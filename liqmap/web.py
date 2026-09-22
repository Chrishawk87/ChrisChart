"""HTTP service: dashboard, JSON API, and a background worker.

Built for Railway, which means three things shape the design:

  PORT       Railway injects it and health-checks the bound port. Binding
             anything else marks the deploy as crashed even though the build
             succeeded. The Procfile passes $PORT through.

  EPHEMERAL  The container filesystem is rebuilt on every deploy. Without a
             mounted volume, SQLite goes with it -- every sweep, every change
             event, every resolved outcome. `startup_report()` shouts about
             this because losing the history is losing the only thing here
             that takes time to accumulate.

  ONE PROC   The web service can also run the collectors on a background
             thread, so a single Railway service does everything. Splitting
             the worker into its own service is cleaner at scale and the
             Procfile offers both.

Auth: every data and mutating route requires LIQMAP_TOKEN. Only /health is
open, because Railway needs it. If the token is unset the service starts in a
locked state that serves the health check and refuses everything else -- a
public URL carrying your trading configuration should not be the default.
"""

from __future__ import annotations

import os
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Any

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from . import consensus as consensus_mod
from .bucket import build_map, render
from .history import History, render_changes
from .settings import Settings, SettingsStore, default_db_path
from .strength import fragility_weighter, score_all, summarise

APP_TITLE = "liqmap"


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------

class Runtime:
    """Shared, lazily built state. One database, reused connections."""

    def __init__(self) -> None:
        self.db_path = default_db_path()
        self.settings_store = SettingsStore(self.db_path)
        self.history = History(self.db_path)
        self.token = os.environ.get("LIQMAP_TOKEN", "").strip()
        self.wallets: list[str] = []
        self.last_error: str | None = None
        self.last_sweep: dict[str, str] = {}
        self.worker_started = False
        self.harvest: dict[str, Any] = {"running": False, "found": 0,
                                        "started": None, "finished": None,
                                        "error": None, "minutes": 0}
        # One level watch at a time. Watching several would multiply the book
        # polls against a shared rate-limit budget that the position sweep
        # also draws on, and in practice you are looking at one level.
        self.watcher = None
        self.watch_info: dict[str, Any] = {"running": False, "coin": None,
                                           "level": None, "started": None,
                                           "finished": None, "error": None}
        self._lock = threading.Lock()
        self._client = None
        self._store = None

    @property
    def locked(self) -> bool:
        return not self.token

    def settings(self) -> Settings:
        return self.settings_store.load()

    def client(self):
        if self._client is None:
            from .hl import InfoClient
            self._client = InfoClient()
        return self._client

    def store(self):
        if self._store is None:
            from .store import Store
            self._store = Store(self.db_path)
        return self._store

    def load_wallets(self) -> list[str]:
        """Wallet universe from the env var or a file alongside the database."""
        if self.wallets:
            return self.wallets

        raw = os.environ.get("LIQMAP_WALLETS", "").strip()
        if raw:
            self.wallets = [w.strip() for w in raw.split(",")
                            if w.strip().startswith("0x")]
            return self.wallets

        import json
        from pathlib import Path
        for candidate in (Path(self.db_path).parent / "wallets.json",
                          Path("wallets.json")):
            if candidate.exists():
                try:
                    self.wallets = json.loads(candidate.read_text())
                    return self.wallets
                except (json.JSONDecodeError, OSError):
                    continue
        return []

    def set_wallets(self, wallets: list[str]) -> int:
        import json
        from pathlib import Path

        clean = [w.strip() for w in wallets if str(w).strip().startswith("0x")]
        self.wallets = clean
        target = Path(self.db_path).parent / "wallets.json"
        try:
            target.write_text(json.dumps(clean, indent=2))
        except OSError:
            pass
        return len(clean)

    # -- bootstrapping a wallet universe -----------------------------------

    def start_harvest(self, minutes: float, then_sweep: bool = True,
                      coins: list[str] | None = None) -> dict[str, Any]:
        """Listen to the public trade feed and collect addresses.

        Without this the first deploy is a dead end: no wallets means no
        sweeps, no sweeps means no map, and the only way to get wallets was a
        CLI command on your own machine. Running it here makes the service
        able to bootstrap itself from an empty database.

        Returns immediately; the work happens on a thread. Poll the same
        endpoint for progress.
        """
        if self.harvest["running"]:
            # Spread FIRST, then the explicit keys. The other way round, the
            # harvest dict's own `error: None` silently overwrites the
            # message and the caller gets a 409 with no reason attached.
            return {**self.harvest, "ok": False,
                    "error": "a harvest is already running"}

        cfg = self.settings()
        # Which tapes to listen to. Discovery is per-coin: a wallet only turns
        # up if it trades a coin you subscribed to. Listening only to BTC
        # finds BTC traders, which is why a universe built that way looks
        # empty on everything else — their other positions do come back in the
        # sweep, but a trader who only trades SOL is never found at all.
        listen = [c.strip().upper() for c in (coins or cfg.coins) if c.strip()]
        self.harvest.update({"running": True, "found": 0, "error": None,
                             "finished": None, "minutes": minutes,
                             "coins": listen,
                             "started": datetime.now(timezone.utc).isoformat()})

        def run() -> None:
            try:
                from .hl import TradeHarvester
                h = TradeHarvester(listen, min_notional=cfg.min_wallet_notional)
                h.run(seconds=minutes * 60,
                      on_progress=lambda n: self.harvest.update({"found": n}))

                wallets = h.top_wallets(cfg.wallet_limit)
                self.set_wallets(wallets)
                self.harvest.update({"found": len(wallets)})

                if then_sweep and wallets:
                    # One sweep now covers every coin these wallets hold, so
                    # the old per-coin loop would just re-read all of them.
                    try:
                        self.sweep()
                    except Exception as exc:
                        self.last_error = f"post-harvest sweep: {exc}"
            except Exception as exc:
                self.harvest["error"] = str(exc)
                self.last_error = f"harvest: {exc}"
            finally:
                self.harvest["running"] = False
                self.harvest["finished"] = datetime.now(timezone.utc).isoformat()

        threading.Thread(target=run, daemon=True).start()
        return {"ok": True, **self.harvest}

    # -- watching a level --------------------------------------------------

    def start_watch(self, coin: str, level: float, minutes: float,
                    band_bps: float = 10.0,
                    window_s: float = 300.0) -> dict[str, Any]:
        """Watch one price level: book depth, refills, and aggression into it.

        Starting a new watch replaces the previous one rather than queueing.
        The point of this is to look at the level in front of you, and a
        stale watch on yesterday's level is worse than none.
        """
        if level <= 0:
            return {"ok": False, "error": "level must be a positive price"}
        if self.watch_info["running"]:
            return {**self.watch_info, "ok": False,
                    "error": (f"already watching {self.watch_info['coin']} at "
                              f"{self.watch_info['level']} — stop it first")}

        try:
            from .hl import LevelWatcher
            watcher = LevelWatcher(self.client(), coin, level,
                                   band_bps=band_bps, window_s=window_s)
        except Exception as exc:
            return {"ok": False, "error": f"could not start: {exc}"}

        self.watcher = watcher
        self.watch_info.update({
            "running": True, "coin": coin, "level": level, "error": None,
            "minutes": minutes, "band_bps": band_bps, "window_s": window_s,
            "started": datetime.now(timezone.utc).isoformat(),
            "finished": None})

        def run() -> None:
            try:
                watcher.run(seconds=minutes * 60)
            except Exception as exc:
                self.watch_info["error"] = str(exc)
                self.last_error = f"watch: {exc}"
            finally:
                self.watch_info["running"] = False
                self.watch_info["finished"] = datetime.now(timezone.utc).isoformat()

        threading.Thread(target=run, daemon=True).start()
        return {"ok": True, **self.watch_info}

    def stop_watch(self) -> dict[str, Any]:
        if self.watcher is not None:
            self.watcher.stop()
        self.watch_info["running"] = False
        return {"ok": True, **self.watch_info}

    # -- the actual work ---------------------------------------------------

    # Below this many positions a coin is not worth its own sweep row. The
    # requested and configured coins are recorded regardless of this.
    MIN_POSITIONS_FOR_COIN = 3
    # Bound on how many coins one sweep records, so a single read cannot write
    # a hundred sweep rows into the volume.
    MAX_COINS_PER_SWEEP = 25

    def sweep(self, coin: str | None = None) -> dict[str, Any]:
        """One sweep: prices, positions, maps, change detection — for EVERY
        coin the wallets are holding, not just one.

        `clearinghouseState` returns a wallet's entire portfolio in a single
        weight-2 call. The original version threw away everything except the
        one coin it was asked about, so getting a second coin meant reading
        all 600 wallets a second time for data already in hand. That is why
        only BTC ever had anything in it: whichever coin you asked for got
        recorded and the rest of the response was discarded.

        Now one read populates all of them. Same API cost as before, and
        asking for a coin nobody holds tells you that instead of silently
        recording nothing.

        Serialised behind a lock. Two concurrent sweeps would both consume the
        same rate-limit budget and each end up with a partial picture.
        """
        with self._lock:
            cfg = self.settings()
            wallets = self.load_wallets()
            if not wallets:
                raise RuntimeError(
                    "no wallet universe configured. POST /api/wallets, set "
                    "LIQMAP_WALLETS, or run `harvest` and place wallets.json "
                    "beside the database.")

            from .hl import sweep_positions

            client = self.client()
            # Mids from every perp DEX, not just the canonical crypto one.
            # Without this, gold, oil, FX and the stock perps have no price,
            # get skipped as unlisted, and look like markets nobody trades.
            try:
                mids = client.all_mids_everywhere()
            except Exception as exc:
                self.last_error = f"multi-dex mids failed, using canonical: {exc}"
                mids = client.all_mids()
            positions, failed = sweep_positions(client, wallets[:cfg.wallet_limit])

            held: dict[str, int] = {}
            for p in positions:
                if p.szi:
                    held[p.coin] = held.get(p.coin, 0) + 1

            asked = coin.upper() if coin else None
            # Order matters: what you asked for first, then what you configured,
            # then whatever else the wallets actually hold, busiest first.
            wanted: list[str] = []
            for c in ([asked] if asked else []) + list(cfg.coins) + \
                    sorted(held, key=lambda c: -held[c]):
                if c and c not in wanted:
                    if (c == asked or c in cfg.coins
                            or held.get(c, 0) >= self.MIN_POSITIONS_FOR_COIN):
                        wanted.append(c)
                if len(wanted) >= self.MAX_COINS_PER_SWEEP:
                    break

            recorded: list[dict[str, Any]] = []
            skipped: list[dict[str, str]] = []
            stamp = datetime.now(timezone.utc).isoformat()

            for c in wanted:
                spot = mids.get(c)
                if not spot:
                    skipped.append({"coin": c, "why": "no mid price on the exchange"})
                    continue
                self.store().log_price(c, spot)
                sweep_id, changes = self.history.record_sweep(c, spot, positions)
                self.last_sweep[c] = stamp
                notable = [ch for ch in changes if ch.kind != "HELD"]
                recorded.append({
                    "coin": c, "spot": spot, "sweep_id": sweep_id,
                    "positions": held.get(c, 0),
                    "changes": len(notable),
                    "defended": len([ch for ch in notable if ch.kind == "DEFENDED"]),
                    "liquidated": len([ch for ch in notable if ch.kind == "LIQUIDATED"]),
                })

            primary = next((r for r in recorded if r["coin"] == asked),
                           recorded[0] if recorded else {})
            out = {
                "wallets_swept": len(wallets[:cfg.wallet_limit]),
                "wallets_failed": failed,
                "coins_recorded": [r["coin"] for r in recorded],
                "per_coin": recorded,
                "skipped": skipped,
                **primary,
            }
            if asked and asked not in [r["coin"] for r in recorded]:
                out["warning"] = (
                    f"{asked} was swept but nothing was recorded for it — "
                    f"{'the exchange has no mid price for that symbol' if asked not in mids else 'none of your wallets hold it'}. "
                    f"Coins with data: {', '.join(r['coin'] for r in recorded) or 'none'}.")
            elif asked and not held.get(asked):
                out["warning"] = (
                    f"None of your {len(wallets[:cfg.wallet_limit])} wallets hold "
                    f"{asked} right now. Harvest for longer, or lower the minimum "
                    f"wallet size in Settings so smaller {asked} traders get picked up.")
            return out

    def resolve_symbol(self, typed: str) -> tuple[str, list[str]]:
        """Turn what someone typed into a symbol that exists.

        HIP-3 markets are namespaced `dex:COIN`, so the gold perp is something
        like `vntl:GOLD`, not `GOLD`. Nobody is going to type that, and typing
        `GOLD` finding nothing is the single most confusing thing this app can
        do — it looks like an empty market rather than a near miss.

        Returns (best match, all candidates). Several builders can list the
        same ticker, so an ambiguous match returns every one rather than
        silently picking.
        """
        typed = typed.strip()
        if not typed:
            return "", []

        known = [r["coin"] for r in self.history.coins_with_data()]
        upper = typed.upper()

        for k in known:                                   # exact, case-aware
            if k.upper() == upper:
                return k, [k]

        from .hl import split_symbol
        matches = [k for k in known if split_symbol(k)[1].upper() == upper]
        if matches:
            return matches[0], matches
        return typed.upper(), []

    def no_data_reason(self, coin: str) -> str:
        """Why this coin is empty, and what to do about it.

        "No sweep recorded yet" is true for a coin you never configured, a
        coin nobody holds, and a service that has never swept at all — three
        different problems with three different fixes, and the bare message
        sends you looking in the wrong place for two of them.
        """
        have = [r["coin"] for r in self.history.coins_with_data()]
        if not have:
            return ("Nothing has been swept yet. Click Harvest wallets to "
                    "collect a universe, then Sweep now.")

        # A near miss is far more likely than a missing market: the HIP-3
        # perps are namespaced, so "GOLD" is really "vntl:GOLD" or similar.
        _, candidates = self.resolve_symbol(coin)
        if candidates:
            return (f"{coin} is listed as {', '.join(candidates)} — those are "
                    f"builder-deployed markets, so the venue is part of the "
                    f"symbol. Use the full name.")

        cfg = self.settings()
        others = ", ".join(have[:8])
        if coin in cfg.coins:
            return (f"{coin} is configured but has no positions recorded. None "
                    f"of your wallets are holding it, or the last sweep missed "
                    f"it. Coins with data right now: {others}.")
        return (f"No data for {coin}. Your wallets are holding {others}. "
                f"Run a sweep — one read records every coin they hold — or add "
                f"{coin} to the coin list in Settings so it is always included.")

    def maps(self, coin: str) -> dict[str, Any]:
        """Both maps from the most recent stored sweep."""
        cfg = self.settings()
        coin = self.resolve_symbol(coin)[0] or coin
        latest = self.history.latest_sweep_positions(coin)
        if not latest:
            return {"coin": coin, "error": self.no_data_reason(coin)}

        sweep_id, positions = latest
        meta = next((s for s in self.history.sweeps(coin, limit=1)), {})
        spot = float(meta.get("spot") or 0)
        if spot <= 0:
            return {"coin": coin, "error": "sweep has no spot price"}

        common = dict(bucket_bps=cfg.bucket_bps,
                      max_distance_pct=cfg.max_distance_pct)
        raw = build_map(positions, coin, spot, **common)
        weighted = build_map(
            positions, coin, spot, **common,
            weight_fn=fragility_weighter(spot, cfg.sigma_per_min,
                                         cfg.horizon_minutes))

        strengths = score_all(positions, spot, cfg.sigma_per_min,
                              cfg.horizon_minutes)

        def clusters(lm) -> list[dict[str, Any]]:
            return [{
                "price": b.mid,
                "distance_pct": (b.mid / spot - 1),
                "side": "above" if b.mid > spot else "below",
                "notional": b.total_notional,
                "long_notional": b.long_notional,
                "short_notional": b.short_notional,
                "positions": b.count,
            } for b in lm.clusters(min_notional=0.0, top=15)]

        return {
            "coin": coin,
            "spot": spot,
            "sweep_id": sweep_id,
            "as_of": meta.get("ts"),
            "raw": {"total_notional": raw.total_notional(),
                    "concentration": raw.concentration(),
                    "clusters": clusters(raw)},
            "weighted": {"total_notional": weighted.total_notional(),
                         "concentration": weighted.concentration(),
                         "clusters": clusters(weighted)},
            "cohort": {
                k: v for k, v in vars(summarise(strengths, coin, spot)).items()
            },
            "text": {"raw": render(raw), "weighted": render(weighted)},
        }


RT: Runtime | None = None


def runtime() -> Runtime:
    global RT
    if RT is None:
        RT = Runtime()
    return RT


# --------------------------------------------------------------------------
# auth
# --------------------------------------------------------------------------

def require_token(authorization: str = Header(default=""),
                  token: str = Query(default="")) -> None:
    """Bearer header or ?token= query. Both, because a dashboard link in a
    browser cannot set a header."""
    rt = runtime()
    if rt.locked:
        raise HTTPException(
            503, "LIQMAP_TOKEN is not set, so this service is locked. Set it "
                 "in the Railway variables and redeploy.")

    supplied = token or (authorization[7:] if authorization.lower().startswith("bearer ")
                         else authorization)
    if supplied.strip() != rt.token:
        raise HTTPException(401, "bad or missing token")


# --------------------------------------------------------------------------
# background worker
# --------------------------------------------------------------------------

def _worker_loop(rt: Runtime) -> None:
    last_sweep = 0.0
    last_price = 0.0

    while True:
        try:
            cfg = rt.settings()
            if not cfg.auto_run:
                time.sleep(20)
                continue

            now = time.monotonic()

            if now - last_price >= cfg.price_interval_seconds:
                last_price = now
                try:
                    mids = rt.client().all_mids()
                    for coin in cfg.coins:
                        if coin in mids:
                            rt.store().log_price(coin, mids[coin])
                except Exception as exc:
                    rt.last_error = f"price poll: {exc}"

            if now - last_sweep >= cfg.sweep_interval_minutes * 60:
                last_sweep = now
                try:
                    rt.sweep()          # one read, every coin
                except Exception as exc:
                    rt.last_error = f"sweep: {exc}"

            time.sleep(5)
        except Exception:
            # The worker must never die. A crashed thread in a single-service
            # deployment means collection silently stops while the dashboard
            # keeps serving stale data, which is worse than an obvious outage.
            rt.last_error = traceback.format_exc(limit=3)
            time.sleep(30)


def start_worker(rt: Runtime) -> None:
    if rt.worker_started:
        return
    rt.worker_started = True
    threading.Thread(target=_worker_loop, args=(rt,), daemon=True).start()


def startup_report(rt: Runtime) -> list[str]:
    """Warnings worth seeing in the deploy logs."""
    notes = [f"database: {rt.db_path}"]

    on_volume = rt.db_path.startswith("/data") or os.environ.get("LIQMAP_DB")
    if not on_volume:
        notes.append(
            "WARNING: the database is on ephemeral disk. Railway rebuilds the "
            "container on every deploy, so all sweep history, change events "
            "and resolved outcomes will be DESTROYED on your next push. Mount "
            "a volume at /data in the Railway service settings.")

    if rt.locked:
        notes.append(
            "WARNING: LIQMAP_TOKEN is unset. The service is locked and will "
            "refuse every route except /health.")

    problems = rt.settings().validate()
    notes.extend(f"settings: {p}" for p in problems)

    if not rt.load_wallets():
        notes.append("no wallet universe yet -- POST /api/wallets or set "
                     "LIQMAP_WALLETS")
    return notes


# --------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------

def create_app() -> FastAPI:
    rt = runtime()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Deploy logs are the only place these warnings will be seen, so they
        # go out before anything else happens.
        for line in startup_report(rt):
            print(f"[liqmap] {line}", flush=True)
        if rt.settings().auto_run and not rt.locked:
            start_worker(rt)
            print("[liqmap] background worker started", flush=True)
        yield

    app = FastAPI(title=APP_TITLE, docs_url="/api/docs", redoc_url=None,
                  lifespan=lifespan)

    # -- open ------------------------------------------------------------

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Unauthenticated so Railway's health check can reach it. Carries no
        market data or configuration."""
        return {
            "ok": True,
            "locked": rt.locked,
            "db": rt.db_path,
            "worker": rt.worker_started,
            "time": datetime.now(timezone.utc).isoformat(),
        }

    # -- data ------------------------------------------------------------

    @app.get("/api/status", dependencies=[Depends(require_token)])
    def status() -> dict[str, Any]:
        cfg = rt.settings()
        return {
            "settings": vars(cfg),
            "settings_problems": cfg.validate(),
            "wallets": len(rt.load_wallets()),
            "history": rt.history.counts(),
            "last_sweep": rt.last_sweep,
            "harvest": rt.harvest,
            "last_error": rt.last_error,
            "warnings": startup_report(rt),
        }

    @app.get("/api/map", dependencies=[Depends(require_token)])
    def api_map(coin: str = "BTC") -> dict[str, Any]:
        return rt.maps(coin.upper())

    @app.get("/api/positions", dependencies=[Depends(require_token)])
    def api_positions(coin: str = "BTC", top: int = 50) -> dict[str, Any]:
        cfg = rt.settings()
        coin = rt.resolve_symbol(coin)[0] or coin.upper()
        latest = rt.history.latest_sweep_positions(coin)
        if not latest:
            return {"coin": coin, "error": rt.no_data_reason(coin)}

        _, positions = latest
        meta = next((s for s in rt.history.sweeps(coin, limit=1)), {})
        spot = float(meta.get("spot") or 0)

        scored = sorted(score_all(positions, spot, cfg.sigma_per_min,
                                  cfg.horizon_minutes),
                        key=lambda s: -s.position.notional)[:top]

        return {"coin": coin, "spot": spot, "as_of": meta.get("ts"),
                "positions": [{
                    "wallet": s.position.wallet,
                    "side": "long" if s.position.is_long else "short",
                    "notional": s.position.notional,
                    "entry": s.position.entry_px,
                    "liquidation": s.position.liquidation_px,
                    "unrealized_pnl": s.position.unrealized_pnl,
                    "account_value": s.position.account_value,
                    "leverage": s.position.leverage,
                    "leverage_type": s.position.leverage_type,
                    "liq_distance_sigmas": (
                        None if s.liq_distance_sigmas == float("inf")
                        else s.liq_distance_sigmas),
                    "commitment": s.commitment,
                    "carry_annualised": s.carry_annualised,
                    "maturity": s.maturity,
                    "survivability": s.survivability(),
                    "fragility": s.fragility(),
                } for s in scored]}

    @app.get("/api/consensus", dependencies=[Depends(require_token)])
    def api_consensus(coin: str = "BTC", winners_only: bool = True,
                      min_notional: float = 0.0,
                      min_account: float = 0.0) -> dict[str, Any]:
        """Who is positioned which way, among traders currently in profit."""
        cfg = rt.settings()
        coin = rt.resolve_symbol(coin)[0] or coin.upper()
        latest = rt.history.latest_sweep_positions(coin)
        if not latest:
            return {"coin": coin, "error": rt.no_data_reason(coin)}

        _, positions = latest
        meta = next((s for s in rt.history.sweeps(coin, limit=1)), {})
        spot = float(meta.get("spot") or 0)
        if spot <= 0:
            return {"coin": coin, "error": "sweep has no spot price"}

        v = consensus_mod.build(
            positions, coin, spot, cfg.sigma_per_min, cfg.horizon_minutes,
            min_notional=min_notional, min_account=min_account,
            winners_only=winners_only)

        return {
            "coin": coin, "spot": spot, "as_of": meta.get("ts"),
            "direction": v.direction,
            "strength": v.strength,
            "agreement": v.agreement,
            "n_traders": v.n_traders,
            "n_winning": v.n_winning,
            "n_losing": v.n_losing,
            "long_notional": v.long_notional,
            "short_notional": v.short_notional,
            "winning_long_notional": v.winning_long_notional,
            "winning_short_notional": v.winning_short_notional,
            "avg_entry": v.avg_entry,
            "entry_gap_pct": v.entry_gap_pct,
            "room_sigmas": v.room_sigmas,
            "median_leverage": v.median_leverage,
            "crowded": v.crowded,
            "late": v.late,
            "verdict": v.verdict(),
            "traders": [{
                "wallet": r.wallet, "side": r.side, "notional": r.notional,
                "entry": r.entry_px, "unrealized_pnl": r.unrealized_pnl,
                "return_on_position": r.return_on_position,
                "account_value": r.account_value, "leverage": r.leverage,
                "liq_distance_sigmas": r.liq_distance_sigmas,
                "survivability": r.survivability,
                "entry_gap_pct": r.entry_gap_pct,
                "winning": r.winning,
            } for r in v.rows[:60]],
        }

    @app.get("/api/changes", dependencies=[Depends(require_token)])
    def api_changes(coin: str | None = None, kind: str | None = None,
                    hours: float = 24.0, limit: int = 200) -> dict[str, Any]:
        kinds = [k.strip().upper() for k in kind.split(",")] if kind else None
        rows = rt.history.recent_changes(
            coin=coin.upper() if coin else None, kinds=kinds,
            hours=hours, limit=limit)
        flow = (rt.history.conviction_flow(coin.upper(), hours) if coin else None)
        return {"changes": rows, "conviction_flow": flow,
                "text": render_changes(rows)}

    @app.get("/api/report", response_class=PlainTextResponse,
             dependencies=[Depends(require_token)])
    def api_report(horizon: float | None = None) -> str:
        from .validate import full_report
        cfg = rt.settings()
        obs = rt.store().observations(horizon_minutes=horizon or cfg.horizon_minutes)
        if not obs:
            return ("No resolved observations yet.\n\n"
                    "Snapshots need their horizon to elapse and prices to be "
                    "recorded through it before anything can be scored.")
        return full_report(obs)

    # -- changing things -------------------------------------------------

    @app.get("/api/settings", dependencies=[Depends(require_token)])
    def get_settings() -> dict[str, Any]:
        cfg = rt.settings()
        return {"settings": vars(cfg), "problems": cfg.validate()}

    @app.post("/api/settings", dependencies=[Depends(require_token)])
    def post_settings(changes: dict[str, Any]) -> JSONResponse:
        cfg, problems = rt.settings_store.update(changes)
        if problems:
            return JSONResponse(
                {"ok": False, "problems": problems, "settings": vars(cfg)},
                status_code=400)
        if cfg.auto_run and not rt.worker_started:
            start_worker(rt)
        return JSONResponse({"ok": True, "settings": vars(cfg)})

    @app.post("/api/settings/reset", dependencies=[Depends(require_token)])
    def reset_settings() -> dict[str, Any]:
        return {"ok": True, "settings": vars(rt.settings_store.reset())}

    @app.post("/api/wallets", dependencies=[Depends(require_token)])
    def post_wallets(payload: dict[str, Any]) -> dict[str, Any]:
        wallets = payload.get("wallets") or []
        if isinstance(wallets, str):
            wallets = [w.strip() for w in wallets.replace("\n", ",").split(",")]
        return {"ok": True, "count": rt.set_wallets(list(wallets))}

    @app.get("/api/harvest", dependencies=[Depends(require_token)])
    def get_harvest() -> dict[str, Any]:
        return {"wallets": len(rt.load_wallets()), **rt.harvest}

    # -- liquidity --------------------------------------------------------

    @app.post("/api/watch", dependencies=[Depends(require_token)])
    def api_watch(coin: str = "BTC", level: float = 0.0, minutes: float = 15.0,
                  band_bps: float = 10.0, window_s: float = 300.0,
                  stop: bool = False) -> dict[str, Any]:
        """Start (or stop) a level watch."""
        if stop:
            return rt.stop_watch()
        out = rt.start_watch(coin.upper(), level, minutes,
                             band_bps=band_bps, window_s=window_s)
        if not out.get("ok"):
            raise HTTPException(409, out.get("error") or "could not start watch")
        return out

    @app.get("/api/liquidity", dependencies=[Depends(require_token)])
    def api_liquidity(size: float = 0.0, coin: str = "BTC") -> dict[str, Any]:
        """Current liquidity picture.

        With a watch running this returns the full reading. Without one it
        still answers the question that does not need history — what does the
        book cost you right now — by pulling a single snapshot. That is the
        useful default: slippage is the one number here that is exact rather
        than learned, so it should not be gated behind a fifteen-minute wait.
        """
        if rt.watcher is not None:
            out = rt.watcher.snapshot(size=size)
            out["watch"] = dict(rt.watch_info)
            return out

        try:
            b = rt.client().book(coin.upper())
        except Exception as exc:
            return {"coin": coin.upper(), "error": str(exc)}
        if b.empty:
            return {"coin": coin.upper(), "error": "book came back empty"}

        out: dict[str, Any] = {
            "coin": coin.upper(),
            "watch": dict(rt.watch_info),
            "book": {
                "bid": b.best_bid, "ask": b.best_ask, "mid": b.mid,
                "spread_bps": b.spread_bps,
                "depth_bid_25bps": b.depth(25.0, "buy"),
                "depth_ask_25bps": b.depth(25.0, "sell"),
                "imbalance_25bps": b.imbalance(25.0),
                "shelves": [{"px": s.px, "notional": s.notional,
                             "multiple": s.multiple, "side": s.side}
                            for s in b.shelves()],
            },
        }
        if size > 0:
            buy, sell = b.walk(size, "buy"), b.walk(size, "sell")
            out["cost"] = {
                "size": size,
                "entry_bps": buy.slippage_bps,
                "exit_bps": sell.slippage_bps,
                "round_trip_bps": buy.slippage_bps + sell.slippage_bps,
                "entry_avg_px": buy.avg_px, "exit_avg_px": sell.avg_px,
                "exhausted": buy.exhausted or sell.exhausted,
                "levels": max(buy.levels_consumed, sell.levels_consumed),
            }
        return out

    @app.post("/api/harvest", dependencies=[Depends(require_token)])
    def post_harvest(minutes: float = 5.0, then_sweep: bool = True,
                     coins: str = "") -> JSONResponse:
        """Collect a wallet universe off the trade feed.

        Five minutes is enough to see the thing work. A real universe wants
        an hour or more -- the longer it listens, the more of the large
        participants it catches, and those are the ones whose liquidations
        matter.

        `coins` is a comma-separated list of tapes to listen to, defaulting to
        the configured coins. Discovery only sees wallets that trade a coin you
        subscribed to, so this is what decides which markets your universe can
        ever cover.
        """
        want = [c for c in (coins or "").upper().split(",") if c.strip()]
        result = rt.start_harvest(min(max(minutes, 0.5), 180.0), then_sweep,
                                  coins=want or None)
        return JSONResponse(result, status_code=200 if result.get("ok") else 409)

    @app.post("/api/sweep", dependencies=[Depends(require_token)])
    def post_sweep(coin: str = "") -> JSONResponse:
        """Sweep. One wallet read records every coin they hold; passing a coin
        only decides which one comes back as the headline."""
        try:
            return JSONResponse({"ok": True, **rt.sweep(coin.upper() or None)})
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    @app.get("/api/coins", dependencies=[Depends(require_token)])
    def api_coins() -> dict[str, Any]:
        """Which markets actually have data, and which are merely configured."""
        from .hl import split_symbol

        cfg = rt.settings()
        have = rt.history.coins_with_data()
        names = [r["coin"] for r in have]
        for r in have:
            dex, base = split_symbol(r["coin"])
            r["dex"] = dex
            r["base"] = base
            # HIP-3 markets are oracle-priced synthetics on a builder's own
            # book. They are NOT the underlying futures market, and their
            # liquidity is a fraction of it.
            r["hip3"] = bool(dex)
        return {
            "configured": cfg.coins,
            "with_data": have,
            "never_swept": [c for c in cfg.coins if c not in names],
        }

    @app.get("/api/dexes", dependencies=[Depends(require_token)])
    def api_dexes() -> dict[str, Any]:
        """Which perp DEXes this service can see.

        If this returns only the canonical DEX, every non-crypto market is
        invisible to the sweep and that is why they look empty.
        """
        try:
            return {"dexes": rt.client().perp_dexs(),
                    "check": rt.client().check_dexes()}
        except Exception as exc:
            return {"dexes": [], "error": str(exc)}

    @app.post("/api/resolve", dependencies=[Depends(require_token)])
    def post_resolve(horizon: float | None = None,
                     follow: float = 60.0) -> dict[str, Any]:
        from datetime import timedelta

        from .store import resolve_touch

        cfg = rt.settings()
        h = horizon or cfg.horizon_minutes
        store = rt.store()
        resolved = skipped = 0

        for row in store.unresolved_clusters(h):
            start = datetime.fromisoformat(row["ts"])
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            prices = store.prices_between(
                row["coin"], start, start + timedelta(minutes=h + follow))
            if len(prices) < 5:
                skipped += 1
                continue
            touched, when, move = resolve_touch(
                prices, start, float(row["spot"]), float(row["price"]),
                float(row["sigma_per_min"]), h, follow)
            store.record_outcome(row["cluster_id"], h, touched, when, move)
            resolved += 1

        return {"ok": True, "resolved": resolved, "skipped": skipped,
                "counts": store.counts()}

    # -- dashboard -------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return DASHBOARD

    return app


DASHBOARD = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>liqmap</title>
<style>
  :root{--bg:#0f1216;--panel:#171b21;--line:#262c34;--ink:#e6e4df;--dim:#8b939d;
        --up:#4eae80;--down:#d9685c;--accent:#d5a340;--mono:ui-monospace,"SF Mono",Menlo,monospace}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:14px/1.5 system-ui,-apple-system,sans-serif}
  .wrap{max-width:1180px;margin:0 auto;padding:20px 16px 60px}
  h1{font-size:20px;margin:0 0 2px;letter-spacing:-.01em}
  .sub{color:var(--dim);font-size:13px;margin-bottom:18px}
  .bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:16px}
  input,select,button{font:inherit;background:var(--panel);color:var(--ink);
       border:1px solid var(--line);border-radius:4px;padding:6px 9px}
  input[type=password]{min-width:200px}
  button{cursor:pointer}
  button:hover{border-color:var(--accent)}
  button.go{background:var(--accent);color:#14171b;border-color:var(--accent);font-weight:600}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:5px;padding:14px}
  .panel h2{font-size:11px;text-transform:uppercase;letter-spacing:.1em;
       color:var(--dim);margin:0 0 10px;font-weight:600}
  pre{font-family:var(--mono);font-size:11.5px;overflow-x:auto;margin:0;
       white-space:pre;color:var(--ink)}
  table{width:100%;border-collapse:collapse;font-size:12px}
  th{text-align:left;color:var(--dim);font-weight:600;padding:4px 6px;
       border-bottom:1px solid var(--line);font-size:11px}
  td{padding:4px 6px;border-bottom:1px solid var(--line);font-family:var(--mono)}
  .r{text-align:right}
  .long{color:var(--up)}.short{color:var(--down)}
  .tag{display:inline-block;padding:1px 6px;border-radius:3px;font-size:10px;
       font-weight:700;letter-spacing:.05em}
  .DEFENDED{background:#1b3d2e;color:var(--up)}
  .LIQUIDATED{background:#45241f;color:var(--down)}
  .WEAKENED{background:#3e3318;color:var(--accent)}
  .ADDED,.OPENED{background:#1d2a38;color:#7fa8c6}
  .REDUCED,.CLOSED{background:#22262c;color:var(--dim)}
  .msg{color:var(--dim);font-size:12px;padding:6px 0}
  .err{color:var(--down)}
  .kv{display:grid;grid-template-columns:auto 1fr;gap:3px 12px;font-size:12px}
  .kv span:nth-child(odd){color:var(--dim)}
  .kv span:nth-child(even){font-family:var(--mono)}
  .full{grid-column:1/-1}
  .conbar{display:flex;flex-wrap:wrap;gap:14px;align-items:center;margin-bottom:12px;
          font-size:12px;color:var(--dim)}
  .conbar label{display:flex;align-items:center;gap:5px}
  .verdict-big{font-size:26px;font-weight:700;letter-spacing:-.02em;margin-bottom:2px}
  .verdict-big.long{color:var(--up)}.verdict-big.short{color:var(--down)}
  .verdict-big.split{color:var(--dim)}
  .conrow{display:flex;flex-wrap:wrap;gap:8px 26px;align-items:baseline;margin:10px 0}
  .stat{display:flex;flex-direction:column}
  .stat b{font-family:var(--mono);font-size:15px;font-weight:700}
  .stat span{font-size:11px;color:var(--dim)}
  .gap-bad b{color:var(--down)}
  .gap-ok b{color:var(--up)}
  .flag{display:inline-block;padding:2px 8px;border-radius:3px;font-size:11px;
        font-weight:700;letter-spacing:.05em;margin-right:6px}
  .flag.crowded{background:#45241f;color:var(--down)}
  .flag.late{background:#3e3318;color:var(--accent)}
  .say{font-size:13px;color:var(--ink);line-height:1.6;margin-top:10px;
       padding:10px 12px;background:var(--bg);border-radius:4px;
       border-left:3px solid var(--accent)}
  label{font-size:12px;color:var(--dim);display:block;margin:8px 0 2px}
</style></head><body>
<div class="wrap">
  <h1>liqmap</h1>
  <div class="sub">Live Hyperliquid positions, read straight from the exchange API.
    Nothing to paste in — <b>Harvest wallets</b> listens to the public trade feed to find
    who is active, <b>Sweep now</b> reads every one of their open positions, and the
    background worker re-sweeps on its own every 30 minutes.</div>

  <div class="bar">
    <input type="password" id="tok" placeholder="access token">
    <input id="coin" value="BTC" size="12" list="coinList" onchange="showVenueNote()"
           title="any perp symbol — crypto, or a HIP-3 market like vntl:GOLD">
    <datalist id="coinList"></datalist>
    <button class="go" onclick="loadAll()">Load</button>
    <button onclick="doHarvest()">Harvest wallets</button>
    <input id="hmin" type="number" value="5" min="1" max="180" step="1"
           title="minutes to listen to the trade feed" style="width:62px">
    <span class="msg" style="margin-right:6px">min</span>
    <input id="hcoins" size="18" placeholder="tapes: BTC,ETH,SOL"
           title="which markets to listen to for wallet discovery — blank uses your configured coins">

    <button onclick="doSweep()">Sweep now</button>
    <button onclick="doResolve()">Resolve</button>
    <span id="msg" class="msg"></span>
  </div>

  <div id="venueNote" class="say" style="display:none;margin-bottom:14px"></div>

  <div class="grid">
    <div class="panel full" id="conPanel"><h2>Who is winning, and which way</h2>
      <div class="msg" style="margin-bottom:8px">Of the wallets holding this coin right now,
        how much winning money is on each side — and how much worse you would enter than
        they did.</div>
      <div class="conbar">
        <label><input type="checkbox" id="winOnly" checked onchange="loadConsensus()">
          winners only</label>
        <label>min position $<input id="minNot" type="number" value="0" step="10000"
          style="width:96px" onchange="loadConsensus()"></label>
        <label>min account $<input id="minAcct" type="number" value="0" step="100000"
          style="width:110px" onchange="loadConsensus()"></label>
      </div>
      <div id="consensus" class="msg">—</div>
      <div id="conTraders"></div>
    </div>

    <div class="panel full" id="liqPanel"><h2>Liquidity — what it costs and who is winning the level</h2>
      <div class="msg" style="margin-bottom:8px">Slippage works immediately off a
        single book read. Absorption needs a running watch, because it has to learn
        how far price normally moves per dollar before it can say whether this is
        a lot of volume or a little.</div>
      <div class="conbar">
        <label>your size $<input id="liqSize" type="number" value="25000" step="5000"
          style="width:110px" onchange="loadLiquidity()"></label>
        <label>level <input id="wLevel" type="number" value="0" step="1"
          style="width:110px" title="price to watch"></label>
        <label>±bps <input id="wBand" type="number" value="10" step="1"
          style="width:64px"></label>
        <label>minutes <input id="wMins" type="number" value="15" min="1" max="120"
          style="width:64px"></label>
        <button onclick="startWatch()">Watch level</button>
        <button onclick="stopWatch()">Stop</button>
      </div>
      <div id="liqCost" class="msg">—</div>
      <div id="liqBook" class="msg"></div>
      <div id="liqAbs" class="say" style="display:none"></div>
      <div id="liqShelves"></div>
    </div>

    <div class="panel"><h2>Status</h2><div id="status" class="msg">—</div>
      <div id="firstrun" class="msg"></div></div>
    <div class="panel"><h2>Conviction flow · 24h</h2><div id="flow" class="msg">—</div></div>

    <div class="panel full"><h2>Position changes — DEFENDED is the one to watch</h2>
      <div id="changes" class="msg">—</div></div>

    <div class="panel"><h2>Raw map — where leverage sits</h2><pre id="mapRaw">—</pre></div>
    <div class="panel"><h2>Fragility-weighted — where pressure is</h2><pre id="mapW">—</pre></div>

    <div class="panel full"><h2>Positions</h2><div id="pos" class="msg">—</div></div>

    <div class="panel full"><h2>Settings</h2>
      <div id="settings" class="msg">—</div>
      <div id="setForm"></div>
    </div>

    <div class="panel full"><h2>Validation report</h2><pre id="report">—</pre></div>
  </div>
</div>
<script>
const $ = id => document.getElementById(id);
const tok = () => $('tok').value.trim();
const coin = () => $('coin').value.trim().toUpperCase() || 'BTC';
const money = n => n == null ? '—' :
  (Math.abs(n) >= 1e6 ? '$' + (n/1e6).toFixed(2) + 'M'
   : Math.abs(n) >= 1e3 ? '$' + (n/1e3).toFixed(0) + 'k' : '$' + Number(n).toFixed(0));
const pct = n => n == null ? '—' : (n*100).toFixed(2) + '%';

try { const s = localStorage.getItem('liqmap_tok'); if (s) $('tok').value = s; } catch (e) {}

async function api(path, opts) {
  if (!tok()) throw new Error('enter your access token first');
  try { localStorage.setItem('liqmap_tok', tok()); } catch (e) {}
  const r = await fetch(path + (path.includes('?') ? '&' : '?') + 'token=' + encodeURIComponent(tok()),
    Object.assign({headers: {'Content-Type': 'application/json'}}, opts || {}));
  const txt = await r.text();
  if (!r.ok) throw new Error(txt.slice(0, 300));
  try { return JSON.parse(txt); } catch (e) { return txt; }
}

function note(m, bad) { $('msg').textContent = m; $('msg').className = 'msg' + (bad ? ' err' : ''); }

async function loadAll() {
  note('loading…');
  try {
    await Promise.all([loadStatus(), loadCoins(), loadConsensus(), loadLiquidity(),
                       loadMap(), loadChanges(), loadPositions(), loadReport()]);
    note('updated ' + new Date().toLocaleTimeString());
  } catch (e) { note(e.message, true); }
}

let harvestPoll = null;

async function doHarvest() {
  // No prompt() here on purpose: browsers block modal prompts in plenty of
  // situations, and when they do the handler returns with no explanation at
  // all -- which looks exactly like a dead button.
  const mins = parseFloat(($('hmin') && $('hmin').value) || '5');
  if (!isFinite(mins) || mins <= 0) { note('enter a positive number of minutes', true); return; }
  if (!tok()) { note('paste your access token first', true); return; }

  note('starting harvest…');
  try {
    const tapes = (($('hcoins') && $('hcoins').value) || '').trim();
    const r = await api('/api/harvest?minutes=' + encodeURIComponent(mins)
                        + '&then_sweep=true'
                        + (tapes ? '&coins=' + encodeURIComponent(tapes) : ''),
                        {method: 'POST'});
    if (!r || r.ok === false) { note('could not start: ' + ((r && r.error) || 'unknown'), true); return; }
    note(`harvesting for ${mins} min — watching the trade feed…`);

    if (harvestPoll) clearInterval(harvestPoll);
    harvestPoll = setInterval(async () => {
      try {
        const h = await api('/api/harvest');
        if (h.running) {
          note(`harvesting… ${h.found} wallet${h.found === 1 ? '' : 's'} so far`);
        } else {
          clearInterval(harvestPoll); harvestPoll = null;
          if (h.error) { note('harvest failed: ' + h.error, true); }
          else if (!h.wallets) {
            note('harvest finished but found no wallets — try a longer run, or lower '
                 + 'min_wallet_notional in Settings', true);
          } else {
            note(`harvest done — ${h.wallets} wallets. Sweeping…`);
            setTimeout(loadAll, 4000);
          }
        }
      } catch (e) {
        clearInterval(harvestPoll); harvestPoll = null;
        note('lost track of the harvest: ' + e.message, true);
      }
    }, 5000);
  } catch (e) { note(e.message, true); }
}

async function loadStatus() {
  const d = await api('/api/status');
  const h = d.history || {};
  $('status').innerHTML = '<div class="kv">'
    + `<span>sweeps</span><span>${h.sweeps ?? 0}</span>`
    + `<span>changes</span><span>${h.changes ?? 0}</span>`
    + `<span>defended</span><span>${h.defended ?? 0}</span>`
    + `<span>liquidated</span><span>${h.liquidated ?? 0}</span>`
    + `<span>wallets</span><span>${d.wallets}</span>`
    + '</div>'
    + (d.warnings || []).map(w => `<div class="msg ${w.startsWith('WARNING') ? 'err' : ''}">${w}</div>`).join('')
    + (d.last_error ? `<div class="msg err">last error: ${d.last_error}</div>` : '');

  // First-run guidance. An empty dashboard with no explanation is the worst
  // possible first impression of a tool that needs data before it says anything.
  const fr = $('firstrun');
  if (!d.wallets) {
    fr.innerHTML = '<b>No wallet universe yet.</b> Click <b>Harvest wallets</b> — '
      + 'it listens to the public trade feed and collects addresses, then sweeps '
      + 'them automatically. Five minutes is enough to see it work.';
  } else if (!h.sweeps) {
    fr.innerHTML = `<b>${d.wallets} wallets, no sweep yet.</b> Click <b>Sweep now</b>. `
      + 'At the rate limit this takes about ' + Math.ceil(d.wallets / 600) + ' minute(s).';
  } else if (h.sweeps === 1) {
    fr.innerHTML = '<b>One sweep recorded.</b> Change events need two — DEFENDED is '
      + 'defined by the difference between sweeps, so it cannot appear until the next one.';
  } else {
    fr.innerHTML = '';
  }
  renderSettings(d.settings);
}

function renderSettings(s) {
  if (!s) return;
  const editable = ['sigma_per_min','horizon_minutes','bucket_bps','min_cluster_notional',
                    'max_distance_pct','sweep_interval_minutes','price_interval_seconds',
                    'wallet_limit','min_wallet_notional'];
  $('settings').innerHTML = '<div class="kv">'
    + `<span>coins</span><span>${(s.coins||[]).join(', ')}</span>`
    + `<span>auto_run</span><span>${s.auto_run}</span>` + '</div>';
  $('setForm').innerHTML = editable.map(k =>
    `<label>${k}</label><input id="set_${k}" value="${s[k]}">`).join('')
    + `<label>coins (comma separated)</label><input id="set_coins" value="${(s.coins||[]).join(',')}">`
    + `<label>auto_run</label><select id="set_auto_run">
         <option value="true" ${s.auto_run?'selected':''}>true</option>
         <option value="false" ${!s.auto_run?'selected':''}>false</option></select>`
    + `<div style="margin-top:12px"><button class="go" onclick="saveSettings()">Save settings</button></div>`;
}

async function saveSettings() {
  const body = {};
  ['sigma_per_min','horizon_minutes','bucket_bps','min_cluster_notional',
   'max_distance_pct','sweep_interval_minutes','price_interval_seconds'].forEach(k => {
     body[k] = parseFloat($('set_' + k).value); });
  ['wallet_limit'].forEach(k => { body[k] = parseInt($('set_' + k).value, 10); });
  body.min_wallet_notional = parseFloat($('set_min_wallet_notional').value);
  body.coins = $('set_coins').value.split(',').map(c => c.trim().toUpperCase()).filter(Boolean);
  body.auto_run = $('set_auto_run').value === 'true';
  try {
    const r = await api('/api/settings', {method: 'POST', body: JSON.stringify(body)});
    note(r.ok ? 'settings saved' : 'rejected: ' + (r.problems || []).join('; '), !r.ok);
    if (r.ok) loadStatus();
  } catch (e) { note(e.message, true); }
}

let watchPoll = null;

async function startWatch() {
  const lvl = parseFloat($('wLevel').value || '0');
  if (!isFinite(lvl) || lvl <= 0) { note('enter the price level you want watched', true); return; }
  const mins = parseFloat($('wMins').value || '15');
  const band = parseFloat($('wBand').value || '10');
  try {
    const r = await api(`/api/watch?coin=${coin()}&level=${lvl}&minutes=${mins}&band_bps=${band}`,
                        {method: 'POST'});
    if (!r || r.ok === false) { note('could not start watch: ' + ((r && r.error) || 'unknown'), true); return; }
    note(`watching ${coin()} at ${lvl} for ${mins} min`);
    if (watchPoll) clearInterval(watchPoll);
    watchPoll = setInterval(loadLiquidity, 10000);
    loadLiquidity();
  } catch (e) { note(e.message, true); }
}

async function stopWatch() {
  if (watchPoll) { clearInterval(watchPoll); watchPoll = null; }
  try { await api('/api/watch?stop=true', {method: 'POST'}); note('watch stopped'); }
  catch (e) { note(e.message, true); }
  loadLiquidity();
}

async function loadLiquidity() {
  const size = parseFloat($('liqSize').value || '0') || 0;
  let d;
  try {
    d = await api('/api/liquidity?coin=' + coin() + '&size=' + size);
  } catch (e) { $('liqCost').textContent = e.message; return; }

  if (d.error) {
    $('liqCost').textContent = d.error;
    $('liqBook').innerHTML = ''; $('liqAbs').style.display = 'none';
    $('liqShelves').innerHTML = ''; return;
  }

  const c = d.cost, b = d.book;
  if (c) {
    const bad = c.exhausted || c.round_trip_bps > 20;
    $('liqCost').innerHTML = '<div class="conrow">'
      + `<div class="stat"><b>${c.entry_bps.toFixed(1)}bps</b><span>cost to get in</span></div>`
      + `<div class="stat"><b>${c.exit_bps.toFixed(1)}bps</b><span>cost to get out</span></div>`
      + `<div class="stat ${bad ? 'gap-bad' : 'gap-ok'}"><b>${c.round_trip_bps.toFixed(1)}bps</b>`
      + `<span>round trip on ${money(c.size)}</span></div>`
      + `<div class="stat"><b>${c.levels}</b><span>levels eaten</span></div>`
      + '</div>'
      + (c.exhausted
         ? '<div class="say">The book cannot fill that size at all. Anything this '
           + 'tool says about entries at this size is theoretical.</div>' : '');
  } else {
    $('liqCost').textContent = 'set your size above to price the entry and exit';
  }

  if (b) {
    const imb = b.imbalance_25bps;
    $('liqBook').innerHTML = '<div class="conrow">'
      + `<div class="stat"><b>${b.spread_bps.toFixed(2)}bps</b><span>spread</span></div>`
      + `<div class="stat"><b>${money(b.depth_bid_25bps)}</b><span>bid depth ±25bps</span></div>`
      + `<div class="stat"><b>${money(b.depth_ask_25bps)}</b><span>ask depth ±25bps</span></div>`
      + `<div class="stat"><b class="${imb >= 0 ? 'long' : 'short'}">${imb >= 0 ? '+' : ''}`
      + `${imb.toFixed(2)}</b><span>${imb >= 0 ? 'bid heavy' : 'offer heavy'}</span></div>`
      + '</div>';

    const sh = b.shelves || [];
    $('liqShelves').innerHTML = !sh.length ? '' :
      '<table style="margin-top:12px"><tr><th>resting shelf</th><th>side</th>'
      + '<th class="r">size</th><th class="r">vs typical level</th></tr>'
      + sh.map(s => `<tr>
          <td>${Number(s.px).toLocaleString(undefined,{maximumFractionDigits:2})}</td>
          <td class="${s.side === 'buy' ? 'long' : 'short'}">${s.side === 'buy' ? 'support' : 'supply'}</td>
          <td class="r">${money(s.notional)}</td>
          <td class="r">${s.multiple.toFixed(1)}x</td></tr>`).join('')
      + '</table>';
  } else { $('liqBook').innerHTML = ''; $('liqShelves').innerHTML = ''; }

  const a = d.absorption;
  if (a) {
    const band = d.band || {};
    const flag = a.absorbing ? '<span class="flag crowded">ABSORBING</span>'
               : a.thin ? '<span class="flag late">THIN</span>' : '';
    const div = d.divergence && d.divergence.disagrees
      ? `<div style="margin-top:6px;color:var(--dim)">Divergence: ${d.divergence.note}</div>` : '';
    $('liqAbs').style.display = '';
    $('liqAbs').innerHTML = flag + a.verdict
      + `<div style="margin-top:6px;color:var(--dim)">`
      + `${d.trades_seen} trades and ${d.books_seen} book reads so far`
      + (band.refill_events ? ` · ${band.refill_events} refills` : '')
      + `</div>` + div
      + (d.side_report && d.side_report.indexOf('ambiguous') === -1 ? ''
         : `<div style="margin-top:6px;color:var(--dim)">side convention: ${d.side_report || '—'}</div>`);
  } else {
    $('liqAbs').style.display = 'none';
  }
}

async function loadConsensus() {
  const wo = $('winOnly').checked;
  const mn = parseFloat($('minNot').value || '0') || 0;
  const ma = parseFloat($('minAcct').value || '0') || 0;
  let d;
  try {
    d = await api('/api/consensus?coin=' + coin() + '&winners_only=' + wo
                  + '&min_notional=' + mn + '&min_account=' + ma);
  } catch (e) { $('consensus').textContent = e.message; return; }

  if (d.error) { $('consensus').textContent = d.error; $('conTraders').innerHTML = ''; return; }

  const gapBad = d.entry_gap_pct > 0.005;
  const flags = (d.crowded ? '<span class="flag crowded">CROWDED</span>' : '')
              + (d.late ? '<span class="flag late">LATE ENTRY</span>' : '');

  $('consensus').innerHTML =
      `<div class="verdict-big ${d.direction}">${d.direction.toUpperCase()}`
    + `<span style="font-size:15px;color:var(--dim);font-weight:400"> `
    + `&nbsp;${(d.strength*100).toFixed(0)}% of winning size</span></div>`
    + `<div style="font-size:12px;color:var(--dim)">${flags}`
    + `${d.n_winning} winning · ${d.n_losing} losing · ${d.n_traders} tracked</div>`
    + '<div class="conrow">'
    + `<div class="stat"><b>${money(d.winning_long_notional)}</b><span>winning long</span></div>`
    + `<div class="stat"><b>${money(d.winning_short_notional)}</b><span>winning short</span></div>`
    + `<div class="stat"><b>${Number(d.avg_entry).toLocaleString(undefined,{maximumFractionDigits:2})}</b><span>their avg entry</span></div>`
    + `<div class="stat ${gapBad ? 'gap-bad' : 'gap-ok'}"><b>${(d.entry_gap_pct*100).toFixed(2)}%</b>`
    + `<span>${gapBad ? 'WORSE than their entry' : 'your entry gap'}</span></div>`
    + `<div class="stat"><b>${Number(d.room_sigmas).toFixed(1)}sd</b><span>room before liq</span></div>`
    + `<div class="stat"><b>${Number(d.median_leverage).toFixed(0)}x</b><span>median leverage</span></div>`
    + '</div>'
    + `<div class="say">${d.verdict}</div>`;

  const rows = (d.traders || []).filter(t => !wo || t.winning);
  $('conTraders').innerHTML = !rows.length ? '' :
    '<table style="margin-top:14px"><tr><th>wallet</th><th>side</th>'
    + '<th class="r">size</th><th class="r">entry</th><th class="r">your gap</th>'
    + '<th class="r">uPnL</th><th class="r">return</th><th class="r">room</th>'
    + '<th class="r">lev</th></tr>'
    + rows.slice(0,25).map(t => `<tr>
        <td>${t.wallet.slice(0,10)}…</td>
        <td class="${t.side}">${t.side}</td>
        <td class="r">${money(t.notional)}</td>
        <td class="r">${Number(t.entry).toLocaleString(undefined,{maximumFractionDigits:2})}</td>
        <td class="r ${t.entry_gap_pct > 0.005 ? 'short' : 'long'}">${(t.entry_gap_pct*100).toFixed(2)}%</td>
        <td class="r ${t.unrealized_pnl >= 0 ? 'long' : 'short'}">${money(t.unrealized_pnl)}</td>
        <td class="r">${(t.return_on_position*100).toFixed(1)}%</td>
        <td class="r">${t.liq_distance_sigmas.toFixed(1)}sd</td>
        <td class="r">${Number(t.leverage).toFixed(0)}x</td></tr>`).join('')
    + '</table>';
}

async function loadMap() {
  const d = await api('/api/map?coin=' + coin());
  if (d.error) { $('mapRaw').textContent = d.error; $('mapW').textContent = d.error; return; }
  $('mapRaw').textContent = d.text.raw;
  $('mapW').textContent = d.text.weighted;
  const c = d.cohort || {};
  $('flow').innerHTML = '<div class="kv">'
    + `<span>spot</span><span>${Number(d.spot).toLocaleString()}</span>`
    + `<span>raw long</span><span>${money(c.long_notional)}</span>`
    + `<span>raw short</span><span>${money(c.short_notional)}</span>`
    + `<span>fragile long</span><span class="down">${money(c.fragile_long_notional)}</span>`
    + `<span>fragile short</span><span class="long">${money(c.fragile_short_notional)}</span>`
    + `<span>mean survivability</span><span>${(c.mean_survivability ?? 0).toFixed(2)}</span>`
    + `<span>pressure</span><span>${(c.fragile_long_notional > c.fragile_short_notional) ? 'downside' : 'upside'}</span>`
    + '</div>';
}

async function loadChanges() {
  const d = await api('/api/changes?hours=24&coin=' + coin());
  const rows = d.changes || [];
  if (!rows.length) {
    $('changes').textContent = 'no changes in the last 24h — needs at least two sweeps';
    return;
  }
  const order = {LIQUIDATED:0, DEFENDED:1, WEAKENED:2, ADDED:3, REDUCED:4, CLOSED:5, OPENED:6};
  rows.sort((a,b) => (order[a.kind] ?? 9) - (order[b.kind] ?? 9)
    || Math.max(b.prev_notional||0,b.new_notional||0) - Math.max(a.prev_notional||0,a.new_notional||0));
  $('changes').innerHTML = '<table><tr><th>when</th><th>kind</th><th>wallet</th>'
    + '<th class="r">notional</th><th class="r">size</th><th class="r">liq move</th></tr>'
    + rows.slice(0, 40).map(r => `<tr>
        <td>${(r.ts||'').slice(5,16).replace('T',' ')}</td>
        <td><span class="tag ${r.kind}">${r.kind}</span></td>
        <td>${(r.wallet||'').slice(0,12)}…</td>
        <td class="r">${money(Math.max(r.prev_notional||0, r.new_notional||0))}</td>
        <td class="r">${r.size_change_pct ? pct(r.size_change_pct) : '—'}</td>
        <td class="r">${r.liq_move_pct ? pct(r.liq_move_pct) : '—'}</td></tr>`).join('')
    + '</table>';
}

async function loadPositions() {
  const d = await api('/api/positions?coin=' + coin() + '&top=40');
  if (d.error) { $('pos').textContent = d.error; return; }
  $('pos').innerHTML = '<table><tr><th>wallet</th><th>side</th><th class="r">notional</th>'
    + '<th class="r">entry</th><th class="r">liq</th><th class="r">liq dist</th>'
    + '<th class="r">uPnL</th><th class="r">acct</th><th class="r">carry</th>'
    + '<th class="r">surv</th><th class="r">frag</th></tr>'
    + (d.positions||[]).map(p => `<tr>
        <td>${p.wallet.slice(0,10)}…</td>
        <td class="${p.side}">${p.side}</td>
        <td class="r">${money(p.notional)}</td>
        <td class="r">${p.entry ? Number(p.entry).toFixed(2) : '—'}</td>
        <td class="r">${p.liquidation ? Number(p.liquidation).toFixed(2) : 'none'}</td>
        <td class="r">${p.liq_distance_sigmas == null ? '∞' : p.liq_distance_sigmas.toFixed(1)+'sd'}</td>
        <td class="r ${p.unrealized_pnl >= 0 ? 'long' : 'short'}">${money(p.unrealized_pnl)}</td>
        <td class="r">${money(p.account_value)}</td>
        <td class="r">${pct(p.carry_annualised)}</td>
        <td class="r">${p.survivability.toFixed(2)}</td>
        <td class="r">${p.fragility.toFixed(2)}</td></tr>`).join('')
    + '</table>';
}

async function loadReport() { $('report').textContent = await api('/api/report'); }

async function doSweep() {
  note('sweeping — this takes a while at the rate limit…');
  try {
    const r = await api('/api/sweep?coin=' + coin(), {method: 'POST'});
    if (!r.ok) { note(r.error, true); return; }
    const coins = r.coins_recorded || [];
    note(`${coin()}: ${r.positions || 0} positions, ${r.changes || 0} changes `
         + `(${r.defended || 0} defended, ${r.liquidated || 0} liquidated) · `
         + `also recorded ${coins.length} coin${coins.length === 1 ? '' : 's'}: `
         + coins.slice(0, 12).join(' '));
    if (r.warning) note(r.warning, true);
    loadCoins();
    loadAll();
  } catch (e) { note(e.message, true); }
}

let hip3Markets = {};

async function loadCoins() {
  try {
    const d = await api('/api/coins');
    const rows = d.with_data || [];
    hip3Markets = {};
    rows.forEach(r => { if (r.hip3) hip3Markets[r.coin] = r.dex; });
    const have = rows.map(r => r.coin);
    $('coinList').innerHTML = rows.map(r =>
        `<option value="${r.coin}">${r.hip3 ? r.base + ' · ' + r.dex : r.base}</option>`)
      .concat((d.configured || []).filter(c => have.indexOf(c) === -1)
              .map(c => `<option value="${c}">`)).join('');
    showVenueNote();
  } catch (e) { /* the coin box still works as free text */ }
}

function showVenueNote() {
  const el = $('venueNote');
  if (!el) return;
  const dex = hip3Markets[coin()];
  if (!dex) { el.style.display = 'none'; return; }
  el.style.display = '';
  el.innerHTML = `<b>${coin()}</b> is a builder-deployed market on <b>${dex}</b>, `
    + 'priced against an oracle rather than matched on the underlying exchange. '
    + 'The book here is this venue’s own and is far thinner than the futures '
    + 'market it tracks — depth, slippage and absorption describe THIS venue, '
    + 'not COMEX or CME. Oracle dislocations and flash moves have happened on '
    + 'these markets.';
}

async function doResolve() {
  try {
    const r = await api('/api/resolve', {method: 'POST'});
    note(`resolved ${r.resolved}, skipped ${r.skipped}`);
    loadReport();
  } catch (e) { note(e.message, true); }
}
</script></body></html>
"""


app = create_app()
