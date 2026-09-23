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

import json
import os
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Sequence

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
        self.last_traceback: str | None = None
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
        self._calibrations: dict[tuple[str, str], Any] = {}
        # Wall-clock stamps for freshness reporting. Every reading on the
        # dashboard is a snapshot from a poll, not a stream, and a number with
        # no age on it is indistinguishable from a number that stopped
        # updating half an hour ago.
        self.last_price_ts: float = 0.0
        self.last_price: dict[str, float] = {}
        self.last_book_ts: float = 0.0
        self.feed = None          # live.LiveFeed, when one is running
        self._markets_cache: dict[str, Any] | None = None
        self._markets_ts: float = 0.0
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

    # The listed universe changes when a builder deploys a market, not by the
    # second. Refetching it on every dashboard poll would spend rate-limit
    # budget the sweep needs.
    MARKETS_TTL_S = 300.0

    def markets(self, force: bool = False) -> dict[str, Any]:
        """Every market you can actually trade, across every perp DEX.

        The picker used to be built from swept history, which meant it listed
        only markets the wallet universe happened to hold -- in practice BTC
        and nothing else. What you can trade and what has been swept are
        different questions, and the picker was answering the wrong one.
        """
        now = time.time()
        if (not force and self._markets_cache
                and now - self._markets_ts < self.MARKETS_TTL_S):
            return self._markets_cache

        from .assetclass import classify, label as class_label
        from .hl import split_symbol

        try:
            mids = self.client().all_mids_everywhere()
        except Exception as exc:
            if self._markets_cache:
                return {**self._markets_cache, "stale": True, "error": str(exc)}
            return {"markets": [], "error": str(exc), "venues": []}

        have = {r["coin"] for r in self.history.coins_with_data()}
        rows = []
        for sym, px in mids.items():
            dex, base = split_symbol(sym)
            klass = classify(sym, dex)
            rows.append({"symbol": sym, "base": base, "dex": dex,
                         "hip3": bool(dex), "price": px,
                         "klass": klass, "klass_label": class_label(klass),
                         "has_data": sym in have})

        # Canonical crypto first, then each builder's markets, alphabetical
        # within a venue so the list is scannable rather than hash-ordered.
        from .assetclass import ORDER

        order = {k: i for i, k in enumerate(ORDER)}
        rows.sort(key=lambda r: (order.get(r["klass"], 99), r["base"]))
        venues = sorted({r["dex"] for r in rows})

        counts: dict[str, int] = {}
        for r in rows:
            counts[r["klass"]] = counts.get(r["klass"], 0) + 1

        self._markets_cache = {
            "markets": rows, "venues": venues, "count": len(rows),
            "classes": [{"klass": k, "label": class_label(k), "n": counts[k]}
                        for k in ORDER if k in counts],
            "fetched": datetime.now(timezone.utc).isoformat(),
        }
        self._markets_ts = now
        return self._markets_cache

    # -- the live feed -----------------------------------------------------

    def start_feed(self, coin: str,
                   intervals: Sequence[str] = ("1m", "5m", "15m", "30m")
                   ) -> dict[str, Any]:
        """Open a persistent socket for one market and build its candles.

        Replaces polling for the current bar entirely: open, high, low, last,
        volume and the buy/sell split all come from fills as they print. The
        builder is seeded once from `candleSnapshot` so the open of the bar
        already in progress is real rather than whatever price happened to be
        trading when the socket connected.

        One market at a time. A second feed doubles the socket traffic for a
        market you are not looking at.
        """
        from .live import LiveFeed

        coin = self.resolve_symbol(coin)[0] or coin
        if self.feed is not None and self.feed.coin == coin and self.feed.running:
            return {"ok": True, **self.feed.status(), "already": True}

        self.stop_feed()
        feed = LiveFeed(coin, intervals=intervals)

        seeded: list[str] = []
        for iv in intervals:
            try:
                feed.seed(iv, self.client().candles(coin, iv, bars=200))
                seeded.append(iv)
            except Exception as exc:
                feed.errors.append(f"seed {iv}: {exc}")

        feed.start()
        self.feed = feed
        return {"ok": True, "seeded": seeded, **feed.status()}

    def stop_feed(self) -> dict[str, Any]:
        if self.feed is not None:
            self.feed.stop()
            out = {"ok": True, **self.feed.status()}
            self.feed = None
            return out
        return {"ok": True, "running": False}

    def feed_for(self, coin: str):
        """The feed, only if it is live on this market and actually ticking."""
        f = self.feed
        if f is None or f.coin != coin or not f.running:
            return None
        return f

    def resolve_reads(self, limit: int = 300) -> dict[str, Any]:
        """Settle recorded reads whose candle has closed.

        Groups by (market, interval) so one candle fetch settles every read
        waiting on it, rather than one request per row.
        """
        pending = self.history.pending_reads(time.time(), limit=limit)
        if not pending:
            return {"ok": True, "resolved": 0, "pending": 0}

        by_market: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in pending:
            by_market.setdefault((row["coin"], row["interval"]), []).append(row)

        resolved = failed = 0
        errors: list[str] = []
        client = self.client()

        for (coin, interval), rows in by_market.items():
            try:
                bars = client.candles(coin, interval, bars=200)
            except Exception as exc:
                errors.append(f"{coin} {interval}: {exc}")
                failed += len(rows)
                continue

            step = client.INTERVALS.get(interval, 900)
            for row in rows:
                # Match within half a bar rather than exactly. A read taken
                # against a synthesised boundary carries a grid timestamp that
                # can sit a fraction off the feed's own, and an exact match
                # would leave those permanently unresolved.
                want = float(row["candle_ts"])
                best = min(bars, key=lambda b: abs(b.ts - want), default=None)
                px = (best.close if best is not None
                      and abs(best.ts - want) <= step / 2 else None)
                if px is None:
                    # Candle rolled out of the window before it was resolved.
                    # Leave it unresolved rather than guessing a price; a
                    # fabricated outcome poisons the measurement permanently.
                    failed += 1
                    continue
                if self.history.resolve_read(row["id"], px):
                    resolved += 1

        return {"ok": True, "resolved": resolved, "unresolvable": failed,
                "pending": len(self.history.pending_reads(time.time())),
                "errors": errors[:5]}

    def now(self, coin: str) -> dict[str, Any]:
        """Spot price plus how old everything else on the dashboard is.

        This exists because a reading with no price beside it cannot be
        checked against anything. If the number here does not match the chart
        you are looking at, nothing else on the page is worth reading, and
        that should take one glance to establish rather than an investigation.

        Ages are computed HERE, on the server, and the client counts up from
        when it received them. Computing them in the browser from timestamps
        means a machine whose clock is a few minutes off reports everything as
        fresh, or everything as stale, and both are worse than no age at all.
        """
        coin = self.resolve_symbol(coin)[0] or coin.upper()
        now = time.time()
        out: dict[str, Any] = {
            "coin": coin,
            "server_time": datetime.now(timezone.utc).isoformat(),
            "server_epoch": now,
        }

        try:
            mids = self.client().all_mids_everywhere()
            self.last_price.update(mids)
            self.last_price_ts = now
            spot = mids.get(coin)
            out["spot"] = spot
            out["spot_age_s"] = 0.0
            if spot is None:
                out["spot_error"] = (
                    f"{coin} has no mid price on the exchange right now")
        except Exception as exc:
            spot = self.last_price.get(coin)
            out["spot"] = spot
            out["spot_age_s"] = (now - self.last_price_ts
                                 if self.last_price_ts else None)
            out["spot_error"] = f"price fetch failed: {exc}"

        # How stale is each panel's underlying data.
        sweeps = self.history.sweeps(coin, limit=1)
        sweep_age = None
        if sweeps:
            try:
                ts = datetime.fromisoformat(sweeps[0]["ts"])
                sweep_age = now - ts.timestamp()
            except (ValueError, KeyError, TypeError):
                sweep_age = None

        watching = bool(self.watch_info.get("running"))
        out["sources"] = {
            # consensus, positions and the liquidation map all come from here
            "sweep": {"age_s": sweep_age,
                      "interval_s": self.settings().sweep_interval_minutes * 60,
                      "note": "positions, consensus and the liquidation map"},
            "book": {"age_s": (now - self.last_book_ts
                               if self.last_book_ts else None),
                     "note": "depth, slippage and shelves"},
            "watch": {"running": watching,
                      "coin": self.watch_info.get("coin"),
                      "level": self.watch_info.get("level"),
                      "note": "flow and absorption — only live while watching"},
        }
        return out

    def calibration(self, coin: str, interval: str):
        """Measured hit rate per market and timeframe.

        Kept per (coin, interval) because they are genuinely different
        questions. A lean that is worth something on a 15-minute gold candle
        says nothing about a 4-hour BTC one, and pooling them would hide both.
        """
        from .candleread import Calibration
        return self._calibrations.setdefault((coin, interval), Calibration())

    def replay_calibration(self, coin: str, interval: str,
                           bars: int = 500) -> int:
        """Score past closed candles and record how each resolved.

        Uses only what candles alone can reconstruct -- structure, zones,
        VWAP, position in range. Flow, absorption and book depth were not
        recorded historically, so the replay is deliberately blind to the
        three strongest live signals. Treat the resulting rate as a floor,
        not as what the live read achieves.
        """
        from . import candleread
        from .structure import (session_anchor, structure as read_structure,
                                vwap as make_vwap, zones)

        candles = self.client().candles(coin, interval, bars=bars)
        if len(candles) < 60:
            raise RuntimeError(f"only {len(candles)} candles — need 60+ to replay")

        step = self.client().INTERVALS.get(interval, 900)
        cal = self.calibration(coin, interval)
        scored = 0

        # Walk forward. Everything each read sees comes from bars strictly
        # before the one being predicted, so the replay cannot peek.
        for i in range(50, len(candles)):
            past = candles[:i]
            target = candles[i]

            zs = zones(past)
            in_zone = next((z for z in zs if z.contains(target.open)), None)
            vw = make_vwap(past, session_anchor(past))

            r = candleread.read(
                coin=coin, interval_s=float(step), elapsed_s=float(step),
                open_px=target.open, high_px=target.open, low_px=target.open,
                last_px=target.open,
                higher=read_structure(past), zone=in_zone, vw=vw)
            if r.signals:
                cal.observe(r.score, candleread.outcome(target))
                scored += 1
        return scored

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

        # Both what has been swept AND what is listed. Resolving only against
        # swept history means a market you can trade but have not swept fails
        # to resolve, which is the same near-miss confusion in a new place.
        known = [r["coin"] for r in self.history.coins_with_data()]
        if self._markets_cache:
            known += [m["symbol"] for m in self._markets_cache.get("markets", [])
                      if m["symbol"] not in known]
        upper = typed.upper()

        for k in known:                                   # exact, case-aware
            if k.upper() == upper:
                return k, [k]

        from .hl import join_symbol, split_symbol
        matches = [k for k in known if split_symbol(k)[1].upper() == upper]
        if matches:
            return matches[0], matches

        # No match on record. Normalise case WITHOUT touching the DEX prefix:
        # HIP-3 DEX names are lowercase and case-sensitive, so upper-casing
        # the whole thing turns `vntl:GOLD` into `VNTL:GOLD`, which matches
        # nothing on the exchange and looks exactly like a missing market.
        dex, base = split_symbol(typed)
        return join_symbol(dex, base.upper()), []

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
    last_resolve = 0.0

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

            if now - last_resolve >= 120:
                last_resolve = now
                try:
                    rt.resolve_reads(limit=200)
                except Exception as exc:
                    rt.last_error = f"resolve reads: {exc}"

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

    @app.exception_handler(Exception)
    async def unhandled(request, exc):     # noqa: ANN001
        """Never serve a bare "Internal Server Error".

        A 500 with no body on a service you are iterating on costs a round
        trip every single time: the person sees five words, and the only way
        to find out what happened is to ask someone with the logs. The error
        class, its message and the route are safe to return -- they describe
        this service's own code, not the caller's data -- and the traceback
        goes to stdout where Railway keeps it.

        The last one is also kept in memory so `/api/status` can show it
        without anyone having to open a log viewer.
        """
        tb = traceback.format_exc()
        rt.last_error = f"{type(exc).__name__}: {exc}"
        rt.last_traceback = tb
        print(f"[liqmap] 500 on {request.url.path}\n{tb}", flush=True)
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "where": str(request.url.path),
                "query": dict(request.query_params),
                "hint": ("The full traceback is in the Railway deploy logs, "
                         "and /api/status carries the last one."),
            })

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
            "last_traceback": (rt.last_traceback or "").splitlines()[-12:],
            "warnings": startup_report(rt),
        }

    @app.get("/api/map", dependencies=[Depends(require_token)])
    def api_map(coin: str = "BTC") -> dict[str, Any]:
        return rt.maps(coin.upper())

    @app.get("/api/positions", dependencies=[Depends(require_token)])
    def api_positions(coin: str = "BTC", top: int = 50) -> dict[str, Any]:
        cfg = rt.settings()
        coin = rt.resolve_symbol(coin)[0] or coin
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
        coin = rt.resolve_symbol(coin)[0] or coin
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
            rt.last_book_ts = time.time()
        except Exception as exc:
            return {"coin": coin.upper(), "error": str(exc)}
        if b.empty:
            return {"coin": coin.upper(), "error": "book came back empty"}

        out: dict[str, Any] = {
            "coin": coin.upper(),
            "watch": dict(rt.watch_info),
            "book_age_s": 0.0,          # fetched in this request
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

    @app.post("/api/feed", dependencies=[Depends(require_token)])
    def api_feed(coin: str = "BTC", stop: bool = False,
                 intervals: str = "1m,5m,15m,30m") -> dict[str, Any]:
        """Start or stop the WebSocket feed for one market."""
        if stop:
            return rt.stop_feed()
        want = tuple(i.strip() for i in intervals.split(",") if i.strip())
        return rt.start_feed(coin, want or ("1m", "5m", "15m"))

    @app.get("/api/feed", dependencies=[Depends(require_token)])
    def api_feed_status() -> dict[str, Any]:
        return rt.feed.status() if rt.feed else {"running": False}

    @app.get("/api/stream")
    def api_stream(coin: str = "BTC", interval: str = "15m",
                   higher: str = "4h", max_seconds: float = 3600.0,
                   token: str = Query(default=""),
                   authorization: str = Header(default="")) -> Any:
        """Server-sent events: a read pushed on every tick.

        EventSource cannot set headers, so the token comes through the query
        string and is checked here rather than by the dependency.

        The stream is driven by the feed's own updates with a short floor
        between sends -- a busy market prints hundreds of fills a second and
        re-running the read on each one would burn the container for no
        additional information.
        """
        require_token(authorization=authorization, token=token)

        from fastapi.responses import StreamingResponse

        coin_r = rt.resolve_symbol(coin)[0] or coin
        tick = threading.Event()

        feed = rt.feed_for(coin_r)
        wake = (lambda _ch: tick.set())
        if feed is not None:
            feed.on_update(wake)

        def events():
          try:
            last_sent = 0.0
            # Bounded on purpose. An unbounded generator holds a worker for as
            # long as a forgotten tab stays open, and Railway does not have
            # many of them. EventSource reconnects by itself when it ends.
            deadline = time.time() + max(1.0, max_seconds)
            # Tell the client immediately whether this is a live socket or a
            # polled fallback, so a silent stream is never mistaken for a
            # quiet market.
            yield (f"event: hello\ndata: "
                   f"{json.dumps({'feed': bool(feed), 'coin': coin_r})}\n\n")

            last_px = None

            while time.time() < deadline:
                fired = tick.wait(timeout=min(5.0, max(0.1, deadline - time.time())))
                tick.clear()
                now = time.time()
                if now >= deadline:
                    break

                # PRICE FIRST, AND ON EVERY TICK.
                #
                # The full read costs structure, zones and VWAP, so it is
                # coalesced. The price is one float the feed already holds,
                # and in a fast market it is the number that goes stale
                # first -- a four-second-old price during a move is worse
                # than useless, because it reads as current. So it is sent
                # on every batch of fills, independently of the read.
                if feed is not None:
                    bar = feed.candle(interval)
                    px = bar.close if bar else None
                    if px is not None and px != last_px:
                        last_px = px
                        yield ("event: px\ndata: "
                               + json.dumps({"px": px, "ts": now,
                                             "high": bar.high, "low": bar.low,
                                             "open": bar.open,
                                             "trades": bar.trades,
                                             "delta": bar.delta})
                               + "\n\n")

                if fired and now - last_sent < 0.5:
                    continue            # coalesce the expensive part only
                last_sent = now
                try:
                    payload = api_read(coin=coin_r, interval=interval,
                                       higher=higher)
                except Exception as exc:
                    yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
                    time.sleep(2.0)
                    continue
                yield f"data: {json.dumps(payload, default=str)}\n\n"

            yield "event: bye\ndata: {}\n\n"
          finally:
            # The stream is over; stop waking it. EventSource reconnects on
            # its own, so without this a tab left open overnight accumulates
            # dead callbacks that fire on every single fill.
            if feed is not None:
                feed.off_update(wake)

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.get("/api/diag", dependencies=[Depends(require_token)])
    def api_diag(coin: str = "BTC", interval: str = "15m") -> dict[str, Any]:
        """Run every upstream call for one symbol and report each result.

        Built because "Internal Server Error" for a symbol names neither the
        call that failed nor the reason. This walks the same sequence a read
        makes -- resolve, mids, candles, book, structure, the read itself --
        and reports each step separately, so the first FAIL is the answer
        rather than the start of an investigation.
        """
        import traceback as _tb

        steps: list[dict[str, Any]] = []

        def step(name: str, fn) -> Any:
            t0 = time.time()
            try:
                value = fn()
                steps.append({"step": name, "ok": True,
                              "ms": round((time.time() - t0) * 1000),
                              "result": value})
                return value
            except Exception as exc:
                steps.append({"step": name, "ok": False,
                              "ms": round((time.time() - t0) * 1000),
                              "error": f"{type(exc).__name__}: {exc}",
                              "trace": _tb.format_exc().splitlines()[-4:]})
                return None

        raw = coin
        resolved = step("resolve symbol",
                        lambda: rt.resolve_symbol(raw)[0] or raw)
        target = resolved or raw
        client = rt.client()

        step("allMids (canonical)",
             lambda: f"{len(client.all_mids())} symbols")
        step("perpDexs", lambda: [d["name"] for d in client.perp_dexs()])
        mids = step("allMids (every dex)",
                    lambda: client.all_mids_everywhere())
        step(f"mid for {target}",
             lambda: (mids or {}).get(target)
             if (mids or {}).get(target) is not None
             else (_ for _ in ()).throw(
                 KeyError(f"{target} has no mid price — check the exact "
                          f"symbol, including the dex prefix and its case")))

        bars = step(f"candleSnapshot {interval}",
                    lambda: client.candles(target, interval, bars=60))
        step("candle count",
             lambda: len(bars) if bars else (_ for _ in ()).throw(
                 ValueError("no candles returned for this symbol")))
        step("l2Book", lambda: client.check_book(target))

        if bars:
            from .structure import structure as read_structure, zones
            step("structure", lambda: read_structure(bars).direction)
            step("zones", lambda: len(zones(bars)))

        step("full read", lambda: api_read(coin=target, interval=interval)
             .get("lean", "no lean"))

        first_fail = next((s for s in steps if not s["ok"]), None)
        return {
            "coin_asked": raw, "coin_used": target, "interval": interval,
            "all_passed": first_fail is None,
            "first_failure": first_fail,
            "steps": steps,
        }

    @app.get("/api/markets", dependencies=[Depends(require_token)])
    def api_markets(refresh: bool = False) -> dict[str, Any]:
        """Everything listed and tradeable, across every perp DEX."""
        return rt.markets(force=refresh)

    @app.get("/api/now", dependencies=[Depends(require_token)])
    def api_now(coin: str = "BTC") -> dict[str, Any]:
        """Live price and how stale everything else is. Cheap enough to poll."""
        return rt.now(coin)

    @app.get("/api/read", dependencies=[Depends(require_token)])
    def api_read(coin: str = "BTC", interval: str = "15m",
                 higher: str = "4h", size: float = 0.0) -> dict[str, Any]:
        """The live directional read on the candle still forming.

        Everything this service knows, assembled into one answer: flow inside
        the candle, whether it is being absorbed, the book, where price sits
        in its own range and against VWAP, the higher-timeframe structure, the
        zone it is in, and the liquidation fuel either side.

        Missing feeds are simply absent rather than filled with neutral
        values. A read built from four signals says four.
        """
        from . import candleread
        from .structure import (atr, session_anchor, structure as read_structure,
                                vwap as make_vwap, zones)

        coin = rt.resolve_symbol(coin)[0] or coin
        client = rt.client()
        out: dict[str, Any] = {"coin": coin, "interval": interval}

        try:
            bars = client.candles(coin, interval, bars=200)
        except Exception as exc:
            return {**out, "error": f"candles unavailable: {exc}"}
        if len(bars) < 10:
            return {**out, "error": f"only {len(bars)} candles came back — "
                                    f"not enough to read structure"}

        from .structure import Candle

        step = client.INTERVALS.get(interval, 900)
        now_s = time.time()

        # If a socket is feeding this market, the current bar is BUILT from
        # the tape rather than fetched. Nothing to poll, nothing to lag, and
        # no chance of being handed a closed bar and told it is the live one.
        feed = rt.feed_for(coin)
        live_bar = feed.candle(interval) if feed else None
        use_feed = live_bar is not None and live_bar.seeded and not feed.stale

        # Is the last bar actually the one still forming?
        #
        # This is the bug behind "it says up on a candle that is clearly going
        # down". If the feed's last bar has already closed, every signal built
        # from it describes the PREVIOUS candle while the panel presents it as
        # the current one -- and a finished green bar sitting under a market
        # that has since turned is exactly how you get an UP read on a falling
        # candle. So it is detected and corrected rather than assumed.
        last = bars[-1]
        candle_age = now_s - last.ts
        last_is_forming = candle_age < step

        # Live mid, fetched BEFORE assembling anything that depends on price.
        live_spot = None
        try:
            live_spot = rt.client().all_mids_everywhere().get(coin)
            rt.last_price.update({coin: live_spot} if live_spot else {})
            rt.last_price_ts = now_s
        except Exception as exc:
            out["price_error"] = str(exc)

        if use_feed:
            # The socket's own candle wins over anything polled.
            current = live_bar.to_candle()
            closed = feed.history(interval) or bars[:-1]
            elapsed = live_bar.elapsed(now_s)
            live_spot = current.close
            candle_age = now_s - current.ts
            last_is_forming = True
            out["source"] = "websocket"
            out["feed"] = feed.status()
        elif last_is_forming:
            current, closed = last, bars[:-1]
            elapsed = max(0.0, min(float(step), candle_age))
            out["source"] = "polled"
        else:
            # No forming bar in the data. Build one from the live price so the
            # read describes NOW rather than a closed candle.
            boundary = last.ts + step * (int(candle_age // step))
            px = live_spot if live_spot else last.close
            current = Candle(ts=boundary, open=last.close, high=max(last.close, px),
                             low=min(last.close, px), close=px, volume=0.0)
            closed = bars
            elapsed = max(0.0, min(float(step), now_s - boundary))
            out["synthesised_candle"] = True
            out["source"] = "polled (last bar had already closed)"

        # Overlay the live mid onto the forming candle. `candleSnapshot` lags
        # the tape, so its close, high and low for the in-progress bar are
        # behind the market -- and `position_in_range` computed from a stale
        # close is the single most misleading number on the panel.
        # Skipped when the socket built the bar: it is already tick-current,
        # and a polled mid would only drag it backwards.
        if not use_feed and live_spot and live_spot > 0:
            current = Candle(ts=current.ts, open=current.open,
                             high=max(current.high, live_spot),
                             low=min(current.low, live_spot),
                             close=live_spot, volume=current.volume,
                             trades=current.trades)

        # higher timeframe
        higher_struct = None
        try:
            hb = client.candles(coin, higher, bars=200)
            if len(hb) >= 20:
                higher_struct = read_structure(hb)
        except Exception as exc:
            out["higher_error"] = str(exc)

        # zone price is currently sitting in
        zs = zones(closed)
        in_zone = next((z for z in zs if z.contains(current.close)), None)

        vw = make_vwap(bars, session_anchor(bars))

        # book, and absorption if a watch happens to be running on this coin
        book = None
        if use_feed and feed.book is not None:
            book = feed.book           # pushed, not polled
            rt.last_book_ts = time.time()
        else:
            try:
                book = client.book(coin)
                rt.last_book_ts = time.time()
            except Exception as exc:
                out["book_error"] = str(exc)

        absorption = tape = None
        w = rt.watcher
        if w is not None and w.coin == coin:
            absorption = w.watch.absorption()
            tape = w.watch.tape
        elif use_feed:
            # Even without a level watch, the feed's tape gives the flow
            # signal -- it is the same public fills either way.
            tape = feed.tape

        # nearest liquidation cluster, signed by side
        magnet_bps = magnet_notional = 0.0
        try:
            m = rt.maps(coin)
            clusters = (m.get("weighted") or {}).get("clusters") or []
            spot = float(m.get("spot") or current.close)
            best = None
            for cl in clusters:
                px = float(cl.get("price") or 0)
                notional = float(cl.get("notional") or 0)
                if px <= 0 or notional <= 0 or spot <= 0:
                    continue
                bps = (px - spot) / spot * 10_000.0
                pull = notional / max(abs(bps), 1.0)
                if best is None or pull > best[0]:
                    best = (pull, bps, notional)
            if best:
                magnet_bps, magnet_notional = best[1], best[2]
        except Exception:
            pass

        r = candleread.read(
            coin=coin, interval_s=float(step), elapsed_s=elapsed,
            open_px=current.open, high_px=current.high, low_px=current.low,
            last_px=current.close, tape=tape, book=book,
            absorption=absorption, higher=higher_struct, zone=in_zone, vw=vw,
            magnet_bps=magnet_bps, magnet_notional=magnet_notional,
            calibration=rt.calibration(coin, interval))

        # The candle feed and the live mid come from different endpoints with
        # different clocks. The overlay above keeps them in step; this reports
        # how far apart they were, which is the honest staleness measure.
        drift_bps = None
        if live_spot and last.close > 0:
            drift_bps = (live_spot - last.close) / last.close * 10_000.0

        out.update({
            "open": current.open, "high": current.high, "low": current.low,
            "last": current.close,
            "spot": live_spot,
            "spot_vs_candle_bps": drift_bps,
            "stale": bool(drift_bps is not None and abs(drift_bps) > 25),
            "candle_age_s": candle_age,
            "candle_was_forming": last_is_forming,
            "predicts": ("direction of price from here to the close of this "
                         f"{interval} candle"),
            "elapsed_fraction": r.elapsed_fraction,
            "seconds_left": r.seconds_left,
            "change_bps": r.change_bps,
            "position_in_range": r.position_in_range,
            "lean": r.lean, "score": r.score, "agreement": r.agreement,
            "confidence": r.confidence, "early": r.early,
            "verdict": r.verdict(),
            "signals": [{"name": s.name, "direction": s.direction,
                         "strength": s.strength, "weighted": s.weighted(),
                         "note": s.note} for s in
                        sorted(r.signals, key=lambda x: -abs(x.weighted()))],
            "higher_timeframe": (higher_struct.describe()
                                 if higher_struct else None),
            "atr": atr(closed),
            "vwap": (None if vw is None else
                     {"value": vw.value, "band": vw.band_of(current.close)}),
            "zone": (None if in_zone is None else
                     {"kind": in_zone.kind, "low": in_zone.low,
                      "high": in_zone.high, "tested": in_zone.tested,
                      "fresh": in_zone.fresh}),
            "calibration": rt.calibration(coin, interval).table(),
            "has_watch": absorption is not None,
        })
        if book is not None and not book.empty:
            out["book"] = {
                "bid": book.best_bid, "ask": book.best_ask, "mid": book.mid,
                "spread_bps": book.spread_bps,
                "imbalance_25bps": book.imbalance(25.0),
                "pushed": bool(use_feed and feed.book is not None),
            }
        if size > 0 and book is not None and not book.empty:
            out["round_trip_bps"] = book.round_trip_bps(size)

        # Write the read down so it can be scored when the candle closes.
        # This is the only way the flow and absorption signals ever get
        # measured: no exchange serves historical order flow, so a backtest
        # over past candles is permanently blind to them. This accumulates
        # while you use the dashboard.
        try:
            import json as _json
            rid = rt.history.record_read(
                coin=coin, interval=interval, candle_ts=current.ts,
                candle_end=current.ts + step, score=r.score, lean=r.lean,
                confidence=r.confidence, elapsed_frac=r.elapsed_fraction,
                price=current.close, had_flow=absorption is not None,
                signals=_json.dumps({s.name: round(s.weighted(), 4)
                                     for s in r.signals}))
            out["recorded"] = rid is not None
        except Exception as exc:
            out["record_error"] = str(exc)

        out["forward"] = rt.history.read_stats(coin, interval)
        return out

    @app.post("/api/resolve-reads", dependencies=[Depends(require_token)])
    def api_resolve_reads(limit: int = 300) -> dict[str, Any]:
        """Score recorded reads whose candle has now closed."""
        return rt.resolve_reads(limit=limit)

    @app.get("/api/forward", dependencies=[Depends(require_token)])
    def api_forward(coin: str = "", interval: str = "") -> dict[str, Any]:
        """Measured performance of the live read, from recorded outcomes."""
        c = rt.resolve_symbol(coin)[0] if coin else None
        return {"coin": c, "interval": interval or None,
                "stats": rt.history.read_stats(c, interval or None)}

    @app.post("/api/backtest", dependencies=[Depends(require_token)])
    def api_backtest(coin: str = "BTC", interval: str = "15m",
                     bars: int = 1000, train_fraction: float = 0.6,
                     tune: bool = True) -> dict[str, Any]:
        """Walk-forward backtest with a held-out half.

        The number to act on is the held-out one. Weights are searched on the
        earlier portion only and the later portion is never seen by the
        search, so the gap between the two is a direct measure of how much of
        any apparent edge is fitted noise.
        """
        from . import backtest as bt_mod

        coin = rt.resolve_symbol(coin)[0] or coin
        try:
            candles = rt.client().candles(coin, interval,
                                          bars=max(200, min(bars, 5000)))
        except Exception as exc:
            return {"ok": False, "error": f"candles unavailable: {exc}"}

        bt = bt_mod.run(candles, coin, interval,
                        train_fraction=max(0.3, min(train_fraction, 0.85)),
                        do_tune=tune)

        def rep(r) -> dict[str, Any]:
            return {"label": r.label, "n": r.n, "calls": r.calls,
                    "accuracy": r.accuracy, "coverage": r.coverage,
                    "edge_pts": r.edge, "describe": r.describe(),
                    "bands": [{"band": b.band, "n": b.n, "up_rate": b.up_rate}
                              for b in r.bands],
                    "curve": r.curve}

        span = ""
        if candles:
            from datetime import datetime as _dt
            a = _dt.utcfromtimestamp(candles[0].ts).strftime("%Y-%m-%d %H:%M")
            b = _dt.utcfromtimestamp(candles[-1].ts).strftime("%Y-%m-%d %H:%M")
            span = f"{a} to {b} UTC"

        return {
            "ok": True, "coin": coin, "interval": interval,
            "candles": len(candles), "samples": len(bt.samples),
            "history_span": span,
            "source": "real exchange candles via candleSnapshot",
            "train": rep(bt.train), "test": rep(bt.test),
            "baseline_test": rep(bt.baseline_test) if bt.baseline_test else None,
            "tuned_weights": bt.tuned_weights,
            "overfit_gap_pts": bt.overfit_gap,
            "verdict": bt.verdict(),
            "blind_to": ["flow", "absorption", "imbalance", "magnet"],
        }

    @app.post("/api/weights", dependencies=[Depends(require_token)])
    def api_weights(payload: dict[str, float] | None = None,
                    reset: bool = False) -> dict[str, Any]:
        """Apply tuned weights to the live read, or put the defaults back.

        Separate from the backtest on purpose: seeing a tuned result and
        adopting it should be two decisions, not one. A tuned set that only
        improved the training half is exactly the thing you do not want
        silently applied to live readings.
        """
        from . import candleread

        if reset:
            candleread.WEIGHTS.clear()
            candleread.WEIGHTS.update(candleread.DEFAULT_WEIGHTS)
            return {"ok": True, "weights": dict(candleread.WEIGHTS),
                    "defaults": dict(candleread.DEFAULT_WEIGHTS),
                    "note": "defaults restored"}
        if not payload:
            return {"ok": True, "weights": dict(candleread.WEIGHTS),
                    "defaults": dict(candleread.DEFAULT_WEIGHTS)}

        clean = {k: max(0.0, min(float(v), 5.0)) for k, v in payload.items()
                 if k in candleread.DEFAULT_WEIGHTS}
        if not clean:
            return {"ok": False, "error": "no recognised weight names",
                    "known": sorted(candleread.DEFAULT_WEIGHTS)}
        candleread.WEIGHTS.update(clean)
        return {"ok": True, "weights": dict(candleread.WEIGHTS),
                "defaults": dict(candleread.DEFAULT_WEIGHTS),
                "applied": clean}

    @app.post("/api/calibrate", dependencies=[Depends(require_token)])
    def api_calibrate(coin: str = "BTC", interval: str = "15m") -> dict[str, Any]:
        """Score every closed candle and record how it actually resolved.

        This is what turns the lean into a measurable number. It replays
        history with the signals that were available from candles alone --
        NOT flow or absorption, which were not recorded at the time and
        cannot be reconstructed. So the replayed hit rate is a floor: the
        live read has strictly more information than this.
        """
        coin = rt.resolve_symbol(coin)[0] or coin
        try:
            n = rt.replay_calibration(coin, interval)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "coin": coin, "interval": interval,
                "scored": n, "table": rt.calibration(coin, interval).table()}

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

  /* Live price and staleness. Every reading on this page is a snapshot from
     a poll; without an age beside it there is no way to tell a quiet market
     from a feed that stopped. */
  .ticker{display:flex;flex-wrap:wrap;align-items:center;gap:18px;
    padding:12px 16px;margin-bottom:14px;background:var(--panel);
    border:1px solid var(--line);border-radius:6px}
  .tick-px{display:flex;align-items:baseline;gap:10px;font-size:13px;
    color:var(--dim)}
  .tick-px b{font-size:28px;color:var(--fg);font-variant-numeric:tabular-nums}
  .dot{width:9px;height:9px;border-radius:50%;background:var(--long);
    display:inline-block;align-self:center}
  .dot.warn{background:#d6a14a} .dot.bad{background:var(--short)}
  .dot.live{animation:pulse 2s ease-in-out infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
  .tick-age{font-size:12px;font-variant-numeric:tabular-nums}
  .tick-age.warn{color:#d6a14a} .tick-age.bad{color:var(--short)}
  .tick-src{display:flex;flex-wrap:wrap;gap:14px;font-size:11px;color:var(--dim)}
  .tick-src span b{color:var(--fg);font-weight:600}
  .tick-src span.warn b{color:#d6a14a} .tick-src span.bad b{color:var(--short)}
  .stamp{font-size:11px;color:var(--dim);font-weight:400;margin-left:8px}
  .stamp.warn{color:#d6a14a} .stamp.bad{color:var(--short)}

  .alertbar{padding:12px 14px;margin:10px 0;border-radius:5px;font-size:14px;
    font-weight:600;border-left:4px solid}
  .alertbar.up{background:rgba(74,168,116,.14);border-color:var(--long);
    color:var(--long)}
  .alertbar.down{background:rgba(201,90,90,.14);border-color:var(--short);
    color:var(--short)}
  .alertbar small{display:block;font-weight:400;color:var(--dim);margin-top:4px}
  .bt-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
    gap:14px;margin-top:10px}
  .bt-col{padding:10px 12px;background:var(--bg);border-radius:5px;
    border:1px solid var(--line)}
  .bt-col h3{margin:0 0 6px;font-size:12px;text-transform:uppercase;
    letter-spacing:.06em;color:var(--dim)}
  .bt-col.held{border-color:var(--accent)}
  .bt-big{font-size:26px;font-variant-numeric:tabular-nums}
  .wgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
    gap:8px}
  .wgrid div{background:var(--bg);border:1px solid var(--line);border-radius:4px;
    padding:6px 8px}
  .wgrid label{margin:0 0 3px;font-size:11px}
  .wgrid input{width:100%;padding:4px 6px;font-size:13px}
  .wgrid div.changed{border-color:var(--accent)}
</style></head><body>
<div class="wrap">
  <h1>liqmap</h1>
  <div class="sub">Live Hyperliquid positions, read straight from the exchange API.
    Nothing to paste in — <b>Harvest wallets</b> listens to the public trade feed to find
    who is active, <b>Sweep now</b> reads every one of their open positions, and the
    background worker re-sweeps on its own every 30 minutes.</div>

  <div class="bar">
    <input type="password" id="tok" placeholder="access token">
    <select id="klass" onchange="renderCoins()" style="min-width:110px"
            title="asset class"><option value="">all classes</option></select>
    <input id="mktFind" size="10" placeholder="find…" oninput="renderCoins()"
           title="filter by ticker">
    <select id="coin" onchange="onCoinChange()" style="min-width:160px"
            title="every market listed on the exchange — a dot marks ones with swept position data">
      <option value="BTC">BTC</option></select>
    <span id="mktCount" class="msg" style="margin-right:6px">—</span>
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

  <div id="ticker" class="ticker">
    <div class="tick-px"><span id="tickSym">—</span>
      <b id="tickPx">—</b>
      <span id="tickDot" class="dot"></span>
      <span id="tickAge" class="tick-age">connecting…</span></div>
    <div id="tickSources" class="tick-src"></div>
  </div>

  <div id="venueNote" class="say" style="display:none;margin-bottom:14px"></div>

  <div class="grid">
    <div class="panel full" id="readPanel"><h2>Candle read — live<span class="stamp" id="readStamp"></span></h2>
      <div class="msg" style="margin-bottom:8px">Everything this service knows,
        assembled into one direction on the candle still forming. Absorption
        <b>inverts</b> flow: heavy buying that is not moving price is a bearish
        reading, not a bullish one.</div>
      <div class="conbar">
        <label>candle <select id="rInt" onchange="loadRead()">
          <option>1m</option><option>5m</option><option selected>15m</option>
          <option>30m</option><option>1h</option></select></label>
        <label>higher TF <select id="rHigh" onchange="loadRead()">
          <option>15m</option><option>30m</option><option>1h</option>
          <option selected>4h</option><option>12h</option>
          <option>1d</option></select></label>
        <label>alert at <select id="rAlert" onchange="saveAlert()">
          <option value="0">off</option><option value="0.6">60%</option>
          <option value="0.7">70%</option><option value="0.8" selected>80%</option>
          <option value="0.9">90%</option></select> confidence</label>
        <label><input type="checkbox" id="rAuto" onchange="toggleReadAuto()">
          poll 10s</label>
        <button onclick="startFeed()" id="feedBtn">Go live (websocket)</button>
        <button onclick="stopFeed()">Stop feed</button>
        <button onclick="loadRead()">Read now</button>
        <button onclick="doCalibrate()">Calibrate</button>
      </div>
      <div id="feedBar" class="msg" style="margin-bottom:8px">Feed off — the
        current candle is being polled. Go live to build it from the tape instead.</div>
      <div id="alertBar" class="alertbar" style="display:none"></div>
      <div id="readHead" class="msg">—</div>
      <div id="readSignals"></div>
      <div id="readSay" class="say" style="display:none"></div>
    </div>

    <div class="panel full" id="btPanel"><h2>Backtest — measured, not asserted</h2>
      <div class="msg" style="margin-bottom:8px">Two separate measurements.
        <b>Historical replay</b> uses real candles pulled from the exchange — the
        count and date range are shown with the result — split in two by date, with
        weights tuned on the earlier half only. No exchange serves historical order
        flow or book depth, so that replay can only measure the candle-structure
        signals. <b>Forward test</b> is the answer to that: every live read is written
        down and scored when its candle closes, so flow and absorption get measured
        too. It fills up as you use the dashboard.</div>
      <div id="fwd" class="msg" style="margin-bottom:10px">—</div>
      <div class="conbar">
        <label>bars <input id="btBars" type="number" value="1000" min="200" max="5000"
          step="100" style="width:88px"></label>
        <label>train split <select id="btSplit">
          <option value="0.5">50/50</option><option value="0.6" selected>60/40</option>
          <option value="0.7">70/30</option></select></label>
        <label><input type="checkbox" id="btTune" checked> search weights</label>
        <button onclick="runBacktest()">Run backtest</button>
      </div>
      <div id="btHead" class="msg">—</div>
      <div id="btCurve"></div>
      <div id="btSay" class="say" style="display:none"></div>

      <h3 style="margin:18px 0 4px;font-size:12px;text-transform:uppercase;
        letter-spacing:.06em;color:var(--dim)">Signal weights — edit and apply</h3>
      <div class="msg" style="margin-bottom:8px">These are the multipliers the
        live read uses. Running a backtest fills in a tuned set; you can also
        type your own. Nothing is applied until you press Apply.</div>
      <div id="wGrid" class="wgrid"></div>
      <div style="margin-top:8px">
        <button onclick="applyWeights()">Apply weights</button>
        <button onclick="resetWeights()">Reset to defaults</button>
        <button onclick="useTuned()" id="btUseTuned" disabled>Load tuned values</button>
        <span id="wMsg" class="msg"></span>
      </div>
    </div>

    <div class="panel full" id="conPanel"><h2>Who is winning, and which way<span class="stamp" id="conStamp"></span></h2>
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

    <div class="panel full" id="liqPanel"><h2>Liquidity — what it costs and who is winning the level<span class="stamp" id="liqStamp"></span></h2>
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

// Do NOT upper-case the whole symbol. HIP-3 DEX prefixes are lowercase and
// case-sensitive, so `vntl:GOLD` upper-cased is a symbol that does not exist.
const coin = () => {
  const raw = $('coin').value.trim();
  if (!raw) return 'BTC';
  const i = raw.indexOf(':');
  return i < 0 ? raw.toUpperCase()
               : raw.slice(0, i) + ':' + raw.slice(i + 1).toUpperCase();
};

/* Build a query string with every value encoded.
   Concatenating a raw symbol into a URL is what caused "bad or missing
   token": a symbol containing `#` turns everything after it into a fragment,
   the browser never sends `&token=`, and the server correctly rejects the
   request. The symptom looked like an auth bug and was a URL bug.           */
function q(params) {
  return Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== null && v !== '')
    .map(([k, v]) => encodeURIComponent(k) + '=' + encodeURIComponent(v))
    .join('&');
}
const money = n => n == null ? '—' :
  (Math.abs(n) >= 1e6 ? '$' + (n/1e6).toFixed(2) + 'M'
   : Math.abs(n) >= 1e3 ? '$' + (n/1e3).toFixed(0) + 'k' : '$' + Number(n).toFixed(0));
const pct = n => n == null ? '—' : (n*100).toFixed(2) + '%';

try { const s = localStorage.getItem('liqmap_tok'); if (s) $('tok').value = s; } catch (e) {}

async function api(path, opts) {
  if (!tok()) throw new Error('enter your access token first');
  try { localStorage.setItem('liqmap_tok', tok()); } catch (e) {}
  // Token goes in BOTH the header and the query. The header survives any
  // mangling of the query string; the query keeps plain links working.
  const o = Object.assign({}, opts || {});
  o.headers = Object.assign({'Content-Type': 'application/json',
                             'Authorization': 'Bearer ' + tok()},
                            o.headers || {});
  const r = await fetch(path + (path.includes('?') ? '&' : '?')
                        + 'token=' + encodeURIComponent(tok()), o);
  const txt = await r.text();
  if (!r.ok) throw new Error(txt.slice(0, 300));
  try { return JSON.parse(txt); } catch (e) { return txt; }
}

function note(m, bad) { $('msg').textContent = m; $('msg').className = 'msg' + (bad ? ' err' : ''); }

async function loadAll() {
  note('loading…');
  if (!nowPoll) startTicker();
  try {
    await Promise.all([loadStatus(), loadCoins(), loadWeights(), loadRead(),
                       loadForward(), loadConsensus(),
                       loadLiquidity(), loadMap(), loadChanges(), loadPositions(),
                       loadReport()]);
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
    const r = await api('/api/harvest?' + q({minutes: mins, then_sweep: 'true',
                        coins: tapes}), {method: 'POST'});
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

/* ---- live price and staleness -------------------------------------------
   Ages arrive computed by the server and we count up from the moment we
   received them. Deriving them in the browser from timestamps means a clock
   a few minutes out reports everything as fresh, or everything as stale.  */

let nowData = null, nowFetchedAt = 0, nowPoll = null, lastPx = null;

function ageClass(sec, warn, bad) {
  if (sec == null) return 'bad';
  return sec >= bad ? 'bad' : sec >= warn ? 'warn' : '';
}

function ageText(sec) {
  if (sec == null) return 'never';
  if (sec < 60) return Math.floor(sec) + 's ago';
  if (sec < 3600) return Math.floor(sec / 60) + 'm ago';
  return (sec / 3600).toFixed(1) + 'h ago';
}

async function loadNow() {
  try {
    nowData = await api('/api/now?' + q({coin: coin()}));
    nowFetchedAt = Date.now();
  } catch (e) {
    $('tickAge').textContent = e.message;
    $('tickDot').className = 'dot bad';
    return;
  }
  paintTicker();
}

function paintTicker() {
  const d = nowData;
  if (!d) return;
  // A streamed price is always fresher than a polled one. Repainting over it
  // would make a live tape look like it updates every few seconds.
  const streaming = streamedAt && (Date.now() - streamedAt) < 15000;
  const since = (Date.now() - nowFetchedAt) / 1000;

  $('tickSym').textContent = d.coin;
  if (streaming) { paintSources(d, since); return; }
  if (d.spot != null) {
    const px = Number(d.spot);
    const dir = lastPx == null ? '' : px > lastPx ? 'long' : px < lastPx ? 'short' : '';
    $('tickPx').className = dir;
    $('tickPx').textContent = px.toLocaleString(undefined,
      {minimumFractionDigits: 2, maximumFractionDigits: 6});
    lastPx = px;
  } else {
    $('tickPx').textContent = '—';
  }

  const age = (d.spot_age_s == null ? null : d.spot_age_s + since);
  const cls = ageClass(age, 20, 60);
  $('tickDot').className = 'dot ' + (cls || 'live');
  $('tickAge').className = 'tick-age ' + cls;
  $('tickAge').textContent = (d.spot_error ? d.spot_error + ' · ' : '')
    + 'price ' + ageText(age);

  paintSources(d, since);
}

function paintSources(d, since) {
  const s = d.sources || {};
  const bits = [];
  const sweepWarn = (s.sweep && s.sweep.interval_s) ? s.sweep.interval_s * 1.5 : 2700;
  if (s.sweep) {
    const a = s.sweep.age_s == null ? null : s.sweep.age_s + since;
    bits.push(`<span class="${ageClass(a, sweepWarn, sweepWarn * 2)}">`
      + `positions &amp; consensus <b>${ageText(a)}</b></span>`);
  }
  if (s.book) {
    const a = s.book.age_s == null ? null : s.book.age_s + since;
    bits.push(`<span class="${ageClass(a, 30, 120)}">book <b>${ageText(a)}</b></span>`);
  }
  if (s.watch) {
    bits.push(s.watch.running
      ? `<span>flow &amp; absorption <b>live on ${s.watch.coin} @ ${s.watch.level}</b></span>`
      : '<span class="bad">flow &amp; absorption <b>not running</b></span>');
  }
  $('tickSources').innerHTML = bits.join('');
}

function tickFromStream(p) {
  if (!p || p.px == null) return;
  const el = $('tickPx');
  const dir = lastPx == null ? '' : p.px > lastPx ? 'long' : p.px < lastPx ? 'short' : '';
  el.className = dir;
  el.textContent = Number(p.px).toLocaleString(undefined,
    {minimumFractionDigits: 2, maximumFractionDigits: 6});
  lastPx = p.px;
  streamedAt = Date.now();

  $('tickDot').className = 'dot live';
  $('tickAge').className = 'tick-age';
  $('tickAge').textContent = 'live tape'
    + (p.trades ? ` · ${p.trades} fills this candle` : '');
}

let streamedAt = 0;

function startTicker() {
  if (nowPoll) clearInterval(nowPoll);
  loadNow();
  // 5s is fine as a backstop while the socket carries the price. Without a
  // socket this is the only source, so it runs harder — at weight 2 a 2s
  // poll costs 60/minute against a 1200 budget, which the sweep can absorb.
  nowPoll = setInterval(() => {
    const streaming = streamedAt && (Date.now() - streamedAt) < 15000;
    if (!streaming || Math.random() < 0.2) loadNow();
  }, 2000);
  setInterval(paintTicker, 1000);       // keep the age counting between polls
}

let lastCoin = 'BTC';
try { const c = localStorage.getItem('liqmap_coin'); if (c) lastCoin = c; } catch (e) {}
window.addEventListener('DOMContentLoaded', () => {
  try {
    const a = localStorage.getItem('liqmap_alert');
    if (a && $('rAlert')) $('rAlert').value = a;
  } catch (e) {}
});

function onCoinChange() {
  // A feed is bound to one market. Leaving it running on the old one would
  // quietly serve its candles under the new symbol's name.
  if (stream || feedTimer) stopFeed();
  lastCoin = coin();
  try { localStorage.setItem('liqmap_coin', lastCoin); } catch (e) {}
  alertedFor = null;            // a new market starts its own alert history
  showVenueNote();
  loadNow();
  loadRead();
}

function stamp(id, seconds, warn, bad, label) {
  const el = $(id);
  if (!el) return;
  el.className = 'stamp ' + ageClass(seconds, warn, bad);
  el.textContent = (label || '') + ageText(seconds);
}

/* ---- the confidence alert ------------------------------------------------
   Fires once per candle per direction. Three suppressions matter more than
   the alert itself: it will not fire on stale data, it will not fire early
   in a candle, and it will not re-fire for a state it already announced.
   An alert that cries wolf gets ignored, and then the one that matters does
   too.                                                                     */

let alertedFor = null, audioCtx = null;

function alertThreshold() {
  const v = parseFloat(($('rAlert') && $('rAlert').value) || '0.8');
  return isFinite(v) ? v : 0.8;
}

function saveAlert() {
  try { localStorage.setItem('liqmap_alert', $('rAlert').value); } catch (e) {}
  alertedFor = null;
}

function ping() {
  // Synthesised rather than a file: an artifact page cannot load external
  // media, and a two-tone chirp cuts through better than a single beep.
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    [880, 1320].forEach((f, i) => {
      const o = audioCtx.createOscillator(), g = audioCtx.createGain();
      o.frequency.value = f; o.type = 'sine';
      g.gain.setValueAtTime(0.0001, audioCtx.currentTime + i * 0.14);
      g.gain.exponentialRampToValueAtTime(0.25, audioCtx.currentTime + i * 0.14 + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + i * 0.14 + 0.13);
      o.connect(g); g.connect(audioCtx.destination);
      o.start(audioCtx.currentTime + i * 0.14);
      o.stop(audioCtx.currentTime + i * 0.14 + 0.14);
    });
  } catch (e) {}
}

function checkAlert(d) {
  const bar = $('alertBar');
  const thr = alertThreshold();

  if (!thr || d.lean === 'flat' || d.confidence < thr) {
    bar.style.display = 'none';
    if (d.lean === 'flat' || d.confidence < thr * 0.8) alertedFor = null;
    return;
  }
  if (d.early || d.stale) { bar.style.display = 'none'; return; }

  const key = coin() + '|' + $('rInt').value + '|' + d.lean
            + '|' + Math.floor(d.seconds_left / 60);
  const candleKey = coin() + '|' + $('rInt').value + '|' + d.lean;

  bar.className = 'alertbar ' + d.lean;
  bar.style.display = '';
  bar.innerHTML = `${d.lean.toUpperCase()} — ${(d.confidence*100).toFixed(0)}% confidence, `
    + `${(d.agreement*100).toFixed(0)}% of signals agree, `
    + `${(d.seconds_left/60).toFixed(1)} min left`
    + `<small>${d.verdict}</small>`;

  if (alertedFor === candleKey) return;      // already announced this one
  alertedFor = candleKey;
  ping();

  try {
    if (window.Notification && Notification.permission === 'granted') {
      new Notification(`${coin()} ${d.lean.toUpperCase()} ${(d.confidence*100).toFixed(0)}%`,
        {body: `${(d.seconds_left/60).toFixed(1)} min left · `
               + `${(d.agreement*100).toFixed(0)}% agree · score ${d.score.toFixed(2)}`});
    } else if (window.Notification && Notification.permission === 'default') {
      Notification.requestPermission();
    }
  } catch (e) {}
}

/* ---- the live feed -------------------------------------------------------
   Polling asks "what happened"; the stream is told. Once the socket is up,
   every fill pushes a fresh read instead of waiting for the next poll, which
   is the difference between seeing a move and reading about it.            */

let stream = null, feedTimer = null;

async function startFeed() {
  $('feedBtn').disabled = true;
  try {
    const r = await api('/api/feed?' + q({coin: coin(),
                        intervals: '1m,5m,15m,30m'}), {method: 'POST'});
    if (!r.ok) { note(r.error || 'could not start feed', true); return; }
    note(`feed starting on ${coin()} — seeding from history…`);
    openStream();
    if (!feedTimer) feedTimer = setInterval(paintFeed, 2000);
  } catch (e) { note(e.message, true); }
  finally { $('feedBtn').disabled = false; }
}

async function stopFeed() {
  closeStream();
  if (feedTimer) { clearInterval(feedTimer); feedTimer = null; }
  try { await api('/api/feed?stop=true', {method: 'POST'}); } catch (e) {}
  $('feedBar').textContent = 'Feed off — the current candle is being polled.';
  note('feed stopped');
}

function openStream() {
  closeStream();
  // EventSource cannot set headers, so the token travels in the query. It is
  // encoded like every other parameter.
  const url = '/api/stream?' + q({coin: coin(), interval: $('rInt').value,
                                  higher: $('rHigh').value, token: tok()});
  stream = new EventSource(url);

  stream.onmessage = (ev) => {
    try { paintRead(JSON.parse(ev.data)); } catch (e) {}
  };

  // Price arrives on its own channel, on every batch of fills. The full read
  // is coalesced; the price is not, because in a fast market it is the number
  // that goes stale first and a stale price reads as a current one.
  stream.addEventListener('px', (ev) => {
    try {
      const p = JSON.parse(ev.data);
      tickFromStream(p);
    } catch (e) {}
  });
  stream.addEventListener('hello', (ev) => {
    try {
      const h = JSON.parse(ev.data);
      note(h.feed ? 'streaming ' + h.coin + ' from the websocket'
                  : 'streaming, but no socket feed — still polled');
    } catch (e) {}
  });
  stream.addEventListener('error', () => {
    // EventSource reconnects on its own; say so rather than looking dead.
    $('feedBar').innerHTML = '<b>Stream interrupted</b> — reconnecting…';
  });
}

function closeStream() {
  if (stream) { stream.close(); stream = null; }
}

async function paintFeed() {
  let f;
  try { f = await api('/api/feed'); } catch (e) { return; }
  const bar = $('feedBar');
  if (!f.running) {
    bar.textContent = 'Feed off — the current candle is being polled.';
    return;
  }
  const age = f.age_s == null ? null : f.age_s;
  const cls = f.stale ? 'short' : 'long';
  bar.innerHTML = `<b class="${cls}">${f.stale ? 'FEED SILENT' : 'LIVE'}</b> `
    + `${f.coin} · ${f.trades_seen} fills · ${f.book_updates} book updates · `
    + `last message ${ageText(age)}`
    + (f.reconnects ? ` · ${f.reconnects} reconnects` : '')
    + (f.errors && f.errors.length ? `<br><span style="color:var(--short)">${f.errors[f.errors.length-1]}</span>` : '');
}

let readPoll = null;

function toggleReadAuto() {
  if (readPoll) { clearInterval(readPoll); readPoll = null; }
  if ($('rAuto').checked) { readPoll = setInterval(loadRead, 10000); loadRead(); }
}

async function loadRead() {
  let d;
  try {
    d = await api('/api/read?' + q({coin: coin(), interval: $('rInt').value,
                  higher: $('rHigh').value,
                  size: parseFloat($('liqSize').value || '0') || 0}));
  } catch (e) { $('readHead').textContent = e.message; return; }
  paintRead(d);
}

// Split out so the websocket stream can paint without a fetch of its own.
function paintRead(d) {
  if (!d) return;
  if (d.error) {
    $('readHead').textContent = d.error;
    $('readSignals').innerHTML = ''; $('readSay').style.display = 'none'; return;
  }

  const cls = d.lean === 'up' ? 'long' : d.lean === 'down' ? 'short' : '';
  const src = (d.source || '').indexOf('websocket') === 0;
  const flags = (src ? '<span class="flag">LIVE TAPE</span>'
                     : '<span class="flag late">POLLED</span>')
              + (d.early ? '<span class="flag late">EARLY</span>' : '')
              + (d.stale ? '<span class="flag crowded">STALE — candle and live price disagree</span>' : '')
              + (d.has_watch ? '' : '<span class="flag">NO LEVEL WATCH — flow and absorption missing</span>');

  $('readHead').innerHTML =
      `<div class="verdict-big ${cls}">${d.lean.toUpperCase()}`
    + `<span style="font-size:15px;color:var(--dim);font-weight:400">`
    + `&nbsp;score ${d.score >= 0 ? '+' : ''}${d.score.toFixed(2)}</span></div>`
    + `<div style="font-size:12px;color:var(--dim)">${flags}</div>`
    + '<div class="conrow">'
    + `<div class="stat"><b>${(d.seconds_left/60).toFixed(1)}m</b><span>left in candle</span></div>`
    + `<div class="stat"><b>${(d.elapsed_fraction*100).toFixed(0)}%</b><span>elapsed</span></div>`
    + `<div class="stat"><b>${(d.agreement*100).toFixed(0)}%</b><span>signals agree</span></div>`
    + `<div class="stat"><b>${(d.confidence*100).toFixed(0)}%</b><span>confidence</span></div>`
    + `<div class="stat"><b class="${d.change_bps >= 0 ? 'long' : 'short'}">`
    + `${d.change_bps >= 0 ? '+' : ''}${d.change_bps.toFixed(1)}bps</b><span>on the candle</span></div>`
    + `<div class="stat"><b>${(d.position_in_range*100).toFixed(0)}%</b><span>of candle range</span></div>`
    + `<div class="stat"><b>${Number(d.last).toLocaleString(undefined,{maximumFractionDigits:6})}</b><span>candle price</span></div>`
    + (d.spot != null
       ? `<div class="stat ${d.stale ? 'gap-bad' : 'gap-ok'}">`
         + `<b>${Number(d.spot).toLocaleString(undefined,{maximumFractionDigits:6})}</b>`
         + `<span>live mid${d.spot_vs_candle_bps != null ? ' (' + (d.spot_vs_candle_bps >= 0 ? '+' : '') + d.spot_vs_candle_bps.toFixed(1) + 'bps)' : ''}</span></div>`
       : '')
    + (d.round_trip_bps != null
       ? `<div class="stat"><b>${d.round_trip_bps.toFixed(1)}bps</b><span>round trip cost</span></div>` : '')
    + '</div>';

  const sigs = d.signals || [];
  $('readSignals').innerHTML = !sigs.length ? '' :
    '<table style="margin-top:12px"><tr><th>signal</th><th class="r">weight</th>'
    + '<th>reading</th></tr>'
    + sigs.map(s => `<tr>
        <td class="${s.direction === 'up' ? 'long' : s.direction === 'down' ? 'short' : ''}">
          ${s.direction === 'up' ? '▲' : s.direction === 'down' ? '▼' : '·'} ${s.name}</td>
        <td class="r ${s.weighted >= 0 ? 'long' : 'short'}">${s.weighted >= 0 ? '+' : ''}${s.weighted.toFixed(2)}</td>
        <td style="color:var(--dim)">${s.note}</td></tr>`).join('')
    + '</table>';

  stamp('readStamp', 0, 30, 120, 'read ');
  checkAlert(d);
  $('readSay').style.display = '';
  $('readSay').innerHTML = d.verdict
    + (d.higher_timeframe ? `<div style="margin-top:6px;color:var(--dim)">Higher timeframe: ${d.higher_timeframe}</div>` : '')
    + (d.vwap ? `<div style="color:var(--dim)">VWAP ${Number(d.vwap.value).toLocaleString(undefined,{maximumFractionDigits:2})} — ${d.vwap.band}</div>` : '');
}

async function doCalibrate() {
  note('replaying history — this measures the lean instead of asserting it…');
  try {
    const r = await api('/api/calibrate?' + q({coin: coin(),
                        interval: $('rInt').value}), {method: 'POST'});
    if (!r.ok) { note(r.error, true); return; }
    const rows = (r.table || []).filter(t => t.n > 0)
      .map(t => `${t.band}: ${(t.up_rate*100).toFixed(0)}% up (n=${t.n})`);
    note(`scored ${r.scored} past candles — ` + (rows.join(' · ') || 'no bands populated'));
    loadRead();
  } catch (e) { note(e.message, true); }
}

let tunedWeights = null;

async function runBacktest() {
  $('btHead').textContent = 'replaying…';
  $('btSay').style.display = 'none';
  $('btCurve').innerHTML = '';
  let d;
  try {
    d = await api('/api/backtest?' + q({coin: coin(),
                  interval: $('rInt').value,
                  bars: parseInt($('btBars').value, 10) || 1000,
                  train_fraction: $('btSplit').value,
                  tune: $('btTune').checked ? 'true' : 'false'}),
                  {method: 'POST'});
  } catch (e) { $('btHead').textContent = e.message; return; }
  if (!d.ok) { $('btHead').textContent = d.error; return; }

  tunedWeights = d.tuned_weights || null;
  const bu = $('btUseTuned');
  if (bu) bu.disabled = !tunedWeights;
  if (tunedWeights) paintWeights(liveWeights, tunedWeights);

  const col = (r, held) => {
    if (!r) return '';
    const acc = r.calls >= 20 ? (r.accuracy * 100).toFixed(1) + '%' : '—';
    const edgeCls = r.calls < 20 ? '' : r.edge_pts > 0 ? 'long' : 'short';
    return `<div class="bt-col${held ? ' held' : ''}"><h3>${r.label}</h3>`
      + `<div class="bt-big ${edgeCls}">${acc}</div>`
      + `<div style="font-size:12px;color:var(--dim)">`
      + `${r.calls} calls from ${r.n} bars (${(r.coverage*100).toFixed(0)}% coverage)<br>`
      + `${r.calls >= 20 ? (r.edge_pts >= 0 ? '+' : '') + r.edge_pts.toFixed(1) + 'pts vs coin flip'
                          : 'too few calls to judge'}</div></div>`;
  };

  $('btHead').innerHTML =
      `<div style="font-size:12px;color:var(--dim);margin-bottom:6px">`
    + `${d.candles} real candles (${d.history_span || 'span unknown'}) · `
    + `${d.samples} scored bars · replay is blind to `
    + `${(d.blind_to || []).join(', ')}</div>`
    + '<div class="bt-grid">'
    + col(d.train, false) + col(d.baseline_test, false) + col(d.test, true)
    + '</div>'
    + `<div style="margin-top:8px;font-size:12px;color:var(--dim)">`
    + `Train minus held-out: <b class="${d.overfit_gap_pts > 8 ? 'short' : ''}">`
    + `${d.overfit_gap_pts >= 0 ? '+' : ''}${d.overfit_gap_pts.toFixed(1)}pts</b>`
    + ` — anything large here is fitted noise.</div>`;

  const curve = (d.test && d.test.curve) || [];
  $('btCurve').innerHTML = !curve.length ? '' :
    '<table style="margin-top:14px"><tr><th>selectivity (held out)</th>'
    + '<th class="r">calls</th><th class="r">accuracy</th><th class="r">coverage</th></tr>'
    + curve.map(r => `<tr>
        <td>score at or above ${r.threshold.toFixed(2)}</td>
        <td class="r">${r.n}</td>
        <td class="r ${r.accuracy == null ? '' : r.accuracy > 0.55 ? 'long' : r.accuracy < 0.45 ? 'short' : ''}">
          ${r.accuracy == null ? '—' : (r.accuracy*100).toFixed(1) + '%'}</td>
        <td class="r">${(r.coverage*100).toFixed(0)}%</td></tr>`).join('')
    + '</table>'
    + '<div style="margin-top:6px;font-size:12px;color:var(--dim)">'
    + 'Being more selective trades trades for accuracy. The row you can live '
    + 'with is the one that decides whether this is worth trading.</div>';

  $('btSay').style.display = '';
  $('btSay').textContent = d.verdict;
  loadForward();
}

async function loadForward() {
  let d;
  try {
    d = await api('/api/forward?' + q({coin: coin(), interval: $('rInt').value}));
  } catch (e) { $('fwd').textContent = e.message; return; }

  const s = d.stats || {};
  const o = s.overall || {n: 0};
  const cell = (label, v) => {
    if (!v || !v.n) return `<div class="stat"><b>—</b><span>${label} (0)</span></div>`;
    return `<div class="stat"><b class="${v.accuracy > 0.55 ? 'long' : v.accuracy < 0.45 ? 'short' : ''}">`
      + `${(v.accuracy*100).toFixed(0)}%</b><span>${label} (${v.n})</span></div>`;
  };

  if (!o.n) {
    $('fwd').innerHTML = '<b>Forward test:</b> nothing scored yet'
      + (s.pending ? ` — ${s.pending} read${s.pending === 1 ? '' : 's'} waiting for their candle to close.`
                   : '. Leave the read running and it will fill up.');
    return;
  }

  $('fwd').innerHTML = '<b>Forward test — measured on live reads</b>'
    + '<div class="conrow" style="margin-top:6px">'
    + cell('overall', o)
    + cell('with flow &amp; absorption', s.with_flow)
    + cell('candles only', s.without_flow)
    + (s.bands || []).map(b => cell(b.band + ' signals', b)).join('')
    + '</div>'
    + (s.pending ? `<div style="color:var(--dim);font-size:12px">${s.pending} awaiting resolution</div>` : '');
}

let liveWeights = {}, defaultWeights = {};

function wmsg(m, bad) {
  const el = $('wMsg');
  el.textContent = m;
  el.className = 'msg' + (bad ? ' err' : '');
}

function paintWeights(w, tuned) {
  liveWeights = Object.assign({}, w);
  $('wGrid').innerHTML = Object.keys(w).sort().map(k => {
    const t = tuned && tuned[k] != null ? tuned[k] : null;
    const diff = t != null && Math.abs(t - w[k]) > 1e-9;
    return `<div class="${diff ? 'changed' : ''}">`
      + `<label>${k}${t != null ? ` <span style="color:var(--accent)">tuned ${t.toFixed(2)}</span>` : ''}</label>`
      + `<input id="w_${k}" type="number" step="0.05" min="0" max="5" value="${w[k]}"></div>`;
  }).join('');
}

function readWeightInputs() {
  const out = {};
  Object.keys(liveWeights).forEach(k => {
    const el = $('w_' + k);
    if (!el) return;
    const v = parseFloat(el.value);
    if (isFinite(v)) out[k] = Math.max(0, Math.min(v, 5));
  });
  return out;
}

async function loadWeights() {
  try {
    const r = await api('/api/weights', {method: 'POST'});
    defaultWeights = r.defaults || r.weights;
    paintWeights(r.weights, tunedWeights);
  } catch (e) { wmsg(e.message, true); }
}

async function applyWeights() {
  const w = readWeightInputs();
  if (!Object.keys(w).length) { wmsg('nothing to apply', true); return; }
  wmsg('applying…');
  try {
    const r = await api('/api/weights', {method: 'POST', body: JSON.stringify(w)});
    if (!r.ok) { wmsg(r.error || 'rejected', true); return; }
    paintWeights(r.weights, tunedWeights);
    wmsg('applied — live reads now use these');
    loadRead();
  } catch (e) { wmsg(e.message, true); }
}

async function resetWeights() {
  wmsg('resetting…');
  try {
    const r = await api('/api/weights?reset=true', {method: 'POST'});
    paintWeights(r.weights, tunedWeights);
    wmsg('back to defaults');
    loadRead();
  } catch (e) { wmsg(e.message, true); }
}

function useTuned() {
  if (!tunedWeights) return;
  Object.keys(tunedWeights).forEach(k => {
    const el = $('w_' + k);
    if (el) el.value = tunedWeights[k];
  });
  wmsg('tuned values loaded into the boxes — press Apply to use them');
}

let watchPoll = null;

async function startWatch() {
  const lvl = parseFloat($('wLevel').value || '0');
  if (!isFinite(lvl) || lvl <= 0) { note('enter the price level you want watched', true); return; }
  const mins = parseFloat($('wMins').value || '15');
  const band = parseFloat($('wBand').value || '10');
  try {
    const r = await api('/api/watch?' + q({coin: coin(), level: lvl,
                        minutes: mins, band_bps: band}), {method: 'POST'});
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
    d = await api('/api/liquidity?' + q({coin: coin(), size: size}));
  } catch (e) { $('liqCost').textContent = e.message; return; }

  if (d.error) {
    $('liqCost').textContent = d.error;
    $('liqBook').innerHTML = ''; $('liqAbs').style.display = 'none';
    $('liqShelves').innerHTML = ''; return;
  }

  stamp('liqStamp', d.book_age_s != null ? d.book_age_s : 0, 30, 120, 'book ');

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
    d = await api('/api/consensus?' + q({coin: coin(), winners_only: wo,
                  min_notional: mn, min_account: ma}));
  } catch (e) { $('consensus').textContent = e.message; return; }

  if (d.error) { $('consensus').textContent = d.error; $('conTraders').innerHTML = ''; return; }

  // The stamp here is the SWEEP's age, not this fetch's. Consensus is built
  // from stored positions, so a fresh HTTP call can still serve half-hour-old
  // data and stamping the request time would say it was current.
  let sweepAge = null;
  if (d.as_of) {
    const t = Date.parse(d.as_of);
    if (!isNaN(t)) sweepAge = (Date.now() - t) / 1000;
  }
  stamp('conStamp', sweepAge, 2700, 5400, 'swept ');

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
  const d = await api('/api/map?' + q({coin: coin()}));
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
  const d = await api('/api/changes?' + q({hours: 24, coin: coin()}));
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
  const d = await api('/api/positions?' + q({coin: coin(), top: 40}));
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
    const r = await api('/api/sweep?' + q({coin: coin()}), {method: 'POST'});
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

let allMarkets = [];

async function loadCoins() {
  // The picker lists what you can TRADE, not what happens to have been swept.
  // With well over a thousand markets, a flat alphabetical list is a haystack,
  // so they are grouped by asset class and filterable by ticker.
  try {
    const d = await api('/api/markets');
    const rows = d.markets || [];
    if (!rows.length) { $('mktCount').textContent = d.error || 'no markets'; return; }

    allMarkets = rows;
    hip3Markets = {};
    rows.forEach(r => { if (r.hip3) hip3Markets[r.symbol] = r.dex; });

    const cls = d.classes || [];
    $('klass').innerHTML = `<option value="">all (${d.count})</option>`
      + cls.map(c => `<option value="${c.klass}">${c.label} (${c.n})</option>`).join('');

    renderCoins();
    $('mktCount').textContent = (d.stale ? 'cached — exchange unreachable' : '');
  } catch (e) { $('mktCount').textContent = e.message; }
}

function renderCoins() {
  const want = $('klass').value;
  const find = ($('mktFind').value || '').trim().toUpperCase();
  const keep = allMarkets.filter(r =>
    (!want || r.klass === want) && (!find || r.base.toUpperCase().includes(find)));

  if (!keep.length) {
    $('coin').innerHTML = '<option value="">no match</option>';
    $('mktCount').textContent = '0 of ' + allMarkets.length;
    return;
  }

  // Group by class, then by venue inside it, so a builder's gold and the
  // canonical crypto never sit in the same undifferentiated list.
  const groups = {};
  keep.forEach(r => {
    const g = r.klass_label + (r.dex ? ' · ' + r.dex : '');
    (groups[g] || (groups[g] = [])).push(r);
  });

  $('coin').innerHTML = Object.keys(groups).map(g =>
    `<optgroup label="${g}">`
    + groups[g].map(r =>
        `<option value="${r.symbol}">${r.base}${r.has_data ? ' •' : ''}</option>`).join('')
    + '</optgroup>').join('');

  const still = keep.some(r => r.symbol === lastCoin);
  if (still) $('coin').value = lastCoin;
  $('mktCount').textContent = keep.length === allMarkets.length
    ? allMarkets.length + ' markets'
    : keep.length + ' of ' + allMarkets.length;
  showVenueNote();
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
