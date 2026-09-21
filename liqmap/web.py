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

    def start_harvest(self, minutes: float, then_sweep: bool = True) -> dict[str, Any]:
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
        self.harvest.update({"running": True, "found": 0, "error": None,
                             "finished": None, "minutes": minutes,
                             "started": datetime.now(timezone.utc).isoformat()})

        def run() -> None:
            try:
                from .hl import TradeHarvester
                h = TradeHarvester(cfg.coins, min_notional=cfg.min_wallet_notional)
                h.run(seconds=minutes * 60,
                      on_progress=lambda n: self.harvest.update({"found": n}))

                wallets = h.top_wallets(cfg.wallet_limit)
                self.set_wallets(wallets)
                self.harvest.update({"found": len(wallets)})

                if then_sweep and wallets:
                    for coin in cfg.coins:
                        try:
                            self.sweep(coin)
                        except Exception as exc:
                            self.last_error = f"post-harvest sweep {coin}: {exc}"
            except Exception as exc:
                self.harvest["error"] = str(exc)
                self.last_error = f"harvest: {exc}"
            finally:
                self.harvest["running"] = False
                self.harvest["finished"] = datetime.now(timezone.utc).isoformat()

        threading.Thread(target=run, daemon=True).start()
        return {"ok": True, **self.harvest}

    # -- the actual work ---------------------------------------------------

    def sweep(self, coin: str) -> dict[str, Any]:
        """One sweep: prices, positions, map, change detection.

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
            mids = client.all_mids()
            spot = mids.get(coin)
            if not spot:
                raise RuntimeError(f"no mid price for {coin}")

            self.store().log_price(coin, spot)

            positions, failed = sweep_positions(client, wallets[:cfg.wallet_limit])
            sweep_id, changes = self.history.record_sweep(coin, spot, positions)
            self.last_sweep[coin] = datetime.now(timezone.utc).isoformat()

            notable = [c for c in changes if c.kind != "HELD"]
            return {
                "coin": coin,
                "spot": spot,
                "sweep_id": sweep_id,
                "wallets_swept": len(wallets[:cfg.wallet_limit]),
                "wallets_failed": failed,
                "positions": len([p for p in positions if p.coin == coin]),
                "changes": len(notable),
                "defended": len([c for c in notable if c.kind == "DEFENDED"]),
                "liquidated": len([c for c in notable if c.kind == "LIQUIDATED"]),
            }

    def maps(self, coin: str) -> dict[str, Any]:
        """Both maps from the most recent stored sweep."""
        cfg = self.settings()
        latest = self.history.latest_sweep_positions(coin)
        if not latest:
            return {"coin": coin, "error": "no sweep recorded yet"}

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
                for coin in cfg.coins:
                    try:
                        rt.sweep(coin)
                    except Exception as exc:
                        rt.last_error = f"sweep {coin}: {exc}"

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
        coin = coin.upper()
        latest = rt.history.latest_sweep_positions(coin)
        if not latest:
            return {"coin": coin, "error": "no sweep recorded yet"}

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

    @app.post("/api/harvest", dependencies=[Depends(require_token)])
    def post_harvest(minutes: float = 5.0,
                     then_sweep: bool = True) -> JSONResponse:
        """Collect a wallet universe off the trade feed.

        Five minutes is enough to see the thing work. A real universe wants
        an hour or more -- the longer it listens, the more of the large
        participants it catches, and those are the ones whose liquidations
        matter.
        """
        result = rt.start_harvest(min(max(minutes, 0.5), 180.0), then_sweep)
        return JSONResponse(result, status_code=200 if result.get("ok") else 409)

    @app.post("/api/sweep", dependencies=[Depends(require_token)])
    def post_sweep(coin: str = "BTC") -> JSONResponse:
        try:
            return JSONResponse({"ok": True, **rt.sweep(coin.upper())})
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

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
  label{font-size:12px;color:var(--dim);display:block;margin:8px 0 2px}
</style></head><body>
<div class="wrap">
  <h1>liqmap</h1>
  <div class="sub">Liquidation pressure, position strength, and what changed since the last sweep.</div>

  <div class="bar">
    <input type="password" id="tok" placeholder="access token">
    <input id="coin" value="BTC" size="6">
    <button class="go" onclick="loadAll()">Load</button>
    <button onclick="doHarvest()">Harvest wallets</button>
    <button onclick="doSweep()">Sweep now</button>
    <button onclick="doResolve()">Resolve</button>
    <span id="msg" class="msg"></span>
  </div>

  <div class="grid">
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
    await Promise.all([loadStatus(), loadMap(), loadChanges(), loadPositions(), loadReport()]);
    note('updated ' + new Date().toLocaleTimeString());
  } catch (e) { note(e.message, true); }
}

let harvestPoll = null;

async function doHarvest() {
  const mins = prompt('Listen to the trade feed for how many minutes?\n\n' +
    '5 is enough to see it work. An hour or more builds a universe worth trusting.', '5');
  if (!mins) return;
  try {
    const r = await api('/api/harvest?minutes=' + encodeURIComponent(mins) + '&then_sweep=true',
                        {method: 'POST'});
    if (!r.ok) { note(r.error || 'could not start', true); return; }
    note('harvesting… the page will update as wallets come in');
    if (harvestPoll) clearInterval(harvestPoll);
    harvestPoll = setInterval(async () => {
      try {
        const h = await api('/api/harvest');
        if (h.running) { note(`harvesting… ${h.found} wallets so far`); }
        else {
          clearInterval(harvestPoll); harvestPoll = null;
          note(h.error ? ('harvest failed: ' + h.error) : `harvest done — ${h.wallets} wallets, sweeping…`, !!h.error);
          setTimeout(loadAll, 3000);
        }
      } catch (e) { clearInterval(harvestPoll); harvestPoll = null; }
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
    note(r.ok ? `swept ${r.positions} positions, ${r.changes} changes (${r.defended} defended, ${r.liquidated} liquidated)`
              : r.error, !r.ok);
    if (r.ok) loadAll();
  } catch (e) { note(e.message, true); }
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
