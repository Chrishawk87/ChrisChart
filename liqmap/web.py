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

from fastapi import (Depends, FastAPI, File, Header, HTTPException, Query,
                     UploadFile)
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from . import consensus as consensus_mod
from .bucket import build_map, render
from .history import History, render_changes
from .settings import Settings, SettingsStore, default_db_path
from .ledger import Ledger
from dataclasses import replace
from .autopilot import Autopilot, Knobs, KNOBS
from . import score as score_mod
from . import tuner as tuner_mod
from . import sweep as sweep_mod
from . import vote as vote_mod
from .strength import fragility_weighter, score_all, summarise

APP_TITLE = "liqmap"

# Trade size assumed when none is given, in dollars. It is a stated default
# rather than a borrowed one: size determines the round-trip cost, which
# determines whether a suggestion is offered at all, so it must not be tied
# to an unrelated setting that someone might reasonably change.
DEFAULT_TRADE_NOTIONAL = 10_000.0

# Largest chart-history upload accepted, in bytes. Enforced against the
# declared length BEFORE the body is read, because a cap checked after
# `await file.read()` has already allocated the whole thing.
MAX_UPLOAD_BYTES = 32 * 1024 * 1024

# Most bars kept from one upload. A 14MB CSV parses to roughly 400k bars and
# half a gigabyte of resident memory; this bounds that regardless of what the
# file claims to be.
MAX_UPLOAD_BARS = 250_000


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------

class Runtime:
    """Shared, lazily built state. One database, reused connections."""

    def __init__(self) -> None:
        self.db_path = default_db_path()
        self.settings_store = SettingsStore(self.db_path)
        self.history = History(self.db_path)
        # The agent's own book, separate from yours on purpose: yours is
        # filtered by your judgement, which is the thing it is being
        # measured against.
        self.ledger = Ledger(self.db_path)
        self._pilots: dict[tuple[str, str], Autopilot] = {}
        self.autopilot: dict[str, Any] = {
            "on": False, "coin": None, "interval": "15m", "mode": "scalp",
            "notional": 10_000.0, "fee_bps": 0.0,
            "raw": True, "exit_on_invalidation": False,
            "tp_bps": 20.0, "sl_bps": 15.0,
            "last_step": None, "last_decision": None, "steps": 0,
            "error": None}
        self._last_tune = 0.0
        self.token = os.environ.get("LIQMAP_TOKEN", "").strip()
        self.wallets: list[str] = []
        self.last_error: str | None = None
        self.last_traceback: str | None = None
        self.last_sweep: dict[str, str] = {}
        self.worker_started = False
        self.harvest: dict[str, Any] = {"running": False, "found": 0,
                                        "started": None, "finished": None,
                                        "error": None, "minutes": 0}
        self.backfill: dict[str, Any] = {"running": False, "progress": None,
                                         "plan": None, "error": None}
        # Agreement memory, one per (market, timeframe). Held here rather
        # than on the feed so switching markets does not lose the history
        # of the one you came from.
        self._trackers: dict[tuple[str, str], Any] = {}
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

    def suggest_for(self, coin: str, interval: str = "15m",
                    notional: float = 0.0, fee_bps: float = 0.0,
                    record: bool = True, mode: str = "scalp"
                    ) -> dict[str, Any]:
        """A trade to take or ignore on this market, or why there isn't one.

        Everything the suggestion needs comes from the live feed: the book
        read, the book itself, how much of the candle is left and what recent
        bars on this timeframe actually travel. If the feed is not running
        there is no suggestion, and saying so is the correct answer -- the
        alternative is a target sized from a stale snapshot.
        """
        from . import runway as runway_mod
        from .confirm import (AgreementTracker, from_feed as read_price_action,
                              grade_three, participation_from_feed)
        from .delta import from_feed as read_delta_column, sticky as delta_sticky
        from .live import INTERVALS
        from .suggest import assess as make_call

        coin = self.resolve_symbol(coin)[0] or coin

        # An unknown interval must not silently become 900 seconds. The
        # candle grid, the recorded `candle_ts` and the resolver's own
        # lookup would each use a different length, and the row could never
        # be settled -- it would simply sit pending forever.
        if interval not in INTERVALS:
            return {"ok": False, "coin": coin, "take": False,
                    "gate": "bad interval",
                    "detail": f"{interval!r} is not a timeframe this feed "
                              f"builds ({', '.join(sorted(INTERVALS))})",
                    "sentence": f"No trade — {interval!r} is not a known "
                                f"timeframe."}

        feed = self.feed_for(coin)
        if feed is None:
            return {"ok": False, "coin": coin, "take": False,
                    "gate": "no feed",
                    "detail": "no live feed on this market — press Go live",
                    "sentence": "No trade — the feed is not running on this "
                                "market, so there is no book to read."}

        interval_s = float(INTERVALS[interval])
        read = feed.book_call()
        with feed._lock:
            book = feed.book

        live = feed.candle(interval)
        if live is None:
            feed.ensure_interval(interval)
            live = feed.candle(interval)

        now = time.time()
        if live is not None:
            seconds_left = max(0.0, live.end_ts - now)
            candle_ts, candle_end = live.start_ts, live.end_ts
        else:
            from .live import grid_start
            start = grid_start(now, interval_s)
            seconds_left = max(0.0, start + interval_s - now)
            candle_ts, candle_end = start, start + interval_s

        # Closed bars for the range estimate. The feed's own history is
        # preferred -- it is what this candle is actually following -- and
        # the exchange fills in when the feed has not been up long.
        recent = feed.history(interval)
        if len(recent) < 10:
            try:
                recent = self.client().candles(coin, interval, bars=60)[:-1]
            except Exception:
                pass

        # A fixed, stated default rather than `min_wallet_notional`, which is
        # the "smallest whale position worth harvesting" knob and has nothing
        # to do with trade size. Size drives the round trip, which drives the
        # main gate, so an operator lowering an unrelated setting would have
        # silently made every suggestion look cheaper.
        if notional <= 0:
            notional = DEFAULT_TRADE_NOTIONAL

        # What this market and timeframe is actually achieving, so the
        # suggestion can put the rate it NEEDS next to the rate you GET.
        # That pair is the whole decision, and neither half means much
        # alone.
        measured_rate, measured_n = None, 0
        try:
            stats = self.history.suggestion_stats(coin, interval)["overall"]
            measured_rate, measured_n = stats["hit_rate"], stats["n"]
        except Exception:
            pass

        # What price is ACTUALLY doing, from fills only. This is the
        # independent half of the comparison: it never touches the book, so
        # agreement between the two means something.
        #
        # The window is scoped to the timeframe -- thirty seconds is the
        # right question on a 15m bar and far too long on a 1m one.
        window_s = max(10.0, min(interval_s / 30.0, 120.0))
        action = read_price_action(feed, interval, window_s=window_s)

        # Was it paid for? The piece that decides whether an agreement holds
        # together or evaporates between seeing it and acting on it.
        part = participation_from_feed(feed, interval, interval_s, window_s)

        # The third column: who is crossing the spread. Independent of the
        # book by construction — `delta.py` never reads one.
        tape = read_delta_column(feed, interval, interval_s, window_s)

        # Agreement needs an AGE, and age needs memory that survives the
        # poll. One tracker per market and timeframe.
        tracker = self._trackers.setdefault((coin, interval),
                                            AgreementTracker())
        book_score = read.score if read is not None else 0.0
        price_score = action.score if action is not None else 0.0
        tracker.classify(book_score, price_score)
        held, flips = tracker.observe(
            "agree" if (read is not None and action is not None
                        and read.direction == action.direction
                        and read.direction != "flat")
            else "no",
            read.direction if read is not None else "flat", now)

        # Three-way grade. Price is mandatory: it is the only column that
        # reports an outcome rather than an intention.
        tracker_delta = delta_sticky(tape.score if tape else 0.0,
                                     getattr(tracker, "delta_dir", None))
        tracker.delta_dir = tracker_delta
        three = grade_three(
            tracker.book_dir or "flat", tracker_delta,
            tracker.price_dir or "flat")

        call = make_call(
            read, book, coin=coin, interval=interval, interval_s=interval_s,
            seconds_left=seconds_left, recent=recent, notional=notional,
            fee_bps=fee_bps, mode=("scalp" if mode == "scalp" else "range"),
            action=action, require_confirmation=True,
            held_s=held, flips=flips, participation=part,
            measured_rate=measured_rate, measured_n=measured_n)
        out = call.suggestion

        payload = call.to_dict()
        payload["ok"] = True
        payload["take"] = call.tradeable
        payload["candle_ts"] = candle_ts
        payload["notional"] = notional
        payload["mode"] = mode
        payload["feed_age_s"] = feed.age
        payload["book_updates"] = feed.book_updates
        payload["three_way"] = three.to_dict()
        payload["delta"] = tape.to_dict() if tape else None

        # How far can it go? Direction quality and holding distance are
        # different questions and are answered separately.
        live_bar_now = feed.candle(interval)
        prof_now = getattr(live_bar_now, "profile", None) if live_bar_now else None
        entry_px = 0.0
        if out is not None:
            entry_px = out.entry
        elif book is not None and not book.empty:
            entry_px = book.mid
        if entry_px > 0 and three.direction != "flat":
            mag_bps, mag_not = self.magnet_for(coin, entry_px)
            road = runway_mod.measure(
                entry=entry_px,
                side="long" if three.direction == "up" else "short",
                book=book, profile=prof_now, bars=recent,
                magnet_bps=mag_bps, magnet_notional=mag_not)
            payload["runway"] = road.to_dict()
            if out is not None:
                payload["runway"]["holdable"] = road.holdable(out.target_bps)
        payload["updates_per_s"] = feed.updates_per_s
        payload["feed_quality"] = feed.feed_quality
        payload["fast_book"] = feed.fast_book
        if out is not None:
            # Flatten the trade onto the top level too, so the panel and the
            # existing clients can read it without reaching into a nested
            # object.
            payload.update({k: v for k, v in out.to_dict().items()
                            if k not in ("coin", "interval")})

        # Record it so it can be settled later. Only real suggestions are
        # stored: a refusal is not a trade and counting refusals would let
        # the tool improve its own hit rate by declining more often.
        # EVERY candle's agreement state is recorded, tradeable or not, with
        # the raw readings behind it. This is the only way to answer what the
        # confirmation rule is actually worth: measuring only the candles
        # that passed the filter and concluding the filter works is the
        # oldest mistake there is.
        if record and call.confirmation is not None:
            feats: dict[str, Any] = {}
            if read is not None:
                feats.update({
                    "tilt": round(read.tilt, 4),
                    "imbalance": round(read.imbalance, 4),
                    "replenish": round(read.replenish, 4),
                    "depletion": round(read.depletion, 4),
                    "aggression": round(read.aggression, 4),
                    "absorbed": bool(read.absorbed),
                    "spread_bps": round(read.spread_bps, 4),
                    "book_score": round(read.score, 4),
                    "book_samples": read.samples,
                })
            if action is not None:
                feats.update({
                    "slope_bps": round(action.slope_bps, 4),
                    "thrust_bps": round(action.thrust_bps, 4),
                    "position": round(action.position, 4),
                    "extending": action.extending,
                    "price_score": round(action.score, 4),
                })

            # The third source: forced flow. Stored rather than acted on,
            # so the question "does a liquidation cluster in the called
            # direction improve a confirmed candle" becomes something the
            # feature slice can answer from data already collected, instead
            # of a weight guessed at now and argued about later.
            # Stability and participation, stored so the table can later
            # answer "do agreements that held 10s beat fresh ones" from
            # data already collected rather than from a guessed threshold.
            feats["held_s"] = round(held, 2)
            feats["flips"] = flips
            feats["grade_3way"] = three.grade
            feats["agreeing"] = three.agreeing
            if tape is not None:
                feats["delta_score"] = tape.score
                feats["delta_lean"] = round(tape.lean, 4)
                feats["delta_effort"] = round(tape.effort, 3)
                feats["delta_persistence"] = tape.persistence
                feats["delta_absorbed"] = tape.absorbed
                feats["delta_working"] = tape.working
                feats["delta_divergent"] = tape.divergent
            road_d = payload.get("runway")
            if road_d:
                feats["runway_bps"] = road_d.get("clear_bps")
                feats["runway_open"] = road_d.get("open_road")
            if part is not None:
                feats["effort"] = round(part.effort, 3)
                feats["effort_aligned"] = round(part.aligned, 3)

            # Where the business was done inside this bar.
            live_bar = feed.candle(interval)
            prof = getattr(live_bar, "profile", None) if live_bar else None
            if prof is not None and prof.total_notional > 0:
                px = live_bar.close
                feats["vol_position"] = round(prof.position(px), 4)
                feats["vol_poc_bps"] = (
                    round((px - prof.poc) / px * 10_000.0, 2)
                    if prof.poc else None)
                if out is not None:
                    # Volume standing between price and the target: the
                    # number that says whether a thin book ahead is air or
                    # a level that already traded and will be defended.
                    feats["vol_ahead"] = round(
                        prof.ahead_ratio(px, out.target_px), 4)

            spot = book.mid if book is not None and not book.empty else 0.0
            if spot > 0:
                mag_bps, mag_notional = self.magnet_for(coin, spot)
                feats["magnet_bps"] = round(mag_bps, 2)
                feats["magnet_notional"] = round(mag_notional, 0)
                # Signed FOR the book's call: +1 the fuel is where the book
                # wants to go, -1 it is behind us.
                if mag_notional > 0 and call.confirmation.book != "flat":
                    want_up = call.confirmation.book == "up"
                    feats["magnet_with_call"] = (
                        1 if (mag_bps > 0) == want_up else -1)
                else:
                    feats["magnet_with_call"] = 0
            try:
                self.history.record_state(
                    coin=coin, interval=interval, candle_ts=candle_ts,
                    candle_end=candle_end,
                    book_dir=call.confirmation.book,
                    book_strength=call.confirmation.book_strength,
                    price_dir=call.confirmation.candle,
                    price_strength=call.confirmation.candle_strength,
                    verdict=call.confirmation.verdict,
                    price=(book.mid if book is not None and not book.empty
                           else 0.0),
                    elapsed_frac=(1.0 - seconds_left / interval_s
                                  if interval_s > 0 else 0.0),
                    made_at=now, features=feats)
            except Exception as exc:
                payload["state_error"] = str(exc)

        # Only tradeable calls become trades. A graded look at a candle that
        # was never tradeable is not a trade, and scoring refusals would let
        # the tool raise its own hit rate by declining more often.
        if record and out is not None:
            sid = self.history.record_suggestion(
                coin=coin, interval=interval, candle_ts=candle_ts,
                candle_end=candle_end, side=out.side, entry=out.entry,
                target_px=out.target_px, stop_px=out.stop_px,
                target_bps=out.target_bps, risk_bps=out.risk_bps,
                rr=out.rr, cost_bps=out.cost_bps,
                conviction=out.conviction, score=out.score,
                reason="; ".join(out.reasons), made_at=now)
            if sid is None:
                existing = self.history.open_suggestion(coin, interval,
                                                        candle_ts)
                sid = existing["id"] if existing else None
                if existing:
                    payload["decision"] = existing["decision"]
            payload["id"] = sid
        return payload

    def resolve_suggestions(self, limit: int = 300) -> dict[str, Any]:
        """Settle suggestions whose candle has closed.

        Two things make this harder than settling a read, and both of them
        flatter the result if they are got wrong.

        THE WINDOW STARTS WHEN THE SUGGESTION DID. A suggestion made ten
        minutes into a fifteen-minute bar must not be settled against that
        bar's full high and low. A bar that spiked 300bps in its first two
        minutes and came back would record a target hit that happened eight
        minutes before the trade existed. So the settling window runs from
        `made_at` to the candle's close, and it is built from ONE-MINUTE bars
        covering that span.

        A TRADE IS DECIDED BY WHICH LEVEL WAS REACHED, not by the close. A
        close-only resolution records a winner as a loser every time price
        touched the target and came back.
        """
        pending = self.history.pending_suggestions(time.time(), limit=limit)
        if not pending:
            return {"ok": True, "resolved": 0, "pending": 0}

        by_coin: dict[str, list[dict[str, Any]]] = {}
        for row in pending:
            by_coin.setdefault(row["coin"], []).append(row)

        resolved = failed = 0
        errors: list[str] = []
        client = self.client()

        for coin, rows in by_coin.items():
            try:
                # One-minute bars are the finest the exchange serves, and
                # they are what makes a sub-candle window measurable at all.
                fine = client.candles(coin, "1m", bars=1500)
            except Exception as exc:
                errors.append(f"{coin}: {exc}")
                failed += len(rows)
                continue

            for row in rows:
                start = float(row["made_at"] or 0) or float(row["candle_ts"])
                end = float(row["candle_end"])
                window = [b for b in fine if start - 60 <= b.ts < end]
                if not window:
                    # The span rolled out of the 1m history before it could
                    # be settled. Never settle against the wrong bars: a
                    # fabricated outcome poisons the measurement
                    # permanently, while a missing one is merely missing.
                    #
                    # Rows too old to ever come back are abandoned rather
                    # than left pending, because `pending_suggestions` is
                    # ordered oldest-first: one permanently unresolvable row
                    # would otherwise sit at the head of the queue and block
                    # every newer row behind it.
                    if end < time.time() - 86_400:
                        self.history.abandon_suggestion(row["id"])
                    failed += 1
                    continue

                if self.history.resolve_suggestion(
                        row["id"],
                        high=max(b.high for b in window),
                        low=min(b.low for b in window),
                        close=window[-1].close):
                    resolved += 1

        return {"ok": True, "resolved": resolved, "unresolvable": failed,
                "pending": len(self.history.pending_suggestions(time.time())),
                "errors": errors[:5]}

    def start_backfill(self, coin: str, interval: str, days: float,
                       at: float = 0.33, max_gib: float = 2.0
                       ) -> dict[str, Any]:
        """Fill the agreement table from Hyperliquid's own book archive.

        Runs in the background because it is slow and costs money: the
        bucket is requester-pays, so every hour downloaded is on the
        caller's AWS bill. The job reports bytes as it goes and stops at
        the cap rather than discovering the total afterwards.
        """
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz

        from .archive import Archive, estimate
        from .backfill import Progress, backfill as run_backfill
        from .live import INTERVALS

        if self.backfill.get("running"):
            return {**self.backfill, "ok": False,
                    "error": "a backfill is already running"}
        if interval not in INTERVALS:
            return {"ok": False,
                    "error": f"{interval!r} is not a timeframe this service "
                             f"builds"}

        coin = self.resolve_symbol(coin)[0] or coin
        interval_s = float(INTERVALS[interval])

        archive = Archive(max_bytes=int(max(0.1, max_gib) * 1024 ** 3))
        if not archive.credentials_present():
            return {"ok": False,
                    "error": "no AWS credentials found. The Hyperliquid "
                             "archive is requester-pays, so there has to be "
                             "an account to charge — set AWS_ACCESS_KEY_ID "
                             "and AWS_SECRET_ACCESS_KEY in Railway"}

        end = _dt.now(_tz.utc).replace(minute=0, second=0, microsecond=0)
        start = end - _td(days=max(0.04, min(days, 30.0)))

        plan = estimate([coin], start, end)
        prog = Progress(coin=coin, interval=interval)
        self.backfill = {"running": True, "ok": True, "plan": plan,
                         "progress": prog.to_dict(), "error": None}

        def run() -> None:
            try:
                # One-minute candles for the whole window, from the
                # exchange. These are built from FILLS, which is what keeps
                # the price side independent of the book side.
                need = int((end - start).total_seconds() // 60) + 10
                minute_bars = self.client().candles(
                    coin, "1m", bars=min(need, 5000))
                if not minute_bars:
                    prog.error = ("no one-minute candles came back for this "
                                  "window")
                    prog.done = True
                    return

                oldest = min(c.ts for c in minute_bars)
                from_ts = max(start.timestamp(), oldest)
                run_backfill(
                    archive, self.history, coin=coin, interval=interval,
                    interval_s=interval_s,
                    start=_dt.fromtimestamp(from_ts, _tz.utc),
                    end=end, minute_bars=minute_bars, at=at, progress=prog,
                    should_stop=lambda: not self.backfill.get("running"))
            except Exception as exc:                    # noqa: BLE001
                prog.error = f"{type(exc).__name__}: {exc}"
                prog.done = True
            finally:
                self.backfill["running"] = False
                self.backfill["progress"] = prog.to_dict()

        t = threading.Thread(target=run, daemon=True)
        t.start()
        return self.backfill

    def magnet_for(self, coin: str, spot: float) -> tuple[float, float]:
        """The nearest liquidation cluster, signed, and its size.

        A THIRD kind of evidence, and qualitatively different from the other
        two. The order book is resting orders, which are voluntary and can be
        pulled in a microsecond. Price action is what has already traded.
        A liquidation cluster is neither: it is flow that MUST happen if
        price reaches the level, because margin runs out whether anyone
        wants it to or not.

        Returns (bps to the cluster, notional). Positive bps means the
        cluster sits above spot -- shorts liquidating, forced buying, a pull
        upward. Negative means longs below, forced selling, a pull down.
        """
        try:
            m = self.maps(coin)
        except Exception:
            return 0.0, 0.0

        clusters = (m.get("weighted") or {}).get("clusters") or []
        at = float(m.get("spot") or spot)
        if at <= 0:
            return 0.0, 0.0

        best = None
        for cl in clusters:
            px = float(cl.get("price") or 0)
            notional = float(cl.get("notional") or 0)
            if px <= 0 or notional <= 0:
                continue
            bps = (px - at) / at * 10_000.0
            # Pull falls off with distance: a huge cluster 5% away is not
            # acting on this candle, a modest one 10bps away is.
            pull = notional / max(abs(bps), 1.0)
            if best is None or pull > best[0]:
                best = (pull, bps, notional)

        return (best[1], best[2]) if best else (0.0, 0.0)

    def resolve_states(self, limit: int = 300) -> dict[str, Any]:
        """Settle recorded agreement states against their closed candles.

        Settled from the read forward, not from the candle's open — a read
        taken a third of the way in must not be credited with a move that
        happened before it existed. So the window runs from `made_at` to the
        close and is built from one-minute bars.
        """
        pending = self.history.pending_states(time.time(), limit=limit)
        if not pending:
            return {"ok": True, "resolved": 0, "pending": 0}

        by_coin: dict[str, list[dict[str, Any]]] = {}
        for row in pending:
            by_coin.setdefault(row["coin"], []).append(row)

        resolved = failed = 0
        errors: list[str] = []
        client = self.client()

        for coin, rows in by_coin.items():
            try:
                fine = client.candles(coin, "1m", bars=1500)
            except Exception as exc:
                errors.append(f"{coin}: {exc}")
                failed += len(rows)
                continue

            for row in rows:
                start = float(row["made_at"] or 0) or float(row["candle_ts"])
                end = float(row["candle_end"])
                window = [b for b in fine if start - 60 <= b.ts < end]
                if not window:
                    failed += 1
                    continue
                if self.history.resolve_state(
                        row["id"], close_px=window[-1].close,
                        high_px=max(b.high for b in window),
                        low_px=min(b.low for b in window)):
                    resolved += 1

        return {"ok": True, "resolved": resolved, "unresolvable": failed,
                "pending": len(self.history.pending_states(time.time())),
                "errors": errors[:5]}

    # ------------------------------------------------------------ autopilot

    def knobs(self) -> Knobs:
        """Live settings: shipped defaults, displaced by adopted proposals."""
        try:
            return Knobs.from_overrides(self.ledger.overrides())
        except Exception:
            return Knobs()

    def pilot(self, coin: str, interval: str) -> Autopilot:
        """One agent per market and timeframe, rehydrated on first use.

        Built lazily so a restart picks up whatever was open rather than
        abandoning it -- an abandoned position never closes, and a trade
        with no exit silently drops out of every average computed later.
        """
        key = (coin, interval)
        p = self._pilots.get(key)
        if p is None:
            p = Autopilot(
                coin, interval, self.ledger, knobs=self.knobs(),
                size_usd=float(self.autopilot.get("notional") or 0),
                raw=bool(self.autopilot.get("raw", True)),
                exit_on_invalidation=bool(
                    self.autopilot.get("exit_on_invalidation", False)))
            self._pilots[key] = p
        return p

    def autopilot_step(self, coin: str, interval: str,
                       mode: str = "scalp", notional: float = 10_000.0,
                       fee_bps: float = 0.0) -> dict[str, Any]:
        """One decision, on exactly what the panel would have shown.

        The agent reads the same payload you do. Deciding on anything else
        would make the screen and the ledger two different stories, and
        there would be no way to audit a call after the fact.
        """
        payload = self.suggest_for(coin, interval, mode=mode,
                                   notional=notional, fee_bps=fee_bps,
                                   record=True)
        if not payload.get("ok"):
            return {"ok": False, "detail": payload.get("detail",
                                                       "no reading")}

        pilot = self.pilot(coin, interval)
        pilot.knobs = replace(
            self.knobs(),
            tp_bps=float(self.autopilot.get("tp_bps") or 20.0),
            sl_bps=float(self.autopilot.get("sl_bps") or 15.0))
        pilot.raw = bool(self.autopilot.get("raw", True))
        pilot.exit_on_invalidation = bool(
            self.autopilot.get("exit_on_invalidation", False))
        pilot.size_usd = notional

        # The bar's extremes, so a level reached between two polls still
        # counts. Polling the last print only would hand the agent a fill
        # model money will not reproduce.
        high = low = 0.0
        feed = self.feed
        if feed is not None:
            bar = feed.candle(interval)
            if bar is not None:
                high, low = bar.high, bar.low

        now = time.time()
        d = pilot.step(payload, now, high=high, low=low)
        out = d.to_dict()
        out["ok"] = True
        out["coin"] = coin
        out["interval"] = interval
        out["grade"] = payload.get("grade")
        out["three_way"] = payload.get("three_way")
        self.autopilot["last_step"] = now
        self.autopilot["last_decision"] = out
        self.autopilot["steps"] = int(self.autopilot.get("steps") or 0) + 1
        return out

    def signals_from_history(self, coin: str, interval: str,
                             since: float | None = None
                             ) -> list[sweep_mod.Signal]:
        """Every stored candle reading, re-read as a book+delta+price vote.

        The readings were stored raw rather than as a verdict, so a rule
        invented today can be scored against candles collected weeks ago.
        That is the difference between testing tonight and testing next
        month.
        """
        rows = self.history.states_for_sweep(coin, interval, since=since)
        out = []
        for r in rows:
            v = vote_mod.from_state(r)
            if v.side is None:
                continue          # nothing to read, not a refusal
            out.append(sweep_mod.Signal(
                ts=float(r.get("made_at") or r.get("candle_ts") or 0.0),
                coin=r["coin"], interval=r["interval"], side=v.side,
                entry=float(r["price"]), agreeing=v.agreeing,
                against=v.against, shape=v.shape(), net=v.net,
                book=v.book, delta=v.delta, price=v.price))
        return out

    def bars_for_sweep(self, coin: str, start: float, end: float,
                       interval: str = "1m") -> list[sweep_mod.Bar]:
        """One-minute bars covering the signal span.

        One minute rather than the signal's own timeframe: a bar that
        touches both levels is settled as a stop, so the ambiguous window
        should be as small as the data allows.
        """
        span_min = max(1.0, (end - start) / 60.0) + sweep_mod.DEFAULT_HORIZON
        want = int(min(5000, max(200, span_min)))
        cs = self.client().candles(coin, interval, bars=want)
        return [sweep_mod.Bar(ts=c.ts, open=c.open, high=c.high,
                              low=c.low, close=c.close) for c in cs]

    def run_sweep(self, coin: str, interval: str = "15m", *,
                  tp_from: float = 5.0, tp_to: float = 40.0,
                  tp_step: float = 5.0, sl_from: float = 5.0,
                  sl_to: float = 40.0, sl_step: float = 5.0,
                  cost_bps: float = 0.0,
                  horizon: int = sweep_mod.DEFAULT_HORIZON,
                  min_agreeing: int = 0, max_against: int = 3,
                  side: str = "both", hours: float = 0.0) -> dict[str, Any]:
        """Every target against every stop, over the signals already stored."""
        since = (time.time() - hours * 3600.0) if hours else None
        signals = self.signals_from_history(coin, interval, since=since)
        if not signals:
            span = self.history.state_span(coin, interval)
            return {"ok": False, "signals": 0,
                    "detail": ("no stored readings for that market and "
                               "timeframe yet — leave the feed running and "
                               "they accumulate one per candle"),
                    "span": span}

        lo = min(s.ts for s in signals)
        hi = max(s.ts for s in signals)
        try:
            bars = self.bars_for_sweep(coin, lo, hi)
        except Exception as exc:
            return {"ok": False, "detail": f"could not fetch bars: {exc}"}

        out = sweep_mod.run(
            signals, bars,
            sweep_mod.axis(tp_from, tp_to, tp_step),
            sweep_mod.axis(sl_from, sl_to, sl_step),
            cost_bps=cost_bps, horizon=horizon,
            min_agreeing=min_agreeing, max_against=max_against, side=side)
        out["coin"] = coin
        out["interval"] = interval
        out["available"] = len(signals)
        out["span_hours"] = round((hi - lo) / 3600.0, 1)
        # How each vote shape did at the best cell, so "do two-of-three
        # trades pay" is answered from the same run rather than guessed.
        if out.get("ok"):
            out["by_shape"] = self._shape_breakdown(
                signals, bars, out["best"], cost_bps, horizon, side)
        return out

    @staticmethod
    def _shape_breakdown(signals, bars, best, cost_bps, horizon, side
                         ) -> list[dict[str, Any]]:
        """The best cell, split by how the columns voted."""
        by: dict[str, sweep_mod.Cell] = {}
        stamps = [b.ts for b in bars]
        import bisect as _bisect
        for s in signals:
            if side != "both" and s.side != side:
                continue
            i = _bisect.bisect_right(stamps, s.ts)
            window = bars[i:i + horizon]
            if not window:
                continue
            cell = by.setdefault(s.shape, sweep_mod.Cell(
                tp_bps=best["tp_bps"], sl_bps=best["sl_bps"],
                cost_bps=cost_bps))
            reason, exit_px, held = sweep_mod.resolve(
                s.side, s.entry, best["tp_bps"], best["sl_bps"], window,
                horizon)
            raw = (exit_px - s.entry) / s.entry * 10_000.0
            gross = raw if s.side == "long" else -raw
            cell.add(reason, gross - cost_bps, held)
        rows = []
        for shape, cell in sorted(by.items(), key=lambda kv: -kv[1].n):
            d = cell.to_dict()
            d["shape"] = shape
            rows.append(d)
        return rows

    def scorecard(self, coin: str | None = None, interval: str | None = None
                  ) -> dict[str, Any]:
        """How the agent has actually done, error bars included."""
        positions = self.ledger.closed(coin=coin, interval=interval)
        decisions = self.ledger.decisions(coin=coin, interval=interval,
                                          limit=5000)
        card = score_mod.scorecard(positions, decisions)
        card["ok"] = True
        card["counts"] = self.ledger.counts()
        card["open"] = [p.to_dict() for p in self.ledger.load_open(coin)]
        card["knobs"] = self.knobs().to_dict()
        return card

    def maybe_tune(self, min_gap_s: float = 3600.0) -> list[dict[str, Any]]:
        """Look for a change worth proposing. Files nothing most of the time.

        Rate limited because the answer only moves when the book grows, and
        re-running a walk-forward every five seconds burns CPU to file the
        same proposal repeatedly.
        """
        now = time.time()
        if now - self._last_tune < min_gap_s:
            return []
        self._last_tune = now
        k = replace(self.knobs(),
                    tp_bps=float(self.autopilot.get("tp_bps") or 20.0),
                    sl_bps=float(self.autopilot.get("sl_bps") or 15.0))
        filed = []

        # In raw mode there are no entry gates left to tighten -- the vote
        # refuses nothing. What is worth tuning is where the levels sit, and
        # only the grid can judge that.
        if self.autopilot.get("raw", True):
            coin = self.autopilot.get("coin")
            interval = self.autopilot.get("interval") or "15m"
            if coin:
                try:
                    one = self.propose_levels_now(coin, interval, k, now=now)
                    if one:
                        filed.append(one)
                except Exception as exc:
                    self.last_error = f"level proposal: {exc}"
        else:
            filed.extend(tuner_mod.propose_all(self.ledger, k, now=now))
        return filed

    def propose_levels_now(self, coin: str, interval: str,
                           knobs=None, now: float | None = None
                           ) -> dict[str, Any] | None:
        """Walk the grid forward over stored signals and file the result."""
        k = knobs or replace(
            self.knobs(),
            tp_bps=float(self.autopilot.get("tp_bps") or 20.0),
            sl_bps=float(self.autopilot.get("sl_bps") or 15.0))
        signals = self.signals_from_history(coin, interval)
        if not signals:
            return None
        lo = min(x.ts for x in signals)
        hi = max(x.ts for x in signals)
        bars = self.bars_for_sweep(coin, lo, hi)
        return tuner_mod.propose_levels(
            self.ledger, k, signals, bars,
            cost_bps=float(self.autopilot.get("fee_bps") or 0.0), now=now)

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
            # ONE request. This runs every couple of seconds, so the
            # every-dex sweep that used to live here was the single largest
            # consumer of the rate-limit budget on the whole service.
            spot = self.client().mid(coin)
            if spot is not None:
                self.last_price[coin] = spot
            self.last_price_ts = now
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
    last_pilot = 0.0

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
                # Suggestions settle on the same tick. Without this the
                # table fills up and nothing in it is ever scored, so
                # `suggestion_stats` reports "nothing settled yet" forever
                # while the backlog grows -- and the decision freeze, which
                # keys off the candle clock, has nothing to compare against.
                try:
                    rt.resolve_suggestions(limit=200)
                except Exception as exc:
                    rt.last_error = f"resolve suggestions: {exc}"
                try:
                    rt.resolve_states(limit=300)
                except Exception as exc:
                    rt.last_error = f"resolve states: {exc}"

            # The agent decides on its own cadence, independent of whether
            # anyone has the panel open. That is the point: a book that only
            # fills while you are watching measures your attention, not the
            # strategy.
            ap = rt.autopilot
            if ap.get("on") and ap.get("coin") and now - last_pilot >= 5:
                last_pilot = now
                try:
                    rt.autopilot_step(
                        ap["coin"], ap.get("interval") or "15m",
                        mode=ap.get("mode") or "scalp",
                        notional=float(ap.get("notional") or 10_000.0),
                        fee_bps=float(ap.get("fee_bps") or 0.0))
                    ap["error"] = None
                except Exception as exc:
                    ap["error"] = str(exc)
                    rt.last_error = f"autopilot: {exc}"
                try:
                    rt.maybe_tune()
                except Exception as exc:
                    rt.last_error = f"tuner: {exc}"

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

                # The book call is cheap -- the reader already did the work
                # on each push -- so it goes out at tape speed alongside the
                # price, not on the read's slower cadence.
                if feed is not None:
                    try:
                        bc = api_bookcall(coin=coin_r)
                        yield f"event: book\ndata: {json.dumps(bc, default=str)}\n\n"
                    except Exception:
                        pass

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

    @app.get("/api/bookcall", dependencies=[Depends(require_token)])
    def api_bookcall(coin: str = "BTC", window_s: float = 20.0
                     ) -> dict[str, Any]:
        """The book's call on the current candle. Nothing else consulted.

        No trend, no higher timeframe, no VWAP, no zones. Just: what is the
        order book doing right now, and which way is this candle going.
        """
        coin = rt.resolve_symbol(coin)[0] or coin
        feed = rt.feed_for(coin)
        if feed is None:
            return {"coin": coin,
                    "error": "no live feed on this market — press Go live"}

        r = feed.book_call(window_s=max(2.0, min(window_s, 120.0)))
        if r is None:
            return {"coin": coin, "error": "no book updates yet",
                    "feed": feed.status()}

        return {
            "coin": coin,
            "direction": r.direction, "score": r.score,
            "conviction": r.conviction,
            "mid": r.mid, "microprice": r.microprice,
            "spread_bps": r.spread_bps,
            "tilt": r.tilt, "imbalance": r.imbalance,
            "depletion": r.depletion, "replenish": r.replenish,
            "mid_drift_bps": r.mid_drift_bps,
            "aggression": r.aggression, "absorbed": r.absorbed,
            "samples": r.samples, "window_s": r.window_s,
            "components": r.components(),
            "verdict": r.verdict(),
            "book_updates": feed.book_updates,
            "age_s": feed.age,
        }

    @app.get("/api/suggest", dependencies=[Depends(require_token)])
    def api_suggest(coin: str = "BTC", interval: str = "15m",
                    size: float = 0.0, fee_bps: float = 0.0,
                    record: bool = True, mode: str = "scalp"
                    ) -> dict[str, Any]:
        """A graded call on this candle, tradeable or not.

        Always answers. `tradeable` is the honest flag and only tradeable
        calls are recorded; the rest are graded D or unlettered and name the
        gate that stopped them, so the panel shows the state of the book
        rather than going blank.

        `mode` is `scalp` or `range`. Scalp aims for the smallest target
        that still pays — the highest hit rate available on a trade worth
        taking. Range aims for half of what a bar typically travels.

        `size` is what you would actually trade, because the round trip
        depends on how far into the book you have to reach. `fee_bps` is your
        own round-trip fee; leaving it at zero makes every suggestion look
        cheaper than it is, and the output says so when it is unset.

        This SUGGESTS. It does not place anything.
        """
        return rt.suggest_for(coin, interval, notional=max(size, 0.0),
                              fee_bps=max(fee_bps, 0.0), record=record,
                              mode=mode)

    @app.post("/api/decide", dependencies=[Depends(require_token)])
    def api_decide(id: str, taken: bool) -> dict[str, Any]:
        """Record that a suggestion was taken or ignored.

        Ignored ones matter as much as taken ones: without them there is no
        way to find out whether the filtering being applied is worth
        anything.
        """
        ok = rt.history.decide(id, taken)
        # `decision` reports what was STORED, not what was asked for. The
        # first cut echoed the request either way, so a refused write came
        # back reading "taken" and the page showed a decision that is not in
        # the table.
        return {"ok": ok, "id": id,
                "decision": ("taken" if taken else "ignored") if ok else None,
                "note": "" if ok else
                        "unknown id, or that candle has already closed — a "
                        "decision cannot be recorded or changed once the "
                        "result is visible"}

    @app.post("/api/resolve-suggestions", dependencies=[Depends(require_token)])
    def api_resolve_suggestions(limit: int = 300) -> dict[str, Any]:
        return rt.resolve_suggestions(limit=limit)

    @app.get("/api/decisions", dependencies=[Depends(require_token)])
    def api_decisions(coin: str = "", interval: str = "", recent: int = 25
                      ) -> dict[str, Any]:
        """How the suggestions did, and how the decisions about them did."""
        c = rt.resolve_symbol(coin)[0] if coin else None
        return {
            "ok": True, "coin": c, "interval": interval or None,
            "stats": rt.history.suggestion_stats(c, interval or None),
            "recent": rt.history.recent_suggestions(c, limit=max(1, min(recent, 200))),
        }

    # ---------------------------------------------------------- autopilot

    @app.post("/api/autopilot", dependencies=[Depends(require_token)])
    def api_autopilot(on: bool, coin: str = "", interval: str = "15m",
                      mode: str = "scalp", notional: float = 10_000.0,
                      fee_bps: float = 0.0, tp_bps: float = 20.0,
                      sl_bps: float = 15.0, raw: bool = True,
                      exit_on_invalidation: bool = False) -> dict[str, Any]:
        """Turn the agent's own book on or off.

        It keeps deciding whether or not anyone is watching. A record that
        only fills while the page is open measures your attention rather
        than the strategy.
        """
        c = rt.resolve_symbol(coin)[0] if coin else rt.autopilot.get("coin")
        if on and not c:
            return {"ok": False, "detail": "name a market first"}
        rt.autopilot.update({
            "on": bool(on), "coin": c, "interval": interval or "15m",
            "mode": mode, "notional": float(notional),
            "fee_bps": float(fee_bps), "tp_bps": float(tp_bps),
            "sl_bps": float(sl_bps), "raw": bool(raw),
            "exit_on_invalidation": bool(exit_on_invalidation),
            "error": None})
        return {"ok": True, "autopilot": rt.autopilot,
                "knobs": rt.knobs().to_dict(),
                "note": (("deciding every 5s on the three columns alone, "
                          f"target {tp_bps:.0f}bps / stop {sl_bps:.0f}bps — "
                          "it records, it never places an order")
                         if on else "stopped; open positions stay open")}

    @app.get("/api/autopilot", dependencies=[Depends(require_token)])
    def api_autopilot_state() -> dict[str, Any]:
        coin = rt.autopilot.get("coin")
        return {"ok": True, "autopilot": rt.autopilot,
                "knobs": rt.knobs().to_dict(),
                "counts": rt.ledger.counts(),
                "open": [p.to_dict() for p in rt.ledger.load_open(coin)],
                "recent": rt.ledger.decisions(coin=coin, limit=40)}

    @app.post("/api/autopilot/step", dependencies=[Depends(require_token)])
    def api_autopilot_step(coin: str, interval: str = "15m",
                           mode: str = "scalp", notional: float = 10_000.0,
                           fee_bps: float = 0.0) -> dict[str, Any]:
        """One decision now, for testing without waiting on the loop."""
        c = rt.resolve_symbol(coin)[0]
        return rt.autopilot_step(c, interval, mode=mode, notional=notional,
                                 fee_bps=fee_bps)

    @app.post("/api/autopilot/close", dependencies=[Depends(require_token)])
    def api_autopilot_close(id: str, price: float = 0.0) -> dict[str, Any]:
        """Close one by hand. Recorded as `manual`, never as a stop.

        Kept honestly separate: a hand-closed trade says nothing about the
        exit rules, and filing it as one would put your decision inside the
        agent's score.
        """
        for key, pilot in rt._pilots.items():
            pos = pilot.position
            if pos is not None and pos.id == id:
                px = price if price > 0 else pos.entry
                closed = rt.ledger.close_position(pos, exit_px=px,
                                                  reason="manual")
                pilot.position = None
                return {"ok": closed is not None, "closed": closed}
        return {"ok": False, "detail": "no open position with that id"}

    @app.get("/api/sweep", dependencies=[Depends(require_token)])
    def api_sweep(coin: str, interval: str = "15m",
                  tp_from: float = 5.0, tp_to: float = 40.0,
                  tp_step: float = 5.0, sl_from: float = 5.0,
                  sl_to: float = 40.0, sl_step: float = 5.0,
                  cost_bps: float = 0.0, horizon: int = 60,
                  min_agreeing: int = 0, max_against: int = 3,
                  side: str = "both", hours: float = 0.0) -> dict[str, Any]:
        """Every target against every stop, over the readings already stored.

        The signal is independent of where the levels sit, so one pass over
        the candle history scores the whole grid.
        """
        c = rt.resolve_symbol(coin)[0]
        return rt.run_sweep(
            c, interval, tp_from=tp_from, tp_to=tp_to, tp_step=tp_step,
            sl_from=sl_from, sl_to=sl_to, sl_step=sl_step,
            cost_bps=cost_bps, horizon=horizon, min_agreeing=min_agreeing,
            max_against=max_against, side=side, hours=hours)

    @app.get("/api/signals", dependencies=[Depends(require_token)])
    def api_signals(coin: str, interval: str = "15m", limit: int = 50
                    ) -> dict[str, Any]:
        """What the three columns have been saying, candle by candle."""
        c = rt.resolve_symbol(coin)[0]
        sigs = rt.signals_from_history(c, interval)
        span = rt.history.state_span(c, interval)
        shapes: dict[str, int] = {}
        for s2 in sigs:
            shapes[s2.shape] = shapes.get(s2.shape, 0) + 1
        return {"ok": True, "coin": c, "interval": interval,
                "n": len(sigs), "span": span,
                "shapes": sorted(shapes.items(), key=lambda kv: -kv[1]),
                "recent": [s2.to_dict() for s2 in sigs[-limit:]][::-1]}

    @app.get("/api/scorecard", dependencies=[Depends(require_token)])
    def api_scorecard(coin: str = "", interval: str = "") -> dict[str, Any]:
        """The agent marking its own homework, error bars included."""
        c = rt.resolve_symbol(coin)[0] if coin else None
        return rt.scorecard(c, interval or None)

    @app.get("/api/ledger", dependencies=[Depends(require_token)])
    def api_ledger(coin: str = "", interval: str = "", limit: int = 100
                   ) -> dict[str, Any]:
        c = rt.resolve_symbol(coin)[0] if coin else None
        rows = rt.ledger.closed(coin=c, interval=interval or None)
        rows = list(reversed(rows))[:max(1, min(limit, 1000))]
        return {"ok": True, "closed": rows,
                "open": [p.to_dict() for p in rt.ledger.load_open(c)],
                "counts": rt.ledger.counts()}

    @app.get("/api/proposals", dependencies=[Depends(require_token)])
    def api_proposals(status: str = "pending") -> dict[str, Any]:
        """Changes the agent wants to make to itself, and why."""
        raw = bool(rt.autopilot.get("raw", True))
        out = {"ok": True, "raw": raw,
               "proposals": rt.ledger.proposals(status or None),
               "history": rt.ledger.override_history(),
               "knobs": rt.knobs().to_dict()}
        coin = rt.autopilot.get("coin")
        if raw:
            n = 0
            if coin:
                try:
                    n = len(rt.signals_from_history(
                        coin, rt.autopilot.get("interval") or "15m"))
                except Exception:
                    n = 0
            out["readiness"] = tuner_mod.level_readiness(n)
            out["tunes"] = "the target and stop, using the grid"
        else:
            out["readiness"] = tuner_mod.readiness(rt.ledger, coin)
            out["tunes"] = "the entry gates, using closed trades"
        return out

    @app.post("/api/proposals/decide", dependencies=[Depends(require_token)])
    def api_proposal_decide(id: str, adopt: bool) -> dict[str, Any]:
        """Adopt or reject. Only this changes a setting -- nothing auto."""
        row = rt.ledger.decide_proposal(id, adopt)
        if row is None:
            return {"ok": False,
                    "detail": "unknown id, or already decided"}
        # Every pilot picks the new value up on its next step.
        for pilot in rt._pilots.values():
            pilot.knobs = rt.knobs()
        return {"ok": True, "proposal": row,
                "knobs": rt.knobs().to_dict()}

    @app.post("/api/proposals/revert", dependencies=[Depends(require_token)])
    def api_proposal_revert(param: str) -> dict[str, Any]:
        """Put one knob back to the shipped default."""
        ok = rt.ledger.clear_override(param)
        for pilot in rt._pilots.values():
            pilot.knobs = rt.knobs()
        return {"ok": ok, "param": param, "knobs": rt.knobs().to_dict()}

    @app.post("/api/proposals/scan", dependencies=[Depends(require_token)])
    def api_proposal_scan() -> dict[str, Any]:
        """Run the walk-forward now instead of waiting for the hourly one."""
        rt._last_tune = 0.0
        coin = rt.autopilot.get("coin")
        if not coin:
            return {"ok": False,
                    "detail": ("name a market first — pick one above and "
                               "the grid has something to walk over")}
        filed = rt.maybe_tune(min_gap_s=0.0)
        n = 0
        try:
            n = len(rt.signals_from_history(
                coin, rt.autopilot.get("interval") or "15m"))
        except Exception:
            pass
        return {"ok": True, "filed": filed,
                "proposals": rt.ledger.proposals("pending"),
                "readiness": (tuner_mod.level_readiness(n)
                              if rt.autopilot.get("raw", True)
                              else tuner_mod.readiness(rt.ledger, coin))}

    @app.get("/api/chart", dependencies=[Depends(require_token)])
    def api_chart(coin: str = "BTC", interval: str = "15m", bars: int = 120
                  ) -> dict[str, Any]:
        """Candles plus the calls that fired on them, for the live chart.

        Everything comes from this service's own data: bars built from the
        tape by the live feed where it is running, and the recorded
        agreement states. No third-party chart, no embedded widget, nobody
        else's rendering of the same market.

        That matters for more than independence. A chart drawn from the same
        candles the signals were computed on cannot disagree with them, so
        when a marker looks wrong on the chart it IS wrong — which is the
        whole point of looking.
        """
        coin = rt.resolve_symbol(coin)[0] or coin
        n = max(20, min(bars, 500))
        out: dict[str, Any] = {"coin": coin, "interval": interval}

        feed = rt.feed_for(coin)
        candles: list[Any] = []
        source = "polled"

        if feed is not None:
            hist = feed.history(interval)
            if len(hist) >= 10:
                candles = list(hist[-n:])
                source = "websocket (built from fills)"

        if not candles:
            try:
                candles = rt.client().candles(coin, interval, bars=n)
            except Exception as exc:
                return {**out, "error": f"candles unavailable: {exc}"}

        live = feed.candle(interval) if feed else None
        rows = [{"ts": c.ts, "o": c.open, "h": c.high, "l": c.low,
                 "c": c.close, "v": c.volume} for c in candles]
        if live is not None:
            # Shown even when unseeded. An unseeded bar's high, low and close
            # are real fills; only its OPEN is the first trade seen rather
            # than the true open. Hiding the forming candle to avoid a
            # slightly wrong open loses the one bar the operator is actually
            # trading, so it is drawn and flagged instead.
            if rows and abs(rows[-1]["ts"] - live.start_ts) < 1.0:
                rows.pop()
            rows.append({"ts": live.start_ts, "o": live.open, "h": live.high,
                         "l": live.low, "c": live.close, "v": live.volume,
                         "live": True, "seeded": bool(live.seeded)})

        # The markers: the AGENT'S OWN TRADES, entry to exit.
        #
        # These used to come from the suggestion table -- the calls you took
        # or ignored -- which after the agent started keeping its own book
        # meant every marker on the chart read "pending" forever, because
        # nothing decides those rows any more. The chart should show what
        # the agent actually did.
        marks = []
        try:
            span = rows[0]["ts"] if rows else 0.0
            for p in rt.ledger.positions_between(coin, interval,
                                                 since=span - 3600):
                f = p.get("features") or {}
                cols = []
                for name in ("book", "delta", "price"):
                    v = f.get(f"vote_{name}")
                    if isinstance(v, (int, float)) and v:
                        cols.append(f"{name} {'up' if v > 0 else 'down'}")
                    elif f.get(f"{name}_dir") in ("up", "down"):
                        cols.append(f"{name} {f[name + '_dir']}")
                net = p.get("net_bps")
                marks.append({
                    "id": p["id"],
                    "ts": p["candle_ts"],
                    "entry_ts": p["opened_at"],
                    "exit_ts": p.get("closed_at"),
                    "side": p["side"], "entry": p["entry"],
                    "target": p["target_px"], "stop": p["stop_px"],
                    "exit_px": p.get("exit_px"),
                    "exit_reason": p.get("exit_reason"),
                    "open": p["status"] == "open",
                    "net_bps": net,
                    "gross_bps": p.get("gross_bps"),
                    "won": (net > 0) if isinstance(net, (int, float)) else None,
                    "shape": f.get("vote_shape") or p.get("grade") or "",
                    "why": ", ".join(cols),
                    "tp_bps": p.get("target_bps"),
                    "sl_bps": p.get("risk_bps"),
                    "held_s": p.get("held_s"),
                    "mae_bps": p.get("mae_bps"),
                    "mfe_bps": p.get("mfe_bps"),
                })
        except Exception as exc:
            out["marks_error"] = str(exc)

        # Where volume traded inside the forming bar, for the side profile.
        profile = None
        prof = getattr(live, "profile", None) if live is not None else None
        if prof is not None and prof.total_notional > 0:
            levels = prof.levels
            top = sorted(levels, key=lambda x: -x[1])[:40]
            profile = {"poc": prof.poc,
                       "total": round(prof.total_notional, 2),
                       "levels": [{"px": p, "v": round(v, 2)} for p, v in top]}

        return {**out, "source": source, "bars": rows, "marks": marks,
                "profile": profile,
                "interval_s": rt.client().INTERVALS.get(interval, 900)}

    @app.get("/api/agreement", dependencies=[Depends(require_token)])
    def api_agreement(coin: str = "", interval: str = "",
                      min_strength: float = 0.0) -> dict[str, Any]:
        """What actually happens in each agreement state.

        Built from real order book readings recorded as they happened. It
        starts empty and fills while the feed runs — there is no way to
        backfill it, because historical L2 depth does not exist.
        """
        c = rt.resolve_symbol(coin)[0] if coin else None
        return rt.history.agreement_table(c, interval or None, min_strength)

    @app.post("/api/backfill", dependencies=[Depends(require_token)])
    def api_backfill(coin: str = "BTC", interval: str = "15m",
                     days: float = 2.0, at: float = 0.33,
                     max_gib: float = 2.0) -> dict[str, Any]:
        """Fill the agreement table from Hyperliquid's own book archive.

        COSTS REAL MONEY. `s3://hyperliquid-archive` is requester-pays, so
        every byte is charged to your AWS account. The job reports bytes as
        it runs and stops at `max_gib`.
        """
        return rt.start_backfill(coin, interval, days=days, at=at,
                                 max_gib=max_gib)

    @app.get("/api/backfill", dependencies=[Depends(require_token)])
    def api_backfill_status() -> dict[str, Any]:
        return rt.backfill

    @app.post("/api/backfill/stop", dependencies=[Depends(require_token)])
    def api_backfill_stop() -> dict[str, Any]:
        rt.backfill["running"] = False
        return rt.backfill

    @app.get("/api/backfill/estimate", dependencies=[Depends(require_token)])
    def api_backfill_estimate(coin: str = "BTC", days: float = 2.0
                              ) -> dict[str, Any]:
        """What a job would cost, before spending anything."""
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz

        from .archive import Archive, estimate

        end = _dt.now(_tz.utc).replace(minute=0, second=0, microsecond=0)
        start = end - _td(days=max(0.04, min(days, 30.0)))
        out = estimate([coin], start, end)
        out["credentials"] = Archive().credentials_present()
        out["bucket"] = "s3://hyperliquid-archive (requester-pays)"
        return out

    @app.post("/api/resolve-states", dependencies=[Depends(require_token)])
    def api_resolve_states(limit: int = 300) -> dict[str, Any]:
        return rt.resolve_states(limit=limit)

    @app.get("/api/feature-slice", dependencies=[Depends(require_token)])
    def api_feature_slice(coin: str = "", interval: str = "",
                          feature: str = "tilt", verdict: str = "confirmed"
                          ) -> dict[str, Any]:
        """How the outcome varies with one stored reading.

        The tuning surface: ask what confirmed candles look like when tilt is
        above 0.5 versus below, and get the answer from candles already
        recorded rather than from three more weeks of collecting.
        """
        c = rt.resolve_symbol(coin)[0] if coin else None
        out = rt.history.feature_slice(c, interval or None, feature=feature,
                                       verdict=verdict)
        out["available"] = rt.history.stored_features()
        return out

    @app.post("/api/agreement-replay", dependencies=[Depends(require_token)])
    def api_agreement_replay(coin: str = "BTC", interval: str = "15m",
                             bars: int = 500, at: float = 0.33,
                             dataset: str = "") -> dict[str, Any]:
        """The price-action base rate over history.

        Deliberately measures ONE side. The order book half cannot be
        replayed — no exchange serves historical depth — and a book proxy
        built from candles would be price action agreeing with itself. This
        is the number the full rule has to beat.
        """
        from . import replay as replay_mod
        from .live import INTERVALS

        if interval not in INTERVALS:
            return {"ok": False, "error": f"{interval!r} is not a timeframe "
                                          f"this service builds"}
        interval_s = float(INTERVALS[interval])
        if interval_s <= 60:
            return {"ok": False,
                    "error": "this replay stands inside a candle using "
                             "one-minute sub-bars, so it needs a timeframe "
                             "larger than 1m"}

        coin = rt.resolve_symbol(coin)[0] or coin
        try:
            if dataset:
                stored = rt.history.dataset(dataset)
                if stored is None:
                    return {"ok": False, "error": f"no dataset {dataset!r}"}
                main = stored["candles"]
                coin, interval = stored["coin"], stored["interval"]
                interval_s = float(INTERVALS.get(interval, interval_s))
                sub = rt.client().candles(coin, "1m", bars=5000)
            else:
                n = max(100, min(bars, 2000))
                main = rt.client().candles(coin, interval, bars=n)
                need = int(n * interval_s / 60.0)
                sub = rt.client().candles(coin, "1m", bars=min(need, 5000))
        except Exception as exc:
            return {"ok": False, "error": f"candles unavailable: {exc}"}

        rows = replay_mod.replay_price_action(
            main, sub, interval_s=interval_s, sub_s=60.0,
            at=max(0.1, min(at, 0.8)))
        out = replay_mod.table(rows)
        out.update({"ok": True, "coin": coin, "interval": interval,
                    "bars": len(main), "sub_bars": len(sub),
                    "read_at": f"{max(0.1, min(at, 0.8)):.0%} into the bar"})
        return out

    @app.post("/api/upload-history", dependencies=[Depends(require_token)])
    async def api_upload_history(file: UploadFile = File(...),
                                 coin: str = "BTC",
                                 merge_into: str = "") -> dict[str, Any]:
        """Take an OHLCV export and store it for replaying.

        The report that comes back always states what candle-only history can
        and cannot train. It cannot train the book signals, which are most of
        the weight in the book call, because an OHLCV file does not contain a
        book and no exchange serves historical depth at this resolution.
        """
        from .ingest import parse as parse_history

        too_big = {"ok": False,
                   "error": f"file is over "
                            f"{MAX_UPLOAD_BYTES // (1024 * 1024)}MB — split "
                            f"it or upload a shorter span"}

        # Checked against the declared size first: reading a two-gigabyte
        # body and THEN rejecting it means the allocation already happened.
        declared = getattr(file, "size", None)
        if declared is not None and declared > MAX_UPLOAD_BYTES:
            return too_big

        # Read in bounded chunks so a request that lies about its length
        # cannot get further than one chunk past the cap.
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                return too_big
            chunks.append(chunk)
        raw = b"".join(chunks)
        del chunks

        ing = parse_history(raw, source=file.filename or "upload")
        if len(ing.candles) > MAX_UPLOAD_BARS:
            return {"ok": False,
                    "error": f"{len(ing.candles):,} bars is more than this "
                             f"service will hold in one dataset "
                             f"({MAX_UPLOAD_BARS:,}) — upload a shorter span "
                             f"or a higher timeframe"}
        out = ing.to_dict()
        if not ing.ok:
            out["ok"] = False
            return out

        try:
            did = rt.history.save_dataset(
                name=file.filename or "upload",
                coin=rt.resolve_symbol(coin)[0] or coin,
                interval=ing.interval_name, interval_s=ing.interval_s,
                candles=ing.candles, report=ing.describe(),
                merge_into=merge_into or None)
        except ValueError as exc:
            return {**out, "ok": False, "error": str(exc)}

        out["ok"] = True
        out["dataset_id"] = did
        # Read the count back from the listing, which does not carry the
        # bars. Reloading and re-parsing the whole blob just to count it
        # doubles the memory cost of every upload.
        meta = next((d for d in rt.history.datasets() if d["id"] == did), None)
        out["stored_bars"] = meta["bars"] if meta else len(ing.candles)
        return out

    @app.get("/api/datasets", dependencies=[Depends(require_token)])
    def api_datasets(coin: str = "") -> dict[str, Any]:
        c = rt.resolve_symbol(coin)[0] if coin else None
        return {"ok": True, "datasets": rt.history.datasets(c)}

    @app.post("/api/datasets/delete", dependencies=[Depends(require_token)])
    def api_delete_dataset(id: str) -> dict[str, Any]:
        return {"ok": rt.history.delete_dataset(id), "id": id}

    @app.get("/api/read", dependencies=[Depends(require_token)])
    def api_read(coin: str = "BTC", interval: str = "15m",
                 higher: str = "4h", size: float = 0.0,
                 timeframes: str = "") -> dict[str, Any]:
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
        if feed is not None and interval not in feed.builders:
            # The timeframe selector can name any interval. A feed opened on
            # a different set has nothing for it, which looked exactly like
            # the socket having stopped.
            try:
                feed.ensure_interval(interval, bars)
            except Exception as exc:
                out["feed_interval_error"] = str(exc)
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
            live_spot = rt.client().mid(coin)
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
            # signal -- it is the same public fills either way. And the feed
            # learns its own impact baseline, so absorption comes with it:
            # absorption is a property of the tape, not of a level you
            # happened to point at.
            tape = feed.tape
            # Scope the absorption window to THIS candle. A fixed five
            # minutes answers a different question on a 1-minute chart than
            # on a 30-minute one, and the whole point is to read the bar in
            # front of you. Floored at a quarter of the bar so a just-opened
            # candle is not judged on three seconds of tape.
            absorption = feed.absorption(
                window_s=max(min(elapsed, float(step)), float(step) * 0.25))
            # Impact buckets scale with the timeframe too: 15-second buckets
            # are right for a 15-minute bar and far too coarse for a 1-minute.
            feed.bucket_s = max(2.0, min(float(step) / 60.0, 60.0))

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

        try:
            from .pressure import price_action
            pa_read = price_action(closed + [current])
        except Exception:
            pa_read = None

        r = candleread.read(
            coin=coin, interval_s=float(step), elapsed_s=elapsed,
            open_px=current.open, high_px=current.high, low_px=current.low,
            last_px=current.close, tape=tape, book=book,
            absorption=absorption, higher=higher_struct, zone=in_zone, vw=vw,
            magnet_bps=magnet_bps, magnet_notional=magnet_notional,
            pa=pa_read, calibration=rt.calibration(coin, interval))

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
            "price_action": (None if pa_read is None else {
                "rejection": pa_read.rejection, "sweep": pa_read.sweep,
                "acceptance": pa_read.acceptance, "signed": pa_read.signed,
                "notes": pa_read.notes}),
            "vwap": (None if vw is None else
                     {"value": vw.value, "band": vw.band_of(current.close)}),
            "zone": (None if in_zone is None else
                     {"kind": in_zone.kind, "low": in_zone.low,
                      "high": in_zone.high, "tested": in_zone.tested,
                      "fresh": in_zone.fresh}),
            "calibration": rt.calibration(coin, interval).table(),
            "inputs": {
                "flow": tape is not None,
                "absorption": bool(absorption is not None
                                   and absorption.confident),
                "absorption_warming": bool(absorption is not None
                                           and not absorption.confident),
                "book": bool(book is not None and not book.empty),
                "level_watch": bool(w is not None and w.coin == coin),
                "baseline_samples": (feed.baseline.samples
                                     if use_feed and feed else 0),
            },
            "has_watch": absorption is not None,
        })
        # ---- who is winning, on every timeframe at once -----------------
        #
        # Not averaged into the score. A 4-hour that belongs to sellers and a
        # 15-minute that belongs to buyers is a specific, tradeable shape;
        # blending them to zero destroys the only thing worth knowing.
        try:
            from . import pressure as pr

            tf_list = [t.strip() for t in (timeframes or "").split(",") if t.strip()]
            if not tf_list:
                tf_list = [higher, "1h", interval, "5m"]
            seen, want = set(), []
            for t in tf_list:
                if t in client.INTERVALS and t not in seen:
                    seen.add(t)
                    want.append(t)

            readings = []
            for tf in want:
                iv = float(client.INTERVALS[tf])
                bar = feed.candle(tf) if (use_feed and feed) else None
                if bar is not None and bar.seeded:
                    cov = (min(1.0, (now_s - feed.started_at) / iv)
                           if feed.started_at else 0.0)
                    readings.append(pr.from_live(tf, iv, bar, book,
                                                 coverage=max(cov, 0.05)))
                    continue
                try:
                    tf_bars = (bars if tf == interval
                               else client.candles(coin, tf, bars=6))
                except Exception:
                    continue
                if not tf_bars:
                    continue
                # Aggregate the recent bars rather than reading only the one
                # in progress: two minutes into a 4-hour candle there is
                # nothing in it, and the higher timeframe would go silent
                # exactly when it matters most.
                r_ = pr.from_candles(
                    tf, iv, tf_bars, lookback=3, book=book,
                    elapsed_s=max(0.0, min(iv, now_s - tf_bars[-1].ts)))
                if r_ is not None:
                    readings.append(r_)

            conf = pr.confront(readings)
            out["timeframes"] = [{
                "timeframe": r.timeframe, "winner": r.winner,
                "aggressor": r.aggressor, "strength": r.strength,
                "signed": r.signed, "absorbing": r.absorbing,
                "move_bps": r.move_bps, "lean": r.lean,
                "buy_notional": r.buy_notional,
                "sell_notional": r.sell_notional,
                "measured": r.measured, "trades": r.trades,
                "position_in_range": r.position_in_range,
                "describe": r.describe(),
            } for r in conf.ordered]
            out["confrontation"] = {
                "aligned": conf.aligned, "conflicted": conf.conflicted,
                "consensus": conf.consensus, "verdict": conf.verdict(),
                "higher": conf.higher.timeframe if conf.higher else None,
                "lower": conf.lower.timeframe if conf.lower else None,
            }
        except Exception as exc:
            out["pressure_error"] = f"{type(exc).__name__}: {exc}"

        if book is not None and not book.empty:
            out["book"] = {
                "bid": book.best_bid, "ask": book.best_ask, "mid": book.mid,
                "spread_bps": book.spread_bps,
                "imbalance_25bps": book.imbalance(25.0),
                "pushed": bool(use_feed and feed.book is not None),
                # Scoped to the candle: what the book has been doing across
                # this bar, not only what it looks like this instant.
                "trend": (feed.spread_trend(
                    window_s=max(min(elapsed, float(step)), 30.0))
                    if use_feed and feed else None),
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
                     tune: bool = True, dataset: str = "") -> dict[str, Any]:
        """Walk-forward backtest with a held-out half.

        The number to act on is the held-out one. Weights are searched on the
        earlier portion only and the later portion is never seen by the
        search, so the gap between the two is a direct measure of how much of
        any apparent edge is fitted noise.

        With `dataset`, the replay runs over uploaded chart history instead
        of the exchange's own candles. That buys length -- years rather than
        the few thousand bars the API serves -- and buys nothing at all for
        the book signals, which are not in an OHLCV file. `blind_to` in the
        response says so on every run, uploaded or not.
        """
        from . import backtest as bt_mod

        source = "real exchange candles via candleSnapshot"
        if dataset:
            stored = rt.history.dataset(dataset)
            if stored is None:
                return {"ok": False, "error": f"no dataset {dataset!r}"}
            candles = stored["candles"]
            coin, interval = stored["coin"], stored["interval"]
            source = (f"uploaded history: {stored['name']} "
                      f"({len(candles):,} bars)")
        else:
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
            "source": source,
            "dataset": dataset or None,
            "train": rep(bt.train), "test": rep(bt.test),
            "baseline_test": rep(bt.baseline_test) if bt.baseline_test else None,
            "tuned_weights": bt.tuned_weights,
            "overfit_gap_pts": bt.overfit_gap,
            "verdict": bt.verdict(),
            "blind_to": ["flow", "absorption", "imbalance", "magnet"],
            "caveat": (
                "A replay over bars can only train what a bar contains. "
                "Microprice tilt, replenishment, depletion and absorption "
                "are not in an OHLCV series at any length, so uploading more "
                "history improves the candle-geometry half and leaves the "
                "book half exactly where it was. That half is measured "
                "forward, live, in the suggestions table."),
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
  .chart-key{display:flex;flex-wrap:wrap;gap:4px 14px;margin-top:6px;
       font-size:11px;color:var(--dim);align-items:center}
  .chart-key span{display:inline-flex;align-items:center;gap:5px}
  .chart-key i{display:inline-block;flex:none}
  .k-arrow{width:0;height:0;border-left:5px solid transparent;
       border-right:5px solid transparent}
  .k-arrow.k-long{border-top:8px solid var(--up)}
  .k-arrow.k-short{border-bottom:8px solid var(--down);border-top:none}
  /* What the canvas actually draws for a trade: long is an UP triangle in
     the up colour, short is a DOWN triangle in the down colour. */
  .k-arrow.k-entry-long{border-bottom:8px solid var(--up);border-top:none}
  .k-arrow.k-entry-short{border-top:8px solid var(--down);border-bottom:none}
  .k-arrow.k-hollow{opacity:.4}
  .k-dot{width:7px;height:7px;border-radius:50%}
  .k-dot.k-long{background:var(--up)}
  .k-dot.k-short{background:var(--down)}
  .k-dash{width:16px;height:0;border-top:1px dashed var(--dim)}
  .k-box{width:9px;height:9px;border:1.5px solid var(--ink);opacity:.6}
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
  /* Hit rate as a RANGE, never a point: a rate with no sample size behind
     it is a mood, not a measurement. */
  .band{position:relative;height:16px;background:var(--grid);border-radius:3px;
        overflow:hidden;min-width:140px}
  .band .fill{position:absolute;top:0;bottom:0;border-radius:3px;opacity:.55}
  .band .fill.clear{background:var(--up)}
  .band .fill.short{background:var(--down)}
  .band .fill.undecided{background:var(--dim)}
  .band .need{position:absolute;top:-2px;bottom:-2px;width:2px;
              background:var(--ink)}
  .band .point{position:absolute;top:3px;width:2px;height:10px;
               background:var(--ink);opacity:.9}
  .band-key{display:flex;flex-wrap:wrap;gap:4px 14px;margin:6px 0 2px;
            font-size:11px;color:var(--dim)}
  .band-key span{display:inline-flex;align-items:center;gap:5px}
  .band-key i{display:inline-block;flex:none}
  /* The grid. Diverging, because the metric has a real zero: blue pays,
     red loses, neutral is breaking even. Blue-red rather than the app's
     green-red because dozens of adjacent cells have to be told apart --
     measured, green/red collapses to deltaE 5 under deuteranopia while
     blue/red holds at 19. Every cell prints its number as well, so the
     colour never carries the meaning by itself. */
  .heat{border-collapse:separate;border-spacing:2px;font-size:11px}
  .heat th{font-weight:600;color:var(--dim);font-size:11px;padding:2px 6px;
           white-space:nowrap}
  .heat td{padding:5px 7px;text-align:right;font-family:var(--mono);
           color:var(--ink);border-radius:3px;min-width:56px;cursor:default}
  .heat td.sig{outline:1.5px solid var(--ink);outline-offset:-1.5px}
  .heat td.best{outline:2px solid var(--accent);outline-offset:-2px}
  .heat-key{display:flex;flex-wrap:wrap;gap:6px 16px;align-items:center;
            margin:8px 0;font-size:11px;color:var(--dim)}
  .heat-key i{display:inline-block;height:10px;border-radius:2px}
  .prop{border:1px solid var(--grid);border-radius:4px;padding:10px 12px;
        margin-bottom:8px}
  .prop pre{white-space:pre-wrap;font-family:var(--mono);font-size:11px;
            color:var(--dim);margin:6px 0}
  .thin{font-size:11px;color:var(--dim)}
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
  .tf{display:grid;grid-template-columns:62px 92px 1fr 88px;gap:10px;
    align-items:center;padding:7px 10px;border-radius:4px;margin-bottom:4px;
    background:var(--bg);border-left:3px solid var(--line);font-size:13px}
  .tf.buyers{border-left-color:var(--long)}
  .tf.sellers{border-left-color:var(--short)}
  .tf b{font-variant-numeric:tabular-nums}
  .tf .who{font-weight:600;letter-spacing:.03em}
  .tf .bar{height:6px;background:var(--line);border-radius:3px;position:relative}
  .tf .bar i{position:absolute;top:0;bottom:0;border-radius:3px}
  .tf .bar i.buyers{left:50%;background:var(--long)}
  .tf .bar i.sellers{right:50%;background:var(--short)}
  .tf .meta{font-size:11px;color:var(--dim);text-align:right}
  .call{display:flex;align-items:baseline;gap:16px;flex-wrap:wrap;
    padding:10px 0 4px}
  .call b{font-size:44px;letter-spacing:.02em;line-height:1}
  .call .sub{font-size:13px;color:var(--dim);font-variant-numeric:tabular-nums}
  .comp{display:grid;grid-template-columns:132px 64px 1fr;gap:10px;
    align-items:center;font-size:12px;padding:4px 0;
    border-bottom:1px solid var(--line)}
  .comp .v{font-variant-numeric:tabular-nums;text-align:right}
  .comp .m{height:5px;background:var(--line);border-radius:3px;position:relative}
  .comp .m i{position:absolute;top:0;bottom:0;border-radius:3px}
  .comp .m i.pos{left:50%;background:var(--long)}
  .comp .m i.neg{right:50%;background:var(--short)}
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
    <div class="panel full" id="agentPanel"><h2>The call — take it or leave it
      <span class="stamp" id="sugStamp"></span></h2>
      <div class="msg" style="margin-bottom:8px">Two independent readings.
        The <b>book</b> says where the pressure is; <b>price</b> says whether
        the pressure is winning. A trade only when they agree — if the book
        favours buyers and price is falling, somebody is absorbing them and
        we stand aside. <b>Nothing here places an order.</b></div>
      <div class="conbar">
        <label>candle <select id="sInt" onchange="loadSuggest()">
          <option>1m</option><option>5m</option><option selected>15m</option>
          <option>30m</option><option>1h</option></select></label>
        <label>aim <select id="sMode" onchange="loadSuggest()">
          <option value="scalp" selected>scalp — nearest that pays</option>
          <option value="range">range — half the bar</option></select></label>
        <label>size $<input id="sSize" value="10000" size="8"
          onchange="loadSuggest()"></label>
        <label>round-trip fee <input id="sFee" value="0" size="4"
          onchange="loadSuggest()">bps</label>
        <label><input type="checkbox" id="sAuto" onchange="toggleSuggestAuto()">
          poll 5s</label>
        <button class="go" onclick="startFeed()" id="sugLive">Go live</button>
        <button onclick="stopFeed()">Stop feed</button>
        <button onclick="loadSuggest()">Ask</button>
      </div>
      <div id="sugFeed" class="msg" style="display:none;margin-bottom:8px"></div>
      <div id="sugThree" style="display:none;margin-bottom:10px"></div>
      <div id="chartWrap" style="display:none;margin-bottom:12px">
        <canvas id="sugChart" height="300"
          style="width:100%;height:300px;display:block;
                 border:1px solid var(--line);border-radius:4px"></canvas>
        <!-- A legend, not a caption. Identity is never carried by colour
             alone: filled vs hollow says taken vs ignored, and the arrow
             direction says which side. -->
        <div class="chart-key">
          <span><i class="k-arrow k-entry-long"></i>went long</span>
          <span><i class="k-arrow k-entry-short"></i>went short</span>
          <span><i class="k-dot k-long"></i>closed a winner</span>
          <span><i class="k-dot k-short"></i>closed a loser</span>
          <span>the line joins entry to exit — its length is how long it
            held</span>
          <span><i class="k-dash"></i>target and stop</span>
          <span>hollow triangle = still open</span>
          <span><i class="k-box"></i>bar still forming</span>
        </div>
        <div id="chartNote" class="msg" style="margin-top:4px"></div>
      </div>
      <div id="sugCard" class="msg">Press <b>Go live</b>, give it ten seconds
        of book pushes, then <b>Ask</b>.</div>
      <div id="sugActions" style="display:none;margin-top:10px">
        <button class="go" onclick="decide(true)">Take it</button>
        <button onclick="decide(false)">Ignore</button>
        <span id="sugDecided" class="msg" style="margin-left:10px"></span>
      </div>
      <div id="sugScore" class="say" style="display:none;margin-top:12px"></div>
    </div>

    <div class="panel full" id="pilotPanel"><h2>The agent's own book
      <span class="stamp" id="apStamp"></span></h2>
      <div class="msg" style="margin-bottom:8px">Its decisions, not yours.
        Book, delta and price vote; whichever way they add up is the
        direction. Nothing is refused except all three going flat, which is
        no reading rather than a veto. Levels are the ones you set below —
        the same ones the grid tests — so a result up there transfers down
        here unchanged. <b>Nothing places an order.</b></div>

      <div class="conbar">
        <label>candle <select id="apInt">
          <option>1m</option><option>5m</option><option selected>15m</option>
          <option>30m</option><option>1h</option></select></label>
        <label>size $<input id="apSize" value="10000" size="8"></label>
        <label><b>take profit</b> <input id="apTp" value="20" size="3">bps</label>
        <label><b>stop loss</b> <input id="apSl" value="15" size="3">bps</label>
        <label>round-trip fee <input id="apFee" value="0" size="4">bps</label>
        <label title="Off keeps the live book identical to the grid, so a
result from the sweep transfers unchanged."><input type="checkbox"
          id="apInval">also exit when the read flips</label>
        <button class="go" id="apToggle" onclick="toggleAutopilot()">Let it
          decide</button>
        <button onclick="stepAutopilot()">Decide once</button>
        <button onclick="loadScorecard()">Refresh score</button>
      </div>

      <div id="apState" class="msg">Off. Turn it on and leave it — a book
        that only fills while you are watching measures your attention, not
        the strategy.</div>
      <div id="apPos" style="display:none;margin-top:10px"></div>
      <div id="apFeed" style="display:none;margin-top:10px"></div>

      <h3 style="margin:16px 0 6px">How it is doing</h3>
      <div id="apHead" class="say">Nothing settled yet.</div>
      <div id="apStats" style="margin-top:10px"></div>
      <div id="apSlices" style="margin-top:10px"></div>

      <h3 style="margin:16px 0 6px">Changes it wants to make</h3>
      <div class="msg" style="margin-bottom:6px">It picks a value on older
        trades and scores it on newer ones it has never seen. Nothing changes
        until you say so.
        <button style="margin-left:8px" onclick="scanProposals()">Look
          now</button></div>
      <div id="apProposals"></div>
    </div>

    <div class="panel full" id="sweepPanel"><h2>Target and stop — test every pair
      <span class="stamp" id="swStamp"></span></h2>
      <div class="msg" style="margin-bottom:8px">A signal does not know where
        its target is — book, delta and price point a direction at a price,
        and that is the whole decision. So one pass over the candles you have
        already recorded scores <b>every</b> target against <b>every</b> stop.
        Set the ranges and look at the shape of the surface, not at the best
        square.</div>

      <div class="conbar">
        <label>candle <select id="swInt">
          <option>1m</option><option>5m</option><option selected>15m</option>
          <option>30m</option><option>1h</option></select></label>
        <label><b>take profit</b> <input id="swTpFrom" value="5" size="3">to
          <input id="swTpTo" value="40" size="3">step
          <input id="swTpStep" value="5" size="3">bps</label>
        <label><b>stop loss</b> <input id="swSlFrom" value="5" size="3">to
          <input id="swSlTo" value="40" size="3">step
          <input id="swSlStep" value="5" size="3">bps</label>
        <label>round-trip fee <input id="swFee" value="0" size="3">bps</label>
        <label>give it <input id="swHorizon" value="60" size="3">
          minutes</label>
        <label>votes <select id="swAgree">
          <option value="0" selected>any — every signal</option>
          <option value="2">2 of 3 or better</option>
          <option value="3">all three only</option></select></label>
        <label>side <select id="swSide">
          <option value="both" selected>both</option>
          <option value="long">long only</option>
          <option value="short">short only</option></select></label>
        <button class="go" onclick="runSweep()">Run the grid</button>
        <button onclick="loadSignals()">What have I got?</button>
      </div>

      <div id="swAvail" class="msg"></div>
      <div id="swVerdict" class="say" style="display:none;margin:10px 0"></div>
      <div id="swGrid" style="overflow-x:auto"></div>
      <div id="swShapes" style="margin-top:14px"></div>
    </div>

    <div class="panel full" id="bookPanel"><h2>Book call — this candle, right now
      <span class="stamp" id="bookStamp"></span></h2>
      <div class="msg" style="margin-bottom:8px">The order book is the structure.
        No trend, no higher timeframe, no VWAP. Updates on every book push.</div>
      <div id="bookHead" class="msg">Press <b>Go live</b> to open the book feed.</div>
      <div id="bookComps"></div>
      <div id="bookSay" class="say" style="display:none"></div>
    </div>

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
        <label>book window <select id="bkWin" onchange="loadBookCall()">
          <option value="5">5s</option><option value="10">10s</option>
          <option value="20" selected>20s</option><option value="60">60s</option>
        </select></label>
        <button onclick="startFeed()" id="feedBtn">Go live (websocket)</button>
        <button onclick="stopFeed()">Stop feed</button>
        <button onclick="loadRead()">Read now</button>
        <button onclick="doCalibrate()">Calibrate</button>
      </div>
      <div id="feedBar" class="msg" style="margin-bottom:8px">Feed off — the
        current candle is being polled. Go live to build it from the tape instead.</div>
      <div id="alertBar" class="alertbar" style="display:none"></div>
      <div id="tfLadder"></div>
      <div id="tfSay" class="say" style="display:none;margin-bottom:12px"></div>
      <div id="readHead" class="msg">—</div>
      <div id="readSignals"></div>
      <div id="readSay" class="say" style="display:none"></div>
    </div>

    <div class="panel full" id="agreePanel"><h2>Book vs price — what actually
      happens<span class="stamp" id="agreeStamp"></span></h2>
      <div class="msg" style="margin-bottom:8px">Every candle is recorded,
        tradeable or not, with the raw book and price readings behind it.
        That is the only way to find out what the confirmation rule is worth:
        measuring just the candles that passed the filter and concluding the
        filter works answers a different question. <b>This cannot be
        backfilled</b> — no exchange serves historical order book depth, so it
        fills as the feed runs.</div>
      <div class="conbar">
        <label>market <select id="agInt" onchange="loadAgreement()">
          <option value="">all timeframes</option>
          <option>1m</option><option>5m</option><option selected>15m</option>
          <option>30m</option><option>1h</option></select></label>
        <label><input type="checkbox" id="agThis" checked
          onchange="loadAgreement()"> this coin only</label>
        <button onclick="loadAgreement()">Refresh</button>
        <button onclick="runReplay()">Historical base rate</button>
      </div>
      <div id="agSay" class="say" style="display:none;margin-bottom:10px"></div>
      <div id="agTable"></div>

      <h3 style="margin:18px 0 4px;font-size:12px;text-transform:uppercase;
        letter-spacing:.06em;color:var(--dim)">Backfill from the archive</h3>
      <div class="msg" style="margin-bottom:8px">Hyperliquid publishes real
        book snapshots twice a second to
        <code>s3://hyperliquid-archive</code>, going back years — higher
        resolution than the live public feed currently gives. This replays
        them through the same reader the live feed uses, so the table above
        fills in this afternoon instead of over three weeks.
        <b class="short">This is a requester-pays bucket: every byte is
        charged to your own AWS account.</b> Set
        <code>AWS_ACCESS_KEY_ID</code> and <code>AWS_SECRET_ACCESS_KEY</code>
        in Railway first.</div>
      <div class="conbar">
        <label>days back <input id="bfDays" type="number" value="2" min="1"
          max="30" step="1" style="width:70px" onchange="estimateBackfill()"></label>
        <label>read at <select id="bfAt">
          <option value="0.25">25% in</option>
          <option value="0.33" selected>33% in</option>
          <option value="0.5">50% in</option></select></label>
        <label>stop at <input id="bfCap" type="number" value="2" min="0.1"
          max="50" step="0.5" style="width:70px">GiB</label>
        <button onclick="estimateBackfill()">Estimate cost</button>
        <button class="go" onclick="startBackfill()">Run backfill</button>
        <button onclick="stopBackfill()">Stop</button>
      </div>
      <div id="bfSay" class="msg">—</div>

      <h3 style="margin:18px 0 4px;font-size:12px;text-transform:uppercase;
        letter-spacing:.06em;color:var(--dim)">Which reading matters</h3>
      <div class="msg" style="margin-bottom:8px">The tuning surface. Pick a
        stored reading and see how the outcome varies across its range — a
        signal that matters shows a rising line, one that does not shows
        noise. Answered from candles already recorded.</div>
      <div class="conbar">
        <label>reading <select id="agFeat" onchange="loadSlice()"></select></label>
        <label>state <select id="agVerdict" onchange="loadSlice()">
          <option value="confirmed" selected>confirmed</option>
          <option value="conflict">conflict</option>
          <option value="unconfirmed">unconfirmed</option></select></label>
      </div>
      <div id="agSlice"></div>
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

      <h3 style="margin:14px 0 4px;font-size:12px;text-transform:uppercase;
        letter-spacing:.06em;color:var(--dim)">Upload chart history</h3>
      <div class="msg" style="margin-bottom:8px">A CSV or JSON OHLCV export —
        TradingView, an exchange, anything with time, open, high, low and
        close. Column names and order are detected. This extends the
        historical replay past what the exchange API serves, and it trains
        the candle-geometry signals only: <b>microprice tilt, replenishment,
        depletion and absorption are not in an OHLCV file at any length</b>,
        so that half stays where it is and is measured forward instead.</div>
      <div class="conbar">
        <input type="file" id="histFile" accept=".csv,.txt,.json,.tsv">
        <label>merge into <select id="histMerge">
          <option value="">— new dataset —</option></select></label>
        <button onclick="uploadHistory()">Upload</button>
      </div>
      <div id="histSay" class="msg" style="margin-bottom:6px">—</div>
      <div id="histList"></div>

      <h3 style="margin:18px 0 4px;font-size:12px;text-transform:uppercase;
        letter-spacing:.06em;color:var(--dim)">Run a replay</h3>
      <div class="conbar">
        <label>source <select id="btSource">
          <option value="">exchange candles</option></select></label>
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
/* Escape before putting any server string into innerHTML. Coin names,
   uploaded filenames and error text all reach the page this way, and a
   symbol or filename containing a bracket would otherwise break the markup
   around it -- or worse. */
function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
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
  // A FormData body must set its own Content-Type, because only the browser
  // knows the multipart boundary it generated. Forcing application/json onto
  // it makes the server unable to find any of the parts, and the failure
  // looks like an empty upload rather than a header problem.
  const base = {'Authorization': 'Bearer ' + tok()};
  if (!(o.body instanceof FormData)) base['Content-Type'] = 'application/json';
  o.headers = Object.assign(base, o.headers || {});
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
    await Promise.all([loadStatus(), loadCoins(), loadBookCall(), loadWeights(), loadRead(),
                       loadForward(), loadConsensus(), loadDecisions(),
                       loadDatasets(), loadAgreement(), loadAutopilot(),
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

let bookPoll = null;

async function loadBookCall() {
  let d;
  try {
    d = await api('/api/bookcall?' + q({coin: coin(),
                  window_s: parseFloat($('bkWin').value || '20')}));
  } catch (e) { $('bookHead').textContent = e.message; return; }
  paintBookCall(d);
}

function paintBookCall(d) {
  if (!d) return;
  if (d.error) {
    $('bookHead').textContent = d.error;
    $('bookComps').innerHTML = '';
    $('bookSay').style.display = 'none';
    return;
  }

  const cls = d.direction === 'up' ? 'long' : d.direction === 'down' ? 'short' : '';
  $('bookHead').innerHTML =
      `<div class="call"><b class="${cls}">${d.direction.toUpperCase()}</b>`
    + `<span class="sub">score ${d.score >= 0 ? '+' : ''}${d.score.toFixed(3)}`
    + ` · conviction ${(d.conviction*100).toFixed(0)}%`
    + ` · ${d.samples} book updates / ${d.window_s.toFixed(0)}s</span></div>`
    + '<div class="conrow">'
    + `<div class="stat"><b>${Number(d.mid).toLocaleString(undefined,{maximumFractionDigits:6})}</b><span>mid</span></div>`
    + `<div class="stat"><b class="${d.microprice > d.mid ? 'long' : d.microprice < d.mid ? 'short' : ''}">`
    + `${Number(d.microprice).toLocaleString(undefined,{maximumFractionDigits:6})}</b><span>microprice</span></div>`
    + `<div class="stat"><b>${d.spread_bps.toFixed(3)}bps</b><span>spread</span></div>`
    + `<div class="stat"><b>${d.book_updates}</b><span>book pushes</span></div>`
    + '</div>';

  $('bookComps').innerHTML = (d.components || []).map(c => {
    const w = Math.round(Math.abs(c.value) * 50);
    return `<div class="comp">
      <span>${c.name}</span>
      <span class="v ${c.value > 0 ? 'long' : c.value < 0 ? 'short' : ''}">
        ${c.value >= 0 ? '+' : ''}${c.value.toFixed(2)}</span>
      <span class="m"><i class="${c.value >= 0 ? 'pos' : 'neg'}" style="width:${w}%"></i>
        <span style="position:absolute;left:0;top:8px;color:var(--dim);font-size:11px">${c.note}</span></span>
    </div>`;
  }).join('') + '<div style="height:14px"></div>';

  $('bookSay').style.display = '';
  $('bookSay').textContent = d.verdict;
  stamp('bookStamp', d.age_s == null ? 0 : d.age_s, 5, 30, 'book ');
}

/* ---- the call: take it or leave it ---------------------------------- */

let sugPoll = null;
let sugId = null;

/* ---------------------------------------------------------------------
   The agent's own book.

   Everything here is a record of decisions, not an instruction to anyone.
   The panel never offers to place an order because nothing behind it can.
   --------------------------------------------------------------------- */

let apOn = false;

function apCfg() {
  return {coin: coin(), interval: $('apInt').value,
          notional: parseFloat($('apSize').value || '10000'),
          fee_bps: parseFloat($('apFee').value || '0'),
          tp_bps: parseFloat($('apTp').value || '20'),
          sl_bps: parseFloat($('apSl').value || '15'),
          raw: true,
          exit_on_invalidation: $('apInval').checked};
}

async function toggleAutopilot() {
  const c = apCfg();
  let d;
  try {
    d = await api('/api/autopilot?' + q(Object.assign({on: !apOn}, c)),
                  {method: 'POST'});
  } catch (e) { $('apState').textContent = e.message; return; }
  if (!d.ok) { $('apState').textContent = d.detail || 'could not start'; return; }
  apOn = d.autopilot.on;
  $('apToggle').textContent = apOn ? 'Stop deciding' : 'Let it decide';
  loadAutopilot();
  if (apOn && !apPoll) apPoll = setInterval(loadAutopilot, 5000);
  if (!apOn && apPoll) { clearInterval(apPoll); apPoll = null; }
}

let apPoll = null;

async function stepAutopilot() {
  const c = apCfg();
  try {
    const d = await api('/api/autopilot/step?' + q(c), {method: 'POST'});
    if (!d.ok) { $('apState').textContent = d.detail || 'no reading'; return; }
  } catch (e) { $('apState').textContent = e.message; return; }
  loadAutopilot();
}

async function loadAutopilot() {
  let d;
  try { d = await api('/api/autopilot'); }
  catch (e) { $('apState').textContent = e.message; return; }
  apOn = !!(d.autopilot && d.autopilot.on);
  $('apToggle').textContent = apOn ? 'Stop deciding' : 'Let it decide';
  $('apStamp').textContent = d.autopilot.last_step
    ? 'decided ' + Math.round(Date.now()/1000 - d.autopilot.last_step) + 's ago'
    : '';

  const c = d.counts || {};
  const last = d.autopilot.last_decision;
  $('apState').innerHTML = apOn
    ? `<b>Running on ${esc(d.autopilot.coin || '—')} `
      + `${esc(d.autopilot.interval || '')}</b> at `
      + `${d.autopilot.tp_bps}bps target / ${d.autopilot.sl_bps}bps stop — `
      + `${c.closed || 0} closed, `
      + `${c.open || 0} open, ${c.decisions || 0} decisions logged. `
      + (last ? 'Last: ' + esc(last.sentence) : '')
      + (d.autopilot.error ? ` <span class="err">${esc(d.autopilot.error)}</span>` : '')
    : `Off. ${c.closed || 0} trades in the book so far. Turn it on and leave `
      + `it — a book that only fills while you are watching measures your `
      + `attention, not the strategy.`;

  paintPosition((d.open || [])[0]);
  paintDecisionFeed(d.recent || []);
  loadScorecard();
}

function paintPosition(p) {
  const el = $('apPos');
  if (!p) { el.style.display = 'none'; return; }
  el.style.display = '';
  const dirCls = p.side === 'long' ? 'long' : 'short';
  el.innerHTML =
    `<div class="verdict-big ${dirCls}">${p.side.toUpperCase()} `
    + `${esc(p.coin)} <span class="thin">open</span></div>`
    + '<div class="conrow">'
    + `<div class="stat"><b>${chPx(p.entry)}</b><span>entry</span></div>`
    + `<div class="stat"><b>${chPx(p.target_px)}</b><span>target</span></div>`
    + `<div class="stat"><b>${chPx(p.stop_px)}</b><span>invalid at</span></div>`
    + `<div class="stat"><b>${p.mfe_bps.toFixed(1)}bps</b>`
      + '<span>best it saw</span></div>'
    + `<div class="stat"><b>${p.mae_bps.toFixed(1)}bps</b>`
      + '<span>worst it saw</span></div>'
    + (p.against_s > 0
        ? `<div class="stat gap-bad"><b>${p.against_s.toFixed(0)}s</b>`
          + '<span>read against it</span></div>' : '')
    + '</div>'
    + `<div class="msg"><b>Needs ${(p.breakeven*100).toFixed(0)}%</b> to break `
    + `even. ${p.agreeing} of three agreed at entry; runway was `
    + `${p.runway_bps.toFixed(0)}bps. `
    + `<button style="margin-left:6px" onclick="closePaper('${esc(p.id)}')">`
    + 'Close it by hand</button></div>';
}

async function closePaper(id) {
  try { await api('/api/autopilot/close?' + q({id}), {method: 'POST'}); }
  catch (e) { $('apState').textContent = e.message; return; }
  loadAutopilot();
}

const AP_WORD = {enter_long: 'LONG', enter_short: 'SHORT', exit: 'EXIT',
                 hold: 'hold', stand_aside: 'aside'};

function paintDecisionFeed(rows) {
  const el = $('apFeed');
  if (!rows.length) { el.style.display = 'none'; return; }
  el.style.display = '';
  // Holds are the overwhelming majority and say nothing on their own;
  // showing them would bury the decisions that changed something.
  // Runs of the same stand-aside are collapsed. Eighty identical rows is
  // not a feed, and it buries the decisions that changed something.
  const runs = [];
  for (const r of rows.filter(r => r.action !== 'hold')) {
    const last = runs[runs.length - 1];
    const same = last && last.action === r.action && last.gate === r.gate
                 && r.action === 'stand_aside';
    if (same) { last.count++; last.ts = Math.min(last.ts, r.ts); }
    else runs.push({action: r.action, gate: r.gate, reason: r.reason,
                    ts: r.ts, count: 1});
  }
  el.innerHTML = '<table><thead><tr><th>when</th><th>did</th>'
    + '<th>why</th></tr></thead><tbody>'
    + runs.slice(0, 12).map(r => {
        const cls = r.action === 'enter_long' ? 'long'
                  : r.action === 'enter_short' ? 'short' : '';
        const times = r.count > 1 ? ` <span class="thin">×${r.count}</span>` : '';
        return `<tr><td class="thin">`
          + new Date(r.ts * 1000).toLocaleTimeString()
          + `</td><td><b class="${cls}">${AP_WORD[r.action] || r.action}</b>`
          + `${times}</td><td class="thin">`
          + esc(r.gate ? r.gate + ' — ' + (r.reason || '') : (r.reason || ''))
          + '</td></tr>';
      }).join('')
    + '</tbody></table>'
    + '<div class="thin" style="margin-top:4px">Holds are left out and '
    + 'repeats collapsed — only decisions that changed something.</div>';
}

/* ------------------------------------------------------------ scorecard */

function band(b) {
  /* Hit rate drawn as the RANGE it honestly is, against the rate the
     entries needed. This is a diagnostic of the ENTRY SHAPE only — it is
     not the verdict, because once an exit rule can close a trade between
     the levels, clearing this line stops implying making money. The
     verdict column is expectancy. */
  const verdict = b.beats_breakeven === true ? 'clear'
                : b.beats_breakeven === false ? 'short' : 'undecided';
  const lo = b.ci_low * 100, hi = b.ci_high * 100;
  return `<div class="band" title="${esc(b.verdict)}">`
    + `<div class="fill ${verdict}" style="left:${lo}%;width:${hi-lo}%"></div>`
    + `<div class="point" style="left:${b.hit_rate*100}%"></div>`
    + `<div class="need" style="left:${b.needed*100}%"></div></div>`;
}

const BAND_KEY =
  '<div class="thin" style="margin-top:8px">The bar below is about entry '
  + '<i>shape</i> — did it hit often enough for the way the trades were '
  + 'sized. The money column is the verdict; they can disagree, and when '
  + 'they do the row says why.</div>'
  + '<div class="band-key">'
  + '<span><i style="width:18px;height:9px;background:var(--dim);opacity:.55;'
  + 'border-radius:2px"></i>where the true hit rate honestly sits</span>'
  + '<span><i style="width:2px;height:11px;background:var(--ink);'
  + 'opacity:.9"></i>what it actually hit</span>'
  + '<span><i style="width:2px;height:13px;background:var(--ink)"></i>'
  + 'what it had to hit to break even</span>'
  + '<span>green = the whole range clears it · red = none of it does · '
  + 'grey = too close to call</span></div>';

async function loadScorecard() {
  let d;
  try { d = await api('/api/scorecard?' + q({coin: coin()})); }
  catch (e) { $('apHead').textContent = e.message; return; }
  if (!d.ok) return;

  const o = d.overall;
  $('apHead').innerHTML = `<b>${esc(d.headline)}</b>`;

  $('apStats').innerHTML = '<div class="conrow">'
    + `<div class="stat"><b>${o.n}</b><span>closed</span></div>`
    + `<div class="stat"><b class="${o.profitable === true ? 'long'
        : o.profitable === false ? 'short' : ''}">`
      + `${o.expectancy_bps > 0 ? '+' : ''}${o.expectancy_bps}bps</b>`
      + `<span>per trade${o.exp_low == null ? '' :
          ' · ' + (o.exp_low > 0 ? '+' : '') + o.exp_low + ' to '
          + (o.exp_high > 0 ? '+' : '') + o.exp_high + ' honest range'}</span></div>`
    + `<div class="stat"><b>${(o.hit_rate*100).toFixed(0)}%</b>`
      + `<span>hit · needed ${(o.needed*100).toFixed(0)}%</span></div>`
    + `<div class="stat"><b>${o.avg_r}</b><span>average R</span></div>`
    + `<div class="stat"><b>${o.cost_drag_bps}bps</b>`
      + '<span>cost per trade</span></div>'
    + '</div>'
    + (d.exits && d.exits.n
        ? `<div class="msg"><b>Exits:</b> ${esc(d.exits.note)}</div>` : '')
    + (d.stops && d.stops.n
        ? `<div class="msg"><b>Stops:</b> ${esc(d.stops.note)}.</div>` : '');

  const slice = (title, rows) => !rows.length ? '' :
    `<h3 style="margin:14px 0 4px;font-size:13px">${title}</h3>`
    + '<table><thead><tr><th>slice</th><th>n</th>'
    + '<th>entry shape: hit vs needed</th>'
    + '<th>per trade, after cost</th><th>reads as</th></tr></thead><tbody>'
    + rows.map(b => {
        const cls = b.profitable === true ? 'long'
                  : b.profitable === false ? 'short' : '';
        const range = (b.exp_low == null) ? ''
          : `<span class="thin"> (${b.exp_low > 0 ? '+' : ''}${b.exp_low}`
            + ` to ${b.exp_high > 0 ? '+' : ''}${b.exp_high})</span>`;
        return `<tr><td>${esc(b.label)}</td><td>${b.n}</td>`
          + `<td style="width:180px">${band(b)}</td>`
          + `<td class="${cls}">${b.expectancy_bps > 0 ? '+' : ''}`
          + `${b.expectancy_bps}bps${b.meaningful ? range : ''}</td>`
          + `<td class="thin">${b.meaningful ? esc(b.verdict)
              : 'too few to read'}`
          + (b.divergence ? `<br><b>${esc(b.divergence)}</b>` : '')
          + '</td></tr>';
      }).join('')
    + '</tbody></table>';

  $('apSlices').innerHTML = BAND_KEY
    + slice('By how many columns agreed', d.by_agreeing)
    + slice('By grade', d.by_grade)
    + slice('By how it ended', d.by_exit)
    + slice('By how much room it had', d.by_runway)
    + (d.gates && d.gates.length
        ? '<h3 style="margin:14px 0 4px;font-size:13px">What it refused</h3>'
          + '<table><thead><tr><th>gate</th><th>blocked</th>'
          + '<th>of all polls</th></tr></thead><tbody>'
          + d.gates.slice(0, 8).map(g => `<tr><td>${esc(g.gate)}</td>`
              + `<td>${g.blocked}</td>`
              + `<td>${(g.share_of_all*100).toFixed(0)}%</td></tr>`).join('')
          + '</tbody></table>'
          + '<div class="thin">A gate that fires on most polls IS the '
          + 'strategy, whatever the description says.</div>'
        : '');

  loadProposals();
}

/* ------------------------------------------------------------ proposals */

async function loadProposals() {
  let d;
  try { d = await api('/api/proposals'); }
  catch (e) { $('apProposals').textContent = e.message; return; }

  const knobs = d.knobs || {};
  const changed = Object.entries(knobs).filter(([, v]) => v.changed);
  const current = '<div class="msg"><b>Settings now:</b> '
    + Object.entries(knobs).map(([k, v]) =>
        `${esc(k)} ${v.value}${v.changed
          ? ` <span class="thin">(default ${v.default}, `
            + `<a href="#" onclick="revertKnob('${esc(k)}');return false">`
            + 'put back</a>)</span>' : ''}`).join(' · ')
    + '</div>';

  const r = d.readiness || {};
  if (!d.proposals.length) {
    $('apProposals').innerHTML = current
      + `<div class="msg"><b>Tuning ${esc(d.tunes || '')}.</b> `
      + `${esc(r.note || '')}`
      + (r.ready ? ' Nothing to propose right now — it only asks when a '
                 + 'pair beats what is running on signals it has never '
                 + 'seen, and most of the time nothing does.' : '')
      + '</div>'
      + (d.raw
          ? '<div class="thin">There are no entry gates to tune any more — '
            + 'the vote refuses nothing, so the only thing left worth '
            + 'changing is where the levels sit. That is judged by the grid '
            + 'above, not by closed trades: a target changes what happens '
            + '<i>during</i> a trade, which an outcome cannot replay. '
            + 'Target and stop always move together.</div>'
          : '<div class="thin">Loosening is never proposed — the book has '
            + 'no result for trades it refused to take.</div>');
    return;
  }

  $('apProposals').innerHTML = current + d.proposals.map(p => {
    const head = p.param2
      ? `${esc(p.param)} + ${esc(p.param2)}: `
        + `${p.current_val}/${p.current_val2} → `
        + `${p.proposed_val}/${p.proposed_val2}`
      : `${esc(p.param)}: ${p.current_val} → ${p.proposed_val}`;
    return `
    <div class="prop">
      <b>${head}</b>
      ${p.param2 ? '<div class="thin">Adopted together — half of a tested '
        + 'pair is a setting nobody tested.</div>' : ''}
      <pre>${esc(p.rationale)}</pre>
      <button class="go" onclick="decideProposal('${esc(p.id)}',true)">
        Adopt</button>
      <button onclick="decideProposal('${esc(p.id)}',false)">Reject</button>
    </div>`; }).join('');
}

async function decideProposal(id, adopt) {
  try { await api('/api/proposals/decide?' + q({id, adopt}), {method: 'POST'}); }
  catch (e) { $('apProposals').textContent = e.message; return; }
  loadProposals();
}

async function revertKnob(param) {
  try { await api('/api/proposals/revert?' + q({param}), {method: 'POST'}); }
  catch (e) { return; }
  loadProposals();
}

async function scanProposals() {
  $('apProposals').textContent = 'walking forward…';
  try { await api('/api/proposals/scan', {method: 'POST'}); }
  catch (e) { $('apProposals').textContent = e.message; return; }
  loadProposals();
}

/* ---------------------------------------------------------------------
   The target / stop grid.

   Diverging colour, because the metric has a real zero. The pair is
   blue-red rather than the app's green-red for one measured reason: a
   grid asks you to tell dozens of adjacent cells apart, and green-red
   separates at deltaE 5 under deuteranopia where blue-red holds at 19.
   Every cell carries its number too, so colour is never doing the work
   alone.
   --------------------------------------------------------------------- */

const HEAT_POS = [42, 107, 176];    // pays
const HEAT_NEG = [196, 69, 54];     // loses
const HEAT_MID = [44, 49, 56];      // breaking even

function heatColor(v, scale) {
  if (v == null || !scale) return 'rgb(44,49,56)';
  const t = Math.max(-1, Math.min(1, v / scale));
  const pole = t >= 0 ? HEAT_POS : HEAT_NEG;
  const k = Math.abs(t);
  const c = HEAT_MID.map((m, i) => Math.round(m + (pole[i] - m) * k));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function swCfg() {
  return {
    coin: coin(), interval: $('swInt').value,
    tp_from: parseFloat($('swTpFrom').value || '5'),
    tp_to: parseFloat($('swTpTo').value || '40'),
    tp_step: parseFloat($('swTpStep').value || '5'),
    sl_from: parseFloat($('swSlFrom').value || '5'),
    sl_to: parseFloat($('swSlTo').value || '40'),
    sl_step: parseFloat($('swSlStep').value || '5'),
    cost_bps: parseFloat($('swFee').value || '0'),
    horizon: parseInt($('swHorizon').value || '60', 10),
    min_agreeing: parseInt($('swAgree').value || '0', 10),
    side: $('swSide').value,
  };
}

async function loadSignals() {
  let d;
  try {
    d = await api('/api/signals?' + q({coin: coin(),
                                       interval: $('swInt').value}));
  } catch (e) { $('swAvail').textContent = e.message; return; }
  if (!d.ok) { $('swAvail').textContent = d.detail || 'nothing stored'; return; }
  const hrs = (d.span && d.span.first && d.span.last)
    ? ((d.span.last - d.span.first) / 3600).toFixed(1) : '0';
  $('swAvail').innerHTML = `<b>${d.n} signals</b> on ${esc(d.coin)} `
    + `${esc(d.interval)} over ${hrs}h. `
    + (d.shapes || []).map(([sh, n]) =>
        `${esc(sh)}: ${n}`).join(' · ')
    + (d.n < 30 ? ' — <b>too few to test yet.</b> Leave the feed running; '
                + 'one lands per candle.' : '');
}

async function runSweep() {
  $('swVerdict').style.display = '';
  $('swVerdict').textContent = 'walking every pair over every signal…';
  $('swGrid').innerHTML = '';
  $('swShapes').innerHTML = '';
  let d;
  try { d = await api('/api/sweep?' + q(swCfg())); }
  catch (e) { $('swVerdict').textContent = e.message; return; }

  if (!d.ok) {
    $('swVerdict').innerHTML = `<b>${esc(d.detail || 'no result')}</b>`;
    if (d.span && d.span.n) {
      $('swAvail').textContent = `${d.span.n} readings stored for this `
        + 'market, but none matched.';
    }
    return;
  }

  $('swStamp').textContent = `${d.signals} signals · ${d.span_hours}h`;
  $('swVerdict').innerHTML = `<b>${esc(d.verdict)}</b>`;
  $('swAvail').innerHTML = `<b>${d.available} signals</b> available, `
    + `${d.signals} with price after them. Cost ${d.cost_bps}bps a round `
    + `trip, ${d.horizon} minutes given to each trade.`;
  paintHeat(d);
  paintShapes(d);
}

function paintHeat(d) {
  const scale = Math.max(...d.cells.map(c => Math.abs(c.expectancy)), 0.01);
  const by = {};
  for (const c of d.cells) by[c.tp_bps + '|' + c.sl_bps] = c;
  const bestKey = d.best.tp_bps + '|' + d.best.sl_bps;

  let h = '<table class="heat"><thead><tr>'
        + '<th style="text-align:left">stop \\u2193 / target \\u2192</th>'
        + d.tp_values.map(t => `<th>${t}</th>`).join('')
        + '</tr></thead><tbody>';
  for (const sl of d.sl_values) {
    h += `<tr><th style="text-align:left">${sl}</th>`;
    for (const tp of d.tp_values) {
      const c = by[tp + '|' + sl];
      if (!c) { h += '<td></td>'; continue; }
      const key = tp + '|' + sl;
      const cls = (key === bestKey ? 'best' : (c.significant ? 'sig' : ''));
      const tip = `target ${tp}bps / stop ${sl}bps · ${c.rr}R\n`
        + `${c.expectancy >= 0 ? '+' : ''}${c.expectancy}bps a trade `
        + `over ${c.n}\n`
        + `honest range ${c.ci_low} to ${c.ci_high}\n`
        + `hit ${(c.hit_rate*100).toFixed(0)}% · needs `
        + `${(c.breakeven*100).toFixed(0)}%\n`
        + `${c.targets} targets, ${c.stops} stops, ${c.timeouts} ran out\n`
        + `held ${c.avg_bars} min on average`;
      h += `<td class="${cls}" style="background:`
        + `${heatColor(c.expectancy, scale)}" title="${esc(tip)}">`
        + `${c.expectancy >= 0 ? '+' : ''}${c.expectancy.toFixed(1)}</td>`;
    }
    h += '</tr>';
  }
  h += '</tbody></table>';

  const swatch = v =>
    `<i style="width:22px;background:${heatColor(v, scale)}"></i>`;
  h += '<div class="heat-key">'
    + `<span>${swatch(-scale)}${swatch(-scale/2)}${swatch(0)}`
    + `${swatch(scale/2)}${swatch(scale)}</span>`
    + `<span>&minus;${scale.toFixed(1)}bps &rarr; breaking even &rarr; `
    + `+${scale.toFixed(1)}bps a trade, after cost</span>`
    + '<span><i style="width:12px;outline:1.5px solid var(--ink);'
    + 'outline-offset:-1.5px;background:transparent"></i>the whole honest '
    + 'range is on one side of zero</span>'
    + '<span><i style="width:12px;outline:2px solid var(--accent);'
    + 'outline-offset:-2px;background:transparent"></i>best cell</span>'
    + '<span>hover any square for its full numbers</span></div>'
    + '<div class="thin">Both levels touched inside one minute counts as a '
    + '<b>stop</b> — nothing in a bar says which came first, and taking the '
    + 'good one is what makes a grid look better than it is. Trades that '
    + 'never reach either level are marked out at the close, not dropped.</div>';

  $('swGrid').innerHTML = h;
}

function paintShapes(d) {
  const rows = d.by_shape || [];
  if (!rows.length) return;
  $('swShapes').innerHTML =
    '<h3 style="margin:0 0 4px;font-size:13px">How each vote shape did at '
    + `the best cell (${d.best.tp_bps}bps target, ${d.best.sl_bps}bps stop)`
    + '</h3>'
    + '<div class="thin" style="margin-bottom:6px">agreeing&ndash;against. '
    + '<b>3&ndash;0</b> is all three columns; <b>2&ndash;1</b> is two '
    + 'outvoting one; <b>1&ndash;0</b> is one column with two staying '
    + 'quiet. The cell was picked using all of them, so treat one '
    + 'shape standing out here as a lead to check, not a finding.'
    + '</div>'
    + '<table><thead><tr><th>shape</th><th>n</th><th>per trade</th>'
    + '<th>honest range</th><th>hit</th><th>needed</th>'
    + '<th>reads as</th></tr></thead><tbody>'
    + rows.map(r => `<tr><td><b>${esc(r.shape)}</b></td><td>${r.n}</td>`
        + `<td class="${r.expectancy > 0 ? 'long' : 'short'}">`
        + `${r.expectancy >= 0 ? '+' : ''}${r.expectancy.toFixed(2)}bps</td>`
        + `<td class="thin">${r.ci_low == null ? '—'
            : (r.ci_low >= 0 ? '+' : '') + r.ci_low.toFixed(1) + ' to '
              + (r.ci_high >= 0 ? '+' : '') + r.ci_high.toFixed(1)}</td>`
        + `<td>${(r.hit_rate*100).toFixed(0)}%</td>`
        + `<td>${(r.breakeven*100).toFixed(0)}%</td>`
        + `<td class="thin">${r.significant
            ? 'real at this sample size'
            : (r.n < 30 ? 'too few to read' : 'inside the noise')}</td>`
        + '</tr>').join('')
    + '</tbody></table>';
}

async function loadSuggest() {
  let d;
  try {
    d = await api('/api/suggest?' + q({
      coin: coin(), interval: $('sInt').value,
      mode: $('sMode').value,
      size: parseFloat($('sSize').value || '10000'),
      fee_bps: parseFloat($('sFee').value || '0')}));
  } catch (e) { $('sugCard').textContent = e.message; return; }
  paintSuggest(d);
  loadDecisions();
}

/* Book against price, always on show. This is the most useful thing on the
   panel precisely when there is NO trade, so it is painted before and
   independently of the call itself. */
function paintCompare(d) {
  const box = $('sugCompare');
  const c = d.confirmation;
  if (!c) { box.style.display = 'none'; return; }

  const cls = v => v === 'up' ? 'long' : v === 'down' ? 'short' : '';
  const arrow = v => v === 'up' ? '▲' : v === 'down' ? '▼' : '—';
  const verdictCls = c.verdict === 'confirmed' ? 'long'
                   : c.verdict === 'conflict' ? 'short' : '';

  // Held time and participation are what separate an agreement you can act
  // on from one that evaporates while you reach for the mouse.
  const p = c.participation;
  const heldCls = c.settled ? 'long' : c.agree ? 'short' : '';
  const partCls = !p ? '' : p.backed ? 'long' : 'short';

  box.style.display = '';
  box.innerHTML =
      '<div class="conrow">'
    + `<div class="stat"><b class="${cls(c.book)}">${arrow(c.book)} `
    + `${esc((c.book || '').toUpperCase())}</b><span>order book</span></div>`
    + `<div class="stat"><b class="${cls(c.candle)}">${arrow(c.candle)} `
    + `${esc((c.candle || '').toUpperCase())}</b><span>price action</span></div>`
    + (p ? `<div class="stat"><b class="${partCls}">`
           + `${(p.effort * 100).toFixed(0)}%</b>`
           + '<span>paid for</span></div>' : '')
    + `<div class="stat"><b class="${heldCls}">${(c.held_s || 0).toFixed(0)}s</b>`
    + `<span>held${c.flips ? ' · ' + c.flips + ' flips' : ''}</span></div>`
    + `<div class="stat"><b class="${verdictCls}">`
    + `${esc((c.verdict || '').toUpperCase())}</b><span>verdict</span></div>`
    + '</div>'
    + `<div class="msg" style="margin-top:6px">${esc(c.detail || '')}</div>`
    + (c.instability
        ? `<div class="msg" style="margin-top:4px;color:var(--down)">`
          + `<b>Agreeing but not settled</b> — ${esc(c.instability)}.</div>`
        : '');
}

/* Three independent columns and a letter. Never blended — averaging three
   directions turns a disagreement into a confident-looking number, and the
   disagreement is the part worth keeping. */
function paintThreeWay(d) {
  const box = $('sugThree');
  const t = d.three_way;
  if (!t) { box.style.display = 'none'; return; }

  const cls = v => v === 'up' ? 'long' : v === 'down' ? 'short' : '';
  const arrow = v => v === 'up' ? '▲' : v === 'down' ? '▼' : '—';
  const gradeCls = t.grade === 'A' ? 'long' : t.grade === 'X' ? 'short' : '';
  const road = d.runway;
  const dl = d.delta;
  // Stability, folded in here rather than repeated in a second row.
  const c = d.confirmation || {};
  const p = c.participation;
  const heldCls = c.settled ? 'long' : c.agree ? 'short' : '';
  const partCls = !p ? '' : p.backed ? 'long' : 'short';

  box.style.display = '';
  box.innerHTML =
      '<div class="conrow">'
    + `<div class="stat"><b class="${cls(t.book)}">${arrow(t.book)} `
    + `${esc((t.book || '').toUpperCase())}</b><span>book (resting)</span></div>`
    + `<div class="stat"><b class="${cls(t.delta)}">${arrow(t.delta)} `
    + `${esc((t.delta || '').toUpperCase())}</b><span>delta (crossing)</span></div>`
    + `<div class="stat"><b class="${cls(t.price)}">${arrow(t.price)} `
    + `${esc((t.price || '').toUpperCase())}</b><span>price (outcome)</span></div>`
    + `<div class="stat"><b class="${gradeCls}" style="font-size:20px">`
    + `${esc(t.grade)}</b><span>grade · ${t.agreeing}/3</span></div>`
    + (road ? `<div class="stat"><b>${road.clear_bps.toFixed(0)}bps</b>`
              + `<span>runway${road.open_road ? ' · open' : ''}</span></div>` : '')
    + (p ? `<div class="stat"><b class="${partCls}">`
           + `${(p.effort * 100).toFixed(0)}%</b><span>paid for</span></div>` : '')
    + `<div class="stat"><b class="${heldCls}">${(c.held_s || 0).toFixed(0)}s</b>`
    + `<span>held${c.flips ? ' · ' + c.flips + ' flips' : ''}</span></div>`
    + '</div>'
    + `<div class="msg" style="margin-top:6px">${esc(t.detail || '')}</div>`
    + (road ? `<div class="msg" style="margin-top:4px"><b>How far:</b> `
              + `${esc(road.describe || '')}.</div>` : '')
    + (dl && (dl.absorbed || dl.divergent || dl.working)
        ? `<div class="msg" style="margin-top:4px"><b>Tape:</b> `
          + `${esc(dl.describe || '')}.</div>` : '')
    + (c.instability
        ? `<div class="msg" style="margin-top:4px;color:var(--down)">`
          + `<b>Agreeing but not settled</b> — ${esc(c.instability)}.</div>`
        : '');
}

function paintSuggest(d) {
  if (!d) return;
  sugId = d.id || null;
  const act = $('sugActions');
  paintThreeWay(d);
  loadChart();

  // The feed rate decides whether anything below it is worth reading. Every
  // book signal measures CHANGE, so the rate change arrives at IS the
  // resolution of the signal — and a throttled feed looks exactly like a
  // quiet market from an empty panel.
  const fq = $('sugFeed');
  if (d.feed_quality) {
    const slow = (d.updates_per_s || 0) < 0.35;
    fq.style.display = '';
    fq.className = 'msg';
    fq.innerHTML = (slow ? '<b class="short">SLOW FEED</b> — ' : '')
      + esc(d.feed_quality)
      + (d.fast_book ? '' : ' (fast stream not confirmed)');
  } else { fq.style.display = 'none'; }

  if (!d.take) {
    // A refusal is an answer, not an error, and it still carries a grade and
    // a lean. "No trade but leaning long" and "no trade, book balanced" are
    // different states and only one of them is worth watching.
    const lean = d.side ? `${d.side.toUpperCase()} lean` : 'no lean';
    const cls = d.side === 'long' ? 'long' : d.side === 'short' ? 'short' : '';
    $('sugCard').innerHTML =
        `<div class="call"><b class="${cls}">${esc(d.grade || '—')}</b>`
      + `<span class="sub">${esc(lean)} · not tradeable`
      + (d.blocked_by ? ` · ${esc(d.blocked_by)}` : '') + '</span></div>'
      + `<div class="msg" style="margin-top:6px">${esc(d.detail || d.sentence || '')}</div>`;
    act.style.display = 'none';
    stamp('sugStamp', d.feed_age_s == null ? 0 : d.feed_age_s, 5, 30, 'book ');
    return;
  }

  const cls = d.side === 'long' ? 'long' : 'short';
  const sign = d.side === 'long' ? '+' : '-';
  const px = v => Number(v).toLocaleString(undefined, {maximumFractionDigits: 6});

  // The pair that decides it: what this trade NEEDS against what you GET.
  const need = d.breakeven == null ? null : (d.breakeven * 100);
  const got = d.measured_rate == null ? null : (d.measured_rate * 100);
  const edgeCls = d.edge_pts == null ? '' : d.edge_pts >= 0 ? 'long' : 'short';

  $('sugCard').innerHTML =
      `<div class="call"><b class="${cls}">${d.side.toUpperCase()}</b>`
    + `<span class="sub">grade ${esc(d.grade || '')} · ${d.target_ticks} ticks`
    + ` · ${sign}${d.target_bps.toFixed(1)}bps · ${d.rr}R`
    + ` · conviction ${(d.conviction*100).toFixed(0)}%</span></div>`
    + '<div class="conrow">'
    + `<div class="stat"><b>${px(d.entry)}</b><span>entry</span></div>`
    + `<div class="stat"><b class="${cls}">${px(d.target_px)}</b><span>target</span></div>`
    + `<div class="stat"><b class="${d.side === 'long' ? 'short' : 'long'}">${px(d.stop_px)}</b><span>invalid</span></div>`
    + `<div class="stat"><b>${d.cost_bps.toFixed(1)}bps</b><span>round trip</span></div>`
    + (need == null ? ''
       : `<div class="stat"><b>${need.toFixed(0)}%</b><span>needs to win</span></div>`)
    + (got == null ? ''
       : `<div class="stat"><b class="${edgeCls}">${got.toFixed(0)}%</b>`
         + `<span>you win (${d.measured_n})</span></div>`)
    + '</div>'
    + (d.edge_pts == null ? ''
       : `<div class="msg" style="margin-top:6px"><b class="${edgeCls}">`
         + `${d.edge_pts >= 0 ? '+' : ''}${d.edge_pts} points `
         + `${d.edge_pts >= 0 ? 'clear of' : 'short of'} breakeven</b>`
         + ` on ${d.measured_n} settled trades.</div>`)
    + (d.reasons && d.reasons.length
        ? `<div class="msg" style="margin-top:8px"><b>Why:</b> ${esc(d.reasons.join('; '))}.</div>` : '')
    + (d.cautions && d.cautions.length
        ? `<div class="msg" style="margin-top:4px;color:var(--down)"><b>Against:</b> ${esc(d.cautions.join('; '))}.</div>` : '')
    + (d.fees_included ? ''
        : `<div class="msg" style="margin-top:4px">Cost excludes fees — put your
           round-trip fee in the box above or every suggestion looks cheaper
           than it is.</div>`);

  act.style.display = '';
  $('sugDecided').textContent = d.decision && d.decision !== 'pending'
      ? 'already ' + d.decision + ' for this candle' : '';
  stamp('sugStamp', d.feed_age_s == null ? 0 : d.feed_age_s, 5, 30, 'book ');
}

async function decide(taken) {
  if (!sugId) { $('sugDecided').textContent = 'nothing to decide on'; return; }
  try {
    const r = await api('/api/decide?' + q({id: sugId, taken: taken}),
                        {method: 'POST'});
    // Only claim a decision was stored when the server says it was. It
    // refuses once the candle has closed, and reporting it anyway would
    // show a decision that is not in the table.
    $('sugDecided').textContent = r.ok && r.decision
        ? 'recorded: ' + r.decision : (r.note || 'could not record');
  } catch (e) { $('sugDecided').textContent = e.message; }
  loadDecisions();
}

async function loadDecisions() {
  let d;
  try {
    d = await api('/api/decisions?' + q({coin: coin(),
                                         interval: $('sInt').value}));
  } catch (e) { return; }
  const s = d.stats || {};
  const box = $('sugScore');
  const pct = v => v == null ? '—' : (v * 100).toFixed(0) + '%';
  const o = s.overall || {}, t = s.taken || {}, i = s.ignored || {};
  box.style.display = '';
  const bps = v => v == null ? '—' : (v >= 0 ? '+' : '') + v.toFixed(1) + 'bps';
  box.innerHTML =
      `<b>${esc(s.verdict || '')}</b>`
    + `<div class="msg" style="margin-top:6px">`
    + `all ${o.n || 0} settled · hit ${pct(o.hit_rate)} · `
    + `taken ${t.n || 0} (${pct(t.hit_rate)}) · `
    + `ignored ${i.n || 0} (${pct(i.hit_rate)}) · `
    + `${s.pending || 0} waiting on a close</div>`
    // Gross and net side by side. They answer different questions: gross
    // says whether the book read was right, net says whether the trade made
    // money, and only one of those pays for anything.
    + (o.n ? `<div class="msg" style="margin-top:4px">`
        + `avg per trade ${bps(o.avg_gross_bps)} gross, `
        + `<b class="${(o.avg_pnl_bps || 0) >= 0 ? 'long' : 'short'}">`
        + `${bps(o.avg_pnl_bps)} after cost</b> · `
        + `total ${bps(o.total_pnl_bps)} · `
        + `cost has taken ${bps(-(o.cost_drag_bps || 0))}</div>` : '');
}

function toggleSuggestAuto() {
  if (sugPoll) { clearInterval(sugPoll); sugPoll = null; }
  if ($('sAuto').checked) { loadSuggest(); sugPoll = setInterval(loadSuggest, 5000); }
}

function inputFlags(d) {
  // Report each input separately. One banner saying "flow and absorption
  // missing" while flow was visibly working in the signal list was simply
  // untrue, and an untrue status line costs more trust than a missing one.
  const i = d.inputs || {};
  let out = '';
  if (!i.flow) out += '<span class="flag late">NO FLOW</span>';
  if (i.absorption_warming)
    out += `<span class="flag late">ABSORPTION WARMING (${i.baseline_samples||0}/20)</span>`;
  else if (!i.absorption) out += '<span class="flag late">NO ABSORPTION</span>';
  if (!i.book) out += '<span class="flag late">NO BOOK</span>';
  return out;
}

function paintLadder(d) {
  const tfs = d.timeframes || [];
  const el = $('tfLadder'), say = $('tfSay');
  if (!tfs.length) {
    el.innerHTML = '';
    say.style.display = 'none';
    if (d.pressure_error) el.innerHTML = '<div class="msg err">' + d.pressure_error + '</div>';
    return;
  }

  el.innerHTML = tfs.map(t => {
    const w = Math.round(Math.abs(t.signed) * 50);
    return `<div class="tf ${t.winner}">
      <b>${t.timeframe}</b>
      <span class="who ${t.winner === 'buyers' ? 'long' : t.winner === 'sellers' ? 'short' : ''}">
        ${t.winner.toUpperCase()}</span>
      <span class="bar"><i class="${t.winner}" style="width:${w}%"></i></span>
      <span class="meta">${t.move_bps >= 0 ? '+' : ''}${t.move_bps.toFixed(1)}bps
        ${t.absorbing ? '· absorbed' : ''}
        ${t.measured ? '' : '· inferred'}</span>
    </div>`;
  }).join('');

  const c = d.confrontation;
  if (!c) { say.style.display = 'none'; return; }
  say.style.display = '';
  say.innerHTML = (c.conflicted ? '<span class="flag crowded">CONFLICT</span>'
                   : c.aligned ? '<span class="flag">ALIGNED</span>' : '')
                + c.verdict;
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
  stream.addEventListener('book', (ev) => {
    try { paintBookCall(JSON.parse(ev.data)); } catch (e) {}
  });

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
  paintLadder(d);
  const src = (d.source || '').indexOf('websocket') === 0;
  const flags = (src ? '<span class="flag">LIVE TAPE</span>'
                     : '<span class="flag late">POLLED</span>')
              + (d.early ? '<span class="flag late">EARLY</span>' : '')
              + (d.stale ? '<span class="flag crowded">STALE — candle and live price disagree</span>' : '')
              + inputFlags(d);

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

/* ---- the live chart, drawn from OUR OWN data -------------------------
   Candles the feed built from fills, and the calls that fired on them, on
   one canvas. A chart drawn from the same bars the signals were computed
   on cannot disagree with them — so when a marker looks wrong here, it IS
   wrong, which is the entire reason to look.                            */

let chartData = null;
/* Viewport over the bar array. `count` is how many bars are visible and
   `offset` is how many are hidden off the right edge, so offset 0 always
   means "pinned to the live bar" and a new candle arriving does not shove
   the view sideways while you are reading it. */
let chartView = {count: 70, offset: 0};
let chartHover = null;      // {x, y} in CSS pixels, or null
let chartDrag = null;

const CH_PAD = {l: 8, r: 64, t: 10, b: 22};
const CH_PROF = 46;   // width of the volume-profile gutter

function chartGeom() {
  const cv = $('sugChart');
  const w = cv.clientWidth, h = cv.clientHeight;
  // The forming bar's volume profile gets its own gutter against the price
  // axis. Drawn over the candles it lands squarely on the live bar, which is
  // the one bar you are actually watching.
  const pr = chartData && chartData.profile && chartData.profile.levels
             && chartData.profile.levels.length ? CH_PROF : 0;
  return {w, h, prof: pr, plotW: w - CH_PAD.l - CH_PAD.r - pr,
          plotH: h - CH_PAD.t - CH_PAD.b};
}

/* The slice of bars currently on screen, plus the index maths the pointer
   handlers need. Kept in one place so drawing and hit-testing can never
   disagree about which bar is under the cursor. */
function chartSlice() {
  if (!chartData || !chartData.bars) return null;
  const all = chartData.bars;
  const count = Math.max(10, Math.min(chartView.count, all.length));
  const maxOff = Math.max(0, all.length - count);
  const off = Math.max(0, Math.min(chartView.offset, maxOff));
  chartView.offset = off;
  const end = all.length - off;
  return {all, bars: all.slice(end - count, end), start: end - count,
          count, maxOff};
}

/* Decimals chosen from the price's own magnitude. A fixed 6 gives
   "3,458.740157" on an instrument that ticks in cents, which is unreadable
   at a glance and implies a precision the venue does not have. */
function chPx(v) {
  const a = Math.abs(v);
  const d = a >= 1000 ? 2 : a >= 10 ? 3 : a >= 0.1 ? 5 : 8;
  return Number(v).toLocaleString(undefined,
    {minimumFractionDigits: d, maximumFractionDigits: d});
}

function fmtClock(ts) {
  const d = new Date(ts * 1000);
  const p = n => String(n).padStart(2, '0');
  return p(d.getHours()) + ':' + p(d.getMinutes());
}

function fmtDay(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleDateString(undefined, {month: 'short', day: 'numeric'});
}

function drawChart() {
  const wrap = $('chartWrap');
  const s = chartSlice();
  if (!s || s.bars.length < 2) { wrap.style.display = 'none'; return; }
  wrap.style.display = '';

  const cv = $('sugChart');
  const dpr = window.devicePixelRatio || 1;
  const {w, h, prof: profW, plotW, plotH} = chartGeom();
  cv.width = w * dpr; cv.height = h * dpr;
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);

  const css = getComputedStyle(document.documentElement);
  const col = n => css.getPropertyValue(n).trim() || '#888';
  const up = col('--up'), down = col('--down'), dim = col('--dim'),
        line = col('--line'), ink = col('--ink'), panel = col('--panel');

  const bars = s.bars;
  let lo = Infinity, hi = -Infinity;
  for (const b of bars) { if (b.l < lo) lo = b.l; if (b.h > hi) hi = b.h; }
  const marks = (chartData.marks || []).filter(m => {
    const i = markIndex(m, s);
    return i >= 0 && i < bars.length;
  });
  for (const m of marks) {
    if (m.target) { lo = Math.min(lo, m.target); hi = Math.max(hi, m.target); }
    if (m.stop) { lo = Math.min(lo, m.stop); hi = Math.max(hi, m.stop); }
  }
  if (!isFinite(lo) || !isFinite(hi) || hi <= lo) return;
  const pad = (hi - lo) * 0.06; lo -= pad; hi += pad;

  const y = p => CH_PAD.t + (hi - p) / (hi - lo) * plotH;
  const step = plotW / bars.length;
  const xOf = i => CH_PAD.l + i * step + step / 2;
  const bw = Math.max(1, Math.min(step * 0.7, 16));

  g.clearRect(0, 0, w, h);
  g.font = '10px ui-monospace, monospace';

  /* Grid and price scale. Recessive on purpose — the marks are the data,
     the axes are furniture. */
  g.strokeStyle = line; g.fillStyle = dim; g.textAlign = 'left';
  for (let i = 0; i <= 4; i++) {
    const p = lo + (hi - lo) * i / 4, yy = Math.round(y(p)) + 0.5;
    g.beginPath(); g.moveTo(CH_PAD.l, yy); g.lineTo(CH_PAD.l + plotW, yy); g.stroke();
    g.fillText(chPx(p), CH_PAD.l + plotW + profW + 5, yy + 3);
  }

  /* Time axis. Ticks are spaced by pixels rather than by bar count so the
     labels stay readable at every zoom level instead of colliding when you
     zoom out. */
  g.textAlign = 'center';
  const wantEvery = Math.max(1, Math.ceil(64 / step));
  let lastDay = null;
  for (let i = 0; i < bars.length; i++) {
    if (i % wantEvery) continue;
    const x = xOf(i), ts = bars[i].ts;
    g.strokeStyle = line;
    g.beginPath();
    g.moveTo(Math.round(x) + 0.5, CH_PAD.t);
    g.lineTo(Math.round(x) + 0.5, CH_PAD.t + plotH);
    g.globalAlpha = 0.35; g.stroke(); g.globalAlpha = 1;
    g.fillStyle = dim;
    const day = fmtDay(ts);
    g.fillText(day !== lastDay ? day : fmtClock(ts), x, h - 7);
    lastDay = day;
  }

  /* Volume profile of the forming bar, along the right edge. */
  const prof = chartData.profile;
  if (profW) {
    const right = CH_PAD.l + plotW + profW;
    const maxV = Math.max(...prof.levels.map(l => l.v));
    for (const l of prof.levels) {
      if (l.px < lo || l.px > hi) continue;
      const lw = Math.max(1, (l.v / maxV) * (profW - 4));
      const poc = prof.poc && Math.abs(l.px - prof.poc) < 1e-9;
      g.fillStyle = poc ? ink : dim;
      g.globalAlpha = poc ? 0.55 : 0.3;
      g.fillRect(right - lw, y(l.px) - 1, lw, 2);
    }
    g.globalAlpha = 1;
  }

  // candles
  bars.forEach((b, i) => {
    const x = xOf(i), rising = b.c >= b.o;
    g.strokeStyle = g.fillStyle = rising ? up : down;
    g.beginPath();
    g.moveTo(Math.round(x) + 0.5, y(b.h));
    g.lineTo(Math.round(x) + 0.5, y(b.l));
    g.stroke();
    const top = y(Math.max(b.o, b.c)), bot = y(Math.min(b.o, b.c));
    g.fillRect(x - bw / 2, top, bw, Math.max(1, bot - top));
    if (b.live) {                       // mark the bar still forming
      g.strokeStyle = ink; g.globalAlpha = 0.35; g.setLineDash([2, 2]);
      g.strokeRect(x - bw / 2 - 2.5, top - 2.5, bw + 5,
                   Math.max(1, bot - top) + 5);
      g.setLineDash([]); g.globalAlpha = 1;
    }
  });

  // the agent's trades, entry through exit
  for (const m of marks) {
    const i = markIndex(m, s), x = xOf(i);
    const long = m.side === 'long', c = long ? up : down;
    const xi = exitIndex(m, s);
    const xe = xi == null ? null : xOf(xi);

    // Target and stop, spanning the life of the trade rather than a fixed
    // stub, so you can see which one price actually reached.
    const right = xe == null ? x + step * 2.2 : Math.max(xe, x + step * 0.5);
    g.strokeStyle = c; g.globalAlpha = 0.4; g.setLineDash([2, 3]);
    for (const p of [m.target, m.stop]) {
      if (!p) continue;
      g.beginPath();
      g.moveTo(x - step * 0.4, y(p)); g.lineTo(right, y(p));
      g.stroke();
    }
    g.setLineDash([]); g.globalAlpha = 1;

    const yy = y(m.entry);

    // Entry to exit, so the hold shows as a line you can read length off.
    if (xe != null && m.exit_px) {
      g.strokeStyle = m.won ? up : down;
      g.globalAlpha = 0.7; g.lineWidth = 1.5;
      g.beginPath(); g.moveTo(x, yy); g.lineTo(xe, y(m.exit_px)); g.stroke();
      g.lineWidth = 1; g.globalAlpha = 1;
    }

    // The entry triangle: filled once the trade is done, hollow while it
    // is still running.
    g.beginPath();
    if (long) { g.moveTo(x, yy + 9); g.lineTo(x - 5, yy + 18); g.lineTo(x + 5, yy + 18); }
    else { g.moveTo(x, yy - 9); g.lineTo(x - 5, yy - 18); g.lineTo(x + 5, yy - 18); }
    g.closePath();
    g.fillStyle = c; g.strokeStyle = c; g.lineWidth = 1.5;
    if (m.open) g.stroke(); else g.fill();
    g.lineWidth = 1;

    // The exit dot: coloured by WON or LOST, not by which level it was.
    // Hitting the target is not the question; keeping money is.
    if (xe != null && m.exit_px) {
      g.fillStyle = m.won ? up : down;
      g.strokeStyle = panel; g.lineWidth = 1.5;
      g.beginPath(); g.arc(xe, y(m.exit_px), 3.5, 0, Math.PI * 2);
      g.fill(); g.stroke(); g.lineWidth = 1;
    }
  }

  // crosshair and readout
  if (chartHover) {
    const i = Math.max(0, Math.min(bars.length - 1,
                                   Math.floor((chartHover.x - CH_PAD.l) / step)));
    const b = bars[i];
    if (b) {
      const x = xOf(i);
      g.strokeStyle = dim; g.globalAlpha = 0.6; g.setLineDash([3, 3]);
      g.beginPath();
      g.moveTo(Math.round(x) + 0.5, CH_PAD.t);
      g.lineTo(Math.round(x) + 0.5, CH_PAD.t + plotH);
      g.moveTo(CH_PAD.l, Math.round(chartHover.y) + 0.5);
      g.lineTo(CH_PAD.l + plotW, Math.round(chartHover.y) + 0.5);
      g.stroke();
      g.setLineDash([]); g.globalAlpha = 1;

      // price under the cursor, on the scale
      const pv = hi - (chartHover.y - CH_PAD.t) / plotH * (hi - lo);
      g.fillStyle = ink;
      g.fillRect(CH_PAD.l + plotW + profW + 2, chartHover.y - 7,
                 CH_PAD.r - 4, 14);
      g.fillStyle = panel; g.textAlign = 'left';
      g.fillText(chPx(pv), CH_PAD.l + plotW + profW + 5, chartHover.y + 3);

      // A trade is findable from EITHER end: you hover where you see a
      // mark, and both ends carry marks.
      const hit = marks.find(m => markIndex(m, s) === i)
               || marks.find(m => exitIndex(m, s) === i);
      const px = chPx;
      const rows = [
        fmtDay(b.ts) + ' ' + fmtClock(b.ts),
        'O ' + px(b.o) + '  H ' + px(b.h),
        'L ' + px(b.l) + '  C ' + px(b.c),
      ];
      if (hit) {
        rows.push('');
        rows.push((hit.side || '').toUpperCase() + ' @ ' + px(hit.entry)
                  + (hit.shape ? '   ' + hit.shape : ''));
        // Why it took it: the columns, in words. This is the thing that
        // makes a marker worth hovering over.
        if (hit.why) rows.push('why: ' + hit.why);
        if (hit.tp_bps != null && hit.sl_bps != null) {
          rows.push('target ' + Number(hit.tp_bps).toFixed(0) + 'bps · stop '
                    + Number(hit.sl_bps).toFixed(0) + 'bps');
        }
        if (hit.open) {
          rows.push('still open');
        } else {
          const w = hit.won ? 'WIN' : 'LOSS';
          const n = hit.net_bps == null ? ''
            : '  ' + (hit.net_bps >= 0 ? '+' : '')
              + Number(hit.net_bps).toFixed(1) + 'bps net';
          rows.push(w + ' — ' + (hit.exit_reason || 'closed') + n);
          if (hit.exit_px) rows.push('out at ' + px(hit.exit_px));
          if (hit.held_s != null) {
            const h = Number(hit.held_s);
            rows.push('held ' + (h >= 90 ? (h / 60).toFixed(0) + 'm'
                                         : h.toFixed(0) + 's')
                      + (hit.mfe_bps != null
                         ? '   best ' + (hit.mfe_bps >= 0 ? '+' : '')
                           + Number(hit.mfe_bps).toFixed(1) + 'bps' : ''));
          }
        }
      }
      const bw2 = 212, bh = 8 + rows.length * 13;
      let bx = x + 12, by = CH_PAD.t + 6;
      if (bx + bw2 > CH_PAD.l + plotW) bx = x - 12 - bw2;
      g.fillStyle = panel; g.globalAlpha = 0.95;
      g.fillRect(bx, by, bw2, bh);
      g.globalAlpha = 1; g.strokeStyle = line; g.strokeRect(bx + 0.5, by + 0.5, bw2, bh);
      g.fillStyle = ink; g.textAlign = 'left';
      rows.forEach((r, k) => g.fillText(r, bx + 6, by + 15 + k * 13));
    }
  }
}

/* Which visible slot a call belongs in. Derived from the bar timestamps
   rather than from a stored index, so it stays correct while panning. */
function markIndex(m, s) {
  const iv = chartData.interval_s || 900;
  return Math.round(((m.entry_ts || m.ts) - s.bars[0].ts) / iv);
}

/* Which slot the EXIT landed in. Drawing the exit on the entry bar --
   which is what the first version did -- hides the one thing a chart is
   good at showing: how long the trade took and where it ended. */
function exitIndex(m, s) {
  if (!m.exit_ts) return null;
  const iv = chartData.interval_s || 900;
  return Math.round((m.exit_ts - s.bars[0].ts) / iv);
}

/* ---- interaction ------------------------------------------------------ */

function chartPoint(ev) {
  const r = $('sugChart').getBoundingClientRect();
  const t = ev.touches && ev.touches[0];
  return {x: (t ? t.clientX : ev.clientX) - r.left,
          y: (t ? t.clientY : ev.clientY) - r.top};
}

function wireChart() {
  const cv = $('sugChart');
  if (!cv || cv.dataset.wired) return;
  cv.dataset.wired = '1';
  cv.style.cursor = 'crosshair';

  cv.addEventListener('wheel', ev => {
    if (!chartData) return;
    ev.preventDefault();
    const s = chartSlice(); if (!s) return;
    const {plotW} = chartGeom();
    // Zoom about the cursor, so the bar under the pointer stays put.
    const frac = Math.max(0, Math.min(1, (chartPoint(ev).x - CH_PAD.l) / plotW));
    const before = chartView.count;
    const next = Math.round(before * (ev.deltaY > 0 ? 1.15 : 0.87));
    chartView.count = Math.max(10, Math.min(next, s.all.length));
    const grew = chartView.count - before;
    chartView.offset = Math.max(0, chartView.offset + Math.round(grew * (1 - frac)));
    drawChart();
  }, {passive: false});

  const start = ev => {
    chartDrag = {x: chartPoint(ev).x, offset: chartView.offset};
    cv.style.cursor = 'grabbing';
  };
  const move = ev => {
    const p = chartPoint(ev);
    chartHover = p;
    if (chartDrag) {
      const s = chartSlice();
      if (s) {
        const step = chartGeom().plotW / s.count;
        chartView.offset = Math.max(0, Math.min(
          s.maxOff, chartDrag.offset + Math.round((p.x - chartDrag.x) / step)));
      }
    }
    drawChart();
  };
  const end = () => { chartDrag = null; cv.style.cursor = 'crosshair'; };

  cv.addEventListener('mousedown', start);
  cv.addEventListener('mousemove', move);
  cv.addEventListener('mouseup', end);
  cv.addEventListener('mouseleave', () => {
    chartHover = null; end(); drawChart();
  });
  cv.addEventListener('touchstart', ev => { start(ev); }, {passive: true});
  cv.addEventListener('touchmove', ev => { move(ev); }, {passive: true});
  cv.addEventListener('touchend', end);
  cv.addEventListener('dblclick', resetChartView);
}

function resetChartView() {
  chartView = {count: 70, offset: 0};
  drawChart();
}

async function loadChart() {
  try {
    chartData = await api('/api/chart?' + q({
      coin: coin(), interval: $('sInt').value, bars: 400}));
  } catch (e) { return; }
  if (chartData && chartData.error) { chartData = null; return; }
  wireChart();
  drawChart();
  const n = (chartData.marks || []).length;
  $('chartNote').innerHTML =
      `${esc(chartData.interval || '')} · ${esc(chartData.source || '')} · `
    + `${n} call${n === 1 ? '' : 's'} · scroll to zoom, drag to pan, `
    + 'double-click to reset';
}

window.addEventListener('resize', () => { if (chartData) drawChart(); });

/* ---- book vs price: the table ---------------------------------------- */

const bpsCell = v => v == null ? '—'
  : `<b class="${v > 0 ? 'long' : v < 0 ? 'short' : ''}">`
    + `${v >= 0 ? '+' : ''}${v.toFixed(1)}</b>`;
const pctCell = v => v == null ? '—' : (v * 100).toFixed(0) + '%';

function agreementRows(rows, withSides) {
  return rows.map(r => {
    // A row with nothing in it is greyed rather than hidden: "we have never
    // seen this state" is itself worth knowing.
    const dim = r.n ? '' : ' style="opacity:.35"';
    return `<tr${dim}><td><b>${esc(r.state)}</b><br>`
      + `<span style="color:var(--dim);font-size:11px">${esc(r.detail || '')}</span></td>`
      + `<td>${r.n}</td>`
      + `<td>${r.share == null ? '—' : (r.share * 100).toFixed(0) + '%'}</td>`
      + `<td>${bpsCell(r.avg_called_bps)}</td>`
      + `<td>${r.median_called_bps == null ? '—' : r.median_called_bps.toFixed(1)}</td>`
      + `<td>${pctCell(r.hit_rate)}</td>`
      + (withSides
          ? `<td>${pctCell(r.book_right)}</td><td>${pctCell(r.price_right)}</td>`
          : '')
      + `<td>${r.avg_mfe_bps == null ? '—' : '+' + r.avg_mfe_bps.toFixed(1)}</td>`
      + `<td>${r.avg_mae_bps == null ? '—' : r.avg_mae_bps.toFixed(1)}</td>`
      + '</tr>';
  }).join('');
}

async function loadAgreement() {
  let d;
  try {
    d = await api('/api/agreement?' + q({
      coin: $('agThis').checked ? coin() : '',
      interval: $('agInt').value}));
  } catch (e) { $('agSay').style.display = ''; $('agSay').textContent = e.message; return; }

  $('agSay').style.display = '';
  $('agSay').innerHTML = `<b>${esc(d.verdict || '')}</b>`
    + `<div class="msg" style="margin-top:6px">${d.n} settled · `
    + `${d.pending || 0} waiting on a close · ${esc(d.source || '')}</div>`;

  $('agTable').innerHTML =
      '<table><tr><th>state</th><th>n</th><th>share</th>'
    + '<th>avg move<br>called (bps)</th><th>median</th><th>closed<br>that way</th>'
    + '<th>book<br>right</th><th>price<br>right</th>'
    + '<th>avg best</th><th>avg worst</th></tr>'
    + agreementRows(d.table || [], true) + '</table>';

  loadSlice();
}

async function runReplay() {
  $('agSay').style.display = '';
  $('agSay').textContent = 'replaying price action over history…';
  let d;
  try {
    d = await api('/api/agreement-replay?' + q({
      coin: coin(), interval: $('agInt').value || '15m', bars: 800}),
      {method: 'POST'});
  } catch (e) { $('agSay').textContent = e.message; return; }
  if (!d.ok) { $('agSay').textContent = d.error; return; }

  $('agSay').innerHTML = `<b>${esc(d.verdict || '')}</b>`
    + `<div class="msg" style="margin-top:6px">${esc(d.caveat || '')}</div>`;
  $('agTable').innerHTML =
      `<div class="msg" style="margin-bottom:6px">${d.bars} bars, `
    + `${d.sub_bars} one-minute sub-bars, read ${esc(d.read_at || '')}. `
    + '<b>Price action only — the book half is not in this table.</b></div>'
    + '<table><tr><th>state</th><th>n</th><th>share</th>'
    + '<th>avg move<br>called (bps)</th><th>median</th><th>closed<br>that way</th>'
    + '<th>avg best</th><th>avg worst</th></tr>'
    + agreementRows(d.table || [], false) + '</table>';
}

async function loadSlice() {
  let d;
  try {
    d = await api('/api/feature-slice?' + q({
      coin: $('agThis').checked ? coin() : '',
      interval: $('agInt').value,
      feature: $('agFeat').value || 'tilt',
      verdict: $('agVerdict').value}));
  } catch (e) { return; }

  const sel = $('agFeat');
  const keep = sel.value;
  const names = d.available && d.available.length ? d.available
              : ['tilt', 'replenish', 'depletion', 'imbalance', 'slope_bps',
                 'position', 'magnet_bps', 'magnet_with_call'];
  sel.innerHTML = names.map(n =>
    `<option value="${esc(n)}">${esc(n)}</option>`).join('');
  if (keep && names.includes(keep)) sel.value = keep;

  const rows = (d.buckets || []);
  $('agSlice').innerHTML =
      '<table><tr><th>range</th><th>n</th><th>avg move called (bps)</th>'
    + '<th>closed that way</th></tr>'
    + rows.map(b => `<tr${b.n ? '' : ' style="opacity:.35"'}>`
        + `<td>${b.from.toFixed(2)} → ${b.to.toFixed(2)}</td>`
        + `<td>${b.n}</td><td>${bpsCell(b.avg_called_bps)}</td>`
        + `<td>${pctCell(b.hit_rate)}</td></tr>`).join('')
    + '</table>'
    + `<div class="msg" style="margin-top:6px">${esc(d.note || '')}</div>`;
}

/* ---- backfill from the S3 archive ------------------------------------ */

let bfPoll = null;

async function estimateBackfill() {
  let d;
  try {
    d = await api('/api/backfill/estimate?' + q({
      coin: coin(), days: parseFloat($('bfDays').value || '2')}));
  } catch (e) { $('bfSay').textContent = e.message; return; }

  $('bfSay').innerHTML =
      `About <b>${d.estimated_gib} GiB</b> across ${d.object_count} hourly `
    + `objects — roughly <b>$${d.estimated_usd}</b> on your AWS bill. `
    + (d.credentials
        ? '<span class="long">AWS credentials found.</span>'
        : '<b class="short">No AWS credentials — set AWS_ACCESS_KEY_ID and '
          + 'AWS_SECRET_ACCESS_KEY in Railway.</b>')
    + `<div class="msg" style="margin-top:4px">${esc(d.note || '')}</div>`;
}

async function startBackfill() {
  let d;
  try {
    d = await api('/api/backfill?' + q({
      coin: coin(), interval: $('agInt').value || '15m',
      days: parseFloat($('bfDays').value || '2'),
      at: parseFloat($('bfAt').value || '0.33'),
      max_gib: parseFloat($('bfCap').value || '2')}), {method: 'POST'});
  } catch (e) { $('bfSay').textContent = e.message; return; }

  if (d.ok === false) { $('bfSay').innerHTML = `<b class="short">${esc(d.error)}</b>`; return; }
  if (!bfPoll) bfPoll = setInterval(pollBackfill, 2000);
  pollBackfill();
}

async function stopBackfill() {
  try { await api('/api/backfill/stop', {method: 'POST'}); } catch (e) {}
  pollBackfill();
}

async function pollBackfill() {
  let d;
  try { d = await api('/api/backfill'); } catch (e) { return; }
  const p = d.progress || {};

  $('bfSay').innerHTML =
      (d.running ? `<b>Running — ${p.pct || 0}%</b> · ` : '<b>Finished.</b> ')
    + `${p.hours_done || 0}/${p.hours_total || 0} hours · `
    + `${p.bars_seen || 0} bars read · <b>${p.states || 0} states recorded</b>`
    + (p.transfer ? `<div class="msg" style="margin-top:4px">${esc(p.transfer)}</div>` : '')
    + (p.error ? `<div class="msg" style="margin-top:4px"><b class="short">${esc(p.error)}</b></div>` : '')
    + (p.skipped && Object.keys(p.skipped).length
        ? `<div class="msg" style="margin-top:4px">skipped: `
          + Object.entries(p.skipped).map(([k, v]) => `${v} ${esc(k)}`).join(', ')
          + '</div>' : '');

  if (!d.running) {
    if (bfPoll) { clearInterval(bfPoll); bfPoll = null; }
    loadAgreement();          // the table just grew
  }
}

/* ---- uploaded chart history ------------------------------------------ */

async function uploadHistory() {
  const f = $('histFile').files[0];
  if (!f) { $('histSay').textContent = 'pick a file first'; return; }
  $('histSay').textContent = 'reading ' + f.name + '…';

  const body = new FormData();
  body.append('file', f);
  try {
    const d = await api('/api/upload-history?' + q({
                          coin: coin(), merge_into: $('histMerge').value}),
                        {method: 'POST', body: body});
    $('histSay').innerHTML = (d.ok ? '' : '<b>Rejected.</b> ')
      + esc(d.describe || d.error || '');
  } catch (e) { $('histSay').textContent = e.message; return; }
  loadDatasets();
}

async function loadDatasets() {
  let d;
  try { d = await api('/api/datasets?' + q({coin: coin()})); }
  catch (e) { return; }

  const rows = d.datasets || [];
  const when = ts => new Date(ts * 1000).toISOString().slice(0, 10);

  $('histList').innerHTML = !rows.length ? ''
    : '<table><tr><th>file</th><th>market</th><th>tf</th><th>bars</th>'
      + '<th>span</th><th></th></tr>'
      + rows.map(r => `<tr><td>${esc(r.name)}</td><td>${esc(r.coin)}</td>`
          + `<td>${esc(r.interval)}</td><td>${r.bars.toLocaleString()}</td>`
          + `<td>${when(r.start_ts)} → ${when(r.end_ts)}</td>`
          + `<td><button onclick="dropDataset('${esc(r.id)}')">delete</button></td>`
          + '</tr>').join('') + '</table>';

  // Both selectors are rebuilt from the same list, keeping whatever was
  // chosen if it still exists.
  for (const [id, head] of [['histMerge', '— new dataset —'],
                            ['btSource', 'exchange candles']]) {
    const sel = $(id);
    if (!sel) continue;
    const keep = sel.value;
    sel.innerHTML = `<option value="">${head}</option>`
      + rows.map(r => `<option value="${esc(r.id)}">${esc(r.name)} · `
          + `${esc(r.interval)} · ${r.bars.toLocaleString()} bars</option>`).join('');
    if (keep && rows.some(r => r.id === keep)) sel.value = keep;
  }
}

async function dropDataset(id) {
  try { await api('/api/datasets/delete?' + q({id: id}), {method: 'POST'}); }
  catch (e) { $('histSay').textContent = e.message; return; }
  loadDatasets();
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
                  dataset: $('btSource') ? $('btSource').value : '',
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
