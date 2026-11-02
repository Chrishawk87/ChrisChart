"""Closed CME time candles, kept across restarts.

WHY THIS EXISTS.

The CME tape builds its candles from prints as they arrive and holds
nothing from before the engine started. Every restart therefore began
with an empty 1m, 5m and 15m, and `/api/read` refuses to build a
ladder until the traded interval has ten closed bars -- two and a half
hours of live session at 15m. For that whole time the stack showed
rows with no direction that never resolved, and there was nothing on
the screen to say the cause was an empty history rather than a quiet
market.

The OFI bars already survive a restart; `ofistore` reloads them from
SQLite on boot. That is exactly why the footprint panel came back
instantly while the stack beside it had nothing. This does the same
job for the time candles, on the same file, following the same shape:
a store that only ever appends, and a recorder on its own thread that
the tape never waits for.

THE FLOW IS PERSISTED, NOT JUST THE PRICES.

A candle restored as OHLCV alone is useless to the thing that needs
it. The ladder's direction comes from `buy_notional` against
`sell_notional`, and `CandleBuilder.seed` -- the obvious way to put
history back -- sets `flow_measured=False` precisely because a candle
fetched as OHLCV never saw an aggressor. Restoring that way would
refill the chart, clear the ten-bar gate, and leave every row reading
contested: a fix that looks complete and reads nothing. So the rows
carry the flow, and they go back as `LiveCandle` into the builder's
own list rather than through `seed`.

Nothing here places, sizes or times an order.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .db import ThreadedDB
from .live import LiveCandle

SCHEMA = """
CREATE TABLE IF NOT EXISTS cme_candles (
    symbol      TEXT    NOT NULL,
    interval    TEXT    NOT NULL,
    start_ts    REAL    NOT NULL,
    interval_s  REAL    NOT NULL,
    open        REAL    NOT NULL,
    high        REAL    NOT NULL,
    low         REAL    NOT NULL,
    close       REAL    NOT NULL,
    volume      REAL    NOT NULL DEFAULT 0,
    trades      INTEGER NOT NULL DEFAULT 0,
    buy_notional  REAL  NOT NULL DEFAULT 0,
    sell_notional REAL  NOT NULL DEFAULT 0,
    flow_measured INTEGER NOT NULL DEFAULT 1,
    seeded        INTEGER NOT NULL DEFAULT 1,
    written_at  REAL    NOT NULL,
    PRIMARY KEY (symbol, interval, start_ts)
);
CREATE INDEX IF NOT EXISTS cme_candles_recent
    ON cme_candles (symbol, interval, start_ts DESC);
"""

_COLUMNS = ("symbol", "interval", "start_ts", "interval_s", "open",
            "high", "low", "close", "volume", "trades", "buy_notional",
            "sell_notional", "flow_measured", "seeded", "written_at")

# A day of minutes, which is also five days of 5m and fifteen of 15m.
# The tape itself keeps 1,500 closed bars per interval; holding more on
# disk than the builder will accept back is storage nobody reads.
KEEP_PER_INTERVAL = 1_500
FLUSH_S = 20.0


def _row_of(symbol: str, interval: str, c: LiveCandle,
            now: float) -> tuple:
    return (
        str(symbol), str(interval), float(c.start_ts),
        float(c.interval_s), float(c.open), float(c.high),
        float(c.low), float(c.close), float(c.volume or 0.0),
        int(c.trades or 0), float(getattr(c, "buy_notional", 0.0) or 0.0),
        float(getattr(c, "sell_notional", 0.0) or 0.0),
        1 if getattr(c, "flow_measured", True) else 0,
        1 if getattr(c, "seeded", True) else 0,
        float(now),
    )


def _candle_of(r: sqlite3.Row) -> LiveCandle:
    return LiveCandle(
        interval_s=float(r["interval_s"]), start_ts=float(r["start_ts"]),
        open=float(r["open"]), high=float(r["high"]),
        low=float(r["low"]), close=float(r["close"]),
        volume=float(r["volume"]), trades=int(r["trades"]),
        buy_notional=float(r["buy_notional"]),
        sell_notional=float(r["sell_notional"]),
        # BOTH FLAGS COME BACK AS THEY WERE WRITTEN. A restored bar
        # that claims flow it never measured would put an invented
        # direction into the ladder; one that disclaims flow it did
        # measure throws away the only thing that makes it useful.
        flow_measured=bool(r["flow_measured"]),
        seeded=bool(r["seeded"]),
    )


class CandleStore(ThreadedDB):
    """Closed time candles, one table across every symbol and interval."""

    def __init__(self, path: str | Path = "liqmap.db"):
        super().__init__(path, SCHEMA)

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # -- writing ---------------------------------------------------------

    def record_many(self, symbol: str, interval: str, candles) -> int:
        """Write closed candles. Returns how many rows were new.

        A candle already on disk is IGNORED rather than updated. A
        closed bar is a fact that does not change, so a second write of
        one can only be a replay, and overwriting it would let a
        reload-then-reflush cycle rewrite history with whatever this
        build happened to compute.
        """
        rows, now = [], time.time()
        for c in candles:
            if c is None or float(getattr(c, "interval_s", 0.0)) <= 0:
                continue
            rows.append(_row_of(symbol, interval, c, now))
        if not rows:
            return 0
        sql = ("INSERT OR IGNORE INTO cme_candles (" + ",".join(_COLUMNS)
               + ") VALUES (" + ",".join("?" * len(_COLUMNS)) + ")")
        with self._tx() as conn:
            before = conn.total_changes
            conn.executemany(sql, rows)
            return conn.total_changes - before

    def prune(self, keep: int = KEEP_PER_INTERVAL) -> int:
        """Trim each symbol/interval to its most recent `keep` rows."""
        keep = max(1, int(keep))
        removed = 0
        with self._tx() as conn:
            pairs = conn.execute(
                "SELECT DISTINCT symbol, interval FROM cme_candles"
            ).fetchall()
            for sym, iv in pairs:
                cur = conn.execute(
                    "DELETE FROM cme_candles WHERE symbol=? AND interval=?"
                    " AND start_ts NOT IN ("
                    "  SELECT start_ts FROM cme_candles"
                    "  WHERE symbol=? AND interval=?"
                    "  ORDER BY start_ts DESC LIMIT ?)",
                    (sym, iv, sym, iv, keep))
                removed += cur.rowcount or 0
        return removed

    # -- reading ---------------------------------------------------------

    def recent(self, symbol: str, interval: str,
               limit: int = KEEP_PER_INTERVAL) -> list:
        """The newest `limit` candles, OLDEST FIRST.

        Oldest first because that is the order a builder's history is
        in and the order every reader downstream assumes. Returning
        newest-first here would put a reversed chart on the screen
        after a restart and nowhere else, which is the kind of bug that
        survives a long time.
        """
        rows = self._conn.execute(
            "SELECT * FROM cme_candles WHERE symbol=? AND interval=?"
            " ORDER BY start_ts DESC LIMIT ?",
            (str(symbol), str(interval), max(1, int(limit)))).fetchall()
        return [_candle_of(r) for r in reversed(rows)]

    def last_start(self, symbol: str, interval: str) -> float | None:
        """The newest stored start, or None when nothing is stored.

        NONE, NOT ZERO. Zero is a real timestamp, and a sentinel that
        collides with a real value is a bug waiting for the one input
        that looks like it: with `start_ts > mark` as the filter, a
        marker of 0.0 standing for "nothing stored" silently drops the
        candle at 0.0 on the first write.
        """
        r = self._conn.execute(
            "SELECT MAX(start_ts) FROM cme_candles"
            " WHERE symbol=? AND interval=?",
            (str(symbol), str(interval))).fetchone()
        return float(r[0]) if r and r[0] is not None else None

    def count(self, symbol: str | None = None,
              interval: str | None = None) -> int:
        sql = "SELECT COUNT(*) FROM cme_candles"
        args: list = []
        where = []
        if symbol:
            where.append("symbol=?")
            args.append(str(symbol))
        if interval:
            where.append("interval=?")
            args.append(str(interval))
        if where:
            sql += " WHERE " + " AND ".join(where)
        r = self._conn.execute(sql, args).fetchone()
        return int(r[0]) if r else 0

    def intervals(self, symbol: str) -> list:
        rows = self._conn.execute(
            "SELECT DISTINCT interval FROM cme_candles WHERE symbol=?",
            (str(symbol),)).fetchall()
        return [r[0] for r in rows]


class CandleRecorder:
    """Keeps a tape's closed candles and the store in step.

    Off the tape thread on purpose: it wakes on a timer, takes what has
    closed since it last looked, and writes that batch. If the disk is
    slow, or missing, or the write raises, the tape does not notice and
    the feed does not stall.
    """

    def __init__(self, store: CandleStore, tape,
                 period_s: float = FLUSH_S,
                 keep: int = KEEP_PER_INTERVAL) -> None:
        self.store = store
        self.tape = tape
        self.period_s = float(period_s)
        self.keep = int(keep)
        self.written = 0
        self.errors = 0
        self.restored = 0
        self.last_flush = 0.0
        # Newest start_ts written per interval, taken from disk at boot
        # so a reload does not immediately rewrite what it just read.
        # None means nothing is stored for that interval yet, which is
        # NOT the same as a marker at zero -- see `last_start`.
        self._mark: dict[str, float | None] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def symbol(self) -> str:
        return str(getattr(self.tape, "symbol", "") or "ES")

    def seed(self) -> None:
        for name in list(getattr(self.tape, "builders", {}) or {}):
            try:
                self._mark[name] = self.store.last_start(self.symbol, name)
            except Exception:                           # noqa: BLE001
                self._mark[name] = None

    def load_into_builders(self, limit: int = KEEP_PER_INTERVAL) -> int:
        """Put the stored history back. Returns candles restored.

        ONLY INTO A BUILDER WITH NOTHING OF ITS OWN. Merging a disk
        history into a live one needs the two to agree about where they
        overlap, and there is no need to find out what happens when
        they do not: a builder that has already closed bars this
        session is ahead of the disk by definition.

        The bar in progress is never restored. It was in progress when
        the process died, which means its open is real and everything
        after it is a guess.
        """
        total = 0
        builders = getattr(self.tape, "builders", {}) or {}
        for name, b in builders.items():
            if getattr(b, "closed", None):
                continue
            try:
                got = self.store.recent(self.symbol, name, limit)
            except Exception:                           # noqa: BLE001
                self.errors += 1
                continue
            if not got:
                continue
            keep = int(getattr(b, "max_closed", 0) or 0) or len(got)
            b.closed.extend(got[-keep:])
            # The tape's own mirror of the same bars, so a reader that
            # goes through `closed` rather than the builder sees the
            # restored history too.
            try:
                self.tape.closed.setdefault(name, None)
                dq = self.tape.closed.get(name)
                if dq is not None and not dq:
                    dq.extend(got[-keep:])
            except Exception:                           # noqa: BLE001
                pass
            total += len(got[-keep:])
        self.restored += total
        return total

    def flush(self) -> int:
        """Write every closed candle newer than the place marker."""
        n = 0
        builders = getattr(self.tape, "builders", {}) or {}
        for name, b in list(builders.items()):
            closed = list(getattr(b, "closed", []) or [])
            if not closed:
                continue
            mark = self._mark.get(name)
            fresh = (list(closed) if mark is None
                     else [c for c in closed if float(c.start_ts) > mark])
            if not fresh:
                continue
            try:
                n += self.store.record_many(self.symbol, name, fresh)
            except Exception:                           # noqa: BLE001
                self.errors += 1
                continue
            self._mark[name] = max(float(c.start_ts) for c in fresh)
        self.written += n
        self.last_flush = time.time()
        return n

    # -- the thread ------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="cme-candle-store")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=2.0)
        self._thread = None

    def _run(self) -> None:
        pruned_at = 0.0
        while not self._stop.wait(self.period_s):
            try:
                self.flush()
                now = time.time()
                if now - pruned_at > 3600:
                    self.store.prune(self.keep)
                    pruned_at = now
            except Exception:                           # noqa: BLE001
                self.errors += 1

    def status(self) -> dict:
        try:
            held = {n: self.store.count(self.symbol, n)
                    for n in (getattr(self.tape, "builders", {}) or {})}
        except Exception:                               # noqa: BLE001
            held = {}
        return {"symbol": self.symbol, "written": self.written,
                "restored": self.restored, "errors": self.errors,
                "last_flush": self.last_flush, "on_disk": held,
                "running": self._thread is not None}
