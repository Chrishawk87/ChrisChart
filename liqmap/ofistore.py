"""Closed OFI bars on disk, so a restart does not erase the chart.

WHAT THIS FIXES. The builders hold their bars in memory. A Railway
redeploy, a crash, or an idle restart drops every one of them, and the
chart comes back empty with no indication that anything was lost. At
target 2,000 a bar is minutes wide, so that is days of structure gone
-- and the swing levels go with it, because a level is only as old as
the bars it was found in.

WHERE IT GOES. `liqmap.settings.default_db_path` resolves to the
Railway volume at /data when one is mounted. Without a volume this
writes to ephemeral disk and the redeploy wipes it anyway; that is not
this module's problem to detect, and `web.startup_report()` already
says so loudly.

THE WRITER DOES NOT RUN ON THE TAPE THREAD. The record loop is the
hot path -- five builders per book update at the rate the feed
arrives -- and putting a disk write in it would couple the tape's
throughput to the filesystem. Instead a flusher wakes on a timer,
asks each builder what has closed since it last looked, and writes
that batch. A bar is never written twice: `(target, ts_open,
ts_close)` is unique, and a re-insert of the same bar is ignored
rather than duplicated.

THE BARS ARE NOT RE-DERIVED ON RELOAD. What comes back is exactly
what was written, including the delta and aggression accumulators. A
bar rebuilt from a formula at load time would be a second, fallible
copy of something already stated, and any drift between the two would
be invisible -- the chart would stay plausible and be wrong.

PRICES STAY IN RAW FEED UNITS, as everywhere else in the OFI path.
The route converts to dollars once, on the way out.

NOTHING HERE PREDICTS, and nothing here decides anything. It is a
cupboard.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .db import ThreadedDB
from .ofibar import Episode, OfiBar

SCHEMA = """
CREATE TABLE IF NOT EXISTS ofi_bars (
    id          INTEGER PRIMARY KEY,
    target      INTEGER NOT NULL,
    ts_open     REAL    NOT NULL,
    ts_close    REAL    NOT NULL,
    open        REAL    NOT NULL,
    high        REAL    NOT NULL,
    low         REAL    NOT NULL,
    close       REAL    NOT NULL,
    direction   TEXT    NOT NULL,
    ofi         INTEGER NOT NULL,
    updates     INTEGER NOT NULL,
    ofi_high    INTEGER NOT NULL DEFAULT 0,
    ofi_low     INTEGER NOT NULL DEFAULT 0,
    buy_agg     REAL    NOT NULL DEFAULT 0,
    sell_agg    REAL    NOT NULL DEFAULT 0,
    buy_burst   REAL    NOT NULL DEFAULT 0,
    sell_burst  REAL    NOT NULL DEFAULT 0,
    buy_vol     REAL    NOT NULL DEFAULT 0,
    sell_vol    REAL    NOT NULL DEFAULT 0,
    trade_count INTEGER NOT NULL DEFAULT 0,
    trade_high  REAL    NOT NULL DEFAULT 0,
    trade_low   REAL    NOT NULL DEFAULT 0,
    trade_last  REAL    NOT NULL DEFAULT 0,
    call_side   TEXT    NOT NULL DEFAULT '',
    call_px     REAL    NOT NULL DEFAULT 0,
    call_ts     REAL    NOT NULL DEFAULT 0,
    push_up     INTEGER NOT NULL DEFAULT 0,
    push_dn     INTEGER NOT NULL DEFAULT 0,
    episodes    TEXT    NOT NULL DEFAULT '',
    written_at  REAL    NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ofi_bars_one
    ON ofi_bars(target, ts_open, ts_close);
CREATE INDEX IF NOT EXISTS ofi_bars_recent
    ON ofi_bars(target, ts_close);
"""

# How often the flusher wakes. Ten seconds is under one bar at the
# smallest target, so at worst a restart loses a handful of bars from
# the fastest chart and none from the slower ones.
FLUSH_S = 10.0

# Rows kept per target. Matches the builders' own in-memory cap, so a
# reload fills the chart to the same depth it had before.
KEEP_PER_TARGET = 20_000

_COLUMNS = ("target", "ts_open", "ts_close", "open", "high", "low",
            "close", "direction", "ofi", "updates", "ofi_high", "ofi_low",
            "buy_agg", "sell_agg", "buy_burst", "sell_burst",
            "buy_vol", "sell_vol", "trade_count",
            "trade_high", "trade_low", "trade_last",
            "call_side", "call_px", "call_ts",
            "push_up", "push_dn", "episodes", "written_at")

# Columns added after the table first shipped. A CREATE TABLE IF NOT
# EXISTS does nothing to a table that already exists, so a database
# written by an earlier build needs these put on by hand -- otherwise
# every insert fails with "no such column" and the chart quietly stops
# being saved, which looks exactly like it working.
_ADDED = (("trade_high", "REAL NOT NULL DEFAULT 0"),
          ("trade_low", "REAL NOT NULL DEFAULT 0"),
          ("trade_last", "REAL NOT NULL DEFAULT 0"),
          ("call_side", "TEXT NOT NULL DEFAULT ''"),
          ("call_px", "REAL NOT NULL DEFAULT 0"),
          ("call_ts", "REAL NOT NULL DEFAULT 0"),
          ("episodes", "TEXT NOT NULL DEFAULT ''"),
          ("push_up", "INTEGER NOT NULL DEFAULT 0"),
          ("push_dn", "INTEGER NOT NULL DEFAULT 0"))


def _row_of(target: int, b: OfiBar, now: float) -> tuple:
    return (int(target), float(b.ts_open), float(b.ts_close),
            float(b.open), float(b.high), float(b.low), float(b.close),
            str(b.direction), int(b.ofi), int(b.updates),
            int(b.ofi_high), int(b.ofi_low),
            float(b.buy_agg), float(b.sell_agg),
            float(b.buy_burst), float(b.sell_burst),
            float(b.buy_vol), float(b.sell_vol), int(b.trade_count),
            float(b.trade_high), float(b.trade_low), float(b.trade_last),
            str(b.call_side or ""), float(b.call_px), float(b.call_ts),
            int(b.push_up), int(b.push_dn), _eps_out(b), float(now))


def _eps_out(b: OfiBar) -> str:
    """The bar's observations as JSON, or empty for a bar that made none.

    JSON in a column rather than a table of its own: they are only
    ever read back with the bar they belong to, and nothing queries
    across them. A second table would buy a join and cost a migration.
    """
    eps = getattr(b, "episodes", None)
    if not eps:
        return ""
    try:
        return json.dumps([e.as_dict() for e in eps], separators=(",", ":"))
    except Exception:                                   # noqa: BLE001
        return ""


def _eps_in(raw) -> list:
    """Back into Episodes. A row that cannot be parsed yields none --
    an unreadable record is no record, and inventing one would be
    worse than a bar that made no observations."""
    if not raw:
        return []
    try:
        rows = json.loads(raw)
    except Exception:                                   # noqa: BLE001
        return []
    out = []
    for d in rows:
        try:
            first = {int(k): v for k, v in (d.get("first") or {}).items()}
            out.append(Episode(
                side=str(d.get("side", "")),
                level=int(d.get("level", 0) or 0),
                px0=float(d.get("px0", 0.0)),
                ts0=float(d.get("ts0", 0.0)),
                px_end=float(d.get("px_end", 0.0)),
                ts_end=float(d.get("ts_end", 0.0)),
                updates=int(d.get("updates", 0)),
                mfe=float(d.get("mfe", 0.0)), mae=float(d.get("mae", 0.0)),
                end=str(d.get("end", "")), first=first))
        except Exception:                               # noqa: BLE001
            continue
    return out


def _bar_of(r: sqlite3.Row) -> OfiBar:
    """A row back into a bar.

    Only the columns a bar cannot exist without are read directly.
    Everything the table grew later goes through `_opt`, so a row
    written by any earlier build loads with zeros in the fields it
    never had rather than raising -- and a zero there says exactly
    what it should: that bar recorded none.
    """
    return OfiBar(ts_open=r["ts_open"], ts_close=r["ts_close"],
                  open=r["open"], high=r["high"], low=r["low"],
                  close=r["close"], direction=r["direction"],
                  ofi=r["ofi"], updates=r["updates"],
                  ofi_high=int(_opt(r, "ofi_high")),
                  ofi_low=int(_opt(r, "ofi_low")),
                  buy_agg=_opt(r, "buy_agg"), sell_agg=_opt(r, "sell_agg"),
                  buy_burst=_opt(r, "buy_burst"),
                  sell_burst=_opt(r, "sell_burst"),
                  buy_vol=_opt(r, "buy_vol"), sell_vol=_opt(r, "sell_vol"),
                  trade_count=int(_opt(r, "trade_count")),
                  trade_high=_opt(r, "trade_high"),
                  trade_low=_opt(r, "trade_low"),
                  trade_last=_opt(r, "trade_last"),
                  call_side=_txt(r, "call_side"),
                  call_px=_opt(r, "call_px"),
                  call_ts=_opt(r, "call_ts"),
                  push_up=int(_opt(r, "push_up")),
                  push_dn=int(_opt(r, "push_dn")),
                  episodes=_eps_in(_txt(r, "episodes")))


def _txt(r: sqlite3.Row, name: str) -> str:
    """A text column an older row may not carry."""
    try:
        v = r[name]
    except (IndexError, KeyError):
        return ""
    return str(v) if v is not None else ""


def _opt(r: sqlite3.Row, name: str) -> float:
    """A column an older row may not carry. Zero means no print."""
    try:
        v = r[name]
    except (IndexError, KeyError):
        return 0.0
    return float(v) if v is not None else 0.0


class OfiBarStore(ThreadedDB):
    """Closed OFI bars, one table across every target."""

    def __init__(self, path: str | Path = "liqmap.db"):
        super().__init__(path, SCHEMA)

    def _bootstrap(self, conn: sqlite3.Connection) -> None:
        """Create the table, then add any column a later build grew.

        Bars written before the traded-price columns existed keep their
        rows and read back with zeros in them, which is the same thing
        a bar with no prints says. Nothing is rewritten: a closed bar
        is a fact, and inventing traded prices for one that never
        recorded any would be worse than having none.
        """
        super()._bootstrap(conn)
        have = {r[1] for r in
                conn.execute("PRAGMA table_info(ofi_bars)").fetchall()}
        for name, decl in _ADDED:
            if name not in have:
                conn.execute(
                    f"ALTER TABLE ofi_bars ADD COLUMN {name} {decl}")
        conn.commit()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # -- writing ---------------------------------------------------------

    def record_many(self, target: int, bars) -> int:
        """Write closed bars. Returns how many rows were new.

        A bar already on disk is IGNORED rather than updated. A closed
        bar is a fact that does not change, so a second write of one
        can only be a replay -- and quietly overwriting it would let a
        reload-then-reflush cycle rewrite history with whatever the
        current build happened to compute.
        """
        rows = []
        now = time.time()
        for b in bars:
            if b is None or b.updates <= 0:
                continue
            rows.append(_row_of(target, b, now))
        if not rows:
            return 0
        sql = ("INSERT OR IGNORE INTO ofi_bars (" + ",".join(_COLUMNS) +
               ") VALUES (" + ",".join("?" * len(_COLUMNS)) + ")")
        with self._tx() as conn:
            before = conn.total_changes
            conn.executemany(sql, rows)
            return conn.total_changes - before

    def record(self, target: int, bar: OfiBar) -> int:
        return self.record_many(target, [bar])

    def prune(self, keep: int = KEEP_PER_TARGET) -> int:
        """Trim each target to its most recent `keep` bars."""
        keep = max(1, int(keep))
        removed = 0
        with self._tx() as conn:
            for (t,) in conn.execute(
                    "SELECT DISTINCT target FROM ofi_bars").fetchall():
                cur = conn.execute(
                    "DELETE FROM ofi_bars WHERE target=? AND id NOT IN ("
                    " SELECT id FROM ofi_bars WHERE target=?"
                    " ORDER BY ts_close DESC LIMIT ?)",
                    (t, t, keep))
                removed += cur.rowcount or 0
        return removed

    # -- reading ---------------------------------------------------------

    def recent(self, target: int, limit: int = KEEP_PER_TARGET) -> list:
        """The last `limit` bars for one target, OLDEST FIRST.

        Oldest first because that is the order a builder's own list is
        in, and handing back the reverse would put the chart's history
        on backwards with every candle individually correct.
        """
        rows = self._conn.execute(
            "SELECT * FROM ofi_bars WHERE target=?"
            " ORDER BY ts_close DESC LIMIT ?",
            (int(target), int(max(1, limit)))).fetchall()
        return [_bar_of(r) for r in reversed(rows)]

    def last_ts(self, target: int) -> float:
        """Close time of the newest bar on disk, or 0.0 for none."""
        r = self._conn.execute(
            "SELECT MAX(ts_close) AS t FROM ofi_bars WHERE target=?",
            (int(target),)).fetchone()
        return float(r["t"]) if r and r["t"] is not None else 0.0

    def last_mark(self, target: int) -> tuple[float, float] | None:
        """Identity of the newest bar on disk: (ts_open, ts_close).

        An identity rather than a timestamp, because the recorder uses
        it to find its place in a live list. See `OfiBarRecorder.flush`
        for why a watermark alone is not enough.
        """
        r = self._conn.execute(
            "SELECT ts_open, ts_close FROM ofi_bars WHERE target=?"
            " ORDER BY ts_close DESC LIMIT 1", (int(target),)).fetchone()
        return (float(r["ts_open"]), float(r["ts_close"])) if r else None

    def count(self, target: int | None = None) -> int:
        if target is None:
            r = self._conn.execute(
                "SELECT COUNT(*) AS n FROM ofi_bars").fetchone()
        else:
            r = self._conn.execute(
                "SELECT COUNT(*) AS n FROM ofi_bars WHERE target=?",
                (int(target),)).fetchone()
        return int(r["n"]) if r else 0

    def targets(self) -> list[int]:
        rows = self._conn.execute(
            "SELECT DISTINCT target FROM ofi_bars ORDER BY target").fetchall()
        return [int(r["target"]) for r in rows]


class OfiBarRecorder:
    """Keeps a set of builders and the store in step.

    Off the tape thread on purpose: it wakes on a timer, takes what has
    closed since it last looked, and writes that batch. If the disk is
    slow, or missing, or the write raises, the tape does not notice.
    """

    def __init__(self, store: OfiBarStore, builders: dict,
                 period_s: float = FLUSH_S,
                 keep: int = KEEP_PER_TARGET) -> None:
        self.store = store
        self.builders = builders
        self.period_s = float(period_s)
        self.keep = int(keep)
        self.written = 0
        self.errors = 0
        self.last_flush = 0.0
        # The last bar written, per target, as (ts_open, ts_close).
        # Seeded from the store so a reload does not immediately
        # rewrite everything it just read back.
        self._mark: dict[int, tuple[float, float] | None] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def seed(self) -> None:
        """Take the place marker from what is already on disk."""
        for t in self.builders:
            try:
                self._mark[t] = self.store.last_mark(t)
            except Exception:                           # noqa: BLE001
                self._mark[t] = None

    def load_into_builders(self, limit: int = KEEP_PER_TARGET) -> int:
        """Fill each builder's history from disk. Returns bars restored.

        Only ever called on a builder with no bars of its own. Merging
        a disk history into a live one would need the two to agree
        about where they overlap, and there is no need to find out what
        happens when they do not.
        """
        total = 0
        for t, b in self.builders.items():
            if b.bars:
                continue
            try:
                got = self.store.recent(t, limit)
            except Exception:                           # noqa: BLE001
                self.errors += 1
                continue
            if got:
                b.bars.extend(got)
                total += len(got)
        return total

    def flush(self) -> int:
        """Write everything that has closed since the last call.

        THE PLACE MARKER IS A BAR, NOT A TIMESTAMP. A watermark of
        "newest close already written" is the obvious design and it
        fails silently the moment feed time stops increasing -- a
        replay from an archive, a session restarted from a file, a
        clock correction. Every bar after that reads as older than the
        watermark and is dropped forever, and the chart simply stops
        growing on disk with nothing to say why.

        So the marker is the identity of the last bar written, located
        in the live list each time. If it cannot be found -- trimmed
        off the front, or the feed restarted -- the whole list is
        offered instead. That costs one large batch, once, and
        `record_many` ignores the rows that are already there.
        """
        n = 0
        for t, b in self.builders.items():
            bars = b.bars
            if not bars:
                continue
            mark = self._mark.get(t)
            start = 0
            if mark is not None:
                start = None
                # Scanned from the END: in steady running the marker is
                # within a few bars of it, so this is a handful of
                # comparisons rather than a walk of twenty thousand.
                for i in range(len(bars) - 1, -1, -1):
                    if (bars[i].ts_open, bars[i].ts_close) == mark:
                        start = i + 1
                        break
                if start is None:
                    start = 0
            fresh = bars[start:]
            if not fresh:
                continue
            try:
                n += self.store.record_many(t, fresh)
            except Exception:                           # noqa: BLE001
                self.errors += 1
                continue
            self._mark[t] = (fresh[-1].ts_open, fresh[-1].ts_close)
        self.written += n
        self.last_flush = time.time()
        return n

    # -- the timer -------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="ofibar-recorder")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        pruned_at = 0.0
        while not self._stop.wait(self.period_s):
            try:
                self.flush()
                # Pruning is far more expensive than a flush and the cap
                # is twenty thousand rows a target, so once an hour is
                # ample. Doing it every flush would spend the service's
                # disk budget deleting nothing.
                now = time.time()
                if now - pruned_at > 3600:
                    self.store.prune(self.keep)
                    pruned_at = now
            except Exception:                           # noqa: BLE001
                self.errors += 1

    def status(self, target: int | None = None) -> dict:
        """Cheap enough to call on every dashboard poll.

        The row count is for ONE target, not all five. A dashboard
        polling once a second would otherwise run five counts a second
        for a number that changes slowly and is only ever read as
        "roughly how much is kept".
        """
        out = {"written": self.written, "errors": self.errors,
               "last_flush": self.last_flush,
               "running": bool(self._thread and self._thread.is_alive())}
        if target is not None:
            try:
                out["on_disk"] = self.store.count(target)
            except Exception:                           # noqa: BLE001
                out["on_disk"] = None
        return out
