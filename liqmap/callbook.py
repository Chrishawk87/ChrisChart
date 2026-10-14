"""Every call the engine makes, written down before the answer exists.

WHY THIS IS THE MISSING PIECE

The ladder, the stack reader, the seven layers and the trajectory are all
built, all plausible, and none of them have ever been measured. Plausible
is not the same as right, and this project has already produced one engine
that scored 98.4% on a pure random walk.

So: record the call, wait, record what happened, and divide.

THE RECORD IS WRITTEN BEFORE THE OUTCOME EXISTS

`record` cannot write an outcome -- the columns are not in its statement,
and a test fails if they ever are. `resolve` fills them in afterwards, from
the bar's close. That separation is the whole guarantee against the failure
mode this file is built to catch: a scorer that sees the answer while it is
writing the question will report whatever its author hoped for.

A RATE WITHOUT A BASELINE IS NOT EVIDENCE

This is the part that matters most, and the part that is easiest to leave
out.

A bar that is already up at the halfway mark usually closes up. So an
engine that says "up" when price is up will score well above 50% while
contributing nothing whatsoever. Against a coin-flip baseline it looks like
an edge; against what it actually has to beat, it may be worth nothing.

So every call is stored beside the NAIVE call -- the direction price was
already going at the moment of the read -- and both are scored on the same
bars. Three numbers come out, and only the comparison between them means
anything:

    the engine's hit rate
    the naive hit rate on the same bars
    how often the two agreed in the first place

That third one is the quiet killer. An engine that agrees with the naive
read 95% of the time cannot have an edge over it, whatever either rate
says, because they are nearly the same call wearing different clothes.

AND HOW OFTEN IT DECLINES TO CALL AT ALL

The stack reader stands aside constantly -- no commit, pullback, unread.
A read that calls one bar in twenty and gets 80% of them is a different
proposition from one that calls everything and gets 55%, and a hit rate
printed without its coverage hides which of the two you have.

WHEN IN THE BAR

Three checkpoints: a quarter, a half and three quarters of the way
through. The same bar is called three times, scored three times, and
reported separately, because "does this read work" and "how early does it
work" are different questions and only one of them is answered by a single
number.

A read taken at 99% of the bar would score beautifully and be useless.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal, Sequence

from .db import ThreadedDB

Way = Literal["up", "down"]
Side = Literal["buy", "sell"]

# How far through the bar each call is taken. A read at 99% of the bar
# would score beautifully and tell you nothing you could trade on.
MARKS = (25, 50, 75)

# A bar closing within this many basis points of its open went nowhere.
# It is not a win for either side and it is not a loss for either side, so
# it is counted apart rather than handed to whoever guessed. Same reasoning
# as discarding mid prints in the aggressor audit: they carry no
# information about the thing being measured.
FLAT_BPS = 0.5

# Kept long enough to accumulate a sample worth reading, and no longer.
KEEP_S = 90 * 86400.0

SCHEMA = """
-- One row per (bar, kind of read, checkpoint).
--
-- The primary key is the call itself, so a page polling four times a
-- second overwrites its own row rather than stacking up ninety copies of
-- one opinion. That matters more here than anywhere else in this project:
-- a hit rate is a mean, and ninety rows for one bar would let a single
-- lucky fifteen minutes outvote a week.
--
-- The resolution columns are NULL until the bar closes. They are not in
-- the insert statement at all, which is what stops a call being written
-- with its own answer already in it.
CREATE TABLE IF NOT EXISTS calls (
    coin        TEXT NOT NULL,
    timeframe   TEXT NOT NULL,
    bar_ts      REAL NOT NULL,
    kind        TEXT NOT NULL,
    mark        INTEGER NOT NULL,
    made_at     REAL NOT NULL,
    side        TEXT NOT NULL DEFAULT '',
    detail      TEXT NOT NULL DEFAULT '',
    open_px     REAL NOT NULL DEFAULT 0,
    call_px     REAL NOT NULL DEFAULT 0,
    naive       TEXT NOT NULL DEFAULT '',
    resolved_at REAL,
    close_px    REAL,
    outcome     TEXT,
    PRIMARY KEY (coin, timeframe, bar_ts, kind, mark)
);
CREATE INDEX IF NOT EXISTS ix_calls_open
    ON calls (coin, timeframe, bar_ts) WHERE resolved_at IS NULL;
CREATE INDEX IF NOT EXISTS ix_calls_done
    ON calls (kind, mark) WHERE resolved_at IS NOT NULL;
"""


def _bps(a: float, b: float) -> float:
    return ((a - b) / b * 10_000.0) if b else 0.0


def way_of(px: float, open_px: float, flat_bps: float = FLAT_BPS) -> str:
    """Which way price sits against the bar's open: up, down, or flat."""
    d = _bps(float(px), float(open_px))
    if d > flat_bps:
        return "up"
    if d < -flat_bps:
        return "down"
    return "flat"


def mark_for(elapsed_s: float, interval_s: float) -> int | None:
    """The checkpoint this moment belongs to, or None if it is before the
    first one.

    The LATEST checkpoint already passed, so a read arriving at 60% of the
    bar counts as the halfway call. Awarding it to the next one up would
    file a read as earlier than it was, which flatters exactly the thing
    this file is trying to measure.
    """
    if interval_s <= 0:
        return None
    share = (float(elapsed_s) / float(interval_s)) * 100.0
    got = [m for m in MARKS if share >= m]
    return max(got) if got else None


@dataclass(frozen=True)
class Call:
    """One read, as it was at the time it was made."""

    coin: str
    timeframe: str
    bar_ts: float
    kind: str                  # cascade | trace | whatever is added next
    mark: int
    side: str = ""             # buy | sell | "" when it declined to call
    detail: str = ""
    open_px: float = 0.0
    call_px: float = 0.0

    @property
    def naive(self) -> str:
        """Where price was ALREADY going when the call was made.

        The thing the engine has to beat. Computed here rather than passed
        in, so a caller cannot supply a flattering one.
        """
        return way_of(self.call_px, self.open_px)

    @property
    def called(self) -> bool:
        return self.side in ("buy", "sell")


@dataclass
class Tally:
    """One kind of read, at one checkpoint, scored against its baseline."""

    kind: str = ""
    mark: int = 0
    offered: int = 0           # bars where a read existed at all
    called: int = 0            # ...and it committed to a side
    scored: int = 0            # ...and the bar then went somewhere
    hits: int = 0
    naive_hits: int = 0
    agreed: int = 0            # the engine said what the naive read said
    flat: int = 0              # bars that closed where they opened

    @property
    def rate(self) -> float | None:
        return (self.hits / self.scored) if self.scored else None

    @property
    def naive_rate(self) -> float | None:
        return (self.naive_hits / self.scored) if self.scored else None

    @property
    def edge(self) -> float | None:
        """The only number that means anything, and only with `scored`."""
        r, n = self.rate, self.naive_rate
        return None if r is None or n is None else r - n

    @property
    def coverage(self) -> float | None:
        """How often it committed to anything at all."""
        return (self.called / self.offered) if self.offered else None

    @property
    def agreement(self) -> float | None:
        """How often it said what price was already saying.

        THE QUIET KILLER. Near 1.0 and there can be no edge over the naive
        read whatever the rates say -- they are the same call in different
        clothes.
        """
        return (self.agreed / self.scored) if self.scored else None

    def add(self, side: str, naive: str, outcome: str) -> None:
        self.offered += 1
        if side not in ("buy", "sell"):
            return
        self.called += 1
        want: Way = "up" if side == "buy" else "down"
        if outcome == "flat":
            # Counted, so the coverage is honest, but not scored. A bar
            # that went nowhere is not a win for whoever guessed.
            self.flat += 1
            return
        self.scored += 1
        self.hits += int(outcome == want)
        self.naive_hits += int(naive == outcome)
        self.agreed += int(naive == want)

    def describe(self) -> str:
        if not self.scored:
            return (f"{self.kind} @{self.mark}%: nothing scored yet "
                    f"({self.offered} reads, {self.called} calls)")
        bits = [f"{self.rate:.0%} of {self.scored}",
                f"naive {self.naive_rate:.0%}"]
        e = self.edge
        bits.append(f"edge {e:+.0%}" if e is not None else "edge unknown")
        if self.agreement is not None and self.agreement >= 0.9:
            bits.append(f"but agrees with naive {self.agreement:.0%} of the "
                        f"time -- it is nearly the same call")
        if self.coverage is not None:
            bits.append(f"called {self.coverage:.0%} of bars")
        return f"{self.kind} @{self.mark}%: " + ", ".join(bits)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "mark": self.mark, "offered": self.offered,
                "called": self.called, "scored": self.scored,
                "hits": self.hits, "flat": self.flat,
                "rate": self.rate, "naive_rate": self.naive_rate,
                "edge": self.edge, "coverage": self.coverage,
                "agreement": self.agreement, "describe": self.describe()}


class CallBook(ThreadedDB):
    """The calls, and what became of them."""

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

    # -- writing the question ---------------------------------------------

    def record(self, call: Call, now: float | None = None) -> bool:
        """Write a call. The outcome columns are untouchable from here.

        They are not in this statement and they are not in the conflict
        clause, so a resolved row is never reopened and a new row is never
        born with its answer attached. That is the whole guarantee.
        """
        if call.mark not in MARKS or call.open_px <= 0:
            return False
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO calls (coin, timeframe, bar_ts, kind, mark,"
                " made_at, side, detail, open_px, call_px, naive)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(coin, timeframe, bar_ts, kind, mark) DO NOTHING",
                (call.coin, call.timeframe, float(call.bar_ts), call.kind,
                 int(call.mark), time.time() if now is None else float(now),
                 call.side, call.detail, float(call.open_px),
                 float(call.call_px), call.naive))
        return True

    # -- writing the answer, later ----------------------------------------

    def resolve(self, coin: str, timeframe: str, bar_ts: float,
                close_px: float, now: float | None = None) -> int:
        """Fill in what the bar did, for every unresolved call on it.

        Only rows that are still open. A resolved call is never rewritten,
        so a bar re-closed by a reconnecting feed cannot quietly change an
        answer that has already been counted.
        """
        close_px = float(close_px)
        if close_px <= 0:
            return 0
        stamp = time.time() if now is None else float(now)
        with self._tx() as conn:
            cur = conn.execute(
                "UPDATE calls SET resolved_at=?, close_px=?,"
                " outcome=CASE"
                "  WHEN (?-open_px)/open_px*10000.0 >  ? THEN 'up'"
                "  WHEN (?-open_px)/open_px*10000.0 < -? THEN 'down'"
                "  ELSE 'flat' END"
                " WHERE coin=? AND timeframe=? AND bar_ts=?"
                " AND resolved_at IS NULL AND open_px > 0",
                (stamp, close_px, close_px, FLAT_BPS, close_px, FLAT_BPS,
                 coin, timeframe, float(bar_ts)))
            return cur.rowcount or 0

    # -- reading it back ---------------------------------------------------

    def scorecard(self, coin: str = "", timeframe: str = "",
                  kind: str = "") -> list[Tally]:
        """Every resolved call, tallied by kind and checkpoint.

        NOTHING IS FILTERED OUT. A scorer that drops rows is
        indistinguishable from a better strategy, so the only arguments
        here narrow which market is being asked about -- never which
        results count.
        """
        where = ["resolved_at IS NOT NULL"]
        args: list = []
        for col, val in (("coin", coin), ("timeframe", timeframe),
                         ("kind", kind)):
            if val:
                where.append(f"{col}=?")
                args.append(val)
        rows = self._conn.execute(
            "SELECT kind, mark, side, naive, outcome FROM calls WHERE "
            + " AND ".join(where), args).fetchall()

        out: dict[tuple[str, int], Tally] = {}
        for r in rows:
            key = (r["kind"], int(r["mark"]))
            t = out.get(key)
            if t is None:
                t = out[key] = Tally(kind=r["kind"], mark=int(r["mark"]))
            t.add(r["side"] or "", r["naive"] or "", r["outcome"] or "")
        return [out[k] for k in sorted(out)]

    def open_bars(self, coin: str = "", limit: int = 200) -> list[tuple]:
        """Bars with calls still waiting on an outcome."""
        args: list = []
        where = ["resolved_at IS NULL"]
        if coin:
            where.append("coin=?")
            args.append(coin)
        rows = self._conn.execute(
            "SELECT DISTINCT coin, timeframe, bar_ts FROM calls WHERE "
            + " AND ".join(where) + " ORDER BY bar_ts DESC LIMIT ?",
            args + [int(limit)]).fetchall()
        return [(r["coin"], r["timeframe"], float(r["bar_ts"])) for r in rows]

    def counts(self) -> dict:
        r = self._conn.execute(
            "SELECT COUNT(*) AS n,"
            " SUM(CASE WHEN resolved_at IS NOT NULL THEN 1 ELSE 0 END) AS done"
            " FROM calls").fetchone()
        n = int(r["n"] or 0)
        done = int(r["done"] or 0)
        return {"calls": n, "resolved": done, "open": n - done}

    def prune(self, older_than_s: float = KEEP_S,
              now: float | None = None) -> int:
        cut = (time.time() if now is None else now) - older_than_s
        with self._tx() as conn:
            cur = conn.execute("DELETE FROM calls WHERE bar_ts < ?", (cut,))
            return cur.rowcount or 0


def summarise(tallies: Sequence[Tally]) -> str:
    """One line for the panel, naming the honest state of the evidence."""
    scored = sum(t.scored for t in tallies)
    if not scored:
        offered = sum(t.offered for t in tallies)
        return (f"nothing scored yet -- {offered} reads recorded, waiting "
                f"for their bars to close")
    best = max((t for t in tallies if t.scored),
               key=lambda t: (t.edge if t.edge is not None else -1))
    e = best.edge
    if e is None:
        return f"{scored} calls scored, no baseline to compare against"
    if scored < 100:
        return (f"{scored} calls scored -- too few to conclude anything. "
                f"Best so far: {best.describe()}")
    return f"best: {best.describe()}"
