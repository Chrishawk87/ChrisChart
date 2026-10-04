"""Every bar's buy/sell split, kept after the bar closes.

WHY THIS EXISTS

CVD on a timeframe is the sum of that timeframe's bar deltas. The deltas
only exist in the live tape, because an OHLCV bar does not record who
crossed the spread -- so a CVD for any timeframe longer than the tape's
memory simply cannot be built from the tape.

The rolling tape holds about an hour. A one-minute rung can be built from
it sixty times over; a four-hour rung has seen a quarter of one bar. That
is why every slow row on the ladder read NOT COVERED and stayed there.

The fix is not a bigger tape. Keeping four hours of raw fills to extract
one number from them is the wrong trade, and it still starts from nothing
after every restart. What is worth keeping is the ONE ROW PER BAR that the
fills reduce to: buy, sell, trades, volume. Each row is tiny, the feed
already produces exactly this when a bar closes, and it survives restarts
-- so the four-hour CVD fills in over a day of uptime instead of never.

WHAT IS NOT STORED, AND WHY IT MATTERS MORE THAN WHAT IS

A bar adopted from the exchange at connect -- the bar that was already in
progress, and the history fetched behind it -- has no aggressor data at
all. Its buy and sell are zero, and a zero delta looks exactly like a
balanced one. Those bars are NOT written, and every read reports how many
bars it actually found, so a CVD built from two bars is never mistaken for
one built from six.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import sqlite3

from .db import ThreadedDB

SCHEMA = """
-- One row per closed bar per timeframe per market.
--
-- The primary key is the bar itself, so a feed that reconnects and
-- re-closes the same bar overwrites rather than double-counting it. That
-- matters: CVD is a sum, and a duplicated bar is indistinguishable from a
-- real one twice the size.
CREATE TABLE IF NOT EXISTS bar_flow (
    coin        TEXT NOT NULL,
    timeframe   TEXT NOT NULL,
    bar_ts      REAL NOT NULL,
    buy         REAL NOT NULL,
    sell        REAL NOT NULL,
    trades      INTEGER NOT NULL DEFAULT 0,
    volume      REAL NOT NULL DEFAULT 0,
    open_px     REAL NOT NULL DEFAULT 0,
    close_px    REAL NOT NULL DEFAULT 0,
    written_at  REAL NOT NULL,
    PRIMARY KEY (coin, timeframe, bar_ts)
);
CREATE INDEX IF NOT EXISTS ix_bar_flow_recent
    ON bar_flow (coin, timeframe, bar_ts DESC);
"""

# Bars older than this are dropped. Six four-hour bars is a day, so a week
# covers every timeframe on the ladder with room to spare, and the table
# stays small enough to never think about.
KEEP_S = 7 * 86400.0


@dataclass(frozen=True)
class BarFlow:
    """One closed bar's aggression, as it was counted."""

    coin: str
    timeframe: str
    bar_ts: float
    buy: float
    sell: float
    trades: int = 0
    volume: float = 0.0
    open_px: float = 0.0
    close_px: float = 0.0

    @property
    def delta(self) -> float:
        return self.buy - self.sell

    @property
    def total(self) -> float:
        return self.buy + self.sell

    def to_dict(self) -> dict:
        return {"coin": self.coin, "timeframe": self.timeframe,
                "bar_ts": self.bar_ts, "buy": round(self.buy, 2),
                "sell": round(self.sell, 2), "delta": round(self.delta, 2),
                "trades": self.trades, "volume": round(self.volume, 6),
                "open_px": self.open_px, "close_px": self.close_px}


@dataclass
class CVD:
    """A timeframe's cumulative delta, and how much of it is real."""

    timeframe: str
    cvd: float = 0.0
    bars: int = 0             # completed bars that contributed
    asked: int = 0            # completed bars that were wanted
    oldest_ts: float = 0.0
    live_delta: float = 0.0   # the bar in progress, counted separately
    source: str = "none"      # stored | memory | both | none

    @property
    def complete(self) -> bool:
        """Did every bar asked for turn up."""
        return self.asked > 0 and self.bars >= self.asked

    @property
    def total(self) -> float:
        return self.cvd + self.live_delta

    def to_dict(self) -> dict:
        return {"timeframe": self.timeframe, "cvd": round(self.cvd, 2),
                "bars": self.bars, "asked": self.asked,
                "oldest_ts": self.oldest_ts,
                "live_delta": round(self.live_delta, 2),
                "total": round(self.total, 2),
                "complete": self.complete, "source": self.source}


class BarFlowStore(ThreadedDB):
    """Closed-bar aggression, written once and read back as CVD."""

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

    # -- writing ----------------------------------------------------------

    def record(self, flow: BarFlow) -> bool:
        """Write one closed bar. Returns False if it was not worth keeping.

        A bar with no aggression counted in it is REFUSED rather than
        written as zero. The two are not the same thing, and once a zero
        is in the table nothing downstream can tell them apart.
        """
        if flow.total <= 0 and flow.trades <= 0:
            return False
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO bar_flow (coin, timeframe, bar_ts, buy, sell,"
                " trades, volume, open_px, close_px, written_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(coin, timeframe, bar_ts) DO UPDATE SET"
                " buy=excluded.buy, sell=excluded.sell,"
                " trades=excluded.trades, volume=excluded.volume,"
                " open_px=excluded.open_px, close_px=excluded.close_px,"
                " written_at=excluded.written_at",
                (flow.coin, flow.timeframe, float(flow.bar_ts),
                 float(flow.buy), float(flow.sell), int(flow.trades),
                 float(flow.volume), float(flow.open_px),
                 float(flow.close_px), time.time()))
        return True

    def prune(self, older_than_s: float = KEEP_S,
              now: float | None = None) -> int:
        cut = (time.time() if now is None else now) - older_than_s
        with self._tx() as conn:
            cur = conn.execute("DELETE FROM bar_flow WHERE bar_ts < ?", (cut,))
            return cur.rowcount or 0

    # -- reading ----------------------------------------------------------

    def recent(self, coin: str, timeframe: str, bars: int = 6,
               before_ts: float | None = None) -> list[BarFlow]:
        """The last `bars` closed bars, oldest first.

        `before_ts` excludes the bar in progress: pass its open and nothing
        from it can leak into a figure described as completed bars.
        """
        sql = ("SELECT * FROM bar_flow WHERE coin=? AND timeframe=?")
        args: list = [coin, timeframe]
        if before_ts is not None:
            sql += " AND bar_ts < ?"
            args.append(float(before_ts))
        sql += " ORDER BY bar_ts DESC LIMIT ?"
        args.append(int(max(1, bars)))
        rows = self._conn.execute(sql, args).fetchall()
        out = [BarFlow(coin=r["coin"], timeframe=r["timeframe"],
                       bar_ts=r["bar_ts"], buy=r["buy"], sell=r["sell"],
                       trades=r["trades"], volume=r["volume"],
                       open_px=r["open_px"], close_px=r["close_px"])
               for r in rows]
        out.reverse()
        return out

    def cvd(self, coin: str, timeframe: str, bars: int = 6,
            before_ts: float | None = None,
            live_delta: float = 0.0) -> CVD:
        rows = self.recent(coin, timeframe, bars=bars, before_ts=before_ts)
        return summarise(timeframe, rows, asked=bars, live_delta=live_delta,
                         source="stored" if rows else "none")


def summarise(timeframe: str, rows: Sequence[BarFlow], asked: int,
              live_delta: float = 0.0, source: str = "stored") -> CVD:
    """Add up what turned up, and say how much of it there was.

    The bar count is not decoration. Six bars of CVD and two bars of CVD
    are different claims, and the ladder draws them the same width.
    """
    out = CVD(timeframe=timeframe, asked=int(max(0, asked)),
              live_delta=float(live_delta))
    for r in rows:
        out.cvd += r.delta
        out.bars += 1
    out.oldest_ts = rows[0].bar_ts if rows else 0.0
    out.source = source if rows else "none"
    return out


def merge(timeframe: str, stored: Sequence[BarFlow],
          memory: Sequence[BarFlow], bars: int,
          live_delta: float = 0.0) -> CVD:
    """Combine what the database kept with what this feed has seen.

    The FEED WINS on any bar both of them have. The in-memory bar is the
    one this process counted fill by fill; the stored row may be from an
    older run that saw a different slice of the same bar, and two partial
    counts of one bar do not add up to that bar.
    """
    by_ts: dict[float, BarFlow] = {float(r.bar_ts): r for r in stored}
    had_stored = bool(by_ts)
    for r in memory:
        by_ts[float(r.bar_ts)] = r
    rows = [by_ts[k] for k in sorted(by_ts)][-max(1, bars):]
    src = ("both" if had_stored and memory else
           "memory" if memory else "stored" if had_stored else "none")
    return summarise(timeframe, rows, asked=bars, live_delta=live_delta,
                     source=src)
