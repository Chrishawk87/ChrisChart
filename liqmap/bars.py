"""OHLCV bars from Databento, for the auction pipeline.

WHY THIS IS NOT escondense

`escondense` exists to pair a book snapshot with the candle that followed
it, and it drops any row without resting depth -- correctly, because a row
with no book carries no read to grade. An `ohlcv-1s` file has no book at
all, so feeding one to that condenser produces zero rows after a twenty
minute parse and looks like a broken file rather than the wrong tool.

The auction pipeline needs nothing but timestamp, OHLC and volume. So this
is the loader for it: small, and about a hundred and fifty times cheaper to
buy than the order-book schema nothing in Steps 1-5 reads.

THE ROLL

A continuous front-month series (`ES.c.0`) stitches one contract to the
next at each quarterly expiry, and the two contracts trade at different
prices -- ES rolls are typically several points apart. That discontinuity is
synthetic. A profile spanning it is meaningless, and the prior session's
value area across a roll boundary points at prices in the old contract.

Six months of ES crosses two rolls. So `sessions()` flags any session whose
open gaps from the previous close by more than `ROLL_GAP_TICKS` and marks it
`suspect`, and the study drops those sessions rather than quietly building
events out of a bookkeeping artifact. That test cannot distinguish a roll
from a genuine news gap, and it does not try to -- both are sessions whose
relationship to the prior day's structure is broken.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Iterator, Sequence

from . import dbn
from .levels import Phase, phase, trade_date

# A gap beyond this is a data break, not a move.
#
# The first version used 40 ticks -- ten ES points -- on the reasoning that
# it sat above ordinary overnight drift. It does not. At ES 6000-7800 a ten
# point overnight move is routine, and the detector threw away 352 of 507
# sessions on consecutive weekdays: 69% of two years of data, none of it
# rolls. Two hundred ticks is fifty points, which on a session-to-session
# basis really is a discontinuity rather than a market.
ROLL_GAP_TICKS = 200.0

# Sessions within this many days of a quarterly expiry are dropped.
#
# Rolls are not inferred from price at all now. They happen on known dates --
# the third Friday of March, June, September and December -- and a
# continuous series stitches contracts around them. Knowing the date is
# strictly better than guessing from a gap, because the roll's price
# discontinuity can be small (making it invisible to a gap test) while its
# effect on the prior session's profile is total.
ROLL_WINDOW_DAYS = 4


def quarterly_expiry(year: int, month: int) -> date:
    """Third Friday of a month. ES expiries are Mar, Jun, Sep, Dec."""
    d = date(year, month, 1)
    fridays = 0
    while True:
        if d.weekday() == 4:
            fridays += 1
            if fridays == 3:
                return d
        d = date(d.year, d.month, d.day + 1)


def roll_dates(first: date, last: date) -> list[date]:
    """Every ES quarterly expiry in a span."""
    out: list[date] = []
    for y in range(first.year, last.year + 1):
        for m in (3, 6, 9, 12):
            e = quarterly_expiry(y, m)
            if first <= e <= last:
                out.append(e)
    return out


def near_roll(day: date, rolls: Sequence[date],
              window: int = ROLL_WINDOW_DAYS) -> bool:
    return any(abs((day - r).days) <= window for r in rolls)

TICK = 0.25


@dataclass(slots=True)
class Bar:
    """Exactly what the profile engine reads. Nothing else.

    Slotted because two years of five-second RTH bars is about 2.4 million
    of these. Without slots each carries a dict and the set runs to roughly
    700MB; with them it is a few hundred.
    """

    ts: float
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def range_ticks(self) -> float:
        return (self.high - self.low) / TICK

    @property
    def valid(self) -> bool:
        """Rejects the shapes that should never exist.

        A high below the low, or an open outside its own range, means the
        record was misread -- and a profile built from one would put volume
        at prices that never traded.
        """
        return (self.high >= self.low
                and self.low <= self.open <= self.high
                and self.low <= self.close <= self.high
                and self.low > 0 and self.volume >= 0)


@dataclass
class Session:
    """One RTH session's bars, and whether it can be trusted."""

    day: date
    bars: list[Bar] = field(default_factory=list)
    suspect: bool = False
    reason: str = ""

    @property
    def range_ticks(self) -> float:
        if not self.bars:
            return 0.0
        return (max(b.high for b in self.bars)
                - min(b.low for b in self.bars)) / TICK


def from_records(records: Iterable[Any]) -> Iterator[Bar]:
    """DBN OHLCV records to Bars, skipping anything malformed.

    Databento's OHLCV messages carry the same fixed-precision prices as the
    book schemas, so they go through the same scaling rather than a second
    copy of it.
    """
    for rec in records:
        ts = dbn.seconds(dbn._get(rec, "ts_event"))
        o = dbn.price(dbn._get(rec, "open"))
        h = dbn.price(dbn._get(rec, "high"))
        l = dbn.price(dbn._get(rec, "low"))
        c = dbn.price(dbn._get(rec, "close"))
        v = float(dbn._get(rec, "volume", 0) or 0)
        if ts <= 0 or min(o, h, l, c) <= 0:
            continue
        b = Bar(ts=ts, open=o, high=h, low=l, close=c, volume=v)
        if b.valid:
            yield b


def resample(bars: Sequence[Bar], seconds: float) -> list[Bar]:
    """Aggregate to a coarser grid, aligned to the interval.

    One-second bars are what is cheap to buy; five-second bars are what the
    profile was sized for. Aggregating is exact -- open of the first, high
    and low of the extremes, close of the last, volume summed -- unlike
    going the other way, which cannot be done at all.
    """
    if seconds <= 0 or not bars:
        return list(bars)
    out: list[Bar] = []
    cur: Bar | None = None
    end = 0.0
    for b in bars:
        start = b.ts - (b.ts % seconds)
        if cur is None or start >= end:
            if cur is not None:
                out.append(cur)
            cur = Bar(ts=start, open=b.open, high=b.high, low=b.low,
                      close=b.close, volume=b.volume)
            end = start + seconds
        else:
            cur.high = max(cur.high, b.high)
            cur.low = min(cur.low, b.low)
            cur.close = b.close
            cur.volume += b.volume
    if cur is not None:
        out.append(cur)
    return out


def sessions(bars: Sequence[Bar], rth_only: bool = True,
             roll_gap_ticks: float = ROLL_GAP_TICKS) -> list[Session]:
    """Group into CME trade dates, flagging discontinuities.

    A session is marked suspect when its first price gaps from the previous
    session's last by more than `roll_gap_ticks`. The prior session's value
    area is the input to every state decision, and across a roll it refers
    to a different contract -- so the study drops these rather than treating
    a bookkeeping artifact as a breakout.
    """
    grouped: dict[date, list[Bar]] = defaultdict(list)
    for b in bars:
        if rth_only and phase(b.ts) is not Phase.RTH:
            continue
        grouped[trade_date(b.ts)].append(b)

    days = sorted(grouped)
    rolls = roll_dates(days[0], days[-1]) if days else []

    out: list[Session] = []
    prev_close: float | None = None
    for day in days:
        rows = sorted(grouped[day], key=lambda x: x.ts)
        s = Session(day=day, bars=rows)

        # Known roll dates first. This is a calendar fact, not an inference
        # from price, and it catches rolls whose price discontinuity is too
        # small for any gap test to see.
        if near_roll(day, rolls):
            s.suspect = True
            s.reason = "within the quarterly roll window"
        elif prev_close is not None and rows:
            gap = abs(rows[0].open - prev_close) / TICK
            if gap > roll_gap_ticks:
                s.suspect = True
                s.reason = f"opened {gap:.0f} ticks from the prior close"

        if rows:
            prev_close = rows[-1].close
        out.append(s)
    return out


def resample_stream(records: Iterable[Bar], seconds: float) -> list[Bar]:
    """Aggregate an ITERABLE to a coarser grid, holding only the result.

    Identical arithmetic to `resample`, and the difference is memory
    rather than numbers. Two years of one-second ES is about twenty
    million records; materialising those as Python objects before
    aggregating costs something like 2.5GB and is pure waste when the
    five-second result is a fifth the size. The records go in a bar at a
    time and only the aggregate is kept.

    Requires ascending timestamps, which is how a DBN file arrives.
    """
    out: list[Bar] = []
    cur: Bar | None = None
    end = 0.0
    for b in records:
        if seconds <= 0:
            out.append(b)
            continue
        start = b.ts - (b.ts % seconds)
        if cur is None or start >= end:
            if cur is not None:
                out.append(cur)
            cur = Bar(ts=start, open=b.open, high=b.high, low=b.low,
                      close=b.close, volume=b.volume)
            end = start + seconds
        else:
            cur.high = max(cur.high, b.high)
            cur.low = min(cur.low, b.low)
            cur.close = b.close
            cur.volume += b.volume
    if cur is not None:
        out.append(cur)
    return out


def load(path: str, seconds: float = 5.0) -> list[Bar]:
    """Read a .dbn.zst OHLCV file and resample as it reads."""
    store = dbn.open_file(path)
    return resample_stream(from_records(store), seconds)
