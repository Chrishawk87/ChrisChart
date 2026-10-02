"""The 4H / 15m / 1m entry engine. Suggestions only.

WHAT THIS IS

Chris's spec: a 4-hour directional filter, a 15-minute colour flip, and a
1-minute breakout trigger. It marks where those three line up and what the
target and stop would be. It does not place, size or time anything, and it
does not claim the setup works.

THREE TIMEFRAMES, ONE SOURCE

The 15m and 4H series are RESAMPLED from the 1m bars rather than fetched
separately. Three feeds are three chances for the boundaries to disagree --
a 15m bar whose open is not the open of the 1m bar at its start is a bug
you find weeks later, in a signal that fired on a bar that never existed.
Aggregating is exact and alignment is then true by construction.

THE REPAINT, WHICH IS THE WHOLE DIFFICULTY

The trigger asks whether the CURRENT 15-minute candle is green. On a
1-minute close that candle is unfinished. It can be green now and close
red, and a chart that quietly forgets this shows you, in replay, signals
that were never on the screen at the time.

Two things keep that honest here:

  * the forming 15m bar is accumulated from the 1m bars as they arrive,
    never read off the resampled series. Reading the finished bar to decide
    something at minute three is look-ahead, and it is the same error that
    made a pure coin score 63.7% earlier in this project.

  * every signal carries a STATUS. It is born `provisional`. When its 15m
    bar closes it becomes `confirmed` or `withdrawn`, and a withdrawn one
    stays in the record rather than disappearing. How often provisional
    signals survive is a fact worth having, and you only get it by keeping
    the ones that did not.

THE 4H STATE HAS A BUFFER, AND THE SPEC DID NOT

Price against the 4H open with nothing else flips on every tick while
price is near that open -- which is most of the first hour of every block.
As written the filter whipsaws between BULLISH and BEARISH and means
nothing. So the state here requires price to be a few ticks clear of the
open and to STAY there for a couple of closes, and it starts each block
UNDECIDED rather than guessing. No signal fires while it is undecided.

THE BAR INDEX IS DERIVED, NOT COUNTED

A 4H block is exactly sixteen 15m bars, so both boundaries land on the same
timestamp every single block. An incrementing counter that is reset by one
branch and advanced by another is then order-dependent, and the order is
not stated anywhere. Here the index is arithmetic on the timestamp, which
cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from . import bars as bars_mod
from . import profile as pf

# ---------------------------------------------------------------------------
# FIXED CONVENTIONS. Stated, not tuned.
# ---------------------------------------------------------------------------

H4_SECONDS = 14400.0
H1_SECONDS = 3600.0
M15_SECONDS = 900.0

# How far past the block's own open price must be before that block is
# called, and how many 1m closes it must hold there. Applied to BOTH the 4H
# and the 1H. Without the pair of them the state flips on noise, which is
# most of what price does near an open it has just left.
STATE_BUFFER_TICKS = 4.0
STATE_HOLD_BARS = 2

# 15m bars at the top of a 4H block during which no signal may fire. The
# spec's "opening filter": one bar.
SKIP_15M_BARS = 1

# Target, in ticks. From the spec.
TARGET_TICKS = 10.0

# The structural stop is the extreme of the forming 15m bar and this many
# CLOSED 15m bars before it. The spec said "Structural_15M_Low" without
# saying which low, and that number is the entire risk -- so it is pinned
# here rather than left to the reader.
STOP_LOOKBACK_15M = 2

# Smallest distance from the trigger back to the 15m open that still
# describes a trade. Below this, price is effectively AT the open and the
# premise has gone before the trigger printed.
MIN_STOP_TICKS_FLOOR = 1.0

# A round trip's cost in ticks. ES: $17.50 against $12.50 a tick. Only
# meaningful on a tick-denominated instrument; pass your own elsewhere.
COST_TICKS = 1.4

# Minimum distance from trigger to stop. A stop a tick away is not a
# structural stop, it is a rounding error, and the breakeven rate it
# implies is a fantasy.
MIN_STOP_TICKS = MIN_STOP_TICKS_FLOOR

BULLISH = "BULLISH"
BEARISH = "BEARISH"
UNDECIDED = "UNDECIDED"

PROVISIONAL = "provisional"
CONFIRMED = "confirmed"
WITHDRAWN = "withdrawn"


@dataclass
class Signal:
    """One marked setup. A suggestion for a human, never an instruction."""

    ts: float
    side: str                    # long | short
    trigger: float               # the 1m close that fired it
    target: float
    stop: float

    status: str = PROVISIONAL
    m15_start: float = 0.0
    m15_open: float = 0.0        # the stop, and the trade's premise
    minutes_in: float = 0.0      # how far into the 15m candle it fired
    expires: float = 0.0         # no trade outlives its own 15m candle
    h4_open: float = 0.0
    h1_open: float = 0.0
    bar_15_index: int = 0

    target_ticks: float = 0.0
    stop_ticks: float = 0.0
    driftless: float = 0.0       # P(target before stop) for a random walk
    breakeven: float = 0.0       # the same, with cost added
    note: str = ""

    def to_dict(self) -> dict:
        return {"ts": self.ts, "side": self.side,
                "trigger": round(self.trigger, 6),
                "target": round(self.target, 6),
                "stop": round(self.stop, 6),
                "status": self.status, "m15_start": self.m15_start,
                "m15_open": round(self.m15_open, 6),
                "minutes_in": round(self.minutes_in, 1),
                "expires": self.expires,
                "h4_open": round(self.h4_open, 6),
                "h1_open": round(self.h1_open, 6),
                "bar_15_index": self.bar_15_index,
                "target_ticks": round(self.target_ticks, 1),
                "stop_ticks": round(self.stop_ticks, 1),
                "driftless": round(self.driftless, 4),
                "breakeven": round(self.breakeven, 4),
                "note": self.note}


def _norm(bars: Sequence) -> list:
    """Sorted, de-duplicated, and nothing with a broken timestamp."""
    out = [b for b in bars if float(getattr(b, "ts", 0) or 0) > 0]
    out.sort(key=lambda b: float(b.ts))
    seen: list = []
    last = None
    for b in out:
        if last is not None and float(b.ts) == last:
            seen[-1] = b
            continue
        seen.append(b)
        last = float(b.ts)
    return seen


def rates(target_ticks: float, stop_ticks: float,
          cost_ticks: float = COST_TICKS) -> tuple[float, float]:
    """(driftless hit rate, break-even hit rate) for this barrier pair.

    For a driftless walk, P(+T before -S) = S / (T + S) -- gambler's ruin,
    and it depends only on the distances. Adding a round trip's cost moves
    the bar to (S + C) / (T + S).

    A tight target against a wide stop therefore wins most of the time by
    construction, which is why a high hit rate on this shape means nothing
    on its own. The two numbers are reported together so the gap is visible
    rather than inferred.
    """
    total = target_ticks + stop_ticks
    if total <= 0:
        return 0.0, 0.0
    return stop_ticks / total, min(1.0, (stop_ticks + cost_ticks) / total)


class _Forming:
    """The 15m bar being built, from the 1m bars seen so far.

    Never the resampled one. The resampled bar knows how it ends.
    """

    __slots__ = ("start", "open", "high", "low", "close")

    def __init__(self, start: float, bar) -> None:
        self.start = start
        self.open = float(bar.open)
        self.high = float(bar.high)
        self.low = float(bar.low)
        self.close = float(bar.close)

    def add(self, bar) -> None:
        self.high = max(self.high, float(bar.high))
        self.low = min(self.low, float(bar.low))
        self.close = float(bar.close)

    @property
    def green(self) -> bool:
        return self.close > self.open

    @property
    def red(self) -> bool:
        return self.close < self.open


def scan(bars: Sequence, tick: float | None = None,
         cost_ticks: float = COST_TICKS,
         target_ticks: float = TARGET_TICKS,
         max_minutes_in: float | None = None,
         require_1h: bool = False,
         block_s: float = H4_SECONDS,
         trade_s: float = M15_SECONDS,
         trigger_s: float = 60.0) -> list[Signal]:
    """Every setup in these 1-minute bars, oldest first.

    `require_1h` adds the hour to the agreement. The rule as finally
    stated is that the 4H and the 15m agree and the minute is trending
    with them; the hour was an earlier version of it. It is kept as a
    switch rather than deleted, because which of the two is better is a
    question with an answer and guessing at it is not.

    `max_minutes_in` refuses a trigger that prints more than that many
    minutes into its own 15-minute candle. The stop is the candle's open,
    so a late trigger is a long way from it -- on a random walk the whole
    population averages twenty ticks of stop, which is not the trade. A
    tighter window is a tighter stop by construction, and it also throws
    away signals, so it is a parameter to be SEARCHED on one half of the
    data and judged on the other, never chosen by looking at the answer.

    THE THREE TIMEFRAMES ARE PARAMETERS

    `block_s` sets the direction, `trade_s` is the candle being traded --
    it supplies the stop, which is its open, and the clock, which is its
    close -- and `trigger_s` is the bar the continuation is read on. The
    defaults are the 4H / 15m / 1m the rule was written in.

    They are parameters because the choice has an answer and guessing at
    it is not the same as knowing. Be careful what the answer costs,
    though: a coarser trigger enters further from the traded candle's
    open, so it widens the stop, and a smaller target makes the fixed
    cost a larger share of what is being won. Both raise the hit rate
    required. Neither is visible in a win rate on its own.

    Each signal is decided using only bars at or before its own timestamp.
    Status is resolved afterwards, from the close of its 15m bar, which is
    a later fact about an earlier signal -- not an input to it.
    """
    rows = _norm(bars)
    # The whole read happens on the TRIGGER grid. Aggregating first means
    # the forming trade candle, the block opens and the continuation bar
    # all come off one series and cannot disagree about a boundary.
    if trigger_s > 0:
        rows = bars_mod.resample_stream(iter(rows), trigger_s)
    if len(rows) < 4:
        return []
    if tick is None:
        tick = pf.infer_tick(rows)
    if tick <= 0:
        return []

    out: list[Signal] = []

    # Two independent block states, tracked the same way. The 15m only
    # counts when BOTH agree: the 4H says which way the day is leaning and
    # the 1H says the hour has not turned against it.
    h4_start = h1_start = -1.0
    h4_open = h1_open = 0.0
    state4 = state1 = UNDECIDED
    up4 = dn4 = up1 = dn1 = 0

    cur15: _Forming | None = None
    closed15: list[_Forming] = []       # most recent last
    fired: set[tuple[float, str]] = set()

    for i, b in enumerate(rows):
        ts = float(b.ts)

        # -- 4H block. The open is the open of the first 1m bar in it, so
        #    it is known the moment the block starts and reading it is not
        #    look-ahead. Nothing else about the 4H bar is touched.
        start4 = ts - (ts % block_s)
        if start4 != h4_start:
            h4_start = start4
            h4_open = float(b.open)
            state4 = UNDECIDED
            up4 = dn4 = 0

        # The hour, read exactly the same way. A new hour starts undecided
        # too -- carrying the last hour's lean into a fresh one is reading
        # a filter about a period that has ended.
        start1 = ts - (ts % H1_SECONDS)
        if start1 != h1_start:
            h1_start = start1
            h1_open = float(b.open)
            state1 = UNDECIDED
            up1 = dn1 = 0

        # -- 15m bar, accumulated rather than looked up.
        start15 = ts - (ts % trade_s)
        if cur15 is None or start15 != cur15.start:
            if cur15 is not None:
                closed15.append(cur15)
                if len(closed15) > STOP_LOOKBACK_15M + 2:
                    del closed15[:-(STOP_LOOKBACK_15M + 2)]
            cur15 = _Forming(start15, b)
        else:
            cur15.add(b)

        # -- both states, each with a buffer and a hold.
        px_now = float(b.close)

        d4 = (px_now - h4_open) / tick
        if d4 >= STATE_BUFFER_TICKS:
            up4, dn4 = up4 + 1, 0
        elif d4 <= -STATE_BUFFER_TICKS:
            dn4, up4 = dn4 + 1, 0
        else:
            up4 = dn4 = 0
        if up4 >= STATE_HOLD_BARS:
            state4 = BULLISH
        elif dn4 >= STATE_HOLD_BARS:
            state4 = BEARISH

        d1 = (px_now - h1_open) / tick
        if d1 >= STATE_BUFFER_TICKS:
            up1, dn1 = up1 + 1, 0
        elif d1 <= -STATE_BUFFER_TICKS:
            dn1, up1 = dn1 + 1, 0
        else:
            up1 = dn1 = 0
        if up1 >= STATE_HOLD_BARS:
            state1 = BULLISH
        elif dn1 >= STATE_HOLD_BARS:
            state1 = BEARISH

        # THE AGREEMENT. The four hours set the direction; the fifteen
        # has to lean the same way (checked below, where the candle is
        # known); the minute has to be trending with them.
        #
        # The hour is OPTIONAL. It was in an earlier statement of the rule
        # and not in the final one, and rather than pick, both are
        # measurable.
        if i == 0 or state4 == UNDECIDED:
            continue
        if require_1h and state1 != state4:
            continue
        state = state4

        idx15 = int((ts - h4_start) // trade_s)
        if idx15 < SKIP_15M_BARS:
            continue

        minutes_in = (ts - cur15.start) / 60.0
        if max_minutes_in is not None and minutes_in > max_minutes_in:
            continue
        if not closed15:
            continue

        prior15 = closed15[-1]
        upto15 = [b for b in rows[max(0, i - 15):i + 1]
                  if float(b.ts) >= cur15.start]
        prev1 = rows[i - 1]
        px = float(b.close)

        # CONTINUATION, not a breakout print. The bar must make a higher
        # high AND a higher low than the one before it -- structure moving
        # up, rather than one spike through a prior extreme that closes
        # back inside the last bar's range.
        up = (float(b.high) > float(prev1.high)
              and float(b.low) > float(prev1.low))
        down = (float(b.low) < float(prev1.low)
                and float(b.high) < float(prev1.high))

        side = None
        # ALL FOUR STATES THE SAME.
        #
        # The earlier rule wanted the PREVIOUS 15m to be the opposite
        # colour -- a reversal condition, and the wrong shape entirely.
        # What is being traded is agreement: the four hours, the hour and
        # the fifteen all leaning the same way, with the minute
        # continuing. The prior bar's colour is not part of that.
        m15_bull = cur15.green
        m15_bear = cur15.red
        if state == BULLISH and m15_bull and up:
            side = "long"
        elif state == BEARISH and m15_bear and down:
            side = "short"
        if side is None:
            continue

        # One per 15m bar per side. There is no position to be flat of --
        # this tool suggests and never executes -- so the rearm is the bar
        # rather than a fill.
        key = (cur15.start, side)
        if key in fired:
            continue

        # THE STOP IS THE 15m CANDLE'S OWN OPEN.
        #
        # Not a structural low two bars back. The trade's premise is that
        # this candle is leaning with everything above it; price trading
        # back through the candle's open says the premise is gone, and
        # there is nothing left to wait for.
        #
        # It is also what makes the arithmetic workable. A structural low
        # sat 27 ticks away and demanded a 76.6% hit rate against a ten
        # tick target. This stop is the distance from the entry back to
        # the open -- usually single digits -- which is a different trade.
        stop = cur15.open
        if side == "long":
            target = px + target_ticks * tick
            stop_t = (px - stop) / tick
        else:
            target = px - target_ticks * tick
            stop_t = (stop - px) / tick

        # Price is already back at or through the open, so the premise has
        # gone before the trigger even printed.
        if stop_t < MIN_STOP_TICKS:
            continue

        fired.add(key)
        drift, be = rates(target_ticks, stop_t, cost_ticks)
        out.append(Signal(
            ts=ts, side=side, trigger=px, target=target, stop=stop,
            m15_start=cur15.start, m15_open=cur15.open,
            minutes_in=minutes_in,
            expires=cur15.start + trade_s,
            h4_open=h4_open, h1_open=h1_open, bar_15_index=idx15,
            target_ticks=target_ticks, stop_ticks=stop_t,
            driftless=drift, breakeven=be,
            note=(f"4H and 15m {state.lower()}"
                  f"{' with the 1H' if require_1h else ''} "
                  f"({h4_open:,.2f} / {cur15.open:,.2f}); "
                  f"bar {idx15} of the block, {minutes_in:.0f} min in; 1m made "
                  f"a {'higher high and higher low' if side == 'long' else 'lower low and lower high'}"
                  f"; out at the 15m open or its close, whichever first")))

    _resolve(out, rows, trade_s)
    return out


def _resolve(signals: list[Signal], rows: list,
             trade_s: float = M15_SECONDS) -> None:
    """Mark each signal confirmed or withdrawn once its 15m bar has closed.

    This is a LATER fact about an EARLIER signal. It never feeds back into
    whether the signal fired -- `scan` has already finished deciding that
    by the time this runs, which is why it is a separate pass rather than
    a branch inside the loop.
    """
    if not signals or not rows:
        return
    final = {b.ts: b for b in bars_mod.resample(rows, trade_s)}
    last_ts = float(rows[-1].ts)
    for s in signals:
        bar = final.get(s.m15_start)
        if bar is None or last_ts < s.m15_start + trade_s - 1e-9:
            s.status = PROVISIONAL        # still forming
            continue
        green = float(bar.close) > float(bar.open)
        held = green if s.side == "long" else (float(bar.close)
                                               < float(bar.open))
        s.status = CONFIRMED if held else WITHDRAWN


def survival(signals: Sequence[Signal]) -> dict:
    """How often the provisional signals survived their own 15m close.

    The number that says whether the provisional marker is worth watching.
    It is NOT a win rate -- nothing here follows a signal to its target or
    its stop, and this says nothing about whether the setup makes money.
    """
    done = [s for s in signals if s.status in (CONFIRMED, WITHDRAWN)]
    kept = [s for s in done if s.status == CONFIRMED]
    return {"signals": len(signals), "resolved": len(done),
            "confirmed": len(kept),
            "still_forming": len(signals) - len(done),
            "survival": round(len(kept) / len(done), 4) if done else None}
