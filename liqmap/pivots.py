"""Swing highs and lows, and the honesty about which ones are settled.

A pivot high is a bar whose high stands above the `left` bars before it
and the `right` bars after it. A pivot low is the mirror. That is the
whole definition; the care is all in the edges.

LEFT AND RIGHT DO DIFFERENT JOBS, so they are set separately.

`left` is how many bars must be cleared to establish that a bar is a
local extreme at all -- that is a statement about structure, and it
wants to be wide. `right` is only how long you are willing to wait
before believing it, and every bar of it is pure latency: at three
bars either side on target 250 a swing is marked about thirty-six
seconds after it happened, which is far too late to be looking at.

So the default is asymmetric -- wide left, narrow right -- and
`find_provisional` reports the turns in the unsettled region as well,
clearing whatever bars exist after them so far. The caller draws those
differently, because they can still be taken out.

A PIVOT IS NOT CONFIRMED UNTIL `right` BARS HAVE PASSED. The most
recent bars cannot be evaluated -- the next one may take out the high
and the pivot never existed. Drawing one there would put a level on the
chart that disappears a moment later, and a level that comes and goes
is worse than no level: you remember it was there and not that it was
provisional. So `find` returns settled pivots only, and reports how
many bars at the right edge are still pending.

AN ORDINAL EXTREME IS NOT A SWING, and this was the bug.

Standing above the neighbours by one tick and standing above them by
eight points were the same thing to this module: both were pivots,
both drew a level, both got a chevron. Measured on 26,000 bars, THE
MEDIAN PIVOT CLEARED ITS NEIGHBOURS BY ZERO TICKS -- more than half
of every mark on the chart was a bar that merely tied the bars
around it, and 59% cleared by less than a tick. Those are not weak
turns; they are not turns.

So a pivot now also has to stand clear by `min_clear`. Zero keeps
the old behaviour exactly, which is how the two can be compared
rather than argued about.

CLEARANCE IGNORES THE PLATEAU, which is the whole difficulty. A
genuine double top has two bars at the same high, and measuring
clearance against the nearest neighbour would score it zero and
throw away exactly the turn most worth keeping. So clearance is
measured against the best neighbour STRICTLY BELOW the pivot --
the plateau is part of the pivot, not competition for it.

PLATEAUS RESOLVE TO THEIR LAST BAR. Two bars sharing the same high are
ordinary on a tick grid, and a strict comparison on both sides would
find no pivot at all in that case -- quietly dropping some of the most
obvious turns on the chart. The comparison is therefore `>=` looking
left and `>` looking right, which marks exactly one pivot per plateau,
at its right-hand end.

Nothing here predicts. A pivot is a description of bars that have
already printed.
"""

from __future__ import annotations

from dataclasses import dataclass

HIGH = "high"
LOW = "low"


@dataclass(frozen=True, slots=True)
class Pivot:
    index: int = 0          # position in the series it was found in
    kind: str = HIGH
    price: float = 0.0
    ts: float = 0.0
    left: int = 0
    right: int = 0

    @property
    def depth(self) -> int:
        """How many bars on the shorter side had to clear it."""
        return min(self.left, self.right)

    def as_dict(self) -> dict:
        return {"index": self.index, "kind": self.kind,
                "price": self.price, "ts": self.ts,
                "left": self.left, "right": self.right}


def clearance(series: list[float], i: int, left: int, right: int,
              kind: str) -> float:
    """How far bar `i` stands clear of its neighbourhood.

    MEASURED AGAINST THE BEST NEIGHBOUR STRICTLY BELOW IT (above, for
    a low), so a plateau does not score itself to zero. Two bars at
    the same high are one turn, not two bars competing; measuring
    against the nearer of them would throw away the double top, which
    is the turn most worth keeping.

    Returns 0.0 when every neighbour ties the bar -- a perfectly flat
    stretch has no extreme in it, whatever the comparisons say.
    """
    px = series[i]
    lo_i = max(0, i - left)
    hi_i = min(len(series), i + right + 1)
    if kind == HIGH:
        others = [series[k] for k in range(lo_i, hi_i)
                  if k != i and series[k] < px]
        return (px - max(others)) if others else 0.0
    others = [series[k] for k in range(lo_i, hi_i)
              if k != i and series[k] > px]
    return (min(others) - px) if others else 0.0


def find(highs: list[float], lows: list[float],
         left: int = 5, right: int = 1,
         ts: list[float] | None = None,
         min_clear: float = 0.0) -> list[Pivot]:
    """Settled pivots in `highs`/`lows`, oldest first.

    `left` and `right` are how many bars either side must be cleared.
    Larger means fewer, more structural turns.

    `min_clear` is how far the bar must stand clear of its
    neighbourhood, in the same price units as the series. Zero is the
    old behaviour: any ordinal extreme counts, including one that
    merely ties everything around it.
    """
    if left < 1 or right < 1:
        raise ValueError("left and right must be at least 1")
    n = len(highs)
    if len(lows) != n:
        raise ValueError("highs and lows must be the same length")
    if ts is not None and len(ts) != n:
        raise ValueError("ts must match the series length")

    out: list[Pivot] = []
    # The scan stops `right` short of the end. Everything past that is
    # unsettled, and `pending()` is how a caller finds out.
    for i in range(left, n - right):
        h = highs[i]
        is_high = True
        for k in range(i - left, i):
            if highs[k] > h:
                is_high = False
                break
        if is_high:
            for k in range(i + 1, i + right + 1):
                if highs[k] >= h:
                    is_high = False
                    break
        if is_high and min_clear > 0:
            is_high = clearance(highs, i, left, right, HIGH) >= min_clear
        if is_high:
            out.append(Pivot(index=i, kind=HIGH, price=h,
                             ts=(ts[i] if ts else 0.0),
                             left=left, right=right))

        lo = lows[i]
        is_low = True
        for k in range(i - left, i):
            if lows[k] < lo:
                is_low = False
                break
        if is_low:
            for k in range(i + 1, i + right + 1):
                if lows[k] <= lo:
                    is_low = False
                    break
        if is_low and min_clear > 0:
            is_low = clearance(lows, i, left, right, LOW) >= min_clear
        if is_low:
            out.append(Pivot(index=i, kind=LOW, price=lo,
                             ts=(ts[i] if ts else 0.0),
                             left=left, right=right))

    out.sort(key=lambda p: p.index)
    return out


def find_provisional(highs: list[float], lows: list[float],
                     left: int = 5, right: int = 1,
                     ts: list[float] | None = None,
                     min_clear: float = 0.0) -> list[Pivot]:
    """Turns in the unsettled region, cleared only by the bars so far.

    These appear the moment a bar is a local extreme against everything
    to its left, without waiting out the confirmation window. They are
    the ones that show up on time -- and the ones that can still be
    taken out by the next bar, which is why they come back separately
    from `find` rather than mixed in with it.
    """
    if left < 1 or right < 1:
        raise ValueError("left and right must be at least 1")
    n = len(highs)
    if len(lows) != n:
        raise ValueError("highs and lows must be the same length")
    if ts is not None and len(ts) != n:
        raise ValueError("ts must match the series length")

    out: list[Pivot] = []
    start = max(left, n - right)
    for i in range(start, n):
        h, lo = highs[i], lows[i]
        is_high = all(highs[k] <= h for k in range(i - left, i))
        is_low = all(lows[k] >= lo for k in range(i - left, i))
        # Whatever bars DO exist to the right still have to be cleared.
        for k in range(i + 1, n):
            if highs[k] >= h:
                is_high = False
            if lows[k] <= lo:
                is_low = False
        # AND THE SAME CLEARANCE THE SETTLED ONES NEED. A provisional
        # mark held to a looser standard would seed the chart with
        # exactly the ties this exists to remove, and they would
        # simply look fainter while doing it.
        if min_clear > 0:
            if is_high and clearance(highs, i, left, n - i - 1,
                                     HIGH) < min_clear:
                is_high = False
            if is_low and clearance(lows, i, left, n - i - 1,
                                    LOW) < min_clear:
                is_low = False
        when = ts[i] if ts else 0.0
        if is_high:
            out.append(Pivot(index=i, kind=HIGH, price=h, ts=when,
                             left=left, right=n - 1 - i))
        if is_low:
            out.append(Pivot(index=i, kind=LOW, price=lo, ts=when,
                             left=left, right=n - 1 - i))
    out.sort(key=lambda p: p.index)
    return out


def pending(n: int, right: int = 1) -> int:
    """Bars at the right edge that cannot be judged yet."""
    return min(n, max(0, right))


def levels(pivots: list[Pivot], each: int = 2) -> list[Pivot]:
    """The most recent `each` highs and `each` lows.

    These are the ones worth extending across the chart as reference
    lines. Extending every pivot turns the chart into a grid and the
    levels stop meaning anything.
    """
    highs = [p for p in pivots if p.kind == HIGH][-each:]
    lows = [p for p in pivots if p.kind == LOW][-each:]
    return sorted(highs + lows, key=lambda p: p.index)
