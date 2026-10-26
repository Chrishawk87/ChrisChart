"""Swing highs and lows, and the honesty about which ones are settled.

A pivot high is a bar whose high stands above the `left` bars before it
and the `right` bars after it. A pivot low is the mirror. That is the
whole definition; the care is all in the edges.

A PIVOT IS NOT CONFIRMED UNTIL `right` BARS HAVE PASSED. The most
recent bars cannot be evaluated -- the next one may take out the high
and the pivot never existed. Drawing one there would put a level on the
chart that disappears a moment later, and a level that comes and goes
is worse than no level: you remember it was there and not that it was
provisional. So `find` returns settled pivots only, and reports how
many bars at the right edge are still pending.

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


def find(highs: list[float], lows: list[float],
         left: int = 3, right: int = 3,
         ts: list[float] | None = None) -> list[Pivot]:
    """Settled pivots in `highs`/`lows`, oldest first.

    `left` and `right` are how many bars either side must be cleared.
    Larger means fewer, more structural turns.
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
        if is_low:
            out.append(Pivot(index=i, kind=LOW, price=lo,
                             ts=(ts[i] if ts else 0.0),
                             left=left, right=right))

    out.sort(key=lambda p: p.index)
    return out


def pending(n: int, right: int = 3) -> int:
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
