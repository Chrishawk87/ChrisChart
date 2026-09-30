"""Read the book at its own rate, not once a candle.

WHAT WAS THROWN AWAY

The five-day file holds 44 million book updates. Every test so far used
5,969 of them -- one per candle, 0.013% of what was bought. The book moves
roughly 660 times a second during the cash session and we were looking at
it once a minute.

So this samples every few seconds instead, which turns 5,969 observations
into something like 74,000, and adds the two things a single snapshot
cannot carry: how fast the book is CHANGING, and what the tape did over
the same few seconds.

THE TRAP THIS MODULE IS BUILT AROUND: OVERLAP

Sampling every 5 seconds and measuring 30 seconds forward means
consecutive observations share five sixths of their outcome. They are not
independent, and a Wilson interval computed over them is far too narrow --
74,000 overlapping looks can carry less information than 6,000 independent
ones, while reporting an interval six times tighter. That is how a study
produces 52.1% (51.8%-52.4%) from pure noise and calls it real.

So every horizon is scored on a STRIDE of at least its own length: to
measure 30 seconds forward, observations are taken 30 seconds apart. Fewer
rows, honest intervals. `stride_for` enforces it and `test_intrabar` pins
that a shorter stride is refused rather than quietly accepted.

WHAT COUNTS AS A RESULT

Not a hit rate. The cost of a round trip in ES is the commission plus the
spread a taker gives up -- about $17.50 all in, not the $5 used earlier in
this project -- and against that, direction has to clear
`cost / (width x tick value)` over the barrier being traded. Both numbers
are printed side by side so the gap is visible rather than inferred.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

# A taker's true round trip on one ES contract: commission and fees, plus
# the one-tick spread given up buying the ask and selling the bid.
COST_COMMISSION = 5.00
COST_SPREAD = 12.50
COST_TAKER = COST_COMMISSION + COST_SPREAD

ES_TICK = 0.25
ES_TICK_USD = 12.50


@dataclass
class Read:
    """The book and the tape at one instant, plus how they are changing."""

    index: int
    ts: float
    mid: float

    imb1: float = 0.0          # touch only
    imb3: float = 0.0
    imb5: float = 0.0
    imb10: float = 0.0         # everything MBP-10 carries

    velocity: float = 0.0      # change in imb5 since the previous read
    flow: float = 0.0          # signed volume share over the window
    trades: int = 0

    @property
    def usable(self) -> bool:
        return self.mid > 0


def _imbalance(bids, asks, depth: int) -> float:
    b = sum(sz for _, sz in bids[:depth])
    a = sum(sz for _, sz in asks[:depth])
    return (b - a) / (b + a) if (b + a) > 0 else 0.0


def reads(slices: Sequence) -> list[Read]:
    """One Read per slice, with velocity differenced against the previous.

    Velocity comes from the PREVIOUS row, never the next one. Differencing
    forward would put a few seconds of the answer into the question, and on
    a five-second grid that is most of a short horizon.
    """
    out: list[Read] = []
    prev5: float | None = None

    for i, s in enumerate(slices):
        mid = s.mid if s.mid > 0 else s.open
        r = Read(index=i, ts=s.ts, mid=mid)
        if s.bids and s.asks:
            r.imb1 = _imbalance(s.bids, s.asks, 1)
            r.imb3 = _imbalance(s.bids, s.asks, 3)
            r.imb5 = _imbalance(s.bids, s.asks, 5)
            r.imb10 = _imbalance(s.bids, s.asks, 10)
        total = s.buy_vol + s.sell_vol
        r.flow = (s.delta / total) if total > 0 else 0.0
        r.trades = s.trades_in
        r.velocity = 0.0 if prev5 is None else (r.imb5 - prev5)
        prev5 = r.imb5
        out.append(r)
    return out


def stride_for(horizon_rows: int, stride: int | None = None) -> int:
    """Rows between observations, never fewer than the horizon.

    Overlapping observations share their outcome, so the interval computed
    over them understates the uncertainty -- badly. A caller asking for a
    shorter stride gets the horizon instead.
    """
    if horizon_rows < 1:
        return 1
    return max(horizon_rows, stride or 0)


def forward_bps(slices: Sequence, i: int, rows: int) -> float | None:
    """Mid-to-mid move from row i to row i+rows, in bps."""
    j = i + rows
    if i < 0 or j >= len(slices):
        return None
    a = slices[i].open
    b = slices[j].open
    if a <= 0 or b <= 0:
        return None
    return (b - a) / a * 10_000.0


def direction(slices: Sequence, sides: Sequence[int], horizon_rows: int,
              stride: int | None = None) -> dict:
    """How often the call matched the move, on non-overlapping rows."""
    step = stride_for(horizon_rows, stride)
    hit = n = 0
    for i in range(0, len(slices) - horizon_rows, step):
        side = sides[i]
        if side == 0:
            continue
        mv = forward_bps(slices, i, horizon_rows)
        if mv is None or mv == 0.0:
            continue
        n += 1
        if (mv > 0) == (side > 0):
            hit += 1
    if n < 30:
        return {"ready": False, "n": n}
    lo, hi = _wilson(hit, n)
    return {"ready": True, "n": n, "rate": hit / n, "lo": lo, "hi": hi,
            "real": lo > 0.5 or hi < 0.5, "stride": step}


def needed_edge(take_ticks: float, stop_ticks: float,
                cost_usd: float = COST_TAKER,
                tick_usd: float = ES_TICK_USD) -> float:
    """Percentage points above the no-skill rate that costs demand.

    For a market with no predictable drift the no-skill rate at +T/-S is
    S/(T+S), and the breakeven after costs is (S*v+c)/((T+S)*v). Almost
    everything cancels: what is left is cost over the TOTAL barrier width.
    The asymmetry does not appear at all -- widening the target lowers the
    breakeven and the achieved rate by the same amount.
    """
    width = (take_ticks + stop_ticks) * tick_usd
    return cost_usd / width if width > 0 else 1.0


def no_skill(take_ticks: float, stop_ticks: float) -> float:
    return stop_ticks / (take_ticks + stop_ticks)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - s) / d), min(1.0, (c + s) / d)


def signal(rs: Sequence[Read], name: str, cut: float) -> list[int]:
    """Turn one feature into calls: +1 above the cut, -1 below, else 0."""
    out = []
    for r in rs:
        v = getattr(r, name, 0.0)
        out.append(1 if v >= cut else -1 if v <= -cut else 0)
    return out


def quantile_cut(rs: Sequence[Read], name: str, q: float) -> float:
    vals = sorted(abs(getattr(r, name, 0.0)) for r in rs)
    if not vals:
        return 0.0
    return vals[min(len(vals) - 1, int(q * len(vals)))]
