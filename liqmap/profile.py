"""Volume profile from 5-second bars. Causal by construction.

THE TRAP THIS MODULE IS BUILT AROUND

A session's value area is not known until the session is over. Classifying a
10:15 bar against the day's final VAH is look-ahead, and it is the single
most common way an auction-theory backtest produces a result that vanishes
live: every "price rejected the value area high" is scored against a level
that was partly determined by the rejection itself.

So this module produces two different things and names them differently:

    final(bars)       the completed profile. For DISPLAY and for the PRIOR
                      session, which is genuinely known at today's open.

    developing(bars)  the profile as it stood at each bar, using only bars
                      up to and including that one. For CLASSIFICATION.

Nothing that decides a trade may read `final` for the session it is trading.
`test_profile` pins this with a session whose second half is deliberately
displaced: if the developing profile at bar k has seen bar k+1, its POC
moves and the test fails.

VOLUME DISTRIBUTION

Each bar's volume is spread across the price rows its range covers, in
proportion to how much of the row the bar overlaps. A bar from 7000.10 to
7000.40 puts 40% of its volume in the 7000.00-7000.25 row and 60% in the
7000.25-7000.50 row. A bar with no range puts everything in one row.

This is an approximation and it is worth being honest about what it assumes:
that trading within the bar was uniform across its range. It was not. The
true distribution is in the tape, and with the MBO or trades schema the
profile could be built from actual prints instead. On 5-second bars the
error is small because the ranges are small -- typically one or two ticks --
but it is an assumption, not a measurement, and it is the reason the row
size is fixed at one tick rather than something finer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# FIXED CONVENTIONS. Stated here, not tuned. Changing any of these is a new
# hypothesis and must be counted as one.
# ---------------------------------------------------------------------------

# Row height. One ES tick. Chosen because it is the instrument's own quantum
# and therefore requires no judgement -- any coarser value would be a free
# parameter, and any finer would be inventing resolution the data lacks.
ROW_TICKS = 1
TICK = 0.25

# Value area share. 70% is Steidlmayer's convention, an approximation of one
# standard deviation. Not chosen by fit.
VALUE_AREA = 0.70

# Node detection, both as a fraction of the POC row's volume. The 30/70 pair
# is the common desk convention. A row must ALSO be a local extreme over the
# window below, so a broad shoulder does not read as a string of HVNs.
LVN_FRACTION = 0.30
HVN_FRACTION = 0.70

# Half-width of the local-extreme window, in rows. Four ticks = one ES point.
NODE_WINDOW = 4

# An LVN must be a VALLEY: real accepted volume on BOTH sides.
#
# Without this, the thin tail of a trend day reports as a string of LVNs. On
# 2026-09-21 that produced 29 "low volume nodes", every one of them below the
# value area low, in the empty ground the session opened from and left. Those
# are range edges, not the thin shelf between two accepted areas that "LVN
# traversal" means -- and an extractor reading them would build a study out
# of where each day happened to start.
#
# The first attempt at this fix asked for a neighbour carrying 3x the row's
# own volume, and it made things WORSE -- 135 tail nodes instead of 29 --
# because in thin ground a row holding 8 contracts sits beside one holding
# 24, and three times almost nothing is still almost nothing. The test has to
# be ABSOLUTE: each side must hold a real share of the profile's peak.
VALLEY_FLOOR = 0.30

# And it must look far enough to find the shelves. The local-extreme window
# is one ES point, which cannot see accepted volume two points away -- the
# first version missed a deliberately planted valley for exactly that reason.
VALLEY_WINDOW = 12

# Rolling profile length, in seconds. One hour.
ROLLING_SECONDS = 3600.0


# How many rows a profile should have, whatever the instrument.
#
# THE ROW SIZE CANNOT BE A CONSTANT
#
# It was one tick of ES -- 0.25 -- which is correct for ES and nonsense
# everywhere else. On BTC at 105,000 a 3,000-dollar range became 12,000 rows
# twenty-five cents wide, and the node window of +/- 4 rows spanned one
# dollar: it detected nothing. On a sub-penny token the whole session range
# was smaller than a single row, so there was no profile at all.
#
# Targeting a ROW COUNT instead makes the shape mean the same thing on every
# market, and snapping up to the instrument's own tick keeps ES at exactly
# one tick per row rather than some fraction of one.
TARGET_ROWS = 400

# Distinct traded prices needed before an inferred tick is trustworthy.
MIN_PRICES_FOR_TICK = 20


def _relative_step(price: float) -> float:
    """About one basis point of price, snapped to a clean power of ten.

    The fallback when the bars are too sparse to show their own increment.
    Scale-correct everywhere, exact nowhere -- which is the right trade for
    a guess that only applies when the real answer is unavailable.
    """
    import math
    if price <= 0:
        return TICK
    return 10.0 ** math.floor(math.log10(price * 1e-4))


def infer_tick(bars: Sequence, floor: float = 1e-9) -> float:
    """The instrument's own price increment, read off the bars.

    The smallest positive gap between distinct traded prices. Taken from the
    data rather than configured, because the chart serves a thousand markets
    and a per-market tick table is a thousand chances to be wrong about one.
    """
    seen = set()
    for b in bars[:4000]:
        for v in (b.open, b.high, b.low, b.close):
            if v > 0:
                seen.add(round(float(v), 10))

    # Too few distinct prices to read an increment from. Three bars priced
    # five points apart would otherwise "infer" a five point tick, and the
    # profile would be three rows wide. Fall back to a fraction of price,
    # which is at least scale-correct on any instrument.
    if len(seen) < MIN_PRICES_FOR_TICK:
        mid = (max(seen) + min(seen)) / 2 if seen else 0.0
        return _relative_step(mid) if mid > 0 else floor

    ordered = sorted(seen)
    gaps = [b - a for a, b in zip(ordered, ordered[1:]) if b - a > floor]
    return min(gaps) if gaps else floor


def row_size(bars: Sequence, target_rows: int = TARGET_ROWS) -> float:
    """Row height for these bars: about `target_rows` rows, tick-aligned."""
    if not bars:
        return TICK
    hi = max(b.high for b in bars)
    lo = min(b.low for b in bars)
    tick = infer_tick(bars)
    span = hi - lo
    if span <= 0 or tick <= 0:
        return max(tick, TICK)
    ideal = span / max(1, target_rows)
    # Never finer than the instrument trades, and always a whole number of
    # ticks so a row boundary is a price the market can actually print.
    steps = max(1, int(ideal / tick + 0.999999))
    return steps * tick


@dataclass
class Profile:
    """A completed or in-progress volume profile."""

    lo_row: int                      # row index of the lowest populated row
    volumes: np.ndarray              # volume per row, ascending in price
    tick: float = TICK

    poc: float = 0.0
    vah: float = 0.0
    val: float = 0.0
    total: float = 0.0
    n_bars: int = 0

    hvn: list[float] = field(default_factory=list)
    lvn: list[float] = field(default_factory=list)

    @property
    def va_width_ticks(self) -> float:
        return (self.vah - self.val) / self.tick if self.tick > 0 else 0.0

    @property
    def empty(self) -> bool:
        return self.total <= 0

    def price_of(self, row: int) -> float:
        """Centre price of a row index, absolute."""
        return (self.lo_row + row + 0.5) * self.tick

    def inside(self, price: float) -> bool:
        return self.val <= price <= self.vah

    def to_dict(self) -> dict:
        return {"poc": round(self.poc, 2), "vah": round(self.vah, 2),
                "val": round(self.val, 2),
                "va_width_ticks": round(self.va_width_ticks, 1),
                "total": round(self.total, 1), "n_bars": self.n_bars,
                "hvn": [round(x, 2) for x in self.hvn],
                "lvn": [round(x, 2) for x in self.lvn]}


def _rows_for(bars: Sequence, tick: float) -> tuple[int, int]:
    lo = min(b.low for b in bars)
    hi = max(b.high for b in bars)
    return int(np.floor(lo / tick)), int(np.floor(hi / tick))


def accumulate(bars: Sequence, tick: float = TICK
               ) -> tuple[int, np.ndarray]:
    """Volume per row, distributed proportionally across each bar's range.

    Returns (lowest row index, volume array ascending in price).
    """
    if not bars:
        return 0, np.zeros(0)

    lo_row, hi_row = _rows_for(bars, tick)
    n = hi_row - lo_row + 1
    vol = np.zeros(n, dtype=float)

    for b in bars:
        v = float(b.volume)
        if v <= 0:
            continue
        lo, hi = float(b.low), float(b.high)
        if hi < lo:
            lo, hi = hi, lo
        a = int(np.floor(lo / tick)) - lo_row
        z = int(np.floor(hi / tick)) - lo_row
        if z <= a:
            vol[a] += v                       # no range: one row
            continue
        # Overlap of each row with [lo, hi], as a share of the bar's range.
        span = hi - lo
        edges_lo = (np.arange(a, z + 1) + lo_row) * tick
        edges_hi = edges_lo + tick
        overlap = np.minimum(edges_hi, hi) - np.maximum(edges_lo, lo)
        overlap = np.clip(overlap, 0.0, None)
        s = overlap.sum()
        if s <= 0:
            vol[a] += v
        else:
            vol[a:z + 1] += v * (overlap / s)

    return lo_row, vol


def _value_area(vol: np.ndarray, share: float) -> tuple[int, int, int]:
    """(poc row, val row, vah row) by the standard expansion.

    Start at the POC and repeatedly take whichever neighbouring row holds
    more volume, until `share` of the total is enclosed. Ties expand
    downward first, which is arbitrary but fixed -- leaving it to
    floating-point ordering would make the result depend on the machine.
    """
    if vol.size == 0 or vol.sum() <= 0:
        return 0, 0, 0
    poc = int(np.argmax(vol))
    target = vol.sum() * share
    lo = hi = poc
    got = vol[poc]
    while got < target and (lo > 0 or hi < vol.size - 1):
        below = vol[lo - 1] if lo > 0 else -1.0
        above = vol[hi + 1] if hi < vol.size - 1 else -1.0
        if below >= above:
            lo -= 1
            got += below
        else:
            hi += 1
            got += above
    return poc, lo, hi


def _traded_min(window: np.ndarray) -> float:
    """Smallest NON-ZERO volume in a window, or infinity if all are zero.

    An untraded row is not a competitor for "thinnest traded price". A
    genuine valley with an untouched gap beside it was being rejected
    because zero is lower than it -- so the thin shelf between two shelves,
    which is the entire object "LVN traversal" refers to, never qualified.
    """
    nz = window[window > 0]
    return float(nz.min()) if nz.size else float("inf")


def _nodes(vol: np.ndarray, lo_row: int, tick: float
           ) -> tuple[list[float], list[float]]:
    """High and low volume rows, as absolute prices.

    A row qualifies only if it is BOTH beyond the volume threshold and a
    local extreme over +/- NODE_WINDOW rows. Without the local test a broad
    high-volume shelf reports as twenty adjacent HVNs, which is not what the
    concept means and would let an event extractor fire twenty times on one
    feature.
    """
    if vol.size == 0 or vol.max() <= 0:
        return [], []
    peak = vol.max()
    w = NODE_WINDOW

    hi_rows: list[int] = []
    lo_rows: list[int] = []
    for i in range(vol.size):
        a, z = max(0, i - w), min(vol.size, i + w + 1)
        window = vol[a:z]
        if vol[i] >= HVN_FRACTION * peak and vol[i] >= window.max():
            hi_rows.append(i)
        elif vol[i] <= LVN_FRACTION * peak and vol[i] <= _traded_min(window):
            # A row with nothing in it is not a low-volume node, it is a
            # gap. Reporting untraded prices as LVNs would scatter targets
            # across places the market never visited.
            if vol[i] <= 0:
                continue
            # And it must be a VALLEY -- genuinely accepted ground on BOTH
            # sides -- rather than a point on the slope out of the range.
            va, vz = max(0, i - VALLEY_WINDOW), min(vol.size,
                                                    i + VALLEY_WINDOW + 1)
            below = vol[va:i]
            above = vol[i + 1:vz]
            if below.size == 0 or above.size == 0:
                continue
            floor = VALLEY_FLOOR * peak
            if below.max() >= floor and above.max() >= floor:
                lo_rows.append(i)

    def _cluster(rows: list[int]) -> list[float]:
        """Collapse each run of adjacent qualifying rows into one node.

        A flat shelf satisfies ">= the window max" at EVERY row, because
        every row ties. The first version reported twelve adjacent HVNs for
        one shelf, and an event extractor reading that would fire twelve
        times on a single feature. A shelf is one node, reported at its
        heaviest row.
        """
        out: list[float] = []
        run: list[int] = []
        for r in rows:
            if run and r == run[-1] + 1:
                run.append(r)
                continue
            if run:
                best = max(run, key=lambda k: vol[k])
                out.append((lo_row + best + 0.5) * tick)
            run = [r]
        if run:
            best = max(run, key=lambda k: vol[k])
            out.append((lo_row + best + 0.5) * tick)
        return out

    return _cluster(hi_rows), _cluster(lo_rows)


def final(bars: Sequence, tick: float | None = None,
          share: float = VALUE_AREA) -> Profile:
    """The completed profile over `bars`.

    `tick` None means "work it out from the bars", which is what every
    caller should want -- a fixed row size is correct for exactly one
    instrument. Pass a value only to pin it deliberately.

    DISPLAY AND PRIOR SESSIONS ONLY. Using this for the session being traded
    is look-ahead -- see the module docstring.
    """
    if tick is None:
        tick = row_size(bars)
    lo_row, vol = accumulate(bars, tick)
    p = Profile(lo_row=lo_row, volumes=vol, tick=tick, n_bars=len(bars))
    if vol.size == 0 or vol.sum() <= 0:
        return p
    poc, val_r, vah_r = _value_area(vol, share)
    p.total = float(vol.sum())
    p.poc = (lo_row + poc + 0.5) * tick
    p.val = (lo_row + val_r) * tick
    p.vah = (lo_row + vah_r + 1) * tick
    p.hvn, p.lvn = _nodes(vol, lo_row, tick)
    return p


def developing(bars: Sequence, stride: int = 1, warmup: int = 60,
               tick: float | None = None, share: float = VALUE_AREA
               ) -> list[tuple[int, Profile]]:
    """The profile as it stood at each bar, causally.

    Returns (bar index, profile) pairs. The profile at index i is built from
    bars[0..i] inclusive and has seen nothing after i.

    `stride` exists because recomputing the value area at every one of
    64,000 bars is wasted work when decisions are made hourly; it changes
    WHERE the profile is reported, never WHAT it contains.

    `warmup` suppresses profiles built on too few bars. A value area over
    three bars is a number, not a value area, and an event extractor reading
    it would fire on the first minute of every session.
    """
    out: list[tuple[int, Profile]] = []
    if not bars:
        return out
    if tick is None:
        tick = row_size(bars)

    lo_row, _ = _rows_for(bars, tick)
    hi_row = max(int(np.floor(b.high / tick)) for b in bars)
    vol = np.zeros(hi_row - lo_row + 1, dtype=float)

    for i, b in enumerate(bars):
        # Add this bar, then report. The bar being classified is CLOSED, so
        # its own volume is known; the bars after it are not touched.
        _, one = accumulate([b], tick)
        a = int(np.floor(min(b.low, b.high) / tick)) - lo_row
        vol[a:a + one.size] += one

        if i + 1 < warmup or (i % stride) != 0:
            continue
        p = Profile(lo_row=lo_row, volumes=vol.copy(), tick=tick,
                    n_bars=i + 1)
        if vol.sum() > 0:
            poc, val_r, vah_r = _value_area(vol, share)
            p.total = float(vol.sum())
            p.poc = (lo_row + poc + 0.5) * tick
            p.val = (lo_row + val_r) * tick
            p.vah = (lo_row + vah_r + 1) * tick
            p.hvn, p.lvn = _nodes(vol, lo_row, tick)
        out.append((i, p))
    return out


def rolling(bars: Sequence, seconds: float = ROLLING_SECONDS,
            stride: int = 1, tick: float | None = None,
            share: float = VALUE_AREA) -> list[tuple[int, Profile]]:
    """Profile over a trailing window ending at each reported bar.

    Trailing, never centred. A centred window would be half made of bars the
    decision has not seen.
    """
    out: list[tuple[int, Profile]] = []
    if not bars:
        return out
    start = 0
    for i, b in enumerate(bars):
        while start < i and (b.ts - bars[start].ts) > seconds:
            start += 1
        if (i % stride) != 0 or (i - start) < 10:
            continue
        out.append((i, final(bars[start:i + 1], tick, share)))
    return out
