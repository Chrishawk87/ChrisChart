"""Derived features, and the regimes to slice them by.

Raw imbalance has been tested and it is a coin. These are the second-order
reads -- not the book's state, but what the book's state means given what
price did with it.

EVERY WINDOW IS TRAILING, WITHOUT EXCEPTION

A z-score computed over a window that includes the bar being scored, or
centred on it, leaks the answer. So does a mean taken over the whole file
and applied to every bar: on 21 September it encodes what volume looked
like on the 25th. Both are easy to write, neither crashes, and both produce
a study that says yes.

So every statistic here is computed from bars STRICTLY BEFORE the one being
described, and a bar without a full trailing window gets None rather than a
value computed from a short one. `test_features` pins this with a fixture
whose tail is deliberately extreme.

THE FEATURES

ABSORPTION -- delta over displacement. The one genuine microstructure read
that aggregated depth can support: a large signed flow that moved price
barely at all means somebody sat there and took the other side. Large flow
with a large move is ordinary initiative. The ratio separates them, and
neither half does on its own.

ANOMALY -- volume z times range z. Unusual volume with an unusual range is
a different animal from unusual volume with a normal range: the first is a
move, the second is a fight.

LEVEL DISTANCE -- ticks to the nearest live key level, from `levels.py`. A
trap at a random price is noise; a trap at yesterday's high is fuel. This
is the feature that turns that sentence into something testable.

REGIME -- trend or range, quiet or busy, at a level or in the middle. Not
features but slices: the question they answer is whether a read that is
flat on average is flat everywhere, or strong somewhere and drowned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Sequence

from .levels import LevelBook, Bar as LevelBar

# Trailing window for the rolling statistics, in bars.
WINDOW = 60

# Trend/range split: |net move| over the window against total travel.
# Above this the window went somewhere; below it, it milled about.
TREND_EFFICIENCY = 0.35

# "At a level" means within this many ticks.
NEAR_TICKS = 4.0


@dataclass
class Feat:
    """One bar's derived reads. None means not computable yet."""

    ts: float
    index: int

    imbalance: float = 0.0
    absorption: float | None = None
    vol_z: float | None = None
    range_z: float | None = None
    anomaly: float | None = None
    level_ticks: float | None = None
    level_kind: str | None = None

    efficiency: float | None = None

    @property
    def ready(self) -> bool:
        return (self.absorption is not None and self.vol_z is not None
                and self.anomaly is not None)

    @property
    def trending(self) -> bool | None:
        if self.efficiency is None:
            return None
        return self.efficiency >= TREND_EFFICIENCY

    @property
    def busy(self) -> bool | None:
        return None if self.vol_z is None else self.vol_z > 0.0

    def near_level(self, ticks: float = NEAR_TICKS) -> bool:
        return self.level_ticks is not None and abs(self.level_ticks) <= ticks

    def regime(self) -> str:
        """A single label, for slicing."""
        if self.efficiency is None or self.vol_z is None:
            return "unknown"
        return (f"{'trend' if self.trending else 'range'}-"
                f"{'busy' if self.busy else 'quiet'}-"
                f"{'at-level' if self.near_level() else 'mid'}")


def _mean_sd(xs: Sequence[float]) -> tuple[float, float]:
    n = len(xs)
    if n < 2:
        return (xs[0] if xs else 0.0), 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)


def _z(x: float, window: Sequence[float]) -> float | None:
    """Z-score against a trailing window. None if the window is flat.

    A zero standard deviation makes the score undefined, not infinite. A
    window of identical values has told us nothing about what is unusual.
    """
    if len(window) < 10:
        return None
    m, sd = _mean_sd(window)
    if sd <= 1e-12:
        return None
    return (x - m) / sd


def build(slices: Sequence, tick: float = 0.25,
          window: int = WINDOW) -> list[Feat]:
    """Features for every bar, trailing windows only.

    The level book is fed bar by bar in order, so the levels a bar sees are
    the ones that existed when it opened -- never a level defined by a
    session that had not finished.
    """
    book = LevelBook(tick=tick)
    out: list[Feat] = []

    vols: list[float] = []
    rngs: list[float] = []
    closes: list[float] = []

    for i, s in enumerate(slices):
        f = Feat(ts=s.ts, index=i, imbalance=s.imbalance(levels=5))

        # -- absorption: signed flow per basis point of displacement -----
        move = abs(s.prior_move_bps)
        if s.trades_in > 0:
            total = s.buy_vol + s.sell_vol
            lean = (s.delta / total) if total > 0 else 0.0
            # A floor on the denominator rather than a guard against zero:
            # flow that moved price by a thousandth of a basis point is
            # absorption, and dividing by that raw number would report it
            # as several thousand rather than "a lot".
            f.absorption = lean / max(move, 0.05)

        # -- anomaly: how unusual, against what came before --------------
        f.vol_z = _z(s.volume, vols)
        bar_range = ((s.high - s.low) / s.open * 10_000.0
                     if s.open > 0 else 0.0)
        f.range_z = _z(bar_range, rngs)
        if f.vol_z is not None and f.range_z is not None:
            f.anomaly = f.vol_z * f.range_z

        # -- trend or range, over the trailing window --------------------
        if len(closes) >= 10:
            net = abs(closes[-1] - closes[0])
            travel = sum(abs(closes[k] - closes[k - 1])
                         for k in range(1, len(closes)))
            f.efficiency = (net / travel) if travel > 0 else 0.0

        # -- distance to the nearest live level --------------------------
        # Read BEFORE this bar is fed to the book, so the level set is the
        # one that existed at the open rather than one this bar helped set.
        if s.open > 0:
            near = book.nearest(s.open)
            if near is not None:
                f.level_ticks = near.distance(s.open, tick)
                f.level_kind = near.kind.value

        book.observe(LevelBar(ts=s.ts, open=s.open, high=s.high,
                              low=s.low, close=s.close, volume=s.volume))

        out.append(f)

        vols.append(s.volume)
        rngs.append(bar_range)
        closes.append(s.close)
        del vols[:-window]
        del rngs[:-window]
        del closes[:-window]

    return out


def regimes(feats: Sequence[Feat]) -> dict[str, list[int]]:
    """Bar indices grouped by regime label."""
    out: dict[str, list[int]] = {}
    for f in feats:
        out.setdefault(f.regime(), []).append(f.index)
    return out


def values(feats: Sequence[Feat], name: str) -> list[float | None]:
    return [getattr(f, name, None) for f in feats]
