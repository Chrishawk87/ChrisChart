"""Replay the candle read over history, and measure it honestly.

WHAT THIS IS FOR

"Train the model to perform better" is a reasonable thing to want and the
easiest thing in the world to fake. Tune weights until the numbers look good,
report those same numbers, and you have built a machine that describes the
past. It will not survive contact with a live candle.

So this module is built around one rule: THE NUMBER YOU ACT ON COMES FROM DATA
THE TUNING NEVER SAW. History is split chronologically -- earlier half to
train, later half held back -- weights are searched on the training half only,
and both results are reported side by side. When the training number is far
better than the test number, that gap IS the finding: the search fitted noise.
`Report.overfit_gap` puts a figure on it rather than leaving it to be noticed.

The split is chronological, never random. Shuffling candles and splitting
randomly leaks the future into the past through overlapping structure, and
produces spectacular backtests that mean nothing.

WHAT THE REPLAY CAN AND CANNOT SEE

It reconstructs what candles alone provide: trend, structure, zones, VWAP,
position in range. It CANNOT reconstruct order flow, absorption or book depth,
because those were never recorded historically. Those three are the strongest
live signals, so every number here is a FLOOR on what the live read has
available -- not an estimate of it, and not a promise about it either.

No lookahead. Swings are computed once over the whole series, but each bar
only ever sees swings whose `confirmed_at` has already passed, zones created
by breaks that already happened, and a VWAP accumulated from bars already
closed. The outcome being predicted is always the NEXT bar.

THE METRIC THAT MATTERS

Not overall accuracy -- a model that leans flat on everything and calls three
candles correctly has 100% accuracy and is useless. What matters is accuracy
AT A GIVEN SELECTIVITY: if you only take the strongest tenth of readings, how
often are you right, and how many trades is that? `threshold_curve()` answers
exactly that, because it is the question that decides whether this is tradeable.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Sequence

from .candleread import WEIGHTS, Calibration
from .structure import Candle, Swing, Zone, swings

MIN_BARS = 80
DEFAULT_TRAIN_FRACTION = 0.6


@dataclass
class Sample:
    """One replayed bar: what was knowable, and what happened next."""

    index: int
    ts: float
    # Signal name -> signed strength, BEFORE weighting. Storing it unweighted
    # is what lets the weight search re-score thousands of candidate weight
    # sets without recomputing any structure.
    contributions: dict[str, float]
    went_up: bool
    move_bps: float

    def score(self, weights: dict[str, float]) -> float:
        if not self.contributions:
            return 0.0
        total = sum(abs(weights.get(k, 1.0)) for k in self.contributions)
        if total <= 0:
            return 0.0
        return sum(v * weights.get(k, 1.0)
                   for k, v in self.contributions.items()) / total


@dataclass
class BandStat:
    band: str
    n: int
    up_rate: float | None


@dataclass
class Report:
    n: int
    accuracy: float           # among non-flat calls
    coverage: float           # share of bars that produced a call at all
    calls: int
    bands: list[BandStat] = field(default_factory=list)
    curve: list[dict] = field(default_factory=list)
    label: str = ""

    @property
    def edge(self) -> float:
        """Accuracy above a coin flip, in percentage points."""
        return (self.accuracy - 0.5) * 100.0

    def describe(self) -> str:
        if self.calls < 20:
            return (f"{self.label}: only {self.calls} directional calls in "
                    f"{self.n} bars — too few to conclude anything.")
        return (f"{self.label}: {self.accuracy:.1%} correct on {self.calls} "
                f"calls ({self.coverage:.0%} of {self.n} bars), "
                f"{self.edge:+.1f}pts vs a coin flip.")


@dataclass
class Backtest:
    coin: str
    interval: str
    samples: list[Sample]
    train: Report
    test: Report
    tuned_weights: dict[str, float] | None = None
    baseline_test: Report | None = None

    @property
    def overfit_gap(self) -> float:
        """Train accuracy minus test accuracy, in points.

        The single most informative number here. A large positive gap means
        the search found patterns that existed only in the training half.
        """
        return (self.train.accuracy - self.test.accuracy) * 100.0

    def verdict(self) -> str:
        if self.test.calls < 20:
            return ("Not enough held-out calls to judge. Pull more history "
                    "before reading anything into this.")

        parts = [self.test.describe()]

        if self.tuned_weights and self.baseline_test:
            delta = (self.test.accuracy - self.baseline_test.accuracy) * 100.0
            if delta > 1.0:
                parts.append(f"Tuning improved held-out accuracy by "
                             f"{delta:+.1f}pts, so the gain is real rather "
                             f"than fitted.")
            else:
                parts.append(f"Tuning changed held-out accuracy by "
                             f"{delta:+.1f}pts — it did not generalise. "
                             f"Keep the default weights.")

        if self.overfit_gap > 8:
            parts.append(f"Training beat the held-out half by "
                         f"{self.overfit_gap:.0f}pts. That gap is fitted "
                         f"noise, not skill — trust the held-out number.")

        if self.test.accuracy > 0.85 and self.test.calls >= 20:
            # Directional prediction on real price data does not do this.
            # When it appears, the cause is almost always a data artifact --
            # synthetic or smoothed candles, a repeated series, or a leak --
            # and reporting it as an edge is the most expensive thing this
            # module could do.
            parts.append(f"{self.test.accuracy:.0%} is not a plausible hit "
                         f"rate on real candles. Check the data before "
                         f"believing it: smoothed or synthetic series, a "
                         f"repeated range, or too few independent bars will "
                         f"all produce this.")
            return " ".join(parts)

        if self.test.accuracy < 0.52:
            parts.append("At this accuracy the read is not an edge on its own. "
                         "It is still worth having as one input, but do not "
                         "size off it.")
        return " ".join(parts)


# --------------------------------------------------------------------------
# walk-forward replay
# --------------------------------------------------------------------------

def replay(candles: Sequence[Candle], strength: int = 3,
           min_impulse_bps: float = 30.0) -> list[Sample]:
    """Score every bar from what was knowable before it.

    Swings are computed once over the whole series -- they are deterministic,
    and `confirmed_at` is what enforces causality, not the order of
    computation. Recomputing them per bar would be O(n^2) for an identical
    answer.
    """
    n = len(candles)
    if n < MIN_BARS:
        return []

    sw = swings(candles, strength)
    sw_by_confirm = sorted(sw, key=lambda s: s.confirmed_at)

    known: list[Swing] = []
    ptr = 0
    zones: list[Zone] = []
    tested: dict[int, int] = {}

    vol_sum = 0.0
    pv_sum = 0.0
    pv2_sum = 0.0

    out: list[Sample] = []

    for i in range(n - 1):
        cur = candles[i]
        nxt = candles[i + 1]

        # admit swings that became knowable at or before this bar
        while ptr < len(sw_by_confirm) and sw_by_confirm[ptr].confirmed_at <= i:
            known.append(sw_by_confirm[ptr])
            ptr += 1

        highs = [s for s in known if s.is_high]
        lows = [s for s in known if not s.is_high]

        # running VWAP over closed bars only
        if cur.volume > 0:
            vol_sum += cur.volume
            pv_sum += cur.typical * cur.volume
            pv2_sum += cur.typical * cur.typical * cur.volume

        # zone creation on a structural break
        if highs and lows:
            broke_down = cur.close < lows[-1].px
            broke_up = cur.close > highs[-1].px
            if broke_down or broke_up:
                want_bull = broke_down
                origin = None
                for j in range(i - 1, max(i - 12, -1), -1):
                    if (candles[j].bullish if want_bull else candles[j].bearish):
                        origin = j
                        break
                if origin is not None:
                    o = candles[origin]
                    if want_bull:
                        lo, hi = o.body_top, max(o.high, o.body_top)
                    else:
                        lo, hi = min(o.low, o.body_bottom), o.body_bottom
                    if hi > lo:
                        mid = (lo + hi) / 2.0
                        imp = abs(cur.close - mid) / mid * 10_000.0 if mid else 0.0
                        if imp >= min_impulse_bps:
                            zones.append(Zone(
                                kind="supply" if want_bull else "demand",
                                low=lo, high=hi, index=origin, ts=o.ts,
                                impulse_bps=imp))

        # tests accumulate as price revisits a zone
        for zi, z in enumerate(zones):
            if cur.low <= z.high and cur.high >= z.low and zi < len(zones) - 1:
                tested[zi] = tested.get(zi, 0) + 1

        contrib: dict[str, float] = {}
        px = cur.close

        # position within the bar
        rng = cur.high - cur.low
        if rng > 0:
            pos = (px - cur.low) / rng
            off = (pos - 0.5) * 2.0
            if abs(off) > 0.1:
                contrib["position"] = max(-1.0, min(1.0, off))

        # trend
        if len(highs) >= 2 and len(lows) >= 2:
            hh, hl = highs[-1].px > highs[-2].px, lows[-1].px > lows[-2].px
            lh, ll = highs[-1].px < highs[-2].px, lows[-1].px < lows[-2].px
            if hh and hl:
                contrib["structure"] = 0.6
            elif lh and ll:
                contrib["structure"] = -0.6

        # zone containing price
        for zi in range(len(zones) - 1, -1, -1):
            z = zones[zi]
            if z.low <= px <= z.high:
                t = tested.get(zi, 0)
                strength_v = 0.8 if t == 0 else 0.35
                contrib["zone"] = -strength_v if z.kind == "supply" else strength_v
                break

        # VWAP
        if vol_sum > 0:
            mean = pv_sum / vol_sum
            var = max(pv2_sum / vol_sum - mean * mean, 0.0)
            sd = var ** 0.5
            if mean > 0 and sd > 0:
                if px >= mean + 2 * sd:
                    contrib["vwap"] = -0.7
                elif px <= mean - 2 * sd:
                    contrib["vwap"] = 0.7
                else:
                    d = (px - mean) / mean * 10_000.0
                    if abs(d) > 5:
                        contrib["vwap"] = 0.35 if d > 0 else -0.35

        if not contrib:
            continue

        move = ((nxt.close - nxt.open) / nxt.open * 10_000.0
                if nxt.open > 0 else 0.0)
        out.append(Sample(index=i, ts=nxt.ts, contributions=contrib,
                          went_up=nxt.close > nxt.open, move_bps=move))

    return out


# --------------------------------------------------------------------------
# scoring a set of samples
# --------------------------------------------------------------------------

def evaluate(samples: Sequence[Sample], weights: dict[str, float],
             flat_band: float = 0.15, label: str = "") -> Report:
    """Accuracy among directional calls, plus the selectivity curve."""
    if not samples:
        return Report(n=0, accuracy=0.0, coverage=0.0, calls=0, label=label)

    cal = Calibration()
    right = calls = 0
    scored: list[tuple[float, bool]] = []

    for s in samples:
        sc = s.score(weights)
        cal.observe(sc, s.went_up)
        scored.append((sc, s.went_up))
        if abs(sc) >= flat_band:
            calls += 1
            if (sc > 0) == s.went_up:
                right += 1

    bands = [BandStat(band=row["band"], n=row["n"], up_rate=row["up_rate"])
             for row in cal.table()]

    return Report(
        n=len(samples),
        accuracy=(right / calls) if calls else 0.0,
        coverage=(calls / len(samples)) if samples else 0.0,
        calls=calls, bands=bands,
        curve=threshold_curve(scored), label=label)


def threshold_curve(scored: Sequence[tuple[float, bool]]) -> list[dict]:
    """Accuracy as a function of how selective you are.

    The question that decides whether this is tradeable is not "how accurate
    is it" but "how accurate is it when it speaks loudly, and how often does
    it speak loudly". A 54% model that is 68% on its top decile is useful; a
    54% model that is 54% everywhere is not.
    """
    out = []
    for cut in (0.15, 0.25, 0.35, 0.45, 0.55, 0.65):
        taken = [(s, up) for s, up in scored if abs(s) >= cut]
        if not taken:
            out.append({"threshold": cut, "n": 0, "accuracy": None,
                        "coverage": 0.0})
            continue
        right = sum(1 for s, up in taken if (s > 0) == up)
        out.append({"threshold": cut, "n": len(taken),
                    "accuracy": right / len(taken),
                    "coverage": len(taken) / len(scored)})
    return out


def split(samples: Sequence[Sample],
          train_fraction: float = DEFAULT_TRAIN_FRACTION
          ) -> tuple[list[Sample], list[Sample]]:
    """Chronological split. Never random -- see the module docstring."""
    cut = int(len(samples) * train_fraction)
    return list(samples[:cut]), list(samples[cut:])


# --------------------------------------------------------------------------
# tuning
# --------------------------------------------------------------------------

def tune(samples: Sequence[Sample], base: dict[str, float] | None = None,
         rounds: int = 3, flat_band: float = 0.15) -> dict[str, float]:
    """Coarse coordinate search over the weights.

    Deliberately coarse. A fine search over eight weights on a few hundred
    samples finds a global optimum for THIS data and nothing else; the grid
    below is wide enough to discover that a signal is worthless or
    backwards, and too blunt to memorise the series.

    Only ever call this on the training split.
    """
    weights = dict(base or WEIGHTS)
    names = sorted({k for s in samples for k in s.contributions})
    if not samples or not names:
        return weights

    grid = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)

    def acc(w: dict[str, float]) -> float:
        right = calls = 0
        for s in samples:
            sc = s.score(w)
            if abs(sc) >= flat_band:
                calls += 1
                if (sc > 0) == s.went_up:
                    right += 1
        # A weight set that almost never speaks can score 100% on four calls.
        # Require it to cover a tenth of the bars before its accuracy counts.
        if calls < max(20, len(samples) * 0.10):
            return 0.0
        return right / calls

    best = acc(weights)
    for _ in range(rounds):
        improved = False
        for name in names:
            current = weights.get(name, 1.0)
            for candidate in grid:
                if candidate == current:
                    continue
                trial = dict(weights)
                trial[name] = candidate
                score = acc(trial)
                if score > best + 1e-9:
                    best, weights, improved = score, trial, True
        if not improved:
            break
    return weights


def run(candles: Sequence[Candle], coin: str, interval: str,
        train_fraction: float = DEFAULT_TRAIN_FRACTION,
        do_tune: bool = True) -> Backtest:
    """Full walk-forward backtest with a held-out half."""
    samples = replay(candles)
    if len(samples) < 40:
        empty = Report(n=len(samples), accuracy=0.0, coverage=0.0, calls=0)
        return Backtest(coin=coin, interval=interval, samples=samples,
                        train=empty, test=empty)

    tr, te = split(samples, train_fraction)
    baseline_train = evaluate(tr, WEIGHTS, label="train (default weights)")
    baseline_test = evaluate(te, WEIGHTS, label="held out (default weights)")

    if not do_tune:
        return Backtest(coin=coin, interval=interval, samples=samples,
                        train=baseline_train, test=baseline_test)

    tuned = tune(tr)
    return Backtest(
        coin=coin, interval=interval, samples=samples,
        train=evaluate(tr, tuned, label="train (tuned)"),
        test=evaluate(te, tuned, label="held out (tuned)"),
        tuned_weights=tuned, baseline_test=baseline_test)
