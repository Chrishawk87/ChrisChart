"""The harness that decides whether the map is worth anything.

Every commercial liquidation-heatmap product stops at drawing the picture.
The picture is always compelling, because clusters form near price and price
moves around near price, so clusters get touched constantly. That proves
nothing.

Three questions are asked here, in increasing order of how much they matter:

  1. TOUCH LIFT. Does price reach cluster levels more often than an arbitrary
     level at the same sigma-distance, over the same horizon? The baseline is
     the closed-form first-passage probability, and optionally a matched
     placebo level with no cluster on it. A lift of 1.0 means the map adds
     nothing.

  2. CALIBRATION. Treating the baseline probability as a forecast, does
     knowing the cluster's size improve it? If big clusters and small clusters
     get touched at the same rate relative to baseline, then size -- the thing
     the whole map is built on -- carries no information.

  3. WHAT HAPPENS AFTER. Conditional on a touch, does price accelerate
     through (a cascade) or reverse (absorption)? This is the only one that
     pays. A level that gets touched exactly as often as chance predicts but
     reliably produces a violent move afterwards is still tradeable.

On statistics: cluster observations are NOT independent. Several clusters
share one snapshot, and consecutive snapshots overlap in time. Treating them
as independent inflates significance enormously -- the classic way a backtest
convinces you of something that is not there. Everything here uses a block
bootstrap that resamples whole snapshots.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Sequence

import numpy as np

from .firstpassage import distance_in_sigmas, hit_probability

Side = Literal["above", "below"]


@dataclass
class ClusterObservation:
    """One cluster, its baseline, and what actually happened to it."""

    snapshot_id: str
    ts: datetime
    coin: str
    spot: float
    sigma_per_min: float
    cluster_price: float
    cluster_notional: float
    side: Side
    horizon_minutes: float

    touched: bool
    minutes_to_touch: float | None = None
    # Signed move over the K minutes after the touch, expressed in the
    # direction the touch implies (positive = continuation, negative =
    # reversal). None when never touched.
    move_after_touch_sigmas: float | None = None

    is_placebo: bool = False

    # Computed once on first access. The bootstrap resamples the same
    # observations thousands of times, so recomputing these per access turns a
    # two-second report into a ten-minute one.
    _baseline: float | None = field(default=None, repr=False, compare=False)
    _sigma_dist: float | None = field(default=None, repr=False, compare=False)

    @property
    def baseline_prob(self) -> float:
        if self._baseline is None:
            self._baseline = hit_probability(
                self.spot, self.cluster_price, self.horizon_minutes,
                self.sigma_per_min)
        return self._baseline

    @property
    def sigma_distance(self) -> float:
        if self._sigma_dist is None:
            self._sigma_dist = distance_in_sigmas(
                self.spot, self.cluster_price, self.horizon_minutes,
                self.sigma_per_min)
        return self._sigma_dist


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------

MIN_BLOCKS = 8


def _block_bootstrap_ratio(numerators: dict[str, float],
                           denominators: dict[str, float],
                           n_boot: int = 2000, seed: int = 0
                           ) -> tuple[float, float, float]:
    """Bootstrap a ratio-of-sums by resampling whole SNAPSHOTS.

    Clusters within a snapshot share a price, a volatility and a moment in
    time, so they rise and fall together. Resampling them individually would
    treat correlated observations as fresh evidence and shrink the interval to
    a fiction -- which is exactly how a backtest talks you into a signal that
    is not there.

    Both statistics in this module are ratios of sums (touches over expected
    touches; total move over number of touches), so one routine covers them.
    Everything is pre-aggregated per block and resampled as a matrix, which
    keeps a full report in the low seconds rather than minutes.

    Returns (point estimate, 2.5th percentile, 97.5th percentile).
    """
    keys = [k for k in numerators if k in denominators]
    if not keys:
        return float("nan"), float("nan"), float("nan")

    num = np.array([numerators[k] for k in keys], dtype=float)
    den = np.array([denominators[k] for k in keys], dtype=float)

    total_den = den.sum()
    point = float(num.sum() / total_den) if total_den > 0 else float("nan")

    if len(keys) < MIN_BLOCKS:
        # Too few independent blocks for an interval to mean anything.
        return point, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    boot_num = num[idx].sum(axis=1)
    boot_den = den[idx].sum(axis=1)

    valid = boot_den > 0
    if valid.sum() < 50:
        return point, float("nan"), float("nan")

    draws = boot_num[valid] / boot_den[valid]
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _group_sums(obs: Sequence[ClusterObservation], value_fn, weight_fn
                ) -> tuple[dict[str, float], dict[str, float]]:
    """Aggregate values and weights per snapshot for the bootstrap."""
    num: dict[str, float] = {}
    den: dict[str, float] = {}
    for o in obs:
        num[o.snapshot_id] = num.get(o.snapshot_id, 0.0) + value_fn(o)
        den[o.snapshot_id] = den.get(o.snapshot_id, 0.0) + weight_fn(o)
    return num, den


# --------------------------------------------------------------------------
# 1. touch lift
# --------------------------------------------------------------------------

@dataclass
class LiftResult:
    n: int
    n_snapshots: int
    observed_rate: float
    expected_rate: float
    lift: float
    lift_lo: float
    lift_hi: float

    @property
    def significant(self) -> bool:
        """The interval excludes 1.0, i.e. the map differs from chance."""
        return (math.isfinite(self.lift_lo) and math.isfinite(self.lift_hi)
                and (self.lift_lo > 1.0 or self.lift_hi < 1.0))

    def render(self) -> str:
        ci = (f"[{self.lift_lo:.2f}, {self.lift_hi:.2f}]"
              if math.isfinite(self.lift_lo) else "[too few snapshots]")
        verdict = ("clusters ARE touched more than chance" if self.lift > 1 and self.significant
                   else "clusters are touched LESS than chance" if self.lift < 1 and self.significant
                   else "indistinguishable from chance")
        return "\n".join([
            "TOUCH LIFT",
            f"  observations      {self.n} across {self.n_snapshots} snapshots",
            f"  observed touches  {self.observed_rate:.3f}",
            f"  baseline expects  {self.expected_rate:.3f}",
            f"  lift              {self.lift:.2f}x  95% CI {ci}",
            f"  verdict           {verdict}",
        ])


def touch_lift(obs: Sequence[ClusterObservation], n_boot: int = 2000,
               seed: int = 0) -> LiftResult:
    """Do clusters get touched more than a distance-matched random level?"""
    real = [o for o in obs if not o.is_placebo]
    if not real:
        return LiftResult(0, 0, float("nan"), float("nan"),
                          float("nan"), float("nan"), float("nan"))

    observed = float(np.mean([o.touched for o in real]))
    expected = float(np.mean([o.baseline_prob for o in real]))

    # Lift = total touches / total expected touches, bootstrapped by snapshot.
    num, den = _group_sums(real,
                           value_fn=lambda o: float(o.touched),
                           weight_fn=lambda o: o.baseline_prob)
    point, lo, hi = _block_bootstrap_ratio(num, den, n_boot, seed)

    return LiftResult(
        n=len(real),
        n_snapshots=len({o.snapshot_id for o in real}),
        observed_rate=observed,
        expected_rate=expected,
        lift=point,
        lift_lo=lo,
        lift_hi=hi,
    )


def lift_by_distance(obs: Sequence[ClusterObservation],
                     edges: Sequence[float] = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
                     ) -> str:
    """Lift broken out by how far the cluster sits from spot.

    Worth looking at even when the headline lift is flat. Any real effect
    should be concentrated near price and fade with distance; a "signal" that
    is strongest three sigma away is almost certainly an artifact.
    """
    real = [o for o in obs if not o.is_placebo]
    if not real:
        return "no observations"

    lines = ["LIFT BY DISTANCE",
             f"  {'sigma dist':>14} {'n':>6} {'observed':>9} {'baseline':>9} {'lift':>7}",
             "  " + "-" * 50]

    bounds = [0.0, *edges, float("inf")]
    for lo, hi in zip(bounds, bounds[1:]):
        group = [o for o in real if lo <= abs(o.sigma_distance) < hi]
        if len(group) < 10:
            continue
        observed = float(np.mean([o.touched for o in group]))
        expected = float(np.mean([o.baseline_prob for o in group]))
        lift = observed / expected if expected > 0 else float("nan")
        label = f"{lo:.2f}-{hi:.2f}" if math.isfinite(hi) else f"{lo:.2f}+"
        lines.append(f"  {label:>14} {len(group):>6} {observed:>9.3f} "
                     f"{expected:>9.3f} {lift:>7.2f}x")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 2. does size matter?
# --------------------------------------------------------------------------

@dataclass
class SizeBand:
    cut: float
    n: int
    observed: float
    expected: float
    lift: float
    lift_lo: float
    lift_hi: float


@dataclass
class SizeResult:
    n: int
    bands: list[SizeBand]

    @property
    def separated(self) -> bool:
        """Do the smallest and largest bands have non-overlapping intervals?

        Without this check, ordinary sampling noise across bands reads as a
        size effect. On correlated data a spread of 0.3x between bands is
        entirely routine under the null, so point estimates alone will invent
        a relationship that is not there.
        """
        if len(self.bands) < 2:
            return False
        lo, hi = self.bands[0], self.bands[-1]
        if not all(math.isfinite(v) for v in
                   (lo.lift_lo, lo.lift_hi, hi.lift_lo, hi.lift_hi)):
            return False
        return hi.lift_lo > lo.lift_hi or hi.lift_hi < lo.lift_lo

    def render(self) -> str:
        if not self.bands:
            return "CLUSTER SIZE\n  not enough observations"

        lines = ["CLUSTER SIZE",
                 f"  {'notional >=':>16} {'n':>6} {'observed':>9} {'baseline':>9} "
                 f"{'lift':>7}  95% CI",
                 "  " + "-" * 68]
        for b in self.bands:
            ci = (f"[{b.lift_lo:.2f}, {b.lift_hi:.2f}]"
                  if math.isfinite(b.lift_lo) else "[too few snapshots]")
            lines.append(f"  ${b.cut:>15,.0f} {b.n:>6} {b.observed:>9.3f} "
                         f"{b.expected:>9.3f} {b.lift:>7.2f}x  {ci}")
        lines.append("")

        if not self.separated:
            lines.append("  Lift does not separate across size once the intervals are "
                         "accounted for.")
            lines.append("  The notional weighting -- the premise of the whole map -- is "
                         "not doing any work here.")
        elif self.bands[-1].lift > self.bands[0].lift:
            lines.append("  Bigger clusters show more lift, and the intervals separate. "
                         "Size carries information.")
        else:
            lines.append("  Bigger clusters show LESS lift, with separated intervals. "
                         "Suspect the map is")
            lines.append("  inverted, or that large clusters simply sit further from price.")
        return "\n".join(lines)


def size_effect(obs: Sequence[ClusterObservation], n_groups: int = 3,
                n_boot: int = 1000, seed: int = 0) -> SizeResult:
    """Split clusters by notional and compare lift across the groups.

    Each band gets its own block-bootstrapped interval, because the whole
    point is to decide whether the bands genuinely differ -- and they
    routinely appear to when they do not.
    """
    real = [o for o in obs if not o.is_placebo]
    if len(real) < n_groups * 20:
        return SizeResult(len(real), [])

    notionals = np.array([o.cluster_notional for o in real])
    cuts = np.quantile(notionals, np.linspace(0, 1, n_groups + 1)[:-1])

    bands: list[SizeBand] = []
    for i, cut in enumerate(cuts):
        hi = cuts[i + 1] if i + 1 < len(cuts) else float("inf")
        group = [o for o in real if cut <= o.cluster_notional < hi]
        if len(group) < 10:
            continue

        observed = float(np.mean([o.touched for o in group]))
        expected = float(np.mean([o.baseline_prob for o in group]))
        num, den = _group_sums(group,
                               value_fn=lambda o: float(o.touched),
                               weight_fn=lambda o: o.baseline_prob)
        lift, lo, hi_ci = _block_bootstrap_ratio(num, den, n_boot, seed + i)

        bands.append(SizeBand(float(cut), len(group), observed, expected,
                              lift, lo, hi_ci))

    return SizeResult(len(real), bands)


# --------------------------------------------------------------------------
# 3. what happens after the touch -- the part that pays
# --------------------------------------------------------------------------

@dataclass
class PostTouchResult:
    n_touched: int
    mean_move: float
    median_move: float
    ci_lo: float
    ci_hi: float
    continuation_rate: float

    def render(self) -> str:
        ci = (f"[{self.ci_lo:+.3f}, {self.ci_hi:+.3f}]"
              if math.isfinite(self.ci_lo) else "[too few snapshots]")
        if not math.isfinite(self.mean_move):
            return "AFTER THE TOUCH\n  no touches recorded"

        if math.isfinite(self.ci_lo) and self.ci_lo > 0:
            verdict = "cascades -- price accelerates through the level"
        elif math.isfinite(self.ci_hi) and self.ci_hi < 0:
            verdict = "absorbs -- price reverses off the level"
        else:
            verdict = "no reliable direction after touch"

        return "\n".join([
            "AFTER THE TOUCH  (move in sigma, signed toward continuation)",
            f"  touches           {self.n_touched}",
            f"  mean move         {self.mean_move:+.3f} sigma   95% CI {ci}",
            f"  median move       {self.median_move:+.3f} sigma",
            f"  continued         {self.continuation_rate:.1%} of the time",
            f"  verdict           {verdict}",
        ])


def post_touch_behavior(obs: Sequence[ClusterObservation], n_boot: int = 2000,
                        seed: int = 0) -> PostTouchResult:
    """Conditional on a touch, does price carry on through or turn around?

    The move is signed so that positive always means continuation in the
    direction that touching the cluster implies -- through an overhead short
    cluster is up, through a long cluster below is down. That makes cascade
    and reversal readable as one number regardless of side.
    """
    touched = [o for o in obs
               if not o.is_placebo and o.touched
               and o.move_after_touch_sigmas is not None]
    if not touched:
        return PostTouchResult(0, float("nan"), float("nan"),
                               float("nan"), float("nan"), float("nan"))

    moves = np.array([o.move_after_touch_sigmas for o in touched], dtype=float)

    # Mean move = total move / number of touches, bootstrapped by snapshot.
    num, den = _group_sums(touched,
                           value_fn=lambda o: float(o.move_after_touch_sigmas),
                           weight_fn=lambda o: 1.0)
    point, lo, hi = _block_bootstrap_ratio(num, den, n_boot, seed)

    return PostTouchResult(
        n_touched=len(touched),
        mean_move=point,
        median_move=float(np.median(moves)),
        ci_lo=lo,
        ci_hi=hi,
        continuation_rate=float(np.mean(moves > 0)),
    )


# --------------------------------------------------------------------------
# placebo comparison
# --------------------------------------------------------------------------

def placebo_comparison(obs: Sequence[ClusterObservation]) -> str:
    """Real clusters against matched levels that had no cluster on them.

    Stronger than the analytic baseline, because it controls for any way the
    random-walk model misdescribes the asset. If real and placebo levels get
    touched at the same rate, the map is decoration.
    """
    real = [o for o in obs if not o.is_placebo]
    placebo = [o for o in obs if o.is_placebo]
    if not real or not placebo:
        return "PLACEBO\n  no placebo observations -- generate them to run this check"

    r_obs = float(np.mean([o.touched for o in real]))
    p_obs = float(np.mean([o.touched for o in placebo]))
    r_dist = float(np.mean([abs(o.sigma_distance) for o in real]))
    p_dist = float(np.mean([abs(o.sigma_distance) for o in placebo]))

    lines = [
        "PLACEBO COMPARISON",
        f"  real clusters     {r_obs:.3f} touched   (n={len(real)}, "
        f"mean distance {r_dist:.2f} sigma)",
        f"  matched placebos  {p_obs:.3f} touched   (n={len(placebo)}, "
        f"mean distance {p_dist:.2f} sigma)",
    ]
    if abs(r_dist - p_dist) > 0.15:
        lines.append("  WARNING: distances are not well matched; this comparison "
                     "is measuring distance, not clusters.")
    ratio = r_obs / p_obs if p_obs > 0 else float("nan")
    lines.append(f"  ratio             {ratio:.2f}x")
    return "\n".join(lines)


# --------------------------------------------------------------------------

def full_report(obs: Sequence[ClusterObservation]) -> str:
    real = [o for o in obs if not o.is_placebo]
    n_snap = len({o.snapshot_id for o in real})

    parts = [
        "=" * 64,
        "  LIQUIDATION MAP VALIDATION",
        "=" * 64,
        "",
        touch_lift(obs).render(),
        "",
        lift_by_distance(obs),
        "",
        size_effect(obs).render(),
        "",
        post_touch_behavior(obs).render(),
        "",
        placebo_comparison(obs),
        "",
    ]

    if n_snap < 30:
        parts.append(
            f"  NOTE: {n_snap} independent snapshots is far too few to conclude "
            "anything.\n  The confidence intervals are the honest part of this "
            "report -- read those,\n  not the point estimates.")
    return "\n".join(parts)
