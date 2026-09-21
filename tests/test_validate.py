"""The most important tests in the package.

A validation harness that finds signal in pure noise is worse than no harness
at all -- it launders randomness into confidence. So the first thing checked
here is that on data generated with NO effect whatsoever, the thing reports no
effect. Then that it can still detect a real one when it is there.
"""

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from liqmap.validate import (ClusterObservation, full_report, post_touch_behavior,
                             size_effect, touch_lift)

START = datetime(2026, 9, 21, tzinfo=timezone.utc)
SPOT = 100_000.0
SIGMA = 0.0006
HORIZON = 240.0


def simulate_touches(rng, spot, barriers, horizon_minutes, sigma,
                     steps_per_min=2):
    """Walk ONE path and test every barrier against it.

    This is deliberately how reality works: all the clusters in a snapshot are
    evaluated against the same price series, so their outcomes are correlated.
    A quiet window leaves every cluster untouched at once; a violent one
    sweeps several together.

    That correlation is the whole reason the bootstrap resamples snapshots
    rather than observations, so the test data has to contain it.
    """
    barriers = np.asarray(barriers, dtype=float)
    n = len(barriers)
    n_steps = max(1, int(horizon_minutes * steps_per_min))
    dt = horizon_minutes / n_steps
    sd = sigma * math.sqrt(dt)
    drift = -0.5 * sigma ** 2 * dt

    log_b = np.log(barriers)
    log_spot = math.log(spot)
    up = log_b > log_spot

    log_p = log_spot
    touched = np.zeros(n, dtype=bool)
    when = np.full(n, np.nan)

    for step in range(n_steps):
        log_p += drift + sd * rng.standard_normal()
        hit_now = np.where(up, log_p >= log_b, log_p <= log_b) & ~touched
        when[hit_now] = (step + 1) * dt
        touched |= hit_now

    return touched, when


def make_null_observations(n_snapshots=160, clusters_per=4, seed=1):
    """Clusters placed at random levels on an ordinary random walk.

    There is no relationship whatsoever between where a cluster sits and where
    price goes. A correct harness must report a lift of about 1.0 here.
    """
    rng = np.random.default_rng(seed)
    obs = []
    s = SIGMA * math.sqrt(HORIZON)

    for i in range(n_snapshots):
        ts = START + timedelta(hours=i)
        dists = rng.uniform(0.3, 2.5, clusters_per) * rng.choice([-1, 1], clusters_per)
        barriers = SPOT * np.exp(dists * s)
        touched, when = simulate_touches(rng, SPOT, barriers, HORIZON, SIGMA)

        for j in range(clusters_per):
            obs.append(ClusterObservation(
                snapshot_id=f"snap-{i}",
                ts=ts, coin="BTC", spot=SPOT, sigma_per_min=SIGMA,
                cluster_price=float(barriers[j]),
                cluster_notional=float(rng.lognormal(13, 1.2)),
                side="above" if dists[j] > 0 else "below",
                horizon_minutes=HORIZON,
                touched=bool(touched[j]),
                minutes_to_touch=None if np.isnan(when[j]) else float(when[j]),
                move_after_touch_sigmas=(
                    float(rng.standard_normal() * 0.5) if touched[j] else None),
            ))
    return obs


# --------------------------------------------------------------------------
# the null must come back null
# --------------------------------------------------------------------------

def test_no_effect_produces_lift_near_one():
    obs = make_null_observations()
    result = touch_lift(obs, n_boot=800, seed=3)

    assert result.n > 500
    # The analytic baseline and the simulated reality must agree closely.
    assert result.lift == pytest.approx(1.0, abs=0.10), (
        f"lift {result.lift:.3f} on data with no effect -- the baseline is "
        "miscalibrated and every downstream number is wrong")


def test_no_effect_is_reported_as_not_significant():
    obs = make_null_observations(seed=11)
    result = touch_lift(obs, n_boot=1500, seed=4)
    assert not result.significant, (
        "harness claims significance on pure noise -- it would launder "
        "randomness into a trading decision")


def test_no_effect_confidence_interval_contains_one():
    obs = make_null_observations(seed=21)
    result = touch_lift(obs, n_boot=1500, seed=5)
    assert result.lift_lo < 1.0 < result.lift_hi


def test_null_size_effect_reports_no_separation():
    """Cluster notional is pure noise in the null data.

    Point estimates across bands WILL differ by a few tenths -- that is
    ordinary sampling noise on correlated observations. What matters is that
    the harness does not call it an effect."""
    obs = make_null_observations(seed=31)
    res = size_effect(obs, n_boot=800)
    assert len(res.bands) >= 2
    assert not res.separated
    assert "not doing any work" in res.render()


def test_size_effect_detects_a_real_relationship():
    """When touch probability genuinely rises with notional, the bands must
    separate and the verdict must say so."""
    rng = np.random.default_rng(99)
    obs = []
    s = SIGMA * math.sqrt(HORIZON)

    for i in range(200):
        ts = START + timedelta(hours=i)
        dists = rng.uniform(0.3, 2.5, 6) * rng.choice([-1, 1], 6)
        barriers = SPOT * np.exp(dists * s)
        touched, when = simulate_touches(rng, SPOT, barriers, HORIZON, SIGMA)
        notionals = rng.lognormal(13, 1.2, 6)

        # Big clusters pull price in; small ones do nothing.
        big = notionals > np.exp(13.8)
        bonus = (~touched) & big & (rng.random(6) < 0.5)
        touched = touched | bonus

        for j in range(6):
            obs.append(ClusterObservation(
                snapshot_id=f"snap-{i}", ts=ts, coin="BTC", spot=SPOT,
                sigma_per_min=SIGMA, cluster_price=float(barriers[j]),
                cluster_notional=float(notionals[j]),
                side="above" if dists[j] > 0 else "below",
                horizon_minutes=HORIZON, touched=bool(touched[j]),
            ))

    res = size_effect(obs, n_boot=1200)
    assert res.separated
    assert res.bands[-1].lift > res.bands[0].lift
    assert "Size carries information" in res.render()


def test_null_post_touch_move_is_centred_on_zero():
    obs = make_null_observations(seed=41)
    res = post_touch_behavior(obs, n_boot=800, seed=6)
    assert res.mean_move == pytest.approx(0.0, abs=0.12)
    assert res.ci_lo < 0 < res.ci_hi


# --------------------------------------------------------------------------
# but a real effect must be found
# --------------------------------------------------------------------------

def make_effect_observations(extra_touch_prob=0.25, n_snapshots=160,
                             clusters_per=4, seed=2):
    """Same as the null, except clusters genuinely attract price.

    With some probability the cluster is treated as magnetic and gets touched
    regardless of what the path did. That is the effect the harness exists to
    detect.
    """
    rng = np.random.default_rng(seed)
    obs = []
    s = SIGMA * math.sqrt(HORIZON)

    for i in range(n_snapshots):
        ts = START + timedelta(hours=i)
        dists = rng.uniform(0.3, 2.5, clusters_per) * rng.choice([-1, 1], clusters_per)
        barriers = SPOT * np.exp(dists * s)
        touched, when = simulate_touches(rng, SPOT, barriers, HORIZON, SIGMA)

        # The injected magnetism: some untouched clusters get touched anyway.
        bonus = (~touched) & (rng.random(clusters_per) < extra_touch_prob)
        when = np.where(bonus, rng.uniform(10, HORIZON, clusters_per), when)
        touched = touched | bonus

        for j in range(clusters_per):
            obs.append(ClusterObservation(
                snapshot_id=f"snap-{i}",
                ts=ts, coin="BTC", spot=SPOT, sigma_per_min=SIGMA,
                cluster_price=float(barriers[j]),
                cluster_notional=float(rng.lognormal(13, 1.2)),
                side="above" if dists[j] > 0 else "below",
                horizon_minutes=HORIZON,
                touched=bool(touched[j]),
                minutes_to_touch=None if np.isnan(when[j]) else float(when[j]),
                move_after_touch_sigmas=(
                    float(rng.standard_normal() * 0.5 + 0.4) if touched[j] else None),
            ))
    return obs


def test_real_effect_is_detected():
    obs = make_effect_observations()
    result = touch_lift(obs, n_boot=1500, seed=7)
    assert result.lift > 1.2
    assert result.significant


def test_real_post_touch_drift_is_detected():
    """Injected continuation of +0.4 sigma must show up as a cascade."""
    obs = make_effect_observations(seed=12)
    res = post_touch_behavior(obs, n_boot=1500, seed=8)
    assert res.mean_move > 0.2
    assert res.ci_lo > 0
    assert "cascade" in res.render()


# --------------------------------------------------------------------------
# the bootstrap must respect dependence
# --------------------------------------------------------------------------

def test_bootstrap_refuses_interval_with_too_few_snapshots():
    """Five snapshots is not five independent observations no matter how many
    clusters they contain."""
    obs = [o for o in make_null_observations(n_snapshots=5, clusters_per=40)]
    result = touch_lift(obs, n_boot=500)
    assert result.n > 100
    assert math.isnan(result.lift_lo)
    assert not result.significant


def test_block_bootstrap_is_wider_than_treating_clusters_as_independent():
    """The point of resampling snapshots instead of observations.

    Clusters in a snapshot share a price path, so they carry far less
    information than their raw count suggests. A naive bootstrap that
    resamples observations individually produces an interval that is too
    narrow -- which is precisely how a backtest manufactures confidence.
    """
    obs = make_null_observations(n_snapshots=40, clusters_per=12, seed=77)
    real = [o for o in obs]

    block = touch_lift(obs, n_boot=2000, seed=9)
    block_width = block.lift_hi - block.lift_lo

    # Naive alternative: resample the observations themselves.
    rng = np.random.default_rng(9)
    touched = np.array([float(o.touched) for o in real])
    baseline = np.array([o.baseline_prob for o in real])
    idx = rng.integers(0, len(real), size=(2000, len(real)))
    draws = touched[idx].sum(axis=1) / baseline[idx].sum(axis=1)
    naive_width = float(np.percentile(draws, 97.5) - np.percentile(draws, 2.5))

    assert block_width > naive_width * 1.15, (
        f"block CI {block_width:.3f} vs naive {naive_width:.3f} -- the "
        "bootstrap is not accounting for within-snapshot correlation")


# --------------------------------------------------------------------------

def test_empty_input_does_not_crash():
    res = touch_lift([])
    assert res.n == 0
    assert math.isnan(res.lift)
    assert not res.significant
    assert "no touches" in post_touch_behavior([]).render()


def test_report_warns_on_small_samples():
    obs = make_null_observations(n_snapshots=10, clusters_per=3)
    text = full_report(obs)
    assert "far too few" in text


def test_report_runs_end_to_end():
    text = full_report(make_null_observations(n_snapshots=60))
    for heading in ("TOUCH LIFT", "LIFT BY DISTANCE", "CLUSTER SIZE",
                    "AFTER THE TOUCH"):
        assert heading in text
