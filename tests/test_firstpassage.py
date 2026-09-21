"""Verify the barrier-hit maths against simulation and known identities.

If this baseline is wrong, every lift number the validator produces is wrong
in the same direction, and a worthless map looks predictive. So it gets
checked hard.
"""

import math

import pytest

from liqmap.firstpassage import (distance_in_sigmas, hit_probability,
                                 hit_probability_both,
                                 monte_carlo_hit_probability)

SPOT = 100_000.0        # BTC-ish
SIGMA = 0.0006          # 6 bp/min


@pytest.mark.parametrize("pct", [0.002, 0.005, 0.01, 0.02, 0.04])
@pytest.mark.parametrize("horizon", [30.0, 120.0, 480.0])
@pytest.mark.parametrize("side", ["above", "below"])
def test_closed_form_matches_simulation(pct, horizon, side):
    barrier = SPOT * (1 + pct) if side == "above" else SPOT * (1 - pct)
    analytic = hit_probability(SPOT, barrier, horizon, SIGMA)
    simulated = monte_carlo_hit_probability(
        SPOT, barrier, horizon, SIGMA, steps_per_min=8, n=60_000, seed=5)

    # Discrete sampling can step over the barrier without recording a touch,
    # so the simulation is biased slightly low. Allow for that plus MC noise.
    assert simulated - 0.02 <= analytic <= simulated + 0.06


def test_reflection_principle_at_zero_drift():
    """With no drift, P(touch) is exactly twice P(finish beyond). This is the
    identity the whole baseline rests on."""
    from scipy import stats

    horizon, pct = 240.0, 0.01
    barrier = SPOT * (1 + pct)

    p_touch = hit_probability(SPOT, barrier, horizon, SIGMA, drift_per_min=0.0)

    s = SIGMA * math.sqrt(horizon)
    p_terminal = float(stats.norm.sf(math.log(barrier / SPOT) / s))

    assert p_touch == pytest.approx(2 * p_terminal, rel=1e-9)


def test_touch_always_exceeds_terminal():
    """A barrier can be touched and given back. Using a terminal probability
    where a touch probability belongs understates it by roughly half -- the
    single most common way this analysis breaks."""
    from scipy import stats

    for pct in (0.005, 0.01, 0.03):
        barrier = SPOT * (1 + pct)
        touch = hit_probability(SPOT, barrier, 120.0, SIGMA)
        s = SIGMA * math.sqrt(120.0)
        terminal = float(stats.norm.sf(math.log(barrier / SPOT) / s))
        assert touch > terminal * 1.7


def test_probability_falls_with_distance():
    probs = [hit_probability(SPOT, SPOT * (1 + p), 120.0, SIGMA)
             for p in (0.002, 0.005, 0.01, 0.02, 0.05)]
    assert all(a > b for a, b in zip(probs, probs[1:]))


def test_probability_rises_with_horizon():
    probs = [hit_probability(SPOT, SPOT * 1.01, h, SIGMA)
             for h in (15.0, 60.0, 240.0, 1440.0)]
    assert all(a < b for a, b in zip(probs, probs[1:]))


def test_probability_rises_with_volatility():
    probs = [hit_probability(SPOT, SPOT * 1.01, 120.0, s)
             for s in (0.0002, 0.0005, 0.001, 0.002)]
    assert all(a < b for a, b in zip(probs, probs[1:]))


def test_percentage_barriers_are_not_equidistant():
    """A barrier at -1% is FARTHER in log space than one at +1%
    (|ln 0.99| = 0.01005 vs ln 1.01 = 0.00995), so it is slightly harder to
    reach. Worth pinning down: clusters must be compared in log or sigma
    distance, never in raw percent, or the downside is systematically handed
    a lower baseline than it deserves."""
    up = hit_probability(SPOT, SPOT * 1.01, 120.0, SIGMA)
    down = hit_probability(SPOT, SPOT * 0.99, 120.0, SIGMA)
    assert down < up
    assert down == pytest.approx(up, rel=0.05)


def test_drift_makes_downside_marginally_easier_at_equal_log_distance():
    """With the percentage artifact removed, the martingale drift (-sigma^2/2
    in logs) should tilt things very slightly toward the downside barrier."""
    a = 0.01
    up = hit_probability(SPOT, SPOT * math.exp(a), 120.0, SIGMA)
    down = hit_probability(SPOT, SPOT * math.exp(-a), 120.0, SIGMA)
    assert down > up
    assert down == pytest.approx(up, rel=0.02)

    # With the drift switched off the two sides are exactly symmetric.
    up0 = hit_probability(SPOT, SPOT * math.exp(a), 120.0, SIGMA, drift_per_min=0.0)
    down0 = hit_probability(SPOT, SPOT * math.exp(-a), 120.0, SIGMA, drift_per_min=0.0)
    assert down0 == pytest.approx(up0, rel=1e-9)


def test_degenerate_inputs():
    assert hit_probability(SPOT, SPOT * 1.01, 0.0, SIGMA) == 0.0
    assert hit_probability(SPOT, SPOT * 1.01, 100.0, 0.0) == 0.0
    assert hit_probability(SPOT, SPOT, 100.0, SIGMA) == 1.0
    with pytest.raises(ValueError):
        hit_probability(-1.0, SPOT, 100.0, SIGMA)


def test_far_barrier_does_not_overflow():
    p = hit_probability(SPOT, SPOT * 12.0, 30.0, SIGMA)
    assert 0.0 <= p < 1e-6


def test_both_sides_returned_independently():
    up, down = hit_probability_both(SPOT, SPOT * 1.02, SPOT * 0.98, 240.0, SIGMA)
    assert 0 < up < 1 and 0 < down < 1
    # Both can happen in the same window, so there is no constraint that they
    # sum to at most one.
    assert up + down > 0


def test_distance_in_sigmas_is_consistent():
    horizon = 100.0
    s = SIGMA * math.sqrt(horizon)
    barrier = SPOT * math.exp(2.0 * s)
    assert distance_in_sigmas(SPOT, barrier, horizon, SIGMA) == pytest.approx(2.0, rel=1e-9)

    below = SPOT * math.exp(-1.5 * s)
    assert distance_in_sigmas(SPOT, below, horizon, SIGMA) == pytest.approx(-1.5, rel=1e-9)


def test_equal_sigma_distance_gives_equal_probability():
    """Two barriers at the same sigma-distance but different horizons and
    volatilities must carry nearly the same touch probability. This is what
    makes distance-matching a valid control."""
    a = hit_probability(SPOT, SPOT * math.exp(2 * 0.0006 * math.sqrt(100)), 100.0, 0.0006)
    b = hit_probability(SPOT, SPOT * math.exp(2 * 0.0012 * math.sqrt(400)), 400.0, 0.0012)
    assert a == pytest.approx(b, rel=0.02)
