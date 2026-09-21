"""Barrier-hit probability: the baseline every cluster must beat.

The whole validation rests on one question. A liquidation cluster sits at some
distance from spot. Price reaches it, or it doesn't. Does it reach it MORE
OFTEN than an arbitrary level at the same distance would?

That baseline is not a guess -- it has a closed form. Under a driftless random
walk, the probability of touching a barrier within a horizon is given by the
reflection principle, and for the martingale case (drift -sigma^2/2 in logs)
there is an exact expression with the drift carried.

This matters more than it sounds. Liquidation clusters form NEAR price,
because that is where leverage was put on. So "clusters get touched a lot" is
trivially true and means nothing. The only honest test compares each cluster
against a random level at the same distance, over the same horizon, at the
same volatility. Everything in `validate.py` is built on this function.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

Direction = Literal["above", "below"]


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via erfc.

    Mathematically identical to scipy.stats.norm.cdf for scalars, and roughly
    two orders of magnitude faster because it is a C builtin rather than a
    dispatch through a distribution object. That gap matters: the bootstrap in
    validate.py evaluates this millions of times.
    """
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def hit_probability(spot: float, barrier: float, horizon_minutes: float,
                    sigma_per_min: float, drift_per_min: float | None = None
                    ) -> float:
    """P(price touches `barrier` at any point within the horizon).

    This is a FIRST PASSAGE probability -- it asks whether the barrier is ever
    reached, not whether price ends up beyond it. Those differ by roughly a
    factor of two for a distant barrier, and using the terminal probability
    where a touch probability belongs is the most common way this analysis
    gets quietly broken.

    Uses the standard result for Brownian motion with drift: for a barrier at
    log-distance a > 0 with drift mu and volatility s over time T,

        P(hit) = Phi((-a + mu*T) / (s*sqrt(T)))
               + exp(2*mu*a / s^2) * Phi((-a - mu*T) / (s*sqrt(T)))

    `drift_per_min` defaults to -sigma^2/2, which makes the PRICE a martingale
    (zero expected return). At the horizons this is used for, the drift term
    is nearly irrelevant, but carrying it costs nothing and keeps the maths
    honest at longer ones.
    """
    if spot <= 0 or barrier <= 0:
        raise ValueError("spot and barrier must be positive")
    if horizon_minutes <= 0 or sigma_per_min <= 0:
        return 0.0

    s = sigma_per_min * math.sqrt(horizon_minutes)
    if s <= 0:
        return 0.0

    log_dist = math.log(barrier / spot)
    if abs(log_dist) < 1e-12:
        return 1.0                       # already touching

    if drift_per_min is None:
        drift_per_min = -0.5 * sigma_per_min ** 2
    mu_t = drift_per_min * horizon_minutes

    # Work with a positive barrier distance; flip the drift for a downside
    # barrier so one formula covers both.
    a = abs(log_dist)
    if log_dist < 0:
        mu_t = -mu_t

    var = sigma_per_min ** 2 * horizon_minutes
    term1 = _norm_cdf((-a + mu_t) / s)

    exponent = 2.0 * mu_t * a / var if var > 0 else 0.0
    # Guard the exponential: for a far barrier with adverse drift this
    # underflows harmlessly, but a large positive value would overflow.
    if exponent > 700:
        return 1.0
    term2 = math.exp(exponent) * _norm_cdf((-a - mu_t) / s)

    return float(min(1.0, max(0.0, term1 + term2)))


def hit_probability_both(spot: float, upper: float, lower: float,
                         horizon_minutes: float, sigma_per_min: float
                         ) -> tuple[float, float]:
    """Touch probabilities for an upper and a lower barrier, independently.

    These do NOT sum to anything meaningful -- both can be touched in the same
    window. They are reported separately on purpose.
    """
    return (hit_probability(spot, upper, horizon_minutes, sigma_per_min),
            hit_probability(spot, lower, horizon_minutes, sigma_per_min))


def distance_in_sigmas(spot: float, barrier: float, horizon_minutes: float,
                       sigma_per_min: float) -> float:
    """Signed barrier distance in standard deviations of the horizon.

    The natural unit for comparing clusters across assets and volatility
    regimes. A cluster 2% away in a quiet hour is a very different object from
    one 2% away during a cascade.
    """
    if sigma_per_min <= 0 or horizon_minutes <= 0:
        return 0.0
    s = sigma_per_min * math.sqrt(horizon_minutes)
    return math.log(barrier / spot) / s


def monte_carlo_hit_probability(spot: float, barrier: float,
                                horizon_minutes: float, sigma_per_min: float,
                                steps_per_min: int = 4, n: int = 100_000,
                                seed: int = 0) -> float:
    """Simulate the path and check whether the barrier is ever touched.

    Used by the tests to verify the closed form. Note that a discretely
    sampled path can step OVER a barrier without registering a touch, which
    biases this estimate DOWNWARD -- so the tests allow for it, and the
    discretisation is fine enough to keep the gap small.
    """
    rng = np.random.default_rng(seed)
    n_steps = max(1, int(round(horizon_minutes * steps_per_min)))
    dt = horizon_minutes / n_steps

    sd = sigma_per_min * math.sqrt(dt)
    drift = -0.5 * sigma_per_min ** 2 * dt

    log_spot = math.log(spot)
    log_barrier = math.log(barrier)
    up = log_barrier > log_spot

    paths = np.full(n, log_spot)
    hit = np.zeros(n, dtype=bool)

    for _ in range(n_steps):
        paths += drift + sd * rng.standard_normal(n)
        hit |= (paths >= log_barrier) if up else (paths <= log_barrier)

    return float(hit.mean())
