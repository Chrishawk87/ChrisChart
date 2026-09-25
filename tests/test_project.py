"""The projected candle, and the scoring that decides whether to believe it.

The tests that matter most here are the calibration ones. A projection
that looks plausible and is badly calibrated is worse than no projection:
it puts a number on a guess, and a number invites a position size.
"""

from __future__ import annotations

import math
import random

import pytest

from liqmap import project as pj


def bar(c, o=None, h=None, l=None, v=10.0):
    o = c if o is None else o
    return {"o": o, "h": h if h is not None else max(o, c),
            "l": l if l is not None else min(o, c), "c": c, "v": v}


# ------------------------------------------------------------ volatility

def test_a_flat_series_has_no_volatility():
    bars = [bar(100.0) for _ in range(40)]
    assert pj.realised_sigma_bps(bars, 900.0) == 0.0


def test_volatility_scales_per_second():
    """Doubling the bar length halves the per-second figure, near enough."""
    rng = random.Random(3)
    px, bars = 100.0, []
    for _ in range(200):
        px *= 1 + rng.gauss(0, 0.002)
        bars.append(bar(px))
    a = pj.realised_sigma_bps(bars, 900.0)
    b = pj.realised_sigma_bps(bars, 3600.0)
    assert a > 0 and b > 0
    assert b == pytest.approx(a / 2.0, rel=0.02)


def test_too_little_history_is_no_reading_not_a_guess():
    assert pj.realised_sigma_bps([bar(100.0)], 900.0) == 0.0


# ----------------------------------------------------------------- drift

def test_no_read_means_no_drift():
    assert pj.drift_from_read(0.0, 2.0, 900.0) == 0.0


def test_drift_takes_its_sign_from_the_read():
    assert pj.drift_from_read(1.0, 2.0, 900.0) > 0
    assert pj.drift_from_read(-1.0, 2.0, 900.0) < 0


def test_drift_is_squashed_so_a_huge_read_stays_sane():
    small = pj.drift_from_read(1.0, 2.0, 900.0)
    huge = pj.drift_from_read(50.0, 2.0, 900.0)
    assert huge < small * 3          # tanh, not linear


def test_drift_is_relative_to_volatility():
    """The same coefficient must mean the same thing in a quiet market
    and a busy one."""
    quiet = pj.drift_from_read(1.0, 1.0, 900.0)
    busy = pj.drift_from_read(1.0, 4.0, 900.0)
    assert busy == pytest.approx(quiet * 4, rel=1e-9)


def test_the_claim_is_the_same_size_at_every_horizon():
    """The bug that shipped: a per-second drift accumulates linearly in T
    while the noise around it grows only as sqrt(T), so the same
    coefficient claims more and more as the horizon lengthens.

    On a 15m bar it was asserting a 3.4 sigma move — which read as "the
    signal is hurting" even on data where the signal was real, because a
    wildly over-confident drift is worse than none whichever way it
    points.
    """
    k, sigma = 0.08, 2.0
    for T in (60.0, 300.0, 900.0, 3600.0):
        d = pj.drift_from_read(1.5, sigma, T, k)
        in_sd = (d * T) / (sigma * math.sqrt(T))
        assert in_sd == pytest.approx(k * math.tanh(1.0), rel=1e-6), T


def test_the_projection_reports_its_claim_in_sigma():
    out = pj.project(price=100, open_px=100, high_so_far=100,
                     low_so_far=100, remaining_s=600, sigma_bps_s=2.0,
                     net=1.5, seed=3)
    assert abs(out["expected_move_sd"]) < 0.1


# ------------------------------------------------------------ projecting

def test_a_bar_with_no_time_left_is_not_projected():
    out = pj.project(price=100, open_px=100, high_so_far=101,
                     low_so_far=99, remaining_s=0, sigma_bps_s=2.0)
    assert out["ready"] is False and "no time left" in out["detail"]


def test_a_market_with_no_measured_volatility_is_not_projected():
    out = pj.project(price=100, open_px=100, high_so_far=100,
                     low_so_far=100, remaining_s=300, sigma_bps_s=0.0)
    assert out["ready"] is False


def test_the_projected_high_can_only_extend_what_already_printed():
    """The bar's own high is a fact. A projection must never walk it back."""
    out = pj.project(price=100, open_px=99, high_so_far=105, low_so_far=95,
                     remaining_s=300, sigma_bps_s=2.0, seed=1)
    for q in out["signal"]["high"].values():
        assert q >= 105 - 1e-9
    for q in out["signal"]["low"].values():
        assert q <= 95 + 1e-9


def test_the_bands_are_ordered():
    out = pj.project(price=100, open_px=100, high_so_far=100.5,
                     low_so_far=99.5, remaining_s=600, sigma_bps_s=3.0,
                     seed=2)
    c = out["signal"]["close"]
    assert c["q5"] <= c["q25"] <= c["q50"] <= c["q75"] <= c["q95"]


def test_uncertainty_shrinks_as_the_bar_fills():
    """The whole reason completing a bar beats predicting one."""
    wide = pj.project(price=100, open_px=100, high_so_far=100,
                      low_so_far=100, remaining_s=800, sigma_bps_s=2.0,
                      seed=4)["signal"]["close"]
    tight = pj.project(price=100, open_px=100, high_so_far=100,
                       low_so_far=100, remaining_s=30, sigma_bps_s=2.0,
                       seed=4)["signal"]["close"]
    assert (tight["q95"] - tight["q5"]) < (wide["q95"] - wide["q5"]) / 3


def test_a_positive_read_moves_the_middle_up():
    up = pj.project(price=100, open_px=100, high_so_far=100, low_so_far=100,
                    remaining_s=600, sigma_bps_s=2.0, net=2.0, k=0.5,
                    seed=5)
    assert up["signal"]["close"]["q50"] > up["null"]["close"]["q50"]
    assert up["signal"]["p_up"] > up["null"]["p_up"]


def test_the_null_carries_no_opinion():
    out = pj.project(price=100, open_px=100, high_so_far=100,
                     low_so_far=100, remaining_s=600, sigma_bps_s=2.0,
                     net=3.0, k=0.5, seed=6)
    assert out["null"]["p_up"] == pytest.approx(0.5, abs=0.06)


def test_the_two_projections_do_not_share_draws():
    """Sharing a random stream would make the comparison look tighter
    than it is."""
    out = pj.project(price=100, open_px=100, high_so_far=100,
                     low_so_far=100, remaining_s=600, sigma_bps_s=2.0,
                     net=0.0, seed=7)
    assert out["_signal_samples"] != out["_null_samples"]


# --------------------------------------------------------------- scoring

def test_pit_finds_where_the_outcome_landed():
    samples = [float(i) for i in range(100)]
    assert pj.pit(50.0, samples) == pytest.approx(0.5, abs=0.01)
    assert pj.pit(-10.0, samples) == 0.0
    assert pj.pit(1000.0, samples) == 1.0


def test_a_correct_model_produces_flat_pits():
    """The definition of calibrated, checked by construction."""
    rng = random.Random(11)
    pits = []
    for _ in range(600):
        samples = [rng.gauss(0, 1) for _ in range(400)]
        actual = rng.gauss(0, 1)          # drawn from the same law
        pits.append(pj.pit(actual, samples))
    out = pj.calibration_score(pits)
    assert out["ready"] and out["deviation"] < 0.03
    assert "calibrated" in out["verdict"]
    assert out["coverage_50"] == pytest.approx(0.5, abs=0.06)


def test_an_over_confident_model_is_caught():
    """Bands too narrow: outcomes pile up at both ends."""
    rng = random.Random(12)
    pits = []
    for _ in range(600):
        samples = [rng.gauss(0, 0.3) for _ in range(400)]   # too sure
        actual = rng.gauss(0, 1)
        pits.append(pj.pit(actual, samples))
    out = pj.calibration_score(pits)
    assert out["deviation"] > 0.05
    assert "over-confident" in out["verdict"]
    assert out["coverage_50"] < 0.35


def test_a_model_pointing_the_wrong_way_is_caught():
    rng = random.Random(13)
    pits = []
    for _ in range(600):
        samples = [rng.gauss(-1.0, 1) for _ in range(400)]  # biased low
        actual = rng.gauss(0, 1)
        pits.append(pj.pit(actual, samples))
    out = pj.calibration_score(pits)
    assert out["mean_pit"] > 0.56
    assert "leans low" in out["verdict"]


def test_a_thin_sample_says_so_rather_than_scoring_it():
    out = pj.calibration_score([0.5] * 10)
    assert out["ready"] is False and "too few" in out["note"]


def test_the_histogram_is_flat_for_a_good_model():
    rng = random.Random(14)
    pits = [rng.random() for _ in range(2000)]
    rows = pj.reliability(pits)
    assert len(rows) == 10
    for r in rows:
        assert r["share"] == pytest.approx(0.1, abs=0.03)


# ----------------------------------------------- signal against the null

def test_a_signal_with_no_information_reads_as_no_information():
    """The finding that matters, and the one nobody wants: if the drifted
    projection is no better than the flat one, the read carries nothing at
    this horizon."""
    rng = random.Random(15)
    sig, null = [], []
    for _ in range(600):
        actual = rng.gauss(0, 1)
        sig.append(pj.pit(actual, [rng.gauss(0, 1) for _ in range(300)]))
        null.append(pj.pit(actual, [rng.gauss(0, 1) for _ in range(300)]))
    out = pj.compare(sig, null)
    assert out["ready"]
    assert "no directional information" in out["verdict"]


def test_a_signal_that_hurts_is_named_as_hurting():
    rng = random.Random(16)
    sig, null = [], []
    for _ in range(600):
        actual = rng.gauss(0, 1)
        # The "signal" leans hard the wrong way.
        sig.append(pj.pit(actual, [rng.gauss(-1.5, 1) for _ in range(300)]))
        null.append(pj.pit(actual, [rng.gauss(0, 1) for _ in range(300)]))
    out = pj.compare(sig, null)
    assert "WORSE" in out["verdict"] or "hurting" in out["verdict"]


def test_comparing_without_enough_bars_refuses_to_conclude():
    out = pj.compare([0.5] * 5, [0.5] * 5)
    assert out["ready"] is False


# ------------------------------------------------ does it call direction

def _rows(n, right_rate, seed=1):
    import random
    rng = random.Random(seed)
    out = []
    for i in range(n):
        up = rng.random() < 0.5
        correct = rng.random() < right_rate
        moved_up = up if correct else not up
        out.append({"net": 1.0 if up else -1.0, "price": 100.0,
                    "close_px": 100.5 if moved_up else 99.5})
    return out


def test_a_coin_reads_as_a_coin():
    out = pj.directional_edge(_rows(400, 0.5, seed=2))
    assert out["ready"] and out["real"] is False
    assert "does not call direction" in out["verdict"]


def test_a_real_edge_is_found():
    out = pj.directional_edge(_rows(400, 0.62, seed=3))
    assert out["real"] is True and out["ci_low"] > 0.5
    assert "carries direction" in out["verdict"]


def test_an_inverted_signal_is_named_as_inverted():
    """Predicting with the sign backwards is information, not noise, and
    saying 'no edge' would throw it away."""
    out = pj.directional_edge(_rows(400, 0.35, seed=4))
    assert out["real"] is True and out["ci_high"] < 0.5
    assert "sign inverted" in out["verdict"]


def test_a_thin_sample_refuses_to_conclude():
    out = pj.directional_edge(_rows(10, 0.9))
    assert out["ready"] is False and "too few" in out["note"]


def test_bars_where_the_read_was_flat_are_not_counted():
    rows = _rows(100, 0.9) + [{"net": 0.0, "price": 100.0,
                               "close_px": 101.0}] * 500
    out = pj.directional_edge(rows)
    assert out["n"] == 100


def test_direction_is_far_more_sensitive_than_band_shape():
    """The reason both tests are reported.

    A small directional drift barely moves the PIT histogram, so grading
    only the bands and concluding the signal is worthless would be a
    mistake of instrument rather than of data.
    """
    out = pj.directional_edge(_rows(400, 0.58, seed=5))
    assert out["real"] is True          # binomial catches it easily
