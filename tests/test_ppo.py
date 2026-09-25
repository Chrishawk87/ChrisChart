"""Chris's Tick Counter PPO, checked against hand-computed values.

An indicator is the worst kind of thing to get subtly wrong: a line that
is 10% off still looks like a line, still crosses its signal, and still
produces something you would trade on. So the pieces are checked against
numbers worked out by hand rather than against each other.
"""

from __future__ import annotations

import math

import pytest

from liqmap import ppo


def bar(o, h, l, c, v):
    return {"o": o, "h": h, "l": l, "c": c, "v": v}


# ------------------------------------------------- money flow volume

def test_a_close_on_the_high_is_the_whole_volume():
    # (2*10 - 0 - 10) / (10 - 0) = 1.0, times 100.
    assert ppo.money_flow_volume(bar(5, 10, 0, 10, 100)) == pytest.approx(100)


def test_a_close_on_the_low_is_minus_the_whole_volume():
    assert ppo.money_flow_volume(bar(5, 10, 0, 0, 100)) == pytest.approx(-100)


def test_a_close_in_the_middle_is_nothing():
    assert ppo.money_flow_volume(bar(5, 10, 0, 5, 100)) == pytest.approx(0)


def test_a_close_three_quarters_up_is_half_the_volume():
    # (2*7.5 - 0 - 10) / 10 = 0.5
    assert ppo.money_flow_volume(bar(5, 10, 0, 7.5, 80)) == pytest.approx(40)


def test_a_bar_with_no_range_is_zero_not_a_division_by_zero():
    assert ppo.money_flow_volume(bar(5, 5, 5, 5, 100)) == 0.0


def test_volume_scales_it():
    a = ppo.money_flow_volume(bar(5, 10, 0, 10, 10))
    b = ppo.money_flow_volume(bar(5, 10, 0, 10, 1000))
    assert b == pytest.approx(a * 100)


# ------------------------------------------------------ signed volume

def test_volume_is_signed_by_the_close_against_the_last_one():
    bars = [bar(1, 1, 1, 10, 5), bar(1, 1, 1, 11, 7), bar(1, 1, 1, 9, 3),
            bar(1, 1, 1, 9, 4)]
    assert ppo.signed_volume(bars) == [0.0, 7.0, -3.0, 0.0]


def test_the_first_bar_has_nothing_to_compare_against():
    assert ppo.signed_volume([bar(1, 1, 1, 10, 99)]) == [0.0]


# ------------------------------------------------------------- stdev

def test_stdev_is_none_until_the_window_is_full():
    out = ppo.stdev([1, 2, 3, 4], 3)
    assert out[0] is None and out[1] is None
    assert out[2] is not None


def test_stdev_is_the_population_one_as_pine_uses():
    # mean 2, deviations -1/0/1, population variance 2/3.
    out = ppo.stdev([1, 2, 3], 3)
    assert out[2] == pytest.approx(math.sqrt(2 / 3))
    # The sample version would be 1.0; being wrong here scales the whole
    # indicator by a constant that looks like nothing.
    assert out[2] != pytest.approx(1.0)


def test_a_flat_window_has_no_spread():
    assert ppo.stdev([5, 5, 5, 5], 4)[3] == 0.0


# --------------------------------------------------------------- ema

def test_a_one_length_ema_is_the_input():
    xs = [1.0, 2.0, 3.0]
    assert ppo.ema(xs, 1) == xs


def test_the_ema_is_pines_recursive_one():
    # alpha = 2/(3+1) = 0.5, seeded on the first value.
    out = ppo.ema([1.0, 3.0, 5.0], 3)
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(0.5 * 3 + 0.5 * 1)      # 2.0
    assert out[2] == pytest.approx(0.5 * 5 + 0.5 * 2)      # 3.5


def test_leading_gaps_are_carried_not_counted_as_zero():
    out = ppo.ema([None, None, 4.0, 4.0], 3)
    assert out[0] is None and out[1] is None
    assert out[2] == pytest.approx(4.0)
    assert out[3] == pytest.approx(4.0)


# ------------------------------------------------------ normalisation

def test_normalising_bounds_the_result():
    xs = [0.0] * 9 + [1000.0]
    out = ppo._norm(xs, 10, 3.0)
    assert out[9] == pytest.approx(3.0)


def test_a_window_that_never_moved_has_no_reading():
    """Zero spread makes the ratio undefined, not infinite."""
    out = ppo._norm([5.0] * 10, 10, 3.0)
    assert out[9] is None


# ------------------------------------------------------- the whole thing

def _series(n=120, seed=5):
    import random
    rng = random.Random(seed)
    out, px = [], 100.0
    for i in range(n):
        o = px
        c = o * (1 + rng.gauss(0.0003, 0.002))
        hi = max(o, c) * (1 + abs(rng.gauss(0, 0.0008)))
        lo = min(o, c) * (1 - abs(rng.gauss(0, 0.0008)))
        out.append(bar(o, hi, lo, c, 50 + rng.random() * 100))
        px = c
    return out


def test_every_series_matches_the_bar_count():
    bars = _series()
    out = ppo.tick_ppo(bars)
    for k in ("ppo", "signal", "hist", "ad_norm", "mom_norm"):
        assert len(out[k]) == len(bars), k


def test_nothing_is_reported_before_the_window_fills():
    bars = _series(n=60)
    out = ppo.tick_ppo(bars, stdev_len=50)
    assert all(x is None for x in out["ppo"][:48])
    assert out["ppo"][-1] is not None


def test_a_short_series_reports_that_it_is_not_ready():
    out = ppo.tick_ppo(_series(n=10), stdev_len=50)
    assert out["ready"] is False
    assert all(x is None for x in out["hist"])


def test_an_empty_series_does_not_raise():
    out = ppo.tick_ppo([])
    assert out["ppo"] == [] and out["ready"] is False


def test_the_histogram_is_the_line_minus_its_signal():
    bars = _series()
    out = ppo.tick_ppo(bars)
    for p, s, h in zip(out["ppo"], out["signal"], out["hist"]):
        if p is None or s is None:
            assert h is None
        else:
            # Each series is rounded to four places on its own, so the
            # difference of two rounded numbers can sit 1e-4 from the
            # rounded difference.
            assert h == pytest.approx(p - s, abs=2e-4)


def test_smoothing_of_one_leaves_the_line_as_the_combination():
    """The original's default. Smoothing by one must not change anything."""
    bars = _series()
    out = ppo.tick_ppo(bars, smooth=1)
    for a, m, p in zip(out["ad_norm"], out["mom_norm"], out["ppo"]):
        if a is None and m is None:
            assert p is None
        else:
            assert p == pytest.approx((a or 0) * 0.7 + (m or 0) * 0.3,
                                      abs=1e-4)


def test_the_weights_are_seventy_thirty_and_they_move_it():
    bars = _series()
    a = ppo.tick_ppo(bars, ad_weight=1.0, mom_weight=0.0)
    b = ppo.tick_ppo(bars, ad_weight=0.0, mom_weight=1.0)
    assert a["ppo"][-1] != b["ppo"][-1]
    assert a["settings"]["ad_weight"] == 1.0


def test_everything_is_bounded_by_the_clamp():
    bars = _series()
    out = ppo.tick_ppo(bars, clamp=3.0)
    for x in out["ppo"]:
        if x is not None:
            assert -3.0 <= x <= 3.0


def test_an_unvarying_series_has_no_reading_at_all():
    """Not a bug — the normalisation is relative.

    Identical bars have zero standard deviation, so the ratio is
    undefined. A line pinned at zero would be a claim about a market that
    has told us nothing.
    """
    bars = [bar(100, 101, 99, 101, 100.0) for _ in range(120)]
    out = ppo.tick_ppo(bars)
    assert out["ppo"][-1] is None
    assert out["ready"] is False


def test_a_run_of_closes_on_the_high_reads_positive():
    """The direction has to be right, or every sign on the panel is.

    Volume varies so the window has a spread to normalise against; the
    closes stay on their highs so the sign cannot be in doubt.
    """
    import random
    rng = random.Random(1)
    bars, px = [], 100.0
    for i in range(120):
        o = px
        c = o * (1 + 0.0005 + rng.random() * 0.001)
        bars.append(bar(o, c, o, c, 50 + rng.random() * 100))
        px = c
    out = ppo.tick_ppo(bars)
    assert out["ppo"][-1] > 0


def test_a_run_of_closes_on_the_low_reads_negative():
    import random
    rng = random.Random(2)
    bars, px = [], 100.0
    for i in range(120):
        o = px
        c = o * (1 - 0.0005 - rng.random() * 0.001)
        bars.append(bar(o, o, c, c, 50 + rng.random() * 100))
        px = c
    out = ppo.tick_ppo(bars)
    assert out["ppo"][-1] < 0


def test_it_reads_a_candle_object_as_well_as_a_dict():
    from liqmap.structure import Candle
    objs = [Candle(ts=float(i), open=100, high=101, low=99, close=101,
                   volume=10) for i in range(60)]
    out = ppo.tick_ppo(objs)
    assert len(out["ppo"]) == 60


def test_the_cumulative_in_the_original_cancels_out():
    """`cum(x) - cum(x)[1]` is `x`.

    The Pine builds a running total and then subtracts the same total one
    bar back. Translating that literally would carry a cumulative sum that
    contributes nothing, and would invite someone later to 'fix' the
    indicator by removing the subtraction instead.
    """
    bars = _series(n=30)
    per_bar = [ppo.money_flow_volume(b) for b in bars]
    cum, running = [], 0.0
    for x in per_bar:
        running += x
        cum.append(running)
    differenced = [cum[0]] + [cum[i] - cum[i - 1] for i in range(1, len(cum))]
    assert differenced == pytest.approx(per_bar)
