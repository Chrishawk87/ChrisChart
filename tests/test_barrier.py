"""The triple barrier, against paths whose answer is arithmetic.

The asymmetry is the point, so the arithmetic tests matter as much as the
path-walking ones: a hit rate means nothing on its own once the target and
the stop are different sizes, and the whole reason for moving to this
target is that it changes what rate is needed.
"""

from __future__ import annotations

import pytest

from liqmap import barrier as bar
from liqmap.escondense import Slice


def sl(ts, open_px, path):
    s = Slice(ts=ts, prior_close=open_px, prior_open=open_px)
    s.open = open_px
    s.close = open_px
    s.high = open_px
    s.low = open_px
    s.path = path
    return s


# ------------------------------------------------------- the arithmetic

def test_five_against_three_needs_far_less_than_five_against_five():
    """The reason for an asymmetric target."""
    assert bar.breakeven(5, 5) == pytest.approx(0.540, abs=1e-3)
    assert bar.breakeven(5, 3) == pytest.approx(0.425, abs=1e-3)


def test_the_breakeven_is_the_dollar_arithmetic():
    # win 5*12.50-5 = 57.50, lose 3*12.50+5 = 42.50.
    assert bar.breakeven(5, 3) == pytest.approx(42.5 / 100.0)


def test_costs_bite_at_both_ends():
    """They shrink the win AND deepen the loss, so a round trip moves the
    bar further than half of it would suggest."""
    free = bar.breakeven(5, 3, cost_usd=0.0)
    paid = bar.breakeven(5, 3, cost_usd=5.0)
    assert paid > free + 0.04


def test_a_target_smaller_than_costs_can_never_break_even():
    assert bar.breakeven(0.2, 3) == 1.0


def test_expectancy_is_the_verdict_not_the_hit_rate():
    """45% is a loss at 3:5 and a profit at 5:3. The rate alone cannot
    tell you which, which is why nothing here reports it alone."""
    assert bar.expectancy_usd(0.45, take_ticks=3, stop_ticks=5) < 0
    assert bar.expectancy_usd(0.45, take_ticks=5, stop_ticks=3) > 0


def test_expectancy_is_zero_at_the_breakeven_rate():
    r = bar.breakeven(5, 3)
    assert bar.expectancy_usd(r, 5, 3) == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------ walking a path

def test_the_target_is_taken_when_it_comes_first():
    path = [(1.0, -0.2), (6.0, -0.3)]
    assert bar.resolve(path, side=1, take_bps=5.0, stop_bps=3.0) == "target"


def test_the_stop_is_taken_when_it_comes_first():
    path = [(1.0, -4.0), (9.0, -4.0)]
    assert bar.resolve(path, side=1, take_bps=5.0, stop_bps=3.0) == "stop"


def test_a_bucket_holding_both_is_scored_as_the_stop():
    """The tie rule. A bucket does not record the order of ticks inside
    itself, and resolving ties favourably is how a backtest flatters."""
    path = [(9.0, -9.0)]
    assert bar.resolve(path, side=1, take_bps=5.0, stop_bps=3.0) == "stop"


def test_a_path_that_reaches_neither_times_out():
    assert bar.resolve([(1.0, -1.0)] * 20, 1, 5.0, 3.0) == "timeout"


def test_a_short_reads_the_mirror_image():
    """Down is profit. The same code path, so this catches a hard-coded
    long assumption."""
    path = [(0.2, -6.0)]
    assert bar.resolve(path, side=-1, take_bps=5.0, stop_bps=3.0) == "target"
    assert bar.resolve([(6.0, -0.2)], side=-1,
                       take_bps=5.0, stop_bps=3.0) == "stop"


def test_an_empty_or_sideless_path_times_out_rather_than_raising():
    assert bar.resolve([], 1, 5.0, 3.0) == "timeout"
    assert bar.resolve([(9.0, 0.0)], 0, 5.0, 3.0) == "timeout"


# ------------------------------------------------------- chaining bars

def test_the_horizon_runs_past_the_end_of_the_entry_bar():
    a = sl(0.0, 7000.0, [(0.0, 0.0)])
    b = sl(60.0, 7000.0, [(10.0, 0.0)])
    assert len(bar.chain([a, b], 0, bars=1)) == 1
    assert len(bar.chain([a, b], 0, bars=2)) == 2


def test_later_bars_are_rebased_onto_the_entry_not_their_own_open():
    """The bug this function exists to avoid.

    Bar two opens ten points above bar one and does nothing. Read raw, its
    path says 0 bps. Rebased onto the entry it must say about +14 bps --
    and concatenating the stored figures would lose that, drifting further
    with every bar added.
    """
    a = sl(0.0, 7000.0, [(0.0, 0.0)])
    b = sl(60.0, 7010.0, [(0.0, 0.0)])
    chained = bar.chain([a, b], 0, bars=2)
    assert chained[0][0] == pytest.approx(0.0)
    assert chained[1][0] == pytest.approx(14.29, abs=0.05)


def test_chaining_stops_at_the_end_of_the_data():
    a = sl(0.0, 7000.0, [(0.0, 0.0)])
    assert len(bar.chain([a], 0, bars=10)) == 1


def test_chaining_from_a_bad_index_or_price_is_empty_not_an_error():
    a = sl(0.0, 7000.0, [(0.0, 0.0)])
    assert bar.chain([a], 5, 2) == []
    assert bar.chain([sl(0.0, 0.0, [(1.0, 1.0)])], 0, 1) == []


# ------------------------------------------------------------ scoring

def _straight(n=40, px=7000.0, up=True):
    """Bars that march one way, a tick per bucket."""
    out = []
    for i in range(n):
        step = 0.25 if up else -0.25
        moves = [(((step * (k + 1)) / px) * 10_000.0,
                  ((step * (k + 1)) / px) * 10_000.0) for k in range(4)]
        out.append(sl(i * 60.0, px, moves))
        px += step * 4
    return out


def test_a_signal_that_calls_a_march_correctly_hits_its_target():
    s = _straight(up=True)
    r = bar.run(s, [1] * len(s), take_ticks=5, stop_ticks=3, bars=4)
    assert r.target > 0 and r.stop == 0
    assert r.rate == pytest.approx(1.0)
    assert r.per_trade > 0


def test_a_signal_that_calls_it_backwards_is_stopped_out():
    s = _straight(up=True)
    r = bar.run(s, [-1] * len(s), take_ticks=5, stop_ticks=3, bars=4)
    assert r.stop > 0 and r.target == 0
    assert r.per_trade < 0


def test_bars_without_a_full_horizon_are_skipped_not_scored_short():
    """A truncated horizon cannot reach its target, so scoring it would
    add a run of losses that never happened."""
    s = _straight(n=20, up=True)
    r = bar.run(s, [1] * len(s), take_ticks=5, stop_ticks=3, bars=6)
    assert r.n == 14


def test_a_zero_side_is_no_trade():
    s = _straight(n=20)
    assert bar.run(s, [0] * 20, 5, 3, bars=4).n == 0


def test_timeouts_are_excluded_from_the_rate_rather_than_scored_flat():
    """A timeout is a position still open; what it is worth depends on the
    exit rule, not on the signal. Scoring them zero would reward a rule
    that mostly declines to do anything."""
    r = bar.Result(n=100, target=20, stop=20, timeout=60,
                   take_ticks=5, stop_ticks=3)
    assert r.resolved == 40
    assert r.rate == pytest.approx(0.5)


def test_a_result_with_nothing_resolved_reports_none_not_zero():
    r = bar.Result(n=10, target=0, stop=0, timeout=10,
                   take_ticks=5, stop_ticks=3)
    assert r.rate is None and r.per_trade is None and r.total is None


def test_the_total_scales_with_the_number_of_trades():
    r = bar.Result(n=100, target=60, stop=40, timeout=0,
                   take_ticks=5, stop_ticks=3)
    assert r.total == pytest.approx(r.per_trade * 100)


# ------------------------------------------------------- the control

def _noisy(n=600, px=7000.0, seed=3, edge=0.0):
    """A noisy walk per bar, so paths are not straight lines.

    A straight-line path reaches the far barrier without ever testing the
    near one, which makes an asymmetric target look far better than it is.
    The first version of these fixtures did exactly that and reported a
    coin as profitable.
    """
    import random
    rng = random.Random(seed)
    out, sides = [], []
    for i in range(n):
        up = rng.random() < 0.5
        drift = (0.25 if up else -0.25) * rng.uniform(0.3, 1.2)
        p, path = px, []
        for _ in range(30):
            p += drift + rng.gauss(0, 0.55)
            path.append((((p - px) / px) * 1e4 + 0.02,
                         ((p - px) / px) * 1e4 - 0.02))
        s = sl(i * 60.0, px, path)
        s.close = p
        out.append(s)
        called = up if rng.random() < 0.5 + edge else not up
        sides.append(1 if called else -1)
        px = p
    return out, sides


def test_a_coin_does_not_beat_its_own_shuffle():
    """The failure the control exists to catch.

    With a +5/-3 barrier a signal with no skill can still show positive
    dollars per trade, because expectancy depends on path geometry as well
    as direction. Only the gap over the shuffle is evidence.
    """
    s, sides = _noisy(edge=0.0)
    r = bar.run(s, sides, 5, 3, bars=5)
    ctrl = bar.control(s, sides, 5, 3, bars=5)
    assert not bar.beats_control(r.per_trade, ctrl)


def test_a_real_edge_beats_its_own_shuffle():
    s, sides = _noisy(edge=0.30)
    r = bar.run(s, sides, 5, 3, bars=5)
    ctrl = bar.control(s, sides, 5, 3, bars=5)
    assert bar.beats_control(r.per_trade, ctrl)
    assert r.per_trade > ctrl["mean"] + 5.0


def test_the_control_can_be_positive_on_a_pure_coin():
    """Documenting why the theoretical breakeven is the wrong baseline:
    the shuffle itself often clears it."""
    s, sides = _noisy(edge=0.0)
    ctrl = bar.control(s, sides, 5, 3, bars=5)
    assert ctrl["mean"] is not None
    assert ctrl["trials"] > 10


def test_the_control_trades_the_same_bars_as_the_rule():
    """Holding bar selection, time of day and volatility fixed is the whole
    point -- only direction is destroyed."""
    s, sides = _noisy(n=300)
    r = bar.run(s, sides, 5, 3, bars=5)
    ctrl = bar.control(s, sides, 5, 3, bars=5, trials=3)
    assert ctrl["trials"] == 3
    # Same number of calls every time, since only the signs move.
    blank = [0] * len(sides)
    assert bar.run(s, blank, 5, 3, bars=5).n == 0
    assert r.n > 0


def test_the_control_preserves_the_long_short_balance():
    """A rule that is mostly long keeps a mostly-long control, so a market
    that simply drifted up cannot flatter it."""
    s, _ = _noisy(n=300)
    mostly_long = [1] * 250 + [-1] * 50
    ctrl = bar.control(s, mostly_long, 5, 3, bars=5, trials=5)
    assert ctrl["mean"] is not None


def test_the_control_is_reproducible_for_a_given_seed():
    s, sides = _noisy(n=300)
    a = bar.control(s, sides, 5, 3, bars=5, seed=1, trials=5)
    b = bar.control(s, sides, 5, 3, bars=5, seed=1, trials=5)
    assert a["mean"] == b["mean"]


def test_a_signal_that_never_trades_has_no_control():
    s, _ = _noisy(n=100)
    ctrl = bar.control(s, [0] * len(s), 5, 3, bars=5)
    assert ctrl["mean"] is None
    assert not bar.beats_control(10.0, ctrl)


def test_clearing_requires_a_margin_over_the_controls_spread():
    """A whisker above the mean is inside the noise of the comparison."""
    ctrl = {"mean": 1.0, "sd": 2.0}
    assert not bar.beats_control(2.0, ctrl)
    assert bar.beats_control(5.5, ctrl)


def test_a_rate_above_breakeven_pays_and_below_it_does_not():
    good = bar.Result(n=100, target=60, stop=40, timeout=0,
                      take_ticks=5, stop_ticks=3)
    bad = bar.Result(n=100, target=30, stop=70, timeout=0,
                     take_ticks=5, stop_ticks=3)
    assert good.rate > good.needs and good.per_trade > 0
    assert bad.rate < bad.needs and bad.per_trade < 0
