"""Volume pace: has this candle got the volume to move, by this point in it.

Three traps, and the first is the one that makes the whole thing worth
building.

A BAR IS NOT QUIET JUST BECAUSE IT IS YOUNG. Six minutes into a fifteen,
comparing what it has done against a FULL bar reports "quiet" every time
until the bar is nearly over. The comparison has to be against what a
normal bar has by this point, which means knowing the shape of volume
inside a bar.

THE SHAPE IS NOT FLAT, and assuming it flat is wrong in a known direction:
the start of every slow bar reads quiet and the end of every one reads
heavy.

VOLUME ALONE NEVER SAYS MOVE-OR-HOLD. The same heavy volume means opposite
things depending on what it bought -- a wide range is a move being paid
for, no range is that move being absorbed.
"""

from __future__ import annotations

import pytest

from liqmap import vpace as vp


def flat_bar(slices=12, each=10.0):
    return [each] * slices


def front_bar(slices=12, heavy=4.0):
    """Half the volume in the first quarter."""
    n = max(1, slices // 4)
    return [heavy] * n + [1.0] * (slices - n)


class T:
    def __init__(self, ts, notional):
        self.ts, self.notional = float(ts), float(notional)


class M:
    """A one-minute record, as the bar store hands them back."""

    def __init__(self, bar_ts, total):
        self.bar_ts, self.total = float(bar_ts), float(total)


# ------------------------------------------------------- learning the shape

def test_an_even_market_learns_a_straight_line():
    c = vp.curve_from_bars("1m", 60.0, [flat_bar() for _ in range(20)])
    assert c.learned
    assert c.share_by(30.0) == pytest.approx(0.5, abs=0.02)
    assert c.share_by(60.0) == pytest.approx(1.0)


def test_a_front_loaded_market_is_not_reported_as_even():
    """THE reason the shape is learned. A quarter of the way into a bar
    that normally does half its volume there, an even spread expects 25%
    and sees 50%, and reports a perfectly ordinary bar as twice normal."""
    c = vp.curve_from_bars("15m", 900.0, [front_bar() for _ in range(20)])
    assert c.learned
    early = c.share_by(225.0)             # a quarter of the way in
    assert early > 0.4
    assert early > vp.even_curve("15m", 900.0).share_by(225.0) * 1.5


def test_each_bar_is_normalised_before_the_bars_are_combined():
    """One enormous bar must not set the shape for all of them. Here the
    giant is back-loaded and everything else is front-loaded; the median
    of SHARES still describes the habit, not the giant."""
    ordinary = [front_bar() for _ in range(19)]
    giant = [[1.0] * 9 + [10_000.0] * 3]
    c = vp.curve_from_bars("15m", 900.0, ordinary + giant)
    assert c.share_by(225.0) > 0.4


def test_the_median_stops_one_bar_bending_the_shape():
    odd = [[0.0] * 11 + [100.0] for _ in range(3)]
    normal = [flat_bar() for _ in range(17)]
    c = vp.curve_from_bars("1m", 60.0, normal + odd)
    assert c.share_by(30.0) == pytest.approx(0.5, abs=0.05)


def test_a_cumulative_share_never_goes_backwards_or_misses_the_end():
    """Asserted rather than enforced.

    It holds by construction: every row is a cumulative share of its own
    bar, so every row is non-decreasing and ends at exactly 1.0, and an
    order statistic of non-decreasing sequences is non-decreasing. A
    fix-up pass in the builder would be dead code pretending to hold the
    invariant up -- mutating it away changes nothing, which is how we
    know. So the test guards the construction instead, and a change that
    broke it would fail here.
    """
    noisy = [[1.0, 9.0, 1.0, 9.0, 1.0, 9.0] for _ in range(5)]
    noisy += [[9.0, 1.0, 9.0, 1.0, 9.0, 1.0] for _ in range(5)]
    noisy += [[0.0, 0.0, 50.0, 0.0, 0.0, 1.0] for _ in range(4)]
    c = vp.curve_from_bars("1m", 60.0, noisy)
    assert all(b >= a for a, b in zip(c.shares, c.shares[1:]))
    assert c.shares[-1] == pytest.approx(1.0)
    assert c.share_by(60.0) == pytest.approx(1.0)


def test_too_few_bars_assumes_an_even_spread_and_says_so():
    c = vp.curve_from_bars("4h", 14400.0, [flat_bar() for _ in range(3)])
    assert not c.learned
    assert c.source == "even"
    assert "learn" in c.note


def test_bars_with_no_volume_in_them_are_not_counted_as_shape():
    c = vp.curve_from_bars("1m", 60.0, [[0.0] * 12 for _ in range(20)])
    assert not c.learned


# -------------------------------------------- where a fast bar's shape comes from

def test_a_fast_bars_shape_comes_from_the_fills_themselves():
    """Nothing finer than a one-minute bar is stored anywhere -- but the
    fills ARE finer than any bar, and an hour of them is in the tape. A
    one-minute bar cuts into twelve five-second slices with sixty
    completed bars to learn from."""
    trades = []
    for bar in range(30):                      # 30 completed minutes
        base = bar * 60.0
        for i in range(12):
            # front-loaded: the first quarter carries most of it
            size = 40.0 if i < 3 else 2.0
            trades.append(T(base + i * 5.0 + 1.0, size))
    sliced = vp.slices_from_tape(trades, 60.0, now=30 * 60.0 + 20.0)
    assert len(sliced) >= 25
    c = vp.curve_from_bars("1m", 60.0, sliced, source="tape")
    assert c.learned and c.source == "tape"
    assert c.share_by(15.0) > 0.6


def test_the_bar_in_progress_is_left_out_of_its_own_shape():
    """A partial bar normalised against its own partial total reports a
    completed bar's shape every time, and it would be the loudest sample
    in the set."""
    trades = [T(i * 5.0 + 1.0, 10.0) for i in range(12)]      # bar 0
    trades += [T(60.0 + i * 5.0, 10.0) for i in range(12)]    # bar 1, whole
    trades += [T(120.0 + 1.0, 500.0)]                         # bar 2, partial
    sliced = vp.slices_from_tape(trades, 60.0, now=130.0)
    # Bar 1 only: bar 2 is in progress, and bar 0 is the one the tape's
    # earliest fill lands inside, so it may itself be partial.
    assert len(sliced) == 1
    assert sum(sliced[0]) == pytest.approx(120.0)


def test_the_first_bar_the_tape_saw_is_dropped_as_possibly_partial():
    """There is no way to tell from here whether the tape was listening
    at that bar's open. It would look back-loaded when it was not."""
    trades = [T(i * 5.0 + 1.0, 10.0) for i in range(12)]
    assert vp.slices_from_tape(trades, 60.0, now=130.0) == []


def test_a_bar_the_tape_only_caught_the_end_of_is_dropped():
    """It would look back-loaded, and it was not -- the tape just arrived
    late."""
    trades = [T(45.0, 10.0)]                       # bar 0, tail only
    trades += [T(60.0 + i * 5.0, 10.0) for i in range(12)]    # bar 1, whole
    sliced = vp.slices_from_tape(trades, 60.0, now=130.0)
    assert len(sliced) == 1


def test_an_empty_tape_is_not_a_crash():
    assert vp.slices_from_tape([], 60.0, now=0.0) == []
    assert vp.slices_from_tape(None, 60.0, now=0.0) == []


# -------------------------------------------- where a slow bar's shape comes from

def test_a_slow_bars_shape_comes_from_the_minutes_inside_it():
    rows = []
    for bar in range(12):
        for m in range(15):
            rows.append(M(bar * 900.0 + m * 60.0, 50.0 if m < 4 else 5.0))
    sliced = vp.slices_from_minutes(rows, 900.0)
    assert len(sliced) == 12
    c = vp.curve_from_bars("15m", 900.0, sliced)
    assert c.learned
    assert c.share_by(240.0) > 0.5          # four minutes in


def test_a_bar_missing_a_minute_is_dropped_not_filled_in():
    """A gap reads as a quiet patch inside the bar and bends the shape
    towards whatever the feed happened to miss."""
    rows = [M(m * 60.0, 10.0) for m in range(15) if m != 7]
    rows += [M(900.0 + m * 60.0, 10.0) for m in range(15)]
    assert len(vp.slices_from_minutes(rows, 900.0)) == 1


def test_the_slices_divide_the_bar_into_whole_minutes():
    """A slice boundary inside a minute cannot be built from minute bars."""
    for interval, minutes in ((900.0, 15), (1800.0, 30), (3600.0, 60),
                              (14400.0, 240)):
        n = vp.minute_slices(interval)
        assert n > 1
        assert minutes % n == 0


def test_a_one_minute_bar_has_no_minute_sliced_shape():
    """Which is exactly why the fast path reads the tape instead."""
    assert vp.minute_slices(60.0) == 0


# --------------------------------------------------------------- the pace

def kit(done, normal=1000.0, elapsed=30.0, interval=60.0, curve=None,
        range_bps=5.0, normal_range=10.0):
    return vp.VolumePace(timeframe="1m", interval_s=interval, done=done,
                         normal_full=normal, elapsed_s=elapsed,
                         range_bps=range_bps, normal_range_bps=normal_range,
                         curve=curve)


def test_a_young_bar_is_judged_against_a_young_bars_volume():
    """Half a normal bar's volume, halfway through. That is exactly
    normal, and against a full bar it would read 0.5x and look dead."""
    assert kit(done=500.0).pace == pytest.approx(1.0, abs=0.01)


def test_the_pace_projects_where_the_bar_finishes():
    p = kit(done=1000.0)                 # a full bar's volume, halfway
    assert p.projected == pytest.approx(2000.0, rel=0.01)
    assert p.projected_x == pytest.approx(2.0, rel=0.01)


def test_the_learned_shape_changes_the_verdict():
    """The same bar, the same volume, read against the two yardsticks.
    Front-loaded market: an even spread calls this heavy, the learned
    shape calls it ordinary."""
    curve = vp.curve_from_bars("15m", 900.0,
                               [front_bar() for _ in range(20)])
    done, elapsed = 500.0, 225.0
    even = kit(done, normal=1000.0, elapsed=elapsed, interval=900.0)
    learned = kit(done, normal=1000.0, elapsed=elapsed, interval=900.0,
                  curve=curve)
    assert even.pace > 1.8
    assert learned.pace < 1.3


def test_no_yardstick_means_unknown_rather_than_zero():
    p = kit(done=500.0, normal=0.0)
    assert not p.known
    assert p.state == "unknown"
    assert "yardstick" in p.describe()


# ------------------------------------------------- effort against result

def test_heavy_volume_and_a_wide_range_is_the_move_being_paid_for():
    p = kit(done=1500.0, range_bps=20.0, normal_range=10.0)
    assert p.heavy and p.wide
    assert p.state == "paid_for"


def test_heavy_volume_and_no_range_is_absorption_not_a_move():
    """The same volume saying the opposite thing. Reading it as fuel for
    a move is the expensive way to get this wrong."""
    p = kit(done=1500.0, range_bps=0.3, normal_range=10.0)
    assert p.heavy and not p.wide
    assert p.state == "absorbed"


def test_a_wide_range_on_light_volume_is_thin_not_strong():
    p = kit(done=200.0, range_bps=20.0, normal_range=10.0)
    assert p.quiet and p.wide
    assert p.state == "thin"


def test_light_volume_and_no_range_is_nothing_happening():
    p = kit(done=200.0, range_bps=0.2, normal_range=10.0)
    assert p.state == "no_fuel"
    assert "nothing is happening" in vp.MEANING[p.state]


def test_a_young_bar_is_not_expected_to_have_a_full_bars_range():
    """The range yardstick scales with elapsed for the same reason the
    volume one does, and by the square root for the same reason again."""
    young = kit(done=1500.0, elapsed=3.0, range_bps=3.0, normal_range=10.0)
    old = kit(done=1500.0, elapsed=60.0, range_bps=3.0, normal_range=10.0)
    assert young.wide
    assert not old.wide


def test_efficiency_is_range_bought_per_unit_of_normal_volume():
    easy = kit(done=500.0, range_bps=20.0)
    grinding = kit(done=1500.0, range_bps=20.0)
    assert easy.efficiency > grinding.efficiency


def test_every_state_has_something_to_say():
    for s in ("paid_for", "absorbed", "thin", "no_fuel", "unknown"):
        assert len(vp.MEANING[s]) > 20


def test_the_reading_says_which_yardstick_it_used():
    curve = vp.curve_from_bars("1m", 60.0, [flat_bar() for _ in range(20)])
    assert "shape" in kit(done=500.0, curve=curve).describe()
    assert "even spread" in kit(done=500.0).describe()


# ------------------------------------------------- on the page, not gating


def _src():
    import inspect

    from liqmap import web as w
    return inspect.getsource(w)


def test_the_volume_read_is_reported_and_gates_nothing():
    """A quiet bar about to break out would go blank exactly when it
    mattered, so nothing on the row is hidden by a low reading."""
    src = _src()
    cell = src[src.index("function fuelCell(name, v, call) {"):]
    cell = cell[:cell.index("\nfunction renderLadder(")]
    assert "v.known" in cell
    # The direction half is built without ever consulting the volume.
    read = src[src.index("function readHalf(t) {"):]
    read = read[:read.index("\nfunction flowHalf(")]
    assert "volume" not in read


def test_the_gauge_sits_with_the_candle_not_inside_one_read():
    """Both halves are looking at the same bar. Putting the volume inside
    one of them would imply it belonged to that read."""
    src = _src()
    row = src[src.index("function renderLadder() {"):]
    row = row[:row.index("\nfunction paintLadder(")]
    assert "fuelCell(n, t && t.volume," in row


def test_the_row_shows_effort_next_to_its_result():
    src = _src()
    cell = src[src.index("function fuelCell(name, v, call) {"):]
    cell = cell[:cell.index("\nfunction renderLadder(")]
    for tag in ("paid", "held", "thin", "none"):
        assert f"'{tag}'" in cell


def test_a_fast_timeframe_is_curved_from_the_tape_and_a_slow_one_from_bars():
    src = _src()
    fn = src[src.index("def _build_curve("):]
    fn = fn[:fn.index("\n    def ")]
    assert "slices_from_tape" in fn and "slices_from_minutes" in fn
    assert fn.index("slices_from_tape") < fn.index("slices_from_minutes")
    assert "FAST_CURVE_S" in fn
    assert "even_curve" in fn, "with neither source it must say so"


def test_the_curve_is_cached_rather_than_rebuilt_every_poll():
    """The tape path walks every fill the tape holds, and the dashboard
    polls this several times a minute across five timeframes."""
    src = _src()
    fn = src[src.index("def volume_curve("):]
    fn = fn[:fn.index("\n    def ")]
    assert "CURVE_TTL_S" in fn and "_curves" in fn


def test_an_aggregated_reading_gets_no_pace():
    """Its volume spans a window while the shape describes one bar, and
    dividing one by the other would be a number rather than a
    measurement."""
    src = _src()
    loop = src[src.index("for pres in readings:"):]
    loop = loop[:loop.index("out[\"timeframes\"]")]
    assert "pres.span_bars > 1" in loop
    assert "continue" in loop


def test_the_range_yardstick_is_the_range_not_the_move():
    """A bar can travel a long way and come back, so its range is always
    at least its move and usually more. Judging 'has it gone anywhere'
    against the move median would call ordinary bars wide."""
    from liqmap.pressure import typical, typical_range
    from liqmap.structure import Candle

    bars = [Candle(ts=i * 60.0, open=100.0, high=101.0, low=99.0,
                   close=100.0, volume=5.0) for i in range(10)]
    moves, _ = typical(bars)
    assert moves == pytest.approx(0.0)
    assert typical_range(bars) == pytest.approx(200.0, rel=0.01)
