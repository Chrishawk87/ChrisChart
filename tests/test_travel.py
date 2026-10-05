"""Which way the on-candle move is travelling.

The level and its direction of travel are different questions, and only the
second is about right now. +3t rising and +3t falling are the same number
and opposite situations, which is why a snapshot could never say it.

Four things would make this lie, and each has a test that fails if the code
does the convenient thing:

    measuring across a bar roll, which reports the biggest retracement of
    the day every single bar;

    counting samples instead of seconds, so the same slope means ten
    seconds on a quiet tape and two on a busy one;

    calling any non-zero change a direction, when price always moves a
    little;

    reading the slope alone and missing a spike that happened between two
    samples -- which is what the giveback is for.
"""

from __future__ import annotations

import pytest

from liqmap import travel as tv


def track(points, timeframe="15m", bar_ts=0.0):
    t = tv.Track(timeframe=timeframe)
    for ts, move in points:
        t.add(ts, move, bar_ts)
    return t


# ----------------------------------------------------------- the direction

def test_a_rising_move_on_an_up_candle_is_extending():
    r = track([(0.0, 1.0), (2.0, 3.0), (4.0, 5.0), (6.0, 7.0)]).read(now=6.0)
    assert r.state == "extending"
    assert r.extension > 0


def test_a_falling_move_on_an_up_candle_is_a_pullback():
    """The same +3 reading, going the other way. A snapshot cannot tell
    these two apart and they are opposite situations."""
    r = track([(0.0, 9.0), (2.0, 7.0), (4.0, 5.0), (6.0, 3.0)]).read(now=6.0)
    assert r.state == "retracing"
    assert r.extension < 0


def test_a_falling_move_on_a_DOWN_candle_is_also_extending():
    """-3 going to -5 is extending exactly as much as +3 going to +5, and
    a reader should not have to work out the sign of the sign."""
    r = track([(0.0, -1.0), (2.0, -3.0), (4.0, -5.0),
               (6.0, -7.0)]).read(now=6.0)
    assert r.state == "extending"
    assert r.extension > 0
    assert r.change < 0          # the raw change is still negative


def test_a_rising_move_on_a_down_candle_is_a_pullback():
    r = track([(0.0, -9.0), (2.0, -7.0), (4.0, -5.0),
               (6.0, -3.0)]).read(now=6.0)
    assert r.state == "retracing"


# ------------------------------------------------------------- the bar roll

def test_a_new_candle_starts_a_new_track():
    """THE trap. The move resets to about zero when the candle rolls, so a
    slope measured across the boundary reports the biggest retracement of
    the day every single bar."""
    t = tv.Track(timeframe="15m")
    for ts, m in ((0.0, 8.0), (2.0, 9.0), (4.0, 10.0)):
        t.add(ts, m, bar_ts=0.0)
    t.add(6.0, 0.2, bar_ts=900.0)          # the roll
    t.add(8.0, 0.6, bar_ts=900.0)
    t.add(10.0, 1.1, bar_ts=900.0)
    r = t.read(now=10.0)
    assert r.bar_ts == 900.0
    assert r.samples == 3
    assert r.was == pytest.approx(0.2)
    assert r.state != "retracing"


def test_the_roll_is_detected_by_the_bar_not_by_the_drop():
    """A genuine collapse inside one candle must still read as a
    retracement -- it is only a reset when the BAR changed."""
    t = track([(0.0, 10.0), (2.0, 6.0), (4.0, 0.2)])
    assert t.read(now=4.0).state == "retracing"


# --------------------------------------------------------- the clock, not N

def test_the_window_is_wall_clock_rather_than_a_number_of_samples():
    """A busy tape produces many more reads per second. Counting samples
    would make the same slope mean different things exactly when it
    mattered."""
    busy = track([(i * 0.2, i * 0.2) for i in range(60)])      # 12s, 60 reads
    r = busy.read(now=12.0, window_s=6.0)
    assert r.span_s == pytest.approx(6.0, abs=0.3)
    assert r.samples < 40


def test_the_span_actually_covered_is_reported():
    r = track([(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]).read(now=2.0)
    assert r.span_s == pytest.approx(2.0)


def test_a_slow_poll_still_reads_rather_than_going_blank():
    """Ten seconds apart with a six second window: rather than report
    nothing, it reaches back one more sample and says how far it had to
    reach."""
    r = track([(0.0, 2.0), (10.0, 9.0)]).read(now=10.0, window_s=6.0)
    assert r.samples == 2
    assert r.span_s == pytest.approx(10.0)


def test_too_little_watched_says_so_rather_than_guessing():
    r = track([(0.0, 1.0), (0.3, 2.0)]).read(now=0.3)
    assert not r.enough
    assert r.state == "unknown"
    assert "not enough" in r.describe()


def test_a_single_read_is_not_a_slope():
    r = track([(0.0, 5.0)]).read(now=0.0)
    assert r.state == "unknown"
    assert r.samples == 1


# ------------------------------------------------------------ the flat band

def test_a_change_under_one_increment_is_not_a_direction():
    """Price always moves a little. Below one tick, one point or one pip,
    nothing has happened."""
    r = track([(0.0, 3.0), (2.0, 3.2), (4.0, 3.4), (6.0, 3.6)]).read(now=6.0)
    assert abs(r.change) < 1.0
    assert r.state == "flat"


def test_one_whole_increment_is_a_direction():
    r = track([(0.0, 3.0), (2.0, 3.6), (4.0, 4.1)]).read(now=4.0)
    assert r.state == "extending"


# ------------------------------------------------- the giveback, which is exact

def test_a_candle_well_off_its_best_is_retracing_even_on_a_flat_slope():
    """The slope can miss a spike that happened between two samples. The
    giveback cannot: it comes from the candle's own high."""
    r = track([(0.0, 3.0), (2.0, 3.1), (4.0, 3.0),
               (6.0, 3.05)]).read(now=6.0, best=9.0)
    assert abs(r.change) < 1.0
    assert r.giveback == pytest.approx(5.95, abs=0.01)
    assert r.giveback_share > 0.6
    assert r.state == "retracing"


def test_a_small_giveback_does_not_overrule_a_flat_slope():
    r = track([(0.0, 9.0), (2.0, 9.1), (4.0, 9.0)]).read(now=4.0, best=9.5)
    assert r.state == "flat"


def test_the_giveback_is_measured_from_the_candles_own_extreme():
    assert tv.best_of(100.0, 109.0, 99.0, 103.0) == pytest.approx(9.0)
    assert tv.best_of(100.0, 101.0, 91.0, 97.0) == pytest.approx(-9.0)


def test_a_candle_sitting_on_its_open_takes_the_further_extreme():
    """It has no direction yet, so the one that matters is the one it
    would have to give back."""
    assert tv.best_of(100.0, 102.0, 95.0, 100.0) == pytest.approx(-5.0)


def test_a_candle_that_never_moved_has_nothing_to_give_back():
    r = track([(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]).read(now=4.0, best=0.0)
    assert r.giveback == 0.0
    assert r.giveback_share == 0.0
    assert r.state == "flat"


# ---------------------------------------------------------------- the rate

def test_the_rate_is_per_minute_from_the_span_it_measured():
    r = track([(0.0, 0.0), (30.0, 5.0)]).read(now=30.0, window_s=60.0)
    assert r.per_min == pytest.approx(10.0, rel=0.01)


def test_no_span_is_not_a_division_by_zero():
    r = track([(5.0, 1.0), (5.0, 2.0), (5.0, 3.0)]).read(now=5.0)
    assert r.per_min == 0.0


# --------------------------------------------------------------- the tracker

def test_two_timeframes_of_one_market_are_two_different_moves():
    t = tv.Tracker()
    t.record("BTC", "1m", 0.0, 1.0, 0.0)
    t.record("BTC", "15m", 0.0, 9.0, 0.0)
    assert len(t) == 2
    assert t.read("BTC", "1m").now == 1.0
    assert t.read("BTC", "15m").now == 9.0


def test_a_timeframe_never_recorded_reads_as_nothing():
    assert tv.Tracker().read("BTC", "4h") is None


def test_switching_market_can_drop_the_old_ones_tracks():
    t = tv.Tracker()
    t.record("BTC", "1m", 0.0, 1.0, 0.0)
    t.record("ETH", "1m", 0.0, 1.0, 0.0)
    assert t.forget("BTC") == 1
    assert t.read("BTC", "1m") is None
    assert t.read("ETH", "1m") is not None


def test_the_track_does_not_grow_without_limit():
    """A tab left open overnight at two reads a second would otherwise
    hold a quarter of a million points per timeframe."""
    t = tv.Track(timeframe="1m")
    for i in range(5_000):
        t.add(i * 0.5, float(i), bar_ts=0.0)
    assert len(t.points) <= tv.MAX_SAMPLES


def test_every_state_says_what_it_means():
    for s in ("extending", "retracing", "flat", "unknown"):
        assert len(tv.MEANING[s]) > 20


# ------------------------------------------------- on the page and the route


def _src():
    import inspect

    from liqmap import web as w
    return inspect.getsource(w)


def test_the_row_keeps_bps_as_the_reading():
    """I replaced the bps figure with ticks and an extending/retracing
    word, and the row got worse and read wrong. bps is what every
    threshold in this project is expressed in and what makes two markets
    comparable; the tick count is an addition beside it, never a swap.

    This test exists so that mistake cannot be made twice.
    """
    src = _src()
    fn = src[src.index("function moveText(t) {"):]
    fn = fn[:fn.index("\n}")]
    assert "toFixed(2) + 'bps'" in fn, "bps stopped being the reading"
    assert "t.move_units" in fn, "the tick count is not beside it"
    for gone in ("'ext'", "'back'", "u.state", "u.extension"):
        assert gone not in fn, (
            "the direction-of-travel wording is back on the row")


def test_the_tick_count_is_the_same_move_not_a_different_one():
    """It is the bps figure converted, so the two can never disagree."""
    src = _src()
    i = src.index('"move_units"')
    assert "unit.of(r.last_px - r.open_px)" in src[i:i + 80]
    assert '"unit_label"' in src


def test_the_route_records_the_move_before_reading_the_trend():
    """Reading before recording would always be one sample behind, which
    on a fast tape is the whole window."""
    src = _src()
    block = src[src.index("travels: dict[str, Any] = {}"):]
    block = block[:block.index("paces: dict[str, Any] = {}")]
    assert block.index("rt.travel.record(") < block.index("rt.travel.read(")


def test_the_track_is_keyed_to_the_bar_so_a_roll_clears_it():
    src = _src()
    block = src[src.index("travels: dict[str, Any] = {}"):]
    block = block[:block.index("paces: dict[str, Any] = {}")]
    assert "bar_open" in block
    assert "now_s % pres.interval_s" in block


def test_the_best_point_comes_from_the_candles_own_extremes():
    src = _src()
    block = src[src.index("travels: dict[str, Any] = {}"):]
    block = block[:block.index("paces: dict[str, Any] = {}")]
    assert "tvl.best_of(" in block
    assert "pres.high_px" in block and "pres.low_px" in block


def test_the_move_is_emitted_in_the_markets_own_unit():
    src = _src()
    i = src.index('"move_units"')
    assert "unit.of(" in src[i:i + 80]
    assert '"unit"' in src


def test_the_read_follows_the_chart_timeframe():
    """Two candle selectors on one screen is a way to end up reading the
    5m while looking at the 15m and never noticing."""
    src = _src()
    fn = src[src.index("function chartTfChanged() {"):]
    fn = fn[:fn.index("\nfunction readTfChanged(")]
    assert "followChartTf()" in fn and "loadRead()" in fn
    assert 'onchange="chartTfChanged()"' in src


def test_setting_the_reads_own_timeframe_breaks_the_link():
    """An override that gets silently reverted on the next chart change is
    worse than no override."""
    src = _src()
    fn = src[src.index("function readTfChanged() {"):]
    fn = fn[:fn.index("\nasync function loadRead(")]
    assert "box.checked = false" in fn


def test_the_link_does_not_reload_when_nothing_changed():
    """The chart's own timeframe changing to the one the read is already
    on must not cost a fetch."""
    src = _src()
    fn = src[src.index("function followChartTf() {"):]
    fn = fn[:fn.index("\nfunction chartTfChanged(")]
    assert "read.value === chart.value" in fn
    assert "return false" in fn
