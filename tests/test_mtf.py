"""The 4H / 15m / 1m entry engine.

Two things here are worth more than the rest put together.

CAUSALITY. The trigger asks whether the current 15-minute candle is green,
on a 1-minute close, when that candle is unfinished. If the engine reads
the FINISHED 15m bar to answer that, every signal is scored against a bar
that partly resulted from the move being signalled -- the exact error that
made a pure coin score 63.7% earlier in this project. The decisive test
builds a 15m bar that is green at minute three and red at its close, and
requires a signal at minute three.

PRECEDENCE AND BOUNDARIES. A 4H block is exactly sixteen 15m bars, so both
boundaries land on the same timestamp every block, every day. The opening
filter has to skip the right number of bars without depending on which
branch of a counter ran first.
"""

from __future__ import annotations

import pytest

from liqmap import mtf


class Bar:
    __slots__ = ("ts", "open", "high", "low", "close", "volume")

    def __init__(self, ts, o, h, l, c, v=100.0):
        self.ts, self.open, self.high = float(ts), o, h
        self.low, self.close, self.volume = l, c, v


def m1(ts, o, c, hi=None, lo=None):
    """One 1-minute bar. Wicks default to the body."""
    return Bar(ts, o, hi if hi is not None else max(o, c),
               lo if lo is not None else min(o, c), c)


def flat(start, n, px, step=60.0):
    return [m1(start + i * step, px, px) for i in range(n)]


def ramp(start, n, px, per, step=60.0):
    """`n` bars walking by `per` each, oldest first."""
    out = []
    for i in range(n):
        o = px + i * per
        out.append(m1(start + i * step, o, o + per))
    return out


# A 4H block boundary on the epoch grid.
T0 = 1_758_000_000.0 - (1_758_000_000.0 % mtf.H4_SECONDS)


# --------------------------------------------------------- the arithmetic

def test_the_driftless_rate_is_gamblers_ruin():
    """A tight target against a wide stop wins most of the time by
    construction. Reporting the hit rate without this number beside it
    makes geometry look like skill."""
    drift, _ = mtf.rates(10.0, 30.0, cost_ticks=0.0)
    assert drift == pytest.approx(0.75)


def test_cost_raises_the_bar_it_never_lowers_it():
    drift, be = mtf.rates(10.0, 30.0, cost_ticks=1.4)
    assert be > drift
    assert be == pytest.approx((30.0 + 1.4) / 40.0)


def test_a_tighter_target_needs_more_edge_not_less():
    """The intuition runs the other way and it is wrong: the cost is the
    same either way, so the narrower the total width the larger a share of
    it the cost is."""
    _, tight = mtf.rates(1.0, 30.0)
    _, wide = mtf.rates(40.0, 30.0)
    assert tight > wide


def test_a_zero_width_pair_does_not_divide_by_zero():
    assert mtf.rates(0.0, 0.0) == (0.0, 0.0)


# ------------------------------------------------------------- causality

def _bullish_runup(px=100.0):
    """Enough bars above the 4H open to put the state in BULLISH."""
    bars = [m1(T0, px, px)]
    bars += ramp(T0 + 60, 6, px, 0.5)        # climbs clear of the open
    return bars


def test_a_signal_fires_on_a_15m_bar_that_later_closes_red():
    """THE causality test.

    The 15m bar is green at minute three and red at its close. A signal
    must fire at minute three -- that is what was on the screen -- and it
    must be marked provisional, then withdrawn once the bar closes.

    An engine that reads the finished 15m bar sees red the whole time and
    fires nothing, which looks tidier and is a lie about the past.
    """
    px = 100.0
    bars = _bullish_runup(px)

    # Close out a RED 15m bar so the flip has something to flip from.
    t = T0 + mtf.M15_SECONDS
    bars += [m1(t + i * 60, 103.0 - i * 0.1, 102.9 - i * 0.1)
             for i in range(15)]

    # The 15m bar that goes green, then red. Up for four minutes...
    t2 = t + mtf.M15_SECONDS
    bars += [m1(t2, 101.5, 102.0)]
    bars += [m1(t2 + 60, 102.0, 102.6)]
    bars += [m1(t2 + 120, 102.6, 103.4)]      # clears the prior 1m high
    # ...then it collapses and closes below its own open.
    bars += [m1(t2 + 180 + i * 60, 103.0 - i * 0.4, 102.6 - i * 0.4)
             for i in range(11)]
    # One more 15m bar so the signalling bar is definitely closed.
    bars += flat(t2 + mtf.M15_SECONDS, 15, 98.0)

    sigs = mtf.scan(bars, tick=0.25)
    longs = [s for s in sigs if s.side == "long" and s.m15_start == t2]
    assert longs, "no signal on a bar that was green when it mattered"
    assert longs[0].ts <= t2 + 240
    assert longs[0].status == mtf.WITHDRAWN


def test_the_signal_is_decided_only_from_bars_up_to_its_own_timestamp():
    """Truncating the series after a signal must not move or remove it.
    If it does, something later was feeding the decision."""
    px = 100.0
    bars = _bullish_runup(px)
    t = T0 + mtf.M15_SECONDS
    bars += [m1(t + i * 60, 103.0 - i * 0.1, 102.9 - i * 0.1)
             for i in range(15)]
    t2 = t + mtf.M15_SECONDS
    bars += [m1(t2, 101.5, 102.0), m1(t2 + 60, 102.0, 102.6),
             m1(t2 + 120, 102.6, 103.4)]
    bars += flat(t2 + 180, 12, 103.4)
    bars += flat(t2 + mtf.M15_SECONDS, 15, 103.4)

    full = mtf.scan(bars, tick=0.25)
    assert full
    first = full[0]
    cut = [b for b in bars if b.ts <= first.ts]
    again = mtf.scan(cut, tick=0.25)
    assert again, "the signal vanished when the future was removed"
    assert again[0].ts == first.ts
    assert again[0].trigger == pytest.approx(first.trigger)
    assert again[0].stop == pytest.approx(first.stop)


def test_a_still_forming_bar_leaves_its_signal_provisional():
    px = 100.0
    bars = _bullish_runup(px)
    t = T0 + mtf.M15_SECONDS
    bars += [m1(t + i * 60, 103.0 - i * 0.1, 102.9 - i * 0.1)
             for i in range(15)]
    t2 = t + mtf.M15_SECONDS
    bars += [m1(t2, 101.5, 102.0), m1(t2 + 60, 102.0, 102.6),
             m1(t2 + 120, 102.6, 103.4)]        # series ends here
    sigs = mtf.scan(bars, tick=0.25)
    assert sigs and sigs[0].status == mtf.PROVISIONAL


def test_a_withdrawn_signal_stays_in_the_record():
    """Deleting it would make the chart's history prettier than the past,
    and would hide the only number that says whether a provisional marker
    is worth watching."""
    import inspect
    src = inspect.getsource(mtf._resolve)
    assert "WITHDRAWN" in src
    assert "remove" not in src and "del " not in src


# ------------------------------------------------------- the 4H filter

def test_no_signal_fires_while_the_4h_state_is_undecided():
    """The spec called the block bullish or bearish the instant it opened.
    Price sits on its own open at that moment, so that is a coin flip
    dressed as a filter."""
    bars = [m1(T0 + i * 60, 100.0, 100.0) for i in range(20)]
    t = T0 + mtf.M15_SECONDS
    bars += [m1(t + i * 60, 100.2 - i * 0.01, 100.1 - i * 0.01)
             for i in range(15)]
    t2 = t + mtf.M15_SECONDS
    bars += [m1(t2, 100.0, 100.4), m1(t2 + 60, 100.4, 100.9)]
    assert mtf.scan(bars, tick=0.25) == []


def test_the_state_needs_the_buffer_not_just_the_side():
    """One tick above the open is not a bullish four hours; it is where
    price always is at the start of one."""
    bars = [m1(T0, 100.0, 100.0)]
    bars += [m1(T0 + i * 60, 100.05, 100.05) for i in range(1, 12)]
    sigs = mtf.scan(bars + flat(T0 + 720, 10, 100.05), tick=0.25)
    assert sigs == []


def test_the_state_needs_to_hold_not_just_touch():
    assert mtf.STATE_HOLD_BARS >= 2
    assert mtf.STATE_BUFFER_TICKS > 0


def test_a_new_block_resets_its_own_state_to_undecided():
    """Carrying the old block's direction into a new one is reading a
    filter that is about a period that has ended."""
    import inspect
    src = inspect.getsource(mtf.scan)
    i = src.index("if start4 != h4_start:")
    assert "state4 = UNDECIDED" in src[i:i + 400]
    j = src.index("if start1 != h1_start:")
    assert "state1 = UNDECIDED" in src[j:j + 400]


# --------------------------------------------------- the opening filter

def test_the_first_15m_bar_of_a_block_is_skipped():
    assert mtf.SKIP_15M_BARS == 1


def test_the_bar_index_is_arithmetic_not_a_counter():
    """A counter that one branch resets and another advances is
    order-dependent, and both branches fire on the same timestamp every
    single 4H block."""
    import inspect
    src = inspect.getsource(mtf.scan)
    assert "idx15 = int((ts - h4_start) // trade_s)" in src
    assert "counter" not in src.lower()


def test_sixteen_fifteens_make_a_four_hour_block():
    assert mtf.H4_SECONDS / mtf.M15_SECONDS == 16


def test_no_signal_inside_the_opening_bar():
    px = 100.0
    bars = _bullish_runup(px)
    # Everything that follows is still inside the FIRST 15m bar.
    bars += [m1(T0 + 420, 103.0, 103.6), m1(T0 + 480, 103.6, 104.4)]
    assert [s for s in mtf.scan(bars, tick=0.25)
            if s.bar_15_index == 0] == []


# -------------------------------------------------------------- the stop

def test_the_stop_is_defined_and_sits_on_the_losing_side():
    px = 100.0
    bars = _bullish_runup(px)
    t = T0 + mtf.M15_SECONDS
    bars += [m1(t + i * 60, 103.0 - i * 0.1, 102.9 - i * 0.1)
             for i in range(15)]
    t2 = t + mtf.M15_SECONDS
    bars += [m1(t2, 101.5, 102.0), m1(t2 + 60, 102.0, 102.6),
             m1(t2 + 120, 102.6, 103.4)]
    sigs = mtf.scan(bars, tick=0.25)
    assert sigs
    s = sigs[0]
    assert s.stop < s.trigger < s.target
    assert s.stop_ticks >= mtf.MIN_STOP_TICKS


def test_a_trigger_already_back_at_the_open_is_not_a_signal():
    """The stop IS the 15m open now, so a trigger sitting on it describes
    a trade with no room in it -- the premise went before the trigger
    printed."""
    assert mtf.MIN_STOP_TICKS >= 1.0
    import inspect
    assert "if stop_t < MIN_STOP_TICKS:" in inspect.getsource(mtf.scan)


def test_the_target_is_the_spec_distance():
    assert mtf.TARGET_TICKS == 10.0


def test_every_signal_carries_both_rates():
    px = 100.0
    bars = _bullish_runup(px)
    t = T0 + mtf.M15_SECONDS
    bars += [m1(t + i * 60, 103.0 - i * 0.1, 102.9 - i * 0.1)
             for i in range(15)]
    t2 = t + mtf.M15_SECONDS
    bars += [m1(t2, 101.5, 102.0), m1(t2 + 60, 102.0, 102.6),
             m1(t2 + 120, 102.6, 103.4)]
    for s in mtf.scan(bars, tick=0.25):
        assert 0.0 < s.driftless < 1.0
        assert s.breakeven >= s.driftless


# ------------------------------------------------------------- the rearm

def test_one_signal_per_15m_bar_per_side():
    """There is no position to be flat of -- this tool suggests and never
    executes -- so the bar is the rearm."""
    px = 100.0
    bars = _bullish_runup(px)
    t = T0 + mtf.M15_SECONDS
    bars += [m1(t + i * 60, 103.0 - i * 0.1, 102.9 - i * 0.1)
             for i in range(15)]
    t2 = t + mtf.M15_SECONDS
    bars += [m1(t2, 101.5, 102.0)]
    # Six consecutive 1m bars each taking out the prior high.
    for k in range(6):
        o = 102.0 + k * 0.6
        bars.append(m1(t2 + 60 * (k + 1), o, o + 0.6))
    longs = [s for s in mtf.scan(bars, tick=0.25)
             if s.side == "long" and s.m15_start == t2]
    assert len(longs) == 1


# --------------------------------------------------------------- shorts

def test_the_short_side_mirrors_the_long_side():
    px = 100.0
    bars = [m1(T0, px, px)]
    bars += [m1(T0 + 60 * (i + 1), px - i * 0.5, px - (i + 1) * 0.5)
             for i in range(6)]
    t = T0 + mtf.M15_SECONDS
    bars += [m1(t + i * 60, 96.0 + i * 0.1, 96.1 + i * 0.1)
             for i in range(15)]           # a GREEN 15m bar
    t2 = t + mtf.M15_SECONDS
    bars += [m1(t2, 97.6, 97.2), m1(t2 + 60, 97.2, 96.8),
             m1(t2 + 120, 96.8, 96.0)]
    sigs = [s for s in mtf.scan(bars, tick=0.25) if s.side == "short"]
    assert sigs
    s = sigs[0]
    assert s.stop > s.trigger > s.target


# ------------------------------------------------------------- survival

def test_survival_counts_only_resolved_signals():
    sigs = [mtf.Signal(ts=1, side="long", trigger=1, target=2, stop=0,
                       status=mtf.CONFIRMED),
            mtf.Signal(ts=2, side="long", trigger=1, target=2, stop=0,
                       status=mtf.WITHDRAWN),
            mtf.Signal(ts=3, side="long", trigger=1, target=2, stop=0,
                       status=mtf.PROVISIONAL)]
    out = mtf.survival(sigs)
    assert out["resolved"] == 2 and out["confirmed"] == 1
    assert out["survival"] == pytest.approx(0.5)
    assert out["still_forming"] == 1


def test_survival_is_not_called_a_win_rate():
    """Nothing here follows a signal to its target or its stop. Calling
    this a win rate would be the single most misleading thing the module
    could say."""
    doc = mtf.survival.__doc__ or ""
    assert "NOT a win rate" in doc


def test_survival_of_nothing_is_none_not_zero():
    assert mtf.survival([])["survival"] is None


# ---------------------------------------------------------- the basics

def test_no_bars_is_no_signals():
    assert mtf.scan([]) == []
    assert mtf.scan([m1(T0, 1.0, 1.0)]) == []


def test_out_of_order_bars_are_sorted_not_trusted():
    px = 100.0
    bars = _bullish_runup(px)
    shuffled = list(reversed(bars))
    assert mtf.scan(shuffled, tick=0.25) == mtf.scan(bars, tick=0.25)


def test_the_three_timeframes_come_from_one_series():
    """Three feeds are three chances for the boundaries to disagree."""
    import inspect
    src = inspect.getsource(mtf)
    assert "resample" in src
    assert "def scan(bars" in src


def test_the_dict_carries_every_field_the_chart_shows():
    d = mtf.Signal(ts=1, side="long", trigger=1, target=2, stop=0).to_dict()
    for k in ("ts", "side", "trigger", "target", "stop", "status",
              "target_ticks", "stop_ticks", "driftless", "breakeven",
              "note"):
        assert k in d


def test_nothing_in_here_places_an_order():
    import inspect
    src = inspect.getsource(mtf).lower()
    for word in ("place_order", "submit", "api_key", "sign(", "private_key"):
        assert word not in src


# ------------------------------------------------- the chart integration

def _src():
    return open("liqmap/web.py").read()


def test_the_route_exists_and_is_token_protected():
    import re
    src = _src()
    assert '@app.get("/api/mtf"' in src
    m = re.search(r'@app\.get\("/api/mtf"[^)]*\)', src)
    assert "require_token" in m.group(0)


def test_the_route_is_defined_exactly_once():
    assert _src().count('@app.get("/api/mtf"') == 1


def test_the_route_always_reads_one_minute_bars():
    """Whatever the chart is showing. The engine resamples 15m and 4H from
    these, and asking the venue for each separately is three chances for
    the boundaries to disagree."""
    src = _src()
    route = src[src.index('@app.get("/api/mtf"'):
                src.index('@app.get("/api/chart"')]
    assert 'feed.history("1m")' in route
    assert 'rt.client().candles(coin, "1m"' in route
    assert '"15m"' not in route and '"4h"' not in route


def test_the_route_returns_the_survival_count():
    src = _src()
    route = src[src.index('@app.get("/api/mtf"'):
                src.index('@app.get("/api/chart"')]
    assert "_mtf.survival(" in route


def test_withdrawn_signals_are_drawn_faint_not_deleted():
    src = _src()
    assert "const MTF_ALPHA" in src
    i = src.index("const MTF_ALPHA")
    block = src[i:src.index("\n", src.index("}", i))]
    assert "withdrawn" in block


def test_the_bubble_fill_says_confirmed_not_the_colour():
    """Identity never rides on colour alone -- the fill says confirmed and
    the side is already carried by the letter and the position."""
    src = _src()
    fn = src[src.index("function drawMTF("):]
    fn = fn[:fn.index("\nfunction markIndex(")]
    assert "sig.status === 'confirmed'" in fn
    assert "g.fillText(long ? 'L' : 'S'" in fn


def test_the_setups_are_drawn_over_the_candles_not_under_them():
    """The profile belongs behind the price action. A signal marker does
    not -- hidden behind a candle it is not a marker."""
    src = _src()
    assert src.index("drawProfile(g, hi, lo, plotW, plotH);") \
        < src.index("drawMTF(g, s, xOf, step, y, hi, lo, plotW")


def test_the_breakeven_rate_is_shown_beside_the_signal():
    """A 10 tick target against a wide stop wins most of the time by
    construction. The number that says how much of that is geometry
    belongs where the signal is read, not in a tooltip."""
    src = _src()
    fn = src[src.index("async function loadMTF("):]
    fn = fn[:fn.index("async function loadProfile(")]
    assert "break even" in fn
    assert "last.driftless" in fn and "last.breakeven" in fn


def test_every_function_the_markup_calls_is_defined():
    src = _src()
    for fn in ("toggleMTF", "loadMTF", "drawMTF"):
        assert f"function {fn}(" in src or f"async function {fn}(" in src


def test_the_readout_element_the_script_writes_to_exists():
    src = _src()
    assert 'id="mtfRead"' in src and "$('mtfRead')" in src


def test_the_setups_refresh_with_the_chart():
    """Two clocks would let the bubbles drift out of step with the
    candles they sit on."""
    src = _src()
    i = src.index("  loadProfile();\n  loadMTF();")
    assert i > 0



# ------------------------------------------------- the 1H agreement

def _four_up_hour_down():
    """4H bullish, the HOUR bearish, and a 15m bounce inside it.

    Hour one rises clean, so both blocks turn bullish. Hour two opens at
    the top and falls away: price is well below the hour's own open while
    still well above the four-hour one. The bounce at the end is a valid
    red-to-green 15m with a 1m higher-high-and-higher-low -- everything
    the trade asks for EXCEPT the hour.
    """
    bars = [m1(T0, 100.0, 100.0)]
    for i in range(1, 60):
        o = 100.0 + (i - 1) * 0.2
        bars.append(m1(T0 + i * 60, o, o + 0.2))
    base = bars[-1].close
    t2 = T0 + mtf.H1_SECONDS
    for i in range(30):
        o = base - i * 0.25
        bars.append(m1(t2 + i * 60, o, o - 0.25))
    t3 = t2 + 30 * 60
    lo = bars[-1].close
    for i in range(4):
        o = lo + i * 0.5
        bars.append(m1(t3 + i * 60, o, o + 0.5, hi=o + 0.5, lo=o - 0.05))
    return bars


def test_the_hour_is_not_required_by_default():
    """The rule as finally stated is 4H + 15m + a trending minute. The
    hour was an earlier version of it, and the default follows the final
    statement rather than the first."""
    import inspect
    assert inspect.signature(mtf.scan).parameters["require_1h"].default \
        is False


def test_asking_for_the_hour_blocks_a_trade_the_hour_disagrees_with():
    """Kept as a switch because which version is better has an answer.
    The fixture is built so the two settings DISAGREE on these bars -- a
    fixture that produced nothing either way would pass while testing
    nothing, which an earlier version of this test did."""
    bars = _four_up_hour_down()
    cut = T0 + mtf.H1_SECONDS
    with_hour = [g for g in mtf.scan(bars, tick=0.25, require_1h=True)
                 if g.ts >= cut]
    without = [g for g in mtf.scan(bars, tick=0.25, require_1h=False)
               if g.ts >= cut]
    assert without, "the fixture stopped exercising the hour filter"
    assert with_hour == [], "traded while the hour disagreed"


def test_the_signal_records_both_opens():
    px = 100.0
    bars = _bullish_runup(px)
    t = T0 + mtf.M15_SECONDS
    bars += [m1(t + i * 60, 103.0 - i * 0.1, 102.9 - i * 0.1)
             for i in range(15)]
    t2 = t + mtf.M15_SECONDS
    bars += [m1(t2, 101.5, 102.0), m1(t2 + 60, 102.0, 102.6),
             m1(t2 + 120, 102.6, 103.4)]
    sigs = mtf.scan(bars, tick=0.25)
    assert sigs
    assert sigs[0].h4_open > 0 and sigs[0].h1_open > 0
    assert "h1_open" in sigs[0].to_dict()


def test_an_hour_is_a_quarter_of_a_block():
    assert mtf.H4_SECONDS / mtf.H1_SECONDS == 4
    assert mtf.H1_SECONDS / mtf.M15_SECONDS == 4


# ------------------------------------------- the continuation trigger

def test_a_spike_that_closes_back_inside_is_not_a_continuation():
    """A bar that pokes above the prior high but makes a LOWER low is one
    spike, not structure moving up. The old rule took it."""
    px = 100.0
    bars = _bullish_runup(px)
    t = T0 + mtf.M15_SECONDS
    bars += [m1(t + i * 60, 103.0 - i * 0.1, 102.9 - i * 0.1)
             for i in range(15)]
    t2 = t + mtf.M15_SECONDS
    # Inside bars: no continuation, so nothing fires before the spike.
    bars += [m1(t2, 101.5, 102.0, hi=102.4, lo=101.4),
             m1(t2 + 60, 102.0, 102.1, hi=102.2, lo=101.9)]
    # Higher high, but a LOWER low: one wide spike, not structure moving.
    bars += [m1(t2 + 120, 102.0, 103.0, hi=103.4, lo=101.0)]
    assert [s for s in mtf.scan(bars, tick=0.25)
            if s.m15_start == t2] == []


def test_a_higher_high_and_higher_low_is_a_continuation():
    px = 100.0
    bars = _bullish_runup(px)
    t = T0 + mtf.M15_SECONDS
    bars += [m1(t + i * 60, 103.0 - i * 0.1, 102.9 - i * 0.1)
             for i in range(15)]
    t2 = t + mtf.M15_SECONDS
    bars += [m1(t2, 101.5, 102.0), m1(t2 + 60, 102.0, 102.6),
             m1(t2 + 120, 102.8, 103.4, hi=103.4, lo=102.7)]
    assert [s for s in mtf.scan(bars, tick=0.25) if s.m15_start == t2]


def test_the_trigger_reads_the_bars_extremes_not_just_its_close():
    import inspect
    src = inspect.getsource(mtf.scan)
    assert "float(b.high) > float(prev1.high)" in src
    assert "float(b.low) > float(prev1.low)" in src


# --------------------------------------- the 15m must agree too

def test_no_long_is_ever_taken_below_its_own_15m_open():
    """The invariant Chris flagged: the 15m being traded has to lean the
    same way as the blocks above it.

    Asserted as a PROPERTY over a whole random series rather than on one
    hand-built bar, because two separate rules enforce it -- the explicit
    state check, and the stop being the 15m open (which for a long in a
    red candle would sit above the entry and is rejected). A fixture
    aimed at one of them passes while the other does the work.
    """
    import random
    rng = random.Random(21)
    px = 5000.0
    rows = []
    for i in range(20000):
        o = px
        px = round((px + rng.gauss(0, 0.6)) * 4) / 4
        rows.append(m1(T0 + i * 60, o, px,
                       hi=max(o, px) + 0.25 * rng.randint(0, 2),
                       lo=min(o, px) - 0.25 * rng.randint(0, 2)))
    sigs = mtf.scan(rows, tick=0.25)
    assert len(sigs) > 50, "fixture produced too few signals to judge"
    for s in sigs:
        if s.side == "long":
            assert s.trigger > s.m15_open, (
                f"long at {s.trigger} below its 15m open {s.m15_open}")
        else:
            assert s.trigger < s.m15_open, (
                f"short at {s.trigger} above its 15m open {s.m15_open}")


def test_the_state_check_names_the_15m_explicitly():
    """Weak on its own -- the property test above is what has teeth -- but
    it catches the rule being deleted silently on the assumption that the
    stop covers it."""
    import inspect
    src = inspect.getsource(mtf.scan)
    assert "m15_bull = cur15.green" in src
    assert "m15_bear = cur15.red" in src


def test_the_prior_15m_colour_is_no_longer_required():
    """It was a reversal condition and the wrong shape: what is traded is
    agreement, not a flip."""
    import inspect
    src = inspect.getsource(mtf.scan)
    assert "prior15.red" not in src and "prior15.green" not in src


# ------------------------------------------ the stop and the clock

def test_the_stop_is_the_15m_candles_own_open():
    import random
    rng = random.Random(8)
    px, rows = 5000.0, []
    for i in range(12000):
        o = px
        px = round((px + rng.gauss(0, 0.6)) * 4) / 4
        rows.append(m1(T0 + i * 60, o, px,
                       hi=max(o, px) + 0.25, lo=min(o, px) - 0.25))
    sigs = mtf.scan(rows, tick=0.25)
    assert sigs
    for s in sigs:
        assert s.stop == s.m15_open


def test_every_signal_expires_at_its_own_15m_close():
    """No trade outlives the candle it was taken on."""
    import random
    rng = random.Random(9)
    px, rows = 5000.0, []
    for i in range(12000):
        o = px
        px = round((px + rng.gauss(0, 0.6)) * 4) / 4
        rows.append(m1(T0 + i * 60, o, px,
                       hi=max(o, px) + 0.25, lo=min(o, px) - 0.25))
    sigs = mtf.scan(rows, tick=0.25)
    assert sigs
    for s in sigs:
        assert s.expires == s.m15_start + mtf.M15_SECONDS
        assert s.ts < s.expires


def test_the_stop_is_much_closer_than_the_structural_one_was():
    """The whole point of the change. The old stop averaged 27 ticks and
    demanded 76.6% against a ten tick target; this one is the distance
    back to the candle's open."""
    import random
    rng = random.Random(10)
    px, rows = 5000.0, []
    for i in range(20000):
        o = px
        px = round((px + rng.gauss(0, 0.6)) * 4) / 4
        rows.append(m1(T0 + i * 60, o, px,
                       hi=max(o, px) + 0.25, lo=min(o, px) - 0.25))
    sigs = mtf.scan(rows, tick=0.25)
    avg = sum(s.stop_ticks for s in sigs) / len(sigs)
    assert avg < 20, f"stops averaging {avg:.0f} ticks is not the open"


# ------------------------------------------- the entry window

def _noise(n=20000, seed=33):
    import random
    rng = random.Random(seed)
    px, rows = 5000.0, []
    for i in range(n):
        o = px
        px = round((px + rng.gauss(0, 0.6)) * 4) / 4
        rows.append(m1(T0 + i * 60, o, px,
                       hi=max(o, px) + 0.25 * rng.randint(0, 2),
                       lo=min(o, px) - 0.25 * rng.randint(0, 2)))
    return rows


def test_a_late_trigger_is_refused_when_a_window_is_set():
    rows = _noise()
    for cap in (2.0, 5.0, 10.0):
        sigs = mtf.scan(rows, tick=0.25, max_minutes_in=cap)
        assert sigs, f"no signals at all within {cap} minutes"
        assert max(s.minutes_in for s in sigs) <= cap


def test_no_window_means_the_whole_candle_is_fair_game():
    sigs = mtf.scan(_noise(), tick=0.25)
    assert max(s.minutes_in for s in sigs) > 10


def test_a_tighter_window_means_a_tighter_stop():
    """The mechanism, not a preference. The stop IS the candle's open, so
    a trigger that fires two minutes in is near it and one that fires at
    minute thirteen is a long way from it."""
    rows = _noise()
    tight = mtf.scan(rows, tick=0.25, max_minutes_in=3.0)
    loose = mtf.scan(rows, tick=0.25)
    assert tight and loose
    a = sum(s.stop_ticks for s in tight) / len(tight)
    b = sum(s.stop_ticks for s in loose) / len(loose)
    assert a < b, f"tight window gave {a:.1f}t, loose gave {b:.1f}t"


def test_a_tighter_window_also_throws_signals_away():
    """Which is the cost, and the reason it has to be searched on one half
    of the data and judged on the other."""
    rows = _noise()
    assert len(mtf.scan(rows, tick=0.25, max_minutes_in=3.0)) < \
        len(mtf.scan(rows, tick=0.25))


def test_the_window_never_changes_a_signal_it_keeps():
    """Filtering must drop signals, never alter the ones it allows. If it
    moved a trigger or a stop, every comparison across windows would be
    comparing two different strategies."""
    rows = _noise()
    loose = {s.ts: s for s in mtf.scan(rows, tick=0.25)}
    for s in mtf.scan(rows, tick=0.25, max_minutes_in=4.0):
        o = loose.get(s.ts)
        assert o is not None
        assert (s.trigger, s.stop, s.side) == (o.trigger, o.stop, o.side)


# ------------------------------------------- the timeframes as inputs

def test_the_three_timeframes_are_parameters_with_the_stated_defaults():
    import inspect
    pr = inspect.signature(mtf.scan).parameters
    assert pr["block_s"].default == mtf.H4_SECONDS
    assert pr["trade_s"].default == mtf.M15_SECONDS
    assert pr["trigger_s"].default == 60.0


def test_a_different_trade_candle_moves_the_stop_and_the_clock():
    """The traded candle supplies both: its open is the stop and its
    close is the clock. Changing it has to change both or the two have
    drifted apart."""
    rows = _noise()
    for trade_s in (900.0, 1800.0):
        sigs = mtf.scan(rows, tick=0.25, block_s=14400.0, trade_s=trade_s)
        assert sigs
        for s in sigs:
            assert s.expires == s.m15_start + trade_s
            assert s.stop == s.m15_open
            assert s.m15_start % trade_s == 0


def test_a_coarser_trigger_widens_the_stop():
    """The arithmetic Chris asked about, as a test rather than a claim.
    The stop is the distance back to the traded candle's open, so reading
    the continuation on a bigger bar enters further from it."""
    rows = _noise(n=40000)
    fine = mtf.scan(rows, tick=0.25, trigger_s=60.0)
    coarse = mtf.scan(rows, tick=0.25, trigger_s=300.0)
    assert fine and coarse
    a = sum(s.stop_ticks for s in fine) / len(fine)
    b = sum(s.stop_ticks for s in coarse) / len(coarse)
    assert b > a, f"1m trigger gave {a:.1f}t, 5m trigger gave {b:.1f}t"


def test_a_smaller_target_needs_a_higher_hit_rate():
    """The other half of the same answer. Cost is fixed, so the smaller
    the target the bigger a share of it the fee is."""
    for stop in (8.0, 14.5, 25.0):
        _, five = mtf.rates(5.0, stop)
        _, ten = mtf.rates(10.0, stop)
        _, twenty = mtf.rates(20.0, stop)
        assert five > ten > twenty


def test_a_15m_block_with_a_5m_trigger_is_buildable():
    """The configuration Chris proposed, end to end."""
    sigs = mtf.scan(_noise(n=40000), tick=0.25,
                    block_s=3600.0, trade_s=900.0, trigger_s=300.0)
    assert sigs
    for s in sigs:
        assert s.m15_start % 900.0 == 0
