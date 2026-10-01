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


def test_a_new_4h_block_resets_the_state_to_undecided():
    """Carrying the old block's direction into a new one is reading a
    filter that is about a period that has ended."""
    import inspect
    src = inspect.getsource(mtf.scan)
    i = src.index("if start4 != h4_start:")
    assert "state = UNDECIDED" in src[i:i + 400]


# --------------------------------------------------- the opening filter

def test_the_first_15m_bar_of_a_block_is_skipped():
    assert mtf.SKIP_15M_BARS == 1


def test_the_bar_index_is_arithmetic_not_a_counter():
    """A counter that one branch resets and another advances is
    order-dependent, and both branches fire on the same timestamp every
    single 4H block."""
    import inspect
    src = inspect.getsource(mtf.scan)
    assert "idx15 = int((ts - h4_start) // M15_SECONDS)" in src
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


def test_a_stop_too_close_to_the_trigger_is_not_a_signal():
    """A stop a tick away is a rounding error, and the break-even rate it
    implies is a fantasy."""
    assert mtf.MIN_STOP_TICKS >= 2.0
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
