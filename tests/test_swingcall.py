"""Which swing forms next, and which side of the last one it lands.

Three traps, and the first two are the ones that turn a predictor into a
machine for grading itself on the past.

SOME OF THE ANSWER IS ALREADY DETERMINED. If price has traded above the
prior high since the last pivot, the next high WILL be a higher high --
there is nothing left to forecast. Counting those among the correct
predictions is scoring yourself on what already happened.

A SWING IS NOT VISIBLE ON ITS OWN BAR. `confirmed_at` is `strength` bars
later, and reading a pivot before that is lookahead -- the most common way
a structure backtest lies.

AND N RIDES WITH THE RATE. Seven out of ten is 70% and is also noise
wearing a decimal point.
"""

from __future__ import annotations

import pytest

from liqmap import swingcall as sc
from liqmap.structure import Candle, swings


def bars(highs, lows=None):
    """Candles from a list of highs; lows mirror them unless given."""
    lows = lows if lows is not None else [h - 1.0 for h in highs]
    return [Candle(ts=i * 60.0, open=(h + l) / 2, high=h, low=l,
                   close=(h + l) / 2, volume=1.0)
            for i, (h, l) in enumerate(zip(highs, lows))]


def peak(at, n, base=100.0, size=5.0):
    """Highs with a single clean peak at index `at`."""
    return [base + (size if i == at else 0.0) for i in range(n)]


# A clean alternating structure: a swing HIGH at index 4 and a swing LOW
# at index 8, both far enough from the ends to survive a strength-3
# window. `tail` continues it, so a test can decide whether price has
# taken out the prior high or not.
#
# The pivot has to sit at least `strength` bars from either end or the
# detector cannot see it -- which is how the first draft of these
# fixtures produced one swing and let a test pass without asserting
# anything.
SHAPE = [100, 101, 102, 103, 108, 103, 102, 101, 100]


def structured(tail):
    highs = SHAPE + list(tail)
    return bars(highs, [h - 3 for h in highs])


# ------------------------------------------------- only what was knowable

def test_a_swing_is_not_usable_on_its_own_bar():
    """`confirmed_at` is strength bars after the pivot. Using it earlier is
    reading the future."""
    cs = bars(peak(4, 12))
    sw = swings(cs, strength=3)
    assert sw, "fixture produced no swings"
    pivot = sw[0]
    assert pivot.confirmed_at > pivot.index
    assert sc.knowable(sw, pivot.index) == []
    assert sc.knowable(sw, pivot.confirmed_at) == [pivot]


def test_too_little_structure_is_said_rather_than_guessed():
    out = sc.call_next(bars([100.0] * 5), [], side="buy")
    assert out.call is None
    assert "confirmed swings" in out.why


def test_a_high_and_a_low_are_both_needed():
    cs = bars(peak(4, 12))
    sw = [s for s in swings(cs, strength=3) if s.is_high]
    out = sc.call_next(cs, sw, side="buy", strength=3)
    assert out.call is None


# -------------------------------------------- already determined, not called

def test_price_already_through_the_level_is_settled_not_predicted():
    """THE trap. Whenever the next high confirms it is going to be a higher
    high. That is waiting, not forecasting."""
    cs = structured([101, 105, 109, 110])        # takes out the 108 high
    out = sc.call_next(cs, swings(cs, strength=3), side="sell", strength=3)
    assert out.next_kind == "high"
    assert out.call == "HH"
    assert out.settled and not out.predicted
    assert "already" in out.why


def test_a_settled_call_ignores_what_the_flow_is_leaning():
    """It is a fact about price, not an opinion about flow."""
    cs = structured([101, 105, 109, 110])
    sw = swings(cs, strength=3)
    buy = sc.call_next(cs, sw, side="buy", strength=3)
    sell = sc.call_next(cs, sw, side="sell", strength=3)
    assert buy.settled and sell.settled
    assert buy.call == sell.call == "HH"


def test_only_the_bars_since_the_last_swing_count_as_the_running_extreme():
    """The pivot's own bar and everything before it belong to a swing that
    has already been counted.

    This needs a shape where it matters: a deep low early on, a HIGHER low
    that becomes the reference, and a confirmed high after it. Measuring
    from the start of the series finds the old low, marks LL already set,
    and reports a question that is still wide open as settled history.
    """
    lows = [90, 96, 97, 98, 99, 100, 99, 98, 97, 98, 99, 100, 101, 102,
            101, 100, 99]
    cs = bars([l + 3 for l in lows], lows)
    sw = swings(cs, strength=3)
    out = sc.call_next(cs, sw, side="buy", strength=3)
    assert out.next_kind == "low"
    assert out.reference == pytest.approx(97.0)   # the higher low, not the 90
    assert out.running == pytest.approx(99.0)     # since that swing only
    assert not out.settled
    assert out.call == "HL"


def test_not_exceeding_the_level_settles_nothing():
    """The asymmetry. Price has the rest of the swing to go and do it, so
    a level still untouched is still an open question."""
    cs = structured([101, 102, 103, 104])        # never reaches the 108
    out = sc.call_next(cs, swings(cs, strength=3), side="buy", strength=3)
    assert out.next_kind == "high"
    assert not out.settled and out.predicted


# ------------------------------------------------------- the actual call

def test_the_next_swing_is_the_opposite_kind_of_the_last_one():
    cs = structured([101, 102, 103, 104])
    sw = swings(cs, strength=3)
    last = max(sc.knowable(sw, len(cs) - 1),
               key=lambda s: (s.confirmed_at, s.index))
    assert not last.is_high, "fixture should end on a confirmed low"
    out = sc.call_next(cs, sw, side="buy", strength=3)
    assert out.next_kind == "high"


def test_the_flow_decides_which_side_of_the_level_it_lands():
    cs = structured([101, 102, 103, 104])
    sw = swings(cs, strength=3)
    up = sc.call_next(cs, sw, side="buy", strength=3)
    down = sc.call_next(cs, sw, side="sell", strength=3)
    assert up.call == "HH" and down.call == "LH"
    assert not up.settled and not down.settled


def test_no_side_and_an_untouched_level_is_no_call():
    cs = structured([101, 102, 103, 104])
    out = sc.call_next(cs, swings(cs, strength=3), side=None, strength=3)
    assert out.next_kind == "high"
    assert out.call is None
    assert "no side" in out.why


def test_the_call_names_the_price_it_needs_and_the_price_that_kills_it():
    """A direction with no invalidation is an opinion, not a prediction."""
    cs = structured([101, 102, 103, 104])
    out = sc.call_next(cs, swings(cs, strength=3), side="buy", strength=3)
    assert out.needs == pytest.approx(108.0)      # the prior high
    assert out.fails_at == pytest.approx(97.0)    # the prior low
    assert "108" in out.describe()


def test_the_call_says_how_long_confirmation_takes():
    cs = structured([101, 102, 103, 104])
    out = sc.call_next(cs, swings(cs, strength=3), side="buy", strength=3)
    assert out.confirm_bars == 3


def test_no_bars_since_the_last_swing_is_not_a_crash():
    cs = structured([101, 102, 103, 104])
    sw = swings(cs, strength=3)
    out = sc.call_next(cs[:5], sw, side="buy", strength=3)
    assert out.call is None


def test_no_candles_at_all_is_not_a_crash():
    assert sc.call_next([], [], side="buy").call is None


# ---------------------------------------------------------------- scoring

class S:
    def __init__(self, px, is_high):
        self.px, self.is_high = px, is_high


def test_a_swing_is_graded_against_its_prior_level():
    assert sc.actual_call(S(110.0, True), 105.0) == "HH"
    assert sc.actual_call(S(100.0, True), 105.0) == "LH"
    assert sc.actual_call(S(99.0, False), 95.0) == "HL"
    assert sc.actual_call(S(90.0, False), 95.0) == "LL"


def test_settled_calls_are_counted_apart_from_predicted_ones():
    """They were already determined when they were made. Folding them in
    measures how often the tool can read the present."""
    predicted = sc.SwingCall(call="HH", settled=False)
    already = sc.SwingCall(call="HH", settled=True)
    r = sc.score([(predicted, S(110.0, True), 105.0),
                  (already, S(110.0, True), 105.0)])
    assert r.n == 1 and r.hits == 1
    assert r.settled_n == 1 and r.settled_hits == 1
    assert r.rate == pytest.approx(1.0)


def test_a_wrong_call_counts_against_the_rate():
    r = sc.score([(sc.SwingCall(call="HH"), S(100.0, True), 105.0)])
    assert r.n == 1 and r.hits == 0
    assert r.rate == pytest.approx(0.0)


def test_calls_that_said_nothing_are_not_scored():
    r = sc.score([(sc.SwingCall(call=None), S(110.0, True), 105.0)])
    assert r.n == 0 and r.settled_n == 0


def test_the_sample_size_rides_with_the_rate():
    """Seven out of ten is 70% and is also noise wearing a decimal point,
    and the two are indistinguishable once the count is dropped."""
    r = sc.Rate()
    for _ in range(7):
        r.add(sc.Scored("HH", "HH", settled=False))
    for _ in range(3):
        r.add(sc.Scored("HH", "LH", settled=False))
    assert r.rate == pytest.approx(0.7)
    assert "of 10" in r.describe()
    assert r.to_dict()["n"] == 10


def test_a_rate_with_nothing_behind_it_is_none_not_zero():
    r = sc.Rate()
    assert r.rate is None
    assert "no predicted swings scored yet" in r.describe()


def test_settled_calls_alone_do_not_produce_a_rate():
    r = sc.Rate()
    r.add(sc.Scored("HH", "HH", settled=True))
    assert r.rate is None
    assert r.settled_n == 1
    assert "already set" in r.describe()


def test_nothing_here_places_an_order():
    import inspect

    src = inspect.getsource(sc).lower()
    for word in ("def submit", "def execute", "def place", "def order"):
        assert word not in src


def test_swings_all_of_one_kind_says_which_kind_is_missing():
    """Saying 'not enough swings' when there are twenty of them sends you
    looking in the wrong place. A straight trend on a short window really
    does produce pivots of only one kind."""
    from liqmap.structure import Swing

    cs = structured([101, 102, 103, 104])
    sw = [Swing(index=4, ts=240.0, px=108.0, kind="high", confirmed_at=7),
          Swing(index=9, ts=540.0, px=104.0, kind="high", confirmed_at=12)]
    out = sc.call_next(cs, sw, side="buy", strength=3)
    assert out.call is None
    assert "2 confirmed swings" in out.why
    assert "no lows" in out.why


# ------------------------------------------------------ route and panel


def _src():
    import inspect

    from liqmap import web as w
    return inspect.getsource(w)


def test_the_route_calls_the_swing_from_knowable_structure_only():
    src = _src()
    block = src[src.index('out["swing"] = swc.call_next('):]
    block = block[:block.index('out["timeframes"] = [')]
    assert "_swings(bars" in block
    assert "strength=_SWING_STRENGTH" in block


def test_the_route_feeds_the_swing_call_the_engines_side():
    """The structural frame comes from price; which side of it comes from
    the order flow."""
    src = _src()
    block = src[src.index("chosen = lay.get(interval)"):]
    block = block[:block.index('out["timeframes"] = [')]
    assert 'get("result") or {}).get("side")' in block
    assert "side=lead" in block


def test_the_panel_draws_settled_and_predicted_differently():
    """Price having already traded through the level is a fact. Drawing it
    as a forecast lets the tool grade itself on the past."""
    src = _src()
    fn = src[src.index("const S = (d && d.swing) || {};"):]
    fn = fn[:fn.index("const st = $('layerStamp');")]
    assert "S.settled" in fn
    assert "ALREADY SET" in fn and "PREDICTED" in fn


def test_the_panel_says_why_when_there_is_no_call():
    src = _src()
    fn = src[src.index("const S = (d && d.swing) || {};"):]
    fn = fn[:fn.index("const st = $('layerStamp');")]
    assert "S.why" in fn


def test_the_swing_line_sits_with_the_verdict_not_in_the_ladder():
    src = _src()
    assert 'id="swingCall"' in src
    panel = src[src.index('id="layerPanel"'):]
    assert panel.index('id="swingCall"') < panel.index('id="layerBody"')
