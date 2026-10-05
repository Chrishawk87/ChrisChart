"""Layer seven: one word for the candle, and the side it points.

Three things would make this dishonest, and each has a test that fails if
the code takes the easy path.

FOUR OF THE NINE ARE NOT KNOWABLE LIVE. Failed breakout, failed breakdown,
exhaustion and reversal describe how something turned out, and inside a
bar that has not closed the most that can be said is "so far". A
classifier that reports them with the confidence of a closed bar is using
information the moment did not have -- the exact shape of the bug that
scored 98.4% on a coin flip earlier in this project.

ABSORPTION INVERTS THE SIDE. Heavy buying that will not move price means
somebody is selling into all of it, and they are winning. A bar with heavy
buying and a rising price and a bar with heavy buying and a stuck price
both satisfy "the buyers are attacking", and only one of them is bullish.

A MISSING LAYER IS NOT A NEUTRAL ONE. No word at all beats a confident one
built on a feed that has not warmed up.
"""

from __future__ import annotations

import pytest

from liqmap import layers as ly
from liqmap import result as rs


def layers(who="buyers", pace=1.0, pace_known=True, impact=1.0,
           impact_measured=True, move=4.0, wide=True, position=0.5,
           refs=(), high=0.0, low=0.0, price=100.0, unit="t",
           intent_measured=True, response_measured=True):
    """A Layers built directly, so each test states only what it is about."""
    out = ly.Layers(timeframe="15m", interval_s=900.0)
    out.intent = ly.Intent(who=who, lean=0.7, trades=200,
                           measured=intent_measured)
    out.force = ly.Force(pace=pace, pace_known=pace_known)
    out.resistance = ly.Resistance(absorbing=impact < 0.5,
                                   impact_ratio=impact,
                                   measured=impact_measured)
    with_intent = None
    if who in ("buyers", "sellers"):
        want = 1.0 if who == "buyers" else -1.0
        with_intent = (move * want) > 0
    out.response = ly.Response(move=move, range_=abs(move) * 2, unit=unit,
                               impact=impact, with_intent=with_intent,
                               position=position, wide=wide,
                               measured=response_measured)
    out.location = ly.Location(
        price=price, unit=unit, high=high or price, low=low or price,
        near=[ly.Reference(n, p, (p - price) / 0.25, unit)
              for n, p in refs])
    return out


# ------------------------------------------------- missing beats confident

def test_no_word_at_all_when_the_tape_has_not_warmed_up():
    r = rs.classify(layers(intent_measured=False))
    assert r.state is rs.State.UNKNOWN
    assert r.side is None
    assert "intent" in r.missing


def test_a_bar_with_no_price_response_yet_says_nothing():
    r = rs.classify(layers(response_measured=False))
    assert r.state is rs.State.UNKNOWN
    assert "response" in r.missing


def test_unknown_is_never_actionable():
    assert not rs.classify(layers(intent_measured=False)).actionable


# ----------------------------------------------------- absorption inverts

def test_heavy_buying_that_will_not_move_price_is_a_sell():
    """THE one a naive reading gets backwards."""
    r = rs.classify(layers(who="buyers", pace=2.0, impact=0.2, move=0.2,
                           wide=False))
    assert r.state is rs.State.ABSORPTION
    assert r.side == "sell"
    assert not r.provisional


def test_heavy_selling_that_will_not_move_price_is_a_buy():
    r = rs.classify(layers(who="sellers", pace=2.0, impact=0.2, move=-0.2,
                           wide=False))
    assert r.state is rs.State.ABSORPTION
    assert r.side == "buy"


def test_absorption_is_decided_before_continuation():
    """Both bars satisfy 'the buyers are attacking'. Only one is bullish,
    and the order of the checks is what keeps them apart."""
    stuck = rs.classify(layers(who="buyers", pace=2.0, impact=0.2, move=0.2,
                               wide=False))
    paid = rs.classify(layers(who="buyers", pace=2.0, impact=1.0, move=4.0))
    assert stuck.side == "sell" and paid.side == "buy"


def test_light_volume_going_nowhere_is_not_absorption():
    """Four fills that happen to lean one way is a quiet bar, not the
    passive side taking size -- and not continuation either. Without the
    width test on continuation, a tenth of a tick of drift satisfies "the
    buyers are crossing and price went up"."""
    r = rs.classify(layers(who="buyers", pace=0.4, impact=0.2, move=0.1,
                           wide=False))
    assert r.state is rs.State.CONSOLIDATION


# ----------------------------------------------- what is knowable right now

def test_the_aggressor_getting_paid_is_continuation():
    r = rs.classify(layers(who="buyers", pace=1.2, impact=1.0, move=5.0))
    assert r.state is rs.State.CONTINUATION
    assert r.side == "buy" and not r.provisional


def test_price_travelling_further_than_the_flow_justifies_is_expansion():
    r = rs.classify(layers(who="buyers", pace=1.0, impact=3.0, move=9.0))
    assert r.state is rs.State.EXPANSION
    assert r.side == "buy" and not r.provisional


def test_reaching_a_level_and_coming_back_is_rejection():
    # Reached it and turned WITHOUT trading through -- the high stops
    # short of the level. Through-and-back is a failed break, which is a
    # different word and a provisional one.
    r = rs.classify(layers(who="buyers", move=1.0, position=0.1,
                           price=100.0, high=100.9, low=99.9,
                           refs=[("PDH", 100.95)]))
    assert r.state is rs.State.REJECTION
    assert r.side == "sell" and not r.provisional


def test_rejection_from_below_points_the_other_way():
    r = rs.classify(layers(who="sellers", move=-1.0, position=0.9,
                           price=100.0, high=100.1, low=99.1,
                           refs=[("PDL", 99.05)]))
    assert r.state is rs.State.REJECTION
    assert r.side == "buy"


def test_a_quiet_narrow_bar_is_consolidation_and_points_nowhere():
    r = rs.classify(layers(pace=0.3, move=0.1, wide=False))
    assert r.state is rs.State.CONSOLIDATION
    assert r.side is None
    assert not r.actionable


# --------------------------------------------- the four that are not settled

def test_trading_through_a_level_and_coming_back_is_provisional():
    """The bar can still close back through it, and then the word was
    wrong. Live, the most that can be said is 'for now'."""
    r = rs.classify(layers(who="buyers", move=1.0, price=100.0,
                           high=101.0, low=99.5, refs=[("PDH", 100.5)]))
    assert r.state is rs.State.FAILED_BREAKOUT
    assert r.side == "sell"
    assert r.provisional
    assert not r.actionable


def test_a_failed_breakdown_points_the_other_way():
    r = rs.classify(layers(who="sellers", move=-1.0, price=100.0,
                           high=100.5, low=99.0, refs=[("PDL", 99.5)]))
    assert r.state is rs.State.FAILED_BREAKDOWN
    assert r.side == "buy" and r.provisional


def test_a_move_that_has_run_and_stopped_paying_is_exhaustion():
    """Same heavy-and-stuck shape as absorption, but after the bar has
    already travelled -- the move running out rather than being met at a
    wall. Whether it has actually ended is not knowable until the close."""
    r = rs.classify(layers(who="buyers", pace=2.2, impact=0.2, move=12.0,
                           wide=True))
    assert r.state is rs.State.EXHAUSTION
    assert r.side == "sell" and r.provisional


def test_absorption_and_exhaustion_are_told_apart_by_whether_it_moved():
    wall = rs.classify(layers(who="buyers", pace=2.2, impact=0.2, move=0.3,
                              wide=False))
    spent = rs.classify(layers(who="buyers", pace=2.2, impact=0.2,
                               move=12.0, wide=True))
    assert wall.state is rs.State.ABSORPTION and not wall.provisional
    assert spent.state is rs.State.EXHAUSTION and spent.provisional


def test_turning_against_the_previous_bar_is_a_reversal():
    before = rs.Result(state=rs.State.CONTINUATION, side="buy")
    r = rs.classify(layers(who="sellers", pace=1.0, impact=1.0, move=-5.0),
                    previous=before)
    assert r.state is rs.State.REVERSAL
    assert r.side == "sell" and r.provisional


def test_reversal_is_unavailable_rather_than_guessed_without_a_previous():
    r = rs.classify(layers(who="sellers", pace=1.0, impact=1.0, move=-5.0))
    assert r.state is not rs.State.REVERSAL


def test_a_previous_bar_that_pointed_nowhere_cannot_be_reversed():
    """Treating 'no side' as a direction would manufacture reversals out
    of quiet."""
    before = rs.Result(state=rs.State.CONSOLIDATION, side=None)
    r = rs.classify(layers(who="sellers", pace=1.0, impact=1.0, move=-5.0),
                    previous=before)
    assert r.state is not rs.State.REVERSAL


def test_agreeing_with_the_previous_bar_is_not_a_reversal():
    before = rs.Result(state=rs.State.CONTINUATION, side="sell")
    r = rs.classify(layers(who="sellers", pace=1.0, impact=1.0, move=-5.0),
                    previous=before)
    assert r.state is rs.State.CONTINUATION


# ----------------------------------------------------------- the contract

def test_every_state_that_is_not_settled_is_marked_as_such():
    assert rs.PROVISIONAL == {rs.State.FAILED_BREAKOUT,
                              rs.State.FAILED_BREAKDOWN,
                              rs.State.EXHAUSTION, rs.State.REVERSAL}


def test_a_provisional_word_is_never_actionable():
    for st in rs.PROVISIONAL:
        assert not rs.Result(state=st, side="buy", provisional=True).actionable


def test_every_state_says_what_it_means():
    for st in rs.State:
        assert len(rs.MEANING[st]) > 20


def test_all_nine_of_chriss_states_exist():
    for name in ("continuation", "expansion", "absorption", "exhaustion",
                 "rejection", "failed_breakout", "failed_breakdown",
                 "reversal", "consolidation"):
        assert rs.State(name)


def test_the_word_reads_as_a_sentence():
    r = rs.classify(layers(who="buyers", pace=2.0, impact=0.2, move=0.2,
                           wide=False))
    text = r.describe()
    assert "ABSORPTION" in text and "SELL" in text


def test_a_provisional_word_says_the_bar_can_still_close_otherwise():
    r = rs.classify(layers(who="buyers", pace=2.2, impact=0.2, move=12.0))
    assert "provisional" in r.describe()
    assert "close the other way" in r.describe()


def test_nothing_here_places_an_order():
    import inspect

    src = inspect.getsource(rs).lower()
    for word in ("def submit", "def execute", "def place", "def order"):
        assert word not in src


def test_the_dict_carries_the_side_and_whether_it_is_settled():
    d = rs.classify(layers(who="buyers", pace=1.2, impact=1.0,
                           move=5.0)).to_dict()
    for k in ("state", "side", "provisional", "actionable", "why",
              "meaning"):
        assert k in d


# --------------------------------------------- the memory between two bars


def _src():
    import inspect

    from liqmap import web as w
    return inspect.getsource(w)


def test_the_previous_bar_survives_the_whole_of_this_one():
    """THE bug in the obvious version. Keep one entry and reversal fires
    once at the bar boundary and then vanishes: the first read after the
    roll sees the old bar, and every read after that sees this one and
    finds nothing to turn against."""
    from liqmap import web as w

    rt = w.Runtime.__new__(w.Runtime)
    rt._last_state = {}
    first = rs.Result(state=rs.State.CONTINUATION, side="buy")
    rt.remember_state("BTC", "15m", 0.0, first)

    rt.remember_state("BTC", "15m", 900.0,
                      rs.Result(state=rs.State.CONTINUATION, side="sell"))
    assert rt.previous_state("BTC", "15m", 900.0) is first
    # Re-read the same bar several times; the previous one must not move.
    for _ in range(5):
        rt.remember_state("BTC", "15m", 900.0,
                          rs.Result(state=rs.State.ABSORPTION, side="sell"))
        assert rt.previous_state("BTC", "15m", 900.0) is first


def test_a_bar_is_never_its_own_previous():
    """Otherwise a reading reverses against itself and oscillates for the
    whole candle."""
    from liqmap import web as w

    rt = w.Runtime.__new__(w.Runtime)
    rt._last_state = {}
    rt.remember_state("BTC", "1m", 60.0,
                      rs.Result(state=rs.State.CONTINUATION, side="buy"))
    assert rt.previous_state("BTC", "1m", 60.0) is None


def test_the_roll_carries_the_last_reading_forward_not_the_first():
    """A bar is re-read many times and only the last one is what it
    settled on."""
    from liqmap import web as w

    rt = w.Runtime.__new__(w.Runtime)
    rt._last_state = {}
    rt.remember_state("BTC", "5m", 0.0,
                      rs.Result(state=rs.State.CONSOLIDATION, side=None))
    settled = rs.Result(state=rs.State.CONTINUATION, side="buy")
    rt.remember_state("BTC", "5m", 0.0, settled)
    rt.remember_state("BTC", "5m", 300.0,
                      rs.Result(state=rs.State.ABSORPTION, side="sell"))
    assert rt.previous_state("BTC", "5m", 300.0) is settled


def test_markets_and_timeframes_keep_their_own_memory():
    from liqmap import web as w

    rt = w.Runtime.__new__(w.Runtime)
    rt._last_state = {}
    a = rs.Result(state=rs.State.CONTINUATION, side="buy")
    rt.remember_state("BTC", "1m", 0.0, a)
    rt.remember_state("BTC", "1m", 60.0, rs.Result())
    rt.remember_state("ETH", "1m", 60.0, rs.Result())
    assert rt.previous_state("BTC", "1m", 60.0) is a
    assert rt.previous_state("ETH", "1m", 60.0) is None
    assert rt.previous_state("BTC", "5m", 60.0) is None


# ------------------------------------------------------------- the panel


def test_the_verdict_sits_above_the_layers_that_produced_it():
    src = _src()
    assert 'id="layerResult"' in src
    panel = src[src.index('id="layerPanel"'):]
    assert panel.index('id="layerResult"') < panel.index('id="layerBody"')


def test_a_provisional_word_is_drawn_differently_from_a_settled_one():
    """Drawing them the same hands a word the confidence the moment does
    not have."""
    src = _src()
    fn = src[src.index("function paintLayers(d) {"):]
    fn = fn[:fn.index("\nfunction paintLadder(")]
    assert "R.provisional" in fn
    assert "close the" in fn


def test_the_route_classifies_with_the_previous_bar():
    src = _src()
    block = src[src.index("lay: dict[str, Any] = {}"):]
    block = block[:block.index('out["timeframes"] = [')]
    assert "rsl.classify(" in block
    assert "previous=rt.previous_state(" in block
    assert block.index("rsl.classify(") < block.index("rt.remember_state(")
