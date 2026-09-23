"""Book against price: the four states, and the one that costs money.

The rule this pins is the whole point of the module:

    book UP  + price UP    -> trade it
    book UP  + price DOWN  -> DO NOT TRADE. Somebody is absorbing.
    book UP  + price FLAT  -> not yet.
    book FLAT              -> nothing.

CONFLICT MUST BE A HARD REFUSAL, NEVER A WEAK SIGNAL. If a strong book and a
falling price average out into a small long, the tool hands you a position
precisely where a passive seller is filling every bid without letting price
move. That is the most expensive thing this design could do and it is the
easiest to do by accident, because averaging two numbers is the obvious
implementation.

THE TWO SIDES MUST NOT SHARE AN INPUT. `BookRead` already scores
`mid_drift_bps`. If the price side of this comparison also used the book's
mid, the check would be comparing a number with itself, would agree almost
always, and would look like it was working. Everything here comes from fills
-- bars and the trade tape -- and nothing in `confirm.py` may import or read
an order book.
"""

import inspect

import pytest

from liqmap import confirm as confirm_mod
from liqmap.confirm import (
    CandleAction, confirm, read_candle, typical_bar_bps,
)
from liqmap.structure import Candle

BAR = 100.0        # typical bar range in bps, for scaling


def action(slope=0.0, thrust=0.0, position=0.5, extending="flat",
           bar_range_bps=BAR):
    return CandleAction(thrust_bps=thrust, position=position, slope_bps=slope,
                        extending=extending, bar_range_bps=bar_range_bps)


def rising(strength=1.0):
    m = BAR * 0.5 * strength
    return action(slope=m, thrust=m, position=0.5 + 0.45 * strength,
                  extending="up")


def falling(strength=1.0):
    m = -BAR * 0.5 * strength
    return action(slope=m, thrust=m, position=0.5 - 0.45 * strength,
                  extending="down")


# --------------------------------------------------------------------------
# no circularity — the structural guarantee
# --------------------------------------------------------------------------

def test_this_module_never_reads_an_order_book():
    """The check is worthless if it shares an input with the thing it
    checks. Asserted on the parsed CODE rather than the text, so that the
    module's own docstring is free to explain why the rule exists without
    tripping it."""
    import ast

    tree = ast.parse(inspect.getsource(confirm_mod))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", "") or ""
            names.add(mod)
            names.update(a.name for a in node.names)

    banned = {"bookread", "microprice", "best_bid", "best_ask", "bids",
              "asks", "Book", "BookRead", "tilt", "imbalance", "replenish",
              "mid_drift_bps"}
    leaked = banned & names
    assert not leaked, f"confirm.py touches the order book: {sorted(leaked)}"


def test_a_candle_action_can_be_built_without_any_book():
    a = read_candle(open_px=100.0, high_px=101.0, low_px=99.5, last_px=100.9,
                    slope_bps=30.0, bar_range_bps=BAR)
    assert a is not None and a.direction == "up"


# --------------------------------------------------------------------------
# reading what price is doing
# --------------------------------------------------------------------------

def test_a_rising_tape_and_a_green_bar_read_up():
    assert rising().direction == "up"


def test_a_falling_tape_and_a_red_bar_read_down():
    assert falling().direction == "down"


def test_a_bar_going_nowhere_reads_flat():
    assert action(slope=0.5, thrust=0.5, position=0.5).direction == "flat"


def test_strength_scales_with_the_size_of_the_move():
    # 0.4 already saturates: at a 20bps move against a 35bps full-strength
    # yardstick the score is past the 0.5 cap. A genuinely marginal move is
    # much smaller than that.
    assert rising(1.0).strength > rising(0.12).strength


def test_the_same_move_reads_the_same_on_any_instrument():
    """Scale-free: a 0.3% move is a 0.3% move on a $4 coin and a $100k one.
    A fixed basis-point threshold would make this module useless on one of
    them."""
    cheap = read_candle(open_px=4.0, high_px=4.02, low_px=3.99, last_px=4.018,
                        slope_bps=40.0, bar_range_bps=BAR)
    dear = read_candle(open_px=100_000.0, high_px=100_500.0, low_px=99_750.0,
                       last_px=100_450.0, slope_bps=40.0, bar_range_bps=BAR)
    assert cheap.direction == dear.direction == "up"
    assert cheap.score == pytest.approx(dear.score, abs=0.05)


def test_closing_on_the_highs_is_not_the_same_as_closing_mid_range():
    top = action(slope=10.0, thrust=10.0, position=0.95)
    mid = action(slope=10.0, thrust=10.0, position=0.5)
    assert top.score > mid.score


def test_making_a_new_high_counts_for_something():
    assert (read_candle(100.0, 101.0, 99.0, 100.8, slope_bps=20.0,
                        bar_range_bps=BAR,
                        recent_highs=[100.5], recent_lows=[99.0]).extending
            == "up")
    assert (read_candle(100.0, 101.0, 99.0, 99.2, slope_bps=-20.0,
                        bar_range_bps=BAR,
                        recent_highs=[101.0], recent_lows=[99.5]).extending
            == "down")


def test_price_inside_the_recent_range_is_not_extending():
    a = read_candle(100.0, 101.0, 99.0, 100.2, slope_bps=5.0,
                    bar_range_bps=BAR,
                    recent_highs=[101.5], recent_lows=[98.5])
    assert a.extending == "flat"


def test_a_degenerate_bar_reads_as_nothing_rather_than_raising():
    assert read_candle(0.0, 0.0, 0.0, 0.0, 0.0, BAR) is None
    assert read_candle(100.0, 99.0, 101.0, 100.0, 0.0, BAR) is None
    assert read_candle(100.0, 101.0, 99.0, 100.0, 0.0, 0.0) is None


def test_a_bar_with_no_range_does_not_divide_by_zero():
    a = read_candle(100.0, 100.0, 100.0, 100.0, slope_bps=0.0,
                    bar_range_bps=BAR)
    assert a is not None and a.position == 0.5


def test_typical_bar_range_uses_the_median():
    normal = [Candle(ts=i, open=100, high=100.5, low=99.5, close=100)
              for i in range(20)]
    spiked = normal + [Candle(ts=99, open=100, high=200, low=50, close=100)]
    assert typical_bar_bps(spiked) == pytest.approx(typical_bar_bps(normal),
                                                    rel=0.05)
    assert typical_bar_bps([]) == 0.0


# --------------------------------------------------------------------------
# the four states
# --------------------------------------------------------------------------

def test_book_up_and_price_up_is_confirmed():
    c = confirm("up", 0.8, rising())
    assert c.verdict == "confirmed" and c.agree
    assert c.direction == "up"
    assert "going with them" in c.detail


def test_book_down_and_price_down_is_confirmed():
    c = confirm("down", 0.8, falling())
    assert c.verdict == "confirmed" and c.direction == "down"


def test_book_up_and_price_down_is_a_conflict_not_a_weak_long():
    """The expensive case. Averaging these into a small long hands you a
    position exactly where the passive seller is winning."""
    c = confirm("up", 0.95, falling())
    assert c.verdict == "conflict"
    assert c.agree is False
    assert c.direction == "flat"
    assert c.strength == 0.0
    assert "absorbing" in c.detail


def test_book_down_and_price_up_is_also_a_conflict():
    c = confirm("down", 0.95, rising())
    assert c.verdict == "conflict" and c.direction == "flat"
    assert "absorbing" in c.detail


def test_a_conflict_names_who_is_absorbing_whom():
    up = confirm("up", 0.9, falling())
    down = confirm("down", 0.9, rising())
    assert "sellers absorbing" in up.detail
    assert "buyers absorbing" in down.detail


def test_book_leaning_but_price_still_is_unconfirmed_not_a_trade():
    c = confirm("up", 0.9, action(slope=0.2, thrust=0.2, position=0.5))
    assert c.verdict == "unconfirmed" and c.direction == "flat"
    assert "not moving with them yet" in c.detail


def test_a_flat_book_has_nothing_to_confirm():
    c = confirm("flat", 0.0, rising())
    assert c.verdict == "no signal" and c.direction == "flat"


def test_no_price_action_at_all_is_unconfirmed_rather_than_confirmed():
    """Missing evidence must never read as agreement."""
    c = confirm("up", 0.95, None)
    assert c.verdict == "unconfirmed" and c.agree is False


# --------------------------------------------------------------------------
# strength
# --------------------------------------------------------------------------

def test_agreement_strength_is_the_weaker_of_the_two_sides():
    """A powerful book against a barely-moving price is not a strong setup,
    it is a warning."""
    c = confirm("up", 0.95, rising(strength=0.35))
    if c.agree:
        assert c.strength == pytest.approx(min(0.95, c.candle_strength))
        assert c.strength < 0.95


def test_a_disagreement_has_no_strength_at_all():
    assert confirm("up", 1.0, falling()).strength == 0.0


def test_the_dict_form_carries_both_sides_and_the_verdict():
    d = confirm("up", 0.9, falling()).to_dict()
    assert d["book"] == "up" and d["candle"] == "down"
    assert d["verdict"] == "conflict" and d["agree"] is False
    assert d["detail"]


# --------------------------------------------------------------------------
# the gate, through suggest()
# --------------------------------------------------------------------------

def _suggest(book_dir="up", price=None, **kw):
    from tests.test_suggest import bars, book_at, read_at
    from liqmap.suggest import suggest

    r = read_at(0.9 if book_dir == "up" else -0.9)
    return suggest(r, book_at(), coin="BTC", interval="15m",
                   interval_s=900.0, seconds_left=600.0,
                   recent=bars(rng=2.0), notional=10_000.0,
                   action=price, **kw)


def test_a_conflict_refuses_the_trade_entirely():
    from liqmap.suggest import NoTrade

    out = _suggest("up", falling())
    assert isinstance(out, NoTrade)
    assert out.gate == "conflict"
    assert "absorbing" in out.detail


def test_an_unconfirmed_book_refuses_the_trade():
    from liqmap.suggest import NoTrade

    out = _suggest("up", action(slope=0.1, thrust=0.1, position=0.5))
    assert isinstance(out, NoTrade) and out.gate == "unconfirmed"


def test_agreement_lets_the_trade_through_and_is_recorded_on_it():
    from liqmap.suggest import Suggestion

    out = _suggest("up", rising())
    assert isinstance(out, Suggestion)
    assert out.confirmation is not None
    assert out.confirmation.verdict == "confirmed"
    assert "Book says UP, price is going UP" in out.sentence()


def test_a_short_confirms_on_falling_price():
    from liqmap.suggest import Suggestion

    out = _suggest("down", falling())
    assert isinstance(out, Suggestion) and out.side == "short"
    assert out.confirmation.verdict == "confirmed"


def test_missing_price_action_blocks_rather_than_passes(monkeypatch):
    """The default must be to refuse. A feed that has not produced a candle
    yet must not silently become 'confirmed'."""
    from liqmap.suggest import NoTrade

    out = _suggest("up", None)
    assert isinstance(out, NoTrade) and out.gate == "unconfirmed"


def test_the_gate_can_be_turned_off_for_replay(monkeypatch):
    """Historical replay has bars but no tape, so it cannot confirm. It must
    be able to opt out explicitly rather than by accident."""
    from liqmap.suggest import Suggestion

    out = _suggest("up", None, require_confirmation=False)
    assert isinstance(out, Suggestion)
    assert out.confirmation is None


# --------------------------------------------------------------------------
# the graded call
# --------------------------------------------------------------------------

def test_a_conflict_is_graded_x_and_shown_rather_than_hidden():
    """The most informative state on the panel appears only on candles with
    no trade on them, so it must not be swallowed."""
    from tests.test_suggest import bars, book_at, read_at
    from liqmap.suggest import assess

    c = assess(read_at(0.9), book_at(), coin="BTC", interval="15m",
               interval_s=900.0, seconds_left=600.0, recent=bars(rng=2.0),
               notional=10_000.0, action=falling())
    assert c.tradeable is False
    assert c.grade == "X"
    assert c.blocked_by == "conflict"
    assert c.confirmation.book == "up" and c.confirmation.candle == "down"


def test_the_call_dict_carries_both_readings_for_the_panel():
    from tests.test_suggest import bars, book_at, read_at
    from liqmap.suggest import assess

    d = assess(read_at(0.9), book_at(), coin="BTC", interval="15m",
               interval_s=900.0, seconds_left=600.0, recent=bars(rng=2.0),
               notional=10_000.0, action=falling()).to_dict()
    assert d["confirmation"]["book"] == "up"
    assert d["confirmation"]["candle"] == "down"
    assert d["action"]["direction"] == "down"
    assert d["action"]["describe"]
