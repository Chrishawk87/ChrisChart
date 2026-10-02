"""Velocity, instrument economics, and whether a limit order filled.

Two of these matter far more than the rest.

THE BASELINE MUST NOT CONTAIN THE BURST. A baseline window running up to
the present includes the spike being measured, so the loudest bursts --
the ones worth acting on -- are the ones the ratio understates most.

TOUCH IS NOT A FILL. A tester that fills a limit order because price
touched its level books the best price of a move it was never in, and it
does so in exactly one direction. Both have a test that fails if the code
does the convenient thing.
"""

from __future__ import annotations

import pytest

from liqmap import orderflow as of
from liqmap.flow import Book, Level


# ------------------------------------------------------- the arithmetic

def test_a_smaller_target_needs_a_higher_hit_rate():
    """The point Chris keeps arriving back at. Cost does not shrink with
    the target, so the smaller the prize the bigger a share of it the fee
    is -- a five tick scalp needs MORE accuracy than a twenty tick one."""
    five = of.ES.breakeven(5.0, 10.0)
    ten = of.ES.breakeven(10.0, 10.0)
    twenty = of.ES.breakeven(20.0, 10.0)
    assert five > ten > twenty


def test_the_es_round_trip_is_the_real_one():
    """$2.50 a side plus one tick of spread. Charging commission alone
    understates it by more than a tick, which is most of a five tick
    target."""
    assert of.ES.taker_cost() == pytest.approx(17.50)
    assert of.ES.cost_ticks(taker=True) == pytest.approx(1.4)


def test_posting_instead_of_crossing_removes_the_spread():
    assert of.ES.maker_cost() == pytest.approx(5.0)
    assert of.ES.cost_ticks(taker=False) == pytest.approx(0.4)
    assert of.ES.maker_cost() < of.ES.taker_cost()


def test_the_spread_is_charged_once_not_twice():
    """You give it up getting in; the exit is priced from where you
    already are. Charging it both ways double-counts one half-turn."""
    assert of.ES.taker_cost() == pytest.approx(
        2 * of.ES.commission + of.ES.tick_value)


def test_a_target_that_cannot_pay_for_itself_is_refused():
    """Not a hard trade -- an arithmetic problem. No entry rule supplies
    that much edge, so the configuration is refused before anything is
    tested on it."""
    assert not of.ES.viable(2.0, 30.0)
    assert of.ES.viable(20.0, 20.0)


def test_the_micro_contract_costs_proportionally_more():
    """MES is a tenth the tick value but nowhere near a tenth the
    commission, so the same tick target is a higher bar on it."""
    assert of.MES.breakeven(10.0, 10.0) > of.ES.breakeven(10.0, 10.0)


def test_equities_are_handled_by_the_same_arithmetic():
    e = of.EQUITY
    assert e.tick_size == 0.01
    assert e.breakeven(10.0, 10.0, taker=True) > 0.5


def test_every_spec_is_reachable_by_symbol():
    for sym in ("ES", "MES", "NQ", "EQUITY"):
        assert of.SPECS[sym].symbol == sym


# ---------------------------------------------------------- the spread

def test_the_spread_gate_scales_with_the_target():
    """The same spread is trivial against forty ticks and fatal against
    five, so the gate is a share rather than a tick count."""
    assert of.spread_ok(1.0, 40.0)
    assert not of.spread_ok(1.0, 4.0)


def test_a_zero_target_is_never_tradeable():
    assert not of.spread_ok(0.0, 0.0)


# --------------------------------------------------------- the velocity

def even(start, n, hz):
    return [start + i / hz for i in range(n)]


def test_the_baseline_excludes_the_burst_it_is_measuring():
    """THE velocity test.

    Ten minutes at 1/s, then fifteen seconds at 10/s. A baseline running
    up to the present includes those fifteen seconds and reports well
    under 10x; one that stops where the burst starts reports 10x.
    """
    base = even(0.0, 600, 1.0)
    burst = even(600.0, 150, 10.0)
    v = of.velocity(base + burst, now=615.0)
    assert v.confident
    assert v.ratio == pytest.approx(10.0, rel=0.1)
    assert v.spiking(2.0)


def test_a_quiet_tape_does_not_read_as_a_spike():
    v = of.velocity(even(0.0, 700, 1.0), now=700.0)
    assert v.confident
    assert v.ratio == pytest.approx(1.0, abs=0.2)
    assert not v.spiking(2.0)


def test_too_little_history_refuses_to_call_a_spike():
    """Three prints in ten minutes then a flurry is not a 50x burst, it
    is a market nobody is trading."""
    v = of.velocity(even(0.0, 5, 0.01) + even(600.0, 40, 10.0), now=604.0)
    assert not v.confident
    assert not v.spiking(2.0)
    assert "not enough" in v.note


def test_no_prints_is_not_a_crash():
    v = of.velocity([])
    assert v.now_hz == 0.0 and not v.confident


def test_the_windows_are_stated():
    assert of.BURST_S == 15.0 and of.BASE_S == 600.0
    assert of.MIN_BASE_PRINTS >= 40


def test_the_dict_carries_what_the_panel_shows():
    d = of.velocity(even(0.0, 700, 1.0), now=700.0).to_dict()
    for k in ("now_hz", "base_hz", "ratio", "confident", "note"):
        assert k in d


# ------------------------------------------------------------ the queue

def book_at(bid, ask, bid_sz=100.0, ask_sz=100.0, tick=0.25):
    return Book(coin="ES", ts=0.0,
                bids=[Level(px=bid - i * tick, sz=bid_sz) for i in range(5)],
                asks=[Level(px=ask + i * tick, sz=ask_sz) for i in range(5)])


def test_the_queue_ahead_is_read_off_the_book():
    b = book_at(100.00, 100.25, bid_sz=250.0)
    assert of.queue_ahead(b, 100.00, "buy") == pytest.approx(250.0)
    assert of.queue_ahead(b, 99.75, "buy") == pytest.approx(250.0)
    assert of.queue_ahead(b, 90.00, "buy") == 0.0


def test_a_touch_that_does_not_clear_the_queue_is_not_a_fill():
    """THE fill test.

    Two hundred resting in front, forty trade, price leaves. A tester
    that fills here has handed itself the best price of a move it was
    never in -- and it does that in one direction only, because the
    times the queue DOES clear are the times price went through.
    """
    prints = [(1.0, 100.00, 20.0), (2.0, 100.00, 20.0),
              (3.0, 100.25, 50.0), (4.0, 100.50, 50.0)]
    f = of.resting_fill(100.00, "buy", ahead=200.0, prints=prints, tick=0.25)
    assert not f.filled
    assert f.traded_through == pytest.approx(40.0)
    assert "never reached it" in f.reason


def test_clearing_the_queue_is_a_fill():
    prints = [(1.0, 100.00, 120.0), (2.0, 100.00, 120.0)]
    f = of.resting_fill(100.00, "buy", ahead=200.0, prints=prints, tick=0.25)
    assert f.filled and not f.adverse
    assert f.ts == 2.0


def test_price_through_the_level_fills_and_is_marked_adverse():
    """The asymmetry the model exists for: a resting order fills most
    reliably exactly when it is about to be wrong."""
    prints = [(1.0, 100.00, 10.0), (2.0, 99.75, 50.0)]
    f = of.resting_fill(100.00, "buy", ahead=5000.0, prints=prints,
                        tick=0.25)
    assert f.filled and f.adverse
    assert "traded through" in f.reason


def test_the_short_side_mirrors():
    prints = [(1.0, 100.25, 10.0), (2.0, 100.50, 50.0)]
    f = of.resting_fill(100.25, "sell", ahead=5000.0, prints=prints,
                        tick=0.25)
    assert f.filled and f.adverse


def test_the_naive_model_fills_where_the_real_one_does_not():
    """Kept so the two can be compared rather than argued about. On the
    same tape the touch model is in at the best price and the queue model
    is not in at all."""
    prints = [(1.0, 100.00, 20.0), (2.0, 100.25, 50.0)]
    naive = of.touch_fill(100.00, "buy", prints, tick=0.25)
    real = of.resting_fill(100.00, "buy", ahead=200.0, prints=prints,
                           tick=0.25)
    assert naive.filled
    assert not real.filled


def test_an_empty_queue_fills_on_the_first_print_at_the_level():
    prints = [(1.0, 100.00, 1.0)]
    f = of.resting_fill(100.00, "buy", ahead=0.0, prints=prints, tick=0.25)
    assert f.filled


def test_every_fill_states_what_it_assumed():
    """Cancellations ahead are invisible and hidden size is not counted.
    The two do not cancel and nobody knows the net, so the assumptions
    ride along with the answer rather than living in a docstring."""
    f = of.resting_fill(100.00, "buy", ahead=10.0,
                        prints=[(1.0, 100.00, 20.0)], tick=0.25)
    assert len(f.optimism) >= 2
    assert any("cancellation" in o for o in f.optimism)
    assert any("hidden" in o or "iceberg" in o for o in f.optimism)


def test_an_empty_tape_fills_nothing():
    assert not of.resting_fill(100.0, "buy", 10.0, [], 0.25).filled
    assert not of.touch_fill(100.0, "buy", [], 0.25).filled


def test_the_clear_factor_is_stated_and_not_below_one():
    """Exactly one fills on the last contract of the queue, which already
    assumes nothing joined behind it. Below one would be inventing
    priority."""
    assert of.CLEAR_FACTOR >= 1.0
