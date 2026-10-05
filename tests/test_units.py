"""The unit a market is read in, and where that unit came from."""
from __future__ import annotations
import pytest
from liqmap import units as un
from liqmap.flow import Book, Level


def book_at(bid, ask, tick=0.25, n=6):
    return Book(coin="X", ts=0.0,
                bids=[Level(px=bid - i * tick, sz=10.0) for i in range(n)],
                asks=[Level(px=ask + i * tick, sz=10.0) for i in range(n)])


def test_an_observed_increment_beats_an_assumed_one():
    """The book is the only figure here that was measured. A per-market
    table would be a thousand chances to be wrong about one market."""
    u = un.from_book("SP500", book_at(7731.0, 7731.1, tick=0.1))
    assert u.name == "tick"
    assert u.measured
    assert u.size == pytest.approx(0.1)


def test_fx_is_read_in_pips_whatever_the_book_shows():
    """A EURUSD book quoted to five decimals has a tenth-pip increment.
    Reporting a move as 35 ticks when every FX trader would say 3.5 pips
    is accuracy that costs comprehension."""
    u = un.from_book("EURUSD", book_at(1.08000, 1.08001, tick=0.00001))
    assert u.name == "pip"
    assert u.size == pytest.approx(un.PIP)


def test_the_yen_pip_is_the_second_decimal():
    assert un.for_market("USDJPY").size == pytest.approx(un.JPY_PIP)
    assert un.for_market("EURUSD").size == pytest.approx(un.PIP)


def test_an_index_with_no_book_is_read_in_points():
    u = un.for_market("SP500", price=7731.0)
    assert u.name == "point" and u.size == pytest.approx(1.0)
    assert not u.measured


def test_a_unit_always_says_where_it_came_from():
    """An assumed tick and a measured one produce the same number on the
    screen and mean different things."""
    assert un.from_book("SP500", book_at(7731.0, 7731.1, tick=0.1)).source == "book"
    assert un.for_market("SP500", price=7731.0).source == "class"
    assert not un.for_market("SP500", price=7731.0).measured


def test_an_unknown_market_still_gets_a_scale_correct_unit():
    """Something quoted at 7,731 does not move in hundredths."""
    big = un.for_market("WEIRDPERP", price=7731.0)
    small = un.for_market("WEIRDPERP", price=0.004)
    assert big.size > small.size
    assert big.source == "price"


def test_bps_converts_to_the_unit_and_back():
    u = un.for_market("SP500", price=7731.0)
    n = u.from_bps(10.0, 7731.0)
    assert n == pytest.approx(7.731, rel=0.01)
    assert u.to_bps(n, 7731.0) == pytest.approx(10.0, rel=0.01)


def test_a_price_difference_converts_straight_across():
    u = un.for_market("ES", price=5000.0, tick=0.25)
    assert u.of(1.25) == pytest.approx(5.0)
    assert u.of(-1.25) == pytest.approx(-5.0)


def test_the_number_is_formatted_with_its_label_and_sign():
    u = un.for_market("SP500", price=7731.0)
    assert u.fmt(3.0) == "+3pt"
    assert u.fmt(-3.0) == "-3pt"
    assert u.fmt(3.0, sign=False) == "3pt"


def test_a_coarse_increment_is_not_quoted_to_a_tenth():
    """A tenth of a point is not a thing that happens on an index."""
    assert un.for_market("SP500", price=7731.0).places == 0
    assert un.for_market("ES", price=5000.0, tick=0.25).places == 1


def test_no_book_and_no_price_is_not_a_crash():
    u = un.from_book("BTC", None)
    assert u.size > 0
    assert u.of(1.0) != 0.0


def test_a_zero_sized_unit_never_divides():
    u = un.Unit(name="tick", size=0.0, label="t", source="book")
    assert u.of(5.0) == 0.0
    assert u.from_bps(10.0, 100.0) == 0.0


def test_bars_are_a_fallback_source_and_are_labelled_as_such():
    from liqmap.structure import Candle

    bars = [Candle(ts=i * 60.0, open=100.0 + i * 0.25, high=100.5 + i * 0.25,
                   low=99.75 + i * 0.25, close=100.25 + i * 0.25, volume=1.0)
            for i in range(40)]
    u = un.from_bars("WEIRD", bars)
    assert u.source == "bars" and u.measured
