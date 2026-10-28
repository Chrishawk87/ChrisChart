"""OFI agreeing with delta: did price go that way, and by how much.

In order of how badly each would mislead:

  folding flats into the right or the wrong column. At one bar on a
  one-tick market more than three quarters of outcomes are exactly
  zero, so the same 50% rule reads as 11% or as 89% depending which
  way you fold -- the single largest lever on the headline number

  quoting one percentage without the other. "Of all calls" counts
  flats against you and "of decided" sets them aside; both are true
  and they are far apart

  scoring calls that have not finished. The last h bars have no
  outcome at horizon h, and a zero for them lands in the flat column

  a thin bar's delta making a call. One contract flips the sign
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from liqmap import agree as A  # noqa: E402
from liqmap.ofibar import OfiBarBuilder  # noqa: E402

RAW = 250_000_000


class Bar:
    def __init__(self, direction="chop", delta=0.0, volume=100.0,
                 close=0.0, ofi=0, trade_count=5, ts_close=0.0):
        self.direction = direction
        self.delta = delta
        self.volume = volume
        self.close = close
        self.ofi = ofi
        self.trade_count = trade_count
        self.ts_close = ts_close


class TestTheRule:

    def test_green_with_positive_delta_is_an_up_call(self):
        assert A.call_of(Bar("green", delta=50)) == A.UP

    def test_red_with_negative_delta_is_a_down_call(self):
        assert A.call_of(Bar("red", delta=-50)) == A.DOWN

    def test_green_with_negative_delta_is_no_call(self):
        assert A.call_of(Bar("green", delta=-50)) == A.NOCALL

    def test_red_with_positive_delta_is_no_call(self):
        assert A.call_of(Bar("red", delta=50)) == A.NOCALL

    def test_a_chop_bar_is_never_a_call(self):
        assert A.call_of(Bar("chop", delta=50)) == A.NOCALL
        assert A.call_of(Bar("chop", delta=-50)) == A.NOCALL

    def test_zero_delta_is_no_call(self):
        assert A.call_of(Bar("green", delta=0)) == A.NOCALL

    def test_a_thin_bar_makes_no_call(self):
        """One contract either way flips the sign."""
        assert A.call_of(Bar("green", delta=3, volume=4)) == A.NOCALL

    def test_the_floor_is_on_volume_not_delta(self):
        assert A.call_of(Bar("green", delta=1, volume=500)) == A.UP

    def test_nothing_else_is_consulted(self):
        """No level, no price path, no target, no stop."""
        import re
        src = (ROOT / "liqmap" / "agree.py").read_text()
        i = src.index("def call_of")
        body = src[i:src.index("def entry_of")]
        read = set(re.findall(r'getattr\(bar,\s*"([a-z_]+)"', body))
        assert read == {"volume", "delta", "direction", "call_side"}, read


class TestTheCallIsMadeInsideTheBar:
    """Not after it closes. The close is minutes of information later,
    and measuring from it credits the reading with a move that had
    already happened before it could be taken."""

    def test_the_latched_intrabar_side_wins(self):
        b = Bar("green", delta=-50)         # the CLOSING state says no call
        b.call_side = "up"                  # but it fired up inside the bar
        assert A.call_of(b) == A.UP

    def test_a_bar_that_never_agreed_is_no_call(self):
        b = Bar("green", delta=50)
        b.call_side = ""
        assert A.call_of(b) == A.UP, "falls back to the closing form"

    def test_the_entry_is_the_price_it_fired_at(self):
        b = Bar("green", delta=50, close=100.0)
        b.call_px = 97.0
        assert A.entry_of(b) == 97.0

    def test_without_a_call_price_the_close_stands_in(self):
        assert A.entry_of(Bar("green", delta=50, close=100.0)) == 100.0

    def test_the_move_runs_from_the_call_price(self):
        a = Bar("green", delta=50, close=10.0)
        a.call_px = 7.0
        bars = [a, Bar("chop", close=12.0)]
        r = A.measure(bars, horizons=(1,))[0]
        assert r.mean_ticks == pytest.approx(5.0), \
            "measured from the close it would read 2.0"

    def test_the_catalogue_carries_both_prices(self):
        a = Bar("green", delta=50, close=10.0)
        a.call_px, a.call_ts = 7.0, 1234.5
        rows = A.catalogue([a, Bar("chop", close=12.0)], horizons=(1,))
        assert rows[0]["entry"] == 7.0
        assert rows[0]["bar_close"] == 10.0
        assert rows[0]["call_ts"] == 1234.5

    def test_the_builder_records_it_when_the_imbalance_crosses(self):
        from liqmap.ofibar import OfiBarBuilder
        base = 7_700 * 10 ** 9
        b = OfiBarBuilder(tick_size=0.25, threshold=2000, max_s=900.0,
                          min_updates=2, call_level=500)
        b.on_book(0.0, base, 100, base + RAW, 100)
        assert b.live_call[0] == ""
        sz = 100
        for i in range(1, 200):
            sz += 60
            b.on_trade(i * 0.01, 8, "buy", base + RAW)
            b.on_book(i * 0.01, base, sz, base + RAW, 100)
        assert b.bars, "the feed should close at least one bar"
        called = [x for x in b.bars if x.call_side]
        assert called and called[0].call_side == A.UP
        assert called[0].call_px > 0

    def test_the_call_is_not_latched(self):
        """It was, and that was the bug: a bar whose imbalance, delta
        and body all went up still reported DOWN, because the call
        fired once early and stuck."""
        from liqmap.ofibar import OfiBarBuilder
        base = 7_700 * 10 ** 9
        b = OfiBarBuilder(tick_size=0.25, threshold=10 ** 7, max_s=1e9,
                          min_updates=2, call_level=500)
        b.on_book(0.0, base, 100, base + RAW, 100)
        ts, sz = 0.0, 100
        for _ in range(60):                     # sellers first
            ts += 0.01
            sz += 60
            b.on_trade(ts, 40, "sell", base)
            b.on_book(ts, base, 100, base + RAW, sz)
        assert b.live_call[0] == A.DOWN
        sz = 100
        for _ in range(400):                    # then buyers, hard
            ts += 0.01
            sz += 60
            b.on_trade(ts, 40, "buy", base + RAW)
            b.on_book(ts, base, sz, base + RAW, 100)
        assert len(b.bars) == 0, "it must all be one bar"
        assert b.live_call[0] == A.UP, "it never took the down call back"


class TestWhatCountsAsFinished:

    def test_a_call_is_skipped_at_a_horizon_it_cannot_reach(self):
        bars = [Bar("green", delta=50, close=float(i)) for i in range(5)]
        rows = {r.horizon: r for r in A.measure(bars, horizons=(1, 4))}
        assert rows[1].n == 4
        assert rows[4].n == 1

    def test_an_unreachable_horizon_produces_no_row_at_all(self):
        bars = [Bar("green", delta=50, close=float(i)) for i in range(3)]
        assert not [r for r in A.measure(bars, horizons=(10,))]

    def test_the_move_runs_from_the_called_bar(self):
        bars = [Bar("green", delta=50, close=0.0),
                Bar("chop", close=3.0), Bar("chop", close=7.0)]
        r = A.measure(bars, horizons=(2,))[0]
        assert r.mean_ticks == pytest.approx(7.0)

    def test_a_down_call_is_scored_in_its_own_direction(self):
        """A short that made two ticks reads +2."""
        bars = [Bar("red", delta=-50, close=10.0), Bar("chop", close=8.0)]
        r = A.measure(bars, horizons=(1,))[0]
        assert r.call == A.DOWN and r.mean_ticks == pytest.approx(2.0)

    def test_the_tick_divides_the_move(self):
        bars = [Bar("green", delta=50, close=0.0),
                Bar("chop", close=float(RAW))]
        r = A.measure(bars, horizons=(1,), tick=float(RAW))[0]
        assert r.mean_ticks == pytest.approx(1.0)


class TestTheCatalogue:

    def test_one_row_per_call_and_nothing_else(self):
        bars = [Bar("green", delta=50, close=0.0),
                Bar("chop", delta=0.0, close=1.0),
                Bar("red", delta=-50, close=2.0),
                Bar("chop", delta=0.0, close=3.0)]
        rows = A.catalogue(bars, horizons=(1,))
        assert [r["call"] for r in rows] == [A.UP, A.DOWN]

    def test_every_horizon_gets_a_column(self):
        bars = [Bar("green", delta=50, close=float(i)) for i in range(8)]
        rows = A.catalogue(bars, horizons=(1, 3))
        assert "h1" in rows[0] and "h3" in rows[0]

    def test_an_unfinished_horizon_is_empty_not_zero(self):
        """A zero there would read as a flat outcome."""
        bars = [Bar("green", delta=50, close=float(i)) for i in range(3)]
        rows = A.catalogue(bars, horizons=(1, 10))
        assert rows[0]["h10"] is None
        assert rows[0]["h1"] == 1.0

    def test_it_carries_what_made_the_call(self):
        bars = [Bar("green", delta=50, volume=300, ofi=260, close=1.0,
                    trade_count=9),
                Bar("chop", close=2.0)]
        r = A.catalogue(bars, horizons=(1,))[0]
        assert r["ofi"] == 260 and r["delta"] == 50
        assert r["volume"] == 300 and r["trades"] == 9

    def test_it_serialises(self):
        import json
        bars = walk(60_000)
        json.dumps(A.catalogue(bars, horizons=(1, 3), tick=RAW))


def walk(n=200_000, seed=7):
    import random
    rng = random.Random(seed)
    b = OfiBarBuilder(tick_size=0.25, threshold=250, max_s=900.0,
                      min_updates=2, max_bars=10 ** 7)
    bid, ts = 7_700 * 10 ** 9, 0.0
    for _ in range(n):
        if rng.random() < 0.08:
            bid += RAW if rng.random() < 0.5 else -RAW
        ts += 0.01
        if rng.random() < 0.35:
            px = bid if rng.random() < 0.5 else bid + RAW
            b.on_trade(ts, rng.randint(1, 20),
                       "buy" if rng.random() < 0.5 else "sell", px)
        b.on_book(ts, bid, rng.randint(50, 600), bid + RAW,
                  rng.randint(50, 600))
    return b.bars


class TestOnAMarketWithNothingInIt:
    """A market with no structure must come back at a coin flip."""

    def test_decided_calls_land_near_fifty_percent(self):
        bars = walk(400_000)
        rows = A.measure(bars, horizons=(1, 3, 10), tick=RAW)
        assert rows
        for r in rows:
            assert r.decided > 1_000, r.as_dict()
            assert 0.45 < r.pct_of_decided < 0.55, r.as_dict()

    def test_the_mean_move_is_near_zero(self):
        bars = walk(400_000)
        for r in A.measure(bars, horizons=(1, 3), tick=RAW):
            assert abs(r.mean_ticks) < 0.1, r.as_dict()

    def test_ups_and_downs_are_called_about_equally(self):
        bars = walk(400_000)
        rows = {r.call: r for r in A.measure(bars, horizons=(1,), tick=RAW)}
        a, b = rows[A.UP].n, rows[A.DOWN].n
        assert 0.8 < a / b < 1.25, (a, b)


class TestItDecidesNothing:

    def test_no_order_code(self):
        src = (ROOT / "liqmap" / "agree.py").read_text()
        for w in ("def submit(", "def execute(", "def place(", "def order("):
            assert w not in src, w

    def test_it_models_no_trade(self):
        """No fills, no costs, no size, no risk."""
        src = (ROOT / "liqmap" / "agree.py").read_text().lower()
        for w in ("stop loss", "take profit", "break_even", "position",
                  "slippage", "commission"):
            assert w not in src, w

    def test_it_makes_no_claim_about_what_comes_next(self):
        src = (ROOT / "liqmap" / "agree.py").read_text().lower()
        assert "nothing here predicts" in src
        for w in ("will rise", "will fall", "buy signal", "sell signal"):
            assert w not in src, w
