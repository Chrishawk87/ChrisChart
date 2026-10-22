"""Can the aggressive flow get through what is standing in front of it?

Four failure modes this file exists to prevent, all of them silent:

  comparing a short flow window against a long one, which makes the
  filter hundreds of times too strict and reads as a quiet market

  truncating the obstacle at the visible depth without saying so,
  which UNDERSTATES resistance and lets through exactly the trades
  the filter was built to block

  reporting "no visible resistance" as "clear", when hidden size
  appears at no depth at all

  counting a wall that withdrew as a wall that was eaten
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from liqmap.obstacle import (                              # noqa: E402
    BLOCKED, BUY, CLEARS, SELL, UNKNOWN, Clearance, Flow, Obstacle,
    Obstacles)

PT, TICK = 1_000_000_000, 250_000_000
BASE, T0 = 5800 * PT, 1_700_000_000.0


def book(ts, asks, bids=None, gap=1):
    bids = bids or [(100, 10)] * len(asks)
    o = type("R", (), {})()
    o.ts_event = int(ts * 1e9)
    o.action, o.side = "A", "B"
    lv = []
    for k, ((asz, act), (bsz, bct)) in enumerate(zip(asks, bids)):
        L = type("L", (), {})()
        L.bid_px, L.bid_sz, L.bid_ct = BASE - k * gap * TICK, bsz, bct
        L.ask_px, L.ask_sz, L.ask_ct = (BASE + (k + 1) * gap * TICK,
                                        asz, act)
        lv.append(L)
    o.levels = lv
    return o


class TestTheObstacleIsOnlyWhatYouCanSee:
    """MBP-10 is ten levels. Four points is sixteen ticks. Truncating
    silently understates resistance, making the ratio easier to clear
    and passing the trades the filter exists to stop."""

    def test_a_short_book_is_reported_as_truncated(self):
        ob = Obstacles(reach_ticks=16)
        ob.observe_book(book(T0, [(50, 5)] * 10))
        o = ob.obstacle(BUY)
        assert o.truncated is True
        assert o.covered_ticks < o.asked_ticks
        assert o.levels == 10

    def test_a_book_that_reaches_far_enough_is_not(self):
        ob = Obstacles(reach_ticks=5)
        ob.observe_book(book(T0, [(50, 5)] * 10))
        o = ob.obstacle(BUY)
        assert o.truncated is False
        assert o.covered_ticks == 5

    def test_the_reach_is_measured_from_real_prices(self):
        """A gappy book covers more ground with the same ten levels.
        Assuming one-tick spacing would misreport both."""
        dense = Obstacles(reach_ticks=16)
        dense.observe_book(book(T0, [(50, 5)] * 10, gap=1))
        gappy = Obstacles(reach_ticks=16)
        gappy.observe_book(book(T0, [(50, 5)] * 10, gap=2))
        assert (gappy.obstacle(BUY).covered_ticks
                > dense.obstacle(BUY).covered_ticks)

    def test_levels_beyond_the_reach_are_excluded(self):
        ob = Obstacles(reach_ticks=3)
        ob.observe_book(book(T0, [(50, 5)] * 10))
        assert ob.obstacle(BUY).levels < 10

    def test_the_reach_is_inclusive_of_the_quote_it_starts_from(self):
        """To move four points up you must eat the best ask AND the
        sixteen ticks above it, so a reach of N spans N+1 levels on a
        one-tick book. Excluding the anchor would understate the
        obstacle by the one level most likely to stop the trade."""
        ob = Obstacles(reach_ticks=3)
        ob.observe_book(book(T0, [(50, 5)] * 10))
        o = ob.obstacle(BUY)
        assert o.levels == 4
        assert o.nearest_px == 5800.25 and o.farthest_px == 5801.00

    def test_orders_are_counted_as_well_as_contracts(self):
        ob = Obstacles(reach_ticks=5)
        ob.observe_book(book(T0, [(50, 5)] * 10))
        o = ob.obstacle(BUY)
        assert o.size == 300 and o.orders == 30

    def test_a_buy_is_blocked_by_asks_and_a_sell_by_bids(self):
        ob = Obstacles(reach_ticks=4)
        ob.observe_book(book(T0, [(80, 8)] * 10, [(20, 2)] * 10))
        assert ob.obstacle(BUY).side == "ask"
        assert ob.obstacle(SELL).side == "bid"
        assert ob.obstacle(BUY).size > ob.obstacle(SELL).size


class TestNoVisibleResistanceIsNotClear:
    """Icebergs and reserve orders appear at no depth. The obstacle is
    a LOWER bound on resistance, so the ratio is an UPPER bound on how
    easily anything gets through."""

    def test_an_empty_book_reports_unknown(self):
        ob = Obstacles(reach_ticks=16)
        ob.observe_book(book(T0, [(0, 0)] * 10))
        assert ob.clearance(BUY).verdict == UNKNOWN

    def test_unknown_does_not_pass(self):
        ob = Obstacles(reach_ticks=16)
        ob.observe_book(book(T0, [(0, 0)] * 10))
        assert ob.clearance(BUY).passes is False

    def test_no_book_at_all_reports_unknown(self):
        assert Obstacles().clearance(BUY).verdict == UNKNOWN


class TestLikeForLike:
    """Flow is contracts-over-time, an obstacle is contracts. The
    headline is SECONDS TO CLEAR, which is a unit a person can argue
    with -- not a ratio of two different windows."""

    def _fed(self, obstacle=600, per=20, n=40, step=0.25, thr=1.2):
        ob = Obstacles(reach_ticks=16, window_s=10.0, threshold=thr)
        ob.observe_book(book(T0, [(obstacle // 10, 6)] * 10))
        t = T0
        for _ in range(n):
            t += step
            ob.observe_trade(t, "buy", per, price=5800.25)
        return ob

    def test_heavy_flow_clears_a_light_obstacle(self):
        c = self._fed(obstacle=600, per=20).clearance(BUY)
        assert c.verdict == CLEARS and c.ratio > 1.2

    def test_light_flow_is_blocked_by_a_heavy_obstacle(self):
        c = self._fed(obstacle=6000, per=2).clearance(BUY)
        assert c.verdict == BLOCKED

    def test_seconds_to_clear_is_reported(self):
        c = self._fed().clearance(BUY)
        assert c.seconds_to_clear is not None
        assert c.seconds_to_clear == pytest.approx(
            c.obstacle.size / c.rate, rel=0.01)

    def test_the_window_is_the_flows_own_span_not_an_assumption(self):
        c = self._fed(n=8, step=0.25).clearance(BUY)
        assert c.window_s == pytest.approx(1.75, abs=0.3)

    def test_old_flow_ages_out_of_the_window(self):
        ob = Obstacles(reach_ticks=16, window_s=2.0)
        ob.observe_book(book(T0, [(60, 6)] * 10))
        for i in range(20):
            ob.observe_trade(T0 + i * 0.05, "buy", 50, price=5800.25)
        heavy = ob.clearance(BUY).flow
        ob.observe_trade(T0 + 30, "buy", 1, price=5800.25)
        assert ob.clearance(BUY).flow < heavy

    def test_the_threshold_is_an_argument(self):
        ob = self._fed()
        assert ob.clearance(BUY, threshold=0.5).verdict == CLEARS
        assert ob.clearance(BUY, threshold=99.0).verdict == BLOCKED

    def test_opposing_flow_is_reported_not_netted(self):
        """Netting would hide a fight. Two hundred buying against two
        hundred selling is not the same as nobody trading."""
        ob = Obstacles(reach_ticks=16, window_s=10.0)
        ob.observe_book(book(T0, [(60, 6)] * 10))
        for i in range(10):
            ob.observe_trade(T0 + i * 0.1, "buy", 20, price=5800.25)
            ob.observe_trade(T0 + i * 0.1, "sell", 20, price=5800.00)
        c = ob.clearance(BUY)
        assert c.flow == 200 and c.opposing == 200


class TestAWallThatLeftIsNotAWallThatWasEaten:
    """Flow eating an obstacle is absorption. An obstacle withdrawing
    as flow approaches is something else, and a filter that cannot
    tell them apart keeps reporting breakthroughs of walls that
    simply moved."""

    def _shrink(self, with_trade: bool):
        ob = Obstacles(reach_ticks=16)
        ob.observe_book(book(T0, [(500, 50)] + [(50, 5)] * 9))
        if with_trade:
            ob.observe_trade(T0 + 1, "buy", 400, price=5800.25)
        ob.observe_book(book(T0 + 2, [(100, 10)] + [(50, 5)] * 9))
        return ob.clearance(BUY)

    def test_size_that_traded_away_is_consumed(self):
        c = self._shrink(True)
        assert c.consumed == 400 and c.pulled == 0

    def test_size_that_simply_left_is_pulled(self):
        c = self._shrink(False)
        assert c.pulled == 400 and c.consumed == 0

    def test_the_same_book_change_splits_both_ways(self):
        """Identical before and after. Only the tape distinguishes
        them."""
        a, b = self._shrink(True), self._shrink(False)
        assert a.obstacle.size == b.obstacle.size
        assert a.pulled_share == 0.0 and b.pulled_share == 1.0

    def test_a_partial_fill_splits_proportionally(self):
        ob = Obstacles(reach_ticks=16)
        ob.observe_book(book(T0, [(500, 50)] + [(50, 5)] * 9))
        ob.observe_trade(T0 + 1, "buy", 100, price=5800.25)
        ob.observe_book(book(T0 + 2, [(100, 10)] + [(50, 5)] * 9))
        c = ob.clearance(BUY)
        assert c.consumed == 100 and c.pulled == 300

    def test_growing_size_is_neither(self):
        ob = Obstacles(reach_ticks=16)
        ob.observe_book(book(T0, [(100, 10)] * 10))
        ob.observe_book(book(T0 + 1, [(400, 40)] * 10))
        c = ob.clearance(BUY)
        assert c.consumed == 0 and c.pulled == 0

    def test_the_two_sides_are_attributed_separately(self):
        ob = Obstacles(reach_ticks=16)
        ob.observe_book(book(T0, [(500, 50)] * 10, [(500, 50)] * 10))
        ob.observe_book(book(T0 + 1, [(100, 10)] * 10,
                             [(500, 50)] * 10))
        assert ob.clearance(BUY).pulled > 0
        assert ob.clearance(SELL).pulled == 0


class TestScopedToTheCandle:

    def test_reset_clears_the_counters(self):
        ob = Obstacles(reach_ticks=16)
        ob.observe_book(book(T0, [(500, 50)] * 10))
        ob.observe_trade(T0 + 1, "buy", 100, price=5800.25)
        ob.observe_book(book(T0 + 2, [(100, 10)] * 10))
        assert ob.clearance(BUY).consumed > 0
        ob.reset()
        c = ob.clearance(BUY)
        assert c.consumed == 0 and c.pulled == 0 and c.flow == 0

    def test_reset_keeps_the_standing_book(self):
        ob = Obstacles(reach_ticks=5)
        ob.observe_book(book(T0, [(50, 5)] * 10))
        before = ob.obstacle(BUY).size
        ob.reset()
        assert ob.obstacle(BUY).size == before


class TestFlowWindow:

    def test_it_counts_each_side(self):
        f = Flow(10.0)
        f.add(T0, "buy", 5)
        f.add(T0, "sell", 3)
        assert f.buy == 5 and f.sell == 3

    def test_zero_size_is_ignored(self):
        f = Flow(10.0)
        f.add(T0, "buy", 0)
        assert f.buy == 0

    def test_trimming_is_by_time(self):
        f = Flow(2.0)
        f.add(T0, "buy", 10)
        f.add(T0 + 5, "buy", 1)
        assert f.buy == 1


class TestItDecidesNothing:

    def test_no_order_code(self):
        src = (ROOT / "liqmap" / "obstacle.py").read_text()
        for w in ("def submit(", "def execute(", "def place(",
                  "def order("):
            assert w not in src, w

    def test_it_says_nothing_about_who_is_trading(self):
        """Size is size. A 500-lot from a bank and a 500-lot from an
        aggregated router are indistinguishable in this data, and
        naming one institutional is a story laid over a number."""
        src = (ROOT / "liqmap" / "obstacle.py").read_text().lower()
        for word in ("institution", "retail", "smart money", "whale"):
            assert word not in src, word

    def test_it_serialises(self):
        import json
        ob = Obstacles(reach_ticks=16)
        ob.observe_book(book(T0, [(50, 5)] * 10))
        ob.observe_trade(T0 + 1, "buy", 10, price=5800.25)
        json.dumps(ob.clearance(BUY).as_dict())
