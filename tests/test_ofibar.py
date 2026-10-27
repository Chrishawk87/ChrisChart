"""OFI-closed candles, and the ways the construction goes wrong.

In order of how badly each would mislead:

  a global accumulator that never resets. Cumulative OFI tracks
  cumulative price, so on a trending session it walks away from zero
  and the threshold latches -- every bar after the first is green and
  the chart has only one word. This is the first thing tested, with
  the mirror case beside it

  a bar that closes on one enormous update, giving open == high ==
  low == close: a candle with no body and no range, which looks like
  a data error on the chart and is not

  a chop bar with no terminating condition, so a quiet stretch
  produces a bar that never closes and a chart that stops updating
  while appearing healthy

  contributing a zero from a crossed or absent touch, which sums like
  a measurement and is not one
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from liqmap.ofi import Touch                       # noqa: E402
from liqmap.ofibar import (CHOP, GREEN, RED, ROW_COLUMNS,  # noqa: E402
                           OfiBar, OfiBarBuilder)


def t(bid_px, bid_sz, ask_px=None, ask_sz=400):
    return Touch(bid_px=bid_px, bid_sz=bid_sz,
                 ask_px=ask_px if ask_px is not None else bid_px + 1,
                 ask_sz=ask_sz)


class TestTheAccumulatorResets:
    """The bug this module exists to not have."""

    def _drift(self, b, n=60_000, up_prob=0.52, seed=4):
        rng = random.Random(seed)
        bid, ts = 20_000, 0.0
        for _ in range(n):
            if rng.random() < 0.08:
                bid += 1 if rng.random() < up_prob else -1
            ts += 0.01
            b.on_book(ts, bid, rng.randint(50, 600), bid + 1,
                      rng.randint(50, 600))
        return b.bars

    def test_a_trending_book_still_prints_both_colours(self):
        """With a global accumulator this came out 99.9% green."""
        bars = self._drift(OfiBarBuilder(threshold=250, max_s=30.0))
        greens = sum(1 for x in bars if x.direction == GREEN)
        reds = sum(1 for x in bars if x.direction == RED)
        assert len(bars) > 50, "the chart must actually produce bars"
        assert greens > 0 and reds > 0
        share = greens / (greens + reds)
        assert 0.3 < share < 0.75, \
            f"green share {share:.1%} -- the threshold is latching"

    def test_and_the_mirror_on_a_falling_book(self):
        bars = self._drift(OfiBarBuilder(threshold=250, max_s=30.0),
                           up_prob=0.48, seed=9)
        greens = sum(1 for x in bars if x.direction == GREEN)
        reds = sum(1 for x in bars if x.direction == RED)
        assert greens > 0 and reds > 0
        assert 0.25 < greens / (greens + reds) < 0.70

    def test_the_accumulator_is_zero_after_every_close(self):
        b = OfiBarBuilder(threshold=100, max_s=5.0)
        b.on_book(0.0, 20_000, 100, 20_001, 100)
        closed = None
        ts = 0.0
        while closed is None and ts < 50.0:
            ts += 0.01
            closed = b.on_book(ts, 20_000, 1_000, 20_001, 100)
        assert closed is not None and closed.direction == GREEN
        assert b.accumulated == 0, "a close must zero the accumulator"


class TestTheThreshold:

    def test_a_bar_closes_green_on_reaching_it(self):
        b = OfiBarBuilder(threshold=300, max_s=60.0)
        b.on_book(0.0, 20_000, 100, 20_001, 400)
        out = b.on_book(0.1, 20_000, 400, 20_001, 400)   # +300 on the bid
        assert out is not None
        assert out.direction == GREEN and out.ofi == 300

    def test_a_bar_closes_red_on_the_negative_line(self):
        b = OfiBarBuilder(threshold=300, max_s=60.0)
        b.on_book(0.0, 20_000, 400, 20_001, 400)
        out = b.on_book(0.1, 20_000, 100, 20_001, 400)   # -300
        assert out is not None
        assert out.direction == RED and out.ofi == -300

    def test_below_the_line_nothing_closes(self):
        b = OfiBarBuilder(threshold=300, max_s=60.0)
        b.on_book(0.0, 20_000, 100, 20_001, 400)
        assert b.on_book(0.1, 20_000, 350, 20_001, 400) is None
        assert b.accumulated == 250

    def test_the_sign_is_not_inverted(self):
        """Size ARRIVING on the bid is buying pressure. An inverted
        sign would colour every bar backwards, which no amount of
        later analysis would recover."""
        b = OfiBarBuilder(threshold=100, max_s=60.0)
        b.on_book(0.0, 20_000, 100, 20_001, 100)
        out = b.on_book(0.1, 20_000, 300, 20_001, 100)
        assert out.direction == GREEN

        b2 = OfiBarBuilder(threshold=100, max_s=60.0)
        b2.on_book(0.0, 20_000, 100, 20_001, 100)
        out2 = b2.on_book(0.1, 20_000, 100, 20_001, 300)   # size on the OFFER
        assert out2.direction == RED


class TestTheChopBar:

    def test_a_quiet_stretch_terminates(self):
        """Without the cap this bar never closes and the chart freezes
        while looking perfectly healthy."""
        b = OfiBarBuilder(threshold=10_000, max_s=2.0)
        b.on_book(0.0, 20_000, 400, 20_001, 400)
        out = None
        ts = 0.0
        while out is None and ts < 10.0:
            ts += 0.1
            out = b.on_book(ts, 20_000, 400, 20_001, 400)
        assert out is not None and out.direction == CHOP
        assert out.seconds == pytest.approx(2.0, abs=0.15)

    def test_price_can_move_a_long_way_inside_a_chop_bar(self):
        """That is the entire point: price jumping without the flow
        committing is what the bar is reporting."""
        b = OfiBarBuilder(threshold=100_000, max_s=5.0)
        b.on_book(0.0, 20_000, 400, 20_001, 400)
        for i in range(40):
            px = 20_000 + (i % 8) - 4
            b.on_book(0.1 * (i + 1), px, 400, px + 1, 400)
        out = b.flush(10.0)
        assert out is not None and out.direction == CHOP
        assert out.high > out.low, "price moved"


class TestOHLC:

    def test_the_bar_tracks_its_own_extremes(self):
        b = OfiBarBuilder(threshold=100_000, max_s=100.0)
        b.on_book(0.0, 20_000, 400, 20_001, 400)
        for px in (20_002, 20_006, 19_996, 20_000):
            b.on_book(0.1, px, 400, px + 1, 400)
        cur = b.current
        assert cur.high == pytest.approx(20_006.5)
        assert cur.low == pytest.approx(19_996.5)
        assert cur.close == pytest.approx(20_000.5)

    def test_a_one_update_close_is_refused_when_min_updates_is_set(self):
        """One enormous size change can clear the line by itself and
        leave a candle with no range at all."""
        b = OfiBarBuilder(threshold=100, max_s=60.0, min_updates=3)
        b.on_book(0.0, 20_000, 100, 20_001, 100)
        assert b.on_book(0.1, 20_000, 5_000, 20_001, 100) is None
        assert b.on_book(0.2, 20_001, 5_000, 20_002, 100) is None
        out = b.on_book(0.3, 20_002, 5_000, 20_003, 100)
        assert out is not None and out.high > out.low

    def test_a_new_bar_opens_where_the_last_one_closed(self):
        b = OfiBarBuilder(threshold=100, max_s=60.0)
        b.on_book(0.0, 20_000, 100, 20_001, 100)
        out = b.on_book(0.1, 20_000, 300, 20_001, 100)
        assert b.current.open == pytest.approx(out.close), "no gap"


class TestInvalidBooks:

    def test_a_crossed_book_contributes_nothing(self):
        b = OfiBarBuilder(threshold=100, max_s=60.0)
        b.on_book(0.0, 20_000, 100, 20_001, 100)
        before = b.accumulated
        assert b.on_touch(0.1, Touch(bid_px=20_005, bid_sz=900,
                                     ask_px=20_001, ask_sz=100)) is None
        assert b.accumulated == before, "a crossed book is not a measurement"

    def test_an_absent_level_contributes_nothing(self):
        b = OfiBarBuilder(threshold=100, max_s=60.0)
        b.on_book(0.0, 20_000, 100, 20_001, 100)
        before = b.accumulated
        b.on_touch(0.1, Touch(bid_px=0, bid_sz=0, ask_px=0, ask_sz=0))
        assert b.accumulated == before


class TestTheEmitter:

    def test_the_row_is_the_column_order_the_chart_expects(self):
        b = OfiBarBuilder(threshold=100, max_s=60.0)
        b.on_book(0.0, 20_000, 100, 20_001, 100)
        out = b.on_book(0.5, 20_000, 300, 20_001, 100)
        row = out.as_row()
        assert len(row) == len(ROW_COLUMNS) == 9
        ts, o, h, l, c, colour, strength, pai, agg = row
        assert ts == pytest.approx(0.5), "stamped at the CLOSE"
        assert colour == GREEN
        assert strength == out.ofi
        assert l <= o <= h and l <= c <= h
        assert 0.0 <= pai <= 1.0
        assert agg in ("BUY_PRICE_AGGRESSIVE", "SELL_PRICE_AGGRESSIVE",
                       "BALANCED_PRICE_FLOW")

    def test_it_serialises(self):
        import json
        b = OfiBarBuilder(threshold=100, max_s=60.0)
        b.on_book(0.0, 20_000, 100, 20_001, 100)
        out = b.on_book(0.5, 20_000, 300, 20_001, 100)
        json.dumps(out.as_dict())
        json.dumps(out.as_row())

    def test_bars_accumulate_and_reset(self):
        b = OfiBarBuilder(threshold=100, max_s=60.0)
        b.on_book(0.0, 20_000, 100, 20_001, 100)
        b.on_book(0.5, 20_000, 300, 20_001, 100)
        assert len(b.bars) == 1
        b.reset()
        assert b.bars == [] and b.accumulated == 0


class TestItDecidesNothing:

    def test_no_order_code(self):
        src = (ROOT / "liqmap" / "ofibar.py").read_text()
        for w in ("def submit(", "def execute(", "def place(",
                  "def order("):
            assert w not in src, w

    def test_it_reuses_the_formula_rather_than_copying_it(self):
        """A second copy of e_n is a second thing to keep correct."""
        src = (ROOT / "liqmap" / "ofibar.py").read_text()
        assert "from .ofi import" in src
        assert "level_ofi" in src
        assert "bid_px > prev" not in src, "the formula was reimplemented"

    def test_it_does_not_claim_to_predict(self):
        src = (ROOT / "liqmap" / "ofibar.py").read_text().lower()
        assert "does not predict" in src or "not answer" in src
        for w in ("institution", "smart money", "whale", "guarantee"):
            assert w not in src, w

    def test_degenerate_settings_are_refused(self):
        for kw in ({"tick_size": 0}, {"threshold": 0},
                   {"max_s": 0}, {"min_updates": 0}):
            with pytest.raises(ValueError):
                OfiBarBuilder(**kw)


class TestPriceAggression:
    """PAI colours the body. What it must not be is the body.

    In order of how badly each would mislead:

      measuring the last five milliseconds and calling it the bar. At
      target 250 a bar runs twelve seconds, so that window is 0.04% of
      it -- a candle coloured from almost nothing

      measuring NET movement instead of the PATH, which makes PAI a
      restatement of close-versus-open and adds nothing to a chart
      that already draws the body

      carrying aggression across a close, so every bar inherits the
      first one's character
    """

    def _walk(self, path, threshold=10 ** 9):
        b = OfiBarBuilder(threshold=threshold, max_s=10 ** 9)
        ts = 0.0
        for px in path:
            ts += 0.1
            b.on_book(ts, px, 100, px + 1, 100)
        return b

    def test_a_clean_climb_is_buy_aggressive(self):
        b = self._walk(list(range(0, 12)))
        assert b.live_pai == pytest.approx(1.0)

    def test_a_clean_fall_is_sell_aggressive(self):
        b = self._walk(list(range(0, -12, -1)))
        assert b.live_pai == pytest.approx(0.0)

    def test_it_measures_the_path_not_the_close(self):
        """THE WHOLE POINT. Two bars with the same open and close: one
        grinds there, one chops there. A measure that cannot tell them
        apart is just the candle body again."""
        clean = self._walk([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        chop = self._walk([0, 4, 1, 6, 2, 8, 3, 9, 5, 10, 10])
        assert clean.live_pai == pytest.approx(1.0)
        assert 0.45 < chop.live_pai < 0.70
        assert clean.live_pai > chop.live_pai + 0.3

    def test_the_labels_sit_on_the_stated_thresholds(self):
        from liqmap.ofibar import (BALANCED_FLOW, BUY_AGGRESSIVE,
                                   SELL_AGGRESSIVE, OfiBar)
        assert OfiBar._label(0.70) == BUY_AGGRESSIVE
        assert OfiBar._label(0.6999) == BALANCED_FLOW
        assert OfiBar._label(0.30) == SELL_AGGRESSIVE
        assert OfiBar._label(0.3001) == BALANCED_FLOW
        assert OfiBar._label(0.5) == BALANCED_FLOW

    def test_a_bar_with_no_price_movement_is_balanced(self):
        from liqmap.ofibar import BALANCED_FLOW
        b = self._walk([5, 5, 5, 5, 5])
        assert b.live_pai == pytest.approx(0.5)
        bar = OfiBar()
        assert bar.pai == pytest.approx(0.5)
        assert bar.aggression == BALANCED_FLOW

    def test_aggression_resets_with_the_accumulator(self):
        """Carried forward, every bar inherits the first one's path."""
        b = OfiBarBuilder(threshold=200, max_s=10 ** 9)
        b.on_book(0.0, 20_000, 100, 20_001, 100)
        ts, px = 0.0, 20_000
        closed = 0
        while closed < 2 and ts < 400:
            ts += 0.1
            px += 1
            if b.on_book(ts, px, 400, px + 1, 100) is not None:
                closed += 1
                assert b._buy_agg == 0.0 and b._sell_agg == 0.0

    def test_the_burst_window_is_separate_from_the_bar(self):
        """A bar that climbed all the way and then ticked DOWN once at
        the close must read buy-aggressive overall and sell-aggressive
        in the burst. One number cannot do both jobs."""
        b = OfiBarBuilder(threshold=10 ** 9, max_s=10 ** 9)
        ts = 0.0
        for px in range(0, 40):
            ts += 0.1
            b.on_book(ts, px, 100, px + 1, 100)
        # Ten milliseconds, so the preceding uptick falls OUT of the
        # 5ms window. Spaced 1ms the window holds both and reads 0.5,
        # which is correct behaviour and a useless fixture.
        ts += 0.010
        b.on_book(ts, 38, 100, 39, 100)      # one downtick, alone in the window
        bar = b._close(GREEN, ts, b._bar.close)
        assert bar.pai > 0.9, "the bar as a whole climbed"
        assert bar.pai_burst == pytest.approx(0.0), "the last 5ms fell"
        assert bar.aggression != bar.aggression_burst

    def test_agrees_is_false_when_flow_and_path_conflict(self):
        from liqmap.ofibar import SELL_AGGRESSIVE
        bar = OfiBar(direction=GREEN, buy_agg=1.0, sell_agg=9.0)
        assert bar.aggression == SELL_AGGRESSIVE
        assert bar.agrees is False, "green flow, falling price: absorption"
        mirror = OfiBar(direction=RED, buy_agg=9.0, sell_agg=1.0)
        assert mirror.agrees is False

    def test_agrees_is_true_when_they_match(self):
        assert OfiBar(direction=GREEN, buy_agg=9.0, sell_agg=1.0).agrees
        assert OfiBar(direction=RED, buy_agg=1.0, sell_agg=9.0).agrees
        assert OfiBar(direction=CHOP, buy_agg=9.0, sell_agg=1.0).agrees

    def test_a_balanced_path_never_counts_as_disagreement(self):
        """Only an OPPOSITE path is absorption. A two-sided one is just
        a quiet bar, and flagging it would bury the real cases."""
        assert OfiBar(direction=GREEN, buy_agg=5.0, sell_agg=5.0).agrees
        assert OfiBar(direction=RED, buy_agg=5.0, sell_agg=5.0).agrees

    def test_the_ratio_needs_no_tick_size(self):
        """Units cancel, so raw feed integers and tick indices give the
        same answer -- which is why the live wiring can pass raw ints."""
        small = self._walk([0, 1, 2, 3])
        big = OfiBarBuilder(threshold=10 ** 9, max_s=10 ** 9)
        ts = 0.0
        for k in range(4):
            ts += 0.1
            px = 20_000_000_000 + k * 250_000_000
            big.on_book(ts, px, 100, px + 250_000_000, 100)
        assert small.live_pai == pytest.approx(big.live_pai)

    def test_it_is_in_the_payload(self):
        import json
        b = self._walk([0, 1, 2, 3, 4])
        bar = b._close(GREEN, 1.0, b._bar.close)
        d = bar.as_dict()
        for k in ("price_agg_ratio", "price_aggression",
                  "price_agg_burst", "aggression_burst", "agrees"):
            assert k in d, k
        json.dumps(d)


class TestTheHistoryIsBounded:
    """A long-running service with five builders must not grow forever."""

    def _fill(self, cap, n=600):
        b = OfiBarBuilder(threshold=1, max_s=10 ** 9, max_bars=cap)
        b.on_book(0.0, 20_000, 100, 20_001, 100)
        ts, px = 0.0, 20_000
        for i in range(n):
            ts += 0.01
            px += 1 if i % 2 else -1
            b.on_book(ts, px, 100 + (i % 7) * 50, px + 1, 100)
        return b

    def test_the_list_never_exceeds_the_cap(self):
        b = self._fill(50, n=4000)
        assert len(b.bars) <= 50
        assert b.dropped > 0

    def test_the_newest_bars_are_the_ones_kept(self):
        b = self._fill(50, n=4000)
        times = [x.ts_close for x in b.bars]
        assert times == sorted(times)
        assert b.bars[-1].ts_close == pytest.approx(max(times))

    def test_dropped_counts_what_fell_off_the_front(self):
        b = self._fill(50, n=4000)
        assert b.dropped + len(b.bars) >= 50

    def test_trimming_happens_in_blocks_not_one_at_a_time(self):
        """Popping one per close copies the whole list every bar."""
        src = (ROOT / "liqmap" / "ofibar.py").read_text()
        assert "del self.bars[:cut]" in src
        assert "self.bars.pop(0)" not in src

    def test_reset_clears_the_dropped_count(self):
        b = self._fill(50, n=2000)
        b.reset()
        assert b.dropped == 0 and b.bars == []

    def test_a_tiny_cap_is_refused(self):
        with pytest.raises(ValueError):
            OfiBarBuilder(max_bars=5)


class TestTheDeltaColumn:
    """Paid-for flow, alongside the book and the price path.

    In order of how badly each would mislead:

      delta carried across a close. It would be a running total wearing
      a per-bar label -- the same latching failure the accumulator
      reset exists to prevent, and it looks perfectly normal on screen

      a print closing a bar. The boundary is order-flow imbalance and
      nothing else; if trades could end one too, the threshold would
      mean two things at once and no bar would say which closed it

      an empty bar reported as a conflict. At the larger targets plenty
      of bars contain no prints at all, and calling those disagreements
      puts a reading on the chart that nothing supports

      a print arriving before the first book touch being dropped. It
      belongs to the bar that touch opens
    """

    def _b(self, threshold=50):
        return OfiBarBuilder(tick_size=0.25, threshold=threshold,
                             max_s=900.0, min_updates=2)

    def _walk(self, b, n=12, size=20, side="buy", bid=20_000):
        b.on_book(0.0, bid, 100, bid + 1, 100)
        for i in range(1, n):
            b.on_trade(i * 0.01, size, side)
            b.on_book(i * 0.01, bid, 100 + i * 20, bid + 1, 100)
        return b

    def test_delta_is_buy_minus_sell(self):
        b = self._b()
        b.on_book(0.0, 20_000, 100, 20_001, 100)
        b.on_trade(0.1, 7, "buy")
        b.on_trade(0.2, 3, "sell")
        assert b.live_delta == 4
        assert b.live_trades == 2

    def test_delta_does_not_carry_across_a_close(self):
        """The latching failure, in a second place."""
        b = self._walk(self._b(), n=20)
        assert len(b.bars) >= 2
        per = [x.buy_vol + x.sell_vol for x in b.bars]
        assert max(per) < sum(per), "a bar is holding the running total"
        for x in b.bars:
            assert x.trade_count <= 20

    def test_each_bar_counts_only_its_own_prints(self):
        b = self._b()
        b.on_book(0.0, 20_000, 100, 20_001, 100)
        b.on_trade(0.01, 11, "buy")
        # Push it over the line so the bar closes.
        for i in range(1, 8):
            b.on_book(i * 0.01, 20_000, 100 + i * 40, 20_001, 100)
        first = b.bars[0]
        assert first.trade_count == 1 and first.buy_vol == 11
        b.on_trade(0.2, 4, "sell")
        assert b.live_delta == -4, "the new bar starts from nothing"

    def test_a_trade_never_closes_a_bar(self):
        b = self._b()
        b.on_book(0.0, 20_000, 100, 20_001, 100)
        for i in range(200):
            b.on_trade(i * 0.001, 10_000, "buy")
        assert b.bars == [], "only order-flow imbalance ends a bar"

    def test_a_print_before_the_first_touch_is_kept(self):
        b = self._b()
        b.on_trade(0.0, 9, "buy")
        b.on_book(0.01, 20_000, 100, 20_001, 100)
        assert b.live_delta == 9

    def test_an_unknown_side_is_refused_not_counted_as_balanced(self):
        b = self._b()
        b.on_book(0.0, 20_000, 100, 20_001, 100)
        b.on_trade(0.1, 50, "none")
        b.on_trade(0.1, 50, "")
        assert b.live_trades == 0 and b.live_delta == 0

    def test_a_zero_or_negative_size_is_refused(self):
        b = self._b()
        b.on_book(0.0, 20_000, 100, 20_001, 100)
        b.on_trade(0.1, 0, "buy")
        b.on_trade(0.1, -5, "sell")
        assert b.live_trades == 0

    def test_a_bar_with_no_prints_agrees_by_default(self):
        """Nothing there to disagree with."""
        b = self._walk(self._b(), n=20, size=0)
        assert b.bars
        for x in b.bars:
            assert x.trade_count == 0
            assert x.delta_agrees, "an empty bar is not a conflict"
            assert x.delta == 0 and x.delta_ratio == 0.5

    def test_a_green_bar_sold_into_does_not_agree(self):
        """The case the column exists for: the bid filling up while
        prints hit it."""
        b = self._walk(self._b(), n=20, side="sell")
        greens = [x for x in b.bars if x.direction == GREEN]
        assert greens, "the feed should produce green bars"
        for x in greens:
            assert x.delta < 0 and not x.delta_agrees

    def test_a_green_bar_bought_into_agrees(self):
        b = self._walk(self._b(), n=20, side="buy")
        greens = [x for x in b.bars if x.direction == GREEN]
        assert greens
        for x in greens:
            assert x.delta > 0 and x.delta_agrees

    def test_delta_and_the_price_path_are_separate_readings(self):
        """`agrees` asks whether PRICE followed the book. `delta_agrees`
        asks whether anyone PAID. They answer different questions and
        must not be wired to the same source."""
        src = (ROOT / "liqmap" / "ofibar.py").read_text()
        i = src.index("def delta_agrees")
        body = src[i:src.index("def _label")]
        assert "self.buy_agg" not in body
        assert "self.pai" not in body and "aggression" not in body

    def test_the_ratio_is_a_share_of_traded_size(self):
        b = self._b()
        b.on_book(0.0, 20_000, 100, 20_001, 100)
        b.on_trade(0.1, 30, "buy")
        b.on_trade(0.2, 10, "sell")
        for i in range(1, 8):
            b.on_book(0.2 + i * 0.01, 20_000, 100 + i * 40, 20_001, 100)
        x = b.bars[0]
        assert x.volume == 40
        assert x.delta_ratio == pytest.approx(0.75)

    def test_reset_clears_the_delta_too(self):
        b = self._walk(self._b(), n=20)
        b.reset()
        assert b.live_delta == 0 and b.live_trades == 0

    def test_it_serialises(self):
        import json
        b = self._walk(self._b(), n=20)
        for x in b.bars:
            d = json.loads(json.dumps(x.as_dict()))
            for k in ("delta", "buy_vol", "sell_vol", "trade_count",
                      "delta_ratio", "delta_agrees"):
                assert k in d, k

    def test_the_aggressor_is_never_inferred_here(self):
        """CME states the side. A tick rule is wrong on a meaningful
        share of trades, worst when the market is fast."""
        src = (ROOT / "liqmap" / "ofibar.py").read_text()
        i = src.index("def on_trade")
        body = src[i:i + 1600]
        for w in ("last_px", "> last", "uptick", "tick rule"):
            assert w not in body, w
