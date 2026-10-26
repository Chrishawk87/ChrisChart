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
        assert len(row) == len(ROW_COLUMNS) == 7
        ts, o, h, l, c, colour, strength = row
        assert ts == pytest.approx(0.5), "stamped at the CLOSE"
        assert colour == GREEN
        assert strength == out.ofi
        assert l <= o <= h and l <= c <= h

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
