"""One verdict on the footprint, and the ways it could flatter itself.

What this guards, in order of how badly each would mislead:

  a directional bias in the scoring, so the same bar mirrored does
  not give the mirrored answer -- the one failure that would make
  every reading wrong in the same direction

  calling a bar "strong buyers" because the delta is large, when
  price did not move at all: that is absorption, and reading the
  effort while ignoring the result is the classic footprint mistake

  inventing a position reading on a bar with imbalances on one side
  only, where there is no contrast to measure

  a run of six counting the same as two runs of three
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from liqmap.footprint import FootprintCandle, TickScale     # noqa: E402
from liqmap.fpread import (BALANCED, BUYERS, SELLERS, WEIGHTS,  # noqa: E402
                           FootprintReader, Verdict, read_candle)

SC = TickScale(0.25)


def bar(levels: dict, open_t: int, close_t: int) -> FootprintCandle:
    """levels: {price_tick: (sell_size, buy_size)}."""
    c = FootprintCandle(ts_open=0.0)
    for p, (b, a) in sorted(levels.items()):
        if b:
            c.add(p, b, "sell", 0.0)
        if a:
            c.add(p, a, "buy", 0.0)
    c.open_ticks, c.close_ticks = open_t, close_t
    return c


def mirror(levels: dict) -> dict:
    """Same bar with the two sides swapped at every level."""
    return {p: (a, b) for p, (b, a) in levels.items()}


UP = {100: (10, 10), 101: (5, 40), 102: (5, 45),
      103: (4, 50), 104: (3, 60)}


class TestItHasNoSideOfItsOwn:
    """The failure that would make every reading wrong the same way."""

    def test_a_mirrored_bar_gives_the_mirrored_score(self):
        a = read_candle(bar(UP, 100, 104), SC)
        b = read_candle(bar(mirror(UP), 104, 100), SC)
        assert a.side == BUYERS and b.side == SELLERS
        assert a.score == pytest.approx(-b.score, abs=1e-9)

    def test_and_the_same_strength(self):
        a = read_candle(bar(UP, 100, 104), SC)
        b = read_candle(bar(mirror(UP), 104, 100), SC)
        assert a.strength == b.strength

    def test_every_part_mirrors_too(self):
        a = read_candle(bar(UP, 100, 104), SC)
        b = read_candle(bar(mirror(UP), 104, 100), SC)
        for k in a.parts:
            assert a.parts[k] == pytest.approx(-b.parts[k], abs=1e-9), k

    def test_a_balanced_bar_calls_no_winner(self):
        v = read_candle(bar({100: (20, 20), 101: (20, 20),
                             102: (20, 20)}, 101, 101), SC)
        assert v.side == BALANCED and v.strength == "none"
        assert v.score == pytest.approx(0.0)


class TestEffortWithoutResult:
    """Heavy buying that moves nothing is absorption. Reading the
    delta and ignoring the travel is how a footprint flatters the
    side that is losing."""

    def test_flow_with_no_travel_is_called_out(self):
        v = read_candle(bar({100: (10, 60), 101: (10, 55),
                             102: (10, 50)}, 101, 101), SC)
        assert "absorption" in v.note
        assert v.parts["delta"] > 0.15
        assert abs(v.parts["follow"]) < 0.15

    def test_it_is_not_allowed_to_read_as_strong(self):
        v = read_candle(bar({100: (10, 60), 101: (10, 55),
                             102: (10, 50)}, 101, 101), SC)
        assert v.strength != "strong", "effort alone earned a strong call"

    def test_flow_against_travel_is_also_absorption(self):
        v = read_candle(bar({100: (10, 60), 101: (10, 55),
                             102: (10, 50)}, 102, 100), SC)
        assert "absorption" in v.note
        assert v.parts["delta"] > 0 and v.parts["follow"] < 0

    def test_flow_that_does_travel_is_left_alone(self):
        v = read_candle(bar(UP, 100, 104), SC)
        assert "absorption" not in v.note
        assert v.strength == "strong"


class TestPositionIsAContrastOrNothing:
    """With flags on one side only there is no contrast to measure.
    An early version squeezed it into a signed number anyway and
    scored a seller-dominated bar's geometry at +0.25 -- bullish."""

    def test_one_sided_flags_score_no_position(self):
        v = read_candle(bar(UP, 100, 104), SC)
        assert v.buy_flags > 0 and v.sell_flags == 0
        assert v.parts["location"] == 0.0
        assert "one side only" in v.note

    def test_a_seller_bar_is_never_given_a_bullish_position(self):
        v = read_candle(bar(mirror(UP), 104, 100), SC)
        assert v.parts["location"] <= 0.0

    def test_buyers_high_and_sellers_low_reads_bullish(self):
        """Both sides present: buyers working the top of the range
        against sellers working the bottom."""
        lv = {100: (50, 4), 101: (45, 5),        # sellers low
              102: (20, 20),
              103: (5, 45), 104: (4, 50)}        # buyers high
        v = read_candle(bar(lv, 100, 104), SC)
        if v.buy_flags and v.sell_flags:
            assert v.parts["location"] > 0.0

    def test_and_the_mirror_reads_bearish(self):
        lv = {100: (50, 4), 101: (45, 5), 102: (20, 20),
              103: (5, 45), 104: (4, 50)}
        v = read_candle(bar(mirror(lv), 104, 100), SC)
        if v.buy_flags and v.sell_flags:
            assert v.parts["location"] < 0.0


class TestStacksCountLength:
    """A run of six is not two runs of three -- that is the whole
    difference between 'overall' and 'every level'."""

    def test_a_longer_run_counts_for_more(self):
        short = {100: (10, 10), 101: (4, 40), 102: (4, 45),
                 103: (4, 50), 104: (10, 10)}
        long_ = {100: (10, 10), 101: (4, 40), 102: (4, 45),
                 103: (4, 50), 104: (4, 55), 105: (4, 60),
                 106: (4, 65), 107: (10, 10)}
        a = read_candle(bar(short, 100, 104), SC)
        b = read_candle(bar(long_, 100, 107), SC)
        if a.buy_stacks and b.buy_stacks:
            assert b.parts["stacks"] >= a.parts["stacks"]

    def test_scattered_singles_make_no_stack(self):
        lv = {100: (4, 40), 101: (20, 20), 102: (4, 45),
              103: (20, 20), 104: (4, 50)}
        v = read_candle(bar(lv, 100, 104), SC)
        assert v.buy_stacks == 0, "non-consecutive flags formed a run"


class TestTheEdges:

    def test_an_empty_bar_says_so(self):
        c = FootprintCandle(ts_open=0.0)
        v = read_candle(c, SC)
        assert v.side == BALANCED and "no prints" in v.note

    def test_no_candle_at_all_is_survivable(self):
        assert read_candle(None, SC).side == BALANCED

    def test_a_one_tick_bar_claims_no_geometry(self):
        v = read_candle(bar({100: (10, 40)}, 100, 100), SC)
        assert v.span_ticks == 0
        assert v.parts["location"] == 0.0
        assert v.parts["follow"] == 0.0

    def test_the_score_stays_inside_minus_one_and_one(self):
        huge = {p: (1, 5000) for p in range(100, 140)}
        v = read_candle(bar(huge, 100, 139), SC)
        assert -1.0 <= v.score <= 1.0

    def test_it_serialises(self):
        import json
        json.dumps(read_candle(bar(UP, 100, 104), SC).as_dict())


class TestTheWeightsAreAnArgument:

    def test_they_are_published_with_the_verdict(self):
        v = read_candle(bar(UP, 100, 104), SC)
        assert set(v.weights) == set(WEIGHTS)
        assert v.parts.keys() == WEIGHTS.keys()

    def test_they_sum_to_one_so_the_score_reads_as_a_fraction(self):
        assert sum(WEIGHTS.values()) == pytest.approx(1.0)

    def test_a_caller_can_override_them(self):
        only_delta = {"delta": 1.0, "flags": 0.0, "stacks": 0.0,
                      "location": 0.0, "follow": 0.0}
        c = bar(UP, 100, 104)
        v = read_candle(c, SC, weights=only_delta)
        assert v.score == pytest.approx(v.parts["delta"])

    def test_every_part_is_shown_not_just_the_headline(self):
        """If the headline disagrees with the part you trust, you have
        to be able to see the part."""
        v = read_candle(bar(UP, 100, 104), SC)
        for k in ("delta", "flags", "stacks", "location", "follow"):
            assert k in v.parts


class TestTheReader:

    class _Stream:
        def __init__(self, live, history):
            self.live = live
            self.history = history

    def test_it_reads_the_live_bar(self):
        s = self._Stream(bar(UP, 100, 104), [])
        assert FootprintReader().live(s).side == BUYERS

    def test_it_reads_the_closed_ones(self):
        s = self._Stream(bar(UP, 100, 104),
                         [bar(UP, 100, 104), bar(mirror(UP), 104, 100)])
        out = FootprintReader().closed(s)
        assert [v.side for v in out] == [BUYERS, SELLERS]

    def test_it_counts_how_long_one_side_has_held(self):
        s = self._Stream(bar(UP, 100, 104),
                         [bar(mirror(UP), 104, 100)] + [bar(UP, 100, 104)] * 3)
        assert FootprintReader().as_dict(s)["bars_held"] == 3

    def test_a_flip_resets_the_count(self):
        s = self._Stream(bar(UP, 100, 104),
                         [bar(UP, 100, 104)] * 3 + [bar(mirror(UP), 104, 100)])
        assert FootprintReader().as_dict(s)["bars_held"] == 0

    def test_it_carries_no_state_between_reads(self):
        """A reader that accumulated would drift from the footprint it
        is supposed to be reading."""
        r = FootprintReader()
        s = self._Stream(bar(UP, 100, 104), [])
        first = r.live(s).score
        for _ in range(5):
            assert r.live(s).score == pytest.approx(first)

    def test_no_live_bar_is_survivable(self):
        assert FootprintReader().live(self._Stream(None, [])).side == BALANCED


class TestItDecidesNothing:

    def test_no_order_code(self):
        src = (ROOT / "liqmap" / "fpread.py").read_text()
        for w in ("def submit(", "def execute(", "def place(",
                  "def order("):
            assert w not in src, w

    def test_it_does_not_claim_to_predict(self):
        """It reports the state of the auction. Five sections of
        measurement found nothing in this family that calls where
        price goes next, and the module must not imply otherwise."""
        src = (ROOT / "liqmap" / "fpread.py").read_text().lower()
        assert "does not predict" in src
        for word in ("institution", "retail", "smart money", "whale"):
            assert word not in src, word


class TestItFindsTheFormingBarWhereverItLives:
    """Guessing one attribute name and getting a silent 'no live bar'
    is how a reader ends up describing nothing while looking fine.
    That is what happened the first time this was wired: the real
    FootprintStream keeps the open bar on its engine, not on itself."""

    def test_it_reads_a_real_footprint_stream(self):
        import random

        from liqmap.footprint import Tick, VolumeRule
        from liqmap.footprint_live import FootprintStream

        rng = random.Random(5)
        st = FootprintStream(tick_size=0.25, rule=VolumeRule(400))
        px, ts = 5800.0, 1_700_078_400.0
        for i in range(4000):
            ts += 0.05
            up = (i // 500) % 2 == 0
            if rng.random() < 0.25:
                px += 0.25 if up else -0.25
            side = ("buy" if rng.random() < 0.5 + (0.22 if up else -0.22)
                    else "sell")
            st.on_tick(Tick(ts=ts, price=px, size=rng.choice((1, 2, 3, 8)),
                            aggressor=side))

        d = FootprintReader().as_dict(st)
        assert d["now"]["note"] != "no live bar"
        assert d["now"]["volume"] > 0, "the forming bar was never found"
        assert len(d["history"]) > 10

    def test_an_engine_works_too(self):
        from liqmap.footprint import FootprintEngine, Tick, VolumeRule
        e = FootprintEngine(tick_size=0.25, rule=VolumeRule(400))
        for i in range(50):
            e.on_tick(Tick(ts=1.0 + i, price=5800.0, size=2,
                           aggressor="buy"))
        assert FootprintReader().live(e).volume > 0

    def test_a_stream_with_no_bar_open_still_answers(self):
        from liqmap.footprint import FootprintEngine, VolumeRule
        e = FootprintEngine(tick_size=0.25, rule=VolumeRule(400))
        assert FootprintReader().live(e).side == BALANCED
