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
                           Lead, LeadWatcher, Pace, PaceWatcher,
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

    def test_the_panel_says_what_the_buy_sell_is_not(self):
        """The call puts BUY or SELL on screen in a box. Nothing
        else on the page does that, and a trader reading it at a
        glance will take it for a signal unless the page says
        plainly that it is not one.

        So the disclaimer is not decoration -- it is the only thing
        standing between a measured null and a trade. If a future
        edit tidies it away, this fails.
        """
        src = (ROOT / "liqmap" / "web.py").read_text()
        assert "fpPaceRow" in src, "the panel itself must still exist"
        lower = src.lower()
        assert "it is not a forecast" in lower
        assert "does not precede price" in lower
        assert "not a trigger" in lower

    def test_the_panel_has_no_order_code_either(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        for w in ("def submit(", "def execute(", "def place(",
                  "def order("):
            assert w not in src, w


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


class TestTheLeadAndWhenItStops:
    """The running total can say 'buyers +250' long after buyers
    stopped winning at +400. The high-water mark is what notices."""

    def _push(self, w, n, side, size=10, px0=5800.0, step=0.025, ts0=0.0):
        ts, px = ts0, px0
        for _ in range(n):
            ts += 1.0
            px += step if side == "buy" else -step
            w.on_trade(ts, px, size, side)
        return ts, px

    def test_the_peak_is_remembered_after_it_is_given_back(self):
        w = LeadWatcher(stall_frac=0.35, stall_min=50)
        w.on_bar("a")
        self._push(w, 40, "buy")
        assert w.cur.peak == 400
        self._push(w, 15, "sell", ts0=40)
        assert w.cur.delta == 250, "the running total"
        assert w.cur.peak == 400, "the high-water mark was forgotten"
        assert w.cur.give_back == 150

    def test_the_stall_fires_when_enough_is_given_back(self):
        w = LeadWatcher(stall_frac=0.35, stall_min=50)
        w.on_bar("a")
        self._push(w, 40, "buy")
        assert not w.cur.stalled
        self._push(w, 15, "sell", ts0=40)
        assert w.cur.stalled and w.cur.stall_side == BUYERS

    def test_the_total_can_still_favour_the_side_that_stalled(self):
        """The whole point: the ladder still reads buyers."""
        w = LeadWatcher(stall_frac=0.35, stall_min=50)
        w.on_bar("a")
        self._push(w, 40, "buy")
        self._push(w, 15, "sell", ts0=40)
        assert w.cur.stalled
        assert w.cur.delta > 0 and w.cur.leader == BUYERS

    def test_sellers_stall_the_same_way(self):
        w = LeadWatcher(stall_frac=0.35, stall_min=50)
        w.on_bar("a")
        self._push(w, 40, "sell")
        assert w.cur.trough == -400
        self._push(w, 15, "buy", ts0=40)
        assert w.cur.stalled and w.cur.stall_side == SELLERS

    def test_a_small_lead_cannot_stall(self):
        """A lead of twelve giving back five is noise, not a turn."""
        w = LeadWatcher(stall_frac=0.35, stall_min=50)
        w.on_bar("a")
        self._push(w, 3, "buy", size=4)       # lead of 12
        self._push(w, 2, "sell", size=4, ts0=3)
        assert not w.cur.stalled

    def test_a_bigger_lead_must_give_back_more(self):
        """The threshold is a fraction of the leader's own best, so a
        quiet bar and a busy one are held to the same standard."""
        # Identical give-back of 40 contracts against two very
        # different leads: 40 is 40% of a 100 lead and 4% of a 1000
        # one, so only the small lead has actually turned.
        small = LeadWatcher(stall_frac=0.35, stall_min=50)
        small.on_bar("a")
        self._push(small, 10, "buy")          # peak 100
        self._push(small, 4, "sell", ts0=10)  # gave back 40
        big = LeadWatcher(stall_frac=0.35, stall_min=50)
        big.on_bar("a")
        self._push(big, 100, "buy")           # peak 1000
        self._push(big, 4, "sell", ts0=100)   # gave back 40
        assert small.cur.give_back == big.cur.give_back == 40
        assert small.cur.stalled, "40 of a 100 lead is a turn"
        assert not big.cur.stalled, "40 of a 1000 lead is noise"

    def test_one_stall_per_bar(self):
        w = LeadWatcher(stall_frac=0.35, stall_min=50)
        w.on_bar("a")
        self._push(w, 40, "buy")
        self._push(w, 15, "sell", ts0=40)
        first = w.cur.stall_ts
        self._push(w, 20, "sell", ts0=60)
        assert w.cur.stall_ts == first, "the stall was re-armed mid-bar"

    def test_a_new_bar_starts_clean(self):
        w = LeadWatcher(stall_frac=0.35, stall_min=50)
        w.on_bar("a")
        self._push(w, 40, "buy")
        self._push(w, 15, "sell", ts0=40)
        assert w.on_bar("b") is True
        assert w.cur.peak == 0 and not w.cur.stalled
        assert len(w.history) == 1

    def test_the_same_bar_key_does_not_roll(self):
        w = LeadWatcher()
        w.on_bar("a")
        assert w.on_bar("a") is False

    def test_the_stall_records_where_it_happened(self):
        w = LeadWatcher(stall_frac=0.35, stall_min=50)
        w.on_bar("a")
        _, top = self._push(w, 40, "buy")
        self._push(w, 15, "sell", ts0=40)
        assert w.cur.peak_px == pytest.approx(top)
        assert w.cur.stall_px < top, "the stall price is not below the peak"

    def test_a_degenerate_fraction_is_refused(self):
        for bad in (0.0, 1.0, -0.2, 1.5):
            with pytest.raises(ValueError):
                LeadWatcher(stall_frac=bad)

    def test_it_serialises(self):
        import json
        w = LeadWatcher()
        w.on_bar("a")
        self._push(w, 40, "buy")
        json.dumps(w.cur.as_dict())

    def test_the_history_is_bounded(self):
        w = LeadWatcher(history=5)
        for k in range(50):
            w.on_bar(k)
            self._push(w, 5, "buy")
        assert len(w.history) <= 5


class TestSpeedNotLevel:
    """buyers 4932 / sellers 4992 is a sellers' bar by the total and a
    buyers' bar if buyers are adding forty a second against fifteen.
    The rate is a different question from the level, and the useful
    state is when the two disagree."""

    def _feed(self, w, n, ts, buy=0, sell=0, every=0.1):
        for i in range(n):
            ts += every
            if buy:
                w.on_trade(ts, buy, "buy")
            if sell:
                w.on_trade(ts, sell, "sell")
        return ts

    def test_the_side_behind_on_totals_can_be_the_one_gaining(self):
        w = PaceWatcher(tau_s=10.0, dead=0.20)
        ts = self._feed(w, 600, 0.0, buy=4, sell=8)
        assert w.reading(ts).level == SELLERS
        ts = self._feed(w, 60, ts, buy=20, sell=1)
        r = w.reading(ts)
        assert r.level == SELLERS, "the total should still favour sellers"
        assert r.pace == BUYERS
        assert r.taking_control == BUYERS

    def test_and_the_mirror(self):
        w = PaceWatcher(tau_s=10.0, dead=0.20)
        ts = self._feed(w, 600, 0.0, buy=8, sell=4)
        assert w.reading(ts).level == BUYERS
        ts = self._feed(w, 60, ts, buy=1, sell=20)
        r = w.reading(ts)
        assert r.level == BUYERS and r.pace == SELLERS
        assert r.taking_control == SELLERS

    def test_agreement_is_not_called_a_takeover(self):
        w = PaceWatcher(tau_s=10.0, dead=0.20)
        ts = self._feed(w, 400, 0.0, buy=10, sell=2)
        r = w.reading(ts)
        assert r.level == BUYERS and r.pace == BUYERS
        assert r.taking_control == ""
        assert r.diverging is False

    def test_the_deadband_stops_it_flickering_at_the_crossover(self):
        """Without it the two rates cross constantly and the reading
        changes several times a second."""
        w = PaceWatcher(tau_s=10.0, dead=0.20)
        ts = self._feed(w, 300, 0.0, buy=10, sell=10)
        assert w.reading(ts).pace == BALANCED

    def test_a_rate_is_per_second_not_per_trade(self):
        """Twenty prints is a moment at the open and minutes
        overnight. The same flow delivered either way must read the
        same."""
        burst = PaceWatcher(tau_s=10.0)
        trickle = PaceWatcher(tau_s=10.0)
        ts = 0.0
        for i in range(400):          # 40 contracts/sec, 1 print per 0.1s
            ts += 0.1
            burst.on_trade(ts, 4, "buy")
        ts2 = 0.0
        for i in range(40):           # 40 contracts/sec, 1 print per second
            ts2 += 1.0
            trickle.on_trade(ts2, 40, "buy")
        a = burst.reading(ts).buy_rate
        b = trickle.reading(ts2).buy_rate
        assert a == pytest.approx(b, rel=0.15), (
            f"same flow, different print counts, different rate: {a} vs {b}")

    def test_a_steady_flow_converges_on_its_own_rate(self):
        w = PaceWatcher(tau_s=5.0)
        ts = 0.0
        for i in range(2000):         # exactly 30 contracts a second
            ts += 0.1
            w.on_trade(ts, 3, "buy")
        assert w.reading(ts).buy_rate == pytest.approx(30.0, rel=0.05)

    def test_silence_decays_the_rate(self):
        """A side that stopped a minute ago must not still read at the
        rate it had when it stopped."""
        w = PaceWatcher(tau_s=10.0)
        ts = self._feed(w, 300, 0.0, buy=10)
        hot = w.reading(ts).buy_rate
        cold = w.reading(ts + 90).buy_rate
        assert hot > 10.0 and cold < hot * 0.01

    def test_the_totals_are_untouched_by_silence(self):
        w = PaceWatcher(tau_s=10.0)
        ts = self._feed(w, 300, 0.0, buy=10)
        before = w.reading(ts).buy_total
        assert w.reading(ts + 90).buy_total == before

    def test_reset_clears_it(self):
        w = PaceWatcher(tau_s=10.0)
        ts = self._feed(w, 200, 0.0, buy=10, sell=3)
        w.reset()
        r = w.reading(ts + 1)
        assert r.buy_total == 0 and r.buy_rate == 0.0
        assert r.level == BALANCED

    def test_degenerate_settings_are_refused(self):
        with pytest.raises(ValueError):
            PaceWatcher(tau_s=0.0)
        with pytest.raises(ValueError):
            PaceWatcher(dead=-0.1)

    def test_it_serialises(self):
        import json
        w = PaceWatcher()
        ts = self._feed(w, 100, 0.0, buy=5, sell=9)
        json.dumps(w.reading(ts).as_dict())


class TestThePaceCallNamesASideWithoutPredicting:
    """A buy/sell in the language of a trade, over a measured null.

    The call exists so the panel can say which side flow is on in
    the words a trader uses. cvd_study section 10 established that
    these states do not precede price at 15s, 30s or 60s over 102.5
    hours of ES, so the one thing these tests must never let slip is
    the call quietly becoming a forecast -- either in the code, by
    disagreeing with the state it claims to describe, or on the
    page, by losing the sentence that says what it is not.
    """

    def _feed(self, w, n, ts, buy, sell, step=0.25):
        for _ in range(n):
            ts += step
            if buy:
                w.on_trade(ts, buy, "buy")
            if sell:
                w.on_trade(ts, sell, "sell")
        return ts

    def test_buyers_taking_control_reads_buy(self):
        w = PaceWatcher(tau_s=10.0, dead=0.20)
        ts = self._feed(w, 600, 0.0, buy=4, sell=8)
        ts = self._feed(w, 60, ts, buy=20, sell=1)
        r = w.reading(ts)
        assert r.taking_control == BUYERS
        assert r.call == "buy"
        assert r.basis == "taking", "behind on totals, faster now"

    def test_and_the_mirror_reads_sell(self):
        w = PaceWatcher(tau_s=10.0, dead=0.20)
        ts = self._feed(w, 600, 0.0, buy=8, sell=4)
        ts = self._feed(w, 60, ts, buy=1, sell=20)
        r = w.reading(ts)
        assert r.taking_control == SELLERS
        assert r.call == "sell"
        assert r.basis == "taking"

    def test_a_side_already_ahead_and_still_faster_is_extending(self):
        w = PaceWatcher(tau_s=10.0, dead=0.20)
        ts = self._feed(w, 400, 0.0, buy=10, sell=2)
        r = w.reading(ts)
        assert r.taking_control == ""
        assert r.call == "buy"
        assert r.basis == "extending", "ahead and pulling away is not a takeover"

    def test_extending_mirrors_too(self):
        w = PaceWatcher(tau_s=10.0, dead=0.20)
        ts = self._feed(w, 400, 0.0, buy=2, sell=10)
        r = w.reading(ts)
        assert r.call == "sell" and r.basis == "extending"

    def test_balanced_pace_makes_no_call_at_all(self):
        """The deadband exists so the reading does not flicker. A
        call that flickered with it would be worse than none."""
        w = PaceWatcher(tau_s=10.0, dead=0.20)
        ts = self._feed(w, 300, 0.0, buy=10, sell=10)
        r = w.reading(ts)
        assert r.pace == BALANCED
        assert r.call == "" and r.basis == ""

    def test_lopsided_totals_do_not_force_a_call_when_pace_is_even(self):
        """Who is AHEAD is not the question the call answers."""
        w = PaceWatcher(tau_s=10.0, dead=0.20)
        ts = self._feed(w, 600, 0.0, buy=12, sell=2)
        ts = self._feed(w, 200, ts, buy=6, sell=6)
        r = w.reading(ts)
        assert r.level == BUYERS, "buyers are far ahead on the total"
        assert r.pace == BALANCED
        assert r.call == "", "even pace, so no side to name"

    def test_the_call_can_never_contradict_taking_control(self):
        """Both follow `pace`. If one is ever derived separately
        the panel would show BUY beside SELLERS TAKING CONTROL."""
        w = PaceWatcher(tau_s=8.0, dead=0.15)
        ts = 0.0
        seen = set()
        for i in range(300):
            ts += 0.2
            w.on_trade(ts, 3 + (i % 11), "buy" if (i // 17) % 2 else "sell")
            r = w.reading(ts)
            seen.add(r.call)
            if r.taking_control:
                want = "buy" if r.taking_control == BUYERS else "sell"
                assert r.call == want
            if r.call:
                assert r.pace == (BUYERS if r.call == "buy" else SELLERS)
        assert {"buy", "sell"} <= seen, "the fixture should exercise both"

    def test_it_is_carried_in_the_payload(self):
        import json
        w = PaceWatcher()
        ts = self._feed(w, 200, 0.0, buy=12, sell=2)
        d = w.reading(ts).as_dict()
        assert d["call"] == "buy" and d["basis"] == "extending"
        json.dumps(d)
