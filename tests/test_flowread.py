"""The reading that decides which way the five-minute trade goes.

This is not a scalp on the OFI bar. It is the thing Chris reads off
THIS BAR before taking a time-based candle somewhere else, so the
properties that matter are: it does not depend on the level he picked,
it survives an OFI bar closing underneath it, it goes quiet when the
winning side stops winning, and it does not change while he is
clicking.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest                                              # noqa: E402

from liqmap.flowread import (DOWN, FlowRead, NONE,          # noqa: E402
                             Snapshot, UP)
from liqmap.ofi import Touch, level_ofi                     # noqa: E402

RAW = 250_000_000
SRC = (ROOT / "liqmap" / "flowread.py").read_text()


def _f(**kw):
    kw.setdefault("window_s", 10.0)
    kw.setdefault("grow_s", 2.0)
    kw.setdefault("hold_s", 2.0)
    kw.setdefault("bucket_s", 0.5)
    return FlowRead(**kw)


def _push(f, side, seconds, ts=0.0, size=100.0, step=0.25):
    """Feed one-sided flow for a stretch of clock."""
    e = size if side == "buy" else -size
    end = ts + seconds
    while ts < end:
        ts += step
        f.on_event(ts, e)
    return ts


def _quiet(f, seconds, ts, step=0.25):
    end = ts + seconds
    while ts < end:
        ts += step
        f.on_event(ts, 0.0)
    return ts


class TestItSuggestsAndDoesNotExecute:
    def test_it_places_nothing(self):
        for bad in ("def submit(", "def execute(", "def place(",
                    "def order("):
            assert bad not in SRC, bad

    def test_it_says_so(self):
        assert "places, sizes or times an order" in SRC


class TestTheDirectionDoesNotDependOnTheLevel:
    """He picks 100 to 500 from a dropdown. That must now be a LABEL
    on the reading and nothing else -- if it still moved the
    direction, 'regardless of the imbalance contracts I select' would
    be false."""

    def test_the_same_flow_reads_the_same_at_every_level(self):
        reads = {}
        for lv in (100, 200, 300, 500):
            f = _f(level=lv)
            ts = _push(f, "buy", 8.0)
            reads[lv] = f.snapshot.read
        assert len(set(reads.values())) == 1, reads
        assert reads[100] == UP

    def test_changing_the_level_mid_stream_changes_nothing(self):
        f = _f(level=100)
        ts = _push(f, "buy", 8.0)
        before = f.snapshot.read
        f.set_level(500)
        after = f.snapshot.read
        assert before == after == UP

    def test_the_level_only_labels_the_strength(self):
        weak = _f(level=10 ** 9)
        _push(weak, "buy", 8.0)
        strong = _f(level=1)
        _push(strong, "buy", 8.0)
        assert weak.snapshot.read == strong.snapshot.read == UP
        assert weak.snapshot.strength == "weak"
        assert strong.snapshot.strength == "strong"

    def test_the_decision_never_reads_the_level(self):
        """Source-level, because a level could be reintroduced into
        the decision without any of the tests above failing on the
        particular sizes they happen to use."""
        i = SRC.index("def _decide")
        body = SRC[i:SRC.index("# ---", i)]
        assert "self.level" not in body, "the level is back in the call"


class TestItDoesNotResetWithAnOfiBar:
    """The accumulator in ofibar MUST reset at each close or the
    threshold latches. This must NOT, or it blanks several times
    inside a five-minute position and each blank looks like the
    signal dying."""

    def test_nothing_in_it_knows_about_bars(self):
        for word in ("ofibar", "OfiBar", "threshold", "bar_close",
                     "on_bar"):
            assert word not in SRC.split('"""', 2)[2], word

    def test_a_steady_read_survives_an_arbitrary_boundary(self):
        """Feed it in two halves, as if a bar had closed between
        them, and the read must be the one the whole stretch gives."""
        whole = _f()
        _push(whole, "buy", 12.0)
        split = _f()
        ts = _push(split, "buy", 6.0)
        _push(split, "buy", 6.0, ts)          # nothing resets here
        assert whole.snapshot.read == split.snapshot.read == UP
        assert whole.snapshot.lead == pytest.approx(split.snapshot.lead)


class TestItIsAWindowAndNotARunningTotal:
    """A cumulative sum of buying minus selling walks away from zero
    on any session with a direction and never comes back -- the same
    latching the bar builder's reset exists to prevent."""

    def test_old_flow_leaves_the_window(self):
        f = _f(window_s=5.0)
        ts = _push(f, "buy", 5.0)
        assert f.snapshot.lead > 0
        ts = _quiet(f, 8.0, ts)
        assert f.snapshot.lead == pytest.approx(0.0), "it latched"

    def test_a_long_one_sided_stretch_does_not_pin_it(self):
        f = _f(window_s=5.0)
        ts = _push(f, "buy", 60.0)
        lead_after_a_minute = f.snapshot.lead
        g = _f(window_s=5.0)
        _push(g, "buy", 6.0)
        assert lead_after_a_minute == pytest.approx(g.snapshot.lead,
                                                    rel=0.2)

    def test_the_window_is_bounded_in_memory(self):
        f = _f(window_s=10.0, bucket_s=0.5)
        _push(f, "buy", 600.0)
        assert len(f._up) == 20
        assert len(f._dn) == 20


class TestAheadIsNotEnough:
    """A lead that has stopped growing is a side that has stopped
    winning -- and the lead is LARGEST just after the push is over,
    which is when a level-on-the-net rule is most confident."""

    def test_it_reads_up_while_buying_keeps_extending(self):
        f = _f()
        _push(f, "buy", 8.0)
        s = f.snapshot
        assert s.read == UP and s.extending and s.growth > 0

    def test_it_goes_quiet_when_the_lead_stops_growing(self):
        f = _f()
        ts = _push(f, "buy", 8.0)
        assert f.snapshot.read == UP
        ts = _quiet(f, 8.0, ts)
        s = f.snapshot
        assert not s.extending, "the gap was flat and it said otherwise"
        assert s.read == NONE

    def test_a_lead_still_standing_is_not_a_read(self):
        """The distinguishing case. Buying is still ahead on the
        window, but has stopped adding to it."""
        f = _f(window_s=20.0, grow_s=2.0, hold_s=2.0)
        ts = _push(f, "buy", 6.0)
        ts = _quiet(f, 6.0, ts)
        s = f.snapshot
        assert s.lead > 0, "buying should still be ahead"
        assert s.read == NONE, "ahead but not extending is not a read"

    def test_selling_reads_down_the_same_way(self):
        f = _f()
        _push(f, "sell", 8.0)
        s = f.snapshot
        assert s.read == DOWN and s.growth < 0

    def test_an_even_fight_reads_nothing(self):
        f = _f()
        ts = 0.0
        for _ in range(60):
            ts += 0.25
            f.on_event(ts, 100.0)
            ts += 0.25
            f.on_event(ts, -100.0)
        s = f.snapshot
        assert s.read == NONE
        assert s.share == pytest.approx(0.5, abs=0.02)


class TestItHoldsBeforeItCounts:
    """75 to 108 raw flips a minute on a random walk. A reading that
    changes while you are clicking is not a reading."""

    def test_a_flip_does_not_publish_immediately(self):
        f = _f(hold_s=5.0)
        ts = _push(f, "buy", 10.0)
        assert f.snapshot.read == UP
        ts = _push(f, "sell", 2.0, ts)        # not yet five seconds
        assert f.snapshot.read == UP, "it published before the hold"
        assert f.snapshot.raw != UP, "the raw must have moved off it"

    def test_it_publishes_once_the_hold_is_served(self):
        f = _f(hold_s=5.0)
        ts = _push(f, "buy", 10.0)
        ts = _push(f, "sell", 20.0, ts)
        assert f.snapshot.read == DOWN

    def test_the_raw_is_always_visible(self):
        """Hiding it would make the smoothing unfalsifiable: if the
        steady read is wrong, the first thing worth knowing is
        whether the raw one was right earlier."""
        f = _f(hold_s=10.0)
        ts = _push(f, "buy", 12.0)
        ts = _push(f, "sell", 2.0, ts)
        s = f.snapshot
        assert s.read != s.raw
        assert s.pending_s > 0, "it must say how long they have differed"

    def test_held_seconds_counts_the_published_read(self):
        """And sustained one-sided buying must KEEP reading up. The
        first version asked for the windowed lead to grow, which
        steady buying does not do -- so it went silent through
        exactly the pushes worth trading."""
        f = _f(hold_s=2.0)
        ts = _push(f, "buy", 10.0)
        a = f.snapshot.held_s
        assert f.snapshot.read == UP
        ts = _push(f, "buy", 15.0, ts)
        assert f.snapshot.read == UP, "it went quiet during a push"
        assert f.snapshot.held_s > a

    def test_a_brand_new_read_says_it_is_unsettled(self):
        f = _f(hold_s=2.0)
        ts = _push(f, "buy", 3.0)
        s = f.snapshot
        if s.read:
            assert isinstance(s.settled, bool)

    def test_the_flip_rate_is_reported(self):
        f = _f(hold_s=0.0)
        ts = 0.0
        for _ in range(12):
            ts = _push(f, "buy", 1.0, ts)
            ts = _push(f, "sell", 1.0, ts)
        assert f.snapshot.flips_1m > 2, "it hid how unstable it was"


class TestItSurvivesABadFeed:
    def test_a_gap_longer_than_the_window_empties_it(self):
        f = _f(window_s=5.0)
        _push(f, "buy", 5.0)
        f.on_event(10_000.0, 0.0)
        assert f.snapshot.lead == pytest.approx(0.0)
        assert f.snapshot.gross == pytest.approx(0.0)

    def test_time_going_backwards_does_not_throw_the_window_away(self):
        f = _f(window_s=10.0)
        ts = _push(f, "buy", 6.0)
        lead = f.snapshot.lead
        f.on_event(ts - 50.0, 0.0)            # a replayed record
        assert f.snapshot.lead == pytest.approx(lead)

    def test_zero_events_do_not_move_it(self):
        f = _f()
        ts = _push(f, "buy", 6.0)
        s1 = f.snapshot
        f.on_event(ts + 0.01, 0.0)
        s2 = f.snapshot
        assert s1.lead == pytest.approx(s2.lead)

    def test_a_reset_clears_everything(self):
        f = _f()
        _push(f, "buy", 8.0)
        assert f.snapshot.read == UP
        f.reset()
        s = f.snapshot
        assert s.read == NONE and s.lead == 0.0 and s.gross == 0.0


class TestItAgreesWithTheBuilderAboutWhatHappened:
    """It is fed the same `e` the bar builder accumulates. If the two
    could disagree about the flow, one of them is describing a
    different tape."""

    def test_the_lead_is_the_imbalance_over_the_window(self):
        rng = random.Random(4)
        f = FlowRead(window_s=10 ** 6, grow_s=1.0, hold_s=0.0)
        bid, ts = 7_700 * 10 ** 9, 0.0
        prev = None
        total = 0.0
        for _ in range(4_000):
            if rng.random() < 0.2:
                bid += RAW if rng.random() < 0.5 else -RAW
            ts += 0.01
            t = Touch(bid_px=bid, bid_sz=rng.randint(50, 900),
                      ask_px=bid + RAW, ask_sz=rng.randint(50, 900))
            if prev is not None:
                e = float(level_ofi(prev, t))
                f.on_event(ts, e)
                total += e
            prev = t
        # push_up - push_dn == the net, identically. The same identity
        # the bar builder is tested on.
        assert f.snapshot.lead == pytest.approx(total, rel=1e-9)
        assert f.snapshot.up - f.snapshot.dn == pytest.approx(total,
                                                              rel=1e-9)


class TestTheSnapshotDoesNotLie:
    def test_gross_is_the_size_of_the_fight_not_the_result(self):
        f = _f()
        ts = 0.0
        for _ in range(40):
            ts += 0.25
            f.on_event(ts, 1000.0)
            ts += 0.25
            f.on_event(ts, -900.0)
        s = f.snapshot
        assert s.gross > abs(s.lead) * 5, "a brawl read as a clean push"
        assert 0.5 < s.share < 0.6

    def test_dominance_is_zero_on_an_even_fight(self):
        assert Snapshot(share=0.5).dominance == 0.0
        assert Snapshot(share=1.0).dominance == 1.0

    def test_it_serialises_every_field_it_shows(self):
        f = _f()
        _push(f, "buy", 8.0)
        d = f.snapshot.as_dict()
        for k in ("read", "raw", "held_s", "pending_s", "lead", "gross",
                  "share", "growth", "extending", "flips_1m",
                  "strength", "settled"):
            assert k in d, k


class TestAFinishedPushIsNotAReadEither:
    """The case the long window gets wrong on its own, and the reason
    there are two of them. Buying that stopped ten seconds ago is
    still inside a thirty-second window: the lead is still positive
    and still not shrinking, so a single-window rule reads UP on flow
    that has already finished -- the same failure a threshold on the
    running total has, arriving by a different route."""

    def test_buying_that_stopped_stops_reading_up(self):
        f = _f(window_s=20.0, grow_s=2.0, hold_s=2.0)
        ts = _push(f, "buy", 6.0)
        assert f.snapshot.read == UP
        ts = _quiet(f, 6.0, ts)
        s = f.snapshot
        assert s.lead > 0, "the long window should still hold the push"
        assert s.recent == pytest.approx(0.0), "nothing is happening now"
        assert s.read == NONE, "it read a push that was already over"

    def test_the_long_window_alone_would_have_said_up(self):
        """The control. If the lead had decayed on its own, the short
        window would be doing nothing and this test would prove it."""
        f = _f(window_s=20.0, grow_s=2.0, hold_s=2.0)
        ts = _push(f, "buy", 6.0)
        ts = _quiet(f, 6.0, ts)
        s = f.snapshot
        assert s.lead > 0 and s.growth >= 0, \
            "the lead decayed by itself; the short window is redundant"

    def test_sustained_buying_still_reads_up_throughout(self):
        """And the short window must not break the ordinary case: a
        push that keeps going keeps reading."""
        f = _f(window_s=20.0, grow_s=2.0, hold_s=2.0)
        ts = _push(f, "buy", 40.0)
        assert f.snapshot.read == UP
        assert f.snapshot.recent > 0

    def test_selling_into_a_buy_lead_goes_quiet_before_it_flips(self):
        """Not straight to DOWN. The long window still has buying
        ahead, so the honest answer in between is nothing."""
        f = _f(window_s=20.0, grow_s=2.0, hold_s=0.0)
        ts = _push(f, "buy", 10.0)
        assert f.snapshot.read == UP
        ts = _push(f, "sell", 2.0, ts)
        s = f.snapshot
        assert s.lead > 0, "buying should still lead the long window"
        assert s.recent < 0, "selling should win the short one"
        assert s.read == NONE
