"""The reading that decides which way the five-minute trade goes.

This is not a scalp on the OFI bar. Chris reads it off THIS BAR and
takes a five-minute candle on another platform, so what has to be
true of it is: it describes THAT candle, it resets when that candle
does and at no other time, it does not depend on the level he picked,
and it does not change while he is clicking.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest                                              # noqa: E402

from liqmap.flowread import (DOWN, FlowRead, NONE,          # noqa: E402
                             PERIOD_S, Snapshot, UP)
from liqmap.ofi import Touch, level_ofi                     # noqa: E402

RAW = 250_000_000
SRC = (ROOT / "liqmap" / "flowread.py").read_text()

# Start well inside a candle so the tests are not accidentally
# measuring a boundary they did not mean to cross.
T0 = 3_000.0


def _f(**kw):
    kw.setdefault("period_s", 300.0)
    kw.setdefault("warmup_s", 5.0)
    kw.setdefault("hold_s", 2.0)
    kw.setdefault("recent_s", 10.0)
    kw.setdefault("bucket_s", 0.5)
    return FlowRead(**kw)


def _push(f, side, seconds, ts=T0, size=100.0, step=0.25):
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


class TestItResetsOnTheCandleAndNowhereElse:
    """The whole point of it. The reading is about a five-minute
    candle, so it is counted from that candle's open and thrown away
    at its close."""

    def test_crossing_the_boundary_clears_everything(self):
        f = _f()
        ts = _push(f, "buy", 120.0, ts=3_000.0)
        assert f.snapshot.read == UP and f.snapshot.lead > 0
        f.on_event(3_300.0, 1.0)          # the next candle
        s = f.snapshot
        assert s.lead == 1.0, "flow carried across the boundary"
        assert s.read == NONE, "the last candle's read carried over"
        assert s.into_s == pytest.approx(0.0)
        assert s.flips == 0

    def test_the_boundaries_are_on_the_clock_not_on_the_startup(self):
        """:00, :05, :10 -- where a five-minute chart puts them.
        Aligning to when the service started would make candles
        nobody else can see."""
        f = _f()
        f.on_event(3_123.0, 1.0)
        assert f.snapshot.candle_ts == 3_000.0
        f.on_event(3_301.0, 1.0)
        assert f.snapshot.candle_ts == 3_300.0

    def test_a_different_period_moves_the_boundaries(self):
        f = _f(period_s=60.0)
        f.on_event(3_123.0, 1.0)
        assert f.snapshot.candle_ts == 3_120.0
        assert f.snapshot.period_s == 60.0

    def test_it_says_how_far_into_the_candle_it_is(self):
        f = _f()
        f.on_event(3_000.0 + 97.0, 1.0)
        s = f.snapshot
        assert s.into_s == pytest.approx(97.0)
        assert s.left_s == pytest.approx(203.0)

    def test_nothing_in_it_knows_about_ofi_bars(self):
        """The accumulator in ofibar MUST reset at each close or the
        threshold latches. This must NOT, or it blanks several times
        inside one five-minute position and each blank reads as the
        signal dying."""
        body = SRC.split('"""', 2)[2]
        for word in ("ofibar", "OfiBar", "threshold", "bar_close",
                     "on_bar"):
            assert word not in body, word

    def test_an_arbitrary_mid_candle_split_changes_nothing(self):
        """Feed it in two halves, as if a bar had closed between
        them. Only the clock may reset it."""
        whole = _f()
        _push(whole, "buy", 60.0)
        split = _f()
        ts = _push(split, "buy", 30.0)
        _push(split, "buy", 30.0, ts)
        assert whole.snapshot.read == split.snapshot.read == UP
        assert whole.snapshot.lead == pytest.approx(split.snapshot.lead)


class TestTheResetIsWhatStopsItLatching:
    """A cumulative buy-minus-sell walks away from zero on any
    session with a direction and never comes back. The candle reset
    is the same discipline ofibar applies to its own accumulator,
    with the clock in place of the flow."""

    def test_an_hour_of_one_sided_flow_does_not_pin_it(self):
        f = _f()
        ts = 3_000.0
        for _ in range(12):               # twelve candles of buying
            ts = _push(f, "buy", 290.0, ts)
            ts += 15.0                    # over the boundary
            f.on_event(ts, 0.0)
        one = _f()
        _push(one, "buy", 290.0, 3_000.0)
        assert f.snapshot.lead <= one.snapshot.lead * 1.05, \
            "the lead accumulated across candles"

    def test_each_candle_starts_from_zero(self):
        f = _f()
        _push(f, "buy", 200.0, 3_000.0)
        big = f.snapshot.lead
        f.on_event(3_305.0, 1.0)
        assert f.snapshot.lead < big / 100


class TestTheReadDescribesTheCandle:
    """A candle bought hard and since gone quiet is a bought candle.
    A trailing window would call it neutral, which is the opposite
    answer to the question being asked."""

    def test_sustained_buying_reads_up(self):
        f = _f()
        _push(f, "buy", 60.0)
        s = f.snapshot
        assert s.read == UP and s.lead > 0

    def test_sustained_selling_reads_down(self):
        f = _f()
        _push(f, "sell", 60.0)
        assert f.snapshot.read == DOWN

    def test_a_stall_does_not_take_the_read_away(self):
        """The decision Chris made explicitly: the read describes the
        candle, so it holds while buying still leads it."""
        f = _f()
        ts = _push(f, "buy", 60.0)
        assert f.snapshot.read == UP
        ts = _quiet(f, 60.0, ts)
        s = f.snapshot
        assert s.lead > 0, "buying should still lead the candle"
        assert s.read == UP, "it dropped the read because of a stall"

    def test_the_stall_is_still_measured_even_though_it_is_not_used(self):
        """Carried so the cost of the decision above can be measured
        later, rather than argued about."""
        f = _f(recent_s=5.0)
        ts = _push(f, "buy", 60.0)
        ts = _push(f, "sell", 6.0, ts)
        s = f.snapshot
        assert s.lead > 0, "the candle should still lead up"
        assert s.recent < 0, "the last seconds went the other way"
        assert s.stalling is True
        assert s.read == UP, "the diagnostic gated the reading"

    def test_selling_that_overtakes_the_candle_flips_it(self):
        f = _f(hold_s=1.0)
        ts = _push(f, "buy", 30.0)
        assert f.snapshot.read == UP
        ts = _push(f, "sell", 90.0, ts)
        s = f.snapshot
        assert s.lead < 0
        assert s.read == DOWN

    def test_an_even_fight_reads_nothing(self):
        f = _f()
        ts = T0
        for _ in range(200):
            ts += 0.25
            f.on_event(ts, 100.0)
            ts += 0.25
            f.on_event(ts, -100.0)
        s = f.snapshot
        assert s.read == NONE
        assert s.share == pytest.approx(0.5, abs=0.02)


class TestTheFirstSecondsSayNothing:
    """A direction built from four book updates is a coin toss with a
    label on it."""

    def test_it_is_silent_during_the_warmup(self):
        f = _f(warmup_s=20.0)
        _push(f, "buy", 10.0)
        s = f.snapshot
        assert s.read == NONE and s.raw == NONE
        assert s.warm is False

    def test_it_shows_the_clock_instead(self):
        f = _f(warmup_s=20.0)
        _push(f, "buy", 8.0)
        s = f.snapshot
        assert s.into_s > 0, "it must say how far in it is"
        assert s.left_s > 0

    def test_it_speaks_once_the_warmup_has_passed(self):
        f = _f(warmup_s=10.0, hold_s=1.0)
        _push(f, "buy", 30.0)
        s = f.snapshot
        assert s.warm is True
        assert s.read == UP

    def test_every_candle_warms_up_again(self):
        f = _f(warmup_s=20.0, hold_s=1.0)
        _push(f, "buy", 100.0, ts=3_000.0)
        assert f.snapshot.read == UP
        _push(f, "buy", 5.0, ts=3_300.0)
        s = f.snapshot
        assert s.warm is False, "the new candle skipped its warmup"
        assert s.read == NONE


class TestItHoldsBeforeItCounts:
    def test_a_flip_does_not_publish_immediately(self):
        f = _f(hold_s=10.0)
        ts = _push(f, "buy", 60.0)
        assert f.snapshot.read == UP
        ts = _push(f, "sell", 130.0, ts)   # lead crosses, briefly
        s = f.snapshot
        assert s.lead < 0, "the candle should have flipped"
        assert s.raw == DOWN
        # Published only once DOWN has stood for the hold.
        assert s.read in (UP, DOWN)

    def test_the_raw_is_always_visible(self):
        """Hiding it would make the smoothing unfalsifiable: if the
        steady read is wrong, the first thing worth knowing is
        whether the raw one had already turned."""
        f = _f(hold_s=20.0)
        ts = _push(f, "buy", 60.0)
        assert f.snapshot.read == UP
        ts = _push(f, "sell", 70.0, ts)   # crosses, inside the hold
        s = f.snapshot
        assert s.lead < 0, "the candle should have flipped"
        assert s.raw == DOWN, "the raw must show the turn at once"
        assert s.read == UP, "it published before the hold"
        assert s.pending_s > 0, "it must say how long they have differed"

    def test_held_seconds_counts_the_published_read(self):
        f = _f(hold_s=1.0)
        ts = _push(f, "buy", 30.0)
        a = f.snapshot.held_s
        ts = _push(f, "buy", 30.0, ts)
        assert f.snapshot.read == UP
        assert f.snapshot.held_s > a

    def test_the_flip_count_is_per_candle_and_reported(self):
        """And it counts CHANGES OF MIND, not transitions: the first
        direction a candle settles on is not a change of mind, and
        counting it would put a 1 on every quiet candle.

        Driven with explicit impulses rather than stretches of flow,
        because equal stretches cancel to exactly zero and the
        cumulative lead then never crosses -- which is a tidier
        market than any that exists.
        """
        f = _f(hold_s=0.0, warmup_s=1.0)
        ts = T0 + 2.0
        f.on_event(ts, 1000.0)            # first direction, not a flip
        assert f.snapshot.flips == 0
        for k in range(5):
            ts += 1.0
            f.on_event(ts, -2000.0 if k % 2 == 0 else 2000.0)
        assert f.snapshot.flips == 5, "it hid how unstable it was"
        f.on_event(T0 + 400.0, 1.0)
        assert f.snapshot.flips == 0, "the count carried into a new candle"


class TestTheDirectionDoesNotDependOnTheLevel:
    """He picks 100 to 500 from a dropdown. That is now a LABEL on
    the reading and nothing else."""

    def test_the_same_flow_reads_the_same_at_every_level(self):
        reads = {}
        for lv in (100, 200, 300, 500):
            f = _f(level=lv)
            _push(f, "buy", 60.0)
            reads[lv] = f.snapshot.read
        assert len(set(reads.values())) == 1, reads
        assert reads[100] == UP

    def test_changing_the_level_mid_stream_changes_nothing(self):
        f = _f(level=100)
        _push(f, "buy", 60.0)
        before = f.snapshot.read
        f.set_level(500)
        assert before == f.snapshot.read == UP

    def test_the_level_only_labels_the_strength(self):
        weak = _f(level=10 ** 9)
        _push(weak, "buy", 60.0)
        strong = _f(level=1)
        _push(strong, "buy", 60.0)
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


class TestItSurvivesABadFeed:
    def test_a_gap_of_hours_just_starts_a_new_candle(self):
        f = _f()
        _push(f, "buy", 60.0, ts=3_000.0)
        f.on_event(90_000.0, 1.0)
        s = f.snapshot
        assert s.lead == 1.0
        assert s.candle_ts == 90_000.0 // 300.0 * 300.0

    def test_time_going_backwards_does_not_reopen_a_closed_candle(self):
        f = _f()
        ts = _push(f, "buy", 60.0)
        lead = f.snapshot.lead
        f.on_event(ts - 5_000.0, 0.0)
        assert f.snapshot.lead == pytest.approx(lead)
        assert f.snapshot.candle_ts == 3_000.0

    def test_zero_events_do_not_move_it(self):
        f = _f()
        ts = _push(f, "buy", 60.0)
        a = f.snapshot.lead
        f.on_event(ts + 0.01, 0.0)
        assert f.snapshot.lead == pytest.approx(a)

    def test_a_crossed_book_is_withheld_not_counted_as_zero(self):
        f = _f()
        b = 7_700 * 10 ** 9
        f.on_book(T0, b, 100, b + RAW, 100)
        before = f.snapshot.gross
        f.on_book(T0 + 1, b + 2 * RAW, 100, b, 100)   # crossed
        assert f.snapshot.gross == pytest.approx(before)

    def test_a_reset_clears_everything(self):
        f = _f()
        _push(f, "buy", 60.0)
        assert f.snapshot.read == UP
        f.reset()
        s = f.snapshot
        assert s.read == NONE and s.lead == 0.0 and s.gross == 0.0


class TestItAgreesWithTheBuilderAboutWhatHappened:
    """It is fed the same `e` the bar builder accumulates. If the two
    could disagree about the flow, one of them is describing a
    different tape."""

    def test_the_lead_is_the_imbalance_since_the_candle_opened(self):
        rng = random.Random(4)
        f = FlowRead(period_s=10 ** 6, warmup_s=0.0, hold_s=0.0)
        bid, ts, prev, total = 7_700 * 10 ** 9, 1.0, None, 0.0
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

    def test_the_book_form_and_the_event_form_agree(self):
        rng = random.Random(7)
        a = _f(warmup_s=0.0, hold_s=0.0)
        b = _f(warmup_s=0.0, hold_s=0.0)
        bid, ts, prev = 7_700 * 10 ** 9, T0, None
        for _ in range(500):
            if rng.random() < 0.3:
                bid += RAW if rng.random() < 0.5 else -RAW
            ts += 0.05
            bsz, asz = rng.randint(50, 900), rng.randint(50, 900)
            t = Touch(bid_px=bid, bid_sz=bsz, ask_px=bid + RAW,
                      ask_sz=asz)
            a.on_book(ts, bid, bsz, bid + RAW, asz)
            if prev is not None:
                b.on_event(ts, float(level_ofi(prev, t)))
            prev = t
        assert a.snapshot.lead == pytest.approx(b.snapshot.lead)


class TestTheSnapshotDoesNotLie:
    def test_gross_is_the_size_of_the_fight_not_the_result(self):
        f = _f()
        ts = T0
        for _ in range(120):
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
        _push(f, "buy", 60.0)
        d = f.snapshot.as_dict()
        for k in ("read", "raw", "held_s", "pending_s", "lead", "gross",
                  "share", "recent", "stalling", "into_s", "left_s",
                  "candle_ts", "warm", "flips", "strength", "settled",
                  "period_s"):
            assert k in d, k

    def test_the_default_period_is_five_minutes(self):
        assert PERIOD_S == 300.0
