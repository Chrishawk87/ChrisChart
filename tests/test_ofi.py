"""The ways an OFI engine can report pressure that was not there.

OFI is a signed sum. Every defect below produces a number that still
looks like a measurement, which is why each gets its own test:

1.  A SIGN. Six branches, three per side. Flip one and the engine
    reports buying while the book is being sold. There is no runtime
    symptom -- the magnitudes stay plausible.

2.  A FLOAT COMPARISON. `Pb' == Pb` is the branch that fires on a
    quiet book. Compare decoded floats instead of raw integers and a
    price equal to itself can test greater-than, turning "nothing
    changed" into "the bid improved" and emitting +Qb' out of nowhere.

3.  MEASURING ACROSS A BOUNDARY. The first event, a roll, a clear and
    a sequence gap all mean `prev` is not the state this record
    followed. Measuring through one is carry or staleness wearing the
    costume of pressure.

4.  A ZERO THAT SUMS. Refusing an event must mean emitting nothing.
    Returning 0 instead puts a real number into the window and drags
    the mean toward the middle.

5.  AN INVALID TOUCH. An absent level is not a price of zero, and a
    crossed book is not a 10,000-tick spread.

6.  REBUILDING THE BOOK. MBP-10 already carries the state after the
    event. A second, locally maintained ladder can drift from the feed
    and nothing would show it.

Nothing here asserts that OFI predicts anything. These tests check
that the number equals the formula on the input given.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap import ofi                                      # noqa: E402

PT = 1_000_000_000              # one index point, raw
TICK = PT // 4                  # 0.25 points
BASE = 5800 * PT


class Lv:
    """One MBP-10 level, as an object rather than a dict.

    Deliberately not a dict: an earlier defect in this codebase
    (`action` decoded by truncating an enum's str) was invisible to
    dict fixtures and only showed up against attribute objects.
    """

    def __init__(self, bid_px=0, bid_sz=0, ask_px=0, ask_sz=0):
        self.bid_px, self.bid_sz = bid_px, bid_sz
        self.ask_px, self.ask_sz = ask_px, ask_sz


class Rec:
    """One MBP-10 record."""

    def __init__(self, levels, action="A", ts=1.0, iid=42, seq=1):
        self.levels = levels
        self.action = action
        self.ts_event = int(ts * 1e9)
        self.instrument_id = iid
        self.sequence = seq


def touch(bid_px, bid_sz, ask_px, ask_sz) -> ofi.Touch:
    return ofi.Touch(bid_px=bid_px, bid_sz=bid_sz,
                     ask_px=ask_px, ask_sz=ask_sz)


def one(bid_px, bid_sz, ask_px, ask_sz, **kw) -> Rec:
    return Rec([Lv(bid_px, bid_sz, ask_px, ask_sz)], **kw)


# ------------------------------------------------- 1. the six branches

class TestBidBranches:

    def test_bid_steps_up_contributes_the_new_size(self):
        p = touch(BASE, 50, BASE + TICK, 40)
        c = touch(BASE + TICK, 7, BASE + 2 * TICK, 40)
        # dv_bid = +7 ; ask stepped up so dv_ask = -40 ; e = 7 + 40
        assert ofi.level_ofi(p, c) == 47

    def test_bid_holds_contributes_the_size_change(self):
        p = touch(BASE, 50, BASE + TICK, 40)
        c = touch(BASE, 62, BASE + TICK, 40)
        assert ofi.level_ofi(p, c) == 12

    def test_bid_steps_down_contributes_minus_the_old_size(self):
        p = touch(BASE, 50, BASE + TICK, 40)
        c = touch(BASE - TICK, 9, BASE + TICK, 40)
        assert ofi.level_ofi(p, c) == -50

    def test_a_shrinking_bid_at_the_same_price_is_negative(self):
        p = touch(BASE, 50, BASE + TICK, 40)
        c = touch(BASE, 10, BASE + TICK, 40)
        assert ofi.level_ofi(p, c) == -40


class TestAskBranches:

    def test_ask_steps_up_is_positive(self):
        """Sellers retreated. dv_ask = -Qa, so e_n = -dv_ask = +Qa."""
        p = touch(BASE, 50, BASE + TICK, 40)
        c = touch(BASE, 50, BASE + 2 * TICK, 11)
        assert ofi.level_ofi(p, c) == 40

    def test_ask_holds_contributes_the_size_change_negated(self):
        p = touch(BASE, 50, BASE + TICK, 40)
        c = touch(BASE, 50, BASE + TICK, 55)
        assert ofi.level_ofi(p, c) == -15

    def test_ask_steps_down_is_negative(self):
        """Sellers pressed. dv_ask = +Qa', so e_n = -Qa'."""
        p = touch(BASE, 50, BASE + 2 * TICK, 40)
        c = touch(BASE, 50, BASE + TICK, 23)
        assert ofi.level_ofi(p, c) == -23


class TestSignConventions:

    def test_both_sides_rising_is_strongly_positive(self):
        p = touch(BASE, 50, BASE + TICK, 40)
        c = touch(BASE + TICK, 30, BASE + 2 * TICK, 25)
        assert ofi.level_ofi(p, c) == 30 + 40

    def test_both_sides_falling_is_strongly_negative(self):
        p = touch(BASE, 50, BASE + TICK, 40)
        c = touch(BASE - TICK, 30, BASE, 25)
        assert ofi.level_ofi(p, c) == -50 - 25

    def test_an_unchanged_book_is_exactly_zero(self):
        p = touch(BASE, 50, BASE + TICK, 40)
        assert ofi.level_ofi(p, p) == 0

    def test_the_measure_is_antisymmetric_under_mirroring(self):
        """Mirror bid and ask and the sign must flip. A formula with a
        sign error on one side only will not survive this."""
        p = touch(BASE, 50, BASE + TICK, 40)
        c = touch(BASE, 70, BASE + TICK, 44)
        mp = touch(p.ask_px, p.ask_sz, p.bid_px, p.bid_sz)
        mc = touch(c.ask_px, c.ask_sz, c.bid_px, c.bid_sz)
        assert ofi.level_ofi(p, c) == -ofi.level_ofi(mp, mc)


# --------------------------------------------- 2. integer arithmetic

class TestIntegerExactness:

    def test_the_result_is_an_int(self):
        p = touch(BASE, 50, BASE + TICK, 40)
        c = touch(BASE, 62, BASE + TICK, 40)
        assert isinstance(ofi.level_ofi(p, c), int)

    def test_equal_prices_take_the_equality_branch(self):
        """The branch that fires on a quiet book. On raw integers this
        is exact; it is the comparison that decoded floats break."""
        p = touch(7827250000000, 5, 7827500000000, 5)
        c = touch(7827250000000, 9, 7827500000000, 5)
        assert ofi.level_ofi(p, c) == 4

    def test_the_module_never_decodes_prices_to_float(self):
        src = (Path(__file__).resolve().parent.parent
               / "liqmap" / "ofi.py").read_text()
        assert "dbn.price(" not in src, (
            "a float price in the state machine reintroduces the "
            "comparison bug the raw integers exist to avoid")


# ------------------------------------------------ 3 & 4. the refusals

class TestRefusals:

    def test_the_first_event_emits_nothing(self):
        st = ofi.BookState()
        assert st.feed(one(BASE, 50, BASE + TICK, 40, seq=1)) is None
        assert st.refused.first == 1

    def test_the_first_event_guard_stands_on_its_own(self):
        """With the sequence check OFF, so the gap guard cannot cover
        for it.

        Both guards refuse the same record for different reasons, and
        either one alone makes the first event return None. Tested
        together, the first-event guard can be deleted and nothing
        fails -- it would then emit an Event whose e_n is a record
        compared against itself, a zero that sums like a measurement.
        """
        st = ofi.BookState(check_sequence=False)
        assert st.feed(one(BASE, 50, BASE + TICK, 40, seq=1)) is None
        assert st.refused.first == 1

    def test_the_second_event_emits(self):
        st = ofi.BookState()
        st.feed(one(BASE, 50, BASE + TICK, 40, seq=1))
        ev = st.feed(one(BASE, 62, BASE + TICK, 40, seq=2))
        assert ev is not None and ev.e_n == 12

    def test_a_boundary_record_is_not_a_book_change(self):
        st = ofi.BookState()
        st.feed(one(BASE, 50, BASE + TICK, 40, seq=1))
        assert st.feed(one(0, 0, 0, 0, action="N", seq=2)) is None
        assert st.refused.boundary == 1

    def test_a_boundary_does_not_disturb_the_previous_state(self):
        """It is not a book change, so the event after it must be
        measured against the record before it."""
        st = ofi.BookState()
        st.feed(one(BASE, 50, BASE + TICK, 40, seq=1))
        st.feed(one(0, 0, 0, 0, action="N", seq=2))
        ev = st.feed(one(BASE, 62, BASE + TICK, 40, seq=2))
        assert ev is not None and ev.e_n == 12

    def test_a_refusal_returns_none_not_zero(self):
        """A zero sums like a measurement; None does not."""
        st = ofi.BookState()
        out = st.feed(one(BASE, 50, BASE + TICK, 40, seq=1))
        assert out is None
        assert out != 0

    def test_a_sequence_gap_is_refused(self):
        st = ofi.BookState()
        st.feed(one(BASE, 50, BASE + TICK, 40, seq=1))
        assert st.feed(one(BASE, 62, BASE + TICK, 40, seq=9)) is None
        assert st.refused.gap == 1

    def test_after_a_gap_the_machine_resumes(self):
        st = ofi.BookState()
        st.feed(one(BASE, 50, BASE + TICK, 40, seq=1))
        st.feed(one(BASE, 62, BASE + TICK, 40, seq=9))
        ev = st.feed(one(BASE, 70, BASE + TICK, 40, seq=10))
        assert ev is not None and ev.e_n == 8

    def test_the_gap_check_can_be_turned_off(self):
        st = ofi.BookState(check_sequence=False)
        st.feed(one(BASE, 50, BASE + TICK, 40, seq=1))
        ev = st.feed(one(BASE, 62, BASE + TICK, 40, seq=9))
        assert ev is not None and ev.e_n == 12

    def test_a_clear_resets_the_state(self):
        """Sequence checking OFF, so the reset is what is tested.

        With it on, the record after a clear is refused as a gap
        whether or not the clear reset anything -- the test would pass
        against a clear that did nothing at all.
        """
        st = ofi.BookState(check_sequence=False)
        st.feed(one(BASE, 50, BASE + TICK, 40, seq=1))
        st.feed(one(0, 0, 0, 0, action="R", seq=2))
        assert st.feed(one(BASE, 62, BASE + TICK, 40, seq=3)) is None
        assert st.refused.first == 2, "the clear did not reset"

    def test_a_roll_resets_rather_than_measuring_carry(self):
        """The next contract trades dollars away. An e_n across that
        boundary is carry, and it would be enormous."""
        st = ofi.BookState()
        st.feed(one(BASE, 50, BASE + TICK, 40, iid=1, seq=1))
        st.feed(one(BASE, 55, BASE + TICK, 40, iid=1, seq=2))
        out = st.feed(one(BASE + 50 * PT, 60, BASE + 51 * PT, 40,
                          iid=2, seq=3))
        assert out is None
        assert st.refused.roll == 1

    def test_after_a_roll_the_new_contract_measures_normally(self):
        st = ofi.BookState()
        st.feed(one(BASE, 50, BASE + TICK, 40, iid=1, seq=1))
        st.feed(one(BASE + 50 * PT, 60, BASE + 51 * PT, 40,
                    iid=2, seq=2))
        ev = st.feed(one(BASE + 50 * PT, 70, BASE + 51 * PT, 40,
                         iid=2, seq=3))
        assert ev is not None and ev.e_n == 10


class TestInvalidTouch:

    def test_an_absent_level_is_refused(self):
        st = ofi.BookState()
        st.feed(one(BASE, 50, BASE + TICK, 40, seq=1))
        assert st.feed(one(0, 0, BASE + TICK, 40, seq=2)) is None
        assert st.refused.invalid == 1

    def test_an_undefined_price_is_refused(self):
        st = ofi.BookState()
        st.feed(one(BASE, 50, BASE + TICK, 40, seq=1))
        out = st.feed(one(ofi.UNDEF_PRICE, 0, BASE + TICK, 40, seq=2))
        assert out is None
        assert st.refused.invalid == 1

    def test_a_crossed_book_is_refused(self):
        st = ofi.BookState()
        st.feed(one(BASE, 50, BASE + TICK, 40, seq=1))
        assert st.feed(one(BASE + 2 * TICK, 50, BASE + TICK, 40,
                           seq=2)) is None
        assert st.refused.crossed == 1

    def test_a_locked_book_counts_as_crossed(self):
        """Bid equal to ask. 'The best bid' and 'the best ask' stop
        meaning what the formula assumes."""
        assert touch(BASE, 5, BASE, 5).crossed

    def test_a_normal_book_is_not_crossed(self):
        assert not touch(BASE, 5, BASE + TICK, 5).crossed

    def test_an_absent_level_is_not_valid(self):
        assert not touch(0, 0, BASE + TICK, 5).valid
        assert not touch(BASE, 5, ofi.UNDEF_PRICE, 5).valid

    def test_refusals_are_counted_not_dropped(self):
        st = ofi.BookState()
        st.feed(one(BASE, 50, BASE + TICK, 40, seq=1))
        st.feed(one(0, 0, 0, 0, action="N", seq=2))
        st.feed(one(BASE + 2 * TICK, 5, BASE + TICK, 5, seq=2))
        assert st.refused.total == 3
        assert set(st.refused.as_dict()) == {
            "first", "boundary", "invalid", "crossed", "gap", "roll"}


# -------------------------------------------------- 6. no rebuilt book

class TestNoRebuild:

    def test_the_state_is_only_the_previous_record(self):
        """MBP-10 carries the book after the event, so the machine
        holds one record and nothing else. A locally rebuilt ladder
        could drift from the feed invisibly."""
        st = ofi.BookState()
        r1 = one(BASE, 50, BASE + TICK, 40, seq=1)
        st.feed(r1)
        assert st.prev.touch == touch(BASE, 50, BASE + TICK, 40)
        r2 = one(BASE, 62, BASE + TICK, 40, seq=2)
        st.feed(r2)
        assert st.prev.touch == touch(BASE, 62, BASE + TICK, 40)

    def test_the_module_does_not_apply_deltas(self):
        src = (Path(__file__).resolve().parent.parent
               / "liqmap" / "ofi.py").read_text()
        for bad in ("def apply", "def _apply", "def mutate"):
            assert bad not in src


# --------------------------------------------------- multi-level OFI

class TestMultiLevel:

    def test_weights_run_from_one_to_the_floor(self):
        w = ofi.multi_level_weights(10, 0.1)
        assert len(w) == 10
        assert w[0] == pytest.approx(1.0)
        assert w[9] == pytest.approx(0.1)

    def test_weights_decay_linearly(self):
        w = ofi.multi_level_weights(10, 0.1)
        gaps = [round(w[i] - w[i + 1], 9) for i in range(9)]
        assert len(set(gaps)) == 1

    def test_weights_descend(self):
        w = ofi.multi_level_weights(10, 0.1)
        assert list(w) == sorted(w, reverse=True)

    def test_a_single_level_has_full_weight(self):
        assert ofi.multi_level_weights(1) == (1.0,)

    def test_multi_equals_touch_when_only_one_level_moves(self):
        st = ofi.BookState()
        a = Rec([Lv(BASE, 50, BASE + TICK, 40),
                 Lv(BASE - TICK, 10, BASE + 2 * TICK, 10)], seq=1)
        b = Rec([Lv(BASE, 62, BASE + TICK, 40),
                 Lv(BASE - TICK, 10, BASE + 2 * TICK, 10)], seq=2)
        st.feed(a)
        ev = st.feed(b)
        assert ev.e_n == 12
        assert ev.e_multi == pytest.approx(12.0)

    def test_a_deeper_level_is_discounted(self):
        st = ofi.BookState()
        a = Rec([Lv(BASE, 50, BASE + TICK, 40),
                 Lv(BASE - TICK, 10, BASE + 2 * TICK, 10)], seq=1)
        b = Rec([Lv(BASE, 50, BASE + TICK, 40),
                 Lv(BASE - TICK, 20, BASE + 2 * TICK, 10)], seq=2)
        st.feed(a)
        ev = st.feed(b)
        assert ev.e_n == 0, "the touch did not move"
        assert ev.e_multi == pytest.approx(0.9 * 10)


# ----------------------------------------------------------- features

class TestFeatures:

    def test_obi_is_bounded(self):
        st = ofi.BookState()
        st.feed(one(BASE, 50, BASE + TICK, 40, seq=1))
        ev = st.feed(one(BASE, 90, BASE + TICK, 10, seq=2))
        assert -1.0 <= ev.obi <= 1.0

    def test_obi_sign_follows_the_heavier_side(self):
        st = ofi.BookState()
        st.feed(one(BASE, 50, BASE + TICK, 40, seq=1))
        ev = st.feed(one(BASE, 90, BASE + TICK, 10, seq=2))
        assert ev.obi == pytest.approx(0.8)

    def test_obi_of_an_empty_level_is_zero_not_an_error(self):
        st = ofi.BookState()
        st.feed(one(BASE, 50, BASE + TICK, 40, seq=1))
        ev = st.feed(one(BASE, 0, BASE + TICK, 0, seq=2))
        assert ev.obi == 0.0

    def test_spread_is_in_raw_units(self):
        assert touch(BASE, 5, BASE + TICK, 5).spread == TICK

    def test_relative_spread_is_a_fraction_of_the_mid(self):
        t = touch(BASE, 5, BASE + TICK, 5)
        assert ofi.relative_spread(t) == pytest.approx(
            TICK / (BASE + TICK / 2), rel=1e-6)

    def test_relative_spread_of_an_invalid_touch_is_zero(self):
        assert ofi.relative_spread(touch(0, 0, 0, 0)) == 0.0

    def test_depth_sums_cover_every_level(self):
        st = ofi.BookState()
        a = Rec([Lv(BASE, 50, BASE + TICK, 40),
                 Lv(BASE - TICK, 10, BASE + 2 * TICK, 7)], seq=1)
        b = Rec([Lv(BASE, 51, BASE + TICK, 40),
                 Lv(BASE - TICK, 10, BASE + 2 * TICK, 7)], seq=2)
        st.feed(a)
        ev = st.feed(b)
        assert ev.bid_depth == 61
        assert ev.ask_depth == 47

    def test_book_slope_needs_a_spread(self):
        assert ofi.book_slope(100, 100, 0) == 0.0

    def test_book_slope_divides_by_spread_and_levels(self):
        assert ofi.book_slope(100, 100, 2, levels=10) == pytest.approx(10.0)

    def test_msg_rate_counts_only_the_window(self):
        f = ofi.Features(window_s=1.0)
        for i in range(5):
            f.observe(100.0 + i * 0.1, "A")
        assert f.msg_rate == 5
        f.observe(102.0, "A")
        assert f.msg_rate == 1, "events older than the window stayed"

    def test_cancel_to_add_is_a_ratio(self):
        f = ofi.Features(window_s=10.0)
        f.observe(100.0, "A")
        f.observe(100.1, "C")
        f.observe(100.2, "C")
        assert f.cancel_to_add == pytest.approx(2.0)

    def test_cancel_to_add_with_no_adds_is_none_not_infinity(self):
        """A ratio with a zero denominator is a question the window
        cannot answer, not an enormous number."""
        f = ofi.Features(window_s=10.0)
        f.observe(100.0, "C")
        assert f.cancel_to_add is None


# ----------------------------------------------------- no order path

class TestSuggestsOnly:

    def test_the_module_cannot_place_an_order(self):
        src = (Path(__file__).resolve().parent.parent
               / "liqmap" / "ofi.py").read_text()
        for bad in ("def submit", "def execute", "def place", "def order",
                    "requests.post", "urllib.request"):
            assert bad not in src, f"ofi.py contains {bad!r}"

    def test_the_module_prescribes_no_threshold(self):
        """Thresholds are the caller's. A constant here would become a
        default, and a default becomes a recommendation.

        Checked against the NAMESPACE, not the text: `var ** 0.5` is a
        square root, and a text match on "0.5" fails on it while
        telling you nothing about whether a threshold exists.
        """
        suspect = ("THRESHOLD", "Z_SCORE", "ZSCORE", "MIN_OFI",
                   "CUTOFF", "SIGNAL", "ENTER", "TRIGGER")
        bound = [n for n, v in vars(ofi).items()
                 if any(k in n.upper() for k in suspect)
                 and isinstance(v, (int, float))
                 and not isinstance(v, bool)]
        assert not bound, f"ofi.py prescribes {bound}"

    def test_side_has_no_default_threshold(self):
        """The signature must REQUIRE the line, so it cannot be
        called without the caller naming one."""
        import inspect
        sig = inspect.signature(ofi.Reading.side)
        p = sig.parameters["z_enter"]
        assert p.default is inspect.Parameter.empty


class TestRace:
    """The rolling window and the z-score.

    The ways a "who is winning" readout lies:

    1.  A Z-SCORE BEFORE THE BASELINE IS FULL. Computed from a handful
        of samples it is enormous, and it is enormous exactly when
        there is least information -- so the engine fires hardest at
        the open and after every gap.
    2.  A FLAT BASELINE. std of zero divides to infinity, or to a zero
        that reads as calm. Neither is a reading.
    3.  THE WRONG BASELINE. The z-score measures a WINDOW SUM, so the
        baseline must be of window sums. Against single events it is
        comparing quantities of different scale and everything is
        extreme.
    4.  A WINDOW THAT NEVER FORGETS. If old events are not subtracted
        the sum is cumulative, and drifts away forever.
    5.  A PRESCRIBED THRESHOLD. `side()` takes the line as an argument.
    """

    def test_no_z_until_the_baseline_is_full(self):
        r = ofi.Race(window=2, baseline=5)
        got = [r.push(i).z for i in (3, 1, 4, 1, 5)]
        assert got[:4] == [None, None, None, None]
        assert got[4] is not None

    def test_warm_says_so(self):
        r = ofi.Race(window=2, baseline=3)
        assert [r.push(i).warm for i in (1, 2, 3)] == [False, False, True]

    def test_a_flat_baseline_has_no_z(self):
        """Every window sum identical. std is 0, and no number divided
        by it is a reading."""
        r = ofi.Race(window=2, baseline=3)
        last = None
        for _ in range(10):
            last = r.push(5)
        assert last.std == 0.0
        assert last.z is None

    def test_the_window_forgets(self):
        r = ofi.Race(window=3, baseline=3)
        for e in (10, 10, 10):
            rd = r.push(e)
        assert rd.window == 30
        rd = r.push(0)
        assert rd.window == 20, "the oldest event was not subtracted"

    def test_the_window_sums_only_its_length(self):
        r = ofi.Race(window=2, baseline=3)
        for e in (1, 2, 3, 4):
            rd = r.push(e)
        assert rd.window == 7

    def test_the_baseline_is_of_window_sums(self):
        """With window=1 the two coincide; with window>1 they must
        not. A baseline of raw events would give a different mean."""
        r = ofi.Race(window=3, baseline=4)
        for e in (10, 10, 10, 10):
            rd = r.push(e)
        assert rd.mean == pytest.approx((10 + 20 + 30 + 30) / 4)

    def test_a_buy_burst_is_positive(self):
        r = ofi.Race(window=2, baseline=10)
        for _ in range(12):
            r.push(1)
        rd = r.push(500)
        assert rd.z is not None and rd.z > 0

    def test_a_sell_burst_is_negative(self):
        r = ofi.Race(window=2, baseline=10)
        for _ in range(12):
            r.push(1)
        rd = r.push(-500)
        assert rd.z is not None and rd.z < 0

    def test_the_measure_is_symmetric(self):
        up, dn = ofi.Race(window=2, baseline=8), ofi.Race(window=2, baseline=8)
        for i in range(10):
            a = up.push((i % 3) - 1)
            b = dn.push(-((i % 3) - 1))
        assert a.z == pytest.approx(-b.z)

    def test_side_takes_the_threshold_as_an_argument(self):
        r = ofi.Race(window=1, baseline=5)
        for e in (0, 0, 0, 1, 10):
            rd = r.push(e)
        assert rd.z is not None
        assert rd.side(0.5) == "buyers"
        assert rd.side(99.0) == "neither", "the caller's line was ignored"

    def test_side_is_neither_without_a_z(self):
        r = ofi.Race(window=1, baseline=5)
        rd = r.push(1)
        assert rd.z is None and rd.side(0.1) == "neither"

    def test_a_nonpositive_threshold_decides_nothing(self):
        r = ofi.Race(window=1, baseline=3)
        for e in (0, 0, 5):
            rd = r.push(e)
        assert rd.side(0.0) == "neither"
        assert rd.side(-1.0) == "neither"

    def test_the_periodic_recompute_changes_nothing(self):
        """Drift correction must correct drift, not the values."""
        import random
        random.seed(5)
        evs = [random.randint(-300, 300) for _ in range(4000)]
        a = ofi.Race(window=50, baseline=200, recompute_every=97)
        b = ofi.Race(window=50, baseline=200, recompute_every=10**9)
        for e in evs:
            ra, rb = a.push(e), b.push(e)
        assert ra.window == rb.window
        assert ra.mean == pytest.approx(rb.mean, rel=1e-9)
        assert ra.std == pytest.approx(rb.std, rel=1e-6)

    def test_variance_never_goes_negative(self):
        """Float drift can push the incremental variance below zero.
        A negative variance has no square root and would raise."""
        import random
        random.seed(7)
        r = ofi.Race(window=5, baseline=20)
        for _ in range(5000):
            rd = r.push(random.choice((1000000, 1000001)))
            assert rd.std >= 0.0

    def test_a_degenerate_window_is_refused(self):
        with pytest.raises(ValueError):
            ofi.Race(window=0)
        with pytest.raises(ValueError):
            ofi.Race(baseline=1)
