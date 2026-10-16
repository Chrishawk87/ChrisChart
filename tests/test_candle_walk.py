"""The specific ways a candle walkthrough can lie.

The page claims to show what each timeframe was showing AT a given
minute inside a 15-minute candle. Every way that claim can be false:

1.  LOOK-AHEAD THROUGH THE 5m BAR. At minute 1 the 5m bar has not
    finished. Reading its CLOSE instead of its open-to-here would show
    a direction that was not knowable for another four minutes. This is
    the same defect found in `tf_backtest` (an unclosed 15m bar being
    read), and it is the one that would make these pages worthless
    while looking completely normal.

2.  LOOK-AHEAD THROUGH THE 15m BAR. Same, with fourteen minutes of
    future instead of four.

3.  THE AGREEMENT RULE. Agreement is all three the same direction AND
    none flat. Counting a flat timeframe as agreeing inflates the count
    on exactly the quiet bars where nothing was happening.

4.  THE SIGN OF `to close`. Positive must mean the move CONTINUED. Drop
    the sign and every giving-back minute reads as a winner.

5.  A FABRICATED ZERO. A flat minute has no direction to sign by. If it
    is signed by zero it prints "+0 ticks to the close" for a minute
    after which price moved a hundred ticks -- a made-up number wearing
    the format of a measurement.

6.  FLOAT NOISE AS DIRECTION. `dbn.price` decodes a tick as
    7827.250000000001. Compared with a bare `>`, two prices that are
    the same tick differ, and every flat bar is classed up or down at
    random.

7.  THE ROLL-UP. The 1m bars must reproduce the 15m bar's open, high,
    low and close -- they are the same prints. The page states this
    check, so the check has to be real.

8.  GROUPING ACROSS A ROLL. A 15m slot holding a roll contains minutes
    from two contracts. Filing them under one candle draws a step of
    carry as if it were a trade.

Nothing here asserts that agreement predicts anything. These tests
check that the page describes the rows it was built from.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import candle_walk as W                                     # noqa: E402
import chart_pdf as C                                       # noqa: E402


def ts(y, m, d, hh=0, mm=0, ss=0) -> float:
    return dt.datetime(y, m, d, hh, mm, ss, tzinfo=C.TZ).timestamp()


BASE = ts(2026, 8, 4, 9, 30)


def bar(slot, o, h, lo, c, v=10, k="ESU6"):
    return (slot, o, h, lo, c, v, k)


def build(minutes, k="ESU6"):
    """A candle from (offset_minutes, open, high, low, close) rows."""
    b15, b5, b1 = W.C.Bars(900), W.C.Bars(300), W.C.Bars(60)
    for off, o, h, lo, c in minutes:
        for b in (b15, b5, b1):
            b.feed(BASE + off * 60, o, h, lo, c, 10, k)
    return W.group_candles(b15.finish(), b5.finish(), b1.finish())[0]


# ------------------------------------------------ 6. float tolerance

class TestDirection:

    def test_up_down_flat(self):
        assert W.direction(10.0, 11.0) == 1
        assert W.direction(11.0, 10.0) == -1
        assert W.direction(10.0, 10.0) == 0

    def test_tick_noise_is_not_a_direction(self):
        """`dbn.price` returns 7827.250000000001 for an exact tick."""
        a = 7827.25
        b = 7827.250000000001
        assert W.direction(a, b) == 0
        assert W.direction(b, a) == 0

    def test_a_real_tick_still_registers(self):
        assert W.direction(7827.25, 7827.50) == 1

    def test_the_tolerance_is_far_below_a_tick(self):
        """A guard that swallowed a whole tick would hide real moves."""
        assert W.EPS < W.TICK / 1000


# ------------------------------------- 1 & 2. the look-ahead guards

class TestNoLookAhead:

    def test_the_5m_direction_uses_open_to_here_not_its_close(self):
        """Minute 0 is UP. Its 5m bar ends DOWN four minutes later.

        At minute 0 the 5m bar had opened at 100 and printed 110, so
        what was on the screen was UP. Reading the 5m bar's eventual
        close (90) would report DOWN -- four minutes of future.
        """
        c = build([
            (0, 100, 110, 100, 110),      # up
            (1, 110, 110, 95, 95),
            (2, 95, 95, 92, 92),
            (3, 92, 92, 91, 91),
            (4, 91, 91, 90, 90),          # the 5m bar closes at 90
        ])
        five = c.bars5[0]
        assert five[1] == 100 and five[4] == 90      # the 5m IS down
        first = c.walk()[0]
        assert first["d5"] == 1, "read the 5m bar's close: look-ahead"

    def test_the_15m_direction_uses_open_to_here_not_its_close(self):
        c = build([
            (0, 100, 110, 100, 110),      # up
            (6, 110, 110, 80, 80),
            (14, 80, 80, 70, 70),         # the 15m closes at 70
        ])
        assert c.bar15[1] == 100 and c.bar15[4] == 70
        first = c.walk()[0]
        assert first["d15"] == 1, "read the 15m bar's close: look-ahead"

    def test_the_last_minute_agrees_with_the_finished_bars(self):
        """At the FINAL minute there is no future left, so the three
        directions must match the completed bars. This is the boundary
        the two tests above approach from the other side."""
        c = build([(i, 100 + i, 101 + i, 99 + i, 101 + i)
                   for i in range(15)])
        last = c.walk()[-1]
        assert last["d5"] == W.direction(c.bars5[-1][1], c.bars5[-1][4])
        assert last["d15"] == W.direction(c.bar15[1], c.bar15[4])

    def test_a_minutes_row_never_depends_on_later_minutes(self):
        """Truncating the candle must not change the rows that remain.

        The strongest statement of causality available here: build the
        same candle twice, once cut short, and the surviving rows must
        be identical.
        """
        rows = [(i, 100 + i, 101 + i, 99 + i, 101 + i)
                for i in range(10)]
        full = build(rows + [(i, 300 - i, 301 - i, 200, 250 - i)
                             for i in range(10, 15)])
        short = build(rows)
        for a, b in zip(full.walk()[:10], short.walk()[:10]):
            assert a["d1"] == b["d1"]
            assert a["d5"] == b["d5"]
            assert a["d15"] == b["d15"]
            assert a["agree"] == b["agree"]


# ---------------------------------------------- 3. the agreement rule

class TestAgreement:

    def test_all_three_up_agrees(self):
        c = build([(i, 100 + i, 101 + i, 99 + i, 101 + i)
                   for i in range(15)])
        assert c.walk()[3]["agree"]

    def test_all_three_down_agrees(self):
        c = build([(i, 200 - i, 201 - i, 198 - i, 198 - i)
                   for i in range(15)])
        r = c.walk()[3]
        assert r["d1"] == r["d5"] == r["d15"] == -1
        assert r["agree"]

    def test_a_disagreeing_minute_does_not_agree(self):
        """Minute 1 is DOWN while the 15m is still up from minute 0."""
        c = build([
            (0, 100, 120, 100, 120),
            (1, 120, 120, 110, 110),      # down, but 110 > open 100
        ])
        r = c.walk()[1]
        assert r["d1"] == -1 and r["d15"] == 1
        assert not r["agree"]

    def test_a_flat_timeframe_blocks_agreement(self):
        """The 1m closes where it opened. Nothing was happening, and
        counting it as agreement inflates the count on exactly the
        quiet bars."""
        c = build([
            (0, 100, 110, 100, 110),
            (1, 110, 112, 108, 110),      # flat 1m, 15m still up
        ])
        r = c.walk()[1]
        assert r["d1"] == 0
        assert not r["agree"]

    def test_three_flats_are_not_three_agreements(self):
        """All three directions are 0, so all three are EQUAL. Without
        the `!= 0` half of the rule this is the case that silently
        reads as unanimous agreement -- and it is the commonest state
        in the overnight session, where price sits on one tick."""
        c = build([
            (0, 100, 100, 100, 100),
            (1, 100, 100, 100, 100),
        ])
        r = c.walk()[1]
        assert r["d1"] == r["d5"] == r["d15"] == 0
        assert not r["agree"]

    def test_the_15m_is_actually_consulted(self):
        """A minute where the 1m and the 5m agree and the 15m does not.

        Only reachable after the first 5m bar, because the first one
        shares its open with the 15m and the two can never disagree
        inside it.
        """
        c = build(
            [(i, 100 - i * 5, 100, 95 - i * 5, 95 - i * 5)
             for i in range(5)]            # falls 100 -> 75
            + [(5, 75, 90, 75, 90)]        # 2nd 5m opens at 75, rises
        )
        r = c.walk()[5]
        assert r["d1"] == 1 and r["d5"] == 1
        assert r["d15"] == -1, "90 is still below the 15m open of 100"
        assert not r["agree"]

    def test_the_5m_is_actually_consulted(self):
        """A minute where the 1m and the 15m agree and the 5m does
        not."""
        c = build(
            [(i, 100 - i * 2, 100, 90 - i * 2, 90 - i * 2)
             for i in range(5)]            # falls 100 -> 82
            + [(5, 82, 96, 82, 95)]        # 2nd 5m opens at 82
            + [(6, 95, 95, 92, 92)]        # down on the minute
        )
        r = c.walk()[6]
        assert r["d1"] == -1, "the minute itself fell"
        assert r["d15"] == -1, "92 is below the 15m open of 100"
        assert r["d5"] == 1, "but 92 is above the 5m open of 82"
        assert not r["agree"]

    def test_the_first_minute_can_agree(self):
        c = build([(0, 100, 110, 100, 110)])
        assert c.walk()[0]["agree"]

    def test_agreement_is_not_simply_always_true(self):
        """A pullback that stays ABOVE the 15m open is the case that
        separates the timeframes: the minute is down while both the
        5m and the 15m are still up from their opens.

        (A fixture that simply falls back through the open makes all
        three agree on the way down, and would pass against a function
        that returned True unconditionally.)
        """
        c = build([
            (0, 100, 120, 100, 120),      # up hard
            (1, 120, 120, 110, 110),      # down, but still above 100
            (2, 110, 125, 110, 125),      # up again
        ])
        flags = [r["agree"] for r in c.walk()]
        assert flags == [True, False, True]


# ----------------------------------- 4 & 5. `to close` and its sign

class TestToClose:

    def test_positive_means_the_move_continued(self):
        c = build([
            (0, 100, 101, 100, 101),      # up
            (14, 101, 120, 101, 120),     # and it kept going
        ])
        assert c.walk()[0]["to_close"] > 0

    def test_negative_means_it_gave_back(self):
        c = build([
            (0, 100, 101, 100, 101),      # up
            (14, 101, 101, 80, 80),       # and it reversed
        ])
        assert c.walk()[0]["to_close"] < 0

    def test_a_down_minute_that_continues_is_also_positive(self):
        """The sign is relative to the direction showing, not to price.
        Without that, every short reads as a loss."""
        c = build([
            (0, 100, 100, 90, 90),        # down
            (14, 90, 90, 70, 70),         # and it kept falling
        ])
        r = c.walk()[0]
        assert r["d1"] == -1
        assert r["to_close"] > 0

    def test_the_last_minute_has_nothing_left(self):
        c = build([(i, 100 + i, 101 + i, 99 + i, 101 + i)
                   for i in range(15)])
        assert c.walk()[-1]["to_close"] == 0

    def test_a_flat_minute_has_no_travel_rather_than_zero(self):
        c = build([
            (0, 100, 105, 95, 100),       # flat
            (14, 100, 200, 100, 200),     # price then moved a long way
        ])
        r = c.walk()[0]
        assert r["d1"] == 0
        assert r["to_close"] is None, "a signed-by-zero fabricated zero"

    def test_travel_is_measured_in_ticks(self):
        c = build([
            (0, 100, 101, 100, 101),
            (14, 101, 102, 101, 102),     # one point beyond
        ])
        assert c.walk()[0]["to_close"] == pytest.approx(1 / W.TICK)


# -------------------------------------------------- 7. the roll-up

class TestRollup:

    # The extremes sit in the MIDDLE of this fixture, not at its end.
    # A monotone fixture lets `high = the last bar's high` pass as
    # `high = max(...)` -- the same weak-fixture defect found earlier
    # in the spreadsheet export.
    ROWS = ([(i, 100 + i, 105 + i, 95 + i, 101 + i) for i in range(6)]
            + [(6, 106, 180, 20, 107)]            # the real extremes
            + [(i, 100 + i, 105 + i, 95 + i, 101 + i)
               for i in range(7, 15)])

    def candle(self):
        return build(self.ROWS)

    def test_the_fixture_hides_its_extremes_in_the_middle(self):
        c = self.candle()
        ones = sorted(c.bars1)
        assert max(b[2] for b in ones) != ones[-1][2]
        assert min(b[3] for b in ones) != ones[-1][3]

    def test_the_ones_reproduce_the_fifteen(self):
        assert self.candle().rollup() == {"open": True, "high": True,
                                          "low": True, "close": True}

    def test_the_first_one_opens_where_the_fifteen_opens(self):
        c = self.candle()
        assert sorted(c.bars1)[0][1] == c.bar15[1]

    def test_the_last_one_closes_where_the_fifteen_closes(self):
        c = self.candle()
        assert sorted(c.bars1)[-1][4] == c.bar15[4]

    @pytest.mark.parametrize("field,index,delta", [
        ("open", 1, 7), ("high", 2, 7), ("low", 3, -7), ("close", 4, 7),
    ])
    def test_each_check_can_fail_on_its_own(self, field, index, delta):
        """A rollup() that hardcoded True for any one field would pass
        every other test here. Each field is tampered with in turn and
        must be the ONLY one that reports a mismatch."""
        c = self.candle()
        bar = list(c.bar15)
        bar[index] += delta
        c.bar15 = tuple(bar)
        r = c.rollup()
        assert r[field] is False, f"{field} mismatch went unreported"
        assert [k for k, v in r.items() if not v] == [field]

    def test_an_empty_candle_has_no_rollup(self):
        c = W.Candle(slot=BASE, contract="ESU6",
                     bar15=bar(BASE, 1, 1, 1, 1))
        assert c.rollup() == {}


# ------------------------------------------- 8. grouping and rolls

class TestGrouping:

    def test_a_candle_collects_its_own_minutes(self):
        c = build([(i, 100 + i, 101 + i, 99 + i, 101 + i)
                   for i in range(15)])
        assert len(c.bars1) == 15
        assert len(c.bars5) == 3

    def test_minutes_from_the_next_candle_are_excluded(self):
        b15, b5, b1 = W.C.Bars(900), W.C.Bars(300), W.C.Bars(60)
        for i in range(30):              # two whole candles
            for b in (b15, b5, b1):
                b.feed(BASE + i * 60, 100, 101, 99, 100.5, 10, "ESU6")
        cands = W.group_candles(b15.finish(), b5.finish(), b1.finish())
        assert len(cands) == 2
        for c in cands:
            assert len(c.bars1) == 15
            assert len(c.bars5) == 3, "the 5m window is not bounded"
            for b in c.bars1 + c.bars5:
                assert c.slot <= b[0] < c.slot + W.SPAN15

    def test_the_two_candles_do_not_share_any_bar(self):
        """The bound is half-open at both ends, so no minute is filed
        under two candles and counted twice in the census."""
        b15, b5, b1 = W.C.Bars(900), W.C.Bars(300), W.C.Bars(60)
        for i in range(30):
            for b in (b15, b5, b1):
                b.feed(BASE + i * 60, 100, 101, 99, 100.5, 10, "ESU6")
        a, z = W.group_candles(b15.finish(), b5.finish(), b1.finish())
        assert not ({b[0] for b in a.bars1} & {b[0] for b in z.bars1})
        assert not ({b[0] for b in a.bars5} & {b[0] for b in z.bars5})

    def test_a_roll_splits_the_slot_into_two_candles(self):
        b15, b5, b1 = W.C.Bars(900), W.C.Bars(300), W.C.Bars(60)
        for i in range(15):
            k = "ESU6" if i < 7 else "ESZ6"
            px = 5800 if i < 7 else 5850       # the carry step
            for b in (b15, b5, b1):
                b.feed(BASE + i * 60, px, px, px, px, 10, k)
        cands = W.group_candles(b15.finish(), b5.finish(), b1.finish())
        assert len(cands) == 2
        assert {c.contract for c in cands} == {"ESU6", "ESZ6"}

    def test_no_candle_mixes_two_contracts(self):
        b15, b5, b1 = W.C.Bars(900), W.C.Bars(300), W.C.Bars(60)
        for i in range(15):
            k = "ESU6" if i < 7 else "ESZ6"
            px = 5800 if i < 7 else 5850
            for b in (b15, b5, b1):
                b.feed(BASE + i * 60, px, px, px, px, 10, k)
        for c in W.group_candles(b15.finish(), b5.finish(),
                                 b1.finish()):
            assert {b[6] for b in c.bars1} == {c.contract}
            assert {b[6] for b in c.bars5} == {c.contract}

    def test_a_roll_candle_does_not_span_the_carry(self):
        b15, b5, b1 = W.C.Bars(900), W.C.Bars(300), W.C.Bars(60)
        for i in range(15):
            k = "ESU6" if i < 7 else "ESZ6"
            px = 5800 if i < 7 else 5850
            for b in (b15, b5, b1):
                b.feed(BASE + i * 60, px, px, px, px, 10, k)
        for c in W.group_candles(b15.finish(), b5.finish(),
                                 b1.finish()):
            assert c.bar15[2] - c.bar15[3] < 1.0

    def test_a_gap_is_reported_as_partial(self):
        c = build([(i, 100, 101, 99, 100.5)
                   for i in range(15) if i not in (5, 6, 7)])
        assert len(c.bars1) == 12
        assert c.partial

    def test_a_whole_candle_is_not_partial(self):
        c = build([(i, 100, 101, 99, 100.5) for i in range(15)])
        assert not c.partial

    def test_a_gap_does_not_invent_bars(self):
        c = build([(i, 100, 101, 99, 100.5)
                   for i in range(15) if i not in (5, 6, 7)])
        mins = {int((b[0] - BASE) // 60) for b in c.bars1}
        assert 5 not in mins and 6 not in mins and 7 not in mins


# ---------------------------------------------------- the aggregate

class TestCensus:

    def test_counts_every_minute(self):
        cen = W.Census()
        c = build([(i, 100 + i, 101 + i, 99 + i, 101 + i)
                   for i in range(15)])
        cen.add(c)
        assert cen.candles == 1
        assert cen.minutes == 15

    def test_a_flat_minute_is_counted_but_not_sampled(self):
        """It is a minute that happened; it is not a travel figure."""
        cen = W.Census()
        c = build([
            (0, 100, 105, 95, 100),       # flat
            (1, 100, 106, 100, 106),      # up
            (14, 106, 110, 106, 110),
        ])
        cen.add(c)
        assert cen.minutes == 3
        assert len(cen.agree_travel) + len(cen.other_travel) == 2

    def test_quartiles_of_nothing_do_not_explode(self):
        assert W.Census.quartiles([]) == (0.0, 0.0, 0.0)

    def test_quartiles_ascend(self):
        q = W.Census.quartiles(list(range(100)))
        assert q[0] < q[1] < q[2]

    def test_partial_candles_are_counted(self):
        cen = W.Census()
        cen.add(build([(i, 100, 101, 99, 100.5)
                       for i in range(15) if i != 4]))
        assert cen.partial == 1


# ------------------------------------------------------- the page

class TestPage:

    def figure(self):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        return Figure(figsize=(11, 8.5), dpi=100)

    def candle(self):
        return build([(i, 100 + i, 105 + i, 95 + i, 101 + i)
                      for i in range(15)])

    def test_a_page_draws_three_panels_and_a_strip(self):
        fig = self.figure()
        W.candle_page(fig, self.candle(), 1)
        assert len(fig.axes) == 4          # 15m, 5m, 1m, agreement

    def test_the_three_panels_share_one_price_axis(self):
        """The 15m bar is the envelope of the 1m bars only if they are
        drawn against the same scale."""
        fig = self.figure()
        W.candle_page(fig, self.candle(), 1)
        ylims = {fig.axes[i].get_ylim() for i in range(3)}
        assert len(ylims) == 1

    def test_every_bar_is_in_the_table(self):
        fig = self.figure()
        W.candle_page(fig, self.candle(), 1)
        text = " ".join(t.get_text() for t in fig.texts)
        assert text.count("1m") >= 15
        assert "15m" in text and "5m" in text

    def test_the_rollup_check_is_printed(self):
        fig = self.figure()
        W.candle_page(fig, self.candle(), 1)
        text = " ".join(t.get_text() for t in fig.texts)
        assert "rolls up to 15m" in text
        assert "MISMATCH" not in text

    def test_a_mismatch_is_printed_when_there_is_one(self):
        fig = self.figure()
        c = self.candle()
        s, o, h, lo, cl, v, k = c.bar15
        c.bar15 = (s, o + 7, h, lo, cl, v, k)
        W.candle_page(fig, c, 1)
        text = " ".join(t.get_text() for t in fig.texts)
        assert "MISMATCH" in text

    def test_a_partial_candle_says_so(self):
        fig = self.figure()
        c = build([(i, 100, 101, 99, 100.5)
                   for i in range(15) if i != 4])
        W.candle_page(fig, c, 1)
        text = " ".join(t.get_text() for t in fig.texts)
        assert "PARTIAL" in text

    def test_a_whole_candle_does_not(self):
        fig = self.figure()
        W.candle_page(fig, self.candle(), 1)
        text = " ".join(t.get_text() for t in fig.texts)
        assert "PARTIAL" not in text

    def test_the_agreement_count_is_on_the_page(self):
        fig = self.figure()
        c = self.candle()
        W.candle_page(fig, c, 1)
        text = " ".join(t.get_text() for t in fig.texts)
        n = c.summary()["agreed"]
        assert f"agreed on {n} of 15 minutes" in text


# ----------------------------------------------------- no order path

class TestSuggestsOnly:

    def test_the_script_cannot_place_an_order(self):
        src = (Path(__file__).resolve().parent.parent
               / "scripts" / "candle_walk.py").read_text()
        for bad in ("def submit", "def execute", "def place", "def order",
                    "requests.post", "urllib.request"):
            assert bad not in src, f"candle_walk.py contains {bad!r}"


class TestNegativeZero:
    """A down minute closing exactly at the candle's close yields
    0.0 * -1 == -0.0, which formats as "-0" -- a sign with no
    magnitude, read as a small loss."""

    def test_a_down_minute_ending_flat_is_positive_zero(self):
        c = build([
            (0, 100, 100, 90, 90),        # down
            (14, 90, 90, 90, 90),         # closes exactly there
        ])
        r = c.walk()[0]
        assert r["d1"] == -1
        assert r["to_close"] == 0
        assert f"{r['to_close']:+.0f}" == "+0"


class TestMarkdown:
    """The text breakdown must be the SAME breakdown as the page.

    It exists to be handed to a reader that parses numbers rather than
    pictures, so the one way it can fail badly is by disagreeing with
    the page it claims to mirror -- a second formatting path drifting
    from the first. Every assertion here ties a cell back to the same
    accessor `candle_page` draws from.
    """

    def candle(self):
        """A candle with BOTH agreeing and non-agreeing minutes.

        An all-agreeing fixture cannot tell "mark the agreeing ones"
        apart from "mark every one" -- the pullbacks below are what
        make the YES column a test rather than a formality.
        """
        rows = []
        for i in range(15):
            if i in (4, 9, 13):
                # Down on the minute but still above the 15m open, so
                # the 1m disagrees with the 5m and the 15m.
                rows.append((i, 100 + i, 105 + i, 95 + i, 99 + i))
            else:
                rows.append((i, 100 + i, 105 + i, 95 + i, 101 + i))
        c = build(rows)
        agreed = c.summary()["agreed"]
        assert 0 < agreed < len(c.bars1), "fixture is not mixed"
        return c

    def rows(self, md):
        return [ln for ln in md.splitlines()
                if ln.startswith("| ") and not ln.startswith("| tf")]

    def test_one_row_per_bar_at_every_timeframe(self):
        c = self.candle()
        assert len(self.rows(markdown_md(c))) == 1 + len(c.bars5) \
            + len(c.bars1)

    def test_a_partial_candle_writes_only_the_bars_it_has(self):
        c = build([(i, 100, 101, 99, 100.5)
                   for i in range(15) if i not in (5, 6)])
        assert len(self.rows(markdown_md(c))) == 1 + len(c.bars5) + 13

    def test_the_15m_numbers_match_the_bar(self):
        c = self.candle()
        first = self.rows(markdown_md(c))[0]
        for v in c.bar15[1:5]:
            assert f"{v:,.2f}" in first

    def test_the_close_agrees_with_the_page_header(self):
        c = self.candle()
        md = markdown_md(c)
        assert f"close {c.bar15[4]:,.2f}" in md

    def test_the_agreement_count_matches_summary(self):
        c = self.candle()
        s = c.summary()
        assert f"agreed on {s['agreed']} of {s['minutes']} minutes" \
            in markdown_md(c)

    def test_the_yes_count_matches_the_agreement_count(self):
        """The table and the header line are two renderings of one
        number and must not drift."""
        c = self.candle()
        md = markdown_md(c)
        yes = sum(1 for r in self.rows(md) if "| YES |" in r)
        assert yes == c.summary()["agreed"]

    def test_the_rollup_line_is_present(self):
        assert "rolls up to 15m" in markdown_md(self.candle())

    def test_a_broken_rollup_says_mismatch(self):
        c = self.candle()
        s, o, h, lo, cl, v, k = c.bar15
        c.bar15 = (s, o + 9, h, lo, cl, v, k)
        assert "MISMATCH" in markdown_md(c)

    def test_a_flat_minute_shows_no_travel(self):
        """Asserted on the LAST cell, not on the row.

        `arrow(0)` also renders "--", so a bare `"| -- |" in md`
        matches the direction column and passes even when the travel
        cell prints a fabricated "+0". The column has to be named.
        """
        c = build([
            (0, 100, 105, 95, 100),       # flat
            (14, 100, 200, 100, 200),
        ])
        row = [r for r in self.rows(markdown_md(c))
               if r.startswith("| 1m |")][0]
        cells = [x.strip() for x in row.strip("|").split("|")]
        assert cells[0] == "1m"
        assert cells[-1] == "--", f"travel cell is {cells[-1]!r}"

    def test_a_directional_minute_shows_a_number(self):
        """The counterpart: "--" must not be what every row says."""
        c = build([(0, 100, 101, 100, 101), (14, 101, 110, 101, 110)])
        row = [r for r in self.rows(markdown_md(c))
               if r.startswith("| 1m |")][0]
        cells = [x.strip() for x in row.strip("|").split("|")]
        assert cells[-1] not in ("--", "")
        assert cells[-1].startswith("+")

    def test_the_directions_match_the_walk(self):
        c = self.candle()
        md = markdown_md(c)
        for w in c.walk():
            if w["agree"]:
                continue
            # every non-agreeing minute must not be marked YES
        marked = [r for r in self.rows(md) if "| YES |" in r]
        assert len(marked) == sum(1 for w in c.walk() if w["agree"])

    def test_ticks_not_points_in_the_change_column(self):
        """A 1-point candle is +4, not +1."""
        c = build([(0, 100.00, 101.00, 100.00, 101.00)])
        assert "| +4 |" in markdown_md(c)

    def test_the_header_explains_the_to_close_trap(self):
        """Handing this to another reader without the warning invites
        exactly the wrong conclusion from the strongest-looking
        column.

        Matched against whitespace-normalised text: the header is
        hard-wrapped prose, and a phrase that straddles a line break
        is not a substring of it. Asserting on the raw string makes
        the test fail on rewrapping rather than on meaning.
        """
        h = flat_text(W.TEXT_HEADER)
        assert "NOT a measure of edge" in h
        assert "deadline the market does not observe" in h

    def test_the_header_states_the_causal_reading(self):
        assert "have NOT finished forming" in flat_text(W.TEXT_HEADER)

    def test_the_header_gives_the_tick_size(self):
        assert "0.25 points" in W.TEXT_HEADER


def markdown_md(c):
    return W.markdown_candle(c)


def flat_text(s: str) -> str:
    """Collapse hard wrapping so a phrase can be matched by meaning."""
    return " ".join(s.split())
