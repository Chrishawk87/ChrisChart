"""The specific ways a page of candles can lie.

A chart is an argument about what happened, so these tests are aimed at
the places where the drawing could disagree with the data:

1.  SESSION BOUNDARY. Sessions run 17:00 to 16:00 Chicago and are named
    for the day they close on. An off-by-one hour at that edge moves
    Sunday evening onto Sunday's page, splits one session across two, or
    silently changes the page count away from the 517 the audit found.
    DST moves the boundary in wall-clock terms and must not break it.

2.  A BAR ACROSS A ROLL. The contract changes and the price steps by
    carry. A bar built through that shows a range nobody traded, and on
    a chart it draws as a huge candle that looks like a real move.

3.  UP VERSUS DOWN. `close > open` is up. If `close == open` were
    classed as up, every page would overstate how often price finished
    higher -- which is the single thing a reader eyeballs these for.

4.  BODY GEOMETRY. The body spans open to close, not low to high. Swap
    them and every candle becomes a full-range block: the chart looks
    plausible and is wrong everywhere.

5.  HIGH AND LOW IN THE HEADER. The session's extremes must come from
    the data, not from the 15m frame's first and last bar. A header that
    disagrees with the panel under it discredits both.

6.  THE RTH WINDOW. 08:30 is in, 15:15 is out. An inclusive close pulls
    in the minute after the cash close.

7.  TIME AXIS. Ticks must fall inside the drawn range and on the hour,
    or the three panels stop lining up with each other.

Nothing here asserts anything about profit, edge, or what a candle
predicts. These tests check that the picture matches the rows.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import chart_pdf as C                                       # noqa: E402


def ts(y, m, d, hh=0, mm=0, ss=0) -> float:
    """An epoch second from a Chicago wall clock."""
    return dt.datetime(y, m, d, hh, mm, ss, tzinfo=C.TZ).timestamp()


# ------------------------------------------------- 1. session boundary

class TestSessionDay:

    def test_morning_belongs_to_its_own_date(self):
        assert C.session_day(ts(2025, 3, 12, 9, 30)) == "2025-03-12"

    def test_1659_is_still_the_closing_day(self):
        assert C.session_day(ts(2025, 3, 12, 16, 59)) == "2025-03-12"

    def test_1700_rolls_to_the_next_day(self):
        assert C.session_day(ts(2025, 3, 12, 17, 0)) == "2025-03-13"

    def test_1659_and_1700_differ(self):
        """The boundary exists at all -- a constant would pass the pair
        of tests above only if they disagree."""
        a = C.session_day(ts(2025, 3, 12, 16, 59, 59))
        b = C.session_day(ts(2025, 3, 12, 17, 0, 0))
        assert a != b

    def test_sunday_evening_is_mondays_session(self):
        # 2025-03-09 is a Sunday.
        assert C.session_day(ts(2025, 3, 9, 18, 0)) == "2025-03-10"

    def test_midnight_belongs_to_the_day_that_closes(self):
        assert C.session_day(ts(2025, 3, 13, 0, 30)) == "2025-03-13"

    def test_one_session_spans_the_boundary_as_one_name(self):
        """17:05 Tuesday and 09:00 Wednesday are the SAME session."""
        a = C.session_day(ts(2025, 3, 11, 17, 5))
        b = C.session_day(ts(2025, 3, 12, 9, 0))
        assert a == b == "2025-03-12"

    def test_survives_the_spring_dst_change(self):
        """2025-03-09 is the US spring-forward. The 17:00 rule is in
        local time, so it holds on both sides without arithmetic on
        fixed offsets."""
        before = C.session_day(ts(2025, 3, 7, 18, 0))   # Fri evening
        after = C.session_day(ts(2025, 3, 10, 18, 0))   # Mon evening
        assert before == "2025-03-08"
        assert after == "2025-03-11"

    def test_survives_the_autumn_dst_change(self):
        assert C.session_day(ts(2025, 11, 3, 17, 30)) == "2025-11-04"

    def test_matches_data_audits_rule(self):
        """The page count must equal the session count already
        reported, so the two rules have to be the same rule."""
        sys.path.insert(
            0, str(Path(__file__).resolve().parent.parent / "scripts"))
        import data_audit
        for t in (ts(2025, 3, 9, 18, 0), ts(2025, 3, 12, 16, 59),
                  ts(2025, 3, 12, 17, 0), ts(2025, 7, 1, 0, 1),
                  ts(2025, 11, 3, 17, 30)):
            assert C.session_day(t) == data_audit._session_day(t)


# ------------------------------------------------------ 6. RTH window

class TestRTH:

    def test_open_is_included(self):
        assert C.in_rth(ts(2025, 3, 12, 8, 30, 0))

    def test_just_before_open_is_out(self):
        assert not C.in_rth(ts(2025, 3, 12, 8, 29, 59))

    def test_mid_session_is_in(self):
        assert C.in_rth(ts(2025, 3, 12, 12, 0))

    def test_close_itself_is_out(self):
        """15:15:00 opens the minute AFTER the cash close."""
        assert not C.in_rth(ts(2025, 3, 12, 15, 15, 0))

    def test_last_second_in_is_included(self):
        assert C.in_rth(ts(2025, 3, 12, 15, 14, 59))

    def test_overnight_is_out(self):
        assert not C.in_rth(ts(2025, 3, 12, 3, 0))


# --------------------------------------------------- 2. bars and rolls

class TestBars:

    def test_one_slot_aggregates(self):
        b = C.Bars(60)
        b.feed(ts(2025, 3, 12, 9, 0, 0), 10, 11, 9, 10.5, 5, "A")
        b.feed(ts(2025, 3, 12, 9, 0, 30), 10.5, 12, 10, 11, 7, "A")
        out = b.finish()
        assert len(out) == 1
        slot, o, h, lo, c, v, k = out[0]
        assert (o, h, lo, c, v, k) == (10, 12, 9, 11, 12, "A")

    def test_high_is_the_max_not_the_last(self):
        """A weak fixture can let `cur.high = h` pass. The extreme is
        put in the MIDDLE so only a real max() survives."""
        b = C.Bars(60)
        t = ts(2025, 3, 12, 9, 0, 0)
        b.feed(t, 10, 10.5, 9.5, 10, 1, "A")
        b.feed(t + 10, 10, 99, 1, 10, 1, "A")      # the extremes
        b.feed(t + 20, 10, 10.5, 9.5, 10, 1, "A")  # a quiet last print
        out = b.finish()
        assert out[0][2] == 99
        assert out[0][3] == 1

    def test_a_new_slot_closes_the_bar(self):
        b = C.Bars(60)
        b.feed(ts(2025, 3, 12, 9, 0, 59), 10, 10, 10, 10, 1, "A")
        b.feed(ts(2025, 3, 12, 9, 1, 0), 20, 20, 20, 20, 1, "A")
        assert len(b.finish()) == 2

    def test_a_roll_closes_the_bar_mid_slot(self):
        """Both prints are in the SAME minute. Only the contract change
        can split them -- this is the roll guard."""
        b = C.Bars(60)
        t = ts(2025, 3, 12, 9, 0, 0)
        b.feed(t, 100, 100, 100, 100, 1, "MAR")
        b.feed(t + 30, 140, 140, 140, 140, 1, "JUN")
        out = b.finish()
        assert len(out) == 2
        assert out[0][6] == "MAR" and out[1][6] == "JUN"

    def test_no_bar_carries_two_contracts(self):
        b = C.Bars(900)
        t = ts(2025, 3, 12, 9, 0, 0)
        for i in range(10):
            b.feed(t + i * 60, 100, 100, 100, 100, 1,
                   "MAR" if i < 5 else "JUN")
        for bar in b.finish():
            assert bar[6] in ("MAR", "JUN")
        spans = {bar[6] for bar in b.finish()}
        assert spans == {"MAR", "JUN"}

    def test_a_roll_bar_does_not_span_the_carry_gap(self):
        """The failure this guards: one bar whose low is the old
        contract and whose high is the new one."""
        b = C.Bars(900)
        t = ts(2025, 3, 12, 9, 0, 0)
        b.feed(t, 5800, 5800, 5800, 5800, 1, "MAR")
        b.feed(t + 60, 5850, 5850, 5850, 5850, 1, "JUN")
        for _slot, _o, h, lo, _c, _v, _k in b.finish():
            assert h - lo < 1.0

    def test_slot_floors_to_the_span(self):
        b = C.Bars(900)
        b.feed(ts(2025, 3, 12, 9, 7, 13), 1, 1, 1, 1, 1, "A")
        slot = b.finish()[0][0]
        assert slot == ts(2025, 3, 12, 9, 0, 0)

    def test_finish_is_idempotent(self):
        b = C.Bars(60)
        b.feed(ts(2025, 3, 12, 9, 0), 1, 1, 1, 1, 1, "A")
        assert len(b.finish()) == 1
        assert len(b.finish()) == 1


# ----------------------------------------- 3 & 4. candle classification

class TestCandleGeometry:

    def test_up_is_hollow_down_is_filled(self):
        bars = [(0, 10, 12, 9, 11, 1, "A")]          # close > open
        uw, dw, ub, db = C.candle_geometry(bars, 60)
        assert len(ub) == 1 and len(db) == 0
        assert len(uw) == 1 and len(dw) == 0

    def test_a_down_bar_goes_to_the_down_collection(self):
        bars = [(0, 11, 12, 9, 10, 1, "A")]
        _uw, dw, ub, db = C.candle_geometry(bars, 60)
        assert len(db) == 1 and len(ub) == 0 and len(dw) == 1

    def test_an_unchanged_bar_is_not_counted_as_up(self):
        """close == open. Classing it up would overstate how often
        price finished higher on every page in the document."""
        bars = [(0, 10, 11, 9, 10, 1, "A")]
        _uw, _dw, ub, db = C.candle_geometry(bars, 60)
        assert len(ub) == 0
        assert len(db) == 1

    def test_an_unchanged_bar_has_no_body_height(self):
        bars = [(0, 10, 11, 9, 10, 1, "A")]
        _uw, _dw, _ub, db = C.candle_geometry(bars, 60)
        ys = {y for _x, y in db[0]}
        assert ys == {10}

    def test_body_spans_open_to_close_not_low_to_high(self):
        bars = [(0, 10, 50, 1, 11, 1, "A")]
        _uw, _dw, ub, _db = C.candle_geometry(bars, 60)
        ys = sorted({y for _x, y in ub[0]})
        assert ys == [10, 11]

    def test_body_is_ordered_low_to_high_for_a_down_bar(self):
        """min/max, not open/close in the order given."""
        bars = [(0, 11, 12, 9, 10, 1, "A")]
        _uw, _dw, _ub, db = C.candle_geometry(bars, 60)
        ys = sorted({y for _x, y in db[0]})
        assert ys == [10, 11]

    def test_the_body_is_a_rectangle_not_a_bowtie(self):
        """The four vertices must walk the outline in order. Swapping a
        pair draws an hourglass that still has the right bounding box,
        so a bounding-box assertion alone cannot see it.

        (Writing the pair as `o, c` instead of `max, min` is an
        EQUIVALENT mutation -- it reverses the winding, and a reversed
        rectangle is the same rectangle. This test pins the shape,
        which is the property that actually matters.)
        """
        for bars in ([(0, 10, 12, 9, 11, 1, "A")],      # up
                     [(0, 11, 12, 9, 10, 1, "A")]):     # down
            _uw, _dw, ub, db = C.candle_geometry(bars, 60)
            poly = (ub or db)[0]
            assert len(poly) == 4
            # Consecutive vertices share exactly one coordinate: that is
            # true of a rectangle's outline and false of a bowtie.
            for i in range(4):
                a, b = poly[i], poly[(i + 1) % 4]
                same = (a[0] == b[0]) + (a[1] == b[1])
                assert same == 1, f"{a} -> {b} is a diagonal"

    def test_wick_spans_low_to_high(self):
        bars = [(0, 10, 50, 1, 11, 1, "A")]
        uw, _dw, _ub, _db = C.candle_geometry(bars, 60)
        (x0, y0), (x1, y1) = uw[0]
        assert sorted((y0, y1)) == [1, 50]
        assert x0 == x1                     # vertical

    def test_body_is_centred_on_the_slot(self):
        bars = [(600, 10, 12, 9, 11, 1, "A")]
        _uw, _dw, ub, _db = C.candle_geometry(bars, 60)
        xs = sorted({x for x, _y in ub[0]})
        assert (xs[0] + xs[1]) / 2 == pytest.approx(630)   # slot + span/2

    def test_body_is_narrower_than_the_slot(self):
        """Adjacent candles must not touch."""
        bars = [(0, 10, 12, 9, 11, 1, "A")]
        _uw, _dw, ub, _db = C.candle_geometry(bars, 60)
        xs = sorted({x for x, _y in ub[0]})
        assert 0 < xs[1] - xs[0] < 60

    def test_every_bar_produces_exactly_one_wick_and_one_body(self):
        bars = [(i * 60, 10, 12, 9, 11 if i % 2 else 9.5, 1, "A")
                for i in range(20)]
        uw, dw, ub, db = C.candle_geometry(bars, 60)
        assert len(uw) + len(dw) == 20
        assert len(ub) + len(db) == 20

    def test_empty_input_draws_nothing(self):
        assert C.candle_geometry([], 60) == ([], [], [], [])


# ------------------------------------------------- 5. the page header

class TestSessionStats:

    def session(self, prints):
        s = C.Session.start("2025-03-12")
        for t, px, k in prints:
            s.feed(t, px, px, px, px, 1, k)
        s.close()
        return s

    def ohlc_session(self, bars):
        """Bars whose open and close DIFFER.

        The single-price helper above gives every bar open == close, so
        a header that printed the open where it meant the close would
        read correctly in all of its fixtures.
        """
        s = C.Session.start("2025-03-12")
        for t, o, h, lo, c, k in bars:
            s.feed(t, o, h, lo, c, 1, k)
        s.close()
        return s

    def test_high_and_low_come_from_the_data(self):
        t = ts(2025, 3, 12, 9, 0)
        s = self.session([(t, 5800, "A"), (t + 30, 5875.75, "A"),
                          (t + 60, 5792.75, "A"), (t + 90, 5809, "A")])
        st = s.stats()
        assert st["high"] == 5875.75
        assert st["low"] == 5792.75

    def test_open_is_the_first_open_not_the_first_close(self):
        t = ts(2025, 3, 12, 9, 0)
        s = self.ohlc_session([
            (t, 5800, 5810, 5795, 5805, "A"),
            (t + 120, 5805, 5860, 5800, 5850, "A"),
        ])
        assert s.stats()["open"] == 5800

    def test_close_is_the_last_close_not_the_last_open(self):
        t = ts(2025, 3, 12, 9, 0)
        s = self.ohlc_session([
            (t, 5800, 5810, 5795, 5805, "A"),
            (t + 120, 5805, 5860, 5800, 5850, "A"),
        ])
        assert s.stats()["close"] == 5850

    def test_the_three_frames_agree_on_the_four_prices(self):
        """The header reads the 1m frame. That is only safe because the
        coarser frames give the same four numbers -- asserted here
        rather than assumed."""
        t = ts(2025, 3, 12, 9, 0)
        s = self.ohlc_session([
            (t + i * 60, 5800 + i, 5805 + i, 5795 + i, 5802 + i, "A")
            for i in range(40)
        ])
        for name in ("1m", "5m", "15m"):
            bars = sorted(s.frames[name], key=lambda b: b[0])
            assert bars[0][1] == s.stats()["open"]
            assert bars[-1][4] == s.stats()["close"]
            assert max(b[2] for b in bars) == s.stats()["high"]
            assert min(b[3] for b in bars) == s.stats()["low"]

    def test_range_in_ticks_uses_the_es_tick(self):
        t = ts(2025, 3, 12, 9, 0)
        s = self.session([(t, 5800, "A"), (t + 60, 5801, "A")])
        assert s.stats()["range"] == pytest.approx(1.0)
        assert s.stats()["ticks"] == 4          # 1 point = 4 ticks

    def test_an_extreme_inside_a_bar_is_not_lost(self):
        """The spike happens mid-bar and is neither the open nor the
        close of any 15m candle. Reading extremes off the coarse frame
        would miss it."""
        t = ts(2025, 3, 12, 9, 7, 0)
        s = self.session([(ts(2025, 3, 12, 9, 0), 5800, "A"),
                          (t, 5999, "A"),
                          (ts(2025, 3, 12, 9, 14), 5801, "A")])
        assert s.stats()["high"] == 5999

    def test_empty_session_has_no_stats(self):
        s = C.Session.start("2025-03-12")
        s.close()
        assert s.stats() == {}
        assert s.empty

    def test_a_roll_is_recorded_with_its_time(self):
        t = ts(2025, 3, 12, 9, 0)
        s = self.session([(t, 5800, "MAR"), (t + 60, 5850, "JUN")])
        assert s.contracts == ["MAR", "JUN"]
        assert len(s.rolls) == 1
        assert s.rolls[0] == t + 60

    def test_no_roll_in_a_single_contract_session(self):
        t = ts(2025, 3, 12, 9, 0)
        s = self.session([(t, 5800, "MAR"), (t + 60, 5801, "MAR")])
        assert s.rolls == []


# ------------------------------------------------------ 7. the x axis

class TestTimeTicks:

    def test_ticks_land_on_the_hour(self):
        lo = ts(2025, 3, 12, 8, 37)
        hi = ts(2025, 3, 12, 15, 15)
        for t in C.time_ticks(lo, hi):
            assert dt.datetime.fromtimestamp(t, C.TZ).minute == 0

    def test_ticks_are_inside_the_range(self):
        lo = ts(2025, 3, 12, 8, 37)
        hi = ts(2025, 3, 12, 15, 15)
        for t in C.time_ticks(lo, hi):
            assert lo <= t <= hi

    def test_a_long_session_does_not_produce_one_tick_an_hour(self):
        lo = ts(2025, 3, 11, 17, 0)
        hi = ts(2025, 3, 12, 16, 0)
        assert 4 <= len(C.time_ticks(lo, hi)) <= 14

    def test_a_short_window_still_gets_ticks(self):
        lo = ts(2025, 3, 12, 8, 30)
        hi = ts(2025, 3, 12, 11, 30)
        assert len(C.time_ticks(lo, hi)) >= 2

    def test_ticks_ascend(self):
        out = C.time_ticks(ts(2025, 3, 12, 0, 0), ts(2025, 3, 12, 16, 0))
        assert out == sorted(out)


# ----------------------------------------------- the whole page, drawn

class TestPage:

    def build(self, day="2025-03-12", contracts=("A",)):
        """A session whose bars have DIFFERENT lows from one another.

        An oscillation that returns to the same extreme every bar gives
        every bar the same low, and then a y-range computed from
        max(lows) instead of min(lows) still contains every bar. The
        drift below keeps the lows distinct so that cannot happen.
        """
        s = C.Session.start(day)
        base = ts(2025, 3, 11, 17, 0)
        px = 5800.0
        for n, i in enumerate(range(0, 4 * 3600, 5)):
            k = contracts[min(len(contracts) - 1, i // (2 * 3600))]
            px += 0.25 if (n * 7919) % 11 < 6 else -0.25
            s.feed(base + i, px, px + 0.5, px - 0.5, px, 3, k)
        s.close()
        lows = [b[3] for b in s.frames["15m"]]
        assert len(set(lows)) > 3, "fixture is too flat to be a test"
        return s

    def figure(self):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        return Figure(figsize=(11, 8.5), dpi=100)

    def test_a_page_draws_three_panels(self):
        fig = self.figure()
        C.session_page(fig, self.build(), 1, rth=False)
        assert len(fig.axes) == 3

    def test_the_three_panels_share_one_x_range(self):
        """The alignment between timeframes is the point of the page."""
        fig = self.figure()
        C.session_page(fig, self.build(), 1, rth=False)
        spans = {ax.get_xlim() for ax in fig.axes}
        assert len(spans) == 1

    def test_the_date_is_on_the_page(self):
        fig = self.figure()
        C.session_page(fig, self.build(), 7, rth=False)
        text = " ".join(t.get_text() for t in fig.texts)
        assert "2025-03-12" in text
        assert "page 7" in text

    def test_a_roll_is_named_on_the_page(self):
        fig = self.figure()
        C.session_page(fig, self.build(contracts=("MAR", "JUN")), 1,
                       rth=False)
        text = " ".join(t.get_text() for t in fig.texts)
        assert "ROLL" in text

    def test_no_roll_notice_without_a_roll(self):
        fig = self.figure()
        C.session_page(fig, self.build(), 1, rth=False)
        text = " ".join(t.get_text() for t in fig.texts)
        assert "ROLL" not in text

    def test_an_empty_session_draws_nothing(self):
        fig = self.figure()
        s = C.Session.start("2025-03-12")
        s.close()
        C.session_page(fig, s, 1, rth=False)
        assert fig.axes == []

    def test_a_panel_with_no_bars_says_so(self):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        fig = Figure()
        ax = fig.add_subplot(111)
        C.draw_panel(ax, [], 60, "1m", (0, 100))
        assert any("no bars" in t.get_text() for t in ax.texts)

    def test_the_y_range_covers_every_bar(self):
        fig = self.figure()
        s = self.build()
        C.session_page(fig, s, 1, rth=False)
        lo_ax, hi_ax = fig.axes[0].get_ylim()
        bars = s.frames["15m"]
        assert lo_ax <= min(b[3] for b in bars)
        assert hi_ax >= max(b[2] for b in bars)


# ----------------------------------------------------- no order path

class TestSuggestsOnly:

    def test_the_script_cannot_place_an_order(self):
        src = (Path(__file__).resolve().parent.parent
               / "scripts" / "chart_pdf.py").read_text()
        for bad in ("def submit", "def execute", "def place", "def order",
                    "requests.post", "urllib.request"):
            assert bad not in src, f"chart_pdf.py contains {bad!r}"
