"""The specific ways a strategy dataset can lie.

This file is the one a strategy would be built on, so every column has
to mean exactly what its name says. The ways that can fail:

1.  FILL TIMING. `entry` must be the first print STRICTLY AFTER the
    minute ended. Taking the close, or a print stamped at the same
    second, enters inside the bar that produced the signal. That exact
    defect produced a 98.4% win rate on a random walk earlier in this
    project, so it gets the most tests here.

2.  FIRST TOUCH. When price reaches +4 and -4 ticks within the same
    window, which came first decides whether the row is a win or a
    loss. Reading it off 1-minute bars cannot answer that; reading it
    off the seconds usually can. When ONE second covers both, the
    honest answer is `ambiguous` -- not a coin flip, and not silently
    preferring the target, which inflates every win rate computed from
    this file.

3.  THE SIGN CONVENTION. `fwd_*`, `mfe`, `mae` and the bracket are all
    signed by the direction that was showing. Lose the sign and every
    short reads as its own opposite.

4.  THE ROLL. A forward walk must stop at a change of instrument. Past
    one, the next price differs by carry, which would resolve brackets
    and fill excursions with a step nobody traded.

5.  THE HORIZON. `fwd_5` is the price five minutes later -- the last
    print at or before that moment, which is what a screen shows. The
    NEXT print after a quiet stretch could be from minutes further on
    and would be reported under this horizon's name.

6.  MFE/MAE ORIENTATION. mfe is the best excursion and is never
    negative; mae is the worst and is never positive. Swap them and
    every risk figure in the file inverts.

7.  THE CAP. Fifteen minutes means fifteen minutes. A walk that runs
    past it finds touches that a 15-minute trade never saw.

Nothing here asserts that any of these columns predicts anything.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import chart_pdf as C                                       # noqa: E402
import walk_export as X                                     # noqa: E402

TICK = X.TICK
BASE = dt.datetime(2026, 8, 4, 9, 30, tzinfo=C.TZ).timestamp()


def tape(rows, contract="ESU6"):
    """rows: (offset_seconds, high, low, close) or with a contract."""
    t = X.Tape()
    for r in rows:
        off, h, lo, c = r[:4]
        k = r[4] if len(r) > 4 else contract
        t.add(BASE + off, h, lo, c, k)
    return t.freeze()


def flat(n, px=100.0, start=0, contract="ESU6"):
    return [(start + i, px, px, px, contract) for i in range(n)]


# --------------------------------------------------- 1. fill timing

class TestEntryTiming:

    def test_entry_is_the_first_print_after_the_moment(self):
        t = tape([(0, 100, 100, 100), (1, 101, 101, 101),
                  (2, 102, 102, 102)])
        i = t.after(BASE + 0)
        assert t.price_at(i) == 101

    def test_a_print_on_the_same_second_is_not_the_fill(self):
        """`after` is STRICTLY after. A print stamped at the instant
        the minute ended belongs to that minute -- filling there is
        entering inside your own signal bar."""
        t = tape([(59, 100, 100, 100), (60, 200, 200, 200)])
        i = t.after(BASE + 59)
        assert t.price_at(i) == 200, "filled on the signal's own print"

    def test_the_fill_can_be_later_than_the_next_second(self):
        """A gap in the tape means the fill is when trading resumed."""
        t = tape([(0, 100, 100, 100), (45, 108, 108, 108)])
        i = t.after(BASE + 0)
        assert t.price_at(i) == 108

    def test_no_print_after_the_moment_has_no_entry(self):
        t = tape([(0, 100, 100, 100)])
        i = t.after(BASE + 0)
        assert t.price_at(i) is None

    def test_entry_lag_is_reported_not_assumed_zero(self):
        t = tape([(0, 100, 100, 100), (30, 105, 105, 105)])
        i = t.after(BASE + 0)
        assert t.ts[i] - (BASE + 0) == 30


# ---------------------------------------------------- 2. first touch

class TestBracket:

    def test_target_first_for_a_long(self):
        t = tape([(0, 100, 100, 100)]
                 + [(1, 101, 100, 101)]          # +4 ticks
                 + [(2, 101, 99, 99)])
        assert t.bracket(0, 100.0, 1)[0] == "target"

    def test_stop_first_for_a_long(self):
        t = tape([(0, 100, 100, 100),
                  (1, 100, 99, 99),              # -4 ticks
                  (2, 101, 99, 101)])
        assert t.bracket(0, 100.0, 1)[0] == "stop"

    def test_the_sides_are_mirrored_for_a_short(self):
        """Price FALLING is the target for a short. A bracket that
        ignored side would call this a stop and invert every short in
        the file."""
        t = tape([(0, 100, 100, 100), (1, 100, 99, 99)])
        assert t.bracket(0, 100.0, -1)[0] == "target"
        assert t.bracket(0, 100.0, 1)[0] == "stop"

    def test_a_second_covering_both_is_ambiguous(self):
        """ONE second's range spans +4 and -4. Which came first is not
        in this data. Guessing it -- in either direction -- biases
        every win rate computed from this file."""
        t = tape([(0, 100, 100, 100), (1, 101.5, 98.5, 100)])
        out, _ = t.bracket(0, 100.0, 1)
        assert out == "ambiguous"

    def test_ambiguity_is_rare_not_the_default(self):
        """A guard that returned `ambiguous` always would pass the test
        above."""
        t = tape([(0, 100, 100, 100), (1, 101, 100.5, 101)])
        assert t.bracket(0, 100.0, 1)[0] == "target"

    def test_neither_when_nothing_is_touched(self):
        t = tape(flat(100, 100.0))
        assert t.bracket(0, 100.0, 1) == ("neither", None)

    def test_the_seconds_resolve_what_a_minute_bar_cannot(self):
        """Within one minute price hits -4 and then +4. A 1m bar shows
        only that both were in range; the seconds say stop."""
        secs = ([(0, 100, 100, 100)]
                + [(1, 100, 99, 99)]             # down first
                + [(2, 101, 99, 101)])           # then up
        t = tape(secs)
        assert t.bracket(0, 100.0, 1)[0] == "stop"
        assert max(r[1] for r in secs) >= 101     # a 1m bar sees both
        assert min(r[2] for r in secs) <= 99

    def test_seconds_to_first_is_measured_from_entry(self):
        t = tape([(0, 100, 100, 100), (7, 101, 100, 101)])
        out, secs = t.bracket(0, 100.0, 1)
        assert out == "target" and secs == 7

    def test_a_flat_side_resolves_nothing(self):
        t = tape([(0, 100, 100, 100), (1, 101, 100, 101)])
        assert t.bracket(0, 100.0, 0) == ("neither", None)


# ------------------------------------------------------- 7. the cap

class TestCap:

    def test_a_touch_after_the_cap_does_not_count(self):
        t = tape(flat(1000, 100.0) + [(1000, 101, 100, 101)])
        assert t.bracket(0, 100.0, 1, cap_s=900)[0] == "neither"

    def test_a_touch_inside_the_cap_counts(self):
        t = tape(flat(800, 100.0) + [(800, 101, 100, 101)])
        assert t.bracket(0, 100.0, 1, cap_s=900)[0] == "target"

    def test_the_cap_boundary_is_inclusive(self):
        t = tape(flat(900, 100.0) + [(900, 101, 100, 101)])
        out, secs = t.bracket(0, 100.0, 1, cap_s=900)
        assert out == "target" and secs == 900

    def test_the_excursion_respects_the_cap(self):
        t = tape(flat(1000, 100.0) + [(1000, 500, 100, 500)])
        best, _ = t.excursion(0, 100.0, 1, cap_s=900)
        assert best == 0.0, "read a high from past the cap"


# -------------------------------------------------------- 4. rolls

class TestRoll:

    def test_a_bracket_does_not_resolve_through_a_roll(self):
        """The new contract is 50 points away. Resolving through it
        would mark this a target on carry alone."""
        t = X.Tape()
        t.add(BASE, 100, 100, 100, "ESU6")
        t.add(BASE + 1, 150, 150, 150, "ESZ6")
        t.freeze()
        assert t.bracket(0, 100.0, 1)[0] == "neither"

    def test_the_excursion_stops_at_a_roll(self):
        t = X.Tape()
        t.add(BASE, 100, 100, 100, "ESU6")
        t.add(BASE + 1, 150, 150, 150, "ESZ6")
        t.freeze()
        best, worst = t.excursion(0, 100.0, 1)
        assert best == 0.0 and worst == 0.0

    def test_the_horizon_stops_at_a_roll(self):
        t = X.Tape()
        for i in range(400):
            t.add(BASE + i, 100, 100, 100, "ESU6")
        for i in range(400, 800):
            t.add(BASE + i, 150, 150, 150, "ESZ6")
        t.freeze()
        assert t.forward(0, 100.0, 1, 1) is not None     # 60s, same
        assert t.forward(0, 100.0, 1, 10) is None        # past the roll

    def test_segments_are_found_from_the_right_place(self):
        """seg_end must be the end of THIS index's run, not the file's
        first or last change."""
        t = X.Tape()
        for i in range(5):
            t.add(BASE + i, 100, 100, 100, "A")
        for i in range(5, 10):
            t.add(BASE + i, 100, 100, 100, "B")
        t.freeze()
        assert int(t.seg_end[0]) == 5
        assert int(t.seg_end[4]) == 5
        assert int(t.seg_end[5]) == 10

    def test_one_contract_has_one_segment(self):
        t = tape(flat(50, 100.0))
        assert int(t.seg_end[0]) == 50


# ----------------------------------------------------- 5. horizons

class TestForward:

    def test_the_horizon_is_the_price_at_that_moment(self):
        rows = [(i, 100 + i * 0.25, 100 + i * 0.25, 100 + i * 0.25)
                for i in range(200)]
        t = tape(rows)
        # 60 seconds after index 0: price 100 + 60*0.25 = 115
        assert t.forward(0, 100.0, 1, 1) == pytest.approx(15 / TICK)

    def test_a_quiet_stretch_uses_the_last_known_price(self):
        """No print lands exactly on the horizon. The price then is the
        last one printed -- what a screen shows. Taking the NEXT print
        instead would report a price from a minute further on."""
        t = tape([(0, 100, 100, 100), (30, 110, 110, 110),
                  (600, 999, 999, 999)])
        assert t.forward(0, 100.0, 1, 1) == pytest.approx(10 / TICK)

    def test_the_sign_follows_the_side(self):
        t = tape([(0, 100, 100, 100), (60, 90, 90, 90)])
        assert t.forward(0, 100.0, 1, 1) < 0      # long, price fell
        assert t.forward(0, 100.0, -1, 1) > 0     # short, price fell

    def test_past_the_data_has_no_value(self):
        t = tape([(0, 100, 100, 100), (5, 101, 101, 101)])
        assert t.forward(0, 100.0, 1, 15) is None

    def test_measured_in_ticks(self):
        t = tape([(0, 100, 100, 100), (60, 101, 101, 101)])
        assert t.forward(0, 100.0, 1, 1) == pytest.approx(4.0)


# ------------------------------------------------ 3 & 6. excursions

class TestExcursion:

    def test_mfe_is_never_negative_and_mae_never_positive(self):
        t = tape([(0, 100, 100, 100), (1, 103, 97, 99)])
        best, worst = t.excursion(0, 100.0, 1)
        assert best >= 0 >= worst

    def test_mfe_finds_the_high_for_a_long(self):
        t = tape([(0, 100, 100, 100), (1, 102, 100, 101),
                  (2, 101, 100, 100)])
        best, _ = t.excursion(0, 100.0, 1)
        assert best == pytest.approx(2 / TICK)

    def test_mae_finds_the_low_for_a_long(self):
        t = tape([(0, 100, 100, 100), (1, 100, 97, 98)])
        _, worst = t.excursion(0, 100.0, 1)
        assert worst == pytest.approx(-3 / TICK)

    def test_a_short_inverts_both(self):
        """For a short, price FALLING is favourable. Without the sign
        every short's risk and reward swap places.

        The range is deliberately ASYMMETRIC about the entry. With a
        symmetric one (say 103/97 about 100) the long and the short
        produce the same pair of numbers, and the short branch can be
        the long formula verbatim without any test noticing.
        """
        t = tape([(0, 100, 100, 100), (1, 110, 97, 99)])
        lb, lw = t.excursion(0, 100.0, 1)
        sb, sw = t.excursion(0, 100.0, -1)
        assert lb == pytest.approx(10 / TICK)      # long: the high
        assert lw == pytest.approx(-3 / TICK)      # long: the low
        assert sb == pytest.approx(3 / TICK)       # short: the low
        assert sw == pytest.approx(-10 / TICK)     # short: the high
        assert lb != sb, "the fixture is symmetric and proves nothing"

    def test_the_extreme_is_not_simply_the_last_bar(self):
        """The extreme sits in the MIDDLE, so `= the last value` cannot
        pass as `max(...)`."""
        t = tape([(0, 100, 100, 100), (1, 120, 80, 100),
                  (2, 101, 99, 100)])
        best, worst = t.excursion(0, 100.0, 1)
        assert best == pytest.approx(20 / TICK)
        assert worst == pytest.approx(-20 / TICK)

    def test_a_flat_side_has_no_excursion(self):
        t = tape([(0, 100, 100, 100), (1, 110, 90, 100)])
        assert t.excursion(0, 100.0, 0) == (None, None)


# ------------------------------------------------- rows end to end

class TestMinuteRows:

    def candle(self, rows=None):
        import candle_walk as W
        rows = rows or [(i, 100 + i, 101 + i, 99 + i, 101 + i)
                        for i in range(15)]
        b15, b5, b1 = C.Bars(900), C.Bars(300), C.Bars(60)
        for off, o, h, lo, c in rows:
            for b in (b15, b5, b1):
                b.feed(BASE + off * 60, o, h, lo, c, 10, "ESU6")
        return W.group_candles(b15.finish(), b5.finish(),
                               b1.finish())[0]

    def full_tape(self):
        t = X.Tape()
        for i in range(60 * 40):
            px = 100 + i / 60.0
            t.add(BASE + i, px + 0.25, px - 0.25, px, "ESU6")
        return t.freeze()

    def test_one_row_per_minute(self):
        rows = X.minute_rows(self.candle(), self.full_tape())
        assert len(rows) == 15

    def test_every_declared_column_is_present(self):
        rows = X.minute_rows(self.candle(), self.full_tape())
        for r in rows:
            assert set(X.MINUTE_COLS) <= set(r)

    def test_idx_is_the_position_inside_the_candle(self):
        rows = X.minute_rows(self.candle(), self.full_tape())
        assert [r["idx"] for r in rows] == list(range(15))

    def test_idx_zero_is_the_candles_first_minute(self):
        """A sequence 0..14 is still 0..14 when the minutes under it
        run backwards, so the numbering has to be tied to the clock."""
        rows = X.minute_rows(self.candle(), self.full_tape())
        assert rows[0]["minute"] == rows[0]["candle"]
        assert [r["minute"] for r in rows] == sorted(
            r["minute"] for r in rows)

    def test_the_utc_stamps_ascend_with_idx(self):
        rows = X.minute_rows(self.candle(), self.full_tape())
        stamps = [r["utc"] for r in rows]
        assert stamps == sorted(stamps)

    def test_the_entry_is_after_the_minute_not_its_close(self):
        """Asserted on the TIMING, not on the price.

        Comparing `entry` to `close` only works when the two fixtures
        happen to disagree numerically; two unrelated synthetic series
        can coincide and the test passes while filling at the close.
        The lag is the property that actually defines the fill.
        """
        rows = X.minute_rows(self.candle(), self.full_tape())
        filled = [r for r in rows if r["entry_lag_s"] != ""]
        assert filled, "the fixture produced no fills at all"
        for r in filled:
            assert float(r["entry_lag_s"]) >= 1, (
                "filled at or before the minute's own last print")

    def test_a_flat_minute_has_no_forward_columns(self):
        c = self.candle([(i, 100, 100, 100, 100) for i in range(15)])
        rows = X.minute_rows(c, self.full_tape())
        assert all(r["d1"] == 0 for r in rows)
        assert all(r["fwd_1"] == "" for r in rows)
        assert all(r["first_4t"] == "neither" for r in rows)

    def test_to_close_is_still_written(self):
        """It is on the pages, so it is in the file -- with the warning
        in the header, not by being silently dropped."""
        rows = X.minute_rows(self.candle(), self.full_tape())
        assert rows[0]["to_close"] != ""

    def test_candle_row_carries_the_rollup_result(self):
        c = self.candle()
        assert X.candle_row(c)["rollup_ok"] == 1

    def test_candle_row_reports_a_broken_rollup(self):
        c = self.candle()
        s, o, h, lo, cl, v, k = c.bar15
        c.bar15 = (s, o + 9, h, lo, cl, v, k)
        assert X.candle_row(c)["rollup_ok"] == 0

    def test_candle_row_has_every_declared_column(self):
        assert set(X.CANDLE_COLS) <= set(X.candle_row(self.candle()))


# ----------------------------------------------------- no order path

class TestSuggestsOnly:

    def test_the_script_cannot_place_an_order(self):
        src = (Path(__file__).resolve().parent.parent
               / "scripts" / "walk_export.py").read_text()
        for bad in ("def submit", "def execute", "def place", "def order",
                    "requests.post", "urllib.request"):
            assert bad not in src, f"walk_export.py contains {bad!r}"


class TestHorizonNeedsData:
    """The tail-of-session defect, pinned.

    "The last print at or before the horizon" is correct only while
    the tape continues past it. Where the session ends first, that
    price is not the price N minutes later -- and without this guard
    the last fifteen minutes of all 517 sessions get fwd_15 values of
    roughly zero, which is a fabricated "price did not move".
    """

    def test_a_horizon_beyond_the_tape_is_unknown(self):
        t = tape([(0, 100, 100, 100), (5, 101, 101, 101)])
        assert t.forward(0, 100.0, 1, 15) is None

    def test_a_horizon_inside_the_tape_is_known(self):
        t = tape(flat(1000, 100.0))
        assert t.forward(0, 100.0, 1, 15) is not None

    def test_the_boundary_is_the_horizon_itself(self):
        exact = tape(flat(61, 100.0))             # reaches 60s
        short = tape(flat(60, 100.0))             # reaches 59s
        assert exact.forward(0, 100.0, 1, 1) is not None
        assert short.forward(0, 100.0, 1, 1) is None

    def test_a_gap_before_the_horizon_still_resolves(self):
        """No print lands ON the horizon, but the tape continues past
        it, so the last known price is the right answer."""
        t = tape([(0, 100, 100, 100), (30, 110, 110, 110),
                  (120, 120, 120, 120)])
        assert t.forward(0, 100.0, 1, 1) == pytest.approx(10 / TICK)
