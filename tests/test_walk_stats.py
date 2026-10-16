"""The specific ways a win-rate table can lie.

This is the script that decides whether a strategy gets built, so the
arithmetic has to be the arithmetic:

1.  THE DENOMINATOR. `win%` is target/(target+stop). Put `neither` in
    it and every rate drops toward zero; put `ambiguous` in it and a
    second that covered both levels gets counted as a loss (or a win)
    that the data cannot support. Either way the number stops meaning
    what its column header says.

2.  EXPECTANCY DROPPING THE UNRESOLVED. A `neither` row is a position
    that was still open at the cap, worth `fwd_15`. Scoring it zero,
    or leaving it out, flatters the result -- unresolved rows cluster
    in the quiet stretches, so they are not a random sample.

3.  THE BREAKEVEN LINE. (stop + cost)/(target + stop). Get it wrong
    and a losing bracket reads as a winning one: at 4 and 4 the bar is
    55% at 0.4 ticks, not 50%.

4.  THE AGREEING COUNT. Counted against the MINUTE's direction. A row
    where the 5m and 15m agree with each other but not with the minute
    is 1 of 3, not 2 -- the minute is what gets entered on.

5.  THE WALK-FORWARD SPLIT. Split on the DATE. Halving the rows puts
    part of one session in both arms, and the two halves stop being
    independent.

Nothing here asserts that any bucket is profitable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

pd = pytest.importorskip("pandas")

import walk_stats as S                                      # noqa: E402


def frame(rows):
    """rows: (first_4t, fwd_15) or (first_4t, fwd_15, d1, d5, d15)."""
    out = []
    for r in rows:
        first, fwd = r[0], r[1]
        d1, d5, d15 = (r[2:5] if len(r) >= 5 else (1, 1, 1))
        out.append({"first_4t": first, "fwd_15": fwd, "d1": d1,
                    "d5": d5, "d15": d15})
    return pd.DataFrame(out)


# ------------------------------------------------- 3. the breakeven

class TestBreakeven:

    def test_a_symmetric_bracket_is_a_coin_before_costs(self):
        assert S.breakeven(4, 4, 0.0) == pytest.approx(0.50)

    def test_cost_raises_the_bar(self):
        assert S.breakeven(4, 4, 0.4) == pytest.approx(0.55)
        assert S.breakeven(4, 4, 1.0) == pytest.approx(0.625)
        assert S.breakeven(4, 4, 1.5) == pytest.approx(0.6875)

    def test_cost_is_added_to_the_stop_not_the_target(self):
        """(stop + cost)/(target + stop). Adding it to the denominator
        instead LOWERS the bar, which is the wrong direction."""
        assert S.breakeven(4, 4, 1.0) > S.breakeven(4, 4, 0.0)

    def test_a_wider_target_lowers_the_bar(self):
        assert S.breakeven(8, 4, 0.0) < S.breakeven(4, 4, 0.0)

    def test_a_wider_stop_raises_it(self):
        assert S.breakeven(4, 8, 0.0) > S.breakeven(4, 4, 0.0)


# ------------------------------------------------ 1. the denominator

class TestWinRate:

    def test_only_resolved_rows_are_counted(self):
        c = {"target": 3, "stop": 1, "neither": 96, "ambiguous": 0}
        assert S.win_rate(c) == pytest.approx(0.75)

    def test_neither_does_not_dilute_the_rate(self):
        a = S.win_rate({"target": 3, "stop": 1, "neither": 0})
        b = S.win_rate({"target": 3, "stop": 1, "neither": 1000})
        assert a == b

    def test_ambiguous_is_not_a_loss(self):
        a = S.win_rate({"target": 5, "stop": 5, "ambiguous": 0})
        b = S.win_rate({"target": 5, "stop": 5, "ambiguous": 500})
        assert a == b == pytest.approx(0.5)

    def test_nothing_resolved_has_no_rate(self):
        assert S.win_rate({"neither": 10, "ambiguous": 2}) is None

    def test_an_all_target_bucket_is_one(self):
        assert S.win_rate({"target": 7, "stop": 0}) == 1.0

    def test_an_all_stop_bucket_is_zero(self):
        assert S.win_rate({"target": 0, "stop": 7}) == 0.0


# ------------------------------------------------- 2. the expectancy

class TestExpectancy:

    def test_a_resolved_win_is_the_target(self):
        f = frame([("target", 0.0)])
        assert S.expectancy(f, 0.0) == pytest.approx(4.0)

    def test_a_resolved_loss_is_the_stop(self):
        f = frame([("stop", 0.0)])
        assert S.expectancy(f, 0.0) == pytest.approx(-4.0)

    def test_an_unresolved_row_is_worth_its_mark(self):
        """Not zero. The position was open and had a value."""
        f = frame([("neither", -6.0)])
        assert S.expectancy(f, 0.0) == pytest.approx(-6.0)

    def test_unresolved_rows_are_not_dropped(self):
        """Dropping them flatters the result: they are not a random
        sample, they cluster in the quiet stretches."""
        kept = S.expectancy(frame([("target", 0.0),
                                   ("neither", -8.0)]), 0.0)
        dropped = S.expectancy(frame([("target", 0.0)]), 0.0)
        assert kept != dropped
        assert kept == pytest.approx((4.0 - 8.0) / 2)

    def test_ambiguous_rows_are_marked_not_guessed(self):
        """An ambiguous row is not scored +4 or -4; it is carried at
        its fwd_15 like any other unresolved row."""
        f = frame([("ambiguous", 2.0)])
        assert S.expectancy(f, 0.0) == pytest.approx(2.0)

    def test_cost_comes_off_every_row(self):
        f = frame([("target", 0.0), ("stop", 0.0)])
        assert S.expectancy(f, 0.0) == pytest.approx(0.0)
        assert S.expectancy(f, 1.0) == pytest.approx(-1.0)

    def test_a_missing_mark_is_not_an_error(self):
        f = frame([("neither", float("nan"))])
        assert S.expectancy(f, 0.0) == pytest.approx(0.0)

    def test_an_empty_bucket_has_no_expectancy(self):
        assert S.expectancy(frame([]), 0.0) is None

    def test_a_coin_flip_bracket_loses_exactly_the_cost(self):
        f = frame([("target", 0.0)] * 50 + [("stop", 0.0)] * 50)
        assert S.expectancy(f, 1.0) == pytest.approx(-1.0)


# ------------------------------------------------ 4. agreeing count

class TestAgreeingCount:

    def test_all_three_up_is_three(self):
        f = frame([("neither", 0.0, 1, 1, 1)])
        assert int(S.agreeing_count(f).iloc[0]) == 3

    def test_all_three_down_is_three(self):
        f = frame([("neither", 0.0, -1, -1, -1)])
        assert int(S.agreeing_count(f).iloc[0]) == 3

    def test_counted_against_the_minute_not_the_majority(self):
        """The 5m and 15m agree with each other and not with the
        minute. The minute is what gets entered on, so this is 1."""
        f = frame([("neither", 0.0, 1, -1, -1)])
        assert int(S.agreeing_count(f).iloc[0]) == 1

    def test_two_of_three(self):
        f = frame([("neither", 0.0, 1, 1, -1)])
        assert int(S.agreeing_count(f).iloc[0]) == 2

    def test_a_flat_timeframe_does_not_count(self):
        f = frame([("neither", 0.0, 1, 1, 0)])
        assert int(S.agreeing_count(f).iloc[0]) == 2

    def test_a_flat_minute_scores_zero(self):
        """There is nothing to enter on, so nothing agrees with it."""
        f = frame([("neither", 0.0, 0, 0, 0)])
        assert int(S.agreeing_count(f).iloc[0]) == 0

    def test_a_flat_minute_with_two_trending_frames_is_still_zero(self):
        f = frame([("neither", 0.0, 0, 1, 1)])
        assert int(S.agreeing_count(f).iloc[0]) == 0


# ------------------------------------------------------ the census

class TestCensus:

    def test_every_outcome_is_present_even_at_zero(self):
        c = S.census(frame([("target", 0.0)]))
        assert set(c) == {"target", "stop", "neither", "ambiguous"}
        assert c["stop"] == 0

    def test_counts_are_counts(self):
        f = frame([("target", 0.0)] * 3 + [("stop", 0.0)] * 2
                  + [("neither", 0.0)])
        c = S.census(f)
        assert (c["target"], c["stop"], c["neither"]) == (3, 2, 1)

    def test_the_line_reports_the_unresolved_share(self):
        f = frame([("target", 0.0)] * 1 + [("neither", 0.0)] * 3)
        out = S.line("x", f)
        assert "75.0%" in out

    def test_an_empty_bucket_says_so(self):
        assert "no rows" in S.line("x", frame([]))


# ----------------------------------------------------- no order path

class TestSuggestsOnly:

    def test_the_script_cannot_place_an_order(self):
        src = (Path(__file__).resolve().parent.parent
               / "scripts" / "walk_stats.py").read_text()
        for bad in ("def submit", "def execute", "def place", "def order",
                    "requests.post", "urllib.request"):
            assert bad not in src, f"walk_stats.py contains {bad!r}"
