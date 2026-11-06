"""Direction that carries, instead of restarting every five minutes.

Chris, after 109 hours said his rule was a coin flip: "the direction
is a continuation and not resetting at ever 5 minute interval".

The rule measured the 5-minute direction as close against open of
the CURRENT candle, so the anchor was discarded every 300 seconds. A
twenty-minute move was re-measured from scratch at each boundary; in
the first seconds of a candle price sits on its open and one tick
decided the direction; and a single pullback candle read as a
reversal because it was the only reference.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from liqmap.trend import MODES, Trend

ROOT = Path(__file__).resolve().parents[1]


def walk(t: Trend, bars) -> list:
    """Feed closed candles, reading the direction before each close."""
    seen = []
    for o, c in bars:
        t.on_price(c, o)
        seen.append(t.direction)
        t.on_close(o, c)
    return seen


UP3_PULLBACK_UP = [(100.0, 101.0), (101.0, 102.0), (102.0, 103.0),
                   (103.0, 102.5), (102.5, 104.0)]


class TestItDoesNotTrade:
    def test_no_order_entry_points(self):
        tree = ast.parse((ROOT / "liqmap" / "trend.py").read_text())
        names = {n.name for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for banned in ("submit", "execute", "place", "order", "send_order"):
            assert banned not in names, banned

    def test_it_imports_nothing(self):
        tree = ast.parse((ROOT / "liqmap" / "trend.py").read_text())
        mods = {n.module.split(".")[0] for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and n.module}
        assert mods <= {"__future__"}, mods


class TestTheControl:
    """BAR is what the rule does now, kept so the others have
    something to be measured against."""

    def test_it_is_this_candle_against_itself(self):
        t = Trend("bar")
        t.on_price(101.0, 100.0)
        assert t.direction == "up"
        t.on_price(99.0, 100.0)
        assert t.direction == "down"

    def test_a_pullback_candle_reads_as_a_reversal(self):
        """The behaviour under suspicion, pinned so the comparison
        is honest about what it is comparing against."""
        assert walk(Trend("bar"), UP3_PULLBACK_UP)[3] == "down"

    def test_it_has_no_memory_at_all(self):
        a = Trend("bar")
        walk(a, UP3_PULLBACK_UP)
        b = Trend("bar")
        # The same forming candle `a` is sitting in, and nothing
        # before it. Feeding `b` a different candle would prove
        # only that different inputs give different answers.
        b.on_price(104.0, 102.5)
        assert a.direction == b.direction, \
            "four candles of history changed nothing"


class TestContinuationHoldsThroughAPullback:
    @pytest.mark.parametrize("mode", ["closed", "anchor", "run"])
    def test_the_pullback_does_not_flip_it(self, mode):
        assert walk(Trend(mode), UP3_PULLBACK_UP)[3] == "up"

    def test_anchor_measures_from_where_the_move_began(self):
        """Not from the current candle's open. That is the literal
        reading of "not resetting at every 5 minute interval"."""
        t = Trend("anchor")
        walk(t, UP3_PULLBACK_UP)
        assert t.anchor == 100.0, "the anchor moved with the candles"

    def test_the_anchor_moves_only_when_the_MOVE_turns(self):
        """Not when a candle does.

        The anchor used to move on any direction-changing close, so
        one pullback candle reset it -- most of the way back to the
        per-bar anchor this mode exists to replace. It now hangs
        off the run, so it holds until the run turns.
        """
        t = Trend("anchor", flip_bars=2)
        walk(t, [(100.0, 101.0), (101.0, 102.0)])
        assert t.anchor == 100.0

        walk(t, [(102.0, 98.0)])         # one down candle
        assert t.anchor == 100.0, "one candle moved the anchor"

        walk(t, [(98.0, 97.0)])          # the second completes the turn
        assert t.anchor == 98.0, "the anchor did not follow the turn"
        assert t.direction == "down"

    def test_closed_lags_by_exactly_one_candle(self):
        seen = walk(Trend("closed"), UP3_PULLBACK_UP)
        assert seen[0] == "flat", "nothing has closed yet"
        assert seen[1] == "up"


class TestRunHasItsOwnMemory:
    """IT DID NOT, AND THAT WAS A BUG.

    RUN shared `_anchor_dir` with ANCHOR, and ANCHOR moved that on
    the FIRST disagreeing close -- so the counter RUN was supposed
    to be protected by reset at the same instant, and RUN flipped
    on one pullback candle exactly like the mode it exists to
    improve on.

    The fix was not to split them further. ANCHOR had the same
    defect for the same reason, so the anchor was hung off the run
    and they now turn TOGETHER, deliberately, on the same counter.
    `_run_dir` stays a separate field because `_anchor_dir` records
    which turn placed the anchor, and the tests below hold the
    counter itself to its promise.
    """

    def test_one_pullback_does_not_turn_it(self):
        bars = [(100.0, 101.0), (101.0, 102.0), (102.0, 103.0),
                (103.0, 102.0), (102.0, 103.0)]
        t = Trend("run", flip_bars=2)
        seen = walk(t, bars)
        assert seen[4] == "up", "one pullback flipped the run"

    def test_two_against_it_do(self):
        bars = [(100.0, 101.0), (101.0, 102.0),
                (102.0, 101.0), (101.0, 100.0), (100.0, 99.0)]
        t = Trend("run", flip_bars=2)
        assert walk(t, bars)[4] == "down"

    def test_the_counter_resets_on_an_agreeing_close(self):
        bars = [(100.0, 101.0), (101.0, 100.5),   # one against
                (100.5, 102.0),                    # agrees, resets
                (102.0, 101.0)]                    # one against again
        t = Trend("run", flip_bars=2)
        assert walk(t, bars)[3] == "up"

    def test_flip_bars_of_one_is_the_closed_mode(self):
        bars = UP3_PULLBACK_UP
        a = walk(Trend("run", flip_bars=1), bars)
        b = walk(Trend("closed"), bars)
        assert a[1:] == b[1:]

    def test_the_anchor_turns_with_the_run_and_not_before(self):
        t = Trend("run", flip_bars=3)
        walk(t, [(100.0, 101.0), (101.0, 100.0)])
        d = t.to_dict()
        assert d["run_dir"] == "up", "one against turned the run"
        assert d["anchor_dir"] == "up", "the anchor ran ahead of the run"
        assert d["against"] == 1
        assert t.anchor == 100.0

        walk(t, [(100.0, 99.0), (99.0, 98.0)])   # two more against
        d = t.to_dict()
        assert d["run_dir"] == "down"
        assert d["anchor_dir"] == "down"
        assert t.anchor == 99.0, "the anchor sits where the turn completed"


class TestTheEdges:
    def test_an_unknown_mode_is_refused_at_construction(self):
        with pytest.raises(ValueError):
            Trend("vibes")

    def test_every_mode_starts_flat_or_bar_like(self):
        for m in MODES:
            t = Trend(m)
            assert t.direction in ("flat", "up", "down")

    def test_a_flat_candle_establishes_nothing(self):
        t = Trend("anchor")
        t.on_price(100.0, 100.0)
        t.on_close(100.0, 100.0)
        assert t.to_dict()["anchor_dir"] == ""

    def test_garbage_prices_do_not_raise(self):
        t = Trend("anchor")
        t.on_price("x", None)
        t.on_close(0.0, 0.0)
        assert t.direction in ("flat", "up", "down")

    def test_flips_are_counted(self):
        t = Trend("closed")
        walk(t, [(100.0, 101.0), (101.0, 100.0), (100.0, 101.0)])
        assert t.flips == 2
        assert t.closes == 3


class TestTheComparisonScript:
    def test_it_changes_only_the_five_minute_leg(self):
        """The 1-minute leg and the exit are untouched, so any
        difference is attributable to the direction and nothing
        else."""
        src = (ROOT / "scripts" / "trend_compare.py").read_text()
        assert "tm.entry_side(d1, d5)" in src
        assert "tm.reversed_against(side_s, d1)" in src
        assert 'tm.direction({"open": b1o, "close": b1c})' in src
        assert "d5 = tr.direction" in src

    def test_gross_is_the_headline(self):
        """Any change that trades less improves NET on a tape with
        no edge, because the round trip is the only drain."""
        src = (ROOT / "scripts" / "trend_compare.py").read_text()
        assert "GROSS IS THE HEADLINE" in src
        assert "flatters every variant that" in src

    def test_it_judges_on_the_holdout(self):
        src = (ROOT / "scripts" / "trend_compare.py").read_text()
        assert 'h["gpt"] > best[2]["gpt"]' in src
        assert 'f["trades"] >= 30 and h["trades"] >= 10' in src

    def test_it_can_say_continuation_is_not_the_answer(self):
        src = (ROOT / "scripts" / "trend_compare.py").read_text()
        assert "Continuation is not the " in src
