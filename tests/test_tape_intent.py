"""Pullbacks and continuations, off the tape, inside the tick bar.

Chris: "down at the bottom of the tick bar there is the tape what we
are going to do with that is show tape intent at that moment during
the tick bar kinda the same way a 1 minute shows the 5 minute candles
intent".

Asked which measurement it should be -- who is crossing the spread,
where the recent slice went, or both with the disagreement flagged --
he answered the question behind the question: "which ever one will
show tick chart pullbacks and continuations thats what i actually
need".

SO THE TEST OF THIS READING IS WHETHER IT NAMES THOSE TWO STATES, and
that is what this file checks. Price, not aggression: a pullback on a
tick chart IS a price event, and buy-against-sell volume can only
imply one. The aggression disagreement is absorption, which this
engine already measures and already labels as deciding nothing.

WHAT THE LINE SHOWED BEFORE. Its projected span against a median --
how FAST the bar was forming. That is a real measurement and it is
kept, as the tail of the line, because a reading that silently stops
being collected is how a screen starts lying. It is not intent: a bar
can quicken while price gives the whole move back, and those two
sentences describe opposite trades.

TWO NUMBERS, BOTH NEEDED.

    DEPTH alone cannot tell a pullback that is still running from one
    that has already turned -- the first says wait, the second says
    the continuation has resumed.

    THE SLICE alone cannot say what the pullback has cost. One tick
    off the high and half the move given back are the same slice
    direction and very different markets.
"""

from __future__ import annotations

import pathlib

import pytest

from liqmap import tapeintent as ti
from liqmap.tickbar import TickBars

WEB = (pathlib.Path(__file__).resolve().parents[1]
       / "liqmap" / "web.py").read_text()


class TestItNamesTheTwoStates:
    def test_the_slice_going_with_the_bar_is_a_continuation(self):
        r = ti.read("up", 7000, 7010, 6999, 7010, "up", 1000)
        assert r["state"] == "continuation"

    def test_the_slice_going_against_it_is_a_pullback(self):
        r = ti.read("up", 7000, 7010, 6999, 7006, "down", 1000)
        assert r["state"] == "pullback"

    def test_it_is_symmetric_on_a_down_bar(self):
        assert ti.read("down", 7000, 7001, 6990, 6990,
                       "down", 1000)["state"] == "continuation"
        assert ti.read("down", 7000, 7001, 6990, 6994,
                       "up", 1000)["state"] == "pullback"

    def test_a_pullback_is_called_early_not_once_it_is_deep(self):
        """One tick off the high with the tape still selling is a
        pullback that has just started. Waiting for depth before
        naming it is what makes the reading useless -- by then it is
        on the screen anyway."""
        r = ti.read("up", 7000, 7010, 6999, 7009.75, "down", 1000)
        assert r["state"] == "pullback"
        assert r["depth"] < 0.05

    def test_a_resumed_continuation_is_not_called_a_pullback(self):
        """THE CASE DEPTH ALONE CANNOT SEE. Price is still well off
        the high, so depth says a pullback happened -- but the tape
        has turned back up, which is the continuation resuming and
        the opposite trade."""
        r = ti.read("up", 7000, 7010, 6999, 7006, "up", 1000)
        assert r["state"] == "continuation"
        assert r["depth"] == 0.4
        assert "turned back" in r["say"]


class TestHowDeepThePullbackIs:
    @pytest.mark.parametrize("close,want", [
        (7010.0, 0.0),      # at the high
        (7007.5, 0.25),
        (7005.0, 0.5),
        (7000.0, 1.0),      # the whole move given back
    ])
    def test_depth_is_the_fraction_of_the_move_given_back(self, close,
                                                          want):
        assert ti.depth("up", 7000, 7010, 6990, close) == want

    def test_it_measures_the_move_made_not_the_whole_range(self):
        """On an up bar the denominator is high-minus-open. Using the
        full high-to-low would mix in a dip that happened before the
        move and answer a different question."""
        shallow = ti.depth("up", 7000, 7010, 6999, 7005)
        deep = ti.depth("up", 7000, 7010, 6900, 7005)
        assert shallow == deep == 0.5, "the low leaked into the maths"

    def test_it_is_symmetric_downwards(self):
        assert ti.depth("down", 7000, 7001, 6990, 6990) == 0.0
        assert ti.depth("down", 7000, 7001, 6990, 6995) == 0.5
        assert ti.depth("down", 7000, 7001, 6990, 7000) == 1.0

    def test_it_never_leaves_nought_to_one(self):
        assert ti.depth("up", 7000, 7010, 6990, 7020) == 0.0
        assert ti.depth("up", 7000, 7010, 6990, 6980) == 1.0

    def test_a_bar_with_no_move_is_not_a_retrace(self):
        """A flat bar dividing by zero would make every reading
        either a crash or a 100% pullback."""
        assert ti.depth("up", 7000, 7000, 7000, 7000) == 0.0
        assert ti.depth("", 7000, 7010, 6990, 7005) == 0.0

    @pytest.mark.parametrize("junk", [None, "x", object()])
    def test_rubbish_prices_do_not_raise(self, junk):
        assert ti.depth("up", junk, junk, junk, junk) == 0.0


class TestItNeverBlanks:
    """The fault that took two goes to get out of the gates
    themselves. A line that blinks out gets read as an absence."""

    def test_a_tie_holds_the_last_state(self):
        prev = {"state": "pullback", "slice_dir": "down"}
        r = ti.read("", 7000, 7000, 7000, 7000, "", 1000, prev=prev)
        assert r["state"] == "pullback"
        assert r["slice_dir"] == "down"

    def test_a_flat_slice_holds_its_direction(self):
        r = ti.read("up", 7000, 7010, 6999, 7006, "", 1000,
                    prev={"state": "pullback", "slice_dir": "down"})
        assert r["slice_dir"] == "down"

    def test_nothing_is_invented_at_the_start(self):
        r = ti.read("", 0, 0, 0, 0, "", 0)
        assert r["state"] == "" and r["slice_dir"] == ""
        assert "no direction" in r["say"]

    def test_a_real_change_passes_straight_through(self):
        r = ti.read("up", 7000, 7010, 6999, 7010, "up", 1000,
                    prev={"state": "pullback", "slice_dir": "down"})
        assert r["state"] == "continuation"
        assert r["slice_dir"] == "up"


class TestTheSliceIsAFifthOfTheBar:
    """Chris's own analogy: "the same way a 1 minute shows the 5
    minute candles intent". A fifth, so it means the same thing at
    every bar size and nothing has to be retuned."""

    def test_it_scales_with_the_bar(self):
        assert ti.slice_prints(5000) == 1000
        assert ti.slice_prints(2000) == 400
        assert ti.slice_prints(10000) == 2000

    def test_a_small_bar_gets_the_floor(self):
        assert ti.slice_prints(50) == ti.MIN_SLICE
        assert ti.slice_prints(0) == ti.MIN_SLICE

    def test_the_builder_sizes_its_slice_to_itself(self):
        assert TickBars(per_bar=5000)._slice.maxlen == 1000
        assert TickBars(per_bar=2000)._slice.maxlen == 400


class TestTheSliceComesOffRealPrints:
    def test_it_reads_close_against_open_on_the_slice(self):
        b = TickBars(per_bar=100)       # slice of 20
        t, px = 0.0, 7000.0
        for _ in range(60):             # drive it up
            t += 1.0
            px += 0.25
            b.on_trade(t, px, 1.0, "B")
        s = b.slice_read()
        assert s["dir"] == "up" and s["n"] == 20
        assert s["close"] > s["open"]
        assert s["ticks"] == 19.0

    def test_the_slice_turns_before_the_bar_does(self):
        """THE WHOLE POINT. The bar is still up on the session while
        the last twenty prints have turned down -- which is a
        pullback, and is invisible to anything reading the bar
        alone."""
        # The slice is sized FROM `per_bar`, so the down leg has to
        # be longer than the slice or the window still spans the up
        # move -- which is the whole reason a fifth is a fifth.
        b = TickBars(per_bar=1000)      # slice of 200
        t, px = 0.0, 7000.0
        for _ in range(400):
            t += 1.0
            px += 0.25
            b.on_trade(t, px, 1.0, "B")
        for _ in range(250):
            t += 1.0
            px -= 0.25
            b.on_trade(t, px, 1.0, "B")
        cur = b.current
        assert cur.direction == "up", "the bar is no longer up"
        assert b.slice_read()["dir"] == "down"
        r = ti.read(cur.direction, cur.open, cur.high, cur.low,
                    cur.close, b.slice_read()["dir"])
        assert r["state"] == "pullback"
        assert 0.0 < r["depth"] < 1.0

    def test_the_slice_does_not_restart_at_a_bar_boundary(self):
        """The same rule the gate's direction follows. A slice that
        emptied at every close would blink the line out once a bar,
        which is the fault this panel has already had twice."""
        b = TickBars(per_bar=10)        # slice floors at MIN_SLICE
        t, px = 0.0, 7000.0
        for _ in range(205):
            t += 1.0
            px += 0.25
            b.on_trade(t, px, 1.0, "B")
        assert b.current.trades == 5, "not mid-bar; the setup moved"
        s = b.slice_read()
        assert s["n"] == ti.MIN_SLICE, "the slice emptied at a close"
        assert s["dir"] == "up"

    def test_too_few_prints_is_not_a_direction(self):
        b = TickBars(per_bar=100)
        b.on_trade(1.0, 7000.0, 1.0, "B")
        assert b.slice_read()["dir"] == ""


class TestItReachesThePanel:
    def test_the_route_builds_the_reading(self):
        i = WEB.index("def _tape_intent(")
        blk = WEB[i:WEB.index("def _m5_now(", i)]
        assert "_ti.read(" in blk
        assert 'tk_ or {}).get("body_dir")' in blk, \
            "it is not reading the bar's own direction"
        assert 'sl.get("dir")' in blk

    def test_it_uses_the_bars_own_direction_not_the_held_one(self):
        """THE GEOMETRY HAS TO MATCH THE PRICES. `depth` measures
        against high-minus-open on an up bar, so a direction the bar
        does not currently have would measure a retrace off the
        wrong extreme. The hold happens on the VERDICT instead."""
        i = WEB.index("def _tape_intent(")
        blk = WEB[i:WEB.index("def _m5_now(", i)]
        assert "_bd" not in blk.split('_ti.read(')[1][:200], \
            "the gate's held direction was fed to the geometry"

    def test_the_tick_route_carries_the_slice_and_the_extremes(self):
        i = WEB.index("def _tick_now(")
        blk = WEB[i:WEB.index("def _tape_intent(", i)]
        assert "bld.slice_read()" in blk
        main = blk[blk.index('"up_prints": cur.up_prints'):]
        assert '"slice": sl' in main
        assert '"high": cur.high' in main, \
            "no extreme to measure the retrace against"

    def test_the_state_carries_it(self):
        from liqmap import stackwin as sw
        st = sw.state({}, "up", {}, 300.0,
                      intent={"state": "pullback", "depth": 0.4})
        assert st["intent"]["state"] == "pullback"
        assert st["intent"]["decides"] is False

    def test_its_memory_rides_home_with_the_gates(self):
        from liqmap import stackwin as sw
        st = sw.state({}, "up", {}, 300.0,
                      intent={"state": "pullback", "slice_dir": "down"})
        assert st["held"]["intent"] == {"state": "pullback",
                                        "slice_dir": "down"}

    def test_the_route_feeds_the_memory_back_in(self):
        i = WEB.index("def _journal_step_un(")
        blk = WEB[i:i + 30000]
        assert "intent=_tape_intent(_tk, _prev)" in blk
        j = WEB.index("def _tape_intent(")
        assert 'prev_ or {}).get("intent")' in WEB[j:j + 1600]

    def test_the_panel_draws_the_two_states(self):
        assert "'PULLBACK'" in WEB and "'CONTINUATION'" in WEB
        assert "SW.intent && SW.intent.state" in WEB

    def test_the_panel_keeps_the_speed_as_a_tail(self):
        """It is not deleted. It is just no longer the headline."""
        i = WEB.index("SW.intent && SW.intent.state")
        blk = WEB[i:i + 1600]
        assert "esc(SW.pace.push)" in blk, "the speed reading was dropped"
        assert "SW.pace.ratio" in blk

    def test_the_panel_says_it_decides_nothing(self):
        i = WEB.index("SW.intent && SW.intent.state")
        assert "decides nothing" in WEB[i:i + 1600]


class TestItDoesNotDecideAnything:
    def test_the_entry_ignores_it_entirely(self):
        from liqmap import stackwin as sw
        r = {"5m": {"present": True, "qualified": "up"}}
        a = sw.state({}, "up", r, 300.0, held_s=99.0,
                     intent={"state": "pullback", "depth": 0.9})
        b = sw.state({}, "up", r, 300.0, held_s=99.0,
                     intent={"state": "continuation", "depth": 0.0})
        assert a["side"] == b["side"] == "up"
        assert a["why"] == b["why"]

    def test_entry_side_cannot_see_it(self):
        import inspect

        from liqmap import stackwin as sw
        assert "intent" not in inspect.signature(sw.entry_side).parameters
        assert "intent" not in inspect.signature(sw.holds).parameters

    def test_nothing_here_submits_an_order(self):
        import ast
        tree = ast.parse(pathlib.Path(ti.__file__).read_text())
        bad = [n.name for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name in ("submit", "execute", "place", "order")]
        assert not bad, f"tapeintent.py grew {bad}"
