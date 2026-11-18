"""Pullbacks and continuations, off the tape, inside the tick bar.

Chris: "down at the bottom of the tick bar there is the tape what we
are going to do with that is show tape intent at that moment during
the tick bar kinda the same way a 1 minute shows the 5 minute candles
intent" -- and, on which measurement, "which ever one will show tick
chart pullbacks and continuations thats what i actually need".

THE FIRST VERSION LAGGED, AND THE LAG WAS STRUCTURAL.

    It compared price against a point 1,000 prints back -- the last
    fifth of a bar's worth, the 1m-inside-a-5m ratio. A window like
    that can only report a turn once the turn is BIGGER THAN THE
    WINDOW'S OWN PRIOR MOVE, so a pullback stayed invisible until
    it had undone everything the window had seen.

    Measured on a tape that pushed up 1,200 prints and then
    reversed: 200 ticks -- fifty points -- off the high and it
    still read CONTINUATION. It did not flip until 42% of the move
    was already gone.

    Chris: "price and tick bar are obviously pulling back but it
    shows a continuation like its making its determination after
    the move has already happened". Exactly right, and no tuning of
    the window would have fixed it -- a shorter one lags less and
    still lags, and a very short one is noise.

THE REFERENCE IS NOW THE BAR'S OWN EXTREME.

    Price off the high of an up bar is a retrace THE INSTANT IT
    HAPPENS, because the high is a fact about this bar and not a
    moving point in the tape. One tick off reads PULLBACK with a
    depth of 1%, which is honest in both directions: it is a
    pullback, and it is a trivial one, and the number says so.

    Depth alone still cannot tell a pullback that is running from
    one that has turned, so the deepest retrace since the extreme
    is carried in the caller's memory. RECOVERING is a qualifier on
    the pullback, not a state of its own: it stays a pullback until
    a new extreme is made, because calling it a continuation before
    that would be predicting the resumption rather than reporting
    it -- the same mistake in the other direction.

PRICE, NOT AGGRESSION. A pullback on a tick chart IS a price event;
buy-against-sell volume can only imply one. Their disagreement is
absorption, which this engine already measures and already labels as
deciding nothing.
"""

from __future__ import annotations

import pathlib

import pytest

from liqmap import tapeintent as ti
from liqmap.tickbar import TickBars

WEB = (pathlib.Path(__file__).resolve().parents[1]
       / "liqmap" / "web.py").read_text()


def bar(up_prints: int, back_prints: int, per_bar: int = 5000):
    """A bar that pushed one way and then retraced, through the real
    builder. Returns the forming bar."""
    b = TickBars(per_bar=per_bar)
    t, px = 0.0, 7000.0
    b.on_trade(t, px, 1.0, "B")
    for _ in range(up_prints):
        t += 1.0
        px += 0.25
        b.on_trade(t, px, 1.0, "B")
    for _ in range(back_prints):
        t += 1.0
        px -= 0.25
        b.on_trade(t, px, 1.0, "B")
    return b.current


def run(c, prev=None):
    return ti.read(c.direction, c.open, c.high, c.low, c.close, prev=prev)


class TestItNamesTheTwoStates:
    def test_a_bar_at_its_extreme_is_a_continuation(self):
        assert run(bar(400, 0))["state"] == "continuation"

    def test_a_bar_off_its_extreme_is_a_pullback(self):
        assert run(bar(400, 100))["state"] == "pullback"

    def test_it_is_symmetric_on_a_down_bar(self):
        b = TickBars(per_bar=5000)
        t, px = 0.0, 7000.0
        b.on_trade(t, px, 1.0, "B")
        for _ in range(400):
            t += 1.0
            px -= 0.25
            b.on_trade(t, px, 1.0, "B")
        assert run(b.current)["state"] == "continuation"
        for _ in range(100):
            t += 1.0
            px += 0.25
            b.on_trade(t, px, 1.0, "B")
        c = b.current
        assert c.direction == "down"
        assert run(c)["state"] == "pullback"


class TestTheLagIsGone:
    """THE DEFECT THIS VERSION EXISTS TO FIX, pinned as a number so
    it cannot creep back."""

    def test_one_tick_off_the_high_is_already_a_pullback(self):
        c = bar(400, 1)
        r = run(c)
        assert r["state"] == "pullback", \
            "it is still waiting for the move to finish"
        assert r["depth"] < 0.02, "and it says the retrace is trivial"

    @pytest.mark.parametrize("back", [1, 5, 20, 50, 200])
    def test_it_calls_it_at_every_depth_not_just_a_deep_one(self, back):
        assert run(bar(400, back))["state"] == "pullback", back

    def test_the_old_window_would_have_missed_these(self):
        """The control. A 1,000-print slice on this tape is still
        net-up after a 200-print retrace, which is what produced
        CONTINUATION while price was fifty points off the high.
        Stated here so the fix is measured against the fault rather
        than asserted."""
        c = bar(1200, 200)
        # What the window WOULD have said: price now against price
        # 1,000 prints ago.
        assert c.close > c.open, "the bar is still up overall"
        # What this reading says.
        assert run(c)["state"] == "pullback"
        assert run(c)["depth"] > 0.1

    def test_no_fixed_window_remains_anywhere_in_the_reading(self):
        import inspect
        src = inspect.getsource(ti)
        for gone in ("SLICE_FRACTION", "slice_prints", "MIN_SLICE",
                     "slice_dir"):
            assert gone not in src, f"{gone} is still in the reading"


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
        assert ti.depth("up", 7000, 7000, 7000, 7000) == 0.0
        assert ti.depth("", 7000, 7010, 6990, 7005) == 0.0

    @pytest.mark.parametrize("junk", [None, "x", object()])
    def test_rubbish_prices_do_not_raise(self, junk):
        assert ti.depth("up", junk, junk, junk, junk) == 0.0


class TestRecoveringIsAQualifierNotAState:
    def test_it_marks_a_pullback_that_has_turned(self):
        deep = run(bar(400, 200))
        prev = {"state": deep["state"], "worst": deep["worst"],
                "bar_dir": deep["bar_dir"]}
        # Same bar, price recovered half of the retrace.
        r = ti.read("up", 7000.0, 7100.0, 6999.0, 7075.0, prev=prev)
        assert r["recovering"] is True
        assert r["state"] == "pullback", \
            "it called the continuation before a new extreme"
        assert "recovering" in r["say"]

    def test_a_new_extreme_is_what_ends_the_pullback(self):
        r = ti.read("up", 7000.0, 7100.0, 6999.0, 7100.0,
                    prev={"state": "pullback", "worst": 0.5,
                          "bar_dir": "up"})
        assert r["state"] == "continuation"
        assert r["worst"] == 0.0, "the old retrace was not cleared"
        assert r["recovering"] is False

    def test_a_deepening_pullback_is_not_recovering(self):
        r = ti.read("up", 7000.0, 7100.0, 6999.0, 7050.0,
                    prev={"state": "pullback", "worst": 0.5,
                          "bar_dir": "up"})
        assert r["recovering"] is False
        assert r["worst"] == 0.5

    def test_the_worst_is_remembered_not_recomputed(self):
        """One instant cannot know how deep a pullback has been.
        Without the memory, a recovery looks like a shallow
        pullback and RECOVERING never appears."""
        r = ti.read("up", 7000.0, 7100.0, 6999.0, 7075.0, prev={})
        assert r["recovering"] is False
        assert r["worst"] == 0.25

    def test_a_bar_turning_over_clears_the_memory(self):
        """The geometry is then measured off the other end of the
        bar, and a retrace depth from the old direction is a number
        about a move that is no longer the one being read."""
        r = ti.read("down", 7000.0, 7001.0, 6900.0, 6950.0,
                    prev={"state": "pullback", "worst": 0.9,
                          "bar_dir": "up"})
        assert r["worst"] == 0.0


class TestItNeverBlanks:
    def test_a_tie_holds_the_last_state(self):
        r = ti.read("", 7000, 7000, 7000, 7000,
                    prev={"state": "pullback", "bar_dir": "up"})
        assert r["state"] == "pullback"

    def test_nothing_is_invented_at_the_start(self):
        r = ti.read("", 0, 0, 0, 0)
        assert r["state"] == ""
        assert "no direction" in r["say"]

    def test_a_real_change_passes_straight_through(self):
        r = ti.read("up", 7000, 7010, 6999, 7010,
                    prev={"state": "pullback", "worst": 0.5,
                          "bar_dir": "up"})
        assert r["state"] == "continuation"


class TestItReachesThePanel:
    def test_the_route_builds_the_reading(self):
        i = WEB.index("def _tape_intent(")
        blk = WEB[i:WEB.index("def _m5_now(", i)]
        assert "_ti.read(" in blk
        assert 'tk_ or {}).get("body_dir")' in blk, \
            "it is not reading the bar's own direction"
        assert "slice" not in blk, "the lagging window is still wired in"

    def test_the_tick_route_carries_the_extremes(self):
        i = WEB.index("def _tick_now(")
        blk = WEB[i:WEB.index("def _tape_intent(", i)]
        main = blk[blk.index('"up_prints": cur.up_prints'):]
        assert '"high": cur.high' in main, \
            "no extreme to measure the retrace against"
        assert '"low": cur.low' in main

    def test_the_builder_no_longer_keeps_a_slice(self):
        b = TickBars(per_bar=5000)
        assert not hasattr(b, "_slice")
        assert not hasattr(b, "slice_read")

    def test_the_state_carries_it(self):
        from liqmap import stackwin as sw
        st = sw.state({}, "up", {}, 300.0,
                      intent={"state": "pullback", "depth": 0.4})
        assert st["intent"]["state"] == "pullback"
        assert st["intent"]["decides"] is False

    def test_its_memory_rides_home_with_the_gates(self):
        from liqmap import stackwin as sw
        st = sw.state({}, "up", {}, 300.0,
                      intent={"state": "pullback", "worst": 0.6,
                              "bar_dir": "up"})
        assert st["held"]["intent"] == {"state": "pullback",
                                        "worst": 0.6, "bar_dir": "up"}

    def test_the_route_feeds_the_memory_back_in(self):
        i = WEB.index("def _journal_step_un(")
        blk = WEB[i:i + 30000]
        assert "intent=_tape_intent(_tk, _prev)" in blk
        j = WEB.index("def _tape_intent(")
        assert 'prev_ or {}).get("intent")' in WEB[j:j + 1800]

    def test_the_panel_draws_the_two_states(self):
        assert "'PULLBACK'" in WEB and "'CONTINUATION'" in WEB
        assert "SW.intent && SW.intent.state" in WEB

    def test_the_panel_marks_a_pullback_that_has_turned(self):
        assert "SW.intent.recovering" in WEB
        assert "TURNING BACK" in WEB

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
