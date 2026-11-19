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
        assert st["intent"]["decides"] is True

    def test_its_memory_rides_home_with_the_gates(self):
        from liqmap import stackwin as sw
        st = sw.state({}, "up", {}, 300.0,
                      intent={"state": "pullback", "worst": 0.6,
                              "bar_dir": "up"})
        held = st["held"]["intent"]
        assert held["state"] == "pullback"
        assert held["worst"] == 0.6
        assert held["bar_dir"] == "up"

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


class TestVolumeIsDrawnAndDoesNotVote:
    """It has had three jobs in three revisions and this is the
    right one.

    It turned a trade round into the retrace -- measured at 3% of
    entries, a special case wearing a rule's clothes. Then it
    decided nothing at all, and the rule opened trades on a tick
    bar that had given back 90% of its move on 1.71x the volume,
    because the BODY still read up.

    It now refuses to call a dead push a continuation. It blocks an
    entry; it never closes a trade and never reverses one."""

    SCREENSHOT = {"state": "pullback", "depth": 0.90, "heavy": True}

    def test_a_dead_push_no_longer_blocks_the_entry(self):
        """IT DID, FOR ONE REVISION, and Chris took it back out by
        name: "Nothing else votes ... the pullback reading -- all
        computed, all drawn, none of them gate anything."
        """
        from liqmap import stackwin as sw
        r = {"5m": {"present": True, "qualified": "up"}}
        st = sw.state({}, "up", r, 300.0, tf_ticks=20.0,
                      intent=self.SCREENSHOT)
        assert st["side"] == "up"

    def test_a_live_push_does_not(self):
        from liqmap import stackwin as sw
        r = {"5m": {"present": True, "qualified": "up"}}
        st = sw.state({}, "up", r, 300.0, tf_ticks=20.0,
                      intent={"state": "continuation", "heavy": False})
        assert st["side"] == "up"

    def test_it_never_returns_the_opposite_side(self):
        """The job it had two revisions ago and must not get back."""
        from liqmap import stackwin as sw
        r = {"5m": {"present": True, "qualified": "up"}}
        for intent in (self.SCREENSHOT,
                       {"state": "pullback", "heavy": False},
                       {"state": "continuation", "heavy": True}):
            st = sw.state({}, "up", r, 300.0, tf_ticks=20.0,
                          intent=intent)
            assert st["side"] in ("up", "")

    def test_it_never_closes_a_trade(self):
        from liqmap import stackwin as sw
        assert sw.holds("up", "up") == (True, "")

    def test_the_reading_is_still_computed(self):
        r = ti.read("up", 7000.0, 7100.0, 6999.0, 7040.0,
                    volume=800.0, bar_ts=1.0,
                    prev={"v_ext": 500.0, "bar_ts": 1.0,
                          "bar_dir": "up"})
        assert r["state"] == "pullback"
        assert r["ratio"] > 0 and isinstance(r["heavy"], bool)

    def test_the_ratio_is_what_sets_heavy(self):
        assert ti.read("up", 7000, 7100, 6999, 7050,
                       volume=100.0, bar_ts=1.0,
                       prev={"v_ext": 20.0, "bar_ts": 1.0,
                             "bar_dir": "up"})["heavy"] is True
        assert ti.read("up", 7000, 7100, 6999, 7050,
                       volume=100.0, bar_ts=1.0,
                       prev={"v_ext": 95.0, "bar_ts": 1.0,
                             "bar_dir": "up"})["heavy"] is False

    def test_nothing_here_submits_an_order(self):
        import ast
        tree = ast.parse(pathlib.Path(ti.__file__).read_text())
        bad = [n.name for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name in ("submit", "execute", "place", "order")]
        assert not bad, f"tapeintent.py grew {bad}"


class TestTheVolumeMaths:
    """Each of these was written after a mutation survived. The
    comparison, the mark and the reset are three separate things and
    breaking any one of them leaves the other two looking right."""

    def test_it_is_effort_per_tick_not_raw_volume(self):
        """THE CASE THAT SEPARATES THEM. A long push on big volume,
        then a short retrace on a fraction of that volume -- but
        over so little distance that it is costing MORE per tick
        than the push did. Raw volume calls it light; effort calls
        it heavy, and effort is the one that describes a fight."""
        r = ti.read("up", 7000.0, 7100.0, 6999.0, 7080.0,
                    volume=1300.0, bar_ts=1.0,
                    prev={"v_ext": 1000.0, "bar_ts": 1.0,
                          "bar_dir": "up"})
        assert r["impulse_v"] == 1000.0 and r["retrace_v"] == 300.0
        assert r["depth"] >= ti.MIN_PULLBACK, "not a real retrace"
        assert r["ratio"] == 1.5, r
        assert r["heavy"] is True
        # Raw volume would have read 0.30 and called it drift.
        assert (r["retrace_v"] / r["impulse_v"]) < ti.HEAVY_RATIO

    def test_the_ratio_is_scale_free(self):
        """The price units cancel, so the same shape reads the same
        on any instrument and no tick size is needed."""
        a = ti.effort_ratio("up", 7000, 7100, 6999, 7090, 1000, 150)
        b = ti.effort_ratio("up", 70000, 71000, 69990, 70900, 1000, 150)
        assert a == b == 1.5

    def test_the_mark_is_taken_at_the_extreme(self):
        """Everything before it is the push, everything after is the
        retrace. Without the mark there is no push to compare
        against and nothing is ever heavy."""
        r = ti.read("up", 7000.0, 7100.0, 6999.0, 7100.0,
                    volume=500.0, bar_ts=1.0)
        assert r["at_extreme"] is True
        assert r["v_ext"] == 500.0, "the push volume was not marked"
        assert r["retrace_v"] == 0.0

    def test_the_mark_is_re_taken_at_every_new_extreme(self):
        first = ti.read("up", 7000.0, 7100.0, 6999.0, 7100.0,
                        volume=500.0, bar_ts=1.0)
        later = ti.read("up", 7000.0, 7200.0, 6999.0, 7200.0,
                        volume=800.0, bar_ts=1.0,
                        prev={"v_ext": first["v_ext"], "bar_ts": 1.0,
                              "bar_dir": "up"})
        assert later["v_ext"] == 800.0

    def test_a_new_bar_clears_the_mark(self):
        """Volume restarts at nought on a new bar. Carrying the old
        mark forward makes the whole of the new bar look like a
        retrace of a move that has left the screen."""
        r = ti.read("up", 7000.0, 7010.0, 6999.0, 7005.0,
                    volume=50.0, bar_ts=2.0,
                    prev={"v_ext": 10.0, "bar_ts": 1.0,
                          "bar_dir": "up", "worst": 0.9})
        assert r["v_ext"] == 50.0, "the previous bar's mark survived"
        assert r["retrace_v"] == 0.0
        assert r["worst"] == 0.0

    def test_the_same_bar_keeps_its_mark(self):
        """The control for the test above -- a reset that fires
        every poll is the same as no mark at all."""
        r = ti.read("up", 7000.0, 7100.0, 6999.0, 7050.0,
                    volume=600.0, bar_ts=1.0,
                    prev={"v_ext": 400.0, "bar_ts": 1.0,
                          "bar_dir": "up"})
        assert r["v_ext"] == 400.0
        assert r["retrace_v"] == 200.0


class TestTheRouteCarriesWhatTheMathsNeeds:
    def test_the_tick_payload_has_volume_and_the_bar_id(self):
        i = WEB.index("def _tick_now(")
        blk = WEB[i:WEB.index("def _tape_intent(", i)]
        main = blk[blk.index('"up_prints": cur.up_prints'):]
        assert '"volume": cur.volume' in main, \
            "no volume, so no pullback can ever read heavy"
        assert '"start_ts": cur.start_ts' in main, \
            "no bar id, so a new bar keeps the old volume mark"

    def test_the_reading_is_handed_both(self):
        i = WEB.index("def _tape_intent(")
        blk = WEB[i:WEB.index("def _m5_now(", i)]
        assert 'volume=float((tk_ or {}).get("volume")' in blk
        assert 'bar_ts=float((tk_ or {}).get("start_ts")' in blk


class TestTheFlagDoesNotChatter:
    """Two defects found by MEASURING the rule, not by reading it.
    Both were mine and both were invisible in a unit test."""

    def test_a_bare_threshold_was_the_first_one(self):
        """`heavy` was `ratio >= 1.0`. A number that moves
        continuously flips every time it crosses, so the side
        inverted about 1,250 times an hour on a simulated tape --
        four out of five side changes were this one flag. The same
        defect `PATH_BAND` was added to fix in the tick vote."""
        assert ti.HEAVY_BAND > 0

    @staticmethod
    def _r(ratio_target, was, depth=0.5):
        """A reading whose effort ratio is `ratio_target`."""
        # push 100 on volume 100 -> 1.0 per unit; back = depth*100.
        back = depth * 100.0
        rv = ratio_target * 100.0 * back / 100.0
        return ti.read("up", 7000.0, 7100.0, 6999.0, 7100.0 - back,
                       volume=100.0 + rv, bar_ts=1.0,
                       prev={"v_ext": 100.0, "bar_ts": 1.0,
                             "bar_dir": "up", "heavy": was})

    def test_it_takes_more_to_claim_heavy_than_to_keep_it(self):
        assert self._r(1.1, was=False)["heavy"] is False
        assert self._r(1.1, was=True)["heavy"] is True

    def test_leaving_heavy_needs_the_lower_edge_not_the_middle(self):
        """BOTH EDGES, OR IT IS HALF A BAND. A mutation that kept
        the entry edge and dropped the exit one survived: 0.9 is
        below 1.0 but inside the band, so it must still read heavy
        once claimed. Without this the flag chatters on every pass
        back through 1.0, which is the whole defect."""
        assert self._r(0.9, was=True)["heavy"] is True
        assert self._r(0.74, was=True)["heavy"] is False

    def test_well_past_either_edge_is_unambiguous(self):
        assert self._r(2.0, was=False)["heavy"] is True
        assert self._r(0.2, was=True)["heavy"] is False

    def test_a_tiny_retrace_can_never_be_heavy(self):
        """THE SECOND DEFECT, and the tell was that a TRENDING tape
        produced more side changes than a choppy one -- backwards.
        The ratio divides by how far price has come back, so one
        tick off the high with any volume read as the move being
        fought, and a trend makes dozens of those."""
        r = self._r(99.0, was=False, depth=0.01)
        assert r["depth"] < ti.MIN_PULLBACK
        assert r["heavy"] is False, "a one-tick retrace turned the trade"

    def test_it_is_still_called_a_pullback_though(self):
        """The depth floor gates the VOLUME verdict only. Naming the
        pullback stays instant -- that was the whole last fix."""
        r = self._r(99.0, was=False, depth=0.01)
        assert r["state"] == "pullback"

    def test_a_real_retrace_is_judged_normally(self):
        r = self._r(2.0, was=False, depth=ti.MIN_PULLBACK + 0.05)
        assert r["heavy"] is True


class TestTheMemoryCarriesEverythingTheReadingNeeds:
    def test_the_state_hands_back_every_field(self):
        """`read` is pure with its memory held outside, so a field
        missing from `held` does not error -- it silently resets
        that part of the reading on every poll."""
        from liqmap import stackwin as sw
        intent = ti.read("up", 7000.0, 7100.0, 6999.0, 7050.0,
                         volume=600.0, bar_ts=7.0,
                         prev={"v_ext": 400.0, "bar_ts": 7.0,
                               "bar_dir": "up", "heavy": True})
        st = sw.state({}, "up", {}, 300.0, intent=intent)
        held = st["held"]["intent"]
        for k in ("state", "worst", "bar_dir", "heavy", "v_ext",
                  "bar_ts"):
            assert k in held, f"{k} is dropped between polls"
        assert held["heavy"] is True
        assert held["v_ext"] == 400.0
        assert held["bar_ts"] == 7.0

    def test_a_round_trip_through_the_memory_is_stable(self):
        from liqmap import stackwin as sw
        prev = {}
        for _ in range(3):
            intent = ti.read("up", 7000.0, 7100.0, 6999.0, 7050.0,
                             volume=600.0, bar_ts=7.0, prev=prev)
            prev = sw.state({}, "up", {}, 300.0,
                            intent=intent)["held"]["intent"]
        assert prev["v_ext"] == 600.0 or prev["v_ext"] == 400.0
        assert isinstance(prev["heavy"], bool)
