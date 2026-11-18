"""The tick bar alone, and what its own volume says about it.

Chris: "we are going to remove the 10 second hold completely its not
a factor in the opening anymore we are going to use volume to confirm
tick bars continuation or pull back we will no longer use time as a
entry point or exit at all just what the tick bar is doing alone" --
and, on what a volume-backed pullback means, "if volume agrees with
tick charts direction for a pull back a trade should be taken in that
pullback direction as well".

THE RULE, WHOLE:

    The 5,000-tick candle's direction is close against open, held
    through the instant close sits on its open so it never blanks.

    AT ITS EXTREME, the bar is extending: trade its direction.

    PULLING BACK ON LIGHT VOLUME, the retrace is drift and the push
    still stands: trade its direction.

    PULLING BACK ON HEAVY VOLUME -- more size per tick to push it
    back than it took to make the move -- the move is being fought:
    trade the RETRACE'S direction.

    There is always a side while the bar has a direction, which is
    what he has now asked for three times in three places.

WHAT CAME OUT, AND EACH OF THEM IS TESTED IN THE NEGATIVE BELOW
because every one of these has been silently re-wired in this file
before:

    THE TEN-SECOND HOLD, with its tracker and its constant.
    THE TIMEFRAME CANDLE GATE -- still read, still drawn, silent.
    THE FIVE-MINUTE TIME STOP.
    THE PER-BAR CAP, which was never meant to be there at all.

THE ONLY SIGNAL EXIT IS THE RULE'S OWN SIDE TURNING OVER. One number
opens a trade and the same number ends it, which is what stops the
engine opening a long on one reading and closing it on another --
it did exactly that until this revision.
"""

from __future__ import annotations

import pytest

from liqmap import stackwin as sw

D = "down"
U = "up"


def rungs(**kw) -> dict:
    """Cascade rows, as the route hands them over."""
    out = {}
    for tf, q in kw.items():
        if q is None:
            out[tf] = {"present": False, "qualified": None,
                       "conflicted": False}
        elif q == "conflict":
            out[tf] = {"present": True, "qualified": None,
                       "conflicted": True}
        else:
            out[tf] = {"present": True, "qualified": q,
                       "conflicted": False, "bps": -1.3}
    return out


def pull(heavy: bool) -> dict:
    """An intent reading that is pulling back, paid for or not."""
    return {"state": "pullback", "heavy": heavy, "depth": 0.3,
            "ratio": 1.8 if heavy else 0.3}


def push() -> dict:
    return {"state": "continuation", "heavy": False, "depth": 0.0,
            "ratio": 0.0}

class TestTheStackIsTheTradedRowAlone:
    def test_five_minutes_reads_the_5m(self):
        r = rungs(**{"15m": U, "5m": D, "1m": U})
        assert sw.stack_side(r, 300.0) == ("down", "")

    def test_fifteen_minutes_reads_the_15m(self):
        r = rungs(**{"15m": D, "5m": U, "1m": U})
        assert sw.stack_side(r, 900.0) == ("down", "")

    def test_the_row_below_is_ignored_entirely(self):
        """Chris: "paying no attention to the 1 minute at all". It
        gated entries until he took it out; it is still read and
        still drawn and it decides nothing."""
        for below in (U, D, None, "conflict"):
            r = rungs(**{"5m": D, "1m": below})
            assert sw.stack_side(r, 300.0)[0] == "down", below

    def test_the_row_above_is_not_consulted(self):
        """On the 5-minute he is trading the 5-minute. A 15-minute
        against it is the pullback he is trading into, not a veto."""
        r = rungs(**{"15m": U, "5m": D, "1m": D})
        assert sw.stack_side(r, 300.0)[0] == "down"

    def test_the_row_below_is_not_consulted_on_fifteen(self):
        r = rungs(**{"15m": D, "5m": D, "1m": U})
        assert sw.stack_side(r, 900.0)[0] == "down"

    def test_a_flat_traded_row_is_no_side(self):
        r = rungs(**{"5m": "flat", "1m": D})
        side, why = sw.stack_side(r, 300.0)
        assert side == "" and "the 5m is flat" in why

    def test_an_unread_traded_row_is_named(self):
        r = rungs(**{"5m": None, "1m": D})
        side, why = sw.stack_side(r, 300.0)
        assert side == "" and "5m has not been read" in why

    def test_an_unseeded_traded_row_has_no_direction(self):
        """Without the bar's open there is no close-against-open."""
        r = {"5m": {"present": True, "qualified": "down",
                    "seeded": False}, "1m": {"present": True,
                                             "qualified": "down"}}
        side, why = sw.stack_side(r, 300.0)
        assert side == "" and "open was never seen" in why


class TestBothGatesMustAgree:
    """Chris, restating the rule he wants back: "Gate 1 -- the 5000t
    candle's direction ... Gate 2 -- the real live timeframe candle
    ... Both gates point the same way. That side is the trade." --
    and, in the message that asked for it, "lets add back the
    timeframe section without the 10 second hold"."""

    @pytest.mark.parametrize("d", [U, D])
    def test_agreement_is_an_entry(self, d):
        assert sw.entry_side(d, d) == (d, "")

    def test_no_tick_direction_is_no_entry(self):
        side, why = sw.entry_side("", U)
        assert side == "" and "tick candle has no direction" in why

    @pytest.mark.parametrize("junk", [None, "flat", "UP", 7])
    def test_junk_is_never_a_side(self, junk):
        assert sw.entry_side(junk, U)[0] == ""
        assert sw.entry_side(U, junk)[0] == ""

    @pytest.mark.parametrize("tick,candle", [(U, D), (D, U)])
    def test_a_disagreeing_candle_is_a_pullback_not_an_entry(
            self, tick, candle):
        """It blocks the ENTRY and it does not close a trade --
        that distinction is the whole of this revision."""
        side, why = sw.entry_side(tick, candle)
        assert side == ""
        assert "pulling back" in why
        assert f"the tick candle is {tick}" in why
        assert f"{candle} timeframe" in why

    def test_no_candle_direction_blocks_it(self):
        side, why = sw.entry_side(U, "")
        assert side == ""
        assert "timeframe candle has no direction" in why


class TestTheTickBarsOwnPushMustStillBeAlive:
    """Chris, on a screenshot of a trade taken against this: "the
    chart book 5000t and the gate 1 tick bar are not synced at all
    ... its taking trades on data that has already happened".

    IT WAS NOT A CLOCK. Both read the same 5,000 builder. `tick` is
    the candle's BODY -- close against open, held through small
    moves -- so a bar that ran twenty ticks up and gave back
    eighteen still reads UP. It is not pushing. It is rolling over,
    and the body is a summary of where it has BEEN.

    From his panel, with the trade taken on it:

        tick bar UP, +2 ticks from this candle's open
        tape PULLBACK -- 90% given back, HEAVY, 1.71x the effort
        THE TWO CHARTS TOGETHER: CONTINUATION

    The reading that said the push was dead was two lines above the
    one that called it a continuation, and it was marked as
    deciding nothing.

    VOLUME DECIDES AGAIN, but not the way it did two revisions ago.
    It does not turn a trade round into the retrace; it refuses to
    call a dead push a continuation."""

    SCREENSHOT = {"state": "pullback", "depth": 0.90, "heavy": True,
                  "ratio": 1.71}
    ALIVE = {"state": "continuation", "depth": 0.0, "heavy": False}

    def test_the_screenshot_is_no_longer_a_continuation(self):
        st = sw.state({}, U, rungs(**{"5m": U}), 300.0, tf_ticks=20.0,
                      intent=self.SCREENSHOT)
        assert st["regime"]["where"] == "pullback"
        assert st["regime"]["tick_failed"] is True
        assert st["side"] == "", "it opened on a push that was over"
        assert "rolling over" in st["why"]

    def test_a_live_push_still_trades(self):
        st = sw.state({}, U, rungs(**{"5m": U}), 300.0, tf_ticks=20.0,
                      intent=self.ALIVE)
        assert st["regime"]["where"] == "continuation"
        assert st["side"] == U

    def test_a_shallow_unpaid_for_dip_still_trades(self):
        """The whole point of the previous revision was holding
        through ordinary retraces. This must not undo it."""
        st = sw.state({}, U, rungs(**{"5m": U}), 300.0, tf_ticks=20.0,
                      intent={"state": "pullback", "depth": 0.2,
                              "heavy": False})
        assert st["regime"]["where"] == "continuation"
        assert st["side"] == U

    @pytest.mark.parametrize("depth,heavy,alive", [
        (0.2, False, True), (0.5, False, True),
        (0.7, False, False), (0.2, True, False),
    ])
    def test_it_uses_the_same_failure_test_as_the_timeframe(
            self, depth, heavy, alive):
        st = sw.state({}, U, rungs(**{"5m": U}), 300.0, tf_ticks=20.0,
                      intent={"state": "pullback", "depth": depth,
                              "heavy": heavy})
        assert (st["regime"]["where"] == "continuation") is alive
        assert sw.pullback_failed(depth, heavy) is (not alive)

    def test_a_bar_at_its_extreme_is_never_a_dead_push(self):
        """The guard that stops a retrace reading leaking onto a bar
        that is EXTENDING. `depth` is 0 at the extreme, but a stray
        `heavy` would otherwise fail a push that is making new
        highs -- a mutation dropping the state check survived until
        this test existed."""
        st = sw.state({}, U, rungs(**{"5m": U}), 300.0, tf_ticks=20.0,
                      intent={"state": "continuation", "depth": 0.0,
                              "heavy": True})
        assert st["regime"]["tick_failed"] is False
        assert st["regime"]["where"] == "continuation"
        assert st["side"] == U

    def test_the_regime_function_takes_it_directly(self):
        assert sw.regime(U, U) == "continuation"
        assert sw.regime(U, U, tick_failed=True) == "pullback"
        assert sw.regime(U, D, tick_failed=False) == "pullback"

    def test_the_entry_reads_the_same_regime_the_panel_shows(self):
        """It recomputed its own and they disagreed -- the box said
        PULLBACK while the side still read UP."""
        side, why = sw.entry_side(U, U, 20.0, tick_failed=True)
        assert side == "" and "rolling over" in why

    def test_the_reading_is_marked_as_deciding(self):
        """It sat on the panel saying "shown only" while the rule
        traded against it."""
        st = sw.state({}, U, rungs(**{"5m": U}), 300.0, tf_ticks=20.0,
                      intent=self.SCREENSHOT)
        assert st["intent"]["decides"] is True

    def test_a_dead_push_does_not_close_an_open_trade(self):
        """It blocks an ENTRY. Closing on it would put the tick
        bar's own retrace back in charge of exits, which is the
        churn the previous revision removed."""
        assert sw.holds(U, U, U) == (True, "")


class TestWhatCameOut:
    """Each in the negative. Every one of these has been removed and
    quietly re-wired in this file before."""

    def test_the_dwell_is_gone_entirely(self):
        assert not hasattr(sw, "Dwell"), "the tracker is still here"
        assert not hasattr(sw, "DWELL_S")
        import inspect
        src = inspect.getsource(sw)
        for gone in ("held_s", "min_hold_s", "has to hold"):
            assert gone not in src, f"{gone} survived"

    def test_the_route_has_no_dwell_left(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("def _journal_step_un(")
        blk = src[i:i + 30000]
        assert "_dw." not in blk and "Dwell" not in blk

    def test_the_time_stop_is_gone(self):
        """`bar_of` is back and is NOT a time stop -- it stamps
        which five minutes a trade opened in so the cap can count
        them. What stays gone is anything that ENDS a trade on a
        clock."""
        for gone in ("exit_reason", "expired"):
            assert not hasattr(sw, gone), f"{gone} is still here"
        import inspect
        src = inspect.getsource(sw.bar_of)
        assert "counter" in src.lower(), \
            "bar_of is not documented as a counter"

    def test_the_route_has_no_clock_left(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        assert "TRADE_BAR_S = " not in src, \
            "the five-minute constant is still defined"
        assert "_sw.exit_reason(" not in src

    def test_the_cap_is_explicit_rather_than_inherited(self):
        """THE BUG THAT MADE THIS WORTH A TEST. The stack branch
        once passed `cap = 0` under a comment reading NO CAP, and
        `decide` does `max(1, max_per_bar or MAX_PER_BAR)` -- so 0
        became 2 and nobody meant it to. Whatever the cap is, it
        has to be said out loud here."""
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index('if strat == "stack":', src.index("cap = 0"))
        blk = src[i:i + 1800]
        assert "cap = _swb.MAX_TRADES_PER_BAR" in blk, \
            "the cap is inherited from journal.decide again"


class TestAPullbackDoesNotCloseATrade:
    """THE CHANGE THIS REVISION EXISTS FOR.

    Chris: "every trade that im loosing is because of flips" -- and
    then the reason, which is the sentence that rewrote the rule:
    "tick chart pushing opposite direction of time is a pullback and
    vic versa a continuation when they both agree on direction".

    The tick bar turning against the timeframe was being read as the
    signal failing, and it closed the trade. It is a PULLBACK. Every
    retrace became a realised loss, which is exactly what he was
    watching and could not do by hand.
    """

    def test_a_tick_flip_alone_is_held_through(self):
        assert sw.holds(D, U, D) == (True, "")
        assert sw.holds(U, D, U) == (True, "")

    def test_a_tick_flip_that_has_failed_does_close_it(self):
        ok, why = sw.holds(D, U, D, failed=True)
        assert ok is False
        assert "went too far to still be a pullback" in why

    def test_the_timeframe_turning_closes_it(self):
        ok, why = sw.holds(D, D, U)
        assert ok is False
        assert "timeframe candle turned up against your down" in why

    def test_the_timeframe_is_blamed_before_the_pullback(self):
        """A log blaming a failed pullback when the trend itself
        had turned sends you to the wrong chart."""
        ok, why = sw.holds(D, U, U, failed=True)
        assert ok is False and "timeframe candle turned" in why

    def test_it_holds_while_both_agree(self):
        assert sw.holds(D, D, D) == (True, "")

    @pytest.mark.parametrize("tick,candle", [
        ("", D), (D, ""), ("", ""), ("flat", "flat"),
    ])
    def test_FLAT_IS_NOT_A_DISAGREEMENT(self, tick, candle):
        assert sw.holds(D, tick, candle) == (True, "")

    def test_a_failed_flag_without_a_flip_changes_nothing(self):
        """`failed` describes a pullback. With the tick bar still
        with you there is no pullback for it to describe."""
        assert sw.holds(D, D, D, failed=True) == (True, "")

    def test_no_open_side_is_never_an_exit(self):
        assert sw.holds("", U, U, failed=True) == (True, "")


class TestWhenAPullbackStopsBeingOne:
    def test_too_deep_fails_it(self):
        assert sw.pullback_failed(0.7) is True
        assert sw.pullback_failed(0.3) is False

    def test_the_threshold_is_named(self):
        assert sw.PULLBACK_FAILS_AT == 0.6
        assert sw.pullback_failed(sw.PULLBACK_FAILS_AT) is True

    def test_volume_fails_it_at_any_depth(self):
        """`heavy` already refuses to call a trivial retrace heavy
        -- `tapeintent` floors it at MIN_PULLBACK -- so this does
        not need a second depth test in front of it."""
        assert sw.pullback_failed(0.2, heavy=True) is True

    def test_shallow_and_unpaid_for_is_the_case_we_hold(self):
        assert sw.pullback_failed(0.2, heavy=False) is False

    @pytest.mark.parametrize("junk", [None, "x", object()])
    def test_junk_does_not_fail_a_pullback(self, junk):
        assert sw.pullback_failed(junk) is False


class TestTheRegimeReading:
    def test_agreement_is_a_continuation(self):
        assert sw.regime(U, U) == "continuation"
        assert sw.regime(D, D) == "continuation"

    def test_disagreement_is_a_pullback(self):
        assert sw.regime(U, D) == "pullback"
        assert sw.regime(D, U) == "pullback"

    @pytest.mark.parametrize("a,b", [("", U), (U, ""), ("flat", U)])
    def test_an_incomplete_pair_is_neither(self, a, b):
        assert sw.regime(a, b) == ""

    def test_the_state_reports_it_and_says_it_decides(self):
        st = sw.state({}, U, rungs(**{"5m": D}), 300.0,
                      tf_ticks=40.0,
                      tf_intent={"depth": 0.2, "heavy": False})
        assert st["regime"]["where"] == "pullback"
        assert st["regime"]["failed"] is False
        assert st["regime"]["decides"] is True
        assert "pullback" in st["regime"]["say"]
        assert "holding through it" in st["regime"]["say"]

    def test_the_state_marks_a_failing_pullback(self):
        st = sw.state({}, U, rungs(**{"5m": D}), 300.0,
                      tf_ticks=40.0,
                      tf_intent={"depth": 0.8, "heavy": False})
        assert st["regime"]["failed"] is True
        assert "FAILING" in st["regime"]["say"]

    def test_a_continuation_is_never_marked_failed(self):
        st = sw.state({}, U, rungs(**{"5m": U}), 300.0,
                      tf_ticks=40.0,
                      tf_intent={"depth": 0.9, "heavy": True})
        assert st["regime"]["where"] == "continuation"
        assert st["regime"]["failed"] is False, \
            "a retrace reading leaked onto a bar that is extending"


class TestNoTradeAtTheTimeframesOpen:
    """Chris: "no trades are ever taken at the opening price of
    whatever timeframe i select on my auto trader"."""

    def test_the_minimum_is_named(self):
        assert sw.MIN_OPEN_TICKS == 4.0

    @pytest.mark.parametrize("ticks", [0.0, 1.0, -1.0, 3.9, -3.9])
    def test_sitting_on_the_open_is_no_entry(self, ticks):
        side, why = sw.entry_side(U, U, ticks)
        assert side == ""
        assert "off its open" in why

    @pytest.mark.parametrize("ticks", [4.0, -4.0, 40.0, -40.0])
    def test_clear_of_the_open_is_fine(self, ticks):
        assert sw.entry_side(U, U, ticks) == (U, "")

    def test_an_unmeasurable_distance_does_not_block_it(self):
        """A caller that cannot see the distance must not be
        silently stopped from ever trading."""
        assert sw.entry_side(U, U, None) == (U, "")

    def test_the_state_applies_it(self):
        assert sw.state({}, U, rungs(**{"5m": U}), 300.0,
                        tf_ticks=1.0)["side"] == ""
        assert sw.state({}, U, rungs(**{"5m": U}), 300.0,
                        tf_ticks=12.0)["side"] == U

    def test_the_route_measures_it_off_the_traded_candle(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("_tf_name = (")
        blk = src[i:i + 1400]
        assert "tf_ticks=" in blk, "the distance never reaches the rule"
        assert "_candle_now(rt_, _tf_name)" in blk


class TestTheRouteFeedsTheReasoning:
    def test_the_timeframe_candle_exposes_its_extremes(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("def _candle_now(")
        # To the NEXT def, whatever it is. `_tf_intent` sits above
        # this one, so slicing to it ran backwards and raised.
        blk = src[i:src.index("\n    def ", i + 10)]
        # THE VALUE, NOT THE KEY. Asserting on the key alone let a
        # mutation that zeroed every one of them survive -- the
        # rule then read a retrace of nothing off a candle with no
        # high, which is a pullback that can never fail.
        for k in ("high", "low", "volume"):
            assert f'"{k}": float(getattr(c, "{k}"' in blk, \
                f"{k} is not read off the candle"

    def test_a_timeframe_candle_reading_carries_real_numbers(self):
        """The guarantee the source check cannot make: a candle
        that has run up and come back must produce a retrace the
        rule can act on."""
        from liqmap import tapeintent as ti
        r = ti.read("up", 7000.0, 7100.0, 6995.0, 7040.0,
                    volume=800.0, bar_ts=1.0,
                    prev={"v_ext": 500.0, "bar_ts": 1.0,
                          "bar_dir": "up"})
        assert r["depth"] == 0.6, r
        assert sw.pullback_failed(r["depth"], r["heavy"]) is True
        shallow = ti.read("up", 7000.0, 7100.0, 6995.0, 7090.0,
                          volume=520.0, bar_ts=1.0,
                          prev={"v_ext": 500.0, "bar_ts": 1.0,
                                "bar_dir": "up"})
        assert shallow["depth"] == 0.1
        assert sw.pullback_failed(shallow["depth"],
                                  shallow["heavy"]) is False

    def test_one_retrace_reader_serves_both_scales(self):
        """A second copy of "how far has this come back" is a second
        thing to keep correct, and the first time they disagreed the
        screen and the rule would describe one retrace differently."""
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("def _tf_intent(")
        blk = src[i:src.index("def _m5_now(", i)]
        assert "_ti.read(" in blk
        assert "tapeintent" in blk

    def test_the_rule_is_handed_the_timeframes_retrace(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("fus = _sw.state(win, _bd, _rungs,")
        blk = src[i:i + 900]
        assert "tf_intent=_tfi" in blk

    def test_the_exit_is_handed_the_failed_flag(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index('elif strat == "stack" and t is not None and t.open:')
        blk = src[i:i + 2000]
        assert 'failed=bool((fus.get("regime") or {}).get("failed"))' \
            in blk, "a pullback can never fail, so nothing closes early"

    def test_the_panel_draws_the_reasoning(self):
        """"invisible" has been half of every complaint on this
        panel. A rule that holds a trade through a retrace has to
        show that it is doing so."""
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("function paintStackWin(")
        blk = src[i:src.index("function paintJournal(", i)]
        assert "SW.regime" in blk, "the reasoning never reaches the box"
        assert "'CONTINUATION'" in blk and "'PULLBACK'" in blk
        assert "holding through it" in blk, \
            "a held pullback is silent -- the exact complaint"
        assert "FAILING" in blk
        assert "move given back" in blk and "fails " in blk
        assert "rg.tf_ticks" in blk, \
            "the distance off the timeframe open is invisible"
        assert "regimeBlock" in blk, "it is built and never drawn"
        assert "rg.tick_failed" in blk, \
            "a dead tick push is invisible -- it sat on this panel " \
            "marked SHOWN ONLY while the rule traded against it"
        assert "own push has gone" in blk

    def test_the_timeframe_reading_survives_between_polls(self):
        """It is a pure function with its memory held outside. A
        field dropped from the carry-over does not error -- it
        silently resets that part of the reading every poll."""
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("rt_._gate_held = ")
        blk = src[i:i + 700]
        for k in ("v_ext", "bar_ts", "heavy", "worst"):
            assert k in blk, f"{k} is not carried for the timeframe"


class TestTwoTradesAFiveMinuteBar:
    """Chris: "we need to set a rule for no more then 2 trades on a
    5 minute timeframe and rules have to be met before a entry"."""

    def test_the_cap_is_two(self):
        assert sw.MAX_TRADES_PER_BAR == 2

    def test_the_bar_is_five_minutes_whatever_the_window_is(self):
        """He said five. The window selector can read 15m, and
        counting against the traded row would make the cap three
        times looser than he asked for."""
        assert sw.CAP_BAR_S == 300.0

    def test_the_buckets_are_true_wall_clock_marks(self):
        b = sw.bar_of(1_700_000_000.0)
        assert b % 300.0 == 0.0, "not on a five-minute mark"
        assert sw.bar_of(b) == b
        assert sw.bar_of(b + 299.0) == b
        assert sw.bar_of(b + 300.0) == b + 300.0

    @pytest.mark.parametrize("bad", [0, -1, None, "", "x"])
    def test_junk_is_not_a_bar(self, bad):
        assert sw.bar_of(bad) == 0.0

    def test_the_route_stamps_and_caps(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index('if strat == "stack":', src.index("cap = 0"))
        blk = src[i:i + 1800]
        assert "_swb.bar_of(time.time(), _swb.CAP_BAR_S)" in blk, \
            "the trade is not stamped, so nothing can count it"
        assert "cap = _swb.MAX_TRADES_PER_BAR" in blk, \
            "the cap is not applied"
        assert "strict=True" in blk, \
            "an unidentifiable bar would permit an entry every poll"

    def test_the_cap_reaches_the_re_entry_path_too(self):
        """It did not once, so after a close the rule fell back to
        the default and could turn round inside the same bar."""
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        # THE ONE INSIDE THE JOURNAL STEP. There is another
        # `if act == "close":` in a hand-trading route further up,
        # and a plain `index` finds that one -- a whole-file search
        # that lands on the wrong block is how the earlier route
        # test passed while the thing it named was untouched.
        i = src.index("def _journal_step_un(")
        blk = src[i:i + 30000]
        j = blk.index('if act == "close":')
        assert "max_per_bar=cap" in blk[j:j + 1200]

    def test_nothing_expires_because_of_the_stamp(self):
        """IT IS A COUNTER, NOT A CLOCK ON THE TRADE. The time stop
        was removed and stays removed -- `holds` is the only thing
        that ends a trade on a signal."""
        import inspect
        import pathlib
        assert "expired" not in inspect.getsource(sw)
        assert not hasattr(sw, "exit_reason")
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        assert "TRADE_BAR_S" not in src


class TestTheStateCarriesTheRule:
    def test_the_side_is_the_agreement(self):
        assert sw.state({}, U, rungs(**{"5m": U}), 300.0)["side"] == U
        assert sw.state({}, U, rungs(**{"5m": D}), 300.0)["side"] == ""

    def test_fifteen_minutes_gates_on_the_15m(self):
        assert sw.state({}, D, rungs(**{"15m": D, "5m": U}),
                        900.0)["side"] == D
        assert sw.state({}, D, rungs(**{"15m": U, "5m": D}),
                        900.0)["side"] == ""

    def test_the_row_below_still_gates_nothing(self):
        for below in (U, D, None, "conflict"):
            r = rungs(**{"5m": D, "1m": below})
            assert sw.state({}, D, r, 300.0)["side"] == D, below

    def test_the_verdict_names_both(self):
        st = sw.state({}, U, rungs(**{"5m": U}), 300.0)
        assert "tick candle" in st["say"] and "5m candle" in st["say"]

    def test_the_panel_is_told_the_gate_decides(self):
        st = sw.state({}, U, rungs(**{"5m": U}), 300.0)
        assert st["stack"]["asked"] is True
        assert st["stack"]["decides"] is True
        assert st["stack"]["agrees"] is True

    def test_the_tick_direction_still_holds_through_a_tie(self):
        st = sw.state({}, "", rungs(**{"5m": D}), 300.0,
                      prev={"tick": D})
        assert st["tick"]["side"] == D
        assert st["side"] == D


class TestItStillTakesAsManyTradesAsTheTapeGives:
    def test_the_cap_does_not_apply_to_this_strategy(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("cap = 0")
        # To the cap itself, not a fixed window -- the stack branch
        # now sits between them and pushed it out of a 700-char
        # slice.
        blk = src[i:src.index("cap = 1", i) + 10]
        assert 'strat in ("fused", "timed")' in blk
        assert 'strat == "stack"' in blk
        head = blk.split('if strat in ("fused", "timed")')[0]
        assert "cap = 1" not in head, \
            "the stack rule was given a one-per-candle cap"


class TestTheEdges:
    def test_the_window_length_picks_the_row(self):
        assert sw.traded_tf(300.0) == "5m"
        assert sw.traded_tf(900.0) == "15m"
        assert sw.traded_tf(600.0) == "15m"
        assert sw.traded_tf(120.0) == "5m"

    @pytest.mark.parametrize("bad", [None, 0, "", "x"])
    def test_junk_never_produces_a_side(self, bad):
        assert sw.traded_tf(bad) in ("", "5m", "15m")
        assert sw.entry_side(bad, push())[0] == ""

    def test_no_rungs_at_all_is_a_reason_not_a_crash(self):
        side, why = sw.stack_side({}, 300.0)
        assert side == "" and why

    def test_state_survives_empty_everything(self):
        st = sw.state(None, "", None, 300.0)
        assert st["side"] == ""
        assert st["tick"]["true"] is False
        # The gate is always consulted now, so `asked` is always
        # true. What an empty stack must still do is BLOCK.
        # The gate is always consulted, so `asked` is always true.
        # What an empty stack must do is BLOCK.
        assert st["stack"]["asked"] is True
        assert st["stack"]["agrees"] is False


class TestItDoesNotTrade:
    def test_nothing_here_submits_an_order(self):
        import ast
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "stackwin.py")
        tree = ast.parse(src.read_text())
        bad = [n.name for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name in ("submit", "execute", "place", "order")]
        assert not bad, f"stackwin.py grew {bad}"


class TestTheWiring:
    def test_the_journal_knows_the_strategy(self):
        from liqmap import journal as jn
        assert "stack" in jn.STRATEGIES

    def test_the_signal_dispatches_to_it(self):
        from liqmap import journal as jn
        side, _why = jn.window_signal({}, "stack", "",
                                      {"side": "down", "why": ""})
        assert side == "down"
        side, why = jn.window_signal({}, "stack", "",
                                     {"side": "", "why": "no window"})
        assert side == "" and why == "no window"

    def test_it_is_not_refused_for_no_window_yet(self):
        from liqmap import journal as jn
        side, _why = jn.window_signal(None, "stack", "",
                                      {"side": "up", "why": ""})
        assert side == "up"

    def test_the_route_computes_it_and_wires_both_ends(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("def _journal_step_un(")
        blk = src[i:i + 30000]
        assert 'strat == "stack"' in blk, "the route never builds it"
        assert "_sw.state(" in blk
        assert '"stackwin"' in blk, "it never reaches the payload"
        assert "_sw.exit_reason(" not in blk, \
            "the time stop is wired back in"
        assert "intent=_tape_intent(" in blk, \
            "the volume reading never reaches the rule"
        assert "_tick_now(rt_, 0, \"raw\", 0.0, 0, FUSION_RESET_S)" in blk, \
            "gate 1 is not reading the Fusion panel's tick series"

    def test_the_panel_labels_what_does_not_decide(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("function paintStackWin(")
        # TO THE END OF THE FUNCTION, not a fixed slice. A byte
        # count silently stopped covering the shown-only block
        # the moment gate 2 was added above it, and a label test
        # that cannot see the label it checks is worse than none.
        blk = src[i:src.index("function paintJournal(", i)]
        assert "style.display = 'none'" in blk
        assert "none of this decides" in blk, \
            "the shown-only block is unlabelled"
        assert "GATE 1" in blk and "GATE 2" in blk, \
            "the two gates are not marked as gates"
        assert "it must agree to open" in blk, \
            "the timeframe candle is not marked as a gate"
        # The row that GATES is out of the shown-only block.
        tail = blk[blk.index("none of this decides"):]
        assert "row(st.trading," not in tail, \
            "the deciding row is drawn under SHOWN ONLY"
        assert "row(st.below)" in tail

    def test_the_dropdown_offers_it(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index('id="jnlStrat"')
        box = src[i:src.index("</select>", i)]
        assert 'value="stack"' in box


class TestTheFootprintNeverDisappears:
    """Chris: "is there a way to keep footprint always visible
    because it disappears".

    THREE STATES WERE COLLAPSING INTO ONE BLANK. The panel's guard
    was `J.footprint && J.footprint.levels`, and `levels` was absent
    from two of the three return paths and zero on the third
    whenever a bar had no imbalanced levels. So no footprint stream,
    a bar boundary, and a quiet bar all rendered as nothing at all.

    That mattered little when the footprint was one of four legs. It
    is now the gate, so a reading that can silently not be there is
    a rule that can silently not fire.
    """

    def test_the_panel_has_no_truthiness_guard_left(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        # THE CODE FORM, NOT THE PROSE. A first attempt at this
        # matched the comment that DESCRIBES the old guard and
        # failed against its own documentation.
        assert "+ ((J.footprint && J.footprint.levels)" not in src, \
            "a zero reading is hidden again"
        i = src.index("function paintJournal(")
        blk = src[i:i + 12000]
        assert "' &middot; footprint '" in blk, "it is not drawn at all"

    def test_every_return_path_carries_the_same_shape(self):
        """A missing key reads as absent; a zero reads as a
        reading. They must not look alike."""
        import ast
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("def _footprint_imbalance(")
        j = src.index("def _window_now(", i)
        # The `def` sits at four spaces and its body at eight, so
        # the whole block dedents by four -- not eight, which was
        # the first attempt and left the docstring unindented.
        fn = ast.parse("\n".join(
            ln[4:] if ln.startswith(" " * 4) else ln
            for ln in src[i:j].splitlines())).body[0]
        rets = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
        assert len(rets) >= 4, "paths were removed"
        for r in rets:
            assert isinstance(r.value, ast.Dict)
            keys = {k.value for k in r.value.keys}
            assert "levels" in keys, \
                "a return path omits levels -- it will read as absent"
            assert "side" in keys and "buy" in keys and "sell" in keys

    def test_a_quiet_bar_reports_zero_rather_than_nothing(self):
        import os
        import tempfile

        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient
        import liqmap.web as web

        old = {k: os.environ.get(k) for k in
               ("LIQMAP_DB", "LIQMAP_TOKEN", "LIQMAP_AUTO")}
        os.environ.update(LIQMAP_DB=tempfile.mktemp(suffix=".db"),
                          LIQMAP_TOKEN="fp-token", LIQMAP_AUTO="false")
        web.RT = None
        try:
            with TestClient(web.create_app()) as c:
                c.headers.update({"Authorization": "Bearer fp-token"})
                # No footprint stream attached at all -- the worst
                # of the three states.
                d = c.get("/api/read?coin=ES&interval=5m").json()
                fp = (d.get("journal") or {}).get("footprint")
                assert fp is not None, "the footprint key vanished"
                assert fp.get("levels") == 0
                assert fp.get("side") == ""
                assert fp.get("why"), "a blank with no reason given"
        finally:
            web.RT = None
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


class TestTheRuleDoesNotWaitForTheSlowRoute:
    """Chris: "the auto trader isnt following the rules we set for
    it" and "lagging behind real price".

    MEASURED, NOT INFERRED. /api/read takes ~4,200ms on an empty
    database; /api/fusion takes ~1.3ms. The stack rows were read off
    `out_["cascade"]`, which /api/read builds -- so every decision
    was made on roughly four-second-old rows.

    Worse: the server-side stepper calls the journal step with an
    EMPTY dict. With no browser open the rungs were {}, gate 2 could
    never pass, and this strategy could not open a trade at all.
    """

    def test_an_empty_stack_blocks_again_and_that_is_the_danger(self):
        """THE ORIGINAL BUG IS LIVE AGAIN BY DESIGN, so the test
        that found it stays pointed at it.

        The server-side stepper calls the journal step with an
        EMPTY dict. With the timeframe candle gating entries, {}
        means no trade -- exactly the state it was in when this
        strategy could not open a trade at all with the browser
        closed.

        What stops it is the NEXT test: the step builds its own
        rows from candles rather than waiting for the slow route to
        hand them over. These two belong together and neither is
        sufficient alone.
        """
        assert sw.stack_side({}, 300.0)[0] == ""
        st = sw.state({}, D, {}, 300.0, intent=push())
        assert st["side"] == "", "an empty stack opened a trade"
        assert "has not been read" in st["stack"]["why"]

    def test_the_step_builds_its_own_rows_from_candles(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("def _journal_step_un(")
        blk = src[i:i + 30000]
        assert "_csc.rows_from_candles(" in blk, \
            "the rule still waits for the route to hand it rows"
        assert "_candle_now(rt_, tf)" in blk
        assert '(out_.get("cascade") or {}).get("rungs")' not in blk, \
            "it is reading the slow route's cascade again"

    def test_it_picks_the_stack_from_the_window_length(self):
        """5m window -> the 5m/1m stack; 15m -> 15m/5m."""
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("def _journal_step_un(")
        blk = src[i:i + 30000]
        assert 'float(s.window_s or 900.0) <= 300.0' in blk

    def test_a_failure_building_the_stack_does_not_take_the_step_down(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("_csc.rows_from_candles(")
        blk = src[max(0, i - 1200):i + 900]
        assert "except Exception" in blk, \
            "a stack that cannot build would kill the journal step"


class TestTheRowsShowRealPrices:
    """Chris: "the Gate 2 stack is still reading BPS not true
    price". The direction was already close against open; a bps
    figure is simply not something anyone can check against a
    broker's chart."""

    def test_a_rung_carries_both_prices_and_the_ticks(self):
        from liqmap import cascade as csc
        rows = csc.rows_from_candles("5m", lambda tf: {
            "5m": {"open": 7438.00, "close": 7432.75},
            "1m": {"open": 7434.00, "close": 7432.75}}.get(tf))
        g = {r.timeframe: r for r in csc.read("5m", rows).rungs}["5m"]
        d = g.to_dict()
        assert d["open_px"] == 7438.0 and d["last_px"] == 7432.75
        assert d["ticks"] == -21.0
        assert "7438 -> 7432.75" in g.describe()

    def test_the_ticks_match_the_price_difference(self):
        from liqmap import cascade as csc
        g = csc.Rung("5m", "none", 0.0, True,
                     open_px=7438.0, last_px=7432.75)
        assert g.to_dict()["ticks"] == (7432.75 - 7438.0) / 0.25

    def test_both_panels_show_the_prices(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        assert src.count("chPx(r.open_px)") >= 2, \
            "one of the two stack panels still shows only bps"

    def test_the_state_carries_them_to_the_gate_two_panel(self):
        r = {"5m": {"present": True, "qualified": "down",
                    "open_px": 7438.0, "last_px": 7432.75, "ticks": -21.0},
             "1m": {"present": True, "qualified": "down",
                    "open_px": 7434.0, "last_px": 7432.75, "ticks": -5.0}}
        st = sw.state({}, D, r, 300.0)
        assert st["stack"]["rows"]["5m"]["open_px"] == 7438.0
        assert st["stack"]["rows"]["5m"]["ticks"] == -21.0




class TestOneTickCannotTurnGateOne:
    """Chris: "every trade that im loosing is because of flips ...
    it gets in when both agree then immediately exits for a loss of
    several ticks ... it just exits on its own with no visable
    disagreement at all".

    MEASURED, AND IT WAS A BAR-BIRTH PROBLEM. Gate 1 is the
    5,000-tick candle's body. A bar that has just opened has close
    EQUAL to open, so the very next print decided its direction --
    one tick, after which the gate had turned and the trade was
    closed. The first fifty prints are 1% of a bar's life and
    produced 9% of direction changes on a choppy simulated tape,
    29% trending and 63% in a strong trend: about 170 times the
    per-print rate of the rest of the bar.

    AND IT WAS INVISIBLE, which is the other half of the complaint.
    The flip was over in a handful of prints and the panel had
    repainted before anyone looked."""

    def test_a_single_tick_does_not_turn_it(self):
        assert sw.body_dir(7000.0, 6999.75, U) == U
        assert sw.body_dir(7000.0, 7000.25, D) == D

    def test_the_full_distance_does(self):
        assert sw.body_dir(7000.0, 6999.00, U) == D
        assert sw.body_dir(7000.0, 7001.00, D) == U

    def test_it_is_four_ticks_one_point_on_es(self):
        assert sw.MIN_TURN_TICKS == 4.0
        assert sw.TICK == 0.25

    def test_the_tick_size_matches_the_fill_module(self):
        """Two tick sizes in one engine is a bug waiting for a
        contract change."""
        from liqmap import quote
        assert sw.TICK == quote.TICK

    @pytest.mark.parametrize("ticks", [3.0, 3.9, -3.0, -3.9])
    def test_just_inside_the_threshold_holds(self, ticks):
        assert sw.body_dir(7000.0, 7000.0 + ticks * 0.25, U) == U

    @pytest.mark.parametrize("ticks", [4.0, 4.1, 9.0])
    def test_at_the_threshold_it_turns(self, ticks):
        assert sw.body_dir(7000.0, 7000.0 - ticks * 0.25, U) == D

    def test_it_never_reads_flat(self):
        """Inside the threshold the standing direction is returned,
        NOT an absence. He has rejected a blank gate three times."""
        for raw in (7000.0, 7000.25, 6999.75):
            assert sw.body_dir(7000.0, raw, U) == U
            assert sw.body_dir(7000.0, raw, D) == D

    def test_nothing_is_invented_before_the_first_direction(self):
        assert sw.body_dir(7000.0, 7000.25, "") == ""
        assert sw.body_dir(7000.0, 7001.00, "") == U

    @pytest.mark.parametrize("o,c", [(0, 7000), (7000, 0), (None, None),
                                     ("x", "y")])
    def test_junk_prices_hold_rather_than_turn(self, o, c):
        assert sw.body_dir(o, c, U) == U

    def test_the_distance_is_reported(self):
        assert sw.turn_ticks(7000.0, 7001.00) == 4.0
        assert sw.turn_ticks(7000.0, 6999.00) == -4.0
        assert sw.turn_ticks(0.0, 7000.0) == 0.0

    def test_the_state_applies_it_when_given_the_prices(self):
        st = sw.state({}, D, rungs(**{"5m": U}), 300.0,
                      tick_open=7000.0, tick_close=6999.75,
                      prev={"tick": U, "candle": U})
        assert st["tick"]["raw"] == D, "the candle really did read down"
        assert st["tick"]["side"] == U, "one tick turned the gate"
        assert st["tick"]["held_for_want_of_move"] is True
        assert st["tick"]["moved_ticks"] == -1.0

    def test_the_state_lets_a_real_turn_through(self):
        st = sw.state({}, D, rungs(**{"5m": D}), 300.0,
                      tick_open=7000.0, tick_close=6999.00,
                      prev={"tick": U, "candle": D})
        assert st["tick"]["side"] == D
        assert st["tick"]["held_for_want_of_move"] is False
        assert st["side"] == D

    def test_without_prices_the_old_behaviour_stands(self):
        """A caller that cannot measure a distance must not be
        silently given a rule it is not applying."""
        st = sw.state({}, D, rungs(**{"5m": D}), 300.0,
                      prev={"tick": U, "candle": D})
        assert st["tick"]["side"] == D

    def test_the_route_hands_over_the_candles_prices(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("fus = _sw.state(win, _bd, _rungs,")
        blk = src[i:i + 700]
        assert 'tick_open=float(_tk.get("open")' in blk, \
            "no open, so there is no distance to measure"
        assert 'tick_close=float(_tk.get("close")' in blk

    def test_the_panel_shows_the_distance_and_the_hold(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("function paintStackWin(")
        blk = src[i:src.index("function paintJournal(", i)]
        assert "f.moved_ticks" in blk, \
            "the distance from the open is invisible again"
        assert "f.min_turn_ticks" in blk and "to turn" in blk
        assert "f.held_for_want_of_move" in blk, \
            "a withheld turn is silent -- the exact complaint"


class TestARefusedEntryIsNeverSilent:
    """Chris: "on some trades tick bar and 5 or 15 min can actually
    be up and no trade it taken like its invisible".

    IT WAS INVISIBLE. `idle` -- the journal's reason for not opening
    -- is drawn by the JOURNAL painter. The stack+window box, the
    one showing the two gates, said TRADE UP and never mentioned
    that the cap had refused it. Two boxes, one of them telling half
    the story, which is the fault this panel has now had three
    times in three different forms."""

    def test_the_route_carries_the_refusal_into_the_gate_payload(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index('"stackwin": (')
        blk = src[i:i + 400]
        assert '"blocked_by": idle_why' in blk, \
            "the journal's refusal never reaches the gate panel"

    def test_the_refusal_is_set_whenever_nothing_opens(self):
        """`idle_why` is blank only when a trade actually opened or
        closed, so a silent no-trade is not reachable."""
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("def _journal_step_un(")
        blk = src[i:i + 30000]
        assert blk.count('idle_why = arg if act == "" else ""') >= 2, \
            "a path through decide leaves the reason unset"

    def test_the_panel_draws_it_under_the_verdict(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("function paintStackWin(")
        blk = src[i:src.index("function paintJournal(", i)]
        assert "SW.blocked_by" in blk, \
            "the gate panel still cannot say why nothing opened"
        assert "NO TRADE" in blk

    def test_the_journal_always_gives_a_reason(self):
        """The policy end of it: `decide` never returns an empty
        reason when it declines."""
        from liqmap import journal as jn
        s_ = jn.Settings()
        s_.auto = True
        act, why = jn.decide(None, s_, 7000.0, "up", ("up",),
                             window_rule=True, max_per_bar=2)
        assert act == "" and why, "it declined without saying why"
        assert "already taken its up" in why
