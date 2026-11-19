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
        assert f"the tick candle is {tick}" in why
        assert f"the timeframe candle is {candle}" in why

    def test_no_candle_direction_blocks_it(self):
        side, why = sw.entry_side(U, "")
        assert side == ""
        assert "timeframe candle has no direction" in why


class TestThePullbackReadingDoesNotVote:
    """Chris, restoring the rule: "Nothing else votes. The
    footprint, OFI, CVD, the window's four legs, the 1m row, the
    vote, the pace, the pullback reading -- all computed, all
    drawn, none of them gate anything."

    It blocked entries for one revision -- a tick chart whose own
    push had gone was refused -- and he has taken that back by
    name. It is still computed and still on the panel."""

    DEAD = {"state": "pullback", "depth": 0.90, "heavy": True}
    ALIVE = {"state": "continuation", "depth": 0.0, "heavy": False}

    def test_a_dead_push_no_longer_blocks_an_entry(self):
        st = sw.state({}, U, rungs(**{"5m": U}), 300.0, tf_ticks=20.0,
                      intent=self.DEAD)
        assert st["side"] == U

    def test_it_makes_no_difference_either_way(self):
        a = sw.state({}, U, rungs(**{"5m": U}), 300.0, tf_ticks=20.0,
                     intent=self.DEAD)
        b = sw.state({}, U, rungs(**{"5m": U}), 300.0, tf_ticks=20.0,
                     intent=self.ALIVE)
        assert a["side"] == b["side"] == U
        assert a["why"] == b["why"]

    def test_entry_side_ignores_the_flag_outright(self):
        assert sw.entry_side(U, U, 20.0, tick_failed=True) == (U, "")

    def test_it_is_still_computed_and_drawn(self):
        st = sw.state({}, U, rungs(**{"5m": U}), 300.0, tf_ticks=20.0,
                      intent=self.DEAD)
        assert st["intent"]["state"] == "pullback"
        assert st["regime"]["where"]


class TestWhatIsBackAndWhatStaysOut:
    """Each in the positive or the negative, because every one of
    these has been added, removed and re-added in this file and the
    tests are the only record of which way round it is now."""

    def test_the_hold_is_back_at_five_seconds(self):
        assert sw.DWELL_S == 5.0
        assert hasattr(sw, "Dwell")

    def test_the_route_tracks_it(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("def _journal_step_un(")
        blk = src[i:i + 30000]
        assert "_sw.Dwell()" in blk, "nothing measures the hold"
        assert "_dw.on_tick(_sw.hold_dir(_bd," in blk, \
            "the hold is fed a reading that blanks, so it restarts " \
            "on an absence and never elapses"
        assert "held_s=_dw.held_s(" in blk, "the rule never sees it"

    def test_the_time_stop_is_back(self):
        assert sw.TRADE_BAR_S == 300.0
        assert hasattr(sw, "expired")
        assert hasattr(sw, "exit_reason")

    def test_the_four_tick_minimum_is_out(self):
        """Chris: gate 2 is "Held through the instant close sits on
        open" -- a tie and nothing wider. The hold debounces it
        now."""
        import inspect
        src = inspect.getsource(sw.state)
        assert "body_dir(" not in src, \
            "the four-tick minimum still gates gate 2"
        assert "hold_dir(tick_raw" in src

    def test_the_cap_is_still_two_a_five_minute_bar(self):
        assert sw.MAX_TRADES_PER_BAR == 2
        assert sw.CAP_BAR_S == 300.0

    def test_the_trade_is_stamped_so_both_the_cap_and_the_clock_work(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index('if strat == "stack":', src.index("cap = 0"))
        blk = src[i:i + 1800]
        assert "_swb.bar_of(time.time(), _swb.CAP_BAR_S)" in blk, \
            "unstamped: the cap cannot count and nothing expires"
        assert "cap = _swb.MAX_TRADES_PER_BAR" in blk


class TestTheExitIsTheClockThenTheTickBar:
    """Chris, restoring the rule, in his order:

        1. Stop loss   2. Take profit
        3. Time stop -- "the trade has outlived the 5-minute bar it
           opened in. Checked before direction, so the log says the
           bar ended rather than blaming the tick bar."
        4. Tick bar flip against the position.

    THE TIMEFRAME IS NOT ON THAT LIST. It closed trades for one
    revision; it gets you in, gate 2 takes you out."""

    NOW = 1_700_000_000.0

    def test_a_healthy_trade_holds(self):
        b = sw.bar_of(self.NOW, sw.TRADE_BAR_S)
        assert sw.exit_reason(D, D, b, b + 60.0) == ""

    def test_a_flipped_tick_bar_ends_it(self):
        b = sw.bar_of(self.NOW, sw.TRADE_BAR_S)
        why = sw.exit_reason(D, U, b, b + 60.0)
        assert "tick bar flipped up against your down trade" in why

    def test_the_bar_ending_ends_it(self):
        b = sw.bar_of(self.NOW, sw.TRADE_BAR_S)
        why = sw.exit_reason(D, D, b, b + 301.0)
        assert "5-minute bar this trade opened in has ended" in why

    def test_the_clock_is_blamed_when_both_are_true(self):
        b = sw.bar_of(self.NOW, sw.TRADE_BAR_S)
        why = sw.exit_reason(D, U, b, b + 301.0)
        assert "has ended" in why, "it blamed the tick bar"

    def test_a_flat_tick_bar_inside_the_bar_holds(self):
        """FLAT IS NOT A FLIP. The series can be flat at the same
        instant a bar ends, and an exit that read an absence as a
        reversal is the churn this has been rewritten to remove."""
        b = sw.bar_of(self.NOW, sw.TRADE_BAR_S)
        assert sw.exit_reason(D, "", b, b + 60.0) == ""

    def test_a_trade_with_no_bar_never_expires(self):
        """Zero is "we do not know". Closing every such position at
        the first boundary is acting on missing data."""
        assert sw.exit_reason(D, D, 0.0, self.NOW) == ""

    def test_no_open_side_is_never_an_exit(self):
        assert sw.exit_reason("", U, 0.0, self.NOW) == ""

    def test_the_timeframe_cannot_close_a_trade(self):
        import inspect
        assert list(inspect.signature(sw.holds).parameters) == \
            ["side", "tick"]

    def test_the_route_delegates_to_it(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index('elif strat == "stack" and t is not None and t.open:')
        blk = src[i:i + 2600]
        assert "_sw.exit_reason(" in blk, "the exit is not wired"
        assert "_sw.TRADE_BAR_S" in blk, "no clock reaches the exit"
        assert '(fus.get("tick") or {}).get("side")' in blk, \
            "the exit recomputes gate 2 instead of reading it"


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
        # From the STATE call, not from wherever `_tf_name` is now
        # assigned -- that moved when the gate-1 sizing was added
        # above it, and a fixed slice from the wrong anchor stopped
        # covering the lines it names.
        i = src.index("fus = _sw.state(win, _bd, _rungs,")
        blk = src[max(0, i - 900):i + 900]
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

    def test_the_exit_is_not_handed_a_pullback_flag(self):
        """The pullback reading does not gate and does not close.
        Pinned in the negative -- it has been wired in twice."""
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index('elif strat == "stack" and t is not None and t.open:')
        blk = src[i:i + 2000]
        assert "failed=" not in blk

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

    def test_the_stamp_serves_the_cap_AND_the_clock(self):
        """ONE STAMP, TWO JOBS, and they are the same five minutes.
        It counts the trades in this bar for the cap, and it is the
        bar the time stop measures against -- Chris restored that
        exit by name after it had been removed."""
        assert sw.CAP_BAR_S == sw.TRADE_BAR_S == 300.0
        b = sw.bar_of(1_700_000_000.0, sw.CAP_BAR_S)
        assert sw.exit_reason(D, D, b, b + 299.0) == ""
        assert "has ended" in sw.exit_reason(D, D, b, b + 301.0)


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
        assert "_sw.exit_reason(" in blk, \
            "the clock-then-tick-bar exit is not wired"
        assert "intent=_tape_intent(" in blk, \
            "the volume reading never reaches the rule"
        assert '_tk = _tick_now(rt_, _g1, "raw", 0.0, 0, FUSION_RESET_S)' \
            in blk, "gate 1 is not reading the continuous tick series"
        assert 'getattr(s, "tick_bar", 0)' in blk, \
            "gate 2's chart is derived again instead of picked"

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
        assert "jn.window_label(float(s.window_s or 900.0))" in blk, \
            "the traded chart is named in more than one place"

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




class TestTheFourTickMinimumIsOut:
    """It was added to stop one-tick flips at a bar's birth, and it
    worked -- direction changes an hour fell 359 to 86 on a choppy
    simulated tape. Chris has taken it back out by name: gate 2 is
    "Held through the instant close sits on open", a tie and nothing
    wider, and the five-second hold is what debounces it now.

    `body_dir` and `turn_ticks` survive because the DISTANCE from
    the open is still worth drawing; only the gating is gone."""

    def test_gate_two_turns_on_one_tick_again(self):
        st = sw.state({}, D, rungs(**{"5m": D}), 300.0, tf_ticks=20.0,
                      tick_open=7000.0, tick_close=6999.75,
                      prev={"tick": U, "candle": D})
        assert st["tick"]["side"] == D, "a minimum is still gating it"

    def test_it_still_holds_through_an_exact_tie(self):
        st = sw.state({}, "", rungs(**{"5m": U}), 300.0, tf_ticks=20.0,
                      tick_open=7000.0, tick_close=7000.0,
                      prev={"tick": U, "candle": U})
        assert st["tick"]["side"] == U
        assert st["tick"]["held_through_flat"] is True

    def test_the_distance_is_still_reported(self):
        st = sw.state({}, U, rungs(**{"5m": U}), 300.0, tf_ticks=20.0,
                      tick_open=7000.0, tick_close=7000.75)
        assert st["tick"]["moved_ticks"] == 3.0

    def test_the_hold_is_what_debounces_it_now(self):
        """The flip gets through to the DIRECTION immediately and is
        stopped from becoming a TRADE until it has stood five
        seconds. That is the trade Chris asked for."""
        st = sw.state({}, D, rungs(**{"5m": D}), 300.0, tf_ticks=20.0,
                      tick_open=7000.0, tick_close=6999.75,
                      held_s=1.0, prev={"tick": U, "candle": D})
        assert st["tick"]["side"] == D
        assert st["side"] == "", "a one-second-old flip opened a trade"
        assert "has to hold 5s" in st["why"]


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


class TestATickChartCanBeTheTradedTimeframe:
    """Chris: "what if we traded the tick chart instead of the time
    chart ... reading the tick chart the same way we would read the
    forming 5000t chart the same way it would for a forming 5 minute
    chart ... a tick chart is really no different then a time chart
    only difference is a tick chart prints every other 5000ts".

    HE IS RIGHT, AND THAT IS WHY THIS IS SMALL. Everything
    downstream of the candle wants an open, a high, a low, a close
    and a volume. None of it cares whether the bar was cut by a
    clock or by a count, so one reader learned to return a tick
    bar's forming candle and the rule was left alone."""

    def test_a_negative_window_names_a_tick_chart(self):
        assert sw.traded_tf(-5000.0) == "5000t"
        assert sw.traded_tf(-10000.0) == "10000t"

    def test_the_time_windows_are_untouched(self):
        assert sw.traded_tf(300.0) == "5m"
        assert sw.traded_tf(900.0) == "15m"
        assert sw.traded_tf(600.0) == "15m"

    def test_the_journal_label_agrees_with_the_rule(self):
        """Two names for one chart is how the gate, the row and the
        panel end up disagreeing about what is being traded."""
        from liqmap import journal as jn
        for w in (300.0, 900.0, -5000.0, -10000.0):
            assert jn.window_label(w) == sw.traded_tf(w), w

    def test_a_tick_window_survives_being_saved(self, tmp_path):
        """BEHAVIOURAL, because the source check was not enough.

        `window_s` is a REAL column that was clamped to 300 or 900
        on EVERY save, so a tick size would have been erased the
        first time the row was written -- silently, and the rule
        would have gone back to trading the 15m without saying so.

        A mutation that kept the new branch and returned 900.0 from
        it survived a test that only read the source."""
        from liqmap.journal import Journal, Settings
        j = Journal(tmp_path / "j.db")
        j.save_settings(Settings(coin="ES", window_s=-5000.0))
        assert j.settings("ES").window_s == -5000.0, \
            "the clamp destroyed the tick window"
        assert sw.traded_tf(j.settings("ES").window_s) == "5000t"
        # And the time windows still clamp, which is why the clamp
        # is there at all.
        j.save_settings(Settings(coin="ES", window_s=301.0))
        assert j.settings("ES").window_s == 900.0

    def test_the_rule_runs_on_a_tick_row_exactly_as_on_a_time_row(self):
        r = {"5000t": {"present": True, "qualified": U, "bps": 3.0}}
        st = sw.state({}, U, r, -5000.0, tf_ticks=20.0,
                      intent={"state": "continuation", "heavy": False},
                      tf_intent={"depth": 0.1, "heavy": False})
        assert st["stack"]["trading"] == "5000t"
        assert st["regime"]["where"] == "continuation"
        assert st["side"] == U

    def test_the_same_refusals_apply(self):
        r = {"5000t": {"present": True, "qualified": U, "bps": 3.0}}
        # sitting on its open
        assert sw.state({}, U, r, -5000.0, tf_ticks=1.0)["side"] == ""
        # pulling back against it
        assert sw.state({}, D, r, -5000.0, tf_ticks=20.0)["side"] == ""

    def test_a_tick_window_has_no_row_below_it(self):
        """`STACKS` is a map of TIME rows -- 15m takes 5m takes 1m.
        A tick chart is not in it, and the row below is simply
        absent, which the panel already knows how to draw."""
        r = {"5000t": {"present": True, "qualified": U, "bps": 3.0}}
        st = sw.state({}, U, r, -5000.0, tf_ticks=20.0)
        assert st["stack"]["below"] == ""
        assert "5000t" in st["stack"]["rows"]

    def test_gate_one_is_always_faster_than_the_traded_chart(self):
        """THE DEGENERATE CASE. Both sides reading one size is the
        identical number agreeing with itself, and gate 2 becomes
        decoration -- the same fault as asking a 5m window to agree
        with a 5m row."""
        from liqmap.tickbar import PER_BAR_CHOICES as P
        for traded in (1000, 2000, 5000, 10000):
            want = max(1, traded // 5)
            g1 = min((n for n in P if n < traded),
                     key=lambda n: abs(n - want), default=want)
            assert g1 < traded, (traded, g1)

    def test_the_route_breaks_a_size_clash(self):
        """If gate 1 is itself a tick chart of the size gate 2 was
        set to, the two sides are the identical number agreeing with
        itself and gate 2 is decoration."""
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("_traded_n = _pi3(_tf_name)")
        blk = src[i:i + 1200]
        assert "_g2_clash" in blk, "a size clash is not detected"
        assert "n < int(_traded_n)" in blk, \
            "the clash is detected and not broken"

    def test_the_candle_reader_serves_both_kinds(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("def _candle_now(")
        blk = src[i:src.index("\n    def ", i + 10)]
        assert "_n = _pi(interval)" in blk, \
            "a tick interval is never parsed, so the branch is dead"
        assert "if _n is not None:" in blk
        assert 'getattr(tape, "tickbars", None)' in blk, \
            "it does not read the continuous tick series"
        assert "tickbars_timed" not in blk, \
            "it reads the clock-cut family, putting the clock back " \
            "into a rule chosen to escape it"
        # THE VALUE, NOT THE KEY. A mutation that zeroed all three
        # on the tick branch survived a test that counted keys --
        # the rule then read a retrace off a candle with no high,
        # which is a pullback that can never fail.
        for k in ("high", "low", "volume"):
            assert blk.count(f'"{k}": float(getattr(c, "{k}"') == 2, \
                f"{k} is not read off the bar for both kinds"

    def test_the_stack_row_follows_the_window(self):
        """A hardcoded name here would draw a 15m row while the rule
        traded a tick chart -- the screen and the rule describing
        different charts, which this panel has done three times."""
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("_tf_for = ")
        blk = src[i:i + 900]
        assert "jn.window_label(" in blk
        assert '_tf_for = "15m"' not in blk and '_tf_for = "5m"' not in blk

    def test_the_selector_offers_them(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index('id="jnlWin"')
        blk = src[i:i + 1200]
        assert 'value="-5000"' in blk and 'value="-10000"' in blk
        assert 'value="300"' in blk and 'value="900"' in blk, \
            "the time windows were dropped"



class TestOneTradePerAgreement:
    """Chris: "take a trade when timeframe and Tick chart match 1
    trade per agreement ... this should kill most of the
    disagreements and multi entries".

    An agreement is an UNBROKEN RUN of the two charts pointing the
    same way. Once it has produced a trade it produces no more --
    they have to stop agreeing before a fresh one can be taken."""

    UP = {"state": "continuation", "heavy": False}

    def _step(self, prev, tick=U, candle=U, took=False):
        st = sw.state({}, tick, rungs(**{"5m": candle}), 300.0,
                      tf_ticks=20.0, intent=self.UP, prev=prev)
        nxt = dict(st["held"])
        if took:
            nxt["episode"] = {**nxt["episode"], "taken": True}
        return st, nxt

    def test_the_first_agreement_trades(self):
        st, _ = self._step({})
        assert st["side"] == U

    def test_the_same_agreement_does_not_trade_twice(self):
        _, prev = self._step({}, took=True)
        st, _ = self._step(prev)
        assert st["side"] == ""
        assert "already been traded" in st["why"]

    def test_a_break_starts_a_fresh_agreement(self):
        _, prev = self._step({}, took=True)
        _, prev = self._step(prev, tick=D)          # pullback
        st, _ = self._step(prev)
        assert st["side"] == U, "a fresh agreement was refused"

    def test_a_flip_to_the_other_side_is_a_fresh_agreement(self):
        _, prev = self._step({}, took=True)
        st, _ = self._step(prev, tick=D, candle=D)
        assert st["side"] == D

    def test_an_agreement_that_never_traded_is_not_spent(self):
        """THE CASE THAT MAKES THE CALLER OWN THE FLAG. The rule
        offering a side is not the journal accepting it -- the cap,
        or AUTO being off, refuses entries the rule offered -- and
        an agreement that produced no trade must stay live."""
        _, prev = self._step({}, took=False)
        st, _ = self._step(prev)
        assert st["side"] == U

    def test_the_run_is_reported(self):
        st, _ = self._step({})
        assert st["episode"]["agreeing"] is True
        assert st["episode"]["side"] == U
        assert st["episode"]["decides"] is True

    def test_a_pullback_is_not_a_run(self):
        st, _ = self._step({}, tick=D)
        assert st["episode"]["agreeing"] is False
        assert st["episode"]["side"] == ""

    def test_the_route_marks_it_only_when_a_trade_opens(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        # INSIDE THE JOURNAL STEP. There is another
        # `if act == "open":` in a hand-trading route above, and a
        # whole-file index lands on that one -- the same way an
        # earlier route test passed while the thing it named was
        # untouched.
        i = src.index("def _journal_step_un(")
        blk = src[i:i + 30000]
        j = blk.index('if act == "open":')
        seg = blk[j:j + 1400]
        assert '"taken": True' in seg, \
            "an agreement is never marked spent, so it trades forever"
        assert "j.open_trade(" in seg


class TestGateTwosChartIsPicked:
    """Chris: "i want to add a few tick profiles for gate number 2
    250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7k, 8k, 9,
    and 10k"."""

    WANTED = (250, 500, 1000, 1500, 2000, 3000, 4000, 5000,
              6000, 7000, 8000, 9000, 10000)

    def test_every_size_he_asked_for_exists(self):
        from liqmap.tickbar import PER_BAR_CHOICES as P
        assert tuple(sorted(P)) == self.WANTED

    def test_the_settings_carry_it(self):
        from liqmap.journal import Settings
        assert Settings().tick_bar == 5000
        assert "tick_bar" in Settings().to_dict()

    def test_it_survives_being_saved(self, tmp_path):
        from liqmap.journal import Journal, Settings
        j = Journal(tmp_path / "j.db")
        j.save_settings(Settings(coin="ES", tick_bar=1500))
        assert j.settings("ES").tick_bar == 1500

    def test_the_selector_offers_all_of_them(self):
        import pathlib
        import re
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index('id="jnlTick"')
        box = src[i:src.index("</select>", i)]
        shown = tuple(sorted(int(m) for m in
                             re.findall(r'value="(\d+)"', box)))
        assert shown == self.WANTED

    def test_the_route_reads_the_setting(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("_traded_n = _pi3(_tf_name)")
        blk = src[i:i + 1200]
        assert 'getattr(s, "tick_bar", 0)' in blk
        assert "_g1 not in _PBC2" in blk, \
            "a size with no builder behind it is not snapped"


class TestGateOneIsALiveTimeframe:
    """Chris: "replace gate ones old tick bar with live timeframes
    that i can select between 1m, 5m and 15minute this has to be
    live timeframe"."""

    def test_one_minute_is_a_window(self):
        from liqmap import journal as jn
        assert jn.window_label(60.0) == "1m"
        assert sw.traded_tf(60.0) == "1m"

    def test_the_clamp_keeps_it(self):
        """It forced every value to 300 or 900, so a 1m window would
        have come back as 5m on its first save -- the same way a
        tick window came back as 15m."""
        from liqmap.journal import _clamp_window
        assert _clamp_window(60.0) == 60.0
        assert _clamp_window(300.0) == 300.0
        assert _clamp_window(900.0) == 900.0

    def test_junk_is_not_a_one_minute_window(self):
        """Adding the 1m choice turned every zero into a 1m row --
        the shortest, busiest chart there is, reached by accident."""
        for bad in (0, None, "", "x"):
            assert sw.traded_tf(bad) == ""

    def test_a_one_minute_window_survives_a_save(self, tmp_path):
        from liqmap.journal import Journal, Settings
        j = Journal(tmp_path / "j.db")
        j.save_settings(Settings(coin="ES", window_s=60.0))
        assert j.settings("ES").window_s == 60.0
        assert sw.traded_tf(j.settings("ES").window_s) == "1m"

    def test_the_selector_offers_it(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index('id="jnlWin"')
        box = src[i:src.index("</select>", i)]
        for v in ('value="60"', 'value="300"', 'value="900"'):
            assert v in box, v

    def test_the_rule_runs_on_a_one_minute_row(self):
        r = {"1m": {"present": True, "qualified": U, "bps": 3.0}}
        st = sw.state({}, U, r, 60.0, tf_ticks=20.0,
                      intent={"state": "continuation", "heavy": False})
        assert st["stack"]["trading"] == "1m"
        assert st["side"] == U

    def test_the_timeframe_is_still_the_trend(self):
        """The slots swapped; the jobs did not. The timeframe closes
        the trade and the tick chart never does."""
        assert sw.holds(U, U) == (True, "")
        assert sw.holds(U, D)[0] is False             # the tick bar


# ---------------------------------------------------------------------
# THE STEP HAS TO ACTUALLY RUN.
#
# Everything above this line reads the route as TEXT. That is useful
# for pinning which number a rule reads, and it is blind to the one
# class of fault that has now shipped three times in this file:
#
#     FUSION_RESET_S  -- a constant used as a default argument before
#                        the line that defined it
#     MAX_PER_BAR     -- the same, in `journal.decide`
#     _tf_name        -- assigned thirty lines BELOW the gate that
#                        read it, so every step raised
#                        UnboundLocalError the moment a tick window
#                        was selected
#
# None of those is visible to a substring check and every one of them
# dies on the first execution. `/api/read` catches the exception and
# puts it in `journal.error`, so nothing crashes loudly -- which is
# exactly why the third one reached Chris's screen.
# ---------------------------------------------------------------------

from tests.test_cme_panels import _cme, client  # noqa: E402,F401

AUTH = {"Authorization": "Bearer t0ken"}


class TestEveryWindowSettingActuallySteps:
    @pytest.mark.parametrize("window_s,label", [
        (300.0, "5m"), (900.0, "15m"),
        (-5000.0, "5000t"), (-10000.0, "10000t"),
    ])
    def test_the_journal_step_runs_without_raising(
            self, client, monkeypatch, window_s, label):
        from liqmap import web
        _cme(monkeypatch)
        rt = web.runtime()
        s = rt.journal.settings("ES")
        s.strategy = "stack"
        s.window_s = window_s
        rt.journal.save_settings(s)
        assert rt.journal.settings("ES").window_s == window_s, \
            "the window did not survive the save"

        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH).json()
        jr = d.get("journal") or {}
        assert "error" not in jr, (
            f"the step raised on a {label} window: {jr.get('error')}")

    def test_a_tick_window_reaches_the_gates(self, client, monkeypatch):
        """Not merely "it did not raise": the payload has to describe
        the tick chart, or the window silently did nothing."""
        from liqmap import web
        _cme(monkeypatch)
        rt = web.runtime()
        s = rt.journal.settings("ES")
        s.strategy = "stack"
        s.window_s = -5000.0
        rt.journal.save_settings(s)

        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH).json()
        sw_ = ((d.get("journal") or {}).get("stackwin")) or {}
        assert sw_, "the stack payload is missing entirely"
        assert (sw_.get("stack") or {}).get("trading") == "5000t", \
            "the rule is still trading a time chart"
