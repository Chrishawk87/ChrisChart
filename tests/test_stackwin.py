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
    def test_a_disagreeing_candle_blocks_it(self, tick, candle):
        side, why = sw.entry_side(tick, candle)
        assert side == ""
        assert f"the tick candle is {tick}" in why
        assert f"the timeframe candle is {candle}" in why

    def test_no_candle_direction_blocks_it(self):
        side, why = sw.entry_side(U, "")
        assert side == ""
        assert "timeframe candle has no direction" in why


class TestVolumeNoLongerTurnsATradeRound:
    """IT DID, FOR ONE REVISION. A volume-backed pullback was traded
    in the retrace's own direction. Measured against the timeframe
    gate it fired on 3% of entries in chop and NONE in a trend,
    because a short inside an up tick bar needs the timeframe down
    as well -- so it was a special case wearing a rule's clothes.

    It is back to being the confirmation he first asked for: "we are
    going to use volume to confirm tick bars continuation or pull
    back". Confirming, and not deciding."""

    @pytest.mark.parametrize("heavy", [True, False])
    def test_the_side_is_the_same_either_way(self, heavy):
        intent = {"state": "pullback", "heavy": heavy, "depth": 0.5}
        assert sw.entry_side(U, U, intent) == (U, "")

    def test_it_never_returns_the_opposite_side(self):
        for tick in (U, D):
            for st in ("pullback", "continuation", ""):
                for heavy in (True, False):
                    side, _ = sw.entry_side(
                        tick, tick, {"state": st, "heavy": heavy})
                    assert side == tick, (tick, st, heavy)

    def test_the_state_agrees(self):
        a = sw.state({}, U, rungs(**{"5m": U}), 300.0,
                     intent={"state": "pullback", "heavy": True})
        b = sw.state({}, U, rungs(**{"5m": U}), 300.0,
                     intent={"state": "pullback", "heavy": False})
        assert a["side"] == b["side"] == U

    def test_the_reading_is_still_carried_and_still_shown(self):
        """Removed from the decision, not from the screen."""
        st = sw.state({}, U, rungs(**{"5m": U}), 300.0,
                      intent={"state": "pullback", "heavy": True,
                              "ratio": 1.9})
        assert st["intent"]["heavy"] is True
        assert st["intent"]["decides"] is False
        assert "heavy" in st["say"] and "does not decide" in st["say"]


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
        for gone in ("exit_reason", "expired", "bar_of"):
            assert not hasattr(sw, gone), f"{gone} is still here"

    def test_the_route_has_no_clock_left(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        assert "TRADE_BAR_S = " not in src, \
            "the five-minute constant is still defined"
        assert "_sw.exit_reason(" not in src

    def test_the_trade_is_not_stamped_with_a_bar(self):
        """THE CAP THAT WAS NEVER MEANT TO BE THERE. The stamp fed
        `sides_in_bar`, and `decide` turns a 0 cap into
        MAX_PER_BAR -- so `agree in taken` blocked a second trade
        the same way inside one 5-minute bar while the comment
        above it read NO CAP."""
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index('if strat == "stack":', src.index("cap = 0"))
        blk = src[i:i + 1400]
        assert "bar_ts = 0.0" in blk, "the trade is stamped again"
        assert "bar_of(" not in blk, "a clock bar is back in the stamp"


class TestTheExitIsTheTickBarAlone:
    """Chris: "we will no longer use time as a entry point or exit
    at all just what the tick bar is doing alone". Putting the entry
    gate back does not take that back -- the timeframe candle gets
    you in, the tick bar keeps you in."""

    def test_it_holds_while_the_tick_bar_agrees(self):
        assert sw.holds(D, D) == (True, "")

    def test_the_opposite_direction_ends_it(self):
        ok, why = sw.holds(D, U)
        assert ok is False
        assert "flipped up against your down trade" in why

    def test_FLAT_IS_NOT_A_FLIP(self):
        assert sw.holds(D, "") == (True, "")

    def test_no_open_side_is_never_an_exit(self):
        assert sw.holds("", U) == (True, "")

    def test_the_timeframe_candle_cannot_close_a_trade(self):
        import inspect
        names = list(inspect.signature(sw.holds).parameters)
        assert names == ["side", "tick"], names

    def test_the_route_reads_the_same_number_the_entry_read(self):
        """THE BUG THIS REPLACES. It called `exit_reason` with
        `dir`, the tick bar's VOTE, while the entry took
        `body_dir`, the candle -- they can point opposite ways on
        one bar, so the engine opened a long and closed it on the
        same poll."""
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index('elif strat == "stack" and t is not None and t.open:')
        blk = src[i:i + 1600]
        assert "_sw.holds(" in blk, "nothing ends a stack trade"
        assert "_sw.hold_dir(_bd," in blk, \
            "the exit is reading something other than the candle"
        assert '_tk_x' not in blk, "a second tick reading is back"


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


