"""The footprint gates, then the stack confirms. In that order.

Chris:

    "we are just going to use the stack and the footprint thats all
    the make entry and exits you are going to auto close trades the
    minute they disagree because right now its holding the trades
    for no reason"

    "on a 5 minute the 5 min and 1 minute must agree ... and on the
    15 min the 15 and 5 must agree"

He first watched all four of candle/OFI/CVD/footprint read DOWN and
no trade follow. Nothing was broken: NO STRATEGY ASKED FOR ALL
FOUR. `ofi` wants three, `footprint` wants two, and the two
intrabar rules return before the window is read. That became this
strategy -- and then the four legs became two, because two readings
he trusts beat four he has to reconcile.

The candle, OFI and CVD are still read and still shown. They decide
nothing, and a test below holds them to that.
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


class TestBothMustAgreeToOpen:
    """Chris: "the tick bar should be a continuous direction tracking
    with the timeframe candle tracing at the same time thats how im
    going to be making trades".

    THE CANDLE IS A GATE AGAIN. It was one, he removed it, and he
    has now put it back -- so these tests are the previous rule's
    tests rewritten rather than deleted, and the pair below pins the
    reversal itself so nobody re-derives the old behaviour from the
    old docstrings.

    EXIT IS UNCHANGED. The tick bar flipping, or the time stop. The
    candle has no say in closing, which is tested in its own class.
    """

    @pytest.mark.parametrize("d", [U, D])
    def test_agreement_is_an_entry(self, d):
        assert sw.entry_side(d, rungs(**{"5m": d}), 300.0) == (d, "")

    def test_no_direction_is_no_entry(self):
        side, why = sw.entry_side("", rungs(**{"5m": U}), 300.0)
        assert side == "" and "tick bar has no direction" in why

    @pytest.mark.parametrize("junk", [None, "flat", "UP", 7])
    def test_junk_is_never_a_side(self, junk):
        assert sw.entry_side(junk, rungs(**{"5m": U}), 300.0)[0] == ""

    @pytest.mark.parametrize("tick,candle", [(U, D), (D, U)])
    def test_a_disagreeing_candle_blocks_it(self, tick, candle):
        side, why = sw.entry_side(tick, rungs(**{"5m": candle}), 300.0)
        assert side == ""
        assert f"the tick bar is {tick}" in why
        assert f"the 5m candle is {candle}" in why

    def test_a_flat_candle_blocks_it_too(self):
        """Flat is not a side, so it is not agreement. A window with
        nothing in it would otherwise read as the strongest there
        is -- the rule this module states in its own header."""
        side, why = sw.entry_side(U, rungs(**{"5m": "flat"}), 300.0)
        assert side == "" and "the 5m is flat" in why

    def test_an_unread_candle_blocks_it(self):
        side, why = sw.entry_side(U, rungs(**{"5m": None}), 300.0)
        assert side == "" and "has not been read" in why

    def test_no_stack_at_all_blocks_it(self):
        """A caller that cannot supply the candle must not trade on
        the tick bar alone. It used to, because the argument was
        accepted and ignored."""
        assert sw.entry_side(U, {}, 300.0)[0] == ""
        assert sw.entry_side(U, None, 300.0)[0] == ""

    def test_the_row_below_still_gates_nothing(self):
        """Chris: "paying no attention to the 1 minute at all". The
        candle coming back is the TRADED row only."""
        for below in (U, D, None, "conflict"):
            r = rungs(**{"5m": D, "1m": below})
            assert sw.entry_side(D, r, 300.0) == ("down", ""), below

    def test_fifteen_minutes_gates_on_the_15m(self):
        assert sw.entry_side(D, rungs(**{"15m": D, "5m": U}),
                             900.0) == ("down", "")
        assert sw.entry_side(D, rungs(**{"15m": U, "5m": D}),
                             900.0)[0] == ""

    def test_the_state_marks_the_stack_as_deciding(self):
        """THE SCREEN HAS TO SAY WHICH THINGS DECIDE. The panel
        reads these flags to choose whether to draw the row as a
        gate or under "shown only", and a live gate drawn as
        scenery is the exact fault this panel was rebuilt after."""
        r = rungs(**{"5m": U, "1m": U})
        st = sw.state({"mid_dir": U, "ofi_dir": U, "cvd_dir": U}, U,
                      r, 300.0, fp=U)
        assert st["side"] == "up"
        assert st["stack"]["decides"] is True
        assert st["stack"]["asked"] is True
        assert st["stack"]["agrees"] is True
        # The four window legs did NOT come back with it.
        assert st["context"]["decides"] is False

    def test_the_state_blocks_when_they_disagree(self):
        r = rungs(**{"5m": U, "1m": U})
        st = sw.state({}, D, r, 300.0)
        assert st["side"] == ""
        assert st["stack"]["agrees"] is False
        assert "the 5m candle is up" in st["why"]

    def test_the_stack_is_still_reported(self):
        r = rungs(**{"5m": U, "1m": D})
        st = sw.state({}, D, r, 300.0)
        assert st["stack"]["rows"]["5m"]["dir"] == "up"
        assert st["stack"]["side"] == "up"

    def test_the_verdict_names_both(self):
        st = sw.state({}, U, rungs(**{"5m": U}), 300.0)
        assert "tick bar" in st["say"] and "5m candle" in st["say"]

    def test_the_footprint_reads_exactly_as_it_did(self):
        """Chris: "we will not change the way the footprint reads on
        this strategy"."""
        st = sw.state({}, D, {}, 300.0, fp=U)
        assert st["context"]["footprint"] == "up"


class TestOnlyTheTickBarFlippingEndsIt:
    def test_it_holds_while_the_tick_bar_agrees(self):
        assert sw.holds(D, D) == (True, "")

    def test_the_opposite_direction_ends_it(self):
        ok, why = sw.holds(D, U)
        assert ok is False
        assert "the tick bar flipped up against your down trade" in why

    def test_FLAT_IS_NOT_A_FLIP(self):
        """It matters more here than anywhere else it has come up.
        The tick bar is cut at every true 5-minute mark, so it has
        no direction for the first prints of every bar. An exit on
        flat would fire twelve times an hour on the clock."""
        assert sw.holds(D, "") == (True, "")
        assert sw.holds(U, "") == (True, "")

    def test_it_is_symmetric(self):
        assert sw.holds(U, U)[0] is True
        assert sw.holds(U, D)[0] is False

    def test_no_open_side_is_never_an_exit(self):
        assert sw.holds("", U)[0] is True

    def test_the_stack_is_not_an_argument_any_more(self):
        """Pinned by signature so it cannot quietly come back."""
        import inspect
        assert list(inspect.signature(sw.holds).parameters) == \
            ["side", "tick"]


class TestNoTradeOutlivesItsFiveMinuteBar:
    """Chris: "no trade will ever last past a 5 minute timeframe"."""

    @staticmethod
    def at(h, m, s=0):
        import datetime as dt
        tz = dt.timezone(dt.timedelta(hours=-5))
        return dt.datetime(2026, 11, 17, h, m, s, tzinfo=tz).timestamp()

    def test_it_survives_inside_its_own_bar(self):
        assert sw.expired(self.at(7, 1), self.at(7, 4, 59))[0] is False

    def test_the_boundary_ends_it(self):
        gone, why = sw.expired(self.at(7, 1), self.at(7, 5))
        assert gone is True and "bar it was opened in has ended" in why

    def test_a_trade_opened_a_second_before_the_boundary_still_ends(self):
        assert sw.expired(self.at(7, 4, 59), self.at(7, 5))[0] is True

    def test_the_buckets_are_true_wall_clock_marks(self):
        import datetime as dt
        tz = dt.timezone(dt.timedelta(hours=-5))
        for h, m in ((7, 0), (7, 5), (7, 10), (13, 35)):
            b = sw.bar_of(self.at(h, m, 30))
            opened = dt.datetime.fromtimestamp(b, tz)
            assert (opened.minute % 5, opened.second) == (0, 0)

    def test_a_trade_with_no_recorded_bar_is_not_expired(self):
        """Zero is "we do not know". Closing every such position at
        the first boundary would be acting on missing data."""
        assert sw.expired(0.0, self.at(7, 5))[0] is False
        assert sw.expired(self.at(7, 1), 0.0)[0] is False

    def test_fifteen_minutes_works_too(self):
        assert sw.expired(self.at(7, 1), self.at(7, 5), 900.0)[0] is False
        assert sw.expired(self.at(7, 1), self.at(7, 15), 900.0)[0] is True

    def test_the_clock_is_checked_before_the_direction(self):
        """So the log says the bar ended rather than blaming a tick
        bar that is merely between bars at that same instant. Held
        by behaviour now rather than by the order of two lines in a
        route -- see TestTheExitPolicyIsOneFunction."""
        import datetime as dt
        tz = dt.timezone(dt.timedelta(hours=-5))
        at = lambda h, m: dt.datetime(2026, 11, 17, h, m,
                                      tzinfo=tz).timestamp()
        why = sw.exit_reason(D, U, at(7, 1), at(7, 5))
        assert "bar it was opened in has ended" in why
        assert "flipped" not in why

    def test_the_route_stamps_the_bar_a_trade_opened_in(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("def _journal_step_un(")
        blk = src[i:i + 30000]
        # TRADE_BAR_S, not FUSION_RESET_S. The two were one
        # constant doing two jobs -- the tick series' reset and the
        # trade time stop -- and making the series continuous would
        # have silently switched the time stop off with it.
        assert "_swb.bar_of(time.time(), TRADE_BAR_S)" in blk
        assert "bar_ts=bar_ts" in blk


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
        assert sw.entry_side(bad, {}, 300.0)[0] == ""

    def test_no_rungs_at_all_is_a_reason_not_a_crash(self):
        side, why = sw.stack_side({}, 300.0)
        assert side == "" and why

    def test_state_survives_empty_everything(self):
        st = sw.state(None, "", None, 300.0)
        assert st["side"] == ""
        assert st["tick"]["true"] is False
        # The gate is always consulted now, so `asked` is always
        # true. What an empty stack must still do is BLOCK.
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
        assert "_sw.exit_reason(" in blk, "the exit is not wired"
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
        # The row that gates is OUT of the shown-only block.
        tail = blk[blk.index("none of this decides"):]
        assert "row(st.trading)" not in tail, \
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
        that found it has to stay pointed at it.

        While the tick bar was the whole rule an empty stack cost
        nothing. The candle is a gate again, so {} means no trade --
        exactly the state the server-side stepper was in when it
        could not open a trade at all with the browser closed.

        What stops it this time is the NEXT test: the step builds
        its own rows from candles instead of waiting for the route
        to hand them over. These two belong together and neither is
        sufficient alone.
        """
        assert sw.stack_side({}, 300.0)[0] == ""
        side, why = sw.entry_side(D, {}, 300.0)
        assert side == "", "an empty stack opened a trade"
        assert "has not been read" in why

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


class TestTheExitPolicyIsOneFunction:
    """A mutation deleting the time stop's effect in the route broke
    no test, because the checks lived in a closure reachable only
    through an HTTP request -- the same shape that let the auto gate
    and the one-per-candle cap each be removed unnoticed."""

    @staticmethod
    def at(h, m, s=0):
        import datetime as dt
        tz = dt.timezone(dt.timedelta(hours=-5))
        return dt.datetime(2026, 11, 17, h, m, s, tzinfo=tz).timestamp()

    def test_a_healthy_trade_holds(self):
        assert sw.exit_reason(D, D, self.at(7, 1), self.at(7, 3)) == ""

    def test_a_flipped_tick_bar_ends_it(self):
        why = sw.exit_reason(D, U, self.at(7, 1), self.at(7, 3))
        assert "flipped up" in why

    def test_the_bar_ending_ends_it(self):
        why = sw.exit_reason(D, D, self.at(7, 1), self.at(7, 5))
        assert "bar it was opened in has ended" in why

    def test_the_clock_wins_when_both_are_true(self):
        """At the boundary the tick series is cut too, so it has no
        direction and could also look like a reason. The log must
        say the bar ended."""
        why = sw.exit_reason(D, U, self.at(7, 1), self.at(7, 5))
        assert "bar it was opened in has ended" in why
        assert "flipped" not in why

    def test_a_flat_tick_bar_inside_the_bar_holds(self):
        assert sw.exit_reason(D, "", self.at(7, 1), self.at(7, 3)) == ""

    def test_no_open_side_never_exits(self):
        assert sw.exit_reason("", U, self.at(7, 1), self.at(7, 9)) == ""

    def test_the_route_delegates_to_it(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("def _journal_step_un(")
        blk = src[i:i + 30000]
        assert "_sw.exit_reason(" in blk, "the policy is inline again"


class TestAFlipHasToHoldBeforeItTrades:
    """Chris: "a flip has to hold for 10 seconds before a trade is
    taken that keeps us from taking wrong direction trades and flips
    that lead to nothing"."""

    def test_the_default_is_ten_seconds(self):
        assert sw.DWELL_S == 10.0

    @pytest.mark.parametrize("held,opens", [
        (0.0, False), (4.0, False), (9.9, False),
        (10.0, True), (37.0, True),
    ])
    def test_it_waits_for_the_full_dwell(self, held, opens):
        side, why = sw.entry_side(D, rungs(**{"5m": D}), 300.0,
                                  held_s=held)
        assert bool(side) is opens
        if not opens:
            assert "has to hold 10s" in why

    def test_the_reason_says_how_long_it_has_held(self):
        assert "turned down 4s ago" in sw.entry_side(
            D, rungs(**{"5m": D}), 300.0, held_s=4.0)[1]

    def test_no_tracking_means_no_check(self):
        """A caller that cannot measure the dwell must not be
        silently blocked from ever trading."""
        assert sw.entry_side(D, rungs(**{"5m": D}), 300.0,
                             held_s=None) == ("down", "")

    def test_a_zero_minimum_disables_it(self):
        assert sw.entry_side(D, rungs(**{"5m": D}), 300.0,
                             held_s=0.0, min_hold_s=0.0) == ("down", "")

    def test_the_exit_is_NOT_delayed(self):
        """The asymmetry is the point: a flip gets you OUT at once
        and only lets you back IN once it has proved it meant
        something. Making the exit wait would hold a losing position
        for ten seconds to protect against a flip that was right."""
        import inspect
        assert "held" not in str(inspect.signature(sw.holds))
        assert sw.holds(D, U)[0] is False


class TestTheDwellTracker:
    def test_it_times_from_the_turn(self):
        d = sw.Dwell()
        d.on_tick("down", 1000.0)
        assert d.direction == "down"
        assert d.held_s(1004.0) == 4.0

    def test_the_opposite_direction_restarts_it(self):
        d = sw.Dwell()
        d.on_tick("down", 1000.0)
        d.on_tick("up", 1020.0)
        assert d.direction == "up"
        assert d.held_s(1021.0) == 1.0
        assert d.flips == 1

    def test_FLAT_DOES_NOT_RESET_IT(self):
        """The tick series is cut at every true 5-minute mark, so it
        has no direction for the first prints of every bar. Counting
        that as a turn would restart the clock twelve times an hour
        and cost ten seconds of every bar for nothing. A gap is not
        a turn."""
        d = sw.Dwell()
        d.on_tick("down", 1000.0)
        d.on_tick("", 1005.0)
        d.on_tick("down", 1006.0)
        assert d.held_s(1010.0) == 10.0, "the gap restarted the clock"
        assert d.flips == 0

    def test_the_same_direction_repeated_does_not_restart_it(self):
        d = sw.Dwell()
        for t in range(1000, 1010):
            d.on_tick("up", float(t))
        assert d.held_s(1010.0) == 10.0

    def test_nothing_held_reads_zero(self):
        d = sw.Dwell()
        assert d.direction == "" and d.held_s(1000.0) == 0.0
        d.on_tick("", 1000.0)
        assert d.held_s(1001.0) == 0.0

    def test_a_clock_going_backwards_is_zero_not_negative(self):
        d = sw.Dwell()
        d.on_tick("up", 1000.0)
        assert d.held_s(990.0) == 0.0

    @pytest.mark.parametrize("junk", [None, "flat", "UP", 7, ""])
    def test_junk_is_not_a_direction(self, junk):
        d = sw.Dwell()
        d.on_tick(junk, 1000.0)
        assert d.direction == ""

    def test_a_junk_clock_does_not_corrupt_it(self):
        d = sw.Dwell()
        d.on_tick("up", 1000.0)
        d.on_tick("up", None)
        d.on_tick("up", "x")
        assert d.held_s(1005.0) == 5.0


class TestTheDwellReachesTheRuleAndThePanel:
    def test_the_state_reports_it(self):
        st = sw.state({}, D, {}, 300.0, held_s=4.0, flips=3)
        assert st["side"] == "", "it traded before the dwell was met"
        assert st["tick"]["held_s"] == 4.0
        assert st["tick"]["ready"] is False
        assert st["tick"]["min_hold_s"] == 10.0
        assert st["tick"]["flips"] == 3

    def test_a_held_direction_reads_ready(self):
        st = sw.state({}, D, rungs(**{"5m": D}), 300.0, held_s=30.0)
        assert st["side"] == "down"
        assert st["tick"]["ready"] is True

    def test_an_untracked_dwell_is_ready(self):
        st = sw.state({}, D, rungs(**{"5m": D}), 300.0)
        assert st["tick"]["ready"] is True
        assert st["side"] == "down"

    def test_the_route_keeps_ONE_tracker_on_the_runtime(self):
        """A second instance would let the screen show a dwell the
        rule never saw -- the fault the 5-minute cell and the OFI
        bar both had."""
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("def _journal_step_un(")
        blk = src[i:i + 30000]
        assert 'getattr(rt_, "_tick_dwell", None)' in blk
        assert "_dw.on_tick(" in blk
        assert "held_s=_dw.held_s(" in blk
        assert blk.count("_sw.Dwell()") == 1, "more than one tracker"

    def test_the_panel_shows_waiting_rather_than_nothing(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("function paintStackWin(")
        blk = src[i:i + 6000]
        assert "HOLDING " in blk, "a waiting entry looks like no signal"
        assert "f.ready === false" in blk


class TestTheTwoFiveMinutesAreSeparate:
    """`FUSION_RESET_S` cut the tick series AND set the period of
    the trade time stop. Making the series continuous would have
    silently switched off "no trade will ever last past a 5 minute
    timeframe" -- a safety rule disappearing as a side effect of a
    display change."""

    def test_the_tick_series_is_continuous(self):
        import liqmap.web as web
        assert web.FUSION_RESET_S == 0.0

    def test_the_time_stop_is_still_five_minutes(self):
        import liqmap.web as web
        assert web.TRADE_BAR_S == 300.0

    def test_they_are_different_constants(self):
        import liqmap.web as web
        assert web.FUSION_RESET_S != web.TRADE_BAR_S

    def test_the_time_stop_still_fires(self):
        """The behaviour, not the constant. A trade opened in one
        5-minute bar must not survive into the next."""
        import datetime as dt
        tz = dt.timezone(dt.timedelta(hours=-5))
        at = lambda h, m: dt.datetime(2026, 11, 17, h, m,
                                      tzinfo=tz).timestamp()
        import liqmap.web as web
        assert sw.expired(at(7, 1), at(7, 5), web.TRADE_BAR_S)[0] is True
        assert sw.expired(at(7, 1), at(7, 4), web.TRADE_BAR_S)[0] is False

    def test_gate_one_reads_the_continuous_bar(self):
        """The same series the 5000t chart draws, so what he watches
        and what trades are one bar."""
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("def _journal_step_un(")
        blk = src[i:i + 30000]
        assert '_tick_now(rt_, 0, "raw", 0.0, 0, FUSION_RESET_S)' in blk
        assert "FUSION_RESET_S = 0.0" in src
