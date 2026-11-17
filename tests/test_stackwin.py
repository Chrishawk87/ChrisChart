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


class TestTheTickBarGatesFirstAndAlone:
    def test_no_tick_direction_never_reaches_the_stack(self):
        r = rungs(**{"5m": D, "1m": D})
        side, why = sw.entry_side("", r, 300.0)
        assert side == ""
        assert "tick bar has no direction" in why
        assert "stack" not in why.lower()

    def test_the_stack_is_marked_not_asked(self):
        r = rungs(**{"5m": D, "1m": D})
        st = sw.state({"mid_dir": D, "ofi_dir": D, "cvd_dir": D}, "",
                      r, 300.0)
        assert st["stack"]["asked"] is False
        assert st["tick"]["true"] is False

    def test_a_tick_direction_with_a_silent_stack_says_so(self):
        # The TRADED row has to be the silent one now. With `5m: D`
        # this entry succeeds -- the 1m is no longer consulted --
        # which is the whole change.
        r = rungs(**{"5m": "flat", "1m": U})
        side, why = sw.entry_side(D, r, 300.0)
        assert side == ""
        assert "the tick bar is down but the stack does not back it" in why
        assert "the 5m is flat" in why

    def test_both_true_and_pointing_apart(self):
        r = rungs(**{"5m": U, "1m": D})
        side, why = sw.entry_side(D, r, 300.0)
        assert side == "" and "point apart" in why

    def test_both_agreeing_is_the_trade(self):
        r = rungs(**{"5m": D, "1m": U})     # the 1m is ignored
        assert sw.entry_side(D, r, 300.0) == ("down", "")

    def test_it_is_symmetric(self):
        r = rungs(**{"5m": U, "1m": D})
        assert sw.entry_side(U, r, 300.0) == ("up", "")


class TestTheOtherLegsDecideNothing:
    def test_all_four_against_it_and_it_still_fires(self):
        """Candle, OFI, CVD and the FOOTPRINT all the other way.
        The footprint used to be gate 1; it decides nothing now."""
        r = rungs(**{"5m": D, "1m": D})
        st = sw.state({"mid_dir": U, "ofi_dir": U, "cvd_dir": U}, D,
                      r, 300.0, fp=U)
        assert st["side"] == "down"
        assert st["context"]["decides"] is False
        assert st["context"]["footprint"] == "up"

    def test_they_are_still_reported(self):
        """Shown, not used. A leg that silently stops counting is
        what made the old window look broken."""
        r = rungs(**{"5m": D, "1m": D})
        cx = sw.state({"mid_dir": U, "ofi_dir": D, "cvd_dir": U}, D,
                      r, 300.0, fp=D)["context"]
        assert cx["candle"] == "up" and cx["ofi"] == "down"
        assert cx["footprint"] == "down"

    def test_the_four_leg_reading_is_kept_for_the_panel(self):
        assert sw.window_side(D, D, D, D) == ("down", "")
        side, why = sw.window_side(D, D, U, U)
        assert side == "" and "two against two" in why


class TestOnlyAFlipOfTheTradedTimeframeEndsIt:
    """Chris: "a disagreement is only when the primary timeframe
    that im trading on flips".

    THE TICK BAR IS NOT CONSULTED AT ALL HERE. It gets you in; the
    traded timeframe keeps you in. It closes on a count of prints,
    so between bars it has no direction -- and with the 5-minute
    reset forced onto that series, an exit that read it would fire
    on the clock at every boundary rather than on the market. That
    was the live risk in the previous version.
    """

    def test_it_holds_while_the_traded_row_still_backs_it(self):
        r = rungs(**{"5m": D, "1m": U})
        assert sw.holds(D, r, 300.0) == (True, "")

    def test_only_the_opposite_side_ends_it(self):
        r = rungs(**{"5m": U, "1m": U})
        ok, why = sw.holds(D, r, 300.0)
        assert ok is False
        assert "the 5m flipped up against your down trade" in why

    def test_a_FLAT_traded_row_is_not_a_flip(self):
        """A candle sitting on its open has gone nowhere; it has
        not turned. Closing on that ends trades for an absence,
        which is the churn this rule has twice been rewritten to
        remove."""
        r = rungs(**{"5m": "flat", "1m": U})
        assert sw.holds(D, r, 300.0) == (True, "")

    def test_an_unread_traded_row_is_not_a_flip_either(self):
        r = rungs(**{"5m": None, "1m": U})
        assert sw.holds(D, r, 300.0)[0] is True

    def test_the_one_minute_cannot_end_it(self):
        for below in (U, D, None, "conflict"):
            r = rungs(**{"5m": D, "1m": below})
            assert sw.holds(D, r, 300.0)[0] is True, below

    def test_the_tick_bar_cannot_end_it(self):
        """It is not even an argument to `holds` any more. Pinned
        by signature so it cannot quietly come back."""
        import inspect
        params = list(inspect.signature(sw.holds).parameters)
        assert params == ["side", "rungs", "period_s"], params

    def test_on_a_fifteen_minute_window_the_15m_decides(self):
        r = rungs(**{"15m": U, "5m": D, "1m": D})
        ok, why = sw.holds(D, r, 900.0)
        assert ok is False and "the 15m flipped up" in why
        assert sw.holds(U, r, 900.0)[0] is True

    def test_it_is_symmetric(self):
        r = rungs(**{"5m": D, "1m": U})
        assert sw.holds(D, r, 300.0)[0] is True
        assert sw.holds(U, r, 300.0)[0] is False

    def test_no_open_side_is_never_an_exit(self):
        r = rungs(**{"5m": U, "1m": U})
        assert sw.holds("", r, 300.0)[0] is True

    def test_a_trade_survives_what_would_block_a_new_one(self):
        """ASYMMETRIC ON PURPOSE, and worth stating. A flat 5m
        blocks an entry -- the tick bar has nothing to agree with --
        and holds an open position, because flat is not a flip."""
        r = rungs(**{"5m": "flat", "1m": U})
        assert sw.entry_side(D, r, 300.0)[0] == ""
        assert sw.holds(D, r, 300.0)[0] is True


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
        assert st["stack"]["asked"] is False


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
        assert "_sw.holds(" in blk, "the tight exit is not wired"
        assert "_tick_now(rt_, 0, \"raw\", 0.0, 0, FUSION_RESET_S)" in blk, \
            "gate 1 is not reading the Fusion panel's tick series"

    def test_the_panel_labels_what_does_not_decide(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("function paintStackWin(")
        blk = src[i:i + 6000]
        assert "style.display = 'none'" in blk
        assert "not asked" in blk, "gate 2 does not say it was skipped"
        assert "do not decide" in blk, "the shown-only legs are unlabelled"

    def test_the_dropdown_offers_it(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index('id="jnlStrat"')
        box = src[i:src.index("</select>", i)]
        assert 'value="stack"' in box


class TestItCanTakeMoreThanOneTrade:
    """Chris: "it takes a trade hits take profit but signal is still
    holding strong then it takes another trade".

    IT ALREADY COULD -- the one-per-candle cap is set only for the
    fused and timed rules, and this one was never in that list. The
    test exists so that adding a third strategy to the cap does not
    quietly take it away."""

    def test_the_cap_does_not_apply_to_this_strategy(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("cap = 0")
        blk = src[i:i + 400]
        assert 'strat in ("fused", "timed")' in blk
        assert '"stack"' not in blk.split("cap = 1")[0], \
            "the stack rule was added to the one-per-candle cap"


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

    def test_an_empty_stack_blocks_gate_two_entirely(self):
        """The bug, pinned. This is what the stepper saw."""
        assert sw.stack_side({}, 300.0)[0] == ""
        assert sw.entry_side(D, {}, 300.0)[0] == ""

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
