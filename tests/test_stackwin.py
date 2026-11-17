"""The window gates, then the stack confirms. In that order.

Chris:

    "a buy or a sell is based on what the stack says and what the
    window says but first the window needs to read true before
    anything else can be done"

    "on a 5 minute the 5 min and 1 minute must agree as well as the
    window must agree candle, OFI, CVD, Footprint. and on the 15 min
    the 15 and 5 must agree with the same 4 from the window"

He watched the window line show all four DOWN and no trade follow.
Nothing was broken: NO STRATEGY ASKED FOR ALL FOUR. `ofi` wants
three, `footprint` wants two, and the two intrabar rules return
before the window is read. These tests pin the new rule and, as
much, pin that the ORDER is load-bearing -- a rule that checks both
gates and reports whichever failed is a different rule that fires
on the same trades and explains them wrongly.
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
        tf = tf.replace("m_", "m")
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


class TestTheWindowNeedsAllFour:
    def test_all_four_down_is_down(self):
        assert sw.window_side(D, D, D, D) == ("down", "")

    def test_all_four_up_is_up(self):
        assert sw.window_side(U, U, U, U) == ("up", "")

    @pytest.mark.parametrize("legs", [
        # one up against three down ...
        (U, D, D, D), (D, U, D, D), (D, D, U, D), (D, D, D, U),
        # ... AND THE MIRROR. Without these a mutation loosening the
        # test to `len(up) >= 3` survived the whole file: every case
        # here was one-up-three-down, so nothing ever asked what
        # three up and one down does.
        (D, U, U, U), (U, D, U, U), (U, U, D, U), (U, U, U, D),
    ])
    def test_one_against_is_no_side(self, legs):
        side, why = sw.window_side(*legs)
        assert side == ""
        assert "disagrees" in why

    def test_the_reason_names_the_minority_not_the_first_found(self):
        """"the candle is down but the CVD is up" sends you to the
        CVD when the footprint was against it too."""
        _s, why = sw.window_side(D, D, U, U)
        assert "CVD" in why and "footprint" in why

    def test_two_against_two_has_no_minority(self):
        _s, why = sw.window_side(D, U, D, U)
        assert "two against two" in why

    @pytest.mark.parametrize("legs", [
        ("", D, D, D), (D, "", D, D), (D, D, "flat", D), (D, D, D, ""),
    ])
    def test_a_flat_leg_blocks_it(self, legs):
        """Flat is not a direction. Four zeros are not four signals
        pointing together."""
        side, why = sw.window_side(*legs)
        assert side == ""
        assert "no direction" in why

    def test_the_reason_names_every_flat_leg(self):
        _s, why = sw.window_side("", D, "", D)
        assert "candle" in why and "CVD" in why and "have" in why


class TestTheStackIsTheTradedRowAndTheOneBelow:
    def test_five_minutes_takes_the_5m_and_the_1m(self):
        r = rungs(**{"15m": U, "5m": D, "1m": D})
        assert sw.stack_side(r, 300.0) == ("down", "")

    def test_fifteen_minutes_takes_the_15m_and_the_5m(self):
        r = rungs(**{"15m": D, "5m": D, "1m": U})
        assert sw.stack_side(r, 900.0) == ("down", "")

    def test_the_row_above_is_not_consulted(self):
        """On the 5-minute he is trading the 5-minute. A 15-minute
        that disagrees is the pullback he is trading into, not a
        veto."""
        r = rungs(**{"15m": U, "5m": D, "1m": D})
        assert sw.stack_side(r, 300.0)[0] == "down"

    def test_the_row_below_is_not_consulted_on_fifteen(self):
        r = rungs(**{"15m": D, "5m": D, "1m": U})
        assert sw.stack_side(r, 900.0)[0] == "down"

    def test_the_two_rows_disagreeing_is_no_side(self):
        r = rungs(**{"5m": D, "1m": U})
        side, why = sw.stack_side(r, 300.0)
        assert side == ""
        assert "5m is down but the 1m is up" in why

    def test_an_unread_row_is_named(self):
        r = rungs(**{"5m": D, "1m": None})
        side, why = sw.stack_side(r, 300.0)
        assert side == "" and "1m has not been read" in why

    def test_a_row_contradicting_itself_is_thrown_out_not_used(self):
        """The cascade sets `qualified` to None when a row's colour
        and its own number disagree. That is not a direction."""
        r = rungs(**{"5m": D, "1m": "conflict"})
        side, why = sw.stack_side(r, 300.0)
        assert side == "" and "contradicts itself" in why


class TestTheOrderIsLoadBearing:
    """Not presentation. When the window has no side the stack is
    never consulted and no reason mentions it."""

    def test_a_false_window_never_reaches_the_stack(self):
        r = rungs(**{"5m": D, "1m": D})
        side, why = sw.entry_side(D, D, U, U, r, 300.0)
        assert side == ""
        assert "window has no side" in why
        assert "stack" not in why.lower()

    def test_the_stack_is_marked_not_asked(self):
        r = rungs(**{"5m": D, "1m": D})
        st = sw.state({"mid_dir": D, "ofi_dir": D, "cvd_dir": U}, U,
                      r, 300.0)
        assert st["stack"]["asked"] is False
        assert st["window"]["true"] is False

    def test_a_true_window_with_a_silent_stack_says_so(self):
        r = rungs(**{"5m": D, "1m": U})
        side, why = sw.entry_side(D, D, D, D, r, 300.0)
        assert side == ""
        assert "the window is down but the stack does not back it" in why

    def test_both_true_and_pointing_apart(self):
        r = rungs(**{"5m": U, "1m": U})
        side, why = sw.entry_side(D, D, D, D, r, 300.0)
        assert side == "" and "point apart" in why

    def test_both_agreeing_is_the_trade(self):
        r = rungs(**{"5m": D, "1m": D})
        assert sw.entry_side(D, D, D, D, r, 300.0) == ("down", "")


class TestTheExitIsAFullReversal:
    """Chris chose this over the tighter exit and has paid for the
    difference twice: exiting the moment agreement breaks took
    twelve trades in an afternoon at two to five ticks each."""

    def test_all_four_the_other_way_closes_it(self):
        out, why = sw.reversed_window(U, D, D, D, D)
        assert out is True and "turned fully down" in why

    def test_losing_agreement_does_not(self):
        assert sw.reversed_window(U, D, D, D, "")[0] is False
        assert sw.reversed_window(U, D, D, U, D)[0] is False

    def test_the_window_still_backing_you_does_not(self):
        assert sw.reversed_window(U, U, U, U, U)[0] is False

    def test_it_is_symmetric(self):
        assert sw.reversed_window(D, U, U, U, U)[0] is True

    def test_no_side_never_exits(self):
        assert sw.reversed_window("", D, D, D, D)[0] is False


class TestTheScreenshots:
    """The two readings he sent, run through the rule."""

    def test_all_four_down_on_the_5m_is_a_trade(self):
        r = rungs(**{"5m": D, "1m": D, "15m": D})
        st = sw.state({"mid_dir": D, "ofi_dir": D, "cvd_dir": D}, D,
                      r, 300.0)
        assert st["side"] == "down"
        assert "both down" in st["say"]

    def test_the_mixed_one_is_not(self):
        r = rungs(**{"5m": D, "1m": D, "15m": D})
        st = sw.state({"mid_dir": D, "ofi_dir": D, "cvd_dir": U}, U,
                      r, 300.0)
        assert st["side"] == ""
        assert st["stack"]["asked"] is False


class TestTheEdges:
    def test_an_unknown_window_length_still_picks_a_row(self):
        assert sw.traded_tf(300.0) == "5m"
        assert sw.traded_tf(900.0) == "15m"
        assert sw.traded_tf(600.0) == "15m"
        assert sw.traded_tf(120.0) == "5m"

    @pytest.mark.parametrize("bad", [None, 0, "", "x"])
    def test_junk_never_produces_a_side(self, bad):
        assert sw.traded_tf(bad) in ("", "5m", "15m")
        assert sw.entry_side(bad, bad, bad, bad, {}, 300.0)[0] == ""

    def test_no_rungs_at_all_is_a_reason_not_a_crash(self):
        side, why = sw.stack_side({}, 300.0)
        assert side == "" and why

    def test_state_survives_empty_everything(self):
        st = sw.state(None, "", None, 300.0)
        assert st["side"] == ""
        assert st["window"]["true"] is False
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
        side, why = jn.window_signal({}, "stack", "",
                                     {"side": "down", "why": ""})
        assert side == "down"
        side, why = jn.window_signal({}, "stack", "",
                                     {"side": "", "why": "no window"})
        assert side == "" and why == "no window"

    def test_it_is_not_refused_for_no_window_yet(self):
        """The guards below the dispatch are written for the
        three-leg rule. This one owns its own flat handling."""
        from liqmap import journal as jn
        side, _why = jn.window_signal(None, "stack", "",
                                      {"side": "up", "why": ""})
        assert side == "up"

    def test_the_route_computes_it_and_the_panel_gets_it(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("def _journal_step_un(")
        blk = src[i:i + 30000]
        assert 'strat == "stack"' in blk, "the route never builds it"
        assert "_sw.state(" in blk
        assert '"stackwin"' in blk, "it never reaches the payload"
        assert "reversed_window(" in blk, "the exit is not wired"

    def test_the_panel_hides_itself_for_other_strategies(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("function paintStackWin(")
        blk = src[i:i + 4000]
        assert "style.display = 'none'" in blk
        assert "not asked" in blk, "gate 2 does not say it was skipped"

    def test_the_dropdown_offers_it(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index('id="jnlStrat"')
        box = src[i:src.index("</select>", i)]
        assert 'value="stack"' in box
