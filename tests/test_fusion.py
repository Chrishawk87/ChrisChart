"""Two series read separately, compared on direction only.

Chris's rule, in his words:

    "the 5000 tick bar to read independent of the OFI the agreement
    comes from both of them agreeing at direction and exiting when
    they disagree reenter when they agree again exit when they
    disagree ... they should read independent but have the same
    directional interest for a trade to happen"

So: independence of the two readings, agreement to enter, loss of
agreement to exit, and re-entry allowed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from liqmap import fusion as fu
from liqmap.journal import Settings, Trade, decide, should_close, \
    window_signal

ROOT = Path(__file__).resolve().parents[1]


def py_func(name: str, path: str = "liqmap/web.py") -> str:
    src = (ROOT / path).read_text()
    i = src.index(f"def {name}(")
    start = src.rfind("\n", 0, i) + 1
    indent = i - start
    lines = src[start:].split("\n")
    out = [lines[0]]
    for line in lines[1:]:
        if line.strip() and (len(line) - len(line.lstrip())) <= indent:
            break
        out.append(line)
    return "\n".join(out)


def js_func(name: str) -> str:
    src = (ROOT / "liqmap" / "web.py").read_text()
    i = src.index(f"function {name}(")
    depth, j = 0, src.index("{", i)
    for k in range(j, len(src)):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return src[i:k + 1]
    raise AssertionError(f"{name} has no closing brace")


class TestItDoesNotTrade:
    def test_no_order_entry_points(self):
        tree = ast.parse((ROOT / "liqmap" / "fusion.py").read_text())
        names = {n.name for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for banned in ("submit", "execute", "place", "order", "buy",
                       "sell", "send_order"):
            assert banned not in names, banned

    def test_it_imports_nothing(self):
        """The comparison is arithmetic on two strings.

        A module that reaches for a builder, a feed or a client to
        answer "do these two point the same way" has grown a second
        source of truth.
        """
        tree = ast.parse((ROOT / "liqmap" / "fusion.py").read_text())
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module:
                mods.add(n.module.split(".")[0])
        assert mods <= {"__future__"}, mods


class TestAgreement:
    def test_both_up_is_up(self):
        assert fu.agreement("up", "up") == ("up", "")

    def test_both_down_is_down(self):
        assert fu.agreement("down", "down") == ("down", "")

    def test_opposite_is_no_side(self):
        side, why = fu.agreement("up", "down")
        assert side == ""
        assert why == "the tick bar is up and the OFI is down"

    @pytest.mark.parametrize("t,o", [("flat", "up"), ("", "up"),
                                     ("up", "flat"), ("up", "")])
    def test_flat_on_either_side_is_no_side(self, t, o):
        """Flat is not a direction.

        Without this a dead market -- two neutrals -- reads as the
        strongest agreement there is.
        """
        side, why = fu.agreement(t, o)
        assert side == ""
        assert "flat" in why

    def test_both_flat_says_so(self):
        side, why = fu.agreement("", "")
        assert side == ""
        assert why == "neither the tick bar nor the OFI has a direction"

    def test_the_reason_is_never_blank_when_the_side_is(self):
        for t in ("up", "down", "flat", "", "garbage"):
            for o in ("up", "down", "flat", "", "garbage"):
                side, why = fu.agreement(t, o)
                if not side:
                    assert why.strip(), (t, o)

    def test_it_is_case_and_space_tolerant(self):
        assert fu.agreement(" UP ", "Up") == ("up", "")

    def test_an_unknown_word_is_not_a_direction(self):
        assert fu.agreement("sideways", "up")[0] == ""


class TestHolds:
    """The exit test, stated apart from the entry test."""

    def test_it_holds_while_both_still_point_that_way(self):
        assert fu.holds("up", "up", "up") is True

    def test_it_breaks_when_one_turns(self):
        assert fu.holds("up", "up", "down") is False
        assert fu.holds("up", "down", "up") is False

    def test_it_breaks_when_one_goes_flat(self):
        """Neutral is a disagreement here, by Chris's rule.

        The reason for the trade was that BOTH pointed the same way.
        One of them no longer does.
        """
        assert fu.holds("up", "up", "flat") is False
        assert fu.holds("up", "flat", "up") is False

    def test_it_does_not_hold_the_wrong_side(self):
        assert fu.holds("down", "up", "up") is False

    def test_entry_and_exit_cannot_drift_apart(self):
        """`holds` is defined in terms of `agreement`.

        Two separate implementations of the same comparison is how a
        rule ends up entering on one definition and exiting on
        another.
        """
        for t in ("up", "down", "flat"):
            for o in ("up", "down", "flat"):
                side = fu.agreement(t, o)[0]
                for guess in ("up", "down"):
                    assert fu.holds(guess, t, o) == (side == guess), (t, o)


class TestState:
    def test_it_reads_each_series_own_field(self):
        """The tick bar reports `dir`; the OFI bar reports `read`.

        They are different builders with different vocabularies and
        this is where that is reconciled -- once.
        """
        st = fu.state({"dir": "up", "per_bar": 5000, "prints": 1200},
                      {"read": "up", "imbalance": 900, "level": 250},
                      {"dir": "up", "seconds": 200})
        assert st["side"] == "up" and st["agree"] is True
        assert st["tick"]["dir"] == "up"
        assert st["ofi"]["dir"] == "up"

    def test_it_carries_the_evidence_for_the_verdict(self):
        st = fu.state({"dir": "down", "per_bar": 5000, "prints": 40,
                       "ticks": -3.0, "seconds": 12.0},
                      {"read": "down", "imbalance": -800, "level": 250,
                       "delta": -120})
        assert st["tick"]["prints"] == 40 and st["tick"]["ticks"] == -3.0
        assert st["ofi"]["imbalance"] == -800
        assert st["ofi"]["level"] == 250

    def test_missing_inputs_are_a_reading_not_a_crash(self):
        st = fu.state(None, None)
        assert st["agree"] is False
        assert st["say"].strip()

    def test_say_is_filled_either_way(self):
        assert "up" in fu.state({"dir": "up"}, {"read": "up"})["say"]
        assert fu.state({"dir": "up"}, {"read": "down"})["say"].strip()


class TestTheStrategyUsesIt:
    def test_agreement_gives_the_side(self):
        st = fu.state({"dir": "up"}, {"read": "up"},
                      {"dir": "up", "seconds": 200})
        assert window_signal({}, "fused", "", st) == ("up", "")

    def test_disagreement_passes_the_reason_through(self):
        st = fu.state({"dir": "up"}, {"read": "down"})
        side, why = window_signal({}, "fused", "", st)
        assert side == ""
        assert why == "the tick bar is up and the OFI is down"

    def test_it_never_reads_the_window(self):
        """Every other strategy refuses on an empty window."""
        st = fu.state({"dir": "down"}, {"read": "down"},
                      {"dir": "down", "seconds": 200})
        assert window_signal({}, "fused", "", st)[0] == "down"
        assert window_signal({}, "ofi")[1] == "no window yet"

    def test_no_reading_is_a_reason(self):
        assert window_signal({}, "fused", "", None) == (
            "", "no reading from the tick bar or the OFI yet")

    def test_the_other_strategies_are_untouched(self):
        win = {"ts_last": 1000.0, "start_ts": 0.0, "mid_dir": "up",
               "ofi_dir": "up", "cvd_dir": "up"}
        assert window_signal(win, "ofi")[0] == "up"
        assert window_signal(win, "candle")[0] == "up"
        assert window_signal(win, "footprint", "up")[0] == "up"


class TestTheRouteReadsBothSeries:
    def test_the_tick_reading_is_the_forming_bar(self):
        blk = py_func("_tick_now")
        assert "bld.current" in blk
        assert "last_closed" not in blk, "the rule is intrabar"

    def test_the_tick_reading_does_not_consult_the_book(self):
        """Independence, enforced where it could quietly be lost."""
        blk = py_func("_tick_now")
        for leak in ("ofibars", "bar_read", "net_ofi", "imbalance"):
            assert leak not in blk, leak

    def test_the_fusion_reading_combines_the_two_sources(self):
        blk = py_func("_fusion_now")
        assert "_tick_now(" in blk
        assert "_bar_read(" in blk

    def test_the_strategy_reads_the_same_function_the_panel_does(self):
        """One source, so the screen cannot show an agreement the
        rule did not see."""
        step = py_func("_journal_step")
        route = py_func("api_fusion")
        assert "_fusion_now(rt_)" in step
        assert "_fusion_now(rt_" in route

    def test_the_five_thousand_tick_series_exists(self):
        from liqmap.tickbar import PER_BAR_CHOICES, STRATEGY_PER_BAR
        assert STRATEGY_PER_BAR == 5000
        assert 5000 in PER_BAR_CHOICES

    def test_the_strategy_defaults_to_it(self):
        blk = py_func("_tick_now")
        assert "STRATEGY_PER_BAR" in blk

    def test_only_fused_trades_are_closed_by_the_5m(self):
        """The exit reading is taken for this strategy and no other."""
        blk = py_func("_journal_step")
        assert 'if strat == "fused" and t is not None and t.open:' in blk
        assert "m5_against = \"\"" in blk


class TestThePanel:
    def test_it_is_its_own_titled_panel(self):
        """The last reading that went inside another section
        rendered correctly and could not be found."""
        src = (ROOT / "liqmap" / "web.py").read_text()
        assert 'id="fusionPanel"' in src
        i = src.index('id="fusionPanel"')
        head = src[i:i + 300]
        assert 'class="panel full"' in src[i - 40:i + 40]
        assert "<h2>" in head

    def test_it_shows_both_series_separately(self):
        blk = js_func("paintFusion")
        assert "TICK BAR" in blk and "OFI BAR" in blk
        assert "prints and price only" in blk
        assert "the resting book only" in blk

    def test_the_verdict_is_the_loud_part(self):
        blk = js_func("paintFusion")
        assert "ALL THREE" in blk and "NO ENTRY" in blk

    def test_it_says_what_actually_ends_a_trade(self):
        """Reading "no entry" as "get out" is the mistake that
        produced fourteen trades in a few minutes."""
        blk = js_func("paintFusion")
        assert "held until the" in blk
        assert "5m" in blk

    def test_the_sizes_come_from_the_server(self):
        """Hard-coded sizes are how a panel offers a series that has
        no builder behind it."""
        blk = js_func("fusLoad")
        assert "d.sizes" in blk
        assert "/api/fusion?" in blk

    def test_it_rides_the_existing_poll(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        assert "try { fusLoad(); } catch (e) {}" in src

    def test_the_route_exists_and_is_behind_the_token(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index('@app.get("/api/fusion"')
        assert "Depends(require_token)" in src[i:i + 120]


class TestTheFiveMinuteHold:
    """Chris, after running it:

        "it literally took 14 trades in a matter of a few minutes
        that closed due to disagreement that cant happen in real
        trading we need a way to verify the hold so we will use the 5
        minute timeframe bar to keep this from bouncing all over the
        place"

    So the fast pair no longer closes anything. Entry needs all
    three; only the 5m turning against an open trade ends it.
    """

    @staticmethod
    def _open(side="up"):
        return Trade(id=1, coin="ES", side=side, mode="auto",
                     open_px=6800.0, strategy="fused")

    @staticmethod
    def _s():
        return Settings(coin="ES", auto=True, tp_on=False, sl_on=False,
                        strategy="fused")

    # -- the settle window ------------------------------------------------

    def test_an_unsettled_bar_has_no_direction(self):
        side, why = fu.m5_direction({"dir": "up", "seconds": 30})
        assert side == ""
        assert "30s old" in why and "90s" in why

    def test_a_settled_bar_does(self):
        assert fu.m5_direction({"dir": "up", "seconds": 120}) == ("up", "")

    def test_the_boundary(self):
        assert fu.m5_direction({"dir": "up",
                                "seconds": fu.M5_SETTLE_S})[0] == "up"
        assert fu.m5_direction({"dir": "up",
                                "seconds": fu.M5_SETTLE_S - 0.1})[0] == ""

    def test_settling_is_what_makes_it_a_filter(self):
        """Thirty seconds in, a 5m bar flips about as freely as the
        tick bar it is meant to steady."""
        assert fu.M5_SETTLE_S >= 60.0

    def test_a_flat_settled_bar_has_no_direction_either(self):
        assert fu.m5_direction({"dir": "flat", "seconds": 300})[0] == ""

    def test_no_bar_is_a_reason(self):
        side, why = fu.m5_direction(None)
        assert side == "" and why.strip()

    # -- entry ------------------------------------------------------------

    def test_entry_needs_all_three(self):
        assert fu.entry_side("up", "up", "up") == ("up", "")
        assert fu.entry_side("down", "down", "down") == ("down", "")

    def test_the_fast_pair_alone_is_not_an_entry(self):
        """The change. This used to open a trade."""
        side, why = fu.entry_side("up", "up", "")
        assert side == ""
        assert "the 5m is flat" in why

    def test_a_5m_against_blocks_entry(self):
        side, why = fu.entry_side("up", "up", "down")
        assert side == ""
        assert "the 5m is down" in why

    def test_the_pair_failing_is_reported_before_the_5m(self):
        """A trader wants to know which reading failed, and "the 5m
        is flat" is unhelpful when the pair never agreed."""
        side, why = fu.entry_side("up", "down", "up")
        assert side == ""
        assert why == "the tick bar is up and the OFI is down"

    def test_nothing_opens_in_the_unsettled_part_of_the_bar(self):
        st = fu.state({"dir": "up"}, {"read": "up"},
                      {"dir": "up", "seconds": 10})
        assert st["side"] == ""
        assert st["pair"] == "up", "the fast pair did agree"
        assert window_signal({}, "fused", "", st)[0] == ""

    # -- the hold ---------------------------------------------------------

    def test_a_flat_5m_holds(self):
        ok, why = fu.hold_against_m5("up", "")
        assert ok is True and why == ""

    def test_an_agreeing_5m_holds(self):
        assert fu.hold_against_m5("up", "up")[0] is True

    def test_only_a_5m_against_ends_it(self):
        ok, why = fu.hold_against_m5("up", "down")
        assert ok is False
        assert why == "the 5m turned down against a up"

    def test_the_fast_pair_no_longer_closes_anything(self):
        """The whole fix, in one assertion.

        Tick and OFI in full disagreement, trade still open.
        """
        assert should_close(self._open(), self._s(), 6800.25, "") == ""

    def test_even_a_full_fast_reversal_is_held(self):
        """Stated out loud because it is a real exposure: you can be
        long while both fast series say down, until the 5m turns.
        The stop is the only fast protection left.

        The signal reaching `should_close` is the ENTRY side, which
        requires all three -- so the pair reversing under a 5m that
        is still up arrives as no signal at all, not as "down".
        """
        pair_reversed = fu.entry_side("down", "down", "up")[0]
        assert pair_reversed == "", "the 5m is still up"
        assert should_close(self._open(), self._s(), 6800.25,
                            pair_reversed) == ""

    def test_a_three_way_reversal_does_close_it(self):
        """And then it is a flip, which is a real turn: the 5m is
        part of that agreement, so this is not the fast noise the
        hold was added to ignore."""
        both = fu.entry_side("down", "down", "down")[0]
        assert both == "down"
        assert should_close(self._open(), self._s(), 6800.25,
                            both) == "flip"

    def test_the_5m_turning_closes_it(self):
        why = should_close(self._open(), self._s(), 6800.25, "",
                           m5_against="the 5m turned down against a up")
        assert why == "m5"

    def test_the_stop_still_wins(self):
        s = Settings(coin="ES", auto=True, sl_on=True, sl_ticks=2)
        why = should_close(self._open(), s, 6799.0, "",
                           m5_against="the 5m turned")
        assert why == "sl"

    def test_the_target_still_wins(self):
        s = Settings(coin="ES", auto=True, tp_on=True, tp_ticks=2)
        why = should_close(self._open(), s, 6800.75, "",
                           m5_against="the 5m turned")
        assert why == "tp"

    def test_a_manual_trade_is_not_closed_by_the_5m(self):
        t = self._open()
        t.mode = "manual"
        act, _ = decide(t, self._s(), 6800.25, "",
                        m5_against="the 5m turned")
        assert act == ""

    def test_an_auto_trade_is(self):
        act, why = decide(self._open(), self._s(), 6800.25, "",
                          m5_against="the 5m turned")
        assert (act, why) == ("close", "m5")

    # -- the wiring -------------------------------------------------------

    def test_the_route_reads_the_ordinary_5m_candle(self):
        """Not a mid-price window and not a counted bar -- the bar
        Chris can see on his own chart."""
        blk = py_func("_m5_now")
        assert 'tape.candle("5m")' in blk

    def test_the_route_only_closes_fused_trades_on_the_5m(self):
        blk = py_func("_journal_step")
        assert "hold_against_m5(" in blk
        assert 'if strat == "fused" and t is not None and t.open:' in blk
        assert "m5_against=m5_against" in blk

    def test_the_state_carries_all_three(self):
        st = fu.state({"dir": "up"}, {"read": "up"},
                      {"dir": "up", "seconds": 200})
        assert st["tick"]["dir"] == "up"
        assert st["ofi"]["dir"] == "up"
        assert st["m5"]["dir"] == "up"
        assert st["side"] == "up"
        assert "all three" in st["say"]

    def test_the_state_separates_the_pair_from_the_verdict(self):
        """Pair agreeing without the 5m, and the pair disagreeing,
        used to look identical on the panel."""
        st = fu.state({"dir": "up"}, {"read": "up"},
                      {"dir": "down", "seconds": 200})
        assert st["pair"] == "up"
        assert st["side"] == ""

    def test_the_panel_shows_the_5m_box(self):
        blk = js_func("paintFusion")
        assert "5 MINUTE BAR" in blk
        assert "the hold" in blk

    def test_the_panel_does_not_say_disagreement_exits(self):
        """It did, and that reading is now wrong."""
        blk = js_func("paintFusion")
        assert "an open trade ends here" not in blk
        assert "held until the" in blk
