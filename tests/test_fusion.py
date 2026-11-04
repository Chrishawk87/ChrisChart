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
                      {"read": "up", "imbalance": 900, "level": 250})
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
        st = fu.state({"dir": "up"}, {"read": "up"})
        assert window_signal({}, "fused", "", st) == ("up", "")

    def test_disagreement_passes_the_reason_through(self):
        st = fu.state({"dir": "up"}, {"read": "down"})
        side, why = window_signal({}, "fused", "", st)
        assert side == ""
        assert why == "the tick bar is up and the OFI is down"

    def test_it_never_reads_the_window(self):
        """Every other strategy refuses on an empty window."""
        st = fu.state({"dir": "down"}, {"read": "down"})
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


class TestExitOnDisagreement:
    @staticmethod
    def _open(side="up"):
        # `open` is derived -- a trade is open while closed_ts is
        # None -- so there is nothing to set. Passing open=True
        # would be setting a property that does not exist.
        return Trade(id=1, coin="ES", side=side, mode="auto",
                     open_px=6800.0, strategy="fused")

    @staticmethod
    def _s():
        return Settings(coin="ES", auto=True, tp_on=False, sl_on=False,
                        strategy="fused")

    def test_it_closes_when_agreement_is_lost(self):
        why = should_close(self._open(), self._s(), 6800.25, "",
                           exit_on_disagree=True)
        assert why == "disagree"

    def test_an_opposite_signal_is_still_called_a_flip(self):
        """Both end the trade; the labels stay distinct so the
        journal can tell them apart afterwards."""
        why = should_close(self._open(), self._s(), 6800.25, "down",
                           exit_on_disagree=True)
        assert why == "flip"

    def test_it_holds_while_they_still_agree(self):
        why = should_close(self._open(), self._s(), 6800.25, "up",
                           exit_on_disagree=True)
        assert why == ""

    def test_it_is_off_for_every_other_strategy(self):
        """The same construction on the window signal closed for -1
        tick over and over: 1,582 trades became 72 when it was
        removed. It stays off unless asked for."""
        why = should_close(self._open(), self._s(), 6800.25, "")
        assert why == ""

    def test_the_stop_still_wins(self):
        s = Settings(coin="ES", auto=True, sl_on=True, sl_ticks=2)
        why = should_close(self._open(), s, 6799.0, "",
                           exit_on_disagree=True)
        assert why == "sl"

    def test_the_target_still_wins(self):
        s = Settings(coin="ES", auto=True, tp_on=True, tp_ticks=2)
        why = should_close(self._open(), s, 6800.75, "",
                           exit_on_disagree=True)
        assert why == "tp"

    def test_a_manual_trade_is_not_closed_by_it(self):
        """A trade he took by hand is his to end."""
        t = self._open()
        t.mode = "manual"
        act, why = decide(t, self._s(), 6800.25, "",
                          exit_on_disagree=True)
        assert act == ""

    def test_an_auto_trade_is(self):
        act, why = decide(self._open(), self._s(), 6800.25, "",
                          exit_on_disagree=True)
        assert (act, why) == ("close", "disagree")


class TestReEntry:
    def test_agreement_returning_opens_again(self):
        """"reenter when they agree again" -- with nothing open."""
        act, arg = decide(None, Settings(coin="ES", auto=True,
                                         strategy="fused"),
                          6800.0, "up", taken=(),
                          exit_on_disagree=True)
        assert (act, arg) == ("open", "up")

    def test_the_route_lifts_the_per_candle_cap_for_this_strategy(self):
        """A one-per-direction cap would allow the first re-entry and
        refuse every one after it, which is not the rule."""
        blk = py_func("_journal_step")
        assert 'if strat == "fused":' in blk
        assert "taken = ()" in blk

    def test_the_cap_still_applies_to_the_others(self):
        act, arg = decide(None, Settings(coin="ES", auto=True),
                          6800.0, "up", taken=("up",))
        assert act == ""
        assert "already taken its up" in arg


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

    def test_exit_on_disagree_is_passed_only_for_fused(self):
        blk = py_func("_journal_step")
        assert 'exit_on_disagree=(strat == "fused")' in blk


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
        assert "AGREE" in blk and "DISAGREE" in blk

    def test_it_says_an_open_trade_ends_on_disagreement(self):
        blk = js_func("paintFusion")
        assert "an open trade ends" in blk

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
