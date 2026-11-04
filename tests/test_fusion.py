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

    def test_only_fused_trades_use_the_three_way_exit(self):
        """The exit reading is taken for this strategy and no other."""
        blk = py_func("_journal_step")
        assert 'if strat == "fused" and t is not None and t.open:' in blk
        assert 'disagreed = ""' in blk


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
        """The entry and the exit are the same test now, and the
        panel has to say the one that is true today."""
        blk = js_func("paintFusion")
        assert "turning closes" in blk

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


class TestTheFiveMinuteLeg:
    """Chris, after running the hold version:

        "the 5 minute minute we use for this is highly inaccurate
        compared to the 5 minute on chris charts it turns extremely
        late also on auto trad after one trade hit the take profit it
        immediately jumps in for another trade without making sure
        the parameters still hold true remember all 3 must agree for
        a entry if one disagrees it exits that simple"

    Three consequences: the 5m is read immediately, the exit is
    symmetric with the entry again, and entries are capped at one per
    5-minute bar.
    """

    @staticmethod
    def _open(side="up"):
        return Trade(id=1, coin="ES", side=side, mode="auto",
                     open_px=6800.0, strategy="fused")

    @staticmethod
    def _s():
        return Settings(coin="ES", auto=True, tp_on=False, sl_on=False,
                        strategy="fused")

    # -- read immediately -------------------------------------------------

    def test_the_5m_is_not_held_back(self):
        """"the 5 minute is the only one that dont need to be settled"."""
        assert fu.M5_SETTLE_S == 0.0
        assert fu.m5_direction({"dir": "up", "seconds": 1}) == ("up", "")

    def test_a_brand_new_bar_still_reads(self):
        assert fu.m5_direction({"dir": "down", "seconds": 0})[0] == "down"

    def test_a_flat_bar_has_no_direction(self):
        """Flat is still not a side -- that never changed."""
        assert fu.m5_direction({"dir": "flat", "seconds": 300})[0] == ""

    def test_a_settle_can_still_be_asked_for(self):
        """Kept as a parameter so it can be put back."""
        assert fu.m5_direction({"dir": "up", "seconds": 10},
                               settle_s=90.0)[0] == ""

    def test_it_matches_what_colours_the_candle(self):
        """Close against open, the same as the chart.

        A confirmation that disagrees with what Chris can see on his
        own screen is worse than no confirmation.
        """
        blk = py_func("_m5_now")
        assert '"up" if cl > o else "down" if cl < o else "flat"' in blk
        assert 'tape.candle("5m")' in blk

    # -- the exit is the entry, run again ---------------------------------

    def test_all_three_holds(self):
        assert fu.holds_three("up", "up", "up", "up") == (True, "")

    @pytest.mark.parametrize("t,o,m", [("down", "up", "up"),
                                       ("up", "down", "up"),
                                       ("up", "up", "down")])
    def test_any_one_turning_ends_it(self, t, o, m):
        ok, why = fu.holds_three("up", t, o, m)
        assert ok is False
        assert why.strip()

    @pytest.mark.parametrize("t,o,m", [("", "up", "up"),
                                       ("up", "", "up"),
                                       ("up", "up", "")])
    def test_any_one_going_flat_ends_it(self, t, o, m):
        """One of them going flat is a disagreement: the reason for
        the trade was that all three pointed the same way."""
        assert fu.holds_three("up", t, o, m)[0] is False

    def test_the_exit_cannot_drift_from_the_entry(self):
        """`holds_three` is defined in terms of `entry_side`.

        This pair has been rewritten twice; entering on one
        definition and exiting on another is the bug that hides
        until a live position is sitting in it.
        """
        for t in ("up", "down", "flat"):
            for o in ("up", "down", "flat"):
                for m in ("up", "down", "flat"):
                    side = fu.entry_side(t, o, m)[0]
                    for guess in ("up", "down"):
                        assert fu.holds_three(guess, t, o, m)[0] == (
                            side == guess), (t, o, m, guess)

    def test_the_journal_closes_on_a_disagreement(self):
        why = should_close(self._open(), self._s(), 6800.25, "",
                           disagreed="the 5m is flat")
        assert why == "disagree"

    def test_it_holds_when_nothing_disagreed(self):
        assert should_close(self._open(), self._s(), 6800.25, "up") == ""

    def test_the_stop_still_wins(self):
        s = Settings(coin="ES", auto=True, sl_on=True, sl_ticks=2)
        assert should_close(self._open(), s, 6799.0, "",
                            disagreed="x") == "sl"

    def test_a_manual_trade_is_not_closed_by_it(self):
        t = self._open()
        t.mode = "manual"
        assert decide(t, self._s(), 6800.25, "", disagreed="x")[0] == ""

    def test_an_auto_trade_is(self):
        assert decide(self._open(), self._s(), 6800.25, "",
                      disagreed="x") == ("close", "disagree")

    # -- one trade per 5m bar ---------------------------------------------

    def test_a_second_entry_in_the_same_bar_is_refused(self):
        """The take-profit bug: it banked a long and stepped straight
        back into the same move."""
        act, why = decide(None, self._s(), 6800.0, "up",
                          taken=("up",), max_per_bar=1)
        assert act == ""

    def test_even_the_other_direction_is_refused(self):
        """A per-direction cap does not fix it -- the trade after a
        long is usually another long, but the other way must be
        capped too or the bar can still take two."""
        act, why = decide(None, self._s(), 6800.0, "down",
                          taken=("up",), max_per_bar=1)
        assert act == ""
        assert "1 trade" in why

    def test_an_empty_bar_opens(self):
        assert decide(None, self._s(), 6800.0, "up", taken=(),
                      max_per_bar=1) == ("open", "up")

    def test_the_other_strategies_keep_their_cap(self):
        act, _ = decide(None, Settings(coin="ES", auto=True),
                        6800.0, "down", taken=("up",))
        assert act == "open", "two per candle, one per direction"

    def test_the_route_counts_against_the_5m_bar(self):
        """Not the 15-minute window: a cap from that clock would
        refuse a good entry because of a trade twelve minutes old."""
        blk = py_func("_journal_step")
        assert 'bar_ts = float((fus.get("m5") or {}).get("start_ts") or 0.0)' \
            in blk
        assert "cap = 1" in blk
        assert "max_per_bar=cap" in blk

    def test_the_route_passes_all_three_to_the_exit(self):
        blk = py_func("_journal_step")
        assert "holds_three(" in blk
        assert "disagreed=disagreed" in blk

    def test_the_cap_makes_the_churn_arithmetically_impossible(self):
        """Fourteen trades in a few minutes cannot recur.

        One per 5-minute bar is twelve an hour, whatever the exit
        does. That is the reason the symmetric exit is safe to use
        again rather than a hope that it behaves.
        """
        taken = ()
        opens = 0
        for _ in range(50):                 # fifty polls, one 5m bar
            act, _why = decide(None, self._s(), 6800.0, "up",
                               taken=taken, max_per_bar=1)
            if act == "open":
                opens += 1
                taken = ("up",)             # the journal records it
        assert opens == 1

    # -- the panel --------------------------------------------------------

    def test_the_panel_says_any_one_turning_closes_it(self):
        blk = js_func("paintFusion")
        assert "any one of the three turning closes" in blk \
            or "any one turning closes it" in blk

    def test_the_panel_states_the_cap(self):
        blk = js_func("paintFusion")
        assert "one trade per 5-minute bar" in blk

    def test_the_panel_no_longer_claims_a_hold(self):
        blk = js_func("paintFusion")
        assert "held until the <b>5m</b> turns" not in blk
