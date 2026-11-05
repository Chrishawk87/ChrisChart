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
        st = fu.state({"dir": "up"}, {"read": "up"}, {"dir": "down"})
        side, why = window_signal({}, "fused", "", st)
        assert side == ""
        assert why == "the tick bar is up and the 5m is down"

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
        assert "5 MINUTE BAR" in blk
        assert "prints and price only" in blk
        assert "decides nothing" in blk, "the OFI box must say so"

    def test_the_verdict_is_the_loud_part(self):
        blk = js_func("paintFusion")
        assert "BOTH" in blk and "NO ENTRY" in blk

    def test_it_says_what_actually_ends_a_trade(self):
        """The entry and the exit are the same test now, and the
        panel has to say the one that is true today."""
        blk = js_func("paintFusion")
        assert "tick bar" in blk and "turns against it" in blk

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

    def test_it_holds_while_both_still_point_that_way(self):
        assert fu.holds_exit("up", "up", "up") == (True, "")

    def test_the_tick_bar_turning_ends_it(self):
        ok, why = fu.holds_exit("up", "down", "up")
        assert ok is False
        assert why == "the tick bar turned down against a up"

    def test_the_5m_turning_ends_it(self):
        ok, why = fu.holds_exit("up", "up", "down")
        assert ok is False
        assert why == "the 5m turned down against a up"

    def test_the_ofi_cannot_end_it(self):
        """"the ones that have to disagree in order for a exit to
        happen is the tick bar and the 5 minute".

        The exit test is not even GIVEN the OFI, so there is no way
        for it to leak back in.
        """
        import inspect
        sig = inspect.signature(fu.holds_exit)
        assert "ofi" not in " ".join(sig.parameters), sig

    def test_the_ofi_does_not_gate_entry_either(self):
        """"a entry follows the same rules both the tick bar and the
        5 minute has to agree". The OFI is shown and ignored."""
        st_against = fu.state({"dir": "up"}, {"read": "down"},
                              {"dir": "up"})
        assert st_against["side"] == "up", "the book disagreeing is not a veto"
        assert st_against["ofi_concurs"] is False

    def test_the_5m_is_required_to_enter(self):
        """A MUTATION SURVIVED HERE and this is why it exists.

        Deleting the 5m check from `entry_side` -- so the fast pair
        alone opened a trade -- broke no test, because the entry
        cases went out with a superseded class and nothing replaced
        them. The exit tests were thorough and the entry had a hole
        straight through it.
        """
        assert fu.entry_side("up", "")[0] == ""
        assert fu.entry_side("up", "flat")[0] == ""

    def test_a_flat_leg_is_named_as_flat_not_left_blank(self):
        """The reason is what shows on the panel.

        Two mutations survived here that changed no decision at all
        -- removing either flat guard still refuses the entry,
        because the directions differ anyway. What they broke was
        the sentence: "the tick bar is up and the 5m is " with
        nothing after it. The rule was fine and the explanation was
        garbage, which on this panel has been the more expensive
        failure.
        """
        assert fu.entry_side("up", "")[1] == (
            "the tick bar is up but the 5m is flat")
        assert fu.entry_side("", "up")[1] == (
            "the 5m is up but the tick bar is flat")
        assert fu.entry_side("up", "flat")[1].endswith("the 5m is flat")
        assert fu.entry_side("flat", "up")[1].endswith(
            "the tick bar is flat")

    def test_no_reason_ends_mid_sentence(self):
        for t in ("up", "down", "flat", ""):
            for m in ("up", "down", "flat", ""):
                why = fu.entry_side(t, m)[1]
                if why:
                    assert not why.rstrip().endswith("is"), (t, m, why)
                    assert why == why.strip(), (t, m, why)

    def test_a_5m_pointing_the_other_way_blocks_entry(self):
        side, why = fu.entry_side("up", "down")
        assert side == ""
        assert "the 5m is down" in why

    def test_every_leg_is_load_bearing_on_the_way_in(self):
        """One at a time, each turned off, nothing opens."""
        assert fu.entry_side("up", "up")[0] == "up"
        for t, m in (("", "up"), ("up", ""), ("flat", "up"),
                     ("up", "flat"), ("down", "up"), ("up", "down")):
            assert fu.entry_side(t, m)[0] == "", (t, m)

    def test_the_strategy_refuses_without_the_5m(self):
        """Through the journal, not just the helper."""
        st = fu.state({"dir": "up"}, {"read": "up"}, {"dir": "flat"})
        assert window_signal({}, "fused", "", st)[0] == ""

    @pytest.mark.parametrize("t,m", [("", "up"), ("up", ""), ("", "")])
    def test_flat_is_not_a_disagreement(self, t, m):
        """Chris's own answer for the 5m, applied to both: "flat is
        not a reason to exit". A close sitting exactly on its open
        has not turned against anything."""
        assert fu.holds_exit("up", t, m) == (True, "")

    def test_a_short_is_the_mirror(self):
        assert fu.holds_exit("down", "down", "down")[0] is True
        assert fu.holds_exit("down", "up", "down")[0] is False
        assert fu.holds_exit("down", "down", "up")[0] is False

    def test_no_side_holds_nothing(self):
        assert fu.holds_exit("", "up", "up")[0] is False

    def test_the_reason_is_never_blank_when_it_closes(self):
        for t in ("up", "down", "flat", ""):
            for m in ("up", "down", "flat", ""):
                ok, why = fu.holds_exit("up", t, m)
                if not ok:
                    assert why.strip(), (t, m)

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

    def test_the_route_passes_only_two_legs_to_the_exit(self):
        blk = py_func("_journal_step")
        assert "holds_exit(" in blk
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

    def test_the_panel_names_the_two_legs_that_close_it(self):
        """And says the OFI does not, because that asymmetry is the
        single most surprising thing about the rule."""
        blk = js_func("paintFusion")
        assert "tick bar" in blk and "5m" in blk
        assert "does not open anything" in blk, \
            "the OFI gating nothing is the surprising part"

    def test_the_panel_states_the_cap(self):
        blk = js_func("paintFusion")
        assert "one trade per 5-minute bar" in blk

    def test_the_panel_no_longer_claims_a_hold(self):
        blk = js_func("paintFusion")
        assert "held until the <b>5m</b> turns" not in blk
