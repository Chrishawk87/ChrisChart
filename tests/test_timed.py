"""The timed strategy: the 5-minute candle, guided by the 1-minute.

The rule, in Chris's words:

    "take trades only based on the 5 minute timeframe using the 1
    minute as a entry que for the direction 1 trade per 5 minute
    timeframe when 1 and 5 agree exit only on take profit, SL or
    candle reversal"

    "the one minute is used for the truth telling for the 5 minute
    it tells the fight thats happening ... allow the 1 minute to
    determine when to get in and out of a 5 minute candle dont wait
    for a full pullback into the negative before exiting a trade but
    to guide it"

Written because the rule it replaces bled out through its exits:
twelve of them in a few hours, 2 to 5 ticks each, every one fired by
two legs momentarily not agreeing rather than by anything the market
did.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from liqmap import timed as tm
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


def candle(o: float, c: float, **kw) -> dict:
    return {"open": o, "close": c, **kw}


class TestItDoesNotTrade:
    def test_no_order_entry_points(self):
        tree = ast.parse((ROOT / "liqmap" / "timed.py").read_text())
        names = {n.name for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for banned in ("submit", "execute", "place", "order", "buy",
                       "sell", "send_order"):
            assert banned not in names, banned

    def test_it_imports_nothing(self):
        tree = ast.parse((ROOT / "liqmap" / "timed.py").read_text())
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module:
                mods.add(n.module.split(".")[0])
        assert mods <= {"__future__"}, mods


class TestDirection:
    def test_close_above_open_is_up(self):
        assert tm.direction(candle(100.0, 101.0)) == "up"

    def test_close_below_open_is_down(self):
        assert tm.direction(candle(101.0, 100.0)) == "down"

    def test_equal_is_flat(self):
        assert tm.direction(candle(100.0, 100.0)) == "flat"

    @pytest.mark.parametrize("c", [None, {}, {"open": 0, "close": 5},
                                   {"open": 5, "close": 0},
                                   {"open": "x", "close": "y"}])
    def test_nothing_readable_is_flat_not_a_crash(self, c):
        assert tm.direction(c) == "flat"

    def test_it_is_the_same_measure_the_reversal_uses(self):
        """"crosses its open" is only meaningful if the direction IS
        which side of the open price is on. A smoothed slope here
        would make "turned" and "crossed" two different events."""
        src = (ROOT / "liqmap" / "timed.py").read_text()
        assert "c > o" in src and "c < o" in src


class TestEntry:
    def test_both_up_is_a_long(self):
        assert tm.entry_side("up", "up") == ("up", "")

    def test_both_down_is_a_short(self):
        assert tm.entry_side("down", "down") == ("down", "")

    def test_a_pullback_is_not_an_entry(self):
        """The 1-minute against the 5-minute IS the pullback. That
        is the state this rule exists to sit out."""
        side, why = tm.entry_side("down", "up")
        assert side == ""
        assert "pullback" in why

    def test_a_flat_minute_is_the_fight_unresolved(self):
        side, why = tm.entry_side("flat", "up")
        assert side == ""
        assert "the fight is open" in why

    def test_a_flat_five_is_no_direction_to_follow(self):
        side, why = tm.entry_side("up", "flat")
        assert side == ""
        assert "the 5m is flat" in why

    def test_both_flat_says_so(self):
        assert tm.entry_side("", "")[1] == (
            "neither the 1m nor the 5m has a direction")

    def test_the_reason_is_never_blank_when_there_is_no_side(self):
        for a in ("up", "down", "flat", ""):
            for b in ("up", "down", "flat", ""):
                side, why = tm.entry_side(a, b)
                if not side:
                    assert why.strip(), (a, b)

    def test_no_reason_ends_mid_sentence(self):
        for a in ("up", "down", "flat", ""):
            for b in ("up", "down", "flat", ""):
                why = tm.entry_side(a, b)[1]
                if why:
                    assert not why.rstrip().endswith("is"), (a, b, why)


class TestTheExitIsAnEventNotAnAbsence:
    """The whole reason this rule exists.

    The fusion rule exited whenever its legs stopped agreeing, and a
    leg could go flat on one print. Here the 1-minute has to have
    closed back THROUGH ITS OPEN, against the position.
    """

    def test_the_minute_turning_against_a_long_ends_it(self):
        out, why = tm.reversed_against("up", "down")
        assert out is True
        assert "closed back through its open" in why

    def test_the_minute_turning_against_a_short_ends_it(self):
        assert tm.reversed_against("down", "up")[0] is True

    def test_a_flat_minute_does_not(self):
        """An absence is not an event. Twelve exits in an afternoon
        were paid for exactly this."""
        assert tm.reversed_against("up", "flat") == (False, "")
        assert tm.reversed_against("up", "") == (False, "")

    def test_the_minute_agreeing_does_not(self):
        assert tm.reversed_against("up", "up")[0] is False

    def test_no_side_cannot_be_reversed(self):
        assert tm.reversed_against("", "down")[0] is False

    def test_it_says_which_way_when_it_fires(self):
        for side, other in (("up", "down"), ("down", "up")):
            out, why = tm.reversed_against(side, other)
            assert out and side in why and other in why


class TestState:
    def test_it_reads_both_candles(self):
        st = tm.state(candle(100.0, 101.0), candle(99.0, 101.0))
        assert st["m1"]["dir"] == "up"
        assert st["m5"]["dir"] == "up"
        assert st["side"] == "up"
        assert "both up" in st["say"]

    def test_it_carries_the_five_minute_bar_identity(self):
        """The one-per-candle cap is counted against it."""
        st = tm.state(candle(100.0, 101.0),
                      candle(99.0, 101.0, start_ts=1_700_000_000.0))
        assert st["m5"]["start_ts"] == 1_700_000_000.0

    def test_a_pullback_shows_as_no_side_with_a_reason(self):
        st = tm.state(candle(101.0, 100.0), candle(99.0, 101.0))
        assert st["side"] == ""
        assert "pullback" in st["why"]

    def test_missing_candles_are_a_reading_not_a_crash(self):
        st = tm.state(None, None)
        assert st["agree"] is False and st["say"].strip()


class TestTheStrategy:
    def test_it_is_registered(self):
        from liqmap.journal import STRATEGIES
        assert "timed" in STRATEGIES
        assert "1m" in STRATEGIES["timed"]

    def test_agreement_gives_the_side(self):
        st = tm.state(candle(100.0, 101.0), candle(99.0, 101.0))
        assert window_signal({}, "timed", "", st) == ("up", "")

    def test_it_never_reads_the_window(self):
        """Every other strategy refuses on an empty window."""
        st = tm.state(candle(101.0, 100.0), candle(101.0, 100.0))
        assert window_signal({}, "timed", "", st)[0] == "down"
        assert window_signal({}, "ofi")[1] == "no window yet"

    def test_the_reason_travels(self):
        st = tm.state(candle(101.0, 100.0), candle(99.0, 101.0))
        assert "pullback" in window_signal({}, "timed", "", st)[1]

    def test_no_reading_is_a_reason(self):
        assert window_signal({}, "timed", "", None) == (
            "", "no reading from the 1m or the 5m yet")

    def test_the_other_strategies_are_untouched(self):
        win = {"ts_last": 1000.0, "start_ts": 0.0, "mid_dir": "up",
               "ofi_dir": "up", "cvd_dir": "up"}
        assert window_signal(win, "ofi")[0] == "up"
        assert window_signal(win, "candle")[0] == "up"


class TestTheJournalExits:
    @staticmethod
    def _open(side="up"):
        return Trade(id=1, coin="ES", side=side, mode="auto",
                     open_px=6800.0, strategy="timed")

    @staticmethod
    def _s(**kw):
        return Settings(coin="ES", auto=True, strategy="timed", **kw)

    def test_a_reversal_closes_it(self):
        why = should_close(self._open(), self._s(), 6800.25, "",
                           disagreed="the 1m closed back through its open")
        assert why == "disagree"

    def test_nothing_else_signal_side_does(self):
        """No signal, no reversal, trade stays open. This is the
        change: the old rule closed here."""
        assert should_close(self._open(), self._s(), 6800.25, "") == ""

    def test_the_stop_still_works(self):
        s = self._s(sl_on=True, sl_ticks=2)
        assert should_close(self._open(), s, 6799.0, "") == "sl"

    def test_the_target_still_works(self):
        s = self._s(tp_on=True, tp_ticks=2)
        assert should_close(self._open(), s, 6800.75, "") == "tp"

    def test_the_stop_beats_the_reversal(self):
        s = self._s(sl_on=True, sl_ticks=2)
        assert should_close(self._open(), s, 6799.0, "",
                            disagreed="reversed") == "sl"


class TestOnePerFiveMinuteCandle:
    @staticmethod
    def _s():
        return Settings(coin="ES", auto=True, strategy="timed")

    def test_a_fresh_candle_allows_one(self):
        assert decide(None, self._s(), 6800.0, "up", taken=(),
                      max_per_bar=1) == ("open", "up")

    def test_a_spent_candle_allows_none(self):
        assert decide(None, self._s(), 6800.0, "up", taken=("up",),
                      max_per_bar=1)[0] == ""

    def test_not_even_the_other_way(self):
        """"A candle that turns you round twice is a candle that
        cost two spreads to end where it started."""
        assert decide(None, self._s(), 6800.0, "down", taken=("up",),
                      max_per_bar=1)[0] == ""

    def test_the_route_caps_it_against_the_five_minute_bar(self):
        blk = py_func("_journal_step_un")
        assert 'if strat in ("fused", "timed"):' in blk
        assert 'bar_ts = float((fus.get("m5") or {}).get("start_ts") or 0.0)' \
            in blk


class TestTheRouteWiring:
    def test_the_rule_reads_two_time_candles(self):
        blk = py_func("_timed_now")
        assert '_candle_now(rt_, "1m")' in blk
        assert '_candle_now(rt_, "5m")' in blk

    def test_there_is_one_candle_reader(self):
        """`_m5_now` had its own copy of the decode. Two readers is
        two places for the price frame to drift."""
        blk = py_func("_m5_now")
        assert '_candle_now(rt_, "5m")' in blk
        assert "tape.candle(" not in blk

    def test_the_step_picks_the_right_reading(self):
        blk = py_func("_journal_step_un")
        assert '_timed_now(rt_) if strat == "timed" else _fusion_now(rt_)' \
            in blk

    def test_the_exit_uses_the_reversal_not_a_disagreement(self):
        blk = py_func("_journal_step_un")
        assert "_tm.reversed_against(" in blk

    def test_the_exit_is_guided_by_the_ONE_minute(self):
        """A MUTATION SURVIVED HERE.

        Pointing the exit at the 5-minute instead of the 1-minute
        broke no test, and that swap is the whole rule: by the time a
        5-minute candle crosses its own open the move is usually
        given back. "dont wait for a full pullback into the negative
        before exiting a trade but to guide it" -- the 1-minute is
        what says so first.
        """
        for fn in ("_journal_step_un", "_advice"):
            blk = py_func(fn)
            i = blk.index("_tm.reversed_against(")
            seg = blk[i:i + 160]
            assert '(fus.get("m1")' in seg, fn
            assert '(fus.get("m5")' not in seg, f"{fn} reads the wrong candle"

    def test_the_advice_path_uses_it_too(self):
        """The alert must speak the same rule the column follows."""
        blk = py_func("_advice")
        assert "_tm.reversed_against(" in blk

    def test_the_panel_shows_what_the_rule_read(self):
        blk = py_func("_journal_step_un")
        assert 'elif strat == "timed":' in blk
        assert '"1m": (fus.get("m1") or {}).get("dir")' in blk

    def test_the_strategy_is_offered_on_the_panel(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index('id="jnlStrat"')
        box = src[i:src.index("</select>", i)]
        assert 'value="timed"' in box
