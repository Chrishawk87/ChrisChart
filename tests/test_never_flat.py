"""Both gates always show a direction, and both show a real candle.

Chris, on the version before this one:

    "there are 2 things that is actually wrong with this one ... the
    gate 1 should never say flat at all as long as there is a 5000t
    candle it should show the trading direction of that candle,
    secondly it isnt showing the real time continuous 15 or 5 minute
    candle because if it was it would see that both the candle and
    the tickbar are actually going the same way and it dont even
    realize it so one always show what the real 15 minute is doing
    and thet other always what the tick chart is doing when they
    agree thats the direction we take"

FAULT ONE: GATE 1 WAS NOT READING THE CANDLE.

    It read a VOTE -- how many prints moved price up against how
    many moved it down -- behind a 2% margin band and a minimum
    count of 250. Both of those produce FLAT, by design, and flat
    was what the panel reported while the 5,000-tick chart in front
    of him drew a plain green candle.

    The band and the floor were put there to stop flipping, and
    they did. They were the wrong instrument: they were applied to
    the thing the gate reports instead of to the thing that opens a
    trade. The dwell is what stops flips turning into trades, and
    the dwell is untouched.

    Gate 1 is now the candle's own direction -- close against open,
    the number the chart colours itself with. The vote is still
    computed and still drawn beside it, because a reading that
    silently stops being collected is how a screen starts lying.

FAULT TWO: FLAT WAS BEING REPORTED AS A READING.

    Both gates blanked when a candle's close sat exactly on its
    open. That is not a market with no direction; it is one tick
    from wherever it just was. Reporting it as no-direction meant an
    entry needing two gates to agree could be cancelled by an
    instant of arithmetic, and the agreement he could see on the
    chart was never recognised.

    A side, once seen, is now HELD until the other side replaces
    it. Nothing is invented -- before the first print there is still
    no direction -- and a real flip passes through untouched, which
    is what keeps the exit honest.

THE SECOND FAULT IS WHY THE FIRST ONE WAS INVISIBLE. Gate 2's data
path was already the live forming candle with no threshold, which
was checked before changing anything: a 15m candle 12 points off its
open reports "up" through `rows_from_candles` -> `qualified` ->
`stack_side`. It only ever blanked at an exact tie or a missing
candle. Gate 1 reading flat is what destroyed the agreement.
"""

from __future__ import annotations

import pathlib

import pytest

from liqmap import cascade as csc
from liqmap import stackwin as sw
from liqmap.tickbar import TickBars

D, U = "down", "up"
WEB = (pathlib.Path(__file__).resolve().parents[1]
       / "liqmap" / "web.py").read_text()

# An extending bar: the rule's neutral input.
PUSH = {"state": "continuation", "heavy": False}


def rungs(**kw) -> dict:
    out = {}
    for tf, q in kw.items():
        if q is None:
            out[tf] = {"present": False, "qualified": None}
        else:
            out[tf] = {"present": True, "qualified": q, "bps": -1.3}
    return out


class TestGateOneIsTheCandle:
    def test_the_route_reads_the_body_not_the_vote(self):
        i = WEB.index("def _journal_step_un(")
        blk = WEB[i:i + 30000]
        assert '_bd = _tk.get("body_dir")' in blk, \
            "gate 1 is not reading the candle's own direction"
        assert "_sw.state(win, _bd," in blk

    def test_the_candle_direction_is_close_against_open(self):
        """The same number the chart colours the candle with. A
        5,000-tick candle trading above its open is UP, whatever the
        print-by-print vote inside it did."""
        b = TickBars(per_bar=10 ** 6)
        t, px = 0.0, 7000.0
        b.on_trade(t, px, 1.0, "B")
        # A grind up on a minority of prints: price ends well above
        # the open while MORE prints ticked down than up.
        for i in range(300):
            t += 1.0
            px += 2.0 if i % 3 == 0 else -0.25
            b.on_trade(t, px, 1.0, "B")
        cur = b.current
        assert cur.close > cur.open
        assert cur.direction == "up", "the body is not the candle"
        assert cur.down_prints > cur.up_prints, \
            "the setup no longer separates body from vote"

    def test_a_candle_has_a_direction_from_its_second_print(self):
        """"as long as there is a 5000t candle". No warmup, no
        minimum count, no band -- the thing the gate reports is the
        thing the chart draws."""
        b = TickBars(per_bar=5000)
        b.on_trade(1.0, 7000.00, 1.0, "B")
        b.on_trade(2.0, 7000.25, 1.0, "B")
        assert b.current.direction == "up"
        # And the vote, at this point, says nothing at all -- which
        # is exactly what was being reported as the direction.
        assert b.current.path_dir == "flat"

    def test_the_panel_names_what_gate_one_reads(self):
        assert "GATE 1 &mdash; THE 5000T CANDLE" in WEB
        assert "close vs open, as the chart draws it" in WEB

    def test_the_vote_is_still_drawn_beside_it(self):
        """It is not deleted. When it disagrees with the body that
        is worth seeing -- it is just not the direction."""
        assert "the vote says " in WEB
        assert "shown only" in WEB


class TestNeitherGateEverReportsFlat:
    @pytest.mark.parametrize("prev", [U, D])
    def test_a_side_is_held_through_a_tie(self, prev):
        assert sw.hold_dir("flat", prev) == prev
        assert sw.hold_dir("", prev) == prev

    def test_a_real_flip_passes_straight_through(self):
        """The hold fills a gap. It must not slow a turn down --
        that is the dwell's job, and doing it twice would hold a
        losing trade open."""
        assert sw.hold_dir(U, D) == U
        assert sw.hold_dir(D, U) == D

    def test_nothing_is_invented_before_the_first_direction(self):
        assert sw.hold_dir("flat", "") == ""
        assert sw.hold_dir("", "") == ""
        assert sw.hold_dir("flat", "sideways") == ""

    def test_the_tick_direction_holds_through_a_tie(self):
        st = sw.state({}, "flat", rungs(**{"5m": U}), 300.0,
                      intent=PUSH, prev={"tick": U})
        assert st["tick"]["side"] == U
        assert st["tick"]["raw"] == "flat"
        assert st["tick"]["held_through_flat"] is True
        assert st["side"] == U, "the held direction did not reach the rule"

    def test_the_timeframe_row_still_holds_its_direction_on_screen(self):
        """It decides nothing now, but it is still drawn, and a row
        that blinks out reads as a dead market."""
        st = sw.state({}, U, rungs(**{"5m": "flat"}), 300.0,
                      intent=PUSH, prev={"candle": U})
        assert st["stack"]["side"] == U
        assert st["stack"]["raw"] == ""
        assert st["stack"]["held_through_flat"] is True

    def test_a_tie_everywhere_still_trades(self):
        """THE CASE HE WAS WATCHING, and it cannot come back now
        that only one reading gates at all."""
        st = sw.state({}, "flat", rungs(**{"5m": "flat"}), 300.0,
                      intent=PUSH, prev={"tick": D, "candle": D})
        assert st["side"] == D

    def test_the_hold_is_not_marked_when_nothing_is_held(self):
        st = sw.state({}, U, rungs(**{"5m": U}), 300.0, intent=PUSH,
                      prev={"tick": U, "candle": U})
        assert st["tick"]["held_through_flat"] is False
        assert st["stack"]["held_through_flat"] is False

    def test_a_flip_is_not_blocked_by_the_held_direction(self):
        st = sw.state({}, D, rungs(**{"5m": D}), 300.0, intent=PUSH,
                      prev={"tick": U, "candle": U})
        assert st["tick"]["side"] == D
        assert st["stack"]["side"] == D
        assert st["side"] == D

    def test_an_empty_start_is_still_empty(self):
        st = sw.state({}, "", {}, 300.0)
        assert st["side"] == ""
        assert st["held"]["tick"] == ""
        assert st["held"]["candle"] == ""


class TestTheMemoryIsTheCallersCom:
    """A module holding this itself would serve the panel and the
    rule from one history, and they would drift -- which is the
    fault the 5-minute cell and the OFI bar both had."""

    def test_the_state_returns_what_it_held(self):
        st = sw.state({}, U, rungs(**{"5m": D}), 300.0)
        assert st["held"]["tick"] == U
        assert st["held"]["candle"] == D

    def test_feeding_it_back_carries_the_direction(self):
        st = sw.state({}, U, rungs(**{"5m": U}), 300.0, intent=PUSH)
        nxt = sw.state({}, "flat", rungs(**{"5m": "flat"}), 300.0,
                       intent=PUSH, prev=st["held"])
        assert nxt["side"] == U

    def test_the_route_stores_it_and_hands_it_back(self):
        i = WEB.index("def _journal_step_un(")
        blk = WEB[i:i + 30000]
        assert 'getattr(rt_, "_gate_held", None)' in blk, \
            "the route keeps no memory, so nothing can be held"
        assert "prev=_prev" in blk
        assert 'rt_._gate_held = {**(fus.get("held")' in blk, \
            "the gates' memory is no longer stored"

    def test_there_is_no_dwell_left_to_feed(self):
        """Chris: "remove the 10 second hold completely". Pinned in
        the negative -- it has been re-wired here before."""
        i = WEB.index("def _journal_step_un(")
        blk = WEB[i:i + 30000]
        assert "_dw." not in blk and "Dwell" not in blk


class TestGateTwoIsTheRealLiveCandle:
    """Checked BEFORE changing it, because the complaint named it.
    The chain carries the live forming candle with no threshold; it
    blanked only at an exact tie, which is fault two."""

    @staticmethod
    def _chain(candles, trading="15m", period=900.0, prev=None):
        rows = csc.rows_from_candles(trading, lambda tf: candles.get(tf))
        rg = {}
        for tf, row in rows.items():
            rg[tf] = {"timeframe": tf, "present": True,
                      "qualified": ("up" if row[1] > 0 else
                                    "down" if row[1] < 0 else None),
                      "seeded": True, "bps": round(row[1], 3)}
        return sw.state({}, U, rg, period, intent=PUSH, prev=prev)

    def test_a_live_candle_above_its_open_reads_up(self):
        """It no longer gates, but it still has to READ correctly --
        it is on the screen and he compares it against Ironbeam."""
        st = self._chain({"15m": {"open": 7000.0, "close": 7012.0},
                          "5m": {"open": 7008.0, "close": 7012.0},
                          "1m": {"open": 7011.0, "close": 7012.0}})
        assert st["stack"]["side"] == U

    def test_one_tick_above_the_open_is_enough(self):
        """No band, no minimum. `qualified` is the sign of the move
        from the bar's own open and nothing else."""
        st = self._chain({"15m": {"open": 7000.0, "close": 7000.25}})
        assert st["stack"]["side"] == U

    def test_exactly_on_the_open_holds_rather_than_blanks(self):
        st = self._chain({"15m": {"open": 7000.0, "close": 7000.0}},
                         prev={"tick": U, "candle": U})
        assert st["stack"]["side"] == U
        assert st["side"] == U

    def test_the_row_carries_the_prices_it_was_computed_from(self):
        """So a disagreement with Ironbeam can be read off the
        screen rather than guessed at."""
        assert "open_px" in WEB and "last_px" in WEB

    def test_the_panel_says_it_is_the_real_live_candle_gate(self):
        assert "GATE 2 &mdash; THE REAL " in WEB
        assert "close vs open, " in WEB
        assert "it must agree to open" in WEB

    def test_the_panel_arrow_shows_what_the_gate_uses(self):
        """An arrow reading "flat" beside a verdict reading AGREES
        UP is the panel contradicting itself in the one place it
        must not."""
        assert "row(st.trading, g2side)" in WEB
        assert "const row = (name, shown)" in WEB


class TestWhatDidNotChange:
    def test_the_dwell_no_longer_gates_entries(self):
        """It did, for two revisions. Chris removed it."""
        st = sw.state({}, U, rungs(**{"5m": U}), 300.0, intent=PUSH)
        assert st["side"] == U
        assert not hasattr(sw, "DWELL_S")

    def test_the_time_stop_is_gone(self):
        assert "TRADE_BAR_S = " not in WEB
        assert not hasattr(sw, "exit_reason")

    def test_it_does_not_trade(self):
        import ast
        tree = ast.parse((pathlib.Path(sw.__file__)).read_text())
        bad = [n.name for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name in ("submit", "execute", "place", "order")]
        assert not bad, f"stackwin.py grew {bad}"
