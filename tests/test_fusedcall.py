"""The reader that weighs the fused panel instead of gating on it.

Chris: "i want you to build in a agent that can read the fused bar
section like a human only whats there ... make a calculated
decision while reading actual price plus what the fused bar section
is saying at the same time".
"""

from __future__ import annotations

import inspect
import pathlib

import pytest

from liqmap import fusedcall as fc

ROOT = pathlib.Path(__file__).resolve().parents[1]

UP = dict(ofi_dir="up", tick_dir="up", where="settling", entry="",
          open=7180.0, body_pct=0.8,
          turning={"dir": "up", "travel": 14.0},
          tf={"on": True, "dir": "up", "name": "5m", "body": 0.7,
              "travel": 20.0},
          tf2={"on": True, "dir": "up", "name": "15m", "body": 0.6,
               "travel": 30.0})


class TestItSuggestsAndNeverExecutes:
    """THE STANDING RULE OF THIS WHOLE PACKAGE. It is checked by
    name here as it is in every other module that produces a
    side."""

    def test_no_function_is_named_as_though_it_traded(self):
        src = (ROOT / "liqmap" / "fusedcall.py").read_text()
        for bad in ("def submit(", "def execute(", "def place(",
                    "def order(", "def buy(", "def sell("):
            assert bad not in src, f"fusedcall has {bad}"

    def test_the_payload_says_it_decides_nothing(self):
        assert fc.read(UP, price=7183.0)["decides"] is False
        assert fc.read({}, price=0.0)["decides"] is False

    def test_the_rule_cannot_see_it(self):
        """A READER THE RULE READS IS NOT A READER, IT IS A RULE.
        Chris has had a panel claim a rule nobody was running once
        already and it cost him a day."""
        from liqmap import fusedread as fr
        for fn in (fr.entry_side, fr.exit_reason, fr.read, fr.state):
            ps = inspect.signature(fn).parameters
            for bad in ("call", "score", "reader", "agent"):
                assert bad not in ps, f"{fn.__name__} reads {bad}"
        src = (ROOT / "liqmap" / "fusedread.py").read_text()
        assert "fusedcall" not in src, \
            "the rule imports the reader"


class TestItWeighsRatherThanGates:
    """THE WHOLE DIFFERENCE FROM THE RULE, and the reason this
    module exists beside `fusedread` rather than inside it."""

    def test_everything_agreeing_calls_the_side(self):
        r = fc.read(UP, price=7183.0)
        assert r["side"] == "up"
        assert r["score"] > 0.9
        assert r["speaking"] == r["of"] == 6

    def test_a_split_panel_calls_nothing(self):
        split = dict(UP, tick_dir="down",
                     turning={"dir": "down", "travel": 3.0})
        r = fc.read(split, price=7179.0)
        assert r["side"] == ""
        assert "split" in r["say"]
        assert "price" in r["say"] and "book" in r["say"]

    def test_one_dissenter_does_not_veto_it(self):
        """THE POINT OF WEIGHING. The rule cannot tell five rows
        agreeing and one blank from five agreeing and one against
        -- both are "no". This can."""
        one_against = dict(UP, tf2={"on": True, "dir": "down",
                                    "name": "15m", "body": 0.2,
                                    "travel": 2.0})
        r = fc.read(one_against, price=7183.0)
        assert r["side"] == "up", "a single weak dissenter vetoed it"
        assert "tf2" in r["against"]

    def test_the_dissent_is_always_named(self):
        """A CALL WITH NO DISSENT LISTED IS A CALL THAT CANNOT BE
        CHECKED."""
        one_against = dict(UP, tf2={"on": True, "dir": "down",
                                    "name": "15m", "body": 0.2,
                                    "travel": 2.0})
        r = fc.read(one_against, price=7183.0)
        assert "against it" in r["say"]
        assert "tf2" in r["say"]

    def test_two_strong_dissenters_do_hold_it_off(self):
        """FOUR AGAINST TWO IS NOT A CALL. Both filters pointing
        the other way with real bodies behind them is a contested
        bar, and saying so is the job."""
        heavy = dict(UP, ofi_dir="down", tick_dir="down",
                     turning={"dir": "down", "travel": 20.0})
        r = fc.read(heavy, price=7175.0)
        assert r["side"] == ""
        assert r["score"] < 0.0, "it did not even lean the right way"
        assert set(r["quiet"]) == set()

    def test_everything_pointing_down_calls_down(self):
        down = dict(UP, ofi_dir="down", tick_dir="down",
                    turning={"dir": "down", "travel": 20.0},
                    tf={"on": True, "dir": "down", "name": "5m",
                        "body": 0.7, "travel": 20.0},
                    tf2={"on": True, "dir": "down", "name": "15m",
                         "body": 0.6, "travel": 30.0})
        r = fc.read(down, price=7175.0)
        assert r["side"] == "down"
        assert r["score"] < -0.9

    def test_the_threshold_is_what_decides_the_call(self):
        lean = dict(UP, tf={"on": False}, tf2={"on": False},
                    turning={"dir": "down", "travel": 2.0})
        strict = fc.read(lean, price=7183.0, call_at=0.95)
        loose = fc.read(lean, price=7183.0, call_at=0.10)
        assert strict["side"] == ""
        assert loose["side"] == "up"
        assert strict["score"] == loose["score"], \
            "the threshold changed the reading, not just the call"


class TestARowThatIsOffIsNotARowVotingZero:
    """Otherwise a switched-off filter would halve every score it
    touched while telling you nothing."""

    def test_a_filter_that_is_off_leaves_the_denominator(self):
        both = fc.read(UP, price=7183.0)
        one = fc.read(dict(UP, tf2={"on": False}), price=7183.0)
        assert one["speaking"] == both["speaking"] - 1
        assert one["score"] > 0.9, \
            "switching a filter off weakened a unanimous panel"
        assert "tf2" in one["quiet"]

    def test_a_blank_row_leaves_it_too(self):
        blank = fc.read(dict(UP, turning={"dir": "", "travel": 0.0}),
                        price=7183.0)
        assert blank["score"] > 0.9
        assert "turning" in blank["quiet"]

    def test_a_silent_panel_scores_zero_and_says_so(self):
        r = fc.read({"open": 7180.0, "tf": {"on": False},
                     "tf2": {"on": False}}, price=7180.0)
        assert r["side"] == ""
        assert r["score"] == 0.0
        assert r["speaking"] == 0
        assert "nothing" in r["say"]

    def test_a_panel_leaning_under_the_line_says_which_way(self):
        """TWO DIFFERENT SILENCES. "nothing is pointing anywhere"
        and "five rows are pointing and disagree" are not the same
        state, and the first version of this said the former on
        both."""
        # ONE WEAK ROW, POINTING UP, AND NOTHING AGAINST IT.
        # Score is that row's own strength, which is under the
        # line -- a lean, not a split and not a silence.
        lean = {"open": 7180.0, "ofi_dir": "", "tick_dir": "",
                "tf": {"on": False},
                "tf2": {"on": True, "dir": "up", "name": "15m",
                        "body": 0.0, "travel": 0.0},
                "turning": {"dir": ""}}
        r = fc.read(lean, price=7180.0)
        assert r["side"] == "", r["say"]
        assert "leaning up" in r["say"]
        assert "nothing" not in r["say"]
        assert "split" not in r["say"]


class TestStrength:
    """Chris: "the direction shows but its strength dont is the
    move strong or just a micro pull back or whatever"."""

    def test_a_strong_row_outweighs_a_weak_one_of_equal_weight(self):
        """TWO FILTERS, SAME WEIGHT, OPPOSITE DIRECTIONS. The one
        with a body and a move behind it has to win, or `body` and
        `travel` are decoration."""
        panel = dict(ofi_dir="", tick_dir="", open=7180.0,
                     turning={"dir": ""},
                     tf={"on": True, "dir": "up", "name": "5m",
                         "body": 0.95, "travel": 30.0},
                     tf2={"on": True, "dir": "down", "name": "15m",
                          "body": 0.02, "travel": 0.0})
        r = fc.read(panel, price=7180.0, call_at=0.25)
        assert r["side"] == "up"
        assert r["score"] > 0.0
        # AND THE OTHER WAY ROUND, so this is not just "tf wins".
        flipped = dict(panel,
                       tf={"on": True, "dir": "up", "name": "5m",
                           "body": 0.02, "travel": 0.0},
                       tf2={"on": True, "dir": "down", "name": "15m",
                            "body": 0.95, "travel": 30.0})
        assert fc.read(flipped, price=7180.0, call_at=0.25)["side"] \
            == "down"

    def test_a_direction_with_nothing_behind_it_still_counts(self):
        """THE FLOOR IS NOT ZERO. A row that points up counts for
        something even on a bar with no body yet -- the point is
        that strong and weak differ, not that weak is worthless."""
        assert fc._strength(body=0.0, travel=0.0) > 0.0
        assert fc._strength(body=1.0, travel=0.0) == pytest.approx(1.0)

    def test_the_better_of_body_and_travel_wins(self):
        """A bar with almost no body that has carried fifteen
        ticks is a strong move read mid-pullback, and averaging
        the two would call it weak."""
        carried = fc._strength(body=0.02, travel=30.0)
        bodied = fc._strength(body=1.0, travel=0.0)
        assert carried == pytest.approx(bodied)

    def test_strength_is_bounded(self):
        for b, t in ((5.0, 500.0), (-3.0, -9.0), (0.0, 0.0)):
            v = fc._strength(body=b, travel=t)
            assert 0.0 <= v <= 1.0


class TestTheLivePriceLeg:
    """Chris asked for a decision made from "real life price and
    the fused bar section ... at the same time". Every other row is
    a reading OF the tape; this one is the ticker against the
    number the bar opened at."""

    def test_it_reads_price_against_the_bars_open(self):
        assert fc.live_dir(7183.0, 7180.0) == "up"
        assert fc.live_dir(7177.0, 7180.0) == "down"

    def test_a_single_tick_is_not_a_direction(self):
        """Price sits on the open constantly; reading that as a
        side would make this row flicker on every print and lend
        its weight to whichever way the last one went."""
        assert fc.live_dir(7180.25, 7180.0) == ""
        assert fc.live_dir(7179.75, 7180.0) == ""
        assert fc.live_dir(7180.50, 7180.0) == "up"

    def test_a_missing_price_is_not_a_direction(self):
        assert fc.live_dir(0.0, 7180.0) == ""
        assert fc.live_dir(7180.0, 0.0) == ""

    def test_the_live_leg_reaches_the_call(self):
        bare = {"open": 7180.0, "tf": {"on": False},
                "tf2": {"on": False}, "turning": {"dir": ""}}
        up = fc.read(bare, price=7185.0, call_at=0.5)
        down = fc.read(bare, price=7175.0, call_at=0.5)
        assert up["side"] == "up"
        assert down["side"] == "down"


class TestItSaysWhenItIsAheadOfTheRule:
    """THE USEFUL PART. Chris: "the move has already happened on
    iron beam by the time it shows up on chris chart". About
    fifteen seconds of that is the rule's own -- a settle gate and
    a five-second hold -- and this waits for neither."""

    def test_it_calls_while_the_rule_is_still_settling(self):
        r = fc.read(UP, price=7183.0)
        assert r["side"] == "up"
        assert r["rule_side"] == ""
        assert r["rule_where"] == "settling"
        assert r["ahead_of_rule"] is True
        assert "settling" in r["say"]

    def test_it_is_not_ahead_when_the_rule_has_fired(self):
        fired = dict(UP, where="together", entry="up")
        r = fc.read(fired, price=7183.0)
        assert r["ahead_of_rule"] is False

    def test_it_is_not_ahead_when_it_has_no_call(self):
        split = dict(UP, tick_dir="down",
                     turning={"dir": "down", "travel": 20.0},
                     tf={"on": True, "dir": "down", "name": "5m",
                         "body": 0.9, "travel": 30.0})
        r = fc.read(split, price=7180.0)
        assert r["side"] == ""
        assert r["ahead_of_rule"] is False


class TestJunkNeverRaises:
    def test_an_empty_panel(self):
        r = fc.read(None, price=0.0)
        assert r["side"] == ""
        assert isinstance(r["score"], float)

    def test_junk_everywhere(self):
        for bad in ("abc", None, [], {}, 0, -1):
            panel = {"ofi_dir": bad, "tick_dir": bad, "open": bad,
                     "body_pct": bad, "entry": bad, "where": bad,
                     "turning": {"dir": bad, "travel": bad},
                     "tf": {"on": bad, "dir": bad, "body": bad,
                            "travel": bad, "name": bad},
                     "tf2": bad}
            r = fc.read(panel, price=bad, call_at=bad)
            assert r["side"] in ("", "up", "down")
            assert isinstance(r["score"], float)
            assert r["decides"] is False

    def test_a_nonsense_threshold_cannot_call_every_bar(self):
        """Zero would make it name a side on every bar, which is
        the same as naming one on none of them."""
        quiet = {"open": 7180.0, "tf": {"on": False},
                 "tf2": {"on": False}, "turning": {"dir": ""}}
        assert fc.read(quiet, price=7180.0, call_at=0.0)["side"] == ""


class TestItIsWiredThrough:
    @staticmethod
    def _gate():
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("def _fused_gate(")
        return src[i:src.index("\n    def ", i + 10)]

    def test_the_gate_runs_it_over_its_own_payload(self):
        """NOT A SECOND READING OF THE SAME BAR. A reader that
        fetched its own copy would eventually disagree with the
        panel beside it, which is how the 5-minute cell and the
        OFI bar each came to contradict the rule they were drawn
        next to."""
        fn = self._gate()
        assert "_fcall.read(out" in fn, \
            "the reader is handed something other than the panel"
        assert "import fusedcall as _fcall" in \
            (ROOT / "liqmap" / "web.py").read_text()

    def test_every_payload_path_carries_a_call(self):
        fn = self._gate()
        assert fn.count('"call"') >= 2, \
            "one of the two payload paths has no reader"

    def test_the_panel_draws_it_and_says_it_is_shown_only(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("function paintFusedGate(")
        box = src[i:src.index("\nfunction ", i + 10)]
        assert "FG.call" in box, "the call is never drawn"
        assert "READS " in box
        assert "shown only" in box, \
            "nothing says the reader opens and closes nothing"
        assert "ahead_of_rule" in box, \
            "the panel cannot say it is ahead of the rule"

    def test_it_actually_renders(self, page_js):
        """RENDERED, NOT SCANNED. The source test above passes
        with the call hidden behind `false` -- the text is still
        in the file -- and it passed with "shown only" deleted
        from this line, because the turning row says it too."""
        from tests.test_fused_gate import TestThePanelActuallyRenders
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**TestThePanelActuallyRenders.BAR,
                       "tick_dir": "up", "ofi_dir": "up",
                       "where": "settling", "agree": True, "side": "",
                       "entry": "",
                       "turning": {"dir": "", "ticks_off": 0.0,
                                   "travel": 0.0, "band": 4.0,
                                   "decides": False},
                       "call": {"side": "up", "score": 0.87,
                                "speaking": 5, "of": 6,
                                "ahead_of_rule": True,
                                "say": "book, price say up",
                                "decides": False}})
        assert "READS UP" in html, "the call is not drawn"
        assert "87%" in html
        assert "5 of 6 rows speaking" in html
        assert "book, price say up" in html
        assert "it opens and closes nothing" in html, \
            "nothing on this line says the reader never trades"
        assert "ahead of the rule" in html

    def test_a_no_call_still_draws_and_still_disclaims(self, page_js):
        from tests.test_fused_gate import TestThePanelActuallyRenders
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**TestThePanelActuallyRenders.BAR,
                       "tick_dir": "up", "ofi_dir": "up",
                       "where": "together", "agree": True, "side": "up",
                       "turning": {"dir": "", "ticks_off": 0.0,
                                   "travel": 0.0, "band": 4.0,
                                   "decides": False},
                       "call": {"side": "", "score": 0.1,
                                "speaking": 4, "of": 6,
                                "ahead_of_rule": False,
                                "say": "split -- book up against price down",
                                "decides": False}})
        assert "NO CALL" in html
        assert "split" in html
        assert "it opens and closes nothing" in html
        assert "ahead of the rule" not in html

    def test_the_live_route_carries_it(self, client, monkeypatch):
        from liqmap import web
        from tests.test_cme_panels import _cme
        _cme(monkeypatch)
        # THE SETTINGS ROW IS WHAT MAKES THE JOURNAL BLOCK APPEAR.
        # Without it the whole block comes back empty and this
        # test passes or fails for a reason that has nothing to do
        # with the reader.
        rt = web.runtime()
        st = rt.journal.settings("ES")
        st.strategy = "together"
        rt.journal.save_settings(st)
        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH).json()
        jr = d.get("journal") or {}
        assert "error" not in jr, jr.get("error")
        call = (jr.get("fusedgate") or {}).get("call")
        assert call is not None, "no reader on the live panel"
        assert call.get("decides") is False


from tests.test_cme_panels import client  # noqa: E402,F401
from tests.test_stack_strip import page_js  # noqa: E402,F401

# THE TOKEN THE TEST APP ACTUALLY RUNS WITH. A wrong one here does
# not fail loudly -- the route returns 401, the journal block comes
# back empty, and the test reports "no reader on the live panel",
# which is a true sentence about entirely the wrong thing.
AUTH = {"Authorization": "Bearer t0ken"}
