"""The tick chart against the clock, which is the push and the pull.

Chris: "what if we created a hybrid chart a real live continuous time
chart paired with a live continuous feed tick chart and traded the
difference in the 2 using the tick chart and the time chart as push
pull kinda like the 5000x2000 ofi we created".

THE DIFFERENCE BETWEEN THOSE TWO CHARTS IS ALREADY A NUMBER.

    A tick bar closes on a count of prints, so the SECONDS it took
    are the tape's speed -- that is the only thing a tick bar
    measures that a time bar cannot. `TickBars.pace()` compares the
    forming bar's projected span against the median of recent closed
    ones. Under 1 the prints are arriving faster than lately and the
    tick series is running ahead of the clock; over 1 it is dragging
    behind it.

    A second chart drawn beside the first would show the same
    relationship and leave the comparison to the eye. This is the
    comparison, computed, off one feed.

IT WAS ALREADY COMPUTED AND NOBODY COULD SEE IT. `pace()` reached the
CHART route and stopped there. The fusion panel and the stack+window
panel -- the two places the direction is actually read -- never got
it, so the number existed and decided nothing and informed nobody.

AND IT STILL DECIDES NOTHING. Fast is not the same as up. The entry
is the vote alone, exactly as it was; this is a reading drawn under a
label that says so. Every unlabelled number on these panels has
eventually been read as part of the rule.
"""

from __future__ import annotations

import pathlib

import pytest

from liqmap import tickbar as tb
from liqmap.tickbar import (FADE_RATIO, PACE_MIN_BARS, PUSH_RATIO,
                            TickBars, pace_push)

WEB = (pathlib.Path(__file__).resolve().parents[1]
       / "liqmap" / "web.py").read_text()


def built(spans, building_s=0.0, built_prints=0, per_bar=1000):
    """A builder with `spans` closed bars of known duration.

    Driven through `on_trade` rather than assembled by hand, because
    a helper that fabricates bars tests the helper.
    """
    b = TickBars(per_bar=per_bar)
    t, px = 0.0, 7000.0
    for span in spans:
        dt = span / per_bar
        for _ in range(per_bar):
            t += dt
            px += 0.25
            b.on_trade(t, px, 1.0, "B")
    if built_prints:
        dt = (building_s / built_prints) if built_prints else 0.0
        for _ in range(built_prints):
            t += dt
            px += 0.25
            b.on_trade(t, px, 1.0, "B")
    return b


class TestTheOneWordReading:
    def test_faster_than_lately_is_a_push(self):
        assert pace_push(0.4, 30) == "push"
        assert pace_push(PUSH_RATIO, 30) == "push"

    def test_slower_than_lately_is_a_fade(self):
        assert pace_push(3.0, 30) == "fade"
        assert pace_push(FADE_RATIO, 30) == "fade"

    def test_in_between_is_even(self):
        assert pace_push(1.0, 30) == "even"

    def test_an_unknown_pace_is_blank_and_not_even(self):
        """A cold start read as "even" is a quiet tape that is not
        quiet. Blank means NOT MEASURED and the panel draws the
        count instead."""
        assert pace_push(0.4, PACE_MIN_BARS - 1) == ""
        assert pace_push(0.0, 30) == ""
        assert pace_push(None, 30) == ""

    @pytest.mark.parametrize("junk", ["", "fast", object()])
    def test_rubbish_does_not_become_a_reading(self, junk):
        assert pace_push(junk, 30) == ""

    def test_the_words_and_the_one_word_cannot_drift(self):
        """Both read the same two constants. They used to be typed
        into `_pace_words` as literals, which is one edit away from
        a panel saying "busier than usual" beside the word EVEN."""
        src = (pathlib.Path(tb.__file__)).read_text()
        i = src.index("def _pace_words(")
        blk = src[i:i + 1400]
        assert "PUSH_RATIO" in blk and "FADE_RATIO" in blk
        assert "PACE_MIN_BARS" in blk
        assert "ratio <= 0.8" not in blk, "a threshold was typed in again"
        assert "ratio < 1.25" not in blk


class TestItComesOffRealBars:
    def test_a_quickening_tape_reads_push(self):
        """Ten closed bars of 100s, then one running at a quarter
        of that pace."""
        b = built([100.0] * 10, building_s=5.0, built_prints=200)
        p = b.pace()
        assert p["bars_measured"] >= PACE_MIN_BARS
        assert p["ratio"] < PUSH_RATIO, p
        assert p["push"] == "push"

    def test_a_stalling_tape_reads_fade(self):
        b = built([10.0] * 10, building_s=40.0, built_prints=200)
        p = b.pace()
        assert p["ratio"] > FADE_RATIO, p
        assert p["push"] == "fade"

    def test_a_steady_tape_reads_even(self):
        b = built([100.0] * 10, building_s=20.0, built_prints=200)
        p = b.pace()
        assert p["push"] == "even", p

    def test_a_cold_start_says_so_rather_than_guessing(self):
        b = built([100.0] * 3, building_s=1.0, built_prints=200)
        p = b.pace()
        assert p["push"] == ""
        assert p["bars_measured"] < PACE_MIN_BARS
        assert "not enough" in p["say"]

    def test_the_word_agrees_with_the_sentence(self):
        """Two readings of one number that disagree on screen is the
        OFI colour-against-body bug wearing different clothes."""
        for spans, bs, bp in (([100.0] * 10, 5.0, 200),
                              ([10.0] * 10, 40.0, 200),
                              ([100.0] * 10, 20.0, 200)):
            p = built(spans, building_s=bs, built_prints=bp).pace()
            if p["push"] == "push":
                assert "busier" in p["say"] or "twice as busy" in p["say"]
            elif p["push"] == "fade":
                assert "slower" in p["say"] or "quiet" in p["say"]
            else:
                assert "normal" in p["say"]


class TestItReachesThePanels:
    """THE WHOLE POINT OF THE CHANGE. `pace()` was correct and
    unreachable: it reached the chart route and stopped, so the two
    panels that read direction never saw it."""

    def test_the_tick_route_carries_it(self):
        """THE RETURN THAT FIRES ON ALMOST EVERY CALL.

        A plain substring check passed this while the main return
        had no pace at all -- the between-bars return further up
        carries the same text, so one site covered for the other
        and the line the panel actually reads was empty. The block
        is cut at the early return so each site is checked alone.
        """
        i = WEB.index("def _tick_now(")
        blk = WEB[i:WEB.index("def _m5_now(", i)]
        assert "bld.pace()" in blk, "the route still does not read it"
        main = blk[blk.index('"up_prints": cur.up_prints'):]
        assert '"pace": pace' in main, \
            "the forming bar's reading is returned without its pace"

    def test_the_route_carries_it_between_bars_too(self):
        """The early return fires when the last print closed a bar.
        Dropping the pace there makes the line blink out once per
        bar, which reads as a fault."""
        i = WEB.index("def _tick_now(")
        blk = WEB[i:WEB.index("def _m5_now(", i)]
        j = blk.index("if cur is None:")
        assert '"pace": pace' in blk[j:j + 600]

    def test_fusion_passes_it_through(self):
        """This dict is a reshape: a field not named here is
        silently dropped. That is how the smoothing once reached the
        route and died one function later."""
        from liqmap import fusion
        st = fusion.state({"dir": "up", "pace": {"push": "push",
                                                 "ratio": 0.4}},
                          {"price_dir": "up"}, {"dir": "up"})
        assert st["tick"]["pace"]["push"] == "push"
        assert st["tick"]["pace"]["ratio"] == 0.4

    def test_fusion_gives_an_empty_dict_rather_than_none(self):
        from liqmap import fusion
        st = fusion.state({"dir": "up"}, {}, {})
        assert st["tick"]["pace"] == {}

    def test_the_stackwin_state_carries_it_under_its_own_key(self):
        """NOT inside `tick`. That block is the whole rule, and a
        field deciding nothing beside fields deciding everything is
        how this engine has twice traded something nobody meant."""
        from liqmap import stackwin as sw
        st = sw.state({}, "up", {}, 300.0,
                      pace={"push": "fade", "ratio": 2.0})
        assert st["pace"]["push"] == "fade"
        assert st["pace"]["decides"] is False
        assert "pace" not in st["tick"]

    def test_the_route_hands_it_the_tick_readings_pace(self):
        i = WEB.index("fus = _sw.state(")
        assert 'pace=_tk.get("pace")' in WEB[i:i + 400]

    def test_the_fusion_panel_draws_it(self):
        """SCOPED TO `paintFusion`. This test passed for the wrong
        reason: the phrase it looked for lived in the STACK+WINDOW
        painter, not this one -- the fusion panel builds the same
        sentence across two template-literal lines, so the
        contiguous string was never in it. Replacing the other
        panel's line is what exposed it.

        Same class as the route test that passed while the main
        return carried no pace at all: a whole-file substring check
        cannot say WHICH panel it found."""
        i = WEB.index("function paintFusion(")
        blk = WEB[i:WEB.index("function ", i + 10)]
        assert "tape ${esc(pc_.push)}" in blk, \
            "the fusion panel does not show the pace"
        assert "bars_measured" in blk and "not enough yet" in blk, \
            "a cold start goes blank instead of saying why"

    def test_the_stackwin_panel_draws_it(self):
        assert "SW.pace && SW.pace.push" in WEB
        assert "decides nothing" in WEB, \
            "the line is drawn without the label that keeps it out " \
            "of the rule"


class TestItIsNotAllowedToTrade:
    """The reading is informational; the rule is the tick bar and
    its volume. Pinned, because an informational panel reading that
    quietly acquires a vote is exactly the failure these tests
    exist for."""

    PUSH = {"state": "continuation", "heavy": False}

    def test_the_entry_ignores_the_pace_entirely(self):
        from liqmap import stackwin as sw
        fast = sw.state({}, "up", {}, 300.0, intent=self.PUSH,
                        pace={"push": "fade", "ratio": 9.0})
        slow = sw.state({}, "up", {}, 300.0, intent=self.PUSH,
                        pace={"push": "push", "ratio": 0.1})
        assert fast["side"] == slow["side"] == "up"
        assert fast["say"] == slow["say"]

    def test_a_push_cannot_rescue_a_heavy_pullback(self):
        """The tempting bug: a very fast tape waved through the
        trade the volume rule had turned round."""
        from liqmap import stackwin as sw
        st = sw.state({}, "up", {}, 300.0,
                      intent={"state": "pullback", "heavy": True},
                      pace={"push": "push", "ratio": 0.05})
        assert st["side"] == "down"

    def test_entry_side_does_not_take_a_pace_at_all(self):
        """The strongest form of the guarantee: the function that
        decides cannot see the number."""
        import inspect

        from liqmap import stackwin as sw
        assert "pace" not in inspect.signature(sw.entry_side).parameters

    def test_nothing_here_submits_an_order(self):
        import ast
        tree = ast.parse(pathlib.Path(tb.__file__).read_text())
        bad = [n.name for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name in ("submit", "execute", "place", "order")]
        assert not bad, f"tickbar.py grew {bad}"
