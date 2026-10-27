"""OFI candles on the live page: the route, the ladder, the panel.

In order of how badly each would mislead:

  a target switch that shows an empty chart. Every threshold must be
  accumulating all the time, or changing it throws away the session
  and you are staring at nothing during the minutes you wanted to
  look at

  the builders fed from the DECODED float book rather than the raw
  integers. The formula branches on price EQUALITY -- "the bid did
  not move" is a different case from "the bid moved" -- and a float
  that round-tripped through a division takes the wrong branch
  silently

  a chart that spaces bars by timestamp. These bars are not time, and
  drawing them on a clock axis squeezes the busy stretches into a
  sliver -- which is exactly the part worth looking at
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from liqmap.ofibar import CHOP, GREEN, RED, OfiBarBuilder   # noqa: E402

WEB = (ROOT / "liqmap" / "web.py").read_text()
TAPE = (ROOT / "liqmap" / "cme_tape.py").read_text()


class TestEveryTargetAccumulatesAtOnce:
    """So switching is instant and carries its whole history."""

    def _ladder(self):
        return {t: OfiBarBuilder(tick_size=0.25, threshold=t,
                                 max_s=900.0, min_updates=2)
                for t in (125, 250, 500, 1000, 2000)}

    def _feed(self, ladder, n=4000, seed=5):
        import random
        rng = random.Random(seed)
        bid, ts = 20_000, 0.0
        for _ in range(n):
            if rng.random() < 0.08:
                bid += 1 if rng.random() < 0.52 else -1
            ts += 0.01
            for b in ladder.values():
                b.on_book(ts, bid, rng.randint(50, 600),
                          bid + 1, rng.randint(50, 600))
        return ladder

    def test_a_small_target_makes_more_bars_than_a_large_one(self):
        lad = self._feed(self._ladder())
        counts = {t: len(b.bars) for t, b in lad.items()}
        ordered = [counts[t] for t in (125, 250, 500, 1000, 2000)]
        assert ordered == sorted(ordered, reverse=True), counts
        assert counts[125] > counts[2000]

    def test_every_target_has_history_at_the_same_moment(self):
        """The point of running them all: no target is ever cold."""
        lad = self._feed(self._ladder())
        for t, b in lad.items():
            assert len(b.bars) > 0, f"target {t} produced nothing"

    def test_switching_target_does_not_disturb_the_others(self):
        lad = self._feed(self._ladder())
        before = {t: len(b.bars) for t, b in lad.items()}
        _ = lad[500].bars[-20:]            # what a switch does: read one
        after = {t: len(b.bars) for t, b in lad.items()}
        assert before == after

    def test_each_builder_keeps_its_own_accumulator(self):
        lad = self._ladder()
        for b in lad.values():
            b.on_book(0.0, 20_000, 100, 20_001, 100)
        lad[125].on_book(0.1, 20_000, 400, 20_001, 100)
        assert lad[250].accumulated == 0, "builders must not share state"


class TestTheTapeFeedsRawIntegers:

    def test_the_tape_reads_bid_px_not_the_decoded_book(self):
        """A float that round-tripped through a division breaks the
        equality branch the formula depends on."""
        i = TAPE.index("if self.ofibars:")
        j = TAPE.index("if self.cvdcross is not None:", i)
        block = TAPE[i:j]
        assert 'dbn._get(l0, "bid_px"' in block
        assert "int(" in block
        assert "bk.bids" not in block, "that is the decoded float book"

    def test_the_tape_cannot_be_stopped_by_the_display(self):
        i = TAPE.index("if self.ofibars:")
        j = TAPE.index("if self.cvdcross is not None:", i)
        assert "except Exception" in TAPE[i:j]

    def test_the_field_exists_on_the_tape(self):
        from liqmap.cme_tape import Tape
        t = Tape(symbol="ES")
        assert hasattr(t, "ofibars")
        assert t.ofibars is None, "attached on demand, not at construction"


class TestTheRouteAndPanel:

    def test_the_ladder_spans_125_to_2000(self):
        from liqmap.ofibar import LADDER
        assert LADDER == (125, 250, 500, 1000, 2000)

    def test_the_ladder_lives_with_the_builder_not_in_a_closure(self):
        """It was briefly a local inside create_app, where the runtime's
        own footprint() could not see it -- eighteen tests died with a
        NameError raised from a method in a different scope."""
        assert "from .ofibar import LADDER" in WEB
        assert "OFI_TARGETS" not in WEB

    def test_an_unknown_target_snaps_to_the_nearest(self):
        """A stale bookmark asking for 1,700 should get 2,000, not an
        error and not an empty chart."""
        i = WEB.index("def api_ofibars")
        j = WEB.index("def api_dexarb")
        body = WEB[i:j]
        assert "min(builders, key=lambda x: abs(x - t))" in body

    def test_the_route_converts_raw_units_to_dollars_once(self):
        i = WEB.index("def api_ofibars")
        j = WEB.index("def api_dexarb")
        body = WEB[i:j]
        assert "sc = 1e-9" in body
        assert "x.open * sc" in body

    def test_the_forming_bar_is_sent_separately(self):
        """It can still change colour, so it must not arrive looking
        like a closed one."""
        i = WEB.index("def api_ofibars")
        j = WEB.index("def api_dexarb")
        assert '"forming"' in WEB[i:j]
        assert "direction: 'forming'" in WEB

    def test_the_chart_spaces_bars_by_index_not_time(self):
        i = WEB.index("function ofiChart")
        j = WEB.index("let _ofiTarget")
        body = WEB[i:j]
        assert "const step = iw / n" in body
        assert "step * (i + 0.5)" in body
        # No timestamp arithmetic anywhere in the geometry.
        assert "ts_close -" not in body and "ts_open -" not in body

    def test_the_chart_draws_a_body_and_a_wick(self):
        i = WEB.index("function ofiChart")
        j = WEB.index("let _ofiTarget")
        body = WEB[i:j]
        assert "<rect" in body and "<line" in body
        assert "b.high" in body and "b.low" in body

    def test_the_panel_has_a_target_selector(self):
        assert "ofiSetTarget" in WEB
        assert "onchange=\\\"ofiSetTarget(this.value)\\\"" in WEB or \
               "onchange=\"ofiSetTarget(this.value)\"" in WEB

    def test_the_panel_has_its_own_timer(self):
        # The callback is wrapped so the tick knows it came from the
        # timer and can decline to refetch while scrolled back.
        assert "_ofiTimer" in WEB
        assert "setInterval(() => ofiTick(true)" in WEB

    def test_the_panel_element_exists(self):
        assert 'id="ofiBars"' in WEB


class TestItDecidesNothing:

    def test_the_route_places_no_orders(self):
        i = WEB.index("def api_ofibars")
        j = WEB.index("def api_dexarb")
        body = WEB[i:j]
        for w in ("def submit(", "def execute(", "def place(", "def order("):
            assert w not in body, w

    def test_the_panel_does_not_claim_a_direction_continues(self):
        i = WEB.index("function ofiTick")
        j = WEB.index("function mgChart")
        body = WEB[i:j].lower()
        for w in ("will continue", "guaranteed", "buy signal", "sell signal"):
            assert w not in body, w

    def test_the_panel_says_what_an_uneven_split_means(self):
        """The one diagnostic a reader needs: lopsided colour is a bug,
        not a market view."""
        assert "latching" in WEB


class TestTheAggressionLayer:
    """Body from price, outline from flow, and the gap between them."""

    def test_the_route_carries_the_aggression_fields(self):
        i = WEB.index("def api_ofibars")
        j = WEB.index("def api_dexarb")
        body = WEB[i:j]
        for k in ("price_agg_ratio", "price_aggression",
                  "price_agg_burst", "agrees"):
            assert f'"{k}"' in body, k

    def test_the_forming_bar_gets_a_live_ratio(self):
        """Otherwise the bar in progress is drawn in a colour it does
        not have, and flips the moment it closes."""
        i = WEB.index("def api_ofibars")
        j = WEB.index("def api_dexarb")
        assert "live_pai" in WEB[i:j]

    def test_the_whole_candle_is_one_colour(self):
        """It was briefly two -- body from price, outline from flow --
        which is a different chart from the one asked for. A candle is
        one object and carries one reading."""
        i = WEB.index("function ofiChart")
        j = WEB.index("let _ofiTarget")
        body = WEB[i:j]
        assert "bodyColour" in body
        assert "ofiColour" not in body, "the outline colour is gone"
        assert "aggColour" not in body
        # The rect must carry no stroke of its own.
        rect = body[body.index("g += `<rect"):body.index("});", body.index("g += `<rect"))]
        assert "stroke=" not in rect, "the body must not be outlined separately"

    def test_the_wick_takes_the_same_colour_as_the_body(self):
        i = WEB.index("function ofiChart")
        j = WEB.index("let _ofiTarget")
        body = WEB[i:j]
        line = body[body.index("g += `<line"):body.index("g += `<rect")]
        assert 'stroke="${c}"' in line

    def test_weak_intent_is_grey(self):
        i = WEB.index("function ofiChart")
        j = WEB.index("let _ofiTarget")
        body = WEB[i:j]
        assert "BUY_PRICE_AGGRESSIVE" in body and "SELL_PRICE_AGGRESSIVE" in body
        # The fall-through case, with no label test, is the grey one.
        assert "return '#8a929b'" in body

    def test_the_panel_explains_the_three_colours(self):
        assert "GREEN" in WEB and "RED" in WEB and "GREY" in WEB
        assert "two-sided" in WEB

    def test_the_panel_says_the_boundaries_are_still_flow(self):
        """Only the colour changed. A reader needs to know the bars
        still END on order-flow imbalance, not on price."""
        assert "BOUNDARIES still come" in WEB

    def test_the_reasoning_for_the_whole_bar_window_is_recorded(self):
        """The brief asked for a trailing 5ms window. At target 250
        that is 0.04% of a bar, which would colour the candle from
        almost nothing. The arithmetic belongs in the module, not on
        the panel -- but it must not be lost."""
        src = (ROOT / "liqmap" / "ofibar.py").read_text()
        assert "0.04%" in src
        assert "pai_burst" in src

    def test_the_panel_does_not_read_like_a_timeframe_chart(self):
        """These bars are not time. A duration in the readout makes an
        OFI-paced chart look like a one-minute one, which is the single
        thing it is not."""
        i = WEB.index("function ofiChart")
        j = WEB.index("function mgChart")
        panel = WEB[i:j]
        readout = panel[panel.index("return `${dPane}"):]
        for w in ("seconds", "toFixed(1)}s", "minute", "timeframe"):
            assert w not in readout, f"{w!r} in the on-screen readout"
        assert "book updates" in readout


class TestScrollback:
    """Pan and zoom over what the service has recorded.

    The failure worth guarding is a view that moves under you: an
    offset that wraps, a forming bar stapled onto the end of the past,
    or pivots that flicker in and out as you scroll because they were
    computed only over the visible slice.
    """

    def _route(self):
        i = WEB.index("def api_ofibars")
        return WEB[i:WEB.index("def api_dexarb")]

    def test_the_offset_is_clamped_not_wrapped(self):
        """Scrolling past the beginning should stop at the beginning,
        not jump to the other end of the history."""
        r = self._route()
        assert "max(0, min(int(offset), max(0, total - n)))" in r

    def test_the_forming_bar_is_dropped_when_scrolled_back(self):
        """It belongs at the live edge. Appended to a historical window
        it staples the present onto the end of the past."""
        r = self._route()
        assert "if off > 0:" in r and "forming = None" in r

    def test_pivots_are_found_over_a_wider_slice_than_the_window(self):
        """Found only within the window, a swing near either edge
        appears and disappears as you scroll, because its neighbours
        were outside the slice."""
        r = self._route()
        assert "wide_lo = max(0, start - pv)" in r
        assert "wide_hi = min(total, end + cf)" in r
        assert "if start <= abs_i < end:" in r

    def test_the_route_reports_where_the_window_sits(self):
        r = self._route()
        for k in ("at_live", "oldest", "offset", "window", "dropped"):
            assert f'"{k}"' in r, k

    def test_unsettled_is_only_reported_at_the_live_edge(self):
        """Scrolled back there is no unsettled region -- every bar in
        view has bars after it."""
        r = self._route()
        assert "if off == 0 else 0" in r

    def test_the_window_size_is_bounded(self):
        r = self._route()
        assert "max(20, min(int(bars), 600))" in r

    def test_the_pivot_window_is_bounded(self):
        r = self._route()
        assert "max(1, min(int(pivot), 20))" in r


class TestThePanAndZoomControls:

    def test_the_timer_does_not_refetch_while_scrolled_back(self):
        """A refetch while you are reading history re-renders the same
        window for nothing, and any drift in the offset arithmetic
        would walk the view out from under you."""
        assert "if (fromTimer && _ofiOffset > 0) return;" in WEB
        assert "setInterval(() => ofiTick(true), 2000)" in WEB

    def test_changing_target_snaps_back_to_live(self):
        """Targets have wildly different bar counts, so an offset in
        bars means something else on each one."""
        i = WEB.index("function ofiSetTarget")
        j = WEB.index("function ofiSetWindow")
        assert "_ofiOffset = 0;" in WEB[i:j]

    def test_zooming_holds_the_right_edge(self):
        i = WEB.index("function ofiSetWindow")
        j = WEB.index("function ofiPan")
        assert "_ofiOffset + (was - _ofiWindow)" in WEB[i:j]

    def test_pan_cannot_go_negative(self):
        i = WEB.index("function ofiPan")
        j = WEB.index("function ofiSetPivot")
        assert "Math.max(0," in WEB[i:j]

    def test_there_is_a_way_back_to_live(self):
        assert "function ofiLive()" in WEB
        assert "ofiLive()" in WEB and "AT LIVE EDGE" in WEB

    def test_the_buttons_disable_at_the_ends(self):
        assert "${d.oldest ? 'disabled' : ''}" in WEB
        assert "${d.at_live ? 'disabled' : ''}" in WEB


class TestThePivotMarks:

    def _chart(self):
        i = WEB.index("function ofiChart")
        return WEB[i:WEB.index("let _ofiTarget")]

    def test_a_caret_is_drawn_for_each_pivot(self):
        c = self._chart()
        assert "<path" in c and "d.pivots" in c

    def test_only_a_few_pivots_extend_as_levels(self):
        """Every pivot extended turns the chart into a grid and the
        levels stop meaning anything. Which few is now the server's
        decision, because it also carries their history."""
        c = self._chart()
        assert "d.levels" in c
        assert "0.42, '3,3'" in c, "an untested level is drawn dashed"
        # The chart extends what it was given and chooses nothing.
        assert "marks.filter" not in c

    def test_a_level_off_the_top_or_bottom_is_not_drawn(self):
        c = self._chart()
        assert "if (yy < padT || yy > padT + ih) continue;" in c

    def test_the_marks_are_bounds_checked(self):
        c = self._chart()
        assert "if (m.i < 0 || m.i >= all.length) return '';" in c


class TestMarksArriveOnTime:
    """Symmetric confirmation put the carets thirty-six seconds late."""

    def _route(self):
        i = WEB.index("def api_ofibars")
        return WEB[i:WEB.index("def api_dexarb")]

    def test_left_and_right_are_set_separately(self):
        """Left establishes structure and wants to be wide; right is
        pure latency and wants to be narrow. One number cannot do both
        jobs."""
        r = self._route()
        assert "pivot: int = 5, confirm: int = 1" in r
        assert "left=pv, right=cf" in r

    def test_provisional_marks_are_returned(self):
        r = self._route()
        assert "find_provisional" in r
        assert '"provisional": prov' in r

    def test_provisional_marks_only_exist_at_the_live_edge(self):
        """Scrolled back, every bar has bars after it and the settled
        pass has already judged them all."""
        r = self._route()
        i = r.index("find_provisional")
        assert "if off == 0:" in r[max(0, i - 400):i]

    def test_the_unsettled_count_uses_the_confirm_window(self):
        r = self._route()
        assert "_pivot_pending(len(window), cf)" in r

    def test_provisional_carets_are_drawn_differently(self):
        """They can still be taken out, so they must not look settled."""
        i = WEB.index("function ofiChart")
        c = WEB[i:WEB.index("let _ofiTimer")]
        assert "d.provisional" in c
        assert "stroke-dasharray=\"1.5,1.5\"" in c
        assert "caret(m, true)" in c and "caret(m, false)" in c

    def test_the_panel_warns_that_dotted_marks_can_vanish(self):
        assert "not confirmed yet and can vanish" in WEB


class TestSettingsSurviveAReload:

    def test_the_chart_settings_are_saved(self):
        assert "liqmap_ofi" in WEB
        assert "localStorage.setItem('liqmap_ofi'" in WEB

    def test_they_are_restored_on_load(self):
        assert "localStorage.getItem('liqmap_ofi'" in WEB
        i = WEB.index("localStorage.getItem('liqmap_ofi'")
        blk = WEB[i - 200:i + 400]
        for k in ("_s.t", "_s.w", "_s.p", "_s.c", "_s.k"):
            assert k in blk, k

    def test_every_setter_persists(self):
        i = WEB.index("function _ofiSave")
        blk = WEB[i:WEB.index("async function ofiTick")]
        for fn in ("ofiSetTarget", "ofiSetWindow", "ofiSetPivot",
                   "ofiSetConfirm", "ofiSetKeep"):
            k = blk.index("function " + fn)
            assert "_ofiSave()" in blk[k:k + 340], fn

    def test_the_scroll_position_is_NOT_saved(self):
        """Coming back to a page scrolled into history, with nothing
        saying why it is not moving, reads as broken."""
        i = WEB.index("function _ofiSave")
        blk = WEB[i:WEB.index("let _ofiTarget")]
        assert "_ofiOffset" not in blk

    def test_storage_failures_do_not_break_the_panel(self):
        """Private mode, blocked storage and quota all throw."""
        i = WEB.index("function _ofiSave")
        blk = WEB[i:WEB.index("async function ofiTick")]
        assert blk.count("catch") >= 2

class TestTheLevelLayer:
    """The last few swings, and what happened to each one since.

    In order of how badly each would mislead:

      replay leaking the future. Scrolled back, a level must be judged
      only by the bars that had printed at that point -- judged against
      the whole session, every level on a historical screen already
      knows how it ended, and the replay teaches you nothing

      a second copy of the wick/close rule. The chart and the session
      ledger would eventually disagree about what happened at a price,
      and nothing on either would say which one was wrong

      reading the state off the current state alone, which cannot tell
      "never closed through" from "closed through, closed back, and has
      been touched since" -- opposite readings at the same price
    """

    def _route(self):
        i = WEB.index("def api_ofibars")
        return WEB[i:WEB.index("def api_dexarb")]

    def _chart(self):
        i = WEB.index("function ofiChart")
        return WEB[i:WEB.index("let _ofiTimer")]

    def test_the_route_takes_a_level_count(self):
        assert "keep: int = 4" in self._route()

    def test_the_levels_are_returned(self):
        r = self._route()
        assert '"levels": lvls' in r and '"keep_n": keep_n' in r

    def test_it_uses_the_shared_rule_and_does_not_restate_it(self):
        """One implementation, or the chart and the ledger drift."""
        r = self._route()
        assert "_lv.swing_levels(" in r
        for w in ("bar.close >", "closed_through", "wicked"):
            assert w not in r, w

    def test_they_are_judged_only_by_bars_that_had_printed(self):
        r = self._route()
        assert "hist = b.bars[look:end]" in r
        assert "b.bars[look:total]" not in r

    def test_the_lookback_runs_further_back_than_the_window(self):
        """The last four swings are not always on screen."""
        assert "look = max(0, end - max(n, 300) - pv)" in self._route()

    def test_only_settled_pivots_become_levels(self):
        """A provisional one can be taken out, and a level that appears
        and then vanishes is worse than no level."""
        r = self._route()
        blk = r[r.index("lvls: list[dict[str, Any]] = []"):]
        assert "_pvm.find(hh, ll, left=pv, right=cf" in blk
        assert "find_provisional" not in blk

    def test_pivots_are_found_on_the_numbers_they_are_judged_against(self):
        """Raw units on one side and dollars on the other would make a
        level that no bar can ever equal."""
        r = self._route()
        assert "hh = [x.high * sc for x in hist]" in r
        assert "high=x.high * sc" in r

    def test_the_count_is_clamped(self):
        assert "keep_n = max(1, min(int(keep), 8))" in self._route()

    def test_the_event_indices_are_window_relative_like_everything_else(self):
        r = self._route()
        assert '"break_i": (brk_i - start) if brk_i >= 0 else -1' in r
        assert '"back_i": (bak_i - start) if bak_i >= 0 else -1' in r

    def test_the_state_is_not_carried_by_colour_alone(self):
        """Colour says which side the level came from. The line style
        and the label say what happened to it."""
        c = self._chart()
        for s in ("'broken'", "'held'", "'failed'"):
            assert "L.state === " + s in c, s
        assert "0.42, '3,3'" in c, "an untested level is drawn dashed"
        assert "L.touches" in c

    def test_a_broken_level_is_dimmed_only_after_the_break(self):
        """Up to that bar it WAS a level, and erasing that erases the
        part that mattered."""
        import re
        c = self._chart()
        blk = c[c.index("L.state === 'broken'"):][:900]
        ops = [float(m) for m in
               re.findall(r"rail\(.*?,\s*[\d.]+,\s*([\d.]+),\s*'", blk)]
        assert len(ops) == 2, ops
        assert ops[0] > ops[1], "the part after the break must be fainter"

    def test_a_failed_break_is_marked_where_it_failed(self):
        c = self._chart()
        blk = c[c.index("L.state === 'failed'"):][:700]
        assert "L.back_i" in blk and "<circle" in blk

    def test_a_level_formed_off_screen_still_rules_the_screen(self):
        assert "L.i < 0 ? padL" in self._chart()

    def test_the_readout_states_every_count(self):
        c = self._chart()
        for w in ("held", "closed through", "broke and came back",
                  "not reached"):
            assert w in c, w

    def test_the_panel_carries_the_count_control(self):
        assert "ofiSetKeep" in WEB and "levels forward" in WEB

    def test_the_panel_states_the_rule_rather_than_offering_a_choice(self):
        """The module settled it, with a reason. A toggle here would
        quietly let the chart disagree with the ledger."""
        assert "only a\n        CLOSE through is a break" in WEB
        assert "ofiSetBreak" not in WEB


class TestLevelsOnBarsTheBuilderMade:
    """The same composition the route does, on real builder output."""

    def _bars(self, n=20000, seed=17, target=250):
        import random
        from liqmap import levels as L
        rng = random.Random(seed)
        b = OfiBarBuilder(tick_size=0.25, threshold=target,
                          max_s=900.0, min_updates=2)
        bid, ts = 20_000, 0.0
        for _ in range(n):
            if rng.random() < 0.08:
                bid += 1 if rng.random() < 0.5 else -1
            ts += 0.01
            b.on_book(ts, bid, rng.randint(50, 600),
                      bid + 1, rng.randint(50, 600))
        sc = 1e-9
        return [L.Bar(ts=x.ts_close, open=x.open * sc, high=x.high * sc,
                      low=x.low * sc, close=x.close * sc) for x in b.bars]

    def _levels(self, rows, keep=4, left=5, right=1):
        from liqmap import levels as L
        from liqmap import pivots as P
        ps = P.find([r.high for r in rows], [r.low for r in rows],
                    left, right)
        return L.swing_levels(ps, rows, keep=keep)

    def test_it_produces_levels_and_every_one_has_a_status(self):
        from liqmap import levels as L
        rows = self._bars()
        assert len(rows) > 40, "the feed should close plenty of bars"
        got = self._levels(rows)
        assert 1 <= len(got) <= 4
        assert all(L.swing_status(s) in L.SWING_STATUS for s in got)

    def test_the_status_agrees_with_the_counters(self):
        from liqmap import levels as L
        for s in self._levels(self._bars(), keep=8):
            st = L.swing_status(s)
            if st == L.SWING_UNTESTED:
                assert s.level.touches == 0 and s.level.breaks == 0
            if st == L.SWING_HELD:
                assert s.level.touches > 0 and s.level.breaks == 0
            if st == L.SWING_FAILED:
                assert s.level.breaks > 0 and s.broke_at >= 0
            if st == L.SWING_BROKEN:
                assert s.broke_at >= 0

    def test_a_break_cannot_un_happen_when_more_bars_arrive(self):
        """Replay honesty, as the route composes it."""
        from liqmap import levels as L
        from liqmap import pivots as P
        rows = self._bars()
        cut = len(rows) - 40
        part = rows[:cut]
        ps = P.find([r.high for r in part], [r.low for r in part], 5, 1)
        early = {s.index: s for s in L.swing_levels(ps, part, keep=8)}
        later = {s.index: s for s in L.swing_levels(ps, rows, keep=8)}
        for i, e in early.items():
            if e.level.breaks:
                assert later[i].level.breaks >= e.level.breaks, i
            assert later[i].level.touches >= e.level.touches, i

    def test_every_level_the_chart_draws_can_be_described(self):
        from liqmap import levels as L
        for s in self._levels(self._bars(), keep=8):
            line = L.describe(s)
            assert f"{s.price:.2f}" in line
            for w in ("will", "should", "expect", "likely"):
                assert w not in line.lower(), line


class TestTheDeltaPane:
    """Paid-for flow, drawn directly above the candles.

    In order of how badly each would mislead:

      a pane on its own x-axis. It is not a second chart, it is a
      second reading of the SAME bars, and any drift between the two
      grids puts a column over the wrong candle -- which looks fine

      delta wired to the price path or the book. Three columns exist
      because they come apart; sourcing two of them from one place
      destroys exactly what makes them worth having

      an empty bar drawn as a conflict. At the larger targets plenty
      of bars contain no prints at all
    """

    def _chart(self):
        i = WEB.index("function ofiChart")
        return WEB[i:WEB.index("let _ofiTimer")]

    def _route(self):
        i = WEB.index("def api_ofibars")
        return WEB[i:WEB.index("def api_dexarb")]

    def test_the_pane_shares_the_candles_grid(self):
        c = self._chart()
        blk = c[c.index("const dh = 64"):c.index("Pivot marks")]
        assert "padL + step * (i + 0.5)" in blk, "same column centres"
        assert "bw / 2" in blk and "bw.toFixed" in blk, "same column width"
        assert "w - padR" in blk, "same right edge"

    def test_the_pane_is_drawn_above_the_candles(self):
        c = self._chart()
        i = c.index("return `${dPane}")
        assert i > 0
        assert c.index("${dPane}", i) < c.index("<svg width=", i + 10)

    def test_the_route_sends_delta_per_bar(self):
        r = self._route()
        assert '"delta": x.delta' in r
        assert '"trade_count": x.trade_count' in r
        assert '"delta_agrees": x.delta_agrees' in r

    def test_the_forming_bar_carries_live_delta(self):
        """Otherwise the newest column is always a bar behind."""
        r = self._route()
        assert '"delta": b.live_delta' in r

    def test_a_disagreement_is_marked_on_the_zero_line(self):
        """A column can be one pixel wide, and an outline on it is
        invisible exactly when the bar is quiet enough to notice."""
        c = self._chart()
        assert "L.delta_agrees" in c or "b.delta_agrees === false" in c
        blk = c[c.index("delta_agrees === false"):][:420]
        assert "dy0 - 3" in blk and "dy0 + 3" in blk

    def test_a_bar_with_no_delta_draws_no_column(self):
        c = self._chart()
        blk = c[c.index("const dh = 64"):c.index("Pivot marks")]
        assert "if (v) {" in blk

    def test_the_pane_says_when_the_window_has_no_prints(self):
        c = self._chart()
        assert "no prints in this window" in c

    def test_the_readout_separates_this_bar_from_the_window(self):
        c = self._chart()
        assert "this bar" in c and "window" in c

    def test_delta_is_not_sourced_from_the_price_path(self):
        c = self._chart()
        blk = c[c.index("const dh = 64"):c.index("Pivot marks")]
        for w in ("price_agg", "aggression", "bodyColour"):
            assert w not in blk, w


class TestHigherTargetLevels:
    """The structure you cannot see from inside the target you trade."""

    def _route(self):
        i = WEB.index("def api_ofibars")
        return WEB[i:WEB.index("def api_dexarb")]

    def _chart(self):
        i = WEB.index("function ofiChart")
        return WEB[i:WEB.index("let _ofiTimer")]

    def test_the_route_returns_them(self):
        r = self._route()
        assert '"higher_levels": hi' in r

    def test_the_cut_is_by_time_not_by_bar_count(self):
        """Each builder has its own bar count, so the same index is a
        different moment on every target. Cutting by index would put a
        level on the screen before it had formed."""
        r = self._route()
        assert "edge_ts = window[-1].ts_close" in r
        assert "key=lambda x: x.ts_close" in r

    def test_only_two_targets_above_and_two_levels_each(self):
        """Every target at once is the grid this is avoiding."""
        r = self._route()
        assert "[x for x in sorted(builders) if x > t][:2]" in r
        i = r.index("_lv.swing_levels(found2, rows2")
        assert "keep=2" in r[i:i + 80]

    def test_they_use_the_same_rule_as_the_rest(self):
        r = self._route()
        assert "_lv.swing_levels(found2" in r
        assert "_lv.swing_status(sw)" in r

    def test_they_are_drawn_edge_to_edge_with_no_column(self):
        """A price is comparable across targets. A bar index is not."""
        c = self._chart()
        blk = c[c.index("d.higher_levels"):][:900]
        assert "rail(padL, xEdge" in blk
        assert "L.i" not in blk, "a higher-target level has no column here"

    def test_the_label_names_the_target_it_came_from(self):
        c = self._chart()
        blk = c[c.index("d.higher_levels"):][:900]
        assert "@${L.target" in blk

    def test_they_are_fainter_than_this_target_s_own(self):
        import re
        c = self._chart()
        blk = c[c.index("d.higher_levels"):][:900]
        ops = [float(m) for m in
               re.findall(r"rail\(padL, xEdge, yy, col, [\d.]+, ([\d.]+)",
                          blk)]
        assert ops and max(ops) < 0.8, ops

    def test_they_can_be_turned_off(self):
        assert "ofiSetHigher" in WEB
        assert "higher: int = 1" in self._route()

    def test_the_choice_is_remembered(self):
        i = WEB.index("function _ofiSave")
        blk = WEB[i:WEB.index("async function ofiTick")]
        assert "h: _ofiHigher" in blk
        k = blk.index("function ofiSetHigher")
        assert "_ofiSave()" in blk[k:k + 200]

    def test_hidden_really_means_not_computed(self):
        """Four extra pivot scans a poll for lines nobody is looking
        at."""
        r = self._route()
        assert "if want_higher and window:" in r


class TestTheHistorySurvivesARestart:

    def _route(self):
        i = WEB.index("def api_ofibars")
        return WEB[i:WEB.index("def api_dexarb")]

    def test_the_builders_are_filled_from_disk_when_created(self):
        i = WEB.index("from .ofistore import")
        blk = WEB[i:i + 400]
        assert "load_into_builders()" in blk
        assert "seed()" in blk and "start()" in blk

    def test_restoring_happens_only_on_fresh_builders(self):
        """Merging a disk history into a live one needs the two to
        agree about where they overlap."""
        i = WEB.index("from .ofistore import")
        blk = WEB[i - 1800:i]
        assert "if not svc.tape.ofibars:" in blk

    def test_the_route_reports_what_is_stored(self):
        r = self._route()
        assert '"saved": _saved' in r
        assert "_rc.status(t)" in r

    def test_a_database_failure_does_not_stop_the_candles(self):
        """Losing history is bad. Losing the live chart because
        yesterday's could not be read is worse."""
        i = WEB.index("from .ofistore import")
        blk = WEB[i - 300:i + 600]
        assert "except Exception" in blk
        assert "self._ofirec = None" in blk

    def test_their_labels_sit_on_the_left(self):
        """The right margin already carries the price axis and four
        level labels; a fifth there reads as another level of THIS
        chart rather than a borrowed one."""
        c = WEB[WEB.index("function ofiChart"):WEB.index("let _ofiTimer")]
        blk = c[c.index("d.higher_levels"):][:1200]
        assert "padL + 3" in blk and "xEdge + 4" not in blk
