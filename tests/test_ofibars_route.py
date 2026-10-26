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
        assert "_ofiTimer" in WEB and "setInterval(ofiTick" in WEB

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
