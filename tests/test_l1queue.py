"""The inside level, and the ways a detector for it flatters itself.

In order of how badly each would mislead:

  calling a level "defended" because one lot came back. Every level
  gets a lot put back. If the refill floor does nothing, FAILED fires
  constantly and whatever it is measured against will look significant

  resolving the window early, so a refill that arrives at 1.9ms is
  reported as a permanent deletion -- which inverts the signal, since
  permanent and failed are supposed to mean opposite things

  treating a BETTER bid arriving as the bid being cleared. A bid that
  steps up is buyers improving, not sellers eating

  carrying the old price's peak size across a price step, which makes
  every step look like a drain of the new level
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from liqmap.l1queue import (ASK, BID, FAILED, PERMANENT,  # noqa: E402
                            L1Event, L1QueueWatcher)

MS = 0.001


def w(**kw) -> L1QueueWatcher:
    kw.setdefault("window_ms", 2.0)
    kw.setdefault("refill_min", 50)
    return L1QueueWatcher(**kw)


class TestThePermanentCase:

    def test_an_ask_that_steps_and_stays_is_permanent(self):
        q = w()
        q.on_book(0.0, 5000.00, 400, 5000.25, 300)
        q.on_book(1 * MS, 5000.00, 400, 5000.50, 250)   # the rung is gone
        out = q.on_book(4 * MS, 5000.00, 400, 5000.50, 250)
        assert len(out) == 1
        e = out[0]
        assert e.kind == PERMANENT and e.side == ASK
        assert e.px == pytest.approx(5000.25)
        assert e.step_px == pytest.approx(5000.50)
        assert e.via == "step"
        assert e.peak_size == 300

    def test_and_the_mirror_on_the_bid(self):
        q = w()
        q.on_book(0.0, 5000.00, 400, 5000.25, 300)
        q.on_book(1 * MS, 4999.75, 350, 5000.25, 300)
        out = q.on_book(4 * MS, 4999.75, 350, 5000.25, 300)
        assert len(out) == 1 and out[0].kind == PERMANENT
        assert out[0].side == BID and out[0].px == pytest.approx(5000.00)

    def test_a_price_that_keeps_going_stays_one_event(self):
        """Three rungs cleared in a row is one attack resolving, not
        three overlapping ones on the same anchor."""
        q = w()
        q.on_book(0.0, 5000.00, 400, 5000.25, 300)
        q.on_book(0.5 * MS, 5000.00, 400, 5000.50, 200)
        q.on_book(1.0 * MS, 5000.00, 400, 5000.75, 150)
        out = q.on_book(5 * MS, 5000.00, 400, 5000.75, 150)
        assert len([e for e in out if e.side == ASK]) == 1
        assert out[0].px == pytest.approx(5000.25), "anchored to the first rung"


class TestTheFailedCase:
    """The golden filter. A level emptied and instantly rebuilt."""

    def test_a_rung_rebuilt_inside_the_window_is_failed(self):
        q = w()
        q.on_book(0.0, 5000.00, 500, 5000.25, 300)
        q.on_book(0.3 * MS, 4999.75, 200, 5000.25, 300)   # bid rung gone
        out = q.on_book(0.8 * MS, 5000.00, 600, 5000.25, 300)  # rebuilt
        assert len(out) == 1
        e = out[0]
        assert e.kind == FAILED and e.side == BID
        assert e.px == pytest.approx(5000.00)
        assert e.refill_size == 600
        assert e.defended is True
        assert e.elapsed_ms == pytest.approx(0.5, abs=1e-6)

    def test_a_token_refill_is_not_a_wall(self):
        """ONE LOT COMING BACK IS NOT A DEFENCE. Without this floor,
        FAILED fires on nearly every step and means nothing."""
        q = w(refill_min=50)
        q.on_book(0.0, 5000.00, 500, 5000.25, 300)
        q.on_book(0.3 * MS, 4999.75, 200, 5000.25, 300)
        q.on_book(0.8 * MS, 5000.00, 3, 5000.25, 300)    # 3 lots back
        out = q.on_book(4 * MS, 5000.00, 3, 5000.25, 300)
        assert len(out) == 1 and out[0].kind == PERMANENT, \
            "three contracts is not an institution defending a price"

    def test_a_refill_after_the_window_is_too_late(self):
        q = w(window_ms=2.0)
        q.on_book(0.0, 5000.00, 500, 5000.25, 300)
        q.on_book(0.3 * MS, 4999.75, 200, 5000.25, 300)
        out = q.on_book(3 * MS, 4999.75, 200, 5000.25, 300)
        assert out and out[0].kind == PERMANENT
        later = q.on_book(5 * MS, 5000.00, 900, 5000.25, 300)
        assert not any(e.kind == FAILED for e in later), \
            "the window closed; a refill now is a different event"

    def test_the_window_boundary_resolves_as_permanent_not_failed(self):
        """A refill at 1.9ms must be FAILED and at 2.1ms PERMANENT. Get
        this backwards and the signal inverts."""
        early = w(window_ms=2.0)
        early.on_book(0.0, 5000.00, 500, 5000.25, 300)
        early.on_book(0.1 * MS, 4999.75, 200, 5000.25, 300)
        out = early.on_book(1.9 * MS, 5000.00, 600, 5000.25, 300)
        assert out and out[0].kind == FAILED

        late = w(window_ms=2.0)
        late.on_book(0.0, 5000.00, 500, 5000.25, 300)
        late.on_book(0.1 * MS, 4999.75, 200, 5000.25, 300)
        out = late.on_book(2.2 * MS, 5000.00, 600, 5000.25, 300)
        assert out and out[0].kind == PERMANENT

    def test_the_ask_mirror(self):
        q = w()
        q.on_book(0.0, 5000.00, 400, 5000.25, 500)
        q.on_book(0.3 * MS, 5000.00, 400, 5000.50, 200)
        out = q.on_book(0.9 * MS, 5000.00, 400, 5000.25, 700)
        assert len(out) == 1
        assert out[0].kind == FAILED and out[0].side == ASK
        assert out[0].refill_size == 700


class TestTheDrainPath:
    """CME publishes incrementally. A 0.5ms empty-and-refill may never
    produce a single book state showing the level vacant, so watching
    only for the price step misses the case worth catching."""

    def test_size_gutted_without_the_price_moving(self):
        q = w(drain_frac=0.90)
        q.on_book(0.0, 5000.00, 500, 5000.25, 300)
        q.on_book(0.5 * MS, 5000.00, 20, 5000.25, 300)    # 96% gone
        out = q.on_book(1.0 * MS, 5000.00, 600, 5000.25, 300)
        assert len(out) == 1
        assert out[0].kind == FAILED and out[0].via == "drain"
        assert out[0].peak_size == 500 and out[0].refill_size == 600

    def test_a_shallow_dent_is_not_a_drain(self):
        q = w(drain_frac=0.90)
        q.on_book(0.0, 5000.00, 500, 5000.25, 300)
        q.on_book(0.5 * MS, 5000.00, 300, 5000.25, 300)   # 40% gone
        out = q.on_book(3.0 * MS, 5000.00, 300, 5000.25, 300)
        assert out == [], "routine trading is not an attack on the level"

    def test_a_drain_on_a_thin_level_is_ignored(self):
        """Ten contracts going to one is 90% of nothing."""
        q = w(drain_frac=0.90, refill_min=50)
        q.on_book(0.0, 5000.00, 10, 5000.25, 300)
        q.on_book(0.5 * MS, 5000.00, 1, 5000.25, 300)
        out = q.on_book(3.0 * MS, 5000.00, 1, 5000.25, 300)
        assert out == []

    def test_a_drain_that_is_not_refilled_is_permanent(self):
        q = w(drain_frac=0.90)
        q.on_book(0.0, 5000.00, 500, 5000.25, 300)
        q.on_book(0.5 * MS, 5000.00, 10, 5000.25, 300)
        out = q.on_book(3.0 * MS, 5000.00, 10, 5000.25, 300)
        assert len(out) == 1 and out[0].kind == PERMANENT
        assert out[0].via == "drain"


class TestItDoesNotInvertTheSides:

    def test_a_better_bid_is_not_the_bid_being_cleared(self):
        """A bid stepping UP is buyers improving. Counting it as a
        deletion would report buying pressure as a sell signal."""
        q = w()
        q.on_book(0.0, 5000.00, 400, 5000.50, 300)
        q.on_book(1 * MS, 5000.25, 150, 5000.50, 300)
        out = q.on_book(4 * MS, 5000.25, 150, 5000.50, 300)
        assert out == []

    def test_a_better_ask_is_not_the_ask_being_cleared(self):
        q = w()
        q.on_book(0.0, 5000.00, 400, 5000.50, 300)
        q.on_book(1 * MS, 5000.00, 400, 5000.25, 120)
        out = q.on_book(4 * MS, 5000.00, 400, 5000.25, 120)
        assert out == []

    def test_the_peak_does_not_survive_a_price_step(self):
        """Carrying 500 across a step onto a 20-lot level makes the new
        level look 96% drained the moment it appears."""
        q = w(drain_frac=0.90)
        q.on_book(0.0, 5000.00, 500, 5000.25, 300)
        q.on_book(1 * MS, 5000.00, 500, 5000.50, 20)
        out = q.on_book(1.5 * MS, 5000.00, 500, 5000.50, 20)
        assert not any(e.via == "drain" for e in out)


class TestBookkeeping:

    def test_the_first_book_opens_no_event(self):
        q = w()
        assert q.on_book(0.0, 5000.00, 400, 5000.25, 300) == []

    def test_flush_resolves_what_is_still_pending(self):
        """Without this the last attack of every session is dropped,
        biasing the sample toward whatever resolves quickly."""
        q = w()
        q.on_book(0.0, 5000.00, 400, 5000.25, 300)
        q.on_book(1 * MS, 5000.00, 400, 5000.50, 250)
        assert q.flush(1.5 * MS) == [], "still inside the window"
        assert len(q.flush(10 * MS)) == 1

    def test_zero_prices_are_ignored(self):
        q = w()
        q.on_book(0.0, 0.0, 0, 0.0, 0)
        assert q.on_book(1 * MS, 0.0, 0, 0.0, 0) == []

    def test_both_sides_can_resolve_on_one_update(self):
        q = w()
        q.on_book(0.0, 5000.00, 400, 5000.25, 300)
        q.on_book(0.5 * MS, 4999.75, 100, 5000.50, 100)
        out = q.on_book(4 * MS, 4999.75, 100, 5000.50, 100)
        assert {e.side for e in out} == {BID, ASK}

    def test_events_accumulate(self):
        q = w()
        q.on_book(0.0, 5000.00, 400, 5000.25, 300)
        q.on_book(1 * MS, 5000.00, 400, 5000.50, 250)
        q.on_book(4 * MS, 5000.00, 400, 5000.50, 250)
        assert len(q.events) == 1
        q.reset()
        assert q.events == []

    def test_it_serialises(self):
        import json
        q = w()
        q.on_book(0.0, 5000.00, 400, 5000.25, 300)
        q.on_book(1 * MS, 5000.00, 400, 5000.50, 250)
        out = q.on_book(4 * MS, 5000.00, 400, 5000.50, 250)
        json.dumps(out[0].as_dict())

    def test_degenerate_settings_are_refused(self):
        for kw in ({"tick_size": 0}, {"window_ms": 0},
                   {"refill_min": 0}, {"drain_frac": 0.0},
                   {"drain_frac": 1.5}):
            with pytest.raises(ValueError):
                L1QueueWatcher(**kw)


class TestItDecidesNothing:

    def test_no_order_code(self):
        src = (ROOT / "liqmap" / "l1queue.py").read_text()
        for word in ("def submit(", "def execute(", "def place(",
                     "def order("):
            assert word not in src, word

    def test_it_does_not_claim_to_predict(self):
        """PERMANENT and FAILED are descriptions of what the inside
        level did. Whether either precedes price is unmeasured at the
        time of writing, and the module must not imply otherwise."""
        src = (ROOT / "liqmap" / "l1queue.py").read_text().lower()
        assert "has not been" in src
        for word in ("institution", "smart money", "whale", "guarantee",
                     "signal", "edge"):
            assert word not in src, word
