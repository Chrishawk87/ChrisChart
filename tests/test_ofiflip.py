"""The imbalance flip, and the ways it would fool itself.

In order of how badly each would mislead:

  no cooldown. The jump condition stays true for as long as the window
  holds the move, so one three-tick drop fires on every book update --
  several hundred times on ES. A sample inflated a thousandfold makes
  anything significant, and the inflation is invisible in the output

  no confirmed states. Price mean-reverts after a sharp move whether
  or not the flow agreed, so a divergent dip bouncing proves nothing
  without a confirmed dip beside it to compare against

  an inverted sign, so a hunted dip is labelled ordinary selling. That
  makes every reading exactly backwards

  scoring +1 contract of imbalance as a side having conviction
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from liqmap.ofi import Touch                                    # noqa: E402
from liqmap.ofiflip import (DIP_CONFIRMED, DIP_DIVERGENT, DIRECTION,  # noqa: E402
                            LONG, POP_CONFIRMED, POP_DIVERGENT, SHORT,
                            STATES, Flip, FlipWatcher)

TICK = 250_000_000
P0 = 20_000 * TICK


def w(**kw):
    kw.setdefault("tick_raw", TICK)
    kw.setdefault("jump_ticks", 3.0)
    kw.setdefault("window_s", 1.0)
    kw.setdefault("ofi_min", 100)
    kw.setdefault("cooldown_s", 2.0)
    return FlipWatcher(**kw)


def book(px_ticks, bid_sz, ask_sz):
    px = P0 + px_ticks * TICK
    return Touch(bid_px=px, bid_sz=bid_sz, ask_px=px + TICK, ask_sz=ask_sz)


def walk(q, ts0, down, steps=3, flat_per_step=12, add=0, resting=100,
         step_dt=0.02, settle_updates=40):
    """Walk price `steps` ticks, with `flat_per_step` book updates at
    each level adding `add` lots to the near side.

    A PRICE STEP IS NOT A NEUTRAL EVENT IN THIS FORMULA. A bid ticking
    down contributes -prev_bid_size and the ask ticking down another
    -cur_ask_size, so three ticks on a 100-lot book is -600 before any
    trading at all. Divergence therefore cannot be produced by walking
    price down and making the bid bigger -- it needs ACCUMULATION AT
    FLAT PRICE large enough to outweigh every step term, which is what
    this fixture builds and what the real state actually is.
    """
    ts = ts0
    px = 0
    sz = resting
    last = None
    for _ in range(steps):
        # STEP FIRST, THEN ACCUMULATE. Stepping THROUGH a pile you just
        # built contributes that whole pile in the direction of the
        # step -- which is right, since the pile was lifted or hit.
        # Divergence is size arriving AFTER the tick, at the new level,
        # and sequencing the fixture the other way round measures the
        # opposite of the state it claims to build.
        ts += step_dt
        px += -1 if down else 1
        sz = resting
        got = q.on_touch(ts, book(px, sz if down else resting,
                                  resting if down else sz))
        if got is not None:
            last = got
        for _ in range(flat_per_step):
            ts += step_dt
            sz += add
            got = q.on_touch(ts, book(px, sz if down else resting,
                                      resting if down else sz))
            # Keep the EVENT, not the last return value. Once one
            # fires the cooldown makes every later call return None,
            # so taking the last result reports "nothing happened"
            # about a walk in which something did.
            if got is not None:
                last = got
    # Keep streaming through the settle window. The watcher arms on the
    # jump and reads the state afterwards, so a fixture that stops at
    # the jump never produces an event at all -- which is a property of
    # the fixture, not of the market.
    for _ in range(settle_updates):
        ts += step_dt
        sz += add
        got = q.on_touch(ts, book(px, sz if down else resting,
                                  resting if down else sz))
        if got is not None:
            last = got
    return last, ts


class TestItLabelsTheFourStates:

    def test_a_dip_with_buying_underneath_is_divergent(self):
        """The stop-hunt claim: price down, flow up. It takes real
        accumulation -- see `walk`."""
        q = w()
        q.on_touch(0.0, book(0, 100, 100))
        f, _ = walk(q, 0.0, down=True, add=60)
        assert f is not None
        assert f.kind == DIP_DIVERGENT
        assert f.ticks < 0 and f.ofi > 0

    def test_a_dip_with_selling_underneath_is_confirmed(self):
        q = w()
        q.on_touch(0.0, book(0, 100, 100))
        f, _ = walk(q, 0.0, down=True, add=0, resting=300)
        assert f is not None and f.kind == DIP_CONFIRMED
        assert f.ticks < 0 and f.ofi < 0

    def test_a_pop_with_selling_underneath_is_divergent(self):
        q = w()
        q.on_touch(0.0, book(0, 100, 100))
        f, _ = walk(q, 0.0, down=False, add=60)
        assert f is not None and f.kind == POP_DIVERGENT
        assert f.ticks > 0 and f.ofi < 0

    def test_a_pop_with_buying_underneath_is_confirmed(self):
        q = w()
        q.on_touch(0.0, book(0, 100, 100))
        f, _ = walk(q, 0.0, down=False, add=0, resting=300)
        assert f is not None and f.kind == POP_CONFIRMED

    def test_divergence_requires_outweighing_every_step_term(self):
        """The quantity that makes this state rare, pinned. Adding a
        little size while price falls is still a confirmed dip."""
        q = w()
        q.on_touch(0.0, book(0, 100, 100))
        f, _ = walk(q, 0.0, down=True, add=2)
        assert f is not None and f.kind == DIP_CONFIRMED, \
            "a trickle of bids does not outweigh three downticks"

    def test_the_directions_fade_divergence_and_follow_confirmation(self):
        assert DIRECTION[DIP_DIVERGENT] == LONG
        assert DIRECTION[POP_DIVERGENT] == SHORT
        assert DIRECTION[DIP_CONFIRMED] == SHORT
        assert DIRECTION[POP_CONFIRMED] == LONG


class TestTheCooldown:
    """Without it, one jump becomes hundreds of 'independent' events."""

    def _drop(self, q, updates, step=0.01):
        q.on_touch(0.0, book(0, 100, 100))
        ts = 0.0
        for i in range(updates):
            ts += step
            px = max(-4, -(i // 3))
            q.on_touch(ts, book(px, 100 + 40 * i, 100))
        # past the settle window, so armed jumps actually resolve
        for i in range(updates, updates + 120):
            ts += step
            q.on_touch(ts, book(-4, 100 + 40 * i, 100))
        return q.events

    def test_one_jump_is_one_event(self):
        q = w(cooldown_s=2.0)
        ev = self._drop(q, 60)          # 0.6s of updates after a 4t drop
        assert len(ev) == 1, f"one drop fired {len(ev)} times"

    def test_with_neither_guard_the_sample_explodes(self):
        """Pinning the failure so nobody removes these thinking they
        are a nicety. The settle window turns out to rate-limit on its
        own, so BOTH have to go to show the damage -- which is worth
        knowing: the spacing does not depend on the cooldown alone."""
        q = w(cooldown_s=0.0, settle_s=0.0)
        ev = self._drop(q, 60)
        assert len(ev) > 10, \
            "with no settle and no cooldown one move should fire repeatedly"

    def test_the_settle_window_alone_still_spaces_events_out(self):
        q = w(cooldown_s=0.0, settle_s=0.5)
        ev = self._drop(q, 60)
        assert len(ev) < 6, f"settle alone should keep this small, got {len(ev)}"

    def test_a_later_jump_still_fires(self):
        q = w(cooldown_s=0.5)
        q.on_touch(0.0, book(0, 100, 100))
        _, ts = walk(q, 0.0, down=True, add=60)
        assert len(q.events) == 1
        # Well past the cooldown, a fresh move must be heard.
        base = ts + 5.0
        q.on_touch(base, book(0, 100, 100))
        walk(q, base, down=True, add=60)
        assert len(q.events) == 2


class TestTheFloors:

    def test_a_small_move_is_not_a_jump(self):
        q = w(jump_ticks=3.0)
        q.on_touch(0.0, book(0, 100, 100))
        for i in range(1, 4):
            q.on_touch(0.1 * i, book(-1, 100 + 500 * i, 100))
        assert q.events == []

    def test_a_token_imbalance_is_not_conviction(self):
        """Without this floor, a handful of contracts reads as
        institutions. Sizes here are tiny so even the step terms stay
        under the line."""
        q = w(ofi_min=500)
        q.on_touch(0.0, book(0, 4, 4))
        f = None
        for i in range(1, 6):
            f = q.on_touch(0.1 * i, book(-i, 4, 4))
        assert f is None and q.events == []

    def test_the_window_forgets(self):
        """OFI from before the window must not count toward the move
        inside it, or an old burst labels a new jump."""
        q = w(window_s=0.5, cooldown_s=0.0)
        q.on_touch(0.0, book(0, 100, 100))
        q.on_touch(0.05, book(0, 5_000, 100))       # big buy, early
        for i in range(1, 12):
            q.on_touch(0.5 + 0.1 * i, book(-i, 100, 100))
        assert all(e.ofi <= 0 for e in q.events), \
            "the early buying leaked past the window"

    def test_a_crossed_book_is_skipped(self):
        q = w()
        q.on_touch(0.0, book(0, 100, 100))
        bad = Touch(bid_px=P0 + 5 * TICK, bid_sz=900,
                    ask_px=P0, ask_sz=100)
        assert q.on_touch(0.1, bad) is None


class TestBookkeeping:

    def test_the_first_touch_opens_nothing(self):
        q = w()
        assert q.on_touch(0.0, book(0, 100, 100)) is None

    def test_reset_clears_it(self):
        q = w()
        q.on_touch(0.0, book(0, 100, 100))
        walk(q, 0.0, down=True, add=60)
        assert q.events
        q.reset()
        assert q.events == [] and q._ofi == 0

    def test_it_serialises(self):
        import json
        q = w()
        q.on_touch(0.0, book(0, 100, 100))
        walk(q, 0.0, down=True, add=60)
        json.dumps(q.events[0].as_dict())

    def test_degenerate_settings_are_refused(self):
        for kw in ({"tick_raw": 0}, {"jump_ticks": 0}, {"window_s": 0},
                   {"ofi_min": 0}, {"cooldown_s": -1}, {"settle_s": -1}):
            with pytest.raises(ValueError):
                w(**kw)


class TestItDecidesNothing:

    def test_no_order_code(self):
        src = (ROOT / "liqmap" / "ofiflip.py").read_text()
        for word in ("def submit(", "def execute(", "def place(",
                     "def order("):
            assert word not in src, word

    def test_the_confirmed_states_exist_as_controls(self):
        """A module that only labelled divergence could not tell a
        finding from the bounce that happens anyway."""
        assert len(STATES) == 4
        assert DIP_CONFIRMED in STATES and POP_CONFIRMED in STATES

    def test_it_reuses_the_formula(self):
        src = (ROOT / "liqmap" / "ofiflip.py").read_text()
        assert "from .ofi import" in src and "level_ofi" in src
