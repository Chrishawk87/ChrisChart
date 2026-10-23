"""The leg reader, and the ways it could flatter itself.

In order of how badly each would mislead:

  calling absorption on a level that simply has not been attacked
  yet -- the reading would fire on every pause in every move, which
  is most of the chart, and be right by accident often enough to
  look like it works

  counting the FILLING side's volume as the driver's effort, which
  makes any two-sided level read as absorbed

  a fixed contract threshold instead of the leg's own going rate, so
  the same reading is too twitchy overnight and too deaf at the open

  a scalp direction that does not mirror -- the same setup on the
  other side must give the opposite answer, or it is reading the
  sample's drift rather than the structure
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from liqmap.push import (ABSORBED, BALANCED, BUYERS, CONTINUATION,  # noqa: E402
                         DOWN, FORMING, SELLERS, UP, Push, PushWatcher)


def drive(w, start, ticks, size=10, side="buy", ts=0.0, step=0.1,
          tick=0.25):
    """Walk price `ticks` in the aggressor's direction, one tick at a
    time, paying `size` at each. Returns the final price and ts."""
    px = start
    up = side == "buy"
    for _ in range(abs(ticks)):
        px = px + (tick if up else -tick)
        ts += step
        w.on_trade(ts, px, size, side)
    return px, ts


def hammer(w, px, n, size=10, side="buy", ts=0.0, step=0.1):
    """Pay repeatedly at ONE price without moving it."""
    for _ in range(n):
        ts += step
        w.on_trade(ts, px, size, side)
    return ts


class TestItReadsTheLeg:

    def test_nothing_until_price_has_actually_gone_somewhere(self):
        w = PushWatcher(window_ticks=4)
        w.on_trade(1.0, 5000.00, 50, "buy")
        r = w.reading()
        assert r.direction == "" and r.travel_ticks == 0
        assert r.verdict == FORMING and r.scalp == ""

    def test_a_short_move_is_still_forming(self):
        w = PushWatcher(window_ticks=4)
        w.on_trade(1.0, 5000.00, 10, "buy")
        drive(w, 5000.00, 3, side="buy", ts=1.0)
        r = w.reading()
        assert r.direction == UP and r.travel_ticks == 3
        assert r.verdict == FORMING, "three ticks is not the four asked for"
        assert r.scalp == ""

    def test_four_ticks_driven_and_held_is_a_continuation(self):
        w = PushWatcher(window_ticks=4)
        w.on_trade(1.0, 5000.00, 10, "buy")
        drive(w, 5000.00, 4, side="buy", ts=1.0)
        r = w.reading()
        assert r.direction == UP and r.travel_ticks == 4
        assert r.eater == BUYERS and r.agrees is True
        assert r.verdict == CONTINUATION
        assert r.scalp == "long"
        assert r.invalidated_at == pytest.approx(5000.00), \
            "a continuation is wrong once the leg is given back"

    def test_and_the_mirror(self):
        w = PushWatcher(window_ticks=4)
        w.on_trade(1.0, 5000.00, 10, "sell")
        drive(w, 5000.00, 4, side="sell", ts=1.0)
        r = w.reading()
        assert r.direction == DOWN and r.eater == SELLERS
        assert r.verdict == CONTINUATION and r.scalp == "short"


class TestAbsorptionIsPayingAndNotGettingIt:
    """The distinction the whole module turns on. Price stops all the
    time; that is not absorption. Absorption is spending the going
    rate for another tick and not being given it."""

    def _four_up(self, w, size=10):
        w.on_trade(1.0, 5000.00, size, "buy")
        px, ts = drive(w, 5000.00, 4, size=size, side="buy", ts=1.0)
        return px, ts

    def test_arriving_and_resting_is_not_absorption(self):
        """Nobody has attacked the level yet. Calling it absorbed
        here would fire on every pause in every move."""
        w = PushWatcher(window_ticks=4)
        px, ts = self._four_up(w, size=10)
        r = w.reading()
        assert r.since_extreme == 0
        assert r.verdict == CONTINUATION

    def test_paying_the_going_rate_at_the_high_and_not_moving_it(self):
        """The reported case: CVD runs hard while price sits."""
        w = PushWatcher(window_ticks=4)
        px, ts = self._four_up(w, size=10)
        r = w.reading()
        assert r.per_tick == pytest.approx(12.5), \
            "50 contracts bought 4 ticks, opening print included"
        hammer(w, px, n=2, size=10, side="buy", ts=ts)
        r = w.reading()
        assert r.since_extreme == 20
        assert r.verdict == ABSORBED
        assert r.scalp == "short", "the driver is the one being filled"
        assert r.invalidated_at == pytest.approx(px), \
            "wrong the moment the high goes"

    def test_half_the_going_rate_is_not_yet_absorption(self):
        w = PushWatcher(window_ticks=4)
        px, ts = self._four_up(w, size=20)
        hammer(w, px, n=1, size=9, side="buy", ts=ts)
        r = w.reading()
        assert 0 < r.since_extreme < r.per_tick
        assert r.verdict == CONTINUATION

    def test_the_mirror_absorption(self):
        w = PushWatcher(window_ticks=4)
        w.on_trade(1.0, 5000.00, 10, "sell")
        px, ts = drive(w, 5000.00, 4, size=10, side="sell", ts=1.0)
        hammer(w, px, n=2, size=12, side="sell", ts=ts)
        r = w.reading()
        assert r.direction == DOWN and r.verdict == ABSORBED
        assert r.scalp == "long", "sellers at the low are the ones filled"

    def test_the_filling_side_is_not_counted_as_effort(self):
        """If the passive side's prints counted, every two-sided
        level would read absorbed -- which is all of them."""
        w = PushWatcher(window_ticks=4)
        px, ts = self._four_up(w, size=10)
        hammer(w, px, n=5, size=40, side="sell", ts=ts)
        r = w.reading()
        assert r.since_extreme == 0, "sellers there are the fill"
        assert r.verdict == CONTINUATION

    def test_the_threshold_scales_with_what_the_leg_cost(self):
        """A fixed contract count would be too twitchy overnight and
        deaf at the open. Same shape, ten times the size, same call."""
        quiet = PushWatcher(window_ticks=4)
        busy = PushWatcher(window_ticks=4)
        for w, sz in ((quiet, 5), (busy, 50)):
            w.on_trade(1.0, 5000.00, sz, "buy")
            px, ts = drive(w, 5000.00, 4, size=sz, side="buy", ts=1.0)
            hammer(w, px, n=2, size=sz, side="buy", ts=ts)
        assert quiet.reading().verdict == ABSORBED
        assert busy.reading().verdict == ABSORBED
        assert quiet.reading().per_tick != busy.reading().per_tick

    def test_absorb_mult_makes_it_slower_to_call(self):
        w = PushWatcher(window_ticks=4, absorb_mult=3.0)
        w.on_trade(1.0, 5000.00, 10, "buy")
        px, ts = drive(w, 5000.00, 4, size=10, side="buy", ts=1.0)
        ts = hammer(w, px, n=2, size=10, side="buy", ts=ts)
        assert w.reading().verdict == CONTINUATION, "one rate is not three"
        hammer(w, px, n=3, size=10, side="buy", ts=ts)
        assert w.reading().verdict == ABSORBED

    def test_giving_the_whole_leg_back_is_absorption_too(self):
        """It got there and did not keep it. Reached on give-back
        rather than on effort, but the same verdict."""
        w = PushWatcher(window_ticks=8, pivot_ticks=99)
        w.on_trade(1.0, 5000.00, 10, "buy")
        px, ts = drive(w, 5000.00, 8, size=10, side="buy", ts=1.0)
        drive(w, px, 8, size=10, side="sell", ts=ts)
        r = w.reading()
        assert r.travel_ticks == 8 and r.give_back_ticks == 8
        assert r.verdict == ABSORBED and r.scalp == "short"


class TestTheLegTurnsOver:

    def test_coming_far_enough_back_starts_a_new_leg(self):
        w = PushWatcher(window_ticks=4, pivot_ticks=2)
        w.on_trade(1.0, 5000.00, 10, "buy")
        px, ts = drive(w, 5000.00, 6, size=10, side="buy", ts=1.0)
        assert w.reading().direction == UP
        px, ts = drive(w, px, 2, size=10, side="sell", ts=ts)
        r = w.reading()
        assert r.direction == DOWN, "the leg in front of you is down now"
        assert r.pivot_px == pytest.approx(5001.50), "the old high"
        assert r.travel_ticks == 2

    def test_a_small_wobble_does_not_restart_it(self):
        w = PushWatcher(window_ticks=4, pivot_ticks=3)
        w.on_trade(1.0, 5000.00, 10, "buy")
        px, ts = drive(w, 5000.00, 6, size=10, side="buy", ts=1.0)
        drive(w, px, 2, size=10, side="sell", ts=ts)
        r = w.reading()
        assert r.direction == UP, "two ticks is inside the tolerance"
        assert r.travel_ticks == 6 and r.give_back_ticks == 2

    def test_the_new_leg_measures_from_the_old_extreme(self):
        w = PushWatcher(window_ticks=4, pivot_ticks=2)
        w.on_trade(1.0, 5000.00, 10, "buy")
        px, ts = drive(w, 5000.00, 10, size=10, side="buy", ts=1.0)
        high = px
        drive(w, px, 6, size=10, side="sell", ts=ts)
        r = w.reading()
        assert r.direction == DOWN
        assert r.pivot_px == pytest.approx(high)
        assert r.eaten_buy == 0, "the new leg carries none of the old one"


class TestItMirrors:
    """The same structure on the other side must give the opposite
    answer. A reading that does not mirror is describing the sample's
    drift rather than the auction."""

    def _run(self, side):
        w = PushWatcher(window_ticks=4)
        start = 5000.00
        w.on_trade(1.0, start, 10, side)
        px, ts = drive(w, start, 4, size=10, side=side, ts=1.0)
        cont = w.reading()
        hammer(w, px, n=2, size=10, side=side, ts=ts)
        return cont, w.reading()

    def test_continuation_and_absorption_both_mirror(self):
        ub, ab = self._run("buy")
        us, as_ = self._run("sell")
        assert ub.verdict == us.verdict == CONTINUATION
        assert ub.scalp == "long" and us.scalp == "short"
        assert ab.verdict == as_.verdict == ABSORBED
        assert ab.scalp == "short" and as_.scalp == "long"
        assert ub.travel_ticks == us.travel_ticks
        assert ab.since_extreme == as_.since_extreme


class TestDriftIsNotDrive:

    def test_price_moving_without_the_matching_aggressor_is_flagged(self):
        """A tick that moves while the other side is doing the
        trading is thin, not driven. The verdict can still be a
        continuation; `agrees` is what says to trust it less."""
        w = PushWatcher(window_ticks=4)
        w.on_trade(1.0, 5000.00, 10, "sell")
        drive(w, 5000.00, 4, size=10, side="sell", ts=1.0)
        # Price walked UP on seller-aggressed prints, which is odd.
        w2 = PushWatcher(window_ticks=4)
        px, ts = 5000.00, 1.0
        for _ in range(4):
            px += 0.25
            ts += 0.1
            w2.on_trade(ts, px, 10, "sell")
        r = w2.reading()
        assert r.direction == UP and r.eater == SELLERS
        assert r.agrees is False


class TestItDecidesNothing:

    def test_no_order_code(self):
        src = (ROOT / "liqmap" / "push.py").read_text()
        for w in ("def submit(", "def execute(", "def place(",
                  "def order("):
            assert w not in src, w

    def test_it_says_the_reading_is_unmeasured(self):
        """Eleven sections have come back null on this data. This
        reading is NOT one of them -- it is a leg, not a bar state --
        but nothing in the file may imply it has been validated when
        it has not."""
        src = (ROOT / "liqmap" / "push.py").read_text().lower()
        assert "not a forecast" in src
        assert "has not been measured" in src
        for word in ("guarantee", "will reverse", "never fails",
                     "high probability", "sure thing"):
            assert word not in src, word

    def test_degenerate_settings_are_refused(self):
        for kw in ({"tick_size": 0}, {"window_ticks": 0},
                   {"pivot_ticks": 0}, {"absorb_mult": 0}):
            with pytest.raises(ValueError):
                PushWatcher(**kw)

    def test_it_serialises(self):
        import json
        w = PushWatcher()
        w.on_trade(1.0, 5000.00, 10, "buy")
        drive(w, 5000.00, 5, side="buy", ts=1.0)
        d = w.reading().as_dict()
        assert d["verdict"] in (FORMING, CONTINUATION, ABSORBED)
        json.dumps(d)

    def test_reset_clears_it(self):
        w = PushWatcher()
        w.on_trade(1.0, 5000.00, 10, "buy")
        drive(w, 5000.00, 6, side="buy", ts=1.0)
        w.reset()
        r = w.reading()
        assert r.direction == "" and r.travel_ticks == 0
        assert r.eaten_buy == 0 and r.verdict == FORMING
