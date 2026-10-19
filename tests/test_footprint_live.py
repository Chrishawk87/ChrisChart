"""The live footprint bar.

THE CENTRAL PROPERTY, and the reason this file exists: the
incremental tracker must agree with the batch pass from phase 1 at
EVERY tick, not just at the close. They are two implementations of
one definition, and if they ever disagree the live panel and the
settled bar tell different stories about the same market -- with
nothing raising.

Everything else here is secondary to that equivalence test.
"""

from __future__ import annotations

import random
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from liqmap.footprint import (                             # noqa: E402
    BUY_IMBALANCE, SELL_IMBALANCE, AnyRule, DeltaRule, ImbalanceConfig,
    Tick, TimeRule, VolumeRule)
from liqmap.footprint_live import (                        # noqa: E402
    CONFIRMED, FADED, UNCONFIRMED, FootprintStream, Pulse, Window)

T0 = 1_700_000_000.0


def stream(rule=None, **cfg):
    return FootprintStream(
        tick_size=0.25, rule=rule or VolumeRule(100_000),
        imbalance=ImbalanceConfig(**cfg) if cfg else ImbalanceConfig())


def live_keys(s):
    return {(p, side) for (p, side) in s.tracker.flags}


def batch_keys(s):
    c = s._engine.current
    if c is None:
        return set()
    return {(f.price_ticks, f.side) for f in c.imbalances(s.cfg)}


# ---------------------------------------------- the equivalence test

class TestTheLiveSetMatchesTheBatchPass:
    """Two implementations of one definition. They must not drift."""

    def _walk(self, s, n=600, seed=1, tick=0.25):
        rng = random.Random(seed)
        px = 5800.0
        for i in range(n):
            px += rng.choice((-tick, 0.0, 0.0, tick))
            s.on_tick(Tick(ts=T0 + i, price=round(px, 4),
                           size=rng.randint(1, 40),
                           aggressor=rng.choice(("buy", "sell"))))
            yield i

    def test_they_agree_after_every_single_tick(self):
        s = stream()
        for i in self._walk(s):
            assert live_keys(s) == batch_keys(s), f"diverged at tick {i}"

    def test_they_agree_with_a_volume_floor(self):
        s = stream(min_volume=25)
        for i in self._walk(s, seed=2):
            assert live_keys(s) == batch_keys(s), f"diverged at tick {i}"

    def test_they_agree_when_empty_diagonals_are_refused(self):
        s = stream(flag_empty_diagonal=False)
        for i in self._walk(s, seed=3):
            assert live_keys(s) == batch_keys(s), f"diverged at tick {i}"

    def test_they_agree_under_the_zero_edge_policy(self):
        s = stream(edge_policy="zero")
        for i in self._walk(s, seed=4):
            assert live_keys(s) == batch_keys(s), f"diverged at tick {i}"

    def test_they_agree_at_a_different_multiplier(self):
        s = stream(multiplier=2.5)
        for i in self._walk(s, seed=5):
            assert live_keys(s) == batch_keys(s), f"diverged at tick {i}"

    def test_the_measured_values_agree_too_not_just_the_keys(self):
        """Equal flag SETS with different volumes behind them would
        still be a divergence, and an easy one to miss."""
        s = stream()
        for _ in self._walk(s, n=400, seed=6):
            live = {(f.price_ticks, f.side): (f.volume, f.against)
                    for f in s.tracker.flags.values()}
            batch = {(f.price_ticks, f.side): (f.volume, f.against)
                     for f in s._engine.current.imbalances(s.cfg)}
            assert live == batch


class TestPriceGapsAndNewExtremes:
    """The edge policy is what makes incremental updating hard. A new
    low turns the OLD low from an un-judgeable edge into an interior
    level -- and a gap puts it outside the four tests a print would
    normally trigger."""

    def test_a_gap_to_a_new_low_revisits_the_old_low(self):
        s = stream()
        s.on_tick(Tick(ts=T0, price=5800.00, size=50, aggressor="buy"))
        assert live_keys(s) == batch_keys(s)
        # four ticks down in one print -- the old low is now interior
        s.on_tick(Tick(ts=T0 + 1, price=5799.00, size=10,
                       aggressor="sell"))
        assert live_keys(s) == batch_keys(s), "old low kept an edge verdict"

    def test_a_gap_to_a_new_high_revisits_the_old_high(self):
        s = stream()
        s.on_tick(Tick(ts=T0, price=5800.00, size=50, aggressor="sell"))
        s.on_tick(Tick(ts=T0 + 1, price=5801.00, size=10,
                       aggressor="buy"))
        assert live_keys(s) == batch_keys(s), "old high kept an edge verdict"

    def test_repeated_gapping_stays_in_agreement(self):
        s = stream()
        rng = random.Random(9)
        px = 5800.0
        for i in range(300):
            px += rng.choice((-1.0, -0.5, 0.25, 0.75, 1.5))
            s.on_tick(Tick(ts=T0 + i, price=round(px, 2),
                           size=rng.randint(1, 30),
                           aggressor=rng.choice(("buy", "sell"))))
            assert live_keys(s) == batch_keys(s), f"diverged at {i}"


# ------------------------------------------------ unconfirmed/confirmed

class TestTheStateTransition:

    def test_a_live_flag_starts_unconfirmed(self):
        s = stream()
        s.on_tick(Tick(ts=T0, price=5800.00, size=10, aggressor="sell"))
        s.on_tick(Tick(ts=T0, price=5800.25, size=100, aggressor="buy"))
        s.on_tick(Tick(ts=T0, price=5800.50, size=5, aggressor="sell"))
        assert s.tracker.flags
        assert all(f.state == UNCONFIRMED
                   for f in s.tracker.flags.values())

    def test_closing_the_bar_confirms_them(self):
        s = stream(VolumeRule(115))
        s.on_tick(Tick(ts=T0, price=5800.00, size=10, aggressor="sell"))
        s.on_tick(Tick(ts=T0, price=5800.25, size=100, aggressor="buy"))
        closed = s.on_tick(Tick(ts=T0, price=5800.50, size=5,
                                aggressor="sell"))
        assert closed, "the bar should have closed"
        got = s.confirmed_flags(-1)
        assert got and all(f.state == CONFIRMED for f in got)

    def test_the_confirmed_set_belongs_to_the_bar_that_closed(self):
        s = stream(VolumeRule(115))
        s.on_tick(Tick(ts=T0, price=5800.00, size=10, aggressor="sell"))
        s.on_tick(Tick(ts=T0, price=5800.25, size=100, aggressor="buy"))
        s.on_tick(Tick(ts=T0, price=5800.50, size=5, aggressor="sell"))
        assert len(s.history) == 1
        assert s.confirmed_flags(0) == s.confirmed_flags(-1)

    def test_the_live_tracker_is_empty_after_a_close(self):
        """Carrying a flag across the boundary would attribute one
        bar's imbalance to the next."""
        s = stream(VolumeRule(115))
        s.on_tick(Tick(ts=T0, price=5800.00, size=10, aggressor="sell"))
        s.on_tick(Tick(ts=T0, price=5800.25, size=100, aggressor="buy"))
        s.on_tick(Tick(ts=T0, price=5800.50, size=5, aggressor="sell"))
        assert not s.tracker.flags

    def test_a_confirmed_flag_is_not_mutated_by_later_ticks(self):
        s = stream(VolumeRule(115))
        s.on_tick(Tick(ts=T0, price=5800.00, size=10, aggressor="sell"))
        s.on_tick(Tick(ts=T0, price=5800.25, size=100, aggressor="buy"))
        s.on_tick(Tick(ts=T0, price=5800.50, size=5, aggressor="sell"))
        # Bar 0 by ABSOLUTE index. `-1` is relative and will point at
        # a different bar once more have closed -- which is correct
        # behaviour and exactly why a frozen bar needs a stable
        # handle to be tested at all.
        before = [(f.price_ticks, f.side, f.volume, f.state)
                  for f in s.confirmed_flags(0)]
        assert before, "bar 0 should have confirmed flags"
        for i in range(50):
            s.on_tick(Tick(ts=T0 + 10 + i, price=5800.25, size=50,
                           aggressor="buy"))
        after = [(f.price_ticks, f.side, f.volume, f.state)
                 for f in s.confirmed_flags(0)]
        assert before == after, "the frozen bar changed"
        assert len(s.history) > 1, "more bars did close"

    def test_a_closed_bar_is_marked_closed(self):
        s = stream(VolumeRule(50))
        s.on_tick(Tick(ts=T0, price=5800.0, size=50, aggressor="buy"))
        assert s.last_closed().closed is True


class TestFading:
    """A level that flashed an imbalance and lost it is a thing that
    happened. A structure holding only the current verdict cannot
    report it, and on a live panel that is the difference between
    'buyers are here' and 'buyers were here and got absorbed'."""

    def test_a_flag_that_stops_qualifying_is_recorded_as_faded(self):
        s = stream()
        s.on_tick(Tick(ts=T0, price=5800.00, size=5, aggressor="sell"))
        s.on_tick(Tick(ts=T0 + 1, price=5800.25, size=100,
                       aggressor="buy"))
        s.on_tick(Tick(ts=T0 + 2, price=5800.50, size=5,
                       aggressor="sell"))
        assert (23201, BUY_IMBALANCE) in live_keys(s)
        # sellers pile into the level below; the diagonal is no longer
        # four to one
        s.on_tick(Tick(ts=T0 + 3, price=5800.00, size=400,
                       aggressor="sell"))
        assert (23201, BUY_IMBALANCE) not in live_keys(s)
        assert any(f.price_ticks == 23201 and f.state == FADED
                   for f in s.tracker.faded)

    def test_a_faded_flag_keeps_how_long_it_held(self):
        s = stream()
        s.on_tick(Tick(ts=T0, price=5800.00, size=5, aggressor="sell"))
        s.on_tick(Tick(ts=T0 + 1, price=5800.25, size=100,
                       aggressor="buy"))
        s.on_tick(Tick(ts=T0 + 2, price=5800.50, size=5,
                       aggressor="sell"))
        s.on_tick(Tick(ts=T0 + 9, price=5800.00, size=400,
                       aggressor="sell"))
        f = next(f for f in s.tracker.faded if f.price_ticks == 23201)
        # First seen at T0+1, when the buy print created it -- NOT at
        # T0, when the bar opened. Faded at T0+9.
        assert f.first_seen_ts == pytest.approx(T0 + 1)
        assert f.held_s == pytest.approx(8.0)

    def test_the_peak_ratio_survives_the_fade(self):
        """What it reached matters as much as where it ended."""
        s = stream()
        s.on_tick(Tick(ts=T0, price=5800.00, size=1, aggressor="sell"))
        s.on_tick(Tick(ts=T0 + 1, price=5800.25, size=100,
                       aggressor="buy"))
        s.on_tick(Tick(ts=T0 + 2, price=5800.50, size=5,
                       aggressor="sell"))
        s.on_tick(Tick(ts=T0 + 3, price=5800.00, size=400,
                       aggressor="sell"))
        f = next(f for f in s.tracker.faded if f.price_ticks == 23201)
        assert f.peak_ratio == pytest.approx(100.0)


# ------------------------------------------------------ delta momentum

class TestPulse:

    def test_every_field_comes_from_one_instant(self):
        """A reader computing `ask - bid` from two separate reads can
        straddle a tick and report a delta that was never true."""
        s = stream()
        for i in range(50):
            s.on_tick(Tick(ts=T0 + i, price=5800.0, size=10,
                           aggressor="buy" if i % 2 else "sell"))
        p = s.pulse()
        assert p.delta == p.ask_volume - p.bid_volume
        assert p.total_volume == p.ask_volume + p.bid_volume

    def test_the_pulse_is_immutable(self):
        s = stream()
        s.on_tick(Tick(ts=T0, price=5800.0, size=1, aggressor="buy"))
        with pytest.raises(Exception):
            s.pulse().delta = 999

    def test_a_held_pulse_does_not_change_underneath_the_holder(self):
        s = stream()
        s.on_tick(Tick(ts=T0, price=5800.0, size=10, aggressor="buy"))
        held = s.pulse()
        snap = held.delta
        for i in range(20):
            s.on_tick(Tick(ts=T0 + i, price=5800.0, size=10,
                           aggressor="buy"))
        assert held.delta == snap, "the held pulse mutated"
        assert s.pulse().delta != snap, "the live one should have moved"

    def test_ratio_is_none_rather_than_infinite(self):
        """A large float invites a threshold comparison that silently
        always passes."""
        s = stream()
        s.on_tick(Tick(ts=T0, price=5800.0, size=10, aggressor="buy"))
        assert s.pulse().ratio is None

    def test_ratio_once_both_sides_have_traded(self):
        s = stream()
        s.on_tick(Tick(ts=T0, price=5800.0, size=100, aggressor="buy"))
        s.on_tick(Tick(ts=T0, price=5800.0, size=25, aggressor="sell"))
        assert s.pulse().ratio == pytest.approx(4.0)

    def test_pressure_is_normalised_so_bars_compare(self):
        """+400 means different things in a 500-lot bar and a
        50,000-lot one."""
        a = stream()
        a.on_tick(Tick(ts=T0, price=5800.0, size=400, aggressor="buy"))
        a.on_tick(Tick(ts=T0, price=5800.0, size=100, aggressor="sell"))
        b = stream()
        b.on_tick(Tick(ts=T0, price=5800.0, size=40_000,
                       aggressor="buy"))
        b.on_tick(Tick(ts=T0, price=5800.0, size=10_000,
                       aggressor="sell"))
        assert a.pulse().pressure == pytest.approx(b.pulse().pressure)

    def test_the_window_can_disagree_with_the_bar(self):
        """A bar can sit positive overall while the last stretch runs
        hard the other way. One number cannot say that."""
        s = FootprintStream(tick_size=0.25, rule=VolumeRule(1_000_000),
                            window_s=5.0)
        for i in range(20):
            s.on_tick(Tick(ts=T0 + i * 0.1, price=5800.0, size=100,
                           aggressor="buy"))
        for i in range(10):
            s.on_tick(Tick(ts=T0 + 20 + i, price=5800.0, size=100,
                           aggressor="sell"))
        p = s.pulse()
        assert p.delta > 0, "the bar is net bought"
        assert p.window_delta < 0, "the recent window is net sold"

    def test_the_window_is_bounded_by_time_not_by_count(self):
        w = Window(seconds=10.0)
        w.push(T0, 100)
        w.push(T0 + 5, 100)
        assert w.total == 200
        w.push(T0 + 20, 50)
        assert w.total == 50, "the old two should have aged out"

    def test_counts_of_flags_match_the_tracker(self):
        s = stream()
        for i in range(200):
            s.on_tick(Tick(ts=T0 + i, price=5800.0 + (i % 7) * 0.25,
                           size=(50 if i % 7 == 3 else 2),
                           aggressor="buy" if i % 3 else "sell"))
        p = s.pulse()
        assert p.unconfirmed == len(s.tracker.flags)
        assert p.buy_flags + p.sell_flags == p.unconfirmed


# ------------------------------------------------------------- the view

class TestView:

    def _fill(self, s, n=120):
        rng = random.Random(11)
        px = 5800.0
        for i in range(n):
            px += rng.choice((-0.25, 0.0, 0.25))
            s.on_tick(Tick(ts=T0 + i, price=round(px, 2),
                           size=rng.randint(1, 60),
                           aggressor=rng.choice(("buy", "sell"))))

    def test_rows_read_high_to_low(self):
        s = stream()
        self._fill(s)
        prices = [r["price_ticks"] for r in s.view().rows]
        assert prices == sorted(prices, reverse=True)

    def test_a_row_carries_its_imbalance_state(self):
        s = stream()
        self._fill(s)
        v = s.view()
        tagged = {r["price_ticks"] for r in v.rows if r["imbalance"]}
        assert tagged == {f["price_ticks"] for f in v.flags}

    def test_the_view_is_a_copy_the_feed_cannot_touch(self):
        s = stream()
        self._fill(s)
        v = s.view()
        before = [dict(r) for r in v.rows]
        self._fill(s, n=50)
        assert [dict(r) for r in v.rows] == before

    def test_it_serialises(self):
        s = stream()
        self._fill(s)
        import json
        json.dumps(s.view().as_dict())


# --------------------------------------------------------- concurrency

class TestNoRaceOnTheReadPath:

    def test_readers_never_see_a_torn_pulse(self):
        """Delta, ask and bid must satisfy their identity in every
        snapshot a reader takes while the feed is running."""
        s = stream()
        stop = threading.Event()
        bad: list = []

        def read():
            while not stop.is_set():
                p = s.pulse()
                if p.total_volume and (
                        p.delta != p.ask_volume - p.bid_volume
                        or p.total_volume != p.ask_volume + p.bid_volume):
                    bad.append(p)

        readers = [threading.Thread(target=read, daemon=True)
                   for _ in range(4)]
        for r in readers:
            r.start()
        for i in range(4000):
            s.on_tick(Tick(ts=T0 + i * 0.001,
                           price=5800.0 + (i % 11) * 0.25,
                           size=(i % 17) + 1,
                           aggressor="buy" if i % 3 else "sell"))
        stop.set()
        for r in readers:
            r.join(timeout=2)
        assert not bad, f"{len(bad)} torn reads"

    def test_two_feed_threads_do_not_lose_contracts(self):
        """One writer is the design, but the lock has to hold if a
        second one appears -- a lost print is silent."""
        s = stream()
        n, size = 500, 3

        def feed(k):
            for i in range(n):
                s.on_tick(Tick(ts=T0 + i * 0.001, price=5800.0,
                               size=size, aggressor="buy"))

        ts_ = [threading.Thread(target=feed, args=(k,)) for k in range(3)]
        for t in ts_:
            t.start()
        for t in ts_:
            t.join(timeout=10)
        assert s.pulse().total_volume == 3 * n * size

    def test_a_view_taken_during_the_feed_is_self_consistent(self):
        s = stream()
        stop = threading.Event()
        bad: list = []

        def read():
            while not stop.is_set():
                v = s.view()
                tot = sum(r["bid_volume"] + r["ask_volume"]
                          for r in v.rows)
                if v.rows and tot != v.pulse.total_volume:
                    bad.append(tot)

        t = threading.Thread(target=read, daemon=True)
        t.start()
        for i in range(2000):
            s.on_tick(Tick(ts=T0 + i * 0.001,
                           price=5800.0 + (i % 9) * 0.25,
                           size=(i % 13) + 1, aggressor="buy"))
        stop.set()
        t.join(timeout=2)
        assert not bad, f"{len(bad)} inconsistent views"


class TestHistory:

    def test_closed_bars_accumulate(self):
        s = stream(VolumeRule(100))
        for i in range(10):
            s.on_tick(Tick(ts=T0 + i, price=5800.0, size=100,
                           aggressor="buy"))
        assert len(s.history) == 10

    def test_history_is_bounded(self):
        s = FootprintStream(tick_size=0.25, rule=VolumeRule(10), keep=5)
        for i in range(40):
            s.on_tick(Tick(ts=T0 + i, price=5800.0, size=10,
                           aggressor="buy"))
        assert len(s.history) == 5

    def test_the_confirmed_map_is_trimmed_with_it(self):
        """Left to grow, it would hold the flags of bars that were
        dropped -- a leak that only shows up after a long session."""
        s = FootprintStream(tick_size=0.25, rule=VolumeRule(10), keep=5)
        for i in range(40):
            s.on_tick(Tick(ts=T0 + i, price=5800.0 + (i % 3) * 0.25,
                           size=10, aggressor="buy"))
        assert len(s.confirmed) <= 5
        assert all(0 <= k < 5 for k in s.confirmed)

    def test_a_time_bar_closes_on_roll_with_no_prints(self):
        s = stream(TimeRule(60.0))
        s.on_tick(Tick(ts=T0, price=5800.0, size=5, aggressor="buy"))
        assert len(s.roll(T0 + 300)) == 1


class TestItDecidesNothing:

    def test_no_order_code(self):
        src = (ROOT / "liqmap" / "footprint_live.py").read_text()
        for w in ("def submit", "def execute", "def place", "def order"):
            assert w not in src, w

    def test_the_tracker_does_not_recompute_the_whole_bar(self):
        """The O(1) claim. A full pass per tick would call
        `imbalances` from the hot path."""
        src = (ROOT / "liqmap" / "footprint_live.py").read_text()
        hot = src.split("def on_trade")[1].split("def confirm")[0]
        assert "imbalances(" not in hot


class TestTheSequenceCounter:
    """`seq` is how a poller tells a new pulse from the one it
    already drew. Off by one it still changes, so nothing breaks
    loudly -- it just stops meaning what it says."""

    def test_it_counts_every_tick_folded_in(self):
        s = stream()
        for i in range(37):
            s.on_tick(Tick(ts=T0 + i, price=5800.0, size=1,
                           aggressor="buy"))
        assert s.pulse().seq == 37

    def test_it_advances_on_every_tick(self):
        s = stream()
        seen = []
        for i in range(10):
            s.on_tick(Tick(ts=T0 + i, price=5800.0, size=1,
                           aggressor="buy"))
            seen.append(s.pulse().seq)
        assert seen == sorted(set(seen)) and len(seen) == 10
