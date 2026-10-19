"""The resting book, read diagonally and accumulated over a candle.

The central claim this file defends: a book is a STATE, not a flow.
Summing snapshots counts the same resting contract once per update,
so a book that is re-reported a thousand times during a burst would
drown out one that stood quietly for a minute. Every accumulation
here is weighted by how long the book actually stood.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from liqmap.bookdiag import (                              # noqa: E402
    BookDiagonal, LevelRead, Reading, _weight)

PT, TICK = 1_000_000_000, 250_000_000
BASE, T0 = 5800 * PT, 1_700_000_000.0


def rec(ts, bids, asks):
    """An MBP-10-shaped record. bids/asks are [(sz, ct), ...] from
    the touch outwards."""
    o = type("R", (), {})()
    o.ts_event = int(ts * 1e9)
    o.action, o.side = "A", "B"
    lv = []
    for k, ((bs, bc), (a_s, ac)) in enumerate(zip(bids, asks)):
        L = type("L", (), {})()
        L.bid_px, L.bid_sz, L.bid_ct = BASE - k * TICK, bs, bc
        L.ask_px, L.ask_sz, L.ask_ct = BASE + (k + 1) * TICK, a_s, ac
        lv.append(L)
    o.levels = lv
    return o


FLAT = [(100, 10)] * 5
HEAVY = [(400, 40)] * 5
LIGHT = [(50, 5)] * 5


class TestItIsTimeWeightedNotSummed:
    """The failure this prevents: a quiet book with a large bid
    accumulating enormous bid pressure purely by sitting still and
    being re-reported."""

    def test_a_burst_does_not_outweigh_a_long_quiet_stretch(self):
        bd = BookDiagonal()
        # one second of bid dominance, reported two hundred times
        for i in range(200):
            bd.observe(rec(T0 + i * 0.005, HEAVY, LIGHT))
        # then four seconds of ask dominance, reported four times
        for i in range(4):
            bd.observe(rec(T0 + 1 + i, LIGHT, HEAVY))
        bd.observe(rec(T0 + 5, LIGHT, HEAVY))
        r = bd.reading()
        assert r.size_mean < 0, (
            "the 200-update burst outvoted four seconds of the book "
            "actually standing the other way")

    def test_the_same_book_for_longer_weighs_more(self):
        """Every interval here stays UNDER the gap cap. Written with
        a 9.9s step it fails, because that gap is excluded entirely
        and the short case then averages over nothing but its one
        heavy snapshot -- which is the cap behaving correctly and the
        test asking the wrong question."""
        short = BookDiagonal(max_gap_s=5.0)
        short.observe(rec(T0, HEAVY, LIGHT))
        short.observe(rec(T0 + 0.5, FLAT, FLAT))
        short.observe(rec(T0 + 4.5, FLAT, FLAT))
        short.observe(rec(T0 + 8.5, FLAT, FLAT))

        long_ = BookDiagonal(max_gap_s=5.0)
        long_.observe(rec(T0, HEAVY, LIGHT))
        long_.observe(rec(T0 + 4, HEAVY, LIGHT))
        long_.observe(rec(T0 + 4.5, FLAT, FLAT))
        long_.observe(rec(T0 + 8.5, FLAT, FLAT))

        assert long_.reading().size_mean > short.reading().size_mean

    def test_update_count_alone_moves_nothing(self):
        a = BookDiagonal()
        b = BookDiagonal()
        for i in range(2):
            a.observe(rec(T0 + i * 5, HEAVY, LIGHT))
        for i in range(50):
            b.observe(rec(T0 + i * 0.2, HEAVY, LIGHT))
        b.observe(rec(T0 + 10, HEAVY, LIGHT))
        a.observe(rec(T0 + 10, HEAVY, LIGHT))
        assert a.reading().size_mean == pytest.approx(
            b.reading().size_mean, abs=0.02)

    def test_a_long_silence_is_not_integrated(self):
        """A gap is a halt or a disconnect, not a book that stood.
        Counting it would let one stale snapshot own the candle."""
        bd = BookDiagonal(max_gap_s=5.0)
        bd.observe(rec(T0, HEAVY, LIGHT))
        bd.observe(rec(T0 + 600, LIGHT, HEAVY))
        bd.observe(rec(T0 + 601, LIGHT, HEAVY))
        assert bd.reading().seconds <= 5.0

    def test_time_in_front_is_tracked_per_side(self):
        bd = BookDiagonal()
        for i in range(3):
            bd.observe(rec(T0 + i, HEAVY, LIGHT))
        for i in range(3, 10):
            bd.observe(rec(T0 + i, LIGHT, HEAVY))
        bd.observe(rec(T0 + 10, LIGHT, HEAVY))
        r = bd.reading()
        assert r.bid_seconds == pytest.approx(3.0)
        assert r.ask_seconds == pytest.approx(7.0)
        assert r.held == pytest.approx(0.3)


class TestNowVersusSinceOpen:
    """A side that is ahead right now and has been ahead all bar is
    a different claim from one that just arrived."""

    def test_they_can_disagree(self):
        bd = BookDiagonal()
        for i in range(20):
            bd.observe(rec(T0 + i, LIGHT, HEAVY))
        bd.observe(rec(T0 + 21, HEAVY, LIGHT))
        r = bd.reading()
        assert r.winning_now() == "bid"
        assert r.winning_since_open() == "ask"

    def test_a_balanced_book_names_no_winner(self):
        bd = BookDiagonal()
        bd.observe(rec(T0, FLAT, FLAT))
        bd.observe(rec(T0 + 1, FLAT, FLAT))
        r = bd.reading()
        assert r.winning_now() == ""
        assert r.winning_since_open() == ""

    def test_the_edge_is_the_callers(self):
        bd = BookDiagonal()
        bd.observe(rec(T0, [(120, 12)] * 5, [(100, 10)] * 5))
        bd.observe(rec(T0 + 1, [(120, 12)] * 5, [(100, 10)] * 5))
        r = bd.reading()
        assert r.winning_now(edge=0.05) == "bid"
        assert r.winning_now(edge=0.5) == ""


class TestTheDiagonalIsPerLevel:
    """Summing a diagonal over every level collapses to total bid
    minus total ask -- the pairing cancels. The structure only
    survives if it is read per level and weighted by distance."""

    def test_the_touch_counts_more_than_the_back(self):
        near = BookDiagonal()
        near.observe(rec(T0, [(400, 40)] + [(100, 10)] * 4,
                         [(100, 10)] * 5))
        far = BookDiagonal()
        far.observe(rec(T0, [(100, 10)] * 4 + [(400, 40)],
                        [(100, 10)] * 5))
        assert near.reading().size_now > far.reading().size_now

    def test_weights_fall_off_from_the_touch(self):
        assert _weight(0) > _weight(1) > _weight(9) > 0

    def test_every_level_is_reported(self):
        bd = BookDiagonal(depth=5)
        bd.observe(rec(T0, FLAT, FLAT))
        assert len(bd.reading().levels) == 5
        assert [lv.depth for lv in bd.reading().levels] == [0, 1, 2, 3, 4]

    def test_a_rung_is_won_by_the_multiplier(self):
        lv = LevelRead(depth=0, bid_sz=400, ask_sz=100)
        assert lv.winner(2.0) == "bid"
        assert lv.winner(8.0) == ""

    def test_an_empty_side_hands_the_rung_over(self):
        assert LevelRead(depth=0, bid_sz=50, ask_sz=0).winner() == "bid"
        assert LevelRead(depth=0, bid_sz=0, ask_sz=50).winner() == "ask"

    def test_an_empty_rung_has_no_winner(self):
        assert LevelRead(depth=0).winner() == ""

    def test_rungs_are_counted_per_side(self):
        bd = BookDiagonal()
        bd.observe(rec(T0, [(400, 40), (400, 40), (100, 10),
                            (10, 1), (10, 1)],
                       [(100, 10), (100, 10), (100, 10),
                        (400, 40), (400, 40)]))
        r = bd.reading()
        assert r.bid_rungs == 2 and r.ask_rungs == 2


class TestContractsAndOrdersAreBothCounted:
    """500 contracts in one order is a single participant who can
    withdraw in one message. 500 across forty orders is forty people
    who would all have to change their minds."""

    def test_they_can_point_opposite_ways(self):
        bd = BookDiagonal()
        # bid has the contracts, ask has the participants
        bd.observe(rec(T0, [(500, 2)] * 5, [(300, 60)] * 5))
        bd.observe(rec(T0 + 1, [(500, 2)] * 5, [(300, 60)] * 5))
        r = bd.reading()
        assert r.size_now > 0, "contracts favour the bid"
        assert r.order_now < 0, "orders favour the ask"

    def test_both_are_accumulated_over_the_candle(self):
        bd = BookDiagonal()
        bd.observe(rec(T0, [(500, 2)] * 5, [(300, 60)] * 5))
        bd.observe(rec(T0 + 4, [(500, 2)] * 5, [(300, 60)] * 5))
        r = bd.reading()
        assert r.size_mean > 0 and r.order_mean < 0

    def test_raw_totals_are_reported_too(self):
        bd = BookDiagonal(depth=5)
        bd.observe(rec(T0, [(100, 10)] * 5, [(50, 5)] * 5))
        r = bd.reading()
        assert r.bid_sz == 500 and r.bid_ct == 50
        assert r.ask_sz == 250 and r.ask_ct == 25


class TestScopedToTheCandle:

    def test_reset_clears_the_accumulation(self):
        bd = BookDiagonal()
        for i in range(10):
            bd.observe(rec(T0 + i, HEAVY, LIGHT))
        assert bd.reading().size_mean > 0
        bd.reset()
        assert bd.reading().seconds == 0.0
        assert bd.reading().updates == 0

    def test_reset_KEEPS_the_standing_book(self):
        """A bar closing does not cancel anyone's resting orders.
        Blanking the snapshot would make the new candle's first
        reading say "no book" about a book that is plainly there."""
        bd = BookDiagonal()
        for i in range(5):
            bd.observe(rec(T0 + i, HEAVY, LIGHT))
        before = bd.reading().size_now
        bd.reset()
        r = bd.reading()
        assert r.size_now == pytest.approx(before)
        assert r.winning_now() == "bid"
        assert r.levels, "the book vanished"

    def test_a_hard_reset_forgets_it(self):
        """For a reconnect or a session break, where the old snapshot
        really has stopped describing anything."""
        bd = BookDiagonal()
        bd.observe(rec(T0, HEAVY, LIGHT))
        bd.reset(keep_book=False)
        assert bd.reading().levels == ()
        assert bd.reading().winning_now() == ""

    def test_after_reset_the_next_candle_starts_clean(self):
        bd = BookDiagonal()
        for i in range(10):
            bd.observe(rec(T0 + i, HEAVY, LIGHT))
        bd.reset()
        for i in range(10, 20):
            bd.observe(rec(T0 + i, LIGHT, HEAVY))
        bd.observe(rec(T0 + 20, LIGHT, HEAVY))
        assert bd.reading().size_mean < 0, "the old candle leaked in"

    def test_a_reading_before_any_record_is_empty_not_wrong(self):
        r = BookDiagonal().reading()
        assert r.updates == 0 and r.winning_now() == ""
        assert r.levels == ()


class TestRecordsWithoutABook:

    def test_a_trade_record_is_ignored(self):
        """`trades` carries no levels. It must not be counted as a
        book of zero size."""
        o = type("R", (), {})()
        o.ts_event = int(T0 * 1e9)
        o.action, o.side = "T", "B"
        o.price, o.size = BASE, 3
        bd = BookDiagonal()
        assert bd.observe(o) is False
        assert bd.reading().updates == 0

    def test_a_book_record_is_counted(self):
        bd = BookDiagonal()
        assert bd.observe(rec(T0, FLAT, FLAT)) is True


class TestItDecidesNothing:

    def test_no_order_code(self):
        src = (ROOT / "liqmap" / "bookdiag.py").read_text()
        # The paren matters: `def order_imb` is an order-COUNT
        # imbalance and would trip a bare "def order" check, which
        # would then have to be loosened everywhere.
        for w in ("def submit(", "def execute(", "def place(",
                  "def order("):
            assert w not in src, w

    def test_the_multiplier_and_edge_are_arguments(self):
        bd = BookDiagonal(multiplier=9.0)
        bd.observe(rec(T0, [(400, 40)] * 5, [(100, 10)] * 5))
        assert bd.reading().bid_rungs == 0, "9x was not applied"

    def test_it_does_not_touch_the_footprint(self):
        """The footprint counts what TRADED. This counts what is
        RESTING. Neither substitutes for the other."""
        src = (ROOT / "liqmap" / "bookdiag.py").read_text()
        assert "import footprint" not in src
        assert "from .footprint" not in src
