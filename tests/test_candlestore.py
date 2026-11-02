"""Closed CME candles survive a restart.

THE FAULT THIS FIXES, in Chris's words: the stack under the timeframe
shows rows that say "building it" and never gain a direction. The
cause was not the ladder and not the feed. `/api/read` refuses to
build any row until the traded interval has ten closed bars, and the
CME tape held nothing from before the engine started -- so after every
restart the panel waited two and a half hours at 15m, with no way to
tell that from a quiet market.

The OFI bars never had this problem because they are already persisted
and reloaded on boot, which is why the footprint panel came back
instantly beside an empty stack.

WHAT THESE TESTS GUARD, in order of how quietly each would fail:

  1. THE FLOW SURVIVES. A candle restored as OHLCV alone clears the
     ten-bar gate and reads contested forever, because the ladder's
     direction comes from buy against sell. `CandleBuilder.seed` sets
     flow_measured=False for exactly that reason -- it is the obvious
     way to restore history and it is the wrong one. This is the
     failure that would look fixed.
  2. order. The store returns oldest-first because that is what every
     reader assumes; newest-first would reverse the chart after a
     restart and nowhere else.
  3. a closed bar is never rewritten. It is a fact, and a
     reload-then-reflush cycle must not overwrite it.
  4. nothing restores over live bars, and the in-progress bar is never
     restored at all -- it was in progress when the process died.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from liqmap.candlestore import CandleRecorder, CandleStore
from liqmap.live import CandleBuilder, LiveCandle

# The CME attach path needs a configured engine and a database, which
# is exactly what the panel suite's fixture already sets up. Imported
# rather than copied so the two cannot drift about what "configured"
# means.
from tests.test_cme_panels import client  # noqa: F401,E402


def candle(start_ts: float, px: float, interval_s: float = 60.0,
           buy: float = 0.0, sell: float = 0.0,
           measured: bool = True) -> LiveCandle:
    return LiveCandle(
        interval_s=interval_s, start_ts=start_ts, open=px,
        high=px + 1.0, low=px - 1.0, close=px + 0.5, volume=100.0,
        trades=20, buy_notional=buy, sell_notional=sell,
        flow_measured=measured)


class Tape:
    """Only what the recorder touches."""

    def __init__(self, symbol="ES.Z26"):
        self.symbol = symbol
        self.builders: dict = {}
        self.closed: dict = {}

    def add_interval(self, name, secs, closed=()):
        b = CandleBuilder(secs)
        b.closed.extend(closed)
        self.builders[name] = b
        self.closed[name] = []
        return b


@pytest.fixture()
def store(tmp_path: Path) -> CandleStore:
    return CandleStore(tmp_path / "candles.db")


class TestTheFlowSurvivesTheRoundTrip:
    """The one that would look fixed and read nothing."""

    def test_buy_and_sell_come_back(self, store):
        store.record_many("ES.Z26", "1m",
                          [candle(0.0, 6700.0, buy=9000.0, sell=1200.0)])
        got = store.recent("ES.Z26", "1m")
        assert len(got) == 1
        assert got[0].buy_notional == 9000.0
        assert got[0].sell_notional == 1200.0

    def test_flow_measured_comes_back_true(self, store):
        store.record_many("ES.Z26", "1m",
                          [candle(0.0, 6700.0, buy=5.0, sell=1.0)])
        assert store.recent("ES.Z26", "1m")[0].flow_measured is True

    def test_flow_measured_false_is_not_promoted(self, store):
        """A bar that never saw an aggressor must not come back
        claiming one. Its zeros are not a balanced bar."""
        store.record_many("ES.Z26", "1m",
                          [candle(0.0, 6700.0, measured=False)])
        assert store.recent("ES.Z26", "1m")[0].flow_measured is False

    def test_a_restored_bar_still_has_a_winning_side(self, store):
        """What the ladder actually asks of it."""
        store.record_many("ES.Z26", "5m",
                          [candle(0.0, 6700.0, 300.0, buy=90.0,
                                  sell=10.0)])
        c = store.recent("ES.Z26", "5m")[0]
        assert c.buy_notional > c.sell_notional
        assert c.flow_measured

    def test_prices_and_counts_survive(self, store):
        c = candle(60.0, 6700.0, buy=3.0, sell=4.0)
        store.record_many("ES.Z26", "1m", [c])
        g = store.recent("ES.Z26", "1m")[0]
        for f in ("start_ts", "interval_s", "open", "high", "low",
                  "close", "volume", "trades"):
            assert getattr(g, f) == getattr(c, f), f


class TestOrderAndIdentity:
    def test_oldest_first(self, store):
        store.record_many("ES.Z26", "1m",
                          [candle(t * 60.0, 6700.0 + t) for t in range(5)])
        got = store.recent("ES.Z26", "1m")
        assert [c.start_ts for c in got] == [0.0, 60.0, 120.0, 180.0,
                                             240.0]

    def test_a_limit_keeps_the_newest(self, store):
        store.record_many("ES.Z26", "1m",
                          [candle(t * 60.0, 6700.0) for t in range(10)])
        got = store.recent("ES.Z26", "1m", limit=3)
        assert [c.start_ts for c in got] == [420.0, 480.0, 540.0]

    def test_a_closed_bar_is_never_rewritten(self, store):
        store.record_many("ES.Z26", "1m", [candle(0.0, 6700.0)])
        n = store.record_many("ES.Z26", "1m", [candle(0.0, 9999.0)])
        assert n == 0
        assert store.recent("ES.Z26", "1m")[0].open == 6700.0

    def test_intervals_do_not_mix(self, store):
        store.record_many("ES.Z26", "1m", [candle(0.0, 6700.0)])
        store.record_many("ES.Z26", "5m", [candle(0.0, 6800.0, 300.0)])
        assert len(store.recent("ES.Z26", "1m")) == 1
        assert store.recent("ES.Z26", "5m")[0].open == 6800.0

    def test_symbols_do_not_mix(self, store):
        store.record_many("ES.Z26", "1m", [candle(0.0, 6700.0)])
        store.record_many("ES.H27", "1m", [candle(0.0, 6800.0)])
        assert store.recent("ES.Z26", "1m")[0].open == 6700.0
        assert store.recent("ES.H27", "1m")[0].open == 6800.0


class TestTheRestartItself:
    """The whole point: an engine that starts with nothing ends up
    with the history it had."""

    def test_a_cold_tape_gets_its_candles_back(self, store):
        before = Tape()
        b = before.add_interval("15m", 900.0,
                                [candle(t * 900.0, 6700.0 + t, 900.0,
                                        buy=90.0, sell=10.0)
                                 for t in range(20)])
        rec = CandleRecorder(store, before)
        rec.seed()
        assert rec.flush() == 20

        after = Tape()
        after.add_interval("15m", 900.0)
        rec2 = CandleRecorder(store, after)
        restored = rec2.load_into_builders()
        assert restored == 20
        got = after.builders["15m"].closed
        assert len(got) == 20
        # AND THE TEN-BAR GATE IS CLEARED, which is the thing Chris
        # was waiting two and a half hours for.
        assert len(got) >= 10
        assert all(c.flow_measured for c in got)
        assert all(c.buy_notional > c.sell_notional for c in got)
        assert b.closed[0].start_ts == got[0].start_ts

    def test_it_does_not_restore_over_live_bars(self, store):
        before = Tape()
        before.add_interval("1m", 60.0,
                            [candle(t * 60.0, 6700.0) for t in range(12)])
        CandleRecorder(store, before).flush()

        live = Tape()
        live.add_interval("1m", 60.0, [candle(9999.0, 7000.0)])
        n = CandleRecorder(store, live).load_into_builders()
        assert n == 0
        assert len(live.builders["1m"].closed) == 1
        assert live.builders["1m"].closed[0].open == 7000.0

    def test_the_bar_in_progress_is_never_stored(self, store):
        """It was in progress when the process died: its open is real
        and everything after it is a guess."""
        t = Tape()
        b = t.add_interval("1m", 60.0,
                           [candle(0.0, 6700.0), candle(60.0, 6701.0)])
        b.current = candle(120.0, 6702.0)
        CandleRecorder(store, t).flush()
        assert store.count("ES.Z26", "1m") == 2
        assert [c.start_ts for c in store.recent("ES.Z26", "1m")] \
            == [0.0, 60.0]

    def test_a_reload_does_not_rewrite_what_it_read(self, store):
        """seed() takes the place marker from disk, so the first flush
        after a restart writes only what is genuinely new."""
        first = Tape()
        first.add_interval("1m", 60.0,
                           [candle(t * 60.0, 6700.0) for t in range(5)])
        CandleRecorder(store, first).flush()

        second = Tape()
        bb = second.add_interval("1m", 60.0)
        rec = CandleRecorder(store, second)
        rec.load_into_builders()
        rec.seed()
        assert rec.flush() == 0
        bb.closed.append(candle(300.0, 6705.0))
        assert rec.flush() == 1

    def test_only_what_closed_since_last_time_is_written(self, store):
        t = Tape()
        b = t.add_interval("1m", 60.0,
                           [candle(x * 60.0, 6700.0) for x in range(3)])
        rec = CandleRecorder(store, t)
        rec.seed()
        assert rec.flush() == 3
        assert rec.flush() == 0
        b.closed.append(candle(180.0, 6703.0))
        assert rec.flush() == 1

    def test_it_respects_the_builders_depth(self, store):
        store.record_many("ES.Z26", "1m",
                          [candle(x * 60.0, 6700.0) for x in range(50)])
        t = Tape()
        b = t.add_interval("1m", 60.0)
        b.max_closed = 10
        CandleRecorder(store, t).load_into_builders()
        assert len(b.closed) == 10
        # The NEWEST ten, not the oldest.
        assert b.closed[-1].start_ts == 49 * 60.0


class TestItNeverStallsTheFeed:
    def test_a_broken_store_is_counted_not_raised(self, store):
        t = Tape()
        t.add_interval("1m", 60.0, [candle(0.0, 6700.0)])

        class Broken:
            def record_many(self, *a, **k):
                raise OSError("disk went away")

            def last_start(self, *a, **k):
                raise OSError("disk went away")

        rec = CandleRecorder(Broken(), t)
        rec.seed()
        assert rec.flush() == 0
        assert rec.errors >= 1

    def test_a_broken_read_leaves_the_builder_alone(self, store):
        t = Tape()
        b = t.add_interval("1m", 60.0)

        class Broken:
            def recent(self, *a, **k):
                raise OSError("disk went away")

        rec = CandleRecorder(Broken(), t)
        assert rec.load_into_builders() == 0
        assert rec.errors >= 1
        assert b.closed == []

    def test_prune_keeps_the_newest(self, store):
        store.record_many("ES.Z26", "1m",
                          [candle(x * 60.0, 6700.0) for x in range(30)])
        store.prune(keep=10)
        got = store.recent("ES.Z26", "1m", limit=100)
        assert len(got) == 10
        assert got[-1].start_ts == 29 * 60.0

    def test_status_says_what_is_held(self, store):
        t = Tape()
        t.add_interval("1m", 60.0, [candle(0.0, 6700.0)])
        rec = CandleRecorder(store, t)
        rec.seed()
        rec.flush()
        st = rec.status()
        assert st["symbol"] == "ES.Z26"
        assert st["written"] == 1
        assert st["on_disk"]["1m"] == 1


class TestItIsActuallyWiredIn:
    """A store nothing calls is a file that never fills.

    These run the real attach path rather than constructing a recorder
    by hand, because the bug being fixed was never in the store -- it
    was that nothing was keeping the candles at all.
    """

    def test_building_the_footprint_attaches_the_recorder(
            self, client, monkeypatch):
        from tests.test_cme_panels import _cme

        rt, tape = _cme(monkeypatch)
        assert getattr(rt, "_candlerec", None) is None
        rt.footprint(rule="1m", volume=0, min_volume=0, multiplier=4.0)
        rec = getattr(rt, "_candlerec", None)
        assert rec is not None, "nothing is saving the CME candles"
        try:
            assert rec.flush() > 0
            on_disk = rec.status()["on_disk"]
            assert on_disk.get("1m", 0) > 0
            assert on_disk.get("15m", 0) > 0
        finally:
            rec.stop()

    def test_a_restart_clears_the_ten_bar_gate(self, client,
                                               monkeypatch):
        """THE FAULT, END TO END. /api/read builds no rows at all until
        the traded interval has ten closed bars, so an engine that
        starts empty shows a directionless stack for two and a half
        hours at 15m."""
        from liqmap.candlestore import CandleRecorder, CandleStore
        from tests.test_cme_panels import _cme

        rt, tape = _cme(monkeypatch)
        rt.footprint(rule="1m", volume=0, min_volume=0, multiplier=4.0)
        rec = getattr(rt, "_candlerec", None)
        assert rec is not None
        rec.flush()
        rec.stop()
        store = rec.store
        symbol = rec.symbol

        # A restart: the builders come back empty, as they do when the
        # process dies and the tape starts from the next print.
        for b in tape.builders.values():
            b.closed.clear()
            b.current = None
        assert len(tape.builders["15m"].closed) == 0

        cold = CandleRecorder(CandleStore(store.path), tape)
        restored = cold.load_into_builders()
        assert restored > 0
        assert len(tape.builders["15m"].closed) >= 10, \
            "the gate is still shut after a restart"
        assert all(c.flow_measured
                   for c in tape.builders["15m"].closed)


class TestTheThread:
    def test_it_starts_and_stops(self, store):
        t = Tape()
        t.add_interval("1m", 60.0, [candle(0.0, 6700.0)])
        rec = CandleRecorder(store, t, period_s=0.05)
        rec.seed()
        rec.start()
        try:
            deadline = time.time() + 3.0
            while time.time() < deadline and rec.written < 1:
                time.sleep(0.02)
            assert rec.written >= 1
        finally:
            rec.stop()
        assert rec.status()["running"] is False
