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


class TestTheStackReadsImmediatelyAfterARestart:
    """Restoring the candles was necessary and not sufficient.

    With the history back the ten-bar gate opens, but there is still
    no bar IN PROGRESS -- it was in progress when the process died --
    and the first bar built mid-period is unseeded by definition,
    because its open was never observed. The live branch was skipped
    either way, every row fell to stored candles that carry no
    aggressor, and the stack came back unmeasured for up to a full
    bar: fifteen minutes of "the stack is incomplete" on the 15m, with
    nothing to tell it from a market nobody was trading.

    Chris saw exactly that after deploying the candle store, which is
    how this was found.
    """

    def _restarted(self, monkeypatch):
        from liqmap.candlestore import CandleRecorder, CandleStore
        from tests.test_cme_panels import _cme

        rt, tape = _cme(monkeypatch)
        rt.footprint(rule="1m", volume=0, min_volume=0, multiplier=4.0)
        rec = rt._candlerec
        rec.flush()
        rec.stop()
        for b in tape.builders.values():
            b.closed.clear()
            b.current = None
        CandleRecorder(CandleStore(rec.store.path),
                       tape).load_into_builders()
        assert tape.builders["15m"].current is None
        return rt, tape

    def _read(self, client):
        r = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers={"Authorization": "Bearer t0ken"})
        assert r.status_code == 200, r.text
        return r.json()

    def test_every_row_is_measured(self, client, monkeypatch):
        self._restarted(monkeypatch)
        rows = self._read(client).get("timeframes") or []
        assert {r["timeframe"] for r in rows} >= {"15m", "5m", "1m"}
        unmeasured = [r["timeframe"] for r in rows if not r["measured"]]
        assert not unmeasured, f"unmeasured after restart: {unmeasured}"

    def test_the_stack_is_not_incomplete(self, client, monkeypatch):
        self._restarted(monkeypatch)
        why = (self._read(client).get("cascade") or {}).get("why") or ""
        assert "stack is incomplete" not in why, why

    def test_the_rungs_carry_a_direction(self, client, monkeypatch):
        """Rows that are present and colourless are the other half of
        what Chris was looking at."""
        self._restarted(monkeypatch)
        rungs = (self._read(client).get("cascade") or {}).get("rungs") or []
        assert rungs
        assert all(g["present"] for g in rungs), rungs
        assert any(g["qualified"] for g in rungs), rungs

    def test_the_row_says_it_is_reading_a_finished_bar(self, client,
                                                       monkeypatch):
        """IT MUST NOT PASS AS LIVE. A reading of the last closed bar
        can be most of an interval old, and reporting the feed's age
        for it would make a fifteen-minute-old row look current."""
        self._restarted(monkeypatch)
        rows = self._read(client).get("timeframes") or []
        byname = {r["timeframe"]: r for r in rows}
        assert byname["15m"]["source"] == "last closed"
        assert byname["15m"]["age_s"] > byname["1m"]["age_s"]

    def test_it_goes_back_to_the_live_bar_once_there_is_one(
            self, client, monkeypatch):
        """The fallback is for the gap, not for good."""
        from liqmap.live import LiveCandle

        _rt, tape = self._restarted(monkeypatch)
        b = tape.builders["1m"]
        last = b.closed[-1]
        b.current = LiveCandle(
            interval_s=60.0, start_ts=last.start_ts + 60.0,
            open=last.close, high=last.close + 1, low=last.close - 1,
            close=last.close + 0.5, volume=50.0, trades=9,
            buy_notional=800.0, sell_notional=100.0, seeded=True)
        rows = self._read(client).get("timeframes") or []
        row = {r["timeframe"]: r for r in rows}["1m"]
        assert row["source"] == "tape", row
        assert row["measured"] is True

    def test_an_unmeasured_closed_bar_is_not_used(self, client,
                                                  monkeypatch):
        """A bar restored without flow would hand the ladder an
        invented direction. Better no row than a made-up one."""
        _rt, tape = self._restarted(monkeypatch)
        for c in tape.builders["5m"].closed:
            c.flow_measured = False
        rows = self._read(client).get("timeframes") or []
        row = {r["timeframe"]: r for r in rows}.get("5m")
        assert row is None or row["measured"] is False


class TestThePanelSaysWhyNotJustWhat:
    """Three separate faults have produced the identical sentence
    "no read on 15m, 5m, 1m -- the stack is incomplete": a venue
    mix-up that emptied the ladder, an empty candle history, and a
    missing bar in progress after a restart. Each cost a round trip to
    tell apart, because the screen could say what was missing and
    never why. The route knew every time.
    """

    def _read(self, client):
        r = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers={"Authorization": "Bearer t0ken"})
        return r.json()

    def _blocked(self, client):
        casc = self._read(client).get("cascade") or {}
        return {b["timeframe"]: b["why"]
                for b in (casc.get("blocked") or [])}

    def test_a_healthy_stack_blocks_nothing(self, client, monkeypatch):
        from tests.test_cme_panels import _cme

        _cme(monkeypatch)
        assert self._blocked(client) == {}

    def test_an_unseeded_bar_with_no_usable_history_says_both(
            self, client, monkeypatch):
        from tests.test_cme_panels import _cme

        _rt, tape = _cme(monkeypatch)
        for b in tape.builders.values():
            if b.current is not None:
                b.current.seeded = False
            for c in b.closed:
                c.flow_measured = False
        got = self._blocked(client)
        assert set(got) >= {"15m", "5m", "1m"}, got
        assert "unseeded" in got["15m"], got["15m"]
        assert "no aggressor" in got["15m"], got["15m"]

    def test_history_without_flow_says_so(self, client, monkeypatch):
        """The failure mode of a candle store that saved prices and
        dropped the buy/sell split -- it clears the bar-count gate and
        reads nothing, and used to look identical to a dead feed."""
        from tests.test_cme_panels import _cme

        _rt, tape = _cme(monkeypatch)
        for b in tape.builders.values():
            b.current = None
            for c in b.closed:
                c.flow_measured = False
        got = self._blocked(client)
        assert "no bar in progress" in got["15m"], got["15m"]
        assert "no aggressor" in got["15m"], got["15m"]

    def test_a_blocked_row_never_goes_unexplained(self, client,
                                                  monkeypatch):
        """A missing row with no reason attached is a bug in this
        reporting, and it says so rather than printing a blank."""
        from tests.test_cme_panels import _cme

        _rt, tape = _cme(monkeypatch)
        for b in tape.builders.values():
            b.current = None
            for c in b.closed:
                c.flow_measured = False
        for why in self._blocked(client).values():
            assert why.strip()
            assert "that is a bug" not in why, why

    def test_the_page_draws_them(self):
        """The reasons reach the screen, not just the payload."""
        web_src = (Path(__file__).resolve().parent.parent / "liqmap"
                   / "web.py").read_text()
        i = web_src.index("function paintCascade(")
        blk = web_src[i:i + 4000]
        assert "C.blocked" in blk
        assert "why these rows did not read" in blk


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
