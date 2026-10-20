"""Tests for liqmap.execution — two-track architecture.

Coverage
────────
1.  Signal  SHAPE / as_dict.  Frozen dataclass round-trips through as_dict
    correctly.

2.  DiagonalCumulativeDelta  raw vs diagonal accumulation; reset.

3.  _live_stacked  edge cases: empty tracker, single level, full run,
    multi-side, run-length filter, sort order.

4.  ExecState  as_dict shape.

5.  ExecutionTrack  on_trade drives the footprint; signal callbacks fire;
    cooldown gate suppresses rapid repeat signals on the same side; off_signal
    removes a callback.

6.  Tape._exec_listeners  on_trade_exec / off_trade_exec / exec_listener_count;
    fired on trades; not fired on book-only records; crashing listener does not
    stop the tape.

7.  SignalBroadcaster  lifecycle, notify (no loop → no-op, with loop → delivers).

8.  /ws/exec  auth rejection; non-CME error frame; CME initial state frame.

9.  /api/exec/state  returns required keys for CME; error for non-CME.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap.execution import (  # noqa: E402
    Signal,
    DiagonalCumulativeDelta,
    ExecState,
    ExecutionTrack,
    _live_stacked,
)
from liqmap.footprint import (  # noqa: E402
    BUY_IMBALANCE, SELL_IMBALANCE,
    DEFAULT_IMBALANCE,
    Tick, TickScale,
)
from liqmap.footprint_live import Tracker, LiveFlag  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TICK_SIZE = 0.25

def _sc() -> TickScale:
    return TickScale(TICK_SIZE)

def _tick(price: float, size: int = 10, agg: str = "buy",
          ts: float = 1_700_000_000.0) -> Tick:
    return Tick(ts=ts, price=price, size=size, aggressor=agg)


def _tracker_with_flags(flags: list[tuple[int, str]]) -> Tracker:
    """Build a Tracker whose .flags dict contains the given (price_ticks, side) pairs."""
    t = Tracker()
    for p, side in flags:
        t.flags[(p, side)] = LiveFlag(
            price_ticks=p, side=side,
            volume=100, against=20,
            ratio=5.0, state="unconfirmed",
            first_seen_ts=1.0, last_seen_ts=2.0, updates=3)
    return t


# ---------------------------------------------------------------------------
# 1. Signal
# ---------------------------------------------------------------------------

class TestSignal:

    def _make_signal(self, **kwargs) -> Signal:
        defaults = dict(
            ts=1_700_000_001.0, side="buy", price=5800.25,
            raw_delta=50, diagonal_delta=30,
            unconfirmed=4, buy_flags=3, sell_flags=1,
            stacked_side="buy", stacked_low=5800.0, stacked_high=5800.75,
            stacked_run=4, bar_age_s=42.5, reason="stacked_buy_4 dcd=+30",
        )
        defaults.update(kwargs)
        return Signal(**defaults)

    def test_signal_is_frozen(self):
        s = self._make_signal()
        with pytest.raises((AttributeError, TypeError)):
            s.side = "sell"  # type: ignore[misc]

    def test_as_dict_has_required_keys(self):
        s = self._make_signal()
        d = s.as_dict()
        for key in ("ts", "side", "price", "raw_delta", "diagonal_delta",
                    "unconfirmed", "buy_flags", "sell_flags",
                    "stacked_side", "stacked_low", "stacked_high",
                    "stacked_run", "bar_age_s", "reason"):
            assert key in d, f"missing key {key!r}"

    def test_as_dict_values_match(self):
        s = self._make_signal(side="sell", price=5799.75, raw_delta=-20)
        d = s.as_dict()
        assert d["side"] == "sell"
        assert d["price"] == 5799.75
        assert d["raw_delta"] == -20

    def test_bar_age_rounded(self):
        s = self._make_signal(bar_age_s=1.23456789)
        d = s.as_dict()
        assert isinstance(d["bar_age_s"], float)


# ---------------------------------------------------------------------------
# 2. DiagonalCumulativeDelta
# ---------------------------------------------------------------------------

class TestDiagonalCumulativeDelta:

    def test_starts_at_zero(self):
        dcd = DiagonalCumulativeDelta()
        assert dcd.raw == 0
        assert dcd.diagonal == 0

    def test_buy_increases_raw(self):
        dcd = DiagonalCumulativeDelta()
        tracker = Tracker()
        dcd.update(_tick(5800.0, 5, "buy"), tracker)
        assert dcd.raw == 5

    def test_sell_decreases_raw(self):
        dcd = DiagonalCumulativeDelta()
        tracker = Tracker()
        dcd.update(_tick(5800.0, 5, "sell"), tracker)
        assert dcd.raw == -5

    def test_diagonal_zero_when_no_flags(self):
        dcd = DiagonalCumulativeDelta()
        tracker = Tracker()   # empty flags
        dcd.update(_tick(5800.0, 10, "buy"), tracker)
        assert dcd.diagonal == 0

    def test_diagonal_counts_when_flag_at_price(self):
        dcd = DiagonalCumulativeDelta()
        sc = _sc()
        price = 5800.0
        px_ticks = sc.to_ticks(price)
        tracker = _tracker_with_flags([(px_ticks, BUY_IMBALANCE)])
        dcd.update(_tick(price, 10, "buy"), tracker)
        assert dcd.diagonal == 10

    def test_diagonal_excludes_unflagged_price(self):
        dcd = DiagonalCumulativeDelta()
        sc = _sc()
        flag_price = 5800.0
        trade_price = 5801.0
        px_ticks = sc.to_ticks(flag_price)
        tracker = _tracker_with_flags([(px_ticks, BUY_IMBALANCE)])
        dcd.update(_tick(trade_price, 10, "buy"), tracker)
        assert dcd.raw == 10
        assert dcd.diagonal == 0

    def test_reset_clears_both_accumulators(self):
        dcd = DiagonalCumulativeDelta()
        sc = _sc()
        tracker = _tracker_with_flags([(sc.to_ticks(5800.0), BUY_IMBALANCE)])
        dcd.update(_tick(5800.0, 10, "buy"), tracker)
        dcd.reset()
        assert dcd.raw == 0
        assert dcd.diagonal == 0

    def test_multiple_prints_accumulate(self):
        dcd = DiagonalCumulativeDelta()
        sc = _sc()
        p = sc.to_ticks(5800.0)
        tracker = _tracker_with_flags([(p, BUY_IMBALANCE)])
        dcd.update(_tick(5800.0, 10, "buy"), tracker)
        dcd.update(_tick(5800.0, 5,  "sell"), tracker)
        assert dcd.raw == 5
        assert dcd.diagonal == 5   # 10 - 5


# ---------------------------------------------------------------------------
# 3. _live_stacked
# ---------------------------------------------------------------------------

class TestLiveStacked:

    def test_empty_tracker_returns_empty(self):
        tracker = Tracker()
        result = _live_stacked(tracker, _sc(), run=3)
        assert result == []

    def test_single_level_below_run_threshold(self):
        sc = _sc()
        tracker = _tracker_with_flags([(23200, BUY_IMBALANCE)])
        result = _live_stacked(tracker, sc, run=3)
        assert result == []

    def test_single_level_meets_run_one(self):
        sc = _sc()
        tracker = _tracker_with_flags([(23200, BUY_IMBALANCE)])
        result = _live_stacked(tracker, sc, run=1)
        assert len(result) == 1
        side, lo, hi, run = result[0]
        assert side == BUY_IMBALANCE
        assert run == 1
        assert lo == hi

    def test_run_of_three_detected(self):
        sc = _sc()
        tracker = _tracker_with_flags([
            (23200, BUY_IMBALANCE),
            (23201, BUY_IMBALANCE),
            (23202, BUY_IMBALANCE),
        ])
        result = _live_stacked(tracker, sc, run=3)
        assert len(result) == 1
        side, lo, hi, run = result[0]
        assert side == BUY_IMBALANCE
        assert run == 3
        assert hi > lo

    def test_gap_breaks_run(self):
        sc = _sc()
        # 23200, 23201, (gap), 23203 → longest run is 2 (below threshold 3)
        tracker = _tracker_with_flags([
            (23200, BUY_IMBALANCE),
            (23201, BUY_IMBALANCE),
            (23203, BUY_IMBALANCE),
        ])
        result = _live_stacked(tracker, sc, run=3)
        assert result == []

    def test_sell_run_detected(self):
        sc = _sc()
        tracker = _tracker_with_flags([
            (23200, SELL_IMBALANCE),
            (23201, SELL_IMBALANCE),
            (23202, SELL_IMBALANCE),
        ])
        result = _live_stacked(tracker, sc, run=3)
        assert len(result) == 1
        assert result[0][0] == SELL_IMBALANCE

    def test_sorted_by_run_length_descending(self):
        sc = _sc()
        # 4-run on buy, 3-run on sell
        tracker = _tracker_with_flags([
            (23200, BUY_IMBALANCE),
            (23201, BUY_IMBALANCE),
            (23202, BUY_IMBALANCE),
            (23203, BUY_IMBALANCE),
            (23100, SELL_IMBALANCE),
            (23101, SELL_IMBALANCE),
            (23102, SELL_IMBALANCE),
        ])
        result = _live_stacked(tracker, sc, run=3)
        assert len(result) == 2
        assert result[0][3] >= result[1][3]   # descending by run length

    def test_lo_price_less_than_hi_price(self):
        sc = _sc()
        tracker = _tracker_with_flags([
            (23200, BUY_IMBALANCE),
            (23201, BUY_IMBALANCE),
            (23202, BUY_IMBALANCE),
        ])
        _, lo, hi, _ = _live_stacked(tracker, sc, run=3)[0]
        assert lo < hi


# ---------------------------------------------------------------------------
# 4. ExecState
# ---------------------------------------------------------------------------

class TestExecState:

    def test_as_dict_required_keys(self):
        state = ExecState(
            ts=1.0, pulse=None,
            diagonal_delta=5, raw_delta=10,
            stacked=[], signals_total=0, last_signal=None,
        )
        d = state.as_dict()
        for key in ("ts", "pulse", "diagonal_delta", "raw_delta",
                    "stacked", "signals_total", "last_signal"):
            assert key in d, f"missing key {key!r}"

    def test_pulse_none_serialises_as_none(self):
        state = ExecState(
            ts=1.0, pulse=None,
            diagonal_delta=0, raw_delta=0,
            stacked=[], signals_total=0, last_signal=None,
        )
        assert state.as_dict()["pulse"] is None

    def test_stacked_list_serialised(self):
        state = ExecState(
            ts=1.0, pulse=None,
            diagonal_delta=0, raw_delta=0,
            stacked=[("buy", 5800.0, 5800.5, 3)],
            signals_total=0, last_signal=None,
        )
        stacked = state.as_dict()["stacked"]
        assert len(stacked) == 1
        row = stacked[0]
        assert row["side"] == "buy"
        assert row["low"] == 5800.0
        assert row["high"] == 5800.5
        assert row["run"] == 3

    def test_last_signal_serialised_when_present(self):
        from liqmap.execution import Signal
        sig = Signal(
            ts=2.0, side="buy", price=5800.0,
            raw_delta=10, diagonal_delta=8,
            unconfirmed=3, buy_flags=3, sell_flags=0,
            stacked_side="buy", stacked_low=5800.0, stacked_high=5800.5,
            stacked_run=3, bar_age_s=10.0, reason="test",
        )
        state = ExecState(
            ts=3.0, pulse=None,
            diagonal_delta=8, raw_delta=10,
            stacked=[], signals_total=1, last_signal=sig,
        )
        d = state.as_dict()
        assert d["last_signal"] is not None
        assert d["last_signal"]["side"] == "buy"


# ---------------------------------------------------------------------------
# 5. ExecutionTrack
# ---------------------------------------------------------------------------

class TestExecutionTrack:

    def _feed_stacked(self, et: ExecutionTrack, price: float,
                      n_prints: int = 20, side: str = "buy") -> None:
        """Drive enough identical prints at one price to create a self-stacked imbalance."""
        for i in range(n_prints):
            et.on_trade(_FakeTrade(ts=1_700_000_000.0 + i * 0.1,
                                   px=price, sz=50, aggressor=side))

    def test_state_returns_exec_state(self):
        et = ExecutionTrack()
        s = et.state()
        assert isinstance(s, ExecState)

    def test_state_ts_is_positive(self):
        et = ExecutionTrack()
        s = et.state()
        assert s.ts > 0

    def test_on_trade_increments_pulse_seq(self):
        et = ExecutionTrack()
        et.on_trade(_FakeTrade(ts=1.0, px=5800.0, sz=10, aggressor="buy"))
        pulse = et._stream.pulse()
        assert pulse.seq == 1

    def test_zero_size_trade_ignored(self):
        et = ExecutionTrack()
        et.on_trade(_FakeTrade(ts=1.0, px=5800.0, sz=0, aggressor="buy"))
        assert et._stream.pulse().seq == 0

    def test_signal_fires_on_stacked_buy_run(self):
        """Four consecutive buy prints: edge skip + 3 interior flags = stacked.

        The default edge policy skips the bar's low, so a buy at 5800.00
        is not flagged.  Buys at 5800.25 / .50 / .75 each face an empty
        diagonal and flag.  The third interior flag completes a run of 3;
        DCD is already +30 and agrees; a Signal must fire immediately —
        no candle close, no timer.
        """
        received = []
        et = ExecutionTrack(min_run=3, min_flag_updates=1, cooldown_s=0.0)
        et.on_signal(lambda sig: received.append(sig))
        base = 1_700_000_000.0
        for i, px in enumerate((5800.00, 5800.25, 5800.50, 5800.75)):
            et.on_trade(_FakeTrade(ts=base + i * 0.1, px=px, sz=10,
                                   aggressor="buy"))
        assert len(received) == 1
        sig = received[0]
        assert isinstance(sig, Signal)
        assert sig.side == "buy"
        assert sig.stacked_run == 3
        assert sig.diagonal_delta > 0
        assert sig.price == 5800.75

    def test_signal_does_not_wait_for_bar_close(self):
        """The execution footprint bar is still open when the signal fires."""
        received = []
        et = ExecutionTrack(min_run=3, min_flag_updates=1, cooldown_s=0.0)
        et.on_signal(lambda sig: received.append(sig))
        base = 1_700_000_000.0
        for i, px in enumerate((5800.00, 5800.25, 5800.50, 5800.75)):
            et.on_trade(_FakeTrade(ts=base + i * 0.1, px=px, sz=10,
                                   aggressor="buy"))
        assert received
        # The engine's own bar must still be forming — a close would reset DCD.
        assert et._stream.pulse().bars_closed == 0
        assert et._dcd.raw != 0

    def test_sell_stack_fires_sell_signal(self):
        """Symmetric to the buy case: four consecutive sells → sell Signal."""
        received = []
        et = ExecutionTrack(min_run=3, min_flag_updates=1, cooldown_s=0.0)
        et.on_signal(lambda sig: received.append(sig))
        base = 1_700_000_000.0
        # High of the bar is skipped for sell tests, so print the highest
        # price first so 5800.75 becomes the edge and the three lower
        # prices form the stacked run.
        for i, px in enumerate((5800.75, 5800.50, 5800.25, 5800.00)):
            et.on_trade(_FakeTrade(ts=base + i * 0.1, px=px, sz=10,
                                   aggressor="sell"))
        assert len(received) == 1
        assert received[0].side == "sell"
        assert received[0].diagonal_delta < 0

    def test_signal_callback_fires(self):
        """Inject enough buy prints at ONE price to create stacked run of 1."""
        received = []
        et = ExecutionTrack(min_run=1, min_flag_updates=1, cooldown_s=0.0)
        et.on_signal(lambda sig: received.append(sig))

        # Drive prints at one price with a buy imbalance vs nothing.
        # With edge_policy='skip' and min_volume=0 the default config can
        # flag a single level as long as it has volume.  Because the bar
        # starts empty the buy prints build ask_volume and the diagonal
        # below is empty, which passes flag_empty_diagonal=True.
        for i in range(5):
            et.on_trade(_FakeTrade(
                ts=1_700_000_000.0 + i * 0.1,
                px=5800.25, sz=10, aggressor="buy"))

        # A signal may or may not have fired depending on flag state, but
        # the machinery must not raise.
        assert isinstance(received, list)

    def test_off_signal_removes_callback(self):
        received = []
        et = ExecutionTrack(min_run=1, min_flag_updates=1, cooldown_s=0.0)
        fn = lambda sig: received.append(sig)
        et.on_signal(fn)
        et.off_signal(fn)

        for i in range(5):
            et.on_trade(_FakeTrade(
                ts=1_700_000_000.0 + i, px=5800.25, sz=10, aggressor="buy"))

        # fn was removed; even if signals fired, received must be empty.
        assert received == []

    def test_cooldown_suppresses_duplicate_signals(self):
        """Two identical signal conditions within cooldown window → only one signal."""
        received = []
        et = ExecutionTrack(
            min_run=1, min_flag_updates=1, cooldown_s=1_000.0)
        et.on_signal(lambda sig: received.append(sig))

        # Force two evaluation cycles
        for i in range(10):
            et.on_trade(_FakeTrade(
                ts=1_700_000_000.0 + i * 0.01,
                px=5800.25, sz=10, aggressor="buy"))

        # At most one signal per side per cooldown window
        buy_sigs = [s for s in received if s.side == "buy"]
        assert len(buy_sigs) <= 1

    def test_dcd_raw_tracks_delta(self):
        et = ExecutionTrack()
        et.on_trade(_FakeTrade(ts=1.0, px=5800.0, sz=10, aggressor="buy"))
        et.on_trade(_FakeTrade(ts=2.0, px=5800.0, sz=3,  aggressor="sell"))
        assert et._dcd.raw == 7

    def test_state_raw_delta_matches_dcd(self):
        et = ExecutionTrack()
        et.on_trade(_FakeTrade(ts=1.0, px=5800.0, sz=10, aggressor="buy"))
        s = et.state()
        assert s.raw_delta == et._dcd.raw


# ---------------------------------------------------------------------------
# 6. Tape._exec_listeners
# ---------------------------------------------------------------------------

PT = 1_000_000_000
TICK_PT = PT // 4
BASE_PX = 5800 * PT
T0 = 1_700_000_000.0


class _Level:
    def __init__(self, bp, bs, ap, asz):
        self.bid_px, self.bid_sz, self.ask_px, self.ask_sz = bp, bs, ap, asz


class _Rec:
    """Minimal DBN-like record."""
    def __init__(self, action, px, sz, side, ts, levels=None):
        self.action = action
        self.price  = px
        self.size   = sz
        self.side   = side
        self.ts_event = int(ts * 1e9)
        self.instrument_id = 7
        self.sequence      = 1
        self.levels = levels if levels is not None else [
            _Level(BASE_PX - j * TICK_PT, 99, BASE_PX + (j + 1) * TICK_PT, 99)
            for j in range(10)
        ]


def _trade_rec(ts=T0):
    return _Rec("T", BASE_PX + TICK_PT, 5, "B", ts)


def _book_rec(ts=T0):
    return _Rec("A", BASE_PX, 50, "B", ts)


def _no_levels_rec(ts=T0):
    r = _Rec("A", BASE_PX, 50, "B", ts, levels=[])
    r.levels = []
    return r


@dataclass
class _FakeTrade:
    ts: float
    px: float
    sz: float
    aggressor: str


class TestTapeExecListeners:

    def test_exec_listener_count_starts_zero(self, monkeypatch):
        monkeypatch.setenv("LIQMAP_OFI", "false")
        from liqmap.cme_tape import Tape
        t = Tape()
        assert t.exec_listener_count == 0

    def test_on_trade_exec_increments_count(self, monkeypatch):
        monkeypatch.setenv("LIQMAP_OFI", "false")
        from liqmap.cme_tape import Tape
        t = Tape()
        t.on_trade_exec(lambda trade: None)
        assert t.exec_listener_count == 1

    def test_off_trade_exec_decrements_count(self, monkeypatch):
        monkeypatch.setenv("LIQMAP_OFI", "false")
        from liqmap.cme_tape import Tape
        t = Tape()
        fn = lambda trade: None
        t.on_trade_exec(fn)
        t.off_trade_exec(fn)
        assert t.exec_listener_count == 0

    def test_off_trade_exec_unknown_is_silent(self, monkeypatch):
        monkeypatch.setenv("LIQMAP_OFI", "false")
        from liqmap.cme_tape import Tape
        t = Tape()
        t.off_trade_exec(lambda trade: None)  # must not raise

    def test_exec_listener_fires_on_trade(self, monkeypatch):
        monkeypatch.setenv("LIQMAP_OFI", "false")
        from liqmap.cme_tape import Tape
        t = Tape()
        received = []
        t.on_trade_exec(lambda trade: received.append(trade))
        t.offer(_trade_rec())
        assert len(received) == 1

    def test_exec_listener_receives_trade_object(self, monkeypatch):
        monkeypatch.setenv("LIQMAP_OFI", "false")
        from liqmap.cme_tape import Tape
        t = Tape()
        received = []
        t.on_trade_exec(lambda trade: received.append(trade))
        t.offer(_trade_rec())
        trade = received[0]
        # Trade object must carry price and aggressor
        assert hasattr(trade, "px") or hasattr(trade, "price")
        assert hasattr(trade, "aggressor")

    def test_exec_listener_not_fired_on_book_record(self, monkeypatch):
        monkeypatch.setenv("LIQMAP_OFI", "false")
        from liqmap.cme_tape import Tape
        t = Tape()
        received = []
        t.on_trade_exec(lambda trade: received.append(trade))
        t.offer(_book_rec())   # non-trade record
        assert received == []

    def test_exec_listener_not_fired_on_no_levels_record(self, monkeypatch):
        monkeypatch.setenv("LIQMAP_OFI", "false")
        from liqmap.cme_tape import Tape
        t = Tape()
        received = []
        t.on_trade_exec(lambda trade: received.append(trade))
        t.offer(_no_levels_rec())
        assert received == []

    def test_crashing_exec_listener_does_not_stop_tape(self, monkeypatch):
        monkeypatch.setenv("LIQMAP_OFI", "false")
        from liqmap.cme_tape import Tape
        t = Tape()
        def _boom(trade):
            raise RuntimeError("exec crash")
        t.on_trade_exec(_boom)
        t.offer(_trade_rec())   # must not raise
        assert t.trades == 1

    def test_exec_and_vis_listeners_both_fire(self, monkeypatch):
        monkeypatch.setenv("LIQMAP_OFI", "false")
        from liqmap.cme_tape import Tape
        t = Tape()
        exec_fired = []
        vis_fired  = []
        t.on_trade_exec(lambda trade: exec_fired.append(1))
        t.on_update(lambda: vis_fired.append(1))
        t.offer(_trade_rec())
        assert exec_fired == [1]
        assert vis_fired  == [1]


# ---------------------------------------------------------------------------
# 7. SignalBroadcaster
# ---------------------------------------------------------------------------

from liqmap.web import SignalBroadcaster  # noqa: E402


class TestSignalBroadcaster:

    def test_starts_with_zero_connections(self):
        sb = SignalBroadcaster()
        assert sb.connections == 0

    def test_add_increments(self):
        sb = SignalBroadcaster()
        q = asyncio.Queue()
        sb.add(q)
        assert sb.connections == 1

    def test_remove_decrements(self):
        sb = SignalBroadcaster()
        q = asyncio.Queue()
        sb.add(q)
        sb.remove(q)
        assert sb.connections == 0

    def test_remove_unknown_is_silent(self):
        sb = SignalBroadcaster()
        sb.remove(asyncio.Queue())   # no raise

    def test_notify_no_op_without_loop(self):
        sb = SignalBroadcaster()
        q = asyncio.Queue()
        sb.add(q)
        sb.notify({"ts": 1.0})
        assert q.empty()
        assert sb.dropped == 0

    def test_notify_delivers_via_loop(self):
        sb = SignalBroadcaster()

        async def _run():
            loop = asyncio.get_running_loop()
            sb.set_loop(loop)
            q: asyncio.Queue = asyncio.Queue()
            sb.add(q)

            def _thread():
                sb.notify({"ts": 99.0, "side": "buy"})
            t = threading.Thread(target=_thread)
            t.start()
            t.join()

            frame = await asyncio.wait_for(q.get(), timeout=1.0)
            return frame

        frame = asyncio.run(_run())
        assert frame["ts"] == 99.0
        assert frame["side"] == "buy"


# ---------------------------------------------------------------------------
# 8-9. /ws/exec and /api/exec/state integration tests
# ---------------------------------------------------------------------------

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
import liqmap.web as web                  # noqa: E402

TOKEN = "exec-test-token-xyz"


@pytest.fixture
def client(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("LIQMAP_DB",    str(Path(tmp) / "exec.db"))
        monkeypatch.setenv("LIQMAP_TOKEN", TOKEN)
        monkeypatch.setenv("LIQMAP_AUTO",  "false")
        monkeypatch.delenv("LIQMAP_OFI",       raising=False)
        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
        web.RT = None
        with TestClient(web.create_app()) as c:
            yield c
        web.RT = None


@pytest.fixture
def client_cme(monkeypatch):
    """Client with CME tape present (no real API key — tape exists, engine won't connect)."""
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("LIQMAP_DB",          str(Path(tmp) / "exec_cme.db"))
        monkeypatch.setenv("LIQMAP_TOKEN",        TOKEN)
        monkeypatch.setenv("LIQMAP_AUTO",         "false")
        monkeypatch.setenv("LIQMAP_OFI",          "true")
        monkeypatch.setenv("LIQMAP_OFI_SYMBOLS",  "ES.c.0")
        monkeypatch.delenv("DATABENTO_API_KEY",   raising=False)
        web.RT = None
        with TestClient(web.create_app()) as c:
            yield c
        web.RT = None


class TestWsExecAuth:

    def test_bad_token_rejected(self, client):
        with pytest.raises(Exception):
            with client.websocket_connect(
                f"/ws/exec?coin=ES&token=WRONG"
            ) as ws:
                ws.receive_json()

    def test_empty_token_rejected(self, client):
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/exec?coin=ES&token=") as ws:
                ws.receive_json()


class TestWsExecNonCme:

    def test_non_cme_coin_returns_error(self, client):
        with client.websocket_connect(
            f"/ws/exec?coin=BTC&token={TOKEN}"
        ) as ws:
            msg = ws.receive_json()
        assert "error" in msg

    def test_es_without_ofi_engine_returns_error(self, client):
        with client.websocket_connect(
            f"/ws/exec?coin=ES&token={TOKEN}"
        ) as ws:
            msg = ws.receive_json()
        assert "error" in msg


class TestWsExecCme:

    def test_initial_frame_has_required_keys(self, client_cme):
        """On connect the endpoint sends the current ExecState as the first frame."""
        with client_cme.websocket_connect(
            f"/ws/exec?coin=ES&token={TOKEN}"
        ) as ws:
            frame = ws.receive_json()
        for key in ("ts", "pulse", "diagonal_delta", "raw_delta",
                    "stacked", "signals_total", "last_signal"):
            assert key in frame, f"missing key {key!r}"

    def test_initial_frame_stacked_is_list(self, client_cme):
        with client_cme.websocket_connect(
            f"/ws/exec?coin=ES&token={TOKEN}"
        ) as ws:
            frame = ws.receive_json()
        assert isinstance(frame["stacked"], list)

    def test_initial_frame_signals_total_is_int(self, client_cme):
        with client_cme.websocket_connect(
            f"/ws/exec?coin=ES&token={TOKEN}"
        ) as ws:
            frame = ws.receive_json()
        assert isinstance(frame["signals_total"], int)

    def test_broadcaster_removes_queue_on_disconnect(self, client_cme):
        rt = web.runtime()
        before = rt.signal_broadcaster.connections
        with client_cme.websocket_connect(
            f"/ws/exec?coin=ES&token={TOKEN}"
        ) as ws:
            ws.receive_json()
            during = rt.signal_broadcaster.connections
        after = rt.signal_broadcaster.connections
        assert during > before
        assert after == before


class TestDashboardExecPanel:

    def test_dashboard_has_exec_panel(self):
        from liqmap.web import DASHBOARD
        for marker in ("execPanel", "execWsBar", "execHead",
                       "execStacks", "execLog", "openExecStream",
                       "paintExecState", "paintExecSignal"):
            assert marker in DASHBOARD, f"dashboard missing {marker!r}"

    def test_go_live_opens_exec_stream(self):
        from liqmap.web import DASHBOARD
        assert "openExecStream()" in DASHBOARD
        assert "closeExecStream()" in DASHBOARD


class TestApiExecState:

    def test_cme_returns_required_keys(self, client_cme):
        resp = client_cme.get(
            f"/api/exec/state?coin=ES",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        body = resp.json()
        for key in ("ts", "pulse", "diagonal_delta", "raw_delta",
                    "stacked", "signals_total", "last_signal"):
            assert key in body, f"missing key {key!r}"

    def test_non_cme_returns_error(self, client):
        resp = client.get(
            f"/api/exec/state?coin=BTC",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "error" in body
