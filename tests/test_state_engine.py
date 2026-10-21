"""Deterministic State Engine — VVR, obstacle ratio, positional logs."""

from __future__ import annotations

import gc
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap.state_engine import (  # noqa: E402
    ACT_ENTER_LONG, ACT_ENTER_SHORT, ACT_EXIT_LONG, ACT_REVERSE_SHORT,
    CANDLE_N, FLAT, LONG, SHORT, L_ALL, L_STACK, L_VVR, L_OBST,
    OBSTACLE_RATIO, STATUS_ABSORB, STATUS_DRIVE, Stance,
    AnsiRouter, MedianRing, StateEngine, Transition, VVR_MULT,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

@dataclass
class T:
    ts: float
    px: float
    sz: float
    aggressor: str


@dataclass
class Lv:
    px: float
    sz: float
    n: int = 1


@dataclass
class Bk:
    bids: list = field(default_factory=list)
    asks: list = field(default_factory=list)


def _thin_ask(px: float = 5801.00, sz: float = 20.0) -> Bk:
    """Ask wall small enough that a 138-lot stack ratio is > 1.2."""
    return Bk(
        bids=[Lv(5799.75, 10.0)],
        asks=[Lv(px, sz), Lv(px + 0.25, 8.0)],
    )


def _thin_bid(px: float = 5799.00, sz: float = 20.0) -> Bk:
    return Bk(
        bids=[Lv(px, sz), Lv(px - 0.25, 8.0)],
        asks=[Lv(5800.25, 10.0)],
    )


def _thick_ask(px: float = 5801.00, sz: float = 400.0) -> Bk:
    """Nearby ask that a 138-lot burst cannot overwhelm (138/400 < 1.2)."""
    return Bk(
        bids=[Lv(5799.75, 10.0)],
        asks=[Lv(px, sz)],
    )


def _thick_bid(px: float = 5799.00, sz: float = 400.0) -> Bk:
    return Bk(
        bids=[Lv(px, sz)],
        asks=[Lv(5800.25, 10.0)],
    )


def _silent_engine(**kw) -> StateEngine:
    return StateEngine(router=AnsiRouter(enabled=False), **kw)


def _seed(eng: StateEngine, vol: float = 20.0, n: int = CANDLE_N) -> None:
    eng.seed_candles(vol, n)


def _drive(eng: StateEngine, t0: float, book: Bk,
           side: str = "buy", base: float = 5800.00,
           sizes: tuple = (40.0, 46.0, 52.0)) -> bool:
    """Three consecutive same-sign levels inside a 100 ms window."""
    prices = (base, base + 0.25, base + 0.50)
    running = 0.0
    fired = False
    for i, (px, sz) in enumerate(zip(prices, sizes)):
        if side == "buy":
            running += sz
        else:
            running -= sz
        fired = eng.on_tick(
            T(t0 + i * 0.01, px, sz, side), dcd=running, book=book
        ) or fired
    return fired


def _go_long(eng: StateEngine, t0: float = 1_700_000_000.0,
             base: float = 5800.00) -> bool:
    _seed(eng)
    return _drive(eng, t0, _thin_ask(), "buy", base)


def _go_short(eng: StateEngine, t0: float = 1_700_000_000.0,
              base: float = 5800.00) -> bool:
    _seed(eng)
    return _drive(eng, t0, _thin_bid(), "sell", base)


# ---------------------------------------------------------------------------
# Stance API
# ---------------------------------------------------------------------------

class TestStance:

    def test_names(self):
        assert Stance.FLAT == 0
        assert Stance.LONG == 1
        assert Stance.SHORT == 2

    def test_starts_flat(self):
        assert _silent_engine().stance == Stance.FLAT


# ---------------------------------------------------------------------------
# Median ring
# ---------------------------------------------------------------------------

class TestMedianRing:

    def test_empty_is_zero(self):
        assert MedianRing().median() == 0.0

    def test_median_of_twenty(self):
        r = MedianRing(20)
        for _ in range(20):
            r.push(20.0)
        assert r.median() == 20.0

    def test_median_odd_and_even(self):
        r = MedianRing(5)
        for v in (1.0, 5.0, 3.0):
            r.push(v)
        assert r.median() == 3.0
        r.push(9.0)
        assert r.median() == 4.0          # (3+5)/2


# ---------------------------------------------------------------------------
# Layers in isolation
# ---------------------------------------------------------------------------

class TestLayers:

    def test_no_book_never_enters(self):
        eng = _silent_engine()
        _seed(eng)
        fired = _drive(eng, 10.0, Bk(), "buy")
        assert fired is False
        assert eng.stance == FLAT

    def test_stack_alone_does_not_transition(self):
        """Raw level counts without VVR (no candle median) stay FLAT."""
        eng = _silent_engine()
        book = _thin_ask()
        _drive(eng, 10.0, book, "buy")
        assert eng.stance == FLAT
        assert eng._stack_run >= 3
        assert eng._stack_side > 0

    def test_unseeded_median_suppresses_as_noise(self):
        eng = _silent_engine()
        fired = _drive(eng, 10.0, _thin_ask(), "buy")
        assert fired is False
        assert eng.stance == FLAT
        assert eng._status == "Micro Noise"
        assert eng._median_vol == 0.0

    def test_retail_noise_below_vvr_multiple(self):
        """100 ms stack volume must clear 1.5× the 20-candle median."""
        eng = _silent_engine()
        _seed(eng, vol=200.0)             # 1.5× = 300; stack is 138
        fired = _drive(eng, 10.0, _thin_ask(), "buy")
        assert fired is False
        assert eng.stance == FLAT
        assert eng._vvr_vol_long == pytest.approx(138.0)
        assert eng._vvr_vol_long < VVR_MULT * 200.0
        assert eng._status == "Micro Noise"

    def test_microsecond_absorption_keeps_flat(self):
        """Aggressive volume that cannot take the nearby ask stays FLAT."""
        eng = _silent_engine()
        _seed(eng)
        fired = _drive(eng, 10.0, _thick_ask(), "buy")
        assert fired is False
        assert eng.stance == FLAT
        assert eng._stack_run >= 3
        assert eng._vvr_vol_long >= VVR_MULT * 20.0
        assert eng._pos_weight <= OBSTACLE_RATIO
        assert eng._status == STATUS_ABSORB

    def test_buy_requires_obstacle_ratio(self):
        eng = _silent_engine()
        _seed(eng)
        assert _drive(eng, 10.0, _thin_ask(sz=20.0), "buy") is True
        # 138 / 28 (20+8) ≈ 4.93
        assert eng.last.pos_weight == pytest.approx(138.0 / 28.0)
        assert eng.last.pos_weight > OBSTACLE_RATIO


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------

class TestTransitions:

    def test_flat_to_long(self):
        eng = _silent_engine()
        fired = _go_long(eng)
        assert fired is True
        assert eng.stance == LONG
        assert eng.last.action == ACT_ENTER_LONG
        assert eng.last.frm == FLAT
        assert eng.last.to == LONG
        assert eng.last.layers == L_ALL
        assert eng.last.status == STATUS_DRIVE
        assert eng.last.tgt_px == 5801.00
        assert eng.last.tgt_side < 0          # ask
        assert eng.last.ts_ms > 0
        assert eng.last.pos_weight > OBSTACLE_RATIO

    def test_flat_to_short(self):
        eng = _silent_engine()
        fired = _go_short(eng)
        assert fired is True
        assert eng.stance == SHORT
        assert eng.last.action == ACT_ENTER_SHORT
        assert eng.last.tgt_side > 0          # bid
        assert eng.last.pos_weight > OBSTACLE_RATIO

    def test_long_to_reverse_short(self):
        eng = _silent_engine()
        assert _go_long(eng)
        t1 = 1_700_000_000.0 + 2.0
        fired = _drive(eng, t1, _thin_bid(px=5808.00), "sell", base=5810.00)
        assert fired is True
        assert eng.stance == SHORT
        assert eng.last.action == ACT_REVERSE_SHORT
        assert eng.last.frm == LONG
        assert eng.last.to == SHORT

    def test_no_repeat_without_change(self):
        eng = _silent_engine()
        assert _go_long(eng)
        seq = eng.seq
        book = _thin_ask()
        fired = _drive(eng, 1_700_000_000.0 + 0.40, book, "buy", 5800.00)
        assert fired is False
        assert eng.seq == seq
        assert eng.stance == LONG

    def test_exit_long_on_opposing_stack_and_vvr(self):
        """Opposite stack+VVR against a thick bid wall → EXIT, not reverse."""
        eng = _silent_engine()
        assert _go_long(eng)
        t1 = 1_700_000_000.0 + 3.0
        # Bids within 4 pts of 5810 that a 138-lot sell cannot overwhelm.
        book = Bk(
            bids=[Lv(5808.00, 400.0)],
            asks=[Lv(5812.00, 20.0)],
        )
        fired = _drive(eng, t1, book, "sell", 5810.00)
        assert fired is True
        assert eng.last.action == ACT_EXIT_LONG
        assert eng.stance == FLAT
        assert (eng.last.layers & (L_STACK | L_VVR)) == (L_STACK | L_VVR)
        assert (eng.last.layers & L_OBST) == 0


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class TestAnsiRouter:

    def test_writes_positional_line(self):
        chunks: list[bytes] = []
        router = AnsiRouter(enabled=True, write=chunks.append)
        eng = StateEngine(router=router)
        _go_long(eng)
        assert chunks
        line = chunks[-1]
        assert b"\xe2\x96\xb6" in line          # ▶
        assert b"LONG" in line
        assert b"POS_WEIGHT:" in line
        assert b"TARGET:" in line
        assert b"5801.00" in line
        assert b"STATUS: Positional Drive" in line
        assert b"\x1b[32m" in line              # green LONG
        assert b"ENTER_LONG" not in line        # old layout is gone
        assert b"sigma" not in line
        assert b"absorb" not in line

    def test_disabled_writes_nothing(self):
        chunks: list[bytes] = []
        router = AnsiRouter(enabled=False, write=chunks.append)
        eng = StateEngine(router=router)
        _go_long(eng)
        assert chunks == []


# ---------------------------------------------------------------------------
# Zero-allocation hot path
# ---------------------------------------------------------------------------

class TestHotPath:

    def test_slots_on_hot_types(self):
        assert "stance" in StateEngine.__slots__
        assert "ts_ms" in Transition.__slots__
        assert "pos_weight" in Transition.__slots__

    def test_no_transition_does_not_grow_heap(self):
        eng = _silent_engine()
        _seed(eng, vol=200.0)                 # VVR will reject 12-lot prints
        book = _thin_ask()
        for i in range(80):
            eng.on_tick(T(1.0 + i * 0.01, 5800.0, 12.0 + (i % 5), "buy"),
                        dcd=12.0, book=book)
        gc.collect()
        before = len(gc.get_objects())
        for i in range(400):
            eng.on_tick(T(10.0 + i * 0.01, 5800.0, 12.0 + (i % 5), "buy"),
                        dcd=12.0, book=book)
        gc.collect()
        after = len(gc.get_objects())
        assert after - before < 80

    def test_reused_frame_is_same_dict(self):
        eng = _silent_engine()
        _go_long(eng)
        a = eng.frame()
        b = eng.frame()
        assert a is b
        assert a["kind"] == "stance"
        assert a["action"] == "ENTER_LONG"
        assert a["status"] == STATUS_DRIVE
        assert a["pos_weight"] > OBSTACLE_RATIO


# ---------------------------------------------------------------------------
# DeltaWorker integration
# ---------------------------------------------------------------------------

class TestWorkerHook:

    def test_worker_fires_stance_on_hub(self):
        import asyncio
        from liqmap.execution import ExecutionTrack
        from liqmap.pipeline import DeltaWorker, IngestHub

        hub = IngestHub()
        et = ExecutionTrack(min_run=3, min_flag_updates=1, cooldown_s=0.0)
        eng = _silent_engine()
        _seed(eng)
        w = DeltaWorker(hub, et, engine=eng, tape=None)

        async def _run():
            hub.set_loop(asyncio.get_running_loop())
            q: asyncio.Queue = asyncio.Queue()
            hub.add(q)
            w.start()

            class _Tape:
                pass
            tape = _Tape()
            tape.book = _thin_ask()
            w.tape = tape

            t0 = 1_700_000_000.0
            prices = (5800.00, 5800.25, 5800.50)
            sizes = (40.0, 46.0, 52.0)
            for i, (px, sz) in enumerate(zip(prices, sizes)):
                w.push_trade(T(t0 + i * 0.01, px, sz, "buy"))

            deadline = time.time() + 3.0
            stance = None
            while time.time() < deadline:
                try:
                    frame = await asyncio.wait_for(q.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                if frame.get("kind") == "stance":
                    stance = frame
                    break
            w.stop()
            return stance

        frame = asyncio.run(_run())
        assert frame is not None
        assert frame["action"] == "ENTER_LONG"
        assert frame["to"] == "LONG"
        assert frame["status"] == STATUS_DRIVE
        assert eng.stance == LONG
