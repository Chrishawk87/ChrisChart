"""Deterministic State Engine — transitions, layers, hot-path discipline."""

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
    FLAT, LONG, SHORT, L_ALL, Stance,
    AnsiRouter, StateEngine, Transition,
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


def _ask_book(px: float = 5802.00, sz: float = 400.0) -> Bk:
    return Bk(
        bids=[Lv(5799.75, 10.0)],
        asks=[Lv(px, sz), Lv(px + 0.25, 8.0)],
    )


def _bid_book(px: float = 5798.00, sz: float = 400.0) -> Bk:
    return Bk(
        bids=[Lv(px, sz), Lv(px - 0.25, 8.0)],
        asks=[Lv(5800.25, 10.0)],
    )


def _silent_engine(**kw) -> StateEngine:
    return StateEngine(router=AnsiRouter(enabled=False), **kw)


def _build_stack(eng: StateEngine, t0: float, book: Bk,
                 side: str = "buy", base: float = 5800.00) -> None:
    """Three consecutive levels with varying size so σ > 0."""
    prices = (base, base + 0.25, base + 0.50)
    sizes = (40.0, 46.0, 52.0)
    for i, (px, sz) in enumerate(zip(prices, sizes)):
        eng.on_tick(T(t0 + i * 0.01, px, sz, side), dcd=sz, book=book)


def _absorb_sells(eng: StateEngine, t0: float, book: Bk,
                  px: float = 5799.50, n: int = 8, sz: float = 22.0) -> bool:
    """Sells off the stacked prices, same px → zero extension."""
    fired = False
    for i in range(n):
        fired = eng.on_tick(
            T(t0 + i * 0.002, px, sz, "sell"), dcd=-sz * (i + 1), book=book
        ) or fired
    return fired


def _absorb_buys(eng: StateEngine, t0: float, book: Bk,
                 px: float = 5801.00, n: int = 8, sz: float = 22.0) -> bool:
    fired = False
    for i in range(n):
        fired = eng.on_tick(
            T(t0 + i * 0.002, px, sz, "buy"), dcd=sz * (i + 1), book=book
        ) or fired
    return fired


def _go_long(eng: StateEngine, t0: float = 1_700_000_000.0,
             base: float = 5800.00) -> bool:
    book = _ask_book()
    _build_stack(eng, t0, book, "buy", base)
    return _absorb_sells(eng, t0 + 0.30, book, base - 0.50)


def _go_short(eng: StateEngine, t0: float = 1_700_000_000.0,
              base: float = 5800.00) -> bool:
    book = _bid_book()
    _build_stack(eng, t0, book, "sell", base)
    return _absorb_buys(eng, t0 + 0.30, book, base + 1.00)


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
# Layers in isolation
# ---------------------------------------------------------------------------

class TestLayers:

    def test_no_book_never_enters(self):
        eng = _silent_engine()
        # Stack + absorb without a book → liquidity bit off → no vote.
        _build_stack(eng, 10.0, Bk(), "buy")
        fired = _absorb_sells(eng, 10.30, Bk())
        assert fired is False
        assert eng.stance == FLAT

    def test_stack_alone_does_not_transition(self):
        eng = _silent_engine()
        book = _ask_book()
        _build_stack(eng, 10.0, book, "buy")
        assert eng.stance == FLAT
        assert eng._stack_run >= 3
        assert eng._stack_side > 0

    def test_sigma_is_dynamic_not_hardcoded(self):
        eng = _silent_engine()
        book = _ask_book()
        for i in range(20):
            eng.on_tick(T(1.0 + i * 0.01, 5800.0, 10.0 + i, "buy"),
                        dcd=10.0, book=book)
        s1 = eng._sigma_now
        for i in range(20):
            eng.on_tick(T(2.0 + i * 0.01, 5800.0, 80.0 + i, "buy"),
                        dcd=80.0, book=book)
        s2 = eng._sigma_now
        assert s2 > s1 > 0.0


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
        assert eng.last.tgt_px == 5802.00
        assert eng.last.tgt_side < 0          # ask
        assert eng.last.ts_ms > 0

    def test_flat_to_short(self):
        eng = _silent_engine()
        fired = _go_short(eng)
        assert fired is True
        assert eng.stance == SHORT
        assert eng.last.action == ACT_ENTER_SHORT
        assert eng.last.tgt_side > 0          # bid

    def test_long_to_reverse_short(self):
        eng = _silent_engine()
        assert _go_long(eng)
        # Opposite structure on a different price band, after the
        # absorb window rolls.  Bid book makes liquidity vote SHORT
        # so this is a reverse, not an exit.
        t1 = 1_700_000_000.0 + 2.0
        fired = _go_short(eng, t1, base=5810.00)
        assert fired is True
        assert eng.stance == SHORT
        assert eng.last.action == ACT_REVERSE_SHORT
        assert eng.last.frm == LONG
        assert eng.last.to == SHORT

    def test_no_repeat_without_change(self):
        eng = _silent_engine()
        assert _go_long(eng)
        seq = eng.seq
        book = _ask_book()
        # More of the same structure must not re-emit ENTER_LONG.
        fired = _absorb_sells(eng, 1_700_000_000.0 + 0.40, book)
        assert fired is False
        assert eng.seq == seq
        assert eng.stance == LONG

    def test_exit_long_on_opposing_stack_and_absorb(self):
        """Opposite stack+absorb without a bid target → EXIT, not reverse."""
        eng = _silent_engine()
        assert _go_long(eng)
        # Keep the ASK book (long liquidity). Build a sell stack and
        # buy-absorption — stack+absorb oppose LONG but liq still long,
        # so vote != SHORT and EXIT_LONG fires.
        t1 = 1_700_000_000.0 + 3.0
        book = _ask_book()
        _build_stack(eng, t1, book, "sell", 5810.00)
        fired = _absorb_buys(eng, t1 + 0.30, book, 5811.00)
        assert fired is True
        assert eng.last.action == ACT_EXIT_LONG
        assert eng.stance == FLAT


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class TestAnsiRouter:

    def test_writes_colour_and_payload(self):
        chunks: list[bytes] = []
        router = AnsiRouter(enabled=True, write=chunks.append)
        eng = StateEngine(router=router)
        _go_long(eng)
        assert chunks
        line = chunks[-1]
        assert b"ENTER_LONG" in line
        assert b"FLAT" in line
        assert b"LONG" in line
        assert b"\x1b[32m" in line          # green LONG
        assert b"px=" in line
        assert b"tgt=" in line
        assert b"5802.00" in line

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

    def test_no_transition_does_not_grow_heap(self):
        eng = _silent_engine()
        book = _ask_book()
        # Warm rings / maps.
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
        # Slack for interpreter noise; a leaking tick would add hundreds.
        assert after - before < 80

    def test_reused_frame_is_same_dict(self):
        eng = _silent_engine()
        _go_long(eng)
        a = eng.frame()
        b = eng.frame()
        assert a is b
        assert a["kind"] == "stance"
        assert a["action"] == "ENTER_LONG"


# ---------------------------------------------------------------------------
# DeltaWorker integration
# ---------------------------------------------------------------------------

class TestWorkerHook:

    def test_worker_fires_stance_on_hub(self):
        import asyncio
        import threading
        from liqmap.execution import ExecutionTrack
        from liqmap.pipeline import DeltaWorker, IngestHub

        hub = IngestHub()
        et = ExecutionTrack(min_run=3, min_flag_updates=1, cooldown_s=0.0)
        eng = _silent_engine()
        w = DeltaWorker(hub, et, engine=eng, tape=None)

        async def _run():
            hub.set_loop(asyncio.get_running_loop())
            q: asyncio.Queue = asyncio.Queue()
            hub.add(q)
            w.start()
            book = _ask_book()
            # Engine needs the book; tape is None so pass via a tiny stub.
            class _Tape:
                pass
            tape = _Tape()
            tape.book = book
            w.tape = tape

            t0 = 1_700_000_000.0
            prices = (5800.00, 5800.25, 5800.50)
            sizes = (40.0, 46.0, 52.0)
            for i, (px, sz) in enumerate(zip(prices, sizes)):
                w.push_trade(T(t0 + i * 0.01, px, sz, "buy"))
            for i in range(8):
                w.push_trade(T(t0 + 0.30 + i * 0.002, 5799.50, 22.0, "sell"))

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
        assert eng.stance == LONG
