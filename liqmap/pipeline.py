"""Event-driven ingest pipeline: queue in, micro-deltas out.

The live path used to have two bottlenecks that made the chart feel like
a polled HTTP page:

  1. /ws/flow rebuilt a full volume-at-price snapshot every 16ms
     (``asyncio.sleep(0.016)``) whether or not anything had traded.
  2. The dashboard called ``/api/ofi``, ``/api/footprint``, ``/api/now``,
     ``/api/read`` and ``/api/feed`` on ``setInterval`` timers.

Neither waits on the network buffer. Both are gone.

WHAT REPLACES THEM

    IngestHub     thread → asyncio.Queue fan-out (same shape as
                  BookBroadcaster). One hub, every kind of micro-delta.
    DeltaWorker   in-memory ``SimpleQueue``. ``push_trade`` returns in
                  microseconds; a dedicated thread owns Diagonal
                  Cumulative Delta and never shares a lock with the UI.
    tick_frame    the processed micro-delta pushed to /ws/flow and
                  /ws/live the instant the worker folds a print.

INGEST IS STILL A SOCKET, NOT A POLL. Databento's Live client is a
blocking iterator (their SDK has no asyncio API). The OFI thread reads
that iterator and the moment a record arrives it is offered to the tape.
The tape's exec listener is ``DeltaWorker.push_trade`` — a non-blocking
``queue.put``. Calculation and WebSocket fan-out happen after that, on
other threads, so the ingest path cannot stall on DCD or on a slow
browser.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable


# --------------------------------------------------------------------------
# IngestHub — one fan-out for every micro-delta kind
# --------------------------------------------------------------------------

class IngestHub:
    """Fan-out from ingest / worker threads to N async WebSocket queues.

    Frames are dicts with a ``kind`` key:
        tick    one processed trade + live DCD
        book    MBP-10 snapshot (same payload as /ws/book)
        ofi     latest OFI reading
        signal  execution Signal
        snap    full flow snapshot (connect only)
    """

    def __init__(self) -> None:
        self._queues: set = set()
        self._lock = threading.Lock()
        self._loop: Any = None
        self.dropped: int = 0
        self.sent: int = 0

    def set_loop(self, loop: Any) -> None:
        self._loop = loop

    def add(self, q: Any) -> None:
        with self._lock:
            self._queues.add(q)

    def remove(self, q: Any) -> None:
        with self._lock:
            self._queues.discard(q)

    def notify(self, frame: dict) -> None:
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        with self._lock:
            queues = list(self._queues)
        for q in queues:
            try:
                loop.call_soon_threadsafe(q.put_nowait, frame)
                self.sent += 1
            except Exception:                           # noqa: BLE001
                self.dropped += 1

    @property
    def connections(self) -> int:
        with self._lock:
            return len(self._queues)


# --------------------------------------------------------------------------
# tick micro-delta
# --------------------------------------------------------------------------

def tick_frame(trade: Any, exec_track: Any) -> dict:
    """One processed print, ready for the WebGL / dashboard buffers.

    The frontend applies this to a typed array / profile buffer. It is
    not a component-state update and it is not a full snapshot.
    """
    st = exec_track.state()
    pulse = st.pulse
    return {
        "kind": "tick",
        "ts": float(getattr(trade, "ts", 0.0) or 0.0),
        "px": float(getattr(trade, "px", 0.0) or 0.0),
        "sz": float(getattr(trade, "sz", 0.0) or 0.0),
        "side": getattr(trade, "aggressor", "") or "",
        "dcd": st.diagonal_delta,
        "raw": st.raw_delta,
        "unconfirmed": pulse.unconfirmed if pulse else 0,
        "buy_flags": pulse.buy_flags if pulse else 0,
        "sell_flags": pulse.sell_flags if pulse else 0,
        "stacked": [
            {"side": s, "low": lo, "high": hi, "run": r}
            for (s, lo, hi, r) in st.stacked
        ],
        "signals_total": st.signals_total,
        "last_signal": st.last_signal.as_dict() if st.last_signal else None,
    }


# --------------------------------------------------------------------------
# DeltaWorker — DCD lives here, not on the UI thread and not on ingest
# --------------------------------------------------------------------------

class DeltaWorker:
    """Background worker that owns Diagonal Cumulative Delta.

    ``push_trade`` is the only method the ingest thread calls. It is a
    ``SimpleQueue.put`` — it does not compute, it does not lock the
    execution track, and it does not touch a WebSocket.
    """

    def __init__(self, hub: IngestHub, exec_track: Any) -> None:
        self.hub = hub
        self.exec_track = exec_track
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.processed: int = 0
        self.dropped: int = 0

    def push_trade(self, trade: Any) -> None:
        """Non-blocking. Called from Tape.offer on the ingest thread."""
        try:
            self._q.put(trade)
        except Exception:                               # noqa: BLE001
            self.dropped += 1

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="delta-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def pending(self) -> int:
        return self._q.qsize()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                trade = self._q.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self.exec_track.on_trade(trade)
                self.processed += 1
                if self.hub.connections:
                    self.hub.notify(tick_frame(trade, self.exec_track))
            except Exception:                           # noqa: BLE001
                self.dropped += 1
