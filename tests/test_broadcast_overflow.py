"""A slow client must not take the server down with it.

THE CRASH THIS FILE EXISTS TO PREVENT

`loop.call_soon_threadsafe(q.put_nowait, frame)` only SCHEDULES the put.
It returns immediately, so a `try/except` wrapped around that call can
never see what the put does. When the client's queue is full the
QueueFull is raised later, inside the event loop's own Handle, with no
handler in reach -- so every overflow becomes an unhandled exception,
and the default handler logs the Handle's repr, which contains the whole
book frame.

On ES that arrives tens of thousands of times a second. The process does
not die of the exception; it dies of writing it down.

AND: when the queue is full, the frame to throw away is the OLDEST one.
A book snapshot is only worth anything while it is current, so a lagging
client should be shown the newest book, not fed a queue of stale ones.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from liqmap.pipeline import IngestHub                      # noqa: E402
from liqmap.web import BookBroadcaster, SignalBroadcaster  # noqa: E402

HUBS = [BookBroadcaster, SignalBroadcaster, IngestHub]


async def _flood(hub_cls, n=40, cap=4):
    """Register one tiny queue, fire n frames at it, collect loop errors."""
    loop = asyncio.get_running_loop()
    errors: list = []
    loop.set_exception_handler(lambda lp, ctx: errors.append(ctx))

    hub = hub_cls()
    hub.set_loop(loop)
    q: asyncio.Queue = asyncio.Queue(maxsize=cap)
    hub.add(q)

    for i in range(n):
        hub.notify({"i": i, "kind": "book"})
    for _ in range(8):                  # drain the scheduled callbacks
        await asyncio.sleep(0)
    return hub, q, errors


@pytest.mark.parametrize("hub_cls", HUBS, ids=lambda c: c.__name__)
def test_an_overflow_does_not_reach_the_loop_exception_handler(hub_cls):
    """The whole bug in one assertion."""
    hub, q, errors = asyncio.run(_flood(hub_cls))
    assert errors == [], (
        f"{len(errors)} unhandled exception(s) escaped into the event "
        f"loop; first: {errors[0].get('exception') if errors else None}")


@pytest.mark.parametrize("hub_cls", HUBS, ids=lambda c: c.__name__)
def test_the_overflow_is_counted_rather_than_silent(hub_cls):
    hub, q, errors = asyncio.run(_flood(hub_cls))
    assert hub.dropped > 0, "frames vanished without being counted"


@pytest.mark.parametrize("hub_cls", HUBS, ids=lambda c: c.__name__)
def test_a_lagging_client_is_left_holding_the_NEWEST_frames(hub_cls):
    """A stale book is worse than no book.

    If the full queue rejects the incoming frame, the client keeps the
    four oldest for ever and the display freezes. Evicting the oldest
    keeps the window moving.
    """
    hub, q, errors = asyncio.run(_flood(hub_cls, n=40, cap=4))
    held = []
    while not q.empty():
        held.append(q.get_nowait()["i"])
    assert held, "the client was left with nothing at all"
    assert max(held) >= 35, (
        f"client is stuck on stale frames {held}; the newest never "
        f"arrived")


@pytest.mark.parametrize("hub_cls", HUBS, ids=lambda c: c.__name__)
def test_a_client_that_keeps_up_loses_nothing(hub_cls):
    """The fix must not cost anything when nobody is behind."""
    async def go():
        loop = asyncio.get_running_loop()
        hub = hub_cls()
        hub.set_loop(loop)
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        hub.add(q)
        for i in range(32):
            hub.notify({"i": i})
        for _ in range(8):
            await asyncio.sleep(0)
        return hub, q
    hub, q = asyncio.run(go())
    assert q.qsize() == 32 and hub.dropped == 0


@pytest.mark.parametrize("hub_cls", HUBS, ids=lambda c: c.__name__)
def test_one_slow_client_does_not_starve_a_fast_one(hub_cls):
    async def go():
        loop = asyncio.get_running_loop()
        errors: list = []
        loop.set_exception_handler(lambda lp, ctx: errors.append(ctx))
        hub = hub_cls()
        hub.set_loop(loop)
        slow: asyncio.Queue = asyncio.Queue(maxsize=2)
        fast: asyncio.Queue = asyncio.Queue(maxsize=256)
        hub.add(slow)
        hub.add(fast)
        for i in range(50):
            hub.notify({"i": i})
        for _ in range(8):
            await asyncio.sleep(0)
        return fast, errors
    fast, errors = asyncio.run(go())
    assert errors == []
    assert fast.qsize() == 50, "the fast client lost frames"


@pytest.mark.parametrize("hub_cls", HUBS, ids=lambda c: c.__name__)
def test_a_disconnected_client_stops_receiving(hub_cls):
    async def go():
        loop = asyncio.get_running_loop()
        hub = hub_cls()
        hub.set_loop(loop)
        q: asyncio.Queue = asyncio.Queue(maxsize=16)
        hub.add(q)
        hub.notify({"i": 0})
        hub.remove(q)
        hub.notify({"i": 1})
        for _ in range(8):
            await asyncio.sleep(0)
        return q
    q = asyncio.run(go())
    assert q.qsize() == 1
