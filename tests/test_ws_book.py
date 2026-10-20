"""Tests for the real-time MBP-10 book WebSocket endpoint and its helpers.

Coverage goals
──────────────
1.  BookBroadcaster LIFECYCLE. add/remove queues; connections count tracks
    them correctly; dropped stays zero while the loop is absent.

2.  BookBroadcaster NOTIFY. When a loop is present, notify() schedules
    put_nowait on every registered queue.

3.  _book_delta SHAPE. The frame always contains the paintBookCall-
    compatible surface (coin, direction, score, mid, etc.) regardless of
    whether the tape has a book.

4.  _book_delta CONTENT. When a book is present the bids/asks arrays
    carry the right prices and sizes; imbalance and direction match.

5.  Tape.on_book_update FIRES ON BOOK RECORDS. A call to tape.offer()
    with a record that carries levels must fire book listeners; one
    without levels must not.

6.  /ws/book BAD TOKEN. Wrong or missing token → rejected before accept.

7.  /ws/book NON-CME COIN. Error frame returned, connection closes.

8.  /ws/book CME COIN. First frame contains the full required surface.

9.  /ws/book DISCONNECTS CLEANLY. The queue is removed from the
    broadcaster on disconnect so it stops receiving frames.
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

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap.web import BookBroadcaster, _book_delta  # noqa: E402

# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------

PT = 1_000_000_000
TICK = PT // 4
BASE = 5800 * PT
T0 = 1_700_000_000.0


class Lv:
    def __init__(self, bp, bs, ap, asz):
        self.bid_px, self.bid_sz, self.ask_px, self.ask_sz = bp, bs, ap, asz


class Rec:
    """Minimal DBN-like record."""
    def __init__(self, action, px, sz, side, ts, lv=None):
        self.action = action
        self.price  = px
        self.size   = sz
        self.side   = side
        self.ts_event = int(ts * 1e9)
        self.instrument_id = 7
        self.sequence      = 1
        self.levels = lv if lv is not None else [
            Lv(BASE - j * TICK, 99, BASE + (j + 1) * TICK, 99)
            for j in range(10)
        ]


def book_rec(ts=T0):
    """A non-trade record that carries 10 levels."""
    return Rec("A", BASE, 50, "B", ts)


def trade_rec(ts=T0):
    """A trade record that also carries levels."""
    return Rec("T", BASE + TICK, 5, "B", ts)


def no_book_rec(ts=T0):
    """A record with no levels field."""
    r = Rec("A", BASE, 50, "B", ts, lv=[])
    r.levels = []          # empty → dbn._get sees it as falsy
    return r


# ---------------------------------------------------------------------------
# 1-2. BookBroadcaster unit tests (no app required)
# ---------------------------------------------------------------------------

class TestBookBroadcaster:

    def test_starts_with_zero_connections(self):
        bc = BookBroadcaster()
        assert bc.connections == 0

    def test_add_increments_connections(self):
        bc = BookBroadcaster()
        q = asyncio.Queue()
        bc.add(q)
        assert bc.connections == 1

    def test_remove_decrements_connections(self):
        bc = BookBroadcaster()
        q = asyncio.Queue()
        bc.add(q)
        bc.remove(q)
        assert bc.connections == 0

    def test_remove_unknown_queue_is_silent(self):
        bc = BookBroadcaster()
        bc.remove(asyncio.Queue())   # should not raise

    def test_dropped_zero_without_loop(self):
        bc = BookBroadcaster()
        q = asyncio.Queue()
        bc.add(q)
        bc.notify({"ts": 1.0})
        assert bc.dropped == 0   # notify is a no-op when loop is absent

    def test_notify_skips_without_loop(self):
        bc = BookBroadcaster()
        q = asyncio.Queue()
        bc.add(q)
        bc.notify({"ts": 1.0})
        # No loop → queue is still empty
        assert q.empty()

    def test_notify_delivers_to_queue_via_loop(self):
        """End-to-end: notify from a thread lands in the queue."""
        bc = BookBroadcaster()

        async def _run():
            loop = asyncio.get_running_loop()
            bc.set_loop(loop)
            q: asyncio.Queue = asyncio.Queue()
            bc.add(q)

            # Simulate OFI thread calling notify
            def _thread():
                bc.notify({"ts": 42.0, "px": 5800.0})
            t = threading.Thread(target=_thread)
            t.start()
            t.join()

            # The call_soon_threadsafe schedules an item; let the loop run
            frame = await asyncio.wait_for(q.get(), timeout=1.0)
            return frame

        frame = asyncio.run(_run())
        assert frame["ts"] == 42.0

    def test_notify_delivers_to_multiple_queues(self):
        bc = BookBroadcaster()

        async def _run():
            loop = asyncio.get_running_loop()
            bc.set_loop(loop)
            qs = [asyncio.Queue() for _ in range(3)]
            for q in qs:
                bc.add(q)

            def _thread():
                bc.notify({"ts": 1.0})
            threading.Thread(target=_thread).start()

            frames = []
            for q in qs:
                frames.append(await asyncio.wait_for(q.get(), timeout=1.0))
            return frames

        frames = asyncio.run(_run())
        assert len(frames) == 3
        assert all(f["ts"] == 1.0 for f in frames)


# ---------------------------------------------------------------------------
# 3-4. _book_delta unit tests
# ---------------------------------------------------------------------------

@dataclass
class _FakeLevel:
    px: float
    sz: float
    n: int = 0


@dataclass
class _FakeBook:
    bids: list = field(default_factory=list)
    asks: list = field(default_factory=list)
    ts: float = T0
    coin: str = "ES"

    def __post_init__(self):
        pass

    @property
    def mid(self):
        if self.bids and self.asks:
            return (self.bids[0].px + self.asks[0].px) / 2
        return self.bids[0].px if self.bids else self.asks[0].px if self.asks else 0.0

    @property
    def spread_bps(self):
        m = self.mid
        if m <= 0 or not self.bids or not self.asks:
            return 0.0
        return (self.asks[0].px - self.bids[0].px) / m * 10_000.0

    @property
    def empty(self):
        return not self.bids and not self.asks


class _FakeTape:
    def __init__(self, last_px=0.0, book=None, books=0, last_ts=0.0,
                 symbol="ES"):
        self.last_px  = last_px
        self.book     = book
        self.books    = books
        self.last_ts  = last_ts
        self.symbol   = symbol


class TestBookDelta:

    def test_required_keys_when_no_book(self):
        tape = _FakeTape()
        out = _book_delta(tape)
        for key in ("ts", "source", "px", "bid", "ask", "mid", "microprice",
                    "spread_bps", "imbalance", "direction", "score",
                    "bids", "asks", "book_updates"):
            assert key in out, f"missing key {key!r}"

    def test_bids_asks_empty_when_no_book(self):
        tape = _FakeTape()
        out = _book_delta(tape)
        assert out["bids"] == []
        assert out["asks"] == []

    def test_source_is_cme(self):
        assert _book_delta(_FakeTape())["source"] == "cme"

    def test_direction_flat_when_balanced(self):
        bids = [_FakeLevel(5800.0, 100.0), _FakeLevel(5799.75, 80.0)]
        asks = [_FakeLevel(5800.25, 100.0), _FakeLevel(5800.50, 80.0)]
        book = _FakeBook(bids=bids, asks=asks)
        tape = _FakeTape(last_px=5800.0, book=book, books=1)
        out = _book_delta(tape)
        assert out["direction"] == "flat"

    def test_direction_up_when_bids_dominate(self):
        bids = [_FakeLevel(5800.0, 500.0)] * 5   # heavy bid side
        asks = [_FakeLevel(5800.25, 10.0)] * 5
        book = _FakeBook(bids=bids, asks=asks)
        tape = _FakeTape(last_px=5800.0, book=book, books=1)
        out = _book_delta(tape)
        assert out["direction"] == "up"
        assert out["imbalance"] > 0.1

    def test_direction_down_when_asks_dominate(self):
        bids = [_FakeLevel(5800.0, 10.0)] * 5
        asks = [_FakeLevel(5800.25, 500.0)] * 5   # heavy ask side
        book = _FakeBook(bids=bids, asks=asks)
        tape = _FakeTape(last_px=5800.0, book=book, books=1)
        out = _book_delta(tape)
        assert out["direction"] == "down"
        assert out["imbalance"] < -0.1

    def test_bids_asks_populated_from_book(self):
        bids = [_FakeLevel(5800.0, 50.0), _FakeLevel(5799.75, 30.0)]
        asks = [_FakeLevel(5800.25, 40.0)]
        book = _FakeBook(bids=bids, asks=asks)
        tape = _FakeTape(last_px=5800.0, book=book, books=2)
        out = _book_delta(tape)
        assert len(out["bids"]) == 2
        assert len(out["asks"]) == 1
        assert out["bids"][0]["px"] == 5800.0
        assert out["asks"][0]["sz"] == 40.0

    def test_bids_capped_at_ten(self):
        bids = [_FakeLevel(5800.0 - i * 0.25, float(i + 1)) for i in range(15)]
        asks = [_FakeLevel(5800.25, 100.0)]
        book = _FakeBook(bids=bids, asks=asks)
        tape = _FakeTape(last_px=5800.0, book=book)
        out = _book_delta(tape)
        assert len(out["bids"]) == 10

    def test_mid_and_spread_computed_correctly(self):
        bids = [_FakeLevel(5800.0, 100.0)]
        asks = [_FakeLevel(5800.5, 100.0)]
        book = _FakeBook(bids=bids, asks=asks)
        tape = _FakeTape(last_px=5800.0, book=book)
        out = _book_delta(tape)
        assert out["mid"] == pytest.approx(5800.25, abs=0.01)
        assert out["spread_bps"] > 0

    def test_book_updates_count_passed_through(self):
        tape = _FakeTape(books=42)
        assert _book_delta(tape)["book_updates"] == 42


# ---------------------------------------------------------------------------
# 5. Tape.on_book_update fires on book records, not on no-book records
# ---------------------------------------------------------------------------

class TestTapeBookListeners:

    def test_book_listener_registered_and_counted(self, monkeypatch):
        monkeypatch.setenv("LIQMAP_OFI", "false")
        from liqmap.cme_tape import Tape
        t = Tape()
        assert t.book_listener_count == 0
        t.on_book_update(lambda: None)
        assert t.book_listener_count == 1

    def test_book_listener_fires_on_record_with_levels(self, monkeypatch):
        monkeypatch.setenv("LIQMAP_OFI", "false")
        from liqmap.cme_tape import Tape
        t = Tape()
        fired = []
        t.on_book_update(lambda: fired.append(1))
        t.offer(book_rec())
        assert len(fired) == 1

    def test_book_listener_fires_on_trade_record_with_levels(self, monkeypatch):
        monkeypatch.setenv("LIQMAP_OFI", "false")
        from liqmap.cme_tape import Tape
        t = Tape()
        fired = []
        t.on_book_update(lambda: fired.append(1))
        t.offer(trade_rec())
        assert len(fired) == 1

    def test_book_listener_does_not_fire_without_levels(self, monkeypatch):
        monkeypatch.setenv("LIQMAP_OFI", "false")
        from liqmap.cme_tape import Tape
        t = Tape()
        fired = []
        t.on_book_update(lambda: fired.append(1))
        t.offer(no_book_rec())
        assert len(fired) == 0

    def test_trade_listener_still_fires_only_on_trades(self, monkeypatch):
        monkeypatch.setenv("LIQMAP_OFI", "false")
        from liqmap.cme_tape import Tape
        t = Tape()
        trade_fired = []
        book_fired  = []
        t.on_update(lambda: trade_fired.append(1))
        t.on_book_update(lambda: book_fired.append(1))
        t.offer(book_rec())    # non-trade: should fire book only
        assert trade_fired == []
        assert book_fired  == [1]
        t.offer(trade_rec())   # trade: should fire both
        assert trade_fired == [1]
        assert book_fired  == [1, 1]

    def test_off_book_update_removes_callback(self, monkeypatch):
        monkeypatch.setenv("LIQMAP_OFI", "false")
        from liqmap.cme_tape import Tape
        t = Tape()
        fired = []
        fn = lambda: fired.append(1)
        t.on_book_update(fn)
        t.offer(book_rec())
        count_after_register = len(fired)
        t.off_book_update(fn)
        t.offer(book_rec())
        assert len(fired) == count_after_register

    def test_crashing_book_listener_does_not_stop_tape(self, monkeypatch):
        monkeypatch.setenv("LIQMAP_OFI", "false")
        from liqmap.cme_tape import Tape
        t = Tape()
        def _boom():
            raise RuntimeError("crash")
        t.on_book_update(_boom)
        t.offer(book_rec())    # must not raise
        assert t.books == 1


# ---------------------------------------------------------------------------
# 6-9. /ws/book WebSocket integration tests
# ---------------------------------------------------------------------------

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
import liqmap.web as web                  # noqa: E402

TOKEN = "ws-book-token-xyz"


@pytest.fixture
def client(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("LIQMAP_DB",    str(Path(tmp) / "ws_book.db"))
        monkeypatch.setenv("LIQMAP_TOKEN", TOKEN)
        monkeypatch.setenv("LIQMAP_AUTO",  "false")
        monkeypatch.delenv("LIQMAP_OFI",         raising=False)
        monkeypatch.delenv("DATABENTO_API_KEY",   raising=False)
        web.RT = None
        with TestClient(web.create_app()) as c:
            yield c
        web.RT = None


@pytest.fixture
def client_cme(monkeypatch):
    """Client with CME/OFI enabled (no real key — tape exists but engine won't connect)."""
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("LIQMAP_DB",        str(Path(tmp) / "ws_book_cme.db"))
        monkeypatch.setenv("LIQMAP_TOKEN",     TOKEN)
        monkeypatch.setenv("LIQMAP_AUTO",      "false")
        monkeypatch.setenv("LIQMAP_OFI",       "true")
        monkeypatch.setenv("LIQMAP_OFI_SYMBOLS", "ES.c.0")
        monkeypatch.delenv("DATABENTO_API_KEY",  raising=False)
        web.RT = None
        with TestClient(web.create_app()) as c:
            yield c
        web.RT = None


class TestWsBookAuth:

    def test_bad_token_rejected_before_accept(self, client):
        with pytest.raises(Exception):
            with client.websocket_connect(
                f"/ws/book?coin=ES&token=WRONG"
            ) as ws:
                ws.receive_json()

    def test_empty_token_rejected(self, client):
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/book?coin=ES&token=") as ws:
                ws.receive_json()


class TestWsBookNonCme:

    def test_non_cme_coin_returns_error_json_then_closes(self, client):
        """Without CME engine any coin is non-CME; endpoint must send error."""
        with client.websocket_connect(
            f"/ws/book?coin=BTC&token={TOKEN}"
        ) as ws:
            msg = ws.receive_json()
        assert "error" in msg

    def test_es_without_ofi_engine_returns_error(self, client):
        with client.websocket_connect(
            f"/ws/book?coin=ES&token={TOKEN}"
        ) as ws:
            msg = ws.receive_json()
        assert "error" in msg


class TestWsBookCme:

    def test_first_frame_has_required_keys(self, client_cme):
        """With CME tape present the first frame is the current book state."""
        with client_cme.websocket_connect(
            f"/ws/book?coin=ES&token={TOKEN}"
        ) as ws:
            frame = ws.receive_json()
        for key in ("ts", "source", "px", "bid", "ask", "mid", "microprice",
                    "spread_bps", "imbalance", "direction", "bids", "asks",
                    "book_updates"):
            assert key in frame, f"missing key {key!r}"

    def test_source_is_cme(self, client_cme):
        with client_cme.websocket_connect(
            f"/ws/book?coin=ES&token={TOKEN}"
        ) as ws:
            frame = ws.receive_json()
        assert frame["source"] == "cme"

    def test_bids_and_asks_are_lists(self, client_cme):
        with client_cme.websocket_connect(
            f"/ws/book?coin=ES&token={TOKEN}"
        ) as ws:
            frame = ws.receive_json()
        assert isinstance(frame["bids"], list)
        assert isinstance(frame["asks"], list)

    def test_direction_is_valid_string(self, client_cme):
        with client_cme.websocket_connect(
            f"/ws/book?coin=ES&token={TOKEN}"
        ) as ws:
            frame = ws.receive_json()
        assert frame["direction"] in ("up", "down", "flat")

    def test_front_month_alias_routes_correctly(self, client_cme):
        """ES.Z26 is a valid alias for the CME tape."""
        rt = web.runtime()
        front, _ = rt.ofi.config.active_contracts
        with client_cme.websocket_connect(
            f"/ws/book?coin={front}&token={TOKEN}"
        ) as ws:
            frame = ws.receive_json()
        # Must not be an error frame
        assert "error" not in frame

    def test_broadcaster_removes_queue_on_disconnect(self, client_cme):
        """After the connection closes the broadcaster has one fewer queue."""
        rt = web.runtime()
        before = rt.book_broadcaster.connections
        with client_cme.websocket_connect(
            f"/ws/book?coin=ES&token={TOKEN}"
        ) as ws:
            ws.receive_json()   # consume first frame
            during = rt.book_broadcaster.connections
        after = rt.book_broadcaster.connections
        assert during > before
        assert after == before
