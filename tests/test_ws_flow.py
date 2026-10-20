"""Tests for the WebGL order-flow WebSocket endpoint and its snapshot helper.

Coverage goals
──────────────
1.  _flow_snapshot OUTPUT SHAPE. The function must always return the
    required keys even when the tape has no trades and no book.

2.  _flow_snapshot AGGREGATION. Buy and sell volumes accumulate per tick.
    The POC is the level with the highest combined volume. Levels farther
    than 100 ticks from the current price are pruned before sending.

3.  _flow_snapshot BAR FILTER. Trades that predate the current bar's open
    are excluded.

4.  _flow_snapshot SORT ORDER. Profile rows are sorted descending by price
    (highest price first) so the WebGL renderer can draw top-to-bottom
    without reversing the array.

5.  /ws/flow BAD TOKEN. A wrong or missing token must be rejected before
    any WebSocket handshake is accepted (WS close code 4001).

6.  /ws/flow NON-CME COIN. Connecting for a coin that has no CME tape
    must result in an accepted connection followed immediately by a JSON
    error frame and a close — not a server crash.

7.  /ws/flow CME COIN. A single snapshot frame sent over the wire must
    contain every required top-level key.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap.web import _flow_snapshot  # noqa: E402 — module-level helper

# ---------------------------------------------------------------------------
# Minimal stubs for the objects _flow_snapshot expects on `tape`
# ---------------------------------------------------------------------------

TICK = 0.25   # ES tick size


@dataclass
class _FakeTrade:
    px: float
    sz: float
    aggressor: str   # "buy" | "sell"
    ts: float


@dataclass
class _FakeFlow:
    _trades: list = field(default_factory=list)


@dataclass
class _FakeLevel:
    px: float
    sz: float


@dataclass
class _FakeBook:
    bids: list = field(default_factory=list)
    asks: list = field(default_factory=list)


@dataclass
class _FakeCandle:
    """Minimal candle returned by tape.candle() — mirrors LiveCandle fields."""
    start_ts: float
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    trades: int = 0
    delta: float = 0.0


class _FakeTape:
    """Stub Tape with configurable last_px, book, flow, and candle."""

    def __init__(
        self,
        last_px: float = 0.0,
        book: Any = None,
        trades: list | None = None,
        candle_obj: Any = None,
    ):
        self.last_px = last_px
        self.book = book
        flow = _FakeFlow()
        if trades:
            flow._trades = trades
        self.flow = flow
        self._candle_obj = candle_obj

    def candle(self, interval: str, now: float | None = None):
        return self._candle_obj


# ---------------------------------------------------------------------------
# _flow_snapshot unit tests
# ---------------------------------------------------------------------------

class TestFlowSnapshotShape:

    def test_required_keys_present_with_empty_tape(self):
        tape = _FakeTape()
        out = _flow_snapshot(tape, "1m")
        for key in ("ts", "px", "poc", "maxv", "profile", "bids", "asks", "bar"):
            assert key in out, f"missing key: {key!r}"

    def test_profile_is_a_list(self):
        tape = _FakeTape()
        assert isinstance(_flow_snapshot(tape, "1m")["profile"], list)

    def test_bids_and_asks_are_lists(self):
        tape = _FakeTape()
        out = _flow_snapshot(tape, "1m")
        assert isinstance(out["bids"], list)
        assert isinstance(out["asks"], list)

    def test_bar_is_none_when_no_candle(self):
        tape = _FakeTape(candle_obj=None)
        assert _flow_snapshot(tape, "1m")["bar"] is None

    def test_bar_fields_match_candle_when_present(self):
        candle = _FakeCandle(
            start_ts=1_700_000_000.0,
            open=5800.0, high=5810.0, low=5795.0, close=5805.0,
            volume=250.0, trades=42, delta=1234.5,
        )
        tape = _FakeTape(last_px=5805.0, candle_obj=candle)
        bar = _flow_snapshot(tape, "1m")["bar"]
        assert bar is not None
        assert bar["open"] == 5800.0
        assert bar["high"] == 5810.0
        assert bar["low"] == 5795.0
        assert bar["close"] == 5805.0
        assert bar["vol"] == 250.0
        assert bar["trades"] == 42

    def test_maxv_defaults_to_one_when_no_trades(self):
        tape = _FakeTape()
        assert _flow_snapshot(tape, "1m")["maxv"] == 1.0

    def test_px_reflects_tape_last_px(self):
        tape = _FakeTape(last_px=5801.25)
        assert _flow_snapshot(tape, "1m")["px"] == 5801.25

    def test_ts_is_a_positive_float(self):
        tape = _FakeTape()
        ts = _flow_snapshot(tape, "1m")["ts"]
        assert isinstance(ts, float) and ts > 0


class TestFlowSnapshotAggregation:

    def _tape_with_trades(self, trades, last_px=5800.0):
        candle = _FakeCandle(
            start_ts=1_700_000_000.0,
            open=last_px, high=last_px, low=last_px, close=last_px,
        )
        return _FakeTape(last_px=last_px, trades=trades, candle_obj=candle)

    def test_single_buy_trade_appears_in_profile(self):
        trades = [_FakeTrade(px=5800.0, sz=5.0, aggressor="buy", ts=1_700_000_001.0)]
        out = _flow_snapshot(self._tape_with_trades(trades), "1m")
        row = next(r for r in out["profile"] if r["px"] == 5800.0)
        assert row["b"] == 5.0
        assert row["s"] == 0.0

    def test_single_sell_trade_appears_in_profile(self):
        trades = [_FakeTrade(px=5800.0, sz=3.0, aggressor="sell", ts=1_700_000_001.0)]
        out = _flow_snapshot(self._tape_with_trades(trades), "1m")
        row = next(r for r in out["profile"] if r["px"] == 5800.0)
        assert row["s"] == 3.0
        assert row["b"] == 0.0

    def test_buys_and_sells_accumulate_separately(self):
        trades = [
            _FakeTrade(px=5800.0, sz=2.0, aggressor="buy",  ts=1_700_000_001.0),
            _FakeTrade(px=5800.0, sz=7.0, aggressor="buy",  ts=1_700_000_002.0),
            _FakeTrade(px=5800.0, sz=4.0, aggressor="sell", ts=1_700_000_003.0),
        ]
        out = _flow_snapshot(self._tape_with_trades(trades), "1m")
        row = next(r for r in out["profile"] if r["px"] == 5800.0)
        assert row["b"] == 9.0
        assert row["s"] == 4.0

    def test_poc_is_level_with_highest_combined_volume(self):
        trades = [
            _FakeTrade(px=5800.0, sz=1.0, aggressor="buy",  ts=1_700_000_001.0),
            _FakeTrade(px=5800.25, sz=10.0, aggressor="sell", ts=1_700_000_002.0),
            _FakeTrade(px=5800.25, sz=5.0,  aggressor="buy",  ts=1_700_000_003.0),
        ]
        out = _flow_snapshot(self._tape_with_trades(trades, last_px=5800.25), "1m")
        # 5800.0: total = 1; 5800.25: total = 15  → POC at 5800.25
        assert out["poc"] == pytest.approx(5800.25)

    def test_maxv_reflects_dominant_level(self):
        trades = [
            _FakeTrade(px=5800.0, sz=5.0, aggressor="buy",  ts=1_700_000_001.0),
            _FakeTrade(px=5800.0, sz=8.0, aggressor="sell", ts=1_700_000_002.0),
        ]
        out = _flow_snapshot(self._tape_with_trades(trades), "1m")
        assert out["maxv"] == pytest.approx(13.0)

    def test_profile_sorted_descending_by_price(self):
        trades = [
            _FakeTrade(px=5800.0,  sz=1.0, aggressor="buy",  ts=1_700_000_001.0),
            _FakeTrade(px=5800.25, sz=2.0, aggressor="sell", ts=1_700_000_002.0),
            _FakeTrade(px=5799.75, sz=3.0, aggressor="buy",  ts=1_700_000_003.0),
        ]
        out = _flow_snapshot(self._tape_with_trades(trades), "1m")
        prices = [r["px"] for r in out["profile"]]
        assert prices == sorted(prices, reverse=True)


class TestFlowSnapshotBarFilter:

    def test_trades_before_bar_start_are_excluded(self):
        bar_start = 1_700_000_100.0
        trades = [
            _FakeTrade(px=5800.0, sz=9.0, aggressor="buy", ts=bar_start - 1.0),  # old
            _FakeTrade(px=5800.0, sz=2.0, aggressor="sell", ts=bar_start + 1.0),  # new
        ]
        candle = _FakeCandle(
            start_ts=bar_start,
            open=5800.0, high=5800.0, low=5800.0, close=5800.0,
        )
        tape = _FakeTape(last_px=5800.0, trades=trades, candle_obj=candle)
        out = _flow_snapshot(tape, "1m")
        row = next((r for r in out["profile"] if r["px"] == 5800.0), None)
        assert row is not None
        # Only the in-bar trade (sz=2 sell) should appear — not the old sz=9 buy
        assert row["b"] == 0.0
        assert row["s"] == 2.0


class TestFlowSnapshotPriceWindow:

    def test_levels_beyond_100_ticks_are_pruned(self):
        last_px = 5800.0
        near  = last_px + 5 * TICK    # ≈ 5801.25 — within ±100 ticks
        far   = last_px + 200 * TICK  # ≈ 5850.00 — outside ±100 ticks
        bar_start = 1_700_000_000.0
        trades = [
            _FakeTrade(px=near, sz=1.0, aggressor="buy",  ts=bar_start + 1.0),
            _FakeTrade(px=far,  sz=1.0, aggressor="sell", ts=bar_start + 2.0),
        ]
        candle = _FakeCandle(
            start_ts=bar_start,
            open=last_px, high=last_px, low=last_px, close=last_px,
        )
        tape = _FakeTape(last_px=last_px, trades=trades, candle_obj=candle)
        out = _flow_snapshot(tape, "1m")
        profile_px = {r["px"] for r in out["profile"]}
        assert near in profile_px
        assert far not in profile_px


class TestFlowSnapshotBook:

    def test_book_levels_appear_in_output(self):
        bids = [_FakeLevel(px=5799.75, sz=50.0), _FakeLevel(px=5799.50, sz=30.0)]
        asks = [_FakeLevel(px=5800.00, sz=40.0)]
        book = _FakeBook(bids=bids, asks=asks)
        tape = _FakeTape(last_px=5800.0, book=book)
        out = _flow_snapshot(tape, "1m")
        assert len(out["bids"]) == 2
        assert len(out["asks"]) == 1
        assert out["bids"][0] == {"px": 5799.75, "sz": 50.0}

    def test_no_book_gives_empty_lists(self):
        tape = _FakeTape(book=None)
        out = _flow_snapshot(tape, "1m")
        assert out["bids"] == []
        assert out["asks"] == []

    def test_book_capped_at_ten_levels(self):
        bids = [_FakeLevel(px=5800.0 - i * TICK, sz=float(i)) for i in range(20)]
        book = _FakeBook(bids=bids, asks=[])
        tape = _FakeTape(last_px=5800.0, book=book)
        out = _flow_snapshot(tape, "1m")
        assert len(out["bids"]) == 10


# ---------------------------------------------------------------------------
# /ws/flow WebSocket endpoint integration tests
# ---------------------------------------------------------------------------

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
import liqmap.web as web                  # noqa: E402

TOKEN = "ws-flow-token-xyz"


@pytest.fixture
def client(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("LIQMAP_DB",    str(Path(tmp) / "ws_flow.db"))
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
    """Client with CME/OFI enabled (no real Databento key — engine won't connect)."""
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("LIQMAP_DB",       str(Path(tmp) / "ws_flow_cme.db"))
        monkeypatch.setenv("LIQMAP_TOKEN",    TOKEN)
        monkeypatch.setenv("LIQMAP_AUTO",     "false")
        monkeypatch.setenv("LIQMAP_OFI",      "true")
        monkeypatch.setenv("LIQMAP_OFI_SYMBOLS", "ES.c.0")
        # No DATABENTO_API_KEY → engine thread exits immediately; tape exists
        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
        web.RT = None
        with TestClient(web.create_app()) as c:
            yield c
        web.RT = None


class TestWsFlowAuth:

    def test_bad_token_rejected_before_handshake(self, client):
        """A wrong token must cause the server to close the WebSocket
        before accepting, so the client never receives a data frame."""
        with pytest.raises(Exception):
            # TestClient raises WebSocketDisconnect or similar when the
            # server closes before accept() — that is the expected outcome.
            with client.websocket_connect(
                f"/ws/flow?coin=ES&token=WRONG&interval=1m"
            ) as ws:
                ws.receive_json()

    def test_empty_token_rejected(self, client):
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/flow?coin=ES&token=&interval=1m") as ws:
                ws.receive_json()


class TestWsFlowNonCme:

    def test_non_cme_coin_returns_error_json(self, client):
        """When there is no CME engine, ES (and any coin) is not CME-routed.
        The endpoint must accept, send a JSON error frame, and then close."""
        with client.websocket_connect(
            f"/ws/flow?coin=BTC&token={TOKEN}&interval=1m"
        ) as ws:
            msg = ws.receive_json()
        assert "error" in msg

    def test_es_without_cme_engine_returns_error(self, client):
        """ES with CME disabled is not routed to any tape. The endpoint must
        send an error frame instead of blocking forever."""
        with client.websocket_connect(
            f"/ws/flow?coin=ES&token={TOKEN}&interval=1m"
        ) as ws:
            msg = ws.receive_json()
        assert "error" in msg


class TestWsFlowSnapshot:

    def test_first_frame_has_required_keys(self, client_cme):
        """With CME enabled (tape exists, no trades yet), the first frame
        must contain every required key even though the tape is empty."""
        with client_cme.websocket_connect(
            f"/ws/flow?coin=ES&token={TOKEN}&interval=1m"
        ) as ws:
            frame = ws.receive_json()
        for key in ("ts", "px", "poc", "maxv", "profile", "bids", "asks", "bar"):
            assert key in frame, f"WebSocket frame missing key: {key!r}"

    def test_profile_is_list_in_first_frame(self, client_cme):
        with client_cme.websocket_connect(
            f"/ws/flow?coin=ES&token={TOKEN}&interval=1m"
        ) as ws:
            frame = ws.receive_json()
        assert isinstance(frame["profile"], list)

    def test_px_is_numeric_in_first_frame(self, client_cme):
        with client_cme.websocket_connect(
            f"/ws/flow?coin=ES&token={TOKEN}&interval=1m"
        ) as ws:
            frame = ws.receive_json()
        assert isinstance(frame["px"], (int, float))
