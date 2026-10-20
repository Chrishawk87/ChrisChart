"""Event-driven ingest pipeline: no 16ms poll, DCD on a worker queue.

Coverage
────────
1. IngestHub lifecycle + thread→queue delivery
2. tick_frame shape
3. DeltaWorker: push_trade is non-blocking; DCD updates off the caller
4. /ws/flow first frame is a snap (not a 16ms rebuild loop)
5. /ws/live auth, non-CME error, CME snap
6. Dashboard no longer HTTP-polls OFI / footprint / now / read / feed
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap.pipeline import IngestHub, DeltaWorker, tick_frame  # noqa: E402
from liqmap.execution import ExecutionTrack  # noqa: E402


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

@dataclass
class _Trade:
    ts: float
    px: float
    sz: float
    aggressor: str


# ---------------------------------------------------------------------------
# IngestHub
# ---------------------------------------------------------------------------

class TestIngestHub:

    def test_starts_empty(self):
        h = IngestHub()
        assert h.connections == 0

    def test_add_remove(self):
        h = IngestHub()
        q = asyncio.Queue()
        h.add(q)
        assert h.connections == 1
        h.remove(q)
        assert h.connections == 0

    def test_notify_noop_without_loop(self):
        h = IngestHub()
        q = asyncio.Queue()
        h.add(q)
        h.notify({"kind": "tick"})
        assert q.empty()

    def test_notify_delivers_via_loop(self):
        h = IngestHub()

        async def _run():
            h.set_loop(asyncio.get_running_loop())
            q: asyncio.Queue = asyncio.Queue()
            h.add(q)

            def _thread():
                h.notify({"kind": "tick", "px": 5800.0})
            t = threading.Thread(target=_thread)
            t.start()
            t.join()
            return await asyncio.wait_for(q.get(), timeout=1.0)

        frame = asyncio.run(_run())
        assert frame["kind"] == "tick"
        assert frame["px"] == 5800.0


# ---------------------------------------------------------------------------
# tick_frame + DeltaWorker
# ---------------------------------------------------------------------------

class TestTickFrame:

    def test_required_keys(self):
        et = ExecutionTrack()
        frame = tick_frame(_Trade(1.0, 5800.0, 4, "buy"), et)
        for key in ("kind", "ts", "px", "sz", "side", "dcd", "raw",
                    "unconfirmed", "buy_flags", "sell_flags", "stacked",
                    "signals_total"):
            assert key in frame
        assert frame["kind"] == "tick"
        assert frame["px"] == 5800.0
        assert frame["side"] == "buy"


class TestDeltaWorker:

    def test_push_does_not_block_or_compute_inline(self):
        """push_trade must return without waiting on DCD."""
        hub = IngestHub()
        et = ExecutionTrack()
        w = DeltaWorker(hub, et)
        t0 = time.perf_counter()
        w.push_trade(_Trade(1.0, 5800.0, 10, "buy"))
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.01
        # Worker not started → DCD still zero (computation did not run inline)
        assert et._dcd.raw == 0

    def test_worker_processes_queued_trade(self):
        hub = IngestHub()
        et = ExecutionTrack()
        w = DeltaWorker(hub, et)
        w.start()
        w.push_trade(_Trade(1.0, 5800.0, 10, "buy"))
        deadline = time.time() + 2.0
        while w.processed < 1 and time.time() < deadline:
            time.sleep(0.01)
        w.stop()
        assert w.processed >= 1
        assert et._dcd.raw == 10

    def test_worker_emits_tick_to_hub(self):
        hub = IngestHub()
        et = ExecutionTrack()
        w = DeltaWorker(hub, et)

        async def _run():
            hub.set_loop(asyncio.get_running_loop())
            q: asyncio.Queue = asyncio.Queue()
            hub.add(q)
            w.start()
            w.push_trade(_Trade(1.0, 5800.25, 5, "sell"))
            frame = await asyncio.wait_for(q.get(), timeout=2.0)
            w.stop()
            return frame

        frame = asyncio.run(_run())
        assert frame["kind"] == "tick"
        assert frame["px"] == 5800.25
        assert frame["side"] == "sell"
        assert frame["raw"] == -5


# ---------------------------------------------------------------------------
# Dashboard: market-data HTTP polls are gone
# ---------------------------------------------------------------------------

class TestNoMarketDataPolling:

    def test_dashboard_dropped_ofi_and_footprint_intervals(self):
        from liqmap.web import DASHBOARD
        # Comments may mention the deleted calls; live code must not.
        assert "setInterval(loadOfi," not in DASHBOARD
        assert "setInterval(fpTick," not in DASHBOARD
        assert "setInterval(paintFeed," not in DASHBOARD
        assert "setInterval(loadRead," not in DASHBOARD
        assert "setInterval(loadSuggest," not in DASHBOARD
        assert "setInterval(loadAutopilot," not in DASHBOARD
        assert "setInterval(loadLiquidity," not in DASHBOARD

    def test_dashboard_opens_live_websocket(self):
        from liqmap.web import DASHBOARD
        assert "openLiveStream" in DASHBOARD
        assert "/ws/live" in DASHBOARD
        assert "applyLiveFrame" in DASHBOARD

    def test_gl_page_applies_ticks_in_place(self):
        from liqmap.web import _GL_PAGE
        assert "applyTick" in _GL_PAGE
        assert "d.kind === 'tick'" in _GL_PAGE

    def test_ws_flow_source_has_no_sixteen_ms_sleep(self):
        import inspect
        import liqmap.web as web
        src = inspect.getsource(web.create_app)
        assert "0.016" not in src
        assert "asyncio.sleep" not in src or "0.016" not in src


# ---------------------------------------------------------------------------
# /ws/live + /ws/flow still serve a snap on connect
# ---------------------------------------------------------------------------

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
import liqmap.web as web                  # noqa: E402

TOKEN = "pipe-token-xyz"


@pytest.fixture
def client(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("LIQMAP_DB", str(Path(tmp) / "pipe.db"))
        monkeypatch.setenv("LIQMAP_TOKEN", TOKEN)
        monkeypatch.setenv("LIQMAP_AUTO", "false")
        monkeypatch.delenv("LIQMAP_OFI", raising=False)
        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
        web.RT = None
        with TestClient(web.create_app()) as c:
            yield c
        web.RT = None


@pytest.fixture
def client_cme(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("LIQMAP_DB", str(Path(tmp) / "pipe_cme.db"))
        monkeypatch.setenv("LIQMAP_TOKEN", TOKEN)
        monkeypatch.setenv("LIQMAP_AUTO", "false")
        monkeypatch.setenv("LIQMAP_OFI", "true")
        monkeypatch.setenv("LIQMAP_OFI_SYMBOLS", "ES.c.0")
        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
        web.RT = None
        with TestClient(web.create_app()) as c:
            yield c
        web.RT = None


class TestWsLive:

    def test_bad_token_rejected(self, client):
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/live?coin=ES&token=WRONG") as ws:
                ws.receive_json()

    def test_non_cme_without_feed_errors(self, client):
        with client.websocket_connect(f"/ws/live?coin=BTC&token={TOKEN}") as ws:
            msg = ws.receive_json()
        assert "error" in msg

    def test_cme_snap_on_connect(self, client_cme):
        with client_cme.websocket_connect(f"/ws/live?coin=ES&token={TOKEN}") as ws:
            frame = ws.receive_json()
        assert frame.get("kind") == "snap"
        assert "px" in frame
        assert "exec" in frame

    def test_hub_cleans_up_on_disconnect(self, client_cme):
        rt = web.runtime()
        before = rt.ingest_hub.connections
        with client_cme.websocket_connect(f"/ws/live?coin=ES&token={TOKEN}") as ws:
            ws.receive_json()
            during = rt.ingest_hub.connections
        after = rt.ingest_hub.connections
        assert during > before
        assert after == before


class TestWsFlowEventDriven:

    def test_first_frame_is_snap(self, client_cme):
        with client_cme.websocket_connect(
            f"/ws/flow?coin=ES&token={TOKEN}&interval=1m"
        ) as ws:
            frame = ws.receive_json()
        assert frame.get("kind") == "snap"
        for key in ("ts", "px", "poc", "maxv", "profile", "bids", "asks", "bar"):
            assert key in frame

    def test_delta_worker_running(self, client_cme):
        rt = web.runtime()
        assert rt.delta_worker.running
