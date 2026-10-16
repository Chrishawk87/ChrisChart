"""The probe must come back. That is the whole requirement.

It exists to tell a market that is closed from a key that is not
entitled, and both of those look like silence. A probe that hangs on
silence answers neither question and takes the terminal with it.

The real hang: iterating the live client blocks in the socket read,
so a deadline checked inside the loop is only ever consulted when a
record has already arrived. A schema that accepted the subscription
and then sent nothing never reached the check.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "dbn_probe", ROOT / "scripts" / "dbn_probe.py")
probe_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe_mod)


# --------------------------------------------------------------- stubs

class FakeLive:
    """A live client that does what the real one does, on command."""

    behaviour = "silent"        # silent | records | refuse | wedged
    records: list = []
    made: list = []

    def __init__(self):
        self.started = False
        self.terminated = False
        self.on_record = None
        self.subs = []
        self._released = threading.Event()
        FakeLive.made.append(self)

    def subscribe(self, **kw):
        if FakeLive.behaviour == "refuse":
            # The gateway refuses at subscribe time for an unentitled
            # schema, which is how mbp-10 actually failed.
            raise RuntimeError("Not authorized for mbp-10 schema")
        self.subs.append(kw)

    def add_callback(self, record_callback, exception_callback=None):
        self.on_record = record_callback

    def start(self):
        self.started = True

    def block_for_close(self, timeout=None):
        if FakeLive.behaviour == "records":
            for r in FakeLive.records:
                self.on_record(r)
            return
        if FakeLive.behaviour == "wedged":
            # The failure the watchdog is for: the package's own
            # timeout does not fire. Only terminate() releases it.
            self._released.wait(timeout=30)
            return
        time.sleep(timeout or 0)        # silent but well-behaved

    def terminate(self):
        self.terminated = True
        self._released.set()

    def stop(self):
        self.terminated = True
        self._released.set()


@pytest.fixture(autouse=True)
def fake_databento(monkeypatch):
    mod = types.ModuleType("databento")
    mod.Live = FakeLive
    monkeypatch.setitem(sys.modules, "databento", mod)
    FakeLive.behaviour = "silent"
    FakeLive.records = []
    FakeLive.made = []
    yield
    FakeLive.behaviour = "silent"


def run(schema="bbo-1s", seconds=0.2, show=3):
    return probe_mod.probe(schema, ["ES.c.0"], "GLBX.MDP3",
                           "continuous", seconds, show)


class Trade:
    """What `trades` delivers: a print, and no book at all."""
    def __init__(self, px=5800_000_000_000, sz=3):
        self.ts_event = 1_700_000_000_000_000_000
        self.sequence, self.instrument_id = 1, 42
        self.action, self.side = "T", "B"
        self.price, self.size = px, sz


class System:
    """Session chatter. Not market data, must not be counted."""
    def __init__(self, msg="heartbeat"):
        self.msg = msg


System.__name__ = "SystemMsg"


# ---------------------------------------------------------- the point

class TestItAlwaysComesBack:

    def test_a_silent_schema_returns(self):
        """THE REGRESSION. A subscription that is accepted and then
        sends nothing used to block forever."""
        t0 = time.time()
        out = run(seconds=0.2)
        assert time.time() - t0 < 5, "it hung"
        assert out["records"] == 0
        assert out["error"] == "", "silence is not an error"

    def test_a_silent_schema_waits_the_window_out(self):
        """It must not return instantly either -- a window that is not
        actually waited makes every schema look dead."""
        t0 = time.time()
        run(seconds=0.4)
        assert time.time() - t0 >= 0.35

    def test_it_reports_how_long_it_really_listened(self):
        out = run(seconds=0.3)
        assert out["elapsed"] >= 0.25

    def test_a_refused_schema_returns_the_reason(self):
        FakeLive.behaviour = "refuse"
        out = run(schema="mbp-10", seconds=5.0)
        assert "Not authorized" in out["error"]
        assert out["records"] == 0

    def test_a_refusal_does_not_wait_the_window_out(self):
        """An entitlement refusal is known immediately; making the
        person sit through six listen windows for six refusals is how
        a 90-second run feels broken."""
        FakeLive.behaviour = "refuse"
        t0 = time.time()
        run(schema="mbp-10", seconds=5.0)
        assert time.time() - t0 < 1.0

    def test_the_watchdog_frees_a_wedged_client(self, monkeypatch):
        """If the package's own timeout does not fire, this does."""
        monkeypatch.setattr(probe_mod, "WATCHDOG_GRACE", 0.3)
        FakeLive.behaviour = "wedged"
        t0 = time.time()
        out = run(seconds=0.2)
        took = time.time() - t0
        assert took < 5, f"the watchdog did not fire ({took:.1f}s)"
        assert FakeLive.made[-1].terminated

    def test_the_watchdog_is_cancelled_when_it_is_not_needed(self):
        """A watchdog left armed would sit out its whole delay on a
        thread, once per schema, after the schema was already done.

        `cancel()` returns before the thread has finished unwinding,
        so the test joins: a cancelled timer dies immediately, an
        uncancelled one would still be waiting out WATCHDOG_GRACE.
        """
        before = {t for t in threading.enumerate()
                  if isinstance(t, threading.Timer)}
        run(seconds=0.2)
        new = [t for t in threading.enumerate()
               if isinstance(t, threading.Timer) and t not in before]
        for t in new:
            t.join(1.0)
            assert not t.is_alive(), f"watchdog still armed: {t}"

    def test_the_client_is_always_terminated(self):
        run(seconds=0.2)
        assert FakeLive.made[-1].terminated


class TestWhatItCounts:

    def test_market_data_is_counted(self):
        FakeLive.behaviour = "records"
        FakeLive.records = [Trade(), Trade(), Trade()]
        assert run()["records"] == 3

    def test_session_chatter_is_not(self):
        FakeLive.behaviour = "records"
        FakeLive.records = [System(), Trade(), System()]
        out = run()
        assert out["records"] == 1, "heartbeats are not market data"

    def test_it_shows_no_more_than_asked(self):
        FakeLive.behaviour = "records"
        FakeLive.records = [Trade() for _ in range(50)]
        out = run(show=3)
        assert out["records"] == 50, "it counts everything"
        assert len(out["rows"]) == 3, "it prints only the first few"

    def test_a_trade_record_is_reported_as_having_no_book(self):
        """The finding that decides the schema: `trades` cannot drive
        OFI, because there is no bid or ask in the record."""
        FakeLive.behaviour = "records"
        FakeLive.records = [Trade()]
        row = run()["rows"][0]
        assert row["trade"] is True
        assert row["has_book"] is False
        assert "NO (no book" in probe_mod.verdict([row])


class TestItSubscribesOnce:

    def test_one_schema_one_subscription(self):
        run(schema="bbo-1s")
        assert len(FakeLive.made[-1].subs) == 1

    def test_the_schema_asked_for_is_the_schema_sent(self):
        run(schema="bbo-1s")
        assert FakeLive.made[-1].subs[0]["schema"] == "bbo-1s"


class TestItPlacesNothing:

    def test_no_order_code(self):
        src = (ROOT / "scripts" / "dbn_probe.py").read_text()
        for word in ("def submit", "def execute", "def place", "def order"):
            assert word not in src, word
