"""The service must say why it is off, and must not die silently.

A panel that cannot tell "waiting for the subscription" from
"the thread crashed" shows the same blank box for both.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap.ofi_service import Config, OfiService            # noqa: E402

PT = 1_000_000_000
TICK = PT // 4
BASE = 5800 * PT


class Lv:
    def __init__(s, bp, bs, ap, asz):
        s.bid_px, s.bid_sz, s.ask_px, s.ask_sz = bp, bs, ap, asz


class Rec:
    def __init__(s, lv, a, t, i, q):
        s.levels, s.action = lv, a
        s.ts_event, s.instrument_id, s.sequence = t, i, q


def recs(n=800):
    random.seed(2)
    bid = BASE
    for i in range(n):
        if random.random() < 0.3:
            bid += random.choice((-TICK, TICK))
        yield Rec([Lv(bid - j * TICK, random.randint(1, 300),
                      bid + (j + 1) * TICK, random.randint(1, 300))
                   for j in range(10)],
                  random.choice("ACM"), int(1.7e9 * 1e9), 7, i + 1)


class TestBlockedReasons:

    def test_disabled_says_so(self):
        s = OfiService(Config(enabled=False))
        assert s.start() is False
        assert "LIQMAP_OFI" in s.snapshot()["blocked_by"]

    def test_a_missing_key_says_so(self, monkeypatch):
        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
        s = OfiService(Config(enabled=True))
        assert s.start() is False
        assert "DATABENTO_API_KEY" in s.snapshot()["blocked_by"]

    def test_a_ready_config_clears_every_user_facing_reason(self,
                                                             monkeypatch):
        """Under pytest the last guard still fires by design, so this
        checks that nothing a PERSON has to fix remains."""
        monkeypatch.setenv("DATABENTO_API_KEY", "x")
        assert Config(enabled=True).blocked_by() == "not started under pytest"

    def test_a_degenerate_window_is_blocked(self, monkeypatch):
        monkeypatch.setenv("DATABENTO_API_KEY", "x")
        assert "window" in Config(enabled=True, window=0).blocked_by()

    def test_snapshot_answers_when_off(self):
        snap = OfiService(Config()).snapshot()
        assert snap["running"] is False
        assert snap["last"] is None
        assert snap["blocked_by"]


class TestReplay:

    def test_readings_accumulate(self):
        s = OfiService(Config(enabled=True, window=20, baseline=50))
        n = s.feed_records(recs())
        assert n > 700
        assert s.snapshot()["readings"] == n

    def test_the_latest_reading_is_served(self):
        s = OfiService(Config(enabled=True, window=20, baseline=50,
                              z_enter=0.5))
        s.feed_records(recs())
        last = s.snapshot()["last"]
        assert last["side"] in ("buyers", "sellers", "neither")
        assert "window" in last and "e_n" in last

    def test_no_side_without_a_line(self):
        s = OfiService(Config(enabled=True, window=20, baseline=50))
        s.feed_records(recs())
        assert "side" not in s.snapshot()["last"]

    def test_history_is_capped_and_ordered(self):
        s = OfiService(Config(enabled=True, window=20, baseline=50))
        s.feed_records(recs())
        rows = s.snapshot(history=5)["history"]
        assert len(rows) == 5
        assert [r["at"] for r in rows] == sorted(r["at"] for r in rows)

    def test_no_history_unless_asked(self):
        s = OfiService(Config(enabled=True, window=20, baseline=50))
        s.feed_records(recs())
        assert "history" not in s.snapshot()

    def test_refusals_are_visible(self):
        s = OfiService(Config(enabled=True, window=20, baseline=50))
        s.feed_records(recs())
        assert s.snapshot()["refused"]["first"] == 1


class TestWiring:

    def test_the_route_exists_and_is_token_guarded(self):
        src = (Path(__file__).resolve().parent.parent
               / "liqmap" / "web.py").read_text()
        i = src.find('@app.get("/api/ofi"')
        assert i > 0, "the route is missing"
        assert "require_token" in src[i:i + 200]

    def test_runtime_holds_the_service(self):
        src = (Path(__file__).resolve().parent.parent
               / "liqmap" / "web.py").read_text()
        assert "self.ofi = _OfiService()" in src

    def test_the_service_cannot_place_an_order(self):
        src = (Path(__file__).resolve().parent.parent
               / "liqmap" / "ofi_service.py").read_text()
        for bad in ("def submit", "def execute", "def place", "def order"):
            assert bad not in src


class TestNeverLiveUnderPytest:
    """A test run must not open a billable market-data subscription.

    `Runtime()` starts the service, and the suite builds many of them.
    With LIQMAP_OFI exported in a shell that would be one live CME
    subscription per Runtime.
    """

    def test_a_fully_configured_engine_still_refuses(self, monkeypatch):
        monkeypatch.setenv("DATABENTO_API_KEY", "real-looking-key")
        cfg = Config(enabled=True)
        assert cfg.blocked_by() == "not started under pytest"
        assert OfiService(cfg).start() is False

    def test_the_real_reasons_are_reported_first(self, monkeypatch):
        """The pytest guard must not mask the message a person needs."""
        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
        assert "DATABENTO_API_KEY" in Config(enabled=True).blocked_by()
        assert Config(enabled=False).blocked_by().startswith("LIQMAP_OFI")


class TestSchemaValidation:
    """A bad schema name must fail before a socket opens.

    Caught at connect time it comes back as a message from the
    gateway that does not say which of your settings was wrong.
    """

    def test_a_good_pair_is_accepted(self, monkeypatch):
        monkeypatch.setenv("DATABENTO_API_KEY", "x")
        c = Config(enabled=True, schema="bbo-1s,trades")
        assert c.schema_list == ["bbo-1s", "trades"]
        assert "unknown schema" not in c.blocked_by()

    def test_a_typo_is_named(self, monkeypatch):
        monkeypatch.setenv("DATABENTO_API_KEY", "x")
        why = Config(enabled=True, schema="bbo1s").blocked_by()
        assert "unknown schema bbo1s" in why
        assert "bbo-1s" in why, "it should list the real names"

    def test_a_duplicate_is_collapsed_not_rejected(self, monkeypatch):
        monkeypatch.setenv("DATABENTO_API_KEY", "x")
        c = Config(enabled=True, schema="bbo-1s,bbo-1s")
        assert c.schema_list == ["bbo-1s"]
        assert "unknown schema" not in c.blocked_by()

    def test_an_empty_schema_is_refused(self, monkeypatch):
        monkeypatch.setenv("DATABENTO_API_KEY", "x")
        assert "empty" in Config(enabled=True, schema="").blocked_by()

    def test_every_live_schema_name_is_known(self):
        """Pinned against the databento package, not a list I typed."""
        db = pytest.importorskip("databento")
        names = {str(getattr(db.Schema, n))
                 for n in dir(db.Schema) if n.isupper()}
        from liqmap.ofi_service import VALID_SCHEMAS
        assert names <= VALID_SCHEMAS, names - VALID_SCHEMAS
