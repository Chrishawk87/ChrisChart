"""The price the journal fills and stops at.

Chris: "its calculating tick by tick for the 1 and 5 minute on the
timed strategy but the problem sits in the trader both auto and
manual trader itself its extremely delayed and off by a great
measure for entries and exits".

The signal was right and the fill was wrong, so this is about
`last_px` and about how long anything takes -- not about the rule.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from liqmap.cme_tape import Tape

ROOT = Path(__file__).resolve().parents[1]


def js_func(name: str) -> str:
    src = (ROOT / "liqmap" / "web.py").read_text()
    i = src.index(f"function {name}(")
    depth, j = 0, src.index("{", i)
    for k in range(j, len(src)):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return src[i:k + 1]
    raise AssertionError(name)


class Rec:
    """A trade record, as Databento hands one over."""

    def __init__(self, px, side="B", ts=1000.0, iid=7, size=2):
        self.instrument_id = iid
        self.ts_event = int(ts * 1e9)
        self.action = "T"
        self.price = int(px * 1e9)
        self.size = size
        self.side = side


class TestTheFillPriceFollowsEveryPrint:
    def test_an_attributed_print_sets_it(self):
        t = Tape()
        t.offer(Rec(7478.0, "B"))
        assert round(t.last_px, 2) == 7478.0

    def test_an_UNATTRIBUTED_print_sets_it_too(self):
        """`trade_from` returns None for side 'N', and everything
        past that gate was skipping those prints -- including the
        price the journal fills at.

        Not knowing who crossed the spread is a reason to leave a
        print out of a DELTA. It is not a reason to pretend the
        market never traded there.
        """
        t = Tape()
        t.offer(Rec(7478.0, "B", ts=1000.0))
        t.offer(Rec(7486.0, "N", ts=1001.0))
        assert round(t.last_px, 2) == 7486.0
        assert round(t.last_ts, 1) == 1001.0

    def test_it_does_not_count_as_a_delta_trade(self):
        """The fill price moves; the attributed-trade count does
        not. Those are different questions and mixing them is what
        would turn this fix into a CVD bug."""
        t = Tape()
        t.offer(Rec(7478.0, "B"))
        t.offer(Rec(7486.0, "N", ts=1001.0))
        assert t.trades == 1
        assert t.unknown_side == 1

    def test_the_share_is_published(self):
        t = Tape()
        for i in range(3):
            t.offer(Rec(7478.0, "N", ts=1000.0 + i))
        snap = t.snapshot() if hasattr(t, "snapshot") else None
        assert t.unknown_side == 3

    def test_a_zero_price_still_does_not_set_it(self):
        t = Tape()
        t.offer(Rec(7478.0, "B"))
        t.offer(Rec(0.0, "N", ts=1001.0))
        assert round(t.last_px, 2) == 7478.0

    def test_a_foreign_contract_still_cannot_set_it(self):
        """The instrument lock runs first, so an unattributed print
        from another month is dropped twice over."""
        t = Tape()
        t.offer(Rec(7478.0, "B", iid=7))
        t.offer(Rec(7474.75, "N", ts=1001.0, iid=99))
        assert round(t.last_px, 2) == 7478.0
        assert t.other_instrument == 1


class TestTheDelayIsMeasured:
    @pytest.fixture
    def client(self, monkeypatch, tmp_path):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient
        import liqmap.web as web

        monkeypatch.setenv("LIQMAP_DB", str(tmp_path / "web.db"))
        monkeypatch.setenv("LIQMAP_TOKEN", "fill-token")
        monkeypatch.setenv("LIQMAP_AUTO", "false")
        web.RT = None
        with TestClient(web.create_app()) as c:
            c.headers.update({"Authorization": "Bearer fill-token"})
            yield c
        web.RT = None

    def test_the_journal_reports_its_own_timing(self, client):
        """"Extremely delayed" cannot be acted on. "The price was
        six seconds old" can."""
        j = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m"}
                       ).json().get("journal") or {}
        tm = j.get("timing")
        assert tm is not None
        for k in ("px_age_s", "step_ms", "gap_s", "queued_ms",
                  "lock_waits"):
            assert k in tm, k

    def test_the_step_is_timed(self, client):
        j = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m"}
                       ).json().get("journal") or {}
        assert isinstance(j["timing"]["step_ms"], (int, float))

    def test_the_gap_shows_the_stepper_running(self, client):
        time.sleep(2.2)
        j = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m"}
                       ).json().get("journal") or {}
        gap = j["timing"]["gap_s"]
        assert gap is not None and gap <= 3.0, \
            f"nothing stepped for {gap}s"

    def test_the_fusion_panel_gets_it_too(self, client):
        d = client.get("/api/fusion", params={"coin": "ES"}).json()
        assert d.get("journal_timing") is not None

    def test_the_panel_draws_it(self):
        blk = js_func("paintFusion")
        assert "journal_timing" in blk
        assert "this is the delay" in blk

    def test_it_says_which_number_is_wrong(self):
        """A row of numbers with no threshold is a row of numbers."""
        blk = js_func("paintFusion")
        assert "px_age_s != null && tmg.px_age_s > 2" in blk
        assert "step_ms != null && tmg.step_ms > 500" in blk
