"""The rule runs with every browser closed.

Chris: "i need it to continue trading even when the laptop is shut
also i dont see the 1 minute live with the 5 minute".

Everything that evaluates the strategy -- the entry test, the exit
test, the cap, the alert -- hung off `/api/read`, and that route is
only called by a page. Close the tab and the rule was not late, it
was gone: an open position sitting there with nothing watching its
stop.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def py_func(name: str, path: str = "liqmap/web.py") -> str:
    src = (ROOT / path).read_text()
    i = src.index(f"def {name}(")
    start = src.rfind("\n", 0, i) + 1
    indent = i - start
    lines = src[start:].split("\n")
    out = [lines[0]]
    for line in lines[1:]:
        if line.strip() and (len(line) - len(line.lstrip())) <= indent:
            break
        out.append(line)
    return "\n".join(out)


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
    raise AssertionError(f"{name} has no closing brace")


@pytest.fixture
def client(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import liqmap.web as web

    monkeypatch.setenv("LIQMAP_DB", str(tmp_path / "web.db"))
    monkeypatch.setenv("LIQMAP_TOKEN", "step-token")
    monkeypatch.setenv("LIQMAP_AUTO", "false")
    monkeypatch.delenv("LIQMAP_WALLETS", raising=False)
    web.RT = None
    with TestClient(web.create_app()) as c:
        c.headers.update({"Authorization": "Bearer step-token"})
        yield c, web
    web.RT = None


class TestItRunsWithoutABrowser:
    def test_the_stepper_starts_with_the_service(self, client):
        _c, web = client
        st = getattr(web.RT, "journal_stepper", None)
        assert st is not None
        assert st.is_alive()

    def test_it_steps_on_its_own(self, client):
        c, web = client
        c.post("/api/journal/settings",
               params={"coin": "ES", "strategy": "timed"})
        st = web.RT.journal_stepper
        before = st.steps
        time.sleep(2.2)
        assert st.steps > before, "nothing stepped without a request"

    def test_it_follows_the_configured_markets(self, client):
        c, web = client
        assert web.RT.journal.coins() == ()
        c.post("/api/journal/settings",
               params={"coin": "ES", "strategy": "timed"})
        assert "ES" in web.RT.journal.coins()

    def test_a_market_that_raises_does_not_stop_the_thread(self, client):
        """A dead stepper is a position nobody is managing, which is
        the failure this exists to prevent."""
        c, web = client
        blk = py_func("run", "liqmap/web.py") if False else None
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("class _JournalStepper")
        body = src[i:i + 2200]
        assert body.count("except Exception") >= 2
        assert "self.errors += 1" in body

    def test_it_is_a_daemon_so_it_cannot_hold_the_process_open(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("class _JournalStepper")
        assert "daemon=True" in src[i:i + 1200]

    def test_it_is_asked_to_stop_on_shutdown(self):
        """A daemon dies with the process anyway; asking first means
        it is never killed halfway through writing a row."""
        src = (ROOT / "liqmap" / "web.py").read_text()
        assert "_st.stop()" in src

    def test_it_reaches_no_broker(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("class _JournalStepper")
        body = src[i:i + 2200]
        for banned in ("requests.", "httpx", "ironbeam", "order"):
            assert banned not in body.lower(), banned


class TestOneStepAtATime:
    def test_the_step_is_serialised(self):
        """The browser polls once a second and the stepper runs on
        its own thread. Without a lock both can be inside the entry
        test at the same instant: both see no open trade, both see
        the signal, both open one. The cap counts rows already
        written and cannot help with a race."""
        src = (ROOT / "liqmap" / "web.py").read_text()
        assert "_step_lock = threading.Lock()" in src
        blk = py_func("_journal_step")
        assert "with _step_lock:" in blk
        assert "_journal_step_un(" in blk

    def test_the_policy_moved_rather_than_being_duplicated(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        assert src.count("def _journal_step_un(") == 1
        assert src.count("def _journal_step(") == 1

    def test_the_stepper_goes_through_the_lock(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("class _JournalStepper")
        body = src[i:i + 2200]
        assert "_journal_step(self.rt, c, {})" in body
        assert "_journal_step_un(" not in body, \
            "the stepper must not bypass the lock"


class TestTheOneMinuteIsVisible:
    def test_the_route_serves_both_candles(self, client):
        c, _web = client
        d = c.get("/api/fusion", params={"coin": "ES"}).json()
        t = d.get("timed") or {}
        assert "m1" in t and "m5" in t

    def test_it_survives_the_tick_bars_being_absent(self, client):
        """The same shape of bug as the trade column dying with the
        ladder. One missing subsystem must not take out the reading
        that says whether the rule is running."""
        c, _web = client
        d = c.get("/api/fusion", params={"coin": "ES"}).json()
        assert d.get("error"), "the tick bars are expected to be absent"
        assert d.get("timed") is not None
        assert d.get("stepper") is not None

    def test_the_panel_draws_both(self):
        blk = js_func("paintFusion")
        assert "1 MINUTE &mdash; the truth teller" in blk
        assert "5 MINUTE &mdash; the direction" in blk

    def test_the_panel_shows_whether_the_rule_is_running(self):
        blk = js_func("paintFusion")
        assert "running on the server" in blk
        assert "NOT" in blk

    def test_the_stepper_line_survives_the_panel_error_path(self):
        blk = js_func("paintFusion")
        i = blk.index("d.error")
        j = blk.index("return;", i)
        assert "stepping on the server" in blk[i:j]


class TestTheHealthIsShownNotClaimed:
    def test_it_reports_steps_and_age(self, client):
        c, web = client
        c.post("/api/journal/settings",
               params={"coin": "ES", "strategy": "timed"})
        time.sleep(2.2)
        sp = c.get("/api/fusion", params={"coin": "ES"}).json()["stepper"]
        assert sp["running"] is True
        assert sp["steps"] >= 1
        assert sp["age_s"] is not None

    def test_it_reports_errors_rather_than_hiding_them(self, client):
        c, _web = client
        sp = c.get("/api/fusion", params={"coin": "ES"}).json()["stepper"]
        for k in ("running", "steps", "errors", "last_error", "age_s"):
            assert k in sp, k
