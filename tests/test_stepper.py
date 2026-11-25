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

def code_only(src: str) -> str:
    """Source with comments and docstrings stripped.

    THREE TESTS IN THIS SUITE HAVE NOW PASSED OR FAILED ON PROSE.
    A comment explaining why `time.time()` is wrong contains
    `time.time()`; a comment explaining that `_journal_step`
    belongs elsewhere contains `_journal_step`. A test asserting
    about code has to be given code.
    """
    out, in_doc = [], False
    for ln in src.splitlines():
        t = ln.strip()
        if t.startswith("#"):
            continue
        if in_doc:
            if '"""' in t:
                in_doc = False
            continue
        if t.startswith('"""') or t.startswith('r"""'):
            # One-line docstrings close on the same line.
            if t.count('"""') == 1:
                in_doc = True
            continue
        out.append(ln)
    return "\n".join(out)


def stepper_src() -> str:
    """The whole `_JournalStepper` class, by indentation.

    THREE TESTS IN THIS FILE READ `src[i:i + 2200]` AND `src[i:i +
    1200]`, and all three failed the moment the class gained a
    longer docstring -- on a change they were meant to approve. A
    fixed character count is a silent assertion about how long the
    code happens to be, and this is the seventh of them across the
    suite.
    """
    src = (ROOT / "liqmap" / "web.py").read_text()
    i = src.index("class _JournalStepper")
    start = src.rfind("\n", 0, i) + 1
    indent = i - start
    lines = src[start:].split("\n")
    out = [lines[0]]
    for line in lines[1:]:
        if line.strip() and (len(line) - len(line.lstrip())) <= indent:
            break
        out.append(line)
    return "\n".join(out)



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
        body = stepper_src()
        assert body.count("except Exception") >= 2
        assert "self.errors += 1" in body

    def test_it_is_a_daemon_so_it_cannot_hold_the_process_open(self):
        assert "daemon=True" in stepper_src()

    def test_it_is_asked_to_stop_on_shutdown(self):
        """A daemon dies with the process anyway; asking first means
        it is never killed halfway through writing a row."""
        src = (ROOT / "liqmap" / "web.py").read_text()
        assert "_st.stop()" in src

    def test_it_reaches_no_broker(self):
        body = stepper_src()
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
        body = stepper_src()
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
        # A stable prefix: the label changed when the 5-minute leg
        # became who is crossing the spread. What this checks is
        # that both cells are drawn.
        assert "5 MINUTE &mdash;" in blk

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


class TestItStepsOnPrintsNotOnAClock:
    """Chris: "there is absolutely no doubt the fixes that you made
    to my chart are all lagging behind extremely bad behind live
    price the real fix is tracking live price not threw snap shots
    but the API from DATABENTO".

    THE FEED WAS NEVER THE SNAPSHOT. `ofi_live` opens a real
    `db.Live()` session and every record goes straight into
    `tape.offer`. The snapshot was this thread: it slept a whole
    second between looks, so a print that should have flipped a
    gate sat untouched while up to a hundred more arrived on top of
    it.
    """

    @staticmethod
    def _stepper(**kw):
        from liqmap import web
        rt = type("RT", (), {"journal": type("J", (), {
            "coins": staticmethod(lambda: [])})()})()
        return web._JournalStepper.__new__(web._JournalStepper), rt

    def test_the_feed_is_the_live_api_not_a_poll(self):
        """Said as a test because it was the premise of the fix,
        and a premise nobody checked is how three days go into the
        wrong place."""
        src = (ROOT / "liqmap" / "ofi_live.py").read_text()
        assert "db.Live()" in src, "the feed is not the streaming API"
        assert "live.subscribe(" in src
        hist = (ROOT / "liqmap" / "ofi_service.py").read_text()
        i = hist.index("db.Historical()")
        # THE ONE HISTORICAL CALL IS THE STARTUP SEED. If it ever
        # ends up on the live path, the chart is replaying.
        assert "seed" in hist[max(0, i - 2000):i].lower()

    def test_it_waits_on_a_print_rather_than_on_the_clock(self):
        body = stepper_src()
        assert "def poke(" in body, "nothing can wake it"
        assert "self._wake.wait(self.every)" in body, \
            "it still sleeps a fixed interval between looks"
        assert "while not self._stop.wait(self.every)" not in body

    def test_poke_does_nothing_but_set_the_event(self):
        """IT RUNS ON THE INGEST THREAD, in front of every other
        consumer of the tape. Work put there is work added to the
        latency of the feed itself."""
        body = stepper_src()
        i = body.index("def poke(")
        code = code_only(body[i:body.index("def run(", i)])
        assert "_journal_step" not in code, "the ingest thread steps"
        assert "self._wake.set()" in code

    def test_the_timeout_ceiling_survives(self):
        """A DEAD TAPE POKES NOBODY. A position with a time stop in
        a market that has stopped printing still has to be looked
        at -- without the timeout the one state that most needs the
        stepper is the one it sleeps through."""
        body = stepper_src()
        assert "self._wake.wait(self.every)" in body
        assert "every: float = 1.0" in body

    def test_there_is_a_floor_and_it_is_after_the_work(self):
        """At a hundred prints a second an unbounded wake runs the
        whole journal step a hundred times a second. Sleeping
        BEFORE the step would add the gap to every step's latency;
        after it bounds the rate and costs the first print
        nothing."""
        body = stepper_src()
        assert "min_gap" in body, "nothing bounds how often it runs"
        a = body.index("_journal_step(self.rt, c, {})")
        b = body.index("self._stop.wait(self.min_gap)")
        assert a < b, "the floor is slept before the work, not after"

    def test_the_tape_is_what_pokes_it(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        assert "_tape0.on_trade_exec(lambda _t: _stepper.poke())" in src, \
            "nothing connects the feed to the stepper"
        i = src.index("_tape0.on_trade_exec")
        # ATTACHED ONCE, where the other tape listeners are wired.
        # Per request it would add a listener on every poll.
        assert "lifespan" in src[max(0, i - 4000):i]

    def test_a_failed_hookup_leaves_the_timer_working(self):
        """A stepper on its 1s ceiling is the old behaviour, which
        worked. A service that will not start because a listener
        could not be attached is not an improvement on it."""
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("_tape0.on_trade_exec")
        seg = src[i:i + 500]
        assert "except Exception" in seg
        assert "timer only" in seg

    def test_stop_wakes_it_so_it_cannot_hang_on_a_dead_tape(self):
        body = stepper_src()
        i = body.index("def stop(")
        fn = body[i:body.index("def poke(", i)]
        assert "self._wake.set()" in fn, \
            "stop leaves it waiting up to a full timeout"


class TestTheStepPushesItsOwnPayload:
    """The page polled `/api/read` on a timer of its own, so what
    you looked at could be a second older than what the rule had
    already decided -- two clocks on the same question."""

    def test_the_frame_is_the_payload_the_rule_used(self):
        """Building a second copy for the screen is how a panel
        comes to disagree with the rule beside it, which this file
        has done with the 5-minute cell and the OFI bar already."""
        body = stepper_src()
        assert "_out = _journal_step(self.rt, c, {})" in body
        assert "self._push(c, _out)" in body
        i = body.index("def _push(")
        fn = code_only(body[i:body.index("def poke(", i)])
        assert '"journal": out' in fn
        assert "_journal_step" not in fn, "the push re-runs the step"

    def test_nobody_watching_costs_nothing(self):
        body = stepper_src()
        i = body.index("def _push(")
        fn = body[i:body.index("def poke(", i)]
        assert 'getattr(hub, "connections", 0)' in fn
        assert "return" in fn

    def test_a_broken_screen_does_not_stop_the_rule(self):
        body = stepper_src()
        i = body.index("def _push(")
        fn = body[i:body.index("def poke(", i)]
        assert "except Exception" in fn

    def test_the_browser_takes_the_frame(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("function applyLiveFrame(")
        fn = src[i:src.index("\nfunction ", i + 10)]
        assert "kind === 'journal'" in fn, "the frame is ignored"
        assert "paintJournal({journal: d.journal})" in fn, \
            "the block is not wrapped the way the painter reads it"
        assert "d.coin === coin()" in fn, \
            "a frame for another market paints this one"

    def test_the_poll_is_not_removed(self):
        """It still paints the chart, the stack and the rungs --
        and it is the fallback if the socket drops, which is the
        one state where a column that had stopped updating would
        be invisible."""
        src = (ROOT / "liqmap" / "web.py").read_text()
        assert "await loadRead();" in src
        i = src.index("let _readBusy")
        blk = src[i:src.index("}, 1000);", i) + len("}, 1000);")]
        assert "}, 1000);" in blk


class TestTheRunningStepper:
    """EXECUTED, NOT SCANNED. `_JournalStepper` is defined inside
    the app factory so it cannot be imported -- but the lifespan
    puts the live instance on the runtime, and that is a real
    object with real threads behind it."""

    @staticmethod
    def _live(client):
        from liqmap import web
        st = getattr(web.runtime(), "journal_stepper", None)
        assert st is not None, "the stepper did not start"
        return st

    def test_it_does_not_spin_when_the_tape_is_quiet(self, client):
        """THE EVENT HAS TO BE CLEARED. Left set, `wait` returns
        immediately on every loop and the stepper runs flat out at
        its floor forever -- twenty full journal steps a second on
        a dead tape, which no source test would notice and which
        looks like the box being slow."""
        import time as _t
        st = self._live(client)
        st.poke()
        _t.sleep(0.15)
        before = st.woke
        _t.sleep(0.40)
        after = st.woke
        assert after - before <= 1, (
            f"it woke {after - before} times with no prints -- "
            "the wake event is never cleared")

    def test_a_poke_wakes_it(self, client):
        import time as _t
        st = self._live(client)
        before = st.woke
        st.poke()
        _t.sleep(0.15)
        assert st.woke > before, "a print did not wake the stepper"
        assert st.pokes > 0

    def test_a_broken_socket_does_not_raise_into_the_rule(self, client):
        """A SCREEN THAT DID NOT UPDATE IS NOT A REASON TO STOP
        MANAGING A POSITION. `_push` returns early when nothing is
        connected, so the handler is only reachable with a hub
        that has connections and then fails."""
        from liqmap import web
        st = self._live(client)
        rt = web.runtime()
        kept = getattr(rt, "ingest_hub", None)

        class _Angry:
            connections = 3

            @staticmethod
            def notify(frame):
                raise RuntimeError("socket is gone")

        try:
            rt.ingest_hub = _Angry()
            st._push("ES", {"settings": {}})
        finally:
            rt.ingest_hub = kept

    def test_the_frame_it_pushes_is_the_block_it_was_handed(self, client):
        from liqmap import web
        st = self._live(client)
        rt = web.runtime()
        kept = getattr(rt, "ingest_hub", None)
        seen = []

        class _Spy:
            connections = 1

            @staticmethod
            def notify(frame):
                seen.append(frame)

        try:
            rt.ingest_hub = _Spy()
            block = {"settings": {"coin": "ES"}, "fusedgate": {"x": 1}}
            st._push("ES", block)
        finally:
            rt.ingest_hub = kept
        assert seen, "nothing was pushed"
        assert seen[0]["kind"] == "journal"
        assert seen[0]["coin"] == "ES"
        assert seen[0]["journal"] is block, \
            "the screen gets a different object from the one the rule used"

    def test_nothing_is_pushed_with_nobody_connected(self, client):
        from liqmap import web
        st = self._live(client)
        rt = web.runtime()
        kept = getattr(rt, "ingest_hub", None)
        seen = []

        class _Quiet:
            connections = 0

            @staticmethod
            def notify(frame):
                seen.append(frame)

        try:
            rt.ingest_hub = _Quiet()
            st._push("ES", {"settings": {}})
        finally:
            rt.ingest_hub = kept
        assert not seen, "a step with nobody watching built a frame"
