"""The ingest thread dies, and the chart must say so.

THE FAILURE THIS FILE EXISTS FOR. OfiService._run iterates the live
feed. When that generator ends or raises -- the session closing on a
Friday, a dropped socket, a credential blip -- the thread recorded
the error and exited, and nothing restarted it. The web server stayed
up serving the bars it already had, so the chart looked completely
normal and was frozen. Over a weekend that is two days of a chart
that reads like a quiet market.

Two things had to be true and neither was: the thread comes back,
and until it does the panel says the tape has stopped.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from liqmap import ofi_service as S                        # noqa: E402

WEB = (ROOT / "liqmap" / "web.py").read_text()
SRC = (ROOT / "liqmap" / "ofi_service.py").read_text()


class _Svc:
    """The supervisor, driven without a real feed behind it."""

    RETRY_MIN_S = S.OfiService.RETRY_MIN_S
    RETRY_MAX_S = S.OfiService.RETRY_MAX_S
    POLL_S = 0.02
    _supervise = S.OfiService._supervise
    _watch = S.OfiService._watch

    def __init__(self, blocked="", lifetime=0.05):
        self._stop = threading.Event()
        self._thread = None
        self._sup = None
        self.started_at = None
        self.restarts = 0
        self.died_at = None
        self.retry_at = None
        self.error = ""
        self.runs = 0
        self.lifetime = lifetime

        class _Cfg:
            def __init__(self, why):
                self._why = why

            def blocked_by(self):
                return self._why
        self.config = _Cfg(blocked)

    def _run(self):
        self.runs += 1
        time.sleep(self.lifetime)       # the feed ends, as it does

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.started_at = time.time()
        self._supervise()


class TestTheThreadComesBack:
    def test_a_feed_that_ends_is_restarted(self):
        svc = _Svc()
        svc.RETRY_MIN_S = 0.05
        svc.start()
        time.sleep(1.2)
        svc._stop.set()
        assert svc.runs > 1, "it died and stayed dead"
        assert svc.restarts > 0

    def test_a_deliberate_stop_is_not_fought(self):
        """Stopping must stop. A supervisor that restarts a service
        someone just turned off is worse than none."""
        svc = _Svc()
        svc.RETRY_MIN_S = 0.05
        svc.start()
        time.sleep(0.2)
        svc._stop.set()
        was = svc.runs
        time.sleep(0.6)
        assert svc.runs == was

    def test_it_backs_off_rather_than_hammering(self):
        """The two reasons a feed will not start are a closed market
        and bad credentials. Retrying either in a tight loop burns
        the box and fixes nothing."""
        svc = _Svc(lifetime=0.01)
        svc.RETRY_MIN_S = 0.05
        svc.RETRY_MAX_S = 0.4
        svc.start()
        time.sleep(1.5)
        svc._stop.set()
        # A tight loop at 10ms a run would be north of a hundred.
        assert svc.restarts < 30, svc.restarts

    def test_a_blocked_config_is_reported_not_retried_blindly(self):
        svc = _Svc(blocked="DATABENTO_API_KEY is not set")
        svc.RETRY_MIN_S = 0.05
        svc._thread = None
        svc._supervise()
        time.sleep(0.9)
        svc._stop.set()
        assert svc.restarts == 0, "it restarted into a known block"
        assert "DATABENTO_API_KEY" in svc.error

    def test_recovery_forgets_the_backoff(self):
        """Otherwise the next outage inherits a minute of waiting
        from the last one."""
        assert "wait = self.RETRY_MIN_S" in SRC
        i = SRC.index("def _watch")
        body = SRC[i:SRC.index("def snapshot", i)]
        assert body.count("wait = self.RETRY_MIN_S") >= 2

    def test_the_supervisor_outlives_the_thread_it_watches(self):
        i = SRC.index("def _supervise")
        body = SRC[i:SRC.index("def _watch")]
        assert "self._sup is not None and self._sup.is_alive()" in body


class TestTheChartSaysWhenTheTapeHasStopped:
    def _fn(self):
        i = WEB.index("const feedLine")
        return WEB[i:WEB.index("liveCall = feedLine")]

    def test_it_is_drawn_above_everything(self):
        assert "liveCall = feedLine(d && d.feed)" in WEB

    def test_it_says_the_bars_are_history(self):
        f = self._fn()
        assert "THE FEED IS NOT RUNNING" in f
        assert "nothing on this chart is live" in f

    def test_it_says_how_old_the_newest_bar_is(self):
        """The number that distinguishes a quiet market from a dead
        one, and the one the chart could never show before."""
        f = self._fn()
        assert "last_bar_age_s" in f
        assert "old" in f

    def test_running_but_producing_nothing_is_also_called_out(self):
        """A running thread that has stopped producing is still a
        stopped chart."""
        f = self._fn()
        assert "running but the newest bar is" in f

    def test_a_healthy_feed_draws_no_banner(self):
        f = self._fn()
        assert "return '';" in f

    def test_the_route_carries_it(self):
        assert '"feed": _feed_health(rt, b)' in WEB
        assert "def _feed_health(" in WEB

    def test_the_age_comes_from_the_bars_not_from_the_thread(self):
        """`running` is about the thread. Only the newest bar says
        whether the CHART is current."""
        i = WEB.index("def _feed_health")
        body = WEB[i:WEB.index("@app.get(\"/api/ofibars\"", i)]
        assert "ts_close" in body
        assert "time.time() - newest" in body
