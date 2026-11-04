"""Fifteen-minute windows rebuilt from raw events.

THE CONSTRAINT THAT SHAPES EVERY TEST: nothing is resampled. The
window grows on each event and locks at the boundary, because that is
what a chart shows and what a trader sees. A batch resample produces
the same closed values and cannot produce the forming one -- so the
tests that matter most are about the FORMING window and about where
the boundary falls.

The others guard the three things most likely to be quietly wrong:
'N'-side prints being dropped, the first event of a window being lost
with the previous book state, and flat counting as agreement.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from liqmap.clockwin import PERIOD_S, ClockWindows, Window, window_start

# The CME attach path needs a configured engine and a database.
from tests.test_cme_panels import client  # noqa: F401,E402
from tests.test_stack_strip import page_js  # noqa: F401,E402

ROOT = Path(__file__).resolve().parent.parent
P = 1_000_000_000
T = P // 4
PX = 6700 * P


def book(cw, ts, px, bsz=500, asz=500):
    return cw.on_book(ts, int(px), bsz, int(px + T), asz)


class TestTheClock:
    def test_label_left_closed_left(self):
        """A bar stamped 10:00 holds up to but not including 10:15."""
        assert window_start(0.0) == 0.0
        assert window_start(899.999) == 0.0
        assert window_start(900.0) == 900.0
        assert window_start(901.0) == 900.0

    def test_a_boundary_closes_the_window(self):
        cw = ClockWindows()
        book(cw, 10.0, PX)
        assert cw.done == []
        finished = book(cw, 900.0, PX)
        assert finished is not None
        assert finished.start_ts == 0.0
        assert finished.closed is True
        assert cw.current.start_ts == 900.0

    def test_a_quiet_window_is_simply_absent(self):
        """Nothing invents an empty bar for a stretch with no events.
        A window that never happened is not a flat one."""
        cw = ClockWindows()
        book(cw, 10.0, PX)
        book(cw, 5000.0, PX)
        assert [w.start_ts for w in cw.done] == [0.0]
        assert cw.current.start_ts == 4500.0

    def test_the_period_is_settable(self):
        cw = ClockWindows(period_s=300.0)
        book(cw, 10.0, PX)
        book(cw, 301.0, PX)
        assert cw.done[0].start_ts == 0.0
        assert cw.current.start_ts == 300.0


class TestItIsIncrementalNotResampled:
    def test_the_forming_window_is_readable_mid_flight(self):
        """THE WHOLE POINT. A resample can produce the closed values
        and cannot produce this one."""
        cw = ClockWindows()
        book(cw, 10.0, PX)
        book(cw, 20.0, PX + 4 * T)
        cur = cw.current
        assert cur.closed is False
        assert cur.books == 2
        assert cur.high > cur.open
        assert cur.close == pytest.approx(PX + 4 * T + T / 2)

    def test_it_updates_on_every_event(self):
        cw = ClockWindows()
        seen = []
        for i in range(5):
            book(cw, 10.0 + i, PX + i * T)
            seen.append(cw.current.close)
        assert len(set(seen)) == 5, seen

    def test_the_high_and_low_are_running(self):
        cw = ClockWindows()
        book(cw, 1.0, PX)
        book(cw, 2.0, PX + 8 * T)
        book(cw, 3.0, PX - 8 * T)
        book(cw, 4.0, PX)
        w = cw.current
        assert w.high == pytest.approx(PX + 8 * T + T / 2)
        assert w.low == pytest.approx(PX - 8 * T + T / 2)
        assert w.open == pytest.approx(PX + T / 2)
        assert w.close == pytest.approx(PX + T / 2)

    def test_no_module_resamples(self):
        """THE CODE, NOT THE PROSE. The docstring names the thing it
        refuses to do, so a plain grep fails on the comment
        explaining the rule."""
        import ast

        src = (ROOT / "liqmap" / "clockwin.py").read_text()
        tree = ast.parse(src)
        called = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                name = getattr(f, "attr", None) or getattr(f, "id", None)
                if name:
                    called.add(name)
        for bad in ("resample", "groupby", "rolling", "agg"):
            assert bad not in called, bad


class TestTheFirstEventOfAWindowIsNotLost:
    def test_the_book_carries_across_the_boundary(self):
        """The OFI of the first event in a window is measured against
        the last book state of the previous one. That change is real
        and belongs to the new window; dropping the previous touch at
        the boundary would discard one event out of every window."""
        cw = ClockWindows()
        book(cw, 10.0, PX, bsz=100)
        book(cw, 20.0, PX, bsz=400)          # +300 in window 0
        assert cw.current.net_ofi == 300
        book(cw, 900.0, PX, bsz=900)         # +500 in window 1
        assert cw.done[0].net_ofi == 300
        assert cw.current.net_ofi == 500

    def test_the_ofi_itself_does_not_carry(self):
        cw = ClockWindows()
        book(cw, 10.0, PX, bsz=100)
        book(cw, 20.0, PX, bsz=900)
        before = cw.current.net_ofi
        book(cw, 900.0, PX, bsz=900)
        assert before > 0
        assert cw.done[0].net_ofi == before
        assert cw.current.net_ofi == 0


class TestUnknownSideIsCountedNotDropped:
    """`dbn.trade_from` returns None for 'N', so every reader
    downstream of it omits those prints silently. A tape full of them
    must not be able to look like a balanced one."""

    def test_an_n_trade_is_counted(self):
        cw = ClockWindows()
        book(cw, 1.0, PX)
        cw.on_trade(2.0, 25.0, "N")
        w = cw.current
        assert w.unknown == 1
        assert w.unknown_size == 25.0
        assert w.trades == 1

    def test_it_does_not_move_the_cvd(self):
        cw = ClockWindows()
        book(cw, 1.0, PX)
        cw.on_trade(2.0, 25.0, "N")
        assert cw.current.net_cvd == 0.0
        assert cw.current.cvd_dir == "flat"

    def test_buys_and_sells_still_net(self):
        cw = ClockWindows()
        book(cw, 1.0, PX)
        cw.on_trade(2.0, 30.0, "B")
        cw.on_trade(3.0, 10.0, "A")
        assert cw.current.net_cvd == 20.0
        assert cw.current.buy_size == 30.0
        assert cw.current.sell_size == 10.0

    def test_the_summary_reports_them(self):
        cw = ClockWindows()
        book(cw, 1.0, PX)
        cw.on_trade(2.0, 5.0, "N")
        book(cw, 900.0, PX)
        assert cw.summary()["unknown_side_trades"] == 1

    def test_the_live_tape_reads_the_raw_side(self):
        """Not through `trade_from`, which would drop them."""
        src = (ROOT / "liqmap" / "cme_tape.py").read_text()
        # Found by what the guard STARTS with, not by the whole line.
        # The condition has had a term added to it once already (the
        # tick bars share this decode), and an exact-line anchor
        # fails on that with its claim still true.
        i = src.index("if self.clockwin is not None")
        # To the end of that block, not a character count: a fixed
        # slice runs on into the next statement, which is the
        # `trade_from` call this is asserting the block avoids.
        j = src.index("t = dbn.trade_from(rec)", i)
        blk = src[i:j]
        assert "dbn._side(rec)" in blk
        assert "on_trade_px" in blk
        assert "trade_from" not in blk


class TestTheComparison:
    def _win(self, mid, ofi, cvd):
        w = Window(start_ts=0.0, open=100.0, close=100.0)
        w.close = 100.0 + (1 if mid == "up" else -1 if mid == "down"
                           else 0)
        w.net_ofi = 1 if ofi == "up" else -1 if ofi == "down" else 0
        w.net_cvd = 1.0 if cvd == "up" else -1.0 if cvd == "down" else 0
        return w

    def test_all_three_up_is_confirmed(self):
        assert self._win("up", "up", "up").agree is True

    def test_all_three_down_is_confirmed(self):
        assert self._win("down", "down", "down").agree is True

    @pytest.mark.parametrize("mid,ofi,cvd", [
        ("up", "up", "down"), ("up", "down", "up"),
        ("down", "up", "up"), ("up", "up", "flat"),
    ])
    def test_any_mismatch_is_flagged(self, mid, ofi, cvd):
        assert self._win(mid, ofi, cvd).agree is False

    def test_three_flats_are_not_agreement(self):
        """Counting them as confirmed would report a dead window as
        the strongest kind there is."""
        assert self._win("flat", "flat", "flat").agree is False

    def test_a_flip_is_between_consecutive_windows(self):
        cw = ClockWindows()
        book(cw, 1.0, PX, bsz=100)
        book(cw, 2.0, PX, bsz=900)           # window 0: OFI up
        book(cw, 901.0, PX, bsz=100)         # window 1 opens
        book(cw, 902.0, PX, asz=900)         # OFI down
        book(cw, 1801.0, PX)                 # close window 1
        assert cw.summary()["ofi_flips"] == 1

    def test_a_flip_is_confirmed_by_its_own_windows_cvd(self):
        cw = ClockWindows()
        book(cw, 1.0, PX, bsz=100)
        book(cw, 2.0, PX, bsz=900)
        cw.on_trade(3.0, 50.0, "B")
        book(cw, 901.0, PX, bsz=900)
        book(cw, 902.0, PX, bsz=100)         # OFI down
        cw.on_trade(903.0, 50.0, "A")        # CVD down: confirmed
        book(cw, 1801.0, PX)
        s = cw.summary()
        assert s["ofi_flips"] == 1
        assert s["flips_cvd_confirmed"] == 1
        assert s["flips_unconfirmed"] == 0

    def test_an_unconfirmed_flip_is_counted_as_such(self):
        cw = ClockWindows()
        book(cw, 1.0, PX, bsz=100)
        book(cw, 2.0, PX, bsz=900)
        book(cw, 901.0, PX, bsz=900)
        book(cw, 902.0, PX, bsz=100)         # OFI down
        cw.on_trade(903.0, 50.0, "B")        # CVD up: not confirmed
        book(cw, 1801.0, PX)
        s = cw.summary()
        assert s["ofi_flips"] == 1
        assert s["flips_cvd_confirmed"] == 0
        assert s["flips_unconfirmed"] == 1


class TestTheOfiIsTheSameFormula:
    def test_it_uses_level_ofi(self):
        """The comparison is worthless if the two versions of OFI are
        computed differently."""
        src = (ROOT / "liqmap" / "clockwin.py").read_text()
        assert "from .ofi import Touch, level_ofi" in src
        assert "level_ofi(self._prev, t)" in src

    def test_a_crossed_book_is_ignored(self):
        cw = ClockWindows()
        book(cw, 1.0, PX)
        n = cw.current.books
        cw.on_book(2.0, int(PX + 4 * T), 100, int(PX), 100)
        assert cw.current.books == n

    def test_it_places_nothing(self):
        for name in ("liqmap/clockwin.py",
                     "scripts/clock_vs_bucket.py"):
            src = (ROOT / name).read_text()
            for bad in ("def submit(", "def execute(", "def place(",
                        "def order("):
                assert bad not in src, (name, bad)


class TestTheTradedCandleIsCarriedToo:
    def test_mid_and_traded_are_separate(self):
        """The mid candle's high and low include levels nothing
        traded at. Both are reported so that is visible."""
        cw = ClockWindows()
        book(cw, 1.0, PX)
        book(cw, 2.0, PX + 20 * T)          # book goes far
        cw.on_trade_px(3.0, 6700.25, 5.0, "B")
        cw.on_trade_px(4.0, 6700.50, 5.0, "B")
        w = cw.current
        assert w.high > w.open
        assert w.last_open == 6700.25
        assert w.last_high == 6700.50
        assert w.last_close == 6700.50
        assert w.last_dir == "up"


class TestItIsWiredIntoTheLiveTape:
    """A state machine nothing feeds is a file that never fills."""

    AUTH = {"Authorization": "Bearer t0ken"}

    def test_the_tape_attaches_it(self, client, monkeypatch):
        from tests.test_cme_panels import _cme

        rt, tape = _cme(monkeypatch)
        assert tape.clockwin is None
        rt.footprint(rule="1m", volume=0, min_volume=0, multiplier=4.0)
        rec = getattr(rt, "_candlerec", None)
        if rec is not None:
            rec.stop()
        assert tape.clockwin is not None

    def test_the_route_serves_the_forming_window(self, client,
                                                  monkeypatch):
        from tests.test_cme_panels import _cme

        rt, tape = _cme(monkeypatch)
        rt.footprint(rule="1m", volume=0, min_volume=0, multiplier=4.0)
        rec = getattr(rt, "_candlerec", None)
        if rec is not None:
            rec.stop()
        # Drive it the way the tape does.
        cw = tape.clockwin
        for i in range(40):
            cw.on_book(i * 10.0, PX, 500 + i * 10, PX + T, 500)
            cw.on_trade_px(i * 10.0 + 1, 6700.0, 5.0, "B")

        d = client.get("/api/clockwindows", params={"coin": "ES"},
                       headers=self.AUTH).json()
        assert d["ok"] is True, d
        assert d["places_orders"] is False
        assert d["candle"].startswith("mid-price")
        assert "MBP-10" in d["ofi"]
        cur = d["current"]
        assert cur is not None
        assert cur["closed"] is False
        assert cur["books"] == 40
        assert cur["net_ofi"] != 0
        assert cur["net_cvd"] > 0

    def test_the_summary_is_served(self, client, monkeypatch):
        from tests.test_cme_panels import _cme

        rt, tape = _cme(monkeypatch)
        rt.footprint(rule="1m", volume=0, min_volume=0, multiplier=4.0)
        rec = getattr(rt, "_candlerec", None)
        if rec is not None:
            rec.stop()
        d = client.get("/api/clockwindows", params={"coin": "ES"},
                       headers=self.AUTH).json()
        for k in ("windows", "agree_pct", "ofi_flips",
                  "flips_cvd_confirmed", "unknown_side_trades"):
            assert k in d["summary"], k

    def test_the_replay_drives_the_same_machine(self):
        """A backtest with its own state machine is a backtest that
        can disagree with the panel about what a window was."""
        src = (ROOT / "scripts" / "clock_vs_bucket.py").read_text()
        assert "from liqmap.clockwin import PERIOD_S, ClockWindows" in src
        assert "resample" not in src.replace(
            "There is no resample here", "")


class TestThePanelDrawsIt:
    """A panel that throws on a forming window is the failure that
    matters here -- the forming one is the whole reason this is event
    driven. Run in node against the served page, because a source
    scan cannot tell that a function raises when it is called.
    """

    @staticmethod
    def _run(payload):
        import json
        import subprocess
        import tempfile

        import pytest as _pt

        from tests.test_stack_strip import HARNESS, NODE, run_js

        if NODE is None:
            _pt.skip("node is not installed")
        return payload

    def test_it_renders_a_forming_window(self, page_js):
        from tests.test_stack_strip import run_js

        out = run_js(page_js, "clock", {
            "ok": True, "period_s": 900,
            "candle": "mid-price from the book",
            "ofi": "level-level at the touch (MBP-10)",
            "current": {"start_ts": 0, "ts_last": 300, "mid_dir": "up",
                        "ticks": 4.0, "ofi_dir": "up", "net_ofi": 1200,
                        "cvd_dir": "up", "net_cvd": 300.0,
                        "agree": True, "books": 900, "trades": 40,
                        "unknown": 0, "closed": False},
            "windows": [], "summary": {"windows": 0, "decided": 0,
                                       "agreed": 0, "agree_pct": 0.0,
                                       "ofi_flips": 0,
                                       "flips_cvd_confirmed": 0,
                                       "flips_unconfirmed": 0,
                                       "flips_per_window": 0.0,
                                       "unknown_side_trades": 0,
                                       "trades": 40}})
        assert "15-MINUTE WINDOW" in out
        assert "all three agree" in out
        assert "no window has closed yet" in out

    def test_it_renders_closed_windows(self, page_js):
        from tests.test_stack_strip import run_js

        w = {"start_ts": 0, "mid_dir": "down", "ticks": -6.0,
             "ofi_dir": "down", "net_ofi": -900, "cvd_dir": "up",
             "net_cvd": 50.0, "agree": False}
        out = run_js(page_js, "clock", {
            "ok": True, "period_s": 900, "candle": "", "ofi": "",
            "current": None, "windows": [w],
            "summary": {"windows": 1, "decided": 1, "agreed": 0,
                        "agree_pct": 0.0, "ofi_flips": 3,
                        "flips_cvd_confirmed": 1,
                        "flips_unconfirmed": 2,
                        "flips_per_window": 3.0,
                        "unknown_side_trades": 0, "trades": 10}})
        assert "flagged" in out
        assert "3 OFI flips" in out
        assert "1 CVD-confirmed" in out

    def test_it_says_when_prints_state_no_side(self, page_js):
        """The thing most likely to be silently wrong, and it is
        wrong everywhere else in this app."""
        from tests.test_stack_strip import run_js

        out = run_js(page_js, "clock", {
            "ok": True, "period_s": 900, "candle": "", "ofi": "",
            "current": None, "windows": [],
            "summary": {"windows": 2, "decided": 2, "agreed": 1,
                        "agree_pct": 0.5, "ofi_flips": 1,
                        "flips_cvd_confirmed": 1,
                        "flips_unconfirmed": 0,
                        "flips_per_window": 0.5,
                        "unknown_side_trades": 77, "trades": 400}})
        assert "77" in out
        assert "no side" in out
        assert "dropped everywhere else" in out

    def test_a_refusal_is_shown_not_thrown(self, page_js):
        from tests.test_stack_strip import run_js

        out = run_js(page_js, "clock",
                     {"ok": False, "why": "the CME engine is not "
                                          "configured"})
        assert "not configured" in out

    def test_it_states_what_it_is_built_from(self, page_js):
        from tests.test_stack_strip import run_js

        out = run_js(page_js, "clock", {
            "ok": True, "period_s": 900,
            "candle": "mid-price from the book",
            "ofi": "level-level at the touch (MBP-10)",
            "current": None, "windows": [], "summary": {}})
        assert "mid-price from the book" in out
        assert "MBP-10" in out
        assert "never resampled" in out


class TestThePanelIsActuallyPolled:
    """A renderer nothing calls draws nothing.

    Found by mutation: deleting the fetch from `ofiTick` left every
    other test in this file green, because they all call the painter
    directly.
    """

    @staticmethod
    def _js_func(name: str) -> str:
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("function " + name + "(")
        depth, j = 0, src.index("{", i)
        for k in range(j, len(src)):
            if src[k] == "{":
                depth += 1
            elif src[k] == "}":
                depth -= 1
                if depth == 0:
                    return src[i:k + 1]
        raise AssertionError(name + " has no closing brace")

    def test_the_ofi_poll_fetches_it(self):
        blk = self._js_func("ofiTick")
        assert "/api/clockwindows" in blk
        assert "paintClockWindows(" in blk

    def test_it_goes_through_the_url_encoder(self):
        """A symbol with a '#' turns the rest of the query into a
        fragment and the token never arrives."""
        blk = self._js_func("ofiTick")
        i = blk.index("/api/clockwindows")
        assert "q({coin:" in blk[i:i + 200]

    def test_a_failure_there_does_not_stop_the_ofi_panel(self):
        """This rides the OFI poll. If it threw, the panel it is
        riding on would go blank with it."""
        blk = self._js_func("ofiTick")
        i = blk.index("paintClockWindows(")
        assert "catch" in blk[i:i + 300]


class TestThePanelCanBeFound:
    """The first version of this was a div inside the footprint
    panel, because it was anchored to `ofiBars` and `ofiBars` lives
    there. It rendered correctly and Chris could not find it -- a
    measurement nobody can locate is one nobody reads.
    """

    @staticmethod
    def _page() -> str:
        return (ROOT / "liqmap" / "web.py").read_text()

    def test_it_is_a_panel_of_its_own(self):
        src = self._page()
        i = src.index('id="clockWin"')
        pan = src.rfind('class="panel', 0, i)
        assert 'id="clockPanel"' in src[pan:pan + 120], \
            "clockWin is nested inside someone else's panel"

    def test_it_has_a_heading(self):
        src = self._page()
        i = src.index('id="clockPanel"')
        blk = src[i:i + 400]
        assert "<h2>" in blk
        assert "Fifteen-minute" in blk

    def test_it_is_on_the_dashboard_tab(self):
        src = self._page()
        i = src.index('id="clockWin"')
        tab = src.rfind('id="tab-', 0, i)
        assert src[tab:tab + 20].startswith('id="tab-dash"')

    def test_it_says_something_before_the_first_poll(self):
        """An empty div under a heading reads as broken."""
        src = self._page()
        i = src.index('id="clockWin"')
        assert "waiting for the first events" in src[i:i + 260]

    def test_it_is_not_inside_the_footprint_panel(self):
        src = self._page()
        fp = src.index('id="fpPanel"')
        cw = src.index('id="clockWin"')
        # The footprint panel closes before this one opens.
        assert cw > fp
        assert src.index('id="clockPanel"') > fp
        nested = src.count('id="fpPanel"', fp, cw)
        assert nested == 1


class TestTheTwoOfisAreLabelledByTheirWindow:
    """Chris watched the footprint candle read UP while this panel
    read DOWN and took it for a misread.

    IT IS NOT ONE. Both are `ofi.level_ofi` over the same raw events
    -- the wiring was checked -- but one is the net over fifteen
    minutes and the other is the bar in progress, so they oppose each
    other often and legitimately. With only one of them on the screen
    that is indistinguishable from the panel reading the wrong thing.
    """

    AUTH = {"Authorization": "Bearer t0ken"}

    def test_the_route_carries_the_bucket_candle(self, client,
                                                  monkeypatch):
        from tests.test_cme_panels import _cme

        rt, tape = _cme(monkeypatch)
        rt.footprint(rule="1m", volume=0, min_volume=0, multiplier=4.0)
        rec = getattr(rt, "_candlerec", None)
        if rec is not None:
            rec.stop()
        cw = tape.clockwin
        for i in range(30):
            cw.on_book(i * 10.0, PX, 500 + i * 10, PX + T, 500)
        d = client.get("/api/clockwindows", params={"coin": "ES"},
                       headers=self.AUTH).json()
        assert "bucket" in d
        for k in ("read", "imbalance", "target", "level"):
            assert k in d["bucket"], k

    def test_the_panel_names_both_windows(self, page_js):
        from tests.test_stack_strip import run_js

        out = run_js(page_js, "clock", {
            "ok": True, "period_s": 900, "candle": "", "ofi": "",
            "bucket": {"read": "up", "imbalance": 900, "target": 2000,
                       "level": 500, "price_dir": "up"},
            "current": {"start_ts": 0, "ts_last": 60, "mid_dir": "up",
                        "ticks": 2.0, "ofi_dir": "down",
                        "net_ofi": -4000, "cvd_dir": "up",
                        "net_cvd": 10.0, "agree": False, "books": 10,
                        "trades": 2, "unknown": 0, "closed": False},
            "windows": [], "summary": {}})
        assert "net over 15 minutes" in out
        assert "bar in progress" in out

    def test_it_says_so_when_they_oppose(self):
        """The sentence that would have saved this round trip."""
        from tests.test_stack_strip import run_js
        import pytest as _pt

        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("function paintClockWindows(")
        blk = src[i:i + 9000]
        assert "not a misread" in blk

    def test_it_is_quiet_when_they_agree(self, page_js):
        from tests.test_stack_strip import run_js

        out = run_js(page_js, "clock", {
            "ok": True, "period_s": 900, "candle": "", "ofi": "",
            "bucket": {"read": "up", "imbalance": 900, "target": 2000,
                       "level": 500, "price_dir": "up"},
            "current": {"start_ts": 0, "ts_last": 60, "mid_dir": "up",
                        "ticks": 2.0, "ofi_dir": "up",
                        "net_ofi": 4000, "cvd_dir": "up",
                        "net_cvd": 10.0, "agree": True, "books": 10,
                        "trades": 2, "unknown": 0, "closed": False},
            "windows": [], "summary": {}})
        assert "not a misread" not in out
        assert "bar in progress" in out

    def test_no_bucket_means_no_line_not_a_crash(self, page_js):
        from tests.test_stack_strip import run_js

        out = run_js(page_js, "clock", {
            "ok": True, "period_s": 900, "candle": "", "ofi": "",
            "current": None, "windows": [], "summary": {}})
        assert "15-MINUTE WINDOW" in out

    def test_both_use_the_same_formula(self):
        """If they ever stop, the disagreement really would be a
        misread and this explanation would be wrong."""
        cwin = (ROOT / "liqmap" / "clockwin.py").read_text()
        obar = (ROOT / "liqmap" / "ofibar.py").read_text()
        assert "level_ofi" in cwin and "level_ofi" in obar
