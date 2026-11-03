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
        i = src.index("if self.clockwin is not None:")
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
