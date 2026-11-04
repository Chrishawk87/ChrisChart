"""The N-print bars, and the wiring that puts them on the chart.

Every test here is written so that DELETING the behaviour it names
breaks it. The suite has been bitten repeatedly by tests that
re-implemented the rule instead of calling it, and by fixed-width
source slices that fail when a function grows while their claim stays
true -- both patterns are avoided deliberately below.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from liqmap import tickbar as tb
from liqmap.tickbar import TickBar, TickBars

ROOT = Path(__file__).resolve().parents[1]


def js_func(name: str, path: str = "liqmap/web.py") -> str:
    """A JS function's whole body, by brace matching.

    Same reason as `py_func` in the journal suite: a fixed-width
    slice is a silent assertion about how long a function is.
    """
    src = (ROOT / path).read_text()
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


def py_func(name: str, path: str = "liqmap/web.py") -> str:
    """A python function's whole body, found by indentation."""
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


def feed(b: TickBars, n: int, *, px: float = 6800.0, step: float = 0.0,
         size: float = 1.0, side: str = "B", t0: float = 1000.0,
         dt: float = 1.0) -> list[TickBar]:
    """n prints into `b`, returning whatever bars closed."""
    out = []
    for i in range(n):
        r = b.on_trade(t0 + i * dt, px + i * step, size, side)
        if r is not None:
            out.append(r)
    return out


class TestItDoesNotTrade:
    """The line this package does not cross."""

    def test_no_order_entry_points(self):
        src = (ROOT / "liqmap" / "tickbar.py").read_text()
        tree = ast.parse(src)
        names = {n.name for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for banned in ("submit", "execute", "place", "order",
                       "buy", "sell", "send_order"):
            assert banned not in names, banned

    def test_imports_no_broker(self):
        """Parsed, not greped.

        An earlier test in this suite greped a module for 'ironbeam'
        and failed on the docstring that said it does NOT reach
        Ironbeam. Prose is not code; read the imports.
        """
        tree = ast.parse((ROOT / "liqmap" / "tickbar.py").read_text())
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module:
                mods.add(n.module.split(".")[0])
        assert not (mods & {"ironbeam", "requests", "httpx", "socket",
                            "urllib", "databento"}), mods


class TestTheBarCloses:
    def test_closes_at_exactly_the_count(self):
        b = TickBars(per_bar=10)
        closed = feed(b, 10)
        assert len(closed) == 1
        assert closed[0].trades == 10

    def test_one_short_does_not_close(self):
        b = TickBars(per_bar=10)
        assert feed(b, 9) == []
        assert b.current is not None and b.current.trades == 9
        assert b.last_closed is None

    def test_every_closed_bar_holds_exactly_n(self):
        b = TickBars(per_bar=7)
        closed = feed(b, 7 * 13)
        assert len(closed) == 13
        assert all(c.trades == 7 for c in closed)

    def test_a_new_bar_starts_at_the_next_print(self):
        """Not at the close.

        Starting the next bar eagerly on close would open it at the
        closing print's price and timestamp, so a quiet stretch would
        show a bar that began minutes before its first trade.
        """
        b = TickBars(per_bar=3)
        feed(b, 3)
        assert b.current is None
        b.on_trade(9000.0, 6801.0, 1.0, "B")
        assert b.current is not None
        assert b.current.start_ts == 9000.0
        assert b.current.open == 6801.0


class TestPrintsThatAreNotTrades:
    @pytest.mark.parametrize("px,size", [(0, 5), (-1, 5), (10, 0), (10, -2)])
    def test_zero_price_or_size_does_not_count(self, px, size):
        b = TickBars(per_bar=3)
        assert b.on_trade(1.0, px, size, "B") is None
        assert b.current is None
        assert b.summary()["prints"] == 0

    def test_garbage_does_not_raise_or_count(self):
        b = TickBars(per_bar=3)
        assert b.on_trade("x", "y", "z", "B") is None
        assert b.on_trade(None, None, None, None) is None
        assert b.summary()["prints"] == 0

    def test_a_rejected_print_cannot_close_a_bar(self):
        """The one that matters.

        If a zero-size record advanced the count, bars would close on
        records that never happened and a 0.0 would land in the OHLC.
        """
        b = TickBars(per_bar=2)
        b.on_trade(1.0, 6800.0, 1.0, "B")
        assert b.on_trade(2.0, 0.0, 5.0, "B") is None
        assert b.last_closed is None
        assert b.current.low == 6800.0


class TestUnknownSide:
    def test_n_advances_the_bar(self):
        """Counted, not dropped.

        `dbn.trade_from` returns None for side 'N', so every reader
        downstream of it omits those prints silently. A tick bar that
        did the same would need N ATTRIBUTABLE prints and close at a
        different rate than the panel claims.
        """
        b = TickBars(per_bar=4)
        closed = feed(b, 4, side="N")
        assert len(closed) == 1
        assert closed[0].trades == 4

    def test_n_does_not_move_delta(self):
        b = TickBars(per_bar=4)
        c = feed(b, 4, side="N")[0]
        assert c.delta == 0.0
        assert c.buy_size == 0.0 and c.sell_size == 0.0

    def test_n_is_reported_so_a_delta_can_be_distrusted(self):
        b = TickBars(per_bar=4)
        b.on_trade(1.0, 10.0, 5.0, "B")
        c = feed(b, 3, side="N", size=2.0, t0=2.0)[0]
        assert c.unknown == 3
        assert c.unknown_size == 6.0
        assert b.summary()["unknown_side"] == 3
        assert b.summary()["unknown_pct"] == 75.0

    def test_volume_counts_unknown_prints(self):
        """They are real trades. They moved real size."""
        b = TickBars(per_bar=2)
        c = feed(b, 2, side="N", size=3.0)[0]
        assert c.volume == 6.0


class TestWhatTheBarSays:
    def test_ohlc_is_the_traded_price(self):
        b = TickBars(per_bar=4)
        for t, p in ((1, 10.0), (2, 12.0), (3, 8.0), (4, 11.0)):
            r = b.on_trade(float(t), p, 1.0, "B")
        assert (r.open, r.high, r.low, r.close) == (10.0, 12.0, 8.0, 11.0)

    def test_delta_is_buy_minus_sell(self):
        b = TickBars(per_bar=4)
        b.on_trade(1, 10, 5, "B")
        b.on_trade(2, 10, 3, "B")
        b.on_trade(3, 10, 2, "A")
        c = b.on_trade(4, 10, 1, "A")
        assert c.buy_size == 8.0 and c.sell_size == 3.0
        assert c.delta == 5.0
        assert c.delta_dir == "up"

    def test_seconds_is_first_print_to_last(self):
        b = TickBars(per_bar=3)
        c = feed(b, 3, t0=100.0, dt=2.5)[0]
        assert c.seconds == 5.0

    @pytest.mark.parametrize("o,c,want", [(10, 11, "up"), (11, 10, "down"),
                                          (10, 10, "flat")])
    def test_direction_is_close_against_open(self, o, c, want):
        bar = TickBar(start_ts=0, open=o, high=max(o, c), low=min(o, c),
                      close=c)
        assert bar.direction == want

    def test_flat_price_agrees_with_nothing(self):
        """However lopsided the delta.

        The same rule the clock windows use: flat is not a side, so it
        cannot be half of an agreement.
        """
        bar = TickBar(start_ts=0, open=10, high=10, low=10, close=10,
                      buy_size=900.0)
        assert bar.delta_dir == "up"
        assert bar.direction == "flat"
        assert bar.agree == ""

    def test_flat_delta_agrees_with_nothing(self):
        bar = TickBar(start_ts=0, open=10, high=11, low=10, close=11)
        assert bar.direction == "up"
        assert bar.delta_dir == "flat"
        assert bar.agree == ""

    def test_agreement_needs_both_the_same_way(self):
        up = TickBar(start_ts=0, open=10, high=11, low=10, close=11,
                     buy_size=5.0)
        split = TickBar(start_ts=0, open=10, high=11, low=10, close=11,
                        sell_size=5.0)
        assert up.agree == "up"
        assert split.agree == ""


class TestRows:
    def test_the_forming_bar_is_drawn_and_flagged(self):
        b = TickBars(per_bar=5)
        feed(b, 7)
        rows = b.rows()
        assert len(rows) == 2
        assert rows[0]["closed"] is True and rows[0]["live"] is False
        assert rows[1]["closed"] is False and rows[1]["live"] is True
        assert rows[1]["trades"] == 2

    def test_it_can_be_left_out(self):
        b = TickBars(per_bar=5)
        feed(b, 7)
        assert len(b.rows(include_live=False)) == 1

    def test_rows_carry_the_painters_key_names(self):
        """The reason a tick series can use the existing chart."""
        b = TickBars(per_bar=2)
        feed(b, 2)
        row = b.rows()[0]
        for k in ("ts", "o", "h", "l", "c", "v"):
            assert k in row, k

    def test_oldest_first(self):
        b = TickBars(per_bar=2)
        feed(b, 10, px=6800.0, step=1.0)
        rows = b.rows(include_live=False)
        assert [r["ts"] for r in rows] == sorted(r["ts"] for r in rows)

    def test_limit_takes_the_most_recent(self):
        b = TickBars(per_bar=2)
        feed(b, 20, t0=0.0)
        rows = b.rows(limit=3, include_live=False)
        assert len(rows) == 3
        assert rows[-1]["ts"] == 18.0

    def test_history_is_bounded(self):
        b = TickBars(per_bar=2, keep=4)
        feed(b, 40)
        assert len(b.done) == 4
        assert b.summary()["bars"] == 20      # built, not held


class TestPace:
    def test_it_refuses_under_eight_bars(self):
        """No median worth comparing to.

        'Running hot' off three samples is confident noise, and this
        panel is read while deciding things.
        """
        b = TickBars(per_bar=2)
        feed(b, 14)                           # 7 closed
        p = b.pace()
        assert p["bars_measured"] == 7
        assert "not enough" in p["say"]

    def test_it_speaks_once_there_are_enough(self):
        b = TickBars(per_bar=2)
        feed(b, 40, dt=1.0)
        assert "not enough" not in b.pace()["say"]

    def test_it_projects_rather_than_comparing_raw_elapsed(self):
        """The bug this is built to avoid.

        A bar 10% built has spent 10% of its time. Comparing that
        against a full bar's median would call every young bar fast,
        every single time.
        """
        b = TickBars(per_bar=10)
        feed(b, 100, dt=1.0)                  # 10 bars, ~9s each
        b.on_trade(10_000.0, 6800.0, 1.0, "B")
        b.on_trade(10_001.0, 6800.0, 1.0, "B")
        p = b.pace()
        assert p["fraction"] == pytest.approx(0.2)
        assert p["elapsed_s"] == pytest.approx(1.0)
        # 1s for a fifth of a bar projects to 5s, not 1s.
        assert p["projected_s"] == pytest.approx(5.0)
        assert p["ratio"] < 1.0

    def test_a_faster_tape_reads_faster(self):
        slow = TickBars(per_bar=10)
        feed(slow, 100, dt=10.0)
        slow.on_trade(99_999.0, 6800.0, 1.0, "B")
        slow.on_trade(100_000.0, 6800.0, 1.0, "B")   # 1s for 2/10
        assert slow.pace()["ratio"] < 0.5
        assert "busy" in slow.pace()["say"]

    def test_a_quiet_tape_reads_slower(self):
        b = TickBars(per_bar=10)
        feed(b, 100, dt=1.0)                  # ~9s a bar
        b.on_trade(10_000.0, 6800.0, 1.0, "B")
        b.on_trade(10_100.0, 6800.0, 1.0, "B")   # 100s for 2/10
        assert b.pace()["ratio"] > 2.0
        assert "quiet" in b.pace()["say"]

    def test_no_forming_bar_is_not_an_error(self):
        b = TickBars(per_bar=2)
        feed(b, 20)
        assert b.current is None
        p = b.pace()
        assert p["projected_s"] == 0.0 and p["ratio"] == 0.0


class TestParseInterval:
    def test_a_tick_size_parses(self):
        assert tb.parse_interval("1000t") == 1000
        assert tb.parse_interval(" 233T ") == 233

    @pytest.mark.parametrize("bad", ["15m", "1h", "30m", "1d", "", None,
                                     "t", "tt", "1.5t", "-5t", "0t", "m15",
                                     "1000", "abc"])
    def test_everything_else_is_none(self, bad):
        """The whole safety of routing tick bars through /api/chart.

        Anything that is not a tick size MUST fall through to the
        time-bar path unchanged. A '15m' that parsed to a number here
        would silently replace every clock chart in the service.
        """
        assert tb.parse_interval(bad) is None

    def test_the_boundary_clamps(self):
        assert tb.parse_interval("1t") == tb.MIN_PER_BAR
        assert tb.parse_interval("99999999t") == tb.MAX_PER_BAR

    def test_the_core_does_not_clamp(self):
        """So a measurement of short bars can be written at all."""
        assert TickBars(per_bar=5).per_bar == 5

    def test_label_round_trips(self):
        for n in tb.PER_BAR_CHOICES:
            assert tb.parse_interval(tb.label(n)) == n


class TestTheTapeFeedsIt:
    def test_the_tape_has_the_field(self):
        from liqmap.cme_tape import Tape
        assert "tickbars" in Tape.__dataclass_fields__

    def test_the_record_loop_feeds_every_builder(self):
        src = (ROOT / "liqmap" / "cme_tape.py").read_text()
        assert "for _tb in (self.tickbars or {}).values():" in src
        assert "_tb.on_trade(_ts, _rpx, _rsz, _sd)" in src

    def test_it_is_fed_from_the_raw_side_not_the_decoded_trade(self):
        """`trade_from` drops side 'N'. The raw record does not.

        If this ever moves below `t = dbn.trade_from(rec)` and starts
        using `t.aggressor`, the bars quietly start needing N
        attributable prints instead of N prints.
        """
        src = (ROOT / "liqmap" / "cme_tape.py").read_text()
        feed_at = src.index("_tb.on_trade(_ts, _rpx, _rsz, _sd)")
        decode_at = src.index("t = dbn.trade_from(rec)")
        assert feed_at < decode_at
        assert "_sd = dbn._side(rec)" in src

    def test_the_guard_does_not_require_clock_windows(self):
        """It used to.

        The decode block was guarded on `clockwin is not None`, so
        tick bars attached to a tape with no clock windows would
        never have been fed a single print.
        """
        src = (ROOT / "liqmap" / "cme_tape.py").read_text()
        assert "if self.clockwin is not None or self.tickbars:" in src

    def test_one_builder_per_offered_size_is_created(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        assert "from .tickbar import PER_BAR_CHOICES, TickBars" in src
        assert "int(n): TickBars(per_bar=int(n))" in src


class TestTheChartRoute:
    def test_it_branches_on_a_tick_interval(self):
        blk = py_func("api_chart")
        assert "_tb.parse_interval(interval)" in blk
        assert "if _per is not None:" in blk

    def test_the_branch_returns_before_the_time_bar_path(self):
        blk = py_func("api_chart")
        i = blk.index("if _per is not None:")
        j = blk.index("_candle_rows(coin, interval, n)")
        assert i < j, "the tick branch must not fall into the time path"

    def test_it_draws_no_trade_markers(self):
        """Honest blank beats plausible wrong.

        Marks are recorded against a TIME bar's timestamp. Drawing
        them on a tick series puts each one on whichever bar spanned
        that second, which looks placed and is not.
        """
        blk = py_func("api_chart")
        branch = blk[blk.index("if _per is not None:"):]
        assert '"marks": []' in branch
        assert "marks_note" in branch

    def test_it_reports_no_interval_seconds(self):
        """A tick bar has no fixed duration.

        Claiming one is what lets the painter draw an evenly spaced
        x-axis and quietly misplace everything on it.
        """
        blk = py_func("api_chart")
        branch = blk[blk.index("if _per is not None:"):]
        assert '"interval_s": None' in branch

    def test_it_carries_the_pace_and_the_sizes(self):
        blk = py_func("api_chart")
        branch = blk[blk.index("if _per is not None:"):]
        for want in ('"pace": bld.pace()', '"per_bar": _per',
                     '"sizes": list(_tb.PER_BAR_CHOICES)'):
            assert want in branch, want

    def test_an_unoffered_size_snaps_to_the_nearest_running_builder(self):
        """Rather than creating one that starts empty."""
        blk = py_func("api_chart")
        branch = blk[blk.index("if _per is not None:"):]
        assert "min(books, key=lambda k: abs(int(k) - _per))" in branch

    def test_no_tape_says_so_instead_of_erroring(self):
        blk = py_func("api_chart")
        branch = blk[blk.index("if _per is not None:"):]
        assert "tick bars are built from the ES tape" in branch


class TestTheRouteActuallyServesThem:
    """Called, not greped.

    Every other test in the two classes above reads source text. This
    one builds the app and asks it, because a branch that is present
    in the file and unreachable at runtime passes all of them.
    """

    @pytest.fixture
    def client(self, monkeypatch, tmp_path):
        fastapi = pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient
        import liqmap.web as web

        monkeypatch.setenv("LIQMAP_DB", str(tmp_path / "web.db"))
        monkeypatch.setenv("LIQMAP_TOKEN", "tick-test-token")
        monkeypatch.setenv("LIQMAP_AUTO", "false")
        monkeypatch.delenv("LIQMAP_WALLETS", raising=False)
        web.RT = None
        with TestClient(web.create_app()) as c:
            c.headers.update({"Authorization": "Bearer tick-test-token"})
            yield c, web
        web.RT = None

    @staticmethod
    def _attach(web, builders):
        class _Tape:
            tickbars = builders
            last_px = 6800.25

        class _Config:
            # Only the TIME path reads this -- `is_cme` asks the
            # service for its config, and the tick branch returns
            # before it gets there. Present so the clock-interval
            # test exercises the real path rather than an
            # AttributeError that would pass for the wrong reason.
            enabled = False

        class _Ofi:
            tape = _Tape()
            config = _Config()

        web.RT.ofi = _Ofi()

    def test_a_tick_interval_returns_tick_bars(self, client):
        c, web = client
        b = TickBars(per_bar=4)
        feed(b, 20, px=6800.0, step=0.25, t0=1_700_000_000.0)
        self._attach(web, {4: b})

        r = c.get("/api/chart", params={"coin": "ES", "interval": "4t"})
        assert r.status_code == 200
        d = r.json()
        assert d.get("error") is None, d.get("error")
        assert d["source"] == "tick"
        assert d["per_bar"] == 4
        assert d["interval"] == "4t"
        assert d["interval_s"] is None
        assert d["marks"] == []
        assert len(d["bars"]) == 5          # 5 closed, none forming
        assert d["bars"][0]["trades"] == 4
        assert "pace" in d and "summary" in d

    def test_a_clock_interval_is_untouched_by_the_branch(self, client):
        """The whole safety of sharing the route.

        If this ever returns source 'tick', every chart in the
        service has silently become a tick chart.
        """
        c, web = client
        self._attach(web, {4: TickBars(per_bar=4)})
        r = c.get("/api/chart", params={"coin": "ES", "interval": "15m"})
        assert r.status_code == 200
        assert r.json().get("source") != "tick"

    def test_an_unoffered_size_snaps_and_says_which(self, client):
        c, web = client
        b = TickBars(per_bar=4)
        feed(b, 20)
        self._attach(web, {4: b})
        d = c.get("/api/chart",
                  params={"coin": "ES", "interval": "997t"}).json()
        assert d["per_bar"] == 4
        assert d["interval"] == "4t"

    def test_no_prints_yet_explains_itself(self, client):
        c, web = client
        self._attach(web, {4: TickBars(per_bar=4)})
        d = c.get("/api/chart",
                  params={"coin": "ES", "interval": "4t"}).json()
        assert "no prints counted yet" in (d.get("error") or "")

    def test_no_tape_explains_itself(self, client):
        c, web = client
        web.RT.ofi = None
        d = c.get("/api/chart",
                  params={"coin": "ES", "interval": "1000t"}).json()
        assert "not running here" in (d.get("error") or "")

    def test_the_forming_bar_comes_through_flagged(self, client):
        c, web = client
        b = TickBars(per_bar=4)
        feed(b, 22)                          # 5 closed + 2 forming
        self._attach(web, {4: b})
        d = c.get("/api/chart",
                  params={"coin": "ES", "interval": "4t"}).json()
        assert d["bars"][-1]["live"] is True
        assert d["bars"][-1]["trades"] == 2
        assert d["bars"][-2]["live"] is False


class TestThePageIsNotCached:
    """A cached page is indistinguishable from a failed deploy.

    The whole dashboard is one HTML string with its JS inside it, so
    a browser or edge cache holding an older copy serves the OLD
    PANEL while every API call underneath returns current data. The
    deploy is green, the numbers move, and a control added in the new
    build is simply missing -- which is exactly what a build that
    never landed looks like.
    """

    def test_the_dashboard_forbids_storing(self):
        fastapi = pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient
        import liqmap.web as web
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            import os
            os.environ["LIQMAP_DB"] = str(Path(tmp) / "web.db")
            os.environ["LIQMAP_TOKEN"] = "cache-test-token"
            os.environ["LIQMAP_AUTO"] = "false"
            web.RT = None
            try:
                with TestClient(web.create_app()) as c:
                    r = c.get("/")
                    assert r.status_code == 200
                    cc = r.headers.get("cache-control", "")
                    assert "no-store" in cc, cc
            finally:
                web.RT = None

    def test_the_page_still_carries_the_tick_sizes(self):
        """The thing the cache was hiding."""
        fastapi = pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient
        import liqmap.web as web
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmp:
            os.environ["LIQMAP_DB"] = str(Path(tmp) / "web.db")
            os.environ["LIQMAP_TOKEN"] = "cache-test-token"
            os.environ["LIQMAP_AUTO"] = "false"
            web.RT = None
            try:
                with TestClient(web.create_app()) as c:
                    body = c.get("/").text
                    assert "by prints" in body
                    assert 'value="1000t"' in body
            finally:
                web.RT = None


class TestThePanel:
    def test_the_interval_box_offers_tick_sizes(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index('id="sInt"')
        box = src[i:src.index("</select>", i)]
        for n in (233, 500, 1000, 2000):
            assert f'value="{n}t"' in box, n
        # and the clock intervals are still there
        for n in ("1m", "5m", "15m"):
            assert f">{n}<" in box, n

    def test_the_offered_sizes_match_the_builders(self):
        """A size on the panel with no builder behind it snaps to
        another one and silently draws bars of the wrong length."""
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index('id="sInt"')
        box = src[i:src.index("</select>", i)]
        shown = {int(m) for m in re.findall(r'value="(\d+)t"', box)}
        assert shown == set(tb.PER_BAR_CHOICES)

    def test_the_read_never_follows_the_chart_onto_a_tick_series(self):
        """The stack is built on clock intervals.

        Handing it '1000t' does not switch it to tick bars; it makes
        every row unreadable and the panel blames the data.
        """
        blk = js_func("followChartTf")
        assert "isTickTf(chart.value)" in blk
        assert "return false" in blk

    def test_there_is_a_tick_interval_test_in_the_page(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        assert "function isTickTf(v)" in src
        assert "/^[0-9]+t$/" in src

    def test_the_projection_is_skipped_on_a_tick_series(self):
        blk = js_func("loadChart")
        assert "chartProjOn && !isTickTf($('sInt').value)" in blk

    def test_an_error_is_shown_rather_than_the_stale_chart(self):
        blk = js_func("loadChart")
        i = blk.index("d.error")
        assert "chartNote" in blk[i:i + 400]

    def test_the_note_reports_the_pace(self):
        blk = js_func("loadChart")
        assert "prints a bar" in blk
        assert "pace" in blk


class TestTheChartKeepsMoving:
    """The bug that made the tick chart look like a dead feed.

    The 1-second repaint was gated on `chartData.interval_s`, which a
    tick series deliberately reports as null -- a tick bar has no
    fixed duration. So the canvas was never redrawn: prints arriving,
    the forming bar growing on the server, and nothing moving on
    screen. It read as a frozen tape, and the tape was fine.
    """

    def test_the_repaint_is_not_gated_on_an_interval(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("setInterval(() => {\n  if (!chartData")
        blk = src[i:i + 400]
        assert "interval_s" not in blk, \
            "the repaint loop must not require a fixed bar duration"

    def test_the_repaint_still_requires_a_forming_bar(self):
        """The guard that was always doing the real work.

        Without it a settled chart repaints once a second forever.
        """
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("setInterval(() => {\n  if (!chartData")
        blk = src[i:i + 400]
        assert "bars[bars.length - 1].live" in blk

    def test_there_is_a_refetch_loop(self):
        """A repaint redraws what is already in hand.

        New bars only arrive from loadChart, which used to be called
        by the suggestion poll and by nothing else -- so with auto
        off the chart never updated at all.
        """
        src = (ROOT / "liqmap" / "web.py").read_text()
        assert "_chartBusy" in src
        i = src.index("let _chartBusy")
        blk = src[i:i + 700]
        assert "await loadChart()" in blk

    def test_the_refetch_does_not_stack_requests(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("let _chartBusy")
        blk = src[i:i + 700]
        assert "if (!chartVisible() || _chartBusy) return" in blk
        assert "finally { _chartBusy = false; }" in blk

    def test_a_tick_chart_refetches_more_often_than_a_time_chart(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("let _chartBusy")
        blk = src[i:i + 700]
        assert "isTickTf(iv) ? 2 : 5" in blk

    def test_both_loops_share_one_visibility_test(self):
        """Two loops disagreeing about visibility is how a hidden
        tab keeps fetching forever."""
        src = (ROOT / "liqmap" / "web.py").read_text()
        assert "function chartVisible()" in src
        assert src.count("chartVisible()") >= 3


class TestDrawingOnATickChart:
    """The x-axis maths that a fixed interval would get wrong."""

    def test_bar_position_walks_the_bars_when_there_is_no_interval(self):
        blk = js_func("barPos")
        assert "chartData.interval_s" in blk
        assert "for (let i = 0" in blk, "it must walk, not divide"

    def test_both_directions_exist(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        assert "function barPos(ts, s)" in src
        assert "function posTs(pos, s)" in src

    def test_the_helpers_use_it(self):
        for name in ("tsX", "xTs", "markIndex", "exitIndex"):
            blk = js_func(name)
            assert ("barPos(" in blk or "posTs(" in blk), name

    def test_no_helper_still_divides_by_a_default_interval(self):
        """`interval_s || 900` in a position helper is the bug.

        On a tick chart that silently treats every bar as fifteen
        minutes wide, so a drawn level lands on the wrong bar and
        stays there looking authoritative.
        """
        for name in ("tsX", "xTs", "markIndex", "exitIndex"):
            assert "interval_s || 900" not in js_func(name), name
