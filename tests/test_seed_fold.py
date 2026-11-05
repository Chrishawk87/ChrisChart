"""The 5m and 15m historical seed, which had never worked.

Chris's service recorded this on every start:

    historical seed (15m): The `schema` was not a valid value of
    Schema, was 'ohlcv-15m'

Databento has ohlcv-1s, 1m, 1h and 1d. The seed built its schema as
f"ohlcv-{interval}", so BOTH the 5m and the 15m asked for something
that does not exist and failed every time, in a field nobody reads.

The visible symptom was the stack reading UNREAD for two and a half
hours after every restart while it rebuilt those bars from live
prints -- /api/read refuses a ladder until the traded interval has
ten closed bars.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from liqmap.ofi_service import _fold
from liqmap.structure import Candle

ROOT = Path(__file__).resolve().parents[1]


def minutes(n: int, t0: float = 1_700_000_000.0) -> list:
    """n one-minute bars, walking up a tick a bar.

    ALIGNED TO THE 15-MINUTE GRID, not just the minute one. Starting
    mid-bucket makes five minutes straddle two 5-minute buckets, and
    then every count below is off by one for a reason that has
    nothing to do with the code under test.
    """
    t0 = float(int(t0 // 900) * 900)
    return [Candle(ts=t0 + i * 60.0, open=100.0 + i * 0.25,
                   high=100.5 + i * 0.25, low=99.75 + i * 0.25,
                   close=100.25 + i * 0.25, volume=10.0 + i)
            for i in range(n)]


class TestTheSchemaBug:
    def test_five_and_fifteen_are_not_requested_as_schemas(self):
        """The line that failed on every start."""
        src = (ROOT / "liqmap" / "ofi_service.py").read_text()
        i = src.index('schema = f"ohlcv-{interval}"')
        # It must be guarded: only native intervals may reach it.
        before = src[max(0, i - 600):i]
        assert "native" in before
        assert "if interval not in native:" in before

    def test_the_native_set_is_what_databento_actually_has(self):
        src = (ROOT / "liqmap" / "ofi_service.py").read_text()
        i = src.index("native = {")
        line = src[i:src.index("}", i) + 1]
        for ok in ("1s", "1m", "1h", "1d"):
            assert ok in line, ok
        for bad in ("5m", "15m", "30m"):
            assert f'"{bad}"' not in line, bad

    def test_one_request_serves_every_derived_interval(self):
        """48h of 1-minute folds into both the fives and the
        fifteens; asking twice is paying twice."""
        src = (ROOT / "liqmap" / "ofi_service.py").read_text()
        assert "want_h = max(hr for _iv, hr in derived)" in src
        # The def line contains the same text, so count calls only.
        calls = src.count("_fetch_ohlcv(client,") - src.count(
            "def _fetch_ohlcv(client,")
        assert calls == 1, calls


class TestTheFold:
    def test_five_minutes_of_minutes_make_one_bar(self):
        out = _fold(minutes(5), 300.0)
        assert len(out) == 1

    def test_thirty_minutes_make_two_fifteens(self):
        assert len(_fold(minutes(30), 900.0)) == 2

    def test_the_open_is_the_first_minutes_open(self):
        mins = minutes(5)
        bar = _fold(mins, 300.0)[0]
        assert bar.open == mins[0].open

    def test_the_close_is_the_last_minutes_close(self):
        mins = minutes(5)
        bar = _fold(mins, 300.0)[0]
        assert bar.close == mins[-1].close

    def test_the_high_and_low_span_the_whole_bucket(self):
        mins = minutes(5)
        bar = _fold(mins, 300.0)[0]
        assert bar.high == max(m.high for m in mins)
        assert bar.low == min(m.low for m in mins)

    def test_volume_adds_up(self):
        mins = minutes(5)
        assert _fold(mins, 300.0)[0].volume == sum(m.volume for m in mins)

    def test_nothing_is_averaged(self):
        """A fold regroups bars that happened. A mean would invent a
        candle the market never printed."""
        mins = minutes(5)
        bar = _fold(mins, 300.0)[0]
        mean_o = sum(m.open for m in mins) / len(mins)
        assert bar.open != pytest.approx(mean_o)

    def test_buckets_sit_on_the_same_grid_the_live_builder_uses(self):
        """Otherwise the history develops a seam at the moment the
        seed stops and the live bars take over."""
        for secs in (300.0, 900.0):
            for bar in _fold(minutes(60), secs):
                assert float(bar.ts) % secs == 0.0

    def test_it_comes_back_in_order(self):
        out = _fold(minutes(60), 300.0)
        assert [b.ts for b in out] == sorted(b.ts for b in out)

    def test_unsorted_input_still_folds_correctly(self):
        mins = minutes(5)
        bar = _fold(list(reversed(mins)), 300.0)[0]
        assert bar.open == mins[0].open
        assert bar.close == mins[-1].close

    def test_a_partial_bucket_is_kept(self):
        """A real bar of a real market. The live builder replaces
        the trailing one as soon as it closes that interval."""
        assert len(_fold(minutes(7), 300.0)) == 2

    @pytest.mark.parametrize("bad", [([], 300.0), (None, 300.0)])
    def test_nothing_in_nothing_out(self, bad):
        assert _fold(*bad) == []

    def test_a_zero_interval_is_refused(self):
        assert _fold(minutes(5), 0.0) == []

    def test_enough_bars_to_clear_the_ten_bar_gate(self):
        """The whole point: /api/read wants ten closed bars before
        it will build a ladder, and at 15m that was 2.5 hours of
        waiting after every restart."""
        out = _fold(minutes(48 * 60), 900.0)
        assert len(out) >= 10, "the seed must clear the gate immediately"


class TestItStaysBestEffort:
    def test_a_failed_fetch_returns_empty_rather_than_raising(self):
        from liqmap.ofi_service import _fetch_ohlcv

        class _Boom:
            class timeseries:
                @staticmethod
                def get_range(**_kw):
                    raise RuntimeError("no")

        assert _fetch_ohlcv(_Boom(), "d", "ES.c.0", "continuous",
                            "ohlcv-1m", 4) == []

    def test_the_helpers_place_no_orders(self):
        tree = ast.parse((ROOT / "liqmap" / "ofi_service.py").read_text())
        names = {n.name for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for banned in ("submit", "place", "send_order"):
            assert banned not in names, banned
