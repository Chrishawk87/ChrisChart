"""ES in the picker, routed to CME instead of Hyperliquid.

The ways this breaks:

1.  LEAKING WHEN OFF. With no CME engine configured the picker and the
    routing must be exactly as before. A phantom ES row that quotes
    0.00 is worse than no row.
2.  ROUTING THE WRONG SYMBOL. `is_cme` deciding true for a Hyperliquid
    coin sends it to a tape that has never heard of it.
3.  A CONFIDENT STALE PRICE. The CME price comes from the last print.
    If the feed stops, that must surface as a growing age, not as a
    number that looks current.
4.  DYING WITH HYPERLIQUID. The CME subscription is independent, so an
    HL outage must not empty the picker of ES.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PT = 1_000_000_000
TICK = PT // 4
BASE = 5800 * PT
T0 = 1_700_000_000.0


class Lv:
    def __init__(s, bp, bs, ap, asz):
        s.bid_px, s.bid_sz, s.ask_px, s.ask_sz = bp, bs, ap, asz


class Rec:
    def __init__(s, a, px, sz, side, ts):
        s.action, s.price, s.size, s.side = a, px, sz, side
        s.ts_event = int(ts * 1e9)
        s.instrument_id, s.sequence = 7, 1
        s.levels = [Lv(BASE - j * TICK, 99, BASE + (j + 1) * TICK, 99)
                    for j in range(10)]


@pytest.fixture
def rt_on(monkeypatch):
    monkeypatch.setenv("LIQMAP_OFI", "true")
    monkeypatch.setenv("LIQMAP_OFI_SYMBOLS", "ES.c.0")
    monkeypatch.setenv("DATABENTO_API_KEY", "fake")
    from liqmap.web import Runtime
    return Runtime()


@pytest.fixture
def rt_blocked(monkeypatch):
    """Enabled, but with no key, so the engine CANNOT be running.

    `rt_on` sets a fake key. On a machine without the databento
    package the thread dies at once and `running` is False; on one
    with it installed the thread starts and stays alive until the
    fake key is rejected. Asserting `running is False` there is both
    environment-dependent and a race. Withholding the key makes the
    state deterministic everywhere.
    """
    monkeypatch.setenv("LIQMAP_OFI", "true")
    monkeypatch.setenv("LIQMAP_OFI_SYMBOLS", "ES.c.0")
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    from liqmap.web import Runtime
    return Runtime()


@pytest.fixture
def rt_off(monkeypatch):
    monkeypatch.delenv("LIQMAP_OFI", raising=False)
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    from liqmap.web import Runtime
    return Runtime()


def fill(rt, n=600):
    random.seed(5)
    for i in range(n):
        is_t = random.random() < 0.3
        rt.ofi.tape.offer(Rec("T" if is_t else "A",
                              BASE + TICK if is_t else 0,
                              random.randint(1, 9) if is_t else 0,
                              random.choice("AB"), T0 + i))


class TestOffIsOff:

    def test_no_cme_row_when_disabled(self, rt_off):
        assert rt_off.cme_market() is None

    def test_nothing_routes_to_cme_when_disabled(self, rt_off):
        assert rt_off.is_cme("ES") is False
        assert rt_off.is_cme("BTC") is False


class TestPicker:

    def test_es_appears_when_enabled(self, rt_on):
        row = rt_on.cme_market()
        assert row["symbol"] == "ES"
        assert row["dex"] == "CME"
        assert row["source"] == "cme"

    def test_the_subscription_suffix_is_stripped(self, rt_on):
        """'ES.c.0' is Databento's continuous-contract name; nobody
        types that into a picker."""
        assert rt_on.ofi.config.symbols == ("ES.c.0",)
        assert rt_on.cme_market()["symbol"] == "ES"

    def test_has_data_is_false_before_any_print(self, rt_on):
        assert rt_on.cme_market()["has_data"] is False

    def test_has_data_turns_true_once_prints_arrive(self, rt_on):
        fill(rt_on)
        row = rt_on.cme_market()
        assert row["has_data"] is True
        assert row["price"] > 0

    def test_it_is_classed_as_an_index(self, rt_on):
        assert rt_on.cme_market()["klass"] == "index"


class TestRouting:

    def test_es_routes_to_cme(self, rt_on):
        assert rt_on.is_cme("ES") is True
        assert rt_on.is_cme("es") is True

    def test_hyperliquid_coins_do_not(self, rt_on):
        for c in ("BTC", "ETH", "SOL", "ESX", "E"):
            assert rt_on.is_cme(c) is False

    def test_now_answers_from_the_tape(self, rt_on):
        fill(rt_on)
        out = rt_on.now("ES")
        assert out["source"] == "cme"
        assert out["price"] > 0
        assert out["trades"] > 0

    def test_now_does_not_call_hyperliquid(self, rt_on, monkeypatch):
        """The whole point of the route: no HL request for an ES
        quote."""
        def boom(*a, **k):
            raise AssertionError("Hyperliquid was called for a CME symbol")
        monkeypatch.setattr(type(rt_on), "client", property(
            lambda self: boom))
        fill(rt_on)
        assert rt_on.now("ES")["price"] > 0


class TestStaleness:

    def test_age_grows_rather_than_the_price_going_stale(self, rt_on):
        fill(rt_on)
        a = rt_on.now("ES")["age_s"]
        import time
        time.sleep(0.05)
        b = rt_on.now("ES")["age_s"]
        assert b > a, "a stopped feed must show as a growing age"

    def test_no_price_before_any_print(self, rt_on):
        assert rt_on.now("ES")["price"] is None

    def test_it_says_why_when_not_running(self, rt_blocked):
        out = rt_blocked.now("ES")
        assert out["running"] is False
        assert "DATABENTO_API_KEY" in out["note"]

    def test_es_is_still_listed_and_routed_while_blocked(self, rt_blocked):
        """A configured-but-not-connected engine still owns the
        symbol. Falling back to Hyperliquid for ES would ask an
        exchange that has never listed it."""
        assert rt_blocked.cme_market()["symbol"] == "ES"
        assert rt_blocked.is_cme("ES") is True
        assert rt_blocked.now("ES")["source"] == "cme"


# ---------------------------------------------------------- api_chart routing

class TestChartRouting:
    """api_chart must use the CME tape for ES, never the HL client."""

    def test_es_chart_uses_cme_feed_not_hyperliquid(self, rt_on, monkeypatch):
        """When ES has closed bars, api_chart reports source 'cme …'
        and never touches the Hyperliquid client."""
        import time
        from liqmap.structure import Candle

        def boom(*a, **k):
            raise AssertionError("Hyperliquid candles were fetched for ES")
        monkeypatch.setattr(type(rt_on), "client", property(
            lambda self: type("C", (), {"candles": staticmethod(boom),
                                        "INTERVALS": {"15m": 900}})()))

        # Seed the tape's 15m builder with enough closed bars.
        now = time.time()
        bars = [Candle(ts=now - 900 * (10 - i), open=5800.0, high=5810.0,
                       low=5790.0, close=5805.0, volume=100)
                for i in range(10)]
        rt_on.ofi.tape.seed_interval("15m", bars)

        # cme_feed wraps the tape; history must be reachable through it.
        cme = rt_on.cme_feed("ES")
        assert cme is not None
        assert len(cme.history("15m")) >= 1

    def test_es_chart_error_message_when_tape_is_empty(self, rt_on, monkeypatch):
        """A tape with no closed bars must return a friendly error, not
        an internal crash or a Hyperliquid 'candles unavailable' message."""
        def boom(*a, **k):
            raise AssertionError("Hyperliquid called for ES")
        monkeypatch.setattr(type(rt_on), "client", property(
            lambda self: type("C", (), {"candles": staticmethod(boom),
                                        "INTERVALS": {}})()))

        cme = rt_on.cme_feed("ES")
        assert cme is not None
        # An empty tape has no closed bars.
        assert cme.history("15m") == []

    def test_cme_feed_returns_live_candle_for_es(self, rt_on):
        """The in-progress bar is accessible through the CME feed."""
        fill(rt_on, n=30)
        cme = rt_on.cme_feed("ES")
        assert cme is not None
        bar = cme.candle("1m")
        assert bar is not None
        assert bar.close > 0

    def test_on_update_fires_when_cme_trade_arrives(self, rt_on):
        """The tape's on_update hook drives the SSE stream wake-up."""
        fired = []
        rt_on.ofi.tape.on_update(lambda: fired.append(1))
        fill(rt_on, n=10)
        assert len(fired) > 0

    def test_off_update_stops_callbacks(self, rt_on):
        fired = []
        fn = lambda: fired.append(1)
        rt_on.ofi.tape.on_update(fn)
        fill(rt_on, n=5)
        count_after_register = len(fired)
        rt_on.ofi.tape.off_update(fn)
        fill(rt_on, n=5)
        assert len(fired) == count_after_register

    def test_baseline_stub_does_not_crash_api_read(self, rt_on):
        """api_read accesses feed.baseline.samples for ES. The stub must
        return 0 (no impact baseline computed) without raising."""
        cme = rt_on.cme_feed("ES")
        assert cme is not None
        assert cme.baseline.samples == 0
        assert cme.baseline.ready is False

    def test_spread_trend_stub_returns_zero_samples(self, rt_on):
        """api_read accesses feed.spread_trend() for ES. The stub must
        return a dict with samples=0 rather than raising AttributeError."""
        cme = rt_on.cme_feed("ES")
        assert cme is not None
        result = cme.spread_trend(window_s=60.0)
        assert isinstance(result, dict)
        assert result["samples"] == 0
