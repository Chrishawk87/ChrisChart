"""The /api/ofibars handler, actually run.

WHY THIS FILE EXISTS. tests/test_ofibars_route.py has two hundred
tests of this route and every one of them reads the SOURCE. That
caught a great deal and could not catch the thing that broke: the
handler scaled a key the snapshot had been renamed out of, the
KeyError hit a broad `except`, and the entire price object vanished
off the panel. Two hundred green tests, one dead feature.

So these boot the app, give it a tape, and call the route. They do
not check wording -- the other file does that. They check that the
payload has the keys it claims and that the values arrive.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest                                              # noqa: E402
from fastapi.testclient import TestClient                  # noqa: E402

from liqmap import web                                     # noqa: E402

TOKEN = "t0ken"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
RAW = 250_000_000
BASE = 7_700 * 10 ** 9


@pytest.fixture
def client(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("LIQMAP_DB", str(Path(tmp) / "web.db"))
        monkeypatch.setenv("LIQMAP_TOKEN", TOKEN)
        monkeypatch.setenv("LIQMAP_AUTO", "false")
        monkeypatch.delenv("LIQMAP_WALLETS", raising=False)
        web.RT = None
        with TestClient(web.create_app()) as c:
            yield c
        web.RT = None


def _tape(client, monkeypatch):
    """A CME tape with real bars, a read and a price behind it."""
    import random

    from liqmap.cme_tape import Tape
    from liqmap.flowread import FlowRead, PriceDir
    from liqmap.ofibar import LADDER, OfiBarBuilder
    from liqmap.regime import Regime

    rt = web.runtime()
    monkeypatch.setattr(rt, "is_cme",
                        lambda c: str(c).upper() == "ES")
    tape = Tape()
    tape.ofibars = {t: OfiBarBuilder(tick_size=0.25, threshold=t,
                                     max_s=900.0, min_updates=2)
                    for t in LADDER}
    tape.flowread = FlowRead()
    tape.regime = Regime()
    tape.pricedir = PriceDir()

    rng = random.Random(5)
    bid, ts = BASE, 1_000.0
    for _ in range(40_000):
        if rng.random() < 0.15:
            bid += RAW if rng.random() < 0.5 else -RAW
        ts += 0.01
        for b in tape.ofibars.values():
            b.on_trade(ts, rng.randint(1, 20),
                       "buy" if rng.random() < 0.55 else "sell",
                       float(bid))
            b.on_book(ts, bid, rng.randint(50, 900), bid + RAW,
                      rng.randint(50, 900))
        tape.flowread.on_book(ts, bid, rng.randint(50, 900), bid + RAW,
                              rng.randint(50, 900))
        tape.pricedir.on_trade(ts, float(bid))
        tape.regime.on_book(ts, bid, 100, bid + RAW, 100, float(RAW))
        tape.regime.on_trade(ts, rng.randint(1, 20),
                             "buy" if rng.random() < 0.55 else "sell")
        tape.regime.on_action(ts, "C" if rng.random() < 0.5 else "A",
                              rng.randint(1, 40))
    monkeypatch.setattr(rt.ofi, "tape", tape, raising=False)
    return rt, tape


def _get(client, **kw):
    params = {"coin": "ES", "target": 250, "bars": 60}
    params.update(kw)
    r = client.get("/api/ofibars", params=params, headers=AUTH)
    assert r.status_code == 200, r.text
    return r.json()


class TestTheRouteActuallyRuns:
    def test_it_returns_bars(self, client, monkeypatch):
        _tape(client, monkeypatch)
        d = _get(client)
        assert d["ok"] is True
        assert d["bars"], "no bars came back"

    def test_nothing_in_it_raised_into_a_none(self, client, monkeypatch):
        """Each of these is built inside its own try/except. A broad
        except turns a typo into a missing feature rather than a
        failure, so their presence is the test."""
        _tape(client, monkeypatch)
        d = _get(client)
        for key in ("read", "regime", "price"):
            assert d.get(key) is not None, f"{key} was swallowed"


class TestThePriceSurvivesTheRoute:
    """The bug this file was written for."""

    def test_the_real_price_is_on_the_payload(self, client, monkeypatch):
        _tape(client, monkeypatch)
        d = _get(client)
        px = d["price"]["px"]
        assert px > 0, "the price number disappeared"
        # Scaled into dollars, not left in raw feed units.
        assert 1_000 < px < 100_000, px

    def test_it_survives_every_reversal_setting(self, client,
                                                monkeypatch):
        _tape(client, monkeypatch)
        for rev in range(1, 13):
            d = _get(client, rev=rev)
            assert d["price"] is not None, f"rev={rev} dropped price"
            assert d["price"]["px"] > 0, rev

    def test_the_setting_is_echoed_back(self, client, monkeypatch):
        """The dropdown marks its selection from this. When it read
        the snapshot instead, a missing snapshot made the control
        fall back to its default on every render and look stuck."""
        _tape(client, monkeypatch)
        for rev in (1, 3, 7, 12):
            assert _get(client, rev=rev)["rev_n"] == rev

    def test_the_setting_reaches_the_reading(self, client, monkeypatch):
        _tape(client, monkeypatch)
        for rev in (1, 12):
            d = _get(client, rev=rev)
            assert d["price"]["reverse_ticks"] == pytest.approx(rev)

    def test_out_of_range_is_clamped_not_crashed(self, client,
                                                 monkeypatch):
        _tape(client, monkeypatch)
        for rev in (-5, 0, 999):
            d = _get(client, rev=rev)
            assert d["price"] is not None
            assert 1.0 <= d["rev_n"] <= 12.0


class TestTheOtherSettingsReachTheRoute:
    def test_the_swing_clearance_is_echoed(self, client, monkeypatch):
        _tape(client, monkeypatch)
        for cl in (0, 0.5, 1, 2):
            d = _get(client, clear=cl)
            assert d["clear_n"] == pytest.approx(cl)

    def test_a_bigger_clearance_never_adds_chevrons(self, client,
                                                    monkeypatch):
        """It can only remove them. If it ever added one, the
        clearance would be selecting rather than filtering."""
        _tape(client, monkeypatch)
        counts = [len(_get(client, clear=c)["pivots"])
                  for c in (0, 1, 2, 3)]
        assert counts == sorted(counts, reverse=True), counts

    def test_the_level_is_echoed(self, client, monkeypatch):
        _tape(client, monkeypatch)
        for lv in (100, 300, 500):
            assert _get(client, level=lv)["call_level"] == lv


class TestTheChartZoomsInFarEnoughToBeUseful:
    def test_a_handful_of_bars_is_allowed(self, client, monkeypatch):
        """Twenty was the old floor and it is not zoomed in. At six, a
        2,000-contract candle fills a sixth of the screen."""
        _tape(client, monkeypatch)
        d = _get(client, bars=6)
        assert len(d["bars"]) <= 6
        assert d["window"] == 6

    def test_below_the_floor_is_clamped_not_emptied(self, client,
                                                     monkeypatch):
        _tape(client, monkeypatch)
        for n in (0, 1, -5):
            d = _get(client, bars=n)
            assert d["bars"], f"bars={n} returned an empty chart"

    def test_the_marks_stay_inside_the_window_at_every_zoom(
            self, client, monkeypatch):
        """An index past the end draws a chevron off the chart, or on
        the wrong candle, which is worse."""
        _tape(client, monkeypatch)
        for n in (6, 12, 60, 300):
            d = _get(client, bars=n)
            w = len(d["bars"])
            for key in ("pivots", "provisional", "trade_arrows",
                        "trade_marks"):
                for m in d.get(key) or ():
                    assert 0 <= m["i"] <= w, (key, m["i"], w, n)

    def test_scrolling_back_returns_different_bars(self, client,
                                                    monkeypatch):
        _tape(client, monkeypatch)
        live = _get(client, bars=30, offset=0)
        back = _get(client, bars=30, offset=60)
        assert live["bars"][-1]["ts_close"] != back["bars"][-1]["ts_close"]
        assert back["at_live"] is False

    def test_scrolling_past_the_start_stops_at_the_start(
            self, client, monkeypatch):
        _tape(client, monkeypatch)
        d = _get(client, bars=30, offset=10 ** 9)
        assert d["oldest"] is True
        assert d["bars"], "it scrolled off the end of the data"
