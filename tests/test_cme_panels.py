"""The panels that used to be crypto-only, serving CME.

THE FAILURE THIS GUARDS AGAINST IS NOT AN ERROR. `rt.client()` is
Hyperliquid. Asked for "ES" it answers about something else, and a
chart full of plausible candles from the wrong instrument cannot be
spotted by looking at it. An empty panel is recoverable; a confident
wrong one is not.

So every test here asks for ES with no crypto feed running, and the
thing being checked is that nothing reaches the crypto venue.
"""
from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest                                              # noqa: E402
from fastapi.testclient import TestClient                  # noqa: E402

from liqmap import web                                     # noqa: E402

WEB = (ROOT / "liqmap" / "web.py").read_text()
TOKEN = "t0ken"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


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


class _Boom:
    """Any touch of the crypto venue is a test failure, not a fallback."""

    def candles(self, *a, **k):
        raise AssertionError("a CME symbol reached the crypto venue")


def _cme(monkeypatch, minutes=1500):
    from liqmap.cme_tape import Tape
    from liqmap.structure import Candle

    rt = web.runtime()
    monkeypatch.setattr(rt, "is_cme", lambda c: str(c).upper() == "ES")
    monkeypatch.setattr(rt, "client", lambda *a, **k: _Boom())
    monkeypatch.setattr(rt, "feed", None, raising=False)

    tape = Tape(symbol="ES")
    rng = random.Random(4)
    px, ts = 6800.0, 1_762_000_000.0
    out = []
    for i in range(minutes):
        o = px
        px += rng.choice((-0.5, -0.25, 0.0, 0.25, 0.5))
        out.append(Candle(ts=ts + i * 60, open=o,
                          high=max(o, px) + 0.25,
                          low=min(o, px) - 0.25, close=px,
                          volume=float(rng.randint(100, 900))))
    for iv in ("1m", "5m", "15m"):
        tape.seed_interval(iv, out)
    monkeypatch.setattr(rt.ofi, "tape", tape, raising=False)
    return rt, tape


def _get(client, path, **params):
    params.setdefault("coin", "ES")
    r = client.get(path, params=params, headers=AUTH)
    assert r.status_code == 200, r.text
    return r.json()


class TestTheStackRunsOnCme:
    """The 4H / 15m / 1m stack under the timeline."""

    def test_it_returns_signals(self, client, monkeypatch):
        _cme(monkeypatch)
        d = _get(client, "/api/mtf")
        assert d["ok"] is True, d.get("why")
        assert d["bars"] > 60
        assert isinstance(d["signals"], list)

    def test_it_never_asks_the_crypto_venue(self, client, monkeypatch):
        """_Boom raises if it is touched, so reaching it fails here
        rather than quietly charting the wrong instrument."""
        _cme(monkeypatch)
        assert _get(client, "/api/mtf")["ok"] is True

    def test_a_tape_with_no_bars_says_so_rather_than_guessing(
            self, client, monkeypatch):
        _cme(monkeypatch, minutes=0)
        d = _get(client, "/api/mtf")
        assert d["ok"] is False
        assert "CME tape" in d["why"] or "one-minute bars" in d["why"]

    def test_a_short_tape_says_how_short(self, client, monkeypatch):
        _cme(monkeypatch, minutes=30)
        d = _get(client, "/api/mtf")
        assert d["ok"] is False
        assert "30" in d["why"], d["why"]


class TestTheOtherPanelsRunOnCmeToo:
    def test_the_profile_does(self, client, monkeypatch):
        _cme(monkeypatch)
        d = _get(client, "/api/profile", interval="15m")
        assert d["ok"] is True, d.get("why")

    def test_the_chart_does(self, client, monkeypatch):
        """This route answers with bars rather than an ok flag."""
        _cme(monkeypatch)
        d = _get(client, "/api/chart", interval="15m")
        assert d.get("bars"), d.get("why")
        assert d["interval_s"] == 900, \
            "the interval came from somewhere other than the label"

    def test_the_candles_do(self, client, monkeypatch):
        _cme(monkeypatch)
        d = _get(client, "/api/ofibars", target=250, bars=60)
        assert d["ok"] is True


class TestTheTapeIsDeepEnoughForTheHigherTimeframes:
    """The stack resamples 4H from MINUTE bars, so what it can see of
    the four-hour chart is bounded by how many minutes are kept."""

    def test_it_holds_more_than_a_single_four_hour_block(self):
        from liqmap.cme_tape import Tape
        t = Tape(symbol="ES")
        b = t.builders["1m"]
        assert b.max_closed >= 1_000, b.max_closed
        hours = b.max_closed / 60.0
        assert hours > 8, f"only {hours:.1f} hours of minutes"

    def test_the_crypto_builder_is_left_alone(self):
        """It refetches deeper history from its venue's REST API, so
        400 is right for it. Raising it there would be a change
        nobody asked for."""
        from liqmap.live import CandleBuilder
        assert CandleBuilder(60.0).max_closed == 400


class TestTheAccessorRefusesTheWrongVenue:
    def test_the_cme_branch_returns_rather_than_falling_through(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("def _candle_rows")
        body = src[i:src.index("def _feed_health")]
        assert "MUST NEVER REACH" in body
        # Measured on the CODE, past the docstring that explains it --
        # the prose names rt.client() before the code reaches it.
        code = body.split('"""', 2)[2]
        cme_at = code.index("rt.cme_feed(")
        rest_at = code.index("rt.client()")
        assert cme_at < rest_at
        assert "return [], (" in code[cme_at:rest_at], \
            "the CME branch can fall through to the crypto venue"


class TestOnePickerDrivesEveryPanel:
    """The bug behind 'the rest of the tabs read Hyperliquid data'.

    The candle panel asked for 'ES' outright. Every other tab read the
    market selector, which defaults to a crypto market. So the app
    showed CME candles beside crypto everything-else, and nothing on
    screen said the two were different markets -- the candles looked
    like proof the whole page was on ES.
    """

    def _tick(self):
        i = WEB.index("async function ofiTick")
        return WEB[i:WEB.index("\n}", i)]

    def test_the_candle_panel_no_longer_hardcodes_a_symbol(self):
        t = self._tick()
        assert "coin=ES" not in t, "the panel still pins its own market"
        assert "'/api/ofibars?coin=' + encodeURIComponent(coin())" in t

    def test_it_asks_for_whatever_the_picker_holds(self):
        t = self._tick()
        assert "coin()" in t

    def test_a_crypto_market_explains_itself(self):
        """'ES is not a CME symbol' is true and useless. The reason the
        panel is blank is that a crypto market is selected."""
        t = self._tick()
        assert "not a CME symbol" in t
        assert "market selector" in t
        assert "every panel, including this one, follows it" in t

    def test_the_selector_opens_on_the_futures_contract(self):
        """Opening on BTC meant every panel but the candles was
        answering about a market the user was not looking at."""
        i = WEB.index("function renderCoins")
        blk = WEB[i:WEB.index("\n}", i)]
        assert "r.source === 'cme'" in blk
        assert "has_data" in blk

    def test_a_chosen_market_still_wins(self):
        """The default must not fight a selection."""
        i = WEB.index("function renderCoins")
        blk = WEB[i:WEB.index("\n}", i)]
        k = blk.index("const still =")
        assert "if (still) {" in blk[k:k + 120]
        assert blk.index("$('coin').value = lastCoin") < blk.index(
            "r.source === 'cme'")

    def test_the_route_accepts_the_dated_contract(self, client,
                                                   monkeypatch):
        """The picker lists ES.Z26, not ES. If the routes only took
        the product name, selecting from the picker would blank every
        CME panel."""
        rt = web.runtime()
        monkeypatch.setattr(rt, "is_cme",
                            lambda c: str(c).upper().startswith("ES"))
        _cme(monkeypatch)
        monkeypatch.setattr(rt, "is_cme",
                            lambda c: str(c).upper().startswith("ES"))
        d = _get(client, "/api/ofibars", coin="ES.Z26", target=250,
                 bars=20)
        assert d["ok"] is True, d.get("why")
