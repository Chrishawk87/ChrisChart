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
        # The CME routes refuse outright when the engine is off, which
        # is correct and is not the state Chris is in -- his candles
        # work. Without this the tests below would pass on the early
        # return and prove nothing.
        monkeypatch.setenv("LIQMAP_OFI", "true")
        monkeypatch.setenv("DATABENTO_API_KEY", "db-test-not-a-real-key")
        monkeypatch.delenv("LIQMAP_WALLETS", raising=False)
        web.RT = None
        with TestClient(web.create_app()) as c:
            yield c
        web.RT = None


class _Boom:
    """Any touch of the crypto venue is a test failure, not a fallback."""

    def candles(self, *a, **k):
        raise AssertionError("a CME symbol reached the crypto venue")


class _Rec:
    """One MBP-10 record, in the shape `dbn._get` reads.

    FED THROUGH THE TAPE'S REAL ENTRY POINT rather than seeded as
    OHLCV, because the difference is the whole bug. `from_candles`
    never marks a reading MEASURED -- only the bar being built from
    actual fills does -- and the cascade reads measured rows only. A
    fixture of seeded candles therefore reproduces "the stack is
    incomplete" for a reason Chris does not have, and would have
    passed whatever the route did.
    """

    __slots__ = ("ts_event", "action", "side", "price", "size",
                 "levels")

    def __init__(self, ts_ns, px_raw, sz, side, bid, ask, bsz, asz,
                 action="T"):
        self.ts_event = ts_ns
        self.action = action
        self.side = side
        self.price = px_raw
        self.size = sz
        self.levels = [_Lvl(bid, bsz, ask, asz)]


class _Lvl:
    __slots__ = ("bid_px", "bid_sz", "ask_px", "ask_sz")

    def __init__(self, bid, bsz, ask, asz):
        self.bid_px, self.bid_sz = bid, bsz
        self.ask_px, self.ask_sz = ask, asz


def _cme(monkeypatch, minutes=400, per_minute=12):
    """A CME tape fed the way the live one is: record by record."""
    import time as _t

    from liqmap.cme_tape import Tape

    rt = web.runtime()
    monkeypatch.setattr(rt, "is_cme",
                        lambda c: str(c).upper().startswith("ES"))
    monkeypatch.setattr(rt, "client", lambda *a, **k: _Boom())
    monkeypatch.setattr(rt, "feed", None, raising=False)

    tape = Tape(symbol="ES")
    rng = random.Random(4)
    tick = 250_000_000
    px = 6800 * 4 * tick // 4            # raw units
    now = _t.time()
    start = now - minutes * 60.0
    step = 60.0 / max(1, per_minute)
    t = start
    while t < now:
        if rng.random() < 0.45:
            px += tick if rng.random() < 0.5 else -tick
        bid, ask = int(px), int(px + tick)
        tape.offer(_Rec(int(t * 1e9), int(px), float(rng.randint(1, 9)),
                        "B" if rng.random() < 0.5 else "A",
                        bid, ask, rng.randint(40, 400),
                        rng.randint(40, 400)))
        t += step
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
        """The number in the message is the number of bars there are.

        Asserted against the tape rather than against the literal the
        fixture was built with: the tape also carries the minute still
        forming, so the two differ by one and the hardcoded figure was
        testing the fixture, not the message.
        """
        _rt, tape = _cme(monkeypatch, minutes=30)
        have = len(list(tape.history("1m", limit=500)))
        d = _get(client, "/api/mtf")
        assert d["ok"] is False
        assert f"only {have} one-minute bars" in d["why"], d["why"]


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


class TestTheStackUnderTheTimeframeReadsCme:
    """Chris saw: UNREAD / no read on 15m, 5m, 1m -- the stack is
    incomplete.

    Three separate things sent every row to the crypto venue, and the
    panel could only report the consequence. The message was accurate
    and said nothing about the cause.
    """

    def _read(self, client, **kw):
        """THE STACK IS SERVED BY /api/read, not /api/flow.

        Worth stating, because the first version of these tests asked
        the wrong route and its empty answer looked exactly like the
        bug they were written to catch.
        """
        params = {"coin": "ES", "interval": "5m",
                  "timeframes": "15m,5m,1m"}
        params.update(kw)
        r = client.get("/api/read", params=params, headers=AUTH)
        assert r.status_code == 200, r.text
        return r.json()

    def test_the_rows_are_there_at_all(self, client, monkeypatch):
        _cme(monkeypatch)
        d = self._read(client)
        assert not d.get("pressure_error"), d["pressure_error"]
        rows = d.get("timeframes") or []
        assert {r["timeframe"] for r in rows} >= {"15m", "5m", "1m"}, rows
        # Measured means the row was built from the bar this service
        # assembled out of fills -- the only kind the stack reads. A
        # row present but unmeasured is the same blank on the panel.
        unmeasured = [r["timeframe"] for r in rows if not r["measured"]]
        assert not unmeasured, f"present but unmeasured: {unmeasured}"
        assert all(r["source"] == "tape" for r in rows), rows

    def test_the_rungs_are_present(self, client, monkeypatch):
        _cme(monkeypatch)
        d = self._read(client)
        casc = d.get("cascade") or {}
        assert "error" not in casc, casc
        rungs = casc.get("rungs") or []
        assert rungs, casc
        missing = [r["timeframe"] for r in rungs if not r.get("present")]
        assert not missing, f"still missing: {missing}"

    def test_it_does_not_say_the_stack_is_incomplete(self, client,
                                                     monkeypatch):
        _cme(monkeypatch)
        why = (self._read(client).get("cascade") or {}).get("why") or ""
        assert "stack is incomplete" not in why, why
        assert "not enough to read" not in why, why

    def test_nothing_asked_the_crypto_venue(self, client, monkeypatch):
        """_Boom raises on any candle fetch, so a single leak fails
        here rather than quietly reading the wrong market."""
        _cme(monkeypatch)
        d = self._read(client)
        assert not d.get("pressure_error"), d["pressure_error"]

    def test_a_label_is_not_gated_on_the_crypto_venues_menu(self, client,
                                                            monkeypatch):
        """The ladder filtered `want` through the Hyperliquid client's
        INTERVALS table. On a market that venue does not list, that
        test emptied the whole stack -- and the panel could only
        report the consequence."""
        _cme(monkeypatch)
        d = self._read(client, timeframes="15m,5m,1m")
        assert len(d.get("timeframes") or []) == 3, d.get("timeframes")
        i = WEB.index("for t in tf_list:")
        assert "client.INTERVALS" not in WEB[i:i + 500]

    def test_every_wanted_timeframe_gets_a_builder(self, client,
                                                    monkeypatch):
        """A tape opened on 1m/15m and asked for the 5m had no builder
        for it, so the row came back missing.

        The guarantee, not the line that provides it: by the end of
        the request the tape is building every timeframe the stack
        asked for, however it got there.
        """
        _rt, tape = _cme(monkeypatch)
        assert "30m" not in tape.builders
        self._read(client, timeframes="30m,15m,1m", interval="15m")
        assert "30m" in tape.builders

    def test_a_brand_new_timeframe_is_absent_not_wrong(self, client,
                                                        monkeypatch):
        """And it does NOT read yet.

        Worth pinning down, because it is the one thing the builder
        cannot buy: a 30m builder started on this request has no
        closed 30m bar behind it, and a row invented for it would be
        a reading of nothing. It appears when its first bar does.
        """
        _cme(monkeypatch)
        rows = self._read(client, timeframes="30m,15m,1m",
                          interval="15m").get("timeframes") or []
        got = {r["timeframe"] for r in rows}
        assert "30m" not in got, got
        assert {"15m", "1m"} <= got, got

    def test_a_timeframe_is_judged_on_its_own_bar(self):
        """`use_feed` is about the interval being traded. Gating the
        whole ladder on it sent every row to the polled fallback the
        moment that one bar was unseeded."""
        i = WEB.index("for tf in want:")
        blk = WEB[i:i + 700]
        assert "bar = feed.candle(tf) if feed else None" in blk
        assert "use_feed and feed" not in blk

    def test_the_fallback_goes_through_the_shared_accessor(self):
        i = WEB.index("for tf in want:")
        blk = WEB[i:i + 2600]
        assert "_candle_rows(" in blk
        assert "client.candles(" not in blk
