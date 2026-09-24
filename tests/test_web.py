"""Web service tests.

The security posture is the part worth guarding hardest. This service is meant
to sit on a public Railway URL carrying position data and trading
configuration, so "locked unless a token is set" and "no route but /health is
open" are behavioural requirements, not preferences.
"""

import os
import tempfile
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import liqmap.web as web  # noqa: E402
from liqmap.settings import Settings, SettingsStore  # noqa: E402

TOKEN = "test-token-abc"


@pytest.fixture
def client(monkeypatch):
    """A fresh app against a throwaway database."""
    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "web.db")
        monkeypatch.setenv("LIQMAP_DB", db)
        monkeypatch.setenv("LIQMAP_TOKEN", TOKEN)
        monkeypatch.setenv("LIQMAP_AUTO", "false")
        monkeypatch.delenv("LIQMAP_WALLETS", raising=False)
        web.RT = None
        with TestClient(web.create_app()) as c:
            yield c
        web.RT = None


@pytest.fixture
def locked_client(monkeypatch):
    """No token configured at all."""
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("LIQMAP_DB", str(Path(tmp) / "web.db"))
        monkeypatch.delenv("LIQMAP_TOKEN", raising=False)
        monkeypatch.setenv("LIQMAP_AUTO", "false")
        web.RT = None
        with TestClient(web.create_app()) as c:
            yield c
        web.RT = None


# -- health and Railway requirements --------------------------------------

def test_health_is_open_because_railway_needs_it(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_health_leaks_no_configuration(client):
    """Railway's health check is unauthenticated, so this payload must not
    carry settings, positions or the token."""
    body = r"" + client.get("/health").text
    assert TOKEN not in body
    assert "sigma" not in body.lower()
    assert "wallet" not in body.lower()


def test_health_works_even_when_locked(locked_client):
    assert locked_client.get("/health").status_code == 200
    assert locked_client.get("/health").json()["locked"] is True


def test_dashboard_is_served_without_a_token(client):
    """The HTML shell is public; every byte of data inside it is not."""
    r = client.get("/")
    assert r.status_code == 200
    assert "liqmap" in r.text
    assert TOKEN not in r.text


# -- auth ------------------------------------------------------------------

DATA_ROUTES = ["/api/status", "/api/map", "/api/positions", "/api/changes",
               "/api/report", "/api/settings"]


@pytest.mark.parametrize("route", DATA_ROUTES)
def test_every_data_route_demands_a_token(client, route):
    assert client.get(route).status_code == 401


@pytest.mark.parametrize("route", DATA_ROUTES)
def test_wrong_token_is_rejected(client, route):
    assert client.get(route, params={"token": "wrong"}).status_code == 401


def test_bearer_header_is_accepted(client):
    r = client.get("/api/status", headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200


def test_query_token_is_accepted_for_browser_use(client):
    assert client.get("/api/status", params={"token": TOKEN}).status_code == 200


def test_mutating_routes_demand_a_token(client):
    for route in ("/api/sweep", "/api/resolve", "/api/settings",
                  "/api/wallets", "/api/settings/reset"):
        assert client.post(route, json={}).status_code == 401


def test_locked_service_refuses_everything_but_health(locked_client):
    """Without a token the service must not serve data to anyone, rather than
    defaulting to open."""
    for route in DATA_ROUTES:
        assert locked_client.get(route).status_code == 503
    assert locked_client.post("/api/sweep").status_code == 503


# -- status and settings ---------------------------------------------------

def test_status_reports_empty_history_cleanly(client):
    d = client.get("/api/status", params={"token": TOKEN}).json()
    assert d["history"]["sweeps"] == 0
    assert d["wallets"] == 0
    assert isinstance(d["warnings"], list)


def test_status_warns_about_a_missing_wallet_universe(client):
    warnings = client.get("/api/status", params={"token": TOKEN}).json()["warnings"]
    assert any("wallet universe" in w for w in warnings)


def test_settings_round_trip(client):
    r = client.post("/api/settings", params={"token": TOKEN},
                    json={"bucket_bps": 50.0, "horizon_minutes": 120.0})
    assert r.status_code == 200 and r.json()["ok"] is True

    got = client.get("/api/settings", params={"token": TOKEN}).json()["settings"]
    assert got["bucket_bps"] == 50.0
    assert got["horizon_minutes"] == 120.0


def test_absurd_sigma_is_rejected_not_stored(client):
    """A bad sigma would corrupt every baseline computed afterwards, and the
    damage would not show up until the validation report read as nonsense."""
    r = client.post("/api/settings", params={"token": TOKEN},
                    json={"sigma_per_min": 0.5})
    assert r.status_code == 400
    assert any("PER-MINUTE" in p for p in r.json()["problems"])

    still = client.get("/api/settings", params={"token": TOKEN}).json()["settings"]
    assert still["sigma_per_min"] != 0.5


def test_negative_values_are_rejected(client):
    for payload in ({"sigma_per_min": -1}, {"bucket_bps": 0},
                    {"horizon_minutes": -5}, {"max_distance_pct": 2.0}):
        assert client.post("/api/settings", params={"token": TOKEN},
                           json=payload).status_code == 400


def test_unknown_settings_keys_are_ignored(client):
    r = client.post("/api/settings", params={"token": TOKEN},
                    json={"nonsense_key": 123, "bucket_bps": 30.0})
    assert r.status_code == 200
    assert "nonsense_key" not in r.json()["settings"]
    assert r.json()["settings"]["bucket_bps"] == 30.0


def test_settings_reset_restores_defaults(client):
    client.post("/api/settings", params={"token": TOKEN}, json={"bucket_bps": 99.0})
    r = client.post("/api/settings/reset", params={"token": TOKEN})
    assert r.json()["settings"]["bucket_bps"] == Settings.defaults().bucket_bps


def test_wallets_accept_a_list_or_a_blob(client):
    r = client.post("/api/wallets", params={"token": TOKEN},
                    json={"wallets": ["0xaaa", "0xbbb", "not-an-address"]})
    assert r.json()["count"] == 2

    r = client.post("/api/wallets", params={"token": TOKEN},
                    json={"wallets": "0x111,0x222\n0x333"})
    assert r.json()["count"] == 3


# -- graceful behaviour with no data --------------------------------------

def test_map_reports_no_sweep_rather_than_crashing(client):
    d = client.get("/api/map", params={"token": TOKEN, "coin": "BTC"}).json()
    # The message has to say what to DO, not just that there is nothing:
    # "no sweep recorded yet" reads identically for a coin you never
    # configured, a coin nobody holds, and a service that has never run.
    assert "Nothing has been swept yet" in d["error"]
    assert "Harvest wallets" in d["error"]


def test_positions_reports_no_sweep(client):
    d = client.get("/api/positions", params={"token": TOKEN}).json()
    assert "Nothing has been swept yet" in d["error"]


def test_report_explains_why_it_is_empty(client):
    text = client.get("/api/report", params={"token": TOKEN}).text
    assert "No resolved observations yet" in text


def test_changes_is_empty_not_broken(client):
    d = client.get("/api/changes", params={"token": TOKEN, "hours": 24}).json()
    assert d["changes"] == []
    assert "no position changes" in d["text"]


def test_sweep_without_wallets_fails_with_a_useful_message(client):
    r = client.post("/api/sweep", params={"token": TOKEN, "coin": "BTC"})
    assert r.status_code == 400
    assert "wallet universe" in r.json()["error"]


# -- data flows through once a sweep exists -------------------------------

def test_recorded_sweep_surfaces_through_the_api(client, monkeypatch):
    """Write a sweep straight into history, then confirm the endpoints render
    it -- no network needed."""
    from liqmap.bucket import Position

    rt = web.runtime()
    spot = 100_000.0
    positions = [
        Position("0xa", "BTC", 1.0, spot, spot * 0.96, 5_000_000.0, 10,
                 "cross", unrealized_pnl=50_000.0, margin_used=500_000.0,
                 account_value=20_000_000.0),
        Position("0xb", "BTC", -1.0, spot, spot * 1.03, 3_000_000.0, 20,
                 "isolated", unrealized_pnl=-20_000.0, margin_used=150_000.0,
                 account_value=1_000_000.0),
    ]
    rt.history.record_sweep("BTC", spot, positions)

    m = client.get("/api/map", params={"token": TOKEN, "coin": "BTC"}).json()
    assert m["spot"] == spot
    assert m["raw"]["clusters"]
    assert m["weighted"]["total_notional"] < m["raw"]["total_notional"]
    assert "<<< spot" in m["text"]["raw"]

    p = client.get("/api/positions", params={"token": TOKEN, "coin": "BTC"}).json()
    assert len(p["positions"]) == 2
    first = p["positions"][0]
    for field in ("entry", "liquidation", "unrealized_pnl", "account_value",
                  "carry_annualised", "survivability", "fragility"):
        assert field in first


def test_defended_appears_in_the_changes_endpoint(client):
    """End to end for the signal this was all for."""
    from liqmap.bucket import Position

    rt = web.runtime()
    spot = 100_000.0

    def p(liq):
        return Position("0xa", "BTC", 1.0, spot, liq, 5_000_000.0, 10,
                        account_value=20_000_000.0)

    rt.history.record_sweep("BTC", spot, [p(spot * 0.95)])
    rt.history.record_sweep("BTC", spot, [p(spot * 0.85)])

    d = client.get("/api/changes",
                   params={"token": TOKEN, "coin": "BTC", "hours": 24}).json()
    kinds = [c["kind"] for c in d["changes"]]
    assert "DEFENDED" in kinds
    assert d["conviction_flow"]["committing_notional"] > 0


def test_changes_can_be_filtered_by_kind_over_http(client):
    from liqmap.bucket import Position

    rt = web.runtime()
    spot = 100_000.0
    a = Position("0xa", "BTC", 1.0, spot, spot * 0.95, 1e6, 10, account_value=1e7)
    b = Position("0xb", "BTC", 1.0, spot, spot * 0.95, 1e6, 10, account_value=1e7)
    a2 = Position("0xa", "BTC", 1.0, spot, spot * 0.85, 1e6, 10, account_value=1e7)
    b2 = Position("0xb", "BTC", 3.0, spot, spot * 0.95, 3e6, 10, account_value=1e7)

    rt.history.record_sweep("BTC", spot, [a, b])
    rt.history.record_sweep("BTC", spot, [a2, b2])

    d = client.get("/api/changes", params={"token": TOKEN, "coin": "BTC",
                                           "kind": "DEFENDED"}).json()
    assert [c["kind"] for c in d["changes"]] == ["DEFENDED"]


# -- volume warning --------------------------------------------------------

def test_ephemeral_database_produces_a_loud_warning(monkeypatch):
    """The single most expensive Railway mistake: no mounted volume, so every
    deploy silently destroys the history."""
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.delenv("LIQMAP_DB", raising=False)
        monkeypatch.setenv("LIQMAP_TOKEN", TOKEN)
        monkeypatch.chdir(tmp)
        web.RT = None
        rt = web.runtime()
        notes = web.startup_report(rt)
        assert any("ephemeral" in n and "DESTROYED" in n for n in notes)
        web.RT = None


def test_explicit_db_path_suppresses_the_warning(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("LIQMAP_DB", str(Path(tmp) / "x.db"))
        monkeypatch.setenv("LIQMAP_TOKEN", TOKEN)
        web.RT = None
        notes = web.startup_report(web.runtime())
        assert not any("ephemeral" in n for n in notes)
        web.RT = None


def test_locked_service_is_warned_about_at_startup(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("LIQMAP_DB", str(Path(tmp) / "x.db"))
        monkeypatch.delenv("LIQMAP_TOKEN", raising=False)
        web.RT = None
        notes = web.startup_report(web.runtime())
        assert any("LIQMAP_TOKEN is unset" in n for n in notes)
        web.RT = None


# -- bootstrap harvest -----------------------------------------------------

def test_harvest_requires_a_token(client):
    assert client.post("/api/harvest").status_code == 401
    assert client.get("/api/harvest").status_code == 401


def test_harvest_status_starts_idle(client):
    d = client.get("/api/harvest", params={"token": TOKEN}).json()
    assert d["running"] is False
    assert d["wallets"] == 0


def test_concurrent_harvests_are_refused(client, monkeypatch):
    """Two harvests would fight over the same websocket budget and neither
    would finish cleanly."""
    rt = web.runtime()
    rt.harvest["running"] = True
    try:
        r = client.post("/api/harvest", params={"token": TOKEN, "minutes": 1})
        assert r.status_code == 409
        assert "already running" in r.json()["error"]
    finally:
        rt.harvest["running"] = False


def test_harvest_duration_is_clamped(client, monkeypatch):
    """A typo of 10000 minutes should not pin a thread for a week."""
    started = {}

    def fake(self, minutes, then_sweep=True, coins=None):
        started["minutes"] = minutes
        started["coins"] = coins
        return {"ok": True, "running": False, "found": 0}

    monkeypatch.setattr(web.Runtime, "start_harvest", fake)
    client.post("/api/harvest", params={"token": TOKEN, "minutes": 10000})
    assert started["minutes"] == 180.0

    client.post("/api/harvest", params={"token": TOKEN, "minutes": 0.01})
    assert started["minutes"] == 0.5


def test_status_includes_harvest_state(client):
    d = client.get("/api/status", params={"token": TOKEN}).json()
    assert "harvest" in d
    assert d["harvest"]["running"] is False


def test_dashboard_guides_a_first_run(client):
    """An empty dashboard with no explanation is the worst first impression
    for a tool that needs data before it can say anything."""
    html = client.get("/").text
    assert "Harvest wallets" in html
    assert "No wallet universe yet" in html
    assert "cannot appear until the next one" in html


# --------------------------------------------------------------------------
# liquidity
# --------------------------------------------------------------------------

def _fake_book(coin="BTC"):
    from liqmap.flow import Book, Level
    bids = [Level(50_000 - i * 5, 0.4) for i in range(30)]
    asks = [Level(50_010 + i * 5, 0.4) for i in range(30)]
    bids.append(Level(49_800, 20.0))          # a deliberate shelf
    return Book(coin=coin, ts=0.0, bids=bids, asks=asks)


class _FakeClient:
    def __init__(self, book=None, fail=False):
        self._book = book
        self.fail = fail

    def book(self, coin):
        if self.fail:
            raise RuntimeError("exchange unreachable")
        return self._book if self._book is not None else _fake_book(coin)


def test_liquidity_requires_a_token(client):
    assert client.get("/api/liquidity").status_code == 401


def test_liquidity_prices_entry_and_exit_without_a_watch(client):
    """Slippage is exact arithmetic on one book read, so it must not be gated
    behind starting a watch and waiting."""
    web.runtime()._client = _FakeClient()
    d = client.get(f"/api/liquidity?coin=BTC&size=250000&token={TOKEN}").json()

    assert d["book"]["bid"] == 50_000 and d["book"]["ask"] == 50_010
    assert d["book"]["spread_bps"] == pytest.approx(2.0, rel=1e-3)
    c = d["cost"]
    assert c["entry_bps"] > 0 and c["exit_bps"] > 0
    assert c["round_trip_bps"] == pytest.approx(c["entry_bps"] + c["exit_bps"])
    assert not c["exhausted"]
    assert "absorption" not in d          # no watch running, so no verdict


def test_liquidity_finds_the_shelf(client):
    web.runtime()._client = _FakeClient()
    d = client.get(f"/api/liquidity?coin=BTC&size=0&token={TOKEN}").json()
    shelves = d["book"]["shelves"]
    assert any(s["px"] == 49_800 and s["side"] == "buy" for s in shelves)


def test_liquidity_reports_a_size_the_book_cannot_fill(client):
    web.runtime()._client = _FakeClient()
    d = client.get(f"/api/liquidity?size=999000000&token={TOKEN}").json()
    assert d["cost"]["exhausted"] is True


def test_liquidity_reports_an_unreachable_exchange_rather_than_500ing(client):
    web.runtime()._client = _FakeClient(fail=True)
    d = client.get(f"/api/liquidity?token={TOKEN}").json()
    assert "exchange unreachable" in d["error"]


def test_liquidity_handles_an_empty_book(client):
    from liqmap.flow import Book
    web.runtime()._client = _FakeClient(book=Book(coin="BTC", ts=0.0))
    d = client.get(f"/api/liquidity?token={TOKEN}").json()
    assert d["error"] == "book came back empty"


def test_watch_rejects_a_missing_level(client):
    r = client.post(f"/api/watch?coin=BTC&level=0&token={TOKEN}")
    assert r.status_code == 409
    assert "positive price" in r.json()["detail"]


def test_watch_refuses_to_start_a_second_one(client):
    rt = web.runtime()
    rt.watch_info.update({"running": True, "coin": "BTC", "level": 50_000})
    try:
        r = client.post(f"/api/watch?coin=ETH&level=3000&token={TOKEN}")
        assert r.status_code == 409
        assert "already watching" in r.json()["detail"]
    finally:
        rt.watch_info["running"] = False


def test_stop_is_safe_with_nothing_running(client):
    r = client.post(f"/api/watch?stop=true&token={TOKEN}")
    assert r.status_code == 200
    assert r.json()["running"] is False


def test_dashboard_carries_the_liquidity_panel(client):
    html = client.get("/").text
    for marker in ("loadLiquidity", "startWatch", "liqSize", "wLevel"):
        assert marker in html


# --------------------------------------------------------------------------
# multi-coin: one wallet read must populate every coin they hold
# --------------------------------------------------------------------------

def _pos(wallet, coin, long=True, entry=100.0, pnl=1_000.0, notional=1_000_000.0):
    from liqmap.bucket import Position
    return Position(
        wallet=wallet, coin=coin, szi=1.0 if long else -1.0, entry_px=entry,
        liquidation_px=entry * (0.9 if long else 1.1),
        position_value=notional, leverage=10.0, leverage_type="cross",
        unrealized_pnl=pnl, margin_used=notional / 10, account_value=5_000_000.0)


@pytest.fixture
def multi_coin(client, monkeypatch):
    """Wallets holding four coins between them."""
    rt = web.runtime()
    rt.wallets = [f"0x{i:040x}" for i in range(5)]

    mids = {"BTC": 100_000.0, "ETH": 3_000.0, "SOL": 150.0,
            "DOGE": 0.20, "HYPE": 25.0}
    positions = []
    for i in range(5):                       # everyone holds BTC
        positions.append(_pos(f"0x{i:040x}", "BTC", entry=99_000.0))
    for i in range(4):                       # four hold ETH
        positions.append(_pos(f"0x{i:040x}", "ETH", entry=2_900.0))
    for i in range(3):                       # three hold SOL
        positions.append(_pos(f"0x{i:040x}", "SOL", entry=140.0))
    positions.append(_pos("0x" + "0" * 40, "DOGE", entry=0.19))   # only one

    class FakeClient:
        def all_mids(self):
            return dict(mids)

    rt._client = FakeClient()
    monkeypatch.setattr(web, "startup_report", lambda _rt: [])
    import liqmap.hl as hl
    monkeypatch.setattr(hl, "sweep_positions",
                        lambda client, wallets, on_progress=None: (positions, 0))
    return rt


def test_one_sweep_records_every_coin_the_wallets_hold(multi_coin, client):
    """THE BUG: clearinghouseState returns a wallet's whole portfolio, but the
    sweep threw away every coin except the one it was asked about. Getting a
    second coin meant re-reading all the wallets for data already in hand —
    so in practice only the coin you asked for ever had anything."""
    r = client.post(f"/api/sweep?coin=BTC&token={TOKEN}").json()
    assert r["ok"]
    assert set(r["coins_recorded"]) >= {"BTC", "ETH", "SOL"}
    assert r["coin"] == "BTC"               # headline is still what you asked


def test_a_coin_nobody_holds_is_not_given_its_own_row(multi_coin, client):
    """One stray position is not a market. DOGE has a single holder and is
    not configured, so it should not get a sweep row of its own."""
    r = client.post(f"/api/sweep?token={TOKEN}").json()
    assert "DOGE" not in r["coins_recorded"]


def test_asking_for_a_coin_records_it_even_if_barely_held(multi_coin, client):
    r = client.post(f"/api/sweep?coin=DOGE&token={TOKEN}").json()
    assert "DOGE" in r["coins_recorded"]


def test_asking_for_a_symbol_the_exchange_does_not_list_says_so(multi_coin, client):
    r = client.post(f"/api/sweep?coin=NOTACOIN&token={TOKEN}").json()
    assert "NOTACOIN" not in r["coins_recorded"]
    assert "no mid price" in r["warning"]
    assert r["coins_recorded"], "the other coins must still be recorded"


def test_a_held_but_unrecorded_coin_warns_about_the_universe(multi_coin, client):
    """HYPE is listed on the exchange but none of these wallets hold it. The
    fix is the wallet universe, so the message has to say that."""
    r = client.post(f"/api/sweep?coin=HYPE&token={TOKEN}").json()
    assert "HYPE" in r["coins_recorded"]     # it is a real market, so recorded
    assert "hold" in r["warning"]


def test_other_coins_are_readable_after_one_sweep(multi_coin, client):
    client.post(f"/api/sweep?coin=BTC&token={TOKEN}")
    for c in ("ETH", "SOL"):
        d = client.get(f"/api/consensus?coin={c}&token={TOKEN}").json()
        assert "error" not in d, f"{c} should have data after a single sweep"
        assert d["n_traders"] > 0


def test_coins_endpoint_separates_having_data_from_being_configured(multi_coin, client):
    client.post(f"/api/sweep?token={TOKEN}")
    d = client.get(f"/api/coins?token={TOKEN}").json()
    names = [r["coin"] for r in d["with_data"]]
    assert "SOL" in names
    assert "BTC" in d["configured"]


def test_empty_coin_message_names_what_does_have_data(multi_coin, client):
    client.post(f"/api/sweep?token={TOKEN}")
    d = client.get(f"/api/map?coin=XRP&token={TOKEN}").json()
    assert "No data for XRP" in d["error"]
    assert "BTC" in d["error"]              # tells you where to look instead


# --------------------------------------------------------------------------
# the live candle read
# --------------------------------------------------------------------------

def _synthetic_candles(n=200, step=900, start=4000.0, drift=-0.4):
    from liqmap.structure import Candle
    import time as _t
    out = []
    px = start
    now = _t.time()
    for i in range(n):
        o = px
        cl = px + drift * (1 if i % 7 else -3)
        out.append(Candle(ts=now - (n - i) * step, open=o,
                          high=max(o, cl) + 2, low=min(o, cl) - 2,
                          close=cl, volume=1000.0))
        px = cl
    return out


class _CandleClient(_FakeClient):
    INTERVALS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400}

    def __init__(self, candles=None, fail=False, **kw):
        super().__init__(**kw)
        self._candles = candles if candles is not None else _synthetic_candles()
        self.candle_fail = fail

    def candles(self, coin, interval="15m", bars=200):
        if self.candle_fail:
            raise RuntimeError("candleSnapshot unavailable")
        return list(self._candles)


def test_read_requires_a_token(client):
    assert client.get("/api/read").status_code == 401


def test_read_assembles_signals_from_candles_alone(client):
    web.runtime()._client = _CandleClient()
    d = client.get(f"/api/read?coin=BTC&interval=15m&token={TOKEN}").json()

    assert "error" not in d
    assert d["lean"] in ("up", "down", "flat")
    assert isinstance(d["signals"], list)
    assert 0.0 <= d["elapsed_fraction"] <= 1.0
    assert d["has_watch"] is False      # no level watch running


def test_read_never_quotes_a_probability_it_has_not_measured(client):
    web.runtime()._client = _CandleClient()
    d = client.get(f"/api/read?token={TOKEN}").json()
    assert "No calibration yet" in d["verdict"]


def test_read_reports_missing_candles_rather_than_guessing(client):
    web.runtime()._client = _CandleClient(fail=True)
    d = client.get(f"/api/read?token={TOKEN}").json()
    assert "candles unavailable" in d["error"]


def test_read_refuses_a_series_too_short_to_mean_anything(client):
    web.runtime()._client = _CandleClient(candles=_synthetic_candles(n=5))
    d = client.get(f"/api/read?token={TOKEN}").json()
    assert "not enough to read structure" in d["error"]


def test_read_rejects_an_interval_the_exchange_does_not_have(client):
    from liqmap.hl import HyperliquidError, InfoClient

    class Strict(_CandleClient):
        def candles(self, coin, interval="15m", bars=200):
            if interval not in InfoClient.INTERVALS:
                raise HyperliquidError(f"unknown interval {interval!r}")
            return list(self._candles)

    web.runtime()._client = Strict()
    d = client.get(f"/api/read?interval=7m&token={TOKEN}").json()
    assert "unknown interval" in d["error"]


def test_calibration_replay_measures_instead_of_asserting(client):
    web.runtime()._client = _CandleClient(candles=_synthetic_candles(n=300))
    r = client.post(f"/api/calibrate?coin=BTC&interval=15m&token={TOKEN}").json()
    assert r["ok"]
    assert r["scored"] > 0
    bands = {row["band"] for row in r["table"]}
    assert bands == {"strong down", "down", "flat", "up", "strong up"}


def test_calibration_refuses_a_series_too_short_to_replay(client):
    web.runtime()._client = _CandleClient(candles=_synthetic_candles(n=30))
    r = client.post(f"/api/calibrate?token={TOKEN}").json()
    assert not r["ok"]
    assert "need 60+" in r["error"]


def test_calibrations_are_kept_per_market_and_timeframe(client):
    """A lean worth something on 15m gold says nothing about 4h BTC."""
    rt = web.runtime()
    a = rt.calibration("BTC", "15m")
    b = rt.calibration("BTC", "1h")
    c2 = rt.calibration("vntl:GOLD", "15m")
    a.observe(0.6, True)
    assert a.samples == 1 and b.samples == 0 and c2.samples == 0
    assert rt.calibration("BTC", "15m") is a


def test_dashboard_carries_the_read_panel(client):
    html = client.get("/").text
    for marker in ("loadRead", "doCalibrate", "Candle read", "toggleReadAuto"):
        assert marker in html


# --------------------------------------------------------------------------
# live price and staleness
# --------------------------------------------------------------------------

class _NowClient(_CandleClient):
    def __init__(self, mids=None, mids_fail=False, **kw):
        super().__init__(**kw)
        self._mids = mids if mids is not None else {"BTC": 101_234.5}
        self.mids_fail = mids_fail

    def all_mids_everywhere(self, dexes=None):
        if self.mids_fail:
            raise RuntimeError("exchange unreachable")
        return dict(self._mids)

    def all_mids(self, dex=""):
        return self.all_mids_everywhere()

    def mid(self, coin):
        """One symbol in one request — what the hot paths call now."""
        return self.all_mids_everywhere().get(coin)

    # The real InfoClient has these; a stub without them makes /api/diag
    # report an AttributeError that no live deployment would ever hit.
    def perp_dexs(self):
        return [{"name": "", "native": True}]

    def check_book(self, coin="BTC"):
        return f"ok -- {coin}"


def test_now_requires_a_token(client):
    assert client.get("/api/now").status_code == 401


def test_now_returns_a_live_price_and_its_age(client):
    web.runtime()._client = _NowClient()
    d = client.get(f"/api/now?coin=BTC&token={TOKEN}").json()
    assert d["spot"] == 101_234.5
    assert d["spot_age_s"] == 0.0
    assert d["server_epoch"] > 0


def test_now_serves_the_last_known_price_with_its_real_age_when_the_feed_dies(client):
    """A price that silently stops updating is worse than no price. When the
    fetch fails the last known value comes back WITH its age attached."""
    rt = web.runtime()
    rt._client = _NowClient()
    client.get(f"/api/now?coin=BTC&token={TOKEN}")          # prime the cache

    rt._client = _NowClient(mids_fail=True)
    d = client.get(f"/api/now?coin=BTC&token={TOKEN}").json()
    assert d["spot"] == 101_234.5
    assert d["spot_age_s"] is not None and d["spot_age_s"] >= 0
    assert "price fetch failed" in d["spot_error"]


def test_now_says_when_a_symbol_has_no_price(client):
    web.runtime()._client = _NowClient(mids={"ETH": 3000.0})
    d = client.get(f"/api/now?coin=BTC&token={TOKEN}").json()
    assert d["spot"] is None
    assert "no mid price" in d["spot_error"]


def test_now_reports_the_age_of_each_panel_source(client):
    web.runtime()._client = _NowClient()
    d = client.get(f"/api/now?coin=BTC&token={TOKEN}").json()
    src = d["sources"]
    assert set(src) == {"sweep", "book", "watch"}
    assert src["sweep"]["age_s"] is None          # nothing swept yet
    assert src["book"]["age_s"] is None
    assert src["watch"]["running"] is False


def test_sweep_age_is_reported_once_something_has_been_swept(multi_coin, client):
    client.post(f"/api/sweep?coin=BTC&token={TOKEN}")
    rt = web.runtime()
    rt._client = _NowClient()
    d = client.get(f"/api/now?coin=BTC&token={TOKEN}").json()
    age = d["sources"]["sweep"]["age_s"]
    assert age is not None and 0 <= age < 60


def test_reading_the_book_makes_its_age_real(client):
    web.runtime()._client = _NowClient()
    assert client.get(f"/api/now?token={TOKEN}").json()["sources"]["book"]["age_s"] is None
    client.get(f"/api/liquidity?coin=BTC&size=1000&token={TOKEN}")
    age = client.get(f"/api/now?token={TOKEN}").json()["sources"]["book"]["age_s"]
    assert age is not None and age < 60


def test_liquidity_stamps_a_freshly_fetched_book_as_current(client):
    web.runtime()._client = _NowClient()
    d = client.get(f"/api/liquidity?coin=BTC&size=1000&token={TOKEN}").json()
    assert d["book_age_s"] == 0.0


def test_read_carries_the_live_mid_beside_the_candle_close(client):
    web.runtime()._client = _NowClient()
    d = client.get(f"/api/read?coin=BTC&token={TOKEN}").json()
    assert d["spot"] == 101_234.5
    assert d["spot_vs_candle_bps"] is not None


def test_read_flags_a_candle_that_disagrees_with_the_live_price(client):
    """Two endpoints, two clocks. If they diverge, one is stale and every
    signal built on the candle is suspect."""
    candles = _synthetic_candles()
    web.runtime()._client = _NowClient(candles=candles,
                                       mids={"BTC": candles[-1].close * 1.05})
    d = client.get(f"/api/read?coin=BTC&token={TOKEN}").json()
    assert d["stale"] is True
    assert abs(d["spot_vs_candle_bps"]) > 25


def test_read_is_not_flagged_stale_when_the_prices_agree(client):
    candles = _synthetic_candles()
    web.runtime()._client = _NowClient(candles=candles,
                                       mids={"BTC": candles[-1].close})
    d = client.get(f"/api/read?coin=BTC&token={TOKEN}").json()
    assert d["stale"] is False


def test_dashboard_carries_the_ticker(client):
    html = client.get("/").text
    for marker in ("loadNow", "paintTicker", "startTicker", "tickPx", "ageText"):
        assert marker in html


# --------------------------------------------------------------------------
# market picker, backtest and weights
# --------------------------------------------------------------------------

def test_markets_lists_the_whole_tradeable_universe(client):
    """THE BUG: the picker was built from swept history, so it listed BTC and
    nothing else. What you can trade and what has been swept are different
    questions."""
    web.runtime()._client = _NowClient(mids={
        "BTC": 100_000.0, "ETH": 3_000.0,
        "vntl:GOLD": 4_100.0, "vntl:CL": 74.0, "para:NVDA": 190.0})
    d = client.get(f"/api/markets?token={TOKEN}").json()

    syms = {m["symbol"] for m in d["markets"]}
    assert syms == {"BTC", "ETH", "vntl:GOLD", "vntl:CL", "para:NVDA"}
    assert d["count"] == 5
    assert set(d["venues"]) == {"", "vntl", "para"}


def test_markets_are_ordered_by_asset_class_then_ticker(client):
    """Venue order is not useful for finding a market. Class order is: with
    1,393 symbols, crypto, metals, energy and so on is how you navigate."""
    web.runtime()._client = _NowClient(mids={
        "ETH": 3_000.0, "BTC": 100_000.0, "vntl:GOLD": 4_100.0,
        "vntl:CL": 74.0})
    rows = client.get(f"/api/markets?token={TOKEN}").json()["markets"]
    assert [r["symbol"] for r in rows] == ["BTC", "ETH", "vntl:GOLD", "vntl:CL"]
    assert [r["klass"] for r in rows] == ["crypto", "crypto", "metals", "energy"]


def test_markets_mark_which_ones_have_swept_data(multi_coin, client):
    client.post(f"/api/sweep?token={TOKEN}")
    web.runtime()._client = _NowClient(mids={"BTC": 100_000.0, "XRP": 2.0})
    rows = client.get(f"/api/markets?refresh=true&token={TOKEN}").json()["markets"]
    by = {r["symbol"]: r for r in rows}
    assert by["BTC"]["has_data"] is True
    assert by["XRP"]["has_data"] is False


def test_markets_are_cached_rather_than_refetched_every_poll(client):
    calls = {"n": 0}

    class Counting(_NowClient):
        def all_mids_everywhere(self, dexes=None):
            calls["n"] += 1
            return {"BTC": 100_000.0}

    web.runtime()._client = Counting()
    for _ in range(4):
        client.get(f"/api/markets?token={TOKEN}")
    assert calls["n"] == 1
    client.get(f"/api/markets?refresh=true&token={TOKEN}")
    assert calls["n"] == 2


def test_markets_serve_the_cache_when_the_exchange_is_unreachable(client):
    rt = web.runtime()
    rt._client = _NowClient(mids={"BTC": 100_000.0})
    client.get(f"/api/markets?token={TOKEN}")
    rt._client = _NowClient(mids_fail=True)
    d = client.get(f"/api/markets?refresh=true&token={TOKEN}").json()
    assert d["stale"] is True
    assert any(m["symbol"] == "BTC" for m in d["markets"])


def test_backtest_reports_train_and_held_out_separately(client):
    web.runtime()._client = _CandleClient(candles=_synthetic_candles(n=600))
    d = client.post(f"/api/backtest?coin=BTC&interval=15m&token={TOKEN}").json()
    assert d["ok"]
    assert "held out" in d["test"]["label"]
    assert d["baseline_test"] is not None
    assert "overfit_gap_pts" in d


def test_backtest_names_what_it_cannot_see(client):
    """The replay is blind to the three strongest live signals. Saying so is
    the difference between a floor and a false promise."""
    web.runtime()._client = _CandleClient(candles=_synthetic_candles(n=600))
    d = client.post(f"/api/backtest?token={TOKEN}").json()
    assert set(d["blind_to"]) >= {"flow", "absorption"}


def test_backtest_reports_missing_candles(client):
    web.runtime()._client = _CandleClient(fail=True)
    d = client.post(f"/api/backtest?token={TOKEN}").json()
    assert not d["ok"] and "candles unavailable" in d["error"]


def test_weights_can_be_read_applied_and_reset(client):
    from liqmap import candleread

    base = dict(candleread.DEFAULT_WEIGHTS)
    try:
        d = client.post(f"/api/weights?token={TOKEN}", json={"flow": 2.5}).json()
        assert d["ok"] and d["weights"]["flow"] == 2.5

        r = client.post(f"/api/weights?reset=true&token={TOKEN}").json()
        assert r["weights"] == base
    finally:
        candleread.WEIGHTS.clear()
        candleread.WEIGHTS.update(base)


def test_weights_are_clamped_and_unknown_names_rejected(client):
    from liqmap import candleread

    base = dict(candleread.DEFAULT_WEIGHTS)
    try:
        d = client.post(f"/api/weights?token={TOKEN}",
                        json={"flow": 999.0, "nonsense": 1.0}).json()
        assert d["weights"]["flow"] == 5.0
        assert "nonsense" not in d["weights"]

        bad = client.post(f"/api/weights?token={TOKEN}",
                          json={"nonsense": 1.0}).json()
        assert not bad["ok"]
    finally:
        candleread.WEIGHTS.clear()
        candleread.WEIGHTS.update(base)


def test_dashboard_carries_the_alert_and_backtest_controls(client):
    html = client.get("/").text
    for marker in ("checkAlert", "runBacktest", "alertBar", "rAlert",
                   "applyWeights", "optgroup"):
        assert marker in html


def test_higher_timeframe_offers_the_shorter_options(client):
    html = client.get("/").text
    sel = html.split('id="rHigh"')[1].split("</select>")[0]
    for tf in ("15m", "30m", "1h", "4h"):
        assert f">{tf}<" in sel


# --------------------------------------------------------------------------
# the 401, and the stale-candle direction bug
# --------------------------------------------------------------------------

def test_a_symbol_with_a_fragment_character_strips_the_query_token(client):
    """Reproduces the reported 401. `#` in an unencoded URL turns everything
    after it into a fragment, so `&token=` is never sent."""
    web.runtime()._client = _NowClient()
    assert client.get(f"/api/now?coin=A#B&token={TOKEN}").status_code == 401


def test_the_header_keeps_auth_working_when_the_query_is_mangled(client):
    """The client now sends the token in the Authorization header as well,
    so a mangled query string cannot take authentication down with it."""
    web.runtime()._client = _NowClient()
    r = client.get("/api/now?coin=A#B",
                   headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200


def test_dashboard_builds_urls_through_the_encoder(client):
    html = client.get("/").text
    assert "function q(params)" in html
    # A blunt guard on purpose: string-concatenating the symbol anywhere is
    # how the token got stripped, so the page must not do it at all.
    assert "+ coin()" not in html, "unencoded symbol concatenated into a URL"
    assert "'Authorization': 'Bearer '" in html


def _stale_series(n=200, step=900, close_age_s=2400, start=4000.0, rise=3.0):
    """A rally whose last candle closed a while ago."""
    import time as _t
    from liqmap.structure import Candle
    now = _t.time()
    out, px = [], start
    for i in range(n):
        o, c = px, px + rise
        out.append(Candle(ts=now - (n - i) * step - close_age_s, open=o,
                          high=c + 1, low=o - 1, close=c, volume=500.0))
        px = c
    return out


class _StaleClient(_NowClient):
    def __init__(self, candles, spot):
        super().__init__(candles=candles, mids={"BTC": spot})


def test_a_closed_last_candle_is_detected_not_treated_as_forming(client):
    bars = _stale_series()
    web.runtime()._client = _StaleClient(bars, bars[-1].close)
    d = client.get(f"/api/read?coin=BTC&interval=15m&token={TOKEN}").json()
    assert d["candle_was_forming"] is False
    assert d.get("synthesised_candle") is True
    assert d["candle_age_s"] > 900


def test_a_stale_green_candle_does_not_read_up_when_price_has_fallen(client):
    """THE REPORTED BUG. The feed's last bar is green and closed. Price has
    since dropped. Reading the stale bar gives position 100% and an UP lean
    on a market that is falling."""
    bars = _stale_series()
    web.runtime()._client = _StaleClient(bars, bars[-1].close * 0.985)
    d = client.get(f"/api/read?coin=BTC&interval=15m&token={TOKEN}").json()

    assert d["last"] < d["open"], "the read must reflect the live price"
    assert d["change_bps"] < 0
    assert d["position_in_range"] < 0.2
    assert d["lean"] != "up"


def test_a_stale_red_candle_does_not_read_down_when_price_has_risen(client):
    bars = _stale_series(rise=-3.0, start=4600.0)
    web.runtime()._client = _StaleClient(bars, bars[-1].close * 1.015)
    d = client.get(f"/api/read?coin=BTC&interval=15m&token={TOKEN}").json()
    assert d["change_bps"] > 0
    assert d["lean"] != "down"


def test_the_live_mid_is_overlaid_onto_a_genuinely_forming_candle(client):
    """Even when the last bar IS the forming one, candleSnapshot lags the
    tape, so its close must not be used as the current price."""
    import time as _t
    from liqmap.structure import Candle
    now = _t.time()
    bars = _stale_series(close_age_s=0)
    bars[-1] = Candle(ts=now - 300, open=4000.0, high=4010.0, low=3995.0,
                      close=4005.0, volume=100.0)
    web.runtime()._client = _StaleClient(bars, 3990.0)      # price below the bar's low

    d = client.get(f"/api/read?coin=BTC&interval=15m&token={TOKEN}").json()
    assert d["candle_was_forming"] is True
    assert d["last"] == 3990.0
    assert d["low"] == 3990.0, "the live price must extend the candle's range"
    assert d["position_in_range"] == pytest.approx(0.0, abs=1e-9)


def test_the_read_states_what_it_predicts(client):
    web.runtime()._client = _NowClient()
    d = client.get(f"/api/read?token={TOKEN}").json()
    assert "predicts" in d and "close of this" in d["predicts"]


# --------------------------------------------------------------------------
# weights, asset classes and the forward test
# --------------------------------------------------------------------------

def test_weights_endpoint_reports_defaults_alongside_live(client):
    d = client.post(f"/api/weights?token={TOKEN}").json()
    assert "weights" in d and "defaults" in d


def test_markets_carry_an_asset_class(client):
    web.runtime()._client = _NowClient(mids={
        "BTC": 100_000.0, "vntl:GOLD": 4_100.0, "vntl:EUR": 1.08,
        "para:NVDA": 190.0, "para:SPY": 600.0, "vntl:JP225": 39_000.0})
    d = client.get(f"/api/markets?token={TOKEN}").json()
    by = {m["symbol"]: m["klass"] for m in d["markets"]}
    assert by["BTC"] == "crypto"
    assert by["vntl:GOLD"] == "metals"
    assert by["vntl:EUR"] == "fx"
    assert by["para:NVDA"] == "equity"
    assert by["para:SPY"] == "etf"
    assert by["vntl:JP225"] == "index"


def test_markets_report_class_counts_for_the_picker(client):
    web.runtime()._client = _NowClient(mids={
        "BTC": 1.0, "ETH": 2.0, "vntl:GOLD": 3.0})
    d = client.get(f"/api/markets?token={TOKEN}").json()
    counts = {c["klass"]: c["n"] for c in d["classes"]}
    assert counts["crypto"] == 2 and counts["metals"] == 1


def test_a_read_is_recorded_once_per_candle(client):
    """Polling every ten seconds must not write ninety rows for one candle
    and let a single lucky fifteen minutes dominate the hit rate."""
    web.runtime()._client = _NowClient()
    first = client.get(f"/api/read?coin=BTC&interval=15m&token={TOKEN}").json()
    second = client.get(f"/api/read?coin=BTC&interval=15m&token={TOKEN}").json()
    assert first["recorded"] is True
    assert second["recorded"] is False


def test_resolution_refuses_to_invent_an_outcome(client):
    """A read whose candle is no longer in the window stays unresolved. A
    fabricated outcome poisons the measurement permanently."""
    rt = web.runtime()
    rt._client = _NowClient()
    client.get(f"/api/read?coin=BTC&interval=15m&token={TOKEN}")
    rt.history._conn.execute(
        "UPDATE reads SET candle_end = 1, candle_ts = 1")
    rt.history._conn.commit()

    d = client.post(f"/api/resolve-reads?token={TOKEN}").json()
    assert d["resolved"] == 0
    assert d["unresolvable"] >= 1


def test_forward_stats_separate_reads_that_had_flow(client):
    """The whole point: a historical replay cannot see flow or absorption, so
    the forward test has to report those cases apart from the rest."""
    rt = web.runtime()
    rt._client = _NowClient()
    d = client.get(f"/api/forward?token={TOKEN}").json()["stats"]
    assert set(d) >= {"overall", "with_flow", "without_flow", "bands", "pending"}


def test_a_correct_read_is_scored_from_the_price_at_the_time(client):
    """Correct means price finished on the leaned side measured from the read
    price — not from the candle's open. That is the question a scalper asks."""
    rt = web.runtime()
    h = rt.history
    rid = h.record_read(coin="BTC", interval="15m", candle_ts=1000.0,
                        candle_end=1900.0, score=0.6, lean="up",
                        confidence=0.8, elapsed_frac=0.5, price=100.0,
                        had_flow=True)
    assert rid
    assert h.resolve_read(rid, 101.0) is True

    stats = h.read_stats("BTC", "15m")
    assert stats["overall"]["n"] == 1
    assert stats["overall"]["accuracy"] == 1.0
    assert stats["with_flow"]["n"] == 1


def test_a_wrong_read_scores_zero(client):
    h = web.runtime().history
    rid = h.record_read(coin="ETH", interval="5m", candle_ts=1.0,
                        candle_end=2.0, score=-0.6, lean="down",
                        confidence=0.8, elapsed_frac=0.9, price=100.0,
                        had_flow=False)
    h.resolve_read(rid, 105.0)
    stats = h.read_stats("ETH", "5m")
    assert stats["overall"]["accuracy"] == 0.0
    assert stats["without_flow"]["n"] == 1


def test_a_flat_read_makes_no_claim_and_is_not_scored(client):
    h = web.runtime().history
    rid = h.record_read(coin="SOL", interval="5m", candle_ts=1.0,
                        candle_end=2.0, score=0.02, lean="flat",
                        confidence=0.1, elapsed_frac=0.9, price=100.0,
                        had_flow=False)
    h.resolve_read(rid, 110.0)
    assert h.read_stats("SOL", "5m")["overall"]["n"] == 0


def test_backtest_states_the_real_history_it_used(client):
    """'How can a backtest run with no history' deserves a concrete answer:
    how many real candles, over what dates."""
    web.runtime()._client = _CandleClient(candles=_synthetic_candles(n=600))
    d = client.post(f"/api/backtest?token={TOKEN}").json()
    assert d["candles"] >= 500
    assert "to" in d["history_span"] and "UTC" in d["history_span"]
    assert "candleSnapshot" in d["source"]


def test_dashboard_carries_the_weight_editor_and_forward_panel(client):
    html = client.get("/").text
    for marker in ("paintWeights", "readWeightInputs", "loadForward",
                   "renderCoins", "mktFind"):
        assert marker in html


# --------------------------------------------------------------------------
# the websocket feed driving the read
# --------------------------------------------------------------------------

def _grid_bars(n=200, step=900, coin="BTC"):
    import time as _t
    from liqmap.live import grid_start
    from liqmap.structure import Candle
    start = grid_start(_t.time(), step)
    out, px = [], 4000.0
    for i in range(n, 0, -1):
        o = px
        c = px + (2.0 if i % 3 else -3.0)
        out.append(Candle(ts=start - i * step, open=o, high=max(o, c) + 1,
                          low=min(o, c) - 1, close=c, volume=400.0))
        px = c
    out.append(Candle(ts=start, open=px, high=px + 2, low=px - 2, close=px,
                      volume=50.0))
    return out


def _attach_feed(rt, coin="BTC", interval="15m", bars=None):
    """A LiveFeed with no socket, marked running, so the read path can be
    exercised without a network."""
    import time as _t
    from liqmap.live import LiveFeed

    bars = bars or _grid_bars()
    f = LiveFeed(coin, intervals=(interval,))
    f.seed(interval, bars)
    f._thread = type("T", (), {"is_alive": lambda self: True})()
    f.last_msg_ts = _t.time()
    rt.feed = f
    return f


def test_read_falls_back_to_polling_with_no_feed(client):
    web.runtime()._client = _NowClient(candles=_grid_bars())
    d = client.get(f"/api/read?coin=BTC&interval=15m&token={TOKEN}").json()
    assert d["source"].startswith("polled")
    assert "feed" not in d


def test_a_live_feed_builds_the_current_candle_from_fills(client):
    import time as _t
    rt = web.runtime()
    bars = _grid_bars()
    rt._client = _NowClient(candles=bars, mids={"BTC": 9999.0})
    f = _attach_feed(rt, bars=bars)

    base = int(_t.time() * 1000)
    start_px = bars[-1].close
    f._handle_trades([{"px": str(start_px - i * 4), "sz": "3", "side": "A",
                       "time": base + i * 400} for i in range(12)])

    d = client.get(f"/api/read?coin=BTC&interval=15m&token={TOKEN}").json()
    assert d["source"] == "websocket"
    assert d["feed"]["trades_seen"] == 12
    # Built from the tape, NOT from the polled mid of 9999.
    assert d["last"] == pytest.approx(start_px - 11 * 4)
    assert d["change_bps"] < 0
    assert d["lean"] != "up"


def test_the_feeds_tape_supplies_flow_without_a_level_watch(client):
    import time as _t
    rt = web.runtime()
    bars = _grid_bars()
    rt._client = _NowClient(candles=bars)
    f = _attach_feed(rt, bars=bars)

    base = int(_t.time() * 1000)
    f._handle_trades([{"px": str(bars[-1].close), "sz": "5", "side": "A",
                       "time": base + i * 300} for i in range(10)])

    d = client.get(f"/api/read?coin=BTC&interval=15m&token={TOKEN}").json()
    flow = next((s for s in d["signals"] if s["name"] == "flow"), None)
    assert flow is not None and flow["direction"] == "down"


def test_a_pushed_book_is_used_instead_of_polling_one(client):
    rt = web.runtime()
    bars = _grid_bars()
    rt._client = _NowClient(candles=bars)
    f = _attach_feed(rt, bars=bars)
    f._handle_book({"levels": [[{"px": "3950", "sz": "2"}],
                               [{"px": "3951", "sz": "9"}]], "time": 1})

    d = client.get(f"/api/read?coin=BTC&interval=15m&token={TOKEN}").json()
    assert d["book"]["bid"] == 3950.0 and d["book"]["ask"] == 3951.0
    assert d["book"]["pushed"] is True


def test_a_stale_feed_is_not_trusted_over_polling(client):
    """A socket that stopped delivering must not keep serving its last candle
    as though it were current."""
    rt = web.runtime()
    bars = _grid_bars()
    rt._client = _NowClient(candles=bars)
    f = _attach_feed(rt, bars=bars)
    f.last_msg_ts = 1.0                      # ancient

    d = client.get(f"/api/read?coin=BTC&interval=15m&token={TOKEN}").json()
    assert d["source"].startswith("polled")


def test_a_feed_on_another_market_is_ignored(client):
    rt = web.runtime()
    rt._client = _NowClient(candles=_grid_bars())
    _attach_feed(rt, coin="ETH")
    d = client.get(f"/api/read?coin=BTC&interval=15m&token={TOKEN}").json()
    assert d["source"].startswith("polled")


def test_feed_status_endpoint_reports_off_when_nothing_runs(client):
    assert client.get(f"/api/feed?token={TOKEN}").json()["running"] is False


def test_stopping_a_feed_that_is_not_running_is_safe(client):
    r = client.post(f"/api/feed?stop=true&token={TOKEN}").json()
    assert r["ok"] is True and r["running"] is False


def test_the_stream_requires_a_token(client):
    with client.stream("GET", "/api/stream?coin=BTC&max_seconds=1") as r:
        assert r.status_code == 401


def test_the_stream_announces_whether_a_socket_is_behind_it(client):
    """A silent stream must never be mistaken for a quiet market."""
    web.runtime()._client = _NowClient(candles=_grid_bars())
    url = f"/api/stream?coin=BTC&max_seconds=1&token={TOKEN}"
    with client.stream("GET", url) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        for line in r.iter_lines():
            if line.startswith("data:"):
                import json as _j
                hello = _j.loads(line[5:])
                assert hello["feed"] is False     # no socket attached
                assert hello["coin"] == "BTC"
                break


def test_dashboard_carries_the_feed_controls(client):
    html = client.get("/").text
    for marker in ("startFeed", "openStream", "EventSource", "paintRead",
                   "paintFeed"):
        assert marker in html


# --------------------------------------------------------------------------
# diagnosable errors and the fast price channel
# --------------------------------------------------------------------------

def test_a_crash_returns_the_reason_not_five_words(client):
    """'Internal Server Error' costs a round trip every time: the only way to
    learn what happened is to ask someone with the logs."""
    rt = web.runtime()

    class Exploding:
        INTERVALS = {"15m": 900}

        def all_mids_everywhere(self, dexes=None):
            raise ZeroDivisionError("something specific went wrong")

        def all_mids(self, dex=""):
            return self.all_mids_everywhere()

    rt._client = Exploding()
    rt._markets_cache = None

    # markets() catches, so force a path that does not: the diag route's own
    # handler is protected, so assert the handler itself via a bad route arg.
    r = client.get(f"/api/diag?coin=BTC&token={TOKEN}")
    assert r.status_code == 200
    body = r.json()
    assert body["all_passed"] is False
    assert "ZeroDivisionError" in str(body["first_failure"]["error"])


def test_diag_names_the_first_upstream_call_that_fails(client):
    class Broken:
        INTERVALS = {"15m": 900, "4h": 14400}

        def all_mids(self, dex=""):
            return {"BTC": 100_000.0}

        def all_mids_everywhere(self, dexes=None):
            return {"BTC": 100_000.0}

        def perp_dexs(self):
            return [{"name": "", "native": True}]

        def candles(self, coin, interval="15m", bars=200):
            return []

        def check_book(self, coin):
            raise RuntimeError("422 unknown coin")

        def book(self, coin):
            from liqmap.flow import Book
            return Book(coin=coin, ts=0)

    web.runtime()._client = Broken()
    d = client.get(f"/api/diag?coin=GOLD&interval=15m&token={TOKEN}").json()

    assert d["all_passed"] is False
    assert d["first_failure"]["step"] == "mid for GOLD"
    assert "dex prefix" in d["first_failure"]["error"]
    # Every step is reported, not just the failure, so the working half is
    # visible too.
    assert [s["step"] for s in d["steps"]][:3] == [
        "resolve symbol", "allMids (canonical)", "perpDexs"]


def test_diag_passes_cleanly_on_a_healthy_symbol(client):
    web.runtime()._client = _NowClient(candles=_grid_bars())
    d = client.get(f"/api/diag?coin=BTC&interval=15m&token={TOKEN}").json()
    failures = [s for s in d["steps"] if not s["ok"]]
    assert not failures, failures


def test_status_carries_the_last_traceback(client):
    rt = web.runtime()
    rt.last_error = "ValueError: boom"
    rt.last_traceback = "line one\nline two"
    d = client.get(f"/api/status?token={TOKEN}").json()
    assert d["last_error"] == "ValueError: boom"
    assert "line two" in d["last_traceback"]


def test_the_stream_pushes_price_on_every_fill(client):
    """The full read is coalesced because it costs structure and zones. The
    price is one float and goes stale fastest, so it must not be."""
    import json as _j
    import threading
    import time as _t

    rt = web.runtime()
    bars = _grid_bars()
    rt._client = _NowClient(candles=bars)
    f = _attach_feed(rt, bars=bars)

    def fire():
        base = int(_t.time() * 1000)
        for i in range(4):
            _t.sleep(0.2)
            f._handle_trades([{"px": str(4000 + (i + 1) * 3), "sz": "1",
                               "side": "B", "time": base + i * 200}])
            for fn in list(f._listeners):
                fn("trades")

    threading.Thread(target=fire, daemon=True).start()

    prices, reads = [], 0
    url = f"/api/stream?coin=BTC&interval=15m&max_seconds=2&token={TOKEN}"
    with client.stream("GET", url) as r:
        ev = None
        for line in r.iter_lines():
            if line.startswith("event:"):
                ev = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                if ev == "px":
                    prices.append(_j.loads(line[5:])["px"])
                elif ev is None:
                    reads += 1
                ev = None
            elif line == "":
                ev = None

    assert len(prices) >= 3, f"expected a price per fill, got {prices}"
    assert prices == sorted(prices)            # each one newer than the last
    assert reads < len(prices), "reads should be coalesced below price ticks"


def test_a_closed_stream_unregisters_its_listener(client):
    """EventSource reconnects on its own, so a tab left open overnight would
    otherwise leave hundreds of dead callbacks firing on every fill."""
    rt = web.runtime()
    bars = _grid_bars()
    rt._client = _NowClient(candles=bars)
    f = _attach_feed(rt, bars=bars)

    for _ in range(3):
        url = f"/api/stream?coin=BTC&max_seconds=1&token={TOKEN}"
        with client.stream("GET", url) as r:
            for _line in r.iter_lines():
                pass
    assert f.listener_count == 0


# --------------------------------------------------------------------------
# multi-timeframe confrontation, and the fetch storm
# --------------------------------------------------------------------------

def _tf_series(step, n, start, drift, tail, now=None):
    import time as _t
    from liqmap.live import grid_start
    from liqmap.structure import Candle
    now = now or _t.time()
    st = grid_start(now, step)
    out, px = [], start
    for i in range(n, 0, -1):
        o = px
        c = px + drift
        out.append(Candle(ts=st - i * step, open=o,
                          high=max(o, c) + abs(drift) * 0.2,
                          low=min(o, c) - abs(drift) * 0.2,
                          close=c, volume=300.0))
        px = c
    o, c = px, px + tail
    out.append(Candle(ts=st, open=o, high=max(o, c) + 0.2,
                      low=min(o, c) - 0.2, close=c, volume=20.0))
    return out


class _MultiTF:
    """4h and 1h falling, 15m and 5m bouncing — the fade shape."""

    INTERVALS = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400}

    def __init__(self):
        self.series = {
            "4h": _tf_series(14400, 60, 5000.0, -25.0, -8.0),
            "1h": _tf_series(3600, 60, 4200.0, -6.0, -2.0),
            "15m": _tf_series(900, 60, 4020.0, 2.0, 3.0),
            "5m": _tf_series(300, 60, 4040.0, 1.0, 2.0),
        }
        self.requests = 0

    def mid(self, coin):
        self.requests += 1
        return self.series["15m"][-1].close

    def all_mids(self, dex="", max_age=None):
        self.requests += 1
        return {"BTC": self.series["15m"][-1].close}

    def all_mids_everywhere(self, dexes=None):
        return self.all_mids()

    def perp_dexs(self):
        return [{"name": "", "native": True}]

    def candles(self, coin, interval="15m", bars=200):
        return list(self.series[interval])

    def book(self, coin):
        from liqmap.flow import Book, Level
        p = self.series["15m"][-1].close
        return Book(coin=coin, ts=0,
                    bids=[Level(p - 0.1 * i, 4) for i in range(1, 12)],
                    asks=[Level(p + 0.1 * i, 9) for i in range(1, 12)])


def test_the_read_reports_every_timeframe_separately(client):
    web.runtime()._client = _MultiTF()
    d = client.get(f"/api/read?coin=BTC&interval=15m&higher=4h&token={TOKEN}").json()

    assert "pressure_error" not in d, d.get("pressure_error")
    tfs = {t["timeframe"]: t for t in d["timeframes"]}
    assert set(tfs) >= {"4h", "1h", "15m", "5m"}
    # slowest first: the context before the trigger
    assert d["timeframes"][0]["timeframe"] == "4h"


def test_a_falling_higher_timeframe_and_rising_lower_one_is_a_conflict(client):
    """The shape a fade is built on. Averaging these to zero would throw away
    the only thing worth knowing."""
    web.runtime()._client = _MultiTF()
    d = client.get(f"/api/read?coin=BTC&interval=15m&higher=4h&token={TOKEN}").json()

    tfs = {t["timeframe"]: t for t in d["timeframes"]}
    assert tfs["4h"]["winner"] == "sellers"
    assert tfs["15m"]["winner"] == "buyers"

    c = d["confrontation"]
    assert c["conflicted"] is True and c["aligned"] is False
    assert c["consensus"] == "sellers"      # the slow timeframe carries more
    assert "CONFLICT" in c["verdict"]
    assert "not a reason to follow" in c["verdict"]


def test_a_higher_timeframe_is_read_from_recent_bars_not_the_partial_one(client):
    """Two minutes into a 4-hour candle there is nothing in the current bar.
    Reading only that would silence the higher timeframe exactly when it
    matters."""
    web.runtime()._client = _MultiTF()
    d = client.get(f"/api/read?coin=BTC&interval=15m&higher=4h&token={TOKEN}").json()
    four = next(t for t in d["timeframes"] if t["timeframe"] == "4h")
    assert four["winner"] != "contested"
    assert abs(four["move_bps"]) > 50


def test_inferred_readings_are_flagged_as_such(client):
    web.runtime()._client = _MultiTF()
    d = client.get(f"/api/read?token={TOKEN}&coin=BTC&interval=15m").json()
    assert all(t["measured"] is False for t in d["timeframes"])


def test_timeframes_can_be_chosen_explicitly(client):
    web.runtime()._client = _MultiTF()
    d = client.get(f"/api/read?coin=BTC&timeframes=1h,5m&token={TOKEN}").json()
    assert [t["timeframe"] for t in d["timeframes"]] == ["1h", "5m"]


def test_an_unknown_timeframe_is_skipped_not_fatal(client):
    web.runtime()._client = _MultiTF()
    d = client.get(f"/api/read?coin=BTC&timeframes=1h,7m,5m&token={TOKEN}").json()
    assert [t["timeframe"] for t in d["timeframes"]] == ["1h", "5m"]


def test_price_polling_costs_one_request_not_one_per_dex(client):
    """THE FETCH STORM. all_mids_everywhere costs a request per perp DEX plus
    two, and it ran on every price poll and every read. On a venue with a
    dozen builder DEXes that queued behind the rate limiter until the whole
    dashboard felt broken."""
    from liqmap.hl import InfoClient

    seen = {"n": 0}

    class Counting(InfoClient):
        def post(self, body):
            seen["n"] += 1
            t = body.get("type")
            if t == "perpDexs":
                return [None] + [{"name": f"dex{i}"} for i in range(10)]
            if t == "allMids":
                dex = body.get("dex", "")
                return {"BTC": "100000"} if not dex else {f"{dex}:GOLD": "4100"}
            return {}

    web.runtime()._client = Counting()
    for _ in range(5):
        client.get(f"/api/now?coin=BTC&token={TOKEN}")
    assert seen["n"] == 1, f"5 polls made {seen['n']} upstream requests"


def test_mids_are_cached_briefly_so_bursts_collapse(client):
    from liqmap.hl import InfoClient

    seen = {"n": 0}

    class Counting(InfoClient):
        def post(self, body):
            seen["n"] += 1
            return {"BTC": "100000"}

    c = Counting()
    for _ in range(10):
        c.all_mids()
    assert seen["n"] == 1


def test_a_namespaced_symbol_asks_only_its_own_dex(client):
    from liqmap.hl import InfoClient

    asked = []

    class Counting(InfoClient):
        def post(self, body):
            asked.append(body.get("dex", ""))
            return {"vntl:GOLD": "4100"}

    c = Counting()
    assert c.mid("vntl:GOLD") == 4100.0
    assert asked == ["vntl"], "should not sweep every dex for one price"


def test_perp_dexes_are_cached_rather_than_re_asked(client):
    from liqmap.hl import InfoClient

    seen = {"n": 0}

    class Counting(InfoClient):
        def post(self, body):
            seen["n"] += 1
            return [None, {"name": "vntl"}]

    c = Counting()
    for _ in range(5):
        c.perp_dexs()
    assert seen["n"] == 1


def test_dashboard_carries_the_timeframe_ladder(client):
    html = client.get("/").text
    for marker in ("paintLadder", "tfLadder", "CONFLICT"):
        assert marker in html


# --------------------------------------------------------------------------
# the book call
# --------------------------------------------------------------------------

def _real_book():
    """The book exactly as it appears on the venue: heavy bid at the touch,
    thin offer. The offer gets consumed first, so this is UP."""
    from liqmap.flow import Book, Level
    asks = [(7737.3, 29216), (7737.2, 21803), (7737.1, 24418),
            (7737.0, 19992), (7736.8, 9996), (7736.7, 99633),
            (7736.6, 84368), (7736.5, 5052), (7736.3, 23209)]
    bids = [(7736.1, 200009), (7736.0, 774), (7735.9, 67596),
            (7735.8, 25899), (7735.7, 75222), (7735.6, 7921),
            (7735.5, 15100), (7735.4, 403850), (7735.3, 91942)]
    return Book(coin="X", ts=1.0,
                bids=[Level(p, s) for p, s in bids],
                asks=[Level(p, s) for p, s in asks])


def test_bookcall_needs_a_live_feed(client):
    d = client.get(f"/api/bookcall?coin=BTC&token={TOKEN}").json()
    assert "no live feed" in d["error"]


def test_bookcall_reads_a_real_book_as_up(client):
    """200,009 bid against 23,209 offered at the touch. The thin side is
    above, so the next print goes up — and the microprice says so."""
    rt = web.runtime()
    rt._client = _NowClient(candles=_grid_bars())
    f = _attach_feed(rt, bars=_grid_bars())

    b = _real_book()
    for i in range(12):
        f.reader.add(b, now=float(i))
    f.book = b
    f.book_updates = 12

    d = client.get(f"/api/bookcall?coin=BTC&token={TOKEN}").json()
    assert d["direction"] == "up"
    assert d["tilt"] > 0.5, "microprice must lean toward the THIN side"
    assert d["microprice"] > d["mid"]
    assert d["spread_bps"] == pytest.approx(0.259, abs=0.01)


def test_bookcall_inverts_when_the_book_inverts(client):
    """The same shape mirrored must call down. If it does not, every call
    this module makes is inverted."""
    from liqmap.flow import Book, Level

    rt = web.runtime()
    rt._client = _NowClient(candles=_grid_bars())
    f = _attach_feed(rt, bars=_grid_bars())

    b = Book(coin="X", ts=1.0,
             bids=[Level(100.0 - 0.01 * i, 1.0 if i == 0 else 3.0)
                   for i in range(6)],
             asks=[Level(100.02 + 0.01 * i, 200.0 if i == 0 else 3.0)
                   for i in range(6)])
    for i in range(12):
        f.reader.add(b, now=float(i))
    f.book = b

    d = client.get(f"/api/bookcall?coin=BTC&token={TOKEN}").json()
    assert d["direction"] == "down"
    assert d["tilt"] < -0.5


def test_bookcall_reports_every_component(client):
    rt = web.runtime()
    rt._client = _NowClient(candles=_grid_bars())
    f = _attach_feed(rt, bars=_grid_bars())
    for i in range(12):
        f.reader.add(_real_book(), now=float(i))
    f.book = _real_book()

    d = client.get(f"/api/bookcall?coin=BTC&token={TOKEN}").json()
    names = {c["name"] for c in d["components"]}
    # Aggression is NOT here. It comes from the trade feed, and scoring it
    # inside the book put the same input on both sides of the confirmation
    # check — so it moved to its own column in `delta.py`, and absorption
    # went with it.
    assert names == {"microprice tilt", "replenishment", "near imbalance",
                     "queue depletion", "mid drift"}


def test_bookcall_window_is_adjustable(client):
    rt = web.runtime()
    rt._client = _NowClient(candles=_grid_bars())
    f = _attach_feed(rt, bars=_grid_bars())
    for i in range(30):
        f.reader.add(_real_book(), now=float(i))
    f.book = _real_book()

    narrow = client.get(f"/api/bookcall?window_s=5&token={TOKEN}").json()
    wide = client.get(f"/api/bookcall?window_s=60&token={TOKEN}").json()
    assert narrow["samples"] < wide["samples"]


def test_the_feed_updates_the_reader_on_every_book_push(client):
    from liqmap.live import LiveFeed
    f = LiveFeed("BTC")
    assert f.reader.updates == 0
    for i in range(5):
        f._handle_book({"levels": [[{"px": "99.9", "sz": "5"}],
                                   [{"px": "100.1", "sz": "1"}]],
                        "time": 1000 + i})
    assert f.reader.updates == 5
    assert f.status()["book_reads"] == 5


def test_dashboard_leads_with_the_book_call(client):
    html = client.get("/").text
    assert "Book call" in html
    assert "paintBookCall" in html
    # it must come BEFORE the older candle read panel
    assert html.index("bookPanel") < html.index("readPanel")
