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
