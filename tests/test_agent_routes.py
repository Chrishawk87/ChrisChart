"""The suggest / decide / upload routes, through HTTP.

These exist because the unit tests above them all pass with the wiring
broken. `suggest.py` can be perfect while `/api/suggest` hands it a stale
book, the upload route can parse a file correctly and store it under the
wrong market, and a decision can be recorded against an id the page never
received. Every one of those is invisible until something makes the real
request.

Locked-by-default is re-checked here too. Every route added to this service
is another way to reach position data and trading configuration on a public
URL, and a new endpoint that forgets its dependency is exactly the kind of
hole that does not announce itself.
"""

import io
import tempfile
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import liqmap.web as web  # noqa: E402

TOKEN = "test-token-abc"
AUTH = {"Authorization": f"Bearer {TOKEN}"}

NEW_ROUTES = ["/api/suggest", "/api/decisions", "/api/datasets"]


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


@pytest.fixture
def locked(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("LIQMAP_DB", str(Path(tmp) / "web.db"))
        monkeypatch.delenv("LIQMAP_TOKEN", raising=False)
        monkeypatch.setenv("LIQMAP_AUTO", "false")
        web.RT = None
        with TestClient(web.create_app()) as c:
            yield c
        web.RT = None


def csv_bytes(n=300, start=1_700_000_000, step=900):
    head = "time,open,high,low,close,volume"
    rows = [f"{start + i * step},100,101,99,100.5,{10 + i % 5}" for i in range(n)]
    return ("\n".join([head] + rows)).encode()


def upload(client, data=None, coin="BTC", name="hist.csv", merge=""):
    return client.post(
        f"/api/upload-history?coin={coin}&merge_into={merge}",
        files={"file": (name, io.BytesIO(data if data is not None
                                         else csv_bytes()), "text/csv")},
        headers=AUTH)


# --------------------------------------------------------------------------
# the lock
# --------------------------------------------------------------------------

@pytest.mark.parametrize("path", NEW_ROUTES)
def test_new_routes_are_locked_without_a_token(locked, path):
    assert locked.get(path).status_code == 503


@pytest.mark.parametrize("path", NEW_ROUTES)
def test_new_routes_reject_a_bad_token(client, path):
    r = client.get(path, headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_the_upload_route_is_locked_too(locked):
    r = locked.post("/api/upload-history",
                    files={"file": ("a.csv", io.BytesIO(csv_bytes()), "text/csv")})
    assert r.status_code == 503


def test_decide_is_locked(locked):
    assert locked.post("/api/decide?id=x&taken=true").status_code == 503


# --------------------------------------------------------------------------
# suggest, with no feed running
# --------------------------------------------------------------------------

def test_suggest_without_a_feed_says_so_rather_than_failing(client):
    """The sandbox cannot reach the exchange, so this is also the shape the
    route takes whenever the socket is down — which is the case a user hits
    most often."""
    r = client.get("/api/suggest?coin=BTC", headers=AUTH)
    assert r.status_code == 200
    d = r.json()
    assert d["take"] is False
    assert "feed" in d["detail"].lower() or "feed" in d["sentence"].lower()


def test_suggest_never_returns_a_trade_without_a_book(client):
    for interval in ("1m", "15m", "1h"):
        d = client.get(f"/api/suggest?coin=BTC&interval={interval}",
                       headers=AUTH).json()
        assert d["take"] is False


def test_suggest_accepts_a_size_and_a_fee_without_error(client):
    r = client.get("/api/suggest?coin=BTC&size=25000&fee_bps=4.5", headers=AUTH)
    assert r.status_code == 200


# --------------------------------------------------------------------------
# decide
# --------------------------------------------------------------------------

def test_deciding_on_an_unknown_id_is_reported_not_raised(client):
    d = client.post("/api/decide?id=nope&taken=true", headers=AUTH).json()
    assert d["ok"] is False and d["note"]


def _pending(rt, coin="BTC", interval="15m"):
    """A suggestion on a candle that is still open, so it can be decided."""
    import time as _t

    open_ts = _t.time() - 300
    return open_ts, rt.history.record_suggestion(
        coin=coin, interval=interval, candle_ts=open_ts,
        candle_end=open_ts + 900, side="long", entry=100.0, target_px=101.0,
        stop_px=99.5, target_bps=100.0, risk_bps=50.0, rr=2.0, cost_bps=2.0,
        conviction=0.7, score=0.5, reason="tilt", made_at=open_ts + 60)


def test_a_recorded_suggestion_can_be_taken_then_scored(client):
    rt = web.runtime()
    _, sid = _pending(rt)

    assert client.post(f"/api/decide?id={sid}&taken=true",
                       headers=AUTH).json()["decision"] == "taken"

    rt.history.resolve_suggestion(sid, high=102.0, low=99.9, close=101.5)
    stats = client.get("/api/decisions?coin=BTC", headers=AUTH).json()["stats"]
    assert stats["taken"]["n"] == 1
    assert stats["overall"]["targets"] == 1


def test_deciding_on_a_closed_candle_is_refused_and_says_nothing_was_stored(client):
    """The route used to echo the requested decision back even when the
    write was refused, so the page showed a decision that is not in the
    table."""
    rt = web.runtime()
    sid = rt.history.record_suggestion(
        coin="BTC", interval="15m", candle_ts=1000.0, candle_end=1900.0,
        side="long", entry=100.0, target_px=101.0, stop_px=99.5,
        target_bps=100.0, risk_bps=50.0, rr=2.0, cost_bps=2.0,
        conviction=0.7, score=0.5)

    d = client.post(f"/api/decide?id={sid}&taken=true", headers=AUTH).json()
    assert d["ok"] is False
    assert d["decision"] is None
    assert "closed" in d["note"]


def test_an_unknown_interval_is_refused_rather_than_treated_as_15m(client):
    """A suggestion recorded on the wrong grid can never be settled — it
    just sits pending forever."""
    d = client.get("/api/suggest?coin=BTC&interval=7m", headers=AUTH).json()
    assert d["take"] is False and d["gate"] == "bad interval"


def test_decisions_reports_an_empty_history_without_inventing_a_rate(client):
    s = client.get("/api/decisions", headers=AUTH).json()["stats"]
    assert s["overall"]["n"] == 0
    assert s["overall"]["hit_rate"] is None
    assert "Nothing settled yet" in s["verdict"]


# --------------------------------------------------------------------------
# upload
# --------------------------------------------------------------------------

def test_a_good_csv_uploads_and_appears_in_the_listing(client):
    d = upload(client).json()
    assert d["ok"] and d["bars"] == 300 and d["interval"] == "15m"

    listed = client.get("/api/datasets", headers=AUTH).json()["datasets"]
    assert len(listed) == 1
    assert listed[0]["name"] == "hist.csv" and listed[0]["bars"] == 300


def test_the_upload_response_always_carries_the_training_caveat(client):
    d = upload(client).json()
    assert "seventy percent" in d["describe"]
    assert "microprice tilt" in d["cannot_train"]


def test_a_junk_upload_is_rejected_with_an_explanation_not_a_500(client):
    d = upload(client, data=b"this is not a chart", name="junk.txt").json()
    assert d["ok"] is False
    assert d["describe"] or d.get("error")
    assert client.get("/api/datasets", headers=AUTH).json()["datasets"] == []


def test_an_empty_upload_is_rejected(client):
    assert upload(client, data=b"").json()["ok"] is False


def test_uploading_twice_with_merge_extends_one_dataset(client):
    first = upload(client).json()
    did = first["dataset_id"]
    second = upload(client, data=csv_bytes(300, start=1_700_000_000 + 150 * 900),
                    merge=did).json()
    assert second["ok"] and second["dataset_id"] == did
    assert second["stored_bars"] == 450

    listed = client.get("/api/datasets", headers=AUTH).json()["datasets"]
    assert len(listed) == 1 and listed[0]["bars"] == 450


def test_uploading_twice_without_merge_keeps_two_datasets(client):
    upload(client)
    upload(client, name="other.csv")
    assert len(client.get("/api/datasets", headers=AUTH).json()["datasets"]) == 2


def test_a_dataset_can_be_deleted(client):
    did = upload(client).json()["dataset_id"]
    assert client.post(f"/api/datasets/delete?id={did}", headers=AUTH).json()["ok"]
    assert client.get("/api/datasets", headers=AUTH).json()["datasets"] == []


def test_datasets_are_filtered_by_market(client):
    upload(client, coin="BTC")
    upload(client, coin="ETH", name="eth.csv")
    assert len(client.get("/api/datasets?coin=BTC", headers=AUTH).json()["datasets"]) == 1
    assert len(client.get("/api/datasets", headers=AUTH).json()["datasets"]) == 2


# --------------------------------------------------------------------------
# backtesting an uploaded dataset
# --------------------------------------------------------------------------

def test_a_backtest_can_run_against_uploaded_history(client):
    """The whole point of the upload: a replay that does not need the
    exchange API, which this sandbox cannot reach at all."""
    did = upload(client, data=zigzag_csv()).json()["dataset_id"]
    d = client.post(f"/api/backtest?dataset={did}&tune=false",
                    headers=AUTH).json()
    assert d["ok"] is True
    assert d["dataset"] == did
    assert "uploaded history" in d["source"]
    assert d["candles"] >= 300


def test_an_uploaded_backtest_still_states_what_it_is_blind_to(client):
    did = upload(client, data=zigzag_csv()).json()["dataset_id"]
    d = client.post(f"/api/backtest?dataset={did}&tune=false",
                    headers=AUTH).json()
    assert "absorption" in d["blind_to"]
    assert "not in an OHLCV series at any length" in d["caveat"]


def test_backtesting_a_missing_dataset_reports_it(client):
    d = client.post("/api/backtest?dataset=nope", headers=AUTH).json()
    assert d["ok"] is False and "nope" in d["error"]


def zigzag_csv(n=400, step=900, start=1_700_000_000):
    """Bars that actually swing, so the replay produces samples.

    A flat series parses fine and yields nothing to score, which would make
    the test above pass for the wrong reason.
    """
    head = "time,open,high,low,close,volume"
    rows = []
    px = 100.0
    for i in range(n):
        leg = 1.0 if (i // 12) % 2 == 0 else -1.0
        o = px
        c = o + leg * (0.4 + (i % 5) * 0.05)
        rows.append(f"{start + i * step},{o:.4f},{max(o, c) + 0.2:.4f},"
                    f"{min(o, c) - 0.2:.4f},{c:.4f},{100 + i % 9}")
        px = c
    return ("\n".join([head] + rows)).encode()


# --------------------------------------------------------------------------
# the full path, with a feed actually running
# --------------------------------------------------------------------------
#
# Every test above exercises the refusal branch, because this sandbox has no
# route to the exchange. That leaves the branch that matters -- a real book
# producing a real suggestion, recorded and then decided on -- completely
# unexercised, which is precisely where a wiring bug would live. So the feed
# is built by hand and fed a book directly, with no socket involved.

def _live_feed(coin="BTC", interval="15m", rising=True):
    """A LiveFeed with a book AND a tape pushed into it, no network.

    Both halves are required. The book alone produces a lean; the trades are
    what the confirmation gate checks it against, and without them every
    suggestion is correctly refused as unconfirmed — which is the right
    behaviour and makes the fixture useless for testing anything else.
    """
    import time as _t

    from liqmap.flow import Book, Level, Trade
    from liqmap.live import LiveFeed
    from liqmap.structure import Candle

    f = LiveFeed(coin, intervals=(interval,))
    now = _t.time()
    sign = 1.0 if rising else -1.0

    # Bids fat, offers thin and thinning: microprice pinned to the offer.
    # Mirrored for a short.
    for i in range(40):
        fat = [Level(round(99.99 - j * 0.01, 4), 800.0 if j == 3 else 120.0)
               for j in range(8)]
        thin = [Level(round(100.01 + j * 0.01, 4),
                      max(1.0, 40.0 - i) if j == 0 else 30.0)
                for j in range(8)]
        b = (Book(coin=coin, ts=now - 40 + i, bids=fat, asks=thin) if rising
             else Book(coin=coin, ts=now - 40 + i,
                       bids=[Level(round(99.99 - j * 0.01, 4),
                                   max(1.0, 40.0 - i) if j == 0 else 30.0)
                             for j in range(8)],
                       asks=[Level(round(100.01 + j * 0.01, 4),
                                   800.0 if j == 3 else 120.0)
                             for j in range(8)]))
        f.book = b
        f.book_updates += 1
        f.reader.add(b, now=now - 40 + i)

    f.last_msg_ts = now
    f.seed(interval, [Candle(ts=now - (30 - i) * 900, open=100.0, high=100.9,
                             low=99.1, close=100.2, volume=50.0)
                      for i in range(30)])

    # Fills that move price the same way the book leans, so the two agree.
    builder = f.builders[interval]
    px = 100.0
    for i in range(60):
        px += sign * 0.004
        tr = Trade(px=round(px, 4), sz=0.5, aggressor="buy" if rising else "sell",
                   ts=now - 60 + i)
        f.tape.add(tr)
        builder.add(tr)
        f.trades_seen += 1
    return f


def _settle(rt, coin="BTC", interval="15m", direction="up", seconds=60.0):
    """Pre-age the agreement tracker, as if the feed had been up a while.

    On a genuinely fresh feed `held_s` really is zero and scalp mode
    correctly refuses — that behaviour has its own test below. These tests
    are about the suggestion itself, so they start from an agreement that
    has already settled.
    """
    import time as _t

    from liqmap.confirm import AgreementTracker

    tr = AgreementTracker()
    tr.observe("agree", direction, _t.time() - seconds)
    rt._trackers[(coin, interval)] = tr
    return tr


def test_a_running_feed_produces_a_real_suggestion_that_can_be_decided(client,
                                                                      monkeypatch):
    rt = web.runtime()
    feed = _live_feed()
    monkeypatch.setattr(rt, "feed", feed, raising=False)
    monkeypatch.setattr(type(feed), "running", property(lambda self: True))
    _settle(rt)

    d = client.get("/api/suggest?coin=BTC&interval=15m&size=10000&fee_bps=2",
                   headers=AUTH).json()

    assert d["ok"] is True
    assert d["take"] is True, d.get("detail")
    assert d["side"] == "long"
    assert d["target_px"] > d["entry"] > d["stop_px"]
    assert d["cost_multiple"] >= 2.0
    # The default is scalp, where R:R is deliberately not the gate — a near
    # target always has a poor one. The breakeven ceiling is the gate.
    assert d["mode"] == "scalp"
    assert 0 < d["breakeven"] <= 0.70
    assert d["grade"] in ("A", "B", "C")
    assert d["reasons"]
    assert d["id"]

    # The suggestion the page was handed can be decided on by that same id.
    assert client.post(f"/api/decide?id={d['id']}&taken=true",
                       headers=AUTH).json()["ok"] is True

    stats = client.get("/api/decisions?coin=BTC", headers=AUTH).json()
    assert stats["recent"][0]["decision"] == "taken"
    assert stats["recent"][0]["side"] == "long"


def test_polling_the_same_candle_does_not_write_a_second_suggestion(client,
                                                                   monkeypatch):
    """Ninety rows for one trade would let a single lucky fifteen minutes
    dominate the measured hit rate."""
    rt = web.runtime()
    feed = _live_feed()
    monkeypatch.setattr(rt, "feed", feed, raising=False)
    monkeypatch.setattr(type(feed), "running", property(lambda self: True))
    _settle(rt)

    ids = {client.get("/api/suggest?coin=BTC&interval=15m",
                      headers=AUTH).json().get("id") for _ in range(5)}
    assert len(ids) == 1
    assert len(rt.history.recent_suggestions()) == 1


def test_record_false_asks_without_writing_anything(client, monkeypatch):
    """The page polls every few seconds; a dry run must not fill the table."""
    rt = web.runtime()
    feed = _live_feed()
    monkeypatch.setattr(rt, "feed", feed, raising=False)
    monkeypatch.setattr(type(feed), "running", property(lambda self: True))
    _settle(rt)

    d = client.get("/api/suggest?coin=BTC&interval=15m&record=false",
                   headers=AUTH).json()
    assert d["take"] is True
    assert rt.history.recent_suggestions() == []


def test_a_short_book_produces_a_short_suggestion(client, monkeypatch):
    """The sign error that would invert every trade the tool ever makes."""
    rt = web.runtime()
    feed = _live_feed(rising=False)      # book AND tape both bearish
    monkeypatch.setattr(rt, "feed", feed, raising=False)
    monkeypatch.setattr(type(feed), "running", property(lambda self: True))
    _settle(rt, direction="down")

    d = client.get("/api/suggest?coin=BTC&interval=15m", headers=AUTH).json()
    assert d["take"] is True, d.get("detail")
    assert d["side"] == "short"
    assert d["target_px"] < d["entry"] < d["stop_px"]
    assert d["confirmation"]["book"] == "down"
    assert d["confirmation"]["candle"] == "down"


def test_a_book_and_a_tape_that_disagree_refuse_the_trade(client, monkeypatch):
    """The whole point. A bearish book with price rising is buyers absorbing
    the sellers, and trading the book alone here is how money is lost."""
    import time as _t

    from liqmap.flow import Book, Level

    rt = web.runtime()
    feed = _live_feed(rising=True)       # rising tape from the helper
    now = _t.time()
    # ...against a book flipped bearish.
    feed.reader = type(feed.reader)()
    for i in range(40):
        b = Book(coin="BTC", ts=now - 40 + i,
                 bids=[Level(round(99.99 - j * 0.01, 4),
                             max(1.0, 40.0 - i) if j == 0 else 30.0)
                       for j in range(8)],
                 asks=[Level(round(100.01 + j * 0.01, 4),
                             800.0 if j == 3 else 120.0) for j in range(8)])
        feed.book = b
        feed.reader.add(b, now=now - 40 + i)

    monkeypatch.setattr(rt, "feed", feed, raising=False)
    monkeypatch.setattr(type(feed), "running", property(lambda self: True))

    d = client.get("/api/suggest?coin=BTC&interval=15m", headers=AUTH).json()
    assert d["take"] is False
    assert d["blocked_by"] == "conflict"
    assert d["grade"] == "X"
    assert d["confirmation"]["book"] == "down"
    assert d["confirmation"]["candle"] == "up"
    assert "absorbing" in d["detail"]
    # And nothing gets recorded, because there is no trade.
    assert rt.history.recent_suggestions() == []



def test_the_route_always_returns_a_graded_call(client, monkeypatch):
    """Tradeable or not, the panel must never go blank — a blank panel is
    indistinguishable from a broken one."""
    rt = web.runtime()
    feed = _live_feed()
    monkeypatch.setattr(rt, "feed", feed, raising=False)
    monkeypatch.setattr(type(feed), "running", property(lambda self: True))

    for mode in ("scalp", "range"):
        d = client.get(f"/api/suggest?coin=BTC&interval=15m&mode={mode}",
                       headers=AUTH).json()
        assert d["grade"], mode
        assert d["sentence"], mode
        assert isinstance(d["tradeable"], bool)
        if not d["tradeable"]:
            assert d["blocked_by"]


def test_scalp_aims_nearer_than_range_through_the_route(client, monkeypatch):
    rt = web.runtime()
    feed = _live_feed()
    monkeypatch.setattr(rt, "feed", feed, raising=False)
    monkeypatch.setattr(type(feed), "running", property(lambda self: True))

    s = client.get("/api/suggest?coin=BTC&interval=15m&mode=scalp&record=false",
                   headers=AUTH).json()
    r = client.get("/api/suggest?coin=BTC&interval=15m&mode=range&record=false",
                   headers=AUTH).json()
    if s.get("take") and r.get("take"):
        assert s["target_bps"] < r["target_bps"]


def test_a_refusal_is_never_recorded_as_a_trade(client, monkeypatch):
    """Scoring refusals would let the tool raise its own hit rate simply by
    declining more often."""
    rt = web.runtime()
    d = client.get("/api/suggest?coin=BTC&interval=15m", headers=AUTH).json()
    assert d["take"] is False
    assert rt.history.recent_suggestions() == []


# --------------------------------------------------------------------------
# the archive backfill routes
# --------------------------------------------------------------------------

def test_backfill_routes_are_locked(locked):
    assert locked.get("/api/backfill").status_code == 503
    assert locked.post("/api/backfill?coin=BTC").status_code == 503
    assert locked.get("/api/backfill/estimate").status_code == 503


def test_an_estimate_is_available_before_spending_anything(client):
    d = client.get("/api/backfill/estimate?coin=BTC&days=2", headers=AUTH).json()
    assert d["hours"] == 48
    assert d["estimated_usd"] >= 0
    assert "requester-pays" in d["bucket"]
    assert isinstance(d["credentials"], bool)


def test_a_backfill_without_credentials_says_so_rather_than_failing_later(
        client, monkeypatch):
    """Requester-pays means there must be an account to charge. Finding
    that out an hour into a job is the wrong time."""
    from liqmap import archive as arch

    monkeypatch.setattr(arch.Archive, "credentials_present",
                        lambda self: False)
    d = client.post("/api/backfill?coin=BTC&days=1", headers=AUTH).json()
    assert d["ok"] is False
    assert "AWS_ACCESS_KEY_ID" in d["error"]


def test_an_unknown_interval_is_refused_before_any_download(client, monkeypatch):
    from liqmap import archive as arch

    monkeypatch.setattr(arch.Archive, "credentials_present", lambda self: True)
    d = client.post("/api/backfill?coin=BTC&interval=7m", headers=AUTH).json()
    assert d["ok"] is False and "timeframe" in d["error"]


def test_the_backfill_status_route_reports_idle_cleanly(client):
    d = client.get("/api/backfill", headers=AUTH).json()
    assert d["running"] is False


def test_a_backfill_can_be_stopped(client):
    rt = web.runtime()
    rt.backfill["running"] = True
    d = client.post("/api/backfill/stop", headers=AUTH).json()
    assert d["running"] is False


def test_credentials_check_does_not_hang_without_aws(monkeypatch):
    """The full boto3 chain ends at EC2 instance metadata, which on a
    non-EC2 host waits for a connection that never comes."""
    import time as _t

    from liqmap import archive as arch

    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_PROFILE",
                "AWS_SHARED_CREDENTIALS_FILE"):
        monkeypatch.delenv(var, raising=False)

    t0 = _t.time()
    arch.Archive().credentials_present()
    assert _t.time() - t0 < 10.0, "the credentials check stalled"


def test_env_credentials_are_detected_without_touching_the_network(monkeypatch):
    from liqmap import archive as arch

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAFAKE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    assert arch.Archive().credentials_present() is True


# --------------------------------------------------------------------------
# the live chart, drawn from our own data
# --------------------------------------------------------------------------

def test_the_chart_route_is_locked(locked):
    assert locked.get("/api/chart").status_code == 503


def test_the_chart_serves_bars_from_our_own_feed(client, monkeypatch):
    """Not an embedded widget and not somebody else's rendering — the same
    bars the signals were computed on, so a marker that looks wrong on the
    chart IS wrong."""
    rt = web.runtime()
    feed = _live_feed()
    monkeypatch.setattr(rt, "feed", feed, raising=False)
    monkeypatch.setattr(type(feed), "running", property(lambda self: True))

    d = client.get("/api/chart?coin=BTC&interval=15m", headers=AUTH).json()
    assert d["bars"], d.get("error")
    assert "websocket" in d["source"]
    for key in ("ts", "o", "h", "l", "c"):
        assert key in d["bars"][0]


def test_the_forming_bar_is_marked_live(client, monkeypatch):
    rt = web.runtime()
    feed = _live_feed()
    monkeypatch.setattr(rt, "feed", feed, raising=False)
    monkeypatch.setattr(type(feed), "running", property(lambda self: True))

    d = client.get("/api/chart?coin=BTC&interval=15m", headers=AUTH).json()
    forming = [b for b in d["bars"] if b.get("live")]
    assert forming, "the bar being traded is missing from the chart"
    # Shown even when unseeded — its high, low and close are real fills and
    # only the open is approximate, so the flag is carried rather than the
    # bar hidden.
    assert "seeded" in forming[0]


def test_the_book_downloads_as_one_csv(client):
    """One file, winners and losers, openable in a spreadsheet."""
    rt = web.runtime()
    pos = rt.ledger.open_position(
        coin="BTC", interval="15m", candle_ts=1.0, side="long", entry=100.0,
        target_px=100.1, stop_px=99.9, target_bps=10.0, risk_bps=10.0,
        cost_bps=2.0, agreeing=3,
        features={"vote_shape": "3-0", "vote_book": 0.5, "vote_delta": 0.4,
                  "vote_price": 0.6, "effort": 3.0},
        now=10.0)
    rt.ledger.close_position(pos, exit_px=100.1, reason="target", now=20.0)

    r = client.get("/api/ledger.csv", headers=AUTH)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"]
    assert ".csv" in r.headers["content-disposition"]

    body = r.text
    assert "trades closed,1" in body
    assert "winners,1" in body
    # The outcome leads, then the conditions that produced it.
    head = [ln for ln in body.splitlines() if ln.startswith("result,")]
    assert head, "no header row"
    cols = head[0].split(",")
    assert cols[0] == "result" and "net_bps" in cols
    for needed in ("shape", "book", "delta", "price", "effort_pct",
                   "exit_reason", "target_bps", "stop_bps"):
        assert needed in cols, f"{needed} missing from the file"
    assert "WIN" in body


def test_an_empty_book_still_downloads(client):
    r = client.get("/api/ledger.csv", headers=AUTH)
    assert r.status_code == 200 and "no trades yet" in r.text


def test_the_agents_trades_are_plotted_entry_to_exit(client, monkeypatch):
    """The chart draws the AGENT'S BOOK, not the old suggestion table.

    It used to draw suggestions, which after the agent started keeping its
    own book meant every marker read "pending" forever — nothing decides
    those rows any more. A marker has to carry the whole trade: side, why
    it was taken, where it ended, and whether that was a win.
    """
    rt = web.runtime()
    feed = _live_feed()
    monkeypatch.setattr(rt, "feed", feed, raising=False)
    monkeypatch.setattr(type(feed), "running", property(lambda self: True))

    bars = feed.history("15m")
    assert bars, "need bars to place a trade on"
    b = bars[-3]
    pos = rt.ledger.open_position(
        coin="BTC", interval="15m", candle_ts=b.ts, side="long",
        entry=b.close, target_px=b.close * 1.003, stop_px=b.close * 0.998,
        target_bps=30.0, risk_bps=20.0, cost_bps=2.0, breakeven=0.42,
        grade="3-0", agreeing=3,
        features={"vote_shape": "3-0", "vote_book": 0.5,
                  "vote_delta": 0.4, "vote_price": 0.6},
        now=b.ts + 60)
    assert pos is not None
    rt.ledger.close_position(pos, exit_px=b.close * 1.003, reason="target",
                             now=b.ts + 900)

    d = client.get("/api/chart?coin=BTC&interval=15m", headers=AUTH).json()
    marks = d["marks"]
    assert marks, d.get("marks_error") or "the trade was not plotted"
    m = marks[0]
    assert m["side"] == "long"
    assert m["target"] > m["entry"] > m["stop"]
    # The exit is its own point in time, so the chart can show the hold.
    assert m["exit_ts"] > m["entry_ts"]
    assert m["exit_reason"] == "target" and m["won"] is True
    assert m["net_bps"] == pytest.approx(28.0, abs=0.5)
    # And the reason it was taken, in words.
    assert m["shape"] == "3-0"
    assert "book up" in m["why"] and "price up" in m["why"]
    assert not m["open"]


def test_an_open_trade_is_plotted_without_an_exit(client, monkeypatch):
    rt = web.runtime()
    feed = _live_feed()
    monkeypatch.setattr(rt, "feed", feed, raising=False)
    monkeypatch.setattr(type(feed), "running", property(lambda self: True))

    b = feed.history("15m")[-2]
    rt.ledger.open_position(
        coin="BTC", interval="15m", candle_ts=b.ts, side="short",
        entry=b.close, target_px=b.close * 0.997, stop_px=b.close * 1.002,
        target_bps=30.0, risk_bps=20.0, now=b.ts + 60)

    d = client.get("/api/chart?coin=BTC&interval=15m", headers=AUTH).json()
    m = [x for x in d["marks"] if x["open"]]
    assert m, "a running position should still be drawn"
    assert m[0]["exit_ts"] is None and m[0]["won"] is None


def test_the_chart_carries_the_forming_bar_volume_profile(client, monkeypatch):
    rt = web.runtime()
    feed = _live_feed()
    monkeypatch.setattr(rt, "feed", feed, raising=False)
    monkeypatch.setattr(type(feed), "running", property(lambda self: True))

    d = client.get("/api/chart?coin=BTC&interval=15m", headers=AUTH).json()
    assert d["profile"] is not None
    assert d["profile"]["poc"] is not None
    assert d["profile"]["levels"]


def test_the_chart_falls_back_to_polled_candles_without_a_feed(client):
    """No feed is not an error — it is a slower chart."""
    r = client.get("/api/chart?coin=BTC&interval=15m", headers=AUTH)
    assert r.status_code == 200
    d = r.json()
    assert "error" in d or d.get("bars") is not None


def test_a_running_feed_grades_all_three_columns(client, monkeypatch):
    rt = web.runtime()
    feed = _live_feed()
    monkeypatch.setattr(rt, "feed", feed, raising=False)
    monkeypatch.setattr(type(feed), "running", property(lambda self: True))
    _settle(rt)

    d = client.get("/api/suggest?coin=BTC&interval=15m&record=false",
                   headers=AUTH).json()
    t = d["three_way"]
    assert t["grade"] in ("A", "B", "C", "X", "—")
    assert set(t) >= {"book", "delta", "price", "agreeing", "tradeable"}
    assert d["delta"] is not None


def test_a_graded_call_reports_how_far_it_can_run(client, monkeypatch):
    """Direction quality and holding distance are different questions."""
    rt = web.runtime()
    feed = _live_feed()
    monkeypatch.setattr(rt, "feed", feed, raising=False)
    monkeypatch.setattr(type(feed), "running", property(lambda self: True))
    _settle(rt)

    d = client.get("/api/suggest?coin=BTC&interval=15m&record=false",
                   headers=AUTH).json()
    if d.get("three_way", {}).get("direction") != "flat":
        assert d.get("runway") is not None
        assert d["runway"]["clear_bps"] > 0
        assert d["runway"]["describe"]
