"""Reading Hyperliquid's S3 book archive, and backfilling from it.

The archive format is documented by example rather than by schema, so every
fixture here is built to match the one published sample exactly:

    {"time": "2025-03-11T09:00:03.296545473",
     "ver_num": 1,
     "raw": {"channel": "l2Book",
             "data": {"coin": "SOL", "time": 1741683599768,
                      "levels": [[bids...], [asks...]]}}}

What these tests guard:

THE KEY PATH. The bucket uses a NON-zero-padded hour — `.../20230916/9/...`.
Padding it produces a 404 that reads as missing data rather than a bad
path, and the whole backfill silently returns nothing.

TORN LINES DO NOT LOSE THE HOUR. The uploader is not transactional, so a
truncated final line is normal. Losing 7,200 snapshots to one bad line is a
poor trade.

THE TWO SIDES STAY INDEPENDENT. The book side comes from the archive; the
price side must come from fills. If the backfill ever derives price action
from the snapshots' own mid, the confirmation rate becomes the book
agreeing with itself and the whole table is worthless. Asserted
structurally, not by reading the code.

MONEY IS BOUNDED. Requester-pays means a careless date range is an
expensive mistake, so the cap must actually stop the job.
"""

import io
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from liqmap import archive as arch
from liqmap.backfill import backfill
from liqmap.history import History
from liqmap.structure import Candle

T0 = datetime(2025, 3, 11, 9, 0, tzinfo=timezone.utc)


def snapshot(ts_ms, mid=124.0, bid_sz=341.81, ask_sz=427.89, coin="SOL",
             levels=6, tick=0.01):
    bids = [{"px": f"{mid - tick * (i + 1):.4f}",
             "sz": f"{bid_sz:.2f}", "n": 2} for i in range(levels)]
    asks = [{"px": f"{mid + tick * (i + 1):.4f}",
             "sz": f"{ask_sz:.2f}", "n": 5} for i in range(levels)]
    stamp = datetime.fromtimestamp(ts_ms / 1000.0, timezone.utc)
    return json.dumps({
        "time": stamp.isoformat().replace("+00:00", "") + "123456789",
        "ver_num": 1,
        "raw": {"channel": "l2Book",
                "data": {"coin": coin, "time": ts_ms,
                         "levels": [bids, asks]}}})


def hour_blob(start_ms, n=20, step_ms=500, **kw):
    return "\n".join(snapshot(start_ms + i * step_ms, **kw)
                     for i in range(n)).encode()


# --------------------------------------------------------------------------
# the key path
# --------------------------------------------------------------------------

def test_the_hour_in_the_key_is_not_zero_padded():
    """`.../20230916/9/...`, not `/09/`. A padded hour is a 404 that looks
    exactly like an hour the archive never uploaded."""
    key = arch.key_for("SOL", datetime(2023, 9, 16, 9, tzinfo=timezone.utc))
    assert key == "market_data/20230916/9/l2Book/SOL.lz4"


def test_midnight_and_late_hours_build_correctly():
    assert arch.key_for("BTC", datetime(2026, 1, 2, 0, tzinfo=timezone.utc)) \
        == "market_data/20260102/0/l2Book/BTC.lz4"
    assert arch.key_for("BTC", datetime(2026, 1, 2, 23, tzinfo=timezone.utc)) \
        == "market_data/20260102/23/l2Book/BTC.lz4"


def test_hip3_symbols_keep_their_dex_prefix():
    key = arch.key_for("vntl:GOLD", datetime(2026, 1, 2, 5, tzinfo=timezone.utc))
    assert key.endswith("l2Book/vntl:GOLD.lz4")


def test_hours_between_is_half_open_and_ordered():
    hours = arch.hours_between(T0, T0 + timedelta(hours=3))
    assert len(hours) == 3
    assert hours[0] == T0 and hours[-1] == T0 + timedelta(hours=2)
    assert arch.hours_between(T0, T0) == []
    assert arch.hours_between(T0 + timedelta(hours=1), T0) == []


# --------------------------------------------------------------------------
# parsing the documented format
# --------------------------------------------------------------------------

def test_a_documented_line_parses_into_a_book():
    b = arch.parse_snapshot(snapshot(1741683599768))
    assert b is not None
    assert b.coin == "SOL"
    assert b.ts == pytest.approx(1741683599.768, abs=0.001)
    assert b.best_bid < b.best_ask
    assert len(b.bids) == 6 and len(b.asks) == 6


def test_levels_zero_is_bids_and_levels_one_is_asks():
    """Getting this backwards inverts every reading downstream and nothing
    else complains."""
    b = arch.parse_snapshot(snapshot(1741683599768, mid=124.0))
    assert b.best_bid == pytest.approx(123.99)
    assert b.best_ask == pytest.approx(124.01)


def test_sides_are_sorted_even_if_the_file_is_not():
    raw = json.loads(snapshot(1741683599768))
    raw["raw"]["data"]["levels"][0].reverse()      # bids ascending
    raw["raw"]["data"]["levels"][1].reverse()      # asks descending
    b = arch.parse_snapshot(json.dumps(raw))
    assert b.best_bid == pytest.approx(123.99)
    assert b.best_ask == pytest.approx(124.01)


def test_the_exchange_timestamp_is_preferred_over_the_recorder_clock():
    raw = json.loads(snapshot(1741683599768))
    raw["time"] = "2030-01-01T00:00:00.000000000"
    b = arch.parse_snapshot(json.dumps(raw))
    assert b.ts == pytest.approx(1741683599.768, abs=0.001)


def test_a_nanosecond_iso_stamp_is_readable_when_the_exchange_time_is_absent():
    raw = json.loads(snapshot(1741683599768))
    raw["raw"]["data"].pop("time")
    b = arch.parse_snapshot(json.dumps(raw))
    assert b is not None and b.ts > 1_700_000_000


def test_a_bare_websocket_frame_parses_too():
    """The same shape arrives live, so one parser should read a spooled
    frame as well as an archive line."""
    frame = json.dumps({"channel": "l2Book", "data": {
        "coin": "BTC", "time": 1741683599768,
        "levels": [[{"px": "100.0", "sz": "5", "n": 1}],
                   [{"px": "100.1", "sz": "3", "n": 1}]]}})
    b = arch.parse_snapshot(frame)
    assert b is not None and b.coin == "BTC"


@pytest.mark.parametrize("bad", [
    "", "   ", "not json", "{}", "[]", "null",
    '{"raw": {"data": {}}}',
    '{"raw": {"data": {"levels": []}}}',
    '{"raw": {"data": {"levels": [[], []]}}}',
])
def test_malformed_lines_return_none_rather_than_raising(bad):
    assert arch.parse_snapshot(bad) is None


def test_a_crossed_book_is_rejected():
    raw = json.loads(snapshot(1741683599768))
    raw["raw"]["data"]["levels"] = [
        [{"px": "101.0", "sz": "1", "n": 1}],
        [{"px": "100.0", "sz": "1", "n": 1}]]
    assert arch.parse_snapshot(json.dumps(raw)) is None


def test_zero_and_negative_sizes_are_dropped():
    raw = json.loads(snapshot(1741683599768))
    raw["raw"]["data"]["levels"][0].insert(0, {"px": "124.0", "sz": "0", "n": 1})
    b = arch.parse_snapshot(json.dumps(raw))
    assert all(l.sz > 0 for l in b.bids)


# --------------------------------------------------------------------------
# iterating an hour
# --------------------------------------------------------------------------

def test_an_hour_yields_every_snapshot_in_order():
    books = list(arch.iter_snapshots(hour_blob(1741683600000, n=20)))
    assert len(books) == 20
    assert [b.ts for b in books] == sorted(b.ts for b in books)


def test_one_torn_line_does_not_lose_the_hour():
    """The uploader is not transactional; a truncated final line is normal."""
    blob = hour_blob(1741683600000, n=10) + b"\n{\"raw\": {\"da"
    assert len(list(arch.iter_snapshots(blob))) == 10


def test_blank_lines_are_skipped():
    blob = b"\n\n" + hour_blob(1741683600000, n=5) + b"\n\n"
    assert len(list(arch.iter_snapshots(blob))) == 5


def test_stride_thins_the_stream():
    """Two per second for an hour is 7,200 books, and a backfill asking one
    question per bar does not need all of them."""
    blob = hour_blob(1741683600000, n=40, step_ms=500)
    assert len(list(arch.iter_snapshots(blob, stride_s=0.0))) == 40
    thinned = list(arch.iter_snapshots(blob, stride_s=2.0))
    assert 8 <= len(thinned) <= 12


def test_the_coin_hint_fills_in_a_missing_name():
    raw = json.loads(snapshot(1741683599768))
    raw["raw"]["data"].pop("coin")
    b = arch.parse_snapshot(json.dumps(raw), coin_hint="ETH")
    assert b.coin == "ETH"


# --------------------------------------------------------------------------
# money
# --------------------------------------------------------------------------

class FakeS3:
    """An in-memory bucket. `missing` keys 404 the way the real one does."""

    def __init__(self, blobs: dict[str, bytes]):
        self.blobs = blobs
        self.requested: list[str] = []
        self.payers: list[str] = []

    def get_object(self, Bucket, Key, RequestPayer=None):   # noqa: N803
        self.requested.append(Key)
        self.payers.append(RequestPayer)
        if Key not in self.blobs:
            raise Exception("NoSuchKey: the specified key does not exist")
        return {"Body": io.BytesIO(self.blobs[Key])}


def compress(blob: bytes) -> bytes:
    import lz4.frame

    return lz4.frame.compress(blob)


def bucket_for(coin="SOL", hours=2, start=T0, n=1800, step_ms=2_000):
    """A bucket whose hours actually SPAN an hour.

    Two things the fixture has to get right, both of which produced a
    backfill that looked broken and was not:

    SPAN. At 500ms apart, 240 snapshots cover two minutes, so the read
    moment a third of the way into a 15m bar is never reached and nothing
    is ever recorded.

    DENSITY. The reader's dynamics window is 20 seconds and it wants at
    least five samples in it. The real archive is about two snapshots per
    second, so 20s holds roughly forty; a sparse fixture leaves the reader
    permanently "warming" and records nothing for a reason that has nothing
    to do with the code under test.
    """
    blobs = {}
    for i in range(hours):
        when = start + timedelta(hours=i)
        ms = int(when.timestamp() * 1000)
        blobs[arch.key_for(coin, when)] = compress(
            hour_blob(ms, n=n, step_ms=step_ms))
    return blobs


def test_every_request_declares_requester_pays():
    """Without this header the bucket refuses, and the failure looks like a
    permissions problem rather than a missing flag."""
    s3 = FakeS3(bucket_for())
    a = arch.Archive(client=s3)
    list(a.books("SOL", T0, T0 + timedelta(hours=1)))
    assert s3.payers and all(p == "requester" for p in s3.payers)


def test_bytes_are_counted_and_reported_in_money():
    s3 = FakeS3(bucket_for(hours=2))
    a = arch.Archive(client=s3)
    list(a.books("SOL", T0, T0 + timedelta(hours=2)))
    assert a.transfer.objects == 2
    assert a.transfer.bytes_down > 0
    text = a.transfer.describe()
    assert "GiB transferred" in text and "$" in text
    assert "check your own AWS bill" in text


def test_the_cap_actually_stops_the_job():
    """A careless date range on a requester-pays bucket is an expensive
    mistake, so the limit has to bite rather than warn."""
    s3 = FakeS3(bucket_for(hours=6))
    a = arch.Archive(max_bytes=1, client=s3)
    with pytest.raises(arch.TransferCapReached):
        list(a.books("SOL", T0, T0 + timedelta(hours=6)))
    assert a.transfer.objects <= 1


def test_a_missing_hour_is_counted_not_raised():
    """The archive is uploaded roughly monthly with no completeness
    guarantee. Gaps are a property of the source."""
    blobs = bucket_for(hours=3)
    del blobs[arch.key_for("SOL", T0 + timedelta(hours=1))]
    a = arch.Archive(client=FakeS3(blobs))
    books = list(a.books("SOL", T0, T0 + timedelta(hours=3)))
    assert books
    assert a.transfer.missing == 1
    assert "missing from the archive" in a.transfer.describe()


def test_an_estimate_is_available_before_spending_anything():
    e = arch.estimate(["BTC"], T0, T0 + timedelta(days=1))
    assert e["hours"] == 24
    assert e["estimated_gib"] > 0 and e["estimated_usd"] > 0
    assert "estimate" in e["note"].lower()


# --------------------------------------------------------------------------
# the backfill
# --------------------------------------------------------------------------

@pytest.fixture
def hist():
    with tempfile.TemporaryDirectory() as tmp:
        h = History(Path(tmp) / "h.db")
        yield h
        h.close()


def minute_bars(start_ts, n=240, px=124.0, drift=0.002):
    """Fills-derived candles. Deliberately NOT built from the book fixture,
    which is the whole point of the independence rule."""
    out = []
    p = px
    for i in range(n):
        o = p
        c = o + drift
        out.append(Candle(ts=start_ts + i * 60.0, open=o,
                          high=max(o, c) + 0.01, low=min(o, c) - 0.01,
                          close=c, volume=10.0))
        p = c
    return out


def test_a_backfill_writes_settled_states(hist):
    s3 = FakeS3(bucket_for(hours=2))
    a = arch.Archive(client=s3)
    bars = minute_bars(T0.timestamp(), n=120)

    p = backfill(a, hist, coin="SOL", interval="15m", interval_s=900.0,
                 start=T0, end=T0 + timedelta(hours=2), minute_bars=bars)

    assert p.done and not p.error, p.error
    assert p.states > 0
    table = hist.agreement_table(coin="SOL", interval="15m")
    assert table["n"] == p.states


def test_backfilled_rows_are_marked_as_coming_from_the_archive(hist):
    a = arch.Archive(client=FakeS3(bucket_for(hours=2)))
    backfill(a, hist, coin="SOL", interval="15m", interval_s=900.0,
             start=T0, end=T0 + timedelta(hours=2),
             minute_bars=minute_bars(T0.timestamp(), n=120))
    assert "source" in hist.stored_features()


def test_the_backfill_stores_the_raw_book_readings(hist):
    """Storing only the verdict would mean every threshold change restarts
    the measurement from zero."""
    a = arch.Archive(client=FakeS3(bucket_for(hours=2)))
    backfill(a, hist, coin="SOL", interval="15m", interval_s=900.0,
             start=T0, end=T0 + timedelta(hours=2),
             minute_bars=minute_bars(T0.timestamp(), n=120))
    names = hist.stored_features()
    for f in ("tilt", "replenish", "depletion", "slope_bps", "position"):
        assert f in names


def test_without_minute_candles_the_backfill_refuses(hist):
    """Rather than falling back to the book's own mid, which would make the
    comparison the book agreeing with itself."""
    a = arch.Archive(client=FakeS3(bucket_for()))
    p = backfill(a, hist, coin="SOL", interval="15m", interval_s=900.0,
                 start=T0, end=T0 + timedelta(hours=1), minute_bars=[])
    assert p.states == 0
    assert "book's own mid" in p.error


def test_one_state_per_bar_at_most(hist):
    a = arch.Archive(client=FakeS3(bucket_for(hours=2)))
    p = backfill(a, hist, coin="SOL", interval="15m", interval_s=900.0,
                 start=T0, end=T0 + timedelta(hours=2),
                 minute_bars=minute_bars(T0.timestamp(), n=120))
    rows = hist.agreement_table(coin="SOL", interval="15m")["n"]
    assert rows == p.states
    assert p.states <= 8        # two hours of 15m bars


def test_running_twice_does_not_double_count(hist):
    bars = minute_bars(T0.timestamp(), n=120)
    for _ in range(2):
        a = arch.Archive(client=FakeS3(bucket_for(hours=2)))
        backfill(a, hist, coin="SOL", interval="15m", interval_s=900.0,
                 start=T0, end=T0 + timedelta(hours=2), minute_bars=bars)
    first = hist.agreement_table(coin="SOL", interval="15m")["n"]

    a = arch.Archive(client=FakeS3(bucket_for(hours=2)))
    p = backfill(a, hist, coin="SOL", interval="15m", interval_s=900.0,
                 start=T0, end=T0 + timedelta(hours=2), minute_bars=bars)
    assert p.skipped.get("already recorded", 0) > 0
    assert hist.agreement_table(coin="SOL", interval="15m")["n"] == first


def test_the_backfill_can_be_stopped(hist):
    a = arch.Archive(client=FakeS3(bucket_for(hours=4)))
    p = backfill(a, hist, coin="SOL", interval="15m", interval_s=900.0,
                 start=T0, end=T0 + timedelta(hours=4),
                 minute_bars=minute_bars(T0.timestamp(), n=240),
                 should_stop=lambda: True)
    assert p.error == "stopped"


def test_a_transfer_cap_during_a_backfill_is_reported_not_raised(hist):
    a = arch.Archive(max_bytes=1, client=FakeS3(bucket_for(hours=4)))
    p = backfill(a, hist, coin="SOL", interval="15m", interval_s=900.0,
                 start=T0, end=T0 + timedelta(hours=4),
                 minute_bars=minute_bars(T0.timestamp(), n=240))
    assert p.done
    assert "cap" in p.error


def test_progress_is_reportable_as_json(hist):
    a = arch.Archive(client=FakeS3(bucket_for(hours=2)))
    p = backfill(a, hist, coin="SOL", interval="15m", interval_s=900.0,
                 start=T0, end=T0 + timedelta(hours=2),
                 minute_bars=minute_bars(T0.timestamp(), n=120))
    d = p.to_dict()
    json.dumps(d)
    for key in ("coin", "states", "transfer", "done", "pct"):
        assert key in d


# --------------------------------------------------------------------------
# the independence rule, asserted structurally
# --------------------------------------------------------------------------

def test_the_backfill_never_builds_price_action_from_the_book():
    """If the price side ever came from the snapshots' own mid, the
    confirmation rate would be the book agreeing with itself — a superb
    number meaning nothing. Checked on the parsed code, so the module
    docstring explaining the rule does not trip it."""
    import ast
    import inspect

    from liqmap import backfill as bf

    tree = ast.parse(inspect.getsource(bf))
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "read_candle":
            calls.append(node)
    assert calls, "read_candle is not called — has the price side moved?"

    for call in calls:
        for kw in call.keywords:
            src = ast.dump(kw.value)
            for banned in ("'mid'", "'best_bid'", "'best_ask'",
                           "'microprice'", "'book'"):
                assert banned not in src, (
                    f"price action argument {kw.arg!r} is derived from the "
                    f"book — that is the circularity this design forbids")
