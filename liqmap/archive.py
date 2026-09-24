"""Hyperliquid's own S3 archive: real historical order books.

I told Chris repeatedly that historical L2 depth does not exist. That was
wrong, and this module is the correction. Hyperliquid publishes book
snapshots to a public requester-pays bucket:

    s3://hyperliquid-archive/market_data/{YYYYMMDD}/{H}/l2Book/{COIN}.lz4

Twice per second, every level the feed carried, back years. That is a
HIGHER resolution than the throttled live public WebSocket currently
delivers, which makes the archive better evidence than anything recorded
live from the public feed today.

THE FILE FORMAT

LZ4 frame, containing JSON Lines. One snapshot per line:

    {"time": "2025-03-11T09:00:03.296545473",
     "ver_num": 1,
     "raw": {"channel": "l2Book",
             "data": {"coin": "SOL",
                      "time": 1741683599768,
                      "levels": [[{"px": "124.04", "sz": "341.81", "n": 2}, ...],
                                 [{"px": "124.05", "sz": "427.89", "n": 5}, ...]]}}}

`levels[0]` is bids, `levels[1]` is asks. `time` at the top is the
recorder's wall clock; `raw.data.time` is the exchange's, in milliseconds,
and that is the one to trust for ordering.

WHAT IS AND IS NOT IN THIS BUCKET

Only `l2Book`. No trades, no candles, no spot. The `market_data` prefix
carries book hours and nothing else -- confirmed against the live bucket by
more than one independent pipeline, and worth stating because the path
`market_data/.../trades/` looks like it should exist and does not.

That matters for the agreement backfill. The book side comes from here; the
price-action side must come from fills, and taking it from these snapshots'
own mid would be the circularity the whole confirmation design exists to
avoid. So price action is built from one-minute candles, which the exchange
computes from actual trades.

IT COSTS REAL MONEY

Requester-pays: every byte is charged to the caller's AWS account, not to
Hyperliquid. A careless date range is a genuinely expensive mistake, so
this module counts bytes, reports them, and refuses to exceed a cap.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Sequence

from .flow import Book, Level

BUCKET = "hyperliquid-archive"
PREFIX = "market_data"

# Requester-pays means a wide date range quietly becomes a large AWS bill.
# The cap is deliberately low and must be raised on purpose.
DEFAULT_MAX_BYTES = 2 * 1024 ** 3          # 2 GiB

# Archived snapshots arrive about twice a second. Feeding every one of them
# through the reader is wasteful for a backfill whose finest question is
# "what did the book look like partway through this bar", so they are
# thinned to roughly this spacing.
DEFAULT_STRIDE_S = 0.5


def _fmt_hour(when: datetime) -> tuple[str, str]:
    """(YYYYMMDD, H) with the hour NOT zero-padded.

    The bucket really does use `.../20230916/9/...`, so zero-padding here
    produces a key that does not exist and a 404 that looks like missing
    data rather than a bad path.
    """
    return when.strftime("%Y%m%d"), str(when.hour)


def key_for(coin: str, when: datetime) -> str:
    day, hour = _fmt_hour(when)
    return f"{PREFIX}/{day}/{hour}/l2Book/{coin}.lz4"


def hours_between(start: datetime, end: datetime) -> list[datetime]:
    """Every hour boundary in [start, end), oldest first."""
    if end <= start:
        return []
    cur = start.replace(minute=0, second=0, microsecond=0)
    out = []
    while cur < end:
        out.append(cur)
        cur += timedelta(hours=1)
    return out


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def parse_snapshot(line: str | bytes, coin_hint: str = "") -> Book | None:
    """One archived line into a `Book`.

    Returns None rather than raising on anything malformed. An archive hour
    with a few torn lines at the end is normal -- the uploader is not
    transactional -- and losing the whole hour to one bad line would be a
    poor trade.
    """
    if isinstance(line, bytes):
        try:
            line = line.decode("utf-8")
        except UnicodeDecodeError:
            return None
    line = line.strip()
    if not line:
        return None

    try:
        row = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(row, dict):
        return None

    # The payload is nested under `raw` in the archive, but the same shape
    # arrives bare off the WebSocket. Accept either, so this parser works
    # on a spooled live frame too.
    data = row.get("raw")
    data = (data or {}).get("data") if isinstance(data, dict) else None
    if data is None:
        data = row.get("data") if isinstance(row.get("data"), dict) else row
    if not isinstance(data, dict):
        return None

    levels = data.get("levels")
    if not isinstance(levels, (list, tuple)) or len(levels) < 2:
        return None

    coin = data.get("coin") or coin_hint
    ts_ms = data.get("time")
    try:
        ts = float(ts_ms) / 1000.0 if ts_ms is not None else 0.0
    except (TypeError, ValueError):
        ts = 0.0
    if ts <= 0:
        # Fall back to the recorder's clock, which is ISO with nanoseconds.
        ts = _parse_iso(row.get("time")) or 0.0
    if ts <= 0:
        return None

    def side(raw: Any) -> list[Level]:
        out: list[Level] = []
        if not isinstance(raw, (list, tuple)):
            return out
        for lv in raw:
            if not isinstance(lv, dict):
                continue
            try:
                px = float(lv["px"])
                sz = float(lv["sz"])
            except (KeyError, TypeError, ValueError):
                continue
            if px > 0 and sz > 0:
                out.append(Level(px, sz))
        return out

    bids, asks = side(levels[0]), side(levels[1])
    if not bids or not asks:
        return None

    # The archive stores bids descending and asks ascending. Sorting anyway
    # costs nothing on twenty levels and removes a silent assumption: a book
    # in the wrong order makes `best_bid` a deep level and every reading
    # downstream nonsense.
    bids.sort(key=lambda l: -l.px)
    asks.sort(key=lambda l: l.px)
    if bids[0].px >= asks[0].px:
        return None                     # crossed or locked; not usable

    return Book(coin=coin, ts=ts, bids=bids, asks=asks)


def _parse_iso(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    # Python's fromisoformat rejects nanosecond precision; trim to micros.
    if "." in text:
        head, _, tail = text.partition(".")
        digits = ""
        rest = ""
        for i, ch in enumerate(tail):
            if ch.isdigit():
                digits += ch
            else:
                rest = tail[i:]
                break
        text = f"{head}.{digits[:6]}{rest}"
    try:
        d = datetime.fromisoformat(text)
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.timestamp()


def iter_snapshots(blob: bytes, coin_hint: str = "",
                   stride_s: float = 0.0) -> Iterator[Book]:
    """Every usable book in one decompressed hour, oldest first.

    `stride_s` thins the stream: snapshots closer together than this to the
    previous one are skipped. At two per second an hour is 7,200 books, and
    a backfill asking one question per bar does not need all of them.
    """
    last = 0.0
    for raw in io.BytesIO(blob):
        book = parse_snapshot(raw, coin_hint)
        if book is None:
            continue
        if stride_s > 0 and last and (book.ts - last) < stride_s:
            continue
        last = book.ts
        yield book


def decompress(blob: bytes) -> bytes:
    """LZ4 frame -> raw bytes.

    Imported lazily and with a readable failure: `lz4` is a compiled
    dependency, and "No module named lz4" surfacing from inside a
    background backfill is a confusing way to learn it is missing.
    """
    try:
        import lz4.frame
    except ImportError as exc:      # pragma: no cover - environment specific
        raise RuntimeError(
            "the lz4 package is required to read the Hyperliquid archive — "
            "add `lz4` to requirements.txt and redeploy") from exc
    return lz4.frame.decompress(blob)


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------

@dataclass
class Transfer:
    """What a fetch actually cost, in bytes and objects."""

    objects: int = 0
    bytes_down: int = 0
    missing: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def gib(self) -> float:
        return self.bytes_down / 1024 ** 3

    def describe(self) -> str:
        # Priced from AWS's standard egress rate at time of writing. Stated
        # as an estimate on purpose: the real number is on the bill, and
        # rates differ by region and by how much the account already moved
        # this month.
        est = self.gib * 0.09
        return (f"{self.objects} hours, {self.gib:.2f} GiB transferred "
                f"(roughly ${est:.2f} at $0.09/GiB egress — check your own "
                f"AWS bill for the real figure)"
                + (f", {self.missing} hours missing from the archive"
                   if self.missing else ""))


class Archive:
    """Requester-pays reader for the Hyperliquid book archive.

    Credentials come from the usual AWS environment variables or instance
    role. The bucket is public to read but the CALLER pays transfer, so
    anonymous access is not an option -- there has to be an account to
    charge.
    """

    def __init__(self, max_bytes: int = DEFAULT_MAX_BYTES,
                 bucket: str = BUCKET, client: Any = None):
        self.bucket = bucket
        self.max_bytes = max_bytes
        self.transfer = Transfer()
        self._client = client

    @property
    def client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:   # pragma: no cover
                raise RuntimeError(
                    "the boto3 package is required to read the Hyperliquid "
                    "archive — add `boto3` to requirements.txt and "
                    "redeploy") from exc
            self._client = boto3.client(
                "s3", region_name=os.environ.get("AWS_REGION", "ap-northeast-1"))
        return self._client

    def credentials_present(self) -> bool:
        """Whether there is anything to charge.

        Checked before a long job rather than failing on the first object an
        hour in. The environment variables are tested first and answered
        without touching the network: the full boto3 chain ends at EC2
        instance metadata, which on a host that is not EC2 waits for a
        connection that will never come. A credentials check that hangs the
        request is worse than one that says no.
        """
        if os.environ.get("AWS_ACCESS_KEY_ID") and \
                os.environ.get("AWS_SECRET_ACCESS_KEY"):
            return True
        if os.environ.get("AWS_PROFILE") or os.environ.get(
                "AWS_SHARED_CREDENTIALS_FILE"):
            return True
        try:
            import boto3
            from botocore.config import Config

            # One short attempt, no retries. Enough to find a real role,
            # not enough to stall a page load.
            prev = os.environ.get("AWS_METADATA_SERVICE_TIMEOUT")
            os.environ["AWS_METADATA_SERVICE_TIMEOUT"] = "1"
            try:
                session = boto3.Session()
                session._session.set_config_variable(
                    "metadata_service_num_attempts", 1)
                return session.get_credentials() is not None
            finally:
                if prev is None:
                    os.environ.pop("AWS_METADATA_SERVICE_TIMEOUT", None)
                else:
                    os.environ["AWS_METADATA_SERVICE_TIMEOUT"] = prev
        except Exception:
            return False

    def fetch_hour(self, coin: str, when: datetime) -> bytes | None:
        """One hour of snapshots, decompressed. None when absent.

        A missing hour is normal and not an error: the archive is uploaded
        roughly monthly with no guarantee of completeness, so gaps are a
        property of the source rather than a failure of this code.
        """
        if self.transfer.bytes_down >= self.max_bytes:
            raise TransferCapReached(
                f"stopped at {self.transfer.gib:.2f} GiB — the cap is "
                f"{self.max_bytes / 1024 ** 3:.2f} GiB. Raise it on purpose "
                f"if you mean to download more; this is a requester-pays "
                f"bucket and every byte is on your AWS bill")

        key = key_for(coin, when)
        try:
            resp = self.client.get_object(Bucket=self.bucket, Key=key,
                                          RequestPayer="requester")
            blob = resp["Body"].read()
        except Exception as exc:
            name = type(exc).__name__
            if "NoSuchKey" in name or "404" in str(exc) or "NoSuchKey" in str(exc):
                self.transfer.missing += 1
                return None
            self.transfer.errors.append(f"{key}: {exc}")
            return None

        self.transfer.objects += 1
        self.transfer.bytes_down += len(blob)
        try:
            return decompress(blob)
        except Exception as exc:
            self.transfer.errors.append(f"{key}: decompress: {exc}")
            return None

    def books(self, coin: str, start: datetime, end: datetime,
              stride_s: float = DEFAULT_STRIDE_S) -> Iterator[Book]:
        """Every archived book in the window, oldest first."""
        for hour in hours_between(start, end):
            blob = self.fetch_hour(coin, hour)
            if blob is None:
                continue
            yield from iter_snapshots(blob, coin_hint=coin, stride_s=stride_s)


class TransferCapReached(RuntimeError):
    """Raised when a backfill would exceed its byte budget."""


def estimate(coins: Sequence[str], start: datetime, end: datetime,
             gib_per_coin_hour: float = 0.01) -> dict[str, Any]:
    """Rough size of a job before running it.

    The per-hour figure is a placeholder that the first real hour replaces.
    It exists so the dashboard can warn about a three-month, ten-coin
    request BEFORE the money is spent, not after.
    """
    hours = len(hours_between(start, end))
    total = hours * max(len(coins), 1) * gib_per_coin_hour
    return {
        "hours": hours, "coins": len(coins),
        "object_count": hours * max(len(coins), 1),
        "estimated_gib": round(total, 2),
        "estimated_usd": round(total * 0.09, 2),
        "note": ("A rough estimate from an assumed hourly size. The job "
                 "measures and reports actual bytes as it runs, and stops "
                 "at the cap."),
    }
