"""Reading uploaded chart history, and being honest about what it can teach.

You can export weeks of bars from almost anything -- TradingView, an
exchange, a broker -- and this parses those exports into the same `Candle`
the backtest already replays. Column names, column order, delimiter and
timestamp format are all detected rather than required, because an export is
whatever the platform felt like writing that day.

WHAT AN OHLCV FILE CAN TRAIN

Everything derived from candle geometry: where price closed in its range,
wick rejection, sweeps and failed breaks, swing structure, supply and demand
zones, the higher timeframe. Those are real signals and more history makes
them better.

WHAT IT CANNOT TRAIN, AND THIS IS THE IMPORTANT PART

The book signals. Microprice tilt, replenishment, queue depletion and
absorption are roughly seventy percent of the weight in the book call, and
NONE of them exist in an OHLCV file. A bar records four prices and a volume.
It does not record what was resting at the touch, which side kept replacing
size, or whether aggression moved the mid. No exchange serves historical L2
depth at this resolution, so the data does not exist to be uploaded.

That is not a gap this module can close by parsing harder. The book half of
the model can only ever be measured forward, live, while the feed is running
-- which is what the `reads` and `suggestions` tables in `history.py` are
for. Every report produced here says so plainly, because uploading three
months of bars and being told the model is now trained on three months would
be a lie by omission, and it is exactly the lie that would get acted on.

SILENT REPAIRS ARE WORSE THAN LOUD REJECTIONS

Rows that cannot be trusted -- a high below its low, a zero price, a
duplicate timestamp -- are dropped and counted, never patched. The report
names how many and why. A file that arrives 40% malformed should look
alarming, not clean.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from .structure import Candle

# Column aliases, lower-cased and stripped of punctuation. Longest match
# wins, so "close_time" never captures the "close" slot.
ALIASES: dict[str, tuple[str, ...]] = {
    "ts": ("t", "time", "timestamp", "date", "datetime", "opentime",
           "open time", "open_time", "bartime", "unix", "utc", "candlebegin"),
    "open": ("open", "o", "openprice", "open_price"),
    "high": ("high", "h", "highprice", "high_price", "max"),
    "low": ("low", "l", "lowprice", "low_price", "min"),
    "close": ("close", "c", "closeprice", "close_price", "last", "price"),
    "volume": ("volume", "vol", "v", "basevolume", "base_volume", "qty",
               "size", "amount"),
}

# Columns that look like a wanted one but are a different thing entirely.
# "close_time" in a Binance kline is the END of the bar, not its close price.
NEVER: tuple[str, ...] = ("closetime", "close_time", "opentime_ms",
                          "quotevolume", "quote_volume", "numberoftrades",
                          "trades", "takerbuybase", "takerbuyquote", "ignore")

# A file with more than this fraction of rows dropped is not a parse problem,
# it is the wrong file.
MAX_BAD_FRACTION = 0.5


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def _to_ts(value: Any) -> float | None:
    """Epoch seconds from whatever the export wrote.

    Handles epoch seconds, milliseconds, microseconds and ISO 8601 with or
    without a zone. A naive ISO timestamp is read as UTC, which is what every
    export that omits the zone means.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        n = float(value)
    else:
        s = str(value).strip()
        if not s:
            return None
        try:
            n = float(s)
        except ValueError:
            txt = s.replace("Z", "+00:00").replace("/", "-")
            try:
                d = datetime.fromisoformat(txt)
            except ValueError:
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
                            "%Y-%m-%d", "%d-%m-%Y %H:%M", "%m-%d-%Y %H:%M"):
                    try:
                        d = datetime.strptime(txt, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    return None
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            return d.timestamp()

    if not math.isfinite(n) or n <= 0:
        return None
    # Scale by magnitude. 1e11 is year 5138 in seconds and 1973 in ms, so the
    # boundary is never ambiguous for any date anyone has chart data for.
    if n >= 1e17:
        return n / 1e9          # nanoseconds
    if n >= 1e14:
        return n / 1e6          # microseconds
    if n >= 1e11:
        return n / 1e3          # milliseconds
    return n


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if math.isfinite(float(value)) else None
    s = str(value).strip().replace(",", "").replace("$", "")
    if not s or s.lower() in ("nan", "null", "none", "-", "n/a"):
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return f if math.isfinite(f) else None


@dataclass
class Ingested:
    """Parsed bars plus an honest account of the parse."""

    candles: list[Candle]
    interval_s: float
    source: str = ""
    rows_read: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.candles) >= 5

    @property
    def rejected_total(self) -> int:
        return sum(self.rejected.values())

    @property
    def interval_name(self) -> str:
        s = self.interval_s
        if s <= 0:
            return "unknown"
        for secs, name in ((86400, "1d"), (14400, "4h"), (3600, "1h"),
                           (1800, "30m"), (900, "15m"), (300, "5m"),
                           (60, "1m")):
            if abs(s - secs) < max(1.0, secs * 0.02):
                return name
        if s >= 3600:
            return f"{s / 3600:.0f}h"
        return f"{s / 60:.0f}m"

    @property
    def span(self) -> tuple[float, float]:
        if not self.candles:
            return (0.0, 0.0)
        return (self.candles[0].ts, self.candles[-1].ts)

    def gaps(self) -> int:
        """Missing bars, as a count of skipped slots."""
        if self.interval_s <= 0 or len(self.candles) < 2:
            return 0
        missing = 0
        for a, b in zip(self.candles, self.candles[1:]):
            step = round((b.ts - a.ts) / self.interval_s)
            if step > 1:
                missing += step - 1
        return missing

    def describe(self) -> str:
        if not self.candles:
            reasons = ", ".join(f"{n} {k}" for k, n in
                                sorted(self.rejected.items(), key=lambda x: -x[1]))
            return (f"No usable bars in {self.source or 'the file'}. "
                    f"Read {self.rows_read} rows; rejected {reasons or 'all of them'}.")

        start, end = self.span
        fmt = "%Y-%m-%d"
        lines = [
            f"{len(self.candles):,} {self.interval_name} bars from "
            f"{datetime.fromtimestamp(start, timezone.utc).strftime(fmt)} to "
            f"{datetime.fromtimestamp(end, timezone.utc).strftime(fmt)}."
        ]
        if self.rejected_total:
            reasons = ", ".join(f"{n} {k}" for k, n in
                                sorted(self.rejected.items(), key=lambda x: -x[1]))
            lines.append(f"Dropped {self.rejected_total} of {self.rows_read} "
                         f"rows: {reasons}.")
        missing = self.gaps()
        if missing:
            lines.append(f"{missing} bars missing from the series — the "
                         f"replay steps over the holes rather than "
                         f"interpolating them.")
        lines.extend(self.warnings)
        lines.append(TRAINING_CAVEAT)
        return " ".join(lines)

    def to_dict(self) -> dict:
        start, end = self.span
        return {
            "ok": self.ok, "source": self.source,
            "bars": len(self.candles), "rows_read": self.rows_read,
            "rejected": self.rejected, "rejected_total": self.rejected_total,
            "interval_s": self.interval_s, "interval": self.interval_name,
            "start_ts": start, "end_ts": end, "gaps": self.gaps(),
            "warnings": self.warnings,
            "describe": self.describe(),
            "trains": TRAINS, "cannot_train": CANNOT_TRAIN,
        }


TRAINS = ["price action (sweeps, failed breaks, rejection)",
          "position in range", "swing structure", "supply and demand zones",
          "higher timeframe direction"]

CANNOT_TRAIN = ["microprice tilt", "replenishment", "queue depletion",
                "absorption", "aggression"]

TRAINING_CAVEAT = (
    "This trains the candle-geometry half of the model only. Microprice "
    "tilt, replenishment, depletion and absorption are about seventy percent "
    "of the book call's weight and none of them are in an OHLCV file — a bar "
    "records four prices, not what was resting at the touch. No exchange "
    "serves historical book depth at this resolution, so that half can only "
    "be measured forward, live.")


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def _map_columns(header: Sequence[str]) -> dict[str, int]:
    """Which column holds which field. Longest alias wins."""
    out: dict[str, int] = {}
    normed = [_norm(h) for h in header]
    for field_name, aliases in ALIASES.items():
        best: tuple[int, int] | None = None     # (alias length, index)
        for i, h in enumerate(normed):
            if not h or h in NEVER:
                continue
            for a in aliases:
                if h == _norm(a) and (best is None or len(a) > best[0]):
                    best = (len(a), i)
        if best is not None:
            out[field_name] = best[1]
    return out


def _rows_from_csv(text: str) -> tuple[list[list[str]], list[str] | None]:
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delim = dialect.delimiter
    except csv.Error:
        delim = ","

    # newline="" is the documented way to let csv handle line endings itself;
    # without it a stray CR inside a field raises. And the reader can still
    # raise on binary that is not a CSV at all (NUL bytes, absurd field
    # sizes), which is a report to write, not a traceback to return.
    try:
        rows = [r for r in csv.reader(io.StringIO(text, newline=""),
                                      delimiter=delim) if r]
    except (csv.Error, ValueError):
        return [], None
    if not rows:
        return [], None

    # A header row is one whose cells are not numbers.
    first = rows[0]
    numeric = sum(1 for c in first if _to_float(c) is not None)
    if numeric <= len(first) / 2:
        return rows[1:], first
    return rows, None


def _positional(width: int) -> dict[str, int]:
    """Column map for a headerless file.

    Two shapes are common and both start with a timestamp: the five-column
    t,o,h,l,c and the exchange kline t,o,h,l,c,v,... Anything else is a
    guess not worth making.
    """
    if width >= 6:
        return {"ts": 0, "open": 1, "high": 2, "low": 3, "close": 4, "volume": 5}
    if width == 5:
        return {"ts": 0, "open": 1, "high": 2, "low": 3, "close": 4}
    return {}


def parse(data: str | bytes, source: str = "") -> Ingested:
    """Parse CSV or JSON chart history into candles.

    Never raises on malformed input: a bad file comes back as an Ingested
    with no candles and a report saying why, because an upload endpoint that
    500s tells the user nothing about their file.
    """
    if isinstance(data, bytes):
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = data.decode("latin-1", errors="replace")
    else:
        text = data
    text = text.strip()

    if not text:
        return Ingested([], 0.0, source=source, rejected={"empty file": 1})

    rows: list[Sequence[Any]]
    columns: dict[str, int]

    if text[0] in "[{":
        parsed = _json_rows(text)
        if parsed is None:
            return Ingested([], 0.0, source=source,
                            rejected={"unreadable JSON": 1})
        rows, columns = parsed
    else:
        raw, header = _rows_from_csv(text)
        rows = raw
        columns = _map_columns(header) if header else _positional(
            len(raw[0]) if raw else 0)

    missing = [f for f in ("ts", "open", "high", "low", "close")
               if f not in columns]
    if missing:
        return Ingested(
            [], 0.0, source=source, rows_read=len(rows),
            rejected={f"no column for {', '.join(missing)}": 1},
            warnings=["Expected columns for time, open, high, low and close. "
                      "Header names are matched loosely (t/time/date, o/open, "
                      "…), so a file with no header must be in "
                      "time,open,high,low,close order."])

    return _build(rows, columns, source)


def _json_rows(text: str) -> tuple[list[Sequence[Any]], dict[str, int]] | None:
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None

    if isinstance(obj, dict):
        for key in ("candles", "data", "result", "bars", "klines", "rows"):
            if isinstance(obj.get(key), list):
                obj = obj[key]
                break
        else:
            return None
    if not isinstance(obj, list) or not obj:
        return None

    first = obj[0]
    if isinstance(first, dict):
        keys = list(first.keys())
        columns = _map_columns(keys)
        rows = [[r.get(k) for k in keys] for r in obj if isinstance(r, dict)]
        return rows, columns
    if isinstance(first, (list, tuple)):
        rows = [list(r) for r in obj if isinstance(r, (list, tuple))]
        return rows, _positional(len(first))
    return None


def _build(rows: Sequence[Sequence[Any]], columns: dict[str, int],
           source: str) -> Ingested:
    rejected: dict[str, int] = {}
    warnings: list[str] = []
    seen: dict[float, Candle] = {}

    def reject(why: str) -> None:
        rejected[why] = rejected.get(why, 0) + 1

    width = max(columns.values()) + 1
    vol_at = columns.get("volume")

    for row in rows:
        if len(row) < width:
            reject("too few columns")
            continue

        ts = _to_ts(row[columns["ts"]])
        if ts is None:
            reject("unreadable timestamp")
            continue

        o = _to_float(row[columns["open"]])
        h = _to_float(row[columns["high"]])
        lo = _to_float(row[columns["low"]])
        c = _to_float(row[columns["close"]])
        if None in (o, h, lo, c):
            reject("unreadable price")
            continue
        if min(o, h, lo, c) <= 0:
            reject("non-positive price")
            continue
        if h < lo:
            reject("high below low")
            continue
        # The open and close must sit inside the bar's range or the row is
        # not describing one bar. Repairing it would invent a candle.
        if not (lo <= o <= h and lo <= c <= h):
            reject("open or close outside the range")
            continue

        v = 0.0
        if vol_at is not None and vol_at < len(row):
            v = _to_float(row[vol_at]) or 0.0

        if ts in seen:
            reject("duplicate timestamp")
            continue
        seen[ts] = Candle(ts=ts, open=o, high=h, low=lo, close=c,
                          volume=max(v, 0.0))

    candles = [seen[k] for k in sorted(seen)]
    interval = _infer_interval(candles)

    if candles and rejected:
        bad = sum(rejected.values())
        if bad > (bad + len(candles)) * MAX_BAD_FRACTION:
            warnings.append(
                f"More than half the rows were unusable — check this is an "
                f"OHLCV export and not a trade list or a report.")

    if len(candles) < 5:
        warnings.append("Fewer than 5 usable bars; nothing can be replayed "
                        "from this.")
    elif len(candles) < 200:
        warnings.append(
            f"{len(candles)} bars is a small sample — a backtest over it can "
            f"report a hit rate but cannot distinguish it from luck.")

    return Ingested(candles=candles, interval_s=interval, source=source,
                    rows_read=len(rows), rejected=rejected, warnings=warnings)


def _infer_interval(candles: Sequence[Candle]) -> float:
    """Median gap between consecutive bars.

    Median, so that a weekend, a halt or a missing chunk does not become the
    timeframe.
    """
    if len(candles) < 2:
        return 0.0
    deltas = sorted(b.ts - a.ts for a, b in zip(candles, candles[1:])
                    if b.ts > a.ts)
    if not deltas:
        return 0.0
    n = len(deltas)
    return deltas[n // 2] if n % 2 else (deltas[n // 2 - 1] + deltas[n // 2]) / 2.0


def merge(existing: Sequence[Candle], new: Sequence[Candle]) -> list[Candle]:
    """Combine two series, newest version of each timestamp winning.

    Uploading an overlapping export should extend the history, not duplicate
    the overlap into the replay twice.
    """
    by_ts: dict[float, Candle] = {c.ts: c for c in existing}
    by_ts.update({c.ts: c for c in new})
    return [by_ts[k] for k in sorted(by_ts)]
