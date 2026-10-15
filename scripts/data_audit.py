#!/usr/bin/env python3
"""What historical ES data is actually here, and what can be tested with it.

    python scripts/data_audit.py data/
    python scripts/data_audit.py data/ ~/market-data --target 4 --stop 4

PHASE 1. IT READS AND REPORTS. IT WRITES NOTHING AND CHANGES NOTHING.

WHY THIS EXISTS RATHER THAN A BACKTEST

A two-year backtest at a four-tick target is only answerable if the data can
say which of two levels four ticks apart was touched FIRST. On one-minute
OHLCV it usually cannot: ES routinely covers eight ticks inside a minute, and
a bar that contains both the target and the stop records neither order nor
sequence. Crediting the target there is the single easiest way to manufacture
a win rate, and a backtest that does it will look excellent and be worthless.

So before any of that, this measures the thing that decides whether the
question is answerable at all:

    THE BRACKET DIAGNOSTIC -- what share of bars have a range wide enough to
    contain the whole target-and-stop bracket. Those are the bars where the
    outcome is genuinely ambiguous on this data. If that share is large, the
    honest answer is that this dataset cannot test this target, and the
    report says so instead of producing numbers.

It also reports coverage, symbols, rolls, gaps and duplicates, because a
dataset with a four-month hole in it is not two years of anything.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER, AND NOTHING HERE IS A RESULT.
"""

from __future__ import annotations

import argparse
import math
import os
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TICK = 0.25
POINT = 4                       # ticks per ES index point

# File types worth looking inside.
DBN_SUFFIXES = (".dbn", ".dbn.zst")
TEXT_SUFFIXES = (".csv", ".tsv", ".txt")
OTHER_SUFFIXES = (".parquet", ".npy", ".db", ".sqlite", ".sqlite3", ".json")

MAX_PREVIEW = 3
SAMPLE_ROWS = 200_000           # cap on rows read for the bar diagnostics


@dataclass
class Found:
    path: Path
    size: int
    kind: str = "?"
    schema: str = ""
    start: str = ""
    end: str = ""
    symbols: list = field(default_factory=list)
    rows: int | None = None
    note: str = ""

    @property
    def mb(self) -> float:
        return self.size / 1e6


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:,.1f}{unit}"
        n /= 1024
    return f"{n:,.1f}PB"


# ------------------------------------------------------------- finding them

def scan(roots) -> list[Found]:
    out: list[Found] = []
    seen = set()
    for root in roots:
        p = Path(root).expanduser()
        if p.is_file():
            paths = [p]
        elif p.is_dir():
            paths = sorted(q for q in p.rglob("*") if q.is_file())
        else:
            print(f"  (no such path: {p})", file=sys.stderr)
            continue
        for q in paths:
            try:
                real = q.resolve()
            except OSError:
                continue
            if real in seen:
                continue
            name = q.name.lower()
            if not (name.endswith(DBN_SUFFIXES) or name.endswith(TEXT_SUFFIXES)
                    or name.endswith(OTHER_SUFFIXES)):
                continue
            seen.add(real)
            try:
                out.append(Found(path=q, size=q.stat().st_size))
            except OSError:
                continue
    return out


# ---------------------------------------------------------- looking inside

def describe_dbn(f: Found) -> None:
    """Metadata only. A multi-gigabyte file is not parsed to be inventoried."""
    f.kind = "databento DBN"
    try:
        import databento as db
    except ImportError:
        f.note = "databento not importable in this interpreter"
        return
    try:
        store = db.DBNStore.from_file(str(f.path))
    except Exception as exc:                      # noqa: BLE001
        f.note = f"could not open: {exc}"
        return
    m = getattr(store, "metadata", None)
    if m is None:
        f.note = "no metadata block"
        return
    f.schema = str(getattr(m, "schema", "") or "")
    f.start = _stamp(getattr(m, "start", None))
    f.end = _stamp(getattr(m, "end", None))
    syms = getattr(m, "symbols", None) or []
    f.symbols = [str(s) for s in syms][:12]
    n = getattr(store, "nbytes", None)
    if n:
        f.note = f"{human(n)} decoded"


def _stamp(v) -> str:
    if v is None:
        return ""
    try:
        n = int(v)
    except (TypeError, ValueError):
        return str(v)
    if n > 10**15:                                # nanoseconds
        n //= 1_000_000_000
    import datetime as _dt
    try:
        return _dt.datetime.fromtimestamp(n, _dt.timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC")
    except (OverflowError, OSError, ValueError):
        return str(v)


def describe_text(f: Found) -> None:
    f.kind = "delimited text"
    try:
        with open(f.path, "r", errors="replace") as fh:
            head = [next(fh, "") for _ in range(2)]
            f.schema = head[0].strip()[:160]
            if head[1]:
                f.note = "first row: " + head[1].strip()[:120]
    except OSError as exc:
        f.note = f"could not read: {exc}"


def describe_sqlite(f: Found) -> None:
    f.kind = "sqlite"
    try:
        con = sqlite3.connect(f"file:{f.path}?mode=ro", uri=True)
        tabs = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        f.schema = ", ".join(tabs[:10])
        con.close()
    except sqlite3.Error as exc:
        f.note = f"could not open: {exc}"


def describe(f: Found) -> None:
    name = f.path.name.lower()
    if name.endswith(DBN_SUFFIXES):
        describe_dbn(f)
    elif name.endswith(TEXT_SUFFIXES):
        describe_text(f)
    elif name.endswith((".db", ".sqlite", ".sqlite3")):
        describe_sqlite(f)
    elif name.endswith(".parquet"):
        f.kind = "parquet"
        try:
            import pyarrow.parquet as pq
            md = pq.ParquetFile(str(f.path))
            f.rows = md.metadata.num_rows
            f.schema = ", ".join(md.schema.names[:12])
        except Exception as exc:                  # noqa: BLE001
            f.note = f"pyarrow unavailable or failed: {exc}"
    else:
        f.kind = f.path.suffix.lstrip(".")


# --------------------------------------------------- the bracket diagnostic

@dataclass
class BarStats:
    bars: int = 0
    sessions: set = field(default_factory=set)
    gaps: list = field(default_factory=list)
    duplicates: int = 0
    out_of_order: int = 0
    bad_prices: int = 0
    zero_volume: int = 0
    ranges: list = field(default_factory=list)   # bar range in ticks
    symbols: Counter = field(default_factory=Counter)
    first_ts: float | None = None
    last_ts: float = 0.0


def bar_stats(records, expect_s: float, limit: int = SAMPLE_ROWS) -> BarStats:
    """Walk OHLCV records and count what a backtest would trip over."""
    from liqmap import dbn
    st = BarStats()
    prev = None
    for rec in records:
        if st.bars >= limit:
            break
        ts = dbn.seconds(dbn._get(rec, "ts_event"))
        o = dbn.price(dbn._get(rec, "open"))
        h = dbn.price(dbn._get(rec, "high"))
        low = dbn.price(dbn._get(rec, "low"))
        c = dbn.price(dbn._get(rec, "close"))
        v = float(dbn._get(rec, "volume", 0) or 0)
        iid = dbn._get(rec, "instrument_id")
        if iid is not None:
            st.symbols[iid] += 1

        st.bars += 1
        if st.first_ts is None:
            st.first_ts = ts
        st.last_ts = max(st.last_ts, ts)

        if min(o, h, low, c) <= 0 or h < low or h < max(o, c) \
                or low > min(o, c):
            st.bad_prices += 1
        else:
            st.ranges.append((h - low) / TICK)
        if v <= 0:
            st.zero_volume += 1

        st.sessions.add(_session_day(ts))
        if prev is not None:
            d = ts - prev
            if d == 0:
                st.duplicates += 1
            elif d < 0:
                st.out_of_order += 1
            elif d > expect_s * 1.5:
                st.gaps.append((d, prev, ts))
        prev = ts
    st.gaps.sort(reverse=True)
    del st.gaps[20:]
    return st


def _session_day(ts: float) -> str:
    """CME equity sessions run 17:00 to 16:00 Chicago, so a session is
    named for the day it CLOSES on -- Sunday evening belongs to Monday."""
    import datetime as _dt
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("America/Chicago")
    except Exception:                             # noqa: BLE001
        tz = _dt.timezone.utc
    d = _dt.datetime.fromtimestamp(ts, tz)
    if d.hour >= 17:
        d += _dt.timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def bracket_report(ranges: list, target: int, stops) -> None:
    """The question that decides whether the backtest is answerable.

    A bar whose range covers the whole bracket touched BOTH levels, and an
    OHLCV bar records no sequence, so which came first is unknowable. That
    is not a small correction: it is the outcome of the trade.
    """
    if not ranges:
        print("    (no usable bars)")
        return
    n = len(ranges)
    print(f"    {n:,} bars with a usable range; "
          f"median {_median(ranges):.1f} ticks, "
          f"p90 {_pct(ranges, 0.90):.1f}, p99 {_pct(ranges, 0.99):.1f}")
    for stop in stops:
        span = target + stop
        amb = sum(1 for r in ranges if r >= span)
        print(f"    target {target}t / stop {stop}t  -> bracket {span}t: "
              f"{amb:,} of {n:,} bars ({amb / n:.1%}) span it and are "
              f"UNRESOLVABLE on this data")


def _median(xs):
    return _pct(xs, 0.5)


def _pct(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[k]


def breakeven(target: int, stop: int, cost_ticks: float) -> float:
    return (stop + cost_ticks) / (target + stop)


# ------------------------------------------------------------------- report

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+",
                    help="files or directories to inspect (read-only)")
    ap.add_argument("--target", type=int, default=4,
                    help="profit target in ticks (default 4 = 1 point)")
    ap.add_argument("--stops", default="4,6,8")
    ap.add_argument("--costs", default="0.4,1.0,1.5",
                    help="round-trip cost assumptions in ticks")
    ap.add_argument("--bars-of", default="",
                    help="a DBN OHLCV file to run the bar diagnostics on")
    ap.add_argument("--expect-s", type=float, default=60.0,
                    help="seconds per bar in --bars-of")
    args = ap.parse_args(argv)
    stops = [int(s) for s in args.stops.split(",") if s.strip()]
    costs = [float(c) for c in args.costs.split(",") if c.strip()]

    print("=" * 78)
    print("PHASE 1 -- DATA AUDIT. Read-only. Nothing here is a result.")
    print("=" * 78)
    print()

    found = scan(args.roots)
    if not found:
        print("No data files found under:", ", ".join(args.roots))
        print("Nothing can be tested until there is something to test on.")
        return 0

    for f in found:
        describe(f)

    total = sum(f.size for f in found)
    print(f"{len(found)} file(s), {human(total)} on disk")
    print()
    by_kind: dict[str, list] = {}
    for f in found:
        by_kind.setdefault(f.kind, []).append(f)
    for kind, fs in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
        print(f"  {kind}: {len(fs)} file(s), {human(sum(x.size for x in fs))}")
    print()

    print("-" * 78)
    print("FILES")
    print("-" * 78)
    for f in sorted(found, key=lambda x: -x.size):
        print(f"  {f.path}")
        print(f"      {human(f.size):>10}  {f.kind}"
              + (f"  schema={f.schema}" if f.schema else ""))
        if f.start or f.end:
            print(f"      covers {f.start}  ->  {f.end}")
        if f.symbols:
            print(f"      symbols: {', '.join(f.symbols)}")
        if f.rows is not None:
            print(f"      rows: {f.rows:,}")
        if f.note:
            print(f"      {f.note}")
    print()

    print("-" * 78)
    print("WHAT THE TARGET COSTS, BEFORE ANY BACKTEST")
    print("-" * 78)
    print(f"  target {args.target} ticks "
          f"({args.target / POINT:.2f} index points, "
          f"${args.target * 12.5:.2f} per contract)")
    print(f"  {'stop':>6}  " + "  ".join(f"cost {c}t".rjust(11)
                                         for c in costs))
    for stop in stops:
        row = [f"  {stop:>4}t  "]
        for c in costs:
            row.append(f"{breakeven(args.target, stop, c):>10.1%} ")
        print("".join(row))
    print()
    print("  These are the hit rates each geometry must CLEAR to break even.")
    print("  A target smaller than its stop needs a high rate by arithmetic,")
    print("  before any question about whether the signal works.")
    print()

    if args.bars_of:
        print("-" * 78)
        print(f"BAR DIAGNOSTICS -- {args.bars_of}")
        print("-" * 78)
        try:
            from liqmap import dbn
            store = dbn.open_file(args.bars_of)
        except Exception as exc:                  # noqa: BLE001
            print(f"  could not open: {exc}")
            return 0
        st = bar_stats(store, args.expect_s)
        print(f"    {st.bars:,} bars read (cap {SAMPLE_ROWS:,})")
        print(f"    {len(st.sessions)} distinct sessions")
        if st.first_ts is not None:
            print(f"    {_stamp(st.first_ts)}  ->  {_stamp(st.last_ts)}")
        print(f"    instruments: {len(st.symbols)}"
              + (f"  {dict(st.symbols.most_common(5))}" if st.symbols else ""))
        print(f"    duplicate stamps {st.duplicates:,}, "
              f"out of order {st.out_of_order:,}")
        print(f"    bad OHLC rows {st.bad_prices:,}, "
              f"zero volume {st.zero_volume:,}")
        print(f"    gaps over {args.expect_s * 1.5:.0f}s: {len(st.gaps)}"
              + (" (largest shown)" if st.gaps else ""))
        for d, a, b in st.gaps[:8]:
            print(f"        {d / 60:8.1f} min at {_stamp(a)}")
        print()
        print("    THE BRACKET DIAGNOSTIC")
        bracket_report(st.ranges, args.target, stops)
        print()
        print("    A high unresolvable share means this dataset cannot")
        print("    settle this target honestly. Finer bars or the tape")
        print("    would be needed, and no amount of care in the backtest")
        print("    recovers a sequence the data never recorded.")
    else:
        print("  (pass --bars-of <file> to run the bar diagnostics)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
