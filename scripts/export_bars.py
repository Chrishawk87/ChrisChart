#!/usr/bin/env python3
"""Every ES bar Databento gave us, as 1m / 5m / 15m spreadsheets.

    python scripts/export_bars.py data/*.dbn.zst --out ~/Downloads/ES_bars

Writes three files per format:

    ES_1m.csv   ES_5m.csv   ES_15m.csv
    ES_1m.xlsx  ES_5m.xlsx  ES_15m.xlsx     (when openpyxl is installed)

COLUMNS

    date       YYYY-MM-DD, exchange time (America/Chicago)
    time       HH:MM:SS, exchange time, the moment the bar OPENED
    price      the bar's CLOSE -- the single price asked for
    open high low volume
    contract   Databento instrument id; a change of id is a contract roll
    utc        ISO 8601, unambiguous and sortable

`price` is the close rather than the open, the midpoint or an average,
because the close is the only one of a bar's four prices that is a
decision point: it is what was true when the bar finished.

WHY OPEN, HIGH AND LOW ARE THERE TOO

A bar has four prices and three of them were asked to be dropped. Keeping
only the close would make these sheets unable to answer where price
actually went inside the bar -- which is the question every study in this
project has turned on. They cost four columns and cannot be recovered
later from a file that left them out.

TWO SOURCES, NOT ONE

The `ohlcv-1s` file is the authority wherever it reaches. The MBP-10 files
cover days it does not, and bars are built from their TRADE prints to fill
those in; where both cover the same second the OHLCV bar wins, because it
is the venue's own aggregation rather than one assembled here. Every added
day is named in the summary.

A BAR NEVER SPANS A CONTRACT ROLL. Across one the price steps by carry,
and a bar straddling it would show a range nobody traded.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER. It writes rows.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap import dbn                                      # noqa: E402

FRAMES = (("1m", 60), ("5m", 300), ("15m", 900))
EXCHANGE_TZ = "America/Chicago"

HEADERS = ("date", "time", "price", "open", "high", "low", "volume",
           "contract", "utc")

HEARTBEAT = 2_000_000
XLSX_ROW_LIMIT = 1_048_576          # Excel's hard ceiling, header included


class Progress:
    """Heartbeat on stderr. Twenty-three million bars take minutes."""

    def __init__(self, label: str, every: int = HEARTBEAT, out=None) -> None:
        self.label, self.every, self.n = label, max(1, every), 0
        self.extra = ""
        self.out = out if out is not None else sys.stderr
        self.start = time.time()

    def tick(self) -> None:
        self.n += 1
        if self.n % self.every:
            return
        el = max(1e-9, time.time() - self.start)
        print(f"  [{self.label}] {self.n:,}  {el:,.0f}s  "
              f"{self.n / el:,.0f}/s  {self.extra}", file=self.out,
              flush=True)

    def done(self) -> None:
        el = max(1e-9, time.time() - self.start)
        print(f"  [{self.label}] done: {self.n:,} in {el:,.0f}s",
              file=self.out, flush=True)


def _tz():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(EXCHANGE_TZ)
    except Exception:                                   # noqa: BLE001
        return dt.timezone.utc


TZ = _tz()


def stamp(ts: float):
    """(date, time, utc) in exchange time, from an epoch second."""
    d = dt.datetime.fromtimestamp(ts, TZ)
    u = dt.datetime.fromtimestamp(ts, dt.timezone.utc)
    return (d.strftime("%Y-%m-%d"), d.strftime("%H:%M:%S"),
            u.strftime("%Y-%m-%dT%H:%M:%SZ"))


# ------------------------------------------------------------- the bars

@dataclass
class Live:
    """One timeframe's bar in progress."""

    span: int
    slot: float | None = None
    contract: object = None
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0

    def row(self) -> tuple:
        d, t, u = stamp(self.slot)
        return (d, t, round(self.close, 4), round(self.open, 4),
                round(self.high, 4), round(self.low, 4),
                int(self.volume), self.contract, u)


class Builder:
    """Feeds 1-second bars (or prints) in, yields completed bars out.

    A new slot OR a change of contract closes the bar in progress. The
    second of those is the one that matters: a bar built across a roll
    would carry a high and a low from two different instruments.
    """

    def __init__(self, span: int) -> None:
        self.span = span
        self.cur: Live | None = None
        self.rows: list = []

    def feed(self, ts: float, o: float, h: float, lo: float, c: float,
             v: float, contract) -> None:
        s = math.floor(ts / self.span) * self.span
        cur = self.cur
        if cur is None or s != cur.slot or contract != cur.contract:
            if cur is not None:
                self.rows.append(cur.row())
            self.cur = Live(span=self.span, slot=s, contract=contract,
                            open=o, high=h, low=lo, close=c, volume=v)
            return
        cur.high = max(cur.high, h)
        cur.low = min(cur.low, lo)
        cur.close = c
        cur.volume += v

    def finish(self) -> None:
        if self.cur is not None:
            self.rows.append(self.cur.row())
            self.cur = None


@dataclass
class Source:
    path: str
    schema: str = ""
    kept: int = 0
    skipped: int = 0
    first: float | None = None
    last: float = 0.0
    added_days: set = field(default_factory=set)


def read_ohlcv(path: str, builders, src: Source, progress=None,
               limit: int = 0):
    """The venue's own 1-second bars. The authority wherever it reaches."""
    import itertools
    store = dbn.open_file(path)
    if limit:
        store = itertools.islice(store, limit)
    for rec in store:
        if progress is not None:
            progress.tick()
        ts = dbn.seconds(dbn._get(rec, "ts_event"))
        o = dbn.price(dbn._get(rec, "open"))
        h = dbn.price(dbn._get(rec, "high"))
        lo = dbn.price(dbn._get(rec, "low"))
        c = dbn.price(dbn._get(rec, "close"))
        v = float(dbn._get(rec, "volume", 0) or 0)
        iid = dbn._get(rec, "instrument_id")
        if ts <= 0 or min(o, h, lo, c) <= 0:
            src.skipped += 1
            continue
        # A high below its own low is covered by this: `lo <= o <= h`
        # cannot hold when h < lo, so testing for it separately was dead
        # code that looked like a second safeguard.
        if not (lo <= o <= h and lo <= c <= h):
            src.skipped += 1
            continue
        src.kept += 1
        if src.first is None:
            src.first = ts
        src.last = max(src.last, ts)
        for b in builders:
            b.feed(ts, o, h, lo, c, v, iid)


def read_trades(path: str, builders, src: Source, covered,
                progress=None, limit: int = 0):
    """Bars built from MBP-10 TRADE prints, for days OHLCV does not reach.

    A print is a single price, so each becomes a one-tick bar and the
    aggregation does the rest. Seconds already covered by the authority
    are skipped rather than merged -- two aggregations of the same second
    disagree in the fourth decimal and the sheet would carry both.
    """
    import itertools
    store = dbn.open_file(path)
    if limit:
        store = itertools.islice(store, limit)
    lo_c, hi_c = covered
    for rec in store:
        if progress is not None:
            progress.tick()
        t = dbn.trade_from(rec)
        if t is None:
            continue
        ts = t.ts
        if ts <= 0 or t.px <= 0:
            src.skipped += 1
            continue
        if lo_c is not None and lo_c <= ts <= hi_c:
            src.skipped += 1
            continue
        src.kept += 1
        if src.first is None:
            src.first = ts
        src.last = max(src.last, ts)
        src.added_days.add(stamp(ts)[0])
        iid = dbn._get(rec, "instrument_id")
        for b in builders:
            b.feed(ts, t.px, t.px, t.px, t.px, t.sz, iid)


# ------------------------------------------------------------- writing

def write_csv(path: Path, rows) -> int:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADERS)
        w.writerows(rows)
    return len(rows)


def write_xlsx(path: Path, rows, title: str) -> str:
    """One sheet, frozen header, in write-only mode.

    Three hundred thousand rows through the normal object model builds a
    cell object per value and runs into gigabytes; write-only streams
    them. No formulas are written, so there is nothing to recalculate.
    """
    # The row limit is checked BEFORE the import because it is the
    # reason that cannot be fixed. "Install openpyxl" is actionable;
    # "this will not fit in a worksheet" is not, and reporting the
    # install first would send someone to fix the wrong thing.
    if len(rows) + 1 > XLSX_ROW_LIMIT:
        return (f"{len(rows):,} rows exceeds Excel's "
                f"{XLSX_ROW_LIMIT:,} limit -- CSV only")
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
    except ImportError:
        return "openpyxl not installed -- CSV only"
    wb = Workbook(write_only=True)
    ws = wb.create_sheet(title=title)
    ws.freeze_panes = "A2"
    bold = Font(name="Arial", bold=True)
    head = []
    for h in HEADERS:
        from openpyxl.cell import WriteOnlyCell
        c = WriteOnlyCell(ws, value=h)
        c.font = bold
        head.append(c)
    ws.append(head)
    for r in rows:
        ws.append(list(r))
    wb.save(path)
    return f"{len(rows):,} rows"


# ------------------------------------------------------------- the run

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", help=".dbn / .dbn.zst files")
    ap.add_argument("--out", default="ES_bars",
                    help="directory to write into (created if absent)")
    ap.add_argument("--no-xlsx", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N records per file -- a trial run")
    args = ap.parse_args(argv)

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    ohlcv, depth = [], []
    for p in args.paths:
        name = Path(p).name.lower()
        (ohlcv if "ohlcv" in name else depth).append(p)

    print(f"out       {out}")
    print(f"timezone  {EXCHANGE_TZ} for date and time; utc column too")
    print(f"price     the bar's CLOSE")
    print(f"ohlcv     {len(ohlcv)} file(s) -- the authority")
    print(f"depth     {len(depth)} file(s) -- trades, for days OHLCV "
          f"does not reach")
    if args.limit:
        print(f"LIMIT     {args.limit:,} records per file -- a trial run, "
              f"not the full export.")
    print("progress on stderr")
    print()

    builders = [Builder(span) for _, span in FRAMES]
    sources: list[Source] = []

    for p in ohlcv:
        s = Source(path=p, schema="ohlcv-1s")
        pr = Progress(f"ohlcv {Path(p).name[:24]}")
        read_ohlcv(p, builders, s, progress=pr, limit=args.limit)
        pr.done()
        sources.append(s)

    covered = (min((s.first for s in sources if s.first is not None),
                   default=None),
               max((s.last for s in sources), default=0.0))

    for p in depth:
        s = Source(path=p, schema="mbp-10")
        pr = Progress(f"depth {Path(p).name[:24]}")
        read_trades(p, builders, s, covered, progress=pr, limit=args.limit)
        pr.done()
        sources.append(s)

    for b in builders:
        b.finish()

    print("SOURCES")
    for s in sources:
        span = ""
        if s.first is not None:
            span = f"  {stamp(s.first)[0]} -> {stamp(s.last)[0]}"
        extra = ""
        if s.added_days:
            extra = (f"  adds {len(s.added_days)} day(s): "
                     + ", ".join(sorted(s.added_days)[:6]))
        elif s.schema == "mbp-10":
            extra = "  nothing new -- already covered by the OHLCV file"
        print(f"  {Path(s.path).name}")
        print(f"      {s.schema}  kept {s.kept:,}  skipped {s.skipped:,}"
              f"{span}{extra}")
    print()

    print("FILES")
    for (name, _), b in zip(FRAMES, builders):
        rows = sorted(b.rows, key=lambda r: r[8])      # by UTC
        csv_path = out / f"ES_{name}.csv"
        n = write_csv(csv_path, rows)
        print(f"  {csv_path}   {n:,} rows")
        if not args.no_xlsx:
            x_path = out / f"ES_{name}.xlsx"
            note = write_xlsx(x_path, rows, f"ES {name}")
            if note.endswith("CSV only"):
                print(f"      xlsx skipped: {note}")
            else:
                print(f"  {x_path}   {note}")
    print()
    print("`price` is the bar's close. `contract` changing is a roll.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
