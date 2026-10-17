#!/usr/bin/env python3
"""A month of MBP-10 down to one row a second: candles AND order flow.

    python scripts/ofi_replay.py data/*.dbn.zst --out ~/Downloads/ES_sec

WHY A SECOND, AND WHY BOTH IN ONE FILE

`ofi_export.py` writes one row per book event. That is the right
output for studying the engine and the wrong one for a backtest: a
month of ES MBP-10 is tens of millions of events, which is gigabytes
no web service is going to hold, and the rows are not on a grid so
nothing can be joined to them.

A second is the smallest bucket that is still small on disk. One
trading month is roughly 1.8 million rows -- about 110MB as CSV, 20MB
as parquet -- which uploads, and which a backtest can walk in seconds.

The two halves have to come out of the SAME pass. Candles answer
"where did price go", OFI answers "which way was the book leaning
when it went there", and a backtest needs them on one clock. Built
from two files they can disagree about which second a print belongs
to, and every result after that is measuring the join rather than the
market.

WHAT A ROW HOLDS

    ts                    the second, unix
    open high low close   from the prints in that second
    volume trades         contracts and fills
    delta                 buy notional minus sell notional
    ofi                   summed e_n: the CKS order-flow imbalance
    events                book events the engine measured
    bid ask bid_sz ask_sz the touch as the second ended
    spread                ask minus bid, in ticks

A second with no prints still gets a row IF the book moved, because
"nothing traded while the book pulled" is a real observation. Seconds
where nothing at all happened are absent, not zero -- a gap in the
index is a quiet market, and writing zeros there would feed the
backtest a flat bar that never existed.

THE PRICES ARE DECODED BY DIVISION, never by multiplying by 1e-9.
That was a real bug here: 6,329 of 32,001 ES ticks decoded inexactly
with the multiply, and those same ticks failed to snap to their own
grid.

IT READS AND WRITES FILES. It places nothing, suggests nothing, and
decides nothing.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap import dbn, ofi                                 # noqa: E402

COLS = ["ts", "open", "high", "low", "close", "volume", "trades",
        "delta", "ofi", "events", "bid", "ask", "bid_sz", "ask_sz",
        "spread"]

TICK = 0.25             # ES. Only used to report the spread in ticks.


class Progress:
    """A heartbeat, because a month takes a while and silence looks
    exactly like a hang."""

    def __init__(self, label: str, every: int = 2_000_000) -> None:
        self.label, self.every = label, every
        self.n, self.t0, self.extra = 0, time.time(), ""

    def tick(self) -> None:
        self.n += 1
        if self.n % self.every == 0:
            el = time.time() - self.t0
            rate = self.n / el if el > 0 else 0
            print(f"  {self.label}: {self.n:,} records  "
                  f"{rate:,.0f}/s  {self.extra}", file=sys.stderr)

    def done(self) -> None:
        el = time.time() - self.t0
        print(f"  {self.label}: {self.n:,} records in {el:,.0f}s  "
              f"{self.extra}", file=sys.stderr)


class Second:
    """One second's worth, accumulated in place.

    Plain attributes rather than a dataclass: this is built and thrown
    away 1.8 million times a month, and the row it becomes is written
    immediately rather than held.
    """

    __slots__ = ("ts", "open", "high", "low", "close", "volume",
                 "trades", "buy_notional", "sell_notional", "ofi",
                 "events", "bid", "ask", "bid_sz", "ask_sz", "records",
                 "books")

    def __init__(self, ts: float) -> None:
        self.ts = ts
        self.records = 0
        self.books = 0
        self.open = self.high = self.low = self.close = 0.0
        self.volume = self.trades = 0.0
        self.buy_notional = self.sell_notional = 0.0
        self.ofi = 0
        self.events = 0
        self.bid = self.ask = self.bid_sz = self.ask_sz = 0.0

    def add_trade(self, t) -> None:
        if self.trades == 0:
            self.open = self.high = self.low = t.px
        else:
            if t.px > self.high:
                self.high = t.px
            if t.px < self.low:
                self.low = t.px
        self.close = t.px
        self.volume += t.sz
        self.trades += 1
        if t.aggressor == "buy":
            self.buy_notional += t.notional
        else:
            self.sell_notional += t.notional

    def row(self) -> dict:
        spread = ((self.ask - self.bid) / TICK
                  if self.bid > 0 and self.ask > 0 else 0.0)
        # NO PRINT MEANS NO PRICE. An empty field reads back as null;
        # a 0.00 would read back as a price, and a backtest cannot
        # tell a fabricated zero from a real quote. The book columns
        # are still filled, because the book did exist.
        def px(v):
            return f"{v:.2f}" if self.trades else ""

        # Same rule for the book: no quote means no number, not zero.
        def bk(v, fmt="{:.2f}"):
            return fmt.format(v) if self.books else ""
        return {
            "ts": f"{self.ts:.0f}",
            "open": px(self.open), "high": px(self.high),
            "low": px(self.low), "close": px(self.close),
            "volume": f"{self.volume:.0f}", "trades": f"{self.trades:.0f}",
            "delta": f"{self.buy_notional - self.sell_notional:.2f}",
            "ofi": self.ofi, "events": self.events,
            "bid": bk(self.bid), "ask": bk(self.ask),
            "bid_sz": bk(self.bid_sz, "{:.0f}"),
            "ask_sz": bk(self.ask_sz, "{:.0f}"),
            "spread": bk(spread),
        }

    @property
    def empty(self) -> bool:
        """Nothing in this second filled a single column.

        Counted by CONTENT, not by records. A record can arrive and
        contribute nothing at all -- an `ohlcv-1s` record has no book
        to difference and no aggressor to count, so it fills neither
        half. Writing a row for it produces a line of commas: no
        price, no book, no flow.

        That is not a harmless blank. Glob a wrong-schema file in by
        accident and you get one such row PER RECORD -- 22.9 million
        of them from a two-year OHLCV extract, dwarfing the real data
        and dragging the timestamps out of order.

        The first record of a file is still kept: it carries a book
        even though it can produce no event.
        """
        return self.trades == 0 and self.books == 0 and self.events == 0


def replay(paths, writer, limit: int = 0,
           state: ofi.BookState | None = None) -> tuple[int, int]:
    """Stream every record through the engine, emitting one row a second.

    Returns (records seen, rows written). The state machine is shared
    across files so a session boundary inside one day does not reset
    the book -- the refusal counters say what it did at each edge.
    """
    st = state if state is not None else ofi.BookState()
    cur: Second | None = None
    seen = rows = 0
    last_ts = 0.0
    notes: list = []

    for path in paths:
        # Counted from THIS file's records, not from rows written:
        # the first record of a new file flushes the previous file's
        # last second, which would otherwise be credited here and
        # hide a file that contributed nothing.
        f_books = f_trades = 0
        first_ts = 0.0
        import itertools
        store = dbn.open_file(str(path))
        if limit:
            store = itertools.islice(store, limit)
        pr = Progress(Path(path).name[:28])

        for rec in store:
            pr.tick()
            seen += 1

            ts = dbn.seconds(dbn._get(rec, "ts_event"))
            if not ts:
                continue
            bucket = math.floor(ts)

            if cur is not None and bucket != cur.ts:
                if not cur.empty:
                    writer.writerow(cur.row())
                    rows += 1
                cur = None
            if cur is None:
                cur = Second(bucket)
            cur.records += 1

            # The book half. Every record goes to the state machine,
            # including the ones with no levels: filtering them here
            # would skip the branch that counts them and leave its
            # tally reading "there were none" rather than "these were
            # counted elsewhere".
            if not first_ts:
                first_ts = ts
            ev = st.feed(rec)
            if ev is not None:
                cur.ofi += ev.e_n
                cur.events += 1

            # The touch comes from the RECORD, not from the event.
            # The first record of a file carries a real book but can
            # produce no event -- there is nothing behind it to
            # difference against -- and reading the touch off the
            # event stream would throw that book away.
            lv = dbn._get(rec, "levels")
            if lv:
                top = lv[0] if isinstance(lv, (list, tuple)) else lv
                # Decoded by DIVISION. The multiply-by-1e-9 form is
                # inexact for thousands of real ES ticks.
                cur.bid = dbn.price(dbn._get(top, "bid_px", 0))
                cur.ask = dbn.price(dbn._get(top, "ask_px", 0))
                cur.bid_sz = float(dbn._get(top, "bid_sz", 0) or 0)
                cur.ask_sz = float(dbn._get(top, "ask_sz", 0) or 0)
                cur.books += 1
                f_books += 1

            # The tape half, from the same record.
            tr = dbn.trade_from(rec)
            if tr is not None:
                cur.add_trade(tr)
                f_trades += 1

            pr.extra = f"{rows:,} seconds"
        pr.done()

        name = Path(path).name
        if f_books == 0 and f_trades == 0:
            notes.append(
                f"{name}: NO book and NO trade in any record -- this is "
                f"not an mbp-10/mbp-1 extract and contributed nothing")
        elif f_books == 0:
            notes.append(
                f"{name}: {f_trades:,} trades but NO book -- a `trades` "
                f"extract gives candles, never OFI")
        elif f_trades == 0:
            notes.append(
                f"{name}: {f_books:,} book updates but NO trade -- a "
                f"`bbo` extract gives OFI, never candles")
        if first_ts and last_ts and first_ts < last_ts:
            notes.append(
                f"{name}: starts BEFORE the previous file ended "
                f"({_when(first_ts)} < {_when(last_ts)}) -- the output "
                f"is not in time order")
        if cur is not None:
            last_ts = max(last_ts, float(cur.ts))

    if cur is not None and not cur.empty:
        writer.writerow(cur.row())
        rows += 1
    for n in notes:
        print(f"  WARNING  {n}", file=sys.stderr)
    return seen, rows


def _when(ts: float) -> str:
    import datetime as _dt
    return _dt.datetime.fromtimestamp(
        ts, _dt.timezone.utc).strftime("%Y-%m-%d")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", help=".dbn / .dbn.zst files")
    ap.add_argument("--out", required=True,
                    help="output path, without extension")
    ap.add_argument("--limit", type=int, default=0,
                    help="records per file, for a quick look")
    ap.add_argument("--no-parquet", action="store_true")
    ap.add_argument("--no-seq-check", action="store_true",
                    help="stop refusing records whose sequence is not "
                         "prev+1. On a CME file filtered to one symbol "
                         "those skips are OTHER instruments' messages "
                         "on the same channel, not missed ones, so the "
                         "check discards real book transitions")
    args = ap.parse_args(argv)

    paths = []
    for p in args.paths:
        q = Path(p)
        if not q.exists():
            print(f"no such file: {p}", file=sys.stderr)
            return 2
        paths.append(q)

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out.with_suffix(".csv")

    print(f"files   {len(paths)}", file=sys.stderr)
    print(f"out     {csv_path}", file=sys.stderr)
    print(file=sys.stderr)

    st = ofi.BookState(check_sequence=not args.no_seq_check)
    if args.no_seq_check:
        print("sequence check OFF -- every consecutive record is "
              "treated as\nthe next state of this symbol's book\n",
              file=sys.stderr)
    t0 = time.time()
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        writer.writeheader()
        seen, rows = replay(paths, writer, limit=args.limit, state=st)
    el = time.time() - t0

    size_mb = csv_path.stat().st_size / 1e6
    print(f"\n{seen:,} records -> {rows:,} seconds in {el:,.0f}s",
          file=sys.stderr)
    print(f"{csv_path}  ({size_mb:,.1f} MB)", file=sys.stderr)

    r = st.refused
    print(f"refused {r.total:,} {r.as_dict()}", file=sys.stderr)
    if rows == 0:
        print("\nNOTHING WAS WRITTEN. If `no_book` accounts for every "
              "refusal, the file\nhas no book in it -- a `trades` "
              "extract cannot drive the OFI engine.",
              file=sys.stderr)
        return 1

    if not args.no_parquet:
        try:
            import pyarrow.csv as pv
            import pyarrow.parquet as pq
            tbl = pv.read_csv(csv_path)
            pq_path = out.with_suffix(".parquet")
            pq.write_table(tbl, pq_path, compression="zstd")
            print(f"{pq_path}  "
                  f"({pq_path.stat().st_size / 1e6:,.1f} MB)",
                  file=sys.stderr)
        except ImportError:
            print("(pyarrow not installed -- CSV only)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
