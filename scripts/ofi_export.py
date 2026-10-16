#!/usr/bin/env python3
"""MBP-10 to a per-event OFI table.

    python scripts/ofi_export.py data/*.dbn.zst --out ~/Downloads/ES_ofi

Writes one row per measured book event:

    ES_ofi.parquet      every event, every feature  (pyarrow)
    ES_ofi.csv          the same, when pyarrow is absent

COLUMNS

    ts utc instrument sequence action
    e_n                 the CKS contribution at the touch (integer)
    e_multi             the same weighted across all ten levels
    bid_px bid_sz ask_px ask_sz        the touch, in index points
    spread_ticks relative_spread
    bid_depth ask_depth book_slope
    obi obi_1 obi_2                    book imbalance by level
    msg_rate cancel_to_add

NO THRESHOLD, NO SIGNAL, NO POSITION SIZE. This writes the
measurement. Rolling windows, z-scores, thresholds and any execution
rule are the caller's and belong downstream of this file, which is why
none of them appear in it.

WHAT IS NOT IN THE TABLE, AND WHY

Events the state machine refused are absent rather than present with
a zero: the first event of a stream, boundary records, contract rolls,
sequence gaps, and books that were empty or crossed. A refused event
has no measurable `e_n`, and a zero would sum into a window as though
it did. The run prints the tally -- if refusals are a large share of
the file, the table is not what it looks like.

Prices are converted to index points ONLY on the way out. The state
machine works in raw integers throughout; see `liqmap/ofi.py`.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap import dbn, ofi                                 # noqa: E402

COLS = (
    "ts", "utc", "instrument", "sequence", "action",
    "e_n", "e_multi",
    "bid_px", "bid_sz", "ask_px", "ask_sz",
    "spread_ticks", "relative_spread",
    "bid_depth", "ask_depth", "book_slope",
    "obi", "obi_1", "obi_2",
    "msg_rate", "cancel_to_add",
)

# One ES tick in the feed's raw units: 0.25 index points, and Databento
# carries nine decimal places, so 0.25 * 1e9.
TICK_RAW = 250_000_000
HEARTBEAT = 500_000


class Progress:
    def __init__(self, label, every=HEARTBEAT, out=None):
        self.label, self.every, self.n = label, every, 0
        self.extra = ""
        self.out = out if out is not None else sys.stderr
        self.start = time.time()

    def tick(self):
        self.n += 1
        if self.n % self.every:
            return
        el = max(1e-9, time.time() - self.start)
        print(f"  [{self.label}] {self.n:,}  {el:,.0f}s  "
              f"{self.n / el:,.0f}/s  {self.extra}",
              file=self.out, flush=True)

    def done(self):
        el = max(1e-9, time.time() - self.start)
        print(f"  [{self.label}] done: {self.n:,} in {el:,.0f}s",
              file=self.out, flush=True)


def row_from(ev: ofi.Event, feats: ofi.Features) -> dict:
    t = ev.touch
    return {
        "ts": round(ev.ts, 9),
        "utc": dt.datetime.fromtimestamp(
            ev.ts, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "instrument": ev.instrument,
        "sequence": ev.sequence,
        "action": ev.action,
        "e_n": ev.e_n,
        "e_multi": round(ev.e_multi, 4),
        "bid_px": dbn.price(t.bid_px),
        "bid_sz": t.bid_sz,
        "ask_px": dbn.price(t.ask_px),
        "ask_sz": t.ask_sz,
        "spread_ticks": round(t.spread / TICK_RAW, 4),
        "relative_spread": round(ofi.relative_spread(t), 10),
        "bid_depth": ev.bid_depth,
        "ask_depth": ev.ask_depth,
        "book_slope": round(
            ofi.book_slope(ev.bid_depth, ev.ask_depth,
                           max(1, t.spread // TICK_RAW)), 4),
        "obi": round(ev.obi, 6),
        "obi_1": round(ev.obi_at(1), 6),
        "obi_2": round(ev.obi_at(2), 6),
        "msg_rate": feats.msg_rate,
        "cancel_to_add": ("" if feats.cancel_to_add is None
                          else round(feats.cancel_to_add, 4)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", help=".dbn / .dbn.zst files")
    ap.add_argument("--out", default="ES_ofi",
                    help="path without extension")
    ap.add_argument("--depth", type=int, default=ofi.DEPTH)
    ap.add_argument("--floor", type=float, default=0.1,
                    help="weight of the deepest level for e_multi")
    ap.add_argument("--window", type=float, default=1.0,
                    help="seconds for msg_rate and cancel_to_add")
    ap.add_argument("--no-sequence-check", action="store_true",
                    help="measure across sequence gaps -- off by "
                         "default, because a gap means the previous "
                         "state is not what this record followed")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N records per file")
    args = ap.parse_args(argv)

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"out        {out}.parquet / .csv")
    print(f"depth      {args.depth} levels, deepest weighted "
          f"{args.floor}")
    print(f"window     {args.window}s for msg_rate and cancel_to_add")
    seq_note = "NOT checked" if args.no_sequence_check else "checked"
    print(f"sequence   {seq_note}")
    print("no threshold, no signal, no position size -- "
          "this writes the measurement")
    print("progress on stderr")
    print()

    state = ofi.BookState(depth=args.depth, floor=args.floor,
                          check_sequence=not args.no_sequence_check)
    feats = ofi.Features(window_s=args.window)

    # Rows stream to disk as they are produced. Holding them in a list
    # costs ~840 bytes each, which at tens of millions of MBP-10
    # events is tens of gigabytes -- the machine starts swapping and
    # the run appears to hang rather than fail.
    csv_path = out.with_suffix(".csv")
    written = 0
    fh = open(csv_path, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
    writer.writeheader()

    for path in args.paths:
        import itertools
        store = dbn.open_file(path)
        if args.limit:
            store = itertools.islice(store, args.limit)
        pr = Progress(f"mbp-10 {Path(path).name[:22]}")
        for rec in store:
            pr.tick()
            # Every record goes to the state machine, including the
            # ones with no levels. Filtering them here would skip the
            # boundary branch and leave its counter permanently at
            # zero -- which reads as "there were none" rather than
            # "these were counted somewhere else".
            ev = state.feed(rec)
            if ev is None:
                continue
            feats.observe(ev.ts, ev.action)
            writer.writerow(row_from(ev, feats))
            written += 1
            pr.extra = f"{written:,} events"
        pr.done()
    fh.close()

    if not written:
        print("No measurable events. Check the refusal tally below.",
              file=sys.stderr)

    note = ""
    try:
        import pyarrow.csv as pv
        import pyarrow.parquet as pq
        pq.write_table(pv.read_csv(csv_path), out.with_suffix(".parquet"),
                       compression="snappy")
        note = (f"{out.with_suffix('.parquet')}   "
                f"{out.with_suffix('.parquet').stat().st_size / 1e6:,.1f} MB")
    except ImportError:
        note = "pyarrow not installed -- CSV only"

    print("REFUSED (absent from the table, never written as zero)")
    r = state.refused
    for k, v in r.as_dict().items():
        share = 100.0 * v / state.seen if state.seen else 0.0
        print(f"  {k:<10} {v:>10,}  ({share:5.2f}% of records seen)")
    print(f"  {'total':<10} {r.total:>10,}")
    print()
    print(f"{csv_path}   {written:,} rows   "
          f"{csv_path.stat().st_size / 1e6:,.1f} MB")
    if note:
        print(f"{note}")
    print()
    print(f"{state.seen:,} records seen, {written:,} measured.")
    if state.seen and r.total / state.seen > 0.5:
        print("WARNING: more than half of all records were refused. "
              "The table is not")
        print("a sample of the book -- read the tally above before "
              "using it.")
    print()
    print("e_n is the touch contribution; e_multi weights all "
          f"{args.depth} levels.")
    print("Positive is net buy pressure at the touch. Windowing, "
          "z-scoring and any")
    print("threshold are yours to apply downstream.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
