#!/usr/bin/env python3
"""Replay MBP-10 and report who is winning the book race, moment by moment.

    python scripts/ofi_race.py data/*.dbn.zst --window 100 --baseline 1000 \\
        --z 0.5 --out ~/Downloads/ES_race

Every measured event gets a reading:

    window      the summed e_n over the last --window events
    z           that sum, z-scored against the last --baseline sums
    side        buyers / sellers / neither, at the --z line YOU set

`--z` has no default. The line between "winning" and "noise" is a
trading decision and this script will not pick one for you; it reports
which side of your line the measurement fell on.

WHAT IT WRITES

    ES_race.csv         one row per event, every column above
    ES_race.parquet     the same, when pyarrow is present

and a summary at the end: how long each side held the lead, how often
the lead changed, and the distribution of z.

WHAT IT IS NOT

This is not a backtest. There is no entry, no fill, no stop and no
PnL, because none of those follow from the measurement alone -- they
follow from an execution rule, which is yours. What this answers is
the question actually asked: at this instant, which side is winning,
and by how much relative to its own recent behaviour.

THE READING IS WITHHELD UNTIL IT MEANS SOMETHING. No z is reported
before the baseline is full, or while the baseline is flat. Both would
otherwise produce enormous values exactly when there is least
information -- at the start of every session and after every gap.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap import dbn, ofi                                 # noqa: E402

TICK_RAW = 250_000_000
HEARTBEAT = 500_000

COLS = ("ts", "utc", "instrument", "sequence", "action",
        "e_n", "e_multi", "window", "z", "side",
        "bid_px", "bid_sz", "ask_px", "ask_sz",
        "spread_ticks", "obi", "bid_depth", "ask_depth",
        "msg_rate", "cancel_to_add")


class Progress:
    def __init__(self, label, every=HEARTBEAT):
        self.label, self.every, self.n = label, every, 0
        self.extra = ""
        self.start = time.time()

    def tick(self):
        self.n += 1
        if self.n % self.every:
            return
        el = max(1e-9, time.time() - self.start)
        print(f"  [{self.label}] {self.n:,}  {el:,.0f}s  "
              f"{self.n / el:,.0f}/s  {self.extra}",
              file=sys.stderr, flush=True)

    def done(self):
        el = max(1e-9, time.time() - self.start)
        print(f"  [{self.label}] done: {self.n:,} in {el:,.0f}s",
              file=sys.stderr, flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="ES_race")
    ap.add_argument("--window", type=int, default=100,
                    help="events summed into the current OFI")
    ap.add_argument("--baseline", type=int, default=1000,
                    help="window sums the z-score is measured against")
    ap.add_argument("--z", type=float, required=True,
                    help="YOUR line between winning and noise; there "
                         "is deliberately no default")
    ap.add_argument("--no-sequence-check", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    if args.z <= 0:
        print("--z must be positive: it is the distance from the mean "
              "at which you call the race.", file=sys.stderr)
        return 2

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"out        {out}.csv")
    print(f"window     {args.window} events summed")
    print(f"baseline   {args.baseline} window-sums for the z-score")
    print(f"your line  |z| >= {args.z}")
    print("no entry, no fill, no PnL -- this reports the race")
    print("progress on stderr")
    print()

    state = ofi.BookState(check_sequence=not args.no_sequence_check)
    race = ofi.Race(window=args.window, baseline=args.baseline)
    feats = ofi.Features()

    csv_path = out.with_suffix(".csv")
    fh = open(csv_path, "w", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
    w.writeheader()

    sides = Counter()
    held = Counter()           # seconds each side held the lead
    flips = 0
    last_side, last_ts = None, None
    zs: list[float] = []
    written = 0

    for path in args.paths:
        import itertools
        store = dbn.open_file(path)
        if args.limit:
            store = itertools.islice(store, args.limit)
        pr = Progress(f"mbp-10 {Path(path).name[:22]}")
        for rec in store:
            pr.tick()
            ev = state.feed(rec)
            if ev is None:
                continue
            feats.observe(ev.ts, ev.action)
            rd = race.push(ev.e_n, ts=ev.ts)
            side = rd.side(args.z)

            sides[side] += 1
            if rd.z is not None:
                zs.append(rd.z)
            if last_side is not None and last_ts is not None:
                held[last_side] += max(0.0, ev.ts - last_ts)
                if side != last_side and side != "neither":
                    flips += 1
            last_side, last_ts = side, ev.ts

            t = ev.touch
            w.writerow({
                "ts": round(ev.ts, 9),
                "utc": dt.datetime.fromtimestamp(
                    ev.ts, dt.timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%S.%fZ"),
                "instrument": ev.instrument,
                "sequence": ev.sequence,
                "action": ev.action,
                "e_n": ev.e_n,
                "e_multi": round(ev.e_multi, 4),
                "window": rd.window,
                "z": "" if rd.z is None else round(rd.z, 6),
                "side": side,
                "bid_px": dbn.price(t.bid_px),
                "bid_sz": t.bid_sz,
                "ask_px": dbn.price(t.ask_px),
                "ask_sz": t.ask_sz,
                "spread_ticks": round(t.spread / TICK_RAW, 4),
                "obi": round(ev.obi, 6),
                "bid_depth": ev.bid_depth,
                "ask_depth": ev.ask_depth,
                "msg_rate": feats.msg_rate,
                "cancel_to_add": ("" if feats.cancel_to_add is None
                                  else round(feats.cancel_to_add, 4)),
            })
            written += 1
            pr.extra = f"{written:,} readings"
        pr.done()
    fh.close()

    note = ""
    try:
        import pyarrow.csv as pv
        import pyarrow.parquet as pq
        pq.write_table(pv.read_csv(csv_path), out.with_suffix(".parquet"),
                       compression="snappy")
        note = f"{out.with_suffix('.parquet')}"
    except ImportError:
        note = "pyarrow not installed -- CSV only"

    print("REFUSED (no reading taken)")
    for k, v in state.refused.as_dict().items():
        share = 100.0 * v / state.seen if state.seen else 0.0
        print(f"  {k:<10} {v:>10,}  ({share:5.2f}%)")
    print()

    print(f"WHO HELD THE LEAD  (|z| >= {args.z})")
    tot = sum(sides.values()) or 1
    tot_s = sum(held.values()) or 1.0
    for name in ("buyers", "sellers", "neither"):
        print(f"  {name:<9} {sides[name]:>10,} readings "
              f"({100.0 * sides[name] / tot:5.1f}%)   "
              f"{held[name] / 60.0:9,.1f} min "
              f"({100.0 * held[name] / tot_s:5.1f}% of the time)")
    print(f"  lead changed {flips:,} times")
    print()

    if zs:
        zs.sort()
        def q(p):
            return zs[min(len(zs) - 1, int(p * len(zs)))]
        print("Z DISTRIBUTION")
        print(f"  p01 {q(.01):+6.2f}   p25 {q(.25):+6.2f}   "
              f"p50 {q(.50):+6.2f}   p75 {q(.75):+6.2f}   "
              f"p99 {q(.99):+6.2f}")
        print(f"  |z| >= {args.z} on "
              f"{100.0 * sum(1 for z in zs if abs(z) >= args.z) / len(zs):.1f}"
              f"% of readings with a z")
        print()

    print(f"{csv_path}   {written:,} rows   "
          f"{csv_path.stat().st_size / 1e6:,.1f} MB")
    if note:
        print(f"{note}")
    print()
    print("`side` is the measurement against YOUR line. Changing --z "
          "changes nothing about")
    print("the book -- it changes where you drew it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
