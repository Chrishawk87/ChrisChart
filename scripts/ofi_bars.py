#!/usr/bin/env python3
"""OFI-closed candles from real ES, and the threshold that makes them readable.

    # find a threshold first -- 250 is a number from a prompt, not from your data
    python scripts/ofi_bars.py data/ES_c_0_mbp-10_*.dbn.zst --sweep

    # then emit the chart rows
    python scripts/ofi_bars.py data/ES_c_0_mbp-10_*.dbn.zst \\
        --threshold 4000 --csv ofi_bars.csv

WHY THE SWEEP COMES FIRST.

The threshold is the only parameter that matters and it is not
transferable. It is denominated in contracts of net imbalance, so it
depends on how much size your book carries and how often it updates --
quantities that differ between instruments, between sessions, and
between a quiet August and an FOMC afternoon. Picked too low, every
other book update closes a bar: on a synthetic feed, 250 produced
29,247 bars in ten minutes with a median duration of 10ms and no range
at all, which is not a chart. Picked too high, a session is four bars.

So the sweep reports, for each candidate: bars per hour, the green/red
balance, median duration, and median range in ticks. You pick the row
that looks like a chart you would read.

GREEN/RED BALANCE IS A DIAGNOSTIC, NOT A RESULT. It should sit near
even. A long way off means the accumulator is latching or the sign is
inverted, both of which make every bar wrong in the same direction.

The CSV is [timestamp, open, high, low, close, direction, ofi_strength]
with the timestamp at each bar's CLOSE, because that is the instant the
bar became a fact.

Nothing here places an order or claims a direction will continue.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from liqmap import dbn                                  # noqa: E402
from liqmap.ofi import Touch                            # noqa: E402
from liqmap.ofibar import (CHOP, GREEN, RED, ROW_COLUMNS,  # noqa: E402
                           OfiBarBuilder)

BUILD = "2026-10-26a"
SWEEP = (250, 500, 1000, 2000, 4000, 8000, 16000, 32000)


def median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[len(s) // 2]


def stream(paths: list[str]):
    """(ts, Touch) for every usable MBP-10 record, oldest first."""
    for path in paths:
        print(f"reading {path} ...", flush=True)
        for rec in dbn.open_file(path):
            ts = dbn.seconds(dbn._get(rec, "ts_event")) or 0.0
            if ts <= 0.0:
                continue
            lv = dbn._get(rec, "levels")
            if not lv:
                continue
            l0 = lv[0]
            # Raw integer units, which is what Touch and level_ofi want.
            # Converting to float prices here and back would lose the
            # exactness the integer formula depends on.
            yield ts, Touch(bid_px=int(dbn._get(l0, "bid_px", 0) or 0),
                            bid_sz=int(dbn._get(l0, "bid_sz", 0) or 0),
                            ask_px=int(dbn._get(l0, "ask_px", 0) or 0),
                            ask_sz=int(dbn._get(l0, "ask_sz", 0) or 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--threshold", type=int, default=0,
                    help="net imbalance in contracts that closes a bar")
    ap.add_argument("--max-s", type=float, default=30.0,
                    help="cap on a chop bar, so a quiet stretch terminates")
    ap.add_argument("--min-updates", type=int, default=2)
    ap.add_argument("--tick-raw", type=int, default=250_000_000,
                    help="one ES tick in the feed's raw integer units")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--csv", default="")
    ap.add_argument("--out", default="ofi_bars.txt")
    a = ap.parse_args()

    paths = [f for f in a.files if not f.endswith(".ohlcv.dbn.zst")]
    if not paths:
        print("no input files")
        return 2
    if not a.sweep and a.threshold <= 0:
        print("give --threshold, or --sweep to find one")
        return 2

    levels = list(SWEEP) if a.sweep else [a.threshold]
    builders = {t: OfiBarBuilder(threshold=t, max_s=a.max_s,
                                 min_updates=a.min_updates)
                for t in levels}

    n = 0
    first_ts = last_ts = 0.0
    started = time.time()
    for ts, touch in stream(paths):
        if first_ts == 0.0:
            first_ts = ts
        last_ts = ts
        for b in builders.values():
            b.on_touch(ts, touch)
        n += 1
        if n % 2_000_000 == 0:
            print(f"  {n:>12,} records  "
                  f"{n/(time.time()-started)/1000:.0f}k/s", flush=True)
    for b in builders.values():
        b.flush(last_ts)

    hours = (last_ts - first_ts) / 3600.0 if last_ts > first_ts else 0.0
    out = open(a.out, "w")

    def p(s: str = "") -> None:
        print(s)
        out.write(s + "\n")

    p("=" * 76)
    p(f"ES OFI-CLOSED CANDLES   build {BUILD}")
    p("=" * 76)
    p(f"  {n:,} records   {hours:.1f} hours")
    p(f"  chop cap {a.max_s}s   min updates per bar {a.min_updates}")
    p()
    p(f"  {'threshold':>10}{'bars':>10}{'per hour':>10}{'green':>8}"
      f"{'red':>8}{'chop':>8}{'med secs':>10}{'med ticks':>11}")

    for t in levels:
        bars = builders[t].bars
        if not bars:
            p(f"  {t:>10}{'0':>10}   nothing closed")
            continue
        g = sum(1 for x in bars if x.direction == GREEN)
        r = sum(1 for x in bars if x.direction == RED)
        c = sum(1 for x in bars if x.direction == CHOP)
        rng = [(x.high - x.low) / a.tick_raw for x in bars]
        p(f"  {t:>10}{len(bars):>10,}{len(bars)/max(hours,1e-9):>10.0f}"
          f"{g/len(bars)*100:>7.1f}%{r/len(bars)*100:>7.1f}%"
          f"{c/len(bars)*100:>7.1f}%"
          f"{median([x.seconds for x in bars]):>10.2f}"
          f"{median(rng):>11.2f}")

    p()
    p("  PICK THE ROW THAT LOOKS LIKE A CHART YOU WOULD READ. A median")
    p("  range of 0 ticks means the bars have no body -- the threshold")
    p("  is below the noise floor of this book. A handful of bars per")
    p("  hour means it is above anything that happens.")
    p()
    p("  Green and red should sit near even. A long way off means the")
    p("  accumulator is latching or the sign is inverted, and either")
    p("  makes every bar wrong in the same direction.")

    if a.csv and not a.sweep:
        bars = builders[levels[0]].bars
        with open(a.csv, "w", newline="") as fh:
            wtr = csv.writer(fh)
            wtr.writerow(ROW_COLUMNS)
            for x in bars:
                wtr.writerow(x.as_row())
        p()
        p(f"  {len(bars):,} rows -> {a.csv}")
        p(f"  columns: {', '.join(ROW_COLUMNS)}")

    out.close()
    print(f"\nwritten to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
