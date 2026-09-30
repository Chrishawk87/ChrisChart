#!/usr/bin/env python3
"""The book at its own rate. Does it call the next few seconds?

    python scripts/es_intrabar.py data/ES_c_0_mbp-10_2026-09-21.5s.slices.jsonl

THE QUESTION

Every earlier test sampled the book once per candle -- 5,969 looks out of
44 million updates. This samples every few seconds instead and asks, at
each horizon, whether depth imbalance, the speed it is changing, or the
tape over the same window calls the direction of the next few seconds.

WHAT WOULD COUNT

Not a hit rate on its own. A taker's round trip in ES costs commission plus
the spread given up buying the ask and selling the bid -- about $17.50, not
the $5 used earlier in this project. Against that, direction must clear

    cost / (total barrier width x $12.50)

which is 17.5% for a 5/3 scalp and 3.5% for a 20/20 hold. Both the measured
edge and the required edge are printed together, so the gap is visible
rather than argued about.

OVERLAP IS HANDLED, NOT IGNORED

Each horizon is scored on observations spaced at least its own length
apart. Sampling every 5 seconds and measuring 30 forward would share five
sixths of the outcome between neighbours, and the interval over 74,000 such
looks would be several times tighter than the data supports. Fewer rows,
honest intervals.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap import escondense as ec, intrabar as ib          # noqa: E402

FEATURES = ("imb1", "imb3", "imb5", "imb10", "velocity", "flow")

# Horizons in seconds; converted to rows using the file's own spacing.
HORIZONS_S = (5, 10, 30, 60, 300)

# Barriers to quote the requirement against.
BARRIERS = ((5, 3), (10, 10), (20, 20))


def parse(argv):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("slices")
    p.add_argument("--cut", type=float, default=0.80,
                   help="Quantile of |feature| above which a call is made "
                        "(default 0.80 -- the strongest fifth).")
    p.add_argument("--cost", type=float, default=ib.COST_TAKER,
                   help=f"Round-trip cost in dollars "
                        f"(default {ib.COST_TAKER:.2f}: commission plus the "
                        f"one-tick spread a taker gives up).")
    return p.parse_args(argv)


def spacing(slices) -> float:
    if len(slices) < 3:
        return 0.0
    gaps = sorted(slices[i + 1].ts - slices[i].ts
                  for i in range(min(200, len(slices) - 1)))
    return gaps[len(gaps) // 2]


def main(argv=None) -> int:
    a = parse(argv or sys.argv[1:])
    path = Path(a.slices)
    if not path.exists():
        print(f"No such file: {path}\n\nParse one first:\n"
              f"  python scripts/es_study.py <file>.dbn.zst "
              f"--interval 5 --lookback 10", file=sys.stderr)
        return 2

    slices = ec.load(path)
    slices.sort(key=lambda s: s.ts)
    if len(slices) < 2000:
        print(f"Only {len(slices):,} rows. Re-parse at --interval 5.",
              file=sys.stderr)
        return 1

    step = spacing(slices)
    if step <= 0:
        print("Could not determine the sampling interval.", file=sys.stderr)
        return 1

    hours = (slices[-1].ts - slices[0].ts) / 3600.0
    print(f"\n{len(slices):,} samples, {step:g}s apart, over {hours:,.1f} "
          f"hours")
    print(f"round-trip cost ${a.cost:,.2f}")

    print("\nwhat a signal must clear, at this cost:")
    for t, s in BARRIERS:
        need = ib.needed_edge(t, s, cost_usd=a.cost)
        base = ib.no_skill(t, s)
        print(f"  +{t}/-{s:<4} no-skill {base:>5.1%}   "
              f"must reach {base + need:>5.1%}   (+{need:.1%})")

    rs = ib.reads(slices)

    print(f"\n=== direction, by feature and horizon "
          f"(calls on the strongest {int((1-a.cut)*100)}%) ===")
    header = "  " + f"{'feature':<10}" + "".join(
        f"{str(h) + 's':>16}" for h in HORIZONS_S)
    print(header)

    best = None
    for name in FEATURES:
        cut = ib.quantile_cut(rs, name, a.cut)
        if cut <= 0:
            print(f"  {name:<10}  flat -- no spread to threshold on")
            continue
        sides = ib.signal(rs, name, cut)
        cells = []
        for h in HORIZONS_S:
            rows = max(1, int(round(h / step)))
            d = ib.direction(slices, sides, rows)
            if not d["ready"]:
                cells.append(f"{'--':>16}")
                continue
            mark = "*" if d["real"] else " "
            cells.append(f"{d['rate']:>14.1%}{mark} ")
            # Only a cell whose interval clears 50% is a candidate. Taking
            # the furthest-from-50 regardless would crown whichever cell
            # had the fewest observations, since those wander most.
            if d["real"] and (best is None
                              or abs(d["rate"] - 0.5) > abs(best[2] - 0.5)):
                best = (name, h, d["rate"], d)
        print(f"  {name:<10}" + "".join(cells))

    print("\n  * = interval excludes 50%. With "
          f"{len(FEATURES) * len(HORIZONS_S)} cells examined, expect one or "
          "two starred\n    by chance alone -- a star is a reason to test "
          "it properly, not a finding.")

    # How far price actually travels in each horizon. An accuracy figure
    # means nothing if the window is too short to reach the target: calling
    # the next five seconds at 70% is worthless when five seconds moves one
    # tick and the trade needs five.
    print("\n=== can the horizon even reach a target? (median |move|) ===")
    print(f"  {'horizon':<10}{'median':>10}{'   90th pct':>14}"
          f"   reaches 5 ticks?")
    for h in HORIZONS_S:
        rows = max(1, int(round(h / step)))
        moves = []
        for i in range(0, len(slices) - rows, rows):
            mv = ib.forward_bps(slices, i, rows)
            if mv is not None:
                moves.append(abs(mv))
        if len(moves) < 30:
            continue
        moves.sort()
        px = slices[-1].open or 7000.0
        tick_bps = 0.25 / px * 10_000.0
        med = moves[len(moves) // 2] / tick_bps
        p90 = moves[int(0.9 * len(moves))] / tick_bps
        share = sum(1 for m in moves if m / tick_bps >= 5.0) / len(moves)
        print(f"  {str(h) + 's':<10}{med:>7.1f} tk{p90:>11.1f} tk"
              f"      {share:>6.1%} of windows")

    if best is not None:
        name, h, rate, d = best
        print(f"\n  furthest from a coin: {name} at {h}s -- "
              f"{rate:.1%} ({d['lo']:.1%}-{d['hi']:.1%}), n={d['n']:,}, "
              f"stride {d['stride']} rows")
        gap = abs(rate - 0.5)
        print(f"  that is {gap:.1%} away from a coin.")
        for t, s in BARRIERS:
            need = ib.needed_edge(t, s, cost_usd=a.cost)
            verdict = "clears" if gap > need else "short by"
            extra = "" if gap > need else f" {need - gap:.1%}"
            print(f"    +{t}/-{s}: needs {need:.1%} -> {verdict}{extra}")
    else:
        print("\n  Nothing cleared 50% by its own interval. No candidate.")

    print("\n  These are in-sample. Anything starred and large enough to "
          "matter goes\n  to a sealed week before it counts.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
