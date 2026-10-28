#!/usr/bin/env python3
"""Did OFI agreeing with delta precede price? Every call, catalogued.

    source .venv/bin/activate
    python scripts/agree_catalog.py data/ES_*.dbn.zst --target 250

    # a quick look before committing to the whole file
    python scripts/agree_catalog.py data/ES_*.dbn.zst --limit 5000000

    # every target on the ladder, one pass over the data
    python scripts/agree_catalog.py data/ES_*.dbn.zst --all-targets

THE RULE

    UP      the bar closed GREEN and delta is positive
    DOWN    the bar closed RED and delta is negative

THE MEASURE

Did price go that way, and by how much, at one bar, two, three, five,
ten and twenty. No target, no stop, no cost: this is not a backtest
and it does not model a trade.

WHAT COMES OUT

A summary table on screen, and a CSV with one row per call carrying
the OFI, the delta, the volume and the move at every horizon. The CSV
is the record -- the thing to open when a percentage looks surprising
-- and it is written next to the data by default.

UP, FLAT AND DOWN ARE THREE OUTCOMES. At one bar on a one-tick market
more than three quarters of moves are exactly zero. Which column you
put those in moves the headline number by fifty points, so they get
their own, and both percentages are printed.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from liqmap import agree as A                       # noqa: E402
from liqmap.ofibar import LADDER, OfiBarBuilder      # noqa: E402

RAW_TICK = 250_000_000.0
TICK = 0.25


def build(paths: list[str], targets: list[int], limit: int = 0,
          level: int = 500):
    """One pass over the files, feeding every target at once."""
    import databento as db

    from liqmap import dbn

    builders = {t: OfiBarBuilder(tick_size=TICK, threshold=t, max_s=900.0,
                                 min_updates=2, max_bars=10_000_000,
                                 call_level=level)
                for t in targets}
    n = 0
    for path in paths:
        store = db.DBNStore.from_file(path)
        for rec in store:
            t = dbn.trade_from(rec)
            ts = float(dbn.seconds(dbn._get(rec, "ts_event")) or 0.0)
            if t is not None:
                # Raw price off the record. `t.px` is dollars and the
                # bar's prices are raw feed units.
                px = float(dbn._get(rec, "price", 0) or 0)
                for b in builders.values():
                    b.on_trade(ts, float(t.sz), t.aggressor, px)
            lv = dbn._get(rec, "levels")
            if lv:
                l0 = lv[0]
                bpx = int(dbn._get(l0, "bid_px", 0) or 0)
                apx = int(dbn._get(l0, "ask_px", 0) or 0)
                bsz = int(dbn._get(l0, "bid_sz", 0) or 0)
                asz = int(dbn._get(l0, "ask_sz", 0) or 0)
                for b in builders.values():
                    b.on_book(ts, bpx, bsz, apx, asz)
            n += 1
            if limit and n >= limit:
                break
            if n % 5_000_000 == 0:
                print(f"  {n:,} records...", flush=True)
        if limit and n >= limit:
            break
    print(f"  {n:,} records read")
    return builders


def show_observations(target: int, bars) -> None:
    """The strategy as he states it: every stretch where imbalance and
    delta agreed, scored at the distance it was shooting for, inside
    the bar it happened in."""
    t = A.observation_table(bars, points=(1, 2, 3, 4))
    print(f"\n  OBSERVATIONS  {t['observations']:,} across "
          f"{t['bars']:,} bars ({t['per_bar']} a bar, "
          f"{t['silent_bars']:,} bars never called) "
          f"{t['up']:,} up / {t['down']:,} down")
    print(f"  {'distance':<12}{'hit it':>9}{'hit other':>11}"
          f"{'bar ended':>11}{'of decided':>12}{'resolved':>10}")
    for r in t["rows"]:
        rate = f"{r['pct_of_decided']:.1%}" if r["decided"] else "--"
        print(f"  {str(r['points']) + ' point':<12}{r['win']:>9,}"
              f"{r['loss']:>11,}{r['unresolved']:>11,}{rate:>12}"
              f"{r['pct_resolved']:>10.1%}")
    med = t["median_updates"]
    print(f"\n  Median observation is {med:,} updates long. Ended: "
          + ", ".join(f"{v:,} {k}" for k, v in t["ends"].items()) + ".")
    if med < 20:
        print("  THAT IS TOO SHORT TO TRAVEL TWO POINTS. The table above is")
        print("  measuring the call level, not the rule. Raise --level and")
        print("  run it again before reading anything into those"
              " percentages.")
    print("  BAR ENDED FIRST is not a draw and not a loss -- it is price"
          " never")
    print("  getting that far either way. At four points it is usually"
          " most of")
    print("  the sample.")


def show(target: int, bars, horizons) -> list[dict]:
    s = A.summary(bars, horizons=horizons, tick=RAW_TICK)
    print(f"\n{'=' * 78}")
    print(f"TARGET {target:,}   {s['bars']:,} bars   "
          f"{s['called']:,} calls ({s['share_called']:.1%} of bars)")
    print("=" * 78)
    if not s["rows"]:
        print("  no finished calls at any horizon")
        return s["rows"]
    print(f"  {'call':<6}{'bars fwd':>9}{'calls':>9}{'right':>9}"
          f"{'flat':>9}{'wrong':>9}{'of all':>9}{'of decided':>12}"
          f"{'mean tk':>9}")
    for r in s["rows"]:
        print(f"  {r['call']:<6}{r['horizon']:>9}{r['n']:>9,}"
              f"{r['right']:>9,}{r['flat']:>9,}{r['wrong']:>9,}"
              f"{r['pct_of_all']:>9.1%}{r['pct_of_decided']:>12.1%}"
              f"{r['mean_ticks']:>+9.3f}")
    print("\n  OF ALL counts a flat outcome against the call; OF DECIDED"
          " sets flats")
    print("  aside. Both are true. At one bar most outcomes are flat, so"
          " the two")
    print("  are far apart, and quoting either alone is misleading.")
    print("  A market with nothing in it returns 50% OF DECIDED. That is"
          " the line")
    print("  to read these against, not zero.")
    return s["rows"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help=".dbn / .dbn.zst MBP files")
    ap.add_argument("--target", type=int, default=250)
    ap.add_argument("--all-targets", action="store_true",
                    help="every threshold on the ladder, one pass")
    ap.add_argument("--forward", type=int, nargs="+",
                    default=list(A.HORIZONS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--level", type=int, default=500,
                    help="imbalance, in contracts, at which a call is"
                         " made; it reverses coming back through it")
    ap.add_argument("--out", default="",
                    help="CSV path; defaults to agree_calls_<target>.csv"
                         " beside the first input file")
    a = ap.parse_args()

    targets = list(LADDER) if a.all_targets else [a.target]
    horizons = tuple(sorted(set(int(h) for h in a.forward if h > 0)))
    print(f"\nAGREEMENT CATALOGUE   targets {targets}   "
          f"horizons {list(horizons)}")
    print("  building bars...", flush=True)
    unreachable = [t for t in targets if t <= a.level]
    if unreachable:
        print(f"\n  NOTE: a bar ends when the imbalance crosses its"
              f" target, so the")
        print(f"  accumulator never travels past it. At target(s)"
              f" {unreachable} a")
        print(f"  {a.level}-contract call can never fire, and an empty"
              f" table there is")
        print(f"  the arithmetic rather than the market.")
    builders = build(a.files, targets, a.limit, a.level)

    base = Path(a.files[0]).resolve().parent
    for t in targets:
        bars = builders[t].bars
        rows = show(t, bars, horizons)
        show_observations(t, bars)
        if not rows:
            continue
        out = (Path(a.out) if a.out and len(targets) == 1
               else base / f"agree_calls_{t}.csv")
        cat = A.catalogue(bars, horizons=horizons, tick=RAW_TICK)
        if cat:
            cols = list(cat[0].keys())
            with open(out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                w.writerows(cat)
            print(f"\n  {len(cat):,} calls written to {out}")
        else:
            print("\n  no calls to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
