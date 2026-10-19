#!/usr/bin/env python3
"""Measure whether a footprint imbalance predicts anything.

    python scripts/footprint_test.py data/*mbp-10*.dbn.zst

    python scripts/footprint_test.py data/*.dbn.zst --bars 2000 \\
        --min-volume 25 --target 16 --stop 4 --cost 1.0

Replays your raw MBP-10 through the footprint engine and puts every
setup it produces through the SAME three arms that retired the
timeframe-agreement idea: SIGNAL, MIRROR and CONTROL, with costs
pre-registered and a held-out half.

THREE KINDS, PRINTED SIDE BY SIDE

    single    any confirmed imbalance in the bar
    stacked   three or more consecutive levels on one side
    extreme   a stack reaching the bar's own high or low

They are nested -- every extreme is a stack, every stack is a single
-- so a result that improves down that list is a different kind of
evidence from one good cell. Read the sample sizes next to the win
rates: the stricter kinds are rarer, and a win rate over forty trades
is not a measurement.

WHAT WOULD COUNT AS AN EDGE

Signal clearing break-even out of sample, with its interval above it,
AND beating mirror by more than the intervals overlap. Mirror shares
the bracket's geometry and none of the signal, so if the two score
alike the bracket is what was measured.

IT PRINTS. It places nothing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap import dbn                                     # noqa: E402
from liqmap.footprint import (                             # noqa: E402
    AnyRule, ImbalanceConfig, TimeRule, VolumeRule)
from liqmap.footprint_signals import KINDS, Replay         # noqa: E402


def pct(v):
    return "   --  " if v is None else f"{v:7.1%}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", help=".dbn / .dbn.zst files")
    ap.add_argument("--bars", type=int, default=2000,
                    help="contracts per bar; 0 uses --seconds instead")
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--mult", type=float, default=4.0)
    ap.add_argument("--min-volume", type=int, default=25,
                    help="floor the winning side must clear. 0 "
                         "reproduces the bare formula, which tags any "
                         "volume at all against an empty diagonal")
    ap.add_argument("--run", type=int, default=3,
                    help="levels that make a stack")
    ap.add_argument("--target", type=int, default=16)
    ap.add_argument("--stop", type=int, default=4)
    ap.add_argument("--cost", type=float, default=1.0,
                    help="round-trip ticks. PRE-REGISTER THIS")
    ap.add_argument("--hold", type=int, default=900)
    ap.add_argument("--train", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0,
                    help="records per file, for a quick look")
    ap.add_argument("--flip", action="store_true",
                    help="invert the aggressor convention")
    args = ap.parse_args(argv)

    paths = []
    for p in args.paths:
        q = Path(p)
        if not q.exists():
            print(f"no such file: {p}", file=sys.stderr)
            return 2
        paths.append(q)

    rule = (VolumeRule(args.bars) if args.bars > 0
            else TimeRule(args.seconds))
    rep = Replay(rule=rule, run_min=args.run, flip=args.flip,
                 imbalance=ImbalanceConfig(multiplier=args.mult,
                                           min_volume=args.min_volume))

    print(f"files      {len(paths)}")
    print(f"bars       " + (f"{args.bars:,} contracts" if args.bars > 0
                            else f"{args.seconds:.0f}s"))
    print(f"imbalance  {args.mult:g}x, min volume {args.min_volume}, "
          f"stack {args.run}+")
    print(f"bracket    {args.target}t target / {args.stop}t stop at "
          f"{args.cost:g} ticks\n")

    t0 = time.time()
    n = [0]

    def beat():
        n[0] += 1
        if n[0] % 2_000_000 == 0:
            el = time.time() - t0
            print(f"  {n[0]:,} records  {n[0]/el:,.0f}/s  "
                  f"{rep.bars:,} bars  {len(rep.setups):,} setups",
                  file=sys.stderr)

    import itertools
    for path in sorted(paths):
        store = dbn.open_file(str(path))
        if args.limit:
            store = itertools.islice(store, args.limit)
        rep.feed(store, progress=beat)
        print(f"  {path.name}: {rep.records:,} records, "
              f"{rep.bars:,} bars so far", file=sys.stderr)

    el = time.time() - t0
    print(f"\n{rep.records:,} records -> {rep.trades:,} trades -> "
          f"{rep.bars:,} bars in {el:,.0f}s")
    print(f"{len(rep.secs):,} seconds of tape, "
          f"{rep.secs.span_hours():,.0f}h span")
    if not rep.bars:
        print("\nNo bars were built. If the files carry no trades, "
              "this is the wrong\nschema -- the footprint needs the "
              "aggressor, which only trade records have.")
        return 1

    first = rep.score(kind="single", target_ticks=args.target,
                      stop_ticks=args.stop, cost_ticks=args.cost,
                      max_hold_s=args.hold,
                      train_fraction=args.train)
    print(f"break-even {first['break_even']:.1%} -- both sides pay\n")

    print(f"  {'kind':<9}{'setups':>8}{'held':>7}{'signal':>8}"
          f"{'[lo':>8}{'hi]':>8}{'mirror':>8}{'control':>8}"
          f"{'ticks/trade':>13}")
    rows = {}
    for kind in KINDS:
        out = rep.score(kind=kind, target_ticks=args.target,
                        stop_ticks=args.stop, cost_ticks=args.cost,
                        max_hold_s=args.hold,
                        train_fraction=args.train)
        rows[kind] = out
        if not out.get("ok"):
            print(f"  {kind:<9}  {out['why'][:60]}")
            continue
        h = {a: out["arms"][a]["held_out"]
             for a in ("signal", "mirror", "control")}
        s = h["signal"]
        ptr = s["per_trade_ticks"]
        print(f"  {kind:<9}{out['scored_setups']:>8,}"
              f"{s['scored']:>7,}{pct(s['win_rate'])}"
              f"{pct(s['win_rate_lo'])}{pct(s['win_rate_hi'])}"
              f"{pct(h['mirror']['win_rate'])}"
              f"{pct(h['control']['win_rate'])}"
              f"{('     --  ' if ptr is None else f'{ptr:+13.3f}')}")

    print()
    for kind, out in rows.items():
        if out.get("ok"):
            print(f"  {kind:<9} {out['verdict']}")

    print("\nThe kinds are NESTED -- every extreme is a stack, every "
          "stack is a single.\nA result that improves down the list is "
          "evidence; one good row among bad\nones is the row you would "
          "have picked by chance. Check the `held` column:\na win rate "
          "over forty trades is not a measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
