#!/usr/bin/env python3
"""Run the OFI backtest over a per-second file and print the table.

    python scripts/ofi_backtest.py ~/Downloads/ES_sec.csv

    python scripts/ofi_backtest.py ES_sec.csv --z 2.0 \\
        --target 16 --stop 4 --cost 0.4

The file comes from `scripts/ofi_replay.py`. The engine lives in
`liqmap/ofi_backtest.py` so the site can run the same code on the
same data -- a backtest that disagrees with the one on the dashboard
is worse than having only one of them.

READ THE HELD-OUT COLUMN. Nothing is searched here, so the split is
not protecting you from a parameter sweep; it is there so a result
that only holds in the first half is visible as one.

READ MIRROR BEFORE YOU BELIEVE SIGNAL. The same trades taken the
other way round share the bracket's geometry and none of the signal.
If they score alike, the bracket is what was measured.

SWEEPING IS NOT TUNING. `--sweep` prints a row per threshold so the
shape is visible. A shape that improves smoothly with the threshold
is a different kind of evidence from one good cell among bad ones,
and picking that cell is how noise gets shipped.

IT PRINTS. It places nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap import ofi_backtest as bt                       # noqa: E402


def pct(v):
    return "  --  " if v is None else f"{v:6.1%}"


def table(out: dict, split: str = "held_out") -> None:
    print(f"\n{split.replace('_', ' ')}")
    print(f"  {'arm':<9}{'scored':>8}{'win%':>8}{'[lo':>8}{'hi]':>8}"
          f"{'net t':>9}{'per-trade':>11}{'ambig':>7}{'unres':>7}")
    for name in ("signal", "mirror", "control"):
        a = out["arms"][name][split]
        pt = ("   --  " if a["per_trade_ticks"] is None
              else f"{a['per_trade_ticks']:+7.3f}")
        print(f"  {name:<9}{a['scored']:>8,}{pct(a['win_rate'])}"
              f"{pct(a['win_rate_lo'])}{pct(a['win_rate_hi'])}"
              f"{a['net_ticks']:>9,.0f}{pt:>11}"
              f"{a['ambiguous']:>7,}{a['unresolved']:>7,}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="the CSV from ofi_replay.py")
    ap.add_argument("--z", type=float, default=1.5, help="your line")
    ap.add_argument("--window", type=int, default=60,
                    help="seconds summed into the current OFI")
    ap.add_argument("--baseline", type=int, default=600,
                    help="window sums the z is measured against")
    ap.add_argument("--target", type=int, default=16, help="ticks")
    ap.add_argument("--stop", type=int, default=4, help="ticks")
    ap.add_argument("--hold", type=int, default=900,
                    help="seconds before a trade is booked as it stands")
    ap.add_argument("--cost", type=float, default=1.0,
                    help="round-trip ticks. PRE-REGISTER THIS")
    ap.add_argument("--cooldown", type=int, default=60)
    ap.add_argument("--train", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sweep", default="",
                    help="comma separated z values, e.g. 1,1.5,2,2.5")
    args = ap.parse_args(argv)

    if not Path(args.path).exists():
        print(f"no such file: {args.path}", file=sys.stderr)
        return 2

    print(f"loading {args.path} ...", file=sys.stderr)
    secs = bt.load(args.path, limit=args.limit)
    if not len(secs):
        print("the file has no rows", file=sys.stderr)
        return 1
    print(f"{len(secs):,} seconds, {secs.priced:,} with a print, "
          f"{secs.span_hours():,.0f} hours", file=sys.stderr)

    def settings(z):
        return bt.Settings(
            z_enter=z, window=args.window, baseline=args.baseline,
            target_ticks=args.target, stop_ticks=args.stop,
            max_hold_s=args.hold, cost_ticks=args.cost,
            cooldown_s=args.cooldown, train_fraction=args.train)

    be = bt.break_even(args.target, args.stop, args.cost)
    print(f"\n{args.target}t target / {args.stop}t stop at "
          f"{args.cost:g} ticks of cost")
    print(f"break-even {be:.1%} -- both sides pay the cost")

    if args.sweep:
        zs = [float(x) for x in args.sweep.split(",") if x.strip()]
        print(f"\n{'z':>6}{'entries':>9}{'held':>7}{'signal':>9}"
              f"{'mirror':>9}{'control':>9}{'sig/trade':>11}")
        for z in zs:
            out = bt.run(secs, settings(z))
            if not out.get("ok"):
                print(f"{z:>6}  {out['why']}")
                continue
            h = {a: out["arms"][a]["held_out"]
                 for a in ("signal", "mirror", "control")}
            pt = h["signal"]["per_trade_ticks"]
            print(f"{z:>6}{out['entries']:>9,}"
                  f"{h['signal']['scored']:>7,}"
                  f"{pct(h['signal']['win_rate'])}"
                  f"{pct(h['mirror']['win_rate'])}"
                  f"{pct(h['control']['win_rate'])}"
                  f"{('   --  ' if pt is None else f'{pt:+7.3f}'):>11}")
        print("\nA shape that improves smoothly with z is evidence. One "
              "good row\namong bad ones is the row you would have picked "
              "by chance. Read the\n`held` column too: a win rate over "
              "forty trades is not a measurement.")
        return 0

    out = bt.run(secs, settings(args.z))
    if not out.get("ok"):
        print(f"\n{out['why']}")
        return 1
    print(f"entries {out['entries']:,} at |z| >= {args.z:g}  "
          f"(split by entry count, not wall clock)")
    print(f"coverage {out['coverage_hours']:,.0f}h of data across a "
          f"{out['hours']:,.0f}h span")
    table(out, "train")
    table(out, "held_out")
    print(f"\nverdict: {out['verdict']}")
    print("\nAmbiguous trades touched both brackets inside one second; "
          "the file does\nnot say which came first, so they are set "
          "aside rather than guessed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
