#!/usr/bin/env python3
"""Measure the candle read against what the candles actually did.

    python scripts/read_study.py data/ES_c_0_mbp-10_2026-09-21.dbn.zst

Needs an MBP-10 file, not OHLCV. Flow, the book and absorption are three
of the read's seven signals and the most informative three; an OHLCV file
contains none of them, and running this on one would be measuring
position-in-range and VWAP while calling it the candle read.

WHAT YOU WILL SEE

Accuracy per lean, at three points inside the candle, each next to the
BASE RATE and a shuffled control. The control permutes which candle got
which lean, keeping how often the read says up, down and flat exactly as
observed. If the real ordering does not clear it, the read is reporting
the mix of leans meeting the mix of outcomes.

WHAT IT CANNOT TELL YOU

Two days of ES is a couple of hundred candles. The intervals will be very
wide and a bucket under 25 candles is flagged rather than reported. This
is a look at the shape, not a validation -- and the honest path to a
number you can lean on is the Calibration class in `candleread`, fed from
live reads over weeks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from liqmap import dbn, readstudy as rs                        # noqa: E402
from liqmap.levels import trade_date                           # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="MBP-10 .dbn/.dbn.zst files")
    p.add_argument("--interval", type=float, default=900.0,
                   help="candle length in seconds (default 900 = 15m)")
    p.add_argument("--symbol", default="ES")
    p.add_argument("--flip", action="store_true",
                   help="invert the aggressor convention (see the side "
                        "audit in dbn.py before using this)")
    p.add_argument("--json", default=None)
    a = p.parse_args()

    obs: list = []
    for f in a.files:
        path = Path(f)
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return 2
        print(f"reading {path.name} "
              f"({path.stat().st_size / 1e6:,.0f} MB) ...", flush=True)
        store = dbn.open_file(str(path))
        obs.extend(rs.study(dbn.stream(store, symbol=a.symbol,
                                       flip=a.flip),
                            coin=a.symbol, interval_s=a.interval))

    if not obs:
        print("no observations -- is this an MBP-10 file?", file=sys.stderr)
        return 2

    candles = {o.candle_start for o in obs}
    base = rs.base_rate(obs)
    span = sorted(candles)
    print(f"\n{len(candles)} candles of {a.interval/60:g}m, "
          f"{trade_date(span[0])} to {trade_date(span[-1])}")
    fwd = rs.forward_rate(obs)
    print(f"base rate: {base*100:.1f}% of candles closed up; "
          f"price rose after {fwd*100:.1f}% of reads\n")
    print("ACCURACY HERE IS FORWARD: did price move the read's way AFTER\n"
          "the read, not whether the candle closed in the direction it was\n"
          "already going. The second column is that weaker question, kept\n"
          "for comparison -- on a random walk it scores far better, because\n"
          "a candle that is already up usually closes up.\n")

    rows: dict[str, dict] = {}
    for frac in rs.FRACTIONS:
        at = [o for o in obs if o.fraction == frac]
        if not at:
            continue
        table = rs.by_lean(at)
        ctl = rs.control(at)
        print(f"{'=' * 70}")
        print(f"READ TAKEN {frac*100:.0f}% INTO THE CANDLE "
              f"({frac * a.interval / 60:.1f} min in)")
        print(f"{'=' * 70}")
        print(f"  {'lean':<6s} {'n':>5s} {'fwd':>7s} {'95% CI':^13s} "
              f"{'avg fwd':>9s} {'candle':>8s}")
        for lean in ("up", "down", "flat"):
            r = table.get(lean)
            if r is None:
                continue
            if lean == "flat":
                print(f"  {'flat':<6s} {r['n']:>5d} "
                      f"{'(no call)':>7s} {'':^13s} "
                      f"{r['avg_ticks']:>8.1f}t"
                      + ("" if r["enough"] else "   [thin]"))
                continue
            print(f"  {lean:<6s} {r['n']:>5d} {r['rate']*100:>6.1f}% "
                  f"[{r['ci_low']*100:4.1f}-{r['ci_high']*100:4.1f}] "
                  f"{r['avg_ticks']:>8.1f}t {r['candle_rate']*100:>7.1f}%"
                  + ("" if r["enough"] else "   [thin: under "
                     f"{rs.MIN_CANDLES} candles, not worth reading]"))

        if ctl.get("runs"):
            direc = [o for o in at if o.lean != "flat"]
            hits = sum(1 for o in direc if o.hit)
            got = hits / len(direc)
            print(f"\n  directional calls: {len(direc)}, "
                  f"{got*100:.1f}% right")
            print(f"  shuffled control:  {ctl['mean']*100:.1f}% "
                  f"+/- {ctl['sd']*100:.1f}  "
                  f"(95th pct {ctl['p95']*100:.1f}%)")
            print(f"  clears it by 2 sd: "
                  f"{'YES' if rs.beats(got, len(direc), ctl) else 'NO'}")
            rows[f"{frac}"] = {"table": table, "control": ctl,
                               "directional": len(direc), "rate": got}
        print()

    print("=" * 70)
    print("The control permutes WHICH candle got which lean, keeping how\n"
          "often the read says up, down and flat exactly as observed. What\n"
          "it scores is what a read with no information scores -- and that\n"
          "is not 50%, it is the mix of leans meeting the mix of outcomes.")
    print("\nTwo days is a couple of hundred candles. Treat anything here\n"
          "as a look at the shape, not a result.")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"candles": len(candles), "base_rate": base, "by_fraction": rows},
            indent=2, default=str))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
