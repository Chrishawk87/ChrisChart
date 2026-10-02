#!/usr/bin/env python3
"""Print individual setups with their full context, for reading by eye.

    python scripts/mtf_show.py data/ES_c_0_ohlcv-1s_2024-09-30.dbn.zst --n 20

A win rate says the rules lost. It cannot say WHICH rule is wrong. This
prints what the tester actually saw at each entry -- where the 4H and 1H
opens were, what the previous 15m bar looked like, how far the forming one
had travelled, and the two 1-minute bars that made the trigger -- so that
someone who has watched this setup a thousand times can say "I would never
have taken that one, the 15m bar was tiny" or "that is a retest, not a
continuation".

Every one of those is a rule the spec is missing, and no amount of
parameter tuning finds them.

Nothing here is a backtest. It is a window onto the trades one already
made.
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from liqmap import bars as bl                                   # noqa: E402
from liqmap import mtf, mtfsim as sim                           # noqa: E402
from liqmap.levels import ET, trade_date                        # noqa: E402

TICK = 0.25


def clock(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=ET).strftime("%Y-%m-%d %H:%M ET")


def ticks(a: float, b: float) -> float:
    return (a - b) / TICK


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("--fine", type=float, default=5.0)
    p.add_argument("--n", type=int, default=20, help="how many to print")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--only", choices=("win", "loss", "any"), default="any")
    p.add_argument("--hold-min", type=float, default=240.0)
    p.add_argument("--target-ticks", type=float, default=10.0)
    a = p.parse_args()

    fine: list = []
    for f in a.files:
        path = Path(f)
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return 2
        print(f"reading {path.name} ...", flush=True)
        fine.extend(bl.load(str(path), seconds=a.fine))
    fine.sort(key=lambda b: b.ts)

    rolls = bl.roll_dates(trade_date(fine[0].ts), trade_date(fine[-1].ts))
    fine = [b for b in fine if not bl.near_roll(trade_date(b.ts), rolls)]

    # Keep the 1-minute bars per day: the context lines below are read off
    # the same bars the engine decided on, not re-derived from something
    # coarser that might disagree with them.
    minutes: dict = {}
    days: dict = {}
    for b in fine:
        days.setdefault(trade_date(b.ts), []).append(b)

    signals: list = []
    for day in sorted(days):
        m = bl.resample(days[day], 60.0)
        if len(m) < 120:
            continue
        minutes[day] = m
        signals.extend(mtf.scan(m, tick=TICK, target_ticks=a.target_ticks))
    signals.sort(key=lambda s: s.ts)
    if not signals:
        print("no signals")
        return 0

    trades = sim.walk(signals, fine, tick=TICK, hold_minutes=a.hold_min)
    by_ts = {s.ts: s for s in signals}

    pool = [t for t in trades
            if a.only == "any"
            or (a.only == "win" and t.outcome == sim.TARGET)
            or (a.only == "loss" and t.outcome == sim.STOP)]
    if not pool:
        print("nothing matched")
        return 0

    rng = random.Random(a.seed)
    pick = sorted(rng.sample(pool, min(a.n, len(pool))), key=lambda t: t.ts)

    print(f"\n{len(trades):,} trades taken; showing {len(pick)} "
          f"({a.only})\n")

    for t in pick:
        s = by_ts.get(t.ts)
        if s is None:
            continue
        m = minutes.get(trade_date(t.ts), [])
        i = next((k for k, b in enumerate(m) if b.ts == t.ts), None)
        trig = m[i] if i is not None else None
        prev = m[i - 1] if i else None

        # The 15m bars, rebuilt from the minutes so they are exactly what
        # the engine saw rather than a fresh resample of everything.
        same = [b for b in m if s.m15_start <= b.ts < s.m15_start + 900]
        before = [b for b in m
                  if s.m15_start - 900 <= b.ts < s.m15_start]
        upto = [b for b in same if b.ts <= t.ts]

        print("=" * 74)
        mark = {"target": "WIN ", "stop": "LOSS", "unresolved": "OPEN"}
        print(f"{mark.get(t.outcome, '?')}  {t.side.upper():<5s} "
              f"{clock(t.ts)}   held {t.minutes:.0f} min   "
              f"{t.ticks:+.0f}t")
        print(f"   4H open {s.h4_open:>9,.2f}  "
              f"price {ticks(t.entry, s.h4_open):+7.0f}t from it")
        print(f"   1H open {s.h1_open:>9,.2f}  "
              f"price {ticks(t.entry, s.h1_open):+7.0f}t from it")

        if before:
            o, c = before[0].open, before[-1].close
            hi = max(b.high for b in before)
            lo = min(b.low for b in before)
            print(f"   prior 15m  {'GREEN' if c > o else 'RED  '}  "
                  f"body {ticks(c, o):+6.0f}t   "
                  f"range {ticks(hi, lo):5.0f}t")
        if upto:
            o, c = upto[0].open, upto[-1].close
            hi = max(b.high for b in upto)
            lo = min(b.low for b in upto)
            print(f"   this 15m   {'GREEN' if c > o else 'RED  '}  "
                  f"body {ticks(c, o):+6.0f}t   "
                  f"range {ticks(hi, lo):5.0f}t   "
                  f"({len(upto)} of 15 min in, bar {s.bar_15_index} of the "
                  f"4H block)")
        if prev is not None and trig is not None:
            print(f"   1m before  o{prev.open:>9,.2f} h{prev.high:>9,.2f} "
                  f"l{prev.low:>9,.2f} c{prev.close:>9,.2f}")
            print(f"   1m TRIGGER o{trig.open:>9,.2f} h{trig.high:>9,.2f} "
                  f"l{trig.low:>9,.2f} c{trig.close:>9,.2f}   "
                  f"range {ticks(trig.high, trig.low):.0f}t")
        print(f"   fill {t.entry:,.2f}   target {t.target:,.2f} "
              f"({s.target_ticks:.0f}t)   stop {t.stop:,.2f} "
              f"({s.stop_ticks:.0f}t)")
        print(f"   15m closed: {s.status}")

    print("=" * 74)
    print("\nWhat to look for: setups you would have skipped, and why.\n"
          "Common ones worth naming if you see them -- a 15m bar too small\n"
          "to matter, a trigger bar that is one wide spike, an entry miles\n"
          "from the 1H open, a stop so far away the trade was never worth\n"
          "taking, or the same 4H block producing the same trade over and\n"
          "over. Each is a rule, and a rule can be tested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
