#!/usr/bin/env python3
"""Flag LONG / SHORT / NONE per bar. Classification, not prediction.

    python scripts/auction_signals.py data/ES_c_0_ohlcv-1s_2024-09-30.dbn.zst
    python scripts/auction_signals.py <file> --bar 300 --json data/signals.json
    python scripts/auction_signals.py <file> --session 2026-09-25 --show

One row per flagged bar:

    timestamp | state | location | pattern | signal | entry | stop | target

Levels come from the PRIOR session's completed profile -- what is on the
chart at the open. Today's own profile supplies nothing, because a level
derived from the session being classified would be defined partly by the
move being labelled.

Nothing here computes expectancy, applies cost, ranks patterns or says
whether any of it is worth trading. It reports what fired and why.

VALIDATE BY EYE, NOT BY BACKTEST. --show prints the flags for one session
with the levels beside them so they can be checked against a chart. The
question is "would I call that a long?", and only the reader can answer it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap import auction as au, bars as bl, profile as pf  # noqa: E402
from liqmap.levels import Phase, phase, trade_date            # noqa: E402


def parse(argv):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data")
    p.add_argument("--bar", type=float, default=300.0,
                   help="Bar size in seconds for classification "
                        "(default 300 = 5 minutes).")
    p.add_argument("--session", default=None,
                   help="Only this trade date, e.g. 2026-09-25.")
    p.add_argument("--show", action="store_true",
                   help="Print every flag with the session's levels.")
    p.add_argument("--json", default=None,
                   help="Write flags for the chart to overlay.")
    return p.parse_args(argv)


def levels_from(p: pf.Profile) -> au.Levels:
    peak = float(p.volumes.max()) if p.volumes.size else 0.0
    strength = {}
    for px in [p.poc, p.vah, p.val] + list(p.hvn) + list(p.lvn):
        row = int(px / p.tick) - p.lo_row
        if 0 <= row < p.volumes.size and peak > 0:
            strength[round(px, 4)] = ("strong"
                                      if p.volumes[row] >= 0.6 * peak
                                      else "weak")
    return au.Levels(poc=p.poc, vah=p.vah, val=p.val,
                     hvn=list(p.hvn), lvn=list(p.lvn), strength=strength)


def regime_of(p: pf.Profile) -> str:
    """From the PRIOR session, so it is known at the open."""
    w = p.va_width_ticks
    return "balance_day" if w < 40 else "range_day" if w < 90 else "trend_day"


def main(argv=None) -> int:
    args = parse(argv or sys.argv[1:])
    path = Path(args.data)
    if not path.exists():
        print(f"No such file: {path}", file=sys.stderr)
        return 2

    if path.suffix in (".zst", ".dbn"):
        raw = bl.load(str(path), seconds=args.bar)
        found = bl.sessions(raw, rth_only=True)
        days = {s.day: s.bars for s in found if not s.suspect and s.bars}
        dropped = sum(1 for s in found if s.suspect)
    else:
        from liqmap import escondense as ec
        sl = sorted(ec.load(path), key=lambda s: s.ts)
        coarse = bl.resample(
            [bl.Bar(x.ts, x.open, x.high, x.low, x.close, x.volume)
             for x in sl if phase(x.ts) is Phase.RTH], args.bar)
        days, dropped = {}, 0
        for b in coarse:
            days.setdefault(trade_date(b.ts), []).append(b)

    ordered = sorted(days)
    print(f"\n{len(ordered)} sessions, {args.bar:g}s bars"
          + (f", {dropped} dropped at rolls" if dropped else ""))
    print(f"levels: prior session's profile | acceptance "
          f"{au.ACCEPT_CLOSES} closes | rotation "
          f"{au.MIN_ROTATION_TICKS:g} ticks | re-arm {au.REARM_TICKS:g}")

    wanted = None
    if args.session:
        import datetime as dt
        wanted = dt.date.fromisoformat(args.session)

    flags: list[au.Signal] = []
    per_session: Counter = Counter()

    for i in range(1, len(ordered)):
        day = ordered[i]
        if wanted and day != wanted:
            continue
        prior = pf.final(days[ordered[i - 1]])
        if prior.empty:
            continue
        lv = levels_from(prior)
        reg = regime_of(prior)
        clf = au.Classifier(lv)

        if args.show:
            print(f"\n  === {day}  ({reg}) ===")
            print(f"  VAH {lv.vah:,.2f}   POC {lv.poc:,.2f}   "
                  f"VAL {lv.val:,.2f}")
            print(f"  HVN {[round(x, 2) for x in lv.hvn]}")
            print(f"  LVN {[round(x, 2) for x in lv.lvn]}")

        n_here = 0
        for b in days[day]:
            s = clf.step(b, regime=reg)
            if s is None:
                continue
            flags.append(s)
            n_here += 1
            per_session[s.pattern] += 1
            if args.show:
                import datetime as dt
                hhmm = dt.datetime.fromtimestamp(
                    s.ts, tz=__import__("liqmap.levels", fromlist=["ET"]).ET
                ).strftime("%H:%M")
                print(f"    {hhmm}  {s.side.value:<5} {s.pattern:<22}"
                      f" {s.node_kind}@{s.node_price:,.2f}"
                      f"  entry {s.entry:,.2f}  stop {s.stop:,.2f}"
                      f"  target {s.target:,.2f}"
                      f"  ({s.risk_ticks:.0f}t risk / "
                      f"{s.reward_ticks:.0f}t reward)")
        if args.show:
            print(f"    -- {n_here} flags, "
                  f"{len(days[day]) - n_here} bars NONE")

    sessions_used = max(1, (len(ordered) - 1) if not wanted else 1)
    print(f"\n{len(flags):,} flags over {sessions_used} sessions "
          f"= {len(flags) / sessions_used:.1f} per session")
    print("\n  pattern                     count   per session")
    for name, n in per_session.most_common():
        print(f"  {name:<28}{n:>6}{n / sessions_used:>13.1f}")

    longs = sum(1 for s in flags if s.side is au.Side.LONG)
    print(f"\n  LONG {longs:,}   SHORT {len(flags) - longs:,}")

    # The rate is the first thing to sanity-check. Four hundred flags a
    # week means the rules are loose; two means they are dead.
    rate = len(flags) / sessions_used
    if rate > 25:
        print(f"\n  {rate:.0f} per session is high -- the definitions are "
              f"probably too loose\n  to be read on a chart.")
    elif rate < 2:
        print(f"\n  {rate:.1f} per session is sparse -- the definitions may "
              f"be too tight.")

    if args.json:
        out = Path(args.json)
        out.write_text(json.dumps({
            "bar_seconds": args.bar,
            "conventions": {"accept_closes": au.ACCEPT_CLOSES,
                            "at_ticks": au.AT_TICKS,
                            "memory_bars": au.MEMORY_BARS,
                            "rearm_ticks": au.REARM_TICKS,
                            "min_rotation_ticks": au.MIN_ROTATION_TICKS},
            "sessions": sessions_used,
            "flags": [s.to_dict() for s in flags],
        }, indent=1))
        print(f"\n  wrote {out} ({out.stat().st_size / 1000:,.0f} KB)")

    print("\n  Classification only. No expectancy, no cost, no ranking,")
    print("  no view on whether any pattern is worth trading.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
