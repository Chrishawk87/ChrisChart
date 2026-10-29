#!/usr/bin/env python3
"""Every call on the last N bars, one line each, in the order it happened.

CHRIS CHECKED TWELVE BARS BY HAND AND GOT TWELVE. This is the same
check, written down, so the hand count and the tally can be compared
line by line instead of argued about -- and so a disagreement between
them points at a row rather than at a percentage.

One line per call:

    bar   which bar it was in, and that bar's direction
    side  what the call said
    imb   the running imbalance when it fired
    call  the price it fired at -- where the arrow goes
    fill  the price the trade was taken at, after confirmation, or
          "--" if price never confirmed and no trade was taken
    slip  ticks given up waiting
    out   WIN, LOSS, skip, or open
    tks   what it made or cost, in ticks, from the FILL
    why   what ended the call
    len   how many book updates the call lasted. IF THIS IS 2 OR 3,
          THE TRADE NEVER HAD ROOM TO TRAVEL A POINT AND THE OUTCOME
          IS ABOUT THE GATE, NOT THE RULE.
    eff   how much of the bar's movement to that point went anywhere
    tps   ticks a second the bar was covering when it fired

Nothing here predicts, and nothing here places, sizes or times an
order.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap import agree as A                             # noqa: E402
from liqmap.ofibar import (CALL_RULE, LADDER, OfiBarBuilder,   # noqa: E402
                           RAW_TICK)

TICK = 0.25


def load(paths, target: int, level: int, confirm: float, limit: int = 0,
         max_calls: int = 1, widen: bool = True):
    import databento as db

    from liqmap import dbn

    b = OfiBarBuilder(tick_size=TICK, threshold=target, max_s=900.0,
                      min_updates=2, max_bars=10_000_000,
                      call_level=level, objective_points=1.0,
                      confirm_ticks=confirm, max_calls=max_calls,
                      widen=widen)
    n = 0
    for path in paths:
        for rec in db.DBNStore.from_file(path):
            tr = dbn.trade_from(rec)
            ts = float(dbn.seconds(dbn._get(rec, "ts_event")) or 0.0)
            if tr is not None:
                px = float(dbn._get(rec, "price", 0) or 0)
                b.on_trade(ts, float(tr.sz), tr.aggressor, px)
            lv = dbn._get(rec, "levels")
            if not lv:
                continue
            t0 = lv[0]
            b.on_book(ts, int(t0.bid_px), int(t0.bid_sz),
                      int(t0.ask_px), int(t0.ask_sz))
            n += 1
            if limit and n >= limit:
                return b
    return b


def show(bars, level: int) -> None:
    eps = [(i, e) for i, bar in enumerate(bars)
           for e in (bar.episodes or ())
           if e.level == level and e.rule == CALL_RULE]
    if not eps:
        print("  No calls at this level on these bars.")
        return
    print(f"  {'bar':>5} {'dir':>6} {'side':>5} {'imb':>7} {'call':>10} "
          f"{'fill':>10} {'slip':>5} {'out':>5} {'tks':>7} "
          f"{'why':>9} {'len':>5} {'eff':>5} {'tps':>6}")
    for i, e in eps:
        bar = bars[i]
        fill = f"{e.entry_px / 1e9:.2f}" if e.entered else "--"
        out = {"win": "WIN", "loss": "LOSS", "skip": "skip",
               "": "open"}[e.hit]
        print(f"  {i:>5} {bar.direction:>6} {e.side:>5} "
              f"{e.level if e.side == 'up' else -e.level:>7} "
              f"{e.px0 / 1e9:>10.2f} {fill:>10} "
              f"{e.slip:>5.1f} {out:>5} "
              f"{(e.result_ticks if e.entered else 0.0):>+7.2f} "
              f"{(e.end or '-'):>9} {e.updates:>5} "
              f"{bar.efficiency:>5.2f} {e.tps0:>6.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help=".dbn / .dbn.zst MBP files")
    ap.add_argument("--target", type=int, default=2000,
                    choices=list(LADDER))
    ap.add_argument("--level", type=int, default=500)
    ap.add_argument("--confirm", type=float, default=1.0,
                    help="ticks of confirmation before entry")
    ap.add_argument("--max-calls", type=int, default=1,
                    help="calls one candle may make; 0 for no cap")
    ap.add_argument("--no-widen", action="store_true",
                    help="take the entry without requiring the"
                         " imbalance to still be pulling away")
    ap.add_argument("--bars", type=int, default=20,
                    help="how many of the most recent bars to list")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    if a.level >= a.target:
        print(f"  A bar ends when the imbalance crosses {a.target:,}, so"
              f" it can never reach {a.level}.")
        print("  No call is possible at this pair. Use a larger target.")
        return 2

    b = load(a.files, a.target, a.level, a.confirm, a.limit,
             max_calls=a.max_calls, widen=not a.no_widen)
    bars = b.bars[-a.bars:]
    if not bars:
        print("  No bars closed. Check the files and the target.")
        return 2

    print(f"\n{'=' * 100}")
    print(f"TARGET {a.target:,}   LEVEL {a.level}   "
          f"{a.confirm:g}-tick confirmation   "
          f"{'widening required' if not a.no_widen else 'no widen gate'}"
          f"   {a.max_calls or 'no'} call"
          f"{'' if a.max_calls == 1 else 's'} a candle   "
          f"last {len(bars):,} of {len(b.bars):,} bars")
    print("=" * 100)
    show(bars, a.level)

    t = A.observation_table(bars, points=(1,), level=a.level,
                            rule=CALL_RULE)
    print(f"\n  These {len(bars)} bars: {t['won']} won, {t['failed']}"
          f" failed, {t['skipped']} never confirmed,"
          f" {t['open_at_close']} open at the bar close.")
    if t["settled"]:
        print(f"  {100 * t['win_rate']:.1f}% of the {t['settled']} that"
              f" settled · {t['avg_win_ticks']:+.2f} ticks a winner,"
              f" {t['avg_loss_ticks']:+.2f} a loser,"
              f" {t['net_per_trade']:+.2f} a trade.")
    print(f"  Median call length {t['median_updates']} updates.")
    if t["median_updates"] <= 4:
        print("\n  THAT IS TOO SHORT TO MEAN ANYTHING. A call lasting a"
              " handful of book")
        print("  updates cannot travel a point either way, so the column"
              " above is")
        print("  measuring how often the imbalance flickers across the"
              " level -- not")
        print("  whether the rule works. Raise the level, or read the"
              " longer calls")
        print("  on their own.")

    whole = A.observation_table(b.bars, points=(1,), level=a.level,
                                rule=CALL_RULE)
    print(f"\n  All {len(b.bars):,} bars: {whole['won']} won,"
          f" {whole['failed']} failed, {whole['skipped']} skipped"
          f" · {100 * whole['win_rate']:.1f}% of"
          f" {whole['settled']} settled"
          f" · {whole['net_per_trade']:+.2f} ticks a trade.")
    calls_per_bar = whole["observations"] / max(1, len(b.bars))
    print(f"  {calls_per_bar:.2f} calls a candle"
          f" ({whole['observations']:,} across {len(b.bars):,}).")
    if calls_per_bar > 1.05:
        print("  MORE THAN ONE A CANDLE means the cap is off. The same"
              " push crossing the")
        print("  level repeatedly is counted as separate trades, and on"
              " a random walk")
        print("  that is 6.41 a candle with almost all of them dead"
              " inside two updates.")
    print("\n  A random walk through this same builder returns 0.0%, so"
          " the figure to")
    print("  beat is zero rather than fifty. Costs run about 0.4 ticks a"
          " round turn")
    print("  and are not in any number above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
