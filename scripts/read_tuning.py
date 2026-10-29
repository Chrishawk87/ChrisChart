#!/usr/bin/env python3
"""How stable the read is, across every window and hold worth trying.

THE TWO NUMBERS THIS RESTS ON are the window the flow is measured
over and how long a new direction must hold before it is published.
Both were going to be invented, so they are measured instead.

What matters is not the flip rate on its own. A read that never
changes is perfectly stable and perfectly useless. The figure to
read down is HOW LONG A READ LASTS against HOW MUCH OF THE TIME IT
SAYS ANYTHING -- a reading meant to put you in a five-minute candle
has to survive most of one, and has to appear often enough to be
worth watching.

RUN ON A RANDOM WALK unless given files. The walk has no
autocorrelation in its flow, so it flips harder than a real book
does: every figure here is a WORST CASE, and a pairing that holds up
on this holds up on tape. The accuracy question is a different
script -- scripts/read_accuracy.py -- and this one cannot answer it.

Nothing here predicts, and nothing here places, sizes or times an
order.
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap.flowread import FlowRead                      # noqa: E402
from liqmap.ofi import Touch, level_ofi                   # noqa: E402

RAW = 250_000_000


def walk_events(n: int = 400_000, seed: int = 3, p_move: float = 0.12,
                dt: float = 0.01):
    """(ts, e) from a random book, through the real OFI formula."""
    rng = random.Random(seed)
    bid, ts = 7_700 * 10 ** 9, 0.0
    prev = None
    out = []
    for _ in range(n):
        if rng.random() < p_move:
            bid += RAW if rng.random() < 0.5 else -RAW
        ts += dt
        t = Touch(bid_px=bid, bid_sz=rng.randint(50, 900),
                  ask_px=bid + RAW, ask_sz=rng.randint(50, 900))
        if prev is not None:
            out.append((ts, float(level_ofi(prev, t)), bid + RAW / 2.0))
        prev = t
    return out


def stability(events, period_s, warmup_s, hold_s) -> dict:
    """How the reading behaves across whole candles.

    The figures that matter changed with the construction. A trailing
    window was asked how long a read lasts; a candle-aligned one is
    asked how much OF EACH CANDLE it spends saying something, and how
    often it changes its mind inside one. A read that settles at
    twenty seconds and holds to the close is what this is for.
    """
    f = FlowRead(period_s=period_s, warmup_s=warmup_s, hold_s=hold_s,
                 recent_s=10.0)
    per_candle: dict = {}
    cur_candle = None
    first_read_at = None
    spoke = 0.0
    last_ts = events[0][0]
    cur, since = "", None
    for ts, e, _px in events:
        f.on_event(ts, e)
        s = f.snapshot
        if s.candle_ts != cur_candle:
            if cur_candle is not None:
                if cur and since is not None:
                    spoke += ts - since
                per_candle[cur_candle] = {
                    "spoke": spoke,
                    "settled_at": first_read_at,
                    "flips": flips_at_close}
            cur_candle = s.candle_ts
            spoke, first_read_at, cur, since = 0.0, None, "", None
        flips_at_close = s.flips
        r = s.read
        if r != cur:
            if cur and since is not None:
                spoke += ts - since
            if r and first_read_at is None:
                first_read_at = s.into_s
            cur, since = r, ts
        last_ts = ts
    span = last_ts - events[0][0]
    rows = list(per_candle.values())
    if not rows:
        return {"period": period_s, "warmup": warmup_s, "hold": hold_s,
                "candles": 0, "pct_spoken": 0.0, "settled_at": 0.0,
                "flips": 0.0, "never": 100.0, "span": span}
    spoke_pct = [100.0 * r["spoke"] / period_s for r in rows]
    settled = [r["settled_at"] for r in rows
               if r["settled_at"] is not None]
    never = 100.0 * (len(rows) - len(settled)) / len(rows)
    return {"period": period_s, "warmup": warmup_s, "hold": hold_s,
            "candles": len(rows),
            "pct_spoken": statistics.mean(spoke_pct),
            "settled_at": statistics.median(settled) if settled else 0.0,
            "flips": statistics.mean(r["flips"] for r in rows),
            "never": never, "span": span}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--period", type=float, default=300.0)
    ap.add_argument("--warmups", type=float, nargs="+",
                    default=[0, 5, 10, 20, 30])
    ap.add_argument("--holds", type=float, nargs="+",
                    default=[0, 5, 10, 15, 30])
    a = ap.parse_args()

    ev = walk_events(a.updates, a.seed)
    span = (ev[-1][0] - ev[0][0]) / 60.0
    print(f"\n{'=' * 78}")
    print(f"READ STABILITY   {len(ev):,} book updates   "
          f"{span:.0f} minutes   {a.period:g}s candles")
    print("A RANDOM WALK. Its flow has no memory, so it flips harder")
    print("than a real book: every figure here is a worst case.")
    print("=" * 78)
    print(f"  {'warmup':>7}{'hold':>6}{'candles':>9}{'% spoken':>10}"
          f"{'settles at':>12}{'flips/candle':>14}{'never spoke':>13}")
    for w in a.warmups:
        for h in a.holds:
            if w + h >= a.period:
                continue
            r = stability(ev, a.period, w, h)
            print(f"  {w:>7.0f}{h:>6.0f}{r['candles']:>9,}"
                  f"{r['pct_spoken']:>9.1f}%{r['settled_at']:>11.0f}s"
                  f"{r['flips']:>14.1f}{r['never']:>12.1f}%")

    print("\n  % SPOKEN is how much of a candle the reading is saying"
          " something. A")
    print("  reading meant to put you in that candle wants to be"
          " saying it for most")
    print("  of the candle, and to have settled EARLY -- a direction"
          " that arrives at")
    print("  four minutes into five is a direction you cannot trade.")
    worst = max((stability(ev, a.period, w, h)["flips"]
                 for w in a.warmups for h in a.holds
                 if w + h < a.period), default=0.0)
    print("\n  FLIPS/CANDLE is changes of mind in the RAW reading"
          " inside one candle.")
    print("  Above one or two, the cumulative lead is crossing zero"
          " rather than")
    print("  pointing anywhere.")
    if worst > 20:
        print(f"\n  AND IT IS {worst:.0f} HERE, WHICH CHANGES WHAT THE"
              f" COLUMN BEFORE IT MEANS.")
        print("  When the raw reading flips every couple of seconds it"
              " never survives")
        print("  the hold, so the published read never changes either"
              " -- it is frozen")
        print("  on whichever side happened to be ahead when it first"
              " settled, and a")
        print("  high % SPOKEN is measuring that, not stability. On a"
              " random walk this")
        print("  is the correct and expected outcome: there is no"
              " direction to hold.")
        print("  On real tape, a flip count still in the dozens means"
              " the same thing,")
        print("  and the reading should be ignored on those candles"
              " rather than tuned.")
    print("\n  NEVER SPOKE is candles that produced no direction at"
          " all. On a walk")
    print("  that is the honest answer for a candle whose two sides"
          " finished level.")
    print("\n  This says nothing about whether the read is RIGHT."
          " scripts/read_accuracy.py")
    print("  is that question, and it has its own control.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
