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


def stability(events, window_s, grow_s, hold_s) -> dict:
    f = FlowRead(window_s=window_s, grow_s=grow_s, hold_s=hold_s)
    runs, cur, since = [], "", None
    silent = 0.0
    last_ts = events[0][0]
    for ts, e, _px in events:
        f.on_event(ts, e)
        r = f.read
        if r != cur:
            if since is not None and cur:
                runs.append(ts - since)
            if not cur and since is not None:
                silent += ts - since
            cur, since = r, ts
        last_ts = ts
    span = last_ts - events[0][0]
    live = [x for x in runs if x > 0]
    return {"window": window_s, "grow": grow_s, "hold": hold_s,
            "reads": len(live),
            "per_min": round(60.0 * len(live) / span, 2) if span else 0.0,
            "median_s": round(statistics.median(live), 1) if live else 0.0,
            "p90_s": round(sorted(live)[int(len(live) * 0.9)], 1)
            if live else 0.0,
            "longest_s": round(max(live), 1) if live else 0.0,
            "pct_silent": round(100.0 * silent / span, 1) if span else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--windows", type=float, nargs="+",
                    default=[15, 30, 60, 120])
    ap.add_argument("--holds", type=float, nargs="+",
                    default=[0, 2, 5, 10, 20])
    ap.add_argument("--grow", type=float, default=5.0)
    a = ap.parse_args()

    ev = walk_events(a.updates, a.seed)
    span = (ev[-1][0] - ev[0][0]) / 60.0
    print(f"\n{'=' * 78}")
    print(f"READ STABILITY   {len(ev):,} book updates   "
          f"{span:.0f} minutes of clock   grow {a.grow:g}s")
    print("A RANDOM WALK. Its flow has no memory, so it flips harder")
    print("than a real book: every figure here is a worst case.")
    print("=" * 78)
    print(f"  {'window':>7}{'hold':>6}{'reads/min':>11}{'median':>9}"
          f"{'p90':>8}{'longest':>9}{'silent':>8}")
    best = None
    for w in a.windows:
        for h in a.holds:
            r = stability(ev, w, a.grow, h)
            print(f"  {w:>7.0f}{h:>6.0f}{r['per_min']:>11.2f}"
                  f"{r['median_s']:>8.1f}s{r['p90_s']:>7.1f}s"
                  f"{r['longest_s']:>8.1f}s{r['pct_silent']:>7.1f}%")
            # Wanted: a read that usually survives a five-minute
            # candle, and that is not silent most of the time.
            if r["median_s"] >= 60 and r["pct_silent"] <= 70:
                if best is None or r["median_s"] > best["median_s"]:
                    best = r
    print("\n  MEDIAN is how long a published read lasts. For a"
          " five-minute trade it")
    print("  wants to be in the hundreds of seconds, not the tens --"
          " a read that")
    print("  expires a minute in puts you in a position the chart no"
          " longer holds.")
    print("  SILENT is the share of the clock saying nothing, which is"
          " the price of")
    print("  only speaking while a side is still pulling away.")
    if best:
        print(f"\n  Longest-lasting pairing that is not mostly silent:"
              f" window {best['window']:.0f}s, hold {best['hold']:.0f}s"
              f" ({best['median_s']:.0f}s median,"
              f" {best['pct_silent']:.0f}% silent).")
    else:
        print("\n  NO PAIRING HOLDS A READ FOR A MINUTE on this data."
              " On a walk that is")
        print("  the expected answer -- there is nothing to hold onto."
              " Run it against")
        print("  real files before concluding anything about the"
              " settings.")
    print("\n  This says nothing about whether the read is RIGHT."
          " scripts/read_accuracy.py")
    print("  is that question, and it has its own control.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
