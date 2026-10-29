#!/usr/bin/env python3
"""When the read says UP, does price go up over the next five minutes?

THE TRADE IS A FIVE-MINUTE CANDLE, so the measurement is in MINUTES.
Every other measurement in this package counts in OFI bars, which is
right for a scalp on those bars and wrong here: an OFI bar is a
quantity of flow, not a quantity of clock, and a rule read off it to
trade a time candle has to be scored against the clock.

WHAT IS COUNTED

A reading is taken each time the published read CHANGES to up or
down -- the moment it became actionable. Price at that instant is the
entry; price at +1, +3 and +5 minutes is the outcome. One row per
reading, not one per update: scoring every update would count a
single two-minute UP read a thousand times and report its sample size
as a thousand.

UP, FLAT AND DOWN ARE THREE OUTCOMES. On a one-tick market over a
short horizon a real share of moves are exactly zero, and folding
flats into either column is how a 50% rule reads as 33% or 67%
depending which way you fold. Both percentages are printed.

THE CONTROL IS THE SAME READ, BACKWARDS. A rule is only worth
something to the extent it beats itself flipped on the same tape --
that difference is the only part a market that does nothing cannot
produce. On a random walk the two are equal by construction, and
this script says so rather than printing a number that looks like an
edge.

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

from liqmap.flowread import (DOWN, FlowRead, GROW_S,          # noqa: E402
                             HOLD_S, UP, WINDOW_S)
from liqmap.ofi import Touch, level_ofi                       # noqa: E402

RAW_TICK = 250_000_000.0
MINUTES = (1.0, 3.0, 5.0)


def walk(n: int, seed: int, p_move: float = 0.12, dt: float = 0.01):
    """(ts, e, mid) from a random book, through the real OFI formula."""
    rng = random.Random(seed)
    bid, ts, prev = 7_700 * 10 ** 9, 0.0, None
    out = []
    for _ in range(n):
        if rng.random() < p_move:
            bid += RAW_TICK if rng.random() < 0.5 else -RAW_TICK
        ts += dt
        t = Touch(bid_px=bid, bid_sz=rng.randint(50, 900),
                  ask_px=bid + RAW_TICK, ask_sz=rng.randint(50, 900))
        if prev is not None:
            out.append((ts, float(level_ofi(prev, t)),
                        bid + RAW_TICK / 2.0))
        prev = t
    return out


def from_files(paths, limit: int = 0):
    import databento as db

    from liqmap import dbn

    out, prev, n = [], None, 0
    for path in paths:
        for rec in db.DBNStore.from_file(path):
            lv = dbn._get(rec, "levels")
            if not lv:
                continue
            t0 = lv[0]
            t = Touch(bid_px=int(t0.bid_px), bid_sz=int(t0.bid_sz),
                      ask_px=int(t0.ask_px), ask_sz=int(t0.ask_sz))
            if not t.valid or t.crossed:
                continue
            ts = float(dbn.seconds(dbn._get(rec, "ts_event")) or 0.0)
            if prev is not None:
                out.append((ts, float(level_ofi(prev, t)),
                            (t.bid_px + t.ask_px) / 2.0))
            prev = t
            n += 1
            if limit and n >= limit:
                return out
    return out


def readings(events, window_s, grow_s, hold_s):
    """One row each time the published read CHANGES to a direction."""
    f = FlowRead(window_s=window_s, grow_s=grow_s, hold_s=hold_s)
    out, cur = [], ""
    for ts, e, px in events:
        f.on_event(ts, e)
        r = f.read
        if r != cur:
            if r in (UP, DOWN):
                s = f.snapshot
                out.append({"ts": ts, "side": r, "px": px,
                            "lead": s.lead, "gross": s.gross,
                            "share": s.share, "flips": s.flips_1m})
            cur = r
    return out


def outcomes(rows, events, minutes=MINUTES, flip: bool = False):
    """Ticks moved the reading's way at each horizon."""
    times = [e[0] for e in events]
    pxs = [e[2] for e in events]
    import bisect
    cols = {m: [] for m in minutes}
    for r in rows:
        side = r["side"]
        if flip:
            side = DOWN if side == UP else UP
        for m in minutes:
            j = bisect.bisect_left(times, r["ts"] + m * 60.0)
            if j >= len(times):
                continue
            d = (pxs[j] - r["px"]) / RAW_TICK
            cols[m].append(d if side == UP else -d)
    return cols


def table(cols, label: str) -> list[dict]:
    rows = []
    for m, xs in sorted(cols.items()):
        if not xs:
            rows.append({"min": m, "n": 0})
            continue
        right = sum(1 for x in xs if x > 0)
        wrong = sum(1 for x in xs if x < 0)
        flat = len(xs) - right - wrong
        dec = right + wrong
        rows.append({"min": m, "n": len(xs), "right": right,
                     "flat": flat, "wrong": wrong, "decided": dec,
                     "pct_all": right / len(xs),
                     "pct_dec": (right / dec) if dec else 0.0,
                     "mean": statistics.mean(xs),
                     "median": statistics.median(xs),
                     "label": label})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*",
                    help=".dbn / .dbn.zst MBP files; omit for a walk")
    ap.add_argument("--window", type=float, default=WINDOW_S)
    ap.add_argument("--short", type=float, default=GROW_S)
    ap.add_argument("--hold", type=float, default=HOLD_S)
    ap.add_argument("--minutes", type=float, nargs="+",
                    default=list(MINUTES))
    ap.add_argument("--updates", type=int, default=400_000,
                    help="walk length when no files are given")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    if a.files:
        ev = from_files(a.files, a.limit)
        src = f"{len(a.files)} file" + ("" if len(a.files) == 1 else "s")
    else:
        ev = walk(a.updates, a.seed)
        src = "A RANDOM WALK -- there is nothing in it to find"
    if len(ev) < 100:
        print("  Not enough book updates to measure anything.")
        return 2

    span = (ev[-1][0] - ev[0][0]) / 60.0
    rows = readings(ev, a.window, a.short, a.hold)
    print(f"\n{'=' * 86}")
    print(f"READ ACCURACY AGAINST THE CLOCK   {src}")
    print(f"{len(ev):,} book updates   {span:.0f} minutes   "
          f"long {a.window:g}s / short {a.short:g}s / hold {a.hold:g}s")
    print("=" * 86)
    if not rows:
        print("  The read never settled on a direction. Nothing to"
              " score.")
        return 2
    ups = sum(1 for r in rows if r["side"] == UP)
    print(f"  {len(rows):,} readings ({ups:,} up, {len(rows) - ups:,}"
          f" down), one every {span / len(rows):.1f} minutes.")

    real = table(outcomes(rows, ev, a.minutes), "read")
    ctl = table(outcomes(rows, ev, a.minutes, flip=True), "flipped")
    print(f"\n  {'after':>7}{'n':>8}{'right':>8}{'flat':>7}{'wrong':>8}"
          f"{'of all':>9}{'of decided':>12}{'mean tks':>10}"
          f"{'flipped':>9}{'EDGE':>8}")
    for r, c in zip(real, ctl):
        if not r.get("n"):
            print(f"  {r['min']:>6.0f}m{0:>8}   -- no reading had that"
                  f" much clock left after it")
            continue
        print(f"  {r['min']:>6.0f}m{r['n']:>8,}{r['right']:>8,}"
              f"{r['flat']:>7,}{r['wrong']:>8,}"
              f"{100 * r['pct_all']:>8.1f}%{100 * r['pct_dec']:>11.1f}%"
              f"{r['mean']:>+10.2f}{100 * c['pct_dec']:>8.1f}%"
              f"{100 * (r['pct_dec'] - c['pct_dec']):>+7.1f}%")

    best = max((r for r in real if r.get("n")),
               key=lambda r: r["pct_dec"], default=None)
    print("\n  OF ALL counts flats against the read; OF DECIDED sets"
          " them aside. Both")
    print("  are true and they answer different questions -- quoting"
          " one without the")
    print("  other is where this kind of measurement usually goes"
          " wrong.")
    n_max = max((r.get("n", 0) for r in real), default=0)
    if n_max < 200:
        need = 200 * (span / len(rows)) / 60.0 if rows else 0
        print(f"\n  ONLY {n_max} READINGS. THAT IS NOT A SAMPLE, and no"
              f" percentage above")
        print("  means anything yet. This read settles about once every"
              f" {span / len(rows):.0f}")
        print(f"  minutes by design, so two hundred of them is roughly"
              f" {need:.0f} hours of")
        print("  tape -- days, not a session. Until then the columns"
              " below are a")
        print("  demonstration that the machinery runs, not a result.")
    print("\n  EDGE is the read against the SAME READ RUN BACKWARDS on"
          " the same tape.")
    print("  That difference is the only part a market with nothing"
          " in it cannot")
    print("  produce, and it is the column to read. A 60% that beats"
          " its own flip")
    print("  by one point is a 60% worth nothing.")
    if not a.files:
        print("\n  THIS WAS A RANDOM WALK. Any edge printed above is"
              " sampling noise,")
        print("  and that is the point of running it: it is what zero"
              " looks like on")
        print("  this many readings. Run it on real files and compare.")
    elif best:
        print(f"\n  Best horizon: {best['min']:.0f} minutes at"
              f" {100 * best['pct_dec']:.1f}% of decided"
              f" ({best['decided']:,} decided).")
        print("  Before reading anything into it: costs run about 0.4"
              " ticks a round")
        print("  turn and are in none of these numbers, and an edge"
              " that does not")
        print("  survive on days this did not see was a property of"
              " these days.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
