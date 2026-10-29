#!/usr/bin/env python3
"""Which imbalance entry level works better? One pass, every level.

    source .venv/bin/activate
    python scripts/imbalance_sweep.py data/ES_*.dbn.zst --target 2000

    # a quick look before committing to the whole file
    python scripts/imbalance_sweep.py data/ES_*.dbn.zst --limit 5000000

    # compare the targets too
    python scripts/imbalance_sweep.py data/ES_*.dbn.zst --all-targets

THE RULE

    enter   the running imbalance reaches +level (buy) or -level (sell)
    exit    the first of: 3 points, the opposite level, or the bar close

A bar may hold several trades. Nothing survives the bar. After a trade
exits, the same side cannot be taken again until the imbalance has come
back inside the band -- without that a trade that exits at target with
the imbalance unchanged re-enters on the very next update, and one move
is counted as a dozen winners.

TWO EXIT RULES, BOTH MEASURED

    HOLD  while a trade is open, the imbalance falling back inside the
          band does nothing; it runs to the target, the opposite level,
          or the bar close
    BAND  coming back through the level flattens it as well

They answer the same question differently and they change which level
looks best, so neither is assumed.

READ THE CONTROL COLUMN FIRST
---------------------------------------------------------------------
Every level also runs with the side flipped -- buy where the rule
sells, sell where it buys. Same entries, same exits, same bars. A win
rate on its own cannot tell a signal from the shape of a bar, because
most of what drives it is how long a bar runs and how far price
travels inside one, and that is identical for both.

    rule 45%, flipped 55%   the signal points the wrong way
    both near 50%           there is nothing in it
    rule 58%, flipped 42%   something is there

EDGE is the gap. It is the column that matters, and it is the only one
that cannot be produced by a market that does nothing.

This is not a backtest. No fills, no costs, no size, no risk. It
answers whether price travelled three points the called way before the
bar ended.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from liqmap.imbsim import (BAND, HOLD, LEVELS, Sweep,      # noqa: E402
                           bands)
from liqmap.ofibar import LADDER, OfiBarBuilder           # noqa: E402

TICK = 0.25


def run(paths: list[str], targets: list[int], levels, target_points: float,
        limit: int = 0, min_share: float = 0.0,
        confirms: tuple = (0.0,)):
    """One pass over the records, every target and level at once."""
    import databento as db

    from liqmap import dbn

    builders = {t: OfiBarBuilder(tick_size=TICK, threshold=t, max_s=900.0,
                                 min_updates=2, max_bars=10_000_000)
                for t in targets}
    sweeps = {t: Sweep(levels=tuple(levels), target_points=target_points,
                       min_share=min_share, confirms=tuple(confirms))
              for t in targets}
    n = 0
    for path in paths:
        store = db.DBNStore.from_file(path)
        for rec in store:
            tr = dbn.trade_from(rec)
            ts = float(dbn.seconds(dbn._get(rec, "ts_event")) or 0.0)
            if tr is not None:
                px = float(dbn._get(rec, "price", 0) or 0)
                for b in builders.values():
                    b.on_trade(ts, float(tr.sz), tr.aggressor, px)
            lv = dbn._get(rec, "levels")
            if not lv:
                continue
            l0 = lv[0]
            bpx = int(dbn._get(l0, "bid_px", 0) or 0)
            apx = int(dbn._get(l0, "ask_px", 0) or 0)
            bsz = int(dbn._get(l0, "bid_sz", 0) or 0)
            asz = int(dbn._get(l0, "ask_sz", 0) or 0)
            if bpx <= 0 or apx <= 0:
                continue
            mid = (bpx + apx) / 2.0
            for t, b in builders.items():
                closed = b.on_book(ts, bpx, bsz, apx, asz)
                sw = sweeps[t]
                pu, pd = b.live_push
                # The close first, so the trade is flattened at the
                # bar's own closing price rather than at whatever the
                # next bar opens on.
                if closed is not None:
                    sw.bar_close(closed.ts_close, closed.close)
                sw.tick(ts, b.accumulated, mid, b.live_share, pu + pd)
            n += 1
            if limit and n >= limit:
                break
            if n % 5_000_000 == 0:
                print(f"  {n:,} book updates...", flush=True)
        if limit and n >= limit:
            break
    print(f"  {n:,} book updates read")
    return builders, sweeps


POINT = 1_000_000_000


def bar_shape(bars) -> dict:
    """How big a bar is, so an empty table can be explained.

    A three-point objective inside a bar that spans one point is not a
    rule that failed; it is a distance the bar cannot contain. The
    table would be all zeros and say nothing about the signal.
    """
    if not bars:
        return {}
    rng = sorted((b.high - b.low) / POINT for b in bars)
    secs = sorted(b.seconds for b in bars)
    n = len(rng)
    return {"median_range": rng[n // 2], "mean_range": sum(rng) / n,
            "p90_range": rng[int(n * 0.9)],
            "median_seconds": secs[n // 2]}


def show(target: int, bars_list, sweep: Sweep, target_points: float) -> None:
    bars = len(bars_list)
    print(f"\n{'=' * 92}")
    print(f"TARGET {target:,}   {bars:,} bars   "
          f"{target_points:g}-point objective")
    print("=" * 92)
    shape = bar_shape(bars_list)
    if shape:
        print(f"  A bar here spans {shape['median_range']:.2f} points at"
              f" the median, {shape['p90_range']:.2f} at the 90th,"
              f" and runs {shape['median_seconds']:.0f}s.")
        if shape["p90_range"] < target_points:
            print(f"  NINE BARS IN TEN NEVER SPAN {target_points:g} POINTS."
                  f" The objective cannot be")
            print("  reached inside one, so the table below is zeros for a"
                  " reason that is")
            print("  nothing to do with the signal. Use a larger target or"
                  " a smaller")
            print("  objective.")
        elif shape["median_range"] < target_points:
            print(f"  The median bar does not span {target_points:g}"
                  f" points, so most trades will")
            print("  close at the bar rather than at the objective. That is"
                  " the arithmetic")
            print("  of the bar size, not a verdict on the rule.")
    print(f"  {'level':>6}{'rule':>6}{'cfm':>5}{'trades':>9}{'skipped':>9}"
          f"{'/bar':>6}{'hit':>8}"
          f"{'flipped':>9}{'EDGE':>8}{'mean pt':>9}{'flip pt':>9}"
          f"{'EDGE pt':>9}{'upd':>6}")
    rows = sweep.rows()
    for r in rows:
        print(f"  {r['level']:>6}{r['policy']:>6}{r.get('confirm', 0):>5.0f}"
              f"{r['trades']:>9,}{r.get('skipped', 0):>9,}"
              f"{r['trades_per_bar']:>6.2f}"
              f"{100 * r['win_rate']:>7.1f}%"
              f"{100 * r['control_win_rate']:>8.1f}%"
              f"{100 * r['edge']:>+7.1f}%"
              f"{r['mean_points']:>+9.3f}{r['control_mean_points']:>+9.3f}"
              f"{r['edge_points']:>+9.3f}{r['mean_updates']:>6.0f}")

    usable = [r for r in rows if r["trades"] >= 200]
    if not usable:
        print("\n  Too few trades at every level to read anything.")
        return
    best = max(usable, key=lambda r: r["edge"])
    print(f"\n  Largest edge: level {best['level']} on the"
          f" {best['policy'].upper()} rule, "
          f"{100 * best['edge']:+.1f}% against its own flip"
          f" ({best['trades']:,} trades).")
    if best["edge"] <= 0.02:
        print("  WHICH IS NOTHING. No level beats its own flip by more than"
              " two points,")
        print("  and a market with nothing in it produces exactly that. The"
              " entry level")
        print("  is not the thing standing between this and working.")
    else:
        print("  Before reading anything into it: check the trade count,"
              " then run the")
        print("  same thing on data this sweep has not seen. An edge that"
              " does not")
        print("  survive that was a property of these particular days.")
    print("\n  CFM is ticks of confirmation waited before entry, and"
          " SKIPPED is how")
    print("  many calls died before price ever confirmed. Those were"
          " never taken, so")
    print("  they are in neither column -- but a level whose skips dwarf"
          " its trades is")
    print("  a level the confirmation has switched off, whatever its win"
          " rate says.")
    print("\n  HIT counts only trades that reached the objective."
          " Everything else")
    print("  closed at the opposite level or at the bar, wherever price"
          " happened to")
    print("  be. MEAN PT includes those, which is why it is the more"
          " honest column.")


def show_bands(sweep: Sweep, level: int, policy: str,
               target_points: float) -> None:
    """Did one-sidedness separate the winners from the losers?

    The net cannot tell a one-sided push from a brawl that happened to
    tilt, and the brawl is where a trade gets taken just before price
    turns over. If that is true here, the clear rate rises across these
    bands. If it is flat, the two sides were not hiding anything a
    filter could use.
    """
    tr = sweep.sims[(level, policy, False)].trades
    if not tr:
        print(f"\n  No trades at level {level} to break out.")
        return
    print(f"\n  ONE-SIDEDNESS AT ENTRY  (level {level}, {policy.upper()},"
          f" {target_points:g} point)")
    print(f"  {'band':>6}{'trades':>9}{'cleared':>9}{'clear%':>9}"
          f"{'mean pt':>9}{'got to':>9}{'near':>7}   why they did not clear")
    rows = bands(tr)
    for r in rows:
        why = ", ".join(f"{k} {v:,}" for k, v in sorted(r["why"].items())
                        if k != "target")
        print(f"  {r['band']:>6.2f}{r['trades']:>9,}{r['cleared']:>9,}"
              f"{100 * r['clear_rate']:>8.1f}%{r['mean_points']:>+9.3f}"
              f"{r['median_missed_mfe']:>9.2f}{r['missed_near']:>7,}   {why}")
    print("  BAND is how much of the bar's gross flow was on the side"
          " taken: 0.50 is an")
    print("  even fight, 0.70 is one-way traffic. GOT TO is the median"
          " distance the")
    print("  ones that did not clear actually reached -- a failure at"
          " nine tenths is")
    print("  not the same animal as one that never left the entry. NEAR"
          " counts the")
    print("  misses that got at least half a point.")
    usable = [r for r in rows if r["trades"] >= 200]
    if len(usable) >= 2:
        lo, hi = usable[0], usable[-1]
        gap = hi["clear_rate"] - lo["clear_rate"]
        print(f"\n  Across the bands the clear rate moves"
              f" {100 * lo['clear_rate']:.1f}% -> {100 * hi['clear_rate']:.1f}%"
              f" ({100 * gap:+.1f} points).")
        if abs(gap) < 0.03:
            print("  WHICH IS FLAT. One-sidedness is not separating these"
                  " trades, so a")
            print("  filter on it would thin the sample without improving"
                  " it. The two")
            print("  sides were not hiding anything the net was not"
                  " already saying.")
        else:
            print("  Worth a filter. Re-run with --min-share set to the"
                  " band where it")
            print("  turns, then check the trade count has not collapsed"
                  " with it.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help=".dbn / .dbn.zst MBP files")
    ap.add_argument("--target", type=int, default=2000)
    ap.add_argument("--all-targets", action="store_true")
    ap.add_argument("--levels", type=int, nargs="+", default=list(LEVELS),
                    help="imbalance entry levels to compare")
    ap.add_argument("--points", type=float, default=1.0,
                    help="objective in ES points")
    ap.add_argument("--band-level", type=int, default=0,
                    help="entry level to break out by one-sidedness;"
                         " defaults to the one with the largest edge")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--confirm", type=float, nargs="+", default=[0.0, 1.0,
                                                               2.0],
                    help="ticks of confirmation to wait before entry;"
                         " several values fan out over the same tape")
    ap.add_argument("--min-share", type=float, default=0.0,
                    help="require this share of the bar's gross flow on"
                         " the side taken; 0 takes every signal")
    ap.add_argument("--csv", default="",
                    help="write every trade here; defaults to"
                         " imbalance_trades_<target>.csv beside the data")
    a = ap.parse_args()

    targets = list(LADDER) if a.all_targets else [a.target]
    levels = sorted({int(x) for x in a.levels if x > 0})
    too_big = [(t, lv) for t in targets for lv in levels if lv >= t]
    if too_big:
        print("\n  NOTE: a bar ends when the imbalance crosses its target,"
              " so the")
        print("  accumulator never travels past it. These pairs can never"
              " fire and")
        print("  will show zero trades:")
        for t, lv in too_big:
            print(f"    target {t:,} with level {lv}")

    print(f"\nIMBALANCE SWEEP   targets {targets}   levels {levels}")
    print("  building...", flush=True)
    builders, sweeps = run(a.files, targets, levels, a.points, a.limit,
                           a.min_share,
                           tuple(sorted({float(x) for x in a.confirm
                                         if x >= 0})) or (0.0,))

    base = Path(a.files[0]).resolve().parent
    for t in targets:
        show(t, builders[t].bars, sweeps[t], a.points)
        rws = [r for r in sweeps[t].rows() if r["trades"] >= 200]
        pick = (a.band_level if a.band_level else
                (max(rws, key=lambda r: r["edge"])["level"] if rws
                 else levels[0]))
        show_bands(sweeps[t], pick, HOLD, a.points)
        rows = []
        for pol in (HOLD, BAND):
            for tr in sweeps[t].trades(pol):
                d = tr.as_dict()
                d["policy"] = pol
                d["target"] = t
                rows.append(d)
        if not rows:
            continue
        out = (Path(a.csv) if a.csv and len(targets) == 1
               else base / f"imbalance_trades_{t}.csv")
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  {len(rows):,} trades written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
