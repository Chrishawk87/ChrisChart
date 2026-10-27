#!/usr/bin/env python3
"""Does the state of an OFI candle precede what the next one does?

    source .venv/bin/activate
    python scripts/bar_state_study.py data/ES_*.dbn.zst --target 250

    # prove the machinery can see an edge when there is one
    python scripts/bar_state_study.py --selftest

This is the question the chart cannot answer about itself. The candle
reports three columns -- the book that closed it, the price path that
coloured it, the delta that paid for it -- and reporting them is not
the same as them being worth anything. Trading candle by candle is
where a weak edge costs the most, because the trade count is highest,
so the burden of proof is here and not on the screen.

===========================================================================
WHAT IS MEASURED
===========================================================================
For every closed bar, its state is recorded BEFORE the next bar exists,
and the outcome is the move from this bar's close to the close `k` bars
later, in ticks. Nothing about bar i's label is allowed to come from
bar i+1 onwards; that is the only way the answer means anything.

States tested, each a partition of every bar:

    direction     green / red / chop -- which line the flow crossed
    aggression    the price path, banded at 0.70 / 0.30
    delta         the sign of paid-for flow: buy / sell / flat
    agrees        did price follow the book
    delta_agrees  did anyone PAY for what the book was doing
    combined      aggression x delta sign, which is the actual read

===========================================================================
THE CONTROL, AND WHY IT IS A SHUFFLE
===========================================================================
A state whose mean forward move is +0.4 ticks has said nothing until
you know what an arbitrary label scores on the same bars. So the labels
are SHUFFLED across the bars and the whole table recomputed, many
times, giving a null distribution of each state's mean. The reported
z is the observed mean against that distribution.

Shuffling the LABELS and not the outcomes is deliberate. Forward
returns at k > 1 overlap -- bar i and bar i+1 share most of their
future -- so the outcome series has structure that a naive standard
error would ignore and call significance. Permuting labels against the
untouched outcome series keeps that structure in both the observed and
the null, which is exactly what makes them comparable.

===========================================================================
THE PARTITION TRAP
===========================================================================
These states partition the bars: every bar is in exactly one. So the
counts sum to the total and the means are not independent -- if one
state is genuinely positive, the others are pushed negative purely by
arithmetic, and a table of four "findings" can be one finding and its
three shadows. The report says so next to every partitioned table, and
the verdict only ever names the largest |z| in a partition.

===========================================================================
SIGNIFICANCE IS CHEAP AT THIS SAMPLE SIZE. EFFECT SIZE IS NOT.
===========================================================================
Fifty-five million book updates make a hundred thousand bars, and with
n that large a z of 3 costs almost nothing to produce. Run on a pure
random walk, this study's own aggression table returns |z| = 3.49 --
on a mean of 0.0016 ticks, which is two cents against roughly five
dollars of round-trip cost. That is a real statistical difference and
a completely worthless one.

So a finding has to clear TWO gates, not one: the z against its own
shuffled control, AND a mean that beats the cost of taking it. The
default floor is 0.4 ticks, which is about what fees plus a tick of
slippage come to on ES. A table row that clears the first gate and not
the second is noted and not called a finding.

===========================================================================
THE SELFTEST
===========================================================================
A null result is only worth believing from a method that can detect a
real effect. `--selftest` builds a series where a known fraction of the
labels genuinely precede the move, runs the identical pipeline, and
fails loudly if it does not find it. A study that cannot do this is
not evidence of absence; it is evidence of nothing.

Nothing here places, sizes or times an order.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TICK = 0.25
# Feed prices are raw integer units; one ES tick is this many of them.
RAW_TICK = 250_000_000

SHUFFLES = 400

# A state seen fewer times than this is not reported as a finding. With
# 60 observations the mean of a noisy series is mostly noise, and a
# table of tiny cells is where a scan finds whatever it is looking for.
MIN_N = 200


# --------------------------------------------------------------- loading

def iter_records(paths: list[str]):
    """Book touches and trades, in order, from the MBP files."""
    import databento as db

    from liqmap import dbn

    for path in paths:
        store = db.DBNStore.from_file(path)
        for rec in store:
            t = dbn.trade_from(rec)
            lv = dbn._get(rec, "levels")
            ts = float(dbn.seconds(dbn._get(rec, "ts_event")) or 0.0)
            if t is not None:
                yield ("trade", ts, float(t.sz), t.aggressor)
            if lv:
                l0 = lv[0]
                yield ("book", ts,
                       int(dbn._get(l0, "bid_px", 0) or 0),
                       int(dbn._get(l0, "bid_sz", 0) or 0),
                       int(dbn._get(l0, "ask_px", 0) or 0),
                       int(dbn._get(l0, "ask_sz", 0) or 0))


def build(paths: list[str], target: int, limit: int = 0):
    from liqmap.ofibar import OfiBarBuilder

    b = OfiBarBuilder(tick_size=TICK, threshold=target, max_s=900.0,
                      min_updates=2, max_bars=10_000_000)
    n = 0
    for ev in iter_records(paths):
        if ev[0] == "trade":
            b.on_trade(ev[1], ev[2], ev[3])
        else:
            b.on_book(ev[1], ev[2], ev[3], ev[4], ev[5])
        n += 1
        if limit and n >= limit:
            break
        if n % 5_000_000 == 0:
            print(f"  {n:,} records, {len(b.bars):,} bars", flush=True)
    return b.bars


# ---------------------------------------------------------------- states

def states_of(bar) -> dict[str, str]:
    """Every label for one bar, read only from that bar."""
    d = bar.delta
    return {
        "direction": bar.direction,
        "aggression": bar.aggression,
        "delta": "buy" if d > 0 else ("sell" if d < 0 else "flat"),
        "agrees": "yes" if bar.agrees else "no",
        "delta_agrees": "yes" if bar.delta_agrees else "no",
        "combined": f"{bar.aggression.split('_')[0]}/"
                    f"{'buy' if d > 0 else ('sell' if d < 0 else 'flat')}",
    }


def outcomes(bars, k: int) -> list[float]:
    """Move from each bar's close to the close k bars later, in ticks.

    The last k bars have no outcome and are dropped rather than given a
    zero -- a zero would sum like a measurement and pull every mean
    toward nothing.
    """
    out = []
    for i in range(len(bars) - k):
        out.append((bars[i + k].close - bars[i].close) / RAW_TICK)
    return out


# ----------------------------------------------------------- the measure

def table(labels: list[str], moves: list[float], rng: random.Random,
          shuffles: int = SHUFFLES) -> dict:
    """Mean move per label, and each one's z against shuffled labels."""
    by = defaultdict(list)
    for lab, mv in zip(labels, moves):
        by[lab].append(mv)

    base = sum(moves) / len(moves) if moves else 0.0
    obs = {k: (len(v), sum(v) / len(v)) for k, v in by.items()}

    # The null: the same bars, the same outcomes, the labels moved.
    null = defaultdict(list)
    pool = list(labels)
    for _ in range(shuffles):
        rng.shuffle(pool)
        acc = defaultdict(float)
        cnt = defaultdict(int)
        for lab, mv in zip(pool, moves):
            acc[lab] += mv
            cnt[lab] += 1
        for k in obs:
            null[k].append(acc[k] / cnt[k] if cnt[k] else 0.0)

    rows = []
    for k, (n, mean) in sorted(obs.items(), key=lambda x: -x[1][0]):
        xs = null[k]
        mu = sum(xs) / len(xs)
        var = sum((x - mu) ** 2 for x in xs) / max(1, len(xs) - 1)
        sd = var ** 0.5
        z = (mean - mu) / sd if sd > 0 else 0.0
        rows.append({"label": k, "n": n, "mean": mean, "null_sd": sd,
                     "z": z, "enough": n >= MIN_N})
    return {"base": base, "rows": rows, "total": len(moves)}


def report(name: str, res: dict, k: int, cost: float) -> tuple[float, float]:
    """Print one partitioned table.

    Returns (largest |z|, the edge that row carried) where edge is the
    distance of the mean from the base rate, in ticks. Both are needed
    to say anything: see the note on significance above.
    """
    print(f"\n  {name}  (forward {k} bar{'s' if k != 1 else ''})")
    print(f"    base rate across all {res['total']:,} bars: "
          f"{res['base']:+.4f} ticks")
    print(f"    {'state':<26}{'n':>9}{'mean':>10}{'edge':>10}"
          f"{'z':>9}{'   verdict'}")
    best_z, best_edge = 0.0, 0.0
    for r in res["rows"]:
        edge = abs(r["mean"] - res["base"])
        if not r["enough"]:
            verdict = "too few to read"
        elif abs(r["z"]) < 3.0:
            verdict = "noise"
        elif edge < cost:
            verdict = f"below cost ({cost:g}t)"
        else:
            verdict = "WORTH A LOOK"
        print(f"    {r['label']:<26}{r['n']:>9,}{r['mean']:>+10.4f}"
              f"{edge:>10.4f}{r['z']:>+9.2f}   {verdict}")
        if r["enough"] and abs(r["z"]) > best_z:
            best_z, best_edge = abs(r["z"]), edge
    print("    these states partition the bars, so the counts sum to the"
          " total and one")
    print("    real effect pushes every other cell the other way --"
          " read the largest")
    print("    |z| only, not the row of them.")
    return best_z, best_edge


# -------------------------------------------------------------- selftest

def selftest(rng: random.Random) -> int:
    """A series with a known edge, through the identical pipeline.

    A null result from a method that cannot detect a real effect is not
    evidence of absence. This fails loudly rather than reassuring.
    """
    n = 20_000
    moves = [rng.gauss(0, 2.0) for _ in range(n)]
    labels = []
    for mv in moves:
        # 58% of the time the label knows the sign. That is a large edge
        # by any standard and the method must see it easily.
        if rng.random() < 0.58:
            labels.append("up" if mv > 0 else "dn")
        else:
            labels.append("up" if rng.random() < 0.5 else "dn")
    res = table(labels, moves, rng, shuffles=200)
    best = max(abs(r["z"]) for r in res["rows"])
    print(f"  planted edge: largest |z| = {best:.1f}")

    flat = ["a" if rng.random() < 0.5 else "b" for _ in range(n)]
    res2 = table(flat, moves, rng, shuffles=200)
    worst = max(abs(r["z"]) for r in res2["rows"])
    print(f"  no edge:      largest |z| = {worst:.1f}")

    ok = best > 6.0 and worst < 4.0
    print("  SELFTEST", "PASS" if ok else "FAIL")
    if not ok:
        print("  the method cannot tell a planted edge from noise, so a"
              " null result")
        print("  from it would mean nothing. Do not trust the tables.")
    return 0 if ok else 1


# ------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help=".dbn / .dbn.zst MBP files")
    ap.add_argument("--target", type=int, default=250)
    ap.add_argument("--forward", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--shuffles", type=int, default=SHUFFLES)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many records (a quick look)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cost", type=float, default=0.4,
                    help="round-trip cost in ticks a finding must beat")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    if a.selftest:
        print("\nSELFTEST -- can this method see an edge that is there?")
        return selftest(rng)

    if not a.files:
        ap.error("give me some .dbn files, or --selftest")

    print(f"\nBAR STATE STUDY  target {a.target:,}")
    print("  building bars...", flush=True)
    bars = build(a.files, a.target, a.limit)
    print(f"  {len(bars):,} closed bars")
    if len(bars) < 2 * MIN_N:
        print("  too few bars to read anything. Use a smaller target or"
              " more data.")
        return 1

    traded = sum(1 for b in bars if b.trade_count > 0)
    print(f"  {traded:,} of them contain at least one print "
          f"({100.0 * traded / len(bars):.1f}%)")
    if traded < len(bars) * 0.2:
        print("  NOTE: most bars have no prints, so the delta columns"
              " below are")
        print("  mostly the 'flat' bucket and say little.")

    findings = []
    for k in a.forward:
        mv = outcomes(bars, k)
        usable = bars[:len(mv)]
        print(f"\n{'=' * 70}\nFORWARD {k} BAR{'S' if k != 1 else ''}"
              f"  ({len(mv):,} measurable bars)\n{'=' * 70}")
        for name in ("direction", "aggression", "delta", "agrees",
                     "delta_agrees", "combined"):
            labels = [states_of(b)[name] for b in usable]
            res = table(labels, mv, rng, a.shuffles)
            bz, be = report(name, res, k, a.cost)
            findings.append((name, k, bz, be))

    print(f"\n{'=' * 70}\nWHAT THIS SAYS\n{'=' * 70}")
    sig = [f for f in findings if f[2] >= 3.0]
    real = [f for f in sig if f[3] >= a.cost]
    if not sig:
        print("  Nothing reaches |z| = 3 against its own shuffled control.")
    elif not real:
        print(f"  {len(sig)} table(s) clear |z| = 3 and NONE of them clear"
              f" {a.cost:g} ticks")
        print("  of cost. At this sample size a z of 3 is nearly free --"
              " on a pure")
        print("  random walk this same study returns |z| = 3.49 on a mean"
              " of 0.0016")
        print("  ticks. Statistically different, economically nothing.")
        for name, k, z, e in sorted(sig, key=lambda x: -x[2])[:4]:
            print(f"    {name} at forward {k}: |z| = {z:.2f}, "
                  f"edge {e:.4f} ticks")
    if not real:
        print("\n  On this data, at this target, the state of a candle"
              " does not")
        print("  precede the next one by enough to trade candle by candle.")
        print("  That is a result, not a failure -- it is the answer to"
              " the question,")
        print("  and it was cheaper to get here than to find out by"
              " trading it.")
        return 0

    for name, k, z, e in sorted(real, key=lambda x: -x[3]):
        print(f"  {name} at forward {k}: |z| = {z:.2f}, "
              f"edge {e:.4f} ticks")
    print("\n  Both gates cleared, which is as far as a study of ONE"
          " sample can")
    print("  take it. Before sizing anything: break_even ="
          " (stop + cost) /")
    print("  (target + stop), and re-run on data this study has not seen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
