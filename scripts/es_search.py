#!/usr/bin/env python3
"""Search the training half. Promote exactly one candidate. Test it once.

    python scripts/es_search.py TRAIN.slices.jsonl [VALIDATE.slices.jsonl]

    --take 5 --stop 3 --bars 5        the triple barrier
    --promote "<name>"                skip the search, test a named rule

WHAT THIS DOES, AND WHY IT IS SHAPED LIKE THIS

Step 1 and 2 -- segment into regimes, build derived features -- generate a
lot of candidates. Six regimes times four features times two directions is
already dozens of looks, and with that many, several will clear any bar you
set. That is not a flaw in the search; it is what a search is. It is a flaw
only if the winner is then reported as a result.

So the search half reports NOTHING as evidence. It picks one candidate, by
a rule stated before looking: highest dollars per trade, subject to a
minimum number of resolved trades. That one candidate is then run, once, on
the validation file -- and if no second file is given, this refuses to
report a validation number at all rather than quietly scoring the training
half twice.

Step 3's triple barrier is the target throughout: does +5 arrive before -3
within N bars. Not the candle's close, which is not a trade anybody makes.

Step 5 is the part no script can enforce. If the validation number is flat,
the honest move is to stop. Coming back with a different feature and the
same validation file converts it into training, one test at a time, and the
conversion is invisible -- the file does not change, only what it is worth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap import barrier as bar, escondense as ec, features as ft  # noqa: E402

# A candidate needs at least this many resolved trades to be promotable.
# Below it the estimate is noise wearing a decimal point.
MIN_RESOLVED = 120

# Feature thresholds to try, as quantiles of the training distribution.
CUTS = (0.10, 0.20, 0.30)


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("train")
    p.add_argument("validate", nargs="?")
    p.add_argument("--take", type=float, default=5.0)
    p.add_argument("--stop", type=float, default=3.0)
    p.add_argument("--bars", type=int, default=5)
    p.add_argument("--tick", type=float, default=0.25)
    p.add_argument("--promote", default=None)
    return p.parse_args(argv)


def quantile(xs, q):
    v = sorted(x for x in xs if x is not None)
    if not v:
        return None
    return v[min(len(v) - 1, int(q * len(v)))]


def candidates(slices, feats, tick):
    """Every (name, sides) pair the search will consider.

    `sides[i]` is +1 / -1 / 0 for bar i. A candidate is a feature, a
    threshold, a direction convention, and optionally a regime filter.
    """
    out: list[tuple[str, list[int]]] = []
    n = len(slices)

    for name in ("imbalance", "absorption", "anomaly"):
        vals = [getattr(f, name, None) for f in feats]
        real = [v for v in vals if v is not None]
        if len(real) < 200:
            continue
        for q in CUTS:
            lo = quantile(real, q)
            hi = quantile(real, 1.0 - q)
            if lo is None or hi is None or lo >= hi:
                continue
            for sign, label in ((1, "with"), (-1, "against")):
                sides = [0] * n
                for i, v in enumerate(vals):
                    if v is None:
                        continue
                    if v >= hi:
                        sides[i] = sign
                    elif v <= lo:
                        sides[i] = -sign
                out.append((f"{name} {label} sign, top/bottom "
                            f"{int(q*100)}%", sides))

    # The conditional form: the same reads, but only at a key level.
    for base_name, base_sides in list(out):
        sides = [s if feats[i].near_level() else 0
                 for i, s in enumerate(base_sides)]
        out.append((f"{base_name}  [at a level only]", sides))

    # And the regime slices, on the plain imbalance read.
    groups = ft.regimes(feats)
    imb = [f.imbalance for f in feats]
    hi = quantile([abs(x) for x in imb], 0.70) or 0.0
    for label, idx in groups.items():
        if label == "unknown" or len(idx) < 300:
            continue
        keep = set(idx)
        sides = [0] * n
        for i, v in enumerate(imb):
            if i in keep and abs(v) >= hi:
                sides[i] = 1 if v > 0 else -1
        out.append((f"imbalance in regime {label}", sides))

    return out


def show(label, r: bar.Result, indent="  "):
    if r.rate is None:
        print(f"{indent}{label:<52} {r.n:>5} calls, "
              f"{r.resolved} resolved -- too few")
        return
    print(f"{indent}{label:<52} {r.rate:>6.1%}  "
          f"needs {r.needs:.1%}  ${r.per_trade:>7.2f}/trade  "
          f"n={r.resolved:,}")


def main(argv=None) -> int:
    a = parse(argv or sys.argv[1:])

    train_path = Path(a.train)
    if not train_path.exists():
        print(f"No such file: {train_path}", file=sys.stderr)
        return 2
    train = ec.load(train_path)
    train.sort(key=lambda s: s.ts)
    tfeat = ft.build(train, tick=a.tick)

    print(f"\ntarget: +{a.take:g} / -{a.stop:g} ticks within {a.bars} bars")
    print(f"        breakeven {bar.breakeven(a.take, a.stop):.1%}  "
          f"(a symmetric {a.take:g}/{a.take:g} would need "
          f"{bar.breakeven(a.take, a.take):.1%})")
    print(f"\ntraining: {len(train):,} bars from {train_path.name}")

    cands = candidates(train, tfeat, a.tick)
    print(f"\n=== search ({len(cands)} candidates) -- NOT EVIDENCE ===")
    print("  With this many looks several will clear any bar. That is what")
    print("  a search is. Only the validation number below counts.\n")

    scored = []
    for name, sides in cands:
        r = bar.run(train, sides, a.take, a.stop, a.bars, tick=a.tick)
        scored.append((name, sides, r))

    ok = [x for x in scored
          if x[2].resolved >= MIN_RESOLVED and x[2].per_trade is not None]
    ok.sort(key=lambda x: x[2].per_trade, reverse=True)

    for name, _, r in ok[:8]:
        show(name, r)
    if not ok:
        print("  Nothing had enough resolved trades to be promotable.")
        return 1

    # -- promote exactly one -------------------------------------------
    if a.promote:
        pick = next((x for x in scored if x[0] == a.promote), None)
        if pick is None:
            print(f"\nNo candidate named {a.promote!r}.", file=sys.stderr)
            return 2
    else:
        pick = ok[0]
    name, _, train_r = pick

    print(f"\n=== promoted (chosen by dollars per trade, "
          f"min {MIN_RESOLVED} resolved) ===")
    print(f"  {name}")
    print(f"  training: {train_r.rate:.1%}, ${train_r.per_trade:.2f}/trade "
          f"-- selection-biased by construction, not a forecast.")

    if not a.validate:
        print("\nNo validation file given, so there is no result to report.")
        print("Scoring the training half again would measure the fit, not")
        print("the rule. Pass a second file of bars this search never saw.\n")
        return 0

    vpath = Path(a.validate)
    if not vpath.exists():
        print(f"No such file: {vpath}", file=sys.stderr)
        return 2
    val = ec.load(vpath)
    val.sort(key=lambda s: s.ts)
    if val and train and not (val[0].ts > train[-1].ts
                              or val[-1].ts < train[0].ts):
        print("\nREFUSING: the validation file overlaps the training file "
              "in time.\nThat is not out-of-sample.", file=sys.stderr)
        return 1

    vfeat = ft.build(val, tick=a.tick)
    vcands = dict(candidates(val, vfeat, a.tick))
    if name not in vcands:
        print(f"\nThe promoted rule does not exist on the validation set "
              f"(too few bars for its feature).", file=sys.stderr)
        return 1

    r = bar.run(val, vcands[name], a.take, a.stop, a.bars, tick=a.tick)
    print(f"\n=== validation ({len(val):,} bars from {vpath.name}) "
          f"-- THIS IS THE RESULT ===\n")
    show(name, r, indent="  ")

    if r.rate is None:
        print("\n  Nothing resolved. No result.\n")
        return 0

    # The comparison that actually means something: the same bars, the same
    # number of trades, the same long/short balance -- directions shuffled.
    ctrl = bar.control(val, vcands[name], a.take, a.stop, a.bars,
                       tick=a.tick)
    print(f"\n  shuffled control (same bars, directions permuted, "
          f"{ctrl['trials']} runs)")
    if ctrl["mean"] is not None:
        print(f"    {ctrl['mean_rate']:>6.1%}  "
              f"${ctrl['mean']:>7.2f}/trade  sd ${ctrl['sd']:.2f}")

    clear = bar.beats_control(r.per_trade, ctrl)
    edge = (r.per_trade - ctrl["mean"]) if ctrl["mean"] is not None else 0.0

    print(f"\n  rule ${r.per_trade:,.2f}  vs  control "
          f"${ctrl['mean']:,.2f}   ->  ${edge:+,.2f} per trade")
    print(f"  {'CLEARS the control' if clear else 'DOES NOT clear the control'}"
          f" (needs 2 sd = ${2*(ctrl['sd'] or 0):.2f} above it)")
    print(f"\n  training said ${train_r.per_trade:.2f} -- the gap between "
          f"that and\n  the validation number is the fitting.")

    if not clear:
        print("\n  Note that a positive dollars-per-trade on its own is NOT "
              "an edge here.\n  With a +5/-3 barrier, expectancy depends on "
              "path geometry as much as\n  on direction, so a rule that "
              "picks favourable bars clears the\n  theoretical breakeven "
              "while knowing nothing. Only the gap over the\n  shuffle is "
              "evidence.")
        print("\n  Stop here. A different feature against this same file "
              "converts it\n  into training, one test at a time, and "
              "nothing in the output shows it.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
