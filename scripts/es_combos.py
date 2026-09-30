#!/usr/bin/env python3
"""Does sign-correcting the columns rescue the read? Honest version.

    python scripts/es_combos.py data/ES_c_0_mbp-10_2026-09-21.60s.slices.jsonl

THE HYPOTHESIS, AND WHERE IT CAME FROM

The first two runs showed the same thing at both horizons:

    book    300s 53.3%   60s 52.0%   above 50 both times
    delta   300s 48.1%   60s 48.2%   below 50 both times

Resting size predicting continuation while aggressive flow predicts
reversal is a real microstructure story -- aggression into a level is the
thing that gets absorbed. And a vote that ADDS those two cancels them:
+2.0 and -1.8 sum to +0.2, which is the 49.3% the combined read produced.

So the hypothesis is that delta's sign is backwards in the vote.

WHY THIS SCRIPT IS SPLIT IN HALF

That hypothesis was found by looking at the answer. Testing it on the same
data would be measuring how well it was fitted, not whether it is true --
and with a couple of dozen slices examined across the runs so far, a 2%
effect is exactly the size that turns up by chance.

So the signs are chosen on the FIRST 60% of the days and never revisited.
Everything reported as a result comes from the LAST 40%, which had no part
in choosing anything. The split is chronological, never random: shuffling
days would let the afternoon of a day inform its own morning.

If the out-of-sample half disagrees with the in-sample half, the effect was
fitting. That is the outcome this script exists to be able to report.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap import escondense as ec, project as pj           # noqa: E402

ES_TICK_USD = 12.50
ES_RT_COST_USD = 5.00
TARGETS_TICKS = (2, 3, 4, 6)

SPLIT = 0.60


def wilson(k: int, n: int) -> tuple[float, float]:
    return pj._wilson(k, n)


def score(rows: list[dict]) -> dict:
    used = [r for r in rows
            if abs(r["net"]) > 1e-9 and r["price"] > 0 and r["close_px"] > 0]
    if len(used) < 30:
        return {"ready": False, "n": len(used)}
    hit = sum(1 for r in used
              if (r["close_px"] > r["price"]) == (r["net"] > 0))
    lo, hi = wilson(hit, len(used))
    return {"ready": True, "n": len(used), "rate": hit / len(used),
            "lo": lo, "hi": hi, "real": lo > 0.5 or hi < 0.5}


def needs(ticks: int) -> float:
    win = ticks * ES_TICK_USD - ES_RT_COST_USD
    lose = ticks * ES_TICK_USD + ES_RT_COST_USD
    return lose / (win + lose)


def col_values(s: ec.Slice) -> dict[str, float]:
    """Each column as a signed number, before any sign correction."""
    total = s.buy_vol + s.sell_vol
    return {
        "book": s.imbalance(levels=5),
        "delta": (s.delta / total) if total > 0 else 0.0,
        "price": max(-1.0, min(1.0, s.prior_move_bps / 5.0)),
    }


def rows_for(slices, weights: dict[str, float]) -> list[dict]:
    out = []
    for s in slices:
        v = col_values(s)
        net = sum(weights.get(k, 0.0) * v[k] for k in v)
        out.append({"net": net, "price": s.open, "path": s.path,
                    "close_px": s.close, "ts": s.ts})
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    path = Path(argv[1])
    if not path.exists():
        print(f"No such file: {path}\n\nRun es_study.py first -- it writes "
              f"the .slices.jsonl this reads.", file=sys.stderr)
        return 2

    slices = ec.load(path)
    slices.sort(key=lambda s: s.ts)
    if len(slices) < 400:
        print(f"Only {len(slices)} candles. Too few to split.",
              file=sys.stderr)
        return 1

    cut = int(len(slices) * SPLIT)
    # Push the cut to a day boundary so no day straddles the split.
    day = 86_400.0
    cut_day = int(slices[cut].ts // day)
    while cut < len(slices) - 1 and int(slices[cut].ts // day) == cut_day:
        cut += 1

    early, late = slices[:cut], slices[cut:]
    print(f"\n{len(slices):,} candles  ->  choose on {len(early):,}, "
          f"test on {len(late):,} (chronological, day-aligned)")

    # ---- CHOOSE, on the early half only -----------------------------
    print("\n=== choosing signs (early half -- NOT a result) ===")
    signs: dict[str, float] = {}
    for name in ("book", "delta", "price"):
        d = score(rows_for(early, {name: 1.0}))
        if not d["ready"]:
            signs[name] = 0.0
            print(f"  {name:<7} too few bars")
            continue
        signs[name] = 1.0 if d["rate"] >= 0.5 else -1.0
        way = "as-is" if signs[name] > 0 else "INVERTED"
        print(f"  {name:<7}{d['rate']:>6.1%} ({d['lo']:.1%}-{d['hi']:.1%})"
              f"  n={d['n']:>5,}  -> use {way}")

    # ---- TEST, on the late half, signs frozen -----------------------
    print("\n=== out of sample (late half -- this is the result) ===")

    candidates = {
        "book only": {"book": signs["book"]},
        "book + delta, signs corrected": {"book": signs["book"],
                                          "delta": signs["delta"]},
        "all three, signs corrected": {k: signs[k] for k in signs},
    }

    for label, w in candidates.items():
        rows = rows_for(late, w)
        d = score(rows)
        if not d["ready"]:
            print(f"\n  {label}: too few bars")
            continue
        verdict = "REAL" if d["real"] else "coin"
        print(f"\n  {label}")
        print(f"    direction  {d['rate']:.1%} "
              f"({d['lo']:.1%}-{d['hi']:.1%})  n={d['n']:,}  {verdict}")

        dist = pj.distance(rows)
        if dist.get("ready"):
            print(f"    distance   favourable "
                  f"{dist['median_favourable_bps']:+.2f}  adverse "
                  f"{dist['median_adverse_bps']:+.2f}  close "
                  f"{dist['median_close_bps']:+.2f} bps")

        px = late[-1].close
        tick_bps = 0.25 / px * 10_000.0 if px > 0 else 0.32
        table = pj.reach_test(rows, [t * tick_bps for t in TARGETS_TICKS])
        for ticks, row in zip(TARGETS_TICKS, table):
            if not row.get("ready") or row.get("rate") is None:
                continue
            need = needs(ticks)
            lo, hi = row["ci_low"], row["ci_high"]
            mark = ("CLEARS" if lo > need else
                    "loses" if hi < need else "unclear")
            print(f"    reach {ticks}tk  {row['rate']:.1%} "
                  f"({lo:.1%}-{hi:.1%})  needs {need:.1%}   {mark}")

    print("\n  The early half chose the signs and is not evidence. Only the "
          "late half is.\n  If the two halves disagree, the effect was "
          "fitting, not signal.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
