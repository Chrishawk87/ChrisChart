#!/usr/bin/env python3
"""Does the three-column read work on ES?

    python scripts/es_study.py data/ES_c_0_mbp-10_2026-09-25.dbn.zst

Walks the DBN file once, condenses it to one row per candle, then asks the
two questions that matter, in order:

    DIRECTION -- when the read leaned, did the candle close that way?
    REACH     -- did it get to a scalp target BEFORE going the same
                 distance against?

The second is the one that decides whether this is tradeable. A read can
call the sign at 59% -- a real, interval-clear edge -- and still reach a
three-tick target first only 38% of the time, which loses money. Direction
alone would have passed it. That is not hypothetical: it is what the
validation on synthetic data with known truth produced, and it is why both
numbers are printed here rather than just the flattering one.

THE CONDENSED FILE IS CACHED

Parsing 44 million records takes minutes. The condensed rows take
milliseconds to re-score, so the parse happens once and lands beside the
DBN file as .slices.jsonl. Delete that file to force a re-parse. Pass
--interval to change the candle size, which needs its own parse.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap import dbn, escondense as ec, project as pj      # noqa: E402
from liqmap.levels import Phase, phase                       # noqa: E402
from liqmap.vote import cast                                 # noqa: E402

if not hasattr(pj, "reach_test"):                            # pragma: no cover
    raise SystemExit(
        "liqmap/project.py is an older copy without reach_test/distance.\n"
        "Unzip chrischart-reach.zip over the repo root first.")

# ES contract facts. Tick value is fixed in dollars, so the breakeven in
# dollars does not move with the index level -- only the bps framing does.
ES_TICK = 0.25
ES_TICK_USD = 12.50
ES_RT_COST_USD = 5.00           # commission + fees, round turn, one contract

# Scalp targets, in ticks.
TARGETS_TICKS = (2, 3, 4, 5, 6, 8)


def columns(s: ec.Slice) -> dict[str, tuple[str, float]]:
    """The three columns as (direction, strength), before they are combined.

    Kept separate from the vote because a strong-but-noisy column buries a
    genuine one: on synthetic data where only the book carried an edge, the
    combined vote read 50% while the book alone read 72%. If the combined
    number is flat, the per-column numbers are the next question, not a
    footnote.
    """
    imb = s.imbalance(levels=5)
    total = s.buy_vol + s.sell_vol
    lean = (s.delta / total) if total > 0 else 0.0
    move = s.prior_move_bps
    return {
        "book": ("up" if imb > 0.05 else "down" if imb < -0.05 else "flat",
                 abs(imb)),
        "delta": ("up" if lean > 0.05 else "down" if lean < -0.05 else "flat",
                  min(1.0, abs(lean))),
        "price": ("up" if move > 0.1 else "down" if move < -0.1 else "flat",
                  min(1.0, abs(move) / 5.0)),
    }


def signed(col: tuple[str, float]) -> float:
    """One column as a signed number, for scoring it on its own."""
    d, strength = col
    return strength if d == "up" else -strength if d == "down" else 0.0


def read(s: ec.Slice) -> float:
    """The three columns, as they stand at the candle's open.

    BOOK   resting size imbalance over the top five levels.
    DELTA  signed aggression over the window ending at the open.
    PRICE  where the open sits against the prior close.

    Strengths are normalised to roughly [0, 1] so no column can dominate by
    unit choice alone -- delta in contracts would otherwise swamp an
    imbalance in [-1, 1] and the vote would be a delta read wearing three
    hats.
    """
    c = columns(s)
    v = cast(c["book"][0], c["book"][1],
             c["delta"][0], c["delta"][1],
             c["price"][0], c["price"][1])
    return v.net


def rows_from(slices, unanimous: bool = False,
              column: str | None = None) -> list[dict]:
    """Condensed rows to the shape the scoring functions want.

    `column` scores one column on its own instead of the combined vote.
    """
    out = []
    for s in slices:
        net = signed(columns(s)[column]) if column else read(s)
        if unanimous and abs(net) < 0.5:
            continue
        # `close_px` is the name distance() reads; `close` is kept for the
        # direction check below. Two names for one number is ugly, but
        # renaming the field inside project.py would touch the crypto path
        # that is already under test.
        out.append({"net": net, "price": s.open, "path": s.path,
                    "close": s.close, "close_px": s.close, "ts": s.ts})
    return out


def direction(rows) -> dict:
    """Of the bars where the read leaned, how many closed that way."""
    used = [r for r in rows
            if abs(r["net"]) > 1e-9 and r["price"] > 0 and r["close"] > 0]
    if len(used) < 20:
        return {"ready": False, "n": len(used)}
    hit = sum(1 for r in used
              if (r["close"] > r["price"]) == (r["net"] > 0))
    lo, hi = pj._wilson(hit, len(used))
    return {"ready": True, "n": len(used), "rate": hit / len(used),
            "lo": lo, "hi": hi, "real": lo > 0.5 or hi < 0.5}


def needs(target_ticks: int) -> float:
    """Breakeven hit rate for a symmetric target and stop, after costs."""
    win = target_ticks * ES_TICK_USD - ES_RT_COST_USD
    lose = target_ticks * ES_TICK_USD + ES_RT_COST_USD
    return lose / (win + lose)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2

    path = Path(argv[1])
    if not path.exists():
        print(f"No such file: {path}", file=sys.stderr)
        return 2

    interval = 300.0
    if "--interval" in argv:
        interval = float(argv[argv.index("--interval") + 1])

    # The flow window. At candle sizes it defaults to the candle; at
    # intrabar sampling rates a 300s window would smear the tape over
    # sixty samples and every row would read nearly the same thing.
    lookback = interval
    if "--lookback" in argv:
        lookback = float(argv[argv.index("--lookback") + 1])

    cache = path.with_suffix("").with_suffix(
        f".{int(interval)}s.slices.jsonl")

    if cache.exists():
        slices = ec.load(cache)
        print(f"\n{len(slices):,} candles from {cache.name} (cached)")
    else:
        print(f"\nParsing {path.name} ... (once; then it is cached)")
        t0 = time.time()
        try:
            store = dbn.open_file(str(path))
        except ImportError:
            print("pip install databento", file=sys.stderr)
            return 2
        n = ec.save(ec.condense(store, interval_s=interval,
                                lookback_s=lookback), cache)
        slices = ec.load(cache)
        print(f"{n:,} candles in {time.time() - t0:,.0f}s -> {cache.name}")

    if not slices:
        print("No complete candles. The window may be too short.",
              file=sys.stderr)
        return 1

    span_h = (slices[-1].ts - slices[0].ts) / 3600.0
    px = slices[-1].close
    tick_bps = ES_TICK / px * 10_000.0 if px > 0 else 0.0
    print(f"  span {span_h:,.1f} hours   last {px:,.2f}   "
          f"1 tick = {tick_bps:.3f} bps")

    # --- each column on its own ----------------------------------------
    # Before reading anything into the combined vote. A column with a real
    # edge can be buried by a louder column that is pure noise, and the
    # combined number alone cannot tell that apart from "nothing works".
    print("\n=== each column alone ===")
    print(f"  {'column':<10}{'direction':<12}{'interval':<20}{'n':>7}   verdict")
    for name in ("book", "delta", "price"):
        d = direction(rows_from(slices, column=name))
        if not d["ready"]:
            print(f"  {name:<10}too few bars ({d['n']})")
            continue
        v = "REAL" if d["real"] else "coin"
        print(f"  {name:<10}{d['rate']:>7.1%}     "
              f"({d['lo']:.1%}-{d['hi']:.1%})      {d['n']:>7,}   {v}")

    # --- the regular session on its own --------------------------------
    # ES overnight is thin and mostly follows other markets; pooling it
    # with the cash session averages two different markets together.
    rth = [s for s in slices if phase(s.ts) is Phase.RTH]
    if len(rth) >= 40:
        d = direction(rows_from(rth))
        if d["ready"]:
            v = "REAL" if d["real"] else "coin"
            print(f"\n  regular session only (09:30-16:00 ET): "
                  f"{d['rate']:.1%} ({d['lo']:.1%}-{d['hi']:.1%})  "
                  f"n={d['n']:,}  {v}")

    for label, unanimous in (("all bars the read leaned on", False),
                             ("only where the columns agree", True)):
        rows = rows_from(slices, unanimous=unanimous)
        print(f"\n=== {label} ===  n={len(rows):,}")

        d = direction(rows)
        if not d["ready"]:
            print(f"  direction   too few bars ({d['n']})")
            continue
        verdict = "REAL" if d["real"] else "not distinguishable from a coin"
        print(f"  direction   {d['rate']:.1%}  "
              f"({d['lo']:.1%}-{d['hi']:.1%})  n={d['n']:,}  {verdict}")

        dist = pj.distance(rows)
        if dist.get("ready"):
            print(f"  distance    favourable "
                  f"{dist['median_favourable_bps']:+.2f} bps  "
                  f"adverse {dist['median_adverse_bps']:+.2f} bps  "
                  f"close {dist['median_close_bps']:+.2f} bps  (medians)")

        print(f"\n  {'target':<20}{'reached first':<24}{'needs':<9}verdict")
        targets = [t * tick_bps for t in TARGETS_TICKS]
        table = pj.reach_test(rows, targets)
        for ticks, row in zip(TARGETS_TICKS, table):
            gross = ticks * ES_TICK_USD
            need = needs(ticks)
            if not row.get("ready") or row.get("rate") is None:
                print(f"  {ticks} ticks (${gross:>6,.2f})  "
                      f"only {row['resolved']} resolved -- too few")
                continue
            got, lo, hi = row["rate"], row["ci_low"], row["ci_high"]
            if lo > need:
                mark = "CLEARS"
            elif hi < need:
                mark = "loses"
            else:
                mark = "unclear"
            print(f"  {ticks} ticks (${gross:>6,.2f})  "
                  f"{got:>6.1%} ({lo:.1%}-{hi:.1%})    "
                  f"{need:>5.1%}    {mark}")

    print("\n  One bar is one observation, and a bar touching both levels "
          "is scored as the loss.")
    print("  Wilson intervals assume independent bars. Consecutive candles "
          "are not fully")
    print("  independent, so the true intervals are a little wider than "
          "these -- on synthetic")
    print("  coins, 2 runs in 20 were flagged 'real' at a nominal 5%. Treat "
          "anything marginal")
    print("  as unproven, and believe a result that clears by a margin.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
