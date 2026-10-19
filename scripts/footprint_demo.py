#!/usr/bin/env python3
"""Drive the footprint engine with mock ticks and print the flags.

    python scripts/footprint_demo.py
    python scripts/footprint_demo.py --rule volume --volume 2000
    python scripts/footprint_demo.py --mult 3 --min-volume 20

THE GENERATOR IS A PLACEHOLDER, NOT A MARKET. It produces a random
walk with deliberately planted one-sided bursts, so the diagonal
detector has something to find and you can see whether it finds the
thing that was planted rather than decorations of the noise. No
conclusion about any real instrument follows from it.

The `--planted` run prints the bursts it inserted alongside the
flags, so detection can be checked against ground truth instead of
eyeballed.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap.footprint import (                             # noqa: E402
    AnyRule, DeltaRule, FootprintEngine, ImbalanceConfig, Tick,
    TimeRule, VolumeRule, BUY_IMBALANCE)


def mock_ticks(n: int = 4000, start: float = 5800.00,
               tick_size: float = 0.25, seed: int = 7,
               burst_every: int = 400, burst_len: int = 40):
    """A random walk with one-sided bursts planted at known places.

    Yields (tick, planted_side_or_None) so the caller can check what
    was detected against what was put there.
    """
    rng = random.Random(seed)
    ts = 1_700_000_000.0
    px = start
    burst = 0
    side = None

    for i in range(n):
        ts += rng.uniform(0.05, 0.6)

        if burst == 0 and i and i % burst_every == 0:
            burst = burst_len
            side = rng.choice(("buy", "sell"))

        if burst > 0:
            burst -= 1
            # A burst is one aggressor repeatedly paying up, which is
            # exactly what a diagonal imbalance is meant to catch.
            aggressor = side
            size = rng.randint(4, 30)
            if rng.random() < 0.65:
                px += tick_size if side == "buy" else -tick_size
            planted = side
            if burst == 0:
                side = None
        else:
            aggressor = rng.choice(("buy", "sell"))
            size = rng.randint(1, 6)
            px += rng.choice((-tick_size, 0.0, 0.0, tick_size))
            planted = None

        yield Tick(ts=ts, price=round(px, 4), size=size,
                   aggressor=aggressor), planted


def build_rule(args) -> object:
    if args.rule == "time":
        return TimeRule(args.seconds)
    if args.rule == "volume":
        return VolumeRule(args.volume)
    if args.rule == "delta":
        return DeltaRule(args.delta)
    return AnyRule(TimeRule(args.seconds), VolumeRule(args.volume))


def show(bar, eng, cfg, show_matrix: bool) -> None:
    sc = eng.scale
    flags = bar.imbalances(cfg)
    print(f"\n  bar  {bar.ts_open:.0f} -> {bar.ts_close:.0f}"
          f"   closed by {bar.close_reason}")
    print(f"       O {sc.to_price(bar.open_ticks):.2f}"
          f"  H {sc.to_price(bar.high_ticks):.2f}"
          f"  L {sc.to_price(bar.low_ticks):.2f}"
          f"  C {sc.to_price(bar.close_ticks):.2f}")
    poc = bar.point_of_control()
    print(f"       volume {bar.total_volume:,}   delta {bar.delta:+,}"
          f"   levels {len(bar.levels)}"
          + (f"   POC {sc.to_price(poc):.2f}" if poc is not None else ""))

    if show_matrix:
        print(f"       {'price':>10} {'bid':>8} {'ask':>8} {'delta':>8}")
        for p, lv in bar.rows():
            mark = ""
            for f in flags:
                if f.price_ticks == p:
                    mark = ("  <= BUY" if f.side == BUY_IMBALANCE
                            else "  <= SELL")
            print(f"       {sc.to_price(p):>10.2f} {lv.bid_volume:>8,}"
                  f" {lv.ask_volume:>8,} {lv.delta:>+8,}{mark}")

    if not flags:
        print("       no diagonal imbalance at this threshold")
        return
    for f in flags:
        print("       " + f.describe(sc))

    runs = bar.stacked(cfg, run=3)
    for side, lo, hi in runs:
        print(f"       STACKED {side.upper()} "
              f"{sc.to_price(lo):.2f} -> {sc.to_price(hi):.2f} "
              f"({hi - lo + 1} levels)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rule", default="time",
                    choices=("time", "volume", "delta", "any"))
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--volume", type=int, default=3000)
    ap.add_argument("--delta", type=int, default=400)
    ap.add_argument("--mult", type=float, default=4.0,
                    help="imbalance multiplier; 4.0 is 400%%")
    ap.add_argument("--min-volume", type=int, default=0,
                    help="floor the winning side must clear on its own. "
                         "0 reproduces the bare formula, which tags any "
                         "volume at all against an empty diagonal")
    ap.add_argument("--no-empty", action="store_true",
                    help="do not flag against an untraded diagonal")
    ap.add_argument("--edge", default="skip", choices=("skip", "zero"))
    ap.add_argument("--ticks", type=int, default=4000)
    ap.add_argument("--bars", type=int, default=6,
                    help="how many finished bars to print")
    ap.add_argument("--matrix", action="store_true",
                    help="print the full bid/ask matrix per bar")
    ap.add_argument("--tick-size", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    cfg = ImbalanceConfig(multiplier=args.mult,
                          min_volume=args.min_volume,
                          flag_empty_diagonal=not args.no_empty,
                          edge_policy=args.edge)
    eng = FootprintEngine(tick_size=args.tick_size,
                          rule=build_rule(args), imbalance=cfg)

    print(f"rule       {args.rule}")
    print(f"multiplier {args.mult:g}x   min volume {args.min_volume}"
          f"   empty diagonal {'ignored' if args.no_empty else 'counted'}"
          f"   edges {args.edge}")
    print(f"tick size  {args.tick_size}")
    print("\nMOCK DATA. A random walk with planted one-sided bursts -- "
          "it exercises\nthe detector, it says nothing about any real "
          "market.")

    planted_total = 0
    shown = 0
    for tick, planted in mock_ticks(args.ticks, tick_size=args.tick_size,
                                    seed=args.seed):
        if planted:
            planted_total += 1
        for bar in eng.on_tick(tick):
            if shown < args.bars:
                show(bar, eng, cfg, args.matrix)
                shown += 1

    print(f"\n{'=' * 64}")
    print(f"{eng.ticks_seen:,} ticks, {eng.contracts_seen:,} contracts, "
          f"{len(eng.history):,} bars")
    print(f"{planted_total:,} of those ticks were inside a planted burst")

    flagged = sum(len(b.imbalances(cfg)) for b in eng.history)
    stacks = sum(len(b.stacked(cfg, run=3)) for b in eng.history)
    print(f"{flagged:,} levels flagged, {stacks:,} stacked runs of 3+")
    if eng.history:
        per = flagged / len(eng.history)
        print(f"{per:.1f} flags a bar -- if that number is large, the "
              f"threshold is\nnot discriminating and --min-volume is the "
              f"setting to raise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
