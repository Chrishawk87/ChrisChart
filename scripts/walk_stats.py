#!/usr/bin/env python3
"""Ask the walk dataset whether agreement is worth trading.

    python scripts/walk_stats.py ~/Downloads/ES_walk

Reads `ES_walk_minutes.parquet` (or the CSV) and splits the 4-tick
bracket every way that could matter: by agreement, by how many
timeframes agree, by where in the 15-minute candle the minute sits, by
hour, and by half of the sample.

THE QUESTION IT ANSWERS

With a 4-tick target against a 4-tick stop, a trade is worth taking
only if it clears

    breakeven = (stop + cost) / (target + stop)

At 4 and 4 that is 50% before costs, 55% at 0.4 ticks of cost, 62.5% at
a full tick, 68.75% at 1.5. Those are the lines every row of every
table here is measured against. A win rate of 52% is not a small edge
on this bracket; it is a loss.

HOW THE RATES ARE COUNTED

    resolved win%   target / (target + stop)

`neither` and `ambiguous` are NOT in that denominator. A minute whose
bracket never resolved is not half a win, and a second that covered
both levels is not a coin flip -- putting either into the rate invents
precision the data does not have. Both are reported as their own share
so you can see how much is being set aside.

EXPECTANCY USES THE UNRESOLVED ROWS RATHER THAN DROPPING THEM

A `neither` row is a real position that was still open when the clock
ran out. Dropping it flatters the result, because unresolved rows skew
toward the quiet stretches. So expectancy is

    mean over ALL rows of (+target, -stop, or fwd_15 if unresolved)
    minus cost

which is what the account would actually have seen.

ARM AGAINST ARM, NOT ARM AGAINST 50%

Minutes inside a trending session are not independent, and
autocorrelation lifts both arms together. So the number that matters is
the GAP between the agreeing rows and the non-agreeing rows, not either
one against a coin. Both are printed side by side on every split.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER. It counts rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TARGET_TICKS = 4
STOP_TICKS = 4
COSTS = (0.0, 0.4, 1.0, 1.5)        # ticks, round trip
RESOLVED = ("target", "stop")


def breakeven(target: int = TARGET_TICKS, stop: int = STOP_TICKS,
              cost: float = 0.0) -> float:
    """The win rate this bracket needs just to break even."""
    return (stop + cost) / (target + stop)


def win_rate(counts: dict):
    """target / (target + stop). None when nothing resolved.

    `neither` and `ambiguous` are deliberately absent from the
    denominator: an unresolved bracket is not half a win, and a second
    that covered both levels is not a coin flip.
    """
    t = counts.get("target", 0)
    s = counts.get("stop", 0)
    return (t / (t + s)) if (t + s) else None


def expectancy(frame, cost: float = 0.0,
               target: int = TARGET_TICKS, stop: int = STOP_TICKS):
    """Mean ticks per row, counting the unresolved at their mark.

    An unresolved row is a position that was still open when the cap
    ran out, so it is worth `fwd_15` -- not zero, and not nothing.
    Dropping those rows flatters the result because they cluster in
    the quiet stretches.
    """
    import numpy as np
    if not len(frame):
        return None
    out = np.where(
        frame["first_4t"].values == "target", float(target),
        np.where(frame["first_4t"].values == "stop", float(-stop),
                 frame["fwd_15"].fillna(0.0).values))
    return float(out.mean() - cost)


def census(frame) -> dict:
    c = frame["first_4t"].value_counts().to_dict()
    return {k: int(c.get(k, 0))
            for k in ("target", "stop", "neither", "ambiguous")}


def line(label: str, frame) -> str:
    n = len(frame)
    if not n:
        return f"  {label:<22} {'(no rows)':>12}"
    c = census(frame)
    wr = win_rate(c)
    e0 = expectancy(frame, 0.0)
    e1 = expectancy(frame, 1.0)
    wr_s = "   --  " if wr is None else f"{100 * wr:6.2f}%"
    unres = 100.0 * (c["neither"] + c["ambiguous"]) / n
    return (f"  {label:<22} {n:>9,}  {wr_s}  "
            f"{unres:5.1f}%  {e0:+7.3f}  {e1:+7.3f}")


HEAD = (f"  {'bucket':<22} {'rows':>9}  {'win%':>7}  "
        f"{'unres':>6}  {'E(0)':>7}  {'E(1t)':>7}")


def section(title: str, pairs) -> None:
    print(title)
    print(HEAD)
    for label, frame in pairs:
        print(line(label, frame))
    print()


def load(path: Path):
    import pandas as pd
    pq = path / "ES_walk_minutes.parquet"
    csv = path / "ES_walk_minutes.csv"
    if pq.exists():
        df = pd.read_parquet(pq)
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        raise SystemExit(f"no ES_walk_minutes.* in {path}")
    df["first_4t"] = df["first_4t"].astype(str)
    for col in ("fwd_1", "fwd_5", "fwd_15", "mfe", "mae", "to_close"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["hour"] = df["minute"].astype(str).str.slice(0, 2).astype(int)
    df["agreeing"] = agreeing_count(df)
    return df


def agreeing_count(df):
    """How many of the three were showing the SAME non-flat direction.

    Counted against d1's direction, so a row where the 5m and 15m agree
    with each other but not with the minute scores 1, not 2 -- the
    minute is the one being entered on.
    """
    d1 = df["d1"]
    same = df[["d1", "d5", "d15"]].eq(d1, axis=0).sum(axis=1)
    # A flat minute is the only case `eq` gets wrong: every flat frame
    # equals it, so a dead bar would score 3. There is nothing to enter
    # on, so it scores 0.
    #
    # An `& ne(0)` term here would ALSO fix that case, which made the
    # two guards cover for each other: either could be deleted and
    # every test still passed. One guard, tested, beats two that hide
    # each other's removal.
    return same.where(d1 != 0, 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="the ES_walk directory")
    ap.add_argument("--rth", action="store_true",
                    help="08:30-15:14 Chicago only")
    args = ap.parse_args(argv)

    try:
        import pandas  # noqa: F401
    except ImportError:
        print("pandas is not installed in this interpreter.\n"
              "    source .venv/bin/activate && pip install pandas",
              file=sys.stderr)
        return 2

    df = load(Path(args.path).expanduser())
    if args.rth:
        mins = df["minute"].astype(str)
        df = df[(mins >= "08:30") & (mins < "15:15")]

    tradeable = df[df["d1"] != 0]
    agree = tradeable[tradeable["agree"] == 1]
    other = tradeable[tradeable["agree"] == 0]

    print(f"{len(df):,} minutes  ·  {df['session'].nunique():,} sessions"
          f"  ·  {args.path}")
    print(f"{len(tradeable):,} of them had a direction to trade "
          f"({len(agree):,} agreeing, {len(other):,} not)")
    print()
    print(f"A {TARGET_TICKS}-tick target against a {STOP_TICKS}-tick "
          f"stop needs this win rate to break even:")
    for c in COSTS:
        print(f"    {c:>3.1f} ticks of cost   "
              f"{100 * breakeven(cost=c):6.2f}%")
    print()
    print("win% is target/(target+stop); unresolved rows are excluded "
          "from it and shown separately.")
    print("E(0) and E(1t) are mean ticks per row at 0 and 1 tick of "
          "cost, counting unresolved rows at fwd_15.")
    print()

    section("ALL THREE AGREE, OR NOT", [
        ("all three agree", agree),
        ("they do not", other),
        ("every tradeable row", tradeable),
    ])

    section("HOW MANY AGREE WITH THE MINUTE", [
        (f"{n} of 3",
         tradeable[tradeable["agreeing"] == n]) for n in (1, 2, 3)
    ])

    section("WHERE IN THE 15-MINUTE CANDLE (agreeing rows only)", [
        ("idx 0-4  (first 5m)", agree[agree["idx"] < 5]),
        ("idx 5-9  (second)", agree[(agree["idx"] >= 5)
                                    & (agree["idx"] < 10)]),
        ("idx 10-14 (third)", agree[agree["idx"] >= 10]),
    ])

    hours = sorted(agree["hour"].unique())
    section("BY HOUR, CHICAGO (agreeing rows only)", [
        (f"{h:02d}:00", agree[agree["hour"] == h]) for h in hours
    ])

    # Split on the DATE, not on the row number: sessions differ in
    # length, so halving the rows puts part of a session in both arms.
    days = sorted(df["session"].astype(str).unique())
    cut = days[len(days) // 2]
    section(f"TWO HALVES, SPLIT AT {cut} (agreeing rows only)", [
        (f"first half", agree[agree["session"].astype(str) < cut]),
        (f"second half", agree[agree["session"].astype(str) >= cut]),
    ])

    print("FORWARD TICKS, SIGNED BY THE DIRECTION SHOWING")
    print(f"  {'bucket':<22} {'fwd_1':>8} {'fwd_5':>8} "
          f"{'fwd_15':>8} {'mfe':>8} {'mae':>8}")
    for label, frame in (("all three agree", agree),
                         ("they do not", other)):
        if not len(frame):
            continue
        bits = [f"{frame[c].median():8.2f}"
                for c in ("fwd_1", "fwd_5", "fwd_15", "mfe", "mae")]
        print(f"  {label:<22} " + " ".join(bits))
    print("  (medians)")
    print()

    wa, wo = win_rate(census(agree)), win_rate(census(other))
    if wa is not None and wo is not None:
        gap = 100 * (wa - wo)
        print(f"THE GAP: agreeing {100 * wa:.2f}%  vs  "
              f"not agreeing {100 * wo:.2f}%  =  {gap:+.2f} points.")
        print("That gap, not either number against 50%, is the "
              "comparison that survives autocorrelation.")
        need = 100 * breakeven(cost=0.4)
        print(f"Either way, {need:.2f}% is the bar at 0.4 ticks of "
              f"cost, and {100 * breakeven(cost=1.0):.2f}% at a tick.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
