#!/usr/bin/env python3
"""Where do the ticks actually go? Entry selectivity and exit drain.

    python scripts/where_it_goes.py data/ES_*.dbn.zst --tp 4 --sl 4

FOUR DEFINITIONS OF THE 5-MINUTE DIRECTION ALL LOST, and all took
the same number of trades: 1194, 1193, 1194, 1193 over 103 hours.

That is the finding, not the losses. If the 5-minute leg gated
entries, changing its definition would change WHICH bars qualify
and the counts would move. They do not move. So the leg is not
binding, and no definition of it could have helped -- which is why
continuation, run-length and anchoring all landed in the same
place.

This asks the two questions that follow, and asks them separately
because they have different answers and different fixes.

1. IS THE ENTRY SELECTING ANYTHING?

   If nearly every 5-minute bar with prints takes a trade, and it
   takes it in the first seconds of the bar, then the rule is not
   a setup. It is "enter at the start of every bar", and the two
   direction legs are a coin deciding which way. Right after a
   boundary price sits on its open and the 1-minute flips through
   both directions within a few prints, so whichever way the
   5-minute points, the 1-minute agrees almost at once.

2. IS THE EXIT THE DRAIN?

   Symmetric 4-tick TP and 4-tick SL on a tape with no edge should
   give about ZERO gross. It gives -0.379. Something is taking
   ticks that is not the entry.

   The suspect is the reversal exit. Long, and the 1-minute closing
   back through its open means price moved DOWN -- so the reversal
   fires preferentially when the trade is already losing. It is a
   second stop, tighter than the real one and triggered by the
   losing direction, while a winner still has to walk the full 4
   ticks. Small losses get added to the pile and winners do not get
   the matching treatment. That shape produces negative gross out of
   an entry carrying no information at all.

   So the run below reports every exit reason separately, and runs
   the rule once more with the reversal exit OFF. If gross goes to
   about zero with it off, the reversal exit is the whole loss and
   the entry was never the problem.

CONTROLS, because a number without one is an opinion:

   RANDOM SIDE   same entry instants, side by coin flip. If the
                 rule matches this, the two legs carry no
                 information and the direction debate is settled.
   ALWAYS LONG   and ALWAYS SHORT, which expose any drift or any
                 asymmetry in the exit geometry itself.

Reads files. Nothing here places, sizes, routes or times an order.
"""
from __future__ import annotations

import argparse
import glob
import random
import sys

sys.path.insert(0, ".")

from liqmap import timed as tm                            # noqa: E402

# trend_compare is a sibling script, not a module on the path, so
# it is loaded by location. Importing it as `scripts.trend_compare`
# needs a package marker that is not in the repo, and would fail on
# a clean checkout with a confusing error.
import importlib.util as _ilu                             # noqa: E402
import os as _os                                          # noqa: E402

_spec = _ilu.spec_from_file_location(
    "_trend_compare",
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                  "trend_compare.py"))
_tc = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_tc)
load = _tc.load

TICK = 0.25
M1, M5 = 60.0, 300.0


def run(prints, tp: float, sl: float, side_pick=None,
        use_reversal: bool = True):
    """The timed rule as built, instrumented.

    `side_pick(want, rng)` replaces the rule's own side with a
    control's. Entry INSTANTS are untouched either way, so a
    control differs from the rule in exactly one thing.
    """
    rng = random.Random(1234)
    out, b1o, b1c, b5o, b5c = [], 0.0, 0.0, 0.0, 0.0
    k1 = k5 = -1.0
    entry = t_entry = 0.0
    side_s = ""
    entry_bar = 0.0
    taken = -1.0
    bars_seen = set()
    open_at_end = 0

    for ts, px in prints:
        n1 = float(int(ts // M1) * M1)
        n5 = float(int(ts // M5) * M5)
        if n1 != k1:
            k1, b1o = n1, px
        if n5 != k5:
            k5, b5o = n5, px
        b1c = b5c = px
        bars_seen.add(n5)

        d1 = tm.direction({"open": b1o, "close": b1c})
        d5 = tm.direction({"open": b5o, "close": b5c})

        if side_s:
            raw = (px - entry) / TICK
            ticks = raw if side_s == "up" else -raw
            why = ""
            if sl > 0 and ticks <= -sl:
                why = "sl"
            elif tp > 0 and ticks >= tp:
                why = "tp"
            elif use_reversal and tm.reversed_against(side_s, d1)[0]:
                why = "reversal"
            if why:
                out.append({"side": side_s, "t": t_entry, "px": entry,
                            "exit_t": ts, "exit_px": px, "why": why,
                            "ticks": ticks,
                            "into_bar": t_entry - entry_bar,
                            "held": ts - t_entry})
                side_s = ""
            continue

        if taken == n5:
            continue
        want, _w = tm.entry_side(d1, d5)
        if want:
            side_s = side_pick(want, rng) if side_pick else want
            entry, t_entry, taken, entry_bar = px, ts, n5, n5
    if side_s:
        open_at_end = 1
    return out, len(bars_seen), open_at_end


def gross(trades) -> tuple:
    if not trades:
        return 0, 0.0, 0.0
    g = [t["ticks"] for t in trades]
    return len(g), round(sum(g), 1), round(sum(g) / len(g), 3)


def pct(a, b) -> float:
    return (100.0 * a / b) if b else 0.0


def med(xs) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[len(s) // 2]


def report(label, prints, tp, sl):
    trades, bars, dangling = run(prints, tp, sl)
    n, g, gpt = gross(trades)

    print(f"\n  {label.upper()}")
    print("  " + "-" * 67)

    print("\n  IS THE ENTRY SELECTING ANYTHING?\n")
    print(f"    5-minute bars with prints      {bars:>8,}")
    print(f"    bars that took a trade         {n:>8,}"
          f"   ({pct(n, bars):.1f}%)")
    into = [t["into_bar"] for t in trades]
    print(f"    median seconds into the bar    {med(into):>8.1f}s")
    print(f"    90th percentile                "
          f"{(sorted(into)[int(len(into) * 0.9)] if into else 0.0):>8.1f}s")
    if bars and pct(n, bars) > 90.0:
        print("\n    Nearly every bar qualifies. This is not a setup --")
        print("    it is an entry on the clock, with the two direction")
        print("    legs deciding only which way.")

    print("\n  WHERE DO THE TICKS GO?\n")
    print(f"    {'exit':<12}{'n':>7}{'share':>8}{'mean':>9}{'total':>10}")
    for why in ("tp", "sl", "reversal"):
        sub = [t for t in trades if t["why"] == why]
        if not sub:
            continue
        sn, sg, sgpt = gross(sub)
        print(f"    {why:<12}{sn:>7}{pct(sn, n):>7.1f}%"
              f"{sgpt:>9.3f}{sg:>10.0f}")
    print(f"    {'ALL':<12}{n:>7}{100.0:>7.1f}%{gpt:>9.3f}{g:>10.0f}")
    if dangling:
        print(f"    (1 trade still open at the end of the data, excluded)")

    print("\n  THE SAME RULE WITH THE REVERSAL EXIT OFF\n")
    nr, _b, _d = run(prints, tp, sl, use_reversal=False)
    n2, g2, gpt2 = gross(nr)
    print(f"    {'tp/sl only':<12}{n2:>7}{100.0:>7.1f}%"
          f"{gpt2:>9.3f}{g2:>10.0f}")
    if n2 and abs(gpt2) < 0.1 and gpt < -0.2:
        print("\n    The loss is the REVERSAL EXIT, not the entry. It")
        print("    fires when price has moved against the trade, so it")
        print("    is a second stop -- tighter than the real one and")
        print("    triggered by the losing direction -- while a winner")
        print("    still has to walk the full take profit.")
    elif n2 and gpt2 < -0.2:
        print("\n    The loss survives without the reversal exit, so the")
        print("    reversal is not the whole story.")

    print("\n  CONTROLS -- same entry instants, side changed\n")
    print(f"    {'side':<14}{'n':>7}{'gross':>10}{'g/trade':>10}")
    for name, pick in (
        ("the rule", None),
        ("coin flip", lambda w, r: r.choice(("up", "down"))),
        ("always long", lambda w, r: "up"),
        ("always short", lambda w, r: "down"),
    ):
        tr, _b, _d = run(prints, tp, sl, side_pick=pick)
        cn, cg, cgpt = gross(tr)
        print(f"    {name:<14}{cn:>7}{cg:>10.0f}{cgpt:>10.3f}")
    print("\n    If the rule sits among the coin flips, the two")
    print("    direction legs carry no information and no definition")
    print("    of either one will change that.")
    print("\n    Not a perfect control: a different side exits at a")
    print("    different moment, which shifts later entries. Read it")
    print("    as a range to sit inside or outside, not a twin.")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--tp", type=float, default=4.0)
    ap.add_argument("--sl", type=float, default=4.0)
    a = ap.parse_args(argv)

    paths = []
    for p in a.files:
        paths.extend(sorted(glob.glob(p)) or [p])

    spans = []
    for path in paths:
        rows = load([path])
        if not rows:
            print(f"  skipping {path.split('/')[-1]} -- no trade prints")
            continue
        spans.append((path, rows, rows[0][0]))
    if not spans:
        print("  no usable files")
        return 2
    spans.sort(key=lambda r: r[2])

    if len(spans) >= 2:
        *fit_s, hold_s = spans
        fit = [r for _p, rows, _t in fit_s for r in rows]
        fit.sort(key=lambda r: r[0])
        report("fit", fit, a.tp, a.sl)
        report(f"holdout -- {hold_s[0].split('/')[-1]}",
               hold_s[1], a.tp, a.sl)
    else:
        report(spans[0][0].split("/")[-1], spans[0][1], a.tp, a.sl)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
