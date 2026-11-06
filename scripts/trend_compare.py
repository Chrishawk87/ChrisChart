#!/usr/bin/env python3
"""Does the direction carry? Four definitions, measured on real tape.

    python scripts/trend_compare.py data/ES_*.dbn.zst
    python scripts/trend_compare.py data/ES_*.dbn.zst --tp 4 --sl 4

Chris: "the direction is a continuation and not resetting at ever 5
minute interval".

THE RULE AS BUILT measures the 5-minute direction as close against
open OF THE CURRENT CANDLE, so the anchor is thrown away every 300
seconds. A move that has run twenty minutes is re-measured from
scratch at each boundary; in the first seconds of a new candle price
sits on its open and one tick of noise decides the direction; and a
single pullback candle reads as a reversal because that candle is
the only reference.

Measured on 109 hours: -0.379 ticks a trade BEFORE costs, confirmed
at -0.308 on a held-back day. A coin flip -- which is what a
direction measured from an arbitrary recent anchor should give.

So here are four ways to answer "which way is the 5-minute going",
run through the same rule on the same tape:

    BAR      this candle's close against this candle's open. What
             it does now. The control.
    CLOSED   the last CLOSED candle's direction, held until
             another closed candle disagrees.
    ANCHOR   price against the open of the candle where the move
             BEGAN. Moves only when the move turns, on the same
             counter RUN uses, so a pullback does not reset it.
    RUN      the direction holds until N consecutive closed
             candles disagree. A pullback is one candle; a
             reversal is a sequence.

GROSS IS THE HEADLINE, not net. Any change that reduces the trade
count improves net on a tape with no edge, because the round trip is
the only systematic drain -- so net flatters every variant that
trades less. Gross says whether the ENTRIES got better, which is the
only thing that could make this rule work.

Reads files. Nothing here places, sizes, routes or times an order.
"""
from __future__ import annotations

import argparse
import glob
import sys

sys.path.insert(0, ".")

from liqmap import timed as tm                          # noqa: E402
from liqmap.trend import MODES, Trend                   # noqa: E402

TICK = 0.25
M1, M5 = 60.0, 300.0


def load(paths):
    try:
        import databento as db
        from liqmap import dbn
    except ImportError:
        raise SystemExit("pip install databento")
    out = []
    for path in paths:
        for rec in db.DBNStore.from_file(path):
            if dbn._action(rec) not in (dbn.ACTION_TRADE, dbn.ACTION_FILL):
                continue
            ts = float(dbn.seconds(dbn._get(rec, "ts_event")) or 0.0)
            px = dbn.price(dbn._get(rec, "price"))
            if ts > 0 and px > 0:
                out.append((ts, px))
    out.sort(key=lambda r: r[0])
    return out


def run(prints, mode: str, tp: float, sl: float, flip_bars: int) -> list:
    """The timed rule, with the 5-minute direction taken from `mode`.

    The 1-minute leg and the exit are untouched -- only the
    definition of the slow direction changes, so any difference is
    attributable to that and to nothing else.
    """
    tr = Trend(mode=mode, flip_bars=flip_bars)
    out, b1o, b1c, b5o, b5c = [], 0.0, 0.0, 0.0, 0.0
    k1 = k5 = -1.0
    side = entry = t_entry = 0.0
    side_s = ""
    taken = -1.0

    for ts, px in prints:
        n1 = float(int(ts // M1) * M1)
        n5 = float(int(ts // M5) * M5)
        if n1 != k1:
            k1, b1o = n1, px
        if n5 != k5:
            if k5 >= 0:
                tr.on_close(b5o, b5c)       # the candle that just ended
            k5, b5o = n5, px
        b1c = b5c = px
        tr.on_price(px, b5o)

        d1 = tm.direction({"open": b1o, "close": b1c})
        d5 = tr.direction

        if side_s:
            raw = (px - entry) / TICK
            ticks = raw if side_s == "up" else -raw
            why = ""
            if sl > 0 and ticks <= -sl:
                why = "sl"
            elif tp > 0 and ticks >= tp:
                why = "tp"
            elif tm.reversed_against(side_s, d1)[0]:
                why = "reversal"
            if why:
                out.append((side_s, t_entry, entry, ts, px, why, ticks))
                side_s = ""
            continue

        if taken == n5:
            continue
        want, _w = tm.entry_side(d1, d5)
        if want:
            side_s, entry, t_entry, taken = want, px, ts, n5
    return out


def score(trades, cost: float) -> dict:
    if not trades:
        return {"trades": 0, "gross": 0.0, "gpt": 0.0, "net": 0.0,
                "npt": 0.0, "win": 0.0}
    g = [t[6] for t in trades]
    n = [x - cost for x in g]
    return {"trades": len(g),
            "gross": round(sum(g), 1), "gpt": round(sum(g) / len(g), 3),
            "net": round(sum(n), 1), "npt": round(sum(n) / len(n), 3),
            "win": round(100.0 * sum(1 for x in n if x > 0) / len(n), 1)}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--tp", type=float, default=4.0)
    ap.add_argument("--sl", type=float, default=4.0)
    ap.add_argument("--cost", type=float, default=1.0)
    ap.add_argument("--flip-bars", type=int, default=2)
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
    if len(spans) < 2:
        print("  need at least two usable files")
        return 2
    spans.sort(key=lambda r: r[2])
    *fit_s, hold_s = spans
    fit = [r for _p, rows, _t in fit_s for r in rows]
    fit.sort(key=lambda r: r[0])
    hold = hold_s[1]

    print(f"\n  fit    : {len(fit):,} prints "
          f"({(fit[-1][0] - fit[0][0]) / 3600:.0f}h)")
    print(f"  holdout: {hold_s[0].split('/')[-1]}, {len(hold):,} prints "
          f"({(hold[-1][0] - hold[0][0]) / 3600:.0f}h)\n")

    print(f"  {'direction':<10}{'where':<9}{'trades':>7}{'win%':>7}"
          f"{'GROSS':>9}{'g/trade':>9}{'net':>9}{'n/trade':>9}")
    print("  " + "-" * 69)
    best = None
    for mode in MODES:
        f = score(run(fit, mode, a.tp, a.sl, a.flip_bars), a.cost)
        h = score(run(hold, mode, a.tp, a.sl, a.flip_bars), a.cost)
        for lbl, s in (("fit", f), ("holdout", h)):
            print(f"  {mode if lbl == 'fit' else '':<10}{lbl:<9}"
                  f"{s['trades']:>7}{s['win']:>7.1f}{s['gross']:>9.0f}"
                  f"{s['gpt']:>9.3f}{s['net']:>9.0f}{s['npt']:>9.3f}")
        if f["trades"] >= 30 and h["trades"] >= 10:
            if best is None or h["gpt"] > best[2]["gpt"]:
                best = (mode, f, h)
        print()

    print("  " + "=" * 69)
    if best is None:
        print("  No definition took enough trades on both sets to judge.")
        return 0
    mode, f, h = best
    print(f"  BEST ON THE HOLDOUT, BY GROSS: {mode}")
    print(f"    fit     {f['gpt']:+.3f} ticks a trade before costs")
    print(f"    holdout {h['gpt']:+.3f}")
    print()
    if h["gpt"] > 0.0 and f["gpt"] > 0.0:
        print(f"  THE ENTRIES HAVE AN EDGE BEFORE COSTS, on both sets.")
        print(f"  To be worth trading it has to clear the round trip: "
              f"{h['gpt']:+.3f} against {a.cost:.1f}.")
        if h["gpt"] > a.cost:
            print("  It does. That is the first thing here that has.")
        else:
            print("  It does not yet -- the edge is real and smaller than "
                  "the spread.\n  Fewer and better trades is the next "
                  "question, not a new indicator.")
    elif h["gpt"] > 0.0:
        print("  Positive on the holdout and not on the fit. One day is "
              "not enough\n  to call that an edge.")
    else:
        print("  Still no edge before costs. Continuation is not the "
              "missing piece,\n  or not by itself.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
