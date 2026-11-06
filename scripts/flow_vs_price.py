#!/usr/bin/env python3
"""The 5-minute leg: who crossed the spread, against where price ended.

    python scripts/flow_vs_price.py data/ES_*.dbn.zst --tp 4 --sl 4

Chris:

    "all i want is for the direction who is crossing the spread to
    recompute every 5 minutes thats all"

That change is made. This is the number that says whether it was
worth making, on his own tape, with the entry rule and the exits
otherwise untouched.

WHY IT IS A DIFFERENT FACT, not a different smoothing.

    Close-against-open says where price ENDED UP over the window.
    It is the OUTCOME. Signed aggressive volume says who was paying
    the spread to make it happen, which is the CAUSE, and the two
    come apart in the cases that matter most: a candle closing green
    on sellers hitting the bid into a thinning book is a different
    market from one closing green on buyers lifting offers, and
    close-against-open calls them the same thing.

    Four readings of the outcome -- current bar, last closed bar,
    anchored, run-length -- all lost on 109 hours AND all took the
    same 1,194 trades. Same count, different definitions: the leg
    was not selecting anything, so no reading of it could have
    helped. The flow is the first change that alters WHICH bars
    qualify rather than how one of them is phrased.

THE WINDOW CLEARS AT EACH BOUNDARY, which is what was asked for and
is not free: right after a boundary the count is near zero and the
direction is flat until contracts arrive. Flat blocks an entry, so
this also removes the first-seconds-of-the-bar entry the price
version was taking. `flat bars` below says how often that bit.

GROSS IS THE HEADLINE. Any change that trades less improves net on
a tape with no edge, because the round trip is the only systematic
drain. Gross says whether the ENTRIES got better.

Reads files. Nothing here places, sizes, routes or times an order.
"""
from __future__ import annotations

import argparse
import glob
import sys

sys.path.insert(0, ".")

from liqmap import timed as tm                            # noqa: E402
from liqmap.kinetic import Kinetic                        # noqa: E402

TICK = 0.25
M1, M5 = 60.0, 300.0


def load(paths):
    """ts, price, size and the RAW aggressor character.

    The side is kept as Databento spells it, 'N' included, because
    `Kinetic` needs an unattributed print to move the price
    reference the next signed one is measured against even though
    it casts no vote. Dropping them here would quietly change the
    number this script exists to report.
    """
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
            sz = float(dbn._get(rec, "size", 0) or 0)
            if ts > 0 and px > 0 and sz > 0:
                out.append((ts, px, sz, dbn._side(rec)))
    out.sort(key=lambda r: r[0])
    return out


def run(prints, leg: str, tp: float, sl: float):
    """The timed rule. Only the 5-minute direction changes."""
    kin = Kinetic(period_s=M5, aligned=True)
    out = []
    b1o = b1c = b5o = b5c = 0.0
    k1 = k5 = -1.0
    entry = t_entry = 0.0
    side_s = ""
    taken = -1.0
    bars, flat_bars, blocked = set(), set(), 0

    for ts, px, sz, sd in prints:
        kin.on_trade(ts, px, sz, sd)
        n1 = float(int(ts // M1) * M1)
        n5 = float(int(ts // M5) * M5)
        if n1 != k1:
            k1, b1o = n1, px
        if n5 != k5:
            k5, b5o = n5, px
        b1c = b5c = px
        bars.add(n5)

        d1 = tm.direction({"open": b1o, "close": b1c})
        d5 = (kin.direction if leg == "flow"
              else tm.direction({"open": b5o, "close": b5c}))
        if d5 == "flat":
            flat_bars.add(n5)

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
                out.append({"why": why, "ticks": ticks})
                side_s = ""
            continue

        if taken == n5:
            continue
        want, _w = tm.entry_side(d1, d5)
        if want:
            side_s, entry, t_entry, taken = want, px, ts, n5
        elif d5 == "flat":
            blocked += 1
    return out, len(bars), len(flat_bars)


def score(trades, cost: float) -> dict:
    if not trades:
        return {"n": 0, "gross": 0.0, "gpt": 0.0, "npt": 0.0, "win": 0.0}
    g = [t["ticks"] for t in trades]
    n = [x - cost for x in g]
    return {"n": len(g), "gross": round(sum(g), 1),
            "gpt": round(sum(g) / len(g), 3),
            "npt": round(sum(n) / len(n), 3),
            "win": round(100.0 * sum(1 for x in n if x > 0) / len(n), 1)}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--tp", type=float, default=4.0)
    ap.add_argument("--sl", type=float, default=4.0)
    ap.add_argument("--cost", type=float, default=1.0)
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

    known = sum(1 for r in fit if r[3] not in ("N", "", None))
    print(f"\n  fit    : {len(fit):,} prints, "
          f"{100.0 * known / max(1, len(fit)):.1f}% with a known aggressor")
    print(f"  holdout: {hold_s[0].split('/')[-1]}, {len(hold):,} prints")
    if known < 0.5 * len(fit):
        print("\n  LESS THAN HALF THE TAPE HAS AN AGGRESSOR. The flow leg")
        print("  is being built from a minority of prints -- read the")
        print("  numbers below as provisional and check the schema.")

    print(f"\n  {'5m leg':<9}{'where':<9}{'bars':>7}{'flat':>7}"
          f"{'trades':>8}{'win%':>7}{'GROSS':>9}{'g/trade':>9}{'n/trade':>9}")
    print("  " + "-" * 74)
    res = {}
    for leg in ("price", "flow"):
        for lbl, rows in (("fit", fit), ("holdout", hold)):
            tr, bars, flat = run(rows, leg, a.tp, a.sl)
            s = score(tr, a.cost)
            res[(leg, lbl)] = s
            print(f"  {leg if lbl == 'fit' else '':<9}{lbl:<9}{bars:>7}"
                  f"{flat:>7}{s['n']:>8}{s['win']:>7.1f}{s['gross']:>9.0f}"
                  f"{s['gpt']:>9.3f}{s['npt']:>9.3f}")
        print()

    print("  " + "=" * 74)
    p_f, p_h = res[("price", "fit")], res[("price", "holdout")]
    f_f, f_h = res[("flow", "fit")], res[("flow", "holdout")]
    print(f"  fit      price {p_f['gpt']:+.3f}  ->  flow {f_f['gpt']:+.3f}"
          f"   ({f_f['gpt'] - p_f['gpt']:+.3f})")
    print(f"  holdout  price {p_h['gpt']:+.3f}  ->  flow {f_h['gpt']:+.3f}"
          f"   ({f_h['gpt'] - p_h['gpt']:+.3f})")
    print(f"  trades   {p_f['n']} -> {f_f['n']} on the fit")
    print()

    if f_f["n"] == p_f["n"]:
        print("  THE TRADE COUNT DID NOT MOVE. The flow agrees with the")
        print("  candle on every bar that mattered, so this leg is not")
        print("  selecting either. That is the same finding as before.")
    elif f_h["gpt"] > 0 and f_f["gpt"] > 0:
        print("  Positive before costs on BOTH sets -- the first thing")
        print(f"  here that has been. It clears the round trip only if")
        print(f"  {f_h['gpt']:+.3f} beats {a.cost:.1f}: "
              f"{'it does' if f_h['gpt'] > a.cost else 'it does not yet'}.")
    elif f_f["gpt"] > p_f["gpt"] and f_h["gpt"] > p_h["gpt"]:
        print("  Better than price on both sets and still negative. The")
        print("  direction improved; it is not yet an edge.")
    else:
        print("  No better than the candle. Who is crossing the spread is")
        print("  not the missing piece either -- which points at the exit")
        print("  rather than the entry (see scripts/where_it_goes.py).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
