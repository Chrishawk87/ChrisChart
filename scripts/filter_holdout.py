#!/usr/bin/env python3
"""Do the filters help? Fit on some files, confirm on one never seen.

    python scripts/filter_holdout.py data/ES_*.dbn.zst
    python scripts/filter_holdout.py data/ES_*.dbn.zst --tp 4 --sl 4
    python scripts/filter_holdout.py data/ES_*.dbn.zst --csv picks.csv

WHAT IT DOES

    Runs the TIMED rule -- 1-minute and 5-minute agreeing, one trade
    per 5-minute candle, out on take profit, stop, or the 1-minute
    closing back through its open -- and then adds entry FILTERS
    built from the tape:

        PUSH        signed aggressive volume as a ratio in [-1, 1].
                    Must back the trade's direction by at least some
                    amount.
        EFFICIENCY  whether that flow is moving price at all, or
                    being absorbed.
        D SCORE     Chris's slope + distance-from-VWAP, both in
                    ticks so the weights are comparable.

    Every filter only ever VETOES. None of them can open a trade the
    rule did not already want, because a continuous score crosses
    any threshold repeatedly inside a bar and using one as a trigger
    is how the churn came back twice before.

WHY A HOLDOUT, SAID PLAINLY

    There are enough knobs here that SOMETHING will look good on any
    one stretch of tape. That is not a risk to be careful about, it
    is a certainty: with a few hundred combinations and 400 hours,
    the best one is partly skill and partly the luck of this
    particular sample, and nothing in the fit result tells you which.

    So the last file is set aside before anything is tried, every
    combination is scored on the rest, the single best is taken, and
    THEN it is run once on the file it never saw. The gap between
    those two numbers is the size of the self-deception.

    The baseline -- the rule with no filter at all -- is reported on
    both, because a filter that improves the fit and not the holdout
    has told you nothing, and one that beats the baseline on BOTH is
    the only result worth acting on.

Reads files. Nothing here places, sizes, routes or times an order.
"""
from __future__ import annotations

import argparse
import csv
import glob
import itertools
import sys
from dataclasses import dataclass, field

sys.path.insert(0, ".")

from liqmap import timed as tm                          # noqa: E402
from liqmap.dscore import DScore                        # noqa: E402
from liqmap.kinetic import Kinetic                      # noqa: E402

TICK = 0.25
TICK_VALUE = 12.50
ROUND_TRIP = 1.0
M1, M5 = 60.0, 300.0


@dataclass
class Bar:
    start: float
    open: float = 0.0
    close: float = 0.0

    def feed(self, px: float) -> None:
        if not self.open:
            self.open = px
        self.close = px

    def d(self) -> dict:
        return {"open": self.open, "close": self.close}


@dataclass
class Trade:
    side: str
    t0: float
    entry: float
    t1: float = 0.0
    exit: float = 0.0
    why: str = ""

    @property
    def ticks(self) -> float:
        if not self.exit:
            return 0.0
        raw = (self.exit - self.entry) / TICK
        return raw if self.side == "up" else -raw


@dataclass
class Cfg:
    """One combination of filter settings. All-zero is the baseline."""
    push: float = 0.0          # |push_ratio| must reach this
    eff: bool = False          # efficiency must back the side
    dmin: float = 0.0          # |D| must reach this, and agree

    def label(self) -> str:
        if not (self.push or self.eff or self.dmin):
            return "no filter (baseline)"
        bits = []
        if self.push:
            bits.append(f"push>={self.push:.2f}")
        if self.eff:
            bits.append("eff backs it")
        if self.dmin:
            bits.append(f"|D|>={self.dmin:.0f}")
        return ", ".join(bits)


def run(prints, cfg: Cfg, tp: float, sl: float, cost: float,
        w_slope: float, w_vwap: float, slope_s: float) -> list:
    """One pass. Returns the closed trades."""
    kin = Kinetic(period_s=M5)
    ds = DScore(slope_s=slope_s, w_slope=w_slope, w_vwap=w_vwap)
    out: list = []
    b1 = b5 = None
    open_t: Trade | None = None
    taken_bar = -1.0

    for ts, px, size, side in prints:
        kin.on_trade(ts, px, size, side)
        ds.on_trade(ts, px, size)

        k1 = float(int(ts // M1) * M1)
        k5 = float(int(ts // M5) * M5)
        if b1 is None or b1.start != k1:
            b1 = Bar(start=k1)
        if b5 is None or b5.start != k5:
            b5 = Bar(start=k5)
        b1.feed(px)
        b5.feed(px)
        d1 = tm.direction(b1.d())
        d5 = tm.direction(b5.d())

        if open_t is not None:
            t = open_t
            raw = (px - t.entry) / TICK
            ticks = raw if t.side == "up" else -raw
            why = ""
            if sl > 0 and ticks <= -sl:
                why = "sl"
            elif tp > 0 and ticks >= tp:
                why = "tp"
            elif tm.reversed_against(t.side, d1)[0]:
                why = "reversal"
            if why:
                t.t1, t.exit, t.why = ts, px, why
                out.append(t)
                open_t = None
            continue

        if taken_bar == k5:
            continue
        want, _why = tm.entry_side(d1, d5)
        if not want:
            continue

        # ---- the filters, veto only -------------------------------
        if cfg.push:
            r = kin.push_ratio
            if want == "up" and r < cfg.push:
                continue
            if want == "down" and r > -cfg.push:
                continue
        if cfg.eff:
            # Efficiency is the co-movement of price and flow, so it
            # is positive when the push is WORKING whichever way it
            # points. Backing the side means: flow is moving price,
            # not being absorbed.
            if kin.efficiency <= 0:
                continue
            if kin.direction != want:
                continue
        if cfg.dmin and not ds.agrees(want, cfg.dmin):
            continue

        open_t = Trade(side=want, t0=ts, entry=px)
        taken_bar = k5
    return out


def score(trades: list, cost: float) -> dict:
    done = [t for t in trades if t.exit]
    if not done:
        return {"trades": 0, "net": 0.0, "avg": 0.0, "win_pct": 0.0}
    net = [t.ticks - cost for t in done]
    wins = sum(1 for x in net if x > 0)
    return {"trades": len(done),
            "net": round(sum(net), 1),
            "avg": round(sum(net) / len(done), 3),
            "win_pct": round(100.0 * wins / len(done), 1)}


def _gross_tail(s: dict, cost: float) -> None:
    """What the rule does BEFORE costs, which is the real question.

    A net loss of 1.38 ticks a trade against a 1.00 tick round trip
    is a rule that is 0.38 ticks worse than a coin flip -- not a
    rule with a broken exit or a bad filter. The two readings call
    for completely different work, and only the gross one says
    which.
    """
    if not s.get("trades"):
        return
    gross = s["net"] + cost * s["trades"]
    per = gross / s["trades"]
    print(f"\n  BEFORE COSTS it was {gross:+.0f} ticks over "
          f"{s['trades']} trades: {per:+.3f} a trade.")
    if abs(per) < 0.5:
        print("  That is a coin flip. The entries carry no edge, and no "
              "filter on a\n  coin flip produces one -- it only reduces "
              "how often you pay the spread.")
        print(f"  To clear a {cost:.1f} tick round trip the entry has to "
              "be worth MORE than\n  that gross, which is a different "
              "rule rather than a tuned one.")
    elif per > 0:
        print("  The entries are positive before costs and the round trip "
              "is eating them.\n  That is worth attacking: fewer, better "
              "trades, or a cheaper entry.")
    else:
        print("  The entries are negative before costs even enter. The "
              "direction itself\n  is wrong, and trading it less often "
              "only loses more slowly.")


def load_one(path: str) -> list:
    """(ts, px, size, side) for one file. Empty if it has no prints.

    An OHLCV file has no trade records at all -- it is bars, not
    tape -- so it loads as nothing. That is a reason to leave it out
    with a word, not to crash four lines later on an empty list.
    """
    return load([path])


def file_spans(paths: list[str]) -> list:
    """(path, prints, first_ts, last_ts), skipping files with no tape."""
    out = []
    for path in paths:
        rows = load_one(path)
        if not rows:
            print(f"  skipping {path.split('/')[-1]} -- no trade prints "
                  "in it (an OHLCV file is bars, not tape)")
            continue
        out.append((path, rows, rows[0][0], rows[-1][0]))
    return out


def load(paths: list[str]) -> list:
    """(ts, px, size, side) for every print, in order."""
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


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--tp", type=float, default=4.0)
    ap.add_argument("--sl", type=float, default=4.0)
    ap.add_argument("--cost", type=float, default=ROUND_TRIP)
    ap.add_argument("--slope-s", type=float, default=60.0)
    ap.add_argument("--csv", default="")
    a = ap.parse_args(argv)

    paths: list[str] = []
    for p in a.files:
        paths.extend(sorted(glob.glob(p)) or [p])
    spans = file_spans(paths)
    if len(spans) < 2:
        print(f"\n  {len(spans)} usable file(s). Need at least two: one "
              "to fit on, one to hold back.")
        return 2

    # SPLIT BY TIME, NOT BY FILENAME.
    #
    # The holdout has to be the LATEST tape, because that is the
    # only split that matches how the rule would be used: fit on
    # what has happened, test on what happened next. Sorting by
    # name put a 2024 file after three 2026 ones and would have
    # "validated" a 2026 fit on tape from two years earlier -- which
    # is not a holdout, it is a different market.
    spans.sort(key=lambda r: r[2])
    *fit_spans, hold_span = spans

    def when(ts):
        import datetime as _dt
        return _dt.datetime.fromtimestamp(
            ts, _dt.timezone.utc).strftime("%Y-%m-%d")

    print()
    for path, rows, t0, t1 in fit_spans:
        print(f"  fit on   : {path.split('/')[-1]}  ({when(t0)}, "
              f"{len(rows):,} prints)")
    print(f"  held back: {hold_span[0].split('/')[-1]}  "
          f"({when(hold_span[2])}, {len(hold_span[1]):,} prints)  "
          "<- the LATEST tape")
    print("  the holdout is not looked at until one configuration has "
          "been chosen.\n")

    fit = [r for _p, rows, _a, _b in fit_spans for r in rows]
    fit.sort(key=lambda r: r[0])
    hold = hold_span[1]
    print(f"  fit {len(fit):,} prints over "
          f"{(fit[-1][0] - fit[0][0]) / 3600:.0f}h  |  "
          f"holdout {len(hold):,} over "
          f"{(hold[-1][0] - hold[0][0]) / 3600:.0f}h\n")

    # The D score is dominated by its VWAP term, which runs to
    # hundreds of ticks in a trend, so a threshold of 10 filters
    # nothing. The push ratio moves in the other direction -- 0.35
    # sustained over five minutes is rare. Both grids are spread
    # wide enough that the trade count below shows a filter doing
    # nothing and one doing far too much.
    grid = [Cfg()]
    for push in (0.0, 0.05, 0.10, 0.20, 0.35):
        for eff in (False, True):
            for dmin in (0.0, 25.0, 60.0, 150.0, 300.0):
                c = Cfg(push=push, eff=eff, dmin=dmin)
                if c.label() != "no filter (baseline)":
                    grid.append(c)
    weights = ((1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (1.0, -1.0))

    print(f"  trying {len(grid)} filter sets x {len(weights)} weightings"
          f" = {len(grid) * len(weights)} configurations on the fit set")
    print("  (that many tries is exactly why the holdout exists)\n")

    rows = []
    for cfg in grid:
        for ws, wv in weights:
            if not cfg.dmin and (ws, wv) != weights[0]:
                continue        # weights do nothing when D is unused
            s = score(run(fit, cfg, a.tp, a.sl, a.cost, ws, wv,
                          a.slope_s), a.cost)
            rows.append((cfg, ws, wv, s))

    base = next(r for r in rows if r[0].label() == "no filter (baseline)")

    # A CONFIGURATION THAT TOOK FOUR TRADES IS NOT A GOOD ONE, it is
    # an unmeasured one. Without this the grid reliably picks
    # whichever filter happened to leave a handful of winners, and
    # the holdout then has nothing to disagree with. Thirty is the
    # same floor the journal uses before it will call a record
    # readable.
    MIN_TRADES = 30
    eligible = [r for r in rows if r[3]["trades"] >= MIN_TRADES]
    thin = len(rows) - len(eligible)
    rows.sort(key=lambda r: -r[3]["net"])
    eligible.sort(key=lambda r: -r[3]["net"])

    print(f"  {'configuration':<46}{'trades':>7}{'win%':>7}"
          f"{'net':>9}{'avg':>8}")
    print("  " + "-" * 77)
    for cfg, ws, wv, s in eligible[:8]:
        lbl = cfg.label() + (f"  w={ws:g}/{wv:g}" if cfg.dmin else "")
        print(f"  {lbl:<46}{s['trades']:>7}{s['win_pct']:>7.1f}"
              f"{s['net']:>9.0f}{s['avg']:>8.3f}")
    print(f"  {'...':<46}")
    b = base[3]
    print(f"  {base[0].label():<46}{b['trades']:>7}{b['win_pct']:>7.1f}"
          f"{b['net']:>9.0f}{b['avg']:>8.3f}")

    if not eligible:
        print(f"\n  every configuration took fewer than {MIN_TRADES} "
              "trades on the fit set.")
        print("  nothing here is measured well enough to carry to a "
              "holdout.")
        return 1
    print(f"\n  {thin} of {len(rows)} configurations took fewer than "
          f"{MIN_TRADES} trades and were\n  excluded -- a filter that "
          "leaves four trades has not been measured.")
    best_cfg, bws, bwv, best_fit = eligible[0]

    print(f"\n  {'=' * 77}")
    print(f"  THE HOLDOUT -- looked at once, now")
    print(f"  {'=' * 77}")
    hb = score(run(hold, Cfg(), a.tp, a.sl, a.cost, 1.0, 1.0,
                   a.slope_s), a.cost)
    hc = score(run(hold, best_cfg, a.tp, a.sl, a.cost, bws, bwv,
                   a.slope_s), a.cost)

    print(f"  {'':<46}{'trades':>7}{'win%':>7}{'net':>9}{'avg':>8}")
    print(f"  {'baseline, no filter':<46}{hb['trades']:>7}"
          f"{hb['win_pct']:>7.1f}{hb['net']:>9.0f}{hb['avg']:>8.3f}")
    lbl = best_cfg.label() + (f"  w={bws:g}/{bwv:g}" if best_cfg.dmin else "")
    print(f"  {lbl:<46}{hc['trades']:>7}{hc['win_pct']:>7.1f}"
          f"{hc['net']:>9.0f}{hc['avg']:>8.3f}")

    # A CONFIGURATION THAT TOOK NO TRADES ON THE HOLDOUT WAS NOT
    # TESTED, and its average of 0.000 must never be compared
    # against a baseline that traded. The floor was applied when
    # CHOOSING on the fit set and not when SCORING on the holdout,
    # so a filter so strict it never fired came back reading like a
    # success. It is the same mistake in the same script, one step
    # later.
    HOLD_MIN = 10
    if hc["trades"] < HOLD_MIN:
        print(f"\n  THE CHOSEN FILTER TOOK {hc['trades']} TRADES ON THE "
              "HOLDOUT.")
        print("  That is not a result. An average of 0.000 on no trades "
              "is not a win,\n  it is an untested filter -- too strict "
              "to fire on unseen tape at all.")
        print(f"\n  What the holdout DOES say: the rule itself took "
              f"{hb['trades']} trades for\n  {hb['net']:+.0f} ticks "
              f"({hb['avg']:+.3f} a trade), against {best_fit['avg']:+.3f} "
              "on the fit set.\n  The rule is losing at the same rate on "
              "tape it has never seen.")
        _gross_tail(hb, a.cost)
        return 0

    print(f"\n  best on the FIT      {best_fit['avg']:+.3f} ticks a trade")
    print(f"  the same on HOLDOUT  {hc['avg']:+.3f} ticks a trade")
    drop = best_fit["avg"] - hc["avg"]
    print(f"  the gap              {drop:+.3f}  <- how much was this "
          "sample rather than an edge")

    print()
    if hc["net"] > 0 and hc["avg"] > hb["avg"]:
        print("  IT SURVIVED. The filter beat the baseline on tape it had"
              " never seen,\n  and made money doing it. That is worth "
              "acting on -- carefully.")
    elif hc["avg"] > hb["avg"]:
        print("  It beat the baseline on the holdout but is still "
              "negative.\n  The filter helps and the underlying rule is "
              "still losing.")
    else:
        print("  IT DID NOT SURVIVE. Whatever the fit found was this "
              "sample, not\n  an edge. The honest move is to drop it "
              "rather than widen the grid\n  until something sticks.")
    _gross_tail(hb, a.cost)

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["where", "config", "w_slope", "w_vwap", "trades",
                        "win_pct", "net_ticks", "avg_ticks"])
            for cfg, ws, wv, s in rows:
                w.writerow(["fit", cfg.label(), ws, wv, s["trades"],
                            s["win_pct"], s["net"], s["avg"]])
            for lbl, s in (("holdout baseline", hb),
                           ("holdout chosen", hc)):
                w.writerow([lbl, best_cfg.label(), bws, bwv, s["trades"],
                            s["win_pct"], s["net"], s["avg"]])
        print(f"\n  wrote {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
