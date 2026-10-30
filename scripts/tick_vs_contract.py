#!/usr/bin/env python3
"""Does weighting by contracts tell you anything that counting ticks does not?

THE QUESTION THIS ANSWERS, and it is the one the whole chart rests on.

Your candles are CONTRACT BASED: every event is weighted by the size
that appeared or left the inside quote, and the bar closes when those
contracts reach the target. A TICK reading weights every tick move
the same, however much size was behind it. If the two call the same
direction on every candle, the size weighting is decoration and the
chart could be a price chart. If they come apart, the gap is what the
book is telling you that price is not -- and this measures which of
them is right when they do.

THREE READINGS, ALL ON THE SAME CANDLE

    OFI     net contracts of inside-quote imbalance.  THE BOOK:
            resting orders, intent that can still be withdrawn.
    TICK    net ticks the mid travelled, unweighted.  THE PATH:
            what price actually did, with no size in it at all.
    DELTA   traded size by the aggressor CME STATED on each print.
            THE TAPE: intent already paid for.

There is deliberately NO uptick/downtick series here. That is the
other thing called "tick based", and it is a guess at the aggressor
that this feed already states outright -- wrong on a meaningful share
of prints, and worst exactly when the market is fast. Building it
would be adding the inferior version of DELTA.

WHAT IS SCORED

Each reading is taken at a fixed point into the candle -- late enough
to be made of something, early enough to still be tradeable -- and
scored against where that candle closed. Flat closes are counted
separately: on a one-tick market a real share of five-minute candles
end exactly where they started, and folding those into either column
is how a 50% reading comes to look like 60% or 40%.

THE COLUMN THAT MATTERS is not any single accuracy. It is the
DISAGREE rows: when the book and the path point opposite ways, which
one did the candle follow? That is the only number here that says
whether the contract weighting is earning its place.

Nothing here predicts, and nothing here places, sizes or times an
order.
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap.ofi import Touch, level_ofi                       # noqa: E402

RAW_TICK = 250_000_000.0
PERIOD_S = 300.0
# How far into the candle the three readings are taken. Matches what
# the live panel does: past the warmup, past the hold, still inside
# the first tenth of the candle.
AT_S = 25.0

UP, DOWN, NONE = "up", "down", ""


def _dir(x: float) -> str:
    return UP if x > 0 else (DOWN if x < 0 else NONE)


class Candle:
    """One five-minute candle, with all three readings in it."""

    __slots__ = ("open_ts", "open_px", "close_px", "at_px",
                 "ofi", "ticks", "delta", "ofi_at", "ticks_at",
                 "delta_at", "n", "gross")

    def __init__(self, open_ts: float, open_px: float) -> None:
        self.open_ts = open_ts
        self.open_px = open_px
        self.close_px = open_px
        self.at_px = 0.0
        self.ofi = 0.0          # contracts, running
        self.ticks = 0.0        # ticks of mid travel, running
        self.delta = 0.0        # traded contracts by stated aggressor
        self.ofi_at = None      # each, frozen at AT_S
        self.ticks_at = None
        self.delta_at = None
        self.n = 0
        self.gross = 0.0

    def freeze(self) -> None:
        if self.ofi_at is None:
            self.ofi_at = self.ofi
            self.ticks_at = self.ticks
            self.delta_at = self.delta

    @property
    def outcome(self) -> str:
        return _dir(self.close_px - self.at_px)

    @property
    def move_ticks(self) -> float:
        return (self.close_px - self.at_px) / RAW_TICK


def build(events, period_s: float, at_s: float) -> list:
    """(ts, e, mid, delta_sz) -> candles. One pass."""
    out: list = []
    cur: Candle | None = None
    prev_mid = None
    for ts, e, mid, d in events:
        open_ts = (ts // period_s) * period_s
        if cur is None or open_ts != cur.open_ts:
            if cur is not None:
                cur.freeze()
                if cur.at_px == 0.0:
                    cur.at_px = cur.close_px
                out.append(cur)
            cur = Candle(open_ts, mid)
            prev_mid = mid
        into = ts - cur.open_ts
        if into >= at_s and cur.ofi_at is None:
            cur.freeze()
            cur.at_px = mid
        cur.ofi += e
        cur.gross += abs(e)
        if prev_mid is not None:
            cur.ticks += (mid - prev_mid) / RAW_TICK
        cur.delta += d
        cur.close_px = mid
        cur.n += 1
        prev_mid = mid
    if cur is not None:
        cur.freeze()
        if cur.at_px == 0.0:
            cur.at_px = cur.close_px
        out.append(cur)
    return out


def score(candles, pick) -> dict:
    """How often one reading called the candle's close."""
    right = wrong = flat = none = 0
    moves = []
    for c in candles:
        d = _dir(pick(c))
        if not d:
            none += 1
            continue
        o = c.outcome
        if not o:
            flat += 1
            continue
        if d == o:
            right += 1
        else:
            wrong += 1
        moves.append(c.move_ticks if d == UP else -c.move_ticks)
    dec = right + wrong
    return {"n": right + wrong + flat, "right": right, "wrong": wrong,
            "flat": flat, "silent": none, "decided": dec,
            "pct": (right / dec) if dec else 0.0,
            "mean": statistics.mean(moves) if moves else 0.0}


def walk(n: int, seed: int, p_move: float = 0.12, dt: float = 0.01):
    """A control, not data. A book with nothing in it to find."""
    rng = random.Random(seed)
    bid, ts, prev = 7_700 * 10 ** 9, 0.0, None
    out = []
    for _ in range(n):
        if rng.random() < p_move:
            bid += RAW_TICK if rng.random() < 0.5 else -RAW_TICK
        ts += dt
        t = Touch(bid_px=int(bid), bid_sz=rng.randint(50, 900),
                  ask_px=int(bid + RAW_TICK), ask_sz=rng.randint(50, 900))
        if prev is not None:
            d = 0.0
            if rng.random() < 0.3:
                d = rng.randint(1, 20) * (1 if rng.random() < 0.5 else -1)
            out.append((ts, float(level_ofi(prev, t)),
                        bid + RAW_TICK / 2.0, d))
        prev = t
    return out


def from_files(paths, limit: int = 0):
    import databento as db

    from liqmap import dbn

    out, prev, n = [], None, 0
    pending = 0.0
    for path in paths:
        for rec in db.DBNStore.from_file(path):
            ts = float(dbn.seconds(dbn._get(rec, "ts_event")) or 0.0)
            tr = dbn.trade_from(rec)
            if tr is not None:
                # THE STATED AGGRESSOR, not a tick rule.
                sz = float(tr.sz)
                pending += sz if tr.aggressor == "buy" else -sz
            lv = dbn._get(rec, "levels")
            if not lv:
                continue
            t0 = lv[0]
            t = Touch(bid_px=int(dbn._get(t0, "bid_px", 0) or 0),
                      bid_sz=int(dbn._get(t0, "bid_sz", 0) or 0),
                      ask_px=int(dbn._get(t0, "ask_px", 0) or 0),
                      ask_sz=int(dbn._get(t0, "ask_sz", 0) or 0))
            if not t.valid or t.crossed:
                continue
            if prev is not None:
                out.append((ts, float(level_ofi(prev, t)),
                            (t.bid_px + t.ask_px) / 2.0, pending))
                pending = 0.0
            prev = t
            n += 1
            if limit and n >= limit:
                return out
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*",
                    help=".dbn / .dbn.zst MBP files; omit for a control")
    ap.add_argument("--period", type=float, default=PERIOD_S,
                    help="candle length in seconds")
    ap.add_argument("--at", type=float, default=AT_S,
                    help="seconds into the candle the readings are taken")
    ap.add_argument("--updates", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    if a.files:
        ev = from_files(a.files, a.limit)
        src = f"{len(a.files)} file" + ("" if len(a.files) == 1 else "s")
        real = True
    else:
        ev = walk(a.updates, a.seed)
        src = "A CONTROL, NOT DATA -- a random book with nothing in it"
        real = False
    if len(ev) < 100:
        print("  Not enough book updates to measure anything.")
        return 2

    cs = build(ev, a.period, a.at)
    cs = [c for c in cs if c.n > 20]
    if not cs:
        print("  No candle had enough updates in it to read.")
        return 2
    span = (ev[-1][0] - ev[0][0]) / 3600.0

    print(f"\n{'=' * 84}")
    print(f"TICK AGAINST CONTRACT   {src}")
    print(f"{len(ev):,} book updates   {span:.1f} hours   "
          f"{len(cs):,} candles of {a.period:g}s   read at {a.at:g}s in")
    print("=" * 84)

    rows = [("OFI   (contracts, book)", lambda c: c.ofi_at),
            ("TICK  (mid path, unweighted)", lambda c: c.ticks_at),
            ("DELTA (traded, stated side)", lambda c: c.delta_at)]
    print(f"  {'reading':>30}{'called':>8}{'right':>7}{'wrong':>7}"
          f"{'flat':>6}{'of decided':>12}{'mean tks':>10}")
    for name, pick in rows:
        r = score(cs, pick)
        print(f"  {name:>30}{r['decided']:>8,}{r['right']:>7,}"
              f"{r['wrong']:>7,}{r['flat']:>6,}"
              f"{100 * r['pct']:>11.1f}%{r['mean']:>+10.2f}")

    # THE ROWS THAT MATTER. Agreement tells you nothing on its own --
    # two readings of the same tape agree most of the time by
    # construction. The question is who is right when they do not.
    agree = [c for c in cs
             if _dir(c.ofi_at) and _dir(c.ofi_at) == _dir(c.ticks_at)]
    clash = [c for c in cs
             if _dir(c.ofi_at) and _dir(c.ticks_at)
             and _dir(c.ofi_at) != _dir(c.ticks_at)]
    print(f"\n  They agree on {len(agree):,} candles and disagree on "
          f"{len(clash):,}"
          f" ({100 * len(clash) / max(1, len(agree) + len(clash)):.1f}%"
          f" of the ones both called).")
    if agree:
        r = score(agree, lambda c: c.ofi_at)
        print(f"    agreeing  &mdash; {100 * r['pct']:.1f}% of"
              f" {r['decided']:,} decided, {r['mean']:+.2f} ticks"
              .replace("&mdash;", "—"))
    if clash:
        ro = score(clash, lambda c: c.ofi_at)
        rt = score(clash, lambda c: c.ticks_at)
        print(f"    disagreeing, the BOOK's side  —"
              f" {100 * ro['pct']:.1f}% of {ro['decided']:,} decided,"
              f" {ro['mean']:+.2f} ticks")
        print(f"    disagreeing, the PATH's side  —"
              f" {100 * rt['pct']:.1f}% of {rt['decided']:,} decided,"
              f" {rt['mean']:+.2f} ticks")
        if ro["decided"] < 100:
            print(f"\n    ONLY {ro['decided']} DECIDED DISAGREEMENTS."
                  f" That is not a sample, and the two")
            print("    lines above are a demonstration that the"
                  " machinery runs, not a result.")
            # HOW MUCH TAPE THIS ACTUALLY NEEDS, from the rate just
            # observed rather than from a guess. The two readings are
            # mechanically linked -- an uptick contributes the whole
            # new bid size to OFI -- so they rarely point different
            # ways, and the rare case is the only one worth measuring.
            rate = ro["decided"] / max(1, len(cs))
            if rate > 0:
                need_h = (100 / rate) * (a.period / 3600.0)
                print(f"\n    At {100 * rate:.1f} decided"
                      f" disagreements per hundred candles, a hundred"
                      f" of them is about")
                print(f"    {need_h:.0f} hours of tape"
                      f" \u2014 {need_h / 6.5:.0f} sessions, not one."
                      f" The two readings are linked by")
                print("    construction: an uptick contributes the"
                      " whole new bid size to OFI,")
                print("    so they point different ways only"
                      " occasionally, and that occasional")
                print("    case is the entire question.")
        elif ro["pct"] - rt["pct"] > 0.05:
            print("\n    The book wins the disagreements. That is the"
                  " contract weighting")
            print("    earning its place: size is carrying something"
                  " the price path is not.")
        elif rt["pct"] - ro["pct"] > 0.05:
            print("\n    THE PATH WINS THE DISAGREEMENTS. On this data"
                  " the contract")
            print("    weighting is costing you — when the book"
                  " and price differ, price is")
            print("    the one to follow, and the chart is built"
                  " around the other.")
        else:
            print("\n    NEITHER WINS THE DISAGREEMENTS. The size"
                  " weighting is not adding")
            print("    a direction the price path does not already"
                  " have, on this data.")

    if not real:
        print("\n  THIS WAS A CONTROL. Every figure above is what zero"
              " looks like on")
        print("  this many candles, and that is the only reason to run"
              " it without")
        print("  files. Point it at your .dbn files for an answer.")
    else:
        print("\n  Costs are in none of these numbers, and a candle"
              " called correctly is")
        print("  not a candle traded profitably — this scores"
              " direction only. An edge")
        print("  that does not survive on days this did not see was a"
              " property of")
        print("  these days.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
