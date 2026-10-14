#!/usr/bin/env python3
"""Is the book adding anything, or is ES just mean-reverting on the 5m?

    python scripts/book_control.py data/ES_c_0_mbp-10_2026-10-08.dbn.zst
    python scripts/book_control.py <file> --every 1 --json data/control.json

RESEARCH ONLY. Runs `bookread` and `dom` exactly as written. No indicator
is added, no weight is touched, no threshold is chosen here, and nothing
in the trading system is modified.

THE QUESTION THIS SETTLES

`book_persist` found that readings where the book leaned AGAINST the
current 5-minute bar held 55% out to sixty seconds, while readings that
agreed with it decayed to 47%. Two confidence intervals that did not
overlap, on one session.

That result has one obvious alternative explanation, and it has never
been tested: **ES may simply mean-revert on the 5-minute, with the book
along for the ride.** If fading the bar WITHOUT CONSULTING THE BOOK AT
ALL scores the same, then nothing was learned about order flow -- a price
pattern was found, and it needs no depth data to trade.

So the blind fade runs beside the book arm, at the same timestamps, scored
the same way. If the book arm does not beat it, the book is decoration.

AND THE SECOND QUESTION, WHICH IS NEW

Of `bookread`'s score, about 69% comes from the touch, 14% from the mid,
and only 17% from depth beyond it -- five levels, lightly weighted. The
module that reads all ten levels properly is `dom`, and the measure it
currently leans on (`pull_share`) was found saturated: pinned at 1.0 on
both sides, speaking one time in six.

So the depth gets its own arms here, through `dom`'s per-level adds, fills
and cancels. This is the first test that uses what the ten levels actually
contain.

NO THRESHOLD IS PICKED

The depth measures are continuous, so a cut has to be applied somewhere.
Rather than choose one -- which would be fitting the knob to the answer --
every depth arm is reported at THREE cuts, as a row. If the arm only works
at one of them it is noise, and showing the whole row is what makes that
visible.

EVERY ARM IS SCORED IDENTICALLY

Same timestamps, same entry that crosses the spread, same first touch from
the tape, same horizons. Differences between arms cannot come from
differences in method.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from liqmap import dbn                                   # noqa: E402
from liqmap.bookread import BookReader                   # noqa: E402
from liqmap.dom import DomWatch                          # noqa: E402

HORIZONS = (1.0, 5.0, 15.0, 30.0, 60.0)
TICKS = (5, 8, 10)
TICK = 0.25
MAX_HOLD_S = 300.0
SLOW_S = 300.0                 # the 5-minute bar
DOM_WINDOW_S = 20.0

# Cuts for the continuous depth measures. THREE, reported as a row,
# because choosing one is fitting the knob to the answer.
CUTS = (0.10, 0.25, 0.40)

# Below this an arm is a count, not a rate.
MIN_ARM = 30


@dataclass
class Look:
    """One sampled moment, with every arm's opinion of it."""

    ts: float
    mid: float
    bid: float
    ask: float
    slow: str = ""             # what the 5m bar has done so far
    book: str = ""             # bookread's direction, as a side
    cancel_edge: float | None = None   # + favours up, from 10 levels
    depth_edge: float | None = None    # + favours up, near-book notional

    @property
    def fade(self) -> str:
        """The blind control: oppose the 5m bar, book not consulted."""
        if self.slow == "up":
            return "sell"
        if self.slow == "down":
            return "buy"
        return ""

    @property
    def follow(self) -> str:
        """The other blind baseline, free with the same walk."""
        if self.slow == "up":
            return "buy"
        if self.slow == "down":
            return "sell"
        return ""

    def edge_side(self, which: str, cut: float) -> str:
        """Which way a depth measure leans, or nothing below the cut.

        Named explicitly rather than selected by an else-branch: a typo in
        the caller would otherwise silently score one measure twice and
        report it under two labels.
        """
        v = {"cancel": self.cancel_edge, "notional": self.depth_edge}[which]
        if v is None or abs(v) < cut:
            return ""
        return "buy" if v > 0 else "sell"


@dataclass
class Tape:
    ts: list = field(default_factory=list)
    px: list = field(default_factory=list)

    def add(self, t: float, p: float) -> None:
        self.ts.append(t)
        self.px.append(p)

    def first_touch(self, after: float, target: float, stop: float,
                    long: bool) -> bool | None:
        i = bisect.bisect_right(self.ts, after)
        end = after + MAX_HOLD_S
        n = len(self.ts)
        while i < n and self.ts[i] <= end:
            p = self.px[i]
            if long:
                if p >= target:
                    return True
                if p <= stop:
                    return False
            else:
                if p <= target:
                    return True
                if p >= stop:
                    return False
            i += 1
        return None


def walk(records: Iterable[Any], every_s: float,
         flip: bool = False) -> tuple[list[Look], Tape, int]:
    """One pass. Both readers fed on every push, sampled on the cadence."""
    reader = BookReader()
    dom = DomWatch(window_s=DOM_WINDOW_S)
    tape = Tape()
    looks: list[Look] = []
    recent: deque = deque()
    book = None
    next_at = None
    reads = 0
    # None, not 0.0. A session whose first trade falls in the bar
    # starting at timestamp zero would otherwise find slot == bar_ts on
    # the very first print, never record an open, and report every 5m
    # direction as empty for the whole run.
    bar_ts: float | None = None
    bar_px = 0.0

    for kind, item in dbn.stream(records, flip=flip):
        if kind == "clear":
            continue
        if kind == "trade":
            tape.add(item.ts, item.px)
            recent.append(item)
            while recent and recent[0].ts < item.ts - DOM_WINDOW_S:
                recent.popleft()
            slot = math.floor(item.ts / SLOW_S) * SLOW_S
            if slot != bar_ts:
                bar_ts, bar_px = slot, item.px
            continue

        book = item
        reader.add(book, now=book.ts)
        dom.add(book)
        ts = book.ts

        if next_at is None:
            next_at = ts + every_s
            continue
        if ts < next_at:
            continue
        next_at = ts + every_s
        reads += 1

        if book.empty or not book.bids or not book.asks:
            continue
        bid, ask = book.best_bid, book.best_ask
        if not bid or not ask or bid <= 0 or ask <= 0:
            continue

        slow = ""
        if bar_px > 0 and tape.px:
            now_px = tape.px[-1]
            slow = ("up" if now_px > bar_px
                    else "down" if now_px < bar_px else "flat")

        side = ""
        r = reader.read(now=ts)
        if r is not None and r.direction != "flat":
            side = "buy" if r.direction == "up" else "sell"

        d = dom.read(trades=list(recent), window_s=DOM_WINDOW_S,
                     tick=TICK, now=ts)
        c_edge = n_edge = None
        if d.measured:
            churn = d.bid.churn + d.ask.churn
            if churn > 0:
                # Offers being cancelled is the book getting out of the
                # way above; bids being cancelled is the floor coming out.
                c_edge = (d.ask.cancelled - d.bid.cancelled) / churn
        nb = sum(l.notional for l in book.bids[:10])
        na = sum(l.notional for l in book.asks[:10])
        if nb + na > 0:
            n_edge = (nb - na) / (nb + na)

        looks.append(Look(ts=ts, mid=book.mid, bid=float(bid),
                          ask=float(ask), slow=slow, book=side,
                          cancel_edge=c_edge, depth_edge=n_edge))
    return looks, tape, reads


def score_arm(pop: Sequence[tuple[Look, str]], tape: Tape,
              mids: list[tuple[float, float]]) -> dict:
    """Outcomes for one arm: (look, side) pairs, scored identically."""
    ts_only = [t for t, _ in mids]
    moved: dict = {h: [] for h in HORIZONS}
    reach: dict = {t: [] for t in TICKS}

    for l, side in pop:
        long = side == "buy"
        for h in HORIZONS:
            j = bisect.bisect_left(ts_only, l.ts + h)
            if j >= len(mids):
                moved[h].append(None)
                continue
            later = mids[j][1]
            if later == l.mid:
                moved[h].append(None)
                continue
            moved[h].append((later > l.mid) if long else (later < l.mid))
        entry = l.ask if long else l.bid
        for t in TICKS:
            d = t * TICK
            tgt = entry + d if long else entry - d
            stp = entry - d if long else entry + d
            reach[t].append(tape.first_touch(l.ts, tgt, stp, long))
    return {"moved": moved, "reach": reach}


def wilson(hits: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    z = 1.959963985
    p = hits / n
    dd = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - r) / dd), min(1.0, (c + r) / dd)


def rate(vals: Sequence) -> tuple[int, int, int]:
    got = [v for v in vals if v is not None]
    return sum(1 for v in got if v), len(got), len(vals) - len(got)


def cell(vals: Sequence) -> str:
    hits, n, _ = rate(vals)
    if n == 0:
        return "    —      "
    lo, hi = wilson(hits, n)
    return f"{hits / n * 100:>5.1f}({lo * 100:.0f}-{hi * 100:.0f})"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--every", type=float, default=1.0)
    ap.add_argument("--flip", action="store_true",
                    help="invert the aggressor convention -- only if the "
                         "audit said to")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    looks: list[Look] = []
    tape = Tape()
    reads = 0
    for f in a.files:
        p = Path(f)
        if not p.exists():
            print(f"no such file: {p}", file=sys.stderr)
            return 2
        print(f"Parsing {p.name} ...", flush=True)
        try:
            store = dbn.open_file(str(p))
        except ImportError:
            print("pip install databento", file=sys.stderr)
            return 2
        lk, tp, n = walk(store, a.every, flip=a.flip)
        looks.extend(lk)
        tape.ts.extend(tp.ts)
        tape.px.extend(tp.px)
        reads += n

    if not looks:
        print("no readings", file=sys.stderr)
        return 1
    mids = [(l.ts, l.mid) for l in looks]

    # -- the arms ---------------------------------------------------------
    arms: list[tuple[str, list]] = []

    arms.append(("fade the 5m (blind)",
                 [(l, l.fade) for l in looks if l.fade]))
    arms.append(("follow the 5m (blind)",
                 [(l, l.follow) for l in looks if l.follow]))
    arms.append(("book, all",
                 [(l, l.book) for l in looks if l.book]))
    arms.append(("book against the 5m",
                 [(l, l.book) for l in looks
                  if l.book and l.fade and l.book == l.fade]))
    arms.append(("book with the 5m",
                 [(l, l.book) for l in looks
                  if l.book and l.follow and l.book == l.follow]))
    for cut in CUTS:
        arms.append((f"depth cancels @{cut:.2f}",
                     [(l, l.edge_side("cancel", cut)) for l in looks
                      if l.edge_side("cancel", cut)]))
    for cut in CUTS:
        arms.append((f"depth notional @{cut:.2f}",
                     [(l, l.edge_side("notional", cut)) for l in looks
                      if l.edge_side("notional", cut)]))

    print(f"\n=== the walk ===")
    print(f"  {reads:,} sampled reads every {a.every:g}s, "
          f"{len(looks):,} with a usable book")
    print(f"  {len(tape.ts):,} prints")
    told = sum(1 for l in looks if l.cancel_edge is not None)
    print(f"  depth cancels measurable on {told:,} "
          f"({told / max(len(looks), 1):.0%})")

    print(f"\n=== direction: is the mid on the called side later? ===")
    print(f"  (cost ignored. % with 95% interval)")
    head = "  ".join(f"{h:>5.0f}s      " for h in HORIZONS)
    print(f"  {'arm':<24} {'n':>7}  {head}")
    out: dict = {}
    for name, pop in arms:
        if len(pop) < MIN_ARM:
            print(f"  {name:<24} {len(pop):>7,}  "
                  f"below {MIN_ARM} — a count, not a measurement")
            out[name] = {"n": len(pop), "reported": False}
            continue
        s = score_arm(pop, tape, mids)
        row = "  ".join(cell(s["moved"][h]) for h in HORIZONS)
        print(f"  {name:<24} {len(pop):>7,}  {row}")
        out[name] = {"n": len(pop), "reported": True,
                     "direction": {str(h): rate(s["moved"][h])
                                   for h in HORIZONS},
                     "ticks": {str(t): rate(s["reach"][t]) for t in TICKS}}

    print(f"\n=== tradeable: target before stop, entry crossed the spread ===")
    head = "  ".join(f"{t:>3} ticks    " for t in TICKS)
    print(f"  {'arm':<24} {'n':>7}  {head}")
    for name, pop in arms:
        if len(pop) < MIN_ARM:
            continue
        s = score_arm(pop, tape, mids)
        row = "  ".join(cell(s["reach"][t]) for t in TICKS)
        print(f"  {name:<24} {len(pop):>7,}  {row}")

    print("\n  THE COMPARISON THAT MATTERS is 'book against the 5m' versus")
    print("  'fade the 5m (blind)'. If they are the same, the book added")
    print("  nothing and this is a price pattern that needs no depth data.")
    print("  Depth arms are shown at three cuts. An arm that only works at")
    print("  one of them is noise.")
    print("  Readings seconds apart are NOT independent — true intervals "
          "are")
    print("  wider, and the gap BETWEEN two arms is more trustworthy than")
    print("  either arm against 50%, because both carry the same inflation.")
    print("  Unresolved is never counted as a loss. Queue position is not")
    print("  modelled: every entry is assumed to cross.")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"every_s": a.every, "reads": reads, "looks": len(looks),
             "cuts": list(CUTS), "arms": out}, indent=2))
        print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
