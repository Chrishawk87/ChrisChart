#!/usr/bin/env python3
"""Does the book alone call the next few seconds on ES, and for how long?

    python scripts/book_decay.py data/ES_c_0_mbp-10_2026-10-08.dbn.zst
    python scripts/book_decay.py <file> --every 1 --json data/decay.json

RESEARCH ONLY. This runs `bookread` and `suggest` EXACTLY AS WRITTEN. No
indicator is added, no weight is touched, no parameter is tuned, and
nothing in the trading system is modified. It is a reader, not a change.

THE QUESTION, AND WHY IT IS ABOUT TIME

`bookread` answers "which way is this candle going, right now" from
resting orders alone -- microprice tilt, queue depletion, replenishment.
Whether that is TRUE is only half of what matters. The other half is how
long it stays true.

Microprice tilt is the canonical short-horizon book signal and it is
competed for at microsecond latency by people sitting next to the matching
engine. If what it sees decays inside a second, it can be entirely real
and still be unreachable from a web app with a human at the other end. If
it persists for thirty seconds, it is something a person can act on.

So the headline output is a DECAY CURVE: the same reading scored at
widening horizons. The shape of that curve decides whether this direction
is worth pursuing at all, and no amount of hit rate at one horizon answers
it.

TWO WAYS OF BEING RIGHT, KEPT APART

    direction   did the mid end up on the predicted side, h seconds later
    tradeable   from an entry that CROSSED THE SPREAD, did the target come
                before the stop

The first measures information and ignores cost. The second is the
business. They are reported separately because a signal can be clearly
informative and still lose money at a quarter-point spread, and collapsing
them hides which problem you have.

ENTRY PAYS THE SPREAD

A long enters at the ask and a short at the bid, because that is what
crossing costs. Targets and stops are measured from there, not from the
mid. Measuring from the mid would hand back half a tick that nobody ever
gets.

FIRST TOUCH COMES FROM THE TAPE

Not from bars. Every print in the file is kept, so "did price reach the
target" is answered by an actual trade at that price, in the order it
happened, to the nanosecond. There is no bucket to resolve a tie inside
and no high-or-low-first ambiguity to guess at.

WHAT IT CANNOT TELL YOU

Queue position. MBP-10 aggregates by price, so it cannot say where in a
level your order would sit, and nothing here models a passive fill. Every
entry is assumed to cross.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from liqmap import dbn                                   # noqa: E402
from liqmap import suggest as sg                         # noqa: E402
from liqmap.bookread import BookReader                   # noqa: E402
from liqmap.structure import Candle                      # noqa: E402

# Horizons for the decay curve, in seconds.
HORIZONS = (1.0, 2.0, 5.0, 15.0, 30.0, 60.0)

# Targets in ticks. The stop sits at the same distance.
TICKS = (5, 8, 10)

TICK = 0.25

# How long a trade is given to resolve before it is called unresolved.
# Unresolved is reported, never counted as a loss.
MAX_HOLD_S = 300.0

# NO GATE OF MY OWN. `BookRead.direction` already applies the module's
# `FLAT_BAND` and answers up / down / flat, so this script asks it rather
# than comparing the score against a number chosen here. A threshold
# invented in a research script is a different strategy wearing the same
# name.


@dataclass
class Look:
    """One sampled reading, and what happened after it."""

    ts: float
    side: str                  # buy | sell
    score: float
    mid: float
    entry: float               # the price a crossing order would pay
    spread_bps: float
    # direction, per horizon: True right, False wrong, None not enough file
    moved: dict = field(default_factory=dict)
    # tradeable, per tick target
    reached: dict = field(default_factory=dict)


@dataclass
class Tape:
    """Every print, for answering first touch exactly."""

    ts: list = field(default_factory=list)
    px: list = field(default_factory=list)

    def add(self, t: float, p: float) -> None:
        self.ts.append(t)
        self.px.append(p)

    def first_touch(self, after: float, target: float, stop: float,
                    long: bool, max_hold: float = MAX_HOLD_S) -> bool | None:
        """Did target come before stop? None if neither did in time.

        Walks prints in order from the first one strictly after `after`.
        A single print cannot be on both sides at once, so there is no tie
        to resolve and nothing to guess.
        """
        i = bisect.bisect_right(self.ts, after)
        end = after + max_hold
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
         flip: bool = False) -> tuple[list[Look], Counter, Tape, int]:
    """One pass: feed the reader every push, sample it every `every_s`.

    FED ON EVERY PUSH, SAMPLED ON A CADENCE. `bookread` is built to be
    called on every update and its dynamics -- depletion, replenishment --
    are accumulated from the stream. Feeding it only at the sample points
    would hand it a different and much poorer book history than it has
    live.
    """
    reader = BookReader()
    tape = Tape()
    looks: list[Look] = []
    gates: Counter = Counter()
    book = None
    next_at = None
    recent: list[Candle] = []
    reads = 0

    for kind, item in dbn.stream(records, flip=flip):
        if kind == "clear":
            continue
        if kind == "trade":
            tape.add(item.ts, item.px)
            continue
        book = item
        reader.add(book, now=book.ts)
        ts = book.ts

        if next_at is None:
            next_at = ts + every_s
            continue
        if ts < next_at:
            continue
        next_at = ts + every_s
        reads += 1

        r = reader.read(now=ts)
        if r is None:
            gates["no read"] += 1
            continue

        # The gate breakdown runs on EVERY sampled read, including the
        # ones the score gate below will drop, because "how often does
        # the full rule refuse, and why" is one of the two questions.
        call = sg.suggest(r, book, coin="ES", interval="1m",
                          interval_s=60.0, seconds_left=30.0,
                          recent=recent, tick=TICK,
                          require_confirmation=False, mode="scalp")
        gates[getattr(call, "gate", "TRADEABLE")] += 1

        if r.direction == "flat" or book.empty:
            continue
        long = r.direction == "up"
        entry = book.best_ask if long else book.best_bid
        if not entry or entry <= 0:
            continue
        looks.append(Look(ts=ts, side="buy" if long else "sell",
                          score=r.score, mid=book.mid, entry=float(entry),
                          spread_bps=r.spread_bps))
    return looks, gates, tape, reads


def score_looks(looks: list[Look], tape: Tape) -> None:
    """Fill in both kinds of outcome for every sampled reading."""
    mids = [(l.ts, l.mid) for l in looks]
    ts_only = [t for t, _ in mids]

    for l in looks:
        long = l.side == "buy"
        # -- direction, against the mid, cost ignored on purpose --------
        for h in HORIZONS:
            j = bisect.bisect_left(ts_only, l.ts + h)
            if j >= len(mids):
                l.moved[h] = None
                continue
            later = mids[j][1]
            if later == l.mid:
                l.moved[h] = None        # went nowhere: neither right nor wrong
                continue
            l.moved[h] = (later > l.mid) if long else (later < l.mid)

        # -- tradeable, from an entry that crossed the spread -----------
        for t in TICKS:
            d = t * TICK
            tgt = l.entry + d if long else l.entry - d
            stp = l.entry - d if long else l.entry + d
            l.reached[t] = tape.first_touch(l.ts, tgt, stp, long)


def wilson(hits: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    z = 1.959963985
    p = hits / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - r) / d), min(1.0, (c + r) / d)


def rate(vals: Sequence) -> tuple[int, int, int]:
    got = [v for v in vals if v is not None]
    return sum(1 for v in got if v), len(got), len(vals) - len(got)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--every", type=float, default=1.0,
                    help="seconds between sampled reads (default 1)")
    ap.add_argument("--flip", action="store_true",
                    help="invert the aggressor convention -- only if the "
                         "audit said to")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    looks: list[Look] = []
    gates: Counter = Counter()
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
        lk, gt, tp, n = walk(store, a.every, flip=a.flip)
        looks.extend(lk)
        gates.update(gt)
        tape.ts.extend(tp.ts)
        tape.px.extend(tp.px)
        reads += n

    if not looks:
        print("no readings taken", file=sys.stderr)
        return 1

    print(f"  {len(tape.ts):,} prints, {reads:,} sampled reads")
    score_looks(looks, tape)

    longs = sum(1 for l in looks if l.side == "buy")
    print(f"\n=== the readings ===")
    print(f"  sampled every        {a.every:g}s")
    print(f"  direction gate       BookRead.direction != flat "
          f"(bookread's own FLAT_BAND)")
    print(f"  readings with a side {len(looks):,} of {reads:,} "
          f"({len(looks) / max(reads, 1):.0%})")
    print(f"  split                {longs:,} buy, {len(looks) - longs:,} sell")
    sp = sorted(l.spread_bps for l in looks)
    if sp:
        print(f"  median spread        {sp[len(sp) // 2]:.3f} bps")

    print(f"\n=== decay: is the mid on the predicted side later? ===")
    print(f"  (cost ignored -- this measures information, not profit)")
    print(f"  {'horizon':<10} {'right':<22} {'n':>8} {'flat':>8}")
    out_dir = {}
    for h in HORIZONS:
        hits, n, none = rate([l.moved.get(h) for l in looks])
        if not n:
            print(f"  {h:>5.0f}s     no resolved readings")
            continue
        lo, hi = wilson(hits, n)
        print(f"  {h:>5.0f}s     {hits / n * 100:>5.1f}% "
              f"({lo * 100:>4.1f}-{hi * 100:>4.1f}%)   {n:>8,} {none:>8,}")
        out_dir[str(h)] = {"hits": hits, "n": n, "rate": hits / n,
                           "lo": lo, "hi": hi, "unresolved": none}

    print(f"\n=== tradeable: target before stop, entry crossed the spread ===")
    print(f"  {'target':<10} {'hit':<22} {'n':>8} {'unresolved':>11}")
    out_tick = {}
    for t in TICKS:
        hits, n, none = rate([l.reached.get(t) for l in looks])
        if not n:
            print(f"  {t} ticks     no resolved trades")
            continue
        lo, hi = wilson(hits, n)
        print(f"  {t} ticks    {hits / n * 100:>5.1f}% "
              f"({lo * 100:>4.1f}-{hi * 100:>4.1f}%)   {n:>8,} {none:>11,}")
        out_tick[str(t)] = {"hits": hits, "n": n, "rate": hits / n,
                            "lo": lo, "hi": hi, "unresolved": none}

    print(f"\n=== what the full rule did with the same reads ===")
    print(f"  `suggest` as written, scalp mode. A refusal names its gate.")
    total = sum(gates.values())
    for gate, n in gates.most_common():
        print(f"  {gate:<16} {n:>8,}  {n / max(total, 1):>5.1%}")

    print("\n  Entry pays the spread: a long enters at the ask, a short at "
          "the bid.")
    print("  First touch comes from the tape, print by print, so there is "
          "no bar")
    print("  to resolve a tie inside. Unresolved means neither level was "
          "reached")
    print(f"  within {MAX_HOLD_S:.0f}s; it is never counted as a loss.")
    print("  Sampled readings seconds apart are NOT independent, so the "
          "true")
    print("  intervals are wider than these.")
    print("  Queue position is not modelled: every entry is assumed to "
          "cross.")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"every_s": a.every, "reads": reads, "readings": len(looks),
             "direction": out_dir, "ticks": out_tick,
             "gates": dict(gates)}, indent=2))
        print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
