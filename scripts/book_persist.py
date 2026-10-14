#!/usr/bin/env python3
"""Does a book lean that HOLDS predict better than one that just appeared?

    python scripts/book_persist.py data/ES_c_0_mbp-10_2026-10-08.dbn.zst
    python scripts/book_persist.py <file> --every 1 --json data/persist.json

RESEARCH ONLY. Runs `bookread` exactly as written. No indicator is added,
no weight is touched, no threshold is invented here, and nothing in the
trading system is modified.

WHY THIS FOLLOWS FROM THE LAST RESULT

`book_decay` found the instantaneous read carries information for about
two seconds and none after five. Two seconds is unreachable from a web
app with a person entering orders. So the question becomes whether a
DIFFERENT reading of the same book -- one that has held its direction for
a while -- lasts longer.

AND THE REASON IT MIGHT NOT

A lean that has persisted for thirty seconds is a book that keeps
re-forming the same way, which is harder to cancel out of and harder to
fake. That is the case for it.

The case against is in this project already, under the name absorption: a
lean the market has had thirty seconds to act on and HAS NOT is a lean
that is failing. Persistence may mark the reads that are not working
rather than the ones that are.

Both stories are plausible and they predict opposite things, which is
what makes this worth measuring instead of arguing about.

THE COUNT COMES FIRST, AND IT MAY END THE STUDY

Only about one read in eleven had a direction at all on the session this
was written for -- the book is flat most of the time. A lean has to stay
non-flat for thirty consecutive samples to land in the thirty-second
bucket, and there may be almost none of those. So every bucket prints its
size before its hit rate, and a bucket below the stated floor is labelled
as not a measurement rather than quietly reported as one.

THE FIVE-MINUTE CUT, AND ITS LIMIT

The second half splits the same readings by whether the book agrees with
what the current 5-minute bar has done so far.

Be clear about what that is: the 5-minute direction is DERIVED FROM
PRICE. Agreement with it is partly agreement with recent price, which is
the same lagging-confluence problem that showed up in the multi-timeframe
study. It is reported because it is cheap and because disagreement is the
interesting half -- a book leaning against a bar that has been running is
a different event from one leaning with it.

SELF-CONTAINED ON PURPOSE

The tape walk and the interval arithmetic are repeated here rather than
imported from `book_decay`. A research script that another research
script depends on is one edit away from changing a finished result.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from liqmap import dbn                                   # noqa: E402
from liqmap.bookread import BookReader                   # noqa: E402

HORIZONS = (1.0, 2.0, 5.0, 15.0, 30.0, 60.0)
TICKS = (5, 8, 10)
TICK = 0.25
MAX_HOLD_S = 300.0

# How long the lean has held, in seconds. The buckets are wide and few
# because the question is "does holding help", not "how much" -- narrow
# buckets on a thin sample would be reading noise with more decimals.
BUCKETS = ((0.0, 5.0), (5.0, 15.0), (15.0, 30.0), (30.0, 60.0),
           (60.0, float("inf")))

# Below this a bucket is reported as a count and nothing else. Thirty is
# not enough to conclude from either; it is the point below which a
# percentage is actively misleading.
MIN_BUCKET = 30

# The bar the second cut compares against.
SLOW_S = 300.0


@dataclass
class Look:
    ts: float
    side: str
    held_s: float              # how long this direction has been unbroken
    mid: float
    entry: float               # the price a crossing order would pay
    slow: str = ""             # what the 5m bar has done so far: up|down|flat
    moved: dict = field(default_factory=dict)
    reached: dict = field(default_factory=dict)

    @property
    def agrees(self) -> bool:
        return bool(self.slow) and self.slow != "flat" and (
            (self.slow == "up") == (self.side == "buy"))

    @property
    def against(self) -> bool:
        return bool(self.slow) and self.slow != "flat" and not self.agrees


@dataclass
class Tape:
    ts: list = field(default_factory=list)
    px: list = field(default_factory=list)

    def add(self, t: float, p: float) -> None:
        self.ts.append(t)
        self.px.append(p)

    def first_touch(self, after: float, target: float, stop: float,
                    long: bool, max_hold: float = MAX_HOLD_S) -> bool | None:
        """Target before stop, walked forward print by print.

        A single print cannot be on both sides at once, so there is no tie
        to resolve and nothing to guess in the tester's favour.
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
         flip: bool = False) -> tuple[list[Look], Tape, int]:
    """One pass. Reader fed on every push, sampled on the cadence.

    `held_s` is carried across samples: it grows while the direction is
    unchanged and resets the moment it flips OR goes flat. Flat counts as
    a break because a lean that lapsed and came back is not a lean that
    held -- treating it as one is how a thirty-second bucket fills up with
    events that never lasted five.
    """
    reader = BookReader()
    tape = Tape()
    looks: list[Look] = []
    book = None
    next_at = None
    reads = 0

    held_dir = ""
    held_since = 0.0

    # The 5m bar in progress, from the tape.
    bar_open_ts = 0.0
    bar_open_px = 0.0

    for kind, item in dbn.stream(records, flip=flip):
        if kind == "clear":
            continue
        if kind == "trade":
            tape.add(item.ts, item.px)
            slot = math.floor(item.ts / SLOW_S) * SLOW_S
            if slot != bar_open_ts:
                bar_open_ts, bar_open_px = slot, item.px
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
            held_dir, held_since = "", ts
            continue

        # NO THRESHOLD HERE. `BookRead.direction` applies the module's own
        # FLAT_BAND and answers up / down / flat.
        d = r.direction
        if d != held_dir:
            held_dir, held_since = d, ts
        if d == "flat" or book.empty:
            continue

        long = d == "up"
        entry = book.best_ask if long else book.best_bid
        if not entry or entry <= 0:
            continue

        slow = ""
        if bar_open_px > 0 and tape.px:
            now_px = tape.px[-1]
            slow = ("up" if now_px > bar_open_px
                    else "down" if now_px < bar_open_px else "flat")

        looks.append(Look(ts=ts, side="buy" if long else "sell",
                          held_s=ts - held_since, mid=book.mid,
                          entry=float(entry), slow=slow))
    return looks, tape, reads


def score(looks: list[Look], tape: Tape) -> None:
    ts_only = [l.ts for l in looks]
    for l in looks:
        long = l.side == "buy"
        for h in HORIZONS:
            j = bisect.bisect_left(ts_only, l.ts + h)
            if j >= len(looks):
                l.moved[h] = None
                continue
            later = looks[j].mid
            if later == l.mid:
                l.moved[h] = None          # went nowhere: neither
                continue
            l.moved[h] = (later > l.mid) if long else (later < l.mid)
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
    dd = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - r) / dd), min(1.0, (c + r) / dd)


def rate(vals: Sequence) -> tuple[int, int, int]:
    got = [v for v in vals if v is not None]
    return sum(1 for v in got if v), len(got), len(vals) - len(got)


def line(label: str, vals: Sequence, width: int = 10) -> str:
    hits, n, none = rate(vals)
    if n == 0:
        return f"  {label:<{width}} no resolved readings"
    lo, hi = wilson(hits, n)
    return (f"  {label:<{width}} {hits / n * 100:>5.1f}% "
            f"({lo * 100:>4.1f}-{hi * 100:>4.1f}%)  n={n:<6,} "
            f"unres={none:,}")


def report(name: str, pop: Sequence[Look]) -> dict:
    out: dict = {"n": len(pop)}
    print(f"\n--- {name} ---  {len(pop):,} readings")
    if len(pop) < MIN_BUCKET:
        print(f"  below {MIN_BUCKET} — a count, not a measurement. "
              f"Nothing reported.")
        out["reported"] = False
        return out
    out["reported"] = True
    print("  direction (cost ignored):")
    out["direction"] = {}
    for h in HORIZONS:
        print("  " + line(f"{h:.0f}s", [l.moved.get(h) for l in pop]))
        hits, n, none = rate([l.moved.get(h) for l in pop])
        out["direction"][str(h)] = {"hits": hits, "n": n,
                                    "rate": (hits / n) if n else None}
    print("  tradeable (entry crossed the spread):")
    out["ticks"] = {}
    for t in TICKS:
        print("  " + line(f"{t} ticks", [l.reached.get(t) for l in pop]))
        hits, n, none = rate([l.reached.get(t) for l in pop])
        out["ticks"][str(t)] = {"hits": hits, "n": n,
                                "rate": (hits / n) if n else None,
                                "unresolved": none}
    return out


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
        print("no readings with a direction", file=sys.stderr)
        return 1
    score(looks, tape)

    print(f"\n=== how long leans actually hold ===")
    print(f"  {reads:,} sampled reads, {len(looks):,} with a direction "
          f"({len(looks) / max(reads, 1):.0%})")
    print(f"  A lean breaks when it flips OR goes flat.")
    print(f"  {'held for':<14} {'readings':>10} {'share':>8}")
    buckets = []
    for lo, hi in BUCKETS:
        pop = [l for l in looks if lo <= l.held_s < hi]
        tag = f"{lo:.0f}-{hi:.0f}s" if hi != float("inf") else f"{lo:.0f}s+"
        print(f"  {tag:<14} {len(pop):>10,} "
              f"{len(pop) / max(len(looks), 1):>7.0%}")
        buckets.append((tag, pop))

    longest = max((l.held_s for l in looks), default=0.0)
    print(f"  longest unbroken lean in the session: {longest:.0f}s")

    print(f"\n=== does holding help? ===")
    out_b = {}
    for tag, pop in buckets:
        out_b[tag] = report(f"held {tag}", pop)

    print(f"\n=== the 5-minute cut ===")
    print(f"  The 5m direction is derived from PRICE, so agreement with it")
    print(f"  is partly agreement with recent price. Disagreement is the")
    print(f"  half worth looking at.")
    out_s = {}
    for tag, pop in (("agrees with the 5m", [l for l in looks if l.agrees]),
                     ("against the 5m", [l for l in looks if l.against])):
        out_s[tag] = report(tag, pop)

    print(f"\n=== held 30s+ AND against the 5m ===")
    out_s["held30_against"] = report(
        "held 30s+, against",
        [l for l in looks if l.held_s >= 30.0 and l.against])

    print("\n  Readings sampled seconds apart are NOT independent — the "
          "true")
    print("  intervals are much wider than these, and a bucket's readings "
          "may")
    print("  be a handful of episodes rather than that many events.")
    print("  Unresolved means neither level was reached in "
          f"{MAX_HOLD_S:.0f}s; never a loss.")
    print("  Queue position is not modelled: every entry is assumed to "
          "cross.")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"every_s": a.every, "reads": reads, "readings": len(looks),
             "longest_hold_s": longest, "buckets": out_b,
             "slow": out_s}, indent=2))
        print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
