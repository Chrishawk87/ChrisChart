#!/usr/bin/env python3
"""Why is the DOM layer silent on ES? Measure it before changing it.

    python scripts/dom_probe.py data/ES_c_0_mbp-10_2026-10-08.dbn.zst
    python scripts/dom_probe.py <file> --every 5 --json data/dom.json

THE QUESTION

`stack_study` found the book had an opinion on 5 signals out of 27. A
layer that speaks one time in five cannot contribute much whether or not
it is right, so the thing to establish is WHY it is quiet:

    the read comes back unmeasured        a plumbing problem -- not enough
                                          overlapping price levels between
                                          snapshots to compute anything

    it is measured, and the number is     a measurement problem -- the
    simply small                          threshold is asking for something
                                          this market does not do

Those want opposite fixes, and guessing between them is how a threshold
gets tuned until signals appear.

WHAT THIS DELIBERATELY DOES NOT DO

It does not pick a threshold. Choosing the cut that produces the most
signals is fitting the knob to the answer, and a knob fitted that way
will keep working beautifully right up until it is used on data it has
not seen.

What it prints is the DISTRIBUTION -- so the question becomes "does this
measure vary enough to separate anything", which the data can answer, and
"where should the line go", which only an out-of-sample test can.

A measure pinned at one value across the whole session cannot discriminate
at any threshold. That is the single most useful thing this can tell you,
and it costs nothing to find out.

SEVERAL CANDIDATE MEASURES, FOR THE SAME REASON

`pull_share` is a RATIO of what left: cancelled over cancelled-plus-traded.
On a market where both sides churn quotes constantly, both ratios sit near
the top of their range and the difference between them is structurally
tiny -- not because nothing is happening, but because the measure has no
room left. So three other formulations are reported beside it. They are
reported, not ranked: whichever looks usable still has to earn it against
outcomes on data this script never saw.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from liqmap import dbn                                  # noqa: E402
from liqmap.dom import DomWatch, MIN_OVERLAP            # noqa: E402

# Windows to try. The read's own default is 20s; a longer one accumulates
# more change and may simply have more to see, which is a parameter the
# data can settle rather than an opinion.
WINDOWS = (5.0, 10.0, 20.0, 60.0)

# Thresholds the coverage table walks. Printed as a curve, NOT as a
# recommendation -- see the module docstring.
CUTS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.25, 0.40)

TICK = 0.25
LONGEST = max(WINDOWS)


def pct(xs: Sequence[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def spread_of(xs: Sequence[float]) -> str:
    """Percentiles, because a mean hides whether anything varies at all."""
    if not xs:
        return "no samples"
    return (f"p5 {pct(xs, .05):+.3f}  p25 {pct(xs, .25):+.3f}  "
            f"med {pct(xs, .50):+.3f}  p75 {pct(xs, .75):+.3f}  "
            f"p95 {pct(xs, .95):+.3f}")


@dataclass
class Bucket:
    """Everything observed at one window length."""

    window_s: float
    n: int = 0
    measured: int = 0
    bid_only: int = 0
    ask_only: int = 0
    overlaps: list = field(default_factory=list)

    # the four candidate measures, signed so that + favours up
    pull_edge: list = field(default_factory=list)     # ask.pull - bid.pull
    cancel_edge: list = field(default_factory=list)   # cancels, over churn
    net_edge: list = field(default_factory=list)      # resting size change
    depth_edge: list = field(default_factory=list)    # size at the touch

    bid_pull: list = field(default_factory=list)
    ask_pull: list = field(default_factory=list)
    ahead: int = 0
    untested: int = 0

    def observe(self, r: Any, book: Any) -> None:
        self.n += 1
        self.overlaps.append(min(r.bid.overlap, r.ask.overlap))
        if r.bid.measured:
            self.bid_only += 1
        if r.ask.measured:
            self.ask_only += 1
        if not r.measured:
            return
        self.measured += 1

        self.bid_pull.append(r.bid.pull_share)
        self.ask_pull.append(r.ask.pull_share)
        self.pull_edge.append(r.ask.pull_share - r.bid.pull_share)

        churn = r.bid.churn + r.ask.churn
        if churn > 0:
            self.cancel_edge.append(
                (r.ask.cancelled - r.bid.cancelled) / churn)
            self.net_edge.append((r.bid.net - r.ask.net) / churn)
        if book is not None and not book.empty and book.bids and book.asks:
            b, a = book.bids[0].sz, book.asks[0].sz
            if b + a > 0:
                self.depth_edge.append((b - a) / (b + a))
        if r.vanishing_ahead > 0:
            self.ahead += 1
        if r.pulled_untested > 0:
            self.untested += 1

    def coverage(self, xs: Sequence[float], cut: float) -> float:
        """Share of ALL samples that would speak at this threshold.

        Of all samples, not of measured ones. A measure that only exists
        a third of the time cannot cover more than a third however
        generous the cut, and reporting it against the measured subset
        would hide that.
        """
        if not self.n:
            return 0.0
        return sum(1 for x in xs if abs(x) >= cut) / self.n

    def to_dict(self) -> dict:
        return {"window_s": self.window_s, "samples": self.n,
                "measured": self.measured,
                "measured_share": (self.measured / self.n) if self.n else 0.0,
                "median_overlap": pct(self.overlaps, .5),
                "pull_edge": {str(c): self.coverage(self.pull_edge, c)
                              for c in CUTS},
                "cancel_edge": {str(c): self.coverage(self.cancel_edge, c)
                                for c in CUTS},
                "net_edge": {str(c): self.coverage(self.net_edge, c)
                             for c in CUTS},
                "depth_edge": {str(c): self.coverage(self.depth_edge, c)
                               for c in CUTS}}


def probe(records: Iterable[Any], every_s: float,
          flip: bool = False) -> dict[float, Bucket]:
    """Walk the file, reading the book every `every_s` at each window."""
    buckets = {w: Bucket(window_s=w) for w in WINDOWS}
    dom = DomWatch(window_s=LONGEST, max_snaps=int(LONGEST / 0.25) + 40)
    recent: deque = deque()
    book = None
    next_at = None

    for kind, item in dbn.stream(records, flip=flip):
        if kind == "clear":
            continue
        if kind == "book":
            book = item
            dom.add(book)
            ts = book.ts
        else:
            ts = item.ts
            recent.append(item)
            while recent and recent[0].ts < ts - LONGEST:
                recent.popleft()

        if next_at is None:
            # Start a full window in, so the first reads are not short
            # simply because the file had only just begun.
            next_at = ts + LONGEST
            continue
        if ts < next_at:
            continue
        next_at = ts + every_s
        for w, b in buckets.items():
            r = dom.read(trades=[t for t in recent if t.ts >= ts - w],
                         window_s=w, tick=TICK, now=ts)
            b.observe(r, book)
    return buckets


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--every", type=float, default=10.0,
                    help="seconds between reads (default 10)")
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    buckets: dict[float, Bucket] = {}
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
        got = probe(store, a.every, flip=a.flip)
        for w, b in got.items():
            if w not in buckets:
                buckets[w] = b
            else:
                old = buckets[w]
                old.n += b.n
                old.measured += b.measured
                old.bid_only += b.bid_only
                old.ask_only += b.ask_only
                old.ahead += b.ahead
                old.untested += b.untested
                for name in ("overlaps", "pull_edge", "cancel_edge",
                             "net_edge", "depth_edge", "bid_pull", "ask_pull"):
                    getattr(old, name).extend(getattr(b, name))

    print("\n=== is it measured at all? ===")
    print(f"  (a side needs {MIN_OVERLAP} overlapping prices between "
          f"snapshots to say anything)")
    print(f"  {'window':<9} {'reads':>7} {'measured':>10} "
          f"{'bid side':>10} {'ask side':>10} {'median overlap':>15}")
    for w in sorted(buckets):
        b = buckets[w]
        share = (b.measured / b.n) if b.n else 0.0
        print(f"  {w:>5.0f}s   {b.n:>7,} {share:>9.0%} "
              f"{(b.bid_only / b.n if b.n else 0):>9.0%} "
              f"{(b.ask_only / b.n if b.n else 0):>9.0%} "
              f"{pct(b.overlaps, .5):>15.0f}")

    base = buckets.get(20.0) or buckets[sorted(buckets)[0]]
    print("\n=== how much room does each side's pull share have? ===")
    print("  pull_share is cancelled / (cancelled + traded) -- a RATIO of")
    print("  what left. Both sides pinned near the top means the "
          "DIFFERENCE")
    print("  between them has nowhere to live, whatever the threshold.")
    print(f"  bid  {spread_of(base.bid_pull)}")
    print(f"  ask  {spread_of(base.ask_pull)}")

    print("\n=== the candidate measures, at 20s ===  (+ favours up)")
    for name, xs, what in (
            ("pull_edge  ", base.pull_edge,
             "ask.pull_share - bid.pull_share   (what is used today)"),
            ("cancel_edge", base.cancel_edge,
             "(ask.cancelled - bid.cancelled) / total churn"),
            ("net_edge   ", base.net_edge,
             "(bid.net - ask.net) / total churn"),
            ("depth_edge ", base.depth_edge,
             "(bid size - ask size) at the touch")):
        print(f"  {name}  {what}")
        print(f"               {spread_of(xs)}")

    print("\n=== how often each measure would speak ===")
    print("  Share of ALL reads, per threshold. This is a curve, NOT a")
    print("  recommendation: picking the cut that yields the most signals")
    print("  is fitting the knob to the answer.")
    hdr = "  ".join(f"{c:>5.2f}" for c in CUTS)
    print(f"  {'window':<8} {'measure':<12} {hdr}")
    out: dict = {}
    for w in sorted(buckets):
        b = buckets[w]
        for name, xs in (("pull_edge", b.pull_edge),
                         ("cancel_edge", b.cancel_edge),
                         ("net_edge", b.net_edge),
                         ("depth_edge", b.depth_edge)):
            row = "  ".join(f"{b.coverage(xs, c):>5.0%}" for c in CUTS)
            print(f"  {w:>5.0f}s   {name:<12} {row}")
        out[str(w)] = b.to_dict()

    print(f"\n=== other things the read noticed (20s) ===")
    if base.n:
        print(f"  size withdrawn ahead of price   "
              f"{base.ahead / base.n:.0%} of reads")
        print(f"  size that never traded at all   "
              f"{base.untested / base.n:.0%} of reads")

    print("\n  A measure whose percentiles barely move cannot separate "
          "anything")
    print("  at any threshold. One that varies has earned a test against "
          "outcomes")
    print("  -- on sessions this probe has not seen.")

    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
