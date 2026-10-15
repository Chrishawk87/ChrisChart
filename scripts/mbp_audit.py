#!/usr/bin/env python3
"""What the MBP-10 file actually contains, and what the engine throws away.

    python scripts/mbp_audit.py data/ES_c_0_mbp-10_2026-10-08.dbn.zst

THE QUESTION THIS EXISTS TO SETTLE

`dbn.stream` reduces every record to one of three things: a clear, a trade,
or a book snapshot. The `action` field -- A add, C cancel, M modify, T trade,
F fill, R clear -- is read once to decide which, and then discarded. So the
whole engine downstream has never seen a cancellation. `dom.py` recovers
add/cancel by DIFFERENCING consecutive snapshots and subtracting what
printed at each price:

    size that left and printed      -> traded
    size that left and did not      -> cancelled

That is a sound inference and it is not the same measurement. Two orders
that arrive and leave between snapshots net to nothing. A requote at the
same price nets to nothing. The file knows about both; the difference does
not. It is a live hypothesis that this is why `pull_share` saturates at 1.0
on both sides of ES -- an artifact of netting rather than a fact about the
market -- and the only way to find out is to read the actions and compare.

So this counts what is in the file, then puts the two measurements side by
side over the same windows and the same price span.

WHY THE PRICE SPAN MATTERS

`dom.py` only compares prices BOTH snapshots could see, because a price that
left the ten-level window was not cancelled, it stopped being reported.
Comparing its inference against every cancel in the file would charge it for
events it deliberately declines to judge. So the true figure is restricted
to the same overlap before the two are compared.

NOTHING HERE READS A SIGNAL OR SCORES AN OUTCOME. It counts records.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap import dbn                                      # noqa: E402
from liqmap.dom import ladder_of, read as dom_read, volume_by_price  # noqa: E402,E501

WINDOW_S = 20.0            # the window dom_probe and book_control both used
TICK = 0.25
MAX_GAP_REPORT = 10
HEARTBEAT = 500_000        # records between progress lines


class Progress:
    """A heartbeat on stderr while a long pass runs.

    Eight million records take minutes and this printed nothing until the
    pass finished, which is indistinguishable from a hang -- so the first
    thing anyone does is kill it. Goes to stderr so a piped stdout still
    holds nothing but the report.
    """

    def __init__(self, label: str, every: int = HEARTBEAT,
                 out=None) -> None:
        self.label = label
        self.every = max(1, every)
        self.n = 0
        self.out = out if out is not None else sys.stderr
        self.start = time.time()

    def tick(self) -> None:
        self.n += 1
        if self.n % self.every:
            return
        el = max(1e-9, time.time() - self.start)
        print(f"  [{self.label}] {self.n:,} records  {el:,.0f}s  "
              f"{self.n / el:,.0f}/s", file=self.out, flush=True)

    def done(self) -> None:
        el = max(1e-9, time.time() - self.start)
        print(f"  [{self.label}] done: {self.n:,} records in {el:,.0f}s",
              file=self.out, flush=True)


@dataclass
class Census:
    """Counts only. No opinion about what any of it means."""

    records: int = 0
    by_action: Counter = field(default_factory=Counter)
    by_action_side: Counter = field(default_factory=Counter)
    size_by_action: Counter = field(default_factory=Counter)
    instruments: Counter = field(default_factory=Counter)
    symbols: Counter = field(default_factory=Counter)

    ts_first: float = 0.0
    ts_last: float = 0.0
    out_of_order: int = 0
    duplicate_ts: int = 0
    gaps: list = field(default_factory=list)

    crossed: int = 0
    empty_book: int = 0
    level_counts: Counter = field(default_factory=Counter)
    undef_px: int = 0

    def add_gap(self, a: float, b: float) -> None:
        self.gaps.append((b - a, a, b))
        self.gaps.sort(reverse=True)
        del self.gaps[MAX_GAP_REPORT:]


def census(records, gap_s: float = 1.0, progress: Progress | None = None
           ) -> Census:
    c = Census()
    prev_ts = None
    for rec in records:
        if progress is not None:
            progress.tick()
        c.records += 1
        act = dbn._action(rec) or "?"
        side = dbn._side(rec)
        sz = float(dbn._get(rec, "size", 0) or 0)
        c.by_action[act] += 1
        c.by_action_side[(act, side)] += 1
        c.size_by_action[act] += sz

        iid = dbn._get(rec, "instrument_id")
        if iid is not None:
            c.instruments[iid] += 1
        sym = dbn._get(rec, "symbol") or dbn._get(rec, "raw_symbol")
        if sym:
            c.symbols[str(sym)] += 1

        ts = dbn.seconds(dbn._get(rec, "ts_event"))
        if c.ts_first == 0.0:
            c.ts_first = ts
        c.ts_last = max(c.ts_last, ts)
        if prev_ts is not None:
            if ts < prev_ts:
                c.out_of_order += 1
            elif ts == prev_ts:
                c.duplicate_ts += 1
            elif ts - prev_ts >= gap_s:
                c.add_gap(prev_ts, ts)
        prev_ts = ts

        levels = dbn._get(rec, "levels") or []
        if levels:
            c.level_counts[len(levels)] += 1
            bpx = dbn.price(dbn._get(levels[0], "bid_px"))
            apx = dbn.price(dbn._get(levels[0], "ask_px"))
            if bpx <= 0 or apx <= 0:
                c.undef_px += 1
            elif bpx >= apx:
                c.crossed += 1
        else:
            c.empty_book += 1
    if progress is not None:
        progress.done()
    return c


# ------------------------------------------- inferred cancels vs real ones

@dataclass
class Compare:
    windows: int = 0
    true_cancel: float = 0.0
    true_add: float = 0.0
    true_trade: float = 0.0
    inferred_cancel: float = 0.0
    inferred_add: float = 0.0
    inferred_trade: float = 0.0
    pairs: list = field(default_factory=list)   # (true, inferred) per window

    @property
    def cancel_ratio(self) -> float | None:
        if self.true_cancel <= 0:
            return None
        return self.inferred_cancel / self.true_cancel


def _span(lad, side: str) -> tuple[float, float]:
    return lad.span(side)


def compare(records, window_s: float = WINDOW_S, depth: int = 10,
            symbol: str = "ES", progress: Progress | None = None) -> Compare:
    """Walk in windows; at each boundary measure the same thing both ways.

    The inference is `dom.read` exactly as production calls it. The truth is
    the file's own A and C actions, summed over the SAME window and clipped
    to the SAME overlapping price span, because that is all `dom.read`
    claims to judge.
    """
    out = Compare()
    old_lad = None
    old_ts = None
    pending_cancel: dict[str, dict[float, float]] = {"bid": defaultdict(float),
                                                     "ask": defaultdict(float)}
    pending_add: dict[str, dict[float, float]] = {"bid": defaultdict(float),
                                                  "ask": defaultdict(float)}
    trades: list = []

    for rec in records:
        if progress is not None:
            progress.tick()
        act = dbn._action(rec)
        side = dbn._side(rec)
        sz = float(dbn._get(rec, "size", 0) or 0)
        px = dbn.price(dbn._get(rec, "price"))
        ts = dbn.seconds(dbn._get(rec, "ts_event"))
        which = "bid" if side == dbn.SIDE_BID else (
            "ask" if side == dbn.SIDE_ASK else "")

        if act == "C" and which and px > 0:
            pending_cancel[which][px] += sz
        elif act == "A" and which and px > 0:
            pending_add[which][px] += sz

        t = dbn.trade_from(rec)
        if t is not None:
            trades.append(t)

        if not (dbn._get(rec, "levels")):
            continue
        book = dbn.book_from(rec, symbol=symbol, depth=depth)
        lad = ladder_of(book, depth, ts)
        if lad is None:
            continue

        if old_lad is None:
            old_lad, old_ts = lad, ts
            pending_cancel = {"bid": defaultdict(float),
                              "ask": defaultdict(float)}
            pending_add = {"bid": defaultdict(float),
                           "ask": defaultdict(float)}
            trades = []
            continue

        if ts - old_ts < window_s:
            continue

        by_px = volume_by_price(trades, old_ts, ts, tick=TICK)
        d = dom_read(old_lad, lad, by_px, window_s=window_s, samples=2)
        if d.measured:
            inf_c = d.bid.cancelled + d.ask.cancelled
            inf_a = d.bid.added + d.ask.added
            inf_t = d.bid.traded + d.ask.traded

            true_c = true_a = 0.0
            for w in ("bid", "ask"):
                lo_a, hi_a = _span(old_lad, w)
                lo_b, hi_b = _span(lad, w)
                lo, hi = max(lo_a, lo_b), min(hi_a, hi_b)
                if hi < lo:
                    continue
                for p, s in pending_cancel[w].items():
                    if lo <= p <= hi:
                        true_c += s
                for p, s in pending_add[w].items():
                    if lo <= p <= hi:
                        true_a += s

            out.windows += 1
            out.true_cancel += true_c
            out.true_add += true_a
            out.true_trade += sum(t.sz for t in trades)
            out.inferred_cancel += inf_c
            out.inferred_add += inf_a
            out.inferred_trade += inf_t
            out.pairs.append((true_c, inf_c))

        old_lad, old_ts = lad, ts
        pending_cancel = {"bid": defaultdict(float),
                          "ask": defaultdict(float)}
        pending_add = {"bid": defaultdict(float), "ask": defaultdict(float)}
        trades = []

    if progress is not None:
        progress.done()
    return out


def correlation(pairs) -> float | None:
    n = len(pairs)
    if n < 3:
        return None
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in pairs)
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


# ------------------------------------------------------------------ report

ACTION_NAMES = {"A": "add", "C": "cancel", "M": "modify", "T": "trade",
                "F": "fill", "R": "clear", "?": "(none)"}


def capped(records, limit: int):
    """The first `limit` records, or all of them when limit is 0.

    For finding out how long a full run will take before starting one.
    Whatever comes out of a capped run is part of a session, so the
    report says so rather than leaving it to be noticed.
    """
    if limit <= 0:
        return records
    return itertools.islice(records, limit)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path")
    ap.add_argument("--symbol", default="ES")
    ap.add_argument("--window", type=float, default=WINDOW_S)
    ap.add_argument("--gap", type=float, default=1.0,
                    help="seconds of silence worth reporting")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N records -- a trial run, not a result")
    args = ap.parse_args(argv)

    print(f"file   {args.path}")
    print("(two passes over the file; progress on stderr)")
    if args.limit:
        print(f"LIMIT  {args.limit:,} records -- a trial run. The figures "
              f"below cover part of the session and are not a result.")
    print()
    c = census(capped(dbn.open_file(args.path), args.limit), gap_s=args.gap,
               progress=Progress("census"))

    span = max(0.0, c.ts_last - c.ts_first)
    print("RECORDS")
    print(f"  total            {c.records:,}")
    print(f"  span             {span / 3600.0:.2f} h "
          f"({c.ts_first:.6f} -> {c.ts_last:.6f} epoch seconds)")
    if span > 0:
        print(f"  rate             {c.records / span:,.0f} records/s")
    print()

    print("ACTIONS  (this is the field `dbn.stream` discards)")
    for act, n in c.by_action.most_common():
        name = ACTION_NAMES.get(act, act)
        bid = c.by_action_side[(act, dbn.SIDE_BID)]
        ask = c.by_action_side[(act, dbn.SIDE_ASK)]
        none = c.by_action_side[(act, dbn.SIDE_NONE)]
        print(f"  {act} {name:<8} {n:>12,}  "
              f"size {c.size_by_action[act]:>14,.0f}  "
              f"bid {bid:>10,} ask {ask:>10,} none {none:>9,}")
    print()

    print("TIMESTAMPS")
    print(f"  out of order     {c.out_of_order:,}")
    print(f"  duplicate stamps {c.duplicate_ts:,}")
    print(f"  gaps >= {args.gap}s     {len(c.gaps)}"
          f"{' (largest shown)' if c.gaps else ''}")
    for d, a, b in c.gaps[:MAX_GAP_REPORT]:
        print(f"      {d:8.2f}s  at {a:.3f}")
    print()

    print("INSTRUMENTS")
    for iid, n in c.instruments.most_common(8):
        print(f"  id {iid:<12} {n:>12,} records")
    if len(c.instruments) > 8:
        print(f"  ... and {len(c.instruments) - 8} more")
    for sym, n in c.symbols.most_common(8):
        print(f"  symbol {sym:<10} {n:>12,}")
    print()

    print("BOOK INTEGRITY")
    print(f"  crossed books    {c.crossed:,}")
    print(f"  undefined touch  {c.undef_px:,}")
    print(f"  no level block   {c.empty_book:,}")
    for k, n in sorted(c.level_counts.items()):
        print(f"  {k:>2} levels        {n:>12,}")
    print()

    print(f"INFERRED vs ACTUAL  (window {args.window:.0f}s, "
          f"clipped to the overlapping price span)")
    cmp = compare(capped(dbn.open_file(args.path), args.limit),
                  window_s=args.window, symbol=args.symbol,
                  progress=Progress("compare"))
    if cmp.windows == 0:
        print("  no comparable windows")
        return 0
    print(f"  windows          {cmp.windows:,}")
    print(f"  cancelled  file {cmp.true_cancel:>14,.0f}   "
          f"inferred {cmp.inferred_cancel:>14,.0f}")
    print(f"  added      file {cmp.true_add:>14,.0f}   "
          f"inferred {cmp.inferred_add:>14,.0f}")
    print(f"  traded     tape {cmp.true_trade:>14,.0f}   "
          f"inferred {cmp.inferred_trade:>14,.0f}")
    r = cmp.cancel_ratio
    if r is not None:
        print(f"  the inference sees {r:.1%} of the cancelled size the "
              f"file reports")
    rho = correlation(cmp.pairs)
    if rho is not None:
        print(f"  per-window correlation {rho:+.3f}")
    print()
    print("  A ratio near 1.0 and a high correlation mean the inference is")
    print("  recovering what the file already knew. A small ratio means the")
    print("  engine has been reading the net of a far larger flow, and any")
    print("  measure built on `cancelled` is measuring the residue.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
