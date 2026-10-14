#!/usr/bin/env python3
"""Does the stack rule work on ES? The rule as traded, not a proxy for it.

    python scripts/stack_study.py data/ES_c_0_mbp-10_2026-10-08.dbn.zst
    python scripts/stack_study.py data/*.dbn.zst --trading 15m
    python scripts/stack_study.py data/*.dbn.zst --trading 5m --json out.json

THE RULE BEING TESTED

Trading the 15m, take a side only when the 15m, the 5m and the 1m all
qualify the same way -- green with a positive move, or red with a negative
one. A row whose colour fights its own number is thrown out. A pullback is
not a signal and neither is a stack that has not committed.

That is `cascade.read`, and this calls IT rather than reimplementing the
rule. A study that reimplements the thing it is testing is testing the
reimplementation.

THE SIGNAL IS READ INSIDE THE BAR, WHICH IS THE ONLY HONEST WAY

The read is intrabar: each row describes its own bar in progress. So the
decision moment is a point INSIDE the 15m bar -- four, eight and twelve
minutes in -- and at that moment each row sees only what had printed by
then. Everything scored happens strictly after.

A read taken at the bar's close would score beautifully and be
untradeable, which is why the checkpoints are fixed and early.

WHAT IS SCORED

From the signal's price, does the target or the stop come first. First
touch, walked forward one minute at a time. A MINUTE THAT TOUCHED BOTH IS
SCORED AS THE LOSS, because a one-minute bucket does not record the order
of ticks inside itself and guessing in the tester's favour is how a
backtest flatters itself.

Beside every hit rate sits the rate it would have to CLEAR to break even
at that target, and a shuffled control that keeps every signal and every
outcome but severs which went with which.

AND THE COUNT IS THE HEADLINE

This rule fires rarely by design. One session yields single-digit signals,
and single-digit anything is a story rather than a measurement. The fire
rate is printed first, because it is what tells you how many sessions you
would have to buy before any of the rest of this means something.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from liqmap import cascade as csc                      # noqa: E402
from liqmap import dbn                                 # noqa: E402
from liqmap import pressure as pr                      # noqa: E402
from liqmap.live import LiveCandle                     # noqa: E402
from liqmap.structure import Candle                    # noqa: E402

MINUTE = 60.0
SECONDS = {"1m": 60.0, "5m": 300.0, "15m": 900.0, "30m": 1800.0,
           "1h": 3600.0}

# Minutes into the trading bar at which the stack is read. Early on
# purpose: a read taken near the close knows almost everything and could
# never have been acted on.
#
# AND NOT AT A POINT WHERE THE ROWS COINCIDE. Four minutes into a 15m bar,
# the current 5m sub-bar started at the same instant the 15m one did, so
# both rows are built from exactly the same prints and agree by
# construction. A "confluence" rule tested there is testing nothing.
# Six, nine and twelve each sit inside a different 5m sub-bar.
#
# The rule: every confirming row's current bar must have STARTED after
# the trading bar opened. On a 30m that means past the fifteen-minute
# mark, because before it the 15m row is the same window.
CHECKPOINTS = {"15m": (6, 9, 12), "5m": (2, 3, 4), "30m": (18, 22, 26)}

# Targets in ticks, with the stop at the same distance. Symmetric, so the
# break-even rate is simply above half once costs are in.
#
# Five through twenty-plus, because that is the range being asked about.
# The small ones are where the round turn hurts most and the large ones
# are where a bar runs out of time to reach either level -- both ends are
# worth seeing rather than assuming.
TARGETS = (5, 8, 10, 12, 15, 20, 25)

# ES: one tick is 0.25 index points and $12.50 on the full contract,
# $1.25 on the micro. Round turn commission is the number that decides
# whether a two-tick scalp is a business or a hobby, and it is a property
# of the broker rather than of the data, so it is an argument.
TICK = 0.25
DEFAULT_RT_TICKS = 0.4          # ~$5 round turn on ES, in ticks

# Bars of history each row learns "normal" from. The same figure the live
# ladder uses, so the thresholds are the ones that would have been applied.
HISTORY = 20

# A difference in pull share below this is two sides behaving the same
# way. The same figure `trace` uses, deliberately: two modules disagreeing
# about what a thinning book looks like would put a study on screen
# contradicting the panel.
PULL_EDGE = 0.15

# How long a window the DOM read covers, and how much tape it needs beside
# it to tell a fill from a cancel.
DOM_WINDOW_S = 20.0


# ---------------------------------------------------------------------------
# one pass over the file: minute bars with a real buy/sell split
# ---------------------------------------------------------------------------


@dataclass
class Minute:
    """One minute of tape, with who crossed the spread counted."""

    ts: float
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    trades: int = 0
    buy_notional: float = 0.0
    sell_notional: float = 0.0

    def add(self, px: float, sz: float, buy: bool) -> None:
        if self.open <= 0:
            self.open = self.high = self.low = px
        self.high = max(self.high, px)
        self.low = min(self.low, px)
        self.close = px
        self.volume += sz
        self.trades += 1
        if buy:
            self.buy_notional += px * sz
        else:
            self.sell_notional += px * sz

    @property
    def filled(self) -> bool:
        return self.trades > 0


@dataclass
class Snap:
    """What the resting book was doing at one checkpoint."""

    ts: float
    read: Any = None            # dom.DomRead
    mid: float = 0.0

    @property
    def measured(self) -> bool:
        return bool(self.read is not None and self.read.measured)

    @property
    def side(self) -> str:
        """Which way the book leans, from what is being WITHDRAWN.

        Offers vanishing ahead of a rise is the book getting out of the
        way; bids vanishing is the floor coming out. Only the DIFFERENCE
        between the two carries anything -- both sides churn hard in a
        fast market, and one side's pull share on its own mostly measures
        how busy that side is.
        """
        if not self.measured:
            return ""
        edge = self.read.ask.pull_share - self.read.bid.pull_share
        if abs(edge) < PULL_EDGE:
            return ""
        return "buy" if edge > 0 else "sell"


def walk_file(records: Iterable[Any], trading: str, marks: Sequence[int],
              flip: bool = False) -> tuple[list[Minute], dict[float, Snap]]:
    """One pass: minute bars, and the book's state at every checkpoint.

    BOTH IN ONE WALK, because the file is large and the DOM read needs
    something the minute bars throw away -- the ladder as it stood in the
    twenty seconds before the decision, and the tape beside it to tell a
    fill from a cancel.

    The checkpoints are not known in advance from the minutes, so they are
    generated from the clock as it advances: each record says which bar it
    is in, and the moment the stream crosses a checkpoint the book is read
    THERE, using only what had already arrived.
    """
    from collections import deque

    from liqmap.dom import DomWatch

    step = SECONDS[trading]
    mins: dict[float, Minute] = {}
    snaps: dict[float, Snap] = {}
    dom = DomWatch()
    recent: deque = deque()
    book = None
    due: list[float] = []
    seen_bar = None

    def checkpoints_for(bar: float) -> list[float]:
        return [bar + mk * MINUTE for mk in marks]

    for kind, item in dbn.stream(records, flip=flip):
        if kind == "clear":
            continue
        if kind == "book":
            book = item
            dom.add(book)
            ts = book.ts
        else:
            ts = item.ts
            slot = math.floor(ts / MINUTE) * MINUTE
            m = mins.get(slot)
            if m is None:
                m = mins[slot] = Minute(ts=slot)
            m.add(item.px, item.sz, item.aggressor == "buy")
            recent.append(item)
            while recent and recent[0].ts < ts - DOM_WINDOW_S:
                recent.popleft()

        bar = math.floor(ts / step) * step
        if bar != seen_bar:
            seen_bar = bar
            due = [c for c in checkpoints_for(bar) if c not in snaps]

        # FIRE ON THE WAY PAST, NOT AFTER. Everything in `dom` and
        # `recent` at this instant arrived at or before `ts`, so a read
        # taken here cannot contain a print from the future.
        while due and ts >= due[0]:
            at = due.pop(0)
            snaps[at] = Snap(
                ts=at,
                read=dom.read(trades=list(recent), window_s=DOM_WINDOW_S,
                              tick=TICK, now=at),
                mid=(book.mid if book is not None and not book.empty else 0.0))

    return [mins[k] for k in sorted(mins)], snaps


def roll_up(mins: Sequence[Minute], start_ts: float, end_ts: float
            ) -> LiveCandle | None:
    """The minutes in [start_ts, end_ts) as one partial bar.

    END IS EXCLUSIVE AND THAT IS THE WHOLE POINT. The minute the decision
    is taken in has not finished, so including it would hand the reader
    prints that had not happened yet.
    """
    got = [m for m in mins if start_ts <= m.ts < end_ts and m.filled]
    if not got:
        return None
    c = LiveCandle(interval_s=end_ts - start_ts, start_ts=start_ts,
                   open=got[0].open, high=max(m.high for m in got),
                   low=min(m.low for m in got), close=got[-1].close,
                   volume=sum(m.volume for m in got),
                   trades=sum(m.trades for m in got),
                   buy_notional=sum(m.buy_notional for m in got),
                   sell_notional=sum(m.sell_notional for m in got))
    return c


def completed(mins: Sequence[Minute], interval_s: float, before_ts: float,
              want: int = HISTORY) -> list[Candle]:
    """The last `want` COMPLETED bars of this timeframe before a moment.

    What each row learns "normal" from. Strictly before, so the bar being
    judged never helps set the threshold it is judged against.
    """
    out: list[Candle] = []
    end = math.floor(before_ts / interval_s) * interval_s
    for i in range(want, 0, -1):
        a = end - i * interval_s
        c = roll_up(mins, a, a + interval_s)
        if c is not None:
            out.append(Candle(ts=a, open=c.open, high=c.high, low=c.low,
                              close=c.close, volume=c.volume))
    return out


# ---------------------------------------------------------------------------
# the read, through the engine's own code
# ---------------------------------------------------------------------------


def row_at(mins: Sequence[Minute], timeframe: str, at_ts: float,
           book: Any = None) -> tuple[str, float] | None:
    """One ladder row as it stood at `at_ts`: (colour, move_bps).

    Built with `pressure.from_live` and coloured with `cascade.colour_of`
    -- the same two calls the live route makes -- so this cannot drift
    away from what the screen shows.
    """
    step = SECONDS[timeframe]
    start = math.floor(at_ts / step) * step
    bar = roll_up(mins, start, at_ts)
    if bar is None:
        # NOTHING HAS PRINTED IN THIS BAR YET. On the one-minute row at a
        # minute boundary that is always true -- the bar in progress is
        # zero seconds old -- and returning None there left the 1m row
        # permanently unread, which silenced the rule completely.
        #
        # What a trader sees at that instant is the minute that just
        # closed, so that is what this reads.
        start -= step
        bar = roll_up(mins, start, start + step)
        if bar is None:
            return None
    p = pr.from_live(timeframe, step, bar, book,
                     history=completed(mins, step, start), now=at_ts)
    return csc.colour_of(p.winner, forming=p.forming), p.move_bps


@dataclass
class Signal:
    """A committed stack read, and what price did after it."""

    ts: float
    side: str
    px: float
    mark: int                   # minutes into the trading bar
    state: str = ""
    dom: str = ""               # which way the resting book leaned, if at all
    reached: dict = field(default_factory=dict)   # ticks -> True/False/None

    @property
    def dom_agrees(self) -> bool:
        return bool(self.dom) and self.dom == self.side

    @property
    def dom_against(self) -> bool:
        return bool(self.dom) and self.dom != self.side


def walk(mins: Sequence[Minute], snaps: dict, trading: str,
         marks: Sequence[int]) -> tuple[list[Signal], int]:
    """Every checkpoint in the file, and the signals among them.

    Returns (signals, checkpoints_examined) -- the second is what makes
    the fire rate reportable, and a fire rate is the only thing a single
    session can honestly produce.
    """
    step = SECONDS[trading]
    want = csc.STACKS[trading]
    if not mins:
        return [], 0

    first = math.floor(mins[0].ts / step) * step + step    # skip a partial
    last = math.floor(mins[-1].ts / step) * step
    sigs: list[Signal] = []
    looked = 0

    bar = first
    while bar <= last:
        for mk in marks:
            at = bar + mk * MINUTE
            if at >= mins[-1].ts:
                continue
            looked += 1
            rows: dict[str, tuple[str, float]] = {}
            for tf in want:
                got = row_at(mins, tf, at)
                if got is not None:
                    rows[tf] = got
            c = csc.read(trading, rows)
            # ONLY A COMMITTED TREND. A pullback is explicitly not a
            # signal in the rule as stated, and a reversal needs a memory
            # of the previous alignment that a fresh walk does not have.
            if c.state is csc.State.TREND and c.side:
                # The last print STRICTLY BEFORE the checkpoint. The
                # minute the decision falls in has not finished, so its
                # close has not happened yet.
                prior = [m for m in mins if m.ts < at and m.filled]
                px = prior[-1].close if prior else 0.0
                if px > 0:
                    sn = snaps.get(at)
                    sigs.append(Signal(ts=at, side=c.side, px=px, mark=mk,
                                       state=c.state.value,
                                       dom=sn.side if sn else ""))
        bar += step
    return sigs, looked


# ---------------------------------------------------------------------------
# what happened next
# ---------------------------------------------------------------------------


def score(sig: Signal, mins: Sequence[Minute], ticks: int,
          tick: float = TICK) -> bool | None:
    """Did the target come before the stop? None if neither did.

    First touch, minute by minute, STRICTLY AFTER the signal. A minute
    that touched both levels is a loss: the bucket does not record the
    order of ticks inside it, and resolving that ambiguity in the
    tester's favour is how a backtest flatters itself.
    """
    d = ticks * tick
    buy = sig.side == "buy"
    # Named rather than juggled. An earlier version computed these as a
    # pair of conditional expressions and had the short's target and stop
    # the wrong way round -- which does not crash, and turns every losing
    # short into a winner.
    target = sig.px + d if buy else sig.px - d
    stop = sig.px - d if buy else sig.px + d
    for m in mins:
        if m.ts < sig.ts or not m.filled:
            continue
        hit_t = (m.high >= target) if buy else (m.low <= target)
        hit_s = (m.low <= stop) if buy else (m.high >= stop)
        if hit_t and hit_s:
            return False
        if hit_t:
            return True
        if hit_s:
            return False
    return None


def wilson(hits: int, n: int) -> tuple[float, float]:
    """A 95% interval that behaves at small n, where this study lives."""
    if n <= 0:
        return 0.0, 1.0
    z = 1.959963985
    p = hits / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - r) / d), min(1.0, (c + r) / d)


def breakeven(ticks: int, rt_ticks: float) -> float:
    """The hit rate a symmetric target/stop must clear to make money.

    Half, plus what the round turn costs expressed as a share of the
    distance being played for. A two-tick scalp paying nearly half a tick
    each way is a far harder business than the symmetry suggests.
    """
    return 0.5 + (rt_ticks / (4.0 * ticks))


@dataclass
class Table:
    n: int = 0
    hits: int = 0
    unresolved: int = 0


def tally(sigs: Sequence[Signal], ticks: int) -> Table:
    t = Table()
    for s in sigs:
        got = s.reached.get(ticks)
        if got is None:
            t.unresolved += 1
            continue
        t.n += 1
        t.hits += int(got)
    return t


def shuffled(sigs: Sequence[Signal], mins: Sequence[Minute], ticks: int,
             runs: int = 200, seed: int = 11) -> tuple[float, float]:
    """The control: same bars, same mix of sides, RE-SCORED after shuffling
    which bar got which side.

    THE OUTCOMES CANNOT BE SHUFFLED. An earlier version permuted the list
    of results and averaged it, which returns the same number every time
    -- a mean does not care about order. It printed a control that agreed
    with the real rate to the decimal and looked like confirmation.

    So the sides are permuted across the signals and every one is scored
    again from scratch. That keeps the moments the rule chose and the
    balance of longs to shorts, and severs only the thing being tested:
    whether it put the right side on the right bar.
    """
    rnd = random.Random(seed)
    pool = [s for s in sigs if s.reached.get(ticks) is not None]
    if len(pool) < 2:
        return 0.0, 0.0
    sides = [s.side for s in pool]
    rates = []
    for _ in range(runs):
        rnd.shuffle(sides)
        hits = n = 0
        for s, side in zip(pool, sides):
            got = score(Signal(ts=s.ts, side=side, px=s.px, mark=s.mark),
                        mins, ticks)
            if got is None:
                continue
            n += 1
            hits += int(got)
        if n:
            rates.append(hits / n)
    if not rates:
        return 0.0, 0.0
    mean = sum(rates) / len(rates)
    var = sum((r - mean) ** 2 for r in rates) / len(rates)
    return mean, math.sqrt(var)


# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", help="MBP-10 .dbn/.dbn.zst files")
    ap.add_argument("--trading", default="15m", choices=sorted(csc.STACKS),
                   help="the timeframe being traded (default 15m)")
    ap.add_argument("--rt-ticks", type=float, default=DEFAULT_RT_TICKS,
                   help="round-turn cost in ticks (default %(default)s)")
    ap.add_argument("--flip", action="store_true",
                   help="invert the aggressor convention -- only if the "
                        "audit said to")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    marks = CHECKPOINTS.get(a.trading)
    if marks is None:
        print(f"no checkpoints declared for {a.trading}", file=sys.stderr)
        return 2

    mins: list[Minute] = []
    snaps: dict = {}
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
        got, sn = walk_file(store, a.trading, marks, flip=a.flip)
        mins.extend(got)
        snaps.update(sn)
    mins.sort(key=lambda m: m.ts)

    if not mins:
        print("no trades parsed", file=sys.stderr)
        return 1

    hours = (mins[-1].ts - mins[0].ts) / 3600.0
    print(f"  {len(mins):,} minutes of tape, {hours:.1f} hours")

    sigs, looked = walk(mins, snaps, a.trading, marks)
    for s in sigs:
        for t in TARGETS:
            s.reached[t] = score(s, mins, t)

    print(f"\n=== the rule ===")
    print(f"  trading            {a.trading}, confirmed by "
          f"{', '.join(csc.STACKS[a.trading][1:]) or 'nothing'}")
    print(f"  read at            {', '.join(str(m) + 'm' for m in marks)} "
          f"into each bar")
    print(f"  checkpoints seen   {looked:,}")
    rate = (len(sigs) / looked) if looked else 0.0
    print(f"  signals            {len(sigs):,}  ({rate:.0%} of checkpoints)")
    longs = sum(1 for s in sigs if s.side == "buy")
    print(f"  of those           {longs} buy, {len(sigs) - longs} sell")

    if len(sigs) < 30:
        print(f"\n  *** {len(sigs)} signals is a story, not a measurement. ***")
        print(f"  At this fire rate a session yields about "
              f"{len(sigs) / max(hours / 6.5, 0.01):.0f} of them.")
        need = 800
        print(f"  Detecting a small edge wants ~{need}, so roughly "
              f"{need / max(len(sigs) / max(hours / 6.5, 0.01), 0.01):.0f} "
              f"sessions.")
        print(f"  A large edge (clearly above break-even) wants ~200, so "
              f"roughly "
              f"{200 / max(len(sigs) / max(hours / 6.5, 0.01), 0.01):.0f}.")

    # THREE POPULATIONS, SCORED SIDE BY SIDE. The book is only worth
    # consulting if signals it backs do better than signals it does not,
    # so the comparison is built in rather than assumed. If the three
    # columns land on top of each other, the DOM read is decoration.
    groups = (("stack only", list(sigs)),
              ("+ book agrees", [x for x in sigs if x.dom_agrees]),
              ("+ book against", [x for x in sigs if x.dom_against]))

    told = sum(1 for x in sigs if x.dom)
    print(f"\n=== the book ===")
    print(f"  signals where the DOM said anything   {told} of {len(sigs)}")
    print(f"    backing the stack                   "
          f"{sum(1 for x in sigs if x.dom_agrees)}")
    print(f"    against it                          "
          f"{sum(1 for x in sigs if x.dom_against)}")

    out_rows: dict = {}
    for name, pop in groups:
        print(f"\n=== outcomes: {name} ===  n={len(pop)}")
        if not pop:
            print("  nothing in this group")
            continue
        print(f"  {'target':<20} {'reached first':<24} {'needs':<8} "
              f"{'shuffled':<11} verdict")
        rows = {}
        for t in TARGETS:
            tb = tally(pop, t)
            be = breakeven(t, a.rt_ticks)
            if tb.n == 0:
                print(f"  {t} ticks{'':<13} nothing resolved "
                      f"({tb.unresolved} ran out of file)")
                continue
            lo, hi = wilson(tb.hits, tb.n)
            q = tb.hits / tb.n
            ctl_m, _ = shuffled(pop, mins, t)
            verdict = ("clears" if lo > be else
                       "loses" if hi < be else "unclear")
            print(f"  {t} ticks (${t * 12.5:>7.2f})  "
                  f"{q * 100:>5.1f}% ({lo * 100:>4.1f}-{hi * 100:>4.1f}%) "
                  f"n={tb.n:<4} {be * 100:>5.1f}%   {ctl_m * 100:>5.1f}%     "
                  f"{verdict}")
            rows[str(t)] = {"n": tb.n, "hits": tb.hits, "rate": q,
                            "lo": lo, "hi": hi, "breakeven": be,
                            "control": ctl_m, "unresolved": tb.unresolved,
                            "verdict": verdict}
        out_rows[name] = rows

    print("\n  'needs' is half plus the round turn as a share of the "
          "distance played for.")
    print("  'shuffled' re-scores every signal after permuting which bar "
          "got which side.")
    print("  A rate matching it is a rule choosing no better than a hat.")
    print("  A minute touching both levels is scored as the loss, and a "
          "signal whose")
    print("  file ends before either level is reached is counted apart, "
          "never as a loss.")
    print("  Signals inside one bar are not independent, so the true "
          "intervals are wider.")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"trading": a.trading, "marks": list(marks),
             "checkpoints": looked, "signals": len(sigs),
             "fire_rate": rate, "hours": hours,
             "rt_ticks": a.rt_ticks, "targets": out_rows,
             "dom": {"told": told, "agrees":
                     sum(1 for x in sigs if x.dom_agrees),
                     "against": sum(1 for x in sigs if x.dom_against)}},
            indent=2))
        print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
