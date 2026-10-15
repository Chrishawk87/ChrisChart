#!/usr/bin/env python3
"""Causal tick swings on ES, and whether anything predicts the next 5-10 ticks.

    python scripts/swing_study.py data/ES_c_0_mbp-10_2026-10-08.dbn.zst

WHAT A SWING IS HERE, AND WHEN IT IS KNOWN

A leg runs until price gives back `reversal` ticks from its extreme. At that
moment -- and not before -- the extreme becomes a confirmed pivot.

    pivot_ts    when the extreme printed
    ts          when the retracement completed and it became knowable

Those are different timestamps and the gap between them is the whole study.
Every measurement below is taken from prints at or before `ts`. Every outcome
is resolved from prints strictly after `ts`. Dating a decision at `pivot_ts`
would be reading a turning point at the turn, which is the family of bug that
scored 98.4% on a pure random walk earlier in this project.

THE RETRACEMENT IS A GRID, NOT A SETTING

Two ticks of give-back finds many small legs, four finds fewer larger ones,
and the only way to learn which the market has is to print all of them as
rows. Choosing the one with the prettiest column afterwards is fitting.

EVERY ARM IS A SUBSET OF THE SAME BASE CALL

`continuation` (trade the new leg) and `reversal` (trade the completed leg)
are exact mirrors and neither consults anything. Every other arm is one of
those two filtered down -- by structure, by flow, by the book. So "does the
book add value" is answerable the only way it can be: the same call, on the
bars where the book agreed, against the bars where it did not, scored by one
scorer. A filter that improves nothing will land on its parent's number.

MFE AND MAE ARE THE RESULT THAT OUTLIVES THE GEOMETRY

A target-and-stop pair is a structure imposed on the data. The excursion
distributions are not: they say how far price went each way after the call,
and any geometry can be read off them later. When the tables disagree with
the excursions, believe the excursions.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER. It counts outcomes.
"""

from __future__ import annotations

import argparse
import bisect
import math
import random
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap import dbn                                      # noqa: E402
from liqmap.bookread import BookReader                      # noqa: E402
from liqmap.dom import DomWatch                             # noqa: E402

TICK = 0.25

# Ticks of give-back that confirm a pivot. Reported as rows; never selected.
REVERSALS = (2, 3, 4)

# THE TRADE BEING TESTED, IN ITS OWN UNITS.
#
# Four to five POINTS on ES is sixteen to twenty TICKS. Every earlier study
# in this project tested 5, 8 and 10 ticks -- a quarter to a half of that --
# and capped the hold at 120 to 300 seconds against a trade held toward a
# 15-minute close. Those nulls are uninformative about this trade rather
# than evidence against it, and the two are not the same thing.
TARGETS = (8, 12, 16, 20, 24)          # 2 to 6 points; 16 and 20 are the ask

# Absolute, not a multiple of the target. A stop tighter than the target is
# the whole shape of this trade and a symmetric grid cannot express it.
STOP_TICKS = (4, 8, 12)                # 1, 2 and 3 points

# Nothing resolves after the time stop. Reported as a grid because the
# number was not supplied, and picking the one that reads best afterwards
# is fitting.
TIME_STOPS = (180.0, 300.0, 900.0)     # 3, 5 and 15 minutes

HORIZON_S = max(TIME_STOPS)            # the excursion window

# The geometry the tables lead with, in ticks.
HEADLINE = ((16, 4), (20, 4))
DOM_WINDOW_S = 20.0
TREND_LOOKBACK_S = 60.0

# THE LIVE-TAPE CONFIRMATION, as its own window.
#
# "Is the move strong" is a question about the seconds around the entry, not
# about the whole leg. A leg can be bought the whole way up and be dead by
# the time its pivot confirms, and the leg-wide delta cannot tell those
# apart. So this is measured separately and the two are kept apart.
CONFIRM_WINDOW_S = 10.0
MIN_ARM = 30               # below this an arm is a count, not a rate
TRAIN_SHARE = 0.6          # chronological, and nothing is fitted outside it
DEFAULT_RT_TICKS = 1.0     # commission plus the stop exit; entry already
                           # pays the spread in the entry price itself
BAR_S = 60.0


# --------------------------------------------------------------- the tape

class Tape:
    """Every print, in order, with forward-only questions."""

    def __init__(self) -> None:
        self.ts: list[float] = []
        self.px: list[float] = []

    def add(self, ts: float, px: float) -> None:
        self.ts.append(ts)
        self.px.append(px)

    @property
    def n(self) -> int:
        return len(self.ts)

    def _after(self, ts: float) -> int:
        return bisect.bisect_right(self.ts, ts)

    def crossings(self, ts: float, entry: float | None,
                  targets: list[float], stops: list[float],
                  long: bool, cap_s: float):
        """ONE forward walk. When each level was first crossed, plus the
        excursion over the same window.

        Every cell in the study comes out of this. Forty-five separate
        walks per signal gave the same answers and would not finish on a
        real session, but the reason this is one function rather than an
        optimisation bolted alongside the old one is that two walkers
        drift: the fast one grows a subtle difference and the tests still
        pass because they only ever exercise the slow one.

        Prints are taken STRICTLY after `ts`. A print carries one price, so
        it cannot cross a target and a stop at once and there is no tie to
        resolve in anybody's favour.
        """
        i = self._after(ts)
        end = ts + cap_s
        t_at: list[float | None] = [None] * len(targets)
        s_at: list[float | None] = [None] * len(stops)
        left = len(targets) + len(stops)
        fav = adv = None
        while i < self.n and self.ts[i] <= end:
            p, now = self.px[i], self.ts[i]
            if entry is not None:
                d = (p - entry) if long else (entry - p)
                fav = d if fav is None else max(fav, d)
                adv = d if adv is None else min(adv, d)
            if left:
                for k, lv in enumerate(targets):
                    if t_at[k] is None and ((p >= lv) if long else (p <= lv)):
                        t_at[k] = now
                        left -= 1
                for k, lv in enumerate(stops):
                    if s_at[k] is None and ((p <= lv) if long else (p >= lv)):
                        s_at[k] = now
                        left -= 1
            i += 1
        ex = None if fav is None else (fav / TICK, -adv / TICK)
        return t_at, s_at, ex

    @staticmethod
    def decide(t_at: float | None, s_at: float | None,
               deadline: float) -> bool | None:
        """The ONE rule that turns two crossing times into an outcome.

        Unresolved is None, never a loss: a call whose levels were not
        reached says nothing, and counting it against the call would
        punish exactly the conditions it was silent about.
        """
        hit = t_at is not None and t_at <= deadline
        out = s_at is not None and s_at <= deadline
        if hit and (not out or t_at < s_at):
            return True
        if out:
            return False
        return None

    def first_touch(self, ts: float, target: float, stop: float,
                    long: bool, cap_s: float = HORIZON_S) -> bool | None:
        """One pair, for callers and tests that want a single answer.

        Delegates, so there is no second walker and no second rule.
        """
        t_at, s_at, _ = self.crossings(ts, None, [target], [stop], long,
                                       cap_s)
        return self.decide(t_at[0], s_at[0], ts + cap_s)

    def excursion(self, ts: float, entry: float, long: bool,
                  until: float) -> tuple[float, float] | None:
        """(favourable, adverse) in ticks, both non-negative.

        Geometry-free: no target, no stop, just how far it went each way.
        Delegates to the one walker.
        """
        _, _, ex = self.crossings(ts, entry, [], [], long, until - ts)
        return ex


# ------------------------------------------------------------ the detector

@dataclass
class Leg:
    """One completed run, from the previous pivot to this one."""

    kind: str                  # the pivot that ENDED it: "high" or "low"
    px: float
    ts: float                  # when the extreme printed
    start_px: float
    start_ts: float

    @property
    def ticks(self) -> float:
        return abs(self.px - self.start_px) / TICK

    @property
    def seconds(self) -> float:
        return max(0.0, self.ts - self.start_ts)


class Swinger:
    """Reversal-threshold pivots, emitted only when they become knowable.

    The running extreme is not a pivot. It becomes one when price has given
    back `reversal` ticks from it, and the caller hears about it then.
    """

    def __init__(self, reversal_ticks: int) -> None:
        self.give = reversal_ticks * TICK
        self.way = ""                     # "up" | "down" | "" until seeded
        self.ext_px = 0.0
        self.ext_ts = 0.0
        self.start_px = 0.0
        self.start_ts = 0.0
        self.seeded = False

    def feed(self, ts: float, px: float) -> Leg | None:
        if not self.seeded:
            self.ext_px = self.start_px = px
            self.ext_ts = self.start_ts = ts
            self.seeded = True
            return None

        if self.way in ("", "up"):
            if px > self.ext_px:
                self.ext_px, self.ext_ts = px, ts
                self.way = "up"
                return None
            if self.way == "up" and (self.ext_px - px) >= self.give:
                leg = Leg(kind="high", px=self.ext_px, ts=self.ext_ts,
                          start_px=self.start_px, start_ts=self.start_ts)
                self.start_px, self.start_ts = self.ext_px, self.ext_ts
                self.ext_px, self.ext_ts = px, ts
                self.way = "down"
                return leg

        if self.way in ("", "down"):
            if px < self.ext_px:
                self.ext_px, self.ext_ts = px, ts
                self.way = "down"
                return None
            if self.way == "down" and (px - self.ext_px) >= self.give:
                leg = Leg(kind="low", px=self.ext_px, ts=self.ext_ts,
                          start_px=self.start_px, start_ts=self.start_ts)
                self.start_px, self.start_ts = self.ext_px, self.ext_ts
                self.ext_px, self.ext_ts = px, ts
                self.way = "up"
                return leg

        return None


# -------------------------------------------------------------- the look

@dataclass
class Look:
    """Everything knowable at one confirmation, and nothing else."""

    ts: float                        # CONFIRMATION. The only legal decision.
    pivot_kind: str                  # "high" | "low"
    pivot_px: float
    pivot_ts: float

    leg_ticks: float
    leg_s: float
    velocity: float                  # ticks per second over the leg
    retrace_pct: float               # the confirming give-back / leg
    prev_leg_ticks: float
    structure: str                   # HH | LH | HL | LL | ""
    failed_break: bool
    changes_1m: int

    delta: float                     # signed contracts over the leg
    efficiency: float | None         # leg ticks per 100 of |delta|

    imbalance: float | None          # ten-level notional edge
    cancel_edge: float | None
    depth_edge: float | None

    bid: float
    ask: float
    mid: float
    bar_ts: float | None = None      # 1-minute bar containing the decision

    # THE LIVE TAPE AT THE ENTRY, over CONFIRM_WINDOW_S and kept apart from
    # the leg-wide figures above. A leg bought the whole way up can be dead
    # by the time its pivot confirms; `delta` cannot tell those apart and
    # `recent_delta` can.
    recent_delta: float = 0.0
    recent_prints: int = 0
    print_rate: float | None = None  # prints/s now against the leg's own

    # filled by score(): side -> (target crossings, stop crossings, excursion)
    cross: dict = field(default_factory=dict)

    @property
    def expansion(self) -> float | None:
        if self.prev_leg_ticks <= 0:
            return None
        return self.leg_ticks / self.prev_leg_ticks

    # -- the two blind arms, exact mirrors, consulting nothing ------------

    @property
    def continuation(self) -> str:
        """Trade the leg that is starting."""
        return "sell" if self.pivot_kind == "high" else "buy"

    @property
    def reversal(self) -> str:
        """Trade the leg that just finished -- the pivot holds."""
        return "buy" if self.pivot_kind == "high" else "sell"

    # -- the filters. Each is a SUBSET of a base arm, never a new call ----

    def structure_agrees(self) -> bool:
        """Lower high before a down leg; higher low before an up leg."""
        if self.pivot_kind == "high":
            return self.structure == "LH"
        return self.structure == "HL"

    def flow_agrees(self, side: str) -> bool:
        """Aggression over the whole leg, signed the same way as the call."""
        if side == "buy":
            return self.delta > 0
        if side == "sell":
            return self.delta < 0
        return False

    def tape_agrees(self, side: str) -> bool | None:
        """Is the move strong RIGHT NOW -- the live-tape confirmation.

        Two conditions, both at the entry and neither over the leg: the
        recent prints are signed with the call, and they are arriving at
        least as fast as they did over the leg. A quiet tape agreeing is
        not a confirmation, which is why the rate is in here.
        """
        if self.print_rate is None or self.recent_prints <= 0:
            return None
        if self.print_rate < 1.0:
            return False
        if side == "buy":
            return self.recent_delta > 0
        if side == "sell":
            return self.recent_delta < 0
        return False

    def book_agrees(self, side: str, cut: float) -> bool | None:
        if self.depth_edge is None:
            return None
        if abs(self.depth_edge) < cut:
            return None
        way = "buy" if self.depth_edge > 0 else "sell"
        return way == side


BOOK_CUTS = (0.10, 0.25, 0.40)


# -------------------------------------------------------------- the walk

def walk(records, reversal_ticks: int, symbol: str = "ES",
         flip: bool = False):
    """One pass. Returns (looks, tape, prints)."""
    tape = Tape()
    reader = BookReader()
    watch = DomWatch(window_s=DOM_WINDOW_S)
    swinger = Swinger(reversal_ticks)

    looks: list[Look] = []
    last_book = None
    recent_trades: deque = deque(maxlen=20000)
    confirms: deque[float] = deque()
    prev_leg = 0.0
    last_high: float | None = None
    last_low: float | None = None

    for kind, item in dbn.stream(records, symbol=symbol, flip=flip):
        if kind == "clear":
            continue

        if kind == "book":
            reader.add(item, now=item.ts)
            watch.add(item, now=item.ts)
            last_book = item
            continue

        # a trade
        tape.add(item.ts, item.px)
        recent_trades.append(item)

        leg = swinger.feed(item.ts, item.px)
        if leg is None:
            continue

        now = item.ts                      # the confirming print's timestamp
        if last_book is None or last_book.best_bid <= 0 \
                or last_book.best_ask <= 0:
            # Not readable, so not a decision point -- but the leg still
            # happened, and dropping it from the history would make the
            # NEXT pivot compare itself to the wrong predecessor.
            prev_leg = leg.ticks
            if leg.kind == "high":
                last_high = leg.px
            else:
                last_low = leg.px
            continue

        while confirms and confirms[0] < now - TREND_LOOKBACK_S:
            confirms.popleft()
        confirms.append(now)

        # structure, from the previous confirmed pivot of the SAME kind
        structure = ""
        failed = False
        if leg.kind == "high" and last_high is not None:
            structure = "HH" if leg.px > last_high else "LH"
            failed = leg.px <= last_high
        elif leg.kind == "low" and last_low is not None:
            structure = "HL" if leg.px > last_low else "LL"
            failed = leg.px >= last_low

        # aggression over the leg, from prints inside it
        delta = 0.0
        for t in recent_trades:
            if t.ts < leg.start_ts:
                continue
            if t.ts > leg.ts:
                break
            delta += t.sz if t.aggressor == "buy" else -t.sz
        eff = (leg.ticks / abs(delta) * 100.0) if abs(delta) > 0 else None

        imb = _imbalance(last_book)
        c_edge, d_edge = _dom_edges(watch, list(recent_trades), now)

        # THE LIVE TAPE AT THE ENTRY. Counted over the seconds before the
        # confirmation, against the same quantity over the leg, so a quiet
        # tape that happens to agree is not mistaken for a strong one.
        # THE BASELINE EXCLUDES THE WINDOW IT IS JUDGING.
        #
        # Comparing the last ten seconds against the leg does not work: a
        # leg can be a fifth of a second long, and then its own print rate
        # is enormous and nothing is ever "fast". So the comparison is
        # against a trailing stretch that STOPS where the confirmation
        # window starts -- the same shape `orderflow.velocity` uses, and
        # for the same reason: a window that contains the spike cannot
        # measure it.
        r_delta = 0.0
        r_prints = 0
        b_prints = 0
        cut_recent = now - CONFIRM_WINDOW_S
        cut_base = now - TREND_LOOKBACK_S
        for t in reversed(recent_trades):
            if t.ts < cut_base:
                break
            if t.ts >= cut_recent:
                r_delta += t.sz if t.aggressor == "buy" else -t.sz
                r_prints += 1
            else:
                b_prints += 1
        base_s = TREND_LOOKBACK_S - CONFIRM_WINDOW_S
        base = b_prints / base_s
        rate_now = r_prints / CONFIRM_WINDOW_S
        p_rate = (rate_now / base) if base > 0 else None

        looks.append(Look(
            ts=now,
            pivot_kind=leg.kind, pivot_px=leg.px, pivot_ts=leg.ts,
            leg_ticks=leg.ticks, leg_s=leg.seconds,
            velocity=(leg.ticks / leg.seconds) if leg.seconds > 0 else 0.0,
            retrace_pct=((reversal_ticks / leg.ticks)
                         if leg.ticks > 0 else 0.0),
            prev_leg_ticks=prev_leg,
            structure=structure, failed_break=failed,
            changes_1m=len(confirms),
            delta=delta, efficiency=eff,
            imbalance=imb, cancel_edge=c_edge, depth_edge=d_edge,
            bid=last_book.best_bid, ask=last_book.best_ask,
            mid=last_book.mid,
            bar_ts=math.floor(now / BAR_S) * BAR_S,
            recent_delta=r_delta, recent_prints=r_prints, print_rate=p_rate,
        ))

        prev_leg = leg.ticks
        if leg.kind == "high":
            last_high = leg.px
        else:
            last_low = leg.px

    return looks, tape, tape.n


def _imbalance(book) -> float | None:
    nb = sum(l.notional for l in book.bids[:10])
    na = sum(l.notional for l in book.asks[:10])
    tot = nb + na
    return ((nb - na) / tot) if tot > 0 else None


def _dom_edges(watch: DomWatch, trades, now: float):
    """(cancel_edge, depth_edge) from the DOM read, or (None, None).

    Both are the two measures `dom_probe` found had usable shape on ES.
    `pull_share` is not among them: it saturates at 1.0 on both sides.
    """
    d = watch.read(trades=trades, window_s=DOM_WINDOW_S, tick=TICK, now=now)
    if not d.measured:
        return None, None
    churn = d.bid.churn + d.ask.churn
    if churn <= 0:
        return None, None
    c = (d.ask.cancelled - d.bid.cancelled) / churn
    nb, na = d.bid.after, d.ask.after
    tot = nb + na
    dep = ((nb - na) / tot) if tot > 0 else None
    return c, dep


# ------------------------------------------------------------- the scorer

def score(looks: list[Look], tape: Tape) -> None:
    """ONE scorer, run ONCE. Every arm reads what it produced.

    Both possible sides are walked for every look, because the arms are
    mirrors and between them they use both. Thirty arms re-scoring the
    same looks was the whole cost of the run, and re-walking per arm also
    invites a subtle difference between what one arm measured and what
    another did. Walking here, once, makes every table below a view of
    one measurement rather than thirty.

    A filter never changes the side (see `arm_sides`), so the side an arm
    assigns is always one of the two walked here.
    """
    for l in looks:
        l.cross = {}
        for side in ("buy", "sell"):
            long = side == "buy"
            entry = l.ask if long else l.bid
            sgn = 1.0 if long else -1.0
            t_px = [entry + sgn * t * TICK for t in TARGETS]
            s_px = [entry - sgn * s * TICK for s in STOP_TICKS]
            l.cross[side] = tape.crossings(l.ts, entry, t_px, s_px, long,
                                           HORIZON_S)


def outcome(l: Look, side: str, t_i: int, s_i: int,
            cap: float) -> bool | None:
    t_at, s_at, _ = l.cross[side]
    return Tape.decide(t_at[t_i], s_at[s_i], l.ts + cap)


def excursion_of(l: Look, side: str):
    return l.cross[side][2]


def rate(vals) -> tuple[int, int, int]:
    """(hits, resolved, unresolved). Unresolved is never a loss."""
    hits = sum(1 for v in vals if v is True)
    miss = sum(1 for v in vals if v is False)
    return hits, hits + miss, sum(1 for v in vals if v is None)


def breakeven(target: float, stop: float, rt_ticks: float) -> float:
    """The rate a geometry needs before it pays for itself.

        p*T - (1-p)*S - rt = 0   ->   p = (S + rt) / (T + S)

    All three in ticks. A stop tighter than the target drops this fast:
    sixteen ticks against four needs 25%, where ten against ten needs 55%.
    That difference is the whole reason the earlier studies answered a
    question nobody asked.
    """
    if target + stop <= 0:
        return 1.0
    return (stop + rt_ticks) / (target + stop)


def wilson(hits: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    z = 1.96
    p = hits / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - half), min(1.0, c + half)


def cell(vals) -> str:
    hits, n, none = rate(vals)
    if n == 0:
        return "    —    "
    if n < MIN_ARM:
        return f" {hits}/{n}   "
    return f" {hits / n:5.1%} "


# --------------------------------------------------------------- reporting

def arm_sides(looks: list[Look], base: str, filt: str = "",
              cut: float = 0.0) -> dict[int, str]:
    """A side per eligible look. A filter only ever REMOVES looks."""
    out: dict[int, str] = {}
    for i, l in enumerate(looks):
        side = l.continuation if base == "continuation" else l.reversal
        if not side:
            continue
        if filt == "structure" and not l.structure_agrees():
            continue
        if filt == "structure_against" and l.structure_agrees():
            continue
        if filt == "flow" and not l.flow_agrees(side):
            continue
        if filt == "flow_against" and l.flow_agrees(side):
            continue
        if filt in ("tape", "tape_against"):
            ag = l.tape_agrees(side)
            if ag is None:
                continue
            if filt == "tape" and not ag:
                continue
            if filt == "tape_against" and ag:
                continue
        if filt in ("book", "book_against"):
            ag = l.book_agrees(side, cut)
            if ag is None:
                continue
            if filt == "book" and not ag:
                continue
            if filt == "book_against" and ag:
                continue
        if filt == "combined":
            if not l.structure_agrees() or not l.flow_agrees(side):
                continue
            if l.tape_agrees(side) is not True:
                continue
            ag = l.book_agrees(side, cut)
            if ag is not True:
                continue
        out[i] = side
    return out


def shuffled(sides: dict[int, str], seed: int) -> dict[int, str]:
    """The same calls on the same timestamps, with the sides permuted.

    This is the null that matters. It holds the sample, the entries and the
    geometry fixed and varies only the one thing under test -- the call.
    """
    rng = random.Random(seed)
    keys = list(sides)
    vals = [sides[k] for k in keys]
    rng.shuffle(vals)
    return dict(zip(keys, vals))


def report(label: str, looks: list[Look], tape: Tape,
           sides: dict[int, str], rt_ticks: float,
           cap: float = 300.0) -> dict:
    """One arm at the headline geometries, against their own break-evens."""
    if not sides:
        print(f"{label:<32} (no eligible readings)")
        return {"n": 0}

    row = [f"{label:<32}", f"n={len(sides):<6}"]
    out = {"n": len(sides)}
    for t, s in HEADLINE:
        a, b = TARGETS.index(t), STOP_TICKS.index(s)
        vals = [outcome(looks[i], side, a, b, cap)
                for i, side in sides.items()]
        hits, n, _ = rate(vals)
        row.append(cell(vals))
        row.append(_verdict(hits, n, breakeven(t, s, rt_ticks)))
        out[f"{t}/{s}"] = (hits, n)
    ex = [excursion_of(looks[i], side) for i, side in sides.items()]
    out["mfe"] = _median([e[0] for e in ex if e is not None])
    out["mae"] = _median([e[1] for e in ex if e is not None])
    print("".join(row))
    return out


def _verdict(hits: int, n: int, need: float) -> str:
    """Clears its own break-even, or does not. Never a judgement."""
    if n < MIN_ARM:
        return "  ---  "
    return "  over " if (hits / n) > need else "  under"


def surface(looks: list[Look], tape: Tape, sides: dict[int, str],
            rt_ticks: float, cap: float) -> None:
    """Target against stop, so the shape is visible rather than two cells.

    Every cell prints its own break-even underneath. A geometry chosen off
    this table is chosen on the outcome, which is why the break-even is
    printed with it -- the question is which cells clear their own bar, not
    which cell is highest.
    """
    if not sides:
        print("  (no eligible readings)")
        return

    print(f"  {'target':>8}" + "".join(f"{s}t stop".rjust(16)
                                       for s in STOP_TICKS))
    for a, t in enumerate(TARGETS):
        row = [f"  {t:>5}t  "]
        for b, s in enumerate(STOP_TICKS):
            vals = [outcome(looks[i], side, a, b, cap)
                    for i, side in sides.items()]
            hits, n, _ = rate(vals)
            need = breakeven(t, s, rt_ticks)
            if n < MIN_ARM:
                row.append(f"{hits}/{n}".rjust(16))
            else:
                row.append(f"{hits / n:5.1%} vs {need:4.0%}".rjust(16))
        print("".join(row))


def _median(xs):
    if not xs:
        return None
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


def header() -> None:
    bits = [f"{'arm':<32}", f"{'':<8}"]
    for t, s in HEADLINE:
        bits.append(f" {t}t/{s}t ".rjust(9))
        bits.append(f"{'':<7}")
    print("".join(bits))
    print("-" * 66)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path")
    ap.add_argument("--symbol", default="ES")
    ap.add_argument("--flip", action="store_true",
                    help="aggressor convention is inverted in this file")
    ap.add_argument("--rt-ticks", type=float, default=DEFAULT_RT_TICKS)
    ap.add_argument("--seed", type=int, default=20261015)
    args = ap.parse_args(argv)

    print(f"file      {args.path}")
    print(f"costs     {args.rt_ticks} ticks round trip, entry crosses "
          f"the spread")
    print(f"headline  " + ", ".join(
        f"{t}t target / {s}t stop ({t / 4:.0f}pt / {s / 4:.0f}pt), "
        f"needs {breakeven(t, s, args.rt_ticks):.1%}"
        for t, s in HEADLINE))
    print(f"time stop {', '.join(f'{c:.0f}s' for c in TIME_STOPS)} "
          f"(grid -- the number was not supplied)")
    print()

    for rev in REVERSALS:
        store = dbn.open_file(args.path)
        looks, tape, prints = walk(store, rev, symbol=args.symbol,
                                   flip=args.flip)
        cut = len(looks)
        print("=" * 78)
        print(f"REVERSAL {rev} TICKS -- {len(looks)} confirmed pivots over "
              f"{prints:,} prints")
        print("=" * 78)
        if not looks:
            continue

        score(looks, tape)          # once; every table below reads this

        med_leg = _median([l.leg_ticks for l in looks])
        med_s = _median([l.leg_s for l in looks])
        print(f"median leg {med_leg:.1f} ticks over {med_s:.1f}s; "
              f"{len(looks) / max(1.0, (tape.ts[-1] - tape.ts[0]) / 60.0):.1f}"
              f" pivots/min")
        print()
        header()

        mid_cap = TIME_STOPS[len(TIME_STOPS) // 2]
        plan = [
            ("continuation (blind)", "continuation", "", 0.0),
            ("  + structure agrees", "continuation", "structure", 0.0),
            ("  + structure against", "continuation", "structure_against",
             0.0),
            ("  + leg flow agrees", "continuation", "flow", 0.0),
            ("  + leg flow against", "continuation", "flow_against", 0.0),
            ("  + live tape confirms", "continuation", "tape", 0.0),
            ("  + live tape denies", "continuation", "tape_against", 0.0),
            ("reversal (blind)", "reversal", "", 0.0),
            ("  + structure agrees", "reversal", "structure", 0.0),
            ("  + leg flow agrees", "reversal", "flow", 0.0),
            ("  + live tape confirms", "reversal", "tape", 0.0),
        ]
        for label, base_name, filt, c in plan:
            report(label, looks, tape,
                   arm_sides(looks, base_name, filt, c), args.rt_ticks,
                   cap=mid_cap)

        for c in BOOK_CUTS:
            for base_name in ("continuation", "reversal"):
                report(f"{base_name[:4]} + book agrees @{c:.2f}", looks,
                       tape, arm_sides(looks, base_name, "book", c),
                       args.rt_ticks, cap=mid_cap)
                report(f"{base_name[:4]} + book against @{c:.2f}", looks,
                       tape, arm_sides(looks, base_name, "book_against", c),
                       args.rt_ticks, cap=mid_cap)
            report(f"combined @{c:.2f}", looks, tape,
                   arm_sides(looks, "continuation", "combined", c),
                   args.rt_ticks, cap=mid_cap)

        print()
        base = arm_sides(looks, "continuation", "", 0.0)
        report("SHUFFLED CONTROL (continuation)", looks, tape,
               shuffled(base, args.seed), args.rt_ticks, cap=mid_cap)

        print()
        n_tr = int(cut * TRAIN_SHARE)
        for name, lo, hi in (("TRAIN (first 60%)", 0, n_tr),
                             ("HOLDOUT (last 40%)", n_tr, cut)):
            sub = {i: s for i, s in base.items() if lo <= i < hi}
            report(f"{name} continuation", looks, tape, sub, args.rt_ticks,
                   cap=mid_cap)

        print()
        print(f"GEOMETRY SURFACE -- continuation (blind), {mid_cap:.0f}s "
              f"time stop. Each cell is the rate against its own break-even.")
        surface(looks, tape, base, args.rt_ticks, mid_cap)

        print()
        print("TIME STOP SENSITIVITY -- continuation (blind), "
              f"{HEADLINE[0][0]}t / {HEADLINE[0][1]}t")
        t, s = HEADLINE[0]
        a, b = TARGETS.index(t), STOP_TICKS.index(s)
        for c in TIME_STOPS:
            hits, n, none = rate([outcome(looks[i], side, a, b, c)
                                  for i, side in base.items()])
            need = breakeven(t, s, args.rt_ticks)
            if n < MIN_ARM:
                print(f"  {c:>5.0f}s   {hits}/{n}   ({none} unresolved)")
            else:
                print(f"  {c:>5.0f}s   {hits / n:5.1%} vs {need:4.0%} "
                      f"over n={n}   ({none} unresolved)")

        print()
        ex = [e for i, side in base.items()
              if (e := excursion_of(looks[i], side)) is not None]
        if ex:
            print(f"excursion after a confirmed pivot, {len(ex)} readings, "
                  f"{HORIZON_S:.0f}s cap -- median favourable "
                  f"{_median([a for a, _ in ex]):.1f}t, median adverse "
                  f"{_median([b for _, b in ex]):.1f}t")
            for q in (0.25, 0.5, 0.75, 0.9):
                fa = sorted(a for a, _ in ex)
                ad = sorted(b for _, b in ex)
                k = min(len(fa) - 1, int(q * len(fa)))
                print(f"    p{q * 100:.0f}  favourable {fa[k]:6.1f}t   "
                      f"adverse {ad[k]:6.1f}t")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
