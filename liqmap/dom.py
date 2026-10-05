"""What is happening to the resting book: stacking, pulling, migration.

THE ONE THING THAT MAKES THIS WORK, AND ITS ABSENCE MAKES IT NOISE

A level getting smaller is either being TRADED or being PULLED, and the
book alone cannot tell you which. Both look identical: size was there, now
it is not.

They mean opposite things. Size traded away is demand being met -- somebody
wanted it and paid for it. Size pulled is demand withdrawn before anyone
touched it, which is the thing that makes a market fall away under price.
A module that reports "liquidity disappearing" without separating the two
is mostly reporting that trading happened.

So every shrink is checked against the tape at that price:

    cancelled = max(0, size_lost - volume_printed_at_that_price)

That single subtraction is why this file needs the trades as well as the
book, and it is the whole difference between a DOM read and a volume
chart with extra steps.

THE SECOND TRAP IS RANK

A book is naturally indexed from the touch: best bid, second, third. Index
by rank and every price move looks like a catastrophe -- price ticks up
one, the "third level" is now a different price, and the comparison reports
a wall of liquidity that vanished when nothing happened at all.

Everything here is keyed to ABSOLUTE PRICE. A level is the same level only
if it is the same number.

THE THIRD TRAP IS THE EDGE OF THE WINDOW

The feed shows a fixed number of levels. When price moves, prices at the far
end fall out of view -- and a price that left the window has not been
cancelled, it has stopped being reported. Comparing it to zero invents a
withdrawal every time the market moves.

So every comparison runs only over prices BOTH snapshots could actually
see, and the overlap is reported so a reading built on two levels is not
mistaken for one built on twenty.

WHAT IS NOT HERE: SPOOFING

This can see that size appeared at a price and left without anything
trading into it. It cannot see why. A market maker widening ahead of a
number does exactly that, and so does a participant with no intention of
being filled -- the book is identical and intent is not in the data. The
reading is called `pulled_untested`, which is what was observed, and no
word here claims to know the reason.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Literal, Sequence

# Levels kept per side. Deeper than the near-book read, because migration
# is a question about where size is MOVING to, which needs somewhere to
# move to.
DEPTH = 10

# The ladder is sampled at most this often. DOM behaviour is a question
# about seconds; sampling every push would keep thousands of snapshots a
# minute to answer it no better.
SAMPLE_EVERY_S = 0.25

# How far back the reading looks by default.
WINDOW_S = 20.0

# Prices both snapshots must share before a comparison means anything.
MIN_OVERLAP = 3

Side = Literal["bid", "ask"]


@dataclass(frozen=True)
class Rung:
    px: float
    sz: float


@dataclass(frozen=True)
class Ladder:
    """One book photograph, keyed by absolute price."""

    ts: float
    bids: tuple[Rung, ...] = ()
    asks: tuple[Rung, ...] = ()

    @property
    def bid(self) -> float:
        return self.bids[0].px if self.bids else 0.0

    @property
    def ask(self) -> float:
        return self.asks[0].px if self.asks else 0.0

    @property
    def mid(self) -> float:
        b, a = self.bid, self.ask
        return (b + a) / 2.0 if b > 0 and a > 0 else (b or a)

    def side(self, which: Side) -> tuple[Rung, ...]:
        return self.bids if which == "bid" else self.asks

    def sizes(self, which: Side) -> dict[float, float]:
        return {r.px: r.sz for r in self.side(which)}

    def span(self, which: Side) -> tuple[float, float]:
        """The price range this snapshot could actually see on one side."""
        rs = self.side(which)
        if not rs:
            return (0.0, 0.0)
        pxs = [r.px for r in rs]
        return (min(pxs), max(pxs))


def ladder_of(book, depth: int = DEPTH, now: float | None = None
              ) -> Ladder | None:
    if book is None or not getattr(book, "bids", None) \
            or not getattr(book, "asks", None):
        return None
    ts = now
    if ts is None:
        ts = book.ts if getattr(book, "ts", None) is not None else time.time()
    return Ladder(
        ts=float(ts),
        bids=tuple(Rung(float(l.px), float(l.sz)) for l in book.bids[:depth]),
        asks=tuple(Rung(float(l.px), float(l.sz)) for l in book.asks[:depth]),
    )


# ---------------------------------------------------------------------------
# what traded, and where
# ---------------------------------------------------------------------------


def volume_by_price(trades: Sequence, start: float, end: float,
                    tick: float = 0.0) -> dict[float, float]:
    """Size printed at each price in a time window.

    Keyed by the traded price itself. When a tick is supplied the prices
    are snapped to it, because a venue that reports 100.0000001 and a book
    that quotes 100.0 are talking about the same level and a dictionary
    does not know that.
    """
    out: dict[float, float] = {}
    for t in trades or []:
        ts = float(getattr(t, "ts", 0.0))
        if ts < start or ts > end:
            continue
        px = float(getattr(t, "px", 0.0))
        if px <= 0:
            continue
        if tick > 0:
            px = round(px / tick) * tick
            px = round(px, 10)
        out[px] = out.get(px, 0.0) + float(getattr(t, "sz", 0.0))
    return out


# ---------------------------------------------------------------------------
# the reading
# ---------------------------------------------------------------------------


@dataclass
class SideFlow:
    """What happened to one side of the book over the window."""

    side: Side
    added: float = 0.0          # size that appeared at prices already visible
    traded: float = 0.0         # size that left and printed
    cancelled: float = 0.0      # size that left and did not print
    overlap: int = 0            # prices both snapshots could see
    before: float = 0.0         # total resting size then
    after: float = 0.0          # and now

    @property
    def net(self) -> float:
        return self.after - self.before

    @property
    def churn(self) -> float:
        return self.added + self.traded + self.cancelled

    @property
    def pull_share(self) -> float:
        """Of the size that LEFT, how much was withdrawn rather than hit."""
        gone = self.traded + self.cancelled
        return (self.cancelled / gone) if gone > 0 else 0.0

    @property
    def stacking(self) -> bool:
        return self.measured and self.added > self.cancelled and self.net > 0

    @property
    def pulling(self) -> bool:
        return self.measured and self.cancelled > self.added and self.net < 0

    @property
    def measured(self) -> bool:
        return self.overlap >= MIN_OVERLAP

    def to_dict(self) -> dict:
        return {"side": self.side, "added": round(self.added, 4),
                "traded": round(self.traded, 4),
                "cancelled": round(self.cancelled, 4),
                "net": round(self.net, 4),
                "pull_share": round(self.pull_share, 3),
                "overlap": self.overlap, "measured": self.measured,
                "stacking": self.stacking, "pulling": self.pulling}


@dataclass
class DomRead:
    """Layer four: what the resting book is doing."""

    window_s: float = WINDOW_S
    span_s: float = 0.0
    samples: int = 0
    bid: SideFlow = field(default_factory=lambda: SideFlow("bid"))
    ask: SideFlow = field(default_factory=lambda: SideFlow("ask"))

    # Where the weight of resting size sits, as a distance from the mid in
    # price terms. Positive means it moved AWAY from the touch.
    bid_drift: float = 0.0
    ask_drift: float = 0.0

    # Size withdrawn at prices in the direction price is travelling, before
    # price got there.
    vanishing_ahead: float = 0.0
    ahead_side: Side | None = None

    # Size that appeared and left with nothing ever printing at it. What was
    # observed; not a claim about why.
    pulled_untested: float = 0.0

    moved: float = 0.0          # what the mid did over the window

    @property
    def measured(self) -> bool:
        return self.bid.measured or self.ask.measured

    def describe(self) -> str:
        if not self.measured:
            return ("the book has not held still long enough at any price "
                    "to say what it is doing")
        bits = []
        for s in (self.bid, self.ask):
            if s.stacking:
                bits.append(f"{s.side}s stacking")
            elif s.pulling:
                bits.append(f"{s.side}s pulling "
                            f"({s.pull_share:.0%} of what left was withdrawn)")
        if self.vanishing_ahead > 0 and self.ahead_side:
            bits.append(f"{self.vanishing_ahead:,.0f} withdrawn on the "
                        f"{self.ahead_side} ahead of price")
        return "; ".join(bits) or "the book is holding"

    def to_dict(self) -> dict:
        return {"window_s": self.window_s, "span_s": round(self.span_s, 2),
                "samples": self.samples, "measured": self.measured,
                "bid": self.bid.to_dict(), "ask": self.ask.to_dict(),
                "bid_drift": round(self.bid_drift, 6),
                "ask_drift": round(self.ask_drift, 6),
                "vanishing_ahead": round(self.vanishing_ahead, 4),
                "ahead_side": self.ahead_side,
                "pulled_untested": round(self.pulled_untested, 4),
                "moved": round(self.moved, 6),
                "describe": self.describe()}


def _weight_centre(rungs: Sequence[Rung], mid: float) -> float:
    """Size-weighted distance of resting size from the mid."""
    total = sum(r.sz for r in rungs)
    if total <= 0 or mid <= 0:
        return 0.0
    return sum(abs(r.px - mid) * r.sz for r in rungs) / total


def _side_flow(side: Side, old: Ladder, new: Ladder,
               traded: dict[float, float]) -> SideFlow:
    """Compare one side, price by price, over what both could see.

    ONLY OVER THE OVERLAP. A price that left the feed's window has not
    been cancelled, it has stopped being reported, and comparing it to
    zero invents a withdrawal every time the market moves.
    """
    out = SideFlow(side=side)
    a, b = old.sizes(side), new.sizes(side)
    lo_a, hi_a = old.span(side)
    lo_b, hi_b = new.span(side)
    if not a or not b:
        return out
    lo, hi = max(lo_a, lo_b), min(hi_a, hi_b)
    if hi < lo:
        return out

    for px in sorted(set(a) | set(b)):
        if px < lo or px > hi:
            continue
        out.overlap += 1
        was, now = a.get(px, 0.0), b.get(px, 0.0)
        out.before += was
        out.after += now
        d = now - was
        if d > 0:
            out.added += d
        elif d < 0:
            lost = -d
            hit = traded.get(px, 0.0)
            # A SHRINK IS A FILL OR A CANCEL, and the book cannot tell you
            # which. What printed at the price is the part that was paid
            # for; the rest was withdrawn.
            out.traded += min(lost, hit)
            out.cancelled += max(0.0, lost - hit)
    return out


def read(old: Ladder, new: Ladder, traded: dict[float, float] | None = None,
         window_s: float = WINDOW_S, samples: int = 2) -> DomRead:
    """What the book did between two photographs."""
    t = traded or {}
    out = DomRead(window_s=window_s, samples=samples,
                  span_s=max(0.0, new.ts - old.ts))
    out.bid = _side_flow("bid", old, new, t)
    out.ask = _side_flow("ask", old, new, t)

    out.moved = new.mid - old.mid
    out.bid_drift = (_weight_centre(new.bids, new.mid)
                     - _weight_centre(old.bids, old.mid))
    out.ask_drift = (_weight_centre(new.asks, new.mid)
                     - _weight_centre(old.asks, old.mid))

    # Ahead of price: the side price is walking into. Going up, that is the
    # offers; going down, the bids. Withdrawal there is liquidity leaving
    # before price arrives, which is a different thing from it being eaten
    # on arrival -- and the two are only separable because the cancelled
    # figure already had the prints taken out of it.
    if out.moved > 0:
        out.ahead_side = "ask"
        out.vanishing_ahead = out.ask.cancelled
    elif out.moved < 0:
        out.ahead_side = "bid"
        out.vanishing_ahead = out.bid.cancelled

    # Appeared and left with nothing ever printing at it.
    for side in ("bid", "ask"):
        a, b = old.sizes(side), new.sizes(side)
        for px, was in a.items():
            if b.get(px, 0.0) <= 0 and t.get(px, 0.0) <= 0:
                lo, hi = new.span(side)
                if lo <= px <= hi:
                    out.pulled_untested += was
    return out


class DomWatch:
    """Keeps the ladder over a short window and answers layer four.

    Sampled rather than taken on every push: DOM behaviour is a question
    about seconds, and keeping thousands of photographs a minute answers
    it no better while costing real memory on a container that has little.
    """

    def __init__(self, window_s: float = WINDOW_S, depth: int = DEPTH,
                 every_s: float = SAMPLE_EVERY_S, max_snaps: int = 400):
        self.window_s = window_s
        self.depth = depth
        self.every_s = every_s
        self._snaps: deque[Ladder] = deque(maxlen=max_snaps)

    def add(self, book, now: float | None = None) -> bool:
        lad = ladder_of(book, self.depth, now)
        if lad is None:
            return False
        if self._snaps and (lad.ts - self._snaps[-1].ts) < self.every_s:
            return False
        self._snaps.append(lad)
        return True

    def read(self, trades: Sequence = (), window_s: float | None = None,
             tick: float = 0.0, now: float | None = None) -> DomRead:
        w = self.window_s if window_s is None else window_s
        if len(self._snaps) < 2:
            return DomRead(window_s=w, samples=len(self._snaps))
        end = float(now) if now is not None else self._snaps[-1].ts
        cut = end - w
        inside = [s for s in self._snaps if s.ts >= cut]
        if len(inside) < 2:
            inside = list(self._snaps)[-2:]
        old, new = inside[0], inside[-1]
        by_px = volume_by_price(trades, old.ts, new.ts, tick=tick)
        return read(old, new, by_px, window_s=w, samples=len(inside))

    def __len__(self) -> int:
        return len(self._snaps)
