"""The order book calls the candle. Nothing else is consulted.

WHAT THIS DELIBERATELY DOES NOT DO

No trend. No higher timeframe. No VWAP, no supply zones, no liquidation
magnets, no candle-history inference. Those all answer "what tends to happen
next", and every one of them was diluting the only question that matters for
a scalp: which way is THIS candle going, right now.

The book is the structure. It updates thousands of times a minute, it is the
actual mechanism by which price moves, and it is available in full. So this
module reads it and answers one question.

THE SIGNAL THAT DOES MOST OF THE WORK IS MICROPRICE

Mid-price is a fiction: it sits halfway between two queues that are almost
never the same size. The size-weighted mid -- the microprice --

    micro = (bid_px * ask_size + ask_px * bid_size) / (bid_size + ask_size)

leans toward the side with LESS size, because that is the side that gets
consumed first. Ten million on the bid against one million on the offer and
the microprice sits almost on the ask: the next print goes up, and it goes up
because there is nothing in the way.

Expressed against the half-spread it is scale-free and bounded:

    tilt = (micro - mid) / (spread / 2)     in -1 .. +1

That one number is the best short-horizon directional reading a book gives.
Everything else here either confirms it or explains why it is about to fail.

THE DYNAMICS MATTER MORE THAN THE SNAPSHOT

A book photograph tells you what is resting. What you can trade is what is
HAPPENING to it: a bid queue being eaten tick by tick, an offer being pulled
before it is touched, a level that keeps coming back however often it is hit.
Displayed size can be cancelled in a microsecond and frequently is. Size that
gets consumed and REPLACED has been paid for, so replenishment is weighted
harder than depth.

SPEED IS THE FEATURE

Every reading here is O(levels) on a snapshot already in memory, with a short
rolling window for the dynamics. It is meant to run on every book push --
thousands a minute, early morning included -- so nothing in here fetches,
blocks, or allocates more than it must.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from .flow import Book

Direction = Literal["up", "down", "flat"]

# Levels from the touch that count as "the near book". Beyond this, size is
# too far away to be consumed inside one candle of a scalping timeframe.
NEAR_LEVELS = 5

# How long the dynamics window looks back, in seconds. Short on purpose: the
# question is what the book is doing NOW, not what it did a minute ago.
DYNAMICS_WINDOW_S = 20.0

# Below this, the book is balanced and saying nothing.
FLAT_BAND = 0.12

# The most one book update may contribute to the consumed/replaced counters:
# one whole level replacing or clearing itself. Accumulating raw size let a
# single outsized quote decide the call, and allowing more than 1.0 per
# update let that same quote clear the evidence bar below on its own.
MAX_REL_STEP = 1.0

# How much accumulated change the consumed/replaced ratios need before they
# speak at full strength — about two levels' worth. Without this the FIRST
# event reads as a perfect 1.0, because a ratio of two tiny numbers is noise
# scaled up to certainty.
MIN_EVIDENCE = 2.0


@dataclass(frozen=True)
class Snap:
    """One book photograph, reduced to what matters."""

    ts: float
    bid: float
    ask: float
    bid_sz: float          # size at the touch
    ask_sz: float
    near_bid: float        # notional within NEAR_LEVELS
    near_ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        m = self.mid
        return (self.spread / m * 10_000.0) if m > 0 else 0.0

    @property
    def microprice(self) -> float:
        """Size-weighted mid. Leans toward the thinner side, because that is
        the side that gets consumed first."""
        total = self.bid_sz + self.ask_sz
        if total <= 0:
            return self.mid
        return (self.bid * self.ask_sz + self.ask * self.bid_sz) / total

    @property
    def tilt(self) -> float:
        """Microprice against the half-spread. -1 pinned to the bid, +1 to
        the ask. Scale-free, so it reads the same on a $4 coin and a $100k one."""
        half = self.spread / 2.0
        if half <= 0:
            return 0.0
        return max(-1.0, min(1.0, (self.microprice - self.mid) / half))

    @property
    def imbalance(self) -> float:
        """Near-book notional, bid versus ask."""
        t = self.near_bid + self.near_ask
        return (self.near_bid - self.near_ask) / t if t > 0 else 0.0


def snap(book: Book, levels: int = NEAR_LEVELS, now: float | None = None
         ) -> Snap | None:
    """Reduce a full book to the fields the read needs. O(levels)."""
    if book is None or not book.bids or not book.asks:
        return None
    b, a = book.bids[0], book.asks[0]
    if b.px <= 0 or a.px <= 0:
        return None
    # `book.ts or time.time()` is wrong: a timestamp of exactly zero is
    # falsy, so it silently becomes the wall clock and puts that snapshot
    # billions of seconds from its neighbours. Every window then collapses.
    ts = now
    if ts is None:
        ts = book.ts if book.ts is not None else time.time()
    return Snap(
        ts=ts,
        bid=b.px, ask=a.px, bid_sz=b.sz, ask_sz=a.sz,
        near_bid=sum(l.notional for l in book.bids[:levels]),
        near_ask=sum(l.notional for l in book.asks[:levels]),
    )


@dataclass
class BookRead:
    """The answer, with every component that produced it."""

    ts: float
    mid: float
    microprice: float
    spread_bps: float

    tilt: float               # microprice against half-spread
    imbalance: float          # near-book size
    depletion: float          # touch queues shrinking: + bid eaten, - ask eaten
    replenish: float          # which side keeps coming back
    mid_drift_bps: float      # what the mid actually did over the window
    aggression: float         # tape: who is crossing, -1..+1
    absorbed: bool            # aggression not moving the mid

    samples: int = 0
    window_s: float = DYNAMICS_WINDOW_S
    notes: list[str] = field(default_factory=list)

    # Weights. Microprice tilt leads because it is the mechanism; replenish
    # beats raw imbalance because displayed size can be cancelled and
    # replaced size has been paid for.
    W = {"tilt": 1.0, "replenish": 0.8, "imbalance": 0.6,
         "depletion": 0.7, "aggression": 0.6, "drift": 0.5}

    @property
    def score(self) -> float:
        """-1 sellers, +1 buyers. The candle's direction right now."""
        parts = {
            "tilt": self.tilt,
            "replenish": self.replenish,
            "imbalance": self.imbalance,
            "depletion": -self.depletion,   # bid being eaten is bearish
            "aggression": (0.0 if self.absorbed else self.aggression),
            "drift": max(-1.0, min(1.0, self.mid_drift_bps / 10.0)),
        }
        total = sum(self.W.values())
        return round(sum(v * self.W[k] for k, v in parts.items()) / total, 4)

    @property
    def direction(self) -> Direction:
        s = self.score
        if s > FLAT_BAND:
            return "up"
        if s < -FLAT_BAND:
            return "down"
        return "flat"

    @property
    def conviction(self) -> float:
        """0..1. Magnitude, discounted while the window is still filling."""
        warm = min(self.samples / 10.0, 1.0)
        return round(min(abs(self.score) / 0.5, 1.0) * warm, 4)

    def components(self) -> list[dict]:
        return [
            {"name": "microprice tilt", "value": self.tilt,
             "weight": self.W["tilt"],
             "note": ("leaning to the offer — thinner above"
                      if self.tilt > 0 else
                      "leaning to the bid — thinner below"
                      if self.tilt < 0 else "balanced at the touch")},
            {"name": "replenishment", "value": self.replenish,
             "weight": self.W["replenish"],
             "note": ("bids keep coming back" if self.replenish > 0 else
                      "offers keep coming back" if self.replenish < 0 else
                      "neither side is replacing size")},
            {"name": "near imbalance", "value": self.imbalance,
             "weight": self.W["imbalance"],
             "note": f"{abs(self.imbalance):.0%} "
                     f"{'bid' if self.imbalance > 0 else 'offer'} heavy"},
            {"name": "queue depletion", "value": -self.depletion,
             "weight": self.W["depletion"],
             "note": ("the bid is being eaten" if self.depletion > 0 else
                      "the offer is being eaten" if self.depletion < 0 else
                      "queues holding")},
            {"name": "aggression", "value": (0.0 if self.absorbed
                                             else self.aggression),
             "weight": self.W["aggression"],
             "note": ("absorbed — aggression is not moving the mid"
                      if self.absorbed else
                      "buyers crossing" if self.aggression > 0 else
                      "sellers crossing" if self.aggression < 0 else
                      "no aggression")},
            {"name": "mid drift", "value": max(-1.0, min(1.0, self.mid_drift_bps / 10.0)),
             "weight": self.W["drift"],
             "note": f"mid {self.mid_drift_bps:+.2f}bps over {self.window_s:.0f}s"},
        ]

    def verdict(self) -> str:
        if self.samples < 3:
            return (f"Warming up — {self.samples} book updates so far. "
                    f"Needs a few seconds of the feed.")

        head = (f"{self.direction.upper()} — score {self.score:+.2f}, "
                f"conviction {self.conviction:.0%}, "
                f"from {self.samples} book updates over {self.window_s:.0f}s.")

        if self.absorbed:
            head += (" Aggression is being absorbed, so it is not counted: "
                     "somebody is taking the other side without price moving.")
        if abs(self.tilt) > 0.5:
            head += (f" The touch is {abs(self.tilt):.0%} tilted toward the "
                     f"{'offer' if self.tilt > 0 else 'bid'} — the thin side "
                     f"is {'above' if self.tilt > 0 else 'below'}.")
        if self.spread_bps > 0:
            head += f" Spread {self.spread_bps:.2f}bps."
        return head


class BookReader:
    """Rolling reader. Feed it book snapshots; ask it for a direction.

    Holds a short deque of reduced snapshots and nothing else. Designed to be
    called on every book push without thinking about cost.
    """

    def __init__(self, window_s: float = DYNAMICS_WINDOW_S,
                 levels: int = NEAR_LEVELS, max_snaps: int = 2000):
        self.window_s = window_s
        self.levels = levels
        self._snaps: deque[Snap] = deque(maxlen=max_snaps)
        # Replenishment counters, decayed rather than reset so a level that
        # was defended ten seconds ago still counts for something.
        self._bid_replaced = 0.0
        self._ask_replaced = 0.0
        self._bid_consumed = 0.0
        self._ask_consumed = 0.0

    def add(self, book: Book, now: float | None = None) -> None:
        s = snap(book, self.levels, now)
        if s is None:
            return

        prev = self._snaps[-1] if self._snaps else None
        self._snaps.append(s)
        if prev is None:
            return

        # Consumed vs replaced, per side, at the touch. A bid whose price
        # holds while its size falls is being eaten; size returning at the
        # same price is somebody paying to hold the level.
        #
        # Deltas are RELATIVE to the level's own size and capped per update.
        # Accumulating raw size lets one outsized print dominate every
        # counter -- a single five-hundred-lot quote swung the whole call by
        # a third, which is a book flicker deciding the trade.
        def rel(delta: float, base: float) -> float:
            if base <= 0:
                return 0.0
            return min(abs(delta) / base, MAX_REL_STEP)

        if s.bid == prev.bid:
            d = s.bid_sz - prev.bid_sz
            if d < 0:
                self._bid_consumed += rel(d, prev.bid_sz)
            else:
                self._bid_replaced += rel(d, prev.bid_sz)
        elif s.bid < prev.bid:
            # The bid stepped down: the level was cleared out.
            self._bid_consumed += MAX_REL_STEP

        if s.ask == prev.ask:
            d = s.ask_sz - prev.ask_sz
            if d < 0:
                self._ask_consumed += rel(d, prev.ask_sz)
            else:
                self._ask_replaced += rel(d, prev.ask_sz)
        elif s.ask > prev.ask:
            self._ask_consumed += MAX_REL_STEP

        decay = 0.995
        for attr in ("_bid_replaced", "_ask_replaced",
                     "_bid_consumed", "_ask_consumed"):
            setattr(self, attr, getattr(self, attr) * decay)

    def _window(self, now: float | None = None) -> list[Snap]:
        if not self._snaps:
            return []
        end = now if now is not None else self._snaps[-1].ts
        cutoff = end - self.window_s
        return [s for s in self._snaps if s.ts >= cutoff] or [self._snaps[-1]]

    def read(self, aggression: float = 0.0, now: float | None = None
             ) -> BookRead | None:
        """The current call. `aggression` is the tape's -1..+1 lean."""
        window = self._window(now)
        if not window:
            return None

        last = window[-1]
        first = window[0]

        mid_drift = ((last.mid - first.mid) / first.mid * 10_000.0
                     if first.mid > 0 else 0.0)

        # Depletion: positive means the BID queue is shrinking faster than the
        # offer, which is sellers eating support.
        # A ratio of two tiny numbers is noise amplified to full scale: the
        # very first event would read as a perfect 1.0 either way. So both
        # ratios are damped by how much evidence produced them, and only
        # reach full strength once several levels' worth has changed hands.
        def ratio(a: float, b: float) -> float:
            tot = a + b
            if tot <= 0:
                return 0.0
            confidence = min(tot / MIN_EVIDENCE, 1.0)
            return ((a - b) / tot) * confidence

        depletion = ratio(self._bid_consumed, self._ask_consumed)
        replenish = ratio(self._bid_replaced, self._ask_replaced)

        # Absorption, book-side: somebody is crossing and the mid is not
        # moving with them. The aggressor is paying and not being paid.
        absorbed = bool(abs(aggression) > 0.2 and abs(mid_drift) < 1.0)

        # Average the instantaneous readings across the window so a single
        # flickering snapshot cannot swing the call.
        n = len(window)
        tilt = sum(s.tilt for s in window) / n
        imb = sum(s.imbalance for s in window) / n

        return BookRead(
            ts=last.ts, mid=last.mid, microprice=last.microprice,
            spread_bps=last.spread_bps,
            tilt=tilt, imbalance=imb, depletion=depletion,
            replenish=replenish, mid_drift_bps=mid_drift,
            aggression=max(-1.0, min(1.0, aggression)), absorbed=absorbed,
            samples=n, window_s=self.window_s)

    @property
    def updates(self) -> int:
        return len(self._snaps)
