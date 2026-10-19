"""The resting book, read diagonally, accumulated from the candle open.

    from liqmap.bookdiag import BookDiagonal
    bd = BookDiagonal()
    bd.observe(rec)                 # an MBP-10 record
    r = bd.reading()
    print(r.winning_now, r.winning_since_open, r.size_now)

DIFFERENT DATA FROM THE FOOTPRINT, DELIBERATELY. The footprint counts
contracts that TRADED and which side crossed to get them. This counts
contracts that are RESTING and have not traded at all. They answer
different questions and neither substitutes for the other:

    footprint   who paid, and where
    this        who is standing there, and where

Resting size is the weaker evidence of the two -- it can be pulled
the instant it is leaned on, and a print cannot be taken back. That
is a reason to measure it separately and read it as what it is, not
a reason to skip it.

THE FOOTPRINT IS NOT TOUCHED BY THIS MODULE.

WHY "DIAGONAL" HAS TO MEAN PER-LEVEL HERE

Summing a diagonal across every level collapses:

    sum over k of (bid_k - ask_k)  ==  total bid - total ask

The level pairing cancels out, and what is left is plain depth
imbalance with extra steps. So the diagonal is kept per level -- is
the fight at the touch, or four ticks back? -- and the aggregate is
weighted by distance from the touch, where a contract resting one
tick away is not the same promise as one resting nine ticks away.

CONTRACTS AND ORDERS ARE BOTH COUNTED. MBP-10 carries `sz` and `ct`
per level, and they disagree in a way that matters: 500 contracts in
one order is a single participant who can withdraw in one message;
500 across forty orders is forty participants who would all have to
change their minds. The second is harder to move.

ACCUMULATION IS TIME-WEIGHTED, NEVER SUMMED. The book is a STATE,
not a flow. Adding up snapshots counts the same resting contract
once per update, so a quiet book with a large bid would accumulate
enormous "bid pressure" purely by sitting still and being re-reported.
Each snapshot is weighted by how long it stood.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import dbn

DEPTH = 10
DEFAULT_MULTIPLIER = 2.0        # a level is "won" at this ratio
MAX_GAP_S = 5.0                 # a longer silence is not counted


def _weight(k: int) -> float:
    """How much level k counts. 1, 1/2, 1/3 ... from the touch.

    A contract resting one tick from the touch is a different promise
    from one resting nine ticks away: the first is about to be
    traded against, the second has time to leave.
    """
    return 1.0 / (k + 1)


@dataclass(frozen=True, slots=True)
class LevelRead:
    """One rung of the ladder, both sides."""

    depth: int                  # 0 is the touch
    bid_px: float = 0.0
    ask_px: float = 0.0
    bid_sz: int = 0
    ask_sz: int = 0
    bid_ct: int = 0
    ask_ct: int = 0

    @property
    def size_imb(self) -> float:
        t = self.bid_sz + self.ask_sz
        return ((self.bid_sz - self.ask_sz) / t) if t else 0.0

    @property
    def count_imb(self) -> float:
        t = self.bid_ct + self.ask_ct
        return ((self.bid_ct - self.ask_ct) / t) if t else 0.0

    def winner(self, mult: float = DEFAULT_MULTIPLIER) -> str:
        """Which side holds this rung, by the same ratio test the
        footprint uses on traded volume. '' when neither does."""
        b, a = self.bid_sz, self.ask_sz
        if b <= 0 and a <= 0:
            return ""
        if a <= 0 or (b > 0 and b >= mult * a):
            return "bid"
        if b <= 0 or (a > 0 and a >= mult * b):
            return "ask"
        return ""

    def as_dict(self, mult: float = DEFAULT_MULTIPLIER) -> dict:
        return {"depth": self.depth,
                "bid_px": self.bid_px, "ask_px": self.ask_px,
                "bid_sz": self.bid_sz, "ask_sz": self.ask_sz,
                "bid_ct": self.bid_ct, "ask_ct": self.ask_ct,
                "size_imb": round(self.size_imb, 4),
                "count_imb": round(self.count_imb, 4),
                "winner": self.winner(mult)}


@dataclass(frozen=True, slots=True)
class Reading:
    """Who is winning now, and who has been winning since the open."""

    # instantaneous
    size_now: float = 0.0       # distance-weighted, [-1, 1]
    order_now: float = 0.0
    touch_size: float = 0.0     # the touch alone
    touch_order: float = 0.0
    bid_sz: int = 0             # raw totals across the depth shown
    ask_sz: int = 0
    bid_ct: int = 0
    ask_ct: int = 0
    spread_ticks: float = 0.0
    levels: tuple = ()
    bid_rungs: int = 0          # rungs held by each side
    ask_rungs: int = 0

    # since the candle opened
    size_mean: float = 0.0      # time-weighted
    order_mean: float = 0.0
    seconds: float = 0.0
    updates: int = 0
    bid_seconds: float = 0.0    # time each side spent ahead
    ask_seconds: float = 0.0

    @staticmethod
    def _who(v: float, edge: float) -> str:
        if v > edge:
            return "bid"
        if v < -edge:
            return "ask"
        return ""

    def winning_now(self, edge: float = 0.1) -> str:
        return self._who(self.size_now, edge)

    def winning_since_open(self, edge: float = 0.1) -> str:
        return self._who(self.size_mean, edge)

    @property
    def held(self) -> float:
        """Share of the candle the bid side spent ahead, [0, 1].

        A side that is ahead right now and has been ahead for ninety
        per cent of the bar is a different claim from one that just
        got there, and an average alone cannot separate them.
        """
        t = self.bid_seconds + self.ask_seconds
        return (self.bid_seconds / t) if t else 0.5

    def as_dict(self, mult: float = DEFAULT_MULTIPLIER) -> dict:
        return {
            "size_now": round(self.size_now, 4),
            "order_now": round(self.order_now, 4),
            "touch_size": round(self.touch_size, 4),
            "touch_order": round(self.touch_order, 4),
            "bid_sz": self.bid_sz, "ask_sz": self.ask_sz,
            "bid_ct": self.bid_ct, "ask_ct": self.ask_ct,
            "spread_ticks": round(self.spread_ticks, 2),
            "bid_rungs": self.bid_rungs, "ask_rungs": self.ask_rungs,
            "size_mean": round(self.size_mean, 4),
            "order_mean": round(self.order_mean, 4),
            "held": round(self.held, 4),
            "seconds": round(self.seconds, 3),
            "updates": self.updates,
            "winning_now": self.winning_now(),
            "winning_since_open": self.winning_since_open(),
            "levels": [lv.as_dict(mult) for lv in self.levels],
        }


class BookDiagonal:
    """Reads MBP records, scoped to one candle.

    `reset()` at a bar boundary. Nothing here knows what a bar is --
    the caller owns that, so this can be driven by any bar rule
    without the two having to agree about anything else.
    """

    __slots__ = ("depth", "multiplier", "tick_size", "max_gap",
                 "_last", "_last_ts", "_w_size", "_w_order",
                 "_seconds", "updates", "_bid_s", "_ask_s", "opened")

    def __init__(self, depth: int = DEPTH,
                 multiplier: float = DEFAULT_MULTIPLIER,
                 tick_size: float = 0.25,
                 max_gap_s: float = MAX_GAP_S) -> None:
        self.depth = max(1, int(depth))
        self.multiplier = float(multiplier)
        self.tick_size = float(tick_size)
        self.max_gap = float(max_gap_s)
        self.reset()

    def reset(self, keep_book: bool = True) -> None:
        """Start a new candle.

        The STANDING BOOK IS KEPT. A bar closing does not cancel
        anyone's resting orders, so clearing the snapshot would blank
        `size_now` until the next record arrived and make the new
        candle's first reading say "no book" about a book that is
        plainly there. Only the accumulation restarts.

        `keep_book=False` is for a genuine discontinuity -- a
        reconnect, a session break -- where the old snapshot really
        has stopped describing anything.
        """
        if not keep_book or not hasattr(self, "_last"):
            self._last: tuple = ()
            self._last_ts = 0.0
        self._w_size = 0.0
        self._w_order = 0.0
        self._seconds = 0.0
        self.updates = 0
        self._bid_s = 0.0
        self._ask_s = 0.0
        self.opened = 0.0
        self.updates = 0

    # ---------------------------------------------------------- feed

    def observe(self, rec) -> bool:
        """One record in. True when it carried a book.

        The PREVIOUS snapshot is integrated over the time it stood,
        which is only known once the next one arrives. That is what
        makes this an average over the candle rather than an average
        over the update rate -- a book that is re-reported a thousand
        times in a burst must not outweigh one that stood quietly for
        a minute.
        """
        raw = dbn._get(rec, "levels")
        if not raw:
            return False
        if not isinstance(raw, (list, tuple)):
            raw = [raw]

        ts = dbn.seconds(dbn._get(rec, "ts_event")) or 0.0
        rows = []
        for k, lv in enumerate(raw[:self.depth]):
            rows.append(LevelRead(
                depth=k,
                bid_px=dbn.price(dbn._get(lv, "bid_px", 0)),
                ask_px=dbn.price(dbn._get(lv, "ask_px", 0)),
                bid_sz=int(dbn._get(lv, "bid_sz", 0) or 0),
                ask_sz=int(dbn._get(lv, "ask_sz", 0) or 0),
                bid_ct=int(dbn._get(lv, "bid_ct", 0) or 0),
                ask_ct=int(dbn._get(lv, "ask_ct", 0) or 0)))

        if self._last and self._last_ts and ts > self._last_ts:
            dt = ts - self._last_ts
            # A gap longer than the cap is a halt or a disconnect, not
            # a book that stood. Integrating it would let one stale
            # snapshot dominate the whole candle.
            if dt <= self.max_gap:
                s, o = self._aggregate(self._last)
                self._w_size += s * dt
                self._w_order += o * dt
                self._seconds += dt
                if s > 0:
                    self._bid_s += dt
                elif s < 0:
                    self._ask_s += dt

        if not self.opened:
            self.opened = ts
        self._last = tuple(rows)
        self._last_ts = ts
        self.updates += 1
        return True

    # ---------------------------------------------------------- read

    def _aggregate(self, rows) -> tuple:
        """Distance-weighted size and order imbalance of a snapshot."""
        ws = wo = tw = 0.0
        for lv in rows:
            w = _weight(lv.depth)
            tw += w
            ws += w * lv.size_imb
            wo += w * lv.count_imb
        if not tw:
            return (0.0, 0.0)
        return (ws / tw, wo / tw)

    def reading(self) -> Reading:
        rows = self._last
        if not rows:
            return Reading()
        size_now, order_now = self._aggregate(rows)
        touch = rows[0]
        bid_rungs = sum(1 for lv in rows
                        if lv.winner(self.multiplier) == "bid")
        ask_rungs = sum(1 for lv in rows
                        if lv.winner(self.multiplier) == "ask")
        spread = 0.0
        if touch.ask_px > 0 and touch.bid_px > 0 and self.tick_size > 0:
            spread = (touch.ask_px - touch.bid_px) / self.tick_size
        secs = self._seconds
        return Reading(
            size_now=size_now, order_now=order_now,
            touch_size=touch.size_imb, touch_order=touch.count_imb,
            bid_sz=sum(lv.bid_sz for lv in rows),
            ask_sz=sum(lv.ask_sz for lv in rows),
            bid_ct=sum(lv.bid_ct for lv in rows),
            ask_ct=sum(lv.ask_ct for lv in rows),
            spread_ticks=spread, levels=rows,
            bid_rungs=bid_rungs, ask_rungs=ask_rungs,
            size_mean=(self._w_size / secs) if secs else size_now,
            order_mean=(self._w_order / secs) if secs else order_now,
            seconds=secs, updates=self.updates,
            bid_seconds=self._bid_s, ask_seconds=self._ask_s)
