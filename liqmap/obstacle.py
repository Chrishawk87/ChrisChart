"""Can the aggressive flow actually get through what is in front of it?

    from liqmap.obstacle import Obstacles
    ob = Obstacles(tick_size=0.25, reach_ticks=16, window_s=10.0)
    ob.observe_book(rec)
    ob.observe_trade(ts, "buy", 5)
    c = ob.clearance("buy")
    print(c.ratio, c.seconds_to_clear, c.verdict)

The book direction says which way resting size is leaning. This asks
the next question: if buyers keep paying at the rate they have been,
do they eat the asks in front of them, or stall into them.

FOUR THINGS THIS GETS RIGHT THAT ARE EASY TO GET WRONG

1. LIKE FOR LIKE. Aggressive flow is contracts-over-time; a resting
   obstacle is contracts. Comparing a 100ms slice of flow against a
   multi-minute volume median is off by the ratio of the windows --
   a filter built that way is roughly nine hundred times too strict
   and goes silent, which reads as a quiet market rather than a
   broken test. Here the flow window is explicit, both sides are
   contracts, and the headline figure is SECONDS TO CLEAR, which is
   a unit a person can argue with.

2. THE OBSTACLE IS ONLY WHAT YOU CAN SEE. MBP-10 is ten levels. Ask
   for the resting size within four points -- sixteen ticks -- and on
   a one-tick book you can see nine of them. Truncating silently
   UNDERSTATES resistance, which makes the ratio easier to clear and
   lets through exactly the trades the filter exists to block. Every
   reading says how far it could actually see and whether it ran out
   of book.

3. HIDDEN SIZE IS STILL INVISIBLE. Icebergs and reserve orders do not
   appear at any depth. So the obstacle is a LOWER BOUND on real
   resistance, and the clearance ratio is therefore an UPPER BOUND on
   how easily flow gets through. A reading that says "this clears"
   means "nothing visible stops it".

4. A WALL THAT VANISHES WAS NEVER RESISTANCE. Resting size falls for
   two different reasons: it traded, or it left. Flow eating an
   obstacle is absorption; an obstacle withdrawing as flow approaches
   is something else entirely, and a filter that cannot tell them
   apart will keep reporting that buyers "broke through" walls that
   simply moved. Both are attributed and reported.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER. It reports a ratio.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from . import dbn

BUY, SELL = "buy", "sell"
CLEARS, BLOCKED, UNKNOWN = "clears", "blocked", "unknown"


@dataclass(frozen=True, slots=True)
class Obstacle:
    """The visible resting size standing in a direction."""

    side: str                   # "ask" blocks a buy, "bid" blocks a sell
    asked_ticks: int = 0        # how far out we wanted to look
    covered_ticks: int = 0      # how far the book actually reached
    truncated: bool = False     # the book ran out before we did
    size: int = 0               # resting contracts inside the reach
    orders: int = 0
    levels: int = 0
    nearest_px: float = 0.0
    farthest_px: float = 0.0

    @property
    def complete(self) -> bool:
        return not self.truncated

    def as_dict(self) -> dict:
        return {"side": self.side, "asked_ticks": self.asked_ticks,
                "covered_ticks": self.covered_ticks,
                "truncated": self.truncated, "size": self.size,
                "orders": self.orders, "levels": self.levels,
                "nearest_px": self.nearest_px,
                "farthest_px": self.farthest_px}


@dataclass(frozen=True, slots=True)
class Clearance:
    """Flow against the obstacle, in units that can be argued with."""

    direction: str = BUY
    obstacle: Obstacle = field(default_factory=Obstacle)
    flow: int = 0               # aggressive contracts, this direction
    opposing: int = 0           # aggressive contracts the other way
    window_s: float = 0.0
    rate: float = 0.0           # contracts per second
    ratio: float = 0.0          # flow / obstacle size
    seconds_to_clear: float | None = None
    threshold: float = 1.2
    verdict: str = UNKNOWN
    consumed: int = 0           # obstacle size that traded away
    pulled: int = 0             # obstacle size that simply left

    @property
    def passes(self) -> bool:
        return self.verdict == CLEARS

    @property
    def pulled_share(self) -> float:
        """How much of the obstacle's decline was withdrawal.

        High here and the "breakout" is a wall that moved, not one
        that was eaten.
        """
        t = self.consumed + self.pulled
        return (self.pulled / t) if t else 0.0

    def as_dict(self) -> dict:
        return {"direction": self.direction,
                "obstacle": self.obstacle.as_dict(),
                "flow": self.flow, "opposing": self.opposing,
                "window_s": round(self.window_s, 2),
                "rate": round(self.rate, 2),
                "ratio": round(self.ratio, 4),
                "seconds_to_clear": (
                    None if self.seconds_to_clear is None
                    else round(self.seconds_to_clear, 2)),
                "threshold": self.threshold,
                "verdict": self.verdict, "passes": self.passes,
                "consumed": self.consumed, "pulled": self.pulled,
                "pulled_share": round(self.pulled_share, 3)}


class Flow:
    """Aggressive contracts per side over a rolling time window."""

    __slots__ = ("window", "_q", "buy", "sell")

    def __init__(self, window_s: float = 10.0) -> None:
        self.window = float(window_s)
        self._q: deque = deque()
        self.buy = 0
        self.sell = 0

    def add(self, ts: float, aggressor: str, size: int) -> None:
        if size <= 0:
            return
        self._q.append((ts, aggressor, size))
        if aggressor == BUY:
            self.buy += size
        else:
            self.sell += size
        self.trim(ts)

    def trim(self, now: float) -> None:
        cut = now - self.window
        q = self._q
        while q and q[0][0] < cut:
            _, agg, sz = q.popleft()
            if agg == BUY:
                self.buy -= sz
            else:
                self.sell -= sz

    @property
    def seconds(self) -> float:
        if len(self._q) < 2:
            return 0.0
        return min(self.window, self._q[-1][0] - self._q[0][0])

    def reset(self) -> None:
        self._q.clear()
        self.buy = self.sell = 0


class Obstacles:
    """Book plus tape: what is standing there, and who is paying."""

    __slots__ = ("tick_size", "reach_ticks", "threshold", "flow",
                 "_levels", "_prev", "_traded", "consumed", "pulled",
                 "_last_ts")

    def __init__(self, tick_size: float = 0.25, reach_ticks: int = 16,
                 window_s: float = 10.0, threshold: float = 1.2) -> None:
        if tick_size <= 0:
            raise ValueError("tick size must be positive")
        self.tick_size = float(tick_size)
        self.reach_ticks = max(1, int(reach_ticks))
        self.threshold = float(threshold)
        self.flow = Flow(window_s)
        self._levels: tuple = ()
        self._prev: dict = {}          # px(ticks) -> (bid_sz, ask_sz)
        self._traded: dict = {}        # px(ticks) -> contracts since
        self.consumed = {BUY: 0, SELL: 0}
        self.pulled = {BUY: 0, SELL: 0}
        self._last_ts = 0.0

    # ---------------------------------------------------------- feed

    def observe_trade(self, ts: float, aggressor: str,
                      size: int, price: float = 0.0) -> None:
        self.flow.add(ts, aggressor, size)
        self._last_ts = max(self._last_ts, ts)
        if price:
            k = int(round(price / self.tick_size))
            self._traded[k] = self._traded.get(k, 0) + int(size)

    def observe_book(self, rec) -> bool:
        raw = dbn._get(rec, "levels")
        if not raw:
            return False
        if not isinstance(raw, (list, tuple)):
            raw = [raw]
        ts = dbn.seconds(dbn._get(rec, "ts_event")) or 0.0
        self._last_ts = max(self._last_ts, ts)

        rows = []
        cur: dict = {}
        for lv in raw:
            bpx = dbn.price(dbn._get(lv, "bid_px", 0))
            apx = dbn.price(dbn._get(lv, "ask_px", 0))
            bsz = int(dbn._get(lv, "bid_sz", 0) or 0)
            asz = int(dbn._get(lv, "ask_sz", 0) or 0)
            bct = int(dbn._get(lv, "bid_ct", 0) or 0)
            act = int(dbn._get(lv, "ask_ct", 0) or 0)
            rows.append((bpx, bsz, bct, apx, asz, act))
            if bpx > 0:
                k = int(round(bpx / self.tick_size))
                cur[k] = (bsz, cur.get(k, (0, 0))[1])
            if apx > 0:
                k = int(round(apx / self.tick_size))
                cur[k] = (cur.get(k, (0, 0))[0], asz)

        self._attribute(cur)
        self._prev = cur
        self._levels = tuple(rows)
        return True

    def _attribute(self, cur: dict) -> None:
        """Split every decline in resting size into traded or withdrawn.

        A level that shrank by more than traded there was partly
        pulled. Without this split, a wall that simply left reads
        exactly like a wall that was eaten -- and those are opposite
        pieces of evidence about whether anyone can get through.
        """
        if not self._prev:
            self._traded = {}
            return
        for k, (pb, pa) in self._prev.items():
            cb, ca = cur.get(k, (0, 0))
            hit = self._traded.get(k, 0)
            for before, after, who in ((pa, ca, BUY), (pb, cb, SELL)):
                drop = before - after
                if drop <= 0:
                    continue
                eaten = min(drop, hit)
                self.consumed[who] += eaten
                self.pulled[who] += drop - eaten
                hit -= eaten
        self._traded = {}

    def reset(self) -> None:
        """New candle. The standing book is kept; the counters are not."""
        self.flow.reset()
        self.consumed = {BUY: 0, SELL: 0}
        self.pulled = {BUY: 0, SELL: 0}
        self._traded = {}

    # ---------------------------------------------------------- read

    def obstacle(self, direction: str = BUY) -> Obstacle:
        """Visible resting size in the way, and how far we could see.

        The reach is measured from the ACTUAL quoted prices, not from
        an assumed one-tick spacing: a gappy book covers more ground
        with ten levels than a dense one, and assuming otherwise
        would misreport both.
        """
        rows = self._levels
        side = "ask" if direction == BUY else "bid"
        if not rows:
            return Obstacle(side=side, asked_ticks=self.reach_ticks)

        if direction == BUY:
            prices = [(r[3], r[4], r[5]) for r in rows if r[3] > 0]
        else:
            prices = [(r[0], r[1], r[2]) for r in rows if r[0] > 0]
        if not prices:
            return Obstacle(side=side, asked_ticks=self.reach_ticks)

        anchor = prices[0][0]
        size = orders = n = 0
        covered = 0
        far = anchor
        for px, sz, ct in prices:
            d = abs(px - anchor) / self.tick_size
            if d > self.reach_ticks:
                break
            size += sz
            orders += ct
            n += 1
            covered = int(round(d))
            far = px

        # The book ran out before the reach did: every level was used
        # and the last one is still inside the distance asked for.
        truncated = (n == len(prices) and covered < self.reach_ticks)
        return Obstacle(side=side, asked_ticks=self.reach_ticks,
                        covered_ticks=covered, truncated=truncated,
                        size=size, orders=orders, levels=n,
                        nearest_px=anchor, farthest_px=far)

    def clearance(self, direction: str = BUY,
                  threshold: float | None = None) -> Clearance:
        thr = self.threshold if threshold is None else float(threshold)
        self.flow.trim(self._last_ts)
        ob = self.obstacle(direction)
        flow = self.flow.buy if direction == BUY else self.flow.sell
        other = self.flow.sell if direction == BUY else self.flow.buy
        secs = self.flow.seconds
        rate = (flow / secs) if secs > 0 else 0.0

        if ob.size <= 0:
            # No visible resistance is not the same as no resistance:
            # hidden size shows at no depth. Say unknown, not clear.
            ratio, ttc, verdict = 0.0, None, UNKNOWN
        else:
            ratio = flow / ob.size
            ttc = (ob.size / rate) if rate > 0 else None
            verdict = CLEARS if ratio >= thr else BLOCKED

        return Clearance(
            direction=direction, obstacle=ob, flow=flow,
            opposing=other, window_s=secs, rate=rate, ratio=ratio,
            seconds_to_clear=ttc, threshold=thr, verdict=verdict,
            consumed=self.consumed[direction],
            pulled=self.pulled[direction])
