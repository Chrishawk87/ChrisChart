"""The live footprint: one volatile bar, a locked history behind it.

    from liqmap.footprint_live import FootprintStream
    from liqmap.footprint import Tick, VolumeRule

    s = FootprintStream(tick_size=0.25, rule=VolumeRule(2000))
    s.on_tick(Tick(ts=..., price=5800.25, size=4, aggressor="buy"))

    p = s.pulse()                     # lock-free, every field one instant
    print(p.delta, p.pressure, p.unconfirmed)

    view = s.view()                   # the whole matrix, on demand

TWO STATES, DELIBERATELY SEPARATE

    history   bars that are closed and FROZEN. Nothing writes to them
              again, so a reader needs no coordination at all.

    live      one bar being mutated by the feed. Everything about
              reading it safely is below.

WHY THE DIAGONAL IS NOT RECOMPUTED FROM SCRATCH

A tick at price P can only change four tests:

    buy at P        (P   vs P-1)
    sell at P       (P   vs P+1)
    buy at P+1      (P+1 vs P  )
    sell at P-1     (P-1 vs P  )

Nothing else in the bar moved, so nothing else can change. That is
O(1) a tick instead of O(levels), and on a bar holding fifty price
levels during a burst it is the difference between keeping up and
falling behind.

THE EDGE POLICY IS WHAT MAKES THAT HARD

The bar's highest and lowest levels are skipped, because their
diagonal lies outside the bar. So when price sets a NEW low, the old
low stops being an edge and its buy test becomes valid for the first
time -- and if price gapped three ticks down, that old low is outside
the four tests above. A new extreme therefore re-tests the previous
one explicitly. Without it, a level silently keeps a verdict that was
computed under a rule that no longer applies to it.

READING IT WITHOUT A RACE

One writer (the feed) and any number of readers (a panel, a poll, a
logger). Two different reads, with different costs:

    pulse()   an immutable snapshot of the scalars, rebuilt after
              every tick and published by rebinding ONE attribute.
              Readers take no lock. This is the microsecond path.

    view()    the whole matrix. Built under the lock, on demand,
              because it allocates per level.

Rebinding one attribute is atomic under the GIL, so a reader either
sees the whole previous snapshot or the whole new one, never half of
each. READING TWO COUNTERS SEPARATELY IS NOT SAFE even though each
read is atomic: `delta = s.ask_volume - s.bid_volume` can take its
two halves from either side of a tick and report a number that was
never true. Every field in a Pulse comes from one instant.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

from .footprint import (BUY_IMBALANCE, SELL_IMBALANCE, CloseRule,
                        DEFAULT_IMBALANCE, FootprintCandle,
                        ImbalanceConfig, Tick, TickScale, TimeRule)

UNCONFIRMED = "unconfirmed"
CONFIRMED = "confirmed"
FADED = "faded"


@dataclass(slots=True)
class LiveFlag:
    """An imbalance, and its life story inside the bar.

    `state` is the part the spec asks for. The rest is what makes a
    live flag readable: a level that has held its imbalance for the
    whole bar and one that flickered on with the last print are not
    the same observation, and a structure that keeps only the current
    verdict cannot tell them apart.
    """

    price_ticks: int
    side: str
    volume: int = 0
    against: int = 0
    ratio: float | None = None
    state: str = UNCONFIRMED
    first_seen_ts: float = 0.0
    last_seen_ts: float = 0.0
    updates: int = 0            # times it was re-measured while live
    peak_ratio: float | None = None

    @property
    def held_s(self) -> float:
        return max(0.0, self.last_seen_ts - self.first_seen_ts)

    def as_dict(self, scale: TickScale) -> dict:
        return {
            "price": scale.to_price(self.price_ticks),
            "price_ticks": self.price_ticks,
            "side": self.side,
            "state": self.state,
            "volume": self.volume,
            "against": self.against,
            "ratio": self.ratio,
            "peak_ratio": self.peak_ratio,
            "held_s": round(self.held_s, 3),
            "updates": self.updates,
        }


@dataclass(frozen=True, slots=True)
class Pulse:
    """Every scalar a caller might poll, from ONE instant.

    Frozen and published by rebinding a single attribute, so a reader
    never assembles a number out of two different moments.
    """

    ts: float = 0.0
    seq: int = 0                 # ticks folded in when this was taken
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    last: float = 0.0
    total_volume: int = 0
    ask_volume: int = 0          # aggressive buyers
    bid_volume: int = 0          # aggressive sellers
    delta: int = 0
    window_delta: int = 0        # delta over the recent window only
    window_volume: int = 0       # and the volume it was measured over
    trades: int = 0
    unconfirmed: int = 0
    buy_flags: int = 0
    sell_flags: int = 0
    bars_closed: int = 0
    bar_age_s: float = 0.0

    @property
    def ratio(self) -> float | None:
        """Aggressive buyers to aggressive sellers.

        None when one side has not traded: an infinite ratio is not a
        number, and returning a large float instead invites a
        threshold comparison that silently always passes.
        """
        if self.bid_volume <= 0 or self.ask_volume <= 0:
            return None
        return self.ask_volume / self.bid_volume

    @property
    def pressure(self) -> float:
        """Delta normalised to -1..+1.

        Comparable across bars of any size, which raw delta is not:
        +400 means something different in a 500-lot bar and a
        50,000-lot one.
        """
        tot = self.total_volume
        return (self.delta / tot) if tot else 0.0

    @property
    def window_pressure(self) -> float:
        """The same, over the recent window -- the momentum half.

        A bar can sit at +0.4 overall while the last thirty seconds
        ran hard the other way. One number cannot say that; two can.
        """
        return ((self.window_delta / self.window_volume)
                if self.window_volume else 0.0)

    def as_dict(self) -> dict:
        return {
            "ts": self.ts, "seq": self.seq,
            "open": self.open, "high": self.high,
            "low": self.low, "last": self.last,
            "total_volume": self.total_volume,
            "ask_volume": self.ask_volume,
            "bid_volume": self.bid_volume,
            "delta": self.delta,
            "window_delta": self.window_delta,
            "window_volume": self.window_volume,
            "ratio": self.ratio,
            "pressure": round(self.pressure, 4),
            "window_pressure": round(self.window_pressure, 4),
            "trades": self.trades,
            "unconfirmed": self.unconfirmed,
            "buy_flags": self.buy_flags,
            "sell_flags": self.sell_flags,
            "bars_closed": self.bars_closed,
            "bar_age_s": round(self.bar_age_s, 3),
        }



class Tracker:
    """Keeps a bar's imbalance flags current, incrementally.

    Holds no reference to the candle: it is handed the levels it needs
    on each call, so the same tracker can be reused across bars
    without carrying state from the previous one by accident.
    """

    __slots__ = ("cfg", "flags", "faded")

    def __init__(self, cfg: ImbalanceConfig = DEFAULT_IMBALANCE) -> None:
        self.cfg = cfg
        self.flags: dict = {}       # (price, side) -> LiveFlag
        self.faded: list = []       # flags that were live and are not

    def reset(self) -> None:
        self.flags = {}
        self.faded = []

    # -------------------------------------------------------- the test

    def _measure(self, candle: FootprintCandle, p: int, side: str):
        """(volume, against, passes) for one level and one side.

        Returns None when the level cannot be judged at all -- it does
        not exist, or it is an edge whose diagonal lies outside the
        bar.
        """
        lv = candle.levels.get(p)
        if lv is None:
            return None
        cfg = self.cfg

        if side == BUY_IMBALANCE:
            vol = lv.ask_volume
            neighbour = candle.levels.get(p - 1)
            is_edge = (p == candle.low_ticks and neighbour is None)
            ref = neighbour.bid_volume if neighbour is not None else 0
        else:
            vol = lv.bid_volume
            neighbour = candle.levels.get(p + 1)
            is_edge = (p == candle.high_ticks and neighbour is None)
            ref = neighbour.ask_volume if neighbour is not None else 0

        if is_edge and cfg.edge_policy == "skip":
            return None
        if vol <= 0 or vol < cfg.min_volume:
            return (vol, ref, False)
        if ref > 0:
            return (vol, ref, vol >= cfg.multiplier * ref)
        return (vol, ref, cfg.flag_empty_diagonal)

    def _apply(self, candle: FootprintCandle, p: int, side: str,
               ts: float) -> None:
        got = self._measure(candle, p, side)
        key = (p, side)
        existing = self.flags.get(key)

        if got is None or not got[2]:
            if existing is not None:
                # It was an imbalance and is not any more. Kept, not
                # discarded: a level that flashed and faded is a thing
                # that happened, and a structure holding only the
                # current verdict cannot report it.
                existing.state = FADED
                existing.last_seen_ts = ts
                self.faded.append(existing)
                del self.flags[key]
            return

        vol, ref, _ = got
        ratio = (vol / ref) if ref else None
        if existing is None:
            self.flags[key] = LiveFlag(
                price_ticks=p, side=side, volume=vol, against=ref,
                ratio=ratio, state=UNCONFIRMED,
                first_seen_ts=ts, last_seen_ts=ts, updates=1,
                peak_ratio=ratio)
        else:
            existing.volume = vol
            existing.against = ref
            existing.ratio = ratio
            existing.last_seen_ts = ts
            existing.updates += 1
            if ratio is None or existing.peak_ratio is None:
                existing.peak_ratio = None      # once infinite, infinite
            elif ratio > existing.peak_ratio:
                existing.peak_ratio = ratio

    # ------------------------------------------------------ the update

    def on_trade(self, candle: FootprintCandle, p: int, ts: float,
                 old_low: int | None, old_high: int | None) -> None:
        """Re-test only what this print can have changed.

        Four tests, because a change at P moves exactly four
        comparisons -- plus the old extremes when the bar's range
        grew, since the edge policy's verdict on them has just
        changed.
        """
        self._apply(candle, p, BUY_IMBALANCE, ts)
        self._apply(candle, p, SELL_IMBALANCE, ts)
        self._apply(candle, p + 1, BUY_IMBALANCE, ts)
        self._apply(candle, p - 1, SELL_IMBALANCE, ts)

        # A new low makes the PREVIOUS low an interior level, whose
        # buy test was being skipped as an edge and is now real. Price
        # can gap several ticks, putting that level outside the four
        # tests above.
        if old_low is not None and candle.low_ticks < old_low:
            self._apply(candle, old_low, BUY_IMBALANCE, ts)
        if old_high is not None and candle.high_ticks > old_high:
            self._apply(candle, old_high, SELL_IMBALANCE, ts)

    def confirm(self, ts: float) -> list:
        """Freeze what is still standing. The spec's transition."""
        out = []
        for f in self.flags.values():
            f.state = CONFIRMED
            f.last_seen_ts = ts
            out.append(f)
        out.sort(key=lambda f: (f.price_ticks, f.side))
        return out


class Window:
    """Delta over the last `seconds`, for the momentum half.

    A deque of (ts, signed_size) trimmed from the front. Bounded by
    time rather than count, because a burst and a quiet minute should
    be the same window of market, not the same number of prints.
    """

    __slots__ = ("seconds", "_q", "total")

    def __init__(self, seconds: float = 30.0) -> None:
        self.seconds = float(seconds)
        self._q: deque = deque()
        self.total = 0

    def push(self, ts: float, signed: int) -> int:
        self._q.append((ts, signed))
        self.total += signed
        cutoff = ts - self.seconds
        q = self._q
        while q and q[0][0] < cutoff:
            self.total -= q.popleft()[1]
        return self.total

    def volume(self) -> int:
        return sum(abs(v) for _, v in self._q)

    def reset(self) -> None:
        self._q.clear()
        self.total = 0


@dataclass
class LiveView:
    """The whole live bar, built on demand. Safe to hand anywhere:
    every field is a copy, so the feed cannot mutate it afterwards."""

    pulse: Pulse
    rows: list = field(default_factory=list)   # high to low
    flags: list = field(default_factory=list)
    faded: list = field(default_factory=list)
    close_reason: str = ""

    def as_dict(self) -> dict:
        return {
            "pulse": self.pulse.as_dict(),
            "rows": self.rows,
            "flags": self.flags,
            "faded": self.faded,
        }


class FootprintStream:
    """Tick ingestion, one live bar, a frozen history behind it.

    ONE WRITER. `on_tick` takes the lock, so several feed threads are
    safe, but the design assumes one -- two feeds writing one bar
    would interleave prints from different sources into a single
    matrix, which no lock can make meaningful.
    """

    def __init__(self, tick_size: float = 0.25,
                 rule: CloseRule | None = None,
                 imbalance: ImbalanceConfig = DEFAULT_IMBALANCE,
                 window_s: float = 30.0, keep: int = 500) -> None:
        from .footprint import FootprintEngine
        self.scale = TickScale(tick_size)
        self.rule = rule or TimeRule(300.0)
        self.cfg = imbalance
        self.keep = int(keep)

        self._lock = threading.Lock()
        self._engine = FootprintEngine(
            tick_size=tick_size, rule=self.rule, imbalance=imbalance,
            keep=keep)
        self.tracker = Tracker(imbalance)
        self.window = Window(window_s)

        self.history: list = []          # closed, frozen
        self.confirmed: dict = {}        # bar index -> confirmed flags
        self._pulse = Pulse()            # published by rebind
        self.seq = 0

    # ------------------------------------------------------------ feed

    def on_tick(self, tick: Tick) -> list:
        """Fold one print in. Returns any bars it closed.

        The imbalance set is updated INSIDE the lock, with the matrix,
        so the published pulse can never describe a flag count from a
        different instant than its volumes.
        """
        with self._lock:
            px = self.scale.to_ticks(tick.price)
            cur = self._engine.current
            old_low = cur.low_ticks if (cur and not cur.empty) else None
            old_high = cur.high_ticks if (cur and not cur.empty) else None

            closed = self._engine.on_tick(tick)

            if closed:
                # Whatever is standing at the close is confirmed, and
                # the bar is frozen behind it.
                for bar in closed:
                    self._freeze(bar, tick.ts)
                # The split remainder (if any) already opened a new
                # bar, so the tracker starts clean and is brought up
                # to date with what that bar holds so far.
                self.tracker.reset()
                self.window.reset()
                nxt = self._engine.current
                if nxt is not None and not nxt.empty:
                    self.tracker.on_trade(nxt, px, tick.ts, None, None)
            else:
                self.tracker.on_trade(self._engine.current, px, tick.ts,
                                      old_low, old_high)

            signed = tick.size if tick.aggressor == "buy" else -tick.size
            self.window.push(tick.ts, signed)
            self.seq += 1
            self._publish(tick.ts)
            return closed

    def roll(self, now: float) -> list:
        """Close a time bar that expired with no prints."""
        with self._lock:
            closed = self._engine.roll(now)
            for bar in closed:
                self._freeze(bar, now)
            if closed:
                self.tracker.reset()
                self.window.reset()
                self._publish(now)
            return closed

    def _freeze(self, bar: FootprintCandle, ts: float) -> None:
        bar.closed = True
        self.confirmed[len(self.history)] = self.tracker.confirm(ts)
        self.history.append(bar)
        if len(self.history) > self.keep:
            drop = len(self.history) - self.keep
            del self.history[:drop]
            self.confirmed = {k - drop: v
                              for k, v in self.confirmed.items()
                              if k >= drop}

    # ------------------------------------------------------------ read

    def _publish(self, ts: float) -> None:
        """Build the immutable pulse and rebind it. Called under the
        lock; the rebind is what readers rely on."""
        c = self._engine.current
        sc = self.scale
        flags = self.tracker.flags
        buys = sum(1 for k in flags if k[1] == BUY_IMBALANCE)
        p = Pulse(
            ts=ts, seq=self.seq,
            open=sc.to_price(c.open_ticks) if c and not c.empty else 0.0,
            high=sc.to_price(c.high_ticks) if c and not c.empty else 0.0,
            low=sc.to_price(c.low_ticks) if c and not c.empty else 0.0,
            last=sc.to_price(c.close_ticks) if c and not c.empty else 0.0,
            total_volume=c.total_volume if c else 0,
            ask_volume=c.ask_volume if c else 0,
            bid_volume=c.bid_volume if c else 0,
            delta=c.delta if c else 0,
            window_delta=self.window.total,
            window_volume=self.window.volume(),
            trades=c.trades if c else 0,
            unconfirmed=len(flags),
            buy_flags=buys, sell_flags=len(flags) - buys,
            bars_closed=len(self.history),
            bar_age_s=(ts - c.ts_open) if c else 0.0)
        self._pulse = p            # atomic rebind; readers take no lock

    def pulse(self) -> Pulse:
        """The scalars, lock-free, all from one instant."""
        return self._pulse

    def view(self, include_faded: bool = True) -> LiveView:
        """The whole live bar. Allocates per level -- poll it at a
        frame rate, not on every tick."""
        with self._lock:
            c = self._engine.current
            sc = self.scale
            rows = []
            flagged = {}
            for (p, side), f in self.tracker.flags.items():
                flagged.setdefault(p, []).append(side)
            if c is not None:
                for price, lv in c.rows():
                    rows.append({
                        "price": sc.to_price(price),
                        "price_ticks": price,
                        "bid_volume": lv.bid_volume,
                        "ask_volume": lv.ask_volume,
                        "delta": lv.delta,
                        "total": lv.total,
                        "imbalance": flagged.get(price, []),
                    })
            return LiveView(
                pulse=self._pulse,
                rows=rows,
                flags=[f.as_dict(sc) for f in sorted(
                    self.tracker.flags.values(),
                    key=lambda x: (x.price_ticks, x.side))],
                faded=([f.as_dict(sc) for f in self.tracker.faded]
                       if include_faded else []),
                close_reason=c.close_reason if c else "")

    def confirmed_flags(self, index: int = -1) -> list:
        """The confirmed set of a closed bar. Frozen -- no lock."""
        if not self.history:
            return []
        i = index if index >= 0 else len(self.history) + index
        return self.confirmed.get(i, [])

    def last_closed(self) -> FootprintCandle | None:
        return self.history[-1] if self.history else None
