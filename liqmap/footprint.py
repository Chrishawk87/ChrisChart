"""Footprint candles and diagonal order-flow imbalance.

    from liqmap.footprint import FootprintEngine, TimeRule, Tick

    eng = FootprintEngine(tick_size=0.25, rule=TimeRule(300))
    for closed in eng.on_tick(tick):
        for flag in closed.imbalances():
            print(flag)

WHAT A FOOTPRINT IS COUNTING

Every print has an aggressor: the side that crossed the spread. A
trade that happens AT THE BID was taken by an aggressive seller; one
AT THE ASK was taken by an aggressive buyer. Resting orders are
passive by definition, so the aggressor is the only directional
information a print carries.

This module follows that convention exactly, and the naming is the
single most dangerous thing in it:

    PriceLevel.bid_volume   contracts traded AT THE BID -> sellers
    PriceLevel.ask_volume   contracts traded AT THE ASK -> buyers

Some feeds label the opposite way round. Getting it backwards does
not raise anything -- it silently inverts every imbalance in the
system, and the output still looks entirely reasonable. `Tick` takes
an explicit `aggressor` for that reason, and `flip_aggressor` exists
so a feed with the opposite convention is corrected in ONE place
rather than by editing this logic.

PRICES ARE INTEGER TICK INDICES, NOT FLOATS

A level is `round(price / tick_size)`, held as an int. Floats make
poor dictionary keys -- 5800.25 reached by two different arithmetic
paths can be two different keys -- and the diagonal comparison is
"one tick up or down", which on an integer index is simply +/- 1.
`TickScale` converts back for display and nothing else.

THE DIAGONAL

A footprint imbalance compares a price level against the OPPOSITE
side one tick away, not against itself. Buyers lifting the ask at P
are compared with sellers hitting the bid at P-1, because those two
orders could have traded with each other and did not.

    buying  imbalance at P:  Ask(P)  >= multiplier * Bid(P - 1)
    selling imbalance at P:  Bid(P)  >= multiplier * Ask(P + 1)

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER. It counts contracts
and tags levels.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Literal

Aggressor = Literal["buy", "sell"]

BUY_IMBALANCE = "buy"
SELL_IMBALANCE = "sell"


# --------------------------------------------------------------- input

@dataclass(frozen=True, slots=True)
class Tick:
    """One print.

    `price` is the raw price; the engine converts it to a tick index
    once, on the way in. `size` is in contracts and must be positive:
    a zero-size print is not an observation about anything, and
    letting one through puts empty levels in the matrix that the
    imbalance loop then has to special-case.
    """

    ts: float
    price: float
    size: int
    aggressor: Aggressor

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"size must be positive, got {self.size}")
        if self.aggressor not in ("buy", "sell"):
            raise ValueError(
                f"aggressor must be 'buy' or 'sell', got "
                f"{self.aggressor!r}. 'buy' means an aggressive buyer "
                f"lifted the ask; 'sell' means an aggressive seller hit "
                f"the bid.")


def flip_aggressor(tick: Tick) -> Tick:
    """The same print under the opposite convention.

    Exists so a feed that labels its aggressor the other way round is
    corrected HERE, once, rather than by reversing the comparison in
    the imbalance loop -- an edit someone later "fixes" back, silently
    inverting every flag in the system.
    """
    other: Aggressor = "sell" if tick.aggressor == "buy" else "buy"
    return Tick(ts=tick.ts, price=tick.price, size=tick.size,
                aggressor=other)


class TickScale:
    """Raw price <-> integer tick index.

    Rounding is done once, on ingestion. A price that is not on the
    grid is snapped rather than rejected: feeds carry the occasional
    off-tick print (block trades, spread legs) and dropping them would
    silently lose volume from the candle.
    """

    __slots__ = ("size",)

    def __init__(self, size: float = 0.25) -> None:
        if size <= 0:
            raise ValueError("tick size must be positive")
        self.size = float(size)

    def to_ticks(self, price: float) -> int:
        return int(round(float(price) / self.size))

    def to_price(self, ticks: int) -> float:
        return round(ticks * self.size, 10)


# ---------------------------------------------------------- the matrix

@dataclass(slots=True)
class PriceLevel:
    """One row of the footprint: both sides at a single price.

    `bid_volume` is contracts taken by aggressive SELLERS (traded at
    the bid). `ask_volume` is contracts taken by aggressive BUYERS
    (traded at the ask). See the module docstring -- these two being
    the wrong way round is undetectable downstream.
    """

    bid_volume: int = 0
    ask_volume: int = 0

    @property
    def total(self) -> int:
        return self.bid_volume + self.ask_volume

    @property
    def delta(self) -> int:
        """Aggressive buyers minus aggressive sellers at this price."""
        return self.ask_volume - self.bid_volume

    def add(self, size: int, aggressor: Aggressor) -> None:
        if aggressor == "buy":
            self.ask_volume += size
        else:
            self.bid_volume += size


@dataclass(frozen=True, slots=True)
class ImbalanceFlag:
    """One tagged level."""

    price_ticks: int
    side: str                   # BUY_IMBALANCE / SELL_IMBALANCE
    volume: int                 # the aggressive side that won
    against: int                # the diagonal it was measured against
    ratio: float | None         # None when the diagonal was empty

    def describe(self, scale: TickScale) -> str:
        r = "inf" if self.ratio is None else f"{self.ratio:.1f}x"
        tag = "BUY " if self.side == BUY_IMBALANCE else "SELL"
        return (f"{tag} imbalance at {scale.to_price(self.price_ticks):>10.2f}"
                f"  {self.volume:>6,} vs {self.against:>6,}  ({r})")


@dataclass(frozen=True, slots=True)
class ImbalanceConfig:
    """How strict the diagonal test is.

    `multiplier` is the standard footprint threshold -- 4.0 means
    400%.

    `min_volume` is the floor the winning side must clear on its own.
    This is the setting that matters, and the reason is the zero case:
    the test is `ask >= multiplier * bid`, so when the diagonal is
    EMPTY the right-hand side is zero and any volume at all passes.
    Without a floor, a single lot against an untraded level is tagged
    the same as four hundred against a hundred. A floor of zero
    reproduces the bare formula; anything above it asks for the flag
    to be worth looking at.

    `flag_empty_diagonal` decides whether an empty diagonal counts at
    all. Left on, those are the strongest imbalances by construction.
    Turned off, only levels with a real quantity on both sides are
    compared.

    `edge_policy` handles the top and bottom of the candle, whose
    diagonals fall OUTSIDE it. 'skip' refuses to judge them -- the
    missing neighbour is an artifact of where the bar happened to
    close, not a statement that nothing traded there. 'zero' treats it
    as an untraded level, which reliably tags both extremes of every
    candle and is almost never what you want.
    """

    multiplier: float = 4.0
    min_volume: int = 0
    flag_empty_diagonal: bool = True
    edge_policy: Literal["skip", "zero"] = "skip"

    def __post_init__(self) -> None:
        if self.multiplier <= 0:
            raise ValueError("multiplier must be positive")
        if self.min_volume < 0:
            raise ValueError("min_volume cannot be negative")
        if self.edge_policy not in ("skip", "zero"):
            raise ValueError("edge_policy must be 'skip' or 'zero'")


DEFAULT_IMBALANCE = ImbalanceConfig()


# ------------------------------------------------------------- candles

@dataclass(slots=True)
class FootprintCandle:
    """One bar, with both sides of every price it traded at."""

    ts_open: float
    ts_close: float = 0.0
    open_ticks: int = 0
    high_ticks: int = 0
    low_ticks: int = 0
    close_ticks: int = 0
    total_volume: int = 0
    levels: dict = field(default_factory=dict)
    trades: int = 0
    closed: bool = False
    # Set by the rule that ended it, so a bar carries the reason it
    # exists -- useful the moment several rules are composed.
    close_reason: str = ""
    bucket: float = 0.0          # grid slot, for time-based bars

    # ---------------------------------------------------------- build

    def add(self, price_ticks: int, size: int, aggressor: Aggressor,
            ts: float) -> None:
        """Fold one print, or part of one, into this bar."""
        if size <= 0:
            return
        first = self.total_volume == 0
        if first:
            self.open_ticks = price_ticks
            self.high_ticks = price_ticks
            self.low_ticks = price_ticks
        else:
            if price_ticks > self.high_ticks:
                self.high_ticks = price_ticks
            if price_ticks < self.low_ticks:
                self.low_ticks = price_ticks

        lv = self.levels.get(price_ticks)
        if lv is None:
            lv = PriceLevel()
            self.levels[price_ticks] = lv
        lv.add(size, aggressor)

        self.close_ticks = price_ticks
        self.total_volume += size
        self.trades += 1
        self.ts_close = ts

    # ----------------------------------------------------------- read

    @property
    def empty(self) -> bool:
        return self.total_volume == 0

    @property
    def delta(self) -> int:
        """Net aggressive volume: buyers minus sellers, whole bar."""
        return sum(lv.delta for lv in self.levels.values())

    @property
    def bid_volume(self) -> int:
        return sum(lv.bid_volume for lv in self.levels.values())

    @property
    def ask_volume(self) -> int:
        return sum(lv.ask_volume for lv in self.levels.values())

    def level(self, price_ticks: int) -> PriceLevel | None:
        return self.levels.get(price_ticks)

    def rows(self) -> list:
        """Every traded level, HIGH to LOW -- the way a footprint is
        read on a screen."""
        return [(p, self.levels[p]) for p in sorted(self.levels,
                                                    reverse=True)]

    def point_of_control(self) -> int | None:
        """The price with the most volume. Ties go to the one nearest
        the close, because a tie broken by sort order is arbitrary and
        would jitter between bars."""
        if not self.levels:
            return None
        best, best_vol = None, -1
        for p, lv in self.levels.items():
            v = lv.total
            if v > best_vol or (v == best_vol and best is not None
                                and abs(p - self.close_ticks)
                                < abs(best - self.close_ticks)):
                best, best_vol = p, v
        return best

    # ------------------------------------------------------ imbalance

    def imbalances(self, cfg: ImbalanceConfig = DEFAULT_IMBALANCE
                   ) -> list:
        """Diagonal imbalances across this bar's price levels.

        THE COMPARISON IS A MULTIPLICATION, never a division. There is
        no reciprocal to compute and therefore no zero to guard, and
        with integer volumes the test is exact -- no level sits on the
        threshold because of a rounding error.

        GAPS ARE SAFE BECAUSE THE LOOKUP IS BY KEY. The diagonal is
        `levels[p - 1]`, an absolute price, not "the previous row in
        the iteration". An untraded price inside the bar's range is
        simply absent from the dict, so a level above a gap is
        compared against nothing -- never against whatever happened to
        be two ticks away. Iterating the traded levels is therefore
        equivalent to walking every tick between low and high, and on
        a bar that spans a hundred ticks and trades twenty it is five
        times less work.
        """
        if not self.levels:
            return []

        lo, hi = self.low_ticks, self.high_ticks
        out: list = []
        mult = cfg.multiplier
        floor = cfg.min_volume

        # Low to high, so the returned list has a stable order a
        # caller can rely on. The RESULT does not depend on it -- the
        # diagonal is a key lookup -- but a list whose order changes
        # between versions quietly reshuffles anything downstream that
        # takes "the first flag".
        for p in sorted(self.levels):
            lv = self.levels[p]

            # --- buying: aggressive buyers at P vs sellers at P-1
            if lv.ask_volume > 0 and lv.ask_volume >= floor:
                below = self.levels.get(p - 1)
                if p == lo and below is None and cfg.edge_policy == "skip":
                    pass        # its diagonal is outside this bar
                else:
                    ref = below.bid_volume if below is not None else 0
                    ok = (lv.ask_volume >= mult * ref
                          if ref > 0 else cfg.flag_empty_diagonal)
                    if ok:
                        out.append(ImbalanceFlag(
                            price_ticks=p, side=BUY_IMBALANCE,
                            volume=lv.ask_volume, against=ref,
                            ratio=(lv.ask_volume / ref) if ref else None))

            # --- selling: aggressive sellers at P vs buyers at P+1
            if lv.bid_volume > 0 and lv.bid_volume >= floor:
                above = self.levels.get(p + 1)
                if p == hi and above is None and cfg.edge_policy == "skip":
                    pass
                else:
                    ref = above.ask_volume if above is not None else 0
                    ok = (lv.bid_volume >= mult * ref
                          if ref > 0 else cfg.flag_empty_diagonal)
                    if ok:
                        out.append(ImbalanceFlag(
                            price_ticks=p, side=SELL_IMBALANCE,
                            volume=lv.bid_volume, against=ref,
                            ratio=(lv.bid_volume / ref) if ref else None))
        return out

    def stacked(self, cfg: ImbalanceConfig = DEFAULT_IMBALANCE,
                run: int = 3) -> list:
        """Runs of `run` or more consecutive imbalances on one side.

        A single tagged level is noise at any threshold. Consecutive
        ones are the pattern people actually trade off, so the run
        length is a parameter rather than a constant.
        """
        flags = self.imbalances(cfg)
        by_side: dict = {}
        for f in flags:
            by_side.setdefault(f.side, []).append(f.price_ticks)

        out: list = []
        for side, prices in by_side.items():
            prices.sort()
            start = prev = None
            for p in prices:
                if prev is not None and p == prev + 1:
                    prev = p
                    continue
                if start is not None and prev - start + 1 >= run:
                    out.append((side, start, prev))
                start = prev = p
            if start is not None and prev - start + 1 >= run:
                out.append((side, start, prev))
        return out


# --------------------------------------------------------- close rules

class CloseRule(ABC):
    """When does a bar end?

    Three questions, kept apart so a new rule implements only what it
    cares about:

      `starts_new`  this print belongs to a different bar entirely
                    (a time grid crossing)
      `room`        how much of this print fits in the current bar
                    (a volume cap that must land EXACTLY)
      `done`        the bar has met its condition and should close

    A rule that only has one of the three overrides one method.
    """

    name = "rule"

    def starts_new(self, candle: FootprintCandle, tick: Tick,
                   ticks: int) -> bool:
        return False

    def room(self, candle: FootprintCandle, tick: Tick) -> int:
        return tick.size

    def done(self, candle: FootprintCandle) -> bool:
        return False

    def bucket_for(self, ts: float) -> float:
        """The grid slot this timestamp belongs to. 0.0 when the rule
        has no notion of one."""
        return 0.0

    def expired(self, candle: FootprintCandle, now: float) -> bool:
        """True when the bar should close WITHOUT a print arriving.

        Only a time rule can answer yes: a volume or delta bar has by
        definition not met its condition until more trades happen, and
        closing one on a timer would make its threshold a lie.
        """
        return False

    def reset(self) -> None:
        return None


class TimeRule(CloseRule):
    """Fixed intervals on a grid.

    Aligned to the epoch rather than to the first print, so a 5-minute
    bar starts at :00, :05, :10 on every run and on every restart.
    Anchored to the first tick instead, two services watching the same
    market would disagree about where every bar begins.
    """

    name = "time"

    def __init__(self, seconds: float = 300.0) -> None:
        if seconds <= 0:
            raise ValueError("interval must be positive")
        self.seconds = float(seconds)

    def bucket_for(self, ts: float) -> float:
        return math.floor(ts / self.seconds) * self.seconds

    def starts_new(self, candle: FootprintCandle, tick: Tick,
                   ticks: int) -> bool:
        return self.bucket_for(tick.ts) != candle.bucket

    def expired(self, candle: FootprintCandle, now: float) -> bool:
        return self.bucket_for(now) > candle.bucket


class VolumeRule(CloseRule):
    """Closes at EXACTLY N contracts.

    A print that would carry the bar past the threshold is SPLIT: the
    part that fits completes this bar and the remainder opens the
    next. Without that, "10,000 contracts" means "10,000 or a bit
    more, depending on how the last print happened to be sized", and
    bars stop being comparable to each other -- which is the whole
    reason for counting volume instead of time.
    """

    name = "volume"

    def __init__(self, contracts: int = 10_000) -> None:
        if contracts <= 0:
            raise ValueError("volume threshold must be positive")
        self.contracts = int(contracts)

    def room(self, candle: FootprintCandle, tick: Tick) -> int:
        return max(0, self.contracts - candle.total_volume)

    def done(self, candle: FootprintCandle) -> bool:
        return candle.total_volume >= self.contracts


class DeltaRule(CloseRule):
    """Closes when net aggressive volume reaches a threshold.

    Deliberately does NOT split the print that crosses the line.
    Delta is a running difference, so the crossing print is the event
    the bar is about; cutting it in half to land on a round number
    would attribute part of one aggressor's order to a bar that
    closed before it happened.
    """

    name = "delta"

    def __init__(self, threshold: int = 1_000) -> None:
        if threshold <= 0:
            raise ValueError("delta threshold must be positive")
        self.threshold = int(threshold)

    def done(self, candle: FootprintCandle) -> bool:
        return abs(candle.delta) >= self.threshold


class AnyRule(CloseRule):
    """Closes on whichever member fires first.

    The common pairing is time AND volume: a bar every five minutes,
    or sooner if ten thousand contracts trade. `close_reason` on the
    finished bar names which one ended it, because a mixed series
    whose bars do not say why they closed cannot be analysed.
    """

    name = "any"

    def __init__(self, *rules: CloseRule) -> None:
        if not rules:
            raise ValueError("AnyRule needs at least one rule")
        self.rules = rules

    def bucket_for(self, ts: float) -> float:
        for r in self.rules:
            b = r.bucket_for(ts)
            if b:
                return b
        return 0.0

    def starts_new(self, candle, tick, ticks) -> bool:
        for r in self.rules:
            if r.starts_new(candle, tick, ticks):
                # Tagged HERE, where the firing rule is known. The
                # engine closes with "any" otherwise, which is the
                # one thing a composed series must not say.
                candle.close_reason = r.name
                return True
        return False

    def room(self, candle, tick) -> int:
        return min(r.room(candle, tick) for r in self.rules)

    def done(self, candle) -> bool:
        for r in self.rules:
            if r.done(candle):
                candle.close_reason = r.name
                return True
        return False

    def expired(self, candle, now) -> bool:
        return any(r.expired(candle, now) for r in self.rules)

    def reset(self) -> None:
        for r in self.rules:
            r.reset()


# -------------------------------------------------------------- engine

class FootprintEngine:
    """Ticks in, finished footprint candles out.

    Streaming and incremental: a print is folded into the open bar as
    it arrives and nothing is recomputed. The imbalance pass runs once,
    on a finished bar, because a diagonal read off a bar still forming
    changes as the bar fills and is not a measurement of anything.
    """

    def __init__(self, tick_size: float = 0.25,
                 rule: CloseRule | None = None,
                 imbalance: ImbalanceConfig = DEFAULT_IMBALANCE,
                 keep: int = 500) -> None:
        self.scale = TickScale(tick_size)
        self.rule = rule or TimeRule(300.0)
        self.imbalance = imbalance
        self.keep = int(keep)
        self.current: FootprintCandle | None = None
        self.history: list = []
        self.ticks_seen = 0
        self.contracts_seen = 0

    # ---------------------------------------------------------- feed

    def on_tick(self, tick: Tick) -> list:
        """One print in. Returns the bars it finished, usually none.

        A print can finish more than one bar: a 25,000-lot against a
        10,000-contract rule completes two and opens a third.
        """
        self.ticks_seen += 1
        self.contracts_seen += tick.size
        px = self.scale.to_ticks(tick.price)
        finished: list = []
        remaining = tick.size

        while remaining > 0:
            if self.current is None:
                self.current = self._open(tick.ts)
            elif self.rule.starts_new(self.current, tick, px):
                finished.append(
                    self._close(self.current.close_reason or self.rule.name))
                self.current = self._open(tick.ts)

            room = self.rule.room(self.current, tick)
            if room <= 0:
                # The bar is full but its rule did not say `done` --
                # close it rather than spin forever on a print that
                # cannot fit anywhere.
                finished.append(self._close(self.rule.name))
                self.current = self._open(tick.ts)
                room = self.rule.room(self.current, tick)
                if room <= 0:
                    raise RuntimeError(
                        f"{type(self.rule).__name__} leaves no room in a "
                        f"fresh bar -- its threshold is smaller than one "
                        f"contract")

            take = min(remaining, room)
            self.current.add(px, take, tick.aggressor, tick.ts)
            remaining -= take

            if self.rule.done(self.current):
                if not self.current.close_reason:
                    self.current.close_reason = self.rule.name
                finished.append(self._close(self.current.close_reason))
        return finished

    def roll(self, now: float) -> list:
        """Close a time bar whose interval elapsed with no prints.

        Without this, a quiet five minutes leaves the previous bar
        open and the next print appears to belong to it.
        """
        if self.current is None or not self.rule.expired(self.current, now):
            return []
        if self.current.empty:
            self.current = None
            return []
        return [self._close("expired")]

    def feed(self, ticks: Iterable) -> Iterator:
        """Drive the engine from any iterable of ticks."""
        for t in ticks:
            for bar in self.on_tick(t):
                yield bar

    # --------------------------------------------------------- state

    def _open(self, ts: float) -> FootprintCandle:
        return FootprintCandle(ts_open=ts, ts_close=ts,
                               bucket=self.rule.bucket_for(ts))

    def _close(self, reason: str) -> FootprintCandle:
        bar = self.current
        bar.closed = True
        bar.close_reason = reason or self.rule.name
        self.history.append(bar)
        if len(self.history) > self.keep:
            del self.history[:len(self.history) - self.keep]
        self.current = None
        return bar

    def flags(self, bar: FootprintCandle) -> list:
        return bar.imbalances(self.imbalance)
