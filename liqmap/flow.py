"""Liquidity and order flow: what is resting, what is being aggressed, and
whether the resting side is winning.

Nothing here touches the network. It takes book snapshots and trades as plain
objects and computes, so every number below is testable against a book you
write by hand. `hl.py` supplies the live feeds.

FOUR READINGS, IN THE ORDER THEY MATTER

1. SLIPPAGE (`Book.walk`). What a market order of your size actually costs
   right now, in both directions. Not an indicator -- a direct input. This is
   the only number here that is exact rather than inferential: it is arithmetic
   on the book in front of you.

2. DEPTH AND SHELVES (`Book.depth`, `Book.shelves`). Where resting size sits.
   A shelf is a price level holding far more than the levels around it, which
   is where a move has to fight rather than glide.

3. FLOW (`FlowTape`). Aggressive buying minus aggressive selling, cumulative.
   The numbers in and out. Divergence is when this and price disagree.

4. ABSORPTION (`LevelWatch`). The one you actually trade. Aggression arriving
   at a level while price refuses to move.

THE HONEST PART, BECAUSE IT DECIDES HOW MUCH TO TRUST ANY OF IT

Resting size is a promise, not a commitment. It can be pulled the instant it
is about to be hit, and frequently is. A book reading is evidence about intent,
never a guarantee of a fill. Two consequences:

  - `Book.walk` tells you what a sweep costs against the book AS DISPLAYED. A
    real sweep moves faster than the book refreshes and will usually cost more.
    Treat it as a floor on your cost.

  - Displayed size that never gets tested proves nothing. Size that gets eaten
    and comes back (`BandTracker.replenished`) proves somebody is willing to
    keep paying to defend the level. Refills are worth more than depth, which
    is why they are tracked separately and weighted more heavily in the verdict.

ABSORPTION IS MEASURED AS IMPACT, NOT AS VOLUME

The naive version -- "lots of volume and price didn't move" -- has no scale. A
million dollars is enormous on one coin at 3am and unremarkable on another at
the open. So absorption here is relative to how far price NORMALLY moves per
dollar of net aggression, learned from the tape itself (`ImpactBaseline`).

    impact_ratio = observed move / move you would have expected

Below 1 means the level is eating more than its share. Around 1 means nothing
is happening that the tape does not already explain. Above 1 means price is
moving MORE easily than usual, which is thin-book behaviour and the opposite of
absorption -- worth knowing, because it is the shape that gaps through a level
you expected to hold.

The baseline needs samples before it means anything. Until it has them,
`Absorption.confident` is False and the verdict says so rather than dressing up
a guess.
"""

from __future__ import annotations

import statistics
import time
from bisect import insort
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

Side = Literal["buy", "sell"]

# Below this many impact observations the baseline is noise, not a baseline.
MIN_IMPACT_SAMPLES = 20

# A band's size has to change by at least this fraction of its own average for
# the change to count as consumption or replenishment rather than jitter.
BAND_NOISE_FLOOR = 0.02


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Trade:
    """One fill, from the aggressor's point of view.

    `aggressor` is who crossed the spread -- the market order. That is the
    whole point of the tape: resting orders are passive by definition, so the
    only directional information in a fill is which side paid to get filled.
    """

    px: float
    sz: float
    aggressor: Side
    ts: float = 0.0

    @property
    def notional(self) -> float:
        return self.px * self.sz

    @property
    def signed_notional(self) -> float:
        return self.notional if self.aggressor == "buy" else -self.notional


@dataclass(frozen=True)
class Level:
    """One price level of resting size."""

    px: float
    sz: float
    n: int = 0          # orders at this level, when the feed provides it

    @property
    def notional(self) -> float:
        return self.px * self.sz


@dataclass(frozen=True)
class Fill:
    """The result of walking the book with a market order."""

    side: Side
    notional_requested: float
    notional_filled: float
    avg_px: float
    worst_px: float
    levels_consumed: int
    slippage_bps: float       # against mid; always positive = cost to you
    sweep_bps: float          # against the touch; what the walk itself cost
    exhausted: bool           # book ran out before the order was filled

    @property
    def filled_fraction(self) -> float:
        if self.notional_requested <= 0:
            return 0.0
        return self.notional_filled / self.notional_requested


@dataclass(frozen=True)
class Shelf:
    """A price level carrying outsized resting size."""

    px: float
    notional: float
    multiple: float           # how many times the median level it holds
    side: Side                # "buy" = resting bids (support)


# --------------------------------------------------------------------------
# the book
# --------------------------------------------------------------------------

@dataclass
class Book:
    """An L2 snapshot.

    `bids` descend from the touch, `asks` ascend. Both are normalised on
    construction so a feed that arrives in either order still works.
    """

    coin: str
    ts: float
    bids: list[Level] = field(default_factory=list)
    asks: list[Level] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.bids = sorted((b for b in self.bids if b.sz > 0),
                           key=lambda l: -l.px)
        self.asks = sorted((a for a in self.asks if a.sz > 0),
                           key=lambda l: l.px)

    # -- touch ------------------------------------------------------------

    @property
    def best_bid(self) -> float:
        return self.bids[0].px if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].px if self.asks else 0.0

    @property
    def mid(self) -> float:
        if self.bids and self.asks:
            return (self.best_bid + self.best_ask) / 2.0
        return self.best_bid or self.best_ask

    @property
    def spread_bps(self) -> float:
        m = self.mid
        if m <= 0 or not (self.bids and self.asks):
            return 0.0
        return (self.best_ask - self.best_bid) / m * 10_000.0

    @property
    def empty(self) -> bool:
        return not self.bids and not self.asks

    # -- depth ------------------------------------------------------------

    def depth(self, bps: float, side: Side) -> float:
        """Resting notional within `bps` of mid on one side."""
        m = self.mid
        if m <= 0:
            return 0.0
        span = m * bps / 10_000.0
        if side == "buy":
            return sum(l.notional for l in self.bids if l.px >= m - span)
        return sum(l.notional for l in self.asks if l.px <= m + span)

    def imbalance(self, bps: float = 25.0) -> float:
        """(bid - ask) / (bid + ask) within `bps`.

        +1 is all bid, -1 is all offer. Read it as pressure, not prediction:
        a heavy bid is as often a wall someone is leaning on to sell into as
        it is genuine demand.
        """
        b, a = self.depth(bps, "buy"), self.depth(bps, "sell")
        total = b + a
        return (b - a) / total if total > 0 else 0.0

    def band_notional(self, low_px: float, high_px: float) -> float:
        """All resting notional between two prices, both sides."""
        lo, hi = min(low_px, high_px), max(low_px, high_px)
        return sum(l.notional for l in (*self.bids, *self.asks)
                   if lo <= l.px <= hi)

    def shelves(self, bps: float = 50.0, multiple: float = 3.0,
                limit: int = 5) -> list[Shelf]:
        """Levels holding `multiple` times the median level within `bps`.

        These are where a move has to fight. A shelf that is never tested
        tells you nothing; a shelf that gets tested and holds is the thing
        you were already looking for by eye.
        """
        m = self.mid
        if m <= 0:
            return []
        span = m * bps / 10_000.0
        out: list[Shelf] = []

        for side, levels, keep in (("buy", self.bids, lambda l: l.px >= m - span),
                                   ("sell", self.asks, lambda l: l.px <= m + span)):
            near = [l for l in levels if keep(l)]
            if len(near) < 3:
                continue
            med = statistics.median(l.notional for l in near)
            if med <= 0:
                continue
            for l in near:
                mult = l.notional / med
                if mult >= multiple:
                    out.append(Shelf(px=l.px, notional=l.notional,
                                     multiple=mult, side=side))  # type: ignore[arg-type]

        out.sort(key=lambda s: -s.notional)
        return out[:limit]

    # -- the walk ---------------------------------------------------------

    def walk(self, notional: float, side: Side) -> Fill:
        """Fill `notional` by crossing the spread, level by level.

        A buy consumes asks upward, a sell consumes bids downward. Slippage
        comes back positive in both cases, so the number always reads as a
        cost regardless of direction -- same convention as the entry gap in
        `consensus.py`.

        This is the book as displayed. A real sweep will usually cost more,
        because size gets pulled as it is approached. Floor, not estimate.
        """
        levels = self.asks if side == "buy" else self.bids
        m = self.mid
        if notional <= 0 or not levels or m <= 0:
            return Fill(side=side, notional_requested=max(notional, 0.0),
                        notional_filled=0.0, avg_px=0.0, worst_px=0.0,
                        levels_consumed=0, slippage_bps=0.0, sweep_bps=0.0,
                        exhausted=bool(notional > 0))

        touch = levels[0].px
        remaining = notional
        qty = 0.0
        cost = 0.0
        consumed = 0
        worst = touch

        for lv in levels:
            if remaining <= 0:
                break
            take = min(remaining, lv.notional)
            if take <= 0:
                continue
            q = take / lv.px
            qty += q
            cost += take
            remaining -= take
            consumed += 1
            worst = lv.px

        if qty <= 0:
            return Fill(side=side, notional_requested=notional,
                        notional_filled=0.0, avg_px=0.0, worst_px=0.0,
                        levels_consumed=0, slippage_bps=0.0, sweep_bps=0.0,
                        exhausted=True)

        avg = cost / qty
        slip = (avg - m) / m if side == "buy" else (m - avg) / m
        sweep = (avg - touch) / touch if side == "buy" else (touch - avg) / touch

        return Fill(
            side=side,
            notional_requested=notional,
            notional_filled=cost,
            avg_px=avg,
            worst_px=worst,
            levels_consumed=consumed,
            slippage_bps=slip * 10_000.0,
            sweep_bps=sweep * 10_000.0,
            exhausted=remaining > 1e-9,
        )

    def round_trip_bps(self, notional: float) -> float:
        """Cost of entering and exiting `notional` at current depth.

        The number to compare your target against before you take the trade.
        A 2:1 setup targeting 40bps is a different proposition when the round
        trip is 3bps than when it is 25bps.
        """
        return (self.walk(notional, "buy").slippage_bps
                + self.walk(notional, "sell").slippage_bps)


# --------------------------------------------------------------------------
# the tape
# --------------------------------------------------------------------------

@dataclass
class FlowWindow:
    """Aggression over a time window."""

    start_ts: float
    end_ts: float
    buy_notional: float
    sell_notional: float
    trades: int

    @property
    def delta(self) -> float:
        return self.buy_notional - self.sell_notional

    @property
    def total(self) -> float:
        return self.buy_notional + self.sell_notional

    @property
    def lean(self) -> float:
        """-1 all selling, +1 all buying."""
        return self.delta / self.total if self.total > 0 else 0.0


@dataclass
class Divergence:
    """Price and flow disagreeing over a window."""

    price_change_bps: float
    delta_notional: float
    disagrees: bool
    note: str


class FlowTape:
    """Rolling tape with cumulative delta.

    Keeps `max_age` seconds of trades. CVD is cumulative over everything the
    tape has seen, not just the retained window, so it survives pruning.
    """

    def __init__(self, max_age: float = 3600.0):
        self.max_age = max_age
        self._trades: list[Trade] = []
        self._cvd = 0.0
        self._first_px: float | None = None
        self._last_px: float = 0.0

    def add(self, trade: Trade) -> None:
        if trade.sz <= 0 or trade.px <= 0:
            return
        self._trades.append(trade)
        self._cvd += trade.signed_notional
        if self._first_px is None:
            self._first_px = trade.px
        self._last_px = trade.px
        self._prune(trade.ts)

    def extend(self, trades: Iterable[Trade]) -> None:
        for t in trades:
            self.add(t)

    def _prune(self, now: float) -> None:
        if not self._trades:
            return
        cutoff = now - self.max_age
        if self._trades[0].ts >= cutoff:
            return
        self._trades = [t for t in self._trades if t.ts >= cutoff]

    # -- readings ---------------------------------------------------------

    @property
    def cvd(self) -> float:
        """Cumulative aggressive buying minus selling, in notional."""
        return self._cvd

    @property
    def last_px(self) -> float:
        return self._last_px

    def __len__(self) -> int:
        return len(self._trades)

    def window(self, seconds: float, now: float | None = None) -> FlowWindow:
        now = now if now is not None else (self._trades[-1].ts
                                           if self._trades else time.time())
        start = now - seconds
        buy = sell = 0.0
        n = 0
        for t in self._trades:
            if t.ts < start:
                continue
            n += 1
            if t.aggressor == "buy":
                buy += t.notional
            else:
                sell += t.notional
        return FlowWindow(start_ts=start, end_ts=now, buy_notional=buy,
                          sell_notional=sell, trades=n)

    def price_change_bps(self, seconds: float, now: float | None = None) -> float:
        now = now if now is not None else (self._trades[-1].ts
                                           if self._trades else time.time())
        start = now - seconds
        inside = [t for t in self._trades if t.ts >= start]
        if len(inside) < 2 or inside[0].px <= 0:
            return 0.0
        return (inside[-1].px - inside[0].px) / inside[0].px * 10_000.0

    def divergence(self, seconds: float = 300.0,
                   now: float | None = None) -> Divergence:
        """Price and aggression pointing opposite ways.

        This is the A/D divergence idea applied to the tape rather than to a
        derived indicator: buyers are paying up and price is not rewarding
        them, or sellers are hitting bids into a rising market. Either way
        somebody on the resting side is winning, and divergence is the first
        place that shows.
        """
        w = self.window(seconds, now)
        move = self.price_change_bps(seconds, now)

        if w.total <= 0 or abs(move) < 1e-9:
            return Divergence(move, w.delta, False, "not enough to compare")

        disagrees = (move > 0 and w.delta < 0) or (move < 0 and w.delta > 0)
        if not disagrees:
            return Divergence(move, w.delta, False,
                              "price and flow agree — nothing unusual")

        if move > 0:
            note = ("price up while net aggression is selling — bids are being "
                    "hit into a rising market, so passive buyers are lifting it")
        else:
            note = ("price down while net aggression is buying — offers are "
                    "being lifted into a falling market, so passive sellers "
                    "are leaning on it")
        return Divergence(move, w.delta, True, note)


# --------------------------------------------------------------------------
# how far price normally moves per dollar
# --------------------------------------------------------------------------

class ImpactBaseline:
    """Learned scale for "was that a lot of volume or not".

    Records (net aggressive notional, resulting price move) pairs and keeps the
    median bps moved per million. Median rather than mean because the
    distribution is violently fat-tailed -- one liquidation prints an impact
    fifty times the typical one and a mean never recovers.

    Samples with near-zero flow are discarded: dividing a price move by almost
    no volume produces an enormous ratio that says nothing about liquidity.
    """

    def __init__(self, min_notional: float = 10_000.0, keep: int = 500):
        self.min_notional = min_notional
        self.keep = keep
        self._ratios: list[float] = []
        self._n = 0

    def observe(self, delta_notional: float, price_change_bps: float) -> None:
        if abs(delta_notional) < self.min_notional:
            return
        per_million = abs(price_change_bps) / (abs(delta_notional) / 1e6)
        self._n += 1
        insort(self._ratios, per_million)
        if len(self._ratios) > self.keep:
            # drop from whichever end is further from the median, so trimming
            # does not skew the estimate it exists to protect
            mid = len(self._ratios) // 2
            if mid - 0 > len(self._ratios) - 1 - mid:
                self._ratios.pop(0)
            else:
                self._ratios.pop()

    @property
    def samples(self) -> int:
        return self._n

    @property
    def ready(self) -> bool:
        return len(self._ratios) >= MIN_IMPACT_SAMPLES

    @property
    def bps_per_million(self) -> float:
        return statistics.median(self._ratios) if self._ratios else 0.0

    def expected_bps(self, delta_notional: float) -> float:
        """How far price would normally move on this much net aggression."""
        return self.bps_per_million * abs(delta_notional) / 1e6


# --------------------------------------------------------------------------
# eaten versus replenished
# --------------------------------------------------------------------------

@dataclass
class BandStat:
    """What happened to the resting size in a price band over time."""

    low_px: float
    high_px: float
    observations: int
    first_notional: float
    last_notional: float
    min_notional: float
    max_notional: float
    consumed: float           # total decreases
    replenished: float        # total increases
    refill_events: int

    @property
    def net_change(self) -> float:
        return self.last_notional - self.first_notional

    @property
    def replenish_ratio(self) -> float:
        """Replenished over consumed.

        Above 1 means size is being added faster than it is being taken --
        somebody is paying to hold the level. That is the single strongest
        signal in this module, because unlike displayed depth it cannot be
        faked by an order that is never tested.
        """
        return self.replenished / self.consumed if self.consumed > 0 else 0.0

    @property
    def defended(self) -> bool:
        return self.consumed > 0 and self.replenish_ratio >= 0.8 and self.refill_events >= 2


class BandTracker:
    """Watches one price band across book snapshots.

    Deliberately tracks a band rather than an exact level. Levels shift by a
    tick constantly and tracking them exactly turns ordinary book churn into
    fake refill events.
    """

    def __init__(self, low_px: float, high_px: float,
                 noise_floor: float = BAND_NOISE_FLOOR):
        self.low_px = min(low_px, high_px)
        self.high_px = max(low_px, high_px)
        self.noise_floor = noise_floor
        self._series: list[float] = []
        self._consumed = 0.0
        self._replenished = 0.0
        self._refills = 0
        self._was_eaten = False

    def observe(self, book: Book) -> None:
        notional = book.band_notional(self.low_px, self.high_px)
        if self._series:
            prev = self._series[-1]
            scale = max(prev, notional, 1.0)
            change = notional - prev
            if abs(change) / scale >= self.noise_floor:
                if change < 0:
                    self._consumed += -change
                    self._was_eaten = True
                else:
                    self._replenished += change
                    if self._was_eaten:
                        self._refills += 1
                        self._was_eaten = False
        self._series.append(notional)

    def stat(self) -> BandStat:
        s = self._series or [0.0]
        return BandStat(
            low_px=self.low_px, high_px=self.high_px,
            observations=len(self._series),
            first_notional=s[0], last_notional=s[-1],
            min_notional=min(s), max_notional=max(s),
            consumed=self._consumed, replenished=self._replenished,
            refill_events=self._refills,
        )


# --------------------------------------------------------------------------
# absorption
# --------------------------------------------------------------------------

@dataclass
class Absorption:
    """The verdict at a level."""

    level: float
    window_s: float

    direction: Side              # which side is doing the aggressing
    aggressive_notional: float   # net, in that direction
    observed_bps: float          # how far price actually moved
    expected_bps: float          # how far it normally would have
    impact_ratio: float          # observed / expected

    band: BandStat
    confident: bool              # baseline has enough samples
    trades: int

    @property
    def absorbing(self) -> bool:
        """Aggression arriving and price not paying for it."""
        return (self.confident
                and self.aggressive_notional > 0
                and self.impact_ratio < 0.5)

    @property
    def thin(self) -> bool:
        """Price moving more easily than the flow justifies."""
        return (self.confident
                and self.aggressive_notional > 0
                and self.impact_ratio > 2.0)

    def verdict(self) -> str:
        if not self.confident:
            return (f"Baseline not established yet ({self.band.observations} "
                    f"book reads). Watch longer before trusting a reading — "
                    f"until then there is no scale to say whether this is a "
                    f"lot of volume or a little.")

        money = f"${self.aggressive_notional:,.0f}"
        side = "buying" if self.direction == "buy" else "selling"

        if self.absorbing:
            head = (f"ABSORBING. {money} of aggressive {side} into "
                    f"{self.level:,.2f} moved price {self.observed_bps:+.1f}bps "
                    f"when {self.expected_bps:.1f}bps was normal for that size.")
        elif self.thin:
            head = (f"THIN. {money} of aggressive {side} moved price "
                    f"{self.observed_bps:+.1f}bps against {self.expected_bps:.1f}bps "
                    f"expected — the book is giving way, not holding.")
        else:
            head = (f"Nothing unusual. {money} of aggressive {side} moved price "
                    f"about as far as that size normally does "
                    f"({self.observed_bps:+.1f} vs {self.expected_bps:.1f}bps).")

        b = self.band
        if b.consumed <= 0:
            tail = " No resting size has been taken at this level yet."
        elif b.defended:
            tail = (f" ${b.consumed:,.0f} eaten and ${b.replenished:,.0f} "
                    f"re-posted across {b.refill_events} refills — somebody is "
                    f"paying to hold it.")
        elif b.replenish_ratio < 0.3:
            tail = (f" ${b.consumed:,.0f} eaten and only ${b.replenished:,.0f} "
                    f"replaced — the level is being emptied, not defended.")
        else:
            tail = (f" ${b.consumed:,.0f} eaten, ${b.replenished:,.0f} replaced "
                    f"({b.refill_events} refills).")

        return head + tail


class LevelWatch:
    """Watch one price level: tape in, book snapshots in, verdict out.

    Feed it everything -- it filters to the band itself. The baseline is built
    from the same stream, so it learns this coin at this time of day rather
    than importing an assumption from somewhere else.
    """

    def __init__(self, coin: str, level: float, band_bps: float = 10.0,
                 window_s: float = 300.0, bucket_s: float = 15.0):
        self.coin = coin
        self.level = level
        self.band_bps = band_bps
        self.window_s = window_s
        self.bucket_s = bucket_s

        span = level * band_bps / 10_000.0
        self.tape = FlowTape(max_age=max(window_s * 4, 1200.0))
        self.baseline = ImpactBaseline()
        self.band = BandTracker(level - span, level + span)

        self._bucket_start: float | None = None
        self._bucket_delta = 0.0
        self._bucket_open_px = 0.0
        self._bucket_last_px = 0.0

    # -- ingest -----------------------------------------------------------

    def on_trade(self, trade: Trade) -> None:
        self.tape.add(trade)
        self._accumulate(trade)

    def on_book(self, book: Book) -> None:
        self.band.observe(book)

    def _accumulate(self, trade: Trade) -> None:
        """Roll trades into fixed buckets and feed each finished one to the
        baseline. Bucketing matters: impact measured trade by trade is mostly
        spread noise, and measured over long windows averages away the thing
        being measured."""
        if self._bucket_start is None:
            self._bucket_start = trade.ts
            self._bucket_open_px = trade.px

        if trade.ts < self._bucket_start:
            # Time went backwards: a clock fallback, a reconnect replaying
            # history, or out-of-order delivery. Without this the bucket never
            # reaches its own end and the baseline silently never trains --
            # the dashboard would say "not established yet" forever with
            # nothing to show why. Restart the bucket here instead.
            self._bucket_start = trade.ts
            self._bucket_open_px = trade.px
            self._bucket_delta = 0.0

        if trade.ts - self._bucket_start >= self.bucket_s:
            if self._bucket_open_px > 0 and self._bucket_last_px > 0:
                move = ((self._bucket_last_px - self._bucket_open_px)
                        / self._bucket_open_px * 10_000.0)
                self.baseline.observe(self._bucket_delta, move)
            self._bucket_start = trade.ts
            self._bucket_open_px = trade.px
            self._bucket_delta = 0.0

        self._bucket_delta += trade.signed_notional
        self._bucket_last_px = trade.px

    # -- read -------------------------------------------------------------

    def absorption(self, now: float | None = None) -> Absorption:
        w = self.tape.window(self.window_s, now)
        move = self.tape.price_change_bps(self.window_s, now)
        direction: Side = "buy" if w.delta >= 0 else "sell"
        net = abs(w.delta)

        expected = self.baseline.expected_bps(net)
        # Signed so the reading is about the aggressor getting paid, not about
        # raw movement: a buyer pushing price DOWN is not being rewarded.
        signed_move = move if direction == "buy" else -move
        ratio = (signed_move / expected) if expected > 0 else 0.0

        return Absorption(
            level=self.level, window_s=self.window_s,
            direction=direction, aggressive_notional=net,
            observed_bps=signed_move, expected_bps=expected,
            impact_ratio=ratio, band=self.band.stat(),
            confident=self.baseline.ready, trades=w.trades,
        )

    def render(self, book: Book | None = None,
               size: float = 0.0) -> str:
        a = self.absorption()
        lines = [
            f"{self.coin} @ {self.level:,.2f}  "
            f"(±{self.band_bps:.0f}bps, {self.window_s:.0f}s window)",
            f"  trades seen      {len(self.tape)}  "
            f"({a.trades} in window)",
            f"  net aggression   ${a.aggressive_notional:,.0f} "
            f"{'buying' if a.direction == 'buy' else 'selling'}",
            f"  impact           {a.observed_bps:+.1f}bps observed / "
            f"{a.expected_bps:.1f}bps expected",
            f"  eaten/replaced   ${a.band.consumed:,.0f} / "
            f"${a.band.replenished:,.0f}  ({a.band.refill_events} refills)",
        ]
        if book is not None and not book.empty:
            lines.append(f"  spread           {book.spread_bps:.2f}bps")
            lines.append(f"  imbalance ±25bps {book.imbalance(25.0):+.2f}")
            if size > 0:
                buy, sell = book.walk(size, "buy"), book.walk(size, "sell")
                lines.append(
                    f"  ${size:,.0f} costs  {buy.slippage_bps:.1f}bps in / "
                    f"{sell.slippage_bps:.1f}bps out "
                    f"= {buy.slippage_bps + sell.slippage_bps:.1f}bps round trip")
                if buy.exhausted or sell.exhausted:
                    lines.append("  *** book cannot fill that size ***")
        lines += ["", f"  {a.verdict()}"]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# side convention
# --------------------------------------------------------------------------

def check_side_convention(samples: Sequence[tuple[str, float]],
                          mid: float) -> str:
    """Work out, from live data, which raw side code means buy-aggressor.

    Every venue labels this differently and some label it inconsistently. An
    aggressive buy lifts the offer, so it prints at or above mid; an aggressive
    sell hits the bid and prints at or below. Feed this a sample of
    (raw_side, price) pairs and it reports what the codes actually mean rather
    than what the documentation claims.

    Run it once on the live feed before trusting any flow number -- a flipped
    convention inverts every reading in this module.
    """
    if mid <= 0 or not samples:
        return "no samples"

    by_code: dict[str, list[float]] = {}
    for code, px in samples:
        by_code.setdefault(str(code), []).append(px)

    parts = []
    for code, prices in sorted(by_code.items()):
        above = sum(1 for p in prices if p > mid)
        below = sum(1 for p in prices if p < mid)
        if above + below == 0:
            parts.append(f"{code}: all at mid, inconclusive")
            continue
        share = above / (above + below)
        if share >= 0.65:
            parts.append(f"{code} = BUY aggressor ({share:.0%} above mid, n={len(prices)})")
        elif share <= 0.35:
            parts.append(f"{code} = SELL aggressor ({1 - share:.0%} below mid, n={len(prices)})")
        else:
            parts.append(f"{code}: ambiguous ({share:.0%} above mid, n={len(prices)}) "
                         f"— sample more or mid has moved")
    return "; ".join(parts)
