"""From a book read to a trade you can accept or ignore.

`bookread.py` answers "which way is this candle going". That is not yet a
trade. A trade needs four more things, and every one of them has to come from
something measured rather than something asserted:

    WHERE    entry, at the side of the book you would actually cross
    HOW FAR  a target, from what bars on this timeframe actually travel
    WHERE WRONG   an invalidation, at the resting level the read leans on
    WHETHER IT PAYS   the target against the real cost of getting in and out

The fourth is the one that kills most suggestions, and it should. A read can
be right about direction and still be a losing trade, because a 4bps move
does not survive a 6bps round trip. Direction is not edge. Direction that
clears cost with room to spare is edge.

REFUSING IS THE MAIN OUTPUT

Most of the time this returns NoTrade, and NoTrade names which gate stopped
it. That is deliberate. A tool that finds a trade every time it is asked is
not reading the book, it is generating sentences. The gates are ordered so
the reason you get back is the first real problem, not the last one checked.

NOTHING HERE PLACES AN ORDER

The output is a sentence and a set of numbers. Taking the trade is a
decision made by a person, and what that person decides is recorded so the
two can be compared later. That comparison -- what the tool suggested versus
what you took -- is the only way to find out whether either of you is adding
anything.

THE TARGET IS NOT A PREDICTION

It is a statement about range: bars on this instrument and this timeframe
typically travel N basis points, there is this much of the candle left, and
range accumulates roughly with the square root of time. So the target is the
part of a typical bar still available, halved, because catching a whole bar
requires being right at both ends. If recent bars have not been measured,
there is no target and no suggestion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Sequence

from .bookread import BookRead
from .flow import Book, Level
from .structure import Candle

Side = Literal["long", "short"]

# --- gates -----------------------------------------------------------------

# Book updates needed before the dynamics mean anything. Below this the
# replenishment and depletion counters are two or three events of noise.
MIN_SAMPLES = 5

# Conviction below which the book is not saying enough to act on. This is the
# single most important knob and it is deliberately high: the cost of a
# marginal trade is certain and its edge is not.
MIN_CONVICTION = 0.45

# The target has to beat the round trip by this much. At 1.0 you are trading
# for the broker. At 2.0 half of a correct call is still yours.
MIN_COST_MULTIPLE = 2.0

# Reward against risk. Below this the shape is wrong even when the direction
# is right: you need a hit rate you have not measured to break even.
MIN_RR = 1.5

# Inside this many seconds of the close there is not enough candle left for
# the move to happen, whatever the book says.
MIN_SECONDS_LEFT = 20.0

# Fraction of a typical bar's range a correct scalp can expect to capture.
# Not the whole bar: that needs the low and the high, which needs two correct
# calls, not one.
CAPTURE = 0.5

# The invalidation must be at least this fraction of a typical bar's range
# away, whatever the book says.
#
# Without this, a razor-thin spread produces a razor-thin stop. A book with a
# 0.26bp spread and a resting level two ticks behind it yields an
# invalidation under 1bp away on an instrument whose bars routinely travel
# forty -- which reads as a magnificent 22R and is in fact a stop sitting
# inside the noise. It gets hit by the next flicker, and the measured hit
# rate then reports the tool as useless for a reason that has nothing to do
# with whether it read the book correctly.
MIN_RISK_BAR_FRACTION = 0.15

# How many recent bars the range estimate uses.
RANGE_BARS = 20


def typical_range_bps(candles: Sequence[Candle], bars: int = RANGE_BARS
                      ) -> float:
    """Median high-low range of recent CLOSED bars, in basis points.

    Median rather than mean: one news bar with ten times the range would
    otherwise set a target nothing else on the chart can reach.

    Returns 0.0 when there is not enough history, and 0.0 means no
    suggestion. An unmeasured target is an invented one.
    """
    ranges = []
    for c in candles[-bars:]:
        if c.high > 0 and c.low > 0 and c.high >= c.low:
            mid = (c.high + c.low) / 2.0
            if mid > 0:
                ranges.append((c.high - c.low) / mid * 10_000.0)
    if len(ranges) < 5:
        return 0.0
    ranges.sort()
    n = len(ranges)
    return (ranges[n // 2] if n % 2
            else (ranges[n // 2 - 1] + ranges[n // 2]) / 2.0)


def infer_tick(book: Book) -> float:
    """Smallest price increment the book actually shows.

    The minimum positive gap between adjacent levels. A book can never have a
    tick larger than its smallest observed gap, and underestimating is the
    safe direction: it only makes the displayed tick count larger, never the
    target price wrong.
    """
    gaps = []
    for side in (book.bids, book.asks):
        for a, b in zip(side, side[1:]):
            g = abs(b.px - a.px)
            if g > 0:
                gaps.append(g)
    if not gaps:
        # Fall back to the spread, which is at least one tick by definition.
        return book.best_ask - book.best_bid if not book.empty else 0.0
    return min(gaps)


def _shelf(levels: Sequence[Level]) -> Level | None:
    """The biggest resting level behind the touch.

    This is what a long is leaning on. The touch itself is excluded where
    possible: putting the stop one tick under the best bid means any single
    sweep of the front queue takes you out, which is noise, not invalidation.
    """
    if not levels:
        return None
    body = list(levels[1:]) or list(levels[:1])
    return max(body, key=lambda l: l.notional)


@dataclass(frozen=True)
class NoTrade:
    """Why there is no suggestion. The gate that stopped it, named."""

    gate: str
    detail: str
    side: Side | None = None
    conviction: float = 0.0

    take = False

    def sentence(self) -> str:
        return f"No trade — {self.detail}"

    def to_dict(self) -> dict:
        return {"take": False, "gate": self.gate, "detail": self.detail,
                "sentence": self.sentence(), "side": self.side,
                "conviction": round(self.conviction, 4)}


@dataclass(frozen=True)
class Suggestion:
    """A trade to accept or ignore, with every number it rests on."""

    coin: str
    interval: str
    side: Side

    entry: float
    target_px: float
    stop_px: float

    tick: float
    target_ticks: int
    risk_ticks: int
    target_bps: float
    risk_bps: float

    cost_bps: float
    fees_included: bool
    conviction: float
    score: float
    seconds_left: float
    samples: int
    spread_bps: float

    reasons: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)

    take = True

    @property
    def rr(self) -> float:
        return round(self.target_bps / self.risk_bps, 2) if self.risk_bps > 0 else 0.0

    @property
    def cost_multiple(self) -> float:
        """How many times the round trip the target is worth. Under 2 this
        would not have been suggested."""
        return round(self.target_bps / self.cost_bps, 1) if self.cost_bps > 0 else 0.0

    @property
    def direction_word(self) -> str:
        return "LONG" if self.side == "long" else "SHORT"

    def headline(self) -> str:
        sign = "+" if self.side == "long" else "-"
        return (f"{self.direction_word} {self.coin} — {self.target_ticks} ticks "
                f"({sign}{self.target_bps:.0f}bps) from {self.entry:,.4f}".rstrip("0").rstrip("."))

    def sentence(self) -> str:
        """The whole suggestion, in the form a person reads and decides on."""
        word = "above" if self.side == "long" else "below"
        opposite = "below" if self.side == "long" else "above"
        sign = "+" if self.side == "long" else "-"

        lines = [
            f"{self.direction_word} {self.coin} {self.target_ticks} ticks "
            f"({sign}{self.target_bps:.0f}bps) from {self.entry:,.6g} — "
            f"target {self.target_px:,.6g}, invalid {opposite} "
            f"{self.stop_px:,.6g}. {self.rr}R.",
        ]
        if self.reasons:
            lines.append("Why: " + "; ".join(self.reasons) + ".")
        if self.cautions:
            lines.append("Against: " + "; ".join(self.cautions) + ".")

        fee_note = "" if self.fees_included else " (book only, no fees)"
        lines.append(
            f"Cost: round trip {self.cost_bps:.1f}bps{fee_note} — the target "
            f"pays {self.cost_multiple}x.")
        lines.append(
            f"Conviction {self.conviction:.0%} from {self.samples} book "
            f"updates, spread {self.spread_bps:.2f}bps. "
            f"{_clock(self.seconds_left)} left on the {self.interval}.")
        # Stated every time, not as a footnote: the thin side being `word` is
        # the mechanism, and if that stops being true the trade is over
        # before the stop is reached.
        lines.append(f"This is a read of the book right now, not a forecast. "
                     f"If the thin side stops being {word}, the reason is gone.")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "take": True, "coin": self.coin, "interval": self.interval,
            "side": self.side, "entry": self.entry,
            "target_px": self.target_px, "stop_px": self.stop_px,
            "tick": self.tick, "target_ticks": self.target_ticks,
            "risk_ticks": self.risk_ticks,
            "target_bps": round(self.target_bps, 2),
            "risk_bps": round(self.risk_bps, 2), "rr": self.rr,
            "cost_bps": round(self.cost_bps, 3),
            "cost_multiple": self.cost_multiple,
            "fees_included": self.fees_included,
            "conviction": round(self.conviction, 4),
            "score": round(self.score, 4),
            "seconds_left": round(self.seconds_left, 1),
            "samples": self.samples,
            "spread_bps": round(self.spread_bps, 3),
            "reasons": self.reasons, "cautions": self.cautions,
            "headline": self.headline(),
            "sentence": self.sentence(),
        }


def _clock(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


def _reasons(read: BookRead, limit: int = 3) -> list[str]:
    """The strongest components of the read, in plain words.

    Ranked by contribution -- value times weight -- so the sentence names
    what actually decided the call rather than whatever reads well.

    Only components pointing the SAME way as the call are listed. A reason
    is an argument for the trade; a component pulling against it is a
    caution, and mixing the two produces a sentence that argues both sides.
    """
    lean = 1.0 if read.score > 0 else -1.0
    ranked = sorted(read.components(),
                    key=lambda c: -abs(c["value"] * c["weight"]))
    out = []
    for c in ranked:
        if abs(c["value"]) < 0.05 or c["value"] * lean < 0:
            continue
        out.append(c["note"])
        if len(out) >= limit:
            break
    return out


def _cautions(read: BookRead, spread_bps: float, target_bps: float
              ) -> list[str]:
    """What argues against the trade. Never mixed into the reasons for it.

    Absorption is the important one. It is a warning -- somebody is taking
    the other side without price moving -- and listing it among the reasons
    to go long, which an earlier version did, makes a sentence that argues
    for and against itself in the same breath.
    """
    out = []
    if read.absorbed:
        out.append("aggression is being absorbed — somebody is taking the "
                   "other side without the mid moving")
    lean = 1.0 if read.score > 0 else -1.0
    against = [c for c in read.components()
               if c["value"] * lean < -0.15]
    for c in sorted(against, key=lambda c: c["value"] * lean)[:2]:
        out.append(f"{c['name']} disagrees ({c['note']})")
    if spread_bps > target_bps * 0.25:
        out.append(f"the spread is {spread_bps:.1f}bps against a "
                   f"{target_bps:.0f}bps target — a wide fraction of the move")
    return out


def suggest(read: BookRead | None, book: Book | None, *,
            coin: str, interval: str, interval_s: float,
            seconds_left: float, recent: Sequence[Candle],
            notional: float = 10_000.0,
            fee_bps: float = 0.0,
            tick: float | None = None,
            min_conviction: float = MIN_CONVICTION,
            min_rr: float = MIN_RR,
            min_cost_multiple: float = MIN_COST_MULTIPLE,
            ) -> Suggestion | NoTrade:
    """A trade to take or ignore, or a named reason there isn't one.

    `recent` is closed bars on the trading timeframe; they set the target.
    `notional` is the size you would actually trade, because the round trip
    depends on it -- a size that walks four levels costs more than one that
    lifts the touch.
    """
    # -- the book has to be saying something ------------------------------
    if read is None or read.samples < MIN_SAMPLES:
        n = read.samples if read else 0
        return NoTrade("warming", f"the book feed has {n} updates so far; "
                                  f"the dynamics need at least {MIN_SAMPLES}")

    if book is None or book.empty:
        return NoTrade("no book", "no order book on this market right now")

    # One-sided and locked books have to go before anything reads a price off
    # them. With no asks, `best_ask` is 0.0, so the spread becomes the whole
    # bid price and the invalidation lands below zero; worse, the round trip
    # comes back as 0.0 and reads as FREE rather than unmeasurable, which
    # switched the cost gate off entirely and produced a 3R short on a market
    # with nothing to buy it back from.
    if not book.bids or not book.asks:
        return NoTrade("one sided",
                       "only one side of this book has resting orders — "
                       "there is nothing to price the other half of the "
                       "trade against")
    if book.best_bid <= 0 or book.best_ask <= 0:
        return NoTrade("no book", "the touch has no price on it")
    if book.best_ask <= book.best_bid:
        return NoTrade("locked",
                       f"the book is locked or crossed "
                       f"({book.best_bid:,.6g} bid against "
                       f"{book.best_ask:,.6g} offer) — nothing can be priced "
                       f"off it until it uncrosses")

    if read.direction == "flat":
        return NoTrade("flat", f"the book is balanced (score {read.score:+.2f}) "
                               f"— neither side is winning",
                       conviction=read.conviction)

    side: Side = "long" if read.direction == "up" else "short"

    if read.conviction < min_conviction:
        return NoTrade("conviction",
                       f"the book leans {side} but only at "
                       f"{read.conviction:.0%} conviction, under the "
                       f"{min_conviction:.0%} floor",
                       side=side, conviction=read.conviction)

    if seconds_left < MIN_SECONDS_LEFT:
        return NoTrade("too late",
                       f"only {_clock(seconds_left)} left on the {interval} — "
                       f"not enough candle for the move",
                       side=side, conviction=read.conviction)

    # -- how far can it go ------------------------------------------------
    bar_bps = typical_range_bps(recent)
    if bar_bps <= 0:
        return NoTrade("no range",
                       f"no recent {interval} bars to size a target from — "
                       f"needs at least 5 closed bars",
                       side=side, conviction=read.conviction)

    if interval_s <= 0:
        return NoTrade("no clock",
                       f"{interval!r} has no known bar length, so there is no "
                       f"way to tell how much of the candle is left",
                       side=side, conviction=read.conviction)

    # Range grows with the square root of time, not linearly. Half an hour
    # left of a one-hour bar is about 70% of its range, not 50%.
    time_left = min(seconds_left / interval_s, 1.0)
    target_bps = bar_bps * math.sqrt(time_left) * CAPTURE

    # -- entry, at the side you would actually cross ----------------------
    entry = book.best_ask if side == "long" else book.best_bid
    if entry <= 0:
        return NoTrade("no book", "the touch has no price on it")

    t = tick if tick and tick > 0 else infer_tick(book)
    if t <= 0:
        return NoTrade("no tick", "cannot read a tick size from this book")

    # -- where it is wrong, from the book ---------------------------------
    spread = max(book.best_ask - book.best_bid, 0.0)
    shelf = _shelf(book.bids if side == "long" else book.asks)
    if shelf is not None:
        raw_stop = shelf.px - t if side == "long" else shelf.px + t
    else:
        raw_stop = (entry - 3 * spread) if side == "long" else (entry + 3 * spread)

    # A stop inside the noise is not an invalidation, it is a donation. The
    # binding constraint is usually the volatility floor, not the spread.
    vol_floor = entry * (bar_bps * MIN_RISK_BAR_FRACTION) / 10_000.0
    floor = max(3 * spread, 2 * t, vol_floor)
    if abs(entry - raw_stop) < floor:
        raw_stop = (entry - floor) if side == "long" else (entry + floor)

    stop_px = raw_stop
    risk_bps = abs(entry - stop_px) / entry * 10_000.0

    # -- snap the target to ticks, then recompute from the snapped price --
    raw_target = (entry * (1 + target_bps / 10_000.0) if side == "long"
                  else entry * (1 - target_bps / 10_000.0))
    steps = max(1, round(abs(raw_target - entry) / t))
    target_px = entry + steps * t if side == "long" else entry - steps * t
    if target_px <= 0:
        return NoTrade("no target", "the target lands at or below zero")
    target_bps = abs(target_px - entry) / entry * 10_000.0

    # -- does it pay -------------------------------------------------------
    #
    # Walked explicitly rather than through `round_trip_bps`, because the
    # fills carry `exhausted` and the summed number does not. A size larger
    # than the displayed book prices the round trip on whatever fraction
    # could fill, which makes the most expensive trades look like the
    # cheapest ones.
    buy, sell = book.walk(notional, "buy"), book.walk(notional, "sell")
    if buy.exhausted or sell.exhausted:
        return NoTrade(
            "depth",
            f"${notional:,.0f} is more than this book is showing — the round "
            f"trip cannot be priced, and a size that walks past the last "
            f"level costs far more than the part that fills",
            side=side, conviction=read.conviction)

    cost_bps = buy.slippage_bps + sell.slippage_bps + max(fee_bps, 0.0)
    if cost_bps <= 0:
        return NoTrade("no cost",
                       "the round trip on this book prices at zero, which "
                       "means it could not be measured rather than that it "
                       "is free",
                       side=side, conviction=read.conviction)

    if target_bps < cost_bps * min_cost_multiple:
        return NoTrade(
            "cost",
            f"the {interval} is only offering about {target_bps:.0f}bps from "
            f"here and the round trip costs {cost_bps:.1f}bps — that is "
            f"{target_bps / cost_bps:.1f}x, under the {min_cost_multiple:.0f}x "
            f"floor. The read is {side}, the spread just eats it",
            side=side, conviction=read.conviction)

    rr = target_bps / risk_bps if risk_bps > 0 else 0.0
    if rr < min_rr:
        return NoTrade(
            "shape",
            f"the read is {side} but the level it leans on is "
            f"{risk_bps:.0f}bps away against a {target_bps:.0f}bps target — "
            f"{rr:.1f}R, under the {min_rr}R floor",
            side=side, conviction=read.conviction)

    return Suggestion(
        coin=coin, interval=interval, side=side,
        entry=entry, target_px=target_px, stop_px=stop_px,
        tick=t, target_ticks=steps,
        risk_ticks=max(1, round(abs(entry - stop_px) / t)),
        target_bps=target_bps, risk_bps=risk_bps,
        cost_bps=cost_bps, fees_included=fee_bps > 0,
        conviction=read.conviction, score=read.score,
        seconds_left=seconds_left, samples=read.samples,
        spread_bps=read.spread_bps,
        reasons=_reasons(read),
        cautions=_cautions(read, read.spread_bps, target_bps),
    )


# --------------------------------------------------------------------------
# settling a suggestion
# --------------------------------------------------------------------------

Outcome = Literal["target", "stop", "neither"]


def settle(side: Side, target_px: float, stop_px: float,
           high: float, low: float) -> Outcome:
    """Which came first, target or stop, over a period with this high and low.

    When both were touched inside the same period this returns "stop". Bar
    data cannot say which came first, and assuming the good one is how a
    backtest manufactures a hit rate it will never reproduce. The pessimistic
    reading is the only one that does not flatter.
    """
    if high <= 0 or low <= 0 or high < low:
        return "neither"

    if side == "long":
        hit_target = high >= target_px
        hit_stop = low <= stop_px
    else:
        hit_target = low <= target_px
        hit_stop = high >= stop_px

    if hit_stop:
        return "stop"
    if hit_target:
        return "target"
    return "neither"
