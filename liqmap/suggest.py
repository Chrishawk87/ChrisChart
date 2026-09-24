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
from typing import Any, Literal, Sequence

from .bookread import BookRead
from .confirm import CandleAction, Confirmation, confirm
from .flow import Book, Level
from .structure import Candle

Side = Literal["long", "short"]
Mode = Literal["scalp", "range"]


def breakeven_hit_rate(reward_bps: float, risk_bps: float,
                       cost_bps: float = 0.0) -> float:
    """The share of trades that must reach the target to break even.

    Expected value is zero when

        p * reward - (1 - p) * risk - cost = 0

    so p = (risk + cost) / (reward + risk). This single number is what makes
    a 0.6R scalp and a 3R swing comparable: it says what each one demands of
    you, in the same units, with the cost of trading already counted.
    """
    denom = reward_bps + risk_bps
    if denom <= 0:
        return 1.0
    return min(1.0, max(0.0, (risk_bps + cost_bps) / denom))

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

# Reward against risk, in RANGE mode. Below this the shape is wrong even when
# the direction is right: you need a hit rate you have not measured to break
# even.
#
# Scalp mode does NOT use this, and it would refuse almost every scalp if it
# did. That is the point of having two modes -- see MAX_BREAKEVEN.
MIN_RR = 1.5

# The gate that replaces R:R in scalp mode.
#
# A small target has a high hit rate and a bad reward-to-risk ratio, and
# those are the same fact viewed twice. Judging a scalp by R:R refuses every
# scalp; judging it by hit rate alone accepts trades that lose money slowly.
# The honest number is the one that combines them -- the share of trades that
# must reach the target for the whole thing to break even:
#
#     p = (risk + cost) / (reward + risk)
#
# At 3R that is 27%. At 0.67R it is 62%. Both can be good trades; they are
# just different businesses. What is NOT a good trade is one needing a hit
# rate no book read sustains, and this is where that line sits.
MAX_BREAKEVEN = 0.70

# In scalp mode the invalidation may be at most this multiple of the target.
# Beyond it one loss erases too many wins, so the stop is pulled in to the
# cap rather than the trade being refused -- a scalp leans on the book read,
# not on a shelf thirty basis points away.
MAX_RISK_MULTIPLE = 2.5

# And at least this share of the target, so the stop is not inside the noise
# of the very move it is waiting for. Below 1.0 because a scalp wants the
# reward to beat the risk, and much below it the stop is flicker bait.
SCALP_RISK_SHARE = 0.8

# The most of the remaining expected range a target may ask for. Above this
# the trade needs the candle to do more than a candle typically does.
MAX_RANGE_SHARE = 0.8

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
    mode: Mode = "range"
    measured_rate: float | None = None      # what you actually achieve
    measured_n: int = 0
    confirmation: Confirmation | None = None

    reasons: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)

    take = True

    @property
    def rr(self) -> float:
        return round(self.target_bps / self.risk_bps, 2) if self.risk_bps > 0 else 0.0

    @property
    def breakeven(self) -> float:
        """Hit rate this trade needs to pay for itself, cost included."""
        return round(breakeven_hit_rate(self.target_bps, self.risk_bps,
                                        self.cost_bps), 4)

    @property
    def edge_pts(self) -> float | None:
        """Measured hit rate minus the one required, in percentage points.

        None until enough trades have settled. This is the only number that
        says whether to take the trade rather than whether it is well shaped
        -- and it cannot exist until the forward test has run for a while,
        which is the honest answer to "is this any good".
        """
        if self.measured_rate is None or self.measured_n < 20:
            return None
        return round((self.measured_rate - self.breakeven) * 100, 1)

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
        if self.confirmation is not None:
            c = self.confirmation
            lines.append(
                f"Book says {c.book.upper()}, price is going "
                f"{c.candle.upper()} — confirmed.")
        if self.reasons:
            lines.append("Why: " + "; ".join(self.reasons) + ".")
        if self.cautions:
            lines.append("Against: " + "; ".join(self.cautions) + ".")

        fee_note = "" if self.fees_included else " (book only, no fees)"
        lines.append(
            f"Cost: round trip {self.cost_bps:.1f}bps{fee_note} — the target "
            f"pays {self.cost_multiple}x.")

        # The number that decides it. A 0.6R scalp and a 3R swing are not
        # comparable by R; they are comparable by what each demands of you.
        need = f"Needs {self.breakeven:.0%} to break even"
        if self.measured_rate is not None and self.measured_n >= 20:
            gap = self.edge_pts
            verdict = ("clear" if gap > 5 else "short" if gap < -5 else "level")
            need += (f"; you are running {self.measured_rate:.0%} on "
                     f"{self.measured_n} settled — {abs(gap):.0f} points "
                     f"{verdict}")
        elif self.measured_n:
            need += (f"; only {self.measured_n} settled so far, not enough "
                     f"to compare")
        else:
            need += "; nothing settled yet to compare against"
        lines.append(need + ".")
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
            "mode": self.mode,
            "breakeven": self.breakeven,
            "measured_rate": self.measured_rate,
            "measured_n": self.measured_n,
            "edge_pts": self.edge_pts,
            "confirmation": (self.confirmation.to_dict()
                             if self.confirmation else None),
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
            action: CandleAction | None = None,
            require_confirmation: bool = True,
            held_s: float = 0.0, flips: int = 0,
            participation: Any = None,
            mode: Mode = "range",
            min_conviction: float = MIN_CONVICTION,
            min_rr: float = MIN_RR,
            min_cost_multiple: float = MIN_COST_MULTIPLE,
            max_breakeven: float = MAX_BREAKEVEN,
            measured_rate: float | None = None,
            measured_n: int = 0,
            ) -> Suggestion | NoTrade:
    """A trade to take or ignore, or a named reason there isn't one.

    `recent` is closed bars on the trading timeframe; they set the range.
    `notional` is the size you would actually trade, because the round trip
    depends on it -- a size that walks four levels costs more than one that
    lifts the touch.

    TWO MODES, WHICH ARE TWO DIFFERENT BUSINESSES

    `range` aims for half of what a bar typically travels. Fewer wins,
    bigger ones, judged on reward against risk.

    `scalp` aims for the SMALLEST move that still clears the round trip.
    A nearer target is mechanically more likely to be touched, so this is
    the highest hit rate available on a trade that still pays -- which is
    the whole point of hyper-scalping. It also has worse reward-to-risk, and
    those are the same fact seen twice, so scalp mode is judged on the hit
    rate it REQUIRES rather than on R.
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

    # -- does price agree with the book -----------------------------------
    #
    # The hard gate, and it comes before every other calculation because
    # there is no point sizing a trade the price action is arguing against.
    #
    # A HARD gate, never a weight: blending a strong book with falling price
    # averages a disagreement into a weak agreement, and a weak long is
    # exactly the wrong thing to hold where a passive seller is filling
    # every bid without moving the price.
    agreement: Confirmation | None = None
    if action is not None or require_confirmation:
        agreement = confirm(read.direction, read.conviction, action,
                            held_s=held_s, flips=flips,
                            participation=participation)
        if agreement.verdict == "conflict":
            return NoTrade("conflict", agreement.detail,
                           side=side, conviction=read.conviction)
        if agreement.verdict == "unconfirmed":
            return NoTrade("unconfirmed", agreement.detail,
                           side=side, conviction=read.conviction)

        # Agreeing is not the same as holding. A confirmation that appeared
        # on this tick, or one in a market flipping six times a minute, or
        # one nobody paid for, is the confirmation that evaporates between
        # seeing it and acting on it. Scalp mode refuses those; range mode
        # holds long enough not to care.
        if mode == "scalp" and not agreement.backed:
            why = agreement.instability()
            if why:
                return NoTrade("unsettled",
                               f"the book and price agree {agreement.book} — "
                               f"{why}",
                               side=side, conviction=read.conviction)

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
    available_bps = bar_bps * math.sqrt(time_left)

    # -- entry, at the side you would actually cross ----------------------
    entry = book.best_ask if side == "long" else book.best_bid
    if entry <= 0:
        return NoTrade("no book", "the touch has no price on it")

    t = tick if tick and tick > 0 else infer_tick(book)
    if t <= 0:
        return NoTrade("no tick", "cannot read a tick size from this book")

    # -- what the round trip costs ----------------------------------------
    #
    # Priced BEFORE the target, because in scalp mode the target is derived
    # from it. Walked explicitly rather than through `round_trip_bps`: the
    # fills carry `exhausted` and the summed number does not, so a size
    # larger than the displayed book would otherwise price on whatever
    # fraction could fill and make the most expensive trade available look
    # like one of the cheapest.
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

    # -- where it is wrong, before deciding how far to aim -----------------
    #
    # The invalidation is computed first because in scalp mode the target
    # depends on it. A stop inside the noise is not an invalidation, it is a
    # donation, and the floor it has to clear -- three spreads, two ticks --
    # does not shrink just because the target is small.
    spread = max(book.best_ask - book.best_bid, 0.0)
    shelf = _shelf(book.bids if side == "long" else book.asks)
    if shelf is not None:
        shelf_dist = abs(entry - (shelf.px - t if side == "long"
                                  else shelf.px + t))
    else:
        shelf_dist = 3 * spread
    hard_floor = max(3 * spread, 2 * t)

    def stop_for(target_px_bps: float) -> tuple[float, bool]:
        """Invalidation distance in price for a given target, and whether it
        had to be pulled in from the structural level."""
        if mode != "scalp":
            # Held for much of the bar, so the bar's own range is the noise
            # it must survive.
            floor = max(hard_floor,
                        entry * (bar_bps * MIN_RISK_BAR_FRACTION) / 10_000.0)
            return max(shelf_dist, floor), False

        # A scalp is held for the time price takes to travel a few basis
        # points -- seconds, not the candle -- so the noise it must survive
        # scales with the TARGET, not with the bar. Using the bar's range
        # here puts the invalidation thirty basis points from a nine basis
        # point target, which is not a cautious scalp, it is a losing one
        # wearing a cautious stop.
        floor = max(hard_floor,
                    entry * (target_px_bps * SCALP_RISK_SHARE) / 10_000.0)
        cap = entry * (target_px_bps * MAX_RISK_MULTIPLE) / 10_000.0
        # A structural level further away than the cap is not what a scalp
        # leans on -- the book read is, and when that stops being true the
        # reason is gone whether or not a shelf has broken. So it is pulled
        # in rather than the trade refused, and the pull is disclosed.
        if shelf_dist > max(cap, floor):
            return max(cap, floor), True
        return max(shelf_dist, floor), False

    # -- how far to aim ----------------------------------------------------
    scalp_stop_capped = False

    if mode != "scalp":
        steps = max(1, round(available_bps * CAPTURE / 10_000.0 * entry / t))
        risk_dist, _ = stop_for(0.0)
    else:
        # The SMALLEST target that actually pays.
        #
        # "Clears the round trip" is necessary and not sufficient. The stop
        # cannot go below three spreads however small the target is, so a
        # target at twice the cost can still demand an 80% hit rate — which
        # is not a trade, it is a donation with good manners. So the search
        # walks outward one tick at a time and stops at the first target
        # whose REQUIRED hit rate is achievable. That target is the nearest
        # one worth taking, which is exactly what a scalper wants: the
        # highest hit rate available on a trade that pays.
        first = max(1, math.ceil(cost_bps * min_cost_multiple
                                 / 10_000.0 * entry / t))
        ceiling = max(first, int(available_bps * MAX_RANGE_SHARE
                                 / 10_000.0 * entry / t))
        steps = None
        for candidate in range(first, ceiling + 1):
            bps = candidate * t / entry * 10_000.0
            dist, capped = stop_for(bps)
            r_bps = dist / entry * 10_000.0
            if breakeven_hit_rate(bps, r_bps, cost_bps) <= max_breakeven:
                steps, risk_dist, scalp_stop_capped = candidate, dist, capped
                break

        if steps is None:
            # Nothing inside this candle's range demands a rate a book read
            # can reach. Almost always the spread: the trade has to pay for
            # it before it pays you.
            widest = ceiling * t / entry * 10_000.0
            dist, _ = stop_for(widest)
            need = breakeven_hit_rate(widest, dist / entry * 10_000.0, cost_bps)
            return NoTrade(
                "breakeven",
                f"nothing this {interval} can offer pays for a "
                f"{cost_bps:.1f}bps round trip against a "
                f"{hard_floor / entry * 10_000.0:.1f}bps minimum stop — even "
                f"the widest target it has room for ({widest:.1f}bps) would "
                f"need to be right {need:.0%} of the time, above the "
                f"{max_breakeven:.0%} ceiling. The read is {side}; the "
                f"spread is the problem",
                side=side, conviction=read.conviction)

    target_px = entry + steps * t if side == "long" else entry - steps * t
    if target_px <= 0:
        return NoTrade("no target", "the target lands at or below zero")
    target_bps = abs(target_px - entry) / entry * 10_000.0

    if mode == "range":
        risk_dist, _ = stop_for(target_bps)

    stop_px = (entry - risk_dist) if side == "long" else (entry + risk_dist)
    if stop_px <= 0:
        return NoTrade("no target", "the invalidation lands at or below zero")
    risk_bps = risk_dist / entry * 10_000.0
    rr = target_bps / risk_bps if risk_bps > 0 else 0.0

    # Does the candle have room for it?
    if target_bps > available_bps * MAX_RANGE_SHARE:
        return NoTrade(
            "no room",
            f"the smallest trade that pays for a {cost_bps:.1f}bps round trip "
            f"needs {target_bps:.1f}bps, and this {interval} typically has "
            f"about {available_bps:.0f}bps left in it. The read is {side}; "
            f"there is not enough candle left to cover the spread"
            if mode == "scalp" else
            f"a {target_bps:.0f}bps target is more than this {interval} "
            f"typically has left ({available_bps:.0f}bps)",
            side=side, conviction=read.conviction)

    if target_bps < cost_bps * min_cost_multiple:
        return NoTrade(
            "cost",
            f"the {interval} is only offering about {target_bps:.1f}bps from "
            f"here and the round trip costs {cost_bps:.1f}bps — that is "
            f"{target_bps / cost_bps:.1f}x, under the {min_cost_multiple:.0f}x "
            f"floor. The read is {side}, the spread just eats it",
            side=side, conviction=read.conviction)

    # -- is the shape worth it --------------------------------------------
    if mode == "scalp":
        # R:R is deliberately NOT the gate here: a small target always has a
        # poor one, and that is the same fact as its high hit rate counted
        # twice. The search above already guaranteed the breakeven; this is
        # the belt-and-braces check.
        need = breakeven_hit_rate(target_bps, risk_bps, cost_bps)
        if need > max_breakeven:
            return NoTrade(
                "breakeven",
                f"this would need to be right {need:.0%} of the time to pay "
                f"({target_bps:.1f}bps target, {risk_bps:.1f}bps risk, "
                f"{cost_bps:.1f}bps cost) — above the {max_breakeven:.0%} "
                f"ceiling, and no book read sustains that",
                side=side, conviction=read.conviction)
    elif rr < min_rr:
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
        spread_bps=read.spread_bps, mode=mode,
        measured_rate=measured_rate, measured_n=measured_n,
        confirmation=agreement,
        reasons=_reasons(read),
        cautions=_cautions(read, read.spread_bps, target_bps)
        + (["the invalidation is a scalp stop, not a structural level — the "
            "nearest shelf is further away than this trade can carry"]
           if scalp_stop_capped else []),
    )


# --------------------------------------------------------------------------
# the call on every candle
# --------------------------------------------------------------------------

# Grades. These describe the SHAPE of a call, never a prediction: an A is a
# trade that is cheap, well-supported and demands a hit rate a book read
# plausibly reaches. It is not a promise that it wins.
GRADES = (
    ("A", "cheap, well supported, and asks little of you"),
    ("B", "a fair trade — the numbers work with something to spare"),
    ("C", "marginal: it works, but not by much"),
    ("D", "readable but not tradeable"),
    ("X", "book and price disagree — somebody is absorbing"),
    ("—", "the book is not saying anything"),
)


@dataclass(frozen=True)
class Call:
    """What the book says about this candle, tradeable or not.

    `suggest` answers "is there a trade", and most of the time the answer is
    no. That is correct and it is also unreadable: a panel that goes blank
    cannot be told apart from a panel that is broken, and it hides the near
    misses -- which are the ones worth watching, because they are where the
    book is starting to lean.

    So this always returns something. `tradeable` is the honest flag, and
    only tradeable calls are ever recorded or scored. The rest are there to
    be looked at.
    """

    coin: str
    interval: str
    side: Side | None
    grade: str
    tradeable: bool
    detail: str
    conviction: float = 0.0
    score: float = 0.0
    samples: int = 0
    seconds_left: float = 0.0
    spread_bps: float = 0.0
    blocked_by: str | None = None
    suggestion: Suggestion | None = None
    # Carried whether or not the trade happened: "book says UP, price says
    # DOWN" is the most informative thing on the panel precisely on the
    # candles where there is no trade.
    confirmation: Confirmation | None = None
    action: CandleAction | None = None

    @property
    def grade_note(self) -> str:
        return dict(GRADES).get(self.grade, "")

    def sentence(self) -> str:
        if self.suggestion is not None:
            return self.suggestion.sentence()
        lean = (f"{self.side} lean" if self.side else "no lean")
        return (f"{self.grade} — {lean}, not tradeable. {self.detail}")

    def to_dict(self) -> dict:
        out = {
            "coin": self.coin, "interval": self.interval,
            "side": self.side, "grade": self.grade,
            "grade_note": self.grade_note,
            "tradeable": self.tradeable,
            "blocked_by": self.blocked_by,
            "detail": self.detail,
            "conviction": round(self.conviction, 4),
            "score": round(self.score, 4),
            "samples": self.samples,
            "seconds_left": round(self.seconds_left, 1),
            "spread_bps": round(self.spread_bps, 3),
            "sentence": self.sentence(),
        }
        out["suggestion"] = (self.suggestion.to_dict()
                             if self.suggestion else None)
        out["confirmation"] = (self.confirmation.to_dict()
                               if self.confirmation else None)
        out["action"] = None
        if self.action is not None:
            out["action"] = {
                "direction": self.action.direction,
                "strength": self.action.strength,
                "score": self.action.score,
                "thrust_bps": round(self.action.thrust_bps, 3),
                "position": round(self.action.position, 4),
                "slope_bps": round(self.action.slope_bps, 3),
                "extending": self.action.extending,
                "window_s": self.action.window_s,
                "components": self.action.components(),
                "describe": self.action.describe(),
            }
        return out


def _grade(s: Suggestion) -> str:
    """How good the SHAPE is. Never how likely it is to win.

    Three things, all measured: how far the target clears the cost, how much
    room there is under the breakeven ceiling, and how convinced the book
    is. A trade can be graded A and still lose; the grade says the numbers
    are not working against you before you start.
    """
    need = s.breakeven
    points = 0
    points += 2 if s.cost_multiple >= 4 else 1 if s.cost_multiple >= 2.5 else 0
    points += 2 if need <= 0.45 else 1 if need <= 0.58 else 0
    points += 2 if s.conviction >= 0.75 else 1 if s.conviction >= 0.6 else 0
    return "A" if points >= 5 else "B" if points >= 3 else "C"


def assess(read: BookRead | None, book: Book | None, **kw) -> Call:
    """A graded call on this candle, whether or not it is tradeable.

    Takes and forwards everything `suggest` takes. The difference is that
    this never declines to answer: a refusal comes back as an ungraded call
    naming the gate, so the panel always shows the state of the book rather
    than going blank and looking broken.
    """
    coin = kw.get("coin", "")
    interval = kw.get("interval", "")
    action = kw.get("action")
    out = suggest(read, book, **kw)

    conviction = read.conviction if read else 0.0
    score = read.score if read else 0.0
    samples = read.samples if read else 0
    spread = read.spread_bps if read else 0.0

    if isinstance(out, Suggestion):
        return Call(coin=coin, interval=interval, side=out.side,
                    grade=_grade(out), tradeable=True,
                    detail=out.headline(), conviction=conviction,
                    score=score, samples=samples,
                    seconds_left=kw.get("seconds_left", 0.0),
                    spread_bps=spread, suggestion=out,
                    confirmation=out.confirmation, action=action)

    # Not tradeable. Still report which way it leans and what price is
    # doing, because "the book says buy and price is falling" is the most
    # useful thing on the panel and it only ever appears on a candle with no
    # trade on it.
    side = out.side
    agreement = None
    if read is not None and (action is not None
                             or kw.get("require_confirmation", True)):
        agreement = confirm(read.direction, read.conviction, action,
                            held_s=kw.get("held_s", 0.0),
                            flips=kw.get("flips", 0),
                            participation=kw.get("participation"))

    grade = "D" if side else "—"
    if out.gate == "conflict":
        grade = "X"
    return Call(coin=coin, interval=interval, side=side, grade=grade,
                tradeable=False, detail=out.detail, conviction=conviction,
                score=score, samples=samples,
                seconds_left=kw.get("seconds_left", 0.0),
                spread_bps=spread, blocked_by=out.gate,
                confirmation=agreement, action=action)


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
