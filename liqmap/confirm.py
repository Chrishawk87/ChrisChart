"""The book says a direction. Is the candle actually doing it?

THE PROBLEM THIS FIXES

`bookread.py` calls the candle from the order book alone. That is the right
input -- the book is the mechanism by which price moves -- but on its own it
answers the wrong question. It says where the PRESSURE is, not whether the
pressure is WINNING.

Those come apart constantly, and the gap between them is where money is lost.
A book can show ten million on the bid against one million on the offer, the
offers thinning, bids replacing everything that gets hit -- every reading a
buyer could want -- while price goes nowhere or drifts down. That is not a
buy signal that happens to be early. It is a seller quietly filling into
every bid without moving the price, and the book is the last place it shows
up. Trading the book without checking price is trading the losing side of an
absorption.

So this module measures what the candle is actually doing, compares it
against what the book claims, and only calls a trade when the two agree.

    book UP    + candle UP    -> CONFIRMED. Pressure is winning. Take it.
    book UP    + candle DOWN  -> CONFLICT. Buyers are being absorbed.
    book UP    + candle FLAT  -> UNCONFIRMED. Pressure has not paid yet.
    book FLAT                 -> nothing to confirm.

Confirmation is a HARD gate, not a weight. Blending a book score with a price
score averages a disagreement into a weak agreement, which is the single most
expensive thing this design could do: it would produce a small long exactly
where the passive seller is winning. Disagreement means no trade.

THE COMPARISON MUST NOT BE CIRCULAR

This is the part that is easy to get wrong and impossible to notice
afterwards. `BookRead` already contains `mid_drift_bps` -- what the mid did
over its window -- and that is part of its own score. If the candle side of
this comparison also used the book's mid, the two sides would share an input
and would agree with themselves. The check would pass constantly and mean
nothing.

So everything here comes from TRADES and BARS: the candle's open, high, low
and last, and the price change on the tape. Those are fills that actually
happened, at prices somebody actually paid. Nothing in this module reads the
order book.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

from .structure import Candle

Direction = Literal["up", "down", "flat"]

# Below this the candle is not doing anything worth calling a direction.
FLAT_BAND = 0.15

# Thrust is measured against the instrument's own typical bar range rather
# than a fixed basis-point figure, so the same code reads a $4 coin and a
# $100k one. This is the share of a typical bar that counts as a full-
# strength move.
THRUST_FULL = 0.35

# How far into the bar's own range price has to sit before position counts
# as directional. 0.5 is the middle; this is the distance either side.
POSITION_BAND = 0.15

# Hysteresis. A reading sitting near the flat boundary flips between "down"
# and "flat" on changes far too small to be a market event, and every flip
# breaks an agreement that was never really broken. So a direction is harder
# to LEAVE than it was to enter: cross FLAT_BAND to take a side, fall below
# EXIT_BAND to give it up.
EXIT_BAND = 0.08

# How long book and price must agree before the agreement is worth acting
# on. An agreement that appeared on this tick and one that has held for ten
# seconds are not the same thing, and treating them alike is why a
# confirmation can evaporate between seeing it and clicking.
MIN_AGREEMENT_S = 4.0

# Above this many direction changes in the recent window, the market is
# chopping and no reading of it is stable enough to trade.
MAX_FLIPS = 4
FLIP_WINDOW_S = 60.0

# Aggressive volume in the window, as a multiple of what this market
# normally does in that much time. Below this, whatever moved price was not
# paid for.
MIN_PARTICIPATION = 0.6


@dataclass
class CandleAction:
    """What price is doing right now, from fills only.

    Four independent readings, none of which touch the order book:

        thrust      where last sits against the bar's OPEN. The bar's own
                    body -- green or red, and by how much.
        position    where last sits between the bar's low and high. Closing
                    on the highs is a different state from closing mid-range
                    even when both bars are green.
        slope       what the tape has done over the confirmation window.
                    Shorter than the bar, because a scalp cares about the
                    last thirty seconds, not the last fifteen minutes.
        extending   whether the recent window set a new extreme. Price making
                    new highs is doing something price grinding sideways
                    under an old high is not.
    """

    thrust_bps: float
    position: float            # 0..1 within the bar's range
    slope_bps: float           # over `window_s`
    extending: Direction
    bar_range_bps: float       # typical, for scale
    window_s: float = 30.0
    notes: list[str] = field(default_factory=list)

    # Slope leads: it is the most recent evidence and the least laggy.
    # Thrust and position describe the bar as a whole, which matters but is
    # older. Extending is a confirmation of the other three rather than an
    # independent claim, so it carries least.
    W = {"slope": 1.0, "thrust": 0.8, "position": 0.6, "extending": 0.4}

    def _scaled(self, bps: float) -> float:
        """A move, as a share of a full-strength move for this instrument."""
        full = max(self.bar_range_bps * THRUST_FULL, 1e-9)
        return max(-1.0, min(1.0, bps / full))

    @property
    def parts(self) -> dict[str, float]:
        pos = 0.0
        if self.position > 0.5 + POSITION_BAND:
            pos = min((self.position - 0.5 - POSITION_BAND) / (0.5 - POSITION_BAND), 1.0)
        elif self.position < 0.5 - POSITION_BAND:
            pos = -min((0.5 - POSITION_BAND - self.position) / (0.5 - POSITION_BAND), 1.0)

        return {
            "slope": self._scaled(self.slope_bps),
            "thrust": self._scaled(self.thrust_bps),
            "position": pos,
            "extending": (1.0 if self.extending == "up"
                          else -1.0 if self.extending == "down" else 0.0),
        }

    @property
    def score(self) -> float:
        """-1 falling, +1 rising. What price is DOING, not what it should do."""
        p = self.parts
        total = sum(self.W.values())
        return round(sum(v * self.W[k] for k, v in p.items()) / total, 4)

    @property
    def direction(self) -> Direction:
        s = self.score
        if s > FLAT_BAND:
            return "up"
        if s < -FLAT_BAND:
            return "down"
        return "flat"

    @property
    def strength(self) -> float:
        return round(min(abs(self.score) / 0.5, 1.0), 4)

    def components(self) -> list[dict]:
        p = self.parts
        return [
            {"name": "tape slope", "value": p["slope"], "weight": self.W["slope"],
             "note": f"price {self.slope_bps:+.2f}bps over {self.window_s:.0f}s"},
            {"name": "bar body", "value": p["thrust"], "weight": self.W["thrust"],
             "note": (f"{'green' if self.thrust_bps > 0 else 'red' if self.thrust_bps < 0 else 'flat'}"
                      f" — {self.thrust_bps:+.2f}bps from the open")},
            {"name": "position in bar", "value": p["position"],
             "weight": self.W["position"],
             "note": f"{self.position:.0%} of the way up the bar's range"},
            {"name": "extending", "value": p["extending"],
             "weight": self.W["extending"],
             "note": ("making new highs" if self.extending == "up" else
                      "making new lows" if self.extending == "down" else
                      "inside the recent range")},
        ]

    def describe(self) -> str:
        if self.direction == "flat":
            return (f"Price is going nowhere — {self.slope_bps:+.2f}bps over "
                    f"{self.window_s:.0f}s, sitting {self.position:.0%} up "
                    f"the bar.")
        word = "rising" if self.direction == "up" else "falling"
        return (f"Price is {word}: {self.slope_bps:+.2f}bps over "
                f"{self.window_s:.0f}s, {self.thrust_bps:+.2f}bps from the "
                f"open, {self.position:.0%} up the bar's range"
                + (f", making new {'highs' if self.extending == 'up' else 'lows'}"
                   if self.extending != "flat" else "") + ".")


@dataclass
class Participation:
    """Whether the move was PAID FOR.

    This is the piece that decides whether an agreement between the book and
    price action holds together or evaporates, and it is the answer to "what
    holds those two together".

    Book says down, price says down. Two completely different worlds produce
    that reading:

        bids were HIT -- aggressive sellers paying the spread, volume well
        above normal. Somebody with size is working an order and they are
        not finished. The agreement persists because the cause persists.

        bids were PULLED -- quotes withdrawn, price drifted down on almost
        no volume. Nothing was paid for, nobody is positioned, and the next
        quote refresh undoes it. The agreement evaporates within seconds by
        construction.

    Both look identical to price and to the book. Only volume separates
    them, which is why a confirmation rule without this catches vacuum moves
    and real moves in the same net.

    `effort` is aggressive volume against what this market normally trades
    in the same span; `aligned` is whether that volume was on the side the
    call is taking.
    """

    effort: float          # 1.0 == a normal amount of volume for this span
    aligned: float         # -1..+1, signed with the aggressor lean
    notional: float
    window_s: float

    @property
    def backed(self) -> bool:
        return self.effort >= MIN_PARTICIPATION

    def supports(self, direction: Direction) -> bool:
        """Does the flow back a call in this direction?"""
        if direction == "flat" or not self.backed:
            return False
        want = 1.0 if direction == "up" else -1.0
        return self.aligned * want > 0.15

    def describe(self, direction: Direction = "flat") -> str:
        if not self.backed:
            return (f"only {self.effort:.0%} of normal volume — whatever "
                    f"moved price was not paid for, so there is nothing "
                    f"holding this together")
        side = "buyers" if self.aligned > 0 else "sellers"
        agree = self.supports(direction)
        return (f"{self.effort:.0%} of normal volume and {side} are the ones "
                f"paying for it" + ("" if agree or direction == "flat"
                                    else " — which is the wrong side for "
                                         "this call"))

    def to_dict(self) -> dict:
        return {"effort": round(self.effort, 3),
                "aligned": round(self.aligned, 3),
                "notional": round(self.notional, 2),
                "backed": self.backed, "window_s": self.window_s}


def sticky(score: float, previous: Direction | None,
           enter: float = FLAT_BAND, leave: float = EXIT_BAND) -> Direction:
    """Classify a score into a direction, but make it hard to give one up.

    Without hysteresis a score hovering at 0.14 against a 0.15 threshold
    reports down, flat, down, flat on noise that is not a market event —
    and each of those flips breaks an agreement that never actually broke.
    """
    if previous == "up":
        return "up" if score > leave else _fresh(score, enter)
    if previous == "down":
        return "down" if score < -leave else _fresh(score, enter)
    return _fresh(score, enter)


def _fresh(score: float, enter: float) -> Direction:
    if score > enter:
        return "up"
    if score < -enter:
        return "down"
    return "flat"


def typical_bar_bps(bars: Sequence[Candle], n: int = 20) -> float:
    """Median bar range in basis points, for scaling. 0.0 when unmeasurable."""
    out = []
    for c in bars[-n:]:
        mid = (c.high + c.low) / 2.0
        if mid > 0 and c.high >= c.low:
            out.append((c.high - c.low) / mid * 10_000.0)
    if not out:
        return 0.0
    out.sort()
    m = len(out)
    return out[m // 2] if m % 2 else (out[m // 2 - 1] + out[m // 2]) / 2.0


def read_candle(open_px: float, high_px: float, low_px: float, last_px: float,
                slope_bps: float, bar_range_bps: float,
                recent_highs: Sequence[float] = (),
                recent_lows: Sequence[float] = (),
                window_s: float = 30.0) -> CandleAction | None:
    """What the candle is doing, from bar geometry and the tape.

    `slope_bps` is the price change on the TAPE over `window_s` -- fills, not
    quotes. `recent_highs` / `recent_lows` are the prior sub-bars used to
    decide whether the current move is extending or just filling in an old
    range.
    """
    if last_px <= 0 or open_px <= 0 or high_px < low_px:
        return None
    if bar_range_bps <= 0:
        return None

    thrust_bps = (last_px - open_px) / open_px * 10_000.0

    span = high_px - low_px
    position = 0.5 if span <= 0 else (last_px - low_px) / span

    extending: Direction = "flat"
    if recent_highs and last_px > max(recent_highs):
        extending = "up"
    elif recent_lows and last_px < min(recent_lows):
        extending = "down"

    return CandleAction(thrust_bps=thrust_bps, position=position,
                        slope_bps=slope_bps, extending=extending,
                        bar_range_bps=bar_range_bps, window_s=window_s)


# --------------------------------------------------------------------------
# the comparison
# --------------------------------------------------------------------------

Verdict = Literal["confirmed", "conflict", "unconfirmed", "no signal"]


@dataclass(frozen=True)
class Confirmation:
    """Book against candle, and whether to act."""

    book: Direction
    candle: Direction
    book_strength: float
    candle_strength: float
    verdict: Verdict
    detail: str

    # Stability, from the tracker. Defaults keep every existing caller
    # working; a caller that does not supply them simply gets no stability
    # gate rather than a fabricated one.
    held_s: float = 0.0
    flips: int = 0
    participation: Participation | None = None

    @property
    def agree(self) -> bool:
        return self.verdict == "confirmed"

    @property
    def settled(self) -> bool:
        """Agreeing, AND has been for long enough, AND is not chopping.

        The distinction that matters for acting: `agree` is a snapshot,
        `settled` is a state you can still be in a few seconds from now.
        """
        return (self.agree
                and self.held_s >= MIN_AGREEMENT_S
                and self.flips <= MAX_FLIPS)

    @property
    def backed(self) -> bool:
        """Settled and paid for. The full bar for a scalp."""
        if not self.settled:
            return False
        p = self.participation
        return p is None or p.supports(self.book)

    def instability(self) -> str:
        """Why a real agreement is still not tradeable, in plain words."""
        if not self.agree:
            return ""
        if self.flips > MAX_FLIPS:
            return (f"but the read has changed {self.flips} times in the last "
                    f"minute — this market is chopping and nothing read from "
                    f"it will hold")
        if self.held_s < MIN_AGREEMENT_S:
            return (f"but they have only agreed for {self.held_s:.1f}s "
                    f"(needs {MIN_AGREEMENT_S:.0f}s) — a fresh agreement is "
                    f"the one that evaporates before you can act")
        p = self.participation
        if p is not None and not p.supports(self.book):
            return "but " + p.describe(self.book)
        return ""

    @property
    def direction(self) -> Direction:
        """The tradeable direction, or flat when there isn't one."""
        return self.book if self.agree else "flat"

    @property
    def strength(self) -> float:
        """Combined, and deliberately the WEAKER of the two.

        A powerful book and a barely-moving price is not a strong setup, it
        is a warning. Taking the minimum means the pair is only as good as
        its weaker half, which is the honest reading.
        """
        if not self.agree:
            return 0.0
        return round(min(self.book_strength, self.candle_strength), 4)

    def to_dict(self) -> dict:
        return {"book": self.book, "candle": self.candle,
                "book_strength": round(self.book_strength, 4),
                "candle_strength": round(self.candle_strength, 4),
                "verdict": self.verdict, "agree": self.agree,
                "settled": self.settled, "backed": self.backed,
                "held_s": round(self.held_s, 1), "flips": self.flips,
                "instability": self.instability(),
                "participation": (self.participation.to_dict()
                                  if self.participation else None),
                "direction": self.direction,
                "strength": self.strength, "detail": self.detail}


class AgreementTracker:
    """Memory for one market and timeframe, so agreement has an AGE.

    `confirm()` answers "do they agree at this instant". That is a snapshot,
    and a snapshot cannot tell a confirmation that has held for ten seconds
    from one that appeared on this tick — which is exactly why a call can
    evaporate between seeing it and acting on it.

    This holds three things a single reading cannot:

        held_s   how long the current state has been unbroken. A young
                 agreement is the one that dies.
        flips    how often the state has changed recently. A market
                 producing six direction changes a minute is chopping, and
                 no reading of it is stable enough to trade.
        sticky   the previous direction of each side, so hysteresis has
                 something to be sticky about.
    """

    def __init__(self, flip_window_s: float = FLIP_WINDOW_S):
        self.flip_window_s = flip_window_s
        self.book_dir: Direction | None = None
        self.price_dir: Direction | None = None
        self._state: tuple[str, str] | None = None
        self._since: float | None = None
        self._changes: list[float] = []

    def classify(self, book_score: float, price_score: float
                 ) -> tuple[Direction, Direction]:
        """Both sides, with hysteresis applied to each."""
        self.book_dir = sticky(book_score, self.book_dir)
        self.price_dir = sticky(price_score, self.price_dir)
        return self.book_dir, self.price_dir

    def observe(self, verdict: str, direction: Direction,
                now: float) -> tuple[float, int]:
        """Record the current state. Returns (seconds held, recent flips)."""
        key = (verdict, direction)
        if key != self._state:
            if self._state is not None:
                self._changes.append(now)
            self._state = key
            self._since = now

        cutoff = now - self.flip_window_s
        self._changes = [t for t in self._changes if t >= cutoff]
        held = (now - self._since) if self._since is not None else 0.0
        return max(0.0, held), len(self._changes)

    def reset(self) -> None:
        self.book_dir = self.price_dir = None
        self._state = None
        self._since = None
        self._changes.clear()


def confirm(book_dir: Direction, book_strength: float,
            action: CandleAction | None,
            held_s: float = 0.0, flips: int = 0,
            participation: Participation | None = None) -> Confirmation:
    """Does price agree with the book?

    The four states are not symmetric and the language matters, because
    CONFLICT is the one that costs money. "The book says buy but price is
    falling" is not a weak buy. It is somebody selling into every bid without
    moving the price, and the book is the last place that becomes visible.
    """
    extra = dict(held_s=held_s, flips=flips, participation=participation)

    if action is None:
        return Confirmation(book_dir, "flat", book_strength, 0.0,
                            "unconfirmed",
                            "no price action to check the book against — "
                            "the candle has not moved enough to read",
                            **extra)

    candle_dir = action.direction
    cs = action.strength

    if book_dir == "flat":
        return Confirmation(book_dir, candle_dir, book_strength, cs,
                            "no signal",
                            f"the book is balanced. {action.describe()}",
                            **extra)

    if candle_dir == "flat":
        side = "buyers" if book_dir == "up" else "sellers"
        return Confirmation(
            book_dir, candle_dir, book_strength, cs, "unconfirmed",
            f"the book favours {side} but price is not moving with them yet. "
            f"{action.describe()} Nothing to trade until it does", **extra)

    if candle_dir == book_dir:
        side = "buyers" if book_dir == "up" else "sellers"
        return Confirmation(
            book_dir, candle_dir, book_strength, cs, "confirmed",
            f"the book favours {side} and price is going with them. "
            f"{action.describe()}", **extra)

    # The expensive case.
    winning = "buyers" if book_dir == "up" else "sellers"
    losing = "sellers" if book_dir == "up" else "buyers"
    way = "down" if book_dir == "up" else "up"
    return Confirmation(
        book_dir, candle_dir, book_strength, cs, "conflict",
        f"the book favours {winning} but price is going {way}. That is "
        f"{losing} absorbing them — filling every order without letting "
        f"price move. Standing aside; this is where trading the book alone "
        f"loses money. {action.describe()}", **extra)


def participation_from_feed(feed, interval: str, interval_s: float,
                            window_s: float = 30.0) -> Participation | None:
    """Was the recent move paid for?

    `effort` compares aggressive volume in the window against what this
    market normally trades in the same span, taken from the median of recent
    closed bars. A ratio, not an absolute, so the same threshold means
    something on BTC and on a thin HIP-3 perp.
    """
    try:
        w = feed.tape.window(window_s)
    except Exception:
        return None
    if w is None:
        return None

    bars = feed.history(interval)
    notionals = []
    for c in bars[-20:]:
        n = getattr(c, "buy_notional", 0.0) + getattr(c, "sell_notional", 0.0)
        if n <= 0:
            # Closed `Candle` objects carry size, not notional. Price it at
            # the bar's own midpoint rather than skipping the bar, which
            # would make a quiet market look like a busy one.
            mid = (c.high + c.low) / 2.0
            n = c.volume * mid if mid > 0 else 0.0
        if n > 0:
            notionals.append(n)

    if not notionals or interval_s <= 0:
        return None
    notionals.sort()
    m = len(notionals)
    typical_bar = (notionals[m // 2] if m % 2
                   else (notionals[m // 2 - 1] + notionals[m // 2]) / 2)
    expected = typical_bar * (window_s / interval_s)
    if expected <= 0:
        return None

    return Participation(effort=w.total / expected,
                         aligned=w.lean, notional=w.total,
                         window_s=window_s)


def from_feed(feed, interval: str, window_s: float = 30.0
              ) -> CandleAction | None:
    """Build a `CandleAction` from a live feed, using fills only.

    Deliberately never touches `feed.book`. The whole value of this check is
    that it is an independent opinion, and an independent opinion that quietly
    shares an input with the thing it is checking is worth nothing.
    """
    bar = feed.candle(interval)
    if bar is None:
        return None

    history = feed.history(interval)
    bar_bps = typical_bar_bps(history)
    if bar_bps <= 0:
        # Fall back to this bar's own range, which is thin evidence but
        # better than refusing to read a market that has only just started.
        mid = (bar.high + bar.low) / 2.0
        bar_bps = ((bar.high - bar.low) / mid * 10_000.0) if mid > 0 else 0.0
    if bar_bps <= 0:
        return None

    try:
        slope = feed.tape.price_change_bps(window_s)
    except Exception:
        slope = 0.0

    recent = history[-5:]
    return read_candle(
        open_px=bar.open, high_px=bar.high, low_px=bar.low, last_px=bar.close,
        slope_bps=slope, bar_range_bps=bar_bps,
        recent_highs=[c.high for c in recent],
        recent_lows=[c.low for c in recent],
        window_s=window_s)
