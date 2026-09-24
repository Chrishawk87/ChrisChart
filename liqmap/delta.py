"""True delta: who is crossing the spread, how hard, and is it working.

THE THIRD INDEPENDENT COLUMN

    BOOK    what is RESTING. Limit orders. Intent that can be withdrawn.
    DELTA   who is AGGRESSING. Market orders. Intent already paid for.
    PRICE   whether it WORKED.

Those are three genuinely different things and they come apart constantly.
The book can be stacked on the bid while every print is hitting it. Delta
can be strongly negative while price refuses to fall. Each of those gaps is
information, and a system that folds them into one score destroys exactly
the information it most needs.

WHY "TRUE" DELTA

Most platforms infer the aggressor with a tick rule -- if the print is above
the last one, call it a buy -- and get it wrong on a meaningful share of
trades, worst precisely when the market is fast and the answer matters.
Hyperliquid publishes the taker side on every fill, so this is classified
rather than guessed. It is a real advantage and it costs nothing.

ABSORPTION LIVES HERE, NOT IN THE BOOK

`bookread.py` used to own absorption, comparing tape aggression against the
book's own mid. That was the wrong home twice over: it made the book module
depend on the tape, and it made the book side of the confirmation share an
input with the price side.

Absorption is simply delta saying one thing and price saying another --
heavy selling, price will not drop. That is a delta/price divergence and it
belongs in the module that owns delta.

PERSISTENCE IS NOT SIZE

The same net delta arrives two ways, and they mean opposite things for how
long to stay in a trade:

    one enormous print, then nothing
        already over. Whoever wanted size has it. Scalp it.

    a steady accumulation over minutes
        somebody is working an order and is not finished. Hold.

`persistence` separates them by asking how monotonically cumulative delta
moved, rather than how far. It is the difference between a 4-tick scalp and
a trade worth holding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

Direction = Literal["up", "down", "flat"]

# Below this the tape is balanced and saying nothing about direction.
FLAT_BAND = 0.15

# Hysteresis, matching `confirm.py`: a direction is harder to leave than it
# was to enter, so a lean hovering at the threshold does not chatter.
EXIT_BAND = 0.08

# Delta is scaled against what this market normally trades in the window, so
# thresholds mean the same thing on BTC and on a thin HIP-3 perp.
FULL_EFFORT = 1.5

# Price move, in basis points, below which a market is "not moving" for the
# purposes of calling absorption. Scaled by the instrument's own range.
STALL_FRACTION = 0.1

# How one-directional cumulative delta has to be before it reads as somebody
# working an order rather than noise arriving in both directions.
PERSISTENT = 0.6


@dataclass
class DeltaRead:
    """What the tape is doing, independent of the book entirely."""

    delta: float             # signed notional: + aggressive buying
    volume: float            # total notional both ways
    expected: float          # what this market normally does in this window
    price_move_bps: float    # what price did over the same window
    bar_range_bps: float     # for scaling the move
    persistence: float       # -1..+1, how monotonic the accumulation was
    trades: int = 0
    window_s: float = 30.0
    notes: list[str] = field(default_factory=list)

    @property
    def lean(self) -> float:
        """-1 selling, +1 buying. Share of volume that was one-sided."""
        return (self.delta / self.volume) if self.volume > 0 else 0.0

    @property
    def effort(self) -> float:
        """Volume against normal for this span. 1.0 is an ordinary window."""
        return (self.volume / self.expected) if self.expected > 0 else 0.0

    @property
    def score(self) -> float:
        """-1..+1. Lean, scaled by whether anyone actually showed up.

        A 90% one-sided lean on a tenth of normal volume is three prints
        agreeing with each other, not pressure. Multiplying by effort stops
        a quiet tape producing a confident reading.
        """
        weight = min(self.effort / FULL_EFFORT, 1.0)
        return round(max(-1.0, min(1.0, self.lean * weight)), 4)

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

    # -- the divergences --------------------------------------------------

    @property
    def stalled(self) -> bool:
        """Price is not moving, whatever the tape is doing."""
        limit = max(self.bar_range_bps * STALL_FRACTION, 0.5)
        return abs(self.price_move_bps) < limit

    @property
    def absorbed(self) -> bool:
        """Real aggression, going nowhere.

        Somebody is crossing the spread in size and price will not move,
        which means somebody else is filling every one of those orders
        passively without stepping back. The aggressor is paying and not
        being paid, and that is the single most expensive thing to be on the
        wrong side of.
        """
        return bool(abs(self.lean) > 0.25 and self.effort >= 0.8
                    and self.stalled)

    @property
    def efficient(self) -> float:
        """Basis points of price move per unit of normal-sized flow.

        High means price is moving easily -- thin resistance, a little
        pressure goes a long way. Low means it is grinding, which is what
        absorption looks like before it becomes obvious.
        """
        if self.effort <= 0:
            return 0.0
        return round(self.price_move_bps / self.effort, 3)

    @property
    def divergent(self) -> bool:
        """Delta and price pointing opposite ways with both meaning it."""
        if self.direction == "flat" or self.stalled:
            return False
        want = 1.0 if self.direction == "up" else -1.0
        return (self.price_move_bps * want) < 0

    @property
    def working(self) -> bool:
        """Somebody is working size rather than firing one print."""
        return abs(self.persistence) >= PERSISTENT and self.effort >= 0.8

    def supports(self, direction: Direction) -> bool:
        if direction == "flat":
            return False
        want = 1.0 if direction == "up" else -1.0
        return self.score * want > FLAT_BAND

    def describe(self) -> str:
        if self.volume <= 0:
            return "no fills in the window — nobody is trading"
        if self.effort < 0.3:
            return (f"only {self.effort:.0%} of normal volume; the tape is "
                    f"quiet and its lean means little")

        side = "buyers" if self.lean > 0 else "sellers"
        head = (f"{side} are crossing ({abs(self.lean):.0%} one-sided on "
                f"{self.effort:.0%} of normal volume)")

        if self.absorbed:
            other = "sellers" if self.lean > 0 else "buyers"
            return (f"{head} and price has not moved — {other} are absorbing "
                    f"them, filling every order without stepping back")
        if self.divergent:
            return (f"{head} but price is going the other way — they are "
                    f"being run over")
        if self.working:
            return (f"{head}, steadily rather than in one print — somebody is "
                    f"working an order and is not finished")
        return head + f", price {self.price_move_bps:+.1f}bps"

    def to_dict(self) -> dict:
        return {
            "direction": self.direction, "score": self.score,
            "strength": self.strength, "lean": round(self.lean, 4),
            "effort": round(self.effort, 3),
            "delta": round(self.delta, 2), "volume": round(self.volume, 2),
            "persistence": round(self.persistence, 3),
            "price_move_bps": round(self.price_move_bps, 2),
            "absorbed": self.absorbed, "divergent": self.divergent,
            "working": self.working, "stalled": self.stalled,
            "efficient": self.efficient, "trades": self.trades,
            "window_s": self.window_s, "describe": self.describe(),
        }


def persistence_of(signed: Sequence[float]) -> float:
    """How one-directional a run of signed flow was, -1..+1.

    Net delta over a window says how much; this says whether it arrived as
    one shove or as a steady push. The ratio of the net to the total
    absolute movement: a perfectly monotonic accumulation scores 1, flow
    arriving equally in both directions scores 0.

    That distinction is the difference between a trade worth holding and one
    worth taking four ticks out of.
    """
    if not signed:
        return 0.0
    total = sum(abs(x) for x in signed)
    if total <= 0:
        return 0.0
    return round(max(-1.0, min(1.0, sum(signed) / total)), 4)


def sticky(score: float, previous: Direction | None,
           enter: float = FLAT_BAND, leave: float = EXIT_BAND) -> Direction:
    """Same hysteresis as the other columns, so all three chatter alike —
    which is to say, not at all."""
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


def read_delta(delta: float, volume: float, expected: float,
               price_move_bps: float, bar_range_bps: float,
               buckets: Sequence[float] = (), trades: int = 0,
               window_s: float = 30.0) -> DeltaRead | None:
    """Build a reading. `buckets` are signed sub-window deltas, oldest first,
    used only to judge persistence."""
    if volume <= 0 or bar_range_bps <= 0:
        return None
    return DeltaRead(
        delta=delta, volume=volume, expected=max(expected, 0.0),
        price_move_bps=price_move_bps, bar_range_bps=bar_range_bps,
        persistence=persistence_of(buckets), trades=trades,
        window_s=window_s)


def from_feed(feed, interval: str, interval_s: float,
              window_s: float = 30.0, slices: int = 6) -> DeltaRead | None:
    """Read the tape from a live feed. Never touches the order book.

    Persistence is measured by cutting the window into slices and asking how
    consistently delta pointed the same way across them, which is what
    separates an accumulator from a single print.
    """
    from .confirm import typical_bar_bps

    try:
        window = feed.tape.window(window_s)
    except Exception:
        return None
    if window is None or window.total <= 0:
        return None

    bars = feed.history(interval)
    bar_bps = typical_bar_bps(bars)
    if bar_bps <= 0:
        live = feed.candle(interval)
        if live is not None:
            mid = (live.high + live.low) / 2.0
            bar_bps = ((live.high - live.low) / mid * 10_000.0) if mid > 0 else 0.0
    if bar_bps <= 0:
        return None

    # What this market normally trades in a window this long.
    notionals = []
    for c in bars[-20:]:
        n = getattr(c, "buy_notional", 0.0) + getattr(c, "sell_notional", 0.0)
        if n <= 0:
            mid = (c.high + c.low) / 2.0
            n = c.volume * mid if mid > 0 else 0.0
        if n > 0:
            notionals.append(n)
    expected = 0.0
    if notionals and interval_s > 0:
        notionals.sort()
        m = len(notionals)
        typical = (notionals[m // 2] if m % 2
                   else (notionals[m // 2 - 1] + notionals[m // 2]) / 2)
        expected = typical * (window_s / interval_s)

    # Slice the window to judge persistence.
    step = window_s / max(slices, 1)
    buckets: list[float] = []
    try:
        prev_total = 0.0
        for i in range(1, slices + 1):
            w = feed.tape.window(step * i)
            if w is None:
                break
            buckets.append(w.delta - prev_total)
            prev_total = w.delta
    except Exception:
        buckets = []

    try:
        move = feed.tape.price_change_bps(window_s)
    except Exception:
        move = 0.0

    return read_delta(
        delta=window.delta, volume=window.total, expected=expected,
        price_move_bps=move, bar_range_bps=bar_bps,
        buckets=list(reversed(buckets)), trades=getattr(window, "trades", 0),
        window_s=window_s)
