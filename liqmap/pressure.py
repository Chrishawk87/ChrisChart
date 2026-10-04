"""Who is winning, measured from the book and the tape.

THIS IS NOT A PREDICTION, AND THAT DISTINCTION IS THE WHOLE POINT

The candle read blended eight things -- some of them observations, some of
them priors about what price tends to do next -- and averaged them into one
number. That was the wrong shape. A prior about mean reversion has no
business diluting a direct measurement of who is currently winning a fight
you can watch happening.

So this module measures one thing and refuses to guess:

    Aggressors are whoever is crossing the spread. The WINNER is whoever is
    getting the price they want.

That gives four states, and only four:

    buyers aggressing,  price rising    -> BUYERS winning
    buyers aggressing,  price flat/down -> SELLERS winning (passive absorption)
    sellers aggressing, price falling   -> SELLERS winning
    sellers aggressing, price flat/up   -> BUYERS winning (passive absorption)

The two absorption cases are the ones worth trading and the ones a naive
"lots of buying, must be bullish" reading gets backwards. Aggressors pay the
spread and take the worse price; if they are not being rewarded, somebody is
quietly taking the other side in size, and that somebody is winning.

MULTI-TIMEFRAME IS A CONFRONTATION, NOT AN AVERAGE

The same measurement runs on every timeframe at once and the results are set
against each other rather than blended. A 4-hour where sellers are winning
and a 15-minute where buyers are winning is not "neutral" -- it is a specific
and tradeable situation, and averaging it to zero destroys exactly the
information you wanted. `Confrontation` keeps them apart and names the shape.

WHAT IS MEASURED VERSUS WHAT IS INFERRED

With a live tape, the buy/sell split is counted from actual fills. Without
one, it can only be inferred from where each candle closed within its own
range, which is weaker and is labelled as such on every reading
(`Pressure.measured`). The two are never silently mixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

from .flow import Book
from .structure import Candle

Side = Literal["buyers", "sellers", "contested"]
Aggressor = Literal["buy", "sell", "balanced"]

# Below this share of one-sided aggression, nobody is really pressing.
BALANCED_BAND = 0.10

# "PRICE DID NOT GO ANYWHERE" IS NOT A FIXED NUMBER OF BASIS POINTS
#
# This was a flat 2bps on every timeframe in every market, and that single
# constant inverted most of the ladder. A four-hour bar that moved 2bps has
# genuinely gone nowhere. A one-minute bar that moved 2bps has had a large
# minute. Judging both by the same number means nearly every fast row comes
# back "flat", and a flat row with one side aggressing is reported as the
# OTHER side winning by absorption -- so the minute reads UP while price is
# coming down, which is exactly the complaint.
#
# So the band is relative to what bars of this length actually do in this
# market. `typical_bps` is the median absolute move of recent completed
# bars; a move well inside that has not gone anywhere, and a move outside it
# has. When there are no recent bars to learn from, the reference below is
# scaled by the square root of the interval, which is how a random walk's
# range grows -- a rough prior, and better than one number for everything.
# The prior is stated as a TYPICAL MOVE rather than as a band, so that the
# learned path and the fallback path are the same calculation with a
# different input. The old constant was a band, which made the two
# inconsistent: a measured 4-hour band came out near 9bps while the
# constant insisted on 2.
#
# 25bps over four hours is a rough middle across liquid markets and it is a
# placeholder, not a measurement. Every reading carries `learned`, so a row
# standing on the prior can say so instead of being mistaken for one that
# counted.
PRIOR_TYPICAL_BPS = 25.0  # median absolute move of a bar at the reference ...
FLAT_REF_S = 14400.0      # ... which is this long
FLAT_SHARE = 0.35         # share of a typical move that still counts as flat
MIN_FLAT_BPS = 0.05       # below this it is quoting noise as a threshold

# ABSORPTION HAS TO HAVE SOMETHING TO ABSORB
#
# The inverting branch says: this side is paying the spread and not being
# rewarded, so the other side is quietly taking it. That reading is only
# worth anything when there is real size being taken. A bar with four fills
# in it that happen to be 60% one-sided has not been absorbed, it has been
# quiet -- and calling that absorption hands the row a confident, inverted
# direction built on nothing.
PRESSING_SHARE = 0.40     # of what this timeframe normally trades by now
MIN_TRADES = 3            # fills below which a measured bar says nothing


@dataclass
class Pressure:
    """One timeframe's answer to: who is winning right now."""

    timeframe: str
    interval_s: float

    open_px: float
    last_px: float
    high_px: float
    low_px: float

    buy_notional: float
    sell_notional: float
    measured: bool            # True = counted from fills, False = inferred
    trades: int = 0

    bid_depth: float = 0.0
    ask_depth: float = 0.0

    elapsed_s: float = 0.0
    coverage: float = 1.0     # share of the bar the tape actually saw

    # What bars of this length normally do in this market, learned from the
    # recent completed ones. Zero means nothing was available to learn from
    # and the fallbacks below take over.
    typical_bps: float = 0.0
    typical_notional: float = 0.0

    # Whether this bar's OPEN was actually observed. A bar adopted mid-flight
    # has an open that is really "the first price we happened to see", and
    # every bps figure derived from it is measured from the wrong place.
    seeded: bool = True

    # How many bars of this timeframe this reading spans. A polled slow
    # timeframe aggregates several, and a normal move ACROSS three bars is
    # larger than a normal move within one, so every threshold below has to
    # know which it is looking at.
    span_bars: float = 1.0

    # -- the raw facts -----------------------------------------------------

    @property
    def move_bps(self) -> float:
        if self.open_px <= 0:
            return 0.0
        return (self.last_px - self.open_px) / self.open_px * 10_000.0

    @property
    def total_notional(self) -> float:
        return self.buy_notional + self.sell_notional

    @property
    def delta(self) -> float:
        return self.buy_notional - self.sell_notional

    @property
    def lean(self) -> float:
        """-1 all selling, +1 all buying."""
        t = self.total_notional
        return self.delta / t if t > 0 else 0.0

    @property
    def aggressor(self) -> Aggressor:
        if abs(self.lean) < BALANCED_BAND:
            return "balanced"
        return "buy" if self.lean > 0 else "sell"

    @property
    def range_bps(self) -> float:
        if self.open_px <= 0:
            return 0.0
        return (self.high_px - self.low_px) / self.open_px * 10_000.0

    @property
    def elapsed_share(self) -> float:
        """How much of this bar has run, never zero.

        A bar one second old is not a bar that should be judged against a
        whole bar's worth of movement or volume.
        """
        if self.interval_s <= 0:
            return 1.0
        span = max(1.0, self.span_bars)
        done = (span - 1.0) + min(1.0, self.elapsed_s / self.interval_s)
        return min(1.0, max(0.02, done / span))

    @property
    def flat_band(self) -> float:
        """How small a move counts as "nowhere" on THIS bar, in bps.

        Scaled three ways, in order of how much each is worth knowing:
        by what this market's bars of this length actually do, by how much
        of the bar has run so far, and -- with no history to learn from --
        by the square root of the interval against a reference.

        The elapsed term uses a square root because that is how far a
        random walk gets in a fraction of the time. Scaling it linearly
        would call the first tenth of a bar flat when it has already made
        a third of a normal bar's move.
        """
        span = max(1.0, self.span_bars)
        if self.typical_bps > 0:
            normal = self.typical_bps * (span ** 0.5)
        else:
            normal = PRIOR_TYPICAL_BPS * (
                self.interval_s * span / FLAT_REF_S) ** 0.5
        return max(MIN_FLAT_BPS,
                   normal * FLAT_SHARE * (self.elapsed_share ** 0.5))

    @property
    def learned(self) -> bool:
        """Is the band a measurement of this market or a stand-in prior."""
        return self.typical_bps > 0

    @property
    def pressing(self) -> bool:
        """Is there enough business here for "absorbed" to mean anything.

        Compared against what this timeframe normally trades by this point
        in a bar, not against an absolute size -- the same notional is
        enormous at 3am and unremarkable at the open.
        """
        if self.measured and self.trades < MIN_TRADES:
            return False
        if self.typical_notional <= 0:
            # Nothing to compare against. Fall back to the only other
            # evidence of real activity there is.
            return self.total_notional > 0 and (
                not self.measured or self.trades >= MIN_TRADES)
        expected = (self.typical_notional * max(1.0, self.span_bars)
                    * self.elapsed_share)
        return self.total_notional >= PRESSING_SHARE * expected

    @property
    def forming(self) -> bool:
        """This bar cannot support a reading yet, and should not pretend to.

        Either its open was never observed -- so every bps figure is
        measured from a price that is not the open -- or there is not
        enough in it yet to tell a quiet bar from an absorbed one.
        """
        if not self.seeded:
            return True
        if self.measured and self.trades < MIN_TRADES:
            return True
        return self.range_bps < self.flat_band and not self.pressing

    @property
    def position_in_range(self) -> float:
        rng = self.high_px - self.low_px
        if rng <= 0:
            return 0.5
        return (self.last_px - self.low_px) / rng

    @property
    def book_imbalance(self) -> float:
        t = self.bid_depth + self.ask_depth
        return (self.bid_depth - self.ask_depth) / t if t > 0 else 0.0

    # -- the answer --------------------------------------------------------

    @property
    def winner(self) -> Side:
        """Who is getting the price they want.

        Aggressors pay the spread. If price is not moving their way, the
        passive side is absorbing them and the passive side is winning.

        TWO GUARDS ON THE INVERTING BRANCH, both learned the hard way.

        "Not moving their way" is judged against `flat_band`, which knows
        how far bars of this length actually travel here. A fixed number of
        basis points called every fast bar flat and inverted it.

        And absorption requires `pressing`: somebody has to be taking real
        size for the passive side to be taking it. A quiet bar that happens
        to lean one way is contested, not absorbed.
        """
        if self.forming:
            return "contested"

        move = self.move_bps
        band = self.flat_band
        agg = self.aggressor

        if agg == "balanced":
            # Nobody is pressing, so the only evidence is the move itself.
            if abs(move) < band:
                return "contested"
            return "buyers" if move > 0 else "sellers"

        if agg == "buy":
            if move > band:
                return "buyers"           # paying up and being rewarded
            if not self.pressing:
                return "contested"        # too quiet to call it absorption
            return "sellers"              # paying up and going nowhere
        if move < -band:
            return "sellers"
        if not self.pressing:
            return "contested"
        return "buyers"

    @property
    def absorbing(self) -> bool:
        """The aggressor is pressing and not getting paid."""
        if self.forming or self.aggressor == "balanced" or not self.pressing:
            return False
        if self.aggressor == "buy":
            return self.move_bps <= self.flat_band
        return self.move_bps >= -self.flat_band

    @property
    def strength(self) -> float:
        """0..1. How decisive this reading is.

        Three things make a reading strong: one side clearly doing the
        aggressing, enough of it to matter, and price responding (or
        conspicuously refusing to).
        """
        if self.forming:
            return 0.0
        one_sided = min(abs(self.lean) / 0.6, 1.0)
        # A decisive move is one that is large FOR THIS BAR. Against a fixed
        # 20bps every fast timeframe reads weak however hard it ran, which
        # is the same mistake the flat band was making at the other end.
        big = (self.typical_bps * (max(1.0, self.span_bars) ** 0.5) * 2.0
               if self.typical_bps > 0 else 20.0)
        moved = min(abs(self.move_bps) / max(big, MIN_FLAT_BPS), 1.0)
        responded = moved
        if self.absorbing:
            # Absorption is strong when aggression is heavy and price is
            # STILL. Reward the stillness rather than the move.
            responded = 1.0 - moved
        base = (one_sided * 0.5 + responded * 0.5)
        if not self.measured:
            base *= 0.6          # inferred, so discounted rather than trusted
        return round(min(base * max(self.coverage, 0.25), 1.0), 4)

    @property
    def signed(self) -> float:
        """+1 buyers winning, -1 sellers winning, scaled by strength."""
        if self.winner == "contested":
            return 0.0
        return self.strength * (1.0 if self.winner == "buyers" else -1.0)

    def describe(self) -> str:
        who = {"buyers": "BUYERS", "sellers": "SELLERS",
               "contested": "CONTESTED"}[self.winner]
        if self.forming:
            if not self.seeded:
                return (f"{self.timeframe}: forming — this bar's open was "
                        f"never seen, so its move is measured from the first "
                        f"price that arrived, not from the open")
            return (f"{self.timeframe}: forming — {self.trades} fills and a "
                    f"{self.range_bps:.2f}bps range against a "
                    f"{self.flat_band:.2f}bps band. Too little has happened "
                    f"to tell a quiet bar from an absorbed one.")
        if self.winner == "contested":
            return f"{self.timeframe}: contested — no side is pressing"

        agg = ("buyers" if self.aggressor == "buy"
               else "sellers" if self.aggressor == "sell" else "neither side")
        if self.absorbing and self.aggressor != "balanced":
            return (f"{self.timeframe}: {who} winning — {agg} are aggressing "
                    f"${self.total_notional:,.0f} and price has moved "
                    f"{self.move_bps:+.2f}bps, inside the "
                    f"{self.flat_band:.2f}bps this bar calls flat. They are "
                    f"being absorbed.")
        return (f"{self.timeframe}: {who} winning — {agg} aggressing into a "
                f"{self.move_bps:+.1f}bps move.")


@dataclass
class Confrontation:
    """Every timeframe's reading, set against the others."""

    readings: list[Pressure] = field(default_factory=list)

    def by_timeframe(self, tf: str) -> Pressure | None:
        return next((r for r in self.readings if r.timeframe == tf), None)

    @property
    def ordered(self) -> list[Pressure]:
        """Slowest first — the context before the trigger."""
        return sorted(self.readings, key=lambda r: -r.interval_s)

    @property
    def higher(self) -> Pressure | None:
        o = self.ordered
        return o[0] if o else None

    @property
    def lower(self) -> Pressure | None:
        o = self.ordered
        return o[-1] if len(o) > 1 else None

    @property
    def aligned(self) -> bool:
        sides = {r.winner for r in self.readings if r.winner != "contested"}
        return len(sides) == 1

    @property
    def conflicted(self) -> bool:
        sides = {r.winner for r in self.readings if r.winner != "contested"}
        return len(sides) > 1

    @property
    def consensus(self) -> Side:
        """Weighted by strength AND by timeframe: an hour of evidence counts
        for more than a minute of it."""
        score = 0.0
        for r in self.readings:
            score += r.signed * (r.interval_s ** 0.5)
        if abs(score) < 1e-9:
            return "contested"
        return "buyers" if score > 0 else "sellers"

    def verdict(self) -> str:
        if not self.readings:
            return "No timeframes read."

        hi, lo = self.higher, self.lower
        if hi is None:
            return "No timeframes read."

        if lo is None or hi.timeframe == lo.timeframe:
            return hi.describe()

        if self.aligned:
            return (f"ALIGNED — {hi.winner.upper()} winning on both "
                    f"{hi.timeframe} and {lo.timeframe}. "
                    f"Continuation, not a fade.")

        if hi.winner != "contested" and lo.winner != "contested" \
                and hi.winner != lo.winner:
            # This is the shape a counter-trend entry is built on: the slower
            # timeframe still belongs to one side while the faster one runs
            # against it.
            return (f"CONFLICT — {hi.timeframe} belongs to {hi.winner} "
                    f"({hi.strength:.0%} conviction) while {lo.timeframe} is "
                    f"{lo.winner} ({lo.strength:.0%}). The lower timeframe is "
                    f"running against the higher one, which is the setup for a "
                    f"fade back in the {hi.winner}' direction — not a reason "
                    f"to follow {lo.timeframe}.")

        return (f"{hi.timeframe} {hi.winner}, {lo.timeframe} {lo.winner} — "
                f"mixed, nothing decisive.")


# --------------------------------------------------------------------------
# building a reading
# --------------------------------------------------------------------------

# How many completed bars the "what is normal here" figures are learned
# from, and the fewest that make a median worth having.
TYPICAL_BARS = 20
MIN_TYPICAL_BARS = 4


def _median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if not n:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def typical(bars: Sequence[Candle],
            back: int = TYPICAL_BARS) -> tuple[float, float]:
    """What a bar of this timeframe normally does here: (bps, notional).

    The MEDIAN, not the mean. One news bar in twenty would drag a mean far
    enough to make every ordinary bar afterwards look flat, which is the
    failure this whole figure exists to prevent.

    Returns (0, 0) when there are too few bars to learn from, and the
    callers fall back to a scaled prior rather than to a number built from
    three samples.
    """
    window = [b for b in bars[-max(1, back):] if b.open > 0]
    if len(window) < MIN_TYPICAL_BARS:
        return 0.0, 0.0
    moves = [abs(b.close - b.open) / b.open * 10_000.0 for b in window]
    notional = [b.volume * b.close if b.close > 0 else b.volume
                for b in window]
    return _median(moves), _median(notional)


def from_live(timeframe: str, interval_s: float, candle, book: Book | None,
              coverage: float = 1.0, history: Sequence[Candle] = ()
              ) -> Pressure:
    """From a `live.LiveCandle`, where the buy/sell split is counted.

    `history` is this timeframe's recent completed bars, used only to learn
    what a normal bar does here -- how far it travels and how much it
    trades. Without it the reading falls back to scaled priors and says so.
    """
    t_bps, t_notional = typical(history)
    return Pressure(
        timeframe=timeframe, interval_s=interval_s,
        open_px=candle.open, last_px=candle.close,
        high_px=candle.high, low_px=candle.low,
        buy_notional=candle.buy_notional, sell_notional=candle.sell_notional,
        measured=True, trades=candle.trades,
        typical_bps=t_bps, typical_notional=t_notional,
        seeded=bool(getattr(candle, "seeded", True)),
        bid_depth=(book.depth(25.0, "buy") if book and not book.empty else 0.0),
        ask_depth=(book.depth(25.0, "sell") if book and not book.empty else 0.0),
        elapsed_s=candle.elapsed(), coverage=coverage)


def from_candle(timeframe: str, interval_s: float, candle: Candle,
                book: Book | None = None, elapsed_s: float = 0.0) -> Pressure:
    """From an OHLCV bar, where the split has to be inferred.

    A candle does not record who crossed the spread. The usual proxy is where
    it closed within its own range: a bar closing on its high spent the period
    with buyers in control of every attempt to push it down.

    That is a weaker instrument than counting fills and it is marked
    `measured=False` so nothing downstream treats the two as equivalent.
    """
    rng = candle.high - candle.low
    pos = ((candle.close - candle.low) / rng) if rng > 0 else 0.5
    notional = candle.volume * candle.close if candle.close > 0 else candle.volume

    return Pressure(
        timeframe=timeframe, interval_s=interval_s,
        open_px=candle.open, last_px=candle.close,
        high_px=candle.high, low_px=candle.low,
        buy_notional=notional * pos, sell_notional=notional * (1.0 - pos),
        measured=False, trades=candle.trades,
        bid_depth=(book.depth(25.0, "buy") if book and not book.empty else 0.0),
        ask_depth=(book.depth(25.0, "sell") if book and not book.empty else 0.0),
        elapsed_s=elapsed_s, coverage=1.0)


# How many bars a polled reading aggregates, by how long the bar is.
#
# THE THREE BAR WINDOW IS RIGHT FOR SLOW TIMEFRAMES AND WRONG FOR FAST ONES
#
# Two minutes into a four-hour candle there is nothing in it, so reading
# only the bar in progress makes the higher timeframe go silent exactly
# when it matters. That is the argument for the window, and it is a good
# one -- about four-hour bars.
#
# Applied to a one-minute bar it does the opposite. A reversal inside the
# current minute gets averaged with the two minutes before it, so the row
# keeps reading UP while price is already coming down. That is not a lag
# in the feed; it is the window outvoting the present.
#
# So the window scales. A bar short enough to fill up on its own is read
# on its own.
FAST_S = 300.0        # bars this long or shorter are read one at a time


def lookback_for(interval_s: float, slow: int = 3) -> int:
    return 1 if interval_s <= FAST_S else slow


def from_candles(timeframe: str, interval_s: float, bars: Sequence[Candle],
                 lookback: int | None = None, book: Book | None = None,
                 elapsed_s: float = 0.0) -> Pressure | None:
    """Who has been winning over the last few bars of this timeframe.

    Reading only the bar in progress is wrong for a slow timeframe. Two
    minutes into a four-hour candle there is almost nothing in it, so the
    reading comes back "contested" and the higher timeframe -- the whole
    reason for looking at it -- contributes nothing exactly when you need it.

    "The 4-hour is down" does not mean the current 4-hour bar. It means the
    recent ones. So the window is aggregated: open from the first bar in it,
    high and low across it, close from the last, and the inferred buy/sell
    split accumulated bar by bar rather than taken from the final one.
    """
    if lookback is None:
        lookback = lookback_for(interval_s)
    window = [b for b in bars[-max(1, lookback):] if b.high >= b.low]
    if not window:
        return None
    # Learned from the bars BEFORE the window being read, so the window
    # does not help set the threshold it is about to be judged against.
    t_bps, t_notional = typical(bars[:-len(window)] or bars)

    buy = sell = 0.0
    trades = 0
    for b in window:
        rng = b.high - b.low
        pos = ((b.close - b.low) / rng) if rng > 0 else 0.5
        notional = b.volume * b.close if b.close > 0 else b.volume
        buy += notional * pos
        sell += notional * (1.0 - pos)
        trades += b.trades

    return Pressure(
        timeframe=timeframe, interval_s=interval_s,
        open_px=window[0].open, last_px=window[-1].close,
        high_px=max(b.high for b in window),
        low_px=min(b.low for b in window),
        buy_notional=buy, sell_notional=sell,
        measured=False, trades=trades,
        typical_bps=t_bps, typical_notional=t_notional,
        span_bars=float(len(window)),
        bid_depth=(book.depth(25.0, "buy") if book and not book.empty else 0.0),
        ask_depth=(book.depth(25.0, "sell") if book and not book.empty else 0.0),
        elapsed_s=elapsed_s, coverage=1.0)


def confront(readings: Sequence[Pressure]) -> Confrontation:
    return Confrontation(readings=[r for r in readings if r is not None])


# --------------------------------------------------------------------------
# price action: what price DID, not what it tends to do
# --------------------------------------------------------------------------
#
# VWAP distance is a PRIOR. It says "price is stretched, so it tends to come
# back" -- a statement about tendencies, not about this market right now. In a
# read whose whole premise is observation over prediction, it does not belong
# alongside measurements, and it was diluting them.
#
# These three are observations. Each one is a thing that visibly happened and
# that you could point at on the chart:
#
#   REJECTION   a bar printed a high and closed far below it. Sellers took
#               the level back. The wick is the evidence.
#   SWEEP       price traded through a prior high or low and closed back
#               inside. The breakout failed, and whoever bought it is trapped.
#               This is the single most useful scalp signal there is.
#   ACCEPTANCE  price closed beyond a prior range and stayed there. The
#               opposite of a sweep, and the thing that makes a breakout real.
#
# All three read the same bars, and all three are signed the same way as
# everything else: positive favours buyers.

@dataclass
class PriceAction:
    """Observed behaviour over the last few bars."""

    rejection: float = 0.0        # -1 top rejected ... +1 bottom rejected
    sweep: float = 0.0            # -1 high swept and failed ... +1 low swept
    acceptance: float = 0.0       # -1 accepted below ... +1 accepted above
    notes: list[str] = field(default_factory=list)

    @property
    def signed(self) -> float:
        """Combined, bounded to -1..+1.

        A sweep is weighted hardest: a failed break leaves trapped
        participants who have to do something about it, which is a mechanism
        rather than a tendency.
        """
        raw = self.sweep * 0.5 + self.rejection * 0.3 + self.acceptance * 0.2
        return max(-1.0, min(1.0, raw))

    @property
    def active(self) -> bool:
        return abs(self.signed) > 0.05

    def describe(self) -> str:
        if not self.notes:
            return "nothing notable in the price action"
        return "; ".join(self.notes)


def price_action(bars: Sequence[Candle], lookback: int = 12) -> PriceAction:
    """Read rejection, sweeps and acceptance off the bars themselves."""
    pa = PriceAction()
    if len(bars) < 3:
        return pa

    window = list(bars[-max(3, lookback):])
    last = window[-1]
    prior = window[:-1]

    # -- rejection: where did the last bar close inside its own range ------
    rng = last.high - last.low
    if rng > 0:
        upper_wick = last.high - last.body_top
        lower_wick = last.body_bottom - last.low
        # Only the wick matters here. Comparing the close to `body_top` was
        # wrong: on a bullish bar the body top IS the close, so that test
        # could never pass and upper-wick rejection never fired at all.
        # Giving back most of the range is the observation, whichever way the
        # body happens to point.
        # The wick must clearly DOMINATE the other side. A bar with equal
        # wicks both ways is indecision, not rejection, and reading every
        # doji as a rejection would fire this signal on almost every bar.
        if upper_wick / rng > 0.45 and upper_wick > lower_wick * 1.5:
            pa.rejection = -min(upper_wick / rng, 1.0)
            pa.notes.append(
                f"upper wick is {upper_wick / rng:.0%} of the bar — the high "
                f"at {last.high:,.2f} was rejected")
        elif lower_wick / rng > 0.45 and lower_wick > upper_wick * 1.5:
            pa.rejection = min(lower_wick / rng, 1.0)
            pa.notes.append(
                f"lower wick is {lower_wick / rng:.0%} of the bar — the low "
                f"at {last.low:,.2f} was rejected")

    # -- sweep: took out a prior extreme and closed back inside ------------
    prior_high = max(b.high for b in prior)
    prior_low = min(b.low for b in prior)

    if last.high > prior_high and last.close < prior_high:
        depth = (last.high - prior_high) / prior_high * 10_000.0
        pa.sweep = -min(1.0, 0.4 + depth / 25.0)
        pa.notes.append(
            f"swept the prior high at {prior_high:,.2f} by {depth:.1f}bps and "
            f"closed back under it — failed break, buyers trapped")
    elif last.low < prior_low and last.close > prior_low:
        depth = (prior_low - last.low) / prior_low * 10_000.0
        pa.sweep = min(1.0, 0.4 + depth / 25.0)
        pa.notes.append(
            f"swept the prior low at {prior_low:,.2f} by {depth:.1f}bps and "
            f"closed back above it — failed break, sellers trapped")

    # -- acceptance: closed beyond the range and held ----------------------
    elif last.close > prior_high:
        pa.acceptance = 1.0
        pa.notes.append(f"closed above the prior high at {prior_high:,.2f} — "
                        f"accepted, not rejected")
    elif last.close < prior_low:
        pa.acceptance = -1.0
        pa.notes.append(f"closed below the prior low at {prior_low:,.2f} — "
                        f"accepted, not rejected")

    return pa
