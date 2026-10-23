"""The current leg, and whether the side driving it is getting paid.

THE QUESTION THIS ANSWERS, which the bar-level reader does not.

`fpread.read_candle` already measures effort against result: one side
paying hard while price refuses to move is absorption, and it refuses
to call that bar "strong buyers". But it measures across the WHOLE
bar, open to close over the bar's range. On a five-minute ES bar that
is a summary of three hundred seconds, and a scalper deciding what to
do in the next twenty is asking a different question:

    price just went four ticks. Who ate the contracts to do it, and
    are they still being paid for the next one?

That is a reading of the LEG, not of the bar -- it starts at the last
turn in price and ends now, and it may be a quarter of a bar or span
three of them. So it lives here rather than in the candle reader.

WHAT MAKES IT ABSORPTION RATHER THAN A PAUSE. Arriving at a level and
stopping is not absorption; price stops constantly. Absorption is
paying the going rate for another tick and not getting it. The going
rate is measurable: during the push itself, the leg cost some number
of contracts per tick travelled. If the driving side has since spent
that much again at the extreme without moving it, somebody is filling
them there and the effort is going nowhere. That is the comparison
this module makes, and it is why the threshold is not a constant --
a quiet overnight tick and a tick at the open cost different amounts,
and a fixed number would call one of them wrong all session.

IT SUGGESTS. Nothing here places, sizes, or routes anything, and the
tests assert there is no such code in the file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

BUYERS = "buyers"
SELLERS = "sellers"
BALANCED = "balanced"

UP = "up"
DOWN = "down"

FORMING = "forming"
CONTINUATION = "continuation"
ABSORBED = "absorbed"


@dataclass(frozen=True, slots=True)
class Push:
    """One leg of price, and the verdict on it."""

    direction: str = ""          # "up", "down", or "" before a leg exists
    eater: str = BALANCED        # who aggressed the contracts during it
    travel_ticks: int = 0        # extreme, measured from the pivot
    give_back_ticks: int = 0     # how far back off the extreme price is
    eaten_buy: int = 0           # aggressive buys during the leg
    eaten_sell: int = 0          # aggressive sells during the leg
    since_extreme: int = 0       # driver's contracts spent AT the extreme
    per_tick: float = 0.0        # what a tick cost during the push
    window: int = 4              # ticks that make a push worth calling
    pivot_px: float = 0.0
    extreme_px: float = 0.0
    last_px: float = 0.0

    @property
    def verdict(self) -> str:
        """"forming", "continuation", or "absorbed"."""
        if not self.direction or self.travel_ticks < self.window:
            return FORMING
        if self.give_back_ticks >= self.window:
            return ABSORBED             # it got there and handed it all back
        if self.per_tick > 0 and self.since_extreme >= self.per_tick:
            return ABSORBED             # paid for another tick, got nothing
        return CONTINUATION

    @property
    def agrees(self) -> bool:
        """Did the side that ate the contracts match the way price went?

        Normally yes -- that is how a tick moves. When it does not,
        price drifted on thin size rather than being driven, and a
        continuation read off it is worth less.
        """
        if not self.direction or self.eater == BALANCED:
            return False
        return ((self.direction == UP and self.eater == BUYERS)
                or (self.direction == DOWN and self.eater == SELLERS))

    @property
    def scalp(self) -> str:
        """"long", "short", or "" -- A READING, NOT A FORECAST.

        A continuation reads with the leg. Absorption reads against
        it, because the side that drove it is the side being filled:
        buyers who have spent the price of another tick at the high
        and not got it are buying from somebody who is happy to sell
        there, and the inventory sitting against them is the reason
        the level matters.

        Measured over 102.5 hours of ES, the flow states on this
        panel do not precede price at 15s, 30s or 60s. THIS reading
        is not one of those -- it is a leg, not a bar state, and it
        has not been measured yet. Section 11 of cvd_study is what
        would settle it. Until that has run on real bars, this names
        a side and nothing more.
        """
        v = self.verdict
        if v == FORMING:
            return ""
        if v == CONTINUATION:
            return "long" if self.direction == UP else "short"
        return "short" if self.direction == UP else "long"

    @property
    def invalidated_at(self) -> float:
        """The price that says the read was wrong.

        A continuation is wrong once price is back at the pivot: the
        leg did not hold. Absorption is wrong once price takes out
        the extreme: whoever was filling there has stopped, or run
        out, and the move is going after all.
        """
        if self.verdict == CONTINUATION:
            return self.pivot_px
        return self.extreme_px

    def as_dict(self) -> dict:
        return {"direction": self.direction, "eater": self.eater,
                "travel_ticks": self.travel_ticks,
                "give_back_ticks": self.give_back_ticks,
                "eaten_buy": self.eaten_buy,
                "eaten_sell": self.eaten_sell,
                "since_extreme": self.since_extreme,
                "per_tick": round(self.per_tick, 1),
                "window": self.window,
                "pivot_px": self.pivot_px,
                "extreme_px": self.extreme_px,
                "last_px": self.last_px,
                "verdict": self.verdict, "agrees": self.agrees,
                "scalp": self.scalp,
                "invalidated_at": self.invalidated_at}


class PushWatcher:
    """Follows the current leg of price and who is paying for it.

    A leg runs from a pivot to the furthest price reached since. It
    turns over when price comes back `pivot_ticks` off that extreme,
    and the extreme becomes the new pivot -- so the reading always
    describes the move in front of you rather than one from earlier.

    NOT SCOPED TO THE BAR. A leg is a price structure and does not
    care where the five-minute boundaries fall; half the moves worth
    reading start mid-bar and finish in the next one. The bar-scoped
    reading is `fpread`, and the two are meant to disagree sometimes.
    """

    __slots__ = ("scale", "window", "pivot_ticks", "absorb_mult",
                 "direction", "_pivot", "_extreme", "_last",
                 "_buy", "_sell", "_since", "_cost", "_started")

    def __init__(self, tick_size: float = 0.25, window_ticks: int = 4,
                 pivot_ticks: int = 2, absorb_mult: float = 1.0) -> None:
        if tick_size <= 0:
            raise ValueError("tick_size must be positive")
        if window_ticks < 1:
            raise ValueError("window_ticks must be at least 1")
        if pivot_ticks < 1:
            raise ValueError("pivot_ticks must be at least 1")
        if absorb_mult <= 0:
            raise ValueError("absorb_mult must be positive")
        self.scale = float(tick_size)
        self.window = int(window_ticks)
        self.pivot_ticks = int(pivot_ticks)
        # How many times the going rate must be spent at the extreme
        # before it is called absorption. 1.0 means "the price of one
        # more tick". Higher is slower to call it and surer when it
        # does.
        self.absorb_mult = float(absorb_mult)
        self.reset()

    def reset(self) -> None:
        self.direction = ""
        self._pivot = None          # ticks
        self._extreme = None        # ticks
        self._last = None           # ticks
        self._buy = 0               # aggressive buys this leg
        self._sell = 0
        self._since = 0             # driver's contracts since the extreme
        self._cost = 0              # contracts spent reaching the extreme
        self._started = 0.0

    def _ticks(self, px: float) -> int:
        return int(round(float(px) / self.scale))

    def _begin(self, at: int, ts: float) -> None:
        self._pivot = at
        self._extreme = at
        self.direction = ""
        self._buy = self._sell = 0
        self._since = 0
        self._cost = 0
        self._started = ts

    def on_trade(self, ts: float, px: float, size: int,
                 aggressor: str) -> None:
        if size <= 0:
            return
        t = self._ticks(px)
        if self._pivot is None:
            self._begin(t, ts)
        self._last = t

        if aggressor == BUYERS or aggressor == "buy":
            self._buy += int(size)
            driver_side = BUYERS
        else:
            self._sell += int(size)
            driver_side = SELLERS

        # A leg has a direction as soon as price leaves the pivot, and
        # keeps it until price comes far enough back to turn over.
        if not self.direction and t != self._pivot:
            self.direction = UP if t > self._pivot else DOWN

        new_extreme = (self.direction == UP and t > self._extreme) or \
                      (self.direction == DOWN and t < self._extreme)
        if new_extreme:
            self._extreme = t
            # Everything spent so far bought the distance so far. The
            # counter at the extreme restarts, because the question is
            # what has been spent since arriving, not in total.
            self._cost = self._buy + self._sell
            self._since = 0
        elif self.direction:
            # Only the DRIVING side's effort counts as effort at the
            # extreme. The other side trading there is the fill, not
            # the attempt, and counting it would make every level look
            # absorbed as soon as it was two-sided -- which is all of
            # them.
            want = BUYERS if self.direction == UP else SELLERS
            if driver_side == want:
                self._since += int(size)

        # Has it turned over? Measured off the extreme, so a leg ends
        # where the next one starts.
        if self.direction and self._extreme is not None:
            back = abs(t - self._extreme)
            if back >= self.pivot_ticks:
                top = self._extreme
                self._begin(top, ts)
                self._last = t
                self.direction = UP if t > top else DOWN
                self._extreme = t

    def reading(self) -> Push:
        if self._pivot is None or self._extreme is None:
            return Push(window=self.window)
        travel = abs(self._extreme - self._pivot)
        # What a tick cost on the way here. Guarded: a leg that has
        # not moved has no rate, and dividing by it would make every
        # standing bid look like absorption.
        per_tick = (self._cost / travel) * self.absorb_mult if travel else 0.0
        eater = (BALANCED if self._buy == self._sell
                 else (BUYERS if self._buy > self._sell else SELLERS))
        last = self._last if self._last is not None else self._extreme
        return Push(direction=self.direction, eater=eater,
                    travel_ticks=travel,
                    give_back_ticks=abs(self._extreme - last),
                    eaten_buy=self._buy, eaten_sell=self._sell,
                    since_extreme=self._since, per_tick=per_tick,
                    window=self.window,
                    pivot_px=self._pivot * self.scale,
                    extreme_px=self._extreme * self.scale,
                    last_px=last * self.scale)
