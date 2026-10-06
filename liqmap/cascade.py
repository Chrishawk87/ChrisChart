"""The stack below the timeframe you are trading, read as one thing.

THE IDEA

A one-minute candle builds a fifteen-minute candle from the ground up. So
the fast rows are not separate opinions about the market, they are the
fifteen being constructed in front of you. Read that way, the stack says
something a single row cannot:

    all red, all the way down      price is going down
    the 1m flips green             a pullback, and a shallow one
    the 5m flips too               a bigger pullback
    the trading timeframe flips    no longer a pullback -- it turned

Which timeframe is being traded decides the stack: the 15m is read with
the 5m and the 1m beneath it, the 5m with only the 1m, and the 1m stands
alone. Anything ABOVE the timeframe being traded takes no part in this
read at all.

THE CAVEAT THAT MAKES IT WORTH ANYTHING

Colour alone is not enough. A row counts as up only if it is green AND
its move is positive, and as down only if red AND negative. A green row
with a negative move is not a weak up-read, it is a row contradicting
itself, and it is thrown out.

That case is not rare or hypothetical. It is exactly what absorption
does: heavy selling into a bid that will not break reads as BUYERS on a
candle whose price has gone down. The colour and the number disagree
because they are measuring different things, and a rule that takes
either one on its own will be wrong in precisely the moments that matter.

NOTHING IS COMMITTED WHILE THE TRADING TIMEFRAME IS SILENT

If the row being traded does not qualify -- no colour, or a colour
fighting its own number -- then the faster rows agreeing with each other
means nothing. They agree about a bar that has not said anything yet.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Sequence

Way = Literal["up", "down"]
Side = Literal["buy", "sell"]

# The stack for each timeframe that can be traded: itself first, then
# everything beneath it, fastest last.
STACKS: dict[str, tuple[str, ...]] = {
    "1m": ("1m",),
    "5m": ("5m", "1m"),
    "15m": ("15m", "5m", "1m"),
    "30m": ("30m", "15m", "5m", "1m"),
    "1h": ("1h", "15m", "5m", "1m"),
    "4h": ("4h", "1h", "15m", "5m", "1m"),
}


class State(str, Enum):
    TREND = "trend"
    PULLBACK = "pullback"
    REVERSAL = "reversal"
    NO_COMMIT = "no_commit"      # the traded row is not saying anything
    MIXED = "mixed"
    UNREAD = "unread"            # rows missing from the stack


def _side(way: Way | None) -> Side | None:
    return "buy" if way == "up" else "sell" if way == "down" else None


@dataclass
class Rung:
    """One row of the stack, and whether it is allowed to count."""

    timeframe: str
    colour: str = "none"         # green | red | none
    bps: float = 0.0
    present: bool = False

    @property
    def qualified(self) -> Way | None:
        """Up only if green AND positive; down only if red AND negative.

        A row whose colour fights its own number is thrown out rather than
        counted weakly. That disagreement is absorption, and it means the
        two are measuring different things.
        """
        if not self.present:
            return None
        if self.colour == "green" and self.bps > 0:
            return "up"
        if self.colour == "red" and self.bps < 0:
            return "down"
        return None

    @property
    def conflicted(self) -> bool:
        """It had a colour, and the number went the other way."""
        return (self.present and self.colour in ("green", "red")
                and self.qualified is None)

    def describe(self) -> str:
        if not self.present:
            return f"{self.timeframe}: not read"
        if self.colour == "none":
            return f"{self.timeframe}: no colour"
        if self.conflicted:
            return (f"{self.timeframe}: {self.colour} but {self.bps:+.1f}bps "
                    f"-- contradicts itself, thrown out")
        return f"{self.timeframe}: {self.colour} {self.bps:+.1f}bps"

    def to_dict(self) -> dict:
        return {"timeframe": self.timeframe, "colour": self.colour,
                "bps": round(self.bps, 3), "present": self.present,
                "qualified": self.qualified, "conflicted": self.conflicted,
                "describe": self.describe()}


@dataclass
class Cascade:
    """What the stack beneath the traded timeframe is doing."""

    trading: str
    rungs: list[Rung] = field(default_factory=list)
    state: State = State.UNREAD
    trend: Way | None = None       # what the traded row says
    pull: Way | None = None        # which way the fast rows have flipped
    depth: int = 0                 # how many of the lower rows flipped
    side: Side | None = None
    why: str = ""

    @property
    def lower(self) -> list[Rung]:
        return self.rungs[1:]

    @property
    def deep(self) -> bool:
        """Every row beneath the traded one has flipped."""
        return bool(self.lower) and self.depth == len(self.lower)

    def describe(self) -> str:
        if self.state is State.UNREAD:
            return self.why or "the stack has not been read"
        if self.state is State.NO_COMMIT:
            return self.why
        if self.state is State.PULLBACK:
            how = "deep" if self.deep else "shallow"
            return (f"{how} pullback {self.pull} against a "
                    f"{self.trend} {self.trading} -- "
                    f"{self.depth} of {len(self.lower)} flipped")
        if self.state is State.REVERSAL:
            return (f"the {self.trading} has flipped {self.trend} with the "
                    f"whole stack behind it")
        if self.state is State.TREND:
            return (f"{self.trading} and everything under it are "
                    f"{self.trend}")
        return self.why

    def to_dict(self) -> dict:
        return {"trading": self.trading, "state": self.state.value,
                "trend": self.trend, "pull": self.pull, "depth": self.depth,
                "deep": self.deep, "side": self.side, "why": self.why,
                "rungs": [r.to_dict() for r in self.rungs],
                "describe": self.describe()}


def colour_of(winner: str | None, forming: bool = False) -> str:
    """The slider's colour, from the row's winning side."""
    if forming or not winner:
        return "none"
    return {"buyers": "green", "sellers": "red"}.get(winner, "none")


def read(trading: str, rows: dict, previous: Way | None = None) -> Cascade:
    """Read the stack beneath `trading`.

    `rows` maps a timeframe to (colour, bps, present). `previous` is the
    direction the stack was last ALIGNED in -- the only thing that tells a
    turn apart from a trend that has simply been running. Without it a
    flip is reported as a trend, which is true but is not the thing worth
    knowing.
    """
    out = Cascade(trading=trading)
    want = STACKS.get(trading)
    if not want:
        out.why = f"{trading} is not a timeframe this reads"
        return out

    for tf in want:
        got = rows.get(tf)
        if got is None:
            out.rungs.append(Rung(timeframe=tf, present=False))
        else:
            colour, bps = got
            out.rungs.append(Rung(timeframe=tf, colour=colour,
                                  bps=float(bps), present=True))

    missing = [r.timeframe for r in out.rungs if not r.present]
    if missing:
        out.state = State.UNREAD
        out.why = (f"no read on {', '.join(missing)} -- the stack is "
                   f"incomplete")
        return out

    head = out.rungs[0]
    out.trend = head.qualified

    # THE TRADED ROW DECIDES WHETHER ANYTHING IS DECIDED. Faster rows
    # agreeing with each other while this one says nothing is agreement
    # about a bar that has not spoken.
    if out.trend is None:
        out.state = State.NO_COMMIT
        said = ("contradicts itself" if head.conflicted
                else "has no colour")
        out.why = (f"the {trading} {said} -- nothing under it counts yet")
        return out

    other: Way = "down" if out.trend == "up" else "up"
    out.depth = sum(1 for r in out.lower if r.qualified == other)
    with_trend = sum(1 for r in out.lower if r.qualified == out.trend)

    if not out.lower or with_trend == len(out.lower):
        # Everything agrees. Whether that is a turn depends on where it
        # came from, which is the only thing `previous` is for.
        if previous is not None and previous != out.trend:
            out.state = State.REVERSAL
        else:
            out.state = State.TREND
        out.side = _side(out.trend)
        return out

    if out.depth:
        out.state = State.PULLBACK
        out.pull = other
        # No side on a pullback. The stack is mid-argument, and the rule
        # throughout is that nothing is committed until it has finished.
        return out

    out.state = State.MIXED
    out.why = (f"the {trading} is {out.trend} but the rows under it are "
               f"not saying either way")
    return out


def aligned_way(c: Cascade) -> Way | None:
    """The direction to remember, for telling the next turn apart.

    ONLY A FULLY ALIGNED STACK IS WORTH REMEMBERING. Carrying a pullback
    forward would make the next alignment look like a reversal of
    something that never happened.
    """
    return c.trend if c.state in (State.TREND, State.REVERSAL) else None
