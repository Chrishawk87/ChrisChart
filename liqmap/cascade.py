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
    # Was this bar's OPEN actually observed? Without it `bps` is
    # measured from whatever price happened to arrive first, which
    # is not a direction at all -- the one gate below that is about
    # the data rather than about noise.
    seeded: bool = True
    # WHERE THIS ROW CAME FROM AND HOW OLD IT IS. The route has
    # always known -- `srcs` and `ages` are computed beside the
    # reading -- and the stack panel has never shown it, so a row
    # built from three polled bars looked identical to one built
    # from the bar in progress. That is most of why the 5m row and
    # the 5m window could disagree without either looking wrong.
    source: str = ""            # tape | last closed | polled
    age_s: float | None = None
    bars_used: float = 1.0
    # THE ACTUAL PRICES. Chris: "the Gate 2 stack is still reading
    # BPS not true price". The direction IS close against open now,
    # but a bps number is not something anyone can check against a
    # broker's chart -- so the row carries the two prices it was
    # computed from and the panel shows them.
    open_px: float = 0.0
    last_px: float = 0.0

    @property
    def qualified(self) -> Way | None:
        """WHERE PRICE IS AGAINST THIS BAR'S OWN OPEN. Nothing else.

        Chris: "i want the stack timeframes to read like the 5m
        window does thats true direction". The window's `mid_dir` is
        close against open with no threshold -- one tick either way
        decides it -- and these rows now answer the same question on
        their own timeframe.

        WHAT WAS DROPPED, AND WHY IT WAS THERE.

            It used to require the COLOUR and the NUMBER to agree:
            up only if green and bps positive. A row whose colour
            fought its own number was thrown out and counted as no
            direction at all. That disagreement is real -- it is
            absorption, aggression on one side with price going the
            other -- but it is a SECOND measurement, and gating the
            direction on it meant the stack could not answer the
            question the window answers.

            It is still measured and still shown, as `absorbed`. It
            decides nothing now.

            The flat band, the three-fill minimum and the pressing
            share went with it: all three are noise gates, and a
            gate that makes a row silent is a gate that blocks every
            entry built on that row.

        THE ONE GATE KEPT is `seeded`. Without the bar's open there
        is no close-against-open to compute, and a number measured
        from the wrong anchor is wrong rather than merely noisy.

        Zero is flat and flat is not a side, exactly as the window
        treats it.
        """
        if not self.present or not self.seeded:
            return None
        if self.bps > 0:
            return "up"
        if self.bps < 0:
            return "down"
        return None

    @property
    def absorbed(self) -> bool:
        """Aggression one way, price the other. SHOWN, NOT USED.

        It no longer blocks the row. Kept because it is the single
        most informative thing this stack measures -- a 5m that is
        green on aggression while price bleeds lower is a different
        market from one that is simply down -- and because a reading
        that silently stops being collected is how a panel starts
        lying.
        """
        if not self.present or self.colour not in ("green", "red"):
            return False
        want = "up" if self.colour == "green" else "down"
        return self.qualified is not None and self.qualified != want

    @property
    def conflicted(self) -> bool:
        """Retained for the panel's THROWN OUT badge, which no longer
        means thrown out -- it means the two measurements disagree.
        The row still counts, by its price."""
        return self.absorbed

    def describe(self) -> str:
        """Lead with the DIRECTION, because that is what now counts.

        This used to return "no colour" and stop, which was right
        when the colour decided the row. It does not any more: a row
        with no colour and a clear price move has a direction, and
        saying "no colour" about it hides the thing being used.
        """
        if not self.present:
            return f"{self.timeframe}: not read"
        if not self.seeded:
            return (f"{self.timeframe}: this bar's open was never seen "
                    f"-- no direction from it")
        way = self.qualified
        if way is None:
            return f"{self.timeframe}: flat, {self.bps:+.1f}bps"
        if self.absorbed:
            return (f"{self.timeframe}: {way} {self.bps:+.1f}bps -- "
                    f"{self.colour} on aggression, absorbed")
        if self.open_px and self.last_px:
            return (f"{self.timeframe}: {way} "
                    f"{self.open_px:g} -> {self.last_px:g}")
        return f"{self.timeframe}: {way} {self.bps:+.1f}bps"

    def to_dict(self) -> dict:
        return {"timeframe": self.timeframe, "colour": self.colour,
                "bps": round(self.bps, 3), "present": self.present,
                "seeded": bool(self.seeded),
                "source": self.source, "age_s": self.age_s,
                "open_px": round(self.open_px, 4) or None,
                "last_px": round(self.last_px, 4) or None,
                "ticks": (round((self.last_px - self.open_px) / 0.25, 1)
                          if self.open_px and self.last_px else None),
                "bars_used": round(float(self.bars_used or 1.0), 2),
                "qualified": self.qualified, "absorbed": self.absorbed,
                "conflicted": self.conflicted,
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


def rows_from_candles(trading: str, candle_of, aggression: dict | None = None,
                      ) -> dict:
    """Stack rows built from the REAL forming candles.

    Chris: "i want what a real 5 minute 15 minute and 1 minute
    candle is doing not a calculation of it".

    `candle_of(timeframe)` returns that timeframe's forming candle as
    the chart draws it -- a mapping with `open` and `close` -- or
    None. `aggression` optionally maps a timeframe to (colour,
    forming) for the absorption flag, which is shown and decides
    nothing.

    WHY THIS IS A NAMED FUNCTION AND NOT A FEW LINES IN THE ROUTE.
    It lived in a route closure, reachable only through an HTTP
    request, and a test of it had to stand up the whole handler --
    which is how the auto gate and the one-per-candle cap each got
    deleted without breaking anything. The policy is here; the route
    supplies the candles.

    A timeframe with no candle is LEFT OUT rather than given a flat
    row. Absent and flat call for opposite reactions, and a row that
    invents a reading is worse than one that admits it has none.
    """
    out: dict = {}
    for tf in STACKS.get(trading, ()):
        try:
            c = candle_of(tf)
        except Exception:                               # noqa: BLE001
            c = None
        if not c:
            continue
        try:
            o = float(c.get("open") or 0.0)
            cl = float(c.get("close") or 0.0)
        except (TypeError, ValueError, AttributeError):
            continue
        if o <= 0 or cl <= 0:
            continue
        colour, forming = (aggression or {}).get(tf, ("none", False))
        out[tf] = (
            colour_of(colour, forming=forming) if colour in
            ("buyers", "sellers") else (colour if colour in
                                        ("green", "red") else "none"),
            (cl - o) / o * 10_000.0,
            True,
            # The caller says where it found the candle -- the live
            # forming one, or the stored bar it falls back to right
            # after a restart. Either way it is ONE REAL BAR, and
            # the row says which so a stale source stays visible.
            {"source": str(c.get("source") or "candle"),
             "age_s": c.get("seconds"), "bars_used": 1.0,
             "open_px": o, "last_px": cl},
        )
    return out


def read(trading: str, rows: dict, previous: Way | None = None) -> Cascade:
    """Read the stack beneath `trading`.

    `rows` maps a timeframe to (colour, bps), (colour, bps, seeded)
    or (colour, bps, seeded, meta).
    The third element is optional so existing callers keep working;
    without it the bar's open is assumed observed, which is what it
    is for every row that has been measured normally. `previous` is the
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
            colour, bps = got[0], got[1]
            seeded = bool(got[2]) if len(got) > 2 else True
            # Provenance travels as an optional fourth element so
            # every existing caller keeps working untouched.
            meta = got[3] if len(got) > 3 else {}
            out.rungs.append(Rung(
                timeframe=tf, colour=colour, bps=float(bps),
                present=True, seeded=seeded,
                source=str(meta.get("source") or ""),
                age_s=meta.get("age_s"),
                bars_used=float(meta.get("bars_used") or 1.0),
                open_px=float(meta.get("open_px") or 0.0),
                last_px=float(meta.get("last_px") or 0.0)))

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
