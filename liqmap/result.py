"""Layer seven: what the candle is actually doing, and which way that points.

THE SEVENTH LAYER IS A VERDICT AND THE OTHER SIX ARE NOT

`layers` describes the bar. This decides what the description amounts to,
and gives it a side. Keeping them in separate files is what lets a wrong
word be traced back to the reading that produced it rather than to the
whole pile.

FOUR OF THE NINE STATES ARE NOT KNOWABLE YET

Failed breakout, failed breakdown, exhaustion and reversal all describe
how something TURNED OUT. Live, inside a bar that has not closed, the
most that can honestly be said is that the shape is there so far:

    price traded above the level and is back below it -- for now
    the aggression has stopped paying -- so far

Every one of those carries `provisional`, and it means what it says: the
bar can still close the other way and the word will change. The other
five -- continuation, expansion, absorption, rejection, consolidation --
are statements about what has already happened and do not move.

This distinction is not pedantry. A classifier that reports "failed
breakout" live, with the confidence of a closed bar, is using information
the moment did not have, and that is the exact shape of the bug that
scored 98.4% on a coin flip earlier in this project.

ABSORPTION INVERTS THE SIDE

Everywhere else here, the side follows the move. Under absorption it
follows the PASSIVE side instead: heavy buying that will not move price
means somebody is selling into all of it, and they are the ones winning.
That inversion is the single most valuable thing in this file and the
thing a naive reading gets backwards.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER. It returns a word and a
side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

Side = Literal["buy", "sell"]


class State(str, Enum):
    CONTINUATION = "continuation"
    EXPANSION = "expansion"
    ABSORPTION = "absorption"
    EXHAUSTION = "exhaustion"
    REJECTION = "rejection"
    FAILED_BREAKOUT = "failed_breakout"
    FAILED_BREAKDOWN = "failed_breakdown"
    REVERSAL = "reversal"
    CONSOLIDATION = "consolidation"
    UNKNOWN = "unknown"


# The four that describe how something turned out. Live, they are a shape
# so far and nothing more.
PROVISIONAL = {State.FAILED_BREAKOUT, State.FAILED_BREAKDOWN,
               State.EXHAUSTION, State.REVERSAL}

MEANING: dict[State, str] = {
    State.CONTINUATION: "the side doing the attacking is getting paid for it",
    State.EXPANSION: "price is travelling further than this much flow "
                     "normally moves it -- a thin book, so moves extend",
    State.ABSORPTION: "real size is crossing and price will not move, so "
                      "somebody is filling all of it passively",
    State.EXHAUSTION: "the move has already run and the aggression has "
                      "stopped paying for it",
    State.REJECTION: "price reached the level and came back without "
                     "trading through it",
    State.FAILED_BREAKOUT: "price traded above the level and is back below",
    State.FAILED_BREAKDOWN: "price traded below the level and is back above",
    State.REVERSAL: "the bar has turned against the one before it",
    State.CONSOLIDATION: "not enough flow or movement to be doing anything",
    State.UNKNOWN: "not enough of this bar has been measured to say",
}

# Volume against what a normal bar of this length has done by now.
HEAVY = 1.5
QUIET = 0.6

# Price moved, against how far this much flow normally moves it.
LOW_IMPACT = 0.5           # matches `flow.Absorption.absorbing`
HIGH_IMPACT = 2.0          # matches `flow.Absorption.thin`

# How far from its extreme a bar has to come back to have rejected.
REJECT_BACK = 0.35

# Layers that must be measured before any word is worth saying.
REQUIRED = ("intent", "response")


@dataclass
class Result:
    """One word for the candle, and the side it points."""

    state: State = State.UNKNOWN
    side: Side | None = None
    provisional: bool = False
    why: str = ""
    missing: list[str] = field(default_factory=list)

    @property
    def actionable(self) -> bool:
        """A side, from a state that is not still deciding what it is."""
        return self.side is not None and not self.provisional

    def describe(self) -> str:
        if self.state is State.UNKNOWN:
            return (f"not enough measured to say"
                    + (f" -- missing {', '.join(self.missing)}"
                       if self.missing else ""))
        head = self.state.value.replace("_", " ").upper()
        side = f" -> {self.side.upper()}" if self.side else " -> no side"
        prov = " (provisional -- the bar can still close the other way)" \
            if self.provisional else ""
        return f"{head}{side}{prov}: {self.why}"

    def to_dict(self) -> dict:
        return {"state": self.state.value, "side": self.side,
                "provisional": self.provisional,
                "actionable": self.actionable,
                "meaning": MEANING[self.state],
                "why": self.why, "missing": list(self.missing),
                "describe": self.describe()}


def _away(who: str) -> Side | None:
    """The side OPPOSITE whoever is doing the attacking."""
    if who == "buyers":
        return "sell"
    if who == "sellers":
        return "buy"
    return None


def _with_move(move: float) -> Side | None:
    if move > 0:
        return "buy"
    if move < 0:
        return "sell"
    return None


def classify(lay, previous: "Result | None" = None) -> Result:
    """What this candle is doing, from its six layers.

    `previous` is what the SAME timeframe settled on for the bar before
    this one -- the whole Result, not just its word, because turning
    against something requires knowing which way that something pointed.
    Reversal is the only state that needs it, and without it the state is
    simply unavailable rather than guessed at.

    The order below is most-specific first, and it is the whole algorithm.
    Absorption is checked before continuation because a bar with heavy
    buying and a rising price and a bar with heavy buying and a stuck
    price both satisfy "the buyers are attacking", and only one of them is
    bullish.
    """
    out = Result()
    out.missing = [n for n in REQUIRED
                   if not getattr(getattr(lay, n), "measured", False)]
    if out.missing:
        return out

    intent, force, res = lay.intent, lay.force, lay.resistance
    resp, loc = lay.response, lay.location

    who = intent.who
    move = resp.move
    heavy = force.pace_known and force.pace >= HEAVY
    quiet = force.pace_known and force.pace <= QUIET
    impact = res.impact_ratio if res.measured else None

    # 1. TRADED THROUGH AND CAME BACK. The most specific shape there is,
    #    and the only one that needs the map. Provisional: the bar can
    #    still close back through.
    for ref, side in loc.through():
        if side == "above":
            out.state = State.FAILED_BREAKOUT
            out.side = "sell"
        else:
            out.state = State.FAILED_BREAKDOWN
            out.side = "buy"
        out.provisional = True
        out.why = (f"traded {side} {ref.name} and is back, "
                   f"{abs(ref.distance):.1f}{ref.unit} away")
        return out

    # 2. REACHED IT AND TURNED. Not through -- touched and came back,
    #    which is the bar sitting well off the extreme that did the
    #    touching.
    at = loc.at
    if at is not None and resp.wide:
        if at.above and resp.position <= REJECT_BACK:
            out.state, out.side = State.REJECTION, "sell"
            out.why = (f"reached {at.name} and came back to the bottom "
                       f"{resp.position:.0%} of its own range")
            return out
        if not at.above and resp.position >= 1.0 - REJECT_BACK:
            out.state, out.side = State.REJECTION, "buy"
            out.why = (f"reached {at.name} and came back to the top "
                       f"{resp.position:.0%} of its own range")
            return out

    # 3. REAL SIZE, GOING NOWHERE. Checked before continuation, because
    #    "the buyers are attacking" is true of both and only one of them
    #    is bullish.
    if heavy and impact is not None and impact < LOW_IMPACT:
        if resp.wide:
            # Already travelled, and now the aggression has stopped being
            # paid. That is the move running out rather than being met at
            # a wall -- and whether it has actually ended is not knowable
            # until the bar closes.
            out.state = State.EXHAUSTION
            out.side = _away(who)
            out.provisional = True
            out.why = (f"{who} are {force.pace:.1f}x normal volume after a "
                       f"{abs(move):.1f}{resp.unit} move and price is only "
                       f"moving {impact:.0%} of normal")
            return out
        out.state = State.ABSORPTION
        out.side = _away(who)
        out.why = (f"{who} are {force.pace:.1f}x normal volume and price is "
                   f"moving {impact:.0%} of what that normally moves it")
        return out

    # 4. TURNED AGAINST THE BAR BEFORE IT.
    #
    # A previous bar with no side was not pointing anywhere, and treating
    # that as a direction would manufacture reversals out of quiet.
    if previous is not None and previous.side is not None and move != 0:
        now = _with_move(move)
        if now is not None and now != previous.side:
            out.state, out.side = State.REVERSAL, now
            out.provisional = True
            was = previous.state.value.replace("_", " ")
            out.why = (f"the last bar read {was} to the "
                       f"{previous.side} and this one has gone "
                       f"{move:+.1f}{resp.unit}")
            return out

    # 5. FURTHER THAN THE FLOW JUSTIFIES. A thin book: moves extend, and
    #    they reverse just as easily.
    if impact is not None and impact > HIGH_IMPACT and resp.wide:
        out.state, out.side = State.EXPANSION, _with_move(move)
        out.why = (f"price has gone {move:+.1f}{resp.unit}, "
                   f"{impact:.0%} of what this much flow normally moves it")
        return out

    # 6. GETTING PAID. The aggressor is pressing and price is going their
    #    way -- and has actually gone somewhere FOR A BAR OF THIS LENGTH.
    #
    # The width test is not decoration. Without it, a bar with four fills
    # and a tenth of a tick of drift satisfies "the buyers are crossing
    # and price went up", and reads CONTINUATION with a side on it. That
    # is a confident word built on nothing happening.
    if resp.wide and resp.with_intent and who in ("buyers", "sellers"):
        out.state, out.side = State.CONTINUATION, _with_move(move)
        out.why = (f"{who} are crossing and price has gone "
                   f"{move:+.1f}{resp.unit} with them")
        return out

    # 7. NOTHING HAPPENING.
    if quiet or not resp.wide:
        out.state = State.CONSOLIDATION
        out.why = (f"{force.pace:.1f}x normal volume and "
                   f"{abs(move):.1f}{resp.unit} of movement"
                   if force.pace_known
                   else f"{abs(move):.1f}{resp.unit} of movement")
        return out

    out.why = "the layers do not add up to any of the nine shapes"
    return out


