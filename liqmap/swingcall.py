"""Which swing forms next, and whether it comes in above or below the last.

WHAT IS BEING PREDICTED

Structure alternates. After a confirmed high the next swing is a low, and
after a confirmed low the next is a high. So there is only ever one
question open:

    next swing is a HIGH -> does it come in above the last high (HH) or
                            below it (LH)
    next swing is a LOW  -> above the last low (HL) or below it (LL)

The order-flow engine supplies the lean; this file supplies the structural
frame it has to land in, and the two prices that decide it.

THE PART THAT IS NOT A PREDICTION AT ALL

Some of the answer is frequently already determined and merely unconfirmed.
If price has ALREADY traded above the previous high since the last swing,
then whenever the next high confirms it is going to be a higher high --
there is nothing left to forecast, only to wait for. That is reported as
`settled`, and keeping it apart from a genuine prediction matters more
than almost anything else here: a tool that counts those among its correct
calls is scoring itself on the past.

The asymmetry is worth stating. Exceeding the level settles HH (or LL);
NOT exceeding it settles nothing, because price has the rest of the swing
to go and do it.

ONLY SWINGS THAT WERE KNOWABLE

`structure.swings` stamps each pivot with `confirmed_at` -- the bar at
which it could actually have been seen, which is `strength` bars after the
bar it sits on. Reading a swing before that is lookahead, and it is the
single most common way a structure backtest lies. Everything here filters
on `confirmed_at <= the bar in progress`, which means the most recent
pivot is usually several bars old. That is not a defect; it is what was
actually knowable.

AND THE CALL NAMES WHAT WOULD KILL IT

Every call carries the price it needs and the price that breaks it, so it
can be scored rather than argued about. A direction with no invalidation
is not a prediction, it is an opinion.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

Kind = Literal["high", "low"]
Call = Literal["HH", "LH", "HL", "LL"]
Side = Literal["buy", "sell"]

# Swings needed before structure can be read at all: one of each, so there
# is both a thing to form next and a level to measure it against.
MIN_SWINGS = 2


@dataclass
class SwingCall:
    """The next swing: which kind, which side of the last one, and why."""

    next_kind: Kind | None = None
    call: Call | None = None
    settled: bool = False          # already determined, waiting to confirm
    reference: float = 0.0         # the prior swing of the same kind
    reference_ts: float = 0.0
    needs: float = 0.0             # price that has to be taken out
    fails_at: float = 0.0          # price that would break the call
    running: float = 0.0           # the extreme since the last swing
    confirm_bars: int = 0          # bars before it can be known, once formed
    side: Side | None = None       # what the order flow was leaning
    why: str = ""

    @property
    def predicted(self) -> bool:
        """A call about the future, as opposed to one already determined."""
        return self.call is not None and not self.settled

    def describe(self) -> str:
        if self.call is None:
            return self.why or "not enough structure to call the next swing"
        if self.settled:
            return (f"{self.call} is already set -- price took "
                    f"{self.reference:,.4f} and the swing has not confirmed "
                    f"yet ({self.confirm_bars} bars to confirm)")
        return (f"{self.call} next: needs {self.needs:,.4f}, fails below "
                f"{self.fails_at:,.4f}" if self.call in ("HH", "HL")
                else f"{self.call} next: fails above {self.fails_at:,.4f}")

    def to_dict(self) -> dict:
        return {"next_kind": self.next_kind, "call": self.call,
                "settled": self.settled, "predicted": self.predicted,
                "reference": round(self.reference, 8),
                "reference_ts": self.reference_ts,
                "needs": round(self.needs, 8),
                "fails_at": round(self.fails_at, 8),
                "running": round(self.running, 8),
                "confirm_bars": self.confirm_bars, "side": self.side,
                "why": self.why, "describe": self.describe()}


def knowable(sw: Sequence, at_index: int) -> list:
    """The swings that could actually have been seen by `at_index`.

    The filter that keeps this honest. A pivot sits on its bar but is not
    visible until `strength` bars later, so the newest usable swing is
    always several bars old.
    """
    return [s for s in sw if s.confirmed_at <= at_index]


def call_next(candles: Sequence, sw: Sequence, side: Side | None = None,
              strength: int = 3) -> SwingCall:
    """What the next swing will be, from structure that was knowable.

    `sw` are swings from `structure.swings`; `side` is what the order-flow
    engine is leaning, used only when the answer is not already determined.
    """
    out = SwingCall(side=side, confirm_bars=int(strength))
    if not candles:
        out.why = "no candles"
        return out

    last_index = len(candles) - 1
    seen = knowable(sw, last_index)
    highs = [s for s in seen if s.is_high]
    lows = [s for s in seen if not s.is_high]
    if len(seen) < MIN_SWINGS:
        out.why = (f"only {len(seen)} confirmed swing"
                   f"{'' if len(seen) == 1 else 's'} -- there is no "
                   f"structure to extend yet")
        return out
    if not highs or not lows:
        # A run of pivots all the same kind. Real enough -- a straight
        # trend on a short window does it -- and saying "not enough
        # swings" when there are twenty of them sends you looking in the
        # wrong place entirely.
        missing = "lows" if not lows else "highs"
        out.why = (f"{len(seen)} confirmed swings but no {missing} among "
                   f"them -- nothing to measure the next one against")
        return out

    last = max(seen, key=lambda s: (s.confirmed_at, s.index))
    # Structure alternates: a high is followed by a low and the reverse.
    out.next_kind = "low" if last.is_high else "high"
    prior = (highs if out.next_kind == "high" else lows)[-1]
    out.reference = prior.px
    out.reference_ts = prior.ts

    # The extreme since the last confirmed pivot formed. Measured from the
    # bar AFTER it, because the pivot's own bar belongs to the swing that
    # has already been counted.
    after = candles[last.index + 1:]
    if not after:
        out.why = "no bars since the last confirmed swing"
        return out

    if out.next_kind == "high":
        out.running = max(c.high for c in after)
        out.needs, out.fails_at = prior.px, lows[-1].px
        if out.running > prior.px:
            # ALREADY DONE. Whenever the next high confirms it is going to
            # be a higher one; there is nothing left to forecast.
            out.call, out.settled = "HH", True
            out.why = (f"price has already traded {out.running:,.4f} above "
                       f"the prior high at {prior.px:,.4f}")
            return out
    else:
        out.running = min(c.low for c in after)
        out.needs, out.fails_at = prior.px, highs[-1].px
        if out.running < prior.px:
            out.call, out.settled = "LL", True
            out.why = (f"price has already traded {out.running:,.4f} below "
                       f"the prior low at {prior.px:,.4f}")
            return out

    if side is None:
        out.why = ("the level is untouched and the order flow has no side "
                   "-- nothing to call the next swing from")
        return out

    if out.next_kind == "high":
        out.call = "HH" if side == "buy" else "LH"
    else:
        out.call = "HL" if side == "buy" else "LL"
    out.why = (f"next swing is a {out.next_kind}, the prior one is at "
               f"{prior.px:,.4f}, and the flow is leaning {side}")
    return out


# ---------------------------------------------------------------------------
# scoring, for when there are enough of them to count
# ---------------------------------------------------------------------------


@dataclass
class Scored:
    """A call, and the swing that answered it."""

    call: Call
    actual: Call
    settled: bool

    @property
    def correct(self) -> bool:
        return self.call == self.actual


def actual_call(swing, prior_px: float) -> Call:
    """What a confirmed swing turned out to be against its prior level."""
    if swing.is_high:
        return "HH" if swing.px > prior_px else "LH"
    return "HL" if swing.px > prior_px else "LL"


@dataclass
class Rate:
    """A hit rate, with the sample size that produced it.

    N RIDES WITH IT, ALWAYS. Seven out of ten is 70% and is also noise
    wearing a decimal point, and the two are indistinguishable once the
    count is dropped.

    `settled` calls are counted SEPARATELY and never folded into the
    predicted rate. They were already determined when they were made, so
    including them measures how often the tool can read the present.
    """

    hits: int = 0
    n: int = 0
    settled_hits: int = 0
    settled_n: int = 0

    @property
    def rate(self) -> float | None:
        return (self.hits / self.n) if self.n else None

    def add(self, s: Scored) -> None:
        if s.settled:
            self.settled_n += 1
            self.settled_hits += int(s.correct)
        else:
            self.n += 1
            self.hits += int(s.correct)

    def describe(self) -> str:
        if not self.n:
            return (f"no predicted swings scored yet"
                    + (f" ({self.settled_n} were already set when called)"
                       if self.settled_n else ""))
        return (f"{self.rate:.0%} of {self.n} predicted swings"
                + (f"; {self.settled_n} more were already set when called"
                   if self.settled_n else ""))

    def to_dict(self) -> dict:
        return {"hits": self.hits, "n": self.n, "rate": self.rate,
                "settled_hits": self.settled_hits, "settled_n": self.settled_n,
                "describe": self.describe()}


def score(calls: Sequence[tuple[SwingCall, object, float]]) -> Rate:
    """Score calls against the swings that answered them.

    Each entry is (the call, the swing that formed, the prior level it is
    measured against).
    """
    out = Rate()
    for c, swing, prior_px in calls:
        if c.call is None:
            continue
        out.add(Scored(call=c.call, actual=actual_call(swing, prior_px),
                       settled=c.settled))
    return out
