"""Who is winning this footprint -- one answer, continuously.

    from liqmap.fpread import read_candle
    v = read_candle(candle, scale)
    v.side, v.score, v.strength, v.parts

WHAT THIS IS. The footprint already tags every diagonal imbalance
level by level, and finds the runs of them. What it does not do is
say, in one line, who is winning the bar overall. That is what this
adds: a single verdict, recomputed from scratch on every read, so it
is always describing the bar as it stands right now.

WHAT THIS IS NOT. It does not predict. Five sections of measurement
over 102 hours of ES found nothing in this family that calls where
price goes next, and nothing here changes that. This reports the
state of the auction as the footprint has recorded it, which is a
different and useful job -- the same job a speedometer does.

THE WEIGHTS ARE A CHOICE, NOT A DISCOVERY. Five facts go in, each
measured from the bar and each genuinely in the data. How much each
one counts is editorial: nobody has shown these weights are right,
because "right" would require an outcome to be right ABOUT, and that
outcome is what the study could not find. So every part is published
alongside the verdict with its own number. If the headline disagrees
with the part you trust, trust the part -- and change the weights,
which is why they are an argument.

THE FIVE PARTS, each running -1 (sellers) to +1 (buyers):

  delta      who paid. Signed volume over total volume. The most
             direct fact in a footprint and the hardest to argue
             with.

  flags      the count of diagonal imbalances each way. One tagged
             level is noise; the balance of them across a bar is the
             shape of the fight.

  stacks     runs of consecutive imbalances, weighted by run length.
             Three in a row is a different animal from three
             scattered, and this is the part that answers "overall"
             rather than "every level".

  location   WHERE each side's imbalances sit in the bar's range.
             Buy imbalances up near the high mean buyers are the
             ones operating in new ground; buy imbalances down at
             the low mean they are the ones being hit. Same flags,
             opposite stories, and only position tells them apart.

  follow     whether price went where the flow pushed. Strong buying
             that produced no upward travel is absorption, and this
             is the part that turns the verdict down when effort and
             result disagree.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .footprint import (BUY_IMBALANCE, DEFAULT_IMBALANCE, ImbalanceConfig,
                        TickScale)

BUYERS, SELLERS, BALANCED = "buyers", "sellers", "balanced"

# How much each part counts toward the headline. An argument, not a
# finding -- see the module docstring. They sum to 1.0 so the score
# stays inside -1..+1 and can be read as a fraction.
WEIGHTS = {
    "delta": 0.30,
    "flags": 0.20,
    "stacks": 0.25,
    "location": 0.15,
    "follow": 0.10,
}

# Where the verdict stops being a shrug. Below `weak` the two sides
# are close enough that calling a winner is a coin flip dressed up.
BANDS = ((0.45, "strong"), (0.22, "moderate"), (0.08, "weak"))


@dataclass(frozen=True, slots=True)
class Verdict:
    """Who is winning, how strongly, and what that was built from."""

    side: str = BALANCED
    score: float = 0.0              # -1 sellers .. +1 buyers
    strength: str = "none"
    parts: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)
    # the raw counts behind the parts, so nothing is unexplained
    buy_flags: int = 0
    sell_flags: int = 0
    buy_stacks: int = 0
    sell_stacks: int = 0
    delta: int = 0
    volume: int = 0
    levels: int = 0
    span_ticks: int = 0
    note: str = ""

    def as_dict(self) -> dict:
        return {"side": self.side, "score": round(self.score, 4),
                "strength": self.strength,
                "parts": {k: round(v, 4) for k, v in self.parts.items()},
                "weights": self.weights,
                "buy_flags": self.buy_flags, "sell_flags": self.sell_flags,
                "buy_stacks": self.buy_stacks,
                "sell_stacks": self.sell_stacks,
                "delta": self.delta, "volume": self.volume,
                "levels": self.levels, "span_ticks": self.span_ticks,
                "note": self.note}


def _band(score: float) -> str:
    a = abs(score)
    for cut, name in BANDS:
        if a >= cut:
            return name
    return "none"


def _ratio(up: float, down: float) -> float:
    """Signed share, -1..+1, and 0 when there is nothing of either."""
    t = up + down
    return ((up - down) / t) if t > 0 else 0.0


def read_candle(candle, scale: TickScale | None = None,
                cfg: ImbalanceConfig = DEFAULT_IMBALANCE,
                run: int = 3,
                weights: dict | None = None) -> Verdict:
    """One verdict for this bar, computed fresh from the bar itself.

    Takes a FootprintCandle -- live or closed, they are the same type
    -- so a forming bar and a finished one are read identically. No
    state is carried between calls, which is what makes it safe to
    poll at any rate and from any thread.
    """
    w = dict(WEIGHTS)
    if weights:
        w.update(weights)

    if candle is None or candle.empty:
        return Verdict(weights=w, note="no prints in this bar yet")

    rows = candle.levels
    lo = min(rows)
    hi = max(rows)
    span = hi - lo

    flags = candle.imbalances(cfg)
    buy_f = [f for f in flags if f.side == BUY_IMBALANCE]
    sell_f = [f for f in flags if f.side != BUY_IMBALANCE]

    runs = candle.stacked(cfg, run=run)
    buy_runs = [r for r in runs if r[0] == BUY_IMBALANCE]
    sell_runs = [r for r in runs if r[0] != BUY_IMBALANCE]
    # Weighted by LENGTH: a run of six is not two runs of three.
    buy_len = sum(r[2] - r[1] + 1 for r in buy_runs)
    sell_len = sum(r[2] - r[1] + 1 for r in sell_runs)

    vol = candle.total_volume
    parts = {
        "delta": (candle.delta / vol) if vol > 0 else 0.0,
        "flags": _ratio(len(buy_f), len(sell_f)),
        "stacks": _ratio(buy_len, sell_len),
        "location": 0.0,
        "follow": 0.0,
    }

    # WHERE the imbalances sit. Position runs 0 at the bar's low to 1
    # at its high; a side operating up near the high is the side with
    # the initiative. On a one-tick bar there is no range to speak of
    # and this part stays out of it rather than inventing a position.
    if span > 0 and (buy_f or sell_f):
        def pos(fs):
            if not fs:
                return None
            return sum((f.price_ticks - lo) / span for f in fs) / len(fs)
        bp, sp = pos(buy_f), pos(sell_f)
        # CONTRAST ONLY. With flags on both sides, "who is operating
        # higher in the range" is a real comparison. With flags on one
        # side it is not: sell imbalances low in the bar and sell
        # imbalances high in the bar are both sellers, and squeezing
        # that into a signed number invented a reading -- an early
        # version scored a seller-dominated bar's geometry at +0.25,
        # bullish, for exactly that reason. One-sided bars get nothing
        # here and let `flags` carry the direction, which it already
        # does correctly.
        if bp is not None and sp is not None:
            parts["location"] = max(-1.0, min(1.0, (bp - sp) * 2.0))

    # Did price go where the flow pushed? Close against open over the
    # bar's own range. Strong buying with no upward travel is
    # absorption, and this is the part that says so.
    if span > 0:
        travel = (candle.close_ticks - candle.open_ticks) / span
        parts["follow"] = max(-1.0, min(1.0, travel))

    score = sum(parts[k] * w.get(k, 0.0) for k in parts)
    score = max(-1.0, min(1.0, score))
    strength = _band(score)
    side = (BALANCED if strength == "none"
            else (BUYERS if score > 0 else SELLERS))

    # ABSORPTION CAPS THE HEADLINE. One side paying hard while price
    # refuses to go anywhere is the clearest thing a footprint shows,
    # and calling it "strong buyers" because the delta is large reads
    # the effort and ignores the result. The parts stay visible; it is
    # the one-word summary that gets held back.
    absorbed = (span > 0 and abs(parts["delta"]) >= 0.15
                and (abs(parts["follow"]) < 0.15
                     or parts["delta"] * parts["follow"] < 0))

    note = ""
    if not flags:
        note = "no diagonal imbalances yet; the verdict is delta alone"
    elif span == 0:
        note = "one-tick bar: position and follow-through have no range"
    if absorbed:
        if strength == "strong":
            strength = "moderate"
        who = "buyers" if parts["delta"] > 0 else "sellers"
        note = (f"{who} are paying and price is not following -- "
                f"effort without result, which is what absorption "
                f"looks like")
    elif not note and (buy_f and not sell_f or sell_f and not buy_f):
        note = "imbalances on one side only; position is not counted"

    return Verdict(
        side=side, score=score, strength=strength, parts=parts, weights=w,
        buy_flags=len(buy_f), sell_flags=len(sell_f),
        buy_stacks=len(buy_runs), sell_stacks=len(sell_runs),
        delta=candle.delta, volume=vol, levels=len(rows),
        span_ticks=span, note=note)


class FootprintReader:
    """The verdict on the live bar, and on the bars behind it.

    Holds no accumulated state of its own: every verdict is computed
    from the candle it describes. A reader that accumulated would
    drift from the footprint it is supposed to be reading, and the
    whole point is that it always agrees with the chart.
    """

    def __init__(self, cfg: ImbalanceConfig = DEFAULT_IMBALANCE,
                 run: int = 3, weights: dict | None = None,
                 history: int = 40) -> None:
        self.cfg = cfg
        self.run = int(run)
        self.weights = dict(WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.history = int(history)

    @staticmethod
    def _forming(stream):
        """The bar still open, whatever the caller calls it.

        FootprintStream keeps it on its engine as `_engine.current`;
        a FootprintEngine has `current` directly; a test double may
        simply expose `live`. Guessing one name and getting a silent
        "no live bar" is how a reader ends up quietly describing
        nothing -- which is exactly what happened first time.
        """
        for path in ("live", "current"):
            bar = getattr(stream, path, None)
            if bar is not None:
                return bar
        eng = getattr(stream, "_engine", None)
        return getattr(eng, "current", None) if eng is not None else None

    def live(self, stream) -> Verdict:
        """The bar still forming."""
        bar = self._forming(stream)
        if bar is None:
            return Verdict(weights=self.weights, note="no live bar")
        return read_candle(bar, cfg=self.cfg, run=self.run,
                           weights=self.weights)

    def closed(self, stream) -> list:
        """The finished bars, oldest first."""
        hist = list(getattr(stream, "history", ()) or ())
        out = []
        for bar in hist[-self.history:]:
            out.append(read_candle(bar, cfg=self.cfg, run=self.run,
                                   weights=self.weights))
        return out

    def as_dict(self, stream) -> dict:
        now = self.live(stream)
        past = self.closed(stream)
        # How long the same side has held across closed bars. A side
        # that has won six bars running is a different statement from
        # one that has just taken this one.
        held = 0
        for v in reversed(past):
            if v.side == now.side and now.side != BALANCED:
                held += 1
            else:
                break
        return {"now": now.as_dict(),
                "history": [v.as_dict() for v in past],
                "bars_held": held,
                "weights": self.weights,
                "run": self.run}
