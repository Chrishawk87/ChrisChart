"""The confirmation ladder: each timeframe judged by the faster ones.

THE RULE

A timeframe does not get to call a trade on its own. It prints only when
it and the faster timeframes named below it all point the same way:

    5m    1m + 5m
    15m   1m + 5m + 15m
    1h    5m + 15m + 1h
    4h    15m + 1h + 4h

The one-minute is the devil's advocate for the five and the fifteen and
then drops out -- it is too fast to have an opinion about an hour, and
letting it keep voting up the stack would mean one noisy minute could
silence a signal built from hours of agreement. The one-minute row itself
never prints: it argues, it does not decide.

WHAT COUNTS AS A VOTE

The SIGN OF THE MOVE, not the winning side. Those two can disagree --
`pressure` inverts the side on absorption, so a bar can read BUYERS while
its move is negative -- and for this ladder the move is what counts. A
timeframe votes sell at its threshold below zero, buy at its threshold
above, and abstains in between.

An abstention is not an agreement. A set containing one timeframe too
quiet to vote prints nothing, which is the whole point: the 1m sitting at
-0.52 when the bar is 1.0 means there is no trade, not that the 1m is
neutral about it.

THE THRESHOLD SCALES, AND IT HAS TO

One bar of this project's history is the argument for it. A fixed 2bps
band for "price did not move" was applied to every timeframe at once and
inverted most of the ladder, because 2bps is a large minute and a
motionless four hours. The same mistake is available here: 1.0bps is a
real move on a one-minute bar and noise on a four-hour one, so a flat
threshold would let the slow rows wave every set through.

So the bar is 1.0 on the ONE-MINUTE bar and scales by how far bars of
each length actually travel in this market -- the same figure the flat
band learns. With nothing learned yet it falls back to the square root of
the interval, which is how a random walk's distance grows.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER. It prints a word.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

# Which timeframes must agree before each one prints. Order matters only
# for reading; every member is required.
CONFIRMERS: dict[str, tuple[str, ...]] = {
    "5m": ("1m", "5m"),
    "15m": ("1m", "5m", "15m"),
    "1h": ("5m", "15m", "1h"),
    "4h": ("15m", "1h", "4h"),
}

# The bar a ONE-MINUTE bar has to clear, in bps. Every other timeframe's
# threshold is this scaled up.
BASE_BPS = 1.0
ANCHOR_S = 60.0

Side = Literal["buy", "sell"]


@dataclass
class Vote:
    """One timeframe's opinion, and whether it was strong enough to hold."""

    timeframe: str
    interval_s: float = 0.0
    bps: float = 0.0
    threshold: float = BASE_BPS
    side: Side | None = None
    present: bool = True
    scaled: bool = False        # threshold learned rather than assumed

    @property
    def abstains(self) -> bool:
        return self.side is None

    def describe(self) -> str:
        if not self.present:
            return f"{self.timeframe}: not read"
        if self.side is None:
            return (f"{self.timeframe}: {self.bps:+.2f}bps, inside its "
                    f"{self.threshold:.2f} bar -- no vote")
        return (f"{self.timeframe}: {self.bps:+.2f}bps past "
                f"{self.threshold:.2f} -- {self.side}")

    def to_dict(self) -> dict:
        return {"timeframe": self.timeframe, "bps": round(self.bps, 3),
                "threshold": round(self.threshold, 3), "side": self.side,
                "present": self.present, "abstains": self.abstains,
                "scaled": self.scaled, "describe": self.describe()}


@dataclass
class Verdict:
    """What one timeframe prints, and what it took to get there."""

    timeframe: str
    side: Side | None = None
    votes: list[Vote] = field(default_factory=list)
    why: str = ""

    @property
    def agreed(self) -> bool:
        return self.side is not None

    def to_dict(self) -> dict:
        return {"timeframe": self.timeframe, "side": self.side,
                "agreed": self.agreed, "why": self.why,
                "votes": [v.to_dict() for v in self.votes]}


def threshold_for(interval_s: float, typical_bps: float = 0.0,
                  anchor_typical: float = 0.0, base: float = BASE_BPS,
                  anchor_s: float = ANCHOR_S) -> tuple[float, bool]:
    """The bar this timeframe has to clear, and whether it was learned.

    Anchored on the one-minute so that `base` means exactly what it says
    there, and scaled by how far bars of this length actually move here.
    Falls back to the square root of the interval, which is how far a
    random walk gets in a longer stretch of time.

    Never below `base`. A slower bar that happens to have been quieter
    than the minute lately must not become the EASIEST row to satisfy --
    that would let the slowest timeframe, the one with the most weight in
    the stack, be waved through on the least evidence.
    """
    if typical_bps > 0 and anchor_typical > 0:
        out = base * (typical_bps / anchor_typical)
        learned = True
    elif interval_s > 0 and anchor_s > 0:
        out = base * (interval_s / anchor_s) ** 0.5
        learned = False
    else:
        return base, False
    if interval_s >= anchor_s:
        out = max(base, out)
    return out, learned


def vote_on(timeframe: str, interval_s: float, bps: float,
            typical_bps: float = 0.0, anchor_typical: float = 0.0,
            base: float = BASE_BPS) -> Vote:
    """Which way this timeframe votes, from the SIGN of its move.

    Not from the winning side: `pressure` inverts that on absorption, so a
    bar can read BUYERS while price is down, and here the move decides.
    """
    t, learned = threshold_for(interval_s, typical_bps, anchor_typical, base)
    side: Side | None = None
    if bps >= t:
        side = "buy"
    elif bps <= -t:
        side = "sell"
    return Vote(timeframe=timeframe, interval_s=interval_s, bps=bps,
                threshold=t, side=side, present=True, scaled=learned)


def verdict_for(timeframe: str, votes: dict[str, Vote]) -> Verdict:
    """What this timeframe prints, given everyone's vote.

    Every confirmer must be present AND must have voted AND must have
    voted the same way. A missing timeframe is not a silent yes, and an
    abstention is not an agreement.
    """
    out = Verdict(timeframe=timeframe)
    want = CONFIRMERS.get(timeframe)
    if not want:
        out.why = f"{timeframe} is not a timeframe that prints"
        return out

    out.votes = [votes.get(n) or Vote(timeframe=n, present=False)
                 for n in want]

    missing = [v.timeframe for v in out.votes if not v.present]
    if missing:
        out.why = (f"no read on {', '.join(missing)} -- "
                   f"{timeframe} cannot be confirmed")
        return out

    quiet = [v for v in out.votes if v.abstains]
    if quiet:
        out.why = ("; ".join(v.describe() for v in quiet)
                   + f" -- no trade on {timeframe}")
        return out

    sides = {v.side for v in out.votes}
    if len(sides) > 1:
        out.why = (f"{', '.join(v.timeframe + ' ' + str(v.side) for v in out.votes)}"
                   f" -- they disagree, no trade on {timeframe}")
        return out

    out.side = out.votes[0].side
    out.why = (f"{', '.join(v.timeframe for v in out.votes)} all "
               f"{out.side} -- {out.side.upper()} on {timeframe}")
    return out


@dataclass
class Row:
    """What this module needs to know about one timeframe."""

    timeframe: str
    interval_s: float
    bps: float
    typical_bps: float = 0.0


def read(rows: Sequence[Row], base: float = BASE_BPS,
         anchor: str = "1m") -> dict:
    """Every timeframe's vote, and the verdict for the ones that print."""
    by_tf = {r.timeframe: r for r in rows}
    anchor_row = by_tf.get(anchor)
    anchor_typical = anchor_row.typical_bps if anchor_row else 0.0

    votes = {r.timeframe: vote_on(r.timeframe, r.interval_s, r.bps,
                                  r.typical_bps, anchor_typical, base)
             for r in rows}
    verdicts = {tf: verdict_for(tf, votes) for tf in CONFIRMERS}
    return {"base_bps": base, "anchor": anchor,
            "votes": {k: v.to_dict() for k, v in votes.items()},
            "verdicts": {k: v.to_dict() for k, v in verdicts.items()}}
