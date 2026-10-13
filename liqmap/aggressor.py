"""Which side crossed the spread, and whether a feed's label can be trusted.

THE FIELD EVERYTHING ELSE IS BUILT ON

Delta, CVD, intent, absorption, the efficiency check, the trajectory count
-- every one of them starts from "who hit whom". If that is backwards,
nothing in this project crashes. Every number simply reads with its sign
inverted, absorption reads as initiative, and the whole engine is
confidently wrong in a way no test on the arithmetic could catch.

So a new feed is not trusted about this. It is measured.

THE LABEL IS NOT ASSUMED TO MEAN ANYTHING

This is the part that matters, and it is why this file exists rather than a
dictionary mapping strings to sides.

Feeds label trades `B`/`S`, `BUY`/`SELL`, `0`/`1`, `Bid`/`Ask`, and the
last of those is genuinely ambiguous: `Ask` can mean the aggressor LIFTED
the ask -- a buy -- or that the resting side was the ask, which is the same
event seen from the other end. A dictionary has to guess. Documentation
frequently does not say, and when it does it is sometimes wrong about its
own encoder.

So the two labels are kept as opaque buckets, A and B, with no meaning
attached, and the tape is asked which is which:

    a crossing buyer lifts the offer, so buyer-marked trades print
    above the mid

Count how often each bucket prints above the mid and the convention falls
out. No documentation required, and the file answers the question about
itself.

TRADES AT THE MID ARE DISCARDED, NOT SPLIT

In a one-tick-wide market they are much of the tape and they say nothing
about the convention either way. Counting them as half-agreement drags
every verdict towards the middle, which is exactly where "unclear" lives.

AND A FEED WITH NO LABEL AT ALL IS A DIFFERENT ANSWER

Not a failure -- a different situation, and one worth quantifying rather
than just reporting. If the side has to be inferred from the quote, what
bounds the quality of that inference is how many trades print strictly off
the mid, because the ones at the mid cannot be assigned at all. That
fraction is the ceiling on how good an inferred delta could ever be, and
it is reported whether or not a label exists.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Verdict = Literal["a_is_buyer", "b_is_buyer", "unclear",
                  "no label", "not enough samples"]

# Below this there is nothing to conclude. A convention read off twenty
# prints is a coin flip wearing a decimal point.
MIN_SAMPLES = 40

# How lopsided the counts have to be before the convention is called.
# Deliberately wide on both sides: a real convention is near-total, and
# anything in between is a feed that needs looking at by hand rather than
# a close call to be resolved by a threshold.
CLEAR = 0.80


def _norm(side: Any) -> str | None:
    """A trade's side label as an opaque bucket: "a", "b", or unknown.

    NO MEANING IS ATTACHED. The point of this file is that `Ask` could be
    the aggressor or the resting side and the label cannot settle it, so
    the two values are separated and left to be identified by the tape.

    The buckets are assigned by the raw value, consistently, so the same
    string always lands in the same bucket.
    """
    if side is None:
        return None
    s = str(side).strip().lower()
    if not s:
        return None
    if s in ("b", "buy", "bid", "bought", "0", "+1", "1b", "buyer"):
        return "b"
    if s in ("s", "a", "sell", "ask", "offer", "sold", "1", "-1", "seller"):
        return "a"
    return None


@dataclass
class Report:
    """What the tape said about its own convention."""

    n: int = 0                  # labelled prints off the mid, which decide it
    b_above: int = 0            # bucket-b prints above the mid
    a_above: int = 0
    b_total: int = 0
    a_total: int = 0
    seen: int = 0               # every print offered, labelled or not
    off_mid: int = 0            # prints strictly off the mid
    quotable: int = 0           # prints with a usable book beside them
    unlabelled: int = 0

    @property
    def b_share(self) -> float:
        """How often bucket b printed above the mid."""
        return (self.b_above / self.b_total) if self.b_total else 0.0

    @property
    def a_share(self) -> float:
        return (self.a_above / self.a_total) if self.a_total else 0.0

    @property
    def assignable(self) -> float:
        """The ceiling on an INFERRED delta.

        Trades at the mid cannot be assigned from the quote at all, so
        this is the best a feed with no side field could ever do -- before
        any error in assigning the rest.
        """
        return (self.off_mid / self.quotable) if self.quotable else 0.0

    @property
    def verdict(self) -> Verdict:
        if self.seen and self.unlabelled == self.seen:
            return "no label"
        if self.n < MIN_SAMPLES or not self.b_total or not self.a_total:
            return "not enough samples"
        # Both buckets are consulted. One of them being lopsided could be
        # an artefact of a thin sample; the two disagreeing with each
        # other is the convention.
        if self.b_share >= CLEAR and self.a_share <= 1.0 - CLEAR:
            return "b_is_buyer"
        if self.a_share >= CLEAR and self.b_share <= 1.0 - CLEAR:
            return "a_is_buyer"
        return "unclear"

    @property
    def settled(self) -> bool:
        return self.verdict in ("a_is_buyer", "b_is_buyer")

    def buyer_label(self) -> str | None:
        """Which bucket is the aggressive buyer, once it is known."""
        v = self.verdict
        return "b" if v == "b_is_buyer" else "a" if v == "a_is_buyer" else None

    def describe(self) -> str:
        v = self.verdict
        if v == "no label":
            return (f"no side field on any of {self.seen} prints -- the "
                    f"aggressor would have to be inferred, and only "
                    f"{self.assignable:.0%} of prints are off the mid and "
                    f"assignable at all")
        if v == "not enough samples":
            return (f"only {self.n} labelled prints off the mid -- need "
                    f"{MIN_SAMPLES} before the convention means anything")
        if v == "unclear":
            return (f"the labels do not line up with the book: 'b' prints "
                    f"above the mid {self.b_share:.0%} of the time and 'a' "
                    f"{self.a_share:.0%} -- do not trust either until this "
                    f"is understood")
        who = self.buyer_label()
        share = self.b_share if who == "b" else self.a_share
        return (f"'{who}' is the aggressive buyer -- those prints are above "
                f"the mid {share:.0%} of the time across {self.n} samples")

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "settled": self.settled,
                "buyer": self.buyer_label(), "n": self.n, "seen": self.seen,
                "b_above": self.b_above, "b_total": self.b_total,
                "a_above": self.a_above, "a_total": self.a_total,
                "b_share": round(self.b_share, 4),
                "a_share": round(self.a_share, 4),
                "unlabelled": self.unlabelled,
                "off_mid": self.off_mid, "quotable": self.quotable,
                "assignable": round(self.assignable, 4),
                "describe": self.describe()}


@dataclass
class Convention:
    """Accumulates prints and works out what the feed's labels mean.

    Venue-neutral on purpose: it takes four numbers and a label, so the
    same check runs over a DBN file, a broker websocket, or anything else
    that ever gets added.
    """

    keep: int = 4000
    r: Report = field(default_factory=Report)

    def observe(self, px: float, side: Any = None,
                bid: float = 0.0, ask: float = 0.0) -> None:
        """One print, with the book as it stood IMMEDIATELY BEFORE it.

        Before, not after. A book sampled after the fill already shows the
        consequence of the trade -- the level that was just taken may be
        gone -- and a mid computed from it drifts towards the aggressor,
        which makes any convention look correct.
        """
        if self.r.seen >= self.keep:
            return
        px = float(px)
        if px <= 0:
            return
        self.r.seen += 1

        bucket = _norm(side)
        if bucket is None:
            self.r.unlabelled += 1

        bid, ask = float(bid), float(ask)
        if bid <= 0 or ask <= 0 or ask < bid:
            return
        mid = (bid + ask) / 2.0
        self.r.quotable += 1
        if px == mid:
            return
        self.r.off_mid += 1

        if bucket is None:
            return
        above = px > mid
        self.r.n += 1
        if bucket == "b":
            self.r.b_total += 1
            self.r.b_above += int(above)
        else:
            self.r.a_total += 1
            self.r.a_above += int(above)

    def report(self) -> Report:
        return self.r
