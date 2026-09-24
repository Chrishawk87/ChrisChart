"""Three columns, one direction. No gates, no thresholds, no opinions.

WHAT THIS REPLACES

`suggest.py` refuses most candles. It has a dozen reasons to say no --
conviction floors, breakeven ceilings, cost multiples, agreement age -- and
every one of them is a guess I made about what a good trade looks like,
applied before any evidence existed to support it. Some are probably right.
The problem is that nobody can tell which, because a filtered sample cannot
measure its own filter: the trades it refused have no outcome, so the
refusals are unfalsifiable by construction.

This does the opposite. Book, delta and price each point somewhere. Add
them up. Whichever way the sum points is the direction, and that is the
entire decision.

    net = book + delta + price       (signed by direction, sized by strength)

    net > 0  ->  long
    net < 0  ->  short
    net == 0 ->  nothing, because there is nothing to read

Two agreeing is a trade. One column carrying two flat ones is a trade. Two
against one is a trade in the direction of the two. None of that is decided
here -- it falls out of the arithmetic, and `agreeing` is recorded alongside
so the scorecard can tell you afterwards which of those shapes actually
paid. That is the point: the question moves out of my assumptions and into
your data.

THE ONE THING THAT IS NOT A GATE

All three flat produces no signal. That is not a filter refusing a trade,
it is the absence of a reading -- there is no direction to be long or short
of. Everything else trades.

STRENGTH, NOT JUST DIRECTION

A column screaming and a column whispering are not the same vote, so the
sum is weighted by each column's strength. This still never blocks
anything: a weak-but-unanimous read and a strong lone voice both produce a
side. It only decides WHICH side when columns disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Direction = Literal["up", "down", "flat"]
Side = Literal["long", "short"]

COLUMNS = ("book", "delta", "price")


def _signed(direction: str | None, strength: float) -> float:
    if direction == "up":
        return abs(strength)
    if direction == "down":
        return -abs(strength)
    return 0.0


@dataclass(frozen=True)
class Vote:
    """What the three columns add up to, and how they split getting there."""

    side: Side | None
    net: float
    book: float          # each column's signed contribution
    delta: float
    price: float
    agreeing: int        # columns pointing the way the sum points
    against: int         # columns pointing the other way
    flat: int            # columns saying nothing

    @property
    def direction(self) -> Direction:
        return "up" if self.side == "long" else "down" if self.side else "flat"

    @property
    def unanimous(self) -> bool:
        return self.agreeing == 3

    @property
    def split(self) -> bool:
        """Columns actively disagreeing, as opposed to some staying quiet."""
        return self.against > 0

    def shape(self) -> str:
        """The vote in the form the ledger slices on: '2-1', '3-0', '1-0'."""
        return f"{self.agreeing}-{self.against}"

    def describe(self) -> str:
        if self.side is None:
            return "nothing to read — all three columns flat"
        parts = []
        for name, v in (("book", self.book), ("delta", self.delta),
                        ("price", self.price)):
            word = "up" if v > 0 else "down" if v < 0 else "flat"
            parts.append(f"{name} {word}" + (f" {abs(v):.2f}" if v else ""))
        return (f"{self.side.upper()} on {self.shape()} — " + ", ".join(parts)
                + f" (net {self.net:+.2f})")

    def to_dict(self) -> dict[str, Any]:
        return {"side": self.side, "direction": self.direction,
                "net": round(self.net, 4), "book": round(self.book, 4),
                "delta": round(self.delta, 4), "price": round(self.price, 4),
                "agreeing": self.agreeing, "against": self.against,
                "flat": self.flat, "shape": self.shape(),
                "unanimous": self.unanimous, "split": self.split,
                "describe": self.describe()}


def cast(book_dir: str | None, book_strength: float,
         delta_dir: str | None, delta_strength: float,
         price_dir: str | None, price_strength: float) -> Vote:
    """Add up the three columns. Nothing is refused."""
    b = _signed(book_dir, book_strength)
    d = _signed(delta_dir, delta_strength)
    p = _signed(price_dir, price_strength)
    net = b + d + p

    side: Side | None = "long" if net > 0 else "short" if net < 0 else None

    # A dead-even split (one up, one down, one flat, equal strength) reads
    # as nothing, which is honest: the columns cancelled.
    want = 1.0 if net > 0 else -1.0 if net < 0 else 0.0
    agreeing = sum(1 for v in (b, d, p) if want and v * want > 0)
    against = sum(1 for v in (b, d, p) if want and v * want < 0)
    flat = sum(1 for v in (b, d, p) if v == 0)

    return Vote(side=side, net=net, book=b, delta=d, price=p,
                agreeing=agreeing, against=against, flat=flat)


def from_payload(payload: dict) -> Vote:
    """Cast a vote from the payload the panel already renders.

    Reads the same three columns the screen shows. Strengths come from each
    column's own score, so a column is never given a say it did not earn --
    and never denied one it did.
    """
    three = payload.get("three_way") or {}
    conf = payload.get("confirmation") or {}
    dl = payload.get("delta") or {}
    action = payload.get("action") or {}

    book_dir = three.get("book") or conf.get("book") or "flat"
    book_s = abs(float(payload.get("score") or conf.get("book_strength") or 0.0))

    delta_dir = three.get("delta") or "flat"
    delta_s = abs(float(dl.get("score") or dl.get("lean") or 0.0))

    price_dir = three.get("price") or conf.get("candle") or "flat"
    price_s = abs(float(action.get("score")
                        or conf.get("candle_strength") or 0.0))

    return cast(book_dir, book_s, delta_dir, delta_s, price_dir, price_s)


def from_state(row: dict) -> Vote:
    """Cast a vote from a stored `agreement_states` row.

    This is what makes weeks of already-collected candles testable tonight
    instead of next month. The raw readings were stored rather than the
    verdict, precisely so a rule invented later could be scored against
    them.
    """
    feats = row.get("features") or {}
    if isinstance(feats, str):
        import json
        try:
            feats = json.loads(feats)
        except Exception:
            feats = {}

    delta_score = feats.get("delta_score")
    delta_lean = feats.get("delta_lean")
    if isinstance(delta_score, (int, float)) and delta_score:
        delta_dir = "up" if delta_score > 0 else "down"
        delta_s = abs(float(delta_score))
    elif isinstance(delta_lean, (int, float)) and delta_lean:
        delta_dir = "up" if delta_lean > 0 else "down"
        delta_s = abs(float(delta_lean))
    else:
        delta_dir, delta_s = "flat", 0.0

    return cast(row.get("book_dir"), float(row.get("book_strength") or 0.0),
                delta_dir, delta_s,
                row.get("price_dir"), float(row.get("price_strength") or 0.0))
