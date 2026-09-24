"""How far can this go before something stops it?

THE QUESTION THE GRADE DOES NOT ANSWER

Three columns agreeing tells you the direction is probably right. It says
nothing about distance, and those are different questions. Getting the
direction right and then taking four ticks out of a forty-basis-point move
is not a bad read -- it is a good read with no idea how much room it had.

So direction quality and holding distance are computed separately and
reported separately. Grade says whether to take it. Runway says how long to
stay.

WHAT ACTUALLY STOPS A MOVE

Four things, and the nearest one wins:

    RESTING SIZE    a shelf in the book ahead. Real orders that must be
                    consumed before price passes.
    TRADED VOLUME   a price where heavy business was done. People are
                    positioned there and defend it.
    FORCED FLOW     a liquidation cluster. Flow that MUST happen, and which
                    can accelerate through the level rather than stop at it
                    -- so it is reported, but its meaning is direction
                    dependent and it is never treated as a wall.
    STRUCTURE       the last swing high or low. Where the previous attempt
                    turned.

Everything here is measured, and every one of these is already collected
somewhere in this service. Runway is not new data; it is asking the data a
question nobody was asking.

THE TARGET SHOULD NEVER BE BEYOND THE RUNWAY

That is the practical payoff. A target sized from typical bar range can sit
happily on the far side of a shelf the book is already showing. Capping the
target at the first obstacle turns a known unknown into a known.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from .flow import Book
from .structure import Candle

Side = Literal["long", "short"]

# A resting level counts as an obstacle when it holds at least this multiple
# of the average level size nearby. Below that it is ordinary depth, not a
# wall.
SHELF_MULTIPLE = 3.0

# A price bucket counts as a volume obstacle at this share of the bar's
# total traded volume.
VOLUME_SHARE = 0.12

# Obstacles further away than this are not relevant to a scalp and reporting
# them would make every runway look enormous.
MAX_LOOK_BPS = 200.0


@dataclass
class Obstacle:
    kind: str                 # shelf | volume | liquidation | structure
    price: float
    distance_bps: float
    size: float = 0.0
    note: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "price": self.price,
                "distance_bps": round(self.distance_bps, 2),
                "size": round(self.size, 2), "note": self.note}


@dataclass
class Runway:
    """Clear distance ahead, and what ends it."""

    side: Side
    entry: float
    clear_bps: float
    obstacles: list[Obstacle] = field(default_factory=list)
    capped: bool = False          # True when nothing was found inside range

    @property
    def first(self) -> Obstacle | None:
        return self.obstacles[0] if self.obstacles else None

    def holdable(self, target_bps: float) -> bool:
        """Is there room for this target before the first obstacle?"""
        return target_bps <= self.clear_bps

    def cap(self, target_bps: float) -> float:
        """Trim a target back to the runway.

        A target sized from bar range can sit on the far side of a shelf the
        book is showing right now. Knowing about the obstacle and aiming
        past it anyway is the avoidable mistake.
        """
        return min(target_bps, self.clear_bps) if self.clear_bps > 0 else target_bps

    def describe(self) -> str:
        if self.capped:
            return (f"nothing in the way for at least {self.clear_bps:.0f}bps "
                    f"— open road as far as this reads")
        o = self.first
        if o is None:
            return "no runway measured"
        return (f"{self.clear_bps:.0f}bps of clear air, then {o.note} at "
                f"{o.price:,.6g}")

    def to_dict(self) -> dict:
        return {"side": self.side, "entry": self.entry,
                "clear_bps": round(self.clear_bps, 2),
                "open_road": self.capped,
                "first": self.first.to_dict() if self.first else None,
                "obstacles": [o.to_dict() for o in self.obstacles[:5]],
                "describe": self.describe()}


def _ahead(price: float, entry: float, side: Side) -> bool:
    return price > entry if side == "long" else price < entry


def _bps(price: float, entry: float) -> float:
    return abs(price - entry) / entry * 10_000.0 if entry > 0 else 0.0


def book_shelves(book: Book | None, entry: float, side: Side
                 ) -> list[Obstacle]:
    """Resting levels ahead that are unusually large.

    Ordinary depth is not an obstacle -- something has to stand out from its
    neighbours before it is worth calling a wall, or every level qualifies
    and the runway is always one tick.
    """
    if book is None or book.empty or entry <= 0:
        return []

    levels = book.asks if side == "long" else book.bids
    ahead = [l for l in levels if _ahead(l.px, entry, side)
             and _bps(l.px, entry) <= MAX_LOOK_BPS]
    if len(ahead) < 3:
        return []

    sizes = [l.notional for l in ahead if l.notional > 0]
    if not sizes:
        return []
    avg = sum(sizes) / len(sizes)
    if avg <= 0:
        return []

    out = []
    for l in ahead:
        if l.notional >= avg * SHELF_MULTIPLE:
            out.append(Obstacle(
                kind="shelf", price=l.px, distance_bps=_bps(l.px, entry),
                size=l.notional,
                note=f"a resting shelf of ${l.notional:,.0f}"))
    return out


def volume_walls(profile: Any, entry: float, side: Side) -> list[Obstacle]:
    """Prices where heavy business was already done.

    A level that traded size has people positioned at it, and they defend
    it. This is the half of "why is the book thin here" that the book itself
    cannot answer.
    """
    if profile is None or entry <= 0:
        return []
    total = getattr(profile, "total_notional", 0.0)
    if total <= 0:
        return []

    out = []
    for px, notional in profile.levels:
        if not _ahead(px, entry, side):
            continue
        d = _bps(px, entry)
        if d > MAX_LOOK_BPS:
            continue
        if notional >= total * VOLUME_SHARE:
            out.append(Obstacle(
                kind="volume", price=px, distance_bps=d, size=notional,
                note=f"{notional / total:.0%} of the bar's volume traded"))
    return out


def structure_levels(bars: Sequence[Candle], entry: float, side: Side,
                     lookback: int = 20) -> list[Obstacle]:
    """The last swing high or low ahead: where the previous attempt turned."""
    if not bars or entry <= 0:
        return []
    window = list(bars[-lookback:])
    if len(window) < 3:
        return []

    out = []
    if side == "long":
        hi = max(c.high for c in window)
        if _ahead(hi, entry, side) and _bps(hi, entry) <= MAX_LOOK_BPS:
            out.append(Obstacle(kind="structure", price=hi,
                                distance_bps=_bps(hi, entry),
                                note="the recent swing high"))
    else:
        lo = min(c.low for c in window)
        if _ahead(lo, entry, side) and _bps(lo, entry) <= MAX_LOOK_BPS:
            out.append(Obstacle(kind="structure", price=lo,
                                distance_bps=_bps(lo, entry),
                                note="the recent swing low"))
    return out


def liquidation_level(magnet_bps: float, magnet_notional: float,
                      entry: float, side: Side) -> list[Obstacle]:
    """A liquidation cluster ahead.

    Reported, never treated as a wall. Forced flow is the one obstacle that
    can ACCELERATE price through a level rather than stop it there -- longs
    liquidating below produce forced selling, which helps a short rather
    than blocking it. So it is surfaced for the operator and deliberately
    left out of the clear-air calculation.
    """
    if magnet_notional <= 0 or entry <= 0 or magnet_bps == 0:
        return []
    px = entry * (1 + magnet_bps / 10_000.0)
    if not _ahead(px, entry, side):
        return []
    d = _bps(px, entry)
    if d > MAX_LOOK_BPS:
        return []
    return [Obstacle(kind="liquidation", price=px, distance_bps=d,
                     size=magnet_notional,
                     note=f"${magnet_notional:,.0f} of liquidations — fuel, "
                          f"not a wall; it can accelerate price through")]


def measure(entry: float, side: Side, book: Book | None = None,
            profile: Any = None, bars: Sequence[Candle] = (),
            magnet_bps: float = 0.0, magnet_notional: float = 0.0
            ) -> Runway:
    """Clear distance ahead, and everything standing in it."""
    if entry <= 0:
        return Runway(side=side, entry=entry, clear_bps=0.0)

    found = (book_shelves(book, entry, side)
             + volume_walls(profile, entry, side)
             + structure_levels(bars, entry, side)
             + liquidation_level(magnet_bps, magnet_notional, entry, side))
    found.sort(key=lambda o: o.distance_bps)

    # Liquidations are shown but do not shorten the runway -- see above.
    blocking = [o for o in found if o.kind != "liquidation"]
    if blocking:
        clear = blocking[0].distance_bps
        capped = False
    else:
        clear = MAX_LOOK_BPS
        capped = True

    return Runway(side=side, entry=entry, clear_bps=clear,
                  obstacles=found, capped=capped)
