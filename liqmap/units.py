"""The unit a market is actually traded in: ticks, points or pips.

WHY BASIS POINTS ARE THE WRONG THING TO SHOW

Basis points are the right thing to COMPUTE in. They make a one-minute bar
on a $4 token and a four-hour bar on a $100k one comparable, which is what
every threshold in this project depends on.

They are the wrong thing to READ. Nobody watching an index perp thinks in
basis points; they think in points. Nobody on ES thinks in basis points;
they think in ticks, because the tick is what the exchange moves in and
what the fee is quoted against. A number you have to convert in your head
before it means anything is a number you will misread under pressure.

So everything stays in bps underneath and this converts at the edge.

WHERE THE UNIT COMES FROM

In order of how much it is worth trusting:

    THE BOOK. The smallest gap between adjacent levels is the exchange's
    own increment, observed rather than configured. A per-market table
    would be a thousand chances to be wrong about one market.

    THE BARS. Same idea from traded prices when there is no book.

    THE ASSET CLASS. FX moves in pips and an index in points whatever the
    book happens to show, so a known class names the unit even when the
    increment has to be assumed.

    THE PRICE. A last resort that is at least scale-correct: something
    quoted at 7,731 does not move in hundredths.

A unit always says where it came from. An assumed tick and a measured one
produce the same number on the screen and mean different things.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import assetclass as ac

# Pips. The convention is the fourth decimal, and the second for anything
# quoted against the yen -- which is a quirk of the quote convention, not
# of the currency, so it keys off the pair rather than the base.
PIP = 0.0001
JPY_PIP = 0.01

# What a "point" is, by class, where the venue's own increment is unknown.
CLASS_POINT: dict[str, float] = {
    "index": 1.0,
    "equity": 0.01,
    "etf": 0.01,
    "metals": 0.1,
    "energy": 0.01,
}

# Names, in the order a trader would say them.
NAMES = {"tick": ("tick", "t"), "point": ("point", "pt"),
         "pip": ("pip", "pip"), "unit": ("unit", "u")}


@dataclass(frozen=True)
class Unit:
    """One increment of the thing being traded."""

    name: str              # tick | point | pip | unit
    size: float            # how much price moves in one of them
    label: str             # what goes next to the number
    source: str            # book | bars | class | price
    places: int = 1        # decimals worth showing

    @property
    def measured(self) -> bool:
        """Observed from this market, rather than assumed for its kind."""
        return self.source in ("book", "bars")

    def of(self, price_delta: float) -> float:
        """A price difference, in these units. Signed."""
        return (price_delta / self.size) if self.size > 0 else 0.0

    def from_bps(self, bps: float, price: float) -> float:
        """A move in bps, in these units, at this price."""
        if self.size <= 0 or price <= 0:
            return 0.0
        return (bps / 10_000.0 * price) / self.size

    def to_bps(self, n: float, price: float) -> float:
        if price <= 0:
            return 0.0
        return n * self.size / price * 10_000.0

    def fmt(self, n: float, sign: bool = True) -> str:
        s = f"{n:+.{self.places}f}" if sign else f"{n:.{self.places}f}"
        return f"{s}{self.label}"

    def to_dict(self) -> dict:
        return {"name": self.name, "size": self.size, "label": self.label,
                "source": self.source, "places": self.places,
                "measured": self.measured}


def _places(size: float) -> int:
    """Decimals worth showing for a count of these units.

    One decimal on a normal increment; none when the increment is so coarse
    that a tenth of it is not a thing that happens.
    """
    return 0 if size >= 1.0 else 1


def _price_step(price: float) -> float:
    """A scale-correct increment when nothing better is known."""
    if price <= 0:
        return 0.01
    for cut, step in ((10_000.0, 1.0), (1_000.0, 0.5), (100.0, 0.1),
                      (10.0, 0.01), (1.0, 0.001)):
        if price >= cut:
            return step
    return 0.00001


def for_market(symbol: str, price: float = 0.0, tick: float = 0.0,
               klass: str = "") -> Unit:
    """The unit this market is read in.

    `tick` is the increment observed from the book or the bars, and zero
    when there is none to observe. It wins where it exists, because it is
    the only figure here that was measured rather than assumed -- except
    on FX, where the pip is a quote convention the book does not express:
    a EURUSD book quoted to five decimals has a tenth-pip increment, and
    reporting a move as "35 ticks" when every FX trader on earth would say
    "3.5 pips" is accuracy that costs comprehension.
    """
    k = klass or ac.classify(symbol)

    if k == "fx":
        size = JPY_PIP if "JPY" in symbol.upper() else PIP
        return Unit(name="pip", size=size, label="pip", source="class",
                    places=1)

    if tick > 0:
        return Unit(name="tick", size=tick, label="t", source="book",
                    places=_places(tick))

    point = CLASS_POINT.get(k)
    if point:
        return Unit(name="point", size=point, label="pt", source="class",
                    places=_places(point))

    step = _price_step(price)
    return Unit(name="point" if step >= 0.01 else "unit", size=step,
                label="pt" if step >= 0.01 else "u", source="price",
                places=_places(step))


def from_book(symbol: str, book, price: float = 0.0) -> Unit:
    """The unit, with the increment read off the book where there is one."""
    tick = 0.0
    try:
        from .suggest import infer_tick

        if book is not None and not getattr(book, "empty", True):
            tick = float(infer_tick(book) or 0.0)
            price = price or float(getattr(book, "mid", 0.0) or 0.0)
    except Exception:
        tick = 0.0
    return for_market(symbol, price=price, tick=tick)


def from_bars(symbol: str, bars, price: float = 0.0) -> Unit:
    """Same, from traded prices, for a market with no live book."""
    tick = 0.0
    try:
        from .profile import infer_tick as bar_tick

        if bars:
            tick = float(bar_tick(bars) or 0.0)
            price = price or float(getattr(bars[-1], "close", 0.0) or 0.0)
    except Exception:
        tick = 0.0
    u = for_market(symbol, price=price, tick=tick)
    if u.source == "book":
        u = Unit(name=u.name, size=u.size, label=u.label, source="bars",
                 places=u.places)
    return u
