"""One candle, read as six layers. No verdict here.

WHAT THIS IS

Six questions about the bar in progress, asked separately and kept apart:

    1 INTENT      who is crossing the spread
    2 FORCE       how hard, how fast, and against what is normal
    3 RESISTANCE  is somebody taking the other side of it
    4 LIQUIDITY   what the resting book is doing about it
    5 RESPONSE    what price actually did in return
    6 LOCATION    where on the map this is happening

Every one of these is already computed somewhere in this project. The work
here is assembly, not measurement: `pressure` owns intent, `orderflow` and
`vpace` own force, `flow.Absorption` owns resistance and response, `dom`
owns liquidity, and the profile, level and structure modules own location.
This file does not recompute any of them -- it is handed what they
produced and arranges it.

WHY ASSEMBLY IS WORTH ITS OWN FILE

Scattered across eight modules, those six readings can only be compared by
a person holding all of them in their head at once. Together they are a
description of the bar: heavy buying, into a thinning offer, at the prior
day's high, with price refusing to go. Each of those clauses on its own is
nearly useless.

NOTHING IS CLASSIFIED HERE, AND THAT IS DELIBERATE

The seventh layer -- continuation, absorption, exhaustion, reversal and
the rest -- is a separate module on top of this one. Keeping the
description apart from the verdict means the inputs can be checked before
a word is built on them, and a wrong word can be traced to the reading
that produced it rather than to the whole pile.

MISSING IS NOT NEUTRAL

Every layer reports whether it was measured. A feed that has not warmed
up, a baseline with too few samples, a book too thin to read: each says
so. A layer that quietly returned zero would be indistinguishable from
one that measured balance, and the difference matters more than almost
anything else here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

Who = Literal["buyers", "sellers", "neither", "unknown"]

# A reference further than this from price, as a share of the bar's own
# range, is not where this is happening.
NEAR_RANGE = 1.5

# References reported. Beyond a handful it is a list, not a location.
NEAR_MAX = 4


def _d(x: float, n: int = 4) -> float:
    return round(float(x), n)


# ---------------------------------------------------------------- 1 intent


@dataclass
class Intent:
    """Who is crossing the spread."""

    who: Who = "unknown"
    lean: float = 0.0            # -1 all selling, +1 all buying
    trades: int = 0
    notional: float = 0.0
    measured: bool = False       # counted from fills rather than inferred

    def describe(self) -> str:
        if not self.measured:
            return "no tape: who is attacking is inferred, not counted"
        if self.who == "neither":
            return "neither side is pressing"
        return (f"{self.who} are crossing, {abs(self.lean):.0%} one-sided "
                f"across {self.trades:,} fills")

    def to_dict(self) -> dict:
        return {"who": self.who, "lean": _d(self.lean), "trades": self.trades,
                "notional": _d(self.notional, 2), "measured": self.measured,
                "describe": self.describe()}


# ----------------------------------------------------------------- 2 force


@dataclass
class Force:
    """How hard, how fast, and against what is normal here."""

    notional: float = 0.0
    delta: float = 0.0
    velocity: float = 1.0        # prints/sec against this market's baseline
    velocity_known: bool = False
    pace: float = 1.0            # volume against a normal bar BY THIS POINT
    pace_known: bool = False

    @property
    def measured(self) -> bool:
        return self.velocity_known or self.pace_known

    def describe(self) -> str:
        bits = []
        if self.pace_known:
            bits.append(f"{self.pace:.1f}x a normal bar's volume by now")
        if self.velocity_known:
            bits.append(f"tape at {self.velocity:.1f}x its baseline")
        return "; ".join(bits) or "no scale for this market yet"

    def to_dict(self) -> dict:
        return {"notional": _d(self.notional, 2), "delta": _d(self.delta, 2),
                "velocity": _d(self.velocity, 2),
                "velocity_known": self.velocity_known,
                "pace": _d(self.pace, 3), "pace_known": self.pace_known,
                "measured": self.measured, "describe": self.describe()}


# ------------------------------------------------------------ 3 resistance


@dataclass
class Resistance:
    """Is somebody taking the other side of all that."""

    absorbing: bool = False
    impact_ratio: float = 0.0    # how far price moved against how far it should
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    busiest_px: float = 0.0      # where the executions piled up
    busiest_share: float = 0.0   # that price's share of the bar's volume
    measured: bool = False       # the impact baseline has a scale

    @property
    def depth_tilt(self) -> float:
        t = self.bid_depth + self.ask_depth
        return (self.bid_depth - self.ask_depth) / t if t > 0 else 0.0

    def describe(self) -> str:
        if not self.measured:
            return "no impact baseline yet -- absorption cannot be judged"
        if self.absorbing:
            return (f"aggression is moving price {self.impact_ratio:.0%} of "
                    f"what it normally would -- somebody is filling it")
        return (f"price is moving {self.impact_ratio:.0%} of normal for this "
                f"much flow")

    def to_dict(self) -> dict:
        return {"absorbing": self.absorbing,
                "impact_ratio": _d(self.impact_ratio, 3),
                "bid_depth": _d(self.bid_depth, 2),
                "ask_depth": _d(self.ask_depth, 2),
                "depth_tilt": _d(self.depth_tilt),
                "busiest_px": _d(self.busiest_px, 8),
                "busiest_share": _d(self.busiest_share, 3),
                "measured": self.measured, "describe": self.describe()}


# -------------------------------------------------------------- 4 liquidity


@dataclass
class Liquidity:
    """What the resting book is doing. Straight from `dom`."""

    dom: Any = None

    @property
    def measured(self) -> bool:
        return bool(self.dom is not None and getattr(self.dom, "measured",
                                                     False))

    def describe(self) -> str:
        if self.dom is None:
            return "no book history yet"
        return self.dom.describe()

    def to_dict(self) -> dict:
        return {"measured": self.measured,
                "dom": self.dom.to_dict() if self.dom is not None else None,
                "describe": self.describe()}


# --------------------------------------------------------------- 5 response


@dataclass
class Response:
    """What price actually did in return. The layer Chris called critical.

    The three cases he named are one scale, not three categories: attacked
    and went four ticks, attacked and went one then stalled, attacked and
    did not move at all. `impact` is that scale -- how far price travelled
    against how far this much flow normally moves it.
    """

    move: float = 0.0            # in the market's own unit, signed
    range_: float = 0.0
    unit: str = "pt"
    impact: float = 0.0
    with_intent: bool | None = None   # did price go the aggressor's way
    stalled: bool = False
    measured: bool = False

    def describe(self) -> str:
        if not self.measured:
            return "not enough of this bar to say what price did"
        if self.with_intent is False:
            return (f"price went {self.move:+.1f}{self.unit} -- against the "
                    f"side doing the attacking")
        if self.stalled:
            return (f"price has gone {self.move:+.1f}{self.unit} and stopped")
        return f"price has gone {self.move:+.1f}{self.unit}"

    def to_dict(self) -> dict:
        return {"move": _d(self.move, 3), "range": _d(self.range_, 3),
                "unit": self.unit, "impact": _d(self.impact, 3),
                "with_intent": self.with_intent, "stalled": self.stalled,
                "measured": self.measured, "describe": self.describe()}


# --------------------------------------------------------------- 6 location


@dataclass(frozen=True)
class Reference:
    """One thing on the map, and how far price is from it."""

    name: str
    price: float
    distance: float              # signed, in the market's unit: + above price
    unit: str = "pt"

    @property
    def above(self) -> bool:
        return self.distance > 0

    def to_dict(self) -> dict:
        return {"name": self.name, "price": _d(self.price, 8),
                "distance": _d(self.distance, 3), "unit": self.unit,
                "above": self.above}


@dataclass
class Location:
    """Where on the map this is happening."""

    price: float = 0.0
    near: list[Reference] = field(default_factory=list)
    unit: str = "pt"

    @property
    def measured(self) -> bool:
        return bool(self.near)

    @property
    def at(self) -> Reference | None:
        """The nearest reference, if there is one."""
        return self.near[0] if self.near else None

    def describe(self) -> str:
        if not self.near:
            return "nothing on the map within reach of this bar"
        bits = [f"{r.name} {abs(r.distance):.1f}{r.unit} "
                f"{'above' if r.above else 'below'}" for r in self.near]
        return "at " + ", ".join(bits)

    def to_dict(self) -> dict:
        return {"price": _d(self.price, 8), "unit": self.unit,
                "measured": self.measured,
                "at": self.at.to_dict() if self.at else None,
                "near": [r.to_dict() for r in self.near],
                "describe": self.describe()}


def nearby(price: float, refs: Sequence[tuple[str, float]],
           unit_size: float, bar_range: float = 0.0,
           unit: str = "pt", limit: int = NEAR_MAX,
           near_range: float = NEAR_RANGE) -> list[Reference]:
    """The references this bar is actually near, nearest first.

    Filtered by the BAR'S OWN RANGE rather than by a fixed distance. A
    level twenty points away is in play on a four-hour bar and irrelevant
    on a one-minute one, and a fixed cutoff gets one of those wrong
    whichever number is chosen. With no range to go on, everything given
    is kept and the caller can decide.
    """
    if price <= 0 or unit_size <= 0:
        return []
    out = []
    for name, px in refs or ():
        px = float(px or 0.0)
        if px <= 0:
            continue
        gap = (px - price) / unit_size
        if bar_range > 0 and abs(px - price) > bar_range * near_range:
            continue
        out.append(Reference(name=name, price=px, distance=gap, unit=unit))
    out.sort(key=lambda r: abs(r.distance))
    return out[:limit]


# ----------------------------------------------------------------- the whole


@dataclass
class Layers:
    """One candle on one timeframe, described in six layers."""

    timeframe: str
    interval_s: float = 0.0
    intent: Intent = field(default_factory=Intent)
    force: Force = field(default_factory=Force)
    resistance: Resistance = field(default_factory=Resistance)
    liquidity: Liquidity = field(default_factory=Liquidity)
    response: Response = field(default_factory=Response)
    location: Location = field(default_factory=Location)

    @property
    def measured(self) -> list[str]:
        return [n for n in ("intent", "force", "resistance", "liquidity",
                            "response", "location")
                if getattr(getattr(self, n), "measured", False)]

    @property
    def missing(self) -> list[str]:
        have = set(self.measured)
        return [n for n in ("intent", "force", "resistance", "liquidity",
                            "response", "location") if n not in have]

    def describe(self) -> str:
        """The six clauses, in order, as one sentence about the bar."""
        return " | ".join(
            f"{n}: {getattr(self, n).describe()}"
            for n in ("intent", "force", "resistance", "liquidity",
                      "response", "location"))

    def to_dict(self) -> dict:
        return {"timeframe": self.timeframe, "interval_s": self.interval_s,
                "intent": self.intent.to_dict(),
                "force": self.force.to_dict(),
                "resistance": self.resistance.to_dict(),
                "liquidity": self.liquidity.to_dict(),
                "response": self.response.to_dict(),
                "location": self.location.to_dict(),
                "measured": self.measured, "missing": self.missing,
                "describe": self.describe()}


def assemble(timeframe: str, interval_s: float, *,
             pressure=None, velocity=None, pace=None, absorption=None,
             dom_read=None, profile=None, book=None,
             refs: Sequence[tuple[str, float]] = (),
             unit_size: float = 0.0, unit: str = "pt") -> Layers:
    """Arrange what the other modules produced. Nothing is recomputed.

    Every input is optional and every layer says whether it was measured,
    because a feed that has not warmed up and a market in balance produce
    the same zeros otherwise.
    """
    out = Layers(timeframe=timeframe, interval_s=float(interval_s))

    if pressure is not None:
        agg = pressure.aggressor
        out.intent = Intent(
            who=("buyers" if agg == "buy" else
                 "sellers" if agg == "sell" else "neither"),
            lean=pressure.lean, trades=pressure.trades,
            notional=pressure.total_notional,
            measured=bool(pressure.measured))
        out.force.notional = pressure.total_notional
        out.force.delta = pressure.delta

    if velocity is not None:
        out.force.velocity = velocity.ratio
        out.force.velocity_known = bool(velocity.confident)
    if pace is not None:
        out.force.pace = pace.pace
        out.force.pace_known = bool(pace.known)

    if absorption is not None:
        out.resistance.absorbing = bool(absorption.absorbing)
        out.resistance.impact_ratio = float(absorption.impact_ratio)
        out.resistance.measured = bool(absorption.confident)

    if book is not None and not getattr(book, "empty", True):
        try:
            out.resistance.bid_depth = float(book.depth(25.0, "buy"))
            out.resistance.ask_depth = float(book.depth(25.0, "sell"))
        except Exception:
            pass

    if profile is not None:
        try:
            lv = profile.levels()
            total = sum(v for _, v in lv)
            if lv and total > 0:
                px, vol = max(lv, key=lambda x: x[1])
                out.resistance.busiest_px = float(px)
                out.resistance.busiest_share = float(vol) / total
        except Exception:
            pass

    out.liquidity = Liquidity(dom=dom_read)

    if pressure is not None and unit_size > 0:
        move = (pressure.last_px - pressure.open_px) / unit_size
        rng = (pressure.high_px - pressure.low_px) / unit_size
        with_intent: bool | None = None
        if out.intent.who in ("buyers", "sellers"):
            want = 1.0 if out.intent.who == "buyers" else -1.0
            with_intent = (move * want) > 0
        out.response = Response(
            move=move, range_=rng, unit=unit,
            impact=out.resistance.impact_ratio,
            with_intent=with_intent,
            stalled=bool(absorption is not None and absorption.absorbing),
            measured=not bool(getattr(pressure, "forming", False)))
        out.location = Location(
            price=float(pressure.last_px), unit=unit,
            near=nearby(float(pressure.last_px), refs, unit_size,
                        bar_range=float(pressure.high_px - pressure.low_px),
                        unit=unit))
    return out
