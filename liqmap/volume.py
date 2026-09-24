"""Where inside the candle the business actually got done.

WHAT THIS ADDS THAT THE CANDLE DOES NOT HAVE

A bar records four prices, a total volume and a buy/sell split. It does not
record WHERE in its own range that volume traded, and the difference between
two bars with identical OHLC is often the whole trade:

    green bar, closing at the high, all its volume down at the LOW
        a drift up on nothing. Nobody bought up here. It falls back.

    green bar, closing at the high, all its volume up at the HIGH
        acceptance. Real size changed hands at the top of the range.

`position_in_range` scores those identically, because it is geometry. The
volume-weighted version -- what SHARE of the bar's volume traded below where
price is now -- separates them.

THE OTHER USE, WHICH IS THE ONE THAT PAIRS WITH THE BOOK

A thin book above is not one thing. It is two:

    thin above, and little volume ever traded up there
        air. There is nothing to consume and nobody defending. Price runs.

    thin above, and heavy volume traded up there recently
        the book is thin BECAUSE it was already eaten. Whoever traded that
        size is up there, and the level gets defended.

Identical book reading, opposite trade. `ahead()` answers which one it is by
measuring the volume standing between price and its target.

BUCKETS, NOT TRADES

Every fill is folded into a price bucket as it arrives rather than stored.
A busy fifteen-minute bar is tens of thousands of fills and keeping them is
both unnecessary and a memory leak waiting to happen; the profile only ever
answers questions about totals per price.

Bucket width scales with the instrument -- roughly one basis point of price
-- so the same code reads a $4 coin and a $100k one without a table of
per-symbol settings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

# Bucket width as a fraction of price. One basis point is fine enough to
# locate a point of control inside a scalping range and coarse enough that a
# quiet hour does not produce ten thousand near-empty buckets.
BUCKET_FRACTION = 1e-4

# Share of total volume that defines the value area, by the usual
# convention: the contiguous band around the point of control holding this
# much of the bar's business.
VALUE_AREA = 0.70


def bucket_size(px: float) -> float:
    """A price step about one basis point wide, rounded to a clean power.

    Rounding to a power of ten keeps buckets stable as price drifts: without
    it, a bucket computed from a moving price slowly shifts its boundaries
    and the same trade lands in different buckets at different times.
    """
    if px <= 0:
        return 0.0
    raw = px * BUCKET_FRACTION
    return 10.0 ** math.floor(math.log10(raw))


@dataclass
class VolumeProfile:
    """Volume at price for one candle, split by who was aggressing."""

    step: float = 0.0
    buy: dict[float, float] = field(default_factory=dict)
    sell: dict[float, float] = field(default_factory=dict)
    total_notional: float = 0.0
    trades: int = 0

    def add(self, px: float, notional: float, aggressor: str) -> None:
        if px <= 0 or notional <= 0:
            return
        if self.step <= 0:
            self.step = bucket_size(px)
        if self.step <= 0:
            return

        key = round(round(px / self.step) * self.step, 10)
        book = self.buy if aggressor == "buy" else self.sell
        book[key] = book.get(key, 0.0) + notional
        self.total_notional += notional
        self.trades += 1

    # -- reading it -------------------------------------------------------

    def at(self, price: float) -> float:
        """Total notional in the bucket holding `price`."""
        if self.step <= 0:
            return 0.0
        key = round(round(price / self.step) * self.step, 10)
        return self.buy.get(key, 0.0) + self.sell.get(key, 0.0)

    @property
    def levels(self) -> list[tuple[float, float]]:
        """(price, notional) for every bucket, low to high."""
        keys = set(self.buy) | set(self.sell)
        return sorted((k, self.buy.get(k, 0.0) + self.sell.get(k, 0.0))
                      for k in keys)

    @property
    def poc(self) -> float | None:
        """Point of control: the price where the most business was done.

        The bar's centre of gravity, and a far better "fair value" for a
        single candle than its midpoint, which is just geometry.
        """
        best_px, best_v = None, 0.0
        for px, v in self.levels:
            if v > best_v:
                best_px, best_v = px, v
        return best_px

    def below(self, price: float) -> float:
        """Notional that traded strictly below `price`."""
        return sum(v for px, v in self.levels if px < price)

    def above(self, price: float) -> float:
        return sum(v for px, v in self.levels if px > price)

    def position(self, price: float) -> float:
        """Share of the bar's volume that traded below `price`, 0..1.

        The volume-weighted replacement for `(last - low) / (high - low)`.
        Where geometry says "price is at the top of the range", this says
        "price is above 90% of the business done in this bar", which is a
        claim about participants rather than about shape.

        Returns 0.5 when there is nothing to weigh, matching the geometric
        version's neutral value rather than inventing a lean.
        """
        if self.total_notional <= 0:
            return 0.5
        return max(0.0, min(1.0, self.below(price) / self.total_notional))

    def ahead(self, price: float, target: float) -> float:
        """Notional that traded between here and the target.

        The number that conditions the book. Thin offers with nothing ahead
        is air; thin offers with heavy volume ahead is a level that already
        traded and will be defended.
        """
        lo, hi = (price, target) if target >= price else (target, price)
        return sum(v for px, v in self.levels if lo < px <= hi)

    def ahead_ratio(self, price: float, target: float) -> float:
        """`ahead` as a share of the bar's total volume.

        Scale-free, so a threshold means the same thing on a quiet bar and a
        busy one.
        """
        if self.total_notional <= 0:
            return 0.0
        return self.ahead(price, target) / self.total_notional

    def delta_at(self, price: float) -> float:
        """Buy minus sell notional in this bucket -- footprint, one level.

        Signed the obvious way: positive means aggressive buyers did more
        business at this price than aggressive sellers.
        """
        if self.step <= 0:
            return 0.0
        key = round(round(price / self.step) * self.step, 10)
        return self.buy.get(key, 0.0) - self.sell.get(key, 0.0)

    def value_area(self, share: float = VALUE_AREA
                   ) -> tuple[float, float] | None:
        """(low, high) of the band around the POC holding `share` of volume.

        Grown outward from the point of control one bucket at a time, always
        taking the heavier neighbour -- the standard construction. Price
        outside the value area is trading away from where business was done,
        which is what "acceptance" and "rejection" actually mean.
        """
        levels = self.levels
        if not levels or self.total_notional <= 0:
            return None

        prices = [p for p, _ in levels]
        vols = {p: v for p, v in levels}
        poc = self.poc
        if poc is None:
            return None

        i = prices.index(poc)
        lo = hi = i
        acc = vols[poc]
        want = self.total_notional * share

        while acc < want and (lo > 0 or hi < len(prices) - 1):
            down = vols[prices[lo - 1]] if lo > 0 else -1.0
            up = vols[prices[hi + 1]] if hi < len(prices) - 1 else -1.0
            if up >= down:
                hi += 1
                acc += vols[prices[hi]]
            else:
                lo -= 1
                acc += vols[prices[lo]]

        return prices[lo], prices[hi]

    def describe(self, price: float) -> str:
        if self.total_notional <= 0:
            return "no volume recorded in this bar yet"
        poc = self.poc
        pos = self.position(price)
        where = ("above" if poc is not None and price > poc
                 else "below" if poc is not None and price < poc else "at")
        return (f"price is {where} the point of control"
                + (f" ({poc:,.6g})" if poc is not None else "")
                + f", above {pos:.0%} of the bar's volume, "
                f"{self.trades} fills")

    def to_dict(self, price: float = 0.0) -> dict:
        va = self.value_area()
        return {
            "poc": self.poc,
            "total_notional": round(self.total_notional, 2),
            "trades": self.trades,
            "value_area_low": va[0] if va else None,
            "value_area_high": va[1] if va else None,
            "position": round(self.position(price), 4) if price > 0 else None,
            "buckets": len(self.levels),
        }


def merge(profiles: Iterable[VolumeProfile]) -> VolumeProfile:
    """Combine several bars' profiles into one, for a multi-bar view."""
    out = VolumeProfile()
    for p in profiles:
        if out.step <= 0:
            out.step = p.step
        for src, dst in ((p.buy, out.buy), (p.sell, out.sell)):
            for k, v in src.items():
                dst[k] = dst.get(k, 0.0) + v
        out.total_notional += p.total_notional
        out.trades += p.trades
    return out
