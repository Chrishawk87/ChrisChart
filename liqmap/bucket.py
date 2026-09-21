"""Turn raw positions into a liquidation map.

The input is a pile of open positions, each carrying a liquidation price and a
notional. The output is a set of price buckets with the forced-flow sitting in
each one.

The direction convention matters and is easy to get backwards:

    LONG positions liquidate BELOW spot, and their liquidation is a forced
    SELL. So long-liquidation clusters underneath price are downside cascade
    fuel.

    SHORT positions liquidate ABOVE spot, and their liquidation is a forced
    BUY. So short-liquidation clusters overhead are squeeze fuel.

A cluster is therefore not just "a level" -- it has a side and a sign, and
touching it produces flow in a known direction. That is the entire reason this
data is more useful than a generic support/resistance line.

Bucket width is expressed in basis points of spot rather than in absolute
price, so the same configuration works across assets trading at $0.30 and
$100,000.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class Position:
    """One open perp position.

    `szi` is Hyperliquid's signed size: positive is long, negative is short.
    `liquidation_px` may be None -- a position with no liquidation price (very
    low leverage, or fully collateralised) carries no forced flow and is
    excluded from the map rather than treated as zero.

    The fields below `leverage_type` are what make strength scoring possible:
    every one of them comes free in the same `clearinghouseState` response, so
    there is no reason not to carry them.
    """

    wallet: str
    coin: str
    szi: float
    entry_px: float
    liquidation_px: float | None
    position_value: float
    leverage: float
    leverage_type: str = "cross"

    unrealized_pnl: float = 0.0
    return_on_equity: float = 0.0
    margin_used: float = 0.0
    max_leverage: float = 0.0
    # Cumulative funding PAID since the position opened. Hyperliquid's sign
    # convention is that a positive number means the trader paid out.
    funding_since_open: float = 0.0
    # Whole-account equity, from marginSummary. Needed to judge how much of
    # the account a single position represents.
    account_value: float = 0.0

    @property
    def is_long(self) -> bool:
        return self.szi > 0

    @property
    def notional(self) -> float:
        """Absolute dollar size of the position."""
        if self.position_value:
            return abs(self.position_value)
        return abs(self.szi) * self.entry_px

    @property
    def key(self) -> tuple[str, str]:
        """Identity for matching the same position across snapshots."""
        return (self.wallet, self.coin)


@dataclass
class Bucket:
    """One price bin of the map."""

    price_low: float
    price_high: float
    long_notional: float = 0.0      # forced SELL flow if touched
    short_notional: float = 0.0     # forced BUY flow if touched
    long_count: int = 0
    short_count: int = 0

    @property
    def mid(self) -> float:
        return (self.price_low + self.price_high) / 2.0

    @property
    def total_notional(self) -> float:
        return self.long_notional + self.short_notional

    @property
    def net_notional(self) -> float:
        """Positive means net forced BUYING here, negative net forced SELLING."""
        return self.short_notional - self.long_notional

    @property
    def count(self) -> int:
        return self.long_count + self.short_count


@dataclass
class LiquidationMap:
    coin: str
    spot: float
    bucket_bps: float
    buckets: list[Bucket] = field(default_factory=list)
    wallets_seen: int = 0
    positions_used: int = 0
    positions_skipped: int = 0
    weighted: bool = False

    # -- queries ----------------------------------------------------------

    def above(self) -> list[Bucket]:
        """Buckets overhead, nearest first. Mostly short liquidations."""
        return sorted((b for b in self.buckets if b.mid > self.spot),
                      key=lambda b: b.mid)

    def below(self) -> list[Bucket]:
        """Buckets underneath, nearest first. Mostly long liquidations."""
        return sorted((b for b in self.buckets if b.mid < self.spot),
                      key=lambda b: -b.mid)

    def clusters(self, min_notional: float = 0.0, top: int = 10) -> list[Bucket]:
        """Biggest buckets by total notional.

        `min_notional` should be fixed in advance, not tuned after looking at
        results -- picking the threshold that makes the backtest work is how
        this kind of analysis lies to you.
        """
        eligible = [b for b in self.buckets if b.total_notional >= min_notional]
        return sorted(eligible, key=lambda b: -b.total_notional)[:top]

    def total_notional(self) -> float:
        return sum(b.total_notional for b in self.buckets)

    def concentration(self, top: int = 5) -> float:
        """Share of all mapped notional sitting in the biggest few buckets.

        Near 1 means the fuel is piled at a handful of levels. Near 0 means it
        is smeared everywhere, and no individual level is worth targeting.
        A diffuse map is a real result, not a failed one.
        """
        total = self.total_notional()
        if total <= 0:
            return 0.0
        biggest = sorted((b.total_notional for b in self.buckets), reverse=True)
        return sum(biggest[:top]) / total


def build_map(positions: Iterable[Position], coin: str, spot: float,
              bucket_bps: float = 25.0, max_distance_pct: float = 0.25,
              min_notional_per_position: float = 0.0,
              weight_fn: "Callable[[Position], float] | None" = None
              ) -> LiquidationMap:
    """Bucket liquidation prices into a map.

    `bucket_bps` is the bin width in basis points of spot -- 25 bps on a
    $100k asset is a $250 bin.

    `max_distance_pct` drops liquidations further than this fraction away.
    A position liquidating 80% below spot is not information about the next
    few hours; including it inflates the totals and tells you nothing.

    `weight_fn` scales each position's notional -- pass
    `strength.fragility_weighter(spot, sigma)` to weight by how likely each
    position is to actually be forced out. That turns a map of where leverage
    SITS into a map of where PRESSURE is, which are different pictures: a
    billion dollars of comfortable, well-collateralised longs is a level that
    holds, not fuel.
    """
    if spot <= 0:
        raise ValueError("spot must be positive")
    if bucket_bps <= 0:
        raise ValueError("bucket_bps must be positive")

    width = spot * bucket_bps / 10_000.0
    lo_bound = spot * (1 - max_distance_pct)
    hi_bound = spot * (1 + max_distance_pct)

    by_index: dict[int, Bucket] = {}
    wallets: set[str] = set()
    used = skipped = 0

    for p in positions:
        wallets.add(p.wallet)

        if p.coin != coin:
            continue
        if p.liquidation_px is None or p.liquidation_px <= 0:
            skipped += 1
            continue
        if not (lo_bound <= p.liquidation_px <= hi_bound):
            skipped += 1
            continue
        if p.notional < min_notional_per_position:
            skipped += 1
            continue

        # Sanity check against the direction convention. A long liquidating
        # ABOVE spot, or a short liquidating BELOW it, is either bad data or a
        # position already past its liquidation price mid-update. Either way
        # it is not forward-looking fuel.
        if p.is_long and p.liquidation_px > spot:
            skipped += 1
            continue
        if not p.is_long and p.liquidation_px < spot:
            skipped += 1
            continue

        idx = int(math.floor((p.liquidation_px - spot) / width))
        bucket = by_index.get(idx)
        if bucket is None:
            bucket = Bucket(price_low=spot + idx * width,
                            price_high=spot + (idx + 1) * width)
            by_index[idx] = bucket

        weighted = p.notional * (weight_fn(p) if weight_fn else 1.0)
        if p.is_long:
            bucket.long_notional += weighted
            bucket.long_count += 1
        else:
            bucket.short_notional += weighted
            bucket.short_count += 1
        used += 1

    return LiquidationMap(
        coin=coin, spot=spot, bucket_bps=bucket_bps,
        buckets=sorted(by_index.values(), key=lambda b: b.price_low),
        wallets_seen=len(wallets), positions_used=used,
        positions_skipped=skipped, weighted=weight_fn is not None,
    )


def parse_clearinghouse_state(wallet: str, payload: dict) -> list[Position]:
    """Extract positions from a Hyperliquid `clearinghouseState` response.

    Every numeric field arrives as a string, and `liquidationPx` is null for
    positions that cannot be liquidated. Both are handled here so nothing
    downstream has to think about it.

    `accountValue` lives on `marginSummary`, one level up from the positions,
    and is attached to each of them -- judging whether a position is a large
    bet or a rounding error requires knowing the size of the account behind it.
    """
    out: list[Position] = []

    summary = payload.get("marginSummary") or {}
    try:
        account_value = float(summary.get("accountValue") or 0.0)
    except (TypeError, ValueError):
        account_value = 0.0

    for entry in payload.get("assetPositions") or []:
        pos = entry.get("position") or {}
        coin = pos.get("coin")
        if not coin:
            continue

        def num(key: str) -> float | None:
            v = pos.get(key)
            if v in (None, "", "null"):
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        szi = num("szi")
        if szi is None or szi == 0:
            continue

        lev = pos.get("leverage") or {}
        try:
            lev_value = float(lev.get("value", 0) or 0)
        except (TypeError, ValueError):
            lev_value = 0.0

        funding = pos.get("cumFunding") or {}
        try:
            since_open = float(funding.get("sinceOpen") or 0.0)
        except (TypeError, ValueError):
            since_open = 0.0

        try:
            max_lev = float(pos.get("maxLeverage") or 0.0)
        except (TypeError, ValueError):
            max_lev = 0.0

        out.append(Position(
            wallet=wallet,
            coin=coin,
            szi=szi,
            entry_px=num("entryPx") or 0.0,
            liquidation_px=num("liquidationPx"),
            position_value=num("positionValue") or 0.0,
            leverage=lev_value,
            leverage_type=str(lev.get("type", "cross")),
            unrealized_pnl=num("unrealizedPnl") or 0.0,
            return_on_equity=num("returnOnEquity") or 0.0,
            margin_used=num("marginUsed") or 0.0,
            max_leverage=max_lev,
            funding_since_open=since_open,
            account_value=account_value,
        ))

    return out


def render(lm: LiquidationMap, top: int = 8, width: int = 44) -> str:
    """Text view of the map, nearest levels first on each side."""
    lines = [
        f"{lm.coin}  spot {lm.spot:,.4f}   "
        f"{lm.positions_used} positions from {lm.wallets_seen} wallets "
        f"({lm.positions_skipped} skipped)",
        f"bucket width {lm.bucket_bps:.0f} bps   "
        f"{'fragility-weighted' if lm.weighted else 'raw'} notional "
        f"${lm.total_notional():,.0f}   "
        f"top-5 concentration {lm.concentration():.0%}",
        "",
    ]

    scale = max((b.total_notional for b in lm.buckets), default=1.0) or 1.0

    def row(b: Bucket, tag: str) -> str:
        bar = "#" * max(1, int(width * b.total_notional / scale))
        dist = (b.mid / lm.spot - 1) * 100
        return (f"  {b.mid:>12,.4f} {dist:>+6.2f}%  {tag}  "
                f"${b.total_notional:>12,.0f}  {b.count:>4d}  {bar}")

    above = lm.above()[:top]
    for b in reversed(above):
        lines.append(row(b, "SQ"))     # squeeze fuel: forced buying

    lines.append(f"  {lm.spot:>12,.4f} {'':>6}   --  <<< spot")

    for b in lm.below()[:top]:
        lines.append(row(b, "CA"))     # cascade fuel: forced selling

    lines += ["", "  SQ = short liquidations overhead (forced buying)",
              "  CA = long liquidations below (forced selling)"]
    return "\n".join(lines)
