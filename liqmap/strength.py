"""How strong is a position, and what is that good for.

Every field needed is already in `clearinghouseState`: entry, liquidation
price, unrealised PnL, cumulative funding, margin used, and account value off
`marginSummary`. Nothing extra has to be bought or scraped.

Four things can be measured honestly from a snapshot:

  SURVIVABILITY  How far the liquidation price sits from spot, measured in
                 standard deviations of the horizon. This is the real "holds
                 strong" number. Eight sigma of room is a position that can
                 sit through almost anything; half a sigma is a hostage.

  COMMITMENT     Margin used as a share of account value. A trader with 3% of
                 their equity on a position behaves completely differently
                 from one with 60% on it, whatever the notional says.

  CARRY          Funding paid since the position opened, annualised against
                 position value. Someone RECEIVING funding can wait forever.
                 Someone paying 40% a year is on a clock whether they like it
                 or not.

  MATURITY       Unrealised PnL as a share of position value. Read carefully
                 -- see the inversion below.

THE INVERSION, which is the useful part:

The same score means opposite things depending on the question.

                        | to FOLLOW the trader  | as LIQUIDATION FUEL
    --------------------+-----------------------+---------------------
    far from liq        | strong, survives      | inert, won't fire
    near liq            | fragile               | LIVE fuel
    deep in profit      | move already happened | inert, has buffer
    deep in loss        | under pressure        | LIVE fuel
    receiving funding   | free to hold          | inert
    paying heavy funding| on a clock            | LIVE fuel

Following needs a judgment about skill, which the data cannot support. Fuel
needs no judgment at all -- a liquidation is mechanical. So the second column
is the defensible use, and `fragility()` is the number that feeds the map.

One thing worth saying plainly: a position showing large unrealised profit is
a WORSE candidate to copy, not a better one. The move that made that profit
has already happened, and their entry is the part you cannot have.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .bucket import Position

# Funding is charged hourly on Hyperliquid.
HOURS_PER_YEAR = 24 * 365


@dataclass(frozen=True)
class PositionStrength:
    position: Position

    liq_distance_sigmas: float      # adverse move survivable, in sigma
    liq_distance_pct: float         # same, as a fraction of spot
    commitment: float               # margin used / account value
    carry_annualised: float         # funding paid per year / position value
    maturity: float                 # unrealised PnL / position value
    effective_leverage: float       # notional / account value

    @property
    def is_isolated(self) -> bool:
        """Isolated margin cannot be rescued by the rest of the account.

        A cross position on a large account has the whole balance standing
        behind it; an isolated one is on its own. Two positions with identical
        liquidation distance are not equally likely to fire.
        """
        return self.position.leverage_type.lower() == "isolated"

    @property
    def receives_funding(self) -> bool:
        return self.carry_annualised < 0

    def survivability(self) -> float:
        """0 to 1. How well this position can sit through an adverse move.

        Driven by liquidation distance, softened by commitment and carry.
        Deliberately a blunt instrument -- the components are more informative
        individually, and this exists mainly for ranking.
        """
        if self.liq_distance_sigmas <= 0:
            return 0.0

        # Saturating: beyond about four sigma the extra room stops mattering.
        room = 1.0 - math.exp(-self.liq_distance_sigmas / 2.0)

        # Heavy commitment means forced de-risking arrives before liquidation.
        commit_penalty = 1.0 / (1.0 + max(0.0, self.commitment) * 2.0)

        # Paying funding is a clock; receiving it is a small bonus.
        if self.carry_annualised > 0:
            carry_penalty = 1.0 / (1.0 + self.carry_annualised)
        else:
            carry_penalty = 1.0 + min(0.15, abs(self.carry_annualised) * 0.3)

        cross_bonus = 1.0 if self.is_isolated else 1.15

        return float(min(1.0, room * commit_penalty * carry_penalty * cross_bonus))

    def fragility(self) -> float:
        """0 to 1. How likely this position is to actually become forced flow.

        The number that matters for the map. A cluster of notional made up of
        comfortable, well-collateralised positions is not fuel -- it is a
        level that will hold. Weighting raw notional by this is the difference
        between a map of leverage and a map of PRESSURE.
        """
        return float(max(0.0, min(1.0, 1.0 - self.survivability())))

    def render(self) -> str:
        p = self.position
        side = "LONG " if p.is_long else "SHORT"
        carry = (f"pays {self.carry_annualised:>6.1%}/yr"
                 if self.carry_annualised > 0
                 else f"earns {abs(self.carry_annualised):>5.1%}/yr")
        return (f"{side} {p.coin:<7} ${p.notional:>11,.0f}  "
                f"{p.leverage:>3.0f}x {p.leverage_type:<8} "
                f"liq {self.liq_distance_sigmas:>5.1f}sd "
                f"({self.liq_distance_pct:>5.1%})  "
                f"commit {self.commitment:>5.1%}  {carry}  "
                f"pnl {self.maturity:>+7.1%}  "
                f"surv {self.survivability():.2f} frag {self.fragility():.2f}")


def score(position: Position, spot: float, sigma_per_min: float,
          horizon_minutes: float = 240.0,
          hours_held: float | None = None) -> PositionStrength:
    """Score one position against current spot and volatility.

    `hours_held` annualises the funding correctly when known. Without it the
    funding figure is treated as a 24-hour cost, which is conservative for a
    fresh position and understates the drag on an old one.
    """
    notional = position.notional or 1.0
    account = position.account_value or 0.0

    if position.liquidation_px and position.liquidation_px > 0 and spot > 0:
        log_dist = abs(math.log(position.liquidation_px / spot))
        pct = abs(position.liquidation_px - spot) / spot
        sd = sigma_per_min * math.sqrt(max(horizon_minutes, 1e-9))
        sigmas = log_dist / sd if sd > 0 else float("inf")
    else:
        # No liquidation price means it effectively cannot be liquidated.
        log_dist, pct, sigmas = float("inf"), float("inf"), float("inf")

    commitment = (position.margin_used / account) if account > 0 else 0.0
    effective_leverage = (notional / account) if account > 0 else 0.0

    held = hours_held if hours_held and hours_held > 0 else 24.0
    carry = (position.funding_since_open / notional) * (HOURS_PER_YEAR / held)

    maturity = position.unrealized_pnl / notional if notional else 0.0

    return PositionStrength(
        position=position,
        liq_distance_sigmas=sigmas,
        liq_distance_pct=pct,
        commitment=commitment,
        carry_annualised=carry,
        maturity=maturity,
        effective_leverage=effective_leverage,
    )


def score_all(positions: Iterable[Position], spot: float, sigma_per_min: float,
              horizon_minutes: float = 240.0) -> list[PositionStrength]:
    return [score(p, spot, sigma_per_min, horizon_minutes) for p in positions]


# --------------------------------------------------------------------------
# what changed since last time -- worth more than any snapshot
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PositionChange:
    """What a trader DID between two sweeps.

    A snapshot says a position exists. The delta says what the person behind
    it decided, which is the only thing in this data that resembles intent:

        ADDED      size increased -- conviction, or averaging into a loser
        REDUCED    size decreased -- taking profit, or cutting
        DEFENDED   size unchanged but liquidation moved AWAY. They posted more
                   margin. This is the strongest conviction signal available
                   on-chain, and it is invisible in any single snapshot.
        WEAKENED   size unchanged but liquidation moved CLOSER. Margin pulled
                   out, or cross-margin pressure from a different position.
        CLOSED     gone, and not near liquidation when last seen
        LIQUIDATED gone, and last seen within a whisker of its liquidation
    """

    wallet: str
    coin: str
    kind: str
    size_change_pct: float = 0.0
    liq_move_pct: float = 0.0
    prev_notional: float = 0.0
    new_notional: float = 0.0

    @property
    def is_bullish_signal(self) -> bool:
        """Did the trader express MORE confidence? Direction-agnostic."""
        return self.kind in ("ADDED", "DEFENDED")


def diff_positions(before: Sequence[Position], after: Sequence[Position],
                   last_spot: float, size_tolerance: float = 0.02,
                   liq_tolerance: float = 0.005,
                   liquidation_proximity: float = 0.01) -> list[PositionChange]:
    """Compare two sweeps of the same wallets.

    `liquidation_proximity` decides how close to its liquidation price a
    vanished position had to be for its disappearance to count as a
    liquidation rather than a voluntary exit. A closed position and a
    liquidated one look identical in the data apart from this.
    """
    prev = {p.key: p for p in before}
    curr = {p.key: p for p in after}
    changes: list[PositionChange] = []

    for key, old in prev.items():
        new = curr.get(key)

        if new is None:
            near_liq = False
            if old.liquidation_px and last_spot > 0:
                near_liq = (abs(last_spot - old.liquidation_px) / last_spot
                            <= liquidation_proximity)
            changes.append(PositionChange(
                wallet=key[0], coin=key[1],
                kind="LIQUIDATED" if near_liq else "CLOSED",
                size_change_pct=-1.0,
                prev_notional=old.notional, new_notional=0.0))
            continue

        old_sz, new_sz = abs(old.szi), abs(new.szi)
        if old_sz <= 0:
            continue
        size_change = (new_sz - old_sz) / old_sz

        if size_change > size_tolerance:
            kind = "ADDED"
        elif size_change < -size_tolerance:
            kind = "REDUCED"
        else:
            kind = "HELD"

        liq_move = 0.0
        if (kind == "HELD" and old.liquidation_px and new.liquidation_px
                and old.liquidation_px > 0):
            liq_move = (new.liquidation_px - old.liquidation_px) / old.liquidation_px
            # "Away" depends on side: a long's liquidation sits below, so
            # moving away means moving DOWN.
            moved_away = (liq_move < -liq_tolerance if old.is_long
                          else liq_move > liq_tolerance)
            moved_closer = (liq_move > liq_tolerance if old.is_long
                            else liq_move < -liq_tolerance)
            if moved_away:
                kind = "DEFENDED"
            elif moved_closer:
                kind = "WEAKENED"

        changes.append(PositionChange(
            wallet=key[0], coin=key[1], kind=kind,
            size_change_pct=size_change, liq_move_pct=liq_move,
            prev_notional=old.notional, new_notional=new.notional))

    for key, new in curr.items():
        if key not in prev:
            changes.append(PositionChange(
                wallet=key[0], coin=key[1], kind="OPENED",
                size_change_pct=1.0, prev_notional=0.0,
                new_notional=new.notional))

    return changes


# --------------------------------------------------------------------------
# cohort view
# --------------------------------------------------------------------------

@dataclass
class CohortSummary:
    coin: str
    spot: float
    n_positions: int
    long_notional: float
    short_notional: float
    fragile_long_notional: float     # notional weighted by fragility
    fragile_short_notional: float
    mean_survivability: float
    paying_funding_share: float

    @property
    def net_notional(self) -> float:
        return self.long_notional - self.short_notional

    @property
    def net_fragile(self) -> float:
        """Positive means fragile LONGS dominate, so downside has more fuel."""
        return self.fragile_long_notional - self.fragile_short_notional

    def render(self) -> str:
        total = self.long_notional + self.short_notional
        if total <= 0:
            return f"{self.coin}: nothing mapped"

        long_pct = self.long_notional / total
        pressure = ("downside -- fragile longs dominate" if self.net_fragile > 0
                    else "upside -- fragile shorts dominate")

        return "\n".join([
            f"{self.coin} cohort at {self.spot:,.4f}",
            f"  positions          {self.n_positions}",
            f"  raw positioning    {long_pct:.0%} long / {1 - long_pct:.0%} short  "
            f"(${total:,.0f})",
            f"  fragility-weighted ${self.fragile_long_notional:,.0f} long  "
            f"${self.fragile_short_notional:,.0f} short",
            f"  mean survivability {self.mean_survivability:.2f}",
            f"  paying funding     {self.paying_funding_share:.0%} of positions",
            f"  pressure           {pressure}",
            "",
            "  Raw positioning says who is there. Fragility-weighted says who",
            "  can actually be forced out, which is the one that moves price.",
        ])


def summarise(strengths: Sequence[PositionStrength], coin: str,
              spot: float) -> CohortSummary:
    longs = [s for s in strengths if s.position.is_long]
    shorts = [s for s in strengths if not s.position.is_long]

    def total(group, weighted: bool) -> float:
        return sum(s.position.notional * (s.fragility() if weighted else 1.0)
                   for s in group)

    survivabilities = [s.survivability() for s in strengths]
    paying = [s for s in strengths if s.carry_annualised > 0]

    return CohortSummary(
        coin=coin,
        spot=spot,
        n_positions=len(strengths),
        long_notional=total(longs, False),
        short_notional=total(shorts, False),
        fragile_long_notional=total(longs, True),
        fragile_short_notional=total(shorts, True),
        mean_survivability=(sum(survivabilities) / len(survivabilities)
                            if survivabilities else 0.0),
        paying_funding_share=(len(paying) / len(strengths) if strengths else 0.0),
    )


def fragility_weighter(spot: float, sigma_per_min: float,
                       horizon_minutes: float = 240.0):
    """A `weight_fn` for `build_map` that weights notional by fragility.

    Use it when you want a map of PRESSURE rather than a map of leverage.
    Positions that cannot realistically be forced out inside the horizon
    contribute almost nothing, which is the correct treatment -- they are a
    level that holds, not fuel waiting to fire.
    """
    def weight(position: Position) -> float:
        return score(position, spot, sigma_per_min, horizon_minutes).fragility()
    return weight
