"""Directional consensus among traders who are currently winning.

The question this answers: of the wallets holding a position right now, how
many are pointed the same way, how much money is behind each side, and how are
those positions actually doing?

Everything comes from the same `clearinghouseState` sweep already being run --
no extra data source, nothing to paste in.

THE NUMBER THAT MATTERS MOST HERE IS `entry_gap_pct`.

A consensus view will happily tell you that eighty percent of winning notional
is long. What it will not tell you, unless you make it, is that those longs
got in four percent lower than where price is now. Following them at this
moment means taking the same position at a materially worse price, with their
cushion and without their entry.

That gap is the whole reason copying a winning position is not the same trade
as the winning position. So it is computed here as a first-class field, shown
with a sign, and never buried:

    entry_gap_pct > 0   you would enter WORSE than their average entry
    entry_gap_pct < 0   you would enter BETTER (rare, and worth a second look
                        -- it usually means the position is underwater)

`room_sigmas` is the companion number: how much adverse movement the consensus
side can absorb before liquidations start. High agreement with very little
room is a crowded trade, which is the setup that produces cascades rather than
continuation.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Literal, Sequence

from .bucket import Position
from .strength import PositionStrength, score_all

Direction = Literal["long", "short", "split"]


@dataclass
class TraderRow:
    """One trader's position, as it looks to someone thinking of copying it."""

    wallet: str
    is_long: bool
    notional: float
    entry_px: float
    unrealized_pnl: float
    return_on_position: float
    account_value: float
    leverage: float
    liq_distance_sigmas: float
    survivability: float
    entry_gap_pct: float          # how much worse you would enter, right now

    @property
    def side(self) -> str:
        return "long" if self.is_long else "short"

    @property
    def winning(self) -> bool:
        return self.unrealized_pnl > 0


@dataclass
class ConsensusView:
    coin: str
    spot: float

    n_traders: int
    n_winning: int
    n_losing: int

    long_notional: float
    short_notional: float
    winning_long_notional: float
    winning_short_notional: float

    direction: Direction
    strength: float               # share of WINNING notional on the dominant side
    agreement: float              # share of ALL notional on the dominant side

    avg_entry: float              # notional-weighted entry of the dominant side
    entry_gap_pct: float          # signed: positive means you enter worse
    room_sigmas: float            # median liquidation distance, dominant side
    median_leverage: float

    rows: list[TraderRow] = field(default_factory=list)

    @property
    def crowded(self) -> bool:
        """Strong agreement with little room to be wrong.

        This is the shape that produces a cascade rather than a continuation:
        everyone on one side, nobody with margin to survive a move against it.
        """
        return self.strength >= 0.75 and self.room_sigmas < 1.5

    @property
    def late(self) -> bool:
        """The move the consensus is profiting from has largely happened."""
        return self.entry_gap_pct > 0.02

    def verdict(self) -> str:
        if self.n_winning < 3:
            return ("Not enough winning positions to call a direction. Collect "
                    "more wallets or wait for the next sweep.")

        parts = [f"{self.strength:.0%} of winning notional is "
                 f"{self.direction.upper()} across {self.n_winning} positions."]

        if self.late:
            parts.append(
                f"But they entered around {self.avg_entry:,.2f} and price is "
                f"{self.spot:,.2f} — you would be {self.entry_gap_pct:+.1%} worse "
                f"than their average entry. Their cushion is not transferable.")
        elif self.entry_gap_pct < -0.005:
            parts.append(
                f"Price is {abs(self.entry_gap_pct):.1%} BELOW their average entry, "
                f"so the position is closer to underwater than the profit count "
                f"suggests. Check the individual rows.")
        else:
            parts.append(
                f"Price is close to their average entry ({self.avg_entry:,.2f}), "
                f"so you would be entering on comparable terms.")

        if self.crowded:
            parts.append(
                f"CROWDED: {self.strength:.0%} agreement with only "
                f"{self.room_sigmas:.1f} sigma of room before liquidations start. "
                f"That is the shape that cascades, not the shape that continues.")
        elif self.room_sigmas > 3:
            parts.append(
                f"The consensus side has {self.room_sigmas:.1f} sigma of room, "
                f"so it can sit through an adverse move.")

        return " ".join(parts)

    def render(self) -> str:
        total = self.long_notional + self.short_notional
        if total <= 0:
            return f"{self.coin}: no positions found"

        lines = [
            f"{self.coin} consensus at {self.spot:,.2f}",
            f"  traders          {self.n_traders}  "
            f"({self.n_winning} winning, {self.n_losing} losing)",
            f"  all positions    {self.long_notional / total:.0%} long / "
            f"{self.short_notional / total:.0%} short  (${total:,.0f})",
            f"  winners only     ${self.winning_long_notional:,.0f} long / "
            f"${self.winning_short_notional:,.0f} short",
            f"  direction        {self.direction.upper()}  "
            f"{self.strength:.0%} of winning notional",
            f"  their avg entry  {self.avg_entry:,.2f}",
            f"  your entry gap   {self.entry_gap_pct:+.2%}"
            f"{'   << you enter worse' if self.late else ''}",
            f"  room to be wrong {self.room_sigmas:.1f} sigma",
            "",
            f"  {self.verdict()}",
        ]
        return "\n".join(lines)


def _entry_gap(spot: float, entry: float, is_long: bool) -> float:
    """How much worse you would enter than they did, signed for the side.

    A long that got in lower than spot has a gap you pay; a short that got in
    higher than spot likewise. Both come back positive, so the number always
    reads the same way regardless of direction.
    """
    if entry <= 0 or spot <= 0:
        return 0.0
    raw = (spot - entry) / entry
    return raw if is_long else -raw


def build(positions: Sequence[Position], coin: str, spot: float,
          sigma_per_min: float, horizon_minutes: float = 240.0,
          min_notional: float = 0.0, min_account: float = 0.0,
          winners_only: bool = True) -> ConsensusView:
    """Compute the consensus view for one coin.

    `winners_only` decides whether the direction is called from profitable
    positions alone. That is the usual intent -- but note it is a survivorship
    filter: a trader who is up right now is not necessarily good, they may
    simply have been lucky recently or be early in a position that is about to
    turn. The filter tells you who is winning, never who is right.
    """
    on_coin = [p for p in positions
               if p.coin == coin
               and p.notional >= min_notional
               and p.account_value >= min_account
               and p.szi != 0]

    strengths: list[PositionStrength] = score_all(
        on_coin, spot, sigma_per_min, horizon_minutes)

    rows: list[TraderRow] = []
    for s in strengths:
        p = s.position
        rows.append(TraderRow(
            wallet=p.wallet,
            is_long=p.is_long,
            notional=p.notional,
            entry_px=p.entry_px,
            unrealized_pnl=p.unrealized_pnl,
            return_on_position=s.maturity,
            account_value=p.account_value,
            leverage=p.leverage,
            liq_distance_sigmas=(0.0 if s.liq_distance_sigmas == float("inf")
                                 else s.liq_distance_sigmas),
            survivability=s.survivability(),
            entry_gap_pct=_entry_gap(spot, p.entry_px, p.is_long),
        ))

    winners = [r for r in rows if r.winning]
    losers = [r for r in rows if not r.winning]

    def notional(group: Sequence[TraderRow], long: bool) -> float:
        return sum(r.notional for r in group if r.is_long == long)

    long_all, short_all = notional(rows, True), notional(rows, False)
    win_long, win_short = notional(winners, True), notional(winners, False)

    pool = winners if (winners_only and winners) else rows
    pool_long = win_long if (winners_only and winners) else long_all
    pool_short = win_short if (winners_only and winners) else short_all
    pool_total = pool_long + pool_short

    if pool_total <= 0:
        direction: Direction = "split"
        strength = 0.0
    elif pool_long > pool_short:
        direction, strength = "long", pool_long / pool_total
    elif pool_short > pool_long:
        direction, strength = "short", pool_short / pool_total
    else:
        direction, strength = "split", 0.5

    dominant = [r for r in pool
                if (r.is_long if direction == "long" else not r.is_long)] \
        if direction != "split" else list(pool)

    dom_notional = sum(r.notional for r in dominant)
    if dom_notional > 0:
        avg_entry = sum(r.entry_px * r.notional for r in dominant) / dom_notional
        entry_gap = sum(r.entry_gap_pct * r.notional for r in dominant) / dom_notional
    else:
        avg_entry = entry_gap = 0.0

    room = (statistics.median([r.liq_distance_sigmas for r in dominant])
            if dominant else 0.0)
    med_lev = (statistics.median([r.leverage for r in dominant if r.leverage > 0])
               if any(r.leverage > 0 for r in dominant) else 0.0)

    all_total = long_all + short_all
    agreement = (max(long_all, short_all) / all_total) if all_total > 0 else 0.0

    rows.sort(key=lambda r: -r.notional)

    return ConsensusView(
        coin=coin, spot=spot,
        n_traders=len(rows), n_winning=len(winners), n_losing=len(losers),
        long_notional=long_all, short_notional=short_all,
        winning_long_notional=win_long, winning_short_notional=win_short,
        direction=direction, strength=strength, agreement=agreement,
        avg_entry=avg_entry, entry_gap_pct=entry_gap,
        room_sigmas=room, median_leverage=med_lev,
        rows=rows,
    )
