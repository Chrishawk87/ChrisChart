"""Price structure from candles: swings, trend, zones, VWAP.

This is the half of the picture the liquidity work was missing. Depth and
absorption tell you what is resting AT a price. They do not tell you which
prices matter. Structure does that, and the two are only useful together --
a supply zone with no resting size gaps through, and a shelf at a price with
no structural meaning is just a big order somebody will pull.

Everything here is computed from OHLCV, so it is testable against candles
written by hand and carries no opinion about the venue.

WHAT IS DELIBERATELY SIMPLE, AND WHY

Swing detection is an n-bar fractal: a high with no higher high within n bars
either side. Every fancier method -- zigzag, ATR-scaled, fractal-of-fractals --
introduces a lookahead or a tuned parameter that makes backtests look better
than live trading. The fractal has one parameter, it is honest about needing
n bars of hindsight to confirm a swing, and `Swing.confirmed_at` says exactly
when it became knowable. Anything that reads a swing before that index is
cheating, and the validation machinery will happily let you cheat if this
module does not stop you.

Trend is the last two swing highs and last two swing lows: higher highs with
higher lows is an uptrend, lower highs with lower lows is a downtrend, and
anything else is a range. It refuses to call a trend from fewer than two of
each rather than guessing from one.

THE STRUCTURE SHIFT

The thing worth detecting is the change of character: in a downtrend, a close
above the most recent lower high. That is the signal that the sequence making
lower highs has stopped, and it is what a counter-trend entry waits for. It is
distinct from a break of structure, which is a close BEYOND the trend's last
extreme in the trend's own direction -- continuation, not reversal. Confusing
the two inverts the trade, so they are separate types here.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

Direction = Literal["up", "down", "range"]
SwingKind = Literal["high", "low"]
ZoneKind = Literal["supply", "demand"]

# Bars either side of a pivot for it to count as a swing. Three is the usual
# compromise: one is noise, five misses most intraday structure.
DEFAULT_SWING_STRENGTH = 3


@dataclass(frozen=True)
class Candle:
    ts: float                 # open time, seconds
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    trades: int = 0

    @property
    def body_top(self) -> float:
        return max(self.open, self.close)

    @property
    def body_bottom(self) -> float:
        return min(self.open, self.close)

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def bullish(self) -> bool:
        return self.close > self.open

    @property
    def bearish(self) -> bool:
        return self.close < self.open

    @property
    def typical(self) -> float:
        return (self.high + self.low + self.close) / 3.0


@dataclass(frozen=True)
class Swing:
    index: int                # bar the swing sits on
    ts: float
    px: float
    kind: SwingKind
    confirmed_at: int         # bar it became knowable -- NOT `index`

    @property
    def is_high(self) -> bool:
        return self.kind == "high"


@dataclass
class Structure:
    """The trend read on one timeframe."""

    direction: Direction
    swings: list[Swing] = field(default_factory=list)
    last_high: Swing | None = None
    last_low: Swing | None = None
    # The level whose breach would end this trend's character.
    invalidation: float | None = None
    note: str = ""

    @property
    def highs(self) -> list[Swing]:
        return [s for s in self.swings if s.is_high]

    @property
    def lows(self) -> list[Swing]:
        return [s for s in self.swings if not s.is_high]

    def describe(self) -> str:
        if self.direction == "range":
            return f"range — {self.note}"
        arrow = "higher highs and higher lows" if self.direction == "up" \
            else "lower highs and lower lows"
        inv = (f", invalidated on a close beyond {self.invalidation:,.2f}"
               if self.invalidation else "")
        return f"{self.direction} ({arrow}){inv}"


@dataclass(frozen=True)
class Shift:
    """A break of structure or a change of character."""

    kind: Literal["BOS", "CHoCH"]
    direction: Direction        # which way the break points
    index: int
    ts: float
    level: float                # the swing that was taken out
    close: float

    def describe(self) -> str:
        what = ("continuation" if self.kind == "BOS"
                else "character change — the sequence broke")
        return (f"{self.kind} {self.direction} at {self.close:,.2f} "
                f"through {self.level:,.2f} ({what})")


@dataclass(frozen=True)
class Zone:
    """A supply or demand zone: the origin of a move that broke structure."""

    kind: ZoneKind
    low: float
    high: float
    index: int
    ts: float
    impulse_bps: float          # how hard price left the zone
    tested: int = 0             # times price has returned since

    @property
    def mid(self) -> float:
        return (self.low + self.high) / 2.0

    @property
    def height_bps(self) -> float:
        return (self.high - self.low) / self.mid * 10_000.0 if self.mid else 0.0

    def contains(self, px: float) -> bool:
        return self.low <= px <= self.high

    def distance_bps(self, px: float) -> float:
        """Signed distance from px to the zone, 0 when inside it."""
        if self.contains(px) or px <= 0:
            return 0.0
        edge = self.low if px < self.low else self.high
        return (edge - px) / px * 10_000.0

    @property
    def fresh(self) -> bool:
        """Untested zones are the ones worth trading.

        Each test consumes the resting orders that made the zone a zone. By
        the third touch there is usually nothing left to react against, which
        is why a level 'stops working' after it has worked twice.
        """
        return self.tested == 0


@dataclass
class VWAP:
    value: float
    upper_1: float
    lower_1: float
    upper_2: float
    lower_2: float
    anchor_ts: float
    bars: int

    def band_of(self, px: float) -> str:
        if px >= self.upper_2:
            return "above +2σ"
        if px >= self.upper_1:
            return "+1σ to +2σ"
        if px <= self.lower_2:
            return "below -2σ"
        if px <= self.lower_1:
            return "-1σ to -2σ"
        return "inside ±1σ"


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def parse_candles(raw: Iterable[dict]) -> list[Candle]:
    """Hyperliquid `candleSnapshot` entries -> Candles, oldest first.

    Fields are `t` (open ms), `o`, `h`, `l`, `c`, `v`, `n`, all as strings.
    Bad entries are dropped rather than raising: one malformed candle should
    not cost you the timeframe.
    """
    out: list[Candle] = []
    for e in raw or []:
        if not isinstance(e, dict):
            continue
        try:
            c = Candle(
                ts=float(e.get("t", 0) or 0) / 1000.0,
                open=float(e["o"]), high=float(e["h"]),
                low=float(e["l"]), close=float(e["c"]),
                volume=float(e.get("v", 0) or 0),
                trades=int(e.get("n", 0) or 0),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if c.high < c.low or c.high <= 0:
            continue
        out.append(c)
    out.sort(key=lambda c: c.ts)
    return out


# --------------------------------------------------------------------------
# swings
# --------------------------------------------------------------------------

def swings(candles: Sequence[Candle],
           strength: int = DEFAULT_SWING_STRENGTH) -> list[Swing]:
    """n-bar fractal pivots.

    A swing high is a bar whose high is not exceeded within `strength` bars
    either side. Ties are resolved toward the EARLIER bar, so a double top
    reports the first peak rather than inventing two swings a few bars apart.

    `confirmed_at` is index + strength: the bar at which you could actually
    have known. Reading a swing before that is lookahead, and it is the most
    common way a structure backtest lies to you.
    """
    n = len(candles)
    if n < strength * 2 + 1:
        return []

    out: list[Swing] = []
    for i in range(strength, n - strength):
        window = candles[i - strength: i + strength + 1]
        c = candles[i]

        highs = [w.high for w in window]
        if c.high == max(highs) and not any(
                w.high == c.high for w in window[:strength]):
            out.append(Swing(index=i, ts=c.ts, px=c.high, kind="high",
                             confirmed_at=i + strength))

        lows = [w.low for w in window]
        if c.low == min(lows) and not any(
                w.low == c.low for w in window[:strength]):
            out.append(Swing(index=i, ts=c.ts, px=c.low, kind="low",
                             confirmed_at=i + strength))

    out.sort(key=lambda s: (s.index, s.kind))
    return out


def structure(candles: Sequence[Candle],
              strength: int = DEFAULT_SWING_STRENGTH) -> Structure:
    """Classify the trend from the swing sequence.

    Two highs and two lows minimum. With fewer, the honest answer is that
    there is not enough structure to call, not a coin flip dressed up as an
    uptrend.
    """
    sw = swings(candles, strength)
    highs = [s for s in sw if s.is_high]
    lows = [s for s in sw if not s.is_high]

    last_high = highs[-1] if highs else None
    last_low = lows[-1] if lows else None

    if len(highs) < 2 or len(lows) < 2:
        return Structure(direction="range", swings=sw,
                         last_high=last_high, last_low=last_low,
                         note=f"only {len(highs)} highs and {len(lows)} lows — "
                              f"not enough to call a trend")

    hh = highs[-1].px > highs[-2].px
    hl = lows[-1].px > lows[-2].px
    lh = highs[-1].px < highs[-2].px
    ll = lows[-1].px < lows[-2].px

    if hh and hl:
        return Structure(direction="up", swings=sw, last_high=last_high,
                         last_low=last_low, invalidation=lows[-1].px)
    if lh and ll:
        return Structure(direction="down", swings=sw, last_high=last_high,
                         last_low=last_low, invalidation=highs[-1].px)

    mixed = ("higher high but lower low — expansion, not trend"
             if hh and ll else
             "lower high but higher low — compression, not trend"
             if lh and hl else "swings disagree")
    return Structure(direction="range", swings=sw, last_high=last_high,
                     last_low=last_low, note=mixed)


def last_shift(candles: Sequence[Candle],
               strength: int = DEFAULT_SWING_STRENGTH) -> Shift | None:
    """The most recent break of structure or change of character.

    BOS   a close beyond the trend's last extreme IN the trend direction.
          Continuation.
    CHoCH a close beyond the last OPPOSING swing. The sequence has broken,
          which is what a counter-trend entry is waiting for.

    Only closes count. An intrabar poke through a level that closes back
    inside is the level holding, not breaking, and treating wicks as breaks
    turns every stop run into a false signal.
    """
    st = structure(candles, strength)
    if st.direction == "range" or not st.swings:
        return None

    for i in range(len(candles) - 1, -1, -1):
        c = candles[i]
        # Only swings confirmed before this bar are knowable here.
        known = [s for s in st.swings if s.confirmed_at <= i and s.index < i]
        highs = [s for s in known if s.is_high]
        lows = [s for s in known if not s.is_high]
        if not highs or not lows:
            continue

        if st.direction == "down":
            if c.close > highs[-1].px:
                return Shift(kind="CHoCH", direction="up", index=i, ts=c.ts,
                             level=highs[-1].px, close=c.close)
            if c.close < lows[-1].px:
                return Shift(kind="BOS", direction="down", index=i, ts=c.ts,
                             level=lows[-1].px, close=c.close)
        else:
            if c.close < lows[-1].px:
                return Shift(kind="CHoCH", direction="down", index=i, ts=c.ts,
                             level=lows[-1].px, close=c.close)
            if c.close > highs[-1].px:
                return Shift(kind="BOS", direction="up", index=i, ts=c.ts,
                             level=highs[-1].px, close=c.close)
    return None


# --------------------------------------------------------------------------
# zones
# --------------------------------------------------------------------------

def zones(candles: Sequence[Candle], strength: int = DEFAULT_SWING_STRENGTH,
          min_impulse_bps: float = 30.0, limit: int = 8) -> list[Zone]:
    """Supply and demand zones: the origin candles of structural breaks.

    A supply zone is the last UP candle before a down move that took out a
    prior swing low. The logic is that whoever sold there sold enough to
    reverse the market, they are unlikely to have filled their whole order,
    and what is left is still resting there.

    That is a story, not a proven mechanism. What makes it testable is the
    part this module CAN check: whether resting size is actually there now.
    `confluence.py` does that join, and a zone with nothing in it is just a
    rectangle on a chart.

    Zones are returned newest first and counted for tests, because a zone
    that has already been traded through has had its orders consumed.
    """
    sw = swings(candles, strength)
    if not sw:
        return []

    out: list[Zone] = []
    for i in range(strength + 1, len(candles)):
        c = candles[i]
        known = [s for s in sw if s.confirmed_at <= i and s.index < i]
        highs = [s for s in known if s.is_high]
        lows = [s for s in known if not s.is_high]

        broke_down = lows and c.close < lows[-1].px
        broke_up = highs and c.close > highs[-1].px
        if not (broke_down or broke_up):
            continue

        kind: ZoneKind = "supply" if broke_down else "demand"
        want_bullish = broke_down        # supply comes from the last up candle

        origin = None
        for j in range(i - 1, max(i - 12, -1), -1):
            if candles[j].bullish if want_bullish else candles[j].bearish:
                origin = j
                break
        if origin is None:
            continue

        o = candles[origin]
        if kind == "supply":
            lo, hi = o.body_top, max(o.high, o.body_top)
        else:
            lo, hi = min(o.low, o.body_bottom), o.body_bottom
        if hi <= lo:
            lo, hi = min(o.low, o.high), max(o.low, o.high)
        if hi <= lo:
            continue

        mid = (lo + hi) / 2.0
        impulse = abs(c.close - mid) / mid * 10_000.0 if mid else 0.0
        if impulse < min_impulse_bps:
            continue

        tested = sum(1 for later in candles[i + 1:]
                     if later.low <= hi and later.high >= lo)

        out.append(Zone(kind=kind, low=lo, high=hi, index=origin, ts=o.ts,
                        impulse_bps=impulse, tested=tested))

    # Collapse overlapping zones of the same kind, keeping the newest.
    out.sort(key=lambda z: -z.index)
    kept: list[Zone] = []
    for z in out:
        if any(k.kind == z.kind and k.low <= z.high and z.low <= k.high
               for k in kept):
            continue
        kept.append(z)
        if len(kept) >= limit:
            break
    return kept


# --------------------------------------------------------------------------
# VWAP
# --------------------------------------------------------------------------

def vwap(candles: Sequence[Candle], anchor_index: int = 0) -> VWAP | None:
    """Volume-weighted average price with deviation bands.

    The bands are volume-weighted standard deviations of typical price around
    VWAP, which is the usual construction. With no volume in the candles this
    returns None rather than silently falling back to an unweighted mean --
    an unweighted "VWAP" is just a moving average with a misleading name.
    """
    window = list(candles[anchor_index:])
    if not window:
        return None

    total_v = sum(c.volume for c in window)
    if total_v <= 0:
        return None

    mean = sum(c.typical * c.volume for c in window) / total_v
    var = sum(c.volume * (c.typical - mean) ** 2 for c in window) / total_v
    sd = var ** 0.5

    return VWAP(value=mean, upper_1=mean + sd, lower_1=mean - sd,
                upper_2=mean + 2 * sd, lower_2=mean - 2 * sd,
                anchor_ts=window[0].ts, bars=len(window))


def session_anchor(candles: Sequence[Candle], session_seconds: float = 86_400.0
                   ) -> int:
    """Index of the first candle in the current session."""
    if not candles:
        return 0
    last = candles[-1].ts
    start = last - (last % session_seconds)
    for i, c in enumerate(candles):
        if c.ts >= start:
            return i
    return 0


def atr(candles: Sequence[Candle], period: int = 14) -> float:
    """Average true range. Used to size zones and stops in the confluence
    layer rather than picking a percentage out of the air."""
    if len(candles) < 2:
        return 0.0
    trs = []
    for prev, cur in zip(candles[:-1], candles[1:]):
        trs.append(max(cur.high - cur.low,
                       abs(cur.high - prev.close),
                       abs(cur.low - prev.close)))
    if not trs:
        return 0.0
    tail = trs[-period:]
    return statistics.fmean(tail)
