"""Tape velocity, instrument economics, and whether a limit order fills.

WHAT IS HERE AND WHAT IS NOT

The tape metrics this project already has are not repeated. `FlowTape`
carries CVD, interval delta and the price/flow divergence; `Absorption`
carries aggression-versus-impact; `VolumeProfile` carries per-price delta.
This module adds the three pieces that were missing:

    Velocity      prints per second, against a baseline that does not
                  include the spike it is being compared to. Two of
                  them: one for the market right now (`velocity`), and
                  one per timeframe for the bar in progress against the
                  bars before it (`velocity_for`).

    Spec          tick size, tick value and commission per instrument,
                  and the arithmetic that says whether a target of N
                  ticks can pay for itself at all.

    Queue         whether a limit order at a price actually got filled,
                  from how much size was ahead of it and how much traded
                  through.

THE QUEUE MODEL IS THE POINT OF THIS FILE

A backtest that fills a limit order because price touched its level is
not optimistic, it is wrong, and wrong in one direction. At a touch you
are at the BACK of a queue. If the level holds and price reverses, the
queue in front of you never cleared and you were never filled -- but a
touch-fills tester books that as a winning entry at the best price of the
move. If the level breaks, the queue did clear, you were filled, and
price is now going the other way.

So the rule here is: TOUCH IS NOT A FILL. THROUGH IS A FILL. Which means
a limit order fills preferentially when it is about to be wrong, and that
asymmetry is most of the difference between a backtest that works and one
that does not.

WHAT THIS MODEL STILL CANNOT SEE, AND WHY IT IS OPTIMISTIC ANYWAY

Cancellations ahead of you are invisible in MBP data: the queue shrinks
for reasons that are not trades, and some of that shrink is in your
favour. Against that, your own order is assumed to join at the back of
the visible size, when in practice there are hidden and iceberg orders
ahead of it. The two do not cancel, and nobody knows the net. `optimism`
reports which assumptions were made so the number is never mistaken for
a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

# ---------------------------------------------------------------------------
# Instrument economics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Spec:
    """What one tick is worth here, and what a round trip costs.

    `commission` is ONE SIDE. The spread is not in it: whether you pay the
    spread depends on whether you take or post, which is the whole reason
    the queue model below exists.
    """

    symbol: str
    tick_size: float
    tick_value: float          # currency per tick, per contract/share
    commission: float = 0.0    # one side, per contract/share
    typical_spread_ticks: float = 1.0

    def taker_cost(self, qty: float = 1.0) -> float:
        """A round trip that crosses both ways: two commissions, one spread.

        One spread, not two. You give it up getting in and the exit is
        priced from where you are, so charging it twice double-counts the
        same half-turn.
        """
        return qty * (2 * self.commission
                      + self.typical_spread_ticks * self.tick_value)

    def maker_cost(self, qty: float = 1.0) -> float:
        """Both sides posted: commissions only, no spread given up."""
        return qty * 2 * self.commission

    def cost_ticks(self, taker: bool = True, qty: float = 1.0) -> float:
        c = self.taker_cost(qty) if taker else self.maker_cost(qty)
        return c / (self.tick_value * qty) if self.tick_value > 0 else 0.0

    def breakeven(self, target_ticks: float, stop_ticks: float,
                  taker: bool = True) -> float:
        """Hit rate this target and stop need, after cost.

        P(target before stop) = (S + C) / (T + S). The cost does not
        shrink with the target, so the smaller the target the larger a
        share of the total width the cost is -- which is why a five tick
        scalp needs a HIGHER hit rate than a twenty tick one, not a lower.
        """
        total = target_ticks + stop_ticks
        if total <= 0:
            return 1.0
        return min(1.0, (stop_ticks + self.cost_ticks(taker)) / total)

    def viable(self, target_ticks: float, stop_ticks: float,
               taker: bool = True, ceiling: float = 0.75) -> bool:
        """Is this target reachable at all, before any strategy exists.

        A target whose break-even rate is above `ceiling` is not a hard
        trade, it is an arithmetic problem: no entry rule supplies that
        much edge, so the configuration is refused rather than tested.
        """
        return self.breakeven(target_ticks, stop_ticks, taker) <= ceiling


ES = Spec("ES", tick_size=0.25, tick_value=12.50, commission=2.50,
          typical_spread_ticks=1.0)
MES = Spec("MES", tick_size=0.25, tick_value=1.25, commission=0.50,
           typical_spread_ticks=1.0)
NQ = Spec("NQ", tick_size=0.25, tick_value=5.00, commission=2.50,
          typical_spread_ticks=1.0)
EQUITY = Spec("EQUITY", tick_size=0.01, tick_value=0.01, commission=0.0035,
              typical_spread_ticks=1.0)

SPECS = {s.symbol: s for s in (ES, MES, NQ, EQUITY)}


# ---------------------------------------------------------------------------
# Tape velocity
# ---------------------------------------------------------------------------

# Window the burst is measured over, and the trailing window the baseline
# comes from. The baseline window ENDS where the burst window begins.
BURST_S = 15.0
BASE_S = 600.0

# Prints in the baseline window below which it is not a baseline.
MIN_BASE_PRINTS = 40


@dataclass
class Velocity:
    """Prints per second now, against what is normal here lately."""

    now_hz: float = 0.0
    base_hz: float = 0.0
    ratio: float = 1.0
    prints: int = 0
    base_prints: int = 0
    confident: bool = False
    note: str = ""

    def spiking(self, times: float = 2.0) -> bool:
        return self.confident and self.ratio >= times

    def to_dict(self) -> dict:
        return {"now_hz": round(self.now_hz, 2),
                "base_hz": round(self.base_hz, 2),
                "ratio": round(self.ratio, 2), "prints": self.prints,
                "base_prints": self.base_prints,
                "confident": self.confident, "note": self.note}


def velocity(stamps: Sequence[float], now: float | None = None,
             burst_s: float = BURST_S, base_s: float = BASE_S) -> Velocity:
    """Tape speed, measured against a baseline that excludes the burst.

    THE BASELINE MUST NOT CONTAIN THE SPIKE.

    A ten-minute baseline that runs up to the present includes the burst
    being measured. The burst then raises its own benchmark, the ratio
    compresses toward one, and the loudest bursts -- the ones worth
    acting on -- are the ones it understates most. So the baseline window
    ends where the burst window starts.

    `stamps` are trade timestamps in seconds, ascending.
    """
    out = Velocity()
    if not stamps:
        out.note = "no prints"
        return out
    end = float(now) if now is not None else float(stamps[-1])
    cut = end - burst_s
    floor = cut - base_s

    burst = [t for t in stamps if cut < t <= end]
    base = [t for t in stamps if floor < t <= cut]

    out.prints = len(burst)
    out.base_prints = len(base)
    out.now_hz = len(burst) / burst_s if burst_s > 0 else 0.0
    out.base_hz = len(base) / base_s if base_s > 0 else 0.0

    if len(base) < MIN_BASE_PRINTS:
        out.ratio = 1.0
        out.note = (f"only {len(base)} prints in the {base_s:g}s baseline -- "
                    f"not enough to call anything a spike")
        return out

    out.confident = True
    out.ratio = (out.now_hz / out.base_hz) if out.base_hz > 0 else 1.0
    out.note = (f"{out.now_hz:.1f}/s against a {out.base_hz:.1f}/s baseline "
                f"({out.ratio:.1f}x)")
    return out


# Completed bars of its own timeframe a rung's pace is measured against.
BASE_BARS = 6

# Below this many completed bars in the tape there is no "lately" for this
# timeframe, and comparing a 4-hour bar against twenty minutes of history
# would just be the 15-second burst wearing a different label.
MIN_BASE_BARS = 2

# Floors on the bar in progress. One print two seconds in is not a pace.
MIN_BURST_S = 5.0
MIN_BURST_PRINTS = 5


def velocity_for(stamps: Sequence[float], interval_s: float,
                 now: float | None = None,
                 back: int = BASE_BARS) -> Velocity:
    """How fast THIS bar is printing, against the bars before it.

    A different measurement from `velocity` above, and they answer
    different questions. `velocity` asks whether something is happening
    RIGHT NOW: fifteen seconds against ten minutes, one number for the
    whole market. This asks whether the bar in progress is busier than
    its own recent neighbours -- the current 15-minute bar against the
    last six 15-minute bars -- which is the same shape as the delta and
    the CVD sitting beside it on the row, and can be read with them.

    The two genuinely diverge, and the divergence is the point: a burst
    that is loud against ten minutes can still leave a 4-hour bar
    running slower than the four before it.

    THE BASELINE IS COMPLETED BARS ONLY. The bar being measured is never
    in its own benchmark, for the same reason the burst window is not in
    the baseline above: it would raise the bar it is being compared to
    and understate exactly the bars worth noticing.

    WHEN THE TAPE CANNOT ANSWER, IT SAYS SO. A rolling tape that holds an
    hour has not seen two completed 4-hour bars, so there is no pace for
    that rung -- `confident` stays false and `note` says why, rather than
    dividing by whatever fraction happens to be in memory and printing a
    ratio that means nothing.
    """
    out = Velocity()
    if interval_s <= 0:
        out.note = "no interval"
        return out
    if not stamps:
        out.note = "no prints"
        return out

    end = float(now) if now is not None else float(stamps[-1])
    start = end - (end % interval_s)        # this bar's open
    elapsed = max(0.0, end - start)
    floor = start - back * interval_s
    held_from = max(floor, float(stamps[0]))
    span = max(0.0, start - held_from)       # baseline seconds actually held

    burst = [t for t in stamps if start < t <= end]
    base = [t for t in stamps if held_from <= t < start]

    out.prints = len(burst)
    out.base_prints = len(base)
    out.now_hz = (len(burst) / elapsed) if elapsed > 0 else 0.0
    out.base_hz = (len(base) / span) if span > 0 else 0.0

    bars = span / interval_s
    if bars < MIN_BASE_BARS:
        out.ratio = 1.0
        out.note = (f"the tape holds {bars:.1f} completed bars of this "
                    f"timeframe -- not enough to say what normal is here")
        return out
    if elapsed < MIN_BURST_S or len(burst) < MIN_BURST_PRINTS:
        out.ratio = 1.0
        out.note = (f"{len(burst)} print{'' if len(burst) == 1 else 's'} "
                    f"{elapsed:.0f}s into the bar -- too early to call a pace")
        return out
    if len(base) < MIN_BASE_PRINTS:
        out.ratio = 1.0
        out.note = (f"only {len(base)} prints across the last "
                    f"{bars:.0f} bars -- not enough to call anything a spike")
        return out

    out.confident = True
    out.ratio = (out.now_hz / out.base_hz) if out.base_hz > 0 else 1.0
    out.note = (f"{out.now_hz:.1f}/s this bar against {out.base_hz:.1f}/s "
                f"over the last {bars:.0f} ({out.ratio:.1f}x)")
    return out


# ---------------------------------------------------------------------------
# Spread gate
# ---------------------------------------------------------------------------

def spread_ok(spread_ticks: float, target_ticks: float,
              max_share: float = 0.20) -> bool:
    """Refuse an entry whose spread eats `max_share` of the target.

    Stated as a share rather than a tick count because the same spread is
    trivial against a forty tick target and fatal against a five tick one.
    """
    if target_ticks <= 0:
        return False
    return spread_ticks <= max_share * target_ticks


# ---------------------------------------------------------------------------
# The queue
# ---------------------------------------------------------------------------

# A level is "cleared" when traded volume at it reaches this multiple of
# the size that was resting. Exactly 1.0 would fill on the last contract
# of the queue, which assumes nothing was added behind it.
CLEAR_FACTOR = 1.0


@dataclass
class Fill:
    """Whether a resting order filled, and what had to happen for it to."""

    filled: bool = False
    price: float = 0.0
    ts: float = 0.0
    ahead: float = 0.0          # size resting in front when it joined
    traded_through: float = 0.0  # volume that printed at the level after
    reason: str = ""
    adverse: bool = False       # filled because the level broke
    optimism: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"filled": self.filled, "price": round(self.price, 6),
                "ts": self.ts, "ahead": round(self.ahead, 2),
                "traded_through": round(self.traded_through, 2),
                "reason": self.reason, "adverse": self.adverse,
                "optimism": list(self.optimism)}


def queue_ahead(book, price: float, side: str) -> float:
    """Resting size at `price` on the given side, or 0 if nothing is there.

    `side` is the side YOU are posting on: "buy" rests among the bids.
    """
    levels = book.bids if side == "buy" else book.asks
    for lv in levels:
        if abs(float(lv.px) - price) < 1e-9:
            return float(lv.sz)
    return 0.0


def resting_fill(price: float, side: str, ahead: float,
                 prints: Sequence, tick: float,
                 clear_factor: float = CLEAR_FACTOR) -> Fill:
    """Did an order resting at `price` get filled by this tape.

    `prints` are (ts, px, size) trades after the order was placed.

    THE RULE: TOUCH IS NOT A FILL, THROUGH IS A FILL.

    A buy resting at the bid fills only once enough volume has printed AT
    that price to clear the size in front of it. If price touches the
    level and leaves without clearing the queue, there is no fill -- and
    a tester that books one has given itself the best price of a move it
    was never in.

    If price trades BELOW the level (for a buy), the queue necessarily
    cleared and the fill happened; it is marked `adverse`, because the
    market has gone through and the position is already losing. That
    asymmetry is the whole model: a resting order fills most reliably
    exactly when it is about to be wrong.
    """
    out = Fill(price=price, ahead=ahead)
    out.optimism = [
        "cancellations ahead are invisible, so the real queue may clear "
        "sooner than this",
        "the order is assumed to join behind all VISIBLE size, and hidden "
        "or iceberg orders ahead of it are not counted",
    ]
    long = side == "buy"
    need = ahead * clear_factor
    done = 0.0

    for p in prints:
        ts, px, sz = float(p[0]), float(p[1]), float(p[2])
        # Through the level: the queue cleared, whatever the tape showed.
        if (long and px < price - tick / 2) or \
           (not long and px > price + tick / 2):
            out.filled = True
            out.ts = ts
            out.adverse = True
            out.traded_through = done
            out.reason = (f"price traded through {price:,.4f} -- the queue "
                          f"cleared and the market kept going")
            return out
        if abs(px - price) < tick / 2:
            done += sz
            if done >= need:
                out.filled = True
                out.ts = ts
                out.traded_through = done
                out.reason = (f"{done:,.0f} traded at {price:,.4f} against "
                              f"{ahead:,.0f} resting in front")
                return out

    out.traded_through = done
    out.reason = (f"only {done:,.0f} of the {need:,.0f} needed traded at "
                  f"{price:,.4f} -- the queue never reached it")
    return out


def touch_fill(price: float, side: str, prints: Sequence,
               tick: float) -> Fill:
    """The naive model, kept so the two can be compared.

    Fills the moment price touches the level. This is what most testers
    do and it is why their limit entries look free. It exists here to be
    measured against `resting_fill`, not to be used.
    """
    out = Fill(price=price, ahead=0.0)
    out.optimism = ["fills on a touch: assumes no queue at all"]
    long = side == "buy"
    for p in prints:
        ts, px = float(p[0]), float(p[1])
        if (long and px <= price + tick / 2) or \
           (not long and px >= price - tick / 2):
            out.filled = True
            out.ts = ts
            out.reason = f"price touched {price:,.4f}"
            return out
    out.reason = f"price never reached {price:,.4f}"
    return out
