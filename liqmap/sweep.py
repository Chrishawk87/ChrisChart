"""Every take-profit against every stop, over the same signals.

THE ONE THING THAT MAKES THIS CHEAP

A signal does not know where its target is. Book, delta and price point a
direction at a price at a time -- that is the whole decision, and it is
identical whether the target is four ticks away or forty. So the expensive
part happens once: work out where every signal fired. Then scoring a
hundred different target-and-stop pairs is a hundred cheap walks over the
same bars.

Which is why the grid is worth having. Testing one target teaches you about
that target. Testing forty tells you the SHAPE -- whether results fall off a
cliff past six ticks, whether a wider stop earns its cost or just delays the
same loss, whether anything at all works or the whole surface is noise
around zero. One number cannot show you that and a surface can.

HOW A TRADE IS SETTLED

Bars are walked in order from the signal forward. The first bar that touches
a level ends it.

    A BAR THAT TOUCHES BOTH IS A STOP.

That is the rule that decides whether this tool tells the truth. Inside one
bar, OHLC does not say which came first. Taking the good one is how a grid
search produces a beautiful surface that evaporates the moment real money
touches it -- and it flatters the tightest stops most, which are exactly the
cells that would otherwise look best. One minute bars are used rather than
the signal's own timeframe, so the ambiguous window is a minute wide instead
of fifteen.

Nothing touched inside the horizon is closed at the last bar's close and
recorded as a timeout. Those are not dropped. Discarding unresolved trades
is the other classic way to manufacture an edge, because the ones that never
reach a target are disproportionately the ones that went nowhere.

EVERY CELL IS NET OF COST

Round-trip cost comes off every trade, win or lose. A four-tick target at
two basis points of cost is a different proposition from the same target
free, and the grid should show you where cost eats the edge rather than
hiding it in a footnote.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

Side = Literal["long", "short"]
Reason = Literal["target", "stop", "timeout", "candle_end"]

# A grid bigger than this is a mistake in the box, not a research plan.
MAX_CELLS = 900
# Bars walked forward before a trade is marked out at the close.
DEFAULT_HORIZON = 60


@dataclass(frozen=True)
class Signal:
    """A direction at a price at a time. It knows nothing about targets."""

    ts: float
    coin: str
    interval: str
    side: Side
    entry: float
    agreeing: int = 0
    against: int = 0
    shape: str = ""
    # When the candle this fired on closes. The trade ends there whatever
    # price is doing -- see `resolve`.
    deadline: float = 0.0
    # Volume behind the move against normal for this market. None where the
    # candle carried no reading, which is not the same as a quiet one.
    effort: float | None = None
    net: float = 0.0
    book: float = 0.0
    delta: float = 0.0
    price: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"ts": self.ts, "coin": self.coin, "interval": self.interval,
                "side": self.side, "entry": self.entry,
                "agreeing": self.agreeing, "against": self.against,
                "shape": self.shape, "net": round(self.net, 4)}


@dataclass(frozen=True)
class Bar:
    ts: float
    open: float
    high: float
    low: float
    close: float


def resolve(side: Side, entry: float, tp_bps: float, sl_bps: float,
            bars: Sequence[Bar], horizon: int = DEFAULT_HORIZON,
            deadline: float = 0.0, exit_before_s: float = 0.0
            ) -> tuple[Reason, float, int]:
    """Walk forward until a level is touched, or the candle closes.

    `deadline` is when the signal's own candle ends. A trade never outlives
    it: the position was opened on one candle's reading, and once that
    candle is closed the reading it rests on has expired. A five minute
    trade lasts five minutes.

    That is not a risk rule bolted on, it is what makes the grid mean
    anything -- every candle is an independent test, and a trade allowed to
    run into the next one is claiming a result the next candle's signal
    should have earned.
    """
    if entry <= 0 or not bars:
        return ("timeout", entry, 0)

    sgn = 1.0 if side == "long" else -1.0
    target = entry * (1 + sgn * tp_bps / 10_000.0)
    stop = entry * (1 - sgn * sl_bps / 10_000.0)

    out_at = (deadline - exit_before_s) if deadline else 0.0
    last_ok = bars[0]
    for i, b in enumerate(bars[:horizon]):
        # Out before the close, matching the live agent. A grid that holds
        # to the closing print is testing a trade nobody can take.
        if out_at and b.ts >= out_at:
            return ("candle_end", last_ok.close, max(1, i))
        last_ok = b
        if side == "long":
            hit_stop = b.low <= stop
            hit_target = b.high >= target
        else:
            hit_stop = b.high >= stop
            hit_target = b.low <= target
        # Both inside one bar: nothing in OHLC orders them, so the loss is
        # the only assumption that does not flatter.
        if hit_stop:
            return ("stop", stop, i + 1)
        if hit_target:
            return ("target", target, i + 1)

    last = bars[min(len(bars), horizon) - 1]
    return ("timeout", last.close, min(len(bars), horizon))


@dataclass
class Cell:
    """One take-profit / stop pair, scored over every signal."""

    tp_bps: float
    sl_bps: float
    cost_bps: float = 0.0
    n: int = 0
    targets: int = 0
    stops: int = 0
    timeouts: int = 0
    expiries: int = 0
    wins: int = 0
    net_sum: float = 0.0
    net_sq: float = 0.0
    bars_sum: int = 0

    def add(self, reason: Reason, net: float, bars_held: int) -> None:
        self.n += 1
        self.net_sum += net
        self.net_sq += net * net
        self.bars_sum += bars_held
        self.wins += 1 if net > 0 else 0
        if reason == "target":
            self.targets += 1
        elif reason == "stop":
            self.stops += 1
        elif reason == "candle_end":
            self.expiries += 1
        else:
            self.timeouts += 1

    @property
    def expectancy(self) -> float:
        return self.net_sum / self.n if self.n else 0.0

    @property
    def hit_rate(self) -> float:
        return self.wins / self.n if self.n else 0.0

    @property
    def sd(self) -> float:
        if self.n < 2:
            return 0.0
        m = self.expectancy
        return math.sqrt(max(0.0, (self.net_sq - self.n * m * m)
                             / (self.n - 1)))

    @property
    def ci(self) -> tuple[float, float]:
        """Interval on the mean. A grid's best cell is usually luck."""
        if self.n < 2:
            return (float("-inf"), float("inf"))
        se = self.sd / math.sqrt(self.n)
        return (self.expectancy - 1.96 * se, self.expectancy + 1.96 * se)

    @property
    def rr(self) -> float:
        return round(self.tp_bps / self.sl_bps, 2) if self.sl_bps else 0.0

    @property
    def breakeven(self) -> float:
        """Hit rate this pair needs, cost included. The shape, not a result."""
        denom = self.tp_bps + self.sl_bps
        return ((self.sl_bps + self.cost_bps) / denom) if denom else 0.0

    @property
    def significant(self) -> bool:
        """Is the whole interval on one side of zero?"""
        lo, hi = self.ci
        return self.n >= 30 and (lo > 0 or hi < 0)

    def to_dict(self) -> dict[str, Any]:
        lo, hi = self.ci
        return {
            "tp_bps": round(self.tp_bps, 2), "sl_bps": round(self.sl_bps, 2),
            "rr": self.rr, "n": self.n, "wins": self.wins,
            "targets": self.targets, "stops": self.stops,
            "timeouts": self.timeouts, "expiries": self.expiries,
            "hit_rate": round(self.hit_rate, 4),
            "breakeven": round(self.breakeven, 4),
            "expectancy": round(self.expectancy, 3),
            "total": round(self.net_sum, 1),
            "sd": round(self.sd, 2),
            "ci_low": round(lo, 3) if self.n >= 2 else None,
            "ci_high": round(hi, 3) if self.n >= 2 else None,
            "significant": self.significant,
            "avg_bars": round(self.bars_sum / self.n, 1) if self.n else 0.0,
        }


def axis(start: float, stop: float, step: float) -> list[float]:
    """Inclusive range for a box the operator typed into."""
    if step <= 0 or stop < start:
        return [round(start, 4)] if start > 0 else []
    out, v = [], start
    while v <= stop + 1e-9:
        out.append(round(v, 4))
        v += step
    return out


def _index(bars: Sequence[Bar]) -> list[float]:
    return [b.ts for b in bars]


def run(signals: Sequence[Signal], bars: Sequence[Bar],
        tp_values: Sequence[float], sl_values: Sequence[float],
        cost_bps: float = 0.0, horizon: int = DEFAULT_HORIZON,
        min_agreeing: int = 0, max_against: int = 3,
        side: str = "both", min_effort: float = 0.0,
        unanimous: bool = False, exit_before_s: float = 0.0
        ) -> dict[str, Any]:
    """Score every pair over every signal.

    `min_agreeing` and `max_against` are for asking the question rather than
    enforcing an answer -- "did two-of-three actually pay" is a slice of the
    result, not a rule applied before the result exists.
    """
    tp_values = [v for v in tp_values if v > 0]
    sl_values = [v for v in sl_values if v > 0]
    if not tp_values or not sl_values:
        return {"ok": False, "detail": "give me a target and a stop range"}
    if len(tp_values) * len(sl_values) > MAX_CELLS:
        return {"ok": False,
                "detail": (f"{len(tp_values)} x {len(sl_values)} is "
                           f"{len(tp_values) * len(sl_values)} cells — "
                           f"widen the step, the cap is {MAX_CELLS}")}

    def keep(s: Signal) -> bool:
        if s.agreeing < min_agreeing or s.against > max_against:
            return False
        if side != "both" and s.side != side:
            return False
        if unanimous and not (s.agreeing == 3 and s.against == 0):
            return False
        if min_effort > 0:
            # A candle with no volume reading is excluded rather than
            # treated as zero: absent is not the same as quiet, and
            # counting it as a refusal would flatter the filter.
            if s.effort is None or s.effort < min_effort:
                return False
        return True

    picked = [s for s in signals if keep(s)]
    if not picked:
        return {"ok": False, "signals": 0,
                "detail": ("no signals match that filter — try relaxing it, "
                           "or leave the feed running for more candles"),
                "considered": len(signals)}

    bars = sorted(bars, key=lambda b: b.ts)
    stamps = _index(bars)

    # Each signal's forward window, sliced once and reused by every cell.
    windows: list[tuple[Signal, Sequence[Bar]]] = []
    for s in picked:
        i = bisect.bisect_right(stamps, s.ts)
        w = bars[i:i + horizon]
        if w:
            windows.append((s, w))

    if not windows:
        return {"ok": False,
                "detail": ("signals found, but no price bars after them — "
                           "fetch more history for this market"),
                "signals": len(picked)}

    cells = [Cell(tp_bps=tp, sl_bps=sl, cost_bps=cost_bps)
             for tp in tp_values for sl in sl_values]
    for cell in cells:
        for s, window in windows:
            reason, exit_px, held = resolve(s.side, s.entry, cell.tp_bps,
                                            cell.sl_bps, window, horizon,
                                            deadline=s.deadline,
                                            exit_before_s=exit_before_s)
            raw = (exit_px - s.entry) / s.entry * 10_000.0
            gross = raw if s.side == "long" else -raw
            cell.add(reason, gross - cost_bps, held)

    rows = [c.to_dict() for c in cells]
    best = max(cells, key=lambda c: c.expectancy)
    worst = min(cells, key=lambda c: c.expectancy)

    return {
        "ok": True,
        "cells": rows,
        "tp_values": tp_values,
        "sl_values": sl_values,
        "signals": len(windows),
        "considered": len(signals),
        "skipped": len(picked) - len(windows),
        "cost_bps": cost_bps,
        "horizon": horizon,
        "best": best.to_dict(),
        "worst": worst.to_dict(),
        "verdict": _verdict(cells, best, len(windows)),
    }


def _verdict(cells: Sequence[Cell], best: Cell, n: int) -> str:
    """What the surface says, including when it says nothing.

    The best cell of a grid is the maximum of many noisy numbers, so it is
    biased upward by the searching itself. Saying that out loud is the
    difference between a research tool and a way to lose money confidently.
    """
    positive = sum(1 for c in cells if c.expectancy > 0)
    share = positive / len(cells) if cells else 0.0
    lo, hi = best.ci

    if n < 30:
        return (f"{n} signals is too few for any of this to mean anything. "
                f"The surface will move a lot as more arrive.")

    head = (f"Best cell: {best.tp_bps:.0f}bps target against a "
            f"{best.sl_bps:.0f}bps stop, {best.expectancy:+.2f}bps a trade "
            f"over {best.n}.")

    if not best.significant:
        return (head + f" Its honest range is {lo:+.2f} to {hi:+.2f}bps, "
                f"which straddles zero — on this data that cell is not "
                f"distinguishable from breaking even.")

    if share < 0.25:
        return (head + f" But only {share:.0%} of the grid is positive at "
                f"all, so this looks like the best corner of a bad surface "
                f"rather than a setting worth trusting.")

    if share > 0.8:
        return (head + f" {share:.0%} of the whole grid is positive, which "
                f"is the encouraging shape: the edge is in the signal, not "
                f"in one lucky pair of levels.")

    return (head + f" {share:.0%} of the grid is positive. Look at whether "
            f"the good cells sit together — a solid region is a real "
            f"preference, a lone bright square is usually noise.")


def surface(result: dict[str, Any], metric: str = "expectancy"
            ) -> list[list[float | None]]:
    """The grid as rows of stops by columns of targets, for drawing."""
    if not result.get("ok"):
        return []
    tps = result["tp_values"]
    sls = result["sl_values"]
    by = {(c["tp_bps"], c["sl_bps"]): c for c in result["cells"]}
    out = []
    for sl in sls:
        row = []
        for tp in tps:
            c = by.get((round(tp, 2), round(sl, 2)))
            row.append(c.get(metric) if c else None)
        out.append(row)
    return out
