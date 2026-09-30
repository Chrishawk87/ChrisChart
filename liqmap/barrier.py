"""The triple barrier: take profit, stop, and a clock.

WHY THIS REPLACES "DID THE CANDLE CLOSE UP"

Next-candle direction is not the trade. The trade is: enter here, and find
out whether +5 ticks arrives before -3 ticks, within some number of bars.
Those are different questions with different answers, and the second is the
one that pays. A read can call the close correctly and still lose the race,
and it can call the close wrongly and still win it.

It also changes the arithmetic in your favour, which is the point of an
asymmetric target. A symmetric five-tick target and five-tick stop needs
54% after costs. Five against three needs 42.5%. That is not a rounding
difference -- it is the difference between needing an edge and needing to
merely not be wrong too often.

WHAT MAKES THIS HARDER THAN IT LOOKS

The horizon runs past the end of the bar the decision was made on, so the
outcome has to be assembled from several stored bars. Each stored path is
in basis points from ITS OWN open, so chaining them means converting back
to absolute price and rebasing on the entry. Concatenating the stored basis
point figures directly would silently treat every bar as if it started
where the first one did, which drifts further from the truth with every bar
added and is invisible in the output.

THE TIE RULE, AGAIN

A bucket holding both barriers is scored as the stop. Same rule as the
reach test, same reason: a two-second bucket does not record the sequence
of ticks inside itself, and resolving ties in the tester's favour is how a
backtest arrives at a number nobody can trade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

Verdict = Literal["target", "stop", "timeout"]

# ES, one contract.
ES_TICK_USD = 12.50
ES_RT_COST_USD = 5.00


def breakeven(take_ticks: float, stop_ticks: float,
              cost_usd: float = ES_RT_COST_USD,
              tick_usd: float = ES_TICK_USD) -> float:
    """Hit rate needed to break even, after costs.

    Costs land on BOTH sides -- they shrink the win and deepen the loss --
    which is why a cheap-looking round trip moves the bar more than it
    looks like it should.
    """
    win = take_ticks * tick_usd - cost_usd
    lose = stop_ticks * tick_usd + cost_usd
    if win <= 0:
        return 1.0                     # target smaller than costs
    return lose / (win + lose)


def expectancy_usd(rate: float, take_ticks: float, stop_ticks: float,
                   cost_usd: float = ES_RT_COST_USD,
                   tick_usd: float = ES_TICK_USD) -> float:
    """Dollars per trade at a given hit rate.

    The verdict, rather than the hit rate. With asymmetric barriers a hit
    rate on its own says nothing -- 45% is a loss at 3:5 and a profit at
    5:3, and the number alone cannot tell you which.
    """
    win = take_ticks * tick_usd - cost_usd
    lose = stop_ticks * tick_usd + cost_usd
    return rate * win - (1.0 - rate) * lose


def chain(slices: Sequence, i: int, bars: int) -> list[tuple[float, float]]:
    """Path from slice `i`'s open, running `bars` bars forward, in bps.

    Rebased properly: each stored path is relative to its own bar's open, so
    it is converted to absolute price and back against the entry.
    """
    if i < 0 or i >= len(slices):
        return []
    entry = slices[i].open
    if entry <= 0:
        return []

    out: list[tuple[float, float]] = []
    for k in range(i, min(i + bars, len(slices))):
        s = slices[k]
        base = s.open
        if base <= 0:
            continue
        for hi_bps, lo_bps in s.path:
            hi = base * (1.0 + hi_bps / 10_000.0)
            lo = base * (1.0 + lo_bps / 10_000.0)
            out.append((((hi - entry) / entry) * 10_000.0,
                        ((lo - entry) / entry) * 10_000.0))
    return out


def resolve(path: Sequence[tuple[float, float]], side: int,
            take_bps: float, stop_bps: float) -> Verdict:
    """Which barrier came first.

    `side` is +1 long, -1 short. Both thresholds are positive distances.
    """
    if not path or side == 0 or take_bps <= 0 or stop_bps <= 0:
        return "timeout"
    for hi, lo in path:
        fav = hi if side > 0 else -lo
        adv = -lo if side > 0 else hi
        if adv >= stop_bps:
            return "stop"
        if fav >= take_bps:
            return "target"
    return "timeout"


@dataclass
class Result:
    """What a rule did over a set of bars."""

    n: int
    target: int
    stop: int
    timeout: int
    take_ticks: float
    stop_ticks: float
    cost_usd: float = ES_RT_COST_USD
    tick_usd: float = ES_TICK_USD

    @property
    def resolved(self) -> int:
        return self.target + self.stop

    @property
    def rate(self) -> float | None:
        return self.target / self.resolved if self.resolved else None

    @property
    def needs(self) -> float:
        return breakeven(self.take_ticks, self.stop_ticks,
                         self.cost_usd, self.tick_usd)

    @property
    def per_trade(self) -> float | None:
        """Dollars per RESOLVED trade.

        Timeouts are excluded rather than scored as scratches. A timeout is
        a position still open when the clock ran out, and what it is worth
        depends on how it is closed -- which is an exit rule, not a fact
        about the signal. Counting them as zero would quietly reward a rule
        that mostly does nothing.
        """
        r = self.rate
        if r is None:
            return None
        return expectancy_usd(r, self.take_ticks, self.stop_ticks,
                              self.cost_usd, self.tick_usd)

    @property
    def total(self) -> float | None:
        p = self.per_trade
        return None if p is None else p * self.resolved

    def to_dict(self) -> dict:
        return {"n": self.n, "target": self.target, "stop": self.stop,
                "timeout": self.timeout, "resolved": self.resolved,
                "rate": self.rate, "needs": self.needs,
                "per_trade": self.per_trade, "total": self.total}


def control(slices: Sequence, sides: Sequence[int], take_ticks: float,
            stop_ticks: float, bars: int, tick: float = 0.25,
            trials: int = 25, seed: int = 0,
            cost_usd: float = ES_RT_COST_USD,
            tick_usd: float = ES_TICK_USD) -> dict:
    """What a RANDOM signal scores on exactly the same bars.

    THE BASELINE IS NOT THE BREAKEVEN RATE.

    With asymmetric barriers, expectancy depends on the shape of the paths
    as much as on directional skill. A rule that happens to trade bars with
    favourable geometry clears the theoretical breakeven while knowing
    nothing -- and "45.4%, needs 42.5%, PAYS" is exactly how that looks in
    the output. It reads like an edge and is not one.

    So the comparison that means something is against a signal that trades
    the SAME bars, the same number of times, with the directions shuffled.
    That holds the path geometry, the bar selection, the time of day and
    the volatility fixed, and destroys only the thing being tested. If the
    rule cannot beat its own shuffle, whatever it found was the geometry.

    The signs are permuted rather than redrawn, so the long/short balance
    is preserved too -- a rule that is 80% long keeps a control that is 80%
    long, and a market that simply drifted up cannot flatter it.
    """
    import random

    live = [(i, s) for i, s in enumerate(sides) if s != 0]
    if not live:
        return {"trials": 0, "mean": None, "sd": None, "rates": []}

    rng = random.Random(seed)
    signs = [s for _, s in live]
    per_trade: list[float] = []
    rates: list[float] = []

    for _ in range(trials):
        shuffled = signs[:]
        rng.shuffle(shuffled)
        blank = [0] * len(sides)
        for (i, _), s in zip(live, shuffled):
            blank[i] = s
        r = run(slices, blank, take_ticks, stop_ticks, bars, tick=tick,
                cost_usd=cost_usd, tick_usd=tick_usd)
        if r.per_trade is not None:
            per_trade.append(r.per_trade)
            rates.append(r.rate)

    if not per_trade:
        return {"trials": 0, "mean": None, "sd": None, "rates": []}

    m = sum(per_trade) / len(per_trade)
    var = (sum((x - m) ** 2 for x in per_trade) / (len(per_trade) - 1)
           if len(per_trade) > 1 else 0.0)
    return {"trials": len(per_trade), "mean": m, "sd": var ** 0.5,
            "mean_rate": sum(rates) / len(rates), "rates": rates}


def beats_control(observed: float, ctrl: dict, sds: float = 2.0) -> bool:
    """Is the rule clear of its own shuffle by a margin?

    Two standard deviations of the control distribution, not zero. The
    control has spread, and a rule a whisker above its mean is inside the
    noise of the thing it is being compared to.
    """
    if ctrl.get("mean") is None:
        return False
    sd = ctrl.get("sd") or 0.0
    return observed > ctrl["mean"] + sds * sd


def run(slices: Sequence, sides: Sequence[int], take_ticks: float,
        stop_ticks: float, bars: int, tick: float = 0.25,
        cost_usd: float = ES_RT_COST_USD,
        tick_usd: float = ES_TICK_USD) -> Result:
    """Score one signal over a set of bars.

    `sides[i]` is the call on `slices[i]`: +1, -1, or 0 for no trade. Bars
    too close to the end to have a full horizon are skipped rather than
    scored short -- a truncated horizon cannot reach its target and would
    read as a run of losses that never happened.
    """
    target = stop = timeout = n = 0
    last = len(slices) - bars
    for i, side in enumerate(sides):
        if side == 0 or i >= last:
            continue
        entry = slices[i].open
        if entry <= 0:
            continue
        one_tick_bps = tick / entry * 10_000.0
        v = resolve(chain(slices, i, bars), side,
                    take_ticks * one_tick_bps, stop_ticks * one_tick_bps)
        n += 1
        if v == "target":
            target += 1
        elif v == "stop":
            stop += 1
        else:
            timeout += 1
    return Result(n=n, target=target, stop=stop, timeout=timeout,
                  take_ticks=take_ticks, stop_ticks=stop_ticks,
                  cost_usd=cost_usd, tick_usd=tick_usd)
