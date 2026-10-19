"""Does a diagonal imbalance predict anything? Turn it into a setup.

    from liqmap.footprint_signals import Replay
    r = Replay(rule=VolumeRule(2000))
    r.feed(records)                     # raw MBP-10
    out = r.score(target_ticks=16, stop_ticks=4, cost_ticks=1.0)

The footprint engine says WHAT traded. This says whether what it
found was worth anything, using the same three arms that already
retired the timeframe-agreement idea:

    SIGNAL    the side the imbalance points at
    MIRROR    the same entries, the other way round
    CONTROL   the same entry times, a coin flip

ONE BAR IS ONE OBSERVATION. A bar can carry fifteen flags; they share
an entry price, a timestamp and an outcome, so scoring them
separately would multiply the sample size by fifteen without adding a
single independent fact. Confidence intervals computed on that are
not wrong by a little. At most one setup per bar per kind survives.

A BAR THAT FLAGS BOTH WAYS IS SKIPPED. Buyers overwhelming one level
while sellers overwhelm another is not a direction, and picking the
bigger one is a free parameter -- the kind that makes a backtest look
decisive and a live account look nothing like it.

ENTRY IS AT THE BAR'S CLOSE, never at the flagged price. The flag is
only confirmed when the bar ends; the price it occurred at was
available earlier in the bar, and filling there would be reading the
past with the future's information.

THE THREE KINDS

    single    any confirmed imbalance in the bar
    stacked   three or more consecutive levels on one side
    extreme   a stack that reaches the bar's own high or low --
              the classic footprint setup, sellers exhausted at the
              low or buyers at the high

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER. It counts outcomes.
"""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass, field

from . import dbn, ofi_backtest as bt
from .footprint import (BUY_IMBALANCE, SELL_IMBALANCE, CloseRule,
                        DEFAULT_IMBALANCE, FootprintCandle,
                        ImbalanceConfig, Tick, TimeRule)
from .footprint_live import FootprintStream

KINDS = ("single", "stacked", "extreme")


@dataclass(frozen=True, slots=True)
class Setup:
    """One tradeable observation."""

    ts: float                  # the bar's close
    entry_ticks: int           # the bar's closing price
    side: int                  # +1 long, -1 short
    kind: str
    flag_ticks: int            # where the imbalance was
    run: int                   # levels in the run
    bar: int                   # index into history
    volume: int                # the aggressive side's contracts


class Seconds(bt.Seconds):
    """`ofi_backtest.Seconds`, filled from trades as they stream.

    The forward path has to come from the SAME prints the footprint
    was built from. Taken from a separate file it could disagree
    about which second a print landed in, and every outcome after
    that measures the join.
    """

    def feed_trade(self, ts: float, px_ticks: int, size: int) -> None:
        sec = int(math.floor(ts))
        n = len(self.ts)
        if n and self.ts[-1] == sec:
            if px_ticks > self.hi[-1]:
                self.hi[-1] = px_ticks
            if px_ticks < self.lo[-1]:
                self.lo[-1] = px_ticks
            self.cl[-1] = px_ticks
            return
        if n and sec < self.ts[-1]:
            return                      # out of order; the guard in
            # `ofi_backtest.run` would refuse the whole file anyway
        self.add(sec, px_ticks, px_ticks, px_ticks, 0)


def setups_from(bar: FootprintCandle, flags: list, index: int,
                run_min: int = 3, edge_within: int = 1) -> list:
    """Every kind this bar supports, at most one of each.

    Returns [] when the bar flagged both ways -- that is not a
    direction.
    """
    if not flags or bar.empty:
        return []

    buys = sorted(f.price_ticks for f in flags
                  if f.side == BUY_IMBALANCE)
    sells = sorted(f.price_ticks for f in flags
                   if f.side == SELL_IMBALANCE)
    if buys and sells:
        return []

    side = 1 if buys else -1
    prices = buys or sells
    by_price = {f.price_ticks: f for f in flags}

    # the longest consecutive run on that side
    best_lo = best_hi = prices[0]
    lo = prev = prices[0]
    for p in prices[1:]:
        if p == prev + 1:
            prev = p
        else:
            if prev - lo > best_hi - best_lo:
                best_lo, best_hi = lo, prev
            lo = prev = p
    if prev - lo > best_hi - best_lo:
        best_lo, best_hi = lo, prev
    run = best_hi - best_lo + 1

    def vol_of(p):
        f = by_price.get(p)
        return f.volume if f is not None else 0

    out = [Setup(ts=bar.ts_close, entry_ticks=bar.close_ticks,
                 side=side, kind="single",
                 flag_ticks=max(prices, key=vol_of), run=1, bar=index,
                 volume=max(vol_of(p) for p in prices))]

    if run >= run_min:
        out.append(Setup(ts=bar.ts_close, entry_ticks=bar.close_ticks,
                         side=side, kind="stacked",
                         flag_ticks=best_lo if side > 0 else best_hi,
                         run=run, bar=index,
                         volume=sum(vol_of(p)
                                    for p in range(best_lo, best_hi + 1))))
        # A buy stack reaching the bar's LOW is sellers exhausted
        # there; a sell stack reaching the HIGH is buyers exhausted.
        # Anywhere else in the bar it is not the same claim.
        #
        # "REACHING" CANNOT MEAN "AT". The edge policy skips the bar's
        # lowest level for a buy test, because its diagonal lies
        # outside the bar -- so a buy flag AT the low is impossible by
        # construction, and a condition demanding one never fires.
        # The lowest flaggable level is low + 1, which is what
        # `edge_within` defaults to.
        at_edge = (best_lo <= bar.low_ticks + edge_within if side > 0
                   else best_hi >= bar.high_ticks - edge_within)
        if at_edge:
            out.append(Setup(
                ts=bar.ts_close, entry_ticks=bar.close_ticks,
                side=side, kind="extreme",
                flag_ticks=best_lo if side > 0 else best_hi,
                run=run, bar=index,
                volume=sum(vol_of(p)
                           for p in range(best_lo, best_hi + 1))))
    return out


@dataclass
class Replay:
    """Raw records in; setups and the forward path out."""

    rule: CloseRule | None = None
    imbalance: ImbalanceConfig = DEFAULT_IMBALANCE
    tick_size: float = 0.25
    run_min: int = 3
    edge_within: int = 1        # how close to the extreme still counts
    flip: bool = False          # for a feed with the other convention

    setups: list = field(default_factory=list)
    secs: Seconds = field(default_factory=Seconds)
    records: int = 0
    trades: int = 0
    bars: int = 0

    def __post_init__(self) -> None:
        self.stream = FootprintStream(
            tick_size=self.tick_size,
            rule=self.rule or TimeRule(300.0),
            imbalance=self.imbalance)

    def feed(self, records, progress=None) -> None:
        """One pass. The footprint and the forward path are built from
        the same prints, in the same order."""
        scale = self.stream.scale
        for rec in records:
            self.records += 1
            if progress is not None:
                progress()
            t = dbn.trade_from(rec, flip=self.flip)
            if t is None:
                continue
            size = int(round(float(t.sz)))
            if size <= 0:
                continue
            self.trades += 1
            px = scale.to_ticks(t.px)
            self.secs.feed_trade(t.ts, px, size)

            before = len(self.stream.history)
            self.stream.on_tick(Tick(ts=float(t.ts), price=float(t.px),
                                     size=size, aggressor=t.aggressor))
            for i in range(before, len(self.stream.history)):
                self.bars += 1
                bar = self.stream.history[i]
                flags = self.stream.confirmed_flags(i)
                self.setups.extend(
                    setups_from(bar, flags, i, run_min=self.run_min,
                                edge_within=self.edge_within))

    # ----------------------------------------------------------- score

    def score(self, kind: str = "stacked", target_ticks: int = 16,
              stop_ticks: int = 4, cost_ticks: float = 1.0,
              max_hold_s: int = 900, train_fraction: float = 0.5
              ) -> dict:
        """SIGNAL against MIRROR and CONTROL, split train / held out."""
        picked = [s for s in self.setups if s.kind == kind]
        be = bt.break_even(target_ticks, stop_ticks, cost_ticks)
        head = {"kind": kind, "setups": len(picked),
                "bars": self.bars, "trades": self.trades,
                "records": self.records,
                "seconds": len(self.secs),
                "hours": round(self.secs.span_hours(), 1),
                "break_even": round(be, 4),
                "settings": {"target_ticks": target_ticks,
                             "stop_ticks": stop_ticks,
                             "cost_ticks": cost_ticks,
                             "max_hold_s": max_hold_s}}
        if not picked:
            return {**head, "ok": False,
                    "why": f"no {kind} setups in this data"}
        back = self.secs.out_of_order
        if back:
            return {**head, "ok": False,
                    "why": f"the forward path goes backwards at "
                           f"{back:,} places"}

        # the index of each setup's entry second
        idx = {}
        for i, ts in enumerate(self.secs.ts):
            idx.setdefault(int(ts), i)

        arms = {"signal": [], "mirror": [], "control": []}
        used = []
        for s in picked:
            i = idx.get(int(math.floor(s.ts)))
            if i is None:
                continue
            used.append(s)
            for name, side in (("signal", s.side),
                               ("mirror", -s.side),
                               ("control", bt._coin(int(s.ts)))):
                arms[name].append(bt.resolve(
                    self.secs, i, side, target_ticks, stop_ticks,
                    max_hold_s))
        if not used:
            return {**head, "ok": False,
                    "why": "no setup could be located in the tape"}

        k = int(len(used) * train_fraction)
        cut = used[k].ts if 0 < k < len(used) else used[0].ts
        out = {**head, "ok": True, "scored_setups": len(used),
               "split_at": cut, "arms": {}}
        for name, trades in arms.items():
            pairs = list(zip(used, trades))
            out["arms"][name] = {
                "all": bt.score(trades, name).to_dict(cost_ticks),
                "train": bt.score([t for s, t in pairs if s.ts < cut],
                                  name).to_dict(cost_ticks),
                "held_out": bt.score([t for s, t in pairs
                                      if s.ts >= cut],
                                     name).to_dict(cost_ticks)}
        h = out["arms"]
        out["verdict"] = bt.verdict(h["signal"]["held_out"],
                                    h["mirror"]["held_out"],
                                    h["control"]["held_out"],
                                    out["break_even"], cost_ticks)
        return out
