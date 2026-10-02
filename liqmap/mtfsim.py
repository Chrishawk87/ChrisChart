"""Walk the 4H/15m/1m signals through the tape and count what happened.

THE RULE BEING TESTED, AS STATED

Take the signal. Price hits the target, that is a win and the trade is
over. Price hits the stop first, that is a loss. Either way, wait for the
NEXT 15-minute candle before taking anything again -- one position at a
time, no stacking, no re-entry inside the bar you just left.

FOUR THINGS DECIDE WHETHER A NUMBER LIKE THIS IS WORTH ANYTHING

1. WHICH BARRIER CAME FIRST. A bar whose high reaches the target and whose
   low reaches the stop does not record the order of the ticks inside
   itself. Resolving those in the tester's favour is the single easiest
   way to manufacture a win rate nobody can trade, so a bar holding both
   is scored as the STOP. Always. The finer the bars, the less often it
   matters -- which is why this refuses to run on bars as coarse as the
   signal grid, where almost every resolution would be that coin flip.

2. THE ENTRY IS THE NEXT PRICE, NOT THE SIGNAL PRICE. The signal is known
   at the close of a 1-minute bar; you cannot have traded at that close.
   Entry is the open of the following fine bar.

3. THE CONTROL. A 10-tick target against a 20-tick stop wins about 67% of
   the time on a driftless walk -- that is geometry, not skill. So the same
   entries, on the same bars, with the same barrier distances and the
   DIRECTIONS SHUFFLED, are run alongside. If the real thing does not
   clear that distribution, the win rate is the shape of the barriers and
   nothing else.

4. COST. A taker's round trip on one ES contract is $5 commission plus the
   $12.50 spread given up. Gross and net are both reported, because a
   gross number on a 10-tick target is most of a fiction.

WHAT THIS DOES NOT DO

It does not optimise anything. No parameter here is fitted, searched, or
chosen by looking at the result. The barriers are the ones in the spec.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

from .mtf import CONFIRMED, M15_SECONDS, Signal, WITHDRAWN

ES_TICK = 0.25
ES_TICK_USD = 12.50

# A taker's true round trip: commission plus the spread given up.
COST_COMMISSION = 5.00
COST_SPREAD = 12.50
COST_RT = COST_COMMISSION + COST_SPREAD

TARGET = "target"
STOP = "stop"
TIME = "time"            # the 15m candle closed with the trade still on
UNRESOLVED = "unresolved"

# Profit, in ticks, at which the stop is pulled up to the entry. Only used
# in the breakeven variant; None there means it never moves.
BREAKEVEN_AT = 4.0


@dataclass
class Trade:
    """One signal taken, and how it ended."""

    ts: float                 # the signal's timestamp
    side: str
    entry_ts: float
    entry: float
    target: float
    stop: float

    outcome: str = UNRESOLVED
    exit_ts: float = 0.0
    exit: float = 0.0
    ticks: float = 0.0        # signed, in ticks, before cost
    minutes: float = 0.0
    later: str = ""           # what its 15m bar did: confirmed | withdrawn

    @property
    def won(self) -> bool:
        return self.outcome == TARGET

    def to_dict(self) -> dict:
        return {"ts": self.ts, "side": self.side, "entry": round(self.entry, 4),
                "target": round(self.target, 4), "stop": round(self.stop, 4),
                "outcome": self.outcome, "exit_ts": self.exit_ts,
                "exit": round(self.exit, 4), "ticks": round(self.ticks, 2),
                "minutes": round(self.minutes, 1), "later": self.later}


def _next_15m(ts: float) -> float:
    """The start of the 15-minute candle after the one holding `ts`."""
    return ts - (ts % M15_SECONDS) + M15_SECONDS


class Tape:
    """The fine bars as arrays, built once.

    The control runs the whole walk a couple of hundred times. Rebuilding
    and re-scanning a two-year tape in Python for each of those is the
    difference between a minute and an afternoon, so the columns are
    prepared once and every run slices the same arrays.
    """

    __slots__ = ("ts", "hi", "lo", "op", "cl", "n")

    def __init__(self, fine: Sequence) -> None:
        import numpy as np

        rows = sorted(fine, key=lambda b: float(b.ts))
        self.ts = np.fromiter((float(b.ts) for b in rows), float, len(rows))
        self.hi = np.fromiter((float(b.high) for b in rows), float, len(rows))
        self.lo = np.fromiter((float(b.low) for b in rows), float, len(rows))
        self.op = np.fromiter((float(b.open) for b in rows), float, len(rows))
        self.cl = np.fromiter((float(b.close) for b in rows), float,
                              len(rows))
        self.n = len(rows)


def _first_touch(tape: Tape, start: int, end: int, long: bool,
                 target: float, stop: float,
                 breakeven_at: float | None = None,
                 entry: float = 0.0, tick: float = ES_TICK,
                 trail_ticks: float | None = None,
                 trail_arm: float = 3.0) -> tuple[str, int, float]:
    """Which barrier is touched first in [start, end). Returns (outcome, i).

    A bar touching BOTH is the STOP. The bar does not record which tick
    came first, and guessing in the tester's favour is how a backtest
    produces a number that evaporates live. `<=` rather than `<` is where
    that rule lives -- flipping it quietly hands every tie to the winner.

    With `breakeven_at` the stop is pulled to the entry once the trade has
    been that many ticks onside. The same tie rule applies to the moved
    stop, and the move itself is resolved bar by bar from that point --
    scanning the whole window at once would let a later profit move a stop
    that was already hit.

    Returns (outcome, index, exit price).
    """
    import numpy as np

    end = min(end, tape.n)
    if end <= start:
        return UNRESOLVED, start, 0.0
    hi = tape.hi[start:end]
    lo = tape.lo[start:end]

    def _scan(a: int, b: int, st: float) -> tuple[int, int]:
        h = hi[a - start:b - start]
        l = lo[a - start:b - start]
        ht = (h >= target) if long else (l <= target)
        hs = (l <= st) if long else (h >= st)
        it = int(np.argmax(ht)) if ht.any() else -1
        isx = int(np.argmax(hs)) if hs.any() else -1
        return it, isx

    # ---- a trailing stop -------------------------------------------
    #
    # Once the trade is `trail_arm` ticks onside the stop follows the
    # best price by `trail_ticks`, so a move that runs and then turns is
    # not given back all the way to the original stop.
    #
    # THE LAG IS DELIBERATE. The extreme that raises the stop and the
    # low that takes it can be inside the SAME bar, and the bar does not
    # record which came first. Using the running extreme up to the
    # PREVIOUS bar assumes the unfavourable order, which is the same rule
    # the tie gets everywhere else in this file. Including the current
    # bar's own high would let a spike raise the stop above a low that
    # had already happened.
    if trail_ticks is not None:
        run = np.maximum.accumulate(hi) if long else \
            np.minimum.accumulate(lo)
        prev = np.empty_like(run)
        prev[0] = entry
        prev[1:] = run[:-1]
        armed = ((prev >= entry + trail_arm * tick) if long
                 else (prev <= entry - trail_arm * tick))
        trailing = (prev - trail_ticks * tick) if long \
            else (prev + trail_ticks * tick)
        eff = np.where(armed,
                       np.maximum(trailing, stop) if long
                       else np.minimum(trailing, stop),
                       stop)
        ht = (hi >= target) if long else (lo <= target)
        hs = (lo <= eff) if long else (hi >= eff)
        it = int(np.argmax(ht)) if ht.any() else -1
        isx = int(np.argmax(hs)) if hs.any() else -1
        if isx >= 0 and (it < 0 or isx <= it):
            return STOP, start + isx, float(eff[isx])
        if it >= 0:
            return TARGET, start + it, target
        return UNRESOLVED, end - 1, 0.0

    if breakeven_at is None:
        it, isx = _scan(start, end, stop)
        if isx >= 0 and (it < 0 or isx <= it):
            return STOP, start + isx, stop
        if it >= 0:
            return TARGET, start + it, target
        return UNRESOLVED, end - 1, 0.0

    # The stop moves. Find where the trade first goes `breakeven_at`
    # ticks onside; before that the original stop applies, after it the
    # entry does.
    trip = entry + breakeven_at * tick if long else entry - breakeven_at * tick
    tr = (hi >= trip) if long else (lo <= trip)
    mv = (start + int(np.argmax(tr))) if tr.any() else -1

    first_end = end if mv < 0 else mv + 1
    it, isx = _scan(start, first_end, stop)
    if isx >= 0 and (it < 0 or isx <= it):
        return STOP, start + isx, stop
    if it >= 0:
        return TARGET, start + it, target
    if mv < 0 or mv + 1 >= end:
        return UNRESOLVED, end - 1, 0.0

    it2, is2 = _scan(mv + 1, end, entry)
    if is2 >= 0 and (it2 < 0 or is2 <= it2):
        return STOP, mv + 1 + is2, entry       # out at breakeven
    if it2 >= 0:
        return TARGET, mv + 1 + it2, target
    return UNRESOLVED, end - 1, 0.0


def walk(signals: Sequence[Signal], fine: Sequence, tick: float = ES_TICK,
         hold_minutes: float = 240.0,
         sides: Sequence[str] | None = None,
         signal_seconds: float = 60.0,
         expire: bool = True,
         breakeven_at: float | None = None,
         trail_ticks: float | None = None,
         trail_arm: float = 3.0) -> list[Trade]:
    """Take the signals in order, one at a time, and resolve each.

    `sides` overrides the signal directions, which is how the shuffled
    control reuses this function unchanged -- same entries, same bars,
    same barrier distances, only the direction permuted.

    `hold_minutes` caps a trade that never touches either barrier. It is
    a reporting bucket, not a third exit: those trades are counted apart
    from the wins and losses rather than folded in as either.
    """
    import numpy as np

    tape = fine if isinstance(fine, Tape) else Tape(fine)
    if not tape.n or not signals:
        return []

    out: list[Trade] = []
    free_from = 0.0          # no new trade before this timestamp

    for n, sig in enumerate(signals):
        if sig.ts < free_from:
            continue
        side = sides[n] if sides is not None else sig.side

        # ENTRY IS AFTER THE SIGNAL'S 1-MINUTE BAR HAS CLOSED.
        #
        # A signal is stamped with its 1m bar's START, because that is how
        # bars are stamped everywhere in this project. It is not KNOWN
        # until that bar closes, sixty seconds later. Entering at the first
        # fine bar after the stamp means buying at the start of the very
        # minute whose rising close produced the long -- fifty-five seconds
        # before anyone could have seen it.
        #
        # That bug scored 98.4% on a pure random walk against a 79.8%
        # shuffled control. It is the reason the control exists.
        ready = sig.ts + signal_seconds
        i = int(np.searchsorted(tape.ts, ready, side="left"))
        if i >= tape.n:
            break
        entry = float(tape.op[i])

        # The barrier DISTANCES come from the signal; the directions come
        # from `side`. The control therefore holds the geometry fixed and
        # varies only the call.
        t_ticks = sig.target_ticks or 10.0
        s_ticks = sig.stop_ticks or 10.0
        if side == "long":
            target, stop = entry + t_ticks * tick, entry - s_ticks * tick
        else:
            target, stop = entry - t_ticks * tick, entry + s_ticks * tick

        # NO TRADE OUTLIVES ITS OWN 15-MINUTE CANDLE. When that candle
        # closes the position is flat at whatever price is there -- not a
        # win, not a loss, a third thing, and counted as a third thing.
        cap = float(tape.ts[i]) + hold_minutes * 60.0
        if expire and sig.expires:
            cap = min(cap, float(sig.expires))
        end = int(np.searchsorted(tape.ts, cap, side="right"))

        outcome, j, px = _first_touch(
            tape, i, max(end, i + 1), side == "long", target, stop,
            breakeven_at=breakeven_at, entry=entry, tick=tick,
            trail_ticks=trail_ticks, trail_arm=trail_arm)
        if outcome == UNRESOLVED and expire and sig.expires and end <= tape.n:
            outcome = TIME
            j = min(max(end - 1, i), tape.n - 1)
            px = float(tape.cl[j])
        elif outcome == UNRESOLVED:
            px = float(tape.cl[j])
        ts = float(tape.ts[j])

        signed = (px - entry) if side == "long" else (entry - px)
        out.append(Trade(
            ts=sig.ts, side=side, entry_ts=float(tape.ts[i]),
            entry=entry, target=target, stop=stop,
            outcome=outcome, exit_ts=ts, exit=px,
            ticks=signed / tick if tick > 0 else 0.0,
            minutes=(ts - float(tape.ts[i])) / 60.0,
            later=sig.status if sig.status in (CONFIRMED, WITHDRAWN) else ""))

        # One at a time, and nothing new until the NEXT 15-minute candle
        # after the one the exit landed in.
        free_from = _next_15m(ts)

    return out


# ---------------------------------------------------------------- scoring

def summarise(trades: Sequence[Trade], tick_usd: float = ES_TICK_USD,
              cost_usd: float = COST_RT) -> dict:
    """Wins, losses, and what they are worth. No verdict."""
    from .score import wilson

    done = [t for t in trades if t.outcome in (TARGET, STOP)]
    wins = [t for t in done if t.won]
    losses = [t for t in done if not t.won]
    timed = [t for t in trades if t.outcome == TIME]
    unresolved = [t for t in trades if t.outcome == UNRESOLVED]

    # EVERY trade that was actually in the market costs money and makes or
    # loses some, including the ones the clock closed. The win rate is the
    # target-versus-stop race, because that is the question; the money is
    # all of them, because that is the account.
    booked = done + timed
    gross = sum(t.ticks for t in booked) * tick_usd
    net = gross - len(booked) * cost_usd
    # The race is target against stop. A scratch did not run that race,
    # so it is out of the denominator rather than scored as a defeat.
    raced = [t for t in done if not (not t.won and abs(t.ticks) < 0.5)]
    rate = len(wins) / len(raced) if raced else 0.0
    lo, hi = wilson(len(wins), len(raced)) if raced else (0.0, 0.0)

    # The rate this barrier pair needs, averaged over the trades actually
    # taken. Stops vary trade to trade, so a single pair would not describe
    # the set.
    avg_t = (sum(abs(t.target - t.entry) for t in done) / len(done) / ES_TICK
             if done else 0.0)
    avg_s = (sum(abs(t.stop - t.entry) for t in done) / len(done) / ES_TICK
             if done else 0.0)
    total = avg_t + avg_s
    cost_ticks = cost_usd / tick_usd if tick_usd > 0 else 0.0
    driftless = avg_s / total if total > 0 else 0.0
    breakeven = min(1.0, (avg_s + cost_ticks) / total) if total > 0 else 0.0

    # A stop hit AT the entry is not a loss, it is a scratch. Counting it
    # as one makes the breakeven variant look far worse than it is: its
    # win rate collapses while its money barely moves.
    scratch = [t for t in done if not t.won and abs(t.ticks) < 0.5]
    real_losses = [t for t in losses if t not in scratch]
    green = [t for t in booked if t.ticks > 0]
    return {
        "trades": len(trades), "resolved": len(done),
        "wins": len(wins), "losses": len(real_losses),
        "scratched": len(scratch),
        "timed_out": len(timed), "booked": len(booked),
        "profitable": len(green),
        "profitable_rate": round(len(green) / len(booked), 4)
                           if booked else 0.0,
        "unresolved": len(unresolved),
        "win_rate": round(rate, 4),
        "ci_low": round(lo, 4), "ci_high": round(hi, 4),
        "avg_target_ticks": round(avg_t, 2),
        "avg_stop_ticks": round(avg_s, 2),
        "driftless": round(driftless, 4),
        "breakeven": round(breakeven, 4),
        "gross_usd": round(gross, 2), "net_usd": round(net, 2),
        "per_trade_usd": round(net / len(booked), 2) if booked else 0.0,
        "avg_minutes": round(sum(t.minutes for t in booked) / len(booked), 1)
                       if booked else 0.0,
    }


def control(signals: Sequence[Signal], fine: Sequence, runs: int = 200,
            tick: float = ES_TICK, hold_minutes: float = 240.0,
            seed: int = 7, tick_usd: float = ES_TICK_USD,
            cost_usd: float = COST_RT, expire: bool = True,
            breakeven_at: float | None = None,
            trail_ticks: float | None = None,
            trail_arm: float = 3.0) -> dict:
    """The same entries with the directions shuffled, many times over.

    This is the baseline that matters. The theoretical driftless rate
    assumes a random walk; real bars trend, gap and mean-revert, and the
    signals are not spread evenly through the day. Permuting the direction
    holds ALL of that fixed -- the same bars, the same entry times, the
    same barrier widths, the same time-of-day mix -- and varies only the
    one thing the engine claims to know.
    """
    rng = random.Random(seed)
    tape = fine if isinstance(fine, Tape) else Tape(fine)
    rates: list[float] = []
    nets: list[float] = []
    for _ in range(runs):
        flip = [rng.choice(("long", "short")) for _ in signals]
        s = summarise(walk(signals, tape, tick, hold_minutes, sides=flip,
                           expire=expire, breakeven_at=breakeven_at,
                           trail_ticks=trail_ticks, trail_arm=trail_arm),
                      tick_usd, cost_usd)
        if s["booked"]:
            rates.append(s["win_rate"])
            nets.append(s["per_trade_usd"])
    if not rates:
        return {"runs": 0}

    def _ms(xs: list[float]) -> tuple[float, float]:
        m = sum(xs) / len(xs)
        var = sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)
        return m, var ** 0.5

    rm, rs = _ms(rates)
    nm, ns = _ms(nets)
    return {"runs": len(rates),
            "win_rate_mean": round(rm, 4), "win_rate_sd": round(rs, 4),
            "per_trade_mean": round(nm, 2), "per_trade_sd": round(ns, 2),
            "win_rate_p95": round(sorted(rates)[int(len(rates) * 0.95)], 4)}


def beats(observed: dict, ctrl: dict, sds: float = 2.0) -> bool:
    """Does the real win rate clear the shuffled one by `sds` deviations?

    Deliberately strict, and deliberately about the CONTROL rather than
    the theoretical breakeven. Beating a formula is easy; beating the same
    bars with the call taken out is the question.
    """
    if not ctrl.get("runs") or not observed.get("resolved"):
        return False
    sd = ctrl.get("win_rate_sd") or 0.0
    if sd <= 0:
        return observed["win_rate"] > ctrl["win_rate_mean"]
    return observed["win_rate"] >= ctrl["win_rate_mean"] + sds * sd


def split(trades: Sequence[Trade], key, tick_usd: float = ES_TICK_USD,
          cost_usd: float = COST_RT) -> dict[str, dict]:
    """Summaries grouped by `key(trade)`, for the breakdown tables.

    Every slice reported is another chance for one of them to look good by
    accident: eight slices at a nominal 5% give a 34% chance that at least
    one reads significant on noise alone. The caller is expected to say how
    many it looked at.
    """
    groups: dict[str, list[Trade]] = {}
    for t in trades:
        groups.setdefault(str(key(t)), []).append(t)
    return {k: summarise(v, tick_usd, cost_usd)
            for k, v in sorted(groups.items())}
