#!/usr/bin/env python3
"""Does 15m/5m/1m agreement pay at a four-tick target? Two years of ES.

    python scripts/tf_backtest.py data/ES_c_0_ohlcv-1s_2024-09-30.dbn.zst

PHASES 2 TO 6. RESEARCH ONLY. It reads one file and prints a table.

WHAT IS KNOWN WHEN

Every signal is decided from COMPLETED bars. At the close of a 1-minute bar
the engine may look at that bar and every bar before it on every timeframe,
and at nothing else. The fill is the open of the first 1-second bar AFTER
that close. Filling at the signal bar's own open is a fifty-nine second
look-ahead and it scored 98.4% on a pure random walk in this project, which
is why `mtfsim` exists and why the control below is not optional.

TWO DEFINITIONS, KEPT APART

    A  STRUCTURE   higher highs and higher lows, from swings that were
                   CONFIRMED by the decision bar. `structure.swings` dates
                   a pivot at its bar but confirms it `strength` bars
                   later, and only confirmed ones are read.
    B  MOMENTUM    the last completed close against the one before it.
                   Equal closes are neutral.

They are never mixed, and neither is adjusted after seeing a result.

WHAT A BAR CANNOT SETTLE

A one-second bar that spans the target AND the stop touched both and
records no order. That trade is UNRESOLVABLE on this data. It is excluded
and counted, never guessed -- 0.4% of bars over two years, 3.5% in April
2025, measured by `data_audit`. Crediting the target there is the easiest
way in existence to manufacture a win rate.

NO TRADE CROSSES A CONTRACT ROLL

The nine instrument ids in this file are eight rolls. Across one the price
steps by carry, not by anything the market did, so a position held through
one would book a move nobody could have made.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap import dbn                                      # noqa: E402
from liqmap import structure as st                          # noqa: E402

TICK = 0.25
DOLLARS_PER_TICK = 12.50

FRAMES = (("1m", 60.0), ("5m", 300.0), ("15m", 900.0))
SPAN = dict(FRAMES)

TARGET_TICKS = 4                     # the brief: 4 ticks = 1.00 index point
STOP_TICKS = (4, 6, 8)               # reported side by side, none chosen
COSTS = (0.4, 1.0, 1.5)              # round trip in ticks
TIME_CAPS_S = (300.0, 900.0, 1800.0)  # 5, 15 and 30 minutes

SWING_STRENGTH = 2                   # bars either side; confirmation lag
MIN_BARS = 40                        # history a timeframe needs to speak
HEARTBEAT = 2_000_000


class Progress:
    """Heartbeat on stderr; a silent pass over 40M bars reads as a hang."""

    def __init__(self, label: str, every: int = HEARTBEAT, out=None) -> None:
        self.label, self.every, self.n = label, max(1, every), 0
        self.extra = ""
        self.out = out if out is not None else sys.stderr
        self.start = time.time()

    def tick(self) -> None:
        self.n += 1
        if self.n % self.every:
            return
        el = max(1e-9, time.time() - self.start)
        print(f"  [{self.label}] {self.n:,}  {el:,.0f}s  "
              f"{self.n / el:,.0f}/s  {self.extra}", file=self.out,
              flush=True)

    def done(self) -> None:
        el = max(1e-9, time.time() - self.start)
        print(f"  [{self.label}] done: {self.n:,} in {el:,.0f}s",
              file=self.out, flush=True)


# -------------------------------------------------- PHASE 2: the bars

@dataclass
class Seconds:
    """One contract's 1-second bars, as parallel lists.

    Objects would cost gigabytes over two years; these are indexed by the
    fill simulation and never copied.
    """

    ts: list = field(default_factory=list)
    op: list = field(default_factory=list)
    hi: list = field(default_factory=list)
    lo: list = field(default_factory=list)
    cl: list = field(default_factory=list)

    def add(self, ts, o, h, l, c) -> None:
        self.ts.append(ts)
        self.op.append(o)
        self.hi.append(h)
        self.lo.append(l)
        self.cl.append(c)

    def __len__(self) -> int:
        return len(self.ts)


@dataclass
class Candle:
    """A completed bar on one of the three timeframes."""

    ts: float                        # the slot it opened in
    open: float
    high: float
    low: float
    close: float


def segments(records, progress: Progress | None = None):
    """Yield (instrument_id, Seconds) -- one per contract, in order.

    Streaming per contract rather than loading the file keeps peak memory
    to one quarter's bars, and it is also the unit a trade may not cross.
    """
    cur_id = None
    cur = Seconds()
    for rec in records:
        if progress is not None:
            progress.extra = f"{len(cur):,} bars in segment"
            progress.tick()
        iid = dbn._get(rec, "instrument_id")
        ts = dbn.seconds(dbn._get(rec, "ts_event"))
        o = dbn.price(dbn._get(rec, "open"))
        h = dbn.price(dbn._get(rec, "high"))
        lo = dbn.price(dbn._get(rec, "low"))
        c = dbn.price(dbn._get(rec, "close"))
        # `dbn.seconds` returns 0.0 when a stamp cannot be parsed, and no
        # real ES bar is dated at the epoch, so zero means unreadable
        # rather than 1 January 1970. Tests must therefore use realistic
        # stamps; a fixture starting at zero silently loses its first bar.
        if ts <= 0 or min(o, h, lo, c) <= 0:
            continue
        if h < lo or not (lo <= o <= h) or not (lo <= c <= h):
            continue                 # malformed; `bars.Bar.valid`'s rule
        if cur_id is None:
            cur_id = iid
        elif iid != cur_id:
            if len(cur):
                yield cur_id, cur
            cur_id, cur = iid, Seconds()
        cur.add(ts, o, h, lo, c)
    if cur_id is not None and len(cur):
        yield cur_id, cur
    if progress is not None:
        progress.done()


def build(sec: Seconds, span: float) -> list[Candle]:
    """Aggregate 1-second bars onto a clock-aligned grid.

    Open of the first, extremes of the span, close of the last. Exact --
    unlike going the other way, which cannot be done at all.
    """
    out: list[Candle] = []
    cur: Candle | None = None
    slot = -1.0
    for i in range(len(sec)):
        s = math.floor(sec.ts[i] / span) * span
        if cur is None or s != slot:
            if cur is not None:
                out.append(cur)
            cur = Candle(ts=s, open=sec.op[i], high=sec.hi[i],
                         low=sec.lo[i], close=sec.cl[i])
            slot = s
        else:
            cur.high = max(cur.high, sec.hi[i])
            cur.low = min(cur.low, sec.lo[i])
            cur.close = sec.cl[i]
    if cur is not None:
        out.append(cur)
    return out


# ------------------------------------------ PHASE 3: the two definitions

def momentum(cands: list[Candle], i: int) -> str:
    """Definition B, over bars 0..i INCLUSIVE, all of them completed."""
    if i < 1:
        return ""
    a, b = cands[i - 1].close, cands[i].close
    if b > a:
        return "up"
    if b < a:
        return "down"
    return ""                        # equal closes are neutral


def structure_way(cands: list[Candle], i: int,
                  strength: int = SWING_STRENGTH) -> str:
    """Definition A, from swings CONFIRMED by bar i and no later.

    `structure.swings` dates a pivot at its own bar and confirms it
    `strength` bars afterwards. Reading one at its bar is the most common
    way a structure backtest lies to itself, so only confirmed pivots
    count and the newest usable swing is always several bars old.
    """
    if i + 1 < MIN_BARS:
        return ""
    window = [st.Candle(ts=c.ts, open=c.open, high=c.high, low=c.low,
                        close=c.close, volume=0.0, trades=0)
              for c in cands[:i + 1]]
    sw = [s for s in st.swings(window, strength=strength)
          if s.confirmed_at <= i]
    highs = [s for s in sw if s.kind == "high"]
    lows = [s for s in sw if s.kind == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return ""
    hh = highs[-1].px > highs[-2].px
    hl = lows[-1].px > lows[-2].px
    lh = highs[-1].px < highs[-2].px
    ll = lows[-1].px < lows[-2].px
    if hh and hl:
        return "up"
    if lh and ll:
        return "down"
    return ""                        # no clear structure: do not enter


WAY = {"momentum": momentum, "structure": structure_way}


def newest_completed(grid: list[Candle], span: float, now: float,
                     start: int = 0) -> int | None:
    """Index of the newest bar that had CLOSED by `now`, or None.

    The bar in progress has not closed, so its high, low and close are
    not yet facts. Reading it is look-ahead of up to a full bar -- on the
    15-minute frame that is fifteen minutes of the future, which would
    make the whole study meaningless while every individual number in it
    still looked plausible.
    """
    j = start
    n = len(grid)
    if n == 0:
        return None
    while j + 1 < n and grid[j + 1].ts + span <= now:
        j += 1
    if grid[j].ts + span > now:
        return None
    return j


# ------------------------------------------------- PHASE 4: the trade

@dataclass
class Trade:
    signal_ts: float
    entry_ts: float
    entry: float
    side: str                        # buy | sell
    stop_ticks: int
    cap_s: float
    exit_ts: float = 0.0
    exit: float = 0.0
    outcome: str = ""                # target | stop | timeout | ambiguous
    seconds: float = 0.0
    ways: tuple = ()

    @property
    def ticks(self) -> float:
        """Gross ticks, before costs. A timeout books what it was worth."""
        d = (self.exit - self.entry) / TICK
        return d if self.side == "buy" else -d


def resolve(sec: Seconds, i0: int, entry: float, side: str,
            stop_ticks: int, cap_s: float):
    """Walk 1-second bars from index i0. Returns (outcome, exit, ts, secs).

    A BAR THAT SPANS BOTH LEVELS IS AMBIGUOUS. It touched the target and
    the stop and records no order between them, so the trade is excluded
    rather than awarded to whichever reading flatters the result.
    """
    long = side == "buy"
    tgt = entry + TARGET_TICKS * TICK if long else entry - TARGET_TICKS * TICK
    stp = entry - stop_ticks * TICK if long else entry + stop_ticks * TICK
    t0 = sec.ts[i0]
    end = t0 + cap_s
    i = i0
    n = len(sec)
    while i < n and sec.ts[i] <= end:
        hi, lo = sec.hi[i], sec.lo[i]
        hit_t = (hi >= tgt) if long else (lo <= tgt)
        hit_s = (lo <= stp) if long else (hi >= stp)
        if hit_t and hit_s:
            return "ambiguous", 0.0, sec.ts[i], sec.ts[i] - t0
        if hit_t:
            return "target", tgt, sec.ts[i], sec.ts[i] - t0
        if hit_s:
            return "stop", stp, sec.ts[i], sec.ts[i] - t0
        i += 1
    if i >= n:
        # The DATA ran out, not the clock. There is no price at which this
        # trade was closed, so it is not a scratch at some invented value
        # -- it never finished, and inventing an exit here would book a
        # trade nobody could have taken at the end of every contract.
        return "unfinished", 0.0, t0, 0.0
    return "timeout", sec.cl[i - 1], sec.ts[i - 1], sec.ts[i - 1] - t0


def first_at_or_after(sec: Seconds, ts: float, lo: int = 0) -> int:
    """Index of the first 1-second bar at or after `ts`."""
    import bisect
    return bisect.bisect_left(sec.ts, ts, lo)


def walk(sec: Seconds, rule: str, frames: tuple, stop_ticks: int,
         cap_s: float, flip: bool = False, rng=None) -> list[Trade]:
    """One contract, one rule, one geometry. Returns the trades taken.

    `frames` names which timeframes must agree. `flip` reverses every
    call, which is the mirror; `rng` shuffles the sides, which is the
    control. Both keep the sample and the geometry fixed.
    """
    grids = {name: build(sec, SPAN[name]) for name, _ in FRAMES}
    idx = {name: 0 for name in grids}
    out: list[Trade] = []
    free_from = 0.0
    one = grids["1m"]

    for k in range(len(one)):
        close_ts = one[k].ts + SPAN["1m"]        # when this bar completed
        if close_ts < free_from:
            continue
        # Each coarser timeframe's newest COMPLETED bar at that moment.
        ways = []
        ok = True
        for name in frames:
            g = grids[name]
            j = newest_completed(g, SPAN[name], close_ts, idx[name])
            if j is None or j + 1 < MIN_BARS:
                ok = False
                break
            idx[name] = j
            ways.append(WAY[rule](g, j))
        if not ok or not ways or any(w == "" for w in ways):
            continue
        if any(w != ways[0] for w in ways):
            continue

        side = "buy" if ways[0] == "up" else "sell"
        if flip:
            side = "sell" if side == "buy" else "buy"
        if rng is not None:
            side = rng.choice(("buy", "sell"))

        i0 = first_at_or_after(sec, close_ts)
        if i0 >= len(sec):
            break
        entry = sec.op[i0]
        res, px, ts_out, secs = resolve(sec, i0, entry, side, stop_ticks,
                                        cap_s)
        if res == "unfinished":
            break
        t = Trade(signal_ts=close_ts, entry_ts=sec.ts[i0], entry=entry,
                  side=side, stop_ticks=stop_ticks, cap_s=cap_s,
                  exit_ts=ts_out, exit=px, outcome=res, seconds=secs,
                  ways=tuple(ways))
        out.append(t)
        free_from = ts_out           # ONE POSITION AT A TIME
    return out


# ------------------------------------------------- PHASE 5: the scoring

@dataclass
class Result:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    timeouts: int = 0
    ambiguous: int = 0
    longs: int = 0
    shorts: int = 0
    gross: float = 0.0               # ticks, before costs
    gross_win: float = 0.0           # ticks earned, after cost, on winners
    gross_loss: float = 0.0          # ticks given up, after cost, positive
    worst_streak: int = 0
    max_dd: float = 0.0
    tte: list = field(default_factory=list)

    @property
    def scored(self) -> int:
        return self.wins + self.losses + self.timeouts

    @property
    def win_rate(self) -> float | None:
        d = self.wins + self.losses
        return (self.wins / d) if d else None

    def net(self, cost: float) -> float:
        return self.gross - cost * self.scored

    def per_trade(self, cost: float) -> float | None:
        return (self.net(cost) / self.scored) if self.scored else None

    @property
    def profit_factor(self) -> float | None:
        """Ticks earned over ticks given up, both already net of cost.

        A timeout counts on whichever side it landed -- it is a real
        outcome with a real price, not a third category to be quietly
        left out of the ratio.
        """
        if self.gross_loss <= 0:
            return None
        return self.gross_win / self.gross_loss


def score(trades: list[Trade], cost: float) -> Result:
    r = Result()
    equity = peak = 0.0
    streak = run = 0
    for t in trades:
        if t.outcome == "ambiguous":
            r.ambiguous += 1
            continue
        r.trades += 1
        if t.side == "buy":
            r.longs += 1
        else:
            r.shorts += 1
        if t.outcome == "target":
            r.wins += 1
        elif t.outcome == "stop":
            r.losses += 1
        else:
            r.timeouts += 1
        r.gross += t.ticks
        r.tte.append(t.seconds)
        net = t.ticks - cost
        if net >= 0:
            r.gross_win += net
        else:
            r.gross_loss += -net
        equity += net
        peak = max(peak, equity)
        r.max_dd = max(r.max_dd, peak - equity)
        if net < 0:
            run += 1
            streak = max(streak, run)
        else:
            run = 0
    r.worst_streak = streak
    return r


def wilson(hits: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    z = 1.96
    p = hits / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - half), min(1.0, c + half)


def breakeven(stop_ticks: int, cost: float) -> float:
    return (stop_ticks + cost) / (TARGET_TICKS + stop_ticks)


def _median(xs):
    if not xs:
        return None
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


# ------------------------------------------------------------- reporting

ARMS = (
    ("1m alone", ("1m",)),
    ("1m + 5m", ("1m", "5m")),
    ("1m + 15m", ("1m", "15m")),
    ("all three", ("1m", "5m", "15m")),
)


def row(label: str, r: Result, cost: float, stop: int) -> None:
    if not r.scored:
        print(f"  {label:<26} (no trades)")
        return
    wr = r.win_rate
    need = breakeven(stop, cost)
    lo, hi = wilson(r.wins, r.wins + r.losses)
    net = r.net(cost)
    verdict = "  --  " if r.scored < 30 else (
        " over " if (wr is not None and wr > need) else " under")
    print(f"  {label:<26} n={r.scored:<6,} "
          f"{'' if wr is None else f'{wr:5.1%}'} vs {need:4.0%}{verdict}"
          f"  [{lo:.0%}-{hi:.0%}]"
          f"  net {net:+9.1f}t  ${net * DOLLARS_PER_TICK:+11,.0f}"
          f"  /trade {r.per_trade(cost):+5.2f}t"
          f"  pf {'   -' if r.profit_factor is None else f'{r.profit_factor:4.2f}'}"
          f"  dd {r.max_dd:7.1f}t  streak {r.worst_streak:>3}"
          f"  L/S {r.longs}/{r.shorts}  skip {r.ambiguous}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path")
    ap.add_argument("--rule", default="momentum,structure")
    ap.add_argument("--stops", default=",".join(str(s) for s in STOP_TICKS))
    ap.add_argument("--caps", default=",".join(str(int(c))
                                               for c in TIME_CAPS_S))
    ap.add_argument("--cost", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N records -- a trial run, not a result")
    args = ap.parse_args(argv)
    rules = [r.strip() for r in args.rule.split(",") if r.strip()]
    stops = [int(s) for s in args.stops.split(",") if s.strip()]
    caps = [float(c) for c in args.caps.split(",") if c.strip()]

    print(f"file      {args.path}")
    print(f"target    {TARGET_TICKS} ticks ({TARGET_TICKS / 4:.2f} index "
          f"points, ${TARGET_TICKS * DOLLARS_PER_TICK:.2f})")
    print(f"cost      {args.cost} ticks round trip")
    print(f"entry     the open of the first 1-second bar AFTER the "
          f"signal minute closed")
    print(f"exclusion a 1-second bar spanning target AND stop is "
          f"unresolvable and is dropped")
    print(f"rolls     no trade crosses a contract change")
    print("progress on stderr")
    if args.limit:
        print(f"LIMIT     {args.limit:,} records -- a trial run, not a "
              f"result.")
    print()

    import itertools
    store = dbn.open_file(args.path)
    if args.limit:
        store = itertools.islice(store, args.limit)

    folds = []
    prog = Progress("read")
    for iid, sec in segments(store, progress=prog):
        folds.append((iid, sec))
        print(f"  contract {iid}: {len(sec):,} one-second bars",
              file=sys.stderr, flush=True)

    if not folds:
        print("no usable bars")
        return 0

    for rule in rules:
        for stop in stops:
            for cap in caps:
                print("=" * 100)
                print(f"RULE {rule.upper()}   stop {stop}t   "
                      f"time cap {cap / 60:.0f} min   "
                      f"break-even {breakeven(stop, args.cost):.1%}")
                print("=" * 100)
                keep = {}
                for label, frames in ARMS:
                    allt = []
                    for iid, sec in folds:
                        allt += walk(sec, rule, frames, stop, cap)
                    keep[label] = allt
                    row(label, score(allt, args.cost), args.cost, stop)

                base = keep["all three"]
                row("  long only",
                    score([t for t in base if t.side == "buy"], args.cost),
                    args.cost, stop)
                row("  short only",
                    score([t for t in base if t.side == "sell"], args.cost),
                    args.cost, stop)

                mirror = []
                ctrl = []
                rng = random.Random(args.seed)
                for iid, sec in folds:
                    mirror += walk(sec, rule, ("1m", "5m", "15m"), stop,
                                   cap, flip=True)
                    ctrl += walk(sec, rule, ("1m", "5m", "15m"), stop, cap,
                                 rng=rng)
                row("MIRROR (every call flipped)", score(mirror, args.cost),
                    args.cost, stop)
                row("CONTROL (sides shuffled)", score(ctrl, args.cost),
                    args.cost, stop)

                print()
                print("  WALK FORWARD -- one fold per contract, in order")
                for iid, sec in folds:
                    ft = walk(sec, rule, ("1m", "5m", "15m"), stop, cap)
                    row(f"    contract {iid}", score(ft, args.cost),
                        args.cost, stop)
                tte = [t.seconds for t in base
                       if t.outcome == "target"]
                if tte:
                    print(f"  time to target: median {_median(tte):.0f}s")
                print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
