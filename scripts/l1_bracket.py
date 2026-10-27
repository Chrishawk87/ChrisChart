#!/usr/bin/env python3
"""Three questions about the Level-1 deletion trade, on real ES.

    python scripts/l1_bracket.py data/ES_c_0_mbp-10_*.dbn.zst --out l1_bracket.txt

  1. LIFESPAN   when a rung is permanently cleared, how many ticks does
                price actually travel over the next 500 / 1000 / 2000ms
  2. WIN RATE   out of 1,000 such events, how many reach +6 ticks BEFORE
                reaching -4
  3. TIME BOMB  how many reach neither inside 800ms and get flattened

Three things the specification left out, each of which would have made
the answer fiction:

  THE PATH, NOT THE ENDPOINT. "Up 2 ticks at 1000ms" does not mean the
  trade won -- it may have printed -4 at 200ms and stopped you out on
  the way. Every trade here is walked book update by book update until
  something touches. Sampling endpoints is the single commonest way a
  backtest of this shape comes out profitable and isn't.

  HONEST FILLS. A long pays the ASK and exits on the BID; a short does
  the reverse. On ES the spread is a tick, so a +6 target needs SEVEN
  ticks of movement from the prevailing bid and a -4 stop needs only
  THREE against. The bracket is far more lopsided than the numbers
  6 and 4 suggest, and a mid-price simulation hides that entirely.

  LATENCY. You cannot act at the microsecond the deletion confirms. A
  Databento feed into Python into a broker is tens of milliseconds, and
  the whole premise is beating the step -- so entry is deliberately
  delayed and swept across several values. If the edge only survives at
  0ms, there is no edge, because 0ms is not available.

Also reported, because the detector produces them anyway: the mirror
FAILED-deletion trade, taken in the opposite direction, on the thesis
that a defended rung is a wall to fade into.

Nothing here places an order. It reads a file and counts.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from liqmap import dbn                                              # noqa: E402
from liqmap.l1queue import ASK, BID, FAILED, PERMANENT, L1QueueWatcher  # noqa: E402

BUILD = "2026-10-26a"
DOLLARS_PER_TICK = 12.50
LIFESPAN_MS = (500, 1000, 2000)

LONG, SHORT = "long", "short"

# Which way each event says to trade.
#   a cleared offer  -> the step is up, ride it
#   a cleared bid    -> the step is down, ride it
#   a DEFENDED bid   -> the push down failed, fade it up
#   a DEFENDED offer -> the push up failed, fade it down
DIRECTION = {
    "permanent_ask": LONG,  "permanent_bid": SHORT,
    "failed_bid":    LONG,  "failed_ask":    SHORT,
}


class Trade:
    __slots__ = ("state", "side", "entry_ts", "entry_px", "deadline",
                 "target", "stop", "mfe", "mae")

    def __init__(self, state: str, side: str, ts: float, px: float,
                 tick: float, target_t: int, stop_t: int, bomb_s: float):
        self.state, self.side = state, side
        self.entry_ts, self.entry_px = ts, px
        self.deadline = ts + bomb_s
        if side == LONG:
            self.target = px + target_t * tick
            self.stop = px - stop_t * tick
        else:
            self.target = px - target_t * tick
            self.stop = px + stop_t * tick
        self.mfe = 0.0          # best it ever looked, in ticks
        self.mae = 0.0          # worst it ever looked, in ticks


class Outcome:
    __slots__ = ("target", "stop", "timeout", "ticks", "timeout_ticks",
                 "mfe", "mae")

    def __init__(self) -> None:
        self.target = 0
        self.stop = 0
        self.timeout = 0
        self.ticks: list[float] = []
        self.timeout_ticks: list[float] = []
        self.mfe: list[float] = []
        self.mae: list[float] = []

    @property
    def n(self) -> int:
        return self.target + self.stop + self.timeout


class Sim:
    """One (latency, state) simulation, run online over the stream."""

    def __init__(self, tick: float, target_t: int, stop_t: int,
                 bomb_ms: float, latency_ms: float, comm_t: float):
        self.tick = tick
        self.target_t, self.stop_t = target_t, stop_t
        self.bomb_s = bomb_ms / 1000.0
        self.latency_s = latency_ms / 1000.0
        self.comm_t = comm_t
        self.pending: list[tuple[float, str]] = []   # (fire_ts, state)
        self.open: list[Trade] = []
        self.by_state: dict[str, Outcome] = {}

    def arm(self, ts: float, state: str) -> None:
        self.pending.append((ts + self.latency_s, state))

    def _close(self, t: Trade, exit_px: float, how: str) -> None:
        sign = 1.0 if t.side == LONG else -1.0
        ticks = sign * (exit_px - t.entry_px) / self.tick - self.comm_t
        o = self.by_state.setdefault(t.state, Outcome())
        setattr(o, how, getattr(o, how) + 1)
        o.ticks.append(ticks)
        o.mfe.append(t.mfe)
        o.mae.append(t.mae)
        if how == "timeout":
            o.timeout_ticks.append(ticks)

    def on_book(self, ts: float, bid: float, ask: float) -> None:
        # --- fill anything whose latency has elapsed ---------------------
        if self.pending:
            still = []
            for fire_ts, state in self.pending:
                if ts >= fire_ts:
                    side = DIRECTION[state]
                    # Cross the spread to get in.
                    px = ask if side == LONG else bid
                    self.open.append(Trade(state, side, ts, px, self.tick,
                                           self.target_t, self.stop_t,
                                           self.bomb_s))
                else:
                    still.append((fire_ts, state))
            self.pending = still

        if not self.open:
            return

        # --- walk every open trade against this book ---------------------
        still_open: list[Trade] = []
        for t in self.open:
            # Mark to the side you would EXIT on, never the mid.
            mark = bid if t.side == LONG else ask
            sign = 1.0 if t.side == LONG else -1.0
            excursion = sign * (mark - t.entry_px) / self.tick
            if excursion > t.mfe:
                t.mfe = excursion
            if excursion < t.mae:
                t.mae = excursion

            # STOP IS CHECKED FIRST. Within one book update both levels
            # can be through; assuming the good one filled is how a
            # backtest invents money. The pessimistic read is the only
            # defensible one at this resolution.
            hit_stop = (mark <= t.stop) if t.side == LONG else (mark >= t.stop)
            hit_tgt = (mark >= t.target) if t.side == LONG else (mark <= t.target)
            if hit_stop:
                self._close(t, t.stop, "stop")
            elif hit_tgt:
                self._close(t, t.target, "target")
            elif ts >= t.deadline:
                self._close(t, mark, "timeout")
            else:
                still_open.append(t)
        self.open = still_open


class Lifespan:
    """Signed ticks travelled at each horizon, resolved online."""

    def __init__(self, tick: float):
        self.tick = tick
        self.pending: list[tuple[float, str, float, int]] = []
        self.moves: dict[tuple[str, int], list[float]] = {}

    def arm(self, ts: float, state: str, mid: float) -> None:
        for ms in LIFESPAN_MS:
            self.pending.append((ts + ms / 1000.0, state, mid, ms))

    def on_book(self, ts: float, mid: float) -> None:
        if not self.pending:
            return
        still = []
        for due, state, mid0, ms in self.pending:
            if ts >= due:
                sign = 1.0 if DIRECTION[state] == LONG else -1.0
                self.moves.setdefault((state, ms), []).append(
                    sign * (mid - mid0) / self.tick)
            else:
                still.append((due, state, mid0, ms))
        self.pending = still


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(q * (len(s) - 1))))
    return s[i]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--tick", type=float, default=0.25)
    ap.add_argument("--window-ms", type=float, default=2.0)
    ap.add_argument("--refill-min", type=int, default=50)
    ap.add_argument("--drain-frac", type=float, default=0.90)
    ap.add_argument("--target", type=int, default=6, help="ticks")
    ap.add_argument("--stop", type=int, default=4, help="ticks")
    ap.add_argument("--bomb-ms", type=float, default=800.0)
    ap.add_argument("--commission", type=float, default=0.4,
                    help="round-turn cost in ticks (~$5 on ES)")
    ap.add_argument("--latency", default="0,10,25,50,100",
                    help="entry delays in ms to sweep")
    ap.add_argument("--out", default="l1_bracket.txt")
    a = ap.parse_args()

    paths = [f for f in a.files if not f.endswith(".ohlcv.dbn.zst")]
    if not paths:
        print("no input files")
        return 2
    lats = [float(x) for x in a.latency.split(",")]

    q = L1QueueWatcher(tick_size=a.tick, window_ms=a.window_ms,
                       refill_min=a.refill_min, drain_frac=a.drain_frac)
    sims = {l: Sim(a.tick, a.target, a.stop, a.bomb_ms, l, a.commission)
            for l in lats}
    life = Lifespan(a.tick)

    n = 0
    events = 0
    started = time.time()
    last_ts = 0.0
    first_ts = 0.0

    for path in paths:
        print(f"reading {path} ...", flush=True)
        for rec in dbn.open_file(path):
            ts = dbn.seconds(dbn._get(rec, "ts_event")) or 0.0
            if ts <= 0.0:
                continue
            lv = dbn._get(rec, "levels")
            if not lv:
                continue
            l0 = lv[0]
            bid = dbn.price(dbn._get(l0, "bid_px", 0))
            ask = dbn.price(dbn._get(l0, "ask_px", 0))
            bsz = int(dbn._get(l0, "bid_sz", 0) or 0)
            asz = int(dbn._get(l0, "ask_sz", 0) or 0)
            if bid <= 0.0 or ask <= bid:
                continue
            if first_ts == 0.0:
                first_ts = ts
            last_ts = ts
            mid = (bid + ask) / 2.0

            for s in sims.values():
                s.on_book(ts, bid, ask)
            life.on_book(ts, mid)

            for e in q.on_book(ts, bid, bsz, ask, asz):
                side = "ask" if e.side == ASK else "bid"
                kind = "permanent" if e.kind == PERMANENT else "failed"
                state = f"{kind}_{side}"
                events += 1
                for s in sims.values():
                    s.arm(e.ts, state)
                life.arm(e.ts, state, mid)

            n += 1
            if n % 2_000_000 == 0:
                print(f"  {n:>12,} records  {n/(time.time()-started)/1000:.0f}k/s"
                      f"  {events:,} events", flush=True)

    hours = (last_ts - first_ts) / 3600.0 if last_ts > first_ts else 0.0
    out = open(a.out, "w")

    def p(s: str = "") -> None:
        print(s)
        out.write(s + "\n")

    p("=" * 74)
    p(f"ES LEVEL-1 DELETION BRACKET STUDY   build {BUILD}")
    p("=" * 74)
    p(f"  {n:,} records   {hours:.1f} hours   {events:,} resolved events")
    p(f"  target +{a.target}t   stop -{a.stop}t   time bomb {a.bomb_ms:.0f}ms")
    p(f"  commission {a.commission}t round turn   tick = ${DOLLARS_PER_TICK}")
    p(f"  fills cross the spread BOTH ways; stop checked before target")
    be = (a.stop + a.commission) / (a.target + a.stop)
    p(f"  break-even win rate, ignoring timeouts: {be:.1%}")
    p()

    # ---- 1. LIFESPAN ----------------------------------------------------
    p("=" * 74)
    p("1. THE DELETION LIFESPAN -- how far does price actually go")
    p("=" * 74)
    p("  signed in the direction the event says to trade, mid-to-mid")
    p()
    p(f"  {'state':<16}{'ms':>7}{'n':>9}{'mean':>8}{'median':>8}"
      f"{'p25':>7}{'p75':>7}{'>=+' + str(a.target) + 't':>8}")
    for state in DIRECTION:
        for ms in LIFESPAN_MS:
            v = life.moves.get((state, ms), [])
            if not v:
                continue
            reach = sum(1 for x in v if x >= a.target) / len(v)
            p(f"  {state:<16}{ms:>7}{len(v):>9,}{sum(v)/len(v):>8.2f}"
              f"{pct(v,0.5):>8.2f}{pct(v,0.25):>7.2f}{pct(v,0.75):>7.2f}"
              f"{reach*100:>7.1f}%")
        p()
    p("  A mean near zero with wide quartiles is a coin flip with")
    p("  volatility, not a direction. The last column is the fraction")
    p("  that EVER reached the target at that instant -- not the same")
    p("  as reaching it before the stop, which is question 2.")
    p()

    # ---- 2 & 3. BRACKET -------------------------------------------------
    p("=" * 74)
    p("2 & 3. THE BRACKET -- target, stop, or choked out")
    p("=" * 74)
    for lat in lats:
        s = sims[lat]
        p(f"  --- entry delayed {lat:.0f}ms " + "-" * 44)
        p(f"  {'state':<16}{'n':>8}{'target':>9}{'stop':>8}{'timeout':>9}"
          f"{'per 1000':>10}{'exp t':>8}{'$/trade':>9}")
        for state in DIRECTION:
            o = s.by_state.get(state)
            if not o or o.n == 0:
                continue
            exp = sum(o.ticks) / len(o.ticks)
            p(f"  {state:<16}{o.n:>8,}{o.target/o.n*100:>8.1f}%"
              f"{o.stop/o.n*100:>7.1f}%{o.timeout/o.n*100:>8.1f}%"
              f"{o.target/o.n*1000:>9.0f}{exp:>8.3f}"
              f"{exp*DOLLARS_PER_TICK:>9.2f}")
        p()

    p("=" * 74)
    p("3. THE TIME-BOMB DECAY, in detail")
    p("=" * 74)
    lat0 = lats[0]
    s = sims[lat0]
    p(f"  at {lat0:.0f}ms entry delay")
    p(f"  {'state':<16}{'timeouts':>10}{'share':>8}{'mean t':>9}"
      f"{'mean MFE':>10}{'mean MAE':>10}")
    for state in DIRECTION:
        o = s.by_state.get(state)
        if not o or o.n == 0:
            continue
        tt = o.timeout_ticks
        mt = sum(tt) / len(tt) if tt else 0.0
        mfe = sum(o.mfe) / len(o.mfe) if o.mfe else 0.0
        mae = sum(o.mae) / len(o.mae) if o.mae else 0.0
        p(f"  {state:<16}{o.timeout:>10,}{o.timeout/o.n*100:>7.1f}%"
          f"{mt:>9.3f}{mfe:>10.2f}{mae:>10.2f}")
    p()
    p("  A choke-out is NOT a scratch. The flatten crosses the spread,")
    p("  so its mean should be read against zero and will usually be")
    p("  below it. MFE and MAE say how close the trades came: an MFE")
    p(f"  well under +{a.target} means the target was never realistic,")
    p(f"  and an MAE near -{a.stop} means they were nearly all stopped.")
    p()
    p("=" * 74)
    p("READ THIS PART")
    p("=" * 74)
    p("  THE PAIR IS THE TEST. permanent_ask and permanent_bid are the")
    p("  same event mirrored. An expectancy that is positive for one")
    p("  and negative for the other is this sample's drift, not an")
    p("  edge -- a real effect pays on both sides.")
    p()
    p("  LATENCY IS THE WHOLE QUESTION. If expectancy is positive at")
    p("  0ms and gone by 25ms, there is nothing here: 0ms does not")
    p("  exist outside this file. Read the sweep, not the best row.")
    p()
    p("  Expectancy already includes the spread on entry and exit and")
    p("  the commission. A positive number is net.")

    out.close()
    print(f"\nwritten to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
