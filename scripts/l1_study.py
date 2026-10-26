#!/usr/bin/env python3
"""Does a cleared inside level, or a defended one, precede price?

    python scripts/l1_study.py data/ES_c_0_mbp-10_*.dbn.zst --out l1_findings.txt

Four states, two mirror pairs:

    permanent_ask   the offer was cleared and nobody replaced it
    permanent_bid   the bid was cleared and nobody replaced it
    failed_ask      the offer emptied and was instantly rebuilt
    failed_bid      the bid emptied and was instantly rebuilt

The thesis under test says permanent_ask precedes UP and permanent_bid
precedes DOWN (ride the structural step), while failed_bid precedes UP
and failed_ask precedes DOWN (fade into the defended wall).

THE PAIR IS THE TEST, and it is the only thing separating a finding
from this sample's drift. `permanent_ask` and `permanent_bid` are the
same event on opposite sides, so a real effect appears in BOTH with
the same sign once each is read in its own direction. One alone is
the market's drift across these hours wearing the event's name. The
same goes for the failed pair, and the two pairs must point OPPOSITE
ways -- if `permanent_ask` and `failed_ask` both precede the same
direction, the reading is not about defence at all.

WHAT IS MEASURED AGAINST WHAT. Every state is compared to an
unconditional control drawn from the same book updates, never against
50%. ES does not go up half the time at every horizon, and a state
that merely matches the control has added nothing.

Horizons start at 250ms, because the claim is about microstructure.
A 2ms event that only shows up at 30 seconds is not the thing the
thesis describes.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from liqmap import dbn                                       # noqa: E402
from liqmap.l1queue import ASK, BID, FAILED, PERMANENT, L1QueueWatcher  # noqa: E402

BUILD = "2026-10-26a"

HORIZONS_MS = (250, 500, 1000, 2500, 5000, 15000)
STATES = ("permanent_ask", "permanent_bid", "failed_ask", "failed_bid")

# Price is sampled on a grid so the control and the events are drawn
# from the same clock. Sampling at event times only would compare
# "moments when something happened" against "all moments", and busy
# moments move differently from quiet ones for reasons that have
# nothing to do with the event.
GRID_MS = 50


class Track:
    """Mid price on a fixed grid, plus the events, on the same clock."""

    def __init__(self, tick: float) -> None:
        self.tick = tick
        self.ts: list[float] = []
        self.mid: list[float] = []
        self.events: list[tuple[float, str]] = []     # (ts, state)
        self._next_grid = 0.0

    def sample(self, ts: float, bid: float, ask: float) -> None:
        if bid <= 0.0 or ask <= 0.0:
            return
        if self._next_grid == 0.0:
            self._next_grid = ts
        if ts < self._next_grid:
            return
        self.ts.append(ts)
        self.mid.append((bid + ask) / 2.0)
        step = GRID_MS / 1000.0
        # Jump the grid forward rather than stepping it, so a gap in the
        # data does not produce a burst of samples at identical prices.
        while self._next_grid <= ts:
            self._next_grid += step

    def add_event(self, ts: float, state: str) -> None:
        self.events.append((ts, state))


def _index_at_or_after(ts_list: list[float], t: float, lo: int = 0) -> int:
    hi = len(ts_list)
    while lo < hi:
        mid = (lo + hi) // 2
        if ts_list[mid] < t:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _forward(tr: Track, i: int, horizon_s: float) -> float | None:
    """Ticks of mid move from sample i to i + horizon."""
    j = _index_at_or_after(tr.ts, tr.ts[i] + horizon_s, i)
    if j >= len(tr.ts):
        return None
    return (tr.mid[j] - tr.mid[i]) / tr.tick


def _stats(vals: list[float], eff_n: float) -> tuple[float, float, float]:
    """up fraction, mean, and a standard error that respects overlap."""
    if not vals:
        return 0.0, 0.0, 0.0
    up = sum(1 for v in vals if v > 0) / len(vals)
    mean = sum(vals) / len(vals)
    se = math.sqrt(max(1e-12, up * (1.0 - up) / max(1.0, eff_n)))
    return up, mean, se


def _read(paths: list[str], tick: float, window_ms: float,
          refill_min: int, drain_frac: float) -> Track:
    tr = Track(tick)
    q = L1QueueWatcher(tick_size=tick, window_ms=window_ms,
                       refill_min=refill_min, drain_frac=drain_frac)
    n = 0
    started = time.time()
    last_ts = 0.0

    for path in paths:
        print(f"reading {path} ...", flush=True)
        store = dbn.open_file(path)
        for rec in store:
            ts = dbn.seconds(dbn._get(rec, "ts_event")) or 0.0
            if ts <= 0.0:
                continue
            lv = dbn._get(rec, "levels")
            if not lv:
                continue
            l0 = lv[0]
            bpx = dbn.price(dbn._get(l0, "bid_px", 0))
            apx = dbn.price(dbn._get(l0, "ask_px", 0))
            bsz = int(dbn._get(l0, "bid_sz", 0) or 0)
            asz = int(dbn._get(l0, "ask_sz", 0) or 0)
            if bpx <= 0.0 or apx <= 0.0 or apx <= bpx:
                continue

            last_ts = ts
            tr.sample(ts, bpx, apx)
            for e in q.on_book(ts, bpx, bsz, apx, asz):
                side = "ask" if e.side == ASK else "bid"
                kind = "permanent" if e.kind == PERMANENT else "failed"
                tr.add_event(e.ts, f"{kind}_{side}")

            n += 1
            if n % 2_000_000 == 0:
                rate = n / max(1e-9, time.time() - started)
                print(f"  {n:>12,} records  {rate/1000:.0f}k/s  "
                      f"{len(tr.events):,} events", flush=True)

    for e in q.flush(last_ts + 1.0):
        side = "ask" if e.side == ASK else "bid"
        kind = "permanent" if e.kind == PERMANENT else "failed"
        tr.add_event(e.ts, f"{kind}_{side}")

    print(f"  {n:,} records in {time.time()-started:.0f}s   "
          f"{len(tr.ts):,} grid samples   {len(tr.events):,} events")
    return tr


def run(tr: Track, out) -> None:
    def p(s: str = "") -> None:
        print(s)
        out.write(s + "\n")

    p("=" * 72)
    p(f"ES LEVEL-1 QUEUE STUDY   build {BUILD}")
    p("=" * 72)
    p(f"  {len(tr.ts):,} grid samples every {GRID_MS}ms")
    p(f"  {len(tr.events):,} resolved events")
    counts: dict[str, int] = {s: 0 for s in STATES}
    for _, s in tr.events:
        counts[s] = counts.get(s, 0) + 1
    for s in STATES:
        p(f"    {s:<16} {counts[s]:>10,}")
    p()

    if len(tr.ts) < 1000:
        p("  not enough grid samples to say anything.")
        return

    for ms in HORIZONS_MS:
        h = ms / 1000.0
        p("-" * 72)
        p(f"  {ms}ms forward")
        p("-" * 72)

        # The control: every grid sample, unconditionally.
        ctrl = [v for i in range(len(tr.ts))
                if (v := _forward(tr, i, h)) is not None]
        if not ctrl:
            p("    no forward samples at this horizon")
            continue
        # Overlapping windows are not independent observations.
        overlap = max(1.0, h / (GRID_MS / 1000.0))
        c_up, c_mean, _ = _stats(ctrl, len(ctrl) / overlap)

        p(f"    {'state':<16}{'n':>9}{'up%':>9}{'mean t':>10}"
          f"{'vs ctrl':>10}{'z':>8}   verdict")
        p(f"    {'control':<16}{len(ctrl):>9,}{c_up*100:>8.1f}%"
          f"{c_mean:>10.3f}{'--':>10}{'--':>8}   (unconditional)")

        for state in STATES:
            idx = [_index_at_or_after(tr.ts, t)
                   for t, s in tr.events if s == state]
            vals = [v for i in idx
                    if i < len(tr.ts) and (v := _forward(tr, i, h)) is not None]
            if len(vals) < 100:
                p(f"    {state:<16}{len(vals):>9,}{'':>9}{'':>10}"
                  f"{'':>10}{'':>8}   too few to read")
                continue
            # Events are discrete occurrences, not a sliding window, so
            # they count their own independence rather than inheriting
            # the control's overlap divisor.
            up, mean, se = _stats(vals, len(vals))
            z = (up - c_up) / se if se > 0 else 0.0
            verdict = ("REAL" if abs(z) >= 3.0
                       else "marginal" if abs(z) >= 2.0 else "nothing")
            p(f"    {state:<16}{len(vals):>9,}{up*100:>8.1f}%"
              f"{mean:>10.3f}{(up-c_up)*100:>+9.1f}%{z:>8.2f}   {verdict}")
        p()

    p("=" * 72)
    p("HOW TO READ IT")
    p("=" * 72)
    p("  Compare every row to the CONTROL on the same horizon, never")
    p("  to 50%. A state matching the control has added nothing.")
    p()
    p("  THE PAIRS ARE THE TEST.")
    p("    permanent_ask and permanent_bid are the same event on")
    p("    opposite sides. A real effect appears in BOTH, with the")
    p("    same sign once each is read in its own direction. One")
    p("    alone is this sample's drift wearing the event's name.")
    p()
    p("    The same for failed_ask and failed_bid -- AND the two")
    p("    pairs must point OPPOSITE ways. If a cleared offer and a")
    p("    defended offer both precede the same direction, the")
    p("    reading is not about defence at all.")
    p()
    p("  MEAN MATTERS MORE THAN up%. ES costs a tick to cross. A")
    p("  state that is 55% up with a mean of 0.05 ticks is a losing")
    p("  trade taken often. Read the mean against 1.0.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="MBP-10 .dbn or .dbn.zst")
    ap.add_argument("--tick", type=float, default=0.25)
    ap.add_argument("--window-ms", type=float, default=2.0,
                    help="refill window separating failed from permanent")
    ap.add_argument("--refill-min", type=int, default=50,
                    help="contracts that must come back to count as defence")
    ap.add_argument("--drain-frac", type=float, default=0.90)
    ap.add_argument("--out", default="l1_findings.txt")
    a = ap.parse_args()

    paths = [f for f in a.files if not f.endswith(".ohlcv.dbn.zst")]
    if not paths:
        print("no input files")
        return 2

    print(f"window {a.window_ms}ms   refill_min {a.refill_min}   "
          f"drain_frac {a.drain_frac}")
    tr = _read(paths, a.tick, a.window_ms, a.refill_min, a.drain_frac)
    with open(a.out, "w") as fh:
        run(tr, fh)
    print(f"\nwritten to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
