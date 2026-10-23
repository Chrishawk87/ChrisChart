"""What the two CVD lines do before price moves. Real data only.

    python scripts/cvd_study.py data/*.dbn.zst --bar 300

WHAT IT ANSWERS, in order of how much it matters

1. DOES THE BOOK LEAD PRICE AT ALL, AND BY HOW LONG. A lead-lag scan:
   the correlation between the book line's change over one window and
   price's change over a window offset by a lag, swept from -120s to
   +120s. If the book genuinely leads, the correlation peaks at a
   NEGATIVE lag and the peak says by how many seconds. If the peak is
   at zero the two move together and there is nothing to trade; if it
   is positive, price leads the book and the whole premise is backwards.

   This is the measurement the rest of the file is scaffolding for. It
   can come back empty, and an empty answer here is worth more than a
   full page of anything else.

2. WHAT SEPARATES AN UP BAR FROM A DOWN BAR. For every completed bar,
   the two readings as they stood at 20/40/60/80% of the way through,
   grouped by how that bar actually closed. If the readings separate,
   the medians differ at the EARLY checkpoints. Separation at 80% is
   near worthless -- by then the bar has mostly happened.

3. WHETHER THE REGIMES DIFFER. Median travel while each of the four
   states held, which is the comparison about book-above versus
   book-below, stated as a number per side.

4. WHETHER ANY OF IT BEATS THE OBVIOUS. Every rate is printed against
   the base rate -- the share of bars that closed up at all. A caller
   scoring 55% where 55% of bars close green has found nothing, and a
   hit rate without its base rate beside it reads as though it had.

HONESTY NOTES

Every number is computed from readings that existed BEFORE the thing
being predicted. The engine is fed one record at a time in file order
and nothing reaches back.

Sample size is printed beside every figure, and anything under 100
observations is marked THIN. On a month of ES the early checkpoints
will have thousands; the regime splits may not.

This measures; it does not suggest, size, or place anything.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from liqmap import dbn                                   # noqa: E402
from liqmap.cvdcross import BUY, DOWN, FLAT, UP, CVDCross  # noqa: E402

CHECKS = (0.2, 0.4, 0.6, 0.8)
VERSION = "2026-10-23h"
SECTIONS = ("lead-lag", "up-vs-down", "regimes", "bar-calls",
            "remaining-move", "travel", "divergence", "stalls",
            "pushback")


def _median(xs):
    return statistics.median(xs) if xs else float("nan")


def _corr(a, b):
    n = min(len(a), len(b))
    if n < 30:
        return float("nan")
    a, b = a[:n], b[:n]
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 0 or vb <= 0:
        return float("nan")
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    return cov / math.sqrt(va * vb)


FLOOR_TRIALS = 200


def _scan_peak(a, b, steps):
    """The largest |correlation| this scan would report over a, b."""
    best = 0.0
    for k in range(-steps, steps + 1):
        if k < 0:
            x, y = a[:k], b[-k:]
        elif k > 0:
            x, y = a[k:], b[:-k]
        else:
            x, y = a, b
        c = _corr(x, y)
        if not math.isnan(c) and abs(c) > abs(best):
            best = c
    return abs(best)


def _shuffle_floor(a, b, steps, trials=FLOOR_TRIALS, seed=11):
    """What peak this scan reports when there is nothing to find.

    `b` is ROTATED, not shuffled. Shuffling would destroy the
    autocorrelation that makes the floor high in the first place and
    hand back a floor that is far too low -- which is the whole reason
    the formula version failed. A rotation keeps each series exactly
    as it is and only breaks the alignment between them.
    """
    import random as _r
    rng = _r.Random(seed)
    n = len(b)
    if n < 100:
        return float("inf")
    peaks = []
    for _ in range(trials):
        off = rng.randrange(n // 10, n - n // 10)
        rot = b[off:] + b[:off]
        peaks.append(_scan_peak(a, rot, steps))
    peaks.sort()
    # The 99th, not the 95th. A 95th-percentile bar lets one noise run
    # in twenty through, and a false lead here is not a bad grade --
    # it is money put behind randomness. The cost of being too strict
    # is missing a weak edge, which is the cheaper mistake.
    return peaks[min(len(peaks) - 1, int(0.99 * len(peaks)))]


class Study:
    """One pass over the tape, collecting everything at once."""

    def __init__(self, bar_s: float, tick: float, grid_s: float = 5.0):
        self.eng = CVDCross(sample_s=0.25, bar_seconds=bar_s,
                            tick_size=tick)
        self.bar_s = bar_s
        self.tick = tick
        self.grid_s = grid_s

        # evenly spaced samples, for the lead-lag scan
        self.g_ts: list = []
        self.g_book: list = []
        self.g_tape: list = []
        self.g_px: list = []
        self.g_bz: list = []        # book z at that instant
        self.g_tz: list = []        # tape z at that instant
        self.g_pz: list = []        # price z, filled by section 7
        self.g_side: list = []      # who led at that instant
        self.g_frac: list = []      # how much of their best they had given back
        self.g_hw: list = []        # the size of that best, in contracts
        self._next_grid = 0.0

        # per-bar observations
        self.bars: list = []
        self._open_ts = 0.0
        self._open_px = 0.0
        self._taken: dict = {}

        # Who is ahead inside each bar, and when they stop gaining.
        from liqmap.fpread import LeadWatcher
        self.lead = LeadWatcher()
        self.stalls: list = []          # (ts, px, side)
        self._seen_stall = False

        self.records = 0
        self.trades = 0
        self.books = 0
        self.refused = 0
        self.skipped = 0

    def offer(self, rec) -> None:
        self.records += 1
        ts = dbn.seconds(dbn._get(rec, "ts_event")) or 0.0
        if ts <= 0.0:
            self.refused += 1
            return

        lv = dbn._get(rec, "levels")
        if lv:
            # The INSIDE quote only. book_from() builds a sorted Book of
            # all ten levels with an object per level, which costs about
            # 22us a record against 2us for reading level zero -- roughly
            # twenty minutes of pure waste over a month of ES, for nine
            # levels nothing here looks at.
            try:
                l0 = lv[0] if isinstance(lv, (list, tuple)) else lv
                bpx = dbn.price(dbn._get(l0, "bid_px", 0))
                apx = dbn.price(dbn._get(l0, "ask_px", 0))
                bsz = float(dbn._get(l0, "bid_sz", 0) or 0)
                asz = float(dbn._get(l0, "ask_sz", 0) or 0)
                self.eng.on_book(bpx, bsz, apx, asz, ts)
                self.books += 1
            except Exception:                            # noqa: BLE001
                self.refused += 1

        t = dbn.trade_from(rec)
        if t is not None:
            self.eng.on_trade(float(t.ts), t.aggressor,
                              float(t.sz), float(t.px))
            self.trades += 1
            # the lead inside the bar, from the SAME decoded trade
            f = self.eng.bars.get(self.bar_s)
            if self.lead.on_bar(f._open_ts):
                self._seen_stall = False
            self.lead.on_trade(float(t.ts), float(t.px),
                               int(round(float(t.sz))), t.aggressor)
            c = self.lead.cur
            if c.stalled and not self._seen_stall:
                self._seen_stall = True
                self.stalls.append((c.stall_ts, c.stall_px,
                                    c.stall_side))

        # A record that carried neither a book nor a print tells this
        # study nothing, and must not be allowed to advance the clock.
        # A glob like data/*.dbn.zst happily picks up an ohlcv file
        # sitting beside the MBP-10 ones; if its timestamps happen to
        # be NEWER, every one of its rows walks the sampling grid
        # forward on a stale price and the lead-lag scan ends up
        # reading thousands of invented samples. It reports that with
        # a straight face. Skipping here makes the file order and its
        # dates irrelevant.
        if not lv and t is None:
            self.skipped += 1
            return

        self._grid(ts)
        self._bar(ts)

    def _grid(self, ts: float) -> None:
        if ts < self._next_grid:
            return
        self._next_grid = ts + self.grid_s
        s = self.eng.snapshot()
        if s.px <= 0.0:
            return
        self.g_ts.append(ts)
        self.g_book.append(s.book_cvd)
        self.g_tape.append(s.tape_cvd)
        self.g_px.append(s.px)
        self.g_bz.append(s.book_z)
        self.g_tz.append(s.tape_z)
        c = self.lead.cur
        hw = abs(c.high_water)
        self.g_side.append(c.leader)
        self.g_hw.append(hw)
        self.g_frac.append((c.give_back / hw) if hw > 0 else 0.0)

    def _bar(self, ts: float) -> None:
        f = self.eng.bars.get(self.bar_s)
        o = f._open_ts
        if o <= 0.0:
            return
        s = self.eng.snapshot()
        px = s.px
        if px <= 0.0:
            return
        if o != self._open_ts:
            if self._open_ts and self._taken:
                self._taken["close_px"] = px
                self._taken["move"] = (px - self._open_px) / self.tick
                self.bars.append(self._taken)
            self._open_ts = o
            self._open_px = px
            self._taken = {"ts": o, "open_px": px}
            return
        pct = f.elapsed(ts)
        for cp in CHECKS:
            key = f"at{int(cp * 100)}"
            if pct >= cp and key not in self._taken:
                self._taken[key] = {
                    "book_z": s.book_z, "tape_z": s.tape_z,
                    "gap": s.book_z - s.tape_z, "regime": s.regime,
                    "side": s.side, "bar_c": s.bar_c,
                    # The price AT the checkpoint. Without it there is
                    # no way to split what the bar had already done
                    # from what it had left to do, and the whole
                    # separation in section 2 stays ambiguous.
                    "px": s.px,
                }


def lead_lag(st: Study, out) -> None:
    """Does the book line move before price, and by how long?"""
    p = out
    p("\n" + "=" * 72)
    p("1. DOES THE BOOK LEAD PRICE -- and if so by how many seconds")
    p("=" * 72)
    steps = int(120 / st.grid_s)
    n = len(st.g_px)
    if n < 500:
        p(f"  only {n} grid samples; not enough to say anything.")
        return
    # first differences over the grid
    dbook = [st.g_book[i] - st.g_book[i - 1] for i in range(1, n)]
    dpx = [(st.g_px[i] - st.g_px[i - 1]) / st.tick for i in range(1, n)]
    p(f"  {n:,} samples every {st.grid_s:.0f}s "
      f"({n * st.grid_s / 3600:.1f} hours)\n")
    # THE NOISE FLOOR, MEASURED RATHER THAN ASSUMED.
    #
    # This scan reports the biggest correlation out of ~49 lags, and
    # the biggest of many is not a small number. A formula floor of
    # 3/sqrt(n) is not enough here: both series are autocorrelated
    # random walks, so their correlation varies far more than
    # independent samples would, and a scan over UNRELATED data still
    # produced a confident "lead" against that bar.
    #
    # So the floor is measured. One series is rotated by a random
    # offset and the whole scan re-run, many times. A rotation keeps
    # each series' own shape and autocorrelation exactly while
    # destroying any relationship BETWEEN them -- so the peaks it
    # produces are precisely what this scan reports when there is
    # nothing to find. The 95th percentile of those peaks is the bar
    # a real result has to clear.
    floor = _shuffle_floor(dbook, dpx, steps=int(120 / st.grid_s))
    p(f"  noise floor {floor:.4f}, measured by rotating one series")
    p(f"  {FLOOR_TRIALS} times and re-running this same scan. A peak")
    p(f"  below this is what unrelated data produces.\n")
    p(f"  {'lag':>8}  {'corr(book -> price)':>22}   meaning")
    best, best_lag = 0.0, 0
    for k in range(-steps, steps + 1):
        if k < 0:
            a, b = dbook[:k], dpx[-k:]
        elif k > 0:
            a, b = dbook[k:], dpx[:-k]
        else:
            a, b = dbook, dpx
        c = _corr(a, b)
        if not math.isnan(c) and abs(c) > abs(best):
            best, best_lag = c, k
        if k % max(1, steps // 6) == 0:
            lab = ("book leads" if k < 0 else
                   ("same instant" if k == 0 else "price leads"))
            p(f"  {k * st.grid_s:+7.0f}s  {c:>22.4f}   {lab}")
    p("")
    p(f"  STRONGEST: {best:+.4f} at {best_lag * st.grid_s:+.0f}s"
      f"   (floor {floor:.4f})")
    margin = abs(best) - floor
    if abs(best) < floor:
        p("  -> NOTHING HERE. The strongest lag does not clear the")
        p("     noise floor, so it is what a scan over unrelated data")
        p("     produces. The book is not leading price on this set.")
        p("     That is a real answer, and a cheap one.")
    elif margin < floor * 0.2:
        p(f"  -> MARGINAL. It clears the floor by only {margin:+.4f},")
        p("     which is close enough to the bar that another week of")
        p("     data could put it either side. Not something to trade")
        p("     on; something to re-run with more data.")
    elif best_lag < 0:
        p(f"  -> The book moves roughly {abs(best_lag) * st.grid_s:.0f}s")
        p(f"     BEFORE price, clearing the floor by "
          f"{abs(best) - floor:+.4f}. That is the lead, and it is the")
        p("     only thing here that could be worth money.")
    elif best_lag == 0:
        p("  -> They move together. Nothing to act on: by the time the")
        p("     book has moved, price already has.")
    else:
        p(f"  -> PRICE leads the book by {best_lag * st.grid_s:.0f}s.")
        p("     The premise is backwards on this data.")


def bar_separation(st: Study, out) -> None:
    p = out
    p("\n" + "=" * 72)
    p("2. WHAT SEPARATES AN UP BAR FROM A DOWN BAR")
    p("=" * 72)
    ups = [b for b in st.bars if b.get("move", 0) > 0]
    dns = [b for b in st.bars if b.get("move", 0) < 0]
    tot = len(ups) + len(dns)
    if tot < 30:
        p(f"  only {tot} completed bars; not enough.")
        return
    base = len(ups) / tot
    p(f"  {tot:,} completed {st.bar_s:.0f}s bars   "
      f"{len(ups):,} up / {len(dns):,} down")
    p(f"  BASE RATE {base:.1%} up -- always calling one way scores "
      f"{max(base, 1 - base):.1%}")
    p("  Anything below that has found nothing.\n")
    p(f"  {'into bar':>9} {'reading':>9} {'up bars':>10} {'down bars':>10}"
      f" {'gap':>8} {'n':>7}")
    for cp in CHECKS:
        key = f"at{int(cp * 100)}"
        for field in ("book_z", "tape_z", "gap"):
            a = [b[key][field] for b in ups if key in b]
            d = [b[key][field] for b in dns if key in b]
            if len(a) < 10 or len(d) < 10:
                continue
            ma, md = _median(a), _median(d)
            thin = "  THIN" if (len(a) + len(d)) < 100 else ""
            p(f"  {cp:>8.0%} {field:>9} {ma:>10.3f} {md:>10.3f}"
              f" {ma - md:>8.3f} {len(a) + len(d):>7}{thin}")
        p("")
    p("  A reading separates up bars from down bars only if its two")
    p("  medians differ at the EARLY checkpoints. A gap that appears")
    p("  only at 80% is the bar telling you what it already did.")


def regimes(st: Study, out) -> None:
    p = out
    p("\n" + "=" * 72)
    p("3. WHAT PRICE DID INSIDE EACH REGIME")
    p("=" * 72)
    rb = st.eng.regime_board()
    p(f"  {rb['episodes']:,} stretches of at least {rb['min_seconds']}s\n")
    p(f"  {'regime':>18} {'n':>6} {'median net':>11} {'favour':>8}"
      f" {'adverse':>8} {'secs':>7}")
    for k in ("up_book_above", "up_book_below",
              "down_book_above", "down_book_below"):
        d = rb[k]
        p(f"  {k:>18} {d['n']:>6} {d['median_net']:>11} "
          f"{d['median_favour']:>8} {d['median_adverse']:>8} "
          f"{d['median_seconds']:>7}")
    p("")
    for side in (UP, DOWN):
        e = rb[f"{side}_edge"]
        verdict = ("book above runs further" if e["difference"] > 0
                   else ("book BELOW runs further" if e["difference"] < 0
                         else "no difference"))
        tag = "" if e["enough"] else "   (THIN -- not a finding)"
        p(f"  {side:>5}: {verdict}  "
          f"{e['book_above_net']}t vs {e['book_below_net']}t  "
          f"diff {e['difference']:+}t  n {e['n_above']}/{e['n_below']}{tag}")


def calls(st: Study, out) -> None:
    p = out
    p("\n" + "=" * 72)
    p("4. CALLING THE BAR BEFORE IT CLOSES")
    p("=" * 72)
    cb = st.eng.call_board()
    if not cb["bars_settled"]:
        p("  no settled bars.")
        return
    p(f"  {cb['bars_settled']:,} bars settled   base rate "
      f"{cb['base_up']:.1%} up   always-one-way scores "
      f"{cb['naive_rate']:.1%}\n")
    p(f"  {'caller':>9} {'into bar':>9} {'n':>7} {'rate':>8} {'edge':>8}")
    for caller, per in cb["callers"].items():
        for cp in cb["checkpoints"]:
            d = per[str(cp)]
            if d["rate"] is None:
                continue
            tag = "" if d["enough"] else "  THIN"
            p(f"  {caller:>9} {cp:>8.0%} {d['n']:>7} "
              f"{d['rate']:>7.1%} {d['edge']:>+7.1%}{tag}")
    p("")
    p("  EDGE is the rate minus always-one-way. Zero means no better")
    p("  than the dumbest possible rule. Only the early checkpoints")
    p("  would be worth anything.")



def remaining_move(st: Study, out) -> None:
    """The test that decides whether section 2 meant anything.

    Section 2 shows the readings separating up bars from down bars
    even at 20% into the bar. But 20% of a 300s bar is 60 seconds
    already gone, and the book line tracks price at the SAME instant
    (section 1). So that separation may be nothing more than the
    reading describing the fifth of the bar that has already
    happened.

    This splits the bar in two at each checkpoint:

        realised   open  -> checkpoint   (already done, no use)
        remaining  checkpoint -> close   (the only part you could
                                          still trade)

    and asks whether the reading at the checkpoint calls the sign of
    the REMAINING move. A reading that only knows the realised part
    scores at the base rate here and nowhere above it.

    MOMENTUM IS THE CONTROL. "Price has been going up this bar, so
    it keeps going up" is free and needs no order book. If a CVD
    reading cannot beat that, it has added nothing, however good its
    raw hit rate looks.
    """
    p = out
    p("\n" + "=" * 72)
    p("5. DOES THE READING CALL WHAT IS LEFT OF THE BAR")
    p("=" * 72)

    rows = [b for b in st.bars if "move" in b]
    if len(rows) < 100:
        p(f"  only {len(rows)} bars; not enough.")
        return

    for cp in CHECKS:
        key = f"at{int(cp * 100)}"
        obs = []
        for b in rows:
            d = b.get(key)
            if not d or not d.get("px"):
                continue
            realised = (d["px"] - b["open_px"]) / st.tick
            remaining = (b["close_px"] - d["px"]) / st.tick
            if remaining == 0:
                continue
            obs.append((d, realised, remaining))
        n = len(obs)
        if n < 100:
            p(f"\n  {cp:.0%} into the bar: only {n} usable bars. THIN.")
            continue

        base = sum(1 for _, _, r in obs if r > 0) / n
        naive = max(base, 1 - base)
        p(f"\n  {cp:.0%} into the bar -- {n:,} bars, "
          f"{base:.1%} of remainders were up")
        p(f"  always calling one way scores {naive:.1%}; "
          f"beat THAT or nothing was added")
        p(f"    {'reading':>10} {'rate':>8} {'edge':>8} {'z':>7} "
          f"{'corr':>8}   verdict")

        callers = (
            ("book_z", lambda d, rl: d["book_z"]),
            ("tape_z", lambda d, rl: d["tape_z"]),
            ("gap", lambda d, rl: d["gap"]),
            ("momentum", lambda d, rl: rl),
        )
        for name, pick in callers:
            hits = 0
            used = 0
            xs, ys = [], []
            for d, rl, rem in obs:
                v = pick(d, rl)
                xs.append(v)
                ys.append(rem)
                if v == 0:
                    continue
                used += 1
                if (v > 0) == (rem > 0):
                    hits += 1
            if used < 100:
                continue
            rate = hits / used
            edge = rate - naive
            # How many standard errors above the naive rule. With a
            # dozen cells in this section, 2 is not enough; 3 is the
            # bar for calling anything real.
            se = math.sqrt(0.25 / used)
            z = (rate - naive) / se if se > 0 else 0.0
            c = _corr(xs, ys)
            verdict = ("REAL" if z >= 3.0 else
                       ("marginal" if z >= 2.0 else "nothing"))
            p(f"    {name:>10} {rate:>7.1%} {edge:>+7.1%} {z:>7.2f} "
              f"{c:>8.3f}   {verdict}")

    p("")
    p("  READ IT THIS WAY. If `momentum` scores and the CVD readings")
    p("  do not, the separation in section 2 was the bar describing")
    p("  itself. If a CVD reading beats momentum at an EARLY")
    p("  checkpoint, that is the one genuinely useful thing in this")
    p("  whole report, and it is worth re-running on another month")
    p("  before trusting it.")



def _state_table(st: Study, p, state_fn, states, dead) -> None:
    """Forward travel from each state, against the unconditional row.

    Shared by every state-based section so they cannot drift apart in
    how they measure, and so the control is never accidentally left
    out of one of them.
    """
    n = len(st.g_px)
    step = st.grid_s
    horizons = [h for h in (15, 30, 60, 120) if h / step < n / 10]
    for H in horizons:
        k = int(round(H / step))
        rows: dict = {s_: [] for s_ in states}
        allv = []
        for i in range(n - k):
            mv = (st.g_px[i + k] - st.g_px[i]) / st.tick
            allv.append(mv)
            key = state_fn(i)
            if key in rows:
                rows[key].append(mv)
        if not allv:
            continue
        base_up = sum(1 for v in allv if v > 0) / len(allv)
        # OVERLAPPING WINDOWS ARE NOT INDEPENDENT OBSERVATIONS.
        # At a 5s grid with a 60s horizon, neighbouring windows share
        # 55 of their 60 seconds; a thousand of them carry about as
        # much information as eighty. Counting them as a thousand
        # shrinks the standard error more than threefold and turns
        # ordinary noise into a confident finding -- measured: this
        # called a planted-nothing state REAL at z -3.47 before the
        # divisor went in.
        overlap = max(1.0, H / step)
        p(f"  ---- {H}s forward "
          f"------------------------------------------------")
        p(f"    {'state':>11} {'n':>7} {'indep':>6} {'median':>8} "
          f"{'mean':>8} {'|travel|':>9} {'up%':>7} {'z':>7}   verdict")
        p(f"    {'all':>11} {len(allv):>7} "
          f"{len(allv) / overlap:>6.0f} {_median(allv):>8.2f} "
          f"{sum(allv) / len(allv):>8.2f} "
          f"{_median([abs(v) for v in allv]):>9.2f} "
          f"{base_up:>6.1%} {'--':>7}   (the control)")
        for s_ in states:
            v = rows[s_]
            if len(v) / overlap < 50:
                p(f"    {s_:>11} {len(v):>7}   too few to read")
                continue
            up = sum(1 for x in v if x > 0) / len(v)
            eff = len(v) / overlap
            se = math.sqrt(max(1e-9, base_up * (1 - base_up) / eff))
            z = (up - base_up) / se
            verdict = ("REAL" if abs(z) >= 3.0 else
                       ("marginal" if abs(z) >= 2.0 else "nothing"))
            p(f"    {s_:>11} {len(v):>7} {eff:>6.0f} {_median(v):>8.2f} "
              f"{sum(v) / len(v):>8.2f} "
              f"{_median([abs(x) for x in v]):>9.2f} "
              f"{up:>6.1%} {z:>7.2f}   {verdict}")
        p("")


def travel(st: Study, out, dead: float = 0.25) -> None:
    """Both lines positive, both negative, or pulling against each
    other -- and how far price actually travels from there.

    THE STATES, decided on the two readings alone:

        both_up     book and tape both above +dead
        both_down   both below -dead
        book_leads  they disagree and the BOOK has the larger
                    absolute reading
        tape_leads  they disagree and the TAPE does
        flat        neither clears the deadband

    For each state, price is followed forward over fixed clock
    horizons and the travel measured in ticks. Fixed clock, not
    bar-relative, because that is what a scalp is: you are in at a
    moment and out some seconds later regardless of where the candle
    boundary falls.

    EVERY ROW IS AGAINST AN `all` ROW on the same horizon. That row
    is price's unconditional travel -- what you would get entering at
    random with no reading at all. A state that matches it has told
    you nothing, however large its own numbers look. This is the
    whole measurement; the per-state figures mean nothing alone.

    Two different questions get separate columns, because a signal can
    answer one and not the other:

        median      WHICH WAY and HOW FAR, signed. Direction.
        |travel|    HOW FAR in either direction. A state can predict
                    a big move without predicting its side, which is
                    useful for sizing and useless for picking a side.

    `z` is standard errors between this state's up-rate and the
    unconditional up-rate. There are a lot of cells here, so the bar
    for REAL is 3, not 2.
    """
    p = out
    p("\n" + "=" * 72)
    p("6. WHAT PRICE DOES FROM EACH STATE OF THE TWO LINES")
    p("=" * 72)

    n = len(st.g_px)
    if n < 2000:
        p(f"  only {n} grid samples; not enough.")
        return
    step = st.grid_s
    p(f"  {n:,} samples every {step:.0f}s   deadband +/-{dead}\n")

    def state(i):
        b, t = st.g_bz[i], st.g_tz[i]
        if b > dead and t > dead:
            return "both_up"
        if b < -dead and t < -dead:
            return "both_down"
        if abs(b) <= dead and abs(t) <= dead:
            return "flat"
        if (b > dead and t < -dead) or (b < -dead and t > dead):
            return "book_leads" if abs(b) > abs(t) else "tape_leads"
        return "flat"

    states = ["both_up", "both_down", "book_leads", "tape_leads",
              "flat"]
    _state_table(st, p, state, states, dead)
    p("")
    p("  HOW TO READ IT. Compare every state row against the `all`")
    p("  row ABOVE it on the same horizon, never against 50%. If")
    p("  `both_up` shows 52% up and `all` shows 52% up, the state")
    p("  added nothing -- you would have got that entering blind.")
    p("")
    p("  If `|travel|` is larger for a state but `up%` matches the")
    p("  control, that state predicts a BIGGER move without")
    p("  predicting its direction. Worth knowing for sizing, no use")
    p("  for picking a side.")



def divergence(st: Study, out, dead: float = 0.5,
               win_s: float = 60.0) -> None:
    """Book CVD against PRICE ITSELF, not against the traded line.

    The idea this tests: replace the second line with price,
    standardised the same way, so a cross between them is a cross
    between flow and the thing flow is supposed to move.

    WHY IT IS BETTER POSED THAN WHAT CAME BEFORE. Section 1 found
    book CVD and price correlate 0.349 at the SAME instant -- they
    largely move together, and the traded line was a second view of
    the same flow. So the shared part carries nothing new, and
    anything worth having lives in the RESIDUAL: the moments when
    flow and price pull apart. That is what this measures.

        price_up    price pushed above the book  (price running
                    ahead of the flow behind it)
        book_up     book pushed above price      (flow building
                    that price has not paid for)
        together    the two inside the deadband of each other

    Either reading is defensible and they point opposite ways, which
    is exactly why it has to be measured rather than argued. Price
    ahead of flow can mean a move with nothing behind it, due to
    fade; it can equally mean price is simply leading, and fading it
    is standing in front of a train. The table answers it.

    Price is z-scored over `win_s` seconds, matched to the book
    line's own look-back so the two are on the same footing. An
    unmatched window would make one line jumpier than the other and
    the gap would measure the mismatch rather than the market.
    """
    p = out
    p("\n" + "=" * 72)
    p("7. BOOK CVD AGAINST PRICE -- the divergence read")
    p("=" * 72)

    n = len(st.g_px)
    step = st.grid_s
    k = max(3, int(round(win_s / step)))
    if n < max(2000, k * 10):
        p(f"  only {n} grid samples; not enough.")
        return

    # rolling z of price over the same look-back as the book line
    pz: list = [0.0] * n
    run: list = []
    ssum = ssq = 0.0
    for i in range(n):
        x = st.g_px[i]
        run.append(x)
        ssum += x
        ssq += x * x
        if len(run) > k:
            old = run.pop(0)
            ssum -= old
            ssq -= old * old
        m = len(run)
        if m >= 3:
            var = (ssq - ssum * ssum / m) / (m - 1)
            sd = math.sqrt(var) if var > 0 else 0.0
            pz[i] = ((x - ssum / m) / sd) if sd > 0 else 0.0
    st.g_pz = pz

    gaps = [st.g_bz[i] - pz[i] for i in range(n)]
    p(f"  price z-scored over {win_s:.0f}s ({k} samples), matched to")
    p(f"  the book line's look-back   deadband +/-{dead}")
    p(f"  gap = book_z - price_z   median {_median(gaps):+.3f}   "
      f"corr(book_z, price_z) {_corr(st.g_bz, pz):+.3f}\n")

    def state(i):
        g = st.g_bz[i] - pz[i]
        if g > dead:
            return "book_up"
        if g < -dead:
            return "price_up"
        return "together"

    _state_table(st, p, state, ["book_up", "price_up", "together"], dead)
    p("  book_up  = flow has built that price has not yet paid for.")
    p("  price_up = price has run ahead of the flow behind it.")
    p("")
    p("  If either is REAL against the control, the sign of its")
    p("  median says which way to read it, and that is the first")
    p("  thing in this whole report that would be worth putting on")
    p("  a chart. If both come back nothing, the gap between flow")
    p("  and price is noise at these horizons and the crossover")
    p("  would mean no more than the last one did.")



def stalls(st: Study, out) -> None:
    """When the winning side stops winning, does price turn?

    THE EVENT. Inside each bar the footprint totals aggressive buyers
    against aggressive sellers. This watches the running difference
    and marks the moment the side that is ahead stops making new
    highs in its own lead -- giving back a third of its best. The
    running total can still say "buyers +250" long after buyers
    stopped winning at +400; the high-water mark is what notices.

    THE QUESTION. From that moment, does price REVERSE against the
    side that stalled, or CONTINUE their way anyway?

        reverse%   share of stalls where price moved against the
                   side that had been winning

    Measured against the `all` control on the same horizon: what
    price does from a random moment with no stall involved. If the
    two match, a stall tells you nothing about what follows, however
    satisfying it is to watch.

    Sign convention: travel is measured in the STALLED LEADER's
    direction, so a positive median means they kept winning despite
    the stall and a negative one means the turn was real. Buyers and
    sellers are then directly comparable, and a result that only
    appears on one side is a warning rather than a finding.
    """
    p = out
    p("\n" + "=" * 72)
    p("8. WHEN THE WINNING SIDE STOPS WINNING")
    p("=" * 72)

    n = len(st.g_px)
    step = st.grid_s
    if n < 2000 or len(st.stalls) < 50:
        p(f"  {len(st.stalls)} stalls over {n:,} samples; not enough.")
        return

    # index the grid by time so a stall can be placed on it
    t0 = st.g_ts[0]
    def at(ts):
        i = int((ts - t0) / step)
        return i if 0 <= i < n else -1

    p(f"  {len(st.stalls):,} stalls in {len(st.bars):,} bars "
      f"({len(st.stalls) / max(1, len(st.bars)):.0%} of bars)")
    p(f"  a stall is the leader giving back "
      f"{st.lead.stall_frac:.0%} of their own best lead, "
      f"minimum {st.lead.stall_min} contracts\n")

    for H in (15, 30, 60, 120):
        k = int(round(H / step))
        if k >= n // 10:
            continue
        # the control: every grid moment, unsigned
        allv = [(st.g_px[i + k] - st.g_px[i]) / st.tick
                for i in range(n - k)]
        base_rev = sum(1 for v in allv if v < 0) / len(allv)

        # STALLS ARE EVENTS, NOT A ROLLING WINDOW. The control walks
        # every grid sample, so its windows overlap and its count has
        # to be divided down. Stalls are discrete and usually minutes
        # apart, so most of them ARE independent -- dividing them by
        # the same factor threw away five sixths of a perfectly good
        # sample and reported "too few to read" on 157 events.
        # Instead: count a stall, then skip any that begin before its
        # window has closed.
        rows = {"buyers": [], "sellers": []}
        indep = {"buyers": 0, "sellers": 0}
        last_used = {"buyers": -1e18, "sellers": -1e18}
        for ts, px, side in sorted(st.stalls):
            i = at(ts)
            if i < 0 or i + k >= n:
                continue
            mv = (st.g_px[i + k] - st.g_px[i]) / st.tick
            # signed in the STALLED LEADER's direction
            rows[side].append(mv if side == "buyers" else -mv)
            if ts - last_used[side] >= H:
                indep[side] += 1
                last_used[side] = ts

        overlap = max(1.0, H / step)
        p(f"  ---- {H}s after the stall "
          f"--------------------------------------")
        p(f"    {'side':>9} {'n':>7} {'indep':>6} {'median':>8} "
          f"{'mean':>8} {'reverse%':>9} {'z':>7}   verdict")
        p(f"    {'control':>9} {len(allv):>7} "
          f"{len(allv) / overlap:>6.0f} "
          f"{_median(allv):>8.2f} {sum(allv) / len(allv):>8.2f} "
          f"{base_rev:>8.1%} {'--':>7}   (no stall involved)")
        for side in ("buyers", "sellers"):
            v = rows[side]
            eff = indep[side]
            if eff < 30:
                p(f"    {side:>9} {len(v):>7}   too few to read")
                continue
            rev = sum(1 for x in v if x < 0) / len(v)
            se = math.sqrt(max(1e-9, base_rev * (1 - base_rev) / eff))
            z = (rev - base_rev) / se
            verdict = ("REAL" if abs(z) >= 3.0 else
                       ("marginal" if abs(z) >= 2.0 else "nothing"))
            p(f"    {side:>9} {len(v):>7} {eff:>6.0f} "
              f"{_median(v):>8.2f} {sum(v) / len(v):>8.2f} "
              f"{rev:>8.1%} {z:>7.2f}   {verdict}")
        p("")

    p("  Travel is signed in the stalled leader's direction, so a")
    p("  NEGATIVE median means price turned against them -- the")
    p("  reversal -- and a positive one means they carried on")
    p("  regardless.")
    p("")
    p("  A result on one side only is a warning, not a finding: the")
    p("  same event should behave the same way whichever side it")
    p("  happens to, and one-sided results are usually the market's")
    p("  drift over the sample rather than the event.")



BUCKETS = ((0.00, 0.10), (0.10, 0.20), (0.20, 0.35),
           (0.35, 0.50), (0.50, 0.75), (0.75, 2.00))


def pushback(st: Study, out, min_lead: int = 50) -> None:
    """How much offset does it take before price turns?

    Price never stops moving, so the question is not whether the
    other side is pushing back -- they always are -- but HOW MUCH
    push-back the leader has absorbed before their move gives out.

    At every instant with a meaningful lead, two numbers are taken:
    how much of their own best lead the leader has handed back, and
    what price then does over the next N seconds, signed in the
    LEADER's direction. A negative median means price turned against
    them; positive means they carried on.

    Bucketing the give-back is the point. If there is a level at
    which a move reliably gives out, it shows as a MONOTONE slide
    down the buckets: carrying on while they are barely offset,
    turning once they have handed back half. A flat column means the
    amount of push-back does not matter, which is itself an answer
    and the one to expect given everything above.

    Each row is against the `all` control on the same horizon --
    price's unconditional travel from a random moment. A bucket that
    matches the control has told you nothing.
    """
    p = out
    p("\n" + "=" * 72)
    p("9. HOW MUCH PUSH-BACK BEFORE THE MOVE GIVES OUT")
    p("=" * 72)

    n = len(st.g_px)
    step = st.grid_s
    if n < 2000 or not st.g_side:
        p(f"  only {n} samples; not enough.")
        return
    live = sum(1 for h in st.g_hw if h >= min_lead)
    p(f"  {live:,} of {n:,} samples had a lead of at least "
      f"{min_lead} contracts ({live / n:.0%})")
    p(f"  give-back is measured against that leader's OWN best, so a")
    p(f"  quiet bar and a busy one are held to the same standard\n")

    for H in (15, 30, 60):
        k = int(round(H / step))
        if k >= n // 10:
            continue
        allv = [(st.g_px[i + k] - st.g_px[i]) / st.tick
                for i in range(n - k)]
        base_rev = sum(1 for v in allv if v < 0) / len(allv)
        overlap = max(1.0, H / step)

        p(f"  ---- {H}s forward "
          f"------------------------------------------------")
        p(f"    {'given back':>12} {'n':>7} {'indep':>6} {'median':>8} "
          f"{'mean':>8} {'against%':>9} {'z':>7}   verdict")
        p(f"    {'control':>12} {len(allv):>7} "
          f"{len(allv) / overlap:>6.0f} {_median(allv):>8.2f} "
          f"{sum(allv) / len(allv):>8.2f} {base_rev:>8.1%} "
          f"{'--':>7}   (no leader involved)")
        for lo_f, hi_f in BUCKETS:
            v = []
            for i in range(n - k):
                if st.g_hw[i] < min_lead:
                    continue
                side = st.g_side[i]
                if side not in ("buyers", "sellers"):
                    continue
                f = st.g_frac[i]
                if not (lo_f <= f < hi_f):
                    continue
                mv = (st.g_px[i + k] - st.g_px[i]) / st.tick
                v.append(mv if side == "buyers" else -mv)
            eff = len(v) / overlap
            label = f"{lo_f:.0%}-{min(hi_f, 1.0):.0%}"
            if eff < 50:
                p(f"    {label:>12} {len(v):>7}   too few to read")
                continue
            rev = sum(1 for x in v if x < 0) / len(v)
            se = math.sqrt(max(1e-9, base_rev * (1 - base_rev) / eff))
            z = (rev - base_rev) / se
            verdict = ("REAL" if abs(z) >= 3.0 else
                       ("marginal" if abs(z) >= 2.0 else "nothing"))
            p(f"    {label:>12} {len(v):>7} {eff:>6.0f} "
              f"{_median(v):>8.2f} {sum(v) / len(v):>8.2f} "
              f"{rev:>8.1%} {z:>7.2f}   {verdict}")
        p("")

    p("  READ THE COLUMN, NOT THE ROWS. One bucket clearing the bar")
    p("  on its own is one cell out of eighteen and will happen by")
    p("  chance. What would be real is the median sliding steadily")
    p("  DOWN as the give-back grows -- a move that carries on while")
    p("  barely offset and turns once it has handed back half. A")
    p("  flat column says the amount of push-back does not matter.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", help="MBP-10 .dbn or .dbn.zst")
    ap.add_argument("--bar", type=float, default=300.0,
                    help="bar length in seconds (60, 300, 900)")
    ap.add_argument("--tick", type=float, default=0.25)
    ap.add_argument("--grid", type=float, default=5.0,
                    help="lead-lag sampling grid, seconds")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N records (0 = all)")
    ap.add_argument("--out", default="", help="also write the report here")
    ap.add_argument("--dead", type=float, default=0.25,
                    help="deadband in sigma for calling a line up/down")
    ap.add_argument("--gap-dead", type=float, default=0.5,
                    dest="gap_dead",
                    help="deadband in sigma for book-vs-price divergence")
    ap.add_argument("--pwin", type=float, default=60.0,
                    help="seconds of look-back for the price z-score")
    ap.add_argument("--version", action="store_true",
                    help="print which build this is and exit")
    a = ap.parse_args()
    if a.version:
        print(f"cvd_study {VERSION} -- sections: {', '.join(SECTIONS)}")
        return 0
    if not a.files:
        ap.error("the following arguments are required: files")

    lines: list = []

    def p(s="", flush=False):
        print(s, flush=flush)
        lines.append(s)

    st = Study(bar_s=a.bar, tick=a.tick, grid_s=a.grid)
    t0 = time.time()
    for path in a.files:
        p(f"reading {path} ...")
        try:
            store = dbn.open_file(path)
        except Exception as e:                           # noqa: BLE001
            p(f"  could not open: {e}")
            continue
        mark = time.time()
        start_n = st.records
        for rec in store:
            st.offer(rec)
            # A silent long run cannot be told from a hung one.
            if st.records % 1_000_000 == 0:
                now = time.time()
                # Rate over THIS MILLION, not since the run began.
                # Dividing this file's records by the whole run's
                # elapsed time made the second and third files report
                # 0k/s while they were going at full speed.
                span = max(1e-6, now - mark)
                p(f"  {st.records:>12,} records   "
                  f"{1000 / span:,.0f}k/s   "
                  f"{span:4.0f}s for the last million   "
                  f"{st.trades:,} trades", flush=True)
                mark = now
            if a.limit and st.records >= a.limit:
                break
        if a.limit and st.records >= a.limit:
            break

    took = time.time() - t0
    p("")
    p("=" * 72)
    p(f"ES CVD STUDY -- {a.bar:.0f}s bars   build {VERSION}")
    p("=" * 72)
    p(f"  {st.records:,} records in {took:,.0f}s   "
      f"{st.books:,} books   {st.trades:,} trades   "
      f"{st.refused:,} refused   {st.skipped:,} skipped")
    if st.skipped > st.records * 0.2:
        p(f"  NOTE: {st.skipped:,} records carried neither a book nor a")
        p("  print. That usually means a non-MBP file (an ohlcv one, say)")
        p("  was caught by the glob. They were skipped, not analysed.")
    p(f"  {len(st.bars):,} completed bars   "
      f"{len(st.g_px):,} grid samples")

    lead_lag(st, p)
    bar_separation(st, p)
    regimes(st, p)
    calls(st, p)
    remaining_move(st, p)
    travel(st, p, dead=a.dead)
    divergence(st, p, dead=a.gap_dead, win_s=a.pwin)
    stalls(st, p)
    pushback(st, p)

    p("\n" + "=" * 72)
    p("READ THIS PART")
    p("=" * 72)
    p("  Section 1 decides whether any of the rest matters. If the")
    p("  strongest correlation sits at zero lag or at a positive one,")
    p("  the book is not leading price on this data and no amount of")
    p("  regime slicing in sections 2-4 will change that.")
    p("")
    p("  Every rate in sections 2 and 4 is against the base rate. A")
    p("  number that merely matches it is a finding of nothing, which")
    p("  is a real result and worth knowing in an afternoon rather")
    p("  than after a month of live trading.")

    if a.out:
        Path(a.out).write_text("\n".join(lines) + "\n")
        print(f"\nwritten to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
