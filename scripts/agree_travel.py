#!/usr/bin/env python3
"""When 1m, 5m and 15m agree, how far does price go before one disagrees?

    python scripts/agree_travel.py data/ES_c_0_mbp-10_2026-10-08.dbn.zst

THIS IS AN EVENT STUDY, NOT A BACKTEST

No win rate, no expectancy, no verdict. The output is a distribution: given
that all three timeframes agreed at a moment, how far did price travel, for
how long, and what ended it. What to do with that is the reader's business.

WHAT "AGREE" MEANS, AND WHY IT IS CAUSAL

Every bar is stamped at its start and its direction is read the way the live
panel reads it:

    direction = sign(price now - the price this bar opened at)

Both halves are known at the moment they are used. There is no bar close in
any of this, nothing is confirmed later, and no measurement reaches past the
instant it is dated. A reading at 10:04:30 uses the 15m bar that opened at
10:00, the 5m that opened at 10:00 and the 1m that opened at 10:04 -- all
three of which have already happened.

FLAT IS NOT DISAGREEMENT

At the first print of a new bar, price IS the open, so that timeframe is
flat. A convention that ends the episode there would cut every run at the
next minute boundary and report a tick count that is really a measure of
where the minute boundaries fell. So an episode ends when a timeframe turns
OPPOSITE. How long until the weaker condition -- all three still strictly
agreeing -- is measured too, and both are printed, because the gap between
them is itself worth seeing.

AN EPISODE ENDS AT THE 15-MINUTE CLOSE

Whatever else happens. That is how the trade is taken, so that is where the
measurement stops.

THE NULL IS A RANDOM START OF THE SAME LENGTH

Price travels in every window; "it went eight ticks" means nothing on its
own. The obvious null -- the same window measured the other way -- does NOT
work here, and the reason is worth stating because it is not obvious: an
episode ENDS when price turns against the agreement, so the adverse
excursion is capped by the stopping rule itself. It came out at zero beyond
two ticks for every group, which says nothing about the market and
everything about where the measurement was told to stop.

So the null is a window of the SAME DURATION starting at a random moment in
the session, scored in a direction chosen by a coin. Same lengths, same
tape, no agreement. If agreement carries nothing, the two columns meet.

AND THE STOPPING RULE IS NOT THE ONLY VIEW

"How far before one disagrees" is one question and it is the one that was
asked. But its answer is bounded by its own stopping rule, so the same
episodes are also measured over FIXED horizons from the moment of
agreement, which is the number a trade actually lives on.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER. It counts travel.
"""

from __future__ import annotations

import argparse
import bisect
import itertools
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap import dbn                                      # noqa: E402

TICK = 0.25
FRAMES = (("1m", 60.0), ("5m", 300.0), ("15m", 900.0))
TRADE_FRAME_S = 900.0                  # the bar the trade is taken on

# The projection. How often agreement reached each of these before ending.
PROJECTION = (2, 4, 6, 8, 12, 16, 20, 24, 32)

# Where in the 15-minute bar the agreement formed.
OFFSET_BUCKETS = ((0.0, 180.0), (180.0, 360.0), (360.0, 540.0),
                  (540.0, 720.0), (720.0, 900.0))

MIN_GROUP = 30                         # below this a group is a count
HEARTBEAT = 500_000

# Fixed horizons from the moment of agreement, ignoring the stopping rule.
HORIZONS_S = (15.0, 30.0, 60.0, 120.0)

CONTROL_DRAWS = 4                      # random windows per episode


class Progress:
    """A heartbeat on stderr. A pass that prints nothing for minutes is
    indistinguishable from a hang."""

    def __init__(self, label: str, every: int = HEARTBEAT, out=None) -> None:
        self.label = label
        self.every = max(1, every)
        self.n = 0
        self.extra = ""
        self.out = out if out is not None else sys.stderr
        self.start = time.time()

    def tick(self) -> None:
        self.n += 1
        if self.n % self.every:
            return
        el = max(1e-9, time.time() - self.start)
        print(f"  [{self.label}] {self.n:,}  {el:,.0f}s  "
              f"{self.n / el:,.0f}/s  {self.extra}",
              file=self.out, flush=True)

    def done(self) -> None:
        el = max(1e-9, time.time() - self.start)
        print(f"  [{self.label}] done: {self.n:,} in {el:,.0f}s",
              file=self.out, flush=True)


def capped(records, limit: int):
    if limit <= 0:
        return records
    return itertools.islice(records, limit)


# ----------------------------------------------------------------- the bars

class Frame:
    """One timeframe's bar in progress, read from the tape.

    Holds the open of the bar that is forming and nothing else. The open is
    the first price printed in the bar, which is known from that print on.
    """

    def __init__(self, seconds: float) -> None:
        self.span = seconds
        self.bar_ts: float | None = None
        self.open_px = 0.0

    def feed(self, ts: float, px: float) -> bool:
        """Returns True when this print opened a new bar."""
        slot = math.floor(ts / self.span) * self.span
        if self.bar_ts is None or slot != self.bar_ts:
            self.bar_ts = slot
            self.open_px = px
            return True
        return False

    def way(self, px: float) -> str:
        if self.bar_ts is None:
            return ""
        if px > self.open_px:
            return "up"
        if px < self.open_px:
            return "down"
        return "flat"


class Tape:
    """Every print, for measuring windows that are not episodes."""

    def __init__(self) -> None:
        self.ts: list[float] = []
        self.px: list[float] = []

    def add(self, ts: float, px: float) -> None:
        self.ts.append(ts)
        self.px.append(px)

    @property
    def n(self) -> int:
        return len(self.ts)

    def travel(self, i: int, seconds: float) -> tuple[float, float] | None:
        """(up, down) favourable travel in ticks over `seconds` from index i.

        Both non-negative. Returns None when the window runs off the end,
        because a short window is not a window that went nowhere.
        """
        if i >= self.n:
            return None
        start, end = self.px[i], self.ts[i] + seconds
        if self.ts[-1] < end:
            return None
        up = down = 0.0
        j = i
        while j < self.n and self.ts[j] <= end:
            d = (self.px[j] - start) / TICK
            up = max(up, d)
            down = min(down, d)
            j += 1
        return up, -down

    def index_at(self, ts: float) -> int:
        return bisect.bisect_left(self.ts, ts)


def control_rate(tape: Tape, durations: list[float], ticks: int,
                 rng: random.Random, draws: int = CONTROL_DRAWS):
    """How often a RANDOM window of the same length reaches `ticks`.

    The direction is a coin, so each draw contributes half its up result
    and half its down. Windows that run off the end of the session are
    skipped rather than counted as failures -- the same rule the episodes
    get.
    """
    hits = 0.0
    used = 0
    if tape.n < 2:
        return None
    for d in durations:
        for _ in range(draws):
            i = rng.randrange(tape.n)
            got = tape.travel(i, d)
            if got is None:
                continue
            up, down = got
            used += 1
            hits += 0.5 * (up >= ticks) + 0.5 * (down >= ticks)
    return (hits / used) if used else None


def agreed_way(ways) -> str:
    """All three pointing the same way, with none of them flat."""
    if any(w in ("", "flat") for w in ways):
        return ""
    first = ways[0]
    return first if all(w == first for w in ways) else ""


# -------------------------------------------------------------- the episode

@dataclass
class Episode:
    """One run of agreement, from the moment it formed."""

    start_ts: float
    start_px: float
    way: str
    bar15_ts: float
    offset_s: float                  # how far into the 15m bar it formed

    end_ts: float = 0.0
    end_px: float = 0.0
    ended: str = ""                  # flip_1m | flip_5m | flip_15m |
                                     # bar_close | file_end
    first_break: str = ""            # what stopped STRICT agreement first
    break_ts: float | None = None
    break_ticks: float | None = None

    up_mfe: float = 0.0              # ticks, measured from start_px
    up_mae: float = 0.0
    prints: int = 0

    def observe(self, px: float) -> None:
        d = (px - self.start_px) / TICK
        self.up_mfe = max(self.up_mfe, d)
        self.up_mae = min(self.up_mae, d)
        self.prints += 1

    # -- in the direction that was agreed --------------------------------

    @property
    def mfe(self) -> float:
        """Favourable travel in ticks, never negative."""
        return self.up_mfe if self.way == "up" else -self.up_mae

    @property
    def mae(self) -> float:
        """Adverse travel in ticks, returned positive."""
        return -self.up_mae if self.way == "up" else self.up_mfe

    @property
    def net(self) -> float:
        d = (self.end_px - self.start_px) / TICK
        return d if self.way == "up" else -d

    @property
    def seconds(self) -> float:
        return max(0.0, self.end_ts - self.start_ts)

    def reached(self, ticks: int) -> bool:
        return self.mfe >= ticks



# ------------------------------------------------------------------ the walk

def walk(records, symbol: str = "ES", flip: bool = False,
         progress: Progress | None = None):
    """One pass. Returns (episodes, tape, prints, span_s)."""
    frames = [(name, Frame(secs)) for name, secs in FRAMES]
    tape = Tape()
    out: list[Episode] = []
    live: Episode | None = None
    # `None`, not 0.0. A file whose timestamps start at the epoch has a
    # first print at exactly 0.0, and `if first_ts == 0.0` would then
    # reassign on the SECOND print and quietly drop one from the span.
    first_ts: float | None = None
    last_ts = last_px = 0.0
    n = 0

    for kind, item in dbn.stream(records, symbol=symbol, flip=flip):
        if kind != "trade":
            continue
        ts, px = item.ts, item.px
        n += 1
        if progress is not None:
            progress.extra = f"{len(out):,} episodes"
            progress.tick()
        if first_ts is None:
            first_ts = ts
        last_ts, last_px = ts, px
        tape.add(ts, px)

        opened15 = False
        for name, f in frames:
            new = f.feed(ts, px)
            if name == "15m" and new:
                opened15 = True

        # THE 15-MINUTE CLOSE ENDS IT, whatever the timeframes say. The
        # trade is taken on that bar, so the measurement stops there.
        if live is not None and opened15:
            _close(live, ts, px, "bar_close", out)
            live = None

        ways = [f.way(px) for _, f in frames]
        now = agreed_way(ways)

        if live is None:
            if now:
                live = _begin(frames, ts, px, now)
            continue

        live.observe(px)

        # STRICT agreement lapsing is recorded but does not end it.
        if live.first_break == "" and now != live.way:
            live.first_break = _who_broke(frames, ways, live.way)
            live.break_ts = ts
            live.break_ticks = ((px - live.start_px) / TICK
                                if live.way == "up"
                                else (live.start_px - px) / TICK)

        # A FLIP ends it: some timeframe is now pointing the other way.
        other = "down" if live.way == "up" else "up"
        flipped = [nm for (nm, _), w in zip(frames, ways) if w == other]
        if flipped:
            _close(live, ts, px, f"flip_{flipped[0]}", out)
            live = None
            if now:
                live = _begin(frames, ts, px, now)

    if live is not None:
        _close(live, last_ts, last_px, "file_end", out)
    if progress is not None:
        progress.done()
    return out, tape, n, max(0.0, last_ts - (first_ts or 0.0))


def _begin(frames, ts: float, px: float, way: str) -> Episode:
    """Open an episode, dated against the 15-minute bar it sits in.

    `bar_ts or ts` was wrong here: a 15-minute slot of 0.0 is falsy, so a
    file timestamped from the epoch dated every episode against its own
    start and reported an offset of zero for all of them. The same falsy
    zero has now cost this project three times.
    """
    f15 = frames[2][1]
    b15 = f15.bar_ts if f15.bar_ts is not None else ts
    ep = Episode(start_ts=ts, start_px=px, way=way,
                 bar15_ts=b15, offset_s=ts - b15)
    ep.observe(px)
    return ep


def _close(ep: Episode, ts: float, px: float, why: str,
           out: list[Episode]) -> None:
    ep.end_ts, ep.end_px, ep.ended = ts, px, why
    out.append(ep)


def _who_broke(frames, ways, agreed: str) -> str:
    """Which timeframe stopped agreeing first. Flat counts here."""
    for (name, _), w in zip(frames, ways):
        if w != agreed:
            return name
    return ""


# ------------------------------------------------------------------ reporting

def _pct(xs, q: float):
    if not xs:
        return None
    s = sorted(xs)
    k = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[k]


def _median(xs):
    return _pct(xs, 0.5)


def projection(eps: list[Episode], label: str, tape: Tape | None = None,
               rng: random.Random | None = None) -> None:
    """How often agreement reached each distance before it ended, against
    random windows of the same lengths."""
    if not eps:
        print(f"  {label:<22} (none)")
        return
    n = len(eps)
    bits = [f"  {label:<22} n={n:<6}"]
    for t in PROJECTION:
        hit = sum(1 for e in eps if e.reached(t))
        bits.append(f"{hit / n:6.1%}")
    print("".join(bits))

    if tape is None or rng is None or n < MIN_GROUP:
        return
    durations = [e.seconds for e in eps]
    bits = [f"  {'  (random start)':<22} {'':<8}"]
    for t in PROJECTION:
        c = control_rate(tape, durations, t, rng)
        bits.append("     -" if c is None else f"{c:6.1%}")
    print("".join(bits))


def horizon_table(eps: list[Episode], tape: Tape, rng: random.Random,
                  label: str) -> None:
    """Travel over FIXED windows from the agreement, ignoring the stop.

    The episode figures are bounded by their own stopping rule. These are
    not, so they are the ones a held trade lives on.
    """
    if not eps:
        print(f"  {label:<22} (none)")
        return
    for h in HORIZONS_S:
        got = []
        for e in eps:
            i = tape.index_at(e.start_ts)
            tv = tape.travel(i, h)
            if tv is None:
                continue
            up, down = tv
            got.append(up if e.way == "up" else down)
        if not got:
            print(f"  {label:<14} {h:>5.0f}s  (no complete windows)")
            continue
        bits = [f"  {label:<14} {h:>5.0f}s  n={len(got):<6}"]
        for t in PROJECTION:
            bits.append(f"{sum(1 for g in got if g >= t) / len(got):6.1%}")
        print("".join(bits))
        if len(got) >= MIN_GROUP:
            bits = [f"  {'  (random start)':<20} {'':<9}"]
            for t in PROJECTION:
                c = control_rate(tape, [h] * len(eps), t, rng)
                bits.append("     -" if c is None else f"{c:6.1%}")
            print("".join(bits))


def projection_header() -> None:
    bits = [f"  {'group':<22} {'':<8}"]
    for t in PROJECTION:
        bits.append(f"{str(t) + 't':>6}")
    print("".join(bits))
    print("  " + "-" * (len("".join(bits)) - 2))


def spread(eps: list[Episode], label: str) -> None:
    if not eps:
        print(f"  {label:<22} (none)")
        return
    mfe = [e.mfe for e in eps]
    mae = [e.mae for e in eps]
    net = [e.net for e in eps]
    sec = [e.seconds for e in eps]
    print(f"  {label:<22} n={len(eps):<6}"
          f"  travel p50 {_median(mfe):5.1f}t  p75 {_pct(mfe, .75):5.1f}t"
          f"  p90 {_pct(mfe, .90):5.1f}t"
          f"  |  against p50 {_median(mae):5.1f}t"
          f"  |  net p50 {_median(net):5.1f}t"
          f"  |  {_median(sec):5.1f}s")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path")
    ap.add_argument("--symbol", default="ES")
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N records -- a trial run, not a result")
    ap.add_argument("--seed", type=int, default=20261015)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)

    print(f"file      {args.path}")
    print(f"frames    " + ", ".join(n for n, _ in FRAMES)
          + "; direction = price now against this bar's open")
    print(f"episode   starts when all three agree, ends on a FLIP or at "
          f"the 15m close")
    print("progress on stderr")
    if args.limit:
        print(f"LIMIT     {args.limit:,} records -- a trial run, not a "
              f"result.")
    print()

    eps, tape, prints, span = walk(
        capped(dbn.open_file(args.path), args.limit),
        symbol=args.symbol, flip=args.flip, progress=Progress("walk"))
    if not eps:
        print("no agreement episodes found")
        return 0

    bars = len({e.bar15_ts for e in eps})
    print("=" * 78)
    print(f"{len(eps):,} agreement episodes over {prints:,} prints, "
          f"{span / 3600.0:.2f} h, {bars} fifteen-minute bars "
          f"({len(eps) / max(1, bars):.1f} per bar)")
    print("=" * 78)
    print()

    print("WHAT ENDED THEM")
    tot = len(eps)
    by_end: dict[str, int] = {}
    for e in eps:
        by_end[e.ended] = by_end.get(e.ended, 0) + 1
    for why, k in sorted(by_end.items(), key=lambda kv: -kv[1]):
        print(f"  {why:<14} {k:>7,}  {k / tot:5.1%}")
    print()

    print("WHICH TIMEFRAME STOPPED AGREEING FIRST  (flat counts here)")
    by_br: dict[str, int] = {}
    for e in eps:
        by_br[e.first_break or "(never)"] = \
            by_br.get(e.first_break or "(never)", 0) + 1
    for who, k in sorted(by_br.items(), key=lambda kv: -kv[1]):
        print(f"  {who:<14} {k:>7,}  {k / tot:5.1%}")
    brk = [e.break_ticks for e in eps if e.break_ticks is not None]
    if brk:
        print(f"  travel at that first lapse: p50 {_median(brk):.1f}t, "
              f"p75 {_pct(brk, .75):.1f}t, p90 {_pct(brk, .90):.1f}t")
    print()

    print("TRAVEL WHILE AGREED  (ticks, from the price at agreement)")
    spread(eps, "all episodes")
    spread([e for e in eps if e.way == "up"], "agreed up")
    spread([e for e in eps if e.way == "down"], "agreed down")
    print()

    print("TICK PROJECTION -- share of episodes reaching each distance")
    print("  '(random start)' is a window of the SAME LENGTH from a random")
    print("  moment, scored in a coin-flip direction. That is the null.")
    print()
    print("  READ THIS TABLE WITH CARE. An episode is long BECAUSE price")
    print("  kept going, so conditioning on its duration selects for")
    print("  travel. The null is duration-matched but cannot be")
    print("  selection-matched, so a gap here is not by itself an edge --")
    print("  and you do not know an episode's length when you enter it.")
    print("  The fixed-horizon table below is the one a trade lives on.")
    projection_header()
    projection(eps, "all episodes", tape, rng)
    projection([e for e in eps if e.way == "up"], "agreed up", tape, rng)
    projection([e for e in eps if e.way == "down"], "agreed down", tape, rng)
    print()

    print("TRAVEL OVER A FIXED HORIZON from the agreement, ignoring the")
    print("stopping rule -- the number a held trade lives on.")
    projection_header()
    horizon_table(eps, tape, rng, "all")
    print()

    print("BY WHERE IT FORMED IN THE 15-MINUTE BAR")
    projection_header()
    for lo, hi in OFFSET_BUCKETS:
        sub = [e for e in eps if lo <= e.offset_s < hi]
        projection(sub, f"{lo / 60:.0f}-{hi / 60:.0f} min in", tape, rng)
    print()

    print("PER 15-MINUTE BAR")
    per: dict[float, list[Episode]] = {}
    for e in eps:
        per.setdefault(e.bar15_ts, []).append(e)
    counts = [len(v) for v in per.values()]
    firsts = [v[0] for v in per.values()]
    print(f"  episodes per bar: p50 {_median(counts)}, "
          f"p90 {_pct(counts, .90)}, max {max(counts)}")
    print(f"  bars with no agreement at all are not counted here")
    spread(firsts, "first episode of bar")
    projection_header()
    projection(firsts, "first of bar", tape, rng)
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
