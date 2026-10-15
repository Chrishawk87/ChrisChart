#!/usr/bin/env python3
"""When the stack agrees, how far does price go before it disagrees?

    python scripts/agree_travel.py data/ES_c_0_mbp-10_2026-10-08.dbn.zst

THIS IS AN EVENT STUDY, NOT A BACKTEST

No win rate, no expectancy, no verdict. The output is a distribution: given
that the stack agreed at a moment, how far did price travel, for how long,
and what ended it. What to do with that is the reader's business.

TWO RULES, BOTH MEASURED, IN ONE PASS

    open      a row is up when price is above that bar's open.
    cascade   `cascade.read` as the site runs it -- a row counts up only
              when it is GREEN **and** its bps is positive, so price must
              have moved AND the aggressors must have paid for it. The
              stack is agreed only in State.TREND or State.REVERSAL.

`open` is the looser reading and it fired 35 times a 15-minute bar. The
point of running both over the same prints, with the same scorer and the
same null, is that the difference between the columns is the rule and
nothing else. Two scripts could not tell you that.

AND TWO TRADING FRAMES

`cascade.STACKS` decides what sits under each: 15m is read over (15m, 5m,
1m) and 5m over (5m, 1m). Both are measured, so "would this work better on
a 5-minute" is a column rather than another run.

WHY IT IS CAUSAL

Every bar is stamped at its start, and both halves of every row are known
at the moment they are used: the bar's open has happened, the price now has
happened, and the delta is of prints already on the tape. Nothing is
confirmed later and no measurement reaches past the instant it is dated. A
reading at 10:04:30 uses the 15m bar that opened at 10:00, the 5m that
opened at 10:00 and the 1m that opened at 10:04.

FLAT IS NOT DISAGREEMENT

At the first print of a new bar, price IS the open, so that timeframe is
flat. A convention that ends the episode there would cut every run at the
next minute boundary and report a tick count that is really a measure of
where the minute boundaries fell. So an episode ends when a timeframe turns
OPPOSITE. How long until the weaker condition -- all three still strictly
agreeing -- is measured too, and both are printed, because the gap between
them is itself worth seeing.

AN EPISODE ENDS AT ITS OWN TRADING BAR'S CLOSE

Whatever else happens -- the 15m close for a 15m stack, the 5m close for a
5m one. That is how the trade is taken, so that is where it stops.

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
from liqmap import cascade                                  # noqa: E402

TICK = 0.25
FRAMES = (("1m", 60.0), ("5m", 300.0), ("15m", 900.0))
SPAN_OF = dict(FRAMES)
TRADE_FRAME_S = 900.0                  # the longest bar a trade is held in

# The stacks measured, each with the bar its trade is held in. The rows
# under each come from `cascade.STACKS`, not from anything chosen here.
TRADED = ("15m", "5m")

RULES = ("cascade", "open")

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

    Holds the open of the bar that is forming and the signed size traded
    inside it. Both are known from the prints that have already happened,
    which is the whole reason this can be read live.
    """

    def __init__(self, seconds: float) -> None:
        self.span = seconds
        self.bar_ts: float | None = None
        self.open_px = 0.0
        self.delta = 0.0

    def feed(self, ts: float, px: float, signed: float = 0.0) -> bool:
        """Returns True when this print opened a new bar.

        The delta RESETS with the bar. Carrying it across would make a
        row's colour a statement about the previous bar.
        """
        slot = math.floor(ts / self.span) * self.span
        new = self.bar_ts is None or slot != self.bar_ts
        if new:
            self.bar_ts = slot
            self.open_px = px
            self.delta = 0.0
        self.delta += signed
        return new

    def way(self, px: float) -> str:
        if self.bar_ts is None:
            return ""
        if px > self.open_px:
            return "up"
        if px < self.open_px:
            return "down"
        return "flat"

    def bps(self, px: float) -> float:
        if self.bar_ts is None or self.open_px <= 0:
            return 0.0
        return (px - self.open_px) / self.open_px * 10_000.0

    def colour(self) -> str:
        """Green when the aggressors were net buyers in this bar.

        `cascade.colour_of` maps a winning side to a colour; the winning
        side here is the sign of the bar's own delta.
        """
        if self.bar_ts is None or self.delta == 0.0:
            return "none"
        return cascade.colour_of("buyers" if self.delta > 0 else "sellers")


# ------------------------------------------------------------- the rules

def way_open(frames: dict, traded: str, px: float) -> str:
    """Every row in the stack above or below its own bar's open."""
    want = cascade.STACKS.get(traded)
    if not want:
        return ""
    ways = [frames[tf].way(px) for tf in want]
    if any(w in ("", "flat") for w in ways):
        return ""
    first = ways[0]
    return first if all(w == first for w in ways) else ""


def way_cascade(frames: dict, traded: str, px: float) -> str:
    """`cascade.read` exactly as the site runs it.

    A row counts only when its colour and its bps point the same way, and
    the stack counts as agreed only in TREND or REVERSAL -- which is what
    `cascade.aligned_way` returns and nothing else does.
    """
    want = cascade.STACKS.get(traded)
    if not want:
        return ""
    rows = {}
    for tf in want:
        f = frames[tf]
        if f.bar_ts is None:
            return ""
        rows[tf] = (f.colour(), f.bps(px))
    return cascade.aligned_way(cascade.read(traded, rows)) or ""


def rows_open(frames: dict, traded: str, px: float) -> list[tuple[str, str]]:
    """(timeframe, way) per row, with '' and 'flat' meaning no opinion."""
    return [(tf, frames[tf].way(px))
            for tf in cascade.STACKS.get(traded, ())]


def rows_cascade(frames: dict, traded: str,
                 px: float) -> list[tuple[str, str]]:
    """(timeframe, qualified way) per row, using cascade's own Rung.

    A row whose colour fights its own bps is thrown out rather than
    counted weakly -- that is `Rung.qualified`, not a reimplementation of
    it, so the rule here cannot drift from the rule on the site.
    """
    out = []
    for tf in cascade.STACKS.get(traded, ()):
        f = frames[tf]
        if f.bar_ts is None:
            out.append((tf, ""))
            continue
        rung = cascade.Rung(timeframe=tf, colour=f.colour(),
                            bps=f.bps(px), present=True)
        out.append((tf, rung.qualified or ""))
    return out


WAY_FN = {"open": way_open, "cascade": way_cascade}
ROWS_FN = {"open": rows_open, "cascade": rows_cascade}


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
    bar15_ts: float                  # the trading bar it sits in
    offset_s: float                  # how far into that bar it formed
    rule: str = ""                   # open | cascade
    traded: str = ""                 # 15m | 5m

    end_ts: float = 0.0
    end_px: float = 0.0
    ended: str = ""                  # flip | bar_close | file_end
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

class Detector:
    """One (rule, traded frame) pair, keeping its own live episode.

    Several of these run off ONE pass over the tape. Four passes would be
    four readings of the same file taken minutes apart in wall-clock time
    and, more to the point, four chances for the rules to end up measured
    against subtly different prints.
    """

    def __init__(self, rule: str, traded: str) -> None:
        self.rule = rule
        self.traded = traded
        self.fn = WAY_FN[rule]
        self.rows_fn = ROWS_FN[rule]
        self.live: Episode | None = None
        self.episodes: list[Episode] = []

    def feed(self, frames: dict, ts: float, px: float,
             bar_closed: bool) -> None:
        live = self.live

        # THE TRADING BAR'S CLOSE ENDS IT, whatever the rows say.
        if live is not None and bar_closed:
            self._close(live, ts, px, "bar_close")
            live = self.live = None

        now = self.fn(frames, self.traded, px)

        if live is None:
            if now:
                self.live = self._begin(frames, ts, px, now)
            return

        live.observe(px)
        rows = self.rows_fn(frames, self.traded, px)

        # AGREEMENT LAPSING is recorded but does not end it -- at the
        # first print of any new bar that row sits on its own open and
        # reads flat, and ending there would measure the bar grid.
        if live.first_break == "" and now != live.way:
            live.first_break = next(
                (tf for tf, w in rows if w != live.way), "(stack)")
            live.break_ts = ts
            live.break_ticks = ((px - live.start_px) / TICK
                                if live.way == "up"
                                else (live.start_px - px) / TICK)

        # ONE ROW TURNING OPPOSITE ENDS IT. That is the question as it was
        # asked -- how far before ONE disagrees -- so it is not the whole
        # stack agreeing the other way, which would be a far later and far
        # rarer event.
        other = "down" if live.way == "up" else "up"
        against = [tf for tf, w in rows if w == other]
        if against:
            self._close(live, ts, px, f"flip_{against[0]}")
            self.live = self._begin(frames, ts, px, now) if now else None

    def _begin(self, frames: dict, ts: float, px: float,
               way: str) -> Episode:
        """Dated against the bar its trade is held in.

        `bar_ts or ts` was wrong here: a slot of 0.0 is falsy, so a file
        timestamped from the epoch dated every episode against its own
        start and reported an offset of zero for all of them. The same
        falsy zero has now cost this project three times.
        """
        f = frames[self.traded]
        b = f.bar_ts if f.bar_ts is not None else ts
        ep = Episode(start_ts=ts, start_px=px, way=way, bar15_ts=b,
                     offset_s=ts - b, rule=self.rule, traded=self.traded)
        ep.observe(px)
        return ep

    def _close(self, ep: Episode, ts: float, px: float, why: str) -> None:
        ep.end_ts, ep.end_px, ep.ended = ts, px, why
        self.episodes.append(ep)

    def finish(self, ts: float, px: float) -> None:
        if self.live is not None:
            self._close(self.live, ts, px, "file_end")
            self.live = None


def walk(records, symbol: str = "ES", flip: bool = False,
         progress: Progress | None = None,
         rules=RULES, traded=TRADED):
    """ONE pass, every rule and frame at once.

    Returns (detectors, tape, prints, span_s).
    """
    frames = {name: Frame(secs) for name, secs in FRAMES}
    dets = [Detector(r, t) for r in rules for t in traded
            if t in cascade.STACKS]
    tape = Tape()
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
            progress.extra = f"{sum(len(d.episodes) for d in dets):,} eps"
            progress.tick()
        if first_ts is None:
            first_ts = ts
        last_ts, last_px = ts, px
        tape.add(ts, px)

        signed = item.sz if item.aggressor == "buy" else -item.sz
        opened = {name: f.feed(ts, px, signed) for name, f in frames.items()}

        for d in dets:
            d.feed(frames, ts, px, opened.get(d.traded, False))

    for d in dets:
        d.finish(last_ts, last_px)
    if progress is not None:
        progress.done()
    return dets, tape, n, max(0.0, last_ts - (first_ts or 0.0))


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


def report_block(eps, tape, rng, rule: str, traded: str,
                 span_s: float) -> None:
    """Everything about one (rule, frame) pair."""
    title = f"RULE = {rule.upper()}   TRADED ON {traded}   " \
            f"stack {'/'.join(cascade.STACKS.get(traded, ()))}"
    print("=" * 78)
    print(title)
    print("=" * 78)
    if not eps:
        print("  no agreement episodes at all")
        print()
        return

    bars = len({e.bar15_ts for e in eps})
    print(f"  {len(eps):,} episodes, {bars} {traded} bars with any "
          f"agreement ({len(eps) / max(1, bars):.1f} per bar)")
    print()

    tot = len(eps)
    by_end: dict[str, int] = {}
    for e in eps:
        by_end[e.ended] = by_end.get(e.ended, 0) + 1
    print("  WHAT ENDED THEM")
    for why, k in sorted(by_end.items(), key=lambda kv: -kv[1]):
        print(f"    {why:<12} {k:>7,}  {k / tot:5.1%}")

    by_br: dict[str, int] = {}
    for e in eps:
        key = e.first_break or "(never)"
        by_br[key] = by_br.get(key, 0) + 1
    print("  WHICH ROW LAPSED FIRST  (flat counts here; does not end it)")
    for who, k in sorted(by_br.items(), key=lambda kv: -kv[1]):
        print(f"    {who:<12} {k:>7,}  {k / tot:5.1%}")
    brk = [e.break_ticks for e in eps if e.break_ticks is not None]
    if brk:
        print(f"    travel at that lapse: p50 {_median(brk):.1f}t  "
              f"p75 {_pct(brk, .75):.1f}t  p90 {_pct(brk, .90):.1f}t")
    print()

    print("  TRAVEL WHILE AGREED  (ticks, from the price at agreement)")
    spread(eps, "all")
    spread([e for e in eps if e.way == "up"], "agreed up")
    spread([e for e in eps if e.way == "down"], "agreed down")
    print()

    print("  TICK PROJECTION -- episodes reaching each distance.")
    print("  An episode is long BECAUSE price kept going, so this table")
    print("  selects for travel and you do not know the length when you")
    print("  enter. The fixed-horizon table under it is the honest one.")
    projection_header()
    projection(eps, "all episodes", tape, rng)
    print()

    print("  FIXED HORIZON from the agreement, ignoring the stopping rule")
    projection_header()
    horizon_table(eps, tape, rng, "all")
    print()

    print("  BY WHERE IT FORMED IN THE BAR")
    projection_header()
    span = SPAN_OF.get(traded, TRADE_FRAME_S)
    edges = [span * i / 5.0 for i in range(6)]
    for lo, hi in zip(edges, edges[1:]):
        sub = [e for e in eps if lo <= e.offset_s < hi]
        projection(sub, f"{lo / 60:.1f}-{hi / 60:.1f} min in", tape, rng)
    print()

    per: dict[float, list[Episode]] = {}
    for e in eps:
        per.setdefault(e.bar15_ts, []).append(e)
    counts = [len(v) for v in per.values()]
    firsts = [v[0] for v in per.values()]
    print(f"  PER {traded} BAR -- episodes per bar: p50 {_median(counts)}, "
          f"p90 {_pct(counts, .90)}, max {max(counts)}")
    spread(firsts, "first of bar")
    projection_header()
    projection(firsts, "first of bar", tape, rng)
    print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path")
    ap.add_argument("--symbol", default="ES")
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N records -- a trial run, not a result")
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--rules", default=",".join(RULES))
    ap.add_argument("--traded", default=",".join(TRADED))
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    rules = [r.strip() for r in args.rules.split(",") if r.strip()]
    traded = [t.strip() for t in args.traded.split(",") if t.strip()]
    for r in rules:
        if r not in WAY_FN:
            print(f"unknown rule: {r}", file=sys.stderr)
            return 2

    print(f"file      {args.path}")
    print(f"rules     " + ", ".join(rules))
    print(f"traded    " + ", ".join(
        f"{t} over {'/'.join(cascade.STACKS.get(t, ()))}" for t in traded))
    print(f"episode   starts when the stack agrees, ends on a FLIP the "
          f"other way or at the trading bar's close")
    print("progress on stderr; ONE pass drives every block below")
    if args.limit:
        print(f"LIMIT     {args.limit:,} records -- a trial run, not a "
              f"result.")
    print()

    dets, tape, prints, span = walk(
        capped(dbn.open_file(args.path), args.limit),
        symbol=args.symbol, flip=args.flip, progress=Progress("walk"),
        rules=rules, traded=traded)

    print()
    print(f"{prints:,} prints over {span / 3600.0:.2f} h")
    print()
    for d in dets:
        report_block(d.episodes, tape, rng, d.rule, d.traded, span)

    print("=" * 78)
    print("SIDE BY SIDE -- fixed horizon, the share reaching 16 ticks "
          "(4 points)")
    print("=" * 78)
    print(f"  {'rule / frame':<22}" + "".join(f"{h:>9.0f}s"
                                              for h in HORIZONS_S))
    for d in dets:
        if not d.episodes:
            continue
        bits = [f"  {d.rule + ' / ' + d.traded:<22}"]
        for h in HORIZONS_S:
            got = []
            for e in d.episodes:
                tv = tape.travel(tape.index_at(e.start_ts), h)
                if tv is None:
                    continue
                got.append(tv[0] if e.way == "up" else tv[1])
            if len(got) < MIN_GROUP:
                bits.append(f"{len(got):>8}n")
            else:
                hit = sum(1 for g in got if g >= 16) / len(got)
                bits.append(f"{hit:>9.1%}")
        print("".join(bits))
    print(f"  {'(random start null)':<22}", end="")
    for h in HORIZONS_S:
        c = control_rate(tape, [h] * 400, 16, rng)
        print("        -" if c is None else f"{c:>9.1%}", end="")
    print()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
