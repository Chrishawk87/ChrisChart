"""Does the candle read's lean have any relationship to the candle's close?

WHAT THIS MEASURES, AND WHAT IT CANNOT

`candleread.read` combines flow, the book, absorption, position in range,
VWAP and higher-timeframe structure into one lean. Its own docstring is
blunt about the weights: "reasonable, argued for in the comments, and
unvalidated". This is the measurement.

It answers one question -- when the read leans up, does the candle finish
up more often than the base rate -- and it answers it from a stream of
book and trade records, which is the only data that carries flow and
absorption at all.

THREE WAYS THIS KIND OF STUDY LIES, AND WHAT IS DONE ABOUT EACH

1. OVERLAP. Sampling every few seconds inside a 15-minute candle and
   scoring every sample against that candle's close gives hundreds of
   "observations" that share one outcome. The interval computed over them
   is far too tight. So a read is taken at a FIXED elapsed fraction, once
   per candle per fraction, and each fraction is reported separately.
   Within one fraction the observations are one-per-candle and the count
   in the table is a count of candles.

2. THE BASE RATE. If 54% of candles close green, a read that always says
   "up" scores 54% and has found nothing. Accuracy is therefore always
   printed next to the base rate, and the control below is the real test.

3. THE CONTROL. The leans are permuted across candles, keeping how often
   the read says up, down and flat exactly as observed, and keeping the
   candles exactly as they were. Anything the real ordering does not beat
   is the mix of leans meeting the mix of outcomes.

WHAT IT STILL CANNOT FIX

Two days of ES is a couple of hundred candles. The intervals will be
wide enough to drive a bus through, and that is a fact about the data,
not a flaw in the arithmetic. This module reports the width rather than
hiding it, and `enough()` says plainly when a result is too thin to read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from . import candleread as cr
from .flow import Absorption, Book, FlowTape, LevelWatch, Trade

# Where inside the candle the reads are taken, as a share of its length.
# Early, middle and late: the module claims confidence scales with elapsed
# time, so the three are reported apart rather than pooled.
FRACTIONS = (0.25, 0.5, 0.75)

# Below this many candles in a bucket, a hit rate is not worth printing.
MIN_CANDLES = 25

# The absorption watcher is re-centred when price leaves its band, because
# a level watched from two hours ago is not where the market is now.
BAND_BPS = 10.0
RECENTRE_BPS = 25.0


@dataclass
class Observation:
    """One read, and what the candle it was taken in went on to do."""

    candle_start: float
    fraction: float
    lean: str                 # up | down | flat
    score: float
    agreement: float
    signals: int
    went_up: bool            # the CANDLE closed up
    change_ticks: float      # open to close, in ticks
    read_px: float = 0.0     # price at the moment of the read
    forward_ticks: float = 0.0   # read price to the candle's close

    @property
    def forward_up(self) -> bool:
        return self.forward_ticks > 0

    @property
    def hit(self) -> bool:
        """Did price move the read's way AFTER the read.

        NOT whether the candle closed in the direction it was already
        going. At 50% into a candle that is up ten ticks, a read saying
        "up" is right about the close most of the time and has told you
        nothing you could trade -- the move it is describing has already
        happened. On a pure random walk that version of the question
        scores well above its own control, which is how this was caught.
        """
        return ((self.lean == "up" and self.forward_ticks > 0)
                or (self.lean == "down" and self.forward_ticks < 0))

    @property
    def candle_hit(self) -> bool:
        """The weaker question, kept for comparison: did the candle close
        the read's way, whenever it got there."""
        return ((self.lean == "up" and self.went_up)
                or (self.lean == "down" and not self.went_up))


@dataclass
class Forming:
    start: float
    open: float
    high: float
    low: float
    last: float

    def add(self, px: float) -> None:
        self.high = max(self.high, px)
        self.low = min(self.low, px)
        self.last = px


def study(events: Iterable[tuple[str, object]], coin: str = "ES",
          interval_s: float = 900.0,
          fractions: Sequence[float] = FRACTIONS,
          tick: float = 0.25) -> list[Observation]:
    """Walk a stream of ("book", Book) / ("trade", Trade) and read as it goes.

    Everything is causal by construction: the read at 25% of a candle sees
    only trades and books that arrived before that moment, and the outcome
    is attached afterwards from the candle's close.
    """
    tape = FlowTape(max_age=3600.0)
    watch: LevelWatch | None = None
    book: Book | None = None

    cur: Forming | None = None
    taken: set[float] = set()
    pending: list[Observation] = []
    out: list[Observation] = []

    def close_candle() -> None:
        if cur is None:
            return
        up = cur.last > cur.open
        move = (cur.last - cur.open) / tick if tick > 0 else 0.0
        for o in pending:
            o.went_up = up
            o.change_ticks = move
            # The tradeable quantity: what price did from the read to the
            # close, not what the candle did from its own open.
            o.forward_ticks = ((cur.last - o.read_px) / tick
                               if tick > 0 else 0.0)
            out.append(o)
        pending.clear()

    for kind, obj in events:
        if kind == "book":
            book = obj  # type: ignore[assignment]
            if watch is not None:
                watch.on_book(obj)  # type: ignore[arg-type]
            continue
        if kind != "trade":
            continue

        t: Trade = obj  # type: ignore[assignment]
        ts, px = float(t.ts), float(t.px)
        tape.add(t)

        # The absorption watcher is anchored to a level. Re-centre it when
        # price has walked away from the band -- a level from two hours
        # ago is not where the market is now, and its baseline describes
        # a different piece of tape.
        if watch is None or abs(px - watch.level) / px * 10_000.0 > \
                RECENTRE_BPS:
            watch = LevelWatch(coin, px, band_bps=BAND_BPS)
        watch.on_trade(t)

        start = ts - (ts % interval_s)
        if cur is None or start != cur.start:
            close_candle()
            cur = Forming(start, px, px, px, px)
            taken = set()
        else:
            cur.add(px)

        elapsed = ts - cur.start
        frac = elapsed / interval_s if interval_s > 0 else 1.0
        for f in fractions:
            if f in taken or frac < f:
                continue
            taken.add(f)
            absorb: Absorption | None = None
            try:
                absorb = watch.absorption(now=ts)
            except Exception:
                absorb = None
            r = cr.read(coin, interval_s, elapsed,
                        cur.open, cur.high, cur.low, cur.last,
                        tape=tape, book=book, absorption=absorb)
            pending.append(Observation(
                candle_start=cur.start, fraction=f, lean=r.lean,
                score=r.score, agreement=r.agreement,
                signals=len([s for s in r.signals
                             if s.direction != "flat"]),
                went_up=False, change_ticks=0.0, read_px=cur.last))

    close_candle()
    return out


# ---------------------------------------------------------------- scoring

def enough(n: int) -> bool:
    return n >= MIN_CANDLES


def base_rate(obs: Sequence[Observation]) -> float:
    """Share of DISTINCT candles that closed up.

    Distinct, because the same candle appears once per fraction and
    counting it three times would make the base rate look better measured
    than it is.
    """
    seen: dict[float, bool] = {}
    for o in obs:
        seen[o.candle_start] = o.went_up
    return sum(seen.values()) / len(seen) if seen else 0.0


def forward_rate(obs: Sequence[Observation]) -> float:
    """Share of reads after which price rose, whatever the read said.

    The base rate for the question that matters. If price rose after 55%
    of reads, a read that always says up scores 55% on the forward
    question too.
    """
    if not obs:
        return 0.0
    return sum(1 for o in obs if o.forward_up) / len(obs)


def by_lean(obs: Sequence[Observation]) -> dict[str, dict]:
    """Accuracy per lean, at one fraction. One row per lean bucket."""
    from .score import wilson

    groups: dict[str, list[Observation]] = {}
    for o in obs:
        groups.setdefault(o.lean, []).append(o)

    out: dict[str, dict] = {}
    for lean, rows in sorted(groups.items()):
        if lean == "flat":
            up = sum(1 for r in rows if r.went_up)
            out[lean] = {"n": len(rows), "went_up": up,
                         "rate": round(up / len(rows), 4) if rows else 0.0,
                         "ci_low": 0.0, "ci_high": 0.0,
                         "enough": enough(len(rows)),
                         "avg_ticks": round(
                             sum(r.forward_ticks for r in rows)
                             / len(rows), 2)}
            continue
        hits = sum(1 for r in rows if r.hit)
        lo, hi = wilson(hits, len(rows))
        # Signed by the lean, so an "up" read that precedes a fall shows a
        # negative average rather than a flattering absolute one.
        avg = sum((r.forward_ticks if lean == "up" else -r.forward_ticks)
                  for r in rows) / len(rows)
        cavg = sum((r.change_ticks if lean == "up" else -r.change_ticks)
                   for r in rows) / len(rows)
        chits = sum(1 for r in rows if r.candle_hit)
        out[lean] = {"n": len(rows), "hits": hits,
                     "rate": round(hits / len(rows), 4),
                     "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                     "enough": enough(len(rows)),
                     "avg_ticks": round(avg, 2),
                     "candle_rate": round(chits / len(rows), 4),
                     "candle_avg_ticks": round(cavg, 2)}
    return out


def control(obs: Sequence[Observation], runs: int = 400,
            seed: int = 11) -> dict:
    """Shuffle which candle got which lean, many times over.

    Keeps how often the read says up, down and flat EXACTLY as observed,
    and keeps the candles exactly as they were. What survives is the mix
    of leans meeting the mix of outcomes -- which is what a read with no
    information scores, and it is not 50%.
    """
    import random

    directional = [o for o in obs if o.lean != "flat"]
    if len(directional) < MIN_CANDLES:
        return {"runs": 0}
    leans = [o.lean for o in directional]
    ups = [o.forward_up for o in directional]

    rng = random.Random(seed)
    rates: list[float] = []
    for _ in range(runs):
        rng.shuffle(leans)
        hits = sum(1 for lean, up in zip(leans, ups)
                   if (lean == "up") == up)
        rates.append(hits / len(leans))

    mean = sum(rates) / len(rates)
    var = sum((r - mean) ** 2 for r in rates) / max(1, len(rates) - 1)
    return {"runs": len(rates), "mean": round(mean, 4),
            "sd": round(var ** 0.5, 4),
            "p95": round(sorted(rates)[int(len(rates) * 0.95)], 4)}


def beats(rate: float, n: int, ctrl: dict, sds: float = 2.0) -> bool:
    if not ctrl.get("runs") or not enough(n):
        return False
    sd = ctrl.get("sd") or 0.0
    if sd <= 0:
        return rate > ctrl["mean"]
    return rate >= ctrl["mean"] + sds * sd
