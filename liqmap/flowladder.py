"""Delta, CVD and velocity on every timeframe, plus the book at the touch.

WHAT THIS IS FOR

Two different jobs, from two different feeds, read together:

    THE BOOK says whether the trade is worth taking at all -- how wide
    the spread is against the target, and how much size is in front of a
    resting order. That is an economics question and it is answered
    before any signal.

    THE TAPE says whether what is happening is real -- delta that is
    actually printing, at a velocity above what this market has been
    doing lately. Resting size can be pulled the moment you lean on it;
    a print cannot be taken back.

So the book gates and the tape triggers, and neither is asked to do the
other's job.

WHY DELTA IS PER TIMEFRAME AND VELOCITY IS NOT

Delta is a quantity accumulated over a bar, so every timeframe has its
own and they can disagree -- that disagreement is most of the value. The
fifteen can be positive while the one is already negative, and that is
the turn.

Velocity is a rate measured right now. There is no 4-hour version of
"prints per second in the last fifteen seconds"; there is one number and
it belongs to the moment, not to a bar. Reporting it per timeframe would
be printing the same figure five times and implying five measurements.

WHAT THE TAPE CANNOT COVER

The rolling tape holds an hour. A one-minute rung is built from sixty
bars of it; a four-hour rung has seen a quarter of one bar. Each rung
therefore reports how much of itself the tape actually covers, and a
rung with almost no coverage says so rather than printing a delta that
describes fifteen minutes of a four-hour candle as though it were the
candle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from . import orderflow as of

# How many completed bars of each timeframe to accumulate, where the tape
# reaches that far. Enough to show a turn, short enough that a rung is
# about now rather than about this morning.
RUNGS_BACK = 6

# A rung whose tape coverage is below this says so instead of reporting a
# delta that describes a fraction of its own bar.
MIN_COVERAGE = 0.25


@dataclass
class Rung:
    """One timeframe's order flow, as far as the tape can see it."""

    timeframe: str
    interval_s: float

    buy: float = 0.0            # notional, current bar
    sell: float = 0.0
    trades: int = 0

    cvd: float = 0.0            # accumulated over the bars the tape covers
    bars: int = 0               # completed bars contributing to that cvd
    coverage: float = 0.0       # share of the CURRENT bar the tape has seen
    covered: bool = False

    @property
    def delta(self) -> float:
        return self.buy - self.sell

    @property
    def total(self) -> float:
        return self.buy + self.sell

    @property
    def lean(self) -> float:
        """-1 all selling, +1 all buying."""
        return self.delta / self.total if self.total > 0 else 0.0

    @property
    def side(self) -> str:
        if not self.covered or self.total <= 0:
            return "unknown"
        return "buyers" if self.delta > 0 else \
            "sellers" if self.delta < 0 else "even"

    def to_dict(self) -> dict:
        return {"timeframe": self.timeframe,
                "interval_s": self.interval_s,
                "buy": round(self.buy, 2), "sell": round(self.sell, 2),
                "delta": round(self.delta, 2), "lean": round(self.lean, 4),
                "cvd": round(self.cvd, 2), "bars": self.bars,
                "trades": self.trades,
                "coverage": round(self.coverage, 3),
                "covered": self.covered, "side": self.side}


@dataclass
class BookRead:
    """The book at the touch, and what it costs to cross it."""

    bid: float = 0.0
    ask: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    spread_ticks: float = 0.0
    ok: bool = False            # tight enough for the target
    why: str = ""

    def to_dict(self) -> dict:
        return {"bid": round(self.bid, 6), "ask": round(self.ask, 6),
                "bid_size": round(self.bid_size, 2),
                "ask_size": round(self.ask_size, 2),
                "spread_ticks": round(self.spread_ticks, 2),
                "ok": self.ok, "why": self.why}


@dataclass
class Ladder:
    """Everything the two feeds say, assembled. No trade in it."""

    rungs: list[Rung] = field(default_factory=list)
    velocity: of.Velocity = field(default_factory=of.Velocity)
    book: BookRead = field(default_factory=BookRead)
    target_ticks: float = 10.0
    aligned: str = "mixed"      # buyers | sellers | mixed | unknown
    tradeable: bool = False
    why: str = ""

    def to_dict(self) -> dict:
        return {"rungs": [r.to_dict() for r in self.rungs],
                "velocity": self.velocity.to_dict(),
                "book": self.book.to_dict(),
                "target_ticks": self.target_ticks,
                "aligned": self.aligned,
                "tradeable": self.tradeable, "why": self.why}


def read_book(book, spec: of.Spec, target_ticks: float) -> BookRead:
    """The touch, and whether the spread leaves room for the target."""
    out = BookRead()
    if book is None or getattr(book, "empty", True):
        out.why = "no book"
        return out
    if not book.bids or not book.asks:
        out.why = "one-sided book"
        return out

    out.bid = float(book.bids[0].px)
    out.ask = float(book.asks[0].px)
    out.bid_size = float(book.bids[0].sz)
    out.ask_size = float(book.asks[0].sz)
    tick = spec.tick_size if spec.tick_size > 0 else 0.25
    out.spread_ticks = (out.ask - out.bid) / tick

    out.ok = of.spread_ok(out.spread_ticks, target_ticks)
    share = (out.spread_ticks / target_ticks) if target_ticks > 0 else 1.0
    out.why = (f"{out.spread_ticks:.1f} tick spread against a "
               f"{target_ticks:g} tick target -- {share:.0%} of it"
               + ("" if out.ok else ", too wide"))
    return out


def rungs(tape, intervals: Sequence[tuple[str, float]],
          now: float | None = None, back: int = RUNGS_BACK) -> list[Rung]:
    """Delta and CVD per timeframe, built from the rolling tape.

    `intervals` are (name, seconds) pairs. Each rung reports the CURRENT
    bar's delta and a CVD accumulated over the completed bars the tape
    still holds -- with the coverage it had, so a rung the tape cannot
    see is not mistaken for a flat one.
    """
    out: list[Rung] = []
    trades = list(getattr(tape, "_trades", []) or [])
    if not trades:
        return [Rung(timeframe=n, interval_s=s) for n, s in intervals]

    end = float(now) if now is not None else float(trades[-1].ts)
    first = float(trades[0].ts)
    held = max(0.0, end - first)

    for name, iv in intervals:
        r = Rung(timeframe=name, interval_s=iv)
        if iv <= 0:
            out.append(r)
            continue

        start = end - (end % iv)
        r.coverage = min(1.0, held / iv) if iv > 0 else 0.0
        r.covered = (end - start) <= held and r.coverage >= MIN_COVERAGE

        for t in trades:
            ts = float(t.ts)
            if ts < start:
                continue
            if t.aggressor == "buy":
                r.buy += float(t.notional)
            else:
                r.sell += float(t.notional)
            r.trades += 1

        # CVD over whichever completed bars the tape still reaches.
        floor = start - back * iv
        seen: set[float] = set()
        for t in trades:
            ts = float(t.ts)
            if ts < max(floor, first) or ts >= start:
                continue
            seen.add(ts - (ts % iv))
            r.cvd += (float(t.notional) if t.aggressor == "buy"
                      else -float(t.notional))
        r.bars = len(seen)
        r.cvd += r.delta          # include the bar in progress
        out.append(r)
    return out


def agreement(rungs_: Sequence[Rung]) -> str:
    """Do the covered timeframes all lean the same way.

    Only the covered ones vote. A rung the tape cannot see has no opinion,
    and counting it as "even" would let silence outvote the rungs that
    actually measured something.
    """
    sides = {r.side for r in rungs_ if r.covered and r.side != "unknown"}
    if not sides:
        return "unknown"
    if sides == {"buyers"}:
        return "buyers"
    if sides == {"sellers"}:
        return "sellers"
    return "mixed"


def build(tape, book, intervals: Sequence[tuple[str, float]],
          spec: of.Spec, target_ticks: float = 10.0,
          now: float | None = None, spike: float = 2.0) -> Ladder:
    """The whole read: book gate, tape trigger, timeframe agreement.

    `tradeable` means all three said yes. It is not a direction and not a
    suggestion -- it is the statement that nothing is currently refusing.
    """
    out = Ladder(target_ticks=target_ticks)
    out.rungs = rungs(tape, intervals, now=now)
    stamps = [float(t.ts) for t in getattr(tape, "_trades", []) or []]
    out.velocity = of.velocity(stamps, now=now)
    out.book = read_book(book, spec, target_ticks)
    out.aligned = agreement(out.rungs)

    refusals: list[str] = []
    if not out.book.ok:
        refusals.append(out.book.why)
    if not out.velocity.spiking(spike):
        refusals.append(
            f"tape at {out.velocity.ratio:.1f}x its baseline, wants "
            f"{spike:g}x" if out.velocity.confident else out.velocity.note)
    if out.aligned in ("mixed", "unknown"):
        refusals.append(f"timeframes {out.aligned}")

    out.tradeable = not refusals
    out.why = ("; ".join(refusals) if refusals
               else f"spread, tape and every covered timeframe agree "
                    f"({out.aligned})")
    return out
