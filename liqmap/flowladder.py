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

THE TWO VELOCITIES, AND WHY THEY ARE NOT THE SAME NUMBER

Delta is a quantity accumulated over a bar, so every timeframe has its
own and they can disagree -- that disagreement is most of the value. The
fifteen can be positive while the one is already negative, and that is
the turn.

Velocity splits in two, and the split matters. The LADDER's velocity is
the market right now: fifteen seconds against ten minutes, one number,
belonging to the moment rather than to any bar. Printing that same
figure on every rung would imply five measurements where there is one.

Each RUNG's pace is a different measurement with the same units: the
bar in progress against the completed bars of its own timeframe, which
is the shape the delta and CVD beside it already have. A tape running
3x against ten minutes can still leave the current 4-hour bar slower
than the four before it, and a rung that cannot see two completed bars
of itself reports no pace at all rather than a ratio built from
whatever fraction is in memory.

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

# Covered rungs needed before "they agree" means anything.
#
# One rung agreeing with itself is not a consensus, and a panel that says
# "every covered timeframe agrees" over a single measured row is making a
# claim about a stack it cannot see. On a freshly connected feed that is
# the normal state for the first few minutes, so it has to be said rather
# than papered over.
MIN_VOTERS = 2


# The rows the panel asks for when it asks for nothing in particular.
DEFAULT_TIMEFRAMES = ("1m", "5m", "15m", "30m", "1h", "4h")


def wanted(asked: Sequence[str], intervals: dict,
           default: Sequence[str] = DEFAULT_TIMEFRAMES
           ) -> list[tuple[str, float]]:
    """The (name, seconds) pairs to build rungs for.

    The panel draws one row per timeframe with the candle read on one
    side and the flow on the other, so it asks for rungs BY NAME rather
    than taking whatever list this module felt like. A row pairing a
    15-minute read with a 5-minute delta would be worse than no row,
    and two lists that happen to be in the same order today are not a
    guarantee that they will be tomorrow.

    Names this feed does not have are dropped rather than guessed at,
    duplicates collapse, and an empty ask gets the default set -- an
    empty list is a caller with no opinion, not a caller asking for
    nothing.
    """
    names = [a.strip() for a in asked if str(a).strip()] or list(default)
    seen: set[str] = set()
    out: list[tuple[str, float]] = []
    for n in names:
        if n in intervals and n not in seen:
            seen.add(n)
            out.append((n, float(intervals[n])))
    return out


@dataclass
class Rung:
    """One timeframe's order flow, as far as the tape can see it."""

    timeframe: str
    interval_s: float

    buy: float = 0.0            # notional, current bar
    sell: float = 0.0
    trades: int = 0

    cvd: float = 0.0            # accumulated over the bars behind this one
    bars: int = 0               # completed bars contributing to that cvd
    asked: int = 0              # completed bars wanted
    cvd_source: str = "tape"    # tape | stored | memory | both | none
    coverage: float = 0.0       # share of the CURRENT bar the tape has seen
    covered: bool = False       # the tape actually measured this one

    # This bar's print rate against the completed bars of THIS timeframe.
    # Not the ladder's velocity, which is the market in the last fifteen
    # seconds; see the module docstring for why they differ.
    pace: of.Velocity = field(default_factory=of.Velocity)

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
                "asked": self.asked, "cvd_source": self.cvd_source,
                "cvd_complete": self.bars >= self.asked > 0,
                "trades": self.trades,
                "coverage": round(self.coverage, 3),
                "covered": self.covered, "side": self.side,
                "pace": self.pace.to_dict()}


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
          now: float | None = None, back: int = RUNGS_BACK,
          cvd_for=None) -> list[Rung]:
    """Delta and CVD per timeframe.

    `intervals` are (name, seconds) pairs. Each rung reports the CURRENT
    bar's delta, counted off the tape, and a CVD over the completed bars
    behind it -- with the coverage it had, so a rung the tape cannot see is
    not mistaken for a flat one.

    `cvd_for(timeframe, interval_s, bar_open, bars)` supplies the completed
    bars' CVD from wherever they are kept. WITHOUT IT THE CVD IS CAPPED BY
    THE TAPE, which holds about an hour: a 4-hour rung then has no
    completed bars to add up and reports none rather than inventing one
    from the fragment it can see. That is the whole reason the closed-bar
    store exists, and why the slow rows used to sit at NOT COVERED forever.
    """
    out: list[Rung] = []
    trades = list(getattr(tape, "_trades", []) or [])
    if not trades:
        return [Rung(timeframe=n, interval_s=s) for n, s in intervals]

    end = float(now) if now is not None else float(trades[-1].ts)
    first = float(trades[0].ts)
    held = max(0.0, end - first)
    stamps = [float(t.ts) for t in trades]

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

        r.asked = int(max(1, back))
        kept = cvd_for(name, iv, start, r.asked) if cvd_for else None
        if kept is not None:
            # Completed bars as they were counted when they closed, which
            # reaches back further than the tape does and survives a
            # restart. The bar in progress is still the tape's.
            r.cvd = float(kept.cvd) + r.delta
            r.bars = int(kept.bars)
            r.cvd_source = kept.source
        else:
            # Fallback: whatever completed bars the tape itself still
            # reaches. Fine for a minute, nothing at all for four hours.
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
            r.cvd += r.delta      # include the bar in progress
            r.cvd_source = "tape"

        # Its own pace, on its own bars. Measured even on an uncovered
        # rung, because the two refusals are different: coverage is about
        # the bar in progress, pace is about the bars behind it, and a
        # rung can have one without the other.
        r.pace = of.velocity_for(stamps, iv, now=end, back=back)
        out.append(r)
    return out


def voters(rungs_: Sequence[Rung]) -> list[Rung]:
    """The rungs entitled to an opinion: the MEASURED ones.

    A rung the rolling tape has not held long enough to fill has no
    opinion. It is shown, because a blank row tells you the feed is still
    warming up, but it does not vote.
    """
    return [r for r in rungs_ if r.covered and r.side != "unknown"]


def agreement(rungs_: Sequence[Rung]) -> str:
    """Do the measured timeframes all lean the same way.

    Only the covered ones vote. A rung the tape cannot see has no opinion,
    and counting it as "even" would let silence outvote the rungs that
    actually measured something.
    """
    vs = voters(rungs_)
    if len(vs) < MIN_VOTERS:
        return "unknown"
    sides = {r.side for r in vs}
    if not sides:
        return "unknown"
    if sides == {"buyers"}:
        return "buyers"
    if sides == {"sellers"}:
        return "sellers"
    return "mixed"


def build(tape, book, intervals: Sequence[tuple[str, float]],
          spec: of.Spec, target_ticks: float = 10.0,
          now: float | None = None, spike: float = 2.0,
          cvd_for=None) -> Ladder:
    """The whole read: book gate, tape trigger, timeframe agreement.

    `tradeable` means all three said yes. It is not a direction and not a
    suggestion -- it is the statement that nothing is currently refusing.
    """
    out = Ladder(target_ticks=target_ticks)
    out.rungs = rungs(tape, intervals, now=now, cvd_for=cvd_for)
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
    n = len(voters(out.rungs))
    if n < MIN_VOTERS:
        refusals.append(
            f"only {n} timeframe{'' if n == 1 else 's'} measured -- the tape "
            f"has not held long enough to read the rest")
    elif out.aligned in ("mixed", "unknown"):
        refusals.append(f"timeframes {out.aligned}")

    out.tradeable = not refusals
    out.why = ("; ".join(refusals) if refusals
               else f"spread, tape and all {len(voters(out.rungs))} measured "
                    f"timeframes agree ({out.aligned})")
    return out
