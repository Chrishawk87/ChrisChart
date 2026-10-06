"""How the read GOT to where it is, sampled across the bar in progress.

WHAT WAS MISSING

Everything else in this project answers "what is true now". That makes a
bar look the same whether its evidence has been building for seven minutes
or has just fallen apart:

    10:01   3 of 7 for
    10:03   4 of 7 for
    10:05   5 of 7 for
    10:07   6 of 7 for      <- this

    10:01   7 of 7 for
    10:03   6 of 7 for
    10:05   5 of 7 for
    10:07   6 of 7 for      <- and this

are the same instant and nothing like the same bar. This file keeps the
column on the left.

A COUNT, NOT A PROBABILITY

It says "5 of 7 agree", never "71% bullish". The difference is not
cosmetic. A percentage composited out of weighted factors is a confidence
score wearing a percent sign: it reads as a measured frequency while being
nothing of the kind, and it is most persuasive exactly where it is most
wrong. An earlier engine in this project scored 98.4% on a pure random
walk.

A count is checkable. Each of the seven is one named field, from one
module, visible on the screen, and a reader who disagrees can point at the
one that is wrong. When enough closed bars have been scored, "6 of 7" maps
to a hit rate that was MEASURED, and only then does it deserve a percent
sign.

THE DIRECTION BEING VOTED ON IS THE BAR'S OWN

Not the engine's opinion of the bar. If the candidate direction came from
the stack reader, then the stack's vote would be guaranteed and the count
would partly be measuring itself. So the candidate is where price actually
is against its open, and every check -- the stack included -- is a real
vote on whether that holds.

A bar sitting exactly on its open has no candidate, and this says so
rather than picking one.

A FLIP IS NOT A CHANGE IN THE COUNT

When price crosses back through its open, the question changes. "6 of 7
for up" and "6 of 7 for down" are not six and six, they are two different
questions each asked once. So a flip ends the leg and starts a new one,
and nothing is compared across the join.

FOUR ANSWERS, NOT TWO

for / against / quiet / unmeasured. A layer with no baseline yet and a
layer that measured balance are different, and collapsing either of them
into "against" is how a warming-up feed starts reading as bearish.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

Way = Literal["up", "down"]
Verdict = Literal["for", "against", "quiet", "unmeasured"]

# How effective aggression has to be before it counts as working, and how
# ineffective before it counts as being absorbed. Both are ratios against
# how far this much flow normally moves this market, and they are the same
# two numbers `result` classifies on -- deliberately, so the trajectory and
# the verdict cannot disagree about what absorption is.
WORKING = 1.0
ABSORBED = 0.5

# A difference in pull share below this is two sides behaving the same way.
PULL_EDGE = 0.15

# Where in its own range a bar has to be sitting before that says anything.
TOP = 0.66
BOTTOM = 0.34

# The seven, in the order they are shown.
CHECKS = ("stack", "intent", "efficiency", "response", "book", "fuel",
          "location")

# Samples are thinned to roughly this many across a bar, whatever its
# length, so a four-hour bar does not keep thousands and a one-minute bar
# still gets a useful handful.
TARGET_SAMPLES = 150
MIN_GAP_S = 3.0
MAX_SAMPLES = 220


def gap_for(interval_s: float) -> float:
    """The shortest time between kept samples, for a bar this long."""
    if interval_s <= 0:
        return MIN_GAP_S
    return max(MIN_GAP_S, float(interval_s) / TARGET_SAMPLES)


@dataclass(frozen=True)
class Check:
    """One named reading, and which way it votes on the candidate."""

    name: str
    verdict: Verdict = "unmeasured"
    why: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "verdict": self.verdict, "why": self.why}


def _way_of(move_bps: float) -> Way | None:
    """The candidate: where price is against its own open."""
    if move_bps > 0:
        return "up"
    if move_bps < 0:
        return "down"
    return None


def _agrees(side: str | None, way: Way) -> bool | None:
    """Whether a buyers/sellers reading backs the candidate direction."""
    if side == "buyers":
        return way == "up"
    if side == "sellers":
        return way == "down"
    return None


# ---------------------------------------------------------------------------
# the seven checks
# ---------------------------------------------------------------------------
#
# Each one reads ONE already-published field. Nothing is recomputed here,
# for the same reason the layers are assembled rather than measured: a
# number that appears twice will eventually appear twice differently.


def _stack(way: Way, casc: dict) -> Check:
    t = (casc or {}).get("trend")
    if t is None:
        return Check("stack", "unmeasured",
                     "the timeframe you are trading has not qualified")
    if t == way:
        return Check("stack", "for", "the stack under it agrees")
    return Check("stack", "against", f"the stack reads {t}")


def _intent(way: Way, L: dict) -> Check:
    i = (L or {}).get("intent") or {}
    if not i.get("measured"):
        return Check("intent", "unmeasured", "fills are not being counted")
    a = _agrees(i.get("who"), way)
    if a is None:
        return Check("intent", "quiet", "neither side is crossing")
    return Check("intent", "for" if a else "against",
                 f"{i.get('who')} are crossing the spread")


def _efficiency(way: Way, L: dict) -> Check:
    """THE one Chris called the solidifier, and the absorption inversion.

    Effectiveness has no side of its own -- it is the effectiveness of
    whoever is doing the attacking, so the side crossing the spread is what
    gives it a direction. Heavy buying that is NOT moving price is a point
    against up, not a weak point for it.
    """
    r = (L or {}).get("resistance") or {}
    i = (L or {}).get("intent") or {}
    if not r.get("measured"):
        return Check("efficiency", "unmeasured", "no impact baseline yet")
    a = _agrees(i.get("who"), way)
    if a is None:
        return Check("efficiency", "quiet",
                     "nobody is attacking hard enough to judge")
    x = float(r.get("impact_ratio") or 0.0)
    if x >= WORKING:
        return Check("efficiency", "for" if a else "against",
                     f"that aggression is moving price {x:.0%} of normal")
    if x <= ABSORBED:
        # Being absorbed counts AGAINST the side doing the pushing.
        return Check("efficiency", "against" if a else "for",
                     f"it is being absorbed -- {x:.0%} of normal impact")
    return Check("efficiency", "quiet",
                 f"{x:.0%} of normal impact -- neither working nor absorbed")


def _response(way: Way, L: dict) -> Check:
    p = (L or {}).get("response") or {}
    if not p.get("measured"):
        return Check("response", "unmeasured",
                     "not enough of this bar to say what price did")
    if not p.get("wide"):
        return Check("response", "quiet",
                     "price has not gone anywhere for a bar this long")
    mv = float(p.get("move") or 0.0)
    if mv == 0:
        return Check("response", "quiet", "price is on its open")
    went: Way = "up" if mv > 0 else "down"
    return Check("response", "for" if went == way else "against",
                 f"price has gone {mv:+.1f}{p.get('unit') or ''}")


def _book(way: Way, L: dict) -> Check:
    """Which side's resting size is leaving.

    Offers withdrawing ahead of a rise is the book getting out of the way;
    bids withdrawing is the floor coming out. Only the DIFFERENCE between
    the two means anything -- both sides churn in a fast market, and a
    single side's pull share on its own mostly measures how busy it is.
    """
    q = (L or {}).get("liquidity") or {}
    d = q.get("dom") or {}
    if not q.get("measured"):
        return Check("book", "unmeasured", "the book has not been watched "
                                           "long enough")
    bid = float(((d.get("bid") or {}).get("pull_share")) or 0.0)
    ask = float(((d.get("ask") or {}).get("pull_share")) or 0.0)
    edge = ask - bid
    if abs(edge) < PULL_EDGE:
        return Check("book", "quiet", "both sides are pulling about equally")
    pulling = "offers" if edge > 0 else "bids"
    helps: Way = "up" if edge > 0 else "down"
    return Check("book", "for" if helps == way else "against",
                 f"{pulling} are the ones being withdrawn")


def _fuel(way: Way, vol: dict) -> Check:
    """Volume has no side, so this votes on whether the bar can move at all.

    Deliberately NOT tied to the direction: tying it to the aggressor would
    be the efficiency check again under another name, and the same evidence
    counted twice looks like two agreeing readings.
    """
    if not vol or not vol.get("known"):
        return Check("fuel", "unmeasured", "no volume baseline for this bar")
    s = vol.get("state")
    if s == "paid_for":
        return Check("fuel", "for", "the move is being paid for")
    if s == "no_fuel":
        return Check("fuel", "against", "there is no volume behind this bar")
    if s == "absorbed":
        return Check("fuel", "against",
                     "heavy volume and almost no range")
    if s == "thin":
        return Check("fuel", "quiet", "a wide range on light volume")
    return Check("fuel", "quiet", "volume is unremarkable")


def _location(way: Way, L: dict) -> Check:
    """Where in its own range the bar is sitting.

    A bar that ran up and came back to its low is a different thing from
    one sitting on its high, and the move alone cannot tell them apart.
    """
    p = (L or {}).get("response") or {}
    if not p.get("measured"):
        return Check("location", "unmeasured", "the bar has no range yet")
    pos = float(p.get("position") if p.get("position") is not None else 0.5)
    if pos >= TOP:
        return Check("location", "for" if way == "up" else "against",
                     f"sitting at {pos:.0%} of its range")
    if pos <= BOTTOM:
        return Check("location", "for" if way == "down" else "against",
                     f"sitting at {pos:.0%} of its range")
    return Check("location", "quiet",
                 f"mid-range at {pos:.0%} -- it has given back its move")


def evaluate(way: Way, layers: dict, cascade: dict | None = None,
             volume: dict | None = None) -> tuple[Check, ...]:
    """The seven, against a candidate direction.

    Fed the dicts that are already going to the screen rather than the
    objects behind them, so a check can only ever use something the reader
    can see for themselves.
    """
    return (_stack(way, cascade or {}),
            _intent(way, layers),
            _efficiency(way, layers),
            _response(way, layers),
            _book(way, layers),
            _fuel(way, volume or {}),
            _location(way, layers))


# ---------------------------------------------------------------------------
# the trajectory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """The seven, at one instant."""

    ts: float
    way: Way | None = None
    checks: tuple[Check, ...] = ()

    def _n(self, v: Verdict) -> int:
        return sum(1 for c in self.checks if c.verdict == v)

    @property
    def n_for(self) -> int:
        return self._n("for")

    @property
    def n_against(self) -> int:
        return self._n("against")

    @property
    def n_quiet(self) -> int:
        return self._n("quiet")

    @property
    def n_unmeasured(self) -> int:
        return self._n("unmeasured")

    @property
    def counted(self) -> int:
        """Checks that could say anything at all."""
        return len(self.checks) - self.n_unmeasured

    def describe(self) -> str:
        if self.way is None:
            return "price is on its open -- nothing to count for or against"
        bits = [f"{self.n_for} of {self.counted} for {self.way}"]
        if self.n_against:
            bits.append(f"{self.n_against} against")
        if self.n_quiet:
            bits.append(f"{self.n_quiet} quiet")
        if self.n_unmeasured:
            bits.append(f"{self.n_unmeasured} not measured")
        return ", ".join(bits)

    def to_dict(self) -> dict:
        return {"ts": self.ts, "way": self.way,
                "for": self.n_for, "against": self.n_against,
                "quiet": self.n_quiet, "unmeasured": self.n_unmeasured,
                "counted": self.counted,
                "checks": [c.to_dict() for c in self.checks],
                "describe": self.describe()}


@dataclass
class Leg:
    """A run of samples that were all asking the same question.

    A leg ends when price crosses back through its open, because the
    question changes with it and the counts stop being comparable.
    """

    way: Way | None = None
    samples: list[Sample] = field(default_factory=list)

    @property
    def latest(self) -> Sample | None:
        return self.samples[-1] if self.samples else None

    @property
    def first(self) -> Sample | None:
        return self.samples[0] if self.samples else None

    def at_or_before(self, ts: float) -> Sample | None:
        """The newest sample no later than `ts`."""
        out = None
        for s in self.samples:
            if s.ts <= ts:
                out = s
            else:
                break
        return out

    def to_dict(self, keep: int = 0) -> dict:
        got = self.samples[-keep:] if keep else self.samples
        return {"way": self.way, "n": len(self.samples),
                "samples": [s.to_dict() for s in got]}


@dataclass
class Trace:
    """One bar's worth of legs."""

    trading: str = ""
    bar_open: float = 0.0
    interval_s: float = 0.0
    legs: list[Leg] = field(default_factory=list)

    @property
    def leg(self) -> Leg | None:
        return self.legs[-1] if self.legs else None

    @property
    def latest(self) -> Sample | None:
        return self.leg.latest if self.leg else None

    @property
    def flips(self) -> int:
        """How many times the bar has crossed back through its open.

        Counted over legs that actually had a direction: a bar that opened
        flat and then picked a side has not flipped, it has started.
        """
        ways = [l.way for l in self.legs if l.way is not None]
        return max(0, len(ways) - 1)

    def record(self, s: Sample) -> bool:
        """Keep a sample. False if it was too soon after the last one.

        Throttled by the bar's own length so a four-hour bar does not keep
        thousands of samples and a one-minute bar still gets a useful
        handful. A direction change is NEVER throttled away -- it is the
        one event the trajectory exists to show.
        """
        cur = self.leg
        turned = cur is None or cur.way != s.way
        if turned:
            self.legs.append(Leg(way=s.way, samples=[s]))
            return True
        last = cur.latest
        if last is not None and s.ts - last.ts < gap_for(self.interval_s):
            return False
        cur.samples.append(s)
        if len(cur.samples) > MAX_SAMPLES:
            # Drop from the MIDDLE, never an end. The opening of the bar
            # and the last few seconds are the two parts anybody looks at.
            del cur.samples[len(cur.samples) // 2]
        return True

    def change_since(self, seconds: float) -> int | None:
        """How the count has moved over the last `seconds`, within this leg.

        None when there is nothing to compare against -- a leg a few
        seconds old, or one that has only ever held a single sample. Never
        across a flip: the two sides of one are different questions.
        """
        cur, now = self.leg, self.latest
        if cur is None or now is None:
            return None
        then = cur.at_or_before(now.ts - float(seconds))
        if then is None or then is now:
            then = cur.first
        # A leg holding one sample lands here with `then is now`, so this
        # covers the single-sample case as well. Guarding the length
        # separately above reads as a second check and is not one.
        if then is now:
            return None
        return now.n_for - then.n_for

    def changes(self) -> list[str]:
        """Which checks flipped between the last two samples of this leg."""
        cur = self.leg
        if cur is None or len(cur.samples) < 2:
            return []
        was = {c.name: c.verdict for c in cur.samples[-2].checks}
        out = []
        for c in cur.samples[-1].checks:
            before = was.get(c.name)
            if before is not None and before != c.verdict:
                out.append(f"{c.name} {before} → {c.verdict}")
        return out

    def describe(self) -> str:
        now = self.latest
        if now is None:
            return "nothing recorded on this bar yet"
        d = self.change_since(60.0)
        if d is None:
            arrow = "just started"
        elif d > 0:
            arrow = f"strengthening (+{d} in the last minute)"
        elif d < 0:
            arrow = f"weakening ({d} in the last minute)"
        else:
            arrow = "holding"
        return f"{now.describe()} -- {arrow}"

    def to_dict(self, keep: int = 40) -> dict:
        now = self.latest
        return {"trading": self.trading, "bar_open": self.bar_open,
                "flips": self.flips,
                "now": now.to_dict() if now else None,
                "change_60s": self.change_since(60.0),
                "changes": self.changes(),
                "leg": self.leg.to_dict(keep=keep) if self.leg else None,
                "describe": self.describe()}


class Tracer:
    """The traces, one per coin and traded timeframe, reset on the bar roll.

    A bar's trajectory belongs to that bar. Carrying it over the roll would
    draw the previous candle's build-up as though it were this one's.
    """

    def __init__(self) -> None:
        self._by: dict[tuple[str, str], Trace] = {}

    def trace(self, coin: str, trading: str, bar_open: float,
              interval_s: float) -> Trace:
        key = (coin, trading)
        got = self._by.get(key)
        if got is None or got.bar_open != float(bar_open):
            got = Trace(trading=trading, bar_open=float(bar_open),
                        interval_s=float(interval_s))
            self._by[key] = got
        return got

    def observe(self, coin: str, trading: str, bar_open: float,
                interval_s: float, ts: float, move_bps: float,
                layers: dict, cascade: dict | None = None,
                volume: dict | None = None) -> Trace:
        """Read the seven and add them to this bar's trajectory."""
        t = self.trace(coin, trading, bar_open, interval_s)
        way = _way_of(float(move_bps))
        checks = evaluate(way, layers, cascade, volume) if way else ()
        t.record(Sample(ts=float(ts), way=way, checks=checks))
        return t
