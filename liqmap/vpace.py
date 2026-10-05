"""Whether this candle has the volume to move, and where it got the yardstick.

THE QUESTION

Direction says which way the bar wants to go. Volume says whether it can.
A bar leaning hard on a tenth of its usual volume is three traders agreeing
with each other; the same lean on twice its usual volume is a move being
paid for. Both look identical on a direction read.

WHY RAW VOLUME CANNOT ANSWER IT

Six minutes into a fifteen-minute bar, the bar has whatever it has, and
comparing that to a full bar's volume says "quiet" every time until the
bar is nearly over. So the comparison has to be against what a bar of this
length has normally done BY THIS POINT -- which means knowing the shape of
volume inside a bar, not just its total.

That shape is not flat. Bars are front-loaded around the open, back-loaded
into the close, and lumpy in between, and assuming it flat reports the
start of every slow bar as quiet and the end of every one as heavy. So the
shape is learned.

WHERE THE SHAPE COMES FROM, INCLUDING FOR FAST BARS

A bar's internal shape can only be learned from something finer than the
bar. The answer is different at the two ends, and the fast end is the one
that looks impossible:

    FAST BARS (up to about five minutes) -- from the tape itself. The
    rolling tape holds an hour of individual fills with their timestamps,
    so a one-minute bar can be cut into twelve five-second slices and
    sixty completed bars' worth of them are sitting in memory already.
    Nothing finer than a bar is needed because the fills ARE finer than
    the bar.

    SLOW BARS -- from the kept one-minute bars. A fifteen-minute bar is
    fifteen one-minute bars, and those are already being written down as
    they close. Twelve completed fifteen-minute bars is three hours of
    one-minute rows, which survive a restart.

    NEITHER YET -- an even spread, reported as such. Honest, wrong in a
    known direction, and replaced the moment there is enough to learn
    from.

WHAT THE ANSWER IS NOT

Volume on its own does not say move-or-hold. Heavy volume with a wide
range is a move being paid for; heavy volume with no range is absorption,
which is the same volume saying the opposite thing. Light volume with a
wide range is price sliding through a thin book, which moves easily and
reverses just as easily. The four are named in `State` and the pairing is
the point -- effort is only meaningful next to its result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

# How many pieces a bar is cut into when the shape is learned from the
# tape. Twelve is five seconds on a one-minute bar: fine enough to see the
# shape, coarse enough that each slice has prints in it.
TAPE_SLICES = 12

# Slice counts tried, in order, when the shape is learned from one-minute
# bars. The first that divides the bar into whole minutes wins, because a
# slice boundary inside a minute cannot be built from minute bars.
MINUTE_SLICES = (24, 20, 16, 15, 12, 10, 8, 6, 5, 4, 3, 2)

# Completed bars below which a learned shape is not worth having. Four
# bars' medians are three bars and an opinion.
MIN_CURVE_BARS = 8

# A share this far into the bar can never be zero: it divides a projection.
MIN_SHARE = 1e-4

# Where the pace stops being ordinary in each direction.
HEAVY_X = 1.5
QUIET_X = 0.6

# How wide a range has to be, against a normal bar's, to count as having
# gone somewhere. Below it the bar is still.
WIDE_RANGE_X = 0.8

State = Literal["paid_for", "absorbed", "thin", "no_fuel", "unknown"]

MEANING: dict[str, str] = {
    "paid_for": "heavy volume and a wide range -- the move is being paid "
                "for, which is the one that tends to keep going",
    "absorbed": "heavy volume and almost no range -- the size is there and "
                "price will not move, so somebody is filling all of it",
    "thin": "a wide range on light volume -- price is sliding through an "
            "empty book, which extends easily and reverses just as easily",
    "no_fuel": "light volume and no range -- nothing is happening in this "
               "bar yet",
    "unknown": "not enough of this bar, or not enough history, to say",
}


# ---------------------------------------------------------------------------
# the shape of volume inside a bar
# ---------------------------------------------------------------------------


def _median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if not n:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


@dataclass
class Curve:
    """What share of a bar's volume is normally done by each point in it.

    `shares` is CUMULATIVE and ends at 1.0: `shares[i]` is the share done
    by the end of slice i. An even spread is the straight line through it,
    and a front-loaded market bulges above that line.
    """

    timeframe: str
    interval_s: float
    shares: list[float] = field(default_factory=list)
    samples: int = 0
    source: str = "even"      # tape | bars | even
    note: str = ""

    @property
    def learned(self) -> bool:
        return self.source != "even" and self.samples >= MIN_CURVE_BARS

    @property
    def slices(self) -> int:
        return len(self.shares)

    def share_by(self, elapsed_s: float) -> float:
        """Share of a normal bar's volume done by `elapsed_s` into it.

        Interpolated between slice boundaries rather than stepped, so the
        expectation rises smoothly instead of jumping every slice and
        making the pace sawtooth.
        """
        if self.interval_s <= 0:
            return 1.0
        frac = min(1.0, max(0.0, elapsed_s / self.interval_s))
        if not self.shares:
            return max(MIN_SHARE, frac)          # even spread
        n = len(self.shares)
        pos = frac * n
        i = int(pos)
        if i >= n:
            return 1.0
        lo = self.shares[i - 1] if i > 0 else 0.0
        hi = self.shares[i]
        return max(MIN_SHARE, lo + (hi - lo) * (pos - i))

    def to_dict(self) -> dict:
        return {"timeframe": self.timeframe, "slices": self.slices,
                "samples": self.samples, "source": self.source,
                "learned": self.learned, "note": self.note,
                "shares": [round(s, 4) for s in self.shares]}


def even_curve(timeframe: str, interval_s: float, note: str = "") -> Curve:
    return Curve(timeframe=timeframe, interval_s=interval_s, shares=[],
                 samples=0, source="even",
                 note=note or "volume assumed to spread evenly through the "
                              "bar -- not enough history to learn its shape")


def curve_from_bars(timeframe: str, interval_s: float,
                    bars: Sequence[Sequence[float]],
                    source: str = "bars") -> Curve:
    """Learn the shape from completed bars, each cut into slices.

    `bars` is one list of slice volumes per completed bar, all the same
    length.

    EACH BAR IS NORMALISED BEFORE THEY ARE COMBINED. Without that, one
    enormous bar sets the shape for all of them -- the average would
    describe that bar's afternoon rather than the market's habit. And the
    MEDIAN across bars, not the mean, for the same reason one rung back:
    a single news bar should not be able to move the yardstick every
    ordinary bar afterwards is judged against.
    """
    usable = [list(b) for b in bars if b and sum(b) > 0]
    if len(usable) < MIN_CURVE_BARS:
        return even_curve(timeframe, interval_s,
                          note=f"only {len(usable)} completed bars to learn "
                               f"the shape from, wants {MIN_CURVE_BARS}")
    n = min(len(b) for b in usable)
    if n <= 1:
        return even_curve(timeframe, interval_s,
                          note="a bar cut into one piece has no shape")

    cum: list[list[float]] = []
    for b in usable:
        total = sum(b[:n])
        if total <= 0:
            continue
        run = 0.0
        row = []
        for v in b[:n]:
            run += max(0.0, v)
            row.append(run / total)
        cum.append(row)
    if len(cum) < MIN_CURVE_BARS:
        return even_curve(timeframe, interval_s,
                          note="too few bars with volume in them")

    # Taken slice by slice across bars. This cannot dip and cannot end
    # anywhere but 1.0, and neither is enforced here: every row is a
    # cumulative share of its own bar, so every row is non-decreasing and
    # ends at 1.0 -- and an order statistic of non-decreasing sequences is
    # non-decreasing. A fix-up pass would be dead code pretending to hold
    # the invariant up. The test asserts it instead, so a change to the
    # construction that broke it would be caught rather than papered over.
    shares = [_median([row[i] for row in cum]) for i in range(n)]
    return Curve(timeframe=timeframe, interval_s=interval_s, shares=shares,
                 samples=len(cum), source=source,
                 note=f"shape learned from {len(cum)} completed bars, "
                      f"{n} slices each")


def slices_from_tape(trades, interval_s: float, now: float,
                     slices: int = TAPE_SLICES,
                     back: int = 60) -> list[list[float]]:
    """Cut recent completed bars into slices, straight off the fills.

    This is the answer for fast bars. Nothing finer than a one-minute bar
    gets stored anywhere, but the fills themselves are finer than any bar,
    and the tape is holding an hour of them.

    The bar in progress is excluded: a partial bar normalised against its
    own partial total would report a full bar's shape every time.
    """
    out: list[list[float]] = []
    if interval_s <= 0 or slices <= 0:
        return out
    rows = list(trades or [])
    if not rows:
        return out
    cur_start = now - (now % interval_s)
    width = interval_s / slices
    buckets: dict[float, list[float]] = {}
    for t in rows:
        ts = float(getattr(t, "ts", 0.0))
        if ts >= cur_start:
            continue                      # the bar in progress
        start = ts - (ts % interval_s)
        if start < cur_start - back * interval_s:
            continue
        b = buckets.setdefault(start, [0.0] * slices)
        i = min(slices - 1, int((ts - start) / width))
        b[i] += float(getattr(t, "notional", 0.0))

    # The bar the tape's earliest fill falls inside may have begun before
    # the tape was listening, and there is no way to tell from here. It
    # would look back-loaded when it was not, so it goes -- along with the
    # bar in progress, these are the only two that can be partial.
    first = float(getattr(rows[0], "ts", 0.0))
    partial = first - (first % interval_s)
    for start in sorted(buckets):
        if start <= partial:
            continue
        out.append(buckets[start])
    return out


def minute_slices(interval_s: float) -> int:
    """How many pieces a bar can be cut into using whole minutes."""
    minutes = int(round(interval_s / 60.0))
    if minutes < 2:
        return 0
    for s in MINUTE_SLICES:
        if s <= minutes and minutes % s == 0:
            return s
    return 0


def slices_from_minutes(rows, interval_s: float,
                        slices: int | None = None) -> list[list[float]]:
    """Cut completed bars into slices using one-minute bars.

    `rows` are one-minute records with `bar_ts` and a notional total,
    oldest first. This is the slow-bar path: a fifteen-minute bar is
    fifteen one-minute bars and those are already written down.

    A bar missing any of its minutes is DROPPED rather than filled in. A
    gap would read as a quiet patch inside the bar and bend the shape
    towards whatever the feed happened to miss.
    """
    n = slices if slices is not None else minute_slices(interval_s)
    if n <= 0 or interval_s <= 0:
        return []
    per = int(round(interval_s / 60.0)) // n      # minutes per slice
    if per <= 0:
        return []
    want = n * per
    buckets: dict[float, list[float | None]] = {}
    for r in rows or []:
        ts = float(getattr(r, "bar_ts", 0.0))
        start = ts - (ts % interval_s)
        got = buckets.setdefault(start, [None] * want)
        idx = int((ts - start) / 60.0)
        if 0 <= idx < want:
            total = float(getattr(r, "total", 0.0))
            if total <= 0:
                total = float(getattr(r, "volume", 0.0))
            got[idx] = total

    out: list[list[float]] = []
    for start in sorted(buckets):
        mins = buckets[start]
        if any(m is None for m in mins):
            continue                       # an incomplete bar has no shape
        out.append([sum(mins[i * per:(i + 1) * per]) for i in range(n)])
    return out


# ---------------------------------------------------------------------------
# the reading
# ---------------------------------------------------------------------------


@dataclass
class VolumePace:
    """How this bar's volume is running, and what it has bought."""

    timeframe: str
    interval_s: float
    done: float = 0.0          # notional so far in this bar
    normal_full: float = 0.0   # what a whole bar of this length normally does
    elapsed_s: float = 0.0
    range_bps: float = 0.0
    normal_range_bps: float = 0.0
    curve: Curve | None = None

    @property
    def share(self) -> float:
        c = self.curve
        if c is None:
            frac = (min(1.0, self.elapsed_s / self.interval_s)
                    if self.interval_s > 0 else 1.0)
            return max(MIN_SHARE, frac)
        return c.share_by(self.elapsed_s)

    @property
    def expected(self) -> float:
        """What a normal bar has done BY NOW -- not what it does in total."""
        return self.normal_full * self.share

    @property
    def known(self) -> bool:
        return self.normal_full > 0 and self.interval_s > 0

    @property
    def pace(self) -> float:
        """1.0 means exactly on a normal bar's pace for this point in it."""
        e = self.expected
        return (self.done / e) if e > 0 else 0.0

    @property
    def projected(self) -> float:
        """Where this bar's volume finishes if it keeps this pace."""
        return self.done / self.share if self.share > 0 else 0.0

    @property
    def projected_x(self) -> float:
        return (self.projected / self.normal_full) if self.normal_full > 0 \
            else 0.0

    @property
    def heavy(self) -> bool:
        return self.known and self.pace >= HEAVY_X

    @property
    def quiet(self) -> bool:
        return self.known and self.pace <= QUIET_X

    @property
    def wide(self) -> bool:
        """Has this bar actually gone anywhere, for a bar of its length.

        Scaled by how much of the bar has run, and by the square root
        because that is how far a random walk gets in a fraction of the
        time -- the same scaling the flat band uses, for the same reason.
        """
        if self.normal_range_bps <= 0 or self.interval_s <= 0:
            return False
        frac = min(1.0, max(0.02, self.elapsed_s / self.interval_s))
        want = self.normal_range_bps * (frac ** 0.5) * WIDE_RANGE_X
        return self.range_bps >= want

    @property
    def state(self) -> State:
        """Effort against result. Volume alone never answers this.

        The same heavy volume means opposite things depending on what it
        bought: a wide range means the move is being paid for, no range
        means it is being absorbed.
        """
        if not self.known:
            return "unknown"
        if self.heavy:
            return "paid_for" if self.wide else "absorbed"
        if self.quiet:
            return "thin" if self.wide else "no_fuel"
        return "paid_for" if self.wide else "unknown"

    @property
    def efficiency(self) -> float:
        """Basis points of range per unit of normal volume.

        High means price moves easily here; low means it is grinding,
        which is what absorption looks like before it is obvious.
        """
        p = self.pace
        return round(self.range_bps / p, 3) if p > 0 else 0.0

    def describe(self) -> str:
        if not self.known:
            return (f"{self.timeframe}: no volume yardstick yet -- "
                    f"nothing to compare this bar against")
        c = self.curve
        how = (f"shape {c.source}" if c is not None and c.learned
               else "even spread assumed")
        return (f"{self.timeframe}: {self.pace:.2f}x the volume a normal "
                f"bar has by this point, on pace to finish at "
                f"{self.projected_x:.2f}x. {MEANING[self.state]} ({how})")

    def to_dict(self) -> dict:
        return {"timeframe": self.timeframe, "done": round(self.done, 2),
                "expected": round(self.expected, 2),
                "normal_full": round(self.normal_full, 2),
                "share": round(self.share, 4),
                "pace": round(self.pace, 3),
                "projected": round(self.projected, 2),
                "projected_x": round(self.projected_x, 3),
                "heavy": self.heavy, "quiet": self.quiet, "wide": self.wide,
                "state": self.state, "known": self.known,
                "efficiency": self.efficiency,
                "range_bps": round(self.range_bps, 3),
                "normal_range_bps": round(self.normal_range_bps, 3),
                "curve": self.curve.to_dict() if self.curve else None,
                "describe": self.describe()}
