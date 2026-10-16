"""Order Flow Imbalance from MBP-10: who is winning the book race now.

This measures CURRENT pressure at the touch. It does not forecast. No
threshold, no position size and no execution rule appears anywhere in
this module -- those are the caller's, supplied as parameters.

THE MEASUREMENT (Cont-Kukanov-Stoikov)

For each event, compare the best bid and ask BEFORE and AFTER:

    dV_bid = +Qb'        when Pb' > Pb     the bid improved
             Qb' - Qb    when Pb' = Pb     same price, net size change
             -Qb         when Pb' < Pb     the bid was pulled

    dV_ask = -Qa         when Pa' > Pa     the ask retreated
             Qa' - Qa    when Pa' = Pa     same price, net size change
             +Qa'        when Pa' < Pa     the ask pressed down

    e_n = dV_bid - dV_ask

Positive is net buy pressure at the touch, negative net sell pressure.
Summing e_n over a window gives the windowed OFI.

WHY THERE IS NO BOOK TO REBUILD

A natural reading of "apply the event to the book" is that this module
should maintain its own ladder and mutate it per message. It must not.
An MBP-10 record already carries the full ten levels AS OF AFTER that
event, so the previous and current states are simply two consecutive
records. Rebuilding the book from deltas would add a second, fallible
copy of something the feed already states, and any drift between the
two would be invisible -- the OFI would stay plausible and be wrong.

INTEGERS ALL THE WAY THROUGH

Prices and sizes stay in the feed's raw integer units. `dbn.price`
converts to float for display; this module never calls it, because the
formula is a chain of comparisons and subtractions where an exact
integer is worth more than a readable one. A price comparison on floats
is how `Pb' == Pb` silently becomes `Pb' > Pb` and a quiet book starts
reporting pressure.

WHAT IS REFUSED RATHER THAN GUESSED

    the first event             no previous state, so no e_n
    action NONE / undefined px  an event boundary, not a book change
    a change of instrument      a roll; the state machine resets
    a sequence gap              the book is stale, so e_n is withheld
    an empty or crossed touch   reported, never computed through

Every one of those is counted and reported rather than dropped
silently. A run that refuses half its events should look alarming.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import dbn

__all__ = [
    "Touch", "LevelPair", "Event", "BookState", "Features",
    "level_ofi", "multi_level_weights", "UNDEF_PRICE",
]

UNDEF_PRICE = dbn.UNDEF_PRICE
ACTION_NONE = "N"

DEPTH = 10                      # MBP-10


# ----------------------------------------------------------- the touch

@dataclass(frozen=True)
class Touch:
    """One level of the book, in the feed's raw integer units.

    `px` of 0 or UNDEF_PRICE means the level is absent, which `valid`
    reports. An absent level is not the same as a level with no size,
    and neither is treated as a price of zero.
    """

    bid_px: int = 0
    bid_sz: int = 0
    ask_px: int = 0
    ask_sz: int = 0

    @property
    def valid(self) -> bool:
        return (self.bid_px not in (0, UNDEF_PRICE)
                and self.ask_px not in (0, UNDEF_PRICE))

    @property
    def crossed(self) -> bool:
        """Bid at or above ask. Locked counts: both are states in which
        'the best bid' and 'the best ask' stop meaning what the formula
        assumes."""
        return self.valid and self.bid_px >= self.ask_px

    @property
    def spread(self) -> int:
        return self.ask_px - self.bid_px if self.valid else 0

    @property
    def mid2(self) -> int:
        """Twice the mid, so it stays an integer."""
        return self.ask_px + self.bid_px if self.valid else 0


def level_ofi(prev: Touch, cur: Touch) -> int:
    """`e_n` for one level. Integer in, integer out.

    The three branches a side can take are mutually exclusive and
    exhaustive, so this is a total function on valid touches. It does
    NOT check validity -- `BookState` does that before calling, because
    the right response to an invalid touch is to withhold the number,
    not to return a zero that sums like a measurement.
    """
    if cur.bid_px > prev.bid_px:
        dv_bid = cur.bid_sz
    elif cur.bid_px == prev.bid_px:
        dv_bid = cur.bid_sz - prev.bid_sz
    else:
        dv_bid = -prev.bid_sz

    if cur.ask_px > prev.ask_px:
        dv_ask = -prev.ask_sz
    elif cur.ask_px == prev.ask_px:
        dv_ask = cur.ask_sz - prev.ask_sz
    else:
        dv_ask = cur.ask_sz

    return dv_bid - dv_ask


def multi_level_weights(depth: int = DEPTH,
                        floor: float = 0.1) -> tuple[float, ...]:
    """Linear decay from 1.0 at the touch to `floor` at the last level.

    Level 0 is the touch and carries full weight; level `depth-1`
    carries `floor`. With the default depth of 10 that is 1.0, 0.9,
    ... 0.1 -- the decay the spec asks for.
    """
    if depth <= 1:
        return (1.0,)
    step = (1.0 - floor) / (depth - 1)
    return tuple(1.0 - step * i for i in range(depth))


@dataclass
class LevelPair:
    """The ten levels of one record, raw."""

    levels: tuple[Touch, ...] = ()

    @property
    def touch(self) -> Touch:
        return self.levels[0] if self.levels else Touch()

    def depth_sums(self) -> tuple[int, int]:
        return (sum(l.bid_sz for l in self.levels),
                sum(l.ask_sz for l in self.levels))


def levels_from(rec) -> LevelPair:
    """One MBP-10 record to raw integer levels.

    Reads `bid_px`/`ask_px` as the integers the feed carries rather
    than through `dbn.price`, so no float ever enters the comparison.
    """
    raw = getattr(rec, "levels", None)
    if raw is None:
        raw = dbn._get(rec, "levels") or ()

    out: list[Touch] = []
    for lv in raw[:DEPTH]:
        # Direct attribute access first. `dbn._get` is a tolerant
        # getter with isinstance checks, and at four fields across ten
        # levels it is called forty times per record -- which profiled
        # as the single largest cost in the whole engine. A real
        # Databento record takes the fast path every time; the
        # fallback keeps dict-shaped fixtures and odd clients working.
        try:
            out.append(Touch(lv.bid_px or 0, lv.bid_sz or 0,
                             lv.ask_px or 0, lv.ask_sz or 0))
            continue
        except AttributeError:
            pass
        try:
            out.append(Touch(
                bid_px=int(dbn._get(lv, "bid_px", 0) or 0),
                bid_sz=int(dbn._get(lv, "bid_sz", 0) or 0),
                ask_px=int(dbn._get(lv, "ask_px", 0) or 0),
                ask_sz=int(dbn._get(lv, "ask_sz", 0) or 0)))
        except (TypeError, ValueError):
            out.append(Touch())
    return LevelPair(levels=tuple(out))


# ------------------------------------------------------------- events

@dataclass
class Event:
    """One book change, with its OFI contribution and its features."""

    ts: float
    instrument: object
    sequence: int
    action: str
    e_n: int                       # the touch contribution
    e_multi: float                 # weighted across all ten levels
    touch: Touch
    pair: LevelPair

    # Part 4 features, computed per event
    spread: int = 0
    bid_depth: int = 0
    ask_depth: int = 0

    @property
    def obi(self) -> float:
        """Book imbalance at the touch, in [-1, 1]."""
        t = self.touch
        tot = t.bid_sz + t.ask_sz
        return (t.bid_sz - t.ask_sz) / tot if tot else 0.0

    def obi_at(self, i: int) -> float:
        """Book imbalance at level `i`."""
        if i >= len(self.pair.levels):
            return 0.0
        lv = self.pair.levels[i]
        tot = lv.bid_sz + lv.ask_sz
        return (lv.bid_sz - lv.ask_sz) / tot if tot else 0.0


@dataclass
class Refusals:
    """Every event the machine declined to measure, and why.

    Counted rather than dropped: a run that refuses most of its events
    is not a run with a small sample, it is a run whose input is not
    what the code thinks it is.
    """

    first: int = 0
    boundary: int = 0
    invalid: int = 0
    crossed: int = 0
    gap: int = 0
    roll: int = 0

    @property
    def total(self) -> int:
        return (self.first + self.boundary + self.invalid
                + self.crossed + self.gap + self.roll)

    def as_dict(self) -> dict:
        return {"first": self.first, "boundary": self.boundary,
                "invalid": self.invalid, "crossed": self.crossed,
                "gap": self.gap, "roll": self.roll}


class BookState:
    """Event-by-event previous/current state machine over MBP-10.

    Feed records in chronological order; get `Event`s back for those
    that could be measured, and a `Refusals` tally for those that could
    not. The machine holds exactly one thing: the previous record's
    levels. It never rebuilds a ladder -- see the module docstring.
    """

    def __init__(self, depth: int = DEPTH, floor: float = 0.1,
                 check_sequence: bool = True) -> None:
        self.depth = depth
        self.weights = multi_level_weights(depth, floor)
        self.check_sequence = check_sequence
        self.prev: LevelPair | None = None
        self.prev_seq: int | None = None
        self.instrument: object = None
        self.refused = Refusals()
        self.seen = 0

    def reset(self) -> None:
        """Forget the previous state. Used at a roll, and after a gap."""
        self.prev = None
        self.prev_seq = None

    def feed(self, rec) -> Event | None:
        """One record in; an Event out, or None when it was refused."""
        self.seen += 1
        action = dbn._action(rec)
        ts = dbn.seconds(dbn._get(rec, "ts_event"))
        iid = dbn._get(rec, "instrument_id")
        seq = int(dbn._get(rec, "sequence", 0) or 0)

        # A boundary marker carries no book change. The spec is explicit
        # and it matters: applying one would compare a record against
        # itself and emit a zero that sums like a measurement.
        if action == ACTION_NONE:
            self.refused.boundary += 1
            return None
        if action == dbn.ACTION_CLEAR:
            self.reset()
            self.refused.boundary += 1
            return None

        cur = levels_from(rec)

        if iid != self.instrument:
            # A roll. The next contract trades dollars away; an e_n
            # across the boundary would be carry, not pressure.
            if self.instrument is not None:
                self.refused.roll += 1
            self.instrument = iid
            self.prev = cur
            self.prev_seq = seq
            if self.refused.roll == 0:
                self.refused.first += 1
            return None

        if self.prev is None:
            self.prev = cur
            self.prev_seq = seq
            self.refused.first += 1
            return None

        if (self.check_sequence and self.prev_seq is not None
                and seq and seq != self.prev_seq + 1):
            # Messages were missed, so `prev` is not the state this
            # record followed. Withhold rather than measure a gap.
            self.refused.gap += 1
            self.prev = cur
            self.prev_seq = seq
            return None

        prev_t, cur_t = self.prev.touch, cur.touch
        if not (prev_t.valid and cur_t.valid):
            self.refused.invalid += 1
            self.prev = cur
            self.prev_seq = seq
            return None
        if prev_t.crossed or cur_t.crossed:
            self.refused.crossed += 1
            self.prev = cur
            self.prev_seq = seq
            return None

        e_n = level_ofi(prev_t, cur_t)
        e_multi = 0.0
        for i, w in enumerate(self.weights):
            if i >= len(cur.levels) or i >= len(self.prev.levels):
                break
            a, b = self.prev.levels[i], cur.levels[i]
            if not (a.valid and b.valid):
                continue
            e_multi += w * level_ofi(a, b)

        bd, ad = cur.depth_sums()
        ev = Event(ts=ts, instrument=iid, sequence=seq, action=action,
                   e_n=e_n, e_multi=e_multi, touch=cur_t, pair=cur,
                   spread=cur_t.spread, bid_depth=bd, ask_depth=ad)

        self.prev = cur
        self.prev_seq = seq
        return ev


# ----------------------------------------------------------- features

@dataclass
class Features:
    """Part 4, accumulated over a stream of events.

    `msg_rate` and `cancel_to_add` are windowed over wall time; the
    rest are per event. Nothing here thresholds anything.
    """

    window_s: float = 1.0
    _times: list = field(default_factory=list)
    _adds: list = field(default_factory=list)
    _cancels: list = field(default_factory=list)

    def observe(self, ts: float, action: str) -> None:
        self._times.append(ts)
        if action == "A":
            self._adds.append(ts)
        elif action == "C":
            self._cancels.append(ts)
        self._trim(ts)

    def _trim(self, now: float) -> None:
        cut = now - self.window_s
        for seq in (self._times, self._adds, self._cancels):
            i = 0
            while i < len(seq) and seq[i] < cut:
                i += 1
            del seq[:i]

    @property
    def msg_rate(self) -> int:
        return len(self._times)

    @property
    def cancel_to_add(self) -> float | None:
        """Cancels per add in the window. None when nothing was added
        -- a ratio with a zero denominator is not infinity, it is a
        question the window cannot answer."""
        adds = len(self._adds)
        return (len(self._cancels) / adds) if adds else None


def relative_spread(touch: Touch) -> float:
    """Spread as a fraction of the mid. 0.0 when there is no touch."""
    if not touch.valid or touch.mid2 <= 0:
        return 0.0
    return 2.0 * touch.spread / touch.mid2


def book_slope(bid_depth: int, ask_depth: int, spread: int,
               levels: int = DEPTH) -> float:
    """Total depth per unit of spread per level.

    None of the three inputs may be zero for this to mean anything, so
    a zero spread returns 0.0 rather than dividing.
    """
    if spread <= 0 or levels <= 0:
        return 0.0
    return (bid_depth + ask_depth) / (spread * levels)


# -------------------------------------------------- who is winning now

@dataclass(frozen=True)
class Reading:
    """The state of the race at one event.

    `z` is None until the baseline is full and while the baseline is
    flat. Both are real states: a z-score computed from a handful of
    samples, or divided by a standard deviation of zero, is a number
    with no meaning, and it would read as an extreme signal exactly
    when there is least information.
    """

    ts: float
    window: int                    # summed e_n over the last N events
    z: float | None
    mean: float
    std: float
    samples: int
    warm: bool

    def side(self, z_enter: float) -> str:
        """'buyers', 'sellers' or 'neither', at a threshold YOU supply.

        There is no default. The caller names the line; this reports
        which side of it the measurement fell on.
        """
        if self.z is None or z_enter <= 0:
            return "neither"
        if self.z >= z_enter:
            return "buyers"
        if self.z <= -z_enter:
            return "sellers"
        return "neither"


class Race:
    """Rolling OFI window, z-scored against a rolling baseline.

    Two windows, both the caller's to size:

        window    how many events are summed into the current OFI
        baseline  how many of those sums the z-score is measured against

    The baseline is a window of WINDOW SUMS, not of raw e_n. Z-scoring
    a sum against the distribution of single events would compare
    quantities of different scale and report an extreme on every
    reading.

    NOTHING IS THRESHOLDED HERE. `Reading.side` takes the line as an
    argument because the line is a trading decision and this is a
    measurement.
    """

    def __init__(self, window: int = 100, baseline: int = 1000,
                 recompute_every: int = 100_000) -> None:
        if window < 1 or baseline < 2:
            raise ValueError("window >= 1 and baseline >= 2")
        from collections import deque
        self.window = window
        self.baseline = baseline
        self.recompute_every = max(1, recompute_every)
        self._events: deque = deque(maxlen=window)
        self._sums: deque = deque(maxlen=baseline)
        self._run = 0                  # current windowed sum
        self._s1 = 0.0                 # sum of the baseline
        self._s2 = 0.0                 # sum of squares of the baseline
        self._since = 0

    def push(self, e_n: int, ts: float = 0.0) -> Reading:
        """One event in, the state of the race out."""
        if len(self._events) == self.window:
            self._run -= self._events[0]
        self._events.append(e_n)
        self._run += e_n
        cur = self._run

        if len(self._sums) == self.baseline:
            old = self._sums[0]
            self._s1 -= old
            self._s2 -= old * old
        self._sums.append(cur)
        self._s1 += cur
        self._s2 += cur * cur

        self._since += 1
        if self._since >= self.recompute_every:
            # Incremental sums drift over millions of updates. Rebuild
            # exactly, periodically, rather than letting the variance
            # wander somewhere it cannot be checked.
            self._s1 = float(sum(self._sums))
            self._s2 = float(sum(x * x for x in self._sums))
            self._since = 0

        n = len(self._sums)
        warm = n >= self.baseline
        mean = self._s1 / n if n else 0.0
        var = (self._s2 / n - mean * mean) if n else 0.0
        if var < 0.0:
            var = 0.0              # float drift, never a real variance
        std = var ** 0.5

        z = None
        if warm and std > 0.0:
            z = (cur - mean) / std

        return Reading(ts=ts, window=cur, z=z, mean=mean, std=std,
                       samples=n, warm=warm)
