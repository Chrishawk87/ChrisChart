"""Does OFI beat its own control? The one number worth having first.

    from liqmap.ofi_backtest import load, run
    secs = load("ES_sec.csv")
    out = run(secs, z_enter=1.5, target_ticks=16, stop_ticks=4)

Reads the per-second file `scripts/ofi_replay.py` writes and asks one
question: entering on an OFI reading, does the result separate from
the same trades taken the wrong way round, and from the same entry
times taken at random?

WHY THREE ARMS AND NOT ONE

A backtest of a signal on its own reports a number with nothing to
compare it to. 52% looks like an edge until the same rule run
backwards also reports 52%, at which point what was measured was the
bracket's geometry -- a 16-tick target against a 4-tick stop wins
often in both directions -- rather than anything the signal knew.

    SIGNAL    side from the sign of z
    MIRROR    the same entries, the opposite side
    CONTROL   the same entry times, a side chosen by coin flip

SIGNAL must beat MIRROR, and both must be read against CONTROL. An
earlier run of this comparison on timeframe agreement had MIRROR at
50.3% and CONTROL at 50.0% against a signal of 49.7%: indistinguishable,
which is the answer that saves the work built on top.

COSTS ARE PRE-REGISTERED, never fitted. `cost_ticks` is an argument
and the break-even it implies is reported beside every result, so a
win rate is never shown without the number it had to clear. Tightening
cost assumptions until a backtest passes is how a losing rule gets
shipped.

    break-even = (stop + cost) / (target + stop)

Both sides pay. A 4-tick target against a 4-tick stop at one tick of
cost needs 62.5%, not 55% -- the loser pays the cost too.

AMBIGUOUS SECONDS ARE NOT RESOLVED IN YOUR FAVOUR. When one second's
high reaches the target and its low reaches the stop, the file does
not say which came first. Those are counted and reported separately,
never booked as wins.

THE Z HERE IS NOT THE LIVE Z. `scripts/ofi_replay.py` sums OFI into
one-second buckets, so the window and baseline are counted in seconds;
the live engine counts them in book events. It is the same quantity
integrated differently, and a threshold found here does not transfer
to the live panel unchanged. Compare shapes, not numbers.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER. It counts outcomes.
"""

from __future__ import annotations

import csv
import math
from array import array
from dataclasses import dataclass, field

from . import ofi

TICK = 0.25                 # ES
NO_PRICE = 0                # sentinel: that second carried no print


# --------------------------------------------------------------- data

class Seconds:
    """The per-second file, columnar.

    A month is ~1.8 million rows. As objects that is gigabytes; as
    typed arrays it is tens of megabytes, and prices are held as
    INTEGER TICKS rather than floats -- exact, half the width, and it
    removes any question of a comparison failing by a rounding error
    at the exact moment a bracket is touched.
    """

    __slots__ = ("ts", "hi", "lo", "cl", "ofi", "path",
                 "bid_sz", "ask_sz", "delta")

    def __init__(self, path: str = "") -> None:
        self.path = path
        self.ts = array("q")
        self.hi = array("i")
        self.lo = array("i")
        self.cl = array("i")
        self.ofi = array("q")
        # The book and tape columns the non-OFI rules read. Four more
        # arrays over a 1.8M-row month is ~58MB on top of ~72MB --
        # still well inside what a replay machine has, and the
        # alternative is a second pass over the file per rule.
        self.bid_sz = array("i")
        self.ask_sz = array("i")
        self.delta = array("q")

    def __len__(self) -> int:
        return len(self.ts)

    def add(self, ts, hi, lo, cl, of, bid_sz=0, ask_sz=0,
            delta=0) -> None:
        self.ts.append(int(ts))
        self.hi.append(int(hi))
        self.lo.append(int(lo))
        self.cl.append(int(cl))
        self.ofi.append(int(of))
        self.bid_sz.append(int(bid_sz))
        self.ask_sz.append(int(ask_sz))
        self.delta.append(int(delta))

    @property
    def out_of_order(self) -> int:
        """Rows whose timestamp goes backwards.

        `resolve` walks forward from an entry assuming the index is
        time-ordered. Hand it two files concatenated out of order and
        it will "resolve" trades against prices from two years
        earlier and report the result with a straight face.
        """
        n = 0
        for a, b in zip(self.ts, self.ts[1:]):
            if b < a:
                n += 1
        return n

    @property
    def priced(self) -> int:
        """Seconds that carried a print. The rest have a book only."""
        return sum(1 for c in self.cl if c != NO_PRICE)

    def span_hours(self) -> float:
        if len(self) < 2:
            return 0.0
        return (self.ts[-1] - self.ts[0]) / 3600.0


def to_ticks(raw: str) -> int:
    """A price field to integer ticks. Empty means no print."""
    if raw is None or raw == "":
        return NO_PRICE
    return int(round(float(raw) / TICK))


def _num(raw) -> int:
    """A numeric field, with empty meaning absent rather than zero.

    `ofi_replay` leaves the book columns blank when the file carried
    no levels. Zero would be a real size of nothing, which the book
    rules would read as a one-sided market.
    """
    if raw is None or raw == "":
        return 0
    try:
        return int(round(float(raw)))
    except (TypeError, ValueError):
        return 0


def load(path: str, limit: int = 0) -> Seconds:
    """Read the file `ofi_replay.py` wrote.

    Empty price fields stay empty. They are seconds in which the book
    moved and nothing traded, and turning them into a price would
    hand this function a bar that never existed.
    """
    out = Seconds(path)
    with open(path, newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if limit and i >= limit:
                break
            out.add(row["ts"], to_ticks(row.get("high")),
                    to_ticks(row.get("low")), to_ticks(row.get("close")),
                    int(row.get("ofi") or 0),
                    _num(row.get("bid_sz")), _num(row.get("ask_sz")),
                    _num(row.get("delta")))
    return out


# ------------------------------------------------------------ outcome

WIN, LOSS, AMBIGUOUS, TIMEOUT, UNRESOLVED = (
    "win", "loss", "ambiguous", "timeout", "unresolved")


@dataclass
class Trade:
    i: int
    ts: int
    side: int                   # +1 long, -1 short
    entry: int                  # ticks
    outcome: str = UNRESOLVED
    ticks: float = 0.0          # gross, before cost
    held_s: int = 0


def resolve(sec: Seconds, i: int, side: int, target_ticks: int,
            stop_ticks: int, max_hold_s: int) -> Trade:
    """Walk forward until a bracket is touched, or time runs out.

    The entry is the close of second `i`; the walk starts at `i + 1`,
    because a bracket cannot be touched by the bar that created it.
    """
    entry = sec.cl[i]
    t = Trade(i=i, ts=sec.ts[i], side=side, entry=entry)
    if entry == NO_PRICE:
        return t

    up = entry + target_ticks * side
    down = entry - stop_ticks * side
    hi_target, lo_target = (up, down) if side > 0 else (down, up)

    n = len(sec)
    deadline = sec.ts[i] + max_hold_s
    j = i + 1
    last = entry
    while j < n and sec.ts[j] <= deadline:
        if sec.cl[j] != NO_PRICE:
            last = sec.cl[j]
            hit_t = (sec.hi[j] >= hi_target if side > 0
                     else sec.lo[j] <= lo_target)
            hit_s = (sec.lo[j] <= lo_target if side > 0
                     else sec.hi[j] >= hi_target)
            if hit_t and hit_s:
                # One second reached both. Which came first is not in
                # the file, and guessing favourably is how a backtest
                # invents an edge.
                t.outcome = AMBIGUOUS
                t.held_s = sec.ts[j] - sec.ts[i]
                return t
            if hit_t:
                t.outcome, t.ticks = WIN, float(target_ticks)
                t.held_s = sec.ts[j] - sec.ts[i]
                return t
            if hit_s:
                t.outcome, t.ticks = LOSS, float(-stop_ticks)
                t.held_s = sec.ts[j] - sec.ts[i]
                return t
        j += 1

    if j >= n and (n == 0 or sec.ts[n - 1] < deadline):
        # The file ENDED before the horizon. Booking this at the last
        # known price would fill the tail of every run with trades
        # that were never given their time.
        t.outcome = UNRESOLVED
        return t

    t.outcome = TIMEOUT
    t.ticks = float((last - entry) * side)
    t.held_s = max_hold_s
    return t


# -------------------------------------------------------------- score

@dataclass
class Result:
    arm: str = ""
    wins: int = 0
    losses: int = 0
    timeouts: int = 0
    ambiguous: int = 0
    unresolved: int = 0
    gross: float = 0.0

    @property
    def scored(self) -> int:
        """Trades with a known outcome. Ambiguous and unresolved are
        NOT scored -- they are reported so the reader can see how much
        of the sample was set aside."""
        return self.wins + self.losses + self.timeouts

    @property
    def win_rate(self) -> float | None:
        n = self.wins + self.losses
        return (self.wins / n) if n else None

    def net(self, cost: float) -> float:
        return self.gross - cost * self.scored

    def per_trade(self, cost: float) -> float | None:
        return (self.net(cost) / self.scored) if self.scored else None

    def to_dict(self, cost: float) -> dict:
        lo, hi = wilson(self.wins, self.wins + self.losses)
        return {
            "arm": self.arm, "scored": self.scored,
            "wins": self.wins, "losses": self.losses,
            "timeouts": self.timeouts, "ambiguous": self.ambiguous,
            "unresolved": self.unresolved,
            "win_rate": self.win_rate,
            "win_rate_lo": lo, "win_rate_hi": hi,
            "gross_ticks": round(self.gross, 1),
            "net_ticks": round(self.net(cost), 1),
            "per_trade_ticks": (None if self.per_trade(cost) is None
                                else round(self.per_trade(cost), 4)),
        }


def wilson(hits: int, n: int, z: float = 1.96) -> tuple:
    """A 95% interval on the win rate. A point estimate with no
    interval invites reading noise as signal."""
    if n <= 0:
        return (None, None)
    p = hits / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, centre - half), 4),
            round(min(1.0, centre + half), 4))


def break_even(target_ticks: float, stop_ticks: float,
               cost_ticks: float) -> float:
    """The win rate this bracket must clear to make nothing.

    Both sides pay the cost -- that is the whole correction. 4t/4t at
    one tick of cost needs 62.5%, not 55%.
    """
    denom = target_ticks + stop_ticks
    return (stop_ticks + cost_ticks) / denom if denom > 0 else 1.0


def score(trades, arm: str) -> Result:
    r = Result(arm=arm)
    for t in trades:
        if t.outcome == WIN:
            r.wins += 1
            r.gross += t.ticks
        elif t.outcome == LOSS:
            r.losses += 1
            r.gross += t.ticks
        elif t.outcome == TIMEOUT:
            r.timeouts += 1
            r.gross += t.ticks
        elif t.outcome == AMBIGUOUS:
            r.ambiguous += 1
        else:
            r.unresolved += 1
    return r



# --------------------------------------------------------------- rules

RULES = ("ofi", "agree", "book", "delta")


class Frames:
    """Which way each timeframe's current bar is pointing.

    Bars are built from the SAME per-second closes, so the 1m, 5m and
    15m views cannot disagree about where their boundaries are.
    Asking a venue for each separately is three chances for a 15m bar
    whose open is not the open of the 1m bar at its start.

    A bar still forming is counted, because that is what a rule would
    have seen at the time. Waiting for the close would let the
    backtest act on a bar that had not happened yet.
    """

    def __init__(self, spans=(60, 300, 900)) -> None:
        self.spans = tuple(int(s) for s in spans if int(s) > 0)
        self._open = {s: None for s in self.spans}
        self._start = {s: None for s in self.spans}

    def observe(self, ts: int, close: int) -> None:
        for s in self.spans:
            bar = ts - (ts % s)
            if self._start[s] != bar:
                self._start[s] = bar
                self._open[s] = close

    def ways(self, close: int) -> list:
        out = []
        for s in self.spans:
            o = self._open[s]
            out.append(0 if o is None or close == o
                       else (1 if close > o else -1))
        return out

    def agreed(self, close: int) -> int:
        """+1 or -1 only when EVERY frame points the same way."""
        ways = self.ways(close)
        if not ways or 0 in ways:
            return 0
        first = ways[0]
        return first if all(w == first for w in ways) else 0


def book_side(sec: "Seconds", i: int, ratio: float) -> int:
    """Which side of the touch is carrying more size.

    Resting size is the weakest evidence on the book -- it can be
    pulled the instant it is leaned on, and a print cannot be taken
    back. It is offered as a rule so it can be MEASURED against the
    controls rather than argued about.
    """
    b, a = sec.bid_sz[i], sec.ask_sz[i]
    if b <= 0 or a <= 0 or ratio <= 0:
        return 0                    # no book in that second
    if b >= a * ratio:
        return 1
    if a >= b * ratio:
        return -1
    return 0


class Roll:
    """A rolling sum over the last `window` seconds."""

    def __init__(self, window: int) -> None:
        from collections import deque
        self.window = max(1, int(window))
        self._q = deque(maxlen=self.window)
        self.total = 0

    def push(self, v: int) -> int:
        if len(self._q) == self._q.maxlen:
            self.total -= self._q[0]
        self._q.append(v)
        self.total += v
        return self.total

    @property
    def full(self) -> bool:
        return len(self._q) == self._q.maxlen


# --------------------------------------------------------------- run

def _coin(ts: int, salt: int = 0) -> int:
    """A deterministic coin flip from the timestamp.

    Deterministic so a control arm is reproducible: a run whose
    comparison moves when you run it again cannot settle anything.
    """
    x = (ts * 6364136223846793005 + 1442695040888963407 + salt) & ((1 << 64) - 1)
    x ^= x >> 33
    return 1 if (x & 1) else -1


@dataclass
class Settings:
    # Which rule decides the side. One name, or several joined by
    # "+", in which case EVERY named rule must point the same way --
    # an agreement rule over the rules themselves.
    rule: str = "ofi"
    z_enter: float = 1.5
    window: int = 60            # seconds summed into the current OFI
    baseline: int = 600         # window sums the z is measured against
    target_ticks: int = 16
    stop_ticks: int = 4
    max_hold_s: int = 900
    cost_ticks: float = 1.0
    train_fraction: float = 0.5
    cooldown_s: int = 60        # no re-entry while one is open
    frames: tuple = (60, 300, 900)   # the timeframes `agree` reads
    book_ratio: float = 2.0          # size imbalance `book` wants
    delta_window: int = 60           # seconds `delta` sums over

    @property
    def rules(self) -> list:
        """The rule names, split and cleaned.

        PLUS, SPACE OR COMMA all separate. A URL query decodes "+" as
        a space, so `?rule=ofi+agree` arrives here as "ofi agree" --
        accepting only "+" would make every combination silently
        collapse to one unknown rule over HTTP while working fine on
        the command line.
        """
        import re
        return [r.lower() for r in re.split(r"[+,\s]+", str(self.rule))
                if r.strip()]

    def unknown(self) -> list:
        return [r for r in self.rules if r not in RULES]

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["rules"] = self.rules
        d["break_even"] = round(break_even(self.target_ticks,
                                           self.stop_ticks,
                                           self.cost_ticks), 4)
        return d


def entries(sec: Seconds, st: Settings) -> list:
    """Where the rule would have entered, and which way.

    ONE PASS, EVERY RULE. The frame bars, the rolling delta and the
    OFI race are all advanced on every second whether or not the rule
    being tested reads them, so a rule cannot accidentally see a
    different history than another rule would have seen at the same
    moment.

    The z comes from `ofi.Race`, the same class the live engine uses,
    so the backtest cannot drift from the panel in how the number is
    computed -- only in what it is computed over (seconds here,
    events live).

    With several rules joined by "+", every one of them must point the
    same way. A rule that abstains -- no book in that second, a frame
    with no direction yet -- vetoes the entry rather than being
    treated as agreement.
    """
    want = st.rules
    race = ofi.Race(window=st.window, baseline=st.baseline)
    frames = Frames(st.frames)
    dlt = Roll(st.delta_window)

    out = []
    blocked_until = -1
    for i in range(len(sec)):
        close = sec.cl[i]
        rd = race.push(int(sec.ofi[i]), ts=float(sec.ts[i]))
        dsum = dlt.push(int(sec.delta[i]))
        if close != NO_PRICE:
            frames.observe(int(sec.ts[i]), close)

        if close == NO_PRICE or sec.ts[i] < blocked_until:
            continue

        sides = []
        for name in want:
            if name == "ofi":
                sides.append(0 if rd.z is None or abs(rd.z) < st.z_enter
                             else (1 if rd.z > 0 else -1))
            elif name == "agree":
                sides.append(frames.agreed(close))
            elif name == "book":
                sides.append(book_side(sec, i, st.book_ratio))
            elif name == "delta":
                sides.append(0 if not dlt.full or dsum == 0
                             else (1 if dsum > 0 else -1))
            else:
                sides.append(0)

        if not sides or 0 in sides:
            continue
        side = sides[0]
        if any(x != side for x in sides):
            continue

        out.append((i, side, rd.z))
        blocked_until = sec.ts[i] + st.cooldown_s
    return out


def run(sec: Seconds, settings: Settings | None = None) -> dict:
    """SIGNAL, MIRROR and CONTROL, split train / held out.

    The held-out number is the one to read. Nothing is searched here,
    so the split is not protecting against a parameter sweep -- it is
    there so that a result which only holds in the first half is
    visible as such.
    """
    st = settings or Settings()
    back = sec.out_of_order
    if back:
        # Refused, not warned. A result computed over a jumbled index
        # is not a degraded result, it is a meaningless one, and it
        # looks exactly like a real one on the page.
        return {"ok": False,
                "why": (f"the data goes backwards in time at "
                        f"{back:,} places, so this file was written "
                        f"by a replay that did not order its inputs. "
                        f"Re-run scripts/ofi_replay.py -- it now sorts "
                        f"the files by time and skips seconds that "
                        f"overlap a file already read"),
                "settings": st.to_dict(), "seconds": len(sec)}
    bad = st.unknown()
    if bad:
        return {"ok": False,
                "why": (f"unknown rule {', '.join(bad)} -- one of "
                        f"{', '.join(RULES)}, or several joined by '+'"),
                "settings": st.to_dict(), "seconds": len(sec)}
    sigs = entries(sec, st)
    if not sigs:
        return {"ok": False,
                "why": (f"the rule {st.rule!r} never fired over "
                        f"{len(sec):,} seconds"),
                "settings": st.to_dict(), "seconds": len(sec)}

    # SPLIT BY ENTRY COUNT, NOT BY WALL CLOCK. Market data has
    # calendar holes -- weekends, days you did not download -- and a
    # midpoint in TIME can fall inside one of them. Three days of ES
    # spanning seventeen split 2,508 / 200 that way, and every number
    # in the held-out column was then computed on 7% of the sample.
    #
    # Cutting at the k-th entry keeps the halves chronological, so
    # nothing leaks backwards, and makes them comparable in size,
    # which is what the comparison needs.
    k = int(len(sigs) * st.train_fraction)
    if 0 < k < len(sigs):
        cut = float(sec.ts[sigs[k][0]])
    else:
        cut = sec.ts[0] + (sec.ts[-1] - sec.ts[0]) * st.train_fraction
    arms = {"signal": [], "mirror": [], "control": []}
    for i, side, _z in sigs:
        arms["signal"].append(resolve(sec, i, side, st.target_ticks,
                                      st.stop_ticks, st.max_hold_s))
        arms["mirror"].append(resolve(sec, i, -side, st.target_ticks,
                                      st.stop_ticks, st.max_hold_s))
        arms["control"].append(resolve(sec, i, _coin(sec.ts[i]),
                                       st.target_ticks, st.stop_ticks,
                                       st.max_hold_s))

    out = {"ok": True, "settings": st.to_dict(),
           "seconds": len(sec), "priced_seconds": sec.priced,
           "hours": round(sec.span_hours(), 1),
           "entries": len(sigs),
           "split_at": cut,
           "coverage_hours": round(len(sec) / 3600.0, 1),
           "break_even": round(break_even(st.target_ticks, st.stop_ticks,
                                          st.cost_ticks), 4),
           "arms": {}}
    for name, ts_ in arms.items():
        out["arms"][name] = {
            "all": score(ts_, name).to_dict(st.cost_ticks),
            "train": score([t for t in ts_ if t.ts < cut],
                           name).to_dict(st.cost_ticks),
            "held_out": score([t for t in ts_ if t.ts >= cut],
                              name).to_dict(st.cost_ticks),
        }

    sig = out["arms"]["signal"]["held_out"]
    mir = out["arms"]["mirror"]["held_out"]
    ctl = out["arms"]["control"]["held_out"]
    out["verdict"] = verdict(sig, mir, ctl, out["break_even"],
                             st.cost_ticks)
    return out


def verdict(sig: dict, mir: dict, ctl: dict, be: float,
            cost: float) -> str:
    """What the held-out numbers support. Deliberately grudging."""
    if not sig["scored"]:
        return "no scored trades out of sample -- nothing to read"
    wr = sig["win_rate"]
    if wr is None:
        return "no resolved trades out of sample"
    bits = [f"signal {wr:.1%} (needs {be:.1%})"]
    if mir["win_rate"] is not None:
        bits.append(f"mirror {mir['win_rate']:.1%}")
    if ctl["win_rate"] is not None:
        bits.append(f"control {ctl['win_rate']:.1%}")
    head = ", ".join(bits)

    lo = sig["win_rate_lo"]
    if lo is not None and lo > be:
        return (f"{head} -- clears break-even with the interval above "
                f"it. Check it also beats mirror by more than noise.")
    if mir["win_rate"] is not None and sig["win_rate"] <= mir["win_rate"]:
        return (f"{head} -- the SAME trades taken the other way did at "
                f"least as well. That is the bracket's geometry, not "
                f"the signal.")
    return (f"{head} -- does not clear break-even out of sample at "
            f"{cost:g} ticks of cost.")
