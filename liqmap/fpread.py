"""Who is winning this footprint -- one answer, continuously.

    from liqmap.fpread import read_candle
    v = read_candle(candle, scale)
    v.side, v.score, v.strength, v.parts

WHAT THIS IS. The footprint already tags every diagonal imbalance
level by level, and finds the runs of them. What it does not do is
say, in one line, who is winning the bar overall. That is what this
adds: a single verdict, recomputed from scratch on every read, so it
is always describing the bar as it stands right now.

WHAT THIS IS NOT. It does not predict. Five sections of measurement
over 102 hours of ES found nothing in this family that calls where
price goes next, and nothing here changes that. This reports the
state of the auction as the footprint has recorded it, which is a
different and useful job -- the same job a speedometer does.

THE WEIGHTS ARE A CHOICE, NOT A DISCOVERY. Five facts go in, each
measured from the bar and each genuinely in the data. How much each
one counts is editorial: nobody has shown these weights are right,
because "right" would require an outcome to be right ABOUT, and that
outcome is what the study could not find. So every part is published
alongside the verdict with its own number. If the headline disagrees
with the part you trust, trust the part -- and change the weights,
which is why they are an argument.

THE FIVE PARTS, each running -1 (sellers) to +1 (buyers):

  delta      who paid. Signed volume over total volume. The most
             direct fact in a footprint and the hardest to argue
             with.

  flags      the count of diagonal imbalances each way. One tagged
             level is noise; the balance of them across a bar is the
             shape of the fight.

  stacks     runs of consecutive imbalances, weighted by run length.
             Three in a row is a different animal from three
             scattered, and this is the part that answers "overall"
             rather than "every level".

  location   WHERE each side's imbalances sit in the bar's range.
             Buy imbalances up near the high mean buyers are the
             ones operating in new ground; buy imbalances down at
             the low mean they are the ones being hit. Same flags,
             opposite stories, and only position tells them apart.

  follow     whether price went where the flow pushed. Strong buying
             that produced no upward travel is absorption, and this
             is the part that turns the verdict down when effort and
             result disagree.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from .footprint import (BUY_IMBALANCE, DEFAULT_IMBALANCE, ImbalanceConfig,
                        TickScale)

BUYERS, SELLERS, BALANCED = "buyers", "sellers", "balanced"

# How much each part counts toward the headline. An argument, not a
# finding -- see the module docstring. They sum to 1.0 so the score
# stays inside -1..+1 and can be read as a fraction.
WEIGHTS = {
    "delta": 0.30,
    "flags": 0.20,
    "stacks": 0.25,
    "location": 0.15,
    "follow": 0.10,
}

# Where the verdict stops being a shrug. Below `weak` the two sides
# are close enough that calling a winner is a coin flip dressed up.
BANDS = ((0.45, "strong"), (0.22, "moderate"), (0.08, "weak"))




@dataclass(frozen=True, slots=True)
class Pace:
    """How FAST each side is accumulating, not how much they have.

    The footprint's totals say who is ahead. They cannot say who is
    gaining, and those are different questions with different
    answers: buyers 4932 against sellers 4992 is a sellers' bar by
    the total, and a buyers' bar if buyers are adding forty contracts
    a second against the sellers' fifteen.

    So two leaders are reported, and the interesting state is when
    they disagree. `level` is who has traded more so far this bar.
    `pace` is who is trading faster right now. When pace points away
    from level, the side that is behind is taking control -- which is
    the thing that happens BEFORE the total flips, and the only
    reason to watch a rate at all.

    RATES ARE PER SECOND, ON A TIME CONSTANT, not per trade. A
    per-trade measure changes meaning with activity: twenty prints is
    a moment at the open and several minutes overnight, so the same
    market would read as two different ones depending on the hour.
    """

    buy_rate: float = 0.0       # contracts per second
    sell_rate: float = 0.0
    buy_total: int = 0
    sell_total: int = 0
    level: str = BALANCED       # who is ahead on the totals, at all
    control: str = BALANCED     # who is ahead by ENOUGH to count
    took_from: str = ""         # the side this one just took it off
    pace: str = BALANCED        # who is faster right now
    taking_control: str = ""    # set when pace disagrees with level
    ratio: float = 1.0          # faster side over slower
    tau_s: float = 0.0

    @property
    def diverging(self) -> bool:
        return bool(self.taking_control)

    @property
    def state(self) -> str:
        """"reversal", "pullback", "continuation", "holding", or "".

        Four readings off two numbers, and the difference between the
        middle two is the difference between a pause and a turn:

          reversal     the side that was in control is no longer the
                       higher number -- it has actually lost the bar
          pullback     the other side is gaining FASTER, but the side
                       in control is still the higher number
          continuation the side in control is also the faster one
          holding      in control, with neither side pulling away
          ""           nobody is clearly ahead yet

        A pullback that keeps going becomes a reversal, and that is
        the whole point of separating them: the first is counter-flow
        inside a move, the second is the move changing hands. Reading
        one as the other is the mistake this panel exists to avoid.
        """
        if self.control == BALANCED:
            return ""
        if self.took_from:
            return "reversal"
        if self.pace == BALANCED:
            return "holding"
        return "continuation" if self.pace == self.control else "pullback"

    @property
    def call(self) -> str:
        """"buy", "sell", or "" -- A READING OF NOW, NOT A FORECAST.

        This names the side the flow is currently on so it can be
        said in the language of a trade. It does NOT mean the next
        move goes that way, and nothing here should be read as a
        prediction that it will.

        That distinction is measured, not modest. cvd_study section
        10 took 73,778 samples over 102.5 hours of ES and asked
        whether these states precede price at 15s, 30s and 60s. They
        do not: the five states partition the sample exactly, so the
        apparent separation is forced by the balanced state sitting
        high and is not evidence about any of the others. The one
        cell that cleared the bar had a mean of -0.04 ticks against
        a round trip that costs a full tick.

        So: a description of the auction, in trade words, for a
        decision the trader is already making. Not a trigger. The
        panel says as much on screen, and that sentence stays.

        IT FOLLOWS `control`, NOT `pace`. It used to follow the
        rates, and that was wrong in a way worth recording: the rates
        go balanced inside their deadband whenever the two sides
        happen to trade at a similar speed for a few seconds, so a
        bar with sellers two thousand contracts ahead and in complete
        control read NO CALL the moment the pace evened up. The
        direction of a bar is who is winning it, which is a count.
        The rates say what is happening inside that, and that is
        `state` -- continuation or pullback -- not the side.
        """
        if self.control == BUYERS:
            return "buy"
        if self.control == SELLERS:
            return "sell"
        return ""

    def as_dict(self) -> dict:
        return {"buy_rate": round(self.buy_rate, 2),
                "sell_rate": round(self.sell_rate, 2),
                "buy_total": self.buy_total,
                "sell_total": self.sell_total,
                "level": self.level, "control": self.control,
                "took_from": self.took_from,
                "pace": self.pace,
                "taking_control": self.taking_control,
                "ratio": round(self.ratio, 3),
                "diverging": self.diverging,
                "call": self.call,
                "state": self.state,
                "tau_s": self.tau_s}


class PaceWatcher:
    """Contracts per second each way, on a wall-clock time constant.

    THE ESTIMATOR. Each side holds a decaying accumulator. Between
    trades it decays by exp(-dt / tau); on a trade of size s it gains
    s / tau. Steady flow of r contracts a second converges on exactly
    r, and the result depends only on elapsed time -- a burst of
    twenty prints and a trickle of twenty over the same seconds give
    the same rate, which a per-trade average cannot do.

    `dead` is how much faster one side must be before control is
    called, as a fraction. Without it the two rates cross constantly
    and the reading flickers between them several times a second.
    """

    __slots__ = ("tau", "dead", "min_lead", "min_share",
                 "_buy", "_sell", "_ts",
                 "buy_total", "sell_total", "_pace",
                 "_held", "_flip_ts")

    def __init__(self, tau_s: float = 10.0, dead: float = 0.20,
                 min_lead: int = 50, min_share: float = 0.02) -> None:
        if tau_s <= 0:
            raise ValueError("tau_s must be positive")
        if dead < 0:
            raise ValueError("dead must not be negative")
        if min_lead < 0 or min_share < 0:
            raise ValueError("the control threshold must not be negative")
        self.tau = float(tau_s)
        self.dead = float(dead)
        # HOW FAR AHEAD IS "IN CONTROL". Both must be cleared: a
        # count, because a bar's first few prints would otherwise
        # set the direction and flip it on the next one, and a
        # share, because 60 contracts means one thing in a bar of
        # 300 and nothing at all in a bar of 10,000.
        self.min_lead = int(min_lead)
        self.min_share = float(min_share)
        self.reset()

    def _control(self) -> str:
        """Who is ahead by enough to be called in control."""
        total = self.buy_total + self.sell_total
        gap = abs(self.buy_total - self.sell_total)
        if total <= 0 or gap < self.min_lead or gap < self.min_share * total:
            return BALANCED
        return BUYERS if self.buy_total > self.sell_total else SELLERS

    def reset(self) -> None:
        self._buy = 0.0
        self._sell = 0.0
        self._ts = 0.0
        self.buy_total = 0
        self.sell_total = 0
        self._pace = BALANCED
        self._held = BALANCED       # last side that actually held control
        self._flip_ts = 0.0         # when it last changed hands

    def new_bar(self) -> None:
        """A new footprint bar: clear the TOTALS, keep the RATES.

        THE TOTALS ARE PER BAR AND THIS IS NOT OPTIONAL. They are the
        same numbers the footprint prints at the top of the panel --
        `buyers 2,587 / sellers 2,710` -- and control is read off
        them. Left to accumulate across a whole session they drift
        into a different question entirely: after a few hours the
        side that is ahead for the DAY can be the side losing the
        bar in front of you, and the panel then names the wrong one
        with complete confidence. That is exactly what it did.

        The rates survive the boundary, because they are not counts.
        A ten-second measure zeroed every five minutes would read
        0/s at the top of every bar and be wrong there for as long
        as it took to refill.
        """
        self.buy_total = 0
        self.sell_total = 0
        self._held = BALANCED
        self._flip_ts = 0.0

    def _decay(self, ts: float) -> None:
        if self._ts <= 0.0:
            self._ts = ts
            return
        dt = ts - self._ts
        if dt <= 0.0:
            return
        f = math.exp(-dt / self.tau)
        self._buy *= f
        self._sell *= f
        self._ts = ts

    def on_trade(self, ts: float, size: int, aggressor: str) -> None:
        if size <= 0:
            return
        self._decay(ts)
        if aggressor == "buy":
            self._buy += size / self.tau
            self.buy_total += int(size)
        else:
            self._sell += size / self.tau
            self.sell_total += int(size)

    def reading(self, ts: float = 0.0) -> Pace:
        """The state as of `ts`, decaying quiet time properly.

        Passing the current time matters: without it a side that
        stopped trading a minute ago still reads at the rate it had
        when it stopped.
        """
        if ts > 0.0:
            self._decay(ts)
        b, s = self._buy, self._sell

        level = (BUYERS if self.buy_total > self.sell_total
                 else (SELLERS if self.sell_total > self.buy_total
                       else BALANCED))
        hi, loq = max(b, s), min(b, s)
        ratio = (hi / loq) if loq > 0 else (float("inf") if hi > 0 else 1.0)
        if loq > 0 and ratio < 1.0 + self.dead:
            pace = BALANCED
        elif b > s:
            pace = BUYERS
        elif s > b:
            pace = SELLERS
        else:
            pace = BALANCED
        self._pace = pace

        taking = ""
        if pace != BALANCED and level != BALANCED and pace != level:
            taking = pace

        # DID IT CHANGE HANDS. Only a move from one real side to the
        # other counts: coming out of BALANCED is control being
        # established for the first time this bar, which is not a
        # reversal of anything. `_held` therefore remembers the last
        # side that actually held it, not the last value of control.
        control = self._control()
        now = self._ts if self._ts > 0.0 else ts
        if control != BALANCED:
            if self._held != BALANCED and control != self._held:
                self._flip_ts = now
            self._held = control

        # A flip is an instant; the panel refreshes every two seconds
        # and a trader looks up when they look up. So it is reported
        # for `tau` after it happens -- the same window the rates are
        # measured over, rather than another number invented for the
        # purpose. After that the bar reads as whatever it has since
        # become: a continuation if the new side kept going, a
        # pullback if the old one started fighting back.
        took = ""
        if (control != BALANCED and self._flip_ts > 0.0
                and 0.0 <= (now - self._flip_ts) < self.tau):
            took = SELLERS if control == BUYERS else BUYERS

        return Pace(buy_rate=b, sell_rate=s,
                    buy_total=self.buy_total, sell_total=self.sell_total,
                    level=level, control=control, took_from=took,
                    pace=pace, taking_control=taking,
                    ratio=(ratio if ratio != float("inf") else 999.0),
                    tau_s=self.tau)


@dataclass(slots=True)
class Lead:
    """Who is ahead inside this bar, and whether they are still gaining.

    The footprint already totals aggressive buyers against aggressive
    sellers and starts again each bar. This watches the RUNNING
    difference and, more importantly, watches the high-water mark of
    it. The moment a side stops making new highs in its own lead is
    the moment it stopped winning, whatever the total still says.

    That distinction is the whole object. A bar can read "buyers
    +800" right to the close and have stopped being a buyers' bar
    four minutes earlier, when the lead peaked at +1400 and spent the
    rest of the bar bleeding back. The total hides it; the high-water
    mark does not.

    WHAT IT DOES NOT CLAIM. Whether a stall is followed by a reversal
    or a continuation is a question for measurement, not for this
    class. It records the stall and the price it happened at; the
    study says what tends to follow, and so far nothing in this
    family has predicted anything.
    """

    buy: int = 0
    sell: int = 0
    peak: int = 0               # best lead buyers ever held this bar
    peak_ts: float = 0.0
    peak_px: float = 0.0
    trough: int = 0             # best lead sellers ever held (<= 0)
    trough_ts: float = 0.0
    trough_px: float = 0.0
    last_px: float = 0.0
    stalled: bool = False
    stall_ts: float = 0.0
    stall_px: float = 0.0
    stall_side: str = ""
    # how far the leader has bled back from their own high-water mark
    give_back: int = 0

    @property
    def delta(self) -> int:
        return self.buy - self.sell

    @property
    def leader(self) -> str:
        d = self.delta
        return BUYERS if d > 0 else (SELLERS if d < 0 else BALANCED)

    @property
    def high_water(self) -> int:
        """The leader's own best lead, signed."""
        return self.peak if self.delta > 0 else self.trough

    def as_dict(self) -> dict:
        return {"buy": self.buy, "sell": self.sell, "delta": self.delta,
                "leader": self.leader, "peak": self.peak,
                "trough": self.trough, "high_water": self.high_water,
                "give_back": self.give_back, "stalled": self.stalled,
                "stall_side": self.stall_side,
                "stall_px": round(self.stall_px, 4),
                "stall_ts": self.stall_ts,
                "peak_px": round(self.peak_px, 4),
                "trough_px": round(self.trough_px, 4),
                "last_px": round(self.last_px, 4)}


class LeadWatcher:
    """Tracks the lead inside each bar and calls the moment it stops.

    `stall_frac` is of the leader's own high-water mark, so a bar
    that ran to +2000 has to give back more than one that ran to
    +200. A fixed contract threshold would fire constantly on a quiet
    bar and never on a busy one.

    `stall_min` keeps a trivially small lead from stalling at all --
    a lead of 12 giving back 5 is noise, not a turn.
    """

    __slots__ = ("stall_frac", "stall_min", "cur", "_bar_key", "history")

    def __init__(self, stall_frac: float = 0.35, stall_min: int = 50,
                 history: int = 200) -> None:
        if not 0.0 < stall_frac < 1.0:
            raise ValueError("stall_frac must be between 0 and 1")
        self.stall_frac = float(stall_frac)
        self.stall_min = int(stall_min)
        self.cur = Lead()
        self._bar_key = None
        self.history: deque = deque(maxlen=int(history))

    def reset(self) -> None:
        self.cur = Lead()
        self._bar_key = None
        self.history.clear()

    def on_bar(self, key) -> bool:
        """Start a new bar when `key` changes. Returns True on a roll."""
        if key == self._bar_key:
            return False
        if self._bar_key is not None:
            self.history.append(self.cur)
        self.cur = Lead()
        self._bar_key = key
        return True

    def on_trade(self, ts: float, px: float, size: int,
                 aggressor: str) -> None:
        c = self.cur
        if size <= 0:
            return
        if aggressor == "buy":
            c.buy += int(size)
        else:
            c.sell += int(size)
        c.last_px = float(px)
        d = c.delta
        if d > c.peak:
            c.peak, c.peak_ts, c.peak_px = d, ts, px
        if d < c.trough:
            c.trough, c.trough_ts, c.trough_px = d, ts, px

        # Has the leader bled back from their own best?
        hw = c.high_water
        if hw == 0:
            c.give_back = 0
            return
        c.give_back = abs(hw) - abs(d) if (hw > 0) == (d > 0) else abs(hw)
        if c.stalled:
            return
        if abs(hw) < self.stall_min:
            return
        if c.give_back >= self.stall_frac * abs(hw):
            c.stalled = True
            c.stall_ts = ts
            c.stall_px = px
            c.stall_side = BUYERS if hw > 0 else SELLERS


@dataclass(frozen=True, slots=True)
class Verdict:
    """Who is winning, how strongly, and what that was built from."""

    side: str = BALANCED
    score: float = 0.0              # -1 sellers .. +1 buyers
    strength: str = "none"
    parts: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)
    # the raw counts behind the parts, so nothing is unexplained
    buy_flags: int = 0
    sell_flags: int = 0
    buy_stacks: int = 0
    sell_stacks: int = 0
    delta: int = 0
    volume: int = 0
    levels: int = 0
    span_ticks: int = 0
    note: str = ""

    def as_dict(self) -> dict:
        return {"side": self.side, "score": round(self.score, 4),
                "strength": self.strength,
                "parts": {k: round(v, 4) for k, v in self.parts.items()},
                "weights": self.weights,
                "buy_flags": self.buy_flags, "sell_flags": self.sell_flags,
                "buy_stacks": self.buy_stacks,
                "sell_stacks": self.sell_stacks,
                "delta": self.delta, "volume": self.volume,
                "levels": self.levels, "span_ticks": self.span_ticks,
                "note": self.note}


def _band(score: float) -> str:
    a = abs(score)
    for cut, name in BANDS:
        if a >= cut:
            return name
    return "none"


def _ratio(up: float, down: float) -> float:
    """Signed share, -1..+1, and 0 when there is nothing of either."""
    t = up + down
    return ((up - down) / t) if t > 0 else 0.0


def read_candle(candle, scale: TickScale | None = None,
                cfg: ImbalanceConfig = DEFAULT_IMBALANCE,
                run: int = 3,
                weights: dict | None = None) -> Verdict:
    """One verdict for this bar, computed fresh from the bar itself.

    Takes a FootprintCandle -- live or closed, they are the same type
    -- so a forming bar and a finished one are read identically. No
    state is carried between calls, which is what makes it safe to
    poll at any rate and from any thread.
    """
    w = dict(WEIGHTS)
    if weights:
        w.update(weights)

    if candle is None or candle.empty:
        return Verdict(weights=w, note="no prints in this bar yet")

    rows = candle.levels
    lo = min(rows)
    hi = max(rows)
    span = hi - lo

    flags = candle.imbalances(cfg)
    buy_f = [f for f in flags if f.side == BUY_IMBALANCE]
    sell_f = [f for f in flags if f.side != BUY_IMBALANCE]

    runs = candle.stacked(cfg, run=run)
    buy_runs = [r for r in runs if r[0] == BUY_IMBALANCE]
    sell_runs = [r for r in runs if r[0] != BUY_IMBALANCE]
    # Weighted by LENGTH: a run of six is not two runs of three.
    buy_len = sum(r[2] - r[1] + 1 for r in buy_runs)
    sell_len = sum(r[2] - r[1] + 1 for r in sell_runs)

    vol = candle.total_volume
    parts = {
        "delta": (candle.delta / vol) if vol > 0 else 0.0,
        "flags": _ratio(len(buy_f), len(sell_f)),
        "stacks": _ratio(buy_len, sell_len),
        "location": 0.0,
        "follow": 0.0,
    }

    # WHERE the imbalances sit. Position runs 0 at the bar's low to 1
    # at its high; a side operating up near the high is the side with
    # the initiative. On a one-tick bar there is no range to speak of
    # and this part stays out of it rather than inventing a position.
    if span > 0 and (buy_f or sell_f):
        def pos(fs):
            if not fs:
                return None
            return sum((f.price_ticks - lo) / span for f in fs) / len(fs)
        bp, sp = pos(buy_f), pos(sell_f)
        # CONTRAST ONLY. With flags on both sides, "who is operating
        # higher in the range" is a real comparison. With flags on one
        # side it is not: sell imbalances low in the bar and sell
        # imbalances high in the bar are both sellers, and squeezing
        # that into a signed number invented a reading -- an early
        # version scored a seller-dominated bar's geometry at +0.25,
        # bullish, for exactly that reason. One-sided bars get nothing
        # here and let `flags` carry the direction, which it already
        # does correctly.
        if bp is not None and sp is not None:
            parts["location"] = max(-1.0, min(1.0, (bp - sp) * 2.0))

    # Did price go where the flow pushed? Close against open over the
    # bar's own range. Strong buying with no upward travel is
    # absorption, and this is the part that says so.
    if span > 0:
        travel = (candle.close_ticks - candle.open_ticks) / span
        parts["follow"] = max(-1.0, min(1.0, travel))

    score = sum(parts[k] * w.get(k, 0.0) for k in parts)
    score = max(-1.0, min(1.0, score))
    strength = _band(score)
    side = (BALANCED if strength == "none"
            else (BUYERS if score > 0 else SELLERS))

    # ABSORPTION CAPS THE HEADLINE. One side paying hard while price
    # refuses to go anywhere is the clearest thing a footprint shows,
    # and calling it "strong buyers" because the delta is large reads
    # the effort and ignores the result. The parts stay visible; it is
    # the one-word summary that gets held back.
    absorbed = (span > 0 and abs(parts["delta"]) >= 0.15
                and (abs(parts["follow"]) < 0.15
                     or parts["delta"] * parts["follow"] < 0))

    note = ""
    if not flags:
        note = "no diagonal imbalances yet; the verdict is delta alone"
    elif span == 0:
        note = "one-tick bar: position and follow-through have no range"
    if absorbed:
        if strength == "strong":
            strength = "moderate"
        who = "buyers" if parts["delta"] > 0 else "sellers"
        note = (f"{who} are paying and price is not following -- "
                f"effort without result, which is what absorption "
                f"looks like")
    elif not note and (buy_f and not sell_f or sell_f and not buy_f):
        note = "imbalances on one side only; position is not counted"

    return Verdict(
        side=side, score=score, strength=strength, parts=parts, weights=w,
        buy_flags=len(buy_f), sell_flags=len(sell_f),
        buy_stacks=len(buy_runs), sell_stacks=len(sell_runs),
        delta=candle.delta, volume=vol, levels=len(rows),
        span_ticks=span, note=note)


class FootprintReader:
    """The verdict on the live bar, and on the bars behind it.

    Holds no accumulated state of its own: every verdict is computed
    from the candle it describes. A reader that accumulated would
    drift from the footprint it is supposed to be reading, and the
    whole point is that it always agrees with the chart.
    """

    def __init__(self, cfg: ImbalanceConfig = DEFAULT_IMBALANCE,
                 run: int = 3, weights: dict | None = None,
                 history: int = 40) -> None:
        self.cfg = cfg
        self.run = int(run)
        self.weights = dict(WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.history = int(history)

    @staticmethod
    def _forming(stream):
        """The bar still open, whatever the caller calls it.

        FootprintStream keeps it on its engine as `_engine.current`;
        a FootprintEngine has `current` directly; a test double may
        simply expose `live`. Guessing one name and getting a silent
        "no live bar" is how a reader ends up quietly describing
        nothing -- which is exactly what happened first time.
        """
        for path in ("live", "current"):
            bar = getattr(stream, path, None)
            if bar is not None:
                return bar
        eng = getattr(stream, "_engine", None)
        return getattr(eng, "current", None) if eng is not None else None

    def live(self, stream) -> Verdict:
        """The bar still forming."""
        bar = self._forming(stream)
        if bar is None:
            return Verdict(weights=self.weights, note="no live bar")
        return read_candle(bar, cfg=self.cfg, run=self.run,
                           weights=self.weights)

    def closed(self, stream) -> list:
        """The finished bars, oldest first."""
        hist = list(getattr(stream, "history", ()) or ())
        out = []
        for bar in hist[-self.history:]:
            out.append(read_candle(bar, cfg=self.cfg, run=self.run,
                                   weights=self.weights))
        return out

    def as_dict(self, stream) -> dict:
        now = self.live(stream)
        past = self.closed(stream)
        # How long the same side has held across closed bars. A side
        # that has won six bars running is a different statement from
        # one that has just taken this one.
        held = 0
        for v in reversed(past):
            if v.side == now.side and now.side != BALANCED:
                held += 1
            else:
                break
        return {"now": now.as_dict(),
                "history": [v.as_dict() for v in past],
                "bars_held": held,
                "weights": self.weights,
                "run": self.run}
