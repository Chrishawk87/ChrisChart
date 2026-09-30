"""Module 26 -- key levels and what happens when price reaches them.

A trap at a random price is noise. A trap at yesterday's high is fuel. This
module is the bridge: it knows where the levels are, and it knows what state
each one is in, so the tape modules can be asked the right question at the
right price.

THE RULE THAT THE WHOLE MODULE TURNS ON

A wick is a touch. A break needs a confirmed close.

That distinction is not pedantry, it is the difference between a reversal
signal and a continuation signal at the identical price. Price poking through
yesterday's high and closing back under it is rejection -- the most common
single event at a key level. Treating that as a break inverts the read on the
bars where being right matters most. Every state transition below goes
through `_resolve`, which applies the rule once, so it cannot drift apart
between level kinds.

WHICH SIDE IS "THE ORIGINAL SIDE"

Derived from where price actually was when the level became active, never
assumed from the level's name. It is tempting to hard-code that PDH is above
and PDL is below, and it is wrong roughly every time there is a gap -- open
above yesterday's high and PDH is now support, with "broken above" already
true before the session starts. The first observation sets the side and the
level is read relative to that.

SESSIONS ARE IN EXCHANGE TIME, NOT UTC

ES trades 18:00 ET to 17:00 ET the next day, with the regular session from
09:30 to 16:00 ET. Those boundaries move against UTC twice a year, so all
session logic runs in America/New_York and only converts at the edges. A
fixed UTC offset here would silently mis-slice every level for the weeks
between the US and European DST switches.

WHAT THIS MODULE DOES NOT DO

It does not read the tape. When price approaches a level it raises an
`Approach`, and the DOM and footprint modules answer what is happening there.
Keeping those apart means a level can be tested on bar data alone -- which is
why this module is verifiable before any depth data has been bought.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)
GLOBEX_OPEN = time(18, 0)          # session start for the NEXT trade date
SETTLEMENT = time(17, 0)           # daily halt

INITIAL_BALANCE = timedelta(minutes=60)

# Bars of regular session before session VWAP is worth publishing.
VWAP_MIN_BARS = 5

# ES. Other instruments pass their own; nothing below hard-codes it.
ES_TICK = 0.25

# How close counts as "approaching". Two ticks, per the spec.
APPROACH_TICKS = 2.0

# A level goes quiet after this many consecutive bars entirely clear of it.
# Bars rather than minutes so it holds on any interval.
EXPIRE_BARS = 20

# How far is "entirely clear" -- in ticks, from the level. Ten ES points.
# The first draft used three points, which expired PDH while price was still
# close enough to trade back to it inside a single bar. A level is behind you
# when price has spent real time a long way from it, not when it is briefly
# out of arm's reach.
EXPIRE_TICKS = 40.0


class Kind(str, Enum):
    """The levels tracked. Values are what the UI and the ledger store."""

    PDH = "PDH"                    # prior day RTH high
    PDL = "PDL"                    # prior day RTH low
    ONH = "ONH"                    # overnight high
    ONL = "ONL"                    # overnight low
    VWAP = "VWAP"                  # session VWAP, anchored at RTH open
    IBH = "IBH"                    # initial balance high, first 60m of RTH
    IBL = "IBL"                    # initial balance low
    PRIOR_CLOSE = "PRIOR_CLOSE"    # prior RTH close, for gap measurement


class State(str, Enum):
    SET = "SET"
    TOUCHED = "TOUCHED"
    BROKEN_ABOVE = "BROKEN_ABOVE"
    BROKEN_BELOW = "BROKEN_BELOW"
    RECLAIMED = "RECLAIMED"
    EXPIRED = "EXPIRED"


class Outcome(str, Enum):
    """What the interaction at a level amounted to, once it resolved."""

    REJECTION = "REJECTION"        # touched and turned away
    BREAK_HOLD = "BREAK_HOLD"      # closed through and stayed through
    FAILED_BREAK = "FAILED_BREAK"  # closed through, then closed back
    PENDING = "PENDING"            # still at the level


class Phase(str, Enum):
    OVERNIGHT = "overnight"        # 18:00 ET -> 09:30 ET
    RTH = "rth"                    # 09:30 -> 16:00 ET
    POST = "post"                  # 16:00 -> 17:00 ET
    CLOSED = "closed"              # 17:00 -> 18:00 ET, and weekends


# --------------------------------------------------------------- sessions

def et(ts: float) -> datetime:
    """Epoch seconds to exchange-local time."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(ET)


def trade_date(ts: float) -> date:
    """The CME trade date a timestamp belongs to.

    The session that opens Sunday at 18:00 ET is Monday's. So anything at or
    after the Globex open belongs to the next calendar day; everything else,
    including the 16:00-17:00 post-settlement hour, belongs to the current
    one.
    """
    local = et(ts)
    if local.time() >= GLOBEX_OPEN:
        return local.date() + timedelta(days=1)
    return local.date()


def phase(ts: float) -> Phase:
    """Which part of the session a timestamp is in."""
    local = et(ts)
    t = local.time()
    if local.weekday() == 5:                          # Saturday
        return Phase.CLOSED
    if local.weekday() == 6 and t < GLOBEX_OPEN:      # Sunday before reopen
        return Phase.CLOSED
    if local.weekday() == 4 and t >= SETTLEMENT:      # Friday after settle
        return Phase.CLOSED
    if RTH_OPEN <= t < RTH_CLOSE:
        return Phase.RTH
    if RTH_CLOSE <= t < SETTLEMENT:
        return Phase.POST
    if SETTLEMENT <= t < GLOBEX_OPEN:
        return Phase.CLOSED
    return Phase.OVERNIGHT


def rth_open_ts(day: date) -> float:
    """Epoch seconds of the RTH open on a trade date."""
    return datetime.combine(day, RTH_OPEN, tzinfo=ET).timestamp()


# ----------------------------------------------------------------- levels

@dataclass(frozen=True)
class Transition:
    """One state change, kept so a level's history can be replayed."""

    ts: float
    frm: State
    to: State
    price: float               # the bar close that caused it
    note: str = ""


@dataclass
class Level:
    """One tracked price, and everything that has happened to it."""

    kind: Kind
    price: float
    session: date
    state: State = State.SET
    above_at_set: bool | None = None    # was price above this level initially?
    set_ts: float = 0.0
    touches: int = 0
    breaks: int = 0
    reclaims: int = 0
    last_ts: float = 0.0
    clear_bars: int = 0                 # consecutive bars well away from it
    history: list[Transition] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.state is not State.EXPIRED

    @property
    def outcome(self) -> Outcome:
        """What this level's interaction amounted to.

        Read off the counters rather than the current state, because the
        current state cannot distinguish "never broken" from "broken and
        reclaimed" -- and those are opposite readings.
        """
        if self.reclaims > 0:
            return Outcome.FAILED_BREAK
        if self.breaks > 0:
            return Outcome.BREAK_HOLD
        if self.touches > 0:
            return Outcome.REJECTION
        return Outcome.PENDING

    def distance(self, price: float, tick: float = ES_TICK) -> float:
        """Signed distance in ticks. Positive means price is above."""
        return (price - self.price) / tick if tick > 0 else 0.0

    def _go(self, to: State, ts: float, px: float, note: str = "") -> None:
        if to is self.state:
            return
        self.history.append(Transition(ts, self.state, to, px, note))
        self.state = to


@dataclass(frozen=True)
class Approach:
    """Price has come within range of an active level.

    This is the hand-off to the tape modules: it says where to look and what
    question to ask, and carries no opinion of its own about what the book is
    doing there.
    """

    ts: float
    kind: Kind
    level: float
    price: float
    ticks_away: float
    state: State
    approached_from_above: bool


@dataclass
class Bar:
    """The minimum a level needs. Any OHLCV candle satisfies it."""

    ts: float
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


def _as_bar(c) -> Bar:
    """Accept this module's Bar, structure.Candle, or a plain dict."""
    if isinstance(c, Bar):
        return c
    if isinstance(c, dict):
        return Bar(ts=float(c.get("ts", c.get("t", 0.0))),
                   open=float(c.get("open", c.get("o", 0.0))),
                   high=float(c.get("high", c.get("h", 0.0))),
                   low=float(c.get("low", c.get("l", 0.0))),
                   close=float(c.get("close", c.get("c", 0.0))),
                   volume=float(c.get("volume", c.get("v", 0.0))))
    return Bar(ts=float(c.ts), open=float(c.open), high=float(c.high),
               low=float(c.low), close=float(c.close),
               volume=float(getattr(c, "volume", 0.0)))


# ------------------------------------------------------------- the book

class LevelBook:
    """Tracks every key level for the current trade date.

    Fed bars in order. Levels are published when the data that defines them
    is complete -- PDH/PDL when the prior RTH session ends, ONH/ONL at the
    RTH open, IBH/IBL an hour later -- and never before, because a level
    computed from a session that has not finished is a look-ahead leak
    wearing a level's name.
    """

    def __init__(self, tick: float = ES_TICK,
                 approach_ticks: float = APPROACH_TICKS,
                 expire_bars: int = EXPIRE_BARS,
                 expire_ticks: float = EXPIRE_TICKS):
        self.tick = tick
        self.approach_ticks = approach_ticks
        self.expire_bars = expire_bars
        self.expire_ticks = expire_ticks

        self.levels: dict[Kind, Level] = {}
        self.session: date | None = None
        self.approaches: list[Approach] = []

        # Accumulators for the session in progress.
        self._rth: list[Bar] = []
        self._overnight: list[Bar] = []
        self._pv = 0.0                     # sum(typical price * volume)
        self._vol = 0.0
        self._prior_rth: list[Bar] = []
        self._ib_done = False

    # -- publishing ------------------------------------------------------

    def _publish(self, kind: Kind, price: float, ts: float) -> None:
        if price <= 0:
            return
        self.levels[kind] = Level(kind=kind, price=price,
                                  session=self.session or trade_date(ts),
                                  set_ts=ts, last_ts=ts)

    def _roll(self, day: date, ts: float) -> None:
        """A new trade date has begun.

        The prior session's RTH bars become PDH/PDL/PRIOR_CLOSE, and
        everything session-scoped resets. Levels do not carry across the
        roll: yesterday's IBH is not a level today, it is history.
        """
        if self._rth:
            self._prior_rth = list(self._rth)

        self.session = day
        self.levels = {}
        self._rth = []
        self._overnight = []
        self._pv = 0.0
        self._vol = 0.0
        self._ib_done = False

        if self._prior_rth:
            self._publish(Kind.PDH, max(b.high for b in self._prior_rth), ts)
            self._publish(Kind.PDL, min(b.low for b in self._prior_rth), ts)
            self._publish(Kind.PRIOR_CLOSE, self._prior_rth[-1].close, ts)

    # -- the wick/close rule --------------------------------------------

    def _resolve(self, lv: Level, bar: Bar) -> None:
        """Apply the one rule, in one place.

        Order matters. A confirmed close beyond is checked first, because a
        bar that closes through also touched on its way, and reporting that
        as a touch would lose the break entirely.
        """
        if lv.above_at_set is None:
            # First look decides which side we are reading from. Use the
            # open, not the close: the close has already been influenced by
            # whatever happened at the level during this very bar.
            lv.above_at_set = bar.open > lv.price

        started_above = lv.above_at_set
        broke = (State.BROKEN_BELOW if started_above else State.BROKEN_ABOVE)

        closed_through = (bar.close < lv.price if started_above
                          else bar.close > lv.price)
        closed_back = (bar.close > lv.price if started_above
                       else bar.close < lv.price)
        wicked = bar.low <= lv.price <= bar.high

        if lv.state in (State.BROKEN_ABOVE, State.BROKEN_BELOW):
            if closed_back:
                lv.reclaims += 1
                lv._go(State.RECLAIMED, bar.ts, bar.close,
                       "confirmed close back on the original side")
            return

        if closed_through:
            lv.breaks += 1
            lv._go(broke, bar.ts, bar.close, "confirmed close beyond")
            return

        if wicked:
            lv.touches += 1
            lv._go(State.TOUCHED, bar.ts, bar.close,
                   "traded through intrabar, closed back")

    def _age(self, lv: Level, bar: Bar) -> None:
        """Expire a level price has plainly left behind."""
        near = (abs(bar.high - lv.price) <= self.expire_ticks * self.tick or
                abs(bar.low - lv.price) <= self.expire_ticks * self.tick or
                bar.low <= lv.price <= bar.high)
        lv.clear_bars = 0 if near else lv.clear_bars + 1
        if lv.clear_bars >= self.expire_bars:
            lv._go(State.EXPIRED, bar.ts, bar.close,
                   f"{lv.clear_bars} bars clear of it")

    # -- the main entry point -------------------------------------------

    def observe(self, candle) -> list[Approach]:
        """Feed one closed bar. Returns any approaches it produced.

        Only closed bars. Feeding a bar still forming would let a level break
        and un-break within the same bar, and would put the engine's state a
        few seconds into the future of what it is allowed to know.
        """
        bar = _as_bar(candle)
        day = trade_date(bar.ts)
        ph = phase(bar.ts)

        if self.session is None or day != self.session:
            self._roll(day, bar.ts)

        if ph is Phase.OVERNIGHT:
            self._overnight.append(bar)
        elif ph is Phase.RTH:
            if not self._rth and self._overnight:
                # First RTH bar: the overnight range is now complete.
                self._publish(Kind.ONH,
                              max(b.high for b in self._overnight), bar.ts)
                self._publish(Kind.ONL,
                              min(b.low for b in self._overnight), bar.ts)
            self._rth.append(bar)

            typical = (bar.high + bar.low + bar.close) / 3.0
            self._pv += typical * bar.volume
            self._vol += bar.volume
            # VWAP is not published for the first few bars of a session.
            # On bar one it IS the price, by definition, so it sits zero
            # ticks away and fires an approach at every single session open
            # -- a phantom key-level test, every day, before the market has
            # done anything. It only starts meaning something once enough
            # volume has traded for it to diverge from spot.
            if self._vol > 0 and len(self._rth) >= VWAP_MIN_BARS:
                self._publish(Kind.VWAP, self._pv / self._vol, bar.ts)

            if not self._ib_done:
                opened = rth_open_ts(day)
                if bar.ts >= opened + INITIAL_BALANCE.total_seconds():
                    ib = [b for b in self._rth
                          if b.ts < opened + INITIAL_BALANCE.total_seconds()]
                    if ib:
                        self._publish(Kind.IBH,
                                      max(b.high for b in ib), bar.ts)
                        self._publish(Kind.IBL,
                                      min(b.low for b in ib), bar.ts)
                    self._ib_done = True

        found: list[Approach] = []
        for lv in self.levels.values():
            if not lv.active:
                continue
            lv.last_ts = bar.ts

            # VWAP moves every bar, so its "side at set" has to move with it
            # or it reads as permanently broken the moment price crosses.
            if lv.kind is Kind.VWAP and lv.state in (
                    State.BROKEN_ABOVE, State.BROKEN_BELOW, State.RECLAIMED):
                lv.above_at_set = None

            self._resolve(lv, bar)
            self._age(lv, bar)

            gap = abs(bar.close - lv.price) / self.tick
            if gap <= self.approach_ticks and lv.active:
                ap = Approach(ts=bar.ts, kind=lv.kind, level=lv.price,
                              price=bar.close, ticks_away=gap,
                              state=lv.state,
                              approached_from_above=bar.close >= lv.price)
                found.append(ap)
                self.approaches.append(ap)

        del self.approaches[:-500]
        return found

    def extend(self, candles: Iterable) -> list[Approach]:
        out: list[Approach] = []
        for c in candles:
            out.extend(self.observe(c))
        return out

    # -- reading it ------------------------------------------------------

    def active(self) -> list[Level]:
        """Live levels, nearest to the last seen price first."""
        live = [lv for lv in self.levels.values() if lv.active]
        return sorted(live, key=lambda lv: lv.price)

    def nearest(self, price: float, above: bool | None = None) -> Level | None:
        live = [lv for lv in self.levels.values() if lv.active]
        if above is True:
            live = [lv for lv in live if lv.price > price]
        elif above is False:
            live = [lv for lv in live if lv.price < price]
        return min(live, key=lambda lv: abs(lv.price - price)) if live else None

    def gap(self) -> float | None:
        """Today's open against the prior RTH close, in ticks.

        None until both exist. A large gap changes which side of PDH/PDL
        price starts on, which is the case the side-at-set logic exists for.
        """
        prior = self.levels.get(Kind.PRIOR_CLOSE)
        if prior is None or not self._rth:
            return None
        return (self._rth[0].open - prior.price) / self.tick

    def to_dict(self, price: float = 0.0) -> dict:
        """What the panel renders."""
        return {
            "session": self.session.isoformat() if self.session else None,
            "gap_ticks": self.gap(),
            "levels": [
                {
                    "kind": lv.kind.value,
                    "price": round(lv.price, 4),
                    "state": lv.state.value,
                    "outcome": lv.outcome.value,
                    "touches": lv.touches,
                    "breaks": lv.breaks,
                    "reclaims": lv.reclaims,
                    "ticks_away": (round(lv.distance(price, self.tick), 2)
                                   if price > 0 else None),
                }
                for lv in self.active()
            ],
        }
