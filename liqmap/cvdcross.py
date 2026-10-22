"""Book leads, tape confirms: two CVDs crossing, and whether it held.

    from liqmap.cvdcross import CVDCross
    r = CVDCross()
    r.on_book(bid_px, bid_sz, ask_px, ask_sz, ts)
    r.on_trade(ts, "buy", size, price)
    s = r.snapshot()
    s.book_z, s.tape_z, s.side, s.ok, s.held_s

THE TWO LINES

  PRIMARY    book CVD -- net size ADDED or PULLED at the inside quotes.
             The reader. It says which way the resting side is leaning
             before anything has to trade.

  SECONDARY  traded CVD -- aggressor-signed volume, smoothed on
             WALL-CLOCK TIME. The confirmation. It says the tape has
             caught up with the lean.

             Time-based, not trade-based, and the difference is not
             cosmetic. An EMA advanced once per trade has a time
             constant that swings with activity: ES prints twenty
             trades in a fraction of a second at the open and takes
             minutes to print twenty overnight, so a "20-trade" smooth
             hugs the raw line during bursts (crossings everywhere) and
             lags badly when quiet (crossings never). Smoothing on
             elapsed seconds gives the same smoothing at 03:00 as at
             09:30:

                 a = 1 - exp(-dt / tau);  s += a * (x - s)

They are two INDEPENDENT measurements of the same market. That is the
whole reason a cross between them is worth anything: a line smoothed
from itself cannot confirm itself.

WHY BOTH ARE STANDARDISED

Raw, these two do not live on the same scale -- resting size at the
inside churns far harder than volume trades, so a raw plot puts one
line across the top and flattens the other into the axis, and they
never cross. The usual dodge is a second y-axis, which is a lie that
makes any two shapes look related.

Instead both are expressed in standard deviations of their OWN recent
behaviour over the SAME rolling window. That makes them dimensionless
and genuinely comparable, one honest axis, and a crossing is a real
event: the tape moving through where the book already was.

WHAT THE BOOK LINE MEASURES

Size changes at a STANDING quote only. When the inside price steps, the
reference is re-based at the new level and nothing is counted -- because
comparing size at a new price against size at the old one is not a
change in liquidity, it is a change of subject. Counting it makes the
line add contracts on every tick of drift with nothing traded, which
turns the reader into a price odometer that shadows the thing it is
supposed to lead. `price_move="carry"` restores that older behaviour.

STABILITY

Agreement alone is a coin flip at any instant. What is reported is how
LONG the two have agreed, how often they have flipped, and whether the
flipping is bad enough to call it chop -- the same hysteresis-and-dwell
shape the book-direction read already uses, for the same reason: a
signal that changes its mind every few hundred milliseconds is not a
direction, it is noise with a sign.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER. It reports a reading.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass

from . import levels


def _median(xs: list) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    n = len(ys)
    mid = n // 2
    return ys[mid] if n % 2 else (ys[mid - 1] + ys[mid]) / 2.0


BUY, SELL = "buy", "sell"
UP, DOWN, FLAT = "up", "down", "flat"

# "Infinity" for an empty level would poison every statistic that sees
# it: one non-finite sample and the running mean stays non-finite for
# the session, every comparison turns False, and the reader goes quiet
# with nothing to say why. Zero depth is routine on ES.
LCR_MAX = 999.0


class OrderBookDeltaTracker:
    """Book CVD: net size added or pulled at the inside quotes.

    A PRICE STEP IS NOT A SIZE CHANGE. With `price_move="rebaseline"`
    the quote moving re-bases the reference and counts nothing, so the
    line measures liquidity and only liquidity. With "carry" a better
    bid counts its new size as arriving and a worse ask counts its old
    size as leaving, which also makes a quote walking one tick with
    identical size add (bid_sz + ask_sz) per tick with nothing traded.
    """

    __slots__ = ("_bid_px", "_bid_sz", "_ask_px", "_ask_sz", "mode",
                 "book_cvd", "bid_delta", "ask_delta", "updates")

    def __init__(self, price_move: str = "rebaseline") -> None:
        if price_move not in ("rebaseline", "carry"):
            raise ValueError("price_move must be 'rebaseline' or 'carry'")
        self.mode = price_move
        self.reset()

    def reset(self) -> None:
        self._bid_px = 0.0
        self._bid_sz = 0.0
        self._ask_px = 0.0
        self._ask_sz = 0.0
        self.book_cvd = 0.0
        self.bid_delta = 0.0
        self.ask_delta = 0.0
        self.updates = 0

    def _side_delta(self, px, sz, prev_px, prev_sz, better) -> float:
        if prev_px <= 0.0:                      # first sight of this side
            return 0.0
        if px == prev_px:
            return sz - prev_sz
        if self.mode == "rebaseline":
            return 0.0                          # different level, not a delta
        return sz if better(px, prev_px) else -prev_sz

    def update(self, bid_px: float, bid_sz: float,
               ask_px: float, ask_sz: float) -> None:
        bid_px, ask_px = float(bid_px or 0.0), float(ask_px or 0.0)
        bid_sz, ask_sz = float(bid_sz or 0.0), float(ask_sz or 0.0)

        # A bid is better when HIGHER, an ask when LOWER.
        self.bid_delta = self._side_delta(
            bid_px, bid_sz, self._bid_px, self._bid_sz, lambda a, b: a > b)
        self.ask_delta = self._side_delta(
            ask_px, ask_sz, self._ask_px, self._ask_sz, lambda a, b: a < b)

        # Resting bids are buyers, resting asks are sellers -- the same
        # sign convention the traded line uses, so "up" means the same
        # thing on both.
        self.book_cvd += self.bid_delta - self.ask_delta

        self._bid_px, self._bid_sz = bid_px, bid_sz
        self._ask_px, self._ask_sz = ask_px, ask_sz
        self.updates += 1

    @property
    def inside_bid_sz(self) -> float:
        return self._bid_sz

    @property
    def inside_ask_sz(self) -> float:
        return self._ask_sz


def liquidity_consumption_ratio(trade_size: float,
                                resting_size: float) -> float:
    """Aggressive size against the resting size it hit. >1 took the level."""
    t, r = float(trade_size or 0.0), float(resting_size or 0.0)
    if t <= 0.0:
        return 0.0
    if r <= 0.0:
        return LCR_MAX
    return min(LCR_MAX, t / r)


class VPINCalculator:
    """Volume-bucketed flow imbalance, bounded [0, 1].

    A trade that does not fit the current bucket is SPLIT: the part
    that fits closes it, the rest opens the next, and a print larger
    than a whole bucket closes several. Dropping the remainder would
    lose volume on exactly the largest prints.

    NOT the published VPIN, which infers side from price changes
    because its authors had no aggressor field. This uses Databento's
    true aggressor -- better input, different number. Do not compare it
    to figures in the literature.
    """

    __slots__ = ("bucket", "window", "_buckets", "_buy", "_sell", "completed")

    def __init__(self, bucket: int = 1000, window: int = 50) -> None:
        if bucket <= 0:
            raise ValueError("bucket size must be positive")
        if window <= 0:
            raise ValueError("window must be positive")
        self.bucket, self.window = int(bucket), int(window)
        self._buckets: deque = deque(maxlen=self.window)
        self._buy = self._sell = 0.0
        self.completed = 0

    def reset(self) -> None:
        self._buckets.clear()
        self._buy = self._sell = 0.0
        self.completed = 0

    def add(self, aggressor: str, size: float) -> None:
        left = float(size or 0.0)
        if left <= 0.0:
            return
        while left > 0.0:
            room = self.bucket - (self._buy + self._sell)
            take = left if left < room else room
            if aggressor == BUY:
                self._buy += take
            else:
                self._sell += take
            left -= take
            if (self._buy + self._sell) >= self.bucket:
                self._buckets.append(abs(self._buy - self._sell))
                self.completed += 1
                self._buy = self._sell = 0.0

    @property
    def value(self) -> float:
        n = len(self._buckets)
        if n == 0:
            return 0.0
        v = sum(self._buckets) / (n * self.bucket)
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    @property
    def filled(self) -> float:
        return (self._buy + self._sell) / self.bucket


class RollingStats:
    """Mean and standard deviation over a fixed lookback.

    Fixed window, not an all-history running form: both lines are
    CUMULATIVE and drift, so an all-history sigma grows without bound
    and a z-score computed from it shrinks toward zero all session.
    Non-finite samples are refused at the door.
    """

    __slots__ = ("_q", "_sum", "_sumsq")

    def __init__(self, window: int = 240) -> None:
        if window < 2:
            raise ValueError("window must be at least 2")
        self._q: deque = deque(maxlen=int(window))
        self._sum = self._sumsq = 0.0

    def reset(self) -> None:
        self._q.clear()
        self._sum = self._sumsq = 0.0

    def add(self, x: float) -> None:
        x = float(x)
        if not math.isfinite(x):
            return
        if len(self._q) == self._q.maxlen:
            old = self._q[0]
            self._sum -= old
            self._sumsq -= old * old
        self._q.append(x)
        self._sum += x
        self._sumsq += x * x

    @property
    def n(self) -> int:
        return len(self._q)

    @property
    def mean(self) -> float:
        return (self._sum / self.n) if self.n else 0.0

    @property
    def std(self) -> float:
        n = self.n
        if n < 2:
            return 0.0
        var = (self._sumsq - self._sum * self._sum / n) / (n - 1)
        return math.sqrt(var) if var > 0.0 else 0.0

    def z(self, x: float) -> float:
        s = self.std
        if s <= 0.0 or not math.isfinite(x):
            return 0.0
        return (float(x) - self.mean) / s


@dataclass(frozen=True, slots=True)
class Point:
    ts: float
    book: float
    tape: float
    bar: float
    px: float


@dataclass(frozen=True, slots=True)
class Cross:
    """Where the tape crossed the book, and whether it was confirmed.

    `confirmed` is the state AT THE MOMENT of the cross -- the two had
    already been agreeing long enough, and were not chopping. An
    unconfirmed cross is still recorded, because knowing how the
    unconfirmed ones behave is the point of watching.
    """

    ts: float
    px: float
    direction: str          # "up" = tape crossed up through the book
    book_z: float
    tape_z: float
    confirmed: bool
    held_s: float

    def as_dict(self) -> dict:
        return {"ts": self.ts, "px": round(self.px, 4),
                "direction": self.direction,
                "book_z": round(self.book_z, 3),
                "tape_z": round(self.tape_z, 3),
                "confirmed": self.confirmed,
                "held_s": round(self.held_s, 1)}


class BookCVDBars:
    """Book CVD, zeroed at every bar, drawn as OHLC.

    What this is, precisely: inside each bar the book CVD starts at
    zero and accumulates every add and pull at the inside quotes. The
    bar records where that running value opened (always zero), the
    furthest it got each way, and where it finished.

    So the candle is the PATH, not just the result. A bar that closes
    at +40 having been to +180 is a different bar from one that
    climbed straight to +40, and only the wicks tell them apart --
    the first had resting bids stacked and then pulled, the second
    had them stacked and kept.

    Boundaries are absolute wall-clock: a 5m bar starts on the
    5-minute mark, so these line up with the chart's candles rather
    than with whenever the engine booted.
    """

    __slots__ = ("seconds", "keep", "_open_ts", "cur", "hi", "lo",
                 "bars", "closed", "_stats")

    def __init__(self, seconds: float = 300.0, keep: int = 180,
                 window: int = 40) -> None:
        if seconds <= 0:
            raise ValueError("bar length must be positive")
        self.seconds = float(seconds)
        self.keep = int(keep)
        self.closed: deque = deque(maxlen=self.keep)
        self._stats = RollingStats(max(2, int(window)))
        self.reset()

    def reset(self) -> None:
        self._open_ts = 0.0
        self.cur = self.hi = self.lo = 0.0
        self.bars = 0
        self.closed.clear()
        self._stats.reset()

    def _bar_open(self, ts: float) -> float:
        return ts - (ts % self.seconds)

    def roll(self, ts: float) -> bool:
        """Close the bar if `ts` has crossed its boundary.

        Driven by time rather than by activity, so a silent bar still
        closes instead of merging into the next one.
        """
        o = self._bar_open(ts)
        if self._open_ts == 0.0:
            self._open_ts = o
            return False
        if o <= self._open_ts:
            return False
        self.closed.append({"ts": self._open_ts, "o": 0.0,
                            "h": round(self.hi, 1),
                            "l": round(self.lo, 1),
                            "c": round(self.cur, 1)})
        self._stats.add(self.cur)
        self.bars += 1
        self.cur = self.hi = self.lo = 0.0
        self._open_ts = o
        return True

    def add(self, ts: float, delta: float) -> bool:
        rolled = self.roll(ts)
        if delta:
            self.cur += float(delta)
            if self.cur > self.hi:
                self.hi = self.cur
            if self.cur < self.lo:
                self.lo = self.cur
        return rolled

    @property
    def z(self) -> float:
        """The running value against how closed bars of this size behave.

        This is what lets the per-bar line sit on the same axis as the
        two session-cumulative ones: raw, a bar's delta and a day's
        delta are nowhere near each other, and a second y-axis would
        be a lie that makes any two shapes look related.

        Zero until a few bars have closed, so a first bar cannot read
        as extreme merely for being first.
        """
        return self._stats.z(self.cur) if self._stats.n >= 3 else 0.0

    def elapsed(self, ts: float) -> float:
        if self._open_ts <= 0.0:
            return 0.0
        return max(0.0, min(1.0, (ts - self._open_ts) / self.seconds))

    def live(self, ts: float = 0.0) -> dict:
        return {"ts": self._open_ts, "o": 0.0, "h": round(self.hi, 1),
                "l": round(self.lo, 1), "c": round(self.cur, 1),
                "live": True, "elapsed": round(self.elapsed(ts), 3)}

    def series(self, limit: int = 120, ts: float = 0.0) -> list:
        out = list(self.closed)[-int(limit):]
        if self._open_ts > 0.0:
            out = out + [self.live(ts)]
        return out


class BookCVDFrames:
    """The same book CVD kept on 1m, 5m and 15m at once.

    Every frame is fed the same deltas, so switching the panel's
    timeframe is instant and loses no history -- which rebuilding a
    single series from scratch on every change would.
    """

    __slots__ = ("frames",)
    CHOICES = (60.0, 300.0, 900.0)

    def __init__(self, keep: int = 180) -> None:
        self.frames = {int(s): BookCVDBars(seconds=s, keep=keep)
                       for s in self.CHOICES}

    def reset(self) -> None:
        for f in self.frames.values():
            f.reset()

    def add(self, ts: float, delta: float) -> None:
        for f in self.frames.values():
            f.add(ts, delta)

    def roll(self, ts: float) -> None:
        for f in self.frames.values():
            f.roll(ts)

    def get(self, seconds: float) -> BookCVDBars:
        key = int(seconds)
        return self.frames.get(key, self.frames[300])

    def as_dict(self, seconds: float = 300.0, limit: int = 120,
                ts: float = 0.0) -> dict:
        f = self.get(seconds)
        return {"seconds": f.seconds,
                "choices": [int(c) for c in self.CHOICES],
                "bars": f.bars, "elapsed": round(f.elapsed(ts), 3),
                "series": f.series(limit, ts)}


@dataclass(slots=True)
class Episode:
    """One stretch of a single regime, and what price did during it.

    A regime is a STATE, not an event, so what it is worth is measured
    by what price does WHILE it holds -- not by what happens after a
    moment. `favour` and `adverse` are in ticks and signed from the
    regime's own direction, so an "up" and a "down" regime are read
    the same way.
    """

    regime: str
    side: str
    order: str
    ts: float
    px: float
    end_ts: float = 0.0
    end_px: float = 0.0
    hi: float = 0.0
    lo: float = 0.0
    favour: float = 0.0         # ticks, best in the regime's direction
    adverse: float = 0.0        # ticks, worst against (<= 0)
    net: float = 0.0            # ticks, close minus open
    seconds: float = 0.0
    open: bool = True

    @property
    def ticks_per_min(self) -> float:
        return (self.net / (self.seconds / 60.0)) if self.seconds > 0 else 0.0

    def as_dict(self) -> dict:
        return {"regime": self.regime, "side": self.side,
                "order": self.order, "ts": self.ts,
                "px": round(self.px, 4), "end_px": round(self.end_px, 4),
                "favour": round(self.favour, 2),
                "adverse": round(self.adverse, 2),
                "net": round(self.net, 2),
                "seconds": round(self.seconds, 1),
                "ticks_per_min": round(self.ticks_per_min, 2),
                "open": self.open}


@dataclass(slots=True)
class BarCall:
    """A call on the bar still forming, and what the bar actually did.

    Made PART WAY THROUGH the bar, from readings that existed at that
    moment, and settled against the bar's own close. That ordering is
    the whole point: it is trivially easy to "predict" a bar using
    numbers that already contain its outcome, and the result looks
    excellent and means nothing.

    `at_pct` is how far into the bar the call was made. A call at 80%
    of a bar knows most of it already and should be expected to score
    well -- which is exactly why the early checkpoints are the ones
    worth reading.
    """

    bar_ts: float
    bar_s: float
    at_pct: float
    caller: str                 # which reading made the call
    call: str                   # "up" | "down"
    open_px: float
    book_z: float = 0.0
    tape_z: float = 0.0
    bar_z: float = 0.0
    regime: str = "none"
    close_px: float = 0.0
    move_ticks: float = 0.0
    settled: bool = False
    correct: bool = False

    def as_dict(self) -> dict:
        return {"bar_ts": self.bar_ts, "bar_s": self.bar_s,
                "at_pct": round(self.at_pct, 2), "caller": self.caller,
                "call": self.call, "open_px": round(self.open_px, 4),
                "close_px": round(self.close_px, 4),
                "move_ticks": round(self.move_ticks, 2),
                "book_z": round(self.book_z, 3),
                "tape_z": round(self.tape_z, 3),
                "bar_z": round(self.bar_z, 3),
                "regime": self.regime, "settled": self.settled,
                "correct": self.correct}


@dataclass(slots=True)
class Signal:
    """One cross, and what price has done since.

    EVERY cross becomes one of these -- confirmed or not, good or
    bad. Suppressing the ones that look poor is how a reader ends up
    flattering itself: the unconfirmed and the losers are the only
    evidence that the confirmation is worth anything.

    Excursions are in TICKS and signed from the signal's own point of
    view, so `mfe` is always "how far it went my way" whether the
    signal was long or short. `mae` is negative or zero.

    This is forward tracking on live prints, not a backtest. It knows
    only what has happened since the cross, and it says how long that
    has been so a half-formed result is not mistaken for a finished
    one. It suggests nothing and places nothing.
    """

    id: int
    ts: float
    px: float
    side: str                   # "long" | "short"
    confirmed: bool
    book_z: float
    tape_z: float
    held_s: float
    last_px: float = 0.0
    hi: float = 0.0
    lo: float = 0.0
    mfe: float = 0.0            # ticks, best in favour
    mae: float = 0.0            # ticks, worst against (<= 0)
    age_s: float = 0.0
    settled: bool = False
    result: float = 0.0         # ticks at the horizon

    def as_dict(self) -> dict:
        return {"id": self.id, "ts": self.ts, "px": round(self.px, 4),
                "side": self.side, "confirmed": self.confirmed,
                "book_z": round(self.book_z, 3),
                "tape_z": round(self.tape_z, 3),
                "held_s": round(self.held_s, 1),
                "last_px": round(self.last_px, 4),
                "mfe": round(self.mfe, 2), "mae": round(self.mae, 2),
                "age_s": round(self.age_s, 1), "settled": self.settled,
                "result": round(self.result, 2)}


@dataclass(frozen=True, slots=True)
class Reading:
    ts: float = 0.0
    px: float = 0.0
    book_cvd: float = 0.0
    tape_cvd: float = 0.0
    book_z: float = 0.0
    tape_z: float = 0.0
    tape_ema: float = 0.0
    tau_s: float = 0.0
    book_dir: str = FLAT
    tape_dir: str = FLAT
    side: str = FLAT
    agree: bool = False
    held_s: float = 0.0
    flips: int = 0
    chop: bool = False
    ok: bool = False
    spread_z: float = 0.0
    regime: str = "none"
    order: str = ""
    bar_s: float = 0.0
    bar_c: float = 0.0          # book CVD so far INSIDE this bar
    bar_h: float = 0.0
    bar_l: float = 0.0
    bar_dir: str = FLAT
    bar_elapsed: float = 0.0
    bar_z: float = 0.0
    lcr: float = 0.0
    vpin: float = 0.0
    inside_bid: float = 0.0
    inside_ask: float = 0.0
    warm: bool = False
    trades: int = 0
    books: int = 0
    session: str = ""

    def as_dict(self) -> dict:
        return {"ts": self.ts, "px": round(self.px, 4),
                "book_cvd": round(self.book_cvd, 1),
                "tape_cvd": round(self.tape_cvd, 1),
                "book_z": round(self.book_z, 3),
                "tape_z": round(self.tape_z, 3),
                "tape_ema": round(self.tape_ema, 1),
                "tau_s": self.tau_s,
                "book_dir": self.book_dir, "tape_dir": self.tape_dir,
                "side": self.side, "agree": self.agree,
                "held_s": round(self.held_s, 1), "flips": self.flips,
                "chop": self.chop, "ok": self.ok,
                "spread_z": round(self.spread_z, 3),
                "regime": self.regime, "order": self.order,
                "bar_s": self.bar_s,
                "bar_c": round(self.bar_c, 1),
                "bar_h": round(self.bar_h, 1),
                "bar_l": round(self.bar_l, 1),
                "bar_dir": self.bar_dir,
                "bar_elapsed": round(self.bar_elapsed, 3),
                "bar_z": round(self.bar_z, 3),
                "lcr": round(self.lcr, 3), "vpin": round(self.vpin, 4),
                "inside_bid": self.inside_bid, "inside_ask": self.inside_ask,
                "warm": self.warm, "trades": self.trades,
                "books": self.books, "session": self.session}


class CVDCross:
    """Book CVD as the reader, traded CVD as the confirmation."""

    def __init__(self, price_move: str = "rebaseline",
                 stat_window: int = 240, sample_s: float = 0.25,
                 tau_s: float = 5.0,
                 enter: float = 0.40, exit: float = 0.15,
                 dwell_s: float = 2.0, max_flips: int = 4,
                 flip_window_s: float = 60.0, history: int = 1200,
                 crosses: int = 60, bucket: int = 1000,
                 vpin_window: int = 50, warm_at: int = 30,
                 tick_size: float = 0.25, horizon_s: float = 120.0,
                 signals: int = 250, bar_seconds: float = 300.0,
                 episodes: int = 400) -> None:
        if exit > enter:
            raise ValueError("exit threshold must not exceed enter")
        self.book = OrderBookDeltaTracker(price_move=price_move)
        self.vpin = VPINCalculator(bucket=bucket, window=vpin_window)
        self.book_stats = RollingStats(stat_window)
        self.tape_stats = RollingStats(stat_window)
        self.enter, self.exit = float(enter), float(exit)
        self.dwell_s = float(dwell_s)
        self.max_flips = int(max_flips)
        self.flip_window_s = float(flip_window_s)
        self.sample_s = float(sample_s)
        self.warm_at = int(warm_at)
        if tau_s <= 0.0:
            raise ValueError("tau_s must be positive")
        self.tau_s = float(tau_s)

        self.tape_cvd = 0.0
        self.last_px = self.last_ts = 0.0
        self.lcr = 0.0
        self.trades = self.books = 0
        self._session = ""

        self._points: deque = deque(maxlen=int(history))
        self._crosses: deque = deque(maxlen=int(crosses))
        self._flips: deque = deque(maxlen=64)
        self._book_dir = self._tape_dir = FLAT
        self._side = FLAT
        self._since = 0.0
        self._gap_sign = 0
        self._tape_ema = 0.0
        self._ema_ts = 0.0

        if tick_size <= 0:
            raise ValueError("tick size must be positive")
        self.tick_size = float(tick_size)
        self.horizon_s = float(horizon_s)
        self._signals: deque = deque(maxlen=int(signals))
        self._next_id = 1
        # The third reading: book CVD zeroed at every bar, as OHLC,
        # kept on 1m / 5m / 15m at once so the selector is instant.
        self.bars = BookCVDFrames()
        self.bar_seconds = float(bar_seconds)
        self._session_checked = 0.0
        # Calling the forming bar's direction, checked against its close.
        self.checkpoints = (0.2, 0.4, 0.6, 0.8)
        self._calls: deque = deque(maxlen=1500)
        self._bar_open_ts = 0.0
        self._bar_open_px = 0.0
        self._bar_taken: set = set()
        self._bar_closes: deque = deque(maxlen=500)
        # The regime: agreement AND which line is on top.
        self._episodes: deque = deque(maxlen=int(episodes))
        self._regime = "none"
        self._live: Episode | None = None

        self._lock = threading.Lock()
        self._read = Reading()

    # -------------------------------------------------------- session

    def _roll_session(self, ts: float) -> None:
        """Both lines are cumulative, so they need a zero somewhere.

        Called from the BOOK path as well as the trade path, because
        the book line moves with no prints at all -- a quiet session
        would otherwise never roll. Rate-limited to once a minute:
        the date conversion is far too expensive to run on every one
        of twenty-odd thousand book updates a second, and a boundary
        found within a minute is found soon enough.

        The CME session day, not the calendar day: ES runs Sunday 17:00
        to Friday 16:00 Central and the post-settlement hour belongs to
        the session that follows.
        """
        if ts - self._session_checked < 60.0:
            return
        self._session_checked = ts
        try:
            day = levels.trade_date(ts).isoformat()
        except Exception:                               # noqa: BLE001
            return
        if day == self._session:
            return
        if self._session:
            self.reset(keep_session=False)
        self._session = day

    def reset(self, keep_session: bool = True) -> None:
        self.book.reset()
        self.vpin.reset()
        self.book_stats.reset()
        self.tape_stats.reset()
        self.tape_cvd = 0.0
        self.lcr = 0.0
        self._points.clear()
        self._crosses.clear()
        self._flips.clear()
        self._book_dir = self._tape_dir = FLAT
        self._side = FLAT
        self._since = 0.0
        self._gap_sign = 0
        self._tape_ema = 0.0
        self._ema_ts = 0.0
        self._session_checked = 0.0
        self._signals.clear()
        self.bars.reset()
        self._episodes.clear()
        self._regime = "none"
        self._live = None
        self._calls.clear()
        self._bar_open_ts = 0.0
        self._bar_open_px = 0.0
        self._bar_taken = set()
        self._bar_closes.clear()
        if not keep_session:
            self._session = ""

    # ----------------------------------------------------------- feed

    def on_book(self, bid_px: float, bid_sz: float, ask_px: float,
                ask_sz: float, ts: float = 0.0) -> None:
        if ts:
            self._roll_session(ts)
        before = self.book.book_cvd
        self.book.update(bid_px, bid_sz, ask_px, ask_sz)
        # The bar series gets the INCREMENT, not the running total:
        # each bar starts at zero, so what it accumulates is the
        # change since the previous book, not the session's sum.
        if ts:
            self.bars.add(ts, self.book.book_cvd - before)
        self.books += 1
        if ts:
            self.last_ts = ts
            if bid_px and ask_px:
                self.last_px = (float(bid_px) + float(ask_px)) / 2.0
        self._sample(ts or self.last_ts)

    def on_trade(self, ts: float, aggressor: str, size: float,
                 price: float = 0.0) -> None:
        sz = float(size or 0.0)
        if sz <= 0.0:
            return
        self._roll_session(ts)
        self.tape_cvd += sz if aggressor == BUY else -sz
        self.vpin.add(aggressor, sz)
        resting = (self.book.inside_ask_sz if aggressor == BUY
                   else self.book.inside_bid_sz)
        self.lcr = liquidity_consumption_ratio(sz, resting)
        if price:
            self.last_px = float(price)
        self.last_ts = ts
        self.trades += 1
        self._sample(ts)

    # ------------------------------------------------------ direction

    def _direction(self, z: float, prev: str) -> str:
        """Hysteresis: a wide gate to turn on, a narrow one to let go.

        One threshold would flip the read every time the value grazed
        it, which is how a reader ends up changing its mind twenty
        times a second and calling it a signal.
        """
        if z >= self.enter:
            return UP
        if z <= -self.enter:
            return DOWN
        if prev == UP and z > self.exit:
            return UP
        if prev == DOWN and z < -self.exit:
            return DOWN
        return FLAT

    def _sample(self, ts: float) -> None:
        if ts <= 0.0:
            return
        last = self._points[-1].ts if self._points else 0.0
        if last and (ts - last) < self.sample_s:
            self._publish(ts)
            return

        book_v = self.book.book_cvd
        tape_v = self._smooth_tape(ts)
        self.book_stats.add(book_v)
        self.tape_stats.add(tape_v)
        bz = self.book_stats.z(book_v)
        tz = self.tape_stats.z(tape_v)

        self._book_dir = self._direction(bz, self._book_dir)
        self._tape_dir = self._direction(tz, self._tape_dir)

        # Agreement is the reading; how long it has held is the point.
        side = (self._book_dir
                if (self._book_dir == self._tape_dir
                    and self._book_dir != FLAT) else FLAT)
        if side != self._side:
            if side != FLAT:
                self._flips.append(ts)
            self._side = side
            self._since = ts if side != FLAT else 0.0

        self.bars.roll(ts)
        self._regime_step(ts, side, bz, tz)
        self._call_step(ts, side, bz, tz)

        # The cross itself: the tape passing through the book.
        gap = tz - bz
        sign = 1 if gap > 0 else (-1 if gap < 0 else 0)
        if sign and self._gap_sign and sign != self._gap_sign:
            held = (ts - self._since) if self._since else 0.0
            conf = self._ok(ts, side, held)
            self._crosses.append(Cross(
                ts=ts, px=self.last_px,
                direction=UP if sign > 0 else DOWN,
                book_z=bz, tape_z=tz,
                confirmed=conf, held_s=held))
            # EVERY cross raises a signal, confirmed or not. The ones
            # that look poor are the only evidence the confirmation is
            # worth anything.
            if self.last_px > 0.0:
                self._signals.append(Signal(
                    id=self._next_id, ts=ts, px=self.last_px,
                    side="long" if sign > 0 else "short",
                    confirmed=conf, book_z=bz, tape_z=tz, held_s=held,
                    last_px=self.last_px,
                    hi=self.last_px, lo=self.last_px))
                self._next_id += 1
        if sign:
            self._gap_sign = sign

        self._points.append(Point(ts=ts, book=bz, tape=tz,
                                  bar=self.bars.get(self.bar_seconds).z,
                                  px=self.last_px))
        self._track(ts)
        self._publish(ts)

    def _smooth_tape(self, ts: float) -> float:
        """Traded CVD smoothed on ELAPSED SECONDS, not on trade count.

        dt is real time between samples, so a quiet stretch decays the
        smooth exactly as much as a busy one of the same length, and
        the exponential composes exactly: asking twice as often over
        the same span gives the same answer. The first sample seeds at
        whatever the line reads then, which at session open is zero.
        """
        x = self.tape_cvd
        if self._ema_ts <= 0.0:
            self._tape_ema = x
            self._ema_ts = ts
            return x
        dt = ts - self._ema_ts
        if dt <= 0.0:
            return self._tape_ema
        a = 1.0 - math.exp(-dt / self.tau_s)
        self._tape_ema += a * (x - self._tape_ema)
        self._ema_ts = ts
        return self._tape_ema

    def _recent_flips(self, ts: float) -> int:
        cut = ts - self.flip_window_s
        return sum(1 for t in self._flips if t >= cut)

    def _ok(self, ts: float, side: str, held: float) -> bool:
        if side == FLAT:
            return False
        if not self._warm():
            return False
        if held < self.dwell_s:
            return False
        return self._recent_flips(ts) <= self.max_flips

    def _warm(self) -> bool:
        return (self.book_stats.n >= self.warm_at
                and self.tape_stats.n >= self.warm_at)

    def _publish(self, ts: float) -> None:
        _live = self.bars.get(self.bar_seconds)
        book_v = self.book.book_cvd
        tape_v = self._tape_ema
        bz = self.book_stats.z(book_v)
        tz = self.tape_stats.z(tape_v)
        held = (ts - self._since) if self._since else 0.0
        flips = self._recent_flips(ts)
        chop = flips > self.max_flips
        self._read = Reading(
            # tape_cvd is the RAW cumulative line; tape_ema is the
            # time-smoothed one that the z-score and the cross are
            # taken from. Reporting the smooth as the raw would hide
            # how far the second line is lagging, which is the one
            # thing a reader needs to see to trust the cross.
            ts=ts, px=self.last_px, book_cvd=book_v,
            tape_cvd=self.tape_cvd,
            book_z=bz, tape_z=tz, tape_ema=tape_v, tau_s=self.tau_s,
            book_dir=self._book_dir, tape_dir=self._tape_dir,
            side=self._side, agree=self._side != FLAT, held_s=held,
            flips=flips, chop=chop,
            ok=self._ok(ts, self._side, held),
            spread_z=tz - bz, regime=self._regime,
            order=("book_above" if bz > tz else "book_below"),
            bar_s=_live.seconds, bar_c=_live.cur, bar_h=_live.hi,
            bar_l=_live.lo,
            bar_dir=(UP if _live.cur > 0 else
                     (DOWN if _live.cur < 0 else FLAT)),
            bar_elapsed=_live.elapsed(ts), bar_z=_live.z,
            lcr=self.lcr, vpin=self.vpin.value,
            inside_bid=self.book.inside_bid_sz,
            inside_ask=self.book.inside_ask_sz,
            warm=self._warm(), trades=self.trades, books=self.books,
            session=self._session)

    # ----------------------------------------------------------- read

    # -------------------------------------------------------- regime

    def _regime_step(self, ts: float, side: str,
                     bz: float, tz: float) -> None:
        """The reading that matters: agreement AND which line is on top.

        Four live states, not one event:

            up_book_above     both lean up, book CVD above traded
            up_book_below     both lean up, traded above book
            down_book_above   both lean down, book above traded
            down_book_below   both lean down, traded above book

        plus `none` when the two do not agree. A regime is a STATE, so
        it is judged by what price does WHILE it holds, which is why
        each stretch is kept as an Episode with its own excursions
        rather than as a moment with an outcome bolted on afterwards.
        """
        order = "book_above" if bz > tz else "book_below"
        key = "none" if side == FLAT else f"{side}_{order}"
        if key != self._regime:
            self._close_episode(ts)
            self._regime = key
            if key != "none" and self.last_px > 0.0:
                self._live = Episode(
                    regime=key, side=side, order=order, ts=ts,
                    px=self.last_px, hi=self.last_px, lo=self.last_px)
                self._episodes.append(self._live)
            else:
                self._live = None
        self._walk_episode(ts)

    def _walk_episode(self, ts: float) -> None:
        e = self._live
        px = self.last_px
        if e is None or px <= 0.0:
            return
        if px > e.hi:
            e.hi = px
        if px < e.lo:
            e.lo = px
        tick = self.tick_size
        if e.side == UP:
            e.favour = (e.hi - e.px) / tick
            e.adverse = (e.lo - e.px) / tick
            e.net = (px - e.px) / tick
        else:
            e.favour = (e.px - e.lo) / tick
            e.adverse = (e.px - e.hi) / tick
            e.net = (e.px - px) / tick
        e.end_px = px
        e.end_ts = ts
        e.seconds = ts - e.ts

    def _close_episode(self, ts: float) -> None:
        if self._live is not None:
            self._walk_episode(ts)
            self._live.open = False
            self._live = None

    def episodes(self) -> list:
        with self._lock:
            return list(self._episodes)

    def regime_board(self, min_seconds: float = 2.0) -> dict:
        """What price actually does inside each regime.

        The hypothesis this exists to test: that when the book line is
        ABOVE the traded line the move runs harder, and likewise below.
        If that is real it shows up as a higher median net travel and a
        higher ticks-per-minute for the `book_above` regimes than the
        `book_below` ones on the same side.

        Stretches shorter than `min_seconds` are excluded: a regime
        that existed for one sample has no travel to measure and would
        other wise pull every median toward zero.

        LIVE FORWARD TRACKING of real prints, not a backtest. `n` is
        how many stretches, not how many trades, and a handful of
        stretches says nothing however clean the numbers look.
        """
        rows = [e for e in self.episodes() if e.seconds >= min_seconds]
        out = {"min_seconds": min_seconds, "episodes": len(rows),
               "bar_s": self.bar_seconds}
        for key in ("up_book_above", "up_book_below",
                    "down_book_above", "down_book_below"):
            es = [e for e in rows if e.regime == key]
            out[key] = {
                "n": len(es),
                "median_net": round(_median([e.net for e in es]), 2),
                "median_favour": round(_median([e.favour for e in es]), 2),
                "median_adverse": round(_median([e.adverse for e in es]), 2),
                "median_seconds": round(_median([e.seconds for e in es]), 1),
                "median_ticks_per_min": round(
                    _median([e.ticks_per_min for e in es]), 2),
                "total_seconds": round(sum(e.seconds for e in es), 1),
            }
        # The comparison he is actually making, stated outright.
        for side in (UP, DOWN):
            a = out[f"{side}_book_above"]
            b = out[f"{side}_book_below"]
            out[f"{side}_edge"] = {
                "book_above_net": a["median_net"],
                "book_below_net": b["median_net"],
                "difference": round(a["median_net"] - b["median_net"], 2),
                "n_above": a["n"], "n_below": b["n"],
                "enough": a["n"] >= 20 and b["n"] >= 20,
            }
        return out

    # ---------------------------------------------------- bar calls

    def _call_step(self, ts: float, side: str,
                   bz: float, tz: float) -> None:
        """Call the forming bar, then settle it against its own close.

        Two callers are recorded side by side so they can be compared
        against each other and against the market's own base rate:

            regime   the two lines agreeing, up or down
            bar_z    this bar's book CVD against how bars of this
                     size usually behave

        A call is only taken ONCE per checkpoint per bar, and only
        from readings that existed at that moment. Nothing here looks
        at the bar's close before the bar has closed.
        """
        f = self.bars.get(self.bar_seconds)
        px = self.last_px
        if px <= 0.0:
            return
        open_ts = f._open_ts
        if open_ts <= 0.0:
            return

        if open_ts != self._bar_open_ts:
            # the previous bar just ended: settle everything on it
            self._settle(self._bar_open_ts, self._bar_open_px, px)
            self._bar_open_ts = open_ts
            self._bar_open_px = px
            self._bar_taken = set()
            return

        pct = f.elapsed(ts)
        for cp in self.checkpoints:
            if pct < cp or cp in self._bar_taken:
                continue
            self._bar_taken.add(cp)
            bar_z = f.z
            for caller, call in (("regime", side),
                                 ("bar_z", UP if bar_z > 0 else
                                  (DOWN if bar_z < 0 else FLAT))):
                if call == FLAT:
                    continue
                self._calls.append(BarCall(
                    bar_ts=open_ts, bar_s=f.seconds, at_pct=cp,
                    caller=caller, call=call,
                    open_px=self._bar_open_px or px,
                    book_z=bz, tape_z=tz, bar_z=bar_z,
                    regime=self._regime))

    def _settle(self, bar_ts: float, open_px: float,
                close_px: float) -> None:
        if bar_ts <= 0.0 or open_px <= 0.0:
            return
        move = (close_px - open_px) / self.tick_size
        self._bar_closes.append(1 if move > 0 else (0 if move < 0 else -1))
        for c in self._calls:
            if c.settled or c.bar_ts != bar_ts:
                continue
            c.close_px = close_px
            c.move_ticks = move
            c.settled = True
            c.correct = ((move > 0 and c.call == UP)
                         or (move < 0 and c.call == DOWN))

    def calls(self) -> list:
        with self._lock:
            return list(self._calls)

    def call_board(self) -> dict:
        """Can these readings call the bar before it closes?

        Reported against the BASE RATE -- the share of bars that
        closed up at all. That comparison is the whole measurement: in
        a session where 56% of bars close green, a caller scoring 56%
        has found nothing, and saying "56% correct" without the base
        rate beside it would read as though it had.

        `edge` is the caller's rate minus the better of always-up and
        always-down, so zero means no better than the dumbest possible
        rule and negative means worse.

        Broken out by how far into the bar the call was made, because
        a call at 80% already knows most of the bar. The early
        checkpoints are the ones that would be worth money.
        """
        rows = [c for c in self.calls() if c.settled]
        ups = [x for x in self._bar_closes if x >= 0]
        base_up = (sum(1 for x in ups if x == 1) / len(ups)) if ups else 0.0
        naive = max(base_up, 1.0 - base_up) if ups else 0.0
        out = {"bars_settled": len(ups),
               "base_up": round(base_up, 4),
               "naive_rate": round(naive, 4),
               "bar_s": self.bar_seconds,
               "checkpoints": list(self.checkpoints),
               "callers": {}}
        for caller in ("regime", "bar_z"):
            per = {}
            for cp in self.checkpoints:
                cs = [c for c in rows
                      if c.caller == caller and c.at_pct == cp]
                n = len(cs)
                ok = sum(1 for c in cs if c.correct)
                rate = (ok / n) if n else None
                per[str(cp)] = {
                    "n": n, "correct": ok,
                    "rate": round(rate, 4) if rate is not None else None,
                    "edge": (round(rate - naive, 4)
                             if rate is not None and ups else None),
                    "enough": n >= 100,
                }
            out["callers"][caller] = per
        return out

    def _track(self, ts: float) -> None:
        """Walk the open signals forward against the last print.

        Only the UNSETTLED ones are touched, so the cost per sample is
        bounded by how many are still inside the horizon rather than
        by how many have ever fired.
        """
        px = self.last_px
        if px <= 0.0:
            return
        tick = self.tick_size
        for g in self._signals:
            if g.settled:
                continue
            g.last_px = px
            if px > g.hi:
                g.hi = px
            if px < g.lo:
                g.lo = px
            if g.side == "long":
                g.mfe = (g.hi - g.px) / tick
                g.mae = (g.lo - g.px) / tick
                g.result = (px - g.px) / tick
            else:
                g.mfe = (g.px - g.lo) / tick
                g.mae = (g.px - g.hi) / tick
                g.result = (g.px - px) / tick
            g.age_s = ts - g.ts
            if g.age_s >= self.horizon_s:
                g.settled = True

    def signals(self) -> list:
        with self._lock:
            return list(self._signals)

    def scoreboard(self, target: float = 4.0, stop: float = 4.0) -> dict:
        """What the signals have actually done, split by confirmation.

        A LIVE FORWARD TALLY, not a backtest. It counts only what has
        happened since each cross fired, so a young sample says little
        -- which is why `settled` is reported separately from the
        total, and why the break-even for the stop and target asked
        for is printed beside the hit rate rather than left implied.

        `hit` means the signal reached +target ticks before it reached
        -stop. A signal still inside the horizon that has done neither
        counts as neither.
        """
        rows = self.signals()
        out = {"target_ticks": target, "stop_ticks": stop,
               "break_even": round(stop / (target + stop), 4),
               "horizon_s": self.horizon_s}
        for name, sel in (("all", lambda g: True),
                          ("confirmed", lambda g: g.confirmed),
                          ("unconfirmed", lambda g: not g.confirmed)):
            gs = [g for g in rows if sel(g)]
            hit = sum(1 for g in gs if g.mfe >= target and g.mae > -stop)
            miss = sum(1 for g in gs if g.mae <= -stop)
            done = hit + miss
            out[name] = {
                "n": len(gs),
                "settled": sum(1 for g in gs if g.settled),
                "hit": hit, "miss": miss, "open": len(gs) - done,
                "rate": round(hit / done, 4) if done else None,
                "median_mfe": round(_median([g.mfe for g in gs]), 2),
                "median_mae": round(_median([g.mae for g in gs]), 2),
                "mean_result": round(
                    sum(g.result for g in gs) / len(gs), 2) if gs else 0.0,
            }
        return out

    def snapshot(self) -> Reading:
        return self._read

    def points(self) -> list:
        with self._lock:
            return list(self._points)

    def crosses(self) -> list:
        with self._lock:
            return list(self._crosses)

    def as_dict(self, limit: int = 600, since: float = 0.0,
                with_signals: bool = False, target: float = 4.0,
                stop: float = 4.0) -> dict:
        """The reading, and the points the caller does not yet have.

        `since` makes the poll INCREMENTAL. Resending 600 points four
        times a second is tens of kilobytes a second of data the
        browser already holds, and that backlog is what puts a live
        chart behind the market. With `since` the usual frame carries
        one or two points.

        `continuous` says whether `since` was still inside the ring.
        False means the caller was away longer than the ring is long,
        so there is a hole and it must take the full set again --
        otherwise it would splice two disjoint stretches into one line
        and draw a move that never happened.
        """
        pts = self.points()
        continuous = True
        if since > 0.0:
            oldest = pts[0].ts if pts else 0.0
            if oldest and since < oldest:
                continuous = False          # the ring turned over
            else:
                pts = [p for p in pts if p.ts > since]
        pts = pts[-int(limit):]
        return {"now": self.snapshot().as_dict(),
                "continuous": continuous,
                "points": [{"ts": p.ts, "book": round(p.book, 3),
                            "tape": round(p.tape, 3),
                            "bar": round(p.bar, 3),
                            "px": round(p.px, 4)} for p in pts],
                "crosses": [c.as_dict() for c in self.crosses()],
                "sample_s": self.sample_s, "tau_s": self.tau_s,
                "enter": self.enter, "dwell_s": self.dwell_s,
                "capacity": self._points.maxlen,
                # Opt-in: the panel polls this twice a second and does
                # not want the whole signal book on every frame. The
                # agent asks for it explicitly.
                "bars": self.bars.as_dict(self.bar_seconds,
                                          ts=self.last_ts),
                "regime": self.snapshot().regime,
                **({"signals": [g.as_dict() for g in self.signals()],
                    "scoreboard": self.scoreboard(target, stop),
                    "regime_board": self.regime_board(),
                    "call_board": self.call_board(),
                    "episodes": [e.as_dict()
                                 for e in self.episodes()[-60:]]}
                   if with_signals else {})}


def _mock(seconds: float = 900.0, seed: int = 7) -> None:         # pragma: no cover
    """Synthetic stream. Shows it runs flat in memory and nothing else.

    The data is invented, so this proves NOTHING about whether the
    cross means anything. That question belongs to real prints.
    """
    import itertools
    import random
    import time

    rng = random.Random(seed)
    r = CVDCross()
    bid, ask, bsz, asz = 5799.75, 5800.25, 40.0, 40.0
    t0 = ts = time.time()
    print(f"{'elapsed':>8} {'trades':>8} {'book_z':>8} {'tape_z':>8} "
          f"{'side':>5} {'held':>6} {'flips':>5}  state")
    for i in itertools.count():
        ts += 0.002
        bsz = max(1.0, bsz + rng.randint(-3, 3))
        asz = max(1.0, asz + rng.randint(-3, 3))
        if rng.random() < 0.015:
            step = 0.25 if rng.random() < 0.5 else -0.25
            bid, ask = bid + step, ask + step
        r.on_book(bid, bsz, ask, asz, ts)
        if rng.random() < 0.09:
            side = BUY if rng.random() < 0.5 else SELL
            r.on_trade(ts, side, rng.choice((1, 1, 2, 3, 5, 30)),
                       ask if side == BUY else bid)
        if i % 100_000 == 0:
            s = r.snapshot()
            state = ("CHOP" if s.chop else
                     ("CONFIRMED" if s.ok else
                      ("agreeing" if s.agree else "-")))
            print(f"{ts - t0:8.1f} {s.trades:8,} {s.book_z:8.2f} "
                  f"{s.tape_z:8.2f} {s.side:>5} {s.held_s:6.1f} "
                  f"{s.flips:5}  {state}")
        if (ts - t0) >= seconds:
            break
    d = r.as_dict()
    conf = sum(1 for c in d["crosses"] if c["confirmed"])
    print(f"\nmessages {i:,} over {ts - t0:,.0f}s")
    print(f"points {len(d['points'])} of {d['capacity']} (bounded)  "
          f"crosses {len(d['crosses'])} of which {conf} confirmed")
    try:
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"peak RSS {kb / 1024:,.0f} MB")
    except Exception:                                   # noqa: BLE001
        pass


if __name__ == "__main__":                              # pragma: no cover
    _mock()
