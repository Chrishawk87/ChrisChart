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
                 vpin_window: int = 50, warm_at: int = 30) -> None:
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

        self._lock = threading.Lock()
        self._read = Reading()

    # -------------------------------------------------------- session

    def _roll_session(self, ts: float) -> None:
        """Both lines are cumulative, so they need a zero somewhere.

        The CME session day, not the calendar day: ES runs Sunday 17:00
        to Friday 16:00 Central and the post-settlement hour belongs to
        the session that follows.
        """
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
        if not keep_session:
            self._session = ""

    # ----------------------------------------------------------- feed

    def on_book(self, bid_px: float, bid_sz: float, ask_px: float,
                ask_sz: float, ts: float = 0.0) -> None:
        self.book.update(bid_px, bid_sz, ask_px, ask_sz)
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

        # The cross itself: the tape passing through the book.
        gap = tz - bz
        sign = 1 if gap > 0 else (-1 if gap < 0 else 0)
        if sign and self._gap_sign and sign != self._gap_sign:
            held = (ts - self._since) if self._since else 0.0
            self._crosses.append(Cross(
                ts=ts, px=self.last_px,
                direction=UP if sign > 0 else DOWN,
                book_z=bz, tape_z=tz,
                confirmed=self._ok(ts, side, held), held_s=held))
        if sign:
            self._gap_sign = sign

        self._points.append(Point(ts=ts, book=bz, tape=tz, px=self.last_px))
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
            spread_z=tz - bz, lcr=self.lcr, vpin=self.vpin.value,
            inside_bid=self.book.inside_bid_sz,
            inside_ask=self.book.inside_ask_sz,
            warm=self._warm(), trades=self.trades, books=self.books,
            session=self._session)

    # ----------------------------------------------------------- read

    def snapshot(self) -> Reading:
        return self._read

    def points(self) -> list:
        with self._lock:
            return list(self._points)

    def crosses(self) -> list:
        with self._lock:
            return list(self._crosses)

    def as_dict(self, limit: int = 600) -> dict:
        pts = self.points()[-int(limit):]
        return {"now": self.snapshot().as_dict(),
                "points": [{"ts": p.ts, "book": round(p.book, 3),
                            "tape": round(p.tape, 3),
                            "px": round(p.px, 4)} for p in pts],
                "crosses": [c.as_dict() for c in self.crosses()],
                "sample_s": self.sample_s, "tau_s": self.tau_s,
                "enter": self.enter, "dwell_s": self.dwell_s,
                "capacity": self._points.maxlen}


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
