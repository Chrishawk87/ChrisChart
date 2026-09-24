"""The current candle, built from the tape as it happens.

WHY THIS EXISTS

Until now the in-progress candle came from `candleSnapshot`, polled every few
seconds. Three things are wrong with that for a scalp:

  - It lags. By the time a poll returns, the move that would have told you
    something has happened.
  - It can hand back a CLOSED bar and say nothing about it, which is how a
    falling market gets read as a green candle.
  - It costs rate-limit budget on every poll, for data the exchange is
    already pushing for free over the WebSocket.

So the current candle is no longer fetched. It is CONSTRUCTED, trade by
trade, from the same public tape that already drives flow and absorption.
Open, high, low, last, volume and the buy/sell split all come from fills as
they print. There is nothing to poll and nothing to lag behind.

THE GRID IS NOT OPTIONAL

A candle that starts whenever the feed happens to connect does not line up
with the exchange's candles, and a read on a bar that is offset by four
minutes from the one on your chart is worse than no read. So every candle is
aligned to the interval grid -- floor(timestamp / interval) * interval -- the
same boundaries the venue uses. `CandleBuilder.roll_to` makes that explicit
and testable.

SEEDING, AND WHY THE OPEN IS THE HARD PART

Connecting mid-candle means the open already happened and no amount of tape
will tell you what it was. Guessing it from the first trade after connection
makes an in-progress candle look like it started wherever you happened to
tune in, which quietly corrupts position-in-range and the change figure. So a
builder starts UNSEEDED, refuses to report a candle until it is given a real
open or a complete grid boundary passes, and says which it is (`seeded`).

STALENESS IS A FIRST-CLASS READING

A WebSocket that has silently stopped delivering looks exactly like a market
where nothing is trading. `age` and `connected` are reported everywhere so
the difference is visible rather than inferred.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from .flow import Book, FlowTape, ImpactBaseline, Trade
from .structure import Candle
from .volume import VolumeProfile

# Intervals the builder understands, in seconds. Mirrors the exchange's.
INTERVALS: dict[str, int] = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "8h": 28800, "12h": 43200,
    "1d": 86400,
}

# A feed with nothing on it for this long is treated as impaired rather than
# quiet. Thin markets do go minutes between prints, so this is deliberately
# generous and reported rather than acted on.
STALE_AFTER_S = 90.0


def grid_start(ts: float, interval_s: float) -> float:
    """The exchange's candle boundary at or before `ts`."""
    if interval_s <= 0:
        return ts
    return ts - (ts % interval_s)


@dataclass
class LiveCandle:
    """One candle assembled from fills."""

    interval_s: float
    start_ts: float
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    trades: int = 0
    buy_notional: float = 0.0
    sell_notional: float = 0.0
    seeded: bool = True          # False when the open was never observed
    # Where inside this bar the business was done. Built from the same
    # fills that build the OHLC, which were previously being discarded
    # after their four prices had been extracted.
    profile: VolumeProfile = field(default_factory=VolumeProfile)

    @property
    def end_ts(self) -> float:
        return self.start_ts + self.interval_s

    @property
    def delta(self) -> float:
        return self.buy_notional - self.sell_notional

    @property
    def lean(self) -> float:
        total = self.buy_notional + self.sell_notional
        return self.delta / total if total > 0 else 0.0

    def elapsed(self, now: float | None = None) -> float:
        now = time.time() if now is None else now
        return max(0.0, min(self.interval_s, now - self.start_ts))

    def to_candle(self) -> Candle:
        return Candle(ts=self.start_ts, open=self.open, high=self.high,
                      low=self.low, close=self.close, volume=self.volume,
                      trades=self.trades)

    @classmethod
    def from_candle(cls, c: Candle, interval_s: float) -> "LiveCandle":
        """Adopt a fetched bar. Used when seeding history at connect.

        `closed` holds LiveCandles, so fetched bars must be converted rather
        than dropped in alongside them -- a mixed list only fails later, in
        whatever reads it.
        """
        return cls(interval_s=interval_s,
                   start_ts=grid_start(c.ts, interval_s),
                   open=c.open, high=c.high, low=c.low, close=c.close,
                   volume=c.volume, trades=c.trades, seeded=True)


class CandleBuilder:
    """Turns a stream of trades into grid-aligned candles for one interval."""

    def __init__(self, interval_s: float,
                 on_close: Callable[[LiveCandle], None] | None = None):
        self.interval_s = float(interval_s)
        self.on_close = on_close
        self.current: LiveCandle | None = None
        self.closed: list[LiveCandle] = []
        self.max_closed = 400

    # -- seeding ----------------------------------------------------------

    def seed(self, candle: Candle) -> None:
        """Adopt a candle fetched from the exchange as the starting point.

        Only used once, at connect, to learn an open that already happened.
        Everything after comes from the tape.
        """
        start = grid_start(candle.ts, self.interval_s)
        self.current = LiveCandle(
            interval_s=self.interval_s, start_ts=start, open=candle.open,
            high=candle.high, low=candle.low, close=candle.close,
            volume=candle.volume, trades=candle.trades, seeded=True)

    # -- ingest -----------------------------------------------------------

    def add(self, trade: Trade) -> LiveCandle | None:
        """Fold one fill in. Returns the candle that just closed, if any."""
        if trade.px <= 0 or trade.sz <= 0:
            return None

        start = grid_start(trade.ts, self.interval_s)
        rolled: LiveCandle | None = None

        if self.current is None:
            # First trade with no seed: the open is unknown, so this candle is
            # marked unseeded and callers can decide whether to trust it.
            self.current = LiveCandle(
                interval_s=self.interval_s, start_ts=start, open=trade.px,
                high=trade.px, low=trade.px, close=trade.px, seeded=False)
        elif start > self.current.start_ts:
            rolled = self._roll(start, trade.px)
        elif start < self.current.start_ts:
            # An out-of-order or replayed fill from a bar that already closed.
            # Dropping it is correct: folding it into the current candle would
            # put an old price into a new bar's range.
            return None

        c = self.current
        c.high = max(c.high, trade.px)
        c.low = min(c.low, trade.px)
        c.close = trade.px
        c.volume += trade.sz
        c.trades += 1
        if trade.aggressor == "buy":
            c.buy_notional += trade.notional
        else:
            c.sell_notional += trade.notional
        c.profile.add(trade.px, trade.notional, trade.aggressor)
        return rolled

    def roll_to(self, now: float) -> LiveCandle | None:
        """Close the current candle if the clock has passed its boundary.

        Needed because a quiet market produces no trade to trigger the roll,
        and a candle that stays 'current' for three intervals reports an
        elapsed fraction of 100% forever.
        """
        if self.current is None:
            return None
        start = grid_start(now, self.interval_s)
        if start <= self.current.start_ts:
            return None
        return self._roll(start, self.current.close)

    def _roll(self, new_start: float, px: float) -> LiveCandle:
        done = self.current
        assert done is not None
        self.closed.append(done)
        del self.closed[:-self.max_closed]
        if self.on_close:
            try:
                self.on_close(done)
            except Exception:
                pass
        # The new bar opens at `px`, which is the FIRST TRADE of that bar when
        # a fill triggered the roll -- matching how the venue stamps an open.
        # A clock-driven roll passes the previous close instead, because with
        # no trades the bar opens flat at the last price.
        self.current = LiveCandle(
            interval_s=self.interval_s, start_ts=new_start, open=px,
            high=px, low=px, close=px, seeded=True)
        return done

    # -- read -------------------------------------------------------------

    def candle(self, now: float | None = None) -> LiveCandle | None:
        self.roll_to(time.time() if now is None else now)
        return self.current

    def history(self, include_current: bool = False) -> list[Candle]:
        out = [c.to_candle() for c in self.closed]
        if include_current and self.current is not None:
            out.append(self.current.to_candle())
        return out


class LiveFeed:
    """A persistent WebSocket for one market.

    Holds the tape, the book and a candle builder per interval, all driven by
    pushed messages rather than polling. One feed per market; the dashboard
    runs one at a time because that is how many you can look at.

    NOT EXERCISED AGAINST THE LIVE API from the environment this was written
    in. `status()` reports message counts and the age of the last tick, which
    is the fastest way to tell a working feed from a silent one.
    """

    def __init__(self, coin: str, intervals: Sequence[str] = ("1m", "5m", "15m"),
                 ws_url: str = "wss://api.hyperliquid.xyz/ws",
                 tape_seconds: float = 3600.0,
                 buy_codes: Iterable[str] | None = None):
        self.coin = coin
        self.ws_url = ws_url
        self.buy_codes = set(buy_codes) if buy_codes else None

        self.builders: dict[str, CandleBuilder] = {
            name: CandleBuilder(INTERVALS[name])
            for name in intervals if name in INTERVALS}
        self.tape = FlowTape(max_age=tape_seconds)
        self.book: Book | None = None
        self._book_hist: list[tuple[float, float, float, float]] = []

        # The book read itself. Updated on EVERY book push rather than on a
        # candle schedule -- the book is the structure, and it changes
        # thousands of times a minute.
        from .bookread import BookReader
        self.reader = BookReader()

        # Impact baseline, learned from this market's own tape. Without a
        # scale, "ten million of buying" means nothing -- it is enormous at
        # 3am and unremarkable at the open.
        self.baseline = ImpactBaseline()
        self.bucket_s = 15.0
        self._bucket_start: float | None = None
        self._bucket_delta = 0.0
        self._bucket_open_px = 0.0
        self._bucket_last_px = 0.0

        self.trades_seen = 0
        self.book_updates = 0
        # Whether the venue honoured `fast: true`. Not assumed -- set only
        # when a message actually arrives on the fast channel.
        self.fast_book = False
        self.last_trade_ts = 0.0
        self.last_msg_ts = 0.0
        self.connected = False
        self.reconnects = 0
        self.errors: list[str] = []
        self.started_at = 0.0

        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._ws = None
        self._listeners: list[Callable[[str], None]] = []

    # -- lifecycle ---------------------------------------------------------

    def seed(self, interval: str, candles: Sequence[Candle]) -> None:
        """Give a builder the open of the bar already in progress."""
        b = self.builders.get(interval)
        if not b or not candles:
            return
        last = candles[-1]
        conv = [LiveCandle.from_candle(c, b.interval_s) for c in candles]
        if time.time() - last.ts < b.interval_s:
            b.seed(last)                       # the bar still in progress
            b.closed = conv[:-1][-b.max_closed:]
        else:
            b.closed = conv[-b.max_closed:]

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.started_at = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:
            pass

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def on_update(self, fn: Callable[[str], None]) -> None:
        """Register a callback fired on each batch of messages."""
        self._listeners.append(fn)

    def off_update(self, fn: Callable[[str], None]) -> None:
        """Unregister a callback.

        Every stream connection registers one, and EventSource reconnects on
        its own -- so without this a tab left open overnight leaves hundreds
        of dead callbacks being invoked on every fill.
        """
        try:
            self._listeners.remove(fn)
        except ValueError:
            pass

    @property
    def listener_count(self) -> int:
        return len(self._listeners)

    # -- the socket --------------------------------------------------------

    def _run(self) -> None:
        try:
            import websocket
        except ImportError:
            self._note("websocket-client is not installed")
            return

        while not self._stop.is_set():
            try:
                ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=lambda _w, e: self._note(f"ws error: {e}"),
                    on_close=lambda _w, *_a: setattr(self, "connected", False))
                self._ws = ws
                ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as exc:
                self._note(f"ws crashed: {exc}")
            finally:
                self.connected = False

            if self._stop.is_set():
                break
            self.reconnects += 1
            # A tight reconnect loop against a rate-limited venue makes the
            # outage worse. Back off, but stay responsive to stop().
            self._stop.wait(min(2.0 * self.reconnects, 20.0))

    def _on_open(self, ws) -> None:
        """Subscribe, asking for the FAST book.

        This matters more than anything else in this file.

        In June 2026 Hyperliquid slowed the public `l2Book` feed: 20 levels
        every two seconds, moving later to five, and it is currently
        observed arriving around every 5.4 seconds. Everything this service
        reads from the book -- microprice tilt, replenishment, queue
        depletion -- is a measurement of how the book CHANGES. At one update
        every five seconds a twenty-second window holds four samples, the
        dynamics are noise, and the call is being made from a photograph
        taken before the last two candles of a one-minute chart.

        `fast: true` asks for the five-level raw stream instead, which
        arrives around every 500ms -- roughly ten times more often. Five
        levels is not a loss here: `NEAR_LEVELS` is 5, so the near book was
        all that was ever read.

        Both are requested. If the venue ignores `fast` the plain feed still
        arrives and the reader carries on at the slower rate; `updates_per_s`
        in `status()` reports which one is actually being delivered, because
        a silently degraded feed looks exactly like a quiet market.
        """
        self.connected = True
        for sub in ({"type": "trades", "coin": self.coin},
                    {"type": "l2Book", "coin": self.coin, "fast": True},
                    {"type": "l2Book", "coin": self.coin}):
            try:
                ws.send(json.dumps({"method": "subscribe", "subscription": sub}))
            except Exception as exc:
                self._note(f"subscribe {sub['type']}: {exc}")

    def _on_message(self, _ws, payload) -> None:
        try:
            msg = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(msg, dict):
            # Subscription acks and pongs arrive in other shapes. Calling
            # .get on them takes the whole socket thread down.
            return

        channel = msg.get("channel")
        self.last_msg_ts = time.time()

        if channel == "trades":
            self._handle_trades(msg.get("data") or [])
        elif channel in ("l2Book", "fastBook"):
            # The fast stream comes back on its own channel name. Missing
            # this would mean subscribing to the fast feed successfully and
            # then discarding every message it sends.
            if channel == "fastBook":
                self.fast_book = True
            self._handle_book(msg.get("data") or {})
        else:
            return

        for fn in list(self._listeners):
            try:
                fn(channel)
            except Exception:
                pass

    def _handle_trades(self, entries: list) -> None:
        from .hl import parse_trade

        with self._lock:
            for raw in entries:
                t = (parse_trade(raw, self.buy_codes) if self.buy_codes
                     else parse_trade(raw))
                if t is None:
                    continue
                self.tape.add(t)
                for b in self.builders.values():
                    b.add(t)
                self._accumulate(t)
                self.trades_seen += 1
                self.last_trade_ts = max(self.last_trade_ts, t.ts)

    # -- absorption, continuously ------------------------------------------
    #
    # Absorption used to require pointing a LevelWatch at a specific price.
    # That made it unavailable by default, so the panel said "flow and
    # absorption missing" even while the tape was running and flow was
    # plainly working. Absorption is not a property of a level -- it is a
    # property of the tape -- so the feed learns it for itself.

    def _accumulate(self, trade: Trade) -> None:
        """Roll fills into fixed buckets and feed each finished one to the
        impact baseline, so "is this a lot of volume" has a scale."""
        if self._bucket_start is None:
            self._bucket_start = trade.ts
            self._bucket_open_px = trade.px

        if trade.ts < self._bucket_start:        # clock went backwards
            self._bucket_start = trade.ts
            self._bucket_open_px = trade.px
            self._bucket_delta = 0.0

        if trade.ts - self._bucket_start >= self.bucket_s:
            if self._bucket_open_px > 0 and self._bucket_last_px > 0:
                move = ((self._bucket_last_px - self._bucket_open_px)
                        / self._bucket_open_px * 10_000.0)
                self.baseline.observe(self._bucket_delta, move)
            self._bucket_start = trade.ts
            self._bucket_open_px = trade.px
            self._bucket_delta = 0.0

        self._bucket_delta += trade.signed_notional
        self._bucket_last_px = trade.px

    def absorption(self, window_s: float = 300.0):
        """What the tape says about who is being absorbed, right now.

        Returns a `flow.Absorption` or None when the baseline has not seen
        enough to have a scale yet -- it says so rather than guessing.
        """
        from .flow import Absorption, BandStat

        with self._lock:
            w = self.tape.window(window_s)
            move = self.tape.price_change_bps(window_s)
            px = self.tape.last_px
            book = self.book

        direction = "buy" if w.delta >= 0 else "sell"
        net = abs(w.delta)
        expected = self.baseline.expected_bps(net)
        signed_move = move if direction == "buy" else -move
        ratio = (signed_move / expected) if expected > 0 else 0.0

        # Resting size around the current price, for the eaten/replaced line.
        band = BandStat(low_px=px * 0.999, high_px=px * 1.001, observations=0,
                        first_notional=0.0, last_notional=0.0,
                        min_notional=0.0, max_notional=0.0,
                        consumed=0.0, replenished=0.0, refill_events=0)
        if book is not None and not book.empty:
            resting = book.band_notional(px * 0.999, px * 1.001)
            band = BandStat(low_px=px * 0.999, high_px=px * 1.001,
                            observations=1, first_notional=resting,
                            last_notional=resting, min_notional=resting,
                            max_notional=resting, consumed=0.0,
                            replenished=0.0, refill_events=0)

        return Absorption(
            level=px, window_s=window_s, direction=direction,
            aggressive_notional=net, observed_bps=signed_move,
            expected_bps=expected, impact_ratio=ratio, band=band,
            confident=self.baseline.ready, trades=w.trades)

    def _handle_book(self, data: dict) -> None:
        from .hl import parse_book

        try:
            b = parse_book(self.coin, data)
        except Exception as exc:
            self._note(f"book parse: {exc}")
            return
        if b.empty:
            return
        with self._lock:
            self.book = b
            self.book_updates += 1
            self.reader.add(b, now=time.time())
            # Keep a short history of the spread and the imbalance. A single
            # snapshot says what the book looks like; the series says what is
            # happening to it, which is the part you can trade. A spread
            # widening while one side thins is the book getting out of the
            # way -- price action at the finest grain there is.
            self._book_hist.append((time.time(), b.spread_bps,
                                    b.imbalance(25.0), b.mid))
            del self._book_hist[:-600]

    def book_call(self, window_s: float | None = None):
        """The book's call on the current candle. Cheap: the reader has
        already done the work on each push."""
        if window_s is not None:
            self.reader.window_s = window_s
        agg = 0.0
        w = self.tape.window(min(self.reader.window_s * 3, 120.0))
        if w.total > 0:
            agg = w.lean
        return self.reader.read(aggression=agg)

    def spread_trend(self, window_s: float = 60.0) -> dict:
        """What the book has been doing over the window, not just now."""
        with self._lock:
            hist = list(self._book_hist)
        if not hist:
            return {"samples": 0}

        cutoff = time.time() - window_s
        inside = [h for h in hist if h[0] >= cutoff] or hist[-1:]
        spreads = [h[1] for h in inside]
        imbs = [h[2] for h in inside]

        first_half = imbs[: max(1, len(imbs) // 2)]
        second_half = imbs[max(1, len(imbs) // 2):] or first_half

        return {
            "samples": len(inside),
            "spread_bps": spreads[-1],
            "spread_avg": sum(spreads) / len(spreads),
            "spread_widening": spreads[-1] > (sum(spreads) / len(spreads)) * 1.3,
            "imbalance": imbs[-1],
            "imbalance_avg": sum(imbs) / len(imbs),
            # Which way the resting book is TILTING, not where it sits.
            "imbalance_shift": (sum(second_half) / len(second_half)
                                - sum(first_half) / len(first_half)),
        }

    def _note(self, msg: str) -> None:
        self.errors.append(f"{time.strftime('%H:%M:%S')} {msg}")
        del self.errors[:-20]

    # -- read --------------------------------------------------------------

    def ensure_interval(self, name: str,
                        seed_bars: Sequence[Candle] | None = None) -> bool:
        """Start building an interval the feed was not asked for at startup.

        The dashboard's timeframe selector can name any interval, and a feed
        opened on 1m/5m/15m has nothing to offer when you switch to 30m --
        which looked like the socket had stopped working. Adding the builder
        on demand costs nothing: the fills are already arriving.
        """
        if name in self.builders:
            return False
        if name not in INTERVALS:
            return False
        self.builders[name] = CandleBuilder(INTERVALS[name])
        if seed_bars:
            self.seed(name, seed_bars)
        return True

    def candle(self, interval: str) -> LiveCandle | None:
        b = self.builders.get(interval)
        return b.candle() if b else None

    def history(self, interval: str) -> list[Candle]:
        b = self.builders.get(interval)
        return b.history() if b else []

    @property
    def age(self) -> float | None:
        """Seconds since the last message of any kind."""
        return (time.time() - self.last_msg_ts) if self.last_msg_ts else None

    @property
    def updates_per_s(self) -> float:
        """Measured book pushes per second.

        The single most diagnostic number this feed produces. Everything
        read from the book is a measurement of change, so the rate at which
        change arrives IS the resolution of the signal. Around 2/s means the
        fast stream; around 0.2/s means the throttled public feed and a book
        read that cannot see anything inside five seconds.

        Measured rather than assumed, because a feed that quietly degrades
        looks exactly like a quiet market.
        """
        if not self.started_at:
            return 0.0
        up = time.time() - self.started_at
        return round(self.book_updates / up, 2) if up > 1.0 else 0.0

    @property
    def feed_quality(self) -> str:
        """Plain words on whether the book is arriving fast enough."""
        if not self.book_updates:
            return "no book updates yet"
        r = self.updates_per_s
        if r >= 1.0:
            return f"fast book — {r:.1f} updates/s"
        if r >= 0.35:
            return (f"{r:.2f} updates/s — the throttled public feed. The "
                    f"dynamics are thin at this rate")
        return (f"only {r:.2f} updates/s. The book read cannot see anything "
                f"happening inside {1 / max(r, 0.01):.0f}s, so replenishment "
                f"and depletion are close to meaningless — this is the "
                f"slowed public feed, not a quiet market")

    @property
    def stale(self) -> bool:
        a = self.age
        return a is None or a > STALE_AFTER_S

    def status(self) -> dict:
        a = self.age
        return {
            "coin": self.coin,
            "running": self.running,
            "connected": self.connected,
            "stale": self.stale,
            "age_s": a,
            "trades_seen": self.trades_seen,
            "book_updates": self.book_updates,
            "fast_book": self.fast_book,
            "updates_per_s": self.updates_per_s,
            "feed_quality": self.feed_quality,
            "reconnects": self.reconnects,
            "baseline_samples": self.baseline.samples,
            "absorption_ready": self.baseline.ready,
            "book_reads": self.reader.updates,
            "intervals": sorted(self.builders),
            "uptime_s": (time.time() - self.started_at) if self.started_at else 0.0,
            "errors": self.errors[-3:],
        }
