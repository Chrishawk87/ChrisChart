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

from .flow import Book, FlowTape, Trade
from .structure import Candle

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

        self.trades_seen = 0
        self.book_updates = 0
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
        self.connected = True
        for sub in ({"type": "trades", "coin": self.coin},
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
        elif channel == "l2Book":
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
                self.trades_seen += 1
                self.last_trade_ts = max(self.last_trade_ts, t.ts)

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

    def _note(self, msg: str) -> None:
        self.errors.append(f"{time.strftime('%H:%M:%S')} {msg}")
        del self.errors[:-20]

    # -- read --------------------------------------------------------------

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
            "reconnects": self.reconnects,
            "intervals": sorted(self.builders),
            "uptime_s": (time.time() - self.started_at) if self.started_at else 0.0,
            "errors": self.errors[-3:],
        }
