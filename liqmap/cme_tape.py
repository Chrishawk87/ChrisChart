"""CME trade prints into the site's own candles.

An MBP record stream carries trades alongside the book updates, so the
subscription that feeds the OFI engine already contains everything a
chart needs. This turns those prints into the same `LiveCandle` the
Hyperliquid path builds, through the same `live.CandleBuilder` -- not a
second candle implementation, because two of those drift and the
difference shows up as a chart that disagrees with the read drawn
beneath it.

WHAT IT PRODUCES

    candles        one `CandleBuilder` per interval, with the venue's
                   own trade prints and the aggressor side on each
    book           the latest `flow.Book`, for the book call
    last trade     price and side, for the ticker

AGGRESSOR SIDE IS REAL HERE, NOT INFERRED. CME states which side
crossed the spread, so the buy/sell split and the CVD underneath these
candles are measured rather than guessed from an uptick rule.

THE GRID IS THE VENUE'S. `CandleBuilder` floors every trade to
interval boundaries, so a 15-minute candle here starts where the
exchange's does. A candle that began when the feed happened to connect
would not line up with anything you could check it against.

A RECORD THAT IS NOT A TRADE IS NOT AN ERROR. Most records in an MBP
stream are adds, cancels and modifies; `offer` returns False for them
and that is the normal case, not a rejection to count.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from . import dbn
from .live import CandleBuilder

DEFAULT_INTERVALS = (60.0, 300.0, 900.0)      # 1m, 5m, 15m
KEEP_CLOSED = 300                             # per interval


def label(seconds: float) -> str:
    """60 -> '1m', 900 -> '15m', 3600 -> '1h'."""
    s = int(seconds)
    if s % 86400 == 0:
        return f"{s // 86400}d"
    if s % 3600 == 0:
        return f"{s // 3600}h"
    if s % 60 == 0:
        return f"{s // 60}m"
    return f"{s}s"


@dataclass
class Tape:
    """Trades from a CME record stream, as candles per timeframe."""

    symbol: str = "ES"
    intervals: tuple = DEFAULT_INTERVALS
    keep: int = KEEP_CLOSED

    builders: dict = field(default_factory=dict)
    closed: dict = field(default_factory=dict)
    book: object = None
    last_px: float = 0.0
    last_side: str = ""
    last_ts: float = 0.0
    trades: int = 0
    records: int = 0
    books: int = 0

    def __post_init__(self) -> None:
        for s in self.intervals:
            name = label(s)
            self.closed[name] = deque(maxlen=self.keep)
            self.builders[name] = CandleBuilder(
                s, on_close=self._closer(name))

    def _closer(self, name: str):
        def on_close(candle) -> None:
            self.closed[name].append(candle)
        return on_close

    # ------------------------------------------------------------ feed

    def offer(self, rec) -> bool:
        """One record in. True when it carried a trade.

        The book is updated from every record that has levels, because
        the book call wants the state as of now and most records are
        not trades.
        """
        self.records += 1
        if dbn._get(rec, "levels"):
            try:
                self.book = dbn.book_from(rec, symbol=self.symbol)
                self.books += 1
            except Exception:                           # noqa: BLE001
                pass
        t = dbn.trade_from(rec)
        if t is None:
            return False
        self.trades += 1
        self.last_px, self.last_side, self.last_ts = t.px, t.aggressor, t.ts
        for b in self.builders.values():
            b.add(t)
        return True

    def roll(self, now: float) -> None:
        """Close any candle whose interval has elapsed without a trade.

        Without this a quiet stretch leaves the last candle open past
        its own boundary, and the chart shows a bar that the exchange
        considers finished.
        """
        for b in self.builders.values():
            b.roll_to(now)

    # ------------------------------------------------------------ read

    def candle(self, interval: str, now: float | None = None):
        """The candle in progress, on the TAPE's clock by default.

        `CandleBuilder.candle(None)` rolls to the wall clock. Live that
        is right, because the feed's timestamps track it. Replaying a
        file it is not: the records are from last month and the builder
        rolls millions of seconds forward, closing every candle and
        handing back an empty one stamped today. Defaulting to the last
        timestamp the tape actually saw makes replay and live agree,
        and a caller that wants wall clock passes it.
        """
        b = self.builders.get(interval)
        if b is None:
            return None
        return b.candle(now if now is not None else (self.last_ts or None))

    def history(self, interval: str, limit: int = 200,
                include_current: bool = True) -> list:
        b = self.builders.get(interval)
        if b is None:
            return []
        rows = b.history(include_current=include_current)
        return rows[-limit:] if limit else rows

    def snapshot(self, interval: str = "1m", limit: int = 0,
                 now: float | None = None) -> dict:
        """What a panel needs, in one call."""
        cur = self.candle(interval, now)
        out = {
            "symbol": self.symbol,
            "intervals": [label(s) for s in self.intervals],
            "interval": interval,
            "records": self.records,
            "trades": self.trades,
            "books": self.books,
            "last_px": self.last_px,
            "last_side": self.last_side,
            "last_ts": self.last_ts,
            "closed": {k: len(v) for k, v in self.closed.items()},
            "current": None if cur is None else {
                "start_ts": cur.start_ts,
                "interval_s": cur.interval_s,
                "open": cur.open, "high": cur.high,
                "low": cur.low, "close": cur.close,
                "volume": cur.volume, "trades": cur.trades,
                "buy_notional": cur.buy_notional,
                "sell_notional": cur.sell_notional,
            },
        }
        if self.book is not None:
            bids, asks = self.book.bids, self.book.asks
            out["book"] = {
                "ts": self.book.ts,
                "bid": bids[0].px if bids else 0.0,
                "bid_sz": bids[0].sz if bids else 0.0,
                "ask": asks[0].px if asks else 0.0,
                "ask_sz": asks[0].sz if asks else 0.0,
                "levels": min(len(bids), len(asks)),
            }
        if limit:
            out["candles"] = [
                {"ts": c.ts, "open": c.open, "high": c.high,
                 "low": c.low, "close": c.close, "volume": c.volume}
                for c in self.history(interval, limit=limit)
            ]
        return out
