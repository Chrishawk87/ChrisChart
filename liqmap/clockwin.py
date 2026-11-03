"""Fifteen-minute windows rebuilt from raw events, on one clock.

WHY THIS EXISTS. The OFI candles end on a quantity of flow -- every
2,000 contracts of imbalance -- and they flip often enough to whipsaw
the trade. This is the same three measurements on a CLOCK instead, so
the two can be compared: if the time version flips less without
collapsing into one signal an hour, it is the better base.

NOTHING HERE IS RESAMPLED. Every number is accumulated from the raw
event that produced it, in O(1), as it arrives. There is no
`resample('15min')` and no pass over a pre-built frame, because the
point is a window that grows tick by tick and locks at the boundary
-- which is what the chart shows and what a trader sees. A batch
resample produces the same closed values and cannot produce the
forming one.

WHAT IS REBUILT, AND FROM WHAT.

    price   MID-PRICE from the book, (bid + ask) / 2, on every book
            update. Chris's choice, and the one that matches OFI:
            both are book measurements, so they move on the same
            events. The cost is stated rather than hidden -- the high
            and low include levels no trade ever happened at, so this
            candle's range is not a range you could have been filled
            in. `last_*` carries the traded prices beside it.

    OFI     LEVEL-LEVEL at the touch, because the subscription is
            MBP-10. Per-order reconstruction needs MBO and is not
            available here. The formula is `ofi.level_ofi`, which is
            the same function the contract-bucket candles use -- the
            whole comparison would be worthless if the two versions
            of OFI were computed differently.

    CVD     The side CME states on each print. 'B' adds, 'A'
            subtracts, and 'N' is COUNTED, NOT DROPPED: the window
            carries `unknown` and `unknown_size` so a tape full of
            them cannot masquerade as a balanced one. The existing
            decoder returns None for those trades, which means the
            running CVD elsewhere in this service quietly omits them.

Nothing here places, sizes or times an order.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .ofi import Touch, level_ofi

PERIOD_S = 900.0          # fifteen minutes
RAW_TICK = 250_000_000.0  # ES, one quarter point in raw units


def window_start(ts: float, period_s: float = PERIOD_S) -> float:
    """The left edge of the window `ts` falls in.

    LABEL LEFT, CLOSED LEFT, which is what pandas would be told and
    what the chart draws: a bar stamped 10:00 holds everything from
    10:00:00.000 up to but not including 10:15.
    """
    if period_s <= 0:
        return 0.0
    return (ts // period_s) * period_s


@dataclass
class Window:
    """One clock window's running state. Mutated in place."""

    start_ts: float = 0.0
    period_s: float = PERIOD_S

    # Mid-price OHLC, from the book.
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    # Traded prices beside it, so the candle you could have been
    # filled in is visible next to the one the book drew.
    last_open: float = 0.0
    last_high: float = 0.0
    last_low: float = 0.0
    last_close: float = 0.0

    net_ofi: int = 0
    net_cvd: float = 0.0

    books: int = 0
    trades: int = 0
    buy_size: float = 0.0
    sell_size: float = 0.0
    # 'N' -- neither side stated. Counted so a tape full of them
    # cannot look like a balanced one.
    unknown: int = 0
    unknown_size: float = 0.0

    ts_first: float = 0.0
    ts_last: float = 0.0
    closed: bool = False

    @property
    def mid_dir(self) -> str:
        if not self.open or not self.close:
            return "flat"
        d = self.close - self.open
        return "up" if d > 0 else "down" if d < 0 else "flat"

    @property
    def last_dir(self) -> str:
        if not self.last_open or not self.last_close:
            return "flat"
        d = self.last_close - self.last_open
        return "up" if d > 0 else "down" if d < 0 else "flat"

    @property
    def ofi_dir(self) -> str:
        return ("up" if self.net_ofi > 0
                else "down" if self.net_ofi < 0 else "flat")

    @property
    def cvd_dir(self) -> str:
        return ("up" if self.net_cvd > 0
                else "down" if self.net_cvd < 0 else "flat")

    @property
    def agree(self) -> bool:
        """All three the same way, and none of them flat.

        FLAT IS NOT AGREEMENT. Three zeros are not three signals
        pointing together, and counting them as confirmed would
        report a dead window as the strongest kind there is.
        """
        d = self.mid_dir
        return (d in ("up", "down")
                and self.ofi_dir == d and self.cvd_dir == d)

    @property
    def ticks(self) -> float:
        if not self.open:
            return 0.0
        return round((self.close - self.open) / RAW_TICK, 2)

    def to_dict(self) -> dict:
        return {
            "start_ts": self.start_ts, "period_s": self.period_s,
            "open": self.open, "high": self.high, "low": self.low,
            "close": self.close, "ticks": self.ticks,
            "last_open": self.last_open, "last_high": self.last_high,
            "last_low": self.last_low, "last_close": self.last_close,
            "net_ofi": self.net_ofi, "net_cvd": self.net_cvd,
            "mid_dir": self.mid_dir, "last_dir": self.last_dir,
            "ofi_dir": self.ofi_dir, "cvd_dir": self.cvd_dir,
            "agree": self.agree,
            "books": self.books, "trades": self.trades,
            "buy_size": self.buy_size, "sell_size": self.sell_size,
            "unknown": self.unknown, "unknown_size": self.unknown_size,
            "ts_first": self.ts_first, "ts_last": self.ts_last,
            "closed": self.closed,
        }


@dataclass
class ClockWindows:
    """The state machine. One window open, the rest finished.

    Every `on_book` and `on_trade` is O(1): a comparison, an add, and
    at a boundary one list append. Nothing is recomputed and nothing
    is held that is not needed to carry the running state forward.
    """

    period_s: float = PERIOD_S
    keep: int = 400

    current: Window | None = None
    done: list = field(default_factory=list)
    _prev: Touch | None = None
    # Each closed window's OFI direction, so a flip between
    # consecutive windows is a comparison and not a scan.
    _last_dir: str = ""
    flips: int = 0
    flips_confirmed: int = 0

    # -- the clock ------------------------------------------------------

    def _roll(self, ts: float) -> Window | None:
        """Close the open window if `ts` has left it. Returns it."""
        cur = self.current
        start = window_start(ts, self.period_s)
        if cur is not None and start == cur.start_ts:
            return None
        finished = None
        if cur is not None:
            cur.closed = True
            finished = cur
            self.done.append(cur)
            if len(self.done) > self.keep:
                del self.done[0:len(self.done) - self.keep]
            # A FLIP IS BETWEEN WINDOWS, counted as they close.
            d = cur.ofi_dir
            if d in ("up", "down"):
                if self._last_dir and d != self._last_dir:
                    self.flips += 1
                    if cur.cvd_dir == d:
                        self.flips_confirmed += 1
                self._last_dir = d
        self.current = Window(start_ts=start, period_s=self.period_s)
        # THE BOOK CARRIES ACROSS THE BOUNDARY, the OFI does not. The
        # next window's first event is measured against the last book
        # state of the previous one -- that change is real and
        # belongs to the new window. Dropping `_prev` here would
        # discard the first event of every window.
        return finished

    # -- events ---------------------------------------------------------

    def on_book(self, ts: float, bid_px: int, bid_sz: int,
                ask_px: int, ask_sz: int) -> Window | None:
        """One book update. Returns a window if this one closed it."""
        t = Touch(bid_px=bid_px, bid_sz=bid_sz,
                  ask_px=ask_px, ask_sz=ask_sz)
        if not t.valid or t.crossed:
            return None
        finished = self._roll(ts)
        w = self.current
        assert w is not None
        mid = (bid_px + ask_px) / 2.0
        if self._prev is not None:
            w.net_ofi += int(level_ofi(self._prev, t))
        self._prev = t
        if not w.books:
            w.open = w.high = w.low = mid
            w.ts_first = ts
        else:
            if mid > w.high:
                w.high = mid
            if mid < w.low:
                w.low = mid
        w.close = mid
        w.books += 1
        w.ts_last = ts
        return finished

    def on_trade(self, ts: float, size: float,
                 side: str) -> Window | None:
        """One print. `side` is 'B', 'A' or 'N' as Databento states it.

        'N' IS COUNTED, NOT DROPPED. A tape full of unknown-side
        prints must not read as a balanced one, and the decoder this
        service already uses returns None for them -- so the running
        CVD elsewhere omits them silently.
        """
        finished = self._roll(ts)
        w = self.current
        assert w is not None
        s = (side or "").strip().upper()[:1]
        sz = float(size or 0.0)
        if sz > 0:
            if s == "B":
                w.net_cvd += sz
                w.buy_size += sz
            elif s == "A":
                w.net_cvd -= sz
                w.sell_size += sz
            else:
                w.unknown += 1
                w.unknown_size += sz
        w.trades += 1
        w.ts_last = ts
        return finished

    def on_trade_px(self, ts: float, px: float, size: float,
                    side: str) -> Window | None:
        """A print with its price, for the traded-price candle."""
        finished = self.on_trade(ts, size, side)
        w = self.current
        if w is not None and px > 0:
            if not w.last_open:
                w.last_open = w.last_high = w.last_low = px
            else:
                if px > w.last_high:
                    w.last_high = px
                if px < w.last_low:
                    w.last_low = px
            w.last_close = px
        return finished

    # -- reading ---------------------------------------------------------

    def snapshot(self, limit: int = 40) -> dict:
        rows = [w.to_dict() for w in self.done[-limit:]]
        return {
            "period_s": self.period_s,
            "current": self.current.to_dict() if self.current else None,
            "windows": rows,
            "summary": self.summary(),
        }

    def summary(self) -> dict:
        """How often the three agree, and how clean the flips are.

        THE COMPARISON THIS WAS BUILT FOR. The contract-bucket
        candles flip several times a bar; the question is whether
        this clock version flips less WITHOUT collapsing into too few
        signals to trade. Both halves are reported, because either
        one alone can be made to look good.
        """
        done = self.done
        n = len(done)
        decided = [w for w in done if w.mid_dir in ("up", "down")]
        agreed = [w for w in decided if w.agree]
        unknown = sum(w.unknown for w in done)
        return {
            "windows": n,
            "decided": len(decided),
            "agreed": len(agreed),
            "agree_pct": (round(len(agreed) / len(decided), 4)
                          if decided else 0.0),
            "ofi_flips": self.flips,
            "flips_cvd_confirmed": self.flips_confirmed,
            "flips_unconfirmed": self.flips - self.flips_confirmed,
            "flips_per_window": (round(self.flips / n, 3) if n else 0.0),
            "unknown_side_trades": unknown,
            "trades": sum(w.trades for w in done),
        }
