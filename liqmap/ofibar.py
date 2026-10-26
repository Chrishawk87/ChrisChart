"""Candles closed by order-flow imbalance rather than by time or ticks.

A bar ends when the flow that built it crosses a line, so every bar on
the chart is the same quantity of imbalance rather than the same
quantity of clock. A quiet hour is one bar; a violent minute is twenty.

    GREEN   closed because OFI since the last bar reached +threshold
    RED     closed because it reached -threshold
    CHOP    closed on the time cap with neither reached -- price moved,
            flow did not commit, and the bar says so

THE ACCUMULATOR RESETS AT EVERY CLOSE, AND THAT IS NOT A DETAIL.

Run a single global accumulator against a fixed threshold and the
threshold is a latch, not a filter. Cumulative OFI tracks cumulative
price, so on any session with a direction it walks away from zero and
never returns: once it clears +250 it is above +250 for the rest of
the day and every bar after the first is green. Measured on a book
drifting up at 52% of ticks, that construction printed 99.9% green,
0.0% red, 0.1% chop over 60,000 updates -- a chart that cannot say
anything because it only has one word.

Resetting at each close makes the threshold mean "imbalance SINCE the
last bar", which is a quantity that can go either way and has to be
earned again every time.

The e_n formula is `liqmap.ofi.level_ofi`, imported rather than
copied. It is the Cont-Kukanov-Stoikov definition and it is already
exercised by that module's tests; a second copy here would be a
second thing to keep correct.

NOTHING HERE PREDICTS. A green bar says buying pressure crossed a line
you chose. Whether that precedes price is a separate question this
module does not answer and does not assume.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ofi import Touch, level_ofi

GREEN = "green"
RED = "red"
CHOP = "chop"

# The thresholds the live chart can switch between, all accumulating at
# once so a switch is instant and carries its whole history.
#
# Bars scale as 1/target^2 -- the accumulator is a random walk and first
# passage to +/-L goes as L squared. Measured on 55M real ES book
# updates, 4,000 produced one bar every 52 minutes, so this ladder spans
# roughly three seconds to thirteen minutes a bar.
LADDER: tuple[int, ...] = (125, 250, 500, 1000, 2000)


@dataclass(slots=True)
class OfiBar:
    """One closed bar. `ofi` is the accumulated imbalance that ended it."""

    ts_open: float = 0.0
    ts_close: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    direction: str = CHOP
    ofi: int = 0
    updates: int = 0
    ofi_high: int = 0        # furthest positive excursion inside the bar
    ofi_low: int = 0         # furthest negative

    @property
    def seconds(self) -> float:
        return self.ts_close - self.ts_open

    @property
    def strength(self) -> int:
        """Signed imbalance that closed the bar. The spec's OFI_Strength."""
        return self.ofi

    def as_row(self) -> list:
        """[Timestamp, Open, High, Low, Close, Direction, OFI_Strength].

        The column order the charting side expects. Timestamp is the
        bar's CLOSE, because that is the instant the bar became a fact;
        stamping it with the open would place every bar earlier than
        the information that produced it.
        """
        return [self.ts_close, self.open, self.high, self.low,
                self.close, self.direction, self.ofi]

    def as_dict(self) -> dict:
        return {"ts_open": self.ts_open, "ts_close": self.ts_close,
                "open": self.open, "high": self.high, "low": self.low,
                "close": self.close, "direction": self.direction,
                "ofi": self.ofi, "updates": self.updates,
                "ofi_high": self.ofi_high, "ofi_low": self.ofi_low,
                "seconds": round(self.seconds, 6)}


ROW_COLUMNS = ("timestamp", "open", "high", "low", "close",
               "direction", "ofi_strength")


class OfiBarBuilder:
    """Feed it book touches oldest-first; it hands back closed bars.

    `threshold` is in contracts of net imbalance. `max_s` caps how long
    a bar may run without either line being reached -- without it a
    quiet stretch produces a single bar that never ends, and the chart
    stops updating while looking like nothing is happening.
    """

    __slots__ = ("tick", "threshold", "max_s", "min_updates",
                 "_prev", "_acc", "_bar", "bars")

    def __init__(self, tick_size: float = 0.25, threshold: int = 250,
                 max_s: float = 30.0, min_updates: int = 1) -> None:
        if tick_size <= 0:
            raise ValueError("tick_size must be positive")
        if threshold < 1:
            raise ValueError("threshold must be at least 1")
        if max_s <= 0:
            raise ValueError("max_s must be positive")
        if min_updates < 1:
            raise ValueError("min_updates must be at least 1")
        self.tick = float(tick_size)
        self.threshold = int(threshold)
        self.max_s = float(max_s)
        # A guard against a bar closing on a single update, which
        # happens when one enormous size change clears the line by
        # itself and leaves a bar with open == high == low == close.
        self.min_updates = int(min_updates)
        self._prev: Touch | None = None
        self._acc = 0
        self._bar: OfiBar | None = None
        self.bars: list[OfiBar] = []

    def reset(self) -> None:
        self._prev = None
        self._acc = 0
        self._bar = None
        self.bars.clear()

    @property
    def current(self) -> OfiBar | None:
        """The bar still forming. Its `ofi` is live and may yet reverse."""
        return self._bar

    @property
    def accumulated(self) -> int:
        return self._acc

    # ------------------------------------------------------------------
    def on_touch(self, ts: float, touch: Touch) -> OfiBar | None:
        """One book update. Returns a bar if this update closed one."""
        if not touch.valid or touch.crossed:
            # A crossed or absent touch is not a book the formula is
            # defined on. Withhold rather than contribute a zero, which
            # would sum like a measurement.
            return None

        mid = (touch.bid_px + touch.ask_px) / 2.0

        if self._prev is None:
            self._prev = touch
            self._open_bar(ts, mid)
            return None

        e = level_ofi(self._prev, touch)
        self._prev = touch
        self._acc += e

        bar = self._bar
        if bar is None:
            self._open_bar(ts, mid)
            bar = self._bar
            assert bar is not None

        bar.updates += 1
        if mid > bar.high:
            bar.high = mid
        if mid < bar.low:
            bar.low = mid
        bar.close = mid
        bar.ts_close = ts
        if self._acc > bar.ofi_high:
            bar.ofi_high = self._acc
        if self._acc < bar.ofi_low:
            bar.ofi_low = self._acc

        if bar.updates >= self.min_updates:
            if self._acc >= self.threshold:
                return self._close(GREEN, ts, mid)
            if self._acc <= -self.threshold:
                return self._close(RED, ts, mid)
        if (ts - bar.ts_open) >= self.max_s:
            return self._close(CHOP, ts, mid)
        return None

    def on_book(self, ts: float, bid_px: int, bid_sz: int,
                ask_px: int, ask_sz: int) -> OfiBar | None:
        return self.on_touch(ts, Touch(bid_px=bid_px, bid_sz=bid_sz,
                                       ask_px=ask_px, ask_sz=ask_sz))

    def flush(self, ts: float) -> OfiBar | None:
        """Close whatever is open, as CHOP. For end of file or session."""
        if self._bar is None or self._bar.updates == 0:
            return None
        return self._close(CHOP, ts, self._bar.close)

    # ------------------------------------------------------------------
    def _open_bar(self, ts: float, mid: float) -> None:
        self._bar = OfiBar(ts_open=ts, ts_close=ts, open=mid, high=mid,
                           low=mid, close=mid, direction=CHOP)

    def _close(self, direction: str, ts: float, mid: float) -> OfiBar:
        bar = self._bar
        assert bar is not None
        bar.direction = direction
        bar.ts_close = ts
        bar.close = mid
        bar.ofi = self._acc
        self.bars.append(bar)
        # THE RESET. See the module docstring -- without this the
        # threshold latches and every bar after the first is green.
        self._acc = 0
        self._open_bar(ts, mid)
        return bar
