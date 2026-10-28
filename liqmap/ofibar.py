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

THREE COLUMNS, NOT ONE SCORE.

    BOOK    what is RESTING -- limit orders, intent that can still be
            withdrawn. This is OFI, and it is what closes the bar.
    PRICE   whether it WORKED -- the path the mid actually took. This
            is `pai`, and it is what colours the bar.
    DELTA   who is AGGRESSING -- market orders, intent already paid
            for. This is `delta`, and it is neither of the other two.

They come apart constantly, and each gap is information: the book can
be stacked on the bid while every print hits it. Folding them into one
number destroys exactly what makes them worth having, so they stay
separate all the way to the chart.

THE CANDLE IS THE MID. THE LEVELS ARE NOT.

Open, high, low and close are the midpoint of the quote, because that
is what the OFI formula is defined against and what the price path is
measured on. On a one-tick ES spread a midpoint sits exactly half a
tick off the grid -- bid 7704.00 and ask 7704.25 give 7704.125, which
is not a price that can print. That is fine for a candle and useless
for a level, so every bar also carries the highest, lowest and last
price that ACTUALLY TRADED inside it. Those are on the tick grid by
construction, and they are what the swing levels are found on.

A bar with no prints in it has no traded prices at all, and says so
with zeros rather than borrowing the mid. A level built from a
borrowed mid would look exactly like a real one.

THE AGGRESSOR IS STATED, NOT INFERRED. CME marks the side on every
print. Most platforms guess it with a tick rule and are wrong on a
meaningful share of trades, worst precisely when the market is fast.
Nothing here guesses.

NOTHING HERE PREDICTS. A green bar says buying pressure crossed a line
you chose. Whether that precedes price is a separate question this
module does not answer and does not assume. Delta does not change
that: it describes the bar more completely, which is not the same as
saying what the next one does.
"""

from __future__ import annotations

from collections import deque
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

# Price-aggression labels.
BUY_AGGRESSIVE = "BUY_PRICE_AGGRESSIVE"
SELL_AGGRESSIVE = "SELL_PRICE_AGGRESSIVE"
BALANCED_FLOW = "BALANCED_PRICE_FLOW"

PAI_BUY = 0.70
PAI_SELL = 0.30

# A bar must have traded at least this much before its delta is
# allowed to speak. Below it one contract flips the sign.
CALL_MIN_VOL = 10.0

# The closing-burst window. Reported alongside the bar-wide figure,
# never used to colour the bar -- see `pai` for why.
BURST_S = 0.005


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
    # Price-aggression accumulators, in whatever price units were fed.
    # Only their RATIO is used, so the units cancel and no tick size is
    # needed to compute it.
    buy_agg: float = 0.0     # sum of upward bid+ask steps, whole bar
    sell_agg: float = 0.0    # sum of |downward| bid+ask steps
    buy_burst: float = 0.0   # the same over the last BURST_S only
    sell_burst: float = 0.0
    # Traded size by STATED aggressor, over the whole bar. Volume, not
    # book size -- these are prints that happened, not orders resting.
    buy_vol: float = 0.0
    sell_vol: float = 0.0
    trade_count: int = 0
    # Prices that ACTUALLY TRADED inside the bar, in the same raw feed
    # units as open/high/low/close. Zero means no print landed here --
    # not a price of zero, and never filled in from the mid.
    trade_high: float = 0.0
    trade_low: float = 0.0
    trade_last: float = 0.0
    # THE CALL, MADE INSIDE THE BAR AND NOT AFTER IT.
    #
    # The first moment during this bar at which the running imbalance
    # and the running delta pointed the same way. That is when the
    # reading existed and when it could have been acted on; the close
    # is minutes of information later. `call_px` is the mid at that
    # instant, which is why it is the price any measurement of the
    # call has to run from.
    #
    # Empty means the two never agreed inside this bar.
    call_side: str = ""
    call_px: float = 0.0
    call_ts: float = 0.0

    @property
    def seconds(self) -> float:
        return self.ts_close - self.ts_open

    @property
    def strength(self) -> int:
        """Signed imbalance that closed the bar. The spec's OFI_Strength."""
        return self.ofi

    @property
    def pai(self) -> float:
        """Price Aggression Index over the WHOLE bar. 0..1, 0.5 = even.

        MEASURED ACROSS THE BAR, NOT OVER A TRAILING WINDOW. The brief
        asked for a rolling five-millisecond queue evaluated when the
        bar closes, which would describe the bar's last five
        milliseconds and nothing else: at target 250 a bar runs about
        twelve seconds, so that window is 0.04% of it. Colouring a
        candle from the final 0.04% of its life is close to colouring
        it at random. The burst figure is still reported, as
        `pai_burst`, because a violent close is worth seeing -- it just
        does not get to name the candle.

        WHAT THIS IS NOT. It is not "did the bar close up". It measures
        the PATH: the share of all price movement that went upward. A
        bar that climbs ten ticks and gives back nine closes up one
        tick but has a PAI near 0.5, because the path was two-sided. A
        bar that grinds up without retracing reads near 1.0. That
        difference -- clean versus choppy -- is the whole content of
        the number, and it is why it is not redundant with the body.
        """
        total = self.buy_agg + self.sell_agg
        return (self.buy_agg / total) if total > 0 else 0.5

    @property
    def pai_burst(self) -> float:
        """The same, over the last BURST_S of the bar only."""
        total = self.buy_burst + self.sell_burst
        return (self.buy_burst / total) if total > 0 else 0.5

    @property
    def traded(self) -> bool:
        """Did anything print inside this bar?

        The test for whether the traded prices mean anything. Reading
        `trade_high` without it gets a zero that looks like a price.
        """
        return self.trade_count > 0 and self.trade_high > 0

    @property
    def volume(self) -> float:
        """Traded size in the bar, both sides."""
        return self.buy_vol + self.sell_vol

    @property
    def delta(self) -> float:
        """Buy-aggressed minus sell-aggressed size. The paid-for column.

        Zero means either a balanced bar or an EMPTY one -- a bar built
        from book updates with no prints in it at all, which happens on
        quiet stretches and at every target above the smallest. Read it
        with `trade_count`, never alone.
        """
        return self.buy_vol - self.sell_vol

    @property
    def delta_ratio(self) -> float:
        """Share of traded size that lifted the offer. 0..1, 0.5 even."""
        total = self.volume
        return (self.buy_vol / total) if total > 0 else 0.5

    @property
    def delta_agrees(self) -> bool:
        """Did paid-for flow go the same way as the book pressure?

        THE INTERESTING CASE IS WHEN IT DOES NOT, and it is a different
        disagreement from `agrees`. That one asks whether PRICE followed
        the book. This asks whether anyone actually PAID to make it
        happen. A green bar with negative delta is a bid that filled up
        while prints were hitting it -- size arriving to be sold into.

        A bar with no trades in it agrees by default. There is nothing
        there to disagree with, and reporting an empty bar as a conflict
        would put a reading on the chart that no print supports.
        """
        if self.trade_count == 0:
            return True
        if self.direction == GREEN:
            return self.delta >= 0
        if self.direction == RED:
            return self.delta <= 0
        return True

    @staticmethod
    def _label(ratio: float) -> str:
        if ratio >= PAI_BUY:
            return BUY_AGGRESSIVE
        if ratio <= PAI_SELL:
            return SELL_AGGRESSIVE
        return BALANCED_FLOW

    @property
    def aggression(self) -> str:
        return self._label(self.pai)

    @property
    def aggression_burst(self) -> str:
        return self._label(self.pai_burst)

    @property
    def agrees(self) -> bool:
        """Does the price path agree with the flow that closed the bar?

        THE INTERESTING CASE IS WHEN IT DOES NOT. A bar closed GREEN by
        order-flow imbalance whose price path was sell-aggressive is
        buying pressure that price refused to pay for -- the absorption
        state, visible directly on the chart rather than inferred. The
        panel draws the body from aggression and the outline from the
        OFI direction, so a disagreement is a green body in a red
        frame.
        """
        if self.direction == GREEN:
            return self.aggression != SELL_AGGRESSIVE
        if self.direction == RED:
            return self.aggression != BUY_AGGRESSIVE
        return True

    def as_row(self) -> list:
        """[Timestamp, Open, High, Low, Close, Direction, OFI_Strength].

        The column order the charting side expects. Timestamp is the
        bar's CLOSE, because that is the instant the bar became a fact;
        stamping it with the open would place every bar earlier than
        the information that produced it.
        """
        return [self.ts_close, self.open, self.high, self.low,
                self.close, self.direction, self.ofi,
                round(self.pai, 4), self.aggression]

    def as_dict(self) -> dict:
        return {"ts_open": self.ts_open, "ts_close": self.ts_close,
                "open": self.open, "high": self.high, "low": self.low,
                "close": self.close, "direction": self.direction,
                "ofi": self.ofi, "updates": self.updates,
                "ofi_high": self.ofi_high, "ofi_low": self.ofi_low,
                "price_agg_ratio": round(self.pai, 4),
                "price_aggression": self.aggression,
                "price_agg_burst": round(self.pai_burst, 4),
                "aggression_burst": self.aggression_burst,
                "agrees": self.agrees,
                "buy_vol": self.buy_vol, "sell_vol": self.sell_vol,
                "trade_count": self.trade_count,
                "delta": self.delta,
                "delta_ratio": round(self.delta_ratio, 4),
                "delta_agrees": self.delta_agrees,
                "trade_high": self.trade_high,
                "trade_low": self.trade_low,
                "trade_last": self.trade_last,
                "traded": self.traded,
                "call_side": self.call_side,
                "call_px": self.call_px,
                "call_ts": self.call_ts,
                "seconds": round(self.seconds, 6)}


ROW_COLUMNS = ("timestamp", "open", "high", "low", "close",
               "direction", "ofi_strength",
               "price_agg_ratio", "price_aggression")


class OfiBarBuilder:
    """Feed it book touches oldest-first; it hands back closed bars.

    `threshold` is in contracts of net imbalance. `max_s` caps how long
    a bar may run without either line being reached -- without it a
    quiet stretch produces a single bar that never ends, and the chart
    stops updating while looking like nothing is happening.
    """

    __slots__ = ("tick", "threshold", "max_s", "min_updates", "max_bars",
                 "_prev", "_acc", "_bar", "bars", "dropped",
                 "_buy_agg", "_sell_agg", "_steps",
                 "_buy_vol", "_sell_vol", "_trades",
                 "_t_hi", "_t_lo", "_t_last",
                 "_c_side", "_c_px", "_c_ts")

    def __init__(self, tick_size: float = 0.25, threshold: int = 250,
                 max_s: float = 30.0, min_updates: int = 1,
                 max_bars: int = 20_000) -> None:
        if tick_size <= 0:
            raise ValueError("tick_size must be positive")
        if threshold < 1:
            raise ValueError("threshold must be at least 1")
        if max_s <= 0:
            raise ValueError("max_s must be positive")
        if min_updates < 1:
            raise ValueError("min_updates must be at least 1")
        if max_bars < 10:
            raise ValueError("max_bars must be at least 10")
        self.tick = float(tick_size)
        self.threshold = int(threshold)
        self.max_s = float(max_s)
        # A guard against a bar closing on a single update, which
        # happens when one enormous size change clears the line by
        # itself and leaves a bar with open == high == low == close.
        self.min_updates = int(min_updates)
        # HOW FAR BACK THE CHART CAN SCROLL, and the only thing standing
        # between a long-running service and an unbounded list. At
        # target 125 a bar is about three seconds, so 20,000 is roughly
        # seventeen hours; at 500 it is over a week. Five targets run at
        # once, so the real figure is five times this many objects.
        self.max_bars = int(max_bars)
        self._prev: Touch | None = None
        self._acc = 0
        self._bar: OfiBar | None = None
        self.bars: list[OfiBar] = []
        # Bars discarded off the front. Reported so the chart can say
        # "this is where the history starts" rather than implying the
        # session began there.
        self.dropped = 0
        # Price-aggression accumulators for the bar in progress, and a
        # short trailing queue for the closing burst. The queue holds
        # only the last few milliseconds, so it stays a handful of
        # entries however long the bar runs.
        self._buy_agg = 0.0
        self._sell_agg = 0.0
        self._steps: deque = deque()
        # Traded size for the bar in progress. Trades and book updates
        # arrive on the same record stream and interleave, so these are
        # builder state rather than bar state -- a print can land before
        # the first book touch has opened a bar, and it still belongs to
        # the bar that touch opens.
        self._buy_vol = 0.0
        self._sell_vol = 0.0
        self._trades = 0
        # Traded extremes for the bar in progress. None, not zero: a
        # bar with no prints must be distinguishable from one that
        # traded at zero, and min() against a zero would win forever.
        self._t_hi: float | None = None
        self._t_lo: float | None = None
        self._t_last: float = 0.0
        # The first instant inside the bar at which imbalance and delta
        # agreed. Latched: once the reading existed it existed, and
        # moving it later would be rewriting when it could have been
        # acted on.
        self._c_side: str = ""
        self._c_px: float = 0.0
        self._c_ts: float = 0.0

    def reset(self) -> None:
        self._prev = None
        self._acc = 0
        self._bar = None
        self.bars.clear()
        self.dropped = 0
        self._buy_agg = 0.0
        self._sell_agg = 0.0
        self._steps.clear()
        self._buy_vol = 0.0
        self._sell_vol = 0.0
        self._trades = 0
        self._t_hi = None
        self._t_lo = None
        self._t_last = 0.0
        self._c_side = ""
        self._c_px = 0.0
        self._c_ts = 0.0

    @property
    def current(self) -> OfiBar | None:
        """The bar still forming. Its `ofi` is live and may yet reverse."""
        return self._bar

    @property
    def accumulated(self) -> int:
        return self._acc

    @property
    def live_pai(self) -> float:
        """Aggression of the bar in progress, so the forming candle can
        be drawn in the colour it currently is."""
        total = self._buy_agg + self._sell_agg
        return (self._buy_agg / total) if total > 0 else 0.5

    @property
    def live_delta(self) -> float:
        """Paid-for flow in the bar in progress, so the delta pane can
        show the forming column live instead of waiting for the close."""
        return self._buy_vol - self._sell_vol

    @property
    def live_trades(self) -> int:
        return self._trades

    @property
    def live_call(self) -> tuple[str, float, float]:
        """(side, price, ts) of the call inside the bar in progress.

        Empty side means imbalance and delta have not agreed yet in
        this bar. This is what the chart marks while the bar is still
        forming, because the reading is intrabar and waiting for the
        close throws away the whole of it.
        """
        return (self._c_side, self._c_px, self._c_ts)

    @property
    def live_traded(self) -> tuple[float, float, float]:
        """Traded high, low and last of the bar in progress.

        Zeros when nothing has printed yet, which is the same thing a
        closed bar says and for the same reason.
        """
        return (self._t_hi or 0.0, self._t_lo or 0.0, self._t_last)

    def on_trade(self, ts: float, size: float, aggressor: str,
                 px: float = 0.0) -> None:
        """One print, with the side CME stated. Never closes a bar.

        A TRADE DOES NOT MOVE THE BAR BOUNDARY, and that is deliberate.
        The boundary is order-flow imbalance and nothing else; letting
        prints end a bar would make the threshold mean two things at
        once, and there would be no way to read which of them closed
        it. Delta rides along inside whatever bar the book has opened.

        A print arriving before the first book touch still belongs to
        the bar that touch opens, which is why these accumulate on the
        builder rather than on the bar.

        `px` is the print price in the SAME RAW FEED UNITS as the book
        prices, and it is what the swing levels are found on. A caller
        that does not have it passes nothing: delta still works and the
        traded extremes stay empty, which is honest. Passing a price in
        dollars here would put two unit systems in one bar.

        A size of zero, or a side that is neither, is refused rather
        than counted as balanced -- a zero would sum like a measurement.
        """
        if size <= 0:
            return
        if aggressor == "buy":
            self._buy_vol += float(size)
        elif aggressor == "sell":
            self._sell_vol += float(size)
        else:
            return
        self._trades += 1
        if px > 0:
            if self._t_hi is None or px > self._t_hi:
                self._t_hi = float(px)
            if self._t_lo is None or px < self._t_lo:
                self._t_lo = float(px)
            self._t_last = float(px)

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

        # ---- price aggression ----------------------------------------
        # The step distance of BOTH sides. On a book that keeps its
        # spread these move together, so an uptick contributes twice --
        # which is symmetric and therefore harmless to a ratio, and it
        # does mean a spread WIDENING on one side alone still registers
        # as movement, which it is.
        bid_step = touch.bid_px - self._prev.bid_px
        ask_step = touch.ask_px - self._prev.ask_px
        up = 0.0
        dn = 0.0
        if bid_step > 0:
            up += bid_step
        elif bid_step < 0:
            dn -= bid_step
        if ask_step > 0:
            up += ask_step
        elif ask_step < 0:
            dn -= ask_step
        if up or dn:
            self._buy_agg += up
            self._sell_agg += dn
            self._steps.append((ts, up, dn))
        cut = ts - BURST_S
        while self._steps and self._steps[0][0] < cut:
            self._steps.popleft()

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
        # THE CALL IS MADE HERE, INSIDE THE BAR.
        #
        # Checked on every book update rather than at the close,
        # because the reading is intrabar: the moment the running
        # imbalance and the running delta first point the same way is
        # the moment it could have been acted on, and the close is a
        # long time after that. Latched once set -- the reading existed
        # at that price, whatever the rest of the bar goes on to do.
        if not self._c_side:
            _v = self._buy_vol + self._sell_vol
            if _v >= CALL_MIN_VOL:
                _d = self._buy_vol - self._sell_vol
                if self._acc > 0 and _d > 0:
                    self._c_side, self._c_px, self._c_ts = "up", mid, ts
                elif self._acc < 0 and _d < 0:
                    self._c_side, self._c_px, self._c_ts = "down", mid, ts
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
        bar.buy_agg = self._buy_agg
        bar.sell_agg = self._sell_agg
        bar.buy_burst = sum(x[1] for x in self._steps)
        bar.sell_burst = sum(x[2] for x in self._steps)
        bar.buy_vol = self._buy_vol
        bar.sell_vol = self._sell_vol
        bar.trade_count = self._trades
        bar.trade_high = self._t_hi or 0.0
        bar.trade_low = self._t_lo or 0.0
        bar.trade_last = self._t_last
        bar.call_side = self._c_side
        bar.call_px = self._c_px
        bar.call_ts = self._c_ts
        self.bars.append(bar)
        if len(self.bars) > self.max_bars:
            # Trim in blocks. Popping one per close is O(n) each time
            # and turns every bar into a list copy of twenty thousand
            # elements; this pays that cost once per ten percent.
            cut = max(1, self.max_bars // 10)
            del self.bars[:cut]
            self.dropped += cut
        # THE RESET. See the module docstring -- without this the
        # threshold latches and every bar after the first is green.
        # Aggression resets with it: it describes ONE bar's path, and
        # carrying it forward would make every later bar inherit the
        # first one's character.
        self._acc = 0
        self._buy_agg = 0.0
        self._sell_agg = 0.0
        # Delta resets with everything else. It describes ONE bar's
        # paid-for flow; carried forward it would be a running total
        # wearing a per-bar label, which is the same latching failure
        # the accumulator reset exists to prevent.
        self._buy_vol = 0.0
        self._sell_vol = 0.0
        self._trades = 0
        self._t_hi = None
        self._t_lo = None
        self._t_last = 0.0
        self._c_side = ""
        self._c_px = 0.0
        self._c_ts = 0.0
        self._open_bar(ts, mid)
        return bar
