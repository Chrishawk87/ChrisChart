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
from typing import Callable

from . import dbn
from .flow import FlowTape
from .live import STALE_AFTER_S, CandleBuilder, LiveCandle

DEFAULT_INTERVALS = (60.0, 300.0, 900.0)      # 1m, 5m, 15m
KEEP_CLOSED = 300                             # per interval

# How much raw tape the flow ladder can reach back through. An hour is
# what the other venues keep, and the ladder reports coverage honestly,
# so a 4-hour rung simply says it is not covered rather than inventing
# a number from the fragment it can see.
TAPE_SECONDS = 3600.0

# And a hard ceiling on how many trades that hour may contain. Measured
# at 136 bytes a trade, an hour of an active ES session is 100-250MB --
# the age bound alone does not bound memory, because the trade rate is
# the market's decision. 250k trades is about 34MB.
TAPE_MAX_TRADES = 250_000


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


def seconds_of(name: str) -> float:
    """'15m' -> 900. The inverse of `label`, for a selector that can
    name any timeframe."""
    raw = str(name).strip().lower()
    if not raw or not raw[:-1].isdigit():
        return 0.0
    n, unit = int(raw[:-1]), raw[-1]
    return float(n * {"s": 1, "m": 60, "h": 3600, "d": 86400}.get(unit, 0))


@dataclass
class Tape:
    """Trades from a CME record stream, as candles per timeframe."""

    symbol: str = "ES"
    intervals: tuple = DEFAULT_INTERVALS
    keep: int = KEEP_CLOSED

    builders: dict = field(default_factory=dict)
    closed: dict = field(default_factory=dict)
    book: object = None
    # The raw trades, kept as well as folded into candles. The candle
    # builders answer "what did this bar do"; the ladder asks "which
    # side has been paying, over these last N seconds", and a closed
    # bar cannot answer that for the bar still forming.
    flow: object = None
    # A footprint stream, when one is attached. Fed from the SAME
    # trade this method already decoded, so the footprint and the
    # candles can never disagree about which prints existed.
    footprint: object = None
    # The resting book, read diagonally and scoped to the footprint's
    # candle. Separate from `footprint` on purpose: one counts what
    # traded, the other what is standing there, and neither
    # substitutes for the other.
    bookdiag: object = None
    # Whether the flow can actually get through that standing size.
    # Scoped to the same candle as the footprint and the book reading.
    obstacles: object = None
    # Who is ahead inside the footprint's bar, and when they stop
    # gaining. Fed from the same decoded trade as the footprint, so
    # the lead and the ladder can never disagree about the prints.
    lead: object = None
    # How FAST each side is accumulating, which is a different
    # question from who is ahead and the one that turns first.
    pace: object = None
    # The two drawn lines: book CVD as the reader, time-smoothed
    # traded CVD as the confirmation. Fed from the same decoded trade
    # as everything else above, so the lines and the candles can never
    # disagree about which prints existed. Nothing is pushed anywhere
    # -- the panel polls.
    cvdcross: object = None
    last_px: float = 0.0
    last_side: str = ""
    last_ts: float = 0.0
    trades: int = 0
    records: int = 0
    books: int = 0
    # Functions registered by SSE streams; called on each new trade so a
    # stream can push a price update without polling. Same pattern as
    # LiveFeed._listeners -- a snapshot copy before iterating makes
    # concurrent register/unregister safe enough under the GIL.
    _listeners: list = field(default_factory=list)
    # Functions registered by WebSocket book streams; called on EVERY
    # MBP-10 record that carries book levels, not only on trades.
    # This fires hundreds of times per second in a live session and is
    # the hook that lets /ws/book push deltas at tape speed.
    _book_listeners: list = field(default_factory=list)
    # Execution-track callbacks. Called on every TRADE with the decoded
    # Trade object as the sole argument: fn(trade).  Separate from the
    # no-arg _listeners because the execution track needs the raw trade
    # data (price, size, aggressor) without another lookup, and keeping
    # it separate keeps the visualization path free of execution logic.
    _exec_listeners: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.flow is None:
            self.flow = FlowTape(max_age=TAPE_SECONDS,
                                 max_trades=TAPE_MAX_TRADES)
        for s in self.intervals:
            name = label(s)
            self.closed[name] = deque(maxlen=self.keep)
            self.builders[name] = CandleBuilder(
                s, on_close=self._closer(name))

    def _closer(self, name: str):
        def on_close(candle) -> None:
            self.closed[name].append(candle)
        return on_close

    # --------------------------------------------------------- listeners

    def on_update(self, fn: Callable) -> None:
        """Register a callback fired on each trade. Called from the record
        processing thread; the callback should be fast and non-blocking."""
        self._listeners.append(fn)

    def off_update(self, fn: Callable) -> None:
        """Unregister. Mirrors LiveFeed.off_update for the SSE layer."""
        try:
            self._listeners.remove(fn)
        except ValueError:
            pass

    @property
    def listener_count(self) -> int:
        return len(self._listeners)

    def on_book_update(self, fn: Callable) -> None:
        """Register a callback fired on every MBP-10 book record.

        Unlike on_update (trades only), this fires on every record that
        carries book levels -- Add, Modify, Delete, and Trade records all
        update the book.  In a live CME session that is hundreds of times
        per second.  The callback MUST be fast and non-blocking; use
        asyncio.Queue + loop.call_soon_threadsafe for async consumers.
        """
        self._book_listeners.append(fn)

    def off_book_update(self, fn: Callable) -> None:
        """Unregister a book-update callback."""
        try:
            self._book_listeners.remove(fn)
        except ValueError:
            pass

    @property
    def book_listener_count(self) -> int:
        return len(self._book_listeners)

    def on_trade_exec(self, fn: Callable) -> None:
        """Register a callback fired on every trade, receiving the Trade object.

        Called from the record-processing thread; fn must be fast and
        non-blocking.  The Trade object carries ts, px, sz, and aggressor.
        Separate from on_update (no-arg) so execution consumers get the raw
        print data without a second lookup, and so the visualization path
        never touches execution callbacks.
        """
        self._exec_listeners.append(fn)

    def off_trade_exec(self, fn: Callable) -> None:
        """Unregister an execution callback."""
        try:
            self._exec_listeners.remove(fn)
        except ValueError:
            pass

    @property
    def exec_listener_count(self) -> int:
        return len(self._exec_listeners)

    # -------------------------------------------------- historical seed

    def seed_interval(self, interval: str, candles: list) -> None:
        """Pre-populate one builder with historical OHLCV candles.

        The live subscription only carries prints from the moment it
        connects, so a chart opened at 10:00 has no 9:00 context.
        Historical bars fetched at startup fill that gap: closed bars
        are stored in the builder and the in-progress bar is adopted
        as the seed, exactly as `LiveFeed.seed` does for Hyperliquid.

        Records already seen from the live stream are not displaced --
        `seed_interval` is a no-op once the builder has closed bars of
        its own that are newer than anything in the historical batch.
        """
        from .structure import Candle as _Candle
        b = self.builders.get(interval)
        if b is None:
            secs = seconds_of(interval)
            if not secs:
                return
            self.closed.setdefault(interval, deque(maxlen=self.keep))
            b = CandleBuilder(secs, on_close=self._closer(interval))
            self.builders[interval] = b

        if not candles:
            return

        # The builder already has live data that post-dates the history.
        # Seeding over it would regress the candles backwards.
        if b.closed:
            newest_live = b.closed[-1].start_ts
            newest_hist = max(getattr(c, "ts", 0.0) for c in candles)
            if newest_hist <= newest_live:
                return

        import time as _time
        now = _time.time()
        conv: list[LiveCandle] = [
            LiveCandle.from_candle(c, b.interval_s) for c in candles]

        last = candles[-1]
        last_ts = getattr(last, "ts", 0.0)
        if now - last_ts < b.interval_s:
            b.seed(last)                          # bar is still forming
            b.closed = conv[:-1][-b.max_closed:]
        else:
            b.closed = conv[-b.max_closed:]

    # ------------------------------------------------------------ feed

    def offer(self, rec) -> bool:
        """One record in. True when it carried a trade.

        The book is updated from every record that has levels, because
        the book call wants the state as of now and most records are
        not trades.
        """
        self.records += 1
        if self.bookdiag is not None:
            try:
                self.bookdiag.observe(rec)
            except Exception:                           # noqa: BLE001
                pass            # a display must not stop the tape
        if dbn._get(rec, "levels"):
            try:
                self.book = dbn.book_from(rec, symbol=self.symbol)
                self.books += 1
            except Exception:                           # noqa: BLE001
                pass
            for _fn in list(self._book_listeners):
                try:
                    _fn()
                except Exception:                       # noqa: BLE001
                    pass
        t = dbn.trade_from(rec)
        if self.cvdcross is not None:
            # Book first here, deliberately, and the opposite way round
            # from `obstacles` below. The LCR asks what size was
            # standing at the level a print took, so the inside quote
            # must be current before the trade is measured against it.
            try:
                bk = self.book
                if bk is not None and dbn._get(rec, "levels"):
                    bids, asks = (bk.bids or ()), (bk.asks or ())
                    b, a = (bids[0] if bids else None), (asks[0] if asks else None)
                    self.cvdcross.on_book(
                        b.px if b else 0.0, b.sz if b else 0.0,
                        a.px if a else 0.0, a.sz if a else 0.0,
                        float(dbn.seconds(dbn._get(rec, "ts_event")) or 0.0))
                if t is not None:
                    self.cvdcross.on_trade(float(t.ts), t.aggressor,
                                           float(t.sz), float(t.px))
            except Exception:                           # noqa: BLE001
                pass            # a display must not stop the tape
        if self.obstacles is not None:
            # Trade FIRST, then the book from the same record. An
            # MBP-10 trade record carries the book as it stands after
            # the fill, so observing the book first would attribute
            # every fill to a wall that withdrew rather than one that
            # was eaten -- the exact confusion this module exists to
            # resolve.
            try:
                if t is not None:
                    sz = int(round(float(t.sz)))
                    if sz > 0:
                        self.obstacles.observe_trade(
                            float(t.ts), t.aggressor, sz, float(t.px))
                if dbn._get(rec, "levels"):
                    self.obstacles.observe_book(rec)
            except Exception:                           # noqa: BLE001
                pass            # a display must not stop the tape
        if t is None:
            return False
        self.trades += 1
        self.last_px, self.last_side, self.last_ts = t.px, t.aggressor, t.ts
        self.flow.add(t)
        for fn in list(self._exec_listeners):
            try:
                fn(t)
            except Exception:                       # noqa: BLE001
                pass
        if self.footprint is not None:
            self._to_footprint(t)
        for b in self.builders.values():
            b.add(t)
        for fn in list(self._listeners):
            try:
                fn()
            except Exception:
                pass
        return True

    def _to_footprint(self, t) -> None:
        """Hand one decoded trade to the footprint engine.

        Sizes are whole contracts there -- a footprint counts
        contracts at a price, and a fractional one is not a thing CME
        prints. A size that rounds to zero is dropped rather than
        passed as zero, which the engine refuses outright.
        """
        from .footprint import Tick as _FTick
        size = int(round(float(t.sz)))
        if size <= 0:
            return
        try:
            closed = self.footprint.on_tick(_FTick(
                ts=float(t.ts), price=float(t.px), size=size,
                aggressor=t.aggressor))
            if self.lead is not None:
                eng = getattr(self.footprint, "_engine", None)
                cur = getattr(eng, "current", None) if eng else None
                self.lead.on_bar(getattr(cur, "ts_open", None))
                self.lead.on_trade(float(t.ts), float(t.px), size,
                                   t.aggressor)
            if self.pace is not None:
                # ORDER MATTERS, AND IT IS OFF BY ONE PRINT IF WRONG.
                # The tick that closes a bar is the FIRST tick of the
                # next one -- the footprint counts it there. Clearing
                # the totals after attributing it would drop it, so
                # every bar would be short its opening print and the
                # panel's counts would disagree with the ladder above
                # it by exactly that much. `lead` is sequenced the
                # same way, with on_bar before on_trade.
                if closed:
                    self.pace.new_bar()
                self.pace.on_trade(float(t.ts), size, t.aggressor)
            # The book reading is scoped to the footprint's candle,
            # driven by the footprint's OWN close events rather than
            # by a poll -- a reset at poll time would scope it to the
            # refresh rate instead of to the bar. The footprint is
            # not modified to do this; its return value already says
            # when a bar ended.
            if closed:
                if self.bookdiag is not None:
                    self.bookdiag.reset()
                if self.obstacles is not None:
                    self.obstacles.reset()
                # `pace.new_bar()` is NOT here: it runs above, before
                # the closing tick is attributed. See the note there.
        except Exception:                               # noqa: BLE001
            # The footprint is a display. It must never be able to
            # stop the tape that feeds the engine and the candles.
            pass

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


@dataclass
class LadderFeed:
    """The CME tape, shaped like a live feed.

    The flow ladder was written against the Hyperliquid feed, but it
    only ever touches four things: `.tape` (walked for `_trades`),
    `.book` (read at index 0), `.age`, and `.coin`. Presenting those
    four is the whole adapter -- the measurement code is reused as
    written rather than copied and re-pointed, so the ES ladder and
    the BTC ladder cannot drift apart in what they mean.

    WHY IT REPORTS WHICH HALF IS MISSING. An L1 entitlement needs TWO
    subscriptions: `bbo-1s` carries a book and no trade, `trades`
    carries a trade and no book. With only one of them the ladder is
    not wrong, it is half-blind -- and a half-blind ladder draws the
    same empty rungs as a quiet market. `missing()` says which.
    """

    src: Tape
    coin: str = "ES"
    running: bool = False
    schemas: tuple = ()
    # What the service itself knows: a thread that died, or a config
    # condition that was never met. Without it an engine whose thread
    # raised at 3am reports exactly what a closed market reports.
    note: str = ""

    @property
    def tape(self):
        return self.src.flow

    @property
    def book(self):
        return self.src.book

    @property
    def age(self) -> float | None:
        """Seconds since the last print, or None if there has been none.

        None rather than 0.0, for the same reason the live feed says
        None: a tape that has never ticked has not just ticked.
        """
        import time as _t
        return (_t.time() - self.src.last_ts) if self.src.last_ts else None

    @property
    def started_at(self) -> float | None:
        return None

    def candle(self, interval: str, now: float | None = None):
        return self.src.candle(interval, now=now)

    def history(self, interval: str, limit: int = 200):
        return self.src.history(interval, limit=limit)


    # ---- the rest of the feed surface the candle read expects ------
    #
    # The read was written against the Hyperliquid feed and touches ten
    # things on it. Four are real here; the rest are answered honestly
    # rather than approximated, because a plausible number from the
    # wrong source is worse on that panel than a blank.

    bucket_s: float = 15.0          # written by the read; unused here

    @property
    def stale(self) -> bool:
        a = self.age
        return a is None or a > STALE_AFTER_S

    @property
    def builders(self) -> dict:
        return self.src.builders

    def ensure_interval(self, name: str, seed_bars=None) -> bool:
        """Start building a timeframe the tape was not opened with.

        The selector can name any interval. Adding the builder costs
        nothing -- the same prints already arriving feed all of them.
        """
        if name in self.src.builders:
            return False
        secs = seconds_of(name)
        if not secs:
            return False
        self.src.closed.setdefault(name, deque(maxlen=self.src.keep))
        self.src.builders[name] = CandleBuilder(
            secs, on_close=self.src._closer(name))
        return True

    def absorption(self, window_s: float = 300.0):
        """Not computed from the CME tape.

        The Hyperliquid feed measures this against a rolling baseline of
        its own impact buckets, which this tape does not keep. None is
        what the read already shows when a baseline is too thin to have
        a scale, so the panel handles it -- and an invented number
        there would read as a measurement.
        """
        return None

    # ---- stubs for attributes api_read accesses when use_feed is True --
    #
    # The read path reaches these because `feed = rt.cme_feed(coin) or
    # rt.feed_for(coin)` hands it a LadderFeed for ES. Both are honest:
    # the CME tape does not keep the HL feed's impact-bucket baseline, and
    # there is no book-history deque to derive a spread trend from.

    @property
    def baseline(self):
        """Impact baseline stub. Not computed from the CME tape.

        api_read accesses `.baseline.samples` to report how far along
        the absorption warm-up is. CME absorption is not computed here,
        so samples is always 0 rather than a number that would imply
        something is being measured when it isn't.
        """
        from types import SimpleNamespace
        return SimpleNamespace(samples=0, ready=False)

    def spread_trend(self, window_s: float = 60.0) -> dict:
        """Spread trend stub. Not tracked for the CME tape.

        LiveFeed keeps a deque of (time, spread_bps, imbalance, mid)
        snapshots and computes drift from them. LadderFeed has no such
        deque -- the book arrives infrequently and is not archived.
        Returning samples=0 tells the caller there is no history yet,
        which is what the panel already handles for a feed that just
        connected.
        """
        return {"samples": 0}

    def status(self) -> dict:
        return {
            "running": self.running,
            "connected": self.running,
            "coin": self.coin,
            "source": "cme",
            "schemas": list(self.schemas),
            "stale": self.stale,
            "age_s": self.age,
            "trades": self.src.trades,
            "book_updates": self.src.books,
            "records": self.src.records,
            "why": self.missing(),
        }

    def missing(self) -> str:
        """Why the ladder cannot be drawn. '' when it can.

        THREE DIFFERENT SILENCES, which look identical from the panel
        and need opposite fixes:

          nothing subscribed   the engine is off or was never configured
          nothing arriving     the subscription is open and the market
                               is shut, or the entitlement is refused
          arriving, wrong kind records are coming but not the ones the
                               ladder needs -- a schema problem

        `src.records` is what separates the second from the third: it
        counts every record the tape was offered, trade or not.
        """
        if not self.running:
            return ("the CME engine is not running"
                    + (f" -- {self.note}" if self.note else ""))

        if self.src.records == 0:
            # Open, and nothing has come down it. Outside the session
            # this is the correct and expected state, so it is named
            # before anything that sounds like a fault.
            why = ("the subscription is open but no records have "
                   "arrived at all. CME equity index futures trade "
                   "Sunday 17:00 to Friday 16:00 Central, with a break "
                   "16:00-17:00 each day -- outside those hours this is "
                   "a closed market, not a fault")
            if self.schemas:
                why += f" (subscribed: {', '.join(self.schemas)})"
            if self.note:
                why += f". The engine also reports: {self.note}"
            return why

        # Records ARE arriving, so the feed is alive and entitled --
        # they are simply not the kind the ladder reads.
        gaps = []
        if self.src.trades == 0:
            gaps.append(f"{self.src.records:,} records have arrived but no "
                        f"prints -- the ladder reads the aggressor side, "
                        f"which only the `trades` schema carries")
        if self.src.book is None:
            gaps.append(f"{self.src.records:,} records have arrived but no "
                        f"book -- the spread gate needs a bid and an ask, "
                        f"which `trades` does not carry and `bbo-1s` or "
                        f"`mbp-1` does")
        if gaps and self.schemas:
            gaps.append(f"subscribed: {', '.join(self.schemas)}")
        return "; ".join(gaps)
