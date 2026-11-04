"""Candles that close only when BOTH legs are satisfied.

A tick bar closes on a count of prints. An OFI bar closes on a
quantity of order-flow imbalance. Each one is blind to what the other
measures, and each has a characteristic way of lying:

    A TICK BAR closes on four hundred one-lots as readily as on four
    hundred fifty-lots. Activity without size.

    AN OFI BAR closes on one institutional block working the touch.
    Size without participation.

A FUSED BAR closes when both have happened: `tick_target` prints AND
`ofi_target` contracts of signed imbalance since the bar opened. A
quiet burst of tiny trades does not close it -- not enough
contracts. A single huge print does not close it either -- not enough
prints. What survives is the thing worth looking at: meaningful
activity carrying meaningful size.

THE OFI LEG IS SIGNED, AND THAT IS THE WHOLE POINT.

    The accumulator is the one the OFI bars already use: it resets at
    every close and it can go either way, so the bar qualifies when
    it crosses +target or -target. It means THE BOOK COMMITTED IN A
    DIRECTION. A bar that churned two-sided all day never qualifies
    on this leg however busy it was, and that is correct -- churn is
    not commitment.

    Running a global accumulator instead would make the threshold a
    latch rather than a filter: cumulative OFI tracks cumulative
    price, so on any session with a direction it walks away from zero
    and never returns. `ofibar` measured that construction printing
    99.9% green. The reset is not a detail.

THE TRAP, AND THE TIME CAP THAT ANSWERS IT.

    In a market that is all size and few prints -- institutional
    tape, thin hours -- the tick leg never fires and the bar never
    closes. Left alone that is one enormous candle and a chart that
    has stopped saying anything.

    So a bar also closes on `max_s` seconds, and when it does it is
    MARKED, not quietly passed off as a normal one. `close_reason` is
    'both' or 'capped', and a capped bar is almost always untradeable
    by the rule below because it did not reach the print count.

    One honest limitation: this is event-driven, so the cap fires on
    the first event AFTER the deadline, not on a timer. In a market
    quiet enough for that to matter, nothing is arriving anyway.

DIRECTION, COLOUR, AND THE GRAY ONES.

    DIRECTION is the OFI sign at close. COLOUR follows it: green when
    the accumulator ended positive, red when negative. The colour
    describes THE BOOK, which is how the OFI candles on this chart
    already read.

    TRADEABLE is a separate flag and a stricter test: the bar must
    have reached the print count AND its own price direction must
    agree with the OFI sign. When they disagree the candle still
    prints -- it happened, it is data -- but it is marked
    untradeable, drawn gray, and skipped. A bar whose book pushed one
    way while price went the other is exactly the bar that looks like
    a signal and is not.

    Tick direction here is the TRADED price, close against open. Not
    delta, not the mid. The tick leg counts prints, so the direction
    that leg is entitled to an opinion about is the one those prints
    made. `delta_dir` is carried alongside for the cases where the
    two differ, but it does not decide tradeability.

Nothing in this module places, sizes, routes or times an order. It
counts prints, accumulates imbalance, and closes bars.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass

from .ofi import Touch, level_ofi

# The defaults Chris named: 5,000 prints and 2,000 contracts.
TICK_TARGET = 5_000
OFI_TARGET = 2_000
# The fallback. Same value the OFI bars cap a chop bar at.
MAX_S = 900.0

# Bounds for anything arriving from a query string.
MIN_TICKS, MAX_TICKS = 50, 200_000
MIN_OFI, MAX_OFI = 50, 200_000

KEEP = 1_000

# The sizes offered on the panel, as (prints, contracts).
PAIRS: tuple[tuple[int, int], ...] = (
    (1_000, 500),
    (2_500, 1_000),
    (5_000, 2_000),
    (10_000, 4_000),
)

# The pair the fused STRATEGY reads when nothing says otherwise. The
# one Chris named.
DEFAULT_PAIR: tuple[int, int] = (TICK_TARGET, OFI_TARGET)


@dataclass
class FusedBar:
    """One bar, and the two legs that had to agree to close it."""

    start_ts: float
    tick_target: int = TICK_TARGET
    ofi_target: int = OFI_TARGET

    # Traded price. Zero until the first print, which is a real state
    # for a bar in thin tape and must not read as a price of zero.
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0

    # The mid, carried separately because the OFI formula is defined
    # against it. Never mixed into the OHLC above: two price frames in
    # one field is a bug that hides for weeks.
    mid_open: float = 0.0
    mid_close: float = 0.0

    ts_last: float = 0.0
    volume: float = 0.0
    prints: int = 0
    books: int = 0
    buy_size: float = 0.0
    sell_size: float = 0.0
    unknown: int = 0
    unknown_size: float = 0.0
    net_ofi: int = 0
    close_reason: str = ""          # 'both' | 'capped' | ''
    closed: bool = False

    # -- the two legs ----------------------------------------------------

    @property
    def ticks_met(self) -> bool:
        return self.prints >= self.tick_target

    @property
    def ofi_met(self) -> bool:
        return abs(self.net_ofi) >= self.ofi_target

    @property
    def both_met(self) -> bool:
        return self.ticks_met and self.ofi_met

    @property
    def seconds(self) -> float:
        return max(0.0, float(self.ts_last) - float(self.start_ts))

    # -- what it says ----------------------------------------------------

    @property
    def ofi_dir(self) -> str:
        if self.net_ofi > 0:
            return "up"
        if self.net_ofi < 0:
            return "down"
        return "flat"

    @property
    def direction(self) -> str:
        """The bar's direction IS the OFI sign. Chris's rule."""
        return self.ofi_dir

    @property
    def colour(self) -> str:
        return {"up": "green", "down": "red"}.get(self.ofi_dir, "flat")

    @property
    def tick_dir(self) -> str:
        """The traded price, close against open."""
        if not self.open or not self.close:
            return "flat"
        if self.close > self.open:
            return "up"
        if self.close < self.open:
            return "down"
        return "flat"

    @property
    def delta(self) -> float:
        return self.buy_size - self.sell_size

    @property
    def delta_dir(self) -> str:
        if self.delta > 0:
            return "up"
        if self.delta < 0:
            return "down"
        return "flat"

    @property
    def tradeable(self) -> bool:
        """Enough prints, and price agreed with the book.

        Deliberately stricter than the close condition: a bar that
        closed on the time cap has not met the print count, so it
        fails here even though it is a real, drawn bar.
        """
        return (self.ticks_met
                and self.ofi_dir in ("up", "down")
                and self.tick_dir == self.ofi_dir)

    @property
    def why_not(self) -> str:
        """Why this bar is gray. Never blank when it is."""
        if self.tradeable:
            return ""
        if not self.prints:
            return "no prints in the bar at all"
        if not self.ticks_met:
            return (f"closed on the {int(MAX_S)}s cap with "
                    f"{self.prints:,} prints, short of "
                    f"{self.tick_target:,}")
        if self.ofi_dir == "flat":
            return "the OFI accumulator ended exactly flat"
        if self.tick_dir == "flat":
            return "price closed where it opened"
        return (f"the book ended {self.ofi_dir} but price went "
                f"{self.tick_dir}")

    def to_dict(self) -> dict:
        """The chart's row shape, plus everything the bar is made of."""
        return {
            "ts": round(float(self.start_ts), 3),
            "o": self.open, "h": self.high,
            "l": self.low, "c": self.close,
            "v": round(self.volume, 2),
            "ts_last": round(float(self.ts_last), 3),
            "seconds": round(self.seconds, 2),
            "prints": self.prints,
            "books": self.books,
            "net_ofi": int(self.net_ofi),
            "tick_target": self.tick_target,
            "ofi_target": self.ofi_target,
            "ticks_met": self.ticks_met,
            "ofi_met": self.ofi_met,
            "close_reason": self.close_reason,
            "capped": self.close_reason == "capped",
            "dir": self.direction,
            "colour": self.colour,
            "tick_dir": self.tick_dir,
            "delta": round(self.delta, 2),
            "delta_dir": self.delta_dir,
            "buy": round(self.buy_size, 2),
            "sell": round(self.sell_size, 2),
            "unknown": self.unknown,
            "unknown_size": round(self.unknown_size, 2),
            "mid_open": self.mid_open,
            "mid_close": self.mid_close,
            "tradeable": self.tradeable,
            "why_not": self.why_not,
            "closed": bool(self.closed),
            "live": not self.closed,
        }


class FusedBars:
    """Builds dual-trigger bars from book updates and prints.

    O(1) per event. Fed from the RAW record on both sides: the book
    from the integer levels, because the OFI formula compares prices
    for exact equality and a float that round-tripped through a
    division silently takes the wrong branch; the prints from the raw
    side character, because `dbn.trade_from` returns None for side
    'N' and a tick leg fed through it would need N ATTRIBUTABLE
    prints and close at a different rate than the panel claims.
    """

    __slots__ = ("tick_target", "ofi_target", "max_s", "keep", "done",
                 "_cur", "_prev", "_built", "_capped", "_tradeable",
                 "_prints")

    def __init__(self, tick_target: int = TICK_TARGET,
                 ofi_target: int = OFI_TARGET,
                 max_s: float = MAX_S, keep: int = KEEP) -> None:
        # Floors of 1 only. The sane-size clamp belongs at the
        # boundary where an untrusted number arrives, not in the
        # core, where it would make short bars impossible to measure.
        self.tick_target = max(1, int(tick_target or TICK_TARGET))
        self.ofi_target = max(1, int(ofi_target or OFI_TARGET))
        self.max_s = max(1.0, float(max_s or MAX_S))
        self.keep = max(1, int(keep))
        self.done: deque[FusedBar] = deque(maxlen=self.keep)
        self._cur: FusedBar | None = None
        # The previous touch, carried ACROSS bars. The OFI resets at
        # a close but the book does not restart, and treating the
        # first update of a new bar as if there were no previous one
        # would discard a real move at every boundary.
        self._prev: Touch | None = None
        self._built = 0
        self._capped = 0
        self._tradeable = 0
        self._prints = 0

    # -- feeding ---------------------------------------------------------

    def _start(self, ts: float) -> FusedBar:
        return FusedBar(start_ts=ts, ts_last=ts,
                        tick_target=self.tick_target,
                        ofi_target=self.ofi_target)

    def _maybe_close(self, bar: FusedBar) -> FusedBar | None:
        """Close on both legs, or on the cap. Returns the closed bar."""
        reason = ""
        if bar.both_met:
            reason = "both"
        elif bar.seconds >= self.max_s:
            reason = "capped"
        if not reason:
            return None
        bar.close_reason = reason
        bar.closed = True
        self.done.append(bar)
        self._built += 1
        if reason == "capped":
            self._capped += 1
        if bar.tradeable:
            self._tradeable += 1
        self._cur = None
        return bar

    def on_book(self, ts: float, bid_px: int, bid_sz: int,
                ask_px: int, ask_sz: int) -> FusedBar | None:
        """One book update. Moves the OFI leg; never the tick leg."""
        t = Touch(bid_px=bid_px, bid_sz=bid_sz,
                  ask_px=ask_px, ask_sz=ask_sz)
        if not t.valid or t.crossed:
            return None
        bar = self._cur
        if bar is None:
            bar = self._cur = self._start(ts)
        if self._prev is not None:
            bar.net_ofi += int(level_ofi(self._prev, t))
        self._prev = t
        mid = (bid_px + ask_px) / 2.0
        if not bar.books:
            bar.mid_open = mid
        bar.mid_close = mid
        bar.books += 1
        bar.ts_last = max(bar.ts_last, float(ts))
        return self._maybe_close(bar)

    def on_trade(self, ts: float, px: float, size: float,
                 side: str) -> FusedBar | None:
        """One print. Moves the tick leg; never the OFI leg.

        A print with no price or no size is not a trade and does not
        advance the count -- letting a zero through would close bars
        on records that never happened and put a 0.0 in the OHLC.
        """
        try:
            p = float(px or 0.0)
            sz = float(size or 0.0)
            t = float(ts or 0.0)
        except (TypeError, ValueError):
            return None
        if p <= 0 or sz <= 0:
            return None

        self._prints += 1
        bar = self._cur
        if bar is None:
            bar = self._cur = self._start(t)
        if not bar.open:
            bar.open = bar.high = bar.low = p
        else:
            if p > bar.high:
                bar.high = p
            if p < bar.low:
                bar.low = p
        bar.close = p
        bar.volume += sz
        bar.prints += 1
        bar.ts_last = max(bar.ts_last, t)

        s = (side or "").strip().upper()[:1]
        if s == "B":
            bar.buy_size += sz
        elif s == "A":
            bar.sell_size += sz
        else:
            bar.unknown += 1
            bar.unknown_size += sz
        return self._maybe_close(bar)

    # -- reading ---------------------------------------------------------

    @property
    def current(self) -> FusedBar | None:
        return self._cur

    @property
    def last_closed(self) -> FusedBar | None:
        return self.done[-1] if self.done else None

    def rows(self, limit: int = 200, include_live: bool = True) -> list[dict]:
        n = max(1, int(limit))
        out = [b.to_dict() for b in list(self.done)[-n:]]
        if include_live and self._cur is not None:
            out.append(self._cur.to_dict())
        return out

    def progress(self) -> dict:
        """How far the bar being built is along each leg.

        Both numbers, never one: the whole idea is that a bar can sit
        at 98% on one leg and 4% on the other, and a single
        percentage would hide exactly that.
        """
        b = self._cur
        if b is None:
            return {"building": False,
                    "say": "no bar open -- the last print closed one"}
        tf = min(1.0, b.prints / self.tick_target) if self.tick_target else 0.0
        of = (min(1.0, abs(b.net_ofi) / self.ofi_target)
              if self.ofi_target else 0.0)
        waiting = ("the book" if b.ticks_met and not b.ofi_met else
                   "prints" if b.ofi_met and not b.ticks_met else
                   "both" if not b.ticks_met and not b.ofi_met else "")
        return {
            "building": True,
            "prints": b.prints, "tick_target": self.tick_target,
            "tick_pct": round(100.0 * tf, 1),
            "net_ofi": int(b.net_ofi), "ofi_target": self.ofi_target,
            "ofi_pct": round(100.0 * of, 1),
            "elapsed_s": round(b.seconds, 1),
            "cap_s": self.max_s,
            "waiting_on": waiting,
            "say": _progress_words(b, self.tick_target, self.ofi_target,
                                   self.max_s, waiting),
            # THE INTRABAR READ, on the bar being built. This is what
            # the fused strategy trades off -- not the closed bar.
            "live_side": live_signal(b)[0],
            "live_why": live_signal(b)[1],
            "ofi_dir": b.ofi_dir,
            "tick_dir": b.tick_dir,
        }

    def summary(self) -> dict:
        built = self._built
        cap_pct = (100.0 * self._capped / built) if built else 0.0
        tr_pct = (100.0 * self._tradeable / built) if built else 0.0
        return {
            "tick_target": self.tick_target,
            "ofi_target": self.ofi_target,
            "max_s": self.max_s,
            "bars": built,
            "held": len(self.done),
            "prints": self._prints,
            "capped": self._capped,
            "capped_pct": round(cap_pct, 1),
            "tradeable": self._tradeable,
            "tradeable_pct": round(tr_pct, 1),
            # THE NUMBER THAT SAYS WHETHER THE FUSION IS WORKING ON
            # THIS TAPE. A high capped share means the trap is live:
            # the tick leg is not firing and the cap is doing the
            # closing, so these are not really dual-trigger bars.
            "say": _summary_words(built, cap_pct, tr_pct),
        }


def _progress_words(b: FusedBar, tick_target: int, ofi_target: int,
                    max_s: float, waiting: str) -> str:
    left = max(0.0, max_s - b.seconds)
    if waiting == "both":
        return (f"{b.prints:,}/{tick_target:,} prints and "
                f"{abs(b.net_ofi):,}/{ofi_target:,} contracts -- "
                f"{left:.0f}s before the cap closes it")
    if waiting == "prints":
        return (f"the book is there ({b.net_ofi:+,}), waiting on prints: "
                f"{b.prints:,} of {tick_target:,}")
    if waiting == "the book":
        return (f"the prints are there ({b.prints:,}), waiting on the "
                f"book: {abs(b.net_ofi):,} of {ofi_target:,}")
    return "both legs met -- the next event closes it"


def _summary_words(built: int, cap_pct: float, tr_pct: float) -> str:
    if built < 10:
        return (f"only {built} bars so far -- too few to say anything "
                "about how the pair is behaving")
    if cap_pct >= 60:
        return (f"{cap_pct:.0f}% of bars closed on the CAP, not on both "
                "legs. These are mostly not dual-trigger bars: either "
                "the targets are too high for this tape or the trap is "
                "live and the prints are not coming")
    if cap_pct >= 25:
        return (f"{cap_pct:.0f}% closed on the cap -- the tick leg is "
                "missing often enough to be worth lowering it")
    return (f"{cap_pct:.0f}% capped, {tr_pct:.0f}% tradeable -- the pair "
            "is closing on both legs as intended")


# How much of each leg has to be in before an intrabar agreement is
# believed. NOT the full target -- waiting for that is waiting for the
# close, which is the lateness this whole series exists to fix. But
# not zero either: two prints and one book update agreeing is not
# agreement, it is noise with a direction, and Chris's own rule from
# the window work was that nothing is taken at the very opening of a
# candle because that is when it is least settled.
MIN_FRACTION = 0.10


def live_signal(bar: FusedBar | None,
                min_fraction: float = MIN_FRACTION) -> tuple[str, str]:
    """INTRABAR. The direction the two legs agree on right now.

    Returns (side, why_not) and the reason is never blank when the
    side is: a rule that goes quiet without saying why reads as a
    market with nothing in it, and on this panel that has meant a
    missing read, a stale cap and a switch left off.

    THIS DOES NOT WAIT FOR THE BAR TO CLOSE, and that is the whole
    point of it. `FusedBar.tradeable` is the close-time flag -- it
    requires the full print count and it is what greys a finished
    candle on the chart. Using it as the entry test would mean
    entering one bar late every time, which is the complaint that
    started this.

    So the test here is only: enough has arrived to be worth
    believing, the book has a direction, and the traded price agrees
    with it. The moment those hold, the trade is on.
    """
    if bar is None:
        return "", "no bar open -- the last event closed one"

    need_t = max(1, int(bar.tick_target * min_fraction))
    need_o = max(1, int(bar.ofi_target * min_fraction))
    pct = int(round(min_fraction * 100))

    if bar.prints < need_t:
        return "", (f"only {bar.prints:,} prints in -- {need_t:,} "
                    f"({pct}% of the bar) before an agreement counts")
    if abs(bar.net_ofi) < need_o:
        return "", (f"the book has only moved {abs(bar.net_ofi):,} "
                    f"-- {need_o:,} ({pct}% of the bar) before an "
                    "agreement counts")
    if bar.ofi_dir == "flat":
        return "", "the book is exactly flat"
    if bar.tick_dir == "flat":
        return "", "price is where the bar opened"
    if bar.tick_dir != bar.ofi_dir:
        return "", (f"the book is {bar.ofi_dir} but price is "
                    f"{bar.tick_dir}")
    return bar.ofi_dir, ""


_FUSED = re.compile(r"^(\d+)x(\d+)$")


def parse_interval(interval: str) -> tuple[int, int] | None:
    """`'5000x2000'` -> (5000, 2000). Anything else -> None.

    This is what lets a fused series travel through the EXISTING
    chart route. `'15m'` and `'1000t'` both return None and the
    caller falls through to the path it already used, so adding this
    changes no existing behaviour.
    """
    m = _FUSED.match(str(interval or "").strip().lower())
    if not m:
        return None
    ticks = int(m.group(1))
    ofi = int(m.group(2))
    if ticks <= 0 or ofi <= 0:
        return None
    return (max(MIN_TICKS, min(ticks, MAX_TICKS)),
            max(MIN_OFI, min(ofi, MAX_OFI)))


def label(ticks: int, ofi: int) -> str:
    return f"{int(ticks)}x{int(ofi)}"
