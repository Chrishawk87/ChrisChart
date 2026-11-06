"""Candles closed by a count of prints rather than by the clock.

A tick bar ends when N trades have printed, whatever that took. Quiet
hours produce one bar; a violent thirty seconds produces twenty. The
x-axis stops being time and becomes ACTIVITY, which is the whole
reason to look at one.

WHAT "TICK" MEANS HERE, SAID ONCE SO IT IS NOT GUESSED AT.

    A tick is ONE PRINT -- one trade record off the tape -- not one
    contract and not one minimum price increment. A 1,000-tick bar
    holds a thousand prints whether they were one lot or five
    hundred. That is the futures convention and it is deliberately
    NOT the same thing as the OFI bars already on this chart, which
    close on a quantity of contracts.

    The two answer different questions. A contract bar asks "how much
    changed hands"; a tick bar asks "how many decisions were made".
    One large print and four hundred small ones are the same bar to
    the first and wildly different bars to the second.

WHY THIS IS CALLED LEADING, AND WHERE THAT CLAIM STOPS.

    A tick bar carries no information a time bar lacks -- it is the
    same prints, cut differently. What it does is SPEND ITS RESOLUTION
    WHERE THE ACTIVITY IS. When the tape speeds up, a 15-minute candle
    is still one candle and still closes at :15; a tick series prints
    six bars in the same stretch and each one closes, with a
    direction, before the clock bar has an opinion.

    So the honest version of "tick charts lead" is: a rule reading
    tick bars can act at the Nth print, while the same rule reading
    time bars cannot act until the minute ends. The lead is in the
    CLOSING SCHEDULE, not in the data. A tick chart drawn beside a
    rule that still reads the clock changes nothing at all about when
    that rule fires.

THE BAR'S OWN PACE IS THE NUMBER WORTH WATCHING.

    Every bar records how many seconds it took. That series is a
    direct read on participation: `seconds` collapsing from 90 to 8
    means the same thousand decisions are now being made eleven times
    faster, and it says so BEFORE price has travelled far enough for a
    time bar to look unusual. `pace()` compares the bar being built
    against the recent median so the panel can say "this one is
    running hot" while it is still running.

THE CANDLE IS THE TRADED PRICE, NOT THE MID.

    The clock windows in `clockwin` keep both, because the OFI formula
    they carry is defined against the midpoint. Nothing here is: a bar
    that exists because trades happened is drawn from the prices those
    trades happened at.

UNKNOWN-SIDE PRINTS COUNT TOWARD THE BAR AND NOT TOWARD DELTA.

    Databento writes `side='N'` when it cannot attribute the
    aggressor. Those are real trades -- they move price and they fill
    the bar -- so they advance the count. They are not guessed into
    one side of delta; they are held in `unknown` where a reader can
    see how much of the bar's flow was unattributable before
    believing the delta on it.

Nothing in this module places, sizes, routes or times an order. It
counts prints and closes bars.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

# Sizes offered on the panel. ES prints hard enough that anything
# under a few hundred is noise at the chart scale, and anything over a
# few thousand stops being faster than the clock bars it is meant to
# beat. The Fibonacci-ish middle is the conventional futures set.
PER_BAR_CHOICES: tuple[int, ...] = (233, 500, 1000, 2000, 5000, 10000)
DEFAULT_PER_BAR = 1000

# The series the fusion rule reads unless told otherwise. Chris named
# the 5,000-tick bar specifically.
STRATEGY_PER_BAR = 5000

# Floor and ceiling on anything arriving from a query string. A bar of
# one print is a trade list, and a bar of a million never closes.
MIN_PER_BAR = 50
MAX_PER_BAR = 50_000

# How many closed bars each builder keeps. 1,500 at 1,000 prints a bar
# is a million and a half prints of history, which on ES is most of a
# session -- enough to draw the chart and to measure a pace median
# against, and small enough to hold several builders at once.
KEEP = 1_500


@dataclass
class TickBar:
    """One bar: N prints, and what they did.

    `trades` is N for every closed bar WHEN THE BUILDER HAS NO
    PERIOD. Give it one and a bar can also be cut by the clock, in
    which case it closes short and `by_clock` says so -- the
    invariant that used to hold by construction now holds only for
    `period_s == 0`, and anything reading `trades` on a closed bar
    has to look at `by_clock` first.
    """

    start_ts: float
    open: float
    high: float
    low: float
    close: float
    ts_last: float = 0.0
    volume: float = 0.0
    trades: int = 0
    buy_size: float = 0.0
    sell_size: float = 0.0
    unknown: int = 0
    unknown_size: float = 0.0
    closed: bool = False
    # Closed because a time boundary arrived, not because the count
    # was reached. Such a bar is SHORT -- possibly a single print in
    # a quiet minute -- and a direction read off three prints is not
    # the same measurement as one read off five thousand.
    by_clock: bool = False

    @property
    def seconds(self) -> float:
        """Wall time the bar took. The pace signal is built on this."""
        return max(0.0, float(self.ts_last) - float(self.start_ts))

    @property
    def delta(self) -> float:
        """Buy size minus sell size, unknown-side prints excluded."""
        return self.buy_size - self.sell_size

    @property
    def direction(self) -> str:
        """Close against open. Equal is flat, and flat is not a side."""
        if self.close > self.open:
            return "up"
        if self.close < self.open:
            return "down"
        return "flat"

    @property
    def delta_dir(self) -> str:
        d = self.delta
        if d > 0:
            return "up"
        if d < 0:
            return "down"
        return "flat"

    @property
    def agree(self) -> str:
        """The direction price and delta BOTH show, or ''.

        Flat is not agreement on either side. A bar whose price went
        nowhere agrees with nothing, however lopsided its delta.
        """
        d = self.direction
        if d in ("up", "down") and self.delta_dir == d:
            return d
        return ""

    def to_dict(self) -> dict:
        """The chart's row shape, plus what the row is made of.

        `ts`/`o`/`h`/`l`/`c`/`v` are named for the existing chart
        painter so a tick series can be handed to it unchanged.
        """
        return {
            "ts": round(float(self.start_ts), 3),
            "o": self.open, "h": self.high,
            "l": self.low, "c": self.close,
            "v": round(self.volume, 2),
            "ts_last": round(float(self.ts_last), 3),
            "seconds": round(self.seconds, 2),
            "trades": self.trades,
            "buy": round(self.buy_size, 2),
            "sell": round(self.sell_size, 2),
            "delta": round(self.delta, 2),
            "dir": self.direction,
            "delta_dir": self.delta_dir,
            "agree": self.agree,
            "unknown": self.unknown,
            "unknown_size": round(self.unknown_size, 2),
            "closed": bool(self.closed),
            "by_clock": bool(self.by_clock),
            # The chart painter already knows what `live` means and
            # draws that bar differently. Spelling the same state two
            # ways is not redundancy here -- `closed` is this
            # module's word and `live` is the painter's, and renaming
            # either one to suit the other breaks the side that did
            # not change.
            "live": not self.closed,
        }


class TickBars:
    """Builds N-print bars off the raw tape, one print at a time.

    O(1) per print: no buffer of trades is retained and nothing is
    recomputed on close. The builder is deliberately cheap because
    several run at once on the same stream, one per size, so that
    switching the chart's bar size is instant instead of starting an
    empty chart (the same reason `ofibars` keeps one builder per
    target).
    """

    __slots__ = ("per_bar", "keep", "period_s", "done", "_cur",
                 "_prints", "_unknown", "_bars_built", "_clock_cuts",
                 "_last_by_clock")

    def __init__(self, per_bar: int = DEFAULT_PER_BAR,
                 keep: int = KEEP, period_s: float = 0.0) -> None:
        # Only a floor of 1 here, not MIN_PER_BAR. The sane-size
        # clamp belongs at the boundary where an untrusted number
        # arrives -- `parse_interval` -- not in the core, where it
        # would silently turn a caller's 5 into a 50 and make a
        # measurement of short bars impossible to write.
        n = int(per_bar or DEFAULT_PER_BAR)
        self.per_bar = max(1, n)
        self.keep = max(1, int(keep))
        # A TICK BAR THAT ALSO ANSWERS TO THE CLOCK. Chris: "i know
        # that tick bars are not natively time based but i want to
        # make my tick bar time based as well resetting every 5 or
        # 15 minutes".
        #
        # Zero keeps the native behaviour and is the default, so
        # every existing builder is untouched. Set it, and a bar
        # closes on whichever comes first -- the count or the
        # boundary -- so no bar ever straddles one and the series
        # lines up with the 5-minute candle a rule is reading.
        self.period_s = max(0.0, float(period_s or 0.0))
        self.done: deque[TickBar] = deque(maxlen=self.keep)
        self._cur: TickBar | None = None
        self._prints = 0            # prints seen, all time
        self._unknown = 0           # of those, with no side
        self._bars_built = 0
        self._clock_cuts = 0        # of those, closed short by time
        # WHETHER THE MOST RECENT CLOSE WAS A CLOCK CUT, kept here
        # rather than read back off `last_closed`. The intrabar
        # readers are held by test to touching only the FORMING
        # bar -- reaching into a closed one for a flag is how that
        # rule gets eroded one harmless-looking line at a time.
        self._last_by_clock = False

    # -- feeding ---------------------------------------------------------

    def on_trade(self, ts: float, px: float, size: float,
                 side: str) -> TickBar | None:
        """One print. Returns the bar it closed, or None.

        `side` is 'B', 'A' or 'N' exactly as Databento spells it --
        the caller passes the raw record's side rather than a decoded
        trade, because the decoder in `dbn.trade_from` returns None
        for 'N' and would drop those prints before they got here.

        A print with no price or no size is not a trade and does not
        advance the count. Letting a zero through would close bars on
        records that never happened and put a 0.0 in the OHLC.
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
        s = (side or "").strip().upper()[:1]

        b = self._cur
        # THE BOUNDARY IS CHECKED BEFORE THE PRINT IS APPLIED, and
        # the print that crosses it starts the NEW bar rather than
        # finishing the old one. Applying it first would credit a
        # trade to a bar that had already ended by the clock -- the
        # same ordering rule the tape uses for trade-then-book.
        cut = None
        if (b is not None and self.period_s > 0
                and int(t // self.period_s)
                != int(b.start_ts // self.period_s)):
            b.closed = True
            b.by_clock = True
            self.done.append(b)
            self._bars_built += 1
            self._clock_cuts += 1
            self._last_by_clock = True
            cut = b
            b = self._cur = None

        if b is None:
            b = self._cur = TickBar(start_ts=t, open=p, high=p,
                                    low=p, close=p, ts_last=t)
        if p > b.high:
            b.high = p
        if p < b.low:
            b.low = p
        b.close = p
        b.ts_last = t
        b.volume += sz
        b.trades += 1
        if s == "B":
            b.buy_size += sz
        elif s == "A":
            b.sell_size += sz
        else:
            b.unknown += 1
            b.unknown_size += sz
            self._unknown += 1

        if b.trades >= self.per_bar:
            b.closed = True
            self.done.append(b)
            self._bars_built += 1
            self._last_by_clock = False
            self._cur = None
            return b
        # One print can close two bars -- the clock cut above and
        # then a count close, when `per_bar` is 1. Both are in
        # `done`; the return names the LAST one closed, so a caller
        # driving a chart off it never draws a bar out of order.
        return cut

    # -- reading ---------------------------------------------------------

    @property
    def current(self) -> TickBar | None:
        """The bar being built, or None when the last print closed one."""
        return self._cur

    @property
    def last_by_clock(self) -> bool:
        """Was the most recent close a clock cut rather than a count?"""
        return self._last_by_clock

    @property
    def last_closed(self) -> TickBar | None:
        return self.done[-1] if self.done else None

    def rows(self, limit: int = 200, include_live: bool = True) -> list[dict]:
        """Closed bars oldest-first, with the forming bar appended.

        The forming bar is drawn rather than hidden, flagged by
        `closed: false`. Its high, low and close are real prints and
        its open is the true open of the bar -- unlike a time bar
        recovered mid-session, a tick bar has no notion of an open it
        missed, because the bar did not exist before its first print.
        """
        n = max(1, int(limit))
        out = [b.to_dict() for b in list(self.done)[-n:]]
        if include_live and self._cur is not None:
            out.append(self._cur.to_dict())
        return out

    def pace(self, lookback: int = 30) -> dict:
        """How fast bars are closing, and how fast this one is running.

        `median_s` is the median seconds-per-bar over the last
        `lookback` CLOSED bars. `ratio` is the forming bar's elapsed
        time against that median, PROJECTED to a full bar -- a bar 20%
        built in 2 seconds is on pace for 10, and that is the number
        worth comparing, not the 2.

        A ratio under 1 means this bar is closing faster than recent
        ones: more decisions per second. Over 1 means it is dragging.
        Both are stated rather than interpreted -- "fast" is not the
        same as "up", and this module does not pretend otherwise.
        """
        spans = [b.seconds for b in list(self.done)[-max(1, int(lookback)):]
                 if b.seconds > 0]
        med = 0.0
        if spans:
            spans.sort()
            mid = len(spans) // 2
            med = (spans[mid] if len(spans) % 2
                   else (spans[mid - 1] + spans[mid]) / 2.0)

        cur = self._cur
        built = cur.trades if cur is not None else 0
        frac = (built / self.per_bar) if self.per_bar else 0.0
        elapsed = cur.seconds if cur is not None else 0.0
        projected = (elapsed / frac) if frac > 0 else 0.0
        ratio = (projected / med) if (med > 0 and projected > 0) else 0.0

        return {
            "per_bar": self.per_bar,
            "median_s": round(med, 2),
            "bars_measured": len(spans),
            "building": built,
            "fraction": round(frac, 4),
            "elapsed_s": round(elapsed, 2),
            "projected_s": round(projected, 2),
            "ratio": round(ratio, 3),
            # Said in words so the panel does not have to decide what
            # the number means, and so the threshold lives in one
            # place rather than in the page and here.
            "say": _pace_words(med, projected, ratio, len(spans)),
        }

    def summary(self) -> dict:
        unk = (100.0 * self._unknown / self._prints) if self._prints else 0.0
        return {
            "per_bar": self.per_bar,
            "prints": self._prints,
            "bars": self._bars_built,
            "held": len(self.done),
            "unknown_side": self._unknown,
            "unknown_pct": round(unk, 2),
            "pace": self.pace(),
        }


def _pace_words(med: float, projected: float, ratio: float,
                measured: int) -> str:
    """Plain English for the pace, or an honest refusal.

    Under eight closed bars there is no median worth comparing
    against, and saying "running hot" off three samples is the kind
    of confident noise that gets traded on.
    """
    if measured < 8:
        return (f"only {measured} closed bars to compare against -- "
                "not enough to call the pace yet")
    if med <= 0 or projected <= 0:
        return "no pace yet on the bar being built"
    if ratio <= 0.5:
        return (f"this bar is on pace for {projected:.0f}s against a "
                f"{med:.0f}s median -- the tape is roughly twice as busy")
    if ratio <= 0.8:
        return (f"on pace for {projected:.0f}s against {med:.0f}s -- "
                "busier than usual")
    if ratio < 1.25:
        return (f"on pace for {projected:.0f}s against {med:.0f}s -- "
                "normal")
    if ratio < 2.0:
        return (f"on pace for {projected:.0f}s against {med:.0f}s -- "
                "slower than usual")
    return (f"on pace for {projected:.0f}s against a {med:.0f}s median "
            "-- the tape has gone quiet")


def parse_interval(interval: str) -> int | None:
    """`'1000t'` -> 1000. Anything that is not a tick size -> None.

    This is what lets a tick series travel through the EXISTING chart
    route and the existing interval box. `'15m'` returns None and the
    caller falls through to the time-bar path it always used, so no
    time-bar behaviour changes by adding this.
    """
    s = str(interval or "").strip().lower()
    if not s.endswith("t"):
        return None
    head = s[:-1]
    if not head.isdigit():
        return None
    try:
        n = int(head)
    except ValueError:
        return None
    if n <= 0:
        return None
    return max(MIN_PER_BAR, min(n, MAX_PER_BAR))


def label(per_bar: int) -> str:
    """How a size is written on the panel and in a query string."""
    return f"{int(per_bar)}t"
