"""44 million records in, about a thousand rows out.

WHY THIS EXISTS

The smoke test read 150,000 records and covered two and a half minutes of
market. Five days is 44 million. Re-parsing that for every experiment would
put seven or eight minutes between asking a question and seeing the answer,
which is the difference between a study you iterate on and a study you run
once and accept.

So the file is walked exactly once and reduced to one row per candle: the
book as it stood at the open, the flow that led into it, the candle that
followed, and the path price took inside it. Roughly 1,400 rows for five
days of five-minute candles -- small enough to hold in memory, re-score in
under a second, and keep beside the repo.

THE LINE THIS MODULE HAS TO NOT CROSS

Each row contains both a decision input and an outcome, which is exactly the
shape that leaks the future if it is built carelessly. The rule enforced
here:

    - `book` is the last book state at or BEFORE the candle opens.
    - `delta` and `trades` cover a window ENDING at the open.
    - `open/high/low/close`, `volume` and `path` are strictly AFTER it.

Nothing that is read to make a decision is computed from a record stamped at
or after the moment the decision is made. `condense` never looks forward to
fill a decision field, and `test_escondense` pins that with a fixture whose
second half is deliberately explosive -- if any of it leaks backwards into
the decision fields, the numbers move and the test fails.

WHY THE PATH IS MINUTE BUCKETS AND NOT A SUMMARY

Reach asks which side price touched FIRST. A candle's high and low cannot
answer that -- they are unordered. The path keeps one [high, low] pair per
minute inside the candle, in order, which is the coarsest thing that still
carries the ordering. A minute that touched both sides is scored as the
adverse one, because a one-minute bucket does not record the sequence of
ticks inside itself and guessing in the tester's favour is how a backtest
flatters itself.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import dbn
from .flow import Book

# One row per this many seconds.
DEFAULT_INTERVAL = 300.0

# How far back the flow read looks, ending at the open.
DEFAULT_LOOKBACK = 300.0

# Buckets inside a candle, for the path. Roughly this many per candle.
#
# Sixty-second buckets were the first draft and they quietly broke the reach
# test. On five-minute ES the median bar runs about five ticks each way, so a
# one-minute bucket spans roughly 2.3 ticks in each direction -- wider than a
# two- or three-tick scalp target. A bucket that contains both the target and
# the stop is scored as the stop, so almost every bar resolved as a loss and
# the two-tick row came back at 34.9% when a coin would be 50%. The market did
# not do that; the bucket size did.
#
# The rule: a bucket has to be materially finer than the smallest target being
# tested, or reach measures the tester instead of the tape.
PATH_BUCKETS_PER_CANDLE = 30
MIN_PATH_BUCKET = 1.0


def path_bucket_for(interval_s: float) -> float:
    """Bucket width for a candle length. Never below one second."""
    return max(MIN_PATH_BUCKET, interval_s / PATH_BUCKETS_PER_CANDLE)

# Levels of book kept per side. Ten is what MBP-10 carries.
KEEP_LEVELS = 10


def grid(ts: float, interval_s: float) -> float:
    """Floor a timestamp onto the interval grid."""
    return ts - (ts % interval_s) if interval_s > 0 else ts


@dataclass
class Slice:
    """One candle: what was knowable at the open, and what then happened."""

    ts: float                                  # candle start

    # -- knowable at the open ------------------------------------------
    bids: list[tuple[float, float]] = field(default_factory=list)
    asks: list[tuple[float, float]] = field(default_factory=list)
    delta: float = 0.0                         # signed size, lookback window
    buy_vol: float = 0.0
    sell_vol: float = 0.0
    trades_in: int = 0
    prior_close: float = 0.0                   # last trade before the open
    prior_open: float = 0.0                    # first trade IN the window

    # -- what followed --------------------------------------------------
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    trades_out: int = 0
    path: list[tuple[float, float]] = field(default_factory=list)

    @property
    def mid(self) -> float:
        if self.bids and self.asks:
            return (self.bids[0][0] + self.asks[0][0]) / 2.0
        return self.prior_close

    @property
    def prior_move_bps(self) -> float:
        """Price change over the window ENDING at the open.

        This is the price column of the three-column read, and it has to be
        computed from here rather than from `open` against `prior_close`.
        `open` is the first trade INSIDE the candle -- an outcome field --
        and using it lets one trade's worth of the answer into the question.
        On a synthetic feed built to be a pure coin, that leak alone scored
        63.7% and reported a real edge with a clear interval. Nothing
        crashed; the study was simply wrong.
        """
        if self.prior_open <= 0 or self.prior_close <= 0:
            return 0.0
        return (self.prior_close - self.prior_open) / self.prior_open * 10_000.0

    @property
    def complete(self) -> bool:
        """Rows missing either half are not scoreable and are dropped.

        A row with a book but no subsequent trades cannot be graded; a row
        with a candle but no book had no read to grade. Keeping either would
        put unscoreable rows in the denominator.
        """
        return bool(self.bids and self.asks and self.close > 0
                    and self.open > 0)

    def imbalance(self, levels: int = 5) -> float:
        """Bid size minus ask size over their sum, top N levels.

        Size, not notional: across ten levels of ES the prices differ by a
        couple of ticks, so weighting by price adds nothing but a chance to
        divide by a zero mid.
        """
        b = sum(sz for _, sz in self.bids[:levels])
        a = sum(sz for _, sz in self.asks[:levels])
        return (b - a) / (b + a) if (b + a) > 0 else 0.0

    def to_row(self) -> dict:
        d = asdict(self)
        d["bids"] = [[round(p, 4), s] for p, s in self.bids]
        d["asks"] = [[round(p, 4), s] for p, s in self.asks]
        d["path"] = [[round(h, 3), round(l, 3)] for h, l in self.path]
        return d

    @classmethod
    def from_row(cls, d: dict) -> "Slice":
        return cls(
            ts=d["ts"],
            bids=[(p, s) for p, s in d.get("bids", [])],
            asks=[(p, s) for p, s in d.get("asks", [])],
            delta=d.get("delta", 0.0),
            buy_vol=d.get("buy_vol", 0.0),
            sell_vol=d.get("sell_vol", 0.0),
            trades_in=d.get("trades_in", 0),
            prior_close=d.get("prior_close", 0.0),
            prior_open=d.get("prior_open", 0.0),
            open=d.get("open", 0.0), high=d.get("high", 0.0),
            low=d.get("low", 0.0), close=d.get("close", 0.0),
            volume=d.get("volume", 0.0), trades_out=d.get("trades_out", 0),
            path=[(h, l) for h, l in d.get("path", [])],
        )


class _Window:
    """Signed flow over a trailing window, pruned as time advances.

    A deque of (ts, signed_size, size) rather than Trade objects: at 2.2
    million trades the object overhead is the difference between comfortable
    and swapping.
    """

    def __init__(self, seconds: float):
        self.seconds = seconds
        self._items: list[tuple[float, float, float, float]] = []
        self._head = 0

    def add(self, ts: float, signed: float, size: float,
            px: float) -> None:
        self._items.append((ts, signed, size, px))

    def prune(self, now: float) -> None:
        cut = now - self.seconds
        while self._head < len(self._items) and self._items[self._head][0] < cut:
            self._head += 1
        # Compact occasionally so the list does not grow without bound.
        if self._head > 50_000:
            del self._items[:self._head]
            self._head = 0

    def read(self) -> tuple[float, float, float, int, float]:
        """(delta, buy volume, sell volume, count, first price) in window."""
        delta = buy = sell = 0.0
        n = 0
        first = 0.0
        for _, signed, size, px in self._items[self._head:]:
            delta += signed
            if signed > 0:
                buy += size
            else:
                sell += size
            if first == 0.0:
                first = px
            n += 1
        return delta, buy, sell, n, first


def condense(records: Iterable[Any], interval_s: float = DEFAULT_INTERVAL,
             lookback_s: float = DEFAULT_LOOKBACK,
             symbol: str = "ES",
             flip: bool = False,
             path_bucket_s: float | None = None) -> Iterator[Slice]:
    """Walk a DBN stream once, yielding one Slice per completed candle.

    Streaming and memory-bounded: the only things held are the current
    candle, the trailing flow window, and the last book. A completed candle
    is yielded and dropped.
    """
    bucket_s = (path_bucket_for(interval_s) if path_bucket_s is None
                else max(MIN_PATH_BUCKET, path_bucket_s))
    window = _Window(lookback_s)
    last_book: Book | None = None
    last_px = 0.0

    cur: Slice | None = None
    cur_end = 0.0
    buckets: list[tuple[float, float] | None] = []

    def finish(s: Slice) -> Slice:
        """Densify the path: a minute with no trades keeps the previous
        minute's extremes rather than vanishing. Dropping it would compress
        the timeline; zeroing it would invent a round trip to the open."""
        out: list[tuple[float, float]] = []
        last = (0.0, 0.0)
        for hl in buckets:
            last = hl if hl is not None else last
            out.append(last)
        s.path = out
        return s

    for rec in records:
        action = dbn._action(rec)

        # A book reset invalidates resting depth. Carrying levels across one
        # would invent liquidity that the exchange has just told us is gone.
        if action == dbn.ACTION_CLEAR:
            last_book = None
            continue

        ts = dbn.seconds(dbn._get(rec, "ts_event"))
        if ts <= 0:
            continue

        trade = dbn.trade_from(rec, flip=flip)

        # -- close out any candle this record has moved past -------------
        while cur is not None and ts >= cur_end:
            done = finish(cur)
            if done.complete:
                yield done
            cur = None
            buckets = []

        # -- open a new candle when one is due ---------------------------
        start = grid(ts, interval_s)
        if cur is None:
            window.prune(start)
            delta, buy, sell, n, first_px = window.read()
            cur = Slice(ts=start, delta=delta, buy_vol=buy, sell_vol=sell,
                        trades_in=n, prior_close=last_px,
                        prior_open=first_px)
            if last_book is not None and not last_book.empty:
                cur.bids = [(l.px, l.sz)
                            for l in last_book.bids[:KEEP_LEVELS]]
                cur.asks = [(l.px, l.sz)
                            for l in last_book.asks[:KEEP_LEVELS]]
            cur_end = start + interval_s
            buckets = []

        # -- accumulate the outcome --------------------------------------
        if trade is not None:
            px, sz = trade.px, trade.sz
            if cur.open == 0.0:
                cur.open = px
                cur.high = cur.low = px
            cur.high = max(cur.high, px)
            cur.low = min(cur.low, px)
            cur.close = px
            cur.volume += sz
            cur.trades_out += 1

            # Path buckets, in bps from the candle's own open, addressed by
            # index rather than by advancing a cursor -- a minute with no
            # trades in it must still occupy its slot, or every later bucket
            # shifts earlier and reach reads an ordering that never happened.
            if cur.open > 0:
                bps = (px - cur.open) / cur.open * 10_000.0
                idx = int((ts - cur.ts) // bucket_s)
                while len(buckets) <= idx:
                    buckets.append(None)
                hl = buckets[idx]
                buckets[idx] = ((bps, bps) if hl is None
                                else (max(hl[0], bps), min(hl[1], bps)))

            window.add(ts, sz if trade.aggressor == "buy" else -sz,
                       sz, px)
            last_px = px

        if dbn._get(rec, "levels"):
            b = dbn.book_from(rec, symbol=symbol)
            if not b.empty:
                last_book = b

    # The candle in progress when the file ends is dropped, always. The
    # close-out loop above yields every candle whose end the tape actually
    # reached, so whatever is left here is by definition partial -- and a
    # short outcome graded against a full-length decision is a silently
    # pessimistic row in the denominator.


# ------------------------------------------------------------- storage

def save(slices: Iterable[Slice], path: str | Path) -> int:
    """JSON Lines, one Slice per line.

    Line-delimited rather than one big array so a run that dies partway
    through still leaves a readable file, and so it can be streamed back
    without holding the parse in memory twice.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w") as fh:
        for s in slices:
            fh.write(json.dumps(s.to_row()) + "\n")
            n += 1
    return n


def load(path: str | Path) -> list[Slice]:
    out: list[Slice] = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Slice.from_row(json.loads(line)))
    return out
