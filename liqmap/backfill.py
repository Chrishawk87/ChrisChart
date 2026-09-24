"""Filling the agreement table from real history.

THE POINT

The live agreement table answers the right questions and fills at one row
per candle per market. At fifteen minutes a bar that is ninety-six rows a
day, so "when the book and price agree, what happens next" is a question
with a three-week wait attached to it.

The archive removes the wait. Hyperliquid publishes real book snapshots
twice a second, going back years, so the same reader that runs live can be
run over last month and produce the same table this afternoon.

THE TWO SIDES MUST STAY INDEPENDENT

This is the part that decides whether the result means anything.

    book side    archived L2 snapshots, fed through the SAME `BookReader`
                 the live feed uses. Not a re-implementation -- the actual
                 class, so a tuning change applies to history and to live
                 at the same time and cannot drift between them.

    price side   one-minute candles from the exchange, which are built from
                 FILLS. Not from the snapshots' own mid.

That second point is the whole game. The archive's mid is computed from the
book, so using it for the price-action side would be the book agreeing with
itself, the confirmation rate would look superb and the number would be
worth nothing. `confirm.py` has a test that fails if it ever imports a
book; this module is where that discipline would actually get broken, so
it is stated here too.

NO LOOKAHEAD

Each bar is read at a fixed fraction of the way through it, using only book
snapshots and sub-bars from before that moment, and scored on what happens
afterwards. A read scored against the bar it was taken from reports a
magnificent hit rate and is measuring nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

from .archive import Archive, TransferCapReached
from .bookread import BookReader
from .confirm import confirm, read_candle, typical_bar_bps
from .structure import Candle

# Where inside the bar the read is taken. Matches `replay.py` so the
# historical book table and the historical price-action base rate are
# measured at the same moment and can be compared directly.
DEFAULT_AT = 0.33

# Book snapshots feeding the reader before a call is taken seriously. The
# live path uses the same floor.
MIN_BOOK_SAMPLES = 5


@dataclass
class Progress:
    """What a running backfill has done so far."""

    coin: str = ""
    interval: str = ""
    hours_done: int = 0
    hours_total: int = 0
    bars_seen: int = 0
    states: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    transfer: str = ""
    error: str = ""
    done: bool = False

    def skip(self, why: str) -> None:
        self.skipped[why] = self.skipped.get(why, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "coin": self.coin, "interval": self.interval,
            "hours_done": self.hours_done, "hours_total": self.hours_total,
            "bars_seen": self.bars_seen, "states": self.states,
            "skipped": self.skipped, "transfer": self.transfer,
            "error": self.error, "done": self.done,
            "pct": (round(100.0 * self.hours_done / self.hours_total)
                    if self.hours_total else 0),
        }


def _bar_start(ts: float, interval_s: float) -> float:
    return ts - (ts % interval_s)


def backfill(archive: Archive, history, coin: str, interval: str,
             interval_s: float, start: datetime, end: datetime,
             minute_bars: Sequence[Candle],
             at: float = DEFAULT_AT,
             progress: Progress | None = None,
             should_stop: Callable[[], bool] | None = None) -> Progress:
    """Replay archived books against fills-derived candles, one row per bar.

    `minute_bars` are one-minute candles covering the window, fetched from
    the exchange by the caller. They serve two purposes and the separation
    matters: the ones BEFORE the read time build the price-action reading,
    and the ones after it settle the outcome.
    """
    p = progress or Progress()
    p.coin, p.interval = coin, interval

    # Group the sub-bars by which trading bar they belong to. Doing this
    # once keeps the inner loop from re-scanning thousands of candles.
    by_bar: dict[float, list[Candle]] = {}
    for c in minute_bars:
        by_bar.setdefault(_bar_start(c.ts, interval_s), []).append(c)
    for bars in by_bar.values():
        bars.sort(key=lambda c: c.ts)

    if not by_bar:
        p.error = ("no one-minute candles for this window — the price side "
                   "cannot be built, and taking it from the book's own mid "
                   "would make the comparison meaningless")
        p.done = True
        return p

    # Trading-timeframe bars, assembled from the sub-bars so the two always
    # agree about where a bar starts.
    frame: list[Candle] = []
    for slot in sorted(by_bar):
        g = by_bar[slot]
        frame.append(Candle(ts=slot, open=g[0].open,
                            high=max(x.high for x in g),
                            low=min(x.low for x in g), close=g[-1].close,
                            volume=sum(x.volume for x in g)))
    bar_bps = typical_bar_bps(frame)
    if bar_bps <= 0:
        p.error = "could not measure a typical bar range for this market"
        p.done = True
        return p

    per_bar = max(int(round(interval_s / 60.0)), 2)
    cut = max(1, int(per_bar * at))

    reader = BookReader()
    slots = sorted(by_bar)
    p.hours_total = max(1, int((end - start).total_seconds() // 3600))

    # The read moment for each bar: `at` of the way through it.
    read_at = {slot: slot + interval_s * at for slot in slots}
    pending = {slot: True for slot in slots}
    last_hour = None

    try:
        for book in archive.books(coin, start, end):
            if should_stop is not None and should_stop():
                p.error = "stopped"
                break

            hour = int(book.ts // 3600)
            if hour != last_hour:
                last_hour = hour
                p.hours_done += 1

            # Feed EVERY snapshot, including ones already past a read
            # moment. The reader's replenishment and depletion counters are
            # cumulative and decayed; skipping updates would leave them
            # measuring a different thing than they do live.
            reader.add(book, now=book.ts)

            slot = _bar_start(book.ts, interval_s)
            if slot not in pending or book.ts < read_at[slot]:
                continue

            # First snapshot at or past this bar's read moment.
            pending.pop(slot, None)
            p.bars_seen += 1

            read = reader.read(aggression=0.0, now=book.ts)
            if read is None or read.samples < MIN_BOOK_SAMPLES:
                p.skip("book still warming")
                continue

            subs = by_bar.get(slot) or []
            if len(subs) < per_bar * 0.6:
                p.skip("too few sub-bars in the hour")
                continue

            seen, rest = subs[:cut], subs[cut:]
            if not seen or not rest:
                p.skip("nothing left of the bar to settle against")
                continue

            # --- the price side, from FILLS only -------------------------
            o = seen[0].open
            hi = max(s.high for s in seen)
            lo = min(s.low for s in seen)
            last = seen[-1].close
            window = seen[-2:] if len(seen) >= 2 else seen
            first = window[0].open
            slope = ((last - first) / first * 10_000.0) if first > 0 else 0.0

            idx = slots.index(slot) if slot in slots else 0
            prior = [frame[i] for i in range(max(0, idx - 5), idx)]

            action = read_candle(
                open_px=o, high_px=hi, low_px=lo, last_px=last,
                slope_bps=slope, bar_range_bps=bar_bps,
                recent_highs=[c.high for c in prior],
                recent_lows=[c.low for c in prior],
                window_s=60.0 * len(window))

            agreement = confirm(read.direction, read.conviction, action)

            # --- record it, then settle it immediately -------------------
            sid = history.record_state(
                coin=coin, interval=interval, candle_ts=slot,
                candle_end=slot + interval_s,
                book_dir=agreement.book, book_strength=agreement.book_strength,
                price_dir=agreement.candle,
                price_strength=agreement.candle_strength,
                verdict=agreement.verdict, price=last,
                elapsed_frac=at, made_at=book.ts,
                features={
                    "tilt": round(read.tilt, 4),
                    "imbalance": round(read.imbalance, 4),
                    "replenish": round(read.replenish, 4),
                    "depletion": round(read.depletion, 4),
                    "spread_bps": round(read.spread_bps, 4),
                    "book_score": round(read.score, 4),
                    "book_samples": read.samples,
                    "slope_bps": round(action.slope_bps, 4) if action else None,
                    "thrust_bps": round(action.thrust_bps, 4) if action else None,
                    "position": round(action.position, 4) if action else None,
                    "price_score": round(action.score, 4) if action else None,
                    "source": "archive",
                })
            if sid is None:
                p.skip("already recorded")
                continue

            history.resolve_state(
                sid, close_px=rest[-1].close,
                high_px=max(r.high for r in rest),
                low_px=min(r.low for r in rest))
            p.states += 1

    except TransferCapReached as exc:
        p.error = str(exc)
    except Exception as exc:                     # noqa: BLE001
        p.error = f"{type(exc).__name__}: {exc}"

    p.transfer = archive.transfer.describe()
    if archive.transfer.errors:
        p.skipped["s3 errors"] = len(archive.transfer.errors)
    p.done = True
    return p
