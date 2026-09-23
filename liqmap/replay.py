"""What price action alone predicts, measured over historical bars.

WHY THIS CANNOT MEASURE THE BOOK, AND WHY THAT IS SAID SO LOUDLY

The question worth answering is "when the book and price action agree, what
happens next". Half of that is not available historically. No exchange serves
past order-book depth at this resolution -- not Hyperliquid, not anyone -- so
there is no file, no endpoint and no dataset from which microprice tilt,
replenishment or queue depletion can be reconstructed for a bar that closed
last Tuesday.

The tempting move is to build a "book proxy" out of candles. It is always
possible: volume, wick ratios, close-versus-VWAP all correlate with order
flow and would produce a plausible-looking book column. It would also be
worthless and actively misleading, because every one of those is derived from
PRICE. Agreement between a price-derived proxy and price action is price
action agreeing with itself, and the resulting table would show a beautiful
edge that evaporates the moment it meets a real book.

So this module measures exactly one thing: what the price-action reading, on
its own, is worth. That is the BASE RATE. It is the number the full rule has
to beat, and knowing it is what makes the live table interpretable -- without
it, "confirmed candles go +6bps" could just as easily be what any candle
does.

The book half is measured live, as it happens, in `agreement_states`.

HOW THE READ IS TAKEN

For each bar, price action is computed from the sub-bars INSIDE it up to a
chosen fraction of the way through, and scored against the rest of that same
bar. Nothing after the read time is visible to the read. That is the same
anti-lookahead discipline as `backtest.py`, and it matters more here, because
a price-action signal that peeks at its own outcome will report a hit rate
near 100% and look like a discovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .confirm import read_candle, typical_bar_bps
from .structure import Candle

# Where in the bar the read is taken. A third of the way through is early
# enough to be actionable and late enough for the bar to have said something.
DEFAULT_AT = 0.33

# Fewer than this and the table is decoration.
MIN_SAMPLES = 40


@dataclass
class ReplayRow:
    direction: str
    strength: float
    called_bps: float          # move in the called direction, from read to close
    mfe_bps: float             # best excursion in the called direction
    mae_bps: float             # worst against it
    slope_bps: float
    thrust_bps: float
    position: float


def _aggregate(subs: Sequence[Candle]) -> tuple[float, float, float, float]:
    """open, high, low, close of a group of sub-bars."""
    return (subs[0].open, max(s.high for s in subs),
            min(s.low for s in subs), subs[-1].close)


def replay_price_action(bars: Sequence[Candle], sub: Sequence[Candle],
                        interval_s: float, sub_s: float,
                        at: float = DEFAULT_AT) -> list[ReplayRow]:
    """Read price action partway through each bar, score it on the rest.

    `bars` are the trading timeframe; `sub` are finer bars covering the same
    span, used both to build the partial bar and to measure what happened
    afterwards. Without sub-bars there is no way to stand inside a candle,
    and reading a bar from its own finished shape is pure lookahead.
    """
    if not bars or not sub or interval_s <= 0 or sub_s <= 0:
        return []

    per_bar = max(int(round(interval_s / sub_s)), 2)
    cut = max(1, int(per_bar * at))
    if cut >= per_bar:
        return []

    by_start: dict[float, list[Candle]] = {}
    for s in sub:
        slot = s.ts - (s.ts % interval_s)
        by_start.setdefault(slot, []).append(s)

    bar_bps = typical_bar_bps(bars)
    if bar_bps <= 0:
        return []

    out: list[ReplayRow] = []
    history: list[Candle] = []

    for bar in bars:
        slot = bar.ts - (bar.ts % interval_s)
        subs = sorted(by_start.get(slot, []), key=lambda c: c.ts)
        history.append(bar)
        if len(subs) < per_bar * 0.6:
            continue                      # too gappy to stand inside

        seen, rest = subs[:cut], subs[cut:]
        if not rest:
            continue

        o, h, l, c = _aggregate(seen)

        # Slope over the last couple of sub-bars, which is the closest thing
        # bars offer to the live tape's short-horizon reading.
        window = seen[-2:] if len(seen) >= 2 else seen
        first = window[0].open
        slope = ((c - first) / first * 10_000.0) if first > 0 else 0.0

        prior = history[-6:-1]
        action = read_candle(
            open_px=o, high_px=h, low_px=l, last_px=c,
            slope_bps=slope, bar_range_bps=bar_bps,
            recent_highs=[p.high for p in prior],
            recent_lows=[p.low for p in prior],
            window_s=sub_s * len(window))

        if action is None or action.direction == "flat":
            continue

        s = 1 if action.direction == "up" else -1
        entry = c
        if entry <= 0:
            continue

        close = rest[-1].close
        hi = max(r.high for r in rest)
        lo = min(r.low for r in rest)

        out.append(ReplayRow(
            direction=action.direction,
            strength=action.strength,
            called_bps=(close - entry) / entry * 10_000.0 * s,
            mfe_bps=((hi - entry) if s > 0 else (entry - lo)) / entry * 10_000.0,
            mae_bps=((lo - entry) if s > 0 else (entry - hi)) / entry * 10_000.0,
            slope_bps=action.slope_bps,
            thrust_bps=action.thrust_bps,
            position=action.position,
        ))

    return out


def table(rows: Sequence[ReplayRow]) -> dict[str, Any]:
    """The price-action base rate, banded by how strong the reading was."""

    def summarise(subset: Sequence[ReplayRow], label: str,
                  detail: str) -> dict[str, Any]:
        if not subset:
            return {"state": label, "detail": detail, "n": 0, "share": 0.0,
                    "avg_called_bps": None, "median_called_bps": None,
                    "hit_rate": None, "avg_mfe_bps": None, "avg_mae_bps": None}
        moves = [r.called_bps for r in subset]
        ordered = sorted(moves)
        n = len(ordered)
        return {
            "state": label, "detail": detail, "n": n,
            "share": n / len(rows) if rows else 0.0,
            "avg_called_bps": sum(moves) / n,
            "median_called_bps": (ordered[n // 2] if n % 2
                                  else (ordered[n // 2 - 1] + ordered[n // 2]) / 2),
            "hit_rate": sum(1 for m in moves if m > 0) / n,
            "avg_mfe_bps": sum(r.mfe_bps for r in subset) / n,
            "avg_mae_bps": sum(r.mae_bps for r in subset) / n,
        }

    strong = [r for r in rows if r.strength >= 0.6]
    weak = [r for r in rows if r.strength < 0.6]

    out = [
        summarise(rows, "PRICE ACTION (all)", "any directional reading"),
        summarise(strong, "PRICE ACTION strong", "strength 60%+"),
        summarise(weak, "PRICE ACTION weak", "under 60%"),
        summarise([r for r in rows if r.direction == "up"],
                  "PRICE ACTION up", "rising reads only"),
        summarise([r for r in rows if r.direction == "down"],
                  "PRICE ACTION down", "falling reads only"),
    ]

    enough = len(rows) >= MIN_SAMPLES
    all_row = out[0]
    if not enough:
        verdict = (f"Only {len(rows)} readings — too few to mean anything. "
                   f"Needs {MIN_SAMPLES}+.")
    else:
        verdict = (
            f"Price action alone, read a third of the way into the bar, "
            f"closed its own way {all_row['hit_rate']:.0%} of the time for "
            f"{all_row['avg_called_bps']:+.1f}bps on average "
            f"({all_row['n']} bars). That is the BASE RATE. The live table "
            f"shows what requiring the order book to agree adds to it — if "
            f"confirmed candles do not beat this, the book is contributing "
            f"nothing.")

    return {
        "n": len(rows), "table": out, "verdict": verdict,
        "measures": "price action only",
        "book_available": False,
        "caveat": (
            "The order book half CANNOT be measured here. No exchange serves "
            "historical L2 depth at this resolution, so microprice tilt, "
            "replenishment and depletion have no historical values to "
            "replay. Building a book proxy out of candles would make this "
            "table look better and mean less — it would be price action "
            "agreeing with itself. The book half is measured live, as it "
            "happens, and accumulates in the agreement table."),
    }
