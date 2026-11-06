"""The price you would actually get, not the price that last printed.

Chris:

    "when i log a buy or sell hit the tp/sl or close the trade it
    definitely dont match the real entry and exit prices so it shows
    a loss when it would have been a win"

THE ENGINE HAD ONE PRICE AND A TRADE HAS FOUR.

    `_live_px` returned `tape.last_px` -- the last price that
    TRADED -- and that single number was used for the entry fill,
    the exit fill, and the stop and target checks.

    A market buy lifts the offer. A market sell hits the bid. The
    last print is whichever one happened to go through, so it is the
    wrong price for at least one side of every round trip, and
    nothing in the log ever said so.

    Four prices, not one:

        open  long   -> you PAY the ASK
        open  short  -> you RECEIVE the BID
        close long   -> you SELL into the BID
        close short  -> you BUY back the ASK

WHY THIS SHOWS LOSSES THAT DID NOT HAPPEN.

    A long is marked and stopped against the last print. On ES the
    bid and the offer are usually one tick apart, so a single trade
    at the bid prints a price a tick below where a buyer could have
    been filled. If the stop sits there, the trade is recorded as
    stopped out at a price nothing would have filled at -- while the
    market, read properly, never offered that exit at all.

    That is the 7478 -> 7474.75 reading: a print, not a fill. And it
    is asymmetric, which is why it costs rather than averages out.
    The stop fires on ANY adverse print; the target needs a
    favourable one. Noise in the mark therefore manufactures stops
    and withholds targets, every time, in that direction.

    So an OPEN POSITION IS MARKED AT ITS EXIT PRICE -- the bid for a
    long, the ask for a short -- because that is the only price it
    could actually leave at. Stops and targets read that mark.

THE BOOK IS NOT ALWAYS TRUSTWORTHY AND THIS SAYS WHEN.

    A crossed book (bid >= ask) is bad data, not a market. A spread
    of forty ticks is a gap or a feed hiccup, not an executable
    quote. A book last touched eleven seconds ago is a memory. In
    every one of those cases this falls back to the last print and
    SAYS it fell back, because a fill priced off a stale book with
    no warning is the same failure wearing a better name.

Nothing in this module places, sizes, routes or times an order. It
answers what a fill would cost; it does not cause one.
"""

from __future__ import annotations

TICK = 0.25
# Wider than this and the quote is not a market: ES trades inside a
# tick nearly all session, and a 10-tick spread is a gap, a halt or a
# bad decode.
MAX_SPREAD_TICKS = 10.0
# A quote older than this is a memory, not an offer.
MAX_AGE_S = 3.0


def usable(bid: float, ask: float, age_s: float | None = None,
           max_spread_ticks: float = MAX_SPREAD_TICKS,
           max_age_s: float = MAX_AGE_S) -> tuple[bool, str]:
    """Is this quote something a fill could be priced from?

    Returns (ok, why_not). The reason is never blank on a refusal,
    because "the fill price looked wrong" has cost more time on this
    engine than any other complaint and a silent fallback is what
    made it hard to find.
    """
    try:
        b = float(bid or 0.0)
        a = float(ask or 0.0)
    except (TypeError, ValueError):
        return False, "no quote"
    if b <= 0 or a <= 0:
        return False, "no quote"
    if b >= a:
        # Crossed or locked. Real for an instant at an open; not a
        # price to fill against.
        return False, "crossed book"
    spread = (a - b) / TICK
    if spread > max_spread_ticks:
        return False, f"{spread:.0f}-tick spread"
    if age_s is not None and age_s > max_age_s:
        return False, f"quote {age_s:.1f}s old"
    return True, ""


def fill_px(side: str, action: str, bid: float, ask: float,
            last: float, age_s: float | None = None) -> tuple[float, str]:
    """What this fill costs, and which price it came from.

    `side` is "up" or "down", `action` is "open" or "close".
    Returns (price, source) where source is "ask", "bid", or
    "last: <reason>" when the quote could not be used.
    """
    ok, why = usable(bid, ask, age_s)
    fallback = (round(float(last or 0.0), 4), f"last: {why or 'no quote'}")
    if not ok:
        return fallback
    if side not in ("up", "down") or action not in ("open", "close"):
        return fallback

    # Opening a long and closing a short both BUY -> the ask.
    # Opening a short and closing a long both SELL -> the bid.
    buying = (side == "up") == (action == "open")
    return (round(float(ask), 4), "ask") if buying \
        else (round(float(bid), 4), "bid")


def mark_px(side: str, bid: float, ask: float, last: float,
            age_s: float | None = None) -> tuple[float, str]:
    """What an OPEN position is worth right now.

    Its exit price, which is the only price it can actually leave
    at, and therefore the only honest thing for a stop or a target
    to read. Marking a long at the last print lets a single trade at
    the bid stop it out a tick below anywhere it could have exited.
    """
    return fill_px(side, "close", bid, ask, last, age_s)


def spread_ticks(bid: float, ask: float) -> float:
    try:
        b, a = float(bid or 0.0), float(ask or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if b <= 0 or a <= 0 or b >= a:
        return 0.0
    return round((a - b) / TICK, 2)


def round_trip_ticks(bid: float, ask: float) -> float:
    """What a round trip costs before the market moves at all.

    Crossing in and crossing out is one spread each way. Worth
    stating in the open, because every backtest in this repo scores
    gross and a rule whose edge is smaller than this is not a rule.
    """
    return spread_ticks(bid, ask) * 2.0


def read(book: object, last: float,
         age_s: float | None = None) -> dict:
    """The quote as a panel can show it, from whatever the tape has."""
    bid = float(getattr(book, "best_bid", 0.0) or 0.0) if book else 0.0
    ask = float(getattr(book, "best_ask", 0.0) or 0.0) if book else 0.0
    ok, why = usable(bid, ask, age_s)
    return {
        "bid": round(bid, 4) if bid else None,
        "ask": round(ask, 4) if ask else None,
        "last": round(float(last or 0.0), 4) or None,
        "spread_ticks": spread_ticks(bid, ask),
        "round_trip_ticks": round_trip_ticks(bid, ask),
        "age_s": age_s,
        "usable": ok,
        "why": why,
        "buy_at": fill_px("up", "open", bid, ask, last, age_s)[0],
        "sell_at": fill_px("up", "close", bid, ask, last, age_s)[0],
    }
