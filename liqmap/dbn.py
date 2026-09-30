"""Databento DBN -> the engine's own vocabulary.

NOT YET EXERCISED AGAINST A REAL FILE. Written to Databento's published
schema documentation (field names, price scaling, enum values and the
trade-side semantics all verified against their docs). Expect to fix
something on the first real file, and run `audit()` before believing a
single number that comes out of it.

That warning is not boilerplate. The one thing in this module that cannot be
checked against documentation is whether the aggressor convention survives
the trip from CME's matching engine through Databento's encoding into these
records. Databento documents `side` on a trade as "the side that initiates
the event" -- the aggressor. If that is right, a `side == 'B'` trade is a
buy. If it is subtly wrong, every delta in the engine is sign-flipped, every
absorption read is inverted, and nothing crashes. This exact failure mode
has already cost this project once on the crypto feed, which is why
`hl.check_side_convention` exists there and why `SideAudit` exists here.

HOW THE AUDIT WORKS

A buyer who crosses the spread pays the ask. So if the convention is right,
trades marked 'B' should print at or very near the prevailing ask, and 'A'
trades at or near the bid. Compare each trade to the book immediately before
it and the tape tells you its own convention -- no documentation required.
Run it on the first file. If it disagrees with the docs, believe the file.

WHY IT PRODUCES flow.Book AND flow.Trade

Those are what the existing engine already eats. FlowTape, VolumeProfile,
ImpactBaseline and BandTracker are instrument-agnostic arithmetic that has
been under test for months; the only thing that was ever crypto-specific was
the shape of the JSON coming off the wire. So this module's whole job is to
be a second translator into the same vocabulary, and the engine downstream
never learns which venue it is reading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from .flow import Book, Level, Side, Trade

# Databento fixed-precision: every integer unit is 1e-9 of a price.
PRICE_SCALE = 1e-9

# INT64_MAX marks a level that is not populated. Scaled naively it becomes
# 9.2 billion, which as a price would blow up every depth and imbalance
# calculation downstream rather than simply being absent.
UNDEF_PRICE = 2**63 - 1

NANOS = 1e-9

# Enum values, per Databento's standards page.
SIDE_BID = "B"
SIDE_ASK = "A"
SIDE_NONE = "N"

ACTION_TRADE = "T"
ACTION_FILL = "F"
ACTION_CLEAR = "R"


def price(raw: Any) -> float:
    """Fixed-precision integer to a float price. Undefined becomes 0.0."""
    if raw is None:
        return 0.0
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return 0.0
    if n == UNDEF_PRICE or n <= 0:
        return 0.0
    return n * PRICE_SCALE


def seconds(ts_ns: Any) -> float:
    """Nanoseconds since epoch to float seconds."""
    try:
        return int(ts_ns) * NANOS
    except (TypeError, ValueError):
        return 0.0


def _get(rec: Any, name: str, default: Any = None) -> Any:
    """Read a field off a DBN record or a plain dict.

    The Python client hands back record objects; tests and any CSV-derived
    row hand back dicts. Supporting both means the parser can be exercised
    without a multi-gigabyte file present.
    """
    if isinstance(rec, dict):
        return rec.get(name, default)
    return getattr(rec, name, default)


def _side(rec: Any) -> str:
    """The side character, however the client spells it.

    Databento's Python objects may expose this as an enum whose `str` is
    something like 'Side.BID' rather than a bare character, so this takes
    the first character of the final component and is tolerant about it.
    """
    raw = _get(rec, "side", SIDE_NONE)
    s = getattr(raw, "value", raw)
    text = str(s).strip().upper()
    if not text:
        return SIDE_NONE
    if "." in text:                       # 'SIDE.BID' -> 'BID'
        text = text.rsplit(".", 1)[1]
    if text.startswith("B"):
        return SIDE_BID
    if text.startswith("A") or text.startswith("S"):
        return SIDE_ASK
    return SIDE_NONE


def _action(rec: Any) -> str:
    raw = _get(rec, "action", "")
    s = getattr(raw, "value", raw)
    text = str(s).strip().upper()
    if "." in text:
        text = text.rsplit(".", 1)[1]
    return text[:1] if text else ""


# ------------------------------------------------------------------ book

def book_from(rec: Any, symbol: str = "ES", depth: int = 10) -> Book:
    """One MBP-10 record to a Book.

    Levels with an undefined price or zero size are dropped rather than
    carried as zeros -- Book's own normalisation removes zero-size levels,
    but a zero *price* would survive and land at the top of the ask side
    after sorting, making the spread nonsense.
    """
    bids: list[Level] = []
    asks: list[Level] = []

    levels = _get(rec, "levels") or []
    for i, lv in enumerate(levels):
        if i >= depth:
            break
        bpx, bsz = price(_get(lv, "bid_px")), _get(lv, "bid_sz", 0) or 0
        apx, asz = price(_get(lv, "ask_px")), _get(lv, "ask_sz", 0) or 0
        if bpx > 0 and bsz > 0:
            bids.append(Level(px=bpx, sz=float(bsz),
                              n=int(_get(lv, "bid_ct", 0) or 0)))
        if apx > 0 and asz > 0:
            asks.append(Level(px=apx, sz=float(asz),
                              n=int(_get(lv, "ask_ct", 0) or 0)))

    return Book(coin=symbol, ts=seconds(_get(rec, "ts_event")),
                bids=bids, asks=asks)


def trade_from(rec: Any, flip: bool = False) -> Trade | None:
    """A trade record to a Trade, or None if this record is not one.

    `flip` inverts the aggressor mapping. It exists so that if `audit()`
    finds the convention reversed on a real file, the fix is a flag rather
    than an edit to this function -- an edit that someone would later
    "correct" back, silently re-inverting every delta in the system.
    """
    if _action(rec) not in (ACTION_TRADE, ACTION_FILL):
        return None

    px = price(_get(rec, "price"))
    sz = float(_get(rec, "size", 0) or 0)
    if px <= 0 or sz <= 0:
        return None

    s = _side(rec)
    if s == SIDE_NONE:
        return None

    buy = (s == SIDE_BID)
    if flip:
        buy = not buy

    aggressor: Side = "buy" if buy else "sell"
    return Trade(px=px, sz=sz, aggressor=aggressor,
                 ts=seconds(_get(rec, "ts_event")))


# ------------------------------------------------------------- the audit

@dataclass
class SideReport:
    """What the tape says about its own aggressor convention."""

    samples: int
    at_ask: int          # 'B'-marked trades printing at/above the mid
    at_bid: int          # 'A'-marked trades printing at/below the mid
    agree: float         # fraction consistent with the documented meaning
    verdict: str

    @property
    def ok(self) -> bool:
        return self.verdict == "documented"

    @property
    def flip(self) -> bool:
        return self.verdict == "inverted"


class SideAudit:
    """Re-derives the aggressor convention from the data.

    Feed it the book as it stood immediately before each trade. A crossing
    buyer lifts the offer, so a correctly-labelled 'B' trade prints at or
    above the mid. Ties (a trade exactly at the mid) are discarded rather
    than split, because in a one-tick-wide market they are most of the tape
    and they carry no information about the convention either way.
    """

    def __init__(self, keep: int = 2000):
        self.keep = keep
        self.n = 0
        self.consistent = 0
        self.at_ask = 0
        self.at_bid = 0

    def observe(self, rec: Any, book: Book | None) -> None:
        if book is None or book.empty:
            return
        if _action(rec) not in (ACTION_TRADE, ACTION_FILL):
            return
        px = price(_get(rec, "price"))
        if px <= 0 or self.n >= self.keep:
            return
        mid = book.mid
        if mid <= 0 or px == mid:
            return

        s = _side(rec)
        if s == SIDE_NONE:
            return

        self.n += 1
        above = px > mid
        if s == SIDE_BID:
            self.at_ask += 1 if above else 0
            self.consistent += 1 if above else 0
        else:
            self.at_bid += 1 if not above else 0
            self.consistent += 1 if not above else 0

    def report(self) -> SideReport:
        if self.n < 40:
            return SideReport(self.n, self.at_ask, self.at_bid, 0.0,
                              "not enough samples")
        agree = self.consistent / self.n
        if agree >= 0.80:
            verdict = "documented"
        elif agree <= 0.20:
            verdict = "inverted"
        else:
            verdict = "unclear"
        return SideReport(self.n, self.at_ask, self.at_bid,
                          round(agree, 4), verdict)


# ---------------------------------------------------------------- reading

def stream(records: Iterable[Any], symbol: str = "ES",
           flip: bool = False) -> Iterator[tuple[str, Any]]:
    """Records in, ("book", Book) and ("trade", Trade) out, in order.

    A trade record in MBP-10 carries the book *after* the trade, so the book
    is emitted after the trade it accompanies. Emitting it first would let a
    reader see the consequence of a fill at the moment it decides about that
    fill -- a one-record look-ahead leak, and the kind that never shows up
    as a crash.
    """
    for rec in records:
        if _action(rec) == ACTION_CLEAR:
            yield "clear", seconds(_get(rec, "ts_event"))
            continue
        t = trade_from(rec, flip=flip)
        if t is not None:
            yield "trade", t
        if _get(rec, "levels"):
            yield "book", book_from(rec, symbol=symbol)


def audit(records: Iterable[Any], symbol: str = "ES",
          limit: int = 200_000) -> SideReport:
    """Run the convention check over the front of a file.

    Deliberately reads the book *before* each trade, which means holding the
    previous record's book -- the opposite of what `stream` emits, and the
    reason this is a separate pass rather than a flag on that one.
    """
    a = SideAudit()
    prev: Book | None = None
    for i, rec in enumerate(records):
        if i >= limit:
            break
        a.observe(rec, prev)
        if _get(rec, "levels"):
            prev = book_from(rec, symbol=symbol)
    return a.report()


def open_file(path: str):
    """Records from a .dbn or .dbn.zst file.

    Imported lazily so the rest of the module -- and its tests -- work
    without the databento client installed.
    """
    import databento as db

    store = db.DBNStore.from_file(path)
    return store
