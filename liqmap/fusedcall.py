"""A reader for the fused panel: everything on it, weighed, as one call.

Chris: "i want you to build in a agent that can read the fused bar
section like a human only whats there it should be free thinking
reading the outer OFI, inner Ticks turning and anything else i have
on ther and make a calculated decision while reading actual price
plus what the fused bar section is saying at the same time ... what
this does is make decisions not out of thin air but what real life
price and the fused bar section is saying at the same time".

IT WEIGHS. THE RULE GATES. That is the whole difference and it is
the reason this exists beside `fusedread` rather than inside it.

    The rule is all-or-nothing by construction: book and price must
    agree, the agreement must have stood five seconds, the bar must
    have settled, every switched-on filter must agree. Any one of
    those missing and the answer is no, with no notion of how
    nearly it was yes.

    This reads the same rows the way a person reads them. Four
    things pointing up with strong bodies and one blank row is not
    the same as four things pointing up and one pointing down, and
    the rule cannot tell those apart -- both are "no".

WHY THAT MATTERS HERE, SPECIFICALLY. Chris: "the move has already
happened on iron beam by the time it shows up on chris chart so the
entries are always well behind iron beam". About fifteen seconds of
that is the rule's own: a settle gate that wants a tenth of the bar
(~9s at his print rate) and a five-second hold on the agreement.
This reader waits for neither. It is the answer to "what would the
rule say if it were not waiting", available while it is still
waiting.

IT DECIDES NOTHING. It does not open, close, size, route or time
anything, and no function here is named as though it did. The
`decides: False` on its payload is asserted by a test, as is the
fact that `fusedread.entry_side` and `fusedread.exit_reason` cannot
see any of it. Chris has had a panel claim a rule nobody was
running once already and it cost him a day; this one says what it
is on every payload it produces.

IT NAMES WHAT VOTED AGAINST IT. A call with no dissent listed is a
call that cannot be checked, and the complaint this engine keeps
getting rebuilt after is always some version of "it did a thing and
would not say why".
"""

from __future__ import annotations

SIDES = ("up", "down")
TICK = 0.25
EPS = 1e-9

# WHAT EACH ROW IS WORTH.
#
# The two legs of the fused bar carry the most because the bar IS
# those two legs -- everything else on the panel is drawn beside
# them. The timeframes are filters in the rule and weigh like it.
# Turning and live price are the fast rows: they say something
# before the bar has settled, which is the entire reason this
# reader is worth having.
#
# THESE ARE NOT TUNED. They are a starting shape, written to be
# argued with after watching, and the panel prints every row's
# contribution so the argument can be had against real bars rather
# than against this comment.
WEIGHTS = {
    "book": 3.0,       # the outer candle, signed OFI
    "price": 3.0,      # the inner candle, close against open
    "turning": 2.0,    # which way price is going off its extreme
    "tf": 2.0,         # the first live filter
    "tf2": 2.0,        # the second
    "live": 1.0,       # the traded price against this bar's open
}

# HOW SURE IT HAS TO BE BEFORE IT NAMES A SIDE. A score runs -1 to
# +1; below this in either direction it says nothing rather than
# naming a side it half believes. Zero would make it call every
# bar, which is the same as calling none of them.
CALL_AT = 0.45

# WHERE A ROW'S STRENGTH STOPS ADDING. Travel is in ticks and a
# move that has carried twenty is not twice as convincing as one
# that carried ten -- it is the same answer, louder. Past this the
# multiplier is full and stays full.
FULL_TRAVEL = 12.0


def _side(v) -> str:
    s = str(v or "").strip().lower()
    return s if s in SIDES else ""


def _dict(v) -> dict:
    """Whatever arrived, as a dict.

    `dict(v or {})` RAISES ON A STRING. `dict("abc")` is a
    ValueError about update sequences, not a quiet empty dict, and
    this reader is handed a payload built five layers away -- one
    row arriving as a string took the whole panel down inside a
    route whose exceptions are swallowed into `journal.error`.
    """
    return v if isinstance(v, dict) else {}


def _num(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _strength(body: float = 0.0, travel: float = 0.0) -> float:
    """How much of a row's weight its own reading has earned, 0 to 1.

    A DIRECTION WITH NOTHING BEHIND IT IS STILL A DIRECTION, so the
    floor is not zero -- a row that points up counts for something
    even on a bar with no body yet. Chris: "the direction shows but
    its strength dont is the move strong or just a micro pull back".
    The point is that they differ, not that the weak one is worth
    nothing.

    BODY AND TRAVEL ANSWER DIFFERENT QUESTIONS and the better of
    the two is taken rather than their average: a bar with almost
    no body that has carried fifteen ticks is a strong move being
    read mid-pullback, and averaging would call it weak.
    """
    b = max(0.0, min(1.0, _num(body)))
    t = max(0.0, min(1.0, _num(travel) / FULL_TRAVEL))
    return 0.4 + 0.6 * max(b, t)


def live_dir(price: float, open_px: float,
             tick: float = TICK, band_ticks: float = 1.0) -> str:
    """Which way the traded price sits against this bar's open.

    THE "ACTUAL PRICE" LEG. Chris asked for a decision made from
    "real life price and the fused bar section ... at the same
    time", and every other row on the panel is a reading OF the
    tape rather than the tape itself. This one is the number on the
    ticker against the number the bar opened at.

    A BAND, BECAUSE A SINGLE TICK IS NOT A DIRECTION. Price sits on
    the open constantly; reading that as a side would make this row
    flicker on every print and lend its weight to whichever way the
    last one went.
    """
    p, o = _num(price), _num(open_px)
    t = _num(tick, TICK) or TICK
    band = max(0.0, _num(band_ticks)) * t
    if p <= 0.0 or o <= 0.0:
        return ""
    if p - o > band:
        return "up"
    if o - p > band:
        return "down"
    return ""


def read(fused: dict | None = None, price: float = 0.0,
         call_at: float = CALL_AT) -> dict:
    """Everything on the fused panel, weighed, as one call.

    `fused` is the panel's own payload -- the same dict the screen
    draws from, not a second reading of the same bar. A reader that
    fetched its own copy would eventually disagree with the panel
    beside it, and this engine has done that with the 5-minute cell
    and with the OFI bar already.

    Returns the side, the score that produced it, every row's
    contribution, and a sentence. Nothing here is a gate.
    """
    f = _dict(fused)
    tf = _dict(f.get("tf"))
    tf2 = _dict(f.get("tf2"))
    turning = _dict(f.get("turning"))

    rows: list[dict] = []

    def vote(key, direction, strength, note):
        d = _side(direction)
        w = float(WEIGHTS.get(key, 0.0))
        s = max(0.0, min(1.0, float(strength)))
        rows.append({
            "row": key,
            "dir": d,
            # A ROW WITH NO DIRECTION IS NOT A ROW VOTING ZERO. It
            # is a row not voting, and it leaves the denominator as
            # well as the numerator -- otherwise a blank filter
            # would halve every score it touched while telling you
            # nothing.
            "weight": w if d else 0.0,
            "strength": s if d else 0.0,
            "points": (w * s) * (1.0 if d == "up" else -1.0) if d else 0.0,
            "note": note,
        })

    # THE TWO LEGS OF THE BAR. The book has no body or travel of its
    # own on this panel -- it is an accumulator, not a candle -- so
    # it votes at full strength when it points anywhere at all.
    vote("book", f.get("ofi_dir"), 1.0, "the outer candle, the book")
    vote("price", f.get("tick_dir"),
         _strength(body=_num(f.get("body_pct"))),
         "the inner candle, price")

    # THE FAST ROWS. Both say something before the bar has settled,
    # which is the whole reason this reader exists.
    vote("turning", turning.get("dir"),
         _strength(travel=_num(turning.get("travel"))),
         "which way price is going off its extreme")
    vote("live", live_dir(price, f.get("open")),
         1.0, "the traded price against this bar's open")

    # THE FILTERS, EACH ONLY IF IT IS SWITCHED ON. One that is off
    # is not a row with nothing to say; it is not a row.
    for key, row in (("tf", tf), ("tf2", tf2)):
        if not row.get("on"):
            rows.append({"row": key, "dir": "", "weight": 0.0,
                         "strength": 0.0, "points": 0.0,
                         "note": "switched off"})
            continue
        vote(key, row.get("dir"),
             _strength(body=_num(row.get("body")),
                       travel=_num(row.get("travel"))),
             f"the {row.get('name') or key} filter")

    live = [r for r in rows if r["weight"] > 0.0]
    total = sum(r["weight"] for r in live)
    score = (sum(r["points"] for r in live) / total) if total > EPS else 0.0

    cut = max(0.0, min(1.0, _num(call_at, CALL_AT)))
    side = ("up" if score >= cut and cut > 0.0
            else "down" if score <= -cut and cut > 0.0 else "")

    for_ = [r for r in live if side and _side(r["dir"]) == side]
    against = [r for r in live if side and _side(r["dir"]) != side]
    quiet = [r for r in rows if r["weight"] <= 0.0]

    return {
        "side": side,
        "score": round(score, 3),
        "call_at": round(cut, 3),
        # HOW MUCH OF THE PANEL IS EVEN TALKING. Four strong rows
        # out of six is a different reading from two out of six
        # that happen to agree, and the score alone cannot tell
        # them apart.
        "speaking": len(live),
        "of": len(rows),
        "rows": rows,
        "for": [r["row"] for r in for_],
        "against": [r["row"] for r in against],
        "quiet": [r["row"] for r in quiet],
        # WHAT THE RULE SAYS ABOUT THE SAME BAR, carried so the two
        # can be seen disagreeing. That disagreement is the useful
        # part: the rule is silent for about fifteen seconds at the
        # start of every move by its own design, and this says what
        # it would have said.
        "rule_side": str(f.get("entry") or ""),
        "rule_where": str(f.get("where") or ""),
        "ahead_of_rule": bool(side and not f.get("entry")),
        "say": _say(side, score, for_, against, live, f),
        # IT DECIDES NOTHING. Asserted by a test, and said on every
        # payload because a panel that claims a rule nobody is
        # running has cost him a day already.
        "decides": False,
    }


def _say(side: str, score: float, for_: list, against: list,
         live: list, f: dict) -> str:
    """One sentence, naming what carried it and what did not.

    A CALL WITH NO DISSENT LISTED IS A CALL THAT CANNOT BE CHECKED.

    THE NO-CALL CASE READS `live`, NOT `for_`. Those two lists are
    built against the called side, so when there is no side they
    are both empty -- and the first version of this said "nothing
    is pointing anywhere yet" on a panel where five rows were
    pointing and simply disagreed. Two different silences, and the
    one that matters is the one where the panel is arguing with
    itself.
    """
    if not side:
        if not live:
            return "nothing on the panel is pointing anywhere yet"
        ups = [r["row"] for r in live if r["dir"] == "up"]
        downs = [r["row"] for r in live if r["dir"] == "down"]
        if ups and downs:
            return (f"split -- {', '.join(ups)} up against "
                    f"{', '.join(downs)} down, not enough to call")
        lean = "up" if ups else "down"
        return (f"leaning {lean} at {abs(score):.0%}, under the "
                "line to call it")
    names = ", ".join(r["row"] for r in for_) or "nothing"
    out = f"{names} say {side}"
    if against:
        out += " -- against it: " + ", ".join(r["row"] for r in against)
    where = str(f.get("where") or "")
    if not f.get("entry") and where:
        out += f"; the rule is still {where}"
    return out
