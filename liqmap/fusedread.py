"""What the tick chart is doing INSIDE the fused bar.

Chris: "on my chart there is a hybrid that was created its a
10000tick x 4000 contract OFI and it works its a tick chart inside
a OFI candle what i want to see is what the tick chart is doing
inside the OFI, like moving down together, moving up together ofi
remaining down but tick is bouncing from bottom or top or read it
like normal candle and OFI trying to go up or down".

A fused bar carries two readings of the same stretch of tape and
they do not have to agree:

    THE BOOK -- the signed OFI accumulator since the bar opened,
    and how far it has pushed toward its target. This is the bar's
    direction and its colour; `fusedbar` already says so.

    THE PRICE -- open, high, low and close of the prints that
    arrived in the same window. Its own direction, and crucially
    WHERE IN ITS OWN RANGE it currently sits.

This module answers one question off those two: are they going the
same way, and if they are, is price still at the business end of
the move or has it come back off it. That is the difference
between "moving down together" and "OFI remaining down but tick is
bouncing from bottom", which are the two states he named and which
look identical if you only read directions.

THE RETRACE IS `tapeintent.depth`, NOT A SECOND MEASURE OF THE
SAME THING. It answers "how much of this bar's move is gone",
measured against the move the bar MADE rather than its full range,
and it is already the number the tick panel shows. A second
implementation would drift from it and then two panels would
disagree about one bar.

MEASURED AGAINST THE BOOK'S DIRECTION, not price's. His case is
"OFI remaining down but tick is bouncing" -- the book's commitment
is the frame and price's position inside it is the question. On an
OFI-down bar that means the span from the open to the low, so 0.0
is price pinned at the low with the book short, and 1.0 is price
all the way back to where the bar opened.

BOUNCE_AT IS A LABEL BOUNDARY, NOT A FINDING. It decides which
word the panel prints and nothing else; the raw percentage is
drawn beside it either way, so nothing is hidden behind the
threshold and no trade turns on it.

THIS DECIDES NOTHING. Chris: "Agreeing, bounce shown only" -- and
at this point there is no strategy behind it at all. It is a panel
to watch. Every block it emits carries `decides: False`, and a
test asserts that, because on this engine a reading that is drawn
like a gate eventually gets treated as one.
"""

from __future__ import annotations

from .tapeintent import EPS, depth

SIDES = ("up", "down")

# How much of the bar's move has to be given back before the panel
# calls it a bounce rather than a push. A third: enough to be a
# move against the bar rather than a tick of noise at the extreme.
BOUNCE_AT = 0.33

# Where in its own high-low range the close has to sit before the
# panel will say "at the top" or "at the bottom" of the bar.
EDGE = 0.25


def position(high: float, low: float, close: float) -> float | None:
    """Where the close sits in the bar's range, 0 at the low.

    `None` when there is no range yet -- a bar whose prints have
    all landed on one price. That is a real state in thin tape and
    must not read as "at the bottom", which is what 0.0 would.
    """
    try:
        h, lo, c = float(high or 0.0), float(low or 0.0), float(close or 0.0)
    except (TypeError, ValueError):
        return None
    span = h - lo
    if span <= EPS:
        return None
    return max(0.0, min(1.0, (c - lo) / span))


def at_edge(pos: float | None, edge: float = EDGE) -> str:
    """"top", "bottom" or "" for the middle of the range."""
    if pos is None:
        return ""
    if pos >= 1.0 - edge:
        return "top"
    if pos <= edge:
        return "bottom"
    return ""


def legs(net_ofi: float, ofi_target: float,
         prints: float, tick_target: float) -> dict:
    """How far along each leg the forming bar is.

    BOTH NUMBERS, NEVER ONE. A bar can sit at 98% on one leg and 4%
    on the other, and that is the whole reason this bar type
    exists -- `fusedbar.progress` says the same thing and this
    mirrors it so the gate panel does not have to call back into
    the builder for two divisions.
    """
    try:
        no, ot = abs(float(net_ofi or 0.0)), float(ofi_target or 0.0)
        pr, tt = float(prints or 0.0), float(tick_target or 0.0)
    except (TypeError, ValueError):
        return {"ofi_pct": 0.0, "tick_pct": 0.0, "waiting_on": "both"}
    ofi_pct = min(1.0, no / ot) if ot > 0 else 0.0
    tick_pct = min(1.0, pr / tt) if tt > 0 else 0.0
    waiting = ("" if ofi_pct >= 1.0 and tick_pct >= 1.0 else
               "the book" if tick_pct >= 1.0 else
               "prints" if ofi_pct >= 1.0 else "both")
    return {"ofi_pct": round(100.0 * ofi_pct, 1),
            "tick_pct": round(100.0 * tick_pct, 1),
            "waiting_on": waiting}


def _say(where: str, ofi_dir: str, tick_dir: str, pos: float | None,
         back: float, edge: str) -> str:
    """One sentence, in the words he used to describe each state."""
    pct = int(round(100.0 * back))
    if where == "together":
        side = "down" if ofi_dir == "down" else "up"
        end = "low" if side == "down" else "high"
        return (f"moving {side} together -- the book is {side} and "
                f"price is holding the {end}")
    if where == "bounce":
        frm = "bottom" if ofi_dir == "down" else "top"
        return (f"the book is still {ofi_dir} but price is bouncing "
                f"off the {frm} -- {pct}% of the move given back")
    if where == "apart":
        return (f"the book is {ofi_dir} and price is {tick_dir} -- "
                "they are not reading the same bar")
    if not ofi_dir or ofi_dir == "flat":
        return "the book has not committed either way yet"
    if not tick_dir or tick_dir == "flat":
        return (f"the book is {ofi_dir}; price has closed back where "
                "the bar opened")
    if edge:
        return f"price is at the {edge} of the bar"
    return "nothing settled yet"


def read(ofi_dir: str = "", tick_dir: str = "",
         open_px: float = 0.0, high: float = 0.0, low: float = 0.0,
         close: float = 0.0,
         net_ofi: float = 0.0, ofi_target: float = 0.0,
         prints: float = 0.0, tick_target: float = 0.0,
         bounce_at: float = BOUNCE_AT) -> dict:
    """The whole reading of one forming fused bar.

    `where` is the headline and it is one of:

        "together"  both legs the same way, price still at the end
                    of the move it made
        "bounce"    both legs the same way, price has come back off
                    that extreme by more than `bounce_at`
        "apart"     the book and the traded price disagree
        ""          neither has said anything yet

    NOTHING HERE GATES ANYTHING. `decides` is False on every block
    and stays False until he writes the rules: "we can create the
    strategy after i can see what it does ill make entry and exit
    rules then".
    """
    ofi_dir = str(ofi_dir or "")
    tick_dir = str(tick_dir or "")
    pos = position(high, low, close)
    edge = at_edge(pos)
    back = depth(ofi_dir, open_px, high, low, close)
    agree = bool(ofi_dir in SIDES and ofi_dir == tick_dir)

    if ofi_dir not in SIDES:
        where = ""
    elif tick_dir not in SIDES:
        # THE BOOK HAS COMMITTED AND PRICE HAS NOT. Not a
        # disagreement -- price closed exactly where it opened,
        # which is an absence of a reading rather than one against.
        where = ""
    elif not agree:
        where = "apart"
    elif back >= float(bounce_at or 0.0):
        where = "bounce"
    else:
        where = "together"

    return {
        "where": where,
        "agree": agree,
        "ofi_dir": ofi_dir,
        "tick_dir": tick_dir,
        # WHERE PRICE SITS IN ITS OWN RANGE, and how much of the
        # bar's move it has given back. Two different questions and
        # the panel draws both: the first is "top or bottom of the
        # candle", the second is "is this still pushing".
        "pos": pos,
        "at": edge,
        "back": round(back, 3),
        "bounce_at": float(bounce_at or 0.0),
        "legs": legs(net_ofi, ofi_target, prints, tick_target),
        "say": _say(where, ofi_dir, tick_dir, pos, back, edge),
        # SHOWN ONLY. Chris: "Agreeing, bounce shown only", and
        # there is no strategy behind this yet at all.
        "decides": False,
    }
