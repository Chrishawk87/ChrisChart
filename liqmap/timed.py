"""The timed strategy: the 5-minute candle, guided by the 1-minute.

Chris, after watching the fusion rule bleed out through its exits:

    "there where 12 in the last few hours ranging anywhere from 2 to
    5 ticks thats not good ... what if we take trades only based on
    the 5 minute timeframe using the 1 minute as a entry que for the
    direction 1 trade per 5 minute timeframe when 1 and 5 agree exit
    only on take profit, SL or candle reversal"

and then, on what the 1-minute is actually for:

    "the one minute is used for the truth telling for the 5 minute it
    tells the fight thats happening ... consider the 1 minute shows
    continuation and pullbacks so allow the 1 minute to determine
    when to get in and out of a 5 minute candle dont wait for a full
    pullback into the negative before exiting a trade but to guide
    it"

TWO TIMEFRAMES, ONE OF THEM DOING TWO JOBS.

    THE 5-MINUTE is the position. It says which way this stretch of
    the session is going, and a trade belongs to one of its candles.

    THE 1-MINUTE is the truth teller. Inside a 5-minute candle there
    is a fight, and the 1-minute is where it shows: continuation
    while the minutes keep closing the same way, a pullback the
    moment one closes against. It decides WHEN to get in, and WHEN
    to get out, of a move the 5-minute has already named.

WHY THIS IS NOT THE RULE IT REPLACES.

    The fusion rule exited whenever its two legs stopped agreeing,
    and one of those legs could go flat on a single print. Twelve
    exits in a few hours, every one of them paying the spread to
    find out nothing had happened.

    Here an exit is an EVENT, not an absence: a 1-minute candle has
    to close back through its own open, against the position. That
    is a thing the market did. Flat does not do it, a quiet minute
    does not do it, and a leg going briefly undecided does not do
    it.

    The point of using the 1-minute for it rather than the 5-minute
    is the sentence above: not waiting for a full pullback into the
    negative. By the time a 5-minute candle crosses its own open the
    move is usually given back. The 1-minute says so first.

DIRECTION IS CLOSE AGAINST OPEN, on both, which is the same thing
that colours the candle on the chart. There is no third definition
here and no smoothing: "crosses its open" is only meaningful if the
direction IS the side of the open that price is on.

ONE TRADE PER 5-MINUTE CANDLE. Stated twice and enforced by the
caller. A candle that turns you around twice is a candle that cost
you two spreads to end up where you started.

Nothing in this module places, sizes, routes or times an order. It
compares two candles.
"""

from __future__ import annotations

SIDES = ("up", "down")


def direction(candle: dict | None) -> str:
    """Close against open. Flat when they are equal or unknown.

    The same measure the reversal is defined against, deliberately.
    If direction were a different thing -- a smoothed slope, a
    momentum sign -- then "crosses its open" and "turned" would be
    two different events and the rule would have two meanings.
    """
    if not candle:
        return "flat"
    try:
        o = float(candle.get("open") or 0.0)
        c = float(candle.get("close") or 0.0)
    except (TypeError, ValueError):
        return "flat"
    if o <= 0 or c <= 0:
        return "flat"
    if c > o:
        return "up"
    if c < o:
        return "down"
    return "flat"


def entry_side(m1_dir: str, m5_dir: str) -> tuple[str, str]:
    """Both candles the same way, or "" and the reason there is none.

    The 5-minute names the direction; the 1-minute has to be going
    along with it for the entry to be a continuation rather than a
    guess about one.

    Flat is not a side. A 1-minute candle sitting exactly on its
    open is the fight unresolved, which is the one moment this rule
    has no opinion.
    """
    a = (m1_dir or "").strip().lower()
    b = (m5_dir or "").strip().lower()
    if a not in SIDES and b not in SIDES:
        return "", "neither the 1m nor the 5m has a direction"
    if b not in SIDES:
        return "", f"the 1m is {a} but the 5m is flat"
    if a not in SIDES:
        return "", f"the 5m is {b} but the 1m is flat -- the fight is open"
    if a != b:
        return "", f"the 5m is {b} but the 1m is {a} -- that is a pullback"
    return a, ""


def reversed_against(side: str, m1_dir: str) -> tuple[bool, str]:
    """Has the 1-minute closed back through its open, against us?

    THE ONLY SIGNAL EXIT. Not "the legs disagree" -- an event. A
    1-minute candle has to be pointing the OTHER way, which means
    price is on the other side of that candle's open. Flat does not
    end a trade and neither does a quiet minute: those are absences,
    and paying a spread for an absence is what the last rule did
    twelve times in an afternoon.

    Take profit and stop loss are checked before this by the caller,
    and they are the other two ways out.
    """
    if side not in SIDES:
        return False, ""
    d = (m1_dir or "").strip().lower()
    if d in SIDES and d != side:
        return True, (f"the 1m closed back through its open and is {d} "
                      f"against a {side}")
    return False, ""


def state(m1: dict | None, m5: dict | None,
          m5_dir: str = "") -> dict:
    """Everything the rule and the panel need, from one place.

    Built from the candles themselves rather than from a direction
    computed elsewhere: two readings of "which way is this candle
    pointing" is two things to keep correct, and the first time they
    differ the screen and the rule are describing different markets.

    `m5_dir` OVERRIDES THE 5-MINUTE LEG. Chris:

        "all i want is for the direction who is crossing the spread
        to recompute every 5 minutes thats all"

    Close-against-open says which way price ENDED UP. It does not
    say who is crossing the spread to put it there, and four
    different ways of reading it all lost on 109 hours of tape while
    taking the same 1,194 trades -- the leg was not selecting
    anything. The aggressor direction is a different fact about the
    same tape: signed volume that actually crossed, cleared at each
    5-minute boundary.

    It arrives as an argument rather than being computed here
    because the window that produces it is fed print by print off
    the tape and cannot be rebuilt from a candle dict. The 1-minute
    leg and the reversal exit are untouched; both still read price,
    which is what keeps "crosses its open" and "turned" the same
    event on the leg that uses them.

    An empty or unknown override falls back to the candle, so a
    missing tape degrades to the old reading instead of flat-lining
    the rule -- and `m5["source"]` says which one was used, so the
    panel can never imply a reading the rule did not take.
    """
    src = "price"
    if m5_dir in ("up", "down", "flat"):
        d5, src = m5_dir, "flow"
    else:
        d5 = direction(m5)
    d1 = direction(m1)
    side, why = entry_side(d1, d5)
    m1 = m1 or {}
    m5 = m5 or {}
    return {
        "m1": {"dir": d1, "open": m1.get("open"), "close": m1.get("close"),
               "ticks": m1.get("ticks"), "seconds": m1.get("seconds")},
        "m5": {"dir": d5, "open": m5.get("open"), "close": m5.get("close"),
               "ticks": m5.get("ticks"), "seconds": m5.get("seconds"),
               "start_ts": m5.get("start_ts"),
               # Which measure produced `dir`, and the signed
               # contracts behind it when that measure is the flow.
               # A direction with no magnitude cannot be audited.
               "source": src, "push": m5.get("push")},
        "side": side,
        "agree": bool(side),
        "why": why,
        "say": (f"the 1m and the 5m are both {side}" if side
                else why or "no reading yet"),
    }
