"""Two series read separately, compared on direction only.

THIS IS NOT A FUSED CANDLE. `fusedbar` builds one bar that closes
when both a print count and an imbalance threshold are met -- one
bar, two closing conditions, one direction. That is a different idea
and it is not this one.

Here the two series stay completely apart:

    THE TICK BAR closes every N prints and knows nothing about the
    book. Its direction is its own traded price, close against open.

    THE OFI BAR closes on a quantity of order-flow imbalance and
    knows nothing about how many prints it took. Its direction is
    which side of the band the accumulator is on -- the green/red
    Chris reads off the candle.

They start and end at different moments, they are built by different
builders from different events, and neither is allowed to influence
the other. The ONLY thing shared is the question asked of both:
which way are you pointing right now?

    AGREE      -> there is a side
    DISAGREE   -> there is no side, and whatever is open should end

That is the whole rule. Chris: "they should read independent but
have the same directional interest for a trade to happen".

WHY INDEPENDENCE IS THE POINT AND NOT A DETAIL.

    Two readings of the same thing agree by construction and tell
    you nothing. The reason this pairing can say something is that
    one measures what was PAID FOR -- prints that happened, at
    prices that happened -- and the other measures what is RESTING,
    the book's intent, which can be withdrawn. When a series built
    from one agrees with a series built from the other, the
    agreement is evidence. Compute either from the other's events
    and it stops being evidence and becomes arithmetic.

FLAT IS NOT A SIDE, on either series. A tick bar that closed where it
opened agrees with nothing, however hard the book leaned; an OFI
accumulator sitting inside its band agrees with nothing, however far
price travelled. Two neutrals are not a signal, and without this rule
a dead market reads as the strongest agreement there is.

Nothing in this module places, sizes, routes or times an order. It
compares two directions.
"""

from __future__ import annotations

SIDES = ("up", "down")

# How long a 5-minute bar must have been running before its direction
# is believed. Under this it is reported flat, whatever its close is
# doing.
#
# WITHOUT IT THE FILTER IS NOT A FILTER. Thirty seconds into a
# five-minute bar, its direction is a handful of prints against its
# open and it flips about as freely as the tick bar this is meant to
# steady -- so the churn would survive the fix, wearing a slower
# label. It is Chris's own rule about not trading the unstable open,
# applied to the confirmation rather than to the entry.
#
# The cost is stated rather than hidden: because a settled 5m
# direction is required to ENTER, nothing opens in the first 90
# seconds of each 5-minute bar. That is 30% of the clock, deliberately.
M5_SETTLE_S = 90.0


def agreement(tick_dir: str, ofi_dir: str) -> tuple[str, str]:
    """The side both point at, or "" and the reason there is none.

    The reason is never blank when the side is. A rule that goes
    quiet without saying why reads as a market with nothing in it,
    and on this panel that has meant a missing read, a stale cap and
    a switch left off -- each one costing an afternoon before anyone
    thought to doubt the silence.
    """
    t = (tick_dir or "").strip().lower()
    o = (ofi_dir or "").strip().lower()

    if t not in SIDES and o not in SIDES:
        return "", "neither the tick bar nor the OFI has a direction"
    if t not in SIDES:
        return "", f"the OFI is {o} but the tick bar is flat"
    if o not in SIDES:
        return "", f"the tick bar is {t} but the OFI is flat"
    if t != o:
        return "", f"the tick bar is {t} and the OFI is {o}"
    return t, ""


def holds(side: str, tick_dir: str, ofi_dir: str) -> bool:
    """Does an open `side` still have both series behind it?

    The exit test, stated separately from the entry test so the two
    cannot drift apart. Chris's rule is symmetric: in on agreement,
    out the moment it breaks, in again when it returns.
    """
    if side not in SIDES:
        return False
    return agreement(tick_dir, ofi_dir)[0] == side


def m5_direction(m5: dict | None,
                 settle_s: float = M5_SETTLE_S) -> tuple[str, str]:
    """The forming 5-minute bar's direction, or "" and why not.

    Flat until the bar has been running `settle_s`. An unsettled bar
    is not a bar with no opinion -- it is a bar whose opinion is not
    worth having yet, and the two are reported the same way on
    purpose, because the rule treats both as "no confirmation".
    """
    if not m5:
        return "", "no 5-minute bar yet"
    try:
        age = float(m5.get("seconds") or 0.0)
    except (TypeError, ValueError):
        age = 0.0
    if age < settle_s:
        return "", (f"the 5m bar is {int(age)}s old -- it is not read "
                    f"until {int(settle_s)}s, that is when it is least "
                    "settled")
    d = str(m5.get("dir") or "").strip().lower()
    if d not in SIDES:
        return "", "the 5m bar is flat"
    return d, ""


def entry_side(tick_dir: str, ofi_dir: str, m5_dir: str) -> tuple[str, str]:
    """ALL THREE pointing the same way, or "" and the reason.

    The fast pair is checked first so its reason is the one reported
    when it is the thing missing -- a trader looking at this wants to
    know which reading failed, and "the 5m is flat" is unhelpful when
    the tick bar and the OFI were never agreeing in the first place.
    """
    side, why = agreement(tick_dir, ofi_dir)
    if not side:
        return "", why
    m = (m5_dir or "").strip().lower()
    if m not in SIDES:
        return "", f"the tick bar and the OFI are {side} but the 5m is flat"
    if m != side:
        return "", f"the tick bar and the OFI are {side} but the 5m is {m}"
    return side, ""


def hold_against_m5(side: str, m5_dir: str) -> tuple[bool, str]:
    """Does an open `side` survive? Only the 5m can end it.

    CHRIS'S ANSWER, AND THE WHOLE REASON THIS EXISTS. The fused rule
    took fourteen trades in a few minutes, every one closed because
    the tick bar and the OFI stopped agreeing for a moment. So they
    no longer end a trade at all: once in, the position is held until
    the FIVE-MINUTE bar points the other way.

    A flat or unsettled 5m holds. "flat is not a reason to exit" --
    only a 5m pointing AGAINST the trade closes it.

    THE CONSEQUENCE, SAID OUT LOUD: a full reversal of both fast
    series no longer closes anything either. You can be long while
    the tick bar and the OFI both say down, for as long as the 5m
    stays up. That is what was asked for and it is what makes the
    stop loss the only fast protection left on the position.
    """
    if side not in SIDES:
        return False, "no side"
    m = (m5_dir or "").strip().lower()
    if m in SIDES and m != side:
        return False, f"the 5m turned {m} against a {side}"
    return True, ""


def state(tick: dict | None, ofi: dict | None,
          m5: dict | None = None) -> dict:
    """Everything the panel draws, in one shape.

    Built from the two builders' own readings rather than recomputed
    here: a second implementation of "which way is this bar
    pointing" is a second thing to keep correct, and the first time
    they disagree the screen and the rule are telling different
    stories about the same tape.
    """
    tick = tick or {}
    ofi = ofi or {}
    t_dir = str(tick.get("dir") or "")
    o_dir = str(ofi.get("read") or "")
    m_dir, m_why = m5_direction(m5)
    # The fast pair on its own, kept separately from the verdict so
    # the panel can show WHICH of the two tests is failing: the pair
    # disagreeing and the pair agreeing without the 5m behind them
    # are different situations and used to look identical.
    pair, pair_why = agreement(t_dir, o_dir)
    side, why = entry_side(t_dir, o_dir, m_dir)
    return {
        "tick": {
            "dir": t_dir,
            "per_bar": tick.get("per_bar"),
            "prints": tick.get("prints"),
            "open": tick.get("open"),
            "close": tick.get("close"),
            "ticks": tick.get("ticks"),
            "seconds": tick.get("seconds"),
        },
        "ofi": {
            "dir": o_dir,
            "imbalance": ofi.get("imbalance"),
            "level": ofi.get("level"),
            "delta": ofi.get("delta"),
        },
        "m5": {
            "dir": m_dir,
            "raw_dir": str((m5 or {}).get("dir") or ""),
            "seconds": (m5 or {}).get("seconds"),
            "settle_s": M5_SETTLE_S,
            "settled": bool(m_dir) or not m_why.startswith("the 5m bar is")
            or "flat" in m_why,
            "why": m_why,
            "open": (m5 or {}).get("open"),
            "close": (m5 or {}).get("close"),
        },
        # The fast pair alone. NOT the entry -- it was, and that is
        # what took fourteen trades in a few minutes.
        "pair": pair,
        "pair_why": pair_why,
        "side": side,
        "agree": bool(side),
        "why": why,
        "say": (f"all three point {side}" if side
                else why or "no reading yet"),
    }
