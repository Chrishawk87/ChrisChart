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


def state(tick: dict | None, ofi: dict | None) -> dict:
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
    side, why = agreement(t_dir, o_dir)
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
        "side": side,
        "agree": bool(side),
        "why": why,
        "say": (f"both point {side}" if side
                else why or "no reading yet"),
    }
