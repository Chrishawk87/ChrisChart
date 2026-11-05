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

# THE 5-MINUTE BAR IS READ IMMEDIATELY. Chris: "the 5 minute is the
# only one that dont need to be settled".
#
# It was held flat for its first 90 seconds, on the reasoning that a
# young bar's direction is a handful of prints against its open. The
# cost of that was worse than the noise it removed: the 5m is the
# SLOW reading in this rule, the one whose job is to be behind the
# other two, and delaying it further made it turn visibly later than
# the same candle on the chart -- a confirmation that disagreed with
# what Chris could see.
#
# Close against open, exactly what colours the 5m candle on the
# chart. Kept as a parameter rather than deleted so a settle can be
# put back by passing one, but the default is none.
M5_SETTLE_S = 0.0


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


def m5_direction(m5: dict | None,
                 settle_s: float = M5_SETTLE_S) -> tuple[str, str]:
    """The forming 5-minute bar's direction, or "" and why not.

    Close against open -- the same thing that colours this candle on
    the chart, so the rule and the screen cannot disagree about which
    way the 5m is pointing.

    No settle by default. A bar that is genuinely flat still has no
    direction, and that is reported the same way, because the rule
    treats "flat" and "no confirmation" identically.
    """
    if not m5:
        return "", "no 5-minute bar yet"
    try:
        age = float(m5.get("seconds") or 0.0)
    except (TypeError, ValueError):
        age = 0.0
    if settle_s > 0 and age < settle_s:
        return "", (f"the 5m bar is {int(age)}s old -- it is not read "
                    f"until {int(settle_s)}s")
    d = str(m5.get("dir") or "").strip().lower()
    if d not in SIDES:
        return "", "the 5m bar is flat"
    return d, ""


def entry_side(tick_dir: str, m5_dir: str) -> tuple[str, str]:
    """THE TICK BAR AND THE 5-MINUTE BAR, both pointing the same way.

    Chris: "a entry follows the same rules both the tick bar and the
    5 minute has to agree". The same two readings that end a trade
    are the two that start one, so the rule is one sentence in both
    directions and cannot be stated two ways.

    THE OFI IS NO LONGER PART OF THE RULE AT ALL. It is still built,
    still read and still on the panel -- it is worth seeing whether
    the resting book concurs -- but it gates nothing. It flips sign
    about once a bar even against a wide deadband because it
    measures size that is withdrawn and replaced constantly, and
    every version of this rule that let it decide something churned.

    What is left is two measurements of PRICE THAT HAPPENED over
    very different spans: a few thousand prints, and five minutes of
    clock. They are independent in the way that matters -- one is
    cut by activity and the other by time -- so their agreeing is
    still evidence rather than arithmetic.

    Flat is not a side. Both must actually point somewhere.
    """
    t = (tick_dir or "").strip().lower()
    m = (m5_dir or "").strip().lower()
    if t not in SIDES and m not in SIDES:
        return "", "neither the tick bar nor the 5m has a direction"
    if t not in SIDES:
        return "", f"the 5m is {m} but the tick bar is flat"
    if m not in SIDES:
        return "", f"the tick bar is {t} but the 5m is flat"
    if t != m:
        return "", f"the tick bar is {t} and the 5m is {m}"
    return t, ""


def holds_exit(side: str, tick_dir: str, m5_dir: str) -> tuple[bool, str]:
    """Does an open `side` survive? Only two of the three can end it.

    THE OFI IS AN ENTRY FILTER, NOT AN EXIT. Chris: "the ones that
    have to disagree in order for a exit to happen is the tick bar
    and the 5 minute".

    That is the right leg to drop, and the flip counts say why. The
    OFI accumulator changes sign 3.17 times a bar against a zero
    deadband -- 1.00 a bar even at +/-500 -- because it measures
    RESTING size, which is withdrawn and replaced constantly inside
    a move that is going nowhere unusual. It is excellent at saying
    "the book is behind this" at the moment of entry and hopeless as
    a reason to abandon a position thirty seconds later.

    The tick bar and the 5m both measure PRICE THAT HAPPENED, over
    different spans. When either of those turns against the trade,
    the thing being traded has actually moved the other way.

    FLAT IS NOT A DISAGREEMENT HERE, and that is deliberate -- it is
    the answer Chris already gave for the 5m ("flat is not a reason
    to exit"), applied to both. A bar whose close sits exactly on
    its open has not turned against anything; it is momentarily
    uncommitted, and exiting on that is the churn this rule keeps
    being rewritten to remove. Only an OPPOSITE direction ends it.

    Note the asymmetry with the entry, which does require all three
    and does treat flat as "no". In is strict, out is narrow: it
    takes more to start a trade than to keep one, which is the shape
    every version of this rule has been converging on.
    """
    if side not in SIDES:
        return False, "no side"
    t = (tick_dir or "").strip().lower()
    m = (m5_dir or "").strip().lower()
    if t in SIDES and t != side:
        return False, f"the tick bar turned {t} against a {side}"
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
    # The OFI, as information only. It no longer gates anything --
    # `concurs` says whether the resting book happens to be pointing
    # the same way as the decision, which is worth seeing and is not
    # allowed to change it.
    concurs = bool(o_dir in SIDES)
    side, why = entry_side(t_dir, m_dir)
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
            "seconds": (m5 or {}).get("seconds"),
            # The bar this reading belongs to. The caller caps
            # entries at one per 5m bar and needs to know WHICH bar,
            # not just how far into it we are.
            "start_ts": (m5 or {}).get("start_ts"),
            "settle_s": M5_SETTLE_S,
            "why": m_why,
            "open": (m5 or {}).get("open"),
            "close": (m5 or {}).get("close"),
            "ticks": (m5 or {}).get("ticks"),
        },
        # Whether the resting book agrees with the decision. NOT an
        # input to it: shown so a trade taken against the book can
        # be seen being taken, not prevented.
        "ofi_concurs": (concurs and o_dir == side) if side else False,
        "ofi_known": concurs,
        "side": side,
        "agree": bool(side),
        "why": why,
        "say": (f"the tick bar and the 5m both point {side}" if side
                else why or "no reading yet"),
    }
