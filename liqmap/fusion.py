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
    """Kept for callers that want the 5m test on its own.

    NO LONGER THE EXIT. Chris: "all 3 must agree for a entry if one
    disagrees it exits that simple" -- so the exit is symmetric with
    the entry again and lives in `holds_three` below. This remains
    because asking "has the 5m turned against this position" is a
    real question worth being able to ask by itself.
    """
    if side not in SIDES:
        return False, "no side"
    m = (m5_dir or "").strip().lower()
    if m in SIDES and m != side:
        return False, f"the 5m turned {m} against a {side}"
    return True, ""


def holds_three(side: str, tick_dir: str, ofi_dir: str,
                m5_dir: str) -> tuple[bool, str]:
    """Do all three still stand behind an open `side`?

    THE EXIT, AND IT IS THE ENTRY TEST RUN AGAIN. Chris: "all 3 must
    agree for a entry if one disagrees it exits that simple". One of
    them going flat is a disagreement, because the reason for the
    trade was that all three pointed the same way and now they do
    not.

    Defined in terms of `entry_side` rather than reimplemented. A
    rule that enters on one definition and exits on another is a bug
    that hides until a live position is sitting in it, and this pair
    has already been rewritten twice.

    THE CHURN THIS CAN CAUSE IS REAL AND IS CAPPED ELSEWHERE. The
    same construction took fourteen trades in a few minutes when the
    pair alone drove it. What stops it now is not a softer exit but
    a hard limit on entries: one trade per 5-minute bar, enforced by
    the caller. The exit can fire as often as it likes; nothing can
    re-enter until a new 5m bar opens.
    """
    if side not in SIDES:
        return False, "no side"
    now, why = entry_side(tick_dir, ofi_dir, m5_dir)
    if now == side:
        return True, ""
    if now:
        return False, f"all three turned {now} against a {side}"
    return False, why or "the three no longer agree"


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
