"""TWO TICK CHARTS, ONE LEADING AND ONE CONFIRMING.

Chris: "lets build a strategy that is just like the current
strategy but just between gate 2 and gate 3 using gate 2 as the
lead and gate 3 as the confirmation with the same 5 second hold on
gate 3 before a trade is taken".

WHAT IS DIFFERENT FROM `stackwin`. There is no timeframe in this
rule at all. The live 1m/5m/15m candle, the window's four legs, the
stack rows below it, the no-trade band off the timeframe's open and
the time stop all go with it -- not demoted to "shown only",
ACTUALLY GONE, because the thing they were measured against is not
here. What is left is two cuts of the same tape:

    GATE 2  the lead. It names the direction and its flip is the
            only thing that closes a trade.
    GATE 3  the confirmation. It has to agree, and its agreement
            has to stand five seconds before anything opens.

WHAT IS THE SAME. The body of the bar, close against open, held
through the instant close sits on open. One trade per agreement.
Two trades per five-minute bar. Flat is never a flip. Those are not
copied here -- `stackwin` owns them and this module calls them, so
a fix to "flat is not a flip" cannot land in one rule and miss the
other.

NO CLOCK ENDS A TRADE. Asked which exits this strategy should have,
Chris picked "Gate 2's flip only, no clock": the lead tick bar
turning is the exit. The five-minute stamp is still taken, but only
as a COUNTER for the cap -- nothing expires because of it.

THIS MODULE SUGGESTS. It returns a side and a sentence. It does not
place, submit, execute or route anything, and a test asserts that
no function here is named as though it did.
"""

from __future__ import annotations

from .stackwin import (  # noqa: F401
    DWELL_S,
    SIDES,
    Dwell,
    bar_of,
    body_dir,
    hold_dir,
    holds,
)

# TWO TRADES A FIVE-MINUTE BAR, the same ceiling the stack rule
# runs under. Chris: "we need to set a rule for no more then 2
# trades on a 5 minute timeframe". It matters MORE here, not less:
# with no clock on the trade and no band off an open, the cap is
# the only thing bounding how often this rule can churn.
CAP_BAR_S = 300.0
MAX_TRADES_PER_BAR = 2


def entry_side(lead: str, confirm: str,
               held_s: float | None = None,
               min_hold_s: float = DWELL_S,
               episode_taken: bool = False) -> tuple[str, str]:
    """The side both tick charts are offering, and why not.

    THE ORDER IS THE RULE. Each refusal names the thing that
    actually refused, because a panel that says "no trade" without
    saying which gate said no is the complaint this whole engine
    keeps getting rebuilt after.

    THE HOLD IS ON THE CONFIRMATION. Chris: "the same 5 second
    hold on gate 3 before a trade is taken". It is measured on gate
    3's HELD direction, not its raw one -- a reading that blanks at
    a tie would restart the hold on an absence, which is how the
    ten-second version never elapsed.
    """
    if lead not in SIDES:
        return "", "the lead tick chart has no direction yet"
    if confirm not in SIDES:
        return "", "the confirming tick chart has no direction yet"
    if confirm != lead:
        return "", (f"the lead chart is {lead} but the confirming "
                    f"chart is {confirm}")
    if held_s is not None and min_hold_s > 0 and held_s < min_hold_s:
        return "", (f"the confirming chart turned {confirm} "
                    f"{held_s:.0f}s ago -- it has to hold "
                    f"{min_hold_s:.0f}s")
    if episode_taken:
        return "", ("this agreement has already been traded -- they "
                    "have to disagree before another is taken")
    return lead, ""


def exit_reason(side: str, lead: str, confirm: str = "",
                confirm_exits: bool = False) -> str:
    """Why this trade should end now, or "" to hold it.

    THE LEAD TURNING, AND NOTHING ELSE BY DEFAULT. Asked what
    should close a trade here, Chris picked "Gate 2's flip only, no
    clock". There is no time stop in this rule: a trade runs until
    the chart that opened it points the other way.

    THE CONFIRMATION CLOSES TOO ONLY IF THE SWITCH IS ON -- the
    same `g3 exit` switch the stack rule uses, read LAST so that
    with it off this returns exactly what the lead alone decided.
    The switch is honoured here rather than ignored because a
    control that does nothing in one strategy and something in
    another is worse than either behaviour on its own.

    FLAT IS NOT A FLIP, on either chart. An absence must never read
    as a reversal; that is the most expensive mistake this engine
    has made and it has made it three times.
    """
    if side not in SIDES:
        return ""
    ok, why = holds(side, lead)
    if not ok:
        return why.replace("the tick bar", "the lead chart")
    if confirm_exits:
        ok3, why3 = holds(side, confirm)
        if not ok3:
            return why3.replace("the tick bar", "the confirming chart")
    return ""


def _say(side: str, why: str, confirm_exits: bool) -> str:
    """One sentence for the panel, in the rule's own words."""
    if side not in SIDES:
        return why or "waiting on the two tick charts"
    out = "out when the lead chart flips"
    if confirm_exits:
        out += " or the confirming chart flips"
    return (f"both tick charts are {side} and the confirmation has "
            f"held -- in on this, {out}")


def state(lead_raw: str = "", confirm_raw: str = "",
          lead_bar: int = 0, confirm_bar: int = 0,
          held_s: float | None = None,
          min_hold_s: float = DWELL_S,
          confirm_exits: bool = False,
          lead_roll: dict | None = None,
          confirm_roll: dict | None = None,
          episode_taken: bool = False,
          prev: dict | None = None) -> dict:
    """Everything the stepper and the panel need, from one place.

    ONE READING, NOT TWO. The panel is handed what the rule
    decided rather than the inputs to decide it again. A screen
    that recomputes a direction is a screen that eventually
    disagrees with the rule beside it, and that has happened here
    with the 5-minute cell and with the OFI bar already.

    THE MEMORY RIDES HOME IN `held`. These functions are pure; the
    caller keeps the previous reading and hands it back, so both
    charts can hold their direction through a tie without this
    module owning any state.
    """
    prev = prev or {}
    lead = hold_dir(str(lead_raw or ""), str(prev.get("lead") or ""))
    confirm = hold_dir(str(confirm_raw or ""),
                       str(prev.get("confirm") or ""))

    agreeing = bool(lead in SIDES and lead == confirm)
    prev_ep = (prev.get("episode") or {})
    same_run = bool(agreeing and prev_ep.get("side") == lead)
    spent = bool(episode_taken or (same_run and prev_ep.get("taken")))

    side, why = entry_side(lead, confirm, held_s=held_s,
                           min_hold_s=min_hold_s,
                           episode_taken=spent)

    ready = not (held_s is not None and min_hold_s > 0
                 and held_s < min_hold_s)
    return {
        "mode": "ticks",
        "side": side,
        "why": why,
        "say": _say(side, why, confirm_exits),
        # THE LEAD. Its flip is the exit, so it is drawn first and
        # labelled with the job it does rather than its number.
        "lead": {
            "side": lead,
            "raw": str(lead_raw or ""),
            "per_bar": int(lead_bar or 0),
            "true": lead in SIDES,
            "held_through_flat": bool(lead in SIDES
                                      and not (lead_raw or "")),
            "roll": dict(lead_roll or {}),
            "decides": True,
        },
        # THE CONFIRMATION, and the hold that sits on it.
        "confirm": {
            "side": confirm,
            "raw": str(confirm_raw or ""),
            "per_bar": int(confirm_bar or 0),
            "true": confirm in SIDES,
            "held_through_flat": bool(confirm in SIDES
                                      and not (confirm_raw or "")),
            "agrees": agreeing,
            "held_s": (None if held_s is None else float(held_s)),
            "min_hold_s": float(min_hold_s or 0.0),
            "ready": ready,
            # A SIZE THAT MATCHES THE LEAD IS THE SAME READING
            # TWICE. It agrees by construction and confirms
            # nothing -- said out loud rather than silently
            # blocked, because picking it is a legitimate thing to
            # do and a silent no-op is not.
            "same_as_lead": bool(lead_bar and confirm_bar
                                 and int(lead_bar) == int(confirm_bar)),
            "roll": dict(confirm_roll or {}),
            "exits": bool(confirm_exits),
            "decides": True,
        },
        "episode": {
            "side": lead if agreeing else "",
            "agreeing": agreeing,
            "same_run": same_run,
            "taken": spent,
            "decides": True,
        },
        # EVERY FIELD THE NEXT READING NEEDS BACK. A field dropped
        # from here does not error -- it silently resets that part
        # of the reading on every poll, which is how a hold that
        # never elapsed and a direction that chattered both got
        # shipped.
        "held": {
            "lead": lead,
            "confirm": confirm,
            "episode": {"side": lead if agreeing else "",
                        "taken": spent},
        },
    }
