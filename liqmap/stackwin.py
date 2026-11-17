"""The window gates, then the stack confirms. In that order.

Chris:

    "a buy or a sell is based on what the stack says and what the
    window says but first the window needs to read true before
    anything else can be done"

    "on a 5 minute the 5 min and 1 minute must agree as well as the
    window must agree candle, OFI, CVD, Footprint. and on the 15 min
    the 15 and 5 must agree with the same 4 from the window"

WHY THIS IS A NEW STRATEGY AND NOT A FIX.

    He watched the window line show candle, OFI, CVD and footprint
    all DOWN and no trade follow, and read that as the window being
    broken. It was not: NOTHING IN THE APP ASKED FOR ALL FOUR.

        `ofi`        candle + OFI + CVD   (three)
        `footprint`  candle + footprint   (two)
        `timed`      returns before the window is looked at
        `fused`      the same

    So a line showing four agreeing drove nothing, which is the same
    shape as the bug where the column traded the bucket bar while
    the window was only ever a comparison panel. A panel that shows
    an agreement nothing acts on reads as broken whichever way round
    the fault is.

THE ORDER IS LOAD-BEARING, not presentation. The window is a gate:
when it does not read true, the stack is not consulted and no
reason mentioning the stack is produced. Said plainly because a
rule that checks both and reports whichever failed first is a
DIFFERENT rule -- it fires on the same trades but explains them
wrongly, and every wrong explanation here has cost a day.

THE STACK IS THE TRADED ROW AND THE ONE BELOW IT, not the whole
stack. Fifteen takes five; five takes one. The row above is
deliberately not consulted: on the 5-minute he is trading the
5-minute, and a 15-minute that disagrees is the pullback he is
trading into, not a veto.

FLAT IS NOT A DIRECTION, on any of the six readings. Six zeros are
not six signals pointing together, and a window with nothing in it
would otherwise read as the strongest there is.

Nothing in this module places, sizes, routes or times an order.
"""

from __future__ import annotations

SIDES = ("up", "down")
# The row each traded timeframe takes its confirmation from.
BELOW_OF = {"15m": "5m", "5m": "1m"}
# Which row the window's period is trading.
TRADED_OF = {300.0: "5m", 900.0: "15m"}
# The four readings the window is built from, in the order a reason
# names them.
LEGS = ("candle", "OFI", "CVD", "footprint")


def _other(side: str) -> str:
    return "down" if side == "up" else "up" if side == "down" else ""


def traded_tf(period_s: float) -> str:
    """Which stack row a window of this length is trading."""
    try:
        p = float(period_s or 0.0)
    except (TypeError, ValueError):
        return ""
    return TRADED_OF.get(p, "15m" if p >= 600.0 else "5m")


def window_side(candle: str, ofi: str, cvd: str,
                fp: str) -> tuple[str, str]:
    """The window's direction, or why it has none.

    All four the same way. The reason names every leg that is not
    with the majority rather than the first one found, because "the
    candle is down but the CVD is up" sends you looking at the CVD
    when the footprint was also against it.
    """
    legs = dict(zip(LEGS, (candle or "", ofi or "", cvd or "",
                           fp or "")))
    flat = [n for n, v in legs.items() if v not in SIDES]
    if flat:
        return "", ("no side -- " + ", ".join(flat)
                    + (" have" if len(flat) > 1 else " has")
                    + " no direction")
    up = [n for n, v in legs.items() if v == "up"]
    if len(up) == 4:
        return "up", ""
    if not up:
        return "down", ""
    # Mixed. Name the MINORITY, which is the thing to go and look
    # at -- "the candle is down but the CVD is up" sends you to the
    # CVD when the footprint was against it too.
    down = [n for n, v in legs.items() if v == "down"]
    if len(up) == len(down):
        # Two against two has no minority, and picking one would
        # invent a majority that does not exist.
        return "", (f"two against two -- {', '.join(up)} up, "
                    f"{', '.join(down)} down")
    minority, major = (up, "down") if len(up) < len(down) else (down, "up")
    return "", (f"{', '.join(minority)} "
                f"{'disagree' if len(minority) > 1 else 'disagrees'}"
                f" -- the other {4 - len(minority)} are {major}")


def stack_side(rungs: dict, period_s: float) -> tuple[str, str]:
    """The traded row and the one below it, or why they do not say.

    `rungs` maps timeframe to its reading, as the cascade reports
    them: `qualified` is the direction a row is entitled to claim,
    and it is None when the row's colour and its own number
    contradict each other -- which is not a direction and must not
    be read as one.
    """
    tf = traded_tf(period_s)
    below = BELOW_OF.get(tf, "")
    if not tf or not below:
        return "", f"no stack row for a {period_s:.0f}s window"

    out = {}
    for name in (tf, below):
        row = (rungs or {}).get(name) or {}
        if not row.get("present"):
            return "", f"the {name} has not been read yet"
        if row.get("conflicted"):
            return "", (f"the {name} contradicts itself -- "
                        f"thrown out")
        q = row.get("qualified") or ""
        if q not in SIDES:
            return "", f"the {name} has no direction"
        out[name] = q

    if out[tf] != out[below]:
        return "", (f"the {tf} is {out[tf]} but the {below} is "
                    f"{out[below]}")
    return out[tf], ""


def entry_side(fp: str, rungs: dict,
               period_s: float) -> tuple[str, str]:
    """The side to take, or the reason there is none.

    THE FOOTPRINT GATES AND THE STACK CONFIRMS. Chris: "we are just
    going to use the stack and the footprint thats all the make
    entry and exits".

    The candle, the OFI and the CVD are still read and still shown,
    and they decide nothing. They are kept on the panel under their
    own heading rather than deleted, because a leg that quietly
    stops counting is the thing that makes a screen lie -- he has
    already watched four readings agree and drive nothing.

    THE FOOTPRINT IS ASKED FIRST AND ALONE. When it has no
    imbalance the stack is never consulted and the reason does not
    mention it.
    """
    if fp not in SIDES:
        return "", "the footprint shows no imbalance either way"

    s_side, s_why = stack_side(rungs, period_s)
    if not s_side:
        return "", (f"the footprint is {fp} but the stack does not "
                    f"back it: {s_why}")
    if s_side != fp:
        return "", (f"the footprint is {fp} and the stack is "
                    f"{s_side} -- they point apart")
    return fp, ""


def holds(side: str, fp: str, rungs: dict,
          period_s: float) -> tuple[bool, str]:
    """Does the signal still back an open trade?

    THE MOMENT THEY DISAGREE, IT IS OVER. Chris: "you are going to
    auto close trades the minute they disagree because right now its
    holding the trades for no reason".

    THIS IS THE TIGHT EXIT, DELIBERATELY, AND IT HAS A HISTORY. The
    rule that exited the moment agreement broke closed twelve trades
    in an afternoon at two to five ticks each, and was replaced
    twice for it. What is different now is the entry: those exits
    came from a four-leg window where one leg going quiet for an
    instant broke agreement constantly. Two readings break less
    often than four -- but if it churns, the exit is where to look,
    and the reason string below is what will show it.

    A reading going FLAT ends it too, and that is the part most
    likely to bite: a footprint between bars has no side, and that
    is not the same as the market turning. It is counted as a
    disagreement because he asked for the minute they disagree, not
    the minute they oppose.
    """
    if side not in SIDES:
        return True, ""
    now, why = entry_side(fp, rungs, period_s)
    if now == side:
        return True, ""
    if now:
        return False, (f"the signal turned {now} against your "
                       f"{side} trade")
    return False, (why or "the signal no longer backs it")


def reversed_window(side: str, candle: str, ofi: str, cvd: str,
                    fp: str) -> tuple[bool, str]:
    """Has the window turned fully against an open trade?

    ALL FOUR THE OTHER WAY, not merely losing agreement. Chris chose
    this over the tighter exit for a reason he has paid for twice:
    exiting the moment agreement breaks closed twelve trades in an
    afternoon at two to five ticks each, none of which was a
    decision about the market. A reading that has gone quiet is not
    a reading that has gone against you.
    """
    if side not in SIDES:
        return False, ""
    against = _other(side)
    w, _why = window_side(candle, ofi, cvd, fp)
    if w == against:
        return True, (f"the window turned fully {against} against "
                      f"your {side} trade")
    return False, ""


def state(win: dict | None, fp: str, rungs: dict | None,
          period_s: float) -> dict:
    """Everything the rule and the panel need, from one place.

    One source, so the screen cannot show an agreement the rule did
    not see. Every previous panel in this engine that computed its
    own copy of a direction ended up telling a different story from
    the rule beside it.
    """
    win = win or {}
    candle = str(win.get("mid_dir") or "")
    ofi = str(win.get("ofi_dir") or "")
    cvd = str(win.get("cvd_dir") or "")
    fp = str(fp or "")
    tf = traded_tf(period_s)
    below = BELOW_OF.get(tf, "")

    # The four-leg reading is kept and SHOWN. It decides nothing
    # now; leaving it out would make the panel quieter and the rule
    # less checkable, and a leg that silently stops counting is what
    # makes a screen lie.
    w_side, w_why = window_side(candle, ofi, cvd, fp)
    s_side, s_why = stack_side(rungs or {}, period_s)
    side, why = entry_side(fp, rungs or {}, period_s)

    rows = {}
    for name in (tf, below):
        if not name:
            continue
        row = (rungs or {}).get(name) or {}
        rows[name] = {
            "present": bool(row.get("present")),
            "dir": row.get("qualified") or "",
            # Shown beside the direction, not instead of it.
            "absorbed": bool(row.get("absorbed")
                             or row.get("conflicted")),
            "seeded": row.get("seeded", True),
            "bps": row.get("bps"),
            "open_px": row.get("open_px"),
            "last_px": row.get("last_px"),
            "ticks": row.get("ticks"),
        }

    return {
        "footprint": {
            "side": fp,
            "true": fp in SIDES,
            "why": ("" if fp in SIDES
                    else "no imbalance either way"),
        },
        "stack": {
            "trading": tf, "below": below,
            "rows": rows,
            "side": s_side, "why": s_why,
            # Never consulted while the footprint has no side, and
            # the panel says that rather than showing a verdict the
            # rule did not ask for.
            "asked": fp in SIDES,
        },
        # SHOWN, NOT USED. Under its own heading so there is no
        # chance of reading it as part of the decision.
        "context": {
            "period_s": float(period_s or 0.0),
            "candle": candle, "ofi": ofi, "cvd": cvd,
            "side": w_side, "why": w_why,
            "decides": False,
        },
        "side": side,
        "why": why,
        "say": (f"the footprint and the {tf}/{below} stack are both "
                f"{side}" if side else why),
    }
