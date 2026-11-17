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
    """THE TRADED TIMEFRAME, AND NOTHING UNDER IT.

    Chris: "a new trade is taken only when stack 5m and tick bar
    agrees paying no attention to the 1 minute at all".

    It was the traded row AND the one below -- 5m with 1m, 15m with
    5m. The faster row is now ignored entirely for the decision. It
    is still read and still drawn, because a reading that silently
    stops being collected is how a panel starts lying, but it gates
    nothing.

    `qualified` is the direction a row is entitled to claim and it
    is None when the bar's open was never observed, which is not a
    direction and must not be read as one.
    """
    tf = traded_tf(period_s)
    if not tf:
        return "", f"no stack row for a {period_s:.0f}s window"
    row = (rungs or {}).get(tf) or {}
    if not row.get("present"):
        return "", f"the {tf} has not been read yet"
    if not row.get("seeded", True):
        return "", f"the {tf} bar's open was never seen"
    q = row.get("qualified") or ""
    if q not in SIDES:
        return "", f"the {tf} is flat"
    return q, ""


def entry_side(tick: str, rungs: dict | None = None,
               period_s: float = 0.0) -> tuple[str, str]:
    """The side to take, or the reason there is none.

    THE TICK BAR IS THE WHOLE RULE NOW. Chris: "lets do number 2" --
    enter when the tick bar has a direction, exit when it flips.

    `rungs` and `period_s` are still taken and still ignored. The
    stack is computed for the panel because a reading that silently
    stops being collected is how a screen starts lying, and because
    dropping the arguments would break every caller for no gain. It
    gates nothing.

    WHAT THE TICK BAR IS, SINCE IT NOW DECIDES EVERYTHING: a bar
    that closes on a count of prints rather than a clock, whose
    direction is ONE VOTE PER PRINT -- how many prints moved price
    up against how many moved it down -- not where price happened
    to finish. It is also cut at every true 5-minute mark, so it
    carries no direction across a boundary.
    """
    if tick not in SIDES:
        return "", "the tick bar has no direction"
    return tick, ""


def holds(side: str, tick: str) -> tuple[bool, str]:
    """Does the trade still stand?

    ONLY THE TICK BAR FLIPPING ENDS IT.

    FLAT IS NOT A FLIP, and here that matters more than anywhere
    else it has come up: the tick bar is cut at every true 5-minute
    mark, so it has no direction for the first prints of every bar.
    An exit that closed on flat would fire twelve times an hour on
    the clock rather than on the market.

    The five-minute boundary DOES end a trade, but as a time stop
    applied by the caller -- not as a direction reading. The two are
    different events and conflating them is what made the earlier
    version close for an absence.
    """
    if side not in SIDES:
        return True, ""
    if tick == _other(side):
        return False, (f"the tick bar flipped {tick} against your "
                       f"{side} trade")
    return True, ""


def bar_of(ts: float, period_s: float = 300.0) -> float:
    """The TRUE wall-clock bar this instant falls in.

    Unix time divides evenly into 300-second buckets from an epoch
    on a minute boundary, so this lands on :00, :05, :10 of the hour
    in every zone whose offset is a multiple of five minutes --
    which is all of them.
    """
    try:
        t = float(ts or 0.0)
        p = float(period_s or 300.0)
    except (TypeError, ValueError):
        return 0.0
    if t <= 0 or p <= 0:
        return 0.0
    return float(int(t // p) * p)


def expired(bar_ts: float, now_ts: float,
            period_s: float = 300.0) -> tuple[bool, str]:
    """Has this trade outlived its 5-minute bar?

    Chris: "no trade will ever last past a 5 minute timeframe".

    A TIME STOP, NOT A DIRECTION READING, and the distinction is
    the whole reason this is a separate function. The tick bar is
    cut at the same boundary, so at that instant it also has no
    direction -- and an exit that read flat as a reversal would
    close for an absence, which is the churn this rule has been
    rewritten twice to remove. One of these is the clock running
    out; the other is the market turning. They get different
    reasons in the log because they are different events.

    A trade with no recorded bar is NOT expired. Zero is "we do not
    know", and closing every such position at the first boundary
    would be acting on missing data.
    """
    b = bar_of(bar_ts, period_s)
    n = bar_of(now_ts, period_s)
    if b <= 0 or n <= 0:
        return False, ""
    if n > b:
        mins = int(period_s // 60) or 5
        return True, (f"the {mins}-minute bar it was opened in has "
                      f"ended")
    return False, ""


def exit_reason(side: str, tick: str, bar_ts: float, now_ts: float,
                period_s: float = 300.0) -> str:
    """Why this trade should end now, or "" to hold it.

    ONE FUNCTION, IN ORDER, because the order carries meaning. The
    clock is checked FIRST so a trade that outlived its bar is
    logged as the bar ending rather than blamed on a tick bar that
    is merely between bars at that same instant -- the tick series
    is cut on the same boundary, so both are true and only one is
    the reason.

    WHY THIS IS NOT A FEW LINES IN THE ROUTE. It was, and a mutation
    that deleted the time stop's effect broke no test: the route
    checks lived in a closure reachable only through an HTTP
    request. That is how the auto gate and the one-per-candle cap
    were each removed without anything noticing. The policy is here
    and the route calls it.
    """
    if side not in SIDES:
        return ""
    gone, why_gone = expired(bar_ts, now_ts, period_s)
    if gone:
        return why_gone
    ok, why = holds(side, tick)
    return "" if ok else why


def state(win: dict | None, tick: str, rungs: dict | None,
          period_s: float, fp: str = "") -> dict:
    """Everything the rule and the panel need, from one place.

    One source, so the screen cannot show an agreement the rule did
    not see. Every panel in this engine that computed its own copy
    of a direction ended up telling a different story from the rule
    beside it -- the 5-minute cell and the OFI bar both did.
    """
    win = win or {}
    candle = str(win.get("mid_dir") or "")
    ofi = str(win.get("ofi_dir") or "")
    cvd = str(win.get("cvd_dir") or "")
    tick = str(tick or "")
    fp = str(fp or "")
    tf = traded_tf(period_s)
    below = BELOW_OF.get(tf, "")

    w_side, w_why = window_side(candle, ofi, cvd, fp)
    s_side, s_why = stack_side(rungs or {}, period_s)
    side, why = entry_side(tick)

    rows = {}
    for name in (tf, below):
        if not name:
            continue
        row = (rungs or {}).get(name) or {}
        rows[name] = {
            "present": bool(row.get("present")),
            "dir": row.get("qualified") or "",
            "absorbed": bool(row.get("absorbed")
                             or row.get("conflicted")),
            "seeded": row.get("seeded", True),
            "bps": row.get("bps"),
            "open_px": row.get("open_px"),
            "last_px": row.get("last_px"),
            "ticks": row.get("ticks"),
        }

    return {
        "tick": {
            "side": tick,
            "true": tick in SIDES,
            "why": "" if tick in SIDES else "no direction yet",
        },
        "stack": {
            "trading": tf, "below": below,
            "rows": rows,
            "side": s_side, "why": s_why,
            # SHOWN, NOT USED. The tick bar is the whole rule now.
            "asked": False,
            "decides": False,
        },
        # SHOWN, NOT USED. Under its own heading so there is no
        # chance of reading any of it as part of the decision.
        "context": {
            "period_s": float(period_s or 0.0),
            "candle": candle, "ofi": ofi, "cvd": cvd, "footprint": fp,
            "side": w_side, "why": w_why,
            "decides": False,
        },
        "side": side,
        "why": why,
        "say": (f"the tick bar is {side}" if side else why),
    }
