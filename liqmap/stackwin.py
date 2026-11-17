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


def entry_side(tick: str, rungs: dict,
               period_s: float) -> tuple[str, str]:
    """The side to take, or the reason there is none.

    THE TICK BAR GATES AND THE STACK CONFIRMS.

    Chris asked for a hybrid of the tick bar and the 5-minute here,
    replacing the footprint. The 5-minute came out, and the reason
    is worth keeping: GATE 2 ALREADY CHECKS IT. On a 5-minute
    window the stack is 5m + 1m, so a 5m in gate 1 has to agree
    with itself -- free, and the rule takes exactly the same trades
    as the tick bar alone. Same trade set, one more clause to keep
    correct.

    So gate 1 contributes what nothing else in this rule has:
    PRINTS RATHER THAN TIME. The tick bar closes on a count of
    trades, so it moves fast when the tape is busy and slowly when
    it is not, and its direction is one vote per print -- not where
    price happened to finish. The stack is three readings of the
    clock. They are measuring genuinely different things, which is
    the only reason pairing them says anything.

    THE FOOTPRINT IS STILL READ AND STILL SHOWN. It decides nothing
    now. It stays on the panel for the same reason the candle, OFI
    and CVD do: a leg that silently stops being collected is how a
    screen starts lying, and he has already watched four readings
    agree and drive nothing.

    THE TICK BAR IS ASKED FIRST AND ALONE. When it has no direction
    the stack is never consulted and the reason does not mention it.
    """
    if tick not in SIDES:
        return "", "the tick bar has no direction"

    s_side, s_why = stack_side(rungs, period_s)
    if not s_side:
        return "", (f"the tick bar is {tick} but the stack does not "
                    f"back it: {s_why}")
    if s_side != tick:
        return "", (f"the tick bar is {tick} and the "
                    f"{traded_tf(period_s)} is {s_side} -- they "
                    f"point apart")
    return tick, ""


def holds(side: str, rungs: dict,
          period_s: float) -> tuple[bool, str]:
    """Does the trade still stand?

    ONLY A FLIP OF THE TRADED TIMEFRAME ENDS IT. Chris: "a
    disagreement is only when the primary timeframe that im trading
    on flips".

    THE TICK BAR IS NOT CONSULTED HERE AT ALL, and that is the
    point. It gets you in; the traded timeframe keeps you in. It
    closes on a count of prints, so between bars it has no
    direction -- and with the 5-minute reset forced onto that
    series, an exit that read it would fire on the clock at every
    boundary rather than on the market. That was the live risk in
    the previous version.

    FLAT IS NOT A FLIP. A 5-minute candle sitting on its open has
    gone nowhere; it has not turned. Closing on that would end
    trades for an absence, which is the churn this rule has been
    rewritten twice to remove. Only the OPPOSITE side ends it.
    """
    if side not in SIDES:
        return True, ""
    tf = traded_tf(period_s)
    now, why = stack_side(rungs, period_s)
    if now == _other(side):
        return False, (f"the {tf} flipped {now} against your "
                       f"{side} trade")
    return True, ""


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
    side, why = entry_side(tick, rungs or {}, period_s)

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
            # Never consulted while the tick bar has no direction,
            # and the panel says that rather than showing a verdict
            # the rule did not ask for.
            "asked": tick in SIDES,
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
        "say": (f"the tick bar and the {tf} are both {side}"
                if side else why),
    }
