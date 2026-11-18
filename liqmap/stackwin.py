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
# How long a tick-bar direction must stand before it can open a
# trade. Chris: "a flip has to hold for 10 seconds before a trade is
# taken that keeps us from taking wrong direction trades and flips
# that lead to nothing".
#
# ENTRY ONLY. The exit stays immediate, and the asymmetry is the
# point: a flip gets you OUT at once and only lets you back IN once
# it has proved it meant something. Making the exit wait too would
# hold a losing position for ten seconds to protect against a flip
# that was right.
DWELL_S = 10.0


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


class Dwell:
    """How long the tick bar has pointed the way it currently does.

    FLAT DOES NOT RESET IT, and that is deliberate. The tick series
    is cut at every true 5-minute mark, so it has no direction for
    the first prints of every bar -- counting that as a change would
    restart the clock twelve times an hour and cost ten seconds of
    every bar for nothing. A gap is not a turn. Only the OPPOSITE
    direction restarts it.

    Monotonic time is not required: the caller passes the same clock
    it uses everywhere else, and a clock that jumps backwards makes
    `held_s` zero rather than negative.
    """

    __slots__ = ("_dir", "_since", "flips")

    def __init__(self) -> None:
        self._dir = ""
        self._since = 0.0
        self.flips = 0

    def on_tick(self, direction: str, now: float) -> None:
        d = direction if direction in SIDES else ""
        try:
            t = float(now or 0.0)
        except (TypeError, ValueError):
            return
        if not d or t <= 0:
            return              # a gap, not a turn
        if d != self._dir:
            if self._dir:
                self.flips += 1
            self._dir = d
            self._since = t

    @property
    def direction(self) -> str:
        return self._dir

    def held_s(self, now: float) -> float:
        try:
            t = float(now or 0.0)
        except (TypeError, ValueError):
            return 0.0
        if not self._dir or self._since <= 0 or t <= 0:
            return 0.0
        return max(0.0, t - self._since)

    def to_dict(self, now: float = 0.0) -> dict:
        return {"dir": self._dir, "held_s": round(self.held_s(now), 1),
                "flips": self.flips}


def hold_dir(now: str, prev: str = "") -> str:
    """The direction to REPORT: this one, or the last real one.

    Chris: "the gate 1 should never say flat at all as long as there
    is a 5000t candle it should show the trading direction of that
    candle", and "one always show what the real 15 minute is doing
    and the other always what the tick chart is doing".

    FLAT WAS NEVER AN ANSWER, IT WAS AN ABSENCE, and both gates were
    reporting it as though it were a reading. A candle whose close
    sits exactly on its open has not turned neutral -- it is one
    tick from wherever it just was, and the direction it had a
    moment ago is still the best answer to "which way is this
    going". Reporting flat there made the panel claim no direction
    while the chart plainly showed one, and an entry needing two
    gates to agree can never fire if either keeps blanking.

    So a side, once seen, is HELD until the other side replaces it.
    Nothing is invented: "" in and "" remembered is still "", which
    is the honest state before the first print of the session.

    This is not a smoother and it is not a threshold. It fills the
    gap at an exact tie and does nothing else -- a real flip passes
    through it untouched, which is what keeps the exit honest.
    """
    if now in SIDES:
        return now
    return prev if prev in SIDES else ""


def entry_side(tick: str, candle: str = "",
               held_s: float | None = None,
               min_hold_s: float = DWELL_S) -> tuple[str, str]:
    """The side to take, or the reason there is none.

    TWO DIRECTIONS IN, ONE SIDE OUT. Both are already resolved by
    the caller -- the tick candle's own direction and the traded
    timeframe candle's -- so this function cannot disagree with the
    panel about what either of them said. It used to take `rungs`
    and dig the candle out itself, which meant the screen and the
    rule each did their own extraction and could differ.

    BOTH MUST POINT THE SAME WAY. Chris: "one always show what the
    real 15 minute is doing and the other always what the tick chart
    is doing when they agree thats the direction we take".

    A FLIP STILL HAS TO HOLD. Chris: "a flip has to hold for 10
    seconds before a trade is taken ... flips that lead to nothing".
    `held_s` is how long the tick candle has pointed this way, from
    `Dwell`. None means the caller is not tracking it and the check
    is skipped -- a caller that cannot measure the dwell must not be
    silently blocked from ever trading.

    THE ORDER OF THE CHECKS IS THE ORDER OF THE REASONS, and it is
    chosen rather than incidental. The tick candle is settled first
    -- has it a direction, has it held -- and the timeframe candle
    is asked last. A rule that reports whichever test failed first
    is a DIFFERENT rule when the order changes: it fires on the same
    trades and explains them differently, and every wrong
    explanation in this engine has cost a day.

    EXITS DO NOT CONSULT THE CANDLE. `holds` is untouched: a trade
    still ends on the tick candle flipping, or on the caller's time
    stop. A candle that turns mid-trade is the pullback being traded
    into, and closing on it would be a second rule nobody asked for.
    """
    if tick not in SIDES:
        return "", "the tick candle has no direction yet"
    if held_s is not None and min_hold_s > 0 and held_s < min_hold_s:
        return "", (f"the tick candle turned {tick} {held_s:.0f}s ago "
                    f"-- it has to hold {min_hold_s:.0f}s")
    if candle not in SIDES:
        return "", "the timeframe candle has no direction yet"
    if candle != tick:
        return "", (f"the tick candle is {tick} but the timeframe "
                    f"candle is {candle}")
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
          period_s: float, fp: str = "",
          held_s: float | None = None, flips: int = 0,
          min_hold_s: float = DWELL_S,
          pace: dict | None = None,
          roll: dict | None = None,
          intent: dict | None = None,
          prev: dict | None = None) -> dict:
    """Everything the rule and the panel need, from one place.

    One source, so the screen cannot show an agreement the rule did
    not see. Every panel in this engine that computed its own copy
    of a direction ended up telling a different story from the rule
    beside it -- the 5-minute cell and the OFI bar both did.

    `tick` IS THE 5,000-TICK CANDLE'S OWN DIRECTION, close against
    open, exactly as the chart colours it. Chris: "as long as there
    is a 5000t candle it should show the trading direction of that
    candle".

    `prev` IS THE MEMORY AND THIS FUNCTION STAYS PURE. It is the
    `held` block from the previous call, and the one returned by
    this call is what the caller stores. Both gates hold their last
    real direction through an exact tie, and holding it means
    remembering it -- but a module that kept that memory inside
    itself would hand two callers one history, which is how the
    screen and the rule end up disagreeing. The caller owns the
    memory; the rule owns the policy.
    """
    win = win or {}
    w_candle = str(win.get("mid_dir") or "")
    ofi = str(win.get("ofi_dir") or "")
    cvd = str(win.get("cvd_dir") or "")
    tick_raw = str(tick or "")
    fp = str(fp or "")
    tf = traded_tf(period_s)
    below = BELOW_OF.get(tf, "")

    w_side, w_why = window_side(w_candle, ofi, cvd, fp)
    s_raw, s_why = stack_side(rungs or {}, period_s)

    # NEITHER GATE REPORTS FLAT ONCE IT HAS HAD A DIRECTION. The
    # raw readings are kept beside the held ones so the moment a
    # gate is being carried is visible rather than silent.
    prev = prev or {}
    tick = hold_dir(tick_raw, str(prev.get("tick") or ""))
    s_side = hold_dir(s_raw, str(prev.get("candle") or ""))

    side, why = entry_side(tick, s_side,
                           held_s=held_s, min_hold_s=min_hold_s)

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
            # WHAT THE 5,000-TICK CANDLE SAID THIS INSTANT, beside
            # the direction being reported. They differ only while
            # the candle sits exactly on its open, and showing both
            # is what keeps the hold from being invisible.
            "raw": tick_raw,
            "held_through_flat": bool(tick in SIDES
                                      and tick_raw not in SIDES),
            # How long it has pointed this way, and what it has to
            # reach. Shown so a blocked entry reads as waiting
            # rather than as nothing happening.
            "held_s": (round(held_s, 1) if held_s is not None else None),
            "min_hold_s": float(min_hold_s or 0.0),
            "ready": bool(tick in SIDES
                          and (held_s is None or min_hold_s <= 0
                               or held_s >= min_hold_s)),
            "flips": int(flips or 0),
            # THE WINDOW THE DIRECTION CAME OUT OF. `side` above IS
            # this window's reading -- the caller passes it in -- and
            # this is the count behind it, so the panel can show a
            # cold start as warming up rather than as a flat market.
            "roll": dict(roll or {}),
        },
        "stack": {
            "trading": tf, "below": below,
            "rows": rows,
            "side": s_side, "why": s_why,
            # As above: what the timeframe candle said this instant,
            # against what is being reported for it.
            "raw": s_raw,
            "held_through_flat": bool(s_side in SIDES
                                      and s_raw not in SIDES),
            # ASKED AND DECIDING. The traded timeframe's candle is a
            # gate on ENTRY -- both it and the tick bar have to point
            # the same way. The row below it is still read and still
            # drawn and still gates nothing.
            #
            # These two flags are what the panel reads to decide
            # whether to draw this block as a gate or as a reading,
            # so they are not decoration: a screen that labelled a
            # live gate "shown, not used" is the exact failure this
            # module was written after.
            "asked": True,
            "decides": True,
            "agrees": bool(s_side and tick in SIDES and s_side == tick),
        },
        # THE TICK CHART AGAINST THE CLOCK -- the push and the pull.
        #
        # Chris: "a real live continuous time chart paired with a
        # live continuous feed tick chart and traded the difference
        # in the 2 using the tick chart and the time chart as push
        # pull". A tick bar closes on a count, so the seconds it
        # takes ARE the tape's speed: the forming bar's projected
        # span against the median of recent closed ones is that
        # difference, already measured.
        #
        # SHOWN, NOT USED, and under its own key rather than inside
        # `tick` -- the tick block is the whole rule, and a field
        # that decides nothing sitting beside fields that decide
        # everything is how this engine has twice ended up trading
        # something nobody meant it to.
        "pace": {**(pace or {}), "decides": False},
        # IS THIS BAR CONTINUING OR PULLING BACK. Chris: "show tape
        # intent at that moment during the tick bar". Built in
        # `tapeintent` from the bar's own prices and the last fifth
        # of a bar's worth of prints.
        #
        # SHOWN, NOT USED, under its own key for the same reason the
        # pace is: a field that decides nothing sitting inside the
        # block that decides everything is how this engine has twice
        # traded something nobody meant it to.
        "intent": {**(intent or {}), "decides": False},
        # SHOWN, NOT USED. Under its own heading so there is no
        # chance of reading any of it as part of the decision.
        "context": {
            "period_s": float(period_s or 0.0),
            "candle": w_candle, "ofi": ofi, "cvd": cvd,
            "footprint": fp,
            "side": w_side, "why": w_why,
            "decides": False,
        },
        # THE MEMORY THE CALLER STORES AND HANDS BACK AS `prev`.
        # Returned rather than kept, so one module cannot serve two
        # callers one history.
        # The tape reading holds its own last state through a tie,
        # so its memory rides home with the gates' rather than in a
        # second place the caller has to remember to carry.
        "held": {"tick": tick, "candle": s_side,
                 "intent": {
                     "state": str((intent or {}).get("state") or ""),
                     "slice_dir": str((intent or {}).get("slice_dir")
                                      or "")}},
        "side": side,
        "why": why,
        "say": ((f"the tick candle and the {tf} candle are both {side}")
                if side else why),
    }
