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
# THE BAR THE TRADE CAP COUNTS AGAINST. Chris: "we need to set a
# rule for no more then 2 trades on a 5 minute timeframe".
#
# FIVE MINUTES WHATEVER THE WINDOW IS SET TO. He said five, and the
# window selector can read 15m -- counting against the traded row
# would quietly make the cap three times looser than he asked for.
#
# This is a counter, not a time stop. Nothing expires because of
# it; the only question it answers is how many trades this
# five minutes has already had.
CAP_BAR_S = 300.0
MAX_TRADES_PER_BAR = 2
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
               intent: dict | None = None) -> tuple[str, str]:
    """The side to take, or the reason there is none.

    TWO READINGS, BOTH POINTING THE SAME WAY. Chris, restating the
    rule he wants back: "Gate 1 -- the 5000t candle's direction ...
    Gate 2 -- the real live timeframe candle ... Both gates point
    the same way. That side is the trade."

    Both are already resolved by the caller and both are held
    through an exact tie, so this cannot disagree with the panel
    about what either of them said.

    `intent` IS ACCEPTED AND DOES NOT DECIDE. It was allowed to
    turn a trade round for one revision -- a volume-backed pullback
    traded in the retrace's own direction. Measured against the
    timeframe gate, that fired on 3% of entries in chop and none at
    all in a trend, because a short inside an up tick bar needs the
    timeframe down as well. A rule that fires three times in a
    hundred is not a rule, it is a special case, and it is better
    off as the reading it started as. It stays in the signature so
    that `state` can pass it without the caller having to know, and
    a test holds it to changing nothing.

    WHAT IS NOT HERE:

        THE TEN-SECOND HOLD. "lets add back the timeframe section
        without the 10 second hold".

        ANY CLOCK. No time stop, no per-bar cap. Exits are the tick
        bar turning over, or his own stop and target.
    """
    if tick not in SIDES:
        return "", "the tick candle has no direction yet"
    if candle not in SIDES:
        return "", "the timeframe candle has no direction yet"
    if candle != tick:
        return "", (f"the tick candle is {tick} but the timeframe "
                    f"candle is {candle}")
    return tick, ""


def holds(side: str, tick: str, candle: str = "") -> tuple[bool, str]:
    """Does the trade still stand?

    TWO WAYS OUT NOW, AND THE TIMEFRAME ONE IS NEW. Chris: "the
    timeframe is also going to be a exit rule disagree means exit
    because the strategy isnt accounting for the tick charts pull
    backs".

    THE TICK BAR TURNING OVER still ends it -- that is what gets
    you out of a move that has actually reversed.

    THE TIMEFRAME CANDLE TURNING AGAINST YOU now ends it too. It
    gated the entry and it keeps gating while the trade is open, so
    a position cannot survive the condition that opened it going
    away. This reverses "no longer use time as a entry point or
    exit at all" from one revision ago, by name and on purpose.

    FLAT IS NOT A DISAGREEMENT, on either of them. Both directions
    arrive here HELD, so neither blanks in normal running -- but a
    caller that passes a raw reading must not have an absence read
    as a reversal. Exits firing on absences is the churn this rule
    has been rewritten three times to remove, and it is the single
    most expensive mistake in this file's history.

    THE ORDER IS THE ORDER OF THE REASONS. The tick bar is asked
    first because it is the faster of the two and the one that
    usually fires; a log that blamed the timeframe for a tick-bar
    exit would send you looking at the wrong chart.

    IT MUST BE THE SAME NUMBERS THE ENTRY READ. This once took the
    tick bar's VOTE while the entry took the candle's body, and the
    two can point opposite ways on one bar -- the engine opened a
    long and closed it on the same poll.
    """
    if side not in SIDES:
        return True, ""
    other = _other(side)
    if tick == other:
        return False, (f"the tick bar flipped {tick} against your "
                       f"{side} trade")
    if candle == other:
        return False, (f"the timeframe candle turned {candle} against "
                       f"your {side} trade")
    return True, ""


def bar_of(ts: float, period_s: float = CAP_BAR_S) -> float:
    """The TRUE wall-clock bar this instant falls in.

    Unix time divides evenly into 300-second buckets from an epoch
    on a minute boundary, so this lands on :00, :05, :10 of the hour
    in every zone whose offset is a multiple of five minutes --
    which is all of them.

    THIS IS A COUNTER, NOT A CLOCK ON THE TRADE. It stamps which
    five minutes a trade was opened in so the cap can count them.
    Nothing expires because of it: the time stop was removed and
    stays removed. Chris asked for "no more then 2 trades on a 5
    minute timeframe", which needs a bar to count against and
    nothing more.
    """
    try:
        t = float(ts or 0.0)
        p = float(period_s or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if t <= 0 or p <= 0:
        return 0.0
    return float(int(t // p) * int(p))


def _say(side: str, tf: str, intent: dict) -> str:
    """Why the side is the side, in one line."""
    base = f"the tick candle and the {tf or 'timeframe'} candle are "
    if intent.get("state") == "pullback":
        word = "heavy" if intent.get("heavy") else "light"
        return (base + f"both {side} -- pulling back, {word} "
                "(shown, it does not decide)")
    return base + f"both {side} and extending"


def state(win: dict | None, tick: str, rungs: dict | None,
          period_s: float, fp: str = "",
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

    `intent` IS THE OTHER HALF OF THE RULE -- what the tape is
    doing inside this bar and whether volume is paying for it. The
    side comes from those two and nothing else.

    THE TIMEFRAME CANDLE AND THE WINDOW ARE STILL BUILT HERE and
    they decide nothing. Kept because a reading that silently stops
    being collected is how a screen starts lying, and because he
    has asked for each of them on screen at some point.

    `prev` IS THE MEMORY AND THIS FUNCTION STAYS PURE. It is the
    `held` block from the previous call, and the one returned by
    this call is what the caller stores. The tick direction holds
    through an exact tie, and holding it means remembering it --
    but a module that kept that memory inside itself would hand two
    callers one history, which is how the screen and the rule end
    up disagreeing. The caller owns the memory; the rule owns the
    policy.
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

    side, why = entry_side(tick, s_side, intent or {})

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
            # THE VOTE BEHIND THE CANDLE, as information. It was
            # the direction once, behind a band and a floor that
            # both produce flat; it is drawn beside the body and it
            # decides nothing.
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
            # ASKED AND DECIDING ON ENTRY. Chris: "lets add back
            # the timeframe section without the 10 second hold".
            # It does NOT decide exits -- it gets you in, the tick
            # bar keeps you in.
            #
            # These two flags are what the panel reads to decide
            # whether to draw this block as a gate or as a reading,
            # so they are not decoration: a screen that labelled a
            # live gate "shown, not used" is the exact failure this
            # module was written after, and the reverse is just as
            # bad.
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
                 # `worst` and `bar_dir` ride home too: the first is
                 # how deep this pullback has been, which one instant
                 # cannot know, and the second is what tells the next
                 # reading the geometry flipped and that number is
                 # now meaningless.
                 # EVERY FIELD `tapeintent.read` NEEDS BACK. It is
                 # a pure function with the memory held out here, so
                 # a field dropped from this dict does not error --
                 # it silently resets that part of the reading on
                 # every poll. `heavy` is the hysteresis, `v_ext`
                 # is where the push ended, `bar_ts` is how it knows
                 # the bar changed; without them the flag chatters
                 # and the volume mark is re-taken continuously.
                 "intent": {
                     "state": str((intent or {}).get("state") or ""),
                     "worst": float((intent or {}).get("worst") or 0.0),
                     "bar_dir": str((intent or {}).get("bar_dir") or ""),
                     "heavy": bool((intent or {}).get("heavy")),
                     "v_ext": float((intent or {}).get("v_ext") or 0.0),
                     "bar_ts": float((intent or {}).get("bar_ts")
                                     or 0.0)}},
        "side": side,
        "why": why,
        "say": (_say(side, tf, intent or {}) if side else why),
    }
