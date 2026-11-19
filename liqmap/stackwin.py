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
# HOW LONG GATE 2'S DIRECTION MUST STAND BEFORE IT CAN OPEN.
#
# Chris, restoring the rule: "The 5-second hold on that direction".
# It was ten seconds, then it came out entirely, and it is back at
# five -- on the TICK chart, which is gate 2 now.
#
# IT IS THE ONLY THING DEBOUNCING GATE 2. Its direction is held
# through an exact tie and nothing wider, so a bar one tick off its
# open reads a direction; the hold is what stops that becoming a
# trade. That was `MIN_TURN_TICKS`' job and he has asked for the
# hold instead.
DWELL_S = 5.0

# NO TRADE OUTLIVES THIS. "Time stop -- the trade has outlived the
# 5-minute bar it opened in. Checked before direction, so the log
# says the bar ended rather than blaming the tick bar."
TRADE_BAR_S = 300.0

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
    """Which row a window of this length is trading.

    A NEGATIVE WINDOW IS A TICK CHART. Chris: "what if we traded the
    tick chart instead of the time chart ... a tick chart is really
    no different then a time chart only difference is a tick chart
    prints every other 5000ts".

    `window_s` is a REAL column that was clamped to 300 or 900 on
    every save, so a tick size is carried negative -- a window of
    seconds is never below zero, and the clamp cannot destroy it.
    """
    try:
        p = float(period_s or 0.0)
    except (TypeError, ValueError):
        return ""
    if p < 0:
        return f"{int(round(-p))}t"
    if p <= 0.0:
        # NOT A ONE-MINUTE WINDOW. Zero is "no window given", and
        # adding the 1m choice turned every junk value into a 1m
        # row -- the shortest, busiest chart there is, reached by
        # accident.
        return ""
    if p <= 60.0:
        return "1m"
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


# ONE TICK MUST NOT BE ABLE TO TURN THE GATE OVER.
#
# Chris: "every trade that im loosing is because of flips ... it
# gets in when both agree then immediately exits for a loss of
# several ticks ... a lot of the disagrees that happen price
# actually travels that direction it just exits on its own with no
# visable disagreement at all".
#
# MEASURED, AND IT IS A BAR-BIRTH PROBLEM. Gate 1 is the 5,000-tick
# candle's body, close against open. A bar that has just opened has
# close EQUAL to open, so the very next print decides its
# direction -- one tick, after which the gate has turned and the
# trade is closed.
#
# Counted over a simulated hour: the first fifty prints are 1% of a
# bar's life and produced 9% of all direction changes on a choppy
# tape, 29% trending and 63% in a strong trend -- roughly 170 times
# the per-print rate of the rest of the bar. And it is INVISIBLE,
# because the flip is over in a handful of prints and the panel has
# repainted by the time anyone looks, which is exactly the "no
# visable disagreement" he is describing.
#
# So the bar has to have gone somewhere before its body is allowed
# to overrule the direction already standing. This is a distance,
# not a clock and not a count -- he removed both of those by name.
# Under it the previous direction simply stands, so the gate never
# reads flat, which he has insisted on three times.
#
# FOUR TICKS IS ONE FULL POINT ON ES and is the knee of the
# measured curve: direction changes an hour fall 359 -> 86 on a
# choppy tape, 108 -> 25 trending, 44 -> 12 in a strong trend. It
# is one line to change and the panel shows the number.
TICK = 0.25
MIN_TURN_TICKS = 4.0

# NO TRADE WHILE THE TIMEFRAME IS SITTING ON ITS OPEN.
#
# Chris: "no trades are ever taken at the opening price of whatever
# timeframe i select on my auto trader".
#
# A candle whose close is a tick from its open has not said
# anything yet -- its direction is one print old and will be the
# other way as often as not. This is the same objection as the
# newborn tick bar, one timeframe up.
MIN_OPEN_TICKS = 4.0

# WHEN A PULLBACK HAS STOPPED BEING A PULLBACK.
#
# Chris: "tick chart pushing opposite direction of time is a
# pullback and vic versa a continuation when they both agree on
# direction" -- and, asked what should then close a trade,
# "Timeframe, or a pullback that fails".
#
# Measured against the TIMEFRAME's own move, not the tick bar's: a
# retrace that has given back this much of what the candle made is
# no longer a pause in the move, whatever the tick chart is doing.
# Volume failing it is the other half, and that test lives in
# `tapeintent` where the effort comparison already is.
PULLBACK_FAILS_AT = 0.6


def body_dir(open_px: float, close_px: float, prev: str = "",
             min_ticks: float = MIN_TURN_TICKS) -> str:
    """Gate 1: which way this candle has gone, once it has gone far.

    `prev` is the direction already standing. Inside the threshold
    it is returned unchanged -- NOT flat. An absence is not a
    reading, and reporting one here is the fault this gate has
    already been rebuilt twice to remove.

    This replaces a bare `hold_dir(close > open)`, which held the
    direction only through an EXACT tie -- a single print's worth of
    protection, at the one moment in a bar's life when a single
    print is all it takes.
    """
    try:
        o, c = float(open_px or 0.0), float(close_px or 0.0)
        m = max(0.0, float(min_ticks))
    except (TypeError, ValueError):
        return prev if prev in SIDES else ""
    if o <= 0 or c <= 0:
        return prev if prev in SIDES else ""
    moved = (c - o) / TICK
    if moved >= m and moved > 0:
        return "up"
    if moved <= -m and moved < 0:
        return "down"
    return prev if prev in SIDES else ""


def turn_ticks(open_px: float, close_px: float) -> float:
    """How far this candle sits from its own open, in ticks.

    Shown beside the gate so a direction being HELD for want of
    movement is visible rather than silent.
    """
    try:
        o, c = float(open_px or 0.0), float(close_px or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if o <= 0 or c <= 0:
        return 0.0
    return round((c - o) / TICK, 1)


class Dwell:
    """How long gate 2 has pointed the way it currently does.

    FLAT DOES NOT RESET IT. A gap is not a turn: only the OPPOSITE
    direction restarts the clock. Counting an absence as a change
    is what kept the old ten-second hold from ever elapsing.

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
        if not self._dir or self._since <= 0 or t < self._since:
            return 0.0
        return t - self._since


def expired(bar_ts: float, now_ts: float,
            period_s: float = TRADE_BAR_S) -> tuple[bool, str]:
    """Has this trade outlived its five-minute bar?

    A TIME STOP, NOT A DIRECTION READING, and that is why it is its
    own function. The tick series can be flat at the same instant a
    bar ends, and an exit that read flat as a reversal would close
    for an absence. One of these is the clock running out; the other
    is the market turning. They get different reasons in the log
    because they are different events.

    A trade with no recorded bar is NOT expired: zero is "we do not
    know", and closing every such position at the first boundary
    would be acting on missing data.
    """
    b = bar_of(bar_ts, period_s)
    n = bar_of(now_ts, period_s)
    if b <= 0 or n <= 0:
        return False, ""
    if n > b:
        return True, (f"the {int(period_s // 60)}-minute bar this "
                      "trade opened in has ended")
    return False, ""


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


def regime(tick: str, candle: str, tick_failed: bool = False) -> str:
    """Where this move is: "continuation", "pullback", or "".

    Chris: "using the tick chart to confirm what the timeframe is
    actually doing tick chart pushing opposite direction of time is
    a pullback and vic versa a continuation when they both agree on
    direction".

    A DIRECTION IS NOT A PUSH, and that distinction cost him trades.
    Gate 1 is the tick candle's BODY -- close against open, and held
    through small moves so it does not flip on noise. A bar that ran
    twenty ticks up and gave back eighteen still has a positive body
    and still reads UP. It is not pushing up. It is rolling over.

    From his own screen, with the trade taken on it:

        tick bar UP, +2 ticks from this candle's open
        tape PULLBACK -- 90% given back, HEAVY, 1.71x the effort
        THE TWO CHARTS TOGETHER: CONTINUATION

    The reading that said the push was dead was on the panel, two
    lines above the one that called it a continuation, and it was
    marked as deciding nothing. Chris: "its taking trades on data
    that has already happened". It was -- `tick` is a summary of
    where the bar has BEEN.

    `tick_failed` is that same bar's own retrace, judged by
    `pullback_failed`. When the tick chart's push has gone, the two
    charts agreeing is not a continuation -- it is the start of a
    pullback, and the rule now says so.
    """
    if tick not in SIDES or candle not in SIDES:
        return ""
    if tick != candle:
        return "pullback"
    return "pullback" if tick_failed else "continuation"


def pullback_failed(depth: float, heavy: bool = False,
                    fails_at: float = PULLBACK_FAILS_AT) -> bool:
    """Has this retrace stopped being a pause and become a turn?

    Two ways, and either is enough:

        DEPTH -- it has given back `fails_at` of what the timeframe
        candle made. Past that the move it was pausing in is mostly
        gone.

        VOLUME -- it is being paid for. `heavy` comes from
        `tapeintent`, which compares the size behind the retrace
        against the size behind the push, per unit of price moved,
        and already refuses to call a trivial retrace heavy.

    A shallow retrace nobody is leaning on is neither, and that is
    the case this whole revision exists to hold through.
    """
    try:
        d = float(depth or 0.0)
        f = float(fails_at)
    except (TypeError, ValueError):
        return False
    return bool(d >= f or heavy)


def entry_side(tick: str, candle: str = "",
               tf_ticks: float | None = None,
               min_open_ticks: float = MIN_OPEN_TICKS,
               intent: dict | None = None,
               tick_failed: bool = False,
               episode_taken: bool = False,
               held_s: float | None = None,
               min_hold_s: float = DWELL_S) -> tuple[str, str]:
    """The side to take, or the reason there is none.

    THE ORDER IS CHRIS'S, and the order is the rule:

        3. GATE 2 -- the tick candle's direction, close against
           open, held through the instant close sits on its open so
           it never blanks.
        4. THE FIVE-SECOND HOLD on that direction.
        5. GATE 1 -- the real live timeframe candle, close against
           open, no threshold, one tick decides it, also held
           through ties.
        6. Both point the same way. That side is the trade.

    "Nothing else votes. The footprint, OFI, CVD, the window's four
    legs, the 1m row, the vote, the pace, the pullback reading --
    all computed, all drawn, none of them gate anything."

    `intent` and `tick_failed` ARE ACCEPTED AND IGNORED. The
    pullback reading blocked entries for one revision and he has
    taken that back by name. They stay in the signature so callers
    need not know, and a test holds them to changing nothing.

    `episode_taken` and the open-distance check are CAPS, not votes
    -- one trade per agreement, and nothing opened on top of the
    timeframe's open. Both were asked for by name and neither is a
    reading.
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
    if episode_taken:
        return "", ("this agreement has already been traded -- they "
                    "have to disagree before another is taken")
    if tf_ticks is not None and abs(float(tf_ticks)) < float(
            min_open_ticks or 0.0):
        return "", (f"the timeframe candle is {abs(tf_ticks):.0f} "
                    f"ticks off its open -- nothing is taken within "
                    f"{min_open_ticks:.0f}")
    return tick, ""


def holds(side: str, tick: str) -> tuple[bool, str]:
    """Does the trade still stand, as far as DIRECTION goes?

    ONLY THE TICK CANDLE FLIPPING ENDS IT. Chris: "4. Tick bar flip
    against the position." The timeframe closed trades for one
    revision and does not any more -- it gets you in, gate 2 takes
    you out.

    FLAT IS NOT A FLIP. The direction arrives held, so it does not
    blank in normal running, but an absence must never read as a
    reversal. That is the most expensive mistake in this file's
    history and it has been made three times.
    """
    if side not in SIDES:
        return True, ""
    if tick == _other(side):
        return False, (f"the tick bar flipped {tick} against your "
                       f"{side} trade")
    return True, ""


def exit_reason(side: str, tick: str, bar_ts: float, now_ts: float,
                period_s: float = TRADE_BAR_S) -> str:
    """Why this trade should end now, or "" to hold it.

    ONE FUNCTION, IN ORDER, because the order carries meaning.
    Chris: "Time stop ... Checked before direction, so the log says
    the bar ended rather than blaming the tick bar." Both can be
    true at the same instant and only one of them is the reason.

    WHY THIS IS NOT A FEW LINES IN THE ROUTE. It was, and a
    mutation that deleted the time stop's effect broke no test: the
    route checks lived in a closure reachable only through an HTTP
    request. That is how the auto gate and the one-per-candle cap
    were each removed without anything noticing.
    """
    if side not in SIDES:
        return ""
    gone, why_gone = expired(bar_ts, now_ts, period_s)
    if gone:
        return why_gone
    ok, why = holds(side, tick)
    return "" if ok else why


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


def _regime_say(where: str, candle: str, tick: str, depth: float,
                heavy: bool, failed: bool,
                tick_failed: bool = False) -> str:
    """What the two charts say about each other, in one line."""
    if not where:
        return "one of the two has no direction yet"
    if tick_failed and tick == candle:
        return (f"both read {tick}, but the tick chart's own push has "
                "gone -- it is rolling over, not continuing")
    if where == "continuation":
        return (f"the tick chart is pushing {tick} with a {candle} "
                "timeframe -- a continuation")
    pct = int(round(depth * 100))
    base = (f"the tick chart is pushing {tick} against a {candle} "
            f"timeframe -- a pullback, {pct}% of the move given back")
    if failed:
        why = "being paid for" if heavy else "too deep"
        return base + f" and FAILING ({why})"
    return base + ", holding through it"


def _say(side: str, tf: str, intent: dict) -> str:
    """Why the side is the side, in one line."""
    base = f"the tick candle and the {tf or 'timeframe'} candle are "
    return base + f"both {side} -- a continuation"


def state(win: dict | None, tick: str, rungs: dict | None,
          period_s: float, fp: str = "",
          tick_open: float = 0.0, tick_close: float = 0.0,
          pace: dict | None = None,
          roll: dict | None = None,
          intent: dict | None = None,
          tf_intent: dict | None = None,
          tf_ticks: float | None = None,
          episode_taken: bool = False,
          held_s: float | None = None,
          min_hold_s: float = DWELL_S,
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

    # NEITHER GATE REPORTS FLAT ONCE IT HAD A DIRECTION. The raw
    # readings are kept beside the held ones so the moment a gate
    # is being carried is visible rather than silent.
    prev = prev or {}
    prev_tick = str(prev.get("tick") or "")
    # GATE 1 NEEDS THE BAR TO HAVE GONE SOMEWHERE. Given the
    # candle's prices, the direction only turns once it is
    # `MIN_TURN_TICKS` from its own open -- a newborn bar's body is
    # otherwise decided by one print, which is what was closing
    # trades a few ticks after opening them.
    #
    # Without the prices there is no distance to measure, so the
    # old behaviour stands: hold through an exact tie and nothing
    # more. A caller that cannot supply them is not silently given
    # a rule it is not applying.
    # GATE 2 IS HELD THROUGH AN EXACT TIE AND NOTHING WIDER.
    # Chris: "Held through the instant close sits on open, so it
    # never blanks." The four-tick minimum that used to sit here is
    # gone by his instruction -- the five-second hold below is what
    # debounces this gate now. `turn_ticks` stays because the
    # distance is still worth seeing.
    moved = turn_ticks(tick_open, tick_close)
    tick = hold_dir(tick_raw, prev_tick)
    s_side = hold_dir(s_raw, str(prev.get("candle") or ""))

    # WHERE THIS MOVE IS, from the two gates together. Agreement
    # is a continuation; the tick chart pushing the other way is a
    # pullback, not a failure.
    tf_intent = tf_intent or {}
    # THE TICK BAR'S OWN PUSH, AND WHETHER IT IS STILL ALIVE. A
    # positive body is not a push: a bar that ran twenty ticks and
    # gave back eighteen reads UP and is rolling over.
    _ti = intent or {}
    tick_failed = bool(
        _ti.get("state") == "pullback"
        and pullback_failed(float(_ti.get("depth") or 0.0),
                            bool(_ti.get("heavy"))))
    where = regime(tick, s_side, tick_failed)
    tf_depth = float(tf_intent.get("depth") or 0.0)
    tf_heavy = bool(tf_intent.get("heavy"))
    failed = bool(where == "pullback"
                  and pullback_failed(tf_depth, tf_heavy))

    # ONE TRADE PER AGREEMENT. The run of agreement is identified
    # by the side the two charts are agreeing on; it ends the moment
    # that is no longer a continuation, whatever broke it.
    prev_ep = prev.get("episode") or {}
    agreeing = (where == "continuation" and tick in SIDES)
    same_run = bool(agreeing and prev_ep.get("side") == tick)
    spent = bool(episode_taken or (same_run and prev_ep.get("taken")))

    side, why = entry_side(tick, s_side, tf_ticks,
                           intent=intent or {},
                           tick_failed=tick_failed,
                           episode_taken=spent,
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
            # HOW FAR THIS CANDLE IS FROM ITS OWN OPEN, and what it
            # has to reach before it may turn the gate over. Drawn,
            # because a direction being HELD for want of movement
            # was invisible -- the flip was over in a handful of
            # prints and the panel had repainted by the time anyone
            # looked. Chris: "it just exits on its own with no
            # visable disagreement at all".
            "moved_ticks": moved,
            # HOW LONG IT HAS POINTED THIS WAY, and what it has to
            # reach. Shown so a blocked entry reads as WAITING
            # rather than as nothing happening -- the two call for
            # very different reactions.
            "held_s": (round(held_s, 1) if held_s is not None else None),
            "min_hold_s": float(min_hold_s or 0.0),
            "ready": bool(tick in SIDES
                          and (held_s is None or min_hold_s <= 0
                               or held_s >= min_hold_s)),
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
        # THE TICK BAR'S OWN RETRACE. It DECIDES now: a tick chart
        # whose push has gone is not confirming anything, whatever
        # its body still reads. It sat on the panel marked "shown
        # only" while the rule traded against it.
        "intent": {**(intent or {}), "decides": True},
        # WHERE THE MOVE IS, AND WHETHER THE PULLBACK HAS FAILED.
        # This block DOES decide: it is what holds a trade through
        # a retrace and what closes one when the retrace stops
        # being a retrace.
        "regime": {
            "where": where,
            "depth": round(tf_depth, 3),
            "heavy": tf_heavy,
            "failed": failed,
            "fails_at": float(PULLBACK_FAILS_AT),
            "tf_ticks": tf_ticks,
            "min_open_ticks": float(MIN_OPEN_TICKS),
            "decides": True,
            "tick_failed": tick_failed,
            "tick_depth": round(float(_ti.get("depth") or 0.0), 3),
            "tick_heavy": bool(_ti.get("heavy")),
            "say": _regime_say(where, s_side, tick, tf_depth,
                               tf_heavy, failed, tick_failed),
        },
        # SHOWN, NOT USED. Under its own heading so there is no
        # chance of reading any of it as part of the decision.
        "context": {
            "period_s": float(period_s or 0.0),
            "candle": w_candle, "ofi": ofi, "cvd": cvd,
            "footprint": fp,
            "side": w_side, "why": w_why,
            "decides": False,
        },
        # THE RUN OF AGREEMENT, and whether it has been spent.
        # `taken` is set by the CALLER when a trade actually opens,
        # because only the caller knows whether the journal accepted
        # the entry -- the cap or AUTO being off can refuse one the
        # rule offered, and an agreement that never produced a trade
        # must not be marked as having produced one.
        "episode": {
            "side": tick if agreeing else "",
            "agreeing": agreeing,
            "same_run": same_run,
            "taken": spent,
            "decides": True,
        },
        # THE MEMORY THE CALLER STORES AND HANDS BACK AS `prev`.
        # Returned rather than kept, so one module cannot serve two
        # callers one history.
        # The tape reading holds its own last state through a tie,
        # so its memory rides home with the gates' rather than in a
        # second place the caller has to remember to carry.
        "held": {"tick": tick, "candle": s_side,
                 "episode": {"side": tick if agreeing else "",
                             "taken": spent},
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
