"""What the tick chart is doing INSIDE the fused bar.

Chris: "on my chart there is a hybrid that was created its a
10000tick x 4000 contract OFI and it works its a tick chart inside
a OFI candle what i want to see is what the tick chart is doing
inside the OFI, like moving down together, moving up together ofi
remaining down but tick is bouncing from bottom or top or read it
like normal candle and OFI trying to go up or down".

A fused bar carries two readings of the same stretch of tape and
they do not have to agree:

    THE BOOK -- the signed OFI accumulator since the bar opened,
    and how far it has pushed toward its target. This is the bar's
    direction and its colour; `fusedbar` already says so.

    THE PRICE -- open, high, low and close of the prints that
    arrived in the same window. Its own direction, and crucially
    WHERE IN ITS OWN RANGE it currently sits.

This module answers one question off those two: are they going the
same way, and if they are, is price still at the business end of
the move or has it come back off it. That is the difference
between "moving down together" and "OFI remaining down but tick is
bouncing from bottom", which are the two states he named and which
look identical if you only read directions.

IT READS THE CANDLE THE CHART DRAWS. Chris, on the first version:
"its goal is to read the chart the way it is if you look at the
chart there is a outside OFI candle and a inside tick candle it
should read what that is doing".

    THE OUTSIDE IS THE OFI CANDLE -- the high-low range, coloured
    by the sign of the book's accumulator, washed gray when the
    book and price disagree. That is literally what the renderer
    does: `b.dir` picks the colour and `b.tradeable` picks the
    wash.

    THE INSIDE IS THE TICK CANDLE -- the body, open to close, of
    the prints that arrived in the same window.

PRICE IS THE ROLLING DIRECTION, NOT THE BODY. Chris, on the swing
pair: "the tick candle (price) that sits inside of the OFI candle
(book) isnt really saying whats going on at all until tick candle
completely flips the other direction im trying to see that
upwards move the whole time its happening".

    He is right, and it gets worse the longer the bar. `close vs
    open` only changes when price crosses back through ONE old
    number. On a 10,000-print bar that number is ninety seconds
    old; on a 100,000-print bar it is a quarter of an hour old and
    possibly twenty points away. A bar that opened high, sold off
    and is now ripping back up reads DOWN the whole way up.

    So the price leg is `RollingDir` -- which way the last N
    prints went, recomputed on every print, never reset at a
    close. The body is still carried and still drawn, because it
    is the candle's shape and the pullback is measured off it, but
    it no longer decides anything.

So the pullback here is read the way anyone reads a candle: an up
candle with a long upper wick has come back off the top. It is the
distance from the close to the extreme the body ran to, over the
WHOLE range of the outer candle.

WHY NOT `tapeintent.depth`, WHICH THIS USED TO USE. That measures
against the move from the OPEN, and framed on the book's
direction. On a bar where the book was down 6.5 points while price
ran 23.5 up, it divided by the 6.5 and saturated: the panel told
Chris "100% of the move given back" about a bar quietly fading off
its high. The denominator has to be the candle's own range, which
cannot be small relative to its own wick.

BOUNCE_AT IS A LABEL BOUNDARY, NOT A FINDING. It decides which
word the panel prints and nothing else; the raw percentage is
drawn beside it either way, so nothing is hidden behind the
threshold and no trade turns on it.

THE RULE, IN ONE PLACE. In when the outer candle and the inner
body point the same way and have done for five seconds, on a bar
that has settled. Out on the take profit, or on the two coming
apart. Nothing else opens a trade and nothing else closes one.

IT DECIDES NOW, AND IT DID NOT BEFORE. It shipped as a panel --
"Agreeing, bounce shown only" -- he watched it, and then wrote the
rules off what he saw: "the auto trader will then trade the
direction that is set for example: ( book down or up, price down
or up and says moving together) and exit the trade on bounces
never trade when book and price are apart or nothing settled".

So `decides` is True, the panel no longer says SHOWN ONLY, and the
tests that enforced the old promise have been replaced rather than
deleted -- what they guarded was a claim on the screen, and the
screen now makes the opposite claim.

SETTLING, AND WHY IT IS NOT A DETAIL.

    Chris: "everytime a new tick ofi candle prints instead allowing
    it to bounce up and down or not sure of direction let it settle
    and then pick true direction."

    A fused bar's first prints have almost no range and an
    accumulator near zero, so both legs flicker. Reading a
    direction there is reading noise, and a rule that enters on it
    enters at random. Below `MIN_SETTLE` of BOTH legs this reports
    "settling" and offers no direction at all -- the same 10% the
    fused strategy's `live_signal` already waits for, so the two
    warm-ups cannot disagree.

    Once settled, the direction is HELD through a tie, exactly as
    the tick gates hold theirs. A bar that ticks back to flat for
    an instant has not changed its mind.

    THE MEMORY DIES AT THE BAR BOUNDARY. Holding a direction into a
    new bar would defeat the settling it was added for, so `prev`
    carries the bar it belongs to and is dropped when that changes.
"""

from __future__ import annotations

from .stackwin import Dwell, hold_dir  # noqa: F401
from .tapeintent import EPS

SIDES = ("up", "down")

# How much of its run off the open a candle has given back.
#
# IT GATES NOTHING NOW, and that is worth stating rather than
# leaving to be discovered. It was the bounce threshold, then the
# band that decided when price counted as turned; price is the
# inner candle itself again -- "simple downs and ups only i dont
# care how much just what it actually reads" -- so there is
# nothing left for a percentage to decide.
#
# The number is still computed and still rides in the payload,
# because it is a true fact about the candle and costs nothing.
# The selector that set it is off the page: a control that
# changes no reading is worse than no control.
BOUNCE_AT = 0.33
BOUNCE_CHOICES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.33)

# How far along BOTH legs a bar has to be before any direction is
# read off it. The same 10% `fusedbar.live_signal` already waits
# for, imported rather than restated so the two cannot drift.
MIN_SETTLE = 0.10

# The turn has to stand this long before it opens a trade. Chris:
# "after the moving together holds for 5 seconds then reenter the
# trade" -- the same five seconds his tick strategy runs on.
DWELL_S = 5.0

# Where in its own high-low range the close has to sit before the
# panel will say "at the top" or "at the bottom" of the bar.
EDGE = 0.25


def position(high: float, low: float, close: float) -> float | None:
    """Where the close sits in the bar's range, 0 at the low.

    `None` when there is no range yet -- a bar whose prints have
    all landed on one price. That is a real state in thin tape and
    must not read as "at the bottom", which is what 0.0 would.
    """
    try:
        h, lo, c = float(high or 0.0), float(low or 0.0), float(close or 0.0)
    except (TypeError, ValueError):
        return None
    span = h - lo
    if span <= EPS:
        return None
    return max(0.0, min(1.0, (c - lo) / span))


class AgreeHold:
    """How long the two legs have been moving together THIS run.

    NOT `stackwin.Dwell`, and the difference is the whole reason
    this exists. There a flat is an ABSENCE -- the candle sitting
    exactly on its open for an instant -- and ignoring it is what
    finally made a hold elapse at all.

    Here a break is an EVENT. The bar bounced, or the two came
    apart, or a new one started settling; the agreement stopped,
    and Chris's rule is that it has to stand again from scratch:
    "if it bounces but then starts moving together again after the
    moving together holds for 5 seconds then reenter the trade".

    Fed `Dwell` instead, a bounce lasting two minutes left the
    clock untouched and the next `together` reading opened a trade
    on the same instant it appeared.

    A clock that jumps backwards makes `held_s` zero rather than
    negative.
    """

    __slots__ = ("_side", "_since", "runs")

    def __init__(self) -> None:
        self._side = ""
        self._since = 0.0
        self.runs = 0

    def on_tick(self, side: str, now: float) -> None:
        d = side if side in SIDES else ""
        try:
            t = float(now or 0.0)
        except (TypeError, ValueError):
            return
        if not d:
            # THE BREAK RESTARTS IT. This is the one line that
            # differs from `Dwell`, and it is the line Chris's
            # rule lives in.
            self._side = ""
            self._since = 0.0
            return
        if d != self._side:
            self._side = d
            self._since = t
            self.runs += 1

    @property
    def side(self) -> str:
        return self._side

    def held_s(self, now: float) -> float:
        try:
            t = float(now or 0.0)
        except (TypeError, ValueError):
            return 0.0
        if not self._side or self._since <= 0 or t < self._since:
            return 0.0
        return t - self._since


def _num(v) -> float:
    """A number, or 0.0 -- never an exception.

    THE PAYLOAD CARRIES THE RAW FIGURES NOW so they can be lined up
    against the chart header, and the first version wrote
    `int(net_ofi or 0)` straight into it. That raises on a string,
    and the one caller is a route whose exceptions get swallowed
    into `journal.error` where nothing is looking. A test that
    feeds junk to every entry point is what caught it.
    """
    try:
        return float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0


def run_from_open(open_px: float, high: float, low: float
                  ) -> tuple[str, float]:
    """Which way this candle ran off its open, and how far.

    THE OPEN IS THE ANCHOR, not the extremes. Chris: "the extremes
    dont matter what is is doing is it moving to tick charts
    opening or away". The furthest point from the open is used
    only to answer how far away it got -- which way is up or down
    off ONE price, the one the candle started at.
    """
    try:
        o, h, lo = float(open_px or 0.0), float(high or 0.0), \
            float(low or 0.0)
    except (TypeError, ValueError):
        return "", 0.0
    if o <= 0.0:
        return "", 0.0
    up_run, dn_run = max(0.0, h - o), max(0.0, o - lo)
    if up_run <= EPS and dn_run <= EPS:
        return "", 0.0
    return ("up", up_run) if up_run >= dn_run else ("down", dn_run)


def back_to_open(open_px: float, high: float, low: float,
                 close: float) -> float:
    """How much of the run back toward the open, 0 to 1.

    0.0 is price at its furthest from the open -- still going. 1.0
    is all the way back to the open, the whole run given up.

    AGAINST THE RUN, NOT THE RANGE. The question is "is it moving
    to the open or away", so the denominator is how far it got
    from the open, and nothing else enters it.
    """
    d, run = run_from_open(open_px, high, low)
    if not d or run <= EPS:
        return 0.0
    try:
        h, lo, c = float(high or 0.0), float(low or 0.0), \
            float(close or 0.0)
    except (TypeError, ValueError):
        return 0.0
    gone = (h - c) if d == "up" else (c - lo)
    return max(0.0, min(1.0, gone / run))


# THE TURNING ROW'S BAND, IN TICKS, AND WHAT YOU CAN PICK.
#
# THE FIRST VERSION OF THIS ROW WAS WRONG AND THE WAY IT WAS WRONG
# IS WORTH KEEPING WRITTEN DOWN. It measured the close against the
# CANDLE'S HIGH -- a number that only moves when a new high prints.
# So on a bar that ran up and pulled back, it said "down" and went
# on saying "down" while price rallied fourteen points, because
# every one of those closes was still below a high set minutes
# earlier. Chris: "thoss one has a turning but its stuck in place
# doing nothing". It was not frozen. It was anchored.
#
# THE EXTREME HAS TO TRAVEL. While this reads up, the high it
# measures from moves up with every new high; it flips to down only
# when price comes `band` ticks back off THAT high, and then the
# low becomes the reference and travels down. Chris: "it should
# read what the chart is doing in real time", and a reference that
# stopped moving an hour ago is not real time.
#
# TICKS, NOT A PERCENTAGE. A third of a 94-tick run is 31 ticks,
# which is no faster than the thing this was added to fix. A fixed
# few ticks off the live extreme is the same answer on every bar
# however far it ran.
TURN_TICKS = 4.0
TICK = 0.25

# WHAT THE DIAL OFFERS. Chris picked a selector over a fixed
# number -- the right call, because the band that reads as a turn
# and the band that reads as noise is a property of his tape and
# not of this module. In ticks, because that is what the chart is
# ruled in.
TURN_CHOICES = (2, 4, 6, 8, 12, 16, 24)


def turn_state(close: float, band_ticks: float = TURN_TICKS,
               prev: dict | None = None,
               open_px: float = 0.0, bar_ts: float = 0.0,
               tick: float = TICK) -> dict:
    """Which way price is going right now, off a travelling extreme.

    Returns {"dir", "ticks_off", "hi", "lo", "bar_ts"}. `ticks_off`
    is how far price has already come back off the extreme it is
    measured from -- a countdown to the flip, not a size of move.

    INTRABAR, AND IT STARTS BLANK. Asked what this should do when a
    new fused candle prints, Chris chose "Start blank each candle":
    the reference resets to the new candle's open and the row names
    no direction until price has travelled the band off it. That
    blank is the honest state -- a direction carried in from the
    previous bar is a claim about a bar that has not happened yet.

    BOTH EXTREMES ARE TRACKED WHILE IT IS BLANK, so the first
    direction comes off whichever end price actually left, rather
    than off the open it happened to start at.

    SHOWN, NEVER TRADED. The rule reads the inner candle, which is
    what the chart greys on. This answers a different question and
    sits beside it. Three attempts to make a derived reading into
    the price leg each put the panel into contradiction with the
    candle next to it, so it does not get a fourth.
    """
    prev = prev or {}
    try:
        c = float(close or 0.0)
        o = float(open_px or 0.0)
        t = float(tick or TICK) or TICK
        band_t = max(0.0, float(band_ticks or 0.0)) * t
        ts = float(bar_ts or 0.0)
    except (TypeError, ValueError):
        return {"dir": "", "ticks_off": 0.0, "hi": 0.0, "lo": 0.0,
                "bar_ts": 0.0}

    # A NEW BAR WIPES IT. Also the first reading ever, where there
    # is no previous extreme to carry: `hi`/`lo` missing is not
    # zero, it is "nothing has been seen yet", and treating it as
    # zero would put the low at zero and never flip again.
    fresh = bool(float(prev.get("bar_ts") or 0.0) != ts
                 or prev.get("hi") is None or prev.get("lo") is None)
    if fresh:
        d, hi, lo = "", (o or c), (o or c)
    else:
        d = str(prev.get("dir") or "")
        try:
            hi = float(prev.get("hi"))
            lo = float(prev.get("lo"))
        except (TypeError, ValueError):
            d, hi, lo = "", c, c

    if d == "up":
        if c >= hi:
            hi = c                       # the extreme travels
        elif band_t > 0.0 and (hi - c) >= band_t:
            d, lo = "down", c            # and hands over
    elif d == "down":
        if c <= lo:
            lo = c
        elif band_t > 0.0 and (c - lo) >= band_t:
            d, hi = "up", c
    else:
        hi, lo = max(hi, c), min(lo, c)
        if band_t > 0.0 and (c - lo) >= band_t:
            d, hi = "up", c
        elif band_t > 0.0 and (hi - c) >= band_t:
            d, lo = "down", c

    if d == "up":
        off = max(0.0, hi - c) / t
    elif d == "down":
        off = max(0.0, c - lo) / t
    else:
        off = 0.0
    return {"dir": d, "ticks_off": round(off, 1),
            "hi": hi, "lo": lo, "bar_ts": ts}


def pullback(body_dir: str, high: float, low: float,
             close: float) -> float:
    """How far off its extreme the candle has come, 0 to 1.

    THE UPPER WICK, AS A FRACTION OF THE WHOLE CANDLE. On an up
    body: 0.0 is the close sitting on the high and 1.0 is the close
    back at the low. Down body, the other way round.

    THE INNER BODY PICKS THE EXTREME, the outer candle supplies the
    range. A body that closed up ran to the high, so the high is
    what it can come back from -- which is what "pulling back from
    the top" means when you look at the chart.

    FLAT BODY, NO READING. A candle that closed where it opened has
    not run anywhere, so there is no extreme to have come off and
    0.0 would be a claim rather than an answer.
    """
    try:
        h, lo, c = float(high or 0.0), float(low or 0.0), float(close or 0.0)
    except (TypeError, ValueError):
        return 0.0
    span = h - lo
    if span <= EPS or body_dir not in SIDES:
        return 0.0
    if body_dir == "up":
        return max(0.0, min(1.0, (h - c) / span))
    return max(0.0, min(1.0, (c - lo) / span))


def at_edge(pos: float | None, edge: float = EDGE) -> str:
    """"top", "bottom" or "" for the middle of the range."""
    if pos is None:
        return ""
    if pos >= 1.0 - edge:
        return "top"
    if pos <= edge:
        return "bottom"
    return ""


def legs(net_ofi: float, ofi_target: float,
         prints: float, tick_target: float) -> dict:
    """How far along each leg the forming bar is.

    BOTH NUMBERS, NEVER ONE. A bar can sit at 98% on one leg and 4%
    on the other, and that is the whole reason this bar type
    exists -- `fusedbar.progress` says the same thing and this
    mirrors it so the gate panel does not have to call back into
    the builder for two divisions.
    """
    try:
        no, ot = abs(float(net_ofi or 0.0)), float(ofi_target or 0.0)
        pr, tt = float(prints or 0.0), float(tick_target or 0.0)
    except (TypeError, ValueError):
        return {"ofi_pct": 0.0, "tick_pct": 0.0, "waiting_on": "both"}
    ofi_pct = min(1.0, no / ot) if ot > 0 else 0.0
    tick_pct = min(1.0, pr / tt) if tt > 0 else 0.0
    waiting = ("" if ofi_pct >= 1.0 and tick_pct >= 1.0 else
               "the book" if tick_pct >= 1.0 else
               "prints" if ofi_pct >= 1.0 else "both")
    return {"ofi_pct": round(100.0 * ofi_pct, 1),
            "tick_pct": round(100.0 * tick_pct, 1),
            "waiting_on": waiting}


def settled(net_ofi: float, ofi_target: float,
            prints: float, tick_target: float,
            min_settle: float = MIN_SETTLE) -> tuple[bool, str]:
    """Has enough of this bar arrived to read a direction off it?

    BOTH LEGS, NOT EITHER. A bar can be 90% of the way along its
    print count with an accumulator sitting on zero -- busy and
    uncommitted -- and that is precisely the bar whose direction
    means nothing.

    Chris: "everytime a new tick ofi candle prints instead allowing
    it to bounce up and down or not sure of direction let it settle
    and then pick true direction."
    """
    try:
        no, ot = abs(float(net_ofi or 0.0)), float(ofi_target or 0.0)
        pr, tt = float(prints or 0.0), float(tick_target or 0.0)
        frac = float(min_settle or 0.0)
    except (TypeError, ValueError):
        return False, "nothing has arrived in this bar yet"
    if frac <= 0.0:
        return True, ""
    pct = int(round(100.0 * frac))
    need_o, need_t = ot * frac, tt * frac
    if ot > 0 and no < need_o:
        return False, (f"the book has moved {int(no):,} of the "
                       f"{int(need_o):,} ({pct}% of the bar) it takes "
                       "before a direction counts")
    if tt > 0 and pr < need_t:
        return False, (f"{int(pr):,} prints in -- {int(need_t):,} "
                       f"({pct}% of the bar) before a direction "
                       "counts")
    return True, ""


def _say(where: str, ofi_dir: str, tick_dir: str) -> str:
    """One sentence about the two candles, and no numbers in it.

    Chris: "it should be simple downs and ups only i dont care how
    much just what it actually reads".
    """
    if where == "settling":
        return "settling -- not enough of this bar yet"
    if where == "together":
        return (f"the OFI candle is {ofi_dir} and the tick candle "
                f"is {tick_dir} -- they agree")
    if where == "apart":
        return (f"the OFI candle is {ofi_dir} and the tick candle "
                f"is {tick_dir} -- they disagree, and the chart "
                "greys this bar")
    if ofi_dir not in SIDES:
        return "the OFI candle has no direction yet"
    return "the tick candle has no direction yet"


def read(ofi_dir: str = "", tick_dir: str = "",
         open_px: float = 0.0, high: float = 0.0, low: float = 0.0,
         close: float = 0.0,
         net_ofi: float = 0.0, ofi_target: float = 0.0,
         prints: float = 0.0, tick_target: float = 0.0,
         bounce_at: float = BOUNCE_AT,
         min_settle: float = MIN_SETTLE,
         bar_ts: float = 0.0,
         tradeable: bool | None = None,
         prev: dict | None = None,
         # ON THE END, NOT IN THE MIDDLE. Inserting a parameter
         # above `prev` once silently turned every existing
         # caller's `tradeable` into something else; ten tests
         # caught it and nothing else would have.
         turn_ticks: float = TURN_TICKS) -> dict:
    """The whole reading of one forming fused bar.

    `where` is the headline and it is one of:

        "settling"  too little of the bar has arrived to read a
                    direction off it at all
        "together"  both legs the same way, price still at the end
                    of the move it made
        "bounce"    both legs the same way, price has come back off
                    that extreme by more than the dial
        "apart"     the book and the traded price disagree
        ""          settled, but neither leg has said anything

    ONLY "together" IS TRADEABLE, and only after it has held. The
    rest are reasons not to be in: "never trade when book and price
    are apart or nothing settled".
    """
    ofi_dir = str(ofi_dir or "")
    # PRICE IS THE INNER CANDLE. Chris, pointing at the chart:
    # "the outer shale is the body the OFI the inner candle is the
    # tick candle what this is supposed to do it tell me whats
    # actually happening between the 2" -- and then "it should be
    # simple downs and ups only i dont care how much just what it
    # actually reads".
    #
    # SO IT IS `close vs open` AND NOTHING DERIVED. Three readings
    # were tried in its place -- the body's own retrace, a rolling
    # window over the last N prints, and the run off the open --
    # and the last of them put the panel into open contradiction
    # with the candle beside it: on a bar the chart GREYED for the
    # book and price disagreeing, the panel said "together" and
    # offered a short. The chart greys on the body. So does this.
    body_dir = str(tick_dir or "")
    # NOT ASSIGNED TWICE. `tick_dir` is set from the body BELOW,
    # after the hold, and a copy here would be dead -- overwritten
    # on every path. It was, and a mutation inverting it changed
    # nothing and broke no test: dead code absorbs the check that
    # was meant to land on the live line.
    bounce_at = float(bounce_at if bounce_at is not None else BOUNCE_AT)
    prev = prev or {}
    # THE TURNING ROW, READ FROM ITS OWN MEMORY. Its state rides
    # home in `held["turn"]` and comes back in `prev`, the same way
    # every other held reading here does -- this module stays pure
    # and the caller keeps the one copy.
    try:
        turn_band = max(0.0, float(turn_ticks
                                   if turn_ticks is not None
                                   else TURN_TICKS))
    except (TypeError, ValueError):
        turn_band = TURN_TICKS
    turn = turn_state(close, turn_band, (prev or {}).get("turn"),
                      open_px=open_px, bar_ts=bar_ts)
    # THE MEMORY BELONGS TO ONE BAR. Carrying a held direction into
    # a new bar would defeat the settling it was added for -- the
    # new bar would open already pointing somewhere.
    same_bar = bool(float(prev.get("bar_ts") or 0.0)
                    == float(bar_ts or 0.0))
    ok, settle_why = settled(net_ofi, ofi_target, prints,
                             tick_target, min_settle)
    if same_bar:
        # HELD THROUGH A TIE. A candle sitting exactly on its open
        # for an instant has not turned, and an absence must never
        # read as a reversal. This is the only guard the reading
        # needs now: there is nothing else between the print and
        # the word on the screen.
        #
        # THE BODY IS HELD WHETHER OR NOT THE BAR HAS SETTLED.
        #
        # Chris: "the body (shape) keeps blinking in and out its
        # supposed to be as accurate as it is right now just a
        # constant reading never fading".
        #
        # It is drawn and it decides nothing, so the settling gate
        # -- which exists to stop a RULE reading noise -- has no
        # business over it. Behind that gate it fell back to the
        # raw close-against-open on every poll of an unsettled
        # bar, and a candle sits exactly on its open often enough
        # that the row flickered.
        body_dir = hold_dir(body_dir, str(prev.get("body_dir") or ""))
    if ok and same_bar:
        # HELD THROUGH A TIE, once the bar has settled. A bar that
        # ticks back to flat for an instant has not changed its
        # mind, and an absence must never read as a reversal.
        #
        # THESE TWO DECIDE, so they stay behind the gate: a
        # direction held from before the bar settled is a
        # direction read off noise.
        ofi_dir = hold_dir(ofi_dir, str(prev.get("ofi_dir") or ""))
    # PRICE FOLLOWS THE BODY EXACTLY, held the same way and from
    # the same print. They are one reading under two names: the
    # body is what the panel draws and price is what the rule
    # reads, and the day they differ the panel is lying.
    tick_dir = body_dir

    pos = position(high, low, close)
    edge = at_edge(pos)
    # THE INNER BODY PICKS THE EXTREME, the outer candle supplies
    # the range. Framed on the body rather than the book because
    # the body is the thing that bounces: "ofi remaining down but
    # tick is bouncing from bottom or top".
    agree = bool(ofi_dir in SIDES and ofi_dir == tick_dir)

    if not ok:
        where = "settling"
    elif ofi_dir not in SIDES:
        where = ""
    elif tick_dir not in SIDES:
        # NO RUN OFF THE OPEN YET -- every print on one price, or
        # the candle sitting exactly where it started. An absence,
        # not a disagreement.
        where = ""
    elif not agree:
        where = "apart"
    else:
        where = "together"

    return {
        "where": where,
        "settled": ok,
        "settle_why": settle_why,
        # THE SIDE THIS OFFERS, AND ONLY FROM "together". Every
        # other state returns no side at all rather than a side with
        # a flag beside it -- a side that exists is a side something
        # eventually trades.
        "side": ofi_dir if where == "together" else "",
        "agree": agree,
        "ofi_dir": ofi_dir,
        "tick_dir": tick_dir,
        # DRAWN, NOT TRADED. The candle's shape, beside the live
        # direction, so the two can be seen disagreeing -- which is
        # the whole case for the change.
        "body_dir": body_dir,
        # WHICH WAY IT RAN OFF THE OPEN AND HOW FAR, carried for
        # anyone reading the payload. Nothing decides on it and
        # the panel no longer draws it: "i dont care how much just
        # what it actually reads".
        # WHICH WAY PRICE IS GOING RIGHT NOW, off an extreme that
        # travels with it, and how far back off that extreme it has
        # already come. Drawn beside the inner candle, never read
        # by the rule: `decides` is False and a test asserts it.
        "turning": {
            "dir": turn["dir"],
            "ticks_off": turn["ticks_off"],
            "band": float(turn_band),
            "decides": False,
        },
        "run_dir": run_from_open(open_px, high, low)[0],
        "run_pts": round(run_from_open(open_px, high, low)[1], 2),
        # WHERE PRICE SITS IN ITS OWN RANGE, and how much of the
        # bar's move it has given back. Two different questions and
        # the panel draws both: the first is "top or bottom of the
        # candle", the second is "is this still pushing".
        "pos": pos,
        "at": edge,
        # WHAT THE CHART ITSELF SHOWS about this bar. `grayed` is
        # the wash the renderer puts over a bar whose book and
        # price disagree, carried here so the panel and the candle
        # cannot say different things about the same bar.
        "grayed": (None if tradeable is None else not bool(tradeable)),
        "open": _num(open_px),
        "high": _num(high),
        "low": _num(low),
        "close": _num(close),
        "net_ofi": int(_num(net_ofi)),
        "back": round(back_to_open(open_px, high, low, close), 3),
        "bounce_at": float(bounce_at or 0.0),
        "legs": legs(net_ofi, ofi_target, prints, tick_target),
        "say": _say(where, ofi_dir, tick_dir),
        # WHAT THE NEXT READING NEEDS BACK. A field dropped from
        # here does not error -- it silently resets that part of the
        # reading on every poll, which is how a hold that never
        # elapsed and a direction that chattered both got shipped.
        "held": {"ofi_dir": ofi_dir, "tick_dir": tick_dir,
                 "body_dir": body_dir,
                 # THE TURNING ROW'S TRAVELLING EXTREME. Dropped
                 # from here it does not error -- it silently
                 # resets the reference on every poll, which is a
                 # row that can never flip and looks exactly like
                 # the bug this replaced.
                 "turn": turn,
                 "bar_ts": float(bar_ts or 0.0)},
        "decides": True,
    }


def entry_side(where: str, side: str,
               held_s: float | None = None,
               min_hold_s: float = DWELL_S,
               tf_dir: str = "", tf_on: bool = False,
               tf_name: str = "5m",
               tf2_dir: str = "", tf2_on: bool = False,
               tf2_name: str = "15m") -> tuple[str, str]:
    """The side this bar is offering, and why not.

    Chris: "the auto trader will then trade the direction that is
    set for example: ( book down or up, price down or up and says
    moving together) ... never trade when book and price are apart
    or nothing settled".

    SO ONLY "together" OPENS ANYTHING. A bounce does not -- it is
    what closes a trade, and entering on one would be entering
    against the retrace that just stopped the last one.

    THE HOLD IS ON THE AGREEMENT ITSELF. "after the moving together
    holds for 5 seconds then reenter the trade" -- it is the
    together STATE that has to stand, not either leg on its own, so
    a bar that flickers in and out of agreement never accumulates
    five seconds of it.
    """
    if where == "settling":
        return "", "the bar has not settled -- no direction yet"
    if where == "apart":
        return "", "the book and price are apart"
    if where != "together" or side not in SIDES:
        return "", "nothing settled -- neither leg has committed"
    if held_s is not None and min_hold_s > 0 and held_s < min_hold_s:
        return "", (f"they have been moving {side} together for "
                    f"{held_s:.0f}s -- it has to hold "
                    f"{min_hold_s:.0f}s")
    # THE TIMEFRAME, AS A FILTER AND ONLY A FILTER. Chris: "use it
    # as a trade filter to only make a trade if the 5 minute agrees
    # with the book and price".
    #
    # CHECKED AFTER THE HOLD, not folded into it. The five seconds
    # is on the book-and-price agreement, which is what he asked
    # for by name; the timeframe is consulted at the moment of
    # entry. Folding it in would mean a timeframe that turned
    # green five seconds ago restarts a clock that has been
    # running for a minute.
    #
    # ON MEANS ON. A filter switched on with nothing to say must
    # block, or switching it on does nothing until the candle
    # happens to have a body -- which is the gate-3 fault again.
    # TWO OF THEM, EACH ITS OWN GATE. Chris: "lets add another
    # timeframe on the side of that one", same rule. They are
    # checked in order so the panel blames the one that actually
    # refused rather than "a timeframe".
    for d, on, nm in ((tf_dir, tf_on, tf_name),
                      (tf2_dir, tf2_on, tf2_name)):
        if not on:
            continue
        if d not in SIDES:
            return "", f"the {nm} candle has no direction yet"
        if d != side:
            return "", (f"the {nm} candle is {d} against a "
                        f"{side} entry")
    return side, ""


def exit_reason(trade_side: str, where: str,
                tick_dir: str = "", tf_dir: str = "",
                tf_on: bool = False, tf_name: str = "5m",
                tf2_dir: str = "", tf2_on: bool = False,
                tf2_name: str = "15m") -> str:
    """Why this trade should end now, or "" to hold it.

    ONE THING CLOSES IT HERE, and the take profit closes the other
    kind. Chris: "the only exit is TP or when book and price
    disagree ... thats the only exit strategy".

    WHAT WAS REMOVED, AND DELIBERATELY. A bounce closed trades for
    one revision and no longer does: the pullback is a reading he
    wants to SEE, and taking him out on it was cutting winners
    that the take profit was already sized to collect. The bar
    closing closed trades too, and does not: a position now rides
    across bar boundaries, through the settling stretch on the far
    side, because a bar ending is bookkeeping and not a change in
    the market.

    THAT LEAVES NO CLOCK AND NO BAR ON THIS TRADE AT ALL. The only
    things that end it are the take profit, the two legs coming
    apart, a stop if one is set, and the journal's own rule that
    closes a position when the strategy actively offers the
    opposite side.

    SETTLING IS NOT AN EXIT EITHER. A new bar has no direction for
    a while by construction; reading that absence as a reason to
    be out would close every trade at every bar boundary, which is
    the bar-close exit wearing a different name.
    """
    if trade_side not in SIDES:
        return ""
    if where == "apart":
        return "the book and price have come apart"
    # PRICE AGAINST THE TIMEFRAME. Chris: "exit a trade when price
    # disagrees with the 5 minute".
    #
    # PRICE, NOT THE BOOK, and that asymmetry is his: the entry
    # wants all three pointing one way, the exit watches the body
    # against the timeframe. The book coming apart from price is
    # already the line above.
    #
    # FLAT IS NOT A DISAGREEMENT. A timeframe candle sits exactly
    # on its open constantly, and reading that as a reversal would
    # close trades on nothing -- the mistake this engine has made
    # three times.
    if tick_dir in SIDES:
        for d, on, nm in ((tf_dir, tf_on, tf_name),
                          (tf2_dir, tf2_on, tf2_name)):
            if on and d in SIDES and d != tick_dir:
                return (f"price is {tick_dir} against the {nm} "
                        f"candle, which is {d}")
    return ""


def state(ofi_dir: str = "", tick_dir: str = "",
          open_px: float = 0.0, high: float = 0.0, low: float = 0.0,
          close: float = 0.0,
          net_ofi: float = 0.0, ofi_target: float = 0.0,
          prints: float = 0.0, tick_target: float = 0.0,
          bounce_at: float = BOUNCE_AT,
          min_settle: float = MIN_SETTLE,
          bar_ts: float = 0.0,
          tf_dir: str = "", tf_on: bool = False, tf_name: str = "5m",
          tf_left_s: float | None = None,
          tf2_dir: str = "", tf2_on: bool = False,
          tf2_name: str = "15m",
          tradeable: bool | None = None,
          held_s: float | None = None,
          min_hold_s: float = DWELL_S,
          prev: dict | None = None,
          turn_ticks: float = TURN_TICKS) -> dict:
    """The reading and the rule's verdict on it, from one place.

    ONE READING, NOT TWO. The panel is handed what the rule decided
    rather than the inputs to decide it again. A screen that
    recomputes a direction is a screen that eventually disagrees
    with the rule beside it, and that has happened here with the
    5-minute cell and with the OFI bar already.
    """
    out = read(ofi_dir=ofi_dir, tick_dir=tick_dir, open_px=open_px,
               high=high, low=low, close=close, net_ofi=net_ofi,
               ofi_target=ofi_target, prints=prints,
               tick_target=tick_target, bounce_at=bounce_at,
               min_settle=min_settle, bar_ts=bar_ts,
               tradeable=tradeable, prev=prev,
               turn_ticks=turn_ticks)
    side, why = entry_side(out["where"], out["side"],
                           held_s=held_s, min_hold_s=min_hold_s,
                           tf_dir=tf_dir, tf_on=tf_on,
                           tf_name=tf_name, tf2_dir=tf2_dir,
                           tf2_on=tf2_on, tf2_name=tf2_name)
    ready = not (held_s is not None and min_hold_s > 0
                 and held_s < min_hold_s)
    return {**out,
            "entry": side,
            "why": why,
            "held_s": (None if held_s is None else float(held_s)),
            "min_hold_s": float(min_hold_s or 0.0),
            "ready": ready,
            # THE TIMEFRAME ROW, for the panel. Carried whether or
            # not it is on, so the screen can say "off" rather than
            # going silent -- a filter that is invisible when
            # switched off reads as one that is not there.
            "tf": {"dir": str(tf_dir or ""), "on": bool(tf_on),
                   "name": str(tf_name or ""),
                   "agrees": bool(tf_dir in SIDES
                                  and tf_dir == out["side"]),
                   # HOW LONG THIS CANDLE HAS LEFT. Chris: "a
                   # candle countdown that way we aren't making
                   # trades in the middle of nowhere" -- asked
                   # what it should DO, he said "Just show it".
                   # So it is drawn and nothing is gated on it:
                   # `decides` on this field would be a lie the
                   # next revision gets built on.
                   "left_s": (None if tf_left_s is None
                              else max(0.0, float(tf_left_s))),
                   "decides": bool(tf_on)},
            "tf2": {"dir": str(tf2_dir or ""), "on": bool(tf2_on),
                    "name": str(tf2_name or ""),
                    "agrees": bool(tf2_dir in SIDES
                                   and tf2_dir == out["side"]),
                    # ONE COUNTDOWN, ON THE FIRST TIMEFRAME. His
                    # choice: two clocks on one panel is two
                    # numbers to read and neither gates anything.
                    "left_s": None,
                    "decides": bool(tf2_on)},
            "bar_ts": float(bar_ts or 0.0)}
