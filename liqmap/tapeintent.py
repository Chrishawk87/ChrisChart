"""Is this tick bar continuing, or pulling back?

Chris: "down at the bottom of the tick bar there is the tape what we
are going to do with that is show tape intent at that moment during
the tick bar kinda the same way a 1 minute shows the 5 minute candles
intent", and then, asked which measurement: "which ever one will show
tick chart pullbacks and continuations thats what i actually need".

SO THIS IS NOT A SPEED READING. The line it replaces showed how FAST
the bar was forming -- its projected span against a median. Fast is
not the same as continuing, and a bar can quicken while price retraces
the whole move. That reading is kept as a trailing clause because it
is real; it is no longer the headline.

WHAT A PULLBACK ACTUALLY IS, SAID IN NUMBERS.

    A bar going UP has made a high. Two facts decide whether it is
    still going:

      HOW FAR BACK price has come from that high, as a fraction of
      the move the bar made to get there. At the high, nothing --
      the bar is extending. Half the move given back is a different
      market from a tick off the high, and the single number says
      which.

      WHICH WAY THE RECENT TAPE IS GOING. Depth alone cannot tell a
      pullback still running from one that has already turned, and
      those call for opposite actions: the first says wait, the
      second says the continuation has resumed.

    So: the recent slice going WITH the bar is a continuation, the
    recent slice going AGAINST it is a pullback, and the depth says
    how much damage the pullback has done. Both come off price,
    because a pullback on a tick chart IS a price event -- aggression
    can only imply one.

WHY AGGRESSION IS NOT THE HEADLINE. Buy and sell volume are recorded
per bar and are genuinely informative, but they answer a different
question: who was leaning on it, not where it went. The case where
they disagree with price is absorption, which this engine already
measures and already labels as deciding nothing. Putting it here
would answer a question he did not ask with a number that is harder
to read.

THE SLICE DOES NOT RESTART AT A BAR BOUNDARY. It is the last N
prints, whichever bars they landed in -- the same rule the gate's
direction follows, and for the same reason: a reading that restarts
on bookkeeping blinks out and gets read as an absence. "At that
moment" means the last N prints, which is always now.

Nothing in this module places, sizes, routes or times an order. It
decides nothing: the entry is the two gates, and this is drawn under
a label that says so.
"""

from __future__ import annotations

SIDES = ("up", "down")

# Float dust. Prices are quarter-points; anything under this is a
# rounding artefact, not a retrace.
EPS = 1e-9

# WHEN A PULLBACK IS BEING PAID FOR. Chris: "we are going to use
# volume to confirm tick bars continuation or pull back ... if volume
# agrees with tick charts direction for a pull back a trade should be
# taken in that pullback direction as well".
#
# The comparison is EFFORT, not raw volume: how much size it took to
# move price one tick on the way into the extreme, against how much
# it is taking to move it one tick on the way back. Raw volume would
# favour whichever leg has run longer, and a retrace that has only
# gone a tenth of the way would read light however hard it was
# fought.
#
# At 1.0 the retrace is being bought or sold as hard as the push was.
# Above it, harder -- the move is being fought and the pullback is
# the real trade. Below it the push is unopposed and drifting back,
# which is the continuation still standing.
HEAVY_RATIO = 1.0

# AND IT NEEDS A BAND, FOR THE SAME REASON THE VOTE DID.
#
# A bare threshold on a number that moves continuously flips every
# time it crosses. Measured on a simulated hour at 5,000 prints a
# bar: the heavy/light flag turned over about 1,250 times, against
# roughly 300 from the candle itself -- so four out of five side
# changes were this one number chattering around 1.0, and each one
# of them is a round trip.
#
# This is the identical defect `PATH_BAND` was added to fix in the
# tick vote, and it was found the same way. Claiming HEAVY needs
# the ratio above 1.25; losing it needs a drop below 0.75. A state
# is harder to leave than it was to enter, which is the property
# that stops flapping, and nothing in between changes anything.
HEAVY_BAND = 0.25

# AND THE RETRACE HAS TO BE A RETRACE BEFORE ANY OF IT MEANS
# ANYTHING.
#
# `effort_ratio` divides by the distance price has come back. As
# that distance goes to nothing the ratio goes to infinity, so one
# tick off the high with any volume behind it reads as the move
# being fought -- and on a TRENDING tape, which makes dozens of
# one-tick retraces, that is constant. Measured before this line
# existed: a strong trend produced 2,659 side changes an hour
# against 791 on a choppy one, which is backwards and was the tell.
#
# Below this depth the comparison is not wrong, it is undefined:
# there is no retrace yet to weigh the volume against. The reading
# still says PULLBACK -- that part is instant and stays instant --
# it just cannot turn the trade round on arithmetic that has not
# got a denominator yet.
MIN_PULLBACK = 0.15


def depth(direction: str, open_px: float, high: float, low: float,
          close: float) -> float:
    """How far price has come back off the bar's extreme, 0 to 1.

    0.0 is at the extreme -- the bar is at its best price and has
    given nothing back. 1.0 is all the way back to the open, which
    is the whole move retraced.

    Measured against the move the bar MADE, not its full range: on
    an up bar that is high-minus-open, so the number answers "how
    much of this bar's gain is gone" rather than "where does price
    sit in the day's noise".
    """
    try:
        o, h, lo, c = (float(open_px or 0.0), float(high or 0.0),
                       float(low or 0.0), float(close or 0.0))
    except (TypeError, ValueError):
        return 0.0
    if direction == "up":
        span = h - o
        if span <= EPS:
            return 0.0
        return max(0.0, min(1.0, (h - c) / span))
    if direction == "down":
        span = o - lo
        if span <= EPS:
            return 0.0
        return max(0.0, min(1.0, (c - lo) / span))
    return 0.0


def hold(now: str, prev: str = "") -> str:
    """This reading, or the last real one.

    The same rule both gates follow. A bar sitting exactly on its
    open for an instant has no direction to read, and reporting that
    as "no intent" would blink the line out -- which is the fault
    that took two attempts to get out of the gates themselves.
    """
    return now if now else (prev or "")


def effort_ratio(bar_dir: str, open_px: float, high: float, low: float,
                 close: float, impulse_v: float,
                 retrace_v: float) -> float:
    """How hard the retrace is being fought, against the push.

    Volume per unit of price moved, on each leg, as a ratio. The
    units cancel, so no tick size is needed and the number means the
    same thing on any instrument.

    0.0 means "cannot say" -- no push, no retrace, or no volume
    behind one of them. It is never guessed at, because a fabricated
    1.0 would read as a perfectly balanced fight.
    """
    try:
        o, h, lo, c = (float(open_px or 0.0), float(high or 0.0),
                       float(low or 0.0), float(close or 0.0))
        iv, rv = float(impulse_v or 0.0), float(retrace_v or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if bar_dir == "up":
        push, back = h - o, h - c
    elif bar_dir == "down":
        push, back = o - lo, c - lo
    else:
        return 0.0
    if push <= EPS or back <= EPS or iv <= 0 or rv <= 0:
        return 0.0
    return (rv * push) / (iv * back)


def read(bar_dir: str, open_px: float, high: float, low: float,
         close: float, volume: float = 0.0, bar_ts: float = 0.0,
         prev: dict | None = None) -> dict:
    """What the tape is doing inside this bar, right now.

    `bar_dir` is the bar's own direction, close against open -- the
    same number the rule reads, not a second opinion about it.

    THE REFERENCE IS THE BAR'S EXTREME, AND THAT IS WHAT MAKES IT
    IMMEDIATE. Price off the high of an up bar is a pullback the
    instant it happens, because the high is a fact about this bar
    and not a point a fixed distance back in the tape.

    VOLUME SAYS WHETHER THE PULLBACK IS BEING PAID FOR. The size
    behind the push into the extreme against the size behind the
    retrace out of it, per unit of price moved. A retrace nobody is
    paying for is drift and the push still stands; one that costs
    more per tick than the push did is the move being fought, and
    that is a trade in the retrace's own direction.

    `prev` carries what one instant cannot know: the deepest this
    retrace has been, and the bar's volume at the moment the extreme
    was last made. The caller keeps that memory and hands it back,
    exactly as `stackwin.state` does, so one module cannot hand the
    panel and the rule two different histories.
    """
    prev = prev or {}
    d = depth(bar_dir, open_px, high, low, close)
    at_extreme = d <= EPS
    vol = max(0.0, float(volume or 0.0))

    # A NEW BAR WIPES ALL OF IT. Volume restarts at nought and the
    # extremes belong to a bar that has closed, so a retrace depth
    # or a volume mark carried across the boundary is a number about
    # a move that is no longer on the screen.
    fresh = float(bar_ts or 0.0) != float(prev.get("bar_ts") or 0.0)
    turned = bar_dir != str(prev.get("bar_dir") or bar_dir)
    reset = fresh or turned

    worst = 0.0 if (at_extreme or reset) else max(
        d, float(prev.get("worst") or 0.0))

    # THE VOLUME MARK IS TAKEN AT THE EXTREME. Everything before it
    # is the push; everything after is the retrace. Re-marked every
    # time a new extreme is made, which is also when the retrace
    # ceases to exist.
    if at_extreme or reset:
        v_ext = vol
    else:
        v_ext = min(vol, float(prev.get("v_ext") or 0.0))
    impulse_v = v_ext
    retrace_v = max(0.0, vol - v_ext)

    ratio = effort_ratio(bar_dir, open_px, high, low, close,
                         impulse_v, retrace_v)
    # HYSTERESIS, not a second threshold. `prev` already carries
    # the state, so entering and leaving simply use the two edges
    # of the band.
    was = bool(prev.get("heavy"))
    if ratio <= 0 or d < MIN_PULLBACK:
        heavy = False
    elif was:
        heavy = ratio > HEAVY_RATIO - HEAVY_BAND
    else:
        heavy = ratio >= HEAVY_RATIO + HEAVY_BAND

    state = "continuation" if at_extreme else "pullback"
    if bar_dir not in SIDES:
        state = ""
    state = hold(state, str(prev.get("state") or ""))

    # RECOVERING IS A QUALIFIER, NOT A STATE. Price coming back off
    # the pullback's own low is worth seeing, but it is not a
    # continuation until a new extreme is made -- calling it one
    # early would be predicting the resumption rather than
    # reporting it, which is the mistake in the other direction.
    recovering = bool(not at_extreme and worst > 0.0 and d < worst - EPS)

    return {
        "state": state,
        "depth": round(d, 3),
        "worst": round(worst, 3),
        "recovering": recovering,
        "at_extreme": bool(at_extreme),
        "bar_dir": bar_dir,
        # THE VOLUME BEHIND EACH LEG, and the ratio between them.
        # Shown as well as used, because a rule that turns a trade
        # round on a number nobody can see is a rule nobody can
        # argue with.
        "impulse_v": round(impulse_v, 2),
        "retrace_v": round(retrace_v, 2),
        "ratio": round(ratio, 3),
        "heavy": heavy,
        # The memory the caller stores and hands back.
        "v_ext": round(v_ext, 2),
        "bar_ts": float(bar_ts or 0.0),
        # Said here so the panel does not decide what the numbers
        # mean, and so the wording lives in one place.
        "say": _words(state, d, worst, recovering, at_extreme, bar_dir,
                      ratio, heavy),
    }


def _words(state: str, d: float, worst: float, recovering: bool,
           at_extreme: bool, bar_dir: str, ratio: float = 0.0,
           heavy: bool = False) -> str:
    """Plain English, or an honest refusal."""
    if not state or bar_dir not in SIDES:
        return "no direction in the bar yet"
    way = bar_dir.upper()
    if state == "continuation" or at_extreme:
        return f"extending the {way} move, at its best price"
    pct = int(round(d * 100))
    if ratio <= 0:
        vol = "no volume read on the pullback yet"
    elif heavy:
        vol = (f"HEAVY -- {ratio:.2f}x the effort of the push, it is "
               "being fought")
    else:
        vol = f"light -- {ratio:.2f}x the push, nobody is paying for it"
    base = f"pulling back on the {way} move -- {pct}% given back, {vol}"
    if recovering:
        return base + f" (recovering from {int(round(worst * 100))}%)"
    return base
