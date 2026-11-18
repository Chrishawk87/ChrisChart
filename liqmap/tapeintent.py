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


def read(bar_dir: str, open_px: float, high: float, low: float,
         close: float, prev: dict | None = None) -> dict:
    """What the tape is doing inside this bar, right now.

    `bar_dir` is the bar's own direction, close against open -- the
    same number gate 1 reports, not a second opinion about it.

    THE REFERENCE IS THE BAR'S EXTREME, AND THAT IS THE WHOLE FIX.
    Price off the high of an up bar is a pullback the instant it
    happens, because the high is a fact about this bar and not a
    point a fixed distance back in the tape.

    `prev` carries the deepest retrace reached since the current
    extreme, because "is this pullback still going or is it
    recovering" cannot be read from one instant. The caller keeps
    that memory and hands it back, exactly as `stackwin.state`
    does, so one module cannot hand the panel and the rule two
    different histories.
    """
    prev = prev or {}
    d = depth(bar_dir, open_px, high, low, close)
    at_extreme = d <= EPS

    # THE DEEPEST THIS PULLBACK HAS BEEN. Cleared the moment a new
    # extreme is made, because at that point there is no pullback
    # left to be recovering from. Also cleared when the bar's own
    # direction turns over, since the geometry is then measured off
    # the other end of the bar and the old number means nothing.
    turned = bar_dir != str(prev.get("bar_dir") or bar_dir)
    worst = 0.0 if (at_extreme or turned) else max(
        d, float(prev.get("worst") or 0.0))

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
        # Said here so the panel does not decide what the numbers
        # mean, and so the wording lives in one place.
        "say": _words(state, d, worst, recovering, at_extreme, bar_dir),
        # NEVER AN INPUT TO A TRADE. Carried on the reading itself
        # so it travels with it into any payload that reshapes.
        "decides": False,
    }


def _words(state: str, d: float, worst: float, recovering: bool,
           at_extreme: bool, bar_dir: str) -> str:
    """Plain English, or an honest refusal."""
    if not state or bar_dir not in SIDES:
        return "no direction in the bar yet"
    way = bar_dir.upper()
    if state == "continuation" or at_extreme:
        return f"extending the {way} move, at its best price"
    pct = int(round(d * 100))
    if recovering:
        return (f"pulling back on the {way} move -- {pct}% given back, "
                f"recovering from {int(round(worst * 100))}%")
    return f"pulling back on the {way} move -- {pct}% given back"
