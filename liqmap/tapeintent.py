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

# THE SLICE, AS A FRACTION OF THE BAR. A fifth, because that is the
# ratio he described: a 1-minute candle inside a 5-minute one. It
# scales, so the slice means the same thing at 1,000 prints a bar as
# at 10,000 and nothing has to be retuned when the size changes.
SLICE_FRACTION = 0.2
# Below this many prints a slice is a handful of trades and its
# direction is noise. It only ever applies at a cold start.
MIN_SLICE = 20

# Float dust. Prices are quarter-points; anything under this is a
# rounding artefact, not a retrace.
EPS = 1e-9


def slice_prints(per_bar: int) -> int:
    """How many prints the recent slice holds, for this bar size."""
    try:
        n = int(per_bar or 0)
    except (TypeError, ValueError):
        return MIN_SLICE
    return max(MIN_SLICE, int(n * SLICE_FRACTION))


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
         close: float, slice_dir: str, slice_n: int = 0,
         prev: dict | None = None) -> dict:
    """What the tape is doing inside this bar, right now.

    `bar_dir` is the bar's own direction, close against open -- the
    same number gate 1 reports, not a second opinion about it.
    `slice_dir` is the recent slice's direction, close against open
    on the last N prints.

    `prev` is the previous reading, and holding a state means
    remembering it. This function stays pure and the caller keeps
    the memory, exactly as `stackwin.state` does, so one module
    cannot hand the panel and the rule two different histories.
    """
    prev = prev or {}
    d = depth(bar_dir, open_px, high, low, close)
    at_extreme = d <= EPS

    state = ""
    if bar_dir in SIDES and slice_dir in SIDES:
        # WITH the bar is a continuation; AGAINST it is a pullback.
        # Depth is how much it has cost, not what it is called --
        # a bar one tick off its high with the tape still selling
        # is a pullback that has just started, and saying so early
        # is the entire point of watching the slice.
        state = "continuation" if slice_dir == bar_dir else "pullback"

    state = hold(state, str(prev.get("state") or ""))
    slice_dir = hold(slice_dir if slice_dir in SIDES else "",
                     str(prev.get("slice_dir") or ""))

    return {
        "state": state,
        "depth": round(d, 3),
        "at_extreme": bool(at_extreme),
        "slice_dir": slice_dir,
        "bar_dir": bar_dir,
        "slice_n": int(slice_n or 0),
        # Said here so the panel does not decide what the numbers
        # mean, and so the wording lives in one place.
        "say": _words(state, d, at_extreme, bar_dir),
        # NEVER AN INPUT TO A TRADE. Carried on the reading itself
        # so it travels with it into any payload that reshapes.
        "decides": False,
    }


def _words(state: str, d: float, at_extreme: bool, bar_dir: str) -> str:
    """Plain English, or an honest refusal."""
    if not state or bar_dir not in SIDES:
        return "no direction in the bar yet"
    way = bar_dir.upper()
    pct = int(round(d * 100))
    if state == "continuation":
        if at_extreme:
            return f"extending the {way} move, at its best price"
        return (f"the tape has turned back {way} -- {pct}% of the "
                f"move still given back")
    if at_extreme:
        # Possible for an instant: price at the extreme while the
        # slice already points the other way.
        return f"turning away from the {way} high"
    return f"pulling back on the {way} move -- {pct}% given back"
