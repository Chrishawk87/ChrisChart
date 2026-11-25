"""Reading a volume-sized footprint bar: push, activity, and a grade.

Chris, asked what should close a footprint bar: "the bar should mark
push and activity and should be just up strong is strong buy push
up, if normal push just buy , sell normal push down strong down if
volume pushes activity".

SO THE BAR IS CLOSED BY VOLUME AND GRADED BY WHAT HAPPENED INSIDE
IT. Those are two different jobs and putting both on the close
condition is the trap this whole line of work has been stuck in.

WHY DELTA DOES NOT CLOSE THE BAR, THOUGH IT IS THE PUSH.

    Making a bar wait for a delta threshold sounds right -- the bar
    would then mark a push by construction. It is the same mistake
    as the fused bar's OFI leg, for the same reason: delta is a
    running difference that resets every bar and can sit near zero
    for as long as the tape is two-sided. A bar that must reach it
    does not close in churn; the time cap closes it instead, and a
    cap that does the closing is a clock. That is how a 10,000 x
    4,000 fused bar became a 15-minute candle with another name.

    Volume only goes up. It cannot stall, it cannot reverse, and
    10,000 contracts means the same thing at eight in the evening
    as it does at ten in the morning. So volume closes the bar and
    delta grades it, and neither is asked to do the other's job.

WHY THE CLOCK IS THE ACTIVITY READING HERE, WHICH IS BACKWARDS FROM
EVERY TIME CHART.

    `VolumeRule` closes at EXACTLY N contracts -- it splits the
    print that would carry the bar past the line -- so every bar
    holds the same size. Volume therefore says nothing about one
    bar against another: they all have the same amount of it.

    What differs is how long it took. A bar that filled in twenty
    seconds and a bar that took ten minutes traded identical size;
    one of them was a market in a hurry. On a time chart volume is
    the activity reading and duration is fixed. Here duration is
    the activity reading and volume is fixed. Chris: "if volume
    pushes activity".

THE GRADES ARE HIS. "strong buy push up" when the push is one-sided
AND the size arrived fast; plain "buy" when it is one-sided at an
ordinary pace; the same two downward. A bar that is neither gets no
call rather than a weak one, because a grade nobody would act on is
noise on a panel that already has plenty.

NOTHING HERE PLACES, SIZES, ROUTES OR TIMES AN ORDER, and a test
asserts no function is named as though it did.
"""

from __future__ import annotations

SIDES = ("up", "down")
EPS = 1e-9

# HOW ONE-SIDED A BAR HAS TO BE BEFORE IT IS A PUSH AT ALL, as a
# share of the bar's own volume. Delta is bounded by volume, so this
# is a fraction between 0 and 1 and means the same thing on any bar
# size -- which is the point of measuring it against the bar rather
# than in contracts.
#
# THESE ARE A STARTING SHAPE, NOT A TUNING. They are written to be
# argued with after watching real bars, and the panel prints the
# share beside the grade so the argument can be had against the
# tape instead of against this comment.
PUSH_AT = 0.08
STRONG_AT = 0.18

# AND HOW FAST "fast" IS: this bar's duration against the median of
# recent ones. Under this it filled in a hurry. A fraction rather
# than a number of seconds, because the whole point of a volume bar
# is that its duration moves with the session.
FAST_AT = 0.70

# WHAT COUNTS AS DIAGONAL CONFIRMATION. Imbalanced levels on the
# side the push is going, against levels on the other. One level is
# noise on any bar big enough to be worth reading.
MIN_IMBALANCE = 2


def _num(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def share(delta: float, volume: float) -> float:
    """How much of this bar's volume was one-sided, 0 to 1.

    THE PUSH, MEASURED AGAINST THE BAR ITSELF. Delta in contracts
    says nothing without knowing how big the bar is; 400 is a
    shove on a 2,000-contract bar and a rounding error on a
    40,000-contract one. Against the bar it is the same number on
    both.
    """
    v = abs(_num(volume))
    if v <= EPS:
        return 0.0
    return max(0.0, min(1.0, abs(_num(delta)) / v))


def pace(seconds: float, median_s: float) -> float:
    """This bar's duration against the median of recent ones.

    Under 1 is faster than lately, over 1 is slower. `None` has no
    meaning here and 1.0 is the honest stand-in for "no median to
    compare against yet" -- it makes the bar ordinary rather than
    fast, so a fresh series cannot grade everything STRONG before
    it knows what normal looks like.
    """
    s, m = _num(seconds), _num(median_s)
    if s <= EPS or m <= EPS:
        return 1.0
    return s / m


def read(delta: float = 0.0, volume: float = 0.0,
         seconds: float = 0.0, median_s: float = 0.0,
         buy_imbalances: int = 0, sell_imbalances: int = 0,
         push_at: float = PUSH_AT, strong_at: float = STRONG_AT,
         fast_at: float = FAST_AT) -> dict:
    """One footprint bar, as a graded call.

    Returns the side, the grade, the two numbers behind them, and a
    sentence. Nothing here is a gate.
    """
    vol = abs(_num(volume))
    d = _num(delta)
    sh = share(d, vol)
    pc = pace(seconds, median_s)
    try:
        push_cut = max(0.0, min(1.0, float(push_at)))
        strong_cut = max(0.0, min(1.0, float(strong_at)))
        fast_cut = max(0.0, float(fast_at))
    except (TypeError, ValueError):
        push_cut, strong_cut, fast_cut = PUSH_AT, STRONG_AT, FAST_AT
    # A STRONG CUT BELOW THE PUSH CUT WOULD MAKE EVERY PUSH STRONG,
    # silently. Ordering them is cheaper than trusting whoever set
    # them.
    strong_cut = max(strong_cut, push_cut)

    side = ""
    if vol > EPS and sh >= push_cut and push_cut > 0.0:
        side = "up" if d > 0 else "down" if d < 0 else ""

    fast = bool(pc < fast_cut)
    try:
        bi, si = int(buy_imbalances or 0), int(sell_imbalances or 0)
    except (TypeError, ValueError):
        bi = si = 0
    # THE DIAGONAL, AS CONFIRMATION AND NOT AS A VOTE. Buyers
    # lifting the ask where sellers did not show up, on the side
    # the push is already going.
    backing = (bi if side == "up" else si if side == "down" else 0)
    against = (si if side == "up" else bi if side == "down" else 0)
    confirmed = bool(side and backing >= MIN_IMBALANCE
                     and backing > against)

    # STRONG NEEDS BOTH. Chris: "strong is strong buy push up ...
    # if volume pushes activity". One-sided alone is a push; it is
    # the size arriving fast that makes it a strong one. The
    # diagonal can carry a bar that is one-sided enough but not
    # quite fast, because an imbalanced run is the same statement
    # made at prices instead of in time.
    grade = ""
    if side:
        grade = "strong" if (sh >= strong_cut and (fast or confirmed)) \
            else "normal"

    call = ""
    if side == "up":
        call = "STRONG BUY" if grade == "strong" else "BUY"
    elif side == "down":
        call = "STRONG SELL" if grade == "strong" else "SELL"

    return {
        "side": side,
        "grade": grade,
        "call": call,
        "share": round(sh, 4),
        "delta": int(d),
        "volume": int(vol),
        "pace": round(pc, 3),
        "fast": fast,
        "seconds": round(_num(seconds), 1),
        "median_s": round(_num(median_s), 1),
        "buy_imbalances": bi,
        "sell_imbalances": si,
        "confirmed": confirmed,
        "push_at": round(push_cut, 4),
        "strong_at": round(strong_cut, 4),
        "fast_at": round(fast_cut, 3),
        "say": _say(side, grade, sh, pc, fast, confirmed, backing),
        # SHOWN, NEVER TRADED. Same as every other reader added to
        # this panel, and said on the payload rather than left to
        # be remembered.
        "decides": False,
    }


def _say(side: str, grade: str, sh: float, pc: float, fast: bool,
         confirmed: bool, backing: int) -> str:
    """One sentence, with the two numbers that produced the grade."""
    if not side:
        return (f"two-sided -- only {sh:.0%} of this bar's volume "
                "went one way")
    way = "buyers" if side == "up" else "sellers"
    speed = ("filled fast" if fast
             else f"ordinary pace ({pc:.1f}x the usual)")
    out = f"{way} took {sh:.0%} of the volume, {speed}"
    if confirmed:
        out += f" -- {backing} imbalanced levels behind it"
    if grade == "strong":
        out += " -- a strong push"
    return out


def size_for(volume_rate: float, target_s: float = 180.0,
             floor: int = 100, ceiling: int = 2_000_000) -> dict:
    """How many contracts a bar should hold to take `target_s`.

    THE SAME JOB AS SIZING THE FUSED PAIR AND A FAR EASIER ONE.
    Volume is a quantity that only goes up, so twice the time is
    twice the contracts and there is no square root and no
    correction loop -- the thing that made the OFI leg hard to
    size is exactly the thing volume does not do.
    """
    r = _num(volume_rate)
    t = max(1.0, _num(target_s, 180.0))
    if r <= 0.0:
        return {"contracts": 0, "target_s": t, "ready": False,
                "why": "no volume measured yet -- nothing to size from"}
    n = r * t
    # TWO SIGNIFICANT FIGURES so the number reads as a setting
    # rather than as a measurement that happened.
    if n >= 10.0:
        import math as _m
        mag = 10 ** (int(_m.floor(_m.log10(n))) - 1)
        n = round(n / mag) * mag
    n = int(max(int(floor), min(int(n), int(ceiling))))
    return {
        "contracts": n, "target_s": t, "ready": True,
        "rate": round(r, 2),
        "why": (f"{n:,} contracts for a ~{int(t)}s bar -- this tape "
                f"is trading {r:.1f} contracts a second"),
    }
