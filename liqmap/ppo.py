"""Chris's Tick Counter PPO, translated from Pine.

WHAT THE ORIGINAL ACTUALLY COMPUTES

Worth writing down, because two lines of the Pine do less than they look
like they do and a faithful translation has to match the behaviour rather
than the apparent intent.

    ad_formula = ((2*close - low - high) / (high - low)) * volume

That is the Chaikin money-flow multiplier times volume: where the close
sits inside the bar's range, on a scale of -1 (closed on the low) to +1
(closed on the high), weighted by how much traded. A doji or a zero-range
bar is zero by definition, which the original guards explicitly.

    ad_current  = cum(ad_formula)
    ad_previous = cum(ad_formula)[1]
    tickCounter = ad_current - ad_previous

A running total minus the same running total one bar ago is just the
current bar's contribution. So `tickCounter` is `ad_formula` for this bar
-- the cumulative sum cancels out entirely. Keeping the subtraction would
be copying the shape of the code rather than what it computes.

    mtfDelta = (close > close[1] ? volume : 0) - (close < close[1] ? volume : 0)

Signed volume: the bar's whole volume counted up if it closed higher than
the last close, down if lower, nothing if unchanged. The cumulative that
follows it is differenced one line later, which cancels the same way.

Both series are then divided by their own 50-bar standard deviation and
clamped to +/-3, so a quiet market and a busy one produce comparable
numbers, and one enormous bar cannot dominate the line. They are combined
70/30, smoothed, and a signal line is an EMA of the result.

TWO HONEST NOTES

The `showArrows` input in the original is never read -- there are no arrow
plots in the script. It is declared and does nothing.

The original can take A/D and momentum from different timeframes. This
computes both from the chart's own bars. On the 15m chart, with the
original's defaults (A/D blank = chart, momentum = 15), those are the same
thing. On any other timeframe they are not, and this says so on the panel
rather than quietly pretending otherwise.

AND THE ONE THAT MATTERS MOST

These are OUR bars: built from the venue's own fills, with the venue's own
volume. The same indicator on TradingView is reading a different feed, so
the numbers will not match print for print. The shape should agree; the
values are not comparable, and anyone expecting them to be will conclude
something is broken when nothing is.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

# The original's defaults.
STDEV_LEN = 50
AD_WEIGHT = 0.7
MOM_WEIGHT = 0.3
SMOOTH = 1
SIGNAL = 9
CLAMP = 3.0


def money_flow_volume(bar: Any) -> float:
    """Where the close sat in the bar's range, times what traded.

    Zero for a bar with no range, and for one that closed at both its high
    and its low -- the position of the close inside the range is undefined
    there, and the original guards both.
    """
    high = _f(bar, "h", "high")
    low = _f(bar, "l", "low")
    close = _f(bar, "c", "close")
    vol = _f(bar, "v", "volume")
    if high == low:
        return 0.0
    if close == high and close == low:
        return 0.0
    return ((2 * close - low - high) / (high - low)) * vol


def signed_volume(bars: Sequence[Any]) -> list[float]:
    """Volume counted up or down by where this close sits against the last.

    The first bar has nothing to compare against, so it contributes
    nothing rather than being counted as up.
    """
    out = [0.0]
    for i in range(1, len(bars)):
        c = _f(bars[i], "c", "close")
        p = _f(bars[i - 1], "c", "close")
        v = _f(bars[i], "v", "volume")
        out.append(v if c > p else -v if c < p else 0.0)
    return out


def stdev(xs: Sequence[float], length: int) -> list[float | None]:
    """Rolling population standard deviation, as Pine's `ta.stdev` uses.

    None until there is a full window: a standard deviation over eleven
    samples of a fifty-sample window is a different statistic, and using
    it would make the first fifty bars of every chart quietly wrong in a
    way that looks like signal.
    """
    out: list[float | None] = []
    for i in range(len(xs)):
        if i + 1 < length:
            out.append(None)
            continue
        window = xs[i + 1 - length:i + 1]
        mean = sum(window) / length
        var = sum((x - mean) ** 2 for x in window) / length
        out.append(math.sqrt(var))
    return out


def ema(xs: Sequence[float | None], length: int) -> list[float | None]:
    """Pine's recursive EMA, seeded on the first value it is given.

    Leading Nones are carried through rather than treated as zero, which
    would drag the first real values toward the axis.
    """
    if length <= 1:
        return list(xs)
    alpha = 2.0 / (length + 1.0)
    out: list[float | None] = []
    prev: float | None = None
    for x in xs:
        if x is None:
            out.append(None)
            continue
        prev = x if prev is None else alpha * x + (1 - alpha) * prev
        out.append(prev)
    return out


def _norm(xs: Sequence[float], length: int, clamp: float
          ) -> list[float | None]:
    """Each value against its own recent spread, bounded.

    A zero standard deviation means the window never moved; the ratio is
    undefined there rather than infinite, so it is None.
    """
    sd = stdev(xs, length)
    out: list[float | None] = []
    for x, s in zip(xs, sd):
        if s is None or s == 0:
            out.append(None)
            continue
        out.append(max(-clamp, min(clamp, x / s)))
    return out


def tick_ppo(bars: Sequence[Any], *, smooth: int = SMOOTH,
             signal: int = SIGNAL, ad_weight: float = AD_WEIGHT,
             mom_weight: float = MOM_WEIGHT,
             stdev_len: int = STDEV_LEN,
             clamp: float = CLAMP) -> dict[str, Any]:
    """The whole indicator, one value per bar.

    Returns lists the same length as `bars`, with None where there is not
    yet enough history. Drawing a gap is the honest thing; drawing zero
    would put a flat line where there is no reading at all.
    """
    n = len(bars)
    if n == 0:
        return {"ppo": [], "signal": [], "hist": [], "ready": False,
                "warmup": stdev_len}

    ad = [money_flow_volume(b) for b in bars]
    mom = signed_volume(bars)

    ad_n = _norm(ad, stdev_len, clamp)
    mom_n = _norm(mom, stdev_len, clamp)

    combined: list[float | None] = []
    for a, m in zip(ad_n, mom_n):
        if a is None and m is None:
            combined.append(None)
        else:
            combined.append((a or 0.0) * ad_weight + (m or 0.0) * mom_weight)

    ppo = ema(combined, smooth)
    sig = ema(ppo, signal)
    hist = [None if (p is None or s is None) else p - s
            for p, s in zip(ppo, sig)]

    ready = any(h is not None for h in hist)
    return {
        "ppo": [_r(x) for x in ppo],
        "signal": [_r(x) for x in sig],
        "hist": [_r(x) for x in hist],
        "ad_norm": [_r(x) for x in ad_n],
        "mom_norm": [_r(x) for x in mom_n],
        "ready": ready,
        "warmup": stdev_len,
        "settings": {"smooth": smooth, "signal": signal,
                     "ad_weight": ad_weight, "mom_weight": mom_weight,
                     "stdev_len": stdev_len, "clamp": clamp},
    }


def _f(bar: Any, short: str, long: str) -> float:
    """Read a field from a dict bar or a Candle, whichever turned up."""
    if isinstance(bar, dict):
        v = bar.get(short)
        if v is None:
            v = bar.get(long)
    else:
        v = getattr(bar, long, None)
        if v is None:
            v = getattr(bar, short, None)
    try:
        return float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 4)
