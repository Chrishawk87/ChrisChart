"""Completing the candle that is still open, as a distribution.

THE REFRAME

Nothing here predicts a candle. At minute seven of a fifteen minute bar
the open, the high so far, the low so far and the volume so far are all
known exactly -- the only unknown is the remaining eight minutes. So the
question is not "what will this bar do", it is "given everything already
printed, where does the close land". That is a much smaller question, and
its uncertainty collapses as the bar fills.

It also dissolves the horizon problem. Order-book microstructure is worth
something over roughly the next thirty seconds and very little over the
next fifteen minutes. A fifteen minute bar is thirty of those windows. So
the read is not stretched to cover the bar; it is applied to the time
actually remaining and integrated forward.

HOW

Monte Carlo, because the quantity wanted is not just the close but the
high and low too, and those depend on the whole path rather than its
endpoint. Closed forms for the running maximum of a drifted walk exist and
are easy to get subtly wrong; simulating is exact given the model, and a
few hundred paths over a few hundred steps is nothing.

    remaining      T seconds left in the bar
    volatility     per-second sigma, measured from recent bars
    drift          per-second expected move, from the microstructure read

    close  ~  price * (1 + sum of T steps)
    high   ~  max(high so far, the path's own maximum)
    low    ~  min(low so far, the path's own minimum)

TWO PROJECTIONS, ALWAYS, AND THAT IS THE POINT

Every projection is produced twice: once with the drift the signal implies
and once with no drift at all. The second is the null -- the same
volatility cone, no opinion about direction.

Scored against the bars that follow, the difference between them is a
direct measurement of whether the signal carries any directional
information. It is a far faster test than trading: every bar closes, so
every projection is graded within minutes, and a few hundred bars arrive
in a couple of days rather than the weeks a comparable number of trades
would take.

If the drifted projection is not better calibrated than the null, the
signal has no directional content at that horizon, and no amount of
position sizing will rescue it.
"""

from __future__ import annotations

import math
import random
from typing import Any, Sequence

# Enough paths that the quantiles are stable to well under a basis point,
# and few enough to run on every poll without being noticed.
PATHS = 600

# Steps are one second unless that would be silly; a 1h bar does not need
# 3600 of them for a distribution quoted to two decimals.
MAX_STEPS = 240

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def realised_sigma_bps(bars: Sequence[Any], interval_s: float,
                       lookback: int = 30) -> float:
    """Per-second volatility in basis points, from recent closes.

    Close-to-close rather than a range estimator: the projection is driven
    step by step, and a range estimator would describe a different
    quantity than the one being simulated.
    """
    if interval_s <= 0:
        return 0.0
    closes = []
    for b in bars[-(lookback + 1):]:
        c = _f(b, "c", "close")
        if c > 0:
            closes.append(c)
    if len(closes) < 3:
        return 0.0
    rets = [(closes[i] - closes[i - 1]) / closes[i - 1] * 10_000.0
            for i in range(1, len(closes))]
    n = len(rets)
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / max(1, n - 1)
    per_bar = math.sqrt(max(var, 0.0))
    return per_bar / math.sqrt(interval_s) if interval_s > 0 else 0.0


def drift_from_read(net: float, sigma_bps_s: float, remaining_s: float,
                    k: float = 0.08) -> float:
    """Per-second drift the read implies, in basis points.

    `k` is what the signal claims as a fraction of the REMAINING WINDOW's
    standard deviation -- not per second. That distinction is the whole
    thing, and getting it wrong is silent:

        expected move = k * squash(net) * sigma * sqrt(T)

    A per-second drift accumulates linearly in T while the noise around
    it only grows as sqrt(T), so the same coefficient claims more and more
    as the horizon lengthens. At k = 0.15 per second, a fifteen minute bar
    was being told to expect a 3.4 sigma move -- an assertion no market
    makes and no calibration could survive. The first version of this
    shipped that way, and it read as "the signal is actively hurting" on
    data where the signal was real, because a wildly over-confident drift
    is worse than no drift whichever way it points.

    Stated this way `k` is dimensionless and horizon-invariant: 0.08 means
    "about eight percent of a standard deviation", which is roughly what a
    genuine microstructure edge is worth and small enough that being
    wrong about it costs little.
    """
    if sigma_bps_s <= 0 or remaining_s <= 0:
        return 0.0
    # The vote's net is roughly [-3, 3]; squash it so a huge reading does
    # not produce a drift the market has never delivered.
    squashed = math.tanh(net / 1.5)
    move = k * squashed * sigma_bps_s * math.sqrt(remaining_s)
    return move / remaining_s


def _quantiles(xs: list[float], qs: Sequence[float] = QUANTILES
               ) -> dict[str, float]:
    if not xs:
        return {}
    s = sorted(xs)
    n = len(s)
    out = {}
    for q in qs:
        i = min(n - 1, max(0, int(round(q * (n - 1)))))
        out[f"q{int(q * 100)}"] = s[i]
    return out


def _simulate(price: float, high_so_far: float, low_so_far: float,
              steps: int, dt: float, sigma_bps_s: float, drift_bps_s: float,
              paths: int, rng: random.Random) -> dict[str, Any]:
    """Run the remaining time forward `paths` times."""
    sd = sigma_bps_s * math.sqrt(dt)
    mu = drift_bps_s * dt
    closes, highs, lows = [], [], []
    for _ in range(paths):
        x = 0.0
        hi, lo = 0.0, 0.0
        for _ in range(steps):
            x += rng.gauss(mu, sd)
            if x > hi:
                hi = x
            if x < lo:
                lo = x
        closes.append(price * (1 + x / 10_000.0))
        highs.append(max(high_so_far, price * (1 + hi / 10_000.0)))
        lows.append(min(low_so_far, price * (1 + lo / 10_000.0)))
    up = sum(1 for c in closes if c > price) / len(closes)
    return {"close": _quantiles(closes), "high": _quantiles(highs),
            "low": _quantiles(lows), "p_up": round(up, 4),
            "samples": closes}


def project(*, price: float, open_px: float, high_so_far: float,
            low_so_far: float, remaining_s: float, sigma_bps_s: float,
            net: float = 0.0, k: float = 0.08, paths: int = PATHS,
            seed: int | None = None) -> dict[str, Any]:
    """Where this bar closes, with the signal and without it.

    Returns both projections and the numbers behind them. A bar with no
    time left, or a market with no measurable volatility, returns a
    degenerate projection rather than a fabricated one.
    """
    rng = random.Random(seed)
    out: dict[str, Any] = {
        "price": price, "open": open_px, "high_so_far": high_so_far,
        "low_so_far": low_so_far, "remaining_s": round(remaining_s, 1),
        "sigma_bps_s": round(sigma_bps_s, 5), "net": round(net, 4), "k": k,
    }

    if remaining_s <= 0 or sigma_bps_s <= 0 or price <= 0:
        out["ready"] = False
        out["detail"] = ("the bar has no time left" if remaining_s <= 0
                         else "no measurable volatility yet")
        return out

    steps = max(1, min(MAX_STEPS, int(remaining_s)))
    dt = remaining_s / steps
    drift = drift_from_read(net, sigma_bps_s, remaining_s, k)

    with_signal = _simulate(price, high_so_far, low_so_far, steps, dt,
                            sigma_bps_s, drift, paths, rng)
    # The null runs on its own stream, so the two are not coupled by
    # sharing draws -- which would make the comparison look tighter than
    # it is.
    flat = _simulate(price, high_so_far, low_so_far, steps, dt,
                     sigma_bps_s, 0.0, paths,
                     random.Random((seed or 0) + 1))

    out.update({
        "ready": True,
        "drift_bps_s": round(drift, 8),
        "expected_move_bps": round(drift * remaining_s, 3),
        "expected_move_sd": round(
            (drift * remaining_s) / (sigma_bps_s * math.sqrt(remaining_s)), 4)
        if sigma_bps_s > 0 else 0.0,
        "signal": {k2: v for k2, v in with_signal.items() if k2 != "samples"},
        "null": {k2: v for k2, v in flat.items() if k2 != "samples"},
        "_signal_samples": with_signal["samples"],
        "_null_samples": flat["samples"],
    })
    return out


def pit(actual: float, samples: Sequence[float]) -> float | None:
    """Where the outcome landed inside the predicted distribution.

    The probability integral transform: the share of simulated closes
    below what actually happened. A well calibrated model produces PIT
    values spread evenly over [0, 1]; a model that is too confident piles
    them at the ends, and one that is too vague piles them in the middle.

    This is the whole scoring mechanism, and it is worth being clear about
    why it beats counting hits: it grades the SHAPE of the forecast, not
    just its direction, so an over-confident model is caught even on the
    bars it happens to get right.
    """
    if not samples:
        return None
    n = len(samples)
    below = sum(1 for s in samples if s < actual)
    return round(below / n, 6)


def coverage(pits: Sequence[float], width: float = 0.5) -> float | None:
    """Share of outcomes that fell inside the central band of that width.

    A calibrated 50% band contains half of them. More is a model hedging;
    less is one claiming more than it knows.
    """
    vals = [p for p in pits if p is not None]
    if not vals:
        return None
    lo, hi = 0.5 - width / 2, 0.5 + width / 2
    return round(sum(1 for p in vals if lo <= p <= hi) / len(vals), 4)


def reliability(pits: Sequence[float], bins: int = 10) -> list[dict]:
    """The PIT histogram -- flat is calibrated.

    Drawn rather than summarised because the SHAPE says what is wrong: a
    U means over-confident, a hump means under-confident, and a lean means
    the drift is pointing the wrong way.
    """
    vals = [p for p in pits if p is not None]
    if not vals:
        return []
    counts = [0] * bins
    for p in vals:
        i = min(bins - 1, max(0, int(p * bins)))
        counts[i] += 1
    n = len(vals)
    return [{"from": round(i / bins, 2), "to": round((i + 1) / bins, 2),
             "n": c, "share": round(c / n, 4),
             "expected": round(1 / bins, 4)}
            for i, c in enumerate(counts)]


def calibration_score(pits: Sequence[float]) -> dict[str, Any]:
    """How far the PIT histogram is from flat, and what that means.

    The statistic is the mean absolute deviation from uniform across the
    bins. Zero is perfect; it rises as the forecast gets worse in either
    direction, which is why the histogram is reported beside it.
    """
    vals = [p for p in pits if p is not None]
    n = len(vals)
    if n < 20:
        return {"n": n, "ready": False,
                "note": f"{n} resolved — too few to say anything. "
                        f"A few hundred bars is a couple of days."}
    rows = reliability(vals)
    dev = sum(abs(r["share"] - r["expected"]) for r in rows) / len(rows)
    c50 = coverage(vals, 0.5)
    c80 = coverage(vals, 0.8)
    mean = sum(vals) / n

    if dev < 0.02:
        verdict = "well calibrated — the bands mean what they say"
    elif dev < 0.05:
        verdict = "roughly calibrated"
    else:
        ends = rows[0]["share"] + rows[-1]["share"]
        verdict = ("over-confident — outcomes land outside the bands far "
                   "more often than they should" if ends > 0.3 else
                   "poorly calibrated")

    lean = ""
    if mean > 0.56:
        lean = (" It also leans low: closes are landing above the "
                "projection more often than not, so the drift is pointing "
                "the wrong way or is too small.")
    elif mean < 0.44:
        lean = (" It also leans high: closes are landing below the "
                "projection more often than not.")

    return {"n": n, "ready": True, "deviation": round(dev, 4),
            "coverage_50": c50, "coverage_80": c80,
            "mean_pit": round(mean, 4), "histogram": rows,
            "verdict": verdict + lean}


def compare(signal_pits: Sequence[float], null_pits: Sequence[float]
            ) -> dict[str, Any]:
    """Does the signal beat the same cone with no opinion in it?

    This is the question the whole exercise exists to answer, and it is
    worth stating what a negative answer means: if the drifted projection
    is no better calibrated than the flat one, the read carries no
    directional information at this horizon. That is a finding about the
    signal, not about the projection.
    """
    a = calibration_score(signal_pits)
    b = calibration_score(null_pits)
    if not (a.get("ready") and b.get("ready")):
        return {"ready": False, "signal": a, "null": b,
                "note": "not enough resolved bars to compare yet"}

    gap = b["deviation"] - a["deviation"]
    if abs(gap) < 0.005:
        verdict = ("The signal's projection is no better calibrated than "
                   "the one with no opinion in it. On this evidence the "
                   "read carries no directional information at this "
                   "horizon.")
    elif gap > 0:
        verdict = (f"The signal's projection is better calibrated than the "
                   f"null by {gap:.3f}. The read is carrying directional "
                   f"information.")
    else:
        verdict = (f"The signal's projection is WORSE than the null by "
                   f"{abs(gap):.3f} — the drift is actively hurting. Check "
                   f"its sign before anything else.")

    return {"ready": True, "signal": a, "null": b,
            "gap": round(gap, 4), "verdict": verdict}


def _f(bar: Any, short: str, long: str) -> float:
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


# --------------------------------------------------------------------------
# does it predict the direction at all
# --------------------------------------------------------------------------

def _wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    z2 = z * z
    den = 1.0 + z2 / n
    mid = (p + z2 / (2 * n)) / den
    half = (z / den) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (max(0.0, mid - half), min(1.0, mid + half))


def directional_edge(rows: Sequence[dict]) -> dict[str, Any]:
    """When the read leans one way, does the bar finish that way?

    A separate question from whether the bands are honest, and the one
    actually being asked. PIT calibration grades the SHAPE of the
    forecast, so a small directional edge barely moves it -- a drift of a
    few percent of a standard deviation leaves the histogram almost
    exactly as flat as it was. Measuring the bands and concluding the
    signal is worthless would be a mistake of instrument, not of data.

    This is a plain binomial on the sign, which is far more powerful for
    the purpose: of the bars where the read leaned, how many closed that
    way? Fifty percent is a coin. The interval says whether the gap from
    fifty is real.

    `rows` need `net`, `price` and `close_px`.
    """
    used = [r for r in rows
            if r.get("close_px") and r.get("price")
            and abs(float(r.get("net") or 0.0)) > 1e-9]
    n = len(used)
    if n < 30:
        return {"n": n, "ready": False,
                "note": f"{n} bars where the read leaned — too few. "
                        f"Thirty is the floor and a few hundred is a "
                        f"finding."}

    right = 0
    for r in used:
        up = float(r["net"]) > 0
        moved_up = float(r["close_px"]) > float(r["price"])
        right += 1 if up == moved_up else 0

    rate = right / n
    lo, hi = _wilson(right, n)
    real = lo > 0.5 or hi < 0.5

    if not real:
        verdict = (f"{rate:.1%} of {n} bars finished the way the read "
                   f"leaned. The honest range is {lo:.1%}–{hi:.1%}, which "
                   f"contains 50% — on this evidence the read does not "
                   f"call direction at this horizon.")
    elif lo > 0.5:
        verdict = (f"{rate:.1%} of {n} bars finished the way the read "
                   f"leaned, and the whole {lo:.1%}–{hi:.1%} range is "
                   f"above a coin. The read carries direction.")
    else:
        verdict = (f"{rate:.1%} of {n} bars finished the way the read "
                   f"leaned — the whole {lo:.1%}–{hi:.1%} range is BELOW "
                   f"a coin. It is predicting, with the sign inverted.")

    return {"n": n, "ready": True, "rate": round(rate, 4),
            "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "real": real, "edge_pts": round((rate - 0.5) * 100, 2),
            "verdict": verdict}
