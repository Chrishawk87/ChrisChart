"""The Directional Score: momentum and distance from value, in ticks.

Chris proposed:

    D = (Slope * w1) + ((P_current - VWAP) * w2)

THE IDEA IS SOUND AND THE UNITS WERE NOT.

    Slope is price per unit TIME. (P - VWAP) is price. Adding them
    makes w1 and w2 carry a silent unit conversion, so whichever
    term happens to have the larger raw magnitude dominates -- and
    that is decided by whether the slope is measured per second or
    per minute, not by the market. Tuning weights against an
    arbitrary scaling is tuning the scaling.

    So both terms are converted to TICKS before they are weighted:
    the slope becomes ticks per minute, the distance becomes ticks
    from VWAP. The weights are then dimensionless and mean what one
    would want them to mean -- how much to trust momentum against
    how much to trust the distance from value.

THE VWAP ANCHOR IS THE WHOLE VALUE OF THE IDEA.

    Anchored at the session, it is genuinely new information: this
    rule otherwise knows only the direction of two candles and has
    no notion of where price sits relative to what has been paid.

    Anchored at the 5-minute bar, it would be nearly perfectly
    correlated with the slope over that same bar -- one indicator
    with extra arithmetic. So the anchor here is the SESSION, and
    the session is cut at a gap in the tape rather than by a
    calendar, because a replay of arbitrary files has no reliable
    idea of what day it is but a two-hour hole in the prints is
    unmistakable.

THE TWO TERMS CAN DISAGREE, AND THAT IS NOT A FAULT.

    Slope up with price far above VWAP is a strong trend AND a
    stretched one. The formula scores that highest, which is either
    the best or the worst moment to buy depending on the regime.
    That is an empirical question about the sign and size of w2,
    and it is exactly what the holdout is for. Nothing here assumes
    the answer.

Nothing in this module places, sizes, routes or times an order.
"""

from __future__ import annotations

from collections import deque

TICK = 0.25
# A gap this long means the session ended. ES pauses an hour a day;
# two hours is unambiguous without needing a calendar.
SESSION_GAP_S = 7200.0
SLOPE_WINDOW_S = 60.0


class DScore:
    """Rolling slope and session-VWAP distance, both in ticks.

    O(1) amortised: the regression is kept as running sums, so the
    window is never walked.
    """

    __slots__ = ("slope_s", "w_slope", "w_vwap", "gap_s",
                 "_q", "_n", "_st", "_sp", "_stt", "_stp",
                 "_pv", "_v", "_last_ts", "_px", "sessions")

    def __init__(self, slope_s: float = SLOPE_WINDOW_S,
                 w_slope: float = 1.0, w_vwap: float = 1.0,
                 gap_s: float = SESSION_GAP_S) -> None:
        self.slope_s = max(1.0, float(slope_s))
        self.w_slope = float(w_slope)
        self.w_vwap = float(w_vwap)
        self.gap_s = float(gap_s)
        self._q: deque = deque()          # (t, px) inside the window
        self._n = 0
        self._st = self._sp = 0.0         # sum t, sum px
        self._stt = self._stp = 0.0       # sum t*t, sum t*px
        self._pv = self._v = 0.0          # sum px*vol, sum vol
        self._last_ts = 0.0
        self._px = 0.0
        self.sessions = 0

    # -- feeding ---------------------------------------------------------

    def on_trade(self, ts: float, px: float, size: float) -> None:
        """One print. Side is not needed: this measures price, not flow."""
        try:
            t = float(ts or 0.0)
            p = float(px or 0.0)
            v = float(size or 0.0)
        except (TypeError, ValueError):
            return
        if p <= 0 or v <= 0:
            return

        if self._last_ts and (t - self._last_ts) > self.gap_s:
            # A new session. The VWAP anchor resets; carrying
            # yesterday's into today would measure distance from a
            # price nobody is trading against any more.
            self._pv = self._v = 0.0
            self._q.clear()
            self._n = 0
            self._st = self._sp = self._stt = self._stp = 0.0
            self.sessions += 1

        self._last_ts = t
        self._px = p
        self._pv += p * v
        self._v += v

        self._q.append((t, p))
        self._n += 1
        self._st += t
        self._sp += p
        self._stt += t * t
        self._stp += t * p

        cut = t - self.slope_s
        q = self._q
        while q and q[0][0] < cut:
            ot, op = q.popleft()
            self._n -= 1
            self._st -= ot
            self._sp -= op
            self._stt -= ot * ot
            self._stp -= ot * op
        if not q:
            self._n = 0
            self._st = self._sp = self._stt = self._stp = 0.0

    # -- reading ---------------------------------------------------------

    @property
    def vwap(self) -> float:
        return (self._pv / self._v) if self._v > 0 else 0.0

    @property
    def slope_ticks_per_min(self) -> float:
        """Least-squares slope over the window, in ticks a minute.

        Two points cannot describe a trend and a degenerate window
        -- every print at one instant -- has an infinite slope, so
        both return zero rather than a number that will be weighted
        and trusted.
        """
        n = self._n
        if n < 3:
            return 0.0
        denom = n * self._stt - self._st * self._st
        if denom <= 0:
            return 0.0
        per_s = (n * self._stp - self._st * self._sp) / denom
        return (per_s * 60.0) / TICK

    @property
    def vwap_ticks(self) -> float:
        """How far price sits above or below the session VWAP."""
        vw = self.vwap
        if vw <= 0 or self._px <= 0:
            return 0.0
        return (self._px - vw) / TICK

    @property
    def score(self) -> float:
        """D, with both terms in ticks so the weights are comparable."""
        return (self.slope_ticks_per_min * self.w_slope
                + self.vwap_ticks * self.w_vwap)

    @property
    def direction(self) -> str:
        s = self.score
        if s > 0:
            return "up"
        if s < 0:
            return "down"
        return "flat"

    def agrees(self, side: str, threshold: float = 0.0) -> bool:
        """Does the score back `side` by at least `threshold`?

        A FILTER, NOT A TRIGGER. A continuous score crosses any
        threshold repeatedly inside a bar, so using it to open
        trades would reintroduce exactly the churn the rule has
        twice been rewritten to remove. It only ever vetoes.
        """
        s = self.score
        if side == "up":
            return s >= threshold
        if side == "down":
            return s <= -threshold
        return False

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 3),
            "slope_ticks_per_min": round(self.slope_ticks_per_min, 3),
            "vwap": round(self.vwap, 4),
            "vwap_ticks": round(self.vwap_ticks, 2),
            "dir": self.direction,
            "w_slope": self.w_slope,
            "w_vwap": self.w_vwap,
            "slope_s": self.slope_s,
            "sessions": self.sessions,
            "samples": self._n,
        }
