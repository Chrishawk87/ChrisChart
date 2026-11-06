"""Three ways to steady the tick bar, and what each one costs.

Chris:

    "it looks like the tick bar is going to be the best thing i have
    for a true up and down directional signal the only thing i would
    like to do is smooth out the tick bar even at the lower tick
    ranges just for stability and holding a trade longer with pure
    accuracy"

SMOOTHING AND "PURE ACCURACY" PULL AGAINST EACH OTHER, and which
way depends entirely on the method. This is the one thing worth
being blunt about before any of it is tuned, because the tick bar
was chosen in the first place for being LEADING:

    "tick charts are leading and we can use that information to make
    better and more timely entries and exits"

AN AVERAGE IS SMOOTH BECAUSE IT IS LATE. That is not a tuning
problem, it is what an average is. A threshold is stable for a
different reason -- it is not late at all, it simply refuses to
call a small move a reversal. Both hold a trade longer. Only one of
them gives up the leading edge to do it.

    THRESHOLD  direction holds until price travels `flip_ticks`
               back from the best level reached since the last
               flip. ZERO time lag: a genuine reversal registers on
               the print that completes it, and a wiggle never
               registers at all. Stability is a distance you set,
               not a delay you absorb. The cost is honest and
               different: every flip gives back `flip_ticks` from
               the extreme, so it is late in PRICE even though it
               is immediate in TIME.

    HEIKIN     Heikin-Ashi bars. ha_close averages the bar's four
               prices and ha_open carries the previous bar forward,
               so each candle is partly the one before it. That is
               what makes a run of them stay one colour through
               noise, and it is also exactly why it lags: about a
               bar, by construction, and it cannot be tuned out
               without ceasing to be Heikin-Ashi. The drawn bar
               changes, which is the one method here you can SEE.

    EMA        price against an N-bar exponential average.
               Smoothest and most familiar, and the latest: an
               N-bar average sits on average N/2 bars behind. Near
               the line it whipsaws, which is the failure the other
               two are built to avoid, so the slope is reported
               alongside for a second opinion.

    RAW        close against open of the bar itself. No smoothing.
               THE CONTROL -- without it there is no way to say
               whether any of the three helped, and every previous
               idea in this engine that went in without a control
               had to be taken back out.

Nothing in this module places, sizes, routes or times an order.
"""

from __future__ import annotations

TICK = 0.25
MODES = ("raw", "threshold", "heikin", "ema")
DEFAULT_FLIP_TICKS = 4.0
DEFAULT_EMA_N = 9


# Below this, two prices are the same price. Heikin-Ashi averages
# four quarter-point prices and compares the result to another
# average, so exact equality lands a few ulps apart: the first bar
# of a flat open/close read "down" off a difference of 1e-14. Prices
# here are quantised to a quarter point, so a thousandth of a tick
# is far below anything real and far above float dust.
EPS = TICK / 1000.0


def _dir(a: float, b: float) -> str:
    if a <= 0 or b <= 0:
        return "flat"
    if b - a > EPS:
        return "up"
    if a - b > EPS:
        return "down"
    return "flat"


class Smoother:
    """One steadied reading of the tick bar's direction.

    Fed closed bars, and the forming bar on every print, so the
    live direction does not wait for a bar that may be thousands of
    prints from closing. The caller never needs to know which mode
    it asked for.
    """

    __slots__ = ("mode", "flip_ticks", "ema_n",
                 "_dir", "_extreme", "_flip_px",
                 "_ha_open", "_ha_close", "_ha_high", "_ha_low",
                 "_ema", "_ema_prev", "_px",
                 "bars", "flips")

    def __init__(self, mode: str = "raw",
                 flip_ticks: float = DEFAULT_FLIP_TICKS,
                 ema_n: int = DEFAULT_EMA_N) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.flip_ticks = max(1.0, float(flip_ticks))
        self.ema_n = max(2, int(ema_n))
        self._dir = ""
        self._extreme = 0.0      # best level since the last flip
        self._flip_px = 0.0      # where the last flip happened
        self._ha_open = 0.0
        self._ha_close = 0.0
        self._ha_high = 0.0
        self._ha_low = 0.0
        self._ema = 0.0
        self._ema_prev = 0.0
        self._px = 0.0
        self.bars = 0
        self.flips = 0

    # -- feeding ---------------------------------------------------------

    def on_price(self, px: float) -> None:
        """A print. THRESHOLD acts here; the averages cannot.

        This asymmetry is the whole difference between the methods
        stated in code: a threshold can be crossed by the print
        that crosses it, while an average has nothing to say until
        a bar closes and gives it a value to average.
        """
        try:
            p = float(px or 0.0)
        except (TypeError, ValueError):
            return
        if p <= 0:
            return
        self._px = p
        if self.mode == "threshold":
            self._threshold(p, p, p)

    def on_bar(self, o: float, h: float, l: float, c: float) -> None:
        """A tick bar just closed."""
        try:
            o, h, l, c = (float(o or 0.0), float(h or 0.0),
                          float(l or 0.0), float(c or 0.0))
        except (TypeError, ValueError):
            return
        if min(o, h, l, c) <= 0:
            return
        self.bars += 1
        self._px = c

        # Heikin-Ashi. Seeded from the bar itself; carried forward
        # after, which is where both the smoothing and the lag come
        # from -- they are the same property.
        ha_c = (o + h + l + c) / 4.0
        ha_o = ((self._ha_open + self._ha_close) / 2.0
                if self._ha_open > 0 else (o + c) / 2.0)
        self._ha_open, self._ha_close = ha_o, ha_c
        self._ha_high = max(h, ha_o, ha_c)
        self._ha_low = min(l, ha_o, ha_c)

        # EMA of closes.
        k = 2.0 / (self.ema_n + 1.0)
        self._ema_prev = self._ema
        self._ema = c if self._ema <= 0 else (c - self._ema) * k + self._ema

        if self.mode == "threshold":
            self._threshold(h, l, c)

    def _threshold(self, h: float, l: float, c: float) -> None:
        """Hysteresis. The direction stands until price gives back
        `flip_ticks` from the extreme reached since the last flip."""
        gap = self.flip_ticks * TICK
        if not self._dir:
            # Nothing established yet. The first move of a full
            # threshold in either direction sets it -- a direction
            # claimed before that would be claimed on one print.
            if self._extreme <= 0:
                self._extreme = c
                self._flip_px = c
                return
            if c >= self._extreme + gap:
                self._dir, self._extreme, self._flip_px = "up", h, c
                return
            if c <= self._extreme - gap:
                self._dir, self._extreme, self._flip_px = "down", l, c
                return
            # Track the far side while undecided, so the first call
            # is measured from a real extreme rather than wherever
            # the feed happened to start.
            self._extreme = max(self._extreme, h) if c > self._extreme \
                else min(self._extreme, l)
            return

        if self._dir == "up":
            self._extreme = max(self._extreme, h)
            if c <= self._extreme - gap:
                self._dir, self._extreme, self._flip_px = "down", l, c
                self.flips += 1
        else:
            self._extreme = min(self._extreme, l)
            if c >= self._extreme + gap:
                self._dir, self._extreme, self._flip_px = "up", h, c
                self.flips += 1

    # -- reading ---------------------------------------------------------

    @property
    def direction(self) -> str:
        m = self.mode
        if m == "threshold":
            return self._dir or "flat"
        if m == "heikin":
            return _dir(self._ha_open, self._ha_close)
        if m == "ema":
            return _dir(self._ema, self._px)
        return "flat"       # raw is read from the bar by the caller

    @property
    def ema(self) -> float:
        return round(self._ema, 4) if self._ema > 0 else 0.0

    @property
    def ema_slope(self) -> str:
        """A second opinion for the mode that whipsaws on its own
        line. Price can sit either side of a flat average all day."""
        return _dir(self._ema_prev, self._ema)

    @property
    def give_back_ticks(self) -> float:
        """What a THRESHOLD flip costs in price.

        It is immediate in time and late in price, and that is the
        trade this mode makes. Saying the number out loud stops it
        being discovered later as a surprise in the log.
        """
        if self.mode != "threshold" or not self._dir or self._px <= 0:
            return 0.0
        return round(abs(self._extreme - self._px) / TICK, 2)

    def ha_bar(self) -> dict:
        if self._ha_open <= 0:
            return {}
        return {"open": round(self._ha_open, 4),
                "high": round(self._ha_high, 4),
                "low": round(self._ha_low, 4),
                "close": round(self._ha_close, 4),
                "dir": _dir(self._ha_open, self._ha_close)}

    @property
    def say(self) -> str:
        d = self.direction
        if self.mode == "raw":
            return "unsmoothed -- the control"
        if d == "flat":
            if self.mode == "threshold":
                return (f"no direction yet -- needs {self.flip_ticks:.0f} "
                        f"ticks of travel to call one")
            return "no direction yet"
        if self.mode == "threshold":
            return (f"{d}, and it holds until price gives back "
                    f"{self.flip_ticks:.0f} ticks from the extreme "
                    f"({self.give_back_ticks:.1f} back so far)")
        if self.mode == "heikin":
            return f"{d} -- Heikin-Ashi, about a bar behind"
        return (f"{d} -- price against a {self.ema_n}-bar average, "
                f"which trails by about {self.ema_n // 2} bars")

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "dir": self.direction,
            "flip_ticks": self.flip_ticks,
            "ema_n": self.ema_n,
            "ema": self.ema,
            "ema_slope": self.ema_slope,
            "ha": self.ha_bar(),
            "extreme": round(self._extreme, 4) if self._extreme else 0.0,
            "give_back_ticks": self.give_back_ticks,
            "bars": self.bars,
            "flips": self.flips,
            "say": self.say,
        }
