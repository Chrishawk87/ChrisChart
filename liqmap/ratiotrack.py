"""What the push ratio is DOING, not what sign it happens to have.

Chris, looking at the 5-minute candle's reading of +0.0414 on 1,170
contracts:

    "there is a part of this that works better then the direction
    itself ... if we can actually track the 0.0414 number accurately
    we will have a serious advantage not just the color the number
    is which turns red and green but what the number is actually
    doing going up or down"

HE IS RIGHT, AND THE REASON IS IN HIS OWN EXAMPLE.

    +0.0414 is four percent of the way off dead even. Its SIGN is a
    coin flip -- a few contracts either way repaints the number red
    or green while nothing about the market changed. The same defect
    killed the OFI sign, which flipped 3.17 times a bar against a
    zero deadband.

    "Net buying is building" is a different claim entirely, and it
    does not care whether the ratio is sitting at +0.04 or -0.04.

BUT THE NUMBER CANNOT BE DIFFERENCED NAIVELY. Three reasons, and
all three produce confident garbage if ignored.

    IT IS A CUMULATIVE MEAN, SO IT SELF-DAMPS. The bar's ratio is
    every contract since the boundary divided by all of them. At
    1,170 contracts one more 50-lot moves it by about 0.04; at 60
    contracts the same 50-lot moves it by 0.5. So the number goes
    quiet as the bar fills REGARDLESS of what the tape is doing,
    and a slope read off it late in the bar understates a genuine
    reversal -- the flow can flip completely while the cumulative
    line barely bends. That is arithmetic, not a market reading.

    So the marginal ratio is carried alongside it: the same
    quantity over the last `marginal_s` seconds only. The gap
    between marginal and cumulative is what is actually pulling the
    number, and it is the leading half of this whole idea. Rising
    cumulative with a marginal BELOW it means the rise is already
    over; the average just has not caught up.

    IT RESETS AT EVERY BOUNDARY. The window clears each 5 minutes,
    so the ratio drops to zero and starts again. Differenced across
    that edge it reads as a violent move every 300 seconds, twelve
    times an hour, forever. Samples from the previous bar are
    therefore dropped rather than smoothed.

    IT IS MEANINGLESS WHEN THE BAR IS NEARLY EMPTY. The first
    handful of contracts can put the ratio at +1.0 and then -0.6
    within seconds. A slope fitted there is fitting noise, and the
    noise is largest exactly when the number looks most dramatic.
    `ready` is false until `min_vol` contracts have crossed.

THE DEADBAND IS A GUESS UNTIL IT IS MEASURED ON HIS TAPE. A zero
deadband would report "rising" and "falling" alternately all bar,
which is the OFI's failure repeated with a new number. The default
here is deliberately conservative and is exposed as an argument so
it can be set from data rather than from this docstring.

Nothing in this module places, sizes, routes or times an order.
"""

from __future__ import annotations

from collections import deque

PERIOD_S = 300.0        # the candle this tracks, matching kinetic_bar
MARGINAL_S = 60.0       # the trailing window the leading read uses
SLOPE_S = 120.0         # how far back the fitted slope looks
MIN_VOL = 100.0         # contracts before any reading is offered
MIN_SAMPLES = 5         # a line through fewer points is not a trend
# Ratio points a minute. The ratio spans -1..+1, so 0.05/min is a
# twentieth of the full scale in sixty seconds: a real move, not a
# flicker. PROVISIONAL -- set from measured tape, not from here.
DEADBAND = 0.05


class RatioTrack:
    """The bar's push ratio, its slope, and where it is being pulled.

    Fed the same prints the aligned window is fed. O(1) amortised:
    the fit is kept as running sums, so no window is ever walked.
    """

    __slots__ = ("period_s", "marginal_s", "slope_s", "min_vol",
                 "deadband", "_bar",
                 "_buy", "_sell",
                 "_mq", "_mbuy", "_msell",
                 "_sq", "_n", "_st", "_sr", "_stt", "_str",
                 "_last_ts", "bars")

    def __init__(self, period_s: float = PERIOD_S,
                 marginal_s: float = MARGINAL_S,
                 slope_s: float = SLOPE_S,
                 min_vol: float = MIN_VOL,
                 deadband: float = DEADBAND) -> None:
        self.period_s = max(1.0, float(period_s))
        self.marginal_s = max(1.0, float(marginal_s))
        self.slope_s = max(1.0, float(slope_s))
        self.min_vol = max(0.0, float(min_vol))
        self.deadband = max(0.0, float(deadband))
        self._bar: int | None = None
        # Cumulative, cleared at each boundary. This IS the number
        # on the panel.
        self._buy = 0.0
        self._sell = 0.0
        # Marginal: a rolling window that does NOT clear, because
        # what the tape is doing right now does not stop mattering
        # because a clock ticked.
        self._mq: deque = deque()
        self._mbuy = 0.0
        self._msell = 0.0
        # The fit over the cumulative ratio, within this bar only.
        self._sq: deque = deque()
        self._n = 0
        self._st = self._sr = self._stt = self._str = 0.0
        self._last_ts = 0.0
        self.bars = 0

    # -- feeding ---------------------------------------------------------

    def on_trade(self, ts: float, size: float, side: str) -> None:
        """One print. 'B' lifted the offer, 'A' hit the bid.

        An unattributed print ('N') casts no vote and is not an
        error: it simply is not evidence about who crossed.
        """
        try:
            t = float(ts or 0.0)
            v = float(size or 0.0)
        except (TypeError, ValueError):
            return
        if t <= 0 or v <= 0:
            return
        s = (side or "").upper()[:1]
        if s not in ("B", "A"):
            return

        bar = int(t // self.period_s)
        if self._bar is None:
            self._bar = bar
        elif bar != self._bar:
            # A NEW CANDLE. The cumulative side and the fit belong to
            # the bar that ended; carrying either across the boundary
            # would report the reset itself as a move.
            self._bar = bar
            self._buy = self._sell = 0.0
            self._sq.clear()
            self._n = 0
            self._st = self._sr = self._stt = self._str = 0.0
            self.bars += 1

        self._last_ts = t
        if s == "B":
            self._buy += v
            self._mbuy += v
        else:
            self._sell += v
            self._msell += v
        self._mq.append((t, v if s == "B" else 0.0,
                         v if s == "A" else 0.0))

        cut = t - self.marginal_s
        q = self._mq
        while q and q[0][0] < cut:
            _t, b, a = q.popleft()
            self._mbuy -= b
            self._msell -= a
        if not q:
            # Hundreds of thousands of adds and subtracts leave
            # float dust; an empty window must read exactly zero or
            # the marginal reports a side on no data.
            self._mbuy = self._msell = 0.0

        self._push_sample(t)

    def _push_sample(self, t: float) -> None:
        r = self.ratio
        self._sq.append((t, r))
        self._n += 1
        self._st += t
        self._sr += r
        self._stt += t * t
        self._str += t * r
        cut = t - self.slope_s
        q = self._sq
        while q and q[0][0] < cut:
            ot, orr = q.popleft()
            self._n -= 1
            self._st -= ot
            self._sr -= orr
            self._stt -= ot * ot
            self._str -= ot * orr
        if not q:
            self._n = 0
            self._st = self._sr = self._stt = self._str = 0.0

    # -- reading ---------------------------------------------------------

    @property
    def volume(self) -> float:
        """Attributable contracts in this candle so far."""
        return self._buy + self._sell

    @property
    def ratio(self) -> float:
        """THE NUMBER ON THE PANEL. (buy - sell) / (buy + sell)."""
        v = self.volume
        if v <= 0:
            return 0.0
        return (self._buy - self._sell) / v

    @property
    def marginal(self) -> float:
        """The same ratio over the last `marginal_s` seconds only."""
        v = self._mbuy + self._msell
        if v <= 0:
            return 0.0
        return (self._mbuy - self._msell) / v

    @property
    def pull(self) -> float:
        """Marginal minus cumulative: what is moving the number.

        Positive means recent flow is more buy-sided than the bar's
        average, so the average is being pulled up -- whatever sign
        it currently shows. This leads the slope, because the slope
        is the average's response and this is the input to it.
        """
        return self.marginal - self.ratio

    @property
    def ready(self) -> bool:
        """Is there enough in this candle for any of it to mean
        anything? Early in a bar the ratio swings wildly on a
        handful of contracts and a fit through it is fitting noise."""
        return self.volume >= self.min_vol and self._n >= MIN_SAMPLES

    @property
    def slope_per_min(self) -> float:
        """Least-squares slope of the ratio, in ratio points a minute.

        Zero rather than a number when there is too little to fit:
        two points cannot describe a trend, and a degenerate window
        -- every print at one instant -- has an infinite slope.
        """
        if not self.ready:
            return 0.0
        n = self._n
        denom = n * self._stt - self._st * self._st
        if denom <= 0:
            return 0.0
        return ((n * self._str - self._st * self._sr) / denom) * 60.0

    @property
    def trend(self) -> str:
        """rising / falling / steady, with a deadband.

        Without the deadband this alternates all bar and is the OFI
        sign problem wearing a new number.
        """
        if not self.ready:
            return "unknown"
        s = self.slope_per_min
        if s > self.deadband:
            return "rising"
        if s < -self.deadband:
            return "falling"
        return "steady"

    @property
    def damping(self) -> float:
        """How much one more contract can still move the number.

        1/n, roughly: the sensitivity of a cumulative mean. Near the
        end of a busy bar this is tiny, which is WHY the slope goes
        quiet and why `pull` is the one to watch there. Reported so
        the panel can say that rather than imply the tape calmed
        down.
        """
        v = self.volume
        return (1.0 / v) if v > 0 else 0.0

    @property
    def say(self) -> str:
        if not self.ready:
            need = max(0.0, self.min_vol - self.volume)
            return (f"{int(self.volume):,} contracts in this candle -- "
                    f"{int(need):,} more before the trend means anything")
        t = self.trend
        s = self.slope_per_min
        word = {"rising": "net buying is building",
                "falling": "net selling is building",
                "steady": "the balance is holding"}[t]
        out = f"{word} ({s:+.3f} a minute)"
        p = self.pull
        if t != "steady" and p * s < 0:
            out += " -- but the last minute is pulling the other way"
        return out

    def to_dict(self) -> dict:
        return {
            "ratio": round(self.ratio, 4),
            "marginal": round(self.marginal, 4),
            "pull": round(self.pull, 4),
            "slope_per_min": round(self.slope_per_min, 4),
            "trend": self.trend,
            "ready": self.ready,
            "volume": round(self.volume, 1),
            "damping": round(self.damping, 6),
            "deadband": self.deadband,
            "marginal_s": self.marginal_s,
            "slope_s": self.slope_s,
            "say": self.say,
        }
