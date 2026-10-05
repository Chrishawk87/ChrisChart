"""Which way the on-candle move is travelling: extending, or pulling back.

THE THING THE PANEL WAS MISSING

The read has always shown what the move IS -- a number on the candle. It
has never shown which way that number is going. Those are different
questions and only the second one is about right now:

    +3t and rising   the candle is extending, buyers still pressing
    +3t and falling  the move is being given back, this is a pullback
    -3t and falling  extending, the other way
    -3t and rising   a down candle retracing

The level is the same in the first two cases and they are opposite
situations. A snapshot cannot tell them apart, which is why watching the
number felt like it was telling you something the panel would not say.

TWO MEASURES, AND THE SECOND IS THE STRONGER ONE

    SLOPE is the sampled one: how much the move has changed over the last
    few seconds. It is what you watch, and it depends on how often the
    read happened to run.

    GIVEBACK is exact and needs no samples at all: how far price has come
    off the best point this candle reached, in the direction it was
    going. A candle that ran to +9t and sits at +3t has given back
    two-thirds of its move, and that is true however often anyone looked.

Both are reported. When they disagree, the giveback is the one to believe,
because the slope can miss a spike that happened between two samples.

WHAT WOULD MAKE THIS LIE

    A BAR ROLL. When the candle rolls, the move resets to about zero. A
    slope measured across that boundary reports the biggest retracement of
    the day, every single bar. The track is keyed to the bar and cleared
    when it changes.

    A SAMPLE COUNT INSTEAD OF A CLOCK. "The change over the last ten
    reads" means ten seconds on a quiet tape and two on a busy one, so
    the same slope would mean different things exactly when it mattered.
    The window is wall-clock and the span actually covered is reported.

    A THRESHOLD OF ZERO. Price always moves a little, so any non-zero
    change would always be a direction. Below one increment of the thing
    being traded -- one tick, one point, one pip -- nothing has happened.

NOTHING HERE PREDICTS ANYTHING. It reports what the number is doing.
Whether a rising number keeps rising is a separate question, and the only
honest way to answer it is to measure it against a shuffled control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

# How far back the slope looks, and the least it needs before it will call
# a direction. Two samples 200ms apart are not a trend.
WINDOW_S = 6.0
MIN_SPAN_S = 1.5
MIN_SAMPLES = 3

# Samples kept per track. At the stream's two-a-second ceiling this is
# about half a minute, which is more than the window ever asks for.
MAX_SAMPLES = 64

# A move is being given back once this much of its best is gone.
GIVEBACK_SHARE = 0.34

State = Literal["extending", "retracing", "flat", "unknown"]

MEANING: dict[str, str] = {
    "extending": "the move is still growing -- the side that owns this "
                 "candle is still pressing",
    "retracing": "the move is being given back -- price is coming off its "
                 "best point on this candle",
    "flat": "the move has not changed by a whole increment either way",
    "unknown": "not enough of this candle has been watched to say",
}


@dataclass
class Travel:
    """What the on-candle move has been doing, in the market's own unit."""

    timeframe: str
    unit: str = "pt"
    bar_ts: float = 0.0

    now: float = 0.0          # the move now, signed, in units
    was: float = 0.0          # the move at the start of the window
    best: float = 0.0         # the furthest this candle got, same direction
    samples: int = 0
    span_s: float = 0.0
    window_s: float = WINDOW_S

    @property
    def change(self) -> float:
        """How much the move has changed. Signed in price terms."""
        return self.now - self.was

    @property
    def extension(self) -> float:
        """Signed so that POSITIVE always means extending.

        A move of -3 going to -5 is extending just as much as +3 going to
        +5, and a reader should not have to work out the sign of the sign.
        The candle's own direction supplies it; with no direction yet, the
        change supplies its own.
        """
        way = 1.0 if self.now > 0 else -1.0 if self.now < 0 else \
            (1.0 if self.change >= 0 else -1.0)
        return self.change * way

    @property
    def giveback(self) -> float:
        """How far off the candle's best point price has come, in units.

        Exact, and needs no samples: it comes from the candle's own high
        and low rather than from how often anyone looked.
        """
        return max(0.0, abs(self.best) - abs(self.now))

    @property
    def giveback_share(self) -> float:
        b = abs(self.best)
        return (self.giveback / b) if b > 0 else 0.0

    @property
    def per_min(self) -> float:
        """Units per minute, at the rate of the window just measured."""
        return (self.extension / self.span_s * 60.0) if self.span_s > 0 \
            else 0.0

    @property
    def enough(self) -> bool:
        return self.samples >= MIN_SAMPLES and self.span_s >= MIN_SPAN_S

    @property
    def state(self) -> State:
        if not self.enough:
            return "unknown"
        # One increment of the thing being traded. Below that, price has
        # wobbled rather than moved.
        if abs(self.change) < 1.0:
            # The slope says nothing, but a candle well off its best is
            # still giving the move back whatever the last six seconds did.
            if self.giveback >= 1.0 and \
                    self.giveback_share >= GIVEBACK_SHARE:
                return "retracing"
            return "flat"
        return "extending" if self.extension > 0 else "retracing"

    def describe(self) -> str:
        if not self.enough:
            return (f"{self.timeframe}: only {self.samples} reads over "
                    f"{self.span_s:.1f}s of this candle -- not enough to "
                    f"say which way the move is going")
        way = MEANING[self.state]
        return (f"{self.timeframe}: {self.now:+.1f}{self.unit} now against "
                f"{self.was:+.1f}{self.unit} {self.span_s:.0f}s ago, best "
                f"{self.best:+.1f}{self.unit} -- {way}")

    def to_dict(self) -> dict:
        return {"timeframe": self.timeframe, "unit": self.unit,
                "now": round(self.now, 2), "was": round(self.was, 2),
                "best": round(self.best, 2),
                "change": round(self.change, 2),
                "extension": round(self.extension, 2),
                "giveback": round(self.giveback, 2),
                "giveback_share": round(self.giveback_share, 3),
                "per_min": round(self.per_min, 2),
                "samples": self.samples, "span_s": round(self.span_s, 2),
                "enough": self.enough, "state": self.state,
                "describe": self.describe()}


@dataclass
class Track:
    """The recent history of one timeframe's move, on one candle."""

    timeframe: str
    bar_ts: float = 0.0
    points: list[tuple[float, float]] = field(default_factory=list)

    def add(self, ts: float, move: float, bar_ts: float) -> None:
        """Record where the move is now.

        A NEW BAR STARTS A NEW TRACK. The move resets to about zero when
        the candle rolls, and a slope measured across that boundary
        reports the biggest retracement of the day every single bar.
        """
        if bar_ts != self.bar_ts:
            self.bar_ts = bar_ts
            self.points = []
        self.points.append((float(ts), float(move)))
        if len(self.points) > MAX_SAMPLES:
            del self.points[:-MAX_SAMPLES]

    def read(self, unit: str = "pt", best: float = 0.0, now: float = 0.0,
             window_s: float = WINDOW_S) -> Travel:
        """What the move has done over the last `window_s` seconds."""
        out = Travel(timeframe=self.timeframe, unit=unit, bar_ts=self.bar_ts,
                     best=best, window_s=window_s)
        if not self.points:
            return out
        end = float(now) if now else self.points[-1][0]
        out.now = self.points[-1][1]

        # The oldest sample still inside the window -- and if the window
        # has nothing but the latest sample in it, the one before that, so
        # a slow poll reports a long span rather than no reading at all.
        cut = end - window_s
        inside = [p for p in self.points if p[0] >= cut]
        if len(inside) < 2:
            inside = self.points[-2:]
        if len(inside) < 2:
            out.samples = len(self.points)
            return out

        out.was = inside[0][1]
        out.span_s = max(0.0, inside[-1][0] - inside[0][0])
        out.samples = len(inside)
        return out


class Tracker:
    """Every market and timeframe being watched. Small and in memory.

    Keyed by (market, timeframe) because the same market read on two
    timeframes is two different moves, and keyed inside each track by the
    bar, so a roll clears it rather than being measured across.
    """

    def __init__(self) -> None:
        self._tracks: dict[tuple[str, str], Track] = {}

    def record(self, coin: str, timeframe: str, ts: float, move: float,
               bar_ts: float) -> Track:
        key = (coin, timeframe)
        t = self._tracks.get(key)
        if t is None:
            t = self._tracks[key] = Track(timeframe=timeframe)
        t.add(ts, move, bar_ts)
        return t

    def read(self, coin: str, timeframe: str, **kw) -> Travel | None:
        t = self._tracks.get((coin, timeframe))
        return t.read(**kw) if t is not None else None

    def forget(self, coin: str) -> int:
        gone = [k for k in self._tracks if k[0] == coin]
        for k in gone:
            del self._tracks[k]
        return len(gone)

    def __len__(self) -> int:
        return len(self._tracks)


def best_of(open_px: float, high: float, low: float, last: float) -> float:
    """The furthest this candle got, in the direction it is going now.

    Signed in price terms, from the open. A candle currently above its
    open is judged against its high; one below, against its low. A candle
    sitting exactly at its open has no direction yet, so it takes whichever
    extreme is further away -- which is the one it will have to give back.
    """
    if open_px <= 0:
        return 0.0
    up, down = high - open_px, low - open_px
    if last > open_px:
        return up
    if last < open_px:
        return down
    return up if abs(up) >= abs(down) else down


def units_moved(open_px: float, last: float, size: float) -> float:
    return ((last - open_px) / size) if size > 0 else 0.0
