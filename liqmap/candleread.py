"""A directional read on the candle that is still forming.

THE PROBLEM THIS EXISTS FOR

Inside a 15-minute candle there are six or seven things worth knowing at once:
which way aggression is pointing, whether it is being absorbed, where resting
size sits, where price is in the candle's own range, which side of VWAP it is
on, what the higher timeframe is doing, and where the liquidation fuel is.
A person cannot hold all of that and act inside the same candle. A machine
reading it every few seconds can.

So this module does not predict. It reads what is measurable right now,
states each signal separately with its direction, and combines them into one
lean. You still take the trade.

WHY THE OUTPUT IS A LEAN AND NOT A PROBABILITY

It would be easy to print "73% chance this candle closes green". It would also
be a lie, because nothing here has been calibrated against outcomes. The
weights below are priors -- reasonable, argued for in the comments, and
unvalidated.

The honest path from a lean to a probability is measurement, so that is what
`Calibration` does: every read gets logged with its lean, the candle's actual
close gets logged against it, and the hit rate is reported BY LEAN BUCKET from
real outcomes. Until there are enough samples it says so and shows nothing.
A number you measured at 58% is worth more than a number you asserted at 73%.

THE SIGNAL THAT MOST OFTEN POINTS THE WRONG WAY

Heavy aggressive buying looks bullish and frequently is not. If price is not
moving on it, somebody is selling into it in size, and the resolution is
usually against the aggressor once they are done. So absorption INVERTS the
flow signal rather than adding to it, which is the single most important
piece of wiring in this file. Flow and absorption are never both counted in
the same direction.

TIME MATTERS MORE THAN ANY SINGLE SIGNAL

Two minutes into a fifteen-minute candle, everything here can reverse and
usually does. Thirteen minutes in, the same readings are close to settled.
`confidence` scales with elapsed fraction for that reason, and a read taken
in the first third says plainly that it is early.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Sequence

from .flow import Absorption, Book, FlowTape
from .structure import VWAP, Candle, Structure, Zone

Lean = Literal["up", "down", "flat"]

# Unvalidated priors. Each is argued in `read()` where it is applied. They are
# module-level so a calibration run can adjust them without touching logic.
# DEFAULT_WEIGHTS is the frozen original; WEIGHTS is the live, mutable copy
# that tuning writes to, so there is always something to reset back to.
DEFAULT_WEIGHTS = {
    "flow": 1.0,          # net aggression inside this candle
    "position": 0.8,      # where price sits in the candle's own range
    "imbalance": 0.5,     # resting depth, the weakest of the group
    "absorption": 1.4,    # heaviest: it inverts flow and it is hard-won
    "vwap": 0.6,
    "structure": 1.0,     # higher timeframe direction
    "zone": 0.9,          # sitting in a fresh supply or demand zone
    "magnet": 0.7,        # liquidation fuel pulling one way
}

WEIGHTS = dict(DEFAULT_WEIGHTS)

MIN_CALIBRATION_SAMPLES = 40


@dataclass(frozen=True)
class Signal:
    name: str
    direction: Lean
    strength: float        # 0..1 before weighting
    note: str

    @property
    def signed(self) -> float:
        if self.direction == "up":
            return self.strength
        if self.direction == "down":
            return -self.strength
        return 0.0

    def weighted(self) -> float:
        return self.signed * WEIGHTS.get(self.name, 1.0)


@dataclass
class CandleRead:
    coin: str
    interval_s: float
    elapsed_s: float
    open_px: float
    high_px: float
    low_px: float
    last_px: float

    signals: list[Signal] = field(default_factory=list)
    calibration: "Calibration | None" = None

    # -- the candle itself ------------------------------------------------

    @property
    def elapsed_fraction(self) -> float:
        if self.interval_s <= 0:
            return 1.0
        return max(0.0, min(1.0, self.elapsed_s / self.interval_s))

    @property
    def seconds_left(self) -> float:
        return max(0.0, self.interval_s - self.elapsed_s)

    @property
    def change_bps(self) -> float:
        if self.open_px <= 0:
            return 0.0
        return (self.last_px - self.open_px) / self.open_px * 10_000.0

    @property
    def position_in_range(self) -> float:
        """0 at the low, 1 at the high, 0.5 with no range yet."""
        rng = self.high_px - self.low_px
        if rng <= 0:
            return 0.5
        return (self.last_px - self.low_px) / rng

    @property
    def green(self) -> bool:
        return self.last_px > self.open_px

    # -- the read ---------------------------------------------------------

    @property
    def score(self) -> float:
        """Sum of weighted signals, normalised to roughly -1..+1."""
        if not self.signals:
            return 0.0
        total_weight = sum(WEIGHTS.get(s.name, 1.0) for s in self.signals)
        if total_weight <= 0:
            return 0.0
        return sum(s.weighted() for s in self.signals) / total_weight

    @property
    def lean(self) -> Lean:
        if self.score > 0.15:
            return "up"
        if self.score < -0.15:
            return "down"
        return "flat"

    @property
    def agreement(self) -> float:
        """Share of signals pointing the same way as the lean.

        Separate from score on purpose. One enormous signal can carry a score
        while everything else disagrees, and that is a materially different
        situation from six things quietly agreeing.
        """
        directional = [s for s in self.signals if s.direction != "flat"]
        if not directional or self.lean == "flat":
            return 0.0
        same = sum(1 for s in directional if s.direction == self.lean)
        return same / len(directional)

    @property
    def confidence(self) -> float:
        """How much to trust this read, 0..1.

        Three things multiply: how far into the candle we are, how much the
        signals agree, and how strong the score is. Early in a candle nothing
        is settled, so a strong score at 10% elapsed is still a weak read.
        """
        return (self.elapsed_fraction ** 0.5
                * max(self.agreement, 0.3)
                * min(abs(self.score) / 0.5, 1.0))

    @property
    def early(self) -> bool:
        return self.elapsed_fraction < 0.35

    def by_name(self, name: str) -> Signal | None:
        return next((s for s in self.signals if s.name == name), None)

    def verdict(self) -> str:
        mins = self.seconds_left / 60.0
        if not self.signals:
            return "No signals — not enough data to read this candle."

        head = (f"{self.lean.upper()} lean, score {self.score:+.2f}, "
                f"{self.agreement:.0%} of signals agree. "
                f"{mins:.1f} min left in the candle.")

        if self.early:
            head += (" EARLY — under a third of the candle has elapsed, so "
                     "treat this as provisional; most of it can still reverse.")

        absorb = self.by_name("absorption")
        if absorb and absorb.strength > 0.4:
            head += f" {absorb.note}"

        if self.calibration and self.calibration.ready:
            head += " " + self.calibration.describe(self.score)
        else:
            head += (" No calibration yet — this lean has not been measured "
                     "against outcomes, so it is a reading, not a probability.")
        return head

    def render(self) -> str:
        lines = [
            f"{self.coin} — {self.interval_s / 60:.0f}m candle, "
            f"{self.elapsed_fraction:.0%} elapsed",
            f"  open {self.open_px:,.2f}  now {self.last_px:,.2f}  "
            f"({self.change_bps:+.1f}bps, "
            f"{'green' if self.green else 'red'})",
            f"  position in range {self.position_in_range:.0%}",
            "",
        ]
        for s in sorted(self.signals, key=lambda x: -abs(x.weighted())):
            arrow = {"up": "▲", "down": "▼", "flat": "·"}[s.direction]
            lines.append(f"  {arrow} {s.name:11} {s.weighted():+.2f}  {s.note}")
        lines += ["", f"  {self.verdict()}"]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

@dataclass
class Calibration:
    """Measured hit rate by lean bucket.

    Fed (score, went_up) pairs from closed candles. Reports the share that
    closed up in each score band, which is the only defensible way to turn a
    lean into a probability.
    """

    buckets: dict[str, tuple[int, int]] = field(default_factory=dict)

    BANDS = ((-1.01, -0.5, "strong down"), (-0.5, -0.15, "down"),
             (-0.15, 0.15, "flat"), (0.15, 0.5, "up"), (0.5, 1.01, "strong up"))

    @classmethod
    def band_of(cls, score: float) -> str:
        for lo, hi, name in cls.BANDS:
            if lo <= score < hi:
                return name
        return "flat"

    def observe(self, score: float, went_up: bool) -> None:
        name = self.band_of(score)
        up, total = self.buckets.get(name, (0, 0))
        self.buckets[name] = (up + (1 if went_up else 0), total + 1)

    @property
    def samples(self) -> int:
        return sum(t for _, t in self.buckets.values())

    @property
    def ready(self) -> bool:
        return self.samples >= MIN_CALIBRATION_SAMPLES

    def rate(self, score: float) -> tuple[float, int]:
        """(share that closed up, sample count) for this score's band."""
        up, total = self.buckets.get(self.band_of(score), (0, 0))
        return (up / total if total else 0.0), total

    def describe(self, score: float) -> str:
        rate, n = self.rate(score)
        if n < 10:
            return (f"Only {n} past candles in this band — not enough to "
                    f"quote a rate.")
        return (f"Historically {rate:.0%} of candles with this reading closed "
                f"up ({n} samples).")

    def table(self) -> list[dict]:
        out = []
        for _, _, name in self.BANDS:
            up, total = self.buckets.get(name, (0, 0))
            out.append({"band": name, "n": total,
                        "up_rate": (up / total) if total else None})
        return out


# --------------------------------------------------------------------------
# the read
# --------------------------------------------------------------------------

def read(coin: str, interval_s: float, elapsed_s: float,
         open_px: float, high_px: float, low_px: float, last_px: float,
         tape: FlowTape | None = None,
         book: Book | None = None,
         absorption: Absorption | None = None,
         higher: Structure | None = None,
         zone: Zone | None = None,
         vw: VWAP | None = None,
         magnet_bps: float = 0.0,
         magnet_notional: float = 0.0,
         calibration: Calibration | None = None) -> CandleRead:
    """Assemble every available signal into one read.

    Everything is optional. A read with three signals is honest about having
    three; the alternative -- inventing a neutral value for a feed that is not
    connected -- pretends to information that does not exist and quietly
    drags the score toward zero.
    """
    out = CandleRead(coin=coin, interval_s=interval_s, elapsed_s=elapsed_s,
                     open_px=open_px, high_px=high_px, low_px=low_px,
                     last_px=last_px, calibration=calibration)
    sig = out.signals

    # -- flow inside this candle -----------------------------------------
    if tape is not None and elapsed_s > 0:
        w = tape.window(elapsed_s)
        if w.total > 0:
            lean = w.lean                      # -1..+1
            sig.append(Signal(
                name="flow",
                direction="up" if lean > 0.05 else "down" if lean < -0.05 else "flat",
                strength=min(abs(lean), 1.0),
                note=(f"${w.buy_notional:,.0f} bought vs "
                      f"${w.sell_notional:,.0f} sold this candle")))

    # -- where price sits in the candle's own range -----------------------
    rng = high_px - low_px
    if rng > 0:
        pos = out.position_in_range
        # Closing near the high means buyers held every attempt to push it
        # down. It is weak on its own and meaningful alongside flow.
        off_centre = (pos - 0.5) * 2.0
        sig.append(Signal(
            name="position",
            direction="up" if off_centre > 0.1 else "down" if off_centre < -0.1 else "flat",
            strength=min(abs(off_centre), 1.0),
            note=f"trading at {pos:.0%} of the candle's range"))

    # -- resting depth ----------------------------------------------------
    if book is not None and not book.empty:
        imb = book.imbalance(25.0)
        sig.append(Signal(
            name="imbalance",
            direction="up" if imb > 0.1 else "down" if imb < -0.1 else "flat",
            strength=min(abs(imb), 1.0),
            note=(f"book {abs(imb):.0%} {'bid' if imb > 0 else 'offer'} heavy "
                  f"within 25bps")))

    # -- absorption, which INVERTS flow -----------------------------------
    if absorption is not None and absorption.confident:
        if absorption.absorbing and absorption.aggressive_notional > 0:
            # Aggressors are paying and not being rewarded. The resolution
            # usually goes against them once they stop.
            against: Lean = "down" if absorption.direction == "buy" else "up"
            sig.append(Signal(
                name="absorption",
                direction=against,
                strength=min(1.0 - max(absorption.impact_ratio, 0.0), 1.0),
                note=(f"${absorption.aggressive_notional:,.0f} of "
                      f"{'buying' if absorption.direction == 'buy' else 'selling'} "
                      f"absorbed — the passive side is winning, so the flow "
                      f"signal above is pointing the wrong way.")))
        elif absorption.thin:
            # Thin book: price gives way easily, so flow carries further.
            with_it: Lean = "up" if absorption.direction == "buy" else "down"
            sig.append(Signal(
                name="absorption", direction=with_it,
                strength=min((absorption.impact_ratio - 1.0) / 3.0, 1.0),
                note="book is thin — aggression is moving price more easily "
                     "than usual, so moves extend"))

    # -- VWAP --------------------------------------------------------------
    if vw is not None and vw.value > 0 and last_px > 0:
        dist = (last_px - vw.value) / vw.value * 10_000.0
        # Near VWAP is not a signal. Far from it is mean reversion pressure,
        # and the bands are how far is far.
        if last_px >= vw.upper_2 or last_px <= vw.lower_2:
            sig.append(Signal(
                name="vwap", direction="down" if dist > 0 else "up",
                strength=0.7,
                note=f"{vw.band_of(last_px)} of VWAP — stretched"))
        elif abs(dist) > 5:
            sig.append(Signal(
                name="vwap", direction="up" if dist > 0 else "down",
                strength=0.35,
                note=f"holding {'above' if dist > 0 else 'below'} VWAP"))

    # -- higher timeframe --------------------------------------------------
    if higher is not None and higher.direction != "range":
        sig.append(Signal(
            name="structure",
            direction="up" if higher.direction == "up" else "down",
            strength=0.6,
            note=f"higher timeframe is {higher.describe()}"))

    # -- sitting in a zone -------------------------------------------------
    if zone is not None and zone.contains(last_px):
        # Supply above, demand below: a zone pushes AGAINST the move that
        # brought price into it.
        sig.append(Signal(
            name="zone",
            direction="down" if zone.kind == "supply" else "up",
            strength=0.8 if zone.fresh else 0.35,
            note=(f"inside a {'fresh' if zone.fresh else f'{zone.tested}x tested'} "
                  f"{zone.kind} zone {zone.low:,.2f}–{zone.high:,.2f}")))

    # -- liquidation magnet ------------------------------------------------
    if magnet_notional > 0 and magnet_bps != 0:
        # Closer and bigger pulls harder. Decays with distance rather than
        # cutting off, because a huge cluster far away still matters.
        pull = min(magnet_notional / 5e7, 1.0) * math.exp(-abs(magnet_bps) / 50.0)
        if pull > 0.05:
            sig.append(Signal(
                name="magnet",
                direction="up" if magnet_bps > 0 else "down",
                strength=min(pull, 1.0),
                note=(f"${magnet_notional:,.0f} of liquidations "
                      f"{abs(magnet_bps):.0f}bps "
                      f"{'above' if magnet_bps > 0 else 'below'}")))

    return out


def outcome(candle: Candle) -> bool:
    """Did this closed candle finish up. The thing calibration measures."""
    return candle.close > candle.open
