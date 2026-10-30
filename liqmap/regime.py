"""Error bars for the OFI read. Four tick metrics, and what they tag.

THE PRINCIPLE, AND IT IS ENFORCED HERE RATHER THAN TRUSTED.

The imbalance is the signal and nothing in this module may change
its direction. These metrics tag the state the signal arrived in --
they can tell you to stand down or to size smaller, and that is all.
`liqmap.flowread` does not import this module, and there is a test
that it does not, because the moment a regime could flip a read the
chart would have two signals disagreeing and no way to say which one
spoke.

THE FOUR

    SPREAD          ask minus bid, in ticks. ES is one tick almost
                    all of the time; anything wider is a book that
                    has thinned out, and it is measured, not guessed.

    VPIN            how one-sided the traded flow has been, over
                    buckets of equal VOLUME rather than equal clock.
                    High means one side is doing most of the taking.

    TAKING          the same buckets WITHOUT the absolute value, so
                    it carries a side. VPIN says how lopsided; this
                    says which way.

    CANCEL RATE     how much of the size posted at the touch was
                    pulled rather than traded. High means the book
                    being read may not have been there to hit.

    GROSS           the size of the fight behind the net. Carried
                    because a 500 net out of 600 and a 500 net out of
                    9,000 are one number and not one state.

VPIN CANNOT CONFIRM A DIRECTION, AND IT IS WORTH BEING EXACT ABOUT
WHY. It is a mean of |buy - sell| / V. The absolute value is the
metric: a bucket where sellers did 80% of the lifting and one where
buyers did produce the IDENTICAL number. So it can tell you the
taking was lopsided and never which way, which is exactly why it
belongs in the veto-or-size column and nowhere near the direction.

Drop the absolute value and the same buckets give `taking`, which
does carry a side. That one is comparable with the read: the book
says up and the aggressive taking is also up, or it is not. It is
not the same as the delta already on the chart -- that is clocked by
TIME, and this by VOLUME, so 500 contracts of taking weighs the same
whether it took four seconds or four minutes.

VPIN IS BUILT ON THE STATED AGGRESSOR, NOT ON THE PAPER'S ESTIMATE.
The original construction infers the buy/sell split from price
changes, because its authors did not have the side. CME marks it on
every print and this feed carries it, so the inference is skipped
entirely. That makes this a better VPIN than the published one, and
it also means figures here will not match a vendor's VPIN computed
the classical way.

AND VPIN IS CONTESTED, which belongs in the code rather than in a
footnote somewhere. Easley, Lopez de Prado and O'Hara introduced it
as a measure of informed trading; Andersen and Bondarenko (2014)
argued its apparent forecasting power comes mostly from the
volume-time clock and from tracking volatility, not from detecting
informed flow. Nothing here takes a side in that. The metric is
computed, tagged and MEASURED AGAINST OUTCOMES -- which is the only
way this particular argument gets settled on one person's market.

THRESHOLDS ARE PERCENTILES OF THE SESSION, NOT CONSTANTS.

"High VPIN" cannot be a number written here. It depends on the
instrument, the session and the day, and a constant would quietly
mean something different every morning. So a reading is high when it
sits above a percentile of what this session has already produced,
and the panel says how many samples that percentile is made of --
a threshold computed from nine observations is not a threshold.

The one exception is the spread, which has a floor the market
defines: one tick. Wider than the minimum IS wide, on any day.

Nothing here predicts, and nothing here places, sizes or times an
order.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

# Action codes, as they arrive on MBP-10 records.
ADD = "A"
CANCEL = "C"
MODIFY = "M"
TRADE = "T"
FILL = "F"
CLEAR = "R"

# VPIN buckets are a VOLUME, not a stretch of clock -- that is the
# whole of what "volume-synchronised" means. The published work uses
# a fiftieth of a day's volume, which on ES is tens of thousands of
# contracts and a lookback of hours. A five-minute read cannot wait
# for that, so the bucket is small enough that fifty of them span
# minutes rather than sessions.
#
# IT IS THE FIRST THING TO RE-TUNE ON REAL VOLUME. Too small and
# every bucket is one trade and VPIN pins to 1.0; too large and it
# never moves. scripts/regime_outcomes.py sweeps it.
BUCKET_VOL = 500.0
BUCKETS = 50

# How much recent book activity the cancel rate is measured over.
# Seconds, because cancels are not synchronised to anything.
CANCEL_WINDOW_S = 30.0

# Where a reading becomes "high" -- as a percentile of what this
# session has produced, never as an absolute.
HIGH_PCT = 0.70

# Samples before a percentile is allowed to mean anything. Below
# this the panel says so instead of showing a band.
MIN_SAMPLES = 30

# The spread's threshold is the one that can be absolute: the market
# defines the minimum increment, so wider than one tick is wide on
# any day and any instrument.
TIGHT_TICKS = 1.0


@dataclass
class RegimeSnapshot:
    """What the panel draws. Every field measured, none inferred."""

    spread_ticks: float = 0.0
    vpin: float = 0.0
    taking: float = 0.0         # signed: + buyers lifting, - sellers
    cancel_rate: float = 0.0
    gross: float = 0.0

    spread_band: str = ""       # "tight" | "wide"
    vpin_band: str = ""         # "low" | "high" | "" when unknown
    cancel_band: str = ""       # "normal" | "high" | ""

    vpin_n: int = 0             # samples behind the vpin percentile
    cancel_n: int = 0
    buckets: int = 0            # completed volume buckets so far

    @property
    def taking_side(self) -> str:
        """Which way the aggressive taking has gone. '' when level.

        THIS IS NOT A SIGNAL AND MUST NOT BECOME ONE. It is here to
        be compared with the read -- 'the book and the tape agree' --
        and the comparison is drawn by whoever holds both. Nothing in
        this module knows what the read says.
        """
        if self.taking > 0:
            return "up"
        if self.taking < 0:
            return "down"
        return ""

    def agrees_with(self, read: str) -> str:
        """'agree' / 'disagree' / '' against a direction from outside.

        Takes the read as an ARGUMENT rather than reaching for it, so
        that this module still cannot see the reading -- the rule
        that a tag may never change a direction is kept by the shape
        of the code and not by anyone remembering it.
        """
        side = self.taking_side
        if not side or read not in ("up", "down"):
            return ""
        return "agree" if side == read else "disagree"

    @property
    def known(self) -> bool:
        """Whether there is enough history for the bands to mean
        anything. Below it the metrics are still real; the words
        'high' and 'low' are not."""
        return bool(self.vpin_band and self.cancel_band)

    @property
    def tag(self) -> str:
        """The regime, as a key. Empty until every band is known.

        A partial tag would quietly become its own category and
        collect outcomes under a name that means 'we did not know'.
        """
        if not self.known or not self.spread_band:
            return ""
        return f"{self.vpin_band}vpin-{self.spread_band}-{self.cancel_band}cancel"

    def as_dict(self) -> dict:
        return {"spread_ticks": round(self.spread_ticks, 2),
                "vpin": round(self.vpin, 4),
                "taking": round(self.taking, 4),
                "taking_side": self.taking_side,
                "cancel_rate": round(self.cancel_rate, 4),
                "gross": round(self.gross, 1),
                "spread_band": self.spread_band,
                "vpin_band": self.vpin_band,
                "cancel_band": self.cancel_band,
                "vpin_n": self.vpin_n, "cancel_n": self.cancel_n,
                "buckets": self.buckets,
                "known": self.known, "tag": self.tag}


class Percentile:
    """A rolling percentile of what this session has produced.

    A reservoir of recent values, sorted on demand. Bounded, because
    a session is hours and an unbounded list of every reading is a
    leak with a statistic's name on it.
    """

    __slots__ = ("cap", "_xs")

    def __init__(self, cap: int = 4000) -> None:
        self.cap = int(cap)
        self._xs: deque = deque(maxlen=self.cap)

    def add(self, x: float) -> None:
        self._xs.append(float(x))

    def __len__(self) -> int:
        return len(self._xs)

    def at(self, p: float) -> float:
        if not self._xs:
            return 0.0
        xs = sorted(self._xs)
        i = min(len(xs) - 1, max(0, int(p * (len(xs) - 1))))
        return xs[i]

    def band(self, x: float, p: float = HIGH_PCT,
             min_n: int = MIN_SAMPLES) -> str:
        """'high' / 'low', or empty when there is not enough behind it."""
        if len(self._xs) < min_n:
            return ""
        return "high" if x >= self.at(p) else "low"


class Regime:
    """The four metrics, fed from the same tape as everything else.

    Nothing here is per target and nothing here resets with a bar.
    The book's state is the book's state.
    """

    __slots__ = ("bucket_vol", "buckets", "cancel_window_s",
                 "high_pct", "min_samples",
                 "_spread", "_gross", "_last_ts",
                 "_b_buy", "_b_sell", "_b_vol", "_bkts", "_signed",
                 "_vpin", "_taking", "_vp", "_events", "_cp",
                 "_cancel", "_added", "_pulled")

    def __init__(self, bucket_vol: float = BUCKET_VOL,
                 buckets: int = BUCKETS,
                 cancel_window_s: float = CANCEL_WINDOW_S,
                 high_pct: float = HIGH_PCT,
                 min_samples: int = MIN_SAMPLES) -> None:
        if bucket_vol <= 0:
            raise ValueError("bucket_vol must be positive")
        if buckets < 2:
            raise ValueError("buckets must be at least 2")
        self.bucket_vol = float(bucket_vol)
        self.buckets = int(buckets)
        self.cancel_window_s = float(cancel_window_s)
        self.high_pct = float(high_pct)
        self.min_samples = int(min_samples)

        self._spread = 0.0
        self._gross = 0.0
        self._last_ts = 0.0

        # The bucket being filled, and the finished ones.
        self._b_buy = 0.0
        self._b_sell = 0.0
        self._b_vol = 0.0
        self._bkts: deque = deque(maxlen=self.buckets)
        # The same buckets before the absolute value. VPIN is the
        # mean of |these|; `taking` is the mean of these.
        self._signed: deque = deque(maxlen=self.buckets)
        self._vpin = 0.0
        self._taking = 0.0
        self._vp = Percentile()

        # (ts, action, size) for the cancel window.
        self._events: deque = deque()
        # Running totals for the window. Re-summing the deque on
        # every event is O(window) per book update, and at ES rates
        # a thirty-second window holds tens of thousands of entries
        # -- that is real CPU burned on every tick, for a number that
        # can be maintained incrementally.
        self._added = 0.0
        self._pulled = 0.0
        self._cancel = 0.0
        self._cp = Percentile()

    # ------------------------------------------------------------------
    def on_book(self, ts: float, bid_px: int, bid_sz: int,
                ask_px: int, ask_sz: int, tick: float) -> None:
        """The touch. Only the spread is taken from it."""
        if bid_px <= 0 or ask_px <= 0 or bid_px >= ask_px or tick <= 0:
            # Crossed, locked or absent. Withhold rather than record a
            # spread of zero, which would sum like a measurement and
            # would read as the tightest book of the day.
            return
        self._spread = (ask_px - bid_px) / tick
        self._last_ts = ts

    def on_trade(self, ts: float, size: float, aggressor: str) -> None:
        """One print, with the side CME stated on it.

        `aggressor` is 'buy' or 'sell'. Anything else is a print
        whose side the feed did not state, and it is counted in the
        bucket's volume but not in either side -- dropping it would
        make the bucket finish late, and guessing it is the thing
        this whole module refuses to do.
        """
        size = float(size)
        if size <= 0:
            return
        self._last_ts = ts
        if aggressor == "buy":
            self._b_buy += size
        elif aggressor == "sell":
            self._b_sell += size
        self._b_vol += size
        while self._b_vol >= self.bucket_vol:
            # A bucket is a fixed VOLUME. When a print overfills one,
            # the excess belongs to the next -- splitting it keeps the
            # buckets equal, which is the only property the whole
            # construction rests on.
            take = self.bucket_vol
            share = take / self._b_vol if self._b_vol else 0.0
            buy = self._b_buy * share
            sell = self._b_sell * share
            self._bkts.append(abs(buy - sell) / take)
            self._signed.append((buy - sell) / take)
            self._b_buy -= buy
            self._b_sell -= sell
            self._b_vol -= take
            if self._bkts:
                self._vpin = sum(self._bkts) / len(self._bkts)
                self._taking = sum(self._signed) / len(self._signed)
                self._vp.add(self._vpin)

    def on_action(self, ts: float, action: str, size: float) -> None:
        """One book event, by what caused it.

        MBP-10 states the action on every record -- add, cancel,
        modify, trade, fill -- so cancellation is COUNTED here rather
        than inferred from size going down. Inferring it would also
        catch every fill and every modify, which is three different
        things wearing one name.
        """
        size = float(size or 0.0)
        a = (action or "")[:1].upper()
        if a not in (ADD, CANCEL) or size <= 0:
            return
        self._last_ts = ts
        self._events.append((ts, a, size))
        if a == ADD:
            self._added += size
        else:
            self._pulled += size
        cut = ts - self.cancel_window_s
        while self._events and self._events[0][0] < cut:
            _, x, sz = self._events.popleft()
            if x == ADD:
                self._added -= sz
            else:
                self._pulled -= sz
        total = self._added + self._pulled
        self._cancel = (self._pulled / total) if total > 0 else 0.0
        self._cp.add(self._cancel)

    def set_gross(self, gross: float) -> None:
        """The size of the fight behind the net, from the reading."""
        self._gross = float(gross or 0.0)

    # ------------------------------------------------------------------
    @property
    def snapshot(self) -> RegimeSnapshot:
        sp = self._spread
        return RegimeSnapshot(
            spread_ticks=sp,
            vpin=self._vpin,
            taking=self._taking,
            cancel_rate=self._cancel,
            gross=self._gross,
            # The market defines the minimum increment, so this one
            # threshold can be absolute without smuggling in a guess.
            spread_band=("tight" if 0 < sp <= TIGHT_TICKS
                         else "wide" if sp > TIGHT_TICKS else ""),
            vpin_band=self._vp.band(self._vpin, self.high_pct,
                                    self.min_samples),
            cancel_band=(("high" if self._cp.band(
                self._cancel, self.high_pct, self.min_samples) == "high"
                else "normal") if len(self._cp) >= self.min_samples
                else ""),
            vpin_n=len(self._vp), cancel_n=len(self._cp),
            buckets=len(self._bkts))

    def reset(self) -> None:
        self._b_buy = self._b_sell = self._b_vol = 0.0
        self._bkts.clear()
        self._signed.clear()
        self._vpin = 0.0
        self._taking = 0.0
        self._vp = Percentile()
        self._events.clear()
        self._added = self._pulled = 0.0
        self._cancel = 0.0
        self._cp = Percentile()
        self._spread = 0.0
        self._gross = 0.0


# ---------------------------------------------------------------- tally

@dataclass
class RegimeTally:
    """What each regime has actually done, counted as it happens.

    THE TABLE IS A RESULT, NOT AN ASSUMPTION. "High confidence",
    "toxic", "suspect" are hypotheses about one person's market, and
    printing them on a chart before anything has checked them would
    be the chart asserting what it was told. So each regime collects
    outcomes, and the words stay off the panel until there are enough
    of them to say something.
    """

    rows: dict = field(default_factory=dict)

    def record(self, tag: str, read: str, outcome: str) -> None:
        """One finished candle: the regime it was in, the read it
        carried, and which way it closed."""
        if not tag or read not in ("up", "down"):
            return
        r = self.rows.setdefault(tag, {"n": 0, "right": 0, "wrong": 0,
                                       "flat": 0})
        r["n"] += 1
        if not outcome:
            r["flat"] += 1
        elif outcome == read:
            r["right"] += 1
        else:
            r["wrong"] += 1

    def of(self, tag: str, min_n: int = MIN_SAMPLES) -> dict | None:
        """What this regime has done, or None if not enough yet."""
        r = self.rows.get(tag)
        if not r:
            return None
        dec = r["right"] + r["wrong"]
        if dec < min_n:
            return {"tag": tag, "n": r["n"], "decided": dec,
                    "enough": False}
        return {"tag": tag, "n": r["n"], "decided": dec, "enough": True,
                "right": r["right"], "wrong": r["wrong"],
                "flat": r["flat"], "pct": round(r["right"] / dec, 4)}

    def as_dict(self) -> dict:
        return {t: self.of(t) for t in sorted(self.rows)}
