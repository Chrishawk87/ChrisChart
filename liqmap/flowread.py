"""Which side is winning right now, steadily enough to act on.

WHAT THIS IS FOR, because it decides every choice below.

The OFI chart is not the trade. The trade is a five-minute candle on
another platform, and this is the thing that says which way to take
it. So the question is not "did a scalp make four ticks" -- it is
"when this says UP, is it right, and is it still saying UP long
enough to act on".

Two consequences follow immediately and neither is optional.

IT CANNOT RESET AT AN OFI BAR CLOSE. The accumulator inside
`ofibar` resets at every close, and it must: without that the
threshold latches and every bar after the first is green. But a
reading that blanks every time a bar happens to end is useless to
someone holding a five-minute position -- at target 2,000 that is
several blanks per trade, each one looking like the signal died.

So this keeps its own flow, over a TRAILING WINDOW, independent of
where the bars fall.

IT IS A WINDOW, NOT A RUNNING TOTAL. A cumulative sum of buying
minus selling walks away from zero on any session with a direction
and never comes back -- measured on a book drifting up at 52% of
ticks, that construction reads one way for the entire day. A window
forgets, so it can say "the buy side was winning and now it is not",
which is the only thing worth saying.

THE READ

    UP      buying is ahead over the window AND winning right now
    DOWN    selling is ahead over the window AND winning right now
    none    neither -- including a side that is ahead on a push that
            has already finished

The third case is the point of this and not a technicality. A lead
that is being given back is a side whose push is over, and the
cumulative gap is LARGEST at exactly that moment -- which is why a
threshold on the running total is most confident just as the move
ends.

TWO WINDOWS, BOTH REQUIRED. The long one says who is AHEAD; a short
one says who is winning RIGHT NOW, and nothing is published unless
they agree.

The short window is not a refinement. Without it, buying that stopped
ten seconds ago is still sitting inside a thirty-second window, so
the lead is still positive, still not shrinking, and the read says UP
on a push that is already over -- which is the exact failure a
threshold on the running total has, arriving by a different route.

THE LEAD IS THE IMBALANCE. `push_up - push_dn` is the net imbalance
identically, so "buying minus selling" and "the imbalance" are two
names for one number and asking for both adds nothing. What this
adds over a threshold on that number is the DERIVATIVE -- whether
the gap is still opening -- and the window it is measured over.

IT HAS TO HOLD BEFORE IT COUNTS

The raw reading flips several times a minute: the net crosses a line
and falls straight back, which is the same chatter that turned one
push into twenty trades elsewhere in this package. A reading that
changes while you are clicking is not a reading. So the published
read only changes once the new direction has survived `hold_s` of
clock, and the panel shows how long the current one has stood.

The raw sign is kept and reported next to it. Hiding it would make
the smoothing unfalsifiable: if the steady read is wrong, the first
thing worth knowing is whether the raw one was right earlier.

NOTHING HERE PREDICTS. It says which side is winning and whether
that is still true. Whether that precedes price is a question for
`scripts/read_accuracy.py`, which measures it against the clock and
against its own flipped control, and which this module does not
assume the answer to.

Nothing here places, sizes or times an order.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ofi import Touch, level_ofi

UP = "up"
DOWN = "down"
NONE = ""

# THE THREE NUMBERS THIS RESTS ON, taken from a measured grid rather
# than chosen. scripts/read_tuning.py prints it; run it again on real
# files before trusting these, because the grid below was made on a
# random walk.
#
#   long  short  hold   reads/min   median     p90   silent
#     30     10     5        1.06      29.8s    68s      34%
#     30     10    10        0.64      40.3s   110s      30%
#     30     10    15        0.22     190.5s   363s       0%   <- these
#     30     20    15        0.62      59.9s   186s       0%
#     60     10    15        0.22     164.2s   368s      10%
#    120     10    20        0.04     720.1s  1063s       5%
#
# Chosen because the position is a FIVE-MINUTE CANDLE. A read whose
# median life is thirty seconds expires a tenth of the way into the
# trade, which is worse than no read: it puts you in a position the
# chart has already stopped holding. At 30/10/15 the median read runs
# a little over three minutes, nine in ten end inside six, and one
# arrives about every four and a half -- roughly one a candle.
#
# The walk's flow has no memory, so it flips harder than a real book
# and every figure above is a WORST CASE.

# How much recent flow the reading is made from.
WINDOW_S = 30.0

# How far back "still pulling away" looks. The lead now against the
# lead this long ago; if it has not grown, the side is no longer
# extending and the read goes to none.
#
# BELOW ABOUT FIVE SECONDS THIS STOPS WORKING ENTIRELY -- at a one-
# or two-second lookback the gap is never reliably growing and the
# grid above returns no reads at all, at any hold. That is a cliff,
# not a gradient, and it is the reason this is not simply "as recent
# as possible".
GROW_S = 10.0

# How long a new direction must survive before the published read
# changes to it. This is the whole of the smoothing.
#
# There is a cliff on this one too, in the other direction: at a
# twenty-second hold with a five-second grow, nothing is ever
# published, because the raw read does not survive twenty seconds on
# a walk. A hold longer than the raw read's own life is a reading
# that never speaks.
HOLD_S = 15.0

# Resolution of the trailing window. Flow is accumulated into buckets
# of this width rather than kept per update, because ES can print
# thousands of book updates a second and a sixty-second deque of them
# is a quarter of a million objects that all have to be walked. At
# half a second the error is bounded by one bucket at each end of the
# window and the memory is a fixed 120 slots.
BUCKET_S = 0.5


@dataclass
class Snapshot:
    """What the panel draws. Every field is measured, none inferred."""

    read: str = NONE            # the steady direction -- what to act on
    raw: str = NONE             # before the hold; shown so the hold
    #                             can be checked rather than trusted
    held_s: float = 0.0         # how long `read` has stood
    pending_s: float = 0.0      # how long `raw` has disagreed with it
    up: float = 0.0             # buying flow over the window
    dn: float = 0.0             # selling flow over the window
    lead: float = 0.0           # up - dn; the net imbalance
    gross: float = 0.0          # up + dn; the size of the fight
    share: float = 0.5          # buying as a share of gross, 0.5 even
    growth: float = 0.0         # lead now minus lead GROW_S ago
    recent: float = 0.0         # buy minus sell over the last GROW_S
    #                             alone -- who is winning RIGHT NOW
    extending: bool = False     # whether the leader is still pulling away
    flips_1m: int = 0           # raw direction changes in the last minute
    strength: str = ""          # "" | "weak" | "strong", vs the level
    seconds: float = 0.0        # clock covered so far, capped at window

    @property
    def dominance(self) -> float:
        """0 is an even fight, 1 is one-way traffic."""
        return abs(self.share - 0.5) * 2.0

    @property
    def settled(self) -> bool:
        """Whether the published read has stood longer than the hold.

        A read that has only just flipped is a read in the middle of
        changing its mind, and it is worth saying so.
        """
        return self.held_s >= HOLD_S

    def as_dict(self) -> dict:
        return {"read": self.read, "raw": self.raw,
                "held_s": round(self.held_s, 2),
                "pending_s": round(self.pending_s, 2),
                "up": round(self.up, 1), "dn": round(self.dn, 1),
                "lead": round(self.lead, 1),
                "gross": round(self.gross, 1),
                "share": round(self.share, 4),
                "dominance": round(self.dominance, 4),
                "growth": round(self.growth, 1),
                "recent": round(self.recent, 1),
                "extending": self.extending,
                "flips_1m": self.flips_1m,
                "strength": self.strength,
                "settled": self.settled,
                "seconds": round(self.seconds, 2)}


class FlowRead:
    """The rolling buy-versus-sell read. One instance, fed every update.

    Not per target. The bars are five different slicings of one tape;
    the flow underneath them is one thing, and reading it five times
    would produce five answers to a question with one.
    """

    __slots__ = ("window_s", "grow_s", "hold_s", "bucket_s", "level",
                 "_prev",
                 "_n", "_up", "_dn", "_i0", "_t0", "_lead_hist",
                 "_sum_up", "_sum_dn", "_raw", "_raw_since",
                 "_read", "_read_since", "_flips", "_started", "_last")

    def __init__(self, window_s: float = WINDOW_S,
                 grow_s: float = GROW_S, hold_s: float = HOLD_S,
                 bucket_s: float = BUCKET_S, level: int = 500) -> None:
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        if bucket_s <= 0 or bucket_s > window_s:
            raise ValueError("bucket_s must be positive and inside the "
                             "window")
        if grow_s < 0:
            raise ValueError("grow_s cannot be negative")
        if hold_s < 0:
            raise ValueError("hold_s cannot be negative")
        self.window_s = float(window_s)
        self.grow_s = float(grow_s)
        self.hold_s = float(hold_s)
        self.bucket_s = float(bucket_s)
        # Only ever a label on the reading. The direction does not
        # depend on it and there is a test that it does not.
        self.level = max(1, int(level))

        self._n = max(1, int(round(window_s / bucket_s)))
        self._up = [0.0] * self._n
        self._dn = [0.0] * self._n
        self._sum_up = 0.0
        self._sum_dn = 0.0
        self._i0 = 0            # bucket index of the newest bucket
        self._t0 = 0.0          # start time of that bucket
        self._started = False
        self._last = 0.0        # latest timestamp seen
        self._prev: Touch | None = None

        # Lead sampled once per bucket, for the growth comparison.
        self._lead_hist: list = [0.0] * self._n

        self._raw = NONE
        self._raw_since = 0.0
        self._read = NONE
        self._read_since = 0.0
        # (timestamp, new direction) for each raw change, trimmed to a
        # minute. The flip rate is the honest way to say "this is too
        # unstable to take right now".
        self._flips: list = []

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._prev = None
        self._up = [0.0] * self._n
        self._dn = [0.0] * self._n
        self._sum_up = self._sum_dn = 0.0
        self._lead_hist = [0.0] * self._n
        self._i0 = 0
        self._t0 = 0.0
        self._started = False
        self._last = 0.0
        self._raw = self._read = NONE
        self._raw_since = self._read_since = 0.0
        self._flips = []

    def set_level(self, level: int) -> None:
        """The strength label only. Direction is untouched by it."""
        self.level = max(1, int(level))

    # ------------------------------------------------------------------
    def on_event(self, ts: float, e: float) -> None:
        """One signed order-flow imbalance event.

        Positive is buying pressure, negative selling -- the same `e`
        the bar builder accumulates, so the two cannot disagree about
        what happened.
        """
        ts = float(ts)
        if not self._started:
            self._started = True
            self._t0 = ts
            self._last = ts
        elif ts < self._last:
            # Time went backwards: a replayed or out-of-order record.
            # Fold it into the current bucket rather than rolling the
            # window to a past instant, which would discard everything
            # between here and there.
            ts = self._last
        self._roll(ts)
        self._last = ts
        if e > 0:
            self._up[self._i0] += e
            self._sum_up += e
        elif e < 0:
            self._dn[self._i0] -= e
            self._sum_dn -= e
        self._decide(ts)

    def on_book(self, ts: float, bid_px: int, bid_sz: int,
                ask_px: int, ask_sz: int) -> None:
        """One book update, in the same shape the bar builder takes.

        The formula is `liqmap.ofi.level_ofi`, imported rather than
        copied -- a second implementation here would be a second
        thing to keep correct, and the first time the two disagreed
        the chart and this reading would be describing different
        tapes.
        """
        t = Touch(bid_px=bid_px, bid_sz=bid_sz,
                  ask_px=ask_px, ask_sz=ask_sz)
        if not t.valid or t.crossed:
            # Not a book the formula is defined on. Withhold rather
            # than contribute a zero, which would sum like a reading.
            return
        prev, self._prev = self._prev, t
        if prev is None:
            return
        self.on_event(ts, float(level_ofi(prev, t)))

    def _roll(self, ts: float) -> None:
        """Advance to the bucket `ts` falls in, clearing what expired."""
        steps = int((ts - self._t0) // self.bucket_s)
        if steps <= 0:
            return
        if steps >= self._n:
            # The whole window expired. Nothing in it is about now.
            self._up = [0.0] * self._n
            self._dn = [0.0] * self._n
            self._lead_hist = [0.0] * self._n
            self._sum_up = self._sum_dn = 0.0
            self._i0 = 0
            self._t0 += steps * self.bucket_s
            return
        for _ in range(steps):
            # The bucket just completed keeps its lead sample, then
            # the next one is cleared of whatever it held a window ago.
            self._lead_hist[self._i0] = self._sum_up - self._sum_dn
            self._i0 = (self._i0 + 1) % self._n
            self._sum_up -= self._up[self._i0]
            self._sum_dn -= self._dn[self._i0]
            self._up[self._i0] = 0.0
            self._dn[self._i0] = 0.0
        self._t0 += steps * self.bucket_s

    def _lead_ago(self, seconds: float) -> float:
        """The lead as it stood roughly this long ago."""
        back = int(round(seconds / self.bucket_s))
        if back <= 0:
            return self._sum_up - self._sum_dn
        if back >= self._n:
            back = self._n - 1
        return self._lead_hist[(self._i0 - back) % self._n]

    def _recent(self) -> float:
        """Buying minus selling over the last `grow_s` ALONE.

        A SECOND, SHORTER WINDOW, and the reason for it is a case the
        long one gets wrong on its own: buying that stopped ten
        seconds ago is still in a thirty-second window, so the lead is
        still positive and still not shrinking, and the read says UP
        on flow that has already finished. The long window says who is
        ahead; this says who is winning RIGHT NOW. Both have to agree
        before anything is published.
        """
        k = max(1, int(round(self.grow_s / self.bucket_s)))
        if k >= self._n:
            return self._sum_up - self._sum_dn
        tot = 0.0
        i = self._i0
        for _ in range(k):
            tot += self._up[i] - self._dn[i]
            i = (i - 1) % self._n
        return tot

    def _decide(self, ts: float) -> None:
        lead = self._sum_up - self._sum_dn
        growth = lead - self._lead_ago(self.grow_s)
        # PULLING AWAY IS THE WINDOWED LEAD ITSELF, and getting this
        # wrong is the first thing this module did.
        #
        # The gap between total buying and total selling is a
        # cumulative quantity; the lead over a trailing window is its
        # RATE OF CHANGE. So a positive windowed lead already means
        # "the buy side is pulling away from the sell side right now",
        # and a side that was ahead but has stopped buying sees its
        # windowed lead decay to zero on its own -- the window
        # supplies the "still", without anything else being asked for.
        #
        # Requiring the windowed lead to also be GROWING asks for
        # acceleration, which is a different and much stricter claim:
        # steady one-sided buying has a flat windowed lead, and the
        # first version of this read that as "not pulling away" and
        # went silent through exactly the pushes worth trading.
        #
        # What growth is still good for is the other sign. A lead that
        # is SHRINKING is a side whose push is fading, and reading UP
        # into that is the entry that lands just before the turn. So:
        # ahead, and not fading.
        recent = self._recent()
        if lead > 0 and recent > 0:
            raw = UP
        elif lead < 0 and recent < 0:
            raw = DOWN
        else:
            raw = NONE
        if raw != self._raw:
            self._raw = raw
            self._raw_since = ts
            self._flips.append((ts, raw))
            cut = ts - 60.0
            while self._flips and self._flips[0][0] < cut:
                self._flips.pop(0)
        # THE HOLD. The published read changes only once the raw one
        # has stood by its new answer for hold_s. Without this the
        # reading changes while you are clicking.
        if raw != self._read and (ts - self._raw_since) >= self.hold_s:
            self._read = raw
            self._read_since = ts

    # ------------------------------------------------------------------
    @property
    def snapshot(self) -> Snapshot:
        ts = self._last
        up, dn = self._sum_up, self._sum_dn
        lead = up - dn
        gross = up + dn
        growth = lead - self._lead_ago(self.grow_s)
        recent = self._recent()
        extending = (lead > 0 and recent > 0) or (lead < 0 and recent < 0)
        strength = ""
        if self._read:
            strength = "strong" if abs(lead) >= self.level else "weak"
        return Snapshot(
            read=self._read, raw=self._raw,
            held_s=max(0.0, ts - self._read_since) if self._read_since
            else 0.0,
            pending_s=(max(0.0, ts - self._raw_since)
                       if self._raw != self._read else 0.0),
            up=up, dn=dn, lead=lead, gross=gross,
            share=(up / gross) if gross > 0 else 0.5,
            growth=growth, recent=recent, extending=extending,
            flips_1m=len(self._flips),
            strength=strength,
            seconds=min(self.window_s, max(0.0, ts - self._t0
                                           + self.window_s)
                        if self._started else 0.0))

    @property
    def read(self) -> str:
        """The direction to act on, after the hold."""
        return self._read
