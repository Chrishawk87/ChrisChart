"""Which side is winning right now, steadily enough to act on.

WHAT THIS IS FOR, because it decides every choice below.

The OFI chart is not the trade. The trade is a five-minute candle on
another platform, and this is the thing that says which way to take
it. So the question is not "did a scalp make four ticks" -- it is
"when this says UP, is it right, and is it still saying UP long
enough to act on".

Two consequences follow immediately and neither is optional.

IT RESETS ON THE FIVE-MINUTE BOUNDARY, AND ON NOTHING ELSE.

The position is a five-minute candle, so the question is what THAT
candle is doing -- which means the flow is counted from the moment
the candle opened and thrown away when it closes. A reading made
from the last thirty seconds answers a different question, one that
is often the opposite of this one: a candle bought hard for four
minutes and quiet for the last thirty seconds is a bought candle,
and a trailing window calls it neutral.

So the window is the candle. `period_s` is the only thing that
resets it.

IT DOES NOT RESET WHEN AN OFI BAR CLOSES. The accumulator inside
`ofibar` does, and must: without that the threshold latches and
every bar after the first is green. But those bars are a quantity of
flow, not a quantity of clock, and they fall wherever they fall. A
reading that blanked every time one of them ended would blank
several times inside one five-minute position, each blank looking
like the signal dying.

THE RESET IS WHAT KEEPS IT FROM LATCHING. A cumulative sum of buying
minus selling walks away from zero on any session with a direction
and never comes back -- measured on a book drifting up at 52% of
ticks, that construction reads one way for the entire day. Counting
from the candle open and starting again at the next one is the same
discipline `ofibar` applies to its own accumulator, with the clock
in place of the flow.

THE READ

    UP      buying has led this candle since it opened
    DOWN    selling has led this candle since it opened
    none    too early in the candle to say, or the two are level

WHAT THE READ DESCRIBES IS THE CANDLE, NOT THE LAST FEW SECONDS. A
candle that was bought hard and has since gone quiet still reads UP,
because that is what the candle has done. The cost of that is stated
rather than hidden: it will sometimes hold UP through a stall that
turns out to be the top. `recent` is carried in the data for anyone
measuring that, but it does not gate the reading.

THE FIRST SECONDS OF A CANDLE SAY NOTHING, and the panel says so
rather than printing a direction built from four book updates. Until
`warmup_s` of the candle has passed there is no read at all, and the
line shows how far into the candle it is instead.

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

# THE CANDLE. Everything is counted from its open and thrown away at
# its close, so this is the only thing that resets the reading.
#
# Boundaries are clock-aligned on the epoch -- floor(ts / period) --
# which puts them at :00, :05, :10 and so on, the same places a
# five-minute chart puts them. Not aligned to when the service
# started, which would be five-minute candles nobody else can see.
PERIOD_S = 300.0

# How far into a candle before the reading is allowed to speak.
#
# The first seconds of a candle are a handful of book updates, and a
# direction built from those is a coin toss with a label on it. Until
# this much of the candle has passed the panel shows how far in it
# is, and no direction at all.
WARMUP_S = 10.0

# How long a new direction must survive before the published read
# changes to it.
#
# Measured, from scripts/read_tuning.py on a random walk:
#
#   warmup  hold   % of candle spoken   settles at   raw flips
#       10     0                 96.7%         10s         146
#       10     5                 93.4%         19s         146
#       10    10                 90.1%         25s         146   <- this
#       10    15                 87.2%         35s         146
#       10    30                 77.4%         50s         146
#
# Settling EARLY is what this is traded on -- a direction that
# arrives four minutes into five is a direction you cannot take -- so
# the hold is kept to the point where the read is up at about a
# twelfth of the way in and still covers nine tenths of the candle.
#
# AND THE RAW FLIP COUNT IS THE NUMBER THAT MATTERS MORE THAN EITHER.
# At 146 changes of mind a candle -- which is what a driftless walk
# produces, the cumulative lead sitting on zero and crossing it
# constantly -- the raw reading never survives any hold, so the
# published read never changes at all. It is then FROZEN on whichever
# side was ahead when it first settled, and the 90% above is
# measuring that rather than stability. On a walk this is the correct
# outcome: there is nothing to hold. On real tape a flip count still
# in the dozens means the same thing, and the reading should be
# ignored on those candles rather than tuned into looking better.
# That is why `flips` is on the panel.
HOLD_S = 10.0

# A short trailing window, carried for measurement and NOT used to
# decide anything. It answers "is this candle still being bought",
# which is a different question from "has this candle been bought"
# and sometimes the opposite answer. The reading is the second one;
# this is here so the first can be measured against outcomes later.
RECENT_S = 10.0

BUCKET_S = 0.5



@dataclass
class Snapshot:
    """What the panel draws. Every field is measured, none inferred."""

    read: str = NONE            # the steady direction -- what to act on
    raw: str = NONE             # before the hold; shown so the hold
    #                             can be checked rather than trusted
    held_s: float = 0.0         # how long `read` has stood
    pending_s: float = 0.0      # how long `raw` has disagreed with it
    up: float = 0.0             # buying since this candle opened
    dn: float = 0.0             # selling since this candle opened
    lead: float = 0.0           # up - dn; what the candle has done
    gross: float = 0.0          # up + dn; the size of the fight
    share: float = 0.5          # buying as a share of gross, 0.5 even
    recent: float = 0.0         # buy minus sell over the last RECENT_S
    #                             alone. CARRIED, NOT USED -- "is it
    #                             still being bought" is a different
    #                             question from "has it been bought",
    #                             and this reading is the second one.
    stalling: bool = False      # the candle leads one way and the
    #                             last few seconds went the other
    into_s: float = 0.0         # how far into the candle
    left_s: float = 0.0         # how much of it is left
    candle_ts: float = 0.0      # the candle's open, clock-aligned
    warm: bool = False          # past the warmup, allowed to speak
    flips: int = 0              # raw direction changes THIS CANDLE
    strength: str = ""          # "" | "weak" | "strong", vs the level
    period_s: float = PERIOD_S

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
                "recent": round(self.recent, 1),
                "stalling": self.stalling,
                "into_s": round(self.into_s, 1),
                "left_s": round(self.left_s, 1),
                "candle_ts": self.candle_ts,
                "warm": self.warm,
                "flips": self.flips,
                "strength": self.strength,
                "settled": self.settled,
                "period_s": self.period_s}


class FlowRead:
    """Buying against selling, counted from the candle open.

    One instance, fed every book update. Not per target: the five OFI
    builders are five slicings of one tape, and the flow underneath
    them is one thing. Reading it five times would produce five
    answers to a question that has one.
    """

    __slots__ = ("period_s", "warmup_s", "hold_s", "recent_s",
                 "bucket_s", "level", "_prev",
                 "_up", "_dn", "_open_ts", "_last", "_started",
                 "_r_up", "_r_dn", "_r_i", "_r_t0", "_r_n",
                 "_raw", "_raw_since", "_read", "_read_since", "_flips",
                 "_last_dir")

    def __init__(self, period_s: float = PERIOD_S,
                 warmup_s: float = WARMUP_S, hold_s: float = HOLD_S,
                 recent_s: float = RECENT_S, bucket_s: float = BUCKET_S,
                 level: int = 500) -> None:
        if period_s <= 0:
            raise ValueError("period_s must be positive")
        if warmup_s < 0 or warmup_s >= period_s:
            raise ValueError("warmup_s must be inside the candle")
        if hold_s < 0:
            raise ValueError("hold_s cannot be negative")
        if bucket_s <= 0 or bucket_s > max(recent_s, bucket_s):
            raise ValueError("bucket_s must be positive")
        self.period_s = float(period_s)
        self.warmup_s = float(warmup_s)
        self.hold_s = float(hold_s)
        self.recent_s = max(0.0, float(recent_s))
        self.bucket_s = float(bucket_s)
        # Only ever a label on the reading. The direction does not
        # depend on it and there is a test that it does not.
        self.level = max(1, int(level))
        self._prev: Touch | None = None

        # The candle's own flow. Plain running sums -- the window is
        # the candle, so nothing has to expire out of it.
        self._up = 0.0
        self._dn = 0.0
        self._open_ts = 0.0
        self._started = False
        self._last = 0.0

        # The short trailing window, for the stall diagnostic only.
        self._r_n = max(1, int(round(self.recent_s / self.bucket_s)))
        self._r_up = [0.0] * self._r_n
        self._r_dn = [0.0] * self._r_n
        self._r_i = 0
        self._r_t0 = 0.0

        self._raw = NONE
        self._raw_since = 0.0
        self._read = NONE
        self._read_since = 0.0
        self._flips = 0
        self._last_dir = NONE

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._prev = None
        self._started = False
        self._new_candle(0.0)
        self._last = 0.0

    def set_level(self, level: int) -> None:
        """The strength label only. Direction is untouched by it."""
        self.level = max(1, int(level))

    def _candle_of(self, ts: float) -> float:
        """The open of the candle `ts` falls in, aligned to the clock.

        floor(ts / period) puts the boundaries at :00, :05, :10 --
        where a five-minute chart puts them. Aligning to when the
        service happened to start would make five-minute candles
        nobody else can see.
        """
        return (ts // self.period_s) * self.period_s

    def _new_candle(self, open_ts: float) -> None:
        self._up = 0.0
        self._dn = 0.0
        self._open_ts = open_ts
        self._r_up = [0.0] * self._r_n
        self._r_dn = [0.0] * self._r_n
        self._r_i = 0
        self._r_t0 = open_ts
        # NOTHING CARRIES OVER, the published read least of all. It
        # described a candle that has closed, and leaving it up would
        # be the previous candle's answer wearing this one's label.
        self._raw = NONE
        self._read = NONE
        self._raw_since = open_ts
        self._read_since = open_ts
        self._flips = 0
        self._last_dir = NONE

    # ------------------------------------------------------------------
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

    def on_event(self, ts: float, e: float) -> None:
        """One signed order-flow imbalance event.

        Positive is buying pressure, negative selling -- the same `e`
        the bar builder accumulates, so the two cannot disagree about
        what happened.
        """
        ts = float(ts)
        if not self._started:
            self._started = True
            self._new_candle(self._candle_of(ts))
            self._last = ts
        elif ts < self._last:
            # Time went backwards: a replayed or out-of-order record.
            # Treat it as now rather than reopening a closed candle.
            ts = self._last
        open_ts = self._candle_of(ts)
        if open_ts != self._open_ts:
            self._new_candle(open_ts)
        self._last = ts
        self._roll_recent(ts)
        if e > 0:
            self._up += e
            self._r_up[self._r_i] += e
        elif e < 0:
            self._dn -= e
            self._r_dn[self._r_i] -= e
        self._decide(ts)

    def _roll_recent(self, ts: float) -> None:
        steps = int((ts - self._r_t0) // self.bucket_s)
        if steps <= 0:
            return
        if steps >= self._r_n:
            self._r_up = [0.0] * self._r_n
            self._r_dn = [0.0] * self._r_n
            self._r_i = 0
        else:
            for _ in range(steps):
                self._r_i = (self._r_i + 1) % self._r_n
                self._r_up[self._r_i] = 0.0
                self._r_dn[self._r_i] = 0.0
        self._r_t0 += steps * self.bucket_s

    def _recent(self) -> float:
        return sum(self._r_up) - sum(self._r_dn)

    def _decide(self, ts: float) -> None:
        into = ts - self._open_ts
        if into < self.warmup_s:
            # TOO EARLY. A direction built from a handful of book
            # updates is a coin toss with a label on it, and the
            # panel shows the clock instead.
            if self._raw:
                self._raw = NONE
                self._raw_since = ts
            self._read = NONE
            return
        lead = self._up - self._dn
        # THE CANDLE'S OWN LEAD, and nothing else. A candle bought
        # hard and since gone quiet still reads UP, because that is
        # what the candle has done -- `recent` is carried for anyone
        # measuring the stall, and does not gate this.
        raw = UP if lead > 0 else (DOWN if lead < 0 else NONE)
        if raw != self._raw:
            self._raw = raw
            self._raw_since = ts
            # CHANGES OF MIND, not transitions. The first direction a
            # candle settles on is not a change of mind, and counting
            # it as one would put a 1 on every quiet candle and make
            # the figure useless for spotting the noisy ones.
            if raw and self._last_dir and raw != self._last_dir:
                self._flips += 1
            if raw:
                self._last_dir = raw
        if raw != self._read and (ts - self._raw_since) >= self.hold_s:
            self._read = raw
            self._read_since = ts

    # ------------------------------------------------------------------
    @property
    def snapshot(self) -> Snapshot:
        ts = self._last
        up, dn = self._up, self._dn
        lead = up - dn
        gross = up + dn
        into = max(0.0, ts - self._open_ts) if self._started else 0.0
        recent = self._recent() if self._started else 0.0
        strength = ""
        if self._read:
            strength = "strong" if abs(lead) >= self.level else "weak"
        return Snapshot(
            read=self._read, raw=self._raw,
            held_s=max(0.0, ts - self._read_since) if self._read else 0.0,
            pending_s=(max(0.0, ts - self._raw_since)
                       if self._raw != self._read else 0.0),
            up=up, dn=dn, lead=lead, gross=gross,
            share=(up / gross) if gross > 0 else 0.5,
            recent=recent,
            stalling=bool(lead and recent and (lead > 0) != (recent > 0)),
            into_s=into,
            left_s=max(0.0, self.period_s - into) if self._started else 0.0,
            candle_ts=self._open_ts,
            warm=into >= self.warmup_s,
            flips=self._flips,
            strength=strength,
            period_s=self.period_s)

    @property
    def read(self) -> str:
        """The direction to act on, after the hold."""
        return self._read
