"""Auction state, and the ten patterns. Classification, never prediction.

WHAT THIS DOES AND DOES NOT DO

It labels what has ALREADY HAPPENED at a bar close. Given the same
structure it always emits the same label. It does not forecast, it does not
rank patterns, and it does not know whether any of them make money.

Every component of a signal is observable at the close of the bar that
emits it:

    LOCATION       which node price is interacting with
    STATE          balance / imbalance-up / imbalance-down
    INTERACTION    what price did at that location
    CONFIRMATION   the bar-close event that settles it

If a bar satisfies none of the ten patterns, the answer is NONE, and NONE
is the common case. Flat is a position.

STOP AND TARGET COME FROM THE STRUCTURE

Not from a risk preference and not from a fitted parameter. The stop is the
level whose violation means the auction read was wrong; the target is the
next node in the direction of the trade. Both are read off the profile. If
a pattern cannot name its own invalidation from structure, the pattern is
underdefined and is not emitted.

WHICH PROFILE SUPPLIES THE LEVELS

The PRIOR session's completed profile -- what is on the chart at the open,
and knowable without any assumption about today. Today's developing profile
supplies nothing here: a level derived from the session being classified
would be defined partly by the move being labelled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

TICK = 0.25

# ---------------------------------------------------------------------------
# FIXED CONVENTIONS. Stated, not tuned. Changing one changes what the
# classifier means, so each is named and pinned by a test.
# ---------------------------------------------------------------------------

# Closes beyond a level that constitute ACCEPTANCE. Two, per the spec: one
# close is a poke, two is the auction agreeing to trade there.
ACCEPT_CLOSES = 2

# How close counts as "at" a level.
AT_TICKS = 2.0

# How far back a setup may look for its triggering event -- the dip below
# POC, the failed break of VAL. Twelve bars is an hour at five minutes.
MEMORY_BARS = 12

# A pattern may not fire again on the same node until price has left by
# this much. Without it a level that price loiters at emits on every bar.
REARM_TICKS = 8.0

# A rotation has to be a rotation. Both the excursion away from a level
# and the close back through it must clear this.
#
# Without it, price dithering a single tick around the POC -- 7010.25
# then 7009.75 -- satisfied "rallied into fair value, rejected, closed
# below" and emitted a SHORT with a target forty points away. The words
# of the pattern were met; the thing they describe had not happened.
MIN_ROTATION_TICKS = 4.0

# The confirming close must happen NEAR the node it is about.
#
# Without this the confirmation drifts. On 2026-09-25 a Failed_Auction_Up
# confirmed 84 ticks below the VAH it names -- price had simply been
# under the value area for twenty minutes, and every bar re-announced a
# failure that happened once. It also produced a VA_Breakout_Up entered
# 94 ticks above the VAH, which is not a breakout, it is a chase, and
# carried a 95-tick stop because the stop stayed at the level while the
# entry ran away from it.
#
# Twelve ticks is three ES points: close enough that the level is still
# what the bar is about.
CONFIRM_TICKS = 12.0


class State(str, Enum):
    BALANCE = "balance"
    IMBALANCE_UP = "imbalance-up"
    IMBALANCE_DOWN = "imbalance-down"


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


@dataclass(frozen=True)
class Signal:
    """One classified auction event. Every field observable at the close."""

    ts: float
    pattern: str
    side: Side
    state: State

    node_kind: str
    node_price: float
    node_strength: str

    entry: float
    stop: float
    target: float

    regime: str = "unknown"
    session: str = "rth"

    @property
    def risk_ticks(self) -> float:
        return abs(self.entry - self.stop) / TICK

    @property
    def reward_ticks(self) -> float:
        return abs(self.target - self.entry) / TICK

    @property
    def coherent(self) -> bool:
        """Stop and target on the correct sides of entry.

        A long whose stop sits above its entry is not a conservative long,
        it is a bug. Emitting one would put a row on the chart that cannot
        be traded as written.
        """
        if self.side is Side.LONG:
            return self.stop < self.entry < self.target
        if self.side is Side.SHORT:
            return self.target < self.entry < self.stop
        return True

    def row(self) -> str:
        return (f"{self.ts:.0f} | {self.state.value} | "
                f"{self.node_kind} @ {self.node_price:.2f} | "
                f"{self.pattern} | {self.side.value} | "
                f"{self.entry:.2f} | {self.stop:.2f} | {self.target:.2f} | "
                f"{self.regime}")

    def to_dict(self) -> dict:
        return {"ts": self.ts, "pattern": self.pattern,
                "side": self.side.value, "state": self.state.value,
                "node_kind": self.node_kind,
                "node_price": round(self.node_price, 2),
                "node_strength": self.node_strength,
                "entry": round(self.entry, 2), "stop": round(self.stop, 2),
                "target": round(self.target, 2),
                "risk_ticks": round(self.risk_ticks, 1),
                "reward_ticks": round(self.reward_ticks, 1),
                "regime": self.regime, "session": self.session}


# ------------------------------------------------------------- the state

def classify_state(closes: Sequence[float], vah: float, val: float,
                   accept: int = ACCEPT_CLOSES) -> State:
    """Balance, or imbalance in a direction. From closes only.

    Acceptance is `accept` CONSECUTIVE closes beyond the value area. One
    close beyond is a poke and leaves the state unchanged -- that is the
    whole point of the distinction, and it is why a failed auction is a
    pattern rather than a state.
    """
    if not closes or vah <= val:
        return State.BALANCE
    tail = closes[-accept:]
    if len(tail) == accept:
        if all(c > vah for c in tail):
            return State.IMBALANCE_UP
        if all(c < val for c in tail):
            return State.IMBALANCE_DOWN
    return State.BALANCE


# ---------------------------------------------------------- the levels

@dataclass
class Levels:
    """The prior session's structure, as the classifier reads it."""

    poc: float
    vah: float
    val: float
    hvn: list[float] = field(default_factory=list)
    lvn: list[float] = field(default_factory=list)
    strength: dict = field(default_factory=dict)

    @property
    def va_width(self) -> float:
        return self.vah - self.val

    def strength_of(self, px: float) -> str:
        return self.strength.get(round(px, 4), "unknown")

    def next_above(self, px: float) -> float | None:
        """Next node above a price, for targets."""
        cand = [x for x in self.hvn + [self.poc, self.vah] if x > px + TICK]
        return min(cand) if cand else None

    def next_below(self, px: float) -> float | None:
        cand = [x for x in self.hvn + [self.poc, self.val] if x < px - TICK]
        return max(cand) if cand else None

    def nearest_hvn_below(self, px: float) -> float | None:
        cand = [x for x in self.hvn if x < px]
        return max(cand) if cand else None

    def nearest_hvn_above(self, px: float) -> float | None:
        cand = [x for x in self.hvn if x > px]
        return min(cand) if cand else None

    def at(self, px: float, level: float, ticks: float = AT_TICKS) -> bool:
        return abs(px - level) <= ticks * TICK


# --------------------------------------------------------- the patterns

class Classifier:
    """Walks bars and emits LONG / SHORT / NONE at each close.

    Stateful, because four of the ten patterns are defined by a sequence:
    a dip below the POC that was reclaimed, a break of the VAL that failed.
    The memory is bounded to `MEMORY_BARS` so a setup cannot be triggered
    by something that happened two hours ago and has stopped meaning
    anything.
    """

    def __init__(self, levels: Levels, memory: int = MEMORY_BARS,
                 accept: int = ACCEPT_CLOSES, tick: float = TICK):
        self.lv = levels
        self.memory = memory
        self.accept = accept
        self.tick = tick

        self._closes: list[float] = []
        self._bars: list = []
        self._fired: dict[str, float] = {}      # pattern -> price when fired
        self._done: set[str] = set()            # resolved this excursion

    # -- housekeeping ---------------------------------------------------

    def _armed(self, pattern: str, px: float) -> bool:
        """Has price left the level since this pattern last fired here?"""
        last = self._fired.get(pattern)
        if last is None:
            return True
        return abs(px - last) >= REARM_TICKS * self.tick

    def _settled(self, pattern: str) -> bool:
        """Has this pattern already resolved the current excursion?

        An auction fails ONCE. On 2026-09-25 the same break of the VAH
        produced seven Failed_Auction_Up flags because every later bar
        back inside the value area satisfied 'closed back inside'. The
        flag clears when price next goes beyond the level again, which
        is a genuinely new excursion.
        """
        return pattern in self._done

    def _since(self, n: int) -> list:
        return self._bars[-n:] if n > 0 else []

    def _closed_below_recently(self, level: float) -> bool:
        return any(b.close < level for b in self._since(self.memory))

    def _closed_above_recently(self, level: float) -> bool:
        return any(b.close > level for b in self._since(self.memory))

    def _dipped_below(self, level: float) -> bool:
        """A real excursion under the level, not a one-tick graze."""
        floor = level - MIN_ROTATION_TICKS * self.tick
        return any(b.close <= floor for b in self._since(self.memory))

    def _rallied_above(self, level: float) -> bool:
        ceil = level + MIN_ROTATION_TICKS * self.tick
        return any(b.close >= ceil for b in self._since(self.memory))

    def _low_since_break(self, level: float) -> float | None:
        lows = [b.low for b in self._since(self.memory) if b.close < level]
        return min(lows) if lows else None

    def _high_since_break(self, level: float) -> float | None:
        highs = [b.high for b in self._since(self.memory) if b.close > level]
        return max(highs) if highs else None

    def _accepted(self, level: float, above: bool) -> bool:
        tail = self._closes[-self.accept:]
        if len(tail) < self.accept:
            return False
        return all(c > level for c in tail) if above else \
            all(c < level for c in tail)

    # -- the ten patterns ------------------------------------------------

    def _long_patterns(self, bar, state: State) -> tuple | None:
        lv, c = self.lv, bar.close

        # 1. VA BREAKOUT UP. Balance, then acceptance above the VAH.
        #    The state test is on the PREVIOUS bars, because by this close
        #    acceptance has already made the state imbalance-up -- asking
        #    for "state == balance" here would make the pattern
        #    unreachable.
        if (classify_state(self._closes[:-1], lv.vah, lv.val) is State.BALANCE
                and self._accepted(lv.vah, above=True)
                and c <= lv.vah + CONFIRM_TICKS * self.tick):
            target = lv.next_above(c) or (lv.vah + lv.va_width)
            return ("VA_Breakout_Up", "VAH", lv.vah, c, lv.vah - TICK, target)

        # 2. LVN TRAVERSAL UP. Thin ground, entered from below, in an
        #    up-imbalance. Stop is the HVN it came from; target the HVN
        #    across the gap.
        if state is State.IMBALANCE_UP:
            for lvn in lv.lvn:
                if not lv.at(c, lvn) and not (bar.low <= lvn <= bar.high):
                    continue
                if c <= lvn:
                    continue                    # not through it yet
                # An HVN either side is the textbook case, but a trend
                # day has only ONE area of acceptance -- 2026-09-25 had
                # a single HVN with every LVN below it, so this pattern
                # fired ONCE in 392 sessions. The value area edges are
                # structure too, and falling back to them is what the
                # thin ground is actually being traversed between.
                origin = lv.nearest_hvn_below(lvn)
                if origin is None or origin >= lvn:
                    origin = lv.val if lv.val < lvn else None
                across = lv.nearest_hvn_above(lvn)
                if across is None or across <= lvn:
                    across = lv.vah if lv.vah > lvn else None
                if origin is None or across is None or across <= c:
                    continue
                return ("LVN_Traversal_Up", "LVN", lvn, c,
                        origin - TICK, across)

        # 3. HVN REJECTION UP. Tested from above, held, closed back over.
        for hvn in lv.hvn:
            if hvn >= c:
                continue
            if not (bar.low <= hvn + AT_TICKS * TICK):
                continue
            if c <= hvn + TICK:
                continue
            target = lv.next_above(c) or lv.vah
            if target <= c:
                continue
            return ("HVN_Rejection_Up", "HVN", hvn, c, hvn - TICK, target)

        # 4. FAILED AUCTION DOWN. Broke the VAL, never accepted, back in.
        if (self._closed_below_recently(lv.val)
                and not self._accepted(lv.val, above=False)
                and lv.val < c <= lv.val + CONFIRM_TICKS * self.tick
                and not self._settled('Failed_Auction_Down')):
            low = self._low_since_break(lv.val)
            if low is not None and low < lv.val:
                target = lv.poc if lv.poc > c else lv.vah
                if target > c:
                    return ("Failed_Auction_Down", "VAL", lv.val, c,
                            low - TICK, target)

        # 5. POC RECLAIM. Balance, dipped under fair value, took it back.
        if (state is State.BALANCE
                and self._dipped_below(lv.poc)
                and lv.poc + MIN_ROTATION_TICKS * self.tick <= c
                <= lv.poc + CONFIRM_TICKS * self.tick
                and lv.vah > c):
            return ("POC_Reclaim", "POC", lv.poc, c, lv.poc - TICK, lv.vah)

        return None

    def _short_patterns(self, bar, state: State) -> tuple | None:
        lv, c = self.lv, bar.close

        if (classify_state(self._closes[:-1], lv.vah, lv.val) is State.BALANCE
                and self._accepted(lv.val, above=False)
                and c >= lv.val - CONFIRM_TICKS * self.tick):
            target = lv.next_below(c) or (lv.val - lv.va_width)
            return ("VA_Breakdown", "VAL", lv.val, c, lv.val + TICK, target)

        if state is State.IMBALANCE_DOWN:
            for lvn in lv.lvn:
                if not lv.at(c, lvn) and not (bar.low <= lvn <= bar.high):
                    continue
                if c >= lvn:
                    continue
                origin = lv.nearest_hvn_above(lvn)
                if origin is None or origin <= lvn:
                    origin = lv.vah if lv.vah > lvn else None
                across = lv.nearest_hvn_below(lvn)
                if across is None or across >= lvn:
                    across = lv.val if lv.val < lvn else None
                if origin is None or across is None or across >= c:
                    continue
                return ("LVN_Traversal_Down", "LVN", lvn, c,
                        origin + TICK, across)

        for hvn in lv.hvn:
            if hvn <= c:
                continue
            if not (bar.high >= hvn - AT_TICKS * TICK):
                continue
            if c >= hvn - TICK:
                continue
            target = lv.next_below(c) or lv.val
            if target >= c:
                continue
            return ("HVN_Rejection_Down", "HVN", hvn, c, hvn + TICK, target)

        if (self._closed_above_recently(lv.vah)
                and not self._accepted(lv.vah, above=True)
                and lv.vah > c >= lv.vah - CONFIRM_TICKS * self.tick
                and not self._settled('Failed_Auction_Up')):
            high = self._high_since_break(lv.vah)
            if high is not None and high > lv.vah:
                target = lv.poc if lv.poc < c else lv.val
                if target < c:
                    return ("Failed_Auction_Up", "VAH", lv.vah, c,
                            high + TICK, target)

        if (state is State.BALANCE
                and self._rallied_above(lv.poc)
                and lv.poc - CONFIRM_TICKS * self.tick <= c
                <= lv.poc - MIN_ROTATION_TICKS * self.tick
                and lv.val < c):
            return ("POC_Rejection", "POC", lv.poc, c, lv.poc + TICK, lv.val)

        return None

    # -- the entry point -------------------------------------------------

    def step(self, bar, regime: str = "unknown",
             session: str = "rth") -> Signal | None:
        """One bar in, at most one signal out. None is the usual answer.

        The bar is appended BEFORE classification, because every pattern is
        defined on closes up to and including this one. Nothing after it is
        visible.
        """
        self._bars.append(bar)
        self._closes.append(bar.close)
        # Keep a bounded tail. The first version of this line read
        # `del self._bars[-(m+5):-(m+5) or None]`, which deletes an
        # empty slice -- the list grew without limit and a "recent"
        # dip could be two hours old.
        keep = self.memory + 5
        del self._bars[:-keep]
        del self._closes[:-max(keep, self.accept + 2)]

        # A failed auction is settled until price goes beyond the level
        # AGAIN. Re-crossing is a new excursion and may fail again; the
        # bars in between are the same event, not new ones.
        # A new excursion clears BOTH gates. The re-arm rule is keyed on
        # price, so without clearing it too, a second failure at the same
        # price as the first is silently swallowed -- and a level price
        # tests twice from the same place is exactly the case worth seeing.
        for name, beyond in (("Failed_Auction_Up", bar.close > self.lv.vah),
                             ("Failed_Auction_Down", bar.close < self.lv.val)):
            if beyond:
                self._done.discard(name)
                self._fired.pop(name, None)

        state = classify_state(self._closes, self.lv.vah, self.lv.val)

        for finder, side in ((self._long_patterns, Side.LONG),
                             (self._short_patterns, Side.SHORT)):
            hit = finder(bar, state)
            if hit is None:
                continue
            name, kind, node_px, entry, stop, target = hit
            if not self._armed(name, entry):
                continue
            sig = Signal(ts=bar.ts, pattern=name, side=side, state=state,
                         node_kind=kind, node_price=node_px,
                         node_strength=self.lv.strength_of(node_px),
                         entry=entry, stop=stop, target=target,
                         regime=regime, session=session)
            # A pattern that cannot name a coherent stop and target from
            # structure is underdefined, and a row nobody could trade as
            # written is worse than no row.
            if not sig.coherent:
                continue
            self._fired[name] = entry
            if name.startswith('Failed_Auction'):
                self._done.add(name)
            return sig
        return None
