"""Price jumped. Did the flow agree with it?

Four states, and the whole design is in the pairing:

    dip_divergent   price fell, OFI was POSITIVE   -- the stop-hunt claim
    dip_confirmed   price fell, OFI was NEGATIVE   -- ordinary selling
    pop_divergent   price rose, OFI was NEGATIVE   -- the mirror claim
    pop_confirmed   price rose, OFI was POSITIVE   -- ordinary buying

THE CONFIRMED STATES ARE NOT FILLER. They are the control that the
thesis has to beat, and without them the measurement is worthless.

Price mean-reverts after a sharp move whether or not the flow said
anything -- bid-ask bounce does that on its own, and has since long
before anyone computed an order-flow imbalance. So "price dipped three
ticks, OFI was positive, price came back" is not evidence. The
question is whether `dip_divergent` comes back HARDER than
`dip_confirmed`. If the two behave alike, what was measured is the
bounce, and the OFI reading contributed nothing but a story about who
was buying.

THE COOLDOWN IS LOad-BEARING. The jump condition is true on every
book update for as long as the window still contains the move, so a
single three-tick drop would fire several hundred times on ES and
every one of those would be counted as an independent observation. A
sample inflated a thousandfold makes anything look significant. One
jump is one event, and the watcher is deaf until the cooldown expires.

Nothing here predicts. It labels what just happened.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .ofi import Touch, level_ofi

DIP_DIVERGENT = "dip_divergent"
DIP_CONFIRMED = "dip_confirmed"
POP_DIVERGENT = "pop_divergent"
POP_CONFIRMED = "pop_confirmed"

STATES = (DIP_DIVERGENT, DIP_CONFIRMED, POP_DIVERGENT, POP_CONFIRMED)

# What the thesis says to do in each state: fade a divergent move,
# stand aside on a confirmed one. The confirmed states carry a
# direction only so the same forward measurement can be applied to
# all four on equal terms.
LONG, SHORT = "long", "short"
DIRECTION = {
    DIP_DIVERGENT: LONG,     # buy the hunted dip
    POP_DIVERGENT: SHORT,    # sell the hunted pop
    DIP_CONFIRMED: SHORT,    # the move had flow behind it
    POP_CONFIRMED: LONG,
}


@dataclass(frozen=True, slots=True)
class Flip:
    kind: str = ""
    ts: float = 0.0
    mid: float = 0.0
    ticks: float = 0.0      # signed move over the window, in ticks
    ofi: int = 0            # net imbalance over the SAME window
    window_s: float = 0.0

    @property
    def divergent(self) -> bool:
        return self.kind in (DIP_DIVERGENT, POP_DIVERGENT)

    def as_dict(self) -> dict:
        return {"kind": self.kind, "ts": self.ts, "mid": self.mid,
                "ticks": round(self.ticks, 3), "ofi": self.ofi,
                "divergent": self.divergent, "window_s": self.window_s}


class FlipWatcher:
    """Labels sharp moves by whether the flow underneath agreed.

    `jump_ticks` is how far price must travel inside `window_s` to
    count as a jump. `ofi_min` is how much net imbalance must sit on
    one side for the state to be called at all -- without it, a reading
    of +1 contract would be scored as "institutions buying".

    `settle_s` IS WHY THIS IS NOT EVALUATED THE MOMENT THE JUMP LANDS.

    Divergence in this formula is self-cancelling. Size resting against
    a move contributes negatively while it sits, and then POSITIVELY
    when price steps through it -- correctly, because a pile price went
    through was lifted or hit. Classify at the instant the jump
    completes and the last step's contribution is the largest term in
    the window, so nearly everything reads `confirmed` regardless of
    what happens next.

    The claim being tested is about buying that arrives AFTER the dip.
    So the jump arms a timer, and the state is read `settle_s` later,
    over a window that now contains the response rather than only the
    push.
    """

    __slots__ = ("tick_raw", "jump", "window_s", "ofi_min", "cooldown_s",
                 "settle_s", "_prev", "_hist", "_ofi", "_quiet_until",
                 "_armed_at", "_armed_ticks", "_armed_mid", "events")

    def __init__(self, tick_raw: int = 250_000_000, jump_ticks: float = 3.0,
                 window_s: float = 1.0, ofi_min: int = 100,
                 cooldown_s: float = 2.0, settle_s: float = 0.5) -> None:
        if tick_raw <= 0:
            raise ValueError("tick_raw must be positive")
        if jump_ticks <= 0:
            raise ValueError("jump_ticks must be positive")
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        if ofi_min < 1:
            raise ValueError("ofi_min must be at least 1")
        if cooldown_s < 0:
            raise ValueError("cooldown_s must not be negative")
        if settle_s < 0:
            raise ValueError("settle_s must not be negative")
        self.tick_raw = int(tick_raw)
        self.jump = float(jump_ticks)
        self.window_s = float(window_s)
        self.ofi_min = int(ofi_min)
        self.cooldown_s = float(cooldown_s)
        self.settle_s = float(settle_s)
        self._armed_at = 0.0
        self._armed_ticks = 0.0
        self._armed_mid = 0.0
        self._prev: Touch | None = None
        self._hist: deque = deque()      # (ts, mid2, e_n)
        self._ofi = 0
        self._quiet_until = 0.0
        self.events: list[Flip] = []

    def reset(self) -> None:
        self._prev = None
        self._hist.clear()
        self._ofi = 0
        self._quiet_until = 0.0
        self._armed_at = 0.0
        self._armed_ticks = 0.0
        self._armed_mid = 0.0
        self.events.clear()

    def on_touch(self, ts: float, touch: Touch) -> Flip | None:
        if not touch.valid or touch.crossed:
            return None

        e = 0
        if self._prev is not None:
            e = level_ofi(self._prev, touch)
        self._prev = touch

        self._hist.append((ts, touch.mid2, e))
        self._ofi += e
        cut = ts - self.window_s
        while self._hist and self._hist[0][0] < cut:
            self._ofi -= self._hist.popleft()[2]

        if len(self._hist) < 2:
            return None

        # --- a jump already armed, waiting to be read ------------------
        if self._armed_at > 0.0:
            if ts < self._armed_at + self.settle_s:
                return None
            moved = self._armed_ticks
            if abs(self._ofi) < self.ofi_min:
                # Nothing committed either way once the dust settled.
                self._armed_at = 0.0
                self._quiet_until = ts + self.cooldown_s
                return None
            down = moved < 0
            buying = self._ofi > 0
            if down:
                kind = DIP_DIVERGENT if buying else DIP_CONFIRMED
            else:
                kind = POP_CONFIRMED if buying else POP_DIVERGENT
            f = Flip(kind=kind, ts=ts, mid=touch.mid2 / 2.0, ticks=moved,
                     ofi=self._ofi, window_s=self.window_s)
            self.events.append(f)
            self._armed_at = 0.0
            # One jump is one event. See the module docstring.
            self._quiet_until = ts + self.cooldown_s
            return f

        if ts < self._quiet_until:
            return None

        # --- arm on the jump itself ------------------------------------
        # mid2 is twice the mid, kept integer. Two of those over two
        # ticks is the move in ticks.
        moved = (touch.mid2 - self._hist[0][1]) / (2.0 * self.tick_raw)
        if abs(moved) < self.jump:
            return None
        self._armed_at = ts
        self._armed_ticks = moved
        self._armed_mid = touch.mid2 / 2.0
        if self.settle_s <= 0.0:
            return self.on_touch(ts, touch)
        return None

    def on_book(self, ts: float, bid_px: int, bid_sz: int,
                ask_px: int, ask_sz: int) -> Flip | None:
        return self.on_touch(ts, Touch(bid_px=bid_px, bid_sz=bid_sz,
                                       ask_px=ask_px, ask_sz=ask_sz))
