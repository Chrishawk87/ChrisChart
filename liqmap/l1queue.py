"""The inside level: cleared, or defended.

Two events live here, and the whole point is that they are opposites.

    PERMANENT   the inside level empties and the book steps. Nobody
                replaced it. The price rung is gone.

    FAILED      the inside level empties -- or very nearly -- and is
                re-established at the SAME price, with size, before the
                window closes. Somebody is holding that rung.

A level that empties and stays empty, and a level that empties and is
instantly rebuilt, look identical for the first microsecond and mean
opposite things. Separating them is all this module does.

WHY THE WINDOW IS NOT OPTIONAL. Without it there is only one event --
"the size went to zero" -- which happens constantly and says nothing.
The information is entirely in what happens next, and `window_ms` is
how long we are willing to wait to find out. Too short and a slow
refill reads as permanent; too long and the move you were trying to
read is over. It is a parameter, not a constant, and the right value
is an empirical question `scripts/l1_study.py` exists to answer.

TWO WAYS A LEVEL GETS EMPTIED, and both are handled, because only one
of them is reliably visible:

    THE STEP. The inside price moves away. Unambiguous in MBP-10 --
    level 0 simply reports a different price.

    THE DRAIN. The size at the inside collapses without the price ever
    moving, because the refill landed inside the same book update. CME
    publishes incrementally and Databento reconstructs, so a 0.5ms
    round trip from 500 contracts to 0 to 600 may never produce a
    single published book state showing that level empty. Watching only
    for the step would miss exactly the case worth catching.

WHAT THIS MODULE DOES NOT DO. It does not place, route, size, or
suggest a trade, and it holds no position state. It reports what the
inside level did. Whether either event precedes price is not asserted
anywhere here -- it is measured, on real data, by the study script,
and at the time of writing it has not been.
"""

from __future__ import annotations

from dataclasses import dataclass

BID = "bid"
ASK = "ask"

PERMANENT = "permanent"
FAILED = "failed"


@dataclass(frozen=True, slots=True)
class L1Event:
    """One resolved attack on the inside level."""

    kind: str = ""           # PERMANENT or FAILED
    side: str = ""           # which side was attacked
    ts: float = 0.0          # when it RESOLVED, not when it started
    started_ts: float = 0.0  # when the level first emptied or drained
    px: float = 0.0          # the level that was attacked
    peak_size: int = 0       # size standing there before the attack
    low_size: int = 0        # the lowest it reached
    refill_size: int = 0     # size standing after (FAILED only)
    step_px: float = 0.0     # where the inside went (PERMANENT only)
    via: str = ""            # "step" or "drain" -- how it was detected

    @property
    def elapsed_ms(self) -> float:
        return (self.ts - self.started_ts) * 1000.0

    @property
    def defended(self) -> bool:
        return self.kind == FAILED

    def as_dict(self) -> dict:
        return {"kind": self.kind, "side": self.side, "ts": self.ts,
                "started_ts": self.started_ts, "px": self.px,
                "peak_size": self.peak_size, "low_size": self.low_size,
                "refill_size": self.refill_size, "step_px": self.step_px,
                "via": self.via, "elapsed_ms": round(self.elapsed_ms, 3)}


class _SideState:
    """One side's inside level, and any attack in flight on it."""

    __slots__ = ("px", "sz", "peak", "pending_px", "pending_ts",
                 "pending_peak", "pending_low", "pending_via")

    def __init__(self) -> None:
        self.px = 0.0
        self.sz = 0
        self.peak = 0            # largest size seen at THIS price
        self.pending_px = 0.0    # 0.0 when nothing is in flight
        self.pending_ts = 0.0
        self.pending_peak = 0
        self.pending_low = 0
        self.pending_via = ""

    def clear_pending(self) -> None:
        self.pending_px = 0.0
        self.pending_ts = 0.0
        self.pending_peak = 0
        self.pending_low = 0
        self.pending_via = ""


class L1QueueWatcher:
    """Watches both inside levels and resolves attacks on them.

    Fed one book state at a time, oldest first. Returns a list because a
    single update can resolve a pending attack on one side while opening
    one on the other -- a bid that steps down is an ask that moved too.
    """

    __slots__ = ("tick", "window_s", "refill_min", "drain_frac",
                 "_bid", "_ask", "events")

    def __init__(self, tick_size: float = 0.25, window_ms: float = 2.0,
                 refill_min: int = 50, drain_frac: float = 0.90) -> None:
        if tick_size <= 0:
            raise ValueError("tick_size must be positive")
        if window_ms <= 0:
            raise ValueError("window_ms must be positive")
        if refill_min < 1:
            raise ValueError("refill_min must be at least 1")
        if not (0.0 < drain_frac <= 1.0):
            raise ValueError("drain_frac must be in (0, 1]")
        self.tick = float(tick_size)
        self.window_s = float(window_ms) / 1000.0
        # How much size must come back for a refill to count as defence.
        # A level that empties and gets one lot put back is not a wall.
        self.refill_min = int(refill_min)
        # How much of the standing size must vanish for the DRAIN path to
        # fire. 0.90 means nine tenths of it gone with the price unmoved.
        self.drain_frac = float(drain_frac)
        self._bid = _SideState()
        self._ask = _SideState()
        self.events: list[L1Event] = []

    def reset(self) -> None:
        self._bid = _SideState()
        self._ask = _SideState()
        self.events.clear()

    # ------------------------------------------------------------------
    def on_book(self, ts: float, bid_px: float, bid_sz: int,
                ask_px: float, ask_sz: int) -> list[L1Event]:
        out: list[L1Event] = []
        self._side(self._bid, BID, ts, bid_px, int(bid_sz), out)
        self._side(self._ask, ASK, ts, ask_px, int(ask_sz), out)
        self.events.extend(out)
        return out

    # ------------------------------------------------------------------
    def _vacated(self, side: str, old_px: float, new_px: float) -> bool:
        """Did the inside move AWAY, in the direction that means the old
        rung was cleared? A bid that steps DOWN was cleared. A bid that
        steps UP is a new, better bid arriving -- the opposite."""
        if side == BID:
            return new_px < old_px - self.tick / 2.0
        return new_px > old_px + self.tick / 2.0

    def _side(self, st: _SideState, side: str, ts: float,
              px: float, sz: int, out: list[L1Event]) -> None:
        if px <= 0.0:
            return

        first = st.px <= 0.0
        if first:
            st.px, st.sz, st.peak = px, sz, sz
            return

        moved = abs(px - st.px) > self.tick / 2.0

        # --- resolve anything already in flight --------------------------
        if st.pending_px > 0.0:
            back = abs(px - st.pending_px) <= self.tick / 2.0
            # THE WINDOW IS CHECKED FIRST, AND THE ORDER IS THE WHOLE
            # POINT. Asking "did it come back" before "is it too late"
            # resolves a refill that arrived after the deadline as a
            # defence -- which does not merely miss the event, it
            # reports the opposite of it. PERMANENT and FAILED are
            # meant to be opposites, so getting this backwards inverts
            # the reading rather than muting it.
            expired = (ts - st.pending_ts) >= self.window_s
            if back and sz >= self.refill_min and not expired:
                out.append(L1Event(
                    kind=FAILED, side=side, ts=ts,
                    started_ts=st.pending_ts, px=st.pending_px,
                    peak_size=st.pending_peak, low_size=st.pending_low,
                    refill_size=sz, via=st.pending_via))
                st.clear_pending()
            elif expired:
                # The window closed with the rung still gone. For a DRAIN
                # the price never moved, so `step_px` is the same level --
                # what was deleted was the size, not the rung.
                out.append(L1Event(
                    kind=PERMANENT, side=side, ts=ts,
                    started_ts=st.pending_ts, px=st.pending_px,
                    peak_size=st.pending_peak, low_size=st.pending_low,
                    step_px=px, via=st.pending_via))
                st.clear_pending()
            else:
                if sz < st.pending_low or st.pending_low == 0:
                    if not back:
                        st.pending_low = min(st.pending_low or sz, sz)

        # --- open a new attack -------------------------------------------
        if st.pending_px <= 0.0:
            if moved and self._vacated(side, st.px, px):
                # THE STEP. The rung we were standing on is gone.
                st.pending_px = st.px
                st.pending_ts = ts
                st.pending_peak = st.peak
                st.pending_low = 0
                st.pending_via = "step"
            elif not moved and st.peak > 0:
                # THE DRAIN. Same price, size gutted. `refill_min` is not
                # applied here -- the test is how much LEFT, not what is
                # standing, because a level drained from 500 to 40 has been
                # attacked whether or not 40 clears some floor.
                gone = (st.peak - sz) / float(st.peak)
                if gone >= self.drain_frac and st.peak >= self.refill_min:
                    st.pending_px = st.px
                    st.pending_ts = ts
                    st.pending_peak = st.peak
                    st.pending_low = sz
                    st.pending_via = "drain"

        # --- carry the inside forward ------------------------------------
        if moved:
            # A new price is a new queue. The old peak describes a level
            # that is no longer the one being measured, and carrying it
            # over would make every step look like a drain.
            st.px, st.sz, st.peak = px, sz, sz
        else:
            st.sz = sz
            if sz > st.peak:
                st.peak = sz

    # ------------------------------------------------------------------
    def flush(self, ts: float) -> list[L1Event]:
        """Resolve anything still pending, at `ts`.

        Called at the end of a file. Without it the last attack of each
        session is silently dropped, which biases the sample toward
        whatever happens to resolve quickly.
        """
        out: list[L1Event] = []
        for st, side in ((self._bid, BID), (self._ask, ASK)):
            if st.pending_px > 0.0 and (ts - st.pending_ts) >= self.window_s:
                out.append(L1Event(
                    kind=PERMANENT, side=side, ts=ts,
                    started_ts=st.pending_ts, px=st.pending_px,
                    peak_size=st.pending_peak, low_size=st.pending_low,
                    step_px=st.px, via=st.pending_via))
                st.clear_pending()
        self.events.extend(out)
        return out
