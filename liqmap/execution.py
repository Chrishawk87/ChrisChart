"""Execution Track: real-time signal generation from raw MBP-10 prints.

This module is ENTIRELY INDEPENDENT of the visualization track.  It has
its own FootprintStream and its own delta accumulator, fed from the same
raw Trade objects that the visualization track uses -- but neither track
reads the other's data structures.

THE SIGNAL CONTRACT

A Signal fires when three conditions are simultaneously true on the
LIVE, UNCLOSED bar:

  1. STACKED IMBALANCE  — `min_run` or more consecutive diagonal
     imbalances on the same side, as measured by the live FootprintStream
     Tracker (not a closed bar, not a candle boundary).

  2. DIRECTIONAL DELTA  — The Diagonal Cumulative Delta (DCD) is non-zero
     and agrees with the stacked side.  Flat delta with a structural
     pattern is not a signal.

  3. FLAG MATURITY      — At least one flag in the stacked run has been
     re-measured `min_flag_updates` or more times.  A flag that appeared
     on a single tick is noise; one that has held across multiple prints
     is a structure.

A fourth gate -- COOLDOWN -- suppresses repeat signals on the same side
within `cooldown_s` seconds.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from .footprint import (
    BUY_IMBALANCE, SELL_IMBALANCE,
    DEFAULT_IMBALANCE, ImbalanceConfig,
    TimeRule, CloseRule, Tick,
)
from .footprint_live import FootprintStream, Tracker, LiveFlag

# Default 300-second (5-minute) time bars for the execution footprint.
DEFAULT_INTERVAL_S = 300.0


# --------------------------------------------------------------------------
# Signal
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Signal:
    """One execution signal, fully immutable.

    Every field is a measurement from the LIVE bar at the instant the
    signal fired, so a log of signals carries its own audit trail without
    a separate tape lookup.
    """

    ts: float
    side: str                   # "buy" or "sell"
    price: float                # last traded price when signal fired
    raw_delta: int              # total bar delta (all prints)
    diagonal_delta: int         # delta restricted to imbalance prices
    unconfirmed: int            # live flag count at fire time
    buy_flags: int
    sell_flags: int
    stacked_side: str           # "buy" or "sell"
    stacked_low: float          # lowest price in the stacked run
    stacked_high: float         # highest price in the stacked run
    stacked_run: int            # consecutive imbalance count
    bar_age_s: float            # seconds since bar open
    reason: str                 # human-readable trigger description

    def as_dict(self) -> dict:
        return {
            "ts": self.ts,
            "side": self.side,
            "price": self.price,
            "raw_delta": self.raw_delta,
            "diagonal_delta": self.diagonal_delta,
            "unconfirmed": self.unconfirmed,
            "buy_flags": self.buy_flags,
            "sell_flags": self.sell_flags,
            "stacked_side": self.stacked_side,
            "stacked_low": self.stacked_low,
            "stacked_high": self.stacked_high,
            "stacked_run": self.stacked_run,
            "bar_age_s": round(self.bar_age_s, 3),
            "reason": self.reason,
        }


# --------------------------------------------------------------------------
# Diagonal Cumulative Delta
# --------------------------------------------------------------------------

class DiagonalCumulativeDelta:
    """Running delta restricted to price levels with a live imbalance flag.

    Standard CVD accumulates every print.  DCD adds a print to its
    running sum ONLY when the footprint Tracker currently holds an active
    (UNCONFIRMED or CONFIRMED) flag at that price.  Noise prints at
    unstructured levels are excluded.

    Reset on each bar close, matching the footprint bar boundary.

    Thread safety: the Tracker's flag dict is mutated by the feed thread
    under a lock.  We read it here WITHOUT the lock, because:
      - dict key lookup under the GIL is atomic at the Python level.
      - We snapshot `set(tracker.flags)` before iterating -- a snapshot
        that may be one print stale, which is acceptable for a delta
        accumulator whose whole point is to be approximate.
    """

    __slots__ = ("_raw", "_diagonal", "_scale")

    def __init__(self, tick_size: float = 0.25) -> None:
        self._raw: int = 0
        self._diagonal: int = 0
        from .footprint import TickScale
        self._scale = TickScale(tick_size)

    def update(self, tick: Tick, tracker: Tracker) -> None:
        """Fold one print. tracker.flags is read without a lock (see class doc)."""
        signed = tick.size if tick.aggressor == "buy" else -tick.size
        self._raw += signed
        px_ticks = self._scale.to_ticks(tick.price)
        # A print counts toward diagonal delta when ANY flag at that
        # price level is currently live (either side).
        flag_keys = set(tracker.flags)         # snapshot under GIL
        if any(p == px_ticks for (p, _) in flag_keys):
            self._diagonal += signed

    def reset(self) -> None:
        self._raw = 0
        self._diagonal = 0

    @property
    def raw(self) -> int:
        return self._raw

    @property
    def diagonal(self) -> int:
        return self._diagonal


# --------------------------------------------------------------------------
# Live stacked run helper
# --------------------------------------------------------------------------

def _live_stacked(
    tracker: Tracker,
    scale: Any,                 # TickScale
    run: int = 3,
) -> list[tuple[str, float, float, int]]:
    """Return all live stacked imbalance runs of at least `run` length.

    Returns a list of (side, low_price, high_price, run_length) tuples,
    sorted by run_length descending so the longest run is first.

    This reads tracker.flags WITHOUT the outer FootprintStream lock,
    which is safe for the same reason DiagonalCumulativeDelta reads it
    safely: all reads are GIL-protected and a one-tick stale snapshot
    never causes a correctness fault (it might miss or delay a signal,
    never emit a fabricated one).
    """
    by_side: dict[str, list[int]] = {}
    for (p, side), _flag in list(tracker.flags.items()):
        by_side.setdefault(side, []).append(p)

    out: list[tuple[str, float, float, int]] = []
    for side, prices in by_side.items():
        prices.sort()
        if not prices:
            continue
        start = prev = prices[0]
        for p in prices[1:]:
            if p == prev + 1:
                prev = p
            else:
                length = prev - start + 1
                if length >= run:
                    out.append((
                        side,
                        scale.to_price(start),
                        scale.to_price(prev),
                        length,
                    ))
                start = prev = p
        length = prev - start + 1
        if length >= run:
            out.append((
                side,
                scale.to_price(start),
                scale.to_price(prev),
                length,
            ))

    out.sort(key=lambda t: t[3], reverse=True)
    return out


# --------------------------------------------------------------------------
# ExecState
# --------------------------------------------------------------------------

@dataclass
class ExecState:
    """A point-in-time snapshot of the ExecutionTrack, for polling."""

    ts: float
    pulse: Any | None                   # Pulse from the execution footprint
    diagonal_delta: int
    raw_delta: int
    stacked: list                       # list of (side, low, high, run)
    signals_total: int
    last_signal: Signal | None

    def as_dict(self) -> dict:
        return {
            "ts": self.ts,
            "pulse": self.pulse.as_dict() if self.pulse else None,
            "diagonal_delta": self.diagonal_delta,
            "raw_delta": self.raw_delta,
            "stacked": [
                {"side": s, "low": lo, "high": hi, "run": r}
                for (s, lo, hi, r) in self.stacked
            ],
            "signals_total": self.signals_total,
            "last_signal": self.last_signal.as_dict() if self.last_signal else None,
        }


# --------------------------------------------------------------------------
# ExecutionTrack
# --------------------------------------------------------------------------

class ExecutionTrack:
    """Real-time execution signal engine.

    Receives raw Trade objects from cme_tape.Tape._exec_listeners.  Runs
    its own FootprintStream (separate from the visualization footprint) so
    the execution engine's state is never shared with or delayed by the
    charting layer.

    Signal callbacks (on_signal / off_signal) are called SYNCHRONOUSLY from
    the OFI background thread.  Keep them fast: hand frames to an asyncio
    queue and return; never block.
    """

    def __init__(
        self,
        tick_size: float = 0.25,
        rule: CloseRule | None = None,
        imbalance: ImbalanceConfig = DEFAULT_IMBALANCE,
        min_run: int = 3,
        min_flag_updates: int = 1,
        cooldown_s: float = 5.0,
    ) -> None:
        self._tick_size = tick_size
        _rule = rule or TimeRule(DEFAULT_INTERVAL_S)
        self._stream = FootprintStream(
            tick_size=tick_size, rule=_rule, imbalance=imbalance)
        self._dcd = DiagonalCumulativeDelta(tick_size=tick_size)
        self._min_run = int(min_run)
        self._min_flag_updates = int(min_flag_updates)
        self._cooldown_s = float(cooldown_s)

        self._signal_listeners: list[Callable[[Signal], None]] = []
        self._lock = threading.Lock()      # guards signal_listeners + _last_signal_ts

        self._signals_total: int = 0
        self._last_signal: Signal | None = None
        self._last_signal_ts: dict[str, float] = {}  # side -> ts of last signal

    # --------------------------------------------------------- callbacks

    def on_signal(self, fn: Callable[[Signal], None]) -> None:
        """Register a callback invoked on every signal. Must be fast."""
        with self._lock:
            self._signal_listeners.append(fn)

    def off_signal(self, fn: Callable[[Signal], None]) -> None:
        with self._lock:
            try:
                self._signal_listeners.remove(fn)
            except ValueError:
                pass

    # ------------------------------------------------------------ feed

    def on_trade(self, trade: Any) -> None:
        """Process one raw Trade from the OFI tape.

        Called synchronously from the OFI background thread. Must return
        quickly -- no I/O, no blocking, no allocating large objects.
        """
        # Convert the Tape Trade to a footprint Tick.
        size = int(round(float(trade.sz)))
        if size <= 0:
            return
        tick = Tick(
            ts=float(trade.ts),
            price=float(trade.px),
            size=size,
            aggressor=trade.aggressor,
        )

        # Feed into the execution footprint stream.
        closed = self._stream.on_tick(tick)

        # If a bar closed, reset the DCD accumulator.
        if closed:
            self._dcd.reset()

        # Update DCD with the current tracker state AFTER the stream
        # has processed the tick (so tracker flags reflect this print).
        self._dcd.update(tick, self._stream.tracker)

        # Evaluate signal conditions.
        self._evaluate(tick)

    def _evaluate(self, tick: Tick) -> None:
        """Check all signal conditions; fire if met."""
        tracker = self._stream.tracker
        scale = self._stream.scale
        pulse = self._stream.pulse()

        # Gate 1: stacked run
        stacked = _live_stacked(tracker, scale, run=self._min_run)
        if not stacked:
            return

        top_side, top_low, top_high, top_run = stacked[0]

        # Gate 2: diagonal delta agrees
        dd = self._dcd.diagonal
        if top_side == "buy" and dd <= 0:
            return
        if top_side == "sell" and dd >= 0:
            return

        # Gate 3: flag maturity — at least one flag in the stacked run
        #         has been re-measured min_flag_updates times.
        sc = scale
        stacked_prices = set(
            sc.to_ticks(lo + sc.size * i)
            for (s, lo, hi, _) in stacked
            if s == top_side
            for i in range(int(round((hi - lo) / sc.size)) + 1)
        )
        mature = any(
            flag.updates >= self._min_flag_updates
            for (p, side), flag in list(tracker.flags.items())
            if side == top_side and p in stacked_prices
        )
        if not mature:
            return

        # Gate 4: cooldown
        now = tick.ts
        with self._lock:
            last_ts = self._last_signal_ts.get(top_side, 0.0)
            if now - last_ts < self._cooldown_s:
                return
            # --- signal fires ---
            buy_flags = sum(1 for (_, s) in tracker.flags if s == BUY_IMBALANCE)
            sell_flags = sum(1 for (_, s) in tracker.flags if s == SELL_IMBALANCE)
            sig = Signal(
                ts=now,
                side=top_side,
                price=tick.price,
                raw_delta=self._dcd.raw,
                diagonal_delta=dd,
                unconfirmed=pulse.unconfirmed,
                buy_flags=buy_flags,
                sell_flags=sell_flags,
                stacked_side=top_side,
                stacked_low=top_low,
                stacked_high=top_high,
                stacked_run=top_run,
                bar_age_s=pulse.bar_age_s,
                reason=(
                    f"stacked_{top_side}_{top_run} "
                    f"dcd={dd:+d} "
                    f"flags={buy_flags}b/{sell_flags}s"
                ),
            )
            self._last_signal_ts[top_side] = now
            self._last_signal = sig
            self._signals_total += 1
            listeners = list(self._signal_listeners)

        for fn in listeners:
            try:
                fn(sig)
            except Exception:               # noqa: BLE001
                pass

    # ------------------------------------------------------------ read

    def state(self) -> ExecState:
        """A snapshot of the current execution state, safe to serialise."""
        pulse = self._stream.pulse()
        scale = self._stream.scale
        tracker = self._stream.tracker
        stacked = _live_stacked(tracker, scale, run=self._min_run)
        with self._lock:
            last_sig = self._last_signal
            total = self._signals_total
        return ExecState(
            ts=time.time(),
            pulse=pulse,
            diagonal_delta=self._dcd.diagonal,
            raw_delta=self._dcd.raw,
            stacked=stacked,
            signals_total=total,
            last_signal=last_sig,
        )
