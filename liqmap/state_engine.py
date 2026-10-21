"""Deterministic State Engine — in-memory stream processor on DeltaWorker.

Sits on the DeltaWorker thread, immediately after Diagonal Cumulative
Delta is updated, and nowhere else.  It does not read the visualization
footprint, it does not touch a layout, and it does not emit on every
print.  A line is written only when the stance actually changes.

STANCES (ints, not strings, so a comparison is a register test)

    FLAT  = 0
    LONG  = 1
    SHORT = 2

A transition is one of ENTER_LONG, ENTER_SHORT, EXIT_LONG, EXIT_SHORT,
REVERSE_LONG (SHORT→LONG), REVERSE_SHORT (LONG→SHORT).  LONG→REVERSE_SHORT
is the example in the spec: the stance becomes SHORT and the action name
says it was a reverse, not a flat-to-short entry.

THREE LAYERS, ALL REQUIRED TO ENTER OR REVERSE

    STACKED      3+ consecutive price levels whose |level DCD| exceeds
                 a rolling standard deviation of recent print sizes.
                 No hardcoded volume floor.
    ABSORPTION   |effort| (signed size in a microsecond window) is
                 anomalous vs the same rolling σ, and price extension
                 inside that window is at most one tick.
    LIQUIDITY    distance in points from last price to the largest
                 resting block on the live MBP-10 book.

Layer results are packed in a bitfield (STACK=1, ABSORB=2, LIQ=4).
A directional vote is the bitfield equalling 7.

HOT PATH RULES

    * every hot object uses __slots__
    * level map, rings, book cache, sort buffer are preallocated
    * on_tick allocates nothing when stance does not change
    * no lock — DeltaWorker is the only writer
    * the reused Transition and hub frame are overwritten in place
    * stdout is written with a pre-sized bytearray and flushed

Python cannot turn off the GC.  The hot path simply does not create
objects for it to notice.

NOTHING HERE PLACES, SIZES OR TIMES AN ORDER.
"""

from __future__ import annotations

import math
import os
import sys
from array import array
from typing import Any


# --------------------------------------------------------------------------
# Stance / action / layer bits  (ints on purpose)
# --------------------------------------------------------------------------

FLAT = 0
LONG = 1
SHORT = 2

STANCE_NAME = ("FLAT", "LONG", "SHORT")


class Stance:
    """The three market stances.  Ints, not strings."""

    __slots__ = ()
    FLAT = FLAT
    LONG = LONG
    SHORT = SHORT

ACT_NONE = 0
ACT_ENTER_LONG = 1
ACT_ENTER_SHORT = 2
ACT_EXIT_LONG = 3
ACT_EXIT_SHORT = 4
ACT_REVERSE_LONG = 5          # SHORT → LONG
ACT_REVERSE_SHORT = 6         # LONG → SHORT

ACT_NAME = (
    "NONE",
    "ENTER_LONG",
    "ENTER_SHORT",
    "EXIT_LONG",
    "EXIT_SHORT",
    "REVERSE_LONG",
    "REVERSE_SHORT",
)

L_STACK = 1
L_ABSORB = 2
L_LIQ = 4
L_ALL = 7                     # STACK | ABSORB | LIQ


# --------------------------------------------------------------------------
# Rolling σ  — ring of doubles, running sum / sumsq, no list growth
# --------------------------------------------------------------------------

class RollingSigma:
    """Population std of the last `cap` samples.  Fixed ring."""

    __slots__ = ("cap", "buf", "i", "n", "total", "total_sq")

    def __init__(self, cap: int = 64) -> None:
        self.cap = int(cap)
        self.buf = array("d", [0.0] * self.cap)
        self.i = 0
        self.n = 0
        self.total = 0.0
        self.total_sq = 0.0

    def push(self, x: float) -> float:
        """Fold one sample.  Returns the current σ (0.0 until 2 samples)."""
        cap = self.cap
        if self.n == cap:
            old = self.buf[self.i]
            self.total -= old
            self.total_sq -= old * old
        else:
            self.n += 1
        self.buf[self.i] = x
        self.i = (self.i + 1) % cap
        self.total += x
        self.total_sq += x * x
        return self.sigma()

    def sigma(self) -> float:
        n = self.n
        if n < 2:
            return 0.0
        mean = self.total / n
        var = self.total_sq / n - mean * mean
        return math.sqrt(var) if var > 0.0 else 0.0

    def reset(self) -> None:
        for k in range(self.cap):
            self.buf[k] = 0.0
        self.i = 0
        self.n = 0
        self.total = 0.0
        self.total_sq = 0.0


# --------------------------------------------------------------------------
# Open-addressed tick → level-DCD  (no dict)
# --------------------------------------------------------------------------

_EMPTY = 0
_USED = 1
_TOMB = 2


class LevelMap:
    """Fixed-capacity open-addressed map: price-tick → signed volume.

    Keys are integer tick indices (price / tick_size).  Values are the
    running DCD at that price.  Capacity is a power of two so the hash
    is a mask.  Never grows.
    """

    __slots__ = ("cap", "mask", "keys", "vals", "state", "count")

    def __init__(self, cap: int = 128) -> None:
        n = 1
        while n < cap:
            n <<= 1
        self.cap = n
        self.mask = n - 1
        self.keys = array("q", [0] * n)       # signed long long
        self.vals = array("d", [0.0] * n)
        self.state = array("B", [0] * n)
        self.count = 0

    def _probe(self, tick: int) -> int:
        i = tick & self.mask
        first_tomb = -1
        cap = self.cap
        for _ in range(cap):
            st = self.state[i]
            if st == _USED and self.keys[i] == tick:
                return i
            if st == _EMPTY:
                return first_tomb if first_tomb >= 0 else i
            if st == _TOMB and first_tomb < 0:
                first_tomb = i
            i = (i + 1) & self.mask
        return first_tomb if first_tomb >= 0 else 0

    def add(self, tick: int, signed: float) -> float:
        """Accumulate signed size at `tick`.  Returns the new level DCD."""
        i = self._probe(tick)
        if self.state[i] == _USED:
            self.vals[i] += signed
            return self.vals[i]
        if self.count >= (self.cap * 3) // 4:
            # Map is full: evict this slot rather than allocate.
            self.state[i] = _USED
            self.keys[i] = tick
            self.vals[i] = signed
            return signed
        if self.state[i] != _USED:
            self.count += 1
        self.state[i] = _USED
        self.keys[i] = tick
        self.vals[i] = signed
        return self.vals[i]

    def get(self, tick: int) -> float:
        i = tick & self.mask
        cap = self.cap
        for _ in range(cap):
            st = self.state[i]
            if st == _EMPTY:
                return 0.0
            if st == _USED and self.keys[i] == tick:
                return self.vals[i]
            i = (i + 1) & self.mask
        return 0.0

    def pack_sorted(self, out_keys: array, out_vals: array) -> int:
        """Write used (tick, dcd) pairs into preallocated arrays, sorted.

        Insertion sort — n ≤ cap (128).  Returns n.
        """
        n = 0
        cap = self.cap
        for i in range(cap):
            if self.state[i] != _USED:
                continue
            k = self.keys[i]
            v = self.vals[i]
            j = n
            while j > 0 and out_keys[j - 1] > k:
                out_keys[j] = out_keys[j - 1]
                out_vals[j] = out_vals[j - 1]
                j -= 1
            out_keys[j] = k
            out_vals[j] = v
            n += 1
        return n

    def reset(self) -> None:
        for i in range(self.cap):
            self.state[i] = _EMPTY
            self.keys[i] = 0
            self.vals[i] = 0.0
        self.count = 0


# --------------------------------------------------------------------------
# Print ring — absorption window
# --------------------------------------------------------------------------

class PrintRing:
    """Last `cap` prints: ts_us, px, signed size.  Fixed."""

    __slots__ = ("cap", "ts", "px", "sz", "i", "n")

    def __init__(self, cap: int = 256) -> None:
        self.cap = int(cap)
        self.ts = array("q", [0] * self.cap)      # unix µs
        self.px = array("d", [0.0] * self.cap)
        self.sz = array("d", [0.0] * self.cap)
        self.i = 0
        self.n = 0

    def push(self, ts_us: int, px: float, signed: float) -> None:
        self.ts[self.i] = ts_us
        self.px[self.i] = px
        self.sz[self.i] = signed
        self.i = (self.i + 1) % self.cap
        if self.n < self.cap:
            self.n += 1

    def window(self, now_us: int, span_us: int) -> tuple:
        """(effort, lo, hi, count) over prints with ts >= now_us - span_us.

        Walks backward from the newest slot.  No allocation.
        """
        if self.n == 0:
            return 0.0, 0.0, 0.0, 0
        cutoff = now_us - span_us
        cap = self.cap
        idx = (self.i - 1) % cap
        effort = 0.0
        lo = 0.0
        hi = 0.0
        count = 0
        for _ in range(self.n):
            t = self.ts[idx]
            if t < cutoff:
                break
            p = self.px[idx]
            effort += self.sz[idx]
            if count == 0:
                lo = hi = p
            elif p < lo:
                lo = p
            elif p > hi:
                hi = p
            count += 1
            idx = (idx - 1) % cap
        return effort, lo, hi, count

    def reset(self) -> None:
        self.i = 0
        self.n = 0


# --------------------------------------------------------------------------
# Reused transition record
# --------------------------------------------------------------------------

class Transition:
    """One stance change.  Overwritten in place; never constructed per tick."""

    __slots__ = (
        "ts_ms", "frm", "to", "action",
        "px", "tgt_px", "tgt_side", "tgt_sz", "tgt_dist",
        "stack_run", "stack_side", "dcd", "sigma",
        "effort", "extension", "layers", "seq",
    )

    def __init__(self) -> None:
        self.ts_ms = 0
        self.frm = FLAT
        self.to = FLAT
        self.action = ACT_NONE
        self.px = 0.0
        self.tgt_px = 0.0
        self.tgt_side = 0          # +1 bid (below), -1 ask (above)
        self.tgt_sz = 0.0
        self.tgt_dist = 0.0
        self.stack_run = 0
        self.stack_side = 0
        self.dcd = 0.0
        self.sigma = 0.0
        self.effort = 0.0
        self.extension = 0.0
        self.layers = 0
        self.seq = 0

    def as_dict(self, dest: dict) -> dict:
        """Write into `dest` (reused).  No new dict."""
        dest["kind"] = "stance"
        dest["ts_ms"] = self.ts_ms
        dest["from"] = STANCE_NAME[self.frm]
        dest["to"] = STANCE_NAME[self.to]
        dest["action"] = ACT_NAME[self.action]
        dest["px"] = self.px
        dest["tgt_px"] = self.tgt_px
        dest["tgt_side"] = "bid" if self.tgt_side > 0 else (
            "ask" if self.tgt_side < 0 else "")
        dest["tgt_sz"] = self.tgt_sz
        dest["tgt_dist"] = self.tgt_dist
        dest["stack_run"] = self.stack_run
        dest["stack_side"] = ("buy" if self.stack_side > 0
                              else "sell" if self.stack_side < 0 else "")
        dest["dcd"] = self.dcd
        dest["sigma"] = self.sigma
        dest["effort"] = self.effort
        dest["extension"] = self.extension
        dest["layers"] = self.layers
        dest["seq"] = self.seq
        return dest


# --------------------------------------------------------------------------
# ANSI router — flushable stdout, no logging module
# --------------------------------------------------------------------------

# 32 green, 31 red, 33 yellow, 36 cyan, 0 reset
_CLR = (
    b"\x1b[33m",          # FLAT
    b"\x1b[32m",          # LONG
    b"\x1b[31m",          # SHORT
)
_RESET = b"\x1b[0m"
_CYAN = b"\x1b[36m"


class AnsiRouter:
    """Colour-coded line on stdout.  One reused bytearray."""

    __slots__ = ("enabled", "_buf", "_write")

    def __init__(self, enabled: bool | None = None,
                 write: Any = None) -> None:
        if enabled is None:
            enabled = os.environ.get("PYTEST_CURRENT_TEST") is None
        self.enabled = bool(enabled)
        self._buf = bytearray(512)
        self._write = write or sys.stdout.buffer.write

    def emit(self, tr: Transition) -> None:
        if not self.enabled:
            return
        # Build in a small local tuple of bytes pieces, then one join
        # into the reused buffer.  The join allocates one bytes object
        # per transition — acceptable, because this path is rare.
        frm = STANCE_NAME[tr.frm].encode("ascii")
        to = STANCE_NAME[tr.to].encode("ascii")
        act = ACT_NAME[tr.action].encode("ascii")
        side = b"ASK" if tr.tgt_side < 0 else (b"BID" if tr.tgt_side > 0 else b"-")
        line = (
            b"[" + str(tr.ts_ms).encode("ascii") + b"ms] "
            + _CLR[tr.frm] + frm + _RESET
            + b"\xe2\x86\x92"                       # →
            + _CLR[tr.to] + to + _RESET
            + b" " + _CYAN + act + _RESET
            + b" px=" + format(tr.px, ".2f").encode("ascii")
            + b" tgt=" + format(tr.tgt_px, ".2f").encode("ascii")
            + b" " + side
            + b" x" + format(tr.tgt_sz, ".0f").encode("ascii")
            + b" \xce\x94=" + format(tr.tgt_dist, ".2f").encode("ascii")
            + b" | stack=" + str(tr.stack_run).encode("ascii")
            + b" \xcf\x83=" + format(tr.sigma, ".3f").encode("ascii")
            + b" dcd=" + format(tr.dcd, "+.1f").encode("ascii")
            + b" | absorb effort=" + format(tr.effort, "+.1f").encode("ascii")
            + b" result=" + format(tr.extension, ".2f").encode("ascii")
            + b" | layers=" + format(tr.layers, "#05b").encode("ascii")
            + b" seq=" + str(tr.seq).encode("ascii")
            + b"\n"
        )
        try:
            self._write(line)
            flush = getattr(sys.stdout, "flush", None)
            if flush is not None:
                flush()
        except Exception:                               # noqa: BLE001
            pass


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------

class StateEngine:
    """Strict three-stance machine.  Owned by the DeltaWorker thread.

    `on_tick` is the only mutating entry.  It is not safe to call from
    two threads.  Book is read by walking `tape.book` (atomically rebound
    by the ingest thread) into a private 10-level cache.
    """

    __slots__ = (
        "tick_size", "z", "window_us", "min_run",
        "stance", "seq", "transitions",
        "_sigma", "_levels", "_prints",
        "_sort_k", "_sort_v",
        "_bid_px", "_bid_sz", "_ask_px", "_ask_sz",
        "_n_bid", "_n_ask",
        "_stack_run", "_stack_side", "_stack_lo", "_stack_hi",
        "_stack_run_long", "_stack_run_short",
        "_effort", "_extension", "_absorb_side",
        "_tgt_px", "_tgt_sz", "_tgt_side", "_tgt_dist",
        "_sigma_now", "_dcd_now",
        "_last", "_frame",
        "router",
    )

    def __init__(
        self,
        tick_size: float = 0.25,
        z: float = 1.0,
        window_us: int = 250_000,
        min_run: int = 3,
        ring: int = 64,
        router: AnsiRouter | None = None,
    ) -> None:
        if tick_size <= 0:
            raise ValueError("tick_size must be positive")
        if z <= 0:
            raise ValueError("z must be positive")
        self.tick_size = float(tick_size)
        self.z = float(z)
        self.window_us = int(window_us)
        self.min_run = int(min_run)

        self.stance = FLAT
        self.seq = 0
        self.transitions = 0

        self._sigma = RollingSigma(ring)
        self._levels = LevelMap(128)
        self._prints = PrintRing(256)
        self._sort_k = array("q", [0] * 128)
        self._sort_v = array("d", [0.0] * 128)

        self._bid_px = array("d", [0.0] * 10)
        self._bid_sz = array("d", [0.0] * 10)
        self._ask_px = array("d", [0.0] * 10)
        self._ask_sz = array("d", [0.0] * 10)
        self._n_bid = 0
        self._n_ask = 0

        self._stack_run = 0
        self._stack_side = 0
        self._stack_lo = 0
        self._stack_hi = 0
        self._stack_run_long = 0
        self._stack_run_short = 0
        self._effort = 0.0
        self._extension = 0.0
        self._absorb_side = 0
        self._tgt_px = 0.0
        self._tgt_sz = 0.0
        self._tgt_side = 0
        self._tgt_dist = 0.0
        self._sigma_now = 0.0
        self._dcd_now = 0.0

        self._last = Transition()
        self._frame: dict = {
            "kind": "stance", "ts_ms": 0, "from": "FLAT", "to": "FLAT",
            "action": "NONE", "px": 0.0, "tgt_px": 0.0, "tgt_side": "",
            "tgt_sz": 0.0, "tgt_dist": 0.0, "stack_run": 0, "stack_side": "",
            "dcd": 0.0, "sigma": 0.0, "effort": 0.0, "extension": 0.0,
            "layers": 0, "seq": 0,
        }
        self.router = router if router is not None else AnsiRouter()

    # -------------------------------------------------------------- feed

    def on_tick(self, trade: Any, dcd: float = 0.0,
                book: Any = None) -> bool:
        """Fold one print.  True only when stance changed.

        `dcd` is the live Diagonal Cumulative Delta from the worker
        (already updated).  `book` is the latest MBP-10 Book or None.
        """
        px = float(getattr(trade, "px", 0.0) or 0.0)
        sz = float(getattr(trade, "sz", 0.0) or 0.0)
        if sz <= 0.0 or px <= 0.0:
            return False
        agg = getattr(trade, "aggressor", "buy")
        signed = sz if agg == "buy" else -sz
        ts = float(getattr(trade, "ts", 0.0) or 0.0)
        ts_us = int(ts * 1_000_000.0)
        tick = int(round(px / self.tick_size))

        self._dcd_now = float(dcd)
        self._sigma_now = self._sigma.push(abs(signed))
        self._levels.add(tick, signed)
        self._prints.push(ts_us, px, signed)
        if book is not None:
            self._capture_book(book)

        thresh = self._sigma_now * self.z

        long_bits = 0
        short_bits = 0

        if self._eval_stack(thresh):
            if self._stack_run_long >= self.min_run:
                long_bits |= L_STACK
            if self._stack_run_short >= self.min_run:
                short_bits |= L_STACK

        if self._eval_absorb(ts_us, thresh):
            if self._absorb_side > 0:
                long_bits |= L_ABSORB
            elif self._absorb_side < 0:
                short_bits |= L_ABSORB

        if self._eval_liq(px):
            if self._tgt_side < 0:          # ask block above → long target
                long_bits |= L_LIQ
            elif self._tgt_side > 0:        # bid block below → short target
                short_bits |= L_LIQ

        vote = FLAT
        vote_bits = 0
        if long_bits == L_ALL:
            vote = LONG
            vote_bits = long_bits
        elif short_bits == L_ALL:
            vote = SHORT
            vote_bits = short_bits

        # Exit: stacked + absorb oppose the current stance (liquidity
        # may still be the old target).  Reverse takes precedence when
        # the opposite side has all three.
        action = ACT_NONE
        new_stance = self.stance
        if vote == LONG and self.stance != LONG:
            if self.stance == SHORT:
                action = ACT_REVERSE_LONG
            else:
                action = ACT_ENTER_LONG
            new_stance = LONG
        elif vote == SHORT and self.stance != SHORT:
            if self.stance == LONG:
                action = ACT_REVERSE_SHORT
            else:
                action = ACT_ENTER_SHORT
            new_stance = SHORT
        elif self.stance == LONG and (short_bits & (L_STACK | L_ABSORB)) == (L_STACK | L_ABSORB):
            action = ACT_EXIT_LONG
            new_stance = FLAT
            vote_bits = short_bits
        elif self.stance == SHORT and (long_bits & (L_STACK | L_ABSORB)) == (L_STACK | L_ABSORB):
            action = ACT_EXIT_SHORT
            new_stance = FLAT
            vote_bits = long_bits

        if action == ACT_NONE:
            return False

        self.seq += 1
        self.transitions += 1
        tr = self._last
        tr.ts_ms = ts_us // 1000
        tr.frm = self.stance
        tr.to = new_stance
        tr.action = action
        tr.px = px
        tr.tgt_px = self._tgt_px
        tr.tgt_side = self._tgt_side
        tr.tgt_sz = self._tgt_sz
        tr.tgt_dist = self._tgt_dist
        tr.stack_run = self._stack_run
        tr.stack_side = self._stack_side
        tr.dcd = self._dcd_now
        tr.sigma = self._sigma_now
        tr.effort = self._effort
        tr.extension = self._extension
        tr.layers = vote_bits
        tr.seq = self.seq
        self.stance = new_stance
        self.router.emit(tr)
        return True

    def frame(self) -> dict:
        """Reused hub payload.  Safe to hand to call_soon_threadsafe."""
        return self._last.as_dict(self._frame)

    # -------------------------------------------------------------- layers

    def _eval_stack(self, thresh: float) -> bool:
        n = self._levels.pack_sorted(self._sort_k, self._sort_v)
        long_run = short_run = 0
        long_lo = long_hi = short_lo = short_hi = 0
        i = 0
        # A brand-new σ is 0 — refuse to flag on a zero threshold, or
        # every first print would be a stack of one.
        gate = thresh
        while i < n:
            v = self._sort_v[i]
            if gate <= 0.0 or abs(v) <= gate:
                i += 1
                continue
            sign = 1 if v > 0.0 else -1
            j = i + 1
            while j < n and self._sort_k[j] == self._sort_k[j - 1] + 1:
                w = self._sort_v[j]
                wsign = 1 if w > 0.0 else -1
                if wsign != sign or abs(w) <= gate:
                    break
                j += 1
            run = j - i
            if sign > 0 and run > long_run:
                long_run = run
                long_lo = self._sort_k[i]
                long_hi = self._sort_k[j - 1]
            elif sign < 0 and run > short_run:
                short_run = run
                short_lo = self._sort_k[i]
                short_hi = self._sort_k[j - 1]
            i = j
        self._stack_run_long = long_run
        self._stack_run_short = short_run
        if long_run >= short_run:
            self._stack_run = long_run
            self._stack_side = 1 if long_run >= self.min_run else 0
            self._stack_lo = long_lo
            self._stack_hi = long_hi
        else:
            self._stack_run = short_run
            self._stack_side = -1 if short_run >= self.min_run else 0
            self._stack_lo = short_lo
            self._stack_hi = short_hi
        return long_run >= self.min_run or short_run >= self.min_run

    def _eval_absorb(self, now_us: int, thresh: float) -> bool:
        effort, lo, hi, count = self._prints.window(now_us, self.window_us)
        self._effort = effort
        self._extension = (hi - lo) if count else 0.0
        self._absorb_side = 0
        if count < 3:
            return False
        # Anomalous effort: |sum| exceeds z·σ·√n  (√n scales the print σ
        # up to a window sum without a second hardcoded volume).
        bar = thresh * math.sqrt(count)
        if bar <= 0.0 or abs(effort) <= bar:
            return False
        if self._extension > self.tick_size:
            return False
        # Negative effort (sellers hitting) + no down-move → bids absorbed
        # → LONG.  Positive effort + no up-move → asks absorbed → SHORT.
        self._absorb_side = 1 if effort < 0.0 else -1
        return True

    def _eval_liq(self, px: float) -> bool:
        best_sz = 0.0
        best_p = 0.0
        best_side = 0
        for i in range(self._n_bid):
            sz = self._bid_sz[i]
            if sz > best_sz:
                best_sz = sz
                best_p = self._bid_px[i]
                best_side = 1
        for i in range(self._n_ask):
            sz = self._ask_sz[i]
            if sz > best_sz:
                best_sz = sz
                best_p = self._ask_px[i]
                best_side = -1
        self._tgt_px = best_p
        self._tgt_sz = best_sz
        self._tgt_side = best_side
        self._tgt_dist = (best_p - px) if best_sz > 0.0 else 0.0
        return best_sz > 0.0

    def _capture_book(self, book: Any) -> None:
        bids = getattr(book, "bids", None) or ()
        asks = getattr(book, "asks", None) or ()
        nb = 0
        for lv in bids:
            if nb == 10:
                break
            self._bid_px[nb] = float(getattr(lv, "px", 0.0) or 0.0)
            self._bid_sz[nb] = float(getattr(lv, "sz", 0.0) or 0.0)
            nb += 1
        self._n_bid = nb
        na = 0
        for lv in asks:
            if na == 10:
                break
            self._ask_px[na] = float(getattr(lv, "px", 0.0) or 0.0)
            self._ask_sz[na] = float(getattr(lv, "sz", 0.0) or 0.0)
            na += 1
        self._n_ask = na

    # -------------------------------------------------------------- read

    @property
    def last(self) -> Transition:
        return self._last

    def reset(self) -> None:
        self.stance = FLAT
        self.seq = 0
        self.transitions = 0
        self._sigma.reset()
        self._levels.reset()
        self._prints.reset()
        self._n_bid = 0
        self._n_ask = 0
        self._stack_run = 0
        self._stack_side = 0
        self._stack_run_long = 0
        self._stack_run_short = 0
        self._absorb_side = 0
        self._tgt_side = 0
        self._tgt_sz = 0.0
