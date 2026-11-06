"""Push, measured from the tape alone. No book, nothing cancellable.

Chris:

    "traditional Order Flow Imbalance (OFI) metrics -- because
    standard OFI indicators track modifications to the limit order
    book, which algorithms constantly spoof, flash, and cancel ... To
    measure a true push, you must completely abandon static order
    book snapshots and look at Level 1 transactional data (the tape)"

THE DIAGNOSIS IS RIGHT AND THIS SERVICE HAS THE MEASUREMENTS.

    The OFI here is Cont-Kukanov-Stoikov on MBP-10 touch changes. It
    reads resting size at the inside, and it cannot tell size that
    LEFT BECAUSE IT TRADED from size that left because it was
    pulled. Measured on this tape: the sign changes 3.17 times a bar
    against a zero deadband, and still once a bar at +/-500. That is
    what quote flicker looks like in a number, and it is why the OFI
    was taken out of the fusion rule's decisions.

    Every quantity in this module comes from executed prints. A
    trade cannot be cancelled after the fact.

WHY TWO NUMBERS AND NOT ONE.

    There are two independent facts about a stretch of tape: WHO
    CROSSED THE SPREAD, and WHICH WAY PRICE WENT. Two binaries is
    four states, and a scalar carries one sign, so one number can
    separate at most two of them. Any single formula is therefore
    right about half the quadrants and wrong about the other half --
    which is not a bug to be fixed by better weights, it is an
    information-capacity limit.

    Chris's formula, sum(V_ask*dP) - sum(V_bid*dP), reads a steady
    sell-off and a steady rally as the SAME POSITIVE NUMBER, because
    it factors to sum(dP * delta): the co-movement of price and
    flow. That is a real and useful quantity under its own name. It
    is not direction.

    So:

        PUSH        signed aggressive volume. Who is crossing the
                    spread, in contracts. Unfakeable, unambiguous,
                    and the direction.

        EFFICIENCY  Chris's formula, kept as written and normalised
                    by volume: sum(dP * delta) / sum(volume). In
                    POINTS. Positive when flow and price agree,
                    negative when aggression is being absorbed and
                    price goes the other way.

                    WHAT IT IS NOT, because the obvious reading is
                    wrong: it is not price-per-contract in the sense
                    of "did this cost a lot of size". Both sums
                    scale with volume, so the volume cancels --
                    doubling every print's size leaves the number
                    unchanged (measured: 0.243750 at 25, 50, 100 and
                    400 contracts a print). It is a VOLUME-WEIGHTED
                    AVERAGE PRICE CHANGE per unit of net aggression:
                    it says which way price went while this flow was
                    crossing, weighting the busy moments more
                    heavily. Whether a move was expensive in
                    contracts is a different question and this does
                    not answer it.

    Together they name the quadrant:

        push > 0, eff > 0   buying, and it is working
        push < 0, eff > 0   selling, and it is working
        push > 0, eff < 0   buying into absorption -- a warning
        push < 0, eff < 0   selling into absorption -- a warning

A ROLLING WINDOW, NOT A BAR. The window is the last `period_s`
seconds at every moment, so the reading does not reset at a boundary
and does not depend on where the clock happens to be. A 5-minute bar
that is four minutes old and a rolling five minutes are different
things, and the second is the one that answers "what is happening
now".

UNATTRIBUTED PRINTS ARE COUNTED AND EXCLUDED. Databento writes
side='N' when it cannot say who crossed. Those prints cannot be
signed, so they enter neither sum -- and the share that arrives that
way is published, because a push reading built from 60% attributable
tape deserves to be distrusted and there is no way to know that from
the number itself.

Nothing in this module places, sizes, routes or times an order.
"""

from __future__ import annotations

from collections import deque

PERIOD_S = 300.0


class Kinetic:
    """Rolling push and efficiency, from prints only.

    O(1) amortised per print: one append and however many expiries
    fall out of the back. Nothing is recomputed over the window.
    """

    __slots__ = ("period_s", "aligned", "_bar", "_q", "_buy", "_sell",
                 "_work", "_vol", "_prev_px", "prints", "unknown",
                 "_last_ts")

    def __init__(self, period_s: float = PERIOD_S,
                 aligned: bool = False) -> None:
        self.period_s = max(1.0, float(period_s))
        # BAR-ALIGNED OR ROLLING, and they are different numbers.
        #
        # Rolling answers "what is happening now": the last
        # `period_s` seconds at every instant, never resetting.
        # Aligned answers "what has this CANDLE done": it clears at
        # each boundary so the reading lines up exactly with the 5m
        # candle a rule is reading, which is what Chris asked for --
        # "resets cleanly every 5 minutes to synchronize perfectly
        # with your candle charts".
        #
        # Neither is correct in general. The aligned one is directly
        # comparable to the candle and is nearly empty right after a
        # boundary; the rolling one is always full and never lines
        # up. Both are cheap, so both run and the comparison is
        # itself worth having.
        self.aligned = bool(aligned)
        self._bar = None
        # (ts, signed_volume, volume, work) where work = dP * signed.
        self._q: deque = deque()
        self._buy = 0.0        # contracts lifting the offer, in window
        self._sell = 0.0       # contracts hitting the bid, in window
        self._work = 0.0       # sum of dP * signed volume
        self._vol = 0.0        # attributable contracts in window
        self._prev_px = 0.0
        self.prints = 0
        self.unknown = 0
        self._last_ts = 0.0

    # -- feeding ---------------------------------------------------------

    def on_trade(self, ts: float, px: float, size: float,
                 side: str) -> None:
        """One print. `side` is 'B', 'A' or 'N' as Databento spells it.

        'B' is a trade that lifted the offer -- an aggressive BUY --
        matching the convention the rest of this service uses, where
        the clock windows add 'B' to CVD and subtract 'A'.
        """
        try:
            t = float(ts or 0.0)
            p = float(px or 0.0)
            v = float(size or 0.0)
        except (TypeError, ValueError):
            return
        if p <= 0 or v <= 0:
            return

        self.prints += 1
        self._last_ts = max(self._last_ts, t)

        # ROLL THE WINDOW FIRST, THEN ADD. Expiring afterwards threw
        # away the very print that crossed the boundary: it was
        # appended to the old candle and then cleared with it, so
        # the first print of every new 5-minute bar vanished. For a
        # rolling window the order makes no difference -- a new
        # print is never older than the cutoff -- so one ordering
        # serves both and there is no second path to get wrong.
        self._expire(t)

        # The price move this print is measured against: the change
        # from the previous print. Updated for EVERY print, including
        # unattributed ones -- the market moved whether or not the
        # aggressor could be named, and skipping them here would
        # silently attribute their move to the next signed print.
        dp = (p - self._prev_px) if self._prev_px > 0 else 0.0
        self._prev_px = p

        s = (side or "").strip().upper()[:1]
        if s == "B":
            signed = v
            self._buy += v
        elif s == "A":
            signed = -v
            self._sell += v
        else:
            self.unknown += 1
            return

        work = dp * signed
        self._work += work
        self._vol += v
        self._q.append((t, signed, v, work))

    def _expire(self, now: float) -> None:
        if self.aligned:
            bar = int(now // self.period_s)
            if self._bar is None:
                self._bar = bar
            elif bar != self._bar:
                # A new candle. Everything before the boundary
                # belongs to the last one.
                self._bar = bar
                self._q.clear()
                self._work = self._vol = self._buy = self._sell = 0.0
            return
        cut = now - self.period_s
        q = self._q
        while q and q[0][0] < cut:
            _ts, signed, v, work = q.popleft()
            self._work -= work
            self._vol -= v
            if signed > 0:
                self._buy -= signed
            else:
                self._sell += signed      # signed is negative
        # Floating error accumulates over hundreds of thousands of
        # adds and subtracts. An empty window must read exactly zero,
        # not 3e-12, or `dir` reports a side on no data at all.
        if not q:
            self._work = self._vol = self._buy = self._sell = 0.0

    # -- reading ---------------------------------------------------------

    @property
    def push(self) -> float:
        """Signed aggressive volume in the window. THE DIRECTION."""
        return self._buy - self._sell

    @property
    def efficiency(self) -> float:
        """Volume-weighted average price change, in points.

        Chris's formula, normalised. Positive when flow and price
        agree; negative when aggression is being absorbed.

        NOT sensitive to size: both sums scale with volume so it
        cancels. See the module docstring -- the obvious reading of
        "per contract" is the wrong one.
        """
        return (self._work / self._vol) if self._vol > 0 else 0.0

    @property
    def push_ratio(self) -> float:
        """The push as a fraction of the volume that made it.

        BOUNDED TO [-1, 1] BY CONSTRUCTION, not by scaling or
        tuning: push is buy - sell and volume is buy + sell, and
        both are non-negative, so the magnitude can never exceed
        one. +1.0 means every contract in the window lifted the
        offer; -1.0 means every one hit the bid; 0 is balanced.

        This is the normalised number Chris asked for, and it is
        the honest version of it. A single bounded number CANNOT
        also say whether the push is working -- that needs the
        second reading, for the reason in the module docstring.
        """
        vol = self._buy + self._sell
        if vol <= 0:
            return 0.0
        return max(-1.0, min(1.0, (self._buy - self._sell) / vol))

    @property
    def direction(self) -> str:
        if self.push > 0:
            return "up"
        if self.push < 0:
            return "down"
        return "flat"

    @property
    def working(self) -> bool:
        """Is the push achieving anything?"""
        return self.efficiency > 0

    @property
    def quadrant(self) -> str:
        """Which of the four states the tape is in."""
        d = self.direction
        if d == "flat":
            return "balanced"
        if self.efficiency > 0:
            return "buying works" if d == "up" else "selling works"
        if self.efficiency < 0:
            return "buying absorbed" if d == "up" else "selling absorbed"
        return "no price response"

    @property
    def attributed_pct(self) -> float:
        """Share of prints whose aggressor was known, all time.

        A push built from poorly attributed tape deserves to be
        distrusted, and the number itself cannot say so.
        """
        if not self.prints:
            return 0.0
        return round(100.0 * (self.prints - self.unknown) / self.prints, 1)

    def to_dict(self) -> dict:
        return {
            "period_s": self.period_s,
            "aligned": self.aligned,
            "push": round(self.push, 2),
            "push_ratio": round(self.push_ratio, 4),
            "buy": round(self._buy, 2),
            "sell": round(self._sell, 2),
            "volume": round(self._vol, 2),
            "efficiency": round(self.efficiency, 6),
            "eff_ticks": round(self.efficiency / 0.25, 4),
            "eff_note": "volume-weighted average price change, not "
                        "a cost-per-contract",
            "dir": self.direction,
            "working": self.working,
            "quadrant": self.quadrant,
            "prints": self.prints,
            "unknown": self.unknown,
            "attributed_pct": self.attributed_pct,
            "say": self._say(),
        }

    def _say(self) -> str:
        if not self._vol:
            return "no attributable prints in the window yet"
        d = self.direction
        if d == "flat":
            return (f"{self._buy:,.0f} bought against {self._sell:,.0f} "
                    "sold -- balanced")
        eff = self.efficiency / 0.25
        side = "buyers" if d == "up" else "sellers"
        if self.efficiency == 0:
            return (f"{side} are {abs(self.push):,.0f} contracts ahead "
                    "and price has not responded")
        if self.working:
            return (f"{side} are {abs(self.push):,.0f} contracts ahead "
                    f"and price is moving {abs(eff):.2f} ticks with them")
        return (f"{side} are {abs(self.push):,.0f} contracts ahead and "
                f"price is going the OTHER way -- absorbed")
