"""The agent deciding for itself: enter, hold, exit, or stand aside.

WHAT CHANGED

Until now the tool answered a question when asked. `suggest()` produced a
call, you took it or you did not, and that was the end of its involvement.
It never had to live with anything.

This makes it commit. Every poll it says one of five things -- go long, go
short, hold what it has, get out, or stand aside -- and every one of those
is written down with the reading behind it. It then has to manage what it
opened: watch the levels, watch whether the reason it entered is still
true, and close for a stated reason.

IT STILL DOES NOT TRADE. Nothing in this file, or anything it calls, places
an order. It keeps a written book of what it would have done, which is the
only way to find out whether it is any good without paying to find out.

THE EXIT IS THE HARD PART, AND IT IS WHERE MOST OF THE EDGE IS

Entries are what people argue about and exits are what decides the number.
This one exits on INVALIDATION: the trade was taken because the book was
thin one way and price was going that way, so when that stops being true
the trade is over, whether or not the stop has been reached. Sitting in a
position whose premise has evaporated, waiting for an arbitrary price to
prove it, is paying full price for information you already have.

The obvious failure mode of that rule is being trigger-happy, which is the
exact problem this agent was built around: a reading flips to flat for one
poll and back. So:

    FLAT IS NOT AGAINST YOU.  A read going quiet is the absence of a
                              signal, not a signal. Only an OPPOSITE read
                              counts as invalidation.

    IT HAS TO PERSIST.        The opposite read must hold continuously for
                              `invalidate_s`. One poll does not close a
                              position. The timer resets the moment the
                              reading stops being against it.

    QUIET IS ITS OWN EXIT.    A position whose read has gone flat and
                              stayed flat is not invalidated, it is dead.
                              That closes on time, recorded as a different
                              reason, because "I was wrong" and "nothing
                              happened" are different outcomes and lumping
                              them together hides both.

ONE POSITION AT A TIME, ONE ENTRY PER CANDLE

Enforced here and again in the ledger's schema. Two entries on one bar
double-count the same read and quietly inflate every average computed from
the book afterwards.

THE KNOBS

Everything the tuner is allowed to move lives in `Knobs`, with a hard range
on each. That list is the whole surface: a proposal naming anything else is
refused. What is deliberately NOT tunable --

    the cost model            an agent that can lower its own assumed
                              costs will discover it is profitable
    the pessimistic tie-break a bar touching both levels counts as a stop;
                              letting that be tuned invents a hit rate
    the one-per-candle rule   sample-count integrity is not a parameter
    anything about orders     there is nothing to tune; there are no orders

That list is enforced by a test, not by good intentions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from typing import Any, Literal

from .ledger import Action, ExitReason, Ledger, OpenPosition, Side
from .vote import Vote, from_payload as vote_from_payload

Direction = Literal["up", "down", "flat"]


# --------------------------------------------------------------------------
# the tunable surface
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Knob:
    """One parameter the agent may change about itself, and its limits."""

    name: str
    default: float
    lo: float
    hi: float
    step: float
    # True when a HIGHER value is more selective. Used to describe a
    # proposal in words rather than leaving you to work out the direction.
    stricter_up: bool
    note: str

    def clamp(self, v: float) -> float:
        return max(self.lo, min(self.hi, float(v)))

    def candidates(self) -> list[float]:
        out, v = [], self.lo
        while v <= self.hi + 1e-9:
            out.append(round(v, 6))
            v += self.step
        return out


KNOBS: dict[str, Knob] = {k.name: k for k in (
    Knob("min_conviction", 0.45, 0.25, 0.80, 0.05, True,
         "how convinced the book must be before it will act"),
    Knob("max_breakeven", 0.70, 0.45, 0.85, 0.05, False,
         "the highest hit rate it will accept needing"),
    Knob("min_agreement_s", 4.0, 0.0, 30.0, 2.0, True,
         "how long book and price must have agreed before entry"),
    Knob("min_agreeing", 2.0, 1.0, 3.0, 1.0, True,
         "how many of the three columns must point the same way"),
    Knob("invalidate_s", 6.0, 2.0, 40.0, 2.0, True,
         "how long the read must be against a position before it exits"),
    Knob("stale_s", 90.0, 20.0, 600.0, 20.0, True,
         "how long a position's read may be flat before it is dead money"),
    # The two that matter most, and the two the ledger cannot judge --
    # changing them changes what happens during a trade. `sweep.py` scores
    # them properly, by replaying every pair against price bars.
    Knob("tp_bps", 20.0, 1.0, 300.0, 1.0, False,
         "how far the target sits, in basis points"),
    Knob("sl_bps", 15.0, 1.0, 300.0, 1.0, True,
         "how far the stop sits, in basis points"),
    # How much of the market's normal volume had to show up behind the
    # move. 1.0 is a normal amount; 2.0 is twice normal.
    Knob("min_effort", 0.0, 0.0, 10.0, 0.25, True,
         "how much volume must be behind the move, against normal"),
    # How early to be flat. Exiting exactly AT the close assumes a fill at
    # the closing print, which is not a price anyone gets.
    Knob("exit_before_s", 5.0, 0.0, 120.0, 1.0, True,
         "how many seconds before the candle closes it gets out"),
)}


@dataclass(frozen=True)
class Knobs:
    """The live settings. Defaults until a proposal is adopted."""

    min_conviction: float = 0.45
    max_breakeven: float = 0.70
    min_agreement_s: float = 4.0
    min_agreeing: float = 2.0
    invalidate_s: float = 6.0
    stale_s: float = 90.0
    tp_bps: float = 20.0
    sl_bps: float = 15.0
    min_effort: float = 0.0
    exit_before_s: float = 5.0

    @classmethod
    def from_overrides(cls, overrides: dict[str, float] | None) -> "Knobs":
        """Build from adopted overrides, ignoring anything not on the list.

        An override for an unknown or out-of-range parameter is dropped
        rather than applied. The database is not a trusted input: a row
        written by an older version, or by hand, must not be able to put the
        agent somewhere its own limits forbid.
        """
        k = cls()
        if not overrides:
            return k
        clean = {}
        for name, val in overrides.items():
            knob = KNOBS.get(name)
            if knob is None:
                continue
            try:
                clean[name] = knob.clamp(float(val))
            except (TypeError, ValueError):
                continue
        return replace(k, **clean) if clean else k

    def to_dict(self) -> dict[str, Any]:
        out = {}
        for f in fields(self):
            knob = KNOBS[f.name]
            v = getattr(self, f.name)
            out[f.name] = {"value": v, "default": knob.default,
                           "lo": knob.lo, "hi": knob.hi,
                           "note": knob.note,
                           "changed": abs(v - knob.default) > 1e-9}
        return out


# --------------------------------------------------------------------------
# a decision
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Decision:
    """One step's answer, in the form it gets written down."""

    action: Action
    reason: str
    side: Side | None = None
    gate: str | None = None
    price: float = 0.0
    position: OpenPosition | None = None
    closed: dict[str, Any] | None = None

    @property
    def acted(self) -> bool:
        return self.action in ("enter_long", "enter_short", "exit")

    def sentence(self) -> str:
        if self.action == "stand_aside":
            return f"Stand aside — {self.reason}"
        if self.action == "hold":
            return f"Hold — {self.reason}"
        if self.action == "exit":
            return f"Exit — {self.reason}"
        word = "LONG" if self.action == "enter_long" else "SHORT"
        return f"{word} — {self.reason}"

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "reason": self.reason,
                "side": self.side, "gate": self.gate,
                "price": self.price, "sentence": self.sentence(),
                "position": self.position.to_dict() if self.position else None,
                "closed": self.closed}


def _dir_of(side: Side) -> Direction:
    return "up" if side == "long" else "down"


def _opposes(read: Direction, side: Side) -> bool:
    """Is this reading AGAINST the position? Flat is not against."""
    if read == "flat":
        return False
    return read != _dir_of(side)


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------

class Autopilot:
    """One market, one timeframe, one position at a time.

    Fed the same payload the panel renders, so what it decides on is exactly
    what you can see. If the agent acted on something the screen does not
    show, neither of us could audit it afterwards.
    """

    def __init__(self, coin: str, interval: str, ledger: Ledger,
                 knobs: Knobs | None = None, size_usd: float = 0.0,
                 raw: bool = True, exit_on_invalidation: bool = False,
                 require_unanimous: bool = False, unit: str = "bps"):
        self.coin = coin
        self.interval = interval
        self.ledger = ledger
        self.knobs = knobs or Knobs()
        self.size_usd = size_usd
        # RAW: the three columns vote, nothing is refused, and the levels
        # are the ones you set. This is the mode `sweep.py` reproduces
        # exactly -- which is the point of it. A live agent whose exits
        # differ from the backtest's makes the backtest decorative.
        self.raw = raw
        # Off by default in raw mode for the same reason: the sweep cannot
        # replay an invalidation exit from bar data, so leaving it on makes
        # the live book and the grid two different strategies.
        self.exit_on_invalidation = exit_on_invalidation
        # Only 3-0. Not two outvoting one, not one with two quiet.
        #
        # This is a gate, and the reason it is allowed back after the rest
        # were removed is that it is YOUR rule with a number on it, not my
        # guess at what a good trade looks like. The difference matters:
        # the shape it refuses is recorded either way, so the ledger can
        # still tell you afterwards what the refusals would have done.
        self.require_unanimous = require_unanimous
        # "bps" or "ticks" -- ten of the market's own increments, where
        # that is how you think about the target.
        self.unit = unit if unit in ("bps", "ticks") else "bps"
        self.position: OpenPosition | None = None
        self._last_step: float = 0.0
        self._entered_candles: set[float] = set()
        # Rehydrate so a redeploy does not abandon a live position.
        for p in ledger.load_open(coin):
            if p.interval == interval:
                self.position = p
                self._entered_candles.add(p.candle_ts)
                break

    # ------------------------------------------------------------- helpers

    def _elapsed(self, now: float) -> float:
        if self._last_step <= 0:
            return 0.0
        return max(0.0, min(now - self._last_step, 60.0))

    def _interval_s(self, payload: dict) -> float:
        s = payload.get("interval_s")
        if isinstance(s, (int, float)) and s > 0:
            return float(s)
        # Fall back to the clock the panel already shows.
        left = payload.get("seconds_left") or 0.0
        frac = payload.get("elapsed_frac")
        if isinstance(frac, (int, float)) and 0 < frac < 1:
            return float(left) / (1.0 - frac)
        return 900.0

    def _deadline(self, pos: OpenPosition, payload: dict) -> float:
        """When this position must be flat, whatever else is happening.

        Taken from the candle it was OPENED on, not from the clock at
        entry: a signal that fires forty seconds before the close gets
        forty seconds, which is the honest amount of time its reading is
        good for.
        """
        end = payload.get("candle_end")
        if isinstance(end, (int, float)) and end > 0 and pos.candle_ts:
            # Only trust it while it still describes the position's own
            # candle; by the next poll the payload has moved on.
            if end - pos.candle_ts <= self._interval_s(payload) * 1.5:
                if end > pos.candle_ts:
                    return float(end)
        if pos.candle_ts:
            return pos.candle_ts + self._interval_s(payload)
        return 0.0

    @staticmethod
    def _interval_word(payload: dict) -> str:
        return str(payload.get("interval") or "")

    def _gate_for(self, payload: dict) -> tuple[str | None, str]:
        """The agent's own entry gates, on top of `suggest`'s.

        `suggest` already refused most candles. These are the extra
        conditions the agent holds itself to, and they are the ones the
        tuner is allowed to move -- which is why they live here and not
        buried in the suggestion.
        """
        k = self.knobs
        conf = payload.get("confirmation") or {}
        three = payload.get("three_way") or {}
        sug = payload.get("suggestion") or {}

        held = float(conf.get("held_s") or 0.0)
        if held < k.min_agreement_s:
            return ("agreement_age",
                    f"book and price have only agreed {held:.0f}s "
                    f"(wants {k.min_agreement_s:.0f}s)")

        agreeing = int(three.get("agreeing") or 0)
        if agreeing < int(k.min_agreeing):
            return ("agreeing",
                    f"only {agreeing} of three columns agree "
                    f"(wants {int(k.min_agreeing)})")

        conv = float(payload.get("conviction") or 0.0)
        if conv < k.min_conviction:
            return ("conviction",
                    f"conviction {conv:.0%} under {k.min_conviction:.0%}")

        be = sug.get("breakeven")
        if isinstance(be, (int, float)) and be > k.max_breakeven:
            return ("breakeven",
                    f"needs {be:.0%} to break even, ceiling is "
                    f"{k.max_breakeven:.0%}")

        return (None, "")

    # ---------------------------------------------------------------- step

    def step(self, payload: dict, now: float,
             high: float = 0.0, low: float = 0.0,
             record: bool = True) -> Decision:
        """Decide once. `high`/`low` are the bar's extremes since entry.

        Levels are checked against the bar range rather than the last print,
        because a stop that was reached between two polls was still reached.
        Only checking the price at poll time gives the agent a quietly
        generous fill model that money will not reproduce.
        """
        dt = self._elapsed(now)
        self._last_step = now
        price = float(payload.get("price") or payload.get("entry") or 0.0)
        if price <= 0:
            book = payload.get("book") or {}
            price = float(book.get("mid") or 0.0)
        candle_ts = float(payload.get("candle_ts") or 0.0)

        ballot = vote_from_payload(payload)
        read_dir: Direction = (ballot.direction if self.raw
                               else ((payload.get("three_way") or {})
                                     .get("direction") or "flat"))

        if self.position is not None:
            d = self._manage(payload, now, dt, price, high, low, read_dir)
        else:
            d = self._consider(payload, now, price, candle_ts, ballot)

        if record:
            self._log(d, payload, now, candle_ts, price)
        return d

    # ------------------------------------------------------------- manage

    def _manage(self, payload: dict, now: float, dt: float, price: float,
                high: float, low: float, read_dir: Direction) -> Decision:
        pos = self.position
        assert pos is not None

        if price > 0:
            pos.mark(price, now)

        # 1. Did price reach a level? Checked first and against the bar's
        #    range, so a level touched between polls still counts.
        hi = high if high > 0 else price
        lo = low if low > 0 else price
        if hi > 0 and lo > 0:
            hit = pos.touched(hi, lo)
            if hit is not None:
                px = pos.target_px if hit == "target" else pos.stop_px
                return self._close(pos, px, hit, now,
                                   "target reached" if hit == "target"
                                   else "stop reached")

        # 2. Is the reason still true? Only an OPPOSITE read counts, and it
        #    has to hold -- one flickering poll does not close a position.
        against = _opposes(read_dir, pos.side)
        if against:
            pos.against_s += dt
        else:
            pos.against_s = 0.0

        if self.exit_on_invalidation and pos.against_s >= self.knobs.invalidate_s:
            return self._close(
                pos, price, "invalidated", now,
                f"the read has been {read_dir} against this "
                f"{pos.side} for {pos.against_s:.0f}s — the reason is gone")

        # 3. Dead money. Not wrong, just nothing happening. Recorded
        #    separately because it has a different fix.
        # THE CANDLE IS THE TRADE'S WHOLE LIFE.
        #
        # A five minute trade lasts five minutes. It was opened on the
        # strength of one candle's book, delta and price, and once that
        # candle closes the reading it rests on no longer exists -- the
        # position is riding a signal that has already expired.
        #
        # This is also what keeps the grid honest. Every candle is its own
        # independent test, so a trade that runs into the next one is
        # borrowing a result the next candle's signal should have earned.
        deadline = self._deadline(pos, payload)
        # Out BEFORE the close, not on it. Exiting at the deadline assumes
        # a fill at the closing print, which is not a price anyone gets --
        # and it leaves the position alive into the moment the next
        # candle's reading starts forming.
        out_at = deadline - self.knobs.exit_before_s if deadline else 0.0
        if out_at and now >= out_at:
            left = max(0.0, deadline - now)
            return self._close(
                pos, price, "candle_end", now,
                f"{left:.0f}s from the close of the "
                f"{self._interval_word(payload)} candle it was opened on — "
                f"out before the bar prints")

        quiet = read_dir == "flat"
        held = now - pos.opened_at
        if (self.exit_on_invalidation and quiet
                and held >= self.knobs.stale_s and abs(pos.mfe_bps) < 1e-9):
            return self._close(pos, price, "time", now,
                               f"flat for {held:.0f}s and never went "
                               f"in front — dead money")

        room = pos.signed_bps(price)
        return Decision(
            action="hold", side=pos.side, price=price, position=pos,
            reason=(f"{room:+.1f}bps, read still {read_dir or 'flat'}"
                    + (f", against {pos.against_s:.0f}s of "
                       f"{self.knobs.invalidate_s:.0f}" if against else "")))

    def _close(self, pos: OpenPosition, px: float, reason: ExitReason,
               now: float, words: str) -> Decision:
        """Close at `px` -- unless `px` cannot possibly be a price.

        Closing at zero is a 10,000 basis point move, and one of those in a
        book of five makes every average downstream meaningless. A missing
        mark is a reason to keep holding and look again next poll, never a
        reason to book a result.
        """
        if px <= 0 or not math.isfinite(px):
            pos.against_s = 0.0
            return Decision(
                action="hold", side=pos.side, price=0.0, position=pos,
                reason=("no usable price this poll — holding rather than "
                        "booking a result against a missing mark"))

        closed = self.ledger.close_position(pos, exit_px=px, reason=reason,
                                            now=now)
        self.position = None
        net = closed.get("net_bps") if closed else None
        tail = f" ({net:+.1f}bps net)" if isinstance(net, (int, float)) else ""
        return Decision(action="exit", side=pos.side, price=px,
                        reason=words + tail, closed=closed)

    # ------------------------------------------------------------ consider

    def _consider(self, payload: dict, now: float, price: float,
                  candle_ts: float, ballot: Vote) -> Decision:
        if candle_ts and candle_ts in self._entered_candles:
            return Decision(action="stand_aside", price=price,
                            gate="one_per_candle",
                            reason="already traded this candle")

        if self.raw:
            return self._consider_raw(payload, now, price, candle_ts, ballot)

        if not payload.get("take"):
            gate = payload.get("blocked_by") or "no_trade"
            return Decision(action="stand_aside", price=price, gate=gate,
                            reason=str(payload.get("detail") or gate))

        gate, why = self._gate_for(payload)
        if gate is not None:
            return Decision(action="stand_aside", price=price, gate=gate,
                            reason=why)

        sug = payload.get("suggestion") or {}
        side: Side = sug.get("side") or payload.get("side")
        entry = float(sug.get("entry") or price or 0.0)
        target = float(sug.get("target_px") or 0.0)
        stop = float(sug.get("stop_px") or 0.0)
        if side not in ("long", "short") or entry <= 0 or target <= 0 or stop <= 0:
            return Decision(action="stand_aside", price=price,
                            gate="incomplete",
                            reason="the call is missing a level")

        three = payload.get("three_way") or {}
        road = payload.get("runway") or {}
        pos = self.ledger.open_position(
            coin=self.coin, interval=self.interval, candle_ts=candle_ts,
            side=side, entry=entry, target_px=target, stop_px=stop,
            target_bps=float(sug.get("target_bps") or 0.0),
            risk_bps=float(sug.get("risk_bps") or 0.0),
            cost_bps=float(sug.get("cost_bps") or 0.0),
            breakeven=float(sug.get("breakeven") or 0.0),
            grade=str(payload.get("grade") or ""),
            grade_3way=str(three.get("grade") or ""),
            agreeing=int(three.get("agreeing") or 0),
            conviction=float(payload.get("conviction") or 0.0),
            runway_bps=float(road.get("clear_bps") or 0.0),
            size_usd=self.size_usd,
            features=self._features(payload), now=now)
        if pos is None:
            # The unique index caught a double entry on this bar.
            self._entered_candles.add(candle_ts)
            return Decision(action="stand_aside", price=price,
                            gate="one_per_candle",
                            reason="already traded this candle")

        self.position = pos
        self._entered_candles.add(candle_ts)
        if len(self._entered_candles) > 512:
            self._entered_candles = set(
                sorted(self._entered_candles)[-256:])

        return Decision(
            action="enter_long" if side == "long" else "enter_short",
            side=side, price=entry, position=pos,
            reason=(f"{sug.get('target_ticks', '?')} ticks from "
                    f"{entry:,.6g}, invalid at {stop:,.6g} — "
                    f"{three.get('agreeing', 0)} of three agree, "
                    f"needs {float(sug.get('breakeven') or 0):.0%}"))

    def _consider_raw(self, payload: dict, now: float, price: float,
                      candle_ts: float, ballot: Vote) -> Decision:
        """Book, delta and price vote. Nothing else gets a say.

        The only thing that produces no trade is all three columns flat,
        and that is the absence of a reading rather than a refusal -- there
        is no direction to be long or short of.
        """
        if ballot.side is None or price <= 0:
            return Decision(action="stand_aside", price=price,
                            gate="no_read", reason=ballot.describe())

        k = self.knobs

        # All three, or nothing.
        if self.require_unanimous and not ballot.unanimous:
            return Decision(
                action="stand_aside", price=price, gate="not_unanimous",
                reason=(f"{ballot.shape()} — {ballot.describe()}; this "
                        f"wants all three pointing the same way"))

        # And somebody had to pay for it. A direction three columns agree
        # on but nobody is trading through is a reading, not a move.
        effort = self._effort(payload)
        if k.min_effort > 0:
            if effort is None:
                return Decision(
                    action="stand_aside", price=price, gate="no_effort_read",
                    reason="no volume reading on this candle yet")
            if effort < k.min_effort:
                return Decision(
                    action="stand_aside", price=price, gate="not_paid_for",
                    reason=(f"only {effort:.0%} of normal volume behind it "
                            f"(wants {k.min_effort:.0%})"))

        tp_bps, sl_bps, unit_note = self._levels(payload, price)
        sgn = 1.0 if ballot.side == "long" else -1.0
        target = price * (1 + sgn * tp_bps / 10_000.0)
        stop = price * (1 - sgn * sl_bps / 10_000.0)
        cost = float(payload.get("cost_bps")
                     or (payload.get("suggestion") or {}).get("cost_bps")
                     or 0.0)
        denom = tp_bps + sl_bps
        breakeven = ((sl_bps + cost) / denom) if denom else 0.0

        feats = self._features(payload)
        feats.update({"vote_shape": ballot.shape(), "vote_net": ballot.net,
                      "vote_book": ballot.book, "vote_delta": ballot.delta,
                      "vote_price": ballot.price,
                      "vote_against": ballot.against,
                      "tp_bps": tp_bps, "sl_bps": sl_bps,
                      "unit": self.unit, "tick": payload.get("tick"),
                      "effort": effort,
                      "unanimous_required": self.require_unanimous})

        road = payload.get("runway") or {}
        pos = self.ledger.open_position(
            coin=self.coin, interval=self.interval, candle_ts=candle_ts,
            side=ballot.side, entry=price, target_px=target, stop_px=stop,
            target_bps=tp_bps, risk_bps=sl_bps, cost_bps=cost,
            breakeven=breakeven,
            grade=ballot.shape(), grade_3way=ballot.shape(),
            agreeing=ballot.agreeing, conviction=abs(ballot.net),
            runway_bps=float(road.get("clear_bps") or 0.0),
            size_usd=self.size_usd, features=feats, now=now)
        if pos is None:
            self._entered_candles.add(candle_ts)
            return Decision(action="stand_aside", price=price,
                            gate="one_per_candle",
                            reason="already traded this candle")

        self.position = pos
        self._entered_candles.add(candle_ts)
        if len(self._entered_candles) > 512:
            self._entered_candles = set(sorted(self._entered_candles)[-256:])

        return Decision(
            action="enter_long" if ballot.side == "long" else "enter_short",
            side=ballot.side, price=price, position=pos,
            reason=(f"{ballot.describe()}"
                    + (f", {effort:.0%} of normal volume"
                       if effort is not None else "")
                    + f" · target {unit_note}"))

    @staticmethod
    def _effort(payload: dict) -> float | None:
        """Volume behind the move, against what this market normally does.

        1.0 is a normal amount for the span. None means the candle has no
        volume reading yet, which is not the same as a quiet one -- and
        treating it as zero would refuse every trade in the first seconds
        of a bar.
        """
        part = (payload.get("confirmation") or {}).get("participation")
        if not isinstance(part, dict):
            return None
        v = part.get("effort")
        return float(v) if isinstance(v, (int, float)) else None

    def _levels(self, payload: dict, price: float
                ) -> tuple[float, float, str]:
        """Target and stop in basis points, whatever unit they were set in.

        Ten ticks on gold and ten ticks on a small-cap perp are different
        numbers of basis points, so the conversion happens per market at
        the moment of entry, from the book's own increment.
        """
        k = self.knobs
        if self.unit != "ticks":
            return (k.tp_bps, k.sl_bps,
                    f"{k.tp_bps:.0f}bps, stop {k.sl_bps:.0f}bps")
        tick = float(payload.get("tick") or 0.0)
        if tick <= 0 or price <= 0:
            # No usable increment: fall back to reading them as basis
            # points rather than inventing a tick size.
            return (k.tp_bps, k.sl_bps,
                    f"{k.tp_bps:.0f}bps, stop {k.sl_bps:.0f}bps "
                    f"(no tick size — read as bps)")
        per = tick / price * 10_000.0
        tp, sl = k.tp_bps * per, k.sl_bps * per
        return (tp, sl,
                f"{k.tp_bps:.0f} ticks ({tp:.1f}bps), stop "
                f"{k.sl_bps:.0f} ticks ({sl:.1f}bps)")

    @staticmethod
    def _features(payload: dict) -> dict[str, Any]:
        """The reading, frozen at entry.

        Stored with the position so the scorecard can slice outcomes by what
        was true when it committed -- not by what is true now, which is the
        mistake that makes every backtest look brilliant.
        """
        three = payload.get("three_way") or {}
        conf = payload.get("confirmation") or {}
        dl = payload.get("delta") or {}
        road = payload.get("runway") or {}
        sug = payload.get("suggestion") or {}
        return {
            "grade": payload.get("grade"),
            "grade_3way": three.get("grade"),
            "agreeing": three.get("agreeing"),
            "book_dir": three.get("book"),
            "delta_dir": three.get("delta"),
            "price_dir": three.get("price"),
            "conviction": payload.get("conviction"),
            "book_score": payload.get("score"),
            "held_s": conf.get("held_s"),
            "flips": conf.get("flips"),
            "effort": (conf.get("participation") or {}).get("effort"),
            "delta_score": dl.get("score"),
            "delta_absorbed": dl.get("absorbed"),
            "delta_working": dl.get("working"),
            "runway_bps": road.get("clear_bps"),
            "runway_open": road.get("open_road"),
            "spread_bps": payload.get("spread_bps"),
            "target_bps": sug.get("target_bps"),
            "risk_bps": sug.get("risk_bps"),
            "breakeven": sug.get("breakeven"),
            "mode": payload.get("mode"),
            "feed_quality": payload.get("feed_quality"),
            "updates_per_s": payload.get("updates_per_s"),
        }

    # --------------------------------------------------------------- log

    def _log(self, d: Decision, payload: dict, now: float,
             candle_ts: float, price: float) -> None:
        three = payload.get("three_way") or {}
        try:
            self.ledger.record_decision(
                coin=self.coin, interval=self.interval, candle_ts=candle_ts,
                action=d.action, price=price or d.price, side=d.side,
                position_id=(d.position.id if d.position
                             else (d.closed or {}).get("id")),
                gate=d.gate, grade=str(payload.get("grade") or ""),
                agreeing=int(three.get("agreeing") or 0),
                conviction=float(payload.get("conviction") or 0.0),
                reason=d.reason,
                features=(self._features(payload)
                          if d.action != "hold" else None),
                now=now)
        except Exception:
            # A logging failure must not take the loop down. The decision
            # already happened; losing the row costs a sample, not a
            # position.
            pass

    # -------------------------------------------------------------- state

    def to_dict(self) -> dict[str, Any]:
        return {"coin": self.coin, "interval": self.interval,
                "position": self.position.to_dict() if self.position else None,
                "knobs": self.knobs.to_dict(),
                "raw": self.raw,
                "exit_on_invalidation": self.exit_on_invalidation,
                "require_unanimous": self.require_unanimous,
                "unit": self.unit,
                "size_usd": self.size_usd}
