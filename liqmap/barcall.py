"""A reading of every closed candle, and a record of how they turned out.

One call per bar, from the three columns the bar already carries and
where it sits against the levels:

    BOOK    what was RESTING -- the imbalance that closed the bar
    PATH    whether price FOLLOWED -- the aggression of the mid's path
    PAID    who CROSSED THE SPREAD -- delta, by the side CME states
    WHERE   whether this happened at a level, and that level's state

Those four are already measured elsewhere. Nothing here recomputes
them, because a second opinion about what a bar's delta was is a
second thing to keep correct.

===========================================================================
A CALL IS A DESCRIPTION. THE LEAN IS NOT A FORECAST.
===========================================================================
Every call carries a `lean` -- long, short, or none -- because that is
the thing a reading is for. It says which way this bar, read on its
own, points. It does not say what happens next, and this module makes
no claim that it does.

The honest part is `Scoreboard`, which keeps every call and fills in
what price actually did afterwards. A reading that has leaned long
four hundred times and been right half of them is a reading worth
nothing, and the only way to know that is to write the calls down
BEFORE the outcome and count. So the panel shows the call and its
record together, always, and the record names the hit rate the lean
would need just to cover costs.

===========================================================================
THE DISAGREEMENTS ARE THE CONTENT
===========================================================================
When all four say the same thing, the call is easy and it is also the
least informative state on the chart: price has already moved, the
book has already tilted, and somebody has already paid. The reading
that is worth having is the one where they come apart --

    SOLD_INTO       the bid filled up and prints hit it. Buying
                    pressure nobody paid for
    BOUGHT_INTO     the mirror
    PAID_NO_MOVE    somebody crossed the spread hard and price did
                    not go. The earliest of these states, because it
                    happens while the move is still being resisted
    UNPAID_PUSH     price travelled with almost nothing printing
                    behind it
    NO_INTENT       the path was two-sided and nothing was decided

-- which is why those have their own names rather than being folded
into a score. A single number would have made them all the same
number.

Nothing here predicts, and nothing here places, sizes or times an
order. It reads a bar that has already closed.
"""

from __future__ import annotations

from dataclasses import dataclass

# Leans.
LONG = "long"
SHORT = "short"
NONE = "none"

# Verdicts. Every closed bar gets exactly one.
CONFIRMED_UP = "CONFIRMED_UP"
CONFIRMED_DOWN = "CONFIRMED_DOWN"
SOLD_INTO = "SOLD_INTO"
BOUGHT_INTO = "BOUGHT_INTO"
PAID_NO_MOVE = "PAID_NO_MOVE"
UNPAID_PUSH = "UNPAID_PUSH"
NO_INTENT = "NO_INTENT"
NO_PRINTS = "NO_PRINTS"

VERDICTS: tuple[str, ...] = (
    CONFIRMED_UP, CONFIRMED_DOWN, SOLD_INTO, BOUGHT_INTO,
    PAID_NO_MOVE, UNPAID_PUSH, NO_INTENT, NO_PRINTS)

# A bar has to carry at least this much traded size before its delta is
# allowed to name the bar. Below it, one lot either way flips the sign,
# and a verdict that turns on a single contract is noise with a label.
MIN_VOL = 10.0

_SAYS = {
    CONFIRMED_UP: "book, path and prints all going up",
    CONFIRMED_DOWN: "book, path and prints all going down",
    SOLD_INTO: "bid filled up and prints hit it -- buying nobody paid for",
    BOUGHT_INTO: "offer stacked and prints lifted it -- selling nobody paid for",
    PAID_NO_MOVE: "somebody paid up and price did not go",
    UNPAID_PUSH: "price travelled with almost nothing printing behind it",
    NO_INTENT: "two-sided path, nothing decided",
    NO_PRINTS: "no trades in this bar at all",
}


@dataclass(frozen=True, slots=True)
class Call:
    """What one closed bar was, read on its own."""

    index: int = 0
    verdict: str = NO_INTENT
    lean: str = NONE
    says: str = ""
    book: str = ""              # green / red / chop
    path: str = ""              # the bar's aggression label
    paid: float = 0.0           # delta
    volume: float = 0.0
    where: str = ""             # "" / "low" / "high"
    level_state: str = ""       # untested / held / broken / failed
    price: float = 0.0          # the bar's close, for scoring
    ts: float = 0.0

    @property
    def at_level(self) -> bool:
        return bool(self.where)

    def as_dict(self) -> dict:
        return {"index": self.index, "verdict": self.verdict,
                "lean": self.lean, "says": self.says, "book": self.book,
                "path": self.path, "paid": self.paid,
                "volume": self.volume, "where": self.where,
                "level_state": self.level_state, "price": self.price,
                "ts": self.ts}


def _columns(bar) -> tuple[str, int, int]:
    """(book, path sign, paid sign) for one bar, each in -1/0/+1."""
    book = getattr(bar, "direction", "chop")
    agg = getattr(bar, "aggression", "")
    path = 1 if agg == "BUY_PRICE_AGGRESSIVE" else (
        -1 if agg == "SELL_PRICE_AGGRESSIVE" else 0)
    # A thin bar's delta is not allowed to name it: below MIN_VOL one
    # lot either way flips the sign.
    vol = float(getattr(bar, "volume", 0.0) or 0.0)
    d = float(getattr(bar, "delta", 0.0) or 0.0)
    paid = 0 if vol < MIN_VOL else (1 if d > 0 else (-1 if d < 0 else 0))
    return book, path, paid


def call_bar(bar, index: int = 0, where: str = "",
             level_state: str = "") -> Call:
    """Read one closed bar. `where`/`level_state` come from the levels.

    The order of the tests is the order of how much each state tells
    you, not the order they are easiest to check. A disagreement is
    decided before an agreement, because a bar where the columns come
    apart is a different event from one where they do not, and folding
    it into the agreeing case would lose it.
    """
    book, path, paid = _columns(bar)
    vol = float(getattr(bar, "volume", 0.0) or 0.0)
    d = float(getattr(bar, "delta", 0.0) or 0.0)
    book_sign = 1 if book == "green" else (-1 if book == "red" else 0)

    if vol < MIN_VOL:
        verdict, lean = NO_PRINTS, NONE
    elif book_sign > 0 and paid < 0:
        verdict, lean = SOLD_INTO, SHORT
    elif book_sign < 0 and paid > 0:
        verdict, lean = BOUGHT_INTO, LONG
    elif paid and path == 0:
        # Somebody crossed the spread and the path stayed two-sided.
        verdict = PAID_NO_MOVE
        lean = LONG if paid > 0 else SHORT
    elif path and paid == 0:
        verdict, lean = UNPAID_PUSH, NONE
    elif path > 0 and paid > 0:
        verdict, lean = CONFIRMED_UP, LONG
    elif path < 0 and paid < 0:
        verdict, lean = CONFIRMED_DOWN, SHORT
    elif path and paid and path != paid:
        # Path one way, prints the other, book not deciding.
        verdict = SOLD_INTO if paid < 0 else BOUGHT_INTO
        lean = SHORT if paid < 0 else LONG
    else:
        verdict, lean = NO_INTENT, NONE

    says = _SAYS[verdict]
    if where:
        side = "a swing low" if where == "low" else "a swing high"
        state = f" that has {level_state}" if level_state else ""
        says = f"{says}, at {side}{state}"

    return Call(index=index, verdict=verdict, lean=lean, says=says,
                book=book, path=getattr(bar, "aggression", ""),
                paid=d, volume=vol, where=where,
                level_state=level_state,
                price=float(getattr(bar, "close", 0.0)),
                ts=float(getattr(bar, "ts_close", 0.0)))


def call_all(bars, context=None) -> list[Call]:
    """Read every bar. `context` is one dict per bar, or None."""
    out = []
    for i, b in enumerate(bars):
        ctx = (context[i] if context and i < len(context) else None) or {}
        out.append(call_bar(b, index=i, where=ctx.get("side", ""),
                            level_state=ctx.get("status", "")))
    return out


# --------------------------------------------------------- the scoreboard

@dataclass(frozen=True, slots=True)
class Record:
    """How one verdict, leaning one way, has actually done."""

    verdict: str = ""
    lean: str = NONE
    n: int = 0
    hits: int = 0
    misses: int = 0
    flat: int = 0
    mean_ticks: float = 0.0

    @property
    def decided(self) -> int:
        """Calls that went somewhere. A flat outcome is neither."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Share of DECIDED calls that went the lean's way.

        Flat outcomes are out of the denominator, not counted as
        misses. Over three bars of a one-tick market a large share of
        moves are exactly zero, and burying those in the denominator
        drags every hit rate toward a number that says nothing about
        direction. They are reported separately instead, because how
        often a reading goes nowhere is worth knowing on its own.
        """
        return (self.hits / self.decided) if self.decided else 0.0

    def as_dict(self) -> dict:
        return {"verdict": self.verdict, "lean": self.lean, "n": self.n,
                "hits": self.hits, "misses": self.misses,
                "flat": self.flat, "decided": self.decided,
                "hit_rate": round(self.hit_rate, 4),
                "mean_ticks": round(self.mean_ticks, 4)}


def break_even(target: float, stop: float, cost: float) -> float:
    """Share of trades that must go your way just to cover costs.

    The arithmetic that decides whether a reading is worth acting on,
    and the only honest thing to put next to a hit rate. All three
    arguments are in ticks.
    """
    denom = target + stop
    return ((stop + cost) / denom) if denom > 0 else 1.0


def score(calls: list[Call], prices: list[float], horizon: int = 3,
          tick: float = 1.0) -> dict[tuple[str, str], Record]:
    """What price did after each call, by verdict AND lean.

    KEYED ON BOTH, because one verdict can lean either way -- a bar
    where somebody paid up and price did not move leans long, and its
    mirror leans short. Folding them together cancels the two halves
    against each other and prints a row that says nothing while
    looking like the largest sample in the table.

    `prices` is one price per bar in the same order as `calls`; the
    move runs from the called bar to `horizon` bars later, divided by
    `tick`.

    THE LAST `horizon` CALLS ARE NOT SCORED. They have not finished,
    and a zero for them would drag every mean toward nothing while
    making the counts look larger than the evidence behind them.
    """
    acc: dict[tuple[str, str], list] = {}
    n = min(len(calls), len(prices))
    for i in range(max(0, n - horizon)):
        c = calls[i]
        move = (prices[i + horizon] - prices[i]) / (tick or 1.0)
        signed = -move if c.lean == SHORT else move
        row = acc.setdefault((c.verdict, c.lean), [0, 0, 0, 0, 0.0])
        row[0] += 1
        if c.lean == NONE or signed == 0:
            row[3] += 1                      # flat, or nothing to be right about
        elif signed > 0:
            row[1] += 1
        else:
            row[2] += 1
        row[4] += signed
    out: dict[tuple[str, str], Record] = {}
    for key, (cnt, hits, miss, flat, total) in acc.items():
        out[key] = Record(verdict=key[0], lean=key[1], n=cnt, hits=hits,
                          misses=miss, flat=flat,
                          mean_ticks=(total / cnt) if cnt else 0.0)
    return out


# A reading seen fewer times than this is not called either way. The
# hit rate of thirty calls is mostly the hit rate of thirty coin flips.
MIN_CALLS = 100


def scoreboard(calls: list[Call], prices: list[float], horizon: int = 3,
               tick: float = 1.0, target: float = 2.0, stop: float = 2.0,
               cost: float = 0.4) -> dict:
    """The table the panel draws, with the bar each row has to clear.

    `needs` is the hit rate that just covers costs. A row below it is
    losing money however good it looks, and a row above it on sixty
    calls is still sixty coin flips -- `enough` says which.
    """
    rows = score(calls, prices, horizon, tick)
    need = break_even(target, stop, cost)
    out = []
    for r in sorted(rows.values(), key=lambda x: -x.n):
        d = r.as_dict()
        d["needs"] = round(need, 4)
        d["enough"] = r.decided >= MIN_CALLS
        d["clears"] = bool(r.lean != NONE and d["enough"]
                           and r.hit_rate >= need)
        out.append(d)
    return {"horizon": horizon, "needs": round(need, 4),
            "target": target, "stop": stop, "cost": cost,
            "scored": max(0, min(len(calls), len(prices)) - horizon),
            "rows": out}
