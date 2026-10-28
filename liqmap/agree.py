"""Does OFI agreeing with delta precede price? One question, measured.

THE CALL

    UP      the bar closed GREEN (order-flow imbalance crossed
            +threshold) and delta is positive
    DOWN    the bar closed RED and delta is negative
    none    anything else, including every bar with no prints in it

That is the whole rule. No level, no price path, no target, no stop.

THE CALL IS MADE INSIDE THE BAR, NOT AFTER IT.

The moment the running imbalance and the running delta first point the
same way is the moment the reading existed and could have been acted
on. The close is a long time after that -- minutes, at the larger
targets. So the bar records that instant and the price at it
(`call_side`, `call_px`), and every measurement here runs from THAT
price. Measuring from the close would credit the reading with a move
that had already happened before it could be taken, which is the most
flattering mistake available and the easiest one to make.

THE MEASURE

Did price go that way afterwards, and by how much. Measured in ticks
from the price the call fired at to the close a given number of bars
later,
at several horizons, because "afterwards" is not one length and a rule
that works at one bar and not at ten is a different claim from one
that works at both.

UP, FLAT AND DOWN ARE THREE OUTCOMES, NOT TWO. Over a short horizon on
a one-tick market a large share of moves are exactly zero. A flat is
not a failed up-call; it is price not having gone anywhere yet.
Folding flats into the wrong column is the easiest way to make a 50%
rule read as 33% or as 67%, depending which way you fold.

So `right`, `flat` and `wrong` are counted separately, and two
percentages are reported:

    of all calls      right / n          flats count against you
    of decided calls  right / (right+wrong)   flats set aside

Both are true; they answer different questions, and quoting one
without the other is where this kind of measurement usually goes
wrong.

WHAT THIS IS NOT

It is not a backtest and it does not model a trade. Nothing here
knows about fills, costs, size or risk. It answers "did price go the
way the bar said" and refuses to be read as anything more.

Nothing here predicts, and nothing here places, sizes or times an
order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

UP = "up"
DOWN = "down"
NOCALL = ""

# Traded size a bar must carry before its delta is allowed to speak.
# Below it one contract flips the sign, and a call that turns on a
# single lot is noise with a direction attached.
MIN_VOL = 10.0

# Horizons in bars. Not minutes: these bars are not time, and a figure
# in minutes would invite the chart to be read as a timeframe chart.
HORIZONS: tuple[int, ...] = (1, 2, 3, 5, 10, 20)


def call_of(bar, min_vol: float = MIN_VOL) -> str:
    """UP, DOWN, or no call. The entire rule.

    Prefers the INTRABAR call the builder latched -- the first instant
    inside the bar when imbalance and delta agreed. A bar that only
    carries its closing state (one written before the builder recorded
    the instant, or restored from an older database) falls back to the
    closed-bar form of the same rule, which is the same question asked
    later.
    """
    side = getattr(bar, "call_side", "") or ""
    if side in (UP, DOWN):
        return side
    vol = float(getattr(bar, "volume", 0.0) or 0.0)
    if vol < min_vol:
        return NOCALL
    d = float(getattr(bar, "delta", 0.0) or 0.0)
    direction = getattr(bar, "direction", "")
    if direction == "green" and d > 0:
        return UP
    if direction == "red" and d < 0:
        return DOWN
    return NOCALL


def entry_of(bar) -> float:
    """The price the call fired at, or the close if it never fired.

    Measuring from the close would credit the reading with whatever
    price did between the call and the end of the bar -- a move that
    had already happened before it could be taken.
    """
    px = float(getattr(bar, "call_px", 0.0) or 0.0)
    return px if px > 0 else float(getattr(bar, "close", 0.0))


@dataclass
class Row:
    """One call direction at one horizon."""

    call: str = UP
    horizon: int = 1
    n: int = 0
    right: int = 0
    flat: int = 0
    wrong: int = 0
    total_ticks: float = 0.0        # signed the call's way
    best: float = 0.0
    worst: float = 0.0
    moves: list = field(default_factory=list, repr=False)

    @property
    def decided(self) -> int:
        return self.right + self.wrong

    @property
    def pct_of_all(self) -> float:
        """Share of calls that went the right way, flats counted against."""
        return (self.right / self.n) if self.n else 0.0

    @property
    def pct_of_decided(self) -> float:
        """Share that went the right way once flats are set aside."""
        return (self.right / self.decided) if self.decided else 0.0

    @property
    def mean_ticks(self) -> float:
        return (self.total_ticks / self.n) if self.n else 0.0

    @property
    def median_ticks(self) -> float:
        if not self.moves:
            return 0.0
        xs = sorted(self.moves)
        mid = len(xs) // 2
        return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2.0

    def as_dict(self) -> dict:
        return {"call": self.call, "horizon": self.horizon, "n": self.n,
                "right": self.right, "flat": self.flat,
                "wrong": self.wrong, "decided": self.decided,
                "pct_of_all": round(self.pct_of_all, 4),
                "pct_of_decided": round(self.pct_of_decided, 4),
                "mean_ticks": round(self.mean_ticks, 4),
                "median_ticks": round(self.median_ticks, 4),
                "best": round(self.best, 2),
                "worst": round(self.worst, 2)}


def measure(bars, horizons=HORIZONS, tick: float = 1.0,
            min_vol: float = MIN_VOL, keep_moves: bool = True) -> list[Row]:
    """Every call in `bars`, and what price did after it.

    A CALL IS ONLY COUNTED AT A HORIZON IT HAS FINISHED. The last `h`
    bars have no outcome at horizon `h`, so they are left out of that
    horizon rather than given a zero -- a zero would land in the flat
    column and drag the mean toward nothing while making the count look
    larger than the evidence.
    """
    rows: dict[tuple[str, int], Row] = {}
    n = len(bars)
    for i in range(n):
        c = call_of(bars[i], min_vol)
        if not c:
            continue
        close = entry_of(bars[i])
        for h in horizons:
            if i + h >= n:
                continue
            later = float(getattr(bars[i + h], "close", 0.0))
            move = (later - close) / (tick or 1.0)
            signed = move if c == UP else -move
            r = rows.setdefault((c, h), Row(call=c, horizon=h))
            r.n += 1
            if signed > 0:
                r.right += 1
            elif signed < 0:
                r.wrong += 1
            else:
                r.flat += 1
            r.total_ticks += signed
            r.best = max(r.best, signed)
            r.worst = min(r.worst, signed)
            if keep_moves:
                r.moves.append(signed)
    return [rows[k] for k in
            sorted(rows, key=lambda k: (k[0], k[1]))]


def catalogue(bars, horizons=HORIZONS, tick: float = 1.0,
              min_vol: float = MIN_VOL) -> list[dict]:
    """Every call, one row each, with its outcome at every horizon.

    This is the record, not the summary: the thing to open and look
    down when a percentage looks surprising.
    """
    out: list[dict] = []
    n = len(bars)
    for i in range(n):
        c = call_of(bars[i], min_vol)
        if not c:
            continue
        b = bars[i]
        close = entry_of(b)
        row = {"index": i,
               "ts_close": float(getattr(b, "ts_close", 0.0)),
               "call_ts": float(getattr(b, "call_ts", 0.0) or 0.0),
               "call": c,
               "entry": close,
               "bar_close": float(getattr(b, "close", 0.0)),
               "ofi": int(getattr(b, "ofi", 0) or 0),
               "delta": float(getattr(b, "delta", 0.0) or 0.0),
               "volume": float(getattr(b, "volume", 0.0) or 0.0),
               "trades": int(getattr(b, "trade_count", 0) or 0)}
        for h in horizons:
            if i + h >= n:
                row[f"h{h}"] = None
                continue
            later = float(getattr(bars[i + h], "close", 0.0))
            move = (later - close) / (tick or 1.0)
            row[f"h{h}"] = round(move if c == UP else -move, 4)
        out.append(row)
    return out


def summary(bars, horizons=HORIZONS, tick: float = 1.0,
            min_vol: float = MIN_VOL) -> dict:
    """What the panel prints, and what the script prints."""
    rows = measure(bars, horizons, tick, min_vol)
    called = sum(1 for b in bars if call_of(b, min_vol))
    return {"bars": len(bars), "called": called,
            "share_called": round(called / len(bars), 4) if bars else 0.0,
            "horizons": list(horizons),
            "rows": [r.as_dict() for r in rows]}


# ------------------------------------------- observations inside the bar

def observations(bars) -> list:
    """Every observation across these bars, oldest first.

    An observation is one stretch during which the running imbalance
    and the running delta agreed on a direction. It starts when they
    agree, and ends when they stop agreeing, when they agree the other
    way, or when the bar closes. NOTHING OUTLIVES ITS BAR -- at the
    larger targets a bar is minutes, and an observation that ran past
    it would be answering a question about a bar that no longer exists.
    """
    out = []
    for b in bars:
        out.extend(getattr(b, "episodes", ()) or ())
    return out


def observation_table(bars, points=(1, 2, 3, 4)) -> dict:
    """Win, loss and unresolved at each distance, plus how it behaved.

    UNRESOLVED IS NOT A DRAW AND IT IS NOT A LOSS. It means the bar
    ended before price got that far either way, and at four points it
    will often be most of the sample. Reporting it as anything else
    is how a rule that rarely reaches its distance comes to look like
    a rule that reaches it and wins.

    The chatter figures are in here for the same reason. An
    observation three updates long cannot travel two points, so if the
    median is three the table below it is not measuring the rule -- it
    is measuring a gate that is set too low.
    """
    eps = observations(bars)
    n_bars = len(bars)
    rows = []
    for pts in points:
        win = sum(1 for e in eps if e.outcome(pts) == "win")
        loss = sum(1 for e in eps if e.outcome(pts) == "loss")
        unres = len(eps) - win - loss
        dec = win + loss
        rows.append({
            "points": pts, "n": len(eps), "win": win, "loss": loss,
            "unresolved": unres, "decided": dec,
            "pct_of_decided": round(win / dec, 4) if dec else 0.0,
            "pct_resolved": round(dec / len(eps), 4) if eps else 0.0,
        })
    ups = sorted(e.updates for e in eps)
    ends: dict = {}
    for e in eps:
        ends[e.end or "open"] = ends.get(e.end or "open", 0) + 1
    return {
        "bars": n_bars, "observations": len(eps),
        "per_bar": round(len(eps) / n_bars, 2) if n_bars else 0.0,
        "silent_bars": sum(1 for b in bars
                           if not (getattr(b, "episodes", ()) or ())),
        "median_updates": ups[len(ups) // 2] if ups else 0,
        "ends": ends,
        "up": sum(1 for e in eps if e.side == UP),
        "down": sum(1 for e in eps if e.side == DOWN),
        "rows": rows,
    }
