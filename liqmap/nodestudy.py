"""What price does after it reaches a volume node. An event study.

THIS IS NOT A BACKTEST AND MUST NOT BECOME ONE

No expectancy, no costs, no verdicts. The output is a conditional
distribution: given that price arrived at a node of a certain kind from a
certain side under certain conditions, how often did each thing happen next,
how far did it go, and how long did it take. What to do with that is the
reader's business.

The one thing this module insists on is the sample size beside every number.
A catalog split by node type, approach, regime, session, volatility and
trend has hundreds of cells, and several will show a striking split from
nothing at all. A frequency without its n is not a finding, it is a
decoration.

CONDITIONS ARE KNOWABLE AT THE TOUCH; OUTCOMES ARE NOT

"Rejected" and "broke through" are things that have happened by the time you
can say them. So they belong in the outcome, never in the condition -- a
table that conditions on the outcome and then reports the outcome is
circular and will look extremely informative.

What a trader knows at the moment price reaches a node: which node it is,
which side they came from, what the prior session's structure was, what the
session and volatility regime are. That is the conditioning set, and nothing
in it is computed from a bar after the touch.

WHERE THE NODES COME FROM

The PRIOR session's completed profile. That is what is on the chart at the
open and it needs no assumption about the future. Using the current
session's own profile would define a level partly from the very move being
measured, which is the standard way an auction study reports beautiful
numbers that do not survive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

TICK = 0.25

# ---------------------------------------------------------------------------
# FIXED CONVENTIONS. Stated, not tuned. Changing one is a new question.
# ---------------------------------------------------------------------------

# How close counts as reaching the node.
TOUCH_TICKS = 2.0

# How far price must leave before the SAME node can register again. Without
# it, one approach that hovers produces fifty touches and the catalog
# becomes a count of how long price loitered rather than how often it came.
REARM_TICKS = 10.0

# How long to follow each touch.
HORIZON_S = 3600.0

# Movement that settles what happened.
REJECT_TICKS = 8.0        # back against the approach, without closing through
BREAK_TICKS = 8.0         # beyond the node, after closing through

# How far a close must be beyond the node to count as through it at all.
#
# The first version armed on ANY close beyond, and on real data 75-88% of
# every cell came back "break_fail" with a median distance of 2 ticks and a
# median time of 0 minutes. That is not a failed break, it is price wobbling
# one tick across the node and back -- ordinary chop at a level, relabelled
# as an event. A trader would not call it anything at all.
THROUGH_TICKS = 2.0

# A node is "strong" if its row carries at least this share of the profile's
# peak row.
STRONG_FRACTION = 0.60

# Volatility split: the session's range against the trailing median.
VOL_LOOKBACK_SESSIONS = 20


@dataclass(frozen=True)
class Touch:
    """Price reached a node. Everything here is knowable at that instant."""

    ts: float
    index: int
    node_price: float
    node_kind: str                 # HVN | LVN | POC | VAH | VAL
    from_below: bool
    price: float

    session: str = "rth"           # rth | overnight
    regime: str = "unknown"        # balance | trend | range
    vol: str = "unknown"           # high | low
    trend_1h: str = "flat"         # up | down | flat
    strength: str = "unknown"      # strong | weak

    @property
    def approach(self) -> str:
        return "from below" if self.from_below else "from above"


@dataclass
class Outcome:
    """What followed. None of this is knowable at the touch."""

    label: str                     # reject | break_go | break_fail | stall
    dist_ticks: float = 0.0        # travel in the resolving direction
    time_s: float = 0.0
    mae_ticks: float = 0.0         # worst against the approach direction
    mfe_ticks: float = 0.0         # best with the approach direction
    returned: bool = False         # came back to the node within the horizon
    resolved: bool = False

    def to_dict(self) -> dict:
        return {"label": self.label, "dist_ticks": round(self.dist_ticks, 1),
                "time_s": round(self.time_s), "mae": round(self.mae_ticks, 1),
                "mfe": round(self.mfe_ticks, 1), "returned": self.returned}


def find_touches(bars: Sequence, nodes: Sequence[tuple[float, str, str]],
                 tick: float = TICK, touch_ticks: float = TOUCH_TICKS,
                 rearm_ticks: float = REARM_TICKS,
                 context: dict | None = None) -> list[Touch]:
    """Every arrival at a node, de-duplicated by the re-arm rule.

    `nodes` are (price, kind, strength) triples, from the PRIOR session.

    A node is armed when price is more than `rearm_ticks` away from it, and
    fires when price comes within `touch_ticks`. Approach direction is taken
    from where price was when the node armed, not from the touching bar --
    a bar that straddles the node has no direction of its own, and reading
    one off it would label half the sample at random.
    """
    ctx = context or {}
    armed: dict[float, bool] = {p: True for p, _, _ in nodes}
    side: dict[float, bool] = {}
    out: list[Touch] = []

    for i, b in enumerate(bars):
        px = b.close
        for node_px, kind, strength in nodes:
            gap = (px - node_px) / tick
            if abs(gap) >= rearm_ticks:
                armed[node_px] = True
                side[node_px] = gap < 0          # below means we come up
                continue
            if not armed.get(node_px):
                continue
            if abs(gap) <= touch_ticks:
                armed[node_px] = False
                out.append(Touch(
                    ts=b.ts, index=i, node_price=node_px, node_kind=kind,
                    from_below=side.get(node_px, gap < 0), price=px,
                    strength=strength,
                    session=ctx.get("session", "rth"),
                    regime=ctx.get("regime", "unknown"),
                    vol=ctx.get("vol", "unknown"),
                    trend_1h=ctx.get("trend_1h", "flat")))
    return out


def resolve(bars: Sequence, touch: Touch, tick: float = TICK,
            horizon_s: float = HORIZON_S,
            reject_ticks: float = REJECT_TICKS,
            break_ticks: float = BREAK_TICKS) -> Outcome:
    """Follow one touch forward and classify what happened.

    The four labels partition the space, so frequencies within a condition
    sum to one:

        reject      moved `reject_ticks` back against the approach without
                    ever closing through the node
        break_go    closed through, then reached `break_ticks` beyond
        break_fail  closed through, then closed back on the original side
        stall       none of the above inside the horizon

    A bar is used whole: highs and lows are read for excursions, but only a
    CLOSE counts as going through. A wick through a node is a test, not an
    acceptance -- the same rule the key-level engine uses, for the same
    reason.
    """
    start = touch.index
    node = touch.node_price
    up = touch.from_below              # the direction of travel on approach
    out = Outcome(label="stall")

    through = False
    best_fav = 0.0
    worst_adv = 0.0

    # The excursions describe THE LEG, and a leg ends when price comes back
    # to the node. Two earlier definitions were both wrong in instructive
    # ways:
    #
    #   stop at the label   -> MFE sat at the 8-tick trigger by
    #                          construction; every break_go in two years
    #                          read "10t" and the column described my rule.
    #
    #   run the full hour   -> MFE and MAE both came back 28-54 ticks for
    #                          every outcome INCLUDING rejects, because
    #                          that is simply how far ES ranges in an hour.
    #                          A reject with 42 ticks of favourable
    #                          excursion is not a reject.
    #
    # Measuring to the return gives the move itself: how far this leg ran
    # before it was over. `back` says that happens within the hour four
    # times in five, and when it does not the horizon caps it.
    left = False
    ended = False

    for j in range(start, len(bars)):
        b = bars[j]
        dt = b.ts - touch.ts
        if dt > horizon_s:
            break

        gap = abs(b.close - node) / tick
        if gap > REARM_TICKS:
            left = True
        elif left and gap <= TOUCH_TICKS:
            out.returned = True
            ended = True

        if not ended:
            fav_hi = (b.high - node) / tick if up else (node - b.low) / tick
            adv_lo = (node - b.low) / tick if up else (b.high - node) / tick
            best_fav = max(best_fav, fav_hi)
            worst_adv = max(worst_adv, adv_lo)

        if out.resolved:
            continue

        # Through, and back, both need a margin. Without one, a one-tick
        # wobble across the node counts as a break and an instant failure.
        beyond = (b.close - node) / tick if up else (node - b.close) / tick
        closed_through = beyond >= THROUGH_TICKS
        closed_back = beyond <= -THROUGH_TICKS

        if not through:
            if closed_through:
                through = True
                continue
            # Rejection only counts while price has never closed through.
            if adv_lo >= reject_ticks:
                out.label = "reject"
                out.dist_ticks = adv_lo
                out.time_s = dt
                out.resolved = True
        else:
            if closed_back:
                out.label = "break_fail"
                out.dist_ticks = best_fav
                out.time_s = dt
                out.resolved = True
            elif fav_hi >= break_ticks:
                out.label = "break_go"
                out.dist_ticks = fav_hi
                out.time_s = dt
                out.resolved = True

    out.mfe_ticks = best_fav
    out.mae_ticks = worst_adv
    if not out.resolved:
        out.dist_ticks = best_fav
        out.time_s = horizon_s

    # `returned` is set in the walk above, where it also ends the leg. It
    # used to be a second pass over the same bars; keeping two loops that
    # both decide when price came back is how they drift apart.
    return out


# --------------------------------------------------------------- catalog

def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    v = sorted(xs)
    m = len(v) // 2
    return v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2.0


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    v = sorted(xs)
    return v[min(len(v) - 1, int(q * len(v)))]


@dataclass
class Cell:
    """One row of the catalog."""

    key: tuple
    label: str
    n: int
    n_condition: int
    freq: float
    median_dist: float
    median_time_s: float
    median_mae: float
    median_mfe: float
    p25_dist: float
    p75_dist: float
    returned_share: float

    @property
    def thin(self) -> bool:
        """Too few to read.

        Thirty is not a magic number, it is the point below which a
        frequency has a confidence interval wider than the differences
        anyone would act on.
        """
        return self.n_condition < 30

    def to_dict(self) -> dict:
        return {"key": self.key, "outcome": self.label, "n": self.n,
                "n_condition": self.n_condition, "freq": round(self.freq, 4),
                "median_dist_ticks": round(self.median_dist, 1),
                "median_time_min": round(self.median_time_s / 60.0, 1),
                "median_mae": round(self.median_mae, 1),
                "median_mfe": round(self.median_mfe, 1),
                "p25_dist": round(self.p25_dist, 1),
                "p75_dist": round(self.p75_dist, 1),
                "returned_share": round(self.returned_share, 3),
                "thin": self.thin}


LABELS = ("reject", "break_go", "break_fail", "stall")


def catalog(pairs: Sequence[tuple[Touch, Outcome]],
            by: Sequence[str] = ("node_kind", "approach")) -> list[Cell]:
    """Conditional outcome table.

    `by` names Touch attributes to condition on. Frequencies sum to one
    within each condition, and every label is emitted even at zero -- an
    absent row reads as "not measured" when it means "never happened", and
    those are different facts.
    """
    groups: dict[tuple, list[tuple[Touch, Outcome]]] = {}
    for t, o in pairs:
        key = tuple(getattr(t, k) for k in by)
        groups.setdefault(key, []).append((t, o))

    out: list[Cell] = []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        rows = groups[key]
        total = len(rows)
        for label in LABELS:
            hit = [(t, o) for t, o in rows if o.label == label]
            dists = [o.dist_ticks for _, o in hit]
            out.append(Cell(
                key=key, label=label, n=len(hit), n_condition=total,
                freq=len(hit) / total if total else 0.0,
                median_dist=_median(dists),
                median_time_s=_median([o.time_s for _, o in hit]),
                median_mae=_median([o.mae_ticks for _, o in hit]),
                median_mfe=_median([o.mfe_ticks for _, o in hit]),
                p25_dist=_pct(dists, 0.25), p75_dist=_pct(dists, 0.75),
                returned_share=(sum(1 for _, o in hit if o.returned)
                                / len(hit) if hit else 0.0)))
    return out


def histogram(values: Sequence[float], lo: float, hi: float,
              buckets: int = 12) -> list[tuple[float, float, int]]:
    """(bucket low, bucket high, count). The distribution, not the median.

    A median of 8 ticks hides whether the move is reliably 8 or is 2 half
    the time and 30 the rest, and those are different markets.
    """
    if buckets <= 0 or hi <= lo:
        return []
    width = (hi - lo) / buckets
    counts = [0] * buckets
    for v in values:
        k = int((v - lo) / width)
        if 0 <= k < buckets:
            counts[k] += 1
        elif v >= hi:
            counts[-1] += 1
    return [(lo + i * width, lo + (i + 1) * width, counts[i])
            for i in range(buckets)]
