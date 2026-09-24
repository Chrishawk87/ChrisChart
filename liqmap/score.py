"""Marking the agent's own homework, with the error bars left in.

THE NUMBER THAT DECIDES IT IS MONEY PER TRADE

Net basis points per trade, costs already out. Positive and the book pays;
negative and it does not, whatever anything else says.

It is worth being blunt about why this and not hit rate, because the first
cut of this file got it wrong and said so on screen: a book losing 2bps a
trade was reported as "paying for itself" because its hit rate cleared the
rate its entries needed.

Both numbers were right. The inference between them was not. Breakeven hit
rate assumes every loser loses exactly the planned risk and every winner
makes exactly the planned target -- true when the levels are the only way
out, and false the moment an exit rule can close a trade somewhere in
between. This agent exits on invalidation, so most of its trades end
nowhere near either level and the identity quietly stops holding.

So: expectancy decides, with its own interval. Hit rate against breakeven
is kept as a diagnostic of the ENTRY SHAPE, which is the honest thing it
measures, and when the two disagree that disagreement is reported -- it is
one of the more useful things in here, because it says the trades are not
reaching the levels they were sized against.

THE OLD HEADING, STILL TRUE OF THE DIAGNOSTIC

A hit rate on its own says nothing. 40% is excellent on a 3R swing and
ruinous on a four-tick scalp. The comparison that works across both is what
the trade NEEDED versus what it GOT:

    edge = hit rate achieved  -  hit rate required to break even

Positive and the agent is paying for itself. Negative and it is donating,
however good the individual reads looked.

AND THE SECOND NUMBER IS HOW MUCH OF IT IS NOISE

This is the part that self-testing usually skips, and skipping it is how
people convince themselves a strategy works on thirty trades. Eleven wins
from twenty is 55%, and the honest reading of eleven from twenty is
"somewhere between 32% and 77%" -- which is compatible with a great
strategy and with a coin. Reporting 55% and nothing else is not a
measurement, it is a mood.

So every rate here carries a Wilson interval, and every bucket carries the
sample size that produced it. Where the interval spans the breakeven line,
the verdict says so in words instead of letting a number imply confidence
it has not earned.

WHY WILSON AND NOT THE OBVIOUS ONE

The textbook interval -- p +/- 1.96 * sqrt(p(1-p)/n) -- is wrong exactly
where this tool spends its first month: small n and rates near 0 or 1. It
happily returns a lower bound below zero on 2 wins from 3. Wilson does not,
because it solves for the bounds rather than assuming the answer is
symmetric around the point estimate.

WHAT THE BUCKETS ARE FOR

Overall numbers say whether to keep going. The slices say what to change:
by grade, by how many columns agreed, by exit reason, by runway. The gate
audit is the reverse question -- what the filters REFUSED, and whether
refusing paid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# A bucket below this is shown but never described as telling you anything.
MIN_MEANINGFUL = 30

# Where the agent's overall record stops being a curiosity and starts being
# evidence. Deliberately higher than the per-bucket floor.
MIN_CONCLUSIVE = 100


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Confidence interval for a proportion that behaves at small n.

    Returns (low, high). At n = 0 returns the whole range, which is the
    truthful answer to "what is your hit rate" before anything has settled.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


@dataclass
class Bucket:
    """One slice of the book: how it did, and how sure we can be."""

    label: str
    n: int = 0
    wins: int = 0
    net_bps: float = 0.0
    gross_bps: float = 0.0
    r_sum: float = 0.0
    breakeven_sum: float = 0.0
    mae_sum: float = 0.0
    mfe_sum: float = 0.0
    held_sum: float = 0.0
    # Sum of squares, so the mean can carry an interval. Without one,
    # "+2bps a trade" reads as a fact when it is often a coin toss.
    net_sq: float = 0.0

    def add(self, p: dict[str, Any]) -> None:
        self.n += 1
        net = float(p.get("net_bps") or 0.0)
        self.wins += 1 if net > 0 else 0
        self.net_bps += net
        self.net_sq += net * net
        self.gross_bps += float(p.get("gross_bps") or 0.0)
        self.r_sum += float(p.get("r_multiple") or 0.0)
        self.breakeven_sum += float(p.get("breakeven") or 0.0)
        self.mae_sum += float(p.get("mae_bps") or 0.0)
        self.mfe_sum += float(p.get("mfe_bps") or 0.0)
        self.held_sum += float(p.get("held_s") or 0.0)

    @property
    def hit_rate(self) -> float:
        return self.wins / self.n if self.n else 0.0

    @property
    def needed(self) -> float:
        """Average hit rate these trades required to break even."""
        return self.breakeven_sum / self.n if self.n else 0.0

    @property
    def edge_pts(self) -> float:
        """Achieved minus required, in percentage points. The whole game."""
        return (self.hit_rate - self.needed) * 100.0

    @property
    def expectancy_bps(self) -> float:
        """Net basis points per trade. Costs already taken out."""
        return self.net_bps / self.n if self.n else 0.0

    @property
    def cost_drag_bps(self) -> float:
        return (self.gross_bps - self.net_bps) / self.n if self.n else 0.0

    @property
    def net_sd(self) -> float:
        """Spread of per-trade results. Sample standard deviation."""
        if self.n < 2:
            return 0.0
        mean = self.net_bps / self.n
        var = max(0.0, (self.net_sq - self.n * mean * mean) / (self.n - 1))
        return math.sqrt(var)

    @property
    def expectancy_ci(self) -> tuple[float, float]:
        """Where the true per-trade result honestly sits.

        A mean with no interval invites reading a month of luck as an edge.
        Normal approximation, which is fine at the sample sizes this is
        allowed to draw conclusions from and is never used below them.
        """
        if self.n < 2:
            return (float("-inf"), float("inf"))
        se = self.net_sd / math.sqrt(self.n)
        mean = self.net_bps / self.n
        return (mean - 1.96 * se, mean + 1.96 * se)

    @property
    def profitable(self) -> bool | None:
        """The verdict. True / False / None for "cannot tell yet".

        None is the correct answer far more often than anyone likes.
        """
        if self.n < MIN_MEANINGFUL:
            return None
        lo, hi = self.expectancy_ci
        if lo > 0:
            return True
        if hi < 0:
            return False
        return None

    @property
    def shape_divergence(self) -> str:
        """When hit rate and money disagree, say what that means.

        Clearing the required hit rate and still losing means the trades are
        not reaching the levels they were sized against -- the exits are
        landing somewhere in between. That is a fixable, specific problem
        and it is invisible if only one of the two numbers is reported.
        """
        if self.n < MIN_MEANINGFUL:
            return ""
        clears = self.hit_rate > self.needed
        earns = self.expectancy_bps > 0
        if clears and not earns:
            return (f"hits {self.hit_rate:.0%} against the {self.needed:.0%} "
                    f"its entries needed, and still loses "
                    f"{abs(self.expectancy_bps):.1f}bps a trade — the winners "
                    f"are not reaching the target the risk was sized against. "
                    f"Look at where the exits are landing, not at the entries.")
        if earns and not clears:
            return (f"hits only {self.hit_rate:.0%} against {self.needed:.0%} "
                    f"needed and makes money anyway — the winners are running "
                    f"further than the target they were sized for.")
        return ""

    @property
    def meaningful(self) -> bool:
        return self.n >= MIN_MEANINGFUL

    @property
    def interval(self) -> tuple[float, float]:
        return wilson(self.wins, self.n)

    @property
    def beats_breakeven(self) -> bool | None:
        """True / False / None for "can't tell yet".

        None is the honest answer far more often than people like, and it is
        the answer that stops a good month from being mistaken for an edge.
        """
        if self.n < MIN_MEANINGFUL:
            return None
        lo, hi = self.interval
        need = self.needed
        if lo > need:
            return True
        if hi < need:
            return False
        return None

    def verdict(self) -> str:
        """Money first. Hit rate is a diagnostic, not the verdict."""
        if self.n == 0:
            return "nothing settled yet"
        e = self.expectancy_bps
        if self.n < MIN_MEANINGFUL:
            return (f"{self.n} trades at {e:+.1f}bps — too few to read "
                    f"either way.")
        lo, hi = self.expectancy_ci
        band = f"{lo:+.1f} to {hi:+.1f}bps"
        p = self.profitable
        if p is True:
            return (f"pays: {e:+.1f}bps a trade on {self.n}, and the whole "
                    f"{band} range is above zero.")
        if p is False:
            return (f"loses: {e:+.1f}bps a trade on {self.n}, and even the "
                    f"top of {band} is below zero.")
        return (f"undecided: {e:+.1f}bps a trade on {self.n}, but {band} "
                f"straddles zero — that is noise, not an edge.")

    def to_dict(self) -> dict[str, Any]:
        lo, hi = self.interval
        return {
            "label": self.label, "n": self.n, "wins": self.wins,
            "hit_rate": round(self.hit_rate, 4),
            "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "needed": round(self.needed, 4),
            "edge_pts": round(self.edge_pts, 1),
            "expectancy_bps": round(self.expectancy_bps, 2),
            "total_net_bps": round(self.net_bps, 1),
            "avg_r": round(self.r_sum / self.n, 3) if self.n else 0.0,
            "cost_drag_bps": round(self.cost_drag_bps, 2),
            "avg_mae_bps": round(self.mae_sum / self.n, 2) if self.n else 0.0,
            "avg_mfe_bps": round(self.mfe_sum / self.n, 2) if self.n else 0.0,
            "avg_held_s": round(self.held_sum / self.n, 1) if self.n else 0.0,
            "net_sd": round(self.net_sd, 2),
            "exp_low": (round(self.expectancy_ci[0], 2)
                        if self.n >= 2 else None),
            "exp_high": (round(self.expectancy_ci[1], 2)
                         if self.n >= 2 else None),
            "meaningful": self.meaningful,
            "profitable": self.profitable,
            "beats_breakeven": self.beats_breakeven,
            "divergence": self.shape_divergence,
            "verdict": self.verdict(),
        }


def _runway_bucket(p: dict[str, Any]) -> str:
    r = (p.get("runway_bps") or 0.0)
    t = (p.get("target_bps") or 0.0)
    if t <= 0 or r <= 0:
        return "unknown"
    ratio = r / t
    if ratio < 1.0:
        return "target past the first obstacle"
    if ratio < 2.0:
        return "1–2x room"
    return "2x+ room"


def _grouped(rows: Iterable[dict[str, Any]], key, labeller=str
             ) -> list[dict[str, Any]]:
    buckets: dict[Any, Bucket] = {}
    for p in rows:
        k = key(p)
        if k is None:
            k = "unknown"
        buckets.setdefault(k, Bucket(labeller(k))).add(p)
    return [b.to_dict() for b in
            sorted(buckets.values(), key=lambda b: -b.n)]


def scorecard(positions: Sequence[dict[str, Any]],
              decisions: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
    """Everything the book has to say about itself.

    `positions` are closed paper positions. `decisions` are every decision
    including the stand-asides, which is what makes the gate audit possible.
    """
    overall = Bucket("overall")
    for p in positions:
        overall.add(p)

    # MAE against outcome answers a specific, actionable question: were the
    # losers ever in front? If most stopped trades went meaningfully green
    # first, the stop is not the problem and the target is too far.
    stopped = [p for p in positions if p.get("exit_reason") == "stop"]
    was_ahead = [p for p in stopped
                 if float(p.get("mfe_bps") or 0.0) >= float(
                     p.get("target_bps") or 0.0) * 0.5]

    out: dict[str, Any] = {
        "overall": overall.to_dict(),
        "conclusive": overall.n >= MIN_CONCLUSIVE,
        "min_meaningful": MIN_MEANINGFUL,
        "min_conclusive": MIN_CONCLUSIVE,
        "by_grade": _grouped(positions, lambda p: p.get("grade") or "?",
                             lambda k: f"grade {k}"),
        "by_three_way": _grouped(
            positions, lambda p: p.get("grade_3way") or "?",
            lambda k: f"three-way {k}"),
        "by_agreeing": _grouped(
            positions, lambda p: int(p.get("agreeing") or 0),
            lambda k: f"{k} of 3 agreed"),
        "by_exit": _grouped(positions, lambda p: p.get("exit_reason") or "?",
                            lambda k: f"exit: {k}"),
        "by_runway": _grouped(positions, _runway_bucket),
        "by_interval": _grouped(positions, lambda p: p.get("interval") or "?"),
        "by_coin": _grouped(positions, lambda p: p.get("coin") or "?"),
    }

    out["stops"] = {
        "n": len(stopped),
        "was_ahead": len(was_ahead),
        "share_ahead": round(len(was_ahead) / len(stopped), 3) if stopped else 0.0,
        "note": (
            f"{len(was_ahead)} of {len(stopped)} stopped trades were more "
            f"than halfway to target first — the read was right and the "
            f"target was too far"
            if stopped and len(was_ahead) / len(stopped) > 0.4
            else f"{len(was_ahead)} of {len(stopped)} stopped trades got "
                 f"halfway to target first"),
    }

    out["gates"] = gate_audit(decisions)
    out["exits"] = exit_audit(positions)
    out["headline"] = _headline(overall, out)
    return out


def gate_audit(decisions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """What each filter refused, and how often.

    This is a count, not a verdict. Saying a gate was WRONG needs the
    counterfactual -- what the trade it blocked would have done -- and that
    is a different job, done in `backtest.py` against stored features. What
    this answers is the first question: which gate is doing all the work? A
    gate that fires on 90% of candles is the strategy, whatever the docs
    say it is.
    """
    total = len(decisions)
    aside = [d for d in decisions if d.get("action") == "stand_aside"]
    counts: dict[str, int] = {}
    for d in aside:
        counts[d.get("gate") or "unknown"] = counts.get(
            d.get("gate") or "unknown", 0) + 1
    rows = [{"gate": g, "blocked": n,
             "share_of_asides": round(n / len(aside), 3) if aside else 0.0,
             "share_of_all": round(n / total, 3) if total else 0.0}
            for g, n in sorted(counts.items(), key=lambda kv: -kv[1])]
    return rows


def exit_audit(positions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Was exiting on invalidation the right call?

    The question the exit policy raises and the one the ledger can answer:
    trades closed because the read flipped are compared against the ones
    that ran to a level. If the invalidated ones average worse than the
    stops, the rule is cutting trades that would have come back.
    """
    inval = [p for p in positions if p.get("exit_reason") == "invalidated"]
    stops = [p for p in positions if p.get("exit_reason") == "stop"]
    if not inval:
        return {"n": 0,
                "note": "nothing has been closed on invalidation yet"}

    avg_inval = _mean([float(p.get("net_bps") or 0.0) for p in inval])
    avg_stop = _mean([float(p.get("net_bps") or 0.0) for p in stops])
    saved = avg_inval - avg_stop

    if len(inval) < MIN_MEANINGFUL:
        note = (f"{len(inval)} invalidation exits so far, averaging "
                f"{avg_inval:+.1f}bps — too few to compare against the "
                f"{len(stops)} full stops.")
    elif saved > 0:
        note = (f"invalidation exits average {avg_inval:+.1f}bps against "
                f"{avg_stop:+.1f}bps for full stops — cutting early has "
                f"saved about {saved:.1f}bps a trade over {len(inval)}.")
    else:
        note = (f"invalidation exits average {avg_inval:+.1f}bps against "
                f"{avg_stop:+.1f}bps for full stops — cutting early is "
                f"costing about {abs(saved):.1f}bps a trade. The reads may "
                f"be flickering rather than genuinely flipping.")

    return {"n": len(inval), "n_stops": len(stops),
            "avg_invalidated_bps": round(avg_inval, 2),
            "avg_stopped_bps": round(avg_stop, 2),
            "saved_per_trade_bps": round(saved, 2),
            "meaningful": len(inval) >= MIN_MEANINGFUL,
            "note": note}


def _headline(overall: Bucket, out: dict[str, Any]) -> str:
    """The one sentence to read if you read nothing else. Money first."""
    if overall.n == 0:
        return ("The agent has not closed a position yet. Turn it on and "
                "leave it running — nothing here means anything until "
                f"{MIN_MEANINGFUL} trades have settled.")
    if overall.n < MIN_MEANINGFUL:
        return (f"{overall.n} closed, {overall.expectancy_bps:+.1f}bps a "
                f"trade. Far too few to read. {MIN_MEANINGFUL} is where the "
                f"slices start to mean something and {MIN_CONCLUSIVE} is "
                f"where the overall number does.")
    lo, hi = overall.expectancy_ci
    lead = (f"{overall.n} closed at {overall.expectancy_bps:+.1f}bps a trade "
            f"after cost")
    p = overall.profitable
    if p is True:
        tail = (f", and the honest range {lo:+.1f} to {hi:+.1f}bps sits "
                f"entirely above zero — it is paying for itself.")
    elif p is False:
        tail = (f", and the honest range {lo:+.1f} to {hi:+.1f}bps sits "
                f"entirely below zero — it is not clearing its costs.")
    else:
        tail = (f". The honest range is {lo:+.1f} to {hi:+.1f}bps, which "
                f"straddles zero — no edge shown either way yet.")
    div = overall.shape_divergence
    return lead + tail + (f" Worth knowing: it {div}" if div else "")
