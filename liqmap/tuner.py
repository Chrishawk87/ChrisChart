"""The agent proposing changes to itself, and the reasons it usually can't.

SELF-IMPROVEMENT IS MOSTLY A WAY TO FOOL YOURSELF

Point a search at a few dozen trades, let it pick the threshold with the
best record, and it will find one every single time -- on random numbers
too. That is not learning, it is reading tea leaves with a computer. The
found threshold then fails live, and because it was "data-driven" it gets
trusted longer than a guess would have.

Three rules keep this one honest.

    WALK FORWARD.   The threshold is chosen on the OLDER part of the book
                    and scored on the NEWER part it has never seen. A
                    setting that only works on the data that picked it is
                    exactly what this catches, and it is the single most
                    common way a tuner lies.

    BE BOUND.       Every knob has a hard range, a step, and a cap on how
                    far one adoption may move it. A tuner that can move a
                    parameter anywhere will eventually move it somewhere
                    absurd on a run of luck.

    ASK FIRST.      Nothing here changes anything. A proposal is a row with
                    the evidence attached, and it does nothing until it is
                    adopted. The agent you tested is the agent that runs.

THE HONEST LIMIT: THIS BOOK CAN ONLY EVALUATE TIGHTENING

The ledger contains trades the agent TOOK. Raising a threshold asks "how
would these have gone without the weakest of them", and the answer is right
there -- it is a subset of rows we already have. Lowering one asks how the
trades it REFUSED would have gone, and the book has no outcome for those,
because they were never opened.

So loosening is not proposed. Not because it is wrong, but because nothing
here can support it, and a proposal whose evidence is imaginary is worse
than no proposal. Answering it properly needs the stand-aside rows carried
forward to a settled outcome -- which is a real thing to build, and it is
`backtest.py`'s territory, against stored features.

A tightening-only tuner has its own failure mode: a ratchet that filters
until it never trades at all. Hence `MIN_RETAIN` -- a proposal that throws
away more than half the book is refused however good the survivors look.

THE EXIT KNOBS ARE NOT EVALUABLE HERE EITHER

`invalidate_s`, `max_hold_bars` and `stale_s` change what happens DURING a
trade. Re-running them needs the price path, not the outcome, and the
ledger stores outcomes. They are tunable by hand and excluded from
proposals, which is the honest state rather than a silent omission.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .autopilot import KNOBS, Knob, Knobs
from .ledger import Ledger
from . import sweep as sweep_mod

# Positions needed in the fitting half before a candidate is considered.
MIN_FIT = 40
# Positions needed in the held-out half before a result is believed.
MIN_TEST = 30
# A proposal must keep at least this share of the book. Below it, the
# "improvement" is a smaller sample, not a better filter.
MIN_RETAIN = 0.5
# Out-of-sample improvement, in net bps per trade, before it is worth
# proposing. Set above the noise floor of a few dozen scalps.
MIN_GAIN_BPS = 0.75
# Fraction of the book used to CHOOSE the value. The rest scores it.
FIT_SHARE = 0.6
# How far one adoption may move a knob, in steps.
MAX_STEPS = 2


# Which knobs can be judged from closed positions, and how. Each returns
# the stored value a filter would have compared against, or None when the
# position does not carry it (those are excluded rather than assumed).
def _conviction(p: dict) -> float | None:
    v = p.get("conviction")
    return float(v) if isinstance(v, (int, float)) else None


def _breakeven(p: dict) -> float | None:
    v = p.get("breakeven")
    return float(v) if isinstance(v, (int, float)) and v > 0 else None


def _held(p: dict) -> float | None:
    v = (p.get("features") or {}).get("held_s")
    return float(v) if isinstance(v, (int, float)) else None


def _agreeing(p: dict) -> float | None:
    v = p.get("agreeing")
    return float(v) if isinstance(v, (int, float)) else None


# name -> (extractor, keep-if). `keep` answers "would this trade still have
# been taken at this threshold".
EVALUABLE: dict[str, tuple[Callable[[dict], float | None],
                           Callable[[float, float], bool]]] = {
    "min_conviction":   (_conviction, lambda v, t: v >= t),
    "max_breakeven":    (_breakeven,  lambda v, t: v <= t),
    "min_agreement_s":  (_held,       lambda v, t: v >= t),
    "min_agreeing":     (_agreeing,   lambda v, t: v >= t),
}

NOT_EVALUABLE = {
    "invalidate_s": "changes what happens during a trade, not which trades "
                    "are taken — replaying it needs the price path, which "
                    "the ledger does not keep",
    "max_hold_bars": "same: an exit rule cannot be re-run from an outcome",
    "stale_s": "same: an exit rule cannot be re-run from an outcome",
    "tp_bps": "not from the ledger — tuned by the grid instead, which "
              "replays every target against real price bars",
    "sl_bps": "not from the ledger — the grid tests every stop at once, "
              "and a target and a stop are one decision, not two",
}


def _expectancy(rows: Sequence[dict]) -> float:
    """Net basis points per trade. The metric everything is judged on."""
    if not rows:
        return 0.0
    return sum(float(r.get("net_bps") or 0.0) for r in rows) / len(rows)


def _tighter(knob: Knob, current: float) -> list[float]:
    """Candidate values strictly more selective than the current one.

    Capped at MAX_STEPS so one lucky stretch cannot move a threshold across
    its whole range in a single adoption.
    """
    out = []
    for c in knob.candidates():
        if knob.stricter_up and c > current or (not knob.stricter_up
                                                and c < current):
            steps = abs(c - current) / knob.step if knob.step else 99
            if steps <= MAX_STEPS + 1e-9:
                out.append(c)
    return sorted(out, key=lambda c: abs(c - current))


@dataclass
class Candidate:
    param: str
    value: float
    n_fit: int
    fit_metric: float
    n_test: int
    test_metric: float
    base_metric: float
    retain: float

    @property
    def gain(self) -> float:
        return self.test_metric - self.base_metric


def evaluate(param: str, positions: Sequence[dict], knobs: Knobs
             ) -> list[Candidate]:
    """Walk-forward every tighter value of one knob.

    Ordered by out-of-sample gain. An empty list means the book does not
    support changing this knob yet, which is the usual and correct answer.
    """
    spec = EVALUABLE.get(param)
    knob = KNOBS.get(param)
    if spec is None or knob is None:
        return []
    extract, keep = spec
    current = float(getattr(knobs, param))

    rows = [p for p in positions if extract(p) is not None]
    rows.sort(key=lambda p: float(p.get("closed_at") or 0.0))
    if len(rows) < MIN_FIT + MIN_TEST:
        return []

    cut = int(len(rows) * FIT_SHARE)
    fit, test = rows[:cut], rows[cut:]
    if len(fit) < MIN_FIT or len(test) < MIN_TEST:
        return []

    base_test = _expectancy(test)
    out: list[Candidate] = []
    for value in _tighter(knob, current):
        f = [p for p in fit if keep(extract(p), value)]
        if len(f) < MIN_FIT:
            continue
        t = [p for p in test if keep(extract(p), value)]
        if len(t) < MIN_TEST:
            continue
        retain = len(t) / len(test)
        if retain < MIN_RETAIN:
            continue
        out.append(Candidate(param=param, value=value, n_fit=len(f),
                             fit_metric=_expectancy(f), n_test=len(t),
                             test_metric=_expectancy(t),
                             base_metric=base_test, retain=retain))

    # Chosen on the fit half. Scoring by the test half here would be
    # choosing on the data meant to judge the choice, which is the whole
    # error this function exists to avoid.
    out.sort(key=lambda c: -c.fit_metric)
    return out


def _rationale(c: Candidate, knob: Knob, current: float) -> str:
    direction = "tightening" if knob.stricter_up else "loosening the ceiling"
    return (
        f"{knob.name}: {direction} from {current:g} to {c.value:g} "
        f"({knob.note}).\n"
        f"Chosen on the first {c.n_fit} trades, where it averaged "
        f"{c.fit_metric:+.2f}bps a trade.\n"
        f"Then scored on {c.n_test} later trades it had never seen: "
        f"{c.test_metric:+.2f}bps against {c.base_metric:+.2f}bps for the "
        f"current setting — {c.gain:+.2f}bps a trade better.\n"
        f"It keeps {c.retain:.0%} of the book, so the gain is not just a "
        f"smaller sample.")


def propose_all(ledger: Ledger, knobs: Knobs, coin: str | None = None,
                now: float | None = None) -> list[dict[str, Any]]:
    """Look for changes worth asking about. Files the ones that qualify.

    Returns what it filed. An empty list is the expected outcome most of
    the time and is not a failure; it means the book does not yet justify
    changing anything.
    """
    positions = ledger.closed(coin=coin)
    filed = []
    for param in EVALUABLE:
        cands = evaluate(param, positions, knobs)
        if not cands:
            continue
        best = cands[0]
        if best.gain < MIN_GAIN_BPS:
            continue
        knob = KNOBS[param]
        current = float(getattr(knobs, param))
        pid = ledger.propose(
            param=param, current=current, proposed=best.value,
            n_fit=best.n_fit, n_test=best.n_test,
            fit_metric=round(best.fit_metric, 3),
            test_metric=round(best.test_metric, 3),
            base_metric=round(best.base_metric, 3),
            rationale=_rationale(best, knob, current), now=now)
        filed.append({"id": pid, "param": param, "current": current,
                      "proposed": best.value, "gain_bps": round(best.gain, 2),
                      "n_test": best.n_test})
    return filed


def readiness(ledger: Ledger, coin: str | None = None) -> dict[str, Any]:
    """How close the book is to being able to say anything. Plain numbers.

    Asked constantly in the first weeks, and the answer is almost always
    "not yet" -- so it is worth answering precisely rather than with a
    spinner.
    """
    positions = ledger.closed(coin=coin)
    need = MIN_FIT + MIN_TEST
    per: dict[str, Any] = {}
    for param, (extract, _) in EVALUABLE.items():
        have = sum(1 for p in positions if extract(p) is not None)
        per[param] = {"have": have, "need": need, "ready": have >= need,
                      "note": KNOBS[param].note}
    return {
        "closed": len(positions),
        "need_total": need,
        "ready": len(positions) >= need,
        "per_param": per,
        "not_evaluable": NOT_EVALUABLE,
        "note": (
            f"{len(positions)} closed trades. Tuning needs {need} — "
            f"{MIN_FIT} to choose a value on and {MIN_TEST} it has never "
            f"seen to check it against."
            if len(positions) < need else
            f"{len(positions)} closed trades, enough to walk forward."),
    }


# --------------------------------------------------------------------------
# levels, walked forward through the grid
# --------------------------------------------------------------------------

# Signals needed on each side of the split before a level change is worth
# proposing. Lower than the ledger's floors because a signal is cheaper
# than a closed trade -- one lands per candle whether or not it traded.
LEVEL_MIN_FIT = 60
LEVEL_MIN_TEST = 40
# Out-of-sample improvement, in net bps a trade.
LEVEL_MIN_GAIN = 1.0


def propose_levels(ledger: Ledger, knobs: Knobs, signals, bars, *,
                   cost_bps: float = 0.0,
                   horizon: int = sweep_mod.DEFAULT_HORIZON,
                   tp_values=(), sl_values=(), now: float | None = None
                   ) -> dict[str, Any] | None:
    """Pick a target and stop on older signals, score them on newer ones.

    Once the entry is a plain vote, the gates this module was built to
    tighten no longer exist -- there is nothing left to refuse. What
    remains worth tuning is where the levels sit, and the ledger cannot
    judge that: changing a target changes what happens during a trade, not
    which trades were taken.

    The grid can, because it replays levels against real bars. So the same
    discipline moves across: choose on the older half, score on the newer
    half, propose rather than apply.

    The pair moves together. A target that measured well beside a 25bps
    stop is not evidence for that target beside a 10bps one -- adopting
    half a tested pair produces a setting nobody tested.
    """
    if not signals or not bars:
        return None

    ordered = sorted(signals, key=lambda s: s.ts)
    cut = int(len(ordered) * FIT_SHARE)
    fit, test = ordered[:cut], ordered[cut:]
    if len(fit) < LEVEL_MIN_FIT or len(test) < LEVEL_MIN_TEST:
        return None

    tps = list(tp_values) or sweep_mod.axis(5, 50, 5)
    sls = list(sl_values) or sweep_mod.axis(5, 50, 5)

    chosen = sweep_mod.run(fit, bars, tps, sls, cost_bps=cost_bps,
                           horizon=horizon)
    if not chosen.get("ok"):
        return None
    best = chosen["best"]

    # The held-out half, at the chosen pair and at what is running now.
    scored = sweep_mod.run(test, bars, [best["tp_bps"]], [best["sl_bps"]],
                           cost_bps=cost_bps, horizon=horizon)
    current = sweep_mod.run(test, bars, [knobs.tp_bps], [knobs.sl_bps],
                            cost_bps=cost_bps, horizon=horizon)
    if not scored.get("ok") or not current.get("ok"):
        return None

    new_cell = scored["cells"][0]
    now_cell = current["cells"][0]
    if new_cell["n"] < LEVEL_MIN_TEST:
        return None

    gain = new_cell["expectancy"] - now_cell["expectancy"]
    same = (abs(best["tp_bps"] - knobs.tp_bps) < 1e-9
            and abs(best["sl_bps"] - knobs.sl_bps) < 1e-9)
    if same or gain < LEVEL_MIN_GAIN:
        return None

    # A pair whose out-of-sample interval straddles zero is not an
    # improvement worth adopting, however well it did on the fit half.
    lo = new_cell.get("ci_low")
    if lo is None or lo <= 0:
        return None

    rationale = (
        f"target and stop: {knobs.tp_bps:g}/{knobs.sl_bps:g}bps → "
        f"{best['tp_bps']:g}/{best['sl_bps']:g}bps ({new_cell['rr']}R).\n"
        f"Chosen on the first {len(fit)} signals, where that pair averaged "
        f"{best['expectancy']:+.2f}bps a trade.\n"
        f"Then scored on {new_cell['n']} later signals it had never seen: "
        f"{new_cell['expectancy']:+.2f}bps against "
        f"{now_cell['expectancy']:+.2f}bps for what is running now — "
        f"{gain:+.2f}bps a trade better.\n"
        f"Its honest range out of sample is {new_cell['ci_low']:+.2f} to "
        f"{new_cell['ci_high']:+.2f}bps, entirely above zero.\n"
        f"Hit {new_cell['hit_rate']:.0%} against the "
        f"{new_cell['breakeven']:.0%} that pair needs.")

    pid = ledger.propose(
        param="tp_bps", current=knobs.tp_bps, proposed=best["tp_bps"],
        param2="sl_bps", current2=knobs.sl_bps, proposed2=best["sl_bps"],
        n_fit=len(fit), n_test=new_cell["n"],
        fit_metric=round(best["expectancy"], 3),
        test_metric=round(new_cell["expectancy"], 3),
        base_metric=round(now_cell["expectancy"], 3),
        rationale=rationale, now=now)
    return {"id": pid, "param": "tp_bps + sl_bps",
            "current": f"{knobs.tp_bps:g}/{knobs.sl_bps:g}",
            "proposed": f"{best['tp_bps']:g}/{best['sl_bps']:g}",
            "gain_bps": round(gain, 2), "n_test": new_cell["n"]}


def level_readiness(n_signals: int) -> dict[str, Any]:
    """How close the signal history is to supporting a level proposal."""
    need = LEVEL_MIN_FIT + LEVEL_MIN_TEST
    return {"have": n_signals, "need": need, "ready": n_signals >= need,
            "note": (f"{n_signals} signals recorded. Choosing a target and "
                     f"stop needs {need} — {LEVEL_MIN_FIT} to pick on and "
                     f"{LEVEL_MIN_TEST} it has never seen to check against."
                     if n_signals < need else
                     f"{n_signals} signals, enough to walk forward.")}
