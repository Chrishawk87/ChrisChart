"""The agent's own book: does it decide, log, score and tune honestly?

The tests that matter most here are the ones about what it MUST NOT do:
manufacture a hit rate from ambiguous bars, double-count a candle, tune a
knob it was never allowed to touch, or propose a change on evidence it does
not have.
"""

from __future__ import annotations

import ast
import tempfile
import time
from pathlib import Path

import pytest

from liqmap.autopilot import KNOBS, Autopilot, Knobs
from liqmap.ledger import Ledger, OpenPosition
from liqmap import score as score_mod
from liqmap import tuner as tuner_mod


@pytest.fixture()
def ledger(tmp_path: Path) -> Ledger:
    return Ledger(str(tmp_path / "t.db"))


def payload(*, take=True, side="long", entry=100.0, target=101.0,
            stop=99.5, candle_ts=1000.0, conviction=0.8, held_s=10.0,
            agreeing=3, breakeven=0.5, direction="up", grade="A", **kw):
    """A suggest_for payload, in the shape the panel actually receives."""
    d = {
        "ok": True, "take": take, "grade": grade, "candle_ts": candle_ts,
        "conviction": conviction, "score": 0.5, "entry": entry,
        "interval_s": 900.0, "seconds_left": 400.0, "mode": "scalp",
        "three_way": {"direction": direction, "agreeing": agreeing,
                      "grade": "A", "book": direction, "delta": direction,
                      "price": direction},
        "confirmation": {"held_s": held_s, "flips": 0,
                         "candle_strength": 0.6,
                         "participation": {"effort": 0.9}},
        "action": {"score": 0.6},
        "runway": {"clear_bps": 60.0, "open_road": False},
        "delta": {"score": 0.4},
    }
    if take:
        d["suggestion"] = {
            "side": side, "entry": entry, "target_px": target,
            "stop_px": stop, "target_bps": 100.0, "risk_bps": 50.0,
            "cost_bps": 2.0, "breakeven": breakeven, "target_ticks": 10,
        }
    else:
        d["blocked_by"] = kw.get("gate", "conviction")
        d["detail"] = "not enough"
    d.update({k: v for k, v in kw.items() if k != "gate"})
    return d


# --------------------------------------------------------------- the book

def test_opens_and_closes_net_of_cost(ledger):
    pos = ledger.open_position(
        coin="ETH", interval="15m", candle_ts=1.0, side="long", entry=100.0,
        target_px=101.0, stop_px=99.5, target_bps=100.0, risk_bps=50.0,
        cost_bps=2.0, now=0.0)
    assert pos is not None
    out = ledger.close_position(pos, exit_px=101.0, reason="target", now=10.0)
    # 100bps gross, 2bps of round trip taken out. Scoring gross is how a
    # scalping book looks profitable at four ticks and is not.
    assert out["gross_bps"] == pytest.approx(100.0)
    assert out["net_bps"] == pytest.approx(98.0)


def test_one_position_per_candle_enforced_in_the_schema(ledger):
    kw = dict(coin="ETH", interval="15m", candle_ts=1.0, side="long",
              entry=100.0, target_px=101.0, stop_px=99.5, target_bps=100.0,
              risk_bps=50.0)
    assert ledger.open_position(**kw) is not None
    # A retry or a double poll must not be able to book the same read twice.
    assert ledger.open_position(**kw) is None


def test_a_bar_touching_both_levels_counts_as_a_stop():
    pos = OpenPosition(
        id="x", coin="ETH", interval="15m", candle_ts=1.0, side="long",
        entry=100.0, target_px=101.0, stop_px=99.0, target_bps=100.0,
        risk_bps=100.0, cost_bps=0.0, breakeven=0.5, opened_at=0.0)
    # Nothing in OHLC says which came first. Taking the good one invents a
    # hit rate that money will never reproduce.
    assert pos.touched(high=101.5, low=98.5) == "stop"
    assert pos.touched(high=101.5, low=99.5) == "target"
    assert pos.touched(high=100.2, low=99.8) is None


def test_short_levels_are_the_right_way_round():
    pos = OpenPosition(
        id="x", coin="ETH", interval="15m", candle_ts=1.0, side="short",
        entry=100.0, target_px=99.0, stop_px=101.0, target_bps=100.0,
        risk_bps=100.0, cost_bps=0.0, breakeven=0.5, opened_at=0.0)
    assert pos.touched(high=100.1, low=98.9) == "target"
    assert pos.touched(high=101.2, low=99.9) == "stop"
    assert pos.signed_bps(99.0) == pytest.approx(100.0)


def test_open_positions_survive_a_restart(ledger):
    ledger.open_position(coin="ETH", interval="15m", candle_ts=1.0,
                         side="long", entry=100.0, target_px=101.0,
                         stop_px=99.5, target_bps=100.0, risk_bps=50.0)
    # A redeploy that abandons a position leaves a trade with no exit, and
    # a trade with no exit drops silently out of every later average.
    pilot = Autopilot("ETH", "15m", ledger)
    assert pilot.position is not None
    assert pilot.position.entry == 100.0


# ------------------------------------------------------------- deciding

def test_enters_when_everything_passes(ledger):
    p = Autopilot("ETH", "15m", ledger)
    d = p.step(payload(), now=100.0)
    assert d.action == "enter_long"
    assert p.position is not None
    assert ledger.counts()["open"] == 1


def test_stand_aside_is_recorded_with_its_gate(ledger):
    p = Autopilot("ETH", "15m", ledger, raw=False)
    d = p.step(payload(take=False, gate="unconfirmed"), now=100.0)
    assert d.action == "stand_aside"
    assert d.gate == "unconfirmed"
    # Standing aside is a decision and it goes in the book. A strategy that
    # only records its trades cannot say what its filters cost.
    rows = ledger.decisions(action="stand_aside")
    assert len(rows) == 1 and rows[0]["gate"] == "unconfirmed"


def test_its_own_gates_can_refuse_a_tradeable_call(ledger):
    p = Autopilot("ETH", "15m", ledger, raw=False,
                  knobs=Knobs(min_agreement_s=30.0))
    d = p.step(payload(held_s=3.0), now=100.0)
    assert d.action == "stand_aside" and d.gate == "agreement_age"

    p2 = Autopilot("ETH", "15m", ledger, raw=False,
                   knobs=Knobs(min_agreeing=3))
    assert p2.step(payload(agreeing=2, candle_ts=2.0),
                   now=100.0).gate == "agreeing"

    p3 = Autopilot("ETH", "15m", ledger, raw=False,
                   knobs=Knobs(max_breakeven=0.55))
    assert p3.step(payload(breakeven=0.8, candle_ts=3.0),
                   now=100.0).gate == "breakeven"


def test_only_one_entry_per_candle(ledger):
    p = Autopilot("ETH", "15m", ledger)
    assert p.step(payload(), now=100.0).action == "enter_long"
    p.position = None            # pretend it closed
    d = p.step(payload(), now=200.0)
    assert d.action == "stand_aside" and d.gate == "one_per_candle"


# --------------------------------------------------------------- managing

def test_holds_while_the_read_still_agrees(ledger):
    p = Autopilot("ETH", "15m", ledger)
    p.step(payload(), now=100.0)
    d = p.step(payload(candle_ts=1000.0), now=105.0)
    assert d.action == "hold"


def test_raw_voting_is_the_default(ledger):
    """No gates unless asked for, and levels-only exits, so the live book
    and the grid stay the same strategy."""
    p = Autopilot("ETH", "15m", ledger)
    assert p.raw is True
    assert p.exit_on_invalidation is False


def test_flat_is_not_against_you(ledger):
    """The exact problem this agent was built around.

    A reading dropping to flat is the absence of a signal, not a signal.
    Treating it as invalidation closes good positions on a quiet poll.
    """
    p = Autopilot("ETH", "15m", ledger, knobs=Knobs(invalidate_s=4.0),
                  exit_on_invalidation=True)
    p.step(payload(), now=100.0)
    for t in (105.0, 110.0, 115.0, 120.0, 125.0):
        d = p.step(payload(direction="flat"), now=t)
        assert d.action == "hold", f"closed on a flat read at {t}"
    assert p.position.against_s == 0.0


def test_one_opposite_poll_does_not_close_it(ledger):
    p = Autopilot("ETH", "15m", ledger, knobs=Knobs(invalidate_s=10.0),
                  exit_on_invalidation=True)
    p.step(payload(), now=100.0)
    assert p.step(payload(direction="down"), now=103.0).action == "hold"
    # ... and the timer resets the moment it stops being against.
    assert p.step(payload(direction="up"), now=106.0).action == "hold"
    assert p.position.against_s == 0.0


def test_a_persistent_flip_closes_it(ledger):
    p = Autopilot("ETH", "15m", ledger, knobs=Knobs(invalidate_s=8.0),
                  exit_on_invalidation=True)
    p.step(payload(), now=100.0)
    p.step(payload(direction="down"), now=105.0)
    d = p.step(payload(direction="down"), now=110.0)
    assert d.action == "exit"
    assert d.closed["reason"] == "invalidated"
    assert p.position is None


def test_a_level_beats_everything_else(ledger):
    p = Autopilot("ETH", "15m", ledger)
    p.step(payload(), now=100.0)
    # Reached between polls: checked against the bar's range, not the last
    # print, or the agent gets a fill model money will not reproduce.
    d = p.step(payload(direction="down"), now=105.0, high=101.5, low=100.0)
    assert d.action == "exit" and d.closed["reason"] == "target"


def test_a_trade_never_outlives_its_candle(ledger):
    """A five minute trade lasts five minutes.

    The position was opened on one candle's book, delta and price. Once
    that candle closes the reading it rests on no longer exists, and a
    trade allowed to run on is claiming a result the NEXT candle's signal
    should have had to earn.
    """
    p = Autopilot("ETH", "15m", ledger, knobs=Knobs(invalidate_s=999.0))
    p.step(payload(candle_ts=9000.0), now=9100.0)
    assert p.position is not None
    assert p.step(payload(candle_ts=9000.0, direction="flat"),
                  now=9800.0).action == "hold"
    d = p.step(payload(candle_ts=9000.0, direction="up"), now=9901.0)
    assert d.action == "exit" and d.closed["reason"] == "candle_end"


def test_the_deadline_comes_from_the_candle_it_opened_on(ledger):
    """A signal firing forty seconds before the close gets forty seconds,
    not a fresh full bar measured from entry."""
    p = Autopilot("ETH", "15m", ledger, knobs=Knobs(invalidate_s=999.0))
    p.step(payload(candle_ts=9000.0), now=9860.0)
    d = p.step(payload(candle_ts=9000.0), now=9910.0)
    assert d.action == "exit" and d.closed["reason"] == "candle_end"


def test_a_one_minute_trade_lasts_one_minute(ledger):
    p = Autopilot("ETH", "1m", ledger, knobs=Knobs(invalidate_s=999.0))
    pl = dict(payload(candle_ts=600.0), interval_s=60.0, candle_end=660.0)
    p.step(pl, now=610.0)
    assert p.step(pl, now=650.0).action == "hold"
    assert p.step(pl, now=661.0).closed["reason"] == "candle_end"


def test_dead_money_and_being_wrong_are_different_rows(ledger):
    p = Autopilot("ETH", "15m", ledger, exit_on_invalidation=True,
                  knobs=Knobs(stale_s=50.0, invalidate_s=999.0))
    p.step(payload(candle_ts=9000.0), now=9010.0)
    d = p.step(payload(candle_ts=9000.0, direction="flat"), now=9070.0)
    assert d.action == "exit" and d.closed["reason"] == "time"


# ----------------------------------------------------------------- knobs

def test_overrides_outside_the_allowed_range_are_clamped():
    k = Knobs.from_overrides({"min_conviction": 99.0})
    assert k.min_conviction == KNOBS["min_conviction"].hi


def test_unknown_overrides_are_ignored():
    # The database is not a trusted input. A row written by an older build,
    # or by hand, must not put the agent somewhere its limits forbid.
    k = Knobs.from_overrides({"place_orders": 1.0, "min_conviction": 0.6})
    assert not hasattr(k, "place_orders")
    assert k.min_conviction == 0.6


def test_garbage_overrides_do_not_crash_the_agent():
    k = Knobs.from_overrides({"min_conviction": "banana"})
    assert k.min_conviction == Knobs().min_conviction


# ----------------------------------------------------------------- score

def test_wilson_is_honest_about_small_samples():
    lo, hi = score_mod.wilson(11, 20)
    assert lo < 0.40 and hi > 0.70          # 55% means almost nothing
    lo2, hi2 = score_mod.wilson(550, 1000)
    assert hi2 - lo2 < 0.08                 # ... and a lot at n=1000
    # The textbook interval returns a negative lower bound here. Wilson
    # must not.
    assert score_mod.wilson(2, 3)[0] >= 0.0
    assert score_mod.wilson(0, 0) == (0.0, 1.0)


def _pos(net, breakeven=0.5, **kw):
    d = {"net_bps": net, "gross_bps": net + 2, "breakeven": breakeven,
         "r_multiple": net / 50.0, "grade": "A", "agreeing": 3,
         "exit_reason": "target" if net > 0 else "stop",
         "interval": "15m", "coin": "ETH", "closed_at": 0.0,
         "mae_bps": -5.0, "mfe_bps": max(net, 0.0), "held_s": 60.0,
         "target_bps": 100.0, "runway_bps": 200.0, "features": {}}
    d.update(kw)
    return d


def test_a_small_sample_is_reported_as_undecided():
    card = score_mod.scorecard([_pos(10) for _ in range(5)])
    b = card["overall"]
    assert b["beats_breakeven"] is None
    assert "too few" in card["headline"]


def test_a_clear_edge_is_called_clear():
    rows = [_pos(10) for _ in range(80)] + [_pos(-10) for _ in range(20)]
    card = score_mod.scorecard(rows)
    assert card["overall"]["profitable"] is True
    assert "paying for itself" in card["headline"]


def test_a_losing_book_is_called_losing():
    rows = [_pos(10) for _ in range(20)] + [_pos(-10) for _ in range(80)]
    card = score_mod.scorecard(rows)
    assert card["overall"]["profitable"] is False
    assert "not clearing its costs" in card["headline"]


def test_clearing_the_breakeven_hit_rate_and_still_losing_is_not_profitable():
    """The bug the first version of this panel shipped with.

    Breakeven hit rate assumes losers lose exactly the planned risk. This
    agent exits on invalidation, so most trades end nowhere near a level and
    that assumption stops holding -- a book can clear the rate its entries
    needed and still lose money on every one.
    """
    rows = ([_pos(3, breakeven=0.42) for _ in range(92)]
            + [_pos(-30, breakeven=0.42) for _ in range(88)])
    o = score_mod.scorecard(rows)["overall"]
    assert o["hit_rate"] > o["needed"]        # the shape looks fine
    assert o["beats_breakeven"] is True       # ... and says so
    assert o["expectancy_bps"] < 0            # the money says otherwise
    assert o["profitable"] is False           # and the money decides
    assert "not reaching the target" in o["divergence"]


def test_expectancy_carries_its_own_interval():
    """A mean with no interval reads as a fact when it is often a coin toss."""
    noisy = [_pos(200 if i % 2 else -196) for i in range(60)]
    o = score_mod.scorecard(noisy)["overall"]
    assert o["expectancy_bps"] > 0
    assert o["exp_low"] < 0 < o["exp_high"]
    assert o["profitable"] is None            # wide spread, no conclusion

    steady = [_pos(2.0 + (0.1 if i % 2 else -0.1)) for i in range(60)]
    o2 = score_mod.scorecard(steady)["overall"]
    assert o2["profitable"] is True           # same sign, far less spread


def test_a_high_hit_rate_that_does_not_cover_its_breakeven_is_not_an_edge():
    """The whole reason breakeven is carried per trade.

    Seventy percent sounds excellent and is a loss when the trade needed
    ninety. A tool that reports hit rate alone cannot tell these apart.
    """
    rows = ([_pos(1, breakeven=0.9) for _ in range(70)]
            + [_pos(-9, breakeven=0.9) for _ in range(30)])
    card = score_mod.scorecard(rows)
    assert card["overall"]["hit_rate"] == pytest.approx(0.7)
    assert card["overall"]["beats_breakeven"] is False
    assert card["overall"]["profitable"] is False


def test_the_gate_audit_counts_what_was_refused():
    decisions = ([{"action": "stand_aside", "gate": "conviction"}] * 7
                 + [{"action": "stand_aside", "gate": "spread"}] * 3
                 + [{"action": "enter_long", "gate": None}] * 2)
    rows = score_mod.gate_audit(decisions)
    assert rows[0]["gate"] == "conviction" and rows[0]["blocked"] == 7
    assert rows[0]["share_of_all"] == pytest.approx(7 / 12, abs=1e-3)


def test_the_exit_audit_compares_invalidation_against_full_stops():
    rows = ([_pos(-5, exit_reason="invalidated") for _ in range(40)]
            + [_pos(-20, exit_reason="stop") for _ in range(40)])
    out = score_mod.exit_audit(rows)
    assert out["saved_per_trade_bps"] == pytest.approx(15.0)
    assert "saved" in out["note"]


# ----------------------------------------------------------------- tuner

def _book(n, conviction_split=0.5, good=8.0, bad=-8.0):
    """n closed trades where high conviction did better than low."""
    rows = []
    for i in range(n):
        hi = (i % 2 == 0)
        rows.append(_pos(good if hi else bad,
                         conviction=0.8 if hi else 0.3,
                         closed_at=float(i),
                         features={"held_s": 20.0 if hi else 2.0}))
    return rows


def test_no_proposal_without_enough_trades(ledger):
    for r in _book(20):
        pass
    assert tuner_mod.evaluate("min_conviction", _book(20), Knobs()) == []


def test_walk_forward_finds_a_real_split():
    cands = tuner_mod.evaluate("min_conviction", _book(200), Knobs())
    assert cands, "should find a tightening worth looking at"
    best = cands[0]
    assert best.value > Knobs().min_conviction
    # Scored on data the choice never saw.
    assert best.n_test >= tuner_mod.MIN_TEST
    assert best.test_metric > best.base_metric


def test_a_tightening_that_guts_the_book_is_refused():
    """A filter that keeps eight trades is a smaller sample, not an edge."""
    rows = []
    for i in range(200):
        rare = i % 25 == 0
        rows.append(_pos(50.0 if rare else -1.0,
                         conviction=0.8 if rare else 0.3,
                         closed_at=float(i), features={"held_s": 1.0}))
    for c in tuner_mod.evaluate("min_conviction", rows, Knobs()):
        assert c.retain >= tuner_mod.MIN_RETAIN


def test_loosening_is_never_proposed():
    """The book has no outcome for trades it refused to take.

    Proposing a looser threshold from this data would rest the argument on
    rows that do not exist.
    """
    rows = _book(200)
    for c in tuner_mod.evaluate("min_conviction", rows, Knobs()):
        assert c.value >= Knobs().min_conviction
    for c in tuner_mod.evaluate("max_breakeven", rows, Knobs()):
        assert c.value <= Knobs().max_breakeven


def test_exit_knobs_are_not_proposed_from_outcomes():
    """They change what happens DURING a trade; replaying needs the path."""
    for name in ("invalidate_s", "stale_s"):
        assert name not in tuner_mod.EVALUABLE
        assert name in tuner_mod.NOT_EVALUABLE
        assert tuner_mod.evaluate(name, _book(200), Knobs()) == []


def test_a_proposal_changes_nothing_until_it_is_adopted(ledger):
    pid = ledger.propose(param="min_conviction", current=0.45, proposed=0.55,
                         n_fit=50, n_test=40, fit_metric=3.0,
                         test_metric=2.0, base_metric=1.0, rationale="x")
    assert ledger.overrides() == {}
    assert Knobs.from_overrides(ledger.overrides()).min_conviction == 0.45

    ledger.decide_proposal(pid, adopt=True)
    assert ledger.overrides()["min_conviction"] == 0.55
    assert Knobs.from_overrides(ledger.overrides()).min_conviction == 0.55


def test_rejecting_keeps_the_record_and_changes_nothing(ledger):
    pid = ledger.propose(param="min_conviction", current=0.45, proposed=0.55,
                         n_fit=50, n_test=40, fit_metric=3.0,
                         test_metric=2.0, base_metric=1.0, rationale="x")
    row = ledger.decide_proposal(pid, adopt=False)
    assert row["status"] == "rejected"
    assert ledger.overrides() == {}
    assert any(r["status"] == "rejected" for r in ledger.override_history())


def test_a_second_proposal_supersedes_the_first(ledger):
    a = ledger.propose(param="min_conviction", current=0.45, proposed=0.5,
                       n_fit=50, n_test=40, fit_metric=1.0, test_metric=1.0,
                       base_metric=0.0, rationale="a")
    ledger.propose(param="min_conviction", current=0.45, proposed=0.6,
                   n_fit=50, n_test=40, fit_metric=2.0, test_metric=2.0,
                   base_metric=0.0, rationale="b")
    pending = ledger.proposals("pending")
    assert len(pending) == 1 and pending[0]["proposed_val"] == 0.6
    assert ledger.decide_proposal(a, adopt=True) is None


def test_reverting_puts_a_knob_back(ledger):
    pid = ledger.propose(param="min_conviction", current=0.45, proposed=0.6,
                         n_fit=50, n_test=40, fit_metric=1.0,
                         test_metric=1.0, base_metric=0.0, rationale="x")
    ledger.decide_proposal(pid, adopt=True)
    assert ledger.clear_override("min_conviction") is True
    assert Knobs.from_overrides(ledger.overrides()).min_conviction == 0.45


def test_readiness_says_how_far_off_it_is(ledger):
    r = tuner_mod.readiness(ledger)
    assert r["ready"] is False and r["closed"] == 0
    assert "need" in r["note"] or "needs" in r["note"]


# ------------------------------------------------------- the safety rails

AGENT_FILES = ("autopilot.py", "ledger.py", "score.py", "tuner.py",
               "vote.py", "sweep.py")


def _code_only(path: Path) -> str:
    """The file with every comment and string literal removed.

    Scanning raw source for forbidden words flags the paragraph explaining
    why they are forbidden, which trains everyone to delete the paragraph.
    """
    import io
    import tokenize
    out = []
    with open(path, "rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    return " ".join(out)


def test_nothing_in_the_agent_can_place_an_order():
    """The standing constraint, enforced rather than trusted.

    This tool suggests. If a future edit adds order placement, the thing
    being measured has quietly become something else and every number in
    the ledger starts meaning something different.
    """
    banned = ("place_order", "market_order", "limit_order", "submit_order",
              "cancel_order", "exchange", "sign_l1_action", "private_key",
              "wallet.sign", "eth_account")
    root = Path(__file__).resolve().parent.parent / "liqmap"
    for name in AGENT_FILES:
        # Comments and docstrings are stripped first: these files EXPLAIN
        # the rule, and a prose mention of it is not a violation of it.
        code = _code_only(root / name).lower()
        for word in banned:
            assert word not in code, f"{name} calls {word}"


def test_the_tuner_can_only_touch_the_declared_knobs():
    for param in tuner_mod.EVALUABLE:
        assert param in KNOBS, f"{param} is tunable but has no declared range"
    for name, knob in KNOBS.items():
        assert knob.lo < knob.hi and knob.step > 0
        assert knob.lo <= knob.default <= knob.hi


def test_the_tuner_never_writes_a_knob_directly():
    """Adoption goes through the ledger, where it is recorded.

    A tuner that can assign to a Knobs field bypasses the approval it was
    built to require.
    """
    src = (Path(__file__).resolve().parent.parent
           / "liqmap" / "tuner.py").read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "attr", getattr(fn, "id", ""))
            assert name != "decide_proposal", (
                "the tuner must not adopt its own proposals")
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            assert node.attr not in KNOBS, (
                f"tuner assigns to {node.attr} directly")


def test_the_score_module_never_drops_losing_trades():
    """A filter in the scorer is indistinguishable from a better strategy."""
    rows = [_pos(10) for _ in range(50)] + [_pos(-10) for _ in range(50)]
    card = score_mod.scorecard(rows)
    assert card["overall"]["n"] == 100
    assert card["overall"]["wins"] == 50


# ------------------------------------------- the rule: 3-0, paid for, small

def _unanimous(**kw):
    return payload(**kw)


def _split(**kw):
    """Two agreeing, one against — a 2-1."""
    d = payload(**kw)
    d["three_way"]["delta"] = "down"
    d["delta"] = {"score": -0.4}
    return d


def _paid(d, effort):
    d["confirmation"]["participation"] = {"effort": effort}
    return d


def test_only_three_nil_is_taken(ledger):
    p = Autopilot("ETH", "15m", ledger, require_unanimous=True)
    d = p.step(_paid(_split(), 5.0), now=100.0)
    assert d.action == "stand_aside" and d.gate == "not_unanimous"
    assert p.position is None


def test_three_nil_with_volume_is_taken(ledger):
    p = Autopilot("ETH", "15m", ledger, require_unanimous=True,
                  knobs=Knobs(min_effort=2.0, tp_bps=10.0, sl_bps=10.0))
    d = p.step(_paid(_unanimous(), 5.0), now=100.0)
    assert d.action == "enter_long"
    assert p.position.target_bps == 10.0 and p.position.risk_bps == 10.0


def test_a_unanimous_read_nobody_paid_for_is_refused(ledger):
    """Three columns agreeing on a direction nobody is trading through is
    a reading, not a move."""
    p = Autopilot("ETH", "15m", ledger, require_unanimous=True,
                  knobs=Knobs(min_effort=2.0))
    d = p.step(_paid(_unanimous(), 1.2), now=100.0)      # 120% of normal
    assert d.action == "stand_aside" and d.gate == "not_paid_for"
    assert "120%" in d.reason and "200%" in d.reason


def test_two_hundred_percent_is_the_line(ledger):
    p = Autopilot("ETH", "15m", ledger, require_unanimous=True,
                  knobs=Knobs(min_effort=2.0))
    assert p.step(_paid(_unanimous(candle_ts=1.0), 1.99),
                  now=100.0).gate == "not_paid_for"
    p2 = Autopilot("ETH", "15m", ledger, require_unanimous=True,
                   knobs=Knobs(min_effort=2.0))
    assert p2.step(_paid(_unanimous(candle_ts=2.0), 2.0),
                   now=100.0).action == "enter_long"


def test_a_candle_with_no_volume_reading_is_not_treated_as_quiet(ledger):
    """Absent is not the same as zero. Treating it as zero would refuse
    every trade in the first seconds of a bar."""
    p = Autopilot("ETH", "15m", ledger, require_unanimous=True,
                  knobs=Knobs(min_effort=2.0))
    d = _unanimous()
    d["confirmation"].pop("participation", None)
    out = p.step(d, now=100.0)
    assert out.gate == "no_effort_read"
    assert out.gate != "not_paid_for"


def test_ticks_convert_per_market(ledger, tmp_path):
    """Ten ticks on gold and ten ticks on a cheap perp are different
    numbers of basis points."""
    p = Autopilot("ETH", "15m", ledger, unit="ticks",
                  knobs=Knobs(tp_bps=10.0, sl_bps=10.0, min_effort=0.0))
    d = _paid(_unanimous(entry=100.0), 5.0)
    d["tick"] = 0.01                      # 1 tick = 1bps at price 100
    p.step(d, now=100.0)
    assert p.position.target_bps == pytest.approx(10.0)
    assert p.position.target_px == pytest.approx(100.10)

    # Its own book: a second pilot on the same ledger rehydrates the first
    # one's open position and manages that instead.
    other = Ledger(str(tmp_path / "b.db"))
    p2 = Autopilot("ETH", "15m", other, unit="ticks",
                   knobs=Knobs(tp_bps=10.0, sl_bps=10.0, min_effort=0.0))
    d2 = _paid(_unanimous(entry=4000.0, candle_ts=2.0), 5.0)
    d2["tick"] = 0.1                      # 1 tick = 0.25bps at price 4000
    p2.step(d2, now=100.0)
    assert p2.position.target_bps == pytest.approx(2.5)


def test_ticks_without_a_tick_size_fall_back_rather_than_invent_one(ledger):
    p = Autopilot("ETH", "15m", ledger, unit="ticks",
                  knobs=Knobs(tp_bps=10.0, sl_bps=10.0, min_effort=0.0))
    d = _paid(_unanimous(entry=100.0), 5.0)
    d["tick"] = 0.0
    out = p.step(d, now=100.0)
    assert out.action == "enter_long"
    assert p.position.target_bps == 10.0
    assert "read as bps" in out.reason or p.position.target_bps == 10.0


def test_the_refusals_are_still_recorded(ledger):
    """A gate you set is still a gate: the shape it refused has to stay in
    the book, or the ledger cannot tell you later what it cost."""
    p = Autopilot("ETH", "15m", ledger, require_unanimous=True,
                  knobs=Knobs(min_effort=2.0))
    p.step(_paid(_split(candle_ts=1.0), 5.0), now=100.0)
    p.step(_paid(_unanimous(candle_ts=2.0), 0.5), now=200.0)
    gates = {r["gate"] for r in ledger.decisions(action="stand_aside")}
    assert gates == {"not_unanimous", "not_paid_for"}


# -------------------------------------------------- out before the close

def test_it_is_flat_before_the_candle_prints(ledger):
    """Exiting AT the close assumes a fill at the closing print, which is
    not a price anyone gets — and it leaves the position alive into the
    moment the next candle's reading starts forming."""
    p = Autopilot("ETH", "15m", ledger,
                  knobs=Knobs(invalidate_s=999.0, exit_before_s=10.0))
    p.step(payload(candle_ts=9000.0), now=9100.0)
    assert p.step(payload(candle_ts=9000.0), now=9880.0).action == "hold"
    d = p.step(payload(candle_ts=9000.0), now=9891.0)     # 9s before 9900
    assert d.action == "exit" and d.closed["reason"] == "candle_end"


def test_the_target_is_what_normally_takes_it_out(ledger):
    """The candle close is the backstop, not the plan."""
    p = Autopilot("ETH", "15m", ledger,
                  knobs=Knobs(tp_bps=10.0, sl_bps=10.0, invalidate_s=999.0))
    p.step(payload(candle_ts=9000.0, entry=100.0), now=9100.0)
    tgt = p.position.target_px
    d = p.step(payload(candle_ts=9000.0, entry=100.0), now=9200.0,
               high=tgt + 0.01, low=99.99)
    assert d.action == "exit" and d.closed["reason"] == "target"
    assert d.closed["net_bps"] > 0


def test_a_zero_buffer_still_gets_out_at_the_close(ledger):
    p = Autopilot("ETH", "15m", ledger,
                  knobs=Knobs(invalidate_s=999.0, exit_before_s=0.0))
    p.step(payload(candle_ts=9000.0), now=9100.0)
    assert p.step(payload(candle_ts=9000.0), now=9899.0).action == "hold"
    assert p.step(payload(candle_ts=9000.0),
                  now=9901.0).closed["reason"] == "candle_end"


def test_the_export_carries_the_outcome_and_the_conditions(ledger):
    """One row per trade, result first, then what was true when it was
    taken — so sorting by result puts the two populations side by side."""
    pos = ledger.open_position(
        coin="ETH", interval="15m", candle_ts=9000.0, side="long",
        entry=100.0, target_px=100.1, stop_px=99.9, target_bps=10.0,
        risk_bps=10.0, cost_bps=2.0, breakeven=0.6, agreeing=3,
        features={"vote_shape": "3-0", "vote_book": 0.5, "vote_delta": 0.4,
                  "vote_price": 0.6, "effort": 3.2, "unit": "bps"},
        now=9100.0)
    ledger.close_position(pos, exit_px=100.1, reason="target", now=9200.0)
    ledger.open_position(coin="ETH", interval="15m", candle_ts=9900.0,
                         side="short", entry=100.0, target_px=99.9,
                         stop_px=100.1, target_bps=10.0, risk_bps=10.0,
                         now=9950.0)

    rows = ledger.export_rows()
    assert len(rows) == 2
    done = rows[0]
    assert done["result"] == "WIN"
    assert done["net_bps"] == pytest.approx(8.0, abs=0.01)
    assert done["book"] == "up" and done["delta"] == "up"
    assert done["effort_pct"] == pytest.approx(320.0)
    assert done["exit_reason"] == "target"
    assert done["opened_at"].startswith("1970-")
    # The one still running is in the file and marked, not quietly dropped.
    assert rows[1]["result"] == "open" and rows[1]["net_bps"] is None


def test_a_loser_is_labelled_a_loser(ledger):
    pos = ledger.open_position(
        coin="ETH", interval="15m", candle_ts=1.0, side="long", entry=100.0,
        target_px=100.1, stop_px=99.9, target_bps=10.0, risk_bps=10.0,
        cost_bps=2.0, now=10.0)
    ledger.close_position(pos, exit_px=99.9, reason="stop", now=20.0)
    r = ledger.export_rows()[0]
    assert r["result"] == "LOSS" and r["net_bps"] < 0
