"""The vote, and the grid that scores it at every target and stop.

Most of these are about the two ways a grid search lies: settling an
ambiguous bar in your own favour, and quietly dropping the trades that
never resolved. Both make the surface look better than the market was.
"""

from __future__ import annotations

import pytest

from liqmap import sweep as sw
from liqmap import vote as vt
from liqmap.autopilot import Autopilot, Knobs
from liqmap.ledger import Ledger


# ------------------------------------------------------------------ vote

def test_all_three_agreeing_is_a_trade():
    v = vt.cast("up", 0.5, "up", 0.4, "up", 0.6)
    assert v.side == "long" and v.agreeing == 3 and v.against == 0
    assert v.shape() == "3-0" and v.unanimous


def test_two_agreeing_against_one_is_a_trade():
    """Explicitly wanted: they do not all have to agree."""
    v = vt.cast("up", 0.5, "up", 0.4, "down", 0.3)
    assert v.side == "long" and v.agreeing == 2 and v.against == 1
    assert v.shape() == "2-1" and v.split


def test_one_column_with_two_quiet_is_still_a_trade():
    v = vt.cast("down", 0.4, "flat", 0.0, "flat", 0.0)
    assert v.side == "short" and v.agreeing == 1 and v.flat == 2
    assert v.shape() == "1-0"


def test_strength_decides_which_side_when_columns_disagree():
    # Two weak against one strong: the sum, not a head count.
    v = vt.cast("up", 0.9, "down", 0.2, "down", 0.2)
    assert v.side == "long" and v.agreeing == 1 and v.against == 2


def test_all_flat_is_no_reading_not_a_refusal():
    v = vt.cast("flat", 0.0, "flat", 0.0, "flat", 0.0)
    assert v.side is None
    assert "nothing to read" in v.describe()


def test_a_dead_even_split_reads_as_nothing():
    v = vt.cast("up", 0.5, "down", 0.5, "flat", 0.0)
    assert v.side is None


def test_nothing_in_the_vote_can_refuse_a_trade():
    """Every non-flat combination produces a side. That is the whole point.

    A conviction floor here would recreate the filter this module exists to
    remove, and the filtered-out trades would again have no outcome to
    argue with.
    """
    dirs = ("up", "down", "flat")
    for b in dirs:
        for d in dirs:
            for p in dirs:
                v = vt.cast(b, 0.5, d, 0.3, p, 0.7)
                allflat = b == d == p == "flat"
                # The only other None is an exact cancellation.
                if v.side is None:
                    assert allflat or abs(v.net) < 1e-12


def test_a_stored_candle_reading_becomes_a_vote():
    """Weeks of collected candles are testable because the RAW readings
    were stored rather than the verdict."""
    row = {"book_dir": "up", "book_strength": 0.6,
           "price_dir": "up", "price_strength": 0.4,
           "features": {"delta_score": -0.2}}
    v = vt.from_state(row)
    assert v.side == "long" and v.against == 1 and v.shape() == "2-1"


def test_a_reading_with_no_delta_still_votes():
    row = {"book_dir": "down", "book_strength": 0.5,
           "price_dir": "down", "price_strength": 0.5, "features": {}}
    v = vt.from_state(row)
    assert v.side == "short" and v.flat == 1 and v.shape() == "2-0"


# ------------------------------------------------------------- settlement

def bars(*ohlc):
    return [sw.Bar(ts=float(i), open=o, high=h, low=l, close=c)
            for i, (o, h, l, c) in enumerate(ohlc)]


def test_a_bar_touching_both_levels_is_a_stop():
    """The rule the whole grid's honesty rests on.

    Inside one bar OHLC does not order the two touches. Taking the good one
    flatters the tightest stops most — which are exactly the cells that
    would then look best.
    """
    b = bars((100, 102, 98, 100))
    reason, px, held = sw.resolve("long", 100.0, 100.0, 100.0, b)
    assert reason == "stop"

    b2 = bars((100, 102, 98, 100))
    assert sw.resolve("short", 100.0, 100.0, 100.0, b2)[0] == "stop"


def test_the_first_bar_to_touch_ends_it():
    b = bars((100, 100.2, 99.9, 100.1),
             (100.1, 100.2, 99.9, 100.0),
             (100, 101.0, 99.95, 100.9))
    reason, px, held = sw.resolve("long", 100.0, 50.0, 200.0, b)
    assert reason == "target" and held == 3


def test_nothing_touched_is_marked_out_not_dropped():
    """Dropping unresolved trades is the other way to manufacture an edge.

    The ones that never reach a level are disproportionately the ones that
    went nowhere, so discarding them removes mostly flat and mildly bad
    outcomes.
    """
    b = bars((100, 100.1, 99.9, 100.05)) * 3
    reason, px, held = sw.resolve("long", 100.0, 500.0, 500.0, b)
    assert reason == "timeout"
    assert px == pytest.approx(100.05)


def test_the_horizon_is_respected():
    quiet = bars((100, 100.01, 99.99, 100.0)) * 50
    late = bars((100, 200.0, 99.99, 199.0))
    reason, px, held = sw.resolve("long", 100.0, 50.0, 50.0,
                                  quiet + late, horizon=10)
    assert reason == "timeout" and held == 10


def test_a_short_settles_the_right_way_round():
    b = bars((100, 100.05, 99.0, 99.2))
    reason, px, _ = sw.resolve("short", 100.0, 50.0, 50.0, b)
    assert reason == "target"
    assert px == pytest.approx(100.0 * (1 - 50 / 10_000.0))


# ------------------------------------------------------------------ grid

def _walk(n=400, drift=0.0004, seed=3):
    import random
    rng = random.Random(seed)
    out, px = [], 100.0
    for i in range(n):
        o = px
        c = o * (1 + rng.gauss(drift, 0.0009))
        out.append(sw.Bar(ts=float(i * 60), open=o, close=c,
                          high=max(o, c) * 1.0002, low=min(o, c) * 0.9998))
        px = c
    return out


def _signals(bars_, every=10, side="long", agreeing=3):
    return [sw.Signal(ts=b.ts, coin="ETH", interval="15m", side=side,
                      entry=b.close, agreeing=agreeing,
                      against=3 - agreeing, shape=f"{agreeing}-0")
            for i, b in enumerate(bars_[:200]) if i % every == 0]


def test_the_grid_scores_every_pair():
    b = _walk()
    out = sw.run(_signals(b), b, [10, 20], [10, 20], cost_bps=1.0)
    assert out["ok"]
    assert len(out["cells"]) == 4
    assert {(c["tp_bps"], c["sl_bps"]) for c in out["cells"]} == {
        (10, 10), (10, 20), (20, 10), (20, 20)}


def test_cost_comes_off_every_trade():
    b = _walk()
    free = sw.run(_signals(b), b, [20], [20], cost_bps=0.0)
    paid = sw.run(_signals(b), b, [20], [20], cost_bps=3.0)
    a = free["cells"][0]["expectancy"]
    c = paid["cells"][0]["expectancy"]
    assert c == pytest.approx(a - 3.0, abs=1e-6)


def test_every_signal_lands_in_every_cell():
    """No cell may quietly score a different sample from its neighbour."""
    b = _walk()
    out = sw.run(_signals(b), b, [5, 15, 30], [5, 15, 30])
    counts = {c["n"] for c in out["cells"]}
    assert len(counts) == 1
    assert out["signals"] == counts.pop()


def test_outcomes_always_add_up():
    b = _walk()
    out = sw.run(_signals(b), b, [10, 25], [10, 25])
    for c in out["cells"]:
        assert c["targets"] + c["stops"] + c["timeouts"] == c["n"]


def test_an_oversized_grid_is_refused_with_the_arithmetic():
    b = _walk()
    out = sw.run(_signals(b), b, sw.axis(1, 100, 1), sw.axis(1, 100, 1))
    assert not out["ok"] and "cells" in out["detail"]


def test_the_vote_filter_slices_rather_than_gates():
    b = _walk()
    mixed = _signals(b, agreeing=3) + _signals(b, every=11, agreeing=2)
    everything = sw.run(mixed, b, [20], [20], min_agreeing=0)
    strict = sw.run(mixed, b, [20], [20], min_agreeing=3)
    assert strict["signals"] < everything["signals"]


def test_a_thin_sample_is_called_out_before_any_number():
    b = _walk()
    out = sw.run(_signals(b, every=60), b, [20], [20])
    assert "too few" in out["verdict"]


def test_the_best_cell_is_not_sold_as_a_finding_when_it_straddles_zero():
    """A grid's maximum is the max of many noisy numbers, so searching alone
    biases it upward. Saying so is the difference between a research tool
    and a way to lose money confidently."""
    b = _walk(drift=0.0)
    out = sw.run(_signals(b, every=2), b, [10, 20, 30], [10, 20, 30])
    best = out["best"]
    if not best["significant"]:
        assert ("straddles zero" in out["verdict"]
                or "bad surface" in out["verdict"])


def test_the_surface_comes_back_shaped_for_drawing():
    b = _walk()
    out = sw.run(_signals(b), b, [10, 20, 30], [5, 15])
    grid = sw.surface(out)
    assert len(grid) == 2 and all(len(r) == 3 for r in grid)


def test_axis_is_inclusive_of_the_end():
    assert sw.axis(5, 20, 5) == [5, 10, 15, 20]
    assert sw.axis(5, 5, 5) == [5]
    assert sw.axis(5, 20, 0) == [5]


# ------------------------------------------- the live agent matches the grid

def payload(book="up", delta="up", price="up", entry=100.0, candle_ts=1.0):
    return {"ok": True, "take": False, "candle_ts": candle_ts,
            "entry": entry, "score": 0.5, "interval_s": 900.0,
            "three_way": {"book": book, "delta": delta, "price": price,
                          "direction": "flat", "agreeing": 0},
            "delta": {"score": 0.4 if delta == "up"
                      else -0.4 if delta == "down" else 0.0},
            "action": {"score": 0.6},
            "confirmation": {"candle": price, "candle_strength": 0.6}}


def test_the_agent_trades_on_the_vote_with_no_gates(tmp_path):
    """A call `suggest` refused outright still trades in raw mode."""
    led = Ledger(str(tmp_path / "t.db"))
    p = Autopilot("ETH", "15m", led, knobs=Knobs(tp_bps=25.0, sl_bps=10.0))
    d = p.step(payload(), now=100.0)
    assert d.action == "enter_long"
    assert p.position.target_px == pytest.approx(100.0 * 1.0025)
    assert p.position.stop_px == pytest.approx(100.0 * 0.9990)


def test_two_of_three_trades_in_raw_mode(tmp_path):
    led = Ledger(str(tmp_path / "t.db"))
    p = Autopilot("ETH", "15m", led)
    d = p.step(payload(delta="down"), now=100.0)
    assert d.action == "enter_long"
    assert p.position.features["vote_shape"] == "2-1"


def test_all_flat_is_the_only_thing_that_stands_aside(tmp_path):
    led = Ledger(str(tmp_path / "t.db"))
    p = Autopilot("ETH", "15m", led)
    d = p.step(payload(book="flat", delta="flat", price="flat"), now=100.0)
    assert d.action == "stand_aside" and d.gate == "no_read"


def test_raw_mode_exits_only_on_levels_so_the_grid_transfers(tmp_path):
    """The live book and the backtest must be the same strategy.

    An invalidation exit cannot be replayed from bar data, so leaving it on
    by default would make every number in the grid describe something the
    agent is not doing.
    """
    led = Ledger(str(tmp_path / "t.db"))
    p = Autopilot("ETH", "15m", led, knobs=Knobs(invalidate_s=1.0))
    assert p.exit_on_invalidation is False
    p.step(payload(), now=100.0)
    for t in (105.0, 110.0, 115.0, 120.0):
        assert p.step(payload(book="down", delta="down", price="down"),
                      now=t).action == "hold"
    assert p.position is not None


def test_invalidation_can_be_switched_back_on(tmp_path):
    led = Ledger(str(tmp_path / "t.db"))
    p = Autopilot("ETH", "15m", led, knobs=Knobs(invalidate_s=4.0),
                  exit_on_invalidation=True)
    p.step(payload(), now=100.0)
    # 5s of elapsed time against a 4s patience: one step is enough here.
    d = p.step(payload(book="down", delta="down", price="down"), now=105.0)
    assert d.action == "exit" and d.closed["reason"] == "invalidated"


def test_the_agent_and_the_grid_settle_a_trade_identically(tmp_path):
    """Same entry, same levels, same bar — the two must not disagree."""
    led = Ledger(str(tmp_path / "t.db"))
    p = Autopilot("ETH", "15m", led, knobs=Knobs(tp_bps=100.0, sl_bps=100.0))
    p.step(payload(), now=0.0)
    # A bar spanning both levels: the agent must call it a stop, as the
    # grid does.
    d = p.step(payload(), now=60.0, high=102.0, low=98.0)
    assert d.action == "exit" and d.closed["reason"] == "stop"

    reason, _, _ = sw.resolve("long", 100.0, 100.0, 100.0,
                              bars((100, 102, 98, 100)))
    assert reason == d.closed["reason"]


# ------------------------------------------- levels, walked forward

def test_a_level_proposal_needs_signals_on_both_sides_of_the_split(tmp_path):
    from liqmap import tuner
    from liqmap.autopilot import Knobs as K
    led = Ledger(str(tmp_path / "t.db"))
    b = _walk()
    assert tuner.propose_levels(led, K(), _signals(b, every=40), b) is None
    assert led.proposals() == []


def test_a_level_proposal_moves_the_pair_together(tmp_path):
    """Adopting a target without the stop it was measured beside produces a
    setting nobody tested."""
    from liqmap import tuner
    from liqmap.autopilot import Knobs as K
    led = Ledger(str(tmp_path / "t.db"))
    led.propose(param="tp_bps", current=20, proposed=45, n_fit=80, n_test=50,
                fit_metric=8.0, test_metric=7.0, base_metric=2.0,
                rationale="x", param2="sl_bps", current2=15, proposed2=30)
    pid = led.proposals()[0]["id"]
    led.decide_proposal(pid, adopt=True)
    o = led.overrides()
    assert o["tp_bps"] == 45 and o["sl_bps"] == 30


def test_adopted_levels_reach_the_agent(tmp_path):
    from liqmap.autopilot import Knobs as K
    led = Ledger(str(tmp_path / "t.db"))
    pid = led.propose(param="tp_bps", current=20, proposed=45, n_fit=80,
                      n_test=50, fit_metric=8.0, test_metric=7.0,
                      base_metric=2.0, rationale="x", param2="sl_bps",
                      current2=15, proposed2=30)
    led.decide_proposal(pid, adopt=True)
    k = K.from_overrides(led.overrides())
    assert k.tp_bps == 45 and k.sl_bps == 30


def test_a_level_proposal_is_refused_when_it_straddles_zero(tmp_path):
    """Good on the fit half is not enough — the out-of-sample interval has
    to clear zero, or the pair is a coin toss with a nice backstory."""
    from liqmap import tuner
    from liqmap.autopilot import Knobs as K
    led = Ledger(str(tmp_path / "t.db"))
    b = _walk(drift=0.0, seed=17)
    sigs = _signals(b, every=1)
    out = tuner.propose_levels(led, K(), sigs, b)
    if out is not None:
        row = led.proposals()[0]
        assert "entirely above zero" in row["rationale"]


def test_level_readiness_counts_signals_not_trades():
    from liqmap import tuner
    r = tuner.level_readiness(10)
    assert r["ready"] is False and "10 signals" in r["note"]
    assert tuner.level_readiness(500)["ready"] is True
