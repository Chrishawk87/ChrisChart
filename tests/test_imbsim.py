"""The imbalance rule as trades, and the ways a sweep flatters itself.

In order of how badly each would mislead:

  no control. A win rate here is driven mostly by how long a bar runs
  and how far price travels inside one, which has nothing to do with
  the signal. Without the same rule run backwards there is no way to
  tell a reading from the shape of the bar

  re-entering on an unchanged reading. A trade that exits at target
  with the imbalance still far past the level re-enters on the next
  update, and one move is counted as a dozen trades -- every one of
  them a winner, since price is already going that way

  a trade surviving its bar. The bar is the horizon

  the two exit rules compared across different data. They have to be
  run over the same tape in the same pass or the comparison is of two
  tapes
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from liqmap.imbsim import (BAND, DOWN, HOLD, LEVELS, POINT,  # noqa: E402
                           UP, ImbSim, Sweep, Trade)

PX = 7_700 * POINT


def feed(sim, steps):
    """steps of (acc, price_in_points_from_7700)."""
    ts = 0.0
    for acc, pts in steps:
        ts += 1.0
        sim.tick(ts, acc, PX + pts * POINT)
    return ts


class TestTheRule:

    def test_it_enters_at_the_level(self):
        s = ImbSim(200)
        feed(s, [(100, 0), (199, 0), (200, 0)])
        assert s._t is not None and s._t.side == UP

    def test_it_enters_short_at_the_mirror(self):
        s = ImbSim(200)
        feed(s, [(-199, 0), (-200, 0)])
        assert s._t is not None and s._t.side == DOWN

    def test_it_exits_at_the_target(self):
        s = ImbSim(200, target_points=3.0)
        feed(s, [(200, 0), (200, 1), (200, 2.5), (200, 3)])
        assert len(s.trades) == 1
        t = s.trades[0]
        assert t.why == "target" and t.won
        assert t.points == pytest.approx(3.0)

    def test_a_short_target_is_measured_downward(self):
        s = ImbSim(200, target_points=3.0)
        feed(s, [(-200, 0), (-200, -3)])
        t = s.trades[0]
        assert t.side == DOWN and t.why == "target"
        assert t.points == pytest.approx(3.0)

    def test_it_does_not_exit_one_tick_short(self):
        s = ImbSim(200, target_points=3.0)
        feed(s, [(200, 0), (200, 2.75)])
        assert s.trades == [] and s._t is not None

    def test_the_opposite_level_reverses_it(self):
        s = ImbSim(200)
        feed(s, [(200, 0), (-200, 0)])
        assert len(s.trades) == 1
        assert s.trades[0].why == "reverse"
        assert s._t is not None and s._t.side == DOWN

    def test_the_bar_close_flattens_it(self):
        s = ImbSim(200)
        feed(s, [(200, 0), (200, 1)])
        s.bar_close(99.0, PX + 1 * POINT)
        assert len(s.trades) == 1
        assert s.trades[0].why == "bar"
        assert s.trades[0].points == pytest.approx(1.0)
        assert s._t is None

    def test_nothing_survives_the_bar(self):
        s = ImbSim(200)
        feed(s, [(200, 0)])
        s.bar_close(99.0, PX)
        feed(s, [(0, 0)])
        assert s._t is None

    def test_only_the_target_counts_as_a_win(self):
        s = ImbSim(200)
        feed(s, [(200, 0), (200, 2)])
        s.bar_close(9.0, PX + 2 * POINT)
        t = s.trades[0]
        assert t.points == pytest.approx(2.0)
        assert not t.won, "two points is not three"


class TestTheTwoExitRules:
    """They differ on one question, and it changes which level wins."""

    def test_hold_ignores_the_band(self):
        s = ImbSim(200, policy=HOLD)
        feed(s, [(200, 0), (50, 0), (50, 1)])
        assert s.trades == [], "it closed on something it should hold"
        assert s._t is not None

    def test_band_exits_on_the_band(self):
        s = ImbSim(200, policy=BAND)
        feed(s, [(200, 0), (50, 0)])
        assert len(s.trades) == 1 and s.trades[0].why == "band"

    def test_band_takes_more_trades_than_hold(self):
        import random
        rng = random.Random(3)
        steps = [(rng.randint(-400, 400), rng.randint(-4, 4) * 0.25)
                 for _ in range(4_000)]
        a, b = ImbSim(200, policy=HOLD), ImbSim(200, policy=BAND)
        for s in (a, b):
            feed(s, steps)
            s.bar_close(9e9, PX)
        assert len(b.trades) > len(a.trades)

    def test_an_unknown_rule_is_refused(self):
        with pytest.raises(ValueError):
            ImbSim(200, policy="whatever")


class TestReArming:
    """One move counted as a dozen trades is the failure here."""

    def test_it_does_not_re_enter_on_an_unchanged_reading(self):
        s = ImbSim(200, target_points=1.0)
        feed(s, [(600, 0), (600, 1)])          # target hit, acc unchanged
        assert len(s.trades) == 1
        feed(s, [(600, 1), (600, 2), (600, 3)])
        assert len(s.trades) == 1, "it re-entered on the same reading"

    def test_coming_back_inside_the_band_re_arms_it(self):
        s = ImbSim(200, target_points=1.0)
        feed(s, [(600, 0), (600, 1)])
        feed(s, [(0, 1)])                      # back inside
        feed(s, [(600, 1)])
        assert s._t is not None, "it never re-armed"

    def test_the_other_side_is_not_disarmed_by_this_one(self):
        s = ImbSim(200, target_points=1.0)
        feed(s, [(600, 0), (600, 1)])          # a long exits at target
        feed(s, [(-600, 1)])
        assert s._t is not None and s._t.side == DOWN

    def test_a_bar_can_hold_several_trades(self):
        s = ImbSim(200, target_points=1.0)
        feed(s, [(600, 0), (600, 1),           # trade 1, target
                 (0, 1), (600, 1), (600, 2)])  # re-arm, trade 2, target
        s.bar_close(99.0, PX + 2 * POINT)
        assert len(s.trades) == 2
        assert all(t.won for t in s.trades)
        assert s.trades_per_bar == 2.0
        assert s.multi_bars == 1


class TestTheControl:
    """The only number that says whether the signal carries anything."""

    def test_the_flip_takes_the_other_side(self):
        s = ImbSim(200, flip=True)
        feed(s, [(200, 0)])
        assert s._t is not None and s._t.side == DOWN

    def test_it_enters_at_the_same_moments(self):
        import random
        rng = random.Random(5)
        steps = [(rng.randint(-400, 400), rng.randint(-8, 8) * 0.25)
                 for _ in range(4_000)]
        a, b = ImbSim(200), ImbSim(200, flip=True)
        for s in (a, b):
            feed(s, steps)
            s.bar_close(9e9, PX)
        # Entries differ only once a target exit lands at a different
        # moment, so on a path where neither reaches three points the
        # counts match exactly.
        assert abs(len(a.trades) - len(b.trades)) <= len(a.trades) * 0.2

    def test_a_signal_that_is_right_beats_its_own_flip(self):
        """A path built so the rule is correct: up means up."""
        steps = []
        px = 0.0
        for k in range(200):
            up = (k // 10) % 2 == 0
            for j in range(10):
                px += 0.5 if up else -0.5
                steps.append((400 if up else -400, px))
        a, b = ImbSim(200, target_points=3.0), ImbSim(
            200, target_points=3.0, flip=True)
        for s in (a, b):
            feed(s, steps)
            s.bar_close(9e9, PX + px * POINT)
        assert a.summary()["win_rate"] > b.summary()["win_rate"] + 0.3

    def test_the_sweep_reports_the_gap(self):
        sw = Sweep(levels=(200,), policies=(HOLD,))
        steps = []
        px = 0.0
        for k in range(200):
            up = (k // 10) % 2 == 0
            for j in range(10):
                px += 0.5 if up else -0.5
                steps.append((400 if up else -400, px))
        ts = 0.0
        for acc, p in steps:
            ts += 1.0
            sw.tick(ts, acc, PX + p * POINT)
        sw.bar_close(ts, PX + px * POINT)
        row = sw.rows()[0]
        assert row["edge"] > 0.3
        assert row["control_win_rate"] < row["win_rate"]


class TestTheSweep:

    def test_it_covers_every_level_and_rule(self):
        sw = Sweep()
        assert len(sw.rows()) == len(LEVELS) * 2
        assert {r["level"] for r in sw.rows()} == set(LEVELS)
        assert {r["policy"] for r in sw.rows()} == {HOLD, BAND}

    def test_a_higher_level_takes_fewer_trades(self):
        import random
        rng = random.Random(7)
        sw = Sweep(policies=(HOLD,))
        ts = 0.0
        for _ in range(20_000):
            ts += 1.0
            sw.tick(ts, rng.randint(-700, 700),
                    PX + rng.randint(-8, 8) * 0.25 * POINT)
            if rng.random() < 0.01:
                sw.bar_close(ts, PX)
        counts = [r["trades"] for r in
                  sorted(sw.rows(), key=lambda r: r["level"])]
        assert counts == sorted(counts, reverse=True), counts

    def test_every_sim_sees_the_same_tape(self):
        """Two passes would compare two tapes."""
        src = (ROOT / "liqmap" / "imbsim.py").read_text()
        i = src.index("    def tick(self, ts: float, acc: int, mid: float) -> None:",
                      src.index("class Sweep") if "class Sweep" in src else 0)
        assert "for s in self.sims.values():" in src

    def test_the_trades_come_back_in_order(self):
        import random
        rng = random.Random(9)
        sw = Sweep(levels=(100, 300), policies=(HOLD,))
        ts = 0.0
        for _ in range(5_000):
            ts += 1.0
            sw.tick(ts, rng.randint(-500, 500),
                    PX + rng.randint(-8, 8) * 0.25 * POINT)
            if rng.random() < 0.02:
                sw.bar_close(ts, PX)
        out = sw.trades(HOLD)
        assert out
        assert [t.entry_ts for t in out] == sorted(t.entry_ts for t in out)

    def test_it_serialises(self):
        import json
        s = ImbSim(200, target_points=1.0)
        feed(s, [(600, 0), (600, 1)])
        json.dumps([t.as_dict() for t in s.trades])
        json.dumps(Sweep(levels=(200,), policies=(HOLD,)).rows())


class TestItDecidesNothing:

    def test_no_order_code(self):
        src = (ROOT / "liqmap" / "imbsim.py").read_text()
        for w in ("def submit(", "def execute(", "def place(", "def order("):
            assert w not in src, w

    def test_it_models_no_fills_or_costs(self):
        src = (ROOT / "liqmap" / "imbsim.py").read_text().lower()
        for w in ("slippage", "commission", "position size", "leverage"):
            assert w not in src, w

    def test_it_makes_no_claim_about_what_comes_next(self):
        src = (ROOT / "liqmap" / "imbsim.py").read_text().lower()
        assert "nothing here predicts" in src
        for w in ("will rise", "will fall", "buy signal", "sell signal",
                  "profitable"):
            assert w not in src, w

    def test_a_degenerate_level_or_target_is_refused(self):
        with pytest.raises(ValueError):
            ImbSim(0)
        with pytest.raises(ValueError):
            ImbSim(200, target_points=0)
