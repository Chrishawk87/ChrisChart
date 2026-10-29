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
        s = ImbSim(200, target_points=3.0)
        feed(s, [(100, 0), (199, 0), (200, 0)])
        assert s._t is not None and s._t.side == UP

    def test_it_enters_short_at_the_mirror(self):
        s = ImbSim(200, target_points=3.0)
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
        s = ImbSim(200, target_points=3.0)
        feed(s, [(200, 0), (-200, 0)])
        assert len(s.trades) == 1
        assert s.trades[0].why == "reverse"
        assert s._t is not None and s._t.side == DOWN

    def test_the_bar_close_flattens_it(self):
        # Target pinned: these tests are about the exit, not about
        # whatever the default objective happens to be this week.
        s = ImbSim(200, target_points=3.0)
        feed(s, [(200, 0), (200, 1)])
        s.bar_close(99.0, PX + 1 * POINT)
        assert len(s.trades) == 1
        assert s.trades[0].why == "bar"
        assert s.trades[0].points == pytest.approx(1.0)
        assert s._t is None

    def test_nothing_survives_the_bar(self):
        s = ImbSim(200, target_points=3.0)
        feed(s, [(200, 0)])
        s.bar_close(99.0, PX)
        feed(s, [(0, 0)])
        assert s._t is None

    def test_only_the_target_counts_as_a_win(self):
        s = ImbSim(200, target_points=3.0)
        feed(s, [(200, 0), (200, 2)])
        s.bar_close(9.0, PX + 2 * POINT)
        t = s.trades[0]
        assert t.points == pytest.approx(2.0)
        assert not t.won, "two points is not three"


class TestTheTwoExitRules:
    """They differ on one question, and it changes which level wins."""

    def test_hold_ignores_the_band(self):
        s = ImbSim(200, target_points=3.0, policy=HOLD)
        feed(s, [(200, 0), (50, 0), (50, 1)])
        assert s.trades == [], "it closed on something it should hold"
        assert s._t is not None

    def test_band_exits_on_the_band(self):
        s = ImbSim(200, target_points=3.0, policy=BAND)
        feed(s, [(200, 0), (50, 0)])
        assert len(s.trades) == 1 and s.trades[0].why == "band"

    def test_band_takes_more_trades_than_hold(self):
        import random
        rng = random.Random(3)
        steps = [(rng.randint(-400, 400), rng.randint(-4, 4) * 0.25)
                 for _ in range(4_000)]
        a = ImbSim(200, target_points=3.0, policy=HOLD)
        b = ImbSim(200, target_points=3.0, policy=BAND)
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
        s = ImbSim(200, target_points=3.0, flip=True)
        feed(s, [(200, 0)])
        assert s._t is not None and s._t.side == DOWN

    def test_it_enters_at_the_same_moments(self):
        import random
        rng = random.Random(5)
        steps = [(rng.randint(-400, 400), rng.randint(-8, 8) * 0.25)
                 for _ in range(4_000)]
        a = ImbSim(200, target_points=3.0)
        b = ImbSim(200, target_points=3.0, flip=True)
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
        body = src[src.index("class Sweep"):]
        assert "for s in self.sims.values():" in body
        assert "s.tick(ts, acc, mid, share, gross)" in body

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


class TestOneSidednessAtEntry:
    """Does the fight behind the net separate the winners?

    In order of how badly each would mislead:

      banding on the raw share rather than the side taken. A short
      entered at 0.38 is as one-sided as a long at 0.62; grouping them
      by raw share splits every band in two and compares shorts
      against longs

      the control facing a different filter from the rule. Then it is
      not a control of the same thing

      a flat gradient read as a finding
    """

    def test_the_share_is_recorded_on_every_trade(self):
        s = ImbSim(200, target_points=3.0)
        ts = 0.0
        for acc, px, sh in ((0, 0, 0.5), (200, 0, 0.64)):
            ts += 1.0
            s.tick(ts, acc, PX + px * POINT, sh, 5_000)
        assert s._t is not None
        assert s._t.entry_share == pytest.approx(0.64)
        assert s._t.entry_gross == 5_000

    def test_a_short_is_banded_on_the_side_it_took(self):
        from liqmap.imbsim import _band_of
        assert _band_of(0.62, UP) == _band_of(0.38, DOWN)

    def test_a_contested_entry_lands_in_the_lowest_band(self):
        from liqmap.imbsim import _band_of
        assert _band_of(0.51, UP) == 0.50
        assert _band_of(0.49, DOWN) == 0.50

    def test_the_bands_split_outcomes_and_keep_the_reasons(self):
        from liqmap.imbsim import bands
        made = []
        for sh, won in ((0.51, False), (0.51, False), (0.72, True)):
            s = ImbSim(200, target_points=1.0)
            s.tick(1.0, 0, PX, sh, 1000)
            s.tick(2.0, 400, PX, sh, 1000)
            if won:
                s.tick(3.0, 400, PX + 1 * POINT, sh, 1000)
            else:
                s.bar_close(3.0, PX)
            made.extend(s.trades)
        rows = {r["band"]: r for r in bands(made)}
        assert rows[0.50]["trades"] == 2 and rows[0.50]["cleared"] == 0
        assert rows[0.70]["trades"] == 1 and rows[0.70]["cleared"] == 1
        assert rows[0.50]["why"] == {"bar": 2}

    def test_how_far_the_misses_got_is_kept(self):
        """A failure at nine tenths is not the same animal as one that
        never left the entry."""
        from liqmap.imbsim import bands
        s = ImbSim(200, target_points=1.0)
        s.tick(1.0, 0, PX, 0.51, 1000)
        s.tick(2.0, 400, PX, 0.51, 1000)
        s.tick(3.0, 400, PX + 0.9 * POINT, 0.51, 1000)   # nearly there
        s.bar_close(4.0, PX)
        r = bands(s.trades)[0]
        assert r["median_missed_mfe"] == pytest.approx(0.9)
        assert r["missed_near"] == 1

    def test_the_filter_is_off_by_default(self):
        s = ImbSim(200, target_points=3.0)
        s.tick(1.0, 400, PX, 0.50, 1000)
        assert s._t is not None, "it filtered with no filter set"

    def test_the_filter_refuses_a_contested_entry(self):
        s = ImbSim(200, target_points=3.0, min_share=0.60)
        s.tick(1.0, 400, PX, 0.52, 1000)
        assert s._t is None
        s.tick(2.0, 400, PX, 0.65, 1000)
        assert s._t is not None and s._t.side == UP

    def test_the_short_filter_mirrors(self):
        s = ImbSim(200, target_points=3.0, min_share=0.60)
        s.tick(1.0, -400, PX, 0.48, 1000)
        assert s._t is None
        s.tick(2.0, -400, PX, 0.35, 1000)
        assert s._t is not None and s._t.side == DOWN

    def test_the_control_faces_the_same_filter(self):
        """Read on the rule's side, not the flipped one, or it is not
        a control of the same thing."""
        a = ImbSim(200, target_points=3.0, min_share=0.60)
        b = ImbSim(200, target_points=3.0, min_share=0.60, flip=True)
        for s in (a, b):
            s.tick(1.0, 400, PX, 0.52, 1000)
        assert a._t is None and b._t is None
        for s in (a, b):
            s.tick(2.0, 400, PX, 0.65, 1000)
        assert a._t is not None and b._t is not None
        assert a._t.side == UP and b._t.side == DOWN

    def test_the_sweep_passes_both_sides_through(self):
        sw = Sweep(levels=(200,), policies=(HOLD,), target_points=3.0)
        sw.tick(1.0, 400, PX, 0.66, 9_000)
        t = sw.sims[(200, HOLD, 0.0, False)]._t
        assert t is not None and t.entry_share == pytest.approx(0.66)
        assert t.entry_gross == 9_000


class TestConfirmationInTheSimulation:
    """The sweep has to test the SAME entry the chart takes, or it is
    validating a strategy nobody is running."""

    RAW = 250_000_000

    def _sim(self, confirm=1.0, level=100):
        return ImbSim(level=level, target_points=1.0, policy=BAND,
                      confirm_ticks=confirm)

    def test_the_call_alone_does_not_open_a_trade(self):
        s = self._sim()
        s.tick(0.0, 200, 1000 * self.RAW)
        assert not s.trades and s._t is None, "it entered at the call"

    def test_price_confirming_opens_it_at_the_price_not_the_call(self):
        s = self._sim()
        s.tick(0.0, 200, 1000 * self.RAW)
        s.tick(0.1, 200, 1001 * self.RAW)
        assert s._t is not None
        assert s._t.entry_px == 1001 * self.RAW
        assert s._t.call_px == 1000 * self.RAW
        assert s._t.slip == 1.0

    def test_a_call_that_dies_first_is_skipped_not_lost(self):
        s = self._sim()
        s.tick(0.0, 200, 1000 * self.RAW)
        s.tick(0.1, 10, 1000 * self.RAW)          # imbalance collapses
        assert s._t is None
        assert s.skipped == 1
        assert not s.trades, "a trade nobody took was recorded"

    def test_confirmation_costs_the_ticks_it_waited(self):
        """Target measured from the FILL. Four ticks from the call is
        not a win when the fill was one tick in."""
        s = self._sim()
        s.tick(0.0, 200, 1000 * self.RAW)
        s.tick(0.1, 200, 1001 * self.RAW)         # fill at +1
        s.tick(0.2, 200, 1004 * self.RAW)         # +4 from the CALL
        assert not s.trades, "it paid out at the call's distance"
        s.tick(0.3, 200, 1005 * self.RAW)         # +4 from the FILL
        assert len(s.trades) == 1 and s.trades[0].won

    def test_zero_confirmation_is_the_old_entry(self):
        s = self._sim(confirm=0.0)
        s.tick(0.0, 200, 1000 * self.RAW)
        assert s._t is not None
        assert s._t.entry_px == s._t.call_px
        assert s._t.slip == 0.0

    def test_the_flipped_control_waits_the_same_way(self):
        """Otherwise the control gets a better entry than the rule and
        the gap between them stops meaning anything."""
        a = ImbSim(level=100, target_points=1.0, policy=BAND,
                   confirm_ticks=1.0)
        b = ImbSim(level=100, target_points=1.0, policy=BAND,
                   confirm_ticks=1.0, flip=True)
        for s in (a, b):
            s.tick(0.0, 200, 1000 * self.RAW)
            assert s._t is None, "one of them entered at the call"


class TestTheCandleCapInTheSimulation:
    """The sweep has to cap the same way the chart does, or it is
    validating a strategy nobody is running."""

    RAW = 250_000_000

    def _sim(self, cap=1, confirm=0.0, level=100):
        return ImbSim(level=level, target_points=1.0, policy=BAND,
                      confirm_ticks=confirm, max_calls=cap)

    def _chatter(self, s, n=6):
        """Cross the level, fall back inside, repeat."""
        t = 0.0
        for _ in range(n):
            s.tick(t, 200, 1000 * self.RAW)
            t += 0.01
            s.tick(t, 10, 1000 * self.RAW)
            t += 0.01

    def test_one_push_is_not_twenty_trades(self):
        s = self._sim()
        self._chatter(s)
        assert len(s.trades) <= 1, f"{len(s.trades)} trades from one push"

    def test_uncapped_takes_them_all(self):
        """The control. If this did not chatter, the test above would
        be proving nothing."""
        s = self._sim(cap=0)
        self._chatter(s)
        assert len(s.trades) > 1

    def test_the_cap_resets_with_the_candle(self):
        s = self._sim()
        self._chatter(s, n=3)
        first = len(s.trades)
        s.bar_close(9.0, 1000 * self.RAW)
        self._chatter(s, n=3)
        assert len(s.trades) > first, "the next candle got no call"

    def test_an_unconfirmed_call_still_spends_the_candle(self):
        """Counting fills instead of calls lets a candle keep trying
        after every unconfirmed call, which is a cap in name only."""
        s = self._sim(confirm=1.0)
        s.tick(0.0, 200, 1000 * self.RAW)      # calls, does not fill
        s.tick(0.1, 10, 1000 * self.RAW)       # dies -> skip
        assert s.skipped == 1
        s.tick(0.2, 200, 1000 * self.RAW)      # would call again
        s.tick(0.3, 200, 1001 * self.RAW)      # and would fill
        assert not s.trades, "the candle got a second call"
