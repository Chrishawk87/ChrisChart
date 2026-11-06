"""The Directional Score: slope and distance from value, both in ticks.

Chris proposed D = (Slope * w1) + ((P - VWAP) * w2). The idea is
sound; the units were not. Slope is price per TIME and (P - VWAP) is
price, so adding them makes the weights carry a silent unit
conversion and whichever term has the larger raw magnitude wins --
decided by seconds-versus-minutes rather than by the market.

Both terms are converted to ticks before weighting, so the weights
are dimensionless and mean what one would want.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from liqmap.dscore import DScore

ROOT = Path(__file__).resolve().parents[1]


def ramp(d: DScore, n: int, step: float, t0: float = 1000.0,
         px0: float = 6800.0, v: float = 10.0) -> DScore:
    t, px = t0, px0
    for _ in range(n):
        t += 1.0
        px += step
        d.on_trade(t, px, v)
    return d


class TestItDoesNotTrade:
    def test_no_order_entry_points(self):
        tree = ast.parse((ROOT / "liqmap" / "dscore.py").read_text())
        names = {n.name for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for banned in ("submit", "execute", "place", "order", "send_order"):
            assert banned not in names, banned

    def test_it_imports_nothing(self):
        tree = ast.parse((ROOT / "liqmap" / "dscore.py").read_text())
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module:
                mods.add(n.module.split(".")[0])
        assert mods <= {"__future__", "collections"}, mods


class TestBothTermsAreInTicks:
    """The fix to the formula. Without it the weights are tuning a
    unit conversion rather than a preference."""

    def test_a_tick_a_second_is_sixty_ticks_a_minute(self):
        d = ramp(DScore(slope_s=60.0), 200, +0.25)
        assert d.slope_ticks_per_min == pytest.approx(60.0, rel=1e-6)

    def test_falling_is_negative(self):
        d = ramp(DScore(slope_s=60.0), 200, -0.25)
        assert d.slope_ticks_per_min == pytest.approx(-60.0, rel=1e-6)

    def test_flat_is_zero(self):
        d = ramp(DScore(slope_s=60.0), 200, 0.0)
        assert d.slope_ticks_per_min == 0.0

    def test_the_vwap_distance_is_in_ticks(self):
        d = DScore()
        d.on_trade(1.0, 6800.0, 100.0)
        d.on_trade(2.0, 6801.0, 100.0)
        # vwap 6800.50, price 6801.00 -> two ticks above
        assert d.vwap == pytest.approx(6800.5)
        assert d.vwap_ticks == pytest.approx(2.0)

    def test_the_weights_are_dimensionless(self):
        """Both terms are ticks, so w=1/1 adds like with like."""
        d = ramp(DScore(slope_s=60.0, w_slope=1.0, w_vwap=1.0), 200, +0.25)
        assert d.score == pytest.approx(d.slope_ticks_per_min
                                        + d.vwap_ticks, rel=1e-9)

    def test_a_weight_of_zero_removes_a_term(self):
        a = ramp(DScore(slope_s=60.0, w_vwap=0.0), 200, +0.25)
        assert a.score == pytest.approx(a.slope_ticks_per_min)
        b = ramp(DScore(slope_s=60.0, w_slope=0.0), 200, +0.25)
        assert b.score == pytest.approx(b.vwap_ticks)

    def test_a_negative_vwap_weight_is_allowed(self):
        """Extended above VWAP is a strong trend AND a stretched
        one. Which of those it is, is empirical -- so the sign of
        w_vwap has to be testable, not assumed positive."""
        d = ramp(DScore(slope_s=60.0, w_slope=1.0, w_vwap=-1.0),
                 200, +0.25)
        assert d.score == pytest.approx(d.slope_ticks_per_min
                                        - d.vwap_ticks, rel=1e-9)


class TestTheSlopeRefusesWhenItCannot:
    def test_two_points_are_not_a_trend(self):
        d = DScore()
        d.on_trade(1.0, 6800.0, 10.0)
        d.on_trade(2.0, 6800.25, 10.0)
        assert d.slope_ticks_per_min == 0.0

    def test_every_print_at_one_instant_is_not_a_slope(self):
        """A degenerate window has an infinite slope. Zero is the
        only honest answer; a huge number would be weighted and
        believed."""
        d = DScore()
        for _ in range(50):
            d.on_trade(1000.0, 6800.0, 10.0)
        assert d.slope_ticks_per_min == 0.0

    def test_an_empty_score_is_zero_not_undefined(self):
        assert DScore().score == 0.0
        assert DScore().direction == "flat"


class TestTheSessionAnchor:
    def test_the_vwap_resets_after_a_gap(self):
        """Carrying yesterday's anchor measures distance from a
        price nobody is trading against any more."""
        d = ramp(DScore(), 100, +0.25)
        before = d.vwap
        d.on_trade(d._last_ts + 10_000.0, 6900.0, 10.0)
        assert d.sessions == 1
        assert d.vwap == pytest.approx(6900.0)
        assert d.vwap != before

    def test_a_normal_pause_does_not_reset_it(self):
        d = ramp(DScore(), 100, +0.25)
        d.on_trade(d._last_ts + 60.0, 6825.0, 10.0)
        assert d.sessions == 0

    def test_the_slope_window_also_clears_on_a_new_session(self):
        d = ramp(DScore(slope_s=60.0), 200, +0.25)
        assert d.slope_ticks_per_min != 0.0
        d.on_trade(d._last_ts + 10_000.0, 6900.0, 10.0)
        assert d.slope_ticks_per_min == 0.0, \
            "a slope across a session gap is not a slope"

    def test_the_anchor_is_the_session_not_the_bar(self):
        """A 5-minute anchor would be near-perfectly correlated with
        the slope over that same bar -- one indicator with extra
        arithmetic. The whole value of the idea is the longer
        anchor."""
        src = (ROOT / "liqmap" / "dscore.py").read_text()
        assert "SESSION_GAP_S" in src
        assert DScore().gap_s >= 3600.0


class TestItIsAFilterNotATrigger:
    def test_it_backs_a_side_above_the_threshold(self):
        d = ramp(DScore(slope_s=60.0), 200, +0.25)
        assert d.agrees("up", 10.0) is True
        assert d.agrees("down", 10.0) is False

    def test_it_refuses_below_the_threshold(self):
        d = ramp(DScore(slope_s=60.0), 200, +0.25)
        assert d.agrees("up", 10_000.0) is False

    def test_it_backs_nothing_when_flat(self):
        d = ramp(DScore(slope_s=60.0), 200, 0.0)
        assert d.agrees("up", 0.0) is True      # zero threshold, zero score
        assert d.agrees("up", 1.0) is False
        assert d.agrees("down", 1.0) is False

    def test_an_unknown_side_is_never_backed(self):
        d = ramp(DScore(slope_s=60.0), 200, +0.25)
        assert d.agrees("", 0.0) is False
        assert d.agrees("sideways", 0.0) is False

    def test_the_docstring_says_veto_only(self):
        """A continuous score crosses any threshold repeatedly
        inside a bar. Using one as a trigger is how the churn came
        back twice."""
        src = (ROOT / "liqmap" / "dscore.py").read_text()
        assert "A FILTER, NOT A TRIGGER" in src
