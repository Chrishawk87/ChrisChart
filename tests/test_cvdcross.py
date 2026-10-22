"""Book leads, tape confirms -- and the ways that reading goes wrong.

In rough order of how badly each would mislead the chart:

  smoothing the tape by TRADE COUNT instead of elapsed time, which
  makes the second line hug the first during a burst and lag for
  minutes overnight -- the same signal behaving like two different
  ones depending on the hour

  a price STEP counted as a size change, which turns the book reader
  into a price odometer that shadows the thing it is meant to lead

  a single threshold with no hysteresis, which flips the read every
  time the value grazes the line and calls the result a direction

  a cross reported as confirmed before there is enough history to know
  what normal looks like
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from liqmap.cvdcross import (                            # noqa: E402
    BUY, DOWN, FLAT, LCR_MAX, SELL, UP, CVDCross, OrderBookDeltaTracker,
    RollingStats, VPINCalculator, liquidity_consumption_ratio)

T0 = 1_700_000_000.0


class TestTheSecondLineIsTimeBasedNotTradeBased:
    """The whole reason the second line is smoothed on the clock.

    A trade-count EMA has a time constant that swings with activity.
    The same market, sampled at the same wall-clock instants, must
    smooth the same amount whether the prints arrived in a burst or a
    trickle.
    """

    def _decayed(self, step_s: float, span_s: float = 20.0):
        """Put one step into the line, then let the SAME wall-clock
        span elapse, sampled at different densities."""
        r = CVDCross(sample_s=0.0, tau_s=5.0, warm_at=2)
        r.on_book(5800.00, 40, 5800.25, 40, T0)
        r.on_trade(T0, BUY, 1000, 5800.25)
        # Multiplied, not accumulated: adding 0.01 two thousand times
        # drifts by enough to show up against a 1e-6 tolerance, and
        # that drift would be the test's, not the smoothing's.
        n = int(round(span_s / step_s))
        for i in range(1, n + 1):
            r.on_book(5800.00, 40, 5800.25, 40, T0 + i * step_s)
        return r.snapshot().tape_ema

    def test_the_time_constant_is_seconds_not_samples(self):
        """Twenty seconds of smoothing is twenty seconds of smoothing
        whether it arrived in 20 messages or 2,000.

        The exponential composes exactly -- (1-a1)(1-a2)... reduces to
        exp(-sum(dt)/tau) -- so a correct time-based smooth is blind to
        how often it was asked. A per-message or per-trade decay is
        not, and on ES the message rate swings by orders of magnitude
        between the open and the overnight.
        """
        sparse = self._decayed(step_s=1.0)
        dense = self._decayed(step_s=0.01)
        assert sparse == pytest.approx(dense, rel=1e-6), (
            f"same 20s smoothed differently at different message "
            f"rates: {sparse} vs {dense}")

    def test_it_decays_toward_the_line_it_is_following(self):
        """A 1000-lot step, then quiet. After four tau the smooth
        should have closed most of the gap."""
        after = self._decayed(step_s=0.5, span_s=20.0)   # tau=5, so 4 tau
        assert 900.0 < after < 1000.0

    def test_the_smoothing_actually_lags_the_raw_line(self):
        r = CVDCross(sample_s=1.0, tau_s=10.0, warm_at=2)
        r.on_book(5800.00, 40, 5800.25, 40, T0)
        for i in range(1, 20):
            r.on_trade(T0 + i, BUY, 100, 5800.25)
            r.on_book(5800.00, 40, 5800.25, 40, T0 + i)
        s = r.snapshot()
        assert s.tape_ema < s.tape_cvd, "the smooth is not lagging"
        assert s.tape_ema > 0

    def test_a_longer_tau_lags_more(self):
        def ema_after(tau):
            r = CVDCross(sample_s=1.0, tau_s=tau, warm_at=2)
            r.on_book(5800.00, 40, 5800.25, 40, T0)
            for i in range(1, 20):
                r.on_trade(T0 + i, BUY, 100, 5800.25)
                r.on_book(5800.00, 40, 5800.25, 40, T0 + i)
            return r.snapshot().tape_ema
        assert ema_after(30.0) < ema_after(3.0)

    def test_the_smooth_starts_where_the_line_starts(self):
        """At session open both are zero, so easing up from zero is
        the line genuinely starting, not the smooth dragging. The
        first sample seeds at whatever the line reads then."""
        r = CVDCross(sample_s=0.0, tau_s=10.0, warm_at=2)
        r.on_book(5800.00, 40, 5800.25, 40, T0)
        assert r.snapshot().tape_ema == 0.0
        r.on_trade(T0 + 1, BUY, 500, 5800.25)
        s = r.snapshot()
        assert s.tape_cvd == 500.0, "the raw line is the raw line"
        assert 0.0 < s.tape_ema < 500.0, "the smooth eased toward it"

    def test_the_raw_line_is_reported_beside_the_smoothed_one(self):
        """Reporting the smooth as the raw would hide how far the
        second line is lagging -- the one thing needed to judge a
        cross."""
        r = CVDCross(sample_s=0.0, tau_s=30.0, warm_at=2)
        r.on_book(5800.00, 40, 5800.25, 40, T0)
        for i in range(1, 15):
            r.on_trade(T0 + i, BUY, 100, 5800.25)
        s = r.snapshot()
        assert s.tape_cvd == 1400.0
        assert s.tape_ema < s.tape_cvd

    def test_a_degenerate_tau_is_refused(self):
        with pytest.raises(ValueError):
            CVDCross(tau_s=0.0)


class TestTheBookLineMeasuresLiquidityNotPrice:

    def test_a_price_step_counts_nothing(self):
        b = OrderBookDeltaTracker()                 # rebaseline default
        b.update(5800.00, 40, 5800.50, 40)
        b.update(5800.25, 12, 5800.50, 40)          # bid steps up
        assert b.bid_delta == 0.0
        assert b.book_cvd == 0.0

    def test_size_changing_at_a_standing_quote_still_counts(self):
        b = OrderBookDeltaTracker()
        b.update(5800.00, 40, 5800.25, 40)
        b.update(5800.00, 55, 5800.25, 40)
        assert b.bid_delta == 15 and b.book_cvd == 15

    def test_a_drifting_quote_does_not_move_the_reader(self):
        """The regression this mode exists for: a book walking a tick
        at a time with identical size is not liquidity arriving."""
        b = OrderBookDeltaTracker()
        px = 5800.00
        b.update(px, 40, px + 0.25, 40)
        for _ in range(40):
            px += 0.25
            b.update(px, 40, px + 0.25, 40)
        assert b.book_cvd == 0.0

    def test_carry_mode_restores_the_older_behaviour(self):
        b = OrderBookDeltaTracker(price_move="carry")
        px = 5800.00
        b.update(px, 40, px + 0.25, 40)
        for _ in range(20):
            px += 0.25
            b.update(px, 40, px + 0.25, 40)
        assert b.book_cvd == pytest.approx(20 * 80.0)

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError):
            OrderBookDeltaTracker(price_move="whatever")

    def test_resting_bids_are_positive_and_asks_negative(self):
        b = OrderBookDeltaTracker()
        b.update(5800.00, 40, 5800.25, 40)
        b.update(5800.00, 60, 5800.25, 40)
        assert b.book_cvd > 0
        b.update(5800.00, 60, 5800.25, 90)
        assert b.book_cvd < 20


class TestDirectionDoesNotFlapAtTheThreshold:

    def _r(self):
        return CVDCross(enter=0.40, exit=0.15, warm_at=2)

    def test_a_wide_gate_to_turn_on(self):
        r = self._r()
        assert r._direction(0.39, FLAT) == FLAT
        assert r._direction(0.41, FLAT) == UP
        assert r._direction(-0.41, FLAT) == DOWN

    def test_a_narrow_gate_to_let_go(self):
        r = self._r()
        assert r._direction(0.20, UP) == UP, "dropped it too early"
        assert r._direction(0.10, UP) == FLAT

    def test_a_value_grazing_the_line_does_not_flip_the_read(self):
        r = self._r()
        d = FLAT
        for z in (0.41, 0.38, 0.42, 0.37, 0.39, 0.41):
            d = r._direction(z, d)
        assert d == UP, "hysteresis did not hold the read"

    def test_exit_above_enter_is_refused(self):
        with pytest.raises(ValueError):
            CVDCross(enter=0.1, exit=0.5)


class TestConfirmationIsEarned:

    def _driven(self, n=200, **kw):
        kw.setdefault("sample_s", 0.0)
        kw.setdefault("warm_at", 5)
        r = CVDCross(**kw)
        r.on_book(5800.00, 40, 5800.25, 40, T0)
        for i in range(n):
            ts = T0 + i * 0.5
            r.on_book(5800.00, 40 + i, 5800.25, 40, ts)
            r.on_trade(ts, BUY, 10, 5800.25)
        return r

    def test_nothing_is_confirmed_before_it_is_warm(self):
        r = CVDCross(sample_s=0.0, warm_at=50)
        r.on_book(5800.00, 40, 5800.25, 40, T0)
        for i in range(5):
            r.on_trade(T0 + i, BUY, 10, 5800.25)
        s = r.snapshot()
        assert s.warm is False and s.ok is False

    def test_a_fresh_agreement_has_not_held_long_enough(self):
        r = self._driven(dwell_s=1e9)
        assert r.snapshot().ok is False, "dwell was not enforced"

    def test_agreement_that_holds_is_confirmed(self):
        r = self._driven(dwell_s=0.0, max_flips=999)
        s = r.snapshot()
        assert s.agree and s.ok

    def test_too_many_flips_is_chop(self):
        r = CVDCross(sample_s=0.0, warm_at=3, dwell_s=0.0, max_flips=1,
                     enter=0.05, exit=0.01)
        r.on_book(5800.00, 40, 5800.25, 40, T0)
        for i in range(400):
            up = (i // 3) % 2 == 0
            ts = T0 + i * 0.1
            r.on_book(5800.00, 40 + (20 if up else -20), 5800.25, 40, ts)
            r.on_trade(ts, BUY if up else SELL, 40, 5800.25)
        s = r.snapshot()
        assert s.flips > 1
        assert s.chop is True and s.ok is False

    def test_disagreement_is_not_a_side(self):
        r = CVDCross(sample_s=0.0, warm_at=3, dwell_s=0.0)
        r.on_book(5800.00, 40, 5800.25, 40, T0)
        for i in range(60):          # bids stack while the tape sells
            ts = T0 + i * 0.5
            r.on_book(5800.00, 40 + i * 5, 5800.25, 40, ts)
            r.on_trade(ts, SELL, 20, 5800.00)
        s = r.snapshot()
        if s.book_dir != s.tape_dir:
            assert s.side == FLAT and s.ok is False


class TestTheCrossings:

    def _crossing(self):
        r = CVDCross(sample_s=0.0, warm_at=3, dwell_s=0.0)
        r.on_book(5800.00, 40, 5800.25, 40, T0)
        for i in range(300):
            buy = (i // 25) % 2 == 0
            ts = T0 + i * 0.2
            r.on_book(5800.00, 40, 5800.25, 40, ts)
            r.on_trade(ts, BUY if buy else SELL,
                       50 if buy else 100, 5800.25 + i * 0.01)
        return r

    def test_crossings_are_recorded_with_price_and_direction(self):
        cr = self._crossing().crosses()
        assert cr, "the lines never met"
        assert all(c.direction in (UP, DOWN) for c in cr)
        assert all(c.px > 0 for c in cr)

    def test_each_crossing_carries_whether_it_was_confirmed(self):
        cr = self._crossing().crosses()
        assert all(isinstance(c.confirmed, bool) for c in cr)

    def test_a_crossing_is_not_recorded_twice(self):
        ts = [c.ts for c in self._crossing().crosses()]
        assert len(ts) == len(set(ts))

    def test_the_cross_ring_is_bounded(self):
        r = CVDCross(sample_s=0.0, warm_at=3, crosses=5)
        r.on_book(5800.00, 40, 5800.25, 40, T0)
        for i in range(2000):
            buy = (i // 2) % 2 == 0
            ts = T0 + i * 0.05
            r.on_trade(ts, BUY if buy else SELL, 50 if buy else 100, 5800.25)
        assert len(r.crosses()) <= 5


class TestBoundedAndSerialisable:

    def test_the_chart_ring_is_bounded(self):
        r = CVDCross(sample_s=0.0, history=50)
        for i in range(5000):
            r.on_trade(T0 + i * 0.001, BUY, 1, 5800.25)
        assert len(r.points()) == 50

    def test_sampling_thins_the_ring(self):
        r = CVDCross(sample_s=1.0, history=1000)
        for i in range(2000):
            r.on_trade(T0 + i * 0.001, BUY, 1, 5800.25)   # 2 seconds
        assert len(r.points()) <= 4

    def test_a_new_session_zeroes_both_lines(self):
        r = CVDCross(sample_s=0.0)
        r.on_book(5800.00, 40, 5800.25, 40, T0)
        for i in range(50):
            r.on_trade(T0 + i, BUY, 10, 5800.25)
        assert r.tape_cvd > 0
        r.on_trade(T0 + 86_400 * 2, BUY, 1, 5800.25)
        assert r.tape_cvd == 1

    def test_it_serialises(self):
        import json
        r = CVDCross(sample_s=0.0)
        r.on_book(5800.00, 40, 5800.25, 40, T0)
        r.on_trade(T0 + 1, BUY, 10, 5800.25)
        json.dumps(r.as_dict())

    def test_no_reading_holds_a_non_finite_number(self):
        r = CVDCross(sample_s=0.0)
        r.on_book(5800.00, 0, 5800.25, 0, T0)
        r.on_trade(T0 + 1, BUY, 10, 5800.25)
        for k, v in r.snapshot().as_dict().items():
            if isinstance(v, float):
                assert math.isfinite(v), k


class TestInfinityWouldKillTheRead:

    def test_an_empty_level_gives_a_finite_maximum(self):
        v = liquidity_consumption_ratio(10, 0)
        assert v == LCR_MAX and math.isfinite(v)

    def test_a_buy_is_measured_against_the_ask_and_a_sell_the_bid(self):
        r = CVDCross(sample_s=0.0)
        r.on_book(5800.00, 100, 5800.25, 10, T0)
        r.on_trade(T0 + 1, BUY, 20, 5800.25)
        assert r.snapshot().lcr == pytest.approx(2.0)
        r.on_trade(T0 + 2, SELL, 20, 5800.00)
        assert r.snapshot().lcr == pytest.approx(0.2)

    def test_the_statistics_refuse_non_finite_samples(self):
        s = RollingStats(window=10)
        for x in (1.0, 2.0, 3.0):
            s.add(x)
        s.add(float("inf"))
        s.add(float("nan"))
        assert s.n == 3 and math.isfinite(s.mean) and math.isfinite(s.std)


class TestRollingStats:

    def test_it_matches_the_textbook(self):
        import statistics
        xs = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
        s = RollingStats(window=len(xs))
        for x in xs:
            s.add(x)
        assert s.mean == pytest.approx(statistics.fmean(xs))
        assert s.std == pytest.approx(statistics.stdev(xs))

    def test_a_drifting_series_does_not_flatten_the_z_score(self):
        s = RollingStats(window=50)
        for i in range(50):
            s.add(float(i))
        early = s.std
        for i in range(50, 5000):
            s.add(float(i))
        assert s.std == pytest.approx(early, rel=0.05)

    def test_a_degenerate_window_is_refused(self):
        with pytest.raises(ValueError):
            RollingStats(window=1)


class TestVPIN:

    def test_a_trade_bridging_two_buckets_is_split(self):
        v = VPINCalculator(bucket=100)
        v.add(BUY, 60)
        v.add(BUY, 70)
        assert v.completed == 1 and v.filled == pytest.approx(0.30)

    def test_a_trade_larger_than_a_bucket_closes_several(self):
        v = VPINCalculator(bucket=100)
        v.add(BUY, 450)
        assert v.completed == 4 and v.filled == pytest.approx(0.50)

    def test_no_volume_is_lost(self):
        v = VPINCalculator(bucket=100)
        for _ in range(37):
            v.add(BUY, 13)
        assert v.completed * 100 + v.filled * 100 == pytest.approx(37 * 13)

    def test_it_is_bounded_to_the_unit_interval(self):
        import random
        rng = random.Random(3)
        v = VPINCalculator(bucket=50, window=20)
        for _ in range(3000):
            v.add(BUY if rng.random() < 0.5 else SELL, rng.randint(1, 400))
            assert 0.0 <= v.value <= 1.0

    def test_a_bad_bucket_size_is_refused(self):
        with pytest.raises(ValueError):
            VPINCalculator(bucket=0)


class TestItDecidesNothing:

    def test_no_order_code(self):
        src = (ROOT / "liqmap" / "cvdcross.py").read_text()
        for w in ("def submit(", "def execute(", "def place(",
                  "def order("):
            assert w not in src, w

    def test_it_claims_nothing_about_who_is_resting(self):
        src = (ROOT / "liqmap" / "cvdcross.py").read_text().lower()
        for word in ("institution", "retail", "smart money", "whale"):
            assert word not in src, word


class TestThePanel:

    def _fn(self, name):
        import inspect

        from liqmap import web as w
        src = inspect.getsource(w)
        i = src.index("function " + name + "(")
        return src[i:src.index("\nfunction ", i + 10)]

    def test_it_draws_on_one_axis(self):
        """Two y-scales would make any two shapes look related."""
        fn = self._fn("mgChart")
        assert fn.count("const Y = v =>") == 1, "more than one y mapping"

    def test_identity_is_not_carried_by_colour_alone(self):
        fn = self._fn("mgChart")
        assert "stroke-dasharray" in fn
        assert "<text" in fn, "the lines are not labelled at the end"

    def test_the_end_labels_cannot_overprint(self):
        """They converge exactly when a cross is imminent, which is
        when telling the lines apart matters most."""
        fn = self._fn("mgChart")
        assert "Math.abs(yB - yT) < 12" in fn

    def test_confirmed_and_unconfirmed_crosses_are_drawn_differently(self):
        fn = self._fn("mgChart")
        assert "c.confirmed" in fn
        assert "fill=\"none\"" in fn, "no hollow mark for unconfirmed"

    def test_warming_up_is_not_reported_as_a_reading(self):
        fn = self._fn("mgRow")
        assert "n.warm" in fn and "WARMING UP" in fn
