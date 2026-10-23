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
    BookCVDBars, BookCVDFrames, Episode,
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
        """The second line is PRICE now, so price has to move or the
        two can never meet. A fixture holding price flat tests
        nothing."""
        r = CVDCross(sample_s=0.0, warm_at=3, dwell_s=0.0)
        r.on_book(5800.00, 40, 5800.25, 40, T0)
        px = 5800.25
        for i in range(400):
            buy = (i // 25) % 2 == 0
            ts = T0 + i * 0.2
            px += 0.25 if buy else -0.25
            r.on_book(5800.00, 40 + (20 if buy else -20),
                      5800.25, 40, ts)
            r.on_trade(ts, BUY if buy else SELL, 50, px)
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

    def test_both_lines_share_one_exact_mapping(self):
        """One Y. With two scales the distance between the lines, and
        every crossing, would be a rendering choice rather than a
        quantity."""
        fn = self._fn("mgChart")
        assert fn.count("const Y = v =>") == 1, "more than one y mapping"
        assert "line('book')" in fn and "line('price')" in fn

    def test_the_axis_turns_sigma_back_into_dollars(self):
        """Nobody looks at -0.32 and sees 7768.50. price_z is
        (px - mean) / sd, so a sigma level inverts to mean + v*sd."""
        fn = self._fn("mgChart")
        assert "mean + v * sd" in fn
        assert "price_mean" in fn and "price_sd" in fn

    def test_the_engine_publishes_the_window_the_axis_needs(self):
        r = CVDCross(sample_s=0.0, warm_at=2)
        for k in range(1, 60):
            r.on_book(5800.00 + k * 0.25, 40 + k, 5800.25 + k * 0.25,
                      40, T0 + k)
            r.on_trade(T0 + k, BUY, 2, 5800.00 + k * 0.25)
        d = r.snapshot().as_dict()
        assert d["price_sd"] > 0 and d["price_mean"] > 0
        # the inversion has to actually land on the price
        back = d["price_mean"] + d["price_z"] * d["price_sd"]
        assert back == pytest.approx(d["px"], rel=1e-3)

    def test_identity_is_not_carried_by_colour_alone(self):
        fn = self._fn("mgChart")
        assert "stroke-dasharray" in fn
        assert "<text" in fn, "the lines are not labelled"

    def test_the_last_price_is_shown_as_a_number(self):
        """Reading a level off an axis is not the same as being told
        where price is."""
        fn = self._fn("mgChart")
        assert "last.px.toFixed(2)" in fn

    def test_confirmed_and_unconfirmed_crosses_are_drawn_differently(self):
        """Crossings mean something again now the two share a scale."""
        fn = self._fn("mgChart")
        assert "c.confirmed" in fn
        assert 'fill="none"' in fn, "no hollow mark for unconfirmed"

    def test_warming_up_is_not_reported_as_a_reading(self):
        fn = self._fn("mgRow")
        assert "n.warm" in fn and "WARMING UP" in fn


class TestTheLinesAreNotHostageToTheFootprint:
    """The bug this class exists for: nothing was drawn at all.

    The reader used to be rendered at the tail of the footprint's
    loader, and that loader returns early whenever the footprint
    reports no data -- which it does until the first print of the
    session lands. So the panel stayed completely empty through
    exactly the quiet stretches the book line is there to read.

    The book line needs NO trades. It must not inherit the
    footprint's silence.
    """

    def test_the_book_line_moves_with_no_trades_at_all(self):
        r = CVDCross(sample_s=0.25, warm_at=10)
        ts, bsz = T0, 40.0
        for i in range(2000):
            ts += 0.05
            bsz = max(1.0, bsz + (3 if i % 7 else -5))
            r.on_book(5799.75, bsz, 5800.25, 40.0, ts)
        d = r.as_dict()
        assert r.trades == 0
        assert len(d["points"]) > 100, "no points without prints"
        assert d["now"]["book_z"] != 0.0, "the reader never moved"

    def test_the_engine_is_attached_where_the_tape_is_built(self):
        """Not inside the footprint's setup -- it has to be counting
        from the moment records arrive."""
        import inspect

        from liqmap import ofi_service
        src = inspect.getsource(ofi_service)
        i = src.index("self.tape = Tape(symbol=tape_sym)")
        assert "cvdcross" in src[i:i + 700], (
            "the reader is not attached at tape construction")

    def test_the_route_does_not_read_the_footprint(self):
        import inspect

        from liqmap import web as w
        src = inspect.getsource(w)
        i = src.index("def api_cvdcross(")
        fn = src[i:src.index("\n    @app.", i)]
        assert "rt.footprint(" not in fn
        assert "stream" not in fn

    def test_the_panel_has_its_own_loader(self):
        import inspect

        from liqmap import web as w
        src = inspect.getsource(w)
        assert "async function mgLoad()" in src
        fp = src[src.index("async function fpLoad()"):]
        fp = fp[:fp.index("\nfunction fpTick(")]
        assert "mgChart" not in fp, (
            "the reader is still drawn inside the footprint's loader, "
            "so its early return still blanks it")

    def test_both_panels_are_actually_scheduled(self):
        """The freeze that looked like lag.

        The /ws/live migration removed setInterval(fpTick) -- its own
        comment says it replaces it -- but footprint and CVD data
        never moved onto those frames. Anything hanging off fpTick
        therefore painted once at page open and then stood still. A
        frozen panel does not look frozen; it looks like a feed
        running far behind the market.
        """
        import inspect

        from liqmap import web as w
        src = inspect.getsource(w)
        assert "setInterval(mgLoad," in src, "the reader has no clock"
        assert "setInterval(fpTick," in src, (
            "fpTick is defined but never scheduled, so the footprint "
            "and obstacle panels only update on Refresh")
        assert "try { mgStartTimers(); } catch (e) {}" in src

    def test_the_reader_does_not_hang_off_the_footprints_tick(self):
        """Its own timer, because fpTick skips hidden tabs and the
        reader should keep its line continuous regardless."""
        import inspect

        from liqmap import web as w
        src = inspect.getsource(w)
        tick = src[src.index("function fpTick()"):]
        tick = tick[:tick.index("\n}")]
        assert "mgLoad" not in tick


class TestTheIncrementalPoll:
    """Re-sending the whole history twice a second is tens of
    kilobytes the browser already holds, and that backlog is itself a
    source of lag."""

    def _filled(self, n=400):
        r = CVDCross(sample_s=0.25, warm_at=5)
        ts = T0
        for i in range(n):
            ts += 0.25
            r.on_book(5799.75, 40 + i % 7, 5800.25, 40, ts)
        return r

    def test_since_returns_only_what_is_newer(self):
        r = self._filled()
        full = r.as_dict()
        cut = full["points"][-3]["ts"]
        inc = r.as_dict(since=cut)
        assert len(inc["points"]) == 2
        assert all(p["ts"] > cut for p in inc["points"])

    def test_a_fresh_caller_gets_everything(self):
        r = self._filled()
        assert len(r.as_dict()["points"]) == 400

    def test_the_reading_always_comes_whole(self):
        """Only the POINTS are incremental. The current reading is
        small and must never be partial."""
        r = self._filled()
        inc = r.as_dict(since=r.as_dict()["points"][-1]["ts"])
        assert inc["points"] == []
        # Not book_cvd: this fixture walks bid size through a 7-step
        # cycle, and 400 updates is a whole number of cycles, so the
        # deltas telescope to exactly zero. The count cannot.
        assert inc["now"]["books"] == 400
        assert set(inc["now"]) == set(r.as_dict()["now"])

    def test_a_caller_that_fell_out_of_the_ring_is_told(self):
        """Splicing two disjoint stretches would draw a move that
        never happened."""
        r = self._filled()
        stale = r.as_dict(since=T0 - 10_000)
        assert stale["continuous"] is False
        assert len(stale["points"]) == 400, "it must resend the lot"

    def test_a_caller_inside_the_ring_is_continuous(self):
        r = self._filled()
        mid = r.as_dict()["points"][-5]["ts"]
        assert r.as_dict(since=mid)["continuous"] is True

    def test_the_client_starts_clean_on_a_discontinuity(self):
        import inspect

        from liqmap import web as w
        src = inspect.getsource(w)
        fn = src[src.index("async function mgLoad()"):]
        fn = fn[:fn.index("\n}")]
        assert "d.continuous === false" in fn
        assert "mgLast" in fn, "the client never sends a since cursor"


class TestEverySignalIsKept:
    """"No matter how many come through."

    Suppressing the crosses that look poor is how a reader flatters
    itself: the unconfirmed ones and the losers are the only evidence
    that the confirmation is worth anything.
    """

    def _fired(self, n=400, **kw):
        kw.setdefault("sample_s", 0.0)
        kw.setdefault("warm_at", 3)
        kw.setdefault("dwell_s", 0.0)
        r = CVDCross(**kw)
        r.on_book(5800.00, 40, 5800.25, 40, T0)
        px = 5800.25
        for i in range(n):
            buy = (i // 25) % 2 == 0
            ts = T0 + i * 0.2
            px += 0.25 if buy else -0.25
            # the book must move too, or it is a flat line that price
            # crosses once and never again
            r.on_book(5800.00, 40 + (20 if buy else -20),
                      5800.25, 40, ts)
            r.on_trade(ts, BUY if buy else SELL, 50, px)
        return r

    def test_every_cross_raises_a_signal(self):
        """Both are bounded rings, and the signal ring is the larger
        of the two, so once crossings exceed the smaller cap the
        counts stop matching. Compare inside the caps."""
        r = self._fired(n=120)
        assert r.crosses(), "no crossings at all"
        assert len(r.signals()) == len(r.crosses())

    def test_unconfirmed_crosses_are_kept_too(self):
        r = self._fired(dwell_s=1e9)        # nothing can confirm
        sigs = r.signals()
        assert sigs, "no signals at all"
        assert all(not g.confirmed for g in sigs)
        assert r.scoreboard()["unconfirmed"]["n"] == len(sigs)

    def test_a_long_tracks_price_upward_and_a_short_downward(self):
        r = self._fired(n=160)
        for g in r.signals():
            assert g.mfe >= 0.0, "best-case excursion went negative"
            assert g.mae <= 0.0, "worst-case excursion went positive"

    def test_excursions_are_measured_in_ticks(self):
        """Ticks, not price units. One dollar of ES is four ticks and
        reporting 1.0 where 4.0 belongs understates every excursion
        fourfold."""
        r = self._fired(n=120)
        sigs = r.signals()
        assert sigs
        g = sigs[-1]
        # push price a clear dollar beyond anything seen so far, at a
        # timestamp AFTER the fixture ended
        end = T0 + 120 * 0.2 + 5
        far = (g.hi + 1.0) if g.side == "long" else (g.lo - 1.0)
        r.on_trade(end, BUY, 1, far)
        span = ((g.hi - g.px) if g.side == "long"
                else (g.px - g.lo))
        assert g.mfe == pytest.approx(span / 0.25)
        assert g.mfe >= 4.0, "a dollar of ES is four ticks"

    def test_a_signal_settles_at_the_horizon(self):
        r = self._fired(horizon_s=1.0)
        assert any(g.settled for g in r.signals())

    def test_the_signal_ring_is_bounded(self):
        r = self._fired(n=4000, signals=20)
        assert len(r.signals()) <= 20

    def test_a_new_session_clears_them(self):
        r = self._fired()
        assert r.signals()
        r.on_trade(T0 + 86_400 * 2, BUY, 1, 5800.25)
        assert r.signals() == []

    def test_they_serialise(self):
        import json
        json.dumps(self._fired().as_dict(with_signals=True))


class TestTheScoreboardIsHonest:

    def _r(self):
        r = CVDCross(sample_s=0.0, warm_at=3, dwell_s=0.0, tick_size=0.25)
        r.on_book(5800.00, 40, 5800.25, 40, T0)
        px = 5800.25
        for i in range(300):
            buy = (i // 20) % 2 == 0
            px += 0.25 if buy else -0.25
            r.on_trade(T0 + i * 0.2, BUY if buy else SELL,
                       50 if buy else 100, px)
        return r

    def test_the_break_even_is_printed_beside_the_rate(self):
        """A hit rate without its break-even is not a result."""
        sb = self._r().scoreboard(target=4.0, stop=4.0)
        assert sb["break_even"] == pytest.approx(0.5)

    def test_an_asymmetric_target_moves_the_break_even(self):
        sb = self._r().scoreboard(target=16.0, stop=4.0)
        assert sb["break_even"] == pytest.approx(0.2)

    def test_confirmed_and_unconfirmed_are_tallied_separately(self):
        sb = self._r().scoreboard()
        assert (sb["confirmed"]["n"] + sb["unconfirmed"]["n"]
                == sb["all"]["n"])

    def test_an_unfinished_signal_counts_as_neither(self):
        sb = self._r().scoreboard(target=1e9, stop=1e9)
        for k in ("all", "confirmed", "unconfirmed"):
            assert sb[k]["hit"] == 0 and sb[k]["miss"] == 0
            assert sb[k]["open"] == sb[k]["n"]

    def test_an_empty_scoreboard_says_none_rather_than_zero(self):
        """A rate of 0.0 reads as 'it never wins'; None reads as
        'nothing has finished', which is the truth."""
        r = CVDCross(sample_s=0.0)
        sb = r.scoreboard()
        assert sb["all"]["rate"] is None and sb["all"]["n"] == 0

    def test_settled_is_reported_apart_from_the_total(self):
        sb = self._r().scoreboard()
        assert "settled" in sb["all"]
        assert sb["all"]["settled"] <= sb["all"]["n"]


class TestTheAgentRoute:

    def test_it_states_that_it_is_not_a_backtest(self):
        import inspect

        from liqmap import web as w
        src = inspect.getsource(w)
        i = src.index("def api_cvd_signals(")
        fn = src[i:src.index("\n    @app.", i)]
        assert "not a backtest" in fn
        assert "nothing is filtered" in fn

    def test_it_places_nothing(self):
        import inspect

        from liqmap import web as w
        src = inspect.getsource(w)
        i = src.index("def api_cvd_signals(")
        fn = src[i:src.index("\n    @app.", i)]
        for w_ in ("submit", "place_order", "execute("):
            assert w_ not in fn, w_

    def test_the_light_poll_does_not_carry_the_signal_book(self):
        """The panel polls twice a second; the agent asks explicitly."""
        r = CVDCross(sample_s=0.0)
        assert "signals" not in r.as_dict()
        assert "signals" in r.as_dict(with_signals=True)


class TestTheBookCVDBars:
    """Book CVD zeroed at every bar, as OHLC.

    The candle is the PATH, not the result. A bar closing at +40 after
    reaching +180 is a different bar from one that climbed straight to
    +40 -- bids stacked and pulled versus bids stacked and kept -- and
    only the wick separates them.
    """

    def test_each_bar_opens_at_zero(self):
        b = BookCVDBars(seconds=60.0)
        b.add(T0 + 1, 50)
        b.add(T0 + 61, 10)                  # rolls
        assert list(b.closed)[-1]["o"] == 0.0
        assert b.live(T0 + 61)["o"] == 0.0

    def test_the_wick_records_what_was_given_back(self):
        b = BookCVDBars(seconds=60.0)
        for d in (60, 80, 40, -140):        # up to +180, back to +40
            b.add(T0 + 1, d)
        b.add(T0 + 61, 0)
        bar = list(b.closed)[-1]
        assert bar["h"] == 180.0 and bar["c"] == 40.0

    def test_a_straight_climb_has_no_upper_wick(self):
        b = BookCVDBars(seconds=60.0)
        for d in (10, 10, 10, 10):
            b.add(T0 + 1, d)
        b.add(T0 + 61, 0)
        bar = list(b.closed)[-1]
        assert bar["h"] == bar["c"] == 40.0

    def test_the_low_records_the_other_side(self):
        b = BookCVDBars(seconds=60.0)
        for d in (-90, 120):
            b.add(T0 + 1, d)
        b.add(T0 + 61, 0)
        bar = list(b.closed)[-1]
        assert bar["l"] == -90.0 and bar["c"] == 30.0

    def test_boundaries_are_absolute_not_relative_to_startup(self):
        b = BookCVDBars(seconds=300.0)
        b.add(1_700_000_123.0, 10)
        assert b._open_ts % 300.0 == 0.0

    def test_a_silent_bar_still_closes(self):
        """Driven by time, not by activity -- otherwise a quiet bar
        merges into the next one."""
        b = BookCVDBars(seconds=60.0)
        b.add(T0 + 1, 50)
        assert b.roll(T0 + 500) is True
        assert b.bars == 1 and b.cur == 0.0

    def test_it_zeroes_rather_than_carrying_forward(self):
        b = BookCVDBars(seconds=60.0)
        b.add(T0 + 1, 100)
        b.add(T0 + 61, 5)
        assert b.cur == 5.0, "the previous bar's total carried over"

    def test_the_history_is_bounded(self):
        b = BookCVDBars(seconds=1.0, keep=5)
        for k in range(200):
            b.add(T0 + k * 2, 3)
        assert len(b.closed) == 5

    def test_the_series_ends_with_the_live_bar(self):
        b = BookCVDBars(seconds=60.0)
        b.add(T0 + 1, 10)
        b.add(T0 + 61, 7)
        ser = b.series(ts=T0 + 70)
        assert ser[-1]["live"] is True and ser[-1]["c"] == 7.0

    def test_a_degenerate_bar_is_refused(self):
        with pytest.raises(ValueError):
            BookCVDBars(seconds=0.0)


class TestAllThreeTimeframesAtOnce:
    """Switching the selector must not lose history, which rebuilding
    one series from scratch would."""

    def test_every_frame_is_fed(self):
        f = BookCVDFrames()
        for k in range(1, 2000):
            f.add(T0 + k, 1)
        assert all(v.bars > 0 for v in f.frames.values())

    def test_a_longer_frame_holds_fewer_bars(self):
        f = BookCVDFrames()
        for k in range(1, 2000):
            f.add(T0 + k, 1)
        assert f.get(60).bars > f.get(300).bars > f.get(900).bars

    def test_an_unknown_choice_falls_back_rather_than_raising(self):
        f = BookCVDFrames()
        assert f.get(77).seconds == 300.0

    def test_the_choices_are_published(self):
        f = BookCVDFrames()
        assert f.as_dict()["choices"] == [60, 300, 900]

    def test_the_engine_feeds_them_from_book_updates_alone(self):
        """No trades needed: this is the BOOK's CVD."""
        r = CVDCross(sample_s=0.0, bar_seconds=60.0)
        for k in range(1, 300):
            r.on_book(5800.00, 40 + (k % 9), 5800.25, 40, T0 + k)
        assert r.trades == 0
        assert r.bars.get(60).bars > 0
        assert r.snapshot().bar_s == 60.0

    def test_a_new_session_clears_them(self):
        r = CVDCross(sample_s=0.0)
        for k in range(1, 200):
            r.on_book(5800.00, 40 + (k % 9), 5800.25, 40, T0 + k)
        r.on_trade(T0 + 86_400 * 2, BUY, 1, 5800.25)
        assert r.bars.get(60).bars == 0

    def test_the_light_poll_carries_the_selected_frame(self):
        r = CVDCross(sample_s=0.0, bar_seconds=900.0)
        for k in range(1, 300):
            r.on_book(5800.00, 40 + (k % 9), 5800.25, 40, T0 + k)
        d = r.as_dict()
        assert d["bars"]["seconds"] == 900.0
        assert d["bars"]["choices"] == [60, 300, 900]


class TestTheRegimeIsAStateNotAnEvent:
    """What he actually described: the two agreeing, AND which line is
    on top. Judged by what price does WHILE it holds."""

    def _r(self, **kw):
        kw.setdefault("sample_s", 0.0)
        kw.setdefault("warm_at", 3)
        kw.setdefault("dwell_s", 0.0)
        kw.setdefault("tick_size", 0.25)
        r = CVDCross(**kw)
        r.on_book(5800.00, 40, 5800.25, 40, T0)
        return r

    def test_the_four_states_are_named_by_side_and_order(self):
        r = self._r()
        px = 5800.25
        for i in range(300):
            buy = (i // 30) % 2 == 0
            px += 0.25 if buy else -0.25
            ts = T0 + i * 0.2
            r.on_book(5800.00, 40 + (15 if buy else -15), 5800.25, 40, ts)
            r.on_trade(ts, BUY if buy else SELL, 50, px)
        seen = {e.regime for e in r.episodes()}
        assert seen, "no regimes at all"
        assert seen <= {"up_book_above", "up_book_below",
                        "down_book_above", "down_book_below"}

    def test_order_follows_which_z_is_higher(self):
        r = self._r()
        for i in range(80):
            ts = T0 + i * 0.25
            r.on_book(5800.00, 40 + i, 5800.25, 40, ts)
            r.on_trade(ts, BUY, 10, 5800.25)
        s = r.snapshot()
        expect = "book_above" if s.book_z > s.tape_z else "book_below"
        assert s.order == expect
        if s.regime != "none":
            assert s.regime.endswith(expect)

    def test_no_agreement_is_not_a_regime(self):
        r = self._r()
        for i in range(60):          # bids stack while the tape sells
            ts = T0 + i * 0.5
            r.on_book(5800.00, 40 + i * 5, 5800.25, 40, ts)
            r.on_trade(ts, SELL, 20, 5800.00)
        s = r.snapshot()
        if s.side == FLAT:
            assert s.regime == "none"

    def test_excursions_are_signed_by_the_regimes_own_direction(self):
        r = self._r()
        px = 5800.25
        for i in range(400):
            buy = (i // 40) % 2 == 0
            px += 0.25 if buy else -0.25
            ts = T0 + i * 0.2
            r.on_book(5800.00, 40 + (15 if buy else -15), 5800.25, 40, ts)
            r.on_trade(ts, BUY if buy else SELL, 50, px)
        for e in r.episodes():
            assert e.favour >= 0.0, f"{e.regime} favour went negative"
            assert e.adverse <= 0.0, f"{e.regime} adverse went positive"

    def test_a_regime_change_closes_the_stretch(self):
        r = self._r()
        px = 5800.25
        for i in range(300):
            buy = (i // 30) % 2 == 0
            px += 0.25 if buy else -0.25
            ts = T0 + i * 0.2
            r.on_book(5800.00, 40 + (15 if buy else -15), 5800.25, 40, ts)
            r.on_trade(ts, BUY if buy else SELL, 50, px)
        eps = r.episodes()
        assert len(eps) >= 2
        assert all(not e.open for e in eps[:-1]), "a stale stretch stayed open"

    def test_the_episode_ring_is_bounded(self):
        r = self._r(episodes=10)
        px = 5800.25
        for i in range(2000):
            buy = (i // 3) % 2 == 0
            px += 0.25 if buy else -0.25
            ts = T0 + i * 0.1
            r.on_book(5800.00, 40 + (15 if buy else -15), 5800.25, 40, ts)
            r.on_trade(ts, BUY if buy else SELL, 50, px)
        assert len(r.episodes()) <= 10

    def test_a_new_session_clears_them(self):
        r = self._r()
        for i in range(80):
            ts = T0 + i * 0.25
            r.on_book(5800.00, 40 + i, 5800.25, 40, ts)
            r.on_trade(ts, BUY, 10, 5800.25)
        r.on_trade(T0 + 86_400 * 2, BUY, 1, 5800.25)
        assert r.episodes() == []


class TestTheRegimeBoardAnswersTheQuestion:
    """'When book CVD is above traded CVD the move tends to be
    stronger' -- stated as a number, with the sample size beside it."""

    def _board(self):
        r = CVDCross(sample_s=0.0, warm_at=3, dwell_s=0.0, tick_size=0.25)
        r.on_book(5800.00, 40, 5800.25, 40, T0)
        px = 5800.25
        for i in range(1200):
            buy = (i // 25) % 2 == 0
            px += 0.25 if buy else -0.25
            ts = T0 + i * 0.3
            r.on_book(5800.00, 40 + (15 if buy else -15), 5800.25, 40, ts)
            r.on_trade(ts, BUY if buy else SELL, 50, px)
        return r, r.regime_board()

    def test_it_compares_above_against_below_on_each_side(self):
        _, rb = self._board()
        for side in (UP, DOWN):
            e = rb[f"{side}_edge"]
            assert set(e) >= {"book_above_net", "book_below_net",
                              "difference", "n_above", "n_below", "enough"}
            assert e["difference"] == pytest.approx(
                e["book_above_net"] - e["book_below_net"])

    def test_it_refuses_to_call_a_thin_sample_enough(self):
        """Twenty stretches a side is already generous; fewer is not a
        finding however clean the medians look."""
        r = CVDCross(sample_s=0.0, warm_at=3, dwell_s=0.0)
        r.on_book(5800.00, 40, 5800.25, 40, T0)
        for i in range(40):
            ts = T0 + i * 0.25
            r.on_book(5800.00, 40 + i, 5800.25, 40, ts)
            r.on_trade(ts, BUY, 10, 5800.25)
        for side in (UP, DOWN):
            assert r.regime_board()[f"{side}_edge"]["enough"] is False

    def test_momentary_stretches_are_excluded(self):
        """A regime that lasted one sample has no travel to measure
        and would drag every median toward zero."""
        r, _ = self._board()
        loose = r.regime_board(min_seconds=0.0)["episodes"]
        tight = r.regime_board(min_seconds=5.0)["episodes"]
        assert tight <= loose

    def test_every_regime_is_reported_even_when_empty(self):
        r = CVDCross(sample_s=0.0)
        rb = r.regime_board()
        for key in ("up_book_above", "up_book_below",
                    "down_book_above", "down_book_below"):
            assert rb[key]["n"] == 0

    def test_it_serialises(self):
        import json
        r, _ = self._board()
        json.dumps(r.as_dict(with_signals=True))

    def test_the_light_poll_still_does_not_carry_the_board(self):
        r, _ = self._board()
        assert "regime_board" not in r.as_dict()
        assert "regime_board" in r.as_dict(with_signals=True)


class TestThePerBarLineRidesWithTheOthers:
    """It has to sit on the SAME axis as the two cumulative lines.

    Raw, a bar's delta and a session's delta are nowhere near each
    other, so a raw third line would flatten against the axis and the
    usual dodge -- a second y-scale -- is a lie that makes any two
    shapes look related. Z-scoring against closed bars of the same
    size is what makes one axis honest.
    """

    def _fed(self, n=4000, **kw):
        import random
        kw.setdefault("sample_s", 0.25)
        kw.setdefault("bar_seconds", 60.0)
        r = CVDCross(**kw)
        rng = random.Random(2)
        sz = 40.0
        for k in range(1, n):
            sz = max(1.0, sz + rng.randint(-4, 4))
            r.on_book(5800.00, sz, 5800.25, 40, T0 + k * 0.5)
        return r

    def test_every_chart_point_carries_all_three(self):
        d = self._fed().as_dict()
        assert set(d["points"][0]) >= {"ts", "book", "tape", "bar", "px"}

    def test_the_per_bar_line_is_z_scored(self):
        """Not raw contracts -- it would be off the chart."""
        r = self._fed()
        vals = [abs(p["bar"]) for p in r.as_dict()["points"]]
        assert max(vals) < 20.0, "that is not a z-score"
        assert any(v > 0.0 for v in vals), "the line never moved"

    def test_it_waits_for_a_few_closed_bars(self):
        r = self._fed(n=40)
        assert r.snapshot().bar_z == 0.0

    def test_the_per_bar_reading_is_still_published(self):
        """It left the strip when price took the second slot, but the
        engine must still carry it for the pane and the agent."""
        r = self._fed()
        assert "bar" in r.as_dict()["points"][0]


class TestCallingTheBarBeforeItCloses:
    """Can these readings call a 5m bar during its own cycle?

    The danger here is not a wrong answer, it is a flattering one. It
    is trivially easy to "predict" a bar from numbers that already
    contain its close, and the result looks superb and means nothing.
    So: calls are taken part way through the bar from readings that
    existed at that moment, and settled only once the bar has ended.
    """

    def _run(self, n=400_000, bar_s=300.0, seed=77):
        import random
        rng = random.Random(seed)
        r = CVDCross(sample_s=0.25, stat_window=120, warm_at=10,
                     bar_seconds=bar_s)
        ts, bid, ask, bsz, asz, drift = T0, 5799.75, 5800.25, 40.0, 40.0, 0.0
        for i in range(n):
            ts += 0.004
            drift += rng.gauss(0, 0.05)
            up = drift > 0
            bsz = max(1.0, bsz + rng.randint(-4, 4) + (2 if up else -2))
            asz = max(1.0, asz + rng.randint(-4, 4) - (2 if up else -2))
            if rng.random() < 0.012:
                st = 0.25 if rng.random() < 0.5 + 0.09 * (1 if up else -1) \
                    else -0.25
                bid, ask = bid + st, ask + st
            r.on_book(bid, bsz, ask, asz, ts)
            if rng.random() < 0.09:
                sd = BUY if rng.random() < 0.5 + 0.12 * (1 if up else -1) \
                    else SELL
                r.on_trade(ts, sd, rng.choice((1, 1, 2, 3, 5, 30)),
                           ask if sd == BUY else bid)
        return r

    def test_a_call_is_made_before_its_bar_ends(self):
        """The whole measurement rests on this."""
        r = self._run()
        for c in r.calls():
            if c.settled:
                assert c.bar_ts + c.bar_s >= c.bar_ts
                # the call belongs to the bar it was made in
                assert c.at_pct < 1.0

    def test_a_call_settles_against_its_own_bars_close(self):
        r = self._run()
        done = [c for c in r.calls() if c.settled]
        assert done, "nothing settled"
        for c in done:
            move = (c.close_px - c.open_px) / 0.25
            assert c.move_ticks == pytest.approx(move)
            assert c.correct == ((move > 0 and c.call == "up")
                                 or (move < 0 and c.call == "down"))

    def test_an_unfinished_bar_is_not_scored(self):
        r = self._run(n=60_000)
        board = r.call_board()
        for per in board["callers"].values():
            for d in per.values():
                assert d["n"] <= len(r.calls())

    def test_one_call_per_checkpoint_per_bar(self):
        r = self._run()
        seen = set()
        for c in r.calls():
            key = (c.bar_ts, c.caller, c.at_pct)
            assert key not in seen, f"duplicate call {key}"
            seen.add(key)

    def test_the_base_rate_is_reported_beside_the_hit_rate(self):
        """In a session where 56% of bars close green, a caller
        scoring 56% has found nothing."""
        board = self._run().call_board()
        assert "base_up" in board and "naive_rate" in board
        assert 0.0 <= board["base_up"] <= 1.0

    def test_edge_is_measured_against_the_dumbest_rule(self):
        board = self._run().call_board()
        naive = board["naive_rate"]
        for per in board["callers"].values():
            for d in per.values():
                if d["rate"] is not None and d["edge"] is not None:
                    assert d["edge"] == pytest.approx(d["rate"] - naive)

    def test_a_thin_sample_is_not_called_enough(self):
        """A caller scoring 100% on ten bars is noise wearing a
        rosette."""
        board = self._run(n=120_000).call_board()
        for per in board["callers"].values():
            for d in per.values():
                if d["n"] < 100:
                    assert d["enough"] is False

    def test_both_callers_are_reported(self):
        board = self._run().call_board()
        assert set(board["callers"]) == {"regime", "bar_z"}

    def test_the_board_serialises(self):
        import json
        json.dumps(self._run(n=80_000).call_board())

    def test_a_new_session_clears_the_calls(self):
        r = self._run(n=80_000)
        r.on_trade(T0 + 86_400 * 3, BUY, 1, 5800.25)
        assert r.calls() == []


class TestThePaneBesideTheCandles:
    """The two readings on the price chart, in their own pane."""

    def _src(self):
        import inspect

        from liqmap import web as w
        return inspect.getsource(w)

    def test_one_reading_per_bar_not_one_per_sample(self):
        """At chart zoom a bar is a few pixels wide, so 250ms samples
        are sub-pixel. Per-bar IS the chart's resolution, and it
        covers hours instead of the last two minutes."""
        import random
        r = CVDCross(sample_s=0.25, bar_seconds=60.0, warm_at=5)
        rng = random.Random(3)
        sz = 40.0
        for k in range(1, 6000):
            sz = max(1.0, sz + rng.randint(-4, 4))
            r.on_book(5800.00, sz, 5800.25, 40, T0 + k * 0.5)
        marks = r.as_dict()["bar_marks"]
        assert marks
        assert len({m["ts"] for m in marks}) == len(marks)

    def test_the_forming_bar_is_rewritten_not_appended(self):
        r = CVDCross(sample_s=0.0, bar_seconds=300.0, warm_at=2)
        for k in range(1, 60):
            r.on_book(5800.00, 40 + k, 5800.25, 40, T0 + k)
        marks = r.as_dict()["bar_marks"]
        assert len(marks) == 1, "the live bar grew a tail of points"

    def test_marks_carry_the_bars_own_start_time(self):
        r = CVDCross(sample_s=0.0, bar_seconds=60.0, warm_at=2)
        for k in range(1, 400):
            r.on_book(5800.00, 40 + (k % 11), 5800.25, 40, T0 + k)
        for m in r.as_dict()["bar_marks"]:
            assert m["ts"] % 60.0 == 0.0

    def test_the_ring_is_bounded(self):
        r = CVDCross(sample_s=0.0, bar_seconds=60.0)
        for k in range(1, 200_000, 7):
            r.on_book(5800.00, 40 + (k % 11), 5800.25, 40, T0 + k)
        assert len(r.as_dict()["bar_marks"]) <= 400

    def test_a_new_session_drops_the_old_sessions_marks(self):
        r = CVDCross(sample_s=0.0, bar_seconds=60.0)
        for k in range(1, 400):
            r.on_book(5800.00, 40 + (k % 11), 5800.25, 40, T0 + k)
        assert len(r.as_dict()["bar_marks"]) > 1
        later = T0 + 86_400 * 3
        r.on_trade(later, BUY, 1, 5800.25)
        marks = r.as_dict()["bar_marks"]
        # Not empty: the new session opens its own live bar at once.
        # What matters is that nothing from yesterday survived.
        assert all(m["ts"] > T0 + 86_400 for m in marks)
        assert len(marks) == 1

    def test_the_pane_is_its_own_not_an_overlay(self):
        """Overlaying needs a second y-scale on the price pane, and
        two scales on one pane make any two shapes look related."""
        src = self._src()
        assert "cvdTop" in src and "cvdH" in src
        fn = src[src.index("function chartGeom()"):]
        fn = fn[:fn.index("\n}")]
        assert "cvdTop:" in fn, "the pane has no place in the layout"

    def test_the_price_pane_keeps_its_floor(self):
        """A fourth pane must not be able to squash the candles."""
        src = self._src()
        fn = src[src.index("function chartGeom()"):]
        fn = fn[:fn.index("\n}")]
        assert "v + pp + cd > room" in fn

    def test_lines_are_matched_to_candles_by_time_not_position(self):
        """Matching by index would let the two slide against each
        other without ever looking wrong."""
        src = self._src()
        i = src.index("if (cvdPane) {")
        fn = src[i:src.index("if (ppoOn) {", i)]
        assert "cvdMarks.get(bars[i].ts)" in fn

    def test_the_pane_draws_the_book_line_only(self):
        """Price is the candles directly above this pane. Drawing it
        again here would be the same information twice, on a second
        scale, for no gain."""
        src = self._src()
        i = src.index("if (cvdPane) {")
        fn = src[i:src.index("if (ppoOn) {", i)]
        assert "strokeCvd('book', accent, []" in fn
        assert "strokeCvd('price'" not in fn

    def test_a_timeframe_mismatch_draws_nothing_rather_than_a_lie(self):
        src = self._src()
        assert "CVD_FRAMES" in src
        fn = src[src.index("function cvdBarSeconds()"):]
        fn = fn[:fn.index("\n}")]
        assert "$('sInt')" in fn, "it never consults the chart's interval"
