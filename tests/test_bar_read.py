"""The read is the OFI candle now, not a five-minute clock window.

WHAT CHANGED AND WHY IT MATTERS FOR EVERY TEST BELOW.

It was a flow reading: buying minus selling since the last clock
boundary, reset every five minutes whatever the bars were doing. Chris
asked for the other thing -- what the candle on the screen is doing at
this moment -- so it is now where price sits against the open of the
bar in progress, and it resets when that bar does.

That makes it a PRICE measurement on a FLOW window, and both halves
have consequences worth pinning:

  - the window is a quantity of flow, so it can reset twice in a
    minute or not for ten minutes. There is no "time left in the
    candle" and nothing here may pretend to know one;
  - the imbalance decides when the bar ENDS. It must not decide which
    way the bar reads. Those are different questions and the whole
    point of the change is that the read answers the second one;
  - a bar that keeps crossing its own open reads UP or DOWN either
    way. Only the crossing count can say it is going nowhere, so the
    count has to be real -- every crossing the bar made, not the ones
    a polling panel happened to look at.
"""

from __future__ import annotations

import pytest

from liqmap.ofibar import OfiBarBuilder

POINT = 1_000_000_000
TICK = POINT // 4
PX = 6700 * POINT


def builder(target: int = 2000) -> OfiBarBuilder:
    return OfiBarBuilder(tick_size=0.25, threshold=target, max_s=900.0,
                         min_updates=2)


def book(b: OfiBarBuilder, ts: float, bid: int, bsz: int = 100,
         asz: int = 100):
    """One book update at `bid`, one tick wide."""
    return b.on_book(ts, bid, bsz, bid + TICK, asz)


class TestItReadsTheBar:
    def test_nothing_forming_reads_nothing(self):
        r = builder().bar_read
        assert r["read"] == ""
        assert r["forming"] is False

    def test_a_new_bar_opens_flat(self):
        b = builder()
        book(b, 0.0, PX)
        r = b.bar_read
        assert r["read"] == "flat"
        assert r["price_dir"] == "flat"
        assert r["ticks"] == 0.0
        assert r["forming"] is True

    def test_above_its_open_shows_price_up(self):
        """`read` is the OFI now; the price path is `price_dir`."""
        b = builder()
        book(b, 0.0, PX)
        book(b, 1.0, PX + TICK)
        assert b.bar_read["price_dir"] == "up"

    def test_below_its_open_shows_price_down(self):
        b = builder()
        book(b, 0.0, PX)
        book(b, 1.0, PX - TICK)
        assert b.bar_read["price_dir"] == "down"

    def test_back_at_the_open_reads_flat(self):
        """FLAT IS PRINTED, NOT SUPPRESSED. Showing the last direction
        through it would be a reading of a moment that has passed."""
        b = builder()
        book(b, 0.0, PX)
        book(b, 1.0, PX + 4 * TICK)
        assert b.bar_read["price_dir"] == "up"
        book(b, 2.0, PX)
        assert b.bar_read["price_dir"] == "flat"


class TestTheNumberIsTheEvidence:
    """UP by a quarter of a tick and UP by four points are the same
    word and not the same thing."""

    @pytest.mark.parametrize("offset,ticks,points", [
        (TICK, 1.0, 0.25),
        (4 * TICK, 4.0, 1.0),
        (-2 * TICK, -2.0, -0.5),
        (-16 * TICK, -16.0, -4.0),
    ])
    def test_distance_from_the_open(self, offset, ticks, points):
        b = builder()
        book(b, 0.0, PX)
        book(b, 1.0, PX + offset)
        r = b.bar_read
        assert r["ticks"] == ticks
        assert r["points"] == points

    def test_raw_units_are_converted(self):
        """The bar's prices are raw -- a point is a billion of them --
        and the class tick is 0.25. Dividing one by the other gives a
        number four billion times too big, which reads on the panel as
        a one-tick move being a thousand points."""
        b = builder()
        book(b, 0.0, PX)
        book(b, 1.0, PX + TICK)
        assert abs(b.bar_read["ticks"]) < 10

    def test_the_prices_are_the_bars_own(self):
        b = builder()
        book(b, 0.0, PX)
        book(b, 1.0, PX + 4 * TICK)
        r = b.bar_read
        assert r["open_px"] == float(PX + TICK / 2)
        assert r["last_px"] == float(PX + 4 * TICK + TICK / 2)
        assert r["last_px"] - r["open_px"] == float(4 * TICK)


class TestItResetsWithTheBarAndOnNothingElse:
    def _run_to_a_close(self, b, start_ts=0.0, px=PX):
        """Push one-sided flow until a bar closes.

        The imbalance accumulates from CHANGES in the book, not from
        its state: a static book at any size contributes zero however
        long it is held. So the bid grows step by step, which is what
        `level_ofi` reads as buying pressure at an unchanged price.
        """
        ts, n = start_ts, len(b.bars)
        for i in range(600):
            ts += 0.1
            book(b, ts, px, bsz=100 + 20 * (i + 1), asz=100)
            if len(b.bars) > n:
                return ts, len(b.bars) - n
        return ts, 0

    def test_a_closing_bar_restarts_the_read(self):
        b = builder(target=250)
        book(b, 0.0, PX)
        book(b, 0.5, PX + 8 * TICK)
        assert b.bar_read["price_dir"] == "up"
        ts, closed = self._run_to_a_close(b, 0.5)
        assert closed >= 1, "fixture never closed a bar"
        # The new bar's read is measured from ITS open, so a bar that
        # opens where the last one ended starts at flat rather than
        # inheriting eight ticks of someone else's move.
        r = b.bar_read
        assert r["bar_ts"] > 0.0
        assert abs(r["ticks"]) < 8

    def test_the_crossing_count_does_not_survive_the_bar(self):
        """A count that carried over would describe two bars and
        belong to neither."""
        b = builder(target=250)
        book(b, 0.0, PX)
        for i in range(6):
            book(b, 0.1 * (i + 1), PX + (TICK if i % 2 == 0 else -TICK))
        assert b.bar_read["flips"] >= 3
        self._run_to_a_close(b, 1.0)
        assert b.bar_read["flips"] <= 2

    def test_a_clock_does_not_reset_it(self):
        """THE WHOLE POINT OF THE CHANGE. Five minutes passing inside
        one bar leaves the reading exactly where it was."""
        b = builder()
        book(b, 0.0, PX)
        book(b, 1.0, PX + 4 * TICK)
        before = b.bar_read
        book(b, 601.0, PX + 4 * TICK)
        after = b.bar_read
        assert after["price_dir"] == before["price_dir"] == "up"
        assert after["bar_ts"] == before["bar_ts"]
        assert after["into_s"] > 600

    def test_it_reports_age_not_time_remaining(self):
        """There is no time left in an OFI bar. It ends when the
        imbalance says so, and nothing here knows when that is."""
        b = builder()
        book(b, 0.0, PX)
        book(b, 30.0, PX + TICK)
        r = b.bar_read
        assert "left_s" not in r
        assert r["into_s"] == pytest.approx(30.0, abs=0.01)


class TestTheReadIsTheOfiNotThePricePath:
    """This class asserted the opposite and is rewritten, not fixed.

    It existed to pin that the imbalance decides when the bar ENDS
    and never which way it reads. Chris then told me what he was
    actually looking at -- "the OFI candle read is up when the OFI
    candle is green" -- and green means the imbalance crossed the
    level. The read IS the imbalance now; the price path is carried
    beside it as `price_dir`.
    """

    def test_one_sided_buying_reads_up_whatever_price_does(self):
        b = builder()
        book(b, 0.0, PX)
        # Size stacked on the bid while price sits still.
        for i in range(30):
            book(b, 0.1 * (i + 1), PX, bsz=100 + 60 * i, asz=100)
        r = b.bar_read
        assert r["read"] == "up"
        assert r["price_dir"] == "flat"

    def test_the_price_path_is_still_carried(self):
        b = builder()
        book(b, 0.0, PX)
        book(b, 1.0, PX + TICK)
        r = b.bar_read
        assert "price_dir" in r and "ticks" in r
        assert r["ticks"] > 0


class TestHoldAndCrossings:
    def test_it_holds_while_it_stays_one_side(self):
        b = builder()
        book(b, 0.0, PX)
        book(b, 1.0, PX + TICK)
        book(b, 5.0, PX + TICK)
        assert b.bar_read["held_s"] == pytest.approx(4.0, abs=0.01)

    def test_a_crossing_restarts_the_hold(self):
        """`read` is the OFI now, so the price-path fields are
        checked on their own names. The crossing count and the hold
        still describe price against the bar open -- they are
        evidence carried beside the signal, not the signal."""
        b = builder()
        book(b, 0.0, PX)
        book(b, 1.0, PX + TICK)
        book(b, 5.0, PX + TICK)
        book(b, 6.0, PX - TICK)
        r = b.bar_read
        assert r["price_dir"] == "down"
        assert r["held_s"] == pytest.approx(0.0, abs=0.01)
        assert r["flips"] == 1

    def test_flat_reports_no_hold(self):
        """`_r_sign` survives a passage through the open on purpose --
        up, flat, up is one reading -- but a duration printed next to
        the word FLAT would belong to a direction the line is not
        showing."""
        b = builder()
        book(b, 0.0, PX)
        book(b, 1.0, PX + TICK)
        book(b, 5.0, PX)
        r = b.bar_read
        assert r["read"] == "flat"
        assert r["held_s"] == 0.0

    def test_touching_the_open_is_not_a_crossing(self):
        b = builder()
        book(b, 0.0, PX)
        book(b, 1.0, PX + TICK)
        book(b, 2.0, PX)
        book(b, 3.0, PX + TICK)
        assert b.bar_read["flips"] == 0

    def test_every_crossing_counts_not_just_the_observed_ones(self):
        """Derived at read time, this would count only the changes a
        caller happened to look at -- a different number every time a
        polling panel asked."""
        b = builder()
        book(b, 0.0, PX)
        for i in range(10):
            book(b, 0.1 * (i + 1), PX + (TICK if i % 2 == 0 else -TICK))
        once = b.bar_read["flips"]
        assert once == 9, once
        # Asking again changes nothing.
        assert b.bar_read["flips"] == once
        assert b.bar_read["flips"] == once


class TestTheDeadbandStopsTheFlipping:
    """Chris: "the flips in the OFI candles are killing the trades".

    The accumulator crosses zero constantly, so reading its bare sign
    made the candle change direction about three times a bar --
    measured on a synthetic tape, 10,178 sign changes over 3,213
    bars -- and every one was a flip exit on a trade that had not
    gone wrong. Requiring it past the call level before it counts as
    a side took that to 1.0 a bar at 500 and 0.4 at 1,000.

    THE BAND IS THE LEVEL ALREADY ON THE PANEL. A second threshold
    of its own would be a number to keep in step with that one for
    no reason.
    """

    @staticmethod
    def _b(level):
        return OfiBarBuilder(tick_size=0.25, threshold=2000,
                             max_s=900.0, min_updates=2,
                             call_level=level)

    def _lean(self, b, steps, up=True):
        px = PX
        b.on_book(0.0, px, 100, px + TICK, 100)
        for i in range(steps):
            if up:
                b.on_book(1.0 + i, px, 100 + 60 * i, px + TICK, 100)
            else:
                b.on_book(1.0 + i, px, 100, px + TICK, 100 + 60 * i)
        return b.bar_read

    def test_inside_the_band_reads_flat(self):
        r = self._lean(self._b(500), 3)
        assert 0 < r["imbalance"] < 500
        assert r["read"] == "flat"

    def test_past_the_band_reads_up(self):
        r = self._lean(self._b(500), 30)
        assert r["imbalance"] >= 500
        assert r["read"] == "up"

    def test_past_the_band_reads_down(self):
        r = self._lean(self._b(500), 30, up=False)
        assert r["imbalance"] <= -500
        assert r["read"] == "down"

    def test_a_bigger_band_holds_out_longer(self):
        """The control Chris turns when the flips are killing him."""
        small = self._lean(self._b(100), 6)
        big = self._lean(self._b(1500), 6)
        assert small["read"] == "up"
        assert big["read"] == "flat"
        assert big["raw_side"] == "up"

    def test_the_lean_is_still_reported(self):
        """So the panel says "leaning up but inside your level"
        rather than showing a blank, which is how a working band
        gets mistaken for a dead feed."""
        r = self._lean(self._b(1500), 6)
        assert r["read"] == "flat"
        assert r["raw_side"] == "up"
        assert r["level"] == 1500

    def test_the_band_is_the_panels_level(self):
        assert self._b(250).bar_read["level"] == 250
        assert self._b(1000).bar_read["level"] == 1000

    def test_a_zero_band_is_the_old_sign_test(self):
        r = self._lean(self._b(0), 3)
        assert r["read"] == r["raw_side"] == "up"
