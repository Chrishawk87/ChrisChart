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
        assert r["ticks"] == 0.0
        assert r["forming"] is True

    def test_above_its_open_reads_up(self):
        b = builder()
        book(b, 0.0, PX)
        book(b, 1.0, PX + TICK)
        assert b.bar_read["read"] == "up"

    def test_below_its_open_reads_down(self):
        b = builder()
        book(b, 0.0, PX)
        book(b, 1.0, PX - TICK)
        assert b.bar_read["read"] == "down"

    def test_back_at_the_open_reads_flat(self):
        """FLAT IS PRINTED, NOT SUPPRESSED. Showing the last direction
        through it would be a reading of a moment that has passed."""
        b = builder()
        book(b, 0.0, PX)
        book(b, 1.0, PX + 4 * TICK)
        assert b.bar_read["read"] == "up"
        book(b, 2.0, PX)
        assert b.bar_read["read"] == "flat"


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
        assert b.bar_read["read"] == "up"
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
        assert after["read"] == before["read"] == "up"
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


class TestTheImbalanceDoesNotDecideTheDirection:
    """It decides when the bar ENDS. That is a different question, and
    the read answering the second one is the change Chris asked for."""

    def test_one_sided_selling_with_price_up_still_reads_up(self):
        b = builder()
        book(b, 0.0, PX)
        # Size stacked on the bid against a rising price: the
        # imbalance and the price point opposite ways.
        for i in range(20):
            book(b, 0.1 * (i + 1), PX + TICK, bsz=10, asz=900)
        r = b.bar_read
        assert r["read"] == "up", r
        assert r["ticks"] > 0

    def test_the_read_carries_no_imbalance_figure(self):
        b = builder()
        book(b, 0.0, PX)
        book(b, 1.0, PX + TICK)
        r = b.bar_read
        for k in ("ofi", "lead", "gross", "share", "acc", "imbalance"):
            assert k not in r, f"{k} leaked into the read"


class TestHoldAndCrossings:
    def test_it_holds_while_it_stays_one_side(self):
        b = builder()
        book(b, 0.0, PX)
        book(b, 1.0, PX + TICK)
        book(b, 5.0, PX + TICK)
        assert b.bar_read["held_s"] == pytest.approx(4.0, abs=0.01)

    def test_a_crossing_restarts_the_hold(self):
        b = builder()
        book(b, 0.0, PX)
        book(b, 1.0, PX + TICK)
        book(b, 5.0, PX + TICK)
        book(b, 6.0, PX - TICK)
        r = b.bar_read
        assert r["read"] == "down"
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
