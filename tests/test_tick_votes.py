"""A direction needs enough votes, and a wide enough margin, to be one.

Chris: "this is having a bunch of flipping issues what can we do to
keep that from happening because the 10's hold is not holding".

THE HOLD WAS NOT FAILING. IT WAS BEING RESET.

    The tick bar's direction was a bare majority of prints with no
    threshold of any kind, so two things flipped it constantly:

      NEAR PARITY a single print turns it over. 2,501 up against
      2,499 down is a margin of two.

      AT THE START OF A BAR the counts are single digits, where a
      percentage band is no protection at all -- the FIRST print
      makes the share 1.000, one vote to nothing. The bar is cut at
      every true 5-minute mark, so that happens twelve times an
      hour.

    Every turn restarted the entry dwell, so the ten seconds never
    elapsed. Measured over an hour of simulated tape: 359 resets
    with no guards, 15 with both.

THIS IS THE THIRD TIME THIS CLASS OF BUG HAS APPEARED HERE. The
pressure reading carries MIN_TRADES for it and the ratio tracker
carries min_vol. A ratio is only a reading once there is something
to take a ratio of.
"""

from __future__ import annotations

import pytest

from liqmap.tickbar import (MIN_VOTES, PATH_BAND, VOTE_FRACTION,
                            TickBar, TickBars)


def bar(up: int, dn: int, per_bar: int = 5000) -> TickBar:
    """A bar with a known vote split, built through the real path."""
    # The builder is left effectively uncapped so the bar never
    # closes mid-helper -- setting `b.per_bar` here closed it at
    # exactly `per_bar` prints and handed back the NEXT bar, empty.
    # Only the bar's own `per_bar` matters, because that is what
    # `votes_needed` scales against.
    b = TickBars(per_bar=10 ** 9)
    px, t = 7000.0, 1.0
    b.on_trade(t, px, 1.0, "B")
    b._cur.per_bar = per_bar
    for _ in range(up):
        t += 1.0
        px += 0.25
        b.on_trade(t, px, 1.0, "B")
    for _ in range(dn):
        t += 1.0
        px -= 0.25
        b.on_trade(t, px, 1.0, "B")
    return b.current


class TestABarMustCountBeforeItCalls:
    def test_the_first_print_is_not_a_direction(self):
        """It used to be. One vote to nothing is a share of 1.000,
        and a percentage band cannot see that."""
        b = TickBars(per_bar=5000)
        b.on_trade(1.0, 7000.0, 1.0, "B")
        b.on_trade(2.0, 7000.25, 1.0, "B")
        assert b.current.path_share == 1.0
        assert b.current.path_dir == "flat"

    def test_the_minimum_scales_with_the_bar(self):
        assert bar(1, 0, per_bar=5000).votes_needed == 250
        assert bar(1, 0, per_bar=10000).votes_needed == 500

    def test_a_small_bar_gets_the_floor_instead(self):
        """5% of 250 is 12, which would let a dozen prints decide.
        The floor stops a short bar from being louder than a long
        one."""
        assert bar(1, 0, per_bar=250).votes_needed == MIN_VOTES
        assert MIN_VOTES > int(250 * VOTE_FRACTION)

    def test_under_the_minimum_is_flat_however_one_sided(self):
        b = bar(100, 0, per_bar=5000)          # every vote one way
        assert b.path_share == 1.0
        assert b.votes < b.votes_needed
        assert b.path_dir == "flat"

    def test_over_the_minimum_it_speaks(self):
        b = bar(300, 0, per_bar=5000)
        assert b.votes >= b.votes_needed
        assert b.path_dir == "up"

    def test_votes_counts_only_prints_that_moved_price(self):
        b = TickBars(per_bar=5000)
        for i, px in enumerate((7000.0, 7000.0, 7000.0, 7000.25)):
            b.on_trade(float(i + 1), px, 1.0, "B")
        assert b.current.votes == 1, "an unchanged print voted"


class TestTheMarginHasToBeWideEnough:
    def test_a_bare_majority_is_not_a_direction(self):
        """2,501 against 2,499 flipped on a single print."""
        b = bar(2501, 2499)
        assert b.votes >= b.votes_needed
        assert b.path_share < PATH_BAND
        assert b.path_dir == "flat"

    def test_the_band_is_the_threshold(self):
        inside = bar(2520, 2480)                # share 0.008
        outside = bar(2600, 2400)               # share 0.040
        assert inside.path_share < PATH_BAND and inside.path_dir == "flat"
        assert outside.path_share > PATH_BAND and outside.path_dir == "up"

    def test_it_is_symmetric(self):
        assert bar(2400, 2600).path_dir == "down"
        assert bar(2600, 2400).path_dir == "up"

    def test_the_band_gives_hysteresis_for_free(self):
        """Claiming a side needs the band; turning that side over
        needs twice it, because the margin has to cross from one
        edge to the other. A direction is harder to lose than it
        was to gain, which is the property that stops flapping."""
        up = bar(2600, 2400)
        assert up.path_dir == "up"
        # to read DOWN the vote must swing the full width of the
        # band and out the other side
        assert bar(2550, 2450).path_dir == "up"     # still up
        assert bar(2500, 2500).path_dir == "flat"   # inside
        assert bar(2400, 2600).path_dir == "down"   # out the far side


class TestWhatItCosts:
    def test_a_fresh_bar_is_silent_for_its_first_votes(self):
        """Stated as a test because it is the cost, not a defect.
        At 5,000 prints and a busy tape that is roughly fifteen
        seconds of every five-minute bar."""
        b = TickBars(per_bar=5000)
        px, t = 7000.0, 0.0
        b.on_trade(t, px, 1.0, "B")
        for i in range(240):
            t += 1.0
            px += 0.25
            b.on_trade(t, px, 1.0, "B")
        assert b.current.path_dir == "flat", "it called one too early"
        for i in range(30):
            t += 1.0
            px += 0.25
            b.on_trade(t, px, 1.0, "B")
        assert b.current.path_dir == "up"

    def test_the_panel_says_counting_rather_than_going_blank(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        assert "counting &mdash; " in src, \
            "a bar that is still counting looks like a dead market"
        assert "votes_needed" in src


class TestItReachesTheRule:
    def test_the_route_reports_the_counts(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        i = src.index("def _tick_now(")
        blk = src[i:src.index("def _m5_now(", i)]
        assert '"votes": cur.votes' in blk
        assert '"votes_needed": cur.votes_needed' in blk

    def test_fusion_passes_them_through(self):
        """This dict is a reshape: a field not named here is
        silently dropped, which is how the smoothing once reached
        the route and died one function later."""
        from liqmap import fusion
        st = fusion.state({"dir": "up", "votes": 300,
                           "votes_needed": 250},
                          {"price_dir": "up"}, {"dir": "up"})
        assert st["tick"]["votes"] == 300
        assert st["tick"]["votes_needed"] == 250


class TestTheGuardsTogetherStopTheReset:
    """The measurement behind the change, run small enough to live
    in a test: the entry dwell has to survive long enough to
    elapse."""

    @staticmethod
    def _resets(band, frac, minv, bars=4, n=1200, seed=11):
        import random
        import liqmap.tickbar as tb
        from liqmap.stackwin import Dwell

        ob, of, om = tb.PATH_BAND, tb.VOTE_FRACTION, tb.MIN_VOTES
        tb.PATH_BAND, tb.VOTE_FRACTION, tb.MIN_VOTES = band, frac, minv
        try:
            rng = random.Random(seed)
            dw = Dwell()
            px, t, dt = 7000.0, 0.0, 300.0 / n
            for _ in range(bars):
                b = tb.TickBars(per_bar=5000)
                b.on_trade(t, px, 1.0, "B")
                for _ in range(n - 1):
                    t += dt
                    px += 0.25 if rng.random() < 0.502 else -0.25
                    b.on_trade(t, px, 1.0, "B")
                    cur = b.current
                    dw.on_tick(cur.path_dir if cur else "flat", t)
            return dw.flips
        finally:
            tb.PATH_BAND, tb.VOTE_FRACTION, tb.MIN_VOTES = ob, of, om

    def test_both_guards_beat_either_alone(self):
        none = self._resets(0.0, 0.0, 0)
        band = self._resets(PATH_BAND, 0.0, 0)
        votes = self._resets(0.0, VOTE_FRACTION, MIN_VOTES)
        both = self._resets(PATH_BAND, VOTE_FRACTION, MIN_VOTES)
        assert none > 0, "the model produced no flipping to fix"
        assert both < band, (both, band)
        assert both < votes, (both, votes)
        assert both * 4 < none, (both, none)
