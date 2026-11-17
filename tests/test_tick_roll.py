"""A direction that slides instead of stepping.

Chris: "the tick bar should be a continuous direction tracking with
the timeframe candle tracing at the same time thats how im going to
be making trades".

WHAT THE BAR'S VOTE COULD NOT DO.

    A tick bar's vote is thrown away when the bar closes. The next
    bar starts at nought and is FLAT until it has earned
    `votes_needed` again -- roughly fifteen seconds of a busy tape
    at 5,000 prints. So the direction blinked out and restarted
    every 5,000 prints, and the ten-second dwell restarted with it:
    a flip arriving shortly before a close was forgotten shortly
    after, and the hold could not elapse across a boundary however
    steady the tape was.

    That gap was never a reading. It was the bar's bookkeeping
    showing through, read by a rule as "no direction".

    THIS IS THE SAME CLASS OF BUG AS THE CLOCK CUT. Gate 1 used to
    read the clock-cut series while the chart drew the continuous
    one, so after every 5-minute mark the gate restarted from zero
    while the chart was mid-move. That was fixed by pointing the
    gate at the continuous SERIES. This fixes the restart that was
    left -- the one inside the series, at every count close.

WHAT IT KEEPS. Both guards, unchanged: the `PATH_BAND` margin and
the `votes_needed` floor. They are what stopped the flipping and a
new reading that quietly dropped them would bring all of it back.

WHAT IT COSTS, SAID PLAINLY. The window is warm ONCE, at a cold
start, instead of being cold for the first stretch of every bar. It
is not "never flat": a window whose prints nearly all landed at one
price is a dead tape, and it still says flat, which is correct.
"""

from __future__ import annotations

import pathlib

import pytest

from liqmap.tickbar import (MIN_VOTES, PATH_BAND, VOTE_FRACTION,
                            RollingDir, TickBars)

WEB = (pathlib.Path(__file__).resolve().parents[1]
       / "liqmap" / "web.py").read_text()


def feed(r: RollingDir, ups: int = 0, downs: int = 0, flats: int = 0):
    for _ in range(ups):
        r.on_print(0.25)
    for _ in range(downs):
        r.on_print(-0.25)
    for _ in range(flats):
        r.on_print(0.0)
    return r


class TestTheWindowSlides:
    def test_it_holds_only_the_last_n_prints(self):
        r = feed(RollingDir(100), ups=100)
        assert r.prints == 100 and r.path_dir == "up"
        feed(r, downs=100)
        assert r.prints == 100, "the window grew past its size"
        assert r.up_prints == 0, "an evicted print was never subtracted"
        assert r.path_dir == "down"

    def test_the_counts_do_not_drift_as_prints_are_evicted(self):
        """The deque drops its oldest silently. If the running
        tallies are not told what left they climb forever and the
        share becomes meaningless."""
        r = RollingDir(50)
        for i in range(500):
            r.on_print(0.25 if i % 2 else -0.25)
            assert r.up_prints + r.down_prints == r.prints
            assert r.prints <= 50

    def test_an_abstention_takes_a_slot(self):
        """A print at an unchanged price votes on nothing, but it is
        still one of the last N prints. Letting it skip the window
        would keep a direction alive on a handful of ancient moving
        prints while the tape stood still."""
        r = feed(RollingDir(100), ups=60)
        assert r.path_dir == "up"
        feed(r, flats=100)
        assert r.prints == 100
        assert r.votes == 0, "the stalled prints did not displace them"
        assert r.path_dir == "flat"

    def test_it_turns_over_gradually_rather_than_at_once(self):
        """The property the bar could not have. A bar flips the
        instant it resets; a window has to be argued out of its
        direction print by print."""
        r = feed(RollingDir(1000), ups=1000)
        assert r.path_dir == "up"
        feed(r, downs=400)
        assert r.path_dir == "up", "it gave up its direction too easily"
        feed(r, downs=200)
        assert r.path_dir == "down"


class TestItKeepsBothGuards:
    def test_a_bare_majority_is_not_a_direction(self):
        r = feed(RollingDir(5000), ups=2501, downs=2499)
        assert r.votes >= r.votes_needed
        assert r.path_share < PATH_BAND
        assert r.path_dir == "flat"

    def test_the_band_is_the_same_band(self):
        assert feed(RollingDir(5000), ups=2600, downs=2400).path_dir == "up"
        assert feed(RollingDir(5000), ups=2400,
                    downs=2600).path_dir == "down"

    def test_the_floor_is_the_same_floor(self):
        assert RollingDir(5000).votes_needed == 250
        assert RollingDir(250).votes_needed == MIN_VOTES
        assert RollingDir(10000).votes_needed == int(10000 * VOTE_FRACTION)

    def test_a_cold_window_is_flat_however_one_sided(self):
        r = feed(RollingDir(5000), ups=100)
        assert r.path_share == 1.0
        assert r.warm is False
        assert r.path_dir == "flat"

    def test_a_dead_tape_still_reads_flat(self):
        """NOT "never blank". A window whose prints nearly all
        landed at one price has no direction and has to say so."""
        r = feed(RollingDir(1000), ups=20, flats=980)
        assert r.prints == 1000
        assert r.warm is False
        assert r.path_dir == "flat"


class TestItIsWarmOnceNotOncePerBar:
    """THE WHOLE POINT, STATED AS A MEASUREMENT."""

    @staticmethod
    def _flat_runs(use_roll: bool, per_bar: int = 400, prints: int = 4000):
        """How many prints read FLAT over a steady one-way tape.

        A tape that only ever ticks up. Any flat in it is the
        reading restarting, not the market.
        """
        b = TickBars(per_bar=per_bar)
        t, px, flats = 0.0, 7000.0, 0
        for _ in range(prints):
            t += 0.05
            px += 0.25
            b.on_trade(t, px, 1.0, "B")
            if use_roll:
                d = b.roll.path_dir
            else:
                cur = b.current
                d = cur.path_dir if cur is not None else "flat"
            if d != "up":
                flats += 1
        return flats

    def test_the_bar_goes_flat_again_at_every_close(self):
        """The control. Without it the number below means nothing."""
        n = self._flat_runs(use_roll=False)
        assert n > 200, (
            f"only {n} flat prints -- the control stopped reproducing "
            "the restart this change exists to fix")

    def test_the_window_goes_flat_once(self):
        roll = self._flat_runs(use_roll=True)
        bar = self._flat_runs(use_roll=False)
        assert roll < bar / 4, (roll, bar)
        # And what remains is the cold start alone, which is exactly
        # the floor -- not a restart, a warmup.
        assert roll <= RollingDir(400).votes_needed + 1, roll

    def test_it_does_not_reset_when_a_bar_closes(self):
        b = TickBars(per_bar=50)
        t, px = 0.0, 7000.0
        for _ in range(200):
            t += 1.0
            px += 0.25
            b.on_trade(t, px, 1.0, "B")
        assert b.roll.path_dir == "up"
        before = b.roll.votes
        # Drive it exactly onto a close.
        while b.current is not None:
            t += 1.0
            px += 0.25
            b.on_trade(t, px, 1.0, "B")
        assert b.current is None, "the bar did not close"
        assert b.roll.votes >= before, "the window was reset by a close"
        assert b.roll.path_dir == "up"

    def test_it_sees_the_print_before_across_a_boundary(self):
        """A bar's first print has nothing before it INSIDE that
        bar, so the bar casts no vote for it. The window DOES have
        a predecessor -- the last print of the bar before -- and
        reading `b.close` there would have compared the print to
        itself and silently dropped one vote per bar.

        Driven to 505 prints at 10 a bar, so the 10-print window
        straddles a close: five prints either side of it.
        """
        b = TickBars(per_bar=10)
        t, px = 0.0, 7000.0
        for _ in range(505):
            t += 1.0
            px += 0.25
            b.on_trade(t, px, 1.0, "B")
        assert b.current.trades == 5, "not mid-bar; the setup moved"
        assert b.roll.prints == 10
        assert b.roll.votes == 10, (
            "a print at a bar boundary cast no vote: "
            + str(b.roll.as_dict()))
        # THE CONTRAST, pinned beside it. The bar skips its own
        # first print; the window does not skip anything.
        assert b.current.votes == 4 and b.current.trades == 5

    def test_a_clock_cut_does_not_reset_it_either(self):
        """`per_bar` is large enough that only the CLOCK closes a
        bar here, and small enough that the window's floor is
        reachable -- the window is sized to `per_bar`, so a builder
        of a million would need 50,000 votes before it spoke."""
        b = TickBars(per_bar=2000, period_s=300.0)
        t, px = 0.0, 7000.0
        for _ in range(400):
            t += 10.0                      # crosses several boundaries
            px += 0.25
            b.on_trade(t, px, 1.0, "B")
        assert b._clock_cuts > 0, "the clock never cut a bar"
        assert b.roll.warm is True
        assert b.roll.path_dir == "up"


class TestTheBuilderExposesIt:
    def test_every_builder_has_one_sized_to_itself(self):
        for n in (250, 5000):
            assert TickBars(per_bar=n).roll.window == n

    def test_the_reading_is_a_plain_dict(self):
        b = TickBars(per_bar=100)
        t, px = 0.0, 7000.0
        for _ in range(300):
            t += 1.0
            px += 0.25
            b.on_trade(t, px, 1.0, "B")
        d = b.roll.as_dict()
        assert d["dir"] == "up" and d["window"] == 100
        assert d["votes"] == d["up_prints"] + d["down_prints"]
        assert d["warm"] is True

    def test_a_zero_price_print_is_not_a_vote(self):
        b = TickBars(per_bar=100)
        b.on_trade(1.0, 7000.0, 1.0, "B")
        b.on_trade(2.0, 0.0, 1.0, "B")
        b.on_trade(3.0, 7000.0, 1.0, "B")
        assert b.roll.votes == 0, "a rejected print reached the window"

    @pytest.mark.parametrize("junk", [None, "x", object()])
    def test_rubbish_casts_no_vote_and_does_not_raise(self, junk):
        """The guarantee is that junk never becomes a DIRECTION.
        `None` casts an abstention rather than being dropped, which
        is the same thing a zero delta does and is harmless; what
        must never happen is a vote, or a traceback out of the
        feeding path."""
        r = RollingDir(10)
        r.on_print(junk)
        assert r.votes == 0
        assert r.path_dir == "flat"


class TestItReachesTheRule:
    def test_the_route_reads_the_window(self):
        i = WEB.index("def _tick_now(")
        blk = WEB[i:WEB.index("def _m5_now(", i)]
        assert "bld.roll.as_dict()" in blk
        main = blk[blk.index('"up_prints": cur.up_prints'):]
        assert '"roll": roll' in main, \
            "the forming bar's reading is returned without the window"

    def test_gate_one_reads_the_window_not_the_bar(self):
        """THE WIRING IS THE CHANGE. A correct rolling direction that
        the rule never reads is the smoothing bug again -- it got to
        the route and died one function later."""
        i = WEB.index("def _journal_step_un(")
        blk = WEB[i:i + 30000]
        assert '_rd = (_tk.get("roll") or {}).get("dir")' in blk
        assert "_dw.on_tick(_rd," in blk, \
            "the dwell is still fed the bar's own vote"
        assert "_sw.state(win, _rd," in blk, \
            "gate 1 is still fed the bar's own vote"
        # AND THE COUNT BEHIND IT REACHES THE PANEL. Passing {} here
        # left the panel unable to tell a cold start from a flat
        # market -- the exact confusion this change removes -- and
        # a state-level test of `roll` could not see it, because
        # `state` was still perfectly capable of carrying a window
        # nobody handed it.
        assert 'roll=_tk.get("roll")' in blk, \
            "the window's counts never reach the panel"

    def test_the_panel_can_show_a_cold_start_as_warming(self):
        assert "warming up" in WEB
        assert "not every bar" in WEB, \
            "nothing says the warmup happens once"

    def test_the_state_carries_the_window_behind_the_side(self):
        from liqmap import stackwin as sw
        st = sw.state({}, "up", {}, 300.0,
                      roll={"window": 5000, "warm": True})
        assert st["tick"]["roll"]["window"] == 5000


class TestItDoesNotTrade:
    def test_nothing_here_submits_an_order(self):
        import ast
        from liqmap import tickbar as tb
        tree = ast.parse(pathlib.Path(tb.__file__).read_text())
        bad = [n.name for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name in ("submit", "execute", "place", "order")]
        assert not bad, f"tickbar.py grew {bad}"
