"""Push from the tape, and why it takes two numbers.

Chris: "standard OFI indicators track modifications to the limit
order book, which algorithms constantly spoof, flash, and cancel ...
you must completely abandon static order book snapshots and look at
Level 1 transactional data (the tape)".

The diagnosis is right and this service measured the consequence:
the book OFI's sign changes 3.17 times a bar against a zero
deadband. Everything here comes from executed prints, which cannot
be cancelled after the fact.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from liqmap.kinetic import Kinetic

ROOT = Path(__file__).resolve().parents[1]


def feed(k: Kinetic, side: str, n: int, dp: float, v: float = 100.0,
         t0: float = 1000.0, px0: float = 6800.0) -> Kinetic:
    px, t = px0, t0
    for _ in range(n):
        px += dp
        t += 0.5
        k.on_trade(t, px, v, side)
    return k


class TestItDoesNotTrade:
    def test_no_order_entry_points(self):
        tree = ast.parse((ROOT / "liqmap" / "kinetic.py").read_text())
        names = {n.name for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for banned in ("submit", "execute", "place", "order", "send_order"):
            assert banned not in names, banned

    def test_it_touches_no_book(self):
        """The whole premise. A reading built from resting size
        would be exactly the thing this replaces."""
        src = (ROOT / "liqmap" / "kinetic.py").read_text()
        tree = ast.parse(src)
        names = {n.name for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        assert "on_book" not in names
        for leak in ("bid_sz", "ask_sz", "level_ofi", "Touch"):
            assert leak not in src, leak


class TestTheFourQuadrants:
    """One signed number cannot carry two independent binaries.

    Who crossed the spread and which way price went are four states;
    a scalar has one sign and separates two. That is an
    information-capacity limit, not a weighting problem, so there
    are two readings here rather than one.
    """

    def test_buying_that_works(self):
        k = feed(Kinetic(), "B", 40, +0.25)
        assert k.push > 0
        assert k.efficiency > 0
        assert k.quadrant == "buying works"

    def test_selling_that_works(self):
        k = feed(Kinetic(), "A", 40, -0.25)
        assert k.push < 0
        assert k.efficiency > 0, "co-movement, not direction"
        assert k.quadrant == "selling works"

    def test_buying_absorbed(self):
        k = feed(Kinetic(), "B", 40, -0.25)
        assert k.push > 0
        assert k.efficiency < 0
        assert k.quadrant == "buying absorbed"

    def test_selling_absorbed(self):
        k = feed(Kinetic(), "A", 40, +0.25)
        assert k.push < 0
        assert k.efficiency < 0
        assert k.quadrant == "selling absorbed"

    def test_the_two_selling_states_are_distinguishable(self):
        """Chris's original formula gave a steady sell-off and a
        steady rally THE SAME POSITIVE NUMBER. Splitting the reading
        is what fixes that, and this is the assertion that proves
        it."""
        rally = feed(Kinetic(), "B", 40, +0.25)
        selloff = feed(Kinetic(), "A", 40, -0.25)
        assert rally.efficiency == pytest.approx(selloff.efficiency,
                                                 rel=0.05), \
            "efficiency alone cannot tell them apart -- that is the point"
        assert rally.push > 0 > selloff.push, \
            "push is what separates them"

    def test_no_price_response(self):
        k = feed(Kinetic(), "B", 40, 0.0)
        assert k.push > 0
        assert k.efficiency == 0.0
        assert k.quadrant == "no price response"

    def test_balanced_is_not_a_side(self):
        k = Kinetic()
        feed(k, "B", 10, +0.25)
        feed(k, "A", 10, -0.25, t0=1010.0, px0=6802.5)
        assert k.push == 0
        assert k.direction == "flat"
        assert k.quadrant == "balanced"


class TestEfficiencyIsInPrice:
    def test_it_is_points_per_contract(self):
        """Dividing a (volume x price) sum by volume leaves price,
        which makes the number readable rather than an index."""
        k = feed(Kinetic(), "B", 40, +0.25, v=100.0)
        assert 0 < k.efficiency < 1.0
        assert k.to_dict()["eff_ticks"] == pytest.approx(
            k.efficiency / 0.25, rel=1e-6)

    def test_a_bigger_move_on_the_same_volume_scores_higher(self):
        small = feed(Kinetic(), "B", 40, +0.25)
        big = feed(Kinetic(), "B", 40, +1.00)
        assert big.efficiency > small.efficiency

    def test_it_is_NOT_sensitive_to_size(self):
        """I claimed it was, and it is not.

        Both sums scale with volume, so the volume cancels: the
        same move at 50 and at 200 contracts a print reads
        identically. That makes it a volume-weighted average price
        change, not a cost-per-contract -- and "how expensive was
        this move in size" is a question it simply does not answer.
        Pinned so the wrong reading cannot creep back into the
        docstring.
        """
        light = feed(Kinetic(), "B", 40, +0.25, v=50.0)
        heavy = feed(Kinetic(), "B", 40, +0.25, v=200.0)
        assert light.efficiency == pytest.approx(heavy.efficiency,
                                                 rel=1e-9)

    def test_the_docstring_says_so(self):
        src = (ROOT / "liqmap" / "kinetic.py").read_text()
        assert "the volume cancels" in src
        assert "not" in src and "cost-per-contract" in src


class TestTheRollingWindow:
    def test_old_prints_leave(self):
        k = Kinetic(period_s=10.0)
        for i in range(30):
            k.on_trade(1000.0 + i, 6800.0 + i * 0.25, 10.0, "B")
        assert k.prints == 30
        assert k.to_dict()["volume"] < 30 * 10.0

    def test_an_emptied_window_reads_exactly_zero(self):
        """Floating error accumulates over hundreds of thousands of
        adds and subtracts. An empty window reading 3e-12 would
        report a direction on no data at all."""
        k = Kinetic(period_s=5.0)
        for i in range(200):
            k.on_trade(1000.0 + i * 0.1, 6800.0 + i * 0.25, 7.3, "B")
        k.on_trade(9999.0, 6800.0, 1.0, "N")     # far future, unsigned
        assert k.push == 0.0
        assert k.efficiency == 0.0
        assert k.direction == "flat"

    def test_it_does_not_reset_on_a_boundary(self):
        """A rolling window answers "what is happening now". A bar
        that is four minutes old answers something else."""
        k = Kinetic(period_s=300.0)
        feed(k, "B", 40, +0.25, t0=0.0)
        before = k.push
        k.on_trade(301.0, 6900.0, 1.0, "B")
        assert k.push != before


class TestUnattributedPrints:
    def test_they_do_not_enter_either_sum(self):
        k = Kinetic()
        feed(k, "N", 40, +0.25)
        assert k.push == 0.0
        assert k.efficiency == 0.0

    def test_they_are_counted(self):
        k = Kinetic()
        feed(k, "N", 10, +0.25)
        assert k.unknown == 10
        assert k.prints == 10

    def test_they_still_move_the_price_reference(self):
        """Skipping them here would attribute their move to the next
        signed print, which is a bigger lie than ignoring them."""
        a = Kinetic()
        a.on_trade(1.0, 6800.0, 10, "B")
        a.on_trade(2.0, 6810.0, 10, "N")   # ten points, unattributed
        a.on_trade(3.0, 6810.25, 10, "B")  # a quarter point, attributed
        assert a.efficiency == pytest.approx(0.25 * 10 / 20, rel=1e-6), \
            "the 10-point move must not be credited to the last buy"

    def test_the_attributed_share_is_published(self):
        k = Kinetic()
        feed(k, "B", 8, +0.25)
        feed(k, "N", 2, +0.25, t0=1010.0, px0=6802.0)
        assert k.attributed_pct == 80.0

    def test_a_reading_with_no_attribution_says_so(self):
        k = Kinetic()
        feed(k, "N", 5, +0.25)
        assert "no attributable prints" in k.to_dict()["say"]


class TestItIsWiredIn:
    def test_the_tape_carries_it(self):
        from liqmap.cme_tape import Tape
        assert "kinetic" in Tape.__dataclass_fields__

    def test_it_is_fed_from_the_raw_record(self):
        """So an unattributed print still moves the price reference."""
        src = (ROOT / "liqmap" / "cme_tape.py").read_text()
        feed_at = src.index("self.kinetic.on_trade(_ts, _rpx, _rsz, _sd)")
        decode_at = src.index("t = dbn.trade_from(rec)")
        assert feed_at < decode_at

    def test_the_guard_does_not_need_another_builder(self):
        src = (ROOT / "liqmap" / "cme_tape.py").read_text()
        assert "or self.kinetic is not None" in src

    def test_the_service_creates_it(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        assert "svc.tape.kinetic = Kinetic(period_s=300.0)" in src

    def test_the_route_serves_it(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        assert '"kinetic": _kin,' in src

    def test_the_panel_draws_both_numbers(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("function kinBlock(")
        # The block grew when the bounded ratio and the candle
        # reading were added; a fixed slice is an assertion about
        # its length rather than its content.
        blk = src[i:src.index("function paintFusion(", i)]
        assert "DIRECTION" in blk and "EFFICIENCY" in blk
        assert "nothing cancellable" in src

    def test_the_panel_warns_on_poor_attribution(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("function kinBlock(")
        blk = src[i:src.index("function paintFusion(", i)]
        assert "attributed_pct" in blk
        assert "suspicion" in blk


class TestTheThreeGoals:
    """Chris asked whether this achieves three specific things.
    One did, two did not, and these pin all three.
    """

    # 1. SPOOFING -----------------------------------------------------

    def test_it_only_ever_sees_executed_trades(self):
        """"They cannot fake executed volume."

        The guarantee is structural: there is no path into this
        class that carries resting size. A book method appearing
        here later would be the whole premise quietly undone.
        """
        import ast as _ast
        src = (ROOT / "liqmap" / "kinetic.py").read_text()
        fns = {n.name for n in _ast.walk(_ast.parse(src))
               if isinstance(n, _ast.FunctionDef)}
        assert fns & {"on_trade"}
        assert not (fns & {"on_book", "on_quote", "on_depth"})
        for leak in ("bid_sz", "ask_sz", "bid_px", "ask_px",
                     "level_ofi", "Touch", "levels"):
            assert leak not in src, leak

    def test_a_cancelled_order_cannot_reach_it(self):
        """There is no argument through which one could."""
        import inspect
        sig = inspect.signature(Kinetic.on_trade)
        assert list(sig.parameters) == ["self", "ts", "px", "size", "side"]

    # 2. BOUNDED ------------------------------------------------------

    def test_the_ratio_is_plus_one_when_every_contract_bought(self):
        k = feed(Kinetic(), "B", 30, +0.25)
        assert k.push_ratio == pytest.approx(1.0)

    def test_minus_one_when_every_contract_sold(self):
        k = feed(Kinetic(), "A", 30, -0.25)
        assert k.push_ratio == pytest.approx(-1.0)

    def test_zero_when_balanced(self):
        k = Kinetic()
        feed(k, "B", 10, +0.25)
        feed(k, "A", 10, -0.25, t0=1010.0, px0=6802.5)
        assert k.push_ratio == pytest.approx(0.0)

    def test_it_is_bounded_by_construction_not_by_clamping(self):
        """push is buy - sell, volume is buy + sell, both
        non-negative -- the magnitude cannot exceed one. No scaling
        constant, nothing to tune, nothing to drift."""
        import random
        rng = random.Random(4)
        for _ in range(200):
            k = Kinetic()
            t, px = 1000.0, 6800.0
            for _i in range(rng.randint(1, 40)):
                t += 0.5
                px += rng.choice([-0.25, 0.0, 0.25])
                k.on_trade(t, px, rng.randint(1, 500),
                           rng.choice("BAN"))
            assert -1.0 <= k.push_ratio <= 1.0

    def test_an_empty_window_is_zero_not_undefined(self):
        assert Kinetic().push_ratio == 0.0

    def test_one_bounded_number_still_cannot_carry_absorption(self):
        """The ratio answers "who crossed the spread" and nothing
        else. Buying that works and buying that is absorbed are the
        SAME ratio -- which is why efficiency exists."""
        works = feed(Kinetic(), "B", 30, +0.25)
        absorbed = feed(Kinetic(), "B", 30, -0.25)
        assert works.push_ratio == pytest.approx(absorbed.push_ratio)
        assert works.efficiency > 0 > absorbed.efficiency

    # 3. SYNCED TO THE CANDLE -----------------------------------------

    def test_aligned_clears_at_the_boundary(self):
        """"resets cleanly every 5 minutes to synchronize perfectly
        with your candle charts"."""
        k = Kinetic(period_s=300.0, aligned=True)
        for t in (1500.0, 1600.0, 1700.0, 1790.0):
            k.on_trade(t, 6800.0, 10.0, "B")
        assert k.to_dict()["volume"] == 40.0
        k.on_trade(1801.0, 6800.0, 10.0, "B")
        assert k.to_dict()["volume"] == 10.0

    def test_the_print_that_crosses_belongs_to_the_NEW_candle(self):
        """It was being appended to the old bar and then cleared
        with it, so the first print of every candle vanished."""
        k = Kinetic(period_s=300.0, aligned=True)
        k.on_trade(1790.0, 6800.0, 10.0, "B")
        k.on_trade(1801.0, 6800.0, 7.0, "B")
        assert k.to_dict()["volume"] == 7.0, "the crossing print was lost"

    def test_rolling_does_not_clear_at_a_boundary(self):
        k = Kinetic(period_s=300.0, aligned=False)
        for t in (1790.0, 1801.0):
            k.on_trade(t, 6800.0, 10.0, "B")
        assert k.to_dict()["volume"] == 20.0

    def test_both_run_so_they_can_be_compared(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        assert "Kinetic(period_s=300.0,\n                                                   aligned=True)" in src \
            or "aligned=True" in src
        assert '"kinetic_bar": _kin_bar,' in src

    def test_the_tape_feeds_both(self):
        src = (ROOT / "liqmap" / "cme_tape.py").read_text()
        assert "self.kinetic_bar.on_trade(_ts, _rpx, _rsz, _sd)" in src

    def test_the_panel_shows_the_candle_reading_beside_the_rolling(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("function kinBlock(")
        blk = src[i:src.index("function paintFusion(", i)]
        assert "this 5m candle" in blk
        assert "push_ratio" in blk

    def test_the_mode_is_reported(self):
        assert Kinetic(aligned=True).to_dict()["aligned"] is True
        assert Kinetic().to_dict()["aligned"] is False
