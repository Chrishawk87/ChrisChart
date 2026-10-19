"""Footprint candles and the diagonal imbalance test.

The arithmetic here is checked against hand-worked examples, because
the dangerous failures in a footprint engine are all silent:

  the aggressor convention reversed -- every flag inverts, and the
  output still looks entirely reasonable

  a level compared against the wrong neighbour -- a diagonal that
  skips an untraded price is comparing across two ticks, not one

  the extremes of the bar flagged from a missing neighbour that is an
  artifact of where the bar closed
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from liqmap.footprint import (                             # noqa: E402
    BUY_IMBALANCE, SELL_IMBALANCE, AnyRule, DeltaRule, FootprintCandle,
    FootprintEngine, ImbalanceConfig, PriceLevel, Tick, TickScale,
    TimeRule, VolumeRule, flip_aggressor)

T0 = 1_700_000_000.0
TS = 0.25


def eng(rule=None, **cfg):
    return FootprintEngine(
        tick_size=TS, rule=rule or TimeRule(60.0),
        imbalance=ImbalanceConfig(**cfg) if cfg else ImbalanceConfig())


def bar_from(levels, lo=23200):
    """A finished bar with exact (bid, ask) volumes per level.

    `levels` is bottom-up: index 0 is the lowest price.
    """
    c = FootprintCandle(ts_open=T0)
    for i, (bid, ask) in enumerate(levels):
        p = lo + i
        if bid:
            c.add(p, bid, "sell", T0)
        if ask:
            c.add(p, ask, "buy", T0)
    return c


# ------------------------------------------------------- the convention

class TestAggressorConvention:
    """A trade AT THE BID was taken by an aggressive seller; one AT
    THE ASK by an aggressive buyer. Reversed, every imbalance in the
    system inverts and nothing raises."""

    def test_a_buy_lands_in_ask_volume(self):
        lv = PriceLevel()
        lv.add(10, "buy")
        assert lv.ask_volume == 10 and lv.bid_volume == 0

    def test_a_sell_lands_in_bid_volume(self):
        lv = PriceLevel()
        lv.add(10, "sell")
        assert lv.bid_volume == 10 and lv.ask_volume == 0

    def test_delta_is_buyers_minus_sellers(self):
        lv = PriceLevel(bid_volume=30, ask_volume=100)
        assert lv.delta == 70

    def test_an_unknown_aggressor_is_refused_at_the_door(self):
        with pytest.raises(ValueError, match="aggressor"):
            Tick(ts=T0, price=5800.0, size=1, aggressor="BUY")

    def test_a_zero_size_print_is_refused(self):
        """It is not an observation about anything, and it would put
        an empty level in the matrix."""
        with pytest.raises(ValueError, match="size"):
            Tick(ts=T0, price=5800.0, size=0, aggressor="buy")

    def test_flipping_is_available_in_one_place(self):
        """A feed with the opposite convention is corrected here, not
        by reversing the comparison in the imbalance loop."""
        t = Tick(ts=T0, price=5800.0, size=5, aggressor="buy")
        assert flip_aggressor(t).aggressor == "sell"
        assert flip_aggressor(t).size == 5


class TestTickScale:

    def test_a_price_becomes_an_exact_integer_index(self):
        s = TickScale(0.25)
        assert s.to_ticks(5800.00) == 23200
        assert s.to_ticks(5800.25) == 23201

    def test_it_round_trips(self):
        s = TickScale(0.25)
        for px in (5800.0, 5800.25, 5799.75, 1.25):
            assert s.to_price(s.to_ticks(px)) == px

    def test_an_off_grid_price_is_snapped_not_dropped(self):
        """Block trades and spread legs print off-tick. Dropping them
        would silently lose volume from the candle."""
        assert TickScale(0.25).to_ticks(5800.13) == 23201

    def test_a_zero_tick_size_is_refused(self):
        with pytest.raises(ValueError):
            TickScale(0.0)


# -------------------------------------------------------- the diagonal

class TestBuyingImbalance:
    """Ask(P) >= multiplier * Bid(P - 1)."""

    def test_four_times_exactly_is_an_imbalance(self):
        """`>=`, not `>`. 100 against 25 at 4.0 qualifies."""
        c = bar_from([(25, 0), (0, 100), (5, 5)])
        flags = [f for f in c.imbalances() if f.side == BUY_IMBALANCE]
        assert any(f.price_ticks == 23201 for f in flags)

    def test_just_under_four_times_is_not(self):
        c = bar_from([(26, 0), (0, 100), (5, 5)])
        flags = [f for f in c.imbalances() if f.side == BUY_IMBALANCE
                 and f.price_ticks == 23201]
        assert not flags

    def test_it_compares_against_the_level_BELOW(self):
        """The diagonal, not the same row.

        Its own row holds 20 bids against 100 asks, which WOULD pass
        at 4x. The level below holds 40, which blocks it. A loop
        comparing a level with itself flags this; the diagonal does
        not.
        """
        c = bar_from([(40, 1), (20, 100), (1, 1)])
        flags = [f for f in c.imbalances() if f.side == BUY_IMBALANCE
                 and f.price_ticks == 23201]
        assert not flags, "compared against its own row, not the diagonal"

    def test_it_compares_against_BID_volume_below(self):
        """Bid volume at P-1, not ask volume: those are the orders
        that could have traded with each other and did not."""
        c = bar_from([(0, 500), (0, 100), (1, 1)])
        flags = [f for f in c.imbalances() if f.side == BUY_IMBALANCE
                 and f.price_ticks == 23201]
        assert flags, "500 asks below should not block it; 0 bids did"

    def test_the_ratio_is_reported(self):
        c = bar_from([(10, 0), (0, 100), (5, 5)])
        f = next(f for f in c.imbalances()
                 if f.side == BUY_IMBALANCE and f.price_ticks == 23201)
        assert f.volume == 100 and f.against == 10
        assert f.ratio == pytest.approx(10.0)


class TestSellingImbalance:
    """Bid(P) >= multiplier * Ask(P + 1)."""

    def test_four_times_exactly_is_an_imbalance(self):
        c = bar_from([(5, 5), (100, 0), (0, 25)])
        flags = [f for f in c.imbalances() if f.side == SELL_IMBALANCE]
        assert any(f.price_ticks == 23201 for f in flags)

    def test_just_under_is_not(self):
        c = bar_from([(5, 5), (100, 0), (0, 26)])
        flags = [f for f in c.imbalances() if f.side == SELL_IMBALANCE
                 and f.price_ticks == 23201]
        assert not flags

    def test_it_compares_against_the_level_ABOVE(self):
        c = bar_from([(1, 1), (100, 0), (0, 25)])
        f = next(f for f in c.imbalances()
                 if f.side == SELL_IMBALANCE and f.price_ticks == 23201)
        assert f.against == 25, "took the wrong neighbour"

    def test_the_two_sides_are_not_mirror_images(self):
        """A bar can carry both, at different prices. A test that only
        ever sees one side would pass with the loop half wired."""
        c = bar_from([(25, 0), (0, 100), (0, 0), (100, 0), (0, 25)])
        sides = {f.side for f in c.imbalances()}
        assert sides == {BUY_IMBALANCE, SELL_IMBALANCE}


class TestTheEmptyDiagonal:
    """The test is `ask >= multiplier * bid`. With bid == 0 the right
    side is zero and ANY volume passes -- a one-lot is tagged the same
    as four hundred. This is the setting that decides whether the
    output is worth reading."""

    def test_by_default_an_empty_diagonal_counts(self):
        # The level below EXISTS (it has asks) but holds no bids, so
        # the diagonal is empty. A level of (0, 0) is never created at
        # all, which would make this the bar's low and skip it as an
        # edge -- testing nothing.
        c = bar_from([(0, 5), (0, 1), (5, 5)])
        flags = [f for f in c.imbalances() if f.side == BUY_IMBALANCE
                 and f.price_ticks == 23201]
        assert flags and flags[0].ratio is None, "inf, not a number"

    def test_a_floor_removes_the_one_lot(self):
        c = bar_from([(0, 5), (0, 1), (5, 5)])
        cfg = ImbalanceConfig(min_volume=10)
        assert not [f for f in c.imbalances(cfg)
                    if f.price_ticks == 23201 and f.side == BUY_IMBALANCE]

    def test_the_floor_keeps_the_large_one(self):
        c = bar_from([(0, 5), (0, 400), (5, 5)])
        cfg = ImbalanceConfig(min_volume=10)
        assert [f for f in c.imbalances(cfg)
                if f.price_ticks == 23201 and f.side == BUY_IMBALANCE]

    def test_empty_diagonals_can_be_refused_outright(self):
        c = bar_from([(0, 5), (0, 400), (5, 5)])
        cfg = ImbalanceConfig(flag_empty_diagonal=False)
        assert not [f for f in c.imbalances(cfg)
                    if f.price_ticks == 23201 and f.side == BUY_IMBALANCE]

    def test_a_level_with_no_volume_is_never_flagged(self):
        """`0 >= 4 * 0` is true. A level that traded nothing must not
        be tagged as an imbalance of nothing -- and `bar_from` does
        not even create one, so the matrix stays free of empty rows."""
        c = bar_from([(0, 5), (0, 0), (5, 5)])
        assert 23201 not in c.levels
        assert not [f for f in c.imbalances() if f.price_ticks == 23201]

    def test_there_is_no_division_in_the_test(self):
        """A multiplication has no zero case to guard and is exact on
        integers -- no level sits on the threshold by rounding."""
        src = (ROOT / "liqmap" / "footprint.py").read_text()
        body = src.split("def imbalances")[1].split("def stacked")[0]
        assert ">= mult * ref" in body
        assert "/ ref" in body, "only for reporting the ratio"


class TestTheEdgesOfTheBar:
    """The highest level's P+1 and the lowest's P-1 fall OUTSIDE the
    candle. A flag there is an artifact of where the bar closed."""

    def test_the_lowest_level_is_not_flagged_as_buying(self):
        c = bar_from([(0, 100), (5, 5), (5, 5)])
        assert not [f for f in c.imbalances()
                    if f.price_ticks == 23200 and f.side == BUY_IMBALANCE]

    def test_the_highest_level_is_not_flagged_as_selling(self):
        c = bar_from([(5, 5), (5, 5), (100, 0)])
        assert not [f for f in c.imbalances()
                    if f.price_ticks == 23202 and f.side == SELL_IMBALANCE]

    def test_zero_policy_flags_them_and_says_so(self):
        c = bar_from([(0, 100), (5, 5), (5, 5)])
        cfg = ImbalanceConfig(edge_policy="zero")
        assert [f for f in c.imbalances(cfg)
                if f.price_ticks == 23200 and f.side == BUY_IMBALANCE]

    def test_a_single_level_bar_raises_nothing(self):
        c = bar_from([(50, 50)])
        assert c.imbalances() == []

    def test_an_empty_bar_raises_nothing(self):
        assert FootprintCandle(ts_open=T0).imbalances() == []

    def test_an_untraded_price_inside_the_range_is_a_real_gap(self):
        """The diagonal is ONE tick, looked up by absolute price.

        23201 never traded. The 25 bids sit at 23200, TWO ticks below
        23202, and must not be what 23202 is measured against -- that
        would silently turn a diagonal into a two-tick comparison
        wherever the book thinned out.
        """
        c = bar_from([(25, 0), (0, 0), (0, 100)])
        assert 23201 not in c.levels, "the gap is real"
        flags = [f for f in c.imbalances()
                 if f.price_ticks == 23202 and f.side == BUY_IMBALANCE]
        assert flags and flags[0].against == 0, \
            "it reached past the gap to the 25 bids two ticks down"

    def test_a_gap_empties_the_diagonal_on_BOTH_sides_of_it(self):
        """An untraded price is not neutral. The level above it loses
        its bid reference and the level below loses its ask one, so a
        gap creates an empty diagonal in each direction.

        Adjacent, 23201's 100 asks are measured against 23200's 25
        bids. Pushed one tick apart, each level faces nothing.
        """
        packed = bar_from([(25, 0), (0, 100)])
        gapped = bar_from([(25, 0), (0, 0), (0, 100)])
        assert [(f.side, f.against) for f in packed.imbalances()] == [
            (BUY_IMBALANCE, 25)]
        assert sorted((f.side, f.price_ticks, f.against)
                      for f in gapped.imbalances()) == [
            (BUY_IMBALANCE, 23202, 0), (SELL_IMBALANCE, 23200, 0)]

    def test_that_pair_is_identical_however_the_levels_are_walked(self):
        """The diagonal is `levels[p - 1]`, an absolute price, not the
        previous row of the iteration -- so density cannot matter."""
        c = bar_from([(25, 0), (0, 0), (0, 100)])
        by_key = sorted((f.side, f.price_ticks, f.against)
                        for f in c.imbalances())
        # the same levels, inserted in the opposite order
        d = FootprintCandle(ts_open=T0)
        d.add(23202, 100, "buy", T0)
        d.add(23200, 25, "sell", T0)
        assert sorted((f.side, f.price_ticks, f.against)
                      for f in d.imbalances()) == by_key


class TestStacked:

    def test_three_in_a_row_is_a_run(self):
        c = bar_from([(0, 5), (0, 50), (0, 50), (0, 50), (5, 5)])
        runs = c.stacked(run=3)
        assert any(s == BUY_IMBALANCE for s, _, _ in runs)

    def test_two_in_a_row_is_not(self):
        c = bar_from([(0, 5), (0, 50), (0, 50), (90, 0), (5, 5)])
        assert not [r for r in c.stacked(run=3) if r[0] == BUY_IMBALANCE]

    def test_the_run_length_is_the_callers(self):
        c = bar_from([(0, 5), (0, 50), (0, 50), (90, 0), (5, 5)])
        assert [r for r in c.stacked(run=2) if r[0] == BUY_IMBALANCE]


# ----------------------------------------------------------- the bars

class TestTimeRule:

    def test_bars_sit_on_a_grid_not_on_the_first_print(self):
        """Anchored to the first tick, two services watching the same
        market disagree about where every bar begins."""
        e = eng(TimeRule(60.0))
        e.on_tick(Tick(ts=T0 + 37, price=5800.0, size=1, aggressor="buy"))
        b = e.current.bucket
        assert b % 60 == 0, "not on the grid"
        assert b <= T0 + 37 < b + 60, "wrong slot"
        assert b != T0 + 37, "anchored to the first print"

    def test_two_prints_in_one_interval_share_a_bar(self):
        e = eng(TimeRule(60.0))
        e.on_tick(Tick(ts=T0 + 5, price=5800.0, size=1, aggressor="buy"))
        # +5 and +20 sit in the same 60s slot; +50 would NOT, because
        # T0 itself is not on a minute boundary.
        assert e.on_tick(Tick(ts=T0 + 20, price=5800.0, size=1,
                              aggressor="buy")) == []

    def test_crossing_the_boundary_closes_the_bar(self):
        e = eng(TimeRule(60.0))
        e.on_tick(Tick(ts=T0 + 10, price=5800.0, size=1, aggressor="buy"))
        done = e.on_tick(Tick(ts=T0 + 61, price=5800.0, size=1,
                              aggressor="buy"))
        assert len(done) == 1 and done[0].closed

    def test_a_quiet_interval_still_closes_its_bar(self):
        """Otherwise the next print appears to belong to it."""
        e = eng(TimeRule(60.0))
        e.on_tick(Tick(ts=T0 + 5, price=5800.0, size=1, aggressor="buy"))
        assert len(e.roll(T0 + 120)) == 1

    def test_rolling_an_empty_bar_produces_nothing(self):
        e = eng(TimeRule(60.0))
        assert e.roll(T0 + 120) == []


class TestVolumeRule:
    """"Closes after exactly 10,000 contracts" has to mean exactly, or
    bars stop being comparable -- which is the only reason to count
    volume instead of time."""

    def test_it_closes_on_the_exact_count(self):
        e = eng(VolumeRule(100))
        done = []
        for _ in range(10):
            done += e.on_tick(Tick(ts=T0, price=5800.0, size=10,
                                   aggressor="buy"))
        assert len(done) == 1
        assert done[0].total_volume == 100

    def test_a_print_that_overshoots_is_split(self):
        e = eng(VolumeRule(100))
        done = e.on_tick(Tick(ts=T0, price=5800.0, size=150,
                              aggressor="buy"))
        assert len(done) == 1
        assert done[0].total_volume == 100, "overshot its own threshold"
        assert e.current.total_volume == 50, "remainder opens the next"

    def test_one_huge_print_can_finish_several_bars(self):
        e = eng(VolumeRule(100))
        done = e.on_tick(Tick(ts=T0, price=5800.0, size=250,
                              aggressor="buy"))
        assert len(done) == 2
        assert all(b.total_volume == 100 for b in done)
        assert e.current.total_volume == 50

    def test_no_contract_is_lost_or_duplicated_in_a_split(self):
        e = eng(VolumeRule(100))
        done = e.on_tick(Tick(ts=T0, price=5800.0, size=250,
                              aggressor="buy"))
        held = sum(b.total_volume for b in done) + e.current.total_volume
        assert held == 250

    def test_the_split_keeps_the_aggressor(self):
        e = eng(VolumeRule(100))
        done = e.on_tick(Tick(ts=T0, price=5800.0, size=150,
                              aggressor="sell"))
        assert done[0].bid_volume == 100 and done[0].ask_volume == 0
        assert e.current.bid_volume == 50


class TestDeltaRule:

    def test_it_closes_when_net_buying_reaches_the_threshold(self):
        e = eng(DeltaRule(50))
        done = []
        for _ in range(6):
            done += e.on_tick(Tick(ts=T0, price=5800.0, size=10,
                                   aggressor="buy"))
        assert done and done[0].delta >= 50

    def test_net_selling_counts_too(self):
        e = eng(DeltaRule(50))
        done = []
        for _ in range(6):
            done += e.on_tick(Tick(ts=T0, price=5800.0, size=10,
                                   aggressor="sell"))
        assert done and done[0].delta <= -50

    def test_balanced_flow_never_closes_it(self):
        e = eng(DeltaRule(50))
        done = []
        for i in range(40):
            done += e.on_tick(Tick(
                ts=T0, price=5800.0, size=10,
                aggressor="buy" if i % 2 else "sell"))
        assert not done, "delta stayed near zero"

    def test_the_crossing_print_is_not_split(self):
        """Delta is a running difference; the crossing print is the
        event the bar is about. Halving it would put part of one
        order in a bar that closed before it happened."""
        e = eng(DeltaRule(50))
        done = e.on_tick(Tick(ts=T0, price=5800.0, size=500,
                              aggressor="buy"))
        assert len(done) == 1 and done[0].delta == 500


class TestComposedRules:

    def test_either_condition_can_end_the_bar(self):
        e = eng(AnyRule(TimeRule(60.0), VolumeRule(100)))
        done = e.on_tick(Tick(ts=T0, price=5800.0, size=100,
                              aggressor="buy"))
        assert len(done) == 1

    def test_the_bar_records_which_rule_closed_it(self):
        """A mixed series whose bars do not say why they closed cannot
        be analysed."""
        e = eng(AnyRule(TimeRule(60.0), VolumeRule(100)))
        done = e.on_tick(Tick(ts=T0, price=5800.0, size=100,
                              aggressor="buy"))
        assert done[0].close_reason == "volume"

    def test_the_time_half_still_works(self):
        e = eng(AnyRule(TimeRule(60.0), VolumeRule(10_000)))
        e.on_tick(Tick(ts=T0 + 5, price=5800.0, size=1, aggressor="buy"))
        done = e.on_tick(Tick(ts=T0 + 65, price=5800.0, size=1,
                              aggressor="buy"))
        assert len(done) == 1

    def test_a_threshold_below_one_contract_is_refused(self):
        with pytest.raises(ValueError):
            VolumeRule(0)


class TestTheCandleItself:

    def test_ohlc_comes_from_the_prints(self):
        e = eng(VolumeRule(40))
        for px in (5800.00, 5800.75, 5799.50, 5800.25):
            e.on_tick(Tick(ts=T0, price=px, size=10, aggressor="buy"))
        bar = e.history[-1]
        s = e.scale
        assert s.to_price(bar.open_ticks) == 5800.00
        assert s.to_price(bar.high_ticks) == 5800.75
        assert s.to_price(bar.low_ticks) == 5799.50
        assert s.to_price(bar.close_ticks) == 5800.25

    def test_open_is_the_first_print_not_the_lowest(self):
        e = eng(VolumeRule(30))
        for px in (5800.50, 5799.00, 5801.00):
            e.on_tick(Tick(ts=T0, price=px, size=10, aggressor="buy"))
        assert e.scale.to_price(e.history[-1].open_ticks) == 5800.50

    def test_total_volume_is_the_sum_of_the_matrix(self):
        e = eng(VolumeRule(1000))
        for i in range(20):
            e.on_tick(Tick(ts=T0, price=5800.0 + i * TS, size=7,
                           aggressor="buy" if i % 2 else "sell"))
        c = e.current
        assert c.total_volume == sum(lv.total for lv in c.levels.values())
        assert c.total_volume == 140

    def test_rows_read_high_to_low(self):
        c = bar_from([(1, 1), (1, 1), (1, 1)])
        assert [p for p, _ in c.rows()] == [23202, 23201, 23200]

    def test_the_point_of_control_is_the_busiest_price(self):
        c = bar_from([(5, 5), (60, 60), (5, 5)])
        assert c.point_of_control() == 23201


class TestItDecidesNothing:

    def test_no_order_code(self):
        src = (ROOT / "liqmap" / "footprint.py").read_text()
        for word in ("def submit", "def execute", "def place", "def order"):
            assert word not in src, word

    def test_the_multiplier_is_an_argument_not_a_constant(self):
        c = bar_from([(10, 0), (0, 50), (5, 5)])
        assert [f for f in c.imbalances(ImbalanceConfig(multiplier=4.0))
                if f.price_ticks == 23201]
        assert not [f for f in c.imbalances(ImbalanceConfig(multiplier=6.0))
                    if f.price_ticks == 23201 and f.side == BUY_IMBALANCE]


class TestTheOutputOrder:
    """The result does not depend on iteration order -- the diagonal
    is a key lookup. But the ORDER of the returned list is something
    callers rely on, and one that changes between versions quietly
    reshuffles anything taking "the first flag"."""

    def test_flags_come_back_low_to_high(self):
        c = bar_from([(0, 5), (0, 50), (0, 50), (0, 50), (5, 5)])
        prices = [f.price_ticks for f in c.imbalances()]
        assert prices == sorted(prices), prices

    def test_the_matrix_reads_high_to_low(self):
        """Opposite of the flag list, and deliberately so: that is how
        a footprint is read on a screen."""
        c = bar_from([(1, 1), (1, 1), (1, 1)])
        assert [p for p, _ in c.rows()] == sorted(c.levels, reverse=True)


class TestAComposedSeriesSaysWhyEachBarClosed:
    """"any" is the one answer a mixed series must not give: a bar
    that does not say whether it ended on time or on volume cannot be
    compared with its neighbours."""

    def test_the_volume_member_is_named(self):
        e = eng(AnyRule(TimeRule(3600.0), VolumeRule(100)))
        done = e.on_tick(Tick(ts=T0, price=5800.0, size=100,
                              aggressor="buy"))
        assert done[0].close_reason == "volume"

    def test_the_time_member_is_named_too(self):
        e = eng(AnyRule(TimeRule(60.0), VolumeRule(1_000_000)))
        e.on_tick(Tick(ts=T0 + 5, price=5800.0, size=1, aggressor="buy"))
        done = e.on_tick(Tick(ts=T0 + 300, price=5800.0, size=1,
                              aggressor="buy"))
        assert done[0].close_reason == "time", done[0].close_reason

    def test_a_single_rule_names_itself(self):
        e = eng(VolumeRule(50))
        done = e.on_tick(Tick(ts=T0, price=5800.0, size=50,
                              aggressor="buy"))
        assert done[0].close_reason == "volume"
