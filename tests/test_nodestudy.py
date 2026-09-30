"""The node event study, against paths built to produce a known outcome.

Two things here would quietly ruin the catalog. First, one approach that
hovers near a node generating fifty touches, which turns a frequency table
into a measure of how long price loitered. Second, a wick through a node
counting as acceptance, which inverts the reject/break split on exactly the
bars where the distinction matters.
"""

from __future__ import annotations

import pytest

from liqmap import nodestudy as ns


class Bar:
    __slots__ = ("ts", "open", "high", "low", "close", "volume")

    def __init__(self, ts, o, h, l, c, v=100.0):
        self.ts, self.open, self.high = ts, o, h
        self.low, self.close, self.volume = l, c, v


def walk(prices, step=5.0, spread=0.25):
    """Bars whose close follows `prices`, with a small range either side."""
    return [Bar(i * step, p, p + spread, p - spread, p) for i, p in
            enumerate(prices)]


NODE = [(7000.0, "HVN", "strong")]


# ------------------------------------------------------------- touches

def test_a_node_fires_when_price_arrives():
    px = [6990.0, 6994.0, 6998.0, 7000.0, 7002.0]
    t = ns.find_touches(walk(px), NODE)
    assert len(t) == 1
    assert t[0].node_price == 7000.0


def test_the_approach_side_is_taken_from_where_price_armed():
    """A bar straddling the node has no direction of its own. Reading one
    off it would label half the sample at random."""
    up = ns.find_touches(walk([6985.0, 6992.0, 7000.0]), NODE)
    down = ns.find_touches(walk([7015.0, 7008.0, 7000.0]), NODE)
    assert up[0].from_below is True and up[0].approach == "from below"
    assert down[0].from_below is False


def test_hovering_at_a_node_does_not_fire_fifty_times():
    """The re-arm rule. Without it the table counts loitering, not
    arrivals."""
    px = [6985.0] + [7000.0 + (i % 3) * 0.25 for i in range(50)]
    assert len(ns.find_touches(walk(px), NODE)) == 1


def test_leaving_and_coming_back_fires_again():
    px = [6985.0, 7000.0, 6985.0, 7000.0]
    assert len(ns.find_touches(walk(px), NODE)) == 2


def test_price_that_never_reaches_the_node_fires_nothing():
    assert ns.find_touches(walk([6980.0, 6985.0, 6990.0]), NODE) == []


def test_several_nodes_are_tracked_independently():
    nodes = [(7000.0, "HVN", "strong"), (7020.0, "LVN", "weak")]
    px = [6985.0, 7000.0, 7010.0, 7020.0]
    kinds = {t.node_kind for t in ns.find_touches(walk(px), nodes)}
    assert kinds == {"HVN", "LVN"}


def test_context_is_carried_onto_every_touch():
    t = ns.find_touches(walk([6985.0, 7000.0]), NODE,
                        context={"regime": "trend", "vol": "high",
                                 "session": "rth", "trend_1h": "up"})
    assert t[0].regime == "trend" and t[0].vol == "high"


# ------------------------------------------------------------ outcomes

def _one(px):
    bars = walk(px)
    t = ns.find_touches(bars, NODE)[0]
    return bars, t, ns.resolve(bars, t)


def test_turning_away_without_closing_through_is_a_rejection():
    _, _, o = _one([6985.0, 7000.0, 6998.0, 6996.0, 6994.0])
    assert o.label == "reject"
    assert o.resolved and o.dist_ticks >= 8.0


def test_closing_through_and_running_is_a_break():
    _, _, o = _one([6985.0, 7000.0, 7001.0, 7002.0, 7003.0])
    assert o.label == "break_go"
    assert o.dist_ticks >= 8.0


def test_closing_through_then_closing_back_is_a_failed_break():
    _, _, o = _one([6985.0, 7000.0, 7001.0, 6999.0])
    assert o.label == "break_fail"


def test_going_nowhere_is_a_stall():
    _, _, o = _one([6985.0, 7000.0] + [7000.0 + (i % 2) * 0.25
                                       for i in range(20)])
    assert o.label == "stall"
    assert not o.resolved


def test_a_wick_through_is_not_acceptance():
    """A bar that trades above the node but closes back under it is a test.
    Counting it as a break inverts the split on exactly the bars where the
    distinction earns its keep."""
    bars = walk([6985.0, 7000.0])
    bars.append(Bar(ts=100.0, o=7000.0, h=7004.0, l=6999.0, c=6999.0))
    bars += [Bar(ts=105.0 + i * 5, o=6997.0 - i, h=6997.0 - i,
                 l=6996.0 - i, c=6996.5 - i) for i in range(6)]
    t = ns.find_touches(bars, NODE)[0]
    assert ns.resolve(bars, t).label == "reject"


def test_a_one_tick_wobble_across_the_node_is_not_a_failed_break():
    """The bug the output exposed.

    Arming on ANY close beyond made 75-88% of every real cell come back
    'break_fail', with a median distance of 2 ticks and a median time of
    zero minutes. That is price jittering one tick over the node and back
    -- ordinary chop at a level, relabelled as an event. A trader would not
    call it anything at all.
    """
    px = [6985.0, 7000.0, 7000.25, 6999.75, 7000.25, 6999.75]
    bars = walk(px)
    t = ns.find_touches(bars, NODE)[0]
    assert ns.resolve(bars, t).label != "break_fail"


def test_a_real_break_and_return_is_still_a_failed_break():
    """The fix must not swallow the event it is protecting.

    The move beyond has to clear THROUGH_TICKS without reaching
    BREAK_TICKS -- otherwise break_go resolves first and there is nothing
    left to fail.
    """
    px = [6985.0, 7000.0, 7001.0, 7001.0, 6998.0]
    bars = walk(px)
    t = ns.find_touches(bars, NODE)[0]
    assert ns.resolve(bars, t).label == "break_fail"


def test_the_through_margin_is_smaller_than_the_break_distance():
    """Through is 'past it at all'; break_go is 'and then ran'. If the
    margin were the larger, nothing could ever be a failed break."""
    assert 0 < ns.THROUGH_TICKS < ns.BREAK_TICKS


def test_a_short_approach_reads_the_mirror_image():
    px = [7015.0, 7000.0, 7002.0, 7004.0, 7006.0]
    bars = walk(px)
    t = ns.find_touches(bars, NODE)[0]
    o = ns.resolve(bars, t)
    assert t.from_below is False
    assert o.label == "reject"          # moved back UP, against the approach


def test_the_excursion_covers_the_whole_horizon_not_just_the_trigger():
    """The bug the two-year run exposed.

    The first version returned the moment it classified, so MFE was
    whatever the resolving bar reached -- which sits at the threshold by
    construction. Every break_go in two years came back "MFE 10t" against
    an 8-tick trigger, and the column a reader looks at for "how far does
    this usually go" was reporting my own rule.
    """
    bars = walk([6985.0, 7000.0, 7001.0, 7002.5, 7005.0, 7008.0, 7010.0])
    t = ns.find_touches(bars, NODE)[0]
    o = ns.resolve(bars, t)
    assert o.label == "break_go"
    assert o.mfe_ticks > o.dist_ticks * 2


def test_the_excursion_stops_when_price_returns_to_the_node():
    """The second wrong definition, and why this is the third.

    Running the excursions over the whole hour made MFE and MAE both come
    back 28-54 ticks for EVERY outcome in two years of ES -- including
    rejects -- because that is simply how far the market ranges in an
    hour. A reject with 42 ticks of favourable excursion is not a reject;
    it is a measurement of the clock.

    Here the leg runs to +40, comes back to the node, and only then rips
    to +200. The catalog must report the leg.
    """
    bars = walk([6985.0, 7000.0, 7002.0, 7006.0, 7010.0,
                 7002.0, 7000.0, 7020.0, 7050.0])
    t = ns.find_touches(bars, NODE)[0]
    o = ns.resolve(bars, t)
    assert o.returned
    assert o.mfe_ticks < 60          # the post-return rip is a new leg


def test_a_reject_shows_more_adverse_than_favourable_excursion():
    """The sanity check the hour-long version could not pass."""
    bars = walk([6985.0, 7000.0, 6996.0, 6992.0, 6988.0, 6984.0])
    t = ns.find_touches(bars, NODE)[0]
    o = ns.resolve(bars, t)
    assert o.label == "reject"
    assert o.mae_ticks > o.mfe_ticks * 5


def test_the_label_is_still_set_at_the_first_barrier_touched():
    """Walking on for the excursions must not let a later move relabel it.
    Up 8 then down 20 is a break that went wrong, not a rejection."""
    bars = walk([6985.0, 7000.0, 7002.0, 7003.0, 6995.0, 6990.0])
    t = ns.find_touches(bars, NODE)[0]
    assert ns.resolve(bars, t).label == "break_go"


def test_excursions_are_recorded_in_both_directions():
    _, _, o = _one([6985.0, 7000.0, 7003.0, 6996.0, 7005.0])
    assert o.mfe_ticks > 0 and o.mae_ticks > 0


def test_the_horizon_bounds_how_far_forward_it_looks():
    px = [6985.0, 7000.0] + [7000.0] * 50 + [7050.0]
    bars = walk(px, step=120.0)          # two minutes a bar
    t = ns.find_touches(bars, NODE)[0]
    o = ns.resolve(bars, t, horizon_s=600.0)
    assert o.label == "stall"            # the run happens long after


def test_returning_to_the_node_is_flagged_separately_from_the_label():
    px = [6985.0, 7000.0, 7008.0, 7014.0, 7006.0, 7000.0]
    bars = walk(px)
    t = ns.find_touches(bars, NODE)[0]
    o = ns.resolve(bars, t)
    assert o.returned is True


def test_never_leaving_is_not_a_return():
    _, _, o = _one([6985.0, 7000.0, 7000.25, 7000.0])
    assert o.returned is False


# ------------------------------------------------------------- catalog

def _pairs():
    out = []
    for label, px in (
            ("reject", [6985.0, 7000.0, 6997.0, 6994.0, 6991.0]),
            ("break_go", [6985.0, 7000.0, 7002.0, 7004.0, 7006.0]),
            ("break_fail", [6985.0, 7000.0, 7001.0, 6999.0])):
        for _ in range(12):
            bars = walk(px)
            t = ns.find_touches(bars, NODE)[0]
            out.append((t, ns.resolve(bars, t)))
    return out


def test_frequencies_sum_to_one_within_a_condition():
    cells = ns.catalog(_pairs())
    for key in {c.key for c in cells}:
        total = sum(c.freq for c in cells if c.key == key)
        assert total == pytest.approx(1.0)


def test_every_label_is_emitted_even_at_zero():
    """An absent row reads as 'not measured' when it means 'never
    happened', and those are different facts."""
    cells = ns.catalog(_pairs())
    for key in {c.key for c in cells}:
        labels = {c.label for c in cells if c.key == key}
        assert labels == set(ns.LABELS)


def test_the_sample_size_is_on_every_cell():
    for c in ns.catalog(_pairs()):
        assert c.n_condition > 0
        assert "n" in c.to_dict() and "n_condition" in c.to_dict()


def test_a_thin_cell_is_flagged_as_thin():
    few = _pairs()[:6]
    assert all(c.thin for c in ns.catalog(few))


def test_conditioning_can_be_split_further():
    pairs = _pairs()
    wide = ns.catalog(pairs, by=("node_kind",))
    narrow = ns.catalog(pairs, by=("node_kind", "approach", "regime"))
    assert len({c.key for c in narrow}) >= len({c.key for c in wide})


def test_an_empty_catalog_is_empty_not_an_error():
    assert ns.catalog([]) == []


# --------------------------------------------------------- histograms

def test_the_histogram_counts_every_value():
    vals = [1.0, 2.0, 3.0, 9.0, 11.0]
    h = ns.histogram(vals, 0.0, 12.0, buckets=6)
    assert sum(c for _, _, c in h) == len(vals)


def test_values_above_the_top_land_in_the_last_bucket():
    h = ns.histogram([100.0], 0.0, 10.0, buckets=5)
    assert h[-1][2] == 1


def test_a_degenerate_range_gives_no_buckets():
    assert ns.histogram([1.0], 5.0, 5.0) == []


# ------------------------------------------------------- the conventions

def test_the_conventions_are_the_stated_ones():
    assert ns.TOUCH_TICKS == 2.0
    assert ns.REARM_TICKS == 10.0
    assert ns.HORIZON_S == 3600.0
    assert ns.REJECT_TICKS == 8.0
    assert ns.BREAK_TICKS == 8.0


def test_the_labels_partition_the_space():
    assert set(ns.LABELS) == {"reject", "break_go", "break_fail", "stall"}
