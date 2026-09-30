"""Derived features, and the trailing-window rule they all obey.

The central test is `test_no_statistic_can_see_the_future`. A z-score taken
over a window that includes its own bar, or a mean taken over the whole
file, leaks the answer -- and both are easy to write, neither crashes, and
both make a study say yes. So the fixture here is quiet for its whole first
half and violent afterwards: if any statistic on an early bar has seen the
violence, its value moves and the test fails.
"""

from __future__ import annotations

import pytest

from liqmap import features as ft
from liqmap.escondense import Slice


def sl(i, open_px=7000.0, close=None, high=None, low=None, volume=100.0,
       delta=0.0, buy=50.0, sell=50.0, move=1.0, trades_in=10,
       interval=60.0):
    close = open_px if close is None else close
    s = Slice(ts=1_789_999_800.0 + i * interval,
              bids=[(open_px - k * 0.25, 50) for k in range(10)],
              asks=[(open_px + 0.25 + k * 0.25, 50) for k in range(10)],
              delta=delta, buy_vol=buy, sell_vol=sell, trades_in=trades_in,
              prior_close=open_px,
              prior_open=open_px / (1.0 + move / 10_000.0))
    s.open = open_px
    s.close = close
    s.high = high if high is not None else max(open_px, close)
    s.low = low if low is not None else min(open_px, close)
    s.volume = volume
    s.path = [(0.0, 0.0)]
    return s


def quiet(n=80, px=7000.0):
    """Ordinary bars with some spread in volume and range.

    Volume has to VARY. A constant-volume run has zero standard deviation,
    so every z-score is correctly None and any test built on it measures
    the flat-window guard instead of the thing it meant to.
    """
    out = []
    for i in range(n):
        wobble = (i * 7919) % 23                 # deterministic, no rng
        out.append(sl(i, open_px=px + (i % 2) * 0.25,
                      close=px + (i % 2) * 0.25 + (wobble - 11) * 0.05,
                      volume=90.0 + wobble))
    return out


# ------------------------------------------------- THE look-ahead test

def test_no_statistic_can_see_the_future():
    """Quiet for eighty bars, then violent. The early bars must not know."""
    calm = quiet(80)
    loud = [sl(80 + i, open_px=7000.0 + i * 5, close=7000.0 + i * 5 + 4,
               high=7000.0 + i * 5 + 9, low=7000.0 + i * 5 - 9,
               volume=100_000.0) for i in range(40)]

    only_calm = ft.build(calm)
    both = ft.build(calm + loud)

    for a, b in zip(only_calm, both[:len(only_calm)]):
        assert a.vol_z == b.vol_z
        assert a.range_z == b.range_z
        assert a.anomaly == b.anomaly
        assert a.efficiency == b.efficiency


def test_a_bar_is_not_in_its_own_window():
    """One enormous bar in a quiet run must score as an outlier. If it were
    in its own window it would drag the mean toward itself and score far
    less extreme."""
    s = quiet(60) + [sl(60, volume=100_000.0)]
    f = ft.build(s)[-1]
    assert f.vol_z is not None and f.vol_z > 5


def test_nothing_is_reported_before_the_window_has_filled():
    f = ft.build(quiet(5))
    assert all(x.vol_z is None for x in f)
    assert all(not x.ready for x in f)


def test_a_flat_window_gives_no_z_score_rather_than_infinity():
    """Zero spread makes the score undefined, not enormous."""
    s = [sl(i, volume=100.0) for i in range(40)]
    assert ft.build(s)[-1].vol_z is None


# ------------------------------------------------------- absorption

def test_flow_that_moved_price_barely_at_all_reads_as_absorption():
    """The one real microstructure read aggregated depth supports: heavy
    one-sided flow and almost no displacement means somebody took the
    other side of all of it."""
    absorbed = sl(0, delta=90.0, buy=95.0, sell=5.0, move=0.05)
    ordinary = sl(0, delta=90.0, buy=95.0, sell=5.0, move=8.0)
    a = ft.build([absorbed])[0].absorption
    b = ft.build([ordinary])[0].absorption
    assert abs(a) > abs(b) * 5


def test_absorption_is_signed_by_the_flow():
    buys = ft.build([sl(0, delta=90.0, buy=95.0, sell=5.0, move=0.1)])[0]
    sells = ft.build([sl(0, delta=-90.0, buy=5.0, sell=95.0, move=0.1)])[0]
    assert buys.absorption > 0 > sells.absorption


def test_a_tiny_displacement_does_not_blow_the_ratio_up():
    """A floor on the denominator, not a guard against zero. Flow that
    moved price a thousandth of a basis point is absorption; dividing by
    that raw figure would report thousands instead of 'a lot'."""
    f = ft.build([sl(0, delta=50.0, buy=75.0, sell=25.0, move=0.0)])[0]
    assert f.absorption is not None and abs(f.absorption) < 100


def test_a_bar_with_no_flow_has_no_absorption_read():
    assert ft.build([sl(0, trades_in=0)])[0].absorption is None


# ---------------------------------------------------------- anomaly

def test_the_anomaly_needs_both_halves_to_be_unusual():
    """Unusual volume in a normal range is a fight; unusual volume in an
    unusual range is a move. Multiplying keeps them distinguishable."""
    base = quiet(60)
    fight = base + [sl(60, volume=100_000.0, close=7000.0, high=7000.1,
                       low=6999.9)]
    move = base + [sl(60, volume=100_000.0, close=7020.0, high=7021.0,
                      low=6999.0)]
    a = ft.build(fight)[-1]
    b = ft.build(move)[-1]
    assert a.anomaly is not None and b.anomaly is not None
    assert b.anomaly > a.anomaly


# ----------------------------------------------------------- regimes

def test_a_one_way_march_is_a_trend_and_a_chop_is_a_range():
    march = [sl(i, open_px=7000.0 + i, close=7000.0 + i + 1)
             for i in range(60)]
    chop = [sl(i, open_px=7000.0 + (i % 2), close=7000.0 + ((i + 1) % 2))
            for i in range(60)]
    assert ft.build(march)[-1].trending is True
    assert ft.build(chop)[-1].trending is False


def test_the_regime_label_carries_all_three_axes():
    label = ft.build(quiet(60))[-1].regime()
    parts = label.split("-")
    assert parts[0] in ("trend", "range")
    assert parts[1] in ("busy", "quiet")
    assert "-".join(parts[2:]) in ("at-level", "mid")


def test_bars_without_enough_history_are_labelled_unknown():
    assert ft.build(quiet(3))[0].regime() == "unknown"


def test_regimes_groups_every_bar_exactly_once():
    feats = ft.build(quiet(60))
    groups = ft.regimes(feats)
    seen = sorted(i for idx in groups.values() for i in idx)
    assert seen == list(range(len(feats)))


# ------------------------------------------------------ key levels

def test_the_level_distance_is_read_before_the_bar_joins_the_book():
    """Otherwise a bar can be measured against a level it just set, which
    would report a distance of zero on exactly the bars that matter."""
    day = 86_400.0
    s = [sl(i, open_px=7000.0 + (i % 3), interval=300.0) for i in range(200)]
    feats = ft.build(s)
    touched = [f for f in feats if f.level_ticks is not None]
    # Some bars find a level; none of them is at a suspiciously exact zero
    # for every single bar, which is what self-referencing would produce.
    if touched:
        assert not all(abs(f.level_ticks) < 1e-9 for f in touched)


def test_near_level_respects_the_tick_threshold():
    f = ft.Feat(ts=0.0, index=0, level_ticks=3.0)
    assert f.near_level(ticks=4.0) and not f.near_level(ticks=2.0)


def test_a_bar_with_no_level_in_the_book_is_never_near_one():
    assert not ft.Feat(ts=0.0, index=0).near_level()


def test_the_tick_size_is_not_hard_coded():
    s = quiet(40)
    a = ft.build(s, tick=0.25)
    b = ft.build(s, tick=1.0)
    da = [f.level_ticks for f in a if f.level_ticks is not None]
    db = [f.level_ticks for f in b if f.level_ticks is not None]
    if da and db:
        assert da != db


# ----------------------------------------------------------- shape

def test_one_feature_row_per_bar_in_order():
    s = quiet(50)
    feats = ft.build(s)
    assert len(feats) == 50
    assert [f.index for f in feats] == list(range(50))
    assert [f.ts for f in feats] == [x.ts for x in s]


def test_an_empty_input_is_an_empty_output():
    assert ft.build([]) == []
