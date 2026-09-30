"""Volume profile, against distributions whose answer is known by hand.

The load-bearing test is `test_the_developing_profile_cannot_see_later_bars`.
A value area is not known until its session ends, so classifying a bar
against the day's final VA is look-ahead -- and it is the specific way
auction-theory backtests produce results that evaporate live, because every
"rejection at the VA high" gets scored against a level the rejection itself
helped set. Nothing crashes. The study just comes back positive.
"""

from __future__ import annotations

import numpy as np
import pytest

from liqmap import profile as pf


class Bar:
    __slots__ = ("ts", "open", "high", "low", "close", "volume")

    def __init__(self, ts, open, high, low, close, volume):
        self.ts, self.open, self.high = ts, open, high
        self.low, self.close, self.volume = low, close, volume


def bar(i, lo, hi, v=100.0, step=5.0):
    return Bar(ts=1_789_999_800.0 + i * step, open=lo, high=hi, low=lo,
               close=hi, volume=v)


def flat(i, px, v=100.0):
    return bar(i, px, px, v)


# ------------------------------------------------- volume distribution

def test_a_rangeless_bar_puts_everything_in_one_row():
    lo_row, vol = pf.accumulate([flat(0, 7000.10, v=500.0)])
    assert vol.sum() == pytest.approx(500.0)
    assert (vol > 0).sum() == 1


def test_volume_splits_in_proportion_to_the_overlap():
    """7000.10 to 7000.40 covers 0.15 of the 7000.00 row and 0.15 of the
    7000.25 row -- an even split. Worked out by hand, not by the code."""
    lo_row, vol = pf.accumulate([bar(0, 7000.10, 7000.40, v=100.0)])
    assert vol.size == 2
    assert vol[0] == pytest.approx(50.0)
    assert vol[1] == pytest.approx(50.0)


def test_an_uneven_overlap_splits_unevenly():
    # 7000.05 -> 7000.30: 0.20 in the first row, 0.05 in the second.
    _, vol = pf.accumulate([bar(0, 7000.05, 7000.30, v=100.0)])
    assert vol[0] == pytest.approx(80.0)
    assert vol[1] == pytest.approx(20.0)


def test_total_volume_is_conserved():
    bars = [bar(i, 7000.0 + i * 0.1, 7000.0 + i * 0.1 + 0.6, v=37.0)
            for i in range(50)]
    _, vol = pf.accumulate(bars)
    assert vol.sum() == pytest.approx(50 * 37.0)


def test_a_zero_volume_bar_contributes_nothing():
    _, vol = pf.accumulate([flat(0, 7000.0, v=0.0), flat(1, 7001.0, v=10.0)])
    assert vol.sum() == pytest.approx(10.0)


def test_an_empty_input_is_empty_not_an_error():
    assert pf.final([]).empty


# -------------------------------------------------------------- the POC

def test_the_poc_is_the_heaviest_row():
    bars = [flat(0, 7000.0, 10.0), flat(1, 7005.0, 900.0),
            flat(2, 7010.0, 10.0)]
    assert pf.final(bars).poc == pytest.approx(7005.125, abs=0.13)


def test_the_poc_follows_volume_not_time_spent():
    """One heavy print outweighs many light ones. A TPO count would say the
    opposite, and conflating the two is a common porting error."""
    bars = [flat(i, 7000.0, 1.0) for i in range(50)]
    bars.append(flat(50, 7020.0, 5000.0))
    assert pf.final(bars).poc > 7019.0


# ------------------------------------------------------- the value area

def test_the_value_area_holds_about_seventy_percent():
    rng = np.random.default_rng(4)
    bars = [flat(i, round(7000.0 + rng.normal(0, 2) * 4) / 4, 100.0)
            for i in range(4000)]
    p = pf.final(bars)
    inside = sum(b.volume for b in bars if p.val <= b.low <= p.vah)
    assert 0.62 <= inside / p.total <= 0.80


def test_the_value_area_brackets_the_poc():
    rng = np.random.default_rng(7)
    bars = [flat(i, round(7000.0 + rng.normal(0, 3) * 4) / 4, 100.0)
            for i in range(2000)]
    p = pf.final(bars)
    assert p.val <= p.poc <= p.vah


def test_a_tight_distribution_gives_a_narrow_value_area():
    tight = [flat(i, 7000.0 + (i % 3) * 0.25, 100.0) for i in range(600)]
    wide = [flat(i, 7000.0 + (i % 80) * 0.25, 100.0) for i in range(600)]
    assert pf.final(tight).va_width_ticks < pf.final(wide).va_width_ticks


def test_a_single_price_gives_a_one_row_value_area():
    p = pf.final([flat(i, 7000.0, 100.0) for i in range(100)])
    assert p.va_width_ticks == pytest.approx(1.0)


def test_inside_reports_membership_of_the_value_area():
    p = pf.final([flat(i, 7000.0 + (i % 20) * 0.25, 100.0)
                  for i in range(500)])
    assert p.inside(p.poc)
    assert not p.inside(p.vah + 5.0)
    assert not p.inside(p.val - 5.0)


# -------------------------------------------------------------- nodes

def test_a_heavy_isolated_row_is_a_high_volume_node():
    bars = [flat(i, 7000.0 + (i % 40) * 0.25, 10.0) for i in range(400)]
    bars += [flat(500 + i, 7005.0, 400.0) for i in range(30)]
    p = pf.final(bars)
    assert any(abs(x - 7005.0) < 0.5 for x in p.hvn)


def test_a_broad_shelf_does_not_report_as_many_adjacent_nodes():
    """Without the local-extreme test a flat high-volume shelf reports as
    one HVN per row, and an extractor would fire twenty times on one
    feature."""
    bars = [flat(i, 7000.0 + (i % 12) * 0.25, 100.0) for i in range(1200)]
    p = pf.final(bars, tick=0.25)
    assert len(p.hvn) <= 4


def test_an_untraded_price_is_a_gap_not_a_low_volume_node():
    """Reporting prices the market never visited as LVNs would scatter
    targets across places no auction occurred."""
    bars = [flat(i, 7000.0, 500.0) for i in range(100)]
    bars += [flat(200 + i, 7010.0, 500.0) for i in range(100)]
    p = pf.final(bars)
    for x in p.lvn:
        row = int((x - p.tick / 2) / p.tick) - p.lo_row
        assert p.volumes[row] > 0


def test_the_thin_tail_of_a_trend_day_is_not_a_string_of_nodes():
    """The 2026-09-21 failure.

    A session that opened at 7760 and was accepted at 7838 reported 29
    "low volume nodes", every one below the value area, in the ground it
    came from and left. Those are range edges. "LVN traversal" means a thin
    shelf BETWEEN two accepted areas, and an extractor reading tail rows
    would build a study out of where each day happened to start.
    """
    rng = np.random.default_rng(3)
    bars = [flat(i, round((7760.0 + i * 0.125) * 4) / 4, 8.0)
            for i in range(600)]
    bars += [flat(1000 + i, round((7838.0 + rng.normal(0, 3)) * 4) / 4,
                  200.0) for i in range(3000)]
    p = pf.final(bars)
    assert len([x for x in p.lvn if x < p.val]) == 0


def test_a_ratio_test_alone_does_not_fix_the_tail():
    """Why the threshold is absolute rather than relative.

    The first fix asked for a neighbour holding 3x the row's volume. In
    thin ground a row with 8 contracts sits beside one with 24, so it
    passed everywhere and produced 135 tail nodes instead of 29. Three
    times almost nothing is still almost nothing.
    """
    assert pf.VALLEY_FLOOR > 0            # fraction of the PEAK, not of self
    assert pf.VALLEY_WINDOW > pf.NODE_WINDOW


def test_an_untouched_gap_beside_a_valley_does_not_disqualify_it():
    """Zero is lower than thin, so a strict local-minimum test rejected the
    very object the concept names."""
    bars = [flat(i, 7800.0 + (i % 8) * 0.25, 300.0) for i in range(1200)]
    bars += [flat(2000 + i, 7803.0 + (i % 2) * 0.25, 20.0) for i in range(20)]
    bars += [flat(4000 + i, 7805.0 + (i % 8) * 0.25, 300.0)
             for i in range(1200)]
    p = pf.final(bars, tick=0.25)
    assert len(p.lvn) == 1
    assert abs(p.lvn[0] - 7803.0) < 0.5


def test_a_thin_row_between_two_heavy_ones_is_a_low_volume_node():
    bars = [flat(i, 7000.0, 500.0) for i in range(60)]
    bars += [flat(100 + i, 7000.25, 5.0) for i in range(2)]
    bars += [flat(200 + i, 7000.50, 500.0) for i in range(60)]
    p = pf.final(bars)
    assert any(abs(x - 7000.25) < 0.2 for x in p.lvn)


# ============================================================ LOOK-AHEAD

def _two_halves():
    """Quiet at 7000, then displaced to 7050 and HEAVIER.

    The second half has to outweigh the first, or the combined profile has
    two equal peaks, argmax takes the lower one, and the POC never moves --
    which would make the look-ahead test below pass without testing
    anything.
    """
    a = [flat(i, 7000.0 + (i % 4) * 0.25, 100.0) for i in range(300)]
    b = [flat(300 + i, 7050.0 + (i % 4) * 0.25, 400.0) for i in range(300)]
    return a, b


def test_the_developing_profile_cannot_see_later_bars():
    """THE test. The developing profile partway through must be identical
    whether or not the displaced second half exists in the input."""
    a, b = _two_halves()
    only_first = pf.developing(a, stride=50, warmup=60, tick=0.25)
    both = pf.developing(a + b, stride=50, warmup=60, tick=0.25)

    for (i, p), (j, q) in zip(only_first, both[:len(only_first)]):
        assert i == j
        assert p.poc == q.poc
        assert p.vah == q.vah
        assert p.val == q.val
        assert p.total == q.total


def test_the_developing_profile_at_the_end_equals_the_final_one():
    a, b = _two_halves()
    bars = a + b
    last = pf.developing(bars, stride=1, warmup=60)[-1][1]
    whole = pf.final(bars)
    assert last.poc == pytest.approx(whole.poc)
    assert last.total == pytest.approx(whole.total)


def test_the_final_profile_of_a_displaced_session_differs_from_its_middle():
    """Confirms the fixture actually has something to leak -- otherwise the
    look-ahead test above would pass trivially."""
    a, b = _two_halves()
    mid = pf.developing(a, stride=50, warmup=60)[-1][1]
    assert abs(pf.final(a + b).poc - mid.poc) > 5.0


def test_warmup_suppresses_profiles_built_on_too_few_bars():
    a, _ = _two_halves()
    assert all(i + 1 >= 60 for i, _ in pf.developing(a, warmup=60))


def test_stride_changes_where_it_reports_not_what_it_contains():
    a, _ = _two_halves()
    every = dict(pf.developing(a, stride=1, warmup=60))
    some = pf.developing(a, stride=25, warmup=60)
    for i, p in some:
        assert p.poc == every[i].poc
        assert p.total == every[i].total


# ------------------------------------------------------------- rolling

def test_the_rolling_window_is_trailing_not_centred():
    """A centred window is half made of bars the decision has not seen."""
    bars = [flat(i, 7000.0, 100.0) for i in range(400)]
    bars += [flat(400 + i, 7100.0, 100.0) for i in range(400)]
    out = dict(pf.rolling(bars, seconds=300.0, stride=50))
    early = out[350]
    assert early.poc < 7050.0            # has not seen the displacement


def test_the_rolling_window_drops_bars_that_fall_out_of_it():
    bars = [flat(i, 7000.0, 100.0) for i in range(200)]
    bars += [flat(200 + i, 7100.0, 100.0) for i in range(200)]
    out = dict(pf.rolling(bars, seconds=300.0, stride=10))
    late = out[390]
    assert late.poc > 7050.0             # the old regime has aged out


# ------------------------------------------------------------ plumbing

def test_the_conventions_are_the_stated_ones():
    """These are fixed by convention, not fitted. If one changes, it is a
    new hypothesis and has to be counted as one."""
    assert pf.TICK == 0.25
    assert pf.ROW_TICKS == 1
    assert pf.VALUE_AREA == 0.70
    assert pf.LVN_FRACTION == 0.30
    assert pf.HVN_FRACTION == 0.70
    assert pf.NODE_WINDOW == 4


def test_the_dict_carries_every_reported_level():
    p = pf.final([flat(i, 7000.0 + (i % 20) * 0.25, 100.0)
                  for i in range(500)])
    d = p.to_dict()
    for k in ("poc", "vah", "val", "va_width_ticks", "total", "hvn", "lvn"):
        assert k in d


# ------------------------------------------- the row size is not ES's tick

def test_the_row_size_is_read_from_the_bars_not_pinned_to_es():
    """A fixed 0.25 row is right for exactly one instrument.

    On BTC at 105,000 a 3,000-dollar range became 12,000 rows twenty-five
    cents wide, and the node window of +/- 4 rows spanned one dollar -- it
    detected nothing. On a sub-penny token the whole range was smaller than
    a single row, so there was no profile at all.
    """
    import random

    def synth(px, rng, tick, n=500, seed=1):
        r = random.Random(seed)
        out = []
        for _ in range(n):
            a = round((px + r.uniform(-rng / 2, rng / 2)) / tick) * tick
            out.append(Bar(0.0, a, a + tick, a - tick, a, 100.0))
        return out

    for price, rng, tick in ((7800, 90, 0.25),        # ES
                             (105_000, 3000, 1.0),    # BTC
                             (0.38, 0.03, 0.00001)):  # a sub-penny token
        bars = synth(price, rng, tick)
        rows = rng / pf.row_size(bars)
        assert 150 <= rows <= 900, f"{price}: {rows:.0f} rows"


def test_es_still_gets_exactly_one_tick_per_row():
    """Snapping up to the instrument's own tick keeps ES at one tick a row
    rather than some fraction of one."""
    import random
    r = random.Random(1)
    bars = []
    for _ in range(500):
        a = round((7800 + r.uniform(-45, 45)) * 4) / 4
        bars.append(Bar(0.0, a, a + 0.25, a - 0.25, a, 100.0))
    assert pf.row_size(bars) == pytest.approx(0.25)


def test_a_row_boundary_is_always_a_price_the_market_can_print():
    import random
    r = random.Random(3)
    bars = []
    for _ in range(400):
        a = round((105_000 + r.uniform(-1500, 1500)) / 1.0) * 1.0
        bars.append(Bar(0.0, a, a + 1, a - 1, a, 100.0))
    row = pf.row_size(bars)
    assert abs(row / pf.infer_tick(bars) - round(row / pf.infer_tick(bars))) < 1e-6


def test_sparse_prices_fall_back_rather_than_inferring_a_huge_tick():
    """Three bars five points apart would otherwise 'infer' a five point
    tick and give a three-row profile."""
    bars = [Bar(0.0, 7000.0, 7000.0, 7000.0, 7000.0, 10.0),
            Bar(1.0, 7005.0, 7005.0, 7005.0, 7005.0, 900.0),
            Bar(2.0, 7010.0, 7010.0, 7010.0, 7010.0, 10.0)]
    assert pf.infer_tick(bars) < 1.0


def test_an_explicit_tick_still_wins():
    import random
    r = random.Random(2)
    bars = [Bar(0.0, 105_000 + r.uniform(-1500, 1500), 0, 0, 0, 100.0)
            for _ in range(50)]
    for b in bars:
        b.high = b.open + 1; b.low = b.open - 1; b.close = b.open
    assert pf.final(bars, tick=0.25).tick == pytest.approx(0.25)
