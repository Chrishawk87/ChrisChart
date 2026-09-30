"""High-frequency book reads, and the overlap trap.

The test that matters most here is `test_overlapping_observations_are_refused`.
Sampling every 5 seconds and measuring 30 forward means consecutive rows
share five sixths of their outcome; the interval computed over them is far
too narrow, and 74,000 overlapping looks can report six times more
confidence than they hold. Nothing crashes. The study simply comes back
sure of something that is not there.
"""

from __future__ import annotations

import math

import pytest

from liqmap import intrabar as ib
from liqmap.escondense import Slice


def sl(i, px=7000.0, bsz=50, asz=50, delta=0.0, buy=50.0, sell=50.0,
       step=5.0):
    s = Slice(ts=1_789_999_800.0 + i * step,
              bids=[(px - k * 0.25, bsz) for k in range(10)],
              asks=[(px + 0.25 + k * 0.25, asz) for k in range(10)],
              delta=delta, buy_vol=buy, sell_vol=sell, trades_in=8,
              prior_close=px, prior_open=px)
    s.open = px
    s.close = px
    s.high = px
    s.low = px
    s.path = [(0.0, 0.0)]
    return s


# ------------------------------------------------------- the overlap

def test_overlapping_observations_are_refused():
    """A stride shorter than the horizon shares outcome between rows."""
    assert ib.stride_for(6, stride=1) == 6
    assert ib.stride_for(6, stride=3) == 6
    assert ib.stride_for(6, stride=12) == 12


def test_the_default_stride_is_the_horizon():
    assert ib.stride_for(4) == 4
    assert ib.stride_for(1) == 1


def test_a_longer_horizon_yields_fewer_observations():
    """The honest consequence: more forward time means fewer independent
    looks, and the interval has to widen accordingly.

    Price has to MOVE here -- a flat series has every outcome skipped as a
    tie and both counts come back zero, which passes nothing.
    """
    s = [sl(i, px=7000.0 + i * 0.25) for i in range(600)]
    sides = [1] * 600
    short = ib.direction(s, sides, horizon_rows=1)
    long = ib.direction(s, sides, horizon_rows=12)
    assert short["n"] > long["n"] * 5


def test_the_reported_stride_is_the_one_used():
    s = [sl(i, px=7000.0 + i) for i in range(400)]
    out = ib.direction(s, [1] * 400, horizon_rows=6, stride=2)
    assert out["stride"] == 6


# ------------------------------------------------------- the reads

def test_imbalance_is_read_at_several_depths():
    s = sl(0, bsz=90, asz=10)
    r = ib.reads([s])[0]
    assert r.imb1 == pytest.approx(0.8)
    assert r.imb10 == pytest.approx(0.8)


def test_depth_changes_the_reading_when_the_book_is_lopsided():
    """A heavy touch over a thin back book must read differently at 1 and
    10 levels, or the depth parameter is decorative."""
    s = sl(0)
    s.bids = [(7000.0, 900)] + [(7000.0 - k * 0.25, 10) for k in range(1, 10)]
    s.asks = [(7000.25 + k * 0.25, 100) for k in range(10)]
    r = ib.reads([s])[0]
    assert r.imb1 > 0.7
    assert r.imb10 < r.imb1


def test_velocity_differences_against_the_previous_row_not_the_next():
    """Differencing forward would put the next few seconds of the answer
    into the question."""
    a = sl(0, bsz=10, asz=90)      # imb5 = -0.8
    b = sl(1, bsz=90, asz=10)      # imb5 = +0.8
    c = sl(2, bsz=90, asz=10)
    rs = ib.reads([a, b, c])
    assert rs[0].velocity == 0.0                       # nothing before it
    assert rs[1].velocity == pytest.approx(1.6)
    assert rs[2].velocity == pytest.approx(0.0)


def test_flow_is_the_signed_share_of_volume():
    r = ib.reads([sl(0, delta=40.0, buy=70.0, sell=30.0)])[0]
    assert r.flow == pytest.approx(0.4)


def test_a_row_with_no_volume_has_flat_flow_not_a_division_by_zero():
    assert ib.reads([sl(0, buy=0.0, sell=0.0)])[0].flow == 0.0


def test_an_empty_book_reads_flat_rather_than_raising():
    s = sl(0)
    s.bids, s.asks = [], []
    assert ib.reads([s])[0].imb5 == 0.0


def test_one_read_per_slice_in_order():
    rs = ib.reads([sl(i) for i in range(50)])
    assert [r.index for r in rs] == list(range(50))


# ------------------------------------------------------- direction

def test_a_call_that_matches_the_move_scores_high():
    s = [sl(i, px=7000.0 + i * 0.25) for i in range(400)]   # marches up
    out = ib.direction(s, [1] * 400, horizon_rows=2)
    assert out["ready"] and out["rate"] == pytest.approx(1.0)


def test_a_call_that_fights_the_move_scores_low():
    s = [sl(i, px=7000.0 + i * 0.25) for i in range(400)]
    out = ib.direction(s, [-1] * 400, horizon_rows=2)
    assert out["rate"] == pytest.approx(0.0)
    assert out["real"]                      # inverted is still information


def test_unchanged_prices_are_skipped_rather_than_counted_as_wins():
    """A flat outcome is not a correct call. Counting it either way is a
    free win or a free loss on the quietest bars, which are the most
    numerous."""
    s = [sl(i, px=7000.0) for i in range(400)]
    assert not ib.direction(s, [1] * 400, horizon_rows=2)["ready"]


def test_a_zero_side_is_no_call():
    s = [sl(i, px=7000.0 + i * 0.25) for i in range(400)]
    assert not ib.direction(s, [0] * 400, horizon_rows=2)["ready"]


def test_too_few_observations_reports_not_ready_rather_than_a_number():
    s = [sl(i, px=7000.0 + i) for i in range(20)]
    assert ib.direction(s, [1] * 20, horizon_rows=2)["ready"] is False


def test_a_coin_is_not_flagged_real_more_often_than_chance():
    """Averaged over seeds, not asserted on one.

    A single seed either clears the interval or does not, and at a nominal
    95% about one run in twenty will. Pinning the behaviour to one seed
    makes the test a coin flip about a coin flip -- seed 5 fails and seed 6
    passes, and neither says anything about the code. What matters is the
    RATE, which should sit near one in twenty and never near half.
    """
    import random
    import statistics

    flagged, rates = 0, []
    for seed in range(40):
        rng = random.Random(seed)
        px, s = 7000.0, []
        for i in range(3000):
            px += rng.choice([-0.25, 0.25])
            s.append(sl(i, px=px))
        sides = [rng.choice([-1, 1]) for _ in range(3000)]
        d = ib.direction(s, sides, horizon_rows=4)
        rates.append(d["rate"])
        flagged += 1 if d["real"] else 0

    assert flagged <= 5                          # ~2 expected of 40
    assert 0.47 < statistics.mean(rates) < 0.53


# -------------------------------------------------- the cost arithmetic

def test_the_edge_needed_depends_only_on_total_width():
    """The asymmetry cancels. Widening the target lowers the breakeven and
    the achievable rate by the same amount."""
    assert ib.needed_edge(5, 3) == pytest.approx(ib.needed_edge(3, 5))
    assert ib.needed_edge(4, 4) == pytest.approx(ib.needed_edge(5, 3))


def test_a_taker_pays_the_spread_as_well_as_commission():
    assert ib.COST_TAKER == pytest.approx(17.50)
    assert ib.needed_edge(5, 3) > ib.needed_edge(5, 3,
                                                 cost_usd=ib.COST_COMMISSION)


def test_tighter_targets_demand_more_edge_not_less():
    """The opposite of the scalping instinct, and the reason a one-tick
    scalp is the hardest version of the trade rather than the easiest."""
    wide = ib.needed_edge(20, 20)
    tight = ib.needed_edge(1, 1)
    assert tight > wide * 10


def test_the_one_tick_scalp_needs_an_implausible_edge():
    assert ib.needed_edge(1, 1) == pytest.approx(0.70, abs=0.01)


def test_the_no_skill_rate_is_the_gamblers_ruin():
    assert ib.no_skill(5, 3) == pytest.approx(0.375)
    assert ib.no_skill(5, 5) == pytest.approx(0.5)


# ------------------------------------------------------------ signals

def test_a_cut_produces_calls_only_outside_it():
    rs = [ib.Read(index=0, ts=0, mid=1, imb5=0.9),
          ib.Read(index=1, ts=0, mid=1, imb5=0.0),
          ib.Read(index=2, ts=0, mid=1, imb5=-0.9)]
    assert ib.signal(rs, "imb5", 0.5) == [1, 0, -1]


def test_a_higher_quantile_gives_a_higher_cut():
    rs = [ib.Read(index=i, ts=0, mid=1, imb5=(i - 50) / 50.0)
          for i in range(100)]
    assert ib.quantile_cut(rs, "imb5", 0.9) > ib.quantile_cut(rs, "imb5", 0.5)


def test_a_cut_on_an_empty_set_is_zero_not_an_error():
    assert ib.quantile_cut([], "imb5", 0.9) == 0.0
