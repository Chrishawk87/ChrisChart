"""Measuring the candle read against what the candle actually did.

The traps here are all about counting. The same candle supplies a read at
every fraction, so pooling them triples the apparent sample. The base rate
is not 50%, so accuracy without it beside it is meaningless. And the
control has to keep the MIX of leans fixed, or it measures the mix rather
than the ordering.

Each of those has a test that fails if the code does the convenient thing.
"""

from __future__ import annotations

import pytest

from liqmap import readstudy as rs
from liqmap.flow import Book, Level, Trade


def trade(ts, px, sz=1.0, side="buy"):
    return Trade(px=px, sz=sz, aggressor=side, ts=ts)


def book(ts, mid, imb=0.0):
    """A two-sided book leaning by `imb` (-1 offer heavy, +1 bid heavy)."""
    bid_sz = 100.0 * (1.0 + imb)
    ask_sz = 100.0 * (1.0 - imb)
    return Book(coin="ES", ts=ts,
                bids=[Level(px=mid - 0.25 * (i + 1), sz=bid_sz)
                      for i in range(5)],
                asks=[Level(px=mid + 0.25 * (i + 1), sz=ask_sz)
                      for i in range(5)])


def stream(prices, start=1_700_000_000.0, step=5.0, side="buy"):
    """Alternating trade/book events along a price path."""
    out = []
    for i, p in enumerate(prices):
        ts = start + i * step
        out.append(("trade", trade(ts, p, side=side)))
        out.append(("book", book(ts, p)))
    return out


def obs(lean, up, start=0.0, frac=0.5, ticks=0.0, fwd=None):
    """`up` and `ticks` describe the CANDLE; `fwd` is what price did from
    the read to the close, which is the tradeable quantity. It defaults to
    following the candle so the simple fixtures stay readable."""
    return rs.Observation(candle_start=start, fraction=frac, lean=lean,
                          score=0.0, agreement=1.0, signals=2,
                          went_up=up, change_ticks=ticks, read_px=100.0,
                          forward_ticks=(fwd if fwd is not None
                                         else (1.0 if up else -1.0)))


# ------------------------------------------------------------- the walk

def test_a_read_is_taken_once_per_candle_per_fraction():
    """Not once per trade. Every sample inside a candle shares that
    candle's outcome, so sampling densely manufactures a sample size that
    does not exist."""
    prices = [5000.0 + (i % 20) * 0.25 for i in range(400)]
    got = rs.study(stream(prices), interval_s=900.0,
                   fractions=(0.25, 0.5, 0.75))
    assert got
    seen: dict[tuple, int] = {}
    for o in got:
        key = (o.candle_start, o.fraction)
        seen[key] = seen.get(key, 0) + 1
    assert set(seen.values()) == {1}


def test_the_read_only_sees_the_candle_so_far():
    """Causality. A read at 25% that could see the candle's close would
    be scoring itself against its own input."""
    prices = [5000.0] * 50 + [5050.0] * 50      # a big late move
    got = [o for o in rs.study(stream(prices, step=5.0), interval_s=900.0,
                               fractions=(0.1,))]
    assert got
    # At 10% of a 15 minute candle only the flat opening is visible, so
    # the read cannot be leaning hard on a move that has not happened.
    assert all(abs(o.score) < 2.0 for o in got)


def test_the_outcome_is_attached_after_the_candle_closes():
    prices = [5000.0 + i * 0.25 for i in range(400)]
    got = rs.study(stream(prices), interval_s=900.0, fractions=(0.5,))
    assert got
    assert all(o.went_up for o in got)
    assert all(o.change_ticks > 0 for o in got)


def test_a_falling_candle_is_recorded_as_down():
    prices = [5100.0 - i * 0.25 for i in range(400)]
    got = rs.study(stream(prices, side="sell"), interval_s=900.0,
                   fractions=(0.5,))
    assert got
    assert not any(o.went_up for o in got)


def test_no_events_is_no_observations():
    assert rs.study([]) == []


def test_books_without_trades_do_not_make_candles():
    evs = [("book", book(1_700_000_000.0 + i * 5, 5000.0)) for i in range(50)]
    assert rs.study(evs) == []


def test_the_watcher_is_recentred_when_price_walks_away():
    """A level watched from two hours ago is not where the market is, and
    its impact baseline describes a different piece of tape."""
    import inspect
    src = inspect.getsource(rs.study)
    assert "RECENTRE_BPS" in src
    assert rs.RECENTRE_BPS > rs.BAND_BPS


# ------------------------------------------------------- the base rate

def test_the_base_rate_counts_candles_not_reads():
    """The same candle appears once per fraction. Counting it three times
    makes the base rate look better measured than it is."""
    rows = [obs("up", True, start=1.0, frac=f) for f in (0.25, 0.5, 0.75)]
    rows += [obs("down", False, start=2.0, frac=f)
             for f in (0.25, 0.5, 0.75)]
    assert rs.base_rate(rows) == pytest.approx(0.5)


def test_the_base_rate_of_nothing_is_zero_not_a_crash():
    assert rs.base_rate([]) == 0.0


# ----------------------------------------------------------- the table

def test_accuracy_is_per_lean_with_an_interval():
    rows = [obs("up", True, fwd=3.0) for _ in range(30)]
    rows += [obs("up", False, fwd=-3.0) for _ in range(10)]
    out = rs.by_lean(rows)
    assert out["up"]["n"] == 40
    assert out["up"]["rate"] == pytest.approx(0.75)
    assert out["up"]["ci_low"] < 0.75 < out["up"]["ci_high"]


def test_a_thin_bucket_is_flagged_rather_than_quietly_reported():
    out = rs.by_lean([obs("up", True) for _ in range(5)])
    assert out["up"]["enough"] is False


def test_a_down_lean_scores_a_hit_when_price_falls_after_it():
    out = rs.by_lean([obs("down", False, fwd=-4.0) for _ in range(30)])
    assert out["down"]["rate"] == pytest.approx(1.0)


def test_the_average_move_is_signed_by_the_lean():
    """An 'up' read before a fall has to show a NEGATIVE average, or a
    read that is always wrong looks the same as one always right."""
    out = rs.by_lean([obs("up", False, fwd=-8.0) for _ in range(30)])
    assert out["up"]["avg_ticks"] == pytest.approx(-8.0)
    out = rs.by_lean([obs("down", False, fwd=-8.0) for _ in range(30)])
    assert out["down"]["avg_ticks"] == pytest.approx(8.0)


def test_a_read_that_is_right_about_the_candle_but_late_is_not_a_hit():
    """THE correction. At half past a candle that is already up ten
    ticks, 'up' is right about the close and has described a move that
    already happened. Scoring that as a hit is how a read beats its own
    control on a pure random walk."""
    rows = [obs("up", True, ticks=10.0, fwd=-3.0) for _ in range(30)]
    out = rs.by_lean(rows)
    assert out["up"]["rate"] == pytest.approx(0.0)       # forward: wrong
    assert out["up"]["candle_rate"] == pytest.approx(1.0)  # candle: right
    assert out["up"]["avg_ticks"] == pytest.approx(-3.0)


def test_the_forward_base_rate_is_reported_separately():
    rows = [obs("up", True, fwd=2.0) for _ in range(7)]
    rows += [obs("up", True, fwd=-2.0) for _ in range(3)]
    assert rs.forward_rate(rows) == pytest.approx(0.7)
    assert rs.forward_rate([]) == 0.0


def test_flat_reads_are_reported_but_not_scored_as_hits():
    """A flat lean made no call. Scoring it either way invents a
    prediction that was never made."""
    out = rs.by_lean([obs("flat", True) for _ in range(30)])
    assert "hits" not in out["flat"]
    assert out["flat"]["went_up"] == 30


# ---------------------------------------------------------- the control

def test_the_control_keeps_the_mix_of_leans_fixed():
    """Permuting WHICH candle got which lean, not inventing new leans.
    A control that re-rolled the leans would be measuring the mix rather
    than the ordering."""
    rows = [obs("up", True, fwd=1.0) for _ in range(60)]
    rows += [obs("down", True, fwd=1.0) for _ in range(40)]  # all rose
    ctl = rs.control(rows, runs=200)
    # 60% of the calls are "up" and 100% of candles rose, so shuffling
    # scores 60% every time. The spread is zero, not noise.
    assert ctl["mean"] == pytest.approx(0.6)
    assert ctl["sd"] == pytest.approx(0.0)


def test_a_perfect_read_beats_its_control():
    rows = [obs("up", True) for _ in range(50)]
    rows += [obs("down", False) for _ in range(50)]
    ctl = rs.control(rows, runs=200)
    assert rs.beats(1.0, 100, ctl)
    assert ctl["mean"] == pytest.approx(0.5, abs=0.1)


def test_a_read_that_only_matches_the_base_rate_does_not_beat_it():
    """The number that makes raw accuracy meaningless: always saying 'up'
    into a market that rose 70% of the time scores 70% and has found
    nothing."""
    rows = [obs("up", True, fwd=1.0) for _ in range(70)]
    rows += [obs("up", False, fwd=-1.0) for _ in range(30)]
    ctl = rs.control(rows, runs=200)
    assert not rs.beats(0.7, 100, ctl)


def test_flat_reads_are_left_out_of_the_control():
    rows = [obs("flat", True) for _ in range(100)]
    assert rs.control(rows)["runs"] == 0


def test_a_thin_sample_cannot_beat_anything():
    rows = [obs("up", True) for _ in range(10)]
    assert rs.control(rows)["runs"] == 0
    assert not rs.beats(1.0, 10, {"runs": 50, "mean": 0.5, "sd": 0.05})


def test_the_minimum_is_stated_not_hidden():
    assert rs.MIN_CANDLES >= 25
    assert rs.enough(rs.MIN_CANDLES) and not rs.enough(rs.MIN_CANDLES - 1)
