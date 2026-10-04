"""Closed-bar aggression, and the CVD built from it.

The whole reason this file exists is that a CVD for any timeframe longer
than the rolling tape cannot be built from the tape. The tape holds about
an hour; a four-hour rung has seen a quarter of one bar, which is why every
slow row on the ladder sat at NOT COVERED and stayed there.

Two traps run through every test here.

A BAR WITH NO FLOW IS NOT A BALANCED BAR. A bar adopted from the exchange
has zero buy and zero sell because nobody counted them. Stored as zero it
becomes indistinguishable from a bar that really was balanced, and every
CVD containing it is quietly wrong with no way to tell.

TWO PARTIAL COUNTS OF ONE BAR DO NOT ADD UP TO THAT BAR. A reconnect, or a
previous run that saw half of a bar, must overwrite rather than accumulate.
A duplicated bar looks exactly like a real one of twice the size.
"""

from __future__ import annotations

import pytest

from liqmap import barflow as bf


@pytest.fixture()
def store(tmp_path):
    return bf.BarFlowStore(tmp_path / "t.db")


def row(ts, buy=100.0, sell=50.0, tf="15m", coin="BTC", trades=10):
    return bf.BarFlow(coin=coin, timeframe=tf, bar_ts=float(ts), buy=buy,
                      sell=sell, trades=trades)


# ------------------------------------------------------------- writing

def test_a_bar_with_no_aggression_counted_is_refused_not_stored_as_zero():
    """THE trap. Zero because nobody looked is not zero because it was
    balanced, and once the zero is in the table nothing downstream can
    tell them apart."""
    s = bf.BarFlowStore(":memory:")
    assert not s.record(bf.BarFlow("BTC", "15m", 0.0, buy=0.0, sell=0.0,
                                   trades=0))
    assert s.cvd("BTC", "15m").bars == 0


def test_a_genuinely_balanced_bar_is_stored(store):
    """It traded, and both sides were equal. That is a real reading."""
    assert store.record(bf.BarFlow("BTC", "15m", 0.0, buy=500.0, sell=500.0,
                                   trades=40))
    c = store.cvd("BTC", "15m", bars=6)
    assert c.bars == 1 and c.cvd == 0.0


def test_the_same_bar_twice_overwrites_rather_than_doubling(store):
    """A reconnect re-closes the bar it was in the middle of. Adding the
    two counts together would report a bar twice the size it was."""
    store.record(row(900.0, buy=100.0, sell=0.0))
    store.record(row(900.0, buy=180.0, sell=0.0))
    c = store.cvd("BTC", "15m", bars=6)
    assert c.bars == 1
    assert c.cvd == pytest.approx(180.0)


def test_markets_and_timeframes_do_not_bleed_into_each_other(store):
    store.record(row(900.0, tf="15m", buy=100.0, sell=0.0))
    store.record(row(900.0, tf="1h", buy=900.0, sell=0.0))
    store.record(row(900.0, coin="ETH", buy=7.0, sell=0.0))
    assert store.cvd("BTC", "15m").cvd == pytest.approx(100.0)
    assert store.cvd("BTC", "1h").cvd == pytest.approx(900.0)
    assert store.cvd("ETH", "15m").cvd == pytest.approx(7.0)


def test_old_bars_are_pruned(store):
    store.record(row(0.0))
    store.record(row(1_000_000.0))
    assert store.prune(older_than_s=100.0, now=1_000_000.0) == 1
    assert store.cvd("BTC", "15m").bars == 1


# ------------------------------------------------------------- reading

def test_the_cvd_is_the_sum_of_the_bars_behind_this_one(store):
    for i in range(6):
        store.record(row(i * 900.0, buy=100.0, sell=40.0))
    c = store.cvd("BTC", "15m", bars=6)
    assert c.cvd == pytest.approx(360.0)
    assert c.bars == 6 and c.complete


def test_the_bar_in_progress_is_not_counted_as_a_completed_one(store):
    for i in range(4):
        store.record(row(i * 900.0, buy=100.0, sell=0.0))
    c = store.cvd("BTC", "15m", bars=6, before_ts=2700.0)
    assert c.bars == 3
    assert c.cvd == pytest.approx(300.0)


def test_the_live_bar_rides_along_without_being_called_complete(store):
    store.record(row(0.0, buy=100.0, sell=0.0))
    c = store.cvd("BTC", "15m", bars=6, live_delta=-25.0)
    assert c.cvd == pytest.approx(100.0)
    assert c.total == pytest.approx(75.0)


def test_a_short_history_says_how_short(store):
    """Six bars of CVD and two bars of CVD are different claims, and the
    ladder draws them the same width."""
    store.record(row(0.0))
    store.record(row(900.0))
    c = store.cvd("BTC", "15m", bars=6)
    assert c.bars == 2 and c.asked == 6
    assert not c.complete


def test_a_timeframe_with_nothing_stored_reports_none_not_zero(store):
    c = store.cvd("BTC", "4h", bars=6)
    assert c.bars == 0 and c.source == "none"
    assert not c.complete


def test_only_the_most_recent_bars_are_summed(store):
    for i in range(20):
        store.record(row(i * 900.0, buy=10.0, sell=0.0))
    assert store.cvd("BTC", "15m", bars=6).cvd == pytest.approx(60.0)


# ------------------------------------------------- merging the two sources

def test_the_feed_wins_on_a_bar_both_sources_hold():
    """The in-memory bar was counted fill by fill by THIS process. The
    stored row may be an older run's partial view of the same bar, and two
    partial counts of one bar do not add up to that bar."""
    stored = [row(900.0, buy=10.0, sell=0.0)]
    live = [row(900.0, buy=300.0, sell=0.0)]
    c = bf.merge("15m", stored, live, bars=6)
    assert c.bars == 1
    assert c.cvd == pytest.approx(300.0)
    assert c.source == "both"


def test_merging_keeps_bars_only_one_source_has():
    stored = [row(0.0, buy=10.0, sell=0.0)]
    live = [row(900.0, buy=20.0, sell=0.0)]
    c = bf.merge("15m", stored, live, bars=6)
    assert c.bars == 2
    assert c.cvd == pytest.approx(30.0)


def test_merging_reports_which_source_it_stood_on():
    assert bf.merge("15m", [], [row(0.0)], bars=6).source == "memory"
    assert bf.merge("15m", [row(0.0)], [], bars=6).source == "stored"
    assert bf.merge("15m", [], [], bars=6).source == "none"


def test_merging_still_honours_the_bar_count():
    rows = [row(i * 900.0, buy=10.0, sell=0.0) for i in range(12)]
    assert bf.merge("15m", rows, [], bars=4).bars == 4
