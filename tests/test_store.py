"""Store and touch-resolution tests.

`resolve_touch` decides the outcome variable for every observation, so a bug
here corrupts the validation directly and silently.
"""

import math
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from liqmap.bucket import Position, build_map
from liqmap.store import Store, resolve_touch

START = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)
SPOT = 100_000.0
SIGMA = 0.0006


def series(values, step_minutes=1.0, start=START):
    return [(start + timedelta(minutes=i * step_minutes), v)
            for i, v in enumerate(values)]


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmp:
        s = Store(Path(tmp) / "t.db")
        yield s
        s.close()


# -- touch resolution ------------------------------------------------------

def test_upside_touch_is_detected_with_timing():
    prices = series([100_000, 100_200, 100_500, 101_000, 100_800])
    touched, when, _ = resolve_touch(prices, START, SPOT, 100_400.0, SIGMA, 60.0)
    assert touched
    assert when == pytest.approx(2.0)      # third sample, two minutes in


def test_downside_touch_is_detected():
    prices = series([100_000, 99_800, 99_400, 99_000])
    touched, when, _ = resolve_touch(prices, START, SPOT, 99_500.0, SIGMA, 60.0)
    assert touched
    assert when == pytest.approx(2.0)


def test_untouched_level_reports_false():
    prices = series([100_000, 100_100, 100_050, 99_950])
    touched, when, move = resolve_touch(prices, START, SPOT, 101_000.0, SIGMA, 60.0)
    assert not touched
    assert when is None and move is None


def test_touch_after_the_horizon_does_not_count():
    """Price reaches the level, but too late. Counting it would inflate every
    touch rate and manufacture lift out of nothing."""
    prices = series([100_000] * 30 + [101_000])
    touched, _, _ = resolve_touch(prices, START, SPOT, 100_900.0, SIGMA,
                                  horizon_minutes=10.0)
    assert not touched

    touched_longer, _, _ = resolve_touch(prices, START, SPOT, 100_900.0, SIGMA,
                                         horizon_minutes=60.0)
    assert touched_longer


def test_continuation_through_an_upside_level_is_positive():
    prices = series([100_000, 100_500, 101_000, 101_500, 102_000])
    _, _, move = resolve_touch(prices, START, SPOT, 100_400.0, SIGMA,
                               horizon_minutes=60.0, follow_minutes=3.0)
    assert move > 0


def test_reversal_off_an_upside_level_is_negative():
    prices = series([100_000, 100_500, 100_200, 99_800, 99_500])
    _, _, move = resolve_touch(prices, START, SPOT, 100_400.0, SIGMA,
                               horizon_minutes=60.0, follow_minutes=3.0)
    assert move < 0


def test_continuation_through_a_downside_level_is_also_positive():
    """Sign convention: positive always means the move continued in the
    direction the touch implied, whichever side the level sat on."""
    prices = series([100_000, 99_500, 99_000, 98_500, 98_000])
    _, _, move = resolve_touch(prices, START, SPOT, 99_600.0, SIGMA,
                               horizon_minutes=60.0, follow_minutes=3.0)
    assert move > 0


def test_reversal_off_a_downside_level_is_negative():
    prices = series([100_000, 99_500, 99_800, 100_200, 100_600])
    _, _, move = resolve_touch(prices, START, SPOT, 99_600.0, SIGMA,
                               horizon_minutes=60.0, follow_minutes=3.0)
    assert move < 0


def test_move_is_expressed_in_sigma():
    follow = 4.0
    sd = SIGMA * math.sqrt(follow)
    target = 100_400.0
    final = target * math.exp(2.0 * sd)      # exactly two sigma beyond

    prices = series([100_000, 100_500] + [final] * 5)
    _, _, move = resolve_touch(prices, START, SPOT, target, SIGMA,
                               horizon_minutes=60.0, follow_minutes=follow)
    assert move == pytest.approx(2.0, abs=0.05)


def test_empty_price_series_is_safe():
    assert resolve_touch([], START, SPOT, 101_000.0, SIGMA, 60.0) == (False, None, None)


def test_prices_before_the_snapshot_are_ignored():
    """A level touched an hour BEFORE the snapshot is not a prediction."""
    prices = series([101_000, 100_000, 100_050], start=START - timedelta(minutes=10))
    touched, _, _ = resolve_touch(prices, START, SPOT, 100_900.0, SIGMA, 60.0)
    assert not touched


# -- store -----------------------------------------------------------------

def test_snapshot_round_trips_with_placebos(store):
    positions = [
        Position("0xa", "BTC", 1.0, SPOT, SPOT * 0.97, 5e6, 10),
        Position("0xb", "BTC", -1.0, SPOT, SPOT * 1.03, 3e6, 10),
    ]
    lm = build_map(positions, "BTC", SPOT)
    snap_id = store.save_snapshot(lm, SIGMA, min_notional=0.0)

    counts = store.counts()
    assert counts["snapshots"] == 1
    assert counts["clusters"] == 2
    assert counts["placebos"] >= 1

    pending = store.unresolved_clusters(horizon_minutes=0.0)
    assert len(pending) == counts["clusters"] + counts["placebos"]
    assert all(p["snapshot_id"] == snap_id for p in pending)


def test_placebos_avoid_real_cluster_levels(store):
    positions = [Position("0xa", "BTC", 1.0, SPOT, SPOT * 0.97, 5e6, 10)]
    lm = build_map(positions, "BTC", SPOT)
    store.save_snapshot(lm, SIGMA, min_notional=0.0, placebo_per_cluster=3)

    rows = store.unresolved_clusters(0.0)
    real = [r["price"] for r in rows if not r["is_placebo"]]
    placebo = [r["price"] for r in rows if r["is_placebo"]]

    assert placebo
    for p in placebo:
        assert all(abs(p - r) / SPOT >= 0.0015 for r in real)


def test_placebos_are_distance_matched(store):
    """A placebo at a different distance measures distance, not clusters."""
    positions = [Position("0xa", "BTC", 1.0, SPOT, SPOT * 0.95, 5e6, 10)]
    lm = build_map(positions, "BTC", SPOT)
    store.save_snapshot(lm, SIGMA, min_notional=0.0, placebo_per_cluster=5)

    rows = store.unresolved_clusters(0.0)
    real_d = [abs(r["price"] - SPOT) for r in rows if not r["is_placebo"]][0]
    placebo_d = [abs(r["price"] - SPOT) for r in rows if r["is_placebo"]]

    for d in placebo_d:
        assert d == pytest.approx(real_d, rel=0.10)


def test_outcomes_become_observations(store):
    positions = [Position("0xa", "BTC", 1.0, SPOT, SPOT * 0.97, 5e6, 10)]
    lm = build_map(positions, "BTC", SPOT)
    store.save_snapshot(lm, SIGMA, min_notional=0.0)

    for row in store.unresolved_clusters(0.0):
        store.record_outcome(row["cluster_id"], 240.0, True, 30.0, 0.5)

    obs = store.observations()
    assert len(obs) >= 2
    assert all(o.touched for o in obs)
    assert any(o.is_placebo for o in obs)
    assert all(o.horizon_minutes == 240.0 for o in obs)
    assert obs[0].baseline_prob > 0


def test_recording_an_outcome_twice_updates_rather_than_duplicates(store):
    positions = [Position("0xa", "BTC", 1.0, SPOT, SPOT * 0.97, 5e6, 10)]
    lm = build_map(positions, "BTC", SPOT)
    store.save_snapshot(lm, SIGMA, min_notional=0.0)
    cid = store.unresolved_clusters(0.0)[0]["cluster_id"]

    store.record_outcome(cid, 240.0, False, None, None)
    store.record_outcome(cid, 240.0, True, 12.0, 1.5)

    resolved = [o for o in store.observations() if o.touched]
    assert len(resolved) == 1
    assert resolved[0].minutes_to_touch == 12.0


def test_price_history_round_trips(store):
    for i in range(10):
        store.log_price("BTC", 100_000.0 + i, START + timedelta(minutes=i))

    got = store.prices_between("BTC", START, START + timedelta(minutes=5))
    assert len(got) == 6
    assert got[0][1] == 100_000.0
    assert got == sorted(got, key=lambda r: r[0])
