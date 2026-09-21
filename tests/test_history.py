"""Sweep persistence and change detection over time.

DEFENDED is the whole reason this module exists -- it is defined by a
difference between two sweeps and cannot be observed any other way. These
tests confirm it survives the round trip through storage, which is where a
signal like that quietly gets lost.
"""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from liqmap.bucket import Position
from liqmap.history import History, render_changes

SPOT = 100_000.0
T0 = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)


def pos(liq, szi=1.0, notional=1_000_000.0, wallet="0xa", coin="BTC"):
    return Position(wallet=wallet, coin=coin, szi=szi, entry_px=SPOT,
                    liquidation_px=liq, position_value=notional, leverage=10,
                    account_value=10_000_000.0, margin_used=100_000.0)


@pytest.fixture
def hist():
    with tempfile.TemporaryDirectory() as tmp:
        h = History(Path(tmp) / "h.db")
        yield h
        h.close()


# -- the round trip --------------------------------------------------------

def test_first_sweep_produces_no_changes(hist):
    """Nothing to compare against. Expected, not a failure."""
    sweep_id, changes = hist.record_sweep("BTC", SPOT, [pos(SPOT * 0.9)], ts=T0)
    assert sweep_id
    assert changes == []
    assert hist.counts()["sweeps"] == 1
    assert hist.counts()["sweep_positions"] == 1


def test_positions_survive_storage_intact(hist):
    original = pos(SPOT * 0.9, szi=2.5, notional=2_500_000.0)
    sweep_id, _ = hist.record_sweep("BTC", SPOT, [original], ts=T0)

    restored = hist.sweep_positions(sweep_id)[0]
    assert restored.wallet == original.wallet
    assert restored.szi == original.szi
    assert restored.liquidation_px == original.liquidation_px
    assert restored.account_value == original.account_value
    assert restored.margin_used == original.margin_used


def test_defended_is_detected_across_two_sweeps(hist):
    """The signal this module exists for. Same size, liquidation pushed
    further away -- they posted margin."""
    hist.record_sweep("BTC", SPOT, [pos(SPOT * 0.95)], ts=T0)
    _, changes = hist.record_sweep("BTC", SPOT, [pos(SPOT * 0.85)],
                                   ts=T0 + timedelta(minutes=30))

    assert len(changes) == 1
    assert changes[0].kind == "DEFENDED"
    assert hist.counts()["defended"] == 1

    stored = hist.recent_changes(coin="BTC", kinds=["DEFENDED"], hours=24 * 365)
    assert len(stored) == 1
    assert stored[0]["wallet"] == "0xa"
    assert stored[0]["liq_move_pct"] < 0        # long's liquidation moved down


def test_liquidation_is_recorded_when_spot_sat_on_the_price(hist):
    hist.record_sweep("BTC", SPOT, [pos(SPOT * 0.999)], ts=T0)
    _, changes = hist.record_sweep("BTC", SPOT, [], ts=T0 + timedelta(minutes=5))

    assert changes[0].kind == "LIQUIDATED"
    assert hist.counts()["liquidated"] == 1


def test_voluntary_close_is_not_counted_as_a_liquidation(hist):
    hist.record_sweep("BTC", SPOT, [pos(SPOT * 0.70)], ts=T0)
    _, changes = hist.record_sweep("BTC", SPOT, [], ts=T0 + timedelta(minutes=5))

    assert changes[0].kind == "CLOSED"
    assert hist.counts()["liquidated"] == 0


def test_held_positions_are_not_stored_as_changes(hist):
    """HELD is the overwhelming majority and carries nothing. Storing it would
    bury the signal."""
    hist.record_sweep("BTC", SPOT, [pos(SPOT * 0.9)], ts=T0)
    _, changes = hist.record_sweep("BTC", SPOT, [pos(SPOT * 0.9)],
                                   ts=T0 + timedelta(minutes=30))

    assert [c.kind for c in changes] == ["HELD"]
    assert hist.counts()["changes"] == 0


def test_each_sweep_diffs_against_the_previous_one_only(hist):
    """A three-step sequence must produce two transitions, not a comparison
    against the original."""
    hist.record_sweep("BTC", SPOT, [pos(SPOT * 0.95, szi=1.0)], ts=T0)
    hist.record_sweep("BTC", SPOT, [pos(SPOT * 0.95, szi=2.0)],
                      ts=T0 + timedelta(minutes=10))
    _, changes = hist.record_sweep("BTC", SPOT, [pos(SPOT * 0.95, szi=1.0)],
                                   ts=T0 + timedelta(minutes=20))

    assert changes[0].kind == "REDUCED"
    kinds = [c["kind"] for c in hist.recent_changes(hours=24 * 365)]
    assert sorted(kinds) == ["ADDED", "REDUCED"]


def test_coins_are_tracked_independently(hist):
    hist.record_sweep("BTC", SPOT, [pos(SPOT * 0.9, coin="BTC")], ts=T0)
    hist.record_sweep("ETH", 3000.0, [pos(2700.0, coin="ETH")], ts=T0)

    btc = hist.latest_sweep_positions("BTC")
    eth = hist.latest_sweep_positions("ETH")
    assert btc and eth
    assert btc[1][0].coin == "BTC"
    assert eth[1][0].coin == "ETH"


def test_positions_for_other_coins_are_filtered_out(hist):
    mixed = [pos(SPOT * 0.9, coin="BTC"), pos(2700.0, coin="ETH")]
    sweep_id, _ = hist.record_sweep("BTC", SPOT, mixed, ts=T0)
    assert len(hist.sweep_positions(sweep_id)) == 1


def test_multiple_wallets_diff_independently(hist):
    before = [pos(SPOT * 0.95, wallet="0xa"), pos(SPOT * 0.95, wallet="0xb")]
    after = [pos(SPOT * 0.85, wallet="0xa"), pos(SPOT * 0.95, wallet="0xb", szi=2.0)]

    hist.record_sweep("BTC", SPOT, before, ts=T0)
    _, changes = hist.record_sweep("BTC", SPOT, after, ts=T0 + timedelta(minutes=30))

    by_wallet = {c.wallet: c.kind for c in changes}
    assert by_wallet["0xa"] == "DEFENDED"
    assert by_wallet["0xb"] == "ADDED"


# -- conviction flow -------------------------------------------------------

def test_conviction_flow_separates_commitment_from_retreat(hist):
    before = [pos(SPOT * 0.95, wallet=f"0x{i}", notional=1_000_000.0)
              for i in range(4)]
    after = [
        pos(SPOT * 0.85, wallet="0x0", notional=1_000_000.0),   # DEFENDED
        pos(SPOT * 0.95, wallet="0x1", szi=2.0, notional=2_000_000.0),  # ADDED
        pos(SPOT * 0.95, wallet="0x2", szi=0.3, notional=300_000.0),    # REDUCED
        pos(SPOT * 0.99, wallet="0x3", notional=1_000_000.0),   # WEAKENED
    ]
    hist.record_sweep("BTC", SPOT, before, ts=T0)
    hist.record_sweep("BTC", SPOT, after, ts=T0 + timedelta(minutes=30))

    flow = hist.conviction_flow("BTC", hours=24 * 365)
    assert flow["committing_notional"] > 0
    assert flow["retreating_notional"] > 0
    assert set(flow["by_kind"]) == {"DEFENDED", "ADDED", "REDUCED", "WEAKENED"}


def test_liquidations_are_kept_out_of_the_conviction_tally(hist):
    """A forced exit was not a decision. Counting it as one misreads the
    tape in exactly the wrong direction."""
    hist.record_sweep("BTC", SPOT, [pos(SPOT * 0.999, notional=5_000_000.0)], ts=T0)
    hist.record_sweep("BTC", SPOT, [], ts=T0 + timedelta(minutes=5))

    flow = hist.conviction_flow("BTC", hours=24 * 365)
    assert flow["liquidated_count"] == 1
    assert flow["liquidated_notional"] == pytest.approx(5_000_000.0)
    assert flow["retreating_notional"] == 0
    assert flow["net_conviction"] == 0


def test_conviction_flow_on_empty_history(hist):
    flow = hist.conviction_flow("BTC")
    assert flow["by_kind"] == {}
    assert flow["net_conviction"] == 0


# -- queries and housekeeping ---------------------------------------------

def test_recent_changes_respects_the_time_window(hist):
    old = datetime.now(timezone.utc) - timedelta(hours=48)
    hist.record_sweep("BTC", SPOT, [pos(SPOT * 0.95)], ts=old)
    hist.record_sweep("BTC", SPOT, [pos(SPOT * 0.85)],
                      ts=old + timedelta(minutes=30))

    assert hist.recent_changes(hours=1) == []
    assert len(hist.recent_changes(hours=72)) == 1


def test_changes_can_be_filtered_by_kind(hist):
    before = [pos(SPOT * 0.95, wallet="0xa"), pos(SPOT * 0.95, wallet="0xb")]
    after = [pos(SPOT * 0.85, wallet="0xa"), pos(SPOT * 0.95, wallet="0xb", szi=3.0)]
    hist.record_sweep("BTC", SPOT, before, ts=T0)
    hist.record_sweep("BTC", SPOT, after, ts=T0 + timedelta(minutes=5))

    only = hist.recent_changes(kinds=["DEFENDED"], hours=24 * 365)
    assert len(only) == 1 and only[0]["kind"] == "DEFENDED"


def test_pruning_drops_old_sweeps_but_keeps_changes(hist):
    old = datetime.now(timezone.utc) - timedelta(days=60)
    hist.record_sweep("BTC", SPOT, [pos(SPOT * 0.95)], ts=old)
    hist.record_sweep("BTC", SPOT, [pos(SPOT * 0.85)],
                      ts=old + timedelta(minutes=30))

    before_changes = hist.counts()["changes"]
    removed = hist.prune(keep_days=30)

    assert removed == 2
    assert hist.counts()["sweeps"] == 0
    assert hist.counts()["sweep_positions"] == 0
    assert hist.counts()["changes"] == before_changes   # history retained


def test_sweep_listing_is_newest_first(hist):
    for i in range(3):
        hist.record_sweep("BTC", SPOT, [pos(SPOT * 0.9)],
                          ts=T0 + timedelta(minutes=i * 10))
    listed = hist.sweeps("BTC")
    assert len(listed) == 3
    assert listed[0]["ts"] > listed[-1]["ts"]


# -- rendering -------------------------------------------------------------

def test_render_puts_liquidations_and_defends_first(hist):
    changes = [
        {"ts": T0.isoformat(), "kind": "REDUCED", "wallet": "0xr",
         "prev_notional": 9e9, "new_notional": 0, "size_change_pct": -0.5,
         "liq_move_pct": 0},
        {"ts": T0.isoformat(), "kind": "DEFENDED", "wallet": "0xd",
         "prev_notional": 1e5, "new_notional": 1e5, "size_change_pct": 0,
         "liq_move_pct": -0.05},
        {"ts": T0.isoformat(), "kind": "LIQUIDATED", "wallet": "0xl",
         "prev_notional": 1e4, "new_notional": 0, "size_change_pct": -1,
         "liq_move_pct": 0},
    ]
    text = render_changes(changes)
    # Ranked by importance, not by size -- the huge REDUCED goes last.
    assert text.index("LIQUIDATED") < text.index("DEFENDED") < text.index("REDUCED")


def test_render_handles_nothing():
    assert "no position changes" in render_changes([])
