"""Bucketing tests.

The direction convention is the thing most worth guarding. Longs liquidate
BELOW price and produce forced selling; shorts liquidate ABOVE and produce
forced buying. Getting that backwards inverts the entire map while still
producing a plausible-looking picture, which is the worst kind of bug.
"""

import pytest

from liqmap.bucket import (LiquidationMap, Position, build_map,
                           parse_clearinghouse_state, render)

SPOT = 100_000.0


def long_at(liq, notional=1_000_000.0, wallet="0xaaa", coin="BTC", entry=SPOT):
    return Position(wallet=wallet, coin=coin, szi=1.0, entry_px=entry,
                    liquidation_px=liq, position_value=notional, leverage=10)


def short_at(liq, notional=1_000_000.0, wallet="0xbbb", coin="BTC", entry=SPOT):
    return Position(wallet=wallet, coin=coin, szi=-1.0, entry_px=entry,
                    liquidation_px=liq, position_value=notional, leverage=10)


# -- direction convention --------------------------------------------------

def test_longs_land_below_and_are_forced_selling():
    lm = build_map([long_at(SPOT * 0.95)], "BTC", SPOT)
    assert len(lm.buckets) == 1
    b = lm.buckets[0]
    assert b.mid < SPOT
    assert b.long_notional == 1_000_000.0
    assert b.short_notional == 0.0
    assert b.net_notional < 0          # negative = forced selling


def test_shorts_land_above_and_are_forced_buying():
    lm = build_map([short_at(SPOT * 1.05)], "BTC", SPOT)
    b = lm.buckets[0]
    assert b.mid > SPOT
    assert b.short_notional == 1_000_000.0
    assert b.net_notional > 0          # positive = forced buying


def test_impossible_directions_are_rejected():
    """A long liquidating above spot, or a short below it, is stale or bad
    data -- not forward-looking fuel."""
    lm = build_map([long_at(SPOT * 1.05), short_at(SPOT * 0.95)], "BTC", SPOT)
    assert lm.positions_used == 0
    assert lm.positions_skipped == 2


def test_above_and_below_are_ordered_outward_from_spot():
    lm = build_map([short_at(SPOT * 1.01), short_at(SPOT * 1.05),
                    long_at(SPOT * 0.99), long_at(SPOT * 0.95)], "BTC", SPOT)
    above = [b.mid for b in lm.above()]
    below = [b.mid for b in lm.below()]
    assert above == sorted(above)              # nearest first, going up
    assert below == sorted(below, reverse=True)  # nearest first, going down
    assert all(m > SPOT for m in above)
    assert all(m < SPOT for m in below)


# -- bucketing -------------------------------------------------------------

def test_nearby_liquidations_merge_into_one_bucket():
    # 25 bps on 100k is a $250 bin. Bins are anchored at spot, so the one
    # holding these runs [94,750, 95,000).
    lm = build_map([long_at(94_810.0), long_at(94_900.0), long_at(94_990.0)],
                   "BTC", SPOT, bucket_bps=25.0)
    assert len(lm.buckets) == 1
    assert lm.buckets[0].long_count == 3
    assert lm.buckets[0].long_notional == 3_000_000.0


def test_liquidations_across_a_bin_edge_do_not_merge():
    """94,990 and 95,010 are only $20 apart but sit either side of a bin
    boundary. Real behaviour, and worth pinning down so nobody 'fixes' it."""
    lm = build_map([long_at(94_990.0), long_at(95_010.0)],
                   "BTC", SPOT, bucket_bps=25.0)
    assert len(lm.buckets) == 2


def test_narrower_buckets_split_them_apart():
    lm = build_map([long_at(94_810.0), long_at(94_900.0), long_at(94_990.0)],
                   "BTC", SPOT, bucket_bps=1.0)
    assert len(lm.buckets) >= 2


def test_bucket_width_scales_with_price():
    """The same bps setting must work on a $0.30 coin and a $100k one."""
    cheap = build_map([long_at(0.285, coin="DOGE", entry=0.30)],
                      "DOGE", 0.30, bucket_bps=100.0)
    assert cheap.buckets[0].price_high - cheap.buckets[0].price_low == pytest.approx(0.003)

    rich = build_map([long_at(99_000.0)], "BTC", 100_000.0, bucket_bps=100.0)
    assert rich.buckets[0].price_high - rich.buckets[0].price_low == pytest.approx(1000.0)


def test_distant_liquidations_are_excluded():
    lm = build_map([long_at(SPOT * 0.40)], "BTC", SPOT, max_distance_pct=0.25)
    assert lm.positions_used == 0
    assert lm.positions_skipped == 1


def test_positions_without_liquidation_price_are_skipped_not_zeroed():
    p = Position(wallet="0xccc", coin="BTC", szi=1.0, entry_px=SPOT,
                 liquidation_px=None, position_value=5_000_000.0, leverage=1)
    lm = build_map([p], "BTC", SPOT)
    assert lm.positions_used == 0
    assert lm.positions_skipped == 1
    assert lm.total_notional() == 0.0


def test_other_coins_are_ignored_but_wallets_still_counted():
    eth = Position(wallet="0xddd", coin="ETH", szi=1.0, entry_px=3000.0,
                   liquidation_px=2800.0, position_value=100.0, leverage=10)
    lm = build_map([eth, long_at(SPOT * 0.95)], "BTC", SPOT)
    assert lm.positions_used == 1
    assert lm.wallets_seen == 2


def test_min_notional_filter():
    lm = build_map([long_at(SPOT * 0.95, notional=500.0),
                    long_at(SPOT * 0.94, notional=5_000_000.0)],
                   "BTC", SPOT, min_notional_per_position=10_000.0)
    assert lm.positions_used == 1


def test_invalid_configuration_raises():
    with pytest.raises(ValueError):
        build_map([], "BTC", 0.0)
    with pytest.raises(ValueError):
        build_map([], "BTC", SPOT, bucket_bps=0.0)


# -- aggregate views -------------------------------------------------------

def test_clusters_are_ranked_by_notional():
    lm = build_map([long_at(SPOT * 0.99, notional=1e6),
                    long_at(SPOT * 0.97, notional=9e6),
                    short_at(SPOT * 1.02, notional=4e6)], "BTC", SPOT)
    ranked = lm.clusters(top=3)
    assert [round(c.total_notional) for c in ranked] == [9_000_000, 4_000_000, 1_000_000]


def test_concentration_detects_a_spike_versus_a_smear():
    spiky = build_map([long_at(SPOT * 0.95, notional=1e8)]
                      + [long_at(SPOT * (0.90 - i * 0.005), notional=1e4)
                         for i in range(10)], "BTC", SPOT)
    assert spiky.concentration(top=1) > 0.95

    flat = build_map([long_at(SPOT * (0.99 - i * 0.01), notional=1e6)
                      for i in range(20)], "BTC", SPOT)
    assert flat.concentration(top=5) < 0.45


def test_empty_map_is_safe():
    lm = build_map([], "BTC", SPOT)
    assert lm.buckets == []
    assert lm.total_notional() == 0.0
    assert lm.concentration() == 0.0
    assert lm.clusters() == []


# -- parsing ---------------------------------------------------------------

SAMPLE = {
    "assetPositions": [
        {"position": {
            "coin": "ETH", "szi": "0.0335", "entryPx": "2986.3",
            "leverage": {"rawUsd": "-95.06", "type": "isolated", "value": 20},
            "liquidationPx": "2866.26936529", "marginUsed": "4.967826",
            "positionValue": "100.02765", "unrealizedPnl": "-0.0133",
            "returnOnEquity": "-0.0026789", "maxLeverage": 50,
        }},
        {"position": {
            "coin": "BTC", "szi": "-0.5", "entryPx": "99000",
            "leverage": {"type": "cross", "value": 5},
            "liquidationPx": None, "positionValue": "49500",
        }},
        {"position": {"coin": "SOL", "szi": "0"}},
    ]
}


def test_parses_real_response_shape():
    out = parse_clearinghouse_state("0xabc", SAMPLE)
    assert len(out) == 2                    # the zero-size position is dropped

    eth = out[0]
    assert eth.coin == "ETH"
    assert eth.is_long
    assert eth.liquidation_px == pytest.approx(2866.26936529)
    assert eth.leverage == 20
    assert eth.leverage_type == "isolated"
    assert eth.notional == pytest.approx(100.02765)

    btc = out[1]
    assert not btc.is_long
    assert btc.liquidation_px is None       # null preserved, not turned into 0


def test_parsing_survives_junk():
    assert parse_clearinghouse_state("0x", {}) == []
    assert parse_clearinghouse_state("0x", {"assetPositions": None}) == []
    weird = {"assetPositions": [{"position": {"coin": "BTC", "szi": "abc"}}]}
    assert parse_clearinghouse_state("0x", weird) == []


def test_notional_falls_back_to_size_times_entry():
    p = Position(wallet="0x", coin="BTC", szi=2.0, entry_px=50_000.0,
                 liquidation_px=45_000.0, position_value=0.0, leverage=5)
    assert p.notional == pytest.approx(100_000.0)


def test_render_produces_a_readable_map():
    lm = build_map([long_at(SPOT * 0.97, notional=5e6),
                    short_at(SPOT * 1.03, notional=3e6)], "BTC", SPOT)
    text = render(lm)
    assert "<<< spot" in text
    assert "SQ" in text and "CA" in text
    # Overhead levels print above spot, below-levels underneath.
    assert text.index("SQ  ") < text.index("<<< spot") < text.index("CA  ")
