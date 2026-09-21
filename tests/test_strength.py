"""Position strength, fragility, and change detection.

The inversion is the thing to guard: a strong position is weak fuel and a
fragile one is strong fuel. Getting that backwards would weight the map by
exactly the positions that will never fire.
"""

import math

import pytest

from liqmap.bucket import Position, parse_clearinghouse_state
from liqmap.strength import diff_positions, score, score_all, summarise

SPOT = 100_000.0
SIGMA = 0.0006          # 6 bp/min
HORIZON = 240.0
SD = SIGMA * math.sqrt(HORIZON)   # ~0.93% over four hours


def pos(liq, szi=1.0, notional=1_000_000.0, account=10_000_000.0,
        margin=100_000.0, funding=0.0, pnl=0.0, lev=10.0, lev_type="cross",
        wallet="0xa", coin="BTC"):
    return Position(
        wallet=wallet, coin=coin, szi=szi, entry_px=SPOT, liquidation_px=liq,
        position_value=notional, leverage=lev, leverage_type=lev_type,
        unrealized_pnl=pnl, margin_used=margin, funding_since_open=funding,
        account_value=account)


# -- survivability ---------------------------------------------------------

def test_liquidation_distance_is_measured_in_sigma():
    far = score(pos(SPOT * math.exp(-4 * SD)), SPOT, SIGMA, HORIZON)
    assert far.liq_distance_sigmas == pytest.approx(4.0, rel=0.02)

    near = score(pos(SPOT * math.exp(-0.5 * SD)), SPOT, SIGMA, HORIZON)
    assert near.liq_distance_sigmas == pytest.approx(0.5, rel=0.02)


def test_distant_liquidation_survives_better():
    far = score(pos(SPOT * 0.80), SPOT, SIGMA, HORIZON)
    near = score(pos(SPOT * 0.995), SPOT, SIGMA, HORIZON)
    assert far.survivability() > near.survivability()
    assert far.survivability() > 0.7
    assert near.survivability() < 0.5


def test_no_liquidation_price_is_maximally_survivable():
    p = Position("0xa", "BTC", 1.0, SPOT, None, 1e6, 1, account_value=1e7)
    s = score(p, SPOT, SIGMA, HORIZON)
    assert s.liq_distance_sigmas == float("inf")
    assert s.survivability() == pytest.approx(1.0, abs=0.01)
    assert s.fragility() == pytest.approx(0.0, abs=0.01)


def test_heavy_commitment_reduces_survivability():
    light = score(pos(SPOT * 0.90, margin=100_000.0), SPOT, SIGMA, HORIZON)
    heavy = score(pos(SPOT * 0.90, margin=6_000_000.0), SPOT, SIGMA, HORIZON)
    assert heavy.commitment > light.commitment
    assert heavy.survivability() < light.survivability()


def test_paying_funding_reduces_survivability():
    free = score(pos(SPOT * 0.90, funding=0.0), SPOT, SIGMA, HORIZON)
    costly = score(pos(SPOT * 0.90, funding=20_000.0), SPOT, SIGMA, HORIZON)
    assert costly.carry_annualised > free.carry_annualised
    assert costly.survivability() < free.survivability()


def test_receiving_funding_is_a_small_bonus():
    paid = score(pos(SPOT * 0.90, funding=0.0), SPOT, SIGMA, HORIZON)
    earning = score(pos(SPOT * 0.90, funding=-5_000.0), SPOT, SIGMA, HORIZON)
    assert earning.receives_funding
    assert earning.survivability() >= paid.survivability()


def test_isolated_margin_is_more_fragile_than_cross():
    cross = score(pos(SPOT * 0.95, lev_type="cross"), SPOT, SIGMA, HORIZON)
    iso = score(pos(SPOT * 0.95, lev_type="isolated"), SPOT, SIGMA, HORIZON)
    assert iso.is_isolated and not cross.is_isolated
    assert iso.survivability() < cross.survivability()


def test_funding_annualisation_uses_holding_period():
    """The same dollar cost over one hour is a far heavier rate than over a
    week, and the score has to reflect that."""
    fresh = score(pos(SPOT * 0.9, funding=1_000.0), SPOT, SIGMA, HORIZON,
                  hours_held=1.0)
    old = score(pos(SPOT * 0.9, funding=1_000.0), SPOT, SIGMA, HORIZON,
                hours_held=168.0)
    assert fresh.carry_annualised > old.carry_annualised * 100


# -- the inversion ---------------------------------------------------------

def test_fragility_is_the_complement_of_survivability():
    for liq in (SPOT * 0.99, SPOT * 0.95, SPOT * 0.80):
        s = score(pos(liq), SPOT, SIGMA, HORIZON)
        assert s.fragility() == pytest.approx(1.0 - s.survivability(), abs=1e-9)


def test_strong_positions_are_weak_fuel():
    """The point of the whole module. A comfortable position is a level that
    holds, not a level that fires."""
    strong = score(pos(SPOT * 0.75, margin=50_000.0), SPOT, SIGMA, HORIZON)
    fragile = score(pos(SPOT * 0.992, margin=8_000_000.0, funding=50_000.0),
                    SPOT, SIGMA, HORIZON)

    assert strong.survivability() > fragile.survivability()
    assert fragile.fragility() > strong.fragility()
    assert fragile.fragility() > 0.6


def test_scores_stay_in_range_across_extremes():
    for liq in (SPOT * 0.9999, SPOT * 0.5, SPOT * 1.0001, SPOT * 2.0):
        for margin in (0.0, 1e9):
            for funding in (-1e6, 0.0, 1e7):
                s = score(pos(liq, margin=margin, funding=funding),
                          SPOT, SIGMA, HORIZON)
                assert 0.0 <= s.survivability() <= 1.0
                assert 0.0 <= s.fragility() <= 1.0


def test_zero_account_value_does_not_divide_by_zero():
    s = score(pos(SPOT * 0.9, account=0.0), SPOT, SIGMA, HORIZON)
    assert s.commitment == 0.0
    assert s.effective_leverage == 0.0
    assert 0.0 <= s.survivability() <= 1.0


# -- maturity, and why it is a trap ---------------------------------------

def test_maturity_reports_return_on_the_position():
    s = score(pos(SPOT * 0.9, notional=1_000_000.0, pnl=250_000.0),
              SPOT, SIGMA, HORIZON)
    assert s.maturity == pytest.approx(0.25)

    loser = score(pos(SPOT * 0.9, notional=1_000_000.0, pnl=-150_000.0),
                  SPOT, SIGMA, HORIZON)
    assert loser.maturity == pytest.approx(-0.15)


def test_profit_does_not_inflate_survivability_directly():
    """Unrealised PnL is reported, not rewarded. A position deep in profit is
    a worse thing to copy, not a better one, because its move already
    happened -- so it must not quietly raise the score."""
    flat = score(pos(SPOT * 0.9, pnl=0.0), SPOT, SIGMA, HORIZON)
    rich = score(pos(SPOT * 0.9, pnl=5_000_000.0), SPOT, SIGMA, HORIZON)
    assert rich.survivability() == pytest.approx(flat.survivability(), abs=1e-9)


# -- cohort ----------------------------------------------------------------

def test_cohort_separates_raw_from_fragility_weighted():
    """Big comfortable longs and small desperate shorts: raw positioning says
    long, pressure says up."""
    positions = [
        pos(SPOT * 0.70, szi=1.0, notional=10_000_000.0, margin=50_000.0),
        pos(SPOT * 1.004, szi=-1.0, notional=1_000_000.0, margin=9_000_000.0,
            funding=80_000.0),
    ]
    summary = summarise(score_all(positions, SPOT, SIGMA, HORIZON), "BTC", SPOT)

    assert summary.long_notional > summary.short_notional       # raw says long
    assert summary.net_fragile < 0                               # fuel says up
    assert "upside" in summary.render()


def test_cohort_handles_empty_input():
    s = summarise([], "BTC", SPOT)
    assert s.n_positions == 0
    assert "nothing mapped" in s.render()


def test_paying_funding_share_is_counted():
    positions = [pos(SPOT * 0.9, funding=1000.0), pos(SPOT * 0.9, funding=-500.0),
                 pos(SPOT * 0.9, funding=2000.0)]
    s = summarise(score_all(positions, SPOT, SIGMA, HORIZON), "BTC", SPOT)
    assert s.paying_funding_share == pytest.approx(2 / 3)


# -- change detection ------------------------------------------------------

def test_added_and_reduced_are_detected():
    before = [pos(SPOT * 0.9, szi=1.0)]
    assert diff_positions(before, [pos(SPOT * 0.9, szi=1.5)], SPOT)[0].kind == "ADDED"
    assert diff_positions(before, [pos(SPOT * 0.9, szi=0.5)], SPOT)[0].kind == "REDUCED"


def test_tiny_size_wobble_counts_as_held():
    before = [pos(SPOT * 0.9, szi=1.0)]
    after = [pos(SPOT * 0.9, szi=1.005)]
    assert diff_positions(before, after, SPOT)[0].kind == "HELD"


def test_defended_long_moves_liquidation_down():
    """Same size, liquidation pushed further away. They posted margin -- the
    strongest conviction signal available on-chain, and invisible in any
    single snapshot."""
    before = [pos(SPOT * 0.95, szi=1.0)]
    after = [pos(SPOT * 0.85, szi=1.0)]
    c = diff_positions(before, after, SPOT)[0]
    assert c.kind == "DEFENDED"
    assert c.is_bullish_signal


def test_defended_short_moves_liquidation_up():
    before = [pos(SPOT * 1.05, szi=-1.0)]
    after = [pos(SPOT * 1.15, szi=-1.0)]
    assert diff_positions(before, after, SPOT)[0].kind == "DEFENDED"


def test_weakened_long_moves_liquidation_closer():
    before = [pos(SPOT * 0.85, szi=1.0)]
    after = [pos(SPOT * 0.95, szi=1.0)]
    c = diff_positions(before, after, SPOT)[0]
    assert c.kind == "WEAKENED"
    assert not c.is_bullish_signal


def test_weakened_short_moves_liquidation_closer():
    before = [pos(SPOT * 1.15, szi=-1.0)]
    after = [pos(SPOT * 1.05, szi=-1.0)]
    assert diff_positions(before, after, SPOT)[0].kind == "WEAKENED"


def test_liquidation_is_distinguished_from_a_voluntary_close():
    """Both look like a vanished position. The difference is whether spot was
    sitting on the liquidation price when it went."""
    near = [pos(SPOT * 0.999, szi=1.0)]
    assert diff_positions(near, [], SPOT)[0].kind == "LIQUIDATED"

    far = [pos(SPOT * 0.70, szi=1.0)]
    assert diff_positions(far, [], SPOT)[0].kind == "CLOSED"


def test_new_positions_are_flagged_as_opened():
    c = diff_positions([], [pos(SPOT * 0.9)], SPOT)
    assert len(c) == 1 and c[0].kind == "OPENED"


def test_changes_are_matched_per_wallet_and_coin():
    before = [pos(SPOT * 0.9, wallet="0xa", coin="BTC", szi=1.0),
              pos(SPOT * 0.9, wallet="0xb", coin="BTC", szi=1.0)]
    after = [pos(SPOT * 0.9, wallet="0xa", coin="BTC", szi=2.0),
             pos(SPOT * 0.9, wallet="0xb", coin="BTC", szi=1.0)]
    by_wallet = {c.wallet: c.kind for c in diff_positions(before, after, SPOT)}
    assert by_wallet["0xa"] == "ADDED"
    assert by_wallet["0xb"] == "HELD"


def test_empty_comparison_is_safe():
    assert diff_positions([], [], SPOT) == []


# -- parsing the extra fields ---------------------------------------------

PAYLOAD = {
    "marginSummary": {"accountValue": "2500000.5", "totalMarginUsed": "400000"},
    "assetPositions": [
        {"position": {
            "coin": "BTC", "szi": "2.5", "entryPx": "98000",
            "leverage": {"type": "cross", "value": 8},
            "liquidationPx": "91000.5", "marginUsed": "31000.25",
            "positionValue": "248000", "unrealizedPnl": "5000.75",
            "returnOnEquity": "0.16", "maxLeverage": 40,
            "cumFunding": {"allTime": "900.5", "sinceOpen": "420.25",
                           "sinceChange": "12.0"},
        }},
    ],
}


def test_all_the_requested_fields_come_from_one_call():
    """entry, liquidation, unrealised PnL, funding and account value are all
    in the same clearinghouseState response."""
    p = parse_clearinghouse_state("0xabc", PAYLOAD)[0]
    assert p.entry_px == pytest.approx(98_000.0)
    assert p.liquidation_px == pytest.approx(91_000.5)
    assert p.unrealized_pnl == pytest.approx(5_000.75)
    assert p.funding_since_open == pytest.approx(420.25)
    assert p.account_value == pytest.approx(2_500_000.5)
    assert p.margin_used == pytest.approx(31_000.25)
    assert p.max_leverage == 40
    assert p.return_on_equity == pytest.approx(0.16)


def test_missing_margin_summary_defaults_safely():
    p = parse_clearinghouse_state("0x", {"assetPositions": PAYLOAD["assetPositions"]})[0]
    assert p.account_value == 0.0


def test_scoring_a_parsed_position_end_to_end():
    p = parse_clearinghouse_state("0xabc", PAYLOAD)[0]
    s = score(p, 98_500.0, SIGMA, HORIZON)
    assert s.liq_distance_sigmas > 0
    assert 0 < s.commitment < 1
    assert 0.0 <= s.fragility() <= 1.0
    assert "LONG" in s.render()


# -- fragility-weighted map ------------------------------------------------

def test_weighting_the_map_by_fragility_changes_the_picture():
    """A huge comfortable cluster and a small desperate one. Raw notional
    ranks the comfortable one first; fragility ranks the desperate one first,
    and that is the one that can actually fire."""
    from liqmap.bucket import build_map
    from liqmap.strength import fragility_weighter

    # 18% away: inside the map's default 25% window, but ~19 sigma out, so
    # it cannot realistically fire inside the horizon.
    comfortable = pos(SPOT * 0.82, szi=1.0, notional=50_000_000.0,
                      margin=100_000.0, wallet="0xbig")
    desperate = pos(SPOT * 0.992, szi=1.0, notional=2_000_000.0,
                    margin=9_000_000.0, funding=100_000.0, wallet="0xsmall")

    raw = build_map([comfortable, desperate], "BTC", SPOT)
    assert not raw.weighted
    assert raw.clusters(top=1)[0].mid < SPOT * 0.85     # the big one wins

    weighted = build_map([comfortable, desperate], "BTC", SPOT,
                         weight_fn=fragility_weighter(SPOT, SIGMA, HORIZON))
    assert weighted.weighted
    assert weighted.clusters(top=1)[0].mid > SPOT * 0.98  # the fragile one wins
    assert weighted.total_notional() < raw.total_notional()


def test_weighting_shrinks_unreachable_clusters_to_near_nothing():
    from liqmap.bucket import build_map
    from liqmap.strength import fragility_weighter

    far = pos(SPOT * 0.80, notional=10_000_000.0, margin=10_000.0)
    weighted = build_map([far], "BTC", SPOT,
                         weight_fn=fragility_weighter(SPOT, SIGMA, HORIZON))
    assert weighted.total_notional() < 10_000_000.0 * 0.05


def test_render_says_which_map_it_is():
    from liqmap.bucket import build_map, render
    from liqmap.strength import fragility_weighter

    positions = [pos(SPOT * 0.95, notional=5e6)]
    assert "raw notional" in render(build_map(positions, "BTC", SPOT))
    assert "fragility-weighted" in render(
        build_map(positions, "BTC", SPOT,
                  weight_fn=fragility_weighter(SPOT, SIGMA, HORIZON)))
