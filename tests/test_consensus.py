"""Consensus tests.

The field that has to be right is `entry_gap_pct`. It is the difference
between "80% of winners are long" and "80% of winners are long and they got in
4% cheaper than you can". Get its sign wrong and the tool tells people the
opposite of the truth about their entry.
"""

import math

import pytest

from liqmap.bucket import Position
from liqmap.consensus import build

SPOT = 100_000.0
SIGMA = 0.0006
HORIZON = 240.0


def pos(is_long=True, entry=SPOT, pnl=0.0, notional=1_000_000.0, liq=None,
        wallet="0xa", account=10_000_000.0, lev=10.0, coin="BTC"):
    if liq is None:
        liq = SPOT * 0.90 if is_long else SPOT * 1.10
    return Position(
        wallet=wallet, coin=coin, szi=1.0 if is_long else -1.0,
        entry_px=entry, liquidation_px=liq, position_value=notional,
        leverage=lev, leverage_type="cross", unrealized_pnl=pnl,
        margin_used=notional / lev, account_value=account)


def view(positions, **kw):
    return build(positions, "BTC", SPOT, SIGMA, HORIZON, **kw)


# -- direction -------------------------------------------------------------

def test_winning_longs_produce_a_long_consensus():
    positions = [
        pos(True, entry=95_000, pnl=50_000, wallet="0xa"),
        pos(True, entry=96_000, pnl=40_000, wallet="0xb"),
        pos(False, entry=99_000, pnl=-10_000, wallet="0xc"),
    ]
    v = view(positions)
    assert v.direction == "long"
    assert v.strength == 1.0          # every WINNING dollar is long
    assert v.n_winning == 2 and v.n_losing == 1


def test_winning_shorts_produce_a_short_consensus():
    positions = [
        pos(False, entry=105_000, pnl=50_000, wallet="0xa"),
        pos(False, entry=104_000, pnl=40_000, wallet="0xb"),
        pos(True, entry=101_000, pnl=-10_000, wallet="0xc"),
    ]
    v = view(positions)
    assert v.direction == "short"
    assert v.strength == 1.0


def test_consensus_is_weighted_by_notional_not_headcount():
    """One whale on one side outweighs several small accounts on the other."""
    positions = [
        pos(True, entry=95_000, pnl=1_000_000, notional=50_000_000, wallet="0xbig"),
        pos(False, entry=101_000, pnl=1_000, notional=100_000, wallet="0xs1"),
        pos(False, entry=101_000, pnl=1_000, notional=100_000, wallet="0xs2"),
        pos(False, entry=101_000, pnl=1_000, notional=100_000, wallet="0xs3"),
    ]
    v = view(positions)
    assert v.direction == "long"
    assert v.strength > 0.95


def test_even_split_is_reported_as_split():
    positions = [
        pos(True, entry=99_000, pnl=10_000, notional=1_000_000, wallet="0xa"),
        pos(False, entry=101_000, pnl=10_000, notional=1_000_000, wallet="0xb"),
    ]
    v = view(positions)
    assert v.direction == "split"


def test_winners_only_filter_changes_the_answer():
    """Lots of losing longs, a few winning shorts. Who is 'winning' and who is
    'positioned' are different questions."""
    positions = [
        pos(True, entry=105_000, pnl=-500_000, notional=10_000_000, wallet="0xl1"),
        pos(True, entry=106_000, pnl=-600_000, notional=10_000_000, wallet="0xl2"),
        pos(False, entry=105_000, pnl=200_000, notional=2_000_000, wallet="0xs1"),
    ]
    assert view(positions, winners_only=True).direction == "short"
    assert view(positions, winners_only=False).direction == "long"


# -- the entry gap, which is the point ------------------------------------

def test_long_that_entered_lower_means_you_enter_worse():
    v = view([pos(True, entry=95_000, pnl=50_000)])
    # Price is ~5.3% above their entry, so copying now is 5.3% worse.
    assert v.entry_gap_pct == pytest.approx((100_000 - 95_000) / 95_000, rel=1e-6)
    assert v.entry_gap_pct > 0
    assert v.late


def test_short_that_entered_higher_also_means_you_enter_worse():
    """Sign convention: the gap reads the same way whichever side it is."""
    v = view([pos(False, entry=105_000, pnl=50_000)])
    assert v.entry_gap_pct == pytest.approx((105_000 - 100_000) / 105_000, rel=1e-6)
    assert v.entry_gap_pct > 0
    assert v.late


def test_entering_at_their_price_is_a_zero_gap():
    v = view([pos(True, entry=SPOT, pnl=1.0)])
    assert v.entry_gap_pct == pytest.approx(0.0, abs=1e-9)
    assert not v.late


def test_negative_gap_means_you_would_enter_better():
    """A long underwater: price below their entry. You'd get in cheaper than
    they did, which is a very different situation from copying a winner."""
    v = view([pos(True, entry=105_000, pnl=-40_000)], winners_only=False)
    assert v.entry_gap_pct < 0


def test_entry_gap_is_notional_weighted():
    positions = [
        pos(True, entry=90_000, pnl=100_000, notional=9_000_000, wallet="0xbig"),
        pos(True, entry=99_900, pnl=100, notional=1_000, wallet="0xtiny"),
    ]
    v = view(positions)
    # Dominated by the big position's ~11% gap, not averaged evenly with the tiny one.
    assert v.entry_gap_pct > 0.10


def test_verdict_calls_out_a_late_entry():
    v = view([pos(True, entry=90_000, pnl=500_000, wallet=f"0x{i}") for i in range(4)])
    text = v.verdict()
    assert "worse than their average entry" in text
    assert "not transferable" in text


# -- crowding --------------------------------------------------------------

def test_crowded_flag_needs_agreement_and_no_room():
    """Everyone one way, nobody with margin to survive a move against it."""
    positions = [pos(True, entry=99_000, pnl=10_000, liq=SPOT * 0.995,
                     wallet=f"0x{i}") for i in range(5)]
    v = view(positions)
    assert v.strength >= 0.75
    assert v.room_sigmas < 1.5
    assert v.crowded
    assert "CROWDED" in v.verdict()


def test_strong_agreement_with_room_is_not_crowded():
    positions = [pos(True, entry=99_000, pnl=10_000, liq=SPOT * 0.75,
                     wallet=f"0x{i}") for i in range(5)]
    v = view(positions)
    assert v.strength >= 0.75
    assert v.room_sigmas > 3
    assert not v.crowded


def test_room_is_the_median_not_the_mean():
    """One trader with enormous room must not make a fragile cohort look safe."""
    positions = [pos(True, entry=99_000, pnl=1_000, liq=SPOT * 0.999,
                     wallet=f"0x{i}") for i in range(4)]
    positions.append(pos(True, entry=99_000, pnl=1_000, liq=SPOT * 0.50,
                         wallet="0xsafe"))
    v = view(positions)
    assert v.room_sigmas < 2


# -- filters and edges ----------------------------------------------------

def test_min_notional_filters_out_dust():
    positions = [
        pos(True, entry=95_000, pnl=1.0, notional=100.0, wallet="0xdust"),
        pos(False, entry=105_000, pnl=1_000, notional=5_000_000, wallet="0xreal"),
    ]
    v = view(positions, min_notional=10_000)
    assert v.n_traders == 1
    assert v.direction == "short"


def test_min_account_filters_small_accounts():
    positions = [
        pos(True, entry=95_000, pnl=1_000, account=5_000, wallet="0xsmall"),
        pos(False, entry=105_000, pnl=1_000, account=50_000_000, wallet="0xwhale"),
    ]
    v = view(positions, min_account=1_000_000)
    assert v.n_traders == 1
    assert v.direction == "short"


def test_other_coins_are_excluded():
    positions = [
        pos(True, entry=95_000, pnl=1_000, coin="BTC"),
        pos(True, entry=2_900, pnl=1_000, coin="ETH", wallet="0xeth"),
    ]
    assert view(positions).n_traders == 1


def test_no_positions_is_safe():
    v = view([])
    assert v.n_traders == 0
    assert v.direction == "split"
    assert "no positions" in v.render()


def test_too_few_winners_refuses_to_call_a_direction():
    v = view([pos(True, entry=95_000, pnl=1_000)])
    assert "Not enough winning positions" in v.verdict()


def test_rows_are_returned_largest_first():
    positions = [
        pos(True, entry=99_000, pnl=1_000, notional=1_000_000, wallet="0xs"),
        pos(True, entry=99_000, pnl=1_000, notional=9_000_000, wallet="0xl"),
    ]
    v = view(positions)
    assert [r.wallet for r in v.rows] == ["0xl", "0xs"]
    assert all(r.entry_gap_pct > 0 for r in v.rows)


def test_render_shows_the_gap_prominently():
    v = view([pos(True, entry=92_000, pnl=100_000, wallet=f"0x{i}") for i in range(4)])
    text = v.render()
    assert "your entry gap" in text
    assert "you enter worse" in text
