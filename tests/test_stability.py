"""What holds book and price together: age, participation, hysteresis, volume.

The problem these solve, in Chris's words: the book says down and price says
down at the same time, he goes to take it, and one of them flips to flat.

Three distinct causes, three fixes:

FAKE FLATNESS. A score sitting at 0.14 against a 0.15 threshold flips on
noise that is not a market event. Hysteresis makes a direction harder to
leave than it was to enter.

NEWBORN AGREEMENTS. An agreement that appeared on this tick and one that has
held ten seconds are not the same thing, and a snapshot cannot tell them
apart. The tracker gives agreement an age.

VACUUM MOVES. Bids PULLED and bids HIT look identical to price and to the
book. Only volume separates them, and the pulled one dissolves within
seconds by construction. Participation is the difference.
"""

import pytest

from liqmap.confirm import (
    AgreementTracker, CandleAction, Participation, confirm, sticky,
)
from liqmap.volume import VolumeProfile, bucket_size, merge


def rising(strength=1.0, bar=100.0):
    m = bar * 0.5 * strength
    return CandleAction(thrust_bps=m, position=0.5 + 0.45 * strength,
                        slope_bps=m, extending="up", bar_range_bps=bar)


def paid(effort=1.5, aligned=1.0):
    return Participation(effort=effort, aligned=aligned, notional=5e5,
                         window_s=30.0)


# --------------------------------------------------------------------------
# hysteresis
# --------------------------------------------------------------------------

def test_a_direction_is_harder_to_leave_than_to_enter():
    assert sticky(0.20, None) == "up"          # crosses to take a side
    assert sticky(0.10, "up") == "up"          # dips, but holds it
    assert sticky(0.05, "up") == "flat"        # falls below the exit band


def test_hysteresis_kills_boundary_chatter():
    """A score hovering either side of the entry threshold used to flip
    every read, and every flip broke an agreement that never broke."""
    scores = [0.16, 0.14, 0.16, 0.13, 0.17, 0.14]
    prev, flips = None, 0
    for s in scores:
        now = sticky(s, prev)
        if now != prev and prev is not None:
            flips += 1
        prev = now
    assert flips == 0, "the reading is still chattering at the boundary"


def test_without_hysteresis_the_same_series_chatters():
    """The control: proves the fixture would flip under a plain threshold,
    so the test above is measuring the fix and not the fixture."""
    plain = ["up" if s > 0.15 else "flat" for s in
             (0.16, 0.14, 0.16, 0.13, 0.17, 0.14)]
    assert len(set(plain)) > 1


def test_a_real_reversal_still_gets_through():
    assert sticky(-0.4, "up") == "down"


# --------------------------------------------------------------------------
# agreement age
# --------------------------------------------------------------------------

def test_an_unchanged_state_accumulates_time():
    t = AgreementTracker()
    assert t.observe("agree", "up", 100.0) == (0.0, 0)
    held, flips = t.observe("agree", "up", 112.0)
    assert held == pytest.approx(12.0) and flips == 0


def test_a_changed_state_resets_the_clock_and_counts_a_flip():
    t = AgreementTracker()
    t.observe("agree", "up", 100.0)
    t.observe("no", "flat", 105.0)
    held, flips = t.observe("agree", "up", 106.0)
    assert held == pytest.approx(0.0)
    assert flips == 2


def test_old_flips_fall_out_of_the_window():
    t = AgreementTracker(flip_window_s=30.0)
    for i in range(6):
        t.observe("agree" if i % 2 else "no", "up", 100.0 + i)
    assert t.observe("agree", "up", 400.0)[1] == 0


def test_the_tracker_applies_hysteresis_to_both_sides():
    t = AgreementTracker()
    assert t.classify(0.2, 0.2) == ("up", "up")
    assert t.classify(0.10, 0.10) == ("up", "up")     # held, not flipped
    assert t.classify(0.02, 0.02) == ("flat", "flat")


def test_reset_clears_everything():
    t = AgreementTracker()
    t.classify(0.5, 0.5)
    t.observe("agree", "up", 100.0)
    t.reset()
    assert t.observe("agree", "up", 200.0) == (0.0, 0)


# --------------------------------------------------------------------------
# participation — the glue
# --------------------------------------------------------------------------

def test_a_move_on_no_volume_is_not_backed():
    """Bids PULLED, not hit. Nothing was paid for, so there is nothing
    holding the agreement together."""
    p = Participation(effort=0.1, aligned=0.9, notional=1e3, window_s=30.0)
    assert p.backed is False
    assert p.supports("up") is False
    assert "not paid for" in p.describe()


def test_volume_on_the_wrong_side_does_not_support_the_call():
    p = Participation(effort=2.0, aligned=-0.8, notional=1e6, window_s=30.0)
    assert p.backed is True
    assert p.supports("up") is False
    assert "wrong side" in p.describe("up")
    assert p.supports("down") is True


def test_a_paid_for_move_supports_its_direction():
    assert paid().supports("up") is True
    assert paid(aligned=-1.0).supports("down") is True


def test_participation_never_supports_a_flat_call():
    assert paid().supports("flat") is False


# --------------------------------------------------------------------------
# settled vs merely agreeing
# --------------------------------------------------------------------------

def test_agreeing_is_not_the_same_as_settled():
    fresh = confirm("up", 0.9, rising(), held_s=0.5, flips=0,
                    participation=paid())
    assert fresh.agree is True
    assert fresh.settled is False
    assert "evaporates" in fresh.instability()


def test_a_held_paid_for_agreement_is_backed():
    c = confirm("up", 0.9, rising(), held_s=30.0, flips=0,
                participation=paid())
    assert c.agree and c.settled and c.backed
    assert c.instability() == ""


def test_a_chopping_market_is_never_settled():
    c = confirm("up", 0.9, rising(), held_s=60.0, flips=9,
                participation=paid())
    assert c.agree is True and c.settled is False
    assert "chopping" in c.instability()


def test_a_held_agreement_nobody_paid_for_is_not_backed():
    c = confirm("up", 0.9, rising(), held_s=30.0, flips=0,
                participation=Participation(0.1, 0.9, 1e3, 30.0))
    assert c.settled is True
    assert c.backed is False
    assert "not paid for" in c.instability()


def test_missing_participation_does_not_block_a_settled_agreement():
    """Absent evidence must not read as evidence against."""
    c = confirm("up", 0.9, rising(), held_s=30.0, flips=0)
    assert c.backed is True


def test_the_dict_carries_stability_for_the_panel():
    d = confirm("up", 0.9, rising(), held_s=2.0, flips=1,
                participation=paid()).to_dict()
    for key in ("settled", "backed", "held_s", "flips", "instability",
                "participation"):
        assert key in d


# --------------------------------------------------------------------------
# the gate, through suggest()
# --------------------------------------------------------------------------

def _suggest(**kw):
    from tests.test_suggest import bars, book_at, read_at
    from liqmap.suggest import suggest

    kw.setdefault("mode", "scalp")
    kw.setdefault("action", rising())
    kw.setdefault("recent", bars(rng=2.0))
    return suggest(read_at(0.9), book_at(), coin="BTC", interval="15m",
                   interval_s=900.0, seconds_left=600.0,
                   notional=10_000.0, **kw)


def test_scalp_mode_refuses_a_fresh_agreement():
    from liqmap.suggest import NoTrade

    out = _suggest(held_s=0.5, participation=paid())
    assert isinstance(out, NoTrade) and out.gate == "unsettled"
    assert "evaporates" in out.detail


def test_scalp_mode_refuses_an_unpaid_move():
    from liqmap.suggest import NoTrade

    out = _suggest(held_s=30.0,
                   participation=Participation(0.1, 0.9, 1e3, 30.0))
    assert isinstance(out, NoTrade) and out.gate == "unsettled"
    assert "not paid for" in out.detail


def test_scalp_mode_takes_a_settled_paid_for_agreement():
    from liqmap.suggest import Suggestion

    out = _suggest(held_s=30.0, participation=paid())
    assert isinstance(out, Suggestion)


def test_range_mode_does_not_apply_the_scalp_stability_gate():
    """A range trade is held for much of the bar; a two-second wobble in the
    agreement is not the same problem it is for a scalp."""
    from liqmap.suggest import Suggestion

    out = _suggest(mode="range", held_s=0.1, participation=paid())
    assert isinstance(out, Suggestion)


# --------------------------------------------------------------------------
# volume at price
# --------------------------------------------------------------------------

def test_buckets_scale_with_the_instrument():
    """The same code has to read a $4 coin and a $100k one."""
    assert bucket_size(4.0) < bucket_size(100_000.0)
    assert bucket_size(0.0) == 0.0


def test_volume_position_separates_two_identical_looking_bars():
    """The whole point. Both close at the high; only one had business done
    up there."""
    drift = VolumeProfile()
    for _ in range(50):
        drift.add(100.0, 1000.0, "buy")      # all volume at the low
    accept = VolumeProfile()
    for _ in range(50):
        accept.add(101.0, 1000.0, "buy")     # all volume at the high

    assert drift.position(101.0) > 0.9       # price is above all the volume
    assert accept.position(101.0) < 0.1      # price is where volume traded


def test_the_point_of_control_is_where_most_business_was_done():
    p = VolumeProfile()
    p.add(100.0, 100.0, "buy")
    p.add(101.0, 900.0, "sell")
    p.add(102.0, 200.0, "buy")
    assert p.poc == pytest.approx(101.0, abs=0.01)


def test_volume_ahead_distinguishes_air_from_a_defended_level():
    """Thin book above with nothing traded up there is air. Thin book above
    with heavy volume up there is a level that already got eaten."""
    air = VolumeProfile()
    for _ in range(20):
        air.add(100.0, 1000.0, "buy")
    assert air.ahead_ratio(100.0, 101.0) < 0.05

    defended = VolumeProfile()
    for _ in range(10):
        defended.add(100.0, 1000.0, "buy")
        defended.add(100.5, 1000.0, "sell")
    assert defended.ahead_ratio(100.0, 101.0) > 0.4


def test_ahead_works_downward_too():
    p = VolumeProfile()
    p.add(100.0, 1000.0, "buy")
    p.add(99.5, 1000.0, "sell")
    assert p.ahead_ratio(100.0, 99.0) > 0.4


def test_delta_at_price_is_signed_the_obvious_way():
    p = VolumeProfile()
    p.add(100.0, 900.0, "buy")
    p.add(100.0, 300.0, "sell")
    assert p.delta_at(100.0) == pytest.approx(600.0)


def test_an_empty_profile_is_neutral_rather_than_a_divide_by_zero():
    p = VolumeProfile()
    assert p.position(100.0) == 0.5
    assert p.poc is None
    assert p.ahead_ratio(100.0, 101.0) == 0.0
    assert p.value_area() is None
    assert "no volume" in p.describe(100.0)


def test_rubbish_fills_are_ignored():
    p = VolumeProfile()
    for px, n in ((0.0, 100.0), (-1.0, 100.0), (100.0, 0.0), (100.0, -5.0)):
        p.add(px, n, "buy")
    assert p.total_notional == 0.0


def test_the_value_area_holds_most_of_the_volume_around_the_poc():
    p = VolumeProfile()
    for px, n in ((99.0, 50.0), (100.0, 400.0), (100.5, 300.0),
                  (101.0, 100.0), (102.0, 50.0)):
        p.add(px, n, "buy")
    lo, hi = p.value_area(0.70)
    assert lo <= p.poc <= hi
    assert hi > lo


def test_profiles_merge_for_a_multi_bar_view():
    a, b = VolumeProfile(), VolumeProfile()
    a.add(100.0, 500.0, "buy")
    b.add(100.0, 500.0, "sell")
    m = merge([a, b])
    assert m.total_notional == pytest.approx(1000.0)
    assert m.at(100.0) == pytest.approx(1000.0)


def test_the_live_candle_builds_a_profile_from_the_same_fills():
    """The data was already arriving and being thrown away."""
    from liqmap.flow import Trade
    from liqmap.live import CandleBuilder

    import time as _t

    # Timestamps near now, or `candle()` rolls the bar forward to the
    # present and hands back a fresh, empty one.
    #
    # Pinned a safe distance INSIDE the current bar rather than taken raw.
    # A raw clock puts the 20-second window across a bar boundary roughly
    # every 45th run, the early trades land in the previous bar, and the
    # test fails with a count three short of what it asked for -- which
    # reads like a bug in the builder and is a bug in the test.
    now = (_t.time() // 900) * 900 + 100.0
    b = CandleBuilder(900.0)
    for i in range(20):
        b.add(Trade(px=100.0 + i * 0.01, sz=1.0, aggressor="buy",
                    ts=now - 20 + i))
    c = b.candle(now)
    assert c.profile.total_notional > 0
    assert c.profile.poc is not None
    assert c.profile.trades == 20
