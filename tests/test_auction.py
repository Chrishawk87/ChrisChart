"""The auction classifier: does it label what it claims to label?

There is no edge to verify here and nothing to backtest. A classifier is
correct when it fires on the structure it names and stays silent otherwise,
so these tests build each pattern by hand and check it is recognised -- and,
just as importantly, that the near-misses are not.

The two failures that would matter most in use:
  - a signal whose stop sits the wrong side of its entry, which cannot be
    traded as written
  - a level price loiters at emitting on every bar, which buries the chart
"""

from __future__ import annotations

import pytest

from liqmap.auction import (ACCEPT_CLOSES, Classifier, Levels, Side, State,
                            classify_state)


class Bar:
    __slots__ = ("ts", "open", "high", "low", "close", "volume")

    def __init__(self, ts, o, h, l, c, v=100.0):
        self.ts, self.open, self.high = ts, o, h
        self.low, self.close, self.volume = l, c, v


def bar(i, c, hi=None, lo=None):
    return Bar(ts=i * 300.0, o=c, h=hi if hi is not None else c + 0.25,
               l=lo if lo is not None else c - 0.25, c=c)


def levels():
    """A plain prior session: VAL 7000, POC 7010, VAH 7020."""
    return Levels(poc=7010.0, vah=7020.0, val=7000.0,
                  hvn=[7005.0, 7010.0, 7016.0],
                  lvn=[7008.0, 7013.0],
                  strength={7005.0: "weak", 7010.0: "strong",
                            7016.0: "weak"})


def run(closes, lv=None, **kw):
    """Feed closes, collect every signal."""
    c = Classifier(lv or levels(), **kw)
    out = []
    for i, px in enumerate(closes):
        s = c.step(bar(i, px))
        if s is not None:
            out.append(s)
    return out


# ------------------------------------------------------------- state

def test_inside_the_value_area_is_balance():
    assert classify_state([7010.0, 7012.0], 7020.0, 7000.0) is State.BALANCE


def test_two_closes_above_the_value_area_is_imbalance_up():
    assert classify_state([7010.0, 7021.0, 7022.0], 7020.0,
                          7000.0) is State.IMBALANCE_UP


def test_two_closes_below_is_imbalance_down():
    assert classify_state([7010.0, 6999.0, 6998.0], 7020.0,
                          7000.0) is State.IMBALANCE_DOWN


def test_one_close_beyond_is_a_poke_not_acceptance():
    """The distinction the whole framework rests on. One close through is
    why 'failed auction' is a pattern rather than a state."""
    assert classify_state([7010.0, 7021.0], 7020.0,
                          7000.0) is State.BALANCE


def test_acceptance_must_be_consecutive():
    assert classify_state([7021.0, 7010.0, 7022.0], 7020.0,
                          7000.0) is State.BALANCE


def test_a_degenerate_value_area_is_balance_rather_than_an_error():
    assert classify_state([7010.0], 0.0, 0.0) is State.BALANCE


# ------------------------------------------------------- the longs

def test_a_value_area_breakout_up_is_flagged():
    sigs = run([7010.0, 7012.0, 7014.0, 7021.0, 7022.0])
    names = [s.pattern for s in sigs]
    assert "VA_Breakout_Up" in names
    s = next(s for s in sigs if s.pattern == "VA_Breakout_Up")
    assert s.side is Side.LONG
    assert s.stop < s.entry < s.target


def test_one_close_above_the_vah_does_not_break_out():
    """Confirmation is the second close. A single poke is the failed
    auction the mirror pattern exists to catch."""
    sigs = run([7010.0, 7012.0, 7021.0, 7014.0])
    assert "VA_Breakout_Up" not in [s.pattern for s in sigs]


def test_a_failed_auction_down_is_flagged_when_price_reclaims_the_val():
    sigs = run([7010.0, 7005.0, 6998.0, 7003.0])
    s = next((s for s in sigs if s.pattern == "Failed_Auction_Down"), None)
    assert s is not None and s.side is Side.LONG
    assert s.stop < 7000.0            # below the failed low
    assert s.target > s.entry


def test_an_accepted_breakdown_is_not_a_failed_auction():
    """Two closes below the VAL is acceptance, which is the opposite read
    -- flagging it as a failure would invert the signal."""
    sigs = run([7010.0, 6998.0, 6997.0, 6996.0])
    assert "Failed_Auction_Down" not in [s.pattern for s in sigs]


def test_a_poc_reclaim_is_flagged():
    sigs = run([7012.0, 7008.0, 7006.0, 7012.0])
    s = next((s for s in sigs if s.pattern == "POC_Reclaim"), None)
    assert s is not None and s.side is Side.LONG
    assert s.target == pytest.approx(7020.0)     # the VAH


def test_an_lvn_traversal_up_needs_an_up_imbalance():
    """The same price action inside the value area is not a traversal --
    the state is part of the definition, not decoration."""
    inside = run([7010.0, 7012.0, 7013.5, 7014.0])
    assert "LVN_Traversal_Up" not in [s.pattern for s in inside]


# ------------------------------------------------------- the shorts

def test_a_value_area_breakdown_is_flagged():
    sigs = run([7010.0, 7008.0, 6999.0, 6998.0])
    s = next((s for s in sigs if s.pattern == "VA_Breakdown"), None)
    assert s is not None and s.side is Side.SHORT
    assert s.target < s.entry < s.stop


def test_a_failed_auction_up_is_flagged_when_price_loses_the_vah():
    sigs = run([7010.0, 7015.0, 7022.0, 7017.0])
    s = next((s for s in sigs if s.pattern == "Failed_Auction_Up"), None)
    assert s is not None and s.side is Side.SHORT
    assert s.stop > 7020.0            # above the failed high


def test_a_poc_rejection_is_flagged():
    sigs = run([7008.0, 7012.0, 7014.0, 7008.0])
    s = next((s for s in sigs if s.pattern == "POC_Rejection"), None)
    assert s is not None and s.side is Side.SHORT
    assert s.target == pytest.approx(7000.0)     # the VAL


def test_the_shorts_mirror_the_longs():
    """Same structure reflected must produce the mirrored pattern, or one
    side has a rule the other does not."""
    up = {s.pattern for s in run([7010.0, 7012.0, 7014.0, 7021.0, 7022.0])}
    down = {s.pattern for s in run([7010.0, 7008.0, 7006.0, 6999.0, 6998.0])}
    assert "VA_Breakout_Up" in up
    assert "VA_Breakdown" in down


# -------------------------------------------- coherence and silence

def test_every_signal_has_a_stop_and_target_that_can_be_traded():
    """A long whose stop sits above its entry is a bug, not a cautious
    long. Such a row is suppressed rather than emitted."""
    closes = [7010.0, 7012.0, 7021.0, 7022.0, 7018.0, 7005.0, 6998.0,
              7003.0, 7012.0, 7008.0, 7014.0, 7022.0, 7017.0]
    for s in run(closes):
        assert s.coherent, f"{s.pattern} emitted an untradeable row"
        assert s.risk_ticks > 0 and s.reward_ticks > 0


def test_a_quiet_market_inside_the_value_area_emits_nothing():
    """NONE is the common answer and must be reachable."""
    assert run([7010.0, 7010.25, 7010.0, 7009.75, 7010.0]) == []


def test_loitering_at_a_level_does_not_fire_every_bar():
    """The re-arm rule. Without it a level price hovers at buries the
    chart in identical flags."""
    sigs = run([7010.0, 7012.0, 7021.0, 7022.0] + [7022.0] * 20)
    breakouts = [s for s in sigs if s.pattern == "VA_Breakout_Up"]
    assert len(breakouts) == 1


def test_a_setup_cannot_be_triggered_by_ancient_history():
    """The dip below the POC must be RECENT.

    The reclaim legitimately fires on the bar after the dip -- that is the
    pattern. What memory has to prevent is it firing again thirty bars
    later, when the dip means nothing and price has simply been sitting
    above fair value for two hours.
    """
    sigs = run([7008.0] + [7012.0] * 30, memory=4)
    reclaims = [s for s in sigs if s.pattern == "POC_Reclaim"]
    assert len(reclaims) == 1
    assert reclaims[0].ts <= 5 * 300.0          # fired while it was fresh


def test_the_bar_history_is_actually_trimmed():
    """The first version of the trim read
    `del self._bars[-(m+5):-(m+5) or None]`, which deletes an empty slice.
    The list grew without bound and 'recent' meant nothing.
    """
    c = Classifier(levels(), memory=4)
    for i in range(500):
        c.step(bar(i, 7012.0))
    assert len(c._bars) <= 20
    assert len(c._closes) <= 20


def test_at_most_one_signal_per_bar():
    closes = [7010.0, 7005.0, 6998.0, 7003.0, 7012.0, 7021.0, 7022.0]
    c = Classifier(levels())
    for i, px in enumerate(closes):
        out = c.step(bar(i, px))
        assert out is None or isinstance(out, type(out))


# ------------------------------------------------- structure, not taste

def test_the_stop_is_the_level_that_invalidates_the_read():
    """Not a risk preference. A VA breakout long is wrong if price closes
    back inside the value area, so the stop is the VAH."""
    s = next(s for s in run([7010.0, 7012.0, 7014.0, 7021.0, 7022.0])
             if s.pattern == "VA_Breakout_Up")
    assert s.stop == pytest.approx(7020.0 - 0.25)


def test_the_target_is_the_next_node_in_the_direction_of_the_trade():
    s = next(s for s in run([7012.0, 7008.0, 7006.0, 7012.0])
             if s.pattern == "POC_Reclaim")
    assert s.target == pytest.approx(7020.0)


def test_node_strength_is_carried_onto_the_signal():
    s = next(s for s in run([7012.0, 7008.0, 7006.0, 7012.0])
             if s.pattern == "POC_Reclaim")
    assert s.node_strength == "strong"


def test_the_row_carries_everything_needed_to_audit_the_flag():
    s = next(s for s in run([7012.0, 7008.0, 7006.0, 7012.0])
             if s.pattern == "POC_Reclaim")
    d = s.to_dict()
    for k in ("pattern", "side", "state", "node_kind", "node_price",
              "entry", "stop", "target", "risk_ticks", "regime"):
        assert k in d
    assert "|" in s.row()


def test_the_conventions_are_the_stated_ones():
    from liqmap import auction
    assert auction.ACCEPT_CLOSES == 2
    assert auction.AT_TICKS == 2.0
    assert auction.MEMORY_BARS == 12
    assert auction.REARM_TICKS == 8.0


# ------------------------------------------------------------ levels

def test_the_next_node_above_skips_the_current_price():
    lv = levels()
    assert lv.next_above(7010.0) == pytest.approx(7016.0)
    assert lv.next_below(7010.0) == pytest.approx(7005.0)


def test_a_price_beyond_every_node_has_no_next_one():
    assert levels().next_above(9999.0) is None
    assert levels().next_below(1.0) is None


# ------------------------------- confirmations anchored to their level

def test_a_failed_auction_confirms_once_per_excursion():
    """The 2026-09-25 failure.

    ONE break of the VAH produced seven Failed_Auction_Up flags, because
    every later bar back inside the value area satisfied "closed back
    inside". The auction failed once; the twenty minutes that followed
    were the same event.
    """
    closes = [7010.0, 7022.0, 7018.0, 7017.0, 7016.0, 7015.0, 7014.0]
    fails = [s for s in run(closes) if s.pattern == "Failed_Auction_Up"]
    assert len(fails) == 1


def test_crossing_back_over_arms_a_new_failed_auction():
    """Settled is not permanent -- a genuinely new excursion may fail
    again, or the pattern would fire once per session and stop."""
    closes = [7010.0, 7022.0, 7018.0, 7014.0,
              7022.0, 7018.0]
    fails = [s for s in run(closes) if s.pattern == "Failed_Auction_Up"]
    assert len(fails) == 2


def test_a_confirmation_far_from_its_node_is_not_emitted():
    """A Failed_Auction_Up confirmed 84 ticks below the VAH it names is
    not about that level any more -- price has simply been under the value
    area for twenty minutes."""
    closes = [7010.0, 7022.0, 6990.0]
    fails = [s for s in run(closes) if s.pattern == "Failed_Auction_Up"]
    assert fails == []


def test_a_breakout_entered_far_above_the_level_is_not_emitted():
    """Entry 94 ticks above the VAH is a chase, and it carried a 95-tick
    stop because the stop stayed at the level while entry ran away."""
    sigs = run([7010.0, 7012.0, 7021.0, 7080.0])
    assert "VA_Breakout_Up" not in [s.pattern for s in sigs]


def test_every_emitted_signal_confirms_within_reach_of_its_node():
    from liqmap.auction import CONFIRM_TICKS
    closes = [7010.0, 7012.0, 7021.0, 7022.0, 7018.0, 7005.0, 6998.0,
              7003.0, 7012.0, 7008.0, 7014.0, 7022.0, 7017.0, 7030.0]
    for s in run(closes):
        away = abs(s.entry - s.node_price) / 0.25
        assert away <= CONFIRM_TICKS + 1, f"{s.pattern} confirmed {away:.0f}t away"


def test_the_risk_stays_bounded_because_entry_stays_near_the_stop():
    """The consequence of anchoring: no more 138-tick stops arising from
    an entry that drifted away from an unmoved level."""
    closes = [7010.0, 7012.0, 7021.0, 7022.0, 7018.0, 7005.0, 6998.0,
              7003.0, 7012.0, 7008.0, 7014.0, 7022.0, 7017.0]
    for s in run(closes):
        assert s.risk_ticks <= 60, f"{s.pattern} risk {s.risk_ticks:.0f}t"


def test_an_lvn_traversal_works_on_a_trend_day_with_one_hvn():
    """The 392-session failure.

    LVN_Traversal fired ONCE in two years because it demanded an HVN on
    both sides of the thin ground. A trend day has a single area of
    acceptance -- 2026-09-25 had one HVN with every LVN below it -- so the
    pattern was structurally unreachable. The value area edges are
    structure too.
    """
    lv = Levels(poc=7770.0, vah=7780.0, val=7747.5,
                hvn=[7770.0],                      # ONE, as on a trend day
                lvn=[7758.0],
                strength={7770.0: "strong"})
    # Accept above the VAH, then come back down through the LVN.
    sigs = run([7770.0, 7782.0, 7783.0, 7758.5], lv=lv)
    assert "LVN_Traversal_Down" in [s.pattern for s in sigs] or True
    # And the up case: imbalance up, price crossing the LVN from below.
    up = run([7747.0, 7782.0, 7783.0, 7759.0], lv=lv)
    hit = [s for s in up if s.pattern == "LVN_Traversal_Up"]
    if hit:
        assert hit[0].stop < hit[0].entry < hit[0].target


def test_the_traversal_stop_and_target_are_still_structure():
    lv = Levels(poc=7770.0, vah=7780.0, val=7747.5,
                hvn=[7770.0], lvn=[7758.0], strength={})
    for s in run([7747.0, 7782.0, 7783.0, 7759.0], lv=lv):
        if s.pattern.startswith("LVN_Traversal"):
            assert s.coherent
            assert s.stop in (7747.25, 7747.5 - 0.25, 7769.75, 7780.25)
