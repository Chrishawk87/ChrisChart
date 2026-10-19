"""Does a diagonal imbalance predict anything?

The arithmetic of that question is easy to get wrong in ways that
flatter the answer, so the tests here are mostly about the counting:

  a bar with fifteen flags is ONE observation, not fifteen
  a bar flagging both ways is not a direction
  entry is the bar's close, not the flagged price
  "reaching the extreme" cannot mean "at the extreme"
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from liqmap.footprint import (                             # noqa: E402
    BUY_IMBALANCE, SELL_IMBALANCE, FootprintCandle, ImbalanceConfig,
    VolumeRule)
from liqmap.footprint_live import LiveFlag                 # noqa: E402
from liqmap.footprint_signals import (                     # noqa: E402
    KINDS, Replay, Seconds, Setup, setups_from)

T0 = 1_700_000_000.0
PT, TICK = 1_000_000_000, 250_000_000
BASE = 5800 * PT


def bar_with(levels, lo=23200, close=None):
    c = FootprintCandle(ts_open=T0, ts_close=T0 + 60)
    for i, (bid, ask) in enumerate(levels):
        p = lo + i
        if bid:
            c.add(p, bid, "sell", T0)
        if ask:
            c.add(p, ask, "buy", T0)
    if close is not None:
        c.close_ticks = close
    return c


def flag(p, side, vol=100):
    return LiveFlag(price_ticks=p, side=side, volume=vol, against=1,
                    ratio=float(vol), state="confirmed",
                    first_seen_ts=T0, last_seen_ts=T0 + 60)


class TestOneBarIsOneObservation:
    """Fifteen flags in a bar share an entry price, a timestamp and
    an outcome. Scoring them separately multiplies the sample by
    fifteen without adding one independent fact -- and the confidence
    intervals computed on that are not slightly wrong."""

    def test_many_flags_produce_one_single_setup(self):
        bar = bar_with([(0, 50)] * 12)
        flags = [flag(23200 + i, BUY_IMBALANCE) for i in range(12)]
        got = setups_from(bar, flags, 0)
        assert sum(1 for s in got if s.kind == "single") == 1

    def test_at_most_one_setup_of_each_kind(self):
        bar = bar_with([(0, 50)] * 12)
        flags = [flag(23200 + i, BUY_IMBALANCE) for i in range(12)]
        got = setups_from(bar, flags, 0)
        kinds = [s.kind for s in got]
        assert len(kinds) == len(set(kinds)), kinds
        assert set(kinds) <= set(KINDS)

    def test_every_setup_from_one_bar_shares_its_entry(self):
        bar = bar_with([(0, 50)] * 12, close=23205)
        flags = [flag(23200 + i, BUY_IMBALANCE) for i in range(12)]
        got = setups_from(bar, flags, 0)
        assert len({(s.ts, s.entry_ticks) for s in got}) == 1


class TestABarThatFlagsBothWaysIsSkipped:
    """Buyers overwhelming one level while sellers overwhelm another
    is not a direction, and picking the bigger one is a free
    parameter."""

    def test_no_setup_at_all(self):
        bar = bar_with([(0, 50), (50, 0), (0, 50)])
        flags = [flag(23200, BUY_IMBALANCE),
                 flag(23201, SELL_IMBALANCE)]
        assert setups_from(bar, flags, 0) == []

    def test_one_sided_still_produces_one(self):
        bar = bar_with([(0, 50), (0, 50)])
        assert setups_from(bar, [flag(23200, BUY_IMBALANCE)], 0)


class TestDirection:

    def test_a_buy_imbalance_is_long(self):
        bar = bar_with([(0, 50), (0, 50)])
        got = setups_from(bar, [flag(23200, BUY_IMBALANCE)], 0)
        assert got[0].side == 1

    def test_a_sell_imbalance_is_short(self):
        bar = bar_with([(50, 0), (50, 0)])
        got = setups_from(bar, [flag(23200, SELL_IMBALANCE)], 0)
        assert got[0].side == -1


class TestEntryIsTheBarsClose:
    """The flag is only confirmed when the bar ends. Filling at the
    price it occurred at would read the past with the future's
    information."""

    def test_entry_is_the_close_not_the_flagged_price(self):
        bar = bar_with([(0, 50)] * 6, close=23210)
        got = setups_from(bar, [flag(23201, BUY_IMBALANCE)], 0)
        assert got[0].entry_ticks == 23210
        assert got[0].flag_ticks == 23201

    def test_the_timestamp_is_the_bars_close(self):
        bar = bar_with([(0, 50), (0, 50)])
        got = setups_from(bar, [flag(23200, BUY_IMBALANCE)], 0)
        assert got[0].ts == bar.ts_close


class TestStacked:

    def test_three_consecutive_qualify(self):
        bar = bar_with([(0, 50)] * 4)
        flags = [flag(23200 + i, BUY_IMBALANCE) for i in range(3)]
        assert any(s.kind == "stacked" for s in setups_from(bar, flags, 0))

    def test_three_scattered_do_not(self):
        bar = bar_with([(0, 50)] * 8)
        flags = [flag(23200, BUY_IMBALANCE), flag(23203, BUY_IMBALANCE),
                 flag(23206, BUY_IMBALANCE)]
        assert not [s for s in setups_from(bar, flags, 0)
                    if s.kind == "stacked"]

    def test_the_run_length_is_recorded(self):
        bar = bar_with([(0, 50)] * 6)
        flags = [flag(23200 + i, BUY_IMBALANCE) for i in range(5)]
        st = next(s for s in setups_from(bar, flags, 0)
                  if s.kind == "stacked")
        assert st.run == 5

    def test_the_longest_run_wins_when_there_are_two(self):
        bar = bar_with([(0, 50)] * 12)
        flags = [flag(p, BUY_IMBALANCE)
                 for p in (23200, 23201, 23205, 23206, 23207, 23208)]
        st = next(s for s in setups_from(bar, flags, 0)
                  if s.kind == "stacked")
        assert st.run == 4


class TestExtreme:
    """`reaching` cannot mean `at`. The edge policy skips the bar's
    lowest level for a BUY test -- its diagonal lies outside the bar
    -- so a buy flag at the low is impossible by construction, and a
    condition demanding one never fires at all."""

    def test_a_buy_stack_one_tick_above_the_low_counts(self):
        bar = bar_with([(5, 5)] + [(0, 50)] * 5)
        flags = [flag(23201 + i, BUY_IMBALANCE) for i in range(3)]
        assert any(s.kind == "extreme"
                   for s in setups_from(bar, flags, 0))

    def test_a_stack_in_the_middle_does_not(self):
        bar = bar_with([(5, 5)] * 3 + [(0, 50)] * 3 + [(5, 5)] * 4)
        flags = [flag(23203 + i, BUY_IMBALANCE) for i in range(3)]
        assert not [s for s in setups_from(bar, flags, 0)
                    if s.kind == "extreme"]

    def test_a_sell_stack_near_the_high_counts(self):
        bar = bar_with([(5, 5)] * 3 + [(50, 0)] * 3)
        flags = [flag(23203 + i, SELL_IMBALANCE) for i in range(3)]
        assert any(s.kind == "extreme"
                   for s in setups_from(bar, flags, 0))

    def test_the_tolerance_is_a_parameter(self):
        bar = bar_with([(5, 5)] * 4 + [(0, 50)] * 3)
        flags = [flag(23204 + i, BUY_IMBALANCE) for i in range(3)]
        assert not [s for s in setups_from(bar, flags, 0, edge_within=1)
                    if s.kind == "extreme"]
        assert [s for s in setups_from(bar, flags, 0, edge_within=5)
                if s.kind == "extreme"]

    def test_an_extreme_is_always_also_a_stack(self):
        bar = bar_with([(5, 5)] + [(0, 50)] * 5)
        flags = [flag(23201 + i, BUY_IMBALANCE) for i in range(3)]
        kinds = {s.kind for s in setups_from(bar, flags, 0)}
        if "extreme" in kinds:
            assert "stacked" in kinds


# ------------------------------------------------------- the two ends

def records(n, planted, seed):
    """Trade records; `planted` makes one aggressor lead price."""
    rng = random.Random(seed)
    px, ts, burst, side = BASE, T0, 0, None
    for i in range(n):
        ts += 0.1
        if burst == 0 and i % 300 == 0:
            burst, side = 60, rng.choice(("buy", "sell"))
        if burst > 0:
            burst -= 1
            agg = side if planted else rng.choice(("buy", "sell"))
            size = rng.randint(4, 30)
            if planted and rng.random() < 0.55:
                px += TICK if side == "buy" else -TICK
            elif not planted:
                px += rng.choice((-TICK, 0, 0, TICK))
        else:
            agg = rng.choice(("buy", "sell"))
            size = rng.randint(1, 6)
            px += rng.choice((-TICK, 0, 0, TICK))
        o = type("R", (), {})()
        o.ts_event = int(ts * 1e9)
        o.sequence, o.instrument_id = i, 42
        o.action, o.side = "T", ("B" if agg == "buy" else "A")
        o.price, o.size = px, size
        yield o


def run(planted, seed, n=40_000, kind="stacked"):
    r = Replay(rule=VolumeRule(400),
               imbalance=ImbalanceConfig(multiplier=4.0, min_volume=10))
    r.feed(records(n, planted, seed))
    return r, r.score(kind=kind, target_ticks=16, stop_ticks=4,
                      cost_ticks=1.0)


def held(out, arm):
    return out["arms"][arm]["held_out"]


class TestItFindsNothingInNoise:

    def test_the_arms_collapse_together(self):
        _, out = run(False, 9, kind="single")
        rates = [held(out, a)["win_rate"]
                 for a in ("signal", "mirror", "control")]
        assert all(r is not None for r in rates)
        assert max(rates) - min(rates) < 0.12, rates

    def test_it_does_not_clear_break_even(self):
        _, out = run(False, 9, kind="single")
        assert held(out, "signal")["win_rate"] < out["break_even"]

    def test_the_verdict_never_claims_an_edge(self):
        _, out = run(False, 9, kind="single")
        assert "clears break-even" not in out["verdict"]


class TestItFindsAPlantedEdge:

    def test_the_signal_separates_from_its_mirror(self):
        _, out = run(True, 3)
        assert (held(out, "signal")["win_rate"]
                > held(out, "mirror")["win_rate"] + 0.2)

    def test_it_clears_break_even_out_of_sample(self):
        _, out = run(True, 3)
        assert held(out, "signal")["win_rate_lo"] > out["break_even"]

    def test_every_kind_produces_setups(self):
        """`extreme` returned nothing at all until the edge-policy
        bug was found -- an unreachable condition looks exactly like
        'the setup is rare'."""
        r, _ = run(True, 3)
        for kind in KINDS:
            assert [s for s in r.setups if s.kind == kind], kind


class TestTheForwardPath:

    def test_it_is_built_from_the_same_prints(self):
        r, _ = run(True, 3, n=5_000)
        assert len(r.secs) > 0
        assert r.secs.out_of_order == 0

    def test_out_of_order_data_is_refused(self):
        r = Replay(rule=VolumeRule(400))
        r.feed(records(3_000, True, 3))
        r.secs.ts[len(r.secs) // 2] = r.secs.ts[0] - 10_000
        out = r.score(kind="single")
        assert out["ok"] is False and "backwards" in out["why"]

    def test_the_arms_take_identical_entries(self):
        _, out = run(True, 3)
        n = {a: out["arms"][a]["all"]["scored"]
             + out["arms"][a]["all"]["ambiguous"]
             + out["arms"][a]["all"]["unresolved"]
             for a in ("signal", "mirror", "control")}
        assert len(set(n.values())) == 1, n

    def test_break_even_is_reported_with_every_result(self):
        _, out = run(True, 3)
        assert out["break_even"] == pytest.approx(0.25)


class TestItDecidesNothing:

    def test_no_order_code(self):
        src = (ROOT / "liqmap" / "footprint_signals.py").read_text()
        for w in ("def submit", "def execute", "def place", "def order"):
            assert w not in src, w

    def test_cost_is_an_argument(self):
        r, _ = run(True, 3, n=10_000)
        a = r.score(kind="single", cost_ticks=0.0)
        b = r.score(kind="single", cost_ticks=5.0)
        assert (held(a, "signal")["per_trade_ticks"]
                > held(b, "signal")["per_trade_ticks"])
