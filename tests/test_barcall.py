"""A reading of every closed candle, and the ways a scoreboard flatters it.

In order of how badly each would mislead:

  one verdict holding both leans. A reading that leans long half the
  time and short the other half cancels against itself and prints a
  row that says nothing while looking like the largest sample in the
  table -- the most authoritative empty number available

  flat outcomes counted as misses. Over three bars of a one-tick
  market more than half of all moves are exactly zero, and burying
  those in the denominator drags every hit rate toward a figure that
  has nothing to do with direction

  scoring calls that have not finished. The last few have no outcome,
  and a zero for them pulls the mean toward nothing while making the
  count look larger than the evidence

  a thin bar's delta naming the bar. One lot either way flips the
  sign, and a verdict that turns on a single contract is noise wearing
  a label

  a hit rate printed without the rate it has to beat. 51% looks like
  an edge until the number beside it says 60%
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from liqmap import barcall as BC  # noqa: E402
from liqmap.ofibar import OfiBarBuilder  # noqa: E402

RAW = 250_000_000


class Bar:
    """The minimum `call_bar` reads. Any OfiBar satisfies it."""

    def __init__(self, direction="chop", aggression="BALANCED_PRICE_FLOW",
                 delta=0.0, volume=100.0, close=0.0, ts_close=0.0):
        self.direction = direction
        self.aggression = aggression
        self.delta = delta
        self.volume = volume
        self.close = close
        self.ts_close = ts_close


def up(**kw):
    kw.setdefault("aggression", "BUY_PRICE_AGGRESSIVE")
    return Bar(**kw)


def down(**kw):
    kw.setdefault("aggression", "SELL_PRICE_AGGRESSIVE")
    return Bar(**kw)


class TestTheDisagreementsGetTheirOwnNames:
    """Folding them into a score would make them the same number."""

    def test_a_green_bar_sold_into(self):
        c = BC.call_bar(Bar(direction="green", delta=-50))
        assert c.verdict == BC.SOLD_INTO and c.lean == BC.SHORT

    def test_a_red_bar_bought_into(self):
        c = BC.call_bar(Bar(direction="red", delta=50))
        assert c.verdict == BC.BOUGHT_INTO and c.lean == BC.LONG

    def test_paying_up_with_a_two_sided_path(self):
        c = BC.call_bar(Bar(delta=50))
        assert c.verdict == BC.PAID_NO_MOVE and c.lean == BC.LONG

    def test_paying_down_with_a_two_sided_path(self):
        c = BC.call_bar(Bar(delta=-50))
        assert c.verdict == BC.PAID_NO_MOVE and c.lean == BC.SHORT

    def test_a_path_with_nothing_printing_behind_it(self):
        c = BC.call_bar(up(delta=0.0))
        assert c.verdict == BC.UNPAID_PUSH and c.lean == BC.NONE

    def test_everything_agreeing_up(self):
        c = BC.call_bar(up(direction="green", delta=50))
        assert c.verdict == BC.CONFIRMED_UP and c.lean == BC.LONG

    def test_everything_agreeing_down(self):
        c = BC.call_bar(down(direction="red", delta=-50))
        assert c.verdict == BC.CONFIRMED_DOWN and c.lean == BC.SHORT

    def test_a_two_sided_bar_with_nothing_paid(self):
        c = BC.call_bar(Bar(delta=0.0))
        assert c.verdict == BC.NO_INTENT and c.lean == BC.NONE

    def test_the_disagreement_is_decided_before_the_agreement(self):
        """A green bar sold into is a different event from a green bar
        bought into, and the order of the tests is what keeps it."""
        c = BC.call_bar(up(direction="green", delta=-50))
        assert c.verdict == BC.SOLD_INTO, "the agreement swallowed it"


class TestAThinBarCannotNameItself:

    def test_a_bar_under_the_volume_floor_is_no_prints(self):
        c = BC.call_bar(Bar(direction="green", delta=-5, volume=4))
        assert c.verdict == BC.NO_PRINTS and c.lean == BC.NONE

    def test_one_lot_either_way_does_not_flip_a_verdict(self):
        a = BC.call_bar(Bar(direction="green", delta=1, volume=5))
        b = BC.call_bar(Bar(direction="green", delta=-1, volume=5))
        assert a.verdict == b.verdict == BC.NO_PRINTS

    def test_the_floor_is_on_volume_not_on_delta(self):
        """A bar can trade a thousand lots and net zero. That is a real
        reading, not an absent one."""
        c = BC.call_bar(up(direction="green", delta=0.0, volume=1000))
        assert c.verdict != BC.NO_PRINTS

    def test_a_bar_with_no_trades_at_all(self):
        c = BC.call_bar(Bar(volume=0.0, delta=0.0))
        assert c.verdict == BC.NO_PRINTS


class TestEveryBarGetsExactlyOneCall:

    def _bars(self, n=40_000, seed=7):
        import random
        rng = random.Random(seed)
        b = OfiBarBuilder(tick_size=0.25, threshold=250, max_s=900.0,
                          min_updates=2, max_bars=10 ** 7)
        bid, ts = 7_700 * 10 ** 9, 0.0
        for _ in range(n):
            if rng.random() < 0.08:
                bid += RAW if rng.random() < 0.5 else -RAW
            ts += 0.01
            if rng.random() < 0.35:
                px = bid if rng.random() < 0.5 else bid + RAW
                b.on_trade(ts, rng.randint(1, 20),
                           "buy" if rng.random() < 0.5 else "sell", px)
            b.on_book(ts, bid, rng.randint(50, 600), bid + RAW,
                      rng.randint(50, 600))
        return b.bars

    def test_the_verdicts_partition_the_bars(self):
        bars = self._bars()
        assert len(bars) > 500
        calls = BC.call_all(bars)
        assert len(calls) == len(bars)
        assert all(c.verdict in BC.VERDICTS for c in calls)

    def test_the_index_follows_the_bar(self):
        bars = self._bars(8_000)
        calls = BC.call_all(bars)
        assert [c.index for c in calls] == list(range(len(bars)))

    def test_it_reads_a_real_bar_without_being_told_the_columns(self):
        bars = self._bars(8_000)
        c = BC.call_bar(bars[-1], index=len(bars) - 1)
        assert c.book == bars[-1].direction
        assert c.paid == bars[-1].delta
        assert c.price == bars[-1].close

    def test_it_serialises(self):
        import json
        bars = self._bars(8_000)
        json.dumps([c.as_dict() for c in BC.call_all(bars)])


class TestTheLevelContext:

    def test_a_level_is_named_in_the_sentence(self):
        c = BC.call_bar(Bar(delta=50), where="low", level_state="held")
        assert "swing low" in c.says and "held" in c.says
        assert c.at_level

    def test_a_bar_away_from_everything_says_nothing_about_levels(self):
        c = BC.call_bar(Bar(delta=50))
        assert not c.at_level and "swing" not in c.says

    def test_the_context_does_not_change_the_verdict(self):
        """Where a bar happened is not what the bar was."""
        a = BC.call_bar(Bar(direction="green", delta=-50))
        b = BC.call_bar(Bar(direction="green", delta=-50),
                        where="low", level_state="held")
        assert a.verdict == b.verdict and a.lean == b.lean

    def test_call_all_takes_one_context_per_bar(self):
        bars = [Bar(delta=50), Bar(delta=-50), Bar(delta=50)]
        ctx = [{"side": "low", "status": "held"}, {}, None]
        calls = BC.call_all(bars, ctx)
        assert calls[0].where == "low"
        assert calls[1].where == "" and calls[2].where == ""


class TestTheScoreboardDoesNotFlatter:

    def _calls(self, leans, verdict=BC.PAID_NO_MOVE):
        return [BC.Call(index=i, verdict=verdict, lean=ln)
                for i, ln in enumerate(leans)]

    def test_one_verdict_with_two_leans_is_two_rows(self):
        """Folded together they cancel and print the most
        authoritative empty number in the table."""
        calls = self._calls([BC.LONG, BC.SHORT] * 50)
        prices = [float(i) for i in range(len(calls))]
        rows = BC.score(calls, prices, horizon=1)
        assert (BC.PAID_NO_MOVE, BC.LONG) in rows
        assert (BC.PAID_NO_MOVE, BC.SHORT) in rows

    def test_the_two_leans_are_scored_in_their_own_directions(self):
        calls = self._calls([BC.LONG, BC.SHORT] * 50)
        prices = [float(i) for i in range(len(calls))]   # always rising
        rows = BC.score(calls, prices, horizon=1)
        assert rows[(BC.PAID_NO_MOVE, BC.LONG)].hit_rate == 1.0
        assert rows[(BC.PAID_NO_MOVE, BC.SHORT)].hit_rate == 0.0

    def test_a_flat_outcome_is_not_a_miss(self):
        calls = self._calls([BC.LONG] * 10)
        prices = [5.0] * 10                       # price never moves
        r = BC.score(calls, prices, horizon=1)[(BC.PAID_NO_MOVE, BC.LONG)]
        assert r.flat == r.n and r.decided == 0
        assert r.hit_rate == 0.0, "no decided calls means no rate"

    def test_flats_stay_out_of_the_denominator(self):
        calls = self._calls([BC.LONG] * 6)
        # up, flat, flat, up, down -> 2 hits, 1 miss, 2 flat
        prices = [0.0, 1.0, 1.0, 1.0, 2.0, 1.0]
        r = BC.score(calls, prices, horizon=1)[(BC.PAID_NO_MOVE, BC.LONG)]
        assert r.hits == 2 and r.misses == 1 and r.flat == 2
        assert r.hit_rate == pytest.approx(2 / 3)

    def test_unfinished_calls_are_not_scored(self):
        calls = self._calls([BC.LONG] * 10)
        prices = [float(i) for i in range(10)]
        r = BC.score(calls, prices, horizon=3)[(BC.PAID_NO_MOVE, BC.LONG)]
        assert r.n == 7, "the last three have not finished"

    def test_a_call_with_no_lean_can_never_hit(self):
        calls = self._calls([BC.NONE] * 10, verdict=BC.NO_INTENT)
        prices = [float(i) for i in range(10)]
        r = BC.score(calls, prices, horizon=1)[(BC.NO_INTENT, BC.NONE)]
        assert r.hits == 0 and r.decided == 0 and r.n == 9

    def test_the_mean_is_in_the_lean_s_direction(self):
        """A short that made two ticks reads +2, not -2."""
        calls = self._calls([BC.SHORT] * 5)
        prices = [10.0, 8.0, 6.0, 4.0, 2.0]
        r = BC.score(calls, prices, horizon=1)[(BC.PAID_NO_MOVE, BC.SHORT)]
        assert r.mean_ticks == pytest.approx(2.0)

    def test_the_tick_divides_the_move(self):
        calls = self._calls([BC.LONG] * 3)
        prices = [0.0, 250_000_000.0, 500_000_000.0]
        r = BC.score(calls, prices, horizon=1, tick=250_000_000.0)
        assert r[(BC.PAID_NO_MOVE, BC.LONG)].mean_ticks == pytest.approx(1.0)


class TestItPrintsTheRateItHasToBeat:

    def test_break_even_matches_the_arithmetic(self):
        assert BC.break_even(2, 2, 0.4) == pytest.approx(0.6)
        assert BC.break_even(4, 4, 0.4) == pytest.approx(0.55)

    def test_a_wider_target_lowers_the_bar(self):
        assert BC.break_even(8, 2, 0.4) < BC.break_even(2, 2, 0.4)

    def test_cost_raises_it(self):
        assert BC.break_even(2, 2, 1.0) > BC.break_even(2, 2, 0.4)

    def test_every_row_carries_what_it_needs(self):
        calls = [BC.Call(index=i, verdict=BC.CONFIRMED_UP, lean=BC.LONG)
                 for i in range(400)]
        prices = [float(i) for i in range(400)]
        sb = BC.scoreboard(calls, prices, horizon=1)
        assert sb["needs"] == pytest.approx(0.6)
        assert all(r["needs"] == pytest.approx(0.6) for r in sb["rows"])

    def test_a_row_below_the_bar_does_not_clear_it(self):
        calls = [BC.Call(index=i, verdict=BC.CONFIRMED_UP, lean=BC.LONG)
                 for i in range(400)]
        # Rises 55% of the time: a real edge, and still not enough.
        prices = [0.0]
        for i in range(400):
            prices.append(prices[-1] + (1.0 if i % 20 < 11 else -1.0))
        sb = BC.scoreboard(calls, prices, horizon=1)
        row = sb["rows"][0]
        assert row["hit_rate"] > 0.5
        assert not row["clears"], "55% does not cover a 60% break-even"

    def test_a_small_sample_never_clears(self):
        """Sixty calls at 100% is sixty coin flips."""
        calls = [BC.Call(index=i, verdict=BC.CONFIRMED_UP, lean=BC.LONG)
                 for i in range(60)]
        prices = [float(i) for i in range(60)]
        sb = BC.scoreboard(calls, prices, horizon=1)
        assert sb["rows"][0]["hit_rate"] == 1.0
        assert not sb["rows"][0]["clears"]
        assert not sb["rows"][0]["enough"]

    def test_a_genuinely_good_row_does_clear(self):
        calls = [BC.Call(index=i, verdict=BC.CONFIRMED_UP, lean=BC.LONG)
                 for i in range(600)]
        prices = [float(i) for i in range(600)]
        sb = BC.scoreboard(calls, prices, horizon=1)
        assert sb["rows"][0]["clears"]

    def test_the_scored_count_excludes_the_unfinished(self):
        calls = [BC.Call(index=i) for i in range(100)]
        sb = BC.scoreboard(calls, [float(i) for i in range(100)], horizon=5)
        assert sb["scored"] == 95


class TestOnAMarketWithNothingInIt:
    """The result that matters: a reading of noise must read as noise."""

    def test_nothing_clears_break_even_on_a_random_walk(self):
        import random
        rng = random.Random(7)
        b = OfiBarBuilder(tick_size=0.25, threshold=250, max_s=900.0,
                          min_updates=2, max_bars=10 ** 7)
        bid, ts = 7_700 * 10 ** 9, 0.0
        for _ in range(300_000):
            if rng.random() < 0.08:
                bid += RAW if rng.random() < 0.5 else -RAW
            ts += 0.01
            if rng.random() < 0.35:
                px = bid if rng.random() < 0.5 else bid + RAW
                b.on_trade(ts, rng.randint(1, 20),
                           "buy" if rng.random() < 0.5 else "sell", px)
            b.on_book(ts, bid, rng.randint(50, 600), bid + RAW,
                      rng.randint(50, 600))
        calls = BC.call_all(b.bars)
        sb = BC.scoreboard(calls, [x.close for x in b.bars],
                           horizon=3, tick=RAW)
        assert sb["scored"] > 10_000
        assert not any(r["clears"] for r in sb["rows"]), \
            [r for r in sb["rows"] if r["clears"]]
        for r in sb["rows"]:
            if r["lean"] != BC.NONE and r["enough"]:
                assert 0.4 < r["hit_rate"] < 0.6, r


class TestItDecidesNothing:

    def test_no_order_code(self):
        src = (ROOT / "liqmap" / "barcall.py").read_text()
        for w in ("def submit(", "def execute(", "def place(", "def order("):
            assert w not in src, w

    def test_the_lean_is_not_dressed_up_as_a_forecast(self):
        src = (ROOT / "liqmap" / "barcall.py").read_text().lower()
        assert "nothing here predicts" in src
        for w in ("will rise", "will fall", "buy signal", "sell signal",
                  "guaranteed", "high probability"):
            assert w not in src, w

    def test_it_does_not_recompute_what_the_bar_already_carries(self):
        """A second opinion about a bar's delta is a second thing to
        keep correct."""
        import re
        src = (ROOT / "liqmap" / "barcall.py").read_text()
        read = set(re.findall(r'getattr\(bar,\s*"([a-z_]+)"', src))
        assert read == {"direction", "aggression", "delta", "volume",
                        "close", "ts_close"}, read
        # The raw accumulators behind those stay where they are.
        assert "bar.buy_vol" not in src and "bar.pai" not in src
