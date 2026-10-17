"""The backtest has to fail honestly before it is allowed to pass.

Two properties matter more than any feature here:

  on data with NO edge it must report none -- a backtest that finds
  edges in noise is worse than no backtest, because it is believed

  on data with a PLANTED edge it must find it -- one that cannot
  detect a real signal cannot clear one either

Both are tested below, on generated data whose answer is known in
advance. The real ES file is the thing this gets pointed at; these
are the checks that it is safe to point.
"""

from __future__ import annotations

import csv
import random
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from liqmap import ofi_backtest as bt                       # noqa: E402

T0 = 1_700_000_000


def flat(n=200, px=23200, of=0):
    s = bt.Seconds()
    for i in range(n):
        s.add(T0 + i, px, px, px, of)
    return s


class TestBreakEven:
    """The spec said 55% for a 4t/4t bracket at $17 of cost. It is
    67%. The error was computing the winner's net while treating the
    loser as costless -- both sides pay.
    """

    def test_a_symmetric_bracket_with_no_cost_is_a_coin_flip(self):
        assert bt.break_even(4, 4, 0.0) == pytest.approx(0.50)

    def test_cost_moves_it_and_both_sides_pay(self):
        assert bt.break_even(4, 4, 1.0) == pytest.approx(0.625)

    def test_a_wide_target_needs_far_less(self):
        """16t target / 4t stop at 0.4 ticks: 22%, against 55% for
        4t/4t. The bracket matters more than the hit rate."""
        assert bt.break_even(16, 4, 0.4) == pytest.approx(0.22)

    def test_it_is_reported_beside_every_result(self):
        out = bt.run(noise(20_000, seed=3))
        assert "break_even" in out
        assert str(round(out["break_even"] * 100)) in out["verdict"]


class TestResolve:

    def test_a_rising_tape_wins_a_long(self):
        s = bt.Seconds()
        for i in range(400):
            px = 23200 + i
            s.add(T0 + i, px + 1, px - 1, px, 0)
        t = bt.resolve(s, 0, +1, 16, 4, 900)
        assert t.outcome == bt.WIN and t.ticks == 16

    def test_a_rising_tape_loses_a_short(self):
        s = bt.Seconds()
        for i in range(400):
            px = 23200 + i
            s.add(T0 + i, px + 1, px - 1, px, 0)
        t = bt.resolve(s, 0, -1, 16, 4, 900)
        assert t.outcome == bt.LOSS and t.ticks == -4

    def test_a_second_touching_both_is_ambiguous_not_a_win(self):
        """The file does not say which came first, and guessing
        favourably is how a backtest invents an edge."""
        s = bt.Seconds()
        s.add(T0, 23200, 23200, 23200, 0)
        s.add(T0 + 1, 23240, 23100, 23200, 0)
        assert bt.resolve(s, 0, +1, 16, 4, 900).outcome == bt.AMBIGUOUS

    def test_an_ambiguous_trade_is_not_scored(self):
        s = bt.Seconds()
        s.add(T0, 23200, 23200, 23200, 0)
        s.add(T0 + 1, 23240, 23100, 23200, 0)
        r = bt.score([bt.resolve(s, 0, +1, 16, 4, 900)], "x")
        assert r.ambiguous == 1
        assert r.scored == 0, "it must not quietly count as anything"

    def test_a_file_that_ends_early_is_unresolved_not_a_timeout(self):
        """Booking it at the last known price fills the tail of every
        run with trades that were never given their time."""
        s = flat(5)
        assert bt.resolve(s, 0, +1, 16, 4, 900).outcome == bt.UNRESOLVED

    def test_a_trade_that_runs_out_of_patience_books_what_it_was_worth(self):
        s = bt.Seconds()
        for i in range(1200):
            px = 23200 + (1 if i > 10 else 0)
            s.add(T0 + i, px, px, px, 0)
        t = bt.resolve(s, 0, +1, 16, 4, 60)
        assert t.outcome == bt.TIMEOUT
        assert t.ticks == pytest.approx(1.0)

    def test_the_bar_that_created_the_entry_cannot_resolve_it(self):
        s = bt.Seconds()
        s.add(T0, 23300, 23100, 23200, 0)      # huge range AT entry
        for i in range(1, 400):
            s.add(T0 + i, 23200, 23200, 23200, 0)
        t = bt.resolve(s, 0, +1, 16, 4, 900)
        assert t.outcome != bt.WIN, "resolved by its own bar"

    def test_a_second_with_no_print_cannot_be_entered(self):
        s = bt.Seconds()
        s.add(T0, bt.NO_PRICE, bt.NO_PRICE, bt.NO_PRICE, 0)
        for i in range(1, 100):
            s.add(T0 + i, 23200, 23200, 23200, 0)
        assert bt.resolve(s, 0, +1, 16, 4, 900).outcome == bt.UNRESOLVED

    def test_seconds_without_prints_are_skipped_not_treated_as_flat(self):
        """A gap is missing data, not a bar where nothing moved."""
        s = bt.Seconds()
        s.add(T0, 23200, 23200, 23200, 0)
        for i in range(1, 300):
            if i % 2:
                s.add(T0 + i, bt.NO_PRICE, bt.NO_PRICE, bt.NO_PRICE, 0)
            else:
                s.add(T0 + i, 23300, 23290, 23295, 0)
        assert bt.resolve(s, 0, +1, 16, 4, 900).outcome == bt.WIN


# ------------------------------------------------------- the two properties

def noise(n=120_000, seed=7):
    """A random walk with OFI that is pure noise. No edge exists."""
    rng = random.Random(seed)
    s = bt.Seconds()
    px = 23200
    for i in range(n):
        px += rng.choice((-1, 0, 0, 1))
        s.add(T0 + i, px + rng.randint(0, 2), px - rng.randint(0, 2),
              px, rng.randint(-500, 500))
    return s


def planted(n=120_000, seed=11):
    """A burst of one-sided OFI, then a real move the same way."""
    rng = random.Random(seed)
    s = bt.Seconds()
    px, drift, pending = 23200, 0, 0
    for i in range(n):
        if pending == 0 and rng.random() < 0.004:
            drift, pending = rng.choice((1, -1)), 400
            of = drift * rng.randint(1800, 3000)
        elif pending > 0:
            pending -= 1
            of = drift * rng.randint(0, 200)
            px += drift if rng.random() < 0.30 else 0
        else:
            of = rng.randint(-400, 400)
            px += rng.choice((-1, 0, 0, 1))
        s.add(T0 + i, px + rng.randint(0, 2), px - rng.randint(0, 2),
              px, of)
    return s


def held(out, arm):
    return out["arms"][arm]["held_out"]


class TestItReportsNoEdgeWhenThereIsNone:
    """The failure that matters. A backtest that finds edges in noise
    is worse than no backtest, because it is believed."""

    def test_the_three_arms_are_indistinguishable(self):
        out = bt.run(noise())
        rates = [held(out, a)["win_rate"] for a in
                 ("signal", "mirror", "control")]
        assert all(r is not None for r in rates)
        assert max(rates) - min(rates) < 0.08, rates

    def test_the_signal_does_not_clear_break_even(self):
        out = bt.run(noise())
        assert held(out, "signal")["win_rate"] < out["break_even"]

    def test_the_verdict_says_so_in_words(self):
        assert "does not clear" in bt.run(noise())["verdict"]

    def test_per_trade_is_negative_once_costs_are_paid(self):
        out = bt.run(noise())
        assert held(out, "signal")["per_trade_ticks"] < 0


class TestItFindsAnEdgeThatIsReallyThere:
    """A backtest that cannot detect a planted signal cannot clear a
    real one either."""

    def test_the_signal_separates_from_its_mirror(self):
        out = bt.run(planted())
        assert (held(out, "signal")["win_rate"]
                > held(out, "mirror")["win_rate"] + 0.20)

    def test_it_clears_break_even_with_the_interval_above_it(self):
        out = bt.run(planted())
        assert held(out, "signal")["win_rate_lo"] > out["break_even"]

    def test_it_holds_out_of_sample_not_just_in_training(self):
        out = bt.run(planted())
        tr = out["arms"]["signal"]["train"]["win_rate"]
        ho = held(out, "signal")["win_rate"]
        assert ho > 0.35 and abs(tr - ho) < 0.25, (tr, ho)

    def test_the_verdict_says_it_clears(self):
        assert "clears break-even" in bt.run(planted())["verdict"]


class TestTheArmsAreComparable:

    def test_every_arm_takes_exactly_the_same_entries(self):
        """Different entry counts would make the comparison
        meaningless -- the arms must differ only in side."""
        out = bt.run(noise(40_000, seed=5))
        n = {a: out["arms"][a]["all"]["scored"]
             + out["arms"][a]["all"]["ambiguous"]
             + out["arms"][a]["all"]["unresolved"]
             for a in ("signal", "mirror", "control")}
        assert len(set(n.values())) == 1, n

    def test_the_control_is_reproducible(self):
        """A comparison that moves when you run it again settles
        nothing."""
        a = bt.run(noise(40_000, seed=5))
        b = bt.run(noise(40_000, seed=5))
        assert (held(a, "control")["wins"] == held(b, "control")["wins"])

    def test_the_control_is_not_all_one_side(self):
        sides = [bt._coin(T0 + i) for i in range(2000)]
        longs = sides.count(1)
        assert 800 < longs < 1200, f"{longs}/2000 long is not a coin"


class TestCooldown:

    def test_one_position_at_a_time(self):
        s = planted(40_000, seed=3)
        many = bt.entries(s, bt.Settings(z_enter=1.0, cooldown_s=0))
        few = bt.entries(s, bt.Settings(z_enter=1.0, cooldown_s=300))
        assert len(few) < len(many)

    def test_entries_respect_the_gap(self):
        s = planted(40_000, seed=3)
        got = bt.entries(s, bt.Settings(z_enter=1.0, cooldown_s=300))
        stamps = [s.ts[i] for i, _, _ in got]
        assert all(b - a >= 300 for a, b in zip(stamps, stamps[1:]))


class TestLoad:

    def _write(self, rows):
        fh = tempfile.NamedTemporaryFile("w", suffix=".csv",
                                         delete=False, newline="")
        w = csv.DictWriter(fh, fieldnames=["ts", "high", "low",
                                           "close", "ofi"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
        fh.close()
        return fh.name

    def test_an_empty_price_stays_empty(self):
        """`ofi_replay` writes nothing where nothing printed. Reading
        that back as 0.0 would be a price of zero."""
        p = self._write([{"ts": T0, "high": "", "low": "", "close": "",
                          "ofi": 40}])
        s = bt.load(p)
        assert s.cl[0] == bt.NO_PRICE

    def test_prices_become_exact_integer_ticks(self):
        p = self._write([{"ts": T0, "high": "5800.75", "low": "5800.00",
                          "close": "5800.25", "ofi": 0}])
        s = bt.load(p)
        assert s.cl[0] == 23201 and s.hi[0] == 23203

    def test_a_priced_second_is_counted_as_such(self):
        p = self._write([
            {"ts": T0, "high": "", "low": "", "close": "", "ofi": 1},
            {"ts": T0 + 1, "high": "5800.00", "low": "5800.00",
             "close": "5800.00", "ofi": 1}])
        assert bt.load(p).priced == 1


class TestItDecidesNothing:

    def test_no_order_code(self):
        src = (ROOT / "liqmap" / "ofi_backtest.py").read_text()
        for word in ("def submit", "def execute", "def place", "def order"):
            assert word not in src, word

    def test_the_threshold_is_an_argument_not_a_constant(self):
        """The line is a trading decision. This module measures."""
        assert "z_enter" in bt.Settings().__dict__

    def test_cost_is_an_argument_too(self):
        """Tightening cost assumptions until a backtest passes is how
        a losing rule gets shipped."""
        assert "cost_ticks" in bt.Settings().__dict__
        a = bt.run(noise(40_000, seed=5), bt.Settings(cost_ticks=0.0))
        b = bt.run(noise(40_000, seed=5), bt.Settings(cost_ticks=4.0))
        assert (held(a, "signal")["per_trade_ticks"]
                > held(b, "signal")["per_trade_ticks"])
