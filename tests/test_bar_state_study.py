"""Does a candle's state precede the next one -- and can this say so.

In order of how badly each would mislead:

  a scan that cannot detect an edge that is there. Its null result is
  not evidence of absence, it is evidence of nothing, and it is the
  most comforting possible way to be wrong

  calling a significant difference a finding. At a hundred thousand
  bars a z of 3 is nearly free: run on a pure random walk this study's
  own aggression table returns |z| = 3.49 on a mean of 0.0016 ticks --
  two cents against five dollars of cost

  a label that knows the future. One bar of leak turns the whole table
  into a measurement of itself

  shuffling the outcomes instead of the labels. Forward returns at
  k > 1 overlap, and destroying that structure in the null while
  keeping it in the observed manufactures significance
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "bar_state_study", ROOT / "scripts" / "bar_state_study.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

from liqmap.ofibar import OfiBarBuilder  # noqa: E402

RAW = S.RAW_TICK


def walk(n=120_000, seed=5, threshold=250, trades=True):
    rng = random.Random(seed)
    b = OfiBarBuilder(tick_size=0.25, threshold=threshold, max_s=900.0,
                      min_updates=2, max_bars=10_000_000)
    bid, ts = 20_000_000_000, 0.0
    for _ in range(n):
        if rng.random() < 0.08:
            bid += RAW if rng.random() < 0.5 else -RAW
        ts += 0.01
        if trades and rng.random() < 0.3:
            b.on_trade(ts, rng.randint(1, 20),
                       "buy" if rng.random() < 0.5 else "sell")
        b.on_book(ts, bid, rng.randint(50, 600), bid + RAW,
                  rng.randint(50, 600))
    return b.bars


class TestItCanSeeAnEdgeThatIsThere:
    """Otherwise a null result means nothing at all."""

    def test_a_planted_edge_is_found(self):
        rng = random.Random(1)
        moves = [rng.gauss(0, 2.0) for _ in range(20_000)]
        labels = []
        for mv in moves:
            if rng.random() < 0.58:
                labels.append("up" if mv > 0 else "dn")
            else:
                labels.append("up" if rng.random() < 0.5 else "dn")
        res = S.table(labels, moves, rng, shuffles=120)
        assert max(abs(r["z"]) for r in res["rows"]) > 6.0

    def test_a_meaningless_label_is_not_found(self):
        rng = random.Random(2)
        moves = [rng.gauss(0, 2.0) for _ in range(20_000)]
        labels = ["a" if rng.random() < 0.5 else "b" for _ in moves]
        res = S.table(labels, moves, rng, shuffles=120)
        assert max(abs(r["z"]) for r in res["rows"]) < 4.0

    def test_the_selftest_passes(self):
        assert S.selftest(random.Random(3)) == 0


class TestTheLabelsCannotSeeTheFuture:

    def test_a_state_is_read_from_its_own_bar_only(self):
        src = (ROOT / "scripts" / "bar_state_study.py").read_text()
        i = src.index("def states_of")
        body = src[i:src.index("def outcomes")]
        for w in ("bars[", "i + 1", "next", "[i +"):
            assert w not in body, w

    def test_the_outcome_starts_at_the_bar_s_own_close(self):
        bars = walk(20_000)
        mv = S.outcomes(bars, 1)
        want = (bars[1].close - bars[0].close) / RAW
        assert mv[0] == pytest.approx(want)

    def test_the_last_bars_are_dropped_not_zeroed(self):
        """A zero would sum like a measurement and pull every mean
        toward nothing."""
        bars = walk(20_000)
        for k in (1, 2, 4):
            assert len(S.outcomes(bars, k)) == len(bars) - k

    def test_labels_and_outcomes_stay_aligned(self):
        bars = walk(20_000)
        mv = S.outcomes(bars, 2)
        usable = bars[:len(mv)]
        assert len(usable) == len(mv)


class TestTheControl:

    def test_the_outcome_series_is_not_disturbed(self):
        """Forward returns overlap; destroying that in the null while
        keeping it in the observed manufactures significance."""
        src = (ROOT / "scripts" / "bar_state_study.py").read_text()
        i = src.index("def table(")
        body = src[i:src.index("def report")]
        assert "rng.shuffle(pool)" in body
        assert "rng.shuffle(moves)" not in body

    def test_the_null_is_built_from_the_same_labels(self):
        src = (ROOT / "scripts" / "bar_state_study.py").read_text()
        i = src.index("def table(")
        body = src[i:src.index("def report")]
        assert "pool = list(labels)" in body

    def test_every_cell_gets_its_own_null(self):
        rng = random.Random(4)
        moves = [rng.gauss(0, 1.0) for _ in range(4_000)]
        labels = [("a", "b", "c")[i % 3] for i in range(4_000)]
        res = S.table(labels, moves, rng, shuffles=60)
        assert {r["label"] for r in res["rows"]} == {"a", "b", "c"}
        assert all(r["null_sd"] > 0 for r in res["rows"])

    def test_a_rare_cell_is_marked_unreadable(self):
        rng = random.Random(5)
        moves = [rng.gauss(0, 1.0) for _ in range(4_000)]
        labels = ["rare" if i < 10 else "common" for i in range(4_000)]
        res = S.table(labels, moves, rng, shuffles=40)
        rare = [r for r in res["rows"] if r["label"] == "rare"][0]
        assert not rare["enough"]


class TestSignificanceIsNotAnEdge:
    """The trap that makes a large scan look profitable."""

    def test_a_tiny_mean_is_not_called_a_finding(self, capsys):
        rng = random.Random(6)
        n = 60_000
        moves = [rng.gauss(0, 2.0) for _ in range(n)]
        labels = []
        for mv in moves:
            # A real but microscopic effect: enough n makes it
            # significant, and it is worth nothing.
            labels.append("up" if (mv > 0 and rng.random() < 0.51)
                          or (mv <= 0 and rng.random() < 0.49) else "dn")
        res = S.table(labels, moves, rng, shuffles=150)
        S.report("tiny", res, 1, cost=0.4)
        out = capsys.readouterr().out
        assert "WORTH A LOOK" not in out, out

    def test_the_cost_gate_is_applied_against_the_base_rate(self, capsys):
        """A state is only interesting relative to every other bar, not
        relative to zero."""
        rng = random.Random(7)
        moves = [5.0] * 2_000 + [5.0] * 2_000
        labels = ["a"] * 2_000 + ["b"] * 2_000
        res = S.table(labels, moves, rng, shuffles=40)
        S.report("drift", res, 1, cost=0.4)
        out = capsys.readouterr().out
        assert "WORTH A LOOK" not in out, "a shared drift is not an edge"

    def test_a_real_and_large_effect_is_called_out(self, capsys):
        rng = random.Random(8)
        moves, labels = [], []
        for _ in range(8_000):
            up = rng.random() < 0.5
            labels.append("up" if up else "dn")
            moves.append(rng.gauss(2.0 if up else -2.0, 1.0))
        res = S.table(labels, moves, rng, shuffles=120)
        S.report("planted", res, 1, cost=0.4)
        assert "WORTH A LOOK" in capsys.readouterr().out

    def test_the_partition_warning_is_always_printed(self, capsys):
        rng = random.Random(9)
        moves = [rng.gauss(0, 1.0) for _ in range(3_000)]
        labels = [("a", "b")[i % 2] for i in range(3_000)]
        S.report("x", S.table(labels, moves, rng, shuffles=30), 1, 0.4)
        out = " ".join(capsys.readouterr().out.split())
        assert "partition" in out and "largest |z| only" in out


class TestItRunsOnRealBars:

    def test_every_state_is_a_partition_of_the_bars(self):
        bars = walk(80_000)
        assert len(bars) > 1_000
        mv = S.outcomes(bars, 1)
        usable = bars[:len(mv)]
        for name in ("direction", "aggression", "delta", "agrees",
                     "delta_agrees", "combined"):
            labels = [S.states_of(b)[name] for b in usable]
            assert len(labels) == len(usable)
            counts = {}
            for x in labels:
                counts[x] = counts.get(x, 0) + 1
            assert sum(counts.values()) == len(usable), name

    def test_the_delta_state_follows_the_sign(self):
        bars = walk(40_000)
        for b in bars[:200]:
            st = S.states_of(b)["delta"]
            assert st == ("buy" if b.delta > 0
                          else "sell" if b.delta < 0 else "flat")

    def test_a_feed_with_no_prints_puts_every_bar_in_flat(self):
        bars = walk(40_000, trades=False)
        assert bars
        assert all(S.states_of(b)["delta"] == "flat" for b in bars)

    def test_a_random_walk_produces_no_tradeable_finding(self, capsys):
        """The result that matters: this study must not find an edge in
        a market that has none."""
        rng = random.Random(10)
        bars = walk(120_000, seed=10)
        mv = S.outcomes(bars, 1)
        usable = bars[:len(mv)]
        for name in ("direction", "aggression", "delta", "combined"):
            labels = [S.states_of(b)[name] for b in usable]
            S.report(name, S.table(labels, mv, rng, 120), 1, cost=0.4)
        assert "WORTH A LOOK" not in capsys.readouterr().out


class TestItDecidesNothing:

    def test_no_order_code(self):
        src = (ROOT / "scripts" / "bar_state_study.py").read_text()
        for w in ("def submit(", "def execute(", "def place(", "def order("):
            assert w not in src, w

    def test_it_states_the_cost_arithmetic(self):
        src = (ROOT / "scripts" / "bar_state_study.py").read_text()
        assert "break_even = (stop + cost) / (target + stop)" in \
               src.replace("\n", " ").replace("  ", " ") or \
               "break_even =" in src

    def test_a_null_result_is_reported_as_a_result(self):
        src = (ROOT / "scripts" / "bar_state_study.py").read_text()
        assert "That is a result, not a failure" in src


class TestTheThreeConditionRule:
    """Caret level + candle + delta, and what it scores on nothing.

    In order of how badly each would mislead:

      the level's state read AFTER the bar is applied to it, so the
      trigger includes the very touch being used as the trigger

      a pivot admitted before its confirmation window closed. On the
      chart that pivot can still vanish; in a study it is a look-ahead
      leak wearing a level's name

      calling the rule a finding when a random walk scores the same. A
      swing low is lower than its neighbours BY DEFINITION, so the bar
      after one is usually higher -- the rule collects that whether or
      not the market has anything in it
    """

    def test_the_walk_lives_in_one_place(self):
        """The study grades the engine the panel draws, so it imports
        the walk rather than carrying a copy that can drift."""
        src = (ROOT / "scripts" / "bar_state_study.py").read_text()
        i = src.index("def level_walk")
        body = src[i:src.index("def synthetic_bars")]
        assert "from liqmap.levels import walk_levels" in body
        assert "return walk_levels(bars, left, right, keep)" in body

    def test_a_pivot_is_not_used_before_it_is_settled(self):
        src = (ROOT / "liqmap" / "levels.py").read_text()
        body = src[src.index("def walk_levels"):]
        assert "due.setdefault(q.index + right, []).append(q)" in body

    def test_the_level_state_is_read_before_the_bar_is_applied(self):
        src = (ROOT / "liqmap" / "levels.py").read_text()
        body = src[src.index("def walk_levels"):]
        read_at = body.index("status = swing_status")
        apply_at = body.index("for _, _, lv in live:")
        assert read_at < apply_at, "the trigger includes its own touch"

    def test_the_walk_is_causal(self):
        """Truncating the series must not change what earlier bars saw."""
        bars = S.synthetic_bars(900, 250, seed=11)
        i1, c1, x1 = S.level_walk(bars, 5, 1, 4)
        cut = len(bars) - 120
        i2, c2, x2 = S.level_walk(bars[:cut], 5, 1, 4)
        n = len(i2)
        assert n > 50
        assert i1[:n] == i2[:n]
        assert x1[:n] == x2[:n], "a later bar changed an earlier reading"

    def test_the_rule_is_the_predicate_it_says_it_is(self):
        class B:
            aggression = "BUY_PRICE_AGGRESSIVE"
            delta = 5.0

        at_low = {"side": "low", "status": "held"}
        assert S.rule_states(B(), at_low)["rule_triple"] == "long"

        B.delta = -5.0
        assert S.rule_states(B(), at_low)["rule_triple"] == "none"

        B.delta = 5.0
        assert S.rule_states(B(), {"side": "", "status": ""}
                             )["rule_triple"] == "none"

        B.aggression = "BALANCED_PRICE_FLOW"
        assert S.rule_states(B(), at_low)["rule_triple"] == "none"

    def test_absorption_is_delta_paying_without_the_candle(self):
        class B:
            aggression = "BALANCED_PRICE_FLOW"
            delta = 5.0

        ctx = {"side": "low", "status": "held"}
        st = S.rule_states(B(), ctx)
        assert st["rule_absorption"] == "long"
        assert st["rule_triple"] == "none", "the two must not overlap"

    def test_the_legs_are_measured_alone_as_well(self):
        class B:
            aggression = "BALANCED_PRICE_FLOW"
            delta = 5.0

        st = S.rule_states(B(), {"side": "low", "status": "held"})
        assert st["leg_at_level"] == "low"
        assert st["leg_level_delta"] == "long"
        assert st["leg_level_candle"] == "none"

    def test_the_noise_control_runs_the_real_builder(self):
        """A baseline built by another route excuses whatever it fails
        to reproduce."""
        src = (ROOT / "scripts" / "bar_state_study.py").read_text()
        i = src.index("def synthetic_bars")
        body = src[i:src.index("def rule_noise")]
        assert "from liqmap.ofibar import OfiBarBuilder" in body
        assert "b.on_trade(" in body and "b.on_book(" in body

    def test_synthetic_bars_carry_traded_prices(self):
        bars = S.synthetic_bars(400, 250, seed=5)
        assert len(bars) >= 400
        assert any(b.traded for b in bars)

    def test_the_rule_is_not_a_finding_on_a_random_walk(self, capsys):
        """The whole reason the second control exists. Without it this
        returns half a tick at |z| = 27 on data with nothing in it."""
        rng = random.Random(12)
        bars = S.synthetic_bars(30_000, 250, seed=31)
        idx, closes, ctx = S.level_walk(bars, 5, 1, 4)
        mv = [(closes[i + 1] - closes[i]) / RAW
              for i in range(len(closes) - 1)]
        usable = [bars[idx[i]] for i in range(len(mv))]
        noise = S.rule_noise(len(bars), 250, 5, 1, 4, 1,
                             ("rule_triple",), seed=77)
        labels = [S.rule_states(usable[i], ctx[i])["rule_triple"]
                  for i in range(len(mv))]
        res = S.table(labels, mv, rng, 120)
        S.report("rule_triple", res, 1, 0.4, noise.get("rule_triple", {}))
        out = capsys.readouterr().out
        assert "WORTH A LOOK" not in out, out
        assert "MATCHED BY NOISE" in out, out

    def test_without_the_noise_control_it_would_have_been_a_finding(self):
        """Stated as a test so nobody quietly removes the control."""
        rng = random.Random(12)
        bars = S.synthetic_bars(30_000, 250, seed=31)
        idx, closes, ctx = S.level_walk(bars, 5, 1, 4)
        mv = [(closes[i + 1] - closes[i]) / RAW
              for i in range(len(closes) - 1)]
        usable = [bars[idx[i]] for i in range(len(mv))]
        labels = [S.rule_states(usable[i], ctx[i])["rule_triple"]
                  for i in range(len(mv))]
        res = S.table(labels, mv, rng, 120)
        hits = [r for r in res["rows"]
                if r["enough"] and abs(r["z"]) >= 3.0
                and abs(r["mean"] - res["base"]) >= 0.4]
        assert hits, "the shuffle control alone passes this rule"

    def test_the_report_names_the_extremum_bias(self):
        src = (ROOT / "scripts" / "bar_state_study.py").read_text()
        assert "lower than its neighbours BY" in src
        assert "MATCHED BY NOISE" in src


class TestTheLoaderCarriesThePrice:
    """Without it every bar reads as 'no prints', the rule section
    skips in one line, and the study measures nothing while looking
    like it ran."""

    def test_the_trade_event_carries_a_raw_price(self):
        src = (ROOT / "scripts" / "bar_state_study.py").read_text()
        i = src.index("def iter_records")
        body = src[i:src.index("def build")]
        assert 'float(dbn._get(rec, "price", 0) or 0)' in body
        code = [ln for ln in body.splitlines()
                if not ln.strip().startswith("#")]
        assert not any("t.px" in ln for ln in code), \
            "t.px is dollars; the bar's prices are raw units"

    def test_the_builder_is_given_it(self):
        src = (ROOT / "scripts" / "bar_state_study.py").read_text()
        assert "b.on_trade(ev[1], ev[2], ev[3], ev[4])" in src

    def test_a_feed_without_prices_produces_no_traded_bars(self):
        """The failure this guards against, stated directly."""
        from liqmap.ofibar import OfiBarBuilder
        b = OfiBarBuilder(tick_size=0.25, threshold=50, max_s=900.0,
                          min_updates=2)
        bid = 7_700 * 10 ** 9
        b.on_book(0.0, bid, 100, bid + RAW, 100)
        for i in range(1, 20):
            b.on_trade(i * 0.01, 5, "buy")          # no px
            b.on_book(i * 0.01, bid, 100 + i * 40, bid + RAW, 100)
        assert b.bars
        assert not any(x.traded for x in b.bars)
        assert all(x.trade_count > 0 for x in b.bars), \
            "delta still works; only the traded prices are missing"
