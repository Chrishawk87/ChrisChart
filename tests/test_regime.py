"""Error bars for the read, and the one rule they must never break.

The imbalance is the signal. These metrics tag the state it arrived
in -- they may tell you to stand down, and that is all. Most of what
is checked here is that they cannot do more than that, and that the
words on the panel stay off it until something has measured them.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest                                              # noqa: E402

from liqmap.regime import (Percentile, Regime,             # noqa: E402
                           RegimeSnapshot, RegimeTally)

SRC = (ROOT / "liqmap" / "regime.py").read_text()
TICK = 250_000_000.0
BID = 7_700 * 10 ** 9


def _r(**kw):
    kw.setdefault("bucket_vol", 100.0)
    kw.setdefault("buckets", 20)
    kw.setdefault("min_samples", 5)
    return Regime(**kw)


def _book(r, ts, spread_ticks=1.0):
    r.on_book(ts, int(BID), 100, int(BID + spread_ticks * TICK), 100,
              TICK)


class TestItSuggestsAndDoesNotExecute:
    def test_it_places_nothing(self):
        for bad in ("def submit(", "def execute(", "def place(",
                    "def order("):
            assert bad not in SRC, bad

    def test_it_says_so(self):
        assert "places, sizes or times an" in SRC


class TestARegimeCanNeverFlipTheRead:
    """The one rule. A regime that could change the direction would
    put two signals on the chart with no way to say which spoke."""

    def test_flowread_does_not_import_this(self):
        fr = (ROOT / "liqmap" / "flowread.py").read_text()
        assert "regime" not in fr.lower().split('"""', 2)[2], \
            "the reading can see the regime"

    def test_nothing_here_returns_a_direction(self):
        """If it cannot produce 'up' or 'down' it cannot override
        one, whatever anybody wires it to later."""
        r = _r()
        for i in range(400):
            ts = i * 0.05
            _book(r, ts)
            r.on_trade(ts, 10, "buy")
            r.on_action(ts, "A", 20)
        d = r.snapshot.as_dict()
        for k, v in d.items():
            assert v not in ("up", "down"), f"{k} carries a direction"

    def test_the_module_states_the_rule(self):
        assert "nothing in this module may change" in SRC
        assert "veto" in SRC.lower() or "stand down" in SRC.lower()


class TestTheSpread:
    def test_one_tick_is_tight_and_more_is_wide(self):
        r = _r()
        _book(r, 1.0, spread_ticks=1.0)
        assert r.snapshot.spread_band == "tight"
        _book(r, 2.0, spread_ticks=3.0)
        assert r.snapshot.spread_ticks == 3.0
        assert r.snapshot.spread_band == "wide"

    def test_a_crossed_book_is_withheld_not_recorded_as_zero(self):
        """A spread of zero would read as the tightest book of the
        day and would sum like a measurement."""
        r = _r()
        _book(r, 1.0, spread_ticks=2.0)
        before = r.snapshot.spread_ticks
        r.on_book(2.0, int(BID + 2 * TICK), 100, int(BID), 100, TICK)
        assert r.snapshot.spread_ticks == before

    def test_this_is_the_only_absolute_threshold(self):
        """Every other band is a percentile of the session, because
        a constant would quietly mean something different each day.
        The spread can be absolute because the market defines the
        minimum increment."""
        assert "TIGHT_TICKS = 1.0" in SRC
        assert "PERCENTILES OF THE SESSION, NOT CONSTANTS" in SRC


class TestVpin:
    def test_one_sided_flow_drives_it_to_one(self):
        r = _r()
        for i in range(400):
            r.on_trade(i * 0.1, 10, "buy")
        assert r.snapshot.vpin == pytest.approx(1.0)

    def test_balanced_flow_drives_it_to_zero(self):
        r = _r()
        for i in range(400):
            r.on_trade(i * 0.1, 10, "buy" if i % 2 else "sell")
        assert r.snapshot.vpin < 0.05

    def test_buckets_are_equal_volume_not_equal_clock(self):
        """The whole of what volume-synchronised means. A print that
        overfills a bucket has its excess carried to the next."""
        r = _r(bucket_vol=100.0)
        r.on_trade(1.0, 250, "buy")          # two and a half buckets
        assert r.snapshot.buckets == 2
        assert "the excess belongs to the next" in SRC

    def test_it_uses_the_stated_side_and_never_guesses(self):
        assert "STATED AGGRESSOR, NOT ON THE PAPER'S ESTIMATE" in SRC
        assert "tick rule" not in SRC.lower() or "refuses" in SRC

    def test_a_print_with_no_stated_side_is_not_guessed(self):
        """It still fills the bucket -- dropping it would make the
        bucket finish late -- but it joins neither side."""
        r = _r(bucket_vol=100.0)
        for _ in range(10):
            r.on_trade(1.0, 10, "")
        assert r.snapshot.buckets == 1
        assert r.snapshot.vpin == pytest.approx(0.0)

    def test_the_contested_evidence_is_in_the_code(self):
        """Not in a footnote somewhere. Someone reading this later
        has to meet the argument."""
        assert "Andersen and Bondarenko" in SRC
        assert "contested" in SRC.lower()


class TestCancellationIntensity:
    def test_it_counts_stated_actions_rather_than_inferring_them(self):
        """Size going down also happens on every fill and every
        modify -- three different things under one name."""
        assert "COUNTED here rather" in SRC
        assert "three different" in SRC and "one name" in SRC

    def test_all_cancels_reads_one(self):
        r = _r()
        for i in range(60):
            r.on_action(i * 0.1, "C", 10)
        assert r.snapshot.cancel_rate == pytest.approx(1.0)

    def test_all_adds_reads_zero(self):
        r = _r()
        for i in range(60):
            r.on_action(i * 0.1, "A", 10)
        assert r.snapshot.cancel_rate == pytest.approx(0.0)

    def test_trades_and_fills_are_not_cancellations(self):
        r = _r()
        for i in range(60):
            r.on_action(i * 0.1, "T", 10)
            r.on_action(i * 0.1, "F", 10)
        assert r.snapshot.cancel_rate == 0.0
        assert r.snapshot.cancel_n == 0

    def test_old_events_leave_the_window(self):
        r = _r(cancel_window_s=5.0)
        for i in range(40):
            r.on_action(i * 0.1, "C", 10)
        assert r.snapshot.cancel_rate == pytest.approx(1.0)
        for i in range(40):
            r.on_action(10.0 + i * 0.1, "A", 10)
        assert r.snapshot.cancel_rate == pytest.approx(0.0)


class TestTheBandsRefuseToSpeakTooEarly:
    """A threshold computed from nine observations is not a
    threshold, and a band drawn from one is a label with nothing
    behind it."""

    def test_no_band_before_enough_samples(self):
        r = _r(min_samples=200)
        for i in range(30):
            r.on_trade(i * 0.1, 200, "buy")
            r.on_action(i * 0.1, "C", 10)
        s = r.snapshot
        assert s.vpin > 0, "the metric itself should still be real"
        assert s.vpin_band == ""
        assert s.cancel_band == ""
        assert s.known is False

    def test_a_partial_tag_is_no_tag(self):
        """Otherwise 'we did not know' becomes its own regime and
        quietly collects outcomes under a name that means nothing."""
        s = RegimeSnapshot(spread_band="tight", vpin_band="high",
                           cancel_band="")
        assert s.tag == ""
        assert "A partial tag would quietly become" in SRC

    def test_the_sample_count_is_reported(self):
        r = _r(min_samples=5)
        for i in range(200):
            r.on_trade(i * 0.1, 50, "buy" if i % 3 else "sell")
            r.on_action(i * 0.1, "A", 10)
        s = r.snapshot
        assert s.vpin_n > 0 and s.cancel_n > 0

    def test_the_tag_names_every_band_it_is_made_of(self):
        r = _r(min_samples=5)
        for i in range(300):
            _book(r, i * 0.1)
            r.on_trade(i * 0.1, 50, "buy" if i % 4 else "sell")
            r.on_action(i * 0.1, "C" if i % 2 else "A", 10)
        t = r.snapshot.tag
        assert t, "no tag formed"
        assert "vpin" in t and "cancel" in t
        assert "tight" in t or "wide" in t


class TestThePercentile:
    def test_it_is_bounded(self):
        p = Percentile(cap=100)
        for i in range(10_000):
            p.add(i)
        assert len(p) == 100

    def test_it_refuses_a_band_without_enough_behind_it(self):
        p = Percentile()
        for i in range(5):
            p.add(i)
        assert p.band(3, min_n=30) == ""

    def test_high_and_low_split_at_the_percentile(self):
        p = Percentile()
        for i in range(100):
            p.add(i)
        assert p.band(95, 0.70, 10) == "high"
        assert p.band(5, 0.70, 10) == "low"


class TestTheTableIsAResultNotAnAssumption:
    """'High confidence', 'toxic', 'suspect' are hypotheses about one
    person's market. Printing them before anything has checked them
    would be the chart asserting what it was told."""

    def test_no_interpretation_words_are_produced(self):
        """Checked on the OUTPUT, not on the prose. The module is
        free to explain why it refuses to print these words; it must
        not actually print one."""
        rng = random.Random(5)
        r = _r(min_samples=10)
        ts = 0.0
        for _ in range(3_000):
            ts += 0.02
            _book(r, ts, 1.0 if rng.random() < 0.8 else 2.0)
            if rng.random() < 0.4:
                r.on_trade(ts, rng.randint(1, 30),
                           "buy" if rng.random() < 0.6 else "sell")
            if rng.random() < 0.5:
                r.on_action(ts, "C" if rng.random() < 0.5 else "A",
                            rng.randint(1, 40))
        t = RegimeTally()
        for i in range(60):
            t.record(r.snapshot.tag, "up", "up" if i % 3 else "down")
        produced = (repr(r.snapshot.as_dict())
                    + repr(t.as_dict())).lower()
        for word in ("toxic", "dangerous", "high confidence",
                     "suspect", "fragile", "clean", "informed"):
            assert word not in produced, f"an unmeasured verdict: {word}"

    def test_a_regime_with_too_few_outcomes_says_so(self):
        t = RegimeTally()
        for _ in range(5):
            t.record("lowvpin-tight-normalcancel", "up", "up")
        got = t.of("lowvpin-tight-normalcancel", min_n=30)
        assert got["enough"] is False
        assert "pct" not in got

    def test_an_unseen_regime_returns_nothing_rather_than_zero(self):
        assert RegimeTally().of("never-seen") is None

    def test_flats_are_counted_apart_from_wrong(self):
        t = RegimeTally()
        for _ in range(40):
            t.record("x", "up", "")
        for _ in range(30):
            t.record("x", "up", "up")
        got = t.of("x", min_n=10)
        assert got["flat"] == 40
        assert got["right"] == 30 and got["wrong"] == 0
        assert got["pct"] == 1.0, "flats were folded into the rate"

    def test_an_untagged_candle_is_not_recorded(self):
        t = RegimeTally()
        t.record("", "up", "up")
        assert t.rows == {}

    def test_a_candle_with_no_read_is_not_recorded(self):
        t = RegimeTally()
        t.record("x", "", "up")
        assert t.rows == {}


class TestItHoldsUpOnAMessyFeed:
    def test_a_long_random_session_produces_a_tag(self):
        rng = random.Random(11)
        r = _r(min_samples=20)
        ts = 0.0
        for _ in range(5_000):
            ts += 0.02
            _book(r, ts, 1.0 if rng.random() < 0.9 else 2.0)
            if rng.random() < 0.3:
                r.on_trade(ts, rng.randint(1, 30),
                           "buy" if rng.random() < 0.55 else "sell")
            if rng.random() < 0.5:
                r.on_action(ts, "C" if rng.random() < 0.6 else "A",
                            rng.randint(1, 40))
        s = r.snapshot
        assert s.known and s.tag
        assert 0.0 <= s.vpin <= 1.0
        assert 0.0 <= s.cancel_rate <= 1.0

    def test_reset_clears_it(self):
        r = _r()
        for i in range(400):
            _book(r, i * 0.1)
            r.on_trade(i * 0.1, 50, "buy")
            r.on_action(i * 0.1, "C", 10)
        r.reset()
        s = r.snapshot
        assert s.vpin == 0.0 and s.cancel_rate == 0.0
        assert s.buckets == 0 and s.tag == ""


class TestItCanKeepUpWithTheTape:
    """ES prints thousands of book updates a second. A metric that
    re-sums its window on every one of them is real CPU burned on
    every tick, and the first version of this did exactly that."""

    def test_the_cancel_window_is_maintained_incrementally(self):
        i = SRC.index("def on_action")
        body = SRC[i:SRC.index("def set_gross")]
        assert "sum(" not in body, "it re-sums the window every event"
        assert "self._added" in body and "self._pulled" in body

    def test_two_hundred_thousand_events_is_fast(self):
        import time
        rng = random.Random(1)
        r = Regime()
        ts = 0.0
        t0 = time.perf_counter()
        for _ in range(200_000):
            ts += 0.002
            r.on_action(ts, "C" if rng.random() < 0.5 else "A",
                        rng.randint(1, 40))
        assert time.perf_counter() - t0 < 5.0

    def test_the_running_totals_match_a_full_resum(self):
        """The optimisation has to produce the same number, or it is
        a different metric that happens to be quicker."""
        rng = random.Random(9)
        r = Regime(cancel_window_s=5.0)
        ts = 0.0
        for _ in range(4_000):
            ts += 0.005
            r.on_action(ts, "C" if rng.random() < 0.4 else "A",
                        rng.randint(1, 40))
        added = sum(s for _, x, s in r._events if x == "A")
        pulled = sum(s for _, x, s in r._events if x == "C")
        total = added + pulled
        want = (pulled / total) if total else 0.0
        assert r.snapshot.cancel_rate == pytest.approx(want)
