"""The holdout harness, and the guards that make it honest.

A grid of a few hundred configurations against 400 hours will always
produce a winner. The question is whether the winner is skill or
this sample, and only tape the search never touched can answer it.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "fh_mod", ROOT / "scripts" / "filter_holdout.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["fh_mod"] = m
    spec.loader.exec_module(m)
    return m


fh = _load()


def tape(n: int, drift: float, seed: int = 3) -> list:
    rng = random.Random(seed)
    px, t, out = 6800.0, 1_700_000_000.0, []
    for _ in range(n):
        t += 0.4
        if rng.random() < 0.6:
            px += 0.25 if rng.random() < drift else -0.25
        out.append((t, round(px, 2), float(rng.randint(1, 20)),
                    "B" if rng.random() < drift else "A"))
    return out


class TestTheFiltersOnlyVeto:
    """None of them may open a trade the rule did not want."""

    def test_a_filter_never_increases_the_trade_count(self):
        t = tape(60_000, 0.51)
        base = fh.score(fh.run(t, fh.Cfg(), 4, 4, 1.0, 1, 1, 60.0), 1.0)
        for cfg in (fh.Cfg(push=0.05), fh.Cfg(eff=True),
                    fh.Cfg(dmin=60.0), fh.Cfg(push=0.1, dmin=150.0)):
            s = fh.score(fh.run(t, cfg, 4, 4, 1.0, 1, 1, 60.0), 1.0)
            assert s["trades"] <= base["trades"], cfg.label()

    def test_a_stricter_threshold_takes_no_more_trades(self):
        t = tape(60_000, 0.51)
        counts = [fh.score(fh.run(t, fh.Cfg(push=p), 4, 4, 1.0, 1, 1,
                                  60.0), 1.0)["trades"]
                  for p in (0.0, 0.05, 0.10, 0.20, 0.35)]
        assert counts == sorted(counts, reverse=True), counts

    def test_the_baseline_is_the_rule_untouched(self):
        assert fh.Cfg().label() == "no filter (baseline)"


class TestTheGuardsAgainstFoolingOurselves:
    def test_a_thin_configuration_cannot_be_chosen(self):
        """Without a floor the grid reliably picks whichever filter
        left a handful of winners, and the holdout then has nothing
        to disagree with."""
        src = (ROOT / "scripts" / "filter_holdout.py").read_text()
        assert "MIN_TRADES = 30" in src
        assert 'r[3]["trades"] >= MIN_TRADES' in src

    def test_the_excluded_count_is_reported(self):
        src = (ROOT / "scripts" / "filter_holdout.py").read_text()
        assert "were\\n  excluded" in src or "excluded" in src

    def test_the_number_of_configurations_tried_is_printed(self):
        """Honesty about multiple comparisons: a reader has to know
        how many tries produced the winner."""
        src = (ROOT / "scripts" / "filter_holdout.py").read_text()
        assert "configurations" in src
        assert "that many tries is exactly why the holdout exists" in src

    def test_the_holdout_is_named_up_front(self):
        """It is now the chronologically last file rather than the
        last by name -- see TestTheSplitIsByTimeNotByName."""
        src = (ROOT / "scripts" / "filter_holdout.py").read_text()
        assert "*fit_spans, hold_span = spans" in src
        assert "not looked at until one configuration has" in src

    def test_the_baseline_is_scored_on_the_holdout_too(self):
        """A filter that improves the fit and not the baseline-
        relative holdout has told you nothing."""
        src = (ROOT / "scripts" / "filter_holdout.py").read_text()
        assert "baseline, no filter" in src

    def test_the_gap_between_fit_and_holdout_is_reported(self):
        src = (ROOT / "scripts" / "filter_holdout.py").read_text()
        assert "how much was this " in src

    def test_it_says_so_when_nothing_survives(self):
        src = (ROOT / "scripts" / "filter_holdout.py").read_text()
        assert "IT DID NOT SURVIVE" in src
        assert "rather than widen the grid" in src

    def test_two_files_are_required(self):
        src = (ROOT / "scripts" / "filter_holdout.py").read_text()
        assert "Need at least two" in src


class TestTheRuleItWraps:
    def test_it_calls_the_real_entry_and_exit(self):
        """A backtest that reimplements the rule measures the
        backtest."""
        src = (ROOT / "scripts" / "filter_holdout.py").read_text()
        assert "tm.entry_side(d1, d5)" in src
        assert "tm.reversed_against(" in src
        assert "from liqmap import timed as tm" in src

    def test_it_uses_the_real_filters(self):
        src = (ROOT / "scripts" / "filter_holdout.py").read_text()
        assert "from liqmap.kinetic import Kinetic" in src
        assert "from liqmap.dscore import DScore" in src

    def test_one_trade_per_five_minute_candle_still_holds(self):
        t = tape(60_000, 0.51)
        trades = fh.run(t, fh.Cfg(), 4, 4, 1.0, 1, 1, 60.0)
        bars = [int(x.t0 // 300) for x in trades]
        assert len(bars) == len(set(bars)), "a candle took two trades"

    def test_the_stop_is_checked_before_the_target(self):
        src = (ROOT / "scripts" / "filter_holdout.py").read_text()
        i = src.index('why = "sl"')
        j = src.index('why = "tp"')
        assert i < j, "a bar through both must record as stopped out"

    def test_a_cost_is_charged_on_every_trade(self):
        t = tape(30_000, 0.51)
        trades = fh.run(t, fh.Cfg(), 4, 4, 1.0, 1, 1, 60.0)
        free = fh.score(trades, 0.0)
        paid = fh.score(trades, 1.0)
        if free["trades"]:
            assert paid["net"] == pytest.approx(
                free["net"] - free["trades"], rel=1e-6)


class TestTheSplitIsByTimeNotByName:
    """The first real run picked an OHLCV file from 2024 as the
    holdout for three MBP-10 files from 2026, because it sorted by
    filename. Two separate faults, both fixed here.
    """

    def test_a_file_with_no_prints_is_skipped_by_name(self, monkeypatch):
        """An OHLCV file is bars, not tape: it loads as nothing.
        That is a reason to leave it out with a word, not to crash
        four lines later on an empty list."""
        fake = {"good.dbn.zst": [(1.0, 6800.0, 1.0, "B")],
                "bars_ohlcv-1s.dbn.zst": []}
        monkeypatch.setattr(fh, "load_one", lambda p: fake[p])
        spans = fh.file_spans(list(fake))
        assert [s[0] for s in spans] == ["good.dbn.zst"]

    def test_the_holdout_is_the_latest_tape(self, monkeypatch):
        """Fit on what happened, test on what happened next. A
        name-sorted split validated a 2026 fit against 2024 tape,
        which is not a holdout, it is a different market."""
        fake = {
            "z_sept.dbn.zst": [(1_758_400_000.0, 6800.0, 1.0, "B")],
            "a_oct.dbn.zst": [(1_759_900_000.0, 6800.0, 1.0, "B")],
            "m_sept_late.dbn.zst": [(1_758_800_000.0, 6800.0, 1.0, "B")],
        }
        monkeypatch.setattr(fh, "load_one", lambda p: fake[p])
        spans = sorted(fh.file_spans(list(fake)), key=lambda r: r[2])
        assert spans[-1][0] == "a_oct.dbn.zst"
        assert [s[0] for s in spans[:-1]] == ["z_sept.dbn.zst",
                                              "m_sept_late.dbn.zst"]

    def test_the_source_sorts_by_timestamp(self):
        src = (ROOT / "scripts" / "filter_holdout.py").read_text()
        assert "spans.sort(key=lambda r: r[2])" in src
        assert "*fit_spans, hold_span = spans" in src

    def test_it_says_why_the_split_is_chronological(self):
        src = (ROOT / "scripts" / "filter_holdout.py").read_text()
        assert "SPLIT BY TIME, NOT BY FILENAME" in src

    def test_too_few_usable_files_is_a_message_not_a_crash(self):
        src = (ROOT / "scripts" / "filter_holdout.py").read_text()
        assert "usable file(s). Need at least two" in src


class TestAZeroTradeFilterIsNotAWin:
    """The chosen filter took NO trades on the holdout and the
    script reported it as beating the baseline, because 0.000 is
    greater than -1.308.

    The 30-trade floor was applied when CHOOSING on the fit set and
    not when SCORING on the holdout -- the same mistake in the same
    script, one step later.
    """

    def test_the_holdout_has_its_own_floor(self):
        src = (ROOT / "scripts" / "filter_holdout.py").read_text()
        assert "HOLD_MIN = 10" in src
        assert 'hc["trades"] < HOLD_MIN' in src

    def test_it_refuses_to_call_that_a_result(self):
        src = (ROOT / "scripts" / "filter_holdout.py").read_text()
        assert "That is not a result" in src
        assert "is not a win" in src

    def test_it_still_reports_what_the_holdout_did_say(self):
        """The rule's own number on unseen tape is the finding, even
        when the filter was never tested."""
        src = (ROOT / "scripts" / "filter_holdout.py").read_text()
        assert "What the holdout DOES say" in src


class TestGrossIsReported:
    """Net and gross call for completely different work and only
    the gross number says which."""

    def test_the_tail_exists_and_is_called_on_both_paths(self):
        src = (ROOT / "scripts" / "filter_holdout.py").read_text()
        assert "def _gross_tail(" in src
        assert src.count("_gross_tail(hb, a.cost)") == 2

    def test_a_coin_flip_is_named_as_one(self):
        m = fh
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m._gross_tail({"trades": 1194, "net": -1646.0}, 1.0)
        out = buf.getvalue()
        assert "-0.379 a trade" in out
        assert "coin flip" in out
        assert "no filter on a" in out

    def test_a_positive_gross_is_called_a_cost_problem(self):
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fh._gross_tail({"trades": 100, "net": 20.0}, 1.0)
        out = buf.getvalue()
        assert "+1.200 a trade" in out
        assert "round trip is eating them" in out

    def test_a_badly_negative_gross_is_called_a_direction_problem(self):
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fh._gross_tail({"trades": 100, "net": -300.0}, 1.0)
        out = buf.getvalue()
        assert "direction itself" in out

    def test_no_trades_prints_nothing(self):
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fh._gross_tail({"trades": 0, "net": 0.0}, 1.0)
        assert buf.getvalue() == ""
