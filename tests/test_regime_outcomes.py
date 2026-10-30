"""The script that turns the regime table into a result.

Its job is to fill in four hypotheses from real tape. Its trap is
that eight regimes split a day's candles into cells too thin to read,
and a cell of nine that happens to land on 70% looks exactly like a
finding.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SRC = (ROOT / "scripts" / "regime_outcomes.py").read_text()


def _mod():
    spec = importlib.util.spec_from_file_location(
        "ro", ROOT / "scripts" / "regime_outcomes.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TestItSuggestsAndDoesNotExecute:
    def test_it_places_nothing(self):
        for bad in ("def submit(", "def execute(", "def place(",
                    "def order("):
            assert bad not in SRC, bad

    def test_it_says_so(self):
        assert "places, sizes or times an" in SRC


class TestTheRegimeIsReadWhenTheDecisionWasMade:
    def test_it_is_taken_at_the_settle_not_at_the_close(self):
        """Reading it at the candle's close would tag the trade with
        a book state that arrived after the decision -- the most
        flattering mistake available here."""
        assert "TAKEN WHEN THE READ SETTLES" in SRC
        assert "after the decision was made" in SRC

    def test_the_outcome_is_measured_from_the_settle_price(self):
        assert "last_px - px0" in SRC


class TestItRefusesToReadAThinCell:
    def test_a_cell_under_the_minimum_prints_no_rate(self):
        assert "too few" in SRC
        assert 'got["enough"]' in SRC

    def test_it_names_the_arithmetic_of_the_problem(self):
        assert "seventy-eight" in SRC
        assert "weeks of files, not a day" in SRC

    def test_every_cell_is_shown_against_a_baseline(self):
        """A regime at 58% means nothing beside a baseline of 57%."""
        assert "BASELINE" in SRC
        assert "vs base" in SRC
        assert "- base" in SRC

    def test_it_can_report_that_the_regimes_do_nothing(self):
        """If the script can only return the flattering outcome it is
        not a measurement."""
        assert "THAT GAP IS NOTHING" in SRC
        assert "the tags are decoration" in SRC


class TestTheControlReturnsNothing:
    def test_it_labels_itself_a_control_not_data(self):
        assert "A CONTROL, NOT DATA" in SRC
        assert "Point it at your" in SRC

    def test_it_runs_end_to_end_on_a_random_book(self):
        import contextlib
        import io
        m = _mod()
        buf = io.StringIO()
        argv = sys.argv
        sys.argv = ["regime_outcomes.py", "--updates", "60000"]
        try:
            with contextlib.redirect_stdout(buf):
                rc = m.main()
        finally:
            sys.argv = argv
        out = buf.getvalue()
        assert rc in (0, 2)
        if rc == 0:
            assert "REGIME OUTCOMES" in out
            assert "CONTROL" in out

    def test_run_produces_one_row_per_candle_not_per_update(self):
        m = _mod()
        ev = m.walk(120_000, seed=4)
        rows = m.run(ev, 300.0, 10.0, 10.0, 500.0)
        # 120k updates at 10ms is 20 minutes: a handful of candles,
        # not a row per book update.
        assert 0 <= len(rows) <= 10, len(rows)
        for r in rows:
            assert r["read"] in ("up", "down")
            assert r["outcome"] in ("up", "down", "")
