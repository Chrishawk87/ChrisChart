"""Does weighting by contracts say anything counting ticks does not?

The script that answers it has one job and one trap. The job is to
score three readings of the same candle against that candle's close.
The trap is that two readings of one tape agree most of the time by
construction, so an agreement rate looks like a result and is not --
only the disagreements carry the question.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SRC = (ROOT / "scripts" / "tick_vs_contract.py").read_text()


def _mod():
    spec = importlib.util.spec_from_file_location(
        "tvc", ROOT / "scripts" / "tick_vs_contract.py")
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


class TestItDoesNotBuildTheInferiorTapeReading:
    """CME states the aggressor on every print. A tick rule guesses
    it, and guesses worst exactly when the market is fast."""

    def test_there_is_no_uptick_downtick_series(self):
        assert "uptick" in SRC, "the choice should be documented"
        assert "tick_rule" not in SRC
        assert "inferior version of DELTA" in SRC

    def test_delta_comes_from_the_stated_aggressor(self):
        assert "tr.aggressor" in SRC
        assert "THE STATED AGGRESSOR, not a tick rule." in SRC


class TestTheThreeReadingsAreActuallyDifferent:
    def test_the_tick_series_carries_no_size(self):
        """If it did, it would be OFI with extra steps."""
        i = SRC.index("cur.ticks +=")
        line = SRC[i:SRC.index("\n", i)]
        assert "RAW_TICK" in line
        for word in ("sz", "size", "gross", "e)"):
            assert word not in line, f"size leaked into the path: {word}"

    def test_the_ofi_series_is_the_imported_formula(self):
        assert "from liqmap.ofi import Touch, level_ofi" in SRC
        assert "level_ofi(prev, t)" in SRC


class TestItScoresAgainstTheCandleClose:
    def test_a_flat_close_is_its_own_column(self):
        m = _mod()
        assert m._dir(0.0) == m.NONE
        c = m.Candle(0.0, 100.0)
        c.at_px = 100.0
        c.close_px = 100.0
        assert c.outcome == m.NONE

    def test_flats_are_not_folded_into_either_side(self):
        m = _mod()

        class C:
            pass
        flat = m.Candle(0.0, 100.0)
        flat.at_px = 100.0
        flat.close_px = 100.0
        flat.ofi_at = 500.0
        r = m.score([flat], lambda c: c.ofi_at)
        assert r["flat"] == 1
        assert r["right"] == 0 and r["wrong"] == 0
        assert r["decided"] == 0

    def test_a_reading_of_zero_is_silent_not_wrong(self):
        m = _mod()
        c = m.Candle(0.0, 100.0)
        c.at_px = 100.0
        c.close_px = 200.0
        c.ofi_at = 0.0
        r = m.score([c], lambda x: x.ofi_at)
        assert r["silent"] == 1
        assert r["decided"] == 0

    def test_right_and_wrong_are_measured_from_the_reading_not_the_open(self):
        """The reading is taken part-way into the candle. Scoring from
        the candle's OPEN would credit it with the move that happened
        before it existed."""
        assert "self.close_px - self.at_px" in SRC
        m = _mod()
        c = m.Candle(0.0, 100.0)
        c.at_px = 150.0          # price had already run before the read
        c.close_px = 120.0
        assert c.outcome == m.DOWN, "it scored from the open"


class TestTheDisagreementsAreTheQuestion:
    def test_the_script_says_agreement_alone_is_not_a_result(self):
        assert "agree most of the time" in SRC or \
               "agree most of the time by" in SRC
        assert "THE COLUMN THAT MATTERS" in SRC

    def test_it_scores_both_sides_of_a_disagreement(self):
        assert "disagreeing, the BOOK's side" in SRC
        assert "disagreeing, the PATH's side" in SRC

    def test_it_refuses_to_read_a_small_disagreement_sample(self):
        assert 'ro["decided"] < 100' in SRC
        assert "not a sample" in SRC

    def test_it_says_how_much_tape_would_be_needed(self):
        assert "hours of tape" in SRC
        assert "linked by" in SRC and "construction" in SRC

    def test_it_names_the_outcome_where_the_path_wins(self):
        """The result that would say the chart is built around the
        wrong thing. If the script can only report the flattering
        outcome it is not a measurement."""
        assert "THE PATH WINS THE DISAGREEMENTS" in SRC
        assert "contract" in SRC and "costing you" in SRC

    def test_it_names_the_outcome_where_neither_wins(self):
        assert "NEITHER WINS THE DISAGREEMENTS" in SRC


class TestTheControlReturnsNothing:
    def test_a_random_book_scores_at_chance(self):
        """Zero has to look like zero, or no result on real data can
        be believed."""
        m = _mod()
        ev = m.walk(600_000, seed=5)
        cs = [c for c in m.build(ev, 300.0, 25.0) if c.n > 20]
        assert len(cs) >= 10
        for pick in (lambda c: c.ofi_at, lambda c: c.ticks_at,
                     lambda c: c.delta_at):
            r = m.score(cs, pick)
            if r["decided"] >= 10:
                assert 0.2 < r["pct"] < 0.8, r

    def test_it_labels_itself_as_a_control_not_as_data(self):
        assert "A CONTROL, NOT DATA" in SRC
        assert "Point it at your .dbn files" in SRC
