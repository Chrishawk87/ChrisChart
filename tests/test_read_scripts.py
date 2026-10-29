"""The two scripts that decide whether the reading is worth anything.

One picks the window and the hold; the other asks whether the read is
right. Both exist because the numbers they produce were otherwise
going to be asserted from the armchair.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TUNE = (ROOT / "scripts" / "read_tuning.py").read_text()
ACC = (ROOT / "scripts" / "read_accuracy.py").read_text()


class TestNeitherPlacesAnOrder:
    def test_no_execution(self):
        for src, name in ((TUNE, "read_tuning"), (ACC, "read_accuracy")):
            for bad in ("def submit(", "def execute(", "def place(",
                        "def order("):
                assert bad not in src, f"{name}: {bad}"

    def test_both_say_so(self):
        for src in (TUNE, ACC):
            assert "places, sizes or times an" in src


class TestTuningRefusesToFlatter:
    def test_it_says_a_stable_read_is_not_automatically_good(self):
        """A read that never changes is perfectly stable and
        perfectly useless, which is the trap of a flip-rate table."""
        assert "perfectly stable and perfectly useless" in TUNE

    def test_it_states_that_the_walk_is_a_worst_case(self):
        assert "WORST CASE" in TUNE

    def test_it_refuses_to_answer_the_accuracy_question(self):
        assert "cannot answer it" in TUNE
        assert "read_accuracy" in TUNE


class TestAccuracyMeasuresInMinutes:
    """Every other measurement here counts in OFI bars, which is right
    for a scalp on those bars and wrong for a five-minute candle."""

    def test_the_horizons_are_minutes(self):
        assert "MINUTES = (1.0, 3.0, 5.0)" in ACC
        assert "m * 60.0" in ACC

    def test_it_also_scores_the_trade_itself(self):
        """A fixed horizon is not the trade. The position is opened
        on the reading and closed when THAT candle ends, which is a
        different length every time."""
        assert "def to_candle_close" in ACC
        assert "HELD TO THE CANDLE CLOSE" in ACC
        assert "candle_end" in ACC

    def test_the_candle_close_measure_has_its_own_control(self):
        assert "to_candle_close(rows, ev, flip=True)" in ACC

    def test_it_scores_one_row_per_reading_not_per_update(self):
        assert "One row per" in ACC and "not one per update" in ACC
        assert "would count a" in ACC

    def test_it_counts_flats_separately(self):
        assert "UP, FLAT AND DOWN ARE THREE OUTCOMES" in ACC
        assert '"flat"' in ACC
        assert "pct_all" in ACC and "pct_dec" in ACC

    def test_it_has_a_flipped_control(self):
        assert "flip: bool = False" in ACC
        assert "SAME READ RUN BACKWARDS" in ACC

    def test_it_names_the_sample_it_needs(self):
        assert "THAT IS NOT A SAMPLE" in ACC
        assert "n_max < 200" in ACC

    def test_it_names_costs(self):
        assert "0.4" in ACC


class TestTheyRun:
    def _run(self, path, args):
        import contextlib
        import importlib.util
        import io
        spec = importlib.util.spec_from_file_location("m", path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        buf = io.StringIO()
        argv = sys.argv
        sys.argv = [str(path)] + args
        try:
            with contextlib.redirect_stdout(buf):
                rc = m.main()
        finally:
            sys.argv = argv
        return rc, buf.getvalue()

    def test_tuning_produces_a_grid(self):
        rc, out = self._run(ROOT / "scripts" / "read_tuning.py",
                            ["--updates", "40000", "--warmups", "10",
                             "--holds", "5", "15"])
        assert rc == 0
        assert "READ STABILITY" in out
        assert "% spoken" in out
        assert out.count("\n") > 8

    def test_tuning_names_the_frozen_case(self):
        """A read that never changes because its raw sign never
        survives the hold looks identical to a rock-steady one, and
        is the opposite."""
        assert "FROZEN" in TUNE or "frozen" in TUNE
        assert "whichever side happened to be ahead" in TUNE

    def test_accuracy_runs_on_a_walk_and_says_it_found_nothing(self):
        rc, out = self._run(ROOT / "scripts" / "read_accuracy.py",
                            ["--updates", "60000"])
        assert rc in (0, 2)
        if rc == 0:
            assert "RANDOM WALK" in out
            assert "sampling noise" in out
