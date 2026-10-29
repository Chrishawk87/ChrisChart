"""The hand-check, checked.

Chris verified twelve bars by hand and got twelve. This script is that
same check written down, so the thing it must never do is disagree
with the tally it sits underneath -- or quietly place an order.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from liqmap import agree as A                                 # noqa: E402
from liqmap.ofibar import CALL_RULE, OfiBarBuilder            # noqa: E402

RAW = 250_000_000
SRC = (ROOT / "scripts" / "call_log.py").read_text()


def _mod():
    spec = importlib.util.spec_from_file_location(
        "call_log", ROOT / "scripts" / "call_log.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _walk(n=120_000, seed=5, target=2000, level=300, confirm=1.0):
    import random
    rng = random.Random(seed)
    b = OfiBarBuilder(tick_size=0.25, threshold=target, max_s=900.0,
                      min_updates=2, max_bars=10 ** 7,
                      call_level=level, objective_points=1.0,
                      confirm_ticks=confirm)
    bid, ts = 7_700 * 10 ** 9, 0.0
    for _ in range(n):
        if rng.random() < 0.12:
            bid += RAW if rng.random() < 0.5 else -RAW
        ts += 0.01
        b.on_book(ts, bid, rng.randint(50, 900), bid + RAW,
                  rng.randint(50, 900))
    return b


class TestItSuggestsAndDoesNotExecute:
    def test_it_places_nothing(self):
        for bad in ("def submit(", "def execute(", "def place(",
                    "def order("):
            assert bad not in SRC, bad

    def test_it_says_so(self):
        assert "places, sizes or times an" in SRC


class TestEveryLineMatchesTheTally:
    """The listing and the summary are two readings of one set of
    observations. If they can disagree, the listing is decoration."""

    def test_the_rows_are_the_same_observations_the_table_counts(self):
        b = _walk()
        bars = b.bars[-40:]
        m = _mod()
        rows = [(i, e) for i, bar in enumerate(bars)
                for e in (bar.episodes or ())
                if e.level == 300 and e.rule == CALL_RULE]
        t = A.observation_table(bars, points=(1,), level=300,
                                rule=CALL_RULE)
        assert len(rows) == t["observations"]
        assert sum(1 for _, e in rows if e.hit == "win") == t["won"]
        assert sum(1 for _, e in rows if e.hit == "loss") == t["failed"]
        assert sum(1 for _, e in rows if e.hit == "skip") == t["skipped"]
        assert m is not None

    def test_a_skip_has_no_fill_and_no_result(self):
        b = _walk()
        eps = [e for bar in b.bars for e in (bar.episodes or ())
               if e.hit == "skip"]
        assert eps, "the walk produced no skips to check"
        for e in eps:
            assert not e.entered
            assert e.entry_px == 0.0
            assert e.result_ticks == 0.0

    def test_a_win_is_measured_from_the_fill(self):
        b = _walk()
        eps = [e for bar in b.bars for e in (bar.episodes or ())
               if e.hit == "win"]
        for e in eps:
            assert e.entered
            d = e.hit_px - e.entry_px
            fav = d if e.side == "up" else -d
            assert abs(e.result - fav) < 1e-6


class TestItRefusesAnImpossiblePair:
    def test_a_level_above_the_target_is_named_not_run(self):
        m = _mod()
        import io
        import contextlib
        buf = io.StringIO()
        argv = sys.argv
        sys.argv = ["call_log.py", "x.dbn", "--target", "250",
                    "--level", "500"]
        try:
            with contextlib.redirect_stdout(buf):
                rc = m.main()
        finally:
            sys.argv = argv
        assert rc == 2
        assert "never reach 500" in buf.getvalue()


class TestItWarnsWhenItsOwnNumberIsNoise:
    def test_a_short_median_call_is_called_out(self):
        assert "TOO SHORT TO MEAN ANYTHING" in SRC
        assert 'if t["median_updates"] <= 4:' in SRC

    def test_the_baseline_is_stated_as_zero(self):
        assert "returns 0.0%" in SRC
        assert "beat is zero rather than fifty" in SRC

    def test_costs_are_named(self):
        assert "0.4 ticks" in SRC
