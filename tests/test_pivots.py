"""Swing points, and the ways marking them misleads.

In order of how badly each would mislead:

  marking a pivot in the last few bars. It is not settled -- the next
  bar may take the high out and the pivot never existed. A level that
  appears and then vanishes is worse than no level, because you
  remember it was there and not that it was provisional

  finding nothing on a plateau. Two bars sharing a high is ordinary on
  a tick grid, and a strict comparison on both sides silently drops
  some of the most obvious turns on the chart

  marking the same turn twice, which doubles the apparent structure
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from liqmap.pivots import HIGH, LOW, Pivot, find, levels, pending  # noqa: E402


def hl(seq):
    """Treat one series as both high and low -- a simple zig-zag."""
    return list(seq), list(seq)


class TestItFindsTheObviousTurns:

    def test_a_single_peak_is_a_pivot_high(self):
        h, l = hl([1, 2, 3, 9, 3, 2, 1])
        ps = [p for p in find(h, l, 3, 3) if p.kind == HIGH]
        assert len(ps) == 1
        assert ps[0].index == 3 and ps[0].price == 9

    def test_a_single_trough_is_a_pivot_low(self):
        h, l = hl([9, 8, 7, 1, 7, 8, 9])
        ps = [p for p in find(h, l, 3, 3) if p.kind == LOW]
        assert len(ps) == 1
        assert ps[0].index == 3 and ps[0].price == 1

    def test_highs_and_lows_are_found_independently(self):
        highs = [1, 2, 9, 2, 1, 2, 3]
        lows = [1, 0, 1, 1, 0, 1, 2]
        ps = find(highs, lows, 2, 2)
        kinds = {p.kind for p in ps}
        assert HIGH in kinds

    def test_a_long_zigzag_alternates(self):
        h, l = hl([1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1])
        ps = find(h, l, 1, 1)
        kinds = [p.kind for p in ps]
        assert len(kinds) > 4
        # Peaks and troughs must alternate on a clean zigzag.
        for a, b in zip(kinds, kinds[1:]):
            assert a != b, kinds


class TestTheRightEdgeIsNotJudged:
    """The failure that puts a level on the chart and then removes it."""

    def test_no_pivot_is_reported_inside_the_unsettled_window(self):
        h, l = hl([1, 2, 3, 9, 3, 2])        # peak at 3, only 2 bars after
        ps = find(h, l, 3, 3)
        assert ps == [], "the peak is not settled with 3 bars required"

    def test_it_becomes_a_pivot_once_enough_bars_print(self):
        h, l = hl([1, 2, 3, 9, 3, 2, 1])
        assert any(p.index == 3 for p in find(h, l, 3, 3))

    def test_a_later_bar_can_prevent_a_pivot(self):
        """Exactly why the edge is withheld: this high is taken out."""
        h, l = hl([1, 2, 3, 9, 3, 2, 10])
        assert not any(p.index == 3 and p.kind == HIGH
                       for p in find(h, l, 3, 3))

    def test_pending_reports_the_unsettled_count(self):
        assert pending(100, 3) == 3
        assert pending(2, 3) == 2, "a short series cannot owe more than it has"
        assert pending(0, 3) == 0

    def test_a_series_shorter_than_the_window_yields_nothing(self):
        h, l = hl([1, 2, 3])
        assert find(h, l, 3, 3) == []


class TestPlateaus:

    def test_a_flat_top_still_produces_one_pivot(self):
        """Strict on both sides finds nothing here, which drops some of
        the clearest turns on a tick grid."""
        h, l = hl([1, 2, 9, 9, 2, 1, 0])
        ps = [p for p in find(h, l, 2, 2) if p.kind == HIGH]
        assert len(ps) == 1, f"expected one, got {[p.index for p in ps]}"

    def test_the_plateau_pivot_sits_at_its_right_hand_end(self):
        h, l = hl([1, 2, 9, 9, 2, 1, 0])
        ps = [p for p in find(h, l, 2, 2) if p.kind == HIGH]
        assert ps[0].index == 3

    def test_a_flat_bottom_mirrors(self):
        h, l = hl([9, 8, 1, 1, 8, 9, 10])
        ps = [p for p in find(h, l, 2, 2) if p.kind == LOW]
        assert len(ps) == 1 and ps[0].index == 3

    def test_a_wide_plateau_is_still_one_pivot(self):
        h, l = hl([1, 2, 9, 9, 9, 9, 2, 1, 0])
        ps = [p for p in find(h, l, 2, 2) if p.kind == HIGH]
        assert len(ps) == 1

    def test_a_completely_flat_series_has_no_turns(self):
        h, l = hl([5] * 20)
        assert find(h, l, 3, 3) == []


class TestStrength:

    def test_a_wider_window_finds_fewer_pivots(self):
        import random
        rng = random.Random(4)
        seq = [100.0]
        for _ in range(400):
            seq.append(seq[-1] + rng.choice((-1, 0, 1)))
        h, l = hl(seq)
        counts = [len(find(h, l, k, k)) for k in (1, 3, 6, 12)]
        assert counts == sorted(counts, reverse=True), counts

    def test_degenerate_windows_are_refused(self):
        h, l = hl([1, 2, 3, 4, 5])
        for kw in ({"left": 0}, {"right": 0}, {"left": -1}):
            with pytest.raises(ValueError):
                find(h, l, **kw)

    def test_mismatched_series_are_refused(self):
        with pytest.raises(ValueError):
            find([1, 2, 3], [1, 2], 1, 1)
        with pytest.raises(ValueError):
            find([1, 2, 3], [1, 2, 3], 1, 1, ts=[0.0])


class TestLevels:

    def test_it_returns_the_most_recent_of_each_kind(self):
        h, l = hl([1, 5, 1, 6, 1, 7, 1, 8, 1, 9, 1, 2, 3])
        ps = find(h, l, 1, 1)
        lv = levels(ps, each=2)
        assert len([p for p in lv if p.kind == HIGH]) <= 2
        assert len([p for p in lv if p.kind == LOW]) <= 2
        highs = [p for p in ps if p.kind == HIGH]
        assert lv[-1].index >= highs[-2].index

    def test_it_is_ordered_by_position(self):
        h, l = hl([1, 5, 1, 6, 1, 7, 1, 8, 1, 9, 1, 2, 3])
        lv = levels(find(h, l, 1, 1), each=2)
        idx = [p.index for p in lv]
        assert idx == sorted(idx)

    def test_an_empty_list_is_fine(self):
        assert levels([], each=2) == []


class TestHousekeeping:

    def test_timestamps_are_carried_when_given(self):
        h, l = hl([1, 2, 3, 9, 3, 2, 1])
        ts = [float(i) * 10 for i in range(7)]
        ps = find(h, l, 3, 3, ts=ts)
        assert ps[0].ts == pytest.approx(30.0)

    def test_results_are_ordered_by_index(self):
        import random
        rng = random.Random(9)
        seq = [100.0]
        for _ in range(200):
            seq.append(seq[-1] + rng.choice((-1, 0, 1)))
        h, l = hl(seq)
        idx = [p.index for p in find(h, l, 2, 2)]
        assert idx == sorted(idx)

    def test_depth_is_the_shorter_side(self):
        assert Pivot(left=3, right=7).depth == 3

    def test_it_serialises(self):
        import json
        h, l = hl([1, 2, 3, 9, 3, 2, 1])
        json.dumps([p.as_dict() for p in find(h, l, 3, 3)])

    def test_no_order_code(self):
        src = (ROOT / "liqmap" / "pivots.py").read_text()
        for w in ("def submit(", "def execute(", "def place(", "def order("):
            assert w not in src, w

    def test_it_does_not_claim_to_predict(self):
        src = (ROOT / "liqmap" / "pivots.py").read_text().lower()
        assert "predict" in src and "nothing here predicts" in src
        for w in ("support", "resistance", "will hold", "target"):
            assert w not in src, w
