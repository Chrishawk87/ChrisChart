"""The export must not change the measurement on its way out.

`liqmap/ofi.py` works in raw integers; this script converts to index
points for the file. The ways that can go wrong:

1.  A REFUSED EVENT BECOMING A ROW. Refusals are absent, never written
    as a zero. A zero in the `e_n` column sums into any window the
    reader builds and drags it toward the middle.
2.  THE TICK CONSTANT. spread_ticks divides by one ES tick in raw
    units. Get it wrong and every spread figure is off by a factor.
3.  A PRESCRIBED THRESHOLD. None belongs here -- a default becomes a
    recommendation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from liqmap import ofi                                      # noqa: E402
import ofi_export as X                                      # noqa: E402

PT = 1_000_000_000
TICK = PT // 4
BASE = 5800 * PT


class Lv:
    def __init__(s, bp, bs, ap, asz):
        s.bid_px, s.bid_sz, s.ask_px, s.ask_sz = bp, bs, ap, asz


class Rec:
    def __init__(s, lv, action="A", ts=1.0, iid=42, seq=1):
        s.levels, s.action = lv, action
        s.ts_event = int(ts * 1e9)
        s.instrument_id, s.sequence = iid, seq


def event():
    st = ofi.BookState()
    st.feed(Rec([Lv(BASE, 50, BASE + TICK, 40)], seq=1))
    return st.feed(Rec([Lv(BASE, 62, BASE + TICK, 40)], seq=2))


class TestRow:

    def test_every_declared_column_is_written(self):
        f = ofi.Features()
        r = X.row_from(event(), f)
        assert set(X.COLS) <= set(r)

    def test_the_tick_constant_is_one_es_tick(self):
        assert X.TICK_RAW == 250_000_000
        assert X.TICK_RAW * 4 == PT, "four ticks is one index point"

    def test_spread_is_reported_in_ticks(self):
        r = X.row_from(event(), ofi.Features())
        assert r["spread_ticks"] == pytest.approx(1.0)

    def test_prices_come_out_in_index_points(self):
        r = X.row_from(event(), ofi.Features())
        assert r["bid_px"] == pytest.approx(5800.0)
        assert r["ask_px"] == pytest.approx(5800.25)

    def test_e_n_survives_the_conversion_unrounded(self):
        r = X.row_from(event(), ofi.Features())
        assert r["e_n"] == 12
        assert isinstance(r["e_n"], int)

    def test_cancel_to_add_with_no_adds_is_blank_not_zero(self):
        f = ofi.Features(window_s=10.0)
        f.observe(1.0, "C")
        r = X.row_from(event(), f)
        assert r["cancel_to_add"] == ""


class TestNoPrescription:

    def test_the_script_prescribes_no_threshold(self):
        """Checked against the module NAMESPACE, not its text.

        A text match also hits the docstring's own "NO THRESHOLD"
        disclaimer -- the sentence promising there isn't one would
        fail the test for saying so. What matters is whether a
        threshold is actually bound to a value.
        """
        suspect = ("THRESHOLD", "Z_SCORE", "ZSCORE", "MIN_OFI",
                   "CUTOFF", "SIGNAL")
        bound = [n for n in vars(X)
                 if any(k in n.upper() for k in suspect)
                 and isinstance(vars(X)[n], (int, float))]
        assert not bound, f"ofi_export.py prescribes {bound}"

    def test_the_only_numeric_constants_are_mechanical(self):
        """Every module-level number must be a unit or a batch size,
        not a trading parameter."""
        allowed = {"TICK_RAW", "HEARTBEAT"}
        nums = {n for n, v in vars(X).items()
                if n.isupper() and isinstance(v, (int, float))
                and not isinstance(v, bool)}
        assert nums <= allowed, f"unexpected constants: {nums - allowed}"

    def test_the_script_cannot_place_an_order(self):
        src = (Path(__file__).resolve().parent.parent
               / "scripts" / "ofi_export.py").read_text()
        for bad in ("def submit", "def execute", "def place", "def order",
                    "requests.post", "urllib.request"):
            assert bad not in src
