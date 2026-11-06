"""The 5-minute leg is who crossed the spread, not where price ended.

Chris, after four price-based definitions all lost while taking the
same 1,194 trades:

    "all i want is for the direction who is crossing the spread to
    recompute every 5 minutes thats all"

Close-against-open says which way price ENDED UP. Signed aggressive
volume says who put it there. These tests hold the rule to the
second one, and hold the 1-minute leg and the reversal exit to the
first, which is where they have to stay: "crosses its open" and
"turned" are only the same event while that leg reads price.
"""

from __future__ import annotations

import pytest

from liqmap import timed as tm
from liqmap.kinetic import Kinetic

UP = {"open": 100.0, "close": 101.0}
DOWN = {"open": 100.0, "close": 99.0}
FLAT = {"open": 100.0, "close": 100.0}


class TestTheOverrideDecidesTheFiveMinute:
    def test_the_flow_beats_the_candle(self):
        """A candle closing UP while sellers are the ones crossing.

        This is the whole point: the two measures disagree, and the
        rule must take the flow.
        """
        st = tm.state(DOWN, UP, m5_dir="down")
        assert st["m5"]["dir"] == "down", "the candle won"
        assert st["m5"]["source"] == "flow"
        assert st["side"] == "down", "both legs are down"

    def test_it_blocks_an_entry_the_candle_would_have_taken(self):
        st = tm.state(UP, UP, m5_dir="down")
        assert st["side"] == "", "the legs disagree and it entered"

    @pytest.mark.parametrize("d", ["up", "down", "flat"])
    def test_every_direction_is_honoured(self, d):
        assert tm.state(UP, DOWN, m5_dir=d)["m5"]["dir"] == d

    def test_flat_flow_blocks_entry(self):
        """Right after a boundary the cleared window is empty, so
        the push is zero and the direction is flat. Flat must not
        open a trade -- that empty window is exactly where the price
        version was entering on noise."""
        st = tm.state(UP, UP, m5_dir="flat")
        assert st["side"] == ""


class TestTheFallbackIsTheOldReading:
    @pytest.mark.parametrize("bad", ["", None, "sideways", "UP", 7])
    def test_an_unusable_override_falls_back_to_the_candle(self, bad):
        st = tm.state(UP, UP, m5_dir=bad)
        assert st["m5"]["dir"] == "up"
        assert st["m5"]["source"] == "price", \
            "it claimed a flow reading it did not have"

    def test_the_default_is_the_candle(self):
        assert tm.state(UP, DOWN)["m5"]["source"] == "price"

    def test_the_panel_can_always_tell_which_one_ran(self):
        for kw, want in (({}, "price"), ({"m5_dir": "up"}, "flow")):
            assert tm.state(UP, UP, **kw)["m5"]["source"] == want


class TestTheOtherLegIsUntouched:
    def test_the_one_minute_still_reads_price(self):
        assert tm.state(UP, UP, m5_dir="up")["m1"]["dir"] == "up"
        assert tm.state(DOWN, DOWN, m5_dir="down")["m1"]["dir"] == "down"

    def test_the_exit_is_still_the_one_minute_crossing_its_open(self):
        """If the reversal ever read the flow, "crosses its open"
        and "turned" would stop being the same event and the exit
        would have two meanings."""
        assert tm.reversed_against("up", "down")[0] is True
        assert tm.reversed_against("up", "up")[0] is False

    def test_the_override_cannot_reach_the_exit(self):
        import inspect
        src = inspect.getsource(tm.reversed_against)
        assert "m5" not in src and "flow" not in src


class TestTheWindowClearsAtTheBoundary:
    """`kinetic_bar` is the window the rule now reads. It is
    constructed aligned, and aligned is what "recompute every 5
    minutes" means."""

    def test_it_clears_at_the_five_minute_boundary(self):
        k = Kinetic(period_s=300.0, aligned=True)
        for i in range(10):
            k.on_trade(100.0 + i, 7000.0 + i * 0.25, 10.0, "B")
        assert k.direction == "up"
        assert k.push > 0
        k.on_trade(301.0, 7005.0, 1.0, "A")      # the next candle
        assert k.push == -1.0, "the previous candle's flow carried over"
        assert k.direction == "down"

    def test_an_empty_window_reads_flat_not_a_side(self):
        k = Kinetic(period_s=300.0, aligned=True)
        assert k.direction == "flat"
        assert k.push == 0.0

    def test_unattributed_prints_do_not_vote(self):
        k = Kinetic(period_s=300.0, aligned=True)
        k.on_trade(10.0, 7000.0, 50.0, "N")
        assert k.direction == "flat", "a side-less print picked a side"

    def test_the_sign_is_buyers_minus_sellers(self):
        k = Kinetic(period_s=300.0, aligned=True)
        k.on_trade(10.0, 7000.0, 30.0, "B")
        k.on_trade(11.0, 7000.0, 10.0, "A")
        assert k.push == 20.0
        assert k.direction == "up"


class TestItStillDoesNotTrade:
    def test_nothing_here_submits_an_order(self):
        import ast
        import pathlib
        for name in ("timed.py", "kinetic.py"):
            src = pathlib.Path(__file__).resolve().parents[1] / "liqmap" / name
            tree = ast.parse(src.read_text())
            bad = [n.name for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name in ("submit", "execute", "place", "order")]
            assert not bad, f"{name} grew {bad}"


class TestTheLIVERuleReadsTheFlow:
    """PINNING THE WIRING, NOT JUST THE FUNCTION.

    `timed.state` honouring an override proves nothing about the
    running service: deleting `m5_dir=` from the route left every
    other test in this file green while the live rule silently went
    back to reading close-against-open. A rule that quietly reverts
    is worse than one that never changed, because the panel would
    still have said "flow".
    """

    @pytest.fixture
    def client(self, monkeypatch, tmp_path):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient
        import liqmap.web as web

        monkeypatch.setenv("LIQMAP_DB", str(tmp_path / "flow.db"))
        monkeypatch.setenv("LIQMAP_TOKEN", "flow-token")
        monkeypatch.setenv("LIQMAP_AUTO", "false")
        monkeypatch.delenv("LIQMAP_WALLETS", raising=False)
        web.RT = None
        with TestClient(web.create_app()) as c:
            c.headers.update({"Authorization": "Bearer flow-token"})
            yield c, web
        web.RT = None

    @staticmethod
    def _arm(web, side: str, n: int = 5):
        """Put a real aligned window on the tape, with known flow."""
        tape = web.RT.ofi.tape
        k = Kinetic(period_s=300.0, aligned=True)
        for i in range(n):
            k.on_trade(10.0 + i, 7000.0, 25.0, side)
        tape.kinetic_bar = k
        return k

    def test_the_route_takes_the_direction_from_the_tape(self, client):
        c, web = client
        self._arm(web, "A")                   # sellers crossing
        t = c.get("/api/fusion?coin=ES").json()["timed"]
        assert t["m5"]["source"] == "flow", \
            "the live rule fell back to close-against-open"
        assert t["m5"]["dir"] == "down"
        assert t["m5"]["push"] == -125.0, "the magnitude is not auditable"

    def test_buyers_crossing_read_up(self, client):
        c, web = client
        self._arm(web, "B")
        t = c.get("/api/fusion?coin=ES").json()["timed"]
        assert t["m5"]["dir"] == "up"
        assert t["m5"]["push"] == 125.0

    def test_without_a_tape_window_it_says_price_not_flow(self, client):
        c, web = client
        web.RT.ofi.tape.kinetic_bar = None
        t = c.get("/api/fusion?coin=ES").json()["timed"]
        assert t["m5"]["source"] == "price", \
            "it claimed a flow reading with no window to read"
