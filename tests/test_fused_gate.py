"""WHAT THE TICK CHART IS DOING INSIDE THE OFI CANDLE.

Chris: "its a tick chart inside a OFI candle what i want to see is
what the tick chart is doing inside the OFI, like moving down
together, moving up together ofi remaining down but tick is
bouncing from bottom or top or read it like normal candle and OFI
trying to go up or down".

Asked whether it should gate anything: "Agreeing, bounce shown
only", and asked where it should live: "the fused bar should be a
visible gate all on its own that i can look at ... we can create
the strategy after i can see what it does ill make entry and exit
rules then".

SO THE MOST IMPORTANT TESTS HERE ARE THE ONES THAT PROVE IT
DECIDES NOTHING. On this engine a reading drawn like a gate ends
up treated as one -- that is how the OFI bar came to close trades
nobody meant it to.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from liqmap import fusedread as fr

SRC = pathlib.Path(__file__).resolve().parents[1] / "liqmap"


class TestItOnlySuggests:
    def test_nothing_here_is_named_like_an_order(self):
        tree = ast.parse((SRC / "fusedread.py").read_text())
        names = {n.name for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef,
                                   ast.AsyncFunctionDef))}
        for bad in ("submit", "execute", "place", "order"):
            assert bad not in names, f"fusedread.py defines {bad}()"

    def test_it_reaches_no_broker(self):
        src = (SRC / "fusedread.py").read_text().lower()
        for bad in ("ironbeam", "requests.", "httpx", "urllib",
                    "socket"):
            assert bad not in src, f"fusedread.py mentions {bad}"


class TestItDecidesNothing:
    """"Agreeing, bounce shown only" -- and there is no strategy
    behind it at all yet."""

    def test_every_reading_says_it_decides_nothing(self):
        for kw in ({}, {"ofi_dir": "down", "tick_dir": "down",
                        "open_px": 20.0, "high": 20.0, "low": 0.0,
                        "close": 1.0},
                   {"ofi_dir": "up", "tick_dir": "down",
                    "open_px": 0.0, "high": 10.0, "low": 0.0,
                    "close": 1.0}):
            assert fr.read(**kw)["decides"] is False

    def test_it_returns_no_side_to_trade(self):
        """IT HAS NO `side` AND MUST NOT GROW ONE. A reading with a
        side in it is one line away from being wired to an entry."""
        out = fr.read(ofi_dir="down", tick_dir="down", open_px=20.0,
                      high=20.0, low=0.0, close=1.0)
        assert "side" not in out, \
            "the reading offers a side -- it is a gate now"

    def test_the_route_does_not_let_it_block_or_close(self):
        src = (SRC / "web.py").read_text()
        i = src.index("def _fused_gate(")
        fn = src[i:src.index("\n    def ", i + 10)]
        for bad in ("entry_side", "exit_reason", "disagreed",
                    "open_trade", "close_trade", "window_signal"):
            assert bad not in fn, \
                f"the fused gate helper touches {bad}"

    def test_no_rule_module_imports_it(self):
        """IT IS NOT REACHABLE FROM A RULE. The panel reads it; the
        rules must not, until he writes the rules."""
        for mod in ("stackwin.py", "twotick.py", "journal.py"):
            src = (SRC / mod).read_text()
            assert "fusedread" not in src, \
                f"{mod} imports the shown-only reading"


class TestHisFourCases:
    """The states he named, in the order he named them."""

    def test_moving_down_together(self):
        out = fr.read(ofi_dir="down", tick_dir="down",
                      open_px=5020.0, high=5020.5, low=5000.0,
                      close=5001.0)
        assert out["where"] == "together"
        assert out["agree"] is True
        assert "moving down together" in out["say"]

    def test_moving_up_together(self):
        out = fr.read(ofi_dir="up", tick_dir="up", open_px=5000.0,
                      high=5020.0, low=4999.5, close=5019.0)
        assert out["where"] == "together"
        assert "moving up together" in out["say"]

    def test_the_book_stays_down_while_price_bounces_off_the_bottom(self):
        """His third case, and the one that looks identical to the
        first if you only read directions: both legs still say
        down, and price has come 60% of the way back."""
        out = fr.read(ofi_dir="down", tick_dir="down",
                      open_px=5020.0, high=5020.0, low=5000.0,
                      close=5012.0)
        assert out["where"] == "bounce"
        assert out["agree"] is True, \
            "a bounce is not a disagreement -- both legs still say down"
        assert out["ofi_dir"] == "down"
        assert "bouncing off the bottom" in out["say"]
        assert out["back"] == pytest.approx(0.6)

    def test_the_book_stays_up_while_price_fades_off_the_top(self):
        out = fr.read(ofi_dir="up", tick_dir="up", open_px=5000.0,
                      high=5020.0, low=5000.0, close=5008.0)
        assert out["where"] == "bounce"
        assert "bouncing off the top" in out["say"]

    def test_the_book_and_price_disagreeing_is_its_own_state(self):
        out = fr.read(ofi_dir="down", tick_dir="up", open_px=5000.0,
                      high=5010.0, low=4998.0, close=5009.0)
        assert out["where"] == "apart"
        assert out["agree"] is False
        assert "not reading the same bar" in out["say"]

    def test_read_it_like_a_normal_candle(self):
        """"or read it like normal candle" -- where the close sits
        between this bar's own high and low."""
        out = fr.read(ofi_dir="up", tick_dir="up", open_px=5000.0,
                      high=5020.0, low=5000.0, close=5019.0)
        assert out["pos"] == pytest.approx(0.95)
        assert out["at"] == "top"
        low = fr.read(ofi_dir="down", tick_dir="down", open_px=5020.0,
                      high=5020.0, low=5000.0, close=5001.0)
        assert low["at"] == "bottom"
        mid = fr.read(ofi_dir="up", tick_dir="up", open_px=5000.0,
                      high=5020.0, low=5000.0, close=5010.0)
        assert mid["at"] == ""

    def test_ofi_trying_to_go_up_or_down(self):
        """"and OFI trying to go up or down" -- how far the book has
        pushed toward the target that closes the bar."""
        out = fr.read(ofi_dir="down", tick_dir="down", net_ofi=-3000,
                      ofi_target=4000, prints=2000, tick_target=10000,
                      open_px=20.0, high=20.0, low=0.0, close=1.0)
        assert out["legs"]["ofi_pct"] == pytest.approx(75.0)
        assert out["legs"]["tick_pct"] == pytest.approx(20.0)
        assert out["legs"]["waiting_on"] == "both"


class TestTheMeasures:
    def test_a_bar_with_no_range_has_no_position(self):
        """THIN TAPE IS A REAL STATE. Every print on one price must
        not read as "at the bottom", which is what 0.0 would."""
        assert fr.position(5000.0, 5000.0, 5000.0) is None
        out = fr.read(ofi_dir="down", tick_dir="flat", open_px=5000.0,
                      high=5000.0, low=5000.0, close=5000.0)
        assert out["pos"] is None
        assert out["at"] == ""

    def test_the_retrace_is_the_one_the_tick_panel_already_uses(self):
        """NOT A SECOND MEASURE OF THE SAME THING. Two panels that
        measure one bar differently is how they come to disagree."""
        from liqmap import tapeintent as ti
        assert fr.depth is ti.depth

    def test_the_retrace_is_measured_against_the_books_direction(self):
        """His case is "OFI remaining down but tick is bouncing" --
        the book is the frame and price's position inside it is the
        question."""
        out = fr.read(ofi_dir="down", tick_dir="down", open_px=5020.0,
                      high=5020.0, low=5000.0, close=5010.0)
        assert out["back"] == pytest.approx(0.5)

    def test_which_direction_frames_the_retrace_is_visible(self):
        """THE TWO FRAMINGS ONLY DIFFER WHEN THE LEGS DO, which is
        why the test above cannot see the difference: with both
        legs down, measuring against either gives the same number.
        A mutation swapping the book for price survived it.

        Here the book is DOWN and price is UP. Framed on the book,
        the bar's whole downward move is gone -- 1.0. Framed on
        price, the bar is a tick off its high -- 0.1. The panel
        draws this number, so which one it is matters."""
        out = fr.read(ofi_dir="down", tick_dir="up", open_px=5000.0,
                      high=5010.0, low=4998.0, close=5009.0)
        assert out["back"] == pytest.approx(1.0), \
            "the retrace is framed on price, not on the book"
        from liqmap.tapeintent import depth
        assert depth("up", 5000.0, 5010.0, 4998.0, 5009.0) \
            == pytest.approx(0.1), "the two framings are not distinct"

    def test_at_the_extreme_nothing_is_given_back(self):
        out = fr.read(ofi_dir="down", tick_dir="down", open_px=5020.0,
                      high=5020.0, low=5000.0, close=5000.0)
        assert out["back"] == pytest.approx(0.0)
        assert out["where"] == "together"

    def test_the_bounce_threshold_is_a_label_not_a_cliff(self):
        """The raw number is carried whatever the word says, so
        nothing is hidden behind the threshold."""
        near = fr.read(ofi_dir="down", tick_dir="down", open_px=20.0,
                       high=20.0, low=0.0, close=6.0)
        assert near["where"] == "together"
        assert near["back"] == pytest.approx(0.3)
        assert near["bounce_at"] == pytest.approx(fr.BOUNCE_AT)
        over = fr.read(ofi_dir="down", tick_dir="down", open_px=20.0,
                       high=20.0, low=0.0, close=8.0)
        assert over["where"] == "bounce"
        assert over["back"] == pytest.approx(0.4)

    def test_a_flat_book_settles_nothing(self):
        out = fr.read(ofi_dir="flat", tick_dir="up", open_px=5000.0,
                      high=5010.0, low=5000.0, close=5009.0)
        assert out["where"] == ""
        assert "not committed" in out["say"]

    def test_a_flat_price_is_not_a_disagreement(self):
        """Price closed where the bar opened. That is an absence of
        a reading, not one against the book -- the mistake this
        engine has made three times."""
        out = fr.read(ofi_dir="down", tick_dir="flat", open_px=5000.0,
                      high=5010.0, low=4990.0, close=5000.0)
        assert out["where"] == ""
        assert out["where"] != "apart"

    def test_junk_numbers_do_not_raise(self):
        for bad in (None, "", "abc"):
            fr.position(bad, bad, bad)
            fr.legs(bad, bad, bad, bad)
            fr.read(net_ofi=bad, ofi_target=bad, prints=bad,
                    tick_target=bad)

    def test_a_zero_target_does_not_divide(self):
        out = fr.legs(100, 0, 100, 0)
        assert out["ofi_pct"] == 0.0
        assert out["tick_pct"] == 0.0


class TestItIsWiredThrough:
    def _web(self):
        return (SRC / "web.py").read_text()

    def _step(self):
        from tests.test_journal import py_func
        return py_func("_journal_step_un")

    def test_only_the_four_built_pairs_are_offered(self):
        """A BUILDER IS A RUNNING COUNT AND A RUNNING ACCUMULATOR.
        One created on demand starts empty and says nothing until
        it fills, so a pair with no builder is not a slower reading
        -- it is no reading at all."""
        import re
        from liqmap.fusedbar import PAIRS
        src = self._web()
        i = src.index('id="jnlFused"')
        box = src[i:src.index("</select>", i)]
        shown = sorted(tuple(int(x) for x in m.split("x"))
                       for m in re.findall(r'value="(\d+x\d+)"', box))
        assert shown == sorted(tuple(p) for p in PAIRS)

    def test_the_settings_carry_the_pair(self):
        from liqmap.journal import Settings
        s = Settings()
        assert (s.fused_ticks, s.fused_ofi) == (10000, 4000)
        d = s.to_dict()
        assert d["fused_ticks"] == 10000
        assert d["fused_ofi"] == 4000

    def test_it_survives_being_saved(self, tmp_path):
        from liqmap.journal import Journal, Settings
        j = Journal(tmp_path / "j.db")
        j.save_settings(Settings(coin="ES", fused_ticks=2500,
                                 fused_ofi=1000))
        g = j.settings("ES")
        assert (g.fused_ticks, g.fused_ofi) == (2500, 1000)

    def test_a_pair_with_no_builder_is_snapped(self, tmp_path):
        from liqmap.fusedbar import PAIRS
        from liqmap.journal import Journal, Settings
        j = Journal(tmp_path / "j.db")
        j.save_settings(Settings(coin="ES", fused_ticks=9999,
                                 fused_ofi=3999))
        g = j.settings("ES")
        assert (g.fused_ticks, g.fused_ofi) in PAIRS

    def test_the_other_settings_still_round_trip(self):
        """A NEW COLUMN IN THE INSERT is where the old ones get
        knocked out of line with their placeholders."""
        import pathlib as _pl
        import tempfile
        from liqmap.journal import Journal, Settings
        with tempfile.TemporaryDirectory() as d:
            j = Journal(_pl.Path(d) / "j.db")
            j.save_settings(Settings(
                coin="ES", auto=True, tp_on=True, sl_on=True,
                tp_ticks=12.0, sl_ticks=6.0, window_s=300.0,
                tick_bar=2000, tick_bar_3=500, tick3_exit=True,
                fused_ticks=1000, fused_ofi=500, strategy="ticks"))
            g = j.settings("ES")
            assert (g.auto, g.tp_on, g.sl_on) == (True, True, True)
            assert (g.tp_ticks, g.sl_ticks) == (12.0, 6.0)
            assert g.window_s == 300.0
            assert (g.tick_bar, g.tick_bar_3) == (2000, 500)
            assert g.tick3_exit is True
            assert (g.fused_ticks, g.fused_ofi) == (1000, 500)
            assert g.strategy == "ticks"

    def test_the_route_takes_the_pair(self):
        src = self._web()
        assert "fused_ticks: int | None = None," in src
        assert "fused_ofi: int | None = None," in src
        i = src.index("fused_ofi: int | None = None,")
        blk = src[i:i + 2500]
        assert "fused_ticks=(int(getattr(cur" in blk, \
            "the route takes the parameter and drops it"
        assert "fused_ofi=(int(getattr(cur" in blk

    def test_the_page_sends_both_and_reflects_them_back(self):
        src = self._web()
        assert "fused_ticks: $('jnlFused')" in src
        assert "fused_ofi: $('jnlFused')" in src
        assert "S.fused_ticks" in src and "S.fused_ofi" in src, \
            "the selector never shows the saved pair"

    def test_the_step_reads_the_setting(self):
        blk = self._step()
        assert "_fused_gate(" in blk, "the route never builds it"
        assert 'getattr(s, "fused_ticks", 0)' in blk
        assert 'getattr(s, "fused_ofi", 0)' in blk
        assert '"fusedgate"' in blk, "it never reaches the payload"

    def test_it_is_drawn_on_every_strategy(self):
        """NOT TIED TO ONE. The whole point is watching it while
        trading something else -- "a visible gate all on its own
        that i can look at the same way i can see the tick bar
        strategy"."""
        blk = self._step()
        i = blk.index('"fusedgate"')
        seg = blk[i:i + 400]
        assert "strat ==" not in seg, \
            "the fused gate is hidden behind a strategy"
        assert "else None" not in seg

    def test_the_panel_is_drawn_and_shown(self):
        src = self._web()
        assert 'id="jnlFusedGate"' in src, "the box does not exist"
        assert "function paintFusedGate(" in src
        assert "paintFusedGate(J.fusedgate)" in src, \
            "the painter is written and never called"
        i = src.index("function paintFusedGate(")
        fn = src[i:src.index("\nfunction ", i + 10)]
        assert "SHOWN ONLY" in fn, \
            "the panel does not say it decides nothing"
        assert "MOVING" in fn and "BOUNCING OFF THE" in fn
        assert "FG.pos" in fn, "the candle read is never drawn"
        assert "FG.legs" in fn or "FG.legs || {}" in fn


from tests.test_cme_panels import _cme, client  # noqa: E402,F401

AUTH = {"Authorization": "Bearer t0ken"}


class TestItActuallySteps:
    """EXECUTION, NOT SOURCE TEXT. This route has produced three
    UnboundLocalErrors, each swallowed into `journal.error` where
    nothing was looking."""

    @pytest.mark.parametrize("pair", [
        (1000, 500), (2500, 1000), (5000, 2000), (10000, 4000),
    ])
    def test_the_step_runs_on_every_pair(self, client, monkeypatch,
                                         pair):
        from liqmap import web
        _cme(monkeypatch)
        rt = web.runtime()
        s = rt.journal.settings("ES")
        s.fused_ticks, s.fused_ofi = pair
        rt.journal.save_settings(s)
        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH).json()
        jr = d.get("journal") or {}
        assert "error" not in jr, (
            f"the step raised on {pair}: {jr.get('error')}")

    @pytest.mark.parametrize("strat", ["ofi", "stack", "ticks",
                                       "fused", "timed"])
    def test_the_panel_is_present_whatever_the_strategy(
            self, client, monkeypatch, strat):
        from liqmap import web
        _cme(monkeypatch)
        rt = web.runtime()
        s = rt.journal.settings("ES")
        s.strategy = strat
        rt.journal.save_settings(s)
        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH).json()
        jr = d.get("journal") or {}
        assert "error" not in jr, jr.get("error")
        fg = jr.get("fusedgate")
        assert fg is not None, (
            f"the fused gate is missing on the {strat} strategy")
        assert fg.get("decides") is False

    def test_the_payload_names_the_pair_it_read(self, client,
                                                monkeypatch):
        from liqmap import web
        _cme(monkeypatch)
        rt = web.runtime()
        s = rt.journal.settings("ES")
        s.fused_ticks, s.fused_ofi = 2500, 1000
        rt.journal.save_settings(s)
        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH).json()
        fg = ((d.get("journal") or {}).get("fusedgate")) or {}
        assert fg.get("pair") == [2500, 1000]
        assert "2500" in str(fg.get("label") or "")

    def test_the_settings_route_takes_it(self, client):
        d = client.post("/api/journal/settings",
                        params={"coin": "ES", "fused_ticks": "5000",
                                "fused_ofi": "2000"},
                        headers=AUTH).json()
        s = d.get("settings") or d
        assert int(s.get("fused_ticks")) == 5000
        assert int(s.get("fused_ofi")) == 2000
        d = client.post("/api/journal/settings",
                        params={"coin": "ES", "tick_bar": "3000"},
                        headers=AUTH).json()
        s = d.get("settings") or d
        assert int(s.get("fused_ticks")) == 5000, \
            "a setting the call did not mention was reset"
