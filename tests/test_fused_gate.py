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


class TestWhatItDecides:
    """IT DECIDES NOW, AND IT DID NOT BEFORE.

    This class replaced one called `TestItDecidesNothing`, which
    asserted the opposite and was right to. Chris watched the panel
    and then wrote the rules: "the auto trader will then trade the
    direction that is set ... book down or up, price down or up and
    says moving together and exit the trade on bounces never trade
    when book and price are apart or nothing settled". The old
    tests guarded a promise the screen no longer makes; they are
    replaced rather than deleted, so what is guaranteed instead is
    written down.
    """

    def test_it_says_it_decides(self):
        out = fr.read(ofi_dir="down", tick_dir="down", open_px=20.0,
                      high=20.0, low=0.0, close=1.0,
                      net_ofi=-4000, ofi_target=4000,
                      prints=10000, tick_target=10000)
        assert out["decides"] is True

    def test_only_moving_together_offers_a_side(self):
        """NEVER TRADE WHEN THEY ARE APART OR NOTHING IS SETTLED.
        Every other state returns no side at all, rather than a
        side with a flag beside it -- a side that exists is a side
        something eventually trades."""
        full = dict(net_ofi=-4000, ofi_target=4000, prints=10000,
                    tick_target=10000)
        together = fr.read(ofi_dir="down", tick_dir="down",
                           open_px=20.0, high=20.0, low=0.0,
                           close=1.0, **full)
        assert together["where"] == "together"
        assert together["side"] == "down"
        for name, kw in [
            ("bounce", dict(ofi_dir="down", tick_dir="down",
                            open_px=20.0, high=20.0, low=0.0,
                            close=12.0)),
            ("apart", dict(ofi_dir="down", tick_dir="up",
                           open_px=0.0, high=10.0, low=0.0,
                           close=9.0)),
            ("flat book", dict(ofi_dir="flat", tick_dir="up",
                               open_px=0.0, high=10.0, low=0.0,
                               close=9.0)),
        ]:
            out = fr.read(**kw, **full)
            assert out["side"] == "", f"{name} offered a side"

    def test_a_settling_bar_offers_nothing(self):
        out = fr.read(ofi_dir="down", tick_dir="down", open_px=20.0,
                      high=20.0, low=0.0, close=1.0,
                      net_ofi=-10, ofi_target=4000,
                      prints=40, tick_target=10000)
        assert out["where"] == "settling"
        assert out["side"] == ""
        assert fr.entry_side(out["where"], out["side"],
                             held_s=99.0)[0] == ""

    def test_the_tick_strategies_do_not_import_it(self):
        """ONE RULE PER STRATEGY. The tick strategies must not
        reach this reading -- a gate shared between two rules is
        how a change made for one silently moves the other."""
        for mod in ("stackwin.py", "twotick.py"):
            src = (SRC / mod).read_text()
            assert "fusedread" not in src, \
                f"{mod} imports the fused reading"


class TestHisFourCases:
    """The states he named, in the order he named them."""

    def test_moving_down_together(self):
        """It ran DOWN 20 off its open and is only 5% of that back,
        so price is still going down and the book agrees."""
        out = fr.read(ofi_dir="down", open_px=5020.0, high=5020.5,
                      low=5000.0, close=5001.0, **FULL)
        assert out["run_dir"] == "down"
        assert out["tick_dir"] == "down"
        assert out["where"] == "together"
        assert out["agree"] is True

    def test_moving_up_together(self):
        out = fr.read(ofi_dir="up", open_px=5000.0, high=5020.0,
                      low=4999.5, close=5019.0, **FULL)
        assert out["run_dir"] == "up"
        assert out["tick_dir"] == "up"
        assert out["where"] == "together"

    def test_the_book_stays_down_while_price_comes_back(self):
        """His third case. The candle ran DOWN 20 off its open and
        price is 60% of that back toward it -- so price has
        turned, and the book has not. That is a disagreement now,
        which is what it always was underneath."""
        out = fr.read(ofi_dir="down", open_px=5020.0, high=5020.0,
                      low=5000.0, close=5012.0, **FULL)
        assert out["run_dir"] == "down"
        assert out["back"] == pytest.approx(0.6)
        assert out["tick_dir"] == "up"
        assert out["where"] == "apart"
        assert out["agree"] is False

    def test_the_book_stays_up_while_price_comes_back(self):
        out = fr.read(ofi_dir="up", open_px=5000.0, high=5020.0,
                      low=5000.0, close=5008.0, **FULL)
        assert out["run_dir"] == "up"
        assert out["tick_dir"] == "down"
        assert out["where"] == "apart"

    def test_the_book_and_price_disagreeing_is_its_own_state(self):
        out = fr.read(ofi_dir="down", tick_dir="up", open_px=5000.0,
                      high=5010.0, low=4998.0, close=5009.0)
        assert out["where"] == "apart"
        assert out["agree"] is False
        # SAID THE WAY THE CHART SAYS IT. The renderer washes a bar
        # whose book and price disagree, so the panel names the
        # same thing the candle shows rather than inventing a
        # second vocabulary for it.
        assert "greys this bar" in out["say"]
        assert "ran" in out["say"]

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
        out = fr.read(ofi_dir="down", open_px=5000.0, high=5000.0,
                      low=5000.0, close=5000.0, **FULL)
        assert out["pos"] is None
        assert out["at"] == ""

    def test_the_denominator_is_the_run_off_the_open(self):
        """"is it moving to tick charts opening or away" -- so how
        far it got FROM THE OPEN is the denominator, and the range
        does not enter it.

        On Chris's bar the two differ: it ran 23.50 up off the
        open, and its range is 30.00. Ten points back is 43% of
        the run and 33% of the range."""
        out = fr.read(ofi_dir="down", open_px=7160.50, high=7184.00,
                      low=7154.00, close=7174.00, **FULL)
        assert out["run_pts"] == pytest.approx(23.5)
        assert out["back"] == pytest.approx(10.0 / 23.5, abs=1e-3)
        assert out["back"] != pytest.approx(10.0 / 30.0, abs=1e-3)

    def test_the_run_is_the_further_side_of_the_open(self):
        up = fr.run_from_open(100.0, 130.0, 95.0)
        assert up == ("up", pytest.approx(30.0))
        dn = fr.run_from_open(100.0, 105.0, 60.0)
        assert dn == ("down", pytest.approx(40.0))

    def test_a_tie_goes_to_the_upside(self):
        """Equal runs either way is a coin flip and has to land
        somewhere; it lands up, consistently, rather than on
        whichever float comparison happens to win."""
        assert fr.run_from_open(100.0, 110.0, 90.0)[0] == "up"

    def test_at_the_open_nothing_is_given_back(self):
        out = fr.read(ofi_dir="up", open_px=100.0, high=120.0,
                      low=100.0, close=120.0, **FULL)
        assert out["back"] == pytest.approx(0.0)
        assert out["where"] == "together"

    def test_all_the_way_back_is_one(self):
        assert fr.back_to_open(100.0, 120.0, 100.0,
                               100.0) == pytest.approx(1.0)

    def test_past_the_open_does_not_go_above_one(self):
        """A candle that ran up, came back through its open and
        kept going has given up its whole run and no more."""
        assert fr.back_to_open(100.0, 120.0, 90.0,
                               95.0) <= 1.0

    def test_a_flat_book_settles_nothing(self):
        out = fr.read(ofi_dir="flat", open_px=5000.0, high=5010.0,
                      low=5000.0, close=5009.0, **FULL)
        assert out["where"] == ""
        assert "not committed" in out["say"]

    def test_a_candle_that_has_not_run_is_not_a_disagreement(self):
        """Every print on one price. That is an absence of a
        reading, not one against the book -- the mistake this
        engine has made three times."""
        out = fr.read(ofi_dir="down", open_px=5000.0, high=5000.0,
                      low=5000.0, close=5000.0, **FULL)
        assert out["where"] == ""
        assert out["where"] != "apart"

    def test_junk_numbers_do_not_raise(self):
        for bad in (None, "", "abc"):
            fr.position(bad, bad, bad)
            fr.legs(bad, bad, bad, bad)
            fr.run_from_open(bad, bad, bad)
            fr.back_to_open(bad, bad, bad, bad)
            fr.price_dir(bad, bad, bad, bad)
            fr.read(net_ofi=bad, ofi_target=bad, prints=bad,
                    tick_target=bad, open_px=bad, high=bad, low=bad,
                    close=bad)

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
        strategy". On its own strategy it is the reading the rule
        already took; on every other one it is computed fresh. What
        must never happen is `None`."""
        blk = self._step()
        i = blk.index('"fusedgate"')
        seg = blk[i:i + 700]
        assert "else None" not in seg, \
            "the fused gate is hidden behind a strategy"
        assert "_fused_gate(" in seg

    def test_its_own_strategy_reuses_the_reading_it_traded_on(self):
        """A SECOND CALL WOULD READ THE BAR A MOMENT LATER and the
        panel could show a state the rule never saw. That is how
        the 5-minute cell and the OFI bar each came to disagree
        with the rule beside them."""
        blk = self._step()
        i = blk.index('"fusedgate"')
        seg = blk[i:i + 700]
        assert 'if strat == "together"' in seg
        assert "**fus" in seg, \
            "the panel recomputes instead of reusing the reading"

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
        assert "MOVING" in fn and "TURNED BACK TO THE OPEN" in fn
        assert "outer (book)" in fn, \
            "the panel does not name the outer candle"
        assert "price (moving)" in fn, \
            "price is not named as which way it is going"
        assert "body (shape)" in fn, \
            "the candle's shape is nowhere on the panel"
        assert "off the open" in fn, \
            "the run off the open is nowhere on the panel"
        assert "FG.pos" in fn, "the candle read is never drawn"
        assert "FG.legs" in fn or "FG.legs || {}" in fn


from tests.test_cme_panels import _cme, client  # noqa: E402,F401
# THE SERVED PAGE, not web.py's source: reading the script out of
# the Python file would carry that file's backslash escapes and
# test a string nobody is ever sent.
from tests.test_stack_strip import page_js  # noqa: E402,F401

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
        # ONE SHAPE ON EVERY STRATEGY. It reads the same bar
        # whichever rule is running, so a panel that changed shape
        # with the dropdown would be two panels wearing one name.
        for k in ("where", "entry", "label", "held"):
            assert k in fg, f"{k} missing on the {strat} strategy"
        # BUT IT ONLY DECIDES ON ITS OWN. This said True whatever
        # was selected, so with "Fused (intrabar)" running the
        # panel described the moving-together rule while
        # `live_signal` -- which has no pullback test at all --
        # was the thing actually taking trades. Chris saw an entry
        # on a bounce and the panel had told him that could not
        # happen.
        assert fg.get("decides") is (strat == "together"), (
            f"the panel claims to decide on the {strat} strategy")
        assert fg.get("strategy") == strat
        if strat != "together":
            assert fg.get("entry") == "", \
                "it offers a side under a rule that is not reading it"
            assert fg.get("strategy_label"), \
                "the panel cannot name the rule that IS running"

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


FULL = dict(net_ofi=-4000, ofi_target=4000, prints=10000,
            tick_target=10000)


class TestSettling:
    """Chris: "everytime a new tick ofi candle prints instead
    allowing it to bounce up and down or not sure of direction let
    it settle and then pick true direction"."""

    def test_a_new_bar_names_no_direction(self):
        out = fr.read(ofi_dir="down", tick_dir="down", open_px=20.0,
                      high=20.0, low=0.0, close=1.0,
                      net_ofi=-20, ofi_target=4000,
                      prints=50, tick_target=10000)
        assert out["where"] == "settling"
        assert out["side"] == ""
        assert out["settled"] is False
        assert out["settle_why"]

    def test_both_legs_have_to_settle_not_either(self):
        """A bar can be most of the way along its print count with
        an accumulator on zero -- busy and uncommitted -- and that
        is exactly the bar whose direction means nothing."""
        prints_only = fr.read(ofi_dir="down", tick_dir="down",
                              open_px=20.0, high=20.0, low=0.0,
                              close=1.0, net_ofi=-10,
                              ofi_target=4000, prints=9000,
                              tick_target=10000)
        assert prints_only["where"] == "settling"
        book_only = fr.read(ofi_dir="down", tick_dir="down",
                            open_px=20.0, high=20.0, low=0.0,
                            close=1.0, net_ofi=-3900,
                            ofi_target=4000, prints=20,
                            tick_target=10000)
        assert book_only["where"] == "settling"

    def test_ten_percent_of_both_settles_it(self):
        out = fr.read(ofi_dir="down", tick_dir="down", open_px=20.0,
                      high=20.0, low=0.0, close=1.0,
                      net_ofi=-400, ofi_target=4000,
                      prints=1000, tick_target=10000)
        assert out["settled"] is True
        assert out["where"] == "together"

    def test_it_is_the_same_warm_up_the_fused_strategy_uses(self):
        """TWO WARM-UPS THAT COULD DISAGREE would mean one panel
        calling a bar settled while the other does not."""
        from liqmap.fusedbar import MIN_FRACTION
        assert fr.MIN_SETTLE == MIN_FRACTION

    def test_once_settled_the_direction_is_held_through_a_tie(self):
        """"then pick true direction" -- and keep it. A bar that
        ticks back to flat for an instant has not changed its
        mind."""
        first = fr.read(ofi_dir="down", tick_dir="down",
                        open_px=20.0, high=20.0, low=0.0, close=1.0,
                        bar_ts=100.0, **FULL)
        assert first["where"] == "together"
        tie = fr.read(ofi_dir="flat", tick_dir="flat", open_px=20.0,
                      high=20.0, low=0.0, close=1.0, bar_ts=100.0,
                      prev=first["held"], **FULL)
        assert tie["ofi_dir"] == "down"
        assert tie["where"] == "together"

    def test_the_memory_dies_at_the_bar_boundary(self):
        """HOLDING A DIRECTION INTO A NEW BAR would defeat the
        settling it was added for -- the new bar would open already
        pointing somewhere."""
        first = fr.read(ofi_dir="down", tick_dir="down",
                        open_px=20.0, high=20.0, low=0.0, close=1.0,
                        bar_ts=100.0, **FULL)
        nxt = fr.read(ofi_dir="flat", tick_dir="flat", open_px=20.0,
                      high=20.0, low=0.0, close=1.0, bar_ts=200.0,
                      prev=first["held"], **FULL)
        # "flat" is what the builder itself reports for a zero
        # accumulator, and it is passed through honestly. What
        # must not survive is the DOWN from the previous bar.
        assert nxt["ofi_dir"] != "down", \
            "a direction was carried into a new bar"
        assert nxt["where"] != "together"
        assert nxt["side"] == ""

    def test_the_memory_names_the_bar_it_belongs_to(self):
        out = fr.read(ofi_dir="down", tick_dir="down", open_px=20.0,
                      high=20.0, low=0.0, close=1.0, bar_ts=100.0,
                      **FULL)
        assert out["held"]["bar_ts"] == 100.0
        assert set(out["held"]) >= {"ofi_dir", "tick_dir", "bar_ts"}

    def test_an_unsettled_bar_does_not_hold_anything(self):
        """The hold starts when the direction is picked, not
        before -- otherwise the first flicker of a new bar gets
        held for the rest of it."""
        out = fr.read(ofi_dir="flat", tick_dir="flat", open_px=20.0,
                      high=20.0, low=0.0, close=1.0, bar_ts=100.0,
                      prev={"ofi_dir": "down", "tick_dir": "down",
                            "bar_ts": 100.0},
                      net_ofi=-10, ofi_target=4000, prints=20,
                      tick_target=10000)
        assert out["where"] == "settling"


class TestTheTurnDial:
    """The same dial, a clearer job: it decides how much of the
    run off the open price has to give up before it counts as
    having turned. Chris asked for "a list of percentages that i
    can select from starting from 0 to 33%"."""

    def test_the_choices_run_from_zero_to_a_third(self):
        assert fr.BOUNCE_CHOICES[0] == 0.0
        assert fr.BOUNCE_CHOICES[-1] == pytest.approx(0.33)
        assert list(fr.BOUNCE_CHOICES) == sorted(fr.BOUNCE_CHOICES)
        assert len(set(fr.BOUNCE_CHOICES)) == len(fr.BOUNCE_CHOICES)

    # a candle that opened at 100 and ran UP 20 points
    RAN_UP = dict(open_px=100.0, high=120.0, low=100.0)

    @pytest.mark.parametrize("pct,close_px,turned", [
        (0.33, 112.0, True), (0.33, 119.0, False),
        (0.10, 117.0, True), (0.10, 119.5, False),
        (0.05, 118.0, True),
    ])
    def test_a_tighter_setting_calls_the_turn_sooner(self, pct,
                                                     close_px, turned):
        out = fr.read(ofi_dir="up", close=close_px, bounce_at=pct,
                      **self.RAN_UP, **FULL)
        assert (out["tick_dir"] == "down") is turned, out["say"]

    def test_zero_means_any_step_back_is_a_turn(self):
        at = fr.read(ofi_dir="up", close=120.0, bounce_at=0.0,
                     **self.RAN_UP, **FULL)
        assert at["tick_dir"] == "up", \
            "at its furthest it has not turned, at any setting"
        off = fr.read(ofi_dir="up", close=119.75, bounce_at=0.0,
                      **self.RAN_UP, **FULL)
        assert off["tick_dir"] == "down"

    def test_an_open_of_zero_is_a_bar_before_its_first_print(self):
        """`FusedBar.open` is 0.0 until a print arrives, and zero
        is not a price. It must read as "nothing yet" rather than
        as a 20-point run."""
        out = fr.read(ofi_dir="up", open_px=0.0, high=20.0, low=0.0,
                      close=12.0, **FULL)
        assert out["run_dir"] == ""
        assert out["tick_dir"] == ""

    def test_the_setting_reaches_the_reading(self):
        out = fr.read(ofi_dir="up", close=119.0, bounce_at=0.15,
                      **self.RAN_UP, **FULL)
        assert out["bounce_at"] == pytest.approx(0.15)


class TestTheRule:
    """"book down or up, price down or up and says moving together
    ... exit the trade on bounces never trade when book and price
    are apart or nothing settled"."""

    @pytest.mark.parametrize("side", ["up", "down"])
    def test_moving_together_and_held_opens(self, side):
        assert fr.entry_side("together", side, held_s=9.0) == (side, "")

    def test_the_agreement_has_to_hold_five_seconds(self):
        """"after the moving together holds for 5 seconds then
        reenter the trade"."""
        assert fr.DWELL_S == 5.0
        got, why = fr.entry_side("together", "down", held_s=1.0)
        assert got == ""
        assert "moving down together for 1s" in why
        assert fr.entry_side("together", "down", held_s=5.0)[0] == "down"

    @pytest.mark.parametrize("where,word", [
        ("apart", "apart"),
        ("settling", "has not settled"),
        ("", "nothing settled"),
    ])
    def test_nothing_else_opens_a_trade_and_each_says_why(
            self, where, word):
        """THE REFUSAL HAS TO NAME THE STATE THAT REFUSED.

        Each of these is blocked twice over -- the specific check,
        and the catch-all for anything that is not `together`. A
        mutation deleting the bounce check therefore changed no
        behaviour and survived a test that only asked whether a
        trade opened. What it changed was the sentence on the
        panel: a bounce started reading as "nothing settled",
        which is a different market and the wrong thing to do
        about it.
        """
        got, why = fr.entry_side(where, "down", held_s=99.0)
        assert got == "", f"{where} opened a trade"
        assert word in why, f"{where} refused with: {why}"

    def test_an_untracked_hold_does_not_block(self):
        assert fr.entry_side("together", "up", held_s=None) == ("up", "")

    def test_coming_apart_is_the_only_exit_in_this_module(self):
        """Chris: "the only exit is TP or when book and price
        disagree ... thats the only exit strategy". The take
        profit is the journal's; this is the other half."""
        why = fr.exit_reason("up", "apart")
        assert "come apart" in why

    @pytest.mark.parametrize("where", ["bounce", "settling",
                                       "together", ""])
    def test_nothing_else_closes_a_trade(self, where):
        """WHAT WAS REMOVED, AND DELIBERATELY.

        A bounce closed trades for one revision: it was cutting
        winners the take profit was sized to collect, and the
        pullback is a reading he wants to SEE rather than act on.
        The bar closing closed them too -- a bar ending is
        bookkeeping, not a change in the market -- and with it
        gone, settling must not close them either, or the
        bar-close exit comes straight back wearing another name.
        """
        assert fr.exit_reason("up", where) == "", \
            f"{where} still closes a trade"
        assert fr.exit_reason("down", where) == ""

    def test_the_exit_reads_no_bar_and_no_clock(self):
        """A POSITION RIDES ACROSS BAR BOUNDARIES NOW. Nothing
        about which candle it opened in reaches this."""
        import inspect
        sig = inspect.signature(fr.exit_reason)
        assert list(sig.parameters) == [
            "trade_side", "where", "tick_dir", "tf_dir", "tf_on",
            "tf_name", "tf2_dir", "tf2_on", "tf2_name"]
        for bad in ("bar_ts", "open_bar_ts", "held_s", "now_ts",
                    "period_s"):
            assert bad not in sig.parameters, \
                f"exit_reason takes {bad}"

    def test_an_unknown_side_is_never_closed(self):
        assert fr.exit_reason("", "apart") == ""
        assert fr.exit_reason("flat", "apart") == ""

    def test_state_puts_the_reading_and_the_verdict_in_one_place(self):
        out = fr.state(ofi_dir="down", tick_dir="down", open_px=20.0,
                       high=20.0, low=0.0, close=1.0, held_s=9.0,
                       bar_ts=100.0, **FULL)
        assert out["where"] == "together"
        assert out["entry"] == "down"
        assert out["ready"] is True
        assert out["bar_ts"] == 100.0
        assert out["min_hold_s"] == 5.0

    def test_state_counts_the_hold_down(self):
        out = fr.state(ofi_dir="down", tick_dir="down", open_px=20.0,
                       high=20.0, low=0.0, close=1.0, held_s=2.0,
                       bar_ts=100.0, **FULL)
        assert out["entry"] == ""
        assert out["ready"] is False
        assert out["held_s"] == 2.0


class TestTheStrategyIsWiredThrough:
    def _web(self):
        return (SRC / "web.py").read_text()

    def _step(self):
        from tests.test_journal import py_func
        return py_func("_journal_step_un")

    def test_it_is_a_registered_strategy(self):
        from liqmap.journal import STRATEGIES
        assert "together" in STRATEGIES

    def test_it_survives_being_saved(self, tmp_path):
        from liqmap.journal import Journal, Settings
        j = Journal(tmp_path / "j.db")
        j.save_settings(Settings(coin="ES", strategy="together"))
        assert j.settings("ES").strategy == "together", \
            "an unregistered strategy is silently rewritten to ofi"

    def test_the_dropdown_offers_it(self):
        assert 'value="together"' in self._web()

    def test_the_signal_does_not_wait_on_a_window(self):
        from liqmap.journal import window_signal
        side, why = window_signal({}, "together", "",
                                  {"entry": "down", "why": ""})
        assert side == "down", why
        side, why = window_signal({}, "together", "", {})
        assert side == ""
        assert "window" not in why

    def test_the_signal_reads_the_entry_not_the_raw_side(self):
        """`side` is the direction the BAR is pointing; `entry` is
        the one the rule will act on. They differ for the whole of
        the five-second hold, and reading the wrong one would enter
        the moment they agree."""
        side, why = window_signal_safe({"side": "down", "entry": "",
                                        "why": "holding"})
        assert side == ""
        assert "holding" in why

    def test_the_step_builds_it(self):
        blk = self._step()
        assert 'strat == "together"' in blk
        assert "_fused_gate(rt_, _fpair" in blk

    def test_zero_percent_is_never_treated_as_missing(self):
        """`or 33` ANYWHERE ON THIS PATH turns his tightest setting
        into his loosest, silently, on every save."""
        blk = self._step()
        i = blk.index("_fb_pct = ")
        seg = blk[i:i + 420]
        assert "or 33" not in seg
        assert "is None" in seg
        src = self._web()
        assert "fused_bounce: $('jnlBounce') ? $('jnlBounce').value : 33," \
            in src, "the page defaults the bounce with ||"

    def test_the_dwell_is_fed_the_agreement_not_a_leg(self):
        """"after the moving together holds for 5 seconds" -- it is
        the together STATE that has to stand, so a bar flickering
        in and out of agreement never accumulates five seconds."""
        blk = self._step()
        i = blk.index("_fdw.on_tick(")
        seg = blk[i:blk.index("\n", i)]
        assert "_peek" in seg
        assert "ofi_dir" not in seg, \
            "the hold is being fed a leg rather than the agreement"

    def test_it_keeps_its_own_dwell_and_memory(self):
        blk = self._step()
        assert "_fused_dwell" in blk
        assert "_fused_held" in blk
        seg = blk[blk.index('strat == "together"'):]
        seg = seg[:seg.index('strat == "ticks"')] if \
            'strat == "ticks"' in seg else seg
        assert "_ticks_dwell" not in seg
        assert "_tick_dwell" not in seg

    def test_the_exit_is_handed_what_state_resolved(self):
        blk = self._step()
        i = blk.index("_fre.exit_reason(")
        seg = blk[i:i + 420]
        assert 'fus.get("where")' in seg
        # NO BAR STAMP REACHES THE EXIT. A position rides across
        # bar boundaries now, so handing it one would be the
        # bar-close exit coming back by the side door.
        assert "bar_ts" not in seg, \
            "the exit still reads which candle the trade opened in"

    def test_the_cap_does_not_quietly_become_two(self):
        """`decide` reads `max_per_bar or MAX_PER_BAR`, so a cap of
        0 applies the default of two. He asked for no ceiling: "as
        many times as it meets the rule agreement"."""
        blk = self._step()
        # `elif strat == "together":` CONTAINS THIS STRING, so a
        # bare index lands on the step branch instead of the cap.
        i = blk.index('\n        if strat == "together":')
        seg = blk[i:i + 1400]
        assert "cap = 1_000_000" in seg
        assert "cap = 0" not in seg
        assert 'bar_ts = float(fus.get("bar_ts")' in seg, \
            "the cap counts against the wrong candle"

    def test_the_panel_draws_the_new_states(self):
        src = self._web()
        i = src.index("function paintFusedGate(")
        fn = src[i:src.index("\nfunction ", i + 10)]
        assert "SETTLING" in fn
        assert "FG.ready === false" in fn, \
            "the hold countdown is never drawn"
        assert "FG.entry" in fn
        assert "FG.blocked_by" in fn, \
            "a refusal by the journal is invisible on this panel"
        assert "FG.decides" in fn, \
            "the heading cannot stop saying SHOWN ONLY"

    def test_the_bounce_selector_offers_his_list(self):
        import re
        src = self._web()
        i = src.index('id="jnlBounce"')
        box = src[i:src.index("</select>", i)]
        shown = sorted(int(m) for m in
                       re.findall(r'value="(\d+)"', box))
        want = sorted(int(round(100 * c)) for c in fr.BOUNCE_CHOICES)
        assert shown == want

    def test_the_bounce_setting_round_trips_including_zero(self,
                                                           tmp_path):
        from liqmap.journal import Journal, Settings
        j = Journal(tmp_path / "j.db")
        for pct in (0, 5, 10, 15, 20, 25, 33):
            j.save_settings(Settings(coin="ES", fused_bounce=pct))
            assert j.settings("ES").fused_bounce == pct, \
                f"{pct}% did not survive the save"


def window_signal_safe(fused: dict):
    from liqmap.journal import window_signal
    return window_signal({}, "together", "", fused)


class TestTheStrategyActuallySteps:
    @pytest.mark.parametrize("bounce", [0, 5, 15, 33])
    def test_the_step_runs_at_every_bounce_setting(
            self, client, monkeypatch, bounce):
        from liqmap import web
        _cme(monkeypatch)
        rt = web.runtime()
        s = rt.journal.settings("ES")
        s.strategy = "together"
        s.fused_bounce = bounce
        rt.journal.save_settings(s)
        assert rt.journal.settings("ES").fused_bounce == bounce
        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH).json()
        jr = d.get("journal") or {}
        assert "error" not in jr, (
            f"the step raised at {bounce}%: {jr.get('error')}")

    def test_the_panel_carries_the_rules_verdict(self, client,
                                                 monkeypatch):
        from liqmap import web
        _cme(monkeypatch)
        rt = web.runtime()
        s = rt.journal.settings("ES")
        s.strategy = "together"
        rt.journal.save_settings(s)
        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH).json()
        fg = ((d.get("journal") or {}).get("fusedgate")) or {}
        assert fg, "the fused panel is missing on its own strategy"
        # ONE SHAPE WHETHER OR NOT A BAR IS BUILDING. The fixture
        # has no tape engine, so this exercises the path that used
        # to return a short dict with no verdict in it at all.
        assert "entry" in fg, "the panel shows no verdict"
        assert fg.get("decides") is True
        assert "min_hold_s" in fg
        assert "where" in fg and "held" in fg

    def test_its_memory_rides_home_between_polls(self, client,
                                                 monkeypatch):
        """WITHOUT IT NOTHING HOLDS. These functions are pure --
        drop the one assignment and the direction blanks at every
        tie and the hold restarts on an absence. Nothing raises; it
        quietly stops working."""
        from liqmap import web
        _cme(monkeypatch)
        rt = web.runtime()
        s = rt.journal.settings("ES")
        s.strategy = "together"
        rt.journal.save_settings(s)
        for attr in ("_fused_held", "_fused_dwell"):
            try:
                delattr(rt, attr)
            except AttributeError:
                pass
        for _ in range(2):
            client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH)
        # NOT `assert held` -- with no tape engine in the fixture
        # the reading is empty, and an empty dict is the honest
        # answer there. What must never happen is the attribute
        # going unset, which is what dropping the assignment does.
        assert getattr(rt, "_fused_held", None) is not None, \
            "the previous reading is never kept"
        assert getattr(rt, "_fused_dwell", None) is not None

    def test_its_clock_is_not_the_tick_strategys(self, client,
                                                 monkeypatch):
        """Sharing one would carry a hold across a strategy switch,
        from a reading this rule never took."""
        from liqmap import web
        _cme(monkeypatch)
        rt = web.runtime()
        for strat in ("ticks", "together"):
            s = rt.journal.settings("ES")
            s.strategy = strat
            rt.journal.save_settings(s)
            client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH)
        a = getattr(rt, "_fused_dwell", None)
        b = getattr(rt, "_ticks_dwell", None)
        assert a is not None and b is not None
        assert a is not b, "both strategies share one clock"


class TestItReadsTheCandleTheChartDraws:
    """Chris, after comparing the panel against the chart: "its
    goal is to read the chart the way it is if you look at the
    chart there is a outside OFI candle and a inside tick candle it
    should read what that is doing".

    HIS BAR, from the screenshots: O 7160.50 H 7184.00 L 7154.00,
    a 30-point outer candle. He read it as "the OFI is clearly up
    and the tick was up and now pulling back from the top", and the
    panel answered "100% of the move given back".
    """

    BAR = dict(open_px=7160.50, high=7184.00, low=7154.00)

    def test_his_bar_reads_as_coming_back_to_the_open(self):
        """It ran 23.50 up off the open and is 14.50 of that back,
        so price has turned down while the book is still up."""
        out = fr.read(ofi_dir="up", close=7169.50, **self.BAR, **FULL)
        assert out["run_dir"] == "up"
        assert out["back"] == pytest.approx(14.5 / 23.5, abs=1e-3)
        assert out["tick_dir"] == "down"
        assert out["where"] == "apart"
        assert "back toward" in out["say"] or "ran up" in out["say"]

    def test_the_old_framing_saturated_on_exactly_this_bar(self):
        """THE FAULT, WRITTEN DOWN. Framed on the BOOK's move from
        the open, a bar with the book down 6.50 points and price up
        23.50 divided by the 6.50 and pinned at 1.0 -- it reported
        the entire move gone on a candle sitting two thirds of the
        way up its own range.

        It is framed on the candle's own run off the open now, and
        which way the book points does not enter it."""
        from liqmap.tapeintent import depth
        assert depth("down", 7160.50, 7184.00, 7154.00,
                     7174.00) == pytest.approx(1.0)
        for book in ("up", "down"):
            now = fr.read(ofi_dir=book, close=7174.00, **self.BAR,
                          **FULL)
            assert now["back"] == pytest.approx(10.0 / 23.5, abs=1e-3)
            assert now["pos"] == pytest.approx(20.0 / 30.0, abs=1e-3)

    def test_the_panel_and_the_chart_header_can_be_lined_up(self):
        """He could not tell whether the two were even the same bar
        -- they were, minutes apart -- because the panel showed
        only percentages. The candle's four numbers and the signed
        book now ride in the payload."""
        out = fr.read(ofi_dir="down", tick_dir="up", close=7174.00,
                      net_ofi=-1320, ofi_target=4000, prints=39236,
                      tick_target=10000, **self.BAR)
        assert out["open"] == 7160.50
        assert out["high"] == 7184.00
        assert out["low"] == 7154.00
        assert out["close"] == 7174.00
        assert out["net_ofi"] == -1320, \
            "the book's sign is not carried, only its size"

    def test_it_carries_the_wash_the_chart_puts_on_the_bar(self):
        """`tradeable` is what greys a candle in the renderer. The
        panel beside it must not reach a different answer about the
        same bar."""
        grey = fr.read(ofi_dir="down", tick_dir="up", close=7174.00,
                       tradeable=False, **self.BAR, **FULL)
        assert grey["grayed"] is True
        ok = fr.read(ofi_dir="up", tick_dir="up", close=7174.00,
                     tradeable=True, **self.BAR, **FULL)
        assert ok["grayed"] is False
        unknown = fr.read(ofi_dir="up", tick_dir="up", close=7174.00,
                          **self.BAR, **FULL)
        assert unknown["grayed"] is None, \
            "an unknown wash is reported as 'not greyed'"

    def test_the_panel_prints_the_candles_own_numbers(self):
        """WITHOUT THESE HE CANNOT TELL whether the panel and the
        chart are even looking at the same bar -- which is the
        whole complaint. Percentages alone made that impossible;
        a mutation hiding this block changed no reading and broke
        no other test."""
        src = (SRC / "web.py").read_text()
        i = src.index("function paintFusedGate(")
        fn = src[i:src.index("\nfunction ", i + 10)]
        for want in ("chPx(FG.open)", "chPx(FG.high)", "chPx(FG.low)",
                     "chPx(FG.close)", "FG.net_ofi"):
            assert want in fn, f"the panel never draws {want}"
        assert "FG.net_ofi > 0 ? '+' : ''" in fn, \
            "the book is drawn without its sign"
        assert "FG.grayed" in fn, \
            "the panel cannot say the chart greys this bar"
        # `box.innerHTML` appears twice -- the not-building early
        # return writes one too -- so this is the LAST one.
        j = fn.rindex("box.innerHTML")
        assert "chPx(FG.open)" in fn[:j], \
            "the block is built after the page is written"

    def test_the_route_hands_it_the_charts_own_verdict(self):
        src = (SRC / "web.py").read_text()
        i = src.index("def _fused_gate(")
        fn = src[i:src.index("\n    def ", i + 10)]
        assert "tradeable=bool(cur.tradeable)" in fn, \
            "the panel does not know what the chart greys"
        assert "cur.start_ts" in fn

    def test_the_route_reads_the_live_bar_the_chart_draws(self):
        """`rows(include_live=True)` appends `self._cur`; this reads
        `bld.current`, which is the same object. One bar, one
        builder -- the mismatch he reported was the same bar read
        minutes apart, not two sources."""
        from liqmap.fusedbar import FusedBars
        b = FusedBars(tick_target=1000, ofi_target=500)
        src = (SRC / "web.py").read_text()
        i = src.index("def _fused_gate(")
        fn = src[i:src.index("\n    def ", i + 10)]
        assert "bld.current" in fn
        assert hasattr(b, "current")
        assert "rows(" not in fn, \
            "the panel builds its own view of the series"

    @pytest.mark.parametrize("bad", [None, "", "abc", object()])
    def test_junk_never_reaches_int_in_the_payload(self, bad):
        """THE PAYLOAD CARRIES RAW FIGURES NOW, and the first
        version wrote `int(net_ofi or 0)` straight into it. That
        raises on a string, in a route whose exceptions get
        swallowed into `journal.error`."""
        out = fr.read(ofi_dir="up", tick_dir="up", open_px=bad,
                      high=bad, low=bad, close=bad, net_ofi=bad,
                      ofi_target=bad, prints=bad, tick_target=bad)
        assert out["net_ofi"] == 0
        assert out["open"] == 0.0


class TestThePanelActuallyRenders:
    """A SOURCE SCAN CANNOT SEE A DEAD BRANCH.

    A mutation that changed `((FG.open) ? ... : '')` to
    `((false) ? ... : '')` left every word of the block in the
    file, so every text assertion still passed -- and the candle's
    own numbers never reached the screen. Chris needs those
    numbers to line the panel up against the chart; that is the
    whole reason they are there. So this runs the painter in node
    and reads what it wrote.
    """

    BAR = {
        "label": "10000x4000", "pair": [10000, 4000], "building": True,
        "why": "", "where": "apart", "side": "", "entry": "",
        "ofi_dir": "down", "tick_dir": "up", "agree": False,
        "settled": True, "settle_why": "",
        "pos": 0.667, "at": "", "back": 0.333, "bounce_at": 0.33,
        "grayed": True,
        "open": 7160.50, "high": 7184.00, "low": 7154.00,
        "close": 7174.00, "net_ofi": -1320,
        "ofi_target": 4000, "prints": 39236, "tick_target": 10000,
        "legs": {"ofi_pct": 33.0, "tick_pct": 100.0,
                 "waiting_on": "the book"},
        "say": "the outer candle is down and the body is up -- the "
               "chart greys this bar, pulling back from the top by 33%",
        "held": {}, "held_s": 9.0, "min_hold_s": 5.0, "ready": True,
        "bar_ts": 100.0, "decides": True, "blocked_by": "",
    }

    def test_the_candles_four_numbers_reach_the_screen(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate", self.BAR)
        # `chPx` groups thousands, the way the chart header does.
        for want in ("7,160.50", "7,184.00", "7,154.00", "7,174.00"):
            assert want in html, f"{want} is not on the panel"

    def test_the_book_is_drawn_with_its_sign(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate", self.BAR)
        assert "-1,320" in html, html[-900:]
        assert "4,000" in html

    def test_it_says_the_chart_greys_this_bar(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate", self.BAR)
        assert "greys" in html

        # THE SENTENCE CARRIES THE WORD TOO on an apart bar, so a
        # not-greyed payload has to drop both or this proves
        # nothing about the flag.
        ok = {**self.BAR, "grayed": False, "where": "together",
              "ofi_dir": "up", "agree": True,
              "say": "moving up together -- the outer candle is up "
                     "and the body is holding the high"}
        html2 = run_js(page_js, "fusedGate", ok)
        assert "greys" not in html2

    def test_it_names_the_two_candles_the_chart_draws(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate", self.BAR)
        assert "outer (book)" in html
        assert "price (moving)" in html

    def test_the_run_off_the_open_is_drawn(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**self.BAR, "run_dir": "up", "run_pts": 23.5,
                       "back": 0.426, "moving": "back toward"})
        assert "it ran <b>UP</b> 23.50 off the open" in html
        assert "43% of that back" in html
        assert "counts as turned at" in html

    def test_a_settling_bar_says_so(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**self.BAR, "where": "settling", "settled": False,
                       "settle_why": "412 prints in -- 1,000 (10% of "
                                     "the bar) before a direction counts"})
        assert "SETTLING" in html
        assert "412 prints in" in html

    def test_it_no_longer_claims_to_be_shown_only(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate", self.BAR)
        assert "SHOWN ONLY" not in html, \
            "the panel trades now and still says it does not"

    def test_a_bar_that_is_not_building_still_renders(self, page_js):
        """The not-building path writes its own innerHTML, and it
        has to survive the same payload shape."""
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**self.BAR, "building": False,
                       "why": "between bars -- the last event closed one"})
        assert "between bars" in html


class TestAfterTheTakeProfit:
    """Chris: "if we hit take profit another trade is taken as long
    as book and price agree but never trade on settling price and
    book must agree for 5 seconds before a trade is taken"."""

    # AUTO ON, because `decide` only ever OPENS on auto -- a
    # hand-taken row is his to open. Closing on a target happens
    # either way, which is why the close tests below do not need
    # it.
    SET = dict(coin="ES", strategy="together", auto=True, tp_on=True,
               tp_ticks=10.0, sl_on=False)

    def _trade(self):
        from liqmap.journal import Trade
        return Trade(id=1, coin="ES", side="up", mode="auto",
                     opened_ts=0.0, open_px=7160.00,
                     strategy="together")

    def test_the_take_profit_closes_it(self):
        from liqmap.journal import Settings, should_close
        t, s = self._trade(), Settings(**self.SET)
        assert should_close(t, s, 7162.50, "up", window_rule=True,
                            disagreed="") == "tp"

    def test_a_bounce_does_not_take_it_off_first(self):
        """IT WAS CUTTING WINNERS THE TAKE PROFIT WAS SIZED TO
        COLLECT. Nine ticks into a ten-tick target, a pullback
        used to close the trade."""
        from liqmap.journal import Settings, should_close
        t, s = self._trade(), Settings(**self.SET)
        assert should_close(t, s, 7162.25, "", window_rule=True,
                            disagreed=fr.exit_reason("up", "bounce")) == ""

    def test_another_trade_follows_while_they_still_agree(self):
        """The agreement does not have to re-form: it has been
        standing the whole time, so `held_s` is already past five
        seconds and the next look opens again."""
        from liqmap.journal import Settings, decide
        s = Settings(**self.SET)
        act, arg = decide(None, s, 7162.50, "up", (),
                          window_rule=True, max_per_bar=1_000_000)
        assert (act, arg) == ("open", "up")

    def test_but_not_while_the_bar_is_settling(self):
        """"never trade on settling" -- and the rule offers no side
        at all there, so there is nothing for the journal to
        open."""
        from liqmap.journal import Settings, decide
        s = Settings(**self.SET)
        side, _ = fr.entry_side("settling", "", held_s=99.0)
        assert side == ""
        act, arg = decide(None, s, 7162.50, side, (),
                          window_rule=True, max_per_bar=1_000_000)
        assert act != "open", arg

    def test_and_not_before_the_agreement_has_held_five_seconds(self):
        from liqmap.journal import Settings, decide
        s = Settings(**self.SET)
        side, why = fr.entry_side("together", "up", held_s=2.0)
        assert side == ""
        assert "5s" in why
        act, _ = decide(None, s, 7162.50, side, (), window_rule=True,
                        max_per_bar=1_000_000)
        assert act != "open"

    def test_the_route_looks_again_after_closing(self):
        """A CLOSE AND A RE-ENTRY IN ONE POLL. Without the second
        look the next trade waits for the following poll, and on a
        target that is the move already gone."""
        from tests.test_journal import py_func
        blk = py_func("_journal_step_un")
        i = blk.index('if act == "close":')
        seg = blk[i:i + 1600]
        assert "j.close_trade(" in seg
        # THE ASSIGNMENT, NOT THE WORDS. A mutation that moved the
        # call into a lambda and set `act, arg = ("", "")` left
        # every word in place and re-entered nothing; what matters
        # is that the second look lands in the variables the code
        # below acts on.
        assert "act, arg = jn.decide(None," in seg, \
            "nothing re-enters until the next poll"
        k = seg.index("act, arg = jn.decide(None,")
        assert 'if act == "open":' in seg[k:], \
            "the second look is taken and then not acted on"

    def test_the_cap_does_not_stop_the_re_entry(self):
        """The journal counts trades per candle, and a ceiling of
        two would block the third target of a busy bar."""
        from tests.test_journal import py_func
        blk = py_func("_journal_step_un")
        i = blk.index('\n        if strat == "together":')
        assert "cap = 1_000_000" in blk[i:i + 1400]


class TestNothingOpensOnABounceOrASettlingBar:
    """Chris: "never take a buy when pulling back from the top or
    bottom appears or settling -- only when it says both agree and
    its been held for 5 seconds"."""

    @pytest.mark.parametrize("where", ["bounce", "settling", "apart", ""])
    def test_only_together_ever_opens(self, where):
        for side in ("up", "down", ""):
            got, _ = fr.entry_side(where, side, held_s=999.0)
            assert got == "", f"{where}/{side} opened a trade"

    def test_together_and_held_is_the_only_way_in(self):
        assert fr.entry_side("together", "up", held_s=5.0) == ("up", "")
        assert fr.entry_side("together", "up", held_s=4.9)[0] == ""

    def test_the_reading_offers_no_side_once_price_has_turned(self):
        """It ran 23.5 up off the open and is 62% of that back, so
        price is DOWN while the book is UP."""
        out = fr.read(ofi_dir="up", open_px=7160.50, high=7184.00,
                      low=7154.00, close=7169.50, **FULL)
        assert out["tick_dir"] == "down"
        assert out["where"] == "apart"
        assert out["side"] == ""


class TestTheHoldRestartsWhenTheAgreementBreaks:
    """Chris: "if it bounces but then starts moving together again
    after the moving together holds for 5 seconds then reenter the
    trade"."""

    def test_a_bounce_restarts_the_five_seconds(self):
        c = fr.AgreeHold()
        c.on_tick("up", 1000.0)
        assert c.held_s(1010.0) == pytest.approx(10.0)
        c.on_tick("", 1010.0)            # the bar bounces
        c.on_tick("", 1130.0)            # still bouncing
        c.on_tick("up", 1130.0)          # together again
        assert c.held_s(1130.0) == pytest.approx(0.0), \
            "the agreement broke and the clock kept running"
        assert c.held_s(1134.0) == pytest.approx(4.0)
        assert fr.entry_side("together", "up",
                             held_s=c.held_s(1134.0))[0] == ""
        assert fr.entry_side("together", "up",
                             held_s=c.held_s(1136.0))[0] == "up"

    def test_this_is_where_it_differs_from_dwell(self):
        """`Dwell` IGNORES A BLANK ON PURPOSE -- there a gap is the
        candle sitting on its open, and counting it as a turn is
        what kept the old ten-second hold from ever elapsing. Fed
        `Dwell`, a two-minute bounce left this clock untouched and
        the next `together` reading opened a trade on the instant
        it appeared."""
        from liqmap.stackwin import Dwell
        old, new = Dwell(), fr.AgreeHold()
        for c in (old, new):
            c.on_tick("up", 1000.0)
            c.on_tick("", 1010.0)
            c.on_tick("up", 1130.0)
        assert old.held_s(1130.0) == pytest.approx(130.0)
        assert new.held_s(1130.0) == pytest.approx(0.0)

    def test_a_flip_restarts_it_too(self):
        c = fr.AgreeHold()
        c.on_tick("up", 1000.0)
        c.on_tick("down", 1004.0)
        assert c.held_s(1004.0) == pytest.approx(0.0)
        assert c.side == "down"

    def test_an_unbroken_run_keeps_counting(self):
        c = fr.AgreeHold()
        c.on_tick("up", 1000.0)
        for t in range(1001, 1010):
            c.on_tick("up", float(t))
        assert c.held_s(1010.0) == pytest.approx(10.0)
        assert c.runs == 1

    def test_junk_never_raises_and_never_backdates(self):
        c = fr.AgreeHold()
        c.on_tick("up", 1000.0)
        for bad in (None, "", "abc"):
            c.on_tick("up", bad)
        assert c.held_s(1010.0) == pytest.approx(10.0)
        assert c.held_s(900.0) == 0.0, "a backwards clock went negative"
        assert c.held_s("abc") == 0.0

    def test_the_route_uses_it(self):
        from tests.test_journal import py_func
        blk = py_func("_journal_step_un")
        i = blk.index('elif strat == "together":')
        seg = blk[i:i + 2600]
        assert "_fr2.AgreeHold()" in seg
        assert "_fr2.Dwell()" not in seg, \
            "the agreement is on a clock that ignores it breaking"


class TestThePanelNeverClaimsAnotherRulesTrades:
    """Chris: "there is a trade that took place by passed my rule"
    -- an entry on a bounce.

    It had. With "Fused (intrabar)" selected, `live_signal` was
    the rule running: 10% of both legs and the two agreeing, with
    no pullback test anywhere in it. It enters on exactly the
    bounces this panel says it refuses -- and the panel reported
    `decides: True` whatever the dropdown said, describing the
    moving-together rule over the top of a different one.

    A panel that claims a rule nobody is running is worse than one
    that claims nothing: it gets read as evidence about the trade
    that just happened.
    """

    def test_the_other_fused_rule_really_does_enter_on_a_pullback(self):
        """NOT AN ASSUMPTION ABOUT THE OTHER RULE -- its own
        function, on a bar this panel calls a bounce."""
        from liqmap.fusedbar import FusedBar, live_signal
        bar = FusedBar(start_ts=0.0, tick_target=10000,
                       ofi_target=4000)
        bar.open, bar.high, bar.low = 7160.50, 7184.00, 7154.00
        bar.close = 7169.50
        bar.prints, bar.net_ofi = 39236, 2097
        bar.ts_last = 60.0

        side, why = live_signal(bar)
        assert side == "up", why

        ours = fr.read(ofi_dir=bar.ofi_dir, tick_dir=bar.tick_dir,
                       open_px=bar.open, high=bar.high, low=bar.low,
                       close=bar.close, net_ofi=bar.net_ofi,
                       ofi_target=4000, prints=bar.prints,
                       tick_target=10000)
        assert ours["where"] == "apart"
        assert fr.entry_side(ours["where"], ours["side"],
                             held_s=999.0)[0] == "", \
            "the two rules agree, so this explains nothing"

    def test_the_panel_says_which_rule_is_running(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**TestThePanelActuallyRenders.BAR,
                       "decides": False, "entry": "",
                       "strategy": "fused",
                       "strategy_label": "Fused -- the tick bar and "
                                         "the OFI, read separately"})
        assert "SHOWN ONLY" in html
        assert "Fused" in html, html[:400]
        assert "does not read this panel" in html
        assert "in when they move together" not in html, \
            "it still describes a rule that is not running"

    def test_and_says_the_opposite_on_its_own_strategy(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**TestThePanelActuallyRenders.BAR,
                       "decides": True, "strategy": "together",
                       "strategy_label": "Moving together"})
        assert "in when they move together" in html
        assert "SHOWN ONLY" not in html

    def test_it_offers_no_side_under_another_rule(self):
        """`entry` is blanked rather than left standing: a side in
        the payload is a side something eventually reads."""
        from tests.test_journal import py_func
        blk = py_func("_journal_step_un")
        i = blk.index('"fusedgate": (')
        seg = blk[i:i + 1400]
        assert '"decides": False' in seg
        assert '"entry": ""' in seg
        assert '"decides": True' in seg
        assert 'if strat == "together"' in seg


class TestTheTimeframeFilter:
    """Chris: "lets include a 5 minute or a 15 minute live time on
    side of price and use it as a trade filter to only make a trade
    if the 5 minute agrees with the book and price and exit a trade
    when price disagrees with the 5 minute"."""

    @pytest.mark.parametrize("side", ["up", "down"])
    def test_all_three_agreeing_opens(self, side):
        assert fr.entry_side("together", side, held_s=9.0,
                             tf_dir=side, tf_on=True) == (side, "")

    def test_the_timeframe_against_blocks(self):
        got, why = fr.entry_side("together", "up", held_s=9.0,
                                 tf_dir="down", tf_on=True)
        assert got == ""
        assert "5m candle is down" in why

    @pytest.mark.parametrize("tf", ["", "flat"])
    def test_a_filter_switched_on_with_nothing_to_say_blocks(self, tf):
        """ON MEANS ON. Letting a blank through means switching the
        filter on does nothing until the candle happens to have a
        body -- which is the gate-3 fault over again."""
        got, why = fr.entry_side("together", "up", held_s=9.0,
                                 tf_dir=tf, tf_on=True)
        assert got == ""
        assert "no direction yet" in why

    def test_switched_off_it_changes_nothing(self):
        assert fr.entry_side("together", "up", held_s=9.0,
                             tf_dir="down", tf_on=False) == ("up", "")
        assert fr.entry_side("together", "up", held_s=9.0) == ("up", "")

    def test_it_cannot_rescue_a_blocked_entry(self):
        """A FILTER, NOT A VOTE. Agreeing does not overrule the
        bounce, the settling or the five-second hold."""
        for where in ("bounce", "settling", "apart"):
            assert fr.entry_side(where, "up", held_s=9.0,
                                 tf_dir="up", tf_on=True)[0] == ""
        assert fr.entry_side("together", "up", held_s=1.0,
                             tf_dir="up", tf_on=True)[0] == ""

    def test_it_is_checked_after_the_hold_not_folded_into_it(self):
        """The five seconds is on the book-and-price agreement,
        which is what he asked for by name. Folding the timeframe
        in would mean a candle that turned green five seconds ago
        restarts a clock that has been running a minute."""
        _, why = fr.entry_side("together", "up", held_s=1.0,
                               tf_dir="down", tf_on=True)
        assert "has to hold" in why, why
        assert "5m candle" not in why

    def test_the_name_follows_the_selection(self):
        _, why = fr.entry_side("together", "up", held_s=9.0,
                               tf_dir="down", tf_on=True,
                               tf_name="15m")
        assert "15m candle is down" in why

    @pytest.mark.parametrize("trade,tick,tf", [
        ("up", "up", "down"), ("up", "down", "up"),
        ("down", "down", "up"), ("down", "up", "down"),
    ])
    def test_price_against_the_timeframe_closes_it(self, trade, tick,
                                                   tf):
        why = fr.exit_reason(trade, "together", tick, tf, True)
        assert why
        assert "against the 5m candle" in why

    def test_price_with_the_timeframe_holds_it(self):
        assert fr.exit_reason("up", "together", "up", "up", True) == ""

    @pytest.mark.parametrize("tf", ["", "flat"])
    def test_a_flat_timeframe_is_not_a_disagreement(self, tf):
        """A timeframe candle sits exactly on its open constantly;
        reading that as a reversal closes trades on nothing."""
        assert fr.exit_reason("up", "together", "up", tf, True) == ""

    def test_flat_price_is_not_a_disagreement_either(self):
        assert fr.exit_reason("up", "together", "", "down", True) == ""

    def test_switched_off_nothing_about_the_exit_changes(self):
        assert fr.exit_reason("up", "together", "up", "down",
                              False) == ""
        assert fr.exit_reason("up", "apart", "up", "up", True) == \
            "the book and price have come apart"

    def test_coming_apart_keeps_the_blame_when_both_fire(self):
        """ORDER CARRIES MEANING: the book and price parting is the
        older rule and the one he named first."""
        why = fr.exit_reason("up", "apart", "up", "down", True)
        assert "come apart" in why
        assert "5m" not in why

    def test_it_watches_price_not_the_book(self):
        """HIS ASYMMETRY, NOT AN INFERENCE: the entry wants all
        three pointing one way, the exit watches the body against
        the timeframe. The book parting from price is the line
        above."""
        import inspect
        sig = inspect.signature(fr.exit_reason)
        assert "tick_dir" in sig.parameters
        assert "ofi_dir" not in sig.parameters

    def test_the_state_carries_the_row_whether_on_or_off(self):
        """A filter invisible when switched off reads as one that
        is not there, and then a trade it would have blocked looks
        like the rule failing."""
        off = fr.state(ofi_dir="up", tick_dir="up", open_px=20.0,
                       high=20.0, low=0.0, close=20.0, held_s=9.0,
                       **FULL)
        assert off["tf"]["on"] is False
        assert off["tf"]["decides"] is False
        on = fr.state(ofi_dir="up", tick_dir="up", open_px=20.0,
                      high=20.0, low=0.0, close=20.0, held_s=9.0,
                      tf_dir="up", tf_on=True, tf_name="15m", **FULL)
        assert on["tf"] == {"dir": "up", "on": True, "name": "15m",
                            "agrees": True, "left_s": None,
                            "decides": True}
        assert on["entry"] == "up"


class TestTheTimeframeFilterIsWiredThrough:
    def _web(self):
        return (SRC / "web.py").read_text()

    def _step(self):
        from tests.test_journal import py_func
        return py_func("_journal_step_un")

    def test_the_settings_carry_it(self):
        from liqmap.journal import Settings
        s = Settings()
        assert s.fused_tf == 0
        assert s.to_dict()["fused_tf"] == 0
        assert s.to_dict()["fused_tf_label"] == "off"

    @pytest.mark.parametrize("tf,kept", [(0, 0), (60, 60),
                                         (300, 300), (900, 900),
                                         (30, 0), (123, 0),
                                         (600, 0)])
    def test_only_off_five_and_fifteen_survive_a_save(self, tmp_path,
                                                      tf, kept):
        from liqmap.journal import Journal, Settings
        j = Journal(tmp_path / "j.db")
        j.save_settings(Settings(coin="ES", fused_tf=tf))
        assert j.settings("ES").fused_tf == kept

    def test_the_selector_offers_off_one_five_and_fifteen(self):
        import re
        src = self._web()
        i = src.index('id="jnlFusedTf"')
        box = src[i:src.index("</select>", i)]
        shown = sorted(int(m) for m in re.findall(r'value="(\d+)"', box))
        assert shown == [0, 60, 300, 900]

    def test_every_offered_choice_has_a_name_and_a_candle(self):
        """A SELECTOR VALUE THE READER CANNOT NAME is a filter
        that silently does nothing: `window_label` picks the
        interval `_candle_now` is asked for, so a value it maps
        wrongly gates on a timeframe nobody chose."""
        from liqmap.journal import window_label
        assert window_label(60.0) == "1m"
        assert window_label(300.0) == "5m"
        assert window_label(900.0) == "15m"

    def test_off_is_never_read_as_missing(self):
        """`or 300` ANYWHERE turns "off" into a filter he did not
        switch on."""
        src = self._web()
        assert "fused_tf: $('jnlFusedTf') ? $('jnlFusedTf').value : 0," \
            in src
        blk = self._step()
        i = blk.index("_ftf = ")
        assert "or 0" in blk[i:i + 120]

    def test_the_step_reads_it_and_hands_it_over(self):
        blk = self._step()
        assert 'getattr(s, "fused_tf", 0)' in blk
        assert "_fused_gate(rt_, _fpair, _fbounce, None, _fprev,\n" in blk \
            or "_fprev,\n                                _ftf)" in blk

    def test_the_helper_reads_the_live_candle(self):
        src = self._web()
        i = src.index("def _fused_gate(")
        fn = src[i:src.index("\n    def ", i + 10)]
        assert "_candle_now(rt_, nm)" in fn, \
            "the filter is not reading a live timeframe candle"
        assert "_wl(float(secs))" in fn, \
            "the timeframe name is not built from the setting"
        # ONE READER FOR BOTH. A second spelling of this is a
        # second chance to gate on one candle while the panel
        # draws another.
        assert fn.count("_candle_now(rt_, nm)") == 1
        assert "_tf_read(tf_s)" in fn and "_tf_read(tf2_s)" in fn
        assert "from .journal import window_label as _wl" in fn, \
            "it reaches for `jn` through a scope it cannot see"

    def test_the_timeframe_is_named_before_anything_reads_it(self):
        """THE FOURTH NameError OF THIS SHAPE. `_nothing` is a
        closure defined above the assignment and referenced it, so
        the engine-not-running path raised a free-variable error
        into `journal.error`, where nothing was looking."""
        src = self._web()
        i = src.index("def _fused_gate(")
        fn = src[i:src.index("\n    def ", i + 10)]
        assert fn.index("_tf_dir, _tf_name, _tf_left = ") \
            < fn.index("def _nothing("), \
            "the closure is defined before the name it reads"

    def test_the_exit_is_handed_the_filter(self):
        blk = self._step()
        i = blk.index("_fre.exit_reason(")
        seg = blk[i:i + 420]
        assert 'fus.get("tick_dir")' in seg
        assert '_tf.get("dir")' in seg
        assert '_tf.get("on")' in seg

    def test_the_panel_draws_the_row(self):
        src = self._web()
        i = src.index("function paintFusedGate(")
        fn = src[i:src.index("\nfunction ", i + 10)]
        assert "FG.tf" in fn
        assert "no timeframe filter" in fn, \
            "the panel goes silent when the filter is off"

    @pytest.mark.parametrize("tf", [0, 60, 300, 900])
    def test_the_step_runs_at_every_setting(self, client, monkeypatch,
                                            tf):
        """EXECUTION, NOT SOURCE TEXT. Adding this filter shipped a
        NameError on the engine-not-running path that no source
        scan could see -- it raised into `journal.error` and every
        text assertion still passed."""
        from liqmap import web
        _cme(monkeypatch)
        rt = web.runtime()
        s = rt.journal.settings("ES")
        s.strategy = "together"
        s.fused_tf = tf
        rt.journal.save_settings(s)
        assert rt.journal.settings("ES").fused_tf == tf
        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH).json()
        jr = d.get("journal") or {}
        assert "error" not in jr, (
            f"the step raised with tf={tf}: {jr.get('error')}")
        fg = jr.get("fusedgate") or {}
        assert (fg.get("tf") or {}).get("on") is bool(tf)

    def test_the_route_takes_it(self, client):
        d = client.post("/api/journal/settings",
                        params={"coin": "ES", "fused_tf": "900"},
                        headers=AUTH).json()
        s = d.get("settings") or d
        assert int(s.get("fused_tf")) == 900
        assert s.get("fused_tf_label") == "15m"
        d = client.post("/api/journal/settings",
                        params={"coin": "ES", "fused_tf": "0"},
                        headers=AUTH).json()
        s = d.get("settings") or d
        assert int(s.get("fused_tf")) == 0, \
            "switching the filter off was read as leaving it alone"
        assert s.get("fused_tf_label") == "off"


class TestTheSecondTimeframe:
    """Chris: "lets add another timeframe on the side of that one"
    -- same rule: both must agree to open, price turning against
    either one closes the trade."""

    BOTH = dict(tf_on=True, tf_name="5m", tf2_on=True, tf2_name="15m")

    @pytest.mark.parametrize("side", ["up", "down"])
    def test_both_agreeing_opens(self, side):
        assert fr.entry_side("together", side, held_s=9.0,
                             tf_dir=side, tf2_dir=side,
                             **self.BOTH) == (side, "")

    @pytest.mark.parametrize("a,b,named,other", [
        ("down", "up", "5m", "15m"),
        ("up", "down", "15m", "5m"),
        ("down", "down", "5m", "15m"),
    ])
    def test_either_one_against_blocks_and_is_named(self, a, b, named,
                                                    other):
        """THE PANEL HAS TO BLAME THE ONE THAT REFUSED. "a
        timeframe disagrees" is not something you can act on.

        `"5m candle" in why` IS NOT ENOUGH, and a mutation that
        swapped the order proved it: "the 15m candle" contains
        "5m candle" as a substring, so the wrong name passed. The
        article pins it.
        """
        got, why = fr.entry_side("together", "up", held_s=9.0,
                                 tf_dir=a, tf2_dir=b, **self.BOTH)
        assert got == ""
        assert f"the {named} candle" in why, why
        assert f"the {other} candle" not in why, why

    @pytest.mark.parametrize("blank", ["", "flat"])
    def test_a_second_filter_on_with_nothing_to_say_blocks(self, blank):
        got, why = fr.entry_side("together", "up", held_s=9.0,
                                 tf_dir="up", tf2_dir=blank,
                                 **self.BOTH)
        assert got == ""
        assert "15m candle has no direction yet" in why

    def test_one_on_and_one_off_is_the_old_behaviour(self):
        assert fr.entry_side("together", "up", held_s=9.0,
                             tf_dir="up", tf_on=True,
                             tf2_dir="down", tf2_on=False) == ("up", "")

    def test_both_off_changes_nothing(self):
        assert fr.entry_side("together", "up", held_s=9.0,
                             tf_dir="down", tf2_dir="down") == ("up", "")

    @pytest.mark.parametrize("a,b,named,other", [
        ("down", "up", "5m", "15m"),
        ("up", "down", "15m", "5m"),
    ])
    def test_price_against_either_one_closes_it(self, a, b, named,
                                                other):
        why = fr.exit_reason("up", "together", "up", a, True, "5m",
                             b, True, "15m")
        assert why
        assert f"the {named} candle" in why, why
        assert f"the {other} candle" not in why, why

    def test_a_flat_second_timeframe_is_not_a_disagreement(self):
        assert fr.exit_reason("up", "together", "up", "up", True,
                              "5m", "flat", True, "15m") == ""

    def test_the_second_is_silent_when_it_is_off(self):
        assert fr.exit_reason("up", "together", "up", "up", True,
                              "5m", "down", False, "15m") == ""

    def test_the_first_keeps_the_blame_when_both_disagree(self):
        why = fr.exit_reason("up", "together", "up", "down", True,
                             "5m", "down", True, "15m")
        assert "5m" in why and "15m" not in why

    def test_the_state_carries_both_rows(self):
        out = fr.state(ofi_dir="up", tick_dir="up", open_px=20.0,
                       high=20.0, low=0.0, close=20.0, held_s=9.0,
                       tf_dir="up", tf2_dir="up", **self.BOTH, **FULL)
        assert out["tf"]["name"] == "5m"
        assert out["tf2"]["name"] == "15m"
        assert out["tf2"]["on"] is True
        assert out["tf2"]["decides"] is True
        assert out["entry"] == "up"


class TestTheCandleCountdown:
    """Chris: "a candle countdown that way we aren't making trades
    in the middle of nowhere", and asked what it should do: "Just
    show it"."""

    def test_it_rides_on_the_first_timeframe(self):
        out = fr.state(ofi_dir="up", tick_dir="up", open_px=20.0,
                       high=20.0, low=0.0, close=20.0, held_s=9.0,
                       tf_dir="up", tf_on=True, tf_name="5m",
                       tf_left_s=137.4, tf2_dir="up", tf2_on=True,
                       tf2_name="15m", **FULL)
        assert out["tf"]["left_s"] == pytest.approx(137.4)
        assert out["tf2"]["left_s"] is None, \
            "a second clock nobody asked for"

    def test_it_never_goes_negative(self):
        out = fr.state(tf_on=True, tf_left_s=-4.0, **FULL)
        assert out["tf"]["left_s"] == 0.0

    def test_an_unknown_countdown_is_none_not_zero(self):
        """`None` means the candle could not be read. Zero means it
        is about to close, which is a different thing to look at."""
        out = fr.state(tf_on=True, **FULL)
        assert out["tf"]["left_s"] is None

    def test_it_gates_nothing(self):
        """"Just show it". A clock that blocks entries is a rule,
        and this is not one -- the entry function cannot even see
        it."""
        import inspect
        sig = inspect.signature(fr.entry_side)
        for bad in ("left_s", "tf_left_s", "countdown"):
            assert bad not in sig.parameters
        assert "tf_left_s" not in inspect.signature(fr.exit_reason).parameters
        early = fr.state(ofi_dir="up", tick_dir="up", open_px=20.0,
                         high=20.0, low=0.0, close=20.0, held_s=9.0,
                         tf_dir="up", tf_on=True, tf_left_s=299.0,
                         **FULL)
        late = fr.state(ofi_dir="up", tick_dir="up", open_px=20.0,
                        high=20.0, low=0.0, close=20.0, held_s=9.0,
                        tf_dir="up", tf_on=True, tf_left_s=1.0,
                        **FULL)
        assert early["entry"] == late["entry"] == "up", \
            "the clock is blocking entries"

    def test_the_route_computes_it_from_the_period(self):
        src = (SRC / "web.py").read_text()
        i = src.index("def _fused_gate(")
        fn = src[i:src.index("\n    def ", i + 10)]
        assert "float(secs) - float(age)" in fn, \
            "the countdown is not the candle's own remaining time"
        assert "max(0.0," in fn

    def test_the_panel_draws_it_as_minutes_and_seconds(self):
        src = (SRC / "web.py").read_text()
        assert "function mmss(" in src
        i = src.index("function paintFusedGate(")
        fn = src[i:src.index("\nfunction ", i + 10)]
        assert "mmss(FG.tf.left_s)" in fn
        assert "shown only, nothing" in fn, \
            "the panel does not say the clock blocks nothing"

    @pytest.mark.parametrize("tf,tf2", [
        (0, 0), (300, 0), (0, 900), (60, 900), (300, 300),
    ])
    def test_the_step_runs_with_both_set(self, client, monkeypatch,
                                         tf, tf2):
        from liqmap import web
        _cme(monkeypatch)
        rt = web.runtime()
        s = rt.journal.settings("ES")
        s.strategy = "together"
        s.fused_tf, s.fused_tf2 = tf, tf2
        rt.journal.save_settings(s)
        got = rt.journal.settings("ES")
        assert (got.fused_tf, got.fused_tf2) == (tf, tf2)
        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH).json()
        jr = d.get("journal") or {}
        assert "error" not in jr, (
            f"the step raised at {tf}/{tf2}: {jr.get('error')}")
        fg = jr.get("fusedgate") or {}
        assert (fg.get("tf") or {}).get("on") is bool(tf)
        assert (fg.get("tf2") or {}).get("on") is bool(tf2)
        assert (fg.get("tf2") or {}).get("left_s") is None

    @pytest.mark.parametrize("tf2,kept", [(0, 0), (60, 60),
                                          (300, 300), (900, 900),
                                          (30, 0), (600, 0),
                                          (1800, 0)])
    def test_only_the_offered_intervals_survive_a_save(self, tmp_path,
                                                       tf2, kept):
        """THE SECOND ONE NEEDS ITS OWN GUARD. The first had one
        and the second was waved through -- a value with no live
        candle behind it gates on a timeframe nobody chose."""
        from liqmap.journal import Journal, Settings
        j = Journal(tmp_path / "j.db")
        j.save_settings(Settings(coin="ES", fused_tf2=tf2))
        assert j.settings("ES").fused_tf2 == kept

    def test_the_two_are_validated_independently(self, tmp_path):
        from liqmap.journal import Journal, Settings
        j = Journal(tmp_path / "j.db")
        j.save_settings(Settings(coin="ES", fused_tf=900,
                                 fused_tf2=42))
        g = j.settings("ES")
        assert (g.fused_tf, g.fused_tf2) == (900, 0), \
            "one bad value took the other down with it"

    def test_the_second_selector_offers_the_same_four(self):
        import re
        src = (SRC / "web.py").read_text()
        i = src.index('id="jnlFusedTf2"')
        box = src[i:src.index("</select>", i)]
        shown = sorted(int(m) for m in re.findall(r'value="(\d+)"', box))
        assert shown == [0, 60, 300, 900]

    def test_the_route_takes_the_second(self, client):
        d = client.post("/api/journal/settings",
                        params={"coin": "ES", "fused_tf": "300",
                                "fused_tf2": "900"},
                        headers=AUTH).json()
        s = d.get("settings") or d
        assert int(s.get("fused_tf")) == 300
        assert int(s.get("fused_tf2")) == 900
        assert s.get("fused_tf2_label") == "15m"
        d = client.post("/api/journal/settings",
                        params={"coin": "ES", "fused_tf2": "0"},
                        headers=AUTH).json()
        s = d.get("settings") or d
        assert int(s.get("fused_tf2")) == 0
        assert int(s.get("fused_tf")) == 300, \
            "a setting the call did not mention was reset"

    def test_both_panel_rows_render(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**TestThePanelActuallyRenders.BAR,
                       "where": "together", "ofi_dir": "up",
                       "tick_dir": "up", "agree": True, "side": "up",
                       "tf": {"dir": "up", "on": True, "name": "5m",
                              "agrees": True, "left_s": 137.0,
                              "decides": True},
                       "tf2": {"dir": "down", "on": True,
                               "name": "15m", "agrees": False,
                               "left_s": None, "decides": True}})
        assert "5m candle" in html and "15m candle" in html
        assert "2:17" in html, "the countdown is not drawn as m:ss"
        assert "closes in" in html
        assert "must agree to open" in html

    def test_the_panel_says_so_when_both_are_off(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**TestThePanelActuallyRenders.BAR,
                       "tf": {"dir": "", "on": False, "name": "",
                              "agrees": False, "left_s": None,
                              "decides": False},
                       "tf2": {"dir": "", "on": False, "name": "",
                               "agrees": False, "left_s": None,
                               "decides": False}})
        assert "no timeframe filter" in html
        assert "closes in" not in html


class TestTheMeasuredReach:
    """Chris asked for a 1,000,000 x 333,333 pair. The honest
    answer was that nobody knows how far the signed book walks
    over a longer stretch: the accumulator resets at every close
    and wanders, so it does not scale the way a print count does,
    and the only number I had was one extrapolation off one bar.
    He chose to measure first. This is the measurement.
    """

    @staticmethod
    def _bars(rows):
        from liqmap.fusedbar import FusedBar, FusedBars
        b = FusedBars(tick_target=10_000, ofi_target=4_000)
        for secs, pr, ofi, why in rows:
            bar = FusedBar(start_ts=0.0, tick_target=10_000,
                           ofi_target=4_000)
            bar.ts_last, bar.prints, bar.net_ofi = secs, pr, ofi
            bar.close_reason, bar.closed = why, True
            b.done.append(bar)
        return b

    ROWS = [(900, 96_000, 2_100, "capped"),
            (900, 88_000, -2_600, "capped"),
            (900, 101_000, 1_400, "capped"),
            (430, 46_000, 4_050, "both"),
            (610, 65_000, -4_200, "both")]

    def test_nothing_closed_yet_says_so(self):
        r = self._bars([]).reach()
        assert r["bars"] == 0
        assert "nothing measured" in r["say"]
        assert r["ofi_at_cap"] == 0

    def test_it_reports_how_far_the_book_got_at_the_cap(self):
        """THE NUMBER THAT SETTLES THE TARGET. A pair whose
        `ofi_target` sits far above this is a pair whose book leg
        never fires."""
        r = self._bars(self.ROWS).reach()
        assert r["ofi_at_cap"] == 2_100, r
        assert r["capped"] == 3
        assert "2,100 of 4,000" in r["say"]

    def test_the_book_is_measured_by_size_not_sign(self):
        """It is a signed accumulator and it goes both ways. A bar
        that walked 2,600 SHORT got just as far as one that walked
        2,600 long."""
        r = self._bars(self.ROWS).reach()
        assert r["ofi"] == 2_600, "the sign cancelled real distance"

    def test_it_reports_what_the_bars_that_met_both_legs_took(self):
        r = self._bars(self.ROWS).reach()
        assert r["met"] == 2
        assert r["seconds_when_met"] == pytest.approx(520.0)

    def test_medians_not_means(self):
        """One capped bar on a dead afternoon drags an average
        somewhere no bar ever was."""
        from liqmap.fusedbar import _median
        assert _median([1.0, 2.0, 3.0]) == 2.0
        assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5
        assert _median([]) == 0.0
        rows = self.ROWS + [(900, 5_000_000, 50, "capped")]
        r = self._bars(rows).reach()
        assert r["prints"] < 200_000, \
            "one freak bar moved the figure"

    def test_it_counts_only_the_bars_still_held(self):
        """`keep` bars, so a quiet hour ages out rather than
        weighting the answer forever."""
        from liqmap.fusedbar import FusedBars
        b = FusedBars(tick_target=10_000, ofi_target=4_000, keep=3)
        big = self._bars(self.ROWS)
        for bar in big.done:
            b.done.append(bar)
        assert b.reach()["bars"] == 3

    def test_the_route_carries_it(self):
        src = (SRC / "web.py").read_text()
        i = src.index("def _fused_gate(")
        fn = src[i:src.index("\n    def ", i + 10)]
        assert "bld.reach()" in fn
        assert '"reach": {}' in fn, \
            "the not-building path has a different shape"

    def test_the_panel_draws_it(self):
        src = (SRC / "web.py").read_text()
        i = src.index("function paintFusedGate(")
        fn = src[i:src.index("\nfunction ", i + 10)]
        assert "FG.reach" in fn
        assert "MEASURED" in fn, \
            "nothing says these are closed bars rather than the live one"

    def test_it_renders_with_closed_bars(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**TestThePanelActuallyRenders.BAR,
                       "reach": {"bars": 12, "capped": 9, "met": 3,
                                 "say": "12 closed bars: median 91,000 "
                                        "prints and 2,400 contracts in "
                                        "900s"}})
        assert "MEASURED" in html
        assert "91,000" in html

    def test_it_stays_quiet_before_any_bar_closes(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**TestThePanelActuallyRenders.BAR,
                       "reach": {"bars": 0, "say": "no bar has closed "
                                                   "yet"}})
        assert "MEASURED" not in html, \
            "it claims a measurement off no bars"

    def test_the_step_carries_it(self, client, monkeypatch):
        from liqmap import web
        _cme(monkeypatch)
        rt = web.runtime()
        s = rt.journal.settings("ES")
        s.strategy = "together"
        rt.journal.save_settings(s)
        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH).json()
        jr = d.get("journal") or {}
        assert "error" not in jr, jr.get("error")
        assert "reach" in (jr.get("fusedgate") or {})


class TestTheSwingPair:
    """Chris: "i want you to build it in the chart like all the rest
    first then as selectable to be traded with so if is 10000 x 4000
    OFI then it should be 100000 x 40000".

    ONE LEG SCALES, THE OTHER MAY NOT, and that is written down
    rather than argued about: the print leg is a count, so ten
    times the prints is about ten times the time. The book leg is a
    signed accumulator that resets at every close and wanders, so
    it does not cover ten times the ground in ten times the time.
    Whether 40,000 is reachable on this tape is what `reach()`
    answers, and if it is not, that number comes down.
    """

    PAIR = (100_000, 40_000)

    def test_it_is_offered(self):
        from liqmap.fusedbar import PAIRS
        assert self.PAIR in PAIRS

    def test_the_interval_is_not_clamped_away(self):
        """`MAX_OFI` was 200,000 and `MAX_TICKS` 200,000 -- a
        ceiling below an offered pair means the typed interval
        snaps to a size nobody offers."""
        from liqmap.fusedbar import MAX_OFI, MAX_TICKS, parse_interval
        assert parse_interval("100000x40000") == self.PAIR
        assert MAX_TICKS >= self.PAIR[0]
        assert MAX_OFI >= self.PAIR[1]

    def test_every_offered_pair_survives_its_own_parser(self):
        from liqmap.fusedbar import PAIRS, label, parse_interval
        for a, b in PAIRS:
            assert parse_interval(label(a, b)) == (a, b), \
                f"{a}x{b} is offered and then clamped"

    def test_it_runs_under_a_longer_cap(self):
        """At roughly 107 prints a second its print leg needs about
        a quarter of an hour, and much longer in thin hours. Under
        the 900-second default every bar would close short of its
        count and be drawn gray."""
        from liqmap.fusedbar import MAX_S, max_s_for
        assert max_s_for(*self.PAIR) >= 1_800.0
        assert max_s_for(10_000, 4_000) == MAX_S, \
            "the existing pairs were given a different cap"

    def test_a_capped_bar_names_its_own_cap_not_the_default(self):
        """`why_not` printed the module constant, so a bar capped
        at 3,600s would have told him it closed on a 900s cap."""
        from liqmap.fusedbar import FusedBar
        b = FusedBar(start_ts=0.0, tick_target=100_000,
                     ofi_target=40_000, max_s=3_600.0)
        b.ts_last, b.prints, b.net_ofi = 3_600.0, 51_000, 9_000
        b.open, b.close = 100.0, 101.0
        assert not b.tradeable
        assert "3600s cap" in b.why_not, b.why_not
        assert "900s" not in b.why_not

    def test_the_builder_hands_each_bar_its_cap(self):
        from liqmap.fusedbar import FusedBars
        b = FusedBars(tick_target=100_000, ofi_target=40_000,
                      max_s=3_600.0)
        bar = b._start(0.0)
        assert bar.max_s == 3_600.0

    def test_a_bar_that_fills_its_print_leg_is_still_tradeable(self):
        """THE CAP IS A BACKSTOP, NOT A VERDICT. With a cap set
        past what the print leg takes, a bar closed by the clock
        has still met its count -- so it is only gray if the book
        and price actually disagree."""
        from liqmap.fusedbar import FusedBar
        b = FusedBar(start_ts=0.0, tick_target=100_000,
                     ofi_target=40_000, max_s=3_600.0)
        b.ts_last, b.prints, b.net_ofi = 3_600.0, 140_000, 9_000
        b.open, b.close = 100.0, 101.0
        assert b.ticks_met
        assert b.tradeable, b.why_not

    def test_the_chart_offers_it(self):
        src = (SRC / "web.py").read_text()
        i = src.index('value="10000x4000">10,000 ticks')
        assert 'value="100000x40000"' in src[i:i + 500], \
            "it is not on the chart's interval list"

    def test_the_journal_selector_offers_it(self):
        import re
        src = (SRC / "web.py").read_text()
        i = src.index('id="jnlFused"')
        box = src[i:src.index("</select>", i)]
        shown = sorted(tuple(int(x) for x in m.split("x"))
                       for m in re.findall(r'value="(\d+x\d+)"', box))
        from liqmap.fusedbar import PAIRS
        assert shown == sorted(tuple(p) for p in PAIRS)

    def test_it_can_be_saved_as_the_traded_pair(self, tmp_path):
        from liqmap.journal import Journal, Settings
        j = Journal(tmp_path / "j.db")
        j.save_settings(Settings(coin="ES", fused_ticks=100_000,
                                 fused_ofi=40_000))
        g = j.settings("ES")
        assert (g.fused_ticks, g.fused_ofi) == self.PAIR

    def test_the_step_runs_on_it(self, client, monkeypatch):
        from liqmap import web
        _cme(monkeypatch)
        rt = web.runtime()
        s = rt.journal.settings("ES")
        s.strategy = "together"
        s.fused_ticks, s.fused_ofi = self.PAIR
        rt.journal.save_settings(s)
        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH).json()
        jr = d.get("journal") or {}
        assert "error" not in jr, jr.get("error")
        fg = jr.get("fusedgate") or {}
        assert fg.get("pair") == list(self.PAIR)

    def test_the_chart_route_serves_it(self, client, monkeypatch):
        from liqmap import web
        _cme(monkeypatch)
        d = client.get("/api/chart",
                       params={"coin": "ES",
                               "interval": "100000x40000", "n": "50"},
                       headers=AUTH).json()
        assert d.get("source") == "fused", d.get("error") or d
        assert d.get("interval") == "100000x40000"


class TestPriceIsTowardOrAwayFromTheOpen:
    """Chris, after a rolling look-back read nothing like the panel
    he knew: "this is a intrabar strategy it should read the tick
    ofi candle whats happening and when", and then "the extremes
    dont matter what is is doing is it moving to tick charts
    opening or away".

    So price is not close-vs-open (a POSITION, and a stale one on
    a long bar), and not a rolling window (which spans bar
    boundaries and is not intrabar at all). It is: this candle ran
    some distance off its open, and price is either still going
    that way or coming back to it.
    """

    A = dict(open_px=7160.50, high=7184.00, low=7154.00, close=7174.00)
    B = dict(open_px=7180.00, high=7184.00, low=7154.00, close=7174.00)

    def test_bar_a_ran_up_and_is_coming_back(self):
        out = fr.read(ofi_dir="down", **self.A, **FULL)
        assert out["run_dir"] == "up"
        assert out["run_pts"] == pytest.approx(23.5)
        assert out["back"] == pytest.approx(10.0 / 23.5, abs=1e-3)
        assert out["moving"] == "back toward"
        assert out["tick_dir"] == "down"
        assert out["where"] == "together"

    def test_bar_b_ran_down_and_is_rallying(self):
        """THE ONE HE WANTED TO SEE. It opened 7180, sold to 7154
        and is back at 7174 -- the body still reads DOWN, and this
        reads UP, which is what price is doing."""
        out = fr.read(ofi_dir="up", **self.B, **FULL)
        assert out["run_dir"] == "down"
        assert out["run_pts"] == pytest.approx(26.0)
        assert out["back"] == pytest.approx(20.0 / 26.0, abs=1e-3)
        assert out["tick_dir"] == "up"
        assert out["where"] == "together"
        assert out["side"] == "up"

    def test_at_its_furthest_it_is_still_going(self):
        out = fr.read(ofi_dir="up", open_px=7160.50, high=7184.0,
                      low=7154.0, close=7184.0, **FULL)
        assert out["back"] == pytest.approx(0.0)
        assert out["moving"] == "further away from"
        assert out["tick_dir"] == "up"

    def test_the_body_is_drawn_and_decides_nothing(self):
        """Bar B's body is DOWN the whole way up. It rides in the
        payload so the two can be seen disagreeing -- that
        disagreement is the whole case for the change -- and it
        changes no reading."""
        down = fr.read(ofi_dir="up", tick_dir="down", **self.B, **FULL)
        up = fr.read(ofi_dir="up", tick_dir="up", **self.B, **FULL)
        assert down["body_dir"] == "down"
        assert up["body_dir"] == "up"
        for k in ("where", "side", "tick_dir", "back", "run_dir"):
            assert down[k] == up[k], f"the body moved {k}"

    def test_the_extremes_are_only_how_far_it_got(self):
        """"the extremes dont matter" -- the high and low are read
        to answer how far from the OPEN it ran, and nothing is
        measured from them."""
        d, run = fr.run_from_open(7160.50, 7184.0, 7154.0)
        assert (d, run) == ("up", pytest.approx(23.5))
        d2, run2 = fr.run_from_open(7180.0, 7184.0, 7154.0)
        assert (d2, run2) == ("down", pytest.approx(26.0))

    def test_a_candle_sitting_on_its_open_says_nothing(self):
        out = fr.read(ofi_dir="up", open_px=7170.0, high=7170.0,
                      low=7170.0, close=7170.0, **FULL)
        assert out["run_dir"] == ""
        assert out["tick_dir"] == ""
        assert out["where"] == ""
        assert out["side"] == ""

    def test_the_band_is_why_one_tick_is_not_a_turn(self):
        near = fr.read(ofi_dir="up", open_px=100.0, high=120.0,
                       low=100.0, close=119.75, bounce_at=0.33,
                       **FULL)
        assert near["tick_dir"] == "up"
        past = fr.read(ofi_dir="up", open_px=100.0, high=120.0,
                       low=100.0, close=112.0, bounce_at=0.33,
                       **FULL)
        assert past["tick_dir"] == "down"
        assert past["where"] == "apart"

    def test_the_dial_moves_where_the_turn_is(self):
        kw = dict(ofi_dir="up", open_px=100.0, high=120.0, low=100.0,
                  close=117.0)
        assert fr.read(bounce_at=0.33, **kw, **FULL)["tick_dir"] == "up"
        assert fr.read(bounce_at=0.10, **kw, **FULL)["tick_dir"] == "down"

    def test_zero_means_any_step_back_is_a_turn(self):
        kw = dict(ofi_dir="up", open_px=100.0, high=120.0, low=100.0)
        at = fr.read(close=120.0, bounce_at=0.0, **kw, **FULL)
        off = fr.read(close=119.75, bounce_at=0.0, **kw, **FULL)
        assert at["tick_dir"] == "up", \
            "at its furthest it has not turned, at any setting"
        assert off["tick_dir"] == "down"

    def test_there_is_no_bounce_state_any_more(self):
        """It meant "they agree but price has come off its
        extreme", which is now price having TURNED. If the book
        has not turned with it that is a disagreement; if it has,
        they agree again the other way."""
        turned = fr.read(ofi_dir="up", open_px=100.0, high=120.0,
                         low=100.0, close=108.0, **FULL)
        assert turned["where"] == "apart"
        both = fr.read(ofi_dir="down", open_px=100.0, high=120.0,
                       low=100.0, close=108.0, **FULL)
        assert both["where"] == "together"
        assert both["side"] == "down"

    def test_nothing_reads_a_rolling_window_any_more(self):
        """IT SPANS BAR BOUNDARIES, so it is not intrabar. The
        whole reading is this candle's own four numbers."""
        import inspect
        for fn in (fr.read, fr.state, fr.entry_side, fr.exit_reason):
            ps = inspect.signature(fn).parameters
            for bad in ("roll_dir", "roll_window"):
                assert bad not in ps, f"{fn.__name__} takes {bad}"
        src = (SRC / "web.py").read_text()
        i = src.index("def _fused_gate(")
        body = src[i:src.index("\n    def ", i + 10)]
        assert "roll" not in body, \
            "the route still reads a rolling window"

    def test_the_route_passes_only_this_candle(self):
        src = (SRC / "web.py").read_text()
        i = src.index("def _fused_gate(")
        body = src[i:src.index("\n    def ", i + 10)]
        for want in ("open_px=cur.open", "high=cur.high",
                     "low=cur.low", "close=cur.close"):
            assert want in body, f"the reading is not given {want}"

    def test_the_step_carries_the_run(self, client, monkeypatch):
        from liqmap import web
        _cme(monkeypatch)
        rt = web.runtime()
        s = rt.journal.settings("ES")
        s.strategy = "together"
        rt.journal.save_settings(s)
        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH).json()
        jr = d.get("journal") or {}
        assert "error" not in jr, jr.get("error")
        fg = jr.get("fusedgate") or {}
        for k in ("run_dir", "run_pts", "moving", "body_dir"):
            assert k in fg, f"{k} is missing from the payload"

    def test_the_panel_draws_the_run(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**TestThePanelActuallyRenders.BAR,
                       "run_dir": "down", "run_pts": 26.0,
                       "moving": "back toward", "back": 0.769,
                       "tick_dir": "up", "body_dir": "down",
                       "ofi_dir": "up", "where": "together",
                       "agree": True, "side": "up"})
        assert "it ran <b>DOWN</b> 26.00 off the open" in html
        assert "price (moving)" in html
        assert "body (shape)" in html, \
            "the stale body is invisible exactly when it matters"
