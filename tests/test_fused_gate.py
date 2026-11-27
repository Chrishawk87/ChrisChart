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
            ("apart", dict(ofi_dir="down", tick_dir="up",
                           open_px=100.0, high=110.0, low=100.0,
                           close=109.0)),
            ("flat book", dict(ofi_dir="flat", tick_dir="up",
                               open_px=100.0, high=110.0, low=100.0,
                               close=109.0)),
            ("no inner", dict(ofi_dir="down", tick_dir="",
                              open_px=100.0, high=110.0, low=90.0,
                              close=100.0)),
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
        out = fr.read(ofi_dir="down", tick_dir="down",
                      open_px=5020.0, high=5020.5, low=5000.0,
                      close=5001.0, **FULL)
        assert out["tick_dir"] == "down"
        assert out["where"] == "together"
        assert out["agree"] is True

    def test_moving_up_together(self):
        out = fr.read(ofi_dir="up", tick_dir="up", open_px=5000.0,
                      high=5020.0, low=4999.5, close=5019.0, **FULL)
        assert out["tick_dir"] == "up"
        assert out["where"] == "together"

    def test_the_book_down_and_the_inner_candle_up_is_a_disagreement(
            self):
        """His third case. The inner candle closed above its open
        while the book is down -- which is exactly what the chart
        greys a bar for."""
        out = fr.read(ofi_dir="down", tick_dir="up", open_px=5000.0,
                      high=5020.0, low=4990.0, close=5012.0,
                      tradeable=False, **FULL)
        assert out["tick_dir"] == "up"
        assert out["where"] == "apart"
        assert out["agree"] is False
        assert out["grayed"] is True

    def test_the_book_up_and_the_inner_candle_down_is_too(self):
        out = fr.read(ofi_dir="up", tick_dir="down", open_px=5020.0,
                      high=5030.0, low=5000.0, close=5008.0, **FULL)
        assert out["tick_dir"] == "down"
        assert out["where"] == "apart"

    def test_the_book_and_price_disagreeing_is_its_own_state(self):
        out = fr.read(ofi_dir="down", tick_dir="up", open_px=5000.0,
                      high=5010.0, low=4998.0, close=5009.0, **FULL)
        assert out["where"] == "apart"
        assert out["agree"] is False
        # SAID THE WAY THE CHART SAYS IT. The renderer washes a bar
        # whose book and price disagree, so the panel names the
        # same thing the candle shows rather than inventing a
        # second vocabulary for it.
        assert "they disagree" in out["say"]
        assert "greys this bar" in out["say"]

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
        out = fr.read(ofi_dir="up", tick_dir="up", open_px=100.0,
                      high=120.0, low=100.0, close=120.0, **FULL)
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
        out = fr.read(ofi_dir="flat", tick_dir="up", open_px=5000.0,
                      high=5010.0, low=5000.0, close=5009.0, **FULL)
        assert out["where"] == ""
        assert "no direction" in out["say"]

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
        # THE WHOLE ROUTE, NOT 2,500 CHARACTERS OF IT. That slice
        # is a silent assertion that nothing is ever added above
        # the line it looks for, and it broke the moment three
        # settings went in -- on a change that had nothing to do
        # with the pair. Eighth of these across the suite.
        blk = _py_func("api_journal_settings")
        assert "fused_ofi: int | None = None," in blk
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
        # EXECUTED, NOT SCANNED. This read `"_fused_gate(" in seg`
        # and broke the moment the shown-only call was hoisted out
        # of the dict literal so it could keep a memory -- a source
        # shape, not the behaviour. The behaviour is that a
        # strategy which is not `together` still gets a populated
        # panel, and only running it can say so.
        assert "_shown" in seg or "_fused_gate(" in seg
        assert "_shown = _fused_gate(" in blk, \
            "the panel is never built for the other strategies"

    def test_it_is_drawn_while_another_strategy_trades(self, client,
                                                       monkeypatch):
        """THE SAME ASSERTION, RUN. Chris watches this panel while
        the column trades something else, so a payload that is
        empty on every strategy but its own is the feature not
        working at all."""
        from liqmap import web
        _cme(monkeypatch)
        rt = web.runtime()
        s = rt.journal.settings("ES")
        s.strategy = "stackwin"
        rt.journal.save_settings(s)
        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH).json()
        jr = d.get("journal") or {}
        assert "error" not in jr, jr.get("error")
        fg = jr.get("fusedgate")
        assert fg, "no fused panel while another strategy traded"
        assert fg.get("decides") is False
        assert fg.get("turning") is not None, \
            "the turning row is missing off its own strategy"

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
        assert "BOTH " in fn and "THEY DISAGREE" in fn
        assert "outer (OFI)" in fn and "inner (tick)" in fn, \
            "the panel does not name the two candles the chart draws"
        assert "FG.pos" in fn, "the candle read is never drawn"
        assert "FG.legs" in fn or "FG.legs || {}" in fn


from tests.test_cme_panels import _cme, client  # noqa: E402,F401
# THE SERVED PAGE, not web.py's source: reading the script out of
# the Python file would carry that file's backslash escapes and
# test a string nobody is ever sent.
from tests.test_stack_strip import page_js  # noqa: E402,F401

AUTH = {"Authorization": "Bearer t0ken"}


def _py_func(name: str, path: str = "liqmap/web.py") -> str:
    """One whole Python function, by indentation.

    Here because fixed character counts expire. A `src[i:i + N]`
    asserts the code is shorter than N, and this suite has had
    eight of them fail on unrelated changes.
    """
    import pathlib as _pl
    src = (_pl.Path(__file__).resolve().parents[1] / path).read_text()
    i = src.index(f"def {name}(")
    start = src.rfind("\n", 0, i) + 1
    indent = i - start
    lines = src[start:].split("\n")
    out = [lines[0]]
    for line in lines[1:]:
        if line.strip() and (len(line) - len(line.lstrip())) <= indent:
            break
        out.append(line)
    return "\n".join(out)


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

    @pytest.mark.parametrize("where", ["settling", "together", ""])
    def test_nothing_else_closes_a_trade_with_the_switches_off(
            self, where):
        """OFF IS THE DEFAULT, AND OFF MEANS UNCHANGED.

        The bounce, the bar close and the settling exit all came
        back from f175c20 at Chris's ask -- "when we kept
        upgrading the chart it disappeared and dont work like that
        anymore" -- but each is behind its own switch and each
        starts off, because off is how this rule has traded since
        `7d961a9`. Applying the restore must change nothing until
        a box is ticked.

        `bounce` is not in this list because `where` cannot even
        READ "bounce" with the switch off -- that is the point of
        putting the switch on the rung rather than on the exit.
        """
        assert fr.exit_reason("up", where) == "", \
            f"{where} still closes a trade with every switch off"
        assert fr.exit_reason("down", where) == ""
        # And with a bar stamp that HAS changed, which is the
        # bar-close exit's input -- still nothing, switch off.
        assert fr.exit_reason("up", where, bar_ts=2.0,
                              open_bar_ts=1.0) == ""

    def test_the_bounce_rung_is_the_switch(self):
        """ONE SWITCH, ONE PLACE. With it off `where` can never be
        "bounce", so nothing downstream needs a second flag and
        the panel cannot show a state the rule is ignoring."""
        bar = dict(ofi_dir="up", tick_dir="up", open_px=7180.0,
                   high=7190.0, low=7179.0, close=7183.0,
                   bar_ts=1.0, **FULL)
        off = fr.read(**bar)
        on = fr.read(bounce_on=True, **bar)
        assert off["back"] >= 0.33, "this bar is not a bounce"
        assert off["where"] == "together"
        assert on["where"] == "bounce"
        # THE PANEL IS TOLD WHETHER THE RUNG IS LIVE, so it can
        # say "bounce off" rather than drawing a percentage that
        # decides nothing -- which is what that dial did for six
        # commits without anything saying so.
        assert off["bounce_on"] is False
        assert on["bounce_on"] is True
        # THE REASON, NOT JUST THE REFUSAL. Without the bounce
        # branch in `entry_side` the side is still "" -- it falls
        # through to "nothing settled" -- so a test that only
        # checks the side passes while the panel explains the
        # refusal wrongly, which is how a day gets lost here.
        side, why = fr.entry_side(on["where"], on["side"])
        assert side == ""
        assert "bouncing" in why, why
        assert "nothing settled" not in why
        assert "bouncing" in fr.exit_reason("up", on["where"])

    def test_state_carries_the_switch_to_the_reading(self):
        """`state` IS WHAT THE ROUTE CALLS, not `read`. A switch
        that stops at `state`'s door leaves the checkbox, the
        database and the panel all correct and the rung dead."""
        bar = dict(ofi_dir="up", tick_dir="up", open_px=7180.0,
                   high=7190.0, low=7179.0, close=7183.0,
                   bar_ts=1.0, held_s=9.0, **FULL)
        assert fr.state(**bar)["where"] == "together"
        on = fr.state(bounce_on=True, **bar)
        assert on["where"] == "bounce"
        assert on["entry"] == "", "a bounce opened a trade"
        assert on["bounce_on"] is True

    def test_zero_per_cent_is_the_tightest_not_off(self):
        """A DIAL AT ZERO MEANS "any give-back at all". Without
        the `back > 0.0` guard it makes `back >= 0` true on every
        bar including one that has not moved, and the rule never
        trades again -- which looks exactly like the switch being
        stuck on."""
        still = dict(ofi_dir="up", tick_dir="up", open_px=7180.0,
                     high=7180.0, low=7180.0, close=7180.0,
                     bar_ts=1.0, bounce_at=0.0, bounce_on=True,
                     **FULL)
        assert fr.read(**still)["where"] != "bounce"
        moved = dict(still, high=7190.0, close=7189.75)
        assert fr.read(**moved)["where"] == "bounce"

    def test_the_bar_close_and_settling_exits_are_separate(self):
        """IN PRACTICE THEY FIRE AT NEARLY THE SAME MOMENT, since
        a fresh bar is unsettled by construction. They are two
        switches so that overlap is a choice rather than a
        surprise, and each has to work on its own."""
        assert "has closed" in fr.exit_reason(
            "up", "together", bar_ts=2.0, open_bar_ts=1.0,
            bar_exit=True)
        assert fr.exit_reason("up", "settling", bar_ts=2.0,
                              open_bar_ts=1.0, bar_exit=True) \
            == fr.exit_reason("up", "together", bar_ts=2.0,
                              open_bar_ts=1.0, bar_exit=True), \
            "the bar-close reason depends on the reading"
        assert "no longer settled" in fr.exit_reason(
            "up", "settling", settle_exit=True)
        # SETTLING ALONE DOES NOT CLOSE IT when only the bar-close
        # switch is on and the bar has not changed.
        assert fr.exit_reason("up", "settling", bar_ts=1.0,
                              open_bar_ts=1.0, bar_exit=True) == ""

    def test_a_missing_bar_stamp_closes_nothing(self):
        """Zero is "we do not know which bar this opened in", and
        closing every such position at the first boundary is
        acting on missing data. The same guard `stackwin.expired`
        carries."""
        assert fr.exit_reason("up", "together", bar_ts=2.0,
                              open_bar_ts=0.0, bar_exit=True) == ""
        assert fr.exit_reason("up", "together", bar_ts=0.0,
                              open_bar_ts=1.0, bar_exit=True) == ""

    def test_the_exit_reads_no_clock(self):
        """THE BAR STAMP IS BACK, THE CLOCK IS NOT. `bar_ts` and
        `open_bar_ts` return with the bar-close switch; a wall
        clock never does -- "Gate 2's flip only, no clock", and
        nothing here expires on elapsed time.

        THE NINE ORIGINAL PARAMETERS KEEP THEIR POSITIONS. Every
        caller passes them positionally, and an argument slotted
        in among them re-reads a timeframe name as a direction --
        which has happened here once and cost ten tests."""
        import inspect
        sig = inspect.signature(fr.exit_reason)
        assert list(sig.parameters)[:9] == [
            "trade_side", "where", "tick_dir", "tf_dir", "tf_on",
            "tf_name", "tf2_dir", "tf2_on", "tf2_name"]
        assert list(sig.parameters)[9:] == [
            "bar_ts", "open_bar_ts", "bar_exit", "settle_exit"]
        for p in list(sig.parameters.values())[9:]:
            assert p.default in (0.0, False), \
                f"{p.name} does not default to off"
        for bad in ("held_s", "now_ts", "period_s", "min_hold_s"):
            assert bad not in sig.parameters, \
                f"exit_reason takes {bad} -- that is a clock"

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
        """`or 33` ANYWHERE ON THIS PATH would turn a zero into a
        33. The selector is off the page now because the value
        gates nothing, but the stored setting and the route that
        reads it are still here, and the day it gets a job again
        this is the fault it will have."""
        blk = self._step()
        i = blk.index("_fb_pct = ")
        seg = blk[i:i + 420]
        assert "or 33" not in seg
        assert "is None" in seg
        from liqmap.journal import Settings
        assert Settings(fused_bounce=0).to_dict()["fused_bounce"] == 0

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
        # THE WHOLE CALL, NOT A FIXED NUMBER OF CHARACTERS. A
        # slice like `blk[i:i + 420]` is an assertion about how
        # long the call happens to be, and it expired the moment
        # four arguments were added -- the test failed on a change
        # it was meant to approve. Read to the closing paren.
        seg = blk[i:blk.index("or disagreed", i)]
        assert 'fus.get("where")' in seg
        # THE BAR STAMP REACHES IT AGAIN, and it is the TRADE'S
        # stamp against the bar's. Handing it the same value
        # twice, or the live bar for both, is a bar-close exit
        # that can never fire -- and nothing else here would say
        # so, because the switch would look wired.
        assert 'fus.get("bar_ts")' in seg
        assert 'getattr(t, "bar_ts"' in seg, \
            "the exit compares the live bar against itself"
        assert 'getattr(s, "fused_bar_exit"' in seg
        assert 'getattr(s, "fused_settle_exit"' in seg

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

    def test_his_bar_agrees_when_the_inner_candle_does(self):
        """The inner candle closed above its open, so with the book
        up the two agree -- whatever the wick above them says."""
        out = fr.read(ofi_dir="up", tick_dir="up", close=7169.50,
                      **self.BAR, **FULL)
        assert out["tick_dir"] == "up"
        assert out["where"] == "together"
        assert out["side"] == "up"

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
        assert "outer (OFI)" in html
        assert "inner (tick)" in html

    def test_the_reading_carries_no_percentages(self, page_js):
        """"i dont care how much just what it actually reads"."""
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**self.BAR, "run_dir": "up", "run_pts": 23.5,
                       "back": 0.426})
        assert "off the open and is" not in html
        assert "of the way up this" not in html
        assert "counts as turned at" not in html

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
        used to close the trade.

        WITH THE SWITCH OFF -- the default -- that is still true,
        because `where` cannot read "bounce" at all."""
        from liqmap.journal import Settings, should_close
        t, s = self._trade(), Settings(**self.SET)
        bar = dict(ofi_dir="up", tick_dir="up", open_px=7160.0,
                   high=7170.0, low=7159.0, close=7162.25,
                   bar_ts=1.0, **FULL)
        off = fr.read(**bar)
        assert off["where"] != "bounce"
        assert should_close(t, s, 7162.25, "", window_rule=True,
                            disagreed=fr.exit_reason(
                                "up", off["where"])) == ""

    def test_with_the_bounce_on_it_does_close_first(self):
        """THE OTHER HALF, SO THE SWITCH IS NOT COSMETIC. Turned
        on, the same bar closes the trade short of the target --
        which is the behaviour he asked to have back, and the
        reason it is a switch rather than a restore."""
        from liqmap.journal import Settings, should_close
        t, s = self._trade(), Settings(**self.SET)
        bar = dict(ofi_dir="up", tick_dir="up", open_px=7160.0,
                   high=7170.0, low=7159.0, close=7162.25,
                   bar_ts=1.0, bounce_on=True, **FULL)
        on = fr.read(**bar)
        assert on["where"] == "bounce"
        assert should_close(t, s, 7162.25, "", window_rule=True,
                            disagreed=fr.exit_reason(
                                "up", on["where"])) == "disagree"

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

    def test_the_reading_offers_no_side_when_they_disagree(self):
        out = fr.read(ofi_dir="up", tick_dir="down", open_px=7180.00,
                      high=7184.00, low=7154.00, close=7169.50,
                      **FULL)
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

    def test_the_other_fused_rule_has_its_own_entirely(self):
        """NOT AN ASSUMPTION ABOUT THE OTHER RULE -- its own
        function, on a real bar.

        The two can agree on a given bar; that is not the point.
        The point is that `live_signal` asks a different question
        -- 10% of both legs and the two agreeing, with no hold, no
        settling and no timeframe filter -- so the panel
        describing THIS rule while THAT one trades is a claim
        about a trade nobody can check."""
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
        # The readings are built from different questions, and
        # only one of them is the rule actually running.
        assert ours["where"] in ("together", "apart", "")
        import inspect
        assert "min_fraction" in inspect.signature(
            live_signal).parameters, \
            "the other rule no longer has its own warm-up"
        for gate in ("held_s", "min_hold_s"):
            assert gate not in inspect.signature(
                live_signal).parameters, \
                f"the other rule has a {gate} after all"

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
                            "ticks": False, "body": 0.0,
                            "travel": 0.0, "agrees": True,
                            "left_s": None, "decides": True}
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

    def test_the_dial_no_longer_gates_anything(self):
        """It was the bounce threshold, then the band that decided
        when price counted as turned. Price is the inner candle
        itself now, so there is nothing left for a percentage to
        decide -- and a control that changes no reading is worse
        than no control, so it is off the page.

        The value is still stored and still rides in the payload;
        only the selector is gone, which is one line to restore.
        """
        src = self._web()
        assert 'id="jnlBounce"' not in src, \
            "an inert selector is still on the page"
        assert "$('jnlBounce')" not in src
        for pct in (0.0, 0.33):
            a = fr.read(ofi_dir="up", tick_dir="up", open_px=100.0,
                        high=120.0, low=95.0, close=110.0,
                        bounce_at=pct, **FULL)
            assert a["where"] == "together", pct
            assert a["tick_dir"] == "up", pct

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
        assert "_wl(float(n))" in fn, \
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
        assert fn.index("_tf_dir, _tf_name, _tf_left") \
            < fn.index("def _nothing("), \
            "the closure is defined before the name it reads"
        # AND THE STRENGTH NUMBERS TOO. `_nothing` carries `body`
        # and `travel` now, and `_strength` is another closure
        # defined in the same stretch -- the same shape, one
        # revision later.
        assert fn.index("def _strength(") < fn.index("def _nothing("), \
            "the strength helper is defined after the path that reads it"

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
        assert "float(n) - float(age)" in fn, \
            "the countdown is not the candle's own remaining time"
        # AND THE TICK BAR'S HALF OF IT. A filter cut by a count
        # has prints remaining where a candle has seconds, and the
        # two must not be read off the same arithmetic.
        assert "float(-n) - float(done)" in fn, \
            "a tick filter has no countdown of its own"
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


class TestTheTwoCandlesAreOneRowEach:
    """Chris, pointing at the chart: "the outer shale is the body
    the OFI the inner candle is the tick candle what this is
    supposed to do it tell me whats actually happening between the
    2", and then: "it should be simple downs and ups only i dont
    care how much just what it actually reads".

    The body row used to blink because it was drawn only when it
    disagreed with a DERIVED price. There is no derived price any
    more -- the inner candle is the reading -- so there is one row
    for the outer and one for the inner, both always drawn.
    """

    def test_the_two_rows_are_always_drawn(self, page_js):
        from tests.test_stack_strip import run_js
        for ofi, tick in (("up", "up"), ("up", "down"), ("up", ""),
                          ("", "")):
            html = run_js(page_js, "fusedGate",
                          {**TestThePanelActuallyRenders.BAR,
                           "ofi_dir": ofi, "tick_dir": tick,
                           "body_dir": tick})
            assert "outer (OFI)" in html, (ofi, tick)
            assert "inner (tick)" in html, (ofi, tick)

    def test_neither_row_is_behind_a_condition(self):
        src = (SRC / "web.py").read_text()
        i = src.index("function paintFusedGate(")
        fn = src[i:src.index("\nfunction ", i + 10)]
        for row in ("leg('outer (OFI)'", "leg('inner (tick)'"):
            j = fn.index(row)
            line = fn[fn.rindex("\n", 0, j):j]
            assert "?" not in line, f"{row} is drawn behind a condition"

    def test_the_body_is_not_drawn_twice(self):
        """They are one reading under two names now. Two identical
        rows is noise, and a reader checking one against the other
        learns nothing."""
        src = (SRC / "web.py").read_text()
        i = src.index("function paintFusedGate(")
        fn = src[i:src.index("\nfunction ", i + 10)]
        assert "body (shape)" not in fn

    def test_price_and_the_body_can_never_differ(self):
        """THE DAY THEY DIFFER THE PANEL IS LYING. A derived price
        put this panel into open contradiction with the candle
        beside it: on a bar the chart greyed, it said "together"
        and offered a short."""
        for tick in ("up", "down", "", "flat"):
            out = fr.read(ofi_dir="up", tick_dir=tick, open_px=100.0,
                          high=120.0, low=95.0, close=110.0, **FULL)
            assert out["tick_dir"] == out["body_dir"], tick

    def test_it_is_held_through_a_tie_before_the_bar_settles(self):
        """A candle sits exactly on its open often enough that the
        row flickered. The settling gate exists to stop a rule
        reading noise, and this is the reading itself."""
        unsettled = dict(net_ofi=10, ofi_target=4000, prints=20,
                         tick_target=10000)
        first = fr.read(ofi_dir="up", tick_dir="up", open_px=100.0,
                        high=120.0, low=100.0, close=120.0,
                        bar_ts=100.0, **unsettled)
        assert first["settled"] is False
        tie = fr.read(ofi_dir="up", tick_dir="", open_px=100.0,
                      high=120.0, low=100.0, close=100.0,
                      bar_ts=100.0, prev=first["held"], **unsettled)
        assert tie["tick_dir"] == "up"
        assert tie["body_dir"] == "up"

    def test_it_does_not_survive_into_the_next_candle(self):
        kw = dict(open_px=100.0, high=120.0, low=100.0, close=120.0)
        first = fr.read(ofi_dir="up", tick_dir="up", bar_ts=100.0,
                        **kw, **FULL)
        nxt = fr.read(ofi_dir="up", tick_dir="", bar_ts=200.0,
                      prev=first["held"], **kw, **FULL)
        assert nxt["tick_dir"] == ""
        assert nxt["body_dir"] == ""


class TestWhatItSaysAboutTheTwo:
    """"tell me whats actually happening between the 2" -- in ups
    and downs, with no numbers in it."""

    KW = dict(open_px=7160.50, high=7184.00, low=7154.00,
              close=7169.50)

    def test_his_bar_reads_as_a_disagreement(self):
        """Book DOWN 2,097, inner candle UP 9 points. The chart
        greys it; so does this."""
        out = fr.read(ofi_dir="down", tick_dir="up", net_ofi=-2097,
                      ofi_target=4000, prints=39236,
                      tick_target=10000, tradeable=False, **self.KW)
        assert out["where"] == "apart"
        assert out["side"] == ""
        assert out["grayed"] is True
        assert out["say"] == ("the OFI candle is down and the tick "
                              "candle is up -- they disagree, and "
                              "the chart greys this bar")

    def test_agreeing_says_so_and_names_both(self):
        out = fr.read(ofi_dir="up", tick_dir="up", **self.KW, **FULL)
        assert out["where"] == "together"
        assert out["say"] == ("the OFI candle is up and the tick "
                              "candle is up -- they agree")

    @pytest.mark.parametrize("ofi,tick,word", [
        ("", "up", "OFI candle has no direction"),
        ("flat", "up", "OFI candle has no direction"),
        ("up", "", "tick candle has no direction"),
    ])
    def test_a_missing_direction_names_which_one(self, ofi, tick,
                                                 word):
        out = fr.read(ofi_dir=ofi, tick_dir=tick, **self.KW, **FULL)
        assert word in out["say"], out["say"]
        assert out["where"] == ""

    def test_there_are_no_numbers_in_the_sentence(self):
        """"i dont care how much just what it actually reads"."""
        import re
        for ofi, tick in (("up", "up"), ("down", "up"), ("up", ""),
                          ("", "")):
            out = fr.read(ofi_dir=ofi, tick_dir=tick, **self.KW,
                          **FULL)
            assert not re.search(r"\d", out["say"]), out["say"]

    def test_the_panel_draws_no_percentages_in_the_reading(self):
        src = (SRC / "web.py").read_text()
        i = src.index("function paintFusedGate(")
        fn = src[i:src.index("\nfunction ", i + 10)]
        for gone in ("off the open and is", "of the way up this",
                     "counts as turned at", "off its extreme"):
            assert gone not in fn, f"the panel still draws: {gone}"

    def test_the_candles_own_prices_are_still_drawn(self):
        """Those are not a derived reading -- they are what lets
        him line this panel up against the chart header, which is
        how he caught the contradiction in the first place."""
        src = (SRC / "web.py").read_text()
        i = src.index("function paintFusedGate(")
        fn = src[i:src.index("\nfunction ", i + 10)]
        for want in ("chPx(FG.open)", "chPx(FG.high)", "chPx(FG.low)",
                     "chPx(FG.close)", "FG.net_ofi"):
            assert want in fn, f"the panel no longer draws {want}"

    def test_the_panel_says_they_disagree(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**TestThePanelActuallyRenders.BAR,
                       "where": "apart", "ofi_dir": "down",
                       "tick_dir": "up", "body_dir": "up",
                       "agree": False, "side": ""})
        assert "THEY DISAGREE" in html
        assert "outer (OFI)" in html and "inner (tick)" in html

    def test_and_when_they_agree(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**TestThePanelActuallyRenders.BAR,
                       "where": "together", "ofi_dir": "up",
                       "tick_dir": "up", "body_dir": "up",
                       "agree": True, "side": "up"})
        assert "BOTH UP" in html


class TestTheTurningRow:
    """Chris, on the first version: "thoss one has a turning but
    its stuck in place doing nothing", and then "it should read
    what the chart is doing in real time".

    IT WAS NOT STUCK, IT WAS ANCHORED. It measured the close
    against the candle's HIGH -- a number that only moves when a
    new high prints -- so on a bar that ran up and pulled back it
    said "down" and went on saying "down" through a fourteen-point
    rally, because every one of those closes was still below a high
    set minutes earlier.

    THE EXTREME HAS TO TRAVEL. While it reads up the reference high
    moves up with every new high, and it flips only when price
    comes the band back off THAT high.
    """

    BAR = dict(open_px=7160.50, high=7184.00, low=7154.00)

    @staticmethod
    def _walk(path, band=4.0, open_px=7160.50, bar_ts=1.0):
        """The row's reading at each close along a path."""
        st, out = {}, []
        for c in path:
            st = fr.turn_state(c, band, st, open_px=open_px,
                               bar_ts=bar_ts)
            out.append((st["dir"], st["ticks_off"]))
        return out

    def test_it_reads_the_rally_while_the_rally_is_happening(self):
        """THE BUG HE REPORTED, AS A TEST. His own bar: open
        7160.50, high 7184.00, back to 7169.50, then a rally. The
        old row said "down" for every close in that rally because
        they were all below 7184. This one does not."""
        path = [7166, 7172, 7178, 7184, 7180, 7175, 7169.5,
                7173, 7177, 7181]
        got = self._walk(path)
        assert [d for d, _ in got[:4]] == ["up"] * 4
        assert [d for d, _ in got[4:7]] == ["down"] * 3
        # THE RALLY. Three closes, each higher than the last, every
        # one of them below the 7184 the old row was pinned to.
        assert [d for d, _ in got[7:]] == ["up"] * 3, \
            "the row is pinned to a high that stopped moving"

    def test_the_extreme_travels_with_price(self):
        """A NEW HIGH MOVES THE REFERENCE. Without this the band is
        measured off the first high forever, which is the whole of
        the original bug."""
        st = {}
        for c in (7180.25, 7180.50, 7180.75, 7181.00):
            st = fr.turn_state(c, 4.0, st, open_px=7180.0, bar_ts=1.0)
        assert st["dir"] == "up"
        assert st["hi"] == pytest.approx(7181.00)
        # Three ticks back is not a turn; the fourth is.
        for c, want in ((7180.75, "up"), (7180.50, "up"),
                        (7180.25, "up"), (7180.00, "down")):
            st = fr.turn_state(c, 4.0, st, open_px=7180.0, bar_ts=1.0)
            assert st["dir"] == want, f"at {c}"

    def test_the_countdown_to_the_flip_is_visible(self):
        """`ticks_off` is how far back off the extreme price has
        already come -- what makes the flip predictable rather than
        sudden."""
        st = {}
        for c in (7180.25, 7180.50, 7180.75, 7181.00):
            st = fr.turn_state(c, 4.0, st, open_px=7180.0, bar_ts=1.0)
        seen = []
        for c in (7180.75, 7180.50, 7180.25):
            st = fr.turn_state(c, 4.0, st, open_px=7180.0, bar_ts=1.0)
            seen.append(st["ticks_off"])
        assert seen == [1.0, 2.0, 3.0]

    def test_a_new_candle_restarts_the_reference_not_the_direction(
            self):
        """IT USED TO BLANK, AND HE TOOK THAT BACK.

        Asked what the row should do at a new fused candle, Chris
        first chose "Start blank each candle". Having watched it:
        "turning and the 2 time frames goes blank before deciding
        instead of just rolling into the direction". A row that
        goes silent reads as broken, and the silence told him
        nothing he did not already know.

        SO THE EXTREMES RESET AND THE DIRECTION CARRIES. It stays
        an intrabar reading -- the band is still measured inside
        this candle -- and the row never blanks once it has said
        anything at all.
        """
        st = {}
        for c in (7180.25, 7180.50, 7180.75, 7181.00):
            st = fr.turn_state(c, 4.0, st, open_px=7180.0, bar_ts=1.0)
        assert st["dir"] == "up"
        fresh = fr.turn_state(7181.00, 4.0, st, open_px=7181.00,
                              bar_ts=2.0)
        assert fresh["dir"] == "up", "the row blanked at a new candle"
        # THE REFERENCE DID RESET. Four ticks below the NEW open
        # flips it; four ticks below the old high would not have.
        assert fresh["hi"] == pytest.approx(7181.00)
        flipped = fresh
        for c in (7180.75, 7180.50, 7180.25, 7180.00):
            flipped = fr.turn_state(c, 4.0, flipped,
                                    open_px=7181.00, bar_ts=2.0)
        assert flipped["dir"] == "down"

    def test_it_is_blank_only_before_it_has_ever_decided(self):
        """The one blank that is honest: nothing has been seen
        yet, so there is no direction to carry."""
        first = fr.turn_state(7180.25, 4.0, None, open_px=7180.0,
                              bar_ts=1.0)
        assert first["dir"] == ""

    def test_the_first_direction_comes_off_whichever_end_it_left(self):
        """BOTH EXTREMES ARE TRACKED WHILE IT IS BLANK, so a drift
        up and then a drop names the turn off the drift's high
        rather than off the open it happened to start at."""
        st = {}
        for c in (7180.25, 7180.50, 7180.25, 7180.00, 7179.50):
            st = fr.turn_state(c, 4.0, st, open_px=7180.0, bar_ts=1.0)
        assert st["dir"] == "down"
        # Four ticks off 7180.50, not off the 7180.00 open -- a
        # reading taken from the open would still be blank here.
        assert st["hi"] == pytest.approx(7180.50)

    def test_the_band_is_the_setting_not_a_constant(self):
        """Chris picked a selector over a fixed number, so a wider
        band has to actually hold the direction longer."""
        path = [7180.25, 7180.50, 7180.75, 7181.00, 7180.75,
                7180.50, 7180.25, 7180.00]
        tight = self._walk(path, band=4.0, open_px=7180.0)
        wide = self._walk(path, band=16.0, open_px=7180.0)
        assert tight[-1][0] == "down"
        assert wide[-1][0] != "down", \
            "a 16-tick band flipped on a 4-tick pullback"

    def test_read_passes_the_band_through(self):
        near = fr.read(ofi_dir="up", tick_dir="up", close=7183.00,
                       turn_ticks=4.0, **self.BAR, **FULL)
        wide = fr.read(ofi_dir="up", tick_dir="up", close=7183.00,
                       turn_ticks=24.0, **self.BAR, **FULL)
        assert near["turning"]["band"] == pytest.approx(4.0)
        assert wide["turning"]["band"] == pytest.approx(24.0)

    def test_the_band_it_reports_is_the_band_it_flips_on(self):
        """THE LABEL AND THE BEHAVIOUR, CHECKED TOGETHER. Reading
        the setting for the printed number while the row still
        flips on the constant is a panel saying "flipping at 24"
        and flipping at 4 -- a mutation that passed every test
        here until this one existed, and exactly the kind of lie
        this box keeps being rebuilt over.

        So this walks a path through `read`, not `turn_state`."""
        def walk(band):
            prev, seen = None, []
            for c in (7180.25, 7180.50, 7180.75, 7181.00,
                      7180.75, 7180.50, 7180.25, 7180.00):
                out = fr.read(ofi_dir="up", tick_dir="up",
                              open_px=7180.0, high=7181.0,
                              low=7180.0, close=c, bar_ts=1.0,
                              prev=prev, turn_ticks=band, **FULL)
                prev = out["held"]
                seen.append(out["turning"]["dir"])
            return seen
        tight, wide = walk(4.0), walk(24.0)
        assert tight[-1] == "down"
        assert wide[-1] != "down", \
            "the row flipped on 4 ticks while reporting a 24 band"
        assert tight != wide

    def test_the_state_rides_home_in_held(self):
        """A FIELD DROPPED FROM `held` DOES NOT ERROR -- it
        silently resets the reference on every poll, which is a row
        that can never flip and looks exactly like the bug this
        replaced."""
        out = fr.read(ofi_dir="up", tick_dir="up", close=7183.00,
                      bar_ts=1.0, **self.BAR, **FULL)
        assert "turn" in (out.get("held") or {}), \
            "the travelling extreme is not carried back"
        t = out["held"]["turn"]
        for k in ("dir", "hi", "lo", "bar_ts"):
            assert k in t, f"held turn is missing {k}"

    def test_without_the_memory_the_path_stops_mattering(self):
        """THE FAILURE MODE, STATED. Hand it no previous reading
        and every poll re-derives from the open, so how price got
        to a price has no effect at all -- which is a row that
        cannot track a turn, only a distance.

        This is not hypothetical: the shown-only panel call was
        handed `prev=None` and would have behaved exactly like
        this on every strategy but its own."""
        up = [7180.25, 7180.50, 7180.75, 7181.00]
        down = [7179.75, 7179.50, 7179.25, 7181.00]
        last_up = last_down = None
        for c in up:
            last_up = fr.turn_state(c, 4.0, None, open_px=7180.0,
                                    bar_ts=1.0)
        for c in down:
            last_down = fr.turn_state(c, 4.0, None, open_px=7180.0,
                                      bar_ts=1.0)
        # Same close, opposite journeys, identical reading.
        assert last_up == last_down
        # WITH the memory those two paths read differently, which
        # is the entire value of carrying it.
        st_u, st_d = {}, {}
        for c in up:
            st_u = fr.turn_state(c, 4.0, st_u, open_px=7180.0,
                                 bar_ts=1.0)
        for c in down:
            st_d = fr.turn_state(c, 4.0, st_d, open_px=7180.0,
                                 bar_ts=1.0)
        assert st_u["lo"] != st_d["lo"]

    def test_the_inner_candle_is_unchanged_by_it(self):
        """THE ROW IS BESIDE IT, NOT INSTEAD OF IT. Three attempts
        to make a derived reading the price leg each put the panel
        into contradiction with the candle next to it."""
        out = fr.read(ofi_dir="up", tick_dir="up", close=7175.00,
                      **self.BAR, **FULL)
        assert out["tick_dir"] == "up"
        assert out["where"] == "together"
        assert out["side"] == "up"

    def test_it_decides_nothing(self):
        import inspect
        for fn in (fr.entry_side, fr.exit_reason):
            ps = inspect.signature(fn).parameters
            for bad in ("turning", "ticks_off", "turn_ticks"):
                assert bad not in ps, f"{fn.__name__} takes {bad}"
        prev, out = None, None
        for close in (7184.00, 7180.00, 7175.00, 7169.50):
            out = fr.read(ofi_dir="up", tick_dir="up", close=close,
                          bar_ts=1.0, prev=prev, **self.BAR, **FULL)
            prev = out["held"]
            assert out["turning"]["decides"] is False
            assert out["side"] == "up", \
                "the turning row moved the verdict"
        # It read DOWN on the last of those and the rule still
        # offered UP -- the separation, stated rather than implied.
        assert out["turning"]["dir"] == "down"

    def test_a_down_candle_turns_up_off_its_low(self):
        st = {}
        for c in (7183.75, 7183.50, 7183.25, 7183.00):
            st = fr.turn_state(c, 4.0, st, open_px=7184.0, bar_ts=1.0)
        assert st["dir"] == "down"
        for c in (7183.25, 7183.50, 7183.75, 7184.00):
            st = fr.turn_state(c, 4.0, st, open_px=7184.0, bar_ts=1.0)
        assert st["dir"] == "up"

    def test_junk_never_raises(self):
        for bad in (None, "", "abc", [], {}):
            st = fr.turn_state(bad, bad, {"dir": bad, "hi": bad,
                                          "lo": bad, "bar_ts": bad},
                               open_px=bad, bar_ts=bad)
            assert st["dir"] in ("", "up", "down")
            assert isinstance(st["ticks_off"], float)
        out = fr.read(ofi_dir="up", tick_dir="up", close="abc",
                      turn_ticks="abc", **self.BAR, **FULL)
        assert out["turning"]["band"] == pytest.approx(fr.TURN_TICKS)

    def test_a_zero_band_is_not_offered(self):
        """A band of zero flips the row on every print. The snapper
        refuses it; `_snap_bounce` deliberately does NOT, because
        zero is one of the bounce choices -- the two look alike and
        must not be copied from one another."""
        from liqmap.journal import _snap_turn
        assert 0 not in fr.TURN_CHOICES
        assert _snap_turn(0) == int(fr.TURN_TICKS)
        assert _snap_turn(-5) == int(fr.TURN_TICKS)
        assert _snap_turn("junk") == int(fr.TURN_TICKS)
        for c in fr.TURN_CHOICES:
            assert _snap_turn(c) == int(c)
        assert _snap_turn(5) in (4, 6)

    def test_the_band_round_trips_through_the_journal(self, tmp_path):
        from liqmap.journal import Journal
        j = Journal(str(tmp_path / "j.db"))
        s = j.settings("ES")
        s.fused_turn = 16
        assert j.save_settings(s).fused_turn == 16
        assert j.settings("ES").fused_turn == 16
        assert j.settings("ES").to_dict()["fused_turn"] == 16
        s.fused_turn = 999
        assert j.save_settings(s).fused_turn in fr.TURN_CHOICES

    def test_the_selector_exists_and_is_saved(self):
        src = (SRC / "web.py").read_text()
        assert 'id="jnlFusedTurn"' in src, "no dial on the page"
        for c in fr.TURN_CHOICES:
            assert f'value="{int(c)}"' in src
        assert "fused_turn: $('jnlFusedTurn')" in src, \
            "the choice is never sent"
        assert "set('jnlFusedTurn'" in src, \
            "the saved choice never comes back to the dropdown"
        assert "jnlFusedTurn: $('jnlFusedTurn')" in src, \
            "the choice is not pinned and will snap back"

    def test_the_route_takes_it(self, client, monkeypatch):
        _cme(monkeypatch)
        r = client.post("/api/journal/settings",
                        params={"coin": "ES", "fused_turn": 8},
                        headers=AUTH).json()
        assert r["settings"]["fused_turn"] == 8
        # AND A HAND-TYPED JUNK VALUE CANNOT REACH THE ROW.
        r = client.post("/api/journal/settings",
                        params={"coin": "ES", "fused_turn": 4000},
                        headers=AUTH).json()
        assert r["settings"]["fused_turn"] in fr.TURN_CHOICES

    def test_the_panel_draws_it_and_says_it_decides_nothing(self):
        src = (SRC / "web.py").read_text()
        i = src.index("function paintFusedGate(")
        fn = src[i:src.index("\nfunction ", i + 10)]
        assert "leg('turning'" in fn
        assert "shown only" in fn, \
            "nothing says the rule does not read this row"
        assert "running " in fn, \
            "the line does not say what the band is measured from"
        j = fn.index("leg('turning'")
        line = fn[fn.rindex("\n", 0, j):j]
        assert "?" not in line, "the row is drawn behind a condition"

    def test_it_renders(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**TestThePanelActuallyRenders.BAR,
                       "tick_dir": "up", "ofi_dir": "up",
                       "where": "together", "agree": True,
                       "side": "up",
                       "turning": {"dir": "up", "ticks_off": 3.0,
                                   "band": 4.0, "decides": False}})
        assert "turning" in html
        assert "3 ticks back off its running high" in html
        assert "flipping at 4" in html
        assert "shown only" in html

    def test_the_blank_state_says_why_it_is_blank(self, page_js):
        """A ROW THAT GOES SILENT READS AS BROKEN. It starts blank
        at every candle now, so the blank has to explain itself."""
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**TestThePanelActuallyRenders.BAR,
                       "tick_dir": "up", "ofi_dir": "up",
                       "where": "together", "agree": True,
                       "side": "up",
                       "turning": {"dir": "", "ticks_off": 0.0,
                                   "band": 4.0, "decides": False}})
        assert "no direction yet" in html
        assert "4 ticks off" in html

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
        t = (jr.get("fusedgate") or {}).get("turning")
        assert t is not None and t.get("decides") is False

    def test_the_dial_reaches_the_live_panel_on_both_paths(
            self, client, monkeypatch):
        """THE SETTING, THROUGH THE ROUTE, TO THE SCREEN. Dropping
        the band argument at either call site leaves the dropdown
        working, the database correct and the panel flipping on
        the default -- a dial wired to nothing, which every source
        test here passed happily. Only reading the payload back
        catches it, and it has to be read on BOTH strategies
        because the two call sites are separate code.

        The band is checked on its own strategy and on another,
        because the fused panel is drawn either way."""
        from liqmap import web
        _cme(monkeypatch)
        rt = web.runtime()

        def band_for(strategy, ticks):
            s_ = rt.journal.settings("ES")
            s_.strategy = strategy
            s_.fused_turn = ticks
            rt.journal.save_settings(s_)
            d = client.get("/api/read",
                           params={"coin": "ES", "interval": "15m",
                                   "timeframes": "15m,5m,1m"},
                           headers=AUTH).json()
            jr = d.get("journal") or {}
            assert "error" not in jr, jr.get("error")
            return ((jr.get("fusedgate") or {}).get("turning")
                    or {}).get("band")

        for strategy in ("together", "stackwin"):
            assert band_for(strategy, 24) == pytest.approx(24.0), \
                f"the dial never reaches the panel on {strategy}"
            assert band_for(strategy, 2) == pytest.approx(2.0), \
                f"the panel is pinned to one band on {strategy}"

    def test_the_shown_only_path_keeps_its_own_memory(self):
        """OFF ITS OWN STRATEGY THE PANEL IS COMPUTED FRESH, and
        that call used to be handed `prev=None` -- which, for a row
        made entirely of state, means blank on every poll and a
        direction it can never find. Silent rather than wrong,
        which is harder to notice."""
        src = (SRC / "web.py").read_text()
        i = src.index("_shown = _fused_gate(")
        seg = src[i:i + 500]
        assert "_sprev" in seg, "the shown-only call has no memory"
        assert "_fused_shown_held" in src, \
            "nothing keeps the shown-only reading between polls"
        # AND NOT THE TRADED ONE'S. A shared attribute would let a
        # panel poll advance the extreme the live rule is holding.
        j = src.index("rt_._fused_shown_held")
        assert "rt_._fused_held" not in src[i:j], \
            "the shown-only path writes the traded memory"


class TestTheThreeRestoredSwitches:
    """Chris: "when we kept upgrading the chart it disappeared and
    dont work like that anymore now its different", about the
    version at `f175c20 the fused bar traded: settle, agree, hold
    five seconds`.

    THREE THINGS WENT, ALL IN `7d961a9 take profit or apart: the
    only two exits`: the bounce state, the bar-close exit and the
    settling exit. Asked which he wanted back he said all of it,
    as switches. So each is back behind its own box, and each
    starts OFF -- off is how the rule has traded since, and a
    restored feature that switches itself on is a rule change
    nobody asked for, applied to every market at once.
    """

    def test_they_all_default_off(self):
        from liqmap.journal import Settings
        s = Settings(coin="ES")
        assert s.fused_bounce_on is False
        assert s.fused_bar_exit is False
        assert s.fused_settle_exit is False

    def test_they_round_trip_through_the_journal(self, tmp_path):
        from liqmap.journal import Journal
        j = Journal(str(tmp_path / "j.db"))
        s = j.settings("ES")
        assert s.fused_bounce_on is False
        s.fused_bounce_on = True
        s.fused_bar_exit = True
        s.fused_settle_exit = True
        back = j.save_settings(s)
        for k in ("fused_bounce_on", "fused_bar_exit",
                  "fused_settle_exit"):
            assert getattr(back, k) is True, k
            assert j.settings("ES").to_dict()[k] is True, k

    def test_each_switch_is_independent(self, client, monkeypatch):
        """SENDING ONE MUST NOT CLEAR THE OTHERS. Every control on
        this page is saved by one POST that carries the whole
        form, so a field read without its own `is None` guard
        silently resets whatever was not sent."""
        _cme(monkeypatch)
        for k in ("fused_bounce_on", "fused_bar_exit",
                  "fused_settle_exit"):
            client.post("/api/journal/settings",
                        params={"coin": "ES", k: "true"},
                        headers=AUTH)
        got = client.post("/api/journal/settings",
                          params={"coin": "ES"},
                          headers=AUTH).json()["settings"]
        for k in ("fused_bounce_on", "fused_bar_exit",
                  "fused_settle_exit"):
            assert got[k] is True, f"{k} was cleared by another save"
        off = client.post("/api/journal/settings",
                          params={"coin": "ES",
                                  "fused_bounce_on": "false"},
                          headers=AUTH).json()["settings"]
        assert off["fused_bounce_on"] is False
        assert off["fused_bar_exit"] is True, \
            "clearing one cleared another"

    def test_the_page_has_all_four_controls(self):
        src = (SRC / "web.py").read_text()
        for el in ('id="jnlFBounce"', 'id="jnlFBouncePct"',
                   'id="jnlFBarEx"', 'id="jnlFSetEx"'):
            assert el in src, f"{el} is not on the page"
        # SENT, RESTORED AND PINNED -- the three ways a control
        # can look wired and not be.
        for name in ("fused_bounce_on", "fused_bar_exit",
                     "fused_settle_exit"):
            assert f"{name}:" in src, f"{name} is never sent"
        # `"fused_bounce:" in src` IS NOT ENOUGH -- "fused_bounce_on:"
        # contains it, so deleting the percentage line entirely
        # left that assertion passing. Name the control it reads.
        assert "fused_bounce: $('jnlFBouncePct')" in src, \
            "the bounce percentage is never sent"
        for el in ("jnlFBounce", "jnlFBarEx", "jnlFSetEx"):
            assert f"chk('{el}'" in src, f"{el} never comes back"
            assert f"{el}: $('{el}')" in src, f"{el} is not pinned"
        assert "set('jnlFBouncePct'" in src
        assert "jnlFBouncePct: $('jnlFBouncePct')" in src

    def test_the_percentage_offers_every_choice(self):
        src = (SRC / "web.py").read_text()
        for c in fr.BOUNCE_CHOICES:
            assert f'value="{int(round(100.0 * c))}"' in src

    def test_the_dial_reaches_the_live_rule(self, client,
                                            monkeypatch):
        """THE SWITCH, THROUGH THE ROUTE, TO THE READING. Dropping
        `bounce_on` at either gate call leaves the checkbox
        working and the database correct while the rung stays
        dead -- a box wired to nothing, which is exactly what the
        percentage selector has been since `7d961a9`."""
        from liqmap import web
        _cme(monkeypatch)
        rt = web.runtime()

        def probe(strategy, on):
            s_ = rt.journal.settings("ES")
            s_.strategy = strategy
            s_.fused_bounce_on = on
            rt.journal.save_settings(s_)
            d = client.get("/api/read",
                           params={"coin": "ES", "interval": "15m",
                                   "timeframes": "15m,5m,1m"},
                           headers=AUTH).json()
            jr = d.get("journal") or {}
            assert "error" not in jr, jr.get("error")
            return jr.get("fusedgate") or {}

        for strategy in ("together", "stackwin"):
            for on in (True, False):
                fg = probe(strategy, on)
                assert fg, f"no panel on {strategy}"
                assert fg.get("bounce_on") is on, \
                    f"the switch never reaches the panel on {strategy}"
                assert "bounce_at" in fg

    def test_the_restore_changes_nothing_until_a_box_is_ticked(self):
        """THE WHOLE POINT OF SWITCHES. With all three off, every
        reading and every exit must be byte-identical to what the
        rule did before they came back."""
        bar = dict(ofi_dir="up", tick_dir="up", open_px=7160.0,
                   high=7190.0, low=7159.0, close=7165.0,
                   bar_ts=1.0, **FULL)
        out = fr.read(**bar)
        assert out["back"] >= 0.33, "this bar is not a pullback"
        assert out["where"] == "together"
        assert fr.entry_side(out["where"], out["side"],
                             held_s=9.0)[0] == "up"
        assert fr.exit_reason("up", out["where"], bar_ts=2.0,
                              open_bar_ts=1.0) == ""


class TestTickBarsAsFilters:
    """Chris: "since we already have the 2 timeframes on side or
    the turning lets add ticks to it so we will have 1m, 5m 15m,
    233 tick, 512 tick, 500 tick, 1000 tick, 1500 tick, 2000 tick,
    2500 tick 3000 tick and 3500 tick".

    STORED NEGATIVE. These two columns already hold seconds, a
    window of seconds is never below zero, and `window_label` has
    rendered -233 as "233t" since tick windows were added. Reusing
    that beats inventing a second encoding for the same idea.
    """

    TICKS = (233, 500, 512, 1000, 1500, 2000, 2500, 3000, 3500)

    def test_every_size_he_named_has_a_builder(self):
        """`_tick_now` SNAPS AN UNKNOWN SIZE TO THE NEAREST ONE IT
        HAS, silently. A 233 with no builder would have come back
        as a real direction off the 250 bar and nothing on the
        screen would have said so."""
        from liqmap.tickbar import PER_BAR_CHOICES as P
        for n in self.TICKS:
            assert n in P, f"{n} is offered with no builder behind it"

    def test_the_snapper_takes_them_and_refuses_the_rest(self):
        from liqmap.journal import _snap_tf
        for n in self.TICKS:
            assert _snap_tf(-n) == -n, f"{n} tick was refused"
        for secs in (60, 300, 900):
            assert _snap_tf(secs) == secs
        assert _snap_tf(0) == 0
        # A SIZE WITH NO BUILDER, A POSITIVE TICK COUNT, AND JUNK.
        # Each would otherwise reach `_tick_now` and come back as
        # a direction read off some other chart.
        for bad in (-250, -4000, -1, 233, 77, 1000, "x", None, []):
            assert _snap_tf(bad) == 0, f"{bad!r} got through"

    def test_they_round_trip_and_label(self, tmp_path):
        from liqmap.journal import Journal
        j = Journal(str(tmp_path / "j.db"))
        s = j.settings("ES")
        s.fused_tf, s.fused_tf2 = -233, -3500
        back = j.save_settings(s)
        assert back.fused_tf == -233
        assert back.fused_tf2 == -3500
        d = j.settings("ES").to_dict()
        assert d["fused_tf_label"] == "233t"
        assert d["fused_tf2_label"] == "3500t"
        # AND A CLOCK CANDLE STILL LABELS AS ONE.
        s.fused_tf = 300
        assert j.save_settings(s).fused_tf == 300
        assert j.settings("ES").to_dict()["fused_tf_label"] == "5m"

    def test_a_junk_size_cannot_reach_the_rule(self, tmp_path):
        from liqmap.journal import Journal
        j = Journal(str(tmp_path / "j.db"))
        s = j.settings("ES")
        s.fused_tf = -4000
        assert j.save_settings(s).fused_tf == 0, \
            "a size with no builder was stored as a live filter"

    def test_both_selectors_offer_every_one(self):
        src = (SRC / "web.py").read_text()
        for sel in ("jnlFusedTf", "jnlFusedTf2"):
            i = src.index(f'id="{sel}"')
            box = src[i:src.index("</select>", i)]
            for n in self.TICKS:
                assert f'value="-{n}"' in box, f"{sel} has no {n} tick"
            for secs in (0, 60, 300, 900):
                assert f'value="{secs}"' in box, f"{sel} lost {secs}"

    def test_the_reader_branches_on_the_sign(self):
        """ONE FUNCTION, BOTH KINDS. A second reader for tick bars
        is a second chance to be wrong about which bar is which,
        and this engine has made that mistake with the 5-minute
        cell and the OFI bar already."""
        src = (SRC / "web.py").read_text()
        i = src.index("def _tf_read(")
        fn = src[i:src.index("\n        def ", i + 10)]
        assert "_tick_now(" in fn, "a tick filter never reads a tick bar"
        assert "_candle_now(" in fn, "the clock candles stopped working"
        assert "if n < 0:" in fn, "nothing distinguishes the two"
        # THE TICK BAR'S DIRECTION IS THE ENGINE'S, not a second
        # reading invented here. `_tick_now` is what gates 2 and 3
        # run on.
        assert "close" not in fn.split("if n < 0:")[1].split("return")[0], \
            "the tick filter reads close-against-open, not the path vote"

    def test_the_sign_reaches_every_place_that_needs_it(self):
        """SOURCE, NOT EXECUTION, AND THE REASON IS WRITTEN DOWN.

        `_tf_read` and `_tick_now` are closures inside the route
        factory, so no test can call them, and the fused engine is
        not running under the test client -- every live read here
        comes back through `_nothing`. That leaves three wirings
        that nothing else can reach, and all three survived a
        mutation run until this test existed:

            the size handed to `_tick_now` (negated, or it asks
            for a bar of minus 233 prints and gets snapped to
            whatever is nearest)
            the tick flag on each of the two filters

        Each one fails silently -- a direction off the wrong
        chart, or a countdown in the wrong units -- which is why
        they are pinned by name rather than left to the live
        payload that cannot see them.
        """
        src = (SRC / "web.py").read_text()
        i = src.index("def _fused_gate(")
        fn = src[i:src.index("\n    def ", i + 10)]
        assert "_tick_now(rt_, per_bar=-n)" in fn, \
            "the tick filter asks for a negative bar size"
        assert "tf_ticks=bool(int(tf_s or 0) < 0)" in fn, \
            "the first filter never learns it is a tick bar"
        assert "tf2_ticks=bool(int(tf2_s or 0) < 0)" in fn, \
            "the second filter never learns it is a tick bar"

    def test_the_panel_is_told_which_kind_it_is(self):
        """NOT SNIFFED OFF THE LABEL. A rule hiding in the last
        character of a name is a rule nobody can see."""
        on = fr.state(ofi_dir="up", tick_dir="up", open_px=20.0,
                      high=20.0, low=0.0, close=20.0, held_s=9.0,
                      tf_dir="up", tf_on=True, tf_name="233t",
                      tf_ticks=True, tf2_dir="up", tf2_on=True,
                      tf2_name="15m", **FULL)
        assert on["tf"]["ticks"] is True
        assert on["tf2"]["ticks"] is False

    def test_the_countdown_says_prints_for_a_tick_bar(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**TestThePanelActuallyRenders.BAR,
                       "tick_dir": "up", "ofi_dir": "up",
                       "where": "together", "agree": True, "side": "up",
                       "turning": {"dir": "up", "ticks_off": 0.0,
                                   "band": 4.0, "decides": False},
                       "tf": {"dir": "up", "on": True, "name": "233t",
                              "ticks": True, "agrees": True,
                              "left_s": 97.0, "decides": True}})
        assert "97 prints" in html
        assert "233t bar closes in" in html
        assert "shown only" in html

    def test_the_countdown_still_says_time_for_a_candle(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**TestThePanelActuallyRenders.BAR,
                       "tick_dir": "up", "ofi_dir": "up",
                       "where": "together", "agree": True, "side": "up",
                       "turning": {"dir": "up", "ticks_off": 0.0,
                                   "band": 4.0, "decides": False},
                       "tf": {"dir": "up", "on": True, "name": "5m",
                              "ticks": False, "agrees": True,
                              "left_s": 97.0, "decides": True}})
        assert "5m candle closes in" in html
        assert "prints" not in html.split("closes in")[1][:40]

    def test_a_tick_filter_gates_the_same_as_a_candle(self):
        """THE FILTER RULE IS UNCHANGED. It was never about clocks
        -- both must agree to open, price against either closes --
        so a tick bar drops straight into it."""
        blocked = fr.entry_side("together", "up", held_s=9.0,
                                tf_dir="down", tf_on=True,
                                tf_name="233t")
        assert blocked[0] == ""
        assert "233t" in blocked[1]
        opened = fr.entry_side("together", "up", held_s=9.0,
                               tf_dir="up", tf_on=True, tf_name="233t")
        assert opened[0] == "up"
        # PRICE AGAINST THE FILTER. `tick_dir` is price; the
        # filter is the other side of the comparison. Passing them
        # both "down" was my own mistake and the test caught it --
        # two agreeing readings are not an exit.
        assert "233t" in fr.exit_reason("up", "together", "up",
                                        tf_dir="down", tf_on=True,
                                        tf_name="233t")
        assert fr.exit_reason("up", "together", "up", tf_dir="up",
                              tf_on=True, tf_name="233t") == ""

    def test_the_route_takes_a_tick_filter(self, client, monkeypatch):
        _cme(monkeypatch)
        r = client.post("/api/journal/settings",
                        params={"coin": "ES", "fused_tf": -512,
                                "fused_tf2": -2500},
                        headers=AUTH).json()["settings"]
        assert r["fused_tf"] == -512
        assert r["fused_tf_label"] == "512t"
        assert r["fused_tf2"] == -2500
        assert r["fused_tf2_label"] == "2500t"
        bad = client.post("/api/journal/settings",
                          params={"coin": "ES", "fused_tf": -4000},
                          headers=AUTH).json()["settings"]
        assert bad["fused_tf"] == 0

    def test_the_live_panel_carries_it(self, client, monkeypatch):
        from liqmap import web
        _cme(monkeypatch)
        rt = web.runtime()
        s = rt.journal.settings("ES")
        s.strategy = "together"
        s.fused_tf, s.fused_tf2 = -233, 900
        rt.journal.save_settings(s)
        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH).json()
        jr = d.get("journal") or {}
        assert "error" not in jr, jr.get("error")
        fg = jr.get("fusedgate") or {}
        assert (fg.get("tf") or {}).get("ticks") is True
        assert (fg.get("tf") or {}).get("name") == "233t"
        assert (fg.get("tf2") or {}).get("ticks") is False
        assert (fg.get("tf2") or {}).get("name") == "15m"


class TestNothingGoesBlank:
    """Chris, after running it: "turning and the 2 time frames
    goes blank before deciding instead of just rolling into the
    direction".

    THE FILTERS WERE THE ONLY READINGS IN THE ENGINE WITHOUT A
    HOLD. The stack's timeframe candle has one, both two-tick legs
    have one; these two did not. A live candle sits exactly on its
    open constantly -- `_candle_now` returns "flat" there -- and
    `entry_side` reads anything that is not a side as a refusal,
    so the filter blanked and blocked the entry every time price
    touched the open, and again at every new candle.
    """

    FULL_BAR = dict(open_px=20.0, high=21.0, low=19.0, close=21.0)

    def _run(self, raws, **kw):
        prev, out = None, []
        for raw in raws:
            o = fr.state(ofi_dir="up", tick_dir="up", held_s=9.0,
                         tf_dir=raw, tf_on=True, tf_name="5m",
                         prev=prev, **self.FULL_BAR, **FULL, **kw)
            prev = o["held"]
            out.append((o["tf"]["dir"], o["entry"]))
        return out

    def test_a_flat_candle_no_longer_blocks_the_entry(self):
        """THE BUG, AS HE SAW IT. "flat" is what a candle sitting
        on its open reads as, and it used to refuse."""
        got = self._run(["up", "flat", "", "up"])
        assert [d for d, _ in got] == ["up"] * 4, \
            "the filter blanked on a tie"
        assert [e for _, e in got] == ["up"] * 4, \
            "a blank filter refused the entry"

    def test_a_real_turn_still_takes_over_immediately(self):
        """`hold_dir` ONLY FILLS A BLANK. The moment the candle
        has a body of its own it takes over -- this bridges a gap,
        it does not carry a dead candle's opinion forward."""
        got = self._run(["up", "flat", "down"])
        assert got[-1][0] == "down"
        assert got[-1][1] == "", "a filter against the side opened a trade"

    def test_the_second_filter_holds_on_its_own(self):
        """TWO FILTERS, TWO MEMORIES. Sharing one would make the
        pair agree by construction, which is the opposite of what
        a second filter is for."""
        prev, seen = None, []
        for a, b in (("up", "down"), ("flat", "flat"), ("", "")):
            o = fr.state(ofi_dir="up", tick_dir="up", held_s=9.0,
                         tf_dir=a, tf_on=True, tf_name="5m",
                         tf2_dir=b, tf2_on=True, tf2_name="15m",
                         prev=prev, **self.FULL_BAR, **FULL)
            prev = o["held"]
            seen.append((o["tf"]["dir"], o["tf2"]["dir"]))
        assert seen == [("up", "down")] * 3

    def test_the_held_filters_ride_home(self):
        """Dropped from `held` they do not error -- the hold
        silently resets on every poll and the rows blank again,
        with nothing to show for it but a filter that refuses
        entries at random."""
        o = fr.state(ofi_dir="up", tick_dir="up", held_s=9.0,
                     tf_dir="up", tf_on=True, tf2_dir="down",
                     tf2_on=True, **self.FULL_BAR, **FULL)
        assert o["held"]["tf_dir"] == "up"
        assert o["held"]["tf2_dir"] == "down"

    def test_without_the_memory_it_blanks_again(self):
        """The failure mode stated directly."""
        got = [fr.state(ofi_dir="up", tick_dir="up", held_s=9.0,
                        tf_dir=raw, tf_on=True, tf_name="5m",
                        prev=None, **self.FULL_BAR, **FULL)["tf"]["dir"]
               for raw in ("up", "flat", "")]
        assert got == ["up", "", ""], "this test no longer tests that"


class TestStrength:
    """Chris: "the direction shows but its strength dont is the
    move strong or just a micro pull back or whatever" -- and,
    asked what it should measure, "Both, as two numbers"."""

    def test_body_share_is_body_against_the_bars_own_range(self):
        # A close at one end of the range: half the bar is body.
        assert fr.body_share(20.0, 21.0, 19.0, 21.0) == pytest.approx(0.5)
        # A tiny body between two long wicks: it went nowhere.
        assert fr.body_share(20.0, 21.0, 19.0, 20.1) == pytest.approx(0.05)
        # A bar with no range has no shape to measure.
        assert fr.body_share(20.0, 20.0, 20.0, 20.0) == 0.0
        # A CLOSE OUTSIDE THE BAR'S OWN RANGE. It should not
        # happen and it does, because the high and the low are
        # read a moment apart from the close on a live bar. Two
        # and a half times the range has to come back as a full
        # body, not 250% of one.
        assert fr.body_share(20.0, 21.0, 19.0, 25.0) == 1.0
        assert fr.body_share(20.0, 21.0, 19.0, 15.0) == 1.0
        # And it never leaves 0..1, whatever arrives.
        for args in ((20.0, 19.0, 21.0, 25.0), (0, 0, 0, 0),
                     ("a", "b", "c", "d"), (None, None, None, None)):
            v = fr.body_share(*args)
            assert 0.0 <= v <= 1.0

    def test_body_share_is_scale_free(self):
        """AGAINST THE BAR'S OWN RANGE, so it means the same thing
        on a 1-minute candle and a 3,500-tick bar."""
        small = fr.body_share(20.0, 20.5, 19.5, 20.5)
        big = fr.body_share(7000.0, 7050.0, 6950.0, 7050.0)
        assert small == pytest.approx(big)

    def test_travel_counts_up_from_the_last_flip(self):
        st, seen = {}, []
        for c, d in ((20.0, "up"), (20.5, "up"), (21.0, "up")):
            st = fr.leg_travel(c, d, st)
            seen.append(st["ticks"])
        assert seen == [0.0, 2.0, 4.0]

    def test_a_flip_restarts_the_travel(self):
        st = {}
        for c, d in ((20.0, "up"), (21.0, "up")):
            st = fr.leg_travel(c, d, st)
        assert st["ticks"] == 4.0
        st = fr.leg_travel(20.75, "down", st)
        assert st["ticks"] == 0.0, "the new leg inherited the old one"
        st = fr.leg_travel(20.0, "down", st)
        assert st["ticks"] == 3.0

    def test_travel_is_signed_along_the_leg(self):
        """A MOVE GOING THE WRONG WAY IS NOT STRENGTH. An absolute
        distance read a tick of progress on an UP leg that had
        fallen a tick BELOW where it turned."""
        st = fr.leg_travel(20.0, "up", {})
        st = fr.leg_travel(19.75, "up", st)
        assert st["ticks"] == 0.0
        st = fr.leg_travel(19.0, "up", st)
        assert st["ticks"] == 0.0

    def test_the_turning_rows_travel_is_signed_too(self):
        """THE SAME ARITHMETIC IN THE SAME MODULE, TWICE. An
        absolute distance here read progress on an UP leg that had
        fallen below where it turned -- and `leg_travel` being
        correct says nothing about this one."""
        st = {}
        for c in (7180.25, 7180.50, 7180.75, 7181.00):
            st = fr.turn_state(c, 4.0, st, open_px=7180.0, bar_ts=1.0)
        assert st["dir"] == "up"
        back = fr.turn_state(7180.75, 4.0, st, open_px=7180.0,
                             bar_ts=1.0)
        assert back["dir"] == "up"
        assert back["travel"] == 0.0, \
            "a move going the wrong way counted as strength"

    def test_the_turning_row_carries_its_legs_size(self):
        """`ticks_off` COUNTS DOWN TO THE NEXT FLIP, travel counts
        up from the last one. A row showing only the first says a
        turn is near without saying whether what it would be
        turning out of was worth anything."""
        st = {}
        for c in (7180.25, 7180.50, 7180.75, 7181.00, 7182.00):
            st = fr.turn_state(c, 4.0, st, open_px=7180.0, bar_ts=1.0)
        assert st["dir"] == "up"
        assert st["travel"] == 4.0
        assert st["ticks_off"] == 0.0
        out = fr.read(ofi_dir="up", tick_dir="up", open_px=7180.0,
                      high=7182.0, low=7180.0, close=7182.0,
                      bar_ts=1.0, **FULL)
        assert "travel" in out["turning"]

    def test_both_numbers_reach_the_filter_rows(self):
        o = fr.state(ofi_dir="up", tick_dir="up", open_px=20.0,
                     high=21.0, low=19.0, close=21.0, held_s=9.0,
                     tf_dir="up", tf_on=True, tf_body=0.5,
                     tf_travel=12.0, tf2_dir="up", tf2_on=True,
                     tf2_body=0.05, tf2_travel=1.0, **FULL)
        assert o["tf"]["body"] == pytest.approx(0.5)
        assert o["tf"]["travel"] == pytest.approx(12.0)
        assert o["tf2"]["body"] == pytest.approx(0.05)
        assert o["tf2"]["travel"] == pytest.approx(1.0)

    def test_the_strength_is_measured_on_the_held_direction(self):
        """NOT THE RAW ONE. A live candle sits on its open
        constantly; measuring the leg off a reading that blanks
        there would restart the move's length on an absence --
        the same mistake the blanking filter was making one level
        up."""
        src = (SRC / "web.py").read_text()
        i = src.index("def _strength(")
        fn = src[i:src.index("\n        _tf_body", i)]
        assert "hold_dir(" in fn, "the leg is measured off a raw reading"
        assert "leg_travel(" in fn and "body_share(" in fn

    def test_the_computed_numbers_reach_the_rule(self):
        """SOURCE, FOR THE REASON WRITTEN IN THE SIGN TEST ABOVE:
        the fused engine does not run under the test client, so
        every live read comes back through `_nothing`, which
        supplies its own zeros. Hard-coding zeros at the call site
        is therefore invisible to the live payload -- and it
        survived a mutation run until this existed."""
        src = (SRC / "web.py").read_text()
        i = src.index("def _fused_gate(")
        fn = src[i:src.index("\n    def ", i + 10)]
        assert 'tf_body=_tf_body, tf_travel=_tf_leg["ticks"],' in fn, \
            "the first filter's strength is not the one measured"
        assert 'tf2_body=_tf2_body, tf2_travel=_tf2_leg["ticks"],' in fn, \
            "the second filter's strength is not the one measured"

    def test_the_leg_memories_ride_home(self):
        src = (SRC / "web.py").read_text()
        assert 'out["held"]["tf_leg"] = _tf_leg' in src, \
            "the travel restarts on every poll and reads 0 forever"
        assert 'out["held"]["tf2_leg"] = _tf2_leg' in src

    def test_the_panel_draws_both(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**TestThePanelActuallyRenders.BAR,
                       "tick_dir": "up", "ofi_dir": "up",
                       "where": "together", "agree": True, "side": "up",
                       "turning": {"dir": "up", "ticks_off": 1.0,
                                   "travel": 18.0, "band": 4.0,
                                   "decides": False},
                       "tf": {"dir": "up", "on": True, "name": "5m",
                              "ticks": False, "body": 0.72,
                              "travel": 23.0, "agrees": True,
                              "left_s": 97.0, "decides": True}})
        assert "72% body" in html
        assert "carried 23 ticks" in html
        assert "this leg has carried <b>18 ticks</b>" in html

    def test_a_micro_pullback_reads_as_one(self, page_js):
        """THE QUESTION HE ASKED, ON SCREEN. Same direction, two
        very different moves, and the panel has to tell them
        apart."""
        from tests.test_stack_strip import run_js

        def draw(body, travel):
            return run_js(page_js, "fusedGate",
                          {**TestThePanelActuallyRenders.BAR,
                           "tick_dir": "up", "ofi_dir": "up",
                           "where": "together", "agree": True,
                           "side": "up",
                           "turning": {"dir": "up", "ticks_off": 0.0,
                                       "travel": travel, "band": 4.0,
                                       "decides": False},
                           "tf": {"dir": "up", "on": True,
                                  "name": "5m", "ticks": False,
                                  "body": body, "travel": travel,
                                  "agrees": True, "left_s": None,
                                  "decides": True}})
        strong, weak = draw(0.85, 40.0), draw(0.04, 1.0)
        assert "85% body" in strong and "carried 40 ticks" in strong
        assert "4% body" in weak and "carried 1 ticks" in weak
        assert strong != weak

    def test_the_live_panel_carries_both(self, client, monkeypatch):
        from liqmap import web
        _cme(monkeypatch)
        rt = web.runtime()
        s = rt.journal.settings("ES")
        s.strategy = "together"
        s.fused_tf, s.fused_tf2 = 300, -233
        rt.journal.save_settings(s)
        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH).json()
        jr = d.get("journal") or {}
        assert "error" not in jr, jr.get("error")
        fg = jr.get("fusedgate") or {}
        for row in ("tf", "tf2"):
            assert "body" in (fg.get(row) or {}), f"{row} has no body"
            assert "travel" in (fg.get(row) or {}), f"{row} has no travel"
        assert "travel" in (fg.get("turning") or {})


class TestTheLagReadout:
    """Chris: "its like the move has already happened on iron beam
    by the time it shows up on chris chart so the entries are
    always well behind iron beam".

    NOTHING MEASURED IT. Asked which to do first he said "Measure
    the feed first" -- because about fifteen seconds of the delay
    is the rule's own (a ~9s settle gate plus a 5s hold) and until
    the feed leg has a number there is no way to tell a slow feed
    from a rule that waits by design.
    """

    @staticmethod
    def _tape(samples=(), seen=None):
        import time
        from liqmap.cme_tape import Tape
        t = Tape()
        for d in samples:
            t.lag_samples.append(d)
        if seen is not None:
            t.last_seen = time.time() - seen
        return t

    def test_no_prints_is_not_no_lag(self):
        """A MISSING READING AND A ZERO ARE DIFFERENT THINGS. The
        first means the feed is not running, which is the more
        alarming of the two, and the panel's guard cannot tell
        them apart unless the payload does."""
        r = self._tape().lag()
        assert r["feed_ms"] is None
        assert r["since_ms"] is None
        assert r["n"] == 0
        assert r["why"], "nothing says why there is no reading"

    def test_it_is_a_median_not_a_mean(self):
        """One print that arrived during a pause is not the feed
        being slow, and a mean lets it say so for the next two
        hundred samples."""
        clean = self._tape((0.040, 0.050, 0.060)).lag()
        spiked = self._tape((0.040, 0.050, 0.060, 0.050, 4.0)).lag()
        assert clean["feed_ms"] == pytest.approx(50.0)
        assert spiked["feed_ms"] == pytest.approx(50.0), \
            "one outlier moved the reading"

    def test_it_reports_in_milliseconds(self):
        r = self._tape((0.0513,)).lag()
        assert r["feed_ms"] == pytest.approx(51.3, abs=0.2)

    def test_an_even_count_averages_the_middle_two(self):
        r = self._tape((0.040, 0.060)).lag()
        assert r["feed_ms"] == pytest.approx(50.0)

    def test_the_sample_count_is_reported(self):
        """A median of three is not a measurement and the panel
        has to be able to say so."""
        assert self._tape((0.05,) * 7).lag()["n"] == 7

    def test_the_window_is_bounded(self):
        """An unbounded list of every print of the session is a
        leak and a median that stops responding to now."""
        t = self._tape((0.05,) * 5000)
        assert len(t.lag_samples) <= 256

    def test_since_is_measured_on_one_clock(self):
        """`feed_ms` spans the exchange's clock and this one;
        `since_ms` starts and stops here, so a drifting container
        cannot move it."""
        r = self._tape((0.05,), seen=0.25).lag()
        assert r["since_ms"] == pytest.approx(250.0, abs=60.0)

    def test_the_two_clock_caveat_is_carried_not_buried(self):
        """A lag figure nobody can interrogate is worse than none,
        because it gets believed."""
        r = self._tape((0.05,)).lag()
        assert "clock" in r["why"].lower()

    @staticmethod
    def _print_rec(ts, px=7200.0, sz=3):
        """One Databento trade record, as the tape expects it."""
        o = type("R", (), {})()
        o.ts_event = int(ts * 1e9)
        o.sequence, o.instrument_id = 1, 42
        o.action, o.side = "T", "B"
        o.price, o.size = int(px * 1e9), sz
        lv = []
        for k in range(2):
            L = type("L", (), {})()
            L.bid_px, L.bid_sz, L.bid_ct = int((px - k * 0.25) * 1e9), 10, 1
            L.ask_px, L.ask_sz, L.ask_ct = int((px + k * 0.25) * 1e9), 10, 1
            lv.append(L)
        o.levels = lv
        return o

    def test_a_real_print_lands_a_sample(self):
        """EXECUTED, NOT SCANNED. This read the source for
        `lag_samples.append` near `self.trades += 1`, and wrapping
        that line in `if False:` left the text exactly where the
        test was looking. The append has to actually happen."""
        import time
        from liqmap.cme_tape import Tape
        t = Tape()
        before = len(t.lag_samples)
        t.offer(self._print_rec(time.time() - 0.080))
        assert len(t.lag_samples) > before, \
            "a print arrived and no lag sample was taken"
        r = t.lag()
        assert r["n"] >= 1
        assert r["feed_ms"] == pytest.approx(80.0, abs=50.0), \
            "the sample is not the exchange-stamp-to-arrival gap"
        assert r["since_ms"] is not None, "last_seen was never set"

    def test_the_sample_is_taken_at_ingest_not_on_request(self):
        """Sampling when a route asks measures how long ago
        somebody asked, which is a number about the browser."""
        import time
        from liqmap.cme_tape import Tape
        t = Tape()
        t.offer(self._print_rec(time.time() - 0.080))
        first = t.lag()["n"]
        for _ in range(5):
            t.lag()
        assert t.lag()["n"] == first, \
            "asking for the reading added samples to it"

    def test_the_panel_carries_it_on_every_path(self, client,
                                                monkeypatch):
        """INCLUDING THE ENGINE-NOT-RUNNING PATH. That is exactly
        when you most want to know the feed is dead, and it is the
        path that has shipped a short payload four times."""
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
        lag = (jr.get("fusedgate") or {}).get("lag")
        assert lag is not None, "the panel has no lag reading at all"
        for k in ("feed_ms", "since_ms", "n"):
            assert k in lag, f"lag is missing {k}"

    def test_the_panel_draws_all_three(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**TestThePanelActuallyRenders.BAR,
                       "tick_dir": "up", "ofi_dir": "up",
                       "where": "together", "agree": True, "side": "up",
                       "turning": {"dir": "up", "ticks_off": 0.0,
                                   "travel": 0.0, "band": 4.0,
                                   "decides": False},
                       "lag": {"feed_ms": 51.0, "since_ms": 120.0,
                               "n": 256, "why": "two clocks"}})
        assert "behind the exchange by" in html
        assert "51 ms" in html
        assert "120 ms" in html
        assert "256 prints" in html
        assert "two clocks" in html, \
            "the caveat is not on screen beside the number"

    def test_the_panel_says_when_there_is_no_reading(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**TestThePanelActuallyRenders.BAR,
                       "tick_dir": "up", "ofi_dir": "up",
                       "where": "together", "agree": True, "side": "up",
                       "turning": {"dir": "up", "ticks_off": 0.0,
                                   "travel": 0.0, "band": 4.0,
                                   "decides": False},
                       "lag": {"feed_ms": None, "since_ms": None,
                               "n": 0, "why": "no tape is running"}})
        assert "no lag reading" in html
        assert "no tape is running" in html

    def test_the_browser_measures_its_own_round_trip(self):
        """THE ONLY NUMBER WITH ONE CLOCK IN IT END TO END. It
        starts and stops in the browser, so whatever it says, no
        other machine's clock is involved."""
        src = (SRC / "web.py").read_text()
        assert "_readRtt" in src, "no round trip is measured"
        assert "performance.now" in src, \
            "the round trip is measured with a wall clock"
        i = src.index("let _readRtt")
        seg = src[i:i + 700]
        assert "performance.now" in seg
        # THE ASSIGNMENT INSIDE THE TRY, NOT THE DECLARATION.
        # `_readRtt =` matched `let _readRtt = null` and the test
        # passed with the measurement deleted.
        assert "await loadRead(); _readRtt =" in seg, \
            "the round trip is declared but never measured"

    def test_the_building_path_carries_it_too(self):
        """SOURCE, because the fused engine does not run under the
        test client -- every live read here comes back through
        `_nothing`, so dropping the lag from the path that DOES
        have a bar is invisible to the live payload. It survived a
        mutation run until this existed."""
        src = (SRC / "web.py").read_text()
        i = src.index("def _fused_gate(")
        fn = src[i:src.index("\n    def ", i + 10)]
        assert fn.count("_tape_lag(rt_)") >= 2, \
            "only one of the two payload paths carries the lag"

    def test_it_decides_nothing(self):
        """A MEASUREMENT, NOT A GATE. Nothing is blocked or opened
        on how old the reading is -- that would be a rule nobody
        asked for, bolted onto a number added to answer a
        question."""
        import inspect
        for fn in (fr.entry_side, fr.exit_reason):
            ps = inspect.signature(fn).parameters
            for bad in ("lag", "feed_ms", "since_ms", "age"):
                assert bad not in ps, f"{fn.__name__} reads {bad}"


class TestTheAutoSizedPair:
    """Chris sent ChrisChart's "10000x4000" bar beside a 15-minute
    Ironbeam candle. Identical high and low -- 7,189.25 and
    7,176.00 -- because the fused bar closed on its 900s cap every
    time. He chose "Size the pair from the tape"."""

    def test_the_setting_round_trips(self, tmp_path):
        from liqmap.fusedbar import AUTO_CHOICES
        from liqmap.journal import Journal
        j = Journal(str(tmp_path / "j.db"))
        s = j.settings("ES")
        assert s.fused_auto_s == 0, "it switched itself on"
        s.fused_auto_s = 300
        assert j.save_settings(s).fused_auto_s == 300
        assert j.settings("ES").to_dict()["fused_auto_s"] == 300
        s.fused_auto_s = 77
        assert j.save_settings(s).fused_auto_s == 0
        for c in AUTO_CHOICES:
            s.fused_auto_s = c
            assert j.save_settings(s).fused_auto_s == c

    def test_the_route_takes_it(self, client, monkeypatch):
        _cme(monkeypatch)
        r = client.post("/api/journal/settings",
                        params={"coin": "ES", "fused_auto_s": 180},
                        headers=AUTH).json()["settings"]
        assert r["fused_auto_s"] == 180
        bad = client.post("/api/journal/settings",
                          params={"coin": "ES", "fused_auto_s": 9999},
                          headers=AUTH).json()["settings"]
        assert bad["fused_auto_s"] == 0

    def test_the_selector_is_on_the_page(self):
        from liqmap.fusedbar import AUTO_CHOICES
        src = (SRC / "web.py").read_text()
        assert 'id="jnlFusedAuto"' in src
        i = src.index('id="jnlFusedAuto"')
        box = src[i:src.index("</select>", i)]
        for c in AUTO_CHOICES:
            assert f'value="{int(c)}"' in box, f"no {c}s option"
        assert "fused_auto_s: $('jnlFusedAuto')" in src, "never sent"
        assert "set('jnlFusedAuto'" in src, "never restored"
        assert "jnlFusedAuto: $('jnlFusedAuto')" in src, "not pinned"

    def test_an_auto_pair_escapes_the_snap_to_the_fixed_list(self):
        """AN AUTO PAIR IS NOT IN `PAIRS` BY CONSTRUCTION. The
        snap below it would drag 800x1100 back to 1000x500 and
        undo the whole thing, so the order of those two is the
        feature."""
        src = (SRC / "web.py").read_text()
        i = src.index("def _fused_gate(")
        fn = src[i:src.index("\n    def ", i + 10)]
        a = fn.index('if _auto.get("pair"):')
        b = fn.index("elif key not in PAIRS:")
        assert a < b, "the auto pair is snapped to the fixed list"

    def test_the_name_is_bound_before_the_closure_reads_it(self):
        """THE FIFTH TIME THIS SHAPE HAS COME UP HERE. `_nothing`
        references `_auto`; with the assignment below the closure
        it worked only because every call happened to come after
        it, and the previous four of these raised free-variable
        NameErrors into `journal.error` where nothing was
        looking."""
        src = (SRC / "web.py").read_text()
        i = src.index("def _fused_gate(")
        fn = src[i:src.index("\n    def ", i + 10)]
        assert fn.index("_auto = _auto_pair(") < fn.index("def _nothing("), \
            "the closure is defined before the name it reads"

    def test_it_is_rate_limited_and_gated(self):
        """Every change is a new builder and an empty chart."""
        src = (SRC / "web.py").read_text()
        assert "_AUTO_EVERY_S" in src and "_AUTO_MOVE" in src
        i = src.index("def _auto_pair(")
        fn = src[i:src.index("\n    def ", i + 10)]
        assert "_AUTO_EVERY_S" in fn, "it re-sizes on every poll"
        assert "moved < _AUTO_MOVE" in fn, "any wobble rebuilds"
        # THE CALLS, NOT THE NAMES. `"correct_for" in fn` passed
        # with the call replaced by `size_for`, because the
        # docstring above it mentions `correct_for` -- a test
        # reading prose instead of code.
        assert "_fb2.correct_for(src.reach()" in fn, \
            "a running pair is re-seeded instead of corrected"
        assert "_fb2.size_for(seed.reach()" in fn, \
            "a pair with no history is never seeded"
        # AND THE BUILDER IS ACTUALLY MADE. Without this the pair
        # is computed, reported on the panel, and nothing is ever
        # built for it -- the gate then finds no builder and the
        # whole feature reads as "no builder for that pair".
        assert "if new not in books:" in fn, \
            "the builder for a new pair is never created"
        assert "books[new] = _fb2.FusedBars(" in fn

    def test_off_changes_nothing(self, client, monkeypatch):
        """OFF IS THE DEFAULT and off has to mean the fixed pair
        behaves exactly as it did."""
        from liqmap import web
        _cme(monkeypatch)
        rt = web.runtime()
        st = rt.journal.settings("ES")
        st.strategy = "together"
        st.fused_auto_s = 0
        rt.journal.save_settings(st)
        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH).json()
        jr = d.get("journal") or {}
        assert "error" not in jr, jr.get("error")
        auto = (jr.get("fusedgate") or {}).get("auto") or {}
        assert auto.get("on") is False
        assert auto.get("pair") is None

    def test_the_live_panel_carries_the_reason(self, client,
                                               monkeypatch):
        """A PAIR THAT CHANGES UNDER YOU WITH NO EXPLANATION is
        worse than one that is simply wrong -- you cannot tell the
        change from a bug."""
        from liqmap import web
        _cme(monkeypatch)
        rt = web.runtime()
        st = rt.journal.settings("ES")
        st.strategy = "together"
        st.fused_auto_s = 300
        rt.journal.save_settings(st)
        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH).json()
        jr = d.get("journal") or {}
        assert "error" not in jr, jr.get("error")
        auto = (jr.get("fusedgate") or {}).get("auto") or {}
        assert auto.get("on") is True
        assert auto.get("target_s") == 300.0
        assert "why" in auto

    def test_the_panel_draws_it(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**TestThePanelActuallyRenders.BAR,
                       "tick_dir": "up", "ofi_dir": "up",
                       "where": "together", "agree": True, "side": "up",
                       "turning": {"dir": "up", "ticks_off": 0.0,
                                   "travel": 0.0, "band": 4.0,
                                   "decides": False},
                       "auto": {"on": True, "target_s": 300.0,
                                "pair": [800, 1100],
                                "why": "from bars that ran the full 900s cap"}})
        assert "sized from this tape" in html
        assert "~5 min" in html
        assert "800t" in html and "1,100" in html
        assert "900s cap" in html

    def test_the_panel_says_when_it_has_not_sized_yet(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**TestThePanelActuallyRenders.BAR,
                       "tick_dir": "up", "ofi_dir": "up",
                       "where": "together", "agree": True, "side": "up",
                       "turning": {"dir": "", "ticks_off": 0.0,
                                   "travel": 0.0, "band": 4.0,
                                   "decides": False},
                       "auto": {"on": True, "target_s": 300.0,
                                "pair": None,
                                "why": "nothing has closed yet"}})
        assert "not sized yet" in html
        assert "nothing has closed yet" in html


class TestTheRowsReadTwoSeparateCharts:
    """Chris: "i have 2 charts on chris charts one is a OFI chart
    that ranges from 125,250, 500, 1000 and 2000 contracts and the
    other chart is my normal chart ... i want you to make the outer
    OFI actually read the OFI chart and the inner tick read the
    actual tick chart itself 2 different reading from 2 individual
    charts what this does is give a better reading from 2 different
    sources for better trades".

    AND HE IS RIGHT, AGAINST WHAT THIS FILE USED TO ARGUE. The
    fused bar's two legs covering the same prints was defended as a
    virtue -- agreement about one event rather than two. Two legs
    sharing a bar boundary are not independent evidence; they are
    one event described twice.
    """

    def test_the_three_settings_round_trip(self, tmp_path):
        from liqmap.journal import Journal
        from liqmap.ofibar import LADDER
        j = Journal(str(tmp_path / "j.db"))
        s = j.settings("ES")
        assert s.fused_src == 0, "it switched itself on"
        s.fused_src, s.fused_ofi_bar, s.fused_tick_bar = 1, 2000, 233
        back = j.save_settings(s)
        assert (back.fused_src, back.fused_ofi_bar,
                back.fused_tick_bar) == (1, 2000, 233)
        d = j.settings("ES").to_dict()
        assert d["fused_ofi_bar"] == 2000
        for n in LADDER:
            s.fused_ofi_bar = n
            assert j.save_settings(s).fused_ofi_bar == n

    def test_a_size_with_no_builder_cannot_be_saved(self, tmp_path):
        """`_tick_now` AND THE OFI LADDER BOTH SNAP SILENTLY. A
        size with no builder would come back as a real direction
        read off some other chart, and nothing on screen would say
        so."""
        from liqmap.journal import Journal
        j = Journal(str(tmp_path / "j.db"))
        s = j.settings("ES")
        s.fused_ofi_bar, s.fused_tick_bar = 999, 777
        back = j.save_settings(s)
        assert back.fused_ofi_bar == 500
        assert back.fused_tick_bar == 1000

    def test_the_route_takes_them(self, client, monkeypatch):
        _cme(monkeypatch)
        r = client.post("/api/journal/settings",
                        params={"coin": "ES", "fused_src": 1,
                                "fused_ofi_bar": 125,
                                "fused_tick_bar": 2000},
                        headers=AUTH).json()["settings"]
        assert r["fused_src"] == 1
        assert r["fused_ofi_bar"] == 125
        assert r["fused_tick_bar"] == 2000

    def test_the_outer_row_reads_the_ofi_charts_colour(self):
        """`bar_read["read"]` IS THE COLOUR THE CANDLE WILL BE --
        the accumulator's sign past its deadband. NOT price against
        the open; those two disagree often and pairing them refused
        nearly every trade once already."""
        src = (SRC / "web.py").read_text()
        i = src.index("def _ofi_bar_read(")
        fn = src[i:src.index("\n    def ", i + 10)]
        assert "b.bar_read" in fn, "it reads something else"
        assert "ofibars" in fn

    def test_an_unbuilt_ofi_size_returns_nothing_not_the_nearest(self):
        """A ROW QUIETLY READING A DIFFERENT CHART IS WORSE THAN A
        BLANK ROW."""
        src = (SRC / "web.py").read_text()
        i = src.index("def _ofi_bar_read(")
        fn = src[i:src.index("\n    def ", i + 10)]
        code = "\n".join(ln for ln in fn.splitlines()
                         if not ln.strip().startswith("#"))
        assert "min(" not in code and "max(bars)" not in code, \
            "it falls back to the nearest built size"
        assert "if b is None:" in code and "return {}" in code

    def test_the_gate_swaps_only_the_two_directions(self):
        """THE BAR STAYS THE FUSED BAR. It is what the panel draws,
        what the settle gate counts and what the turning row
        measures inside -- swapping it too would change five things
        at once and leave nothing to compare against."""
        src = (SRC / "web.py").read_text()
        i = src.index("def _fused_gate(")
        fn = src[i:src.index("\n    def ", i + 10)]
        assert "ofi_dir=_ofi_row, tick_dir=_tick_row," in fn
        # The bar's own prices still come from `cur`.
        assert "open_px=cur.open, high=cur.high, low=cur.low," in fn
        assert "prints=cur.prints" in fn

    def test_a_chart_with_nothing_to_say_blanks_the_row(self):
        """NOT A FALLBACK TO THE FUSED BAR'S OWN LEG. A row that
        quietly changes source when its own is not ready cannot be
        trusted on the one bar where it matters."""
        src = (SRC / "web.py").read_text()
        i = src.index("def _fused_gate(")
        fn = src[i:src.index("\n    def ", i + 10)]
        j = fn.index("if _src:")
        seg = fn[j:fn.index("out = _fr.state(", j)]
        assert 'str(_ob.get("read") or "")' in seg
        assert "cur.ofi_dir" not in seg, \
            "it falls back to the fused bar's leg"
        assert "cur.tick_dir" not in seg

    def test_each_size_reaches_the_chart_it_names(self):
        """SOURCE, AND THE REASON IS THE USUAL ONE: the fused
        engine does not run under the test client, so every live
        read comes back through `_nothing` and the BUILDING path
        is invisible to the payload. Dropping either size there
        leaves the selectors working, the database correct and the
        panel honest, while both rows read whatever chart happened
        to be default. All four of these survived a mutation run
        until this test existed."""
        src = (SRC / "web.py").read_text()
        i = src.index("def _fused_gate(")
        fn = src[i:src.index("\n    def ", i + 10)]
        assert "_ofi_bar_read(rt_, int(ofi_bar or 0))" in fn, \
            "the outer row does not read the size it was given"
        assert "_tick_now(rt_, per_bar=int(tick_bar or 0))" in fn, \
            "the inner row does not read the size it was given"
        assert '"src": {"two_charts": bool(_src),' in fn, \
            "the building path does not report its own source"

    def test_the_shown_only_call_site_passes_all_three(self):
        """OFF ITS OWN STRATEGY THE PANEL IS COMPUTED FRESH, and
        that call takes every setting positionally -- three
        constants there would make the panel read different charts
        from the rule while looking identical."""
        src = (SRC / "web.py").read_text()
        i = src.index("_shown = _fused_gate(")
        seg = src[i:i + 900]
        assert 'getattr(s, "fused_src"' in seg
        assert 'getattr(s, "fused_ofi_bar"' in seg
        assert 'getattr(s, "fused_tick_bar"' in seg

    def test_off_reads_the_fused_bars_own_legs(self):
        """THE BOOK LEG GOES THROUGH THE SAME BAND. `ofi_dir` on a
        FusedBar is a bare sign of a resetting accumulator, which
        is what was chattering."""
        src = (SRC / "web.py").read_text()
        i = src.index("def _fused_gate(")
        fn = src[i:src.index("\n    def ", i + 10)]
        assert "_fr.ofi_side(cur.net_ofi, bld.ofi_target," in fn
        assert "_tick_row = cur.tick_dir" in fn
        assert "_ofi_row, _tick_row = cur.ofi_dir, cur.tick_dir" not in fn, \
            "the book leg is the bare sign again"

    def test_the_live_panel_says_where_the_rows_came_from(
            self, client, monkeypatch):
        """TWO ROWS THAT LOOK IDENTICAL AND MEAN DIFFERENT THINGS
        is the worst state this box can be in, and it has been in
        it."""
        from liqmap import web
        _cme(monkeypatch)
        rt = web.runtime()
        st = rt.journal.settings("ES")
        st.strategy = "together"
        st.fused_src, st.fused_ofi_bar, st.fused_tick_bar = 1, 1000, 2000
        rt.journal.save_settings(st)
        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH).json()
        jr = d.get("journal") or {}
        assert "error" not in jr, jr.get("error")
        s_ = (jr.get("fusedgate") or {}).get("src") or {}
        assert s_.get("two_charts") is True
        assert s_.get("ofi_bar") == 1000
        assert s_.get("tick_bar") == 2000

    def test_off_says_so_on_every_path(self, client, monkeypatch):
        from liqmap import web
        _cme(monkeypatch)
        rt = web.runtime()
        st = rt.journal.settings("ES")
        st.strategy = "together"
        st.fused_src = 0
        rt.journal.save_settings(st)
        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH).json()
        s_ = ((d.get("journal") or {}).get("fusedgate") or {}).get("src")
        assert s_ is not None, "the path has no src block at all"
        assert s_.get("two_charts") is False

    def test_the_selectors_are_on_the_page(self):
        from liqmap.ofibar import LADDER
        src = (SRC / "web.py").read_text()
        for el in ('id="jnlFusedSrc"', 'id="jnlFusedOfiBar"',
                   'id="jnlFusedTickBar"'):
            assert el in src, f"{el} is not on the page"
        i = src.index('id="jnlFusedOfiBar"')
        box = src[i:src.index("</select>", i)]
        for n in LADDER:
            assert f'value="{n}"' in box, f"no {n}-contract option"
        for name in ("fused_src", "fused_ofi_bar", "fused_tick_bar"):
            assert f"{name}: $('jnlFused" in src, f"{name} is never sent"
        for el in ("jnlFusedSrc", "jnlFusedOfiBar", "jnlFusedTickBar"):
            assert f"set('{el}'" in src, f"{el} never comes back"
            assert f"{el}: $('{el}')" in src, f"{el} is not pinned"

    def test_the_panel_draws_the_source(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**TestThePanelActuallyRenders.BAR,
                       "tick_dir": "up", "ofi_dir": "up",
                       "where": "together", "agree": True, "side": "up",
                       "turning": {"dir": "", "ticks_off": 0.0,
                                   "travel": 0.0, "band": 4.0,
                                   "decides": False},
                       "src": {"two_charts": True, "ofi_bar": 1000,
                               "tick_bar": 2000,
                               "say": "the 1,000-contract OFI chart and "
                                      "the 2,000-print tick chart"}})
        assert "two charts" in html
        assert "1,000-contract OFI chart" in html
        assert "2,000-print tick chart" in html

    def test_the_panel_is_quiet_when_it_is_off(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**TestThePanelActuallyRenders.BAR,
                       "tick_dir": "up", "ofi_dir": "up",
                       "where": "together", "agree": True, "side": "up",
                       "turning": {"dir": "", "ticks_off": 0.0,
                                   "travel": 0.0, "band": 4.0,
                                   "decides": False},
                       "src": {"two_charts": False, "ofi_bar": 0,
                               "tick_bar": 0, "say": ""}})
        assert "two charts" not in html


class TestAChartSourcedRowFollowsItsChart:
    """Chris, after running the rows at 2,000 contracts and 1,500
    ticks: "i noticed that the OFI 2000 contracts can be up and the
    2 charts reading will blank out or still be showing down".

    TWO CAUSES, ONE OF THEM A BUG.

        `bar_read["read"]` RETURNS "flat", NOT "". The OFI chart
        calls a side only once its accumulator clears a CALL
        LEVEL, which is a separate number from the target that
        closes the bar. Flat is a real third state there, not a
        tie.

        AND THE HOLD TURNED THAT INTO A STALE DIRECTION. `hold_dir`
        exists because the fused bar's own price leg sits exactly
        on its open constantly and an absence must not read as a
        reversal. Applied to an independent chart it carried the
        PREVIOUS direction through that chart's flat -- so the row
        said DOWN while the chart he was looking at was UP.
    """

    FULL_BAR = dict(open_px=20.0, high=21.0, low=19.0, close=21.0)

    def _walk(self, raws, hold_legs):
        prev, out = None, []
        for raw in raws:
            o = fr.read(ofi_dir=raw, tick_dir="up", bar_ts=1.0,
                        prev=prev, hold_legs=hold_legs,
                        **self.FULL_BAR, **FULL)
            prev = o["held"]
            out.append(o["ofi_dir"])
        return out

    def test_the_old_hold_showed_a_stale_direction_through_flat(self):
        """The bug, stated as the behaviour it was."""
        got = self._walk(["down", "flat", "flat"], hold_legs=True)
        assert got == ["down", "down", "down"], \
            "this test no longer describes the hold"

    def test_a_chart_sourced_row_goes_flat_when_its_chart_does(self):
        got = self._walk(["down", "flat", "up"], hold_legs=False)
        assert got[1] != "down", "it held a direction the chart dropped"
        assert got[2] == "up"

    def test_it_follows_the_chart_straight_up_from_flat(self):
        got = self._walk(["flat", "flat", "up", "up"], hold_legs=False)
        assert got[-1] == "up"
        assert got[0] not in ("up", "down")

    def test_the_price_leg_is_not_held_either(self):
        """BOTH ROWS COME FROM CHARTS in this mode, so both holds
        have to go -- holding one and not the other is the same
        bug surviving on the other side of the panel."""
        prev, seen = None, []
        for raw in ("up", "", "down"):
            o = fr.read(ofi_dir="up", tick_dir=raw, bar_ts=1.0,
                        prev=prev, hold_legs=False,
                        **self.FULL_BAR, **FULL)
            prev = o["held"]
            seen.append(o["tick_dir"])
        assert seen[1] != "up", "the price row held through a blank"

    def test_the_fused_bars_own_legs_are_still_held(self):
        """OFF ITS OWN SOURCE THE HOLD IS RIGHT and must not be
        collateral damage: a forming fused bar sits exactly on its
        open constantly, and an absence there IS a tie."""
        prev, seen = None, []
        for raw in ("up", "", "up"):
            o = fr.read(ofi_dir="up", tick_dir=raw, bar_ts=1.0,
                        prev=prev, **self.FULL_BAR, **FULL)
            prev = o["held"]
            seen.append(o["tick_dir"])
        assert seen == ["up", "up", "up"], \
            "the tie hold was removed from the fused bar too"

    def test_state_passes_the_flag_down_to_read(self):
        """`state` IS WHAT THE ROUTE CALLS, not `read`. A flag that
        stops at `state`'s door leaves every test above passing
        and the live panel holding stale directions exactly as
        before."""
        prev, seen = None, []
        for raw in ("down", "flat", "flat"):
            o = fr.state(ofi_dir=raw, tick_dir="up", held_s=9.0,
                         bar_ts=1.0, prev=prev, hold_legs=False,
                         **self.FULL_BAR, **FULL)
            prev = o["held"]
            seen.append(o["ofi_dir"])
        assert seen[1] != "down", \
            "state accepted the flag and did not pass it on"
        held, prev2 = [], None
        for raw in ("down", "flat"):
            o = fr.state(ofi_dir=raw, tick_dir="up", held_s=9.0,
                         bar_ts=1.0, prev=prev2,
                         **self.FULL_BAR, **FULL)
            prev2 = o["held"]
            held.append(o["ofi_dir"])
        assert held == ["down", "down"], "state lost the default"

    def test_the_gate_turns_the_hold_off_only_for_two_charts(self):
        src = (SRC / "web.py").read_text()
        i = src.index("def _fused_gate(")
        fn = src[i:src.index("\n    def ", i + 10)]
        assert "hold_legs=not _src," in fn, \
            "the hold is applied to chart-sourced rows"


class TestYouCanSeeWhichChartItRead:
    """Chris: "there is no way to know if it is actually reading
    the settings swapping from fused bar to the 2 charts
    strategy". A row saying UP is not evidence it read the chart
    you picked."""

    def test_the_raw_readings_ride_on_the_payload(self):
        src = (SRC / "web.py").read_text()
        i = src.index("def _fused_gate(")
        fn = src[i:src.index("\n    def ", i + 10)]
        assert '"rows": _src_rows,' in fn
        # THE VALUES, NOT THE KEYS. A key present with a constant
        # behind it reports a call level of zero forever, which
        # reads as "there is no deadband" -- the opposite of the
        # thing that caused the confusion.
        for k in ('"imbalance": int(_ob.get("imbalance")',
                  '"level": int(_ob.get("level")',
                  '"target": int(_ob.get("target")',
                  '"per_bar": int(_tb.get("per_bar")',
                  '"prints": int(_tb.get("prints")'):
            assert k in fn, f"{k} is not read from the chart"
        # AND THE OFI ROW IS THE CANDLE, NOT THE CALLED SIDE.
        #
        # This asserted the opposite for one revision. The called
        # side needs the accumulator past a CALL LEVEL that
        # defaults to 500 contracts -- a quarter of the
        # 2,000-contract bar Chris runs, and larger than the whole
        # target on the 125 and 250 bars, where it could never
        # call a side at all. He watched the row sit FLAT while
        # the chart moved and said so twice.
        #
        # THE CALLED SIDE IS STILL CARRIED, beside it, so a
        # disagreement is visible instead of silent.
        assert '"ofi": {"read": str(_ob.get("raw_side")' in fn, \
            "the row reports a gated side instead of the candle"
        assert '"called": str(_ob.get("read")' in fn, \
            "the called side is dropped rather than shown beside it"
        assert '"tick": {"read": str(_tb.get("body_dir")' in fn, \
            "the tick row reports the path vote, not the candle"
        assert '"path": str(_tb.get("dir")' in fn

    def test_the_gated_readings_are_drawn_but_marked_shown_only(self):
        """A candle can be visibly leaning while the GATED reading
        says FLAT, because the call level is a different number
        from the bar's target. Showing both, with the gated one
        marked, is what turns that from a bug report into a
        reading.

        THE RENDERED STRINGS, NOT THE COMMENTS. The previous
        version of this asserted `"call level" in fn` and the
        panel's own comment contained it in capitals -- a test
        passing on prose, for the fourth time in this suite."""
        src = (SRC / "web.py").read_text()
        i = src.index("function paintFusedGate(")
        fn = src[i:src.index("\nfunction ", i + 10)]
        code = "\n".join(ln for ln in fn.splitlines()
                         if not ln.strip().startswith("/*")
                         and not ln.strip().startswith("*"))
        assert "that closes the bar" in code
        assert "the called side is " in code
        assert "past zero, shown only" in code
        assert "the path vote is " in code
        assert "votes, shown only" in code

    def test_the_panel_shows_both_charts(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**TestThePanelActuallyRenders.BAR,
                       "tick_dir": "up", "ofi_dir": "", "where": "",
                       "turning": {"dir": "", "ticks_off": 0.0,
                                   "travel": 0.0, "band": 4.0,
                                   "decides": False},
                       "src": {"two_charts": True, "ofi_bar": 2000,
                               "tick_bar": 1500,
                               "say": "the 2,000-contract OFI chart",
                               "rows": {
                                   "ofi": {"read": "up",
                                           "called": "flat",
                                           "imbalance": 640,
                                           "level": 800,
                                           "target": 2000,
                                           "held_s": 0.0, "flips": 2},
                                   "tick": {"read": "down",
                                            "path": "flat",
                                            "votes": 40,
                                            "votes_needed": 120,
                                            "per_bar": 1500,
                                            "prints": 310}}}})
        # THE ROW IS THE CANDLE -- up, because the accumulator is
        # +640 -- while the gated reading disagrees at FLAT.
        assert "OFI chart <b>UP</b>" in html
        assert "+640" in html
        assert "of the 2,000 that closes the bar" in html
        assert "2 flips this bar" in html
        # AND THE DISAGREEMENT IS VISIBLE, not silent.
        assert "the called side is FLAT" in html
        assert "needs 800 past zero" in html
        assert "the path vote is FLAT" in html
        assert "40 of 120 votes" in html
        # VISIBLE, not merely present. A row inside a hidden
        # element is still in the HTML and still matches a
        # substring test.
        assert 'class="thin">tick chart <b>DOWN</b>' in html, \
            "the tick chart line is not drawn as a visible row"
        assert "310 of 1,500 prints" in html
        assert "display:none" not in html

    def test_the_four_thousand_size_exists(self):
        from liqmap.journal import _snap_ofi_bar
        from liqmap.ofibar import LADDER
        assert 4000 in LADDER, "the size he asked for is not built"
        assert _snap_ofi_bar(4000) == 4000
        src = (SRC / "web.py").read_text()
        i = src.index('id="jnlFusedOfiBar"')
        box = src[i:src.index("</select>", i)]
        assert 'value="4000"' in box, "it is built but not offered"

    def test_every_ladder_size_has_a_builder(self):
        """A size on the selector with no builder behind it is a
        row reading nothing at all."""
        src = (SRC / "web.py").read_text()
        i = src.index("svc.tape.ofibars = {")
        seg = src[i:i + 300]
        assert "for t in _OFI_LADDER" in seg, \
            "the builders are not made from the ladder"


class TestTheRowsAreTheCandlesThemselves:
    """Chris, watching the two-chart rows at 2,000 contracts and
    1,500 ticks: "these goes flat when there is actual movement in
    the OFI and the tick chart there is no reason for them to be
    flat almost ever if the actual ofi isnt flat or the tick chart
    isnt flat".

    BOTH ROWS WERE READING A GATE, NOT A CHART.

        The OFI row read the CALLED side, which needs the
        accumulator past a call level that defaults to 500
        contracts. On his 2,000-contract bar that is a quarter of
        the whole bar; on the 125 and 250 bars it is larger than
        the target, so the row could never call a side at all.

        The tick row read the PATH VOTE, gated on a 2% margin and
        a minimum vote count, so it reads flat early in every bar
        and whenever the prints are balanced.

    Neither is what the chart draws. `raw_side` is the
    accumulator's sign, which paints the OFI candle; `body_dir` is
    close against open, which is the tick candle.
    """

    def test_the_ofi_row_is_the_accumulators_sign(self):
        """NOT THE CALLED SIDE. A chart that is visibly green has
        a positive accumulator -- that IS the colour."""
        src = (SRC / "web.py").read_text()
        i = src.index("def _fused_gate(")
        fn = src[i:src.index("\n    def ", i + 10)]
        j = fn.index("if _src:")
        seg = fn[j:fn.index("out = _fr.state(", j)]
        code = "\n".join(ln for ln in seg.splitlines()
                         if not ln.strip().startswith("#"))
        # THE CHART'S OWN ACCUMULATOR, THROUGH THE BAND. The row
        # read `raw_side` for one revision -- the bare sign -- and
        # Chris watched it chatter against a chart that does not:
        # "the OFI is bouncing all over the place when im looking
        # at the chart OFI is pretty stable". `ofi_side` takes the
        # accumulator and the chart's own target so the band means
        # the same thing on a 125-contract bar as on a 4,000.
        assert '_fr.ofi_side(_ob.get("imbalance")' in code
        assert '_ob.get("target"), _band, _was)' in code
        assert '_ofi_row = str(_ob.get("read")' not in code

    def test_the_tick_row_is_the_candles_body(self):
        src = (SRC / "web.py").read_text()
        i = src.index("def _fused_gate(")
        fn = src[i:src.index("\n    def ", i + 10)]
        j = fn.index("if _src:")
        seg = fn[j:fn.index("out = _fr.state(", j)]
        code = "\n".join(ln for ln in seg.splitlines()
                         if not ln.strip().startswith("#"))
        assert '_tick_row = str(_tb.get("body_dir") or "")' in code
        assert '_tick_row = str(_tb.get("dir")' not in code

    def test_the_call_level_can_exceed_the_small_bars_target(self):
        """THE ARITHMETIC THAT MADE THIS UNARGUABLE. At the default
        call level a 125-contract bar can never call a side: the
        bar closes before the accumulator could reach it."""
        from liqmap.ofibar import CALL_LEVEL, LADDER
        assert CALL_LEVEL == 500
        unreachable = [t for t in LADDER if t <= CALL_LEVEL]
        assert unreachable, "this test no longer tests anything"
        assert 125 in unreachable and 250 in unreachable

    def test_the_gated_readings_survive_as_evidence(self):
        """THEY WERE PROTECTIONS SOMEBODY MEASURED. Deleting them
        loses the measurement; keeping them in charge overrode the
        chart. Drawn beside, they say when they would disagree."""
        src = (SRC / "web.py").read_text()
        i = src.index("def _fused_gate(")
        fn = src[i:src.index("\n    def ", i + 10)]
        assert '"called": str(_ob.get("read")' in fn
        assert '"path": str(_tb.get("dir")' in fn
        assert '"votes_needed": int(_tb.get("votes_needed")' in fn

    def test_the_panel_only_says_it_when_they_disagree(self, page_js):
        """A LINE THAT IS ALWAYS THERE IS NOT READ. The gated
        reading matters exactly when it differs from the candle."""
        from tests.test_stack_strip import run_js

        def draw(called, path):
            return run_js(page_js, "fusedGate",
                          {**TestThePanelActuallyRenders.BAR,
                           "tick_dir": "up", "ofi_dir": "up",
                           "where": "together", "agree": True,
                           "side": "up",
                           "turning": {"dir": "", "ticks_off": 0.0,
                                       "travel": 0.0, "band": 4.0,
                                       "decides": False},
                           "src": {"two_charts": True,
                                   "ofi_bar": 2000, "tick_bar": 1500,
                                   "say": "two charts",
                                   "rows": {
                                       "ofi": {"read": "up",
                                               "called": called,
                                               "imbalance": 640,
                                               "level": 800,
                                               "target": 2000,
                                               "flips": 0},
                                       "tick": {"read": "up",
                                                "path": path,
                                                "votes": 40,
                                                "votes_needed": 120,
                                                "per_bar": 1500,
                                                "prints": 310}}}})
        agree = draw("up", "up")
        assert "the called side" not in agree
        assert "the path vote" not in agree
        differ = draw("flat", "down")
        assert "the called side is FLAT" in differ
        assert "the path vote is DOWN" in differ


class TestTheBookIsSteadiedWithHysteresis:
    """Chris has now complained about both halves of one dial.

        No band: "the OFI is bouncing all over the place when im
        looking at the chart OFI is pretty stable ofi dont bounce
        all over the place at all its the steady one of the
        bunch".

        The chart's own flat 500-contract level: "these goes flat
        when there is actual movement ... there is no reason for
        them to be flat almost ever".

    Both are true of a plain threshold on a signed accumulator
    that resets every bar. A bare sign chatters through zero; a
    deadband stops the chatter by going FLAT in the middle, which
    is the other complaint. Hysteresis has neither edge.
    """

    def test_a_bare_sign_chatters_and_the_band_does_not(self):
        """THE SAME PATH THROUGH BOTH, so the difference is the
        mechanism and not the data."""
        # ONE REAL PUSH, THEN CHOP AROUND ZERO, THEN A REAL
        # REVERSAL. The first version of this used a path whose
        # bare sign only crossed zero twice, so both readings gave
        # two flips and the test proved nothing while passing.
        path = [210, 50, -40, 60, -70, 30, -210]

        def walk(band):
            prev, flips, out = "", 0, []
            for acc in path:
                now = fr.ofi_side(acc, 2000, band, prev)
                if now and prev and now != prev:
                    flips += 1
                prev = now
                out.append(now)
            return flips, out

        bare, _ = walk(0.0)
        banded, seen = walk(0.10)
        assert bare >= 5, "the path is not noisy enough to test this"
        assert banded == 1, "the band let the chop through"
        assert banded < bare, "the band did not steady anything"
        # AND IT NEVER WENT FLAT once it had called a side.
        after = seen[seen.index("up"):]
        assert "" not in after, "it went flat in the middle of the band"

    def test_it_holds_through_the_band_and_flips_past_it(self):
        prev = ""
        for acc in (150, 210):
            prev = fr.ofi_side(acc, 2000, 0.10, prev)
        assert prev == "up"
        for acc in (80, -50, -150, -190):
            prev = fr.ofi_side(acc, 2000, 0.10, prev)
            assert prev == "up", f"it let go at {acc}"
        assert fr.ofi_side(-205, 2000, 0.10, prev) == "down"

    def test_it_is_flat_only_before_the_first_call(self):
        """THE ONE MOMENT THE BOOK GENUINELY HAS NOT COMMITTED."""
        assert fr.ofi_side(0, 2000, 0.10, "") == ""
        assert fr.ofi_side(150, 2000, 0.10, "") == ""
        assert fr.ofi_side(210, 2000, 0.10, "") == "up"
        assert fr.ofi_side(0, 2000, 0.10, "up") == "up"

    def test_the_band_scales_with_the_bars_own_target(self):
        """A FRACTION, NOT A NUMBER OF CONTRACTS. The chart's own
        call level is a flat 500 whatever the bar is -- larger
        than the whole target on the 125 and 250 bars, where it
        can never call a side at all."""
        assert fr.ofi_side(30, 250, 0.10, "") == "up"      # edge 25
        assert fr.ofi_side(30, 4000, 0.10, "") == ""       # edge 400
        from liqmap.ofibar import CALL_LEVEL, LADDER
        for t in LADDER:
            edge = t * fr.OFI_BAND
            assert edge < t, f"the band is the whole {t} bar"
            if t <= CALL_LEVEL:
                assert edge < CALL_LEVEL, \
                    "the fractional band is no better than the flat one"

    def test_a_band_of_zero_is_the_bare_sign(self):
        """On the dial so the old behaviour can be put back and
        compared rather than argued about."""
        assert fr.ofi_side(1, 2000, 0.0, "") == "up"
        assert fr.ofi_side(-1, 2000, 0.0, "") == "down"
        assert fr.ofi_side(0, 2000, 0.0, "up") == ""

    def test_junk_never_raises(self):
        for bad in (None, "abc", [], {}):
            assert fr.ofi_side(bad, bad, bad, bad) in ("", "up", "down")

    def test_a_junk_previous_side_is_not_held(self):
        """REACHED ONLY WITH VALID NUMBERS. The test above feeds
        junk to every argument, so the conversion fails first and
        the held side is never looked at -- it passed with the
        validation deleted. Inside the band is the only place a
        junk `prev` can escape."""
        for bad in ("sideways", "UP", 7, [], {}, None):
            got = fr.ofi_side(50, 2000, 0.10, bad)
            assert got in ("", "up", "down"), f"{bad!r} got through"
            assert got == "", "a junk side was held as a direction"

    def test_the_band_reaches_the_gate_and_zero_survives(self):
        """SOURCE, because the fused engine does not run under the
        test client. A constant here leaves the selector working,
        the database correct and the row chattering -- and `or`
        on this line would read a 0% band as absent and silently
        substitute the default, which is the one value that
        turns the whole mechanism off."""
        src = (SRC / "web.py").read_text()
        i = src.index("def _fused_gate(")
        fn = src[i:src.index("\n    def ", i + 10)]
        assert "_band = float(ofi_band or 0.0)" in fn, \
            "the band is a constant inside the gate"
        j = src.index("_fband = float(int(getattr(s,")
        seg = src[j:j + 260]
        assert "is not None else 10" in seg, \
            "a 0% band is read as absent and replaced by the default"

    def test_the_called_side_rides_home(self):
        """WITHOUT THE MEMORY THE HYSTERESIS HAS NOTHING TO HOLD
        and the function degenerates into the deadband that
        produced the opposite complaint."""
        src = (SRC / "web.py").read_text()
        assert 'out["held"]["ofi_called"] = _ofi_row' in src
        i = src.index("def _fused_gate(")
        fn = src[i:src.index("\n    def ", i + 10)]
        assert '(prev or {}).get("ofi_called")' in fn

    def test_without_the_memory_it_goes_flat_in_the_middle(self):
        """The failure mode stated directly."""
        for acc in (80, -50, -150):
            assert fr.ofi_side(acc, 2000, 0.10, "") == "", \
                "this test no longer tests that"

    def test_the_setting_round_trips_and_zero_survives(self, tmp_path):
        from liqmap.journal import Journal
        j = Journal(str(tmp_path / "j.db"))
        s = j.settings("ES")
        assert s.fused_ofi_band == 10
        for v in (0, 5, 15, 25):
            s.fused_ofi_band = v
            assert j.save_settings(s).fused_ofi_band == v, \
                f"{v}% did not survive -- zero is a choice"
        s.fused_ofi_band = 7
        assert j.save_settings(s).fused_ofi_band == 10

    def test_the_route_takes_it(self, client, monkeypatch):
        _cme(monkeypatch)
        r = client.post("/api/journal/settings",
                        params={"coin": "ES", "fused_ofi_band": 0},
                        headers=AUTH).json()["settings"]
        assert r["fused_ofi_band"] == 0, "zero was read as absent"

    def test_the_selector_is_on_the_page(self):
        src = (SRC / "web.py").read_text()
        assert 'id="jnlOfiBand"' in src
        i = src.index('id="jnlOfiBand"')
        box = src[i:src.index("</select>", i)]
        for pct in (0, 5, 10, 15, 20, 25):
            assert f'value="{pct}"' in box, f"no {pct}% option"
        assert "fused_ofi_band: $('jnlOfiBand')" in src
        assert "set('jnlOfiBand'" in src
        assert "jnlOfiBand: $('jnlOfiBand')" in src

    def test_the_panel_shows_the_band_and_where_it_sits(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**TestThePanelActuallyRenders.BAR,
                       "tick_dir": "up", "ofi_dir": "up",
                       "where": "together", "agree": True, "side": "up",
                       "turning": {"dir": "", "ticks_off": 0.0,
                                   "travel": 0.0, "band": 4.0,
                                   "decides": False},
                       "band": {"pct": 10.0, "called": "up",
                                "edge": 400, "acc": 640,
                                "target": 4000}})
        assert "book <b" in html and "UP</b>" in html
        assert "+640" in html
        assert "flips past &plusmn;400" in html or "flips past ±400" in html
        assert "10% of 4,000" in html
        assert "holds until it clears the other side" in html

    def test_the_panel_says_when_there_is_no_band(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**TestThePanelActuallyRenders.BAR,
                       "tick_dir": "up", "ofi_dir": "up",
                       "where": "together", "agree": True, "side": "up",
                       "turning": {"dir": "", "ticks_off": 0.0,
                                   "travel": 0.0, "band": 4.0,
                                   "decides": False},
                       "band": {"pct": 0.0, "called": "up",
                                "edge": 0, "acc": 12, "target": 4000}})
        assert "no band" in html
        assert "crosses zero" in html

    def test_the_panel_says_before_the_first_call(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**TestThePanelActuallyRenders.BAR,
                       "tick_dir": "up", "ofi_dir": "", "where": "",
                       "turning": {"dir": "", "ticks_off": 0.0,
                                   "travel": 0.0, "band": 4.0,
                                   "decides": False},
                       "band": {"pct": 10.0, "called": "", "edge": 400,
                                "acc": 120, "target": 4000}})
        assert "not called yet" in html
