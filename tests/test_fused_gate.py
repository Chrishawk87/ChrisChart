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
        assert "pulling back from the bottom" in out["say"]
        # 12 of the candle's 20-point range off the low.
        assert out["back"] == pytest.approx(0.6)

    def test_the_book_stays_up_while_price_fades_off_the_top(self):
        out = fr.read(ofi_dir="up", tick_dir="up", open_px=5000.0,
                      high=5020.0, low=5000.0, close=5008.0)
        assert out["where"] == "bounce"
        assert "pulling back from the top" in out["say"]

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

    def test_the_denominator_is_the_candles_own_range(self):
        """THIS USED TO BE `tapeintent.depth`, which measures
        against the move from the OPEN and was framed on the book.
        On Chris's bar -- the book down 6.5 points while price ran
        23.5 up -- it divided by the 6.5 and saturated at 100%,
        telling him the whole move was gone on a candle that was
        quietly fading off its high.

        The range cannot be small relative to its own wick, so the
        range is the denominator."""
        # O 7160.50  H 7184.00  L 7154.00  C 7174.00, book DOWN.
        out = fr.read(ofi_dir="down", tick_dir="up", open_px=7160.50,
                      high=7184.00, low=7154.00, close=7174.00,
                      **FULL)
        # the payload rounds to three places
        assert out["back"] == pytest.approx(10.0 / 30.0, abs=1e-3), \
            "the retrace is not measured on the candle's range"
        assert out["back"] < 1.0, "it saturated again"

    def test_the_retrace_is_measured_against_the_books_direction(self):
        """His case is "OFI remaining down but tick is bouncing" --
        the book is the frame and price's position inside it is the
        question."""
        out = fr.read(ofi_dir="down", tick_dir="down", open_px=5020.0,
                      high=5020.0, low=5000.0, close=5010.0)
        assert out["back"] == pytest.approx(0.5)

    def test_the_body_picks_the_extreme_not_the_book(self):
        """THE INNER CANDLE IS THE THING THAT BOUNCES. Chris: "ofi
        remaining down but tick is bouncing from bottom or top" --
        the book is the context, the body is what comes off its
        extreme. An up body ran to the high, so the high is what it
        can pull back from, whatever the book says."""
        kw = dict(open_px=7160.50, high=7184.00, low=7154.00,
                  close=7174.00)
        up_body = fr.pullback("up", kw["high"], kw["low"], kw["close"])
        dn_body = fr.pullback("down", kw["high"], kw["low"], kw["close"])
        assert up_body == pytest.approx(10.0 / 30.0)
        assert dn_body == pytest.approx(20.0 / 30.0)
        for book in ("up", "down"):
            out = fr.read(ofi_dir=book, tick_dir="up", **kw, **FULL)
            assert out["back"] == pytest.approx(up_body, abs=1e-3), \
                "the book changed where the retrace is measured from"

    def test_a_flat_body_has_no_extreme_to_come_off(self):
        """A candle that closed where it opened has not run
        anywhere, so 0.0 would be a claim rather than an answer --
        but it is also not a pullback."""
        assert fr.pullback("", 10.0, 0.0, 5.0) == 0.0
        assert fr.pullback("flat", 10.0, 0.0, 5.0) == 0.0

    def test_at_the_extreme_nothing_is_given_back(self):
        out = fr.read(ofi_dir="down", tick_dir="down", open_px=5020.0,
                      high=5020.0, low=5000.0, close=5000.0, **FULL)
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
        assert "MOVING" in fn and "PULLING BACK FROM THE" in fn
        assert "outer (book)" in fn and "body (price)" in fn, \
            "the panel does not name the two candles the chart draws"
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
        assert fg.get("decides") is True
        for k in ("where", "entry", "label", "held"):
            assert k in fg, f"{k} missing on the {strat} strategy"

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


class TestTheBounceDial:
    """Chris: "give me a list of percentages that i can select from
    starting from 0 to 33%"."""

    def test_the_choices_run_from_zero_to_a_third(self):
        assert fr.BOUNCE_CHOICES[0] == 0.0
        assert fr.BOUNCE_CHOICES[-1] == pytest.approx(0.33)
        assert list(fr.BOUNCE_CHOICES) == sorted(fr.BOUNCE_CHOICES)
        assert len(set(fr.BOUNCE_CHOICES)) == len(fr.BOUNCE_CHOICES)

    @pytest.mark.parametrize("pct,close_px,where", [
        (0.33, 12.0, "bounce"), (0.33, 6.0, "together"),
        (0.10, 3.0, "bounce"), (0.10, 1.0, "together"),
        (0.05, 2.0, "bounce"),
    ])
    def test_a_tighter_setting_calls_it_sooner(self, pct, close_px,
                                               where):
        out = fr.read(ofi_dir="down", tick_dir="down", open_px=20.0,
                      high=20.0, low=0.0, close=close_px,
                      bounce_at=pct, **FULL)
        assert out["where"] == where

    def test_zero_means_any_give_back_at_all(self):
        at_low = fr.read(ofi_dir="down", tick_dir="down",
                         open_px=20.0, high=20.0, low=0.0,
                         close=0.0, bounce_at=0.0, **FULL)
        assert at_low["where"] == "together", \
            "at zero every bar reads as a bounce and nothing ever trades"
        off_low = fr.read(ofi_dir="down", tick_dir="down",
                          open_px=20.0, high=20.0, low=0.0,
                          close=0.25, bounce_at=0.0, **FULL)
        assert off_low["where"] == "bounce"

    def test_the_setting_reaches_the_reading(self):
        out = fr.read(ofi_dir="down", tick_dir="down", open_px=20.0,
                      high=20.0, low=0.0, close=1.0, bounce_at=0.15,
                      **FULL)
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
        ("bounce", "bouncing"),
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

    @pytest.mark.parametrize("where,word", [
        ("bounce", "bouncing"), ("apart", "come apart"),
        ("settling", "no longer settled"),
    ])
    def test_his_three_exits(self, where, word):
        why = fr.exit_reason("up", where, "up", 100.0, 100.0)
        assert why, f"{where} did not close the trade"
        assert word in why

    def test_the_bar_closing_closes_the_trade(self):
        why = fr.exit_reason("up", "together", "up", 200.0, 100.0)
        assert "bar this trade opened in has closed" in why

    def test_the_bar_closing_is_blamed_before_anything_else(self):
        """ORDER CARRIES MEANING. A new bar is unsettled by
        construction, so without this the log would blame the
        settling instead of the candle ending."""
        why = fr.exit_reason("up", "settling", "", 200.0, 100.0)
        assert "bar this trade opened in has closed" in why

    def test_moving_together_your_way_holds_the_trade(self):
        assert fr.exit_reason("up", "together", "up", 100.0, 100.0) == ""

    def test_moving_together_the_other_way_is_not_a_hold(self):
        why = fr.exit_reason("up", "together", "down", 100.0, 100.0)
        assert "against you" in why

    def test_an_unknown_bar_stamp_is_not_a_close(self):
        """A trade with no fused stamp on it -- a hand-taken row,
        or one opened before this strategy existed -- must not be
        closed for being in the wrong bar."""
        assert fr.exit_reason("up", "together", "up", 200.0, 0.0) == ""

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
        assert 'fus.get("bar_ts")' in seg
        assert 'getattr(t, "bar_ts"' in seg

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

    def test_his_bar_reads_as_pulling_back_from_the_top(self):
        out = fr.read(ofi_dir="up", tick_dir="up", close=7169.50,
                      **self.BAR, **FULL)
        assert out["where"] == "bounce"
        assert out["back"] == pytest.approx(14.5 / 30.0, abs=1e-3)
        assert "pulling back from the top" in out["say"]

    def test_the_old_framing_saturated_on_exactly_this_bar(self):
        """THE FAULT, WRITTEN DOWN. Framed on the book's move from
        the open, a bar with the book down 6.50 points and price up
        23.50 divides by the 6.50 and pins at 1.0 -- it reported
        the entire move gone on a candle sitting two thirds of the
        way up its own range."""
        from liqmap.tapeintent import depth
        assert depth("down", 7160.50, 7184.00, 7154.00,
                     7174.00) == pytest.approx(1.0)
        now = fr.read(ofi_dir="down", tick_dir="up", close=7174.00,
                      **self.BAR, **FULL)
        assert now["back"] == pytest.approx(10.0 / 30.0, abs=1e-3)
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
        assert "body (price)" in html

    def test_a_pullback_reads_from_the_top(self, page_js):
        from tests.test_stack_strip import run_js
        html = run_js(page_js, "fusedGate",
                      {**self.BAR, "where": "bounce", "ofi_dir": "up",
                       "agree": True})
        assert "PULLING BACK FROM THE TOP" in html

        down = run_js(page_js, "fusedGate",
                      {**self.BAR, "where": "bounce", "ofi_dir": "down",
                       "tick_dir": "down", "agree": True})
        assert "PULLING BACK FROM THE BOTTOM" in down

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
