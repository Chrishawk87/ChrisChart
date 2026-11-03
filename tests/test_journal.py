"""The trade journal under the stack: four must agree.

THE FIRST TEST IN THIS FILE IS THE ONE THAT MATTERS.

This package suggests; it does not execute. The journal is where that
line is most likely to be crossed, because "auto" is a word that means
two completely different things and only one of them is allowed here:
the rows write themselves, the order at Ironbeam is still placed by
hand. `TestItDoesNotTrade` greps for the shapes an execution path
would take, and its job is to fail loudly if anyone -- including a
future me, mid-task, trying to be helpful -- starts down that road.

The rest guard the arithmetic and the things a trade log gets quietly
wrong: a tick is $12.50 and a short is scored the other way round; an
open row's figure is the last print and not money; two rows for one
position make a total that is wrong in a way nobody spots; and costs
are stated rather than subtracted, because a journal averaging under a
tick a trade is describing a loss.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from liqmap.journal import (TICK, TICK_VALUE, Journal, Settings, Trade,
                            decide, should_close)

# The end-to-end class needs a configured CME engine and a database,
# which the panel suite's fixture already provides. Imported rather
# than copied so the two cannot drift about what "configured" means.
from tests.test_cme_panels import client  # noqa: F401,E402

ROOT = Path(__file__).resolve().parent.parent


def py_func(name: str, path: str = "liqmap/web.py") -> str:
    """A python function's whole body, found by indentation.

    SIX TESTS IN THIS SUITE HAVE BROKEN on a fixed-width slice --
    `src[i:i + 3000]` is an assertion about how long a function is,
    and every comment added to it is a chance to fail a test whose
    claim is still true. This reads to the first line indented no
    further than the `def`.
    """
    src = (ROOT / path).read_text()
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


@pytest.fixture()
def jn(tmp_path) -> Journal:
    return Journal(tmp_path / "journal.db")


class TestItDoesNotTrade:
    """The line this package does not cross."""

    def test_the_module_has_no_execution_path(self):
        src = (ROOT / "liqmap" / "journal.py").read_text()
        for bad in ("def submit(", "def execute(", "def place(",
                    "def order(", "def send_order", "def buy(",
                    "def sell("):
            assert bad not in src, bad

    def test_the_module_reaches_no_broker(self):
        """THE CODE, NOT THE PROSE.

        A substring grep over the file fails on the docstring that
        explains the module does not reach Ironbeam -- the comment
        saying so is indistinguishable from the thing it warns
        against. So this parses the module and looks at what it
        actually imports and calls.
        """
        import ast

        tree = ast.parse((ROOT / "liqmap" / "journal.py").read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        for bad in ("requests", "httpx", "urllib", "urllib3", "http",
                    "socket", "websocket", "websockets", "aiohttp",
                    "ironbeam", "ib_insync"):
            assert bad not in imported, f"journal imports {bad}"

    def test_the_module_reads_no_credentials(self):
        import ast

        tree = ast.parse((ROOT / "liqmap" / "journal.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in ("environ", "getenv"), \
                    "the journal has no business reading the environment"

    def test_the_routes_place_nothing(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("def api_journal_trade")
        blk = src[i:i + 3000]
        for bad in ("ironbeam", "submit_order", "place_order",
                    "requests.post"):
            assert bad not in blk.lower(), bad

    def test_the_payload_says_so_on_every_response(self):
        """Stated in the data, not only in a comment, so a reader of
        the API cannot mistake what auto means."""
        assert '"places_orders": False' in py_func("_journal_step")


class TestATickIsTwelveFifty:
    def test_the_constant(self):
        assert TICK == 0.25
        assert TICK_VALUE == 12.50

    @pytest.mark.parametrize("side,open_px,close_px,ticks", [
        ("up", 6700.00, 6701.00, 4.0),
        ("up", 6700.00, 6699.00, -4.0),
        ("down", 6700.00, 6699.00, 4.0),
        ("down", 6700.00, 6701.00, -4.0),
        ("up", 6700.00, 6700.25, 1.0),
        ("down", 6700.00, 6700.25, -1.0),
    ])
    def test_a_short_is_scored_the_other_way(self, side, open_px,
                                             close_px, ticks):
        t = Trade(side=side, open_px=open_px, close_px=close_px,
                  closed_ts=1.0)
        assert t.ticks == pytest.approx(ticks)
        assert t.dollars == pytest.approx(ticks * 12.50)

    def test_four_ticks_is_fifty_dollars(self):
        t = Trade(side="up", open_px=6700.0, close_px=6701.0,
                  closed_ts=1.0)
        assert t.dollars == 50.0

    def test_a_flat_trade_is_neither(self):
        t = Trade(side="up", open_px=6700.0, close_px=6700.0,
                  closed_ts=1.0)
        assert t.ticks == 0.0
        assert not t.won


class TestOneRowPerPosition:
    def test_a_second_open_is_refused(self, jn):
        assert jn.open_trade("ES", "up", 6700.0) is not None
        assert jn.open_trade("ES", "up", 6700.0) is None
        assert jn.open_trade("ES", "down", 6700.0) is None

    def test_closing_frees_it(self, jn):
        jn.open_trade("ES", "up", 6700.0)
        assert jn.close_trade("ES", 6701.0) is not None
        assert jn.open_trade("ES", "down", 6701.0) is not None

    def test_closing_nothing_is_not_an_error(self, jn):
        assert jn.close_trade("ES", 6700.0) is None

    def test_markets_do_not_share_a_position(self, jn):
        assert jn.open_trade("ES", "up", 6700.0) is not None
        assert jn.open_trade("NQ", "up", 24000.0) is not None

    def test_a_bad_price_opens_nothing(self, jn):
        assert jn.open_trade("ES", "up", 0.0) is None
        assert jn.open_trade("ES", "up", -5.0) is None


class TestTheOpenRowIsNotMoney:
    def test_it_is_labelled_unrealised(self, jn):
        jn.open_trade("ES", "up", 6700.0)
        d = jn.open_trade_for("ES").to_dict(6701.0)
        assert d["open"] is True
        assert d["live_ticks"] == 4.0
        assert d["live_dollars"] == 50.0
        # The closed-trade keys must be absent, so nothing adds an
        # open row into a realised total by reading the same name.
        assert "ticks" not in d
        assert "dollars" not in d

    def test_it_is_not_in_the_totals(self, jn):
        jn.open_trade("ES", "up", 6700.0)
        assert jn.stats("ES").n == 0
        assert jn.stats("ES").ticks == 0.0


class TestTheTotals:
    def _three(self, jn):
        jn.open_trade("ES", "up", 6700.0)
        jn.close_trade("ES", 6701.0)          # +4
        jn.open_trade("ES", "down", 6701.0)
        jn.close_trade("ES", 6700.5)          # +2
        jn.open_trade("ES", "up", 6700.5)
        jn.close_trade("ES", 6700.0)          # -2

    def test_wins_losses_and_ticks(self, jn):
        self._three(jn)
        s = jn.stats("ES")
        assert (s.n, s.wins, s.losses) == (3, 2, 1)
        assert s.ticks == pytest.approx(4.0)
        assert s.dollars == pytest.approx(50.0)

    def test_the_split_adds_up(self, jn):
        self._three(jn)
        s = jn.stats("ES")
        assert s.won_ticks == pytest.approx(6.0)
        assert s.lost_ticks == pytest.approx(-2.0)
        assert s.won_ticks + s.lost_ticks == pytest.approx(s.ticks)

    def test_costs_are_stated_not_subtracted(self, jn):
        self._three(jn)
        d = jn.stats("ES").to_dict()
        assert d["ticks"] == pytest.approx(4.0)   # untouched
        assert d["round_trip_ticks"] == 1.0
        assert d["beats_costs"] is True

    def test_a_thin_edge_is_flagged(self, jn):
        jn.open_trade("ES", "up", 6700.0)
        jn.close_trade("ES", 6700.25)             # +1 tick exactly
        d = jn.stats("ES").to_dict()
        assert d["per_trade_ticks"] == pytest.approx(1.0)
        assert d["beats_costs"] is False, \
            "one tick a trade does not beat a one tick round trip"


class TestTheExitRule:
    """Whichever comes first, and the stop is checked first."""

    S = Settings(tp_on=True, sl_on=True, tp_ticks=4.0, sl_ticks=2.0)

    def _open(self, side="up", px=6700.0):
        return Trade(side=side, open_px=px, mode="auto")

    def test_the_target(self):
        assert should_close(self._open(), self.S, 6701.0, "up") == "tp"

    def test_the_stop(self):
        assert should_close(self._open(), self.S, 6699.5, "up") == "sl"

    def test_the_stop_wins_a_tie(self):
        """A bar through both is recorded as stopped out. The other
        way assumes a fill that may not have happened, and assuming a
        fill is how a journal invents money."""
        s = Settings(tp_on=True, sl_on=True, tp_ticks=1.0, sl_ticks=1.0)
        t = self._open()
        # Far enough past both for either to claim it.
        assert should_close(t, s, 6690.0, "up") == "sl"

    def test_a_flip_closes_it(self):
        assert should_close(self._open(), self.S, 6700.0,
                            "down") == "flip"

    def test_neutral_does_not(self):
        """THE CHANGE. Exiting on lost agreement closed for about -1
        tick over and over: the read crosses its bar open constantly,
        so agreement breaks for an instant and the next look re-opens.
        None of those round trips was a decision about the market."""
        assert should_close(self._open(), self.S, 6700.0, "") == ""

    def test_agreement_holds_it(self):
        assert should_close(self._open(), self.S, 6700.25, "up") == ""

    def test_switches_off_means_off(self):
        off = Settings(tp_on=False, sl_on=False, tp_ticks=1.0,
                       sl_ticks=1.0)
        assert should_close(self._open(), off, 6710.0, "up") == ""
        assert should_close(self._open(), off, 6690.0, "up") == ""

    def test_a_zero_level_is_not_a_level(self):
        """Ticked on with nothing in the box must not close every
        trade instantly at zero ticks."""
        z = Settings(tp_on=True, sl_on=True, tp_ticks=0.0, sl_ticks=0.0)
        assert should_close(self._open(), z, 6700.0, "up") == ""

    def test_a_closed_trade_is_left_alone(self):
        t = Trade(side="up", open_px=6700.0, close_px=6701.0,
                  closed_ts=1.0)
        assert should_close(t, self.S, 6690.0, "down") == ""


class TestTheAutoSwitch:
    """`decide` is the whole policy, and it is callable.

    It used to live in a route closure, reachable only through an HTTP
    request to a fixture that never produced agreement -- so deleting
    the auto check broke no test. That is the mutation that survived.
    """

    OPEN = Settings(auto=True)
    OFF = Settings(auto=False)

    def test_auto_off_opens_nothing(self):
        assert decide(None, self.OFF, 6700.0, "up") == ("", "")
        assert decide(None, self.OFF, 6700.0, "down") == ("", "")

    def test_auto_on_opens_on_agreement(self):
        assert decide(None, self.OPEN, 6700.0, "up") == ("open", "up")
        assert decide(None, self.OPEN, 6700.0, "down") == ("open", "down")

    def test_auto_on_opens_nothing_without_agreement(self):
        assert decide(None, self.OPEN, 6700.0, "") == ("", "")

    def test_no_price_does_nothing(self):
        assert decide(None, self.OPEN, 0.0, "up") == ("", "")

    def test_an_auto_row_closes_on_a_flip(self):
        t = Trade(side="up", open_px=6700.0, mode="auto")
        assert decide(t, self.OPEN, 6700.0, "down") == ("close",
                                                        "flip")

    def test_an_auto_row_holds_through_neutral(self):
        t = Trade(side="up", open_px=6700.0, mode="auto")
        assert decide(t, self.OPEN, 6700.0, "") == ("", "")

    def test_a_manual_row_is_not_flipped_out_of(self):
        """Chris took it by hand; it is his to end."""
        t = Trade(side="up", open_px=6700.0, mode="manual")
        assert decide(t, self.OPEN, 6700.0, "down") == ("", "")

    def test_a_manual_row_still_respects_his_stop(self):
        """The levels are settings he turned on. Ignoring them until
        he refreshed would score trades he had said he was out of."""
        s = Settings(auto=False, sl_on=True, sl_ticks=2.0)
        t = Trade(side="up", open_px=6700.0, mode="manual")
        assert decide(t, s, 6699.0, "up") == ("close", "sl")

    def test_a_target_closes_with_auto_off(self):
        s = Settings(auto=False, tp_on=True, tp_ticks=4.0)
        t = Trade(side="up", open_px=6700.0, mode="manual")
        assert decide(t, s, 6701.0, "up") == ("close", "tp")

    def test_it_never_opens_while_one_is_open(self):
        t = Trade(side="up", open_px=6700.0, mode="auto")
        act, _ = decide(t, self.OPEN, 6700.25, "up")
        assert act != "open"


class TestSettingsPersist:
    def test_defaults_without_a_row(self, jn):
        s = jn.settings("ES")
        assert s.auto is False
        assert s.tp_on is False and s.sl_on is False

    def test_they_survive(self, jn, tmp_path):
        jn.save_settings(Settings(coin="ES", auto=True, tp_on=True,
                                  tp_ticks=12.0, sl_ticks=5.0))
        again = Journal(tmp_path / "journal.db")
        s = again.settings("ES")
        assert s.auto is True
        assert s.tp_on is True
        assert s.tp_ticks == 12.0
        assert s.sl_ticks == 5.0

    def test_markets_keep_their_own(self, jn):
        jn.save_settings(Settings(coin="ES", auto=True))
        jn.save_settings(Settings(coin="NQ", auto=False))
        assert jn.settings("ES").auto is True
        assert jn.settings("NQ").auto is False

    def test_a_negative_level_is_clamped(self, jn):
        s = jn.save_settings(Settings(coin="ES", tp_ticks=-5.0))
        assert s.tp_ticks == 0.0


class TestTheTradesSurviveARestart:
    def test_closed_trades_come_back(self, jn, tmp_path):
        jn.open_trade("ES", "up", 6700.0)
        jn.close_trade("ES", 6701.0)
        again = Journal(tmp_path / "journal.db")
        assert again.stats("ES").n == 1
        assert again.stats("ES").ticks == 4.0

    def test_an_open_trade_comes_back_open(self, jn, tmp_path):
        """A restart must not silently flatten a position the log
        says is on."""
        jn.open_trade("ES", "down", 6700.0)
        again = Journal(tmp_path / "journal.db")
        t = again.open_trade_for("ES")
        assert t is not None
        assert t.side == "down"


class TestClearingIsDeliberate:
    def test_it_takes_a_confirmation(self, jn):
        jn.open_trade("ES", "up", 6700.0)
        jn.close_trade("ES", 6701.0)
        assert jn.clear("ES") == 1
        assert jn.stats("ES").n == 0

    def test_the_route_refuses_without_confirm(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("def api_journal_clear")
        blk = src[i:i + 1200]
        assert "confirm" in blk
        assert "cannot be undone" in blk


class TestTheColumnEndToEnd:
    """Through the routes, against the CME fixture.

    A journal that works in isolation and never reaches the panel is
    the failure these catch -- the same class of thing as the price
    that vanished off the read line while 242 tests passed.
    """

    AUTH = {"Authorization": "Bearer t0ken"}

    def _read(self, client):
        r = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=self.AUTH)
        assert r.status_code == 200, r.text
        return r.json()

    def _post(self, client, path, **params):
        r = client.post(path, params=params, headers=self.AUTH)
        assert r.status_code == 200, r.text
        return r.json()

    def test_the_payload_carries_the_column(self, client, monkeypatch):
        from tests.test_cme_panels import _cme

        _cme(monkeypatch)
        j = self._read(client).get("journal")
        assert j is not None
        assert "error" not in j, j
        assert j["places_orders"] is False
        assert "stats" in j and "settings" in j

    def test_it_names_which_of_the_four_disagree(self, client,
                                                 monkeypatch):
        """"no signal" says nothing about which one is holding it
        up, which is the whole reason this column exists."""
        from tests.test_cme_panels import _cme

        _cme(monkeypatch)
        says = self._read(client)["journal"]["says"]
        assert set(says) == {"15m", "5m", "1m", "read"}

    def test_a_manual_trade_opens_and_closes(self, client, monkeypatch):
        from tests.test_cme_panels import _cme

        _cme(monkeypatch)
        d = self._post(client, "/api/journal/trade", coin="ES",
                       action="open", side="up")
        assert d["ok"] is True, d
        assert self._read(client)["journal"]["open"] is not None

        d2 = self._post(client, "/api/journal/trade", coin="ES",
                        action="close")
        assert d2["ok"] is True, d2
        after = self._read(client)["journal"]
        assert after["open"] is None
        assert after["stats"]["trades"] == 1

    def test_a_second_open_is_refused_with_a_reason(self, client,
                                                    monkeypatch):
        from tests.test_cme_panels import _cme

        _cme(monkeypatch)
        self._post(client, "/api/journal/trade", coin="ES",
                   action="open", side="up")
        d = self._post(client, "/api/journal/trade", coin="ES",
                       action="open", side="down")
        assert d["ok"] is False
        assert "already open" in d["why"]

    def test_a_trade_with_no_side_is_refused(self, client, monkeypatch):
        from tests.test_cme_panels import _cme

        _cme(monkeypatch)
        d = self._post(client, "/api/journal/trade", coin="ES",
                       action="open")
        assert d["ok"] is False
        assert "direction" in d["why"]

    def test_the_settings_round_trip(self, client, monkeypatch):
        from tests.test_cme_panels import _cme

        _cme(monkeypatch)
        d = self._post(client, "/api/journal/settings", coin="ES",
                       auto=True, tp_on=True, tp_ticks=20,
                       sl_on=True, sl_ticks=6)
        assert d["settings"]["auto"] is True
        assert d["settings"]["tp_ticks"] == 20.0
        got = self._read(client)["journal"]["settings"]
        assert got["sl_ticks"] == 6.0
        assert got["tick_value"] == 12.50

    def test_only_what_is_sent_changes(self, client, monkeypatch):
        from tests.test_cme_panels import _cme

        _cme(monkeypatch)
        self._post(client, "/api/journal/settings", coin="ES",
                   auto=True, tp_ticks=20)
        d = self._post(client, "/api/journal/settings", coin="ES",
                       sl_ticks=9)
        assert d["settings"]["auto"] is True
        assert d["settings"]["tp_ticks"] == 20.0
        assert d["settings"]["sl_ticks"] == 9.0

    def test_auto_off_opens_nothing(self, client, monkeypatch):
        """With the switch off the column is a record of what Chris
        did, and must never add a row he did not."""
        from tests.test_cme_panels import _cme

        _cme(monkeypatch)
        self._post(client, "/api/journal/settings", coin="ES",
                   auto=False)
        for _ in range(3):
            assert self._read(client)["journal"]["open"] is None

    def test_clearing_needs_confirmation(self, client, monkeypatch):
        from tests.test_cme_panels import _cme

        _cme(monkeypatch)
        self._post(client, "/api/journal/trade", coin="ES",
                   action="open", side="up")
        self._post(client, "/api/journal/trade", coin="ES",
                   action="close")
        d = self._post(client, "/api/journal/clear", coin="ES")
        assert d["ok"] is False
        assert self._read(client)["journal"]["stats"]["trades"] == 1
        d2 = self._post(client, "/api/journal/clear", coin="ES",
                        confirm="yes")
        assert d2["ok"] is True
        assert self._read(client)["journal"]["stats"]["trades"] == 0

    @staticmethod
    def _paint_fn() -> str:
        """The whole of `paintJournal`, to its closing brace.

        A fixed-width slice here is an assertion about how long the
        function is: this one took 6,000 characters and started
        failing the moment the verdict block pushed `jnlRecent` past
        it, with every claim in it still true. That is the fourth
        test to break that way today.
        """
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("function paintJournal(")
        depth, j = 0, src.index("{", i)
        for k in range(j, len(src)):
            if src[k] == "{":
                depth += 1
            elif src[k] == "}":
                depth -= 1
                if depth == 0:
                    return src[i:k + 1]
        raise AssertionError("paintJournal has no closing brace")

    def test_the_page_draws_the_column(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        assert 'id="jnlBox"' in src
        blk = self._paint_fn()
        for want in ("jnlStats", "jnlOpen", "jnlRecent", "jnlAgree",
                     "beats_costs"):
            assert want in blk, want
        # And it is actually called.
        assert "paintJournal(d);" in src


class TestTheFourthSignalIsActuallyThere:
    """Chris: "it doesn't show the read so that it can make the
    decision to trade".

    The column asked for the OFI bar read at `out["read"]` on the
    /api/read payload. That route has never carried it -- it is a
    property of the OFI builder and it is served by /api/ofibars. So
    the fourth signal was None on every single poll, the four could
    never agree, and auto had nothing to fire on. Invisible, because
    "not all four" is also what a quiet market looks like.
    """

    AUTH = {"Authorization": "Bearer t0ken"}

    def _read(self, client, **kw):
        params = {"coin": "ES", "interval": "15m",
                  "timeframes": "15m,5m,1m"}
        params.update(kw)
        r = client.get("/api/read", params=params, headers=self.AUTH)
        assert r.status_code == 200, r.text
        return r.json()

    def _with_ofi(self, monkeypatch):
        from tests.test_cme_panels import _cme

        rt, tape = _cme(monkeypatch)
        # The OFI builders are attached when the footprint is built.
        rt.footprint(rule="1m", volume=0, min_volume=0, multiplier=4.0)
        rec = getattr(rt, "_candlerec", None)
        if rec is not None:
            rec.stop()
        return rt, tape

    def test_the_read_reaches_the_column(self, client, monkeypatch):
        self._with_ofi(monkeypatch)
        j = self._read(client)["journal"]
        assert "read" in j, j.keys()
        assert isinstance(j["read"], dict)

    def test_it_is_the_ofi_bar_read_not_an_empty_dict(self, client,
                                                      monkeypatch):
        """The failure it replaces returned nothing and looked like a
        quiet market."""
        _rt, tape = self._with_ofi(monkeypatch)
        # Drive one OFI bar so there is something to read.
        b = tape.ofibars[max(tape.ofibars)]
        px = 6800 * 1_000_000_000
        t = 1000.0
        for i in range(40):
            b.on_book(t + i, px + i * 250_000_000, 500,
                      px + (i + 1) * 250_000_000, 500)
        r = self._read(client)["journal"]["read"]
        assert r.get("forming") is True, r
        assert r.get("read") in ("up", "down", "flat"), r
        assert "ticks" in r

    def test_a_flat_read_is_not_agreement(self, client, monkeypatch):
        """Price sitting at the bar open is not a direction, and
        counting it would open a trade on a bar that has not moved."""
        from liqmap import journal as jn

        assert jn.decide(None, jn.Settings(auto=True), 6700.0,
                         "flat") == ("", "")

    def test_the_page_draws_the_read(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("function paintJournal(")
        blk = src[i:i + 7000]
        assert "J.read" in blk
        assert "no OFI bar forming" in blk


class TestAutoEntryAndExitFire:
    """Chris: "it has auto record but it dont have auto entry and
    auto exit based on the rules that i gave you".

    With the read permanently absent they could not. These drive the
    policy directly with all four agreeing, so the entry and every
    exit are exercised rather than assumed.
    """

    def _s(self, **kw):
        from liqmap.journal import Settings

        base = dict(coin="ES", auto=True)
        base.update(kw)
        return Settings(**base)

    def test_all_four_agreeing_opens(self):
        from liqmap.journal import decide

        assert decide(None, self._s(), 6700.0, "down") == ("open",
                                                            "down")

    def test_it_exits_when_it_flips(self):
        from liqmap.journal import Trade, decide

        t = Trade(side="down", open_px=6700.0, mode="auto")
        assert decide(t, self._s(), 6700.0, "up") == ("close", "flip")

    def test_it_holds_when_the_signal_merely_goes_quiet(self):
        """Chris's change, and the reason for it: the old rule closed
        for -1 tick over and over on a read crossing its own open."""
        from liqmap.journal import Trade, decide

        t = Trade(side="down", open_px=6700.0, mode="auto")
        assert decide(t, self._s(), 6700.0, "") == ("", "")

    def test_it_exits_on_the_target(self):
        from liqmap.journal import Trade, decide

        s = self._s(tp_on=True, tp_ticks=8.0)
        t = Trade(side="down", open_px=6700.0, mode="auto")
        assert decide(t, s, 6698.0, "down") == ("close", "tp")

    def test_it_exits_on_the_stop(self):
        from liqmap.journal import Trade, decide

        s = self._s(sl_on=True, sl_ticks=4.0)
        t = Trade(side="down", open_px=6700.0, mode="auto")
        assert decide(t, s, 6701.0, "down") == ("close", "sl")

    def test_it_holds_while_all_four_hold(self):
        from liqmap.journal import Trade, decide

        s = self._s(tp_on=True, tp_ticks=8.0, sl_on=True, sl_ticks=4.0)
        t = Trade(side="down", open_px=6700.0, mode="auto")
        assert decide(t, s, 6699.75, "down") == ("", "")

    def test_the_route_applies_the_decision(self):
        """The policy is called AND acted on. A decision nothing acts
        on is the same bug in a different place."""
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("def _journal_step")
        blk = src[i:i + 2500]
        assert "jn.decide(" in blk
        assert 'act == "close"' in blk
        assert 'act == "open"' in blk
        assert "j.close_trade(" in blk
        assert "j.open_trade(" in blk


class TestTheCandleDecidesAndTheStackVetoes:
    """Chris: "its supposed to be tracking the OFI candle making
    trades based on that and nothing more".

    So the three timeframes stopped voting. The direction is the OFI
    bar read and nothing else; the stack can only VETO, and only when
    all three point the other way.

    The three-of-four tests that stood here are gone rather than
    fixed -- the rule they described is not the rule any more, and
    keeping them green would have meant keeping it.
    """

    @staticmethod
    def _side(d15, d5, d1, read):
        """THE REAL RULE, called directly.

        This used to re-implement it, because the rule lived in a
        route closure. That is not a test of anything: deleting the
        veto from the route left all of these green. It is
        `journal.signal` now.
        """
        from liqmap.journal import signal

        # The imbalance is passed as agreeing unless a test says
        # otherwise; its own rule is covered in its own class.
        imb = read if read in ("up", "down") else ""
        return signal({"15m": d15, "5m": d5, "1m": d1}, read, imb)

    def test_the_candle_alone_is_enough(self):
        assert self._side(None, None, None, "up") == ("up", "")

    def test_one_timeframe_against_does_not_stop_it(self):
        assert self._side("down", None, None, "up") == ("up", "")

    def test_two_against_does_not_stop_it(self):
        assert self._side("down", "down", None, "up") == ("up", "")

    def test_all_three_against_vetoes_it(self):
        assert self._side("down", "down", "down", "up") == ("", "stack")

    def test_all_three_with_it_is_fine(self):
        assert self._side("up", "up", "up", "up") == ("up", "")

    def test_a_flat_candle_is_no_signal(self):
        assert self._side("up", "up", "up", "flat") == ("", "")
        assert self._side("up", "up", "up", "") == ("", "")

    def test_a_silent_stack_cannot_veto(self):
        """All three must POINT the other way. Absent is not
        opposed."""
        assert self._side(None, None, None, "down") == ("down", "")
        assert self._side("up", None, "up", "down") == ("down", "")

    def test_the_route_calls_it_rather_than_copying_it(self):
        blk = py_func("_agreement")
        assert "jn_signal(says, r," in blk
        # The vote count is gone, not merely unused.
        assert "votes.count(way) >= MIN_AGREE" not in blk
        # And the rule is not re-written alongside the call.
        assert "all(says[t] == other" not in blk

    def test_the_veto_is_reported(self):
        """A block that fires invisibly is worse than no block -- it
        reads as a market with nothing in it."""
        blk = py_func("_journal_step")
        assert '"blocked"' in blk
        assert '"would_be"' in blk
        src = (ROOT / "liqmap" / "web.py").read_text()
        j = src.index("function paintJournal(")
        assert "VETOED" in src[j:j + 12000]


class TestResettingTheTest:
    """Chris: "i need to be able to reset the test if i choose"."""

    def test_the_button_exists(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        assert "function jnlReset(" in src
        assert "Reset test" in src

    def test_it_asks_first(self):
        """A record a stray click can empty is not a record."""
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("function jnlReset(")
        blk = src[i:i + 900]
        assert "confirm(" in blk
        assert "cannot be undone" in blk
        # Through the URL encoder, not concatenated: a symbol with a
        # '#' in it turns the rest of the query into a fragment and
        # the request arrives without its token.
        assert "confirm: 'yes'" in blk
        assert "q({coin:" in blk

    def test_it_names_the_count_being_deleted(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("function jnlReset(")
        assert "stats.trades" in src[i:i + 500]

    def test_it_clears_only_this_market(self, jn):
        jn.open_trade("ES", "up", 6700.0)
        jn.close_trade("ES", 6701.0)
        jn.open_trade("NQ", "up", 24000.0)
        jn.close_trade("NQ", 24001.0)
        assert jn.clear("ES") == 1
        assert jn.stats("ES").n == 0
        assert jn.stats("NQ").n == 1

    def test_the_test_starts_clean_after_a_reset(self, jn):
        for _ in range(5):
            jn.open_trade("ES", "up", 6700.0)
            jn.close_trade("ES", 6701.0)
        assert jn.stats("ES").verdict()["need"] == 25
        jn.clear("ES")
        v = jn.stats("ES").verdict()
        assert v["need"] == 30
        assert v["enough"] is False


class TestTheAutoButton:
    """One button, two states, and neither of them trades.

    Chris asked for "a button that i can switch between manual
    trading and allowing the sites agent to do it". The switch here
    changes who writes the journal rows. It does not reach a broker,
    and the panel says so in the state the button is in rather than
    only in a paragraph above it.
    """

    def test_the_button_exists(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        assert 'id="jnlAutoBtn"' in src
        assert "function jnlToggleAuto(" in src

    def test_it_shows_the_state_it_is_in(self):
        """A toggle labelled with the mode it would switch TO is the
        commonest way to end up in the one you thought you left."""
        src = (ROOT / "liqmap" / "web.py").read_text()
        # TWO PLACES LABEL IT and both must agree: the click handler
        # paints immediately so the word does not lag a poll, and the
        # painter sets it from the server's state. An earlier version
        # of this test found only the first and would have passed with
        # the second one inverted.
        spots = [i for i in range(len(src))
                 if src.startswith("const btn = $('jnlAutoBtn');", i)]
        assert len(spots) == 2, spots
        labels = [src[i:i + 700] for i in spots]
        assert "c.checked ? 'AUTO' : 'MANUAL'" in labels[0], labels[0]
        assert "S.auto ? 'AUTO' : 'MANUAL'" in labels[1], labels[1]

    def test_neither_state_places_an_order(self):
        """The whole point. Both branches of the switch end at a row
        in a table."""
        from liqmap.journal import Settings, Trade, decide

        for auto in (True, False):
            s = Settings(auto=auto, tp_on=True, tp_ticks=4.0,
                         sl_on=True, sl_ticks=2.0)
            act, _ = decide(None, s, 6700.0, "up")
            assert act in ("", "open")
            t = Trade(side="up", open_px=6700.0, mode="auto")
            act2, _ = decide(t, s, 6701.0, "up")
            assert act2 in ("", "close")

    def test_the_panel_says_where_auto_stops(self):
        """The whole jnlBox, found by its closing div rather than by
        a character count -- this sliced 2,500 and broke when the
        reset button was added, with its claim still true. Fifth test
        today to fail that way."""
        import re

        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index('id="jnlBox"')
        j = src.index('id="jnlRecent"', i)
        # WHITESPACE-FLATTENED. The sentence wraps across a line in
        # the markup, so a raw substring search misses it and reports
        # the panel as silent about the one thing it must say.
        blk = re.sub(r"\s+", " ", src[i:j])
        assert "nothing here reaches <b>Ironbeam" in blk or \
               "nothing here reaches Ironbeam" in blk, blk[:400]
        assert "yours to place" in blk

    def test_the_switch_still_persists(self, jn):
        from liqmap.journal import Settings

        jn.save_settings(Settings(coin="ES", auto=True))
        assert jn.settings("ES").auto is True
        jn.save_settings(Settings(coin="ES", auto=False))
        assert jn.settings("ES").auto is False


class TestWhetherItIsWorthTrusting:
    """Chris: "once it has logged enough trades ill know whether or
    not its worth trusting -- thats what i need it to show".

    The totals answer that badly. What fools you at small sample size
    is a run of wins, and a run of wins is exactly what a promising
    start looks like. So the column reports the average against its
    own standard error and against the cost of trading, and refuses
    to call anything readable before thirty trades.
    """

    @staticmethod
    def _stats(ticks):
        from liqmap.journal import Stats, Trade

        return Stats([Trade(side="up", open_px=6700.0,
                            close_px=6700.0 + t * 0.25, closed_ts=1.0)
                      for t in ticks])

    def test_nothing_is_readable_at_three_trades(self):
        v = self._stats([4, -2, 6]).verdict()
        assert v["enough"] is False
        assert "Under thirty" in v["say"] or "Nothing can be read" \
            in v["say"]

    def test_an_empty_log_says_so_without_dividing_by_zero(self):
        v = self._stats([]).verdict()
        assert v["enough"] is False
        assert v["t"] == 0.0

    def test_one_trade_does_not_crash_the_spread(self):
        s = self._stats([4])
        assert s.spread_ticks == 0.0
        assert s.verdict()["enough"] is False

    def test_a_winning_streak_is_not_enough(self):
        """TEN STRAIGHT WINS AND IT STILL SAYS NO. This is the whole
        point of the verdict: that record is exactly what both a real
        edge and a lucky run look like."""
        v = self._stats([4] * 10).verdict()
        assert v["enough"] is False
        assert v["need"] == 20

    def test_a_losing_record_is_called_what_it_is(self):
        v = self._stats([2, -6] * 20).verdict()
        assert v["enough"] is False
        assert "not paying for itself" in v["say"]

    def test_a_noisy_average_is_not_mistaken_for_an_edge(self):
        """Forty trades averaging well above costs, scattering far
        wider. The average alone reads as a clear win.

        Written out rather than generated from a seed: a seeded
        fixture whose meaning quietly changes is how the first
        version of this test came to assert a mean it did not have.
        """
        s = self._stats([9] * 24 + [-7] * 16)     # mean +2.6 ticks
        d = s.to_dict()
        assert d["per_trade_ticks"] > 2.0
        assert d["spread_ticks"] > 7.0            # scatter dwarfs it
        assert s.verdict()["enough"] is False
        assert "noise" in s.verdict()["say"]

    def test_a_clean_record_clears_the_floor(self):
        """And is still described as a floor, not a result."""
        s = self._stats([6, 5, 7, 6, 5, 6, 7, 5, 6, 6] * 5)
        v = s.verdict()
        assert v["enough"] is True
        assert v["t"] >= 2.0
        assert "not a result" in v["say"]

    def test_the_streaks_are_read_in_order(self):
        """Ten wins then ten losses and an alternating twenty have
        the same totals and are not the same record."""
        a = self._stats([4] * 10 + [-4] * 10).streaks()
        b = self._stats([4, -4] * 10).streaks()
        assert a == {"win": 10, "loss": 10}
        assert b == {"win": 1, "loss": 1}

    def test_the_verdict_reaches_the_payload(self):
        d = self._stats([4, -2, 6]).to_dict()
        assert "verdict" in d and "streaks" in d
        assert "spread_ticks" in d

    def test_the_panel_leads_with_it(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("const V = st.verdict")
        blk = src[i:i + 2500]
        assert "NOT YET READABLE" in blk
        assert "WORTH A CLOSER LOOK" in blk
        # Above the cost caveat, not buried under it.
        assert blk.index("NOT YET READABLE") < blk.index("round trip")


class TestTheImbalanceMustAgree:
    """Chris: "i have seen this make consecutive trades trying to
    trade OFI is up but imbalance is down and it losses the trade ...
    if OFI and imbalance dont agree no trade".

    Two different facts about one bar: where price sits against the
    open, and which side is ahead in the flow that will end the bar.
    A bar where they disagree is being pushed one way on price while
    the book leans the other.
    """

    @staticmethod
    def _s(read, imb, says=None):
        from liqmap.journal import signal

        return signal(says or {}, read, imb)

    def test_both_up_is_a_buy(self):
        assert self._s("up", "up") == ("up", "")

    def test_both_down_is_a_short(self):
        assert self._s("down", "down") == ("down", "")

    def test_candle_up_imbalance_down_is_no_trade(self):
        assert self._s("up", "down") == ("", "imbalance")

    def test_candle_down_imbalance_up_is_no_trade(self):
        assert self._s("down", "up") == ("", "imbalance")

    def test_a_flat_imbalance_is_not_agreement(self):
        """It has not taken a side, so there is nothing to agree
        with."""
        assert self._s("up", "flat") == ("", "imbalance")
        assert self._s("up", "") == ("", "imbalance")

    def test_a_flat_candle_is_no_signal_either(self):
        assert self._s("flat", "up") == ("", "")

    def test_the_stack_veto_still_applies_on_top(self):
        both = {"15m": "down", "5m": "down", "1m": "down"}
        assert self._s("up", "up", both) == ("", "stack")

    def test_the_builder_reports_both_from_one_place(self):
        """A caller reaching into the builder for the second half is
        how the two come to be read at different instants."""
        from liqmap.ofibar import OfiBarBuilder

        b = OfiBarBuilder(tick_size=0.25, threshold=2000,
                          max_s=900.0, min_updates=2)
        P, T = 1_000_000_000, 250_000_000
        px = 6700 * P
        b.on_book(0.0, px, 100, px + T, 100)
        b.on_book(1.0, px + T, 400, px + 2 * T, 100)
        r = b.bar_read
        for k in ("read", "imb", "imbalance", "agrees"):
            assert k in r, k
        assert r["agrees"] is (r["read"] == r["imb"]
                               and r["read"] in ("up", "down"))


class TestTwoTradesToACandle:
    """Chris: "its trading the same ofi candle over and over again it
    gets out of the trade and immediately gets back in ... we trade
    the OFI candle at maximum of 2 times a candle ... its one trade
    per direction".

    Holding to a flip stopped the EXIT churning. It did nothing about
    the entry: with re-entry immediate, one candle could be traded
    the same way again and again, each one the same read on the same
    bar and charged a round trip.
    """

    def _s(self):
        from liqmap.journal import Settings

        return Settings(coin="ES", auto=True)

    def test_the_first_trade_opens(self):
        from liqmap.journal import decide

        assert decide(None, self._s(), 6700.0, "up",
                      taken=()) == ("open", "up")

    def test_the_same_direction_again_is_refused(self):
        from liqmap.journal import decide

        assert decide(None, self._s(), 6700.0, "up",
                      taken=("up",)) == ("", "")

    def test_the_other_direction_is_allowed(self):
        from liqmap.journal import decide

        assert decide(None, self._s(), 6700.0, "down",
                      taken=("up",)) == ("open", "down")

    def test_a_third_trade_is_refused(self):
        from liqmap.journal import decide

        assert decide(None, self._s(), 6700.0, "up",
                      taken=("down", "up")) == ("", "")

    def test_the_cap_holds_on_its_own(self):
        """THE TWO RULES COINCIDE AT TWO DIRECTIONS, which hid a gap:
        deleting the `len(taken) >= MAX_PER_BAR` check broke no test,
        because one-per-direction already caps it at two when up and
        down are the only sides.

        This reaches the cap directly with a candle that somehow took
        the same side twice -- not a state the rule above can
        produce, which is exactly why the guard is worth keeping and
        worth testing separately."""
        from liqmap.journal import decide

        assert decide(None, self._s(), 6700.0, "up",
                      taken=("down", "down")) == ("", "")

    def test_the_cap_is_two(self):
        from liqmap.journal import MAX_PER_BAR

        assert MAX_PER_BAR == 2

    def test_the_count_comes_from_the_table(self, jn):
        """FROM THE TABLE, NOT FROM MEMORY. A cap counted in a
        variable resets on every restart, which on a poll-driven
        column is indistinguishable from having no cap."""
        jn.open_trade("ES", "up", 6700.0, bar_ts=111.0)
        jn.close_trade("ES", 6701.0)
        assert jn.sides_in_bar("ES", 111.0) == ("up",)
        jn.open_trade("ES", "down", 6701.0, bar_ts=111.0)
        jn.close_trade("ES", 6700.0)
        assert jn.sides_in_bar("ES", 111.0) == ("up", "down")

    def test_it_survives_a_restart(self, jn, tmp_path):
        from liqmap.journal import Journal

        jn.open_trade("ES", "up", 6700.0, bar_ts=222.0)
        jn.close_trade("ES", 6701.0)
        again = Journal(tmp_path / "journal.db")
        assert again.sides_in_bar("ES", 222.0) == ("up",)

    def test_the_next_candle_starts_clean(self, jn):
        jn.open_trade("ES", "up", 6700.0, bar_ts=111.0)
        jn.close_trade("ES", 6701.0)
        jn.open_trade("ES", "down", 6701.0, bar_ts=111.0)
        jn.close_trade("ES", 6700.0)
        assert jn.sides_in_bar("ES", 111.0) == ("up", "down")
        assert jn.sides_in_bar("ES", 222.0) == ()

    def test_no_bar_id_means_no_cap_lookup(self, jn):
        """A read with no forming bar must not make every trade ever
        taken look like it belongs to candle zero."""
        jn.open_trade("ES", "up", 6700.0, bar_ts=0.0)
        jn.close_trade("ES", 6701.0)
        assert jn.sides_in_bar("ES", 0.0) == ()

    def test_the_route_passes_the_candle_through(self):
        blk = py_func("_journal_step")
        assert "j.sides_in_bar(coin_, bar_ts)" in blk
        assert "bar_ts=bar_ts" in blk
        assert "jn.decide(t, s, px, side, taken)" in blk

    def test_the_panel_shows_what_the_candle_has_had(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("function paintJournal(")
        blk = src[i:i + 9000]
        assert "taken_this_candle" in blk
        assert "no more until the next one" in blk


class TestItOpensADatabaseFromAnEarlierBuild:
    """Chris, on deploying: OperationalError, table journal_trades
    has no column named bar_ts. The trade button stopped working
    entirely and nothing could be logged.

    CREATE TABLE IF NOT EXISTS DOES NOT ALTER AN EXISTING TABLE. The
    schema gained `bar_ts`, his database was written before it, and
    every insert failed. The store next door (ofistore) has carried a
    migration for exactly this since it shipped; this one did not,
    because I added a column and only ever ran it against a database
    this build had created.
    """

    OLD = """CREATE TABLE journal_trades (
        id INTEGER PRIMARY KEY, coin TEXT NOT NULL, side TEXT NOT NULL,
        mode TEXT NOT NULL DEFAULT 'manual', opened_ts REAL NOT NULL,
        open_px REAL NOT NULL, closed_ts REAL, close_px REAL,
        why TEXT NOT NULL DEFAULT '', tp REAL NOT NULL DEFAULT 0,
        sl REAL NOT NULL DEFAULT 0, note TEXT NOT NULL DEFAULT '')"""

    def _old_db(self, tmp_path, rows=()):
        import sqlite3

        path = tmp_path / "old.db"
        c = sqlite3.connect(path)
        c.execute(self.OLD)
        for coin, side, op, cl in rows:
            c.execute("INSERT INTO journal_trades (coin, side,"
                      " opened_ts, open_px, closed_ts, close_px, why)"
                      " VALUES (?,?,1.0,?,2.0,?,'manual')",
                      (coin, side, op, cl))
        c.commit()
        c.close()
        return path

    def test_it_opens_without_raising(self, tmp_path):
        from liqmap.journal import Journal

        Journal(self._old_db(tmp_path))

    def test_a_trade_can_be_logged_again(self, tmp_path):
        """The symptom: every insert failed."""
        from liqmap.journal import Journal

        j = Journal(self._old_db(tmp_path))
        assert j.open_trade("ES.Z26", "down", 6700.0,
                            bar_ts=99.0) is not None

    def test_the_old_trades_are_kept(self, tmp_path):
        """A migration that drops the record it was migrating is
        worse than the error it replaced."""
        from liqmap.journal import Journal

        j = Journal(self._old_db(
            tmp_path, [("ES.Z26", "up", 6700.0, 6701.0)]))
        assert j.stats("ES.Z26").n == 1
        assert j.stats("ES.Z26").ticks == 4.0

    def test_a_pre_migration_trade_belongs_to_no_candle(self, tmp_path):
        """It reads back with the default, zero, and zero means "no
        candle recorded" -- which is what those trades honestly are.
        Inventing an id would put them under a cap they were never
        subject to."""
        from liqmap.journal import Journal

        j = Journal(self._old_db(
            tmp_path, [("ES.Z26", "up", 6700.0, 6701.0)]))
        assert j.closed("ES.Z26")[0].bar_ts == 0.0
        assert j.sides_in_bar("ES.Z26", 0.0) == ()

    def test_the_cap_works_on_the_migrated_table(self, tmp_path):
        from liqmap.journal import Journal

        j = Journal(self._old_db(tmp_path))
        j.open_trade("ES.Z26", "up", 6700.0, bar_ts=99.0)
        j.close_trade("ES.Z26", 6701.0)
        assert j.sides_in_bar("ES.Z26", 99.0) == ("up",)

    def test_migrating_twice_is_harmless(self, tmp_path):
        from liqmap.journal import Journal

        path = self._old_db(tmp_path)
        Journal(path)
        j = Journal(path)
        assert j.open_trade("ES.Z26", "up", 6700.0,
                            bar_ts=1.0) is not None

    def test_every_added_column_is_declared(self):
        """The list is what the migration walks. A column added to
        SCHEMA and not to `_ADDED` is this bug again."""
        import re

        from liqmap.journal import SCHEMA, _ADDED

        body = SCHEMA[SCHEMA.index("journal_trades"):]
        body = body[:body.index(");")]
        cols = set(re.findall(r"^\s{4}(\w+)\s", body, re.M))
        declared = {n for n, _ in _ADDED}
        assert declared <= cols, declared - cols
        assert "bar_ts" in declared
