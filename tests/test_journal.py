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
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("def _journal_step")
        assert '"places_orders": False' in src[i:i + 3000]


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

    def test_disagreement(self):
        assert should_close(self._open(), self.S, 6700.0,
                            "down") == "disagree"
        assert should_close(self._open(), self.S, 6700.0,
                            "") == "disagree"

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

    def test_an_auto_row_closes_on_disagreement(self):
        t = Trade(side="up", open_px=6700.0, mode="auto")
        assert decide(t, self.OPEN, 6700.0, "down") == ("close",
                                                        "disagree")

    def test_a_manual_row_does_not(self):
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

    def test_the_page_draws_the_column(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        assert 'id="jnlBox"' in src
        i = src.index("function paintJournal(")
        blk = src[i:i + 6000]
        for want in ("jnlStats", "jnlOpen", "jnlRecent", "jnlAgree",
                     "beats_costs"):
            assert want in blk, want
        # And it is actually called.
        assert "paintJournal(d);" in src
