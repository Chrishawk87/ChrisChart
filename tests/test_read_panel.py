"""The trade column must not die with the ladder.

Chris: "their is something wrong with the trades on this rules
section of the chart it doesnt show the trades and i cant see if its
auto trading either".

TWO HALVES OF ONE BUG, and both had to be fixed:

    THE ROUTE. `/api/read` has three early returns -- the CME tape
    has under ten closed bars, the candle fetch failed, too few
    candles came back -- and each returned a payload with no
    `journal` key at all.

    THE PAGE. `paintRead` returned on `d.error` BEFORE it reached
    `paintJournal`, so even a payload that did carry the column was
    never drawn.

The first of those three returns is the UNREAD condition, which is
most of the first two hours after every restart. So the column
disappeared exactly when it was most needed, and an empty column
reads as a broken strategy rather than a stack that is still
filling.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def py_func(name: str, path: str = "liqmap/web.py") -> str:
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


def js_func(name: str) -> str:
    src = (ROOT / "liqmap" / "web.py").read_text()
    i = src.index(f"function {name}(")
    depth, j = 0, src.index("{", i)
    for k in range(j, len(src)):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return src[i:k + 1]
    raise AssertionError(f"{name} has no closing brace")


@pytest.fixture
def client(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import liqmap.web as web

    monkeypatch.setenv("LIQMAP_DB", str(tmp_path / "web.db"))
    monkeypatch.setenv("LIQMAP_TOKEN", "read-panel-token")
    monkeypatch.setenv("LIQMAP_AUTO", "false")
    monkeypatch.delenv("LIQMAP_WALLETS", raising=False)
    web.RT = None
    with TestClient(web.create_app()) as c:
        c.headers.update({"Authorization": "Bearer read-panel-token"})
        yield c
    web.RT = None


class TestTheColumnSurvivesAFailedLadder:
    def test_the_payload_still_carries_it(self, client):
        """The ladder cannot build here -- no market data reaches
        this container -- which is the same shape as the UNREAD
        case on the live service.
        """
        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"}).json()
        assert d.get("error"), "the ladder was expected to fail here"
        assert d.get("journal") is not None, \
            "the column went out with the ladder"

    def test_auto_is_visible_even_then(self, client):
        """"i cant see if its auto trading either"."""
        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m"}).json()
        s = (d.get("journal") or {}).get("settings") or {}
        assert "auto" in s
        assert isinstance(s["auto"], bool)

    def test_the_strategy_is_visible_even_then(self, client):
        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m"}).json()
        s = (d.get("journal") or {}).get("settings") or {}
        assert s.get("strategy")
        assert s.get("strategy_label")

    def test_the_stats_are_visible_even_then(self, client):
        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m"}).json()
        j = d.get("journal") or {}
        assert "stats" in j
        assert "recent" in j

    def test_the_column_says_why_it_is_idle(self, client):
        """An empty column with no reason is what reads as broken."""
        j = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m"}
                       ).json().get("journal") or {}
        assert (j.get("blocked") or j.get("idle") or j.get("error")), \
            "silent and empty is the failure being fixed"

    def test_the_column_itself_did_not_error(self, client):
        j = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m"}
                       ).json().get("journal") or {}
        assert j.get("error") is None, j.get("error")


class TestTheWiring:
    def test_there_is_one_helper_for_it(self):
        """Three early returns each attaching the column by hand is
        three places for the next one to be forgotten."""
        src = (ROOT / "liqmap" / "web.py").read_text()
        assert "def _with_journal(" in src

    def test_every_early_return_uses_it(self):
        blk = py_func("api_read")
        bare = [ln for ln in blk.split("\n")
                if ln.strip().startswith("return {**out")]
        assert not bare, f"an early return skips the column: {bare}"

    def test_the_three_known_ones_are_wrapped(self):
        blk = py_func("api_read")
        assert blk.count("_with_journal(") >= 3

    def test_the_helper_cannot_take_the_route_down(self):
        """A column that raises must not become a 500 on the ladder."""
        blk = py_func("_with_journal")
        assert "except Exception" in blk
        assert '"error"' in blk

    def test_the_page_draws_it_on_the_error_path(self):
        """paintRead returned before paintJournal, so a payload that
        did carry the column was still never drawn."""
        blk = js_func("paintRead")
        i = blk.index("d.error")
        j = blk.index("return;", i)
        assert "paintJournal(d)" in blk[i:j], \
            "the error branch returns without drawing the column"

    def test_it_is_still_drawn_on_the_happy_path(self):
        blk = js_func("paintRead")
        assert blk.count("paintJournal(d)") >= 2


class TestTheHandTradingAlert:
    """Chris cannot get an Ironbeam API key until the account is
    funded: "what if we just send the signals to Iron beam and trade
    the signals that are sent until we have the 1000 dollars".

    So the rule has to say what it WOULD do, loudly, and the exit
    matters more than the entry -- a missed entry costs a trade, a
    missed exit leaves a position nobody is managing.
    """

    def test_the_payload_carries_advice(self, client):
        j = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m"}
                       ).json().get("journal") or {}
        a = j.get("advice")
        assert a is not None
        for k in ("act", "why", "side", "holding", "key"):
            assert k in a, k

    def test_advice_speaks_with_auto_off(self):
        """The acting path refuses; the advice still answers."""
        from liqmap.journal import Settings, decide
        s = Settings(coin="ES", auto=False, strategy="fused")
        assert decide(None, s, 6800.0, "up", taken=(),
                      max_per_bar=1)[0] == ""
        assert decide(None, s, 6800.0, "up", taken=(), max_per_bar=1,
                      advisory=True) == ("open", "up")

    def test_advice_calls_the_exit_on_a_hand_taken_row(self):
        """The acting path leaves a manual row alone -- "a trade he
        took by hand is his to end" -- but he still has to be told."""
        from liqmap.journal import Settings, Trade, decide
        s = Settings(coin="ES", auto=False, strategy="fused")
        t = Trade(id=1, coin="ES", side="up", mode="manual",
                  open_px=6800.0, strategy="fused")
        assert decide(t, s, 6800.0, "",
                      disagreed="the 5m turned down against a up")[0] == ""
        assert decide(t, s, 6800.0, "",
                      disagreed="the 5m turned down against a up",
                      advisory=True) == ("close", "disagree")

    def test_advice_still_respects_the_cap(self):
        """Advisory answers a different question, not a laxer one."""
        from liqmap.journal import Settings, decide
        s = Settings(coin="ES", auto=False, strategy="fused")
        assert decide(None, s, 6800.0, "up", taken=("up",),
                      max_per_bar=1, advisory=True)[0] == ""

    def test_it_is_the_same_function_not_a_second_rule(self):
        """A second implementation would let the alert advertise a
        rule the column does not follow, and he would find out which
        one was right while holding a position."""
        blk = py_func("_advice")
        assert "jn.decide(" in blk
        assert "advisory=True" in blk

    def test_the_key_distinguishes_an_entry_from_an_exit(self):
        blk = py_func("_advice")
        assert 'f"close:{t.id}:{why}"' in blk
        assert 'f"open:{bar_ts:.0f}:{side}"' in blk

    def test_the_alert_refuses_to_repeat_itself(self):
        blk = js_func("checkFusionAlert")
        assert "a.key === fusAlert.last" in blk

    def test_the_exit_sounds_different_from_the_entries(self):
        blk = js_func("fusBeep")
        assert "[880, 880, 880]" in blk

    def test_the_exit_says_which_way_to_trade(self):
        """"SELL to close your LONG" rather than a bare direction --
        under time pressure the translation is where mistakes live."""
        blk = js_func("checkFusionAlert")
        assert "SELL to close your LONG" in blk
        assert "BUY to close your SHORT" in blk

    def test_it_asks_for_the_real_fill(self):
        """The signal price is where the rule fired, not where he was
        filled. Scoring the first as the second flatters every number
        on the panel."""
        blk = js_func("checkFusionAlert")
        assert "log the fill" in blk
        blk2 = js_func("fusLogFill")
        assert "/api/journal/trade" in blk2

    def test_the_logger_uses_the_routes_own_vocabulary(self):
        blk = js_func("fusLogFill")
        assert "action: 'open', side: side" in blk
        assert "action: 'close'" in blk

    def test_the_alert_is_drawn_from_the_column_poll(self):
        blk = js_func("paintJournal")
        assert "checkFusionAlert(" in blk


class TestTheCapLeakAfterAClose:
    def test_the_re_decide_passes_the_cap(self):
        """It did not, so after a close the default of two per candle
        applied: the just-closed side was blocked and the OPPOSITE one
        was free. The rule could turn round inside the same 5-minute
        bar -- the "immediately jumps in for another trade" shape,
        surviving in the one place the cap was not threaded through.
        """
        blk = py_func("_journal_step_un")
        i = blk.index("if act == \"close\":")
        j = blk.index("if act == \"open\":", i)
        assert "max_per_bar=cap" in blk[i:j], \
            "the post-close decide must carry the cap"

    def test_both_decides_carry_it(self):
        blk = py_func("_journal_step_un")
        assert blk.count("max_per_bar=cap") >= 2


class TestTheRuleHasAClock:
    """Chris: "the trade on this rule is behind real time data
    completely by several points at a time".

    It was not late, it was ASLEEP. `/api/read` is the only thing
    that runs the trade column -- entry test, exit test, cap, alert
    -- and nothing polled it. The 10-second poll had been removed on
    the reasoning that "/ws/live already pushes the forming bar",
    which is true and beside the point: the socket updates what is
    DRAWN, it does not evaluate the rule.

    So the strategy stepped only when something happened to call
    loadRead -- a click, a page load, a dropdown -- and the first
    poll after a click acted on a signal that had fired minutes
    earlier at a price long gone. Several points, not several ticks.
    """

    def test_the_read_route_is_polled(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        assert "_readBusy" in src
        i = src.index("let _readBusy")
        blk = src[i:i + 400]
        assert "await loadRead()" in blk
        assert "}, 1000);" in blk

    def test_it_does_not_stack_requests(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("let _readBusy")
        blk = src[i:i + 400]
        assert "if (_readBusy) return;" in blk
        assert "finally { _readBusy = false; }" in blk

    def test_it_does_not_depend_on_the_auto_checkbox(self):
        """The alert needs the rule stepping even with AUTO off --
        that is the whole hand-trading path."""
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("let _readBusy")
        blk = src[i:i + 400]
        assert "rAuto" not in blk

    def test_the_toggle_no_longer_claims_the_socket_runs_the_rule(self):
        blk = js_func("toggleReadAuto")
        assert "ws/live already push" not in blk

    def test_the_journal_step_is_only_reachable_through_that_route(self):
        """If this ever grows a second caller, the comment above the
        poll stops being true and the clock stops being the clock."""
        src = (ROOT / "liqmap" / "web.py").read_text()
        # The def line matches the same text; count call sites only.
        calls = src.count("_journal_step(rt") - src.count(
            "def _journal_step(rt")
        assert calls == 2, f"{calls} call sites -- expected the route " \
            "and the give-up path, and nothing else"
        # And both of them are the read route: the direct call, and
        # the one inside `_with_journal` that the early returns use.
        assert "out[\"journal\"] = _journal_step(rt, coin, out)" in src
        assert "payload[\"journal\"] = _journal_step(rt_, coin_, payload)" \
            in src
