"""TWO TICK CHARTS, ONE LEADING AND ONE CONFIRMING.

Chris: "lets build a strategy that is just like the current
strategy but just between gate 2 and gate 3 using gate 2 as the
lead and gate 3 as the confirmation with the same 5 second hold on
gate 3 before a trade is taken".

Asked what should close a trade, he picked "Gate 2's flip only, no
clock"; asked where the no-trade band should sit with no timeframe
to anchor it, "Drop it entirely". Both of those are absences, and
an absence is the hardest thing to keep: nothing fails when a
clock quietly comes back. There are tests here for each one NOT
being there.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from liqmap import twotick as tt

D = "down"
U = "up"
SRC = pathlib.Path(__file__).resolve().parents[1] / "liqmap"


class TestItOnlySuggests:
    """THE TOOL SUGGESTS, IT DOES NOT EXECUTE. Every module that
    can reach the journal carries this test."""

    def test_nothing_here_is_named_like_an_order(self):
        tree = ast.parse((SRC / "twotick.py").read_text())
        names = {n.name for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef,
                                   ast.AsyncFunctionDef))}
        for bad in ("submit", "execute", "place", "order"):
            assert bad not in names, f"twotick.py defines {bad}()"

    def test_it_reaches_no_broker(self):
        src = (SRC / "twotick.py").read_text().lower()
        for bad in ("ironbeam", "requests.", "httpx", "urllib",
                    "socket"):
            assert bad not in src, f"twotick.py mentions {bad}"


class TestTheEntry:
    @pytest.mark.parametrize("side", [U, D])
    def test_both_agreeing_and_held_opens(self, side):
        assert tt.entry_side(side, side, held_s=9.0) == (side, "")

    @pytest.mark.parametrize("side", [U, D])
    def test_the_confirmation_against_blocks(self, side):
        other = D if side == U else U
        got, why = tt.entry_side(side, other, held_s=9.0)
        assert got == ""
        assert "confirming" in why

    def test_a_blank_lead_blocks_and_names_the_lead(self):
        got, why = tt.entry_side("", U, held_s=9.0)
        assert got == ""
        assert "lead" in why
        assert "no direction" in why

    def test_a_blank_confirmation_blocks_and_names_it(self):
        """AN ABSENCE IS NOT A DISAGREEMENT. Read as one it
        produces "the lead chart is up but the confirming chart
        is" -- a blank where a side should be."""
        got, why = tt.entry_side(U, "", held_s=9.0)
        assert got == ""
        assert "confirming" in why
        assert "no direction" in why
        assert "but" not in why

    def test_the_hold_is_on_the_confirmation(self):
        got, why = tt.entry_side(U, U, held_s=1.0, min_hold_s=5.0)
        assert got == ""
        assert "confirming chart turned up 1s ago" in why
        assert tt.entry_side(U, U, held_s=5.0, min_hold_s=5.0) == (U, "")

    def test_the_hold_defaults_to_five_seconds(self):
        assert tt.DWELL_S == 5.0
        assert tt.entry_side(U, U, held_s=4.9)[0] == ""
        assert tt.entry_side(U, U, held_s=5.0)[0] == U

    def test_an_unknown_hold_does_not_block(self):
        """`None` means "nobody is tracking it", which must not
        read as "zero seconds"."""
        assert tt.entry_side(U, U, held_s=None) == (U, "")

    def test_one_trade_per_agreement(self):
        got, why = tt.entry_side(U, U, held_s=9.0, episode_taken=True)
        assert got == ""
        assert "already been traded" in why


class TestTheExit:
    """"Gate 2's flip only, no clock"."""

    @pytest.mark.parametrize("side,against", [(U, D), (D, U)])
    def test_the_lead_flipping_ends_it(self, side, against):
        why = tt.exit_reason(side, against)
        assert why
        assert "lead chart" in why
        assert "tick bar" not in why

    def test_the_lead_agreeing_holds_it(self):
        assert tt.exit_reason(U, U) == ""

    def test_a_flat_lead_is_not_a_flip(self):
        """FLAT IS NOT A FLIP -- the most expensive mistake this
        engine has made, three times over."""
        assert tt.exit_reason(U, "") == ""

    def test_the_confirmation_is_silent_unless_switched_on(self):
        assert tt.exit_reason(U, U, D) == ""
        assert tt.exit_reason(U, U, D, False) == ""

    def test_switched_on_the_confirmation_closes_it_too(self):
        why = tt.exit_reason(U, U, D, True)
        assert "confirming chart" in why

    def test_switched_on_a_flat_confirmation_is_still_not_a_flip(self):
        assert tt.exit_reason(U, U, "", True) == ""

    def test_the_lead_keeps_the_blame_when_both_flip(self):
        why = tt.exit_reason(U, D, D, True)
        assert "lead chart" in why
        assert "confirming" not in why


class TestThereIsNoClockAndNoBand:
    """TWO ABSENCES HE ASKED FOR BY NAME. Nothing fails when a
    clock quietly comes back, so each one is asserted."""

    def test_no_amount_of_elapsed_time_ends_a_trade(self):
        import inspect
        sig = inspect.signature(tt.exit_reason)
        for bad in ("bar_ts", "now_ts", "period_s"):
            assert bad not in sig.parameters, \
                f"exit_reason takes {bad} -- a clock is back"
        assert tt.exit_reason(U, U) == ""

    def test_the_entry_asks_nothing_about_an_open_price(self):
        import inspect
        sig = inspect.signature(tt.entry_side)
        for bad in ("tf_ticks", "min_open_ticks", "ticks"):
            assert bad not in sig.parameters, \
                f"entry_side takes {bad} -- the band is back"

    def test_it_reads_no_timeframe_at_all(self):
        src = (SRC / "twotick.py").read_text()
        body = src[src.index("def entry_side("):]
        for bad in ("window_s", "stack_side", "rungs", "window_label"):
            assert bad not in body, \
                f"the rule reads {bad} -- it has no timeframe"

    def test_the_cap_survives_though(self):
        """The five-minute stamp stays, as a COUNTER. With no clock
        on the trade and no band off an open it is the only thing
        bounding how often this can churn."""
        assert tt.MAX_TRADES_PER_BAR == 2
        assert tt.CAP_BAR_S == 300.0
        a = tt.bar_of(1_700_000_000.0, tt.CAP_BAR_S)
        assert tt.bar_of(a + 299.0, tt.CAP_BAR_S) == a
        assert tt.bar_of(a + 301.0, tt.CAP_BAR_S) != a


class TestTheState:
    def test_it_holds_each_chart_through_a_tie(self):
        out = tt.state(lead_raw=U, confirm_raw=U, lead_bar=5000,
                       confirm_bar=1000, held_s=9.0)
        assert out["side"] == U
        nxt = tt.state(lead_raw="", confirm_raw="", lead_bar=5000,
                       confirm_bar=1000, held_s=9.0,
                       prev=out["held"])
        assert nxt["lead"]["side"] == U
        assert nxt["confirm"]["side"] == U
        assert nxt["lead"]["held_through_flat"] is True

    def test_a_fresh_reading_with_no_memory_is_flat_not_wrong(self):
        out = tt.state()
        assert out["side"] == ""
        assert "lead" in out["why"]

    def test_the_hold_countdown_reaches_the_panel(self):
        out = tt.state(lead_raw=U, confirm_raw=U, lead_bar=5000,
                       confirm_bar=1000, held_s=2.0, min_hold_s=5.0)
        assert out["side"] == ""
        assert out["confirm"]["ready"] is False
        assert out["confirm"]["held_s"] == 2.0
        assert out["confirm"]["min_hold_s"] == 5.0

    def test_the_sizes_reach_the_panel(self):
        out = tt.state(lead_raw=U, confirm_raw=U, lead_bar=5000,
                       confirm_bar=250, held_s=9.0)
        assert out["lead"]["per_bar"] == 5000
        assert out["confirm"]["per_bar"] == 250
        assert out["confirm"]["same_as_lead"] is False

    def test_the_same_size_twice_is_called_out(self):
        out = tt.state(lead_raw=U, confirm_raw=U, lead_bar=2000,
                       confirm_bar=2000, held_s=9.0)
        assert out["confirm"]["same_as_lead"] is True

    def test_the_switch_reaches_the_panel(self):
        off = tt.state(lead_raw=U, confirm_raw=U, held_s=9.0)
        on = tt.state(lead_raw=U, confirm_raw=U, held_s=9.0,
                      confirm_exits=True)
        assert off["confirm"]["exits"] is False
        assert on["confirm"]["exits"] is True
        assert "confirming chart flips" not in off["say"]
        assert "confirming chart flips" in on["say"]

    def test_an_agreement_is_spent_once_and_not_before(self):
        out = tt.state(lead_raw=U, confirm_raw=U, held_s=9.0)
        assert out["side"] == U
        assert out["episode"]["taken"] is False
        held = {**out["held"],
                "episode": {**out["held"]["episode"], "taken": True}}
        again = tt.state(lead_raw=U, confirm_raw=U, held_s=9.0,
                         prev=held)
        assert again["side"] == ""
        assert "already been traded" in again["why"]

    def test_a_disagreement_frees_the_next_agreement(self):
        out = tt.state(lead_raw=U, confirm_raw=U, held_s=9.0)
        held = {**out["held"],
                "episode": {**out["held"]["episode"], "taken": True}}
        apart = tt.state(lead_raw=U, confirm_raw=D, held_s=9.0,
                         prev=held)
        again = tt.state(lead_raw=U, confirm_raw=U, held_s=9.0,
                         prev=apart["held"])
        assert again["side"] == U

    def test_every_field_the_next_reading_needs_rides_home(self):
        out = tt.state(lead_raw=U, confirm_raw=U, held_s=9.0)
        held = out["held"]
        assert set(held) >= {"lead", "confirm", "episode"}
        assert set(held["episode"]) >= {"side", "taken"}


class TestItIsWiredThrough:
    """The rule can be right and nothing reach it. Every one of
    these has been broken at least once in this engine: a payload
    key nothing populated, a strategy the dropdown never offered, a
    panel built and never shown."""

    def _web(self):
        return (SRC / "web.py").read_text()

    def _step(self):
        src = self._web()
        import re
        i = src.index("def _journal_step_un(")
        col = len(src[:i].split("\n")[-1])
        m = re.compile(r"\n {0,%d}(?! )\S" % col).search(src, i + 10)
        return src[i:m.start()] if m else src[i:]

    def test_it_is_a_registered_strategy(self):
        from liqmap.journal import STRATEGIES
        assert "ticks" in STRATEGIES
        assert "gate 2" in STRATEGIES["ticks"] or \
            "lead" in STRATEGIES["ticks"]

    def test_it_survives_being_saved(self, tmp_path):
        from liqmap.journal import Journal, Settings
        j = Journal(tmp_path / "j.db")
        j.save_settings(Settings(coin="ES", strategy="ticks"))
        assert j.settings("ES").strategy == "ticks", \
            "an unregistered strategy is silently rewritten to ofi"

    def test_the_dropdown_offers_it(self):
        assert 'value="ticks"' in self._web(), \
            "the strategy exists and cannot be picked"

    def test_the_signal_does_not_wait_on_a_window(self):
        """IT HAS NO TIMEFRAME. Refused for "no window yet" it
        would never fire at all."""
        from liqmap.journal import window_signal
        side, why = window_signal({}, "ticks", "",
                                  {"side": "up", "why": ""})
        assert side == "up", why
        side, why = window_signal({}, "ticks", "", {})
        assert side == ""
        assert "window" not in why

    def test_the_step_builds_it(self):
        blk = self._step()
        assert 'strat == "ticks"' in blk, "the route never builds it"
        assert "_tt.state(" in blk
        assert '"twotick"' in blk, "it never reaches the payload"

    def test_the_step_reads_both_tick_sizes(self):
        blk = self._step()
        assert 'getattr(s, "tick_bar", 0)' in blk
        assert 'getattr(s, "tick_bar_3", 0)' in blk
        assert "_lead_n not in _PBCT" in blk, \
            "a size with no builder behind it is not snapped"
        assert "_conf_n not in _PBCT" in blk

    def test_the_hold_runs_on_the_confirmation_not_the_lead(self):
        """Chris: "the same 5 second hold on gate 3"."""
        blk = self._step()
        i = blk.index("_tdw.on_tick(")
        seg = blk[max(0, i - 400):i + 80]
        assert "_conf_held" in seg
        assert "_conf.get" in seg
        assert "_lead.get" not in seg, \
            "the hold is being fed the lead chart"

    def test_it_keeps_its_own_dwell_and_memory(self):
        """SHARING THE STACK RULE'S WOULD CARRY A HOLD ACROSS A
        STRATEGY SWITCH -- from a chart this rule never read."""
        blk = self._step()
        assert "_ticks_dwell" in blk
        assert "_ticks_held" in blk
        assert "_tick_dwell" not in blk.split('strat == "ticks"')[1][:3000]

    def test_the_exit_is_handed_what_state_resolved(self):
        blk = self._step()
        i = blk.index("_tte.exit_reason(")
        seg = blk[i:i + 500]
        assert '(fus.get("lead") or {}).get("side")' in seg
        assert '(fus.get("confirm") or {}).get("side")' in seg
        assert 'getattr(s, "tick3_exit", False)' in seg

    def test_the_exit_consults_no_clock(self):
        """"Gate 2's flip only, no clock"."""
        blk = self._step()
        i = blk.index("_tte.exit_reason(")
        seg = blk[i:i + 500]
        assert "time.time()" not in seg, "a clock reached the exit"
        assert "TRADE_BAR_S" not in seg

    def test_the_agreement_is_spent_only_when_a_trade_opens(self):
        blk = self._step()
        j = blk.index('if act == "open":')
        seg = blk[j:blk.index("\n        st = j.stats(", j)]
        assert "_ticks_held" in seg
        assert '"taken": True' in seg

    def test_the_cap_still_bounds_it(self):
        blk = self._step()
        assert "_ttb.MAX_TRADES_PER_BAR" in blk
        assert "_ttb.bar_of(" in blk

    def test_the_panel_is_drawn_and_shown(self):
        src = self._web()
        assert 'id="jnlTwoTick"' in src, "the box does not exist"
        assert "function paintTwoTick(" in src
        assert "paintTwoTick(J.twotick)" in src, \
            "the painter is written and never called"
        i = src.index("function paintTwoTick(")
        fn = src[i:src.index("\nfunction ", i + 10)]
        assert "TT.lead" in fn and "TT.confirm" in fn
        assert "C.exits" in fn, \
            "the panel cannot say whether the confirmation closes trades"
        assert "No clock ends a trade" in fn, \
            "the panel does not say the clock is gone"


from tests.test_cme_panels import _cme, client  # noqa: E402,F401

AUTH = {"Authorization": "Bearer t0ken"}


class TestItActuallySteps:
    """EXECUTION, NOT SOURCE TEXT. A source-text test cannot see an
    UnboundLocalError, and this route has produced three of them --
    each one broke every window and was swallowed into
    `journal.error`, where nothing was looking."""

    @pytest.mark.parametrize("lead,conf,exit_on", [
        (5000, 1000, False), (250, 250, False),
        (10000, 250, True), (1000, 5000, True),
    ])
    def test_the_step_runs_without_raising(
            self, client, monkeypatch, lead, conf, exit_on):
        from liqmap import web
        _cme(monkeypatch)
        rt = web.runtime()
        s = rt.journal.settings("ES")
        s.strategy = "ticks"
        s.tick_bar = lead
        s.tick_bar_3 = conf
        s.tick3_exit = exit_on
        rt.journal.save_settings(s)
        assert rt.journal.settings("ES").strategy == "ticks", \
            "the strategy did not survive the save"

        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH).json()
        jr = d.get("journal") or {}
        assert "error" not in jr, (
            f"the step raised at {lead}/{conf}: {jr.get('error')}")

    @pytest.mark.parametrize("window_s", [60.0, 300.0, 900.0, -5000.0])
    def test_the_window_setting_cannot_break_it(
            self, client, monkeypatch, window_s):
        """IT HAS NO TIMEFRAME, so every window must be harmless --
        including the tick windows, which are stored negative."""
        from liqmap import web
        _cme(monkeypatch)
        rt = web.runtime()
        s = rt.journal.settings("ES")
        s.strategy = "ticks"
        s.window_s = window_s
        rt.journal.save_settings(s)
        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH).json()
        jr = d.get("journal") or {}
        assert "error" not in jr, (
            f"the step raised on a {window_s}s window: "
            f"{jr.get('error')}")

    def test_the_payload_describes_both_charts(self, client, monkeypatch):
        """Not merely "it did not raise": the panel has to be able
        to name what was read, or the setting did nothing."""
        from liqmap import web
        _cme(monkeypatch)
        rt = web.runtime()
        s = rt.journal.settings("ES")
        s.strategy = "ticks"
        s.tick_bar = 3000
        s.tick_bar_3 = 500
        s.tick3_exit = True
        rt.journal.save_settings(s)
        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH).json()
        tt_ = ((d.get("journal") or {}).get("twotick")) or {}
        assert tt_, "the two-tick payload is missing entirely"
        assert tt_.get("mode") == "ticks"
        assert (tt_.get("lead") or {}).get("per_bar") == 3000
        assert (tt_.get("confirm") or {}).get("per_bar") == 500
        assert (tt_.get("confirm") or {}).get("exits") is True
        assert (tt_.get("confirm") or {}).get("min_hold_s") == 5.0

    def test_the_other_strategies_still_get_their_own_panel(
            self, client, monkeypatch):
        """A NEW PAYLOAD KEY IS WHERE AN OLD ONE GETS KNOCKED OUT.
        On `stack` the two-tick box must be absent, and the other
        way round."""
        from liqmap import web
        _cme(monkeypatch)
        rt = web.runtime()
        s = rt.journal.settings("ES")
        s.strategy = "stack"
        rt.journal.save_settings(s)
        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH).json()
        jr = d.get("journal") or {}
        assert jr.get("stackwin"), "the stack panel went missing"
        assert jr.get("twotick") is None, \
            "the two-tick panel is drawn under the stack rule"

        s = rt.journal.settings("ES")
        s.strategy = "ticks"
        rt.journal.save_settings(s)
        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH).json()
        jr = d.get("journal") or {}
        assert jr.get("twotick"), "the two-tick panel is missing"
        assert jr.get("stackwin") is None, \
            "the stack panel is drawn under the two-tick rule"

    def test_the_memory_rides_home_between_polls(
            self, client, monkeypatch):
        """WITHOUT IT, NOTHING HOLDS. These functions are pure --
        the caller keeps the previous reading and hands it back.
        Drop that one assignment and both charts blank at every tie,
        the hold restarts on an absence, and an agreement is never
        spent. Nothing raises; it just quietly stops working, which
        is why this is an assertion and not a comment."""
        from liqmap import web
        _cme(monkeypatch)
        rt = web.runtime()
        s = rt.journal.settings("ES")
        s.strategy = "ticks"
        rt.journal.save_settings(s)
        try:
            del rt._ticks_held
        except AttributeError:
            pass

        for _ in range(2):
            client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH)
        held = getattr(rt, "_ticks_held", None)
        assert held, "the previous reading is never kept"
        assert set(held) >= {"lead", "confirm", "episode"}, held

    def test_a_held_direction_survives_a_blank_bar(
            self, client, monkeypatch):
        """THE POINT OF KEEPING IT. A direction put into the memory
        comes back out on the next poll even when the bar itself
        reads flat -- an absence is not a reversal."""
        from liqmap import web
        _cme(monkeypatch)
        rt = web.runtime()
        s = rt.journal.settings("ES")
        s.strategy = "ticks"
        rt.journal.save_settings(s)
        rt._ticks_held = {"lead": "up", "confirm": "up",
                          "episode": {"side": "", "taken": False}}
        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH).json()
        tt_ = ((d.get("journal") or {}).get("twotick")) or {}
        assert tt_, "the two-tick payload is missing entirely"
        assert (tt_.get("lead") or {}).get("side") == "up", \
            "the held direction was dropped on the way in"
