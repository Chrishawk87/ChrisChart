"""Volume-sized footprint bars, graded on push and activity.

Chris: "what if we took a actual foot print chart and use that to
build actual candles" -- then, asked what should close one: "the bar
should mark push and activity and should be just up strong is strong
buy push up, if normal push just buy , sell normal push down strong
down if volume pushes activity".
"""

from __future__ import annotations

import pathlib

import pytest

from liqmap import footread as fo

ROOT = pathlib.Path(__file__).resolve().parents[1]
AUTH = {"Authorization": "Bearer t0ken"}


class TestItSuggestsAndNeverExecutes:
    def test_no_function_is_named_as_though_it_traded(self):
        src = (ROOT / "liqmap" / "footread.py").read_text()
        for bad in ("def submit(", "def execute(", "def place(",
                    "def order(", "def buy(", "def sell("):
            assert bad not in src, f"footread has {bad}"

    def test_every_reading_says_it_decides_nothing(self):
        assert fo.read(900, 3000, 40, 180)["decides"] is False
        assert fo.read()["decides"] is False


class TestTheFourGrades:
    """"strong is strong buy push up, if normal push just buy, sell
    normal push down strong down"."""

    def test_one_sided_and_fast_is_a_strong_buy(self):
        r = fo.read(delta=900, volume=3000, seconds=40, median_s=180)
        assert r["call"] == "STRONG BUY"
        assert r["side"] == "up" and r["grade"] == "strong"

    def test_one_sided_at_an_ordinary_pace_is_a_buy(self):
        r = fo.read(delta=900, volume=3000, seconds=200, median_s=180)
        assert r["call"] == "BUY"
        assert r["grade"] == "normal"

    def test_the_same_two_downward(self):
        assert fo.read(-900, 3000, 40, 180)["call"] == "STRONG SELL"
        assert fo.read(-900, 3000, 200, 180)["call"] == "SELL"

    def test_a_two_sided_bar_gets_no_call(self):
        """A GRADE NOBODY WOULD ACT ON IS NOISE on a panel that
        already has plenty."""
        r = fo.read(delta=120, volume=3000, seconds=40, median_s=180)
        assert r["call"] == ""
        assert r["side"] == ""
        assert "two-sided" in r["say"]

    def test_zero_delta_is_never_a_side(self):
        r = fo.read(delta=0, volume=3000, seconds=10, median_s=180)
        assert r["side"] == ""


class TestVolumeClosesItAndDeltaGradesIt:
    """THE TRAP THIS WHOLE LINE OF WORK WAS STUCK IN. Making the
    bar wait for a delta threshold sounds right and is the fused
    bar's OFI leg again: delta resets every bar and sits near zero
    in churn, so the bar does not close, the cap closes it, and a
    cap that does the closing is a clock."""

    def test_the_module_does_not_close_bars_at_all(self):
        src = (ROOT / "liqmap" / "footread.py").read_text()
        for bad in ("DeltaRule", "class CloseRule", "def done("):
            assert bad not in src, f"footread tries to close bars: {bad}"

    def test_the_gate_closes_on_volume(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("def _foot_stream(")
        fn = src[i:src.index("\n    def ", i + 10)]
        assert "VolumeRule" in fn, "the footprint bar is not volume-sized"
        assert "DeltaRule" not in fn, \
            "delta closes the bar -- it will cap in churn"

    def test_the_share_is_measured_against_the_bar(self):
        """400 is a shove on a 2,000-contract bar and a rounding
        error on a 40,000-contract one."""
        small = fo.read(delta=400, volume=2000, seconds=40, median_s=180)
        big = fo.read(delta=400, volume=40000, seconds=40, median_s=180)
        assert small["call"] == "STRONG BUY"
        assert big["call"] == ""
        assert small["share"] > big["share"]

    def test_the_share_never_leaves_its_bounds(self):
        for d, v in ((5000, 1000), (-5000, 1000), (0, 0), (10, 0)):
            assert 0.0 <= fo.share(d, v) <= 1.0


class TestTheClockIsTheActivityReading:
    """BACKWARDS FROM EVERY TIME CHART, on purpose. `VolumeRule`
    closes at exactly N contracts, so every bar holds the same
    size and volume says nothing about one bar against another.
    What differs is how long it took."""

    def test_the_same_push_grades_differently_on_pace_alone(self):
        fast = fo.read(delta=900, volume=3000, seconds=30, median_s=180)
        slow = fo.read(delta=900, volume=3000, seconds=300, median_s=180)
        assert fast["volume"] == slow["volume"]
        assert fast["delta"] == slow["delta"]
        assert fast["call"] == "STRONG BUY"
        assert slow["call"] == "BUY"

    def test_pace_is_against_recent_bars_not_a_fixed_number(self):
        """The whole point of a volume bar is that its duration
        moves with the session, so "fast" has to move with it."""
        quiet = fo.read(900, 3000, seconds=100, median_s=600)
        busy = fo.read(900, 3000, seconds=100, median_s=60)
        assert quiet["fast"] is True
        assert busy["fast"] is False

    def test_fast_alone_is_not_strong(self):
        """STRONG NEEDS BOTH. A bar that filled in a hurry but was
        only just one-sided is activity without a push, and
        grading it STRONG is the reading saying something it
        cannot see."""
        r = fo.read(delta=360, volume=3000, seconds=20, median_s=180)
        assert r["fast"] is True
        assert r["share"] < fo.STRONG_AT
        assert r["call"] == "BUY", "fast alone graded it strong"

    def test_no_median_yet_is_ordinary_not_fast(self):
        """A fresh series must not grade everything STRONG before
        it knows what normal looks like."""
        assert fo.pace(40.0, 0.0) == 1.0
        r = fo.read(delta=900, volume=3000, seconds=40, median_s=0)
        assert r["fast"] is False
        assert r["call"] == "BUY"

    def test_the_elapsed_time_is_the_bars_own_not_a_wall_clock(self):
        """`ts_close` is set on every print, so it is the running
        end of a forming bar. Reading `time.time()` would charge
        the bar for however long the feed has been quiet."""
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("def _foot_gate(")
        fn = src[i:src.index("\n    def ", i + 10)]
        assert 'getattr(cur, "ts_close"' in fn
        assert 'getattr(cur, "ts_open"' in fn
        # CODE, NOT COMMENTS. The comment above that line explains
        # why a wall clock is wrong, which contains the string the
        # test was looking for -- a test reading prose and passing
        # itself. Second time this exact shape has come up.
        code = "\n".join(ln for ln in fn.splitlines()
                         if not ln.strip().startswith("#"))
        assert "time.time()" not in code, \
            "the bar is aged by a wall clock"


class TestTheDiagonalConfirms:
    def test_imbalances_can_carry_a_push_that_is_not_fast(self):
        """An imbalanced run is the same statement made at prices
        instead of in time."""
        bare = fo.read(700, 3000, seconds=190, median_s=180)
        backed = fo.read(700, 3000, seconds=190, median_s=180,
                         buy_imbalances=6, sell_imbalances=1)
        assert bare["call"] == "BUY"
        assert backed["call"] == "STRONG BUY"
        assert backed["confirmed"] is True

    def test_imbalances_on_the_wrong_side_do_not_confirm(self):
        r = fo.read(700, 3000, seconds=190, median_s=180,
                    buy_imbalances=1, sell_imbalances=6)
        assert r["confirmed"] is False
        assert r["call"] == "BUY"

    def test_more_against_than_behind_does_not_confirm(self):
        """IT IS NOT A COUNT, IT IS A COMPARISON. Three buy levels
        under five sell levels is the diagonal arguing with the
        push, not backing it -- and a test using one buy level
        passes whether or not that comparison exists."""
        r = fo.read(700, 3000, seconds=190, median_s=180,
                    buy_imbalances=3, sell_imbalances=5)
        assert r["buy_imbalances"] >= 2, "this no longer tests that"
        assert r["confirmed"] is False
        assert r["call"] == "BUY"

    def test_one_level_is_not_a_run(self):
        r = fo.read(700, 3000, seconds=190, median_s=180,
                    buy_imbalances=1, sell_imbalances=0)
        assert r["confirmed"] is False

    def test_the_diagonal_never_creates_a_side_on_its_own(self):
        """CONFIRMATION, NOT A VOTE. A two-sided bar with buy
        imbalances is still a two-sided bar."""
        r = fo.read(delta=10, volume=3000, seconds=20, median_s=180,
                    buy_imbalances=9, sell_imbalances=0)
        assert r["side"] == ""
        assert r["call"] == ""


class TestTheThresholds:
    def test_a_strong_cut_below_the_push_cut_cannot_make_all_pushes_strong(
            self):
        """Ordering them is cheaper than trusting whoever set them."""
        r = fo.read(300, 3000, 200, 180, push_at=0.09, strong_at=0.01)
        assert r["strong_at"] >= r["push_at"]
        assert r["call"] == "BUY"

    def test_a_zero_push_cut_does_not_call_every_bar(self):
        r = fo.read(1, 3000, 200, 180, push_at=0.0)
        assert r["side"] == ""

    def test_the_cuts_are_carried_so_the_panel_can_show_them(self):
        r = fo.read(900, 3000, 40, 180)
        for k in ("push_at", "strong_at", "fast_at"):
            assert k in r


class TestSizingTheBar:
    def test_it_scales_straight_with_time(self):
        """Volume only goes up, so unlike the OFI leg there is no
        square root and no correction loop."""
        a = fo.size_for(50.0, target_s=60)
        b = fo.size_for(50.0, target_s=120)
        assert b["contracts"] == pytest.approx(2 * a["contracts"],
                                               rel=0.1)

    def test_no_rate_sizes_nothing(self):
        r = fo.size_for(0.0, target_s=180)
        assert r["ready"] is False
        assert r["contracts"] == 0
        assert "nothing" in r["why"]

    def test_it_says_what_it_measured(self):
        r = fo.size_for(50.0, target_s=180)
        assert "contracts a second" in r["why"]
        assert r["ready"] is True

    def test_it_stays_inside_its_bounds(self):
        assert fo.size_for(1e-6, 180)["contracts"] >= 100
        assert fo.size_for(1e9, 180)["contracts"] <= 2_000_000

    def test_junk_never_raises(self):
        for bad in (None, "abc", [], {}, -1):
            r = fo.size_for(bad, target_s=bad)
            assert isinstance(r["contracts"], int)


class TestJunkNeverRaises:
    def test_every_argument(self):
        for bad in (None, "abc", [], {}, 0, -1):
            r = fo.read(delta=bad, volume=bad, seconds=bad,
                        median_s=bad, buy_imbalances=bad,
                        sell_imbalances=bad, push_at=bad,
                        strong_at=bad, fast_at=bad)
            assert r["call"] in ("", "BUY", "SELL",
                                 "STRONG BUY", "STRONG SELL")
            assert r["decides"] is False


class TestItIsWiredThrough:
    def test_the_setting_round_trips(self, tmp_path):
        from liqmap.journal import Journal
        j = Journal(str(tmp_path / "j.db"))
        s = j.settings("ES")
        assert s.foot_s == 0, "it switched itself on"
        s.foot_s = 180
        assert j.save_settings(s).foot_s == 180
        assert j.settings("ES").to_dict()["foot_s"] == 180
        s.foot_s = 77
        assert j.save_settings(s).foot_s == 0

    def test_the_route_takes_it(self, client, monkeypatch):
        from tests.test_cme_panels import _cme
        _cme(monkeypatch)
        r = client.post("/api/journal/settings",
                        params={"coin": "ES", "foot_s": 300},
                        headers=AUTH).json()["settings"]
        assert r["foot_s"] == 300

    def test_it_does_not_touch_the_existing_footprint_panel(self):
        """`tape.footprint` is ONE slot feeding the footprint panel
        on a time rule, and changing its rule rebuilds it and drops
        its history. Chris has already said that panel vanishing is
        a problem."""
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("def _foot_stream(")
        fn = src[i:src.index("\n    def ", i + 10)]
        assert "tape.footprint =" not in fn, \
            "it overwrote the shared footprint stream"
        assert "on_trade_exec" in fn, \
            "it is not attached to the tape as its own listener"

    def test_only_one_listener_is_ever_registered(self):
        """Registering a second on every rebuild would feed every
        stream this session ever made -- a leak that looks like
        the bar being too slow."""
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("def _foot_stream(")
        fn = src[i:src.index("\n    def ", i + 10)]
        assert "_foot_box" in fn
        j = fn.index("on_trade_exec")
        assert "if box is None:" in fn[:j], \
            "the listener is attached outside the once-only guard"

    def test_it_is_rate_limited_and_gated(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        assert "_FOOT_EVERY_S" in src and "_FOOT_MOVE" in src
        i = src.index("def _foot_stream(")
        fn = src[i:src.index("\n    def ", i + 10)]
        assert "_FOOT_EVERY_S" in fn and "moved < _FOOT_MOVE" in fn

    def test_the_route_carries_the_gate(self, client, monkeypatch):
        from liqmap import web
        from tests.test_cme_panels import _cme
        _cme(monkeypatch)
        rt = web.runtime()
        st = rt.journal.settings("ES")
        st.foot_s = 180
        rt.journal.save_settings(st)
        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH).json()
        jr = d.get("journal") or {}
        assert "error" not in jr, jr.get("error")
        fg = jr.get("footgate")
        assert fg is not None, "no footprint gate on the payload"
        assert fg.get("on") is True
        assert fg.get("decides") is False

    def test_off_carries_an_off_gate_rather_than_nothing(self, client,
                                                         monkeypatch):
        from liqmap import web
        from tests.test_cme_panels import _cme
        _cme(monkeypatch)
        rt = web.runtime()
        st = rt.journal.settings("ES")
        st.foot_s = 0
        rt.journal.save_settings(st)
        d = client.get("/api/read",
                       params={"coin": "ES", "interval": "15m",
                               "timeframes": "15m,5m,1m"},
                       headers=AUTH).json()
        fg = (d.get("journal") or {}).get("footgate") or {}
        assert fg.get("on") is False

    def test_the_selector_is_on_the_page(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        assert 'id="jnlFootS"' in src
        assert "foot_s: $('jnlFootS')" in src, "never sent"
        assert "set('jnlFootS'" in src, "never restored"
        assert "jnlFootS: $('jnlFootS')" in src, "not pinned"
        assert 'id="jnlFootGate"' in src, "the box does not exist"
        assert "paintFootGate(J.footgate)" in src, "never painted"

    def test_the_panel_says_it_decides_nothing(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("function paintFootGate(")
        fn = src[i:src.index("\nfunction ", i + 10)]
        assert "shown only" in fn
        assert "opens and closes nothing" in fn


from tests.test_cme_panels import client  # noqa: E402,F401
