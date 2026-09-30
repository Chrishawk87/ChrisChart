"""Two profiles, one inside the other.

The states have to be MUTUALLY EXCLUSIVE and the precedence has to be right.
A value area that has cleared the SVP VAH also sits above the SVP midpoint,
so a classifier that checks "toward the high edge" before "entirely outside"
reports every breakout as a migration -- which is the exact opposite trade.
Most of these tests exist to pin that ordering.

They also pin the shared grid. Two profiles binned differently cannot be
compared, so `study` must hand the same row size to both.
"""

from __future__ import annotations

import pytest

from liqmap import nested as nd
from liqmap import profile as pf


class Bar:
    __slots__ = ("ts", "open", "high", "low", "close", "volume")

    def __init__(self, ts, o, h, l, c, v=100.0):
        self.ts, self.open, self.high = ts, o, h
        self.low, self.close, self.volume = l, c, v


def at(ts, price, vol=100.0):
    return Bar(ts, price, price + 0.25, price - 0.25, price, vol)


def block(start_ts, centre, spread_ticks, cycles=5, vol=100.0, step=1.0):
    """Bars sweeping +/- spread around `centre`, in WHOLE sweeps.

    Whole sweeps matter: a partial one leaves extra weight at whichever end
    it stopped, which drags the POC off centre and makes the fixture's own
    asymmetry the thing under test.

    A flat sweep is not good enough either: a perfectly rectangular profile
    has no argmax, so the POC lands wherever the tie-break points and the
    fixture's own flatness becomes the result. Volume is therefore tented
    toward the centre, which is what a balanced auction actually looks like.
    """
    out = []
    period = 2 * spread_ticks + 1
    for i in range(period * cycles):
        k = i % period
        off = k - spread_ticks
        weight = 1.0 + (spread_ticks - abs(off))
        out.append(at(start_ts + i * step, centre + off * 0.25,
                      vol * weight))
    return out


def prof_at(centre, spread_ticks=8, cycles=5, vol=100.0):
    return pf.final(block(0.0, centre, spread_ticks, cycles, vol), tick=0.25)


# --------------------------------------------------------------- geometry

def test_overlap_is_the_share_of_a_inside_b():
    assert nd._overlap(2.0, 4.0, 0.0, 10.0) == pytest.approx(1.0)
    assert nd._overlap(0.0, 10.0, 5.0, 15.0) == pytest.approx(0.5)
    assert nd._overlap(20.0, 30.0, 0.0, 10.0) == pytest.approx(0.0)


def test_a_zero_width_span_is_inside_or_it_is_not():
    assert nd._overlap(5.0, 5.0, 0.0, 10.0) == pytest.approx(1.0)
    assert nd._overlap(50.0, 50.0, 0.0, 10.0) == pytest.approx(0.0)


# ----------------------------------------------------------- the states

def test_a_micro_area_inside_the_structural_one_is_nested():
    svp = prof_at(7000.0, spread_ticks=40)
    mvp = prof_at(7000.0, spread_ticks=4)
    r = nd.classify(svp, mvp)
    assert r.state == nd.NESTED
    assert r.overlap == pytest.approx(1.0)


def test_a_micro_area_entirely_above_is_a_breakout_not_a_migration():
    """THE precedence test. Above the VAH is also above the midpoint."""
    svp = prof_at(7000.0, spread_ticks=40)
    mvp = prof_at(7030.0, spread_ticks=2)
    r = nd.classify(svp, mvp, drift_ticks=+60.0)
    assert r.state == nd.BREAKOUT_UP
    assert r.state != nd.MIGRATING_UP


def test_a_micro_area_entirely_below_is_a_breakout_down():
    svp = prof_at(7000.0, spread_ticks=40)
    mvp = prof_at(6970.0, spread_ticks=2)
    r = nd.classify(svp, mvp, drift_ticks=-60.0)
    assert r.state == nd.BREAKOUT_DOWN


def test_straddling_the_structural_high_is_a_test_of_it():
    svp = prof_at(7000.0, spread_ticks=40)
    mvp = prof_at(svp.vah, spread_ticks=6)
    r = nd.classify(svp, mvp)
    assert r.state == nd.TESTING_SVP_VAH


def test_straddling_the_structural_low_is_a_test_of_it():
    svp = prof_at(7000.0, spread_ticks=40)
    mvp = prof_at(svp.val, spread_ticks=6)
    r = nd.classify(svp, mvp)
    assert r.state == nd.TESTING_SVP_VAL


def test_a_micro_area_wider_than_the_structure_is_named_not_traded():
    """If the MVP swallows the SVP the hierarchy is gone, and reporting one
    of the four states would be describing a nesting that is not there."""
    svp = prof_at(7000.0, spread_ticks=4)
    mvp = prof_at(7000.0, spread_ticks=40)
    r = nd.classify(svp, mvp)
    assert r.state == nd.ENGULFING
    assert "lengthen the SVP window" in r.playbook


def test_drifting_up_inside_value_is_a_migration():
    svp = prof_at(7000.0, spread_ticks=40)
    mvp = prof_at(7003.0, spread_ticks=2)
    r = nd.classify(svp, mvp, drift_ticks=+10.0)
    assert r.state == nd.MIGRATING_UP


def test_drifting_down_inside_value_is_a_migration():
    svp = prof_at(7000.0, spread_ticks=40)
    mvp = prof_at(6997.0, spread_ticks=2)
    r = nd.classify(svp, mvp, drift_ticks=-10.0)
    assert r.state == nd.MIGRATING_DOWN


def test_a_poc_that_has_barely_moved_is_not_migrating():
    """Below the threshold the POC moves by binning alone, and calling that
    migration would report a direction on every bar."""
    svp = prof_at(7000.0, spread_ticks=40)
    mvp = prof_at(7003.0, spread_ticks=2)
    r = nd.classify(svp, mvp, drift_ticks=+1.0)
    assert r.state == nd.NESTED


def test_a_poc_drifting_but_still_mid_area_is_not_migrating():
    svp = prof_at(7000.0, spread_ticks=40)
    mvp = prof_at(7000.0, spread_ticks=2)
    r = nd.classify(svp, mvp, drift_ticks=+20.0)
    assert r.state == nd.NESTED


def test_every_state_carries_a_playbook_line():
    for s in (nd.NESTED, nd.MIGRATING_UP, nd.MIGRATING_DOWN,
              nd.TESTING_SVP_VAH, nd.TESTING_SVP_VAL, nd.BREAKOUT_UP,
              nd.BREAKOUT_DOWN, nd.ENGULFING, nd.UNKNOWN):
        assert s in nd.PLAYBOOK and len(nd.PLAYBOOK[s]) > 20


def test_the_playbook_never_places_an_order():
    banned = ("submit", "order_id", "api_key", "place_order", "execute(")
    for text in nd.PLAYBOOK.values():
        low = text.lower()
        assert not any(b in low for b in banned)


# ------------------------------------------------------------- movement

def test_falling_window_volume_reads_as_fading():
    svp = prof_at(7000.0, spread_ticks=40)
    mvp = prof_at(7000.0, spread_ticks=4)
    r = nd.classify(svp, mvp, volume_ratio=0.5, has_history=True)
    assert r.volume_trend == "fading"


def test_rising_window_volume_reads_as_building():
    svp = prof_at(7000.0, spread_ticks=40)
    mvp = prof_at(7000.0, spread_ticks=4)
    r = nd.classify(svp, mvp, volume_ratio=2.0, has_history=True)
    assert r.volume_trend == "building"


def test_without_a_previous_window_the_trend_is_not_asserted():
    """One window is not a trend. Reporting 'flat' from a ratio nobody
    measured would be inventing a fact."""
    svp = prof_at(7000.0, spread_ticks=40)
    mvp = prof_at(7000.0, spread_ticks=4)
    r = nd.classify(svp, mvp, volume_ratio=0.1, has_history=False)
    assert r.volume_trend == "flat"
    assert r.has_history is False


# --------------------------------------------------------------- window

def test_the_window_is_trailing_not_centred():
    bars = [at(i * 60.0, 7000.0) for i in range(120)]
    w = nd.window(bars, minutes=10)
    assert w[-1] is bars[-1]
    assert all(b.ts >= bars[-1].ts - 600.0 for b in w)


def test_an_earlier_end_gives_the_same_window_shifted_back():
    bars = [at(i * 60.0, 7000.0) for i in range(120)]
    now = nd.window(bars, 10)
    prev = nd.window(bars, 10, end=float(now[0].ts) - 1e-9)
    assert prev and now
    assert max(b.ts for b in prev) < min(b.ts for b in now)


def test_no_bars_is_an_empty_window_not_a_crash():
    assert nd.window([], 10) == []


# ---------------------------------------------------------------- study

def _two_tier(n=3600, tail=600, tail_lo=7022.0):
    """An hour of 1-second bars that ranges wide, then leaves for the top.

    The first 50 minutes build a balanced structural area between 7000 and
    7020; the last 10 trade above all of it. That is a two-tier auction by
    construction -- the micro window's value has left the structural one.
    """
    bars = [at(float(i), 7000.0 + (i % 81) * 0.25, 100.0)
            for i in range(n - tail)]
    bars += [at(float(n - tail + i), tail_lo + (i % 5) * 0.25, 100.0)
             for i in range(tail)]
    return bars


def test_study_builds_both_profiles_on_the_same_grid():
    """Req 4. Different row sizes make every level comparison an artefact
    of the binning rather than a fact about the auctions."""
    svp, mvp, _ = nd.study(_two_tier(), svp_minutes=60, mvp_minutes=10)
    assert svp.tick == mvp.tick
    assert svp.tick > 0


def test_study_gives_the_two_profiles_different_shapes():
    """If the windows produced the same value area there would be no
    hierarchy to read."""
    svp, mvp, _ = nd.study(_two_tier(), svp_minutes=60, mvp_minutes=5)
    assert mvp.va_width_ticks < svp.va_width_ticks
    assert mvp.n_bars < svp.n_bars


def test_study_measures_drift_against_the_previous_window():
    bars = _two_tier()
    _, _, r = nd.study(bars, svp_minutes=60, mvp_minutes=5)
    assert r.has_history is True


def test_study_classifies_a_settled_top_as_at_or_above_the_high():
    _, _, r = nd.study(_two_tier(tail=130), svp_minutes=60, mvp_minutes=2)
    assert r.state in (nd.TESTING_SVP_VAH, nd.BREAKOUT_UP, nd.MIGRATING_UP)


def test_study_with_no_bars_reads_as_unknown():
    svp, mvp, r = nd.study([], 60, 10)
    assert r.state == nd.UNKNOWN
    assert svp.empty and mvp.empty


def test_study_with_one_bar_does_not_raise():
    svp, mvp, r = nd.study([at(0.0, 7000.0)], 60, 10)
    assert r.state in (nd.UNKNOWN, nd.NESTED, nd.ENGULFING)


def test_the_micro_window_is_the_tail_of_the_structural_one():
    """Nesting means one auction INSIDE another. If the MVP drew on bars
    the SVP never saw, the two would be describing different markets."""
    bars = _two_tier()
    svp_bars = nd.window(bars, 60)
    mvp_bars = nd.window(bars, 5)
    assert set(id(b) for b in mvp_bars) <= set(id(b) for b in svp_bars)


def test_the_dict_carries_every_field_the_panel_shows():
    _, _, r = nd.study(_two_tier(), 60, 5)
    d = r.to_dict()
    for k in ("state", "overlap", "poc_position", "gap_ticks", "drift_ticks",
              "volume_ratio", "volume_trend", "has_history", "note",
              "playbook"):
        assert k in d


def test_the_conventions_are_the_stated_ones():
    assert nd.MIGRATE_TICKS == 4.0
    assert nd.EDGE_SIDE == 0.40
    assert nd.FADING_RATIO == 0.70
    assert nd.BUILDING_RATIO == 1.30


# ------------------------------------------------- the chart integration

def _src():
    return open("liqmap/web.py").read()


def test_the_route_returns_both_profiles_and_the_state():
    src = _src()
    route = src[src.index('@app.get("/api/profile"'):
                src.index('@app.get("/api/chart"')]
    for key in ('"svp"', '"mvp"', '"nested"'):
        assert key in route


def test_the_route_still_returns_the_prior_session_read():
    """Req 5: the existing structural read is preserved, not replaced."""
    src = _src()
    route = src[src.index('@app.get("/api/profile"'):
                src.index('@app.get("/api/chart"')]
    assert "_ms.read(rows[cut:], reference)" in route
    assert '"reference"' in route and '"session"' in route


def test_the_two_profiles_are_built_by_study_not_separately():
    """Two hand-rolled calls would drift apart on the row size, which is
    the one thing that has to be shared."""
    src = _src()
    route = src[src.index('@app.get("/api/profile"'):
                src.index('@app.get("/api/chart"')]
    assert "_nd.study(" in route


def test_the_windows_are_independent_inputs():
    src = _src()
    route = src[src.index('@app.get("/api/profile"'):
                src.index('@app.get("/api/chart"')]
    assert "svp_min" in route and "mvp_min" in route


def test_the_client_keeps_the_profiles_keyed_not_prefixed():
    """Two near-identical global name families in one script is the
    copy-paste failure this project has already been bitten by twice."""
    src = _src()
    assert "let svp_poc" not in src and "let mvp_poc" not in src
    assert "PROFILE_LAYERS" in src


def test_each_layer_has_its_own_width_and_colour():
    """Drawn at the same width and hue the narrow profile vanishes inside
    the wide one, which is the whole thing the second profile is for."""
    src = _src()
    layers = src[src.index("const PROFILE_LAYERS"):]
    layers = layers[:layers.index("\n};") + 3]
    assert layers.count("maxShare") >= 2
    assert layers.count("hue") >= 2


def test_the_drawing_loops_over_the_layers():
    src = _src()
    fn = src[src.index("function drawProfile("):]
    fn = fn[:fn.index("\n}\n")]
    assert "for (const" in fn and "PROFILE_LAYERS" in fn


def test_the_dual_view_is_reachable_from_the_button():
    src = _src()
    fn = src[src.index("function cycleProfile("):]
    fn = fn[:fn.index("\n}\n")]
    assert "dual" in fn


def test_the_nested_state_is_shown_to_the_reader():
    src = _src()
    assert 'id="nestRead"' in src
    assert "$('nestRead')" in src


def test_both_windows_are_editable_from_the_chart():
    src = _src()
    assert 'id="svpMin"' in src and 'id="mvpMin"' in src
    assert "function profileWindows(" in src


def test_the_route_serves_both_profiles_end_to_end(monkeypatch):
    """The grep tests above check the source says the right words. This one
    actually calls it, with a fake venue, and reads the payload."""
    import tempfile
    from pathlib import Path

    from fastapi.testclient import TestClient

    from liqmap import web
    from liqmap.structure import Candle

    token = "t" * 24
    series = [Candle(ts=float(i * 60), open=7000.0 + (i % 81) * 0.25,
                     high=7000.25 + (i % 81) * 0.25,
                     low=6999.75 + (i % 81) * 0.25,
                     close=7000.0 + (i % 81) * 0.25, volume=500.0)
              for i in range(400)]

    class Fake:
        INTERVALS = {"1m": 60, "5m": 300, "15m": 900}

        def candles(self, coin, interval="5m", bars=200):
            return series[-bars:]

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("LIQMAP_DB", str(Path(tmp) / "n.db"))
        monkeypatch.setenv("LIQMAP_TOKEN", token)
        monkeypatch.setenv("LIQMAP_AUTO", "false")
        monkeypatch.delenv("LIQMAP_WALLETS", raising=False)
        web.RT = None
        with TestClient(web.create_app()) as c:
            rt = web.runtime()
            monkeypatch.setattr(rt, "client", lambda: Fake())
            monkeypatch.setattr(rt, "feed_for", lambda coin: None)
            d = c.get("/api/profile",
                      params={"token": token, "coin": "BTC",
                              "svp_min": 60, "mvp_min": 10}).json()
        web.RT = None

    assert d["ok"] is True
    for key in ("svp", "mvp", "nested", "reference", "session", "read"):
        assert key in d
    assert d["svp"]["minutes"] == 60 and d["mvp"]["minutes"] == 10
    # Req 4: one grid, or the levels are not comparable.
    assert d["svp"]["rows"] and d["mvp"]["rows"]
    assert d["nested"]["state"] in nd.PLAYBOOK
    assert d["nested"]["playbook"]


def test_the_route_will_not_let_the_micro_window_exceed_the_structural_one():
    """A micro window longer than the structural one is not a hierarchy,
    it is two profiles with the bigger one called small."""
    src = _src()
    route = src[src.index('@app.get("/api/profile"'):
                src.index('@app.get("/api/chart"')]
    assert "min(float(mvp_min), svp_m)" in route


def test_the_dashboard_script_actually_parses():
    """The page is a Python string, so a backslash escape in the JS is
    eaten by Python before the browser ever sees it -- '\\n' inside a JS
    string literal became a real newline and broke the whole script, with
    nothing in the Python test suite noticing. Parse the rendered page,
    not the source."""
    import shutil
    import subprocess
    import tempfile

    node = shutil.which("node")
    if node is None:
        import pytest as _pt
        _pt.skip("node not available")

    from liqmap.web import DASHBOARD as page
    js = page[page.rindex("<script>") + 8:page.rindex("</script>")]
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(js)
        path = fh.name
    r = subprocess.run([node, "--check", path], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr[:800]
