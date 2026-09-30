"""Panning, zooming, and the two profile window pickers.

A chart you cannot move is a chart that shows you whatever it decided to
show you. Three limits made this one unusable at the zoom levels a node
interaction is read at:

  * the newest bar was welded to the price axis -- zoom in as far as you
    like and the thing you are looking at is always jammed against the
    right edge with nothing in front of it;
  * the price window auto-fitted to the candles and nothing else, so a
    value area above or below them simply was not on the chart;
  * zoom bottomed out at fifteen bars.

The interesting cases are all arithmetic, so most of these run the real
functions out of the rendered page in node rather than grepping for them.
A grep test says the source contains a word. These say it computes the
right number.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile

import pytest

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node not available")


def _src() -> str:
    return open("liqmap/web.py").read()


def _page() -> str:
    from liqmap.web import DASHBOARD
    return DASHBOARD


def _grab(pattern: str) -> str:
    m = re.search(pattern, _page(), re.S)
    assert m, f"could not find {pattern!r} in the rendered page"
    return m.group(0)


def _run(body: str, bars: int = 400) -> dict:
    """Execute the page's own view functions against a fake bar series."""
    js = "\n".join([
        _grab(r"const RIGHT_ROOM = [0-9.]+;"),
        _grab(r"const SHIFT_LIMIT = [0-9.]+;"),
        _grab(r"function rightSlots\(\) \{[^}]*\}"),
        _grab(r"function chartStep\(plotW, n\) \{[^}]*\}"),
        _grab(r"function chartSlice\(\) \{.*?\n\}\n"),
        "let chartProjOn = true;",
        f"let chartData = {{bars: Array.from({{length:{bars}}},"
        "(_,i)=>({h:i,l:i,v:1}))};",
        "let chartView = {count:90, offset:0, scale:1, shift:0};",
        "const out = {};",
        body,
        "console.log(JSON.stringify(out));",
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(js)
        path = fh.name
    r = subprocess.run([NODE, path], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[:900]
    return json.loads(r.stdout)


# ------------------------------------------------------- scrolling past

@needs_node
def test_you_can_scroll_past_the_newest_bar():
    """THE fix. Without room to the right the live bar is stuck against
    the price axis and there is nowhere to read what happens next to it."""
    out = _run("chartView.offset = -9999; const s = chartSlice();"
               "out.off = s.offset; out.shown = s.bars.length;"
               "out.count = s.count;")
    assert out["off"] < 0
    assert out["shown"] < out["count"]        # the shortfall is empty chart


@needs_node
def test_the_room_to_the_right_is_bounded():
    """Unbounded, you can scroll the candles off the screen entirely and
    have no idea which way to drag to find them again."""
    out = _run("chartView.offset = -9999; const s = chartSlice();"
               "out.off = s.offset; out.min = s.minOff;")
    assert out["off"] == out["min"]
    assert out["min"] > -90                    # less than a full window


@needs_node
def test_the_candles_keep_their_width_when_you_scroll_past():
    """The step must come from the window size, not the bar count. Taking
    it from the bars would fatten the candles to fill the width and give
    back the very room you just asked for -- and it would put the drawing
    out of step with the hit testing, which already uses the count."""
    out = _run("const a = chartStep(800, chartSlice().count);"
               "chartView.offset = -30;"
               "const s = chartSlice();"
               "out.same = chartStep(800, s.count) === a;"
               "out.fewer = s.bars.length < s.count;")
    assert out["same"] is True
    assert out["fewer"] is True


@needs_node
def test_scrolling_back_still_stops_at_the_oldest_bar():
    out = _run("chartView.offset = 99999; const s = chartSlice();"
               "out.off = s.offset; out.max = s.all.length - s.count;")
    assert out["off"] == out["max"]


@needs_node
def test_you_can_zoom_in_to_a_handful_of_bars():
    out = _run("chartView.count = 1; out.count = chartSlice().count;")
    assert out["count"] <= 5


def test_the_draw_takes_its_step_from_the_window_not_the_bars():
    src = _src()
    fn = src[src.index("function drawChart() {"):]
    fn = fn[:fn.index("const bw =")]
    assert "chartStep(plotW, s.count)" in fn
    assert "chartStep(plotW, bars.length)" not in fn


# ---------------------------------------------------------- vertical pan

def test_the_price_window_can_be_dragged_off_the_candles():
    """Auto-fitting to the bars alone means a value area above or below
    them is not on the chart and there is nothing you can do about it."""
    src = _src()
    fn = src[src.index("function drawChart() {"):]
    fn = fn[:fn.index("chartScale = {hi, lo")]
    assert "chartView.shift" in fn
    assert "SHIFT_LIMIT" in fn


def test_a_pan_moves_both_axes_in_one_gesture():
    src = _src()
    fn = src[src.index("  const move = ev => {"):]
    fn = fn[:fn.index("  const upEv = ev => {")]
    assert "chartView.offset" in fn and "chartView.shift" in fn


def test_the_pan_can_reach_the_room_to_the_right():
    """Clamping the drag at zero leaves the scroll-past room reachable
    only by the wheel, which is not how most people move a chart."""
    src = _src()
    fn = src[src.index("  const move = ev => {"):]
    fn = fn[:fn.index("  const upEv = ev => {")]
    assert "Math.max(s.minOff" in fn


def test_the_wheel_does_not_clamp_the_offset_itself():
    """chartSlice owns the bounds. Clamping in two places is how the room
    to the right gets quietly taken back on the next scroll."""
    src = _src()
    fn = src[src.index("cv.addEventListener('wheel'"):]
    fn = fn[:fn.index("let drag = null;")]
    assert "chartView.offset = chartView.offset - Math.round" in fn


def test_reset_puts_the_vertical_shift_back_too():
    """A chart dragged off its own candles with no one-click way home is
    a chart you have to reload the page to recover."""
    src = _src()
    fn = src[src.index("function resetChartView() {"):]
    fn = fn[:fn.index("\n}\n")]
    assert "shift: 0" in fn


def test_the_view_starts_with_a_shift():
    src = _src()
    assert "let chartView = {count: 90, offset: 0, scale: 1.0, shift: 0};" \
        in src


# ------------------------------------------------- the window pickers

def test_the_two_windows_are_labelled_timeframes_not_bare_numbers():
    """A spinner showing '60' says nothing about what it is 60 of."""
    page = _page()
    for box in ("svpMin", "mvpMin"):
        i = page.index(f'id="{box}"')
        block = page[i:page.index("</select>", i)]
        assert "<option" in block, f"{box} is not a dropdown"
        assert ">1H<" in block or ">15m<" in block


def test_the_structural_window_offers_the_longer_timeframes():
    page = _page()
    i = page.index('id="svpMin"')
    block = page[i:page.index("</select>", i)]
    for label in (">1H<", ">4H<", ">1D<"):
        assert label in block


def test_the_micro_window_offers_the_shorter_timeframes():
    page = _page()
    i = page.index('id="mvpMin"')
    block = page[i:page.index("</select>", i)]
    for label in (">1m<", ">5m<", ">15m<"):
        assert label in block


def test_a_micro_window_that_is_not_shorter_is_refused_not_clamped():
    """Silently clamping leaves the dropdown showing a timeframe that is
    not the one you get, which is worse than refusing the choice."""
    src = _src()
    assert "function syncWindowOptions(" in src
    fn = src[src.index("function syncWindowOptions("):]
    fn = fn[:fn.index("\n}\n")]
    assert "o.disabled" in fn


def test_changing_either_window_refetches():
    src = _src()
    fn = src[src.index("function setWindows(") :]
    fn = fn[:fn.index("\n}\n")]
    assert "syncWindowOptions()" in fn and "loadProfile()" in fn


def test_the_saved_windows_are_re_validated_on_restore():
    """A pair saved before the options changed can be an impossible pair
    now, and restoring it without the check re-enables the bad state."""
    src = _src()
    fn = src[src.index("function restoreProfileWindows(") :]
    fn = fn[:fn.index("\n}\n")]
    assert "syncWindowOptions()" in fn


# -------------------------------------------------------------- labels

def test_node_labels_are_capped():
    """Zoomed in on a busy profile a dozen nodes can be in frame, and a
    stack of tags reading 'HVN' buries the three levels you came for."""
    src = _src()
    assert "const NODE_LABELS" in src
    fn = src[src.index("function drawProfile("):]
    fn = fn[:fn.index("\n}\n")]
    assert "NODE_LABELS" in fn


def test_a_suppressed_node_label_still_draws_its_line():
    """The level is still a level. Dropping the line with the tag would
    hide structure rather than declutter it."""
    src = _src()
    fn = src[src.index("    const mark = (px, label, colour, dash, quiet)"):]
    fn = fn[:fn.index("    };")]
    assert "g.stroke();" in fn
    assert "if (!quiet) labels.push" in fn


def test_node_labels_carry_their_price():
    src = _src()
    fn = src[src.index("function drawProfile("):]
    fn = fn[:fn.index("\n}\n")]
    assert "'HVN'" in fn and "chPx(nd[1])" in fn


@needs_node
def test_the_dashboard_script_still_parses():
    page = _page()
    js = page[page.rindex("<script>") + 8:page.rindex("</script>")]
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(js)
        path = fh.name
    r = subprocess.run([NODE, "--check", path], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr[:800]
