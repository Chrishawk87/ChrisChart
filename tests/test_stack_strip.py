"""READ, PRICE and THIS BAR, repeated at the foot of the stack panel.

THESE TESTS RUN THE PAGE'S JAVASCRIPT. They are the only ones in the
suite that do, and the reason is specific: the last display bug on this
panel -- price vanishing from the read line because a key had been
renamed in one place and not the other -- went out past 242 passing
tests, every one of which was a scan of the source. A scan cannot tell
that a function throws when it is called. So this builds a DOM stub,
runs the served page in node, calls the painter, and reads what came
out of the element.

What is pinned here:

  - the strip RENDERS -- the painter does not throw, and the three
    readings reach the panel;
  - it is a MIRROR, not a second opinion. The THIS BAR string on the
    stack is character-for-character the one the footprint drew. Two
    copies of that arithmetic is two things to keep in step, and the
    first time they disagreed the screen would be showing two answers
    about one bar;
  - the two READ lines carry DIFFERENT ids, so the per-print repaint
    can find both. One id on two elements leaves one of them frozen at
    whatever it said on load, which looks exactly like a slow market;
  - it goes BLANK, not stale, when the selected market is not the one
    the readings were taken on. A failed OFI poll leaves the last good
    payload in place, and ES's bar under a crypto symbol is the
    wrong-market display this panel was just fixed for;
  - the stack above it is UNTOUCHED -- same rows, same verdict, strip
    or no strip. Chris asked for the readings added at the bottom and
    nothing else changed.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None,
                                reason="node is not installed")

# A DOM stub, not a browser. Every element answers, so the page's
# bootstrap runs to the end instead of dying on the first id it cannot
# find -- which would leave every `const` after that point unreachable.
HARNESS = r"""
const fs = require('fs'), vm = require('vm');
const src = fs.readFileSync(process.argv[2], 'utf8');
const els = {};
function mk(id) {
  return { id, innerHTML: '', outerHTML: '', textContent: '', value: '',
    checked: false, hidden: false, clientWidth: 900, clientHeight: 300,
    offsetWidth: 900, scrollTop: 0, scrollHeight: 0, className: '',
    style: {}, dataset: {}, options: [], children: [], firstChild: null,
    classList: {add(){}, remove(){}, toggle(){}, contains(){return false;}},
    addEventListener(){}, removeEventListener(){}, appendChild(){},
    removeChild(){}, insertBefore(){}, setAttribute(){},
    getAttribute(){ return null; }, remove(){}, focus(){}, blur(){},
    click(){}, querySelector(){ return null; },
    querySelectorAll(){ return []; }, getContext(){ return ctx2d; },
    getBoundingClientRect(){
      return {width:900,height:300,left:0,top:0,right:900,bottom:300}; } };
}
const ctx2d = new Proxy({}, {get: (t,k) => (k in t ? t[k] : (()=>{})),
                             set: (t,k,v) => { t[k]=v; return true; }});
const get = id => (els[id] || (els[id] = mk(id)));
const doc = { hidden:false, cookie:'', readyState:'complete',
  getElementById: get, querySelector: () => mk('q'),
  querySelectorAll: () => [], addEventListener(){},
  removeEventListener(){}, createElement: () => mk('new'),
  body: mk('body'), documentElement: mk('html'), head: mk('head') };
const store = {};
const ls = { getItem: k => (k in store ? store[k] : null),
             setItem: (k,v) => { store[k] = String(v); },
             removeItem: k => { delete store[k]; } };
const ctx = { document: doc, console, localStorage: ls, sessionStorage: ls,
  setTimeout: ()=>0, setInterval: ()=>0, clearTimeout(){}, clearInterval(){},
  requestAnimationFrame: ()=>0, cancelAnimationFrame(){},
  fetch: () => Promise.resolve({ok:true, json: async()=>({}),
                                text: async()=>''}),
  WebSocket: function(){ this.close=()=>{}; this.send=()=>{}; },
  navigator: {userAgent:'node', clipboard:{writeText:async()=>{}}},
  Image: function(){}, URLSearchParams, __els: els };
ctx.window = ctx; ctx.globalThis = ctx; ctx.self = ctx;
ctx.window.location = {href:'http://x/', search:'', hash:'', pathname:'/'};
ctx.window.matchMedia = () => ({matches:false, addEventListener(){},
                                addListener(){}});
ctx.window.devicePixelRatio = 1;
ctx.window.innerWidth = 1200; ctx.window.innerHeight = 800;
ctx.window.addEventListener = ()=>{}; ctx.window.localStorage = ls;
process.on('unhandledRejection', () => {});

/* Top-level `const` and `let` do not become globals, so the probe is
   appended to the page's own source and shares its scope. */
const probe = `
globalThis.__probe = {
  strip(last, thisBar, stripCoin, selected) {
    document.getElementById('coin').value = selected;
    _ofiLast = last; _ofiThisBar = thisBar; _ofiStripCoin = stripCoin;
    paintStackStrip();
    return document.getElementById('stackStrip').innerHTML;
  },
  readLine(r, pr, id) { return ofiReadLine(r, pr, id); },
  clock(payload) {
    paintClockWindows(payload);
    return document.getElementById('clockWin').innerHTML;
  },
  cascade(d, last, thisBar, stripCoin, selected) {
    document.getElementById('coin').value = selected;
    _ofiLast = last; _ofiThisBar = thisBar; _ofiStripCoin = stripCoin;
    paintCascade(d);
    return {rows: document.getElementById('cascRows').innerHTML,
            say: document.getElementById('cascSay').innerHTML,
            strip: document.getElementById('stackStrip').innerHTML};
  }
};
`;
vm.createContext(ctx);
vm.runInContext(src + probe, ctx, {filename: 'page.js'});
const args = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const out = ctx.__probe[args.fn](...args.args);
process.stdout.write(JSON.stringify(out));
"""

READ_UP = {"read": "up", "raw": "up", "warm": True, "held_s": 42.0,
           "pending_s": 0.0, "lead": 1800.0, "gross": 9000.0,
           "share": 0.6, "recent": 0.0, "stalling": False,
           "into_s": 60.0, "left_s": 240.0, "candle_ts": 0.0,
           "flips": 2, "strength": "strong"}
PRICE_UP = {"dir": "up", "px": 6712.25, "still_s": 0.0}
THIS_BAR = ('<span class="stamp" style="width:100%">THIS BAR &mdash; '
            '<b class="long">UP</b> right now &middot; imbalance +640'
            '</span>')

# THE READ AS IT IS NOW: the OFI bar in progress, measured against its
# own open. `ticks` is what tells the two shapes apart, so a payload
# without it still draws the old flow line and this suite covers both.
BAR_READ = {"read": "up", "open_px": 6700.0, "last_px": 6701.0,
            "ticks": 4.0, "points": 1.0, "into_s": 42.0, "held_s": 30.0,
            "flips": 1, "bar_ts": 1000.0, "target": 2000,
            "forming": True}


@pytest.fixture(scope="module")
def page_js() -> str:
    """The script as the browser receives it, not as web.py spells it.

    Taken from the served page on purpose: reading it out of the Python
    source would carry that file's backslash escapes and test a string
    nobody is ever sent.
    """
    import os

    tmp = tempfile.mkdtemp()
    os.environ.setdefault("LIQMAP_TOKEN", "t")
    os.environ["LIQMAP_DB"] = str(Path(tmp) / "page.db")
    os.environ["LIQMAP_AUTO"] = "false"
    from fastapi.testclient import TestClient

    from liqmap import web

    web.RT = None
    with TestClient(web.create_app()) as c:
        r = c.get("/", headers={"Authorization":
                                f"Bearer {os.environ['LIQMAP_TOKEN']}"})
        assert r.status_code == 200, r.status_code
    blocks = re.findall(r"<script>(.*?)</script>", r.text, re.S)
    assert blocks, "the page served no script at all"
    path = Path(tmp) / "page.js"
    path.write_text(max(blocks, key=len))
    return str(path)


def run_js(page_js: str, fn: str, *args):
    with tempfile.NamedTemporaryFile("w", suffix=".json",
                                     delete=False) as fh:
        json.dump({"fn": fn, "args": list(args)}, fh)
        argpath = fh.name
    with tempfile.NamedTemporaryFile("w", suffix=".js",
                                     delete=False) as fh:
        fh.write(HARNESS)
        hpath = fh.name
    p = subprocess.run([NODE, hpath, page_js, argpath],
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, f"node failed:\n{p.stderr[-3000:]}"
    return json.loads(p.stdout)


class TestTheStripRenders:
    def test_the_three_readings_reach_the_panel(self, page_js):
        html = run_js(page_js, "strip", READ_UP, THIS_BAR, "ES", "ES")
        assert "READ &mdash;" in html, html[:400]
        assert "THIS BAR &mdash;" in html, html[:400]

    def test_price_comes_with_the_read(self, page_js):
        """The pairing is the whole point of the line. A read without
        its price beside it is the thing Chris asked to be fixed once
        already, and it failed silently then."""
        html = run_js(page_js, "strip", {**READ_UP}, THIS_BAR, "ES", "ES")
        # No price in the payload, no PRICE on the line -- and no crash.
        assert "PRICE &mdash;" not in html
        html2 = run_js(page_js, "strip",
                       {**READ_UP, "price": None}, THIS_BAR, "ES", "ES")
        assert "READ &mdash;" in html2

    def test_the_price_is_the_real_last_price(self, page_js):
        line = run_js(page_js, "readLine", READ_UP, PRICE_UP,
                      "stackReadLine")
        assert "PRICE &mdash;" in line
        assert "6712.25" in line, line

    def test_this_bar_is_the_footprints_own_string(self, page_js):
        """MIRRORED, NOT RECOMPUTED. If the strip ever starts building
        its own THIS BAR, this is where that shows up."""
        html = run_js(page_js, "strip", READ_UP, THIS_BAR, "ES", "ES")
        assert THIS_BAR in html, html[:600]

    def test_the_two_read_lines_do_not_share_an_id(self, page_js):
        """One id on two elements and getElementById repaints the first
        forever -- the second sits frozen at whatever it said on load,
        which on this panel is indistinguishable from a quiet market."""
        html = run_js(page_js, "strip", READ_UP, THIS_BAR, "ES", "ES")
        assert 'id="stackReadLine"' in html
        assert 'id="ofiReadLine"' not in html

    def test_the_id_defaults_for_the_footprint(self, page_js):
        line = run_js(page_js, "readLine", READ_UP, PRICE_UP, None)
        assert 'id="ofiReadLine"' in line


class TestTheReadIsTheOfiBarNow:
    """It used to be a flow reading on a five-minute clock. Chris asked
    for what the candle on the screen is doing, so it is now price
    against that bar's open, resetting when the bar does."""

    def test_it_renders_at_all(self, page_js):
        line = run_js(page_js, "readLine", BAR_READ, PRICE_UP,
                      "stackReadLine")
        assert "READ &mdash;" in line, line[:300]
        assert "UP" in line

    def test_it_shows_the_distance_from_the_open(self, page_js):
        """UP by a quarter of a tick and UP by four points are the same
        word and not the same thing."""
        line = run_js(page_js, "readLine", BAR_READ, PRICE_UP,
                      "stackReadLine")
        assert "4.00 ticks from its open" in line, line
        assert "+1.00 pts" in line, line

    def test_it_says_how_long_the_bar_has_been_open(self, page_js):
        line = run_js(page_js, "readLine", BAR_READ, PRICE_UP,
                      "stackReadLine")
        assert "this bar has been open" in line

    def test_it_never_claims_time_left_in_the_candle(self, page_js):
        """There is no time left in an OFI bar -- it ends when the
        imbalance says so. The old line said '4.0 min left in this
        candle' because its window was a clock; carrying that wording
        onto a flow window would be inventing a number."""
        line = run_js(page_js, "readLine", BAR_READ, PRICE_UP,
                      "stackReadLine")
        assert "left in this candle" not in line, line

    def test_flat_is_printed(self, page_js):
        line = run_js(page_js, "readLine",
                      {**BAR_READ, "read": "flat", "ticks": 0.0,
                       "points": 0.0, "held_s": 0.0},
                      PRICE_UP, "stackReadLine")
        assert "FLAT" in line
        assert "back at its open" in line, line

    def test_a_bar_crossing_its_own_open_says_so(self, page_js):
        """The word is UP or DOWN either way. Only the count can say
        the bar is going nowhere."""
        quiet = run_js(page_js, "readLine", BAR_READ, PRICE_UP, "x")
        assert "going nowhere" not in quiet
        busy = run_js(page_js, "readLine", {**BAR_READ, "flips": 9},
                      PRICE_UP, "x")
        assert "crossed its open 9 times" in busy, busy
        assert "going nowhere" in busy

    def test_no_bar_forming_says_so(self, page_js):
        line = run_js(page_js, "readLine",
                      {**BAR_READ, "forming": False}, PRICE_UP, "x")
        assert "no bar is forming" in line, line

    def test_price_is_still_its_own_reading(self, page_js):
        """It has no reference point at all -- not this bar's open --
        so it stays a second opinion rather than a restatement."""
        line = run_js(page_js, "readLine", BAR_READ,
                      {"dir": "down", "px": 6699.25}, "x")
        assert "PRICE &mdash;" in line
        assert "6699.25" in line
        assert "opposed" in line, line

    def test_the_old_flow_line_still_draws(self, page_js):
        """The accuracy scripts still score the flow reading, and it is
        still carried in the payload. A payload in the old shape must
        not fall through the new branch."""
        line = run_js(page_js, "readLine", READ_UP, PRICE_UP, "x")
        assert "left in this candle" in line, line


class TestItGoesBlankRatherThanStale:
    def test_another_market_shows_nothing(self, page_js):
        """A failed OFI poll leaves the last good payload in place. ES's
        bar under a crypto symbol is a wrong-market display, which is
        the bug this panel was just fixed for."""
        html = run_js(page_js, "strip", READ_UP, THIS_BAR, "ES", "BTC")
        assert html == "", html[:300]

    def test_no_payload_shows_nothing(self, page_js):
        html = run_js(page_js, "strip", None, "", "", "ES")
        assert html == "", html[:300]


class TestTheStackAboveIsUnchanged:
    """Chris asked for the readings added at the bottom, and for
    everything else left as it was."""

    # EVERY ROW A DIFFERENT SHAPE, deliberately. A fixture whose rungs
    # all qualify the same way makes the comparison below pass by
    # coincidence: a leak that forced every mark to DOWN would produce
    # byte-identical rows and the test would never notice. One of each
    # means any leak changes the output.
    CASC = {"cascade": {
        "trading": "5m", "state": "trend", "trend": "down", "side": "sell",
        "why": "every row under the 5m is red", "depth": 2, "deep": True,
        "describe": "the whole stack is down",
        "rungs": [
            {"timeframe": "5m", "colour": "red", "bps": -11.7,
             "present": True, "qualified": "down", "conflicted": False},
            {"timeframe": "1m", "colour": "green", "bps": 4.2,
             "present": True, "qualified": "up", "conflicted": False},
            {"timeframe": "30s", "colour": "green", "bps": -1.1,
             "present": True, "qualified": None, "conflicted": True},
            {"timeframe": "10s", "colour": "none", "bps": 0.0,
             "present": False, "qualified": None, "conflicted": False},
        ]}}

    def test_the_rows_are_the_same_with_the_strip_and_without(self,
                                                              page_js):
        with_strip = run_js(page_js, "cascade", self.CASC, READ_UP,
                            THIS_BAR, "ES", "ES")
        without = run_js(page_js, "cascade", self.CASC, None, "", "", "ES")
        assert with_strip["rows"] == without["rows"]
        assert with_strip["say"] == without["say"]
        assert with_strip["strip"] != ""
        assert without["strip"] == ""

    def test_the_verdict_still_says_what_it_said(self, page_js):
        got = run_js(page_js, "cascade", self.CASC, READ_UP, THIS_BAR,
                     "ES", "ES")
        assert "TREND" in got["say"]
        assert "SELL" in got["say"]
        # Each row still renders its own shape: a qualified down, a
        # qualified up, one thrown out for fighting its own number, and
        # one that is not there at all.
        assert "DOWN" in got["rows"]
        assert "UP" in got["rows"]
        assert "THROWN OUT" in got["rows"]
        assert "not read" in got["rows"]

    def test_nothing_in_the_rows_came_from_the_readings(self, page_js):
        """The stack's verdict is computed server-side from the bars.
        A strip that leaked into it would make the panel agree with
        itself by construction."""
        up = run_js(page_js, "cascade", self.CASC, READ_UP, THIS_BAR,
                    "ES", "ES")
        down = run_js(page_js, "cascade", self.CASC,
                      {**READ_UP, "read": "down", "raw": "down",
                       "lead": -1800.0},
                      THIS_BAR.replace("long\">UP", "short\">DOWN"),
                      "ES", "ES")
        assert up["rows"] == down["rows"]
        assert up["say"] == down["say"]
