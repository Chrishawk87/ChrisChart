"""The live order-flow ladder: delta per timeframe, the book at the touch.

The traps here are about silence.

A timeframe the tape cannot reach has NO opinion. Reporting it as a flat
delta makes an hour of missing data look like an hour of balance, and
then letting it vote makes silence outweigh the rungs that actually
measured something.

And the two velocities are not the same number. The ladder's is the
market in the last fifteen seconds, one figure belonging to the moment.
A rung's pace is its own bar against its own completed bars. Printing
the first on every row would be one measurement pretending to be five;
printing the second without checking the tape holds those bars would be
a ratio built from whatever happens to be in memory.
"""

from __future__ import annotations

import pytest

from liqmap import flowladder as fl
from liqmap import orderflow as of
from liqmap.flow import Book, FlowTape, Level, Trade

IV = (("1m", 60.0), ("5m", 300.0), ("15m", 900.0), ("1h", 3600.0))


def tape_of(trades):
    t = FlowTape(max_age=86400.0)
    for x in trades:
        t.add(x)
    return t


def tr(ts, px=6000.0, sz=1.0, side="buy"):
    return Trade(px=px, sz=sz, aggressor=side, ts=float(ts))


def book_at(bid, ask, bid_sz=100.0, ask_sz=100.0, tick=0.25):
    return Book(coin="ES", ts=0.0,
                bids=[Level(px=bid - i * tick, sz=bid_sz) for i in range(5)],
                asks=[Level(px=ask + i * tick, sz=ask_sz) for i in range(5)])


# ------------------------------------------------------------ the book

def test_a_tight_spread_passes_the_gate():
    r = fl.read_book(book_at(6000.00, 6000.25), of.ES, target_ticks=10.0)
    assert r.spread_ticks == pytest.approx(1.0)
    assert r.ok


def test_a_wide_spread_against_a_small_target_is_refused():
    """The same spread is trivial against forty ticks and fatal against
    four."""
    r = fl.read_book(book_at(6000.00, 6000.25), of.ES, target_ticks=4.0)
    assert not r.ok
    assert "too wide" in r.why


def test_the_touch_sizes_come_from_the_book():
    r = fl.read_book(book_at(6000.00, 6000.25, bid_sz=250.0, ask_sz=80.0),
                     of.ES, 10.0)
    assert r.bid_size == 250.0 and r.ask_size == 80.0


def test_no_book_is_refused_not_assumed_tight():
    r = fl.read_book(None, of.ES, 10.0)
    assert not r.ok and r.why == "no book"


def test_a_one_sided_book_is_refused():
    b = Book(coin="ES", ts=0.0, bids=[Level(px=6000.0, sz=10.0)], asks=[])
    assert not fl.read_book(b, of.ES, 10.0).ok


# ----------------------------------------------------------- the rungs

def test_delta_is_per_timeframe_and_they_can_disagree():
    """The disagreement is most of the value: the fifteen can still be
    positive while the one has already turned."""
    trades = [tr(t, side="buy", sz=10.0) for t in range(0, 800, 10)]
    trades += [tr(t, side="sell", sz=40.0) for t in range(840, 900, 5)]
    rs = {r.timeframe: r for r in fl.rungs(tape_of(trades), IV, now=899.0)}
    assert rs["1m"].delta < 0, "the current minute is selling"
    assert rs["15m"].delta > 0, "the quarter hour is still net buying"


def test_a_rung_the_tape_cannot_reach_is_marked_uncovered():
    """Not flat. An hour of missing data is not an hour of balance."""
    trades = [tr(t) for t in range(0, 120)]
    rs = {r.timeframe: r for r in fl.rungs(tape_of(trades), IV, now=119.0)}
    assert rs["1m"].covered
    assert not rs["1h"].covered
    assert rs["1h"].side == "unknown"


def test_coverage_is_reported_as_a_share_of_the_bar():
    trades = [tr(t) for t in range(0, 300)]
    rs = {r.timeframe: r for r in fl.rungs(tape_of(trades), IV, now=299.0)}
    assert rs["1m"].coverage == pytest.approx(1.0)
    assert rs["1h"].coverage < 0.1


def test_cvd_accumulates_across_the_bars_the_tape_holds():
    trades = [tr(t, side="buy", sz=5.0) for t in range(0, 600, 10)]
    rs = {r.timeframe: r for r in fl.rungs(tape_of(trades), IV, now=599.0)}
    assert rs["1m"].bars > 1
    assert rs["1m"].cvd > rs["1m"].delta, "cvd is only the current bar"


def test_a_balanced_tape_leans_nowhere():
    trades = []
    for t in range(0, 300, 2):
        trades.append(tr(t, side="buy", sz=5.0))
        trades.append(tr(t + 1, side="sell", sz=5.0))
    rs = {r.timeframe: r for r in fl.rungs(tape_of(trades), IV, now=299.0)}
    assert abs(rs["1m"].lean) < 0.2


def test_an_empty_tape_returns_a_rung_per_timeframe_anyway():
    rs = fl.rungs(tape_of([]), IV)
    assert len(rs) == len(IV)
    assert all(not r.covered for r in rs)


def test_the_dict_carries_what_the_panel_shows():
    d = fl.rungs(tape_of([tr(1)]), IV)[0].to_dict()
    for k in ("timeframe", "delta", "cvd", "lean", "coverage", "covered",
              "side", "trades"):
        assert k in d


# ------------------------------------------------------- the agreement

def test_only_covered_rungs_vote():
    """Silence must not outvote the rungs that measured something."""
    rs = [fl.Rung("1m", 60.0, buy=100.0, covered=True),
          fl.Rung("5m", 300.0, buy=100.0, covered=True),
          fl.Rung("4h", 14400.0, covered=False)]
    assert fl.agreement(rs) == "buyers"


def test_one_rung_the_other_way_is_mixed():
    rs = [fl.Rung("1m", 60.0, sell=100.0, covered=True),
          fl.Rung("5m", 300.0, buy=100.0, covered=True)]
    assert fl.agreement(rs) == "mixed"


def test_nothing_covered_is_unknown_not_agreement():
    assert fl.agreement([fl.Rung("1m", 60.0)]) == "unknown"


# ------------------------------------------------------------- the gate

def busy(n=900, hz=10.0, side="buy"):
    """A quiet baseline then a burst, so velocity has something to see."""
    slow = [tr(i * 1.0, side=side, sz=5.0) for i in range(600)]
    fast = [tr(600.0 + i / hz, side=side, sz=5.0) for i in range(int(15 * hz))]
    return slow + fast


def test_everything_agreeing_is_tradeable():
    lad = fl.build(tape_of(busy()), book_at(6000.00, 6000.25), IV,
                   of.ES, target_ticks=10.0, now=615.0)
    assert lad.velocity.spiking(2.0)
    assert lad.book.ok
    assert lad.tradeable, lad.why


def test_a_wide_spread_alone_refuses_it():
    lad = fl.build(tape_of(busy()), book_at(6000.00, 6000.75), IV,
                   of.ES, target_ticks=10.0, now=615.0)
    assert not lad.tradeable
    assert "spread" in lad.why


def test_a_quiet_tape_alone_refuses_it():
    slow = [tr(i * 1.0, sz=5.0) for i in range(700)]
    lad = fl.build(tape_of(slow), book_at(6000.00, 6000.25), IV,
                   of.ES, target_ticks=10.0, now=699.0)
    assert not lad.tradeable
    assert "baseline" in lad.why


def test_disagreeing_timeframes_refuse_it():
    """The turn: the minute has flipped to sellers while the five and the
    quarter hour are still net buyers. The bar boundaries have to differ
    for the two to be able to disagree at all -- at :00 the 1m and the 5m
    start together and share a window."""
    now = 700.0                      # 1m bar from 660, 5m bar from 600
    trades = [tr(i * 1.0, side="buy", sz=4.0) for i in range(600)]
    trades += [tr(600.0 + i / 2.0, side="buy", sz=40.0) for i in range(170)]
    trades += [tr(685.0 + i / 10.0, side="sell", sz=30.0) for i in range(150)]
    rs = {r.timeframe: r for r in fl.rungs(tape_of(trades), IV, now=now)}
    assert rs["1m"].delta < 0, "the minute should have turned"
    assert rs["5m"].delta > 0, "the five should still be net buying"

    lad = fl.build(tape_of(trades), book_at(6000.00, 6000.25), IV,
                   of.ES, target_ticks=10.0, now=now)
    assert lad.aligned == "mixed"
    assert not lad.tradeable


def test_the_refusal_says_which_gate_said_no():
    """A gate that refuses without saying why is a gate you end up
    disabling."""
    lad = fl.build(tape_of([tr(1.0)]), None, IV, of.ES, now=1.0)
    assert not lad.tradeable
    assert len(lad.why) > 10


def test_the_markets_velocity_is_one_number_not_one_per_rung():
    """The ladder's velocity is the last fifteen seconds against the last
    ten minutes. That belongs to the moment, not to a bar, so it is
    reported once -- repeating it on every row would be one measurement
    pretending to be five."""
    lad = fl.build(tape_of(busy()), book_at(6000.00, 6000.25), IV,
                   of.ES, now=615.0)
    d = lad.to_dict()
    assert "velocity" in d
    assert all("velocity" not in r for r in d["rungs"])


def test_every_rung_carries_its_own_bar_pace():
    """And a rung's pace IS per timeframe, because it is a different
    measurement: this bar against the completed bars of this timeframe,
    the same shape as the delta and CVD beside it on the row."""
    lad = fl.build(tape_of(busy()), book_at(6000.00, 6000.25), IV,
                   of.ES, now=615.0)
    d = lad.to_dict()
    assert all("pace" in r for r in d["rungs"])
    assert all("ratio" in r["pace"] and "note" in r["pace"]
               for r in d["rungs"])


def test_the_rungs_paces_are_not_all_the_same_figure():
    """The test that fails if the per-rung pace is quietly the market
    velocity copied down the column.

    Ten minutes at 1/s then a minute at 5/s: the one-minute rung is
    comparing against six minutes that already contain most of the
    burst, the fifteen against an hour that barely notices it. Equal
    numbers here would mean one measurement wearing five labels.
    """
    trades = [tr(i * 1.0) for i in range(3540)]
    trades += [tr(3540.0 + i / 5.0) for i in range(450)]
    lad = fl.build(tape_of(trades), book_at(6000.00, 6000.25),
                   (("1m", 60.0), ("15m", 900.0)), of.ES, now=3630.0)
    m1, m15 = lad.rungs[0].pace, lad.rungs[1].pace
    assert m1.confident and m15.confident
    assert abs(m15.ratio - m1.ratio) > 1.0
    assert abs(m1.ratio - lad.velocity.ratio) > 0.1


def test_a_rung_whose_history_the_tape_lacks_reports_no_pace():
    """Coverage and pace are different refusals: coverage is about the
    bar in progress, pace is about the bars behind it. An hour of tape
    can cover a 15-minute bar and still have no four-hour history."""
    lad = fl.build(tape_of(busy()), book_at(6000.00, 6000.25),
                   (("1m", 60.0), ("4h", 14400.0)), of.ES, now=615.0)
    four = [r for r in lad.rungs if r.timeframe == "4h"][0]
    assert not four.pace.confident
    assert four.pace.note


def test_tradeable_is_not_a_direction():
    """It says nothing is refusing. Reading it as 'go long' is the one
    way this panel could lose money on its own."""
    lad = fl.build(tape_of(busy()), book_at(6000.00, 6000.25), IV,
                   of.ES, now=615.0)
    assert isinstance(lad.tradeable, bool)
    assert not hasattr(lad, "side")
    assert lad.aligned in ("buyers", "sellers", "mixed", "unknown")


def test_nothing_in_here_places_an_order():
    import inspect
    src = inspect.getsource(fl).lower()
    for word in ("place_order", "submit", "api_key", "private_key"):
        assert word not in src


# ------------------------------------------------- the chart integration

def _src():
    return open("liqmap/web.py").read()


def test_the_route_exists_once_and_is_token_protected():
    import re
    src = _src()
    assert src.count('@app.get("/api/flow"') == 1
    m = re.search(r'@app\.get\("/api/flow"[^)]*\)', src)
    assert "require_token" in m.group(0)


def test_the_route_refuses_without_a_live_feed():
    """Polled candles carry no aggressor and no book. Serving a ladder
    from them would be inventing both."""
    src = _src()
    route = src[src.index('@app.get("/api/flow"'):
                src.index('@app.get("/api/mtf"')]
    assert "no live feed" in route
    assert "feed.tape" in route and "feed.book" in route


def test_the_route_reads_every_timeframe_on_the_ladder():
    """The route no longer carries its own list -- it hands the ask to
    `wanted`, which falls back to this one."""
    src = _src()
    route = src[src.index('@app.get("/api/flow"'):
                src.index('@app.get("/api/mtf"')]
    assert "_fl.wanted(asked, _IV)" in route
    for tf in ("1m", "5m", "15m", "30m", "1h", "4h"):
        assert tf in fl.DEFAULT_TIMEFRAMES


def test_the_tick_is_inferred_rather_than_assumed():
    """The same panel serves a $4 token and a $100k one. A per-market
    table is a chance to be wrong about every market not in it."""
    src = _src()
    route = src[src.index('@app.get("/api/flow"'):
                src.index('@app.get("/api/mtf"')]
    assert "bk.asks[0].px" in route and "bk.bids[0].px" in route


def test_an_uncovered_rung_is_drawn_differently_not_as_flat():
    src = _src()
    fn = src[src.index("function flowHalf(r) {"):]
    fn = fn[:fn.index("\nfunction renderLadder(")]
    assert "NOT COVERED" in fn
    assert "r.coverage" in fn
    # And the refusal is actually wired to coverage, not merely spelled
    # out nearby: a greyed row nobody can reach is a row that reads flat.
    assert "if (!r.covered)" in fn
    assert fn.index("if (!r.covered)") < fn.index("NOT COVERED")
    assert fn.index("NOT COVERED") < fn.index("r.lean")


def test_the_panel_never_calls_the_gate_a_direction():
    src = _src()
    i = src.index('id="flowGate"')
    block = src[i - 1400:i]
    assert "not a direction" in block


def test_the_ladder_refreshes_from_the_paint_not_the_fetch():
    """THE stuck-ladder bug.

    When the socket is live the stream calls paintRead DIRECTLY -- that is
    why it is split out, so a push costs no fetch. Hanging the flow
    refresh off loadRead meant it only ran on the HTTP path: the candle
    read updated from the socket while the ladder sat frozen on whatever
    the last poll returned. Dead-looking data that was really stale data.
    """
    src = _src()
    paint = src[src.index("function paintRead(d) {"):]
    paint = paint[:paint.index("\n}\n")]
    assert "loadFlow()" in paint, "the ladder does not refresh on a push"

    fetch = src[src.index("async function loadRead("):]
    fetch = fetch[:fetch.index("function paintRead(")]
    assert "loadFlow()" not in fetch, (
        "the ladder refreshes twice on the HTTP path")


def test_the_ladder_refresh_is_throttled():
    """paintRead fires many times a second on a busy socket and each flow
    read walks the whole tape."""
    src = _src()
    assert "FLOW_EVERY_MS" in src
    paint = src[src.index("function paintRead(d) {"):]
    paint = paint[:paint.index("\n}\n")]
    assert "FLOW_EVERY_MS" in paint


def test_one_measured_rung_is_not_a_consensus():
    """A panel saying 'every covered timeframe agrees' over a single row
    is making a claim about a stack it cannot see. On a freshly connected
    feed that is the normal state for the first few minutes."""
    rs = [fl.Rung("1m", 60.0, buy=100.0, covered=True),
          fl.Rung("5m", 300.0, covered=False)]
    assert fl.agreement(rs) == "unknown"
    assert len(fl.voters(rs)) == 1


def test_two_measured_rungs_agreeing_is_a_consensus():
    rs = [fl.Rung("1m", 60.0, buy=100.0, covered=True),
          fl.Rung("5m", 300.0, buy=100.0, covered=True)]
    assert fl.agreement(rs) == "buyers"


def test_the_gate_says_when_it_has_only_one_timeframe():
    trades = [tr(i * 1.0, side="buy", sz=5.0) for i in range(600)]
    trades += [tr(600.0 + i / 10.0, side="buy", sz=5.0) for i in range(150)]
    lad = fl.build(tape_of(trades), book_at(6000.00, 6000.25),
                   (("1m", 60.0), ("4h", 14400.0)), of.ES, now=615.0)
    assert not lad.tradeable
    assert "measured" in lad.why


def test_every_function_the_markup_calls_is_defined():
    src = _src()
    for fn in ("loadFlow", "flowNum"):
        assert f"function {fn}(" in src or f"async function {fn}(" in src


def test_the_elements_the_script_writes_to_exist():
    src = _src()
    for el in ("flowGate", "tfLadder", "flowStamp", "flowTgt",
               "flowSpike"):
        assert f'id="{el}"' in src, f"{el} is missing from the markup"
        assert src.count(f'id="{el}"') == 1


def test_every_css_variable_the_stylesheet_uses_is_defined():
    """--long and --short were used in sixteen places and defined in
    none, so every one resolved to nothing and the ladder's bars, the
    status dot and the alert bar all rendered colourless.

    Checked for the whole stylesheet rather than those two, because the
    failure is silent: a browser drops the declaration and paints the
    default, and nothing anywhere says so.
    """
    import re

    from liqmap.web import DASHBOARD as page
    css = page[page.index("<style>"):page.index("</style>")]
    used = set(re.findall(r"var\((--[a-z0-9-]+)\)", css))
    declared = set(re.findall(r"(--[a-z0-9-]+)\s*:", css))
    missing = sorted(used - declared)
    assert not missing, f"used but never defined: {missing}"


# --------------------------------------------- the two ladders, merged
#
# The candle read and the order flow measure the same bar from different
# feeds, and the whole value is in whether they agree. Two ladders a
# screen apart made that a memory test. Merged, they must stay two
# measurements: one row, two halves, neither blended into the other.


def test_one_row_carries_both_readings():
    src = _src()
    row = src[src.index("function renderLadder() {"):]
    row = row[:row.index("\nfunction paintLadder(")]
    assert "readHalf(t)" in row and "flowHalf(r)" in row
    assert "tf2" in row


def test_the_halves_are_matched_by_timeframe_not_by_position():
    """Two lists in the same order today is not the same thing as two
    lists in the same order. A row pairing a 15m read with a 5m delta
    would be worse than no row."""
    src = _src()
    row = src[src.index("function renderLadder() {"):]
    row = row[:row.index("\nfunction paintLadder(")]
    assert "r.timeframe" in row and "t.timeframe" in row
    assert "read[n]" in row and "flow[n]" in row


def test_the_flow_is_asked_for_the_rows_the_read_is_showing():
    """Same reason. The page builds the read's timeframe list, then asks
    the flow route for exactly those."""
    src = _src()
    fn = src[src.index("async function loadFlow("):]
    fn = fn[:fn.index("\nfunction flowNum(")]
    assert "timeframes: readTfList()" in fn

    route = src[src.index('@app.get("/api/flow"'):
                src.index('@app.get("/api/mtf"')]
    assert "timeframes: str" in route


def test_the_rows_asked_for_are_the_rows_built():
    IVS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400}
    assert fl.wanted(["15m", "1m"], IVS) == [("15m", 900.0), ("1m", 60.0)]


def test_a_timeframe_this_feed_does_not_have_is_dropped_not_guessed():
    assert fl.wanted(["15m", "7m"], {"15m": 900}) == [("15m", 900.0)]


def test_asking_twice_for_a_row_does_not_draw_it_twice():
    assert fl.wanted(["1m", "1m"], {"1m": 60}) == [("1m", 60.0)]


def test_an_empty_ask_is_no_opinion_not_no_rows():
    """The panel can be asked for nothing -- by an old client, or before
    the read has chosen its timeframes. That is a caller with no opinion,
    and a blank ladder would read as a dead feed."""
    got = [n for n, _ in fl.wanted([], {k: 1 for k in fl.DEFAULT_TIMEFRAMES})]
    assert got == list(fl.DEFAULT_TIMEFRAMES)
    assert "4h" in got


def test_the_page_builds_the_same_list_the_route_does():
    """The read route's default ladder and the page's copy of it have to
    stay the same list, or the halves drift apart the first time either
    is edited."""
    import inspect

    from liqmap import web as w
    route = inspect.getsource(w.create_app)
    i = route.index("tf_list = [higher")
    server = route[i:i + 80]
    src = _src()
    page = src[src.index("function readTfList() {"):]
    page = page[:page.index("\n}")]
    for tf in ('"1h"', '"5m"', '"1m"'):
        assert tf in server
        assert tf.replace('"', "'") in page
    assert "rHigh" in page and "rInt" in page


def test_a_half_with_nothing_behind_it_says_so_rather_than_reading_flat():
    """The two feeds arrive on their own schedules and either can be
    missing. A silent half looks exactly like a balanced one."""
    src = _src()
    read = src[src.index("function readHalf(t) {"):]
    read = read[:read.index("\nfunction flowHalf(")]
    assert "NO READ" in read

    flow = src[src.index("function flowHalf(r) {"):]
    flow = flow[:flow.index("\nfunction renderLadder(")]
    assert "NO TAPE" in flow


def test_each_rungs_own_pace_is_on_its_row():
    src = _src()
    flow = src[src.index("function flowHalf(r) {"):]
    flow = flow[:flow.index("\nfunction renderLadder(")]
    assert "r.pace" in flow
    assert "p.confident" in flow, "an unmeasured pace must not print a ratio"


def test_the_merged_row_has_a_style_of_its_own():
    from liqmap.web import DASHBOARD as page
    css = page[page.index("<style>"):page.index("</style>")]
    assert ".tf2{" in css
    assert ".tf2 .half{" in css
    # Each half carries its own left edge, so a row where the read and
    # the flow disagree is visible without reading a word.
    assert ".tf2 .half.buyers{" in css and ".tf2 .half.sellers{" in css


def test_the_old_second_ladder_is_gone():
    src = _src()
    assert 'id="flowLadder"' not in src
    assert "Order flow — the book gates" not in src


def test_the_two_halves_of_a_row_cannot_be_different_bars():
    """The guarantee the merge rests on, exercised rather than grepped.

    The page builds the read's timeframe list and asks the flow route for
    exactly it, so every rung that comes back belongs to a row the read
    is drawing -- in the same order, which is what lets the halves be
    matched by name without one of them silently sliding a row.
    """
    IVS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400}
    asked = ["4h", "1h", "15m", "5m", "1m"]          # as the read orders them
    lad = fl.build(tape_of(busy()), book_at(6000.00, 6000.25),
                   fl.wanted(asked, IVS), of.ES, now=615.0)
    assert [r.timeframe for r in lad.rungs] == asked


def test_the_meta_line_does_not_clip_the_age_off_a_stale_row():
    """The row that wraps is always the stale one -- the row you most
    need to read straight. Side by side it is held to one line; stacked,
    where nothing misaligns, it is allowed to wrap rather than ellipsis
    the age away."""
    from liqmap.web import DASHBOARD as page
    css = page[page.index("<style>"):page.index("</style>")]
    wide = css[css.index(".tf2 .meta{"):]
    wide = wide[:wide.index("}")]
    assert "nowrap" in wide
    narrow = css[css.index("@media (max-width:980px)"):]
    narrow = narrow[:narrow.index("\n  }")]
    assert ".tf2 .meta{white-space:normal}" in narrow


# ------------------------------------- CVD that outlives the rolling tape
#
# The tape holds about an hour. A four-hour rung has seen a quarter of one
# bar, so its CVD cannot come from the tape at any price -- which is why
# every slow row sat at NOT COVERED. The completed bars are kept instead.


class FakeCVD:
    def __init__(self, cvd, bars, source="stored"):
        self.cvd, self.bars, self.source = cvd, bars, source


def test_a_rung_takes_its_cvd_from_the_completed_bars_when_it_can():
    seen = []

    def cvd_for(name, iv, bar_open, bars):
        seen.append((name, iv, bar_open, bars))
        return FakeCVD(500_000.0, 6)

    rs = fl.rungs(tape_of(busy()), (("4h", 14400.0),), now=615.0,
                  cvd_for=cvd_for)
    assert seen and seen[0][0] == "4h"
    assert rs[0].bars == 6
    assert rs[0].cvd_source == "stored"


def test_the_bar_in_progress_is_added_to_the_stored_bars_not_counted_twice():
    """The store holds CLOSED bars. The live bar is the tape's, and it has
    to be added exactly once."""
    def cvd_for(name, iv, bar_open, bars):
        return FakeCVD(1000.0, 6)

    rs = fl.rungs(tape_of(busy()), (("1m", 60.0),), now=615.0,
                  cvd_for=cvd_for)
    assert rs[0].cvd == pytest.approx(1000.0 + rs[0].delta)


def test_the_completed_bars_are_asked_for_before_this_bars_open():
    """Anything at or after the open is the bar in progress, and it is
    added separately -- asking for it here would count it twice."""
    grabbed = {}

    def cvd_for(name, iv, bar_open, bars):
        grabbed["open"] = bar_open
        return FakeCVD(0.0, 1)

    fl.rungs(tape_of(busy()), (("1m", 60.0),), now=615.0, cvd_for=cvd_for)
    assert grabbed["open"] == pytest.approx(600.0)


def test_without_a_store_the_cvd_falls_back_to_the_tape_and_says_so():
    rs = fl.rungs(tape_of(busy()), (("1m", 60.0),), now=615.0)
    assert rs[0].cvd_source == "tape"


def test_a_rung_reports_how_many_bars_its_cvd_actually_had():
    def cvd_for(name, iv, bar_open, bars):
        return FakeCVD(10.0, 2)

    rs = fl.rungs(tape_of(busy()), (("4h", 14400.0),), now=615.0,
                  cvd_for=cvd_for)
    d = rs[0].to_dict()
    assert d["bars"] == 2 and d["asked"] == fl.RUNGS_BACK
    assert not d["cvd_complete"]


def test_the_row_shows_a_short_cvd_as_short():
    src = _src()
    fn = src[src.index("function flowHalf(r) {"):]
    fn = fn[:fn.index("\nfunction renderLadder(")]
    assert "cvd_complete" in fn and "r.asked" in fn
    assert "cvd_source === 'none'" in fn


def test_the_route_hands_the_ladder_somewhere_to_get_bars_from():
    src = _src()
    route = src[src.index('@app.get("/api/flow"'):
                src.index('@app.get("/api/mtf"')]
    assert "cvd_for=cvd_for" in route
    assert "rt.bar_cvd(" in route
