"""The live order-flow ladder: delta per timeframe, the book at the touch.

The traps here are about silence.

A timeframe the tape cannot reach has NO opinion. Reporting it as a flat
delta makes an hour of missing data look like an hour of balance, and
then letting it vote makes silence outweigh the rungs that actually
measured something.

And velocity is a rate, not a quantity. There is no four-hour version of
"prints per second in the last fifteen seconds", so printing one per rung
would be the same number five times pretending to be five measurements.
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


def test_velocity_is_one_number_not_one_per_rung():
    """A rate belongs to the moment, not to a bar. Printing it per
    timeframe would be the same figure five times pretending to be five
    measurements."""
    lad = fl.build(tape_of(busy()), book_at(6000.00, 6000.25), IV,
                   of.ES, now=615.0)
    d = lad.to_dict()
    assert "velocity" in d
    assert all("velocity" not in r for r in d["rungs"])


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
    src = _src()
    route = src[src.index('@app.get("/api/flow"'):
                src.index('@app.get("/api/mtf"')]
    for tf in ('"1m"', '"5m"', '"15m"', '"30m"', '"1h"', '"4h"'):
        assert tf in route


def test_the_tick_is_inferred_rather_than_assumed():
    """The same panel serves a $4 token and a $100k one. A per-market
    table is a chance to be wrong about every market not in it."""
    src = _src()
    route = src[src.index('@app.get("/api/flow"'):
                src.index('@app.get("/api/mtf"')]
    assert "bk.asks[0].px" in route and "bk.bids[0].px" in route


def test_an_uncovered_rung_is_drawn_differently_not_as_flat():
    src = _src()
    fn = src[src.index("async function loadFlow("):]
    fn = fn[:fn.index("\nfunction flowNum(")]
    assert "NOT COVERED" in fn
    assert "r.coverage" in fn


def test_the_panel_never_calls_the_gate_a_direction():
    src = _src()
    i = src.index('id="flowGate"')
    block = src[i - 1400:i]
    assert "not a direction" in block


def test_the_ladder_refreshes_with_the_read():
    """Two clocks would let it drift out of step with the panel it sits
    under."""
    src = _src()
    fn = src[src.index("async function loadRead("):]
    fn = fn[:fn.index("function paintRead(")]
    assert "loadFlow()" in fn


def test_every_function_the_markup_calls_is_defined():
    src = _src()
    for fn in ("loadFlow", "flowNum"):
        assert f"function {fn}(" in src or f"async function {fn}(" in src


def test_the_elements_the_script_writes_to_exist():
    src = _src()
    for el in ("flowGate", "flowLadder", "flowStamp", "flowTgt",
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
