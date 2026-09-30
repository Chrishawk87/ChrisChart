"""The live structural read: does it describe what is actually there?

The distinction this module exists to keep is acceptance versus location.
Price pokes outside the value area constantly; the auction agreeing to
trade there is a different fact. Conflating them turns every probe into a
trend, which is the most common way a profile read goes wrong -- so most of
these tests are about that boundary.
"""

from __future__ import annotations

import numpy as np
import pytest

from liqmap import marketstate as ms
from liqmap import profile as pf


class Bar:
    __slots__ = ("ts", "open", "high", "low", "close", "volume")

    def __init__(self, ts, o, h, l, c, v=100.0):
        self.ts, self.open, self.high = ts, o, h
        self.low, self.close, self.volume = l, c, v


def bar(i, c):
    return Bar(i * 300.0, c, c + 0.25, c - 0.25, c)


def bars(closes):
    return [bar(i, c) for i, c in enumerate(closes)]


def prof():
    """A profile with VAL 7000, POC 7010, VAH 7020 and a thin row at 7014."""
    rows = []
    for i in range(3000):
        rows.append(Bar(i, 7010.0, 7010.0, 7010.0, 7010.0, 400.0))
    for i in range(900):
        rows.append(Bar(3000 + i, 7003.0 + (i % 28) * 0.25,
                        7003.0 + (i % 28) * 0.25, 7003.0 + (i % 28) * 0.25,
                        7003.0 + (i % 28) * 0.25, 120.0))
    for i in range(30):
        rows.append(Bar(5000 + i, 7014.0, 7014.0, 7014.0, 7014.0, 5.0))
    for i in range(700):
        rows.append(Bar(6000 + i, 7018.0 + (i % 12) * 0.25,
                        7018.0 + (i % 12) * 0.25, 7018.0 + (i % 12) * 0.25,
                        7018.0 + (i % 12) * 0.25, 150.0))
    return pf.final(rows)


# ------------------------------------------------------------- where

def test_inside_the_value_area_reads_as_inside():
    p = prof()
    r = ms.read(bars([p.poc]), p)
    assert r.zone == "inside value"


def test_above_the_value_area_reads_as_above():
    p = prof()
    r = ms.read(bars([p.vah + 5.0]), p)
    assert r.zone == "above value"


def test_below_the_value_area_reads_as_below():
    p = prof()
    assert ms.read(bars([p.val - 5.0]), p).zone == "below value"


def test_sitting_on_a_node_names_it():
    p = prof()
    r = ms.read(bars([p.poc] * 4), p)
    assert r.at_node is not None
    assert r.node_price == pytest.approx(p.poc, abs=1.0)


def test_being_far_from_every_node_names_none():
    p = prof()
    r = ms.read(bars([p.vah + 40.0]), p)
    assert r.at_node is None


def test_the_nearest_nodes_bracket_the_price():
    p = prof()
    r = ms.read(bars([p.poc + 1.0]), p)
    if r.nearest_above is not None:
        assert r.nearest_above > r.price
    if r.nearest_below is not None:
        assert r.nearest_below < r.price


# ================================================== acceptance vs location

def test_one_close_beyond_the_edge_is_not_acceptance():
    """THE distinction. A single poke outside value is an excursion, and
    calling it acceptance turns every probe into a trend."""
    p = prof()
    r = ms.read(bars([p.poc, p.vah + 2.0]), p)
    assert r.zone == "above value"          # location says beyond
    assert r.accepted == "no"               # acceptance says not yet
    assert r.doing == "probing"


def test_two_closes_beyond_is_acceptance():
    p = prof()
    r = ms.read(bars([p.poc, p.vah + 2.0, p.vah + 3.0]), p)
    assert r.accepted == "above"
    assert r.doing == "leaving"


def test_acceptance_below_is_the_mirror():
    p = prof()
    r = ms.read(bars([p.poc, p.val - 2.0, p.val - 3.0]), p)
    assert r.accepted == "below"


def test_acceptance_must_be_consecutive():
    p = prof()
    r = ms.read(bars([p.vah + 2.0, p.poc, p.vah + 2.0]), p)
    assert r.accepted == "no"


def test_coming_back_inside_ends_acceptance():
    p = prof()
    r = ms.read(bars([p.vah + 2.0, p.vah + 3.0, p.poc, p.poc]), p)
    assert r.accepted == "no"
    assert r.zone == "inside value"


# -------------------------------------------------------------- doing

def test_sitting_on_a_node_reads_as_holding():
    p = prof()
    r = ms.read(bars([p.poc] * 6), p)
    assert r.doing == "holding"
    assert "sitting on" in r.note


def test_crossing_a_node_reads_as_through():
    p = prof()
    r = ms.read(bars([p.poc - 2.0, p.poc - 1.0, p.poc + 1.0, p.poc]), p)
    assert r.doing == "through"


def test_drifting_inside_value_reads_as_rotating():
    p = prof()
    mid = (p.vah + p.val) / 2
    r = ms.read(bars([mid - 3.0, mid - 2.0, mid - 1.0, mid]), p)
    if r.at_node is None:
        assert r.doing == "rotating"


def test_the_note_always_says_something_concrete():
    p = prof()
    for closes in ([p.poc] * 6, [p.vah + 2, p.vah + 3],
                   [p.val - 2, p.val - 3], [p.poc - 5, p.poc]):
        r = ms.read(bars(closes), p)
        assert r.note and len(r.note) > 10


# ------------------------------------------------------------ safety

def test_no_bars_reads_as_no_profile_rather_than_crashing():
    assert ms.read([], prof()).note == "no profile yet"


def test_no_profile_reads_as_no_profile():
    r = ms.read(bars([7010.0]), None)
    assert r.doing == "unclear"


def test_an_empty_profile_does_not_raise():
    assert ms.read(bars([7010.0]), pf.final([])).note == "no profile yet"


def test_the_dict_carries_every_field_the_panel_shows():
    p = prof()
    d = ms.read(bars([p.poc] * 4), p).to_dict()
    for k in ("price", "zone", "at_node", "accepted", "doing",
              "nearest_above", "nearest_below", "note"):
        assert k in d


# --------------------------------------------------------- histogram

def test_the_histogram_merges_only_for_drawing():
    """One tick per row is right to compute on and far too fine to draw --
    a 400-tick session would be 400 bars two pixels apart."""
    p = prof()
    rows = ms.histogram_rows(p, max_rows=40)
    assert len(rows) <= 40
    assert all(r["hi"] > r["lo"] for r in rows)


def test_the_shares_are_normalised_to_the_heaviest_row():
    rows = ms.histogram_rows(prof(), max_rows=40)
    assert max(r["share"] for r in rows) == pytest.approx(1.0)
    assert all(0.0 <= r["share"] <= 1.0 for r in rows)


def test_merging_conserves_volume():
    p = prof()
    rows = ms.histogram_rows(p, max_rows=30)
    assert sum(r["v"] for r in rows) == pytest.approx(p.total, rel=1e-3)


def test_an_empty_profile_has_no_rows():
    assert ms.histogram_rows(pf.final([])) == []


def test_the_conventions_are_the_stated_ones():
    assert ms.ACCEPT_CLOSES == 2
    assert ms.AT_TICKS == 3.0
    assert ms.HOLD_BARS == 3


# ------------------------------------------------- the chart integration

def test_the_profile_route_exists_and_is_token_protected():
    import re
    src = open("liqmap/web.py").read()
    assert '@app.get("/api/profile"' in src
    m = re.search(r'@app\.get\("/api/profile"[^)]*\)', src)
    assert "require_token" in m.group(0)


def test_the_profile_route_is_defined_exactly_once():
    """A second definition on the same path silently replaces the first --
    this project has been bitten by that twice."""
    src = open("liqmap/web.py").read()
    assert src.count('@app.get("/api/profile"') == 1


def test_the_profile_is_drawn_before_the_candles():
    """Painted over the price action it hides the thing it explains."""
    src = open("liqmap/web.py").read()
    hook = src.index("drawProfile(g, hi, lo, plotW, plotH);")
    candles = src.index("/* ---- grid ---")
    assert hook < candles


def test_every_function_the_markup_calls_is_defined():
    src = open("liqmap/web.py").read()
    for fn in ("cycleProfile", "loadProfile", "drawProfile"):
        assert f"function {fn}(" in src or f"async function {fn}(" in src


def test_the_readout_element_the_script_writes_to_exists():
    src = open("liqmap/web.py").read()
    assert 'id="profRead"' in src
    assert "$('profRead')" in src


def test_the_profile_uses_the_same_candle_source_as_the_chart():
    """Two routes disagreeing about where bars come from would draw a
    profile that does not match the candles beside it."""
    src = open("liqmap/web.py").read()
    route = src[src.index('@app.get("/api/profile"'):
                src.index('@app.get("/api/chart"')]
    assert "feed.history(interval)" in route
    assert "rt.client().candles(" in route


def test_the_read_is_taken_against_the_reference_not_the_session():
    """Reading today's own POC as a level price is bouncing off is
    circular -- the bounce is part of what put the POC there."""
    src = open("liqmap/web.py").read()
    route = src[src.index('@app.get("/api/profile"'):
                src.index('@app.get("/api/chart"')]
    assert "_ms.read(rows[cut:], reference)" in route


def test_the_reference_splits_on_a_session_boundary_where_one_exists():
    """A rolling half-and-half split mixes part of today into the shape
    price is supposedly reacting to, which makes today's own structure a
    level. Where the market has sessions, the boundary is the last one."""
    src = open("liqmap/web.py").read()
    route = src[src.index('@app.get("/api/profile"'):
                src.index('@app.get("/api/chart"')]
    assert "trade_date" in route
    assert "_td(stamps[-1])" in route


def test_the_rolling_split_survives_as_a_fallback():
    """Crypto has no session boundary. The fallback is a real limitation
    of the reference on a 24/7 market, so it stays visible rather than
    pretending every market has an open."""
    src = open("liqmap/web.py").read()
    route = src[src.index('@app.get("/api/profile"'):
                src.index('@app.get("/api/chart"')]
    assert "cut = max(20, len(rows) // 2)" in route
    assert "Crypto has no session boundary" in route
