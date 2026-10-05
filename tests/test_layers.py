"""One candle, six layers, no verdict.

The work here is assembly, not measurement, so the tests are about the
things assembly gets wrong.

MISSING IS NOT NEUTRAL. A feed that has not warmed up and a market in
balance produce the same zeros. Every layer has to say which it is, or the
seventh layer will build a confident word on top of nothing.

NOTHING IS RECOMPUTED. If this file starts doing its own arithmetic on the
tape, two modules own the same number and they will disagree.

LOCATION IS RELATIVE TO THE BAR. A level twenty points away is in play on
a four-hour bar and irrelevant on a one-minute one.
"""

from __future__ import annotations

import pytest

from liqmap import layers as ly


class P:
    """Just the fields `assemble` reads off a Pressure."""

    def __init__(self, aggressor="buy", lean=0.8, trades=120,
                 total_notional=1e6, delta=5e5, open_px=100.0, last_px=101.0,
                 high_px=101.5, low_px=99.5, measured=True, forming=False):
        self.aggressor, self.lean, self.trades = aggressor, lean, trades
        self.total_notional, self.delta = total_notional, delta
        self.open_px, self.last_px = open_px, last_px
        self.high_px, self.low_px = high_px, low_px
        self.measured, self.forming = measured, forming


class V:
    def __init__(self, ratio=3.0, confident=True):
        self.ratio, self.confident = ratio, confident


class Pace:
    def __init__(self, pace=1.8, known=True):
        self.pace, self.known = pace, known


class A:
    def __init__(self, absorbing=False, impact_ratio=1.0, confident=True):
        self.absorbing = absorbing
        self.impact_ratio = impact_ratio
        self.confident = confident


class D:
    def __init__(self, measured=True, text="bids stacking"):
        self.measured, self._t = measured, text

    def describe(self):
        return self._t

    def to_dict(self):
        return {"describe": self._t, "measured": self.measured}


class Book:
    empty = False

    def depth(self, bps, side):
        return 900.0 if side == "buy" else 300.0


class Prof:
    def __init__(self, rows):
        self._rows = rows

    def levels(self):
        return self._rows


# ------------------------------------------------------- missing is not neutral

def test_every_layer_says_whether_it_was_measured():
    bare = ly.assemble("15m", 900.0)
    assert bare.measured == []
    assert set(bare.missing) == {"intent", "force", "resistance",
                                 "liquidity", "response", "location"}


def test_an_inferred_intent_is_not_a_counted_one():
    """Without a tape the buy/sell split is guessed from where the bar
    closed in its range, which is a weaker instrument and must not be
    mistaken for fills."""
    out = ly.assemble("15m", 900.0, pressure=P(measured=False))
    assert out.intent.who == "buyers"
    assert not out.intent.measured
    assert "inferred" in out.intent.describe()


def test_a_baseline_with_no_scale_cannot_judge_absorption():
    out = ly.assemble("15m", 900.0, pressure=P(),
                      absorption=A(absorbing=True, confident=False))
    assert not out.resistance.measured
    assert "no impact baseline" in out.resistance.describe()


def test_a_velocity_that_is_not_confident_does_not_count_as_force():
    out = ly.assemble("1m", 60.0, pressure=P(), velocity=V(confident=False))
    assert not out.force.velocity_known
    assert not out.force.measured


def test_one_known_input_is_enough_for_force():
    out = ly.assemble("1m", 60.0, pressure=P(), pace=Pace(known=True))
    assert out.force.measured and out.force.pace_known


def test_a_book_with_no_history_says_so_rather_than_reading_flat():
    out = ly.assemble("15m", 900.0, pressure=P())
    assert not out.liquidity.measured
    assert "no book history" in out.liquidity.describe()


def test_a_bar_still_forming_has_no_price_response():
    out = ly.assemble("1m", 60.0, pressure=P(forming=True), unit_size=0.25)
    assert not out.response.measured


# --------------------------------------------------------------- 1 intent

def test_intent_is_who_is_crossing_the_spread():
    assert ly.assemble("15m", 900.0,
                       pressure=P(aggressor="buy")).intent.who == "buyers"
    assert ly.assemble("15m", 900.0,
                       pressure=P(aggressor="sell")).intent.who == "sellers"
    assert ly.assemble("15m", 900.0,
                       pressure=P(aggressor="balanced")).intent.who == "neither"


# ---------------------------------------------------------------- 2 force

def test_force_carries_all_four_of_its_inputs():
    out = ly.assemble("15m", 900.0, pressure=P(total_notional=2e6, delta=1e6),
                      velocity=V(ratio=4.0), pace=Pace(pace=2.5))
    f = out.force
    assert f.notional == pytest.approx(2e6)
    assert f.delta == pytest.approx(1e6)
    assert f.velocity == pytest.approx(4.0)
    assert f.pace == pytest.approx(2.5)


# ----------------------------------------------------------- 3 resistance

def test_resistance_reports_where_the_executions_piled_up():
    """Repeated executions at one price is the thing you can see in a
    profile and cannot see in a candle."""
    out = ly.assemble("15m", 900.0, pressure=P(),
                      profile=Prof([(100.0, 10.0), (100.25, 70.0),
                                    (100.5, 20.0)]))
    assert out.resistance.busiest_px == pytest.approx(100.25)
    assert out.resistance.busiest_share == pytest.approx(0.7)


def test_an_empty_profile_is_not_a_crash():
    out = ly.assemble("15m", 900.0, pressure=P(), profile=Prof([]))
    assert out.resistance.busiest_px == 0.0


def test_resting_depth_comes_off_the_book():
    out = ly.assemble("15m", 900.0, pressure=P(), book=Book())
    assert out.resistance.bid_depth == pytest.approx(900.0)
    assert out.resistance.depth_tilt > 0


# -------------------------------------------------------------- 5 response

def test_the_three_cases_are_one_scale_not_three_categories():
    """Attacked and went four ticks, attacked and went one then stalled,
    attacked and did not move -- that is impact, measured against how far
    this much flow normally moves price."""
    big = ly.assemble("1m", 60.0, pressure=P(last_px=101.0),
                      absorption=A(impact_ratio=2.0), unit_size=0.25)
    none = ly.assemble("1m", 60.0, pressure=P(last_px=100.0),
                       absorption=A(absorbing=True, impact_ratio=0.1),
                       unit_size=0.25)
    assert big.response.impact > none.response.impact
    assert none.response.stalled and not big.response.stalled


def test_price_going_against_the_aggressor_is_called_out():
    """Buyers crossing while price falls is the case that matters most and
    the one a direction read alone hides."""
    out = ly.assemble("1m", 60.0,
                      pressure=P(aggressor="buy", open_px=100.0,
                                 last_px=99.0, low_px=98.5, high_px=100.0),
                      unit_size=0.25)
    assert out.response.with_intent is False
    assert "against the side" in out.response.describe()


def test_the_move_is_in_the_markets_own_unit():
    out = ly.assemble("1m", 60.0, pressure=P(open_px=100.0, last_px=101.0),
                      unit_size=0.25, unit="t")
    assert out.response.move == pytest.approx(4.0)
    assert out.response.unit == "t"


def test_with_no_unit_there_is_no_invented_one():
    out = ly.assemble("1m", 60.0, pressure=P())
    assert not out.response.measured
    assert not out.location.measured


# -------------------------------------------------------------- 6 location

def test_the_nearest_reference_comes_first():
    out = ly.assemble("15m", 900.0,
                      pressure=P(last_px=100.0, high_px=102.0, low_px=98.0),
                      unit_size=0.25,
                      refs=[("PDH", 101.5), ("VWAP", 100.25), ("POC", 99.0)])
    assert [r.name for r in out.location.near] == ["VWAP", "POC", "PDH"]
    assert out.location.at.name == "VWAP"


def test_distance_is_signed_so_above_and_below_are_readable():
    out = ly.assemble("15m", 900.0,
                      pressure=P(last_px=100.0, high_px=102.0, low_px=98.0),
                      unit_size=0.25, refs=[("PDH", 101.0), ("PDL", 99.0)])
    by = {r.name: r for r in out.location.near}
    assert by["PDH"].above and by["PDH"].distance > 0
    assert not by["PDL"].above and by["PDL"].distance < 0


def test_what_counts_as_near_scales_with_the_bar():
    """A level twenty points away is in play on a four-hour bar and
    irrelevant on a one-minute one. A fixed cutoff gets one of those wrong
    whichever number is chosen."""
    refs = [("PDH", 110.0)]
    tiny = ly.assemble("1m", 60.0,
                       pressure=P(last_px=100.0, high_px=100.2, low_px=99.8),
                       unit_size=0.25, refs=refs)
    wide = ly.assemble("4h", 14400.0,
                       pressure=P(last_px=100.0, high_px=108.0, low_px=92.0),
                       unit_size=0.25, refs=refs)
    assert tiny.location.near == []
    assert [r.name for r in wide.location.near] == ["PDH"]


def test_a_reference_at_zero_is_dropped_rather_than_plotted():
    out = ly.assemble("15m", 900.0,
                      pressure=P(last_px=100.0, high_px=102.0, low_px=98.0),
                      unit_size=0.25, refs=[("IBH", 0.0), ("VWAP", 100.1)])
    assert [r.name for r in out.location.near] == ["VWAP"]


def test_only_a_handful_of_references_are_reported():
    refs = [(f"L{i}", 100.0 + i * 0.01) for i in range(20)]
    out = ly.assemble("15m", 900.0,
                      pressure=P(last_px=100.0, high_px=102.0, low_px=98.0),
                      unit_size=0.25, refs=refs)
    assert len(out.location.near) <= ly.NEAR_MAX


# --------------------------------------------------------------- the whole

def test_the_six_clauses_read_as_one_description():
    out = ly.assemble("15m", 900.0, pressure=P(), velocity=V(), pace=Pace(),
                      absorption=A(absorbing=True, impact_ratio=0.2),
                      dom_read=D(), book=Book(), unit_size=0.25,
                      refs=[("PDH", 101.2)])
    text = out.describe()
    for layer in ("intent", "force", "resistance", "liquidity", "response",
                  "location"):
        assert layer + ":" in text


def test_nothing_is_classified_here():
    """The seventh layer is a separate module. Keeping the description
    apart from the verdict is what lets a wrong word be traced to the
    reading that produced it."""
    import inspect

    src = inspect.getsource(ly)
    # Everything but the module docstring, where the separation is
    # explained. No code below it may name a result state.
    body = src.replace(ly.__doc__ or "", "").lower()
    for word in ("continuation", "exhaustion", "reversal", "failed_break",
                 "rejection", "expansion"):
        assert word not in body


def test_nothing_is_recomputed_from_raw_feeds():
    """If this file starts doing its own arithmetic on the tape, two
    modules own the same number and they will disagree."""
    import inspect

    src = inspect.getsource(ly)
    for leak in ("aggressor ==", "for t in trades", "notional +=",
                 "10_000.0"):
        assert leak not in src


def test_the_dict_carries_every_layer_for_the_panel():
    d = ly.assemble("15m", 900.0, pressure=P(), unit_size=0.25).to_dict()
    for layer in ("intent", "force", "resistance", "liquidity", "response",
                  "location"):
        assert layer in d and "describe" in d[layer]
    assert "measured" in d and "missing" in d


# ------------------------------------------------------ the panel and route


def _src():
    import inspect

    from liqmap import web as w
    return inspect.getsource(w)


def test_the_panel_is_its_own_and_touches_neither_ladder_nor_read():
    src = _src()
    assert 'id="layerPanel"' in src and 'id="layerBody"' in src
    panel = src[src.index('id="layerPanel"'):]
    panel = panel[:panel.index("</div>\n  </div>")]
    for theirs in ('id="tfLadder"', 'id="flowGate"', 'id="readSignals"',
                   'id="readHead"'):
        assert theirs not in panel


def test_the_panel_paints_from_the_read_the_ladder_already_got():
    """Two fetches would be two moments, and the panel explaining a bar
    the ladder is no longer showing is worse than no panel."""
    src = _src()
    fn = src[src.index("function paintRead(d) {"):]
    fn = fn[:fn.index("\n}\n")]
    assert "paintLayers(d)" in fn
    assert "paintLadder(d)" in fn


def test_all_six_layers_are_drawn_in_the_order_they_happen():
    src = _src()
    block = src[src.index("const LAYER_ORDER = ["):]
    block = block[:block.index("];")]
    order = [n for n in ("intent", "force", "resistance", "liquidity",
                         "response", "location")]
    pos = [block.index('"' + n + '"') for n in order]
    assert pos == sorted(pos)


def test_an_unmeasured_layer_is_dimmed_rather_than_shown_as_zero():
    src = _src()
    fn = src[src.index("function paintLayers(d) {"):]
    fn = fn[:fn.index("\nfunction paintLadder(")]
    assert "part.measured ? '' : ' off'" in fn
    css = src[src.index("<style>"):src.index("</style>")]
    assert ".lay.off{" in css


def test_the_panel_follows_the_timeframe_the_read_is_on():
    src = _src()
    fn = src[src.index("function paintLayers(d) {"):]
    fn = fn[:fn.index("\nfunction paintLadder(")]
    assert "$('rInt')" in fn


def test_the_route_shares_the_book_and_the_tape_across_the_rows():
    """They are properties of the moment rather than of a bar. Computing
    them per timeframe would be five chances for one number to come out
    differently."""
    src = _src()
    block = src[src.index("lay: dict[str, Any] = {}"):]
    block = block[:block.index('out["timeframes"] = [')]
    assert block.index("dom_read = feed.dom.read(") < block.index("for pres in readings:")
    assert block.index("vel = _of_velocity(") < block.index("for pres in readings:")


def test_the_route_hands_over_rather_than_recomputing():
    src = _src()
    block = src[src.index("lay: dict[str, Any] = {}"):]
    block = block[:block.index('out["timeframes"] = [')]
    assert "lyr.assemble(" in block
    assert "pressure=pres" in block


def test_a_broken_layer_read_does_not_take_the_read_down():
    src = _src()
    block = src[src.index("lay: dict[str, Any] = {}"):]
    block = block[:block.index('out["timeframes"] = [')]
    assert "except Exception" in block


def test_the_intent_figure_is_not_given_a_direction_it_does_not_have():
    """It is both sides added together. A plus in front of it reads as net
    buying."""
    src = _src()
    fn = src[src.index("function layerNums(name, L) {"):]
    fn = fn[:fn.index("\nfunction paintLayers(")]
    assert "replace('+', '')" in fn
