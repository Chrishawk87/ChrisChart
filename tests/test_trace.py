"""The trajectory of the read across one bar.

Four things would quietly ruin it.

IT MUST NOT VOTE ON ITSELF. If the candidate direction came from the
engine's own read, the stack's vote would be guaranteed and the count
would partly be measuring its own opinion. The candidate is the bar's own
move, so every check is a real vote.

A COUNT IS NOT A PROBABILITY. Nothing here produces a percentage, and a
test fails if one appears.

MISSING IS NOT AGAINST. A layer with no baseline and a layer that measured
balance are different things, and folding either into "against" makes a
warming-up feed read as bearish.

A FLIP IS NOT A CHANGE IN THE COUNT. "6 for up" and "6 for down" are two
different questions asked once each, not a flat trajectory.
"""

from __future__ import annotations

import pytest

from liqmap import trace as tr


# A layers dict with every layer measured and saying nothing in
# particular, so a test can switch on exactly one thing.
def L(**kw):
    base = {
        "intent": {"who": "neither", "measured": True},
        "resistance": {"impact_ratio": 0.75, "measured": True},
        "response": {"move": 0.0, "unit": "pt", "wide": False,
                     "position": 0.5, "measured": True},
        "liquidity": {"measured": True,
                      "dom": {"bid": {"pull_share": 0.3},
                              "ask": {"pull_share": 0.3}}},
    }
    for k, v in kw.items():
        base[k] = {**base.get(k, {}), **v} if isinstance(v, dict) else v
    return base


def verdicts(checks):
    return {c.name: c.verdict for c in checks}


def test_every_check_has_a_name_and_they_are_the_declared_seven():
    got = tr.evaluate("up", L(), {}, {})
    assert tuple(c.name for c in got) == tr.CHECKS
    assert len(got) == 7


# -------------------------------------------------- the absorption inversion

def test_heavy_buying_that_is_not_moving_price_counts_against_up():
    """THE inversion, and the whole reason the count is worth having.
    Buyers crossing hard into a bid that will not break is a point against
    up, not a weak point for it."""
    v = verdicts(tr.evaluate("up", L(intent={"who": "buyers"},
                                     resistance={"impact_ratio": 0.2})))
    assert v["intent"] == "for"          # buyers ARE attacking
    assert v["efficiency"] == "against"  # and getting nowhere


def test_selling_that_is_being_absorbed_counts_for_up():
    v = verdicts(tr.evaluate("up", L(intent={"who": "sellers"},
                                     resistance={"impact_ratio": 0.2})))
    assert v["intent"] == "against"
    assert v["efficiency"] == "for"


def test_buying_that_is_working_counts_for_up():
    v = verdicts(tr.evaluate("up", L(intent={"who": "buyers"},
                                     resistance={"impact_ratio": 2.4})))
    assert v["efficiency"] == "for"


def test_effectiveness_in_the_middle_says_neither():
    v = verdicts(tr.evaluate("up", L(intent={"who": "buyers"},
                                     resistance={"impact_ratio": 0.75})))
    assert v["efficiency"] == "quiet"


def test_effectiveness_with_nobody_attacking_is_quiet_not_a_vote():
    """It is the effectiveness OF somebody. With nobody crossing it has no
    direction to take."""
    v = verdicts(tr.evaluate("up", L(resistance={"impact_ratio": 2.4})))
    assert v["efficiency"] == "quiet"


def test_the_thresholds_are_the_ones_the_verdict_already_classifies_on():
    """Two modules disagreeing about what absorption is would put a
    trajectory on screen contradicting the word above it."""
    from liqmap import result as rsl

    assert tr.ABSORBED == rsl.LOW_IMPACT


# ------------------------------------------------- missing is not against

def test_an_unmeasured_layer_is_its_own_answer():
    v = verdicts(tr.evaluate("up", L(resistance={"measured": False})))
    assert v["efficiency"] == "unmeasured"


def test_a_warming_up_feed_produces_no_votes_against():
    """Every layer cold. If any of these came back 'against', a feed that
    had just connected would read as a case against the move."""
    cold = {"intent": {"measured": False},
            "resistance": {"measured": False},
            "response": {"measured": False},
            "liquidity": {"measured": False, "dom": {}}}
    got = tr.evaluate("up", cold, {}, {})
    assert all(c.verdict == "unmeasured" for c in got), verdicts(got)
    s = tr.Sample(ts=0.0, way="up", checks=got)
    assert s.n_against == 0 and s.n_for == 0
    assert s.counted == 0


def test_an_unmeasured_check_is_not_counted_in_the_denominator():
    """'2 of 7' when five could not be read overstates the disagreement."""
    got = tr.evaluate("up", L(intent={"measured": False},
                              resistance={"measured": False}),
                      {"trend": "up"}, {"known": True, "state": "paid_for"})
    s = tr.Sample(ts=0.0, way="up", checks=got)
    assert s.n_unmeasured == 2        # intent and efficiency
    assert s.counted == 5
    assert "of 5" in s.describe()


def test_the_stack_not_qualifying_is_unmeasured_rather_than_against():
    assert verdicts(tr.evaluate("up", L(), {"trend": None}))["stack"] \
        == "unmeasured"


def test_the_stack_reading_the_other_way_is_against():
    assert verdicts(tr.evaluate("up", L(), {"trend": "down"}))["stack"] \
        == "against"
    assert verdicts(tr.evaluate("up", L(), {"trend": "up"}))["stack"] == "for"


# ------------------------------------------------------- the other checks

def test_price_has_to_have_gone_somewhere_before_it_votes():
    """A tenth of a tick in the right direction is not a response."""
    v = verdicts(tr.evaluate("up", L(response={"move": 0.1, "wide": False})))
    assert v["response"] == "quiet"
    v = verdicts(tr.evaluate("up", L(response={"move": 4.0, "wide": True})))
    assert v["response"] == "for"
    v = verdicts(tr.evaluate("up", L(response={"move": -4.0, "wide": True})))
    assert v["response"] == "against"


def test_only_the_difference_between_the_two_sides_of_the_book_counts():
    """Both sides churn in a fast market. One side's pull share on its own
    mostly measures how busy it is."""
    v = verdicts(tr.evaluate("up", L(liquidity={
        "dom": {"bid": {"pull_share": 0.8}, "ask": {"pull_share": 0.8}}})))
    assert v["book"] == "quiet"
    v = verdicts(tr.evaluate("up", L(liquidity={
        "dom": {"bid": {"pull_share": 0.2}, "ask": {"pull_share": 0.7}}})))
    assert v["book"] == "for"          # offers getting out of the way
    v = verdicts(tr.evaluate("up", L(liquidity={
        "dom": {"bid": {"pull_share": 0.7}, "ask": {"pull_share": 0.2}}})))
    assert v["book"] == "against"      # the floor coming out


def test_volume_votes_on_whether_the_bar_can_move_not_on_which_way():
    """Tying fuel to the aggressor would be the efficiency check again
    under another name, and the same evidence counted twice looks like two
    readings agreeing."""
    for way in ("up", "down"):
        assert verdicts(tr.evaluate(way, L(), {},
                                    {"known": True, "state": "no_fuel"})
                        )["fuel"] == "against"
        assert verdicts(tr.evaluate(way, L(), {},
                                    {"known": True, "state": "paid_for"})
                        )["fuel"] == "for"


def test_an_unlearned_volume_baseline_does_not_vote():
    assert verdicts(tr.evaluate("up", L(), {}, {"known": False,
                                                "state": "paid_for"})
                    )["fuel"] == "unmeasured"


def test_a_bar_that_gave_its_move_back_is_not_sitting_on_its_high():
    v = verdicts(tr.evaluate("up", L(response={"position": 0.5})))
    assert v["location"] == "quiet"
    assert verdicts(tr.evaluate("up", L(response={"position": 0.9})
                                ))["location"] == "for"
    assert verdicts(tr.evaluate("up", L(response={"position": 0.1})
                                ))["location"] == "against"


# ------------------------------------------- the candidate is the bar's own

def test_the_direction_voted_on_comes_from_price_not_from_the_engine():
    """Otherwise the stack's vote is guaranteed and the count is partly
    measuring its own opinion."""
    t = tr.Tracer()
    out = t.observe("ES", "15m", 0.0, 900.0, ts=1.0, move_bps=-2.0,
                    layers=L(), cascade={"trend": "up"})
    assert out.latest.way == "down"
    assert verdicts(out.latest.checks)["stack"] == "against"


def test_a_bar_on_its_open_has_no_candidate_and_says_so():
    t = tr.Tracer()
    out = t.observe("ES", "15m", 0.0, 900.0, ts=1.0, move_bps=0.0,
                    layers=L(), cascade={"trend": "up"})
    assert out.latest.way is None
    assert out.latest.checks == ()
    assert "on its open" in out.latest.describe()


# ------------------------------------------------------------- the legs

def test_crossing_back_through_the_open_starts_a_new_leg():
    t = tr.Trace(trading="15m", interval_s=900.0)
    t.record(tr.Sample(ts=0.0, way="up"))
    t.record(tr.Sample(ts=100.0, way="up"))
    t.record(tr.Sample(ts=200.0, way="down"))
    assert len(t.legs) == 2
    assert [l.way for l in t.legs] == ["up", "down"]
    assert t.flips == 1


def test_a_flip_is_never_thrown_away_by_the_throttle():
    """It is the one event the trajectory exists to show."""
    t = tr.Trace(trading="15m", interval_s=900.0)
    assert t.record(tr.Sample(ts=0.0, way="up"))
    assert not t.record(tr.Sample(ts=0.5, way="up"))    # too soon
    assert t.record(tr.Sample(ts=1.0, way="down"))      # but it turned
    assert len(t.legs) == 2


def test_the_count_is_never_compared_across_a_flip():
    """'6 for up' then '6 for down' is not a flat trajectory, it is two
    different questions asked once each."""
    t = tr.Trace(trading="15m", interval_s=900.0)
    six_up = tuple(tr.Check(n, "for") for n in tr.CHECKS[:6])
    t.record(tr.Sample(ts=0.0, way="up", checks=six_up))
    t.record(tr.Sample(ts=50.0, way="up", checks=six_up))
    assert t.change_since(60.0) == 0
    t.record(tr.Sample(ts=60.0, way="down", checks=six_up))
    assert t.change_since(60.0) is None


def test_a_bar_that_opened_flat_and_picked_a_side_has_not_flipped():
    t = tr.Trace(trading="15m", interval_s=900.0)
    t.record(tr.Sample(ts=0.0, way=None))
    t.record(tr.Sample(ts=10.0, way="up"))
    assert t.flips == 0


# ------------------------------------------------- strengthening or weakening

def _leg(counts, interval_s=900.0, step=30.0):
    """A trace whose `for` count walks through `counts`.

    `step` must clear `gap_for(interval_s)` or the throttle eats the
    samples and the test asserts nothing -- which it did, the first time.
    """
    assert step > tr.gap_for(interval_s), "the throttle would eat these"
    t = tr.Trace(trading="15m", interval_s=interval_s)
    for i, n in enumerate(counts):
        t.record(tr.Sample(ts=i * step, way="up",
                           checks=tuple(tr.Check(c, "for" if k < n else
                                                 "against")
                                        for k, c in enumerate(tr.CHECKS))))
    return t


def test_evidence_building_reads_as_strengthening():
    t = _leg([3, 4, 5, 6])
    assert t.change_since(60.0) > 0
    assert "strengthening" in t.describe()


def test_evidence_falling_apart_reads_as_weakening():
    t = _leg([7, 6, 5, 4])
    assert t.change_since(60.0) < 0
    assert "weakening" in t.describe()


def test_the_same_instant_reached_two_ways_is_not_the_same_bar():
    """The whole point. Both end on 6 of 7."""
    up, down = _leg([3, 4, 5, 6]), _leg([7, 6, 5, 6])
    assert up.latest.n_for == down.latest.n_for == 6
    assert up.change_since(60.0) != down.change_since(60.0)


def test_a_single_sample_has_no_trajectory_yet():
    t = _leg([5])
    assert t.change_since(60.0) is None
    assert "just started" in t.describe()


def test_a_window_older_than_the_leg_falls_back_to_its_start():
    """Otherwise a leg younger than the window reports nothing, which is
    the part of the bar most worth watching."""
    t = _leg([3, 6], step=10.0)
    assert t.change_since(3600.0) == 3


def test_it_names_which_check_flipped():
    t = tr.Trace(trading="15m", interval_s=900.0)
    t.record(tr.Sample(ts=0.0, way="up", checks=(
        tr.Check("intent", "for"), tr.Check("efficiency", "for"))))
    t.record(tr.Sample(ts=60.0, way="up", checks=(
        tr.Check("intent", "for"), tr.Check("efficiency", "against"))))
    assert t.changes() == ["efficiency for → against"]


# -------------------------------------------------------------- the keeping

def test_samples_are_thinned_by_the_bars_own_length():
    """A four-hour bar must not keep thousands; a one-minute bar still
    needs a useful handful."""
    assert tr.gap_for(60.0) == tr.MIN_GAP_S
    assert tr.gap_for(900.0) > tr.MIN_GAP_S
    assert tr.gap_for(14400.0) > tr.gap_for(900.0)
    assert tr.gap_for(0.0) == tr.MIN_GAP_S


def test_a_sample_too_soon_after_the_last_is_refused():
    t = tr.Trace(trading="1m", interval_s=60.0)
    assert t.record(tr.Sample(ts=0.0, way="up"))
    assert not t.record(tr.Sample(ts=1.0, way="up"))
    assert t.record(tr.Sample(ts=10.0, way="up"))
    assert len(t.leg.samples) == 2


def test_the_cap_drops_from_the_middle_never_from_either_end():
    """The opening of the bar and the last few seconds are the two parts
    anybody actually looks at."""
    t = tr.Trace(trading="15m", interval_s=900.0)
    for i in range(tr.MAX_SAMPLES + 40):
        t.record(tr.Sample(ts=i * 999.0, way="up"))
    s = t.leg.samples
    assert len(s) == tr.MAX_SAMPLES
    assert s[0].ts == 0.0
    assert s[-1].ts == (tr.MAX_SAMPLES + 39) * 999.0


def test_the_bar_rolling_starts_a_fresh_trace():
    """Carrying it over would draw the previous candle's build-up as
    though it were this one's."""
    t = tr.Tracer()
    a = t.trace("ES", "15m", 900.0, 900.0)
    a.record(tr.Sample(ts=1.0, way="up"))
    b = t.trace("ES", "15m", 1800.0, 900.0)
    assert b is not a
    assert b.latest is None
    assert t.trace("ES", "15m", 1800.0, 900.0) is b


def test_two_timeframes_keep_separate_traces():
    t = tr.Tracer()
    assert t.trace("ES", "15m", 0.0, 900.0) is not t.trace("ES", "5m", 0.0,
                                                           300.0)


# --------------------------------------------------------------- the honesty

def test_nothing_here_produces_a_percentage():
    """A percentage composited out of weighted factors is a confidence
    score wearing a percent sign. This counts; it does not score."""
    import ast
    import inspect

    # The CODE, not the prose. The module docstring says the word a dozen
    # times explaining why there isn't one, and a grep over the raw source
    # fails on its own explanation -- which it did, the first time.
    tree = ast.parse(inspect.getsource(tr))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body.pop(0)
    code = ast.unparse(tree).lower()

    for word in ("probability", "confidence", "likelihood", "weight",
                 "score"):
        assert word not in code, word
    # And the count is never divided into a rate.
    for frac in ("n_for /", "/ len(self.checks)", "/ self.counted"):
        assert frac not in code, frac


def test_nothing_here_places_an_order():
    import inspect

    src = inspect.getsource(tr).lower()
    for word in ("def submit", "def execute", "def place", "def order"):
        assert word not in src


def test_the_dict_carries_the_leg_for_the_panel():
    t = _leg([3, 4, 5, 6])
    d = t.to_dict()
    for k in ("now", "change_60s", "changes", "leg", "flips", "describe"):
        assert k in d
    assert d["now"]["for"] == 6
    assert [s["for"] for s in d["leg"]["samples"]] == [3, 4, 5, 6]


def test_the_dict_can_be_trimmed_for_the_wire():
    t = _leg([1, 2, 3, 4, 5, 6])
    assert len(t.to_dict(keep=2)["leg"]["samples"]) == 2
    assert t.to_dict(keep=2)["leg"]["n"] == 6


# ------------------------------------------------------ route and panel


def _src():
    import inspect

    from liqmap import web as w
    return inspect.getsource(w)


def _route():
    src = _src()
    block = src[src.index("chosen_r = next((r for r in conf.ordered"):]
    return block[:block.index("except Exception as exc:")]


def test_the_route_votes_on_the_bars_own_move_not_the_engines_read():
    """If the candidate came from the cascade, the stack would be voting
    for its own opinion."""
    block = _route()
    assert "move_bps=chosen_r.move_bps" in block
    assert "move_bps=casc" not in block


def test_the_route_feeds_it_the_timeframe_being_traded():
    block = _route()
    assert "lay.get(interval)" in block
    assert "rt.tracer.observe(" in block


def test_the_route_anchors_the_trace_to_this_bars_open():
    """A trajectory belongs to its bar. Without the open it would never
    reset and would draw the last candle's build-up as this one's."""
    block = _route()
    assert "now_s % step_s" in block


def test_the_route_trims_the_payload():
    assert "keep=_TRACE_KEEP" in _route()


def _panel():
    src = _src()
    fn = src[src.index("function paintTrace(d) {"):]
    return fn[:fn.index("function paintLayers(d) {")]


def test_the_panel_shows_only_the_direction_and_whether_it_is_building():
    """The seven readings are computed and sent; they are not drawn.

    What is on screen is the direction and whether the case for it is
    strengthening or weakening -- one line, under the fastest row.
    """
    fn = _panel()
    assert "'UP'" in fn and "'DOWN'" in fn
    assert "strengthening" in fn and "weakening" in fn
    # None of the seven appear on screen.
    for name in tr.CHECKS:
        assert name not in fn, name
    assert "N.checks" not in fn
    assert "T.leg" not in fn and "samples" not in fn


def test_the_panel_prints_no_number_at_all():
    """Not a count and certainly not a percentage. The count still rides
    in the payload for when there are scored bars behind it."""
    import re

    code = re.sub(r"'[^']*'", "''", _panel())
    for word in ("probability", "likely", "confidence", "percent"):
        assert word not in code.lower(), word
    assert "N['for']" not in code and "n_for" not in code


def test_the_panel_draws_no_trajectory_differently_from_a_flat_one():
    """A leg a few seconds old, or one that has just crossed the open,
    has no trajectory -- drawing that as 'holding' claims a steadiness
    nobody measured."""
    fn = _panel()
    assert "just started" in fn
    assert "holding" in fn
    assert "c === null" in fn


def test_a_cold_feed_does_not_draw_a_direction():
    """A confident arrow over a feed that has told us nothing."""
    fn = _panel()
    assert "!N.counted" in fn
    assert "warming up" in fn


def test_the_panel_is_painted_on_every_read():
    assert "  paintCascade(d);\n  paintTrace(d);\n  paintLayers(d);" in _src()


def test_the_line_sits_under_the_stack_rows_not_in_the_layers_panel():
    src = _src()
    panel = src[src.index('id="stackPanel"'):src.index('id="layerPanel"')]
    assert panel.index('id="cascRows"') < panel.index('id="traceLine"')
    assert 'id="traceBox"' not in src        # the old side column is gone


def test_the_seven_are_still_read_and_still_sent_even_though_hidden():
    """Hidden, not switched off. If the route stopped taking them the
    line above would have nothing behind it."""
    block = _route()
    assert "rt.tracer.observe(" in block
    assert 'out["trace"] = tc.to_dict(' in block
    d = _leg([3, 4, 5, 6]).to_dict()
    assert len(d["now"]["checks"]) == len(tr.CHECKS)
