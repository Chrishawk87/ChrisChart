"""The stack beneath the timeframe being traded.

A one-minute candle builds a fifteen from the ground up, so the fast rows
are not separate opinions -- they are the fifteen being constructed in
front of you.

Three things would break it, and each has a test that fails if the code
takes the easy path.

COLOUR ALONE IS NOT ENOUGH. A green row with a negative move is not a weak
up-read, it is a row contradicting itself, and that case is not
hypothetical: it is exactly what absorption produces. A rule that trusts
either the colour or the number on its own is wrong in precisely the
moments that matter.

THE TRADED ROW DECIDES WHETHER ANYTHING IS DECIDED. Faster rows agreeing
with each other while it says nothing is agreement about a bar that has
not spoken.

A FLIP IS ONLY A TURN IF IT CAME FROM SOMEWHERE. All three red is a down
trend; all three red after having been all three green is the thing worth
being told about.
"""

from __future__ import annotations

import pytest

from liqmap import cascade as cd


def rows(**kw):
    """Timeframes as (colour, bps)."""
    return {tf: v for tf, v in kw.items()}


def stack(m15=("red", -2.0), m5=("red", -1.2), m1=("red", -0.8)):
    return {"15m": m15, "5m": m5, "1m": m1}


# ------------------------------------------- the colour must match the move

def test_a_green_row_with_a_negative_move_is_thrown_out():
    """THE caveat. Not a weak up-read -- a row contradicting itself, which
    is what absorption produces: heavy selling into a bid that will not
    break reads BUYERS on a candle whose price has gone down."""
    r = cd.Rung(timeframe="1m", colour="green", bps=-0.8, present=True)
    assert r.qualified is None
    assert r.conflicted
    assert "contradicts itself" in r.describe()


def test_a_red_row_with_a_positive_move_is_thrown_out_too():
    r = cd.Rung(timeframe="1m", colour="red", bps=0.8, present=True)
    assert r.qualified is None and r.conflicted


def test_colour_and_move_agreeing_is_what_counts():
    assert cd.Rung("1m", "green", 0.8, True).qualified == "up"
    assert cd.Rung("1m", "red", -0.8, True).qualified == "down"


def test_a_row_with_no_colour_is_not_conflicted_it_is_silent():
    r = cd.Rung(timeframe="15m", colour="none", bps=2.0, present=True)
    assert r.qualified is None
    assert not r.conflicted


def test_a_zero_move_qualifies_nothing():
    """Either colour. Price unchanged is not a small move in that colour's
    favour, it is the row having gone nowhere."""
    assert cd.Rung("1m", "green", 0.0, True).qualified is None
    assert cd.Rung("1m", "red", 0.0, True).qualified is None
    assert cd.Rung("1m", "red", 0.0, True).conflicted


# ---------------------------------------------------------- the three cases

def test_all_red_with_negative_moves_is_a_sell():
    """Your words: on a 15 minute, 1, 5 and 15 all red with negative bps,
    that is a sell."""
    c = cd.read("15m", stack())
    assert c.state is cd.State.TREND
    assert c.trend == "down" and c.side == "sell"


def test_all_green_with_positive_moves_is_a_buy():
    c = cd.read("15m", stack(("green", 2.0), ("green", 1.2), ("green", 0.8)))
    assert c.state is cd.State.TREND
    assert c.side == "buy"


def test_the_fast_rows_green_against_a_red_fifteen_is_a_true_pullback():
    """Trading the 15m going down, the 1 and 5 green with positive bps."""
    c = cd.read("15m", stack(("red", -2.0), ("green", 1.2), ("green", 0.8)))
    assert c.state is cd.State.PULLBACK
    assert c.trend == "down" and c.pull == "up"
    assert c.deep and c.depth == 2
    assert "deep pullback" in c.describe()


def test_only_the_minute_flipping_is_the_shallow_one():
    c = cd.read("15m", stack(("red", -2.0), ("red", -1.2), ("green", 0.8)))
    assert c.state is cd.State.PULLBACK
    assert c.depth == 1 and not c.deep
    assert "shallow" in c.describe()


def test_a_pullback_commits_to_nothing():
    """The stack is mid-argument. Nothing is committed until it finishes."""
    c = cd.read("15m", stack(("red", -2.0), ("green", 1.2), ("green", 0.8)))
    assert c.side is None


# ------------------------------------------ the traded row has to speak first

def test_the_lower_rows_agreeing_means_nothing_while_the_traded_one_is_silent():
    """Your case: 1 and 5 green, no colour on the 15. We stand aside."""
    c = cd.read("15m", stack(("none", 0.0), ("green", 1.2), ("green", 0.8)))
    assert c.state is cd.State.NO_COMMIT
    assert c.side is None
    assert "no colour" in c.why


def test_a_traded_row_that_contradicts_itself_also_commits_to_nothing():
    c = cd.read("15m", stack(("green", -2.0), ("green", 1.2), ("green", 0.8)))
    assert c.state is cd.State.NO_COMMIT
    assert "contradicts itself" in c.why


# ----------------------------------------------------------- the stacks

def test_the_five_minute_is_read_against_the_minute_only():
    """Your words: trading the 5m, the only other confluence is the 1m."""
    assert cd.STACKS["5m"] == ("5m", "1m")
    c = cd.read("5m", {"5m": ("red", -1.2), "1m": ("green", 0.8)})
    assert c.state is cd.State.PULLBACK
    assert c.depth == 1 and c.deep


def test_the_minute_stands_alone():
    c = cd.read("1m", {"1m": ("red", -0.8)})
    assert c.state is cd.State.TREND
    assert c.side == "sell"
    assert c.lower == []


def test_nothing_above_the_traded_timeframe_takes_part():
    """Trading the 15m, the hour and the four-hour are not consulted."""
    assert "1h" not in cd.STACKS["15m"]
    assert "4h" not in cd.STACKS["15m"]
    c = cd.read("15m", {**stack(), "1h": ("green", 9.0),
                        "4h": ("green", 20.0)})
    assert [r.timeframe for r in c.rungs] == ["15m", "5m", "1m"]
    assert c.side == "sell"


# ------------------------------------------------------------ the reversal

def test_a_flip_after_the_other_direction_is_a_reversal():
    """Your words: a reversal is when that 15 minute completely flips and
    goes green as well."""
    green = stack(("green", 2.0), ("green", 1.2), ("green", 0.8))
    c = cd.read("15m", green, previous="down")
    assert c.state is cd.State.REVERSAL
    assert c.side == "buy"
    assert "flipped" in c.describe()


def test_the_same_alignment_it_already_had_is_just_a_trend():
    green = stack(("green", 2.0), ("green", 1.2), ("green", 0.8))
    c = cd.read("15m", green, previous="up")
    assert c.state is cd.State.TREND


def test_with_nothing_remembered_a_flip_reads_as_a_trend():
    """True, just not the thing worth being told about."""
    green = stack(("green", 2.0), ("green", 1.2), ("green", 0.8))
    assert cd.read("15m", green).state is cd.State.TREND


def test_only_a_fully_aligned_stack_is_worth_remembering():
    """Carrying a pullback forward would make the next alignment look like
    a reversal of something that never happened."""
    pull = cd.read("15m", stack(("red", -2.0), ("green", 1.2),
                                ("green", 0.8)))
    assert cd.aligned_way(pull) is None
    trend = cd.read("15m", stack())
    assert cd.aligned_way(trend) == "down"
    rev = cd.read("15m", stack(("green", 2.0), ("green", 1.2),
                               ("green", 0.8)), previous="down")
    assert cd.aligned_way(rev) == "up"


# ------------------------------------------------------------- the edges

def test_a_missing_row_leaves_the_stack_unread():
    c = cd.read("15m", {"15m": ("red", -2.0), "5m": ("red", -1.2)})
    assert c.state is cd.State.UNREAD
    assert "1m" in c.why
    assert c.side is None


def test_lower_rows_saying_nothing_is_mixed_rather_than_a_trend():
    """Not a pullback -- nothing flipped -- and not a trend either, since
    the rows beneath are not backing it."""
    c = cd.read("15m", stack(("red", -2.0), ("none", 0.0), ("none", 0.0)))
    assert c.state is cd.State.MIXED
    assert c.side is None


def test_a_timeframe_with_no_stack_is_refused_rather_than_invented():
    c = cd.read("3m", {"3m": ("red", -1.0)})
    assert c.state is cd.State.UNREAD
    assert "not a timeframe" in c.why


def test_the_colour_comes_from_the_winning_side():
    assert cd.colour_of("buyers") == "green"
    assert cd.colour_of("sellers") == "red"
    assert cd.colour_of("contested") == "none"
    assert cd.colour_of(None) == "none"


def test_a_forming_row_has_no_colour_whatever_it_is_winning():
    assert cd.colour_of("buyers", forming=True) == "none"


def test_nothing_here_places_an_order():
    import inspect

    src = inspect.getsource(cd).lower()
    for word in ("def submit", "def execute", "def place", "def order"):
        assert word not in src


# ------------------------------------------------------ route and panel


def _src():
    import inspect

    from liqmap import web as w
    return inspect.getsource(w)


def _route():
    src = _src()
    block = src[src.index("from . import cascade as csc"):]
    return block[:block.index('out["confrontation"] = {')]


def test_the_route_reads_the_rows_already_on_the_ladder():
    """Not anything recomputed. If the panel built its own numbers it
    could contradict the row sitting directly above it."""
    block = _route()
    assert "for r in conf.ordered" in block
    assert "r.move_bps" in block
    assert "csc.colour_of(r.winner" in block


def test_the_route_drops_the_colour_on_a_forming_row():
    assert "forming=r.forming" in _route()


def test_the_route_leaves_an_unmeasured_row_out_rather_than_calling_it_flat():
    """MISSING IS NOT NEUTRAL. A feed that has not warmed up and a market
    in balance produce the same zeros, and a row left in at zero would
    read as a silent row instead of an absent one."""
    assert "if r.measured" in _route()


def test_the_route_uses_the_timeframe_being_traded():
    block = _route()
    assert "csc.read(interval," in block


def test_the_route_remembers_only_an_aligned_stack():
    block = _route()
    assert "rt.remember_aligned(coin, interval, csc.aligned_way(casc))" in block


def test_the_alignment_memory_is_not_bar_scoped():
    """A stack that went green an hour ago is still what a flip now would
    be flipping away from."""
    src = _src()
    fn = src[src.index("def last_aligned(self"):]
    fn = fn[:fn.index("def minute_rows(self")]
    assert "bar_open" not in fn


def test_the_memory_refuses_to_store_nothing():
    src = _src()
    fn = src[src.index("def remember_aligned(self"):]
    fn = fn[:fn.index("def minute_rows(self")]
    assert "if way is not None:" in fn


def _panel():
    src = _src()
    fn = src[src.index("function paintCascade(d) {"):]
    return fn[:fn.index("function paintLayers(d) {")]


def test_the_panel_draws_a_thrown_out_row_differently_from_a_silent_one():
    fn = _panel()
    assert "r.conflicted" in fn and "THROWN OUT" in fn
    assert "no colour" in fn


def test_the_panel_shows_no_call_rather_than_a_greyed_out_side():
    """A faded BUY reads as a weak buy. There is no weak buy here."""
    fn = _panel()
    assert "no call" in fn


def test_the_panel_is_painted_on_every_read():
    assert "  paintCascade(d);\n  paintLayers(d);" in _src()


def test_the_stack_panel_does_not_live_inside_the_ladder_or_the_layers():
    src = _src()
    assert 'id="stackPanel"' in src
    assert src.index('id="stackPanel"') < src.index('id="layerPanel"')
    assert src.index('id="tfLadder"') < src.index('id="stackPanel"')


def test_the_dict_carries_every_rung_for_the_panel():
    d = cd.read("15m", stack()).to_dict()
    assert [r["timeframe"] for r in d["rungs"]] == ["15m", "5m", "1m"]
    for k in ("state", "trend", "pull", "depth", "side", "describe"):
        assert k in d
