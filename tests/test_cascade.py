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


# -------------------------------------- the row reads price, like the window

# Chris: "i want the stack timeframes to read like the 5m window does
# thats true direction". The window's `mid_dir` is close against open
# with no threshold. These rows now answer that question on their own
# timeframe, and the colour -- who won on aggression -- is a SECOND
# measurement that is shown and decides nothing.


def test_the_direction_is_price_against_this_bars_open():
    assert cd.Rung("1m", "green", 0.8, True).qualified == "up"
    assert cd.Rung("1m", "red", -0.8, True).qualified == "down"


def test_a_green_row_with_a_negative_move_counts_by_its_price():
    """IT USED TO BE THROWN OUT. Requiring the colour and the number
    to agree meant the stack could not answer the question the window
    answers, and a silent row blocks every entry built on it."""
    r = cd.Rung(timeframe="1m", colour="green", bps=-0.8, present=True)
    assert r.qualified == "down"
    assert r.absorbed, "the disagreement stopped being measured"
    assert "absorbed" in r.describe()


def test_a_red_row_with_a_positive_move_counts_by_its_price_too():
    r = cd.Rung(timeframe="1m", colour="red", bps=0.8, present=True)
    assert r.qualified == "up" and r.absorbed


def test_a_row_with_no_colour_still_has_a_direction():
    """A quiet bar that has nonetheless moved is not silent. Saying
    "no colour" about it hid the number that now counts."""
    r = cd.Rung(timeframe="15m", colour="none", bps=2.0, present=True)
    assert r.qualified == "up"
    assert not r.absorbed, "no colour cannot disagree with anything"
    assert "up" in r.describe()


def test_a_zero_move_qualifies_nothing():
    """Either colour. Price unchanged is not a small move in that
    colour's favour, it is the row having gone nowhere -- and the
    window treats a flat candle the same way."""
    assert cd.Rung("1m", "green", 0.0, True).qualified is None
    assert cd.Rung("1m", "red", 0.0, True).qualified is None
    assert not cd.Rung("1m", "red", 0.0, True).absorbed, \
        "flat is not absorbed, it is flat"
    assert "flat" in cd.Rung("1m", "red", 0.0, True).describe()


def test_an_unseeded_row_has_no_direction_at_all():
    """THE ONE GATE KEPT. Without the bar's open there is no
    close-against-open to compute, and a number measured from the
    wrong anchor is wrong rather than merely noisy."""
    r = cd.Rung("5m", "green", 5.0, True, seeded=False)
    assert r.qualified is None
    assert "open was never seen" in r.describe()


def test_seeded_defaults_true_so_existing_callers_are_unchanged():
    assert cd.Rung("5m", "green", 1.0, True).seeded is True
    c = cd.read("15m", {"15m": ("green", 1.0), "5m": ("green", 1.0),
                        "1m": ("green", 1.0)})
    assert all(g.seeded for g in c.rungs)


def test_read_accepts_the_seeded_flag_when_given(self=None):
    c = cd.read("15m", {"15m": ("green", 1.0, False),
                        "5m": ("green", 1.0, True),
                        "1m": ("green", 1.0, True)})
    assert c.rungs[0].seeded is False
    assert c.rungs[0].qualified is None


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


def test_a_traded_row_whose_colour_fights_its_price_reads_by_price():
    """IT USED TO COMMIT TO NOTHING. The traded row being green on
    aggression while price fell made the whole stack unreadable, and
    an unreadable traded row blocks every entry beneath it.

    It now reads DOWN by its price, which makes the two green rows
    beneath it a pullback against it -- a reading, where there was
    none. The absorption is still measured on the row itself."""
    c = cd.read("15m", stack(("green", -2.0), ("green", 1.2),
                             ("green", 0.8)))
    assert c.state is cd.State.PULLBACK
    assert c.trend == "down" and c.pull == "up"
    assert c.rungs[0].absorbed, "the disagreement stopped being measured"


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


def test_the_route_takes_the_direction_from_the_candle_and_the_colour_from_the_ladder():
    """TWO SOURCES, ON PURPOSE, AND THEY ANSWER DIFFERENT THINGS.

    The DIRECTION is the real forming candle -- Chris: "i want what
    a real 5 minute 15 minute and 1 minute candle is doing not a
    calculation of it". The COLOUR is still the ladder's own
    aggression row, not recomputed, so the absorption flag cannot
    contradict the row sitting directly above it.
    """
    block = _route()
    assert "csc.rows_from_candles(" in block, "the direction is not the candle"
    assert "_candle_now(rt, tf)" in block
    assert "for r in conf.ordered" in block, "the colour was recomputed"
    assert 'getattr(r, "winner", None)' in block
    # Scoped to the ROW BUILD. `r.move_bps` legitimately survives
    # elsewhere in this slice -- the `timeframes` payload and a
    # trace both want the aggregate -- so a blanket "not in" fails
    # on code it has no opinion about.
    assert "rows_ = {r.timeframe" not in block, "the aggregate is back"


def test_the_route_still_drops_the_colour_on_a_forming_row():
    assert 'getattr(r, "forming", False)' in _route()


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


def test_the_panel_draws_the_direction_before_the_absorption():
    """`conflicted` was checked AHEAD of `qualified` and rendered
    THROWN OUT. That was right while the colour and the number had
    to agree for a row to count; they no longer do, so the old order
    would have printed THROWN OUT on rows that were being traded."""
    fn = _panel()
    assert "r.qualified === 'up'" in fn
    assert "ABSORBED" in fn, "the disagreement is no longer shown"
    # THE CODE, NOT THE PROSE. A first attempt matched the comment
    # that DESCRIBES the old badge -- the second time in two days I
    # have written a test that fails against its own documentation.
    assert "'<span class=\"flag late\">THROWN OUT</span>'" not in fn, \
        "a counted row is still called thrown out"
    i_dir = fn.index("r.qualified === 'up'")
    assert fn.index("r.absorbed") > i_dir, \
        "absorption is still being read before the direction"


def test_the_panel_distinguishes_flat_from_an_unseen_open():
    """Two different silences. One is the bar having gone nowhere,
    the other is not knowing where it started."""
    fn = _panel()
    assert "flat" in fn and "open not seen" in fn


def test_the_panel_shows_no_call_rather_than_a_greyed_out_side():
    """A faded BUY reads as a weak buy. There is no weak buy here."""
    fn = _panel()
    assert "no call" in fn


def test_the_panel_is_painted_on_every_read():
    """Asserts the call, not what sits next to it -- a neighbouring paint
    being added is not this panel breaking."""
    src = _src()
    block = src[src.index("function paintRead(d)"):]
    assert "paintCascade(d);" in block


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


# ------------------------------- the row reads the bar, not the window


def test_bar_move_bps_is_the_current_bar_not_the_aggregate():
    """Chris: "the stack 5 minutes is still reading the old data".

    `from_candles` aggregates three bars and sets `open_px` from the
    FIRST of them. That is right for what it was built for -- "the
    4-hour is down" means the recent ones, not two minutes into the
    current candle -- and wrong for a 5m row that has to answer the
    same question as a 5m window: it measures price against an open
    fifteen minutes old.
    """
    from collections import namedtuple
    from liqmap.pressure import from_candles

    C = namedtuple("C", "open high low close volume trades ts")
    bars = [C(100.0, 100.2, 99.0, 99.2, 1000, 50, 0.0),
            C(99.2, 99.4, 98.5, 98.7, 1000, 50, 300.0),
            C(98.7, 99.9, 98.6, 99.8, 1000, 50, 600.0)]   # this one is UP
    r = from_candles("5m", 300.0, bars, lookback=3)
    assert r.open_px == 100.0, "the window still starts at the oldest bar"
    assert r.bar_open_px == 98.7, "the current bar's open is not carried"
    assert r.move_bps < 0, "the three-bar window is down"
    assert r.bar_move_bps > 0, "the current bar is up"


def test_a_single_bar_reading_is_unchanged():
    """`bar_open_px` must be a no-op wherever one bar is read, or
    this would quietly move every live row as well."""
    from collections import namedtuple
    from liqmap.pressure import from_candles

    C = namedtuple("C", "open high low close volume trades ts")
    bars = [C(100.0, 101.0, 99.0, 100.5, 1000, 50, 0.0)]
    r = from_candles("5m", 300.0, bars, lookback=1)
    assert r.bar_open_px == r.open_px
    assert r.bar_move_bps == r.move_bps


def test_an_unset_bar_open_falls_back_to_the_window_open():
    """Zero means not supplied. A caller that does not set it has to
    behave exactly as it did before."""
    from liqmap.pressure import Pressure
    p = Pressure(timeframe="5m", interval_s=300.0,
                 open_px=100.0, last_px=101.0,
                 high_px=101.0, low_px=99.0,
                 buy_notional=1.0, sell_notional=1.0, measured=True)
    assert p.bar_open_px == 0.0
    assert p.bar_move_bps == p.move_bps


# ------------------------------------------- a row says where it came from


def test_a_rung_carries_its_source_age_and_span():
    r = cd.Rung("5m", "green", 1.0, True, source="polled",
                age_s=47.0, bars_used=3.0)
    d = r.to_dict()
    assert d["source"] == "polled" and d["age_s"] == 47.0
    assert d["bars_used"] == 3.0


def test_the_meta_is_optional_so_old_callers_are_unchanged():
    c = cd.read("15m", {"15m": ("green", 1.0), "5m": ("green", 1.0),
                        "1m": ("green", 1.0)})
    assert all(g.source == "" and g.bars_used == 1.0 for g in c.rungs)


def test_read_takes_the_meta_when_given():
    c = cd.read("5m", {"5m": ("green", 1.0, True,
                              {"source": "polled", "age_s": 12.0,
                               "bars_used": 3}),
                       "1m": ("green", 1.0)})
    assert c.rungs[0].source == "polled"
    assert c.rungs[0].bars_used == 3.0
    assert c.rungs[1].source == ""


def test_the_panel_shows_the_provenance():
    fn = _panel()
    assert "r.source" in fn, "the row does not say where it came from"
    assert "r.bars_used" in fn, "a three-bar row looks like a one-bar row"
    assert "r.age_s" in fn


# --------------------------------- the stack rows ARE the real candles


class TestTheStackReadsTheRealCandle:
    """Chris: "i want what a real 5 minute 15 minute and 1 minute
    candle is doing not a calculation of it".

    The bps came off the pressure reading -- a measurement BUILT
    from bars rather than the bars themselves. Three things it did
    made the row disagree with the candle on the chart: a polled row
    aggregates three bars and takes its open from the OLDEST; it
    falls back to the last CLOSED bar when the forming one is not
    usable; and it is as old as its last write.
    """

    @staticmethod
    def rows(candles: dict, trading: str = "5m", aggression=None):
        return cd.rows_from_candles(
            trading, lambda tf: candles.get(tf), aggression)

    def _read(self, candles: dict, trading: str = "5m"):
        c = cd.read(trading, self.rows(candles, trading))
        return {g.timeframe: g for g in c.rungs}

    def test_each_row_is_its_candles_close_against_its_open(self):
        r = self._read({"5m": {"open": 100.0, "close": 100.5},
                        "1m": {"open": 100.4, "close": 100.1}})
        assert r["5m"].qualified == "up"
        assert r["1m"].qualified == "down"

    def test_a_flat_candle_is_flat(self):
        r = self._read({"5m": {"open": 100.0, "close": 100.0},
                        "1m": {"open": 100.0, "close": 100.5}})
        assert r["5m"].qualified is None
        assert r["1m"].qualified == "up"

    def test_the_row_says_it_came_from_the_candle(self):
        r = self._read({"5m": {"open": 100.0, "close": 100.5, "seconds": 90},
                        "1m": {"open": 100.0, "close": 100.5}})
        assert r["5m"].source == "candle"
        assert r["5m"].bars_used == 1.0, "it is aggregating again"
        assert r["5m"].age_s == 90

    def test_a_missing_candle_leaves_the_row_absent(self):
        """Absent, not invented. A row with no candle has nothing to
        say and must not say flat -- the two call for opposite
        reactions."""
        r = self._read({"5m": {"open": 100.0, "close": 100.5}})
        assert r["5m"].present is True
        assert r["1m"].present is False

    def test_a_zero_or_junk_candle_is_left_out(self):
        for bad in ({"open": 0.0, "close": 100.0},
                    {"open": 100.0, "close": 0.0},
                    {"open": "x", "close": 100.0}, {}):
            assert self.rows({"5m": bad, "1m": {"open": 1.0, "close": 2.0}},
                             "5m").get("5m") is None

    def test_a_raising_candle_source_does_not_take_the_stack_down(self):
        def boom(tf):
            if tf == "5m":
                raise RuntimeError("feed died")
            return {"open": 100.0, "close": 100.5}
        rows = cd.rows_from_candles("5m", boom)
        assert "5m" not in rows and "1m" in rows

    def test_the_fifteen_minute_stack_reads_its_own_three(self):
        r = self._read({"15m": {"open": 100.0, "close": 99.0},
                        "5m": {"open": 99.5, "close": 99.0},
                        "1m": {"open": 99.1, "close": 99.0}}, trading="15m")
        assert [r[t].qualified for t in ("15m", "5m", "1m")] \
            == ["down", "down", "down"]

    def test_it_matches_what_the_window_would_say(self):
        """THE WHOLE POINT. The window's `mid_dir` is close against
        open with no threshold; so is this now."""
        for o, cl, want in ((100.0, 100.25, "up"), (100.0, 99.75, "down"),
                            (100.0, 100.0, None)):
            r = self._read({"5m": {"open": o, "close": cl},
                            "1m": {"open": 100.0, "close": 100.5}})
            mid = ("up" if cl > o else "down" if cl < o else None)
            assert r["5m"].qualified == want == mid

    def test_the_aggression_still_rides_along_as_the_colour(self):
        r = self._read({"5m": {"open": 100.0, "close": 99.0},
                        "1m": {"open": 100.0, "close": 100.5}})
        assert r["5m"].qualified == "down"
        r2 = cd.read("5m", self.rows(
            {"5m": {"open": 100.0, "close": 99.0},
             "1m": {"open": 100.0, "close": 100.5}},
            "5m", {"5m": ("buyers", False)}))
        row = {g.timeframe: g for g in r2.rungs}["5m"]
        assert row.colour == "green"
        assert row.qualified == "down", "the colour decided again"
        assert row.absorbed, "the absorption stopped being flagged"

    def test_the_route_delegates_rather_than_inlining_it(self):
        import pathlib
        src = (pathlib.Path(__file__).resolve().parents[1]
               / "liqmap" / "web.py").read_text()
        assert "csc.rows_from_candles(" in src
        assert "_candle_now(rt, tf)" in src
