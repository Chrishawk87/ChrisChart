"""Agreement episodes across 1m/5m/15m, and how far price goes inside one.

Five ways this would report something other than what it claims.

A DOWN EPISODE IS NOT AN UP EPISODE WITH A MINUS SIGN IN THE WRONG PLACE.
Favourable travel on a short is price FALLING. Getting that backwards turns
every losing run into a winning one and the table still looks plausible --
it has already happened once in this project, in `stack_study`.

FLAT MUST NOT END AN EPISODE. The first print of every new bar sits exactly
on that bar's open, so the timeframe reads flat. A convention that ends the
run there cuts every episode at the next minute boundary, and the tick
figure becomes a measurement of where the boundaries fell.

A FLIP MUST END IT. Otherwise the episode runs on through disagreement and
the travel belongs to a period nobody was agreeing in.

TRAVEL IS MEASURED FROM THE AGREEMENT, not from the bar open. The bar may
have opened a long way back; that distance was not available to anybody
waiting for the three to line up.

AND THE MIRROR MUST BE THE MIRROR. It is the null, so it has to be the same
window and the same elapsed time, differing only in direction.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load():
    path = ROOT / "scripts" / "agree_travel.py"
    spec = importlib.util.spec_from_file_location("agree_travel", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["agree_travel"] = mod
    spec.loader.exec_module(mod)
    return mod


at = _load()
T = at.TICK


def trade(ts, px):
    from liqmap.flow import Trade
    return Trade(px=px, sz=1.0, aggressor="buy", ts=ts)


@pytest.fixture()
def passthrough(monkeypatch):
    from liqmap import dbn
    monkeypatch.setattr(dbn, "stream",
                        lambda recs, symbol="ES", flip=False, **kw:
                        iter(("trade", r) for r in recs))


def tape(pairs):
    return [trade(ts, px) for ts, px in pairs]


# ------------------------------------------------------------- the frames

def test_a_bar_opens_at_its_first_print_and_holds_it():
    f = at.Frame(60.0)
    assert f.feed(0.0, 100.0) is True
    assert f.open_px == 100.0
    assert f.feed(30.0, 105.0) is False
    assert f.open_px == 100.0, "the open moved"


def test_a_new_slot_opens_a_new_bar():
    f = at.Frame(60.0)
    f.feed(10.0, 100.0)
    assert f.feed(60.0, 107.0) is True
    assert f.open_px == 107.0


def test_the_slot_is_the_floor_of_the_span_not_the_first_print():
    """Bars sit on the clock. Starting them at whatever time the file
    happens to begin puts every timeframe on its own private grid."""
    f = at.Frame(900.0)
    f.feed(1000.0, 100.0)
    assert f.bar_ts == 900.0
    f2 = at.Frame(60.0)
    f2.feed(1000.0, 100.0)
    assert f2.bar_ts == 960.0


def test_direction_is_price_now_against_this_bars_open():
    f = at.Frame(60.0)
    f.feed(0.0, 100.0)
    assert f.way(101.0) == "up"
    assert f.way(99.0) == "down"
    assert f.way(100.0) == "flat"


def test_a_frame_with_no_bar_yet_says_nothing():
    assert at.Frame(60.0).way(100.0) == ""


# ----------------------------------------------------------- the agreement

def test_agreement_needs_all_three_and_tolerates_no_flat():
    assert at.agreed_way(("up", "up", "up")) == "up"
    assert at.agreed_way(("down", "down", "down")) == "down"
    assert at.agreed_way(("up", "up", "down")) == ""
    assert at.agreed_way(("up", "up", "flat")) == ""
    assert at.agreed_way(("up", "up", "")) == ""


# ------------------------------------------------------- travel arithmetic

def ep(way="up", start=100.0):
    return at.Episode(start_ts=0.0, start_px=start, way=way, bar15_ts=0.0,
                      offset_s=0.0)


def test_an_up_episode_counts_rising_price_as_favourable():
    e = ep("up")
    for px in (100.0, 101.0, 99.5):
        e.observe(px)
    e.end_ts, e.end_px = 10.0, 99.5
    assert e.mfe == pytest.approx(4.0)      # +1.00 is four ticks
    assert e.mae == pytest.approx(2.0)      # -0.50 is two ticks
    assert e.net == pytest.approx(-2.0)


def test_a_down_episode_counts_falling_price_as_favourable():
    """THE one. A short's favourable travel is price going DOWN."""
    e = ep("down")
    for px in (100.0, 99.0, 100.5):
        e.observe(px)
    e.end_ts, e.end_px = 10.0, 100.5
    assert e.mfe == pytest.approx(4.0)      # -1.00 is four ticks FOR it
    assert e.mae == pytest.approx(2.0)      # +0.50 is two ticks against
    assert e.net == pytest.approx(-2.0)


def test_favourable_and_adverse_are_never_negative():
    for way in ("up", "down"):
        e = ep(way)
        for px in (100.0, 100.25, 99.75):
            e.observe(px)
        e.end_ts, e.end_px = 5.0, 99.75
        assert e.mfe >= 0 and e.mae >= 0


def test_an_asymmetric_window_separates_favourable_from_adverse():
    """The fixture must be ASYMMETRIC.

    With price moving the same distance each way, favourable and adverse
    travel are numerically equal and a function returning the wrong one
    passes anyway.
    """
    up = ep("up")
    down = ep("down")
    for px in (100.0, 103.0, 99.0):        # +12 ticks up, -4 down
        up.observe(px)
        down.observe(px)

    assert up.mfe == pytest.approx(12.0)
    assert up.mae == pytest.approx(4.0)
    assert down.mfe == pytest.approx(4.0)
    assert down.mae == pytest.approx(12.0)

    # each episode is the other's reflection
    assert up.mfe == pytest.approx(down.mae)
    assert down.mfe == pytest.approx(up.mae)


# --------------------------------------------------- the null is a window

def build_tape(pairs):
    t = at.Tape()
    for ts, px in pairs:
        t.add(ts, px)
    return t


def test_a_window_reports_both_directions_as_positive_ticks():
    t = build_tape([(0.0, 100.0), (1.0, 101.0), (2.0, 98.0)])
    up, down = t.travel(0, 2.0)
    assert up == pytest.approx(4.0)
    assert down == pytest.approx(8.0)


def test_a_window_running_past_the_end_is_none_not_a_short_window():
    """A truncated window understates travel, and counting it as a miss
    would quietly bias the null downwards -- which flatters the signal."""
    t = build_tape([(0.0, 100.0), (1.0, 101.0)])
    assert t.travel(0, 60.0) is None
    assert t.travel(0, 1.0) is not None


def test_a_window_past_the_last_index_is_none():
    assert build_tape([(0.0, 100.0)]).travel(5, 1.0) is None


def test_the_null_is_scored_in_a_coin_flip_direction():
    """A one-sided tape: price only ever rises. A null that always scored
    the favourable side would report 100%; a coin should report about
    half."""
    import random as _r
    t = build_tape([(float(i), 100.0 + i * 0.25) for i in range(400)])
    got = at.control_rate(t, [10.0] * 200, 4, _r.Random(1), draws=4)
    assert got is not None
    assert 0.3 < got < 0.7, got


def test_the_null_uses_the_durations_it_is_given():
    import random as _r
    t = build_tape([(float(i), 100.0 + (i % 7) * 0.25) for i in range(600)])
    short = at.control_rate(t, [2.0] * 200, 4, _r.Random(1))
    long_ = at.control_rate(t, [200.0] * 200, 4, _r.Random(1))
    assert short is not None and long_ is not None
    assert long_ > short, (short, long_)


def test_the_null_is_none_rather_than_zero_on_an_empty_tape():
    import random as _r
    assert at.control_rate(at.Tape(), [5.0], 4, _r.Random(1)) is None


def test_the_null_is_reproducible_from_its_seed():
    import random as _r
    t = build_tape([(float(i), 100.0 + (i % 5) * 0.25) for i in range(500)])
    a = at.control_rate(t, [10.0] * 50, 4, _r.Random(7))
    b = at.control_rate(t, [10.0] * 50, 4, _r.Random(7))
    assert a == b


def test_travel_is_measured_from_the_agreement_not_the_bar_open():
    """The bar may have opened far away. That distance was not available
    to anyone waiting for the three to line up."""
    e = ep("up", start=100.0)
    e.observe(100.0)
    e.observe(101.0)
    assert e.mfe == pytest.approx(4.0)       # not 'since the bar open'


def test_reaching_a_distance_is_the_high_water_mark_not_the_close():
    e = ep("up")
    for px in (100.0, 103.0, 100.0):
        e.observe(px)
    e.end_ts, e.end_px = 9.0, 100.0
    assert e.reached(12) is True
    assert e.net == pytest.approx(0.0)


# ------------------------------------------------------------- the walk

def test_flat_at_a_bar_boundary_does_not_end_an_episode(passthrough):
    """The first print of a new minute sits ON that minute's open, so the
    1m reads flat. Ending there would cut every episode at :00 and the
    tick figure would be a measure of where the boundaries fell."""
    rows = [(1.0, 7800.0)]
    px = 7800.0
    for k in range(1, 40):                   # climbs through 60s
        px += 0.25
        rows.append((1.0 + k * 3.0, px))
    eps, _, _, _ = at.walk(tape(rows))
    assert eps
    longest = max(e.seconds for e in eps)
    assert longest > 60.0, [e.seconds for e in eps]


def test_a_flip_ends_the_episode(passthrough):
    rows = [(1.0, 7800.0)]
    px = 7800.0
    for k in range(1, 12):
        px += 0.25
        rows.append((1.0 + k, px))
    for k in range(12, 30):                  # back below the 1m open
        px -= 0.5
        rows.append((1.0 + k, px))
    eps, _, _, _ = at.walk(tape(rows))
    assert eps
    assert any(e.ended.startswith("flip_") for e in eps), \
        [e.ended for e in eps]


def test_the_fifteen_minute_close_ends_it_whatever_else_is_true(
        passthrough):
    rows = []
    px = 7800.0
    t = 880.0                                 # inside the first 15m bar
    for k in range(60):
        px += 0.25
        rows.append((t, px))
        t += 1.0                              # crosses 900.0
    eps, _, _, _ = at.walk(tape(rows))
    assert any(e.ended == "bar_close" for e in eps), [e.ended for e in eps]
    for e in eps:
        if e.ended == "bar_close":
            assert e.end_ts >= 900.0


def test_no_episode_spans_two_fifteen_minute_bars(passthrough):
    rows = []
    px = 7800.0
    t = 600.0
    for k in range(900):
        px += 0.25 if k % 3 else 0.0
        rows.append((t, px))
        t += 1.0
    eps, _, _, _ = at.walk(tape(rows))
    assert eps
    for e in eps:
        assert e.end_ts - e.bar15_ts <= 900.0 + 1e-6


def test_episodes_do_not_overlap_in_time(passthrough):
    rows = []
    px = 7800.0
    t = 10.0
    for k in range(400):
        px += 0.25 if (k // 7) % 2 == 0 else -0.25
        rows.append((t, px))
        t += 2.0
    eps, _, _, _ = at.walk(tape(rows))
    assert len(eps) >= 2
    for a, b in zip(eps, eps[1:]):
        assert a.end_ts <= b.start_ts, (a.end_ts, b.start_ts)


def test_a_flat_tape_produces_no_episodes(passthrough):
    eps, _, n, _ = at.walk(tape([(float(i), 7800.0) for i in range(300)]))
    assert n == 300
    assert eps == []


def test_a_down_run_is_found_as_a_down_episode(passthrough):
    rows = [(1.0, 7800.0)]
    px = 7800.0
    for k in range(1, 40):
        px -= 0.25
        rows.append((1.0 + k, px))
    eps, _, _, _ = at.walk(tape(rows))
    assert eps
    assert all(e.way == "down" for e in eps), [e.way for e in eps]
    assert max(e.mfe for e in eps) > 0


def test_the_episode_records_where_in_the_bar_it_formed(passthrough):
    rows = []
    px = 7800.0
    t = 300.0                                 # five minutes into the bar
    for k in range(30):
        px += 0.25
        rows.append((t + k, px))
    eps, _, _, _ = at.walk(tape(rows))
    assert eps
    assert 0.0 <= eps[0].offset_s < 900.0
    assert eps[0].offset_s == pytest.approx(300.0, abs=2.0)


def test_the_first_lapse_in_strict_agreement_is_recorded_separately(
        passthrough):
    """It does not end the episode, but it is worth knowing when it
    happened and which timeframe it was."""
    rows = [(1.0, 7800.0)]
    px = 7800.0
    for k in range(1, 120):
        px += 0.25
        rows.append((1.0 + k, px))
    eps, _, _, _ = at.walk(tape(rows))
    assert eps
    broke = [e for e in eps if e.first_break]
    assert broke, "no episode ever lapsed out of strict agreement"
    for e in broke:
        assert e.first_break in ("1m", "5m", "15m")
        assert e.break_ts is not None and e.break_ts >= e.start_ts


def test_a_file_timestamped_from_the_epoch_is_dated_correctly(passthrough):
    """The falsy zero, for the third time in this project.

    `bar_ts or ts` treats a 15-minute slot of 0.0 as missing, so every
    episode in the first quarter hour gets dated against its own start and
    reports an offset of zero -- which silently empties the 'where in the
    bar did it form' table into one bucket. `first_ts == 0.0` is the same
    mistake: it reassigns on the second print and drops one from the span.
    """
    rows = []
    px = 7800.0
    for k in range(40):                       # starts at ts == 0.0
        px += 0.25
        rows.append((float(k) * 5.0, px))
    eps, _, n, span = at.walk(tape(rows))
    assert n == 40
    assert span == pytest.approx(195.0)       # 0.0 .. 195.0, not 190.0
    assert eps
    assert eps[0].bar15_ts == 0.0
    assert eps[0].offset_s > 0.0, "episode dated against its own start"


def test_an_episode_later_in_the_session_is_dated_against_its_own_bar(
        passthrough):
    rows = []
    px = 7800.0
    t = 1_791_466_200.0                       # a real-looking stamp
    for k in range(40):
        px += 0.25
        rows.append((t + k * 5.0, px))
    eps, _, _, _ = at.walk(tape(rows))
    assert eps
    for e in eps:
        assert 0.0 <= e.offset_s < 900.0
        assert e.bar15_ts == math.floor(e.start_ts / 900.0) * 900.0


def test_the_walk_counts_only_prints(passthrough):
    eps, _, n, span = at.walk(tape([(float(i), 7800.0 + i * 0.25)
                                 for i in range(50)]))
    assert n == 50
    assert span == pytest.approx(49.0)


# ------------------------------------------------------------- statistics

def test_percentiles_are_ordered_and_hit_the_ends():
    xs = list(range(101))
    assert at._pct(xs, 0.0) == 0
    assert at._pct(xs, 0.5) == 50
    assert at._pct(xs, 1.0) == 100
    assert at._pct([], 0.5) is None


def _group(n, seconds=5.0):
    out = []
    for _ in range(n):
        e = ep("up")
        e.observe(100.0)
        e.end_ts, e.end_px = seconds, 100.0
        out.append(e)
    return out


def _tape_for_null():
    return build_tape([(float(i) * 0.5, 100.0 + (i % 9) * 0.25)
                       for i in range(2000)])


def test_a_thin_group_is_not_given_a_null_row(capsys):
    """A percentage over a handful of episodes reads as a finding, and a
    null printed beside it makes it read as a compared one."""
    import random as _r
    at.projection(_group(5), "tiny", _tape_for_null(), _r.Random(1))
    assert "random start" not in capsys.readouterr().out


def test_a_group_at_the_floor_does_get_its_null(capsys):
    import random as _r
    at.projection(_group(at.MIN_GROUP), "enough", _tape_for_null(),
                  _r.Random(1))
    assert "random start" in capsys.readouterr().out


def test_without_a_tape_no_null_is_invented(capsys):
    at.projection(_group(at.MIN_GROUP), "no tape")
    assert "random start" not in capsys.readouterr().out


def test_the_fixed_horizon_table_reports_its_own_null(capsys):
    import random as _r
    tape = _tape_for_null()
    eps = []
    for i in range(at.MIN_GROUP + 10):
        e = ep("up", start=100.0)
        e.start_ts = float(i)
        e.observe(100.0)
        e.end_ts, e.end_px = float(i) + 5.0, 100.0
        eps.append(e)
    at.horizon_table(eps, tape, _r.Random(1), "all")
    out = capsys.readouterr().out
    assert "random start" in out
    for h in at.HORIZONS_S:
        assert f"{h:.0f}s" in out


def test_the_fixed_horizon_table_scores_a_short_the_other_way(capsys):
    """A table built only from longs cannot tell a direction-aware
    measurement from one that always reads the upside -- and on a tape
    that only rises, the wrong one reports a short as a winner."""
    import random as _r
    rising = build_tape([(float(i) * 0.5, 100.0 + i * 0.25)
                         for i in range(2000)])
    shorts = []
    for i in range(at.MIN_GROUP + 10):
        e = ep("down", start=100.0 + i * 0.25)
        e.start_ts = float(i) * 0.5
        e.observe(e.start_px)
        e.end_ts, e.end_px = e.start_ts + 5.0, e.start_px
        shorts.append(e)
    at.horizon_table(shorts, rising, _r.Random(1), "shorts")
    out = capsys.readouterr().out
    first = [ln for ln in out.splitlines()
             if "shorts" in ln and "random" not in ln][0]
    assert "0.0%" in first, first
    assert "100.0%" not in first, first


def test_the_null_skips_truncated_windows_rather_than_failing_them(capsys):
    """Scoring a window that ran off the end as a miss biases the null
    DOWNWARDS, which flatters whatever it is being compared against."""
    import random as _r
    # a tape that always travels a long way, and durations most of which
    # cannot fit inside it
    t = build_tape([(float(i), 100.0 + i * 0.25) for i in range(200)])
    durations = [150.0] * 60
    got = at.control_rate(t, durations, 4, _r.Random(3), draws=4)
    assert got is not None
    # every window that actually fits travels far past four ticks, so a
    # correct null is near 50% (the coin), not dragged toward zero by the
    # ones that did not fit
    assert got > 0.4, got


def test_the_horizons_are_shorter_than_the_bar_they_sit_in():
    assert max(at.HORIZONS_S) <= at.TRADE_FRAME_S


def test_an_empty_group_says_so_rather_than_dividing_by_zero(capsys):
    at.projection([], "nothing")
    assert "(none)" in capsys.readouterr().out
    at.spread([], "nothing")
    assert "(none)" in capsys.readouterr().out


def test_the_projection_distances_reach_the_traded_size():
    """Four to five points is sixteen to twenty ticks."""
    assert 16 in at.PROJECTION and 20 in at.PROJECTION


def test_the_offset_buckets_tile_the_whole_bar():
    edges = [lo for lo, _ in at.OFFSET_BUCKETS] + [at.OFFSET_BUCKETS[-1][1]]
    assert edges[0] == 0.0
    assert edges[-1] == at.TRADE_FRAME_S
    for (lo, hi), nxt in zip(at.OFFSET_BUCKETS, at.OFFSET_BUCKETS[1:]):
        assert hi == nxt[0]


# ------------------------------------------------- no strategy, no orders

def test_it_scores_no_trade_and_names_no_target():
    import ast
    import inspect

    class Blank(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(value=""), node)
            return node

        def visit_JoinedStr(self, node):
            return ast.copy_location(ast.Constant(value=""), node)

    tree = Blank().visit(ast.parse(inspect.getsource(at)))
    ast.fix_missing_locations(tree)
    code = ast.unparse(tree).lower()
    for word in ("win_rate", "breakeven", "stop_ticks", "first_touch",
                 "expectancy", "optimal", "recommend"):
        assert word not in code, word


def test_no_production_module_is_mutated():
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(at))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute):
                    assert getattr(t.value, "id", "") != "dbn"


def test_nothing_here_places_an_order():
    import inspect
    src = inspect.getsource(at).lower()
    for word in ("def submit", "def execute", "def place", "def order"):
        assert word not in src


def test_progress_never_writes_to_the_report():
    import io
    buf = io.StringIO()
    p = at.Progress("x", every=2, out=buf)
    p.tick(); p.tick(); p.done()
    assert buf.getvalue()
    assert at.Progress("x").out is sys.stderr


def test_a_limit_takes_the_first_records():
    assert list(at.capped(range(10), 3)) == [0, 1, 2]
    assert list(at.capped(range(3), 0)) == [0, 1, 2]
