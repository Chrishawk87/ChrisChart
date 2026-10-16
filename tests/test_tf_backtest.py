"""Two years of ES at a four-tick target. The ways this would lie.

THE FILL MUST FOLLOW THE SIGNAL'S MINUTE. A signal is decided at a
1-minute bar's CLOSE, so the first price anyone could have paid is the
open of the next 1-second bar. Filling inside the signal's own minute is
fifty-nine seconds of look-ahead, and that exact bug scored 98.4% on a
pure random walk in this project.

A BAR THAT SPANS BOTH LEVELS SETTLES NOTHING. It touched the target and
the stop and records no order between them. Awarding it to the target is
the easiest way in existence to manufacture a win rate, so it is dropped
and counted.

A SWING IS NOT KNOWN AT ITS PIVOT. `structure.swings` confirms one
`strength` bars later; reading it at its own bar is the structure version
of the same lie.

NO TRADE CROSSES A ROLL. Across a contract change the price steps by
carry, so a position held through one books a move nobody could make.

AND A SHORT IS NOT A LONG WITH A MINUS SIGN SOMEWHERE. Getting the
target and stop the wrong way round on shorts turns every losing short
into a winner -- it has already happened once in this project.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load():
    path = ROOT / "scripts" / "tf_backtest.py"
    spec = importlib.util.spec_from_file_location("tf_backtest", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tf_backtest"] = mod
    spec.loader.exec_module(mod)
    return mod


tb = _load()
T = tb.TICK
NS = 1_000_000_000
PX = 1_000_000_000


def rec(ts, o, h, l, c, v=10, iid=1):
    return {"ts_event": int(ts * NS), "open": int(o * PX), "high": int(h * PX),
            "low": int(l * PX), "close": int(c * PX), "volume": v,
            "instrument_id": iid}


def secs(rows):
    """rows: (ts, o, h, l, c)."""
    s = tb.Seconds()
    for r in rows:
        s.add(*r)
    return s


def flat(n, px=5000.0, start=0.0, step=1.0):
    return [(start + i * step, px, px, px, px) for i in range(n)]


# ------------------------------------------------- PHASE 2: the bars

# A real-looking epoch stamp. `dbn.seconds` returns 0.0 for a stamp it
# cannot parse and no ES bar is dated 1 January 1970, so the reader treats
# zero as unreadable -- which means a fixture counting from zero quietly
# loses its first bar and the test measures the guard instead of the code.
T0 = 1_727_654_400.0        # 2024-09-30 UTC


def test_a_segment_breaks_at_a_change_of_instrument():
    recs = ([rec(T0 + i, 100, 100, 100, 100, iid=1) for i in range(5)]
            + [rec(T0 + 100 + i, 200, 200, 200, 200, iid=2)
               for i in range(5)])
    got = list(tb.segments(recs))
    assert [iid for iid, _ in got] == [1, 2]
    assert [len(s) for _, s in got] == [5, 5]


def test_a_single_contract_is_one_segment():
    recs = [rec(T0 + i, 100, 100, 100, 100) for i in range(20)]
    got = list(tb.segments(recs))
    assert len(got) == 1 and len(got[0][1]) == 20


def test_an_unreadable_timestamp_is_dropped():
    """Zero means the stamp did not parse, and a bar with no time cannot
    be placed on any grid."""
    recs = [{"ts_event": None, "open": int(100 * PX), "high": int(100 * PX),
             "low": int(100 * PX), "close": int(100 * PX),
             "volume": 1, "instrument_id": 1},
            rec(T0, 100, 100, 100, 100)]
    got = list(tb.segments(recs))
    assert len(got[0][1]) == 1


def test_malformed_rows_are_dropped_not_repaired():
    recs = [rec(T0, 100, 99, 101, 100),         # high below low
            rec(T0 + 1, 100, 101, 99, 100),     # fine
            rec(T0 + 2, 105, 101, 99, 100)]     # open outside the range
    got = list(tb.segments(recs))
    assert len(got[0][1]) == 1


def test_bars_aggregate_onto_a_clock_grid_not_from_the_first_print():
    """Bars sit on the clock. Starting them wherever the data starts puts
    every timeframe on its own private grid and nothing lines up."""
    s = secs([(70.0, 10, 12, 9, 11), (71.0, 11, 13, 10, 12),
              (130.0, 20, 21, 19, 20)])
    out = tb.build(s, 60.0)
    assert [c.ts for c in out] == [60.0, 120.0]
    assert out[0].open == 10 and out[0].high == 13
    assert out[0].low == 9 and out[0].close == 12


def test_aggregation_takes_the_extremes_and_the_last_close():
    s = secs([(0.0, 10, 15, 8, 12), (1.0, 12, 13, 5, 9)])
    c = tb.build(s, 60.0)[0]
    assert (c.open, c.high, c.low, c.close) == (10, 15, 5, 9)


# ------------------------------- PHASE 3: the definitions, kept apart

def cands(closes, highs=None, lows=None, span=60.0):
    highs = highs or [c + 1 for c in closes]
    lows = lows or [c - 1 for c in closes]
    return [tb.Candle(ts=i * span, open=closes[i], high=highs[i],
                      low=lows[i], close=closes[i])
            for i in range(len(closes))]


def test_momentum_reads_the_last_two_completed_closes():
    c = cands([10, 11, 10])
    assert tb.momentum(c, 1) == "up"
    assert tb.momentum(c, 2) == "down"


def test_equal_closes_are_neutral_not_a_direction():
    assert tb.momentum(cands([10, 10]), 1) == ""


def test_momentum_says_nothing_with_no_predecessor():
    assert tb.momentum(cands([10]), 0) == ""


def test_momentum_never_reads_past_the_bar_it_was_given():
    """Index i is the newest COMPLETED bar. Anything after it is the
    future and must not change the answer."""
    rising = cands([10, 11, 12, 13])
    crash = cands([10, 11, 2, 1])
    assert tb.momentum(rising, 1) == tb.momentum(crash, 1)


def test_structure_needs_confirmed_swings_on_both_sides():
    c = cands(list(range(60)))
    assert tb.structure_way(c, 5) == ""          # not enough history


def test_structure_calls_a_clean_climb_up():
    closes, highs, lows = [], [], []
    px = 100.0
    for k in range(80):
        px += 1.0 if (k // 4) % 2 == 0 else -0.5
        closes.append(px)
        highs.append(px + 0.5)
        lows.append(px - 0.5)
    c = cands(closes, highs, lows)
    assert tb.structure_way(c, len(c) - 1) == "up"


def test_structure_calls_a_clean_decline_down():
    closes, highs, lows = [], [], []
    px = 100.0
    for k in range(80):
        px -= 1.0 if (k // 4) % 2 == 0 else -0.5
        closes.append(px)
        highs.append(px + 0.5)
        lows.append(px - 0.5)
    c = cands(closes, highs, lows)
    assert tb.structure_way(c, len(c) - 1) == "down"


def test_structure_reads_no_swing_before_it_was_confirmed():
    """THE structure look-ahead. A pivot sits on its bar but is not
    visible until `strength` bars later; reading it at its bar is the
    same lie as filling inside the signal's own minute."""
    import liqmap.structure as st
    closes, highs, lows = [], [], []
    px = 100.0
    for k in range(80):
        px += 1.0 if (k // 4) % 2 == 0 else -0.5
        closes.append(px)
        highs.append(px + 0.5)
        lows.append(px - 0.5)
    c = cands(closes, highs, lows)
    window = [st.Candle(ts=x.ts, open=x.open, high=x.high, low=x.low,
                        close=x.close, volume=0.0, trades=0) for x in c]
    sw = st.swings(window, strength=tb.SWING_STRENGTH)
    assert sw
    for s in sw:
        assert s.confirmed_at == s.index + tb.SWING_STRENGTH

    import inspect
    src = inspect.getsource(tb.structure_way)
    assert "s.confirmed_at <= i" in src


def test_the_two_definitions_are_separate_functions():
    import inspect
    assert set(tb.WAY) == {"momentum", "structure"}
    a = inspect.getsource(tb.momentum)
    b = inspect.getsource(tb.structure_way)
    assert "swings" not in a
    assert "close" not in b.split("def structure_way")[1].split("return")[0]


# --------------------------------------------- PHASE 4: fill and resolve

def test_the_fill_is_the_bar_after_the_signal_minute_closed():
    s = secs([(59.0, 1, 1, 1, 1), (60.0, 2, 2, 2, 2), (61.0, 3, 3, 3, 3)])
    assert tb.first_at_or_after(s, 60.0) == 1
    assert s.op[tb.first_at_or_after(s, 60.0)] == 2


def test_the_fill_never_lands_inside_the_signals_own_minute():
    """Fifty-nine seconds of look-ahead, and it does not crash."""
    s = secs([(t, 1, 1, 1, 1) for t in range(0, 60)]
             + [(60.0, 9, 9, 9, 9)])
    i = tb.first_at_or_after(s, 60.0)
    assert s.ts[i] == 60.0


def test_a_long_reaching_its_target_first_is_a_win():
    s = secs([(0.0, 100, 100, 100, 100), (1.0, 100, 101.0, 100, 101.0)])
    out, px, _, _ = tb.resolve(s, 0, 100.0, "buy", 4, 300.0)
    assert out == "target" and px == pytest.approx(101.0)


def test_a_long_hitting_its_stop_first_is_a_loss():
    s = secs([(0.0, 100, 100, 100, 100), (1.0, 100, 100, 99.0, 99.0)])
    out, px, _, _ = tb.resolve(s, 0, 100.0, "buy", 4, 300.0)
    assert out == "stop" and px == pytest.approx(99.0)


def test_a_short_is_resolved_the_other_way_round():
    """THE sign bug. A short wins when price FALLS."""
    down = secs([(0.0, 100, 100, 100, 100), (1.0, 100, 100, 99.0, 99.0)])
    up = secs([(0.0, 100, 100, 100, 100), (1.0, 100, 101.0, 100, 101.0)])
    assert tb.resolve(down, 0, 100.0, "sell", 4, 300.0)[0] == "target"
    assert tb.resolve(up, 0, 100.0, "sell", 4, 300.0)[0] == "stop"


def test_a_bar_spanning_both_levels_is_ambiguous_not_a_win():
    """It touched the target AND the stop and records no order. This is
    the single easiest place to manufacture a win rate."""
    s = secs([(0.0, 100, 100, 100, 100),
              (1.0, 100, 102.0, 98.0, 100.0)])
    out, _, _, _ = tb.resolve(s, 0, 100.0, "buy", 4, 300.0)
    assert out == "ambiguous"
    assert tb.resolve(s, 0, 100.0, "sell", 4, 300.0)[0] == "ambiguous"


def test_a_wider_stop_can_make_the_same_bar_resolvable():
    s = secs([(0.0, 100, 100, 100, 100), (1.0, 100, 101.0, 99.5, 100.0)])
    assert tb.resolve(s, 0, 100.0, "buy", 2, 300.0)[0] == "ambiguous"
    assert tb.resolve(s, 0, 100.0, "buy", 8, 300.0)[0] == "target"


def test_a_trade_that_reaches_neither_level_times_out_at_its_price():
    """A timeout is not a win and not a loss. It books what it was
    actually worth when the clock ran out."""
    s = secs([(float(i), 100, 100.25, 99.75, 100.25) for i in range(10)])
    out, px, _, _ = tb.resolve(s, 0, 100.0, "buy", 4, 5.0)
    assert out == "timeout"
    assert px == pytest.approx(100.25)


def test_a_timeout_books_a_loss_when_it_ended_underwater():
    t = tb.Trade(signal_ts=0, entry_ts=0, entry=100.0, side="buy",
                 stop_ticks=4, cap_s=300.0, exit=99.75, outcome="timeout")
    assert t.ticks == pytest.approx(-1.0)


def test_the_time_cap_is_respected():
    s = secs([(float(i), 100, 100, 100, 100) for i in range(20)]
             + [(500.0, 100, 110, 100, 110)])
    assert tb.resolve(s, 0, 100.0, "buy", 4, 10.0)[0] == "timeout"
    assert tb.resolve(s, 0, 100.0, "buy", 4, 600.0)[0] == "target"


def test_a_trade_with_no_bars_left_is_unfinished_not_a_timeout():
    """At the very end of a contract there is no room to resolve, and
    calling that a flat scratch invents a trade nobody could take."""
    s = secs([(0.0, 100, 100, 100, 100)])
    assert tb.resolve(s, 0, 100.0, "buy", 4, 300.0)[0] == "unfinished"


# ---------------------------------------------- the walk and the rolls

def rising(n, start_ts=0.0, px=5000.0, per=0.25):
    rows = []
    for i in range(n):
        p = px + (i // 60) * per
        rows.append((start_ts + i, p, p + 0.25, p - 0.25, p))
    return rows


def test_no_trade_crosses_a_contract_roll():
    """Each contract is walked on its own, so a position cannot be held
    through a price step that is carry rather than a move."""
    import inspect
    src = inspect.getsource(tb.main)
    assert "for iid, sec in folds" in src
    sig = inspect.signature(tb.walk)
    assert "sec" in sig.parameters


def test_only_one_position_is_open_at_a_time():
    s = secs(rising(4000))
    trades = tb.walk(s, "momentum", ("1m",), 4, 300.0)
    for a, b in zip(trades, trades[1:]):
        assert a.exit_ts <= b.entry_ts, (a.exit_ts, b.entry_ts)


def test_every_entry_follows_its_own_signal():
    s = secs(rising(4000))
    for t in tb.walk(s, "momentum", ("1m",), 4, 300.0):
        assert t.entry_ts >= t.signal_ts


def test_a_flat_tape_produces_no_trades():
    s = secs(flat(3000))
    assert tb.walk(s, "momentum", ("1m", "5m", "15m"), 4, 300.0) == []


def test_requiring_more_timeframes_never_adds_trades():
    """A filter removes; it cannot invent an entry."""
    s = secs(rising(8000))
    one = len(tb.walk(s, "momentum", ("1m",), 4, 300.0))
    three = len(tb.walk(s, "momentum", ("1m", "5m", "15m"), 4, 300.0))
    assert three <= one


def test_the_mirror_flips_every_call():
    s = secs(rising(6000))
    a = tb.walk(s, "momentum", ("1m",), 4, 300.0)
    b = tb.walk(s, "momentum", ("1m",), 4, 300.0, flip=True)
    assert a and b
    assert a[0].side != b[0].side


def test_the_control_varies_the_call_and_keeps_the_sample():
    import random
    s = secs(rising(6000))
    a = tb.walk(s, "momentum", ("1m",), 4, 300.0)
    c = tb.walk(s, "momentum", ("1m",), 4, 300.0,
                rng=random.Random(1))
    assert a and c
    assert {t.side for t in c} - {"buy", "sell"} == set()


# ------------------------------------------------ PHASE 5: the scoring

def trade(outcome, side="buy", entry=100.0, exit_=None, secs_=1.0):
    if exit_ is None:
        exit_ = {"target": 101.0, "stop": 99.0, "timeout": 100.0,
                 "ambiguous": 0.0}[outcome]
    return tb.Trade(signal_ts=0.0, entry_ts=0.0, entry=entry, side=side,
                    stop_ticks=4, cap_s=300.0, exit=exit_,
                    outcome=outcome, seconds=secs_)


def test_ambiguous_trades_are_excluded_from_every_figure():
    r = tb.score([trade("ambiguous"), trade("target")], cost=0.0)
    assert r.ambiguous == 1
    assert r.scored == 1
    assert r.wins == 1


def test_the_win_rate_ignores_timeouts_but_the_net_does_not():
    r = tb.score([trade("target"), trade("stop"), trade("timeout")],
                 cost=0.0)
    assert r.win_rate == pytest.approx(0.5)      # 1 of 2 decided
    assert r.scored == 3                          # but 3 cost money


def test_costs_are_charged_on_every_scored_trade():
    r = tb.score([trade("target")] * 10, cost=1.0)
    assert r.gross == pytest.approx(40.0)
    assert r.net(1.0) == pytest.approx(30.0)


def test_a_short_that_fell_is_a_winner():
    r = tb.score([trade("target", side="sell", exit_=99.0)], cost=0.0)
    assert r.gross == pytest.approx(4.0)


def test_drawdown_is_measured_from_the_running_peak():
    seq = [trade("target"), trade("stop"), trade("stop"), trade("target")]
    r = tb.score(seq, cost=0.0)
    assert r.max_dd == pytest.approx(8.0)


def test_the_losing_streak_is_the_longest_run_not_the_total():
    seq = [trade("stop"), trade("target"), trade("stop"), trade("stop"),
           trade("stop"), trade("target")]
    assert tb.score(seq, cost=0.0).worst_streak == 3


def test_profit_factor_is_earned_over_given_up():
    r = tb.score([trade("target"), trade("target"), trade("stop")],
                 cost=0.0)
    assert r.profit_factor == pytest.approx(8.0 / 4.0)


def test_profit_factor_is_none_rather_than_infinite():
    assert tb.score([trade("target")], cost=0.0).profit_factor is None


def test_break_even_rises_with_the_stop_and_the_cost():
    assert tb.breakeven(4, 0.0) == pytest.approx(0.5)
    assert tb.breakeven(4, 1.0) == pytest.approx(0.625)
    assert tb.breakeven(8, 1.0) == pytest.approx(0.75)
    assert tb.breakeven(8, 1.5) > tb.breakeven(8, 1.0)


def test_the_interval_is_wide_at_small_n_and_tight_at_large():
    lo, hi = tb.wilson(6, 9)
    assert hi - lo > 0.4
    lo2, hi2 = tb.wilson(6000, 9000)
    assert hi2 - lo2 < 0.03


def test_a_thin_arm_gets_no_verdict(capsys):
    tb.row("tiny", tb.score([trade("target")] * 5, 1.0), 1.0, 4)
    assert "--" in capsys.readouterr().out


def test_an_empty_arm_says_so_rather_than_dividing_by_zero(capsys):
    tb.row("none", tb.score([], 1.0), 1.0, 4)
    assert "no trades" in capsys.readouterr().out


# ---------------------------------------------- no strategy, no orders

def test_no_threshold_is_chosen_on_the_outcome():
    import ast
    import inspect

    class Blank(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(value=""), node)
            return node

        def visit_JoinedStr(self, node):
            return ast.copy_location(ast.Constant(value=""), node)

    tree = Blank().visit(ast.parse(inspect.getsource(tb)))
    ast.fix_missing_locations(tree)
    code = ast.unparse(tree).lower()
    for word in ("optimal", "recommend", "argmax", "best_stop", "tune"):
        assert word not in code, word
    assert "max(stop_ticks" not in code


def test_the_stops_and_caps_are_grids():
    assert len(tb.STOP_TICKS) >= 3
    assert len(tb.TIME_CAPS_S) >= 3
    assert len(tb.COSTS) >= 3


def test_the_target_is_the_one_the_brief_specified():
    assert tb.TARGET_TICKS == 4
    assert tb.TARGET_TICKS * tb.DOLLARS_PER_TICK == pytest.approx(50.0)


def test_no_production_module_is_mutated():
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(tb))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute):
                    assert getattr(t.value, "id", "") not in ("dbn", "st")


def test_nothing_here_places_an_order():
    import inspect
    src = inspect.getsource(tb).lower()
    for word in ("def submit", "def execute", "def place", "def order"):
        assert word not in src


def test_progress_never_writes_to_the_report():
    import io
    buf = io.StringIO()
    p = tb.Progress("x", every=2, out=buf)
    p.tick(); p.tick(); p.done()
    assert buf.getvalue()
    assert tb.Progress("x").out is sys.stderr


# ------------------------------ the bar in progress is not a fact yet

def grid_of(n, span=900.0, start=0.0):
    return [tb.Candle(ts=start + i * span, open=10.0, high=11.0, low=9.0,
                      close=10.0) for i in range(n)]


def test_a_bar_still_forming_is_never_read():
    """THE 15-minute look-ahead. A bar that has not closed has no final
    high, low or close, so reading it is up to fifteen minutes of the
    future -- and every individual number in the study would still look
    perfectly plausible."""
    g = grid_of(3)                       # bars opening at 0, 900, 1800
    assert tb.newest_completed(g, 900.0, 900.0) == 0
    assert tb.newest_completed(g, 900.0, 1799.0) == 0
    assert tb.newest_completed(g, 900.0, 1800.0) == 1
    assert tb.newest_completed(g, 900.0, 2699.0) == 1


def test_nothing_is_readable_before_the_first_bar_has_closed():
    g = grid_of(3)
    assert tb.newest_completed(g, 900.0, 0.0) is None
    assert tb.newest_completed(g, 900.0, 899.0) is None


def test_an_empty_grid_reads_as_nothing_rather_than_crashing():
    assert tb.newest_completed([], 900.0, 5000.0) is None


def test_the_search_may_start_from_a_hint_without_changing_the_answer():
    """The walk carries an index forward for speed. A hint must not let
    it skip past the right bar or land on a later one."""
    g = grid_of(20)
    for now in (900.0, 4500.0, 9000.0, 17100.0):
        plain = tb.newest_completed(g, 900.0, now)
        for hint in range(0, (plain or 0) + 1):
            assert tb.newest_completed(g, 900.0, now, hint) == plain


def test_a_gap_in_the_grid_does_not_promote_an_unclosed_bar():
    g = [tb.Candle(ts=0.0, open=1, high=1, low=1, close=1),
         tb.Candle(ts=9000.0, open=1, high=1, low=1, close=1)]
    assert tb.newest_completed(g, 900.0, 5000.0) == 0
    assert tb.newest_completed(g, 900.0, 9500.0) == 0
    assert tb.newest_completed(g, 900.0, 9900.0) == 1


def test_the_walk_uses_that_one_rule_rather_than_its_own():
    import inspect
    src = inspect.getsource(tb.walk)
    assert "newest_completed(" in src
    assert "while j + 1" not in src


# ----------------------------- structure needs BOTH halves to agree

def structure_from(highs, lows):
    closes = [(h + l) / 2 for h, l in zip(highs, lows)]
    return [tb.Candle(ts=i * 60.0, open=closes[i], high=highs[i],
                      low=lows[i], close=closes[i])
            for i in range(len(highs))]


def test_higher_highs_with_lower_lows_is_not_an_uptrend():
    """A broadening range makes higher highs AND lower lows. Definition A
    needs both halves pointing the same way; accepting either one on its
    own turns every widening range into a trend call."""
    highs, lows = [], []
    for k in range(80):
        swing = (k // 4) % 2 == 0
        spread = 1.0 + k * 0.10          # widening both ways
        mid = 100.0
        highs.append(mid + (spread if swing else spread * 0.3))
        lows.append(mid - (spread if not swing else spread * 0.3))
    c = structure_from(highs, lows)
    way = tb.structure_way(c, len(c) - 1)
    assert way == "", f"a broadening range was called {way!r}"


def test_lower_highs_with_higher_lows_is_not_a_downtrend():
    """The mirror: a contracting range."""
    highs, lows = [], []
    for k in range(80):
        swing = (k // 4) % 2 == 0
        spread = max(0.5, 12.0 - k * 0.12)
        mid = 100.0
        highs.append(mid + (spread if swing else spread * 0.3))
        lows.append(mid - (spread if not swing else spread * 0.3))
    c = structure_from(highs, lows)
    assert tb.structure_way(c, len(c) - 1) == ""


def test_the_definition_requires_both_halves_in_the_source():
    import inspect
    src = inspect.getsource(tb.structure_way)
    assert "hh and hl" in src
    assert "lh and ll" in src
