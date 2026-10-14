"""The stack rule tested on ES, and the ways that test could lie.

LOOK-AHEAD IS THE WHOLE RISK. Every figure read at a checkpoint must come
from prints before it, and every figure scored must come from prints after.
A study that crosses that line produces a beautiful number and no
information.

THE TIE MUST GO AGAINST THE TESTER. A minute that touched the target and
the stop does not record which came first, and resolving it favourably is
the single easiest way to invent an edge.

AND THE COUNT MUST LEAD. This rule fires rarely by design; a study that
prints a hit rate without the sample behind it reads as a result.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load():
    path = ROOT / "scripts" / "stack_study.py"
    spec = importlib.util.spec_from_file_location("stack_study", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["stack_study"] = mod
    spec.loader.exec_module(mod)
    return mod


ss = _load()
M = ss.MINUTE


def mins(n, start=0.0, px=100.0, step=0.0, buy_share=0.5, vol=100.0):
    """`n` minute bars walking by `step` each, with a given buy share."""
    out = []
    for i in range(n):
        p = px + i * step
        m = ss.Minute(ts=start + i * M)
        # Two prints: one buy, one sell, weighted by buy_share.
        m.add(p, vol * buy_share, True)
        m.add(p, vol * (1 - buy_share), False)
        out.append(m)
    return out


# ------------------------------------------------------- the pass over tape

def test_a_minute_keeps_who_crossed_rather_than_just_volume():
    m = ss.Minute(ts=0.0)
    m.add(100.0, 3.0, True)
    m.add(101.0, 1.0, False)
    assert m.trades == 2 and m.volume == 4.0
    assert m.buy_notional == 300.0 and m.sell_notional == 101.0
    assert m.open == 100.0 and m.close == 101.0
    assert m.high == 101.0 and m.low == 100.0


def test_rolling_up_excludes_the_end_and_that_is_the_point():
    """The minute the decision falls in has not finished. Including it
    hands the reader prints that had not happened yet."""
    ms = mins(10, px=100.0, step=1.0)
    bar = ss.roll_up(ms, 0.0, 5 * M)
    assert bar is not None
    assert bar.open == 100.0
    assert bar.close == 104.0          # minute 4, not minute 5
    assert bar.high == 104.0


def test_rolling_up_an_empty_stretch_is_none_not_a_zero_bar():
    assert ss.roll_up(mins(3), 100 * M, 110 * M) is None


def test_a_row_whose_bar_has_not_started_reads_the_one_that_just_closed():
    """On the 1m row at a minute boundary the bar in progress is zero
    seconds old, so it is always empty. Returning nothing there left the
    1m permanently unread and silenced the rule completely -- zero
    signals, on every file."""
    ms = mins(40, px=100.0, step=1.0)
    at = 20 * M                       # exactly a minute boundary
    got = ss.row_at(ms, "1m", at_ts=at)
    assert got is not None


def test_the_checkpoints_never_land_where_two_rows_see_the_same_prints():
    """Four minutes into a 15m bar, the current 5m sub-bar began at the
    same instant, so both rows are built from identical prints and agree
    by construction. A confluence rule tested there is testing nothing."""
    from liqmap import cascade as csc

    for trading, marks in ss.CHECKPOINTS.items():
        for mk in marks:
            at = mk * M                      # seconds into the trading bar
            for tf in csc.STACKS[trading][1:]:
                sub = ss.SECONDS[tf]
                # The confirming row's own bar must have STARTED after the
                # trading bar opened, or it is reading the same window.
                assert math.floor(at / sub) * sub > 0, (trading, mk, tf)


def test_history_is_strictly_before_the_bar_being_judged():
    """A bar that helps set the threshold it is judged against is a
    threshold fitted to its own answer."""
    ms = mins(60, px=100.0, step=1.0)
    hist = ss.completed(ms, 300.0, before_ts=30 * M, want=3)
    assert hist
    for c in hist:
        assert c.ts + 300.0 <= 30 * M


# ---------------------------------------------------------------- the read

def test_a_row_reads_the_bar_in_progress_not_the_finished_one():
    """`pressure.from_live` would otherwise take the wall clock, decide
    hours have passed, and report every bar complete."""
    ms = mins(40, px=100.0, step=0.5, buy_share=0.9)
    got = ss.row_at(ms, "5m", at_ts=22 * M)
    assert got is not None
    colour, bps = got
    assert colour in ("green", "red", "none")
    assert bps > 0            # price rose through the window


def test_a_row_with_no_prints_yet_is_none_rather_than_flat():
    assert ss.row_at(mins(3), "15m", at_ts=500 * M) is None


def test_the_study_calls_the_engines_own_rule():
    """A study that reimplements what it is testing tests the
    reimplementation."""
    import inspect

    src = inspect.getsource(ss)
    assert "csc.read(" in src
    assert "csc.colour_of(" in src
    assert "pr.from_live(" in src


def test_only_a_committed_trend_counts_as_a_signal():
    """A pullback is explicitly not a signal in the rule as stated."""
    import inspect

    src = inspect.getsource(ss.walk)
    assert "State.TREND" in src
    assert "PULLBACK" not in src


# ------------------------------------------------------------- the scoring

def sig(side="buy", px=100.0, ts=10 * M):
    return ss.Signal(ts=ts, side=side, px=px, mark=4)


def test_the_target_coming_first_is_a_win():
    ms = mins(3, start=10 * M, px=100.0)
    ms[1].add(100.5, 1.0, True)        # +2 ticks
    assert ss.score(sig(), ms, ticks=2) is True


def test_the_stop_coming_first_is_a_loss():
    ms = mins(3, start=10 * M, px=100.0)
    ms[1].add(99.5, 1.0, False)
    assert ss.score(sig(), ms, ticks=2) is False


def test_a_minute_touching_both_is_scored_as_the_loss():
    """THE tie rule. The bucket does not record which came first, and
    resolving it favourably is the easiest way to invent an edge."""
    ms = mins(3, start=10 * M, px=100.0)
    ms[1].add(100.5, 1.0, True)
    ms[1].add(99.5, 1.0, False)
    assert ss.score(sig(), ms, ticks=2) is False


def test_nothing_before_the_signal_can_resolve_it():
    """Look-ahead in reverse: a level touched before the call is not the
    call being right."""
    ms = mins(12, start=0.0, px=100.0)
    ms[2].add(100.5, 1.0, True)        # long before the signal
    assert ss.score(sig(ts=10 * M), ms, ticks=2) is None


def test_a_signal_that_never_resolves_is_none_not_a_loss():
    ms = mins(5, start=10 * M, px=100.0)
    assert ss.score(sig(), ms, ticks=2) is None


def test_unresolved_signals_are_counted_apart_rather_than_dropped():
    s = sig()
    s.reached = {2: True, 3: None}
    assert ss.tally([s], 2).n == 1
    t = ss.tally([s], 3)
    assert t.n == 0 and t.unresolved == 1


def test_a_short_is_scored_the_other_way_round():
    ms = mins(3, start=10 * M, px=100.0)
    ms[1].add(99.5, 1.0, False)
    assert ss.score(sig(side="sell"), ms, ticks=2) is True
    ms2 = mins(3, start=10 * M, px=100.0)
    ms2[1].add(100.5, 1.0, True)
    assert ss.score(sig(side="sell"), ms2, ticks=2) is False


# ------------------------------------------------------- what beats what

def test_break_even_is_above_half_once_the_round_turn_is_in():
    """A two-tick scalp paying most of a tick in costs is a far harder
    business than the symmetry suggests."""
    assert ss.breakeven(2, 0.0) == pytest.approx(0.5)
    assert ss.breakeven(2, 0.4) > ss.breakeven(8, 0.4)
    assert ss.breakeven(2, 0.4) == pytest.approx(0.55)


def test_the_interval_is_wide_at_the_small_samples_this_study_lives_at():
    lo, hi = ss.wilson(6, 9)
    assert lo < 0.4 and hi > 0.85
    lo2, hi2 = ss.wilson(600, 900)
    assert hi2 - lo2 < 0.1


def test_an_empty_sample_has_no_interval_rather_than_a_point():
    assert ss.wilson(0, 0) == (0.0, 1.0)


def _rigged(n=200, seed=3):
    """Tape where price rises every other minute and falls between, so a
    correctly-sided signal is reliably right and a wrongly-sided one is
    reliably wrong."""
    ms = []
    px = 100.0
    for i in range(n):
        m = ss.Minute(ts=i * M)
        for k in range(6):
            px += 0.25 if (i // 10) % 2 == 0 else -0.25
            m.add(px, 2.0, (i // 10) % 2 == 0)
        ms.append(m)
    return ms


def test_the_control_rescores_rather_than_permuting_the_results():
    """THE bug this test exists for. An earlier version shuffled the list
    of outcomes and averaged it -- which returns the same number every
    time, because a mean does not care about order. It printed a control
    that matched the real rate to the decimal and read as confirmation.

    A control that cannot differ from the thing it is controlling is
    worse than no control at all.
    """
    ms = _rigged()
    sigs = []
    for i in range(6, 190, 10):
        s = sig(side="buy" if (i // 10) % 2 == 0 else "sell",
                px=ms[i].close, ts=(i + 1) * M)
        s.reached = {2: ss.score(s, ms, 2)}
        sigs.append(s)
    real = ss.tally(sigs, 2)
    assert real.n >= 10
    rate = real.hits / real.n
    ctl, _ = ss.shuffled(sigs, ms, 2, runs=40)
    # The sides were chosen to be right, so the real rate must beat a
    # permutation of them. If the control simply echoed the outcomes
    # these would be equal.
    assert rate > ctl + 0.1, (rate, ctl)


def test_the_control_keeps_the_moments_and_the_mix_of_sides():
    import inspect

    src = inspect.getsource(ss.shuffled)
    assert "rnd.shuffle(sides)" in src
    assert "score(Signal(" in src          # re-scored, not re-averaged


def test_a_control_needs_something_to_shuffle():
    assert ss.shuffled([], [], 2) == (0.0, 0.0)


def test_nothing_here_places_an_order():
    import inspect

    src = inspect.getsource(ss).lower()
    for word in ("def submit", "def execute", "def place", "def order"):
        assert word not in src


# ------------------------------------------------- the live path is untouched

def test_from_live_still_reads_the_wall_clock_when_not_told_otherwise():
    """The new argument exists for the study. Omitting it must behave
    exactly as it did before, because every caller in the service does."""
    import inspect

    from liqmap import pressure as prs

    sg = inspect.signature(prs.from_live)
    assert sg.parameters["now"].default is None
    assert "candle.elapsed(now)" in inspect.getsource(prs.from_live)


def test_a_simulated_clock_sees_a_partial_bar():
    """With the wall clock, a bar from a file is hours old and reads as
    complete -- so `forming` would never be true and every row would
    qualify."""
    from liqmap import pressure as prs
    from liqmap.live import LiveCandle

    c = LiveCandle(interval_s=900.0, start_ts=1_000_000.0, open=100.0,
                   high=101.0, low=99.0, close=100.5, volume=10.0,
                   trades=5, buy_notional=800.0, sell_notional=200.0)
    early = prs.from_live("15m", 900.0, c, None, now=1_000_060.0)
    late = prs.from_live("15m", 900.0, c, None, now=1_000_899.0)
    assert early.elapsed_s == pytest.approx(60.0)
    assert late.elapsed_s == pytest.approx(899.0)
    assert early.elapsed_share < late.elapsed_share


# ------------------------------------------------- the book, and its timing

def _book(ts, px=100.0, bid_sz=40.0, ask_sz=40.0):
    from liqmap.flow import Book, Level
    return Book(coin="ES", ts=ts,
                bids=[Level(px=px - 0.25 * (j + 1), sz=bid_sz)
                      for j in range(10)],
                asks=[Level(px=px + 0.25 * (j + 1), sz=ask_sz)
                      for j in range(10)])


def _trade(ts, px=100.0, sz=1.0, buy=True):
    from liqmap.flow import Trade
    return Trade(px=px, sz=sz, aggressor="buy" if buy else "sell", ts=ts)


def _as_stream(items, monkeypatch):
    """Feed walk_file directly, bypassing the DBN decoder."""
    from liqmap import dbn

    monkeypatch.setattr(dbn, "stream",
                        lambda recs, flip=False, **kw: iter(recs))
    return items


def test_only_the_difference_between_the_two_sides_of_the_book_counts():
    """Both sides churn hard in a fast market. One side's pull share alone
    mostly measures how busy that side is."""
    class R:
        measured = True

        class bid:
            pull_share = 0.7

        class ask:
            pull_share = 0.7
    s = ss.Snap(ts=0.0, read=R())
    assert s.side == ""
    R.ask.pull_share = 0.9
    R.bid.pull_share = 0.2
    assert ss.Snap(ts=0.0, read=R()).side == "buy"
    R.ask.pull_share = 0.2
    R.bid.pull_share = 0.9
    assert ss.Snap(ts=0.0, read=R()).side == "sell"


def test_an_unmeasured_book_says_nothing_rather_than_guessing():
    class R:
        measured = False
    assert ss.Snap(ts=0.0, read=R()).side == ""
    assert ss.Snap(ts=0.0, read=None).side == ""


def test_the_book_is_read_at_the_checkpoint_not_after_it(monkeypatch):
    """THE look-ahead that would matter here. Everything in the window
    must have arrived at or before the checkpoint."""
    marks = (6, 9, 12)
    items = []
    # Quiet, symmetric book up to minute 6, then the asks vanish.
    for sec in range(0, 13 * 60, 5):
        items.append(("trade", _trade(sec)))
        thin = sec > 6 * 60
        items.append(("book", _book(sec + 0.1,
                                    ask_sz=2.0 if thin else 40.0)))
    _as_stream(items, monkeypatch)
    mins, snaps = ss.walk_file(items, "15m", marks)
    at6 = snaps.get(6 * 60.0)
    assert at6 is not None
    # The collapse happens AFTER minute 6, so the read taken there must
    # not already show offers being pulled.
    assert at6.side != "buy"


def test_a_checkpoint_is_snapped_once(monkeypatch):
    marks = (6,)
    items = []
    for sec in range(0, 13 * 60, 5):
        items.append(("trade", _trade(sec)))
        items.append(("book", _book(sec + 0.1)))
    _as_stream(items, monkeypatch)
    _, snaps = ss.walk_file(items, "15m", marks)
    assert len(snaps) == 1
    assert 360.0 in snaps


def test_the_pass_builds_minutes_and_snaps_together(monkeypatch):
    items = []
    for sec in range(0, 20 * 60, 10):
        items.append(("trade", _trade(sec, px=100.0 + sec / 600.0)))
        items.append(("book", _book(sec + 0.1)))
    _as_stream(items, monkeypatch)
    mins, snaps = ss.walk_file(items, "15m", (6, 9, 12))
    assert len(mins) == 20
    assert snaps                      # the book was kept, not discarded


def test_the_signal_records_which_way_the_book_leaned():
    s = ss.Signal(ts=0.0, side="buy", px=100.0, mark=6, dom="buy")
    assert s.dom_agrees and not s.dom_against
    s2 = ss.Signal(ts=0.0, side="buy", px=100.0, mark=6, dom="sell")
    assert s2.dom_against and not s2.dom_agrees
    s3 = ss.Signal(ts=0.0, side="buy", px=100.0, mark=6, dom="")
    assert not s3.dom_agrees and not s3.dom_against


def test_the_book_is_passed_into_the_row_read():
    import inspect

    assert "book" in inspect.signature(ss.row_at).parameters
    assert "pr.from_live(timeframe, step, bar, book," in inspect.getsource(ss)


def test_the_targets_span_the_range_that_was_asked_for():
    assert min(ss.TARGETS) == 5
    assert max(ss.TARGETS) >= 20


def test_direction_is_never_filtered_out():
    """Up or down, it is a trade. A study that quietly kept only longs
    would halve the sample and flatter a trending session."""
    import inspect

    src = inspect.getsource(ss.walk)
    assert 'c.side == "buy"' not in src
    assert 'side != "sell"' not in src
