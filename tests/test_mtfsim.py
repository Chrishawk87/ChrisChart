"""Walking the signals through the tape.

The tests that matter are about the four ways this kind of number gets
inflated:

  * resolving a bar that touched BOTH barriers in the tester's favour;
  * entering at the signal's own close, which is a price you could not
    have had;
  * stacking trades instead of waiting out the bar you just left;
  * reporting a win rate with no control, when a tight target against a
    wide stop wins most of the time by construction.

Each has a test that fails if the code does the convenient thing.
"""

from __future__ import annotations

import pytest

from liqmap import mtfsim as sim
from liqmap.mtf import M15_SECONDS, Signal


class Bar:
    __slots__ = ("ts", "open", "high", "low", "close", "volume")

    def __init__(self, ts, o, h, l, c, v=100.0):
        self.ts, self.open, self.high = float(ts), o, h
        self.low, self.close, self.volume = l, c, v


T0 = 1_758_000_000.0 - (1_758_000_000.0 % 14400.0)


def sig(ts, side="long", target_ticks=10.0, stop_ticks=20.0, status=""):
    return Signal(ts=ts, side=side, trigger=100.0, target=0.0, stop=0.0,
                  target_ticks=target_ticks, stop_ticks=stop_ticks,
                  status=status, m15_start=ts - (ts % M15_SECONDS))


def path(start, prices, step=1.0, span=0.0):
    """One fine bar per price, with an optional symmetric wick."""
    return [Bar(start + i * step, p, p + span, p - span, p)
            for i, p in enumerate(prices)]


# --------------------------------------------------------- the tie rule

def test_a_bar_touching_both_barriers_is_scored_as_the_stop():
    """THE test. A bar does not record the order of the ticks inside it,
    and resolving that in the tester's favour is the easiest way to
    manufacture a win rate nobody can trade."""
    s = sig(T0 + 60, "long", target_ticks=10, stop_ticks=10)
    # entry 100.00 -> target 102.50, stop 97.50. One bar spans both.
    fine = [Bar(T0 + 120, 100.0, 100.0, 100.0, 100.0),
            Bar(T0 + 125, 100.0, 103.0, 97.0, 100.0)]
    t = sim.walk([s], fine, tick=0.25)[0]
    assert t.outcome == sim.STOP
    assert not t.won


def test_a_clean_target_bar_is_a_win():
    s = sig(T0 + 60, "long", target_ticks=10, stop_ticks=10)
    fine = [Bar(T0 + 120, 100.0, 100.0, 100.0, 100.0),
            Bar(T0 + 125, 100.2, 102.6, 100.1, 102.5)]
    t = sim.walk([s], fine, tick=0.25)[0]
    assert t.outcome == sim.TARGET and t.won
    assert t.ticks == pytest.approx(10.0)


def test_the_stop_is_taken_when_it_is_the_only_one_touched():
    s = sig(T0 + 60, "long", target_ticks=10, stop_ticks=10)
    fine = [Bar(T0 + 120, 100.0, 100.0, 100.0, 100.0),
            Bar(T0 + 125, 99.9, 100.0, 97.4, 97.5)]
    t = sim.walk([s], fine, tick=0.25)[0]
    assert t.outcome == sim.STOP
    assert t.ticks == pytest.approx(-10.0)


def test_the_short_side_mirrors():
    s = sig(T0 + 60, "short", target_ticks=10, stop_ticks=10)
    fine = [Bar(T0 + 120, 100.0, 100.0, 100.0, 100.0),
            Bar(T0 + 125, 99.9, 100.0, 97.4, 97.5)]
    t = sim.walk([s], fine, tick=0.25)[0]
    assert t.outcome == sim.TARGET and t.ticks == pytest.approx(10.0)


# -------------------------------------------------------------- the fill

def test_entry_waits_for_the_signals_own_minute_to_finish():
    """THE look-ahead test, and it is not a near miss.

    A signal is stamped with its 1-minute bar's START, because that is how
    every bar in this project is stamped. It is not KNOWN until that bar
    closes, sixty seconds later. Filling at the first fine bar after the
    stamp buys at the beginning of the very minute whose rising close
    produced the long -- fifty-five seconds before anyone could see it.

    That exact bug scored 98.4% on a pure random walk. The entry must sit
    at or after stamp + 60, never inside the minute."""
    s = sig(T0 + 60, "long")
    fine = [Bar(T0 + 60, 90.0, 90.0, 90.0, 90.0),    # inside the signal bar
            Bar(T0 + 65, 95.0, 95.0, 95.0, 95.0),    # still inside it
            Bar(T0 + 115, 98.0, 98.0, 98.0, 98.0),   # still inside it
            Bar(T0 + 120, 100.0, 100.0, 100.0, 100.0)]  # the first legal fill
    t = sim.walk([s], fine, tick=0.25)[0]
    assert t.entry_ts >= T0 + 120, "filled inside the signal's own minute"
    assert t.entry == pytest.approx(100.0)


def test_the_look_ahead_fill_is_not_merely_off_by_one_bar():
    """Pinning 'the next bar' rather than 'after the minute' is what let
    the bug through the first time: on a 5-second grid the next bar is
    still eleven bars inside the signal's own candle."""
    s = sig(T0 + 60, "long")
    fine = [Bar(T0 + 60 + i * 5, 90.0 + i, 90.0 + i, 90.0 + i, 90.0 + i)
            for i in range(24)]
    t = sim.walk([s], fine, tick=0.25)[0]
    assert t.entry_ts == T0 + 120
    assert t.entry == pytest.approx(102.0)


def test_a_signal_with_no_bar_after_it_is_not_traded():
    s = sig(T0 + 600, "long")
    assert sim.walk([s], path(T0, [100.0] * 5), tick=0.25) == []


# ------------------------------------------------------ one at a time

def test_a_second_signal_inside_the_exit_bar_is_skipped():
    """'TP disappears then we wait for the next 15 minute candle.'"""
    a = sig(T0 + 60, "long", 10, 10)
    b = sig(T0 + 120, "long", 10, 10)         # same 15m candle
    fine = [Bar(T0 + 120, 100.0, 100.0, 100.0, 100.0),
            Bar(T0 + 125, 100.0, 102.6, 100.0, 102.5)]
    fine += path(T0 + 130, [102.5] * 1200)
    out = sim.walk([a, b], fine, tick=0.25)
    assert len(out) == 1


def test_a_signal_in_the_next_15m_candle_is_taken():
    a = sig(T0 + 60, "long", 10, 10)
    b = sig(T0 + M15_SECONDS + 60, "long", 10, 10)
    fine = [Bar(T0 + 120, 100.0, 100.0, 100.0, 100.0),
            Bar(T0 + 125, 100.0, 102.6, 100.0, 102.5)]
    fine += path(T0 + 130, [102.5] * 1800)
    assert len(sim.walk([a, b], fine, tick=0.25)) == 2


def test_the_wait_is_measured_from_the_exit_not_the_entry():
    """A trade that runs for an hour blocks that hour. Re-arming from the
    entry's candle would let a position open while another is still on."""
    a = sig(T0 + 60, "long", 10, 400)          # a stop far away: runs long
    b = sig(T0 + M15_SECONDS + 60, "long", 10, 10)
    fine = path(T0 + 120, [100.0] * 2400)
    fine += path(T0 + 120 + 2400, [100.0 + i * 0.05 for i in range(200)])
    out = sim.walk([a, b], fine, tick=0.25, hold_minutes=240)
    assert len(out) == 1, "a second trade opened while the first was on"


def test_trades_never_overlap():
    sigs = [sig(T0 + 60 + i * 60, "long", 10, 10) for i in range(40)]
    fine = path(T0 + 120, [100.0 + (i % 40) * 0.05 for i in range(4000)])
    out = sim.walk(sigs, fine, tick=0.25)
    for a, b in zip(out, out[1:]):
        assert b.entry_ts > a.exit_ts


# --------------------------------------------------------- the control

def _coin(n=900, seed=3):
    """A pure random walk and signals scattered through it."""
    import random
    rng = random.Random(seed)
    px = 100.0
    rows = []
    for i in range(n * 60):
        px = round(px + rng.choice((-0.25, 0.25)), 4)
        rows.append(Bar(T0 + i, px, px + 0.25, px - 0.25, px))
    sigs = [sig(T0 + i * 900 + 60, "long" if i % 2 else "short", 10, 20)
            for i in range(1, n // 20)]
    return sigs, rows


def test_a_tight_target_beats_a_wide_stop_on_a_pure_coin():
    """The number that makes an unconditioned win rate meaningless: with
    no edge at all, 10 ticks against 20 wins about two thirds of the
    time. Any report that omits this is describing the barriers."""
    sigs, rows = _coin()
    s = sim.summarise(sim.walk(sigs, rows, tick=0.25))
    assert s["resolved"] > 20
    assert s["win_rate"] > 0.5
    assert s["driftless"] == pytest.approx(0.667, abs=0.02)


def test_the_coin_does_not_beat_its_own_shuffled_control():
    sigs, rows = _coin()
    obs = sim.summarise(sim.walk(sigs, rows, tick=0.25))
    ctrl = sim.control(sigs, rows, runs=40, tick=0.25)
    assert ctrl["runs"] > 0
    assert not sim.beats(obs, ctrl)


def test_the_control_holds_the_barrier_widths_fixed():
    """Shuffling the SIDE while re-deriving the stop from the new
    direction would change the geometry, and then the control would be
    measuring the geometry rather than the call."""
    import inspect
    src = inspect.getsource(sim.walk)
    assert "sig.target_ticks" in src and "sig.stop_ticks" in src
    assert "sides[n]" in src


def test_the_control_varies_only_the_direction():
    sigs, rows = _coin()
    flip = [s.side for s in sigs]
    same = sim.walk(sigs, rows, tick=0.25, sides=flip)
    plain = sim.walk(sigs, rows, tick=0.25)
    assert [t.entry for t in same] == [t.entry for t in plain]
    assert [t.outcome for t in same] == [t.outcome for t in plain]


def test_beats_needs_a_real_margin():
    obs = {"win_rate": 0.70, "resolved": 100}
    ctrl = {"runs": 50, "win_rate_mean": 0.68, "win_rate_sd": 0.03}
    assert not sim.beats(obs, ctrl)
    assert sim.beats({"win_rate": 0.78, "resolved": 100}, ctrl)


# ------------------------------------------------------------- scoring

def test_cost_is_charged_on_every_resolved_trade():
    sigs, rows = _coin()
    trades = sim.walk(sigs, rows, tick=0.25)
    s = sim.summarise(trades)
    assert s["gross_usd"] - s["net_usd"] == pytest.approx(
        s["resolved"] * sim.COST_RT)


def test_the_cost_includes_the_spread_not_just_commission():
    """Charging $5 when a taker also gives up $12.50 understates the bar
    by more than a tick, which is most of a 10-tick target."""
    assert sim.COST_RT == pytest.approx(17.50)


def test_unresolved_trades_are_counted_apart_not_as_losses():
    """Folding them into either column invents an outcome the tape did
    not produce."""
    s = sig(T0 + 60, "long", 10, 10)
    fine = path(T0 + 120, [100.0] * 50)
    out = sim.walk([s], fine, tick=0.25, hold_minutes=2)
    assert out[0].outcome == sim.UNRESOLVED
    got = sim.summarise(out)
    assert got["unresolved"] == 1
    assert got["wins"] == 0 and got["losses"] == 0 and got["resolved"] == 0


def test_the_breakeven_rate_is_above_the_driftless_one():
    sigs, rows = _coin()
    s = sim.summarise(sim.walk(sigs, rows, tick=0.25))
    assert s["breakeven"] > s["driftless"]


def test_an_empty_run_summarises_to_zero_not_a_crash():
    s = sim.summarise([])
    assert s["trades"] == 0 and s["win_rate"] == 0.0


def test_the_interval_is_reported_with_the_rate():
    """A rate without one invites reading 68% off 30 trades as a fact."""
    sigs, rows = _coin()
    s = sim.summarise(sim.walk(sigs, rows, tick=0.25))
    assert s["ci_low"] < s["win_rate"] < s["ci_high"]


# ---------------------------------------------------------- the splits

def test_split_groups_and_summarises():
    sigs, rows = _coin()
    trades = sim.walk(sigs, rows, tick=0.25)
    by = sim.split(trades, lambda t: t.side)
    assert set(by) <= {"long", "short"}
    assert sum(v["trades"] for v in by.values()) == len(trades)


def test_split_warns_about_counting_the_slices():
    doc = sim.split.__doc__ or ""
    assert "34%" in doc or "multiple" in doc.lower()


def test_the_later_status_rides_along_for_the_provisional_question():
    """Whether waiting for the 15m close would have helped is the one
    question the provisional marker exists to answer."""
    s = sig(T0 + 60, "long", 10, 10, status="withdrawn")
    fine = [Bar(T0 + 120, 100.0, 100.0, 100.0, 100.0),
            Bar(T0 + 125, 100.0, 102.6, 100.0, 102.5)]
    assert sim.walk([s], fine, tick=0.25)[0].later == "withdrawn"


def test_nothing_here_places_an_order():
    import inspect
    src = inspect.getsource(sim).lower()
    for word in ("place_order", "submit", "api_key", "private_key"):
        assert word not in src


# ------------------------------------------------- the end-to-end guard

def _walk_tape(days=8, seed=5, drift=0.0):
    """A random walk at ES's tick, long enough to produce real signals."""
    import random
    from liqmap import bars as bl
    rng = random.Random(seed)
    start = 1_735_000_000 - (1_735_000_000 % 86400)
    rows, px = [], 6000.0
    for i in range(int(days * 23 * 3600 / 5)):
        o = px
        px = round((px + rng.gauss(drift, 0.9)) * 4) / 4
        rows.append(bl.Bar(ts=start + i * 5, open=o,
                           high=max(o, px) + 0.25 * rng.randint(0, 3),
                           low=min(o, px) - 0.25 * rng.randint(0, 3),
                           close=px, volume=100.0))
    return rows


def _signals_for(rows):
    from liqmap import bars as bl, mtf
    from liqmap.levels import trade_date
    days: dict = {}
    for b in rows:
        days.setdefault(trade_date(b.ts), []).append(b)
    out = []
    for d in sorted(days):
        m = bl.resample(days[d], 60.0)
        if len(m) >= 120:
            out.extend(mtf.scan(m, tick=0.25))
    out.sort(key=lambda s: s.ts)
    return out


def test_the_whole_pipeline_finds_no_edge_on_a_random_walk():
    """THE regression guard, and it has already earned its keep.

    A pure random walk contains nothing to find. If the engine beats its
    own shuffled control here, the backtest is reading the future
    somewhere -- and the first version did, by filling at the first fine
    bar after the signal's stamp, which is inside the very minute whose
    close produced the signal. It scored 98.4% against a 79.8% control.

    This runs the real scan, the real walk and the real control, so it
    catches that class of bug wherever in the chain it is introduced.
    """
    rows = _walk_tape()
    sigs = _signals_for(rows)
    assert len(sigs) > 30, "fixture produced too few signals to judge"
    tape = sim.Tape(rows)
    obs = sim.summarise(sim.walk(sigs, tape, tick=0.25))
    ctrl = sim.control(sigs, tape, runs=25, tick=0.25)
    assert obs["resolved"] > 30
    assert not sim.beats(obs, ctrl), (
        f"found an edge on a coin: {obs['win_rate']:.3f} vs control "
        f"{ctrl['win_rate_mean']:.3f} +/- {ctrl['win_rate_sd']:.3f}")


def test_a_random_walk_still_loses_money():
    """On a coin there is nothing to find, so cost alone decides it."""
    rows = _walk_tape()
    obs = sim.summarise(sim.walk(_signals_for(rows), sim.Tape(rows),
                                 tick=0.25))
    assert obs["booked"] > 30
    assert obs["per_trade_usd"] < 0



# ------------------------------------------- the clock and the stop

def expiring(ts, side="long", target_ticks=10.0, stop_ticks=8.0,
             m15_open=99.0, expires=None):
    from liqmap.mtf import M15_SECONDS
    start = ts - (ts % M15_SECONDS)
    return Signal(ts=ts, side=side, trigger=100.0, target=0.0, stop=m15_open,
                  target_ticks=target_ticks, stop_ticks=stop_ticks,
                  m15_start=start, m15_open=m15_open,
                  expires=expires if expires is not None
                  else start + M15_SECONDS)


def test_a_trade_is_flat_when_its_15m_candle_closes():
    """'No trade ever lasts longer than the 15 minute timeframe.' Neither
    barrier is touched, the clock runs out, and the position is closed at
    whatever price is there."""
    from liqmap.mtf import M15_SECONDS
    s = expiring(T0 + 60)
    end = s.m15_start + M15_SECONDS
    fine = path(T0 + 120, [100.0] * 400, step=5.0)   # dead flat, no touch
    t = sim.walk([s], fine, tick=0.25)[0]
    assert t.outcome == sim.TIME
    assert t.exit_ts <= end


def test_a_time_exit_is_counted_as_neither_a_win_nor_a_loss():
    """It is a third outcome. Folding it into either column invents a
    result the tape did not produce -- but it still costs a round trip
    and still moves the account, so it is booked."""
    s = expiring(T0 + 60)
    fine = path(T0 + 120, [100.0] * 400, step=5.0)
    got = sim.summarise(sim.walk([s], fine, tick=0.25))
    assert got["wins"] == 0 and got["losses"] == 0
    assert got["timed_out"] == 1 and got["booked"] == 1
    assert got["net_usd"] == pytest.approx(-sim.COST_RT)


def test_the_target_still_wins_inside_the_candle():
    s = expiring(T0 + 60)
    fine = [Bar(T0 + 120, 100.0, 100.0, 100.0, 100.0),
            Bar(T0 + 125, 100.0, 102.6, 100.0, 102.5)]
    fine += path(T0 + 130, [102.5] * 200, step=5.0)
    t = sim.walk([s], fine, tick=0.25)[0]
    assert t.outcome == sim.TARGET


def test_the_clock_cannot_be_switched_off_by_accident():
    """`expire=False` exists for comparing against the old model, and the
    default has to be the rule as stated."""
    import inspect
    sig = inspect.signature(sim.walk)
    assert sig.parameters["expire"].default is True


def test_without_the_clock_the_same_trade_runs_on():
    s = expiring(T0 + 60)
    fine = path(T0 + 120, [100.0] * 400, step=5.0)
    t = sim.walk([s], fine, tick=0.25, expire=False)[0]
    assert t.outcome != sim.TIME


# ------------------------------------------------- the breakeven stop

def test_the_stop_moves_to_the_entry_once_the_trade_is_onside():
    """The second reading of the rule: a loser costs only fees."""
    s = expiring(T0 + 60, stop_ticks=8.0)
    fine = [Bar(T0 + 120, 100.0, 100.0, 100.0, 100.0),
            Bar(T0 + 125, 100.0, 101.2, 100.0, 101.2),   # +4.8t: stop moves
            Bar(T0 + 130, 101.2, 101.2, 99.9, 100.0)]    # back to entry
    fine += path(T0 + 135, [100.0] * 100, step=5.0)
    t = sim.walk([s], fine, tick=0.25, breakeven_at=4.0)[0]
    assert t.outcome == sim.STOP
    assert t.exit == pytest.approx(100.0)
    assert t.ticks == pytest.approx(0.0)


def test_the_original_stop_still_applies_before_the_trade_goes_onside():
    s = expiring(T0 + 60, stop_ticks=8.0)
    fine = [Bar(T0 + 120, 100.0, 100.0, 100.0, 100.0),
            Bar(T0 + 125, 100.0, 100.1, 97.9, 98.0)]     # straight down
    fine += path(T0 + 130, [98.0] * 100, step=5.0)
    t = sim.walk([s], fine, tick=0.25, breakeven_at=4.0)[0]
    assert t.outcome == sim.STOP
    assert t.ticks == pytest.approx(-8.0)


def test_a_later_profit_cannot_move_a_stop_that_was_already_hit():
    """Scanning the window in one pass would let a profit at minute ten
    rescue a stop hit at minute two."""
    s = expiring(T0 + 60, stop_ticks=4.0)
    fine = [Bar(T0 + 120, 100.0, 100.0, 100.0, 100.0),
            Bar(T0 + 125, 100.0, 100.0, 98.9, 99.0),      # stop at 99.00
            Bar(T0 + 130, 99.0, 102.0, 99.0, 102.0)]      # then flies up
    fine += path(T0 + 135, [102.0] * 100, step=5.0)
    t = sim.walk([s], fine, tick=0.25, breakeven_at=4.0)[0]
    assert t.outcome == sim.STOP
    assert t.ticks == pytest.approx(-4.0)


def test_the_breakeven_variant_is_off_unless_asked_for():
    import inspect
    assert inspect.signature(sim.walk).parameters[
        "breakeven_at"].default is None


def test_a_moved_stop_is_the_one_in_force_from_then_on():
    """THE breakeven ordering test.

    Price goes onside, the stop moves to the entry, price comes back and
    takes it -- and only AFTERWARDS carries on to where the original stop
    was. The exit is the moved stop, at breakeven. Scanning the window in
    one pass against the original stop reports the full loss instead,
    which is a stop that was not in force by the time it was reached.
    """
    s = expiring(T0 + 60, stop_ticks=8.0)       # original stop at 98.00
    fine = [Bar(T0 + 120, 100.0, 100.0, 100.0, 100.0),
            Bar(T0 + 125, 100.0, 101.3, 100.0, 101.3),   # +5.2t, stop moves
            Bar(T0 + 130, 101.3, 101.3, 99.95, 100.0),   # back to entry
            Bar(T0 + 135, 100.0, 100.0, 97.5, 97.6)]     # on past 98.00
    fine += path(T0 + 140, [97.6] * 100, step=5.0)
    t = sim.walk([s], fine, tick=0.25, breakeven_at=4.0)[0]
    assert t.outcome == sim.STOP
    assert t.ticks == pytest.approx(0.0), (
        "exited at the original stop, but it had already moved to the "
        "entry before price got there")


def test_a_scratch_is_not_counted_as_a_loss():
    """A stop hit at the entry did not lose the race, it did not run it.
    Scoring it as a defeat makes the breakeven variant's win rate collapse
    while its money barely moves -- two numbers telling opposite stories
    about the same trades."""
    s = expiring(T0 + 60, stop_ticks=8.0)
    fine = [Bar(T0 + 120, 100.0, 100.0, 100.0, 100.0),
            Bar(T0 + 125, 100.0, 101.3, 100.0, 101.3),
            Bar(T0 + 130, 101.3, 101.3, 99.95, 100.0)]
    fine += path(T0 + 135, [100.0] * 100, step=5.0)
    got = sim.summarise(sim.walk([s], fine, tick=0.25, breakeven_at=4.0))
    assert got["scratched"] == 1
    assert got["losses"] == 0
    assert got["net_usd"] == pytest.approx(-sim.COST_RT)


# ---------------------------------------------------- the trailing stop

def test_the_trail_follows_the_best_price_once_it_is_armed():
    """Chris's rule: three ticks onside, the stop follows two behind."""
    s = expiring(T0 + 60, stop_ticks=20.0)        # original stop far below
    fine = [Bar(T0 + 120, 100.0, 100.0, 100.0, 100.0),
            Bar(T0 + 125, 100.0, 101.0, 100.0, 101.0),   # +4t: armed
            Bar(T0 + 130, 101.0, 101.0, 100.4, 100.4)]   # back to 100.50
    fine += path(T0 + 135, [100.4] * 100, step=5.0)
    t = sim.walk([s], fine, tick=0.25, trail_ticks=2.0, trail_arm=3.0)[0]
    assert t.outcome == sim.STOP
    assert t.exit == pytest.approx(100.5)       # 101.00 minus two ticks
    assert t.ticks == pytest.approx(2.0)        # a WINNING stop


def test_the_trail_does_nothing_before_it_arms():
    s = expiring(T0 + 60, stop_ticks=8.0)
    fine = [Bar(T0 + 120, 100.0, 100.0, 100.0, 100.0),
            Bar(T0 + 125, 100.0, 100.25, 97.9, 98.0)]    # straight to the stop
    fine += path(T0 + 130, [98.0] * 100, step=5.0)
    t = sim.walk([s], fine, tick=0.25, trail_ticks=2.0, trail_arm=3.0)[0]
    assert t.outcome == sim.STOP
    assert t.ticks == pytest.approx(-8.0)


def test_the_trail_lags_by_a_bar_so_a_spike_cannot_rescue_a_low():
    """THE tie rule, in its trailing form.

    One bar holds both the new high that would raise the stop and the low
    that takes it out, and the bar does not record which came first.
    Setting the stop from the CURRENT bar's own high assumes the
    favourable order and hands the tester free money on every spike.

    The arming bar is kept small here on purpose: a bar that both arms the
    trail and ranges wider than the trail distance stops itself either
    way, which made an earlier version of this test pass against the bug.
    """
    # A far target, so this test is about the trail and nothing else.
    s = expiring(T0 + 60, target_ticks=60.0, stop_ticks=20.0)
    fine = [Bar(T0 + 120, 100.0, 100.0, 100.0, 100.0),      # entry 100.00
            Bar(T0 + 125, 100.0, 100.6, 100.0, 100.6),      # not yet armed
            Bar(T0 + 130, 100.6, 101.0, 100.55, 101.0),     # arms
            # Spikes to 103 and trades down to 100.30 in one bar. Reading
            # its own high the stop would be 102.50 and this a ten tick
            # win; lagged, the stop is 100.50 and it is a two tick one.
            Bar(T0 + 135, 101.0, 103.0, 100.3, 100.3)]
    fine += path(T0 + 140, [100.3] * 100, step=5.0)
    t = sim.walk([s], fine, tick=0.25, trail_ticks=2.0, trail_arm=3.0)[0]
    assert t.outcome == sim.STOP
    assert t.exit == pytest.approx(100.5), (
        "the stop was raised by the same bar's own high")
    assert t.ticks == pytest.approx(2.0)


def test_the_trail_never_moves_the_stop_backwards():
    """A running maximum, not the last bar's high. A pullback must not
    loosen a stop that has already tightened -- and the gap has to be
    wider than the trail for the two to differ at all, which is why this
    uses an eight tick trail rather than two."""
    s = expiring(T0 + 60, target_ticks=60.0, stop_ticks=40.0)
    fine = [Bar(T0 + 120, 100.0, 100.0, 100.0, 100.0),      # entry 100.00
            Bar(T0 + 125, 100.0, 103.0, 100.0, 103.0),      # best 103.00
            Bar(T0 + 130, 103.0, 103.0, 102.5, 102.5),      # stop -> 101.00
            Bar(T0 + 135, 102.5, 102.6, 101.2, 101.2),      # high falls back
            # Running max keeps the stop at 101.00 and this takes it.
            # The last bar's high alone would have loosened it to 100.60.
            Bar(T0 + 140, 101.2, 101.3, 100.9, 100.9)]
    fine += path(T0 + 145, [100.9] * 100, step=5.0)
    t = sim.walk([s], fine, tick=0.25, trail_ticks=8.0, trail_arm=3.0)[0]
    assert t.outcome == sim.STOP
    assert t.exit == pytest.approx(101.0), (
        "a pullback loosened a stop that had already tightened")


def test_the_trail_still_lets_the_target_win():
    s = expiring(T0 + 60, target_ticks=10.0, stop_ticks=20.0)
    fine = [Bar(T0 + 120, 100.0, 100.0, 100.0, 100.0),
            Bar(T0 + 125, 100.0, 102.6, 100.0, 102.5)]
    fine += path(T0 + 130, [102.5] * 100, step=5.0)
    t = sim.walk([s], fine, tick=0.25, trail_ticks=2.0, trail_arm=3.0)[0]
    assert t.outcome == sim.TARGET


def test_the_short_side_trails_the_other_way():
    s = expiring(T0 + 60, side="short", stop_ticks=20.0, m15_open=105.0)
    fine = [Bar(T0 + 120, 100.0, 100.0, 100.0, 100.0),
            Bar(T0 + 125, 100.0, 100.0, 99.0, 99.0),      # -4t: armed
            Bar(T0 + 130, 99.0, 99.6, 99.0, 99.6)]        # back to 99.50
    fine += path(T0 + 135, [99.6] * 100, step=5.0)
    t = sim.walk([s], fine, tick=0.25, trail_ticks=2.0, trail_arm=3.0)[0]
    assert t.outcome == sim.STOP
    assert t.exit == pytest.approx(99.5)
    assert t.ticks == pytest.approx(2.0)


def test_the_trail_is_off_unless_asked_for():
    import inspect
    assert inspect.signature(sim.walk).parameters[
        "trail_ticks"].default is None
