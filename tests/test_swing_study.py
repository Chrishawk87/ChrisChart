"""The swing study: a causal pivot detector and one scorer over four arms.

Five things would turn this from a measurement into a flattering story.

A PIVOT IS NOT KNOWN AT THE PIVOT. The extreme becomes a swing only once
price has given back the reversal distance. If the detector emits at the
extreme, or if anything is scored from `pivot_ts` rather than `ts`, the
study reads the turn at the turn -- which is the family of bug that scored
98.4% on a pure random walk in this project.

A FILTER MAY ONLY REMOVE. Every arm except the two blind ones is a subset
of a base call. If a filter can change the side, the comparison is between
two different strategies and "does the book add value" has no meaning.

NO RETRACEMENT AND NO CUT MAY BE CHOSEN. Both are continuous. Printing the
grid is a measurement; keeping the row that looks best is fitting.

A THIN ARM IS A COUNT. The filters cut hard, and a percentage over nine
readings reads as a finding.

AND UNRESOLVED IS NOT A LOSS. A call whose levels were never reached says
nothing about the call.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load():
    path = ROOT / "scripts" / "swing_study.py"
    spec = importlib.util.spec_from_file_location("swing_study", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["swing_study"] = mod
    spec.loader.exec_module(mod)
    return mod


ss = _load()
T = ss.TICK


# ------------------------------------------- a pivot is not known at the pivot

def test_the_extreme_alone_emits_nothing():
    """Price making a new high is not a swing high. It is a new high."""
    sw = ss.Swinger(2)
    out = [sw.feed(float(i), 100.0 + i * T) for i in range(10)]
    assert all(o is None for o in out)


def test_the_pivot_arrives_only_after_the_giveback_is_complete():
    sw = ss.Swinger(2)
    for i in range(5):                       # up to 101.00 at ts=4
        assert sw.feed(float(i), 100.0 + i * T) is None
    assert sw.feed(5.0, 100.75) is None      # one tick back: not yet
    leg = sw.feed(6.0, 100.50)               # two ticks back: now
    assert leg is not None
    assert leg.kind == "high"
    assert leg.px == pytest.approx(101.0)


def test_the_pivot_is_dated_at_the_extreme_but_confirmed_later():
    """Both timestamps must survive, because the study needs the gap."""
    sw = ss.Swinger(2)
    for i in range(5):
        sw.feed(float(i), 100.0 + i * T)
    sw.feed(5.0, 100.75)
    leg = sw.feed(6.0, 100.50)
    assert leg.ts == 4.0                     # when the high printed
    # and the caller is handed it at 6.0 -- the feed timestamp, not leg.ts


def test_a_deeper_reversal_needs_more_giveback():
    for rev, when in ((2, 100.50), (4, 100.0)):
        sw = ss.Swinger(rev)
        for i in range(5):
            sw.feed(float(i), 100.0 + i * T)
        got = None
        for j, px in enumerate((100.75, 100.50, 100.25, 100.0)):
            got = got or sw.feed(5.0 + j, px)
        assert got is not None and got.px == pytest.approx(101.0)
        sw2 = ss.Swinger(rev)
        for i in range(5):
            sw2.feed(float(i), 100.0 + i * T)
        assert sw2.feed(5.0, when + 0.25) is None or rev == 2


def test_a_low_is_the_mirror_of_a_high():
    sw = ss.Swinger(2)
    for i in range(5):
        sw.feed(float(i), 100.0 - i * T)
    assert sw.feed(5.0, 99.25) is None
    leg = sw.feed(6.0, 99.50)
    assert leg is not None and leg.kind == "low"
    assert leg.px == pytest.approx(99.0)


def test_the_leg_spans_pivot_to_pivot_not_confirmation_to_confirmation():
    sw = ss.Swinger(2)
    for i in range(5):
        sw.feed(float(i), 100.0 + i * T)
    for j, px in enumerate((100.75, 100.50)):
        leg = sw.feed(5.0 + j, px)
    assert leg.start_ts == 0.0 and leg.start_px == pytest.approx(100.0)
    assert leg.ticks == pytest.approx(4.0)


def test_the_next_leg_begins_at_the_pivot_not_at_the_confirming_print():
    """The quiet one. Starting the next leg where the retracement finished
    shortens every leg after the first by the give-back distance, so every
    magnitude, velocity and expansion in the study is understated and the
    expansion ratio in particular becomes nonsense."""
    sw = ss.Swinger(2)
    for i in range(5):                        # 100.00 -> 101.00
        sw.feed(float(i), 100.0 + i * T)
    sw.feed(5.0, 100.75)
    first = sw.feed(6.0, 100.50)              # swing HIGH at 101.00
    assert first is not None
    # now run down to 99.00 and give two ticks back
    for j, px in enumerate((100.25, 100.0, 99.75, 99.5, 99.25, 99.0)):
        assert sw.feed(7.0 + j, px) is None
    second = sw.feed(13.0, 99.25)
    assert second is None                     # one tick back
    second = sw.feed(14.0, 99.50)
    assert second is not None and second.kind == "low"
    # The down leg ran from the HIGH (101.00), not from 100.50 where the
    # high was confirmed.
    assert second.start_px == pytest.approx(101.0)
    assert second.start_ts == 4.0
    assert second.ticks == pytest.approx(8.0)


# --------------------------------------------------- outcomes resolve forward

def tape_of(pairs):
    t = ss.Tape()
    for ts, px in pairs:
        t.add(ts, px)
    return t


def test_prints_before_the_confirmation_cannot_resolve_it():
    """Look-ahead in reverse: a level touched before the call is not the
    call being right."""
    t = tape_of([(1.0, 105.0), (2.0, 100.0)])
    assert t.first_touch(1.5, 101.25, 98.75, True) is None


def test_the_target_coming_first_is_a_win_and_the_stop_a_loss():
    assert tape_of([(1.0, 100.0), (2.0, 101.5)]).first_touch(
        0.5, 101.25, 98.75, True) is True
    assert tape_of([(1.0, 100.0), (2.0, 98.5)]).first_touch(
        0.5, 101.25, 98.75, True) is False


def test_a_short_is_scored_the_other_way_round():
    assert tape_of([(1.0, 100.0), (2.0, 98.5)]).first_touch(
        0.5, 98.75, 101.25, False) is True
    assert tape_of([(1.0, 100.0), (2.0, 101.5)]).first_touch(
        0.5, 98.75, 101.25, False) is False


def test_nothing_resolves_past_the_horizon():
    t = tape_of([(1.0, 100.0), (ss.HORIZON_S + 50.0, 110.0)])
    assert t.first_touch(0.5, 101.25, 98.75, True) is None


def test_a_call_that_never_reached_either_level_is_none_not_a_loss():
    t = tape_of([(1.0, 100.0), (2.0, 100.25)])
    assert t.first_touch(0.5, 101.25, 98.75, True) is None


def test_unresolved_is_counted_apart_from_the_rate():
    assert ss.rate([True, False, None, None]) == (1, 2, 2)


def test_excursion_reports_both_ways_as_positive_ticks():
    t = tape_of([(1.0, 101.0), (2.0, 98.0), (3.0, 100.0)])
    fav, adv = t.excursion(0.5, 100.0, True, 10.0)
    assert fav == pytest.approx(4.0)         # +1.00 is four ticks up
    assert adv == pytest.approx(8.0)         # -2.00 is eight ticks against
    assert fav >= 0 and adv >= 0


def test_excursion_of_a_short_inverts_both():
    t = tape_of([(1.0, 98.0), (2.0, 101.0)])
    fav, adv = t.excursion(0.5, 100.0, False, 10.0)
    assert fav == pytest.approx(8.0)
    assert adv == pytest.approx(4.0)


def test_an_excursion_with_no_prints_ahead_is_none_not_zero():
    """Zero reads as 'it went nowhere'. Nothing to read is a third thing."""
    assert tape_of([(1.0, 100.0)]).excursion(5.0, 100.0, True, 9.0) is None


# ------------------------------------------------- a filter may only remove

def look(kind="high", structure="", delta=0.0, depth=None, ts=0.0,
         bid=99.875, ask=100.125, r_delta=0.0, r_prints=0, p_rate=None):
    return ss.Look(ts=ts, pivot_kind=kind, pivot_px=100.0, pivot_ts=ts - 5,
                   leg_ticks=8.0, leg_s=4.0, velocity=2.0, retrace_pct=0.25,
                   prev_leg_ticks=6.0, structure=structure,
                   failed_break=False, changes_1m=3, delta=delta,
                   efficiency=1.0, imbalance=None, cancel_edge=None,
                   depth_edge=depth, bid=bid, ask=ask, mid=100.0,
                   bar_ts=None, recent_delta=r_delta, recent_prints=r_prints,
                   print_rate=p_rate)


def test_the_two_blind_arms_are_exact_opposites():
    for kind in ("high", "low"):
        l = look(kind=kind)
        assert l.continuation and l.reversal
        assert l.continuation != l.reversal


def test_continuation_trades_the_leg_that_is_starting():
    assert look(kind="high").continuation == "sell"
    assert look(kind="low").continuation == "buy"


def test_neither_blind_arm_consults_a_filter():
    import inspect
    for fn in (ss.Look.continuation.fget, ss.Look.reversal.fget):
        src = inspect.getsource(fn)
        for banned in ("structure", "delta", "depth_edge", "imbalance",
                       "cancel_edge"):
            assert banned not in src, banned


def test_a_filter_never_changes_the_side_only_whether_there_is_one():
    looks = [look(kind="high", structure="LH", delta=-5.0, depth=0.9),
             look(kind="low", structure="HL", delta=5.0, depth=-0.9),
             look(kind="high", structure="HH", delta=5.0, depth=0.1)]
    base = ss.arm_sides(looks, "continuation", "", 0.0)
    for filt in ("structure", "flow", "book", "combined"):
        sub = ss.arm_sides(looks, "continuation", filt, 0.10)
        assert set(sub) <= set(base), filt
        for i, side in sub.items():
            assert side == base[i], filt


def test_the_agreeing_and_disagreeing_halves_are_disjoint_and_both_kept():
    looks = [look(kind="high", structure="LH"), look(kind="high",
                                                     structure="HH")]
    yes = ss.arm_sides(looks, "continuation", "structure", 0.0)
    no = ss.arm_sides(looks, "continuation", "structure_against", 0.0)
    assert set(yes) & set(no) == set()
    assert set(yes) | set(no) == {0, 1}


def test_an_unmeasured_book_is_dropped_from_both_book_halves():
    """Missing is not neutral, and it is certainly not disagreement."""
    looks = [look(kind="high", depth=None)]
    assert ss.arm_sides(looks, "continuation", "book", 0.10) == {}
    assert ss.arm_sides(looks, "continuation", "book_against", 0.10) == {}


def test_a_book_reading_under_its_cut_says_nothing():
    looks = [look(kind="high", depth=0.05)]
    assert ss.arm_sides(looks, "continuation", "book", 0.10) == {}
    assert ss.arm_sides(looks, "continuation", "book_against", 0.10) == {}


def test_structure_agreement_means_a_lower_high_or_a_higher_low():
    assert look(kind="high", structure="LH").structure_agrees()
    assert not look(kind="high", structure="HH").structure_agrees()
    assert look(kind="low", structure="HL").structure_agrees()
    assert not look(kind="low", structure="LL").structure_agrees()
    assert not look(kind="high", structure="").structure_agrees()


def test_flow_agreement_follows_the_sign_of_the_leg_delta():
    assert look(delta=5.0).flow_agrees("buy")
    assert not look(delta=5.0).flow_agrees("sell")
    assert look(delta=-5.0).flow_agrees("sell")
    assert not look(delta=0.0).flow_agrees("buy")


# ----------------------------------------------- the live-tape confirmation

def test_the_tape_confirms_only_when_it_is_signed_and_quick():
    assert look(r_delta=20.0, r_prints=30, p_rate=1.6).tape_agrees("buy")
    assert look(r_delta=-20.0, r_prints=30, p_rate=1.6).tape_agrees("sell")
    assert not look(r_delta=20.0, r_prints=30,
                    p_rate=1.6).tape_agrees("sell")


def test_a_slow_tape_denies_the_call_even_when_it_is_signed_right():
    """A quiet tape that happens to agree is not a strong move, and the
    whole point of the confirmation is that it is about strength."""
    l = look(r_delta=20.0, r_prints=3, p_rate=0.4)
    assert l.tape_agrees("buy") is False


def test_an_unmeasured_tape_is_none_rather_than_a_denial():
    assert look(r_delta=5.0, r_prints=4, p_rate=None).tape_agrees("buy") \
        is None
    assert look(r_delta=5.0, r_prints=0, p_rate=2.0).tape_agrees("buy") \
        is None


def test_the_tape_window_is_its_own_and_not_the_whole_leg():
    """A leg bought the entire way up can be dead by the time its pivot
    confirms. The two measurements must be able to disagree."""
    l = look(delta=500.0, r_delta=-40.0, r_prints=30, p_rate=1.5)
    assert l.flow_agrees("buy")
    assert l.tape_agrees("buy") is False


def test_an_unmeasured_tape_is_dropped_from_both_tape_halves():
    looks = [look(kind="high", p_rate=None)]
    assert ss.arm_sides(looks, "continuation", "tape", 0.0) == {}
    assert ss.arm_sides(looks, "continuation", "tape_against", 0.0) == {}


def test_the_tape_halves_are_disjoint_and_both_kept():
    looks = [look(kind="high", r_delta=-9.0, r_prints=30, p_rate=1.5),
             look(kind="high", r_delta=9.0, r_prints=30, p_rate=1.5)]
    yes = ss.arm_sides(looks, "continuation", "tape", 0.0)
    no = ss.arm_sides(looks, "continuation", "tape_against", 0.0)
    assert set(yes) & set(no) == set()
    assert set(yes) | set(no) == {0, 1}


def test_the_combined_arm_requires_the_tape_as_well():
    """It is the arm that claims every check agreed, so a missing tape
    reading must disqualify it rather than pass silently."""
    base = dict(kind="high", structure="LH", delta=-5.0, depth=-0.9)
    with_tape = look(**base, r_delta=-9.0, r_prints=30, p_rate=1.5)
    no_tape = look(**base, p_rate=None)
    assert ss.arm_sides([with_tape], "continuation", "combined", 0.1)
    assert ss.arm_sides([no_tape], "continuation", "combined", 0.1) == {}


def test_the_walk_measures_the_recent_tape_separately(passthrough):
    looks, _, _ = ss.walk(zigzag([(14, True), (14, False), (14, True)]), 2)
    assert looks
    measured = [l for l in looks if l.print_rate is not None]
    assert measured, "no look carried a tape rate"
    for l in measured:
        assert l.recent_prints > 0


def test_the_tape_baseline_excludes_the_window_it_is_judging(passthrough):
    """Comparing the confirmation window against the leg does not work: a
    leg can be a fraction of a second long, its own print rate is then
    enormous, and nothing is ever fast. A run of such legs produced zero
    eligible readings on the tape arm -- the arm was silently empty, which
    looks the same in a table as an arm that never agreed."""
    # quiet, then a burst: the burst must read as fast
    items = []
    px, t = 7800.0, 0.0
    for _ in range(50):                    # one print a second, quiet
        items.append(("book", book(t, px=px)))
        items.append(("trade", trade(t + 0.1, px)))
        t += 1.0
    for k in range(60):                    # ten a second, rising
        px += 0.25 if k % 4 == 0 else 0.0
        items.append(("book", book(t, px=px)))
        items.append(("trade", trade(t + 0.01, px, side="buy")))
        t += 0.1
    for k in range(10):                    # give back to confirm the high
        px -= 0.25
        items.append(("book", book(t, px=px)))
        items.append(("trade", trade(t + 0.01, px, side="sell")))
        t += 0.1
    looks, _, _ = ss.walk(items, 2)
    rated = [l.print_rate for l in looks if l.print_rate is not None]
    assert rated, "no look carried a tape rate"
    assert max(rated) > 1.0, rated


def test_a_tape_rate_is_none_when_there_is_no_baseline_to_compare_to(
        passthrough):
    """At the very start of a file there is no trailing stretch, so the
    honest answer is that it was not measured -- not that it was slow."""
    items = []
    px, t = 7800.0, 0.0
    for k in range(12):
        px += 0.25
        items.append(("book", book(t, px=px)))
        items.append(("trade", trade(t + 0.01, px)))
        t += 0.05
    for k in range(4):
        px -= 0.25
        items.append(("book", book(t, px=px)))
        items.append(("trade", trade(t + 0.01, px)))
        t += 0.05
    looks, _, _ = ss.walk(items, 2)
    assert looks
    assert all(l.print_rate is None for l in looks)


# ------------------------------------------------------- one scorer, one entry

def test_one_scorer_serves_every_arm():
    import inspect
    src = inspect.getsource(ss)
    assert src.count("def score(") == 1


def test_there_is_exactly_one_tape_walker_and_one_decision_rule():
    """Scoring every cell with its own walk does not finish on a real
    session, so the cells come out of one pass. The danger in that is two
    walkers drifting apart while the tests only ever exercise one, which
    is why `first_touch` and `excursion` delegate rather than duplicate."""
    import inspect
    src = inspect.getsource(ss)
    assert src.count("def crossings(") == 1
    assert src.count("def decide(") == 1
    assert src.count("self._after(") == 1
    for fn in (ss.Tape.first_touch, ss.Tape.excursion):
        body = inspect.getsource(fn)
        assert "self.crossings(" in body
        assert "while" not in body, f"{fn.__name__} walks the tape itself"


def test_the_batched_path_agrees_with_the_single_pair_path():
    """The property that makes the delegation above worth trusting."""
    import random
    rng = random.Random(11)
    for _ in range(120):
        t = ss.Tape()
        px = 100.0
        for k in range(60):
            px += 0.25 * rng.choice((-1, -1, 1, 1, 1))
            t.add(float(k), px)
        long = rng.choice((True, False))
        entry = 100.0
        sgn = 1.0 if long else -1.0
        targets = [entry + sgn * n * ss.TICK for n in ss.TARGETS]
        stops = [entry - sgn * n * ss.TICK for n in ss.STOP_TICKS]
        t_at, s_at, _ = t.crossings(0.0, entry, targets, stops, long, 900.0)
        for a, tt in enumerate(ss.TARGETS):
            for b, sv in enumerate(ss.STOP_TICKS):
                for cap in ss.TIME_STOPS:
                    batched = t.decide(t_at[a], s_at[b], cap)
                    single = t.first_touch(0.0, targets[a], stops[b], long,
                                           cap_s=cap)
                    assert batched == single, (tt, sv, cap, long)


def test_a_level_crossed_later_does_not_overwrite_its_first_crossing():
    t = tape_of([(1.0, 101.0), (2.0, 99.0), (3.0, 102.0)])
    t_at, _, _ = t.crossings(0.0, 100.0, [100.5], [], True, 900.0)
    assert t_at[0] == 1.0


def test_a_level_never_crossed_stays_none():
    t = tape_of([(1.0, 100.0), (2.0, 100.25)])
    t_at, s_at, _ = t.crossings(0.0, 100.0, [110.0], [90.0], True, 900.0)
    assert t_at == [None] and s_at == [None]


def test_the_decision_rule_prefers_whichever_came_first():
    assert ss.Tape.decide(5.0, 9.0, 100.0) is True
    assert ss.Tape.decide(9.0, 5.0, 100.0) is False
    assert ss.Tape.decide(None, 5.0, 100.0) is False
    assert ss.Tape.decide(5.0, None, 100.0) is True
    assert ss.Tape.decide(None, None, 100.0) is None


def test_the_deadline_can_disqualify_a_crossing_that_happened():
    """A win at fifteen minutes is not a win at three."""
    assert ss.Tape.decide(600.0, None, 180.0) is None
    assert ss.Tape.decide(600.0, None, 900.0) is True
    assert ss.Tape.decide(600.0, 700.0, 650.0) is True
    assert ss.Tape.decide(700.0, 600.0, 650.0) is False


def test_entry_crosses_the_spread_for_both_sides():
    import inspect
    src = inspect.getsource(ss.score)
    assert "entry = l.ask if long else l.bid" in src
    assert "l.mid" not in src


def test_the_scorer_resolves_from_the_confirmation_not_the_pivot():
    """The one substitution that would reinstate the 98.4% bug."""
    import inspect
    src = inspect.getsource(ss.score)
    assert "pivot_ts" not in src
    assert "l.ts" in src


def _at(l, side, target, stop, cap):
    return ss.outcome(l, side, ss.TARGETS.index(target),
                      ss.STOP_TICKS.index(stop), cap)


def test_a_long_and_a_short_on_the_same_look_disagree_on_the_outcome():
    looks = [look(kind="high", ts=0.0)]
    t = tape_of([(1.0, 100.0), (2.0, 110.0)])
    ss.score(looks, t)
    assert _at(looks[0], "buy", 16, 4, ss.TIME_STOPS[0]) is True
    assert _at(looks[0], "sell", 16, 4, ss.TIME_STOPS[0]) is False


def test_the_time_stop_can_change_the_outcome_of_the_same_call():
    """A trade that resolves in ten minutes is a win at a 15-minute stop
    and unresolved at a 3-minute one. Collapsing the two hides the hold
    the strategy actually needs."""
    looks = [look(kind="low", ts=0.0)]
    ss.score(looks, tape_of([(1.0, 100.0), (600.0, 110.0)]))
    assert _at(looks[0], "buy", 16, 4, 180.0) is None
    assert _at(looks[0], "buy", 16, 4, 900.0) is True


def test_both_sides_are_walked_so_any_arm_can_read_its_own():
    """A filter never changes the side, but the two base arms are mirrors,
    so between them both sides are used and both must be present."""
    looks = [look(kind="high", ts=0.0)]
    ss.score(looks, tape_of([(1.0, 100.0), (2.0, 101.0)]))
    assert set(looks[0].cross) == {"buy", "sell"}


def test_the_scorer_runs_once_and_every_arm_reads_the_same_measurement():
    """Re-scoring per arm was the cost of the run, and it also lets one
    arm's number drift from another's. Reading must not mutate."""
    looks = [look(kind="high", ts=0.0), look(kind="low", ts=1.0)]
    t = tape_of([(2.0, 100.0), (3.0, 104.0), (4.0, 96.0)])
    ss.score(looks, t)
    before = [dict(l.cross) for l in looks]
    for _ in range(3):
        ss.report("x", looks, t, {0: "buy", 1: "buy"}, 1.0, cap=900.0)
    assert [dict(l.cross) for l in looks] == before


# ------------------------------------------- nothing is chosen on the outcome

def test_the_retracement_and_the_cut_are_grids_not_settings():
    assert len(ss.REVERSALS) >= 3
    assert len(ss.BOOK_CUTS) >= 3


def test_no_selection_vocabulary_survives_in_the_code():
    import ast
    import inspect

    class Blank(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(value=""), node)
            return node

        def visit_JoinedStr(self, node):
            return ast.copy_location(ast.Constant(value=""), node)

    tree = Blank().visit(ast.parse(inspect.getsource(ss)))
    ast.fix_missing_locations(tree)
    code = ast.unparse(tree).lower()
    for word in ("optimal", "recommend", "argmax", "pick_cut", "best_cut",
                 "best_rev", "tune"):
        assert word not in code, word
    assert "max(reversals" not in code
    assert "max(book_cuts" not in code


def test_a_thin_arm_is_a_count_not_a_rate():
    assert ss.MIN_ARM >= 30
    assert "/" in ss.cell([True] * 5)
    assert "%" not in ss.cell([True] * 5)
    assert "%" in ss.cell([True] * ss.MIN_ARM)


def test_the_cell_shows_nothing_when_nothing_resolved():
    assert "—" in ss.cell([None, None])


# --------------------------------------------------------------- the control

def test_the_control_keeps_the_sample_and_varies_only_the_call():
    sides = {i: ("buy" if i % 2 else "sell") for i in range(40)}
    out = ss.shuffled(sides, seed=7)
    assert set(out) == set(sides)
    assert sorted(out.values()) == sorted(sides.values())


def test_the_control_actually_moves_something():
    """A permutation that returns its input is not a control."""
    sides = {i: ("buy" if i < 20 else "sell") for i in range(40)}
    assert ss.shuffled(sides, seed=7) != sides


def test_the_control_is_reproducible_from_its_seed():
    sides = {i: ("buy" if i % 3 else "sell") for i in range(40)}
    assert ss.shuffled(sides, 11) == ss.shuffled(sides, 11)


# ---------------------------------------------------------------- the costs

def test_a_symmetric_geometry_needs_more_than_half():
    assert ss.breakeven(10, 10, 0.0) == pytest.approx(0.5)
    assert ss.breakeven(10, 10, 1.0) == pytest.approx(0.55)
    assert ss.breakeven(5, 5, 1.0) == pytest.approx(0.60)


def test_a_wider_stop_needs_a_much_higher_rate():
    assert ss.breakeven(10, 20, 0.0) == pytest.approx(2 / 3)
    assert ss.breakeven(10, 20, 1.0) > 2 / 3


def test_a_stop_tighter_than_the_target_needs_far_less():
    """The correction that makes every earlier null uninformative about
    this trade: four points against one point needs a quarter, where the
    ten-tick symmetric geometry those studies used needs fifty-five."""
    assert ss.breakeven(16, 4, 1.0) == pytest.approx(0.25)
    assert ss.breakeven(16, 8, 1.0) == pytest.approx(0.375)
    assert ss.breakeven(20, 8, 1.0) == pytest.approx(9 / 28)
    assert ss.breakeven(16, 4, 1.0) < ss.breakeven(10, 10, 1.0)


def test_the_headline_geometry_is_the_trade_being_tested():
    """Four to five points is sixteen to twenty ticks. A study whose
    targets top out at ten is measuring a different trade."""
    assert ss.HEADLINE == ((16, 4), (20, 4))
    for t, s in ss.HEADLINE:
        assert t in ss.TARGETS and s in ss.STOP_TICKS
    assert max(ss.TARGETS) >= 20
    assert min(ss.STOP_TICKS) <= 4


def test_the_time_stop_reaches_a_fifteen_minute_hold():
    assert max(ss.TIME_STOPS) >= 900.0
    assert ss.HORIZON_S >= max(ss.TIME_STOPS)


def test_a_verdict_is_withheld_on_a_thin_arm():
    assert "-" in ss._verdict(5, 9, 0.25)
    assert ss._verdict(60, ss.MIN_ARM + 70, 0.25).strip() in ("over",
                                                              "under")


def test_a_verdict_compares_against_the_geometrys_own_bar():
    n = ss.MIN_ARM + 20
    assert ss._verdict(int(n * 0.30), n, 0.25).strip() == "over"
    assert ss._verdict(int(n * 0.20), n, 0.25).strip() == "under"


def test_costs_are_not_quietly_zero():
    assert ss.DEFAULT_RT_TICKS > 0


def test_the_interval_is_wide_at_small_n_and_tight_at_large():
    lo, hi = ss.wilson(6, 9)
    assert hi - lo > 0.4
    lo2, hi2 = ss.wilson(6000, 9000)
    assert hi2 - lo2 < 0.03


# ------------------------------------------------------------- the walk

def book(ts, px=7800.0, bid_sz=40.0, ask_sz=40.0, n=10):
    from liqmap.flow import Book, Level
    return Book(coin="ES", ts=ts,
                bids=[Level(px=px - 0.25 * (j + 1), sz=bid_sz)
                      for j in range(n)],
                asks=[Level(px=px + 0.25 * (j + 1), sz=ask_sz)
                      for j in range(n)])


def trade(ts, px, side="buy", sz=1.0):
    from liqmap.flow import Trade
    return Trade(px=px, sz=sz, aggressor=side, ts=ts)


@pytest.fixture()
def passthrough(monkeypatch):
    from liqmap import dbn
    monkeypatch.setattr(dbn, "stream",
                        lambda recs, symbol="ES", flip=False, **kw:
                        iter(recs))


def zigzag(legs, px=7800.0, start=0.0):
    """[(ticks, up?)] -> interleaved book pushes and prints."""
    items = []
    t = start
    for ticks, up in legs:
        for _ in range(int(ticks)):
            px += 0.25 if up else -0.25
            items.append(("book", book(t, px=px)))
            items.append(("trade", trade(t + 0.1, px)))
            t += 1.0
    return items


def test_the_walk_keeps_every_print(passthrough):
    items = zigzag([(10, True), (10, False), (10, True)])
    looks, tape, prints = ss.walk(items, 2)
    assert prints == 30 and tape.n == 30


def test_the_walk_finds_the_pivots_a_zigzag_contains(passthrough):
    looks, _, _ = ss.walk(zigzag([(10, True), (10, False), (10, True)]), 2)
    assert len(looks) >= 2
    kinds = [l.pivot_kind for l in looks]
    assert "high" in kinds and "low" in kinds


def test_every_look_is_confirmed_after_its_own_pivot(passthrough):
    """The single most important property in the file."""
    looks, _, _ = ss.walk(zigzag([(12, True), (12, False), (12, True),
                                  (12, False)]), 2)
    assert looks
    for l in looks:
        assert l.ts > l.pivot_ts, f"confirmed at or before its pivot: {l}"


def test_the_bar_slot_is_none_when_unset_rather_than_zero():
    """Zero collides with the first real slot when a file starts at the
    epoch, and the whole bar reads empty for the run."""
    import dataclasses
    f = {f.name: f for f in dataclasses.fields(ss.Look)}["bar_ts"]
    assert f.default is None


def test_a_flat_tape_produces_no_pivots(passthrough):
    items = []
    for s in range(100):
        items.append(("book", book(float(s))))
        items.append(("trade", trade(s + 0.1, 7800.0)))
    looks, _, _ = ss.walk(items, 2)
    assert looks == []


def test_a_pivot_seen_before_any_book_is_skipped_but_not_forgotten(
        passthrough):
    """It cannot be a decision point, but it is still the predecessor the
    next pivot has to be compared against."""
    items = [("trade", trade(float(i), 7800.0 + i * 0.25)) for i in range(10)]
    items += [("trade", trade(10.0 + i, 7802.25 - i * 0.25))
              for i in range(10)]
    items += zigzag([(10, True), (10, False)], px=7800.0, start=30.0)
    looks, _, _ = ss.walk(items, 2)
    assert all(l.bid > 0 and l.ask > 0 for l in looks)


def test_structure_is_read_against_the_previous_pivot_of_the_same_kind(
        passthrough):
    """A label that does not consult its predecessor is a constant, and a
    constant filter either keeps everything or nothing."""
    # up 12, down 6, up 12 (higher high), down 12, up 6 (lower high)
    items = zigzag([(12, True), (6, False), (12, True), (12, False),
                    (6, True), (12, False)])
    looks, _, _ = ss.walk(items, 2)
    highs = [l.structure for l in looks if l.pivot_kind == "high"]
    assert len(highs) >= 3, highs
    assert highs[0] == "", "the first high has no predecessor to compare to"
    assert "HH" in highs and "LH" in highs, highs


def test_a_higher_low_and_a_lower_low_are_both_reachable(passthrough):
    items = zigzag([(12, False), (12, True), (6, False), (12, True),
                    (18, False), (12, True)])
    looks, _, _ = ss.walk(items, 2)
    lows = [l.structure for l in looks if l.pivot_kind == "low"]
    assert len(lows) >= 3, lows
    assert "HL" in lows and "LL" in lows, lows


def test_the_depth_notional_spans_ten_levels(passthrough):
    import inspect
    src = inspect.getsource(ss._imbalance)
    assert "bids[:10]" in src and "asks[:10]" in src


# ------------------------------------------------- no strategy, no production

def test_it_calls_bookread_and_dom_as_written():
    import inspect
    src = inspect.getsource(ss)
    assert "BookReader()" in src
    assert "DomWatch(" in src
    for banned in (".W =", "FLAT_BAND =", "MIN_OVERLAP ="):
        assert banned not in src, banned


def test_no_production_module_is_mutated():
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(ss))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute):
                    assert getattr(t.value, "id", "") not in (
                        "dbn", "br", "dom"), ast.dump(t)


def test_nothing_here_places_an_order():
    import inspect
    src = inspect.getsource(ss).lower()
    for word in ("def submit", "def execute", "def place", "def order"):
        assert word not in src


# ------------------------------------------------------------- progress

def test_progress_never_writes_to_the_report():
    """It goes to stderr. On stdout it would land in the middle of the
    tables and in whatever file the run was piped into."""
    import io
    buf = io.StringIO()
    p = ss.Progress("x", every=2, out=buf)
    for _ in range(4):
        p.tick()
    p.done()
    assert buf.getvalue()
    assert p.out is not sys.stdout


def test_progress_defaults_to_stderr():
    assert ss.Progress("x").out is sys.stderr


def test_progress_prints_on_the_interval_and_not_between():
    import io
    buf = io.StringIO()
    p = ss.Progress("x", every=3, out=buf)
    p.tick(); p.tick()
    assert buf.getvalue() == ""
    p.tick()
    assert "3" in buf.getvalue()


def test_progress_survives_a_zero_interval():
    import io
    p = ss.Progress("x", every=0, out=io.StringIO())
    p.tick()          # must not divide by zero


def test_progress_reports_a_share_only_when_a_total_is_known():
    import io
    a, b = io.StringIO(), io.StringIO()
    p = ss.Progress("x", every=1, total=10, out=a)
    p.tick()
    q = ss.Progress("x", every=1, out=b)
    q.tick()
    assert "%" in a.getvalue()
    assert "%" not in b.getvalue()


def test_the_walk_and_the_scorer_run_without_a_progress_object(passthrough):
    """It is optional, and the tests drive them without one."""
    looks, tape, _ = ss.walk(zigzag([(12, True), (12, False)]), 2)
    ss.score(looks, tape)
    assert all(l.cross for l in looks)


def test_the_walk_reports_the_pivot_count_as_it_goes(passthrough):
    """The number worth seeing early: a threshold producing a pivot every
    few prints will not finish, and that should be visible in seconds
    rather than after the pass."""
    import io
    buf = io.StringIO()
    p = ss.Progress("w", every=10, out=buf)
    ss.walk(zigzag([(12, True), (12, False), (12, True)]), 2, progress=p)
    assert "pivots" in buf.getvalue()


def test_a_limit_takes_the_first_records_and_no_limit_takes_them_all():
    assert list(ss.capped(range(10), 3)) == [0, 1, 2]
    assert list(ss.capped(range(4), 0)) == [0, 1, 2, 3]
    assert list(ss.capped(range(4), -1)) == [0, 1, 2, 3]


def test_a_limit_longer_than_the_file_is_not_an_error():
    assert list(ss.capped(range(3), 99)) == [0, 1, 2]
