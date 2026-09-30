"""The out-of-sample split, against feeds whose truth is known.

This script exists to stop a 2% effect found by looking at the answer from
being mistaken for a finding. So the thing worth testing is not that it
computes a rate -- it is that it says YES to a planted edge and NO to a
coin, using only the half of the data that chose nothing.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from liqmap import escondense as ec                          # noqa: E402

es_combos = pytest.importorskip("es_combos")


def feed(book_edge: float, delta_edge: float, seed: int = 7,
         n: int = 3000) -> list[ec.Slice]:
    """Book predicts WITH its sign; delta predicts AGAINST its sign.

    The shape the ES runs suggested: resting size leads, aggression fades.
    """
    rng = random.Random(seed)
    t0 = 1_789_999_800.0
    out: list[ec.Slice] = []
    px = 7000.0
    for c in range(n):
        up = rng.random() < 0.5
        lean_up = up if rng.random() < 0.5 + book_edge else not up
        d_up = ((not up) if rng.random() < 0.5 + delta_edge else up)
        bsz, asz = (90, 10) if lean_up else (10, 90)

        s = ec.Slice(ts=t0 + c * 60,
                     bids=[(px - i * 0.25, bsz) for i in range(10)],
                     asks=[(px + 0.25 + i * 0.25, asz) for i in range(10)],
                     delta=(30 if d_up else -30), buy_vol=50, sell_vol=50,
                     trades_in=10, prior_close=px, prior_open=px)
        s.open = px
        move = (0.25 if up else -0.25) * rng.randint(1, 6)
        s.close = px + move
        s.high, s.low = max(px, s.close), min(px, s.close)
        s.volume, s.trades_out = 100.0, 20
        step = (s.close - px) / 6.0
        s.path = [(((step * (k + 1)) / px) * 1e4,
                   ((step * (k + 1)) / px) * 1e4) for k in range(6)]
        px = s.close
        out.append(s)
    return out


def split(slices):
    """The same chronological, day-aligned cut the script makes."""
    cut = int(len(slices) * es_combos.SPLIT)
    day = 86_400.0
    d = int(slices[cut].ts // day)
    while cut < len(slices) - 1 and int(slices[cut].ts // day) == d:
        cut += 1
    return slices[:cut], slices[cut:]


def chosen(early):
    signs = {}
    for name in ("book", "delta"):
        d = es_combos.score(es_combos.rows_for(early, {name: 1.0}))
        signs[name] = 1.0 if d["rate"] >= 0.5 else -1.0
    return signs


# ------------------------------------------------------------ the split

def test_the_split_is_chronological_not_random():
    """Shuffling days would let a day's afternoon inform its own morning."""
    s = feed(0.1, 0.1)
    early, late = split(s)
    assert max(x.ts for x in early) < min(x.ts for x in late)


def test_no_day_straddles_the_cut():
    s = feed(0.1, 0.1)
    early, late = split(s)
    day = 86_400.0
    assert int(early[-1].ts // day) != int(late[0].ts // day)


def test_both_halves_are_substantial():
    early, late = split(feed(0.1, 0.1))
    assert len(early) > 500 and len(late) > 300


# ------------------------------------------------- it finds a real edge

def test_a_planted_edge_survives_out_of_sample():
    early, late = split(feed(book_edge=0.10, delta_edge=0.10))
    signs = chosen(early)
    d = es_combos.score(es_combos.rows_for(late, {"book": signs["book"]}))
    assert d["real"] and d["rate"] > 0.55


def test_an_inverted_column_is_identified_as_inverted():
    """Delta predicts against its sign here. Calling that 'no edge' would
    throw away information that is every bit as usable as a positive one."""
    early, _ = split(feed(book_edge=0.10, delta_edge=0.10))
    assert chosen(early)["delta"] == -1.0
    assert chosen(early)["book"] == 1.0


def test_reach_clears_the_breakeven_when_the_edge_is_real():
    """A deliberately generous edge.

    At book_edge=0.10 the out-of-sample interval lands at 56.2% against a
    56.7% requirement -- a correct 'unclear', but a test balanced on that
    boundary flips with any harmless change to the fixture. The point here
    is that a real edge CAN clear, so the edge is planted well clear of it.
    """
    from liqmap import project as pj
    early, late = split(feed(book_edge=0.18, delta_edge=0.10))
    rows = es_combos.rows_for(late, {"book": chosen(early)["book"]})
    tick_bps = 0.25 / 7000.0 * 10_000.0
    table = pj.reach_test(rows, [3 * tick_bps])
    assert table[0]["ci_low"] > es_combos.needs(3)


# -------------------------------------------- it does not invent one

def test_a_coin_is_reported_as_a_coin_out_of_sample():
    """The failure that matters. The early half will always pick a sign for
    every column -- on a coin that sign is noise, and the late half has to
    refuse to confirm it."""
    early, late = split(feed(book_edge=0.0, delta_edge=0.0, seed=11))
    signs = chosen(early)
    d = es_combos.score(es_combos.rows_for(late, {"book": signs["book"]}))
    assert not d["real"]
    assert 0.44 < d["rate"] < 0.56


def test_a_coin_does_not_clear_the_breakeven_at_any_target():
    from liqmap import project as pj
    early, late = split(feed(0.0, 0.0, seed=11))
    rows = es_combos.rows_for(late, {"book": chosen(early)["book"]})
    tick_bps = 0.25 / 7000.0 * 10_000.0
    for ticks in (2, 3, 4):
        row = pj.reach_test(rows, [ticks * tick_bps])[0]
        if row.get("ready"):
            assert row["ci_low"] <= es_combos.needs(ticks)


def test_the_early_half_picks_a_sign_even_on_a_coin():
    """Documenting why the split is necessary rather than optional: the
    choosing step never returns 'no opinion', so its output is not
    evidence of anything on its own."""
    early, _ = split(feed(0.0, 0.0, seed=11))
    signs = chosen(early)
    assert set(signs.values()) <= {1.0, -1.0}


# ----------------------------------------------------------- plumbing

def test_breakeven_matches_the_dollar_arithmetic():
    # 3 ticks: win $37.50-$5, lose $37.50+$5 -> 42.5/75.
    assert es_combos.needs(3) == pytest.approx(42.5 / 75.0)
    assert es_combos.needs(3) == pytest.approx(0.5667, abs=1e-4)


def test_a_bigger_target_needs_a_lower_hit_rate():
    rates = [es_combos.needs(t) for t in (2, 3, 4, 6, 8)]
    assert rates == sorted(rates, reverse=True)


def test_scoring_refuses_a_sample_too_small_to_mean_anything():
    assert not es_combos.score([])["ready"]


def test_a_flat_signal_contributes_no_rows():
    """Weighting a column at zero must leave nothing to score, not a pile
    of coin flips on net == 0."""
    rows = es_combos.rows_for(feed(0.1, 0.1)[:200], {"book": 0.0})
    assert not es_combos.score(rows)["ready"]
