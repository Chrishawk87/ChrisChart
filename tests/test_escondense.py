"""The condenser, and the look-ahead line it must not cross.

The central test here is `test_the_outcome_does_not_leak_into_the_decision`.
Every row this module produces carries both a decision input and the outcome
that followed, which is the exact shape that leaks the future if it is built
carelessly -- and a leak does not crash, it just makes the study come back
with a wonderful result that does not exist. So one fixture is built with a
quiet first half and an explosive second half: if any of the second half
reaches the decision fields, the numbers move and the test fails.
"""

from __future__ import annotations

import json

import pytest

from liqmap import escondense as ec
from liqmap.escondense import Slice, condense, grid


def lvl(bid_px=None, bid_sz=0, ask_px=None, ask_sz=0):
    from liqmap.dbn import UNDEF_PRICE
    return {"bid_px": UNDEF_PRICE if bid_px is None else int(bid_px * 1e9),
            "bid_sz": bid_sz, "bid_ct": 0,
            "ask_px": UNDEF_PRICE if ask_px is None else int(ask_px * 1e9),
            "ask_sz": ask_sz, "ask_ct": 0}


def book(bid=7000.00, ask=7000.25, bsz=50, asz=50, depth=10):
    return [lvl(bid_px=bid - i * 0.25, bid_sz=bsz,
                ask_px=ask + i * 0.25, ask_sz=asz) for i in range(depth)]


def bookrec(ts, bid=7000.00, ask=7000.25, bsz=50, asz=50):
    return {"action": "A", "side": "B", "price": 0, "size": 0,
            "ts_event": int(ts * 1e9), "levels": book(bid, ask, bsz, asz)}


def traderec(ts, px, sz=1, buy=True, levels=True):
    return {"action": "T", "side": "B" if buy else "A",
            "price": int(px * 1e9), "size": sz,
            "ts_event": int(ts * 1e9),
            "levels": book() if levels else []}


T0 = 1_789_999_800.0        # lands on a 300s boundary
assert T0 % 300 == 0


# --------------------------------------------------------------- basics

def test_the_grid_floors_onto_the_interval():
    assert grid(T0 + 137, 300.0) == T0
    assert grid(T0 + 300, 300.0) == T0 + 300


def _one_candle():
    """A quiet candle, then a record past its end to close it out."""
    recs = [bookrec(T0 - 1, bid=7000.00, ask=7000.25, bsz=80, asz=20)]
    for i, px in enumerate([7000.25, 7000.50, 7000.00, 7000.75]):
        recs.append(traderec(T0 + 10 + i * 60, px, sz=2, buy=i % 2 == 0))
    recs.append(traderec(T0 + 305, 7001.0))
    return recs


def test_one_candle_comes_out_of_one_candle_of_tape():
    out = list(condense(_one_candle()))
    assert len(out) == 1
    assert out[0].ts == T0


def test_the_candle_is_built_from_the_trades_that_followed():
    s = list(condense(_one_candle()))[0]
    assert s.open == pytest.approx(7000.25)
    assert s.high == pytest.approx(7000.75)
    assert s.low == pytest.approx(7000.00)
    assert s.close == pytest.approx(7000.75)
    assert s.volume == pytest.approx(8)
    assert s.trades_out == 4


def test_the_book_is_the_one_that_stood_at_the_open():
    s = list(condense(_one_candle()))[0]
    assert s.bids[0] == (pytest.approx(7000.00), 80)
    assert s.asks[0] == (pytest.approx(7000.25), 20)
    assert len(s.bids) == 10 and len(s.asks) == 10


def test_imbalance_reads_off_the_resting_size():
    s = list(condense(_one_candle()))[0]
    # 80 bid against 20 ask at every level -> (400-100)/500.
    assert s.imbalance(levels=5) == pytest.approx(0.6)


def test_imbalance_of_an_empty_book_is_flat_not_a_division_by_zero():
    assert Slice(ts=T0).imbalance() == 0.0


# ------------------------------------------------- THE look-ahead test

def test_the_outcome_does_not_leak_into_the_decision():
    """A quiet run-up, then a violent candle.

    Every decision field on the violent candle must be computed from the
    quiet half alone. If a single record from the explosion reaches the
    book, the delta or the prior close, these numbers move.
    """
    recs = [bookrec(T0 - 200, bid=7000.00, ask=7000.25, bsz=30, asz=30)]

    # Quiet: ten small sells before the open.
    for i in range(10):
        recs.append(traderec(T0 - 150 + i * 10, 7000.00, sz=1, buy=False))

    # The book right at the open, which is what the decision must see.
    recs.append(bookrec(T0 - 0.001, bid=7000.00, ask=7000.25,
                        bsz=99, asz=11))

    # Explosion: heavy buying, far higher prices, a wildly different book.
    for i in range(20):
        recs.append(traderec(T0 + 5 + i * 10, 7050.0 + i, sz=500, buy=True))
        recs.append(bookrec(T0 + 5 + i * 10, bid=7050.0 + i,
                            ask=7050.25 + i, bsz=1, asz=900))
    recs.append(traderec(T0 + 400, 7100.0))

    s = next(x for x in condense(recs) if x.ts == T0)

    # Decision fields: from the quiet half only.
    assert s.bids[0][1] == 99 and s.asks[0][1] == 11
    assert s.delta == pytest.approx(-10.0)      # ten sells of one
    assert s.sell_vol == pytest.approx(10.0)
    assert s.buy_vol == 0.0
    assert s.prior_close == pytest.approx(7000.00)

    # Outcome fields: from the explosion.
    assert s.open == pytest.approx(7050.0)
    assert s.volume == pytest.approx(500 * 20)


def test_the_price_column_is_computed_from_before_the_open():
    """The bug this field exists to prevent.

    The price read must be the move across the window ENDING at the open.
    Computing it as `open` against `prior_close` reaches one trade into the
    candle -- and that one trade carries a slice of the answer. On a
    synthetic feed built to be a pure coin, that leak alone scored 63.7%
    with an interval clear of 50%: a confident finding about nothing.
    """
    recs = [bookrec(T0 - 200)]
    # The window: price walks 7000 -> 7014, a clear +20 bps before the open.
    recs.append(traderec(T0 - 150, 7000.0, sz=1, buy=True))
    recs.append(traderec(T0 - 50, 7014.0, sz=1, buy=True))
    # The candle itself then collapses. None of this may reach the read.
    recs.append(traderec(T0 + 10, 6900.0, sz=1, buy=False))
    recs.append(traderec(T0 + 400, 6900.0))

    s = next(x for x in condense(recs) if x.ts == T0)
    assert s.prior_open == pytest.approx(7000.0)
    assert s.prior_close == pytest.approx(7014.0)
    assert s.prior_move_bps == pytest.approx(20.0, abs=0.01)
    # And emphatically not the -120bps that open-against-prior-close gives.
    assert s.prior_move_bps > 0


def test_the_price_column_is_flat_when_the_window_held_one_trade():
    """One trade means no move to measure. Reporting the move against
    itself as zero is right; inventing one from the candle is not."""
    recs = [bookrec(T0 - 10), traderec(T0 - 5, 7000.0, sz=1),
            traderec(T0 + 10, 7050.0, sz=1), traderec(T0 + 400, 7050.0)]
    s = next(x for x in condense(recs) if x.ts == T0)
    assert s.prior_move_bps == pytest.approx(0.0)


def test_an_empty_window_gives_a_flat_price_column_not_a_crash():
    assert Slice(ts=T0).prior_move_bps == 0.0


def test_the_flow_window_ends_at_the_open_and_does_not_reach_back_forever():
    """Trades older than the lookback are not in the read."""
    recs = [bookrec(T0 - 1000)]
    recs.append(traderec(T0 - 900, 7000.0, sz=77, buy=True))   # far too old
    recs.append(traderec(T0 - 60, 7000.0, sz=3, buy=True))     # inside it
    recs.append(traderec(T0 + 10, 7000.0, sz=1))
    recs.append(traderec(T0 + 400, 7000.0))
    s = next(x for x in condense(recs, lookback_s=300.0) if x.ts == T0)
    assert s.delta == pytest.approx(3.0)
    assert s.trades_in == 1


def test_delta_is_signed_by_the_aggressor():
    recs = [bookrec(T0 - 10)]
    recs.append(traderec(T0 - 5, 7000.0, sz=10, buy=True))
    recs.append(traderec(T0 - 4, 7000.0, sz=4, buy=False))
    recs.append(traderec(T0 + 10, 7000.0))
    recs.append(traderec(T0 + 400, 7000.0))
    s = next(x for x in condense(recs) if x.ts == T0)
    assert s.delta == pytest.approx(6.0)
    assert s.buy_vol == pytest.approx(10.0)
    assert s.sell_vol == pytest.approx(4.0)


# ----------------------------------------------------------- the path

MINUTE = {"path_bucket_s": 60.0}     # for the tests about bucket mechanics


def test_the_path_keeps_one_bucket_per_period_in_order():
    recs = [bookrec(T0 - 1)]
    for i in range(5):
        recs.append(traderec(T0 + 5 + i * 60, 7000.0 + i, sz=1))
    recs.append(traderec(T0 + 400, 7100.0))
    s = next(x for x in condense(recs, **MINUTE) if x.ts == T0)
    assert len(s.path) == 5
    highs = [h for h, _ in s.path]
    assert highs == sorted(highs)          # monotonically rising
    assert highs[0] == pytest.approx(0.0)  # first bucket is the open


def test_a_bucket_with_no_trades_keeps_its_slot():
    """Dropping an empty bucket compresses the timeline, and reach would
    then read an ordering that never happened."""
    recs = [bookrec(T0 - 1)]
    recs.append(traderec(T0 + 5, 7000.0, sz=1))       # bucket 0
    recs.append(traderec(T0 + 245, 7010.0, sz=1))     # bucket 4
    recs.append(traderec(T0 + 400, 7000.0))
    s = next(x for x in condense(recs, **MINUTE) if x.ts == T0)
    assert len(s.path) == 5


def test_an_empty_bucket_carries_the_previous_extremes_forward():
    """Not zero -- zeroing would invent a round trip back to the open."""
    recs = [bookrec(T0 - 1)]
    recs.append(traderec(T0 + 5, 7000.0, sz=1))
    recs.append(traderec(T0 + 65, 7007.0, sz=1))
    recs.append(traderec(T0 + 245, 7007.0, sz=1))
    recs.append(traderec(T0 + 400, 7000.0))
    s = next(x for x in condense(recs, **MINUTE) if x.ts == T0)
    assert s.path[2] == s.path[1]
    assert s.path[2][0] > 0


def test_the_path_is_in_bps_from_the_candles_own_open():
    recs = [bookrec(T0 - 1)]
    recs.append(traderec(T0 + 5, 7000.0, sz=1))
    recs.append(traderec(T0 + 65, 7007.0, sz=1))      # +10 bps
    recs.append(traderec(T0 + 400, 7000.0))
    s = next(x for x in condense(recs, **MINUTE) if x.ts == T0)
    assert s.path[1][0] == pytest.approx(10.0, abs=0.01)


# --------------------------------------------- bucket size vs the target

def test_the_default_bucket_is_far_finer_than_the_candle():
    """The bug this sizing exists to prevent.

    With one-minute buckets on a five-minute ES candle, a bucket spans
    roughly 2.3 ticks each way -- wider than a two- or three-tick target.
    Reach scores a bucket holding both the target and the stop as a stop,
    so nearly every bar resolved as a loss: the two-tick row read 34.9%
    where a coin is 50%. That was the bucket, not the market.
    """
    assert ec.path_bucket_for(300.0) == pytest.approx(10.0)
    assert ec.path_bucket_for(60.0) == pytest.approx(2.0)


def test_the_bucket_never_goes_below_a_second():
    assert ec.path_bucket_for(5.0) == pytest.approx(1.0)
    assert ec.path_bucket_for(0.0) == pytest.approx(1.0)


def test_a_finer_bucket_resolves_an_ordering_a_coarse_one_hides():
    """Up two ticks, then down four, inside one minute.

    At minute resolution both land in one bucket and the favourable move is
    invisible to reach. At ten-second resolution the order is recoverable.
    """
    recs = [bookrec(T0 - 1)]
    recs.append(traderec(T0 + 1, 7000.00, sz=1))
    recs.append(traderec(T0 + 12, 7000.50, sz=1))     # +2 ticks
    recs.append(traderec(T0 + 35, 6999.00, sz=1))     # -4 ticks
    recs.append(traderec(T0 + 400, 7000.0))

    coarse = next(x for x in condense(recs, **MINUTE) if x.ts == T0)
    fine = next(x for x in condense(recs) if x.ts == T0)

    assert len(coarse.path) == 1                      # all one bucket
    assert len(fine.path) > 1

    from liqmap.project import reach
    # Coarse: the single bucket holds both, so it is scored as the stop.
    assert reach(coarse.path, side=1, target_bps=0.6, stop_bps=0.6) == "stop"
    # Fine: the favourable side genuinely came first, and is seen.
    assert reach(fine.path, side=1, target_bps=0.6, stop_bps=0.6) == "target"


def test_a_bucket_touching_both_ways_keeps_both_extremes():
    recs = [bookrec(T0 - 1)]
    recs.append(traderec(T0 + 1, 7000.0, sz=1))
    recs.append(traderec(T0 + 2, 7007.0, sz=1))
    recs.append(traderec(T0 + 3, 6993.0, sz=1))
    recs.append(traderec(T0 + 400, 7000.0))
    s = next(x for x in condense(recs) if x.ts == T0)
    hi, lo = s.path[0]
    assert hi == pytest.approx(10.0, abs=0.01)
    assert lo == pytest.approx(-10.0, abs=0.01)


# -------------------------------------------------------- completeness

def test_a_candle_with_no_book_is_dropped():
    """No read to grade. Keeping it would put an unscoreable row in the
    denominator."""
    recs = [traderec(T0 + 10, 7000.0, levels=False),
            traderec(T0 + 400, 7000.0, levels=False)]
    assert list(condense(recs)) == []


def test_the_partial_candle_at_the_end_of_the_file_is_dropped():
    """A short outcome graded against a full-length decision reads as a
    loss that never had time to happen."""
    recs = [bookrec(T0 - 1), traderec(T0 + 10, 7000.0)]
    assert list(condense(recs)) == []


def test_a_book_clear_drops_the_resting_depth():
    """The exchange has said the book is gone; carrying it forward would
    invent liquidity."""
    recs = [bookrec(T0 - 100),
            {"action": "R", "side": "N", "price": 0, "size": 0,
             "ts_event": int((T0 - 50) * 1e9), "levels": []},
            traderec(T0 + 10, 7000.0, levels=False),
            traderec(T0 + 400, 7000.0, levels=False)]
    assert list(condense(recs)) == []


def test_several_candles_come_out_in_order():
    recs = [bookrec(T0 - 1)]
    for c in range(4):
        for i in range(3):
            recs.append(traderec(T0 + c * 300 + 10 + i * 60,
                                 7000.0 + c, sz=1))
    recs.append(traderec(T0 + 1300, 7010.0))
    out = list(condense(recs))
    assert [s.ts for s in out] == [T0, T0 + 300, T0 + 600, T0 + 900]


def test_a_gap_in_the_tape_does_not_invent_empty_candles():
    """Two candles far apart must yield two rows, not the hundreds of
    empty ones between them."""
    recs = [bookrec(T0 - 1), traderec(T0 + 10, 7000.0, sz=1),
            traderec(T0 + 100_000, 7000.0, sz=1),
            traderec(T0 + 100_400, 7000.0, sz=1)]
    assert len(list(condense(recs))) <= 2


def test_an_empty_stream_yields_nothing():
    assert list(condense([])) == []


# ------------------------------------------------------------ storage

def test_a_round_trip_through_the_file_preserves_everything(tmp_path):
    out = list(condense(_one_candle()))
    p = tmp_path / "es.jsonl"
    assert ec.save(out, p) == 1
    back = ec.load(p)
    assert len(back) == 1
    a, b = out[0], back[0]
    assert b.ts == a.ts
    assert b.bids == a.bids and b.asks == a.asks
    assert b.delta == a.delta
    assert b.close == pytest.approx(a.close)
    assert len(b.path) == len(a.path)


def test_the_file_is_line_delimited_so_a_dead_run_is_still_readable(tmp_path):
    p = tmp_path / "es.jsonl"
    ec.save(condense(_one_candle()), p)
    lines = [l for l in p.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["ts"] == T0
