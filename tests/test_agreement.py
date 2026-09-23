"""The agreement table, and the historical base rate beside it.

What this guards:

EVERY CANDLE IS RECORDED, NOT JUST THE TRADEABLE ONES. Measuring only the
candles that passed the filter and concluding the filter works is the oldest
mistake in the book, and it is invisible in the result — the numbers look
fine, they are just answering a different question.

THE AVERAGE IS TAKEN IN THE CALLED DIRECTION. Averaging raw signed moves
across up-calls and down-calls cancels a working signal to zero. A rule that
is right every time in both directions would report an average of exactly
0.0bps, which reads as useless.

THE HISTORICAL REPLAY NEVER INVENTS A BOOK. There is no historical L2 depth,
so a book proxy built from candles would be price action agreeing with
itself. The replay must say it measures one side, and mean it.

THE REPLAY DOES NOT PEEK. A price-action read scored against the bar it was
taken from would report a near-perfect hit rate and look like a discovery.
"""

import tempfile
import time
from pathlib import Path

import pytest

from liqmap import replay as replay_mod
from liqmap.history import History
from liqmap.structure import Candle

NOW = time.time()
OPEN_TS = NOW - 300


@pytest.fixture
def hist():
    with tempfile.TemporaryDirectory() as tmp:
        h = History(Path(tmp) / "h.db")
        yield h
        h.close()


def add(hist, verdict="confirmed", book="up", price="up", ts=None,
        px=100.0, features=None, coin="BTC", interval="15m"):
    return hist.record_state(
        coin=coin, interval=interval,
        candle_ts=ts if ts is not None else OPEN_TS,
        candle_end=(ts if ts is not None else OPEN_TS) + 900,
        book_dir=book, book_strength=0.8, price_dir=price,
        price_strength=0.7, verdict=verdict, price=px,
        made_at=(ts if ts is not None else OPEN_TS) + 60,
        features=features)


def settle(hist, sid, close, high=None, low=None):
    hist.resolve_state(sid, close_px=close,
                       high_px=high if high is not None else max(close, 100.0),
                       low_px=low if low is not None else min(close, 100.0))


def fill(hist, n, verdict, book, price, close, start=0):
    """n settled states, all the same shape."""
    for i in range(n):
        sid = add(hist, verdict=verdict, book=book, price=price,
                  ts=OPEN_TS + (start + i) * 900)
        if sid:
            settle(hist, sid, close)


# --------------------------------------------------------------------------
# recording every candle
# --------------------------------------------------------------------------

def test_a_state_round_trips(hist):
    sid = add(hist)
    assert sid
    t = hist.agreement_table()
    assert t["pending"] == 1


def test_one_row_per_candle(hist):
    assert add(hist) is not None
    assert add(hist) is None


def test_states_are_recorded_for_every_verdict_not_just_tradeable_ones(hist):
    for i, v in enumerate(("confirmed", "conflict", "unconfirmed",
                           "no signal")):
        assert add(hist, verdict=v, ts=OPEN_TS + i * 900) is not None
    assert hist.agreement_table()["pending"] == 4


def test_a_zero_price_is_refused_rather_than_stored(hist):
    assert add(hist, px=0.0) is None


# --------------------------------------------------------------------------
# settling
# --------------------------------------------------------------------------

def test_settling_records_the_move_and_both_excursions(hist):
    sid = add(hist, px=100.0)
    settle(hist, sid, close=100.5, high=101.0, low=99.8)
    t = hist.agreement_table()
    row = next(r for r in t["table"] if r["state"] == "CONFIRMED up")
    assert row["n"] == 1
    assert row["avg_called_bps"] == pytest.approx(50.0, abs=0.5)
    assert row["avg_mfe_bps"] == pytest.approx(100.0, abs=0.5)
    assert row["avg_mae_bps"] == pytest.approx(-20.0, abs=0.5)


def test_a_down_call_that_wins_reports_a_positive_called_move(hist):
    """The sign convention that decides whether the table is readable at
    all. A correct short must not read as a loss."""
    sid = add(hist, book="down", price="down", px=100.0)
    settle(hist, sid, close=99.5, high=100.1, low=99.4)
    row = next(r for r in hist.agreement_table()["table"]
               if r["state"] == "CONFIRMED down")
    assert row["avg_called_bps"] == pytest.approx(50.0, abs=0.5)
    assert row["hit_rate"] == 1.0
    # And its favourable excursion is the LOW, not the high.
    assert row["avg_mfe_bps"] == pytest.approx(60.0, abs=0.5)


def test_up_and_down_calls_do_not_cancel_each_other_out(hist):
    """A rule right every time in both directions must not average to zero.

    This is the mistake that makes a working signal look like noise: raw
    signed moves cancel, moves in the CALLED direction add up.
    """
    for i in range(10):
        sid = add(hist, book="up", price="up", ts=OPEN_TS + i * 900, px=100.0)
        settle(hist, sid, close=100.4)
    for i in range(10, 20):
        sid = add(hist, book="down", price="down", ts=OPEN_TS + i * 900,
                  px=100.0)
        settle(hist, sid, close=99.6)

    both = next(r for r in hist.agreement_table()["table"]
                if r["state"] == "CONFIRMED (both)")
    assert both["n"] == 20
    assert both["avg_called_bps"] == pytest.approx(40.0, abs=1.0)
    assert both["hit_rate"] == 1.0


def test_resolving_twice_does_not_double_count(hist):
    sid = add(hist)
    settle(hist, sid, close=101.0)
    assert hist.pending_states(now=NOW + 99999) == []


def test_pending_only_lists_closed_candles(hist):
    add(hist, ts=OPEN_TS)
    add(hist, ts=OPEN_TS + 9000)
    assert len(hist.pending_states(now=OPEN_TS + 2000)) == 1


# --------------------------------------------------------------------------
# the question the table exists to answer
# --------------------------------------------------------------------------

def test_the_table_reports_each_state_separately(hist):
    fill(hist, 5, "confirmed", "up", "up", 100.5)
    fill(hist, 4, "conflict", "up", "down", 99.5, start=100)
    fill(hist, 3, "unconfirmed", "up", "flat", 100.1, start=200)

    states = {r["state"]: r for r in hist.agreement_table()["table"]}
    assert states["CONFIRMED (both)"]["n"] == 5
    assert states["CONFLICT (both)"]["n"] == 4
    assert states["UNCONFIRMED"]["n"] == 3
    assert states["CONFIRMED (both)"]["share"] == pytest.approx(5 / 12, abs=0.01)


def test_on_conflicts_the_two_sides_are_scored_separately(hist):
    """The money question: when they disagree, which one is right?"""
    # Book says up, price says down, price wins every time.
    fill(hist, 25, "conflict", "up", "down", 99.5)
    clash = next(r for r in hist.agreement_table()["table"]
                 if r["state"] == "CONFLICT (both)")
    assert clash["price_right"] == 1.0
    assert clash["book_right"] == 0.0


def test_the_verdict_says_which_side_wins_a_conflict(hist):
    fill(hist, 25, "confirmed", "up", "up", 100.5)
    fill(hist, 25, "conflict", "up", "down", 99.5, start=100)
    v = hist.agreement_table()["verdict"]
    assert "PRICE ACTION is right more often" in v
    assert "absorbed" in v


def test_the_verdict_reports_the_book_winning_when_it_does(hist):
    fill(hist, 25, "confirmed", "up", "up", 100.5)
    fill(hist, 25, "conflict", "up", "down", 100.5, start=100)
    v = hist.agreement_table()["verdict"]
    assert "BOOK is right more often" in v


def test_a_coin_flip_conflict_is_reported_as_one(hist):
    for i in range(30):
        sid = add(hist, verdict="conflict", book="up", price="down",
                  ts=OPEN_TS + i * 900, px=100.0)
        settle(hist, sid, close=100.5 if i % 2 else 99.5)
    fill(hist, 25, "confirmed", "up", "up", 100.5, start=100)
    v = hist.agreement_table()["verdict"]
    assert "neither side wins" in v or "coin flip" in v


def test_a_thin_sample_refuses_to_conclude(hist):
    fill(hist, 4, "confirmed", "up", "up", 100.5)
    v = hist.agreement_table()["verdict"]
    assert "too few to conclude" in v


def test_an_empty_table_does_not_divide_by_zero(hist):
    t = hist.agreement_table()
    assert t["n"] == 0
    assert all(r["n"] == 0 for r in t["table"])
    assert all(r["avg_called_bps"] is None for r in t["table"])


def test_the_table_filters_by_market(hist):
    fill(hist, 3, "confirmed", "up", "up", 100.5)
    sid = add(hist, coin="ETH", ts=OPEN_TS + 5000)
    settle(hist, sid, close=100.5)
    assert hist.agreement_table(coin="BTC")["n"] == 3
    assert hist.agreement_table(coin="ETH")["n"] == 1


def test_the_table_says_where_its_numbers_came_from(hist):
    assert "live order book" in hist.agreement_table()["source"]


# --------------------------------------------------------------------------
# the tuning surface
# --------------------------------------------------------------------------

def test_stored_features_can_be_sliced_after_the_fact(hist):
    """The point of storing raw readings: a new threshold can be tested
    against history already collected."""
    for i in range(20):
        tilt = -0.9 + (i / 20.0) * 1.8
        sid = add(hist, ts=OPEN_TS + i * 900, px=100.0,
                  features={"tilt": round(tilt, 3)})
        # Outcome improves with tilt, which the slice should reveal.
        settle(hist, sid, close=100.0 + tilt * 0.5)

    out = hist.feature_slice(feature="tilt")
    filled = [b for b in out["buckets"] if b["n"]]
    assert len(filled) >= 3
    assert filled[0]["avg_called_bps"] < filled[-1]["avg_called_bps"]


def test_the_available_features_are_discoverable(hist):
    add(hist, features={"tilt": 0.4, "slope_bps": 3.0, "position": 0.8})
    names = hist.stored_features()
    assert "tilt" in names and "slope_bps" in names


def test_slicing_with_no_stored_features_is_empty_not_a_crash(hist):
    fill(hist, 5, "confirmed", "up", "up", 100.5)
    out = hist.feature_slice(feature="tilt")
    assert all(b["n"] == 0 for b in out["buckets"])


def test_buckets_do_not_overlap(hist):
    """Half-open ranges, so a value never lands in two."""
    out = hist.feature_slice(feature="tilt", edges=[0.0, 0.5, 1.0])
    for a, b in zip(out["buckets"], out["buckets"][1:]):
        assert a["to"] == b["from"]


# --------------------------------------------------------------------------
# the historical base rate
# --------------------------------------------------------------------------

def series(n=600, step=60.0, start=1_700_000_000, seed=7):
    """A random walk at one-minute resolution. Deliberately unpredictable:
    on a smooth wave any model scores near 100% and the number would be a
    property of the fixture."""
    import random

    rng = random.Random(seed)
    out, px = [], 100.0
    for i in range(n):
        o = px
        c = o + rng.gauss(0.0, 0.05)
        out.append(Candle(ts=start + i * step, open=o, high=max(o, c) + 0.02,
                          low=min(o, c) - 0.02, close=c, volume=10.0))
        px = c
    return out


def group(sub, interval_s=900.0):
    """Aggregate sub-bars into trading-timeframe bars."""
    buckets: dict[float, list[Candle]] = {}
    for s in sub:
        buckets.setdefault(s.ts - (s.ts % interval_s), []).append(s)
    out = []
    for slot in sorted(buckets):
        g = sorted(buckets[slot], key=lambda c: c.ts)
        out.append(Candle(ts=slot, open=g[0].open,
                          high=max(x.high for x in g),
                          low=min(x.low for x in g), close=g[-1].close,
                          volume=sum(x.volume for x in g)))
    return out


def test_the_replay_produces_readings_from_real_sub_bars():
    sub = series(900)
    rows = replay_mod.replay_price_action(group(sub), sub, 900.0, 60.0)
    assert len(rows) > 10


def test_the_replay_never_claims_to_measure_the_book():
    out = replay_mod.table(replay_mod.replay_price_action(
        series(900) and group(series(900)), series(900), 900.0, 60.0))
    assert out["book_available"] is False
    assert out["measures"] == "price action only"
    assert "historical L2 depth" in out["caveat"]
    assert "agreeing with itself" in out["caveat"]


def test_the_replay_does_not_peek_at_its_own_outcome():
    """A read scored against the bar it was taken from reports near-perfect
    accuracy. On a random walk the honest answer is near a coin flip."""
    sub = series(3000, seed=11)
    rows = replay_mod.replay_price_action(group(sub), sub, 900.0, 60.0)
    out = replay_mod.table(rows)
    rate = out["table"][0]["hit_rate"]
    assert rate is not None
    assert 0.3 < rate < 0.7, (
        f"hit rate {rate:.0%} on a random walk — the replay is peeking")


def test_the_replay_refuses_input_it_cannot_stand_inside():
    assert replay_mod.replay_price_action([], [], 900.0, 60.0) == []
    sub = series(100)
    # Sub-bars the same size as the bar: there is no inside to stand in.
    assert replay_mod.replay_price_action(group(sub, 60.0), sub, 60.0, 60.0) == []


def test_the_replay_table_refuses_to_conclude_from_too_little():
    out = replay_mod.table([])
    assert out["n"] == 0
    assert "too few to mean anything" in out["verdict"]


def test_the_replay_calls_itself_the_base_rate():
    sub = series(3000, seed=3)
    out = replay_mod.table(
        replay_mod.replay_price_action(group(sub), sub, 900.0, 60.0))
    assert "BASE RATE" in out["verdict"]
    assert "the book is contributing nothing" in out["verdict"]
