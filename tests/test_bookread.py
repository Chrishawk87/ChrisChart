"""Book read tests.

One question: which way is THIS candle going, from the book alone.

MICROPRICE IS THE CORE. Mid sits halfway between two queues that are almost
never the same size. The size-weighted mid leans toward the THINNER side,
because that is the side that gets consumed first. Ten million bid against
one million offered and the next print goes up — not because of a tendency,
but because there is nothing in the way. If the tilt ever points toward the
HEAVY side, every call this module makes is inverted.

REPLENISHMENT BEATS DEPTH. Displayed size can be cancelled in a microsecond.
Size that was eaten and came back has been paid for.
"""

import pytest

from liqmap.bookread import (
    BookReader, BookRead, Snap, snap,
)
from liqmap.flow import Book, Level


def book(bids, asks, ts=0.0):
    return Book(coin="X", ts=ts,
                bids=[Level(px, sz) for px, sz in bids],
                asks=[Level(px, sz) for px, sz in asks])


def ladder(mid=100.0, spread=0.02, bid_sz=5.0, ask_sz=5.0, n=6, ts=0.0):
    """A symmetric book with the touch sizes set explicitly."""
    b = mid - spread / 2
    a = mid + spread / 2
    bids = [(b - 0.01 * i, bid_sz if i == 0 else 3.0) for i in range(n)]
    asks = [(a + 0.01 * i, ask_sz if i == 0 else 3.0) for i in range(n)]
    return book(bids, asks, ts=ts)


# --------------------------------------------------------------------------
# microprice — the mechanism
# --------------------------------------------------------------------------

def test_microprice_leans_toward_the_thin_side():
    """Heavy bid, thin offer: the offer gets consumed first, so the next
    print goes UP. Tilt must be positive."""
    s = snap(ladder(bid_sz=100.0, ask_sz=1.0))
    assert s.microprice > s.mid
    assert s.tilt > 0.9


def test_a_thin_bid_tilts_down():
    s = snap(ladder(bid_sz=1.0, ask_sz=100.0))
    assert s.microprice < s.mid
    assert s.tilt < -0.9


def test_equal_queues_sit_at_the_mid():
    s = snap(ladder(bid_sz=7.0, ask_sz=7.0))
    assert s.microprice == pytest.approx(s.mid)
    assert s.tilt == pytest.approx(0.0)


def test_tilt_is_bounded_and_scale_free():
    """The same shape on a $4 market and a $100k one must read the same."""
    cheap = snap(ladder(mid=4.0, spread=0.001, bid_sz=50.0, ask_sz=1.0))
    rich = snap(ladder(mid=100_000.0, spread=25.0, bid_sz=50.0, ask_sz=1.0))
    assert cheap.tilt == pytest.approx(rich.tilt, abs=0.02)
    assert -1.0 <= cheap.tilt <= 1.0


def test_a_zero_spread_book_does_not_divide_by_zero():
    s = snap(book([(100.0, 5)], [(100.0, 5)]))
    assert s.tilt == 0.0


def test_near_imbalance_counts_notional_not_levels():
    s = snap(ladder(bid_sz=100.0, ask_sz=1.0))
    assert s.imbalance > 0
    assert s.near_bid > s.near_ask


def test_an_empty_or_one_sided_book_is_unreadable():
    assert snap(book([], [])) is None
    assert snap(book([(100.0, 1)], [])) is None
    assert snap(None) is None


# --------------------------------------------------------------------------
# dynamics
# --------------------------------------------------------------------------

def test_a_bid_being_eaten_reads_bearish():
    r = BookReader()
    for i, sz in enumerate((10.0, 8.0, 6.0, 4.0, 2.0, 1.0)):
        r.add(ladder(bid_sz=sz, ask_sz=5.0, ts=float(i)))
    read = r.read()
    assert read.depletion > 0.5, "the bid queue is the one shrinking"
    # depletion enters the score negatively
    assert any(c["name"] == "queue depletion" and c["value"] < 0
               for c in read.components())


def test_an_offer_being_eaten_reads_bullish():
    r = BookReader()
    for i, sz in enumerate((10.0, 8.0, 6.0, 4.0, 2.0, 1.0)):
        r.add(ladder(bid_sz=5.0, ask_sz=sz, ts=float(i)))
    assert r.read().depletion < -0.5


def test_replenishment_favours_the_side_that_keeps_coming_back():
    """Eaten and replaced repeatedly: somebody is paying to hold the bid."""
    r = BookReader()
    t = 0.0
    for _ in range(4):
        r.add(ladder(bid_sz=10.0, ask_sz=5.0, ts=t)); t += 1
        r.add(ladder(bid_sz=2.0, ask_sz=5.0, ts=t)); t += 1   # eaten
        r.add(ladder(bid_sz=10.0, ask_sz=5.0, ts=t)); t += 1  # replaced
    read = r.read()
    assert read.replenish > 0.5


def test_a_bid_stepping_down_counts_as_consumed():
    r = BookReader()
    r.add(ladder(mid=100.0, bid_sz=10.0, ts=0.0))
    r.add(ladder(mid=99.9, bid_sz=10.0, ts=1.0))     # touch dropped
    assert r.read().depletion > 0


def test_mid_drift_is_measured_over_the_window():
    r = BookReader(window_s=60.0)
    for i in range(10):
        r.add(ladder(mid=100.0 + i * 0.05, ts=float(i)))
    read = r.read()
    assert read.mid_drift_bps > 0
    assert read.samples == 10


# --------------------------------------------------------------------------
# the call
# --------------------------------------------------------------------------

def test_a_heavy_bid_with_the_offer_being_eaten_calls_up():
    r = BookReader()
    for i, ask in enumerate((10.0, 8.0, 6.0, 4.0, 2.0, 1.0)):
        r.add(ladder(mid=100.0 + i * 0.01, bid_sz=50.0, ask_sz=ask,
                     ts=float(i)))
    read = r.read(aggression=0.6)
    assert read.direction == "up"
    assert read.score > 0.2
    assert "UP" in read.verdict()


def test_a_heavy_offer_with_the_bid_being_eaten_calls_down():
    r = BookReader()
    for i, bid in enumerate((10.0, 8.0, 6.0, 4.0, 2.0, 1.0)):
        r.add(ladder(mid=100.0 - i * 0.01, bid_sz=bid, ask_sz=50.0,
                     ts=float(i)))
    read = r.read(aggression=-0.6)
    assert read.direction == "down"
    assert read.score < -0.2


def test_a_balanced_book_calls_flat():
    r = BookReader()
    for i in range(10):
        r.add(ladder(bid_sz=5.0, ask_sz=5.0, ts=float(i)))
    read = r.read(aggression=0.0)
    assert read.direction == "flat"
    assert abs(read.score) < 0.12


def test_aggression_that_is_not_moving_the_mid_is_not_counted():
    """Somebody is crossing and the mid has not moved: they are being
    absorbed, so their aggression is evidence of nothing."""
    r = BookReader()
    for i in range(12):
        r.add(ladder(mid=100.0, bid_sz=5.0, ask_sz=5.0, ts=float(i)))
    read = r.read(aggression=0.9)

    assert read.absorbed is True
    agg = next(c for c in read.components() if c["name"] == "aggression")
    assert agg["value"] == 0.0
    assert "absorbed" in read.verdict()


def test_aggression_that_moves_the_mid_is_counted():
    r = BookReader()
    for i in range(12):
        r.add(ladder(mid=100.0 + i * 0.02, bid_sz=5.0, ask_sz=5.0,
                     ts=float(i)))
    read = r.read(aggression=0.9)
    assert read.absorbed is False
    assert read.score > 0


def test_conviction_is_discounted_while_the_window_is_filling():
    r = BookReader()
    r.add(ladder(bid_sz=100.0, ask_sz=1.0, ts=0.0))
    r.add(ladder(bid_sz=100.0, ask_sz=1.0, ts=1.0))
    early = r.read()

    for i in range(2, 20):
        r.add(ladder(bid_sz=100.0, ask_sz=1.0, ts=float(i)))
    late = r.read()

    assert early.conviction < late.conviction
    assert "Warming up" in early.verdict()


def test_a_reader_with_nothing_in_it_returns_none():
    assert BookReader().read() is None


def test_the_window_drops_stale_snapshots():
    r = BookReader(window_s=5.0)
    for i in range(20):
        r.add(ladder(ts=float(i)))
    read = r.read()
    assert read.samples <= 6
    assert r.updates == 20


def test_one_flickering_snapshot_cannot_swing_the_call():
    """Instantaneous readings are averaged across the window, so a single
    odd book print does not flip the direction."""
    r = BookReader()
    for i in range(15):
        r.add(ladder(bid_sz=5.0, ask_sz=5.0, ts=float(i)))
    before = r.read().score

    r.add(ladder(bid_sz=500.0, ask_sz=0.1, ts=15.0))   # one wild print
    after = r.read().score
    assert abs(after - before) < 0.25


def test_every_component_is_reported_with_its_weight():
    r = BookReader()
    for i in range(12):
        r.add(ladder(ts=float(i)))
    comps = r.read().components()
    names = {c["name"] for c in comps}
    assert names == {"microprice tilt", "replenishment", "near imbalance",
                     "queue depletion", "aggression", "mid drift"}
    assert all(c["weight"] > 0 for c in comps)


def test_the_score_is_bounded():
    r = BookReader()
    for i in range(20):
        r.add(ladder(mid=100.0 + i, bid_sz=1000.0, ask_sz=0.01, ts=float(i)))
    read = r.read(aggression=1.0)
    assert -1.0 <= read.score <= 1.0
    assert 0.0 <= read.conviction <= 1.0
