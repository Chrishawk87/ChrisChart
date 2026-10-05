"""Layer four: what the resting book is doing.

Three traps, and the first one is the whole reason this file needs the tape
as well as the book.

A SHRINKING LEVEL IS A FILL OR A CANCEL and the book cannot tell you which.
Both look identical -- size was there, now it is not -- and they mean
opposite things. Without subtracting what printed at that price, "liquidity
disappearing" is mostly a report that trading happened.

RANK IS NOT IDENTITY. Index a book from the touch and every price move
looks like a catastrophe: price ticks up one, the "third level" is a
different price, and the comparison reports a wall that vanished when
nothing happened.

THE EDGE OF THE WINDOW IS NOT ZERO. A price that fell out of the feed's
visible levels has not been cancelled, it has stopped being reported.
"""

from __future__ import annotations

import pytest

from liqmap import dom


class L:
    def __init__(self, px, sz):
        self.px, self.sz = float(px), float(sz)


class B:
    def __init__(self, bids, asks, ts=0.0):
        self.bids = [L(p, s) for p, s in bids]
        self.asks = [L(p, s) for p, s in asks]
        self.ts = ts


class T:
    def __init__(self, ts, px, sz):
        self.ts, self.px, self.sz = float(ts), float(px), float(sz)


def book(bid=100.0, ask=100.25, n=6, bsz=10.0, asz=10.0, tick=0.25, ts=0.0):
    return B([(bid - i * tick, bsz) for i in range(n)],
             [(ask + i * tick, asz) for i in range(n)], ts=ts)


def lad(b, now=None):
    return dom.ladder_of(b, now=now)


# ------------------------------------------- a fill is not a cancellation

def shrink(b, px, to, side="bids"):
    """One level changed, the rest of the book untouched."""
    out = B([(l.px, l.sz) for l in b.bids], [(l.px, l.sz) for l in b.asks],
            ts=b.ts)
    for l in getattr(out, side):
        if abs(l.px - px) < 1e-9:
            l.sz = float(to)
    return out


def test_size_that_left_and_printed_is_traded_not_pulled():
    """THE test. Both look identical in the book and they mean opposite
    things: size traded away is demand being met, size pulled is demand
    withdrawn before anyone touched it."""
    old = lad(book(ts=0.0))
    new = lad(shrink(book(ts=1.0), 100.0, 4.0))      # best bid 10 -> 4
    traded = {100.0: 6.0}                            # and six printed there
    r = dom.read(old, new, traded)
    assert r.bid.traded == pytest.approx(6.0)
    assert r.bid.cancelled == pytest.approx(0.0)
    assert not r.bid.pulling


def test_the_same_shrink_with_no_print_is_entirely_a_withdrawal():
    old = lad(book(ts=0.0))
    new = lad(shrink(book(ts=1.0), 100.0, 4.0))
    r = dom.read(old, new, {})
    assert r.bid.traded == 0.0
    assert r.bid.cancelled == pytest.approx(6.0)


def test_size_that_left_without_printing_is_a_withdrawal():
    old = lad(book(ts=0.0))
    new = lad(book(bsz=4.0, ts=1.0))
    r = dom.read(old, new, {})                       # nothing traded
    assert r.bid.cancelled == pytest.approx(6.0 * 6)
    assert r.bid.traded == 0.0
    assert r.bid.pulling


def test_a_partly_filled_level_splits_both_ways():
    old = lad(book(ts=0.0))
    new = lad(book(bsz=2.0, ts=1.0))                 # 8 left per level
    r = dom.read(old, new, {100.0: 3.0})             # 3 of them printed
    assert r.bid.traded == pytest.approx(3.0)
    assert r.bid.cancelled == pytest.approx(8.0 * 6 - 3.0)


def test_the_pull_share_is_of_what_left_not_of_the_book():
    old = lad(book(ts=0.0))
    new = lad(book(bsz=5.0, ts=1.0))
    r = dom.read(old, new, {100.0: 5.0})
    assert r.bid.pull_share == pytest.approx(
        r.bid.cancelled / (r.bid.cancelled + r.bid.traded))


def test_a_price_that_never_printed_cannot_have_been_traded():
    old = lad(book(ts=0.0))
    new = lad(book(bsz=1.0, ts=1.0))
    r = dom.read(old, new, {99.99: 500.0})   # volume at a price not on the book
    assert r.bid.traded == 0.0


# --------------------------------------------------------- rank is not price

def test_a_price_move_is_not_liquidity_vanishing():
    """Index from the touch and this reads as the whole book disappearing.
    Keyed by price, the levels that still exist are still there."""
    old = lad(book(bid=100.0, ask=100.25, ts=0.0))
    new = lad(book(bid=100.25, ask=100.50, ts=1.0))   # price ticked up one
    r = dom.read(old, new, {})
    assert r.bid.cancelled == pytest.approx(0.0)
    assert r.ask.cancelled == pytest.approx(0.0)


def test_the_same_price_is_compared_with_itself():
    old = lad(B([(100.0, 10.0), (99.75, 10.0)], [(100.25, 10.0)], ts=0.0))
    new = lad(B([(100.0, 2.0), (99.75, 10.0)], [(100.25, 10.0)], ts=1.0))
    r = dom.read(old, new, {})
    assert r.bid.cancelled == pytest.approx(8.0)


# --------------------------------------------- the edge of the feed's window

def test_a_price_that_left_the_window_was_not_cancelled():
    """It stopped being reported. Comparing it to zero invents a withdrawal
    every time the market moves."""
    old = lad(B([(100.0, 10.0), (99.75, 10.0), (99.50, 10.0)],
                [(100.25, 10.0)], ts=0.0))
    new = lad(B([(100.0, 10.0), (99.75, 10.0)],       # 99.50 fell out of view
                [(100.25, 10.0)], ts=1.0))
    r = dom.read(old, new, {})
    assert r.bid.cancelled == pytest.approx(0.0)


def test_a_comparison_over_too_few_shared_prices_says_so():
    old = lad(B([(100.0, 10.0)], [(100.25, 10.0)], ts=0.0))
    new = lad(B([(100.0, 2.0)], [(100.25, 10.0)], ts=1.0))
    r = dom.read(old, new, {})
    assert not r.bid.measured
    assert not r.bid.pulling
    assert r.bid.overlap < dom.MIN_OVERLAP


def test_the_overlap_is_reported_so_a_thin_reading_is_visible():
    old, new = lad(book(ts=0.0)), lad(book(ts=1.0))
    r = dom.read(old, new, {})
    assert r.bid.overlap == 6 and r.ask.overlap == 6
    assert r.measured


# ----------------------------------------------------------- stacking

def test_size_arriving_at_prices_already_there_is_stacking():
    old = lad(book(bsz=5.0, ts=0.0))
    new = lad(book(bsz=15.0, ts=1.0))
    r = dom.read(old, new, {})
    assert r.bid.added == pytest.approx(60.0)
    assert r.bid.stacking and not r.bid.pulling


def test_a_side_cannot_be_stacking_and_pulling_at_once():
    old = lad(book(bsz=5.0, ts=0.0))
    new = lad(book(bsz=15.0, ts=1.0))
    r = dom.read(old, new, {})
    assert not (r.bid.stacking and r.bid.pulling)


# --------------------------------------------------------- vanishing ahead

def test_withdrawal_on_the_side_price_is_walking_into_is_called_out():
    """Liquidity leaving before price arrives is a different thing from it
    being eaten on arrival, and the two are only separable because the
    prints were already subtracted."""
    old = lad(B([(100.0, 10.0)] + [(100.0 - i * 0.25, 10.0)
                                   for i in range(1, 6)],
                [(100.25 + i * 0.25, 10.0) for i in range(6)], ts=0.0))
    new = lad(B([(100.25, 10.0)] + [(100.25 - i * 0.25, 10.0)
                                    for i in range(1, 6)],
                [(100.50 + i * 0.25, 2.0) for i in range(6)], ts=1.0))
    r = dom.read(old, new, {})
    assert r.moved > 0
    assert r.ahead_side == "ask"
    assert r.vanishing_ahead > 0


def test_a_still_market_has_no_side_ahead_of_price():
    old, new = lad(book(ts=0.0)), lad(book(ts=1.0))
    r = dom.read(old, new, {})
    assert r.ahead_side is None
    assert r.vanishing_ahead == 0.0


def test_liquidity_eaten_on_arrival_is_not_counted_as_vanishing_ahead():
    """Price walked up and the offers shrank -- but every lot that left
    printed. That is liquidity being MET, not withdrawn, and the whole
    point of subtracting the tape is that the two do not get added
    together."""
    old = lad(B([(100.0 - i * 0.25, 10.0) for i in range(6)],
                [(100.25 + i * 0.25, 10.0) for i in range(6)], ts=0.0))
    new = lad(B([(100.25 - i * 0.25, 10.0) for i in range(6)],
                [(100.50 + i * 0.25, 2.0) for i in range(6)], ts=1.0))
    # Everything that left the shared offer prices printed there.
    traded = {round(100.50 + i * 0.25, 4): 8.0 for i in range(5)}
    r = dom.read(old, new, traded)
    assert r.moved > 0 and r.ahead_side == "ask"
    assert r.ask.traded > 0
    assert r.ask.cancelled == pytest.approx(0.0)
    assert r.vanishing_ahead == pytest.approx(0.0)


# ------------------------------------------------- appeared and left untested

def test_size_that_came_and_went_with_no_print_is_reported_as_observed():
    """What was seen: it appeared and left without anything trading into
    it. Not why -- a market maker widening ahead of a number does exactly
    this, and so does someone with no intention of being filled. The book
    is identical and intent is not in the data."""
    old = lad(B([(100.0, 10.0), (99.75, 50.0), (99.50, 10.0)],
                [(100.25, 10.0)], ts=0.0))
    new = lad(B([(100.0, 10.0), (99.75, 0.0), (99.50, 10.0)],
                [(100.25, 10.0)], ts=1.0))
    r = dom.read(old, new, {})
    assert r.pulled_untested == pytest.approx(50.0)


def test_nothing_here_calls_anything_a_spoof():
    import inspect

    src = inspect.getsource(dom)
    assert "spoof" in src.lower(), (
        "the limitation should be stated, not omitted")
    # Everything except the module docstring, where the limitation is
    # explained. No identifier, string or comment below it may claim to
    # know why size was withdrawn.
    body = src.replace(dom.__doc__ or "", "")
    assert "spoof" not in body.lower(), (
        "no identifier may claim to know intent")
    assert "pulled_untested" in body, (
        "the observation itself should still be reported")


# ----------------------------------------------------------- the migration

def test_size_moving_away_from_the_touch_is_drift():
    near = lad(B([(100.0, 100.0), (99.75, 10.0), (99.50, 10.0)],
                 [(100.25, 10.0), (100.50, 10.0), (100.75, 10.0)], ts=0.0))
    far = lad(B([(100.0, 10.0), (99.75, 10.0), (99.50, 100.0)],
                [(100.25, 10.0), (100.50, 10.0), (100.75, 10.0)], ts=1.0))
    r = dom.read(near, far, {})
    assert r.bid_drift > 0


def test_an_unchanged_book_has_not_drifted():
    old, new = lad(book(ts=0.0)), lad(book(ts=1.0))
    r = dom.read(old, new, {})
    assert r.bid_drift == pytest.approx(0.0)
    assert r.ask_drift == pytest.approx(0.0)


# ---------------------------------------------------------- volume by price

def test_prints_are_snapped_to_the_tick_so_they_match_the_book():
    """A venue reporting 100.0000001 and a book quoting 100.0 are talking
    about the same level, and a dictionary does not know that."""
    v = dom.volume_by_price([T(1.0, 100.0000001, 5.0)], 0.0, 2.0, tick=0.25)
    assert v.get(100.0) == pytest.approx(5.0)


def test_prints_outside_the_window_are_not_counted():
    trades = [T(0.5, 100.0, 5.0), T(5.0, 100.0, 7.0)]
    v = dom.volume_by_price(trades, 1.0, 2.0)
    assert v == {}


def test_an_empty_tape_is_not_a_crash():
    assert dom.volume_by_price([], 0.0, 1.0) == {}
    assert dom.volume_by_price(None, 0.0, 1.0) == {}


# ---------------------------------------------------------------- the watch

def test_the_ladder_is_sampled_rather_than_kept_per_push():
    w = dom.DomWatch(every_s=0.25)
    assert w.add(book(ts=0.0), now=0.0)
    assert not w.add(book(ts=0.1), now=0.1)
    assert w.add(book(ts=0.3), now=0.3)
    assert len(w) == 2


def test_the_watch_does_not_grow_without_limit():
    w = dom.DomWatch(every_s=0.0, max_snaps=50)
    for i in range(500):
        w.add(book(ts=i * 1.0), now=i * 1.0)
    assert len(w) <= 50


def test_one_photograph_cannot_be_compared_with_anything():
    w = dom.DomWatch()
    w.add(book(ts=0.0), now=0.0)
    r = w.read()
    assert not r.measured
    assert r.samples == 1


def test_the_watch_reads_over_the_window_it_was_asked_for():
    w = dom.DomWatch(every_s=0.0)
    for i in range(30):
        w.add(book(ts=i * 1.0), now=i * 1.0)
    r = w.read(window_s=5.0, now=29.0)
    assert r.span_s == pytest.approx(5.0, abs=1.0)


def test_an_empty_book_is_ignored_rather_than_recorded():
    w = dom.DomWatch()
    assert not w.add(B([], []), now=0.0)
    assert not w.add(None, now=0.0)
    assert len(w) == 0


def test_the_read_says_what_it_saw_in_words():
    old = lad(book(ts=0.0))
    new = lad(book(bsz=1.0, ts=1.0))
    assert "pulling" in dom.read(old, new, {}).describe()
    assert "holding" in dom.read(lad(book(ts=0.0)),
                                 lad(book(ts=1.0)), {}).describe()


# ------------------------------------------------------ wired into the feed


def test_the_feed_keeps_the_ladder_as_well_as_the_touch():
    """`BookReader` keeps the touch sizes and a five-level aggregate, which
    cannot answer where size is MOVING to. The ladder keeps the depth."""
    from liqmap.live import LiveFeed

    f = LiveFeed("BTC", intervals=("1m",))
    assert isinstance(f.dom, dom.DomWatch)
    assert len(f.dom) == 0


def test_the_ladder_is_captured_on_the_same_push_as_the_touch():
    import inspect

    from liqmap import live

    src = inspect.getsource(live.LiveFeed._handle_book)
    assert "self.dom.add(" in src
    assert src.index("self.reader.add(") < src.index("self.dom.add(")


def test_capturing_the_ladder_never_takes_the_feed_down():
    """It runs inside the socket thread's lock on every book push. A bad
    book must return nothing rather than raise."""
    assert dom.ladder_of(None) is None
    assert dom.ladder_of(B([], [])) is None
    assert dom.ladder_of(B([(100.0, 1.0)], [])) is None
