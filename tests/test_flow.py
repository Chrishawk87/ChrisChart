"""Flow and liquidity tests.

Two things here have to be exactly right rather than approximately right.

The slippage walk is arithmetic on a known book, so it is tested against
hand-computed numbers, not against itself. If this is wrong it is wrong
silently -- it will still return a plausible-looking small number.

The absorption sign convention decides whether the tool says "buyers are being
absorbed" or "buyers are in control". Those are opposite trades. A buyer
pushing price DOWN is not being rewarded, and the signed-move test below is
what keeps that straight.
"""

import pytest

from liqmap.flow import (
    Absorption, Book, BandTracker, FlowTape, ImpactBaseline, Level, LevelWatch,
    Trade, check_side_convention,
)


def book(bids, asks, coin="BTC", ts=0.0):
    return Book(coin=coin, ts=ts,
                bids=[Level(px, sz) for px, sz in bids],
                asks=[Level(px, sz) for px, sz in asks])


# --------------------------------------------------------------------------
# the book
# --------------------------------------------------------------------------

def test_book_normalises_order_it_is_given():
    """A feed may hand levels over in either direction."""
    b = book([(99.0, 1), (100.0, 1), (98.0, 1)],
             [(103.0, 1), (101.0, 1), (102.0, 1)])
    assert [l.px for l in b.bids] == [100.0, 99.0, 98.0]
    assert [l.px for l in b.asks] == [101.0, 102.0, 103.0]
    assert b.best_bid == 100.0 and b.best_ask == 101.0
    assert b.mid == 100.5


def test_zero_size_levels_are_dropped():
    b = book([(100.0, 0), (99.0, 2)], [(101.0, 1)])
    assert [l.px for l in b.bids] == [99.0]


def test_spread_in_bps():
    b = book([(100.0, 1)], [(100.1, 1)])
    # 0.1 on a 100.05 mid
    assert b.spread_bps == pytest.approx(0.1 / 100.05 * 10_000, rel=1e-9)


def test_empty_book_is_safe():
    b = book([], [])
    assert b.empty
    assert b.mid == 0.0
    assert b.imbalance() == 0.0
    f = b.walk(1000.0, "buy")
    assert f.exhausted and f.notional_filled == 0.0


def test_depth_only_counts_inside_the_band():
    b = book([(100.0, 1), (99.0, 1), (90.0, 100)],
             [(101.0, 1), (110.0, 100)])
    # mid 100.5, 100bps = 1.005 either side -> [99.495, 101.505]
    assert b.depth(100.0, "buy") == pytest.approx(100.0 * 1)
    assert b.depth(100.0, "sell") == pytest.approx(101.0 * 1)


def test_imbalance_signs_the_heavier_side():
    heavy_bid = book([(100.0, 10)], [(101.0, 1)])
    heavy_ask = book([(100.0, 1)], [(101.0, 10)])
    assert heavy_bid.imbalance(200.0) > 0.7
    assert heavy_ask.imbalance(200.0) < -0.7


# -- the walk, against hand-computed numbers -------------------------------

def test_walk_consumes_one_level_when_it_fits():
    b = book([(99.0, 10)], [(100.0, 10)])          # 1000 of offer at 100
    f = b.walk(500.0, "buy")
    assert f.avg_px == pytest.approx(100.0)
    assert f.notional_filled == pytest.approx(500.0)
    assert f.levels_consumed == 1
    assert not f.exhausted


def test_walk_across_levels_averages_by_notional():
    """Hand-computed: 100 of notional at px 100 buys 1.0 unit; 300 at px 150
    buys 2.0 units. 400 notional, 3.0 units, average 133.333..."""
    b = book([(50.0, 100)], [(100.0, 1.0), (150.0, 2.0)])
    f = b.walk(400.0, "buy")
    assert f.notional_filled == pytest.approx(400.0)
    assert f.avg_px == pytest.approx(400.0 / 3.0)
    assert f.levels_consumed == 2
    assert f.worst_px == 150.0


def test_walk_partial_fill_on_the_last_level():
    b = book([(50.0, 100)], [(100.0, 1.0), (200.0, 1.0)])
    # 100 notional clears level one; 100 more takes 0.5 units at 200.
    f = b.walk(200.0, "buy")
    assert f.notional_filled == pytest.approx(200.0)
    assert f.avg_px == pytest.approx(200.0 / 1.5)
    assert not f.exhausted


def test_walk_reports_exhaustion_rather_than_pretending():
    b = book([(50.0, 1)], [(100.0, 1.0)])          # only 100 of offer
    f = b.walk(10_000.0, "buy")
    assert f.exhausted
    assert f.notional_filled == pytest.approx(100.0)
    assert f.filled_fraction == pytest.approx(0.01)


def test_slippage_is_positive_for_both_directions():
    """A cost is a cost. Same convention as consensus.entry_gap_pct."""
    b = book([(99.0, 100), (95.0, 100)], [(101.0, 100), (105.0, 100)])
    buy = b.walk(50_000.0, "buy")
    sell = b.walk(50_000.0, "sell")
    assert buy.slippage_bps > 0
    assert sell.slippage_bps > 0


def test_sweep_cost_is_measured_from_the_touch():
    b = book([(50.0, 100)], [(100.0, 1.0), (150.0, 2.0)])
    f = b.walk(400.0, "buy")
    # avg 133.33 against a touch of 100
    assert f.sweep_bps == pytest.approx((400.0 / 3.0 - 100.0) / 100.0 * 10_000, rel=1e-9)
    assert f.sweep_bps > f.slippage_bps or f.slippage_bps > 0


def test_deeper_book_costs_less_than_thin_book():
    deep = book([(99.0, 1000)], [(100.0, 1000), (100.1, 1000)])
    thin = book([(99.0, 1)], [(100.0, 1), (105.0, 1000)])
    size = 50_000.0
    assert deep.walk(size, "buy").slippage_bps < thin.walk(size, "buy").slippage_bps


def test_round_trip_adds_both_sides():
    b = book([(99.0, 100)], [(101.0, 100)])
    rt = b.round_trip_bps(10_000.0)
    assert rt == pytest.approx(b.walk(10_000.0, "buy").slippage_bps
                               + b.walk(10_000.0, "sell").slippage_bps)
    assert rt > 0


def test_shelf_is_found_where_size_is_outsized():
    bids = [(100.0, 1), (99.9, 1), (99.8, 1), (99.7, 50), (99.6, 1)]
    b = book(bids, [(100.1, 1), (100.2, 1), (100.3, 1)])
    shelves = b.shelves(bps=100.0, multiple=3.0)
    assert any(abs(s.px - 99.7) < 1e-9 and s.side == "buy" for s in shelves)


def test_flat_book_has_no_shelves():
    bids = [(100.0 - i * 0.1, 1) for i in range(6)]
    asks = [(100.1 + i * 0.1, 1) for i in range(6)]
    assert book(bids, asks).shelves(bps=200.0, multiple=3.0) == []


# --------------------------------------------------------------------------
# the tape
# --------------------------------------------------------------------------

def test_cvd_signs_the_aggressor():
    t = FlowTape()
    t.add(Trade(px=100.0, sz=10.0, aggressor="buy", ts=1))     # +1000
    t.add(Trade(px=100.0, sz=4.0, aggressor="sell", ts=2))     # -400
    assert t.cvd == pytest.approx(600.0)


def test_cvd_survives_pruning():
    """Old trades leave the window; the cumulative total must not reset."""
    t = FlowTape(max_age=10.0)
    t.add(Trade(px=100.0, sz=10.0, aggressor="buy", ts=0))
    t.add(Trade(px=100.0, sz=1.0, aggressor="buy", ts=1000))
    assert len(t) == 1
    assert t.cvd == pytest.approx(1100.0)


def test_window_counts_only_what_is_inside_it():
    t = FlowTape()
    t.add(Trade(px=100.0, sz=1.0, aggressor="buy", ts=0))
    t.add(Trade(px=100.0, sz=1.0, aggressor="buy", ts=500))
    w = t.window(60.0, now=500)
    assert w.trades == 1
    assert w.buy_notional == pytest.approx(100.0)


def test_lean_reports_which_side_is_paying_up():
    t = FlowTape()
    for i in range(8):
        t.add(Trade(px=100.0, sz=1.0, aggressor="buy", ts=i))
    t.add(Trade(px=100.0, sz=1.0, aggressor="sell", ts=9))
    assert t.window(60.0, now=9).lean > 0.7


def test_zero_size_trades_are_ignored():
    t = FlowTape()
    t.add(Trade(px=100.0, sz=0.0, aggressor="buy", ts=1))
    assert len(t) == 0 and t.cvd == 0.0


# -- divergence ------------------------------------------------------------

def test_divergence_when_price_rises_on_net_selling():
    t = FlowTape()
    t.add(Trade(px=100.0, sz=1.0, aggressor="sell", ts=0))
    for i in range(1, 6):
        t.add(Trade(px=100.0 + i * 0.1, sz=5.0, aggressor="sell", ts=i))
    d = t.divergence(60.0, now=5)
    assert d.disagrees
    assert d.price_change_bps > 0 and d.delta_notional < 0
    assert "passive buyers" in d.note


def test_divergence_when_price_falls_on_net_buying():
    t = FlowTape()
    t.add(Trade(px=100.0, sz=1.0, aggressor="buy", ts=0))
    for i in range(1, 6):
        t.add(Trade(px=100.0 - i * 0.1, sz=5.0, aggressor="buy", ts=i))
    d = t.divergence(60.0, now=5)
    assert d.disagrees
    assert d.price_change_bps < 0 and d.delta_notional > 0
    assert "passive sellers" in d.note


def test_agreement_is_not_reported_as_divergence():
    t = FlowTape()
    for i in range(6):
        t.add(Trade(px=100.0 + i * 0.1, sz=5.0, aggressor="buy", ts=i))
    assert not t.divergence(60.0, now=5).disagrees


# --------------------------------------------------------------------------
# impact baseline
# --------------------------------------------------------------------------

def test_baseline_needs_samples_before_it_claims_anything():
    b = ImpactBaseline()
    for i in range(5):
        b.observe(1_000_000.0, 10.0)
    assert not b.ready


def test_baseline_learns_bps_per_million():
    b = ImpactBaseline()
    for _ in range(30):
        b.observe(1_000_000.0, 8.0)
    assert b.ready
    assert b.bps_per_million == pytest.approx(8.0)
    assert b.expected_bps(2_000_000.0) == pytest.approx(16.0)


def test_baseline_ignores_dust_flow():
    """Dividing a price move by almost no volume produces a meaningless
    ratio, so those samples must not enter the estimate at all."""
    b = ImpactBaseline(min_notional=10_000.0)
    for _ in range(30):
        b.observe(100.0, 50.0)
    assert b.samples == 0 and not b.ready


def test_baseline_uses_median_so_one_cascade_cannot_move_it():
    b = ImpactBaseline()
    for _ in range(40):
        b.observe(1_000_000.0, 8.0)
    b.observe(1_000_000.0, 5_000.0)          # a liquidation print
    assert b.bps_per_million == pytest.approx(8.0)


def test_baseline_direction_does_not_matter():
    up, down = ImpactBaseline(), ImpactBaseline()
    for _ in range(25):
        up.observe(1_000_000.0, 8.0)
        down.observe(-1_000_000.0, -8.0)
    assert up.bps_per_million == pytest.approx(down.bps_per_million)


# --------------------------------------------------------------------------
# eaten versus replenished
# --------------------------------------------------------------------------

def test_band_tracks_consumption():
    t = BandTracker(99.0, 101.0)
    t.observe(book([(100.0, 10)], []))        # 1000
    t.observe(book([(100.0, 4)], []))         # 400
    s = t.stat()
    assert s.consumed == pytest.approx(600.0)
    assert s.replenished == 0.0
    assert s.refill_events == 0


def test_refill_is_counted_only_after_something_was_eaten():
    """Size arriving at an untouched level is not a refill. Only size that
    comes back after being taken proves anyone is defending."""
    t = BandTracker(99.0, 101.0)
    t.observe(book([(100.0, 1)], []))         # 100
    t.observe(book([(100.0, 10)], []))        # 1000, grew but nothing was eaten
    assert t.stat().refill_events == 0

    t2 = BandTracker(99.0, 101.0)
    t2.observe(book([(100.0, 10)], []))       # 1000
    t2.observe(book([(100.0, 2)], []))        # eaten to 200
    t2.observe(book([(100.0, 10)], []))       # back to 1000
    s = t2.stat()
    assert s.refill_events == 1
    assert s.consumed == pytest.approx(800.0)
    assert s.replenished == pytest.approx(800.0)


def test_defended_needs_repeated_refills_not_one():
    t = BandTracker(99.0, 101.0)
    t.observe(book([(100.0, 10)], []))
    t.observe(book([(100.0, 2)], []))
    t.observe(book([(100.0, 10)], []))
    assert not t.stat().defended          # one refill is an accident

    t.observe(book([(100.0, 2)], []))
    t.observe(book([(100.0, 10)], []))
    assert t.stat().defended              # twice is a decision


def test_emptying_level_is_not_defended():
    t = BandTracker(99.0, 101.0)
    for sz in (10, 8, 6, 4, 2, 1):
        t.observe(book([(100.0, sz)], []))
    s = t.stat()
    assert s.consumed > 0
    assert s.replenish_ratio < 0.3
    assert not s.defended


def test_tiny_jitter_is_not_counted_as_flow():
    t = BandTracker(99.0, 101.0, noise_floor=0.05)
    t.observe(book([(100.0, 10.0)], []))
    t.observe(book([(100.0, 10.05)], []))     # 0.5% wobble
    s = t.stat()
    assert s.consumed == 0.0 and s.replenished == 0.0


def test_band_ignores_levels_outside_it():
    t = BandTracker(99.0, 101.0)
    t.observe(book([(100.0, 10), (50.0, 1000)], [(500.0, 1000)]))
    assert t.stat().first_notional == pytest.approx(1000.0)


# --------------------------------------------------------------------------
# absorption
# --------------------------------------------------------------------------

def _train(watch, bps_per_million=10.0, n=40):
    """Teach the baseline that $1M of net aggression normally moves price
    `bps_per_million`.

    The flow and the move it causes have to land in the SAME bucket, which is
    the whole point of bucketing — impact is measured inside a bucket, not
    across the boundary between two. Training data spread across buckets
    teaches the baseline that a million dollars moves price nothing.

    Training runs well before the measurement window so it cannot contaminate
    the reading itself.
    """
    ts = 0.0
    px = 50_000.0
    for _ in range(n):
        open_px = px
        target = open_px * (1 + bps_per_million / 10_000.0)
        # five prints of $200k inside one bucket, walking open -> target
        for k in range(5):
            p = open_px + (target - open_px) * k / 4.0
            watch.on_trade(Trade(px=p, sz=200_000 / p, aggressor="buy",
                                 ts=ts + k))
        px = target
        ts += watch.bucket_s + 1
    watch.on_trade(Trade(px=px, sz=1e-9, aggressor="buy", ts=ts))  # flush last
    return ts, px


def test_absorption_flags_a_level_that_eats_aggression():
    w = LevelWatch("BTC", 50_000.0, window_s=120.0, bucket_s=15.0)
    ts, _ = _train(w)

    # now: heavy buying at the level, price barely moves
    base = ts + 1000
    for i in range(10):
        w.on_trade(Trade(px=50_000.0, sz=1_000_000 / 50_000.0,
                         aggressor="buy", ts=base + i))
    a = w.absorption(now=base + 10)

    assert a.confident
    assert a.direction == "buy"
    assert a.aggressive_notional == pytest.approx(10_000_000.0, rel=1e-6)
    assert a.expected_bps > 50           # 10m at 10bps/m
    assert abs(a.observed_bps) < 1
    assert a.impact_ratio < 0.5
    assert a.absorbing
    assert "ABSORBING" in a.verdict()


def test_thin_book_is_flagged_as_the_opposite():
    w = LevelWatch("BTC", 50_000.0, window_s=120.0, bucket_s=15.0)
    ts, _ = _train(w)

    base = ts + 1000
    px = 50_000.0
    for i in range(6):
        w.on_trade(Trade(px=px, sz=100_000 / px, aggressor="buy", ts=base + i))
        px *= 1.002                       # 20bps a print on small size
    a = w.absorption(now=base + 6)

    assert a.confident
    assert a.impact_ratio > 2.0
    assert a.thin and not a.absorbing
    assert "THIN" in a.verdict()


def test_buyer_pushing_price_down_is_not_called_absorption_of_sellers():
    """The sign that matters: direction is the AGGRESSOR, and the move is
    signed relative to what that aggressor wanted."""
    w = LevelWatch("BTC", 50_000.0, window_s=120.0, bucket_s=15.0)
    ts, _ = _train(w)

    base = ts + 1000
    px = 50_000.0
    for i in range(10):
        w.on_trade(Trade(px=px, sz=1_000_000 / px, aggressor="buy", ts=base + i))
        px *= 0.9995                      # buyers aggressing, price falling
    a = w.absorption(now=base + 10)

    assert a.direction == "buy"
    assert a.observed_bps < 0             # they are not being rewarded
    assert a.impact_ratio < 0
    assert a.absorbing


def test_no_verdict_without_a_baseline():
    w = LevelWatch("BTC", 50_000.0)
    for i in range(5):
        w.on_trade(Trade(px=50_000.0, sz=1.0, aggressor="buy", ts=i))
    a = w.absorption(now=5)
    assert not a.confident
    assert not a.absorbing and not a.thin
    assert "Baseline not established" in a.verdict()


def test_watch_reports_refills_in_the_verdict():
    w = LevelWatch("BTC", 50_000.0, band_bps=50.0, window_s=120.0, bucket_s=15.0)
    ts, _ = _train(w)
    for sz in (100, 20, 100, 20, 100):
        w.on_book(book([(50_000.0, sz)], [], ts=ts))

    base = ts + 1000
    for i in range(10):
        w.on_trade(Trade(px=50_000.0, sz=1_000_000 / 50_000.0,
                         aggressor="buy", ts=base + i))
    text = w.absorption(now=base + 10).verdict()
    assert "re-posted" in text and "paying to hold it" in text


def test_render_includes_slippage_when_given_a_size():
    w = LevelWatch("BTC", 50_000.0, window_s=120.0, bucket_s=15.0)
    ts, _ = _train(w)
    b = book([(49_990.0, 10)], [(50_010.0, 10)], coin="BTC")
    text = w.render(book=b, size=100_000.0)
    assert "round trip" in text
    assert "imbalance" in text


def test_render_warns_when_the_book_cannot_fill_you():
    w = LevelWatch("BTC", 50_000.0)
    b = book([(49_990.0, 0.001)], [(50_010.0, 0.001)])
    assert "cannot fill that size" in w.render(book=b, size=10_000_000.0)


# --------------------------------------------------------------------------
# side convention
# --------------------------------------------------------------------------

def test_side_convention_is_inferred_from_where_prints_land():
    samples = ([("B", 101.0)] * 20) + ([("A", 99.0)] * 20)
    out = check_side_convention(samples, mid=100.0)
    assert "B = BUY aggressor" in out
    assert "A = SELL aggressor" in out


def test_ambiguous_convention_says_so_rather_than_guessing():
    samples = ([("B", 101.0)] * 10) + ([("B", 99.0)] * 10)
    assert "ambiguous" in check_side_convention(samples, mid=100.0)


def test_side_convention_handles_no_samples():
    assert check_side_convention([], mid=100.0) == "no samples"


# --------------------------------------------------------------------------
# parsing the live shapes
# --------------------------------------------------------------------------

from liqmap.hl import parse_book, parse_trade      # noqa: E402


def test_parse_book_reads_the_documented_shape():
    raw = {"coin": "BTC", "time": 1_700_000_000_000,
           "levels": [[{"px": "99.5", "sz": "2", "n": 3},
                       {"px": "99.0", "sz": "1", "n": 1}],
                      [{"px": "100.5", "sz": "4", "n": 2}]]}
    b = parse_book("BTC", raw)
    assert b.best_bid == 99.5 and b.best_ask == 100.5
    assert b.bids[0].n == 3
    assert b.ts == pytest.approx(1_700_000_000.0)


def test_parse_book_drops_bad_levels_instead_of_raising():
    raw = {"levels": [[{"px": "nope", "sz": "1"}, {"px": "99", "sz": "1"},
                       {"px": "98", "sz": "0"}],
                      [{"sz": "1"}, {"px": "101", "sz": "2"}]]}
    b = parse_book("BTC", raw)
    assert [l.px for l in b.bids] == [99.0]
    assert [l.px for l in b.asks] == [101.0]


def test_parse_book_survives_junk():
    for raw in ({}, {"levels": []}, {"levels": [[], []]}, {"levels": None}):
        assert parse_book("BTC", raw).empty


def test_parse_trade_maps_the_side_code():
    buy = parse_trade({"px": "100", "sz": "2", "side": "B", "time": 1_000})
    sell = parse_trade({"px": "100", "sz": "2", "side": "A", "time": 1_000})
    assert buy.aggressor == "buy" and buy.notional == pytest.approx(200.0)
    assert sell.aggressor == "sell"
    assert buy.ts == pytest.approx(1.0)


def test_parse_trade_rejects_unusable_entries():
    assert parse_trade({"px": "0", "sz": "1", "side": "B"}) is None
    assert parse_trade({"px": "100", "sz": "0", "side": "B"}) is None
    assert parse_trade({"sz": "1", "side": "B"}) is None
    assert parse_trade({}) is None


def test_parse_trade_honours_an_overridden_convention():
    """If the live feed turns out to use the opposite codes, the fix is one
    argument rather than an edit to every call site."""
    t = parse_trade({"px": "100", "sz": "1", "side": "A"}, buy_codes={"A"})
    assert t.aggressor == "buy"


def test_zero_timestamp_is_kept_not_replaced_with_the_wall_clock():
    """`float(x or 0)/1000 or time.time()` silently turns a timestamp of 0
    into now, which puts one trade billions of seconds away from its
    neighbours and stops every window and bucket working."""
    t = parse_trade({"px": "100", "sz": "1", "side": "B", "time": 0})
    assert t.ts == 0.0


def test_missing_timestamp_falls_back_to_now():
    import time as _time
    t = parse_trade({"px": "100", "sz": "1", "side": "B"})
    assert abs(t.ts - _time.time()) < 5


def test_baseline_recovers_when_time_goes_backwards():
    """A reconnect replaying history, or one stray wall-clock stamp, must not
    park the bucket in the future forever."""
    w = LevelWatch("BTC", 50_000.0, bucket_s=15.0)
    w.on_trade(Trade(px=50_000.0, sz=1.0, aggressor="buy", ts=1_000_000.0))
    assert not w.baseline.ready

    ts = 0.0
    px = 50_000.0
    for _ in range(30):                      # sane stamps, far in the past
        target = px * 1.001
        for k in range(5):
            p = px + (target - px) * k / 4.0
            w.on_trade(Trade(px=p, sz=200_000 / p, aggressor="buy", ts=ts + k))
        px = target
        ts += 16
    w.on_trade(Trade(px=px, sz=1e-9, aggressor="buy", ts=ts))

    assert w.baseline.ready
    assert w.baseline.bps_per_million == pytest.approx(10.0, rel=1e-6)
