"""Suggestion tests.

The failure this guards against is a tool that always finds a trade. Every
gate below exists because without it the suggestion would be produced from
something that was not measured, and a confident sentence built on an
invented number is worse than silence -- it gets acted on.

Pinned here:

REFUSAL IS THE COMMON CASE. Flat books, thin conviction, costs that eat the
target and stops in the wrong place all produce NoTrade, each naming its own
gate.

THE TARGET COMES FROM MEASURED RANGE. No recent bars, no suggestion.

THE STOP COMES FROM THE BOOK. Not from dividing the target by two.

BOTH-TOUCHED RESOLVES AS THE STOP. Bar data cannot order two touches inside
one bar, and assuming the good one manufactures a hit rate.
"""

import pytest

from liqmap.bookread import BookRead
from liqmap.flow import Book, Level
from liqmap.structure import Candle
from liqmap.suggest import (
    NoTrade, Suggestion, infer_tick, settle, suggest, typical_range_bps,
)

ENTRY = 100.0


def book_at(bid=99.99, ask=100.01, bid_sz=100.0, ask_sz=100.0,
            depth=10, tick=0.01, deep_bid_at=3, deep_sz=5000.0):
    """A book with a fat resting level a few ticks behind the bid."""
    bids = [Level(round(bid - i * tick, 6),
                  deep_sz if i == deep_bid_at else bid_sz)
            for i in range(depth)]
    asks = [Level(round(ask + i * tick, 6),
                  deep_sz if i == deep_bid_at else ask_sz)
            for i in range(depth)]
    return Book(coin="X", ts=0.0, bids=bids, asks=asks)


def read_at(score_driver=0.9, samples=200, absorbed=False):
    """A BookRead whose components all lean the same way, so `score` and
    `conviction` land where the test wants them."""
    v = score_driver
    return BookRead(
        ts=0.0, mid=ENTRY, microprice=ENTRY, spread_bps=2.0,
        tilt=v, imbalance=v, depletion=-v, replenish=v,
        mid_drift_bps=v * 10, aggression=v, absorbed=absorbed,
        samples=samples)


def bars(n=40, rng=0.5, px=ENTRY):
    """Closed bars with a predictable high-low range."""
    return [Candle(ts=i * 900.0, open=px, high=px + rng / 2,
                   low=px - rng / 2, close=px, volume=10.0)
            for i in range(n)]


def acting(direction="up", strength=1.0, bar_range_bps=100.0):
    """A CandleAction that clearly agrees with `direction`.

    Most tests below are about sizing, cost and shape, not about
    confirmation. They supply an agreeing action so the confirmation gate
    passes and the rest of the logic is actually reached; the four-state
    matrix is tested on its own further down.
    """
    from liqmap.confirm import CandleAction

    sign = 1.0 if direction == "up" else -1.0 if direction == "down" else 0.0
    move = sign * strength * bar_range_bps * 0.5
    return CandleAction(
        thrust_bps=move, position=0.5 + sign * strength * 0.45,
        slope_bps=move, bar_range_bps=bar_range_bps,
        extending=("up" if sign > 0 else "down" if sign < 0 else "flat"))


def settled(effort=1.4, aligned=1.0):
    """A held, paid-for agreement.

    Most tests here are about sizing and cost, not about stability, so they
    supply an agreement that has already settled — otherwise scalp mode
    correctly refuses every one of them as too fresh and nothing else gets
    exercised. The stability gate has its own tests.
    """
    from liqmap.confirm import Participation

    return {"held_s": 30.0, "flips": 0,
            "participation": Participation(effort=effort, aligned=aligned,
                                           notional=5e5, window_s=30.0)}


def ask(read=None, book=None, recent=None, **kw):
    for k, v in settled().items():
        kw.setdefault(k, v)
    kw.setdefault("coin", "BTC")
    kw.setdefault("interval", "15m")
    kw.setdefault("interval_s", 900.0)
    kw.setdefault("seconds_left", 600.0)
    kw.setdefault("notional", 10_000.0)
    r = read if read is not None else read_at()
    # Default to price agreeing with the book, so tests of everything else
    # reach everything else.
    kw.setdefault("action", acting(r.direction if r else "up"))
    if r is not None and r.direction == "down":
        kw["participation"] = settled(aligned=-1.0)["participation"]
    return suggest(r, book if book is not None else book_at(),
                   recent=bars() if recent is None else recent, **kw)


# --------------------------------------------------------------------------
# the happy path
# --------------------------------------------------------------------------

def test_a_strongly_bid_book_suggests_a_long():
    s = ask()
    assert isinstance(s, Suggestion)
    assert s.side == "long"
    assert s.target_px > s.entry > s.stop_px


def test_a_strongly_offered_book_suggests_a_short():
    s = ask(read=read_at(-0.9))
    assert isinstance(s, Suggestion)
    assert s.side == "short"
    assert s.target_px < s.entry < s.stop_px


def test_entry_is_the_side_you_actually_cross():
    """A long pays the offer. Using the mid flatters every suggestion by
    half a spread before anything has happened."""
    # A wide book, with a range big enough that the spread does not refuse
    # the trade before the entry side can be checked.
    b = book_at(bid=99.0, ask=101.0)
    wide = bars(rng=60.0)
    assert ask(book=b, recent=wide).entry == 101.0
    assert ask(read=read_at(-0.9), book=b, recent=wide).entry == 99.0


def test_the_sentence_names_direction_size_and_invalidation():
    s = ask()
    text = s.sentence()
    assert "LONG" in text and "ticks" in text
    assert "invalid" in text and "R." in text
    assert "Why:" in text
    assert "Cost:" in text


def test_the_sentence_says_it_is_a_read_not_a_forecast():
    assert "not a forecast" in ask().sentence()


def test_reasons_come_from_the_strongest_components():
    s = ask()
    assert s.reasons
    assert any("thin" in r or "microprice" in r or "offer" in r
               for r in s.reasons)


# --------------------------------------------------------------------------
# the gates — every one of these must refuse
# --------------------------------------------------------------------------

def test_a_cold_feed_refuses():
    r = ask(read=read_at(samples=2))
    assert isinstance(r, NoTrade) and r.gate == "warming"


def test_a_flat_book_refuses():
    r = ask(read=read_at(0.0))
    assert isinstance(r, NoTrade) and r.gate == "flat"


def test_low_conviction_refuses_and_says_which_way_it_leaned():
    # Every component at 0.2 scores 0.2, which is directional but only 40%
    # conviction — under the floor.
    r = ask(read=read_at(0.2))
    assert isinstance(r, NoTrade) and r.gate == "conviction"
    assert r.side == "long"


def test_no_book_refuses():
    r = ask(book=Book(coin="X", ts=0.0, bids=[], asks=[]))
    assert isinstance(r, NoTrade) and r.gate == "no book"


def test_the_last_seconds_of_a_candle_refuse():
    r = ask(seconds_left=5.0)
    assert isinstance(r, NoTrade) and r.gate == "too late"


def test_no_recent_bars_means_no_target_and_no_suggestion():
    """The target is a statement about measured range. Without bars there is
    nothing to measure, and a target invented at that point is the single
    most dangerous number the tool could print."""
    r = ask(recent=[])
    assert isinstance(r, NoTrade) and r.gate == "no range"


def test_a_wide_spread_that_eats_the_target_refuses():
    """Direction can be right and the trade still worthless."""
    wide = book_at(bid=95.0, ask=105.0, bid_sz=1.0, ask_sz=1.0)
    r = ask(book=wide, recent=bars(rng=0.2))
    assert isinstance(r, NoTrade) and r.gate == "cost"
    assert "spread just eats it" in r.detail


def test_fees_push_a_marginal_trade_over_the_line():
    """The same book and the same target, refused once the real cost of
    trading is included."""
    thin = bars(rng=0.45)
    ok = ask(recent=thin, fee_bps=0.0)
    assert isinstance(ok, Suggestion)
    with_fees = ask(recent=thin, fee_bps=ok.target_bps)
    assert isinstance(with_fees, NoTrade) and with_fees.gate == "cost"


def test_a_stop_too_far_for_the_target_refuses_on_shape():
    far = book_at(deep_bid_at=9, depth=10, tick=0.5)
    r = ask(book=far, recent=bars(rng=0.3))
    # A coarse tick on a quiet candle can trip "no room" before the shape is
    # even reached — the smallest whole-tick target is bigger than the bar.
    assert isinstance(r, NoTrade) and r.gate in ("shape", "cost", "no room")


def test_every_refusal_carries_a_readable_sentence():
    for r in (ask(read=read_at(0.0)), ask(recent=[]), ask(seconds_left=1.0)):
        assert isinstance(r, NoTrade)
        assert r.sentence().startswith("No trade — ")
        assert len(r.detail) > 20


# --------------------------------------------------------------------------
# the numbers
# --------------------------------------------------------------------------

def test_the_stop_sits_at_the_resting_level_the_trade_leans_on():
    """Not target/2. The book decides where the read is wrong."""
    b = book_at(deep_bid_at=3, tick=0.05, bid=99.99)
    s = ask(book=b, recent=bars(rng=1.0))
    assert isinstance(s, Suggestion)
    shelf_px = 99.99 - 3 * 0.05
    assert s.stop_px == pytest.approx(shelf_px - 0.05, abs=1e-6)


def test_the_stop_is_never_inside_the_noise():
    """A stop one tick under the touch is taken out by any sweep of the front
    queue, which is not invalidation."""
    b = book_at(bid=99.0, ask=101.0, deep_bid_at=0)
    s = ask(book=b, recent=bars(rng=40.0))
    assert isinstance(s, Suggestion)
    spread = 2.0
    assert abs(s.entry - s.stop_px) >= 3 * spread - 1e-9


def test_a_razor_thin_spread_does_not_produce_a_twenty_r_stop():
    """The bug this catches shipped once: a 0.26bp spread and a resting level
    two ticks behind it gave an invalidation under 1bp away on an instrument
    whose bars travel forty. That reads as a magnificent 22R and is a stop
    sitting inside the noise — it gets hit by the next flicker, and the
    measured hit rate then blames the book read for it."""
    tight = book_at(bid=7736.1, ask=7736.3, tick=0.1, deep_bid_at=1,
                    deep_sz=200_000.0)
    wide_bars = [Candle(ts=i * 900.0, open=7730, high=7745, low=7715,
                        close=7736, volume=10.0) for i in range(30)]
    s = ask(book=tight, recent=wide_bars)
    assert isinstance(s, Suggestion)
    assert s.rr < 6.0, f"{s.rr}R means the stop is inside the noise"
    # At least 15% of a typical bar's range.
    bar_bps = typical_range_bps(wide_bars)
    assert s.risk_bps >= bar_bps * 0.15 - 0.01


def test_a_reason_never_argues_against_its_own_trade():
    """A component pulling the other way is a caution, not a reason. Listing
    it under `Why` produces a sentence that argues both sides at once."""
    mixed = BookRead(ts=0.0, mid=ENTRY, microprice=ENTRY, spread_bps=2.0,
                     tilt=0.9, imbalance=-0.8, depletion=0.0, replenish=0.9,
                     mid_drift_bps=9.0, aggression=0.5, absorbed=False,
                     samples=200)
    s = ask(read=mixed, recent=bars(rng=2.0))
    assert isinstance(s, Suggestion) and s.side == "long"
    assert not any("offer heavy" in r for r in s.reasons)
    assert any("disagrees" in c for c in s.cautions)


def test_absorption_is_a_caution_not_a_reason_to_take_the_trade():
    s = ask(read=read_at(0.9, absorbed=True), recent=bars(rng=2.0))
    assert isinstance(s, Suggestion)
    assert not any("absorb" in r for r in s.reasons)
    assert any("absorb" in c for c in s.cautions)
    assert "Against:" in s.sentence()


def test_target_is_snapped_to_whole_ticks():
    s = ask()
    steps = abs(s.target_px - s.entry) / s.tick
    assert steps == pytest.approx(round(steps), abs=1e-6)
    assert s.target_ticks == round(steps)


def test_a_shorter_remaining_candle_gives_a_nearer_target():
    """Range accumulates with time. Two minutes left is not the same
    opportunity as ten."""
    wide = bars(rng=3.0)
    long_left = ask(seconds_left=800.0, recent=wide)
    short_left = ask(seconds_left=200.0, recent=wide)
    assert isinstance(long_left, Suggestion) and isinstance(short_left, Suggestion)
    assert short_left.target_bps < long_left.target_bps


def test_target_scales_with_measured_range_not_with_conviction():
    """Conviction decides WHETHER to suggest. It must not decide how far,
    or a confident read quietly becomes an ambitious one."""
    a = ask(read=read_at(0.6), recent=bars(rng=1.0))
    b = ask(read=read_at(0.95), recent=bars(rng=1.0))
    assert isinstance(a, Suggestion) and isinstance(b, Suggestion)
    assert a.target_bps == pytest.approx(b.target_bps)

    wider = ask(read=read_at(0.6), recent=bars(rng=2.0))
    assert wider.target_bps > a.target_bps


def test_rr_and_cost_multiple_are_consistent_with_the_parts():
    s = ask()
    assert s.rr == pytest.approx(s.target_bps / s.risk_bps, abs=0.01)
    assert s.cost_multiple == pytest.approx(s.target_bps / s.cost_bps, abs=0.1)


def test_a_bigger_size_costs_more_and_can_refuse_what_a_small_one_allows():
    """The round trip depends on how far into the book you have to reach."""
    small = ask(notional=1_000.0, recent=bars(rng=0.45))
    big = ask(notional=50_000_000.0, recent=bars(rng=0.45))
    assert isinstance(small, Suggestion)
    if isinstance(big, Suggestion):
        assert big.cost_bps > small.cost_bps
    else:
        # "depth" when the size is larger than the whole displayed book,
        # "cost" when it fits but is too expensive. Both are refusals for
        # the right reason.
        assert big.gate in ("cost", "depth")


def test_fees_are_flagged_when_they_were_not_included():
    assert "no fees" in ask().sentence()
    assert "no fees" not in ask(fee_bps=4.0).sentence()


def test_to_dict_round_trips_the_decision_relevant_fields():
    d = ask().to_dict()
    for key in ("side", "entry", "target_px", "stop_px", "rr", "cost_bps",
                "conviction", "sentence", "take"):
        assert key in d
    assert d["take"] is True
    assert ask(read=read_at(0.0)).to_dict()["take"] is False


# --------------------------------------------------------------------------
# range and tick
# --------------------------------------------------------------------------

def test_typical_range_uses_the_median_not_the_mean():
    """One news bar must not set a target nothing else can reach."""
    normal = bars(n=20, rng=1.0)
    spike = normal + [Candle(ts=99999.0, open=100.0, high=150.0, low=100.0,
                             close=150.0, volume=1.0)]
    assert typical_range_bps(spike) == pytest.approx(
        typical_range_bps(normal), rel=0.05)


def test_too_few_bars_gives_no_range_rather_than_a_guess():
    assert typical_range_bps(bars(n=4)) == 0.0


def test_range_ignores_malformed_bars():
    bad = [Candle(ts=i, open=1, high=-1, low=5, close=1) for i in range(10)]
    assert typical_range_bps(bad) == 0.0


def test_tick_is_the_smallest_gap_in_the_book():
    assert infer_tick(book_at(tick=0.05)) == pytest.approx(0.05)
    assert infer_tick(book_at(tick=0.25)) == pytest.approx(0.25)


def test_tick_falls_back_to_the_spread_on_a_one_level_book():
    b = Book(coin="X", ts=0.0, bids=[Level(99.0, 1.0)], asks=[Level(101.0, 1.0)])
    assert infer_tick(b) == pytest.approx(2.0)


def test_an_explicit_tick_overrides_the_inferred_one():
    s = ask(book=book_at(tick=0.05), tick=0.01, recent=bars(rng=3.0))
    assert isinstance(s, Suggestion)
    assert s.tick == 0.01


# --------------------------------------------------------------------------
# settling
# --------------------------------------------------------------------------

def test_target_hit_settles_as_target():
    assert settle("long", 105.0, 95.0, high=106.0, low=99.0) == "target"
    assert settle("short", 95.0, 105.0, high=101.0, low=94.0) == "target"


def test_stop_hit_settles_as_stop():
    assert settle("long", 105.0, 95.0, high=101.0, low=94.0) == "stop"
    assert settle("short", 95.0, 105.0, high=106.0, low=99.0) == "stop"


def test_neither_touched_settles_as_neither():
    assert settle("long", 105.0, 95.0, high=101.0, low=99.0) == "neither"


def test_both_touched_in_one_bar_settles_as_the_stop():
    """Bar data cannot say which came first. Assuming the good one is how a
    backtest reports a hit rate it will never reproduce live."""
    assert settle("long", 105.0, 95.0, high=106.0, low=94.0) == "stop"
    assert settle("short", 95.0, 105.0, high=106.0, low=94.0) == "stop"


def test_a_malformed_bar_settles_as_neither_rather_than_guessing():
    assert settle("long", 105.0, 95.0, high=0.0, low=0.0) == "neither"
    assert settle("long", 105.0, 95.0, high=90.0, low=110.0) == "neither"


# --------------------------------------------------------------------------
# hostile input
# --------------------------------------------------------------------------

def test_nothing_here_raises_on_degenerate_books():
    hostile = [
        Book(coin="X", ts=0.0, bids=[], asks=[]),
        Book(coin="X", ts=0.0, bids=[Level(1e-9, 1e-9)], asks=[Level(1e-9, 1e-9)]),
        Book(coin="X", ts=0.0, bids=[Level(100.0, 0.0)], asks=[Level(100.0, 0.0)]),
    ]
    for b in hostile:
        out = ask(book=b)
        assert isinstance(out, (Suggestion, NoTrade))


def test_a_zero_interval_does_not_divide_by_zero():
    out = ask(interval_s=0.0)
    assert isinstance(out, (Suggestion, NoTrade))


# --------------------------------------------------------------------------
# what the adversarial review found
# --------------------------------------------------------------------------

def test_a_book_with_no_asks_refuses_instead_of_pricing_the_trip_at_zero():
    """The worst bug in the first cut. With no asks, `round_trip_bps` returns
    0.0 — an unfillable walk, not a free one — the cost gate read that as
    free and skipped itself, and the tool offered a 3R short on a market
    with nothing to buy it back from."""
    one_sided = Book(coin="X", ts=0.0,
                     bids=[Level(100.0 - i * 0.01, 50.0) for i in range(5)],
                     asks=[])
    r = ask(read=read_at(-0.9), book=one_sided, recent=bars(rng=5.0))
    assert isinstance(r, NoTrade) and r.gate == "one sided"


def test_a_book_with_no_bids_refuses_too():
    one_sided = Book(coin="X", ts=0.0, bids=[],
                     asks=[Level(100.0 + i * 0.01, 50.0) for i in range(5)])
    r = ask(book=one_sided, recent=bars(rng=5.0))
    assert isinstance(r, NoTrade) and r.gate == "one sided"


def test_a_locked_or_crossed_book_refuses():
    """Zero spread prices the round trip at zero by the same mechanism."""
    locked = Book(coin="X", ts=0.0, bids=[Level(100.0, 10.0), Level(99.99, 10.0)],
                  asks=[Level(100.0, 10.0), Level(100.01, 10.0)])
    r = ask(book=locked, recent=bars(rng=5.0))
    assert isinstance(r, NoTrade) and r.gate == "locked"


def test_a_size_larger_than_the_displayed_book_refuses_on_depth():
    """`walk` fills what it can and reports `exhausted`; summing the
    slippage throws that flag away, so the most expensive trade available
    prices as one of the cheapest."""
    r = ask(notional=1e12, recent=bars(rng=10.0))
    assert isinstance(r, NoTrade) and r.gate == "depth"


def test_an_unknown_bar_length_refuses_rather_than_assuming_a_whole_candle():
    r = ask(interval_s=0.0, recent=bars(rng=5.0))
    assert isinstance(r, NoTrade) and r.gate == "no clock"


def test_a_one_sided_book_never_produces_a_negative_stop_price():
    """Before the one-sided gate, `best_bid` of 0.0 made the spread the whole
    ask price and drove the invalidation below zero."""
    for b in (Book(coin="X", ts=0.0, bids=[], asks=[Level(100.0, 5.0)]),
              Book(coin="X", ts=0.0, bids=[Level(100.0, 5.0)], asks=[])):
        out = ask(book=b, recent=bars(rng=50.0))
        if isinstance(out, Suggestion):
            assert out.stop_px > 0 and out.target_px > 0


# --------------------------------------------------------------------------
# scalp mode: the smallest target that still pays
# --------------------------------------------------------------------------

from liqmap.suggest import Call, assess, breakeven_hit_rate  # noqa: E402


def tight_book(px=7736.2, tick=0.1, spread_ticks=2, deep_at=3):
    """A realistic scalping book: ETH at 7736 with a 0.2 spread.

    The `book_at` fixture above has a 2bps spread, which is enormous for
    scalping — on it the stop floor alone (three spreads) is larger than any
    sensible scalp target, so every scalp is correctly refused and nothing
    about scalp mode gets exercised. Chris's own screenshot showed 0.003%.
    """
    bid = px - spread_ticks * tick / 2
    ask_px = px + spread_ticks * tick / 2
    return Book(
        coin="ETH", ts=0.0,
        bids=[Level(round(bid - i * tick, 4), 200.0 if i == deep_at else 30.0)
              for i in range(10)],
        asks=[Level(round(ask_px + i * tick, 4), 200.0 if i == deep_at else 30.0)
              for i in range(10)])


def wide_bars(n=40, px=7736.2, rng_bps=200.0):
    half = px * rng_bps / 10_000.0 / 2
    return [Candle(ts=i * 900.0, open=px, high=px + half, low=px - half,
                   close=px, volume=10.0) for i in range(n)]


def scalp(**kw):
    kw.setdefault("mode", "scalp")
    kw.setdefault("book", tight_book())
    kw.setdefault("recent", wide_bars())
    return ask(**kw)


def test_breakeven_is_the_ev_zero_point():
    # 2R with no cost: one win pays for two losses, so a third is enough.
    assert breakeven_hit_rate(100.0, 50.0) == pytest.approx(1 / 3, abs=0.001)
    # 1:1 is a coin flip.
    assert breakeven_hit_rate(50.0, 50.0) == pytest.approx(0.5)
    # Cost raises the bar, which is the entire point of tracking it.
    assert breakeven_hit_rate(50.0, 50.0, cost_bps=10.0) == pytest.approx(0.6)


def test_breakeven_never_divides_by_zero():
    assert breakeven_hit_rate(0.0, 0.0) == 1.0


def test_scalp_targets_are_much_nearer_than_range_targets():
    """The whole point: a nearer target is touched more often."""
    r = ask(mode="range", book=tight_book(), recent=wide_bars())
    s = scalp()
    assert isinstance(r, Suggestion) and isinstance(s, Suggestion)
    assert s.target_bps < r.target_bps / 3


def test_the_scalp_target_is_set_by_cost_not_by_range():
    """Double the bar range and the scalp target must not move — it is
    pinned to the round trip, not to volatility."""
    a = scalp(recent=wide_bars(rng_bps=200.0))
    b = scalp(recent=wide_bars(rng_bps=600.0))
    assert isinstance(a, Suggestion) and isinstance(b, Suggestion)
    assert a.target_bps == pytest.approx(b.target_bps, abs=0.01)

    # But double the COST and it must move.
    dearer = scalp(recent=wide_bars(rng_bps=200.0), fee_bps=20.0)
    assert isinstance(dearer, Suggestion)
    assert dearer.target_bps > a.target_bps


def test_a_scalp_target_always_clears_the_round_trip():
    for fee in (0.0, 1.0, 5.0, 15.0):
        s = scalp(fee_bps=fee, recent=wide_bars(rng_bps=800.0))
        if isinstance(s, Suggestion):
            assert s.target_bps >= s.cost_bps * 2.0 - 0.01, f"fee={fee}"


def test_the_scalp_stop_is_scaled_to_the_target_not_to_the_bar():
    """A scalp is held for seconds. Using the bar's range for its stop puts
    the invalidation thirty basis points from a nine basis point target,
    which is a losing trade wearing a cautious stop."""
    s = scalp(recent=wide_bars(rng_bps=800.0))
    assert isinstance(s, Suggestion)
    # Scaled to the target, not to the bar — the bar here is 800bps wide and
    # the risk must be nowhere near that.
    assert s.risk_bps <= s.target_bps * 2.5 + 0.05
    assert s.risk_bps < 20.0


def test_a_scalp_is_judged_on_required_hit_rate_not_on_r():
    """R:R would refuse every scalp, because a small target always has a bad
    one. That is the same fact as the high hit rate, counted twice."""
    s = scalp(recent=wide_bars(rng_bps=800.0))
    assert isinstance(s, Suggestion)
    assert s.rr < 1.5, "this would be refused by the range-mode R floor"
    assert s.breakeven <= 0.70


def test_a_trade_needing_an_impossible_hit_rate_is_refused():
    """Cost so high relative to what the book can give that no book read
    could sustain the rate it demands."""
    r = scalp(fee_bps=400.0, recent=wide_bars(rng_bps=200.0))
    assert isinstance(r, NoTrade)
    assert r.gate in ("breakeven", "no room", "cost")


def test_a_spread_wider_than_the_candle_has_room_for_refuses_with_no_room():
    r = scalp(recent=wide_bars(rng_bps=12.0), fee_bps=8.0)
    assert isinstance(r, NoTrade) and r.gate in ("no room", "cost", "breakeven")
    # Whichever gate catches it, the refusal must name the cause rather than
    # leaving the user to guess at the market.
    assert "spread" in r.detail or "round trip" in r.detail


def test_range_mode_still_uses_the_r_floor():
    assert ask(mode="range", book=tight_book(),
               recent=wide_bars()).rr >= 1.5


def test_every_suggestion_states_the_hit_rate_it_needs():
    for mode in ("scalp", "range"):
        s = ask(mode=mode, book=tight_book(), recent=wide_bars(rng_bps=400.0))
        assert isinstance(s, Suggestion)
        assert "break even" in s.sentence()
        assert 0.0 < s.breakeven < 1.0


def test_measured_performance_is_compared_against_what_is_needed():
    s = ask(mode="scalp", book=tight_book(), recent=wide_bars(rng_bps=800.0),
            measured_rate=0.80, measured_n=200)
    assert isinstance(s, Suggestion)
    assert s.edge_pts == pytest.approx((0.80 - s.breakeven) * 100, abs=0.2)
    assert "you are running 80%" in s.sentence()


def test_a_thin_sample_refuses_to_claim_an_edge():
    """Nineteen trades is an impression."""
    s = ask(mode="scalp", book=tight_book(), recent=wide_bars(rng_bps=800.0),
            measured_rate=0.95, measured_n=19)
    assert isinstance(s, Suggestion)
    assert s.edge_pts is None
    assert "not enough to compare" in s.sentence()


# --------------------------------------------------------------------------
# the call on every candle
# --------------------------------------------------------------------------

def _call(**kw):
    kw.setdefault("coin", "BTC")
    kw.setdefault("interval", "15m")
    kw.setdefault("interval_s", 900.0)
    kw.setdefault("seconds_left", 600.0)
    kw.setdefault("notional", 10_000.0)
    kw.setdefault("recent", bars(rng=2.0))
    r = kw.pop("read", read_at())
    b = kw.pop("book", book_at())
    kw.setdefault("action", acting(r.direction if r else "up"))
    for k, v in settled(aligned=-1.0 if (r and r.direction == "down") else 1.0).items():
        kw.setdefault(k, v)
    return assess(r, b, **kw)


def test_a_call_comes_back_even_when_there_is_no_trade():
    """A blank panel cannot be told apart from a broken one."""
    for read in (read_at(0.0), read_at(0.2), read_at(samples=1)):
        c = _call(read=read)
        assert isinstance(c, Call)
        assert c.tradeable is False
        assert c.blocked_by
        assert c.sentence()


def test_a_flat_book_and_a_blocked_lean_are_different_states():
    """Only one of them is worth watching."""
    flat = _call(read=read_at(0.0))
    leaning = _call(read=read_at(0.2))
    assert flat.side is None and flat.grade == "—"
    assert leaning.side == "long" and leaning.grade == "D"


def test_a_tradeable_call_carries_the_suggestion_and_a_letter_grade():
    c = _call(mode="range")
    assert c.tradeable is True
    assert c.grade in ("A", "B", "C")
    assert c.suggestion is not None
    assert c.sentence() == c.suggestion.sentence()


def test_grade_falls_as_the_required_hit_rate_rises():
    cheap = _call(mode="scalp", book=tight_book(),
                  recent=wide_bars(rng_bps=1200.0), fee_bps=0.0)
    dear = _call(mode="scalp", book=tight_book(),
                 recent=wide_bars(rng_bps=1200.0), fee_bps=6.0)
    assert cheap.tradeable and dear.tradeable
    order = {"A": 3, "B": 2, "C": 1}
    assert order[cheap.grade] >= order[dear.grade]
    assert cheap.suggestion.breakeven <= dear.suggestion.breakeven


def test_a_grade_is_about_shape_never_about_likelihood():
    """Documented here because it is the easiest thing to misread on a
    dashboard: an A says the numbers are not working against you, not that
    the trade wins."""
    c = _call(mode="range")
    assert "not a forecast" in c.sentence()


def test_the_call_dict_is_json_safe_and_complete():
    import json
    for read in (read_at(), read_at(0.0), read_at(0.2)):
        d = _call(read=read).to_dict()
        json.loads(json.dumps(d))
        for k in ("grade", "tradeable", "side", "detail", "sentence",
                  "blocked_by", "suggestion"):
            assert k in d
