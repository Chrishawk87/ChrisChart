"""Candle read tests.

The wiring that must be right, because getting it wrong inverts the trade:

ABSORPTION INVERTS FLOW. Heavy buying that is not moving price is a bearish
reading, not a bullish one. If absorption ever adds to flow in the same
direction, the tool tells you to buy exactly where the passive seller is
winning.

CONFIDENCE MUST RESPECT THE CLOCK. A perfect-looking read two minutes into a
fifteen-minute candle is not a good read. If confidence does not scale with
elapsed time, the tool will hand out its strongest signals at the moment they
are least reliable.

NOTHING CLAIMS A PROBABILITY IT HAS NOT MEASURED.
"""

import pytest

from liqmap.candleread import Calibration, CandleRead, Signal, outcome, read
from liqmap.flow import Absorption, BandStat, Book, FlowTape, Level, Trade
from liqmap.structure import Candle, Structure, VWAP, Zone

OPEN = 100.0


def tape_with(buy_notional=0.0, sell_notional=0.0, px=OPEN, n=10):
    t = FlowTape()
    for i in range(n):
        if buy_notional:
            t.add(Trade(px=px, sz=(buy_notional / n) / px, aggressor="buy", ts=i))
        if sell_notional:
            t.add(Trade(px=px, sz=(sell_notional / n) / px, aggressor="sell", ts=i))
    return t


def book_with(bid_sz=1.0, ask_sz=1.0):
    return Book(coin="X", ts=0.0,
                bids=[Level(99.9, bid_sz)], asks=[Level(100.1, ask_sz)])


def absorption_of(direction="buy", ratio=0.1, notional=5e6, confident=True,
                  thin=False):
    band = BandStat(low_px=99, high_px=101, observations=10,
                    first_notional=1e6, last_notional=1e6, min_notional=5e5,
                    max_notional=1e6, consumed=5e5, replenished=5e5,
                    refill_events=2)
    return Absorption(level=OPEN, window_s=300.0, direction=direction,
                      aggressive_notional=notional,
                      observed_bps=1.0, expected_bps=50.0,
                      impact_ratio=(3.0 if thin else ratio), band=band,
                      confident=confident, trades=40)


def base(**kw):
    kw.setdefault("coin", "BTC")
    kw.setdefault("interval_s", 900.0)
    kw.setdefault("elapsed_s", 700.0)
    kw.setdefault("open_px", OPEN)
    kw.setdefault("high_px", 101.0)
    kw.setdefault("low_px", 99.0)
    kw.setdefault("last_px", 100.5)
    return read(**kw)


# --------------------------------------------------------------------------
# candle geometry
# --------------------------------------------------------------------------

def test_elapsed_fraction_and_time_left():
    r = base(elapsed_s=450.0)
    assert r.elapsed_fraction == pytest.approx(0.5)
    assert r.seconds_left == pytest.approx(450.0)


def test_position_in_range():
    assert base(last_px=101.0).position_in_range == pytest.approx(1.0)
    assert base(last_px=99.0).position_in_range == pytest.approx(0.0)
    assert base(last_px=100.0).position_in_range == pytest.approx(0.5)


def test_a_candle_with_no_range_yet_is_not_a_divide_by_zero():
    r = base(high_px=100.0, low_px=100.0, last_px=100.0)
    assert r.position_in_range == 0.5


def test_change_in_bps_and_colour():
    r = base(last_px=101.0)
    assert r.change_bps == pytest.approx(100.0)
    assert r.green


# --------------------------------------------------------------------------
# individual signals
# --------------------------------------------------------------------------

def test_net_buying_reads_up():
    r = base(tape=tape_with(buy_notional=900_000, sell_notional=100_000))
    s = r.by_name("flow")
    assert s and s.direction == "up" and s.strength > 0.5


def test_net_selling_reads_down():
    r = base(tape=tape_with(buy_notional=100_000, sell_notional=900_000))
    assert r.by_name("flow").direction == "down"


def test_balanced_flow_is_flat():
    r = base(tape=tape_with(buy_notional=500_000, sell_notional=500_000))
    assert r.by_name("flow").direction == "flat"


def test_closing_near_the_high_reads_up():
    assert base(last_px=100.95).by_name("position").direction == "up"
    assert base(last_px=99.05).by_name("position").direction == "down"


def test_bid_heavy_book_reads_up():
    assert base(book=book_with(bid_sz=10, ask_sz=1)).by_name("imbalance").direction == "up"
    assert base(book=book_with(bid_sz=1, ask_sz=10)).by_name("imbalance").direction == "down"


def test_supply_zone_pushes_down_and_demand_pushes_up():
    supply = Zone(kind="supply", low=100.0, high=101.0, index=0, ts=0,
                  impulse_bps=50)
    demand = Zone(kind="demand", low=99.0, high=100.6, index=0, ts=0,
                  impulse_bps=50)
    assert base(zone=supply).by_name("zone").direction == "down"
    assert base(zone=demand).by_name("zone").direction == "up"


def test_a_tested_zone_counts_for_less_than_a_fresh_one():
    fresh = Zone(kind="supply", low=100.0, high=101.0, index=0, ts=0,
                 impulse_bps=50, tested=0)
    worn = Zone(kind="supply", low=100.0, high=101.0, index=0, ts=0,
                impulse_bps=50, tested=3)
    assert base(zone=fresh).by_name("zone").strength > \
           base(zone=worn).by_name("zone").strength


def test_a_zone_price_is_not_inside_produces_no_signal():
    far = Zone(kind="supply", low=200.0, high=210.0, index=0, ts=0,
               impulse_bps=50)
    assert base(zone=far).by_name("zone") is None


def test_higher_timeframe_trend_carries_through():
    down = Structure(direction="down", invalidation=110.0)
    assert base(higher=down).by_name("structure").direction == "down"


def test_a_ranging_higher_timeframe_contributes_nothing():
    assert base(higher=Structure(direction="range")).by_name("structure") is None


def test_stretched_above_vwap_reads_down():
    v = VWAP(value=100.0, upper_1=100.2, lower_1=99.8, upper_2=100.4,
             lower_2=99.6, anchor_ts=0, bars=50)
    assert base(last_px=101.0, vw=v).by_name("vwap").direction == "down"
    assert base(last_px=99.0, vw=v).by_name("vwap").direction == "up"


def test_holding_just_above_vwap_reads_up_weakly():
    v = VWAP(value=100.0, upper_1=101.0, lower_1=99.0, upper_2=102.0,
             lower_2=98.0, anchor_ts=0, bars=50)
    s = base(last_px=100.5, vw=v).by_name("vwap")
    assert s.direction == "up" and s.strength < 0.5


def test_liquidation_magnet_pulls_toward_the_cluster():
    up = base(magnet_bps=20.0, magnet_notional=3e7).by_name("magnet")
    down = base(magnet_bps=-20.0, magnet_notional=3e7).by_name("magnet")
    assert up.direction == "up" and down.direction == "down"


def test_a_distant_magnet_pulls_less_than_a_near_one():
    near = base(magnet_bps=10.0, magnet_notional=3e7).by_name("magnet")
    far = base(magnet_bps=300.0, magnet_notional=3e7).by_name("magnet")
    assert near.strength > (far.strength if far else 0.0)


# --------------------------------------------------------------------------
# absorption inverting flow — the wiring that matters most
# --------------------------------------------------------------------------

def test_absorbed_buying_reads_DOWN_not_up():
    """Heavy buying that is not moving price means somebody is selling into
    it. Reading that as bullish buys the top."""
    r = base(tape=tape_with(buy_notional=9e6, sell_notional=1e6),
             absorption=absorption_of(direction="buy"))
    assert r.by_name("flow").direction == "up"
    assert r.by_name("absorption").direction == "down"
    assert "wrong way" in r.by_name("absorption").note


def test_absorbed_selling_reads_UP():
    r = base(tape=tape_with(buy_notional=1e6, sell_notional=9e6),
             absorption=absorption_of(direction="sell"))
    assert r.by_name("flow").direction == "down"
    assert r.by_name("absorption").direction == "up"


def test_absorption_outweighs_flow_when_they_disagree():
    """It is the heavier weight on purpose: absorption is the harder-won
    observation and it is the one that keeps you out of the trap."""
    r = base(tape=tape_with(buy_notional=9e6, sell_notional=1e6),
             absorption=absorption_of(direction="buy", ratio=0.0))
    flow = r.by_name("flow").weighted()
    absorb = r.by_name("absorption").weighted()
    assert abs(absorb) > abs(flow)


def test_a_thin_book_extends_the_move_instead_of_inverting_it():
    r = base(tape=tape_with(buy_notional=9e6),
             absorption=absorption_of(direction="buy", thin=True))
    assert r.by_name("absorption").direction == "up"
    assert "thin" in r.by_name("absorption").note


def test_an_unconfident_absorption_reading_is_not_used():
    r = base(absorption=absorption_of(confident=False))
    assert r.by_name("absorption") is None


# --------------------------------------------------------------------------
# combining
# --------------------------------------------------------------------------

def test_signals_agreeing_produce_a_strong_lean():
    v = VWAP(value=99.0, upper_1=99.5, lower_1=98.5, upper_2=100.0,
             lower_2=98.0, anchor_ts=0, bars=50)
    r = base(last_px=100.9,
             tape=tape_with(buy_notional=9e6, sell_notional=1e6),
             book=book_with(bid_sz=10, ask_sz=1),
             higher=Structure(direction="up"))
    assert r.lean == "up"
    assert r.score > 0.3
    assert r.agreement > 0.7


def test_signals_disagreeing_flatten_the_lean():
    r = base(last_px=100.9,
             tape=tape_with(buy_notional=9e6, sell_notional=1e6),
             absorption=absorption_of(direction="buy", ratio=0.0),
             book=book_with(bid_sz=1, ask_sz=10),
             higher=Structure(direction="down"))
    assert abs(r.score) < 0.5
    assert r.agreement < 1.0


def test_no_signals_is_flat_and_says_so():
    r = base()
    r.signals.clear()
    assert r.lean == "flat" and r.score == 0.0
    assert "not enough data" in r.verdict()


def test_agreement_is_zero_when_the_lean_is_flat():
    r = base()
    r.signals.clear()
    assert r.agreement == 0.0


# --------------------------------------------------------------------------
# the clock
# --------------------------------------------------------------------------

def test_the_same_signals_are_trusted_less_early_in_the_candle():
    kw = dict(last_px=100.9,
              tape=tape_with(buy_notional=9e6, sell_notional=1e6),
              book=book_with(bid_sz=10, ask_sz=1),
              higher=Structure(direction="up"))
    early = base(elapsed_s=60.0, **kw)
    late = base(elapsed_s=840.0, **kw)

    assert early.score == pytest.approx(late.score)   # same reading
    assert early.confidence < late.confidence          # different trust
    assert early.early and not late.early


def test_an_early_read_says_it_is_early():
    r = base(elapsed_s=60.0, tape=tape_with(buy_notional=9e6))
    assert "EARLY" in r.verdict()


def test_elapsed_fraction_is_clamped():
    assert base(elapsed_s=99_999.0).elapsed_fraction == 1.0
    assert base(elapsed_s=-5.0).elapsed_fraction == 0.0


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def test_no_calibration_refuses_to_quote_a_probability():
    r = base(tape=tape_with(buy_notional=9e6))
    text = r.verdict()
    assert "No calibration yet" in text
    assert "%" not in text.split("No calibration")[1]


def test_calibration_needs_samples_before_it_reports():
    cal = Calibration()
    for _ in range(5):
        cal.observe(0.6, True)
    assert not cal.ready
    assert "not enough" in cal.describe(0.6)


def test_calibration_reports_a_measured_rate():
    cal = Calibration()
    for i in range(50):
        cal.observe(0.6, i % 10 < 7)        # 70% up in the "strong up" band
    assert cal.ready
    rate, n = cal.rate(0.6)
    assert rate == pytest.approx(0.7, abs=0.02)
    assert n == 50
    assert "70%" in cal.describe(0.6)


def test_calibration_separates_bands():
    cal = Calibration()
    for i in range(40):
        cal.observe(0.8, True)              # strong up always resolves up
    for i in range(40):
        cal.observe(-0.8, False)            # strong down always resolves down
    assert cal.rate(0.8)[0] == 1.0
    assert cal.rate(-0.8)[0] == 0.0


def test_a_read_with_calibration_quotes_the_measured_number():
    cal = Calibration()
    for i in range(60):
        cal.observe(0.6, i % 4 != 0)        # 75%
    r = base(tape=tape_with(buy_notional=9e6, sell_notional=1e6),
             book=book_with(bid_sz=10, ask_sz=1),
             higher=Structure(direction="up"), last_px=100.9,
             calibration=cal)
    assert "Historically" in r.verdict()


def test_band_boundaries_are_stable():
    assert Calibration.band_of(0.0) == "flat"
    assert Calibration.band_of(0.2) == "up"
    assert Calibration.band_of(0.9) == "strong up"
    assert Calibration.band_of(-0.2) == "down"
    assert Calibration.band_of(-0.9) == "strong down"
    assert Calibration.band_of(5.0) == "flat"     # out of range, not a crash


def test_calibration_table_covers_every_band():
    cal = Calibration()
    cal.observe(0.6, True)
    names = [row["band"] for row in cal.table()]
    assert names == ["strong down", "down", "flat", "up", "strong up"]
    assert all(row["n"] == 0 or row["up_rate"] is not None for row in cal.table())


def test_outcome_reads_a_closed_candle():
    assert outcome(Candle(ts=0, open=100, high=102, low=99, close=101))
    assert not outcome(Candle(ts=0, open=100, high=102, low=99, close=99.5))


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def test_render_lists_signals_strongest_first():
    r = base(tape=tape_with(buy_notional=9e6, sell_notional=1e6),
             book=book_with(bid_sz=10, ask_sz=1),
             higher=Structure(direction="up"), last_px=100.9)
    text = r.render()
    assert "flow" in text and "imbalance" in text
    body = text.split("\n\n")[1].strip().split("\n")
    weights = [abs(float(line.split()[2])) for line in body]
    assert weights == sorted(weights, reverse=True)
