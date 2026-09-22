"""Structure tests.

Two things must not be got wrong.

LOOKAHEAD. A swing is only knowable `strength` bars after it forms. Every
function that reads swings at bar i must use `confirmed_at <= i`, or the
structure layer quietly reads the future and everything validated on top of it
is worthless. Several tests below exist purely to pin that.

BOS VERSUS CHoCH. A break of structure is continuation; a change of character
is the sequence breaking. Confusing them inverts the trade. They are asserted
separately and by direction.
"""

import pytest

from liqmap.structure import (
    Candle, atr, last_shift, parse_candles, session_anchor, structure, swings,
    vwap, zones,
)


def c(o, h, l, cl, v=100.0, ts=0.0):
    return Candle(ts=ts, open=o, high=h, low=l, close=cl, volume=v)


def series(*bars):
    """(open, high, low, close) tuples -> candles one hour apart."""
    return [Candle(ts=i * 3600.0, open=o, high=h, low=lo, close=cl,
                   volume=100.0)
            for i, (o, h, lo, cl) in enumerate(bars)]


def ramp(start, step, n, spread=1.0, ts0=0.0):
    """A clean directional run, for building trends without hand-writing."""
    out = []
    px = start
    for i in range(n):
        o = px
        cl = px + step
        out.append(Candle(ts=ts0 + i * 3600.0, open=o,
                          high=max(o, cl) + spread, low=min(o, cl) - spread,
                          close=cl, volume=100.0))
        px = cl
    return out


def zigzag(*legs, bars_per_leg=6, spread=0.5):
    """Build a series through explicit turning points.

    Ramps glued end to end do NOT reliably produce swings: a leg that is
    still running at the end of the series never forms a pivot, so a
    "downtrend" built that way can come back with one swing high and get
    classified as a range. Stating the turning points makes the structure
    under test actually exist.
    """
    out: list[Candle] = []
    i = 0
    for a, b in zip(legs[:-1], legs[1:]):
        for k in range(bars_per_leg):
            o = a + (b - a) * k / bars_per_leg
            cl = a + (b - a) * (k + 1) / bars_per_leg
            out.append(Candle(ts=i * 3600.0, open=o,
                              high=max(o, cl) + spread,
                              low=min(o, cl) - spread,
                              close=cl, volume=100.0))
            i += 1
    return out


def downtrend():
    """200 -> 170 -> 190 (lower high) -> 150 (lower low) -> small bounce.

    The trailing bounce is not decoration: without bars after it, the 150 low
    is never confirmed as a pivot and the series reads as a range with two
    highs and one low.
    """
    return zigzag(160, 200, 170, 190, 150, 158)


# --------------------------------------------------------------------------
# candle basics
# --------------------------------------------------------------------------

def test_candle_geometry():
    bar = c(100, 110, 95, 105)
    assert bar.bullish and not bar.bearish
    assert bar.body_top == 105 and bar.body_bottom == 100
    assert bar.range == 15 and bar.body == 5
    assert bar.typical == pytest.approx((110 + 95 + 105) / 3)


def test_parse_candles_reads_the_documented_shape():
    raw = [{"t": 1_700_000_000_000, "o": "100", "h": "110", "l": "95",
            "c": "105", "v": "12.5", "n": 42}]
    out = parse_candles(raw)
    assert len(out) == 1
    assert out[0].ts == pytest.approx(1_700_000_000.0)
    assert out[0].close == 105.0 and out[0].volume == 12.5


def test_parse_candles_sorts_oldest_first():
    raw = [{"t": 2000, "o": "1", "h": "2", "l": "1", "c": "2"},
           {"t": 1000, "o": "1", "h": "2", "l": "1", "c": "2"}]
    assert [x.ts for x in parse_candles(raw)] == [1.0, 2.0]


def test_parse_candles_drops_junk_without_raising():
    raw = [{"t": 1000, "o": "1", "h": "2", "l": "1", "c": "2"},
           {"t": 2000, "o": "bad"}, "not a dict", None,
           {"t": 3000, "o": "1", "h": "0.5", "l": "9", "c": "2"}]  # high < low
    assert len(parse_candles(raw)) == 1


# --------------------------------------------------------------------------
# swings
# --------------------------------------------------------------------------

def test_swing_high_is_the_peak_of_its_window():
    bars = series((1, 2, 0, 1), (1, 3, 1, 2), (2, 4, 2, 3), (3, 9, 3, 8),
                  (8, 5, 3, 4), (4, 4, 2, 3), (3, 3, 1, 2))
    highs = [s for s in swings(bars, strength=3) if s.is_high]
    assert [s.index for s in highs] == [3]
    assert highs[0].px == 9


def test_swing_needs_enough_bars_either_side():
    assert swings(series((1, 2, 0, 1), (1, 3, 1, 2)), strength=3) == []


def test_confirmation_lags_the_swing_by_its_strength():
    """The whole point: you could not have known at the time."""
    bars = series((1, 2, 0, 1), (1, 3, 1, 2), (2, 4, 2, 3), (3, 9, 3, 8),
                  (8, 5, 3, 4), (4, 4, 2, 3), (3, 3, 1, 2))
    s = [x for x in swings(bars, strength=3) if x.is_high][0]
    assert s.index == 3
    assert s.confirmed_at == 6


def test_a_double_top_reports_one_swing_not_two():
    bars = series((1, 2, 0, 1), (1, 3, 1, 2), (2, 4, 2, 3), (3, 9, 3, 8),
                  (8, 9, 6, 7), (7, 5, 3, 4), (4, 4, 2, 3), (3, 3, 1, 2))
    highs = [s for s in swings(bars, strength=3) if s.is_high]
    assert len(highs) <= 1


def test_swing_low_mirrors_the_high():
    bars = series((9, 9, 8, 8), (8, 8, 7, 7), (7, 7, 6, 6), (6, 6, 1, 2),
                  (2, 4, 2, 3), (3, 5, 3, 4), (4, 6, 4, 5))
    lows = [s for s in swings(bars, strength=3) if not s.is_high]
    assert [s.index for s in lows] == [3]
    assert lows[0].px == 1


# --------------------------------------------------------------------------
# trend
# --------------------------------------------------------------------------

def test_higher_highs_and_higher_lows_is_an_uptrend():
    bars = (ramp(100, 2, 6) + ramp(112, -1, 4)
            + ramp(108, 3, 6) + ramp(126, -1, 4) + ramp(122, 2, 6))
    st = structure(bars)
    assert st.direction == "up"
    assert st.invalidation is not None
    assert "higher highs" in st.describe()


def test_lower_highs_and_lower_lows_is_a_downtrend():
    st = structure(downtrend())
    assert st.direction == "down"
    assert "lower highs" in st.describe()


def test_not_enough_swings_refuses_to_call_a_trend():
    st = structure(ramp(100, 1, 10))
    assert st.direction == "range"
    assert "not enough" in st.note


def test_expansion_is_a_range_not_a_trend():
    """Higher high AND lower low is not an uptrend, however bullish it looks.
    Widening swings are volatility, and calling them a trend is how you end
    up buying the top of a range."""
    bars = zigzag(150, 180, 140, 200, 120, 130, bars_per_leg=6)
    st = structure(bars)
    assert st.direction == "range"
    assert "expansion" in st.note


# --------------------------------------------------------------------------
# BOS versus CHoCH
# --------------------------------------------------------------------------

def test_downtrend_taking_out_its_low_is_a_bos_not_a_choch():
    bars = downtrend() + zigzag(158, 130, bars_per_leg=10)
    sh = last_shift(bars)
    assert sh is not None
    assert sh.kind == "BOS" and sh.direction == "down"
    assert "continuation" in sh.describe()


def test_downtrend_closing_above_its_lower_high_is_a_choch():
    """The structure shift a counter-trend entry waits for."""
    bars = downtrend() + zigzag(158, 205, bars_per_leg=10)
    sh = last_shift(bars)
    assert sh is not None
    assert sh.kind == "CHoCH" and sh.direction == "up"
    assert "character change" in sh.describe()


def test_a_wick_through_a_level_is_not_a_break():
    """Only closes count. Treating a stop run as a break makes every sweep a
    false signal."""
    base = downtrend()
    st = structure(base)
    highs = [s for s in st.swings if s.is_high]
    assert highs, "test needs a swing high to poke through"
    level = highs[-1].px

    poke = Candle(ts=base[-1].ts + 3600, open=base[-1].close,
                  high=level + 5, low=base[-1].close - 2,
                  close=base[-1].close, volume=100.0)
    sh = last_shift(base + [poke])
    assert sh is None or not (sh.kind == "CHoCH" and sh.index == len(base))


def test_no_shift_in_a_range():
    assert last_shift(ramp(100, 0.01, 12)) is None


def test_shift_uses_only_swings_confirmed_before_the_bar():
    """A shift that reads a swing confirmed after the breaking bar is
    lookahead. Every reported shift must satisfy the constraint."""
    bars = downtrend() + zigzag(158, 130, bars_per_leg=10)
    sh = last_shift(bars)
    assert sh is not None
    confirmed = [s for s in swings(bars)
                 if s.confirmed_at <= sh.index and s.index < sh.index]
    assert any(abs(s.px - sh.level) < 1e-9 for s in confirmed)


# --------------------------------------------------------------------------
# zones
# --------------------------------------------------------------------------

def test_supply_zone_comes_from_the_last_up_candle_before_the_drop():
    bars = downtrend() + zigzag(158, 130, bars_per_leg=10)
    zs = [z for z in zones(bars, min_impulse_bps=10.0) if z.kind == "supply"]
    assert zs, "a break below the prior low should leave a supply zone"
    z = zs[0]
    assert z.high >= z.low
    assert z.impulse_bps >= 10.0


def test_demand_zone_is_the_mirror():
    bars = zigzag(200, 160, 190, 170, 210, bars_per_leg=6) + zigzag(210, 240, bars_per_leg=8)
    zs = [z for z in zones(bars, min_impulse_bps=10.0) if z.kind == "demand"]
    assert zs


def test_a_weak_move_does_not_make_a_zone():
    bars = ramp(100, 0.001, 30)
    assert zones(bars, min_impulse_bps=100.0) == []


def test_untested_zone_is_fresh_and_a_retouched_one_is_not():
    bars = downtrend() + zigzag(158, 130, bars_per_leg=10)
    zs = zones(bars, min_impulse_bps=10.0)
    assert zs
    z = zs[0]
    assert (z.tested == 0) == z.fresh


def test_zone_containment_and_distance():
    bars = downtrend() + zigzag(158, 130, bars_per_leg=10)
    z = zones(bars, min_impulse_bps=10.0)[0]
    assert z.contains(z.mid)
    assert z.distance_bps(z.mid) == 0.0
    below = z.low * 0.9
    assert z.distance_bps(below) > 0          # zone is above you
    above = z.high * 1.1
    assert z.distance_bps(above) < 0          # zone is below you


def test_overlapping_zones_collapse():
    bars = downtrend() + zigzag(158, 130, 145, 110, bars_per_leg=8)
    zs = zones(bars, min_impulse_bps=10.0)
    supply = [z for z in zs if z.kind == "supply"]
    for i, a in enumerate(supply):
        for b in supply[i + 1:]:
            assert not (a.low <= b.high and b.low <= a.high), "zones overlap"


# --------------------------------------------------------------------------
# VWAP
# --------------------------------------------------------------------------

def test_vwap_is_volume_weighted_not_a_mean():
    bars = [Candle(ts=0, open=100, high=100, low=100, close=100, volume=1),
            Candle(ts=60, open=200, high=200, low=200, close=200, volume=99)]
    v = vwap(bars)
    assert v is not None
    assert v.value == pytest.approx((100 * 1 + 200 * 99) / 100)
    assert v.value > 198        # nowhere near the unweighted 150


def test_vwap_without_volume_returns_none_rather_than_a_moving_average():
    bars = [Candle(ts=0, open=100, high=101, low=99, close=100, volume=0)]
    assert vwap(bars) is None


def test_vwap_bands_widen_with_dispersion():
    tight = [Candle(ts=i, open=100, high=100, low=100, close=100, volume=10)
             for i in range(10)]
    wide = [Candle(ts=i, open=100 + (i % 2) * 50, high=100 + (i % 2) * 50,
                   low=100 + (i % 2) * 50, close=100 + (i % 2) * 50, volume=10)
            for i in range(10)]
    assert vwap(tight).upper_1 - vwap(tight).value == pytest.approx(0.0)
    assert vwap(wide).upper_1 - vwap(wide).value > 10


def test_vwap_band_labels():
    bars = [Candle(ts=i, open=100 + (i % 2) * 10, high=100 + (i % 2) * 10,
                   low=100 + (i % 2) * 10, close=100 + (i % 2) * 10, volume=10)
            for i in range(20)]
    v = vwap(bars)
    assert v.band_of(v.value) == "inside ±1σ"
    assert "above" in v.band_of(v.upper_2 + 1)
    assert "below" in v.band_of(v.lower_2 - 1)


def test_session_anchor_finds_the_days_first_candle():
    day = 86_400.0
    bars = [Candle(ts=day * 3 - 7200, open=1, high=1, low=1, close=1),
            Candle(ts=day * 3 - 3600, open=1, high=1, low=1, close=1),
            Candle(ts=day * 3, open=1, high=1, low=1, close=1),
            Candle(ts=day * 3 + 3600, open=1, high=1, low=1, close=1)]
    assert session_anchor(bars, day) == 2


def test_empty_inputs_are_safe():
    assert swings([]) == []
    assert structure([]).direction == "range"
    assert zones([]) == []
    assert vwap([]) is None
    assert last_shift([]) is None
    assert atr([]) == 0.0


# --------------------------------------------------------------------------
# ATR
# --------------------------------------------------------------------------

def test_atr_uses_true_range_including_gaps():
    bars = [Candle(ts=0, open=100, high=102, low=98, close=100),
            Candle(ts=60, open=120, high=122, low=118, close=120)]
    # True range of bar two is 122 - 100 = 22, not its 4-point bar range.
    assert atr(bars) == pytest.approx(22.0)


def test_atr_is_zero_on_a_flat_market():
    bars = [Candle(ts=i, open=100, high=100, low=100, close=100)
            for i in range(20)]
    assert atr(bars) == pytest.approx(0.0)
