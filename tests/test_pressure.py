"""Pressure tests.

The four states are the whole module, and two of them are the ones a naive
reading gets backwards:

    buyers aggressing, price NOT rising  -> SELLERS are winning
    sellers aggressing, price NOT falling -> BUYERS are winning

Aggressors pay the spread. If they are not being rewarded, the passive side
is absorbing them, and calling that bullish buys the top. Every one of those
cases is asserted by name below.

The multi-timeframe part must CONFRONT rather than average. A 4-hour where
sellers are winning and a 15-minute where buyers are winning is a specific
tradeable shape; averaging it to zero throws away the only information that
mattered.
"""

import pytest

from liqmap.flow import Book, Level
from liqmap.pressure import (
    Pressure, confront, from_candle, from_live,
)
from liqmap.structure import Candle


def p(tf="15m", interval=900, open_px=100.0, last=100.0, high=None, low=None,
      buy=0.0, sell=0.0, measured=True, **kw):
    return Pressure(
        timeframe=tf, interval_s=interval, open_px=open_px, last_px=last,
        high_px=high if high is not None else max(open_px, last),
        low_px=low if low is not None else min(open_px, last),
        buy_notional=buy, sell_notional=sell, measured=measured, **kw)


# --------------------------------------------------------------------------
# the four states
# --------------------------------------------------------------------------

def test_buyers_aggressing_into_a_rising_price_means_buyers_win():
    r = p(open_px=100.0, last=101.0, buy=900_000, sell=100_000)
    assert r.aggressor == "buy"
    assert r.winner == "buyers"
    assert not r.absorbing


def test_buyers_aggressing_into_a_flat_price_means_SELLERS_win():
    """The case a naive reading inverts. Ten million of buying that moves
    nothing means somebody is selling into all of it."""
    r = p(open_px=100.0, last=100.0, buy=9_000_000, sell=500_000)
    assert r.aggressor == "buy"
    assert r.winner == "sellers"
    assert r.absorbing
    assert r.signed < 0


def test_sellers_aggressing_into_a_falling_price_means_sellers_win():
    r = p(open_px=100.0, last=99.0, buy=100_000, sell=900_000)
    assert r.aggressor == "sell"
    assert r.winner == "sellers"
    assert not r.absorbing


def test_sellers_aggressing_into_a_flat_price_means_BUYERS_win():
    r = p(open_px=100.0, last=100.0, buy=500_000, sell=9_000_000)
    assert r.aggressor == "sell"
    assert r.winner == "buyers"
    assert r.absorbing
    assert r.signed > 0


def test_buyers_aggressing_while_price_FALLS_is_still_sellers():
    r = p(open_px=100.0, last=99.0, buy=9_000_000, sell=500_000)
    assert r.winner == "sellers" and r.absorbing


# --------------------------------------------------------------------------
# balanced and contested
# --------------------------------------------------------------------------

def test_balanced_aggression_with_no_move_is_contested():
    r = p(open_px=100.0, last=100.0, buy=500_000, sell=500_000)
    assert r.aggressor == "balanced"
    assert r.winner == "contested"
    assert r.signed == 0.0


def test_balanced_aggression_with_a_move_follows_the_move():
    assert p(open_px=100.0, last=101.0, buy=5e5, sell=5e5).winner == "buyers"
    assert p(open_px=100.0, last=99.0, buy=5e5, sell=5e5).winner == "sellers"


def test_no_activity_at_all_is_contested_not_a_crash():
    r = p(open_px=0.0, last=0.0)
    assert r.winner == "contested"
    assert r.move_bps == 0.0
    assert r.lean == 0.0


# --------------------------------------------------------------------------
# strength
# --------------------------------------------------------------------------

def test_one_sided_aggression_scores_stronger_than_a_split():
    lopsided = p(open_px=100.0, last=101.0, buy=950_000, sell=50_000)
    even = p(open_px=100.0, last=101.0, buy=550_000, sell=450_000)
    assert lopsided.strength > even.strength


def test_absorption_is_strongest_when_price_refuses_to_move():
    """Both are absorption — the aggressor is not being paid. The one where
    price has not budged at all is the harder wall."""
    still = p(open_px=100.0, last=100.0, buy=9e6, sell=1e5)
    drifting = p(open_px=100.0, last=100.015, buy=9e6, sell=1e5)   # 1.5bps
    assert still.absorbing and drifting.absorbing
    assert still.strength > drifting.strength


def test_a_move_beyond_the_flat_band_is_no_longer_absorption():
    """Fifteen basis points is price going somewhere. Calling that
    'absorbed' would flip a working breakout into a fade."""
    moving = p(open_px=100.0, last=100.15, buy=9e6, sell=1e5)
    assert not moving.absorbing
    assert moving.winner == "buyers"


def test_an_inferred_reading_is_discounted_against_a_measured_one():
    """A candle cannot tell you who crossed the spread. Treating the proxy as
    equivalent to counted fills would launder a guess into a measurement."""
    counted = p(open_px=100.0, last=101.0, buy=9e5, sell=1e5, measured=True)
    guessed = p(open_px=100.0, last=101.0, buy=9e5, sell=1e5, measured=False)
    assert guessed.strength < counted.strength


def test_partial_tape_coverage_reduces_strength():
    full = p(open_px=100.0, last=101.0, buy=9e5, sell=1e5, coverage=1.0)
    partial = p(open_px=100.0, last=101.0, buy=9e5, sell=1e5, coverage=0.3)
    assert partial.strength < full.strength


def test_strength_is_bounded():
    r = p(open_px=100.0, last=180.0, buy=1e12, sell=1.0)
    assert 0.0 <= r.strength <= 1.0


# --------------------------------------------------------------------------
# geometry and the book
# --------------------------------------------------------------------------

def test_position_in_range_and_imbalance():
    r = p(open_px=100.0, last=104.0, high=105.0, low=95.0,
          buy=1.0, sell=1.0, bid_depth=300.0, ask_depth=100.0)
    assert r.position_in_range == pytest.approx(0.9)
    assert r.book_imbalance == pytest.approx(0.5)


def test_a_flat_bar_does_not_divide_by_zero():
    r = p(open_px=100.0, last=100.0, high=100.0, low=100.0)
    assert r.position_in_range == 0.5
    assert r.book_imbalance == 0.0


# --------------------------------------------------------------------------
# building readings
# --------------------------------------------------------------------------

def test_from_live_counts_the_real_split():
    from liqmap.live import LiveCandle
    lc = LiveCandle(interval_s=900, start_ts=0, open=100.0, high=102.0,
                    low=99.0, close=101.5, volume=10.0, trades=40,
                    buy_notional=800_000.0, sell_notional=200_000.0)
    # Levels have to sit inside the 25bps band or the depth reads zero —
    # which is correct, and makes a wide-tick fixture misleading.
    book = Book(coin="X", ts=0,
                bids=[Level(101.4, 5), Level(101.3, 5)],
                asks=[Level(101.6, 1)])
    r = from_live("15m", 900, lc, book)

    assert r.measured is True
    assert r.buy_notional == 800_000.0
    assert r.winner == "buyers"
    assert r.bid_depth > r.ask_depth


def test_from_candle_infers_the_split_from_the_close():
    """Closing on the high means buyers held every attempt to push it down."""
    strong = Candle(ts=0, open=100.0, high=105.0, low=99.0, close=104.9,
                    volume=1000.0)
    weak = Candle(ts=0, open=100.0, high=105.0, low=99.0, close=99.1,
                  volume=1000.0)

    a = from_candle("15m", 900, strong)
    b = from_candle("15m", 900, weak)
    assert a.measured is False
    assert a.buy_notional > a.sell_notional
    assert b.sell_notional > b.buy_notional


def test_from_candle_on_a_doji_is_balanced():
    doji = Candle(ts=0, open=100.0, high=102.0, low=98.0, close=100.0,
                  volume=1000.0)
    r = from_candle("15m", 900, doji)
    assert r.aggressor == "balanced"


# --------------------------------------------------------------------------
# confrontation — the part that must not average
# --------------------------------------------------------------------------

def test_agreeing_timeframes_read_as_aligned():
    c = confront([p("4h", 14400, open_px=100, last=97, buy=1e5, sell=9e5),
                  p("15m", 900, open_px=100, last=99.5, buy=1e5, sell=9e5)])
    assert c.aligned and not c.conflicted
    assert c.consensus == "sellers"
    assert "ALIGNED" in c.verdict()


def test_a_higher_and_lower_timeframe_at_odds_is_named_not_averaged():
    """The exact shape a fade is built on: the slow timeframe still belongs
    to sellers while the fast one runs up into it."""
    c = confront([p("4h", 14400, open_px=100, last=96, buy=1e5, sell=9e5),
                  p("15m", 900, open_px=100, last=101, buy=9e5, sell=1e5)])

    assert c.conflicted and not c.aligned
    text = c.verdict()
    assert "CONFLICT" in text
    assert "4h belongs to sellers" in text
    assert "not a reason to follow" in text


def test_the_higher_timeframe_carries_more_weight_in_the_consensus():
    """An hour of evidence counts for more than a minute of it."""
    c = confront([p("4h", 14400, open_px=100, last=96, buy=1e5, sell=9e5),
                  p("1m", 60, open_px=100, last=100.2, buy=9e5, sell=1e5)])
    assert c.consensus == "sellers"


def test_ordered_puts_the_slowest_timeframe_first():
    c = confront([p("1m", 60), p("4h", 14400), p("15m", 900)])
    assert [r.timeframe for r in c.ordered] == ["4h", "15m", "1m"]
    assert c.higher.timeframe == "4h"
    assert c.lower.timeframe == "1m"


def test_a_single_timeframe_just_describes_itself():
    c = confront([p("15m", 900, open_px=100, last=101, buy=9e5, sell=1e5)])
    assert c.lower is None
    assert "15m" in c.verdict()


def test_no_readings_is_safe():
    c = confront([])
    assert c.verdict() == "No timeframes read."
    assert c.consensus == "contested"
    assert not c.aligned and not c.conflicted


def test_contested_timeframes_do_not_count_as_conflict():
    c = confront([p("4h", 14400, open_px=100, last=100, buy=5e5, sell=5e5),
                  p("15m", 900, open_px=100, last=101, buy=9e5, sell=1e5)])
    assert not c.conflicted


def test_by_timeframe_lookup():
    c = confront([p("4h", 14400), p("15m", 900)])
    assert c.by_timeframe("4h") is not None
    assert c.by_timeframe("nope") is None


# --------------------------------------------------------------------------
# price action — observations, not priors
# --------------------------------------------------------------------------

from liqmap.pressure import price_action    # noqa: E402


def bar(o, h, l, c, ts=0.0, v=100.0):
    return Candle(ts=ts, open=o, high=h, low=l, close=c, volume=v)


def flat_bars(n=10, px=100.0, spread=0.5):
    return [bar(px, px + spread, px - spread, px, ts=i * 60.0)
            for i in range(n)]


def test_a_long_upper_wick_reads_as_the_high_being_rejected():
    bars = flat_bars() + [bar(100.0, 110.0, 99.5, 100.5, ts=999)]
    pa = price_action(bars)
    assert pa.rejection < 0
    assert "rejected" in pa.describe()


def test_a_long_lower_wick_reads_as_the_low_being_rejected():
    bars = flat_bars() + [bar(100.0, 100.5, 90.0, 99.5, ts=999)]
    pa = price_action(bars)
    assert pa.rejection > 0


def test_a_symmetric_bar_is_indecision_not_rejection():
    """Equal wicks both ways is a doji. Reading every doji as a rejection
    would fire this on almost every bar and make it worthless."""
    bars = flat_bars() + [bar(100.0, 102.0, 98.0, 100.0, ts=999)]
    assert price_action(bars).rejection == 0.0


def test_a_swept_high_that_closes_back_inside_is_bearish():
    """The failed break. Whoever bought the breakout is trapped and has to do
    something about it — a mechanism, not a tendency."""
    bars = flat_bars(px=100.0, spread=0.5)
    bars.append(bar(100.0, 105.0, 99.8, 100.2, ts=999))   # through 100.5, back under
    pa = price_action(bars)
    assert pa.sweep < 0
    assert "failed break" in pa.describe() and "buyers trapped" in pa.describe()
    assert pa.signed < 0


def test_a_swept_low_that_closes_back_inside_is_bullish():
    bars = flat_bars(px=100.0, spread=0.5)
    bars.append(bar(100.0, 100.2, 95.0, 99.8, ts=999))
    pa = price_action(bars)
    assert pa.sweep > 0
    assert "sellers trapped" in pa.describe()
    assert pa.signed > 0


def test_closing_beyond_the_range_is_acceptance_not_a_sweep():
    """The opposite of a sweep, and what makes a breakout real."""
    bars = flat_bars(px=100.0, spread=0.5)
    bars.append(bar(100.0, 106.0, 100.0, 105.5, ts=999))
    pa = price_action(bars)
    assert pa.acceptance > 0
    assert pa.sweep == 0.0
    assert "accepted" in pa.describe()


def test_closing_below_the_range_is_downside_acceptance():
    bars = flat_bars(px=100.0, spread=0.5)
    bars.append(bar(100.0, 100.0, 94.0, 94.5, ts=999))
    pa = price_action(bars)
    assert pa.acceptance < 0


def test_a_sweep_outweighs_a_rejection_in_the_combined_score():
    """A wick says the level was defended. A failed break says somebody is
    trapped and has to buy or sell their way out — a mechanism, so it counts
    for more. The wicked bar here stays INSIDE the prior range so only the
    rejection fires."""
    swept = price_action(flat_bars() + [bar(100.0, 105.0, 99.8, 100.2, ts=9)])
    wicked = price_action(flat_bars() + [bar(100.2, 100.45, 99.6, 100.35, ts=9)])

    assert swept.sweep != 0.0 and wicked.sweep == 0.0
    assert wicked.rejection != 0.0
    assert abs(swept.signed) > abs(wicked.signed)


def test_a_deeper_sweep_reads_stronger_than_a_shallow_one():
    shallow = price_action(flat_bars() + [bar(100.0, 100.6, 99.8, 100.2, ts=9)])
    deep = price_action(flat_bars() + [bar(100.0, 108.0, 99.8, 100.2, ts=9)])
    assert abs(deep.sweep) > abs(shallow.sweep)


def test_a_quiet_range_produces_nothing():
    pa = price_action(flat_bars())
    assert not pa.active
    assert pa.signed == 0.0
    assert "nothing notable" in pa.describe()


def test_too_few_bars_is_safe():
    assert price_action([]).signed == 0.0
    assert price_action([bar(1, 2, 0.5, 1.5)]).signed == 0.0


def test_the_combined_score_is_bounded():
    bars = flat_bars() + [bar(100.0, 200.0, 99.9, 100.1, ts=9)]
    assert -1.0 <= price_action(bars).signed <= 1.0
