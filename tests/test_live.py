"""Live feed tests.

What has to be right, in order of how much damage it does when wrong:

GRID ALIGNMENT. A candle that starts when the socket happened to connect does
not line up with the one on your chart. A read on a bar offset by four minutes
from the venue's own is worse than no read, because it looks right.

ROLLOVER WITHOUT TRADES. A quiet market produces nothing to trigger a roll, so
a clock-driven roll is required or the candle stays "current" forever and
reports 100% elapsed for three intervals running.

OUT-OF-ORDER FILLS. A replayed fill from a closed bar must not widen the
current bar's range with a price from ten minutes ago.

THE UNKNOWN OPEN. Connecting mid-candle means the open already happened.
Pretending the first trade you saw was the open corrupts position-in-range and
the change figure, so it is marked rather than guessed.
"""

import time

import pytest

from liqmap.live import (
    INTERVALS, CandleBuilder, LiveCandle, LiveFeed, grid_start,
)
from liqmap.flow import Trade
from liqmap.structure import Candle


def tr(px, sz=1.0, side="buy", ts=0.0):
    return Trade(px=px, sz=sz, aggressor=side, ts=ts)


# --------------------------------------------------------------------------
# the grid
# --------------------------------------------------------------------------

def test_grid_start_snaps_to_the_interval_boundary():
    assert grid_start(1000.0, 900) == 900.0
    assert grid_start(1799.0, 900) == 900.0
    assert grid_start(1800.0, 900) == 1800.0


def test_grid_start_is_idempotent():
    for ts in (0.0, 900.0, 12345.0):
        s = grid_start(ts, 900)
        assert grid_start(s, 900) == s


def test_a_zero_interval_does_not_divide_by_zero():
    assert grid_start(123.0, 0) == 123.0


# --------------------------------------------------------------------------
# building a candle from fills
# --------------------------------------------------------------------------

def test_a_candle_is_assembled_from_trades():
    b = CandleBuilder(900)
    b.add(tr(100.0, 1.0, "buy", 900.0))
    b.add(tr(105.0, 2.0, "buy", 910.0))
    b.add(tr(98.0, 1.0, "sell", 920.0))
    b.add(tr(102.0, 1.0, "buy", 930.0))

    c = b.candle(now=940.0)
    assert (c.open, c.high, c.low, c.close) == (100.0, 105.0, 98.0, 102.0)
    assert c.volume == 5.0 and c.trades == 4
    assert c.start_ts == 900.0


def test_the_candle_starts_on_the_grid_not_at_the_first_trade():
    b = CandleBuilder(900)
    b.add(tr(100.0, ts=1234.0))          # mid-bar
    assert b.candle(now=1240.0).start_ts == 900.0


def test_buy_and_sell_notional_are_split():
    b = CandleBuilder(900)
    b.add(tr(100.0, 3.0, "buy", 900.0))
    b.add(tr(100.0, 1.0, "sell", 905.0))
    c = b.candle(now=910.0)
    assert c.buy_notional == pytest.approx(300.0)
    assert c.sell_notional == pytest.approx(100.0)
    assert c.delta == pytest.approx(200.0)
    assert c.lean == pytest.approx(0.5)


def test_lean_of_an_empty_candle_is_flat_not_a_crash():
    c = LiveCandle(interval_s=900, start_ts=0, open=1, high=1, low=1, close=1)
    assert c.lean == 0.0 and c.delta == 0.0


def test_zero_size_and_zero_price_fills_are_ignored():
    b = CandleBuilder(900)
    assert b.add(tr(0.0, 1.0, ts=900.0)) is None
    assert b.add(tr(100.0, 0.0, ts=900.0)) is None
    assert b.current is None


# --------------------------------------------------------------------------
# rollover
# --------------------------------------------------------------------------

def test_a_trade_past_the_boundary_closes_the_candle():
    b = CandleBuilder(900)
    b.add(tr(100.0, 1.0, "buy", 900.0))
    closed = b.add(tr(110.0, 1.0, "buy", 1800.0))

    assert closed is not None
    assert closed.start_ts == 900.0 and closed.close == 100.0
    assert b.current.start_ts == 1800.0


def test_a_bar_opened_by_a_trade_opens_at_that_trade():
    """The venue stamps a candle's open as its first fill, not as the previous
    bar's close — the two differ across a gap, and using the wrong one shifts
    every change and position figure on that bar."""
    b = CandleBuilder(900)
    b.add(tr(100.0, 1.0, "buy", 900.0))
    b.add(tr(110.0, 1.0, "buy", 1800.0))
    assert b.current.open == 110.0
    assert b.current.high == b.current.low == 110.0


def test_a_bar_opened_by_the_clock_opens_flat_at_the_last_price():
    """No trades means no first fill, so the bar opens where price was."""
    b = CandleBuilder(900)
    b.add(tr(100.0, 1.0, "buy", 900.0))
    c = b.candle(now=1900.0)
    assert c.open == c.close == 100.0
    assert c.trades == 0


def test_a_quiet_market_still_rolls_on_the_clock():
    """No trade arrives to trigger the roll. Without a clock-driven roll the
    candle reports 100% elapsed for as long as the silence lasts."""
    b = CandleBuilder(900)
    b.add(tr(100.0, 1.0, "buy", 900.0))

    c = b.candle(now=3700.0)             # two boundaries later
    assert c.start_ts == 3600.0
    assert c.open == c.close == 100.0
    assert c.trades == 0


def test_rolling_more_than_one_boundary_lands_on_the_right_bar():
    b = CandleBuilder(900)
    b.add(tr(100.0, 1.0, "buy", 900.0))
    closed = b.add(tr(120.0, 1.0, "buy", 5400.0))
    assert closed.start_ts == 900.0
    assert b.current.start_ts == 5400.0


def test_closed_candles_are_kept_in_order():
    b = CandleBuilder(60)
    for i in range(5):
        b.add(tr(100.0 + i, 1.0, "buy", i * 60.0))
    assert [c.start_ts for c in b.closed] == [0.0, 60.0, 120.0, 180.0]


def test_closed_history_is_capped():
    b = CandleBuilder(60)
    b.max_closed = 3
    for i in range(10):
        b.add(tr(100.0, 1.0, "buy", i * 60.0))
    assert len(b.closed) == 3


def test_on_close_fires_once_per_completed_candle():
    seen = []
    b = CandleBuilder(900, on_close=seen.append)
    b.add(tr(100.0, 1.0, "buy", 900.0))
    b.add(tr(101.0, 1.0, "buy", 1800.0))
    b.add(tr(102.0, 1.0, "buy", 2700.0))
    assert [c.start_ts for c in seen] == [900.0, 1800.0]


def test_a_throwing_callback_does_not_break_the_feed():
    def boom(_c):
        raise RuntimeError("callback exploded")

    b = CandleBuilder(900, on_close=boom)
    b.add(tr(100.0, 1.0, "buy", 900.0))
    b.add(tr(101.0, 1.0, "buy", 1800.0))      # must not raise
    assert b.current.start_ts == 1800.0


# --------------------------------------------------------------------------
# out-of-order fills
# --------------------------------------------------------------------------

def test_a_fill_from_a_closed_bar_is_dropped():
    """A replayed fill must not put a ten-minute-old price into the current
    bar's high or low."""
    b = CandleBuilder(900)
    b.add(tr(100.0, 1.0, "buy", 1800.0))
    before = (b.current.high, b.current.low, b.current.trades)

    b.add(tr(500.0, 1.0, "buy", 900.0))       # from the previous bar
    assert (b.current.high, b.current.low, b.current.trades) == before


# --------------------------------------------------------------------------
# the unknown open
# --------------------------------------------------------------------------

def test_a_builder_that_starts_mid_candle_marks_the_open_as_unobserved():
    b = CandleBuilder(900)
    b.add(tr(100.0, 1.0, "buy", 1200.0))      # bar opened at 900, we joined late
    assert b.candle(now=1300.0).seeded is False


def test_seeding_supplies_the_real_open():
    b = CandleBuilder(900)
    b.seed(Candle(ts=900.0, open=95.0, high=101.0, low=94.0, close=100.0,
                  volume=12.0))
    c = b.candle(now=1000.0)
    assert c.seeded is True
    assert c.open == 95.0 and c.high == 101.0


def test_trades_after_a_seed_extend_the_seeded_candle():
    b = CandleBuilder(900)
    b.seed(Candle(ts=900.0, open=95.0, high=101.0, low=94.0, close=100.0,
                  volume=12.0))
    b.add(tr(110.0, 1.0, "buy", 1000.0))
    c = b.candle(now=1010.0)
    assert c.open == 95.0            # the seeded open survives
    assert c.high == 110.0           # the new trade extends the range
    assert c.close == 110.0


def test_a_candle_opened_by_a_rollover_is_seeded():
    b = CandleBuilder(900)
    b.add(tr(100.0, 1.0, "buy", 1200.0))      # unseeded
    b.add(tr(101.0, 1.0, "buy", 1800.0))      # rolled — open is a real close
    assert b.current.seeded is True


# --------------------------------------------------------------------------
# candle geometry passed downstream
# --------------------------------------------------------------------------

def test_elapsed_is_clamped_to_the_interval():
    b = CandleBuilder(900)
    b.add(tr(100.0, 1.0, "buy", 900.0))
    c = b.current
    assert c.elapsed(now=1350.0) == pytest.approx(450.0)
    assert c.elapsed(now=99_999.0) == 900.0
    assert c.elapsed(now=0.0) == 0.0


def test_conversion_to_a_structure_candle_keeps_the_numbers():
    b = CandleBuilder(900)
    b.add(tr(100.0, 2.0, "buy", 900.0))
    b.add(tr(104.0, 1.0, "sell", 930.0))
    c = b.candle(now=940.0).to_candle()
    assert isinstance(c, Candle)
    assert (c.open, c.high, c.low, c.close) == (100.0, 104.0, 100.0, 104.0)
    assert c.volume == 3.0


def test_history_excludes_the_forming_candle_by_default():
    b = CandleBuilder(900)
    b.add(tr(100.0, 1.0, "buy", 900.0))
    b.add(tr(101.0, 1.0, "buy", 1800.0))
    assert len(b.history()) == 1
    assert len(b.history(include_current=True)) == 2


# --------------------------------------------------------------------------
# the feed
# --------------------------------------------------------------------------

def test_a_feed_builds_every_interval_it_was_asked_for():
    f = LiveFeed("BTC", intervals=("1m", "15m"))
    assert sorted(f.builders) == ["15m", "1m"]
    assert f.builders["15m"].interval_s == 900


def test_an_unknown_interval_is_ignored_rather_than_crashing():
    f = LiveFeed("BTC", intervals=("15m", "7m"))
    assert sorted(f.builders) == ["15m"]


def test_feed_routes_trades_into_the_tape_and_every_builder():
    f = LiveFeed("BTC", intervals=("1m", "5m"))
    f._handle_trades([
        {"px": "100", "sz": "2", "side": "B", "time": 60_000},
        {"px": "101", "sz": "1", "side": "A", "time": 62_000},
    ])
    assert f.trades_seen == 2
    assert len(f.tape) == 2
    assert f.tape.cvd == pytest.approx(200.0 - 101.0)
    for name in ("1m", "5m"):
        assert f.builders[name].current.trades == 2


def test_feed_ignores_unparseable_fills():
    f = LiveFeed("BTC", intervals=("1m",))
    f._handle_trades([{"px": "bad"}, {}, {"px": "100", "sz": "1",
                                          "side": "B", "time": 60_000}])
    assert f.trades_seen == 1


def test_feed_stores_book_updates():
    f = LiveFeed("BTC")
    f._handle_book({"levels": [[{"px": "99", "sz": "1"}],
                               [{"px": "101", "sz": "1"}]],
                    "time": 1000})
    assert f.book is not None
    assert f.book.best_bid == 99.0 and f.book.best_ask == 101.0
    assert f.book_updates == 1


def test_an_empty_book_update_is_not_stored():
    f = LiveFeed("BTC")
    f._handle_book({"levels": [[], []]})
    assert f.book is None and f.book_updates == 0


def test_a_silent_feed_is_reported_stale_rather_than_quiet():
    """A socket that stopped delivering looks exactly like a market where
    nothing is trading unless the age is shown."""
    f = LiveFeed("BTC")
    assert f.stale is True and f.age is None

    f.last_msg_ts = time.time()
    assert f.stale is False

    f.last_msg_ts = time.time() - 600
    assert f.stale is True
    assert f.age > 500


def test_status_reports_enough_to_tell_working_from_silent():
    f = LiveFeed("BTC", intervals=("1m",))
    s = f.status()
    assert set(s) >= {"coin", "running", "connected", "stale", "age_s",
                      "trades_seen", "book_updates", "reconnects"}
    assert s["running"] is False


def test_seeding_a_feed_adopts_the_in_progress_bar():
    f = LiveFeed("BTC", intervals=("15m",))
    now = time.time()
    start = grid_start(now, 900)
    bars = [Candle(ts=start - 900 * i, open=10.0, high=11.0, low=9.0,
                   close=10.5, volume=1.0) for i in range(5, 0, -1)]
    bars.append(Candle(ts=start, open=20.0, high=21.0, low=19.0, close=20.5,
                       volume=2.0))

    f.seed("15m", bars)
    c = f.candle("15m")
    assert c is not None and c.open == 20.0 and c.seeded


def test_seeding_with_only_closed_bars_leaves_no_current_candle():
    f = LiveFeed("BTC", intervals=("15m",))
    old = grid_start(time.time(), 900) - 900 * 10
    bars = [Candle(ts=old + 900 * i, open=1.0, high=2.0, low=0.5, close=1.5,
                   volume=1.0) for i in range(4)]
    f.seed("15m", bars)
    assert f.candle("15m") is None
    assert len(f.builders["15m"].closed) == 4


def test_stop_is_safe_on_a_feed_that_never_started():
    LiveFeed("BTC").stop()          # must not raise


def test_update_listeners_are_called_and_a_broken_one_is_survivable():
    f = LiveFeed("BTC", intervals=("1m",))
    seen = []
    f.on_update(lambda ch: seen.append(ch))
    f.on_update(lambda ch: (_ for _ in ()).throw(RuntimeError("bad listener")))

    f._on_message(None, '{"channel":"trades","data":[{"px":"100","sz":"1",'
                        '"side":"B","time":60000}]}')
    assert seen == ["trades"]


def test_malformed_socket_payloads_are_dropped():
    f = LiveFeed("BTC", intervals=("1m",))
    for bad in ("not json", "", "[]", '{"channel":"other"}'):
        f._on_message(None, bad)
    assert f.trades_seen == 0


def test_seeded_history_is_converted_not_mixed_in():
    """`closed` holds LiveCandles. Dropping fetched Candles in alongside them
    only fails later, in whatever reads the list."""
    f = LiveFeed("BTC", intervals=("15m",))
    start = grid_start(time.time(), 900)
    bars = [Candle(ts=start - 900 * i, open=1.0, high=2.0, low=0.5, close=1.5,
                   volume=1.0) for i in range(4, 0, -1)]
    f.seed("15m", bars)

    assert all(isinstance(c, LiveCandle) for c in f.builders["15m"].closed)
    hist = f.history("15m")
    assert hist and all(isinstance(c, Candle) for c in hist)


def test_converted_history_lands_on_the_grid():
    c = Candle(ts=1234.0, open=1.0, high=2.0, low=0.5, close=1.5, volume=1.0)
    lc = LiveCandle.from_candle(c, 900)
    assert lc.start_ts == 900.0 and lc.seeded


# --------------------------------------------------------------------------
# absorption without a level watch, and candle-scoped reading
# --------------------------------------------------------------------------

def _train_feed(f, bps_per_million=10.0, buckets=45, start=4000.0):
    """Teach the feed's baseline that $1m of aggression normally moves price
    `bps_per_million`. Flow and its resulting move must land in the SAME
    bucket or the baseline learns that a million dollars moves nothing."""
    ts, px = 0.0, start
    for _ in range(buckets):
        target = px * (1 + bps_per_million / 10_000.0)
        for k in range(5):
            p = px + (target - px) * k / 4.0
            f._handle_trades([{"px": str(p), "sz": str(200_000 / p),
                               "side": "B", "time": int((ts + k) * 1000)}])
        px = target
        ts += f.bucket_s + 1
    f._handle_trades([{"px": str(px), "sz": "0.000001", "side": "B",
                       "time": int(ts * 1000)}])
    return ts, px


def test_absorption_needs_no_level_watch():
    """It used to require pointing a LevelWatch at a price, so the panel said
    'flow and absorption missing' while the tape was plainly running."""
    f = LiveFeed("BTC", intervals=("15m",))
    ts, px = _train_feed(f)
    assert f.baseline.ready

    base = ts + 5000
    for i in range(12):
        f._handle_trades([{"px": str(px), "sz": str(1_000_000 / px),
                           "side": "B", "time": int((base + i) * 1000)}])

    a = f.absorption(window_s=120.0)
    assert a is not None
    assert a.confident
    assert a.direction == "buy"
    assert a.absorbing, "heavy buying with no move is absorption"


def test_an_untrained_feed_says_it_is_not_confident_rather_than_guessing():
    f = LiveFeed("BTC", intervals=("15m",))
    f._handle_trades([{"px": "100", "sz": "1", "side": "B", "time": 1000}])
    a = f.absorption()
    assert a is not None and not a.confident


def test_status_reports_baseline_progress():
    f = LiveFeed("BTC", intervals=("15m",))
    s = f.status()
    assert s["baseline_samples"] == 0
    assert s["absorption_ready"] is False


def test_an_interval_can_be_added_after_the_feed_started():
    """The timeframe selector can name any interval. A feed opened on a
    different set had nothing to offer, which looked like a dead socket."""
    f = LiveFeed("BTC", intervals=("1m",))
    assert f.candle("30m") is None

    assert f.ensure_interval("30m") is True
    assert f.ensure_interval("30m") is False      # already there
    assert f.ensure_interval("7m") is False       # not a real interval

    f._handle_trades([{"px": "100", "sz": "2", "side": "B",
                       "time": int(time.time() * 1000)}])
    assert f.candle("30m") is not None


def test_a_new_interval_can_be_seeded_with_history():
    f = LiveFeed("BTC", intervals=("1m",))
    start = grid_start(time.time(), 1800)
    bars = [Candle(ts=start - 1800 * i, open=10.0, high=11.0, low=9.0,
                   close=10.5, volume=1.0) for i in range(4, 0, -1)]
    bars.append(Candle(ts=start, open=20.0, high=21.0, low=19.0, close=20.5,
                       volume=2.0))
    f.ensure_interval("30m", bars)
    c = f.candle("30m")
    assert c is not None and c.open == 20.0 and c.seeded


def test_the_spread_series_shows_what_the_book_is_doing_not_just_its_shape():
    """A snapshot says what the book looks like. The series says what is
    happening to it, which is the part you can trade."""
    f = LiveFeed("BTC")
    # Levels must sit inside the 25bps depth band or both sides read zero
    # and the imbalance is 0 — correct, but it makes a wide fixture useless.
    for i in range(10):
        wide = 0.02 + i * 0.01
        f._handle_book({"levels": [[{"px": str(100 - wide), "sz": "5"}],
                                   [{"px": str(100 + wide), "sz": "1"}]],
                        "time": 1000 + i})

    t = f.spread_trend(window_s=3600.0)
    assert t["samples"] == 10
    assert t["spread_bps"] > t["spread_avg"]
    assert t["spread_widening"] is True
    assert t["imbalance"] > 0            # bid-heavy throughout


def test_spread_trend_on_an_empty_book_history_is_safe():
    assert LiveFeed("BTC").spread_trend()["samples"] == 0
