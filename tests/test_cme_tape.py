"""CME prints into candles, and the clock that decides when a bar ends.

1.  THE CLOCK. `CandleBuilder.candle(None)` rolls to the WALL clock.
    Live that is right; replaying a file it closes every candle and
    returns an empty one stamped today. The tape's own last timestamp
    is the default so both paths agree.
2.  TRADES ONLY. Most records are adds and cancels. Treating one as a
    trade would invent volume that never printed.
3.  THE GRID. Candles floor to interval boundaries, so a 15m bar
    starts where the exchange's does.
4.  AGGRESSOR SIDE. CME states who crossed; the split must follow it
    rather than an uptick rule.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liqmap.cme_tape import Tape, label                     # noqa: E402

PT = 1_000_000_000
TICK = PT // 4
BASE = 5800 * PT
T0 = 1_700_000_000.0


class Lv:
    def __init__(s, bp, bs, ap, asz):
        s.bid_px, s.bid_sz, s.ask_px, s.ask_sz = bp, bs, ap, asz


class Rec:
    def __init__(s, a, px, sz, side, ts, lv=None):
        s.action, s.price, s.size, s.side = a, px, sz, side
        s.ts_event = int(ts * 1e9)
        s.instrument_id, s.sequence = 7, 1
        s.levels = lv if lv is not None else [Lv(BASE, 50, BASE + TICK, 40)]


def trade(px, sz, side, ts):
    return Rec("T", px, sz, side, ts)


def quote(ts):
    return Rec("A", 0, 0, "B", ts)


class TestLabels:

    def test_seconds_to_names(self):
        assert label(60) == "1m"
        assert label(300) == "5m"
        assert label(900) == "15m"
        assert label(3600) == "1h"
        assert label(86400) == "1d"
        assert label(45) == "45s"


class TestOffer:

    def test_a_trade_is_taken(self):
        t = Tape()
        assert t.offer(trade(BASE, 3, "B", T0)) is True
        assert t.trades == 1

    def test_a_quote_is_not_a_trade(self):
        t = Tape()
        assert t.offer(quote(T0)) is False
        assert t.trades == 0

    def test_a_quote_still_updates_the_book(self):
        t = Tape()
        t.offer(quote(T0))
        assert t.book is not None
        assert t.books == 1

    def test_every_record_is_counted(self):
        t = Tape()
        for i in range(5):
            t.offer(quote(T0 + i))
        t.offer(trade(BASE, 1, "B", T0 + 5))
        assert t.records == 6 and t.trades == 1

    def test_the_last_trade_is_remembered(self):
        t = Tape()
        t.offer(trade(BASE + TICK, 4, "A", T0 + 9))
        assert t.last_px == pytest.approx(5800.25)
        assert t.last_side == "sell"
        assert t.last_ts == pytest.approx(T0 + 9)

    def test_a_quote_does_not_move_the_last_trade(self):
        t = Tape()
        t.offer(trade(BASE, 1, "B", T0))
        t.offer(quote(T0 + 5))
        assert t.last_ts == pytest.approx(T0)


class TestCandles:

    def fill(self, n=900, step=1.0):
        t = Tape()
        random.seed(4)
        px = BASE
        for i in range(n):
            px += random.choice((-TICK, 0, TICK))
            t.offer(trade(px, random.randint(1, 9),
                          random.choice("AB"), T0 + i * step))
        return t

    def test_candles_close_on_the_grid(self):
        """`closed` holds LiveCandle, whose boundary field is
        `start_ts` -- `to_candle()` is what renames it to `ts`."""
        t = self.fill()
        for c in t.closed["1m"]:
            assert c.start_ts % 60 == 0

    def test_fifteen_minutes_of_prints_make_fifteen_bars(self):
        """T0 sits 20s into a minute, so 900 seconds of prints cross
        fifteen boundaries and leaves a sixteenth bar open."""
        t = self.fill(900, 1.0)
        assert len(t.closed["1m"]) == 15
        assert t.candle("1m").volume > 0

    def test_no_closed_candle_is_empty(self):
        t = self.fill()
        assert all(c.volume > 0 for c in t.closed["1m"])

    def test_every_interval_is_built(self):
        t = self.fill()
        assert set(t.closed) == {"1m", "5m", "15m"}
        assert len(t.closed["1m"]) > len(t.closed["5m"])

    def test_the_current_candle_uses_the_tapes_clock(self):
        """The defect this guards: with no `now`, the builder rolls to
        the wall clock, and on replayed data that is years ahead -- so
        every candle closes and the current one comes back empty."""
        t = self.fill()
        cur = t.candle("1m")
        assert cur is not None
        assert cur.volume > 0, "rolled to the wall clock and emptied"
        assert 0 <= t.last_ts - cur.start_ts < 60

    def test_an_explicit_clock_still_wins(self):
        t = self.fill()
        far = t.last_ts + 86400
        cur = t.candle("1m", now=far)
        assert cur is None or cur.volume == 0

    def test_ohlc_brackets_the_prints(self):
        t = Tape()
        for px, ts in ((BASE, T0), (BASE + 8 * TICK, T0 + 1),
                       (BASE - 4 * TICK, T0 + 2), (BASE + TICK, T0 + 3)):
            t.offer(trade(px, 1, "B", ts))
        c = t.candle("1m")
        assert c.high == pytest.approx(5802.0)
        assert c.low == pytest.approx(5799.0)
        assert c.open == pytest.approx(5800.0)
        assert c.close == pytest.approx(5800.25)

    def test_the_buy_sell_split_follows_the_aggressor(self):
        t = Tape()
        t.offer(trade(BASE, 10, "B", T0))        # buy aggressor
        t.offer(trade(BASE, 4, "A", T0 + 1))     # sell aggressor
        c = t.candle("1m")
        assert c.buy_notional > c.sell_notional
        assert c.buy_notional > 0 and c.sell_notional > 0

    def test_roll_closes_a_bar_with_no_trades(self):
        t = Tape()
        t.offer(trade(BASE, 1, "B", T0))
        before = len(t.closed["1m"])
        t.roll(T0 + 180)
        assert len(t.closed["1m"]) > before


class TestSnapshot:

    def test_it_reports_without_any_data(self):
        s = Tape().snapshot()
        assert s["trades"] == 0 and s["current"] is None

    def test_it_carries_the_current_candle(self):
        t = Tape()
        for i in range(30):
            t.offer(trade(BASE + i * TICK, 2, "B", T0 + i))
        s = t.snapshot("1m")
        assert s["current"]["volume"] > 0
        assert s["last_px"] > 0

    def test_the_book_appears_once_seen(self):
        t = Tape()
        t.offer(quote(T0))
        s = t.snapshot()
        assert s["book"]["bid"] > 0 and s["book"]["ask"] > s["book"]["bid"]

    def test_candles_only_when_asked(self):
        t = Tape()
        t.offer(trade(BASE, 1, "B", T0))
        assert "candles" not in t.snapshot("1m")
        assert "candles" in t.snapshot("1m", limit=5)


class TestSuggestsOnly:

    def test_the_module_cannot_place_an_order(self):
        src = (Path(__file__).resolve().parent.parent
               / "liqmap" / "cme_tape.py").read_text()
        for bad in ("def submit", "def execute", "def place", "def order"):
            assert bad not in src

    def test_it_reuses_the_sites_candle_builder(self):
        src = (Path(__file__).resolve().parent.parent
               / "liqmap" / "cme_tape.py").read_text()
        assert "from .live import CandleBuilder" in src
        assert "class CandleBuilder" not in src, "a second implementation"
