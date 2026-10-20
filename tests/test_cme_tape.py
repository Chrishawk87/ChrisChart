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

from liqmap import cme_tape                                 # noqa: E402
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
        assert "import" in src and "CandleBuilder" in src
        assert "from .live import" in src, "not its own copy"
        assert "class CandleBuilder" not in src, "a second implementation"


# --------------------------------------------------------- the ladder

PT_ = 1_000_000_000
TICK_ = 250_000_000
BASE_ = 5800 * PT_


class _BBO:
    """BBOMsg as databento delivers it: a book, and no action at all."""
    def __init__(self, i, ts, bid_sz=120, ask_sz=100, px=BASE_):
        self.ts_event = int(ts * 1e9)
        self.sequence, self.instrument_id = i, 42
        self.action, self.side = "", "N"
        self.levels = [type("L", (), {
            "bid_px": px, "bid_sz": bid_sz,
            "ask_px": px + TICK_, "ask_sz": ask_sz})()]


class _TRD:
    """TradeMsg: a print with an aggressor, and no levels."""
    def __init__(self, i, ts, px=BASE_, sz=4, buy=True):
        self.ts_event = int(ts * 1e9)
        self.sequence, self.instrument_id = i, 42
        self.action = "T"
        self.side = "B" if buy else "A"
        self.price, self.size = px, sz


class TestTheTapeKeepsItsTrades:
    """The candle builders answer "what did this bar do". The ladder
    asks "which side has been paying, over these last N seconds",
    which a closed bar cannot answer for the bar still forming.
    """

    def test_a_print_lands_on_the_flow_tape(self):
        t = cme_tape.Tape(symbol="ES")
        t.offer(_TRD(1, 1_700_000_000.0))
        assert len(t.flow._trades) == 1

    def test_a_book_update_does_not(self):
        t = cme_tape.Tape(symbol="ES")
        t.offer(_BBO(1, 1_700_000_000.0))
        assert t.flow._trades == [], "a quote is not a fill"
        assert t.book is not None, "but it is still a book"

    def test_the_aggressor_survives(self):
        """The only directional information in a fill."""
        t = cme_tape.Tape(symbol="ES")
        t.offer(_TRD(1, 1_700_000_000.0, buy=True))
        t.offer(_TRD(2, 1_700_000_001.0, buy=False))
        assert [x.aggressor for x in t.flow._trades] == ["buy", "sell"]

    def test_it_is_bounded_by_count_not_only_by_age(self):
        """An hour is a promise about time. How many trades fit in an
        hour is the market's decision: measured at 136 bytes a trade,
        an active ES hour is 100-250MB."""
        t = cme_tape.Tape(symbol="ES")
        t.flow.max_trades = 100
        base = 1_700_000_000.0
        for i in range(500):
            t.offer(_TRD(i, base + i * 0.001))      # 0.5s, no age pruning
        assert len(t.flow._trades) == 100

    def test_the_cap_drops_the_oldest(self):
        """Trimmed from the wrong end, a tape reports a stale delta as
        current -- worse than reporting a short one."""
        t = cme_tape.Tape(symbol="ES")
        t.flow.max_trades = 10
        base = 1_700_000_000.0
        for i in range(50):
            t.offer(_TRD(i, base + i * 0.001))
        assert t.flow._trades[-1].ts == pytest.approx(base + 49 * 0.001)
        assert t.flow._trades[0].ts > base

    def test_the_default_tape_is_uncapped(self):
        """The venues already using FlowTape must be untouched."""
        from liqmap.flow import FlowTape, Trade
        tp = FlowTape(max_age=3600.0)
        assert tp.max_trades is None
        for i in range(300):
            tp.add(Trade(px=1.0, sz=1.0, aggressor="buy",
                         ts=1_700_000_000.0 + i * 0.001))
        assert len(tp._trades) == 300


class TestLadderFeed:
    """The ladder only ever touches .tape, .book, .age and .coin.
    Presenting those is the whole adapter -- so the ES ladder and the
    BTC ladder cannot drift apart in what they mean.
    """

    def _fed(self, trades=True, books=True, n=40, running=True):
        t = cme_tape.Tape(symbol="ES")
        base = 1_700_000_000.0
        for i in range(n):
            if books:
                t.offer(_BBO(i, base + i))
            if trades:
                t.offer(_TRD(i, base + i))
        return cme_tape.LadderFeed(t, coin="ES", running=running)

    def test_it_exposes_the_trade_list_the_ladder_walks(self):
        f = self._fed()
        assert hasattr(f.tape, "_trades") and f.tape._trades

    def test_it_exposes_a_book_the_ladder_can_read_at_the_touch(self):
        f = self._fed()
        assert f.book.bids and f.book.asks
        assert f.book.asks[0].px > f.book.bids[0].px

    def test_age_is_none_before_the_first_print(self):
        """Not 0.0 -- a tape that has never ticked has not just
        ticked, and that is the bug that 500'd /api/flow."""
        f = cme_tape.LadderFeed(cme_tape.Tape(symbol="ES"), coin="ES")
        assert f.age is None

    def test_both_halves_present_means_nothing_is_missing(self):
        assert self._fed().missing() == ""

    def test_prints_without_a_book_is_named(self):
        why = self._fed(books=False).missing()
        assert "no book" in why and "trades" in why

    def test_a_book_without_prints_is_named(self):
        why = self._fed(trades=False).missing()
        assert "no prints" in why and "aggressor" in why

    def test_an_engine_that_is_off_says_so_first(self):
        assert "not running" in self._fed(running=False).missing()


class TestTheLadderActuallyBuilds:
    """The point of the adapter: the existing measurement code runs
    unchanged against CME records."""

    def _ladder(self):
        from liqmap import flowladder as fl, orderflow as of
        from liqmap.live import INTERVALS
        t = cme_tape.Tape(symbol="ES")
        now = 1_700_000_000.0
        for i in range(600):
            px = BASE_ + (i // 120) * TICK_
            t.offer(_BBO(i, now - 600 + i, px=px))
            if i % 2 == 0:
                t.offer(_TRD(i, now - 600 + i, px=px, buy=(i % 5 != 0)))
        f = cme_tape.LadderFeed(t, coin="ES", running=True)
        want = fl.wanted(["15m", "5m", "1m"], INTERVALS)
        return fl.build(f.tape, f.book, want, of.SPECS["ES"],
                        target_ticks=10.0, spike=2.0, now=now).to_dict()

    def test_the_rungs_carry_real_deltas(self):
        rows = {r["timeframe"]: r for r in self._ladder()["rungs"]}
        assert rows["1m"]["trades"] > 0
        assert rows["1m"]["delta"] != 0

    def test_the_spread_is_read_against_the_es_tick(self):
        """0.25, from the published spec -- not inferred from a spread
        that happens to be two ticks wide at the moment of the read."""
        book = self._ladder()["book"]
        assert book["spread_ticks"] == pytest.approx(1.0)

    def test_a_tape_shorter_than_the_bar_is_not_called_covered(self):
        """Ten minutes of tape cannot report a 15-minute bar."""
        rows = {r["timeframe"]: r for r in self._ladder()["rungs"]}
        assert rows["15m"]["covered"] is False
        assert rows["1m"]["covered"] is True


class TestTheThreeSilences:
    """A closed market, a dead thread and a wrong schema all draw the
    same empty panel, and each needs a different fix. The message has
    to separate them or it is not worth printing.
    """

    def _feed(self, **kw):
        kw.setdefault("running", True)
        return cme_tape.LadderFeed(kw.pop("src", cme_tape.Tape(symbol="ES")),
                                   coin="ES", **kw)

    def test_an_engine_that_is_off_reports_the_reason_it_is_off(self):
        why = self._feed(running=False,
                         note="LIQMAP_OFI is not set to true").missing()
        assert "not running" in why
        assert "LIQMAP_OFI is not set to true" in why

    def test_a_subscription_with_nothing_on_it_is_not_called_a_fault(self):
        why = self._feed(schemas=("bbo-1s", "trades")).missing()
        assert "no records have arrived" in why
        assert "closed market, not a fault" in why

    def test_it_gives_the_session_hours_to_check_against(self):
        why = self._feed().missing()
        assert "Sunday" in why and "Friday" in why and "Central" in why

    def test_it_names_what_was_subscribed(self):
        why = self._feed(schemas=("bbo-1s", "trades")).missing()
        assert "bbo-1s, trades" in why

    def test_a_thread_error_is_surfaced_even_while_it_still_runs(self):
        why = self._feed(note="BentoError: Not authorized").missing()
        assert "Not authorized" in why

    def test_records_arriving_without_prints_is_a_schema_problem(self):
        """Distinct from silence: the feed is alive and entitled, it
        is just not carrying what the ladder reads."""
        t = cme_tape.Tape(symbol="ES")
        for i in range(5):
            t.offer(_BBO(i, 1_700_000_000.0 + i))
        why = self._feed(src=t, schemas=("bbo-1s",)).missing()
        assert "no prints" in why
        assert "5 records have arrived" in why
        assert "closed market" not in why, "this one IS a fault"

    def test_records_arriving_without_a_book_is_too(self):
        t = cme_tape.Tape(symbol="ES")
        for i in range(5):
            t.offer(_TRD(i, 1_700_000_000.0 + i))
        why = self._feed(src=t, schemas=("trades",)).missing()
        assert "no book" in why and "5 records" in why

    def test_both_halves_still_means_nothing_missing(self):
        t = cme_tape.Tape(symbol="ES")
        for i in range(5):
            t.offer(_BBO(i, 1_700_000_000.0 + i))
            t.offer(_TRD(i, 1_700_000_000.0 + i))
        assert self._feed(src=t).missing() == ""


class TestTheTapeFeedsTheBookDiagonal:
    """One record stream, two readings: what traded (footprint) and
    what is resting (book diagonal). The footprint is not modified to
    make this work -- its own return value already says when a bar
    ended."""

    def _rec(self, i, ts, px, bsz, asz, trade_sz=0, buy=True):
        o = type("R", (), {})()
        o.ts_event = int(ts * 1e9)
        o.sequence, o.instrument_id = i, 42
        if trade_sz:
            o.action, o.side = "T", ("B" if buy else "A")
            o.price, o.size = px, trade_sz
        else:
            o.action, o.side = "A", "B"
        lv = []
        for k in range(5):
            L = type("L", (), {})()
            L.bid_px, L.bid_sz, L.bid_ct = px - k * TICK, bsz, bsz // 10
            L.ask_px, L.ask_sz, L.ask_ct = (px + (k + 1) * TICK, asz,
                                            asz // 10)
            lv.append(L)
        o.levels = lv
        return o

    def _wired(self, contracts=100):
        from liqmap.bookdiag import BookDiagonal
        from liqmap.footprint import VolumeRule
        from liqmap.footprint_live import FootprintStream
        t = cme_tape.Tape(symbol="ES")
        t.footprint = FootprintStream(tick_size=0.25,
                                      rule=VolumeRule(contracts))
        t.bookdiag = BookDiagonal(tick_size=0.25)
        return t

    def test_book_records_reach_the_diagonal(self):
        t = self._wired()
        for i in range(10):
            t.offer(self._rec(i, 1_700_000_000.0 + i, BASE, 400, 100))
        assert t.bookdiag.reading().updates == 10

    def test_it_reads_the_resting_sizes_not_the_traded_ones(self):
        t = self._wired()
        for i in range(6):
            t.offer(self._rec(i, 1_700_000_000.0 + i, BASE, 400, 100,
                              trade_sz=5, buy=False))
        r = t.bookdiag.reading()
        assert r.winning_now() == "bid", "resting bid is the heavy side"
        assert t.footprint.pulse().delta < 0, "but sellers are paying"

    def test_a_closing_bar_restarts_the_accumulation(self):
        t = self._wired(contracts=20)
        base = 1_700_000_000.0
        for i in range(4):
            t.offer(self._rec(i, base + i, BASE, 400, 100, trade_sz=5))
        assert len(t.footprint.history) == 1
        # the accumulation restarted, the standing book did not vanish
        assert t.bookdiag.reading().seconds == 0.0
        assert t.bookdiag.reading().levels

    def test_without_a_diagonal_attached_nothing_changes(self):
        """It is optional. A tape with no book reader must behave
        exactly as it did before this existed."""
        t = cme_tape.Tape(symbol="ES")
        for i in range(5):
            t.offer(self._rec(i, 1_700_000_000.0 + i, BASE, 400, 100,
                              trade_sz=5))
        assert t.trades == 5 and t.books == 5


# ---------------------------------------------------------- on_update / seed

class TestOnUpdate:

    def test_listener_is_called_on_each_trade(self):
        t = Tape()
        called = []
        t.on_update(lambda: called.append(1))
        t.offer(trade(BASE, 1, "B", T0))
        t.offer(trade(BASE + TICK, 2, "A", T0 + 1))
        assert len(called) == 2

    def test_listener_is_not_called_on_book_updates(self):
        t = Tape()
        called = []
        t.on_update(lambda: called.append(1))
        t.offer(quote(T0))
        assert called == []

    def test_listener_can_be_removed(self):
        t = Tape()
        called = []
        fn = lambda: called.append(1)
        t.on_update(fn)
        t.offer(trade(BASE, 1, "B", T0))
        t.off_update(fn)
        t.offer(trade(BASE + TICK, 2, "A", T0 + 1))
        assert len(called) == 1

    def test_removing_unknown_listener_is_silent(self):
        t = Tape()
        t.off_update(lambda: None)  # should not raise

    def test_listener_count_reflects_registered_count(self):
        t = Tape()
        assert t.listener_count == 0
        fn = lambda: None
        t.on_update(fn)
        assert t.listener_count == 1
        t.off_update(fn)
        assert t.listener_count == 0

    def test_a_crashing_listener_does_not_stop_the_tape(self):
        t = Tape()
        good = []
        t.on_update(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        t.on_update(lambda: good.append(1))
        t.offer(trade(BASE, 1, "B", T0))
        assert good == [1]
        assert t.trades == 1


class TestSeedInterval:

    def _candle(self, ts, o=5800.0, h=5801.0, lo=5799.0, c=5800.5, v=100):
        from liqmap.structure import Candle
        return Candle(ts=ts, open=o, high=h, low=lo, close=c, volume=v)

    def test_seed_adds_closed_bars_to_builder(self):
        import time
        t = Tape()
        now = time.time()
        candles = [self._candle(now - 300 * (3 - i)) for i in range(3)]
        t.seed_interval("5m", candles)
        assert len(t.builders["5m"].closed) >= 2

    def test_seed_creates_builder_for_unknown_interval(self):
        t = Tape()
        assert "30m" not in t.builders
        import time
        now = time.time()
        candles = [self._candle(now - 1800 * (3 - i)) for i in range(3)]
        t.seed_interval("30m", candles)
        assert "30m" in t.builders

    def test_seed_does_not_regress_live_data(self):
        """Seeding with old history must not overwrite newer live bars."""
        import time
        t = Tape()
        now = time.time()
        # Feed a live trade so the builder has a closed bar from now.
        ts_live = now - 60
        t.offer(trade(int(BASE), 2, "B", ts_live))
        t.offer(trade(int(BASE + TICK), 1, "A", ts_live + 30))
        closed_before = len(t.builders["1m"].closed)

        # Now seed with older historical bars.
        old_candles = [self._candle(ts_live - 300 * (2 - i)) for i in range(2)]
        t.seed_interval("1m", old_candles)
        # The live closed bar must still be there.
        assert len(t.builders["1m"].closed) >= closed_before

    def test_seed_with_empty_list_is_silent(self):
        t = Tape()
        t.seed_interval("1m", [])  # should not raise

    def test_seed_with_unknown_interval_name_is_silent(self):
        t = Tape()
        t.seed_interval("99x", [self._candle(T0)])  # should not raise
