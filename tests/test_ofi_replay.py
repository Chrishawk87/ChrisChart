"""A month of MBP-10 down to one row a second.

The two halves must come out of the SAME pass. Candles say where
price went, OFI says which way the book was leaning when it went
there, and a backtest needs them on one clock. Built from two files
they can disagree about which second a print belongs to, and every
result after that measures the join rather than the market.
"""

from __future__ import annotations

import csv
import importlib.util
import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "ofi_replay", ROOT / "scripts" / "ofi_replay.py")
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)

PT = 1_000_000_000
TICK = PT // 4
BASE = 5800 * PT
T0 = 1_700_000_000


class _L:
    def __init__(self, b, bs, a, asz):
        self.bid_px, self.bid_sz = b, bs
        self.ask_px, self.ask_sz = a, asz


class _MBP10:
    """MBP10Msg: ten levels AND an action -- book and tape together."""

    def __init__(self, i, sec, px=BASE, bid_sz=100, ask_sz=100,
                 trade=None, frac=0):
        self.ts_event = int((T0 + sec) * 1e9) + frac
        self.sequence, self.instrument_id = i, 42
        if trade is not None:
            sz, buy = trade
            self.action, self.side = "T", ("B" if buy else "A")
            self.price, self.size = px, sz
        else:
            self.action, self.side = "A", "B"
        self.levels = [_L(px, bid_sz, px + TICK, ask_sz)]


def run(records):
    """Replay a list of records, returning the rows as dicts."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=rp.COLS, extrasaction="ignore")
    w.writeheader()
    real = rp.dbn.open_file
    rp.dbn.open_file = lambda p: records
    try:
        seen, rows = rp.replay(["x.dbn"], w)
    finally:
        rp.dbn.open_file = real
    buf.seek(0)
    return seen, rows, list(csv.DictReader(buf))


class TestOneRowPerSecond:

    def test_records_in_one_second_become_one_row(self):
        recs = [_MBP10(i, 0, frac=i * 1_000_000) for i in range(50)]
        seen, rows, out = run(recs)
        assert seen == 50
        assert rows == 1 and len(out) == 1

    def test_each_second_gets_its_own_row(self):
        recs = [_MBP10(i, i, bid_sz=100 + i) for i in range(6)]
        _, rows, out = run(recs)
        assert rows == 6
        assert [int(r["ts"]) - T0 for r in out] == [0, 1, 2, 3, 4, 5]

    def test_rows_come_out_in_time_order(self):
        recs = [_MBP10(i, i) for i in range(20)]
        _, _, out = run(recs)
        stamps = [int(r["ts"]) for r in out]
        assert stamps == sorted(stamps)


class TestGapsStayGaps:
    """A second where nothing happened is ABSENT, not zero.

    Writing a zero row there hands the backtest a flat bar that never
    existed -- and a flat bar is not a neutral observation, it is a
    claim that price did not move during a second the market was shut.
    """

    def test_a_quiet_second_is_not_written(self):
        recs = [_MBP10(1, 0, bid_sz=100), _MBP10(2, 5, bid_sz=120)]
        _, rows, out = run(recs)
        assert rows == 2, "only the two seconds that had records"
        assert [int(r["ts"]) - T0 for r in out] == [0, 5]

    def test_the_gap_is_visible_as_a_jump_in_the_index(self):
        recs = [_MBP10(1, 0), _MBP10(2, 3600)]
        _, _, out = run(recs)
        gap = int(out[1]["ts"]) - int(out[0]["ts"])
        assert gap == 3600, "an hour of closed market, readable as one"


class TestBothHalvesOnOneClock:

    def test_a_trade_and_the_book_land_in_the_same_row(self):
        recs = [_MBP10(1, 0, bid_sz=100),
                _MBP10(2, 0, bid_sz=140, trade=(5, True), frac=1)]
        _, _, out = run(recs)
        assert len(out) == 1
        row = out[0]
        assert float(row["volume"]) == 5, "the tape half"
        assert int(row["events"]) >= 1, "the book half"

    def test_a_second_with_book_movement_but_no_trade_still_counts(self):
        """"Nothing traded while the book pulled" is an observation,
        not an absence."""
        recs = [_MBP10(1, 0, bid_sz=100), _MBP10(2, 1, bid_sz=400)]
        _, rows, out = run(recs)
        assert rows == 2
        assert int(out[1]["trades"]) == 0
        assert int(out[1]["events"]) == 1
        assert int(out[1]["ofi"]) != 0, "the book moved and it was measured"

    def test_ofi_is_summed_over_the_second(self):
        recs = [_MBP10(1, 0, bid_sz=100),
                _MBP10(2, 1, bid_sz=200, frac=0),
                _MBP10(3, 1, bid_sz=300, frac=1)]
        _, _, out = run(recs)
        # two events in second 1: +100 then +100
        assert int(out[1]["ofi"]) == 200


class TestTheCandleIsRight:

    def _sec(self, prices):
        recs = [_MBP10(1, 0, px=BASE, bid_sz=100)]
        for i, p in enumerate(prices):
            recs.append(_MBP10(i + 2, 1, px=p, trade=(1, True),
                               frac=i * 1_000_000))
        _, _, out = run(recs)
        return out[-1]

    def test_open_is_the_first_print_not_the_lowest(self):
        row = self._sec([BASE + TICK, BASE, BASE + 2 * TICK])
        assert float(row["open"]) == 5800.25

    def test_close_is_the_last_print_not_the_highest(self):
        row = self._sec([BASE + 2 * TICK, BASE + TICK, BASE])
        assert float(row["close"]) == 5800.00

    def test_high_is_the_highest(self):
        row = self._sec([BASE, BASE + 3 * TICK, BASE + TICK])
        assert float(row["high"]) == 5800.75

    def test_low_is_the_lowest(self):
        row = self._sec([BASE + 3 * TICK, BASE, BASE + TICK])
        assert float(row["low"]) == 5800.00

    def test_volume_adds_up(self):
        recs = [_MBP10(1, 0, bid_sz=100)]
        for i in range(4):
            recs.append(_MBP10(i + 2, 1, trade=(3, True), frac=i))
        _, _, out = run(recs)
        assert float(out[-1]["volume"]) == 12
        assert int(out[-1]["trades"]) == 4


class TestDelta:

    def _delta(self, buys, sells):
        recs = [_MBP10(1, 0, bid_sz=100)]
        n = 2
        for _ in range(buys):
            recs.append(_MBP10(n, 1, trade=(1, True), frac=n)); n += 1
        for _ in range(sells):
            recs.append(_MBP10(n, 1, trade=(1, False), frac=n)); n += 1
        _, _, out = run(recs)
        return float(out[-1]["delta"])

    def test_buyers_paying_is_positive(self):
        assert self._delta(5, 1) > 0

    def test_sellers_paying_is_negative(self):
        assert self._delta(1, 5) < 0

    def test_balanced_is_zero(self):
        assert self._delta(3, 3) == pytest.approx(0.0)


class TestPrices:

    def test_decoded_by_division(self):
        """6,329 of 32,001 real ES ticks decode inexactly when the
        raw integer is MULTIPLIED by 1e-9, and those same ticks then
        fail to snap to their own grid."""
        recs = [_MBP10(1, 0, px=BASE, bid_sz=100)]
        _, _, out = run(recs)
        assert float(out[0]["bid"]) == 5800.00
        assert float(out[0]["ask"]) == 5800.25

    def test_every_price_sits_on_the_es_tick(self):
        recs = [_MBP10(i, i, px=BASE + i * TICK, bid_sz=100 + i)
                for i in range(40)]
        _, _, out = run(recs)
        for r in out:
            for k in ("bid", "ask"):
                cents = round(float(r[k]) * 100)
                assert cents % 25 == 0, f"{k}={r[k]} is off the grid"

    def test_the_spread_is_reported_in_ticks(self):
        recs = [_MBP10(1, 0, px=BASE, bid_sz=100)]
        _, _, out = run(recs)
        assert float(out[0]["spread"]) == pytest.approx(1.0)


class TestItStaysSmall:

    def test_a_row_is_small_enough_for_a_month(self):
        """1.8M seconds a month. At 200 bytes a row that is 360MB and
        does not upload; the format has to stay lean."""
        recs = [_MBP10(i, i, px=BASE + (i % 4) * TICK,
                       bid_sz=100 + i, trade=(2, True))
                for i in range(50)]
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=rp.COLS, extrasaction="ignore")
        w.writeheader()
        real = rp.dbn.open_file
        rp.dbn.open_file = lambda p: recs
        try:
            rp.replay(["x.dbn"], w)
        finally:
            rp.dbn.open_file = real
        body = buf.getvalue().splitlines()[1:]
        per_row = sum(len(x) for x in body) / len(body)
        assert per_row < 120, f"{per_row:.0f} bytes a row is too fat"


class TestItPlacesNothing:

    def test_no_order_code(self):
        src = (ROOT / "scripts" / "ofi_replay.py").read_text()
        for word in ("def submit", "def execute", "def place", "def order"):
            assert word not in src, word


class TestNoFabricatedNumbers:
    """A zero is a number. A backtest cannot tell a fabricated 0.00
    from a real quote, and will happily trade against it.
    """

    def test_a_second_with_no_print_has_no_price(self):
        recs = [_MBP10(1, 0, bid_sz=100), _MBP10(2, 1, bid_sz=400)]
        _, _, out = run(recs)
        row = out[1]
        assert int(row["trades"]) == 0
        for k in ("open", "high", "low", "close"):
            assert row[k] == "", f"{k} invented a price of {row[k]!r}"

    def test_but_the_book_is_still_there(self):
        """The book DID exist in that second. Only the print did not."""
        recs = [_MBP10(1, 0, bid_sz=100), _MBP10(2, 1, bid_sz=400)]
        _, _, out = run(recs)
        assert float(out[1]["bid"]) == 5800.00
        assert float(out[1]["ask"]) == 5800.25

    def test_a_trades_only_file_has_no_book_columns(self):
        """`trades` carries no levels. Those columns must come back
        empty rather than as a zero-wide spread at a price of zero."""
        class _TRD:
            def __init__(self, i, sec):
                self.ts_event = int((T0 + sec) * 1e9) + i
                self.sequence, self.instrument_id = i, 42
                self.action, self.side = "T", "B"
                self.price, self.size = BASE, 3
        _, _, out = run([_TRD(i, 0) for i in range(3)])
        row = out[0]
        assert float(row["close"]) == 5800.00, "the tape half works"
        for k in ("bid", "ask", "spread", "bid_sz", "ask_sz"):
            assert row[k] == "", f"{k} invented {row[k]!r} with no book"

    def test_the_first_record_of_a_file_is_not_thrown_away(self):
        """It carries a real book, but can produce no OFI event --
        there is nothing behind it to difference against."""
        _, rows, out = run([_MBP10(1, 0, bid_sz=100)])
        assert rows == 1
        assert float(out[0]["bid"]) == 5800.00
        assert int(out[0]["events"]) == 0, "no event, but still a book"


class TestTheOutputIsInTimeOrder:
    """Databento files are named for a start date but can run for
    days, so consecutive downloads overlap at the seam. A real case:
    a file named 2026-09-21 covered 95 hours and ran into Sep 25,
    where the next file began.

    One backward step is enough for the backtest to refuse the whole
    file -- `resolve` walks forward assuming the index is ordered --
    so the replay has to produce order, not just report disorder.
    """

    def test_an_overlapping_second_is_not_written_twice(self):
        a = [_MBP10(i, i, bid_sz=100 + i) for i in range(10)]
        b = [_MBP10(i, 8 + i, bid_sz=200 + i) for i in range(10)]
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=rp.COLS,
                           extrasaction="ignore")
        w.writeheader()
        real = rp.dbn.open_file
        files = {"a": a, "b": b}
        rp.dbn.open_file = lambda p: files[p]
        try:
            rp.replay(["a", "b"], w)
        finally:
            rp.dbn.open_file = real
        buf.seek(0)
        stamps = [int(r["ts"]) for r in csv.DictReader(buf)]
        assert stamps == sorted(stamps), "went backwards"
        assert len(stamps) == len(set(stamps)), "wrote a second twice"

    def test_files_globbed_out_of_order_still_come_out_ordered(self):
        """The shell sorts by name. Names do not sort by time."""
        late = [_MBP10(i, 500 + i, bid_sz=100 + i) for i in range(10)]
        early = [_MBP10(i, i, bid_sz=100 + i) for i in range(10)]
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=rp.COLS,
                           extrasaction="ignore")
        w.writeheader()
        real = rp.dbn.open_file
        files = {"z_late": late, "a_early": early}
        rp.dbn.open_file = lambda p: files[p]
        try:
            rp.replay(["z_late", "a_early"], w)   # worst order
        finally:
            rp.dbn.open_file = real
        buf.seek(0)
        stamps = [int(r["ts"]) for r in csv.DictReader(buf)]
        assert stamps == sorted(stamps), stamps[:5]

    def test_the_backtest_accepts_what_the_replay_produces(self):
        """The two halves have to agree about what ordered means."""
        from liqmap import ofi_backtest as _bt
        a = [_MBP10(i, i, bid_sz=100 + i) for i in range(10)]
        b = [_MBP10(i, 8 + i, bid_sz=200 + i) for i in range(10)]
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=rp.COLS,
                           extrasaction="ignore")
        w.writeheader()
        real = rp.dbn.open_file
        files = {"a": a, "b": b}
        rp.dbn.open_file = lambda p: files[p]
        try:
            rp.replay(["a", "b"], w)
        finally:
            rp.dbn.open_file = real
        secs = _bt.Seconds()
        buf.seek(0)
        for row in csv.DictReader(buf):
            secs.add(row["ts"], _bt.to_ticks(row.get("high")),
                     _bt.to_ticks(row.get("low")),
                     _bt.to_ticks(row.get("close")),
                     int(row.get("ofi") or 0))
        assert secs.out_of_order == 0

    def test_the_seam_is_reported_to_the_second(self):
        """A date alone renders a same-day overlap as
        '2026-09-25 < 2026-09-25', which reads as a broken check."""
        assert ":" in rp._when(1_700_000_061.0)
