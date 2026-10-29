"""Closed OFI bars on disk, and the ways persistence lies.

In order of how badly each would mislead:

  a reload that rewrites what it just read. The flusher would see the
  restored bars as new, write them back under the current build's
  idea of the columns, and history would quietly become whatever the
  latest deploy computed rather than what actually happened

  bars coming back in the wrong order. Every candle individually
  correct and the chart drawn backwards is the hardest kind of wrong
  to notice, because nothing on screen looks broken

  a disk write on the tape thread. The record loop runs five builders
  per book update at feed rate, and coupling that to the filesystem
  makes a slow volume look like a slow market

  a database failure taking the chart down with it. Losing history is
  bad; losing the live candles because yesterday's could not be read
  is worse
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from liqmap.ofibar import LADDER, OfiBar, OfiBarBuilder  # noqa: E402
from liqmap.ofistore import (OfiBarRecorder, OfiBarStore,  # noqa: E402
                             KEEP_PER_TARGET)


def builders(targets=LADDER):
    return {t: OfiBarBuilder(tick_size=0.25, threshold=t, max_s=900.0,
                             min_updates=2) for t in targets}


def feed(lad, n=8000, seed=3, t0=0.0):
    """Feed book updates and prints. `t0` continues an earlier clock --
    restarting it at zero makes the new bars collide with the old ones,
    which is a property of the harness and not of the code."""
    import random
    rng = random.Random(seed)
    bid, ts = 20_000, float(t0)
    for _ in range(n):
        if rng.random() < 0.08:
            bid += 1 if rng.random() < 0.5 else -1
        ts += 0.01
        for b in lad.values():
            b.on_trade(ts, rng.randint(1, 20),
                       "buy" if rng.random() < 0.5 else "sell")
            b.on_book(ts, bid, rng.randint(50, 600),
                      bid + 1, rng.randint(50, 600))
    return ts


@pytest.fixture
def store(tmp_path):
    return OfiBarStore(tmp_path / "t.db")


class TestARestartKeepsTheChart:

    def test_bars_come_back_after_a_restart(self, tmp_path):
        lad = builders()
        feed(lad)
        assert len(lad[250].bars) > 100
        rec = OfiBarRecorder(OfiBarStore(tmp_path / "t.db"), lad)
        rec.seed()
        rec.flush()

        fresh = builders()
        assert all(not b.bars for b in fresh.values())
        r2 = OfiBarRecorder(OfiBarStore(tmp_path / "t.db"), fresh)
        restored = r2.load_into_builders()
        assert restored > 0
        for t in LADDER:
            assert len(fresh[t].bars) == len(lad[t].bars), t

    def test_every_column_survives_the_round_trip(self, tmp_path):
        """A bar rebuilt from a formula at load time would be a second,
        fallible copy of something already stated."""
        lad = builders((250,))
        feed(lad)
        rec = OfiBarRecorder(OfiBarStore(tmp_path / "t.db"), lad)
        rec.seed()
        rec.flush()
        fresh = builders((250,))
        OfiBarRecorder(OfiBarStore(tmp_path / "t.db"),
                       fresh).load_into_builders()
        for a, b in zip(lad[250].bars, fresh[250].bars):
            assert a.as_dict() == b.as_dict()

    def test_the_delta_survives_too(self, tmp_path):
        lad = builders((250,))
        feed(lad)
        rec = OfiBarRecorder(OfiBarStore(tmp_path / "t.db"), lad)
        rec.seed()
        rec.flush()
        fresh = builders((250,))
        OfiBarRecorder(OfiBarStore(tmp_path / "t.db"),
                       fresh).load_into_builders()
        got = fresh[250].bars
        assert any(b.trade_count > 0 for b in got)
        assert any(b.delta != 0 for b in got)

    def test_they_come_back_oldest_first(self, tmp_path):
        """Individually correct and drawn backwards is the hardest kind
        of wrong to spot."""
        lad = builders((250,))
        feed(lad)
        rec = OfiBarRecorder(OfiBarStore(tmp_path / "t.db"), lad)
        rec.seed()
        rec.flush()
        got = OfiBarStore(tmp_path / "t.db").recent(250)
        ts = [b.ts_close for b in got]
        assert ts == sorted(ts)

    def test_a_reload_does_not_rewrite_what_it_just_read(self, tmp_path):
        """Otherwise history becomes whatever the latest deploy
        computed rather than what happened."""
        lad = builders((250,))
        feed(lad)
        rec = OfiBarRecorder(OfiBarStore(tmp_path / "t.db"), lad)
        rec.seed()
        first = rec.flush()
        assert first > 0

        fresh = builders((250,))
        r2 = OfiBarRecorder(OfiBarStore(tmp_path / "t.db"), fresh)
        r2.load_into_builders()
        r2.seed()
        assert r2.flush() == 0, "it rewrote the history it just loaded"

    def test_loading_leaves_a_builder_that_already_has_bars_alone(self, tmp_path):
        lad = builders((250,))
        feed(lad)
        rec = OfiBarRecorder(OfiBarStore(tmp_path / "t.db"), lad)
        rec.seed()
        rec.flush()
        n = len(lad[250].bars)
        assert rec.load_into_builders() == 0
        assert len(lad[250].bars) == n


class TestWritingIsIdempotent:

    def test_the_same_bar_twice_is_one_row(self, store):
        b = OfiBar(ts_open=1.0, ts_close=2.0, open=1, high=2, low=0,
                   close=1, direction="green", ofi=250, updates=9)
        assert store.record(250, b) == 1
        assert store.record(250, b) == 0
        assert store.count(250) == 1

    def test_a_second_flush_writes_nothing_new(self, tmp_path):
        lad = builders((250,))
        feed(lad)
        rec = OfiBarRecorder(OfiBarStore(tmp_path / "t.db"), lad)
        rec.seed()
        assert rec.flush() > 0
        assert rec.flush() == 0

    def test_only_new_bars_are_written_after_more_feed(self, tmp_path):
        lad = builders((250,))
        end = feed(lad, n=4000, seed=1)
        rec = OfiBarRecorder(OfiBarStore(tmp_path / "t.db"), lad)
        rec.seed()
        rec.flush()
        on_disk = rec.store.count(250)
        before = len(lad[250].bars)
        feed(lad, n=4000, seed=2, t0=end + 1.0)
        added = len(lad[250].bars) - before
        assert added > 0
        assert rec.flush() == added
        assert rec.store.count(250) == on_disk + added

    def test_an_empty_bar_is_not_written(self, store):
        assert store.record(250, OfiBar()) == 0
        assert store.count() == 0

    def test_targets_do_not_collide(self, store):
        b = OfiBar(ts_open=1.0, ts_close=2.0, open=1, high=2, low=0,
                   close=1, direction="green", ofi=250, updates=9)
        store.record(125, b)
        store.record(250, b)
        assert store.count(125) == 1 and store.count(250) == 1
        assert store.targets() == [125, 250]


class TestTheCap:

    def test_pruning_keeps_the_most_recent(self, store):
        for i in range(40):
            store.record(250, OfiBar(ts_open=float(i), ts_close=float(i) + 1,
                                     open=1, high=1, low=1, close=1,
                                     direction="green", ofi=1, updates=2))
        store.prune(keep=10)
        got = store.recent(250, 100)
        assert len(got) == 10
        assert got[-1].ts_close == 40.0
        assert got[0].ts_close == 31.0

    def test_pruning_is_per_target(self, store):
        for t in (125, 250):
            for i in range(20):
                store.record(t, OfiBar(ts_open=float(i),
                                       ts_close=float(i) + 1, open=1,
                                       high=1, low=1, close=1,
                                       direction="green", ofi=1, updates=2))
        store.prune(keep=5)
        assert store.count(125) == 5 and store.count(250) == 5

    def test_the_cap_matches_what_a_builder_holds(self):
        """A reload should fill the chart to the depth it had."""
        assert KEEP_PER_TARGET == OfiBarBuilder().max_bars

    def test_last_ts_is_zero_on_an_empty_target(self, store):
        assert store.last_ts(2000) == 0.0


class TestItStaysOffTheTapeThread:

    def test_the_recorder_writes_on_its_own_thread(self, tmp_path):
        lad = builders((250,))
        feed(lad, n=3000)
        rec = OfiBarRecorder(OfiBarStore(tmp_path / "t.db"), lad,
                             period_s=0.05)
        rec.seed()
        here = threading.current_thread().name
        rec.start()
        import time
        for _ in range(60):
            if rec.written:
                break
            time.sleep(0.05)
        rec.stop()
        assert rec.written > 0, "the timer never wrote anything"
        assert here == threading.current_thread().name

    def test_the_tape_never_calls_the_store(self):
        """A disk write per book update would couple the feed's
        throughput to the filesystem."""
        src = (ROOT / "liqmap" / "cme_tape.py").read_text()
        for w in ("OfiBarStore", "ofistore", "record_many"):
            assert w not in src, w

    def test_starting_twice_does_not_make_two_threads(self, tmp_path):
        rec = OfiBarRecorder(OfiBarStore(tmp_path / "t.db"),
                             builders((250,)), period_s=5.0)
        rec.start()
        first = rec._thread
        rec.start()
        assert rec._thread is first
        rec.stop()


class TestFailuresDoNotTakeTheChartDown:

    def test_a_broken_store_does_not_raise_out_of_flush(self, tmp_path):
        class Broken(OfiBarStore):
            def record_many(self, target, bars):
                raise RuntimeError("disk is gone")

        lad = builders((250,))
        feed(lad, n=3000)
        rec = OfiBarRecorder(Broken(tmp_path / "t.db"), lad)
        rec.seed()
        assert rec.flush() == 0
        assert rec.errors == 1

    def test_a_broken_store_does_not_raise_out_of_load(self, tmp_path):
        class Broken(OfiBarStore):
            def recent(self, target, limit=10):
                raise RuntimeError("disk is gone")

        lad = builders((250,))
        rec = OfiBarRecorder(Broken(tmp_path / "t.db"), lad)
        assert rec.load_into_builders() == 0
        assert rec.errors == 1
        assert lad[250].bars == []

    def test_the_service_wires_it_up_defensively(self):
        web = (ROOT / "liqmap" / "web.py").read_text()
        i = web.index("from .ofistore import")
        blk = web[i - 400:i + 500]
        assert "try:" in blk and "except Exception" in blk

    def test_status_is_cheap_enough_to_poll(self, tmp_path):
        """Five row counts a second, for a number read as 'roughly how
        much is kept', is not worth it."""
        src = (ROOT / "liqmap" / "ofistore.py").read_text()
        i = src.index("def status")
        assert "for t in self.builders" not in src[i:]


class TestItDecidesNothing:

    def test_no_order_code(self):
        src = (ROOT / "liqmap" / "ofistore.py").read_text()
        for w in ("def submit(", "def execute(", "def place(", "def order("):
            assert w not in src, w

    def test_it_makes_no_claim_about_what_comes_next(self):
        src = (ROOT / "liqmap" / "ofistore.py").read_text().lower()
        assert "nothing here predicts" in src
        for w in ("will hold", "buy signal", "sell signal", "profit"):
            assert w not in src, w


class TestTheClockGoingBackwards:
    """A watermark of 'newest close already written' is the obvious
    design, and it stops writing forever the moment feed time stops
    increasing -- a replay from an archive, a restart from a file, a
    clock correction. Nothing on the chart says why."""

    def test_bars_with_earlier_timestamps_are_still_written(self, tmp_path):
        lad = builders((250,))
        feed(lad, n=4000, seed=1, t0=10_000.0)
        rec = OfiBarRecorder(OfiBarStore(tmp_path / "t.db"), lad)
        rec.seed()
        rec.flush()
        on_disk = rec.store.count(250)

        # A restart that replays an OLDER stretch of tape.
        lad[250].reset()
        feed(lad, n=4000, seed=9, t0=0.0)
        assert lad[250].bars, "the replay should close bars"
        wrote = rec.flush()
        assert wrote > 0, "a watermark would have dropped every one"
        assert rec.store.count(250) == on_disk + wrote

    def test_a_marker_that_aged_off_the_front_re_offers_everything(
            self, tmp_path):
        lad = {250: OfiBarBuilder(tick_size=0.25, threshold=250,
                                  max_s=900.0, min_updates=2,
                                  max_bars=40)}
        feed(lad, n=4000, seed=4)
        rec = OfiBarRecorder(OfiBarStore(tmp_path / "t.db"), lad)
        rec.seed()
        rec.flush()
        # Trim past the marker, then keep going.
        feed(lad, n=4000, seed=5, t0=1_000.0)
        assert lad[250].dropped > 0, "the front should have been trimmed"
        assert rec.flush() > 0


class TestTheIntrabarCallSurvivesToo:
    """It is the reading, so losing it on a restart loses the record."""

    def test_the_call_round_trips(self, tmp_path):
        lad = builders((250,))
        feed(lad)
        rec = OfiBarRecorder(OfiBarStore(tmp_path / "t.db"), lad)
        rec.seed()
        rec.flush()
        fresh = builders((250,))
        OfiBarRecorder(OfiBarStore(tmp_path / "t.db"),
                       fresh).load_into_builders()
        got = fresh[250].bars
        assert any(b.call_side for b in got), "no call was ever latched"
        for a, b in zip(lad[250].bars, got):
            assert a.call_side == b.call_side
            assert a.call_px == b.call_px

    def test_an_older_row_reads_back_with_no_call(self, tmp_path):
        """Bars written before the column existed never recorded one,
        and inventing a price for them would be worse than having none."""
        import sqlite3
        path = tmp_path / "old.db"
        conn = sqlite3.connect(path)
        conn.executescript(
            "CREATE TABLE ofi_bars (id INTEGER PRIMARY KEY,"
            " target INTEGER NOT NULL, ts_open REAL NOT NULL,"
            " ts_close REAL NOT NULL, open REAL NOT NULL,"
            " high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,"
            " direction TEXT NOT NULL, ofi INTEGER NOT NULL,"
            " updates INTEGER NOT NULL, written_at REAL NOT NULL);"
            "CREATE UNIQUE INDEX ofi_bars_one"
            " ON ofi_bars(target, ts_open, ts_close);")
        conn.execute(
            "INSERT INTO ofi_bars (target,ts_open,ts_close,open,high,low,"
            "close,direction,ofi,updates,written_at)"
            " VALUES (250,1,2,10,11,9,10,'green',250,5,0)")
        conn.commit()
        conn.close()

        st = OfiBarStore(path)
        b = st.recent(250)[0]
        assert b.call_side == "" and b.call_px == 0.0
        assert st.count(250) == 1, "the old row survived the migration"


class TestTheObservationsSurvive:
    """At target 2,000 a bar is minutes, so four thousand bars is days.
    A restart that lost the observations would empty the table that
    measures the strategy and take days to refill it."""

    def test_they_round_trip(self, tmp_path):
        lad = builders((250,))
        feed(lad)
        assert any(b.episodes for b in lad[250].bars)
        rec = OfiBarRecorder(OfiBarStore(tmp_path / "t.db"), lad)
        rec.seed()
        rec.flush()
        fresh = builders((250,))
        OfiBarRecorder(OfiBarStore(tmp_path / "t.db"),
                       fresh).load_into_builders()
        for a, b in zip(lad[250].bars, fresh[250].bars):
            assert len(a.episodes) == len(b.episodes)
            for x, y in zip(a.episodes, b.episodes):
                assert x.as_dict() == y.as_dict()

    def test_the_win_loss_record_comes_back_with_them(self, tmp_path):
        lad = builders((250,))
        feed(lad)
        rec = OfiBarRecorder(OfiBarStore(tmp_path / "t.db"), lad)
        rec.seed()
        rec.flush()
        fresh = builders((250,))
        OfiBarRecorder(OfiBarStore(tmp_path / "t.db"),
                       fresh).load_into_builders()
        from liqmap import agree as A
        a = A.observation_table(lad[250].bars, points=(1, 2))
        b = A.observation_table(fresh[250].bars, points=(1, 2))
        assert a["observations"] == b["observations"]
        assert a["rows"] == b["rows"]

    def test_a_bar_with_no_observations_stores_nothing(self, store):
        assert store.record(250, OfiBar(ts_open=1.0, ts_close=2.0, open=1,
                                        high=2, low=0, close=1,
                                        direction="green", ofi=250,
                                        updates=9)) == 1
        assert store.recent(250)[0].episodes == []

    def test_an_unreadable_record_yields_none_not_a_guess(self, tmp_path):
        """Inventing an observation is worse than having none."""
        from liqmap.ofistore import _eps_in
        assert _eps_in("not json at all") == []
        assert _eps_in("") == []
        assert _eps_in("[{}]")[0].side == ""


class TestBothSidesSurviveTheRestart:
    """They are the only record of how contested a closed bar was, and
    they cannot be recomputed from anything else the row carries."""

    def test_they_round_trip(self, tmp_path):
        lad = builders((250,))
        feed(lad)
        assert any(b.gross for b in lad[250].bars)
        rec = OfiBarRecorder(OfiBarStore(tmp_path / "t.db"), lad)
        rec.seed()
        rec.flush()
        fresh = builders((250,))
        OfiBarRecorder(OfiBarStore(tmp_path / "t.db"),
                       fresh).load_into_builders()
        for a, b in zip(lad[250].bars, fresh[250].bars):
            assert a.push_up == b.push_up and a.push_dn == b.push_dn
            assert b.push_up - b.push_dn == b.ofi

    def test_an_older_row_reads_back_as_no_flow_not_as_selling(self, store):
        """Zero on both sides reports an even share, which is what
        'nothing recorded' should look like."""
        store.record(250, OfiBar(ts_open=1.0, ts_close=2.0, open=1, high=2,
                                 low=0, close=1, direction="green",
                                 ofi=250, updates=9))
        got = store.recent(250)[0]
        assert got.gross == 0 and got.share == 0.5


class TestTheTradeRecordSurvivesTheRestart:
    """The fixtures above only ever produce skips, so every field that
    only a TAKEN trade carries -- the fill, the result, the speed it
    fired into -- would round-trip untested on them. This builds the
    four outcomes by hand instead of hoping a feed happens to make
    them."""

    def _bar(self):
        from liqmap.ofibar import OfiBar, Episode
        b = OfiBar(ts_open=1.0, ts_close=9.0, open=100.0, high=140.0,
                   low=90.0, close=130.0, direction="green", ofi=600,
                   updates=40, path=321.0)
        b.episodes = [
            Episode(side="up", rule="imb-confirm-v2", level=500,
                    px0=100.0, ts0=1.0, px_end=130.0, ts_end=5.0,
                    took=True, entry_px=101.0, entry_ts=1.5,
                    atr0=12.5, tps0=3.25, updates=7,
                    mfe=30.0, mae=4.0, end="bar",
                    hit="win", hit_px=125.0, hit_ts=4.0,
                    first={1: "win", 2: "win"}),
            Episode(side="down", rule="imb-confirm-v2", level=500,
                    px0=130.0, ts0=5.0, px_end=136.0, ts_end=6.0,
                    took=True, entry_px=129.0, entry_ts=5.2,
                    atr0=20.0, tps0=1.5, updates=3,
                    mfe=1.0, mae=7.0, end="reversal",
                    hit="loss", hit_px=136.0, hit_ts=6.0,
                    first={1: "loss", 2: "loss"}),
            Episode(side="up", rule="imb-confirm-v2", level=500,
                    px0=136.0, ts0=6.0, px_end=136.0, ts_end=7.0,
                    took=False, updates=2, end="reversal", hit="skip",
                    hit_px=136.0, hit_ts=7.0),
            Episode(side="up", rule="imb-confirm-v2", level=500,
                    px0=136.0, ts0=7.0, px_end=138.0, ts_end=9.0,
                    took=True, entry_px=137.0, entry_ts=7.5,
                    atr0=30.0, tps0=0.5, updates=5, end="bar"),
        ]
        return b

    def test_all_four_outcomes_come_back_exactly(self, tmp_path):
        st = OfiBarStore(tmp_path / "t.db")
        bar = self._bar()
        st.record_many(250, [bar])
        back = st.recent(250, limit=10)
        assert len(back) == 1
        got = back[0]
        assert [e.hit for e in got.episodes] == ["win", "loss", "skip", ""]
        for a, b in zip(bar.episodes, got.episodes):
            assert a.as_dict() == b.as_dict()

    def test_the_fill_and_the_result_survive(self, tmp_path):
        st = OfiBarStore(tmp_path / "t.db")
        st.record_many(250, [self._bar()])
        win, loss, skip, live = st.recent(250, limit=10)[0].episodes
        assert win.entered and win.entry_px == 101.0
        assert win.result == 125.0 - 101.0, "the win's result moved"
        assert loss.entered and loss.result == 129.0 - 136.0
        assert not skip.entered and skip.result == 0.0
        assert live.entered and live.hit == ""

    def test_the_speed_it_fired_into_survives(self, tmp_path):
        st = OfiBarStore(tmp_path / "t.db")
        st.record_many(250, [self._bar()])
        got = st.recent(250, limit=10)[0]
        assert got.path == 321.0
        assert [round(e.atr0, 2) for e in got.episodes] == [12.5, 20.0,
                                                            0.0, 30.0]
        assert [round(e.tps0, 2) for e in got.episodes] == [3.25, 1.5,
                                                            0.0, 0.5]

    def test_a_row_written_before_any_of_this_still_loads(self, tmp_path):
        """The field that matters is `took`: an older row has no such
        key, and a trade it never took must not come back as one."""
        import json
        import sqlite3
        st = OfiBarStore(tmp_path / "t.db")
        st.record_many(250, [self._bar()])
        old = json.dumps([{"side": "up", "px0": 100.0, "px_end": 110.0,
                           "hit": "win", "hit_px": 110.0, "level": 500,
                           "first": {"1": "win"}}])
        with sqlite3.connect(tmp_path / "t.db") as cx:
            cx.execute("UPDATE ofi_bars SET episodes=?, path=0", (old,))
        got = st.recent(250, limit=10)[0]
        e = got.episodes[0]
        assert not e.took and not e.entered
        assert e.entry_px == 0.0 and e.atr0 == 0.0
        assert e.rule == "", "an older row claimed the current rule"
        assert got.path == 0.0 and got.efficiency == 0.0
