"""The database under concurrent use, which is how it actually runs.

This service has a background worker sweeping and settling while FastAPI
serves dashboard polls from a threadpool. For most of this project's life
every store shared ONE `sqlite3` connection opened with
`check_same_thread=False` — which does not make a connection thread safe, it
only removes the check that would have said so.

It worked by luck: the worker wrote rarely enough that collisions were
uncommon. Adding a second resolver pass to the worker loop was enough to
tip it, and the live service began returning

    IndexError: tuple index out of range

from `/api/read`, thrown inside `dict(row)` in `coins_with_data` — a
function with nothing wrong with it. A `sqlite3.Row` keeps its column
description separately from its value tuple, so when another thread resets
the shared prepared statement mid-flight the two disagree and the failure
lands somewhere innocent. That is the worst property of this bug: the
traceback names the wrong module, and the real cause is in a different
thread.

These tests hammer the real classes from many threads and assert zero
errors. They are slow by design — a race that only shows up sometimes needs
sustained contention to show up reliably — but a few seconds here is worth
considerably more than another afternoon spent reading `coins_with_data`.
"""

import tempfile
import threading
from collections import Counter
from pathlib import Path

import pytest

from liqmap.bucket import Position
from liqmap.history import History
from liqmap.settings import SettingsStore
from liqmap.store import Store

# Long enough to produce collisions reliably on a shared connection (the old
# code failed within two seconds), short enough to keep the suite usable.
SECONDS = 4.0


def pos(i, coin="BTC"):
    return Position(wallet=f"0x{i:040x}", coin=coin, szi=1.0, entry_px=100.0,
                    liquidation_px=90.0, position_value=1e6, leverage=10,
                    account_value=1e7, margin_used=1e5)


def hammer(workers, seconds=SECONDS):
    """Run every callable in `workers` in its own thread. Return the errors."""
    errors: list[str] = []
    ops = [0]
    stop = threading.Event()
    lock = threading.Lock()

    def wrap(fn):
        def run():
            while not stop.is_set():
                try:
                    fn()
                    with lock:
                        ops[0] += 1
                except Exception as exc:          # noqa: BLE001
                    errors.append(f"{type(exc).__name__}: {exc}")
        return run

    threads = [threading.Thread(target=wrap(fn), daemon=True) for fn in workers]
    for t in threads:
        t.start()
    stop.wait(seconds)
    stop.set()
    for t in threads:
        t.join(timeout=10)
    return errors, ops[0]


@pytest.fixture
def hist():
    with tempfile.TemporaryDirectory() as tmp:
        h = History(Path(tmp) / "h.db")
        for i, coin in enumerate(["BTC", "ETH", "xyz:GOLD", "SOL"]):
            h.record_sweep(coin, 100.0, [pos(i, coin)])
        yield h
        h.close()


@pytest.mark.slow
def test_readers_and_writers_do_not_corrupt_each_other(hist):
    """The exact shape that broke production: dashboard reads against worker
    writes."""
    workers = (
        [lambda: hist.coins_with_data()] * 6
        + [lambda: hist.read_stats(), lambda: hist.suggestion_stats(),
           lambda: hist.counts(), lambda: hist.sweeps(limit=5)]
        + [(lambda n: lambda: hist.record_sweep("BTC", 100.0, [pos(n)]))(i)
           for i in range(4)]
    )
    errors, ops = hammer(workers)
    assert ops > 1000, f"only {ops} operations — the hammer did not run"
    assert not errors, f"{len(errors)} errors: {Counter(errors).most_common(3)}"


@pytest.mark.slow
def test_the_exact_call_that_failed_in_production(hist):
    """`dict(row)` in `coins_with_data`, under write pressure."""
    workers = ([lambda: [r["coin"] for r in hist.coins_with_data()]] * 8
               + [(lambda n: lambda: hist.record_sweep(
                   "ETH", 100.0, [pos(n, "ETH")]))(i) for i in range(3)])
    errors, ops = hammer(workers)
    assert not errors, f"{len(errors)} errors: {Counter(errors).most_common(3)}"


@pytest.mark.slow
def test_suggestions_survive_concurrent_writes_and_reads(hist):
    """Recording, deciding and settling all at once, which is what the
    dashboard and the worker do to each other."""
    import time as _t

    base = _t.time() - 300
    made = []

    def record(n):
        def go():
            sid = hist.record_suggestion(
                coin="BTC", interval="15m", candle_ts=base + n * 900,
                candle_end=base + n * 900 + 900, side="long", entry=100.0,
                target_px=101.0, stop_px=99.0, target_bps=100.0,
                risk_bps=100.0, rr=1.0, cost_bps=2.0, conviction=0.7,
                score=0.5, made_at=base + n * 900 + 60)
            if sid:
                made.append(sid)
        return go

    workers = ([record(i) for i in range(6)]
               + [lambda: hist.suggestion_stats()] * 4
               + [lambda: hist.pending_suggestions(now=base + 1e6)]
               + [lambda: made and hist.decide(made[0], True)])
    errors, ops = hammer(workers)
    assert not errors, f"{len(errors)} errors: {Counter(errors).most_common(3)}"


@pytest.mark.slow
def test_the_unique_index_holds_under_a_race(hist):
    """Two threads writing the same candle must produce exactly one row.

    The check-then-insert in `record_suggestion` cannot do this alone; the
    unique index is what makes it true.
    """
    import time as _t

    ts = _t.time() - 300
    got: list[str] = []

    def go():
        sid = hist.record_suggestion(
            coin="BTC", interval="15m", candle_ts=ts, candle_end=ts + 900,
            side="long", entry=100.0, target_px=101.0, stop_px=99.0,
            target_bps=100.0, risk_bps=100.0, rr=1.0, cost_bps=2.0,
            conviction=0.7, score=0.5, made_at=ts + 60)
        if sid:
            got.append(sid)

    errors, _ = hammer([go] * 8, seconds=2.0)
    assert not errors, f"{len(errors)} errors: {Counter(errors).most_common(3)}"
    assert len(got) == 1, f"{len(got)} rows written for one candle"


@pytest.mark.slow
def test_store_and_settings_are_thread_safe_too(tmp_path):
    """Same latent bug, same fix. They share the database file with History
    in a deployed service."""
    st = Store(tmp_path / "s.db")
    cfg = SettingsStore(str(tmp_path / "c.db"))
    try:
        workers = ([lambda: st.log_price("BTC", 100.0)] * 3
                   + [lambda: cfg.load()] * 4
                   + [lambda: cfg.update({"bucket_bps": 25.0})] * 2)
        errors, ops = hammer(workers)
        assert not errors, f"{len(errors)} errors: {Counter(errors).most_common(3)}"
    finally:
        st.close()
        cfg.close()


def test_each_thread_gets_its_own_connection(hist):
    """The mechanism, asserted directly, so a refactor back to a shared
    connection fails here rather than in production a week later."""
    seen: list[int] = []
    lock = threading.Lock()

    def grab():
        with lock:
            seen.append(id(hist._conn))

    threads = [threading.Thread(target=grab) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(set(seen)) == 5, "threads shared a connection"
    # And the same thread keeps its own rather than opening a new one per call.
    assert id(hist._conn) == id(hist._conn)


def test_wal_is_enabled_so_readers_do_not_block_the_writer(hist):
    mode = hist._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_closing_one_thread_does_not_break_another(hist):
    """`close()` must only close the caller's connection. Closing another
    thread's is the cross-thread use this whole module exists to prevent."""
    done = threading.Event()
    failed = []

    def other():
        try:
            hist.coins_with_data()
            done.wait(2)
            hist.coins_with_data()
        except Exception as exc:                  # noqa: BLE001
            failed.append(str(exc))

    t = threading.Thread(target=other)
    t.start()
    hist.close()                 # closes THIS thread's connection only
    done.set()
    t.join(timeout=5)
    assert not failed, failed
