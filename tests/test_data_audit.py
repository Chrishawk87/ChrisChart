"""The Phase 1 audit: it reads, it reports, it changes nothing.

Four things would make it dangerous rather than useless.

IT MUST NOT WRITE. It is pointed at the only copy of two years of data.
Anything that opens a file for writing, moves one, or deletes one is a
catastrophe, not a bug.

THE BRACKET DIAGNOSTIC MUST COUNT THE RIGHT BARS. A bar is unresolvable
when its range spans target PLUS stop, because then both levels were
touched inside it and an OHLCV row records no sequence. Counting only
bars that span the target would understate the problem and wave through
a backtest that cannot be run.

BREAK-EVEN MUST INCLUDE THE COST. A four-tick target against an eight-tick
stop needs three quarters of its trades even before commission; a formula
that drops the cost reports a bar anyone can clear.

AND A MISSING FILE IS NOT AN EMPTY DATASET. Reporting "0 files" when a path
was mistyped invites the conclusion that there is nothing to test.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load():
    path = ROOT / "scripts" / "data_audit.py"
    spec = importlib.util.spec_from_file_location("data_audit", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["data_audit"] = mod
    spec.loader.exec_module(mod)
    return mod


da = _load()


# ------------------------------------------------------------- read only

def test_it_opens_nothing_for_writing():
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(da))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", "") or \
                getattr(node.func, "attr", "")
            if name == "open":
                for kw in node.keywords:
                    if kw.arg == "mode":
                        assert "w" not in getattr(kw.value, "value", "")
                if len(node.args) > 1:
                    mode = getattr(node.args[1], "value", "")
                    assert "w" not in mode and "a" not in mode, mode


def test_it_names_no_destructive_operation():
    import inspect
    src = inspect.getsource(da).lower()
    for word in ("shutil.move", "shutil.rmtree", "os.remove", "os.unlink",
                 "os.rename", ".write_text", ".write_bytes", "to_csv",
                 "to_parquet"):
        assert word not in src, word


def test_sqlite_is_opened_read_only():
    import inspect
    src = inspect.getsource(da.describe_sqlite)
    assert "mode=ro" in src


def test_nothing_here_places_an_order():
    import inspect
    src = inspect.getsource(da).lower()
    for word in ("def submit", "def execute", "def place", "def order"):
        assert word not in src


def test_it_reports_no_outcome_and_claims_no_edge():
    import ast
    import inspect

    class Blank(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(value=""), node)
            return node

        def visit_JoinedStr(self, node):
            return ast.copy_location(ast.Constant(value=""), node)

    tree = Blank().visit(ast.parse(inspect.getsource(da)))
    ast.fix_missing_locations(tree)
    code = ast.unparse(tree).lower()
    for word in ("win_rate", "profit_factor", "equity", "sharpe",
                 "optimal", "recommend"):
        assert word not in code, word


# ----------------------------------------------------------- finding files

def test_it_finds_data_files_and_ignores_the_rest(tmp_path):
    (tmp_path / "a.dbn.zst").write_bytes(b"x")
    (tmp_path / "b.csv").write_text("ts,px\n1,2\n")
    (tmp_path / "notes.md").write_text("hello")
    (tmp_path / "pic.png").write_bytes(b"x")
    got = {f.path.name for f in da.scan([tmp_path])}
    assert got == {"a.dbn.zst", "b.csv"}


def test_it_looks_in_subdirectories():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        (root / "2024").mkdir()
        (root / "2024" / "es.dbn.zst").write_bytes(b"x")
        assert len(da.scan([root])) == 1


def test_a_single_file_can_be_audited_directly(tmp_path):
    f = tmp_path / "one.csv"
    f.write_text("ts,open,high,low,close\n")
    assert [x.path.name for x in da.scan([f])] == ["one.csv"]


def test_a_missing_path_is_reported_not_silently_empty(tmp_path, capsys):
    da.scan([tmp_path / "nope"])
    assert "no such path" in capsys.readouterr().err


def test_the_same_file_reached_twice_is_counted_once(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text("a\n")
    assert len(da.scan([tmp_path, f])) == 1


def test_a_csv_header_is_read_without_loading_the_file(tmp_path):
    f = tmp_path / "bars.csv"
    f.write_text("ts_event,open,high,low,close,volume\n1,2,3,4,5,6\n")
    found = da.Found(path=f, size=f.stat().st_size)
    da.describe(found)
    assert found.kind == "delimited text"
    assert "open" in found.schema and "volume" in found.schema


def test_a_sqlite_file_lists_its_tables(tmp_path):
    import sqlite3
    f = tmp_path / "store.db"
    con = sqlite3.connect(f)
    con.execute("CREATE TABLE bars (ts REAL)")
    con.commit()
    con.close()
    found = da.Found(path=f, size=f.stat().st_size)
    da.describe(found)
    assert found.kind == "sqlite"
    assert "bars" in found.schema


def test_an_unreadable_file_is_noted_rather_than_crashing(tmp_path):
    f = tmp_path / "broken.dbn.zst"
    f.write_bytes(b"not a dbn file")
    found = da.Found(path=f, size=f.stat().st_size)
    da.describe(found)          # must not raise
    assert found.kind == "databento DBN"


class _FakeStore:
    class metadata:
        schema = "ohlcv-1m"
        start = 1_791_466_200 * 10**9
        end = 1_791_499_200 * 10**9
        symbols = ["ESZ6"]
    nbytes = 1234


def _fake_databento(exc=None):
    """Stands in for the real client, which is not installed everywhere.

    Without this the whole DBN branch is dead in the test run and a
    `describe_dbn` that crashes on a corrupt file passes every test.
    """
    import types
    mod = types.ModuleType("databento")

    class DBNStore:
        @staticmethod
        def from_file(path):
            if exc is not None:
                raise exc
            return _FakeStore()

    mod.DBNStore = DBNStore
    return mod


def test_a_corrupt_dbn_is_noted_whatever_the_client_raises(monkeypatch,
                                                           tmp_path):
    """Databento raises its own error types, not ValueError. Narrowing the
    guard means one bad file aborts the audit of the other seven hundred."""
    monkeypatch.setitem(sys.modules, "databento",
                        _fake_databento(exc=RuntimeError("bad magic")))
    f = tmp_path / "corrupt.dbn.zst"
    f.write_bytes(b"xx")
    found = da.Found(path=f, size=2)
    da.describe_dbn(found)
    assert "could not open" in found.note
    assert "bad magic" in found.note


def test_a_readable_dbn_reports_its_metadata_without_parsing(monkeypatch,
                                                             tmp_path):
    monkeypatch.setitem(sys.modules, "databento", _fake_databento())
    f = tmp_path / "es.dbn.zst"
    f.write_bytes(b"xx")
    found = da.Found(path=f, size=2)
    da.describe_dbn(found)
    assert found.schema == "ohlcv-1m"
    assert found.symbols == ["ESZ6"]
    assert "2026" in found.start and "2026" in found.end


def test_a_dbn_with_no_metadata_block_is_noted(monkeypatch, tmp_path):
    import types
    mod = types.ModuleType("databento")

    class Bare:
        metadata = None

    class DBNStore:
        @staticmethod
        def from_file(path):
            return Bare()

    mod.DBNStore = DBNStore
    monkeypatch.setitem(sys.modules, "databento", mod)
    found = da.Found(path=tmp_path / "x.dbn", size=1)
    da.describe_dbn(found)
    assert "no metadata" in found.note


# ------------------------------------------------------ the bracket maths

def test_a_bar_is_unresolvable_when_it_spans_TARGET_PLUS_STOP(capsys):
    """The one that decides whether a four-tick backtest is possible.

    A bar that covers eight ticks contains both a +4 target and a -4 stop,
    so both were touched and the row records no order. Counting only bars
    that span the TARGET would call those resolvable and wave through a
    backtest whose every outcome is a guess.
    """
    ranges = [7.0] * 50 + [8.0] * 30 + [20.0] * 20
    da.bracket_report(ranges, target=4, stops=[4])
    out = capsys.readouterr().out
    assert "50.0%" in out, out          # the 8s and the 20s, not the 7s


def test_a_wider_stop_makes_more_bars_resolvable(capsys):
    ranges = [9.0] * 100
    da.bracket_report(ranges, target=4, stops=[4, 8])
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if "UNRESOLVABLE" in ln]
    assert "100.0%" in lines[0]        # bracket 8, every bar spans it
    assert "0.0%" in lines[1]          # bracket 12, none do


def test_an_empty_range_list_says_so_rather_than_dividing_by_zero(capsys):
    da.bracket_report([], target=4, stops=[4])
    assert "no usable bars" in capsys.readouterr().out


def test_break_even_includes_the_cost():
    assert da.breakeven(4, 4, 0.0) == pytest.approx(0.5)
    assert da.breakeven(4, 4, 1.0) == pytest.approx(0.625)
    assert da.breakeven(4, 6, 1.0) == pytest.approx(0.70)
    assert da.breakeven(4, 8, 1.0) == pytest.approx(0.75)


def test_a_target_smaller_than_its_stop_needs_more_than_half():
    for stop in (6, 8, 12):
        assert da.breakeven(4, stop, 0.0) > 0.5


def test_break_even_rises_with_the_stop_and_with_the_cost():
    assert da.breakeven(4, 8, 1.0) > da.breakeven(4, 4, 1.0)
    assert da.breakeven(4, 4, 1.5) > da.breakeven(4, 4, 0.4)


# ---------------------------------------------------------- the bar walk

NS = 1_000_000_000
PX = 1_000_000_000


def bar(ts, o=100.0, h=100.5, low=99.5, c=100.25, v=10, iid=1):
    return {"ts_event": int(ts * NS), "open": int(o * PX),
            "high": int(h * PX), "low": int(low * PX),
            "close": int(c * PX), "volume": v, "instrument_id": iid}


def test_the_walk_counts_bars_sessions_and_instruments():
    recs = [bar(float(i) * 60.0) for i in range(10)]
    st = da.bar_stats(recs, expect_s=60.0)
    assert st.bars == 10
    assert len(st.symbols) == 1
    assert st.sessions


def test_a_hole_in_the_series_is_reported_as_a_gap():
    recs = [bar(0.0), bar(60.0), bar(6000.0)]
    st = da.bar_stats(recs, expect_s=60.0)
    assert st.gaps and st.gaps[0][0] == pytest.approx(5940.0)


def test_a_bar_arriving_on_time_is_not_a_gap():
    st = da.bar_stats([bar(float(i) * 60.0) for i in range(5)],
                      expect_s=60.0)
    assert st.gaps == []


def test_duplicate_and_backwards_stamps_are_counted_separately():
    recs = [bar(120.0), bar(120.0), bar(60.0)]
    st = da.bar_stats(recs, expect_s=60.0)
    assert st.duplicates == 1
    assert st.out_of_order == 1


def test_an_impossible_ohlc_row_is_flagged_and_left_out_of_the_ranges():
    """high below low, or a high under the close, is a corrupt row and
    its range would poison the diagnostic."""
    recs = [bar(0.0, h=99.0, low=100.0), bar(60.0)]
    st = da.bar_stats(recs, expect_s=60.0)
    assert st.bad_prices == 1
    assert len(st.ranges) == 1


def test_a_range_is_reported_in_ticks():
    st = da.bar_stats([bar(0.0, h=101.0, low=100.0)], expect_s=60.0)
    assert st.ranges == [pytest.approx(4.0)]


def test_zero_volume_bars_are_counted():
    st = da.bar_stats([bar(0.0, v=0), bar(60.0, v=5)], expect_s=60.0)
    assert st.zero_volume == 1


def test_the_walk_stops_at_its_cap():
    recs = (bar(float(i) * 60.0) for i in range(10_000))
    st = da.bar_stats(recs, expect_s=60.0, limit=100)
    assert st.bars == 100


def test_a_session_is_named_for_the_day_it_closes_on():
    """CME equity sessions run 17:00 to 16:00 Chicago, so Sunday evening
    belongs to Monday. Naming it by the calendar day would split every
    session in two and double the session count."""
    import datetime as _dt
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("America/Chicago")
    except Exception:                                  # noqa: BLE001
        pytest.skip("no zoneinfo")
    sun = _dt.datetime(2026, 10, 11, 18, 0, tzinfo=tz)   # Sunday evening
    mon = _dt.datetime(2026, 10, 12, 9, 0, tzinfo=tz)    # Monday morning
    assert da._session_day(sun.timestamp()) == \
        da._session_day(mon.timestamp())


def test_percentiles_hit_the_ends():
    xs = list(range(101))
    assert da._pct(xs, 0.0) == 0
    assert da._pct(xs, 1.0) == 100
    assert da._median(xs) == 50
