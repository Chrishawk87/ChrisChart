"""The MBP-10 audit: a census, and one fair comparison.

Three things would make it lie.

THE COMPARISON MUST BE CLIPPED THE SAME WAY. `dom.read` judges only prices
both snapshots could see. Summing every cancel in the file against that
charges the inference for events it deliberately declines to judge, and
would report it as blind when it is being careful.

THE WINDOWS MUST NOT OVERLAP OR LEAK. Pending adds and cancels belong to
the window that closed, and must be cleared when it does. Carrying them
forward double-counts every one.

AND IT MUST READ THE ACTION FIELD, not re-derive it. The whole point is to
see what `dbn.stream` discards; a version that goes through `stream` can
only ever confirm itself.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load():
    path = ROOT / "scripts" / "mbp_audit.py"
    spec = importlib.util.spec_from_file_location("mbp_audit", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mbp_audit"] = mod
    spec.loader.exec_module(mod)
    return mod


ma = _load()
NS = 1_000_000_000
PX = 1_000_000_000          # one price unit at PRICE_SCALE 1e-9


def levels(bid_px, bid_sz, ask_px, ask_sz, n=10):
    out = []
    for i in range(n):
        out.append({"bid_px": int((bid_px - 0.25 * i) * PX),
                    "bid_sz": bid_sz,
                    "ask_px": int((ask_px + 0.25 * i) * PX),
                    "ask_sz": ask_sz,
                    "bid_ct": 1, "ask_ct": 1})
    return out


def book_rec(ts, bid=99.75, ask=100.0, bsz=50, asz=50, action="A",
             side="B", size=0, price=0.0, iid=1):
    return {"ts_event": int(ts * NS), "action": action, "side": side,
            "size": size, "price": int(price * PX) if price else 0,
            "instrument_id": iid,
            "levels": levels(bid, bsz, ask, asz)}


def plain(ts, action, side, size, price, iid=1):
    """A record with no level block -- an action, nothing else."""
    return {"ts_event": int(ts * NS), "action": action, "side": side,
            "size": size, "price": int(price * PX), "instrument_id": iid}


# ------------------------------------------------------------- the census

def test_the_census_counts_every_action_separately():
    recs = [plain(1.0, "A", "B", 10, 99.75),
            plain(1.1, "C", "B", 4, 99.75),
            plain(1.2, "C", "A", 7, 100.0),
            plain(1.3, "T", "B", 2, 100.0),
            plain(1.4, "M", "A", 3, 100.0)]
    c = ma.census(recs)
    assert c.records == 5
    assert c.by_action["A"] == 1
    assert c.by_action["C"] == 2
    assert c.by_action["T"] == 1
    assert c.by_action["M"] == 1


def test_the_census_keeps_the_side_of_each_action():
    recs = [plain(1.0, "C", "B", 10, 99.75), plain(1.1, "C", "A", 5, 100.0)]
    c = ma.census(recs)
    assert c.by_action_side[("C", "B")] == 1
    assert c.by_action_side[("C", "A")] == 1


def test_the_census_sums_size_not_only_count():
    recs = [plain(1.0, "C", "B", 10, 99.75), plain(1.1, "C", "B", 5, 99.75)]
    c = ma.census(recs)
    assert c.size_by_action["C"] == pytest.approx(15.0)


def test_out_of_order_and_duplicate_stamps_are_both_counted():
    recs = [plain(2.0, "A", "B", 1, 99.75),
            plain(2.0, "A", "B", 1, 99.75),      # duplicate
            plain(1.0, "A", "B", 1, 99.75)]      # backwards
    c = ma.census(recs)
    assert c.duplicate_ts == 1
    assert c.out_of_order == 1


def test_a_silent_stretch_is_reported_as_a_gap():
    recs = [plain(1.0, "A", "B", 1, 99.75), plain(9.0, "A", "B", 1, 99.75)]
    c = ma.census(recs, gap_s=1.0)
    assert c.gaps and c.gaps[0][0] == pytest.approx(8.0)


def test_a_quiet_market_is_not_a_gap_when_under_the_threshold():
    recs = [plain(1.0, "A", "B", 1, 99.75), plain(1.5, "A", "B", 1, 99.75)]
    assert ma.census(recs, gap_s=1.0).gaps == []


def test_a_crossed_book_is_counted():
    c = ma.census([book_rec(1.0, bid=100.25, ask=100.0)])
    assert c.crossed == 1


def test_an_ordinary_book_is_not_counted_as_crossed():
    assert ma.census([book_rec(1.0, bid=99.75, ask=100.0)]).crossed == 0


def test_more_than_one_instrument_is_visible():
    c = ma.census([plain(1.0, "A", "B", 1, 99.75, iid=1),
                   plain(2.0, "A", "B", 1, 99.75, iid=2)])
    assert len(c.instruments) == 2


def test_the_census_reads_the_action_field_rather_than_the_stream():
    """Going through `dbn.stream` could only confirm what it already does."""
    import inspect
    src = inspect.getsource(ma)
    assert "dbn._action(" in src
    assert "dbn.stream(" not in src


class _Enum:
    """What the Databento client actually hands back.

    `str(rec.action)` is 'Action.CANCEL', not 'C'. Anything that takes the
    first character of that gets 'A' for every record in the file --
    including every cancel -- and the census comes back saying the market
    only ever adds. `dbn._action` splits on the dot for exactly this
    reason, and a fixture built from plain dicts cannot tell the two apart.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return f"Action.{self.name}"


class _Rec:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_an_enum_shaped_action_is_decoded_not_truncated():
    recs = [_Rec(ts_event=NS, action=_Enum("CANCEL"), side=_Enum("BID"),
                 size=7, price=int(99.75 * PX), instrument_id=1),
            _Rec(ts_event=2 * NS, action=_Enum("ADD"), side=_Enum("ASK"),
                 size=3, price=int(100.0 * PX), instrument_id=1),
            _Rec(ts_event=3 * NS, action=_Enum("TRADE"), side=_Enum("BID"),
                 size=2, price=int(100.0 * PX), instrument_id=1)]
    c = ma.census(recs)
    assert c.by_action["C"] == 1, dict(c.by_action)
    assert c.by_action["A"] == 1, dict(c.by_action)
    assert c.by_action["T"] == 1, dict(c.by_action)
    assert c.by_action_side[("C", "B")] == 1
    assert c.by_action_side[("A", "A")] == 1
    assert c.size_by_action["C"] == pytest.approx(7.0)


def _enum_levels(bsz, asz, n=10):
    return [_Rec(bid_px=int((99.75 - 0.25 * i) * PX), bid_sz=bsz,
                 ask_px=int((100.0 + 0.25 * i) * PX), ask_sz=asz,
                 bid_ct=1, ask_ct=1) for i in range(n)]


def test_the_comparison_also_survives_enum_shaped_records():
    """The census and the comparison decode the field independently, so
    both need the enum shape exercised."""
    recs = [_Rec(ts_event=0, action=_Enum("ADD"), side=_Enum("BID"), size=0,
                 price=0, instrument_id=1, levels=_enum_levels(50, 50)),
            _Rec(ts_event=NS, action=_Enum("CANCEL"), side=_Enum("BID"),
                 size=20, price=int(99.75 * PX), instrument_id=1),
            _Rec(ts_event=11 * NS, action=_Enum("ADD"), side=_Enum("BID"),
                 size=0, price=0, instrument_id=1,
                 levels=_enum_levels(30, 50))]
    out = ma.compare(recs, window_s=10.0)
    assert out.windows == 1
    assert out.true_cancel == pytest.approx(20.0)
    assert out.inferred_cancel > 0


# -------------------------------------------------------- the comparison

def _run(window=10.0):
    """A book that holds still, with real cancels and adds in between."""
    recs = [book_rec(0.0, bsz=50, asz=50)]
    for i in range(1, 10):
        recs.append(plain(i * 0.5, "C", "B", 4, 99.75))
        recs.append(plain(i * 0.5 + 0.1, "A", "B", 4, 99.75))
    recs.append(book_rec(window + 1.0, bsz=50, asz=50))
    return ma.compare(recs, window_s=window)


def test_the_file_sees_churn_that_the_difference_nets_to_nothing():
    """THE finding this script exists to produce, as a unit test.

    Nine cancels and nine adds of four lots each at the same price leave
    the book exactly as it was. Differencing two snapshots reports no
    cancellation whatsoever; the file reports thirty-six lots of it.
    """
    out = _run()
    assert out.windows == 1
    assert out.true_cancel == pytest.approx(36.0)
    assert out.inferred_cancel == pytest.approx(0.0)
    assert out.cancel_ratio == pytest.approx(0.0)


def test_a_real_shrink_is_seen_by_both():
    recs = [book_rec(0.0, bsz=50, asz=50),
            plain(1.0, "C", "B", 20, 99.75),
            book_rec(11.0, bsz=30, asz=50)]
    out = ma.compare(recs, window_s=10.0)
    assert out.windows == 1
    assert out.true_cancel == pytest.approx(20.0)
    assert out.inferred_cancel > 0


def test_cancels_outside_the_shared_price_span_are_not_charged():
    """The inference declines to judge prices that left the window, so the
    truth it is compared against must decline too."""
    recs = [book_rec(0.0, bid=99.75, ask=100.0),
            plain(1.0, "C", "B", 99, 10.0),       # nowhere near the book
            book_rec(11.0, bid=99.75, ask=100.0)]
    out = ma.compare(recs, window_s=10.0)
    assert out.true_cancel == pytest.approx(0.0)


def test_each_window_starts_from_empty_pending_counters():
    """Carrying pending events forward double-counts every one of them."""
    recs = [book_rec(0.0)]
    for w in range(3):
        recs.append(plain(w * 11 + 1.0, "C", "B", 5, 99.75))
        recs.append(book_rec(w * 11 + 11.0))
    out = ma.compare(recs, window_s=10.0)
    assert out.windows == 3
    assert out.true_cancel == pytest.approx(15.0)


def test_a_window_shorter_than_the_setting_does_not_close():
    recs = [book_rec(0.0), book_rec(1.0), book_rec(2.0)]
    assert ma.compare(recs, window_s=10.0).windows == 0


def test_the_ratio_is_none_rather_than_a_division_by_zero():
    out = ma.Compare()
    assert out.cancel_ratio is None


def test_the_correlation_needs_more_than_a_couple_of_points():
    assert ma.correlation([(1.0, 1.0), (2.0, 2.0)]) is None
    assert ma.correlation([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]) == \
        pytest.approx(1.0)


def test_a_flat_series_has_no_correlation_rather_than_one():
    assert ma.correlation([(1.0, 5.0), (2.0, 5.0), (3.0, 5.0)]) is None


def test_the_comparison_calls_dom_read_as_written():
    import inspect
    src = inspect.getsource(ma)
    assert "dom_read(" in src
    for banned in ("MIN_OVERLAP =", "WINDOW_S =20", ".pull_share ="):
        assert banned not in src, banned


# --------------------------------------------------------- no strategy here

def test_it_scores_no_outcome_and_reads_no_signal():
    import inspect
    src = inspect.getsource(ma).lower()
    for word in ("first_touch", "win_rate", "target", "stop_ticks",
                 "bookreader"):
        assert word not in src, word


def test_no_production_module_is_mutated():
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(ma))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute):
                    assert getattr(t.value, "id", "") not in ("dbn", "dom")


def test_nothing_here_places_an_order():
    import inspect
    src = inspect.getsource(ma).lower()
    for word in ("def submit", "def execute", "def place", "def order"):
        assert word not in src


# ------------------------------------------------------------- progress

def test_progress_never_writes_to_the_report():
    """Two passes over eight million records printed nothing until each
    finished, which looks exactly like a hang. The heartbeat fixes that,
    but on stdout it would corrupt the report it is reporting on."""
    import io
    buf = io.StringIO()
    p = ma.Progress("x", every=2, out=buf)
    for _ in range(4):
        p.tick()
    p.done()
    assert buf.getvalue()
    assert p.out is not sys.stdout


def test_progress_defaults_to_stderr():
    assert ma.Progress("x").out is sys.stderr


def test_progress_survives_a_zero_interval():
    import io
    ma.Progress("x", every=0, out=io.StringIO()).tick()


def test_both_passes_accept_a_progress_object():
    import io
    a, b = io.StringIO(), io.StringIO()
    recs = [plain(float(i), "C", "B", 1, 99.75) for i in range(6)]
    ma.census(recs, progress=ma.Progress("c", every=2, out=a))
    ma.compare(recs, progress=ma.Progress("p", every=2, out=b))
    assert a.getvalue() and b.getvalue()


def test_both_passes_run_without_one():
    recs = [plain(float(i), "C", "B", 1, 99.75) for i in range(4)]
    assert ma.census(recs).records == 4
    assert ma.compare(recs).windows == 0


def test_a_limit_takes_the_first_records_and_no_limit_takes_them_all():
    assert list(ma.capped(range(10), 3)) == [0, 1, 2]
    assert list(ma.capped(range(4), 0)) == [0, 1, 2, 3]
    assert list(ma.capped(range(4), -1)) == [0, 1, 2, 3]


def test_a_capped_run_says_so_in_the_report():
    """A partial figure that does not announce itself gets quoted as a
    result for the session."""
    import inspect
    src = inspect.getsource(ma.main)
    assert "LIMIT" in src and "not a result" in src
