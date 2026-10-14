"""The DOM probe: measuring why a layer is quiet, without fixing it by feel.

The danger here is subtle and it is not correctness. A diagnostic that
reports "threshold 0.02 would give you 40% coverage" invites picking 0.02,
and a knob chosen that way works beautifully until it meets data it has not
seen. So the tests below pin two things:

THE COVERAGE IS OF EVERY READ, NOT OF THE MEASURED ONES. A measure that
only exists a third of the time cannot cover more than a third, and scoring
it against the subset where it exists hides exactly that.

AND IT RECOMMENDS NOTHING. No "best", no sort by coverage, no suggested
value. It prints a curve and says what a curve is for.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load():
    path = ROOT / "scripts" / "dom_probe.py"
    spec = importlib.util.spec_from_file_location("dom_probe", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dom_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


dp = _load()


def book(ts, px=100.0, bid_sz=40.0, ask_sz=40.0, n=10):
    from liqmap.flow import Book, Level
    return Book(coin="ES", ts=ts,
                bids=[Level(px=px - 0.25 * (j + 1), sz=bid_sz)
                      for j in range(n)],
                asks=[Level(px=px + 0.25 * (j + 1), sz=ask_sz)
                      for j in range(n)])


def trade(ts, px=100.0, sz=1.0, buy=True):
    from liqmap.flow import Trade
    return Trade(px=px, sz=sz, aggressor="buy" if buy else "sell", ts=ts)


@pytest.fixture()
def passthrough(monkeypatch):
    from liqmap import dbn

    monkeypatch.setattr(dbn, "stream",
                        lambda recs, flip=False, **kw: iter(recs))


# ------------------------------------------------------- percentiles, not means

def test_the_spread_is_reported_as_percentiles():
    """A mean hides whether anything varies, which is the one thing this
    probe exists to find out."""
    line = dp.spread_of([0.0] * 50 + [1.0] * 50)
    for tag in ("p5", "p25", "med", "p75", "p95"):
        assert tag in line


def test_a_measure_that_never_moves_reads_as_flat():
    line = dp.spread_of([0.3] * 200)
    assert "p5 +0.300" in line and "p95 +0.300" in line


def test_no_samples_says_so_rather_than_printing_zeros():
    assert dp.spread_of([]) == "no samples"


def test_percentiles_land_where_they_should():
    xs = [i / 100.0 for i in range(101)]
    assert dp.pct(xs, 0.5) == pytest.approx(0.5, abs=0.01)
    assert dp.pct(xs, 0.0) == pytest.approx(0.0)
    assert dp.pct(xs, 1.0) == pytest.approx(1.0)


# ------------------------------------------------ coverage counts every read

def test_coverage_is_of_every_read_not_of_the_measured_subset():
    """A measure that exists a third of the time cannot cover more than a
    third, and dividing by the subset where it exists would report it as
    covering everything."""
    b = dp.Bucket(window_s=20.0)
    b.n = 300                       # three hundred reads
    b.pull_edge = [0.9] * 100       # the measure existed on a hundred
    assert b.coverage(b.pull_edge, 0.15) == pytest.approx(1 / 3)


def test_coverage_counts_both_directions():
    b = dp.Bucket(window_s=20.0)
    b.n = 4
    assert b.coverage([0.5, -0.5, 0.01, -0.01], 0.15) == pytest.approx(0.5)


def test_coverage_with_nothing_read_is_zero_not_a_crash():
    assert dp.Bucket(window_s=20.0).coverage([], 0.15) == 0.0


# ------------------------------------------------------- it recommends nothing

def test_the_probe_never_picks_a_threshold():
    """Choosing the cut that yields the most signals is fitting the knob
    to the answer."""
    import ast
    import inspect

    # The CODE, not what it prints. The output deliberately warns against
    # picking a threshold, so a grep over the text fails on the warning --
    # and a regex over quotes mis-pairs across f-strings, so every string
    # constant is blanked through the tree instead.
    class Blank(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(value=""), node)
            return node

        def visit_JoinedStr(self, node):
            return ast.copy_location(ast.Constant(value=""), node)

    tree = Blank().visit(ast.parse(inspect.getsource(dp)))
    ast.fix_missing_locations(tree)
    code = ast.unparse(tree).lower()
    for word in ("best", "recommend", "optimal", "suggest", "tune"):
        assert word not in code, word
    # And it never sorts or maximises the measures by how much they fire.
    assert "sorted(cuts" not in code
    assert "max(cuts" not in code
    assert "key=lambda" not in code


def test_the_output_says_what_the_curve_is_for():
    src = (ROOT / "scripts" / "dom_probe.py").read_text()
    assert "NOT a" in src and "fitting the knob" in src


def test_nothing_here_places_an_order():
    import inspect

    src = inspect.getsource(dp).lower()
    for word in ("def submit", "def execute", "def place", "def order"):
        assert word not in src


# ------------------------------------------------------------- the walk

def _stream(seconds=400, bid_sz=40.0, ask_sz=40.0, churn=None):
    items = []
    for s in range(0, seconds):
        for k in range(4):
            ts = s + k * 0.25
            b, a = (churn(s) if churn else (bid_sz, ask_sz))
            items.append(("book", book(ts, bid_sz=b, ask_sz=a)))
            if k == 0:
                items.append(("trade", trade(ts + 0.01)))
    return items


def test_the_probe_reads_every_window_at_each_sample(passthrough):
    got = dp.probe(_stream(400), every_s=20.0)
    assert set(got) == set(dp.WINDOWS)
    assert all(b.n > 0 for b in got.values()), {w: b.n for w, b in got.items()}


def test_it_waits_a_full_window_before_its_first_read(passthrough):
    """Otherwise the earliest reads are short simply because the file had
    only just started, and they drag every percentile."""
    got = dp.probe(_stream(400), every_s=20.0)
    longest = got[max(dp.WINDOWS)]
    # 400s of tape, first read at 60s, every 20s after.
    assert longest.n <= (400 - dp.LONGEST) / 20 + 1


def test_a_book_that_never_changes_produces_no_edge(passthrough):
    """Nothing added, nothing cancelled, so there is nothing to lean."""
    got = dp.probe(_stream(300), every_s=20.0)
    b = got[20.0]
    assert all(abs(x) < 1e-9 for x in b.pull_edge)


def test_a_side_being_pulled_shows_up_in_the_edge(passthrough):
    """The ask thinning steadily while the bid holds."""
    def churn(s):
        return 40.0, max(2.0, 40.0 - (s % 50) * 0.8)

    got = dp.probe(_stream(400, churn=churn), every_s=20.0)
    b = got[20.0]
    assert b.measured > 0
    assert any(abs(x) > 0 for x in b.cancel_edge), "no cancels detected"


def test_every_bucket_reports_how_often_it_could_measure_anything(
        passthrough):
    got = dp.probe(_stream(300), every_s=20.0)
    for w, b in got.items():
        d = b.to_dict()
        assert 0.0 <= d["measured_share"] <= 1.0
        assert set(d["pull_edge"]) == {str(c) for c in dp.CUTS}


def test_the_longest_window_is_the_one_the_watcher_is_sized_for():
    """Reading a sixty-second window off a watcher that only keeps twenty
    would silently report the shorter one."""
    import inspect

    src = inspect.getsource(dp.probe)
    assert "window_s=LONGEST" in src
    assert "max_snaps=int(LONGEST" in src
