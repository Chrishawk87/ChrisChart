"""Does a held book lean predict better than a fresh one?

The measurement is simple; the ways it lies are not.

A LAPSED LEAN DID NOT HOLD. If `held_s` survives the book going flat,
then a direction that appeared, vanished for twenty seconds and came back
lands in the thirty-second bucket having never lasted five. The bucket
fills with the opposite of what it claims to contain.

A THIN BUCKET IS A COUNT, NOT A RATE. The book is flat most of the time,
so long holds may be rare. A percentage printed over nine readings reads
as a finding.

AND THE FIVE-MINUTE CUT IS PRICE. Agreement with a price-derived direction
is partly agreement with recent price, which is the lagging-confluence
problem again. The split has to keep the disagreeing half visible.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load():
    path = ROOT / "scripts" / "book_persist.py"
    spec = importlib.util.spec_from_file_location("book_persist", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["book_persist"] = mod
    spec.loader.exec_module(mod)
    return mod


bp = _load()


def book(ts, px=7800.0, bid_sz=40.0, ask_sz=40.0, n=10):
    from liqmap.flow import Book, Level
    return Book(coin="ES", ts=ts,
                bids=[Level(px=px - 0.25 * (j + 1), sz=bid_sz)
                      for j in range(n)],
                asks=[Level(px=px + 0.25 * (j + 1), sz=ask_sz)
                      for j in range(n)])


def trade(ts, px):
    from liqmap.flow import Trade
    return Trade(px=px, sz=1.0, aggressor="buy", ts=ts)


@pytest.fixture()
def passthrough(monkeypatch):
    from liqmap import dbn

    monkeypatch.setattr(dbn, "stream",
                        lambda recs, flip=False, **kw: iter(recs))


def stream(spans, px=7800.0):
    """`spans` is [(seconds, lean)] — lean>0 thins the ask.

    Microprice leans toward the side with LESS size, so a thin ask is what
    reads up. Getting this backwards is how a fixture quietly tests the
    opposite of what it claims.
    """
    items = []
    t = 0.0
    for secs, lean in spans:
        for _ in range(int(secs)):
            for k in range(4):
                items.append(("book", book(t + k * 0.25,
                                           px=px,
                                           bid_sz=40.0 + lean * 25,
                                           ask_sz=max(2.0, 40.0 - lean * 25))))
            items.append(("trade", trade(t + 0.9, px)))
            t += 1.0
    return items


# ---------------------------------------------- a lapsed lean did not hold

def test_going_flat_breaks_the_hold(passthrough):
    """THE one that silently fills the long buckets with short events."""
    items = stream([(40, 1), (20, 0), (40, 1)])
    looks, _, _ = bp.walk(items, every_s=1.0)
    assert looks
    # After the flat stretch the hold must restart, so nothing in the
    # second run can claim to have held through it.
    late = [l for l in looks if l.ts > 65.0]
    assert late, "no readings after the flat stretch"
    assert max(l.held_s for l in late) < 40.0


def test_flipping_breaks_the_hold(passthrough):
    looks, _, _ = bp.walk(stream([(40, 1), (40, -1)]), every_s=1.0)
    sells = [l for l in looks if l.side == "sell"]
    assert sells
    assert max(s.held_s for s in sells) < 45.0


def test_a_lean_that_holds_accumulates(passthrough):
    looks, _, _ = bp.walk(stream([(90, 1)]), every_s=1.0)
    assert looks
    assert max(l.held_s for l in looks) > 60.0
    assert all(l.side == "buy" for l in looks)


def test_the_hold_starts_at_zero_not_at_the_sample_cadence(passthrough):
    looks, _, _ = bp.walk(stream([(30, 1)]), every_s=1.0)
    assert min(l.held_s for l in looks) == pytest.approx(0.0, abs=1.5)


# --------------------------------------------- a thin bucket is not a rate

def test_a_bucket_below_the_floor_reports_a_count_and_nothing_else(capsys):
    pop = [bp.Look(ts=float(i), side="buy", held_s=1.0, mid=100.0,
                   entry=100.25) for i in range(5)]
    for l in pop:
        l.moved = {h: True for h in bp.HORIZONS}
        l.reached = {t: True for t in bp.TICKS}
    out = bp.report("tiny", pop)
    text = capsys.readouterr().out
    assert out["reported"] is False
    assert "not a measurement" in text
    assert "100.0%" not in text          # no rate printed at all


def test_a_bucket_at_the_floor_does_report(capsys):
    pop = [bp.Look(ts=float(i), side="buy", held_s=1.0, mid=100.0,
                   entry=100.25) for i in range(bp.MIN_BUCKET)]
    for l in pop:
        l.moved = {h: True for h in bp.HORIZONS}
        l.reached = {t: True for t in bp.TICKS}
    out = bp.report("enough", pop)
    assert out["reported"] is True
    assert "direction" in out and "ticks" in out


def test_the_floor_is_not_so_low_that_a_handful_reads_as_a_finding():
    assert bp.MIN_BUCKET >= 30


def test_the_buckets_cover_every_hold_with_no_gap_or_overlap():
    edges = [lo for lo, _ in bp.BUCKETS] + [bp.BUCKETS[-1][1]]
    assert edges[0] == 0.0
    assert edges[-1] == float("inf")
    for (lo, hi), nxt in zip(bp.BUCKETS, bp.BUCKETS[1:]):
        assert hi == nxt[0]


# ------------------------------------------------------- the 5-minute cut

def test_agreement_and_disagreement_are_both_kept():
    up_with = bp.Look(ts=0.0, side="buy", held_s=1.0, mid=100.0,
                      entry=100.25, slow="up")
    up_against = bp.Look(ts=0.0, side="buy", held_s=1.0, mid=100.0,
                         entry=100.25, slow="down")
    assert up_with.agrees and not up_with.against
    assert up_against.against and not up_against.agrees


def test_a_flat_five_minute_bar_is_neither():
    l = bp.Look(ts=0.0, side="buy", held_s=1.0, mid=100.0, entry=100.25,
                slow="flat")
    assert not l.agrees and not l.against
    l2 = bp.Look(ts=0.0, side="buy", held_s=1.0, mid=100.0, entry=100.25)
    assert not l2.agrees and not l2.against


def test_the_five_minute_direction_is_measured_from_the_bars_open(
        passthrough):
    """Not from the previous bar's close, and not from the session open."""
    import inspect

    src = inspect.getsource(bp.walk)
    assert "math.floor(item.ts / SLOW_S) * SLOW_S" in src
    assert "bar_open_px" in src


def test_the_slow_bar_is_five_minutes():
    assert bp.SLOW_S == 300.0


# ----------------------------------------------- no new strategy in here

def test_it_invents_no_threshold_of_its_own():
    import ast
    import inspect

    class Blank(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(value=""), node)
            return node

        def visit_JoinedStr(self, node):
            return ast.copy_location(ast.Constant(value=""), node)

    tree = Blank().visit(ast.parse(inspect.getsource(bp)))
    ast.fix_missing_locations(tree)
    code = ast.unparse(tree)
    assert "r.direction" in code
    assert "abs(r.score)" not in code
    assert "r.score >" not in code


def test_it_calls_bookread_as_written():
    import inspect

    src = inspect.getsource(bp)
    assert "BookReader()" in src
    for banned in (".W =", "FLAT_BAND =", "min_conviction"):
        assert banned not in src, banned


def test_no_production_module_is_mutated():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(bp))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute):
                    assert getattr(t.value, "id", "") not in ("dbn", "br")


# ------------------------------------------------------------- the scoring

def tape_of(pairs):
    t = bp.Tape()
    for ts, px in pairs:
        t.add(ts, px)
    return t


def test_the_target_before_the_stop_is_a_win():
    t = tape_of([(1.0, 100.0), (2.0, 101.25)])
    assert t.first_touch(0.5, 101.25, 98.75, True) is True


def test_prints_before_the_reading_cannot_resolve_it():
    t = tape_of([(1.0, 105.0), (2.0, 100.0)])
    assert t.first_touch(1.5, 101.25, 98.75, True) is None


def test_a_mid_that_went_nowhere_is_neither_right_nor_wrong():
    looks = [bp.Look(ts=0.0, side="buy", held_s=1.0, mid=100.0,
                     entry=100.25),
             bp.Look(ts=1.0, side="buy", held_s=2.0, mid=100.0,
                     entry=100.25)]
    bp.score(looks, bp.Tape())
    assert looks[0].moved[1.0] is None


def test_unresolved_is_counted_apart_from_the_rate():
    assert bp.rate([True, False, None]) == (1, 2, 1)


def test_entry_crosses_the_spread():
    import inspect

    assert "book.best_ask if long else book.best_bid" in inspect.getsource(
        bp.walk)


def test_nothing_here_places_an_order():
    import inspect

    src = inspect.getsource(bp).lower()
    for word in ("def submit", "def execute", "def place", "def order"):
        assert word not in src
