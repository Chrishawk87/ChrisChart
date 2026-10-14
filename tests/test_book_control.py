"""The control study: is the book adding anything, or is it a price pattern?

The whole value of this script is that two arms are comparable. Everything
below protects that.

THE ARMS MUST BE SCORED IDENTICALLY. Same timestamps, same crossed entry,
same first touch. If the blind fade and the book arm differ in method, the
comparison measures the method.

THE BLIND ARM MUST BE BLIND. `fade` may not consult the book in any way.
If it does, there is no control, only two views of the same thing.

NO CUT MAY BE CHOSEN. The depth measures are continuous. Picking the
threshold that makes an arm look best is fitting the knob to the answer,
so all three are reported as a row.

AND A THIN ARM IS A COUNT. A percentage over nine readings reads as a
finding.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load():
    path = ROOT / "scripts" / "book_control.py"
    spec = importlib.util.spec_from_file_location("book_control", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["book_control"] = mod
    spec.loader.exec_module(mod)
    return mod


bc = _load()


def look(slow="up", bookside="", cancel=None, notional=None, ts=0.0,
         mid=100.0, bid=99.875, ask=100.125):
    return bc.Look(ts=ts, mid=mid, bid=bid, ask=ask, slow=slow,
                   book=bookside, cancel_edge=cancel, depth_edge=notional)


# ------------------------------------------------------ the blind arm is blind

def test_the_fade_opposes_the_bar_and_never_looks_at_the_book():
    assert look(slow="up").fade == "sell"
    assert look(slow="down").fade == "buy"
    assert look(slow="flat").fade == ""
    assert look(slow="").fade == ""
    # The book says one thing; the fade is unmoved by it.
    assert look(slow="up", bookside="buy").fade == "sell"
    assert look(slow="up", bookside="sell").fade == "sell"


def test_the_fade_property_cannot_reach_the_book_field():
    """A control that consults what it controls is not a control."""
    import inspect

    src = inspect.getsource(bc.Look.fade.fget)
    assert "self.book" not in src
    assert "cancel_edge" not in src and "depth_edge" not in src


def test_follow_is_the_other_blind_baseline():
    assert look(slow="up").follow == "buy"
    assert look(slow="down").follow == "sell"
    assert look(slow="flat").follow == ""


def test_the_two_blind_arms_are_exact_opposites():
    for s in ("up", "down"):
        f, g = look(slow=s).fade, look(slow=s).follow
        assert f and g and f != g


# --------------------------------------------------------- no cut is chosen

def test_three_cuts_are_reported_and_none_is_preferred():
    assert len(bc.CUTS) == 3
    import ast
    import inspect

    class Blank(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(value=""), node)
            return node

        def visit_JoinedStr(self, node):
            return ast.copy_location(ast.Constant(value=""), node)

    tree = Blank().visit(ast.parse(inspect.getsource(bc)))
    ast.fix_missing_locations(tree)
    code = ast.unparse(tree).lower()
    # Specific enough not to collide with the book API: `best_bid` is
    # not a recommendation.
    for word in ("optimal", "recommend", "argmax", "pick_cut",
                 "best_cut", "tune"):
        assert word not in code, word
    assert "max(cuts" not in code
    assert "sorted(cuts" not in code


def test_a_measure_below_its_cut_says_nothing():
    assert look(cancel=0.05).edge_side("cancel", 0.10) == ""
    assert look(cancel=0.50).edge_side("cancel", 0.10) == "buy"
    assert look(cancel=-0.50).edge_side("cancel", 0.10) == "sell"


def test_an_unmeasured_depth_reading_says_nothing():
    assert look(cancel=None).edge_side("cancel", 0.10) == ""
    assert look(notional=None).edge_side("notional", 0.10) == ""


def test_the_two_depth_measures_are_not_the_same_field():
    """Selecting by an else-branch would score one measure twice and
    report it under two labels."""
    l = look(cancel=0.9, notional=-0.9)
    assert l.edge_side("cancel", 0.1) == "buy"
    assert l.edge_side("notional", 0.1) == "sell"


def test_an_unknown_measure_raises_rather_than_falling_through():
    with pytest.raises(KeyError):
        look(cancel=0.9).edge_side("typo", 0.1)


# ------------------------------------------------- every arm scored the same

def test_entry_crosses_the_spread_for_both_sides():
    import inspect

    src = inspect.getsource(bc.score_arm)
    assert "entry = l.ask if long else l.bid" in src
    assert "l.mid" not in src.split("entry =")[1].split("for t in")[0]


def test_one_scorer_serves_every_arm():
    """If arms had their own scorers the comparison would measure the
    scorers."""
    import inspect

    src = inspect.getsource(bc)
    assert src.count("def score_arm") == 1
    assert src.count("first_touch(l.ts") == 1


def test_a_short_is_scored_the_other_way_round():
    t = bc.Tape()
    for ts, px in ((1.0, 100.0), (2.0, 98.0)):
        t.add(ts, px)
    looks = [look(ts=0.0)]
    mids = [(0.0, 100.0)]
    down = bc.score_arm([(looks[0], "sell")], t, mids)
    assert down["reach"][5][0] is True
    up = bc.score_arm([(looks[0], "buy")], t, mids)
    assert up["reach"][5][0] is False


def test_a_mid_that_went_nowhere_is_neither_right_nor_wrong():
    looks = [look(ts=0.0, mid=100.0), look(ts=1.0, mid=100.0)]
    mids = [(l.ts, l.mid) for l in looks]
    s = bc.score_arm([(looks[0], "buy")], bc.Tape(), mids)
    assert s["moved"][1.0][0] is None


def test_prints_before_the_reading_cannot_resolve_it():
    t = bc.Tape()
    t.add(1.0, 105.0)
    t.add(2.0, 100.0)
    assert t.first_touch(1.5, 101.25, 98.75, True) is None


def test_unresolved_is_apart_from_the_rate():
    assert bc.rate([True, False, None, None]) == (1, 2, 2)


def test_a_thin_arm_is_a_count_not_a_rate():
    assert bc.MIN_ARM >= 30


def test_the_cell_shows_nothing_when_nothing_resolved():
    assert "—" in bc.cell([None, None])


# ------------------------------------------------------- no strategy in here

def test_it_invents_no_direction_threshold():
    import inspect

    src = inspect.getsource(bc.walk)
    assert 'r.direction != "flat"' in src
    assert "abs(r.score)" not in src


def test_it_calls_bookread_and_dom_as_written():
    import inspect

    src = inspect.getsource(bc)
    assert "BookReader()" in src
    assert "DomWatch(" in src
    for banned in (".W =", "FLAT_BAND =", "MIN_OVERLAP ="):
        assert banned not in src, banned


def test_no_production_module_is_mutated():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(bc))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute):
                    assert getattr(t.value, "id", "") not in ("dbn",)


def test_nothing_here_places_an_order():
    import inspect

    src = inspect.getsource(bc).lower()
    for word in ("def submit", "def execute", "def place", "def order"):
        assert word not in src


# ------------------------------------------------------------- the walk

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


def test_the_five_minute_direction_restarts_each_bar(passthrough):
    """Measured from THIS bar's open, not the session's and not the
    previous bar's close."""
    items = []
    px = 7800.0
    for s in range(700):
        # price rises through bar one, falls through bar two
        px += 0.25 if s < 300 else -0.25
        for k in range(4):
            items.append(("book", book(s + k * 0.25, px=px)))
        items.append(("trade", trade(s + 0.9, px)))
    looks, _, _ = bc.walk(items, every_s=10.0)
    first = [l for l in looks if 60 < l.ts < 290]
    second = [l for l in looks if 360 < l.ts < 590]
    assert first and second
    assert all(l.slow == "up" for l in first)
    assert all(l.slow == "down" for l in second)


def test_the_walk_keeps_every_print_and_samples_the_book(passthrough):
    items = []
    for s in range(120):
        for k in range(4):
            items.append(("book", book(s + k * 0.25)))
        items.append(("trade", trade(s + 0.9, 7800.0)))
    looks, tape, reads = bc.walk(items, every_s=10.0)
    assert len(tape.ts) == 120
    assert reads > 0 and len(looks) <= reads


def test_the_depth_notional_spans_ten_levels_not_five(passthrough):
    """The point of the arm is to read what the ten levels contain."""
    import inspect

    src = inspect.getsource(bc.walk)
    assert "book.bids[:10]" in src and "book.asks[:10]" in src
