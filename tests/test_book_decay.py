"""The book-decay runner: a reader over `bookread`, not a new strategy.

Four things would turn this from a measurement into a flattering story.

IT MUST NOT INVENT A THRESHOLD. `BookRead.direction` already applies the
module's own FLAT_BAND. A script that compares the score against a number
chosen here is a different strategy wearing the same name.

ENTRY MUST PAY THE SPREAD. Measuring from the mid hands back half a tick
nobody ever gets, and at a 5-tick target that is 10% of the move.

FIRST TOUCH MUST COME FROM THE TAPE, in order. Prints are ordered and a
single print cannot be on both sides, so there is no tie to resolve
favourably -- but only if the walk is actually forward from the reading.

AND UNRESOLVED IS NOT A LOSS. A reading whose levels were never reached
says nothing; counting it as a loss would punish quiet markets and
counting it as a win would be worse.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load():
    path = ROOT / "scripts" / "book_decay.py"
    spec = importlib.util.spec_from_file_location("book_decay", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["book_decay"] = mod
    spec.loader.exec_module(mod)
    return mod


bd = _load()


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


# ------------------------------------------- it owns no threshold of its own

def test_the_direction_comes_from_bookread_not_from_a_number_here():
    """A gate invented in a research script is a different strategy."""
    import ast
    import inspect

    class Blank(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(value=""), node)
            return node

        def visit_JoinedStr(self, node):
            return ast.copy_location(ast.Constant(value=""), node)

    tree = Blank().visit(ast.parse(inspect.getsource(bd)))
    ast.fix_missing_locations(tree)
    code = ast.unparse(tree)
    assert 'r.direction ==' in code
    # No comparison of the score against a locally chosen number.
    assert "abs(r.score)" not in code
    assert "flat_gate" not in code


def test_it_calls_bookread_and_suggest_as_written():
    import inspect

    src = inspect.getsource(bd)
    assert "BookReader()" in src
    assert "sg.suggest(" in src
    # No reweighting, no re-scoring, no parameter overrides on the gates.
    for banned in ("min_rr=", "min_conviction=", "max_breakeven=",
                   "min_cost_multiple=", ".W =", "W["):
        assert banned not in src, banned


def test_the_reader_is_fed_every_push_not_only_at_the_samples():
    """Its dynamics are accumulated from the stream. Feeding it only at
    the sample points gives it a far poorer history than it has live."""
    import inspect

    src = inspect.getsource(bd.walk)
    add = src.index("reader.add(book, now=book.ts)")
    gate = src.index("if ts < next_at:")
    assert add < gate, "the sample gate must come after the feed"


# ------------------------------------------------------- entry pays the spread

def test_a_long_enters_at_the_ask_and_a_short_at_the_bid():
    import inspect

    src = inspect.getsource(bd.walk)
    assert "book.best_ask if long else book.best_bid" in src


def test_targets_are_measured_from_the_entry_not_the_mid():
    import inspect

    src = inspect.getsource(bd.score_looks)
    assert "l.entry + d if long else l.entry - d" in src
    assert "l.mid + d" not in src


# ------------------------------------------------------------- first touch

def tape_of(pairs):
    t = bd.Tape()
    for ts, px in pairs:
        t.add(ts, px)
    return t


def test_the_target_coming_first_is_a_win():
    t = tape_of([(1.0, 100.0), (2.0, 101.25), (3.0, 98.0)])
    assert t.first_touch(0.5, target=101.25, stop=98.75, long=True) is True


def test_the_stop_coming_first_is_a_loss():
    t = tape_of([(1.0, 100.0), (2.0, 98.5), (3.0, 101.5)])
    assert t.first_touch(0.5, target=101.25, stop=98.75, long=True) is False


def test_prints_before_the_reading_cannot_resolve_it():
    """Look-ahead in reverse: a level touched before the call is not the
    call being right."""
    t = tape_of([(1.0, 105.0), (2.0, 100.0)])
    assert t.first_touch(1.5, target=101.25, stop=98.75, long=True) is None


def test_a_reading_that_never_resolves_is_none_not_a_loss():
    t = tape_of([(1.0, 100.0), (2.0, 100.25)])
    assert t.first_touch(0.5, target=101.25, stop=98.75, long=True) is None


def test_nothing_resolves_past_the_holding_limit():
    t = tape_of([(1.0, 100.0), (bd.MAX_HOLD_S + 50.0, 105.0)])
    assert t.first_touch(0.5, target=101.25, stop=98.75, long=True) is None


def test_a_short_is_scored_the_other_way_round():
    t = tape_of([(1.0, 100.0), (2.0, 98.75)])
    assert t.first_touch(0.5, target=98.75, stop=101.25, long=False) is True
    t2 = tape_of([(1.0, 100.0), (2.0, 101.25)])
    assert t2.first_touch(0.5, target=98.75, stop=101.25, long=False) is False


# ----------------------------------------------------- unresolved is counted

def test_unresolved_readings_are_reported_apart_from_the_rate():
    hits, n, none = bd.rate([True, True, False, None, None])
    assert (hits, n, none) == (2, 3, 2)


def test_a_rate_over_nothing_is_not_a_crash():
    assert bd.rate([None, None]) == (0, 0, 2)


def test_the_interval_is_wide_at_small_n_and_tight_at_large():
    lo, hi = bd.wilson(6, 9)
    assert hi - lo > 0.4
    lo2, hi2 = bd.wilson(6000, 9000)
    assert hi2 - lo2 < 0.03


# --------------------------------------------------------------- the decay

def test_a_mid_that_went_nowhere_is_neither_right_nor_wrong():
    """Counting a dead flat move as a win for whoever guessed is the
    easiest way to manufacture a hit rate."""
    looks = [bd.Look(ts=0.0, side="buy", score=0.5, mid=100.0, entry=100.25,
                     spread_bps=0.3),
             bd.Look(ts=1.0, side="buy", score=0.5, mid=100.0, entry=100.25,
                     spread_bps=0.3)]
    bd.score_looks(looks, bd.Tape())
    assert looks[0].moved[1.0] is None


def test_the_mid_moving_the_predicted_way_is_right():
    looks = [bd.Look(ts=0.0, side="buy", score=0.5, mid=100.0, entry=100.25,
                     spread_bps=0.3),
             bd.Look(ts=1.0, side="buy", score=0.5, mid=100.5, entry=100.75,
                     spread_bps=0.3)]
    bd.score_looks(looks, bd.Tape())
    assert looks[0].moved[1.0] is True


def test_a_short_whose_mid_fell_is_right():
    looks = [bd.Look(ts=0.0, side="sell", score=-0.5, mid=100.0,
                     entry=99.75, spread_bps=0.3),
             bd.Look(ts=1.0, side="sell", score=-0.5, mid=99.5,
                     entry=99.25, spread_bps=0.3)]
    bd.score_looks(looks, bd.Tape())
    assert looks[0].moved[1.0] is True


def test_a_horizon_past_the_end_of_the_file_is_unresolved():
    looks = [bd.Look(ts=0.0, side="buy", score=0.5, mid=100.0, entry=100.25,
                     spread_bps=0.3)]
    bd.score_looks(looks, bd.Tape())
    assert all(looks[0].moved[h] is None for h in bd.HORIZONS)


def test_the_horizons_span_a_second_to_a_minute():
    assert min(bd.HORIZONS) <= 1.0
    assert max(bd.HORIZONS) >= 60.0


# ------------------------------------------------------------- the walk

def _stream(seconds=200, lean=0.0):
    """Book pushes four times a second, a print every second.

    `lean` thins the ask, which is what tilts the microprice up.
    """
    items = []
    for s in range(seconds):
        for k in range(4):
            ts = s + k * 0.25
            items.append(("book", book(ts, bid_sz=40.0,
                                       ask_sz=max(2.0, 40.0 - lean))))
        items.append(("trade", trade(s + 0.9, 7800.0)))
    return items


def test_the_walk_samples_on_the_cadence_and_keeps_every_print(passthrough):
    items = _stream(120)
    looks, gates, tape, reads = bd.walk(items, every_s=10.0)
    assert len(tape.ts) == 120            # every print kept
    assert reads <= 13                    # ~120s at 10s, after one warmup
    assert sum(gates.values()) == reads


def test_a_balanced_book_produces_no_readings(passthrough):
    """Equal size either side is exactly the case `direction` calls flat."""
    looks, _, _, reads = bd.walk(_stream(120, lean=0.0), every_s=10.0)
    assert reads > 0
    assert looks == []


def test_a_thin_ask_produces_buy_readings(passthrough):
    """Less size on the offer is the side that gets consumed first, which
    is what microprice tilt is built on."""
    looks, _, _, _ = bd.walk(_stream(200, lean=36.0), every_s=10.0)
    assert looks, "no readings taken from a leaning book"
    assert all(l.side == "buy" for l in looks)
    assert all(l.entry > l.mid for l in looks)      # paid the ask


def test_every_sampled_read_gets_a_gate_even_when_it_is_not_a_call(
        passthrough):
    """How often the full rule refuses, and why, is one of the two
    questions -- so the breakdown must cover every read, not only the
    ones that survive to become readings."""
    looks, gates, _, reads = bd.walk(_stream(120, lean=0.0), every_s=10.0)
    assert not looks
    assert sum(gates.values()) == reads > 0


def test_nothing_here_places_an_order():
    import inspect

    src = inspect.getsource(bd).lower()
    for word in ("def submit", "def execute", "def place", "def order"):
        assert word not in src


def test_no_production_module_is_mutated():
    """A research script that writes into the engine is a change to the
    engine."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(bd))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Attribute):
                    base = getattr(t.value, "id", "")
                    assert base not in ("sg", "br", "dbn"), ast.dump(t)
