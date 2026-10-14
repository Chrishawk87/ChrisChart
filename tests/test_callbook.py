"""Every call written down before its answer exists, and scored honestly.

Four ways this would become a machine for flattering the engine, and a
test for each.

THE QUESTION MUST BE WRITTEN BEFORE THE ANSWER. A scorer that can see the
outcome while recording the call will report whatever its author hoped for.
`record` must not be able to touch the outcome columns at all.

A RATE NEEDS A BASELINE. A bar already up at halfway usually closes up, so
an engine that says "up" when price is up scores well above a coin flip
while contributing nothing. The naive call is scored on the same bars.

AND AGREEMENT IS THE QUIET KILLER. Two reads that say the same thing 95% of
the time cannot have an edge over each other, whatever their rates are.

AND NOTHING IS DROPPED. A filter in the scorer is indistinguishable from a
better strategy.
"""

from __future__ import annotations

import pytest

from liqmap import callbook as cb


def _code_of(fn) -> str:
    """A function's CODE, with its docstring removed.

    These functions EXPLAIN the rules they follow, so a grep over the raw
    source fails on the explanation rather than on a violation -- which it
    did, every time, until this existed.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            b = node.body
            if (b and isinstance(b[0], ast.Expr)
                    and isinstance(b[0].value, ast.Constant)
                    and isinstance(b[0].value.value, str)):
                b.pop(0)
    return ast.unparse(tree)


@pytest.fixture()
def book(tmp_path):
    return cb.CallBook(tmp_path / "calls.db")


def call(side="buy", mark=50, bar_ts=0.0, open_px=100.0, call_px=100.0,
         kind="cascade", coin="BTC", timeframe="15m", detail=""):
    return cb.Call(coin=coin, timeframe=timeframe, bar_ts=bar_ts, kind=kind,
                   mark=mark, side=side, detail=detail, open_px=open_px,
                   call_px=call_px)


# ------------------------------------- the question comes before the answer

def test_record_cannot_write_an_outcome():
    """THE guarantee. The columns are not in the statement, so a call
    cannot be born with its own answer attached."""
    for col in ("outcome", "close_px", "resolved_at"):
        assert col not in _code_of(cb.CallBook.record), col
    # And resolve is the only thing that can write them.
    for col in ("outcome", "close_px", "resolved_at"):
        assert col in _code_of(cb.CallBook.resolve), col


def test_a_fresh_call_has_no_outcome(book):
    book.record(call())
    assert book.counts() == {"calls": 1, "resolved": 0, "open": 1}
    assert book.scorecard() == []


def test_the_outcome_arrives_only_from_resolve(book):
    book.record(call(side="buy", open_px=100.0, call_px=100.2))
    assert book.resolve("BTC", "15m", 0.0, close_px=101.0) == 1
    t = book.scorecard()[0]
    assert t.scored == 1 and t.hits == 1


def test_a_resolved_call_is_never_rewritten(book):
    """A feed that reconnects and re-closes a bar must not be able to
    change an answer that has already been counted."""
    book.record(call(side="buy"))
    book.resolve("BTC", "15m", 0.0, close_px=101.0)
    assert book.resolve("BTC", "15m", 0.0, close_px=99.0) == 0
    assert book.scorecard()[0].hits == 1


def test_polling_the_same_bar_does_not_stack_up_opinions(book):
    """A page polling four times a second would otherwise file ninety
    rows for one bar, and let a lucky fifteen minutes outvote a week."""
    for _ in range(20):
        book.record(call(side="buy"))
    assert book.counts()["calls"] == 1


def test_the_first_read_at_a_checkpoint_is_the_one_kept(book):
    """Not the last. The last is the read taken closest to the close,
    which is the one that knows the most."""
    book.record(call(side="buy", call_px=100.1))
    book.record(call(side="sell", call_px=100.9))
    book.resolve("BTC", "15m", 0.0, close_px=101.0)
    t = book.scorecard()[0]
    assert t.hits == 1          # the buy, recorded first, was right


# ---------------------------------------------------------- the checkpoints

def test_a_read_lands_in_the_latest_checkpoint_it_has_passed():
    """Awarding it to the next one up would file a read as earlier than it
    was, which flatters exactly what this measures."""
    assert cb.mark_for(60.0, 100.0) == 50
    assert cb.mark_for(25.0, 100.0) == 25
    assert cb.mark_for(99.0, 100.0) == 75


def test_a_read_before_the_first_checkpoint_is_not_recorded_at_all():
    assert cb.mark_for(1.0, 100.0) is None
    assert cb.mark_for(24.9, 100.0) is None


def test_a_bar_with_no_length_has_no_checkpoint():
    assert cb.mark_for(10.0, 0.0) is None


def test_the_three_checkpoints_are_kept_apart(book):
    for m in cb.MARKS:
        book.record(call(side="buy", mark=m))
    book.resolve("BTC", "15m", 0.0, close_px=101.0)
    assert [t.mark for t in book.scorecard()] == list(cb.MARKS)


def test_a_checkpoint_nobody_declared_is_refused(book):
    assert book.record(call(mark=99)) is False
    assert book.counts()["calls"] == 0


def test_a_call_with_no_open_price_is_refused(book):
    """Everything here is measured against the bar's open. Without one
    there is nothing to be right or wrong about."""
    assert book.record(call(open_px=0.0)) is False


# ------------------------------------------------- the baseline it must beat

def test_the_naive_call_is_where_price_was_already_going():
    assert call(open_px=100.0, call_px=100.5).naive == "up"
    assert call(open_px=100.0, call_px=99.5).naive == "down"
    assert call(open_px=100.0, call_px=100.0).naive == "flat"


def test_the_naive_call_is_computed_here_not_supplied(book):
    """A caller cannot hand in a flattering baseline, because it is not
    an input."""
    import inspect

    assert "naive" not in inspect.signature(cb.Call.__init__).parameters


def test_an_engine_that_only_follows_price_shows_no_edge(book):
    """THE case this file exists for. Saying 'up' because price is up
    scores far above a coin flip and is worth nothing."""
    for i in range(60):
        up = i % 2 == 0
        book.record(call(bar_ts=i, side="buy" if up else "sell",
                         open_px=100.0, call_px=100.5 if up else 99.5))
        book.resolve("BTC", "15m", i, close_px=101.0 if up else 99.0)
    t = book.scorecard()[0]
    assert t.rate == pytest.approx(1.0)
    assert t.naive_rate == pytest.approx(1.0)
    assert t.edge == pytest.approx(0.0)
    assert t.agreement == pytest.approx(1.0)
    assert "nearly the same call" in t.describe()


def test_an_engine_that_calls_against_price_and_is_right_shows_an_edge(book):
    """The shape of a real finding: it disagrees with the naive read and
    turns out to be the one that was right."""
    for i in range(60):
        book.record(call(bar_ts=i, side="sell", open_px=100.0,
                         call_px=100.5))          # price up, it says down
        book.resolve("BTC", "15m", i, close_px=99.0)
    t = book.scorecard()[0]
    assert t.rate == pytest.approx(1.0)
    assert t.naive_rate == pytest.approx(0.0)
    assert t.edge == pytest.approx(1.0)
    assert t.agreement == pytest.approx(0.0)


def test_the_edge_is_none_with_nothing_scored():
    t = cb.Tally(kind="cascade", mark=50)
    assert t.rate is None and t.naive_rate is None and t.edge is None
    assert "nothing scored yet" in t.describe()


# -------------------------------------------------- what is and is not scored

def test_declining_to_call_counts_in_coverage_but_is_never_scored(book):
    """The stack reader stands aside constantly. A rate printed without
    its coverage hides whether it called one bar in twenty or all of
    them."""
    for i in range(10):
        book.record(call(bar_ts=i, side="" if i < 8 else "buy",
                         open_px=100.0, call_px=100.5))
        book.resolve("BTC", "15m", i, close_px=101.0)
    t = book.scorecard()[0]
    assert t.offered == 10 and t.called == 2 and t.scored == 2
    assert t.coverage == pytest.approx(0.2)
    assert "called 20% of bars" in t.describe()


def test_a_bar_that_went_nowhere_is_not_a_win_for_whoever_guessed(book):
    """Same reasoning as discarding mid prints in the aggressor audit: it
    carries no information about the thing being measured."""
    for i in range(10):
        book.record(call(bar_ts=i, side="buy", open_px=100.0, call_px=100.5))
        book.resolve("BTC", "15m", i, close_px=100.0)      # closed flat
    t = book.scorecard()[0]
    assert t.called == 10
    assert t.flat == 10
    assert t.scored == 0
    assert t.rate is None


def test_a_wrong_call_counts_against_the_rate(book):
    for i in range(10):
        book.record(call(bar_ts=i, side="buy", open_px=100.0, call_px=100.5))
        book.resolve("BTC", "15m", i, close_px=99.0)
    t = book.scorecard()[0]
    assert t.scored == 10 and t.hits == 0
    assert t.rate == pytest.approx(0.0)


def test_the_scorecard_drops_nothing(book):
    """A filter in the scorer is indistinguishable from a better
    strategy."""
    assert "LIMIT" not in _code_of(cb.CallBook.scorecard).upper()
    for i in range(20):
        book.record(call(bar_ts=i, side="buy", open_px=100.0, call_px=100.5))
        book.resolve("BTC", "15m", i, close_px=99.0 if i < 15 else 101.0)
    t = book.scorecard()[0]
    assert t.scored == 20 and t.hits == 5


def test_the_scorecard_can_be_narrowed_by_market_not_by_result(book):
    import inspect

    params = inspect.signature(cb.CallBook.scorecard).parameters
    assert set(params) - {"self"} == {"coin", "timeframe", "kind"}
    book.record(call(bar_ts=1, coin="BTC", side="buy", call_px=100.5))
    book.record(call(bar_ts=1, coin="ETH", side="buy", call_px=100.5))
    book.resolve("BTC", "15m", 1, close_px=101.0)
    book.resolve("ETH", "15m", 1, close_px=99.0)
    assert book.scorecard(coin="BTC")[0].hits == 1
    assert book.scorecard(coin="ETH")[0].hits == 0
    assert book.scorecard()[0].scored == 2


def test_two_kinds_of_read_are_tallied_separately(book):
    book.record(call(bar_ts=1, kind="cascade", side="buy", call_px=100.5))
    book.record(call(bar_ts=1, kind="trace", side="sell", call_px=100.5))
    book.resolve("BTC", "15m", 1, close_px=101.0)
    got = {t.kind: t for t in book.scorecard()}
    assert got["cascade"].hits == 1 and got["trace"].hits == 0


# ---------------------------------------------------------------- the edges

def test_resolving_a_bar_nobody_called_is_not_an_error(book):
    assert book.resolve("BTC", "15m", 999.0, close_px=101.0) == 0


def test_a_close_of_zero_resolves_nothing(book):
    book.record(call(side="buy"))
    assert book.resolve("BTC", "15m", 0.0, close_px=0.0) == 0
    assert book.counts()["open"] == 1


def test_open_bars_lists_what_is_still_waiting(book):
    book.record(call(bar_ts=1, side="buy"))
    book.record(call(bar_ts=2, side="buy"))
    book.resolve("BTC", "15m", 1, close_px=101.0)
    assert book.open_bars() == [("BTC", "15m", 2.0)]


def test_pruning_drops_only_old_bars(book):
    book.record(call(bar_ts=1000.0, side="buy"))
    book.record(call(bar_ts=9_000_000.0, side="buy"))
    assert book.prune(older_than_s=100.0, now=9_000_050.0) == 1
    assert book.counts()["calls"] == 1


def test_a_flat_band_keeps_noise_out_of_both_columns():
    assert cb.way_of(100.001, 100.0) == "flat"
    assert cb.way_of(100.5, 100.0) == "up"
    assert cb.way_of(99.5, 100.0) == "down"
    assert cb.way_of(100.0, 0.0) == "flat"


def test_nothing_here_places_an_order():
    import inspect

    src = inspect.getsource(cb).lower()
    for word in ("def submit", "def execute", "def place", "def order"):
        assert word not in src


# --------------------------------------------------------------- the summary

def test_the_summary_refuses_to_conclude_from_a_handful(book):
    for i in range(20):
        book.record(call(bar_ts=i, side="sell", open_px=100.0, call_px=100.5))
        book.resolve("BTC", "15m", i, close_px=99.0)
    line = cb.summarise(book.scorecard())
    assert "too few to conclude" in line


def test_the_summary_says_when_nothing_has_closed_yet(book):
    book.record(call(side="buy"))
    assert "nothing scored yet" in cb.summarise(book.scorecard())
    assert "nothing scored yet" in cb.summarise([])


def test_the_summary_reports_the_best_once_there_is_a_sample(book):
    for i in range(150):
        book.record(call(bar_ts=i, side="sell", open_px=100.0, call_px=100.5))
        book.resolve("BTC", "15m", i, close_px=99.0)
    line = cb.summarise(book.scorecard())
    assert "best:" in line and "edge +100%" in line
