"""Three independent columns, a grade, and how far it can run.

What this pins:

THE COLUMNS DO NOT SHARE INPUTS. `delta.py` must never read an order book
and `bookread.py` must no longer score the tape. Two modules agreeing
because they read the same number is not confirmation, and it is invisible
in the output.

PRICE IS MANDATORY. It is the only column reporting an OUTCOME rather than
an intention. Book and delta can both be screaming while they are being
absorbed, which is the case the whole design exists to refuse.

THE GRADE IS A COUNT, NOT AN AVERAGE. Averaging three directions turns a
disagreement into a confident-looking number and destroys the information
worth keeping.

DIRECTION QUALITY IS NOT HOLDING DISTANCE. Three columns agreeing says the
direction is probably right and says nothing about how far it goes. Runway
answers the second question separately, and a target must never be set
beyond a shelf the book is already showing.
"""

import ast
import inspect

import pytest

from liqmap import delta as delta_mod
from liqmap import runway as runway_mod
from liqmap.bookread import BookRead
from liqmap.confirm import ThreeWay, grade_three
from liqmap.delta import DeltaRead, persistence_of, read_delta
from liqmap.flow import Book, Level
from liqmap.structure import Candle
from liqmap.volume import VolumeProfile


def tape(delta=5e5, volume=1e6, expected=5e5, move=8.0, bar=100.0,
         buckets=(), trades=40):
    return read_delta(delta=delta, volume=volume, expected=expected,
                      price_move_bps=move, bar_range_bps=bar,
                      buckets=buckets, trades=trades)


# --------------------------------------------------------------------------
# independence, asserted structurally
# --------------------------------------------------------------------------

def test_the_delta_module_never_reads_an_order_book():
    tree = ast.parse(inspect.getsource(delta_mod))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.add(getattr(node, "module", "") or "")
            names.update(a.name for a in node.names)
    banned = {"bookread", "BookRead", "Book", "best_bid", "best_ask",
              "microprice", "tilt", "replenish", "bids", "asks"}
    assert not (banned & names), sorted(banned & names)


def test_the_book_no_longer_scores_the_tape():
    """Aggression came from the trade feed and was weighted into the book,
    which put the same input on both sides of the confirmation."""
    assert "aggression" not in BookRead.W
    r = BookRead(ts=0, mid=100, microprice=100, spread_bps=1, tilt=0.8,
                 imbalance=0.5, depletion=-0.4, replenish=0.6,
                 mid_drift_bps=5, samples=50)
    assert not any("aggression" in c["name"] for c in r.components())


def test_setting_aggression_no_longer_moves_the_book_score():
    base = dict(ts=0, mid=100, microprice=100, spread_bps=1, tilt=0.8,
                imbalance=0.5, depletion=-0.4, replenish=0.6,
                mid_drift_bps=5, samples=50)
    quiet = BookRead(**base)
    loud = BookRead(**base, aggression=1.0, absorbed=False)
    assert quiet.score == loud.score


# --------------------------------------------------------------------------
# the delta column
# --------------------------------------------------------------------------

def test_buying_reads_up_and_selling_reads_down():
    assert tape(delta=8e5, volume=1e6).direction == "up"
    assert tape(delta=-8e5, volume=1e6).direction == "down"


def test_a_balanced_tape_is_flat():
    assert tape(delta=0.0, volume=1e6).direction == "flat"


def test_a_one_sided_lean_on_no_volume_is_not_pressure():
    """Three prints agreeing with each other is not the market speaking."""
    loud = tape(delta=9e5, volume=1e6, expected=5e5)
    quiet = tape(delta=9e4, volume=1e5, expected=5e6)
    assert loud.lean == pytest.approx(quiet.lean, abs=0.01)
    assert abs(quiet.score) < abs(loud.score)
    assert quiet.direction == "flat"


def test_absorption_is_aggression_that_goes_nowhere():
    a = tape(delta=8e5, volume=1e6, expected=5e5, move=0.1, bar=100.0)
    assert a.absorbed is True
    assert "absorbing" in a.describe()


def test_aggression_that_moves_price_is_not_absorption():
    assert tape(delta=8e5, volume=1e6, move=25.0).absorbed is False


def test_divergence_is_flow_being_run_over():
    d = tape(delta=8e5, volume=1e6, move=-25.0)
    assert d.divergent is True
    assert "run over" in d.describe()


def test_persistence_separates_an_accumulator_from_one_print():
    """Same net delta, opposite holding decision."""
    steady = persistence_of([100, 110, 95, 105, 100])
    spike = persistence_of([510, -100, 100, -100, 100])
    assert steady > 0.9
    assert spike < steady


def test_persistence_of_nothing_is_zero():
    assert persistence_of([]) == 0.0
    assert persistence_of([0, 0, 0]) == 0.0


def test_working_needs_both_persistence_and_volume():
    steady = [100, 100, 100, 100]
    assert tape(volume=1e6, expected=5e5, buckets=steady).working is True
    assert tape(volume=1e4, expected=5e6, buckets=steady).working is False


def test_supports_is_signed_correctly():
    up = tape(delta=8e5, volume=1e6)
    assert up.supports("up") is True
    assert up.supports("down") is False
    assert up.supports("flat") is False


def test_no_volume_reads_as_nothing_rather_than_raising():
    assert read_delta(0, 0, 1e5, 5.0, 100.0) is None
    assert read_delta(1e5, 1e5, 1e5, 5.0, 0.0) is None


def test_the_dict_is_complete_for_the_panel():
    d = tape().to_dict()
    for k in ("direction", "score", "lean", "effort", "persistence",
              "absorbed", "divergent", "working", "describe"):
        assert k in d


# --------------------------------------------------------------------------
# the grade
# --------------------------------------------------------------------------

def test_all_three_aligned_is_an_a():
    g = grade_three("down", "down", "down")
    assert g.grade == "A" and g.direction == "down" and g.agreeing == 3


def test_price_plus_one_is_a_b():
    assert grade_three("down", "flat", "down").grade == "B"
    assert grade_three("flat", "down", "down").grade == "B"


def test_price_alone_is_a_c():
    g = grade_three("flat", "flat", "down")
    assert g.grade == "C" and g.agreeing == 1


def test_price_against_both_is_a_refusal_not_a_weak_call():
    """Book and delta screaming while price does the opposite is somebody
    absorbing, and averaging it into a small position is the expensive
    mistake."""
    g = grade_three("down", "down", "up")
    assert g.grade == "X"
    assert g.direction == "flat"
    assert g.tradeable is False
    assert "absorbing" in g.detail


def test_intent_without_price_is_not_a_trade():
    """The book and tape can lean for a long time before anything happens.
    Pressure that has not paid is not a setup."""
    g = grade_three("down", "down", "flat")
    assert g.grade == "—" and g.tradeable is False
    assert "has not paid" in g.detail or "not moved" in g.detail


def test_nothing_at_all_says_so():
    assert grade_three("flat", "flat", "flat").grade == "—"


def test_a_split_still_trades_with_price():
    """Book one way, delta the other, price siding with one of them."""
    g = grade_three("up", "down", "down")
    assert g.grade == "B" and g.direction == "down"


def test_the_grade_is_json_safe():
    import json

    json.dumps(grade_three("up", "up", "up").to_dict())


# --------------------------------------------------------------------------
# runway
# --------------------------------------------------------------------------

def book_with(entry=100.0, tick=0.01, shelf_at=None, shelf_sz=50_000.0):
    bids, asks = [], []
    for i in range(20):
        sz = shelf_sz if shelf_at == i else 100.0
        asks.append(Level(round(entry + (i + 1) * tick, 6), sz / (entry or 1)))
        bids.append(Level(round(entry - (i + 1) * tick, 6), sz / (entry or 1)))
    return Book(coin="X", ts=0.0, bids=bids, asks=asks)


def test_a_clear_book_reports_open_road():
    r = runway_mod.measure(100.0, "long", book=book_with())
    assert r.capped is True
    assert r.clear_bps >= 100


def test_a_shelf_ahead_ends_the_runway():
    r = runway_mod.measure(100.0, "long", book=book_with(shelf_at=5))
    assert r.capped is False
    assert r.first.kind == "shelf"
    assert r.clear_bps == pytest.approx(6.0, abs=1.0)
    assert "resting shelf" in r.describe()


def test_a_shelf_behind_you_is_not_an_obstacle():
    """Only what is ahead counts. A wall below does not stop a long."""
    r = runway_mod.measure(100.0, "long", book=book_with(shelf_at=5))
    for o in r.obstacles:
        assert o.price > 100.0


def test_heavy_traded_volume_ahead_is_an_obstacle():
    """The half of 'why is the book thin here' the book cannot answer."""
    p = VolumeProfile()
    for _ in range(10):
        p.add(100.0, 1000.0, "buy")
        p.add(100.05, 4000.0, "sell")
    r = runway_mod.measure(100.0, "long", profile=p)
    assert any(o.kind == "volume" for o in r.obstacles)


def test_the_recent_swing_is_an_obstacle():
    bars = [Candle(ts=i, open=100, high=100.4, low=99.6, close=100)
            for i in range(20)]
    r = runway_mod.measure(100.0, "long", bars=bars)
    assert any(o.kind == "structure" for o in r.obstacles)


def test_a_liquidation_cluster_is_reported_but_does_not_shorten_the_runway():
    """Forced flow is the one obstacle that can ACCELERATE price through a
    level rather than stop it there."""
    r = runway_mod.measure(100.0, "long", book=book_with(),
                           magnet_bps=20.0, magnet_notional=5e6)
    assert any(o.kind == "liquidation" for o in r.obstacles)
    assert r.capped is True                  # still open road
    assert r.clear_bps > 20.0
    liq = next(o for o in r.obstacles if o.kind == "liquidation")
    assert "not a wall" in liq.note


def test_the_nearest_obstacle_wins():
    p = VolumeProfile()
    for _ in range(10):
        p.add(100.0, 1000.0, "buy")
        p.add(100.30, 9000.0, "sell")
    r = runway_mod.measure(100.0, "long", book=book_with(shelf_at=2),
                           profile=p)
    assert r.first.kind == "shelf"           # 3bps beats 30bps


def test_a_target_is_capped_at_the_runway():
    """A target sized from bar range can sit on the far side of a shelf the
    book is showing right now."""
    r = runway_mod.measure(100.0, "long", book=book_with(shelf_at=3))
    assert r.cap(100.0) == pytest.approx(r.clear_bps)
    assert r.cap(1.0) == 1.0
    assert r.holdable(1.0) is True
    assert r.holdable(100.0) is False


def test_a_short_looks_downward():
    r = runway_mod.measure(100.0, "short", book=book_with(shelf_at=4))
    assert r.first is not None
    for o in r.obstacles:
        assert o.price < 100.0


def test_degenerate_input_does_not_raise():
    assert runway_mod.measure(0.0, "long").clear_bps == 0.0
    r = runway_mod.measure(100.0, "long",
                           book=Book(coin="X", ts=0, bids=[], asks=[]))
    assert isinstance(r.clear_bps, float)
    import json

    json.dumps(r.to_dict())
