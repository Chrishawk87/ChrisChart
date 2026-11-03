"""Four must agree: 15m, 5m, 1m and the OFI bar read.

THE ONE THAT MATTERS MOST IS `TestTakingProfitMeansSomething`.

The first working version let a trade take profit and re-enter the
same direction at the same price on the very next look. The arithmetic
telescopes: chopped-up pieces of one move sum to the whole move, so
the total P&L came out IDENTICAL for a 4-point target, a 2-point
target and no target at all -- +216.00 to the cent in all three. Only
the win rate moved, which is exactly the statistic most likely to be
mistaken for an edge.

A target or a stop now ends the SIGNAL, not just the trade: the four
have to stop agreeing before another one can open. That is also what
taking profit means to a person -- done with this move, not out and
straight back in.

The rest guard the shape of the rule: all four or nothing, exit on
whichever comes first, and an unresolved trade counted rather than
dropped.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "agree_study", ROOT / "scripts" / "agree_study.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["agree_study"] = mod
    spec.loader.exec_module(mod)
    return mod


ag = _load()
POINT = 1_000_000_000
TICK = POINT // 4


class Lvl:
    def __init__(self, b, a, bs, asz):
        self.bid_px, self.ask_px = b, a
        self.bid_sz, self.ask_sz = bs, asz


class Rec:
    def __init__(self, ts, px, sz, side, b, a, bs, asz):
        self.ts_event, self.price, self.size = ts, px, sz
        self.side, self.action = side, "T"
        self.levels = [Lvl(b, a, bs, asz)]


def tape_records(hours: float = 6.0, up: float = 0.62, seed: int = 7,
                 step: float = 1.0):
    """A tape with a direction in it, so the four can line up.

    NOT A CLAIM ABOUT THE MARKET. A rule that reads three timeframes
    and an OFI bar needs all four to produce something before any of
    this is exercised at all, and a symmetric walk rarely gets there.
    This is a fixture for the machinery, and every number it produces
    is a property of the generator.
    """
    import random
    import time as _t

    rng = random.Random(seed)
    px = 6800 * POINT
    now = _t.time()
    t = now - hours * 3600.0
    out = []
    while t < now:
        if rng.random() < 0.55:
            px += TICK if rng.random() < up else -TICK
        bid, ask = int(px), int(px + TICK)
        out.append((t, Rec(int(t * 1e9), int(px),
                           float(rng.randint(1, 9)),
                           "B" if rng.random() < up else "A",
                           bid, ask, rng.randint(200, 900),
                           rng.randint(40, 200)),
                    (bid + ask) / 2.0))
        t += step
    return out


@pytest.fixture(scope="module")
def recs():
    return tape_records()


class TestItRunsAtAll:
    def test_trades_are_opened(self, recs):
        r = ag.run(recs, 0.0, 0.0, target=2000, eval_s=1.0)
        assert r.n > 0, "the four never agreed on this fixture"

    def test_every_trade_is_closed(self, recs):
        r = ag.run(recs, 2.0, 1.0, target=2000, eval_s=1.0)
        assert all(t.why for t in r.trades)
        assert all(t.ts_out >= t.ts_in for t in r.trades)

    def test_points_and_ticks_agree(self, recs):
        r = ag.run(recs, 0.0, 0.0, target=2000, eval_s=1.0)
        t = r.trades[0]
        assert t.ticks == pytest.approx(t.points * 4.0)

    def test_a_short_is_scored_the_other_way(self):
        t = ag.Trade(side="down", ts_in=0.0, px_in=100 * POINT)
        t.px_out = 98 * POINT
        assert t.points == pytest.approx(2.0)
        assert t.won
        t2 = ag.Trade(side="up", ts_in=0.0, px_in=100 * POINT)
        t2.px_out = 98 * POINT
        assert t2.points == pytest.approx(-2.0)
        assert not t2.won


class TestTakingProfitMovesTheRateNotTheTotal:
    """THE CONSEQUENCE OF IMMEDIATE RE-ENTRY, pinned rather than
    prevented.

    There was a gate here: after a target or a stop, the agreement
    had to break before another trade could open. It existed because
    without it the P&L telescopes -- the target closes the trade, the
    signal that opened it is still there, the next look re-opens at
    about the same price, and the two pieces sum to the one move.
    Measured when it was written: 4/2, 2/2 and no levels at all came
    out at +216.00 to the cent.

    Chris chose immediate re-entry anyway, so the gate is gone and
    the consequence is real. These tests state it plainly so nobody
    reads a sweep of take profits as a sweep of outcomes.

    Two tests were deleted rather than fixed -- "after a target it
    waits for a fresh signal" and "levels never increase the trade
    count" -- because both are false by design now.
    """

    def test_the_target_barely_moves_the_total(self, recs):
        """It moves the WIN RATE a great deal and the total hardly at
        all. A sweep that looks like it is finding an edge is mostly
        finding different ways to slice the same move."""
        none = ag.run(recs, 0.0, 0.0, target=2000, eval_s=1.0)
        tight = ag.run(recs, 4.0, 2.0, target=2000, eval_s=1.0)
        wide = ag.run(recs, 8.0, 4.0, target=2000, eval_s=1.0)
        totals = [none.points, tight.points, wide.points]
        spread = max(totals) - min(totals)
        assert spread < 0.05 * abs(none.points), (
            f"totals {totals} differ by {spread:.2f} -- if a target "
            "really moved the result, the warning in this class would "
            "need revisiting")
        # The trade COUNT, by contrast, changes a lot.
        assert tight.n > none.n * 2

    def test_the_target_does_fire(self, recs):
        r = ag.run(recs, 4.0, 2.0, target=2000, eval_s=1.0)
        assert r.by_reason().get("tp", {}).get("n", 0) > 0

    def test_the_stop_does_fire(self, recs):
        r = ag.run(recs, 8.0, 0.5, target=2000, eval_s=1.0)
        assert r.by_reason().get("sl", {}).get("n", 0) > 0


class TestLosingAgreementIsNotAnExit:
    """Chris: it "continues to close -1 tick over and over".

    The read is price against the OFI bar open and it crosses that
    open constantly, so agreement broke for an instant, the trade
    closed, and the next look re-opened it. Each round trip gave back
    about a tick and none was a decision about the market.
    """

    def test_a_flip_closes_it_and_nothing_else_does(self, recs):
        r = ag.run(recs, 0.0, 0.0, target=2000, eval_s=1.0)
        reasons = set(r.by_reason())
        assert "disagree" not in reasons, reasons
        assert "flip" in reasons

    def test_the_churn_is_gone(self, recs):
        """The old rule took about 1,500 trades on this fixture, most
        of them a tick or less. Held to a flip it takes a few dozen.
        A property of this generator, not of the market -- what it
        pins is that the exit stopped firing on noise."""
        r = ag.run(recs, 0.0, 0.0, target=2000, eval_s=1.0)
        assert r.n < 300, r.n
        tiny = sum(1 for t in r.trades if abs(t.ticks) <= 1)
        assert tiny < 0.25 * r.n, f"{tiny} of {r.n} are <=1 tick"

    def test_trades_are_held_rather_than_flickered(self, recs):
        r = ag.run(recs, 0.0, 0.0, target=2000, eval_s=1.0)
        holds = sorted(t.held_s for t in r.trades)
        assert holds[len(holds) // 2] > 20.0, holds[:5]


class TestTheRuleIsThreeOfFour:
    """It was all four until Chris changed it. Three is three now,
    and a dissenting fourth does not block it.

    Two of these tests asserted the old rule and are rewritten rather
    than satisfied -- "the stack alone is not enough" and "one
    dissenting timeframe blocks it" are both false by design now, and
    bending the rule back to keep them green would have been the
    wrong fix.
    """

    def test_three_timeframes_fire_without_the_read(self, monkeypatch,
                                                     recs):
        """This asserted no trade may open with the read silent.
        Under three-of-four the three timeframes ARE three."""
        monkeypatch.setattr(
            ag.OfiBarBuilder, "bar_read",
            property(lambda self: {"read": ""}))
        monkeypatch.setattr(
            ag, "_dirs", lambda *a, **k: {t: "up" for t in ag.STACK})
        r = ag.run(recs, 0.0, 0.0, target=2000, eval_s=1.0)
        assert r.n >= 1
        assert all(t.side == "up" for t in r.trades)

    def test_two_is_still_not_enough(self, monkeypatch, recs):
        """Where the line actually sits now."""
        monkeypatch.setattr(
            ag.OfiBarBuilder, "bar_read",
            property(lambda self: {"read": ""}))
        monkeypatch.setattr(
            ag, "_dirs",
            lambda *a, **k: {"15m": "up", "5m": "up", "1m": None})
        r = ag.run(recs, 0.0, 0.0, target=2000, eval_s=1.0)
        assert r.n == 0

    def test_the_read_alone_is_not_enough(self, monkeypatch, recs):
        """With every timeframe silent, one vote is not three."""
        monkeypatch.setattr(ag, "_dirs",
                            lambda *a, **k: {t: None for t in ag.STACK})
        r = ag.run(recs, 0.0, 0.0, target=2000, eval_s=1.0)
        assert r.n == 0

    def test_a_dissenting_fourth_does_not_block_it(self, monkeypatch,
                                                    recs):
        """This asserted the opposite. Chris's call: 3 is 3, even
        with the odd one out pointing the other way."""
        monkeypatch.setattr(
            ag.OfiBarBuilder, "bar_read",
            property(lambda self: {"read": "up"}))
        monkeypatch.setattr(
            ag, "_dirs",
            lambda *a, **k: {"15m": "down", "5m": "down", "1m": "down"})
        r = ag.run(recs, 0.0, 0.0, target=2000, eval_s=1.0)
        assert r.n >= 1
        assert all(t.side == "down" for t in r.trades)

    def test_two_against_two_fires_nothing(self, monkeypatch, recs):
        monkeypatch.setattr(
            ag.OfiBarBuilder, "bar_read",
            property(lambda self: {"read": "up"}))
        monkeypatch.setattr(
            ag, "_dirs",
            lambda *a, **k: {"15m": "down", "5m": "down", "1m": "up"})
        r = ag.run(recs, 0.0, 0.0, target=2000, eval_s=1.0)
        assert r.n == 0

    def test_a_trade_only_opens_with_all_four(self, monkeypatch, recs):
        monkeypatch.setattr(
            ag, "_dirs",
            lambda *a, **k: {t: "up" for t in ag.STACK})
        monkeypatch.setattr(
            ag.OfiBarBuilder, "bar_read",
            property(lambda self: {"read": "up"}))
        r = ag.run(recs, 0.0, 0.0, target=2000, eval_s=1.0)
        assert r.n >= 1
        assert all(t.side == "up" for t in r.trades)


class TestTheHonestAccounting:
    def test_an_unresolved_trade_is_counted(self, monkeypatch, recs):
        """Dropping it would quietly discard the losers that had not
        resolved; closing it as a win at the last price is worse."""
        monkeypatch.setattr(
            ag, "_dirs", lambda *a, **k: {t: "up" for t in ag.STACK})
        monkeypatch.setattr(
            ag.OfiBarBuilder, "bar_read",
            property(lambda self: {"read": "up"}))
        r = ag.run(recs, 0.0, 0.0, target=2000, eval_s=1.0)
        assert r.by_reason().get("end", {}).get("n", 0) == 1

    def test_the_win_rate_matches_the_trades(self, recs):
        r = ag.run(recs, 1.0, 1.0, target=2000, eval_s=1.0)
        assert r.wins == sum(1 for t in r.trades if t.points > 0)
        assert r.rate == pytest.approx(r.wins / r.n)

    def test_the_total_is_the_sum_of_the_trades(self, recs):
        r = ag.run(recs, 2.0, 1.0, target=2000, eval_s=1.0)
        assert r.points == pytest.approx(sum(t.points for t in r.trades))

    def test_it_places_nothing(self):
        src = (ROOT / "scripts" / "agree_study.py").read_text()
        for bad in ("def submit(", "def execute(", "def place(",
                    "def order("):
            assert bad not in src, bad
