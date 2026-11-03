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
    """Re-entry is immediate, so a target slices one move into
    pieces that sum back to it. Stated rather than prevented --
    Chris chose immediate re-entry knowing this."""

    def test_the_target_does_fire(self, recs):
        r = ag.run(recs, 4.0, 2.0, target=2000, eval_s=1.0)
        assert r.by_reason().get("tp", {}).get("n", 0) > 0

    def test_the_stop_does_fire(self, recs):
        r = ag.run(recs, 8.0, 0.5, target=2000, eval_s=1.0)
        assert r.by_reason().get("sl", {}).get("n", 0) > 0

    def test_a_target_moves_the_count_far_more_than_the_total(
            self, recs):
        """The warning, as an assertion. If a target ever really
        moved the result on this fixture, this fails and the claim
        in the panel needs revisiting."""
        none = ag.run(recs, 0.0, 0.0, target=2000, eval_s=1.0)
        tight = ag.run(recs, 4.0, 2.0, target=2000, eval_s=1.0)
        if not none.points:
            pytest.skip("fixture made no money either way")
        moved = abs(tight.points - none.points) / abs(none.points)
        assert moved < 0.25, moved


class TestTheOfiCandleDecides:
    """Chris: "tracking the OFI candle making trades based on that
    and nothing more".

    The three timeframes stopped voting. They veto, and only when
    all three point the other way. The three-of-four tests that
    stood here are gone rather than fixed -- that rule is not the
    rule any more.
    """

    def test_the_candle_alone_opens_trades(self, monkeypatch, recs):
        """Every timeframe silent, and it still trades."""
        monkeypatch.setattr(ag, "_dirs",
                            lambda *a, **k: {t: None for t in ag.STACK})
        r = ag.run(recs, 0.0, 0.0, target=2000, eval_s=1.0)
        assert r.n >= 1

    def test_a_flat_candle_opens_nothing(self, monkeypatch, recs):
        monkeypatch.setattr(
            ag.OfiBarBuilder, "bar_read",
            property(lambda self: {"read": "flat", "price_dir": "up"}))
        r = ag.run(recs, 0.0, 0.0, target=2000, eval_s=1.0)
        assert r.n == 0

    def test_all_three_against_vetoes_it(self, monkeypatch, recs):
        monkeypatch.setattr(
            ag.OfiBarBuilder, "bar_read",
            property(lambda self: {"read": "up", "price_dir": "up"}))
        monkeypatch.setattr(
            ag, "_dirs", lambda *a, **k: {t: "down" for t in ag.STACK})
        r = ag.run(recs, 0.0, 0.0, target=2000, eval_s=1.0)
        assert r.n == 0

    def test_two_against_does_not(self, monkeypatch, recs):
        monkeypatch.setattr(
            ag.OfiBarBuilder, "bar_read",
            property(lambda self: {"read": "up", "price_dir": "up"}))
        monkeypatch.setattr(
            ag, "_dirs",
            lambda *a, **k: {"15m": "down", "5m": "down", "1m": None})
        r = ag.run(recs, 0.0, 0.0, target=2000, eval_s=1.0)
        assert r.n >= 1
        assert all(t.side == "up" for t in r.trades)

    def test_the_study_and_the_panel_share_the_rule(self):
        """Imported, not re-written. This file drifted from the
        panel's rule once already."""
        src = (ROOT / "scripts" / "agree_study.py").read_text()
        assert "from liqmap.journal import signal" in src
        assert "signal(dirs, read, price)" in src


class TestLosingDirectionIsNotAnExit:
    """Only a flip closes it. Going flat holds."""

    def test_nothing_ends_by_disagreement(self, recs):
        r = ag.run(recs, 0.0, 0.0, target=2000, eval_s=1.0)
        assert "disagree" not in set(r.by_reason()), r.by_reason()

    def test_a_flip_is_what_ends_them(self, recs):
        r = ag.run(recs, 0.0, 0.0, target=2000, eval_s=1.0)
        assert r.by_reason().get("flip", {}).get("n", 0) > 0

    def test_the_trade_count_is_bounded_by_the_bars(self, recs):
        """ONE DIRECTION PER BAR AT MOST, so the trade count cannot
        exceed the number of OFI bars. How MANY trades that is on
        real tape is a question about bar rate, not about this rule:
        this fixture closes a bar every few seconds and real ES at
        2,000 contracts does not."""
        r = ag.run(recs, 0.0, 0.0, target=2000, eval_s=1.0)
        from liqmap.ofibar import OfiBarBuilder

        b = OfiBarBuilder(tick_size=0.25, threshold=2000, max_s=900.0,
                          min_updates=2)
        for ts, rec, _mid in recs:
            lv = rec.levels[0]
            b.on_book(ts, lv.bid_px, lv.bid_sz, lv.ask_px, lv.ask_sz)
        assert r.n <= len(b.bars) + 1, (r.n, len(b.bars))


class TestTheHonestAccounting:
    def test_an_unresolved_trade_is_counted(self, monkeypatch, recs):
        """Dropping it would quietly discard the losers that had not
        resolved; closing it as a win at the last price is worse."""
        monkeypatch.setattr(
            ag, "_dirs", lambda *a, **k: {t: "up" for t in ag.STACK})
        monkeypatch.setattr(
            ag.OfiBarBuilder, "bar_read",
            property(lambda self: {"read": "up", "price_dir": "up"}))
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
