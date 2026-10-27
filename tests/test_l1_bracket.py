"""The bracket simulator, and the ways it could invent money.

In order of how badly each would mislead:

  resolving on the ENDPOINT instead of the path. A trade that prints
  -4 at 200ms and +6 at 700ms is a loss, and a simulator that only
  looks at 700ms calls it a win. This is the commonest way a backtest
  of this shape comes out profitable and isn't, and it is the first
  thing tested here

  filling at the mid. A long that enters at the mid and exits at the
  mid never pays the spread, which on ES is the entire edge being
  argued about

  resolving the TARGET first when one book update puts both levels
  through. At this resolution there is no way to know which came
  first, and assuming the good one is free money

  ignoring latency, so the entry happens at an instant that does not
  exist outside the file
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "l1_bracket", ROOT / "scripts" / "l1_bracket.py")
l1b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(l1b)                 # type: ignore[union-attr]

TICK = 0.25
MS = 0.001


def sim(latency_ms=0.0, comm=0.0, bomb_ms=800.0, target=6, stop=4):
    return l1b.Sim(TICK, target, stop, bomb_ms, latency_ms, comm)


def feed(s, path, bid0=5000.00, spread_t=1, t0=0.0, step_ms=10.0):
    """Walk the bid through `path` (offsets in ticks); ask = bid + spread."""
    ts = t0
    for d in path:
        bid = bid0 + d * TICK
        s.on_book(ts, bid, bid + spread_t * TICK)
        ts += step_ms * MS
    return ts


class TestItWalksThePathNotTheEndpoint:

    def test_a_dip_to_the_stop_before_the_target_is_a_LOSS(self):
        """THE TEST THIS FILE EXISTS FOR. Price goes to -4, then to
        +10. An endpoint simulator calls that a winner."""
        s = sim()
        s.arm(0.0, "permanent_ask")              # long
        # entry fills at the ask on the first book: bid 5000.00, ask 5000.25
        feed(s, [0, -1, -2, -4, -5, 0, +4, +8, +10, +10])
        o = s.by_state["permanent_ask"]
        assert o.n == 1
        assert o.stop == 1 and o.target == 0, \
            "it touched the stop first; the later rally is irrelevant"
        assert o.ticks[0] == pytest.approx(-4.0)

    def test_a_clean_run_to_target_is_a_win(self):
        s = sim()
        s.arm(0.0, "permanent_ask")
        feed(s, [0, 1, 2, 3, 4, 5, 6, 7])
        o = s.by_state["permanent_ask"]
        assert o.target == 1 and o.stop == 0
        assert o.ticks[0] == pytest.approx(6.0)

    def test_the_stop_wins_when_one_update_crosses_both(self):
        """A single book update can leave both levels through. There is
        no way to know which came first, so the pessimistic read is the
        only defensible one."""
        s = sim()
        s.arm(0.0, "permanent_ask")
        s.on_book(0.0, 5000.00, 5000.25)        # fill long at 5000.25
        # Next update is 40 ticks away -- target and stop both through.
        s.on_book(10 * MS, 4990.00, 4990.25)
        o = s.by_state["permanent_ask"]
        assert o.stop == 1 and o.target == 0


class TestFillsCrossTheSpread:

    def test_a_long_pays_the_ask_and_exits_on_the_bid(self):
        s = sim()
        s.arm(0.0, "permanent_ask")
        s.on_book(0.0, 5000.00, 5000.25)
        t = s.open[0]
        assert t.entry_px == pytest.approx(5000.25), "paid the offer"
        assert t.target == pytest.approx(5000.25 + 6 * TICK)
        assert t.stop == pytest.approx(5000.25 - 4 * TICK)

    def test_a_short_pays_the_bid_and_exits_on_the_ask(self):
        s = sim()
        s.arm(0.0, "permanent_bid")
        s.on_book(0.0, 5000.00, 5000.25)
        t = s.open[0]
        assert t.side == l1b.SHORT
        assert t.entry_px == pytest.approx(5000.00), "hit the bid"
        assert t.target == pytest.approx(5000.00 - 6 * TICK)

    def test_the_spread_makes_the_target_further_than_the_stop(self):
        """+6/-4 in PnL is 7-to-win and 3-to-lose in MOVEMENT, because
        entry crosses. A mid-price simulation hides this entirely."""
        s = sim()
        s.arm(0.0, "permanent_ask")
        s.on_book(0.0, 5000.00, 5000.25)
        t = s.open[0]
        bid_move_to_win = (t.target - 5000.00) / TICK
        bid_move_to_lose = (5000.00 - t.stop) / TICK
        assert bid_move_to_win == pytest.approx(7.0)
        assert bid_move_to_lose == pytest.approx(3.0)

    def test_a_long_is_immediately_down_the_spread(self):
        s = sim()
        s.arm(0.0, "permanent_ask")
        s.on_book(0.0, 5000.00, 5000.25)
        assert s.open[0].mae == pytest.approx(-1.0)


class TestTheTimeBomb:

    def test_a_trade_that_goes_nowhere_is_flattened(self):
        s = sim(bomb_ms=800.0)
        s.arm(0.0, "permanent_ask")
        feed(s, [0] * 100, step_ms=10.0)         # 1000ms of nothing
        o = s.by_state["permanent_ask"]
        assert o.timeout == 1 and o.target == 0 and o.stop == 0

    def test_a_choke_out_is_a_loss_not_a_scratch(self):
        """It exits by crossing the spread. Reporting it as flat is the
        difference between a losing system and a break-even one."""
        s = sim(bomb_ms=800.0)
        s.arm(0.0, "permanent_ask")
        feed(s, [0] * 100, step_ms=10.0)
        o = s.by_state["permanent_ask"]
        assert o.timeout_ticks[0] == pytest.approx(-1.0), "the spread"
        assert o.timeout_ticks[0] < 0

    def test_commission_comes_off_every_outcome(self):
        s = sim(comm=0.4)
        s.arm(0.0, "permanent_ask")
        feed(s, [0, 1, 2, 3, 4, 5, 6, 7])
        assert s.by_state["permanent_ask"].ticks[0] == pytest.approx(5.6)

    def test_the_bomb_does_not_fire_early(self):
        s = sim(bomb_ms=800.0)
        s.arm(0.0, "permanent_ask")
        feed(s, [0] * 50, step_ms=10.0)          # 500ms
        assert "permanent_ask" not in s.by_state
        assert len(s.open) == 1


class TestLatency:

    def test_entry_is_delayed_by_the_configured_amount(self):
        s = sim(latency_ms=50.0)
        s.arm(0.0, "permanent_ask")
        s.on_book(10 * MS, 5000.00, 5000.25)
        assert s.open == [], "too early"
        s.on_book(60 * MS, 5002.00, 5002.25)
        assert len(s.open) == 1
        assert s.open[0].entry_px == pytest.approx(5002.25), \
            "filled at the price AFTER the delay, not the signal price"

    def test_latency_changes_the_outcome(self):
        """If the move is over before you arrive, you buy the high.
        This is the whole reason the sweep exists."""
        path = [0, 4, 8, 8, 6, 4, 2, 0, -2, -4, -6, -8, -8]
        fast, slow = sim(latency_ms=0.0), sim(latency_ms=30.0)
        for s in (fast, slow):
            s.arm(0.0, "permanent_ask")
            feed(s, path, step_ms=10.0)
        f = fast.by_state["permanent_ask"]
        sl = slow.by_state["permanent_ask"]
        assert f.ticks[0] != sl.ticks[0], \
            "arriving 30ms later must not give the identical fill"


class TestItMirrors:

    def test_the_short_side_is_the_long_side_reflected(self):
        up = [0, 1, 2, 3, 4, 5, 6, 7]
        down = [-d for d in up]
        a, b = sim(), sim()
        a.arm(0.0, "permanent_ask")              # long
        feed(a, up)
        b.arm(0.0, "permanent_bid")              # short
        feed(b, down)
        assert a.by_state["permanent_ask"].target == 1
        assert b.by_state["permanent_bid"].target == 1
        assert (a.by_state["permanent_ask"].ticks[0]
                == pytest.approx(b.by_state["permanent_bid"].ticks[0]))

    def test_the_failed_states_trade_the_other_way(self):
        assert l1b.DIRECTION["failed_bid"] == l1b.LONG
        assert l1b.DIRECTION["failed_ask"] == l1b.SHORT
        assert l1b.DIRECTION["permanent_ask"] == l1b.LONG
        assert l1b.DIRECTION["permanent_bid"] == l1b.SHORT


class TestExcursions:

    def test_mfe_and_mae_are_marked_on_the_exit_side(self):
        s = sim()
        s.arm(0.0, "permanent_ask")
        feed(s, [0, 2, 3, 1, -1, 0] + [0] * 100, step_ms=10.0)
        o = s.by_state["permanent_ask"]
        # entry 5000.25; best bid reached 5000.75 = +2 ticks on the exit side
        assert o.mfe[0] == pytest.approx(2.0)
        assert o.mae[0] == pytest.approx(-2.0)


class TestItDecidesNothing:

    def test_no_order_code(self):
        src = (ROOT / "scripts" / "l1_bracket.py").read_text()
        for word in ("def submit(", "def execute(", "def place(",
                     "def order("):
            assert word not in src, word


class TestARandomWalkMustLose:
    """The baseline that makes every other number readable.

    Entries placed at random times in a driftless market must lose
    roughly the spread plus commission per trade, on both sides, every
    time. If this comes back positive the simulator is manufacturing
    money and nothing it says about real data means anything.

    It also sets the hurdle: the deletion signal is only worth
    something if it beats this, not if it beats zero.
    """

    def _walk(self, seed, n_events=1500, comm=0.4, lat=0.0):
        import random
        rng = random.Random(seed)
        s = sim(latency_ms=lat, comm=comm)
        bid, ts, nxt = 5000.00, 0.0, 0
        for i in range(150_000):
            bid += rng.choice((-TICK, 0.0, 0.0, 0.0, TICK))
            ts += 0.005
            s.on_book(ts, bid, bid + TICK)
            if i == nxt and n_events > 0:
                s.arm(ts, rng.choice(("permanent_ask", "permanent_bid")))
                n_events -= 1
                nxt = i + 90
        return s

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_random_entries_lose_about_the_spread_plus_commission(self, seed):
        s = self._walk(seed)
        ticks = [t for o in s.by_state.values() for t in o.ticks]
        assert len(ticks) > 500
        exp = sum(ticks) / len(ticks)
        assert exp < 0, f"a driftless market paid {exp:+.3f}t per trade"
        assert -2.5 < exp < -0.8, \
            f"expected roughly -(spread + commission), got {exp:+.3f}t"

    def test_both_sides_lose_about_equally(self):
        """A simulator that is kind to one side would make the mirror
        test useless on real data -- the asymmetry would be mine."""
        s = self._walk(7)
        a = s.by_state["permanent_ask"]
        b = s.by_state["permanent_bid"]
        ea = sum(a.ticks) / len(a.ticks)
        eb = sum(b.ticks) / len(b.ticks)
        assert abs(ea - eb) < 0.5, f"long {ea:+.3f}t vs short {eb:+.3f}t"

    def test_a_closer_stop_than_target_loses_more_often_than_it_wins(self):
        """With +6/-4 the stop is nearer, so on noise the stop fires
        more often. A simulator showing the reverse has its comparisons
        inverted somewhere."""
        s = self._walk(5)
        tgt = sum(o.target for o in s.by_state.values())
        stp = sum(o.stop for o in s.by_state.values())
        assert stp > tgt, f"target {tgt} vs stop {stp} on pure noise"

    def test_zero_commission_still_loses_the_spread(self):
        s = self._walk(11, comm=0.0)
        ticks = [t for o in s.by_state.values() for t in o.ticks]
        exp = sum(ticks) / len(ticks)
        assert exp < 0, "crossing twice is a cost even with no commission"
