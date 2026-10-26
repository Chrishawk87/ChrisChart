"""Cycle finding, and the ways it would report money that is not there.

In order of how badly each would mislead:

  finding an "arbitrage" between pools that agree. Two pools at the
  same price must yield nothing; a search that returns a cycle there
  is measuring its own rounding and will fill the panel with noise

  sizing past the book. The quote is only approximately right for a
  size the pool can absorb, so an optimiser that walks to the reserve
  is reporting a fill nobody would get

  reusing a pool inside one cycle, which trades against yourself and
  manufactures an edge out of the fee being counted twice

  reporting gross as though it were profit. It is not, and the module
  says so in several places that these tests check are still there
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from liqmap.dexarb import (Q96, V2, V3, Cycle, DexArb, Hop,  # noqa: E402
                           Pool, PoolState, quote_v2, quote_v3)

WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
DAI = "0x6B175474E89094C44Da98b954EedeAC495271d0F"

E18 = 10 ** 18
E6 = 10 ** 6


def v2(addr, t0, t1, label=""):
    return Pool(addr, V2, t0, t1, 3000, label or addr)


def arb(pools, **kw):
    return DexArb(pools, **kw)


class TestTheQuotes:

    def test_v2_matches_the_canonical_formula(self):
        r_in, r_out, amt = 5_000_000 * E6, 2_000 * E18, 10_000 * E6
        want = (amt * 997 * r_out) // (r_in * 1000 + amt * 997)
        assert quote_v2(amt, r_in, r_out, 3000) == want

    def test_v2_is_monotonic_and_bounded(self):
        r_in, r_out = 1_000_000 * E6, 400 * E18
        prev = 0
        for i in range(1, 40):
            out = quote_v2(10_000 * E6 * i, r_in, r_out, 3000)
            assert out >= prev
            assert out < r_out, "cannot take more than the pool holds"
            prev = out

    def test_v2_degenerate_inputs_yield_nothing(self):
        assert quote_v2(0, 10, 10, 3000) == 0
        assert quote_v2(10, 0, 10, 3000) == 0
        assert quote_v2(10, 10, 0, 3000) == 0

    def test_a_lower_fee_returns_more(self):
        r_in, r_out, amt = 1_000_000 * E6, 400 * E18, 50_000 * E6
        assert (quote_v2(amt, r_in, r_out, 100)
                > quote_v2(amt, r_in, r_out, 10_000))

    def test_v3_quotes_both_directions(self):
        sqrt_p = 1771845812700903892492222464
        liq = 12_000_000_000_000_000_000
        assert quote_v3(E18, sqrt_p, liq, True, 500) > 0
        assert quote_v3(E18, sqrt_p, liq, False, 500) > 0
        assert quote_v3(E18, sqrt_p, 0, True, 500) == 0


class TestItFindsNothingWhenThereIsNothing:
    """The first thing a cycle search must get right."""

    def _balanced(self):
        """Two pools quoting WETH/USDC at exactly the same price."""
        a = v2("0xAAA", USDC, WETH, "pool A")
        b = v2("0xBBB", USDC, WETH, "pool B")
        g = arb([a, b])
        for p in (a, b):
            g.set_state(p.address, PoolState(reserve0=4_000_000 * E6,
                                             reserve1=2_000 * E18))
        return g

    def test_two_pools_at_the_same_price_yield_no_cycle(self):
        g = self._balanced()
        assert g.find(WETH) == [], "identical pools are not an arbitrage"

    def test_and_the_fee_makes_it_strictly_negative(self):
        """Round-tripping through two 0.30% pools must lose 0.60%."""
        g = self._balanced()
        hops = g.enumerate(WETH)
        assert hops, "the cycles exist, they are just not profitable"
        for h in hops:
            out = g.quote_path(h, 10 * E18)
            assert out < 10 * E18

    def test_an_empty_pool_is_skipped_not_treated_as_free(self):
        a = v2("0xAAA", USDC, WETH)
        b = v2("0xBBB", USDC, WETH)
        g = arb([a, b])
        g.set_state(a.address, PoolState(reserve0=4_000_000 * E6,
                                         reserve1=2_000 * E18))
        g.set_state(b.address, PoolState())      # never fetched
        assert g.find(WETH) == []


class TestItFindsWhatIsThere:

    def _skewed(self, skew=1.05):
        """Pool B prices WETH `skew` higher than pool A."""
        a = v2("0xAAA", USDC, WETH, "cheap")
        b = v2("0xBBB", USDC, WETH, "rich")
        g = arb([a, b])
        g.set_state(a.address, PoolState(reserve0=4_000_000 * E6,
                                         reserve1=2_000 * E18))
        g.set_state(b.address, PoolState(reserve0=int(4_000_000 * E6 * skew),
                                         reserve1=2_000 * E18))
        return g

    def test_a_real_dislocation_is_found(self):
        g = self._skewed(1.05)
        found = g.find(WETH)
        assert found, "a 5% gap between two pools is an arbitrage"
        c = found[0]
        assert c.gross > 0 and c.amount_in > 0
        assert c.bps > 0
        assert len(c.hops) == 2

    def test_a_bigger_gap_is_worth_more(self):
        small = g1 = self._skewed(1.01).find(WETH)
        big = self._skewed(1.10).find(WETH)
        assert small and big
        assert big[0].gross > small[0].gross

    def test_the_cycle_closes_on_the_asset_it_started_from(self):
        g = self._skewed(1.05)
        for c in g.find(WETH):
            assert c.hops[0].token_in.lower() == WETH.lower()
            assert c.hops[-1].token_out.lower() == WETH.lower()

    def test_results_are_ordered_best_first(self):
        g = self._skewed(1.08)
        found = g.find(WETH)
        grosses = [c.gross for c in found]
        assert grosses == sorted(grosses, reverse=True)

    def test_min_bps_filters(self):
        g = self._skewed(1.001)
        loose = g.find(WETH)
        g.min_bps = 1000.0
        assert len(g.find(WETH)) < len(loose) or loose == []


class TestSizing:

    def test_the_size_respects_the_impact_cap(self):
        """An optimiser that walks to the reserve reports a fill
        nobody would get."""
        a = v2("0xAAA", USDC, WETH)
        b = v2("0xBBB", USDC, WETH)
        g = arb([a, b], max_impact_bps=30)
        g.set_state(a.address, PoolState(reserve0=4_000_000 * E6,
                                         reserve1=2_000 * E18))
        g.set_state(b.address, PoolState(reserve0=4_400_000 * E6,
                                         reserve1=2_000 * E18))
        found = g.find(WETH)
        assert found
        # 0.30% of the thinnest input-side reserve
        assert found[0].amount_in <= 2_000 * E18 * 30 // 10_000

    def test_a_wider_cap_allows_a_bigger_trade(self):
        def build(cap):
            a = v2("0xAAA", USDC, WETH)
            b = v2("0xBBB", USDC, WETH)
            g = arb([a, b], max_impact_bps=cap)
            g.set_state(a.address, PoolState(reserve0=4_000_000 * E6,
                                             reserve1=2_000 * E18))
            g.set_state(b.address, PoolState(reserve0=4_400_000 * E6,
                                             reserve1=2_000 * E18))
            return g.find(WETH)[0].amount_in
        assert build(300) > build(30)

    def test_sizing_a_dead_pool_returns_nothing(self):
        a = v2("0xAAA", USDC, WETH)
        g = arb([a])
        hop = Hop(a.address, WETH, USDC, V2, False)
        assert g.optimal_size((hop,)) == (0, 0)


class TestEnumeration:

    def test_a_pool_is_never_used_twice_in_one_cycle(self):
        """Trading against yourself manufactures an edge from the fee
        being counted twice."""
        a = v2("0xAAA", USDC, WETH)
        b = v2("0xBBB", USDC, WETH)
        c = v2("0xCCC", USDC, DAI)
        d = v2("0xDDD", DAI, WETH)
        g = arb([a, b, c, d], cycle_lengths=(2, 3))
        for h in g.enumerate(WETH):
            keys = [x.pool.lower() for x in h]
            assert len(keys) == len(set(keys))

    def test_three_leg_cycles_are_found(self):
        a = v2("0xAAA", USDC, WETH)
        c = v2("0xCCC", USDC, DAI)
        d = v2("0xDDD", DAI, WETH)
        g = arb([a, c, d], cycle_lengths=(3,))
        hops = g.enumerate(WETH)
        assert any(len(h) == 3 for h in hops)

    def test_a_token_with_no_pool_yields_nothing(self):
        a = v2("0xAAA", USDC, DAI)
        g = arb([a])
        assert g.enumerate(WETH) == []


class TestItDecidesNothing:

    def test_no_order_code(self):
        src = (ROOT / "liqmap" / "dexarb.py").read_text()
        for w in ("def submit(", "def execute(", "def place(",
                  "def order(", "def sign(", "private_key", "PRIVATE_KEY"):
            assert w not in src, w

    def test_it_says_gross_is_not_profit(self):
        """The one misreading that would cost money: treating the
        number on the panel as take-home."""
        src = (ROOT / "liqmap" / "dexarb.py").read_text().lower()
        assert "is not profit" in src
        assert "builder" in src, "the bid is the term that decides it"
        for w in ("guaranteed", "risk-free", "free money"):
            assert w not in src, w

    def test_it_admits_the_v3_quote_is_approximate(self):
        src = (ROOT / "liqmap" / "dexarb.py").read_text().lower()
        assert "optimistic" in src and "single-tick" in src

    def test_state_carries_the_block_it_came_from(self):
        """A stale read must never be mistakable for a live one."""
        st = PoolState(reserve0=1, reserve1=1, block=21_000_000)
        assert st.block == 21_000_000

    def test_degenerate_settings_are_refused(self):
        a = v2("0xAAA", USDC, WETH)
        with pytest.raises(ValueError):
            DexArb([])
        with pytest.raises(ValueError):
            DexArb([a], max_impact_bps=0)
        with pytest.raises(ValueError):
            DexArb([a], sizing_iterations=1)

    def test_it_serialises(self):
        import json
        g = DexArb([v2("0xAAA", USDC, WETH), v2("0xBBB", USDC, WETH)])
        g.set_state("0xAAA", PoolState(reserve0=4_000_000 * E6,
                                       reserve1=2_000 * E18))
        g.set_state("0xBBB", PoolState(reserve0=4_400_000 * E6,
                                       reserve1=2_000 * E18))
        found = g.find(WETH)
        assert found
        json.dumps(found[0].as_dict())


class TestTheRoute:
    """The panel's contract with the page, and what it must never leak."""

    def _client(self, monkeypatch):
        import tests.test_web as T                      # noqa: F401
        return T

    def test_it_reports_a_missing_rpc_rather_than_erroring(self, monkeypatch):
        monkeypatch.delenv("ETH_RPC_URL", raising=False)
        monkeypatch.delenv("RPC_URL", raising=False)
        from liqmap import dexfeed
        assert dexfeed.rpc_url() == ""

    def test_an_unknown_asset_lists_the_known_ones(self):
        from liqmap.dexarb import TOKENS
        assert set(TOKENS) >= {"WETH", "USDC", "USDT", "WBTC"}
        for name, (addr, dec) in TOKENS.items():
            assert addr.startswith("0x") and len(addr) == 42
            assert 0 < dec <= 18

    def test_the_pool_set_is_internally_consistent(self):
        from liqmap.dexarb import MAINNET_POOLS, TOKENS
        known = {a.lower() for a, _ in TOKENS.values()}
        seen = set()
        for p in MAINNET_POOLS:
            assert p.kind in ("v2", "v3")
            assert p.token0.lower() in known, p.label
            assert p.token1.lower() in known, p.label
            assert p.token0.lower() != p.token1.lower()
            assert 0 < p.fee_ppm <= 100_000
            assert p.key not in seen, f"duplicate pool {p.label}"
            seen.add(p.key)
            assert len(p.address) == 42 and p.address.startswith("0x")

    def test_the_pool_set_actually_forms_cycles(self):
        """A pool list with no closed loop makes the panel permanently
        empty, which looks like a quiet market."""
        from liqmap.dexarb import MAINNET_POOLS, TOKENS, DexArb
        g = DexArb(MAINNET_POOLS)
        hops = g.enumerate(TOKENS["WETH"][0])
        assert len(hops) >= 4
        assert any(len(h) == 3 for h in hops), "no triangles in the set"

    def test_the_panel_says_gross_is_not_profit(self):
        """The one misreading that costs money. If a later edit tidies
        this sentence off the page, this fails."""
        src = (ROOT / "liqmap" / "web.py").read_text()
        assert "dexArbRow" in src
        low = src.lower()
        assert "gross is not profit" in low
        assert "builder" in low

    def test_the_panel_has_its_own_timer(self):
        """Drawn at the tail of fpLoad it would go dark outside RTH,
        when the CME tape is silent but the pools are still trading."""
        src = (ROOT / "liqmap" / "web.py").read_text()
        assert "_dexTimer" in src and "setInterval(dexTick" in src

    def test_the_route_signs_nothing(self):
        src = (ROOT / "liqmap" / "web.py").read_text()
        i = src.index("def api_dexarb")
        j = src.index("@app.get(\"/api/footprint\"")
        body = src[i:j]
        for w in ("eth_sendRawTransaction", "sendBundle", "private",
                  "signTransaction", "Wallet("):
            assert w not in body, w

    def test_token0_sorts_below_token1_in_every_pool(self):
        """Uniswap orders a pair's tokens by ascending address. Get one
        backwards and `zero_for_one` is inverted for that pool, so every
        quote through it reads the wrong side of the book -- silently,
        with plausible numbers."""
        from liqmap.dexarb import MAINNET_POOLS
        for p in MAINNET_POOLS:
            assert int(p.token0, 16) < int(p.token1, 16), (
                f"{p.label}: token0 {p.token0} should sort below "
                f"token1 {p.token1}")


class TestImplausibleEdgesAreRefused:
    """A broken read produces an enormous, confident number. A panel
    that prints it teaches you to trust its largest row."""

    def _mixed(self, max_bps=1000.0):
        from liqmap.dexarb import MAINNET_POOLS
        g = DexArb(MAINNET_POOLS, max_bps=max_bps)
        for p in MAINNET_POOLS:
            if p.kind == "v2":
                g.set_state(p.address, PoolState(reserve0=40_000_000 * E6,
                                                 reserve1=20_000 * E18,
                                                 block=21_000_000))
            else:
                # A sqrtPriceX96 that does NOT mean what the code assumes
                # -- the exact shape of a decimals mistake.
                g.set_state(p.address,
                            PoolState(sqrt_price_x96=1771845812700903892492222464,
                                      liquidity=30_000_000_000_000_000_000,
                                      block=21_000_000))
        return g

    def test_a_miscalibrated_pool_does_not_reach_the_panel(self):
        g = self._mixed()
        found = g.find(WETH)
        assert g.rejected > 0, "the broken cycles should have been caught"
        for c in found:
            assert c.bps <= 1000.0

    def test_raising_the_ceiling_lets_them_through(self):
        """So the guard is demonstrably the thing filtering them."""
        loose = self._mixed(max_bps=1e18).find(WETH)
        assert loose and max(c.bps for c in loose) > 1000.0

    def test_a_real_dislocation_is_well_under_the_ceiling(self):
        a = v2("0xAAA", USDC, WETH)
        b = v2("0xBBB", USDC, WETH)
        g = DexArb([a, b])
        g.set_state("0xAAA", PoolState(reserve0=4_000_000 * E6,
                                       reserve1=2_000 * E18))
        g.set_state("0xBBB", PoolState(reserve0=4_040_000 * E6,
                                       reserve1=2_000 * E18))
        found = g.find(WETH)
        assert found and found[0].bps < 1000.0
        assert g.rejected == 0


class TestCyclesAreNotDuplicated:

    def test_a_two_leg_loop_is_listed_once(self):
        """It closes during the depth-3 walk too. Listed twice it
        reads as two opportunities and doubles the apparent count."""
        a = v2("0xAAA", USDC, WETH)
        b = v2("0xBBB", USDC, WETH)
        g = DexArb([a, b], cycle_lengths=(2, 3))
        hops = g.enumerate(WETH)
        sigs = [tuple((h.pool.lower(), h.zero_for_one) for h in c) for c in hops]
        assert len(sigs) == len(set(sigs)), "duplicate cycles enumerated"

    def test_and_the_results_are_unique_too(self):
        a = v2("0xAAA", USDC, WETH)
        b = v2("0xBBB", USDC, WETH)
        g = DexArb([a, b], cycle_lengths=(2, 3))
        g.set_state("0xAAA", PoolState(reserve0=4_000_000 * E6,
                                       reserve1=2_000 * E18))
        g.set_state("0xBBB", PoolState(reserve0=4_200_000 * E6,
                                       reserve1=2_000 * E18))
        found = g.find(WETH)
        sigs = [tuple(h.pool for h in c.hops) for c in found]
        assert len(sigs) == len(set(sigs))

    def test_both_directions_of_a_loop_are_still_distinct(self):
        """Deduping must not collapse A->B and B->A: only one of them
        is the profitable direction."""
        a = v2("0xAAA", USDC, WETH)
        b = v2("0xBBB", USDC, WETH)
        g = DexArb([a, b], cycle_lengths=(2,))
        assert len(g.enumerate(WETH)) == 2
