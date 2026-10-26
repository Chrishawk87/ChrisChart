"""Mispriced cycles between DEX pools, as a reading.

Three pools quoting the same pair at different prices is an arbitrage,
and the size of it is computable from their reserves. This module finds
those cycles, sizes them, and reports what they were worth.

IT DOES NOT TRADE THEM. There is no key, no contract, no transaction
and no submission here -- the same rule the rest of this package keeps,
for the same reason: a number on a screen you can argue with is worth
more than an autonomous process you have to trust. Taking one of these
would need a flash-loan contract and a funded wallet, which is a
separate decision from being able to see them.

WHAT THE NUMBERS MEAN, AND WHAT THEY DO NOT

`gross` is the cycle's output minus its input at the quoted reserves.
It is not profit. Anyone acting on it would pay, in order: the flash
loan premium, gas, and -- the term that actually decides it -- whatever
they had to bid a block builder to be included ahead of everyone else
who could see the same thing. On a contested path that bid approaches
the whole gross, which is why a long list of large opportunities here
is evidence that they are hard to take, not that they are free.

So read this as a measurement of how often the pools disagree and by
how much. If `gross` is usually under a few dollars, nothing downstream
could have worked. If it is routinely large, the question becomes
latency, and this panel cannot answer that one.

V3 QUOTING IS DELIBERATELY OPTIMISTIC. `quote_v3` holds liquidity
constant and is therefore wrong, in the favourable direction, the
moment a swap would cross a tick. It is a scanner heuristic chosen for
speed over a curated pool list. `max_impact_bps` caps the size so the
error stays small; widen that and this stops being approximately right.
"""

from __future__ import annotations

from dataclasses import dataclass, field

V2 = "v2"
V3 = "v3"

Q96 = 1 << 96
MAX_UINT = (1 << 256) - 1


@dataclass(frozen=True, slots=True)
class Pool:
    address: str
    kind: str
    token0: str
    token1: str
    fee_ppm: int           # 3000 = 0.30%
    label: str = ""

    @property
    def key(self) -> str:
        return self.address.lower()


@dataclass(slots=True)
class PoolState:
    reserve0: int = 0
    reserve1: int = 0
    sqrt_price_x96: int = 0
    liquidity: int = 0
    block: int = 0

    @property
    def live(self) -> bool:
        return (self.reserve0 > 0 and self.reserve1 > 0) or (
            self.sqrt_price_x96 > 0 and self.liquidity > 0)


@dataclass(frozen=True, slots=True)
class Hop:
    pool: str
    token_in: str
    token_out: str
    kind: str
    zero_for_one: bool     # token_in is the pool's token0


@dataclass(frozen=True, slots=True)
class Cycle:
    """A closed loop back to the asset it started from."""

    hops: tuple[Hop, ...]
    asset: str
    amount_in: int = 0
    gross_out: int = 0
    labels: tuple[str, ...] = ()

    @property
    def gross(self) -> int:
        return self.gross_out - self.amount_in

    @property
    def bps(self) -> float:
        """Edge as basis points of the amount risked."""
        return (self.gross / self.amount_in * 10_000.0) if self.amount_in else 0.0

    def as_dict(self, decimals: int = 18) -> dict:
        unit = 10 ** decimals
        return {"asset": self.asset,
                "hops": [{"pool": h.pool, "kind": h.kind,
                          "token_in": h.token_in, "token_out": h.token_out}
                         for h in self.hops],
                "labels": list(self.labels),
                "amount_in": self.amount_in,
                "gross_out": self.gross_out,
                "gross": self.gross,
                "amount_in_f": self.amount_in / unit,
                "gross_f": self.gross / unit,
                "bps": round(self.bps, 3),
                "legs": len(self.hops)}


def quote_v2(amount_in: int, reserve_in: int, reserve_out: int,
             fee_ppm: int) -> int:
    """Constant product with the fee folded in. Exact integer maths."""
    if amount_in <= 0 or reserve_in <= 0 or reserve_out <= 0:
        return 0
    after_fee = amount_in * (1_000_000 - fee_ppm)
    out = (after_fee * reserve_out) // (reserve_in * 1_000_000 + after_fee)
    return out if out < reserve_out else 0


def quote_v3(amount_in: int, sqrt_price_x96: int, liquidity: int,
             zero_for_one: bool, fee_ppm: int) -> int:
    """Single-tick approximation. See the module docstring."""
    if amount_in <= 0 or liquidity <= 0 or sqrt_price_x96 <= 0:
        return 0
    after_fee = (amount_in * (1_000_000 - fee_ppm)) // 1_000_000
    if after_fee <= 0:
        return 0
    if zero_for_one:
        denom = liquidity * Q96 + after_fee * sqrt_price_x96
        if denom <= 0:
            return 0
        nxt = (liquidity * Q96 * sqrt_price_x96) // denom
        if nxt <= 0 or nxt >= sqrt_price_x96:
            return 0
        return (liquidity * (sqrt_price_x96 - nxt)) // Q96
    nxt = sqrt_price_x96 + (after_fee * Q96) // liquidity
    if nxt <= sqrt_price_x96:
        return 0
    return (liquidity * Q96 * (nxt - sqrt_price_x96)) // (nxt * sqrt_price_x96)


class DexArb:
    """Holds the pool set and finds cycles in whatever state it is given.

    State is supplied rather than fetched, so the search is testable
    without a network and a stale read can never be mistaken for a live
    one -- `PoolState.block` says which block each reading came from.
    """

    def __init__(self, pools: list[Pool], cycle_lengths: tuple[int, ...] = (2, 3),
                 max_impact_bps: int = 30, sizing_iterations: int = 24,
                 min_bps: float = 0.0, max_bps: float = 1000.0) -> None:
        if not pools:
            raise ValueError("no pools")
        if max_impact_bps <= 0:
            raise ValueError("max_impact_bps must be positive")
        if sizing_iterations < 4:
            raise ValueError("sizing_iterations must be at least 4")
        self.pools = list(pools)
        self.by_key = {p.key: p for p in pools}
        self.cycle_lengths = tuple(cycle_lengths)
        self.max_impact_bps = int(max_impact_bps)
        self.sizing_iterations = int(sizing_iterations)
        self.min_bps = float(min_bps)
        # A SANITY CEILING, NOT A PREFERENCE. Real cross-venue
        # dislocations on deep pairs run from a fraction of a basis
        # point to tens of them; anything claiming a double-digit
        # PERCENT edge on WETH/USDC is a broken read -- a stale state, a
        # pool whose decimals were assumed wrong, a sqrtPriceX96 that
        # does not mean what the code thinks. Those produce enormous
        # confident numbers, and a panel that prints them teaches you to
        # trust the largest row on it.
        self.max_bps = float(max_bps)
        self.rejected = 0        # cycles dropped as implausible
        self.states: dict[str, PoolState] = {}

    # ------------------------------------------------------------------
    def set_state(self, pool: str, st: PoolState) -> None:
        self.states[pool.lower()] = st

    def state(self, pool: str) -> PoolState | None:
        return self.states.get(pool.lower())

    def quote_hop(self, h: Hop, amount_in: int) -> int:
        p = self.by_key.get(h.pool.lower())
        st = self.state(h.pool)
        if p is None or st is None:
            return 0
        if p.kind == V2:
            r_in, r_out = ((st.reserve0, st.reserve1) if h.zero_for_one
                           else (st.reserve1, st.reserve0))
            return quote_v2(amount_in, r_in, r_out, p.fee_ppm)
        return quote_v3(amount_in, st.sqrt_price_x96, st.liquidity,
                        h.zero_for_one, p.fee_ppm)

    def quote_path(self, hops: tuple[Hop, ...], amount_in: int) -> int:
        amt = amount_in
        for h in hops:
            amt = self.quote_hop(h, amt)
            if amt <= 0:
                return 0
        return amt

    # ------------------------------------------------------------------
    def _within_impact(self, hops: tuple[Hop, ...], amount_in: int) -> bool:
        """Is every leg of this path inside the impact cap?

        EACH LEG IS CHECKED IN ITS OWN TOKEN. An earlier version took
        the minimum of the per-pool caps, which compares a depth in
        WETH against a depth in USDC and keeps the smaller number --
        so a 2,000 WETH pool and a 4,000,000 USDC pool produced a cap
        of twelve thousand wei, and every real dislocation was sized
        down to nothing and reported as no opportunity. A search that
        silently finds nothing is worse than one that errors.
        """
        amt = amount_in
        for h in hops:
            p = self.by_key.get(h.pool.lower())
            st = self.state(h.pool)
            if p is None or st is None:
                return False
            if amt <= 0:
                # Quoted down to nothing. Trivially inside the cap --
                # this is the non-monotonicity the comment below warns
                # about, and treating it as a failure is what made the
                # binary search return zero for every dislocation.
                return True
            if p.kind == V2:
                depth = st.reserve0 if h.zero_for_one else st.reserve1
            else:
                if st.sqrt_price_x96 <= 0:
                    return False
                depth = ((st.liquidity * Q96) // st.sqrt_price_x96
                         if h.zero_for_one
                         else (st.liquidity * st.sqrt_price_x96) // Q96)
            if depth <= 0 or amt * 10_000 > depth * self.max_impact_bps:
                return False
            amt = self.quote_hop(h, amt)
        # IMPACT ONLY. Whether the path yields anything is a separate
        # question and must not be folded in here: a one-wei input
        # quotes to zero output, so a combined predicate is false at
        # the bottom, true in the middle and false at the top -- not
        # monotonic, and a binary search over it returns zero for every
        # real dislocation. Profitability is judged in `optimal_size`,
        # where a zero-output size simply scores no profit.
        return True

    def _ceiling(self, hops: tuple[Hop, ...]) -> int:
        """Largest input whose every leg stays inside the cap.

        Binary search rather than arithmetic, because the size of leg
        two depends on the output of leg one and there is no closed
        form across mixed V2/V3 quoting. `quote_path` is monotonic in
        the input, so the predicate is too.
        """
        first = self.state(hops[0].pool) if hops else None
        p0 = self.by_key.get(hops[0].pool.lower()) if hops else None
        if first is None or p0 is None:
            return 0
        if p0.kind == V2:
            depth0 = first.reserve0 if hops[0].zero_for_one else first.reserve1
        else:
            if first.sqrt_price_x96 <= 0:
                return 0
            depth0 = ((first.liquidity * Q96) // first.sqrt_price_x96
                      if hops[0].zero_for_one
                      else (first.liquidity * first.sqrt_price_x96) // Q96)
        hi = depth0 * self.max_impact_bps // 10_000
        if hi <= 0 or not self._within_impact(hops, 1):
            return 0   # even a wei is over the cap: the book is empty
        if self._within_impact(hops, hi):
            return hi
        lo = 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._within_impact(hops, mid):
                lo = mid
            else:
                hi = mid - 1
        return lo

    def optimal_size(self, hops: tuple[Hop, ...]) -> tuple[int, int]:
        """Ternary search. Profit over input is unimodal for a cycle --
        it rises while the edge outweighs impact and falls after -- so
        this converges without a derivative and works for V2 and V3
        quoting alike."""
        hi = self._ceiling(hops)
        if hi <= 0:
            return 0, 0
        lo = max(1, hi // 10_000)

        def profit(x: int) -> int:
            return self.quote_path(hops, x) - x if x > 0 else 0

        for _ in range(self.sizing_iterations):
            if hi <= lo:
                break
            third = (hi - lo) // 3
            if third <= 0:
                break
            m1, m2 = lo + third, hi - third
            if profit(m1) < profit(m2):
                lo = m1 + 1
            else:
                hi = m2 - 1
        out = self.quote_path(hops, lo)
        return (lo, out) if out > lo else (0, 0)

    # ------------------------------------------------------------------
    def _hop(self, p: Pool, token_in: str) -> Hop | None:
        t = token_in.lower()
        if t == p.token0.lower():
            return Hop(p.address, p.token0, p.token1, p.kind, True)
        if t == p.token1.lower():
            return Hop(p.address, p.token1, p.token0, p.kind, False)
        return None

    def enumerate(self, start: str) -> list[tuple[Hop, ...]]:
        """Every distinct directed cycle back to `start`.

        DEDUPLICATED. A two-leg loop closes during the depth-3 walk as
        well as the depth-2 one, so without this the panel shows the
        same cycle twice with identical numbers -- which reads as two
        separate opportunities and doubles the apparent count.
        """
        found: list[tuple[Hop, ...]] = []
        seen: set[tuple] = set()
        s = start.lower()

        def walk(token: str, used: set[str], acc: list[Hop], depth: int) -> None:
            if acc and token.lower() == s:
                sig = tuple((h.pool.lower(), h.zero_for_one) for h in acc)
                if sig not in seen:
                    seen.add(sig)
                    found.append(tuple(acc))
                return
            if depth == 0:
                return
            for p in self.pools:
                if p.key in used:
                    continue
                h = self._hop(p, token)
                if h is None:
                    continue
                used.add(p.key)
                acc.append(h)
                walk(h.token_out, used, acc, depth - 1)
                acc.pop()
                used.discard(p.key)

        for n in self.cycle_lengths:
            walk(start, set(), [], n)
        return found

    def find(self, start: str, limit: int = 10) -> list[Cycle]:
        """Profitable cycles from `start`, best first, gross only."""
        out: list[Cycle] = []
        self.rejected = 0
        for hops in self.enumerate(start):
            amt, gross_out = self.optimal_size(hops)
            if amt <= 0:
                continue
            c = Cycle(hops=hops, asset=start, amount_in=amt,
                      gross_out=gross_out,
                      labels=tuple(
                          (self.by_key[h.pool.lower()].label
                           or h.pool[:10]) for h in hops))
            if c.gross <= 0 or c.bps < self.min_bps:
                continue
            if c.bps > self.max_bps:
                self.rejected += 1
                continue
            out.append(c)
        out.sort(key=lambda c: c.gross, reverse=True)
        return out[:limit]


# ---------------------------------------------------------------------------
# A curated mainnet pool set.
#
# SHORT ON PURPOSE. The cycle search is combinatorial in the pool count
# and the panel has one block to finish in, so a long list costs the
# refresh. These are the deepest WETH/USDC/USDT/WBTC venues, which is
# where the quoting approximations hold best.
# ---------------------------------------------------------------------------

TOKENS = {
    "WETH": ("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", 18),
    "USDC": ("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", 6),
    "USDT": ("0xdAC17F958D2ee523a2206206994597C13D831ec7", 6),
    "WBTC": ("0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599", 8),
}

_W = TOKENS["WETH"][0]
_U = TOKENS["USDC"][0]
_T = TOKENS["USDT"][0]
_B = TOKENS["WBTC"][0]

MAINNET_POOLS: list[Pool] = [
    Pool("0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc", V2, _U, _W, 3000,
         "UniV2 USDC/WETH"),
    Pool("0x0d4a11d5EEaaC28EC3F61d100daF4d40471f1852", V2, _W, _T, 3000,
         "UniV2 WETH/USDT"),
    Pool("0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640", V3, _U, _W, 500,
         "UniV3 USDC/WETH .05"),
    Pool("0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8", V3, _U, _W, 3000,
         "UniV3 USDC/WETH .30"),
    Pool("0x4e68Ccd3E89f51C3074ca5072bbAC773960dFa36", V3, _W, _T, 3000,
         "UniV3 WETH/USDT .30"),
    Pool("0xCBCdF9626bC03E24f779434178A73a0B4bad62eD", V3, _B, _W, 3000,
         "UniV3 WBTC/WETH .30"),
    # THE TWO CROSS PAIRS. Without them every pool touches WETH, the
    # graph is a star, and only two-leg cycles exist -- the triangles
    # where a third venue's disagreement shows up are unreachable.
    Pool("0x3416cF6C708Da44DB2624D63ea0AAef7113527C6", V3, _U, _T, 100,
         "UniV3 USDC/USDT .01"),
    Pool("0x99ac8cA7087fA4A2A1FB6357269965A2014ABc35", V3, _B, _U, 3000,
         "UniV3 WBTC/USDC .30"),
]
