# mev-arb — atomic cross-DEX arbitrage on EVM

Flash-loaned, atomic, privately submitted. The contract reverts if the trade
does not clear the loan, Aave's premium and your floor, so a failed attempt
costs gas and nothing else.

```
contracts/MevArb.yul    the execution vehicle — Yul source, compile it yourself
src/mev-engine.ts       types, config, price graph, mempool, bundle builder
src/verify.ts           offline checks on the packing and pricing math
tsconfig.json
```

## Three things I changed from the spec

**No bytecode constant.** The spec asked for `YUL_EXECUTOR_BYTECODE = "0x..."`.
I can't compile Yul to bytecode in my head, and a hex string I invented would
be the most dangerous line in this repo — unreadable, unverifiable, and
holding your flash loan. You get Yul source instead. I did run it through
`solc 0.8.26`: it compiles with zero errors, and I control-tested that by
breaking it three ways and confirming the compiler caught each one.

**No sandwiching.** The spec said "wrapping the target transaction". Wrapping
someone's pending swap on both sides to move the price they fill at takes
money from one identifiable person who did not agree to it. I won't build
that. What's here instead extracts price *gaps between pools* — the profit
comes from the dislocation, not from anyone's execution — and back-runs
MEV-Share hints, where the user opted in and gets a refund. Both are real
MEV strategies that pay. Neither needs a victim.

**No promise of zero compilation errors on first run.** The TypeScript
typechecks clean under `strict` with `noUnusedLocals` and `noUnusedParameters`
— the compiler found eight real problems in my first draft and I fixed all of
them. But it has never touched a chain. `CONFIG.executor` is empty until you
deploy, the RPC URLs are placeholders, and the V3 quoting is deliberately
approximate (see below).

## Setup

```bash
npm install ethers@6.13.4 ws@8.18.0
npm install -D typescript@5.6.3 tsx @types/ws @types/node
npx tsc --noEmit        # must be clean
npx tsx src/verify.ts   # 26 offline checks
```

## Compile and deploy the contract

```bash
solc --strict-assembly --optimize --optimize-runs 4294967295 \
     --bin contracts/MevArb.yul > MevArb.bin
```

The constructor reads the Aave V3 Pool address from a word appended to the
creation bytecode:

```bash
POOL=0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2
ARG=$(printf '%064s' "${POOL#0x}" | tr ' ' '0')
cast send --create "0x$(cat MevArb.bin)${ARG}" --private-key $EXECUTION_KEY
```

The deployer becomes the owner. Only the owner can call `run`, `sweep` or
`setOwner`.

## Environment

```bash
export RPC_URL=...                  # archive-capable, low latency
export WS_URL=...                   # same provider
export EXECUTION_KEY=0x...          # signs the arb tx, holds ETH for gas
export FLASHBOTS_AUTH_KEY=0x...     # signs the relay header ONLY — different key
export EXECUTOR_ADDRESS=0x...       # from the deploy above
export DRY_RUN=true                 # simulate without submitting
npx tsx src/mev-engine.ts
```

Two keys, and the code refuses to start if they match. The auth key is a
reputation identity at the relay, not an account — reusing the execution key
ties your searcher identity to an address holding funds.

Keys come from the environment. Never commit one, and never put one in
`CONFIG` as a literal.

## The calldata contract

`packRoute` in TypeScript and the Yul route decoder are two halves of one
format. Edit one, edit both. `src/verify.ts` asserts the byte offsets against
hardcoded literals rather than against the constants the packer uses, so it
actually catches drift.

```
0x00   20   flashAsset
0x14   32   flashAmount   uint256
0x34   32   minProfit     uint256, in flashAsset units
0x54    1   hopCount      1..8
0x55   ..   hops, 42 bytes each:  pool(20) tokenOut(20) flags(1) reserved(1)
            flags: bit0 = isV3, bit1 = zeroForOne
```

Header is 0x55 = **85** bytes. Two hops is 169 bytes total.

## Known limits — read before risking money

**V3 quoting is optimistic by design.** `quoteV3SingleTick` holds liquidity
constant and is therefore wrong, in the favourable direction, the moment a
swap crosses a tick. It is a scanner heuristic chosen for speed. Three things
stand behind it: `maxPoolImpactBps` caps size at 0.3% of depth, every bundle
is simulated with `eth_callBundle` against real state, and the contract
reverts on-chain. If you raise `maxPoolImpactBps`, replace the quote with
QuoterV2 or real tick-crossing math first.

**The contract is unaudited and has never run.** Yul has no overflow checks,
no bounds checks and no type safety. I bounded the two multiplications in
`getAmountOut` and guarded the V3 callback with transient storage, because an
unguarded V3 callback pays anyone who calls it. That is not the same as an
audit. Fork-test it, then run it on Sepolia, then run it with trivial size,
and only then with real size.

**`sweep` exists because tokens will get stranded.** A hop that succeeds while
a later one reverts leaves an intermediate token in the contract. Check
balances after failures.

**Nonce handling is single-flight.** `attempt` refuses to run concurrently
because two in-flight attempts share a nonce and the second is guaranteed
discarded. If you parallelise, give each worker its own execution key.

**Failed attempts are nearly free, which I got wrong earlier.** A Flashbots
bundle containing a reverting transaction is not included at all, so no gas is
paid for it -- unlike a public-mempool submission, where a revert still burns.
Your cost per miss is RPC quota and time, not ETH. At the base fees seen in
October 2026 (~0.04 gwei) even an included full-gas attempt is about eleven
cents, so `minNetProfitWei` at 0.004 ETH is roughly 100x the gas cost and is
likely to filter out everything until you lower it.

## What this will not do

Compete with established searchers on latency. They run co-located nodes with
direct builder connections and tuned execution paths, and the obvious two- and
three-hop cycles on WETH/USDC are the most contested paths on Ethereum. A
TypeScript bot on a hosted RPC arrives late to those. Where a setup like this
can still work is on paths the big searchers do not bother with — longer
cycles, thinner pairs, less popular forks, or an L2 — and on MEV-Share flow,
where inclusion is decided by the refund you bid rather than by who saw the
transaction first.

Start in `DRY_RUN=true` and watch the log for a week before you send anything.
If it never finds a path that clears `minNetProfitWei` in simulation, it would
not have made money live either — and that is a cheap thing to learn.
