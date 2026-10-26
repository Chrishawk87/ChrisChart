/*
 * MevArb.yul — atomic cross-DEX arbitrage executor
 *
 * Compile:
 *   solc --strict-assembly --optimize --optimize-runs 4294967295 \
 *        --bin contracts/MevArb.yul
 *
 * Deploy: append the Aave V3 Pool address, left-padded to 32 bytes, to the
 * creation bytecode. The constructor reads it from there (see below).
 *
 * ===========================================================================
 * WHAT IT DOES
 * ===========================================================================
 * Owner calls run(bytes route). The contract takes an Aave V3 simple flash
 * loan of route.flashAsset for route.flashAmount, walks N swap hops across
 * Uniswap V2- and V3-style pools, and must finish holding at least
 *     flashAmount + premium + minProfit
 * of flashAsset. If it does not, it executes revert(0, 0) and the whole
 * transaction — flash loan included — is undone. The only cost of a losing
 * attempt is gas.
 *
 * ===========================================================================
 * MEMORY DISCIPLINE  (this is a pure Yul object — read this before editing)
 * ===========================================================================
 * There is NO Solidity free-memory pointer here. Nothing maintains the word
 * at 0x40, and nothing may assume it is meaningful. Memory is a flat, hand-
 * partitioned scratchpad, and every region below has exactly one owner:
 *
 *   0x0000 - 0x007F   CALL SCRATCH. Rebuilt immediately before every
 *                     external call: selector at 0x1C (so the 4 bytes sit
 *                     flush against the args that follow at 0x20), then up
 *                     to three argument words. Never read across a call.
 *
 *   0x0080 - 0x00BF   RETURNDATA LANDING, small. One or two words. Used by
 *                     balanceOf, transfer, approve.
 *
 *   0x00C0 - 0x011F   RETURNDATA LANDING, wide. Three words, for
 *                     getReserves() (reserve0, reserve1, blockTimestampLast)
 *                     and for the two-word int256 pair a V3 swap returns.
 *
 *   0x0120 - 0x01FF   V3 CALLBACK DATA. The `bytes data` argument handed to
 *                     pool.swap() and handed back to us in
 *                     uniswapV3SwapCallback. One word of payload (the token
 *                     we owe), ABI-encoded as offset+length+value.
 *
 *   0x0200 +          ROUTE COPY. The caller's packed route, copied out of
 *                     calldata once at the top of executeOperation so the
 *                     hop loop can index it with mload instead of repeated
 *                     calldatacopy. Sized by the route; 8 hops is 421 bytes,
 *                     so this never reaches 0x0400.
 *
 * THE OVERWRITE HAZARD, stated plainly: the call scratch at 0x00 is reused
 * by every single external call. A value that must survive a call lives in a
 * local (a stack slot) or above 0x0200 — never in scratch. The hop loop
 * keeps amountIn and tokenIn on the stack for exactly this reason. If you
 * add a call inside the loop, re-stage its arguments; do not assume what you
 * wrote at 0x20 three lines ago is still there.
 *
 * The scratch is also why the selector is stored at 0x1C rather than 0x00:
 * mstore writes a full 32-byte word, so storing the selector as a plain
 * uint32 at 0x1C puts its 4 significant bytes at 0x1C..0x1F, directly
 * adjacent to the first argument word at 0x20. The call then starts at
 * offset 0x1C with length 4 + 32*nargs. Storing it at 0x00 would require
 * shifting it left 224 bits and leave a 28-byte hole.
 *
 * ===========================================================================
 * PACKED ROUTE LAYOUT  (the `bytes` payload — not ABI-encoded, byte-packed)
 * ===========================================================================
 *   0x00   20 bytes   flashAsset    token borrowed and repaid
 *   0x14   32 bytes   flashAmount   uint256
 *   0x34   32 bytes   minProfit     uint256, denominated in flashAsset
 *   0x54    1 byte    hopCount      uint8, 1..8
 *   0x55   ...        hops, 42 bytes each:
 *
 *        +0x00   20 bytes   pool       pair / pool address
 *        +0x14   20 bytes   tokenOut   token this hop yields
 *        +0x28    1 byte    flags      bit0 = isV3
 *                                      bit1 = zeroForOne (tokenIn is token0)
 *        +0x29    1 byte    reserved   must be zero
 *
 * Total = 0x55 + 42 * hopCount. The last hop's tokenOut MUST equal
 * flashAsset; the contract checks this rather than trusting the caller.
 *
 * ===========================================================================
 * TRUST BOUNDARIES
 * ===========================================================================
 *  - run() is owner-only.
 *  - executeOperation accepts calls only from the stored Aave pool, and only
 *    when initiator == address(this). Without the second check anyone could
 *    flash-loan into this contract and drive the hop loop with their own
 *    route.
 *  - uniswapV3SwapCallback accepts calls only from the pool address this
 *    contract set in transient storage immediately before calling swap().
 *    An unguarded V3 callback is a drain: anyone can call it and be paid.
 *
 * Transient storage (EIP-1153) is used for the callback guard because it is
 * written and read inside a single transaction and must not persist. Slots:
 *    t[0]  expected V3 pool for the in-flight swap  (0 when none)
 *    t[1]  token this contract owes that pool
 */

object "MevArb" {

    // -----------------------------------------------------------------
    // CONSTRUCTOR
    // -----------------------------------------------------------------
    // The Aave V3 Pool address is appended to the creation bytecode as a
    // single left-padded word. It is read from code, not from calldata,
    // because a Yul object's constructor has no calldata of its own.
    code {
        let runtimeOffset := dataoffset("runtime")
        let runtimeSize := datasize("runtime")

        // The appended argument begins immediately after the runtime code.
        // Copy it to scratch and persist it BEFORE the runtime overwrites
        // that same region.
        codecopy(0x00, add(runtimeOffset, runtimeSize), 0x20)
        let pool := shr(96, shl(96, mload(0x00)))
        if iszero(pool) { revert(0x00, 0x00) }

        sstore(0x00, caller())     // owner
        sstore(0x01, pool)         // Aave V3 Pool

        codecopy(0x00, runtimeOffset, runtimeSize)
        return(0x00, runtimeSize)
    }

    object "runtime" {
        code {

            // =========================================================
            // DISPATCH
            // =========================================================
            // calldataload(0) puts the selector in the HIGH 4 bytes of the
            // word; shifting right 224 bits leaves it as a bare uint32.
            if lt(calldatasize(), 4) { revert(0x00, 0x00) }
            let sig := shr(224, calldataload(0x00))

            switch sig

            // ---- run(bytes) ------------------------------------------
            case 0x3a276523 {
                requireOwner()
                // ABI: [0x04] offset to bytes, [offset] length, [.. ] data
                let off := add(0x04, calldataload(0x04))
                let len := calldataload(off)
                let src := add(off, 0x20)
                if lt(len, 0x7f) { revert(0x00, 0x00) }   // 0x55 + one hop

                // flashAsset and flashAmount are read straight out of the
                // route so the loan matches what the hops expect.
                let asset := shr(96, calldataload(src))
                let amount := calldataload(add(src, 0x14))

                // flashLoanSimple(receiver, asset, amount, params, referral)
                // params is a dynamic bytes argument, so its head holds an
                // offset (0xa0: five head words) and its tail holds
                // length + content.
                mstore(0x1c, 0x42b0b77c)
                mstore(0x20, address())
                mstore(0x40, asset)
                mstore(0x60, amount)
                mstore(0x80, 0xa0)
                mstore(0xa0, 0x00)                  // referralCode
                mstore(0xc0, len)
                calldatacopy(0xe0, src, len)

                let tail := add(0xe0, roundUp32(len))
                let ok := call(gas(), sload(0x01), 0,
                               0x1c, sub(tail, 0x1c), 0x00, 0x00)
                if iszero(ok) { bubbleRevert() }
                stop()
            }

            // ---- executeOperation(...) — Aave V3 flash loan callback --
            case 0x1b11d0ff {
                if iszero(eq(caller(), sload(0x01))) { revert(0x00, 0x00) }

                let asset     := shr(96, shl(96, calldataload(0x04)))
                let amount    := calldataload(0x24)
                let premium   := calldataload(0x44)
                let initiator := shr(96, shl(96, calldataload(0x64)))

                // Without this check, any third party could flash-loan into
                // this contract and supply their own route.
                if iszero(eq(initiator, address())) { revert(0x00, 0x00) }

                let pOff := add(0x04, calldataload(0x84))
                let pLen := calldataload(pOff)
                calldatacopy(0x0200, add(pOff, 0x20), pLen)

                runRoute(0x0200, pLen, asset, amount, premium)

                // Aave reclaims principal + premium with transferFrom.
                let owed := add(amount, premium)
                mstore(0x1c, 0x095ea7b3)            // approve(address,uint256)
                mstore(0x20, sload(0x01))
                mstore(0x40, owed)
                if iszero(call(gas(), asset, 0, 0x1c, 0x44, 0x80, 0x20)) {
                    bubbleRevert()
                }

                mstore(0x00, 0x01)                  // return true
                return(0x00, 0x20)
            }

            // ---- uniswapV3SwapCallback(int256,int256,bytes) -----------
            case 0xfa461e33 {
                // An unguarded V3 callback pays anyone who asks. t[0] is set
                // only for the duration of our own swap() call.
                let expected := tload(0x00)
                if iszero(expected) { revert(0x00, 0x00) }
                if iszero(eq(caller(), expected)) { revert(0x00, 0x00) }

                let d0 := calldataload(0x04)
                let d1 := calldataload(0x24)

                // The positive delta is what the pool is owed. Exactly one
                // of the two is positive on an exact-input swap.
                let owed := 0
                if sgt(d0, 0) { owed := d0 }
                if sgt(d1, 0) { owed := d1 }
                if iszero(owed) { revert(0x00, 0x00) }

                mstore(0x1c, 0xa9059cbb)            // transfer(address,uint256)
                mstore(0x20, caller())
                mstore(0x40, owed)
                if iszero(call(gas(), tload(0x01), 0, 0x1c, 0x44, 0x80, 0x20)) {
                    bubbleRevert()
                }
                requireTruthyReturn()
                stop()
            }

            // ---- sweep(address) --------------------------------------
            case 0x01681a62 {
                requireOwner()
                let token := shr(96, shl(96, calldataload(0x04)))
                switch token
                case 0 {
                    let ok := call(gas(), caller(), selfbalance(),
                                   0x00, 0x00, 0x00, 0x00)
                    if iszero(ok) { revert(0x00, 0x00) }
                }
                default {
                    mstore(0x1c, 0x70a08231)        // balanceOf(address)
                    mstore(0x20, address())
                    if iszero(staticcall(gas(), token, 0x1c, 0x24, 0x80, 0x20)) {
                        bubbleRevert()
                    }
                    let bal := mload(0x80)
                    mstore(0x1c, 0xa9059cbb)
                    mstore(0x20, caller())
                    mstore(0x40, bal)
                    if iszero(call(gas(), token, 0, 0x1c, 0x44, 0x80, 0x20)) {
                        bubbleRevert()
                    }
                    requireTruthyReturn()
                }
                stop()
            }

            // ---- setOwner(address) -----------------------------------
            case 0x13af4035 {
                requireOwner()
                let next := shr(96, shl(96, calldataload(0x04)))
                if iszero(next) { revert(0x00, 0x00) }
                sstore(0x00, next)
                stop()
            }

            default { revert(0x00, 0x00) }


            // =========================================================
            // ROUTE EXECUTION
            // =========================================================
            function runRoute(ptr, len, asset, amount, premium) {
                // The route copy starts at `ptr`. Field offsets below are
                // the packed layout documented in the header.
                let routeAsset := shr(96, mload(ptr))
                if iszero(eq(routeAsset, asset)) { revert(0x00, 0x00) }

                let minProfit := mload(add(ptr, 0x34))
                let hops := byte(0, mload(add(ptr, 0x54)))
                if iszero(hops) { revert(0x00, 0x00) }
                if gt(hops, 8) { revert(0x00, 0x00) }
                if iszero(eq(len, add(0x55, mul(42, hops)))) {
                    revert(0x00, 0x00)
                }

                let tokenIn := asset
                let amountIn := amount

                for { let i := 0 } lt(i, hops) { i := add(i, 1) } {
                    let h := add(add(ptr, 0x55), mul(i, 42))
                    let pool := shr(96, mload(h))
                    let tokenOut := shr(96, mload(add(h, 0x14)))
                    let flags := byte(0, mload(add(h, 0x28)))

                    // A hop that yields the token it consumed is either a
                    // packing error or an attempt to make the profit check
                    // trivially pass.
                    if eq(tokenIn, tokenOut) { revert(0x00, 0x00) }

                    switch and(flags, 1)
                    case 0 {
                        amountIn := hopV2(pool, tokenIn, amountIn,
                                          and(shr(1, flags), 1))
                    }
                    default {
                        amountIn := hopV3(pool, tokenIn, amountIn,
                                          and(shr(1, flags), 1))
                    }
                    if iszero(amountIn) { revert(0x00, 0x00) }
                    tokenIn := tokenOut
                }

                // The cycle must close on the borrowed asset, or the
                // balance check below is measuring the wrong token.
                if iszero(eq(tokenIn, asset)) { revert(0x00, 0x00) }

                // ---- THE ATOMIC UNDO ---------------------------------
                // Everything above is undone by this revert if the trade
                // did not clear the loan, Aave's premium, and the caller's
                // own floor. Balance is read from the token rather than
                // tracked locally: fee-on-transfer and rebasing tokens make
                // the accumulator lie, and the only number that matters is
                // what this contract can actually hand back.
                mstore(0x1c, 0x70a08231)            // balanceOf(address)
                mstore(0x20, address())
                if iszero(staticcall(gas(), asset, 0x1c, 0x24, 0x80, 0x20)) {
                    bubbleRevert()
                }
                let bal := mload(0x80)
                let floor := add(add(amount, premium), minProfit)
                if lt(bal, floor) { revert(0x00, 0x00) }
            }

            // ---- Uniswap V2-style hop --------------------------------
            // Pay the pair first, then call swap(). V2 pairs compute the
            // input from their own balance delta, so the transfer must land
            // before the call.
            function hopV2(pool, tokenIn, amountIn, zeroForOne) -> out {
                mstore(0x1c, 0x0902f1ac)            // getReserves()
                if iszero(staticcall(gas(), pool, 0x1c, 0x04, 0xc0, 0x60)) {
                    bubbleRevert()
                }
                let r0 := mload(0xc0)
                let r1 := mload(0xe0)

                let rIn := r1
                let rOut := r0
                if zeroForOne { rIn := r0 rOut := r1 }
                if or(iszero(rIn), iszero(rOut)) { revert(0x00, 0x00) }

                out := getAmountOut(amountIn, rIn, rOut)
                if iszero(out) { revert(0x00, 0x00) }
                if iszero(lt(out, rOut)) { revert(0x00, 0x00) }

                mstore(0x1c, 0xa9059cbb)            // transfer(address,uint256)
                mstore(0x20, pool)
                mstore(0x40, amountIn)
                if iszero(call(gas(), tokenIn, 0, 0x1c, 0x44, 0x80, 0x20)) {
                    bubbleRevert()
                }
                requireTruthyReturn()

                // swap(uint amount0Out, uint amount1Out, address to, bytes)
                let a0 := out
                let a1 := 0
                if zeroForOne { a0 := 0 a1 := out }

                mstore(0x1c, 0x022c0d9f)
                mstore(0x20, a0)
                mstore(0x40, a1)
                mstore(0x60, address())
                mstore(0x80, 0x80)                  // offset to bytes
                mstore(0xa0, 0x00)                  // zero-length data
                if iszero(call(gas(), pool, 0, 0x1c, 0xa4, 0x00, 0x00)) {
                    bubbleRevert()
                }
            }

            // ---- Uniswap V3-style hop --------------------------------
            // V3 is pay-on-callback: swap() calls us back mid-execution and
            // we transfer then. The transient guard is set immediately
            // before the call and cleared immediately after, so the window
            // in which the callback will accept anything is exactly the
            // duration of our own call.
            function hopV3(pool, tokenIn, amountIn, zeroForOne) -> out {
                if gt(amountIn, 0x7fffffffffffffffffffffffffffffff) {
                    revert(0x00, 0x00)              // must fit int256 safely
                }

                tstore(0x00, pool)
                tstore(0x01, tokenIn)

                // sqrtPriceLimitX96: the extreme of the legal range, so the
                // pool never stops the swap early on price. Slippage is
                // enforced by the profit check, not here.
                let limit := 1461446703485210103287273052203988822378723970341
                if zeroForOne { limit := 4295128740 }

                mstore(0x1c, 0x128acb08)
                mstore(0x20, address())
                mstore(0x40, zeroForOne)
                mstore(0x60, amountIn)              // positive = exact input
                mstore(0x80, limit)
                mstore(0xa0, 0xa0)                  // offset to bytes
                mstore(0xc0, 0x20)                  // length: one word
                mstore(0xe0, tokenIn)
                if iszero(call(gas(), pool, 0, 0x1c, 0xe4, 0x0100, 0x40)) {
                    bubbleRevert()
                }

                tstore(0x00, 0x00)
                tstore(0x01, 0x00)

                // Returns (int256 amount0, int256 amount1). What we receive
                // is the negative one; negate it back to a magnitude.
                // zeroForOne means we paid token0 and received token1, so
                // the credit is amount1; otherwise it is amount0.
                let recv := mload(0x0100)           // amount0
                if zeroForOne { recv := mload(0x0120) }   // amount1
                if iszero(slt(recv, 0)) { revert(0x00, 0x00) }
                out := sub(0, recv)
            }

            // ---- x*997*rOut / (rIn*1000 + x*997) ---------------------
            // The V2 constant-product quote with the 0.3% fee folded in.
            // Both multiplications are bounded before they happen: reserves
            // are uint112 by the pair's own storage, and amountIn is capped
            // at 2^128, so neither product can wrap a uint256.
            function getAmountOut(amountIn, rIn, rOut) -> out {
                if gt(amountIn, 0xffffffffffffffffffffffffffffffff) {
                    revert(0x00, 0x00)
                }
                let inWithFee := mul(amountIn, 997)
                let numerator := mul(inWithFee, rOut)
                let denominator := add(mul(rIn, 1000), inWithFee)
                if iszero(denominator) { revert(0x00, 0x00) }
                out := div(numerator, denominator)
            }

            // ---- helpers ---------------------------------------------
            function requireOwner() {
                if iszero(eq(caller(), sload(0x00))) { revert(0x00, 0x00) }
            }

            // ERC20s disagree about return values: some return true, some
            // return nothing, some (USDT) return nothing and are still
            // fine. Accept empty returndata; reject an explicit false.
            function requireTruthyReturn() {
                if gt(returndatasize(), 0) {
                    if iszero(mload(0x80)) { revert(0x00, 0x00) }
                }
            }

            function roundUp32(n) -> r {
                r := mul(div(add(n, 31), 32), 32)
            }

            // Forward the callee's revert reason instead of swallowing it.
            // Costs a few gas on the failure path and saves hours when a
            // pool reverts for a reason worth reading.
            function bubbleRevert() {
                let s := returndatasize()
                if iszero(s) { revert(0x00, 0x00) }
                returndatacopy(0x0300, 0x00, s)
                revert(0x0300, s)
            }
        }
    }
}
