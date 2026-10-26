// ============================================================================
// MONOLITHIC EVM ARBITRAGE SYSTEM
// ============================================================================
//
// Cross-DEX atomic arbitrage over Uniswap V2/V3 forks, executed through the
// MevArb Yul contract (contracts/MevArb.yul) and submitted privately via
// Flashbots. Flash-loaned principal, so the only capital at risk per attempt
// is gas.
//
// WHAT THIS DOES AND DOES NOT DO
// ------------------------------
// It finds cycles where a token is mispriced BETWEEN pools and trades the
// cycle closed. That profit comes from the price gap itself; nobody's pending
// order is made worse by it. It also back-runs MEV-Share hints, where the
// originating user has opted into sharing their orderflow and receives a
// refund of what the back-run earns.
//
// It does NOT front-run or sandwich. There is no code here that places a
// transaction ahead of an identified victim's swap to move the price they
// execute at. See README.md.
//
// RUN:  npm i ethers@6.13.4 ws@8.18.0
//       npx tsx src/mev-engine.ts
//
// ============================================================================

import { ethers } from 'ethers';
import WebSocket from 'ws';
import { EventEmitter } from 'node:events';
import * as path from 'node:path';
import { loadEnvAndReport } from './env';

// Populate process.env from mev/.env BEFORE CONFIG is built, since CONFIG
// reads process.env at module load. Real environment variables always win,
// so Railway — which sets them directly and ships no .env — is unaffected.
loadEnvAndReport(path.resolve(__dirname, '..'));

// ============================================================================
// 1. TYPE DEFINITIONS & ENVIRONMENT GLOBAL CONSTANTS
// ============================================================================

export type PoolKind = 'v2' | 'v3';

export interface TokenMeta {
  readonly address: string;
  readonly symbol: string;
  readonly decimals: number;
}

export interface PoolMeta {
  readonly address: string;
  readonly kind: PoolKind;
  readonly token0: string;
  readonly token1: string;
  /** V2: 30 (0.30%). V3: the tier in hundredths of a bip — 100/500/3000/10000. */
  readonly feePpm: number;
  readonly label: string;
}

/** Live pool state. V2 carries reserves; V3 carries price and in-range depth. */
export interface PoolState {
  readonly address: string;
  readonly kind: PoolKind;
  reserve0: bigint;
  reserve1: bigint;
  sqrtPriceX96: bigint;
  liquidity: bigint;
  blockNumber: number;
}

export interface Hop {
  readonly pool: string;
  readonly tokenIn: string;
  readonly tokenOut: string;
  readonly kind: PoolKind;
  /** True when tokenIn is the pool's token0. Packed into the route flags. */
  readonly zeroForOne: boolean;
}

export interface ArbitragePath {
  readonly hops: readonly Hop[];
  /** The borrowed and repaid token. Equals hops[0].tokenIn. */
  readonly flashAsset: string;
  flashAmount: bigint;
  /** Gross output of the cycle, before loan premium and gas. */
  grossOut: bigint;
  /** grossOut - flashAmount - aavePremium. Still before gas. */
  netBeforeGas: bigint;
  id: string;
}

export interface MempoolTx {
  readonly hash: string;
  readonly to: string | null;
  readonly from: string;
  readonly input: string;
  readonly value: bigint;
  readonly maxFeePerGas: bigint | null;
  readonly maxPriorityFeePerGas: bigint | null;
  readonly gasLimit: bigint;
  readonly seenAt: number;
}

/** A MEV-Share hint. `hash` is all we are given for a fully private tx. */
export interface ShareHint {
  readonly hash: string;
  readonly logs: readonly { address: string; topics: string[]; data: string }[];
  readonly txs: readonly { to?: string; functionSelector?: string }[];
  readonly seenAt: number;
}

export interface BundleResult {
  readonly ok: boolean;
  readonly bundleHash?: string;
  readonly error?: string;
  readonly simProfitWei?: bigint;
  readonly simGasUsed?: bigint;
}

// ----------------------------------------------------------------------------
// Addresses. Ethereum mainnet. Every one of these is load-bearing — a wrong
// address here is a silent failure at best and a loss at worst, so they are
// declared once, here, and never inlined anywhere below.
// ----------------------------------------------------------------------------

export const TOKENS: Record<string, TokenMeta> = {
  WETH: { address: '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2', symbol: 'WETH', decimals: 18 },
  USDC: { address: '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48', symbol: 'USDC', decimals: 6  },
  USDT: { address: '0xdAC17F958D2ee523a2206206994597C13D831ec7', symbol: 'USDT', decimals: 6  },
  DAI:  { address: '0x6B175474E89094C44Da98b954EedeAC495271d0F', symbol: 'DAI',  decimals: 18 },
  WBTC: { address: '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599', symbol: 'WBTC', decimals: 8  },
};

export const CONFIG = {
  chainId: 1,

  // ---- Endpoints ----------------------------------------------------------
  rpcUrl:        process.env.RPC_URL        ?? 'https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY',
  wsUrl:         process.env.WS_URL         ?? 'wss://eth-mainnet.g.alchemy.com/v2/YOUR_KEY',
  flashbotsRelay: process.env.FB_RELAY      ?? 'https://relay.flashbots.net',
  mevShareSse:   process.env.MEV_SHARE_SSE  ?? 'https://mev-share.flashbots.net',

  // ---- Keys. Never literals. ---------------------------------------------
  // EXECUTION_KEY signs the arbitrage transaction and must hold ETH for gas.
  // AUTH_KEY signs the Flashbots request header only; it is an identity, not
  // a funded account, and must be a DIFFERENT key. Reusing the execution key
  // publishes your searcher identity alongside every bundle.
  executionKey: process.env.EXECUTION_KEY ?? '',
  authKey:      process.env.FLASHBOTS_AUTH_KEY ?? '',

  // ---- Deployed contracts -------------------------------------------------
  executor:   process.env.EXECUTOR_ADDRESS ?? '',
  aaveV3Pool: '0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2',
  multicall3: '0xcA11bde05977b3631167028862bE2a173976CA11',

  // ---- Economics ----------------------------------------------------------
  /** Aave V3 flashLoanSimple premium, in basis points (0.05%). */
  aavePremiumBps: 5n,
  /** Abandon anything that does not clear this much after gas and the tip. */
  minNetProfitWei: ethers.parseEther('0.004'),
  /** Share of net profit paid to the builder. The rest is yours. */
  builderTipBps: 5000n,
  /** Hard ceiling on gas for one attempt. A route that needs more is a bug. */
  gasLimit: 900_000n,
  /** Refuse to bid above this effective gas price, whatever the profit says. */
  maxFeePerGasWei: ethers.parseUnits('250', 'gwei'),
  /** Fraction of a pool's reserve we will route through it, in bips. */
  maxPoolImpactBps: 30n,
  /** Blocks ahead to target. 1 = the next block. */
  targetBlockOffset: 1,

  // ---- Search -------------------------------------------------------------
  /** Cycle lengths to enumerate. 2 = A->B->A across two pools; 3 = triangle. */
  cycleLengths: [2, 3] as const,
  /** Binary-search steps when sizing a path. 24 gets within ~0.1% of optimal. */
  sizingIterations: 24,
  /** Stop scanning a block after this long; the next block is more valuable. */
  scanBudgetMs: 120,

  // ---- Connection health --------------------------------------------------
  wsPingIntervalMs: 10_000,
  wsPongTimeoutMs: 8_000,
  wsReconnectBaseMs: 250,
  wsReconnectMaxMs: 15_000,

  dryRun: process.env.DRY_RUN !== 'false',
} as const;

/**
 * The pools to watch. Keep this short and curated: the scan is O(pools^cycle)
 * and a long list costs you the block. These are real mainnet addresses.
 */
export const POOLS: PoolMeta[] = [
  { address: '0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc', kind: 'v2',
    token0: TOKENS.USDC.address, token1: TOKENS.WETH.address, feePpm: 3000,
    label: 'UniV2 USDC/WETH' },
  { address: '0x0d4a11d5EEaaC28EC3F61d100daF4d40471f1852', kind: 'v2',
    token0: TOKENS.WETH.address, token1: TOKENS.USDT.address, feePpm: 3000,
    label: 'UniV2 WETH/USDT' },
  { address: '0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640', kind: 'v3',
    token0: TOKENS.USDC.address, token1: TOKENS.WETH.address, feePpm: 500,
    label: 'UniV3 USDC/WETH 0.05%' },
  { address: '0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8', kind: 'v3',
    token0: TOKENS.USDC.address, token1: TOKENS.WETH.address, feePpm: 3000,
    label: 'UniV3 USDC/WETH 0.30%' },
  { address: '0x4e68Ccd3E89f51C3074ca5072bbAC773960dFa36', kind: 'v3',
    token0: TOKENS.WETH.address, token1: TOKENS.USDT.address, feePpm: 3000,
    label: 'UniV3 WETH/USDT 0.30%' },
  { address: '0xCBCdF9626bC03E24f779434178A73a0B4bad62eD', kind: 'v3',
    token0: TOKENS.WBTC.address, token1: TOKENS.WETH.address, feePpm: 3000,
    label: 'UniV3 WBTC/WETH 0.30%' },
];

// ============================================================================
// 2. THE EXECUTION VEHICLE — calldata packing for contracts/MevArb.yul
// ============================================================================
//
// There is no bytecode constant here, and that is deliberate. A hex string I
// could not compile and you could not read would be the single most dangerous
// line in this repository. Compile the Yul yourself:
//
//   solc --strict-assembly --optimize --optimize-runs 4294967295 \
//        --bin contracts/MevArb.yul
//
// then append the Aave pool address as a left-padded word to the creation
// bytecode (see README). ROUTE_LAYOUT below is the contract's calldata
// contract, byte for byte; if you edit one side, edit both.
//
//   0x00   20   flashAsset
//   0x14   32   flashAmount   uint256
//   0x34   32   minProfit     uint256, in flashAsset units
//   0x54    1   hopCount      uint8, 1..8
//   0x55   ..   hops, 42 bytes each:
//                 +0x00  20  pool
//                 +0x14  20  tokenOut
//                 +0x28   1  flags: bit0 isV3, bit1 zeroForOne
//                 +0x29   1  reserved, zero
// ============================================================================

export const ROUTE_HEADER_BYTES = 0x55;
export const ROUTE_HOP_BYTES = 42;
export const MAX_HOPS = 8;

export const EXECUTOR_ABI = [
  'function run(bytes route)',
  'function sweep(address token)',
  'function setOwner(address next)',
] as const;

const EXECUTOR_IFACE = new ethers.Interface(EXECUTOR_ABI as unknown as string[]);

function addrBytes(a: string): Uint8Array {
  return ethers.getBytes(ethers.getAddress(a));          // throws on bad input
}

function u256Bytes(v: bigint): Uint8Array {
  if (v < 0n || v > (1n << 256n) - 1n) throw new RangeError(`u256 out of range: ${v}`);
  return ethers.getBytes(ethers.zeroPadValue(ethers.toBeHex(v), 32));
}

/** Byte-for-byte mirror of the Yul route decoder. */
export function packRoute(path: ArbitragePath, minProfit: bigint): string {
  if (path.hops.length === 0 || path.hops.length > MAX_HOPS) {
    throw new RangeError(`hop count ${path.hops.length} outside 1..${MAX_HOPS}`);
  }
  const last = path.hops[path.hops.length - 1];
  if (ethers.getAddress(last.tokenOut) !== ethers.getAddress(path.flashAsset)) {
    // The contract rejects this too. Catching it here saves the gas.
    throw new Error('route does not close on the flash asset');
  }

  const out = new Uint8Array(ROUTE_HEADER_BYTES + ROUTE_HOP_BYTES * path.hops.length);
  out.set(addrBytes(path.flashAsset), 0x00);
  out.set(u256Bytes(path.flashAmount), 0x14);
  out.set(u256Bytes(minProfit), 0x34);
  out[0x54] = path.hops.length;

  path.hops.forEach((h, i) => {
    const at = ROUTE_HEADER_BYTES + i * ROUTE_HOP_BYTES;
    out.set(addrBytes(h.pool), at + 0x00);
    out.set(addrBytes(h.tokenOut), at + 0x14);
    out[at + 0x28] = (h.kind === 'v3' ? 1 : 0) | (h.zeroForOne ? 2 : 0);
    out[at + 0x29] = 0;
  });
  return ethers.hexlify(out);
}

export function encodeRunCall(path: ArbitragePath, minProfit: bigint): string {
  return EXECUTOR_IFACE.encodeFunctionData('run', [packRoute(path, minProfit)]);
}

// ============================================================================
// 3. LOW-LATENCY GRAPH MATH & PRICING NORMALIZATION ENGINE
// ============================================================================

const SEL = {
  getReserves: '0x0902f1ac',
  slot0:       '0x3850c7bd',
  liquidity:   '0x1a686502',
} as const;

const MULTICALL3_ABI = [
  'function aggregate3((address target, bool allowFailure, bytes callData)[] calls) payable returns ((bool success, bytes returnData)[])',
] as const;

const Q96 = 1n << 96n;

/** Integer sqrt, Newton's method. Needed for optimal V2 cycle sizing. */
export function isqrt(n: bigint): bigint {
  if (n < 0n) throw new RangeError('isqrt of negative');
  if (n < 2n) return n;
  let x = n, y = (x + 1n) >> 1n;
  while (y < x) { x = y; y = (x + n / x) >> 1n; }
  return x;
}

export class PriceGraph {
  private readonly states = new Map<string, PoolState>();
  private readonly byPool = new Map<string, PoolMeta>();
  private readonly multicall: ethers.Contract;

  constructor(
    provider: ethers.JsonRpcProvider,
    private readonly pools: readonly PoolMeta[],
  ) {
    for (const p of pools) this.byPool.set(p.address.toLowerCase(), p);
    this.multicall = new ethers.Contract(CONFIG.multicall3, MULTICALL3_ABI as unknown as string[], provider);
  }

  meta(pool: string): PoolMeta | undefined {
    return this.byPool.get(pool.toLowerCase());
  }

  state(pool: string): PoolState | undefined {
    return this.states.get(pool.toLowerCase());
  }

  /**
   * One multicall for every pool. A per-pool eth_call round trip costs tens of
   * milliseconds each and you have roughly one block — twelve seconds, most of
   * which belongs to the builder — so batching is not an optimization here,
   * it is the difference between landing and not.
   */
  async refresh(blockNumber: number): Promise<void> {
    const calls: { target: string; allowFailure: boolean; callData: string }[] = [];
    const plan: { pool: PoolMeta; field: 'reserves' | 'slot0' | 'liquidity' }[] = [];

    for (const p of this.pools) {
      if (p.kind === 'v2') {
        calls.push({ target: p.address, allowFailure: true, callData: SEL.getReserves });
        plan.push({ pool: p, field: 'reserves' });
      } else {
        calls.push({ target: p.address, allowFailure: true, callData: SEL.slot0 });
        plan.push({ pool: p, field: 'slot0' });
        calls.push({ target: p.address, allowFailure: true, callData: SEL.liquidity });
        plan.push({ pool: p, field: 'liquidity' });
      }
    }

    const results: { success: boolean; returnData: string }[] =
      await this.multicall.aggregate3.staticCall(calls, { blockTag: blockNumber });

    for (let i = 0; i < plan.length; i++) {
      const { pool, field } = plan[i];
      const r = results[i];
      if (!r?.success || r.returnData === '0x') continue;

      const key = pool.address.toLowerCase();
      const cur: PoolState = this.states.get(key) ?? {
        address: pool.address, kind: pool.kind,
        reserve0: 0n, reserve1: 0n, sqrtPriceX96: 0n, liquidity: 0n,
        blockNumber,
      };

      const words = ethers.getBytes(r.returnData);
      const word = (n: number): bigint =>
        BigInt(ethers.hexlify(words.slice(n * 32, n * 32 + 32)));

      if (field === 'reserves') {
        // getReserves packs uint112,uint112,uint32 into three ABI words.
        cur.reserve0 = word(0);
        cur.reserve1 = word(1);
      } else if (field === 'slot0') {
        cur.sqrtPriceX96 = word(0);      // first member of slot0
      } else {
        cur.liquidity = word(0);
      }
      cur.blockNumber = blockNumber;
      this.states.set(key, cur);
    }
  }

  // --------------------------------------------------------------------
  // Quoting
  // --------------------------------------------------------------------

  /** Uniswap V2 constant product with the fee folded in. Exact. */
  static quoteV2(amountIn: bigint, reserveIn: bigint, reserveOut: bigint, feePpm: number): bigint {
    if (amountIn <= 0n || reserveIn <= 0n || reserveOut <= 0n) return 0n;
    const feeNum = BigInt(1_000_000 - feePpm);
    const inAfterFee = amountIn * feeNum;
    const out = (inAfterFee * reserveOut) / (reserveIn * 1_000_000n + inAfterFee);
    return out < reserveOut ? out : 0n;
  }

  /**
   * Uniswap V3, SINGLE TICK ONLY.
   *
   * This treats in-range liquidity as constant, which is true until the swap
   * crosses a tick boundary and false the moment it does — at which point this
   * OVERSTATES the output, because real liquidity thins as you move away from
   * the current price. It is a scanner heuristic, fast enough to run over every
   * candidate path in a block, and it is wrong in the optimistic direction.
   *
   * That is survivable only because it is not the last word: the size is capped
   * by maxPoolImpactBps, the bundle is simulated against real state before it is
   * sent, and the contract reverts on-chain if the profit is not actually there.
   * If you widen maxPoolImpactBps, replace this with QuoterV2 or real tick-
   * crossing math first.
   */
  static quoteV3SingleTick(
    amountIn: bigint, sqrtPriceX96: bigint, liquidity: bigint,
    zeroForOne: boolean, feePpm: number,
  ): bigint {
    if (amountIn <= 0n || liquidity <= 0n || sqrtPriceX96 <= 0n) return 0n;
    const afterFee = (amountIn * BigInt(1_000_000 - feePpm)) / 1_000_000n;
    if (afterFee <= 0n) return 0n;

    if (zeroForOne) {
      // token0 in, token1 out. sqrtP falls.
      const denom = liquidity * Q96 + afterFee * sqrtPriceX96;
      if (denom === 0n) return 0n;
      const nextSqrt = (liquidity * Q96 * sqrtPriceX96) / denom;
      if (nextSqrt <= 0n || nextSqrt >= sqrtPriceX96) return 0n;
      return (liquidity * (sqrtPriceX96 - nextSqrt)) / Q96;
    }
    // token1 in, token0 out. sqrtP rises.
    const nextSqrt = sqrtPriceX96 + (afterFee * Q96) / liquidity;
    if (nextSqrt <= sqrtPriceX96) return 0n;
    const num = liquidity * Q96 * (nextSqrt - sqrtPriceX96);
    return num / (nextSqrt * sqrtPriceX96);
  }

  quoteHop(h: Hop, amountIn: bigint): bigint {
    const meta = this.meta(h.pool);
    const st = this.state(h.pool);
    if (!meta || !st) return 0n;

    if (meta.kind === 'v2') {
      const [rIn, rOut] = h.zeroForOne ? [st.reserve0, st.reserve1] : [st.reserve1, st.reserve0];
      return PriceGraph.quoteV2(amountIn, rIn, rOut, meta.feePpm);
    }
    return PriceGraph.quoteV3SingleTick(
      amountIn, st.sqrtPriceX96, st.liquidity, h.zeroForOne, meta.feePpm);
  }

  quotePath(hops: readonly Hop[], amountIn: bigint): bigint {
    let amt = amountIn;
    for (const h of hops) {
      amt = this.quoteHop(h, amt);
      if (amt <= 0n) return 0n;
    }
    return amt;
  }

  /** Largest input we will push through the thinnest pool on the path. */
  private sizeCeiling(hops: readonly Hop[]): bigint {
    let cap = ethers.MaxUint256;
    for (const h of hops) {
      const st = this.state(h.pool);
      if (!st) return 0n;
      let depth: bigint;
      if (st.kind === 'v2') {
        depth = h.zeroForOne ? st.reserve0 : st.reserve1;
      } else {
        // Rough in-range depth of the input token, from L and sqrtP.
        depth = h.zeroForOne
          ? (st.liquidity * Q96) / (st.sqrtPriceX96 === 0n ? 1n : st.sqrtPriceX96)
          : (st.liquidity * st.sqrtPriceX96) / Q96;
      }
      const allowed = (depth * CONFIG.maxPoolImpactBps) / 10_000n;
      if (allowed < cap) cap = allowed;
    }
    return cap === ethers.MaxUint256 ? 0n : cap;
  }

  /**
   * Size the trade. Profit as a function of input is unimodal for a cycle —
   * it rises while the edge outweighs price impact and falls after — so a
   * ternary search converges without needing a derivative, and works
   * unchanged across V2 and V3 quoting.
   */
  optimalSize(hops: readonly Hop[]): { amountIn: bigint; grossOut: bigint } {
    const hi0 = this.sizeCeiling(hops);
    if (hi0 <= 0n) return { amountIn: 0n, grossOut: 0n };

    const profit = (x: bigint): bigint => (x <= 0n ? 0n : this.quotePath(hops, x) - x);

    let lo = hi0 / 10_000n;
    let hi = hi0;
    if (lo <= 0n) lo = 1n;

    for (let i = 0; i < CONFIG.sizingIterations && hi > lo; i++) {
      const third = (hi - lo) / 3n;
      if (third <= 0n) break;
      const m1 = lo + third;
      const m2 = hi - third;
      if (profit(m1) < profit(m2)) lo = m1 + 1n; else hi = m2 - 1n;
    }

    const best = lo;
    const out = this.quotePath(hops, best);
    return out > best ? { amountIn: best, grossOut: out } : { amountIn: 0n, grossOut: 0n };
  }

  // --------------------------------------------------------------------
  // Cycle enumeration
  // --------------------------------------------------------------------

  private hopsFor(pool: PoolMeta, tokenIn: string): Hop | null {
    const t0 = pool.token0.toLowerCase();
    const t1 = pool.token1.toLowerCase();
    const tin = tokenIn.toLowerCase();
    if (tin === t0) {
      return { pool: pool.address, tokenIn: pool.token0, tokenOut: pool.token1,
               kind: pool.kind, zeroForOne: true };
    }
    if (tin === t1) {
      return { pool: pool.address, tokenIn: pool.token1, tokenOut: pool.token0,
               kind: pool.kind, zeroForOne: false };
    }
    return null;
  }

  /**
   * Every directed cycle of the configured lengths that starts and ends on
   * `start`, with no pool used twice. Enumerated rather than Bellman-Ford:
   * with a curated pool list the search space is small, the constant factor
   * matters more than the asymptotics, and an explicit walk lets us drop
   * paths on the thin-pool check before they are ever quoted.
   */
  enumerate(start: string, touchedPools?: ReadonlySet<string>): Hop[][] {
    const found: Hop[][] = [];
    const startLc = start.toLowerCase();

    const walk = (token: string, used: Set<string>, acc: Hop[], depth: number): void => {
      if (acc.length > 0 && token.toLowerCase() === startLc) {
        if (touchedPools && !acc.some((h) => touchedPools.has(h.pool.toLowerCase()))) return;
        found.push([...acc]);
        return;
      }
      if (depth === 0) return;
      for (const p of this.pools) {
        const key = p.address.toLowerCase();
        if (used.has(key)) continue;
        const hop = this.hopsFor(p, token);
        if (!hop) continue;
        used.add(key);
        acc.push(hop);
        walk(hop.tokenOut, used, acc, depth - 1);
        acc.pop();
        used.delete(key);
      }
    };

    for (const len of CONFIG.cycleLengths) {
      walk(start, new Set<string>(), [], len);
    }
    return found;
  }

  /** Score every cycle from `start` and return the profitable ones, best first. */
  findPaths(start: string, touchedPools?: ReadonlySet<string>): ArbitragePath[] {
    const deadline = Date.now() + CONFIG.scanBudgetMs;
    const out: ArbitragePath[] = [];

    for (const hops of this.enumerate(start, touchedPools)) {
      if (Date.now() > deadline) break;
      const { amountIn, grossOut } = this.optimalSize(hops);
      if (amountIn <= 0n) continue;

      const premium = (amountIn * CONFIG.aavePremiumBps) / 10_000n;
      const net = grossOut - amountIn - premium;
      if (net <= 0n) continue;

      out.push({
        hops, flashAsset: start, flashAmount: amountIn,
        grossOut, netBeforeGas: net,
        id: hops.map((h) => `${h.pool.slice(0, 8)}${h.zeroForOne ? '>' : '<'}`).join(''),
      });
    }
    out.sort((a, b) => (b.netBeforeGas > a.netBeforeGas ? 1 : b.netBeforeGas < a.netBeforeGas ? -1 : 0));
    return out;
  }
}

// ============================================================================
// 4. TRANSACTION STREAMING & NODE OPTIMIZATION PIPELINE
// ============================================================================

/**
 * A self-healing WebSocket subscriber.
 *
 * The failure this is built around is not the socket closing — that is loud
 * and easy. It is the socket staying OPEN while the node behind it stops
 * sending. From the client's side a dead feed and a quiet market are
 * identical, and a bot that cannot tell them apart will sit happily on stale
 * state and bid into prices that moved minutes ago. So liveness is asserted,
 * not assumed: a ping goes out on a timer and a missing pong is treated as a
 * dead connection even though nothing errored.
 */
export class MempoolScanner extends EventEmitter {
  private ws: WebSocket | null = null;
  private pingTimer: NodeJS.Timeout | null = null;
  private pongTimer: NodeJS.Timeout | null = null;
  private reconnectAt: number = CONFIG.wsReconnectBaseMs;
  private stopped = false;
  private nextId = 1;

  /** Pending subscription/request ids awaiting a reply. */
  private readonly inflight = new Map<number, (r: any) => void>();

  constructor(private readonly url: string) { super(); }

  start(): void { this.stopped = false; this.open(); }

  stop(): void {
    this.stopped = true;
    this.clearTimers();
    this.ws?.close();
    this.ws = null;
  }

  private clearTimers(): void {
    if (this.pingTimer) { clearInterval(this.pingTimer); this.pingTimer = null; }
    if (this.pongTimer) { clearTimeout(this.pongTimer); this.pongTimer = null; }
  }

  private open(): void {
    if (this.stopped) return;
    const ws = new WebSocket(this.url, { handshakeTimeout: 7_000 });
    this.ws = ws;

    ws.on('open', () => {
      this.reconnectAt = CONFIG.wsReconnectBaseMs;    // reset the backoff
      this.emit('open');
      this.send('eth_subscribe', ['newHeads']);
      // alchemy_pendingTransactions gives full bodies; newPendingTransactions
      // gives hashes only and costs an extra round trip per tx. Fall back
      // automatically if the node rejects the Alchemy method.
      this.send('eth_subscribe', ['alchemy_pendingTransactions'], (r) => {
        if (r?.error) this.send('eth_subscribe', ['newPendingTransactions']);
      });
      this.armHeartbeat(ws);
    });

    ws.on('message', (raw: WebSocket.RawData) => this.onMessage(raw));

    ws.on('pong', () => {
      if (this.pongTimer) { clearTimeout(this.pongTimer); this.pongTimer = null; }
    });

    ws.on('error', (err: Error) => this.emit('error', err));

    ws.on('close', (code: number, reason: Buffer) => {
      this.clearTimers();
      this.emit('close', code, reason?.toString());
      if (!this.stopped) this.scheduleReconnect();
    });
  }

  private armHeartbeat(ws: WebSocket): void {
    this.clearTimers();
    this.pingTimer = setInterval(() => {
      if (ws.readyState !== WebSocket.OPEN) return;
      // If a pong is already outstanding, the previous one never came back.
      if (this.pongTimer) return;
      try { ws.ping(); } catch { /* the close handler will deal with it */ }
      this.pongTimer = setTimeout(() => {
        this.emit('stale');
        // terminate(), not close(): a half-open socket will not complete a
        // graceful close handshake, and close() would hang there forever.
        try { ws.terminate(); } catch { /* already gone */ }
      }, CONFIG.wsPongTimeoutMs);
    }, CONFIG.wsPingIntervalMs);
  }

  private scheduleReconnect(): void {
    const wait = this.reconnectAt;
    // Exponential backoff with jitter. Without the jitter, every bot on a
    // provider reconnects in lockstep after an outage and knocks it over again.
    this.reconnectAt = Math.min(this.reconnectAt * 2, CONFIG.wsReconnectMaxMs);
    const jittered = wait / 2 + Math.random() * wait;
    setTimeout(() => this.open(), jittered);
  }

  private send(method: string, params: unknown[], onReply?: (r: any) => void): void {
    const id = this.nextId++;
    if (onReply) this.inflight.set(id, onReply);
    this.ws?.send(JSON.stringify({ jsonrpc: '2.0', id, method, params }));
  }

  private onMessage(raw: WebSocket.RawData): void {
    let msg: any;
    try { msg = JSON.parse(raw.toString()); } catch { return; }

    if (msg.id !== undefined && this.inflight.has(msg.id)) {
      this.inflight.get(msg.id)!(msg);
      this.inflight.delete(msg.id);
      return;
    }
    if (msg.method !== 'eth_subscription') return;

    const result = msg.params?.result;
    if (!result) return;

    if (result.number !== undefined && result.parentHash !== undefined) {
      this.emit('block', Number(BigInt(result.number)), result);
      return;
    }
    if (typeof result === 'string') { this.emit('pendingHash', result); return; }
    if (result.hash) this.emit('pending', toMempoolTx(result));
  }
}

function toMempoolTx(r: any): MempoolTx {
  const big = (v: unknown): bigint | null =>
    v === undefined || v === null ? null : BigInt(v as string);
  return {
    hash: r.hash,
    to: r.to ?? null,
    from: r.from,
    input: r.input ?? r.data ?? '0x',
    value: big(r.value) ?? 0n,
    maxFeePerGas: big(r.maxFeePerGas),
    maxPriorityFeePerGas: big(r.maxPriorityFeePerGas),
    gasLimit: big(r.gas) ?? 0n,
    seenAt: Date.now(),
  };
}

/**
 * MEV-Share hint stream. Consent-based by construction: the originating user
 * opted their transaction into the auction and is refunded a share of what a
 * back-run earns, which is why back-running these is a different act from
 * back-running something scraped out of the public mempool.
 */
export class MevShareStream extends EventEmitter {
  private controller: AbortController | null = null;
  private stopped = false;
  private backoff: number = CONFIG.wsReconnectBaseMs;

  constructor(private readonly url: string) { super(); }

  stop(): void { this.stopped = true; this.controller?.abort(); }

  async start(): Promise<void> {
    this.stopped = false;
    while (!this.stopped) {
      this.controller = new AbortController();
      try {
        const res = await fetch(this.url, {
          headers: { Accept: 'text/event-stream' },
          signal: this.controller.signal,
        });
        if (!res.ok || !res.body) throw new Error(`SSE ${res.status}`);
        this.backoff = CONFIG.wsReconnectBaseMs;
        this.emit('open');

        const reader = res.body.getReader();
        const dec = new TextDecoder();
        let buf = '';
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          // SSE frames are separated by a blank line; a partial frame stays
          // in the buffer until the rest of it arrives.
          let sep: number;
          while ((sep = buf.indexOf('\n\n')) !== -1) {
            const frame = buf.slice(0, sep);
            buf = buf.slice(sep + 2);
            for (const line of frame.split('\n')) {
              if (!line.startsWith('data:')) continue;
              try {
                const d = JSON.parse(line.slice(5).trim());
                this.emit('hint', {
                  hash: d.hash, logs: d.logs ?? [], txs: d.txs ?? [],
                  seenAt: Date.now(),
                } as ShareHint);
              } catch { /* keepalive or partial frame */ }
            }
          }
        }
      } catch (err) {
        if (this.stopped) return;
        this.emit('error', err);
      }
      if (this.stopped) return;
      const wait = this.backoff / 2 + Math.random() * this.backoff;
      this.backoff = Math.min(this.backoff * 2, CONFIG.wsReconnectMaxMs);
      await new Promise((r) => setTimeout(r, wait));
    }
  }
}

// ============================================================================
// 5. ATOMIC FLASHBOTS BUNDLE BUILDER & SUBMISSION CONTROLLER
// ============================================================================

export class FlashbotsBundleManager {
  private readonly authSigner: ethers.Wallet;
  private readonly execSigner: ethers.Wallet;

  constructor(
    private readonly provider: ethers.JsonRpcProvider,
    executionKey: string,
    authKey: string,
    private readonly relay: string = CONFIG.flashbotsRelay,
  ) {
    if (!executionKey) throw new Error('EXECUTION_KEY is not set');
    if (!authKey) throw new Error('FLASHBOTS_AUTH_KEY is not set');
    this.execSigner = new ethers.Wallet(executionKey, provider);
    this.authSigner = new ethers.Wallet(authKey);
    if (this.execSigner.address === this.authSigner.address) {
      // Same key for both publishes your searcher identity with every bundle
      // and ties your reputation to an address holding funds.
      throw new Error('auth key must differ from execution key');
    }
  }

  get executor(): string { return this.execSigner.address; }

  /**
   * Flashbots authenticates the BODY, not the connection: the header is
   * `address:signature` where the signature is a personal_sign over the hex
   * keccak of the exact JSON string sent. Re-serializing the body would
   * change the hash and the relay would reject it, so the string is built
   * once and both signed and sent.
   */
  private async post(method: string, params: unknown[]): Promise<any> {
    const body = JSON.stringify({ jsonrpc: '2.0', id: 1, method, params });
    const signature =
      `${this.authSigner.address}:${await this.authSigner.signMessage(ethers.id(body))}`;

    const res = await fetch(this.relay, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Flashbots-Signature': signature },
      body,
    });
    const text = await res.text();
    if (!res.ok) throw new Error(`relay ${res.status}: ${text.slice(0, 300)}`);
    return JSON.parse(text);
  }

  /**
   * Gas pricing. The builder is paid through priority fee rather than a
   * coinbase transfer: a coinbase transfer costs an extra ~9k gas and, more
   * to the point, pays the builder even when the arbitrage reverts, whereas
   * priority fee on a reverting transaction is still charged but the whole
   * bundle is dropped before it lands. Priority fee also keeps the bid legible
   * to every builder without needing them to special-case your contract.
   */
  computeFees(baseFeeWei: bigint, netProfitWei: bigint, gasEstimate: bigint): {
    maxFeePerGas: bigint; maxPriorityFeePerGas: bigint; bidTotal: bigint;
  } {
    const tipBudget = (netProfitWei * CONFIG.builderTipBps) / 10_000n;
    let priority = gasEstimate > 0n ? tipBudget / gasEstimate : 0n;
    if (priority < 1n) priority = 1n;

    // Next block's base fee can rise 12.5%; pad so a bundle is not orphaned
    // by a fee bump between simulation and inclusion.
    const headroom = (baseFeeWei * 1125n) / 1000n;
    let maxFee = headroom + priority;
    if (maxFee > CONFIG.maxFeePerGasWei) {
      maxFee = CONFIG.maxFeePerGasWei;
      priority = maxFee > headroom ? maxFee - headroom : 1n;
    }
    return { maxFeePerGas: maxFee, maxPriorityFeePerGas: priority,
             bidTotal: priority * gasEstimate };
  }

  async buildSignedTx(
    path: ArbitragePath, minProfit: bigint,
    baseFeeWei: bigint, netProfitWei: bigint,
  ): Promise<{ raw: string; hash: string; fees: ReturnType<FlashbotsBundleManager['computeFees']> }> {
    const fees = this.computeFees(baseFeeWei, netProfitWei, CONFIG.gasLimit);
    const nonce = await this.provider.getTransactionCount(this.execSigner.address, 'pending');

    const tx: ethers.TransactionRequest = {
      chainId: CONFIG.chainId,
      to: CONFIG.executor,
      data: encodeRunCall(path, minProfit),
      gasLimit: CONFIG.gasLimit,
      maxFeePerGas: fees.maxFeePerGas,
      maxPriorityFeePerGas: fees.maxPriorityFeePerGas,
      nonce,
      type: 2,
      value: 0n,
    };
    const raw = await this.execSigner.signTransaction(tx);
    return { raw, hash: ethers.keccak256(raw), fees };
  }

  /**
   * Simulate before sending. This is the step that separates a bot that loses
   * gas occasionally from one that loses it constantly: the scanner's V3
   * quoting is deliberately optimistic, and eth_callBundle runs the real
   * bytecode against real state at the real block.
   */
  async simulate(signedTxs: string[], blockNumber: number): Promise<BundleResult> {
    try {
      const r = await this.post('eth_callBundle', [{
        txs: signedTxs,
        blockNumber: ethers.toBeHex(blockNumber),
        stateBlockNumber: 'latest',
      }]);
      if (r.error) return { ok: false, error: r.error.message ?? JSON.stringify(r.error) };

      const results = r.result?.results ?? [];
      for (const x of results) {
        if (x.error || x.revert) {
          return { ok: false, error: `reverted: ${x.revert ?? x.error}` };
        }
      }
      return {
        ok: true,
        simProfitWei: BigInt(r.result?.coinbaseDiff ?? 0),
        simGasUsed: BigInt(r.result?.totalGasUsed ?? 0),
      };
    } catch (err) {
      return { ok: false, error: (err as Error).message };
    }
  }

  /** Private submission. Nothing here ever touches the public mempool. */
  async sendBundle(signedTxs: string[], blockNumber: number): Promise<BundleResult> {
    try {
      const r = await this.post('eth_sendBundle', [{
        txs: signedTxs,
        blockNumber: ethers.toBeHex(blockNumber),
      }]);
      if (r.error) return { ok: false, error: r.error.message ?? JSON.stringify(r.error) };
      return { ok: true, bundleHash: r.result?.bundleHash };
    } catch (err) {
      return { ok: false, error: (err as Error).message };
    }
  }

  /**
   * MEV-Share back-run. The body references the user's transaction by HASH —
   * we never see or reconstruct their calldata, and the relay places our
   * transaction after theirs. `refundPercent` is what goes back to them.
   */
  async sendShareBackrun(
    targetHash: string, signedTx: string, blockNumber: number,
    refundPercent = 90,
  ): Promise<BundleResult> {
    try {
      const r = await this.post('mev_sendBundle', [{
        version: 'v0.1',
        inclusion: {
          block: ethers.toBeHex(blockNumber),
          maxBlock: ethers.toBeHex(blockNumber + 2),
        },
        body: [{ hash: targetHash }, { tx: signedTx, canRevert: false }],
        validity: { refund: [{ bodyIdx: 0, percent: refundPercent }] },
      }]);
      if (r.error) return { ok: false, error: r.error.message ?? JSON.stringify(r.error) };
      return { ok: true, bundleHash: r.result?.bundleHash };
    } catch (err) {
      return { ok: false, error: (err as Error).message };
    }
  }
}

// ============================================================================
// 6. MAIN EXECUTION EVENT LOOP LIFECYCLE
// ============================================================================

function fmtEth(wei: bigint): string {
  return `${Number(ethers.formatEther(wei)).toFixed(6)} ETH`;
}

export async function startMevEngine(): Promise<void> {
  const provider = new ethers.JsonRpcProvider(CONFIG.rpcUrl, CONFIG.chainId, {
    staticNetwork: true,       // skip the chainId probe on every call
  });

  if (!ethers.isAddress(CONFIG.executor)) {
    throw new Error('EXECUTOR_ADDRESS is not set — deploy contracts/MevArb.yul first');
  }

  const graph = new PriceGraph(provider, POOLS);
  const bundles = new FlashbotsBundleManager(provider, CONFIG.executionKey, CONFIG.authKey);
  const scanner = new MempoolScanner(CONFIG.wsUrl);
  const share = new MevShareStream(CONFIG.mevShareSse);

  console.log(`executor contract : ${CONFIG.executor}`);
  console.log(`execution address : ${bundles.executor}`);
  console.log(`mode              : ${CONFIG.dryRun ? 'DRY RUN — simulate only' : 'LIVE'}`);
  console.log(`pools watched     : ${POOLS.length}`);

  let head = await provider.getBlockNumber();
  let baseFee = (await provider.getBlock('latest'))?.baseFeePerGas ?? 0n;
  let working = false;

  /** Pools touched by pending transactions since the last block. */
  let touched = new Set<string>();
  const poolAddrs = new Set(POOLS.map((p) => p.address.toLowerCase()));

  async function attempt(reason: string, hint?: ShareHint): Promise<void> {
    // One attempt at a time. Overlapping attempts reuse the same nonce and
    // the second is guaranteed to be discarded, having cost a simulation.
    if (working) return;
    working = true;
    const started = Date.now();
    try {
      const target = head + CONFIG.targetBlockOffset;
      await graph.refresh(head);

      const paths = graph.findPaths(
        TOKENS.WETH.address, touched.size > 0 ? touched : undefined);
      if (paths.length === 0) return;

      const best = paths[0];
      const fees = bundles.computeFees(baseFee, best.netBeforeGas, CONFIG.gasLimit);
      const gasCost = fees.maxFeePerGas * CONFIG.gasLimit;
      const net = best.netBeforeGas - gasCost;

      if (net < CONFIG.minNetProfitWei) {
        console.log(`[${reason}] best ${best.id} nets ${fmtEth(net)} after gas — below floor`);
        return;
      }

      // The on-chain floor. Set below the expected profit but above zero so a
      // path that decays between simulation and inclusion reverts rather than
      // landing for a loss. This is the number the Yul compares against.
      const minProfit = best.netBeforeGas / 2n;

      const { raw } = await bundles.buildSignedTx(
        best, minProfit, baseFee, best.netBeforeGas);

      const sim = await bundles.simulate([raw], target);
      if (!sim.ok) {
        console.log(`[${reason}] ${best.id} failed simulation: ${sim.error}`);
        return;
      }

      console.log(
        `[${reason}] ${best.id} | in ${best.flashAmount} | net ${fmtEth(net)} ` +
        `| sim gas ${sim.simGasUsed} | ${Date.now() - started}ms`);

      if (CONFIG.dryRun) { console.log('   dry run — not submitting'); return; }

      const res = hint
        ? await bundles.sendShareBackrun(hint.hash, raw, target)
        : await bundles.sendBundle([raw], target);

      console.log(res.ok
        ? `   submitted for block ${target}: ${res.bundleHash}`
        : `   submit failed: ${res.error}`);
    } catch (err) {
      console.error(`[${reason}] ${(err as Error).message}`);
    } finally {
      working = false;
    }
  }

  scanner.on('open', () => console.log('mempool socket open'));
  scanner.on('stale', () => console.warn('no pong — treating feed as dead, reconnecting'));
  scanner.on('close', (c, r) => console.warn(`socket closed ${c} ${r ?? ''}`));
  scanner.on('error', (e) => console.warn(`socket error: ${(e as Error).message}`));

  scanner.on('block', (n: number, header: any) => {
    head = n;
    if (header?.baseFeePerGas) baseFee = BigInt(header.baseFeePerGas);
    touched = new Set<string>();      // state is stale the moment a block lands
    void attempt('block');
  });

  scanner.on('pending', (tx: MempoolTx) => {
    // A pending swap moves a pool's price; that dislocation is what the cycle
    // closes. Only transactions aimed at a pool we track are worth a rescan.
    const to = tx.to?.toLowerCase();
    if (to && poolAddrs.has(to)) {
      touched.add(to);
      void attempt('pending');
    }
  });

  share.on('open', () => console.log('mev-share hint stream open'));
  share.on('error', (e) => console.warn(`mev-share: ${(e as Error).message}`));
  share.on('hint', (h: ShareHint) => {
    const hit = h.logs.some((l) => poolAddrs.has(l.address?.toLowerCase() ?? ''));
    if (!hit) return;
    for (const l of h.logs) {
      const a = l.address?.toLowerCase();
      if (a && poolAddrs.has(a)) touched.add(a);
    }
    void attempt('mev-share', h);
  });

  scanner.start();
  void share.start();

  const shutdown = (): void => {
    console.log('\nshutting down');
    scanner.stop();
    share.stop();
    provider.destroy();
    process.exit(0);
  };
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

if (require.main === module) {
  startMevEngine().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
