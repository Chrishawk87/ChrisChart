import { ethers } from 'ethers';
import { packRoute, PriceGraph, isqrt, FlashbotsBundleManager, TOKENS,
         ROUTE_HEADER_BYTES, ROUTE_HOP_BYTES, ArbitragePath, Hop } from './mev-engine';

let fails = 0;
const ok = (c: boolean, m: string) => { console.log(`${c ? 'PASS' : 'FAIL'}  ${m}`); if (!c) fails++; };

// ---- 1. Packed route must match the Yul decoder byte for byte -------------
const hops: Hop[] = [
  { pool: '0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc', tokenIn: TOKENS.WETH.address,
    tokenOut: TOKENS.USDC.address, kind: 'v2', zeroForOne: false },
  { pool: '0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640', tokenIn: TOKENS.USDC.address,
    tokenOut: TOKENS.WETH.address, kind: 'v3', zeroForOne: true },
];
const path: ArbitragePath = { hops, flashAsset: TOKENS.WETH.address,
  flashAmount: ethers.parseEther('12.5'), grossOut: 0n, netBeforeGas: 0n, id: 't' };
const packed = ethers.getBytes(packRoute(path, 7n));

// Pinned to literals on purpose. Deriving the expected length from the same
// constants the packer uses makes the assertion tautological: it would pass
// even if both sides drifted away from what the Yul decoder expects.
// 0x55 = 85 header bytes + 42 per hop. Two hops = 169.
ok(packed.length === 169, `packed length ${packed.length} === 169 (0x55 + 2*42)`);
ok(ROUTE_HEADER_BYTES === 85 && ROUTE_HOP_BYTES === 42,
   `constants match the Yul offsets (${ROUTE_HEADER_BYTES}, ${ROUTE_HOP_BYTES})`);
ok(ethers.hexlify(packed.slice(0x00, 0x14)).toLowerCase() === TOKENS.WETH.address.toLowerCase(),
   'flashAsset at 0x00..0x14');
ok(BigInt(ethers.hexlify(packed.slice(0x14, 0x34))) === ethers.parseEther('12.5'),
   'flashAmount at 0x14..0x34');
ok(BigInt(ethers.hexlify(packed.slice(0x34, 0x54))) === 7n, 'minProfit at 0x34..0x54');
ok(packed[0x54] === 2, 'hopCount at 0x54');

const h0 = ROUTE_HEADER_BYTES;
ok(packed[h0 + 0x28] === 0, 'hop0 flags: v2 + !zeroForOne = 0b00');
const h1 = ROUTE_HEADER_BYTES + ROUTE_HOP_BYTES;
ok(packed[h1 + 0x28] === 3, 'hop1 flags: v3 + zeroForOne = 0b11');
ok(ethers.hexlify(packed.slice(h1 + 0x14, h1 + 0x28)).toLowerCase() === TOKENS.WETH.address.toLowerCase(),
   'last hop tokenOut closes on the flash asset');

// A route that does not close must be refused before it costs gas.
let threw = false;
try { packRoute({ ...path, hops: [hops[0]] }, 0n); } catch { threw = true; }
ok(threw, 'open-ended route rejected client-side');

// ---- 2. V2 quote against the canonical Uniswap formula --------------------
const rIn = 5_000_000n * 10n ** 6n, rOut = 2_000n * 10n ** 18n;
const amtIn = 10_000n * 10n ** 6n;
const expect = (amtIn * 997n * rOut) / (rIn * 1000n + amtIn * 997n);
ok(PriceGraph.quoteV2(amtIn, rIn, rOut, 3000) === expect, 'quoteV2 matches x*997*rOut/(rIn*1000+x*997)');
ok(PriceGraph.quoteV2(0n, rIn, rOut, 3000) === 0n, 'zero in -> zero out');
ok(PriceGraph.quoteV2(amtIn, 0n, rOut, 3000) === 0n, 'empty reserve -> zero out');

// More in must never yield less out.
let mono = true, prev = 0n;
for (let i = 1n; i <= 50n; i++) {
  const o = PriceGraph.quoteV2(amtIn * i, rIn, rOut, 3000);
  if (o < prev) mono = false;
  prev = o;
}
ok(mono, 'quoteV2 is monotonic in amountIn');

// Fee must reduce output.
ok(PriceGraph.quoteV2(amtIn, rIn, rOut, 100) > PriceGraph.quoteV2(amtIn, rIn, rOut, 10000),
   'lower fee tier yields more out');

// ---- 3. V3 single-tick quote ---------------------------------------------
const sqrtP = 1771845812700903892492222464n;   // ~WETH/USDC range
const L = 12_000_000_000_000_000_000n;
const q0 = PriceGraph.quoteV3SingleTick(10n ** 18n, sqrtP, L, true, 500);
const q1 = PriceGraph.quoteV3SingleTick(10n ** 18n, sqrtP, L, false, 500);
ok(q0 > 0n && q1 > 0n, `v3 quotes both directions (${q0}, ${q1})`);
ok(PriceGraph.quoteV3SingleTick(10n ** 18n, sqrtP, 0n, true, 500) === 0n, 'no liquidity -> zero');
ok(PriceGraph.quoteV3SingleTick(10n ** 18n, sqrtP, L, true, 500) >
   PriceGraph.quoteV3SingleTick(10n ** 18n, sqrtP, L, true, 10000), 'v3 fee tier reduces output');

// ---- 4. isqrt -------------------------------------------------------------
ok(isqrt(0n) === 0n && isqrt(1n) === 1n && isqrt(144n) === 12n, 'isqrt exact squares');
const big = (10n ** 30n) ** 2n;
ok(isqrt(big) === 10n ** 30n, 'isqrt of a 60-digit square');
ok(isqrt(99n) === 9n, 'isqrt floors');

// ---- 5. Fee computation ---------------------------------------------------
const w1 = ethers.Wallet.createRandom(), w2 = ethers.Wallet.createRandom();
const prov = new ethers.JsonRpcProvider('http://127.0.0.1:1', 1, { staticNetwork: true });
const mgr = new FlashbotsBundleManager(prov, w1.privateKey, w2.privateKey);
const base = ethers.parseUnits('20', 'gwei');
const f = mgr.computeFees(base, ethers.parseEther('0.5'), 900_000n);
ok(f.maxPriorityFeePerGas > 0n, 'priority fee is positive');
ok(f.maxFeePerGas > base, 'maxFee exceeds current base fee');
ok(f.bidTotal <= ethers.parseEther('0.5'), 'builder bid never exceeds the profit it is paid from');
const capped = mgr.computeFees(ethers.parseUnits('400', 'gwei'), ethers.parseEther('50'), 900_000n);
ok(capped.maxFeePerGas <= ethers.parseUnits('250', 'gwei'), 'maxFeePerGas ceiling is respected');

// Reusing one key for both roles must be refused.
threw = false;
try { new FlashbotsBundleManager(prov, w1.privateKey, w1.privateKey); } catch { threw = true; }
ok(threw, 'auth key must differ from execution key');

prov.destroy();
console.log(fails === 0 ? '\nALL CHECKS PASSED' : `\n${fails} FAILED`);
process.exit(fails === 0 ? 0 : 1);
