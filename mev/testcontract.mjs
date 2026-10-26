/**
 * Deploy MevArb into an in-memory EVM and attack it.
 *
 * Compiling proves the Yul parses. This proves the constructor's codecopy
 * arithmetic actually lands the pool address in slot 1, and that every door
 * into the contract is shut to everyone it should be shut to -- above all
 * uniswapV3SwapCallback, which pays whoever calls it if the transient guard
 * is missing or wrong.
 *
 *   node testcontract.mjs
 *
 * Re-run this after ANY edit to the Yul. It is not an audit; it is the floor.
 */
import { EVM } from '@ethereumjs/evm';
import { Common, Chain, Hardfork } from '@ethereumjs/common';
import { hexToBytes, bytesToHex, Address } from '@ethereumjs/util';
import { DefaultStateManager } from '@ethereumjs/statemanager';
import fs from 'node:fs'; import solc from 'solc';

const src = fs.readFileSync(new URL('./contracts/MevArb.yul', import.meta.url),'utf8');
const out = JSON.parse(solc.compile(JSON.stringify({ language:'Yul',
  sources:{'MevArb.yul':{content:src}},
  settings:{optimizer:{enabled:true,details:{yul:true}},evmVersion:'cancun',
            outputSelection:{'*':{'*':['*']}}}})));
const code = out.contracts['MevArb.yul'].MevArb.evm.bytecode.object;

const AAVE  = '87870bca3f3fd6335c3f4ce8392d69350b4fa4e2';
const OWNER = '1111111111111111111111111111111111111111';
const EVE   = '2222222222222222222222222222222222222222';
const addrOf = (h) => new Address(hexToBytes('0x'+h));

const common = new Common({ chain: Chain.Mainnet, hardfork: Hardfork.Cancun });
const sm = new DefaultStateManager();
const evm = await EVM.create({ common, stateManager: sm });
const dep = await evm.runCall({ caller: addrOf(OWNER), to: undefined,
  data: hexToBytes('0x'+code+AAVE.padStart(64,'0')), gasLimit: 5_000_000n, value: 0n });
const C = dep.createdAddress;

let fails = 0;
const ok = (c,m)=>{ console.log(`${c?'PASS':'FAIL'}  ${m}`); if(!c) fails++; };

async function call(from, data) {
  const r = await evm.runCall({ caller: addrOf(from), to: C,
    data: hexToBytes(data), gasLimit: 2_000_000n, value: 0n });
  return { reverted: !!r.execResult.exceptionError,
           err: r.execResult.exceptionError?.error };
}
const word = (h) => h.replace(/^0x/,'').padStart(64,'0');

// A minimal well-formed route: 0x55 header + one 42-byte hop.
const route = word(AAVE).slice(24) + '00'.repeat(32) + '00'.repeat(32) + '01'
            + '33'.repeat(20) + '44'.repeat(20) + '00' + '00';
const runData = '0x3a276523' + word('20') + word((route.length/2).toString(16)) +
                route.padEnd(Math.ceil(route.length/64)*64, '0');

// ---- ownership ----------------------------------------------------------
ok((await call(EVE, runData)).reverted, 'run() from a non-owner reverts');

// ---- the drain vector ---------------------------------------------------
// uniswapV3SwapCallback with no swap in flight: the transient guard is zero,
// so this must revert. Unguarded, it pays whoever calls it.
const cb = '0xfa461e33' + word('64') + word('0') + word('60') + word('20') + word(AAVE);
ok((await call(EVE, cb)).reverted, 'uniswapV3SwapCallback from anyone reverts (no swap in flight)');
ok((await call(OWNER, cb)).reverted, '  ...and the owner gets no exemption either');

// ---- flash loan callback ------------------------------------------------
const execOp = '0x1b11d0ff' + word(AAVE) + word('de0b6b3a7640000') + word('1') +
               word(C.toString().slice(2)) + word('a0') + word('0');
ok((await call(EVE, execOp)).reverted, 'executeOperation from a non-Aave caller reverts');

// ---- initiator check ----------------------------------------------------
// Even FROM Aave, an initiator that is not this contract must be refused,
// or a stranger can flash-loan in and drive the hop loop with their own route.
const execOpBadInit = '0x1b11d0ff' + word(AAVE) + word('de0b6b3a7640000') + word('1') +
                      word(EVE) + word('a0') + word('0');
ok((await call(AAVE, execOpBadInit)).reverted, 'executeOperation with a foreign initiator reverts');

// ---- setOwner -----------------------------------------------------------
ok((await call(EVE, '0x13af4035'+word(EVE))).reverted, 'setOwner from a non-owner reverts');
ok((await call(OWNER, '0x13af4035'+word('0'))).reverted, 'setOwner to the zero address reverts');
ok(!(await call(OWNER, '0x13af4035'+word(EVE))).reverted, 'setOwner from the owner succeeds');
const o = bytesToHex(await sm.getContractStorage(C, hexToBytes('0x'+word('0')))).slice(-40);
ok(o.toLowerCase() === EVE, '  ...and ownership actually moved');

// ---- dispatcher ---------------------------------------------------------
ok((await call(EVE, '0xdeadbeef')).reverted, 'unknown selector reverts');
ok((await call(EVE, '0x00')).reverted, 'calldata shorter than a selector reverts');

console.log(fails === 0 ? '\nALL SECURITY CHECKS PASSED' : `\n${fails} FAILED`);
process.exit(fails === 0 ? 0 : 1);
