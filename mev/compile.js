/**
 * Compile contracts/MevArb.yul and produce deployable bytecode.
 *
 * Needs solc >= 0.8.28. Earlier versions ANALYSE Yul through standard JSON
 * but emit no bytecode at all -- zero errors and an empty contracts object,
 * which looks exactly like success. 0.8.26 does this; do not pin it back.
 *
 *   node compile.js [aavePoolAddress]
 *
 * Writes MevArb.deploy.hex: creation bytecode with the pool address appended
 * as a left-padded word, which is where the constructor reads it from.
 */
const fs = require('fs');
const path = require('path');
const solc = require('solc');

const AAVE_V3_POOL_MAINNET = '0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2';
const pool = (process.argv[2] || process.env.AAVE_POOL || AAVE_V3_POOL_MAINNET)
  .trim().toLowerCase().replace(/^0x/, '');

if (!/^[0-9a-f]{40}$/.test(pool)) {
  console.error(`not a 20-byte address: ${process.argv[2]}`);
  process.exit(1);
}

const [maj, min, pat] = solc.version().split('+')[0].split('.').map(Number);
if (maj === 0 && (min < 8 || (min === 8 && pat < 28))) {
  console.error(`solc ${solc.version().split('+')[0]} emits no Yul bytecode. Need >= 0.8.28.`);
  process.exit(1);
}

const file = path.join(__dirname, 'contracts', 'MevArb.yul');
const out = JSON.parse(solc.compile(JSON.stringify({
  language: 'Yul',
  sources: { 'MevArb.yul': { content: fs.readFileSync(file, 'utf8') } },
  settings: {
    optimizer: { enabled: true, runs: 4294967295, details: { yul: true } },
    evmVersion: 'cancun',
    outputSelection: { '*': { '*': ['*'] } },
  },
})));

const errors = (out.errors || []).filter((e) => e.severity === 'error');
for (const e of (out.errors || [])) {
  if (e.severity === 'error') console.error(e.formattedMessage);
  else console.warn(`warning: ${e.message.split('\n')[0].slice(0, 140)}`);
}
if (errors.length) { console.error(`\n${errors.length} error(s)`); process.exit(1); }

const c = out.contracts?.['MevArb.yul']?.MevArb;
const code = c?.evm?.bytecode?.object;
if (!code) {
  console.error('compiled with no errors but emitted no bytecode — check the solc version');
  process.exit(1);
}

const deploy = '0x' + code + pool.padStart(64, '0');
const dest = path.join(__dirname, 'MevArb.deploy.hex');
fs.writeFileSync(dest, deploy + '\n');

console.log(`solc              : ${solc.version().split('+')[0]}`);
console.log(`creation bytecode : ${code.length / 2} bytes`);
console.log(`aave v3 pool      : 0x${pool}`);
console.log(`deployable        : ${deploy.length / 2 - 1} bytes -> ${path.basename(dest)}`);
console.log(`\nverify it first   : node testcontract.mjs`);
console.log(`then deploy with  : cast send --create "$(cat MevArb.deploy.hex)" --private-key $EXECUTION_KEY --rpc-url "$RPC_URL"`);
