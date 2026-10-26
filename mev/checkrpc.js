// Verifies RPC_URL and WS_URL without ever printing the key.
const { ethers } = require('ethers');
const WebSocket = require('ws');

// Read mev/.env directly rather than relying on the shell.
//
// This used to need RPC_URL=... WS_URL=... prefixed on the command line with
// backslash continuations, and zsh ate the `node` off the end of it. A long
// quoted URL in front of a command is a paste waiting to go wrong -- so the
// values are read from the file they already live in, and the command is just
// `node checkrpc.js` with nothing in front of it.
function loadEnv() {
  // Bootstrap copy of src/env.ts. This file must run before `npm run build`,
  // so it cannot import the compiled version. Keep the two in step.
  const fs = require('fs');
  const path = require('path');
  const file = path.join(__dirname, '.env');
  if (!fs.existsSync(file)) { console.log('NOTE  no .env found next to this script'); return; }
  let text = fs.readFileSync(file, 'utf8');
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
  if (text.startsWith('{\\rtf')) {
    console.log('FAIL  .env is RTF, not plain text — TextEdit saved it in rich-text mode.');
    console.log('      Rewrite it from the terminal, or use Format > Make Plain Text.');
    process.exit(1);
  }
  const SMART = /[\u201C\u201D\u2018\u2019]/g;
  for (const raw of text.split('\n')) {
    const line = raw.replace(/\r$/, '').trim();
    if (!line || line.startsWith('#')) continue;
    const eq = line.indexOf('=');
    if (eq < 1) continue;
    const key = line.slice(0, eq).trim();
    let val = line.slice(eq + 1).trim();
    if (SMART.test(val)) {
      console.log(`NOTE  ${key} had curly quotes (TextEdit smart quotes) — normalised`);
      val = val.replace(SMART, '"');
    }
    if (val.length > 1 && (val[0] === '"' || val[0] === "'") && val[val.length - 1] === val[0]) {
      val = val.slice(1, -1);
    }
    if (val === '') { console.log(`NOTE  ${key} is empty in .env`); continue; }
    if (!(key in process.env)) process.env[key] = val;
  }
}
loadEnv();

const rpc = process.env.RPC_URL || '';
const ws  = process.env.WS_URL  || '';
const mask = (u) => { try { const x = new URL(u); return `${x.protocol}//${x.host}/…`; }
                      catch { return '(unparseable)'; } };

(async () => {
  let bad = 0;

  if (!rpc.startsWith('https://')) { console.log('FAIL  RPC_URL must start with https://'); bad++; }
  else {
    try {
      const p = new ethers.JsonRpcProvider(rpc, 1, { staticNetwork: true });
      const t0 = Date.now();
      const n = await p.getBlockNumber();
      const ms = Date.now() - t0;
      const b = await p.getBlock('latest');
      console.log(`PASS  RPC ${mask(rpc)} block ${n} in ${ms}ms`);
      console.log(`      base fee ${ethers.formatUnits(b?.baseFeePerGas ?? 0n, 'gwei')} gwei`);
      if (ms > 400) console.log(`WARN  ${ms}ms round trip — slow for this use`);
      p.destroy();
    } catch (e) { console.log(`FAIL  RPC ${mask(rpc)}: ${e.shortMessage || e.message}`); bad++; }
  }

  if (!ws.startsWith('wss://')) { console.log('FAIL  WS_URL must start with wss://'); bad++; }
  else {
    const got = await new Promise((res) => {
      const s = new WebSocket(ws, { handshakeTimeout: 8000 });
      const done = (v) => { try { s.close(); } catch {} res(v); };
      const timer = setTimeout(() => done('no newHeads within 20s'), 20000);
      s.on('open', () => s.send(JSON.stringify(
        { jsonrpc: '2.0', id: 1, method: 'eth_subscribe', params: ['newHeads'] })));
      s.on('message', (m) => {
        const d = JSON.parse(m.toString());
        if (d.error) { clearTimeout(timer); done(`rejected: ${d.error.message}`); }
        if (d.method === 'eth_subscription') { clearTimeout(timer); done(true); }
      });
      s.on('error', (e) => { clearTimeout(timer); done(e.message); });
    });
    if (got === true) console.log(`PASS  WS ${mask(ws)} streaming newHeads`);
    else { console.log(`FAIL  WS ${mask(ws)}: ${got}`); bad++; }
  }

  // The enhanced pending-tx feed is the one that decides whether the
  // mempool trigger does anything at all.
  if (ws.startsWith('wss://')) {
    const sub = await new Promise((res) => {
      const s = new WebSocket(ws, { handshakeTimeout: 8000 });
      const done = (v) => { try { s.close(); } catch {} res(v); };
      const timer = setTimeout(() => done('timeout'), 12000);
      s.on('open', () => s.send(JSON.stringify(
        { jsonrpc: '2.0', id: 1, method: 'eth_subscribe', params: ['alchemy_pendingTransactions'] })));
      s.on('message', (m) => { const d = JSON.parse(m.toString());
        if (d.id === 1) { clearTimeout(timer); done(d.error ? d.error.message : 'ok'); } });
      s.on('error', (e) => { clearTimeout(timer); done(e.message); });
    });
    console.log(sub === 'ok'
      ? 'PASS  full pending-tx bodies available — mempool trigger will work'
      : `NOTE  no alchemy_pendingTransactions (${sub}) — runs on block cadence only`);
  }

  console.log(bad === 0 ? '\nendpoint is usable' : `\n${bad} problem(s)`);
  process.exit(bad === 0 ? 0 : 1);
})();
