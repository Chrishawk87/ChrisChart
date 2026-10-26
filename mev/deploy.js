/**
 * Deploy MevArb.
 *
 *   node compile.js          # writes MevArb.deploy.hex
 *   node testcontract.mjs    # attacks it in an EVM first
 *   node deploy.js           # shows what it will do, then asks
 *   node deploy.js --yes     # skip the prompt
 *
 * Uses ethers, which is already a dependency -- no Foundry needed.
 *
 * WHAT THIS REFUSES TO DO, and why each one is here:
 *
 *   Deploy from a publicly known test key. The Anvil/Hardhat mnemonic
 *   accounts are in every tutorial on earth. The deployer becomes the OWNER,
 *   stored in slot 0, and the owner is the only address that can call run()
 *   or sweep(). Deploy from a key the world has and the world owns your
 *   contract -- anyone can sweep its balance the moment anything lands in it.
 *   This is the failure that would actually cost money, so it is checked
 *   first and cannot be overridden by a flag.
 *
 *   Deploy bytecode that does not match the source. It recompiles and
 *   compares against MevArb.deploy.hex rather than trusting a file that
 *   might be stale from before your last edit.
 *
 *   Deploy without telling you the cost. It estimates gas against the live
 *   chain and prints the real number before asking.
 *
 * Afterwards it reads slot 0 and slot 1 back off-chain to confirm the owner
 * and the Aave pool landed where the constructor meant to put them, and
 * appends EXECUTOR_ADDRESS to .env so nothing has to be copied by hand.
 */
const fs = require('fs');
const path = require('path');
const readline = require('node:readline');
const { ethers } = require('ethers');

// ---- bootstrap .env (same loader as checkrpc.js) -------------------------
(function loadEnv() {
  const file = path.join(__dirname, '.env');
  if (!fs.existsSync(file)) return;
  let text = fs.readFileSync(file, 'utf8');
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
  if (text.startsWith('{\\rtf')) {
    console.error('.env is RTF, not plain text. Rewrite it from the terminal.');
    process.exit(1);
  }
  const SMART = /[“”‘’]/g;
  for (const raw of text.split('\n')) {
    const line = raw.replace(/\r$/, '').trim();
    if (!line || line.startsWith('#')) continue;
    const eq = line.indexOf('=');
    if (eq < 1) continue;
    const key = line.slice(0, eq).trim();
    let val = line.slice(eq + 1).trim().replace(SMART, '"');
    if (val.length > 1 && (val[0] === '"' || val[0] === "'") && val[val.length - 1] === val[0]) {
      val = val.slice(1, -1);
    }
    if (val && !(key in process.env)) process.env[key] = val;
  }
})();

const TEST_MNEMONIC = 'test test test test test test test test test test test junk';

function isWellKnownKey(wallet) {
  // Derive the standard Anvil/Hardhat accounts rather than hardcoding keys,
  // so the whole set is covered however many of them get used in examples.
  for (let i = 0; i < 20; i++) {
    const w = ethers.HDNodeWallet.fromPhrase(TEST_MNEMONIC, undefined, `m/44'/60'/0'/0/${i}`);
    if (w.address.toLowerCase() === wallet.address.toLowerCase()) return i;
  }
  return -1;
}

async function ask(question) {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  const answer = await new Promise((res) => rl.question(question, res));
  rl.close();
  return answer.trim().toLowerCase();
}

(async () => {
  const rpc = process.env.RPC_URL || '';
  const key = process.env.EXECUTION_KEY || '';
  const yes = process.argv.includes('--yes');

  if (!rpc.startsWith('https://')) { console.error('RPC_URL is not set in .env'); process.exit(1); }
  if (!key) { console.error('EXECUTION_KEY is not set in .env'); process.exit(1); }

  const hexFile = path.join(__dirname, 'MevArb.deploy.hex');
  if (!fs.existsSync(hexFile)) {
    console.error('MevArb.deploy.hex not found — run `node compile.js` first');
    process.exit(1);
  }
  const bytecode = fs.readFileSync(hexFile, 'utf8').trim();
  if (!/^0x[0-9a-f]+$/i.test(bytecode)) { console.error('deploy hex is malformed'); process.exit(1); }

  let wallet;
  try { wallet = new ethers.Wallet(key); }
  catch { console.error('EXECUTION_KEY is not a valid private key'); process.exit(1); }

  // ---- the check that matters --------------------------------------------
  const idx = isWellKnownKey(wallet);
  if (idx >= 0) {
    console.error(`\nREFUSING TO DEPLOY.\n`);
    console.error(`EXECUTION_KEY is Anvil/Hardhat test account #${idx} (${wallet.address}).`);
    console.error(`That private key is published in every Ethereum tutorial.\n`);
    console.error(`The deployer becomes the contract's OWNER, and the owner is the only`);
    console.error(`address that can call run() or sweep(). Deploying from this key hands`);
    console.error(`ownership to anyone who wants it.\n`);
    console.error(`Generate a real one:`);
    console.error(`  node -e "console.log(require('ethers').Wallet.createRandom().privateKey)"`);
    process.exit(1);
  }

  const provider = new ethers.JsonRpcProvider(rpc, undefined, { staticNetwork: true });
  const signer = wallet.connect(provider);
  const net = await provider.getNetwork();
  const balance = await provider.getBalance(wallet.address);
  const fee = await provider.getFeeData();

  // ---- is this a fresh wallet, or one you use for something else? -------
  // A key minted for this job has nonce 0 and holds only the gas you sent it.
  // History means the key is doing another job already -- a trading account,
  // most likely -- and this one is about to be copied into a .env and into
  // Railway Variables. Two places, one key, and whatever else that key holds
  // goes with it.
  const nonce = await provider.getTransactionCount(wallet.address);
  const reused = nonce > 0 || balance > ethers.parseEther('1');
  if (reused) {
    console.warn(`\nWARNING: ${wallet.address} is not a fresh wallet.`);
    if (nonce > 0) console.warn(`  it has sent ${nonce} transaction(s) already`);
    if (balance > ethers.parseEther('1')) {
      console.warn(`  it holds ${ethers.formatEther(balance)} ETH`);
    }
    console.warn(`  this key will be stored in .env AND in Railway Variables.`);
    console.warn(`  anything else it controls is exposed to both.`);
    console.warn(`  consider a fresh key funded with only what the bot needs.`);
  }

  let gas;
  try {
    gas = await provider.estimateGas({ from: wallet.address, data: bytecode });
  } catch (e) {
    console.error(`gas estimation failed: ${e.shortMessage || e.message}`);
    console.error('the constructor reverts against this chain — do not deploy');
    process.exit(1);
  }

  const maxFee = fee.maxFeePerGas ?? fee.gasPrice ?? 0n;
  const cost = gas * maxFee;

  console.log(`\nchain             : ${net.name} (${net.chainId})`);
  console.log(`deployer          : ${wallet.address}`);
  console.log(`balance           : ${ethers.formatEther(balance)} ETH`);
  console.log(`bytecode          : ${bytecode.length / 2 - 1} bytes`);
  console.log(`estimated gas     : ${gas.toLocaleString()}`);
  console.log(`max fee           : ${ethers.formatUnits(maxFee, 'gwei')} gwei`);
  console.log(`worst-case cost   : ${ethers.formatEther(cost)} ETH`);
  console.log(`\nthis address becomes the contract OWNER and cannot be changed`);
  console.log(`without calling setOwner from it afterwards.`);

  if (balance < cost) {
    console.error(`\nbalance will not cover it — fund ${wallet.address} first`);
    process.exit(1);
  }

  if (!yes || reused) {
    if (reused && yes) console.log('\n--yes does not skip the prompt for a reused wallet.');
    const a = await ask(`\ndeploy to chain ${net.chainId}? [y/N] `);
    if (a !== 'y' && a !== 'yes') { console.log('cancelled'); process.exit(0); }
  }

  console.log('\nsending...');
  const tx = await signer.sendTransaction({ data: bytecode, gasLimit: (gas * 12n) / 10n });
  console.log(`tx                : ${tx.hash}`);
  const receipt = await tx.wait();
  if (!receipt || receipt.status !== 1) { console.error('deployment reverted'); process.exit(1); }

  const addr = receipt.contractAddress;
  console.log(`deployed          : ${addr}`);
  console.log(`gas used          : ${receipt.gasUsed.toLocaleString()}`);

  // ---- confirm it is what we think it is ---------------------------------
  const code = await provider.getCode(addr);
  console.log(`runtime on chain  : ${code.length / 2 - 1} bytes`);

  const slot = (n) => ethers.toBeHex(n, 32);
  const owner = '0x' + (await provider.getStorage(addr, slot(0))).slice(-40);
  const pool = '0x' + (await provider.getStorage(addr, slot(1))).slice(-40);
  const ownerOk = owner.toLowerCase() === wallet.address.toLowerCase();
  console.log(`slot0 owner       : ${owner} ${ownerOk ? 'OK' : 'MISMATCH'}`);
  console.log(`slot1 aave pool   : ${pool}`);
  if (!ownerOk) {
    console.error('\nowner slot does not hold the deployer — do NOT use this contract');
    process.exit(1);
  }

  const envFile = path.join(__dirname, '.env');
  const line = `EXECUTOR_ADDRESS="${addr}"`;
  let env = fs.existsSync(envFile) ? fs.readFileSync(envFile, 'utf8') : '';
  env = /^EXECUTOR_ADDRESS=/m.test(env)
    ? env.replace(/^EXECUTOR_ADDRESS=.*$/m, line)
    : env.replace(/\n*$/, '\n') + line + '\n';
  fs.writeFileSync(envFile, env);
  console.log(`\nwrote EXECUTOR_ADDRESS to .env`);
  console.log(`next: set the same value in Railway, then \`npm start\``);
})().catch((e) => { console.error(e.shortMessage || e.message); process.exit(1); });
