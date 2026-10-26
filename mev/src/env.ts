/**
 * .env loader that refuses to be quietly wrong.
 *
 * WHY THIS IS NOT ONE LINE OF `split('=')`.
 *
 * On macOS the obvious way to edit this file is `open -e .env`, which is
 * TextEdit, and TextEdit does two things that silently destroy an env file:
 *
 *   SMART QUOTES. It rewrites "  as  U+201C and U+201D. The file still looks
 *   correct in any editor that renders them, the key is still visibly there,
 *   and `grep -c YOUR_KEY` still reports 0 — but the parsed value now begins
 *   with a curly quote, so every URL check fails and the reported error points
 *   at the URL rather than at the quotes around it.
 *
 *   RTF. If TextEdit is in rich-text mode, Save writes `{\rtf1\ansi...` into a
 *   file still named `.env`. Nothing about the filename or the editor's own
 *   display gives this away.
 *
 * Both were reproduced against the real failure, and both produce the same
 * unhelpful message from a naive parser: "RPC_URL must start with https://".
 * That message sends you to check your URL, which is correct, which is the
 * worst possible place to be sent.
 *
 * So: curly quotes are normalised, RTF is DETECTED AND REPORTED rather than
 * tolerated — a file that is secretly RTF has other problems and silently
 * half-reading it is worse than refusing it — and anything that parses to an
 * empty value is named explicitly.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';

export interface EnvLoadResult {
  /** Absolute path we looked at. */
  readonly file: string;
  readonly found: boolean;
  /** Names set from the file. Values are never included — they are secrets. */
  readonly loaded: readonly string[];
  /** Human-readable problems, safe to print. Never contains a value. */
  readonly problems: readonly string[];
}

const SMART_QUOTES = /[“”‘’]/g;

export function loadEnvFile(dir: string = process.cwd()): EnvLoadResult {
  const file = path.resolve(dir, '.env');
  const loaded: string[] = [];
  const problems: string[] = [];

  if (!fs.existsSync(file)) {
    return { file, found: false, loaded, problems };
  }

  let text = fs.readFileSync(file, 'utf8');

  // A UTF-8 BOM makes the first key literally "﻿RPC_URL", which never
  // matches anything and is invisible in every editor.
  if (text.charCodeAt(0) === 0xfeff) {
    text = text.slice(1);
    problems.push('stripped a UTF-8 byte-order mark from the start of the file');
  }

  if (text.startsWith('{\\rtf')) {
    problems.push(
      'this file is RTF, not plain text — TextEdit saved it in rich-text mode. ' +
      'Rewrite it from the terminal, or in TextEdit use Format > Make Plain Text ' +
      'before saving.');
    return { file, found: true, loaded, problems };
  }

  for (const raw of text.split('\n')) {
    const line = raw.replace(/\r$/, '').trim();
    if (!line || line.startsWith('#')) continue;

    const eq = line.indexOf('=');
    if (eq < 1) continue;

    const key = line.slice(0, eq).trim();
    let val = line.slice(eq + 1).trim();

    if (SMART_QUOTES.test(val)) {
      problems.push(
        `${key}: curly quotes were converted to straight ones — TextEdit's ` +
        `smart quotes. Turn them off under Edit > Substitutions, or edit this ` +
        `file from the terminal.`);
      val = val.replace(SMART_QUOTES, '"');
    }

    // Strip exactly one matching surrounding pair, after the curly-quote fix
    // so that a smart-quoted value is unwrapped too.
    if (val.length > 1 && (val[0] === '"' || val[0] === "'") && val[val.length - 1] === val[0]) {
      val = val.slice(1, -1);
    }

    if (val === '') {
      problems.push(`${key} is empty`);
      continue;
    }

    // A real environment variable always wins. Railway sets them directly and
    // must never be overridden by a file that happened to ship in the image.
    if (!(key in process.env)) {
      process.env[key] = val;
      loaded.push(key);
    }
  }

  return { file, found: true, loaded, problems };
}

/** Load, and print anything worth knowing. Never prints a value. */
export function loadEnvAndReport(dir: string = process.cwd()): EnvLoadResult {
  const r = loadEnvFile(dir);
  if (r.found && r.loaded.length > 0) {
    console.log(`loaded from .env    : ${r.loaded.join(', ')}`);
  }
  for (const p of r.problems) console.warn(`.env: ${p}`);
  return r;
}
