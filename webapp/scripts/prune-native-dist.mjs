#!/usr/bin/env node
/**
 * Trim dist/ for release native (iOS) builds:
 *
 * - Developer test pages and fixtures, so an App Store reviewer (or a
 *   universal link) can't land on them. Kept when KB_TEST_PAGES=1 — the
 *   on-device suite (npm run test:ios) needs them.
 * - ONNX Runtime WASM binaries (~100 MB). The native app always runs the
 *   custom WebGL / pure-TS inference path (ai.worker.ts routes capacitor:
 *   origins there), so they're never loaded.
 */

import { rmSync, existsSync, readdirSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const dist = join(dirname(fileURLToPath(import.meta.url)), '..', 'dist');
const keepTestPages = process.env.KB_TEST_PAGES === '1';

const TEST_FILES = [
  'test-mcts.html',
  'test-native.html',
  'test-chrome.html',
  'test-webgl-inference.html',
  'inference-fixtures.json',
  'inference-fixtures-v2.json',
];

let freed = 0;
function remove(rel) {
  const p = join(dist, rel);
  if (!existsSync(p)) return;
  freed += statSync(p).size;
  rmSync(p);
  console.log(`  removed ${rel}`);
}

if (keepTestPages) {
  console.log('KB_TEST_PAGES=1 — keeping test pages');
} else {
  TEST_FILES.forEach(remove);
}

for (const dir of ['.', 'assets']) {
  for (const f of readdirSync(join(dist, dir))) {
    if (/^ort-wasm.*\.(wasm|mjs)$/.test(f)) remove(join(dir, f));
  }
}

console.log(`Pruned ${(freed / 1e6).toFixed(1)} MB from dist/`);
