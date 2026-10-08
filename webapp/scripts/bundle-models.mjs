#!/usr/bin/env node
/**
 * Copy the AI models used by the difficulty levels into dist/bundled-models/
 * so the native app plays every level offline (see src/engine/bundledModels.ts).
 *
 * Run after `vite build` — `npm run build:ios` does both. Models are taken
 * from, in order: $BUNDLED_MODELS_DIR, ../engine/output/models (engine host),
 * or .model-cache/ (downloaded from knightball.org on first use). Every file
 * is checksum-verified, so a stale or corrupt model fails the build instead
 * of shipping an AI that plays badly.
 *
 * To change the bundled set: keep in sync with the models in
 * src/utils/autoMatch.ts, ordered weakest → strongest (the last one is the
 * default when no level is selected). distill_* are the distilled students
 * (engine/scripts/distill/), served by the API like any other model.
 */

import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync, copyFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const MODELS = [
  { version: 'distill_v2_16x2', sha256: '53f56803d504868b1c3b929bc034dbb6a59d4e5d35b4a484784e9d896f93c439' },
  { version: 'distill_v2_24x3', sha256: '77521ea3f8d19bd83f6a902d657152bf0abbb8470e39064d867930fd2a2ec6f7' },
  { version: 'distill_v2_32x4', sha256: 'b791f756c852e55b691775962ceb81bb8748c4b53ab1d5bce74823d38209f632' },
  { version: 'distill_v2_48x6', sha256: '2591c72bbbd92dcc465b7f76526834cdc5cd8d090d83b6e4dfd306acaccb2b54' },
  { version: 'distill_v2_64x8', sha256: '876f761a3f88c0f2c3fcd1df0645c0b8ba72b23e33f391232e4ad84c4d476627' },
  { version: 'distill_v2_96x12', sha256: '22a5da34b981af85f9d059cb2eb257a7b0001827154b07bdedc953fa0e3ed753' },
];

const DOWNLOAD_BASE = 'https://knightball.org/api/models/onnx';

const webappDir = join(dirname(fileURLToPath(import.meta.url)), '..');
const outDir = join(webappDir, 'dist', 'bundled-models');
const cacheDir = join(webappDir, '.model-cache');
const sourceDirs = [
  process.env.BUNDLED_MODELS_DIR,
  join(webappDir, '..', 'engine', 'output', 'models'),
  cacheDir,
].filter(Boolean);

const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');

async function obtain({ version, sha256: expected }) {
  const file = `${version}.onnx`;
  for (const dir of sourceDirs) {
    const path = join(dir, file);
    if (existsSync(path) && sha256(readFileSync(path)) === expected) return path;
  }
  console.log(`  downloading ${file}...`);
  const resp = await fetch(`${DOWNLOAD_BASE}/${file}`);
  if (!resp.ok) throw new Error(`download ${file}: HTTP ${resp.status}`);
  const buf = Buffer.from(await resp.arrayBuffer());
  const actual = sha256(buf);
  if (actual !== expected) throw new Error(`${file}: checksum mismatch (got ${actual})`);
  mkdirSync(cacheDir, { recursive: true });
  const path = join(cacheDir, file);
  writeFileSync(path, buf);
  return path;
}

if (!existsSync(join(webappDir, 'dist', 'index.html'))) {
  console.error('dist/ not found — run `vite build` first (or use `npm run build:ios`).');
  process.exit(1);
}

mkdirSync(outDir, { recursive: true });
const manifest = { models: [] };
for (const model of MODELS) {
  const src = await obtain(model);
  const file = `${model.version}.onnx`;
  copyFileSync(src, join(outDir, file));
  const size = readFileSync(src).length;
  manifest.models.push({ version: model.version, file, size_bytes: size });
  console.log(`  bundled ${file} (${(size / 1e6).toFixed(1)} MB)`);
}
writeFileSync(join(outDir, 'manifest.json'), JSON.stringify(manifest, null, 2));
console.log(`Bundled ${MODELS.length} models into dist/bundled-models/`);
