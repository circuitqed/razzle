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
  { version: 'pegasus_iter_050', sha256: '564ea2f2ab70d95d49691c67baf18000b9fd68512a9a1aac1d30be494484fe5a' },
  { version: 'distill_s32x4', sha256: '265b8fb515e7f27b091c74079c757db9432e5898b2d690fa01cf188fc1d5839d' },
  { version: 'distill_s48x6', sha256: 'ebd5b2bcb29d9b48692b20e7739afa49bb9da1a2cb6dc4daa716a515626b9652' },
  { version: 'distill_s64x8', sha256: '50dfa5d2adcfd0b8f57a10d3e087f5889af51956b4f9a3c159b9f4a4283e651f' },
  { version: 'distill_s96x12', sha256: '10586464ebe90e14119fb5b521c977d2f42b4340dcb4c4dccb4b14048423a309' },
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
