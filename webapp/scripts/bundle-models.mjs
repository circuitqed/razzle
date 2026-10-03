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
 * default when no level is selected).
 */

import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync, copyFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const MODELS = [
  { version: 'pegasus_iter_050', sha256: '564ea2f2ab70d95d49691c67baf18000b9fd68512a9a1aac1d30be494484fe5a' },
  { version: 'pegasus_iter_100', sha256: 'b7a527eff5b2d4a07eb8edb6140cf0e7de24d517c89a4ee31a216ad726838f76' },
  { version: 'pegasus_iter_150', sha256: '569ee2cbba80ceb32b368fa3579668eea7ee6731c688040cc319744df1c1e7c5' },
  { version: 'pegasus_iter_200', sha256: '259c9bbee223fe546f280d79193941a20f59d1fcc9c4c2862926a11e35ad8735' },
  { version: 'pegasus_iter_250', sha256: 'a5ca4d532fd6ce2950d3d5b00f4196c6f08ee868b7c066138365a27d6149fbba' },
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
