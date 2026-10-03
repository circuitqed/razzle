/**
 * AI models shipped inside the native app bundle, so every difficulty level
 * works offline with no first-run download.
 *
 * `npm run build:ios` (scripts/bundle-models.mjs) copies the ONNX files plus a
 * manifest.json into dist/bundled-models/. Web builds don't include them; the
 * manifest fetch then 404s and callers fall back to the server.
 */

import { isNativeApp } from '../api/base';
import { getCachedModel } from './modelCache';
import { getOnnxModelInfo, getOnnxModelInfoByName, type OnnxModelInfo } from '../api/engine';

export const BUNDLED_MODELS_PATH = 'bundled-models';

interface BundledManifest {
  models: { version: string; file: string; size_bytes: number }[];
}

let manifestPromise: Promise<BundledManifest | null> | null = null;

function loadManifest(): Promise<BundledManifest | null> {
  if (!manifestPromise) {
    manifestPromise = fetch(`/${BUNDLED_MODELS_PATH}/manifest.json`)
      .then(r => (r.ok ? r.json() : null))
      .catch(() => null);
  }
  return manifestPromise;
}

/** Info for a bundled model by version (e.g. "pegasus_iter_250"), or null. */
export async function getBundledModelInfo(version: string): Promise<OnnxModelInfo | null> {
  if (!isNativeApp) return null;
  const manifest = await loadManifest();
  const entry = manifest?.models.find(m => m.version === version);
  if (!entry) return null;
  return {
    version: entry.version,
    // Absolute URL: the worker resolves relative URLs against its own script path.
    url: new URL(`/${BUNDLED_MODELS_PATH}/${entry.file}`, globalThis.location.href).href,
    size_bytes: entry.size_bytes,
  };
}

/** The strongest bundled model (last in the manifest), used when no level is chosen. */
async function getDefaultBundledModelInfo(): Promise<OnnxModelInfo | null> {
  if (!isNativeApp) return null;
  const manifest = await loadManifest();
  const last = manifest?.models[manifest.models.length - 1];
  return last ? getBundledModelInfo(last.version) : null;
}

/**
 * Resolve where to load a model from, preferring sources that work offline:
 *   1. bundled in the native app
 *   2. the server (also triggers on-demand ONNX export)
 *   3. a copy already in the IndexedDB cache (offline web/PWA)
 *
 * `filename` is a .pt name like "pegasus_iter_250.pt", or null for the
 * server's latest model.
 */
export async function resolveModelInfo(filename: string | null): Promise<OnnxModelInfo> {
  const version = filename?.replace(/\.pt$/, '') ?? null;

  const bundled = version ? await getBundledModelInfo(version) : await getDefaultBundledModelInfo();
  if (bundled) return bundled;

  try {
    return filename ? await getOnnxModelInfoByName(filename) : await getOnnxModelInfo();
  } catch (err) {
    // The worker checks the IndexedDB cache by version before fetching, so an
    // empty URL is fine when the model is already cached.
    if (version && (await getCachedModel(version))) {
      return { version, url: '', size_bytes: 0 };
    }
    throw err;
  }
}
