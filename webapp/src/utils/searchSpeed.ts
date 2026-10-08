/**
 * Measured AI search speed on this device, for "time per move" estimates.
 *
 * Every finished search records simulations/second for its (model, backend)
 * pair in localStorage (an exponential moving average, so a one-off slow
 * search after a tab switch fades out). Estimates for a model that has not
 * run here yet are scaled from a measured one by network cost
 * (filters² × blocks, parsed from names like distill_v2_96x12): only ever
 * scaled *down* in speed, because small networks are overhead-bound and
 * don't get proportionally faster.
 */

const STORAGE_KEY = 'knightball_search_speed';
/** Searches shorter than this are dominated by setup cost; don't record them. */
const MIN_SIMS_TO_RECORD = 32;
const EMA_WEIGHT = 0.3;

interface SpeedEntry { simsPerSec: number; samples: number }
type SpeedTable = Record<string, SpeedEntry>;

const modelKey = (model: string) => model.replace(/\.(pt|onnx)$/, '');
const entryKey = (model: string, backend: string) => `${modelKey(model)}|${backend}`;

function readTable(): SpeedTable {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

/** Relative evaluation cost of a network from its name, or null if unknown. */
export function networkCost(model: string): number | null {
  const m = /(\d+)x(\d+)/.exec(modelKey(model));
  if (!m) return null;
  const filters = parseInt(m[1], 10);
  const blocks = parseInt(m[2], 10);
  return filters * filters * blocks;
}

/** Record a finished search (called by useGame after every AI search). */
export function recordSearchSpeed(model: string, backend: string, simsDone: number, searchMs: number): void {
  if (!model || simsDone < MIN_SIMS_TO_RECORD || searchMs <= 0) return;
  const sps = (1000 * simsDone) / searchMs;
  const table = readTable();
  const key = entryKey(model, backend);
  const prev = table[key];
  table[key] = prev
    ? { simsPerSec: prev.simsPerSec * (1 - EMA_WEIGHT) + sps * EMA_WEIGHT, samples: prev.samples + 1 }
    : { simsPerSec: sps, samples: 1 };
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(table)); } catch { /* ignore */ }
}

/** Estimated simulations/second for `model` on this device, or null if nothing is known. */
export function estimateSimsPerSec(model: string): number | null {
  const table = readTable();
  const target = modelKey(model);
  const entries = Object.entries(table).map(([key, e]) => {
    const [m, backend] = key.split('|');
    return { model: m, backend, ...e };
  });
  if (entries.length === 0) return null;

  // Measured directly: the fastest backend seen (the app picks the best one available).
  const direct = entries.filter((e) => e.model === target);
  if (direct.length > 0) return Math.max(...direct.map((e) => e.simsPerSec));

  // Otherwise scale from the most-sampled measured model with a known cost.
  const targetCost = networkCost(target);
  if (targetCost == null) return null;
  const ref = entries
    .filter((e) => networkCost(e.model) != null)
    .sort((a, b) => b.samples - a.samples)[0];
  if (!ref) return null;
  const ratio = networkCost(ref.model)! / targetCost;
  return ref.simsPerSec * Math.min(1, ratio);
}

/** Estimated seconds per AI move at (model, sims), capped by `capMs` if given; null if unknown. */
export function estimateMoveSeconds(model: string, sims: number, capMs = 0): number | null {
  const sps = estimateSimsPerSec(model);
  if (sps == null || sps <= 0) return null;
  const secs = sims / sps;
  return capMs > 0 ? Math.min(secs, capMs / 1000) : secs;
}

/** Short human label for a duration in seconds: "8 s", "45 s", "2 min". */
export function formatSeconds(secs: number): string {
  if (secs < 10) return `${Math.max(1, Math.round(secs))} s`;
  if (secs < 90) return `${Math.round(secs / 5) * 5} s`;
  return `${Math.round(secs / 60)} min`;
}
