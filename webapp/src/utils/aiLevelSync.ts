/**
 * Keeps the auto-match level and "highest AI level beaten" in sync between
 * this device (localStorage) and the signed-in account, so progress follows
 * the player across devices. Anonymous players stay localStorage-only.
 *
 * Merge rules:
 *  - highest level beaten: max of local and server (it never goes down)
 *  - current level: whichever side changed it most recently
 */

import {
  getAutoMatchLevel, setAutoMatchLevel, getTierSettings, LEVEL_UPDATED_AT_KEY, TIERS,
} from './autoMatch';
import { getAIProgress, putAIProgress, type AIProgress } from '../api/history';

const HIGHEST_BEATEN_KEY = 'knightball_ai_highest_beaten';

export interface LocalProgress {
  level: number;
  /** ISO time the level last changed on this device; null = never (default level). */
  levelUpdatedAt: string | null;
  highestBeaten: number;
}

export function readLocalProgress(): LocalProgress {
  let levelUpdatedAt: string | null = null;
  let highestBeaten = 0;
  try {
    levelUpdatedAt = localStorage.getItem(LEVEL_UPDATED_AT_KEY);
    const n = parseInt(localStorage.getItem(HIGHEST_BEATEN_KEY) ?? '0', 10);
    if (Number.isFinite(n) && n > 0) highestBeaten = n;
  } catch { /* ignore */ }
  return { level: getAutoMatchLevel(), levelUpdatedAt, highestBeaten };
}

/** Record a win against the AI at `level` (keeps the local peak). */
export function recordLevelBeaten(level: number): void {
  if (level <= readLocalProgress().highestBeaten) return;
  try { localStorage.setItem(HIGHEST_BEATEN_KEY, String(level)); } catch { /* ignore */ }
}

function time(iso: string | null): number {
  const t = iso ? Date.parse(iso) : NaN;
  return Number.isFinite(t) ? t : -Infinity;
}

/**
 * Merge local and server progress. `push` = the server is behind and should
 * be sent the merged record.
 */
export function mergeProgress(
  local: LocalProgress,
  server: AIProgress,
): { merged: LocalProgress; push: boolean } {
  const highestBeaten = Math.max(local.highestBeaten, server.highest_level_beaten || 0);
  const serverHasLevel = server.current_level != null;
  const localNewer = local.levelUpdatedAt != null
    && (!serverHasLevel || time(local.levelUpdatedAt) > time(server.current_level_updated_at));

  const merged: LocalProgress = localNewer || !serverHasLevel
    ? { level: local.level, levelUpdatedAt: local.levelUpdatedAt, highestBeaten }
    : { level: server.current_level!, levelUpdatedAt: server.current_level_updated_at, highestBeaten };

  const push = localNewer || highestBeaten > (server.highest_level_beaten || 0);
  return { merged, push };
}

function applyLocal(merged: LocalProgress, local: LocalProgress): void {
  if (merged.level !== local.level) setAutoMatchLevel(merged.level); // also resets the loss streak
  try {
    if (merged.levelUpdatedAt && merged.levelUpdatedAt !== localStorage.getItem(LEVEL_UPDATED_AT_KEY)) {
      localStorage.setItem(LEVEL_UPDATED_AT_KEY, merged.levelUpdatedAt);
    }
    if (merged.highestBeaten > local.highestBeaten) {
      localStorage.setItem(HIGHEST_BEATEN_KEY, String(merged.highestBeaten));
    }
  } catch { /* ignore */ }
}

let chain: Promise<unknown> = Promise.resolve();

/**
 * Pull the account's progress, merge it into localStorage, and push back
 * whatever the server is missing. Call only when signed in. Best-effort:
 * network/server errors leave local state untouched. Calls are serialized.
 * Resolves to the merged progress, or null if the sync failed.
 */
export function syncAIProgress(): Promise<LocalProgress | null> {
  const run = chain.then(async () => {
    try {
      const server = await getAIProgress();
      const local = readLocalProgress();
      const { merged, push } = mergeProgress(local, server);
      applyLocal(merged, local);
      if (push) {
        await putAIProgress({
          current_level: getAutoMatchLevel(),
          current_level_updated_at: merged.levelUpdatedAt,
          highest_level_beaten: merged.highestBeaten,
        });
      }
      return readLocalProgress();
    } catch {
      return null;
    }
  });
  chain = run;
  return run;
}

/**
 * The ladder level an AI opponent corresponds to, for game history: the
 * current auto-match level when it matches, else the first tier with the same
 * model + sims (presets are tiers), else undefined (custom settings).
 */
export function levelForSettings(
  model: string | undefined,
  sims: number,
  difficulty: string | undefined,
): number | undefined {
  if (!model) return undefined;
  if (difficulty === 'auto') {
    const level = getAutoMatchLevel();
    const tier = getTierSettings(level);
    if (tier.model === model && tier.sims === sims) return level;
  }
  const idx = TIERS.findIndex(t => t.model === model && t.sims === sims);
  return idx >= 0 ? idx + 1 : undefined;
}
