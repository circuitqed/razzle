/**
 * Auto-matching difficulty system for KnightBall.
 *
 * Players start at level 1 (very easy) and climb/drop one level per win/loss.
 * Each level maps to a specific model iteration + simulation count, providing
 * a smooth difficulty curve from random-ish play up to near-maximum strength.
 */

const STORAGE_KEY = 'knightball_ai_level';

export interface TierSettings {
  model: string;
  sims: number;
  label: string;
}

// Use 'latest' to always resolve to the best available model.
// Difficulty is controlled purely by simulation count until we have
// enough training iterations to differentiate by model strength.
export const TIERS: TierSettings[] = [
  { model: 'latest', sims: 1,     label: 'Level 1 — Beginner' },
  { model: 'latest', sims: 4,     label: 'Level 2 — Beginner' },
  { model: 'latest', sims: 16,    label: 'Level 3 — Beginner' },
  { model: 'latest', sims: 32,    label: 'Level 4 — Easy' },
  { model: 'latest', sims: 64,    label: 'Level 5 — Easy' },
  { model: 'latest', sims: 128,   label: 'Level 6 — Intermediate' },
  { model: 'latest', sims: 256,   label: 'Level 7 — Intermediate' },
  { model: 'latest', sims: 512,   label: 'Level 8 — Medium' },
  { model: 'latest', sims: 1024,  label: 'Level 9 — Medium' },
  { model: 'latest', sims: 2048,  label: 'Level 10 — Hard' },
  { model: 'latest', sims: 4096,  label: 'Level 11 — Hard' },
  { model: 'latest', sims: 8192,  label: 'Level 12 — Expert' },
  { model: 'latest', sims: 16384, label: 'Level 13 — Expert' },
  { model: 'latest', sims: 32768, label: 'Level 14 — Master' },
  { model: 'latest', sims: 65536, label: 'Level 15 — Master' },
];

export const MAX_LEVEL = TIERS.length;

/** Read the current auto-match level from localStorage (1-indexed, default 1). */
export function getAutoMatchLevel(): number {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const n = parseInt(raw, 10);
      if (n >= 1 && n <= MAX_LEVEL) return n;
    }
  } catch { /* ignore */ }
  return 1;
}

/** Persist the auto-match level to localStorage. */
export function setAutoMatchLevel(level: number): void {
  const clamped = Math.max(1, Math.min(MAX_LEVEL, level));
  try {
    localStorage.setItem(STORAGE_KEY, String(clamped));
  } catch { /* ignore */ }
}

/**
 * Adjust level after a game result.
 * Win => level + 1, Loss => level - 1, clamped to [1, MAX_LEVEL].
 * Returns the new level.
 */
export function adjustAfterGame(won: boolean): number {
  const current = getAutoMatchLevel();
  const next = won ? current + 1 : current - 1;
  const clamped = Math.max(1, Math.min(MAX_LEVEL, next));
  setAutoMatchLevel(clamped);
  return clamped;
}

/** Get the model + sims config for a given level (1-indexed). */
export function getTierSettings(level: number): TierSettings {
  const idx = Math.max(0, Math.min(TIERS.length - 1, level - 1));
  return TIERS[idx];
}

/** Get the display label for a level, e.g. "Level 3 — Beginner". */
export function getLevelLabel(level: number): string {
  return getTierSettings(level).label;
}
