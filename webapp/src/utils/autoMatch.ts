/**
 * Auto-matching difficulty system for KnightBall.
 *
 * Players start at level 1 (very easy). A win promotes one level immediately;
 * a demotion requires LOSS_STREAK_TO_DEMOTE consecutive losses (hysteresis —
 * without it, a player at their skill boundary ping-pongs between two levels
 * every other game, which feels bad). Each level maps to a specific model
 * (model, simulation count) pair calibrated in arena games, providing a smooth
 * difficulty curve from random-ish play up to near-maximum strength.
 */

const STORAGE_KEY = 'knightball_ai_level';
const LOSS_STREAK_KEY = 'knightball_ai_loss_streak';
/** ISO time the level last changed on this device (for cross-device sync). */
export const LEVEL_UPDATED_AT_KEY = 'knightball_ai_level_updated_at';

/** Consecutive losses at a level required before dropping down. */
export const LOSS_STREAK_TO_DEMOTE = 2;

export interface TierSettings {
  model: string;
  sims: number;
  label: string;
  /** Approximate displayed rating (see below). */
  rating: number;
}

// Calibrated ladder (Oct 2026). Each (model, sims) pair was placed on one Elo
// scale with ~20k arena games (engine/scripts/distill/calibration.py); steps
// are ~45-55 points at the bottom (new players) and ~70-90 higher up.
// Ratings are shown on a human-friendly scale anchored so that ~1500 is an
// even game for a ~1500 chess player (displayed = 880 + calibrated Elo).
// Levels marked "interp." are interpolated in log(sims) between measured
// neighbours. Models: pegasus_iter_050 (weak, for beginners) and the distilled
// students distill_s{filters}x{blocks} (fast enough for phones).
export const TIERS: TierSettings[] = [
  { model: 'pegasus_iter_050.pt', sims: 1,    rating: 880,  label: 'Level 1 — Beginner' },
  { model: 'pegasus_iter_050.pt', sims: 4,    rating: 945,  label: 'Level 2 — Beginner' },
  { model: 'pegasus_iter_050.pt', sims: 12,   rating: 990,  label: 'Level 3 — Beginner' },
  { model: 'pegasus_iter_050.pt', sims: 24,   rating: 1030, label: 'Level 4 — Beginner' },
  { model: 'pegasus_iter_050.pt', sims: 32,   rating: 1085, label: 'Level 5 — Easy' },
  { model: 'distill_s32x4.pt',    sims: 2,    rating: 1140, label: 'Level 6 — Easy' },
  { model: 'distill_s32x4.pt',    sims: 16,   rating: 1235, label: 'Level 7 — Easy' },
  { model: 'distill_s32x4.pt',    sims: 32,   rating: 1315, label: 'Level 8 — Intermediate' },
  { model: 'distill_s48x6.pt',    sims: 16,   rating: 1365, label: 'Level 9 — Intermediate' },
  { model: 'distill_s48x6.pt',    sims: 32,   rating: 1440, label: 'Level 10 — Intermediate' },
  { model: 'distill_s48x6.pt',    sims: 45,   rating: 1510, label: 'Level 11 — Medium' },      // interp.
  { model: 'distill_s48x6.pt',    sims: 64,   rating: 1580, label: 'Level 12 — Medium' },
  { model: 'distill_s48x6.pt',    sims: 90,   rating: 1650, label: 'Level 13 — Advanced' },    // interp.
  { model: 'distill_s48x6.pt',    sims: 128,  rating: 1725, label: 'Level 14 — Advanced' },
  { model: 'distill_s48x6.pt',    sims: 180,  rating: 1790, label: 'Level 15 — Advanced' },    // interp.
  { model: 'distill_s48x6.pt',    sims: 256,  rating: 1860, label: 'Level 16 — Expert' },
  { model: 'distill_s96x12.pt',   sims: 256,  rating: 1920, label: 'Level 17 — Expert' },
  { model: 'distill_s64x8.pt',    sims: 512,  rating: 2005, label: 'Level 18 — Expert' },
  { model: 'distill_s96x12.pt',   sims: 720,  rating: 2100, label: 'Level 19 — Master' },      // interp.
  { model: 'distill_s96x12.pt',   sims: 1024, rating: 2160, label: 'Level 20 — Master' },
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

/** Persist the auto-match level to localStorage. Resets the loss streak —
 * any level change (earned or manual) starts fresh at the new level. */
export function setAutoMatchLevel(level: number): void {
  const clamped = Math.max(1, Math.min(MAX_LEVEL, level));
  try {
    localStorage.setItem(STORAGE_KEY, String(clamped));
    localStorage.setItem(LOSS_STREAK_KEY, '0');
    localStorage.setItem(LEVEL_UPDATED_AT_KEY, new Date().toISOString());
  } catch { /* ignore */ }
}

function getLossStreak(): number {
  try {
    const n = parseInt(localStorage.getItem(LOSS_STREAK_KEY) ?? '0', 10);
    if (Number.isFinite(n) && n >= 0) return n;
  } catch { /* ignore */ }
  return 0;
}

function setLossStreak(n: number): void {
  try {
    localStorage.setItem(LOSS_STREAK_KEY, String(n));
  } catch { /* ignore */ }
}

/**
 * Adjust level after a game result.
 * Win => level + 1 immediately (and the loss streak resets).
 * Loss => level - 1 only after LOSS_STREAK_TO_DEMOTE consecutive losses;
 * a single loss at a freshly reached level keeps you there (hysteresis).
 * Clamped to [1, MAX_LEVEL]. Returns the new level.
 */
export function adjustAfterGame(won: boolean): number {
  const current = getAutoMatchLevel();
  if (won) {
    const next = Math.min(MAX_LEVEL, current + 1);
    setAutoMatchLevel(next); // also resets the loss streak
    return next;
  }
  const streak = getLossStreak() + 1;
  if (streak >= LOSS_STREAK_TO_DEMOTE) {
    const next = Math.max(1, current - 1);
    setAutoMatchLevel(next); // also resets the loss streak
    return next;
  }
  setLossStreak(streak);
  return current;
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
