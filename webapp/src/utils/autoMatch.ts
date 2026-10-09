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

import { isDesktopWeb } from './device';

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
  /** Deep-search level offered only in desktop browsers (see isDesktopWeb). */
  desktopOnly?: boolean;
}

// Calibrated ladder (Oct 2026, v2). Each (model, sims) pair was placed on one
// Elo scale with ~45k arena games (engine/scripts/distill/calibration.py);
// steps are ~55-100 points. Ratings are shown on a human-friendly scale
// (displayed = 880 + calibrated Elo, where pegasus_iter_050 at 1 sim = 0).
// Levels marked "interp." are interpolated in log(sims) between measured
// neighbours. Models are v2 students distill_v2_{filters}x{blocks}, distilled
// from phoenix3: each level uses the smallest network that reaches its
// rating, so the low levels cost almost nothing per move.
export const TIERS: TierSettings[] = [
  { model: 'distill_v2_16x2.pt',  sims: 1,    rating: 815,  label: 'Level 1 — Beginner' },
  { model: 'distill_v2_16x2.pt',  sims: 8,    rating: 880,  label: 'Level 2 — Beginner' },
  { model: 'distill_v2_16x2.pt',  sims: 32,   rating: 975,  label: 'Level 3 — Beginner' },
  { model: 'distill_v2_24x3.pt',  sims: 1,    rating: 1035, label: 'Level 4 — Beginner' },
  { model: 'distill_v2_24x3.pt',  sims: 16,   rating: 1115, label: 'Level 5 — Easy' },
  { model: 'distill_v2_32x4.pt',  sims: 1,    rating: 1180, label: 'Level 6 — Easy' },
  { model: 'distill_v2_32x4.pt',  sims: 16,   rating: 1255, label: 'Level 7 — Easy' },
  { model: 'distill_v2_32x4.pt',  sims: 32,   rating: 1330, label: 'Level 8 — Intermediate' },
  { model: 'distill_v2_32x4.pt',  sims: 48,   rating: 1410, label: 'Level 9 — Intermediate' },  // interp.
  { model: 'distill_v2_32x4.pt',  sims: 64,   rating: 1475, label: 'Level 10 — Intermediate' },
  { model: 'distill_v2_48x6.pt',  sims: 45,   rating: 1570, label: 'Level 11 — Medium' },       // interp.
  { model: 'distill_v2_48x6.pt',  sims: 64,   rating: 1655, label: 'Level 12 — Medium' },
  { model: 'distill_v2_48x6.pt',  sims: 90,   rating: 1715, label: 'Level 13 — Advanced' },     // interp.
  { model: 'distill_v2_48x6.pt',  sims: 128,  rating: 1775, label: 'Level 14 — Advanced' },
  { model: 'distill_v2_48x6.pt',  sims: 200,  rating: 1865, label: 'Level 15 — Advanced' },     // interp.
  { model: 'distill_v2_64x8.pt',  sims: 200,  rating: 1930, label: 'Level 16 — Expert' },       // interp.
  { model: 'distill_v2_64x8.pt',  sims: 320,  rating: 2010, label: 'Level 17 — Expert' },       // interp.
  { model: 'distill_v2_64x8.pt',  sims: 512,  rating: 2080, label: 'Level 18 — Expert' },
  { model: 'distill_v2_96x12.pt', sims: 640,  rating: 2165, label: 'Level 19 — Master' },       // interp.
  { model: 'distill_v2_96x12.pt', sims: 1024, rating: 2225, label: 'Level 20 — Master' },
  // Desktop browsers only: deep searches that would take minutes per move on a
  // phone. Measured with ~7k extra games at 2048-8192 sims: strength levels off
  // here (1024 -> 8192 sims is only ~+100), and 128x16 is within noise of 96x12
  // at equal sims, so it is used only for the top level.
  { model: 'distill_v2_96x12.pt',  sims: 2048, rating: 2270, label: 'Level 21 — Grandmaster', desktopOnly: true },
  { model: 'distill_v2_96x12.pt',  sims: 4096, rating: 2305, label: 'Level 22 — Grandmaster', desktopOnly: true },
  { model: 'distill_v2_128x16.pt', sims: 8192, rating: 2340, label: 'Level 23 — Grandmaster', desktopOnly: true },
];

export const MAX_LEVEL = TIERS.length;

/** Highest level offered on this device: desktop browsers get the desktopOnly levels. */
export function maxLevelForDevice(desktop: boolean = isDesktopWeb()): number {
  if (desktop) return MAX_LEVEL;
  const firstDesktop = TIERS.findIndex((t) => t.desktopOnly);
  return firstDesktop < 0 ? MAX_LEVEL : firstDesktop;
}

/** Read the current auto-match level from localStorage (1-indexed, default 1). */
export function getAutoMatchLevel(): number {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const n = parseInt(raw, 10);
      // A desktop-only level synced to a phone plays as the phone's top level.
      if (n >= 1) return Math.min(n, maxLevelForDevice());
    }
  } catch { /* ignore */ }
  return 1;
}

/** Persist the auto-match level to localStorage. Resets the loss streak —
 * any level change (earned or manual) starts fresh at the new level. */
export function setAutoMatchLevel(level: number): void {
  // Clamped to the whole ladder, not this device: a desktop-only level synced
  // from another device is kept (and plays as this device's top level).
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
 * Clamped to [1, maxLevelForDevice()]. Returns the new level.
 */
export function adjustAfterGame(won: boolean): number {
  const current = getAutoMatchLevel();
  if (won) {
    const next = Math.min(maxLevelForDevice(), current + 1);
    // At this device's top level there is nowhere to go; leave the stored
    // level alone so a higher desktop level synced from elsewhere survives.
    if (next === current) { setLossStreak(0); return current; }
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
  const idx = Math.max(0, Math.min(maxLevelForDevice() - 1, level - 1));
  return TIERS[idx];
}

/** Get the display label for a level, e.g. "Level 3 — Beginner". */
export function getLevelLabel(level: number): string {
  return getTierSettings(level).label;
}

/** The ladder level that plays as (model, sims), or null if it isn't one (custom games, old models). */
export function levelForConfig(model: string | null | undefined, sims: number): number | null {
  if (!model) return null;
  const file = model.split('/').pop()!.replace(/\.(pt|onnx)$/, '');
  const i = TIERS.findIndex((t) => t.model.replace(/\.pt$/, '') === file && t.sims === sims);
  return i < 0 ? null : i + 1;
}
