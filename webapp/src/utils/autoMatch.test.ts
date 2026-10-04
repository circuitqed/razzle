import { describe, it, expect, beforeEach } from 'vitest';
import {
  adjustAfterGame,
  getAutoMatchLevel,
  setAutoMatchLevel,
  LOSS_STREAK_TO_DEMOTE,
  MAX_LEVEL,
  TIERS,
} from './autoMatch';

describe('autoMatch level hysteresis', () => {
  beforeEach(() => {
    localStorage.clear();
    setAutoMatchLevel(5);
  });

  it('promotes immediately on a win', () => {
    expect(adjustAfterGame(true)).toBe(6);
    expect(getAutoMatchLevel()).toBe(6);
  });

  it('does NOT demote on a single loss', () => {
    expect(adjustAfterGame(false)).toBe(5);
    expect(getAutoMatchLevel()).toBe(5);
  });

  it('demotes after LOSS_STREAK_TO_DEMOTE consecutive losses', () => {
    for (let i = 0; i < LOSS_STREAK_TO_DEMOTE - 1; i++) {
      expect(adjustAfterGame(false)).toBe(5);
    }
    expect(adjustAfterGame(false)).toBe(4);
  });

  it('a win resets the loss streak (no ping-pong at the boundary)', () => {
    // W L W L W L — the classic boundary sequence must hold the level
    adjustAfterGame(true); // 5 -> 6
    expect(adjustAfterGame(false)).toBe(6); // single loss holds
    adjustAfterGame(true); // 6 -> 7
    expect(adjustAfterGame(false)).toBe(7); // streak was reset by the win
    expect(getAutoMatchLevel()).toBe(7);
  });

  it('demotion resets the streak (takes another full streak to drop again)', () => {
    for (let i = 0; i < LOSS_STREAK_TO_DEMOTE; i++) adjustAfterGame(false); // 5 -> 4
    expect(getAutoMatchLevel()).toBe(4);
    expect(adjustAfterGame(false)).toBe(4); // first loss at 4 holds
  });

  it('manual level change resets the streak', () => {
    adjustAfterGame(false); // streak 1 at level 5
    setAutoMatchLevel(9);
    expect(adjustAfterGame(false)).toBe(9); // fresh streak at the new level
  });

  it('clamps at the bottom and top', () => {
    setAutoMatchLevel(1);
    for (let i = 0; i < LOSS_STREAK_TO_DEMOTE * 2; i++) adjustAfterGame(false);
    expect(getAutoMatchLevel()).toBe(1);
    setAutoMatchLevel(MAX_LEVEL);
    expect(adjustAfterGame(true)).toBe(MAX_LEVEL);
  });
});

describe('calibrated ladder', () => {
  it('has strictly increasing ratings and labels matching the level number', () => {
    TIERS.forEach((t, i) => {
      expect(t.label.startsWith(`Level ${i + 1} — `)).toBe(true);
      if (i > 0) expect(t.rating).toBeGreaterThan(TIERS[i - 1].rating);
    });
  });

  it('only uses models that the native app bundles (offline play)', async () => {
    const { readFileSync } = await import('fs');
    const src = readFileSync('scripts/bundle-models.mjs', 'utf8')  // vitest runs from webapp/;
    const bundled = new Set([...src.matchAll(/version: '([^']+)'/g)].map(m => m[1]));
    for (const t of TIERS) expect(bundled.has(t.model.replace(/\.pt$/, ''))).toBe(true);
  });
});
