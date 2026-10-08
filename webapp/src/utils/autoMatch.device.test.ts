import { describe, it, expect, beforeEach, vi } from 'vitest';

// A phone or the native app: no desktop-only levels.
vi.mock('./device', () => ({ isDesktopWeb: () => false }));

import {
  adjustAfterGame, getAutoMatchLevel, setAutoMatchLevel, getTierSettings, maxLevelForDevice, MAX_LEVEL, TIERS,
} from './autoMatch';

const DEVICE_MAX = TIERS.findIndex((t) => t.desktopOnly);

describe('autoMatch on a device without desktop-only levels', () => {
  beforeEach(() => localStorage.clear());

  it('the ladder has desktop-only levels after level 20', () => {
    expect(DEVICE_MAX).toBe(20);
    expect(MAX_LEVEL).toBeGreaterThan(20);
    expect(TIERS.slice(DEVICE_MAX).every((t) => t.desktopOnly)).toBe(true);
    expect(maxLevelForDevice()).toBe(20);
    expect(maxLevelForDevice(true)).toBe(MAX_LEVEL);
  });

  it('a desktop level synced from another device plays as the top level here', () => {
    setAutoMatchLevel(DEVICE_MAX + 2);
    expect(localStorage.getItem('knightball_ai_level')).toBe(String(DEVICE_MAX + 2));
    expect(getAutoMatchLevel()).toBe(DEVICE_MAX);
    expect(getTierSettings(DEVICE_MAX + 2)).toBe(TIERS[DEVICE_MAX - 1]);
  });

  it('a win at the top level does not overwrite a higher synced level', () => {
    setAutoMatchLevel(DEVICE_MAX + 2);
    expect(adjustAfterGame(true)).toBe(DEVICE_MAX);
    expect(localStorage.getItem('knightball_ai_level')).toBe(String(DEVICE_MAX + 2));
  });

  it('wins stop at the device top level', () => {
    setAutoMatchLevel(DEVICE_MAX - 1);
    expect(adjustAfterGame(true)).toBe(DEVICE_MAX);
    expect(adjustAfterGame(true)).toBe(DEVICE_MAX);
  });

  it('losses still demote from the effective level', () => {
    setAutoMatchLevel(DEVICE_MAX + 2);
    adjustAfterGame(false);
    expect(adjustAfterGame(false)).toBe(DEVICE_MAX - 1);
  });
});
