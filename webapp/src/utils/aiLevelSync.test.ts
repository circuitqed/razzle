import { describe, it, expect, beforeEach, vi } from 'vitest';

const api = vi.hoisted(() => ({
  getAIProgress: vi.fn(),
  putAIProgress: vi.fn(),
}));
vi.mock('../api/history', () => api);

import {
  mergeProgress, readLocalProgress, recordLevelBeaten, syncAIProgress, levelForSettings,
} from './aiLevelSync';
import { getAutoMatchLevel, setAutoMatchLevel, getTierSettings, LEVEL_UPDATED_AT_KEY, TIERS } from './autoMatch';

const T1 = '2026-01-01T00:00:00.000Z';
const T2 = '2026-01-02T00:00:00.000Z';

describe('mergeProgress', () => {
  it('takes the max of highest level beaten', () => {
    const { merged, push } = mergeProgress(
      { level: 3, levelUpdatedAt: T1, highestBeaten: 2 },
      { current_level: 3, current_level_updated_at: T1, highest_level_beaten: 6 },
    );
    expect(merged.highestBeaten).toBe(6);
    expect(push).toBe(false);
  });

  it('pushes when the local peak is higher', () => {
    const { merged, push } = mergeProgress(
      { level: 3, levelUpdatedAt: T1, highestBeaten: 7 },
      { current_level: 3, current_level_updated_at: T1, highest_level_beaten: 6 },
    );
    expect(merged.highestBeaten).toBe(7);
    expect(push).toBe(true);
  });

  it('prefers the more recently updated current level', () => {
    const serverNewer = mergeProgress(
      { level: 8, levelUpdatedAt: T1, highestBeaten: 0 },
      { current_level: 4, current_level_updated_at: T2, highest_level_beaten: 0 },
    );
    expect(serverNewer.merged.level).toBe(4);
    expect(serverNewer.push).toBe(false);

    const localNewer = mergeProgress(
      { level: 4, levelUpdatedAt: T2, highestBeaten: 0 },
      { current_level: 8, current_level_updated_at: T1, highest_level_beaten: 0 },
    );
    expect(localNewer.merged.level).toBe(4);
    expect(localNewer.push).toBe(true);
  });

  it('a device that never changed its level adopts the account level', () => {
    const { merged, push } = mergeProgress(
      { level: 1, levelUpdatedAt: null, highestBeaten: 0 },
      { current_level: 9, current_level_updated_at: T1, highest_level_beaten: 8 },
    );
    expect(merged.level).toBe(9);
    expect(push).toBe(false);
  });

  it('seeds an empty account from the device', () => {
    const { merged, push } = mergeProgress(
      { level: 5, levelUpdatedAt: T1, highestBeaten: 4 },
      { current_level: null, current_level_updated_at: null, highest_level_beaten: 0 },
    );
    expect(merged).toEqual({ level: 5, levelUpdatedAt: T1, highestBeaten: 4 });
    expect(push).toBe(true);
  });
});

describe('local progress', () => {
  beforeEach(() => localStorage.clear());

  it('setAutoMatchLevel stamps the change time', () => {
    expect(readLocalProgress().levelUpdatedAt).toBeNull();
    setAutoMatchLevel(4);
    expect(readLocalProgress().levelUpdatedAt).not.toBeNull();
  });

  it('recordLevelBeaten only raises the peak', () => {
    recordLevelBeaten(5);
    recordLevelBeaten(3);
    expect(readLocalProgress().highestBeaten).toBe(5);
  });
});

describe('syncAIProgress', () => {
  beforeEach(() => {
    localStorage.clear();
    api.getAIProgress.mockReset();
    api.putAIProgress.mockReset().mockImplementation(async (p) => p);
  });

  it('applies a newer account level locally without pushing', async () => {
    setAutoMatchLevel(2);
    localStorage.setItem(LEVEL_UPDATED_AT_KEY, T1);
    api.getAIProgress.mockResolvedValue({ current_level: 7, current_level_updated_at: T2, highest_level_beaten: 6 });
    const merged = await syncAIProgress();
    expect(getAutoMatchLevel()).toBe(7);
    expect(localStorage.getItem(LEVEL_UPDATED_AT_KEY)).toBe(T2);
    expect(merged?.highestBeaten).toBe(6);
    expect(api.putAIProgress).not.toHaveBeenCalled();
  });

  it('pushes newer local progress to the account', async () => {
    setAutoMatchLevel(6);
    localStorage.setItem(LEVEL_UPDATED_AT_KEY, T2);
    recordLevelBeaten(5);
    api.getAIProgress.mockResolvedValue({ current_level: 3, current_level_updated_at: T1, highest_level_beaten: 2 });
    await syncAIProgress();
    expect(api.putAIProgress).toHaveBeenCalledWith({
      current_level: 6, current_level_updated_at: T2, highest_level_beaten: 5,
    });
    expect(getAutoMatchLevel()).toBe(6);
  });

  it('leaves local state alone when the server is unreachable', async () => {
    setAutoMatchLevel(4);
    api.getAIProgress.mockRejectedValue(new TypeError('offline'));
    expect(await syncAIProgress()).toBeNull();
    expect(getAutoMatchLevel()).toBe(4);
  });
});

describe('levelForSettings', () => {
  beforeEach(() => localStorage.clear());

  it('uses the auto-match level when its tier matches', () => {
    setAutoMatchLevel(3);
    const t = getTierSettings(3);
    expect(levelForSettings(t.model, t.sims, 'auto')).toBe(3);
  });

  it('maps preset/custom settings that match a tier, else undefined', () => {
    const t = TIERS[1];
    expect(levelForSettings(t.model, t.sims, 'beginner')).toBe(2);
    expect(levelForSettings('nonexistent.pt', 5, 'custom')).toBeUndefined();
    expect(levelForSettings(undefined, 5, 'custom')).toBeUndefined();
  });
});
