import { describe, it, expect, beforeEach } from 'vitest';
import { recordSearchSpeed, estimateSimsPerSec, estimateMoveSeconds, networkCost, formatSeconds } from './searchSpeed';

describe('searchSpeed', () => {
  beforeEach(() => localStorage.clear());

  it('knows nothing before any search', () => {
    expect(estimateSimsPerSec('distill_v2_96x12.pt')).toBeNull();
    expect(estimateMoveSeconds('distill_v2_96x12.pt', 1024)).toBeNull();
  });

  it('uses the measured speed for a model, ignoring tiny searches', () => {
    recordSearchSpeed('distill_v2_96x12.pt', 'webgpu', 8, 1000); // too few sims: ignored
    expect(estimateSimsPerSec('distill_v2_96x12.pt')).toBeNull();
    recordSearchSpeed('distill_v2_96x12.pt', 'webgpu', 1024, 4096); // 250 sims/s
    expect(estimateSimsPerSec('distill_v2_96x12')).toBeCloseTo(250);
    expect(estimateMoveSeconds('distill_v2_96x12.pt', 2048)).toBeCloseTo(8.192);
  });

  it('averages repeated measurements', () => {
    recordSearchSpeed('distill_v2_96x12.pt', 'webgpu', 1000, 10_000); // 100
    recordSearchSpeed('distill_v2_96x12.pt', 'webgpu', 1000, 5_000);  // 200
    expect(estimateSimsPerSec('distill_v2_96x12.pt')).toBeCloseTo(130);
  });

  it('scales a bigger unmeasured network down by cost, never a smaller one up', () => {
    recordSearchSpeed('distill_v2_96x12.pt', 'webgpu', 1024, 4096); // 250
    const ratio = networkCost('distill_v2_96x12')! / networkCost('distill_v2_128x16')!;
    expect(estimateSimsPerSec('distill_v2_128x16.pt')).toBeCloseTo(250 * ratio);
    expect(estimateSimsPerSec('distill_v2_16x2.pt')).toBeCloseTo(250);
  });

  it('formats durations', () => {
    expect(formatSeconds(4.4)).toBe('4 s');
    expect(formatSeconds(23)).toBe('25 s');
    expect(formatSeconds(150)).toBe('3 min');
  });
});
