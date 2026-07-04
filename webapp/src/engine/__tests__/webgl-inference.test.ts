/**
 * Validate pure-TS CPU inference against Python reference fixtures.
 *
 * This tests that createModelFromOnnx + forward() produces the same
 * policy/value outputs as the Python PyTorch model. Runs in vitest (jsdom).
 *
 * A separate Playwright test (e2e/webgl-inference.spec.ts) validates
 * that the WebGL GEMM produces identical results to this CPU path.
 */

import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import { createModelFromOnnx } from '../inference';

const FIXTURES_PATH = path.resolve(__dirname, 'inference-fixtures.json');

// Skip if fixtures or model not available
const fixturesExist = fs.existsSync(FIXTURES_PATH);

// The ONNX model name is derived from the fixtures' "model" field (a .pt name,
// e.g. "pegasus_iter_250.pt") so regenerating fixtures against a new model
// doesn't require touching this test. Regenerate with
// engine/scripts/regen_inference_fixtures.py.
const onnxName = fixturesExist
  ? JSON.parse(fs.readFileSync(FIXTURES_PATH, 'utf-8')).model.replace(/\.pt$/, '.onnx')
  : 'missing.onnx';

describe.skipIf(!fixturesExist)('Pure-TS Inference vs Python Reference', () => {
  // Load fixtures
  const fixtures = fixturesExist
    ? JSON.parse(fs.readFileSync(FIXTURES_PATH, 'utf-8'))
    : { positions: [] };

  it('should have fixtures with correct structure', () => {
    expect(fixtures.positions.length).toBeGreaterThan(0);
    const pos = fixtures.positions[0];
    expect(pos.tensor).toHaveLength(7);
    expect(pos.tensor[0]).toHaveLength(8);
    expect(pos.tensor[0][0]).toHaveLength(7);
    expect(pos.policy).toHaveLength(3137);
    expect(typeof pos.value).toBe('number');
  });

  it('should produce matching policy/value for all fixture positions', async () => {
    // We need the ONNX model file. Try to fetch it or skip.
    // Try multiple paths for the ONNX model
    const modelPaths = [
      `/app/models/${onnxName}`,
      `/tmp/models/${onnxName}`,
      path.resolve(__dirname, `../../../../models/${onnxName}`),
      path.resolve(__dirname, `../../../../engine/output/models/${onnxName}`),
    ];
    let modelBuffer: ArrayBuffer | null = null;
    for (const mp of modelPaths) {
      if (fs.existsSync(mp)) {
        const buf = fs.readFileSync(mp);
        // Convert Node Buffer to ArrayBuffer
        modelBuffer = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
        console.log(`Loaded model from ${mp} (${buf.length} bytes)`);
        break;
      }
    }
    if (!modelBuffer) {
      console.warn('No ONNX model file found, skipping. Tried:', modelPaths);
      return;
    }

    const model = createModelFromOnnx(modelBuffer);
    console.log('Model config:', model.config);

    let maxPolicyDiff = 0;
    let maxValueDiff = 0;
    let passed = 0;

    for (let p = 0; p < fixtures.positions.length; p++) {
      const pos = fixtures.positions[p];

      // Flatten tensor from [7][8][7] to Float32Array(392)
      const tensor = new Float32Array(7 * 8 * 7);
      for (let c = 0; c < 7; c++) {
        for (let r = 0; r < 8; r++) {
          for (let col = 0; col < 7; col++) {
            tensor[c * 56 + r * 7 + col] = pos.tensor[c][r][col];
          }
        }
      }

      const { policy, value } = model.forward(tensor);

      // Compare policy (log-softmax values)
      let posMaxDiff = 0;
      for (let i = 0; i < 3137; i++) {
        const diff = Math.abs(policy[i] - pos.policy[i]);
        if (diff > posMaxDiff) posMaxDiff = diff;
      }

      // Compare value
      const valueDiff = Math.abs(value - pos.value);

      if (posMaxDiff > maxPolicyDiff) maxPolicyDiff = posMaxDiff;
      if (valueDiff > maxValueDiff) maxValueDiff = valueDiff;

      // Log detailed info for failing positions
      if (posMaxDiff > 0.01 || valueDiff > 0.01) {
        console.log(`Position ${p}: policy maxΔ=${posMaxDiff.toExponential(2)}, value Δ=${valueDiff.toExponential(2)}`);
        console.log(`  CPU value=${value.toFixed(6)}, Python value=${pos.value.toFixed(6)}`);
        // Show top 3 policy differences
        const diffs = Array.from({ length: 3137 }, (_, i) => ({
          i, diff: Math.abs(policy[i] - pos.policy[i]),
          cpu: policy[i], py: pos.policy[i],
        }));
        diffs.sort((a, b) => b.diff - a.diff);
        for (const d of diffs.slice(0, 3)) {
          console.log(`  Action ${d.i}: cpu=${d.cpu.toFixed(4)}, py=${d.py.toFixed(4)}, Δ=${d.diff.toExponential(2)}`);
        }
      } else {
        passed++;
      }
    }

    console.log(`\nResults: ${passed}/${fixtures.positions.length} positions within tolerance`);
    console.log(`Max policy diff: ${maxPolicyDiff.toExponential(3)}`);
    console.log(`Max value diff: ${maxValueDiff.toExponential(3)}`);

    // Float32 vs Float64 precision: expect ~1e-3 to 1e-2 differences
    // (Python uses float64 for some ops, our TS uses Float32Array)
    expect(maxPolicyDiff).toBeLessThan(0.05); // generous tolerance for float32 vs float64
    expect(maxValueDiff).toBeLessThan(0.01);
  });
});
