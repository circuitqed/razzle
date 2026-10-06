/**
 * v2 networks (9 input planes, spatial policy head) vs Python reference.
 *
 * Fixtures: inference-fixtures-v2.json, from engine/scripts/gen_v2_inference_fixtures.py
 * (positions from real self-play games: player-1, pass-chain, forced-pass and
 * "must pass but no pass available" positions, with PyTorch log-policy/value).
 *
 * Checks:
 *  - replaying the fixture move prefix with the TS engine reaches the same state,
 *  - stateToTensor(state, _, 9) equals Python's to_tensor(num_planes=9) exactly,
 *    and its first 7 planes equal the v1 tensor,
 *  - the spatial action index matches network.py's spatial_action_index(),
 *  - pure-TS inference matches PyTorch (needs the fixture model's .onnx; skipped
 *    with a warning if not found, like the v1 test).
 */

import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import { createModelFromOnnx } from '../inference';
import { onnxInputPlanes } from '../onnxWeights';
import { stateToTensor, isForcedPass } from '../tensor';
import { newGame, applyMove, type EngineState } from '../state';
import { getLegalMoves } from '../moves';
import { PureTSEvaluator } from '../evaluator';
import { rotatePolicy180 } from '../symmetry';
import { SPATIAL_ACTION_INDEX, SPATIAL_PAD_SLOT } from '../modelConfig';

const FIXTURES_PATH = path.resolve(__dirname, 'inference-fixtures-v2.json');
const fixturesExist = fs.existsSync(FIXTURES_PATH);

interface FixtureState {
  pieces: [string, string];
  balls: [string, string];
  currentPlayer: number;
  touchedMask: string;
  hasPassed: boolean;
  lastKnightDst: number;
  ply: number;
}
interface V2Position {
  game: number;
  moves: number[];
  state: FixtureState;
  categories: string[];
  forcedPass: boolean;
  tensor: number[][][];
  policy: number[];
  value: number;
}
interface V2Fixtures {
  model: string;
  num_input_planes: number;
  policy_head: string;
  spatial_action_index: number[];
  positions: V2Position[];
}

const fixtures: V2Fixtures = fixturesExist
  ? JSON.parse(fs.readFileSync(FIXTURES_PATH, 'utf-8'))
  : ({ positions: [] } as unknown as V2Fixtures);

function toEngineState(s: FixtureState): EngineState {
  return {
    pieces: [BigInt(s.pieces[0]), BigInt(s.pieces[1])],
    balls: [BigInt(s.balls[0]), BigInt(s.balls[1])],
    currentPlayer: s.currentPlayer,
    touchedMask: BigInt(s.touchedMask),
    hasPassed: s.hasPassed,
    lastKnightDst: s.lastKnightDst,
    ply: s.ply,
  };
}

function flatten(t: number[][][]): Float32Array {
  const out = new Float32Array(t.length * 56);
  for (let c = 0; c < t.length; c++)
    for (let r = 0; r < 8; r++)
      for (let col = 0; col < 7; col++)
        out[c * 56 + r * 7 + col] = t[c][r][col];
  return out;
}

function argmax(a: ArrayLike<number>, idx?: number[]): number {
  const cand = idx ?? Array.from({ length: a.length }, (_, i) => i);
  let best = cand[0];
  for (const i of cand) if (a[i] > a[best]) best = i;
  return best;
}

/** Find the fixture model's ONNX export (same search paths as the v1 test). */
function loadModelBuffer(onnxName: string): ArrayBuffer | null {
  const candidates = [
    `/app/models/${onnxName}`,
    `/tmp/models/${onnxName}`,
    path.resolve(__dirname, `../../../../models/${onnxName}`),
    path.resolve(__dirname, `../../../../engine/output/models/${onnxName}`),
  ];
  for (const mp of candidates) {
    if (fs.existsSync(mp)) {
      const buf = fs.readFileSync(mp);
      console.log(`Loaded v2 model from ${mp} (${buf.length} bytes)`);
      return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    }
  }
  console.warn('No v2 ONNX model file found, skipping parity check. Tried:', candidates);
  return null;
}

describe.skipIf(!fixturesExist)('v2 network inputs (9 planes)', () => {
  it('fixtures cover the v2-specific situations', () => {
    const ps = fixtures.positions;
    expect(fixtures.num_input_planes).toBe(9);
    expect(ps.length).toBeGreaterThanOrEqual(20);
    expect(ps.some(p => p.state.currentPlayer === 1)).toBe(true);
    expect(ps.some(p => p.state.hasPassed)).toBe(true);
    expect(ps.some(p => p.forcedPass)).toBe(true);
    expect(ps.some(p => p.categories.includes('must_pass_no_pass_available'))).toBe(true);
    expect(ps.some(p => p.state.lastKnightDst < 0)).toBe(true);
    expect(ps[0].tensor).toHaveLength(9);
  });

  it('replaying the move prefix reaches the fixture state', () => {
    for (const p of fixtures.positions) {
      const s = newGame();
      for (const mv of p.moves) applyMove(s, mv);
      expect(s).toEqual(toEngineState(p.state));
    }
  });

  it('stateToTensor(9) matches Python to_tensor(num_planes=9) exactly', () => {
    for (const p of fixtures.positions) {
      const s = toEngineState(p.state);
      const ref = flatten(p.tensor);
      const t9 = stateToTensor(s, undefined, 9);
      expect(t9.length).toBe(9 * 56);
      expect(Array.from(t9)).toEqual(Array.from(ref));
      // Same result through a reused output buffer (plane count from its length)
      const buf = new Float32Array(9 * 56).fill(7);
      stateToTensor(s, buf);
      expect(Array.from(buf)).toEqual(Array.from(ref));
      // v1 tensor is unchanged and equals the first 7 planes
      const t7 = stateToTensor(s);
      expect(t7.length).toBe(7 * 56);
      expect(Array.from(t7)).toEqual(Array.from(ref.subarray(0, 7 * 56)));
      expect(isForcedPass(s)).toBe(p.forcedPass);
    }
  });

  it('forced-pass positions only allow passes', () => {
    for (const p of fixtures.positions.filter(q => q.forcedPass)) {
      const s = toEngineState(p.state);
      const ballSq = Number(s.balls[s.currentPlayer].toString(2).length - 1);
      for (const mv of getLegalMoves(s)) expect(Math.floor(mv / 56)).toBe(ballSq);
    }
  });

  it('SPATIAL_ACTION_INDEX matches network.py spatial_action_index()', () => {
    expect(Array.from(SPATIAL_ACTION_INDEX)).toEqual(fixtures.spatial_action_index);
    // Every legal move of every fixture position must map to a real logit
    for (const p of fixtures.positions) {
      for (const mv of getLegalMoves(toEngineState(p.state))) {
        expect(SPATIAL_ACTION_INDEX[mv === -1 ? 3136 : mv]).not.toBe(SPATIAL_PAD_SLOT);
      }
    }
  });
});

describe.skipIf(!fixturesExist)('v2 pure-TS inference vs PyTorch', () => {
  const onnxName = (fixtures.model ?? 'missing.pt').replace(/\.pt$/, '.onnx');

  it('detects the architecture and matches the reference outputs', async () => {
    const buffer = loadModelBuffer(onnxName);
    if (!buffer) return;

    expect(onnxInputPlanes(buffer)).toBe(9);
    const model = createModelFromOnnx(buffer);
    console.log('v2 model config:', model.config);
    expect(model.config.numInputPlanes).toBe(9);
    expect(model.config.policyHead).toBe('spatial');

    let maxLogpDiff = 0;   // over actions that a move can produce (non-PAD)
    let maxPadDiff = 0;    // PAD actions (logit -1e4: float32 rounding is ~1e-3 there)
    let maxProbDiff = 0;
    let maxValueDiff = 0;
    let topMatches = 0;
    let legalTopMatches = 0;
    const n = fixtures.positions.length;

    for (const p of fixtures.positions) {
      const s = toEngineState(p.state);
      const input = stateToTensor(s, undefined, 9);
      const { policy, value } = model.forward(input);
      for (let a = 0; a < 3137; a++) {
        const d = Math.abs(policy[a] - p.policy[a]);
        if (SPATIAL_ACTION_INDEX[a] === SPATIAL_PAD_SLOT) maxPadDiff = Math.max(maxPadDiff, d);
        else maxLogpDiff = Math.max(maxLogpDiff, d);
        maxProbDiff = Math.max(maxProbDiff, Math.abs(Math.exp(policy[a]) - Math.exp(p.policy[a])));
      }
      maxValueDiff = Math.max(maxValueDiff, Math.abs(value - p.value));
      if (argmax(policy) === argmax(p.policy)) topMatches++;
      // Top legal move (network orientation: rotate the legal moves for player 1)
      const legal = getLegalMoves(s).map(m => (m === -1 ? 3136 : m));
      const probs = Float32Array.from(policy, Math.exp);
      const oriented = s.currentPlayer === 1 ? rotatePolicy180(probs) : probs;
      const refProbs = Float32Array.from(p.policy, Math.exp);
      const refOriented = s.currentPlayer === 1 ? rotatePolicy180(refProbs) : refProbs;
      if (argmax(oriented, legal) === argmax(refOriented, legal)) legalTopMatches++;
    }

    console.log(`v2 parity over ${n} positions: max|dlogp| (movable actions)=${maxLogpDiff.toExponential(2)}, ` +
      `max|dlogp| (PAD)=${maxPadDiff.toExponential(2)}, max|dp|=${maxProbDiff.toExponential(2)}, ` +
      `max|dv|=${maxValueDiff.toExponential(2)}, top move ${topMatches}/${n}, top legal move ${legalTopMatches}/${n}`);

    expect(maxLogpDiff).toBeLessThan(1e-3);
    expect(maxPadDiff).toBeLessThan(1e-2);
    expect(maxProbDiff).toBeLessThan(1e-4);
    expect(maxValueDiff).toBeLessThan(1e-4);
    expect(topMatches).toBe(n);
    expect(legalTopMatches).toBe(n);
  }, 60_000);

  it('v1 models are still detected as 7-plane / FC head', () => {
    const v1Path = path.resolve(__dirname, 'inference-fixtures.json');
    if (!fs.existsSync(v1Path)) return;
    const v1Name = JSON.parse(fs.readFileSync(v1Path, 'utf-8')).model.replace(/\.pt$/, '.onnx');
    const buffer = loadModelBuffer(v1Name);
    if (!buffer) return;
    expect(onnxInputPlanes(buffer)).toBe(7);
    const model = createModelFromOnnx(buffer);
    expect(model.config.numInputPlanes).toBe(7);
    expect(model.config.policyHead).toBe('fc');
  });

  it('PureTSEvaluator returns board-oriented probabilities for v2', async () => {
    const buffer = loadModelBuffer(onnxName);
    if (!buffer) return;
    const evaluator = new PureTSEvaluator(createModelFromOnnx(buffer));
    for (const p of fixtures.positions.slice(0, 6)) {
      const s = toEngineState(p.state);
      const { policy, value } = await evaluator.evaluate(s);
      const ref = Float32Array.from(p.policy, Math.exp);
      const refOriented = s.currentPlayer === 1 ? rotatePolicy180(ref) : ref;
      let maxDiff = 0;
      for (let a = 0; a < 3137; a++) maxDiff = Math.max(maxDiff, Math.abs(policy[a] - refOriented[a]));
      expect(maxDiff).toBeLessThan(1e-4);
      expect(Math.abs(value - p.value)).toBeLessThan(1e-4);
      // Probability mass should sit on legal moves
      let legalMass = 0;
      for (const m of getLegalMoves(s)) legalMass += policy[m === -1 ? 3136 : m];
      expect(legalMass).toBeGreaterThan(0.5);
    }
  }, 60_000);
});
