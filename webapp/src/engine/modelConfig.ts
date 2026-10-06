/**
 * Network architecture detection + the v2 spatial policy head's action map.
 *
 * Two network generations exist (engine/razzle/ai/network.py):
 *  - v1: 7 input planes, FC policy head
 *        (1x1 conv -> BN -> ReLU -> flatten -> Linear(policy_filters*56 -> 3137)).
 *  - v2: 9 input planes (adds last-knight-destination + forced-pass planes) and a
 *        spatial policy head: h = ReLU(BN(Conv3x3(F->F)(tower))),
 *        planes = Conv1x1(F->64, bias)(h) flattened plane-major (64*56),
 *        end = Linear(F->1)(mean of h over the 56 squares), pad = -1e4,
 *        logits = [planes, end, pad][SPATIAL_ACTION_INDEX].
 *
 * The TS backends parse ONNX initializers and run a hand-written forward pass,
 * so the generation is inferred from the weights:
 *  - input planes = in-channels of the first conv (7 or 9),
 *  - policy head  = 'spatial' iff `policy_out.weight` exists (the spatial head's
 *    unfused 1x1 conv; BN-fused convs are anonymous `onnx::Conv_N`).
 */

import type { WeightTensor } from './onnxWeights';

const ROWS = 8;
const COLS = 7;
const NUM_SQUARES = ROWS * COLS; // 56
const END_TURN_ACTION = NUM_SQUARES * NUM_SQUARES; // 3136
const NUM_ACTIONS = END_TURN_ACTION + 1; // 3137

export type PolicyHead = 'fc' | 'spatial';

export interface ModelConfig {
  /** 7 (v1) or 9 (v2). */
  numInputPlanes: number;
  numFilters: number;
  numBlocks: number;
  policyHead: PolicyHead;
  /** FC head only (spatial head has no policy_filters). */
  policyFilters: number;
  valueFilters: number;
  valueHidden: number;
  /** FC head only: bottleneck hidden size, 0 = direct FC. */
  policyHidden: number;
}

// ---- Spatial policy geometry (must match network.py exactly) ----

const KNIGHT_OFFSETS: ReadonlyArray<[number, number]> = [
  [-2, -1], [-2, 1], [-1, -2], [-1, 2], [1, -2], [1, 2], [2, -1], [2, 1],
];
const LINE_DIRS: ReadonlyArray<[number, number]> = [
  [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1],
];
const MAX_DIST = 7;
export const SPATIAL_POLICY_PLANES = KNIGHT_OFFSETS.length + LINE_DIRS.length * MAX_DIST; // 64
/** Slot of the END_TURN logit in [planes (64*56), END, PAD]. */
export const SPATIAL_END_SLOT = SPATIAL_POLICY_PLANES * NUM_SQUARES; // 3584
/** Slot of the padding logit (actions no move can produce). */
export const SPATIAL_PAD_SLOT = SPATIAL_END_SLOT + 1; // 3585
export const SPATIAL_PAD_LOGIT = -1e4;

/** Port of network.py spatial_action_index(): action -> slot in [planes, END, PAD]. */
export function spatialActionIndex(): Int32Array {
  const idx = new Int32Array(NUM_ACTIONS).fill(SPATIAL_PAD_SLOT);
  idx[END_TURN_ACTION] = SPATIAL_END_SLOT;
  for (let src = 0; src < NUM_SQUARES; src++) {
    const sr = Math.floor(src / COLS);
    const sc = src % COLS;
    for (let k = 0; k < KNIGHT_OFFSETS.length; k++) {
      const r = sr + KNIGHT_OFFSETS[k][0];
      const c = sc + KNIGHT_OFFSETS[k][1];
      if (r >= 0 && r < ROWS && c >= 0 && c < COLS) {
        idx[src * NUM_SQUARES + r * COLS + c] = k * NUM_SQUARES + src;
      }
    }
    for (let d = 0; d < LINE_DIRS.length; d++) {
      for (let dist = 1; dist <= MAX_DIST; dist++) {
        const r = sr + LINE_DIRS[d][0] * dist;
        const c = sc + LINE_DIRS[d][1] * dist;
        if (!(r >= 0 && r < ROWS && c >= 0 && c < COLS)) break;
        const plane = KNIGHT_OFFSETS.length + d * MAX_DIST + (dist - 1);
        idx[src * NUM_SQUARES + r * COLS + c] = plane * NUM_SQUARES + src;
      }
    }
  }
  return idx;
}

export const SPATIAL_ACTION_INDEX: Int32Array = spatialActionIndex();

/**
 * Gather spatial-head logits into the 3137-action layout.
 * @param planes 64*56 plane-major logits (plane * 56 + square)
 * @param end    END_TURN logit
 * @param out    3137 action logits (written)
 */
export function gatherSpatialLogits(planes: Float32Array, end: number, out: Float32Array): void {
  const idx = SPATIAL_ACTION_INDEX;
  for (let a = 0; a < NUM_ACTIONS; a++) {
    const slot = idx[a];
    out[a] = slot < SPATIAL_END_SLOT ? planes[slot]
      : slot === SPATIAL_END_SLOT ? end : SPATIAL_PAD_LOGIT;
  }
}

// ---- Config inference from ONNX weights ----

/** BN-fused convs (`onnx::Conv_N`), in graph order. */
export function sortedConvTensors(weights: Iterable<WeightTensor>): WeightTensor[] {
  return [...weights]
    .filter(t => t.name.startsWith('onnx::Conv_'))
    .sort((a, b) => parseInt(a.name.split('_').pop()!) - parseInt(b.name.split('_').pop()!));
}

/**
 * Infer the architecture from ONNX initializers.
 *
 * Fused conv order (both generations): input conv, 2 per residual block, policy
 * conv, value conv, difficulty conv. The v2 policy conv is a 3x3 F->F conv (so
 * it counts among the 3x3s); the v1 policy conv is 1x1 F->policy_filters.
 */
export function inferModelConfig(weights: Map<string, WeightTensor>): ModelConfig {
  const convs = sortedConvTensors(weights.values()).filter(t => t.shape.length === 4);
  if (convs.length === 0) throw new Error('ONNX model has no conv weights');
  const inputConv = convs[0];
  const numInputPlanes = inputConv.shape[1];
  const numFilters = inputConv.shape[0];
  if (numInputPlanes !== 7 && numInputPlanes !== 9) {
    throw new Error(`Unsupported network: first conv has ${numInputPlanes} input planes (expected 7 or 9)`);
  }

  const policyHead: PolicyHead = weights.has('policy_out.weight') ? 'spatial' : 'fc';

  const conv3x3Count = convs.filter(t => t.shape[2] === 3).length;
  // Subtract the input conv (and the spatial policy conv for v2).
  const numBlocks = (conv3x3Count - (policyHead === 'spatial' ? 2 : 1)) / 2;
  if (!Number.isInteger(numBlocks) || numBlocks < 0) {
    throw new Error(`Cannot infer residual block count from ${conv3x3Count} 3x3 convs (${policyHead} head)`);
  }

  const conv1x1s = convs.filter(t => t.shape[2] === 1);
  let policyFilters = 0;
  let valueFilters: number;
  if (policyHead === 'spatial') {
    valueFilters = conv1x1s[0]?.shape[0] ?? 1;
    const po = weights.get('policy_out.weight')!;
    if (po.shape[0] !== SPATIAL_POLICY_PLANES || po.shape[1] !== numFilters) {
      throw new Error(`Unexpected policy_out shape ${JSON.stringify(po.shape)}`);
    }
    if (!weights.has('policy_out.bias') || !weights.has('policy_end.weight') || !weights.has('policy_end.bias')) {
      throw new Error('Spatial policy head is missing policy_out.bias / policy_end weights');
    }
  } else {
    policyFilters = conv1x1s[0]?.shape[0] ?? 2;
    valueFilters = conv1x1s[1]?.shape[0] ?? 1;
  }

  const valueFc1 = weights.get('value_fc1.weight');
  const valueHidden = valueFc1 ? valueFc1.shape[0] : 256;
  const policyFc1 = weights.get('policy_fc1.weight');
  const policyHidden = policyFc1 ? policyFc1.shape[0] : 0;

  return {
    numInputPlanes, numFilters, numBlocks, policyHead,
    policyFilters, valueFilters, valueHidden, policyHidden,
  };
}
