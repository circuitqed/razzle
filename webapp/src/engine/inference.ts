/**
 * Pure TypeScript neural network inference for RazzleNet.
 *
 * Implements the AlphaZero-style architecture without WASM or ONNX Runtime.
 * Uses im2col + GEMM for cache-friendly convolution. All operations use
 * Float32Array for performance.
 *
 * This is the iOS-compatible inference path. On desktop, OnnxEvaluator
 * is preferred for its WASM SIMD acceleration.
 *
 * Performance optimizations:
 * - All scratch buffers pre-allocated (zero GC pressure per forward pass)
 * - GEMM inner loop unrolled 4x
 * - Bias+ReLU branches hoisted out of inner loops
 */

import { type WeightTensor, parseOnnxWeights } from './onnxWeights';

// Board dimensions
const ROWS = 8;
const COLS = 7;
const HW = ROWS * COLS; // 56

// ---- Core math operations ----

/**
 * General matrix multiply: C = A @ B
 * A: (M, K), B: (K, N), C: (M, N)
 *
 * Tiled for L1 cache: processes K in blocks of TK so that the active
 * slice of B (~TK*N floats) fits in L1. Within each tile, accumulates
 * 4 output elements in local variables (registers) to avoid repeated
 * C array loads/stores in the inner loop.
 */
function gemm(
  A: Float32Array, B: Float32Array, C: Float32Array,
  M: number, K: number, N: number,
): void {
  C.fill(0);
  const TK = 64; // K-tile: 64 * 56 * 4 bytes = 14KB, fits in L1
  const N4 = N - (N % 4);

  for (let kt = 0; kt < K; kt += TK) {
    const kEnd = Math.min(kt + TK, K);
    for (let m = 0; m < M; m++) {
      const aRowOff = m * K;
      const cRowOff = m * N;
      // Process 4 N-columns at a time with register accumulation
      let n = 0;
      for (; n < N4; n += 4) {
        const ci = cRowOff + n;
        let c0 = C[ci], c1 = C[ci + 1], c2 = C[ci + 2], c3 = C[ci + 3];
        for (let k = kt; k < kEnd; k++) {
          const a = A[aRowOff + k];
          const bi = k * N + n;
          c0 += a * B[bi];
          c1 += a * B[bi + 1];
          c2 += a * B[bi + 2];
          c3 += a * B[bi + 3];
        }
        C[ci] = c0; C[ci + 1] = c1; C[ci + 2] = c2; C[ci + 3] = c3;
      }
      // Remainder columns
      for (; n < N; n++) {
        let c = C[cRowOff + n];
        for (let k = kt; k < kEnd; k++) {
          c += A[aRowOff + k] * B[k * N + n];
        }
        C[cRowOff + n] = c;
      }
    }
  }
}

/**
 * im2col for 3x3 conv with padding=1.
 * Input: (C, H, W) in CHW order
 * Output: (C*9, H*W) column matrix
 */
function im2col3x3(
  input: Float32Array, C: number, H: number, W: number,
  col: Float32Array,
): void {
  const colW = H * W;
  col.fill(0);

  for (let c = 0; c < C; c++) {
    const inOff = c * H * W;
    for (let kh = 0; kh < 3; kh++) {
      for (let kw = 0; kw < 3; kw++) {
        const colRow = (c * 9 + kh * 3 + kw);
        const colRowOff = colRow * colW;
        // Clamp row range to avoid per-element bounds check
        const hStart = Math.max(0, 1 - kh);
        const hEnd = Math.min(H, H + 1 - kh);
        for (let h = hStart; h < hEnd; h++) {
          const ih = h + kh - 1;
          // Clamp col range to avoid per-element bounds check
          const wStart = Math.max(0, 1 - kw);
          const wEnd = Math.min(W, W + 1 - kw);
          const rowOutOff = colRowOff + h * W;
          const rowInOff = inOff + ih * W;
          for (let w = wStart; w < wEnd; w++) {
            col[rowOutOff + w] = input[rowInOff + w + kw - 1];
          }
        }
      }
    }
  }
}

/**
 * Conv2d (3x3, padding=1) + bias + ReLU.
 * BN is already fused into weight/bias by the ONNX exporter.
 */
function conv3x3relu(
  input: Float32Array, weight: Float32Array, bias: Float32Array,
  inC: number, outC: number, H: number, W: number,
  col: Float32Array, output: Float32Array,
): void {
  im2col3x3(input, inC, H, W, col);
  gemm(weight, col, output, outC, inC * 9, H * W);
  const hw = H * W;
  for (let oc = 0; oc < outC; oc++) {
    const b = bias[oc];
    const off = oc * hw;
    for (let i = 0; i < hw; i++) {
      output[off + i] = Math.max(0, output[off + i] + b);
    }
  }
}

/**
 * Conv2d (3x3, padding=1) + bias (no ReLU).
 */
function conv3x3noRelu(
  input: Float32Array, weight: Float32Array, bias: Float32Array,
  inC: number, outC: number, H: number, W: number,
  col: Float32Array, output: Float32Array,
): void {
  im2col3x3(input, inC, H, W, col);
  gemm(weight, col, output, outC, inC * 9, H * W);
  const hw = H * W;
  for (let oc = 0; oc < outC; oc++) {
    const b = bias[oc];
    const off = oc * hw;
    for (let i = 0; i < hw; i++) {
      output[off + i] += b;
    }
  }
}

/**
 * Conv2d (1x1) + bias + ReLU. No im2col needed.
 * Input treated as (inC, H*W) matrix.
 */
function conv1x1(
  input: Float32Array, weight: Float32Array, bias: Float32Array,
  inC: number, outC: number, HW: number,
  output: Float32Array,
): void {
  gemm(weight, input, output, outC, inC, HW);
  for (let oc = 0; oc < outC; oc++) {
    const b = bias[oc];
    const off = oc * HW;
    for (let i = 0; i < HW; i++) {
      output[off + i] = Math.max(0, output[off + i] + b);
    }
  }
}

/**
 * Linear (fully connected) layer: output = input @ weight^T + bias
 */
function linear(
  input: Float32Array, weight: Float32Array, bias: Float32Array,
  outFeatures: number, inFeatures: number,
  output: Float32Array,
): void {
  const inF4 = inFeatures - (inFeatures % 4);
  for (let m = 0; m < outFeatures; m++) {
    let sum = bias[m];
    const wOff = m * inFeatures;
    let n = 0;
    for (; n < inF4; n += 4) {
      sum += weight[wOff + n] * input[n]
           + weight[wOff + n + 1] * input[n + 1]
           + weight[wOff + n + 2] * input[n + 2]
           + weight[wOff + n + 3] * input[n + 3];
    }
    for (; n < inFeatures; n++) {
      sum += weight[wOff + n] * input[n];
    }
    output[m] = sum;
  }
}

/**
 * Log-softmax: output[i] = input[i] - log(sum(exp(input)))
 * Uses the max trick for numerical stability.
 */
function logSoftmax(input: Float32Array, output: Float32Array, len: number): void {
  let max = -Infinity;
  for (let i = 0; i < len; i++) {
    if (input[i] > max) max = input[i];
  }
  let sumExp = 0;
  for (let i = 0; i < len; i++) {
    sumExp += Math.exp(input[i] - max);
  }
  const logSumExp = max + Math.log(sumExp);
  for (let i = 0; i < len; i++) {
    output[i] = input[i] - logSumExp;
  }
}

// ---- Model ----

interface ConvLayer {
  weight: Float32Array;
  bias: Float32Array;
}

interface LinearLayer {
  weight: Float32Array;
  bias: Float32Array;
  outFeatures: number;
  inFeatures: number;
}

interface ModelConfig {
  numFilters: number;
  numBlocks: number;
  policyFilters: number;
  valueFilters: number;
  valueHidden: number;
  policyHidden: number;
}

export class PureTSModel {
  readonly config: ModelConfig;

  // Layers
  private inputConv: ConvLayer;
  private resBlocks: Array<{ conv1: ConvLayer; conv2: ConvLayer }>;
  private policyConv: ConvLayer;
  private valueConv: ConvLayer;
  private policyFc: LinearLayer;
  private policyFc1: LinearLayer | null; // bottleneck hidden layer
  private valueFc1: LinearLayer;
  private valueFc2: LinearLayer;

  // Pre-allocated scratch buffers — zero allocations during forward()
  private inputColBuf: Float32Array;
  private colBuf: Float32Array;
  private towerBuf1: Float32Array;
  private towerBuf2: Float32Array;
  private headBuf: Float32Array;
  private policyHiddenBuf: Float32Array | null;
  private policyLogitsBuf: Float32Array;
  private policyOutBuf: Float32Array;
  private valueHiddenBuf: Float32Array;
  private valueOutBuf: Float32Array;

  constructor(config: ModelConfig, weights: Map<string, WeightTensor>) {
    this.config = config;
    const f = config.numFilters;

    // Sort conv tensors by ONNX node number
    const convTensors = [...weights.values()]
      .filter(t => t.name.startsWith('onnx::Conv_'))
      .sort((a, b) => {
        const numA = parseInt(a.name.split('_').pop()!);
        const numB = parseInt(b.name.split('_').pop()!);
        return numA - numB;
      });

    // Pair consecutive conv tensors as (weight, bias). If the export omitted
    // bias tensors (some seed models were exported with bias=False), bail out
    // with a clear error instead of crashing on undefined.data.
    if (convTensors.length % 2 !== 0) {
      throw new Error(
        `ONNX has ${convTensors.length} conv tensors (odd count) — expected paired weight+bias. ` +
        `This model was likely exported with bias=False; pick a different model.`,
      );
    }
    const convPairs: ConvLayer[] = [];
    for (let i = 0; i < convTensors.length; i += 2) {
      convPairs.push({
        weight: convTensors[i].data,
        bias: convTensors[i + 1].data,
      });
    }

    // Assign conv layers in model order:
    // 0: input conv
    // 1..2*numBlocks: residual blocks (2 convs each)
    // 2*numBlocks+1: policy conv
    // 2*numBlocks+2: value conv
    // 2*numBlocks+3: difficulty conv (ignored)
    let idx = 0;
    this.inputConv = convPairs[idx++];

    this.resBlocks = [];
    for (let b = 0; b < config.numBlocks; b++) {
      this.resBlocks.push({
        conv1: convPairs[idx++],
        conv2: convPairs[idx++],
      });
    }

    this.policyConv = convPairs[idx++];
    this.valueConv = convPairs[idx++];
    // difficulty conv skipped (idx++)

    // FC layers — these have proper names
    const getFC = (prefix: string): LinearLayer => {
      const w = weights.get(`${prefix}.weight`)!;
      const b = weights.get(`${prefix}.bias`)!;
      return {
        weight: w.data,
        bias: b.data,
        outFeatures: w.shape[0],
        inFeatures: w.shape[1],
      };
    };

    if (config.policyHidden > 0) {
      this.policyFc1 = getFC('policy_fc1');
      this.policyFc = getFC('policy_fc2');
    } else {
      this.policyFc1 = null;
      this.policyFc = getFC('policy_fc');
    }
    this.valueFc1 = getFC('value_fc1');
    this.valueFc2 = getFC('value_fc2');

    // Pre-allocate ALL scratch buffers — forward() does zero allocations
    this.inputColBuf = new Float32Array(7 * 9 * HW);
    this.colBuf = new Float32Array(f * 9 * HW);
    this.towerBuf1 = new Float32Array(f * HW);
    this.towerBuf2 = new Float32Array(f * HW);
    this.headBuf = new Float32Array(Math.max(
      config.policyFilters * HW,
      config.valueFilters * HW,
    ));
    this.policyHiddenBuf = this.policyFc1
      ? new Float32Array(this.policyFc1.outFeatures) : null;
    this.policyLogitsBuf = new Float32Array(this.policyFc.outFeatures);
    this.policyOutBuf = new Float32Array(this.policyFc.outFeatures);
    this.valueHiddenBuf = new Float32Array(this.valueFc1.outFeatures);
    this.valueOutBuf = new Float32Array(1);
  }

  /**
   * Run forward pass.
   * @param input Float32Array of shape (7, 8, 7) = 392 elements in CHW order
   * @returns policy (log-probs, 3137 elements) and value (scalar in [-1, 1])
   *
   * NOTE: The returned policy Float32Array is reused across calls.
   * Callers must copy it if they need to retain it.
   */
  forward(input: Float32Array): { policy: Float32Array; value: number } {
    const f = this.config.numFilters;

    // Input conv (7 → f) + fused BN + ReLU
    conv3x3relu(input, this.inputConv.weight, this.inputConv.bias,
      7, f, ROWS, COLS, this.inputColBuf, this.towerBuf1);

    // Residual tower
    let current = this.towerBuf1;
    let scratch = this.towerBuf2;
    const col = this.colBuf;
    const fHW = f * HW;

    for (const block of this.resBlocks) {
      // conv1 + fused BN + ReLU
      conv3x3relu(current, block.conv1.weight, block.conv1.bias,
        f, f, ROWS, COLS, col, scratch);

      // conv2 + fused BN (no ReLU yet)
      conv3x3noRelu(scratch, block.conv2.weight, block.conv2.bias,
        f, f, ROWS, COLS, col, scratch);

      // Residual add + ReLU (in-place into scratch)
      for (let i = 0; i < fHW; i++) {
        scratch[i] = Math.max(0, scratch[i] + current[i]);
      }

      // Swap buffers
      const tmp = current;
      current = scratch;
      scratch = tmp;
    }

    // Tower output is in `current`

    // --- Policy head ---
    const pf = this.config.policyFilters;
    const policyHead = this.headBuf;
    conv1x1(current, this.policyConv.weight, this.policyConv.bias,
      f, pf, HW, policyHead);

    const policyFlat = policyHead.subarray(0, pf * HW);
    const policyLogits = this.policyLogitsBuf;

    if (this.policyFc1) {
      // Bottleneck: FC1 + ReLU + FC2
      const hidden = this.policyHiddenBuf!;
      linear(policyFlat, this.policyFc1.weight, this.policyFc1.bias,
        this.policyFc1.outFeatures, this.policyFc1.inFeatures, hidden);
      for (let i = 0; i < hidden.length; i++) {
        hidden[i] = Math.max(0, hidden[i]);
      }
      linear(hidden, this.policyFc.weight, this.policyFc.bias,
        this.policyFc.outFeatures, this.policyFc.inFeatures, policyLogits);
    } else {
      linear(policyFlat, this.policyFc.weight, this.policyFc.bias,
        this.policyFc.outFeatures, this.policyFc.inFeatures, policyLogits);
    }

    // Log-softmax
    const policy = this.policyOutBuf;
    logSoftmax(policyLogits, policy, policyLogits.length);

    // --- Value head ---
    const vf = this.config.valueFilters;
    const valueHead = this.headBuf;
    conv1x1(current, this.valueConv.weight, this.valueConv.bias,
      f, vf, HW, valueHead);

    const valueFlat = valueHead.subarray(0, vf * HW);

    // FC1 + ReLU
    const vh = this.valueHiddenBuf;
    linear(valueFlat, this.valueFc1.weight, this.valueFc1.bias,
      this.valueFc1.outFeatures, this.valueFc1.inFeatures, vh);
    for (let i = 0; i < vh.length; i++) {
      vh[i] = Math.max(0, vh[i]);
    }

    // FC2 + tanh
    const vOut = this.valueOutBuf;
    linear(vh, this.valueFc2.weight, this.valueFc2.bias,
      this.valueFc2.outFeatures, this.valueFc2.inFeatures, vOut);
    const value = Math.tanh(vOut[0]);

    return { policy, value };
  }
}

/**
 * Create a PureTSModel from an ONNX model buffer.
 * Parses the protobuf to extract weight tensors and infers the model config.
 */
export function createModelFromOnnx(buffer: ArrayBuffer): PureTSModel {
  const tensors = parseOnnxWeights(buffer);
  const tensorMap = new Map<string, WeightTensor>();
  for (const t of tensors) {
    tensorMap.set(t.name, t);
  }

  // Infer config from weight shapes
  const convTensors = tensors.filter(t => t.name.startsWith('onnx::Conv_'));
  const firstConv = convTensors.find(t => t.shape.length === 4 && t.shape[1] !== t.shape[0]);
  const numFilters = firstConv ? firstConv.shape[0] : 96;

  // Count res block convs: total conv3x3 pairs minus input conv = 2 * numBlocks
  const conv3x3Count = convTensors.filter(t => t.shape.length === 4 && t.shape[2] === 3).length;
  const numBlocks = (conv3x3Count - 1) / 2; // subtract input conv

  // Policy/value filter counts from 1x1 conv shapes
  const conv1x1s = convTensors.filter(t => t.shape.length === 4 && t.shape[2] === 1);
  const policyFilters = conv1x1s[0]?.shape[0] ?? 2;
  const valueFilters = conv1x1s[1]?.shape[0] ?? 1;

  // FC hidden sizes
  const valueFc1 = tensorMap.get('value_fc1.weight');
  const valueHidden = valueFc1 ? valueFc1.shape[0] : 256;

  const policyFc1 = tensorMap.get('policy_fc1.weight');
  const policyHidden = policyFc1 ? policyFc1.shape[0] : 0;

  const config: ModelConfig = {
    numFilters,
    numBlocks,
    policyFilters,
    valueFilters,
    valueHidden,
    policyHidden,
  };

  return new PureTSModel(config, tensorMap);
}
