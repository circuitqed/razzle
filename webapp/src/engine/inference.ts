/**
 * Pure TypeScript neural network inference for RazzleNet.
 *
 * Implements the AlphaZero-style architecture without WASM or ONNX Runtime.
 * Uses im2col + GEMM for cache-friendly convolution. All operations use
 * Float32Array for performance.
 *
 * This is the iOS-compatible inference path. On desktop, OnnxEvaluator
 * is preferred for its WASM SIMD acceleration.
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
 * Uses MKN loop order for better cache behavior on B.
 */
function gemm(
  A: Float32Array, B: Float32Array, C: Float32Array,
  M: number, K: number, N: number,
): void {
  C.fill(0);
  for (let m = 0; m < M; m++) {
    const aRowOff = m * K;
    const cRowOff = m * N;
    for (let k = 0; k < K; k++) {
      const a = A[aRowOff + k];
      if (a === 0) continue;
      const bRowOff = k * N;
      for (let n = 0; n < N; n++) {
        C[cRowOff + n] += a * B[bRowOff + n];
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
        for (let h = 0; h < H; h++) {
          const ih = h + kh - 1;
          if (ih < 0 || ih >= H) continue;
          for (let w = 0; w < W; w++) {
            const iw = w + kw - 1;
            if (iw < 0 || iw >= W) continue;
            col[colRowOff + h * W + w] = input[inOff + ih * W + iw];
          }
        }
      }
    }
  }
}

/**
 * Conv2d (3x3, padding=1) + bias + optional ReLU.
 * BN is already fused into weight/bias by the ONNX exporter.
 */
function conv3x3(
  input: Float32Array, weight: Float32Array, bias: Float32Array,
  inC: number, outC: number, H: number, W: number,
  col: Float32Array, output: Float32Array,
  relu: boolean,
): void {
  // im2col: input (inC, H, W) → col (inC*9, H*W)
  im2col3x3(input, inC, H, W, col);

  // GEMM: weight (outC, inC*9) @ col (inC*9, H*W) → output (outC, H*W)
  gemm(weight, col, output, outC, inC * 9, H * W);

  // Add bias (+ optional ReLU)
  for (let oc = 0; oc < outC; oc++) {
    const b = bias[oc];
    const off = oc * H * W;
    for (let i = 0; i < H * W; i++) {
      const v = output[off + i] + b;
      output[off + i] = relu ? Math.max(0, v) : v;
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
  // weight: (outC, inC), input: (inC, HW) → output: (outC, HW)
  gemm(weight, input, output, outC, inC, HW);

  // Add bias + ReLU
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
  for (let m = 0; m < outFeatures; m++) {
    let sum = bias[m];
    const wOff = m * inFeatures;
    for (let n = 0; n < inFeatures; n++) {
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

  // Pre-allocated scratch buffers for inference
  private colBuf: Float32Array;
  private towerBuf1: Float32Array;
  private towerBuf2: Float32Array;
  private headBuf: Float32Array;

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

    // Pair consecutive conv tensors as (weight, bias)
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

    // Pre-allocate scratch buffers
    this.colBuf = new Float32Array(f * 9 * HW);
    this.towerBuf1 = new Float32Array(f * HW);
    this.towerBuf2 = new Float32Array(f * HW);
    this.headBuf = new Float32Array(Math.max(
      config.policyFilters * HW,
      config.valueFilters * HW,
    ));
  }

  /**
   * Run forward pass.
   * @param input Float32Array of shape (7, 8, 7) = 392 elements in CHW order
   * @returns policy (log-probs, 3137 elements) and value (scalar in [-1, 1])
   */
  forward(input: Float32Array): { policy: Float32Array; value: number } {
    const f = this.config.numFilters;
    const inC = 7; // input channels
    const col = this.colBuf;

    // Input conv (7 → f) + fused BN + ReLU
    // Need larger col buffer for input conv (7 channels vs f channels)
    const inputCol = new Float32Array(inC * 9 * HW);
    conv3x3(input, this.inputConv.weight, this.inputConv.bias,
      inC, f, ROWS, COLS, inputCol, this.towerBuf1, true);

    // Residual tower
    let current = this.towerBuf1;
    let scratch = this.towerBuf2;

    for (const block of this.resBlocks) {
      // conv1 + fused BN + ReLU
      conv3x3(current, block.conv1.weight, block.conv1.bias,
        f, f, ROWS, COLS, col, scratch, true);

      // conv2 + fused BN (no ReLU yet)
      conv3x3(scratch, block.conv2.weight, block.conv2.bias,
        f, f, ROWS, COLS, col, scratch, false);

      // Residual add + ReLU (in-place into scratch)
      for (let i = 0; i < f * HW; i++) {
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

    // Flatten: (pf, H, W) → (pf * H * W)
    const policyFlat = policyHead.subarray(0, pf * HW);

    let policyLogits: Float32Array;
    if (this.policyFc1) {
      // Bottleneck: FC1 + ReLU + FC2
      const hidden = new Float32Array(this.policyFc1.outFeatures);
      linear(policyFlat, this.policyFc1.weight, this.policyFc1.bias,
        this.policyFc1.outFeatures, this.policyFc1.inFeatures, hidden);
      for (let i = 0; i < hidden.length; i++) {
        hidden[i] = Math.max(0, hidden[i]);
      }
      policyLogits = new Float32Array(this.policyFc.outFeatures);
      linear(hidden, this.policyFc.weight, this.policyFc.bias,
        this.policyFc.outFeatures, this.policyFc.inFeatures, policyLogits);
    } else {
      // Direct FC
      policyLogits = new Float32Array(this.policyFc.outFeatures);
      linear(policyFlat, this.policyFc.weight, this.policyFc.bias,
        this.policyFc.outFeatures, this.policyFc.inFeatures, policyLogits);
    }

    // Log-softmax
    const policy = new Float32Array(policyLogits.length);
    logSoftmax(policyLogits, policy, policyLogits.length);

    // --- Value head ---
    const vf = this.config.valueFilters;
    const valueHead = this.headBuf;
    conv1x1(current, this.valueConv.weight, this.valueConv.bias,
      f, vf, HW, valueHead);

    // Flatten: (vf, H, W) → (vf * H * W)
    const valueFlat = valueHead.subarray(0, vf * HW);

    // FC1 + ReLU
    const vh = new Float32Array(this.valueFc1.outFeatures);
    linear(valueFlat, this.valueFc1.weight, this.valueFc1.bias,
      this.valueFc1.outFeatures, this.valueFc1.inFeatures, vh);
    for (let i = 0; i < vh.length; i++) {
      vh[i] = Math.max(0, vh[i]);
    }

    // FC2 + tanh
    const vOut = new Float32Array(1);
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

  console.log('[inference] Model config inferred:', config);

  return new PureTSModel(config, tensorMap);
}
