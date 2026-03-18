/**
 * WebGL-accelerated neural network inference for RazzleNet.
 *
 * Uses WebGLGemm for GPU-accelerated matrix multiplication while keeping
 * all other operations (im2col, bias, ReLU, softmax) in plain TypeScript.
 * This gives correct results (unlike ONNX Runtime's WebGL backend) with
 * GPU acceleration for the compute-heavy GEMM operations.
 *
 * The interface matches PureTSModel — same forward() signature and output.
 */

import { type WeightTensor, parseOnnxWeights } from './onnxWeights';
import { WebGLGemm } from './webglGemm';

const ROWS = 8;
const COLS = 7;
const HW = ROWS * COLS; // 56

// ---- Lightweight ops (CPU) ----

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
        const colRowOff = (c * 9 + kh * 3 + kw) * colW;
        const hStart = Math.max(0, 1 - kh);
        const hEnd = Math.min(H, H + 1 - kh);
        for (let h = hStart; h < hEnd; h++) {
          const rowOutOff = colRowOff + h * W;
          const rowInOff = inOff + (h + kh - 1) * W;
          const wStart = Math.max(0, 1 - kw);
          const wEnd = Math.min(W, W + 1 - kw);
          for (let w = wStart; w < wEnd; w++) {
            col[rowOutOff + w] = input[rowInOff + w + kw - 1];
          }
        }
      }
    }
  }
}

function addBiasRelu(output: Float32Array, bias: Float32Array, outC: number, hw: number): void {
  for (let oc = 0; oc < outC; oc++) {
    const b = bias[oc];
    const off = oc * hw;
    for (let i = 0; i < hw; i++) {
      output[off + i] = Math.max(0, output[off + i] + b);
    }
  }
}

function addBias(output: Float32Array, bias: Float32Array, outC: number, hw: number): void {
  for (let oc = 0; oc < outC; oc++) {
    const b = bias[oc];
    const off = oc * hw;
    for (let i = 0; i < hw; i++) {
      output[off + i] += b;
    }
  }
}

function linearCPU(
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
  weightKey: string; // key for GPU texture
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

export class WebGLModel {
  readonly config: ModelConfig;
  private gpu: WebGLGemm;

  // Layers
  private inputConv: ConvLayer;
  private resBlocks: Array<{ conv1: ConvLayer; conv2: ConvLayer }>;
  private policyConv: ConvLayer;
  private valueConv: ConvLayer;
  private policyFc: LinearLayer;
  private policyFc1: LinearLayer | null;
  private valueFc1: LinearLayer;
  private valueFc2: LinearLayer;

  // Pre-allocated scratch buffers
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

  constructor(config: ModelConfig, weights: Map<string, WeightTensor>, canvas?: HTMLCanvasElement | OffscreenCanvas) {
    this.config = config;
    this.gpu = new WebGLGemm(canvas);
    const f = config.numFilters;

    // Sort and pair conv tensors (same as PureTSModel)
    const convTensors = [...weights.values()]
      .filter(t => t.name.startsWith('onnx::Conv_'))
      .sort((a, b) => {
        const numA = parseInt(a.name.split('_').pop()!);
        const numB = parseInt(b.name.split('_').pop()!);
        return numA - numB;
      });

    const convPairs: ConvLayer[] = [];
    for (let i = 0; i < convTensors.length; i += 2) {
      const key = `conv_${i / 2}`;
      const layer: ConvLayer = {
        weight: convTensors[i].data,
        bias: convTensors[i + 1].data,
        weightKey: key,
      };
      // Upload weight to GPU
      this.gpu.uploadWeights(key, layer.weight, convTensors[i].shape[0], convTensors[i].data.length / convTensors[i].shape[0]);
      convPairs.push(layer);
    }

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

    // Upload 1x1 conv weights too
    this.gpu.uploadWeights(this.policyConv.weightKey, this.policyConv.weight, config.policyFilters, f);
    this.gpu.uploadWeights(this.valueConv.weightKey, this.valueConv.weight, config.valueFilters, f);

    // FC layers
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

    // Pre-allocate buffers
    this.inputColBuf = new Float32Array(7 * 9 * HW);
    this.colBuf = new Float32Array(f * 9 * HW);
    this.towerBuf1 = new Float32Array(f * HW);
    this.towerBuf2 = new Float32Array(f * HW);
    this.headBuf = new Float32Array(Math.max(config.policyFilters * HW, config.valueFilters * HW));
    this.policyHiddenBuf = this.policyFc1 ? new Float32Array(this.policyFc1.outFeatures) : null;
    this.policyLogitsBuf = new Float32Array(this.policyFc.outFeatures);
    this.policyOutBuf = new Float32Array(this.policyFc.outFeatures);
    this.valueHiddenBuf = new Float32Array(this.valueFc1.outFeatures);
    this.valueOutBuf = new Float32Array(1);
  }

  forward(input: Float32Array): { policy: Float32Array; value: number } {
    const f = this.config.numFilters;
    const gpu = this.gpu;

    // Input conv (7→f) via GPU GEMM + CPU bias/ReLU
    im2col3x3(input, 7, ROWS, COLS, this.inputColBuf);
    gpu.gemm(this.inputConv.weightKey, this.inputColBuf, this.towerBuf1, f, 7 * 9, HW);
    addBiasRelu(this.towerBuf1, this.inputConv.bias, f, HW);

    // Residual tower
    let current = this.towerBuf1;
    let scratch = this.towerBuf2;
    const col = this.colBuf;
    const fHW = f * HW;

    for (const block of this.resBlocks) {
      // conv1 + ReLU
      im2col3x3(current, f, ROWS, COLS, col);
      gpu.gemm(block.conv1.weightKey, col, scratch, f, f * 9, HW);
      addBiasRelu(scratch, block.conv1.bias, f, HW);

      // conv2 (no ReLU)
      im2col3x3(scratch, f, ROWS, COLS, col);
      gpu.gemm(block.conv2.weightKey, col, scratch, f, f * 9, HW);
      addBias(scratch, block.conv2.bias, f, HW);

      // Residual add + ReLU
      for (let i = 0; i < fHW; i++) {
        scratch[i] = Math.max(0, scratch[i] + current[i]);
      }

      const tmp = current;
      current = scratch;
      scratch = tmp;
    }

    // Policy head: 1x1 conv via GPU
    const pf = this.config.policyFilters;
    gpu.gemm(this.policyConv.weightKey, current, this.headBuf, pf, f, HW);
    addBiasRelu(this.headBuf, this.policyConv.bias, pf, HW);

    const policyFlat = this.headBuf.subarray(0, pf * HW);
    const policyLogits = this.policyLogitsBuf;

    if (this.policyFc1) {
      const hidden = this.policyHiddenBuf!;
      linearCPU(policyFlat, this.policyFc1.weight, this.policyFc1.bias,
        this.policyFc1.outFeatures, this.policyFc1.inFeatures, hidden);
      for (let i = 0; i < hidden.length; i++) hidden[i] = Math.max(0, hidden[i]);
      linearCPU(hidden, this.policyFc.weight, this.policyFc.bias,
        this.policyFc.outFeatures, this.policyFc.inFeatures, policyLogits);
    } else {
      linearCPU(policyFlat, this.policyFc.weight, this.policyFc.bias,
        this.policyFc.outFeatures, this.policyFc.inFeatures, policyLogits);
    }

    const policy = this.policyOutBuf;
    logSoftmax(policyLogits, policy, policyLogits.length);

    // Value head: 1x1 conv via GPU
    const vf = this.config.valueFilters;
    gpu.gemm(this.valueConv.weightKey, current, this.headBuf, vf, f, HW);
    addBiasRelu(this.headBuf, this.valueConv.bias, vf, HW);

    const valueFlat = this.headBuf.subarray(0, vf * HW);
    const vh = this.valueHiddenBuf;
    linearCPU(valueFlat, this.valueFc1.weight, this.valueFc1.bias,
      this.valueFc1.outFeatures, this.valueFc1.inFeatures, vh);
    for (let i = 0; i < vh.length; i++) vh[i] = Math.max(0, vh[i]);

    const vOut = this.valueOutBuf;
    linearCPU(vh, this.valueFc2.weight, this.valueFc2.bias,
      this.valueFc2.outFeatures, this.valueFc2.inFeatures, vOut);
    const value = Math.tanh(vOut[0]);

    return { policy, value };
  }

  dispose(): void {
    this.gpu.dispose();
  }
}

/**
 * Create a WebGLModel from an ONNX model buffer.
 */
export function createWebGLModelFromOnnx(buffer: ArrayBuffer, canvas?: HTMLCanvasElement | OffscreenCanvas): WebGLModel {
  const tensors = parseOnnxWeights(buffer);
  const tensorMap = new Map<string, WeightTensor>();
  for (const t of tensors) tensorMap.set(t.name, t);

  const convTensors = tensors.filter(t => t.name.startsWith('onnx::Conv_'));
  const firstConv = convTensors.find(t => t.shape.length === 4 && t.shape[1] !== t.shape[0]);
  const numFilters = firstConv ? firstConv.shape[0] : 96;

  const conv3x3Count = convTensors.filter(t => t.shape.length === 4 && t.shape[2] === 3).length;
  const numBlocks = (conv3x3Count - 1) / 2;

  const conv1x1s = convTensors.filter(t => t.shape.length === 4 && t.shape[2] === 1);
  const policyFilters = conv1x1s[0]?.shape[0] ?? 2;
  const valueFilters = conv1x1s[1]?.shape[0] ?? 1;

  const valueFc1 = tensorMap.get('value_fc1.weight');
  const valueHidden = valueFc1 ? valueFc1.shape[0] : 256;

  const policyFc1 = tensorMap.get('policy_fc1.weight');
  const policyHidden = policyFc1 ? policyFc1.shape[0] : 0;

  return new WebGLModel({
    numFilters, numBlocks, policyFilters, valueFilters, valueHidden, policyHidden,
  }, tensorMap, canvas);
}
