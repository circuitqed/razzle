/**
 * GPU-resident neural network forward pass using WebGL2.
 *
 * Keeps all intermediate activations on the GPU as textures.
 * Only reads back the final policy logits and value head output.
 * This eliminates the per-layer readPixels bottleneck that made
 * the previous WebGL GEMM approach slower than CPU.
 *
 * Architecture:
 * - Conv3x3 as a single fragment shader (no im2col needed)
 * - Conv1x1 as a fragment shader
 * - Residual add+ReLU as a fragment shader
 * - Render-to-texture between layers (no CPU round-trips)
 * - FC layers on CPU (small enough that GPU overhead isn't worth it)
 *
 * Data layout:
 * - Activation textures: (HW, C) — texel (hw, c) = value at channel c, spatial hw
 * - Weight textures: varies by op (see individual shaders)
 * - Bias textures: (1, C) — texel (0, c) = bias[c]
 */

import { type WeightTensor, parseOnnxWeights } from './onnxWeights';

const ROWS = 8;
const COLS = 7;
const HW = ROWS * COLS; // 56

// ---- Shader sources ----

const VERT_SRC = `#version 300 es
in vec2 aPosition;
void main() { gl_Position = vec4(aPosition, 0.0, 1.0); }
`;

// Conv 3x3 with padding=1 + bias + optional ReLU
// Supports batching: input texture is (HW*N, inC) where N = batch size.
// uSingleHW = HW (56) tells the shader where batch boundaries are.
// Padding does NOT bleed across batch boundaries.
const CONV3X3_RELU_SRC = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uInput;
uniform sampler2D uWeight;
uniform sampler2D uBias;
uniform int uInC;
uniform int uH;
uniform int uW;
uniform int uSingleHW;
out vec4 fragColor;
void main() {
  int globalHW = int(gl_FragCoord.x);
  int oc = int(gl_FragCoord.y);
  int batchStart = (globalHW / uSingleHW) * uSingleHW;
  int localHW = globalHW - batchStart;
  int h = localHW / uW;
  int w = localHW - h * uW;
  float sum = texelFetch(uBias, ivec2(oc, 0), 0).r;
  for (int ic = 0; ic < uInC; ic++) {
    int wBase = ic * 9;
    for (int kh = 0; kh < 3; kh++) {
      int ih = h + kh - 1;
      if (ih < 0 || ih >= uH) continue;
      for (int kw = 0; kw < 3; kw++) {
        int iw = w + kw - 1;
        if (iw < 0 || iw >= uW) continue;
        sum += texelFetch(uWeight, ivec2(wBase + kh * 3 + kw, oc), 0).r
             * texelFetch(uInput, ivec2(batchStart + ih * uW + iw, ic), 0).r;
      }
    }
  }
  fragColor = vec4(max(0.0, sum), 0.0, 0.0, 1.0);
}
`;

const CONV3X3_NORELU_SRC = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uInput;
uniform sampler2D uWeight;
uniform sampler2D uBias;
uniform int uInC;
uniform int uH;
uniform int uW;
uniform int uSingleHW;
out vec4 fragColor;
void main() {
  int globalHW = int(gl_FragCoord.x);
  int oc = int(gl_FragCoord.y);
  int batchStart = (globalHW / uSingleHW) * uSingleHW;
  int localHW = globalHW - batchStart;
  int h = localHW / uW;
  int w = localHW - h * uW;
  float sum = texelFetch(uBias, ivec2(oc, 0), 0).r;
  for (int ic = 0; ic < uInC; ic++) {
    int wBase = ic * 9;
    for (int kh = 0; kh < 3; kh++) {
      int ih = h + kh - 1;
      if (ih < 0 || ih >= uH) continue;
      for (int kw = 0; kw < 3; kw++) {
        int iw = w + kw - 1;
        if (iw < 0 || iw >= uW) continue;
        sum += texelFetch(uWeight, ivec2(wBase + kh * 3 + kw, oc), 0).r
             * texelFetch(uInput, ivec2(batchStart + ih * uW + iw, ic), 0).r;
      }
    }
  }
  fragColor = vec4(sum, 0.0, 0.0, 1.0);
}
`;

// Residual add + ReLU: out = max(0, a + b)
const RESIDUAL_RELU_SRC = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uA;
uniform sampler2D uB;
out vec4 fragColor;
void main() {
  ivec2 pos = ivec2(gl_FragCoord.xy);
  float a = texelFetch(uA, pos, 0).r;
  float b = texelFetch(uB, pos, 0).r;
  fragColor = vec4(max(0.0, a + b), 0.0, 0.0, 1.0);
}
`;

// Conv 1x1 + bias + ReLU
// Input: (HW, inC)  Weight: (inC, outC)  Bias: (1, outC)
// Output: (HW, outC)
const CONV1X1_RELU_SRC = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uInput;
uniform sampler2D uWeight;
uniform sampler2D uBias;
uniform int uInC;
out vec4 fragColor;
void main() {
  int hw = int(gl_FragCoord.x);
  int oc = int(gl_FragCoord.y);
  float sum = texelFetch(uBias, ivec2(oc, 0), 0).r;
  for (int ic = 0; ic < uInC; ic++) {
    sum += texelFetch(uWeight, ivec2(ic, oc), 0).r
         * texelFetch(uInput, ivec2(hw, ic), 0).r;
  }
  fragColor = vec4(max(0.0, sum), 0.0, 0.0, 1.0);
}
`;

// ---- RGBA-packed shader sources (4 channels per texel) ----

// Conv 3x3 with RGBA input → RGBA output + bias + ReLU
// Each fragment writes 4 output channels. Input is RGBA-packed (inC/4 height).
// Weight texture: (inC*9, outC/4) RGBA32F
// Bias texture: (outC/4, 1) RGBA32F
const CONV3X3_RGBA_RELU_SRC = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uInput;
uniform sampler2D uWeight;
uniform sampler2D uBias;
uniform int uInC4;
uniform int uH;
uniform int uW;
uniform int uSingleHW;
out vec4 fragColor;
void main() {
  int globalHW = int(gl_FragCoord.x);
  int oc4 = int(gl_FragCoord.y);
  int batchStart = (globalHW / uSingleHW) * uSingleHW;
  int localHW = globalHW - batchStart;
  int h = localHW / uW;
  int w = localHW - h * uW;
  vec4 sum = texelFetch(uBias, ivec2(oc4, 0), 0);
  for (int ic4 = 0; ic4 < uInC4; ic4++) {
    for (int kh = 0; kh < 3; kh++) {
      int ih = h + kh - 1;
      if (ih < 0 || ih >= uH) continue;
      for (int kw = 0; kw < 3; kw++) {
        int iw = w + kw - 1;
        if (iw < 0 || iw >= uW) continue;
        vec4 inp4 = texelFetch(uInput, ivec2(batchStart + ih * uW + iw, ic4), 0);
        int kIdx = kh * 3 + kw;
        vec4 w0 = texelFetch(uWeight, ivec2((ic4 * 4 + 0) * 9 + kIdx, oc4), 0);
        vec4 w1 = texelFetch(uWeight, ivec2((ic4 * 4 + 1) * 9 + kIdx, oc4), 0);
        vec4 w2 = texelFetch(uWeight, ivec2((ic4 * 4 + 2) * 9 + kIdx, oc4), 0);
        vec4 w3 = texelFetch(uWeight, ivec2((ic4 * 4 + 3) * 9 + kIdx, oc4), 0);
        sum += inp4.x * w0 + inp4.y * w1 + inp4.z * w2 + inp4.w * w3;
      }
    }
  }
  fragColor = max(vec4(0.0), sum);
}
`;

// Conv 3x3 with RGBA input → RGBA output + bias, no ReLU
const CONV3X3_RGBA_NORELU_SRC = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uInput;
uniform sampler2D uWeight;
uniform sampler2D uBias;
uniform int uInC4;
uniform int uH;
uniform int uW;
uniform int uSingleHW;
out vec4 fragColor;
void main() {
  int globalHW = int(gl_FragCoord.x);
  int oc4 = int(gl_FragCoord.y);
  int batchStart = (globalHW / uSingleHW) * uSingleHW;
  int localHW = globalHW - batchStart;
  int h = localHW / uW;
  int w = localHW - h * uW;
  vec4 sum = texelFetch(uBias, ivec2(oc4, 0), 0);
  for (int ic4 = 0; ic4 < uInC4; ic4++) {
    for (int kh = 0; kh < 3; kh++) {
      int ih = h + kh - 1;
      if (ih < 0 || ih >= uH) continue;
      for (int kw = 0; kw < 3; kw++) {
        int iw = w + kw - 1;
        if (iw < 0 || iw >= uW) continue;
        vec4 inp4 = texelFetch(uInput, ivec2(batchStart + ih * uW + iw, ic4), 0);
        int kIdx = kh * 3 + kw;
        vec4 w0 = texelFetch(uWeight, ivec2((ic4 * 4 + 0) * 9 + kIdx, oc4), 0);
        vec4 w1 = texelFetch(uWeight, ivec2((ic4 * 4 + 1) * 9 + kIdx, oc4), 0);
        vec4 w2 = texelFetch(uWeight, ivec2((ic4 * 4 + 2) * 9 + kIdx, oc4), 0);
        vec4 w3 = texelFetch(uWeight, ivec2((ic4 * 4 + 3) * 9 + kIdx, oc4), 0);
        sum += inp4.x * w0 + inp4.y * w1 + inp4.z * w2 + inp4.w * w3;
      }
    }
  }
  fragColor = sum;
}
`;

// Conv 3x3: R32F input (7 channels) → RGBA output + bias + ReLU
// Input conv: input channels not divisible by 4, read one at a time from R32F
// Weight texture: (inC*9, outC/4) RGBA32F
// Bias texture: (outC/4, 1) RGBA32F
const CONV3X3_R2RGBA_RELU_SRC = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uInput;
uniform sampler2D uWeight;
uniform sampler2D uBias;
uniform int uInC;
uniform int uH;
uniform int uW;
uniform int uSingleHW;
out vec4 fragColor;
void main() {
  int globalHW = int(gl_FragCoord.x);
  int oc4 = int(gl_FragCoord.y);
  int batchStart = (globalHW / uSingleHW) * uSingleHW;
  int localHW = globalHW - batchStart;
  int h = localHW / uW;
  int w = localHW - h * uW;
  vec4 sum = texelFetch(uBias, ivec2(oc4, 0), 0);
  for (int ic = 0; ic < uInC; ic++) {
    for (int kh = 0; kh < 3; kh++) {
      int ih = h + kh - 1;
      if (ih < 0 || ih >= uH) continue;
      for (int kw = 0; kw < 3; kw++) {
        int iw = w + kw - 1;
        if (iw < 0 || iw >= uW) continue;
        float inp = texelFetch(uInput, ivec2(batchStart + ih * uW + iw, ic), 0).r;
        vec4 w4 = texelFetch(uWeight, ivec2(ic * 9 + kh * 3 + kw, oc4), 0);
        sum += inp * w4;
      }
    }
  }
  fragColor = max(vec4(0.0), sum);
}
`;

// Residual add + ReLU for RGBA textures: out = max(0, a + b)
const RESIDUAL_RGBA_RELU_SRC = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uA;
uniform sampler2D uB;
out vec4 fragColor;
void main() {
  ivec2 pos = ivec2(gl_FragCoord.xy);
  vec4 a = texelFetch(uA, pos, 0);
  vec4 b = texelFetch(uB, pos, 0);
  fragColor = max(vec4(0.0), a + b);
}
`;

// Conv 1x1: RGBA input → R32F output + bias + ReLU
// For policy/value heads where outC is small (2 or 4), not RGBA-packed
// Input: (HW, inC/4) RGBA32F   Weight: (inC, outC) R32F   Bias: (outC, 1) R32F
// Output: (HW, outC) R32F
const CONV1X1_RGBA_TO_R_RELU_SRC = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uInput;
uniform sampler2D uWeight;
uniform sampler2D uBias;
uniform int uInC4;
out vec4 fragColor;
void main() {
  int hw = int(gl_FragCoord.x);
  int oc = int(gl_FragCoord.y);
  float sum = texelFetch(uBias, ivec2(oc, 0), 0).r;
  for (int ic4 = 0; ic4 < uInC4; ic4++) {
    vec4 inp4 = texelFetch(uInput, ivec2(hw, ic4), 0);
    sum += texelFetch(uWeight, ivec2(ic4 * 4 + 0, oc), 0).r * inp4.x
         + texelFetch(uWeight, ivec2(ic4 * 4 + 1, oc), 0).r * inp4.y
         + texelFetch(uWeight, ivec2(ic4 * 4 + 2, oc), 0).r * inp4.z
         + texelFetch(uWeight, ivec2(ic4 * 4 + 3, oc), 0).r * inp4.w;
  }
  fragColor = vec4(max(0.0, sum), 0.0, 0.0, 1.0);
}
`;

// ---- GPU Context ----

interface ConvLayerGPU {
  weightTex: WebGLTexture;
  biasTex: WebGLTexture;
  inC: number;
  outC: number;
}

// RGBA-packed conv layer: weights and biases store 4 output channels per texel
interface ConvLayerRGBA {
  weightTex: WebGLTexture; // (inC*kSize, outC/4) RGBA32F
  biasTex: WebGLTexture;   // (outC/4, 1) RGBA32F
  inC: number;
  outC: number;
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

interface ShaderProgram {
  program: WebGLProgram;
  uniforms: Record<string, WebGLUniformLocation>;
}

export class GPUForwardPass {
  readonly config: ModelConfig;
  private gl: WebGL2RenderingContext;

  // Shader programs (R32F — kept for backward compat)
  private conv3x3Relu: ShaderProgram;
  private conv3x3NoRelu: ShaderProgram;
  private residualRelu: ShaderProgram;
  private conv1x1Relu: ShaderProgram;

  // Shader programs (RGBA-packed)
  private conv3x3RgbaRelu: ShaderProgram;
  private conv3x3RgbaNoRelu: ShaderProgram;
  private conv3x3R2RgbaRelu: ShaderProgram;
  private residualRgbaRelu: ShaderProgram;
  private conv1x1RgbaToRRelu: ShaderProgram;

  // Full-screen quad
  private vao: WebGLVertexArrayObject;

  // Model layers — R32F (kept for backward compat)
  private inputConv: ConvLayerGPU;
  private resBlocks: Array<{ conv1: ConvLayerGPU; conv2: ConvLayerGPU }>;
  private policyConv: ConvLayerGPU;
  private valueConv: ConvLayerGPU;

  // Model layers — RGBA-packed for tower convolutions
  private inputConvRgba: ConvLayerRGBA;
  private resBlocksRgba: Array<{ conv1: ConvLayerRGBA; conv2: ConvLayerRGBA }>;
  private policyConvRgba: ConvLayerRGBA; // Conv1x1 RGBA→R: weight R32F, bias R32F
  private valueConvRgba: ConvLayerRGBA;  // Conv1x1 RGBA→R: weight R32F, bias R32F

  // FC layers (CPU — small enough that GPU overhead isn't worth it)
  private policyFc: LinearLayer;
  private policyFc1: LinearLayer | null;
  private valueFc1: LinearLayer;
  private valueFc2: LinearLayer;

  // Activation textures (ping-pong) — stay on GPU (R32F, kept for backward compat)
  private actTexA: WebGLTexture;
  private actTexB: WebGLTexture;
  private actFbA: WebGLFramebuffer;
  private actFbB: WebGLFramebuffer;

  // RGBA activation textures (ping-pong) — (HW*MAX_BATCH, maxC/4) RGBA32F
  private actTexRgbaA: WebGLTexture;
  private actTexRgbaB: WebGLTexture;
  private actFbRgbaA: WebGLFramebuffer;
  private actFbRgbaB: WebGLFramebuffer;

  // Head output texture + framebuffer (for policy/value conv) — R32F
  private headTex: WebGLTexture;
  private headFb: WebGLFramebuffer;

  // RGBA head texture (temporary for residual blocks)
  private headTexRgba: WebGLTexture;
  private headFbRgba: WebGLFramebuffer;

  // Input texture (uploaded each forward pass)
  private inputTex: WebGLTexture;
  private inputFb: WebGLFramebuffer;

  // CPU buffers for readback and FC layers
  private readBuf: Float32Array;
  private policyHiddenBuf: Float32Array | null;
  private policyLogitsBuf: Float32Array;
  private policyOutBuf: Float32Array;
  private valueHiddenBuf: Float32Array;
  private valueOutBuf: Float32Array;

  constructor(config: ModelConfig, weights: Map<string, WeightTensor>, canvas?: HTMLCanvasElement | OffscreenCanvas) {
    this.config = config;
    const f = config.numFilters;

    // Create WebGL2 context
    const cvs = canvas ?? (typeof OffscreenCanvas !== 'undefined'
      ? new OffscreenCanvas(1, 1)
      : document.createElement('canvas'));

    const gl = cvs.getContext('webgl2', {
      alpha: false, depth: false, stencil: false, antialias: false,
      premultipliedAlpha: false, preserveDrawingBuffer: false,
    }) as WebGL2RenderingContext | null;

    if (!gl) throw new Error('WebGL2 not available');
    if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('EXT_color_buffer_float not available');
    this.gl = gl;

    // Compile shaders (R32F — kept for backward compat)
    this.conv3x3Relu = this.createShaderProgram(CONV3X3_RELU_SRC, ['uInput', 'uWeight', 'uBias', 'uInC', 'uH', 'uW', 'uSingleHW']);
    this.conv3x3NoRelu = this.createShaderProgram(CONV3X3_NORELU_SRC, ['uInput', 'uWeight', 'uBias', 'uInC', 'uH', 'uW', 'uSingleHW']);
    this.residualRelu = this.createShaderProgram(RESIDUAL_RELU_SRC, ['uA', 'uB']);
    this.conv1x1Relu = this.createShaderProgram(CONV1X1_RELU_SRC, ['uInput', 'uWeight', 'uBias', 'uInC']);

    // Compile shaders (RGBA-packed)
    this.conv3x3RgbaRelu = this.createShaderProgram(CONV3X3_RGBA_RELU_SRC, ['uInput', 'uWeight', 'uBias', 'uInC4', 'uH', 'uW', 'uSingleHW']);
    this.conv3x3RgbaNoRelu = this.createShaderProgram(CONV3X3_RGBA_NORELU_SRC, ['uInput', 'uWeight', 'uBias', 'uInC4', 'uH', 'uW', 'uSingleHW']);
    this.conv3x3R2RgbaRelu = this.createShaderProgram(CONV3X3_R2RGBA_RELU_SRC, ['uInput', 'uWeight', 'uBias', 'uInC', 'uH', 'uW', 'uSingleHW']);
    this.residualRgbaRelu = this.createShaderProgram(RESIDUAL_RGBA_RELU_SRC, ['uA', 'uB']);
    this.conv1x1RgbaToRRelu = this.createShaderProgram(CONV1X1_RGBA_TO_R_RELU_SRC, ['uInput', 'uWeight', 'uBias', 'uInC4']);

    this.vao = this.createQuadVAO(this.conv3x3Relu.program);

    // Upload model weights to GPU
    const convTensors = [...weights.values()]
      .filter(t => t.name.startsWith('onnx::Conv_'))
      .sort((a, b) => parseInt(a.name.split('_').pop()!) - parseInt(b.name.split('_').pop()!));

    // Pair consecutive tensors as (weight, bias) — R32F layout
    const makeConvGPU = (weightIdx: number): ConvLayerGPU => {
      const wt = convTensors[weightIdx];
      const bt = convTensors[weightIdx + 1];
      const outC = wt.shape[0];
      const inC = wt.shape.length === 4 ? wt.shape[1] : wt.data.length / (outC * (wt.shape[2] === 3 ? 9 : 1));
      const kSize = wt.shape.length === 4 ? wt.shape[2] * wt.shape[3] : 1;
      return {
        // Weight: (outC, inC*kSize) stored as texture (inC*kSize, outC)
        weightTex: this.createTex(wt.data, inC * kSize, outC),
        // Bias: (outC,) stored as texture (outC, 1)
        biasTex: this.createTex(bt.data, outC, 1),
        inC,
        outC,
      };
    };

    // RGBA-packed conv3x3: pack 4 output channels per RGBA texel
    // Weight ONNX layout: (outC, inC, kH, kW) row-major
    // Target texture: (inC*9, outC/4) RGBA32F
    //   texel(k, oc4).rgba = { w[oc4*4+0][k], w[oc4*4+1][k], w[oc4*4+2][k], w[oc4*4+3][k] }
    //   where k = ic*9 + kh*3 + kw
    // Bias texture: (outC/4, 1) RGBA32F
    //   texel(oc4, 0).rgba = { b[oc4*4+0], b[oc4*4+1], b[oc4*4+2], b[oc4*4+3] }
    const makeConv3x3Rgba = (weightIdx: number): ConvLayerRGBA => {
      const wt = convTensors[weightIdx];
      const bt = convTensors[weightIdx + 1];
      const outC = wt.shape[0];
      const inC = wt.shape[1];
      const kSize = 9; // 3x3
      const outC4 = outC / 4;
      const wWidth = inC * kSize;

      // Pack weights: texture is (wWidth, outC4) RGBA32F = wWidth * outC4 * 4 floats
      const packedW = new Float32Array(wWidth * outC4 * 4);
      for (let oc4 = 0; oc4 < outC4; oc4++) {
        for (let k = 0; k < wWidth; k++) {
          const texelIdx = (oc4 * wWidth + k) * 4;
          for (let ch = 0; ch < 4; ch++) {
            const oc = oc4 * 4 + ch;
            // ONNX weight layout: w[oc, ic, kh, kw] = wt.data[oc * inC * kSize + k]
            packedW[texelIdx + ch] = wt.data[oc * inC * kSize + k];
          }
        }
      }

      // Pack bias: texture is (outC4, 1) RGBA32F = outC4 * 4 floats
      const packedB = new Float32Array(outC4 * 4);
      for (let oc4 = 0; oc4 < outC4; oc4++) {
        for (let ch = 0; ch < 4; ch++) {
          packedB[oc4 * 4 + ch] = bt.data[oc4 * 4 + ch];
        }
      }

      return {
        weightTex: this.createTexRgba(packedW, wWidth, outC4),
        biasTex: this.createTexRgba(packedB, outC4, 1),
        inC,
        outC,
      };
    };

    // RGBA-packed input conv (R32F input → RGBA output):
    // Input has 7 channels (not divisible by 4), read one at a time.
    // Weight texture: (inC*9, outC/4) RGBA32F — same packing as makeConv3x3Rgba
    // Bias texture: (outC/4, 1) RGBA32F
    const makeConvR2Rgba = (weightIdx: number): ConvLayerRGBA => {
      // Same RGBA packing as makeConv3x3Rgba — input is R32F but weights are RGBA
      return makeConv3x3Rgba(weightIdx);
    };

    // Conv1x1 RGBA→R: input is RGBA-packed, output is R32F (small outC)
    // Weight ONNX layout: (outC, inC, 1, 1) → R32F texture (inC, outC)
    // Bias: R32F texture (outC, 1)
    const makeConv1x1RgbaToR = (weightIdx: number): ConvLayerRGBA => {
      const wt = convTensors[weightIdx];
      const bt = convTensors[weightIdx + 1];
      const outC = wt.shape[0];
      const inC = wt.shape[1];
      return {
        // Weight stays R32F: (inC, outC)
        weightTex: this.createTex(wt.data, inC, outC),
        // Bias stays R32F: (outC, 1)
        biasTex: this.createTex(bt.data, outC, 1),
        inC,
        outC,
      };
    };

    let idx = 0;
    this.inputConv = makeConvGPU(idx);
    this.inputConvRgba = makeConvR2Rgba(idx);
    idx += 2;

    this.resBlocks = [];
    this.resBlocksRgba = [];
    for (let b = 0; b < config.numBlocks; b++) {
      this.resBlocks.push({
        conv1: makeConvGPU(idx),
        conv2: makeConvGPU(idx + 2),
      });
      this.resBlocksRgba.push({
        conv1: makeConv3x3Rgba(idx),
        conv2: makeConv3x3Rgba(idx + 2),
      });
      idx += 4;
    }
    this.policyConv = makeConvGPU(idx);
    this.policyConvRgba = makeConv1x1RgbaToR(idx);
    idx += 2;
    this.valueConv = makeConvGPU(idx);
    this.valueConvRgba = makeConv1x1RgbaToR(idx);
    idx += 2;

    // FC layers (CPU)
    const getFC = (prefix: string): LinearLayer => {
      const w = weights.get(`${prefix}.weight`)!;
      const b = weights.get(`${prefix}.bias`)!;
      return { weight: w.data, bias: b.data, outFeatures: w.shape[0], inFeatures: w.shape[1] };
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

    // Max batch size for GPU batching (allocation limit)
    const MAX_BATCH = 16;

    // Create R32F activation textures (HW*MAX_BATCH x maxC) — kept for backward compat
    const maxC = Math.max(f, config.policyFilters, config.valueFilters);
    this.actTexA = this.createEmptyTex(HW * MAX_BATCH, maxC);
    this.actTexB = this.createEmptyTex(HW * MAX_BATCH, maxC);
    this.actFbA = this.createFb(this.actTexA);
    this.actFbB = this.createFb(this.actTexB);

    // Create RGBA activation textures (HW*MAX_BATCH, maxC/4) RGBA32F
    const maxC4 = Math.ceil(maxC / 4);
    this.actTexRgbaA = this.createEmptyTexRgba(HW * MAX_BATCH, maxC4);
    this.actTexRgbaB = this.createEmptyTexRgba(HW * MAX_BATCH, maxC4);
    this.actFbRgbaA = this.createFb(this.actTexRgbaA);
    this.actFbRgbaB = this.createFb(this.actTexRgbaB);

    // R32F head texture (for policy/value head output)
    this.headTex = this.createEmptyTex(HW * MAX_BATCH, maxC);
    this.headFb = this.createFb(this.headTex);

    // RGBA head texture (temporary for residual blocks)
    this.headTexRgba = this.createEmptyTexRgba(HW * MAX_BATCH, maxC4);
    this.headFbRgba = this.createFb(this.headTexRgba);

    // Input texture (7 channels, wide enough for batch) — R32F
    this.inputTex = this.createEmptyTex(HW * MAX_BATCH, 7);
    this.inputFb = this.createFb(this.inputTex);

    // CPU buffers (sized for max batch)
    const maxHeadSize = Math.max(config.policyFilters, config.valueFilters) * HW * MAX_BATCH;
    this.readBuf = new Float32Array(maxHeadSize * 4); // RGBA
    this.policyHiddenBuf = this.policyFc1 ? new Float32Array(this.policyFc1.outFeatures) : null;
    this.policyLogitsBuf = new Float32Array(this.policyFc.outFeatures);
    this.policyOutBuf = new Float32Array(this.policyFc.outFeatures);
    this.valueHiddenBuf = new Float32Array(this.valueFc1.outFeatures);
    this.valueOutBuf = new Float32Array(1);
  }

  /**
   * Run the full forward pass on the GPU.
   * Uses R32F shaders (RGBA was tested but slower on mobile tile-based GPUs).
   * Only 2 readPixels calls (policy head + value head output).
   */
  forward(input: Float32Array): { policy: Float32Array; value: number } {
    const gl = this.gl;
    const f = this.config.numFilters;

    // Upload input tensor (7, 8, 7) → R32F texture (HW=56, C=7)
    gl.bindTexture(gl.TEXTURE_2D, this.inputTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, HW, 7, gl.RED, gl.FLOAT, input);

    // === Input conv (7 → f) + ReLU ===
    this.runConv3x3(this.conv3x3Relu, this.inputTex, this.inputConv, HW, f, this.actTexA, this.actFbA);

    // === Residual tower ===
    let currentTex = this.actTexA;
    let currentFb = this.actFbA;
    let scratchTex = this.actTexB;
    let scratchFb = this.actFbB;

    for (const block of this.resBlocks) {
      this.runConv3x3(this.conv3x3Relu, currentTex, block.conv1, HW, f, scratchTex, scratchFb);
      this.runConv3x3(this.conv3x3NoRelu, scratchTex, block.conv2, HW, f, this.headTex, this.headFb);
      this.runResidualRelu(this.headTex, currentTex, HW, f, scratchTex, scratchFb);
      const tmpTex = currentTex; currentTex = scratchTex; scratchTex = tmpTex;
      const tmpFb = currentFb; currentFb = scratchFb; scratchFb = tmpFb;
    }

    // === Policy head: 1x1 conv + ReLU ===
    const pf = this.config.policyFilters;
    this.runConv1x1(currentTex, this.policyConv, HW, pf, this.headTex, this.headFb);

    // Read back policy head (pf * HW floats, R32F output)
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.headFb);
    const policySize = pf * HW;
    if (this.readBuf.length < policySize * 4) {
      this.readBuf = new Float32Array(policySize * 4);
    }
    gl.readPixels(0, 0, HW, pf, gl.RGBA, gl.FLOAT, this.readBuf);
    const policyFlat = new Float32Array(policySize);
    for (let i = 0; i < policySize; i++) {
      policyFlat[i] = this.readBuf[i * 4]; // Extract R channel
    }

    // === Value head: 1x1 conv + ReLU ===
    const vf = this.config.valueFilters;
    this.runConv1x1(currentTex, this.valueConv, HW, vf, this.headTex, this.headFb);

    // Read back value head (vf * HW floats, R32F output)
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.headFb);
    const valueSize = vf * HW;
    if (this.readBuf.length < valueSize * 4) {
      this.readBuf = new Float32Array(valueSize * 4);
    }
    gl.readPixels(0, 0, HW, vf, gl.RGBA, gl.FLOAT, this.readBuf);
    const valueFlat = new Float32Array(valueSize);
    for (let i = 0; i < valueSize; i++) {
      valueFlat[i] = this.readBuf[i * 4];
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // === FC layers on CPU ===

    // Policy FC
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

    // Value FC
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

  /**
   * Run batched forward pass: N inputs processed in parallel on the GPU.
   * Conv3x3 shaders handle batch boundaries (no padding bleed).
   * Only 2 readPixels total (policy heads + value heads for all N).
   */
  forwardBatch(inputs: Float32Array[]): Array<{ policy: Float32Array; value: number }> {
    const gl = this.gl;
    const f = this.config.numFilters;
    const N = inputs.length;
    if (N === 0) return [];
    if (N === 1) return [this.forward(inputs[0])];

    const totalHW = HW * N;

    // Stack all input tensors into one wide texture (HW*N, 7)
    // Each input is CHW (7 * HW floats). Stack side by side in x-dimension.
    const stacked = new Float32Array(7 * totalHW);
    for (let b = 0; b < N; b++) {
      const input = inputs[b];
      for (let c = 0; c < 7; c++) {
        for (let hw = 0; hw < HW; hw++) {
          stacked[c * totalHW + b * HW + hw] = input[c * HW + hw];
        }
      }
    }

    gl.bindTexture(gl.TEXTURE_2D, this.inputTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, totalHW, 7, gl.RED, gl.FLOAT, stacked);

    // === Conv tower (R32F, same as single but with totalHW width) ===
    this.runConv3x3(this.conv3x3Relu, this.inputTex, this.inputConv, totalHW, f, this.actTexA, this.actFbA);

    let currentTex = this.actTexA;
    let currentFb = this.actFbA;
    let scratchTex = this.actTexB;
    let scratchFb = this.actFbB;

    for (const block of this.resBlocks) {
      this.runConv3x3(this.conv3x3Relu, currentTex, block.conv1, totalHW, f, scratchTex, scratchFb);
      this.runConv3x3(this.conv3x3NoRelu, scratchTex, block.conv2, totalHW, f, this.headTex, this.headFb);
      this.runResidualRelu(this.headTex, currentTex, totalHW, f, scratchTex, scratchFb);
      const tmpTex = currentTex; currentTex = scratchTex; scratchTex = tmpTex;
      const tmpFb = currentFb; currentFb = scratchFb; scratchFb = tmpFb;
    }

    // === Policy head (all N at once) ===
    const pf = this.config.policyFilters;
    this.runConv1x1(currentTex, this.policyConv, totalHW, pf, this.headTex, this.headFb);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.headFb);
    const policyTotalSize = pf * totalHW;
    if (this.readBuf.length < policyTotalSize * 4) {
      this.readBuf = new Float32Array(policyTotalSize * 4);
    }
    gl.readPixels(0, 0, totalHW, pf, gl.RGBA, gl.FLOAT, this.readBuf);

    // Extract per-element policy flats
    const policyFlats: Float32Array[] = [];
    for (let b = 0; b < N; b++) {
      const flat = new Float32Array(pf * HW);
      for (let oc = 0; oc < pf; oc++) {
        for (let hw = 0; hw < HW; hw++) {
          const readIdx = (oc * totalHW + b * HW + hw) * 4;
          flat[oc * HW + hw] = this.readBuf[readIdx];
        }
      }
      policyFlats.push(flat);
    }

    // === Value head (all N at once) ===
    const vf = this.config.valueFilters;
    this.runConv1x1(currentTex, this.valueConv, totalHW, vf, this.headTex, this.headFb);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.headFb);
    const valueTotalSize = vf * totalHW;
    if (this.readBuf.length < valueTotalSize * 4) {
      this.readBuf = new Float32Array(valueTotalSize * 4);
    }
    gl.readPixels(0, 0, totalHW, vf, gl.RGBA, gl.FLOAT, this.readBuf);

    const valueFlats: Float32Array[] = [];
    for (let b = 0; b < N; b++) {
      const flat = new Float32Array(vf * HW);
      for (let oc = 0; oc < vf; oc++) {
        for (let hw = 0; hw < HW; hw++) {
          const readIdx = (oc * totalHW + b * HW + hw) * 4;
          flat[oc * HW + hw] = this.readBuf[readIdx];
        }
      }
      valueFlats.push(flat);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // === FC layers on CPU (per element) ===
    const results: Array<{ policy: Float32Array; value: number }> = [];
    for (let b = 0; b < N; b++) {
      // Policy FC
      const policyLogits = new Float32Array(this.policyFc.outFeatures);
      if (this.policyFc1) {
        const hidden = new Float32Array(this.policyFc1.outFeatures);
        linearCPU(policyFlats[b], this.policyFc1.weight, this.policyFc1.bias,
          this.policyFc1.outFeatures, this.policyFc1.inFeatures, hidden);
        for (let i = 0; i < hidden.length; i++) hidden[i] = Math.max(0, hidden[i]);
        linearCPU(hidden, this.policyFc.weight, this.policyFc.bias,
          this.policyFc.outFeatures, this.policyFc.inFeatures, policyLogits);
      } else {
        linearCPU(policyFlats[b], this.policyFc.weight, this.policyFc.bias,
          this.policyFc.outFeatures, this.policyFc.inFeatures, policyLogits);
      }
      const policy = new Float32Array(policyLogits.length);
      logSoftmax(policyLogits, policy, policyLogits.length);

      // Value FC
      const vh = new Float32Array(this.valueFc1.outFeatures);
      linearCPU(valueFlats[b], this.valueFc1.weight, this.valueFc1.bias,
        this.valueFc1.outFeatures, this.valueFc1.inFeatures, vh);
      for (let i = 0; i < vh.length; i++) vh[i] = Math.max(0, vh[i]);
      const vOut = new Float32Array(1);
      linearCPU(vh, this.valueFc2.weight, this.valueFc2.bias,
        this.valueFc2.outFeatures, this.valueFc2.inFeatures, vOut);

      results.push({ policy, value: Math.tanh(vOut[0]) });
    }

    return results;
  }

  dispose(): void {
    const gl = this.gl;
    // Delete all textures, framebuffers, programs, VAO
    const textures = [
      this.actTexA, this.actTexB, this.headTex, this.inputTex,
      this.actTexRgbaA, this.actTexRgbaB, this.headTexRgba,
    ];
    const fbs = [
      this.actFbA, this.actFbB, this.headFb, this.inputFb,
      this.actFbRgbaA, this.actFbRgbaB, this.headFbRgba,
    ];
    // R32F layers
    for (const layer of [this.inputConv, this.policyConv, this.valueConv]) {
      textures.push(layer.weightTex, layer.biasTex);
    }
    for (const block of this.resBlocks) {
      textures.push(block.conv1.weightTex, block.conv1.biasTex);
      textures.push(block.conv2.weightTex, block.conv2.biasTex);
    }
    // RGBA layers
    for (const layer of [this.inputConvRgba, this.policyConvRgba, this.valueConvRgba]) {
      textures.push(layer.weightTex, layer.biasTex);
    }
    for (const block of this.resBlocksRgba) {
      textures.push(block.conv1.weightTex, block.conv1.biasTex);
      textures.push(block.conv2.weightTex, block.conv2.biasTex);
    }
    for (const t of textures) gl.deleteTexture(t);
    for (const f of fbs) gl.deleteFramebuffer(f);
    for (const p of [
      this.conv3x3Relu, this.conv3x3NoRelu, this.residualRelu, this.conv1x1Relu,
      this.conv3x3RgbaRelu, this.conv3x3RgbaNoRelu, this.conv3x3R2RgbaRelu,
      this.residualRgbaRelu, this.conv1x1RgbaToRRelu,
    ]) {
      gl.deleteProgram(p.program);
    }
    gl.deleteVertexArray(this.vao);
  }

  // ---- GPU operations (R32F — kept for backward compat) ----

  private runConv3x3(
    shader: ShaderProgram, inputTex: WebGLTexture,
    conv: ConvLayerGPU, totalWidth: number, outC: number,
    _outTex: WebGLTexture, outFb: WebGLFramebuffer,
  ): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, outFb);
    gl.viewport(0, 0, totalWidth, outC);
    gl.useProgram(shader.program);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, inputTex);
    gl.uniform1i(shader.uniforms.uInput, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, conv.weightTex);
    gl.uniform1i(shader.uniforms.uWeight, 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, conv.biasTex);
    gl.uniform1i(shader.uniforms.uBias, 2);

    gl.uniform1i(shader.uniforms.uInC, conv.inC);
    gl.uniform1i(shader.uniforms.uH, ROWS);
    gl.uniform1i(shader.uniforms.uW, COLS);
    gl.uniform1i(shader.uniforms.uSingleHW, HW);

    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  private runConv1x1(
    inputTex: WebGLTexture, conv: ConvLayerGPU,
    totalWidth: number, outC: number,
    _outTex: WebGLTexture, outFb: WebGLFramebuffer,
  ): void {
    const gl = this.gl;
    const shader = this.conv1x1Relu;
    gl.bindFramebuffer(gl.FRAMEBUFFER, outFb);
    gl.viewport(0, 0, totalWidth, outC);
    gl.useProgram(shader.program);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, inputTex);
    gl.uniform1i(shader.uniforms.uInput, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, conv.weightTex);
    gl.uniform1i(shader.uniforms.uWeight, 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, conv.biasTex);
    gl.uniform1i(shader.uniforms.uBias, 2);

    gl.uniform1i(shader.uniforms.uInC, conv.inC);

    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  private runResidualRelu(
    aTex: WebGLTexture, bTex: WebGLTexture,
    totalWidth: number, channels: number,
    _outTex: WebGLTexture, outFb: WebGLFramebuffer,
  ): void {
    const gl = this.gl;
    const shader = this.residualRelu;
    gl.bindFramebuffer(gl.FRAMEBUFFER, outFb);
    gl.viewport(0, 0, totalWidth, channels);
    gl.useProgram(shader.program);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, aTex);
    gl.uniform1i(shader.uniforms.uA, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, bTex);
    gl.uniform1i(shader.uniforms.uB, 1);

    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }


  // ---- Helper methods ----

  private createShaderProgram(fsSrc: string, uniformNames: string[]): ShaderProgram {
    const gl = this.gl;
    const vs = this.compileShader(gl.VERTEX_SHADER, VERT_SRC);
    const fs = this.compileShader(gl.FRAGMENT_SHADER, fsSrc);
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`Shader link failed: ${gl.getProgramInfoLog(prog)}`);
    }
    gl.deleteShader(vs);
    gl.deleteShader(fs);

    const uniforms: Record<string, WebGLUniformLocation> = {};
    for (const name of uniformNames) {
      uniforms[name] = gl.getUniformLocation(prog, name)!;
    }
    return { program: prog, uniforms };
  }

  private compileShader(type: number, source: string): WebGLShader {
    const gl = this.gl;
    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(`Shader compile failed: ${gl.getShaderInfoLog(shader)}`);
    }
    return shader;
  }

  private createQuadVAO(program: WebGLProgram): WebGLVertexArrayObject {
    const gl = this.gl;
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    const buf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
    const posLoc = gl.getAttribLocation(program, 'aPosition');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    return vao;
  }

  private createTex(data: Float32Array, width: number, height: number): WebGLTexture {
    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  }

  private createEmptyTex(width: number, height: number): WebGLTexture {
    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  }

  /** Create an RGBA32F texture with initial data. */
  private createTexRgba(data: Float32Array, width: number, height: number): WebGLTexture {
    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  }

  /** Create an empty RGBA32F texture (for activation framebuffers). */
  private createEmptyTexRgba(width: number, height: number): WebGLTexture {
    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  }

  private createFb(tex: WebGLTexture): WebGLFramebuffer {
    const gl = this.gl;
    const fb = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`Framebuffer incomplete: ${status}`);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return fb;
  }
}

// ---- CPU helpers for FC layers ----

function linearCPU(
  input: Float32Array, weight: Float32Array, bias: Float32Array,
  outFeatures: number, inFeatures: number, output: Float32Array,
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
  for (let i = 0; i < len; i++) if (input[i] > max) max = input[i];
  let sumExp = 0;
  for (let i = 0; i < len; i++) sumExp += Math.exp(input[i] - max);
  const logSumExp = max + Math.log(sumExp);
  for (let i = 0; i < len; i++) output[i] = input[i] - logSumExp;
}

// ---- Factory ----

export function createGPUModelFromOnnx(buffer: ArrayBuffer, canvas?: HTMLCanvasElement | OffscreenCanvas): GPUForwardPass {
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

  return new GPUForwardPass({
    numFilters, numBlocks, policyFilters, valueFilters, valueHidden, policyHidden,
  }, tensorMap, canvas);
}
