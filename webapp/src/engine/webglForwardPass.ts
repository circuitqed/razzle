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
// Input texture: (HW, inC)  Weight texture: (inC*9, outC)  Bias texture: (1, outC)
// Output: (HW, outC)
const CONV3X3_RELU_SRC = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uInput;
uniform sampler2D uWeight;
uniform sampler2D uBias;
uniform int uInC;
uniform int uH;
uniform int uW;
out vec4 fragColor;
void main() {
  int hw = int(gl_FragCoord.x);
  int oc = int(gl_FragCoord.y);
  int h = hw / uW;
  int w = hw - h * uW;
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
             * texelFetch(uInput, ivec2(ih * uW + iw, ic), 0).r;
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
out vec4 fragColor;
void main() {
  int hw = int(gl_FragCoord.x);
  int oc = int(gl_FragCoord.y);
  int h = hw / uW;
  int w = hw - h * uW;
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
             * texelFetch(uInput, ivec2(ih * uW + iw, ic), 0).r;
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

// ---- GPU Context ----

interface ConvLayerGPU {
  weightTex: WebGLTexture;
  biasTex: WebGLTexture;
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

  // Shader programs
  private conv3x3Relu: ShaderProgram;
  private conv3x3NoRelu: ShaderProgram;
  private residualRelu: ShaderProgram;
  private conv1x1Relu: ShaderProgram;

  // Full-screen quad
  private vao: WebGLVertexArrayObject;

  // Model layers (GPU textures)
  private inputConv: ConvLayerGPU;
  private resBlocks: Array<{ conv1: ConvLayerGPU; conv2: ConvLayerGPU }>;
  private policyConv: ConvLayerGPU;
  private valueConv: ConvLayerGPU;

  // FC layers (CPU — small enough that GPU overhead isn't worth it)
  private policyFc: LinearLayer;
  private policyFc1: LinearLayer | null;
  private valueFc1: LinearLayer;
  private valueFc2: LinearLayer;

  // Activation textures (ping-pong) — stay on GPU
  private actTexA: WebGLTexture;
  private actTexB: WebGLTexture;
  private actFbA: WebGLFramebuffer;
  private actFbB: WebGLFramebuffer;

  // Head output texture + framebuffer (for policy/value conv)
  private headTex: WebGLTexture;
  private headFb: WebGLFramebuffer;

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

    // Compile shaders
    this.conv3x3Relu = this.createShaderProgram(CONV3X3_RELU_SRC, ['uInput', 'uWeight', 'uBias', 'uInC', 'uH', 'uW']);
    this.conv3x3NoRelu = this.createShaderProgram(CONV3X3_NORELU_SRC, ['uInput', 'uWeight', 'uBias', 'uInC', 'uH', 'uW']);
    this.residualRelu = this.createShaderProgram(RESIDUAL_RELU_SRC, ['uA', 'uB']);
    this.conv1x1Relu = this.createShaderProgram(CONV1X1_RELU_SRC, ['uInput', 'uWeight', 'uBias', 'uInC']);
    this.vao = this.createQuadVAO(this.conv3x3Relu.program);

    // Upload model weights to GPU
    const convTensors = [...weights.values()]
      .filter(t => t.name.startsWith('onnx::Conv_'))
      .sort((a, b) => parseInt(a.name.split('_').pop()!) - parseInt(b.name.split('_').pop()!));

    // Pair consecutive tensors as (weight, bias)
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

    let idx = 0;
    this.inputConv = makeConvGPU(idx); idx += 2;
    this.resBlocks = [];
    for (let b = 0; b < config.numBlocks; b++) {
      this.resBlocks.push({
        conv1: makeConvGPU(idx),
        conv2: makeConvGPU(idx + 2),
      });
      idx += 4;
    }
    this.policyConv = makeConvGPU(idx); idx += 2;
    this.valueConv = makeConvGPU(idx); idx += 2;

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

    // Create activation textures (HW × maxC)
    const maxC = Math.max(f, config.policyFilters, config.valueFilters);
    this.actTexA = this.createEmptyTex(HW, maxC);
    this.actTexB = this.createEmptyTex(HW, maxC);
    this.actFbA = this.createFb(this.actTexA);
    this.actFbB = this.createFb(this.actTexB);

    // Head texture
    this.headTex = this.createEmptyTex(HW, maxC);
    this.headFb = this.createFb(this.headTex);

    // Input texture (7 channels)
    this.inputTex = this.createEmptyTex(HW, 7);
    this.inputFb = this.createFb(this.inputTex);

    // CPU buffers
    const maxHeadSize = Math.max(config.policyFilters, config.valueFilters) * HW;
    this.readBuf = new Float32Array(maxHeadSize * 4); // RGBA
    this.policyHiddenBuf = this.policyFc1 ? new Float32Array(this.policyFc1.outFeatures) : null;
    this.policyLogitsBuf = new Float32Array(this.policyFc.outFeatures);
    this.policyOutBuf = new Float32Array(this.policyFc.outFeatures);
    this.valueHiddenBuf = new Float32Array(this.valueFc1.outFeatures);
    this.valueOutBuf = new Float32Array(1);
  }

  /**
   * Run the full forward pass on the GPU.
   * Only 2 readPixels calls (policy head + value head output).
   */
  forward(input: Float32Array): { policy: Float32Array; value: number } {
    const gl = this.gl;
    const f = this.config.numFilters;

    // Upload input tensor (7, 8, 7) → texture (HW=56, C=7)
    // Input is in CHW order: input[c*HW + h*W + w]
    // Texture stores (hw, c) = input[c * HW + hw]
    // Need to transpose from CHW to HWC for texture layout
    const inputHWC = new Float32Array(7 * HW);
    for (let c = 0; c < 7; c++) {
      for (let hw = 0; hw < HW; hw++) {
        inputHWC[hw * 7 + c] = input[c * HW + hw];
      }
    }
    // Actually, our texture layout is (HW, C) where texel(hw, c) is at pixel (hw, c)
    // texImage2D stores row-major: pixel at (x, y) = data[y * width + x]
    // Width = HW, Height = C  → data[c * HW + hw] = input[c * HW + hw]
    // So CHW order IS correct for texture (HW, C)!
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
      // conv1 + ReLU: current → scratch
      this.runConv3x3(this.conv3x3Relu, currentTex, block.conv1, HW, f, scratchTex, scratchFb);

      // conv2 (no ReLU): scratch → head (temporary)
      this.runConv3x3(this.conv3x3NoRelu, scratchTex, block.conv2, HW, f, this.headTex, this.headFb);

      // Residual add + ReLU: head + current → scratch
      this.runResidualRelu(this.headTex, currentTex, HW, f, scratchTex, scratchFb);

      // Swap
      const tmpTex = currentTex; currentTex = scratchTex; scratchTex = tmpTex;
      const tmpFb = currentFb; currentFb = scratchFb; scratchFb = tmpFb;
    }

    // Tower output is in currentTex

    // === Policy head: 1x1 conv + ReLU ===
    const pf = this.config.policyFilters;
    this.runConv1x1(currentTex, this.policyConv, HW, pf, this.headTex, this.headFb);

    // Read back policy head (pf * HW floats)
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

    // Read back value head (vf * HW floats)
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

  dispose(): void {
    const gl = this.gl;
    // Delete all textures, framebuffers, programs, VAO
    const textures = [this.actTexA, this.actTexB, this.headTex, this.inputTex];
    const fbs = [this.actFbA, this.actFbB, this.headFb, this.inputFb];
    for (const layer of [this.inputConv, this.policyConv, this.valueConv]) {
      textures.push(layer.weightTex, layer.biasTex);
    }
    for (const block of this.resBlocks) {
      textures.push(block.conv1.weightTex, block.conv1.biasTex);
      textures.push(block.conv2.weightTex, block.conv2.biasTex);
    }
    for (const t of textures) gl.deleteTexture(t);
    for (const f of fbs) gl.deleteFramebuffer(f);
    for (const p of [this.conv3x3Relu, this.conv3x3NoRelu, this.residualRelu, this.conv1x1Relu]) {
      gl.deleteProgram(p.program);
    }
    gl.deleteVertexArray(this.vao);
  }

  // ---- GPU operations ----

  private runConv3x3(
    shader: ShaderProgram, inputTex: WebGLTexture,
    conv: ConvLayerGPU, hw: number, outC: number,
    _outTex: WebGLTexture, outFb: WebGLFramebuffer,
  ): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, outFb);
    gl.viewport(0, 0, hw, outC);
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

    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  private runConv1x1(
    inputTex: WebGLTexture, conv: ConvLayerGPU,
    hw: number, outC: number,
    _outTex: WebGLTexture, outFb: WebGLFramebuffer,
  ): void {
    const gl = this.gl;
    const shader = this.conv1x1Relu;
    gl.bindFramebuffer(gl.FRAMEBUFFER, outFb);
    gl.viewport(0, 0, hw, outC);
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
    hw: number, channels: number,
    _outTex: WebGLTexture, outFb: WebGLFramebuffer,
  ): void {
    const gl = this.gl;
    const shader = this.residualRelu;
    gl.bindFramebuffer(gl.FRAMEBUFFER, outFb);
    gl.viewport(0, 0, hw, channels);
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
