/**
 * GPUForwardPass (WebGL2, used on iOS) run against a fake WebGL2 context.
 *
 * Real WebGL can't run headless here, so this exercises everything except the
 * GPU itself: GPUForwardPass's host code (weight/bias texture layout, input
 * upload, viewports, framebuffer ping-pong, readPixels indexing, batching, the
 * v2 spatial-head readback + CPU gather) runs unchanged, and each fragment
 * shader is replaced by a line-by-line JS port of its GLSL source. A strict
 * texelFetch (throws when out of bounds) catches layout/size mistakes.
 *
 * What it can NOT catch: GLSL compile errors on a real driver, precision
 * differences of the device GPU, and real-driver readPixels/format quirks.
 * Those are covered by the on-device suite (npm run test:ios, `inference` group).
 */

import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import { GPUForwardPass, GPU_SHADER_SOURCES as S } from '../webglForwardPass';
import { createModelFromOnnx } from '../inference';
import { parseOnnxWeights, type WeightTensor } from '../onnxWeights';
import { inferModelConfig } from '../modelConfig';
import { stateToTensor } from '../tensor';
import type { EngineState } from '../state';

// ---- Fake WebGL2 ----

const GL = {
  TEXTURE_2D: 0x0de1, R32F: 0x822e, RGBA32F: 0x8814, RED: 0x1903, RGBA: 0x1908, FLOAT: 0x1406,
  FRAMEBUFFER: 0x8d40, COLOR_ATTACHMENT0: 0x8ce0, FRAMEBUFFER_COMPLETE: 0x8cd5,
  TEXTURE_MIN_FILTER: 0x2801, TEXTURE_MAG_FILTER: 0x2800, TEXTURE_WRAP_S: 0x2802, TEXTURE_WRAP_T: 0x2803,
  NEAREST: 0x2600, CLAMP_TO_EDGE: 0x812f, VERTEX_SHADER: 0x8b31, FRAGMENT_SHADER: 0x8b30,
  COMPILE_STATUS: 0x8b81, LINK_STATUS: 0x8b82, ARRAY_BUFFER: 0x8892, STATIC_DRAW: 0x88e4,
  TRIANGLE_STRIP: 0x0005, TEXTURE0: 0x84c0, TEXTURE1: 0x84c1, TEXTURE2: 0x84c2, TEXTURE3: 0x84c3,
};

class Tex {
  w = 0; h = 0; rgba = false; data = new Float32Array(0);
  fetch(x: number, y: number, ch = 0): number {
    if (x < 0 || y < 0 || x >= this.w || y >= this.h) {
      throw new Error(`texelFetch out of bounds (${x}, ${y}) on ${this.w}x${this.h} texture`);
    }
    return this.data[(y * this.w + x) * 4 + ch];
  }
}
interface Prog { fs: string; uniforms: Map<string, number> }
interface Fb { tex: Tex | null }

type Pass = (ctx: { u: (n: string) => number; t: (n: string) => Tex; out: Tex; vw: number; vh: number }) => void;

/** JS ports of the fragment shaders. Each writes out(x, y) for the viewport. */
function write(out: Tex, x: number, y: number, r: number, g = 0, b = 0, a = 1) {
  if (x >= out.w || y >= out.h) throw new Error(`draw outside ${out.w}x${out.h} target at (${x}, ${y})`);
  const i = (y * out.w + x) * 4;
  if (out.rgba) { out.data[i] = r; out.data[i + 1] = g; out.data[i + 2] = b; out.data[i + 3] = a; }
  else { out.data[i] = r; out.data[i + 1] = 0; out.data[i + 2] = 0; out.data[i + 3] = 1; }
}
function conv3x3(relu: boolean): Pass {
  return ({ u, t, out, vw, vh }) => {
    const inp = t('uInput'), wt = t('uWeight'), bias = t('uBias');
    const inC = u('uInC'), H = u('uH'), W = u('uW'), single = u('uSingleHW');
    for (let oc = 0; oc < vh; oc++) {
      for (let g = 0; g < vw; g++) {
        const batchStart = Math.floor(g / single) * single;
        const local = g - batchStart;
        const h = Math.floor(local / W), w = local - h * W;
        let sum = bias.fetch(oc, 0);
        for (let ic = 0; ic < inC; ic++) {
          const wBase = ic * 9;
          for (let kh = 0; kh < 3; kh++) {
            const ih = h + kh - 1;
            if (ih < 0 || ih >= H) continue;
            for (let kw = 0; kw < 3; kw++) {
              const iw = w + kw - 1;
              if (iw < 0 || iw >= W) continue;
              sum += wt.fetch(wBase + kh * 3 + kw, oc) * inp.fetch(batchStart + ih * W + iw, ic);
            }
          }
        }
        write(out, g, oc, relu ? Math.max(0, sum) : sum);
      }
    }
  };
}
function conv1x1(relu: boolean): Pass {
  return ({ u, t, out, vw, vh }) => {
    const inp = t('uInput'), wt = t('uWeight'), bias = t('uBias');
    const inC = u('uInC');
    for (let oc = 0; oc < vh; oc++) {
      for (let hw = 0; hw < vw; hw++) {
        let sum = bias.fetch(oc, 0);
        for (let ic = 0; ic < inC; ic++) sum += wt.fetch(ic, oc) * inp.fetch(hw, ic);
        write(out, hw, oc, relu ? Math.max(0, sum) : sum);
      }
    }
  };
}
const residual: Pass = ({ t, out, vw, vh }) => {
  const a = t('uA'), b = t('uB');
  for (let y = 0; y < vh; y++) for (let x = 0; x < vw; x++) write(out, x, y, Math.max(0, a.fetch(x, y) + b.fetch(x, y)));
};
const meanPool: Pass = ({ u, t, out, vw, vh }) => {
  const inp = t('uInput'), single = u('uSingleHW');
  for (let c = 0; c < vh; c++) {
    for (let b = 0; b < vw; b++) {
      let sum = 0;
      for (let i = 0; i < single; i++) sum += inp.fetch(b * single + i, c);
      write(out, b, c, sum / single);
    }
  }
};
const unused: Pass = () => { throw new Error('RGBA shader is not used by forward() and not emulated'); };

const PASSES = new Map<string, Pass>([
  [S.CONV3X3_RELU_SRC, conv3x3(true)],
  [S.CONV3X3_NORELU_SRC, conv3x3(false)],
  [S.RESIDUAL_RELU_SRC, residual],
  [S.CONV1X1_RELU_SRC, conv1x1(true)],
  [S.CONV1X1_NORELU_SRC, conv1x1(false)],
  [S.MEAN_POOL_SRC, meanPool],
  [S.CONV3X3_RGBA_RELU_SRC, unused],
  [S.CONV3X3_RGBA_NORELU_SRC, unused],
  [S.CONV3X3_R2RGBA_RELU_SRC, unused],
  [S.RESIDUAL_RGBA_RELU_SRC, unused],
  [S.CONV1X1_RGBA_TO_R_RELU_SRC, unused],
]);

function makeFakeGL() {
  const units: (Tex | null)[] = new Array(16).fill(null);
  let active = 0;
  let program: Prog | null = null;
  let fb: Fb | null = null;
  let viewport = [0, 0, 0, 0];
  let draws = 0;
  const gl = {
    ...GL,
    get draws() { return draws; },
    getExtension: () => ({}),
    isContextLost: () => false,
    createShader: (type: number) => ({ type, src: '' }),
    shaderSource: (sh: { src: string }, src: string) => { sh.src = src; },
    compileShader: () => {},
    getShaderParameter: () => true,
    getShaderInfoLog: () => '',
    deleteShader: () => {},
    createProgram: (): Prog => ({ fs: '', uniforms: new Map() }),
    attachShader: (p: Prog, sh: { type: number; src: string }) => { if (sh.type === GL.FRAGMENT_SHADER) p.fs = sh.src; },
    linkProgram: (p: Prog) => { if (!PASSES.has(p.fs)) throw new Error('unknown fragment shader'); },
    getProgramParameter: () => true,
    getProgramInfoLog: () => '',
    getUniformLocation: (p: Prog, name: string) => ({ p, name }),
    useProgram: (p: Prog) => { program = p; },
    uniform1i: (loc: { p: Prog; name: string }, v: number) => {
      if (loc.p !== program) throw new Error(`uniform ${loc.name} set on a program that is not in use`);
      loc.p.uniforms.set(loc.name, v);
    },
    deleteProgram: () => {},
    createVertexArray: () => ({}), bindVertexArray: () => {}, deleteVertexArray: () => {},
    createBuffer: () => ({}), bindBuffer: () => {}, bufferData: () => {},
    getAttribLocation: () => 0, enableVertexAttribArray: () => {}, vertexAttribPointer: () => {},
    createTexture: () => new Tex(),
    deleteTexture: () => {},
    activeTexture: (u: number) => {
      if (!(u >= GL.TEXTURE0 && u < GL.TEXTURE0 + 16)) throw new Error(`bad texture unit ${u}`);
      active = u - GL.TEXTURE0;
    },
    bindTexture: (_t: number, tex: Tex | null) => { units[active] = tex; },
    texParameteri: () => {},
    texImage2D: (_t: number, _l: number, internal: number, w: number, h: number, _b: number,
      format: number, _type: number, data: Float32Array | null) => {
      const tex = units[active]!;
      tex.w = w; tex.h = h; tex.rgba = internal === GL.RGBA32F;
      tex.data = new Float32Array(w * h * 4);
      if (!data) return;
      const comps = format === GL.RGBA ? 4 : 1;
      if (data.length < w * h * comps) throw new Error(`texImage2D: data too small (${data.length} < ${w * h * comps})`);
      for (let i = 0; i < w * h; i++) {
        for (let c = 0; c < comps; c++) tex.data[i * 4 + c] = data[i * comps + c];
        if (comps === 1) tex.data[i * 4 + 3] = 1;
      }
    },
    texSubImage2D: (_t: number, _l: number, x: number, y: number, w: number, h: number,
      format: number, _type: number, data: Float32Array) => {
      const tex = units[active]!;
      if (format !== GL.RED) throw new Error('texSubImage2D: only RED uploads emulated');
      if (x + w > tex.w || y + h > tex.h) throw new Error(`texSubImage2D: ${w}x${h} at (${x},${y}) exceeds ${tex.w}x${tex.h}`);
      if (data.length < w * h) throw new Error('texSubImage2D: data too small');
      for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) tex.data[((y + r) * tex.w + x + c) * 4] = data[r * w + c];
    },
    createFramebuffer: (): Fb => ({ tex: null }),
    deleteFramebuffer: () => {},
    bindFramebuffer: (_t: number, f: Fb | null) => { fb = f; },
    framebufferTexture2D: (_t: number, _a: number, _tt: number, tex: Tex) => { fb!.tex = tex; },
    checkFramebufferStatus: () => GL.FRAMEBUFFER_COMPLETE,
    viewport: (x: number, y: number, w: number, h: number) => { viewport = [x, y, w, h]; },
    drawArrays: () => {
      if (!program || !fb?.tex) throw new Error('drawArrays without program/framebuffer');
      const prog = program;
      const pass = PASSES.get(prog.fs)!;
      const out = fb.tex;
      if (viewport[2] > out.w || viewport[3] > out.h) {
        throw new Error(`viewport ${viewport[2]}x${viewport[3]} exceeds target ${out.w}x${out.h}`);
      }
      const u = (n: string) => {
        const v = prog.uniforms.get(n);
        if (v === undefined) throw new Error(`uniform ${n} not set`);
        return v;
      };
      const t = (n: string) => {
        const tex = units[u(n)];
        if (!tex) throw new Error(`no texture bound for ${n}`);
        if (tex === out) throw new Error(`feedback loop: ${n} is the render target`);
        return tex;
      };
      pass({ u, t, out, vw: viewport[2], vh: viewport[3] });
      draws++;
    },
    readPixels: (x: number, y: number, w: number, h: number, format: number, _type: number, buf: Float32Array) => {
      const tex = fb?.tex;
      if (!tex) throw new Error('readPixels without framebuffer');
      if (format !== GL.RGBA) throw new Error('readPixels: only RGBA/FLOAT emulated');
      if (x + w > tex.w || y + h > tex.h) throw new Error(`readPixels ${w}x${h} exceeds ${tex.w}x${tex.h}`);
      if (buf.length < w * h * 4) throw new Error('readPixels: buffer too small');
      for (let r = 0; r < h; r++) for (let c = 0; c < w; c++)
        for (let k = 0; k < 4; k++) buf[(r * w + c) * 4 + k] = tex.data[((y + r) * tex.w + x + c) * 4 + k];
    },
  };
  return gl;
}

function makeGPUModel(buffer: ArrayBuffer): GPUForwardPass {
  const map = new Map<string, WeightTensor>();
  for (const t of parseOnnxWeights(buffer)) map.set(t.name, t);
  const gl = makeFakeGL();
  const canvas = { getContext: () => gl } as unknown as OffscreenCanvas;
  return new GPUForwardPass(inferModelConfig(map), map, canvas);
}

// ---- Fixtures / models ----

function findModel(name: string): ArrayBuffer | null {
  for (const dir of ['/app/models', '/tmp/models', path.resolve(__dirname, '../../../../engine/output/models')]) {
    const p = path.join(dir, name);
    if (fs.existsSync(p)) {
      const b = fs.readFileSync(p);
      return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
    }
  }
  console.warn(`[webgl-emulated] ${name} not found, skipping`);
  return null;
}

function readJson(name: string) {
  const p = path.resolve(__dirname, name);
  return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, 'utf-8')) : null;
}

const v2fix = readJson('inference-fixtures-v2.json');
const v1fix = readJson('inference-fixtures.json');

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

describe('GPUForwardPass on emulated WebGL2', () => {
  it.skipIf(!v2fix)('v2 (9 planes, spatial head): single + batched forward match pure TS and PyTorch', () => {
    const buffer = findModel(v2fix.model.replace(/\.pt$/, '.onnx'));
    if (!buffer) return;
    const gpu = makeGPUModel(buffer);
    const cpu = createModelFromOnnx(buffer);
    expect(gpu.config.numInputPlanes).toBe(9);
    expect(gpu.config.policyHead).toBe('spatial');

    // forced-pass, pass-chain, player-1 and no-last-knight positions
    const picks = [0, 5, 10, 11, 12].map(i => v2fix.positions[i]);
    const inputs = picks.map((p: { state: Record<string, unknown> }) => {
      const s = p.state as unknown as { pieces: string[]; balls: string[]; touchedMask: string };
      const st: EngineState = {
        ...(p.state as unknown as EngineState),
        pieces: [BigInt(s.pieces[0]), BigInt(s.pieces[1])],
        balls: [BigInt(s.balls[0]), BigInt(s.balls[1])],
        touchedMask: BigInt(s.touchedMask),
      };
      return stateToTensor(st, undefined, 9);
    });

    let dCpu = 0, dPy = 0, dV = 0;
    const single = inputs.slice(0, 2).map(x => {
      const r = gpu.forward(x);
      return { policy: Float32Array.from(r.policy), value: r.value };
    });
    const batch = gpu.forwardBatch(inputs.slice(2));
    const all = [...single, ...batch];
    all.forEach((r, i) => {
      const c = cpu.forward(inputs[i]);
      dCpu = Math.max(dCpu, maxAbsDiff(r.policy, c.policy));
      dPy = Math.max(dPy, maxAbsDiff(r.policy.map(Math.exp), picks[i].policy.map(Math.exp)));
      dV = Math.max(dV, Math.abs(r.value - picks[i].value), Math.abs(r.value - c.value));
    });
    console.log(`[webgl-emulated v2] max|dlogp| vs pure TS=${dCpu.toExponential(2)}, ` +
      `max|dp| vs PyTorch=${dPy.toExponential(2)}, max|dv|=${dV.toExponential(2)}`);
    expect(dCpu).toBeLessThan(2e-3);   // includes PAD logits (-1e4) where float32 ulp is ~1e-3
    expect(dPy).toBeLessThan(1e-4);
    expect(dV).toBeLessThan(1e-4);
    gpu.dispose();
  }, 120_000);

  it.skipIf(!v1fix)('v1 (7 planes, FC head) is unchanged', () => {
    const buffer = findModel(v1fix.model.replace(/\.pt$/, '.onnx'));
    if (!buffer) return;
    const gpu = makeGPUModel(buffer);
    expect(gpu.config.numInputPlanes).toBe(7);
    expect(gpu.config.policyHead).toBe('fc');
    const inputs = v1fix.positions.slice(0, 4).map((p: { tensor: number[][][] }) => {
      const t = new Float32Array(7 * 56);
      for (let c = 0; c < 7; c++) for (let r = 0; r < 8; r++) for (let col = 0; col < 7; col++)
        t[c * 56 + r * 7 + col] = p.tensor[c][r][col];
      return t;
    });
    const results = [
      ...inputs.slice(0, 2).map((x: Float32Array) => {
        const r = gpu.forward(x);
        return { policy: Float32Array.from(r.policy), value: r.value };
      }),
      ...gpu.forwardBatch(inputs.slice(2)),
    ];
    let dP = 0, dV = 0;
    results.forEach((r, i) => {
      dP = Math.max(dP, maxAbsDiff(r.policy, v1fix.positions[i].policy));
      dV = Math.max(dV, Math.abs(r.value - v1fix.positions[i].value));
    });
    console.log(`[webgl-emulated v1] max|dlogp| vs PyTorch=${dP.toExponential(2)}, max|dv|=${dV.toExponential(2)}`);
    expect(dP).toBeLessThan(0.05);
    expect(dV).toBeLessThan(0.01);
    gpu.dispose();
  }, 120_000);
});
