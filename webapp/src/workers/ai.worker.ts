/**
 * Web Worker for client-side AI computation.
 *
 * Handles model loading and MCTS search off the main thread.
 * BigInt values are serialized as strings in messages (BigInt can't cross postMessage).
 *
 * Backend selection:
 * - iOS: MCTS in worker, inference on main thread via WebGL (RemoteEvaluator)
 * - Desktop with WebGPU: ONNX Runtime with WebGPU backend
 * - Desktop fallback: ONNX Runtime with WASM backend
 */

import { OnnxEvaluator, PureTSEvaluator, GPUEvaluator, RandomEvaluator } from '../engine/evaluator';
import type { Evaluator } from '../engine/evaluator';
import { createModelFromOnnx } from '../engine/inference';
import { createGPUModelFromOnnx } from '../engine/webglForwardPass';
import { search, type MCTSConfig, DEFAULT_CONFIG } from '../engine/mcts';
import type { EngineState } from '../engine/state';
import { getCachedModel, cacheModel } from '../engine/modelCache';

// ONNX Runtime is loaded dynamically to handle import failures gracefully
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let ort: any = null;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let session: any = null;
let evaluator: Evaluator | null = null;
let abortFlag = { aborted: false };
let activeBackend: string = 'wasm';
let ortLoadFailed = false;


/** Detect iOS/iPadOS (all browsers on iOS use WebKit) */
const isIOS = /iPhone|iPad|iPod/i.test(navigator.userAgent) ||
  (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);

/**
 * Check if WebGPU is actually usable (not just present in the API).
 * On iOS, navigator.gpu exists but WebGPU inference in Workers crashes the process.
 */
async function isWebGPUUsable(): Promise<boolean> {
  if (isIOS) {
    console.log('[ai.worker] Skipping WebGPU on iOS');
    return false;
  }
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) return false;
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const gpu = (navigator as any).gpu;
    const adapter = await gpu.requestAdapter();
    return adapter !== null;
  } catch {
    return false;
  }
}

/**
 * Dynamically load ONNX Runtime.
 * On platforms with working WebGPU, loads the 'all' bundle (WebGPU + WASM).
 * Otherwise, loads the base WASM-only bundle.
 */
async function ensureOrt(needsWebGPU: boolean): Promise<boolean> {
  if (ort) return true;
  if (ortLoadFailed) return false;

  if (needsWebGPU) {
    try {
      ort = await import('onnxruntime-web/all');
    } catch {
      console.warn('[ai.worker] Full ONNX bundle failed, trying WASM-only');
      needsWebGPU = false; // fall through to WASM-only
    }
  }

  if (!ort) {
    try {
      ort = await import('onnxruntime-web');
    } catch (err) {
      console.error('[ai.worker] Failed to load ONNX Runtime:', err);
      ortLoadFailed = true;
      return false;
    }
  }

  // Make ort available globally for the evaluator
  (self as unknown as Record<string, unknown>).ort = ort;

  // Tell ONNX Runtime where to find WASM files (copied to root by vite-plugin-static-copy)
  ort.env.wasm.wasmPaths = '/';

  return true;
}

// Types for messages to/from worker
interface LoadMessage {
  type: 'load';
  modelUrl: string;
  modelVersion: string;
  useRandom?: boolean;
}

interface SearchMessage {
  type: 'search';
  state: SerializedEngineState;
  config?: Partial<MCTSConfig>;
}

// BigInt can't be serialized via postMessage, so we use strings
interface SerializedEngineState {
  pieces: [string, string];
  balls: [string, string];
  currentPlayer: number;
  touchedMask: string;
  hasPassed: boolean;
  lastKnightDst: number;
  ply: number;
}

function deserializeState(s: SerializedEngineState): EngineState {
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

/**
 * Download (or load from cache) the ONNX model file.
 */
async function getModelBuffer(msg: LoadMessage): Promise<ArrayBuffer> {
  let modelBuffer = await getCachedModel(msg.modelVersion);

  if (!modelBuffer) {
    self.postMessage({ type: 'loading_progress', stage: 'downloading' });
    const response = await fetch(msg.modelUrl);
    if (!response.ok) {
      throw new Error(`Failed to download model: ${response.status}`);
    }
    modelBuffer = await response.arrayBuffer();
    await cacheModel(msg.modelVersion, modelBuffer);
  }

  return modelBuffer;
}

/**
 * Load model for iOS — try WebGL GEMM first (GPU-accelerated, accurate),
 * fall back to pure TypeScript if WebGL is unavailable.
 */
async function loadiOS(msg: LoadMessage): Promise<void> {
  const modelBuffer = await getModelBuffer(msg);

  self.postMessage({ type: 'loading_progress', stage: 'initializing' });
  const t0 = performance.now();

  // GPU-resident forward pass: all convolutions on GPU, only 2 readPixels per forward.
  // Accuracy verified on iOS Safari via test-webgl-inference.html (OffscreenCanvas path).
  try {
    if (typeof OffscreenCanvas === 'undefined') throw new Error('No OffscreenCanvas');
    const canvas = new OffscreenCanvas(1, 1);
    const gl = canvas.getContext('webgl2');
    if (!gl) throw new Error('No WebGL2 in worker');
    if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('No EXT_color_buffer_float');

    const gpuModel = createGPUModelFromOnnx(modelBuffer, canvas);
    const elapsed = ((performance.now() - t0)).toFixed(0);
    console.log(`[ai.worker] GPU forward pass model created in ${elapsed}ms:`, gpuModel.config);

    evaluator = new GPUEvaluator(gpuModel);
    activeBackend = 'gpu';
    self.postMessage({ type: 'loaded', success: true, isRandom: false, backend: activeBackend });
    return;
  } catch (gpuErr) {
    console.warn('[ai.worker] GPU forward pass unavailable, using pure TypeScript:', gpuErr);
  }

  // Fall back to pure TypeScript inference
  const model = createModelFromOnnx(modelBuffer);
  const elapsed = ((performance.now() - t0)).toFixed(0);
  console.log(`[ai.worker] Pure TS model created in ${elapsed}ms:`, model.config);

  evaluator = new PureTSEvaluator(model);
  activeBackend = 'pure-ts';
  self.postMessage({ type: 'loaded', success: true, isRandom: false, backend: activeBackend });
}

/**
 * Load model using ONNX Runtime (WASM or WebGPU).
 */
async function loadOnnxRuntime(msg: LoadMessage): Promise<void> {
  const useWebGPU = await isWebGPUUsable();
  console.log('[ai.worker] WebGPU usable:', useWebGPU);

  const ortReady = await ensureOrt(useWebGPU);
  if (!ortReady) {
    throw new Error('ONNX Runtime not available on this device');
  }

  const modelBuffer = await getModelBuffer(msg);

  self.postMessage({ type: 'loading_progress', stage: 'initializing' });

  if (useWebGPU) {
    try {
      session = await ort.InferenceSession.create(modelBuffer, {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all',
      });
      activeBackend = 'webgpu';
    } catch (gpuErr) {
      console.warn('[ai.worker] WebGPU session failed, falling back to WASM:', gpuErr);
      session = await ort.InferenceSession.create(modelBuffer, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      activeBackend = 'wasm';
    }
  } else {
    session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    activeBackend = 'wasm';
  }

  evaluator = new OnnxEvaluator(session!);
  self.postMessage({ type: 'loaded', success: true, isRandom: false, backend: activeBackend });
}

async function handleLoad(msg: LoadMessage): Promise<void> {
  try {
    if (msg.useRandom) {
      evaluator = new RandomEvaluator();
      self.postMessage({ type: 'loaded', success: true, isRandom: true });
      return;
    }

    if (isIOS) {
      // iOS: try WebGL GEMM (custom shaders, GPU-accelerated, accurate).
      // Falls back to pure TypeScript if WebGL unavailable.
      await loadiOS(msg);
    } else {
      // Desktop: use ONNX Runtime for WASM SIMD / WebGPU acceleration
      await loadOnnxRuntime(msg);
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error loading model';
    self.postMessage({ type: 'loaded', success: false, error: message });
  }
}

async function handleSearch(msg: SearchMessage): Promise<void> {
  if (!evaluator) {
    self.postMessage({
      type: 'search_result',
      success: false,
      error: 'No model loaded',
    });
    return;
  }

  abortFlag = { aborted: false };

  try {
    const state = deserializeState(msg.state);
    const config = { ...DEFAULT_CONFIG, ...msg.config };
    const searchStart = performance.now();

    const result = await search(state, evaluator, config, abortFlag, (progress) => {
      self.postMessage({
        type: 'search_progress',
        simsDone: progress.simsDone,
        totalSims: progress.totalSims,
        bestMove: progress.bestMove,
        value: progress.value,
      });
    });

    // Extract scalars before dropping the tree reference.
    const { bestMove, simsDone, value } = result;
    const searchMs = performance.now() - searchStart;
    result.rootNode.children.clear();

    self.postMessage({
      type: 'search_result',
      success: true,
      bestMove,
      simsDone,
      value,
      searchMs,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Search failed';
    self.postMessage({
      type: 'search_result',
      success: false,
      error: message,
    });
  }
}

// Signal that the worker is alive (even before ONNX loads)
self.postMessage({ type: 'worker_ready' });

self.onmessage = (event: MessageEvent) => {
  const msg = event.data;
  switch (msg.type) {
    case 'load':
      handleLoad(msg);
      break;
    case 'search':
      handleSearch(msg);
      break;
    case 'abort':
      abortFlag.aborted = true;
      break;
  }
};
