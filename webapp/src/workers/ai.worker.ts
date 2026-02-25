/**
 * Web Worker for client-side AI computation.
 *
 * Handles model loading and MCTS search off the main thread.
 * BigInt values are serialized as strings in messages (BigInt can't cross postMessage).
 *
 * Backend selection:
 * - iOS: Pure TypeScript inference (no WASM, no SharedArrayBuffer)
 * - Desktop with WebGPU: ONNX Runtime with WebGPU backend
 * - Desktop fallback: ONNX Runtime with WASM backend
 */

import { OnnxEvaluator, PureTSEvaluator, RandomEvaluator } from '../engine/evaluator';
import { createModelFromOnnx } from '../engine/inference';
import { search, type MCTSConfig, DEFAULT_CONFIG } from '../engine/mcts';
import type { EngineState } from '../engine/state';
import { getCachedModel, cacheModel } from '../engine/modelCache';

// ONNX Runtime is loaded dynamically to handle import failures gracefully
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let ort: any = null;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let session: any = null;
let evaluator: OnnxEvaluator | PureTSEvaluator | RandomEvaluator | null = null;
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
    console.log('[ai.worker] Skipping WebGPU on iOS — using WASM backend');
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

  // On iOS, limit WASM threads to reduce memory pressure.
  // Default is navigator.hardwareConcurrency (6 on iPhone 15) which spawns
  // 5 pthread workers + the AI worker = 7 JS contexts with shared memory.
  // With numThreads=2, we get: AI worker + 1 pthread = 3 contexts total.
  if (isIOS) {
    ort.env.wasm.numThreads = 2;
    console.log('[ai.worker] iOS: limiting WASM threads to 2');
  }

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

interface AbortMessage {
  type: 'abort';
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
 * Load model using pure TypeScript inference (no WASM).
 * Parses ONNX protobuf for weight tensors and runs inference in JS.
 */
async function loadPureTS(msg: LoadMessage): Promise<void> {
  const modelBuffer = await getModelBuffer(msg);

  self.postMessage({ type: 'loading_progress', stage: 'initializing' });
  const t0 = performance.now();
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
      // iOS: use pure TypeScript inference to avoid WASM memory issues.
      // WASM linear memory only grows (never shrinks), and iOS kills tabs
      // that exceed memory limits after just a few MCTS searches.
      console.log('[ai.worker] iOS detected — using pure TypeScript inference');
      await loadPureTS(msg);
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
    // The MCTS tree holds hundreds of EngineState objects with BigInt bitboards —
    // clearing the reference immediately lets GC reclaim that memory between moves.
    const { bestMove, simsDone, value } = result;
    result.rootNode.children.clear();

    self.postMessage({
      type: 'search_result',
      success: true,
      bestMove,
      simsDone,
      value,
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
  const msg = event.data as LoadMessage | SearchMessage | AbortMessage;
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
