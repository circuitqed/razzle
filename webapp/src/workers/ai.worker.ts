/**
 * Web Worker for client-side AI computation.
 *
 * Handles ONNX model loading and MCTS search off the main thread.
 * BigInt values are serialized as strings in messages (BigInt can't cross postMessage).
 *
 * GPU acceleration: uses WebGPU when available, falls back to WASM.
 * WebGPU is fastest (direct GPU compute shaders), WASM is the CPU fallback.
 */

// Import 'all' bundle to get WebGPU + WASM backends.
// We detect WebGPU availability before trying it to avoid poisoning initWasm().
import * as ort from 'onnxruntime-web/all';
import { OnnxEvaluator, RandomEvaluator } from '../engine/evaluator';
import { search, type MCTSConfig, DEFAULT_CONFIG } from '../engine/mcts';
import type { EngineState } from '../engine/state';
import { getCachedModel, cacheModel } from '../engine/modelCache';

// Make ort available globally for the evaluator
(self as unknown as Record<string, unknown>).ort = ort;

// Tell ONNX Runtime where to find WASM files (copied to root by vite-plugin-static-copy)
ort.env.wasm.wasmPaths = '/';

let session: ort.InferenceSession | null = null;
let evaluator: OnnxEvaluator | RandomEvaluator | null = null;
let abortFlag = { aborted: false };
let activeBackend: string = 'wasm';

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

async function handleLoad(msg: LoadMessage): Promise<void> {
  try {
    if (msg.useRandom) {
      evaluator = new RandomEvaluator();
      self.postMessage({ type: 'loaded', success: true, isRandom: true });
      return;
    }

    // Try IndexedDB cache first
    let modelBuffer = await getCachedModel(msg.modelVersion);

    if (!modelBuffer) {
      // Download model
      self.postMessage({ type: 'loading_progress', stage: 'downloading' });
      const response = await fetch(msg.modelUrl);
      if (!response.ok) {
        throw new Error(`Failed to download model: ${response.status}`);
      }
      modelBuffer = await response.arrayBuffer();

      // Cache for next time
      await cacheModel(msg.modelVersion, modelBuffer);
    }

    // Create ONNX session — try WebGPU if available, otherwise WASM.
    // We check navigator.gpu first to avoid trying WebGPU when it's not
    // supported, which would poison onnxruntime's initWasm() cache and
    // prevent the WASM fallback from working.
    self.postMessage({ type: 'loading_progress', stage: 'initializing' });

    const hasWebGPU = typeof navigator !== 'undefined' && 'gpu' in navigator;

    if (hasWebGPU) {
      try {
        session = await ort.InferenceSession.create(modelBuffer, {
          executionProviders: ['webgpu'],
          graphOptimizationLevel: 'all',
        });
        activeBackend = 'webgpu';
      } catch (gpuErr) {
        console.warn('[ai.worker] WebGPU failed, falling back to WASM:', gpuErr);
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

    self.postMessage({
      type: 'search_result',
      success: true,
      bestMove: result.bestMove,
      simsDone: result.simsDone,
      value: result.value,
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
