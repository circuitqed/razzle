/**
 * React hook for managing the client-side AI Web Worker.
 *
 * Handles worker lifecycle, model loading, and MCTS search.
 * BigInt values are serialized as strings when communicating with the worker.
 *
 * IMPORTANT: The worker is terminated and recreated after each search to fully
 * release WASM linear memory. Without this, WASM memory grows monotonically
 * (pages are never returned to the OS) and iOS kills the tab after a few moves.
 * The model is cached in IndexedDB so reloading is fast (~300ms).
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import type { EngineState } from '../engine/state';
import type { MCTSConfig } from '../engine/mcts';
import { GL_CONTEXT_LOST } from '../engine/evaluator';

interface AIWorkerState {
  isLoaded: boolean;
  isLoading: boolean;
  isSearching: boolean;
  isRandom: boolean;
  /** Which backend is running: 'webgpu', 'webgl', or 'wasm' */
  backend: string | null;
  loadError: string | null;
  progress: { simsDone: number; totalSims: number; bestMove: number; value: number } | null;
}

interface UseAIWorkerReturn extends AIWorkerState {
  loadModel: (modelUrl: string, modelVersion: string) => void;
  loadRandomEvaluator: () => void;
  search: (state: EngineState, config?: Partial<MCTSConfig>) => Promise<{ bestMove: number; simsDone: number; value: number; searchMs?: number }>;
  abort: () => void;
  /**
   * Stop any in-flight search immediately. The pending search() rejects with
   * SEARCH_CANCELLED and the worker is rebuilt (terminating is the only
   * reliable stop: GPU/pure-TS searches block the worker thread, so an
   * 'abort' message isn't processed until the search ends anyway).
   */
  cancelSearch: () => void;
  /** Resolves true when model finishes loading, false if load fails or wasn't started */
  waitForLoad: () => Promise<boolean>;
}

export const SEARCH_CANCELLED = 'Search cancelled';

// Serialize EngineState for postMessage (BigInt -> string)
function serializeState(s: EngineState) {
  return {
    pieces: [s.pieces[0].toString(), s.pieces[1].toString()] as [string, string],
    balls: [s.balls[0].toString(), s.balls[1].toString()] as [string, string],
    currentPlayer: s.currentPlayer,
    touchedMask: s.touchedMask.toString(),
    hasPassed: s.hasPassed,
    lastKnightDst: s.lastKnightDst,
    ply: s.ply,
  };
}

export function useAIWorker(): UseAIWorkerReturn {
  const workerRef = useRef<Worker | null>(null);
  // Track model info for reloading after worker recycle
  const modelInfoRef = useRef<{ url: string; version: string; isRandom: boolean } | null>(null);
  // Use refs for internal state to avoid stale closures
  const isLoadedRef = useRef(false);
  const activeBackendRef = useRef<string | null>(null);

  const [state, setState] = useState<AIWorkerState>({
    isLoaded: false,
    isLoading: false,
    isSearching: false,
    isRandom: false,
    backend: null,
    loadError: null,
    progress: null,
  });

  // Pending promise resolvers
  const searchResolverRef = useRef<{
    resolve: (result: { bestMove: number; simsDone: number; value: number; searchMs?: number }) => void;
    reject: (error: Error) => void;
  } | null>(null);
  const loadResolverRef = useRef<{
    resolve: (success: boolean) => void;
  } | null>(null);
  const loadPromiseRef = useRef<Promise<boolean> | null>(null);

  // Track if component is mounted to avoid state updates after unmount
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  // Create a new worker and wire up message handlers
  const createWorker = useCallback((): Worker | null => {
    try {
      const worker = new Worker(
        new URL('../workers/ai.worker.ts', import.meta.url),
        { type: 'module' },
      );

      worker.onerror = (event) => {
        console.warn('[useAIWorker] Worker error:', event.message);
        event.preventDefault();
        // Clear dead worker refs so waitForLoad() returns false and search() throws immediately
        workerRef.current = null;
        loadPromiseRef.current = null;
        isLoadedRef.current = false;
        if (mountedRef.current) {
          setState((prev) => ({
            ...prev,
            isLoaded: false,
            isLoading: false,
            loadError: event.message || 'Worker initialization failed',
          }));
        }
        loadResolverRef.current?.resolve(false);
        loadResolverRef.current = null;
        searchResolverRef.current?.reject(new Error(event.message || 'Worker error'));
        searchResolverRef.current = null;
        // Auto-recover: recreate worker and reload model after a brief delay
        if (mountedRef.current && modelInfoRef.current) {
          setTimeout(() => {
            if (!mountedRef.current || !modelInfoRef.current) return;
            const newWorker = createWorker();
            if (newWorker) sendLoadMessage(newWorker, modelInfoRef.current);
          }, 1000);
        }
      };

      worker.onmessage = (event) => {
        const msg = event.data;
        switch (msg.type) {
          case 'loaded':
            if (msg.success) {
              isLoadedRef.current = true;
              activeBackendRef.current = msg.backend ?? (msg.isRandom ? 'random' : 'wasm');
              if (mountedRef.current) {
                setState((prev) => ({
                  ...prev,
                  isLoaded: true,
                  isLoading: false,
                  isRandom: msg.isRandom ?? false,
                  backend: activeBackendRef.current,
                  loadError: null,
                }));
              }
              loadResolverRef.current?.resolve(true);
            } else {
              isLoadedRef.current = false;
              if (mountedRef.current) {
                setState((prev) => ({
                  ...prev,
                  isLoaded: false,
                  isLoading: false,
                  loadError: msg.error,
                }));
              }
              loadResolverRef.current?.resolve(false);
            }
            loadResolverRef.current = null;
            break;

          case 'loading_progress':
            break;

          case 'search_progress':
            if (mountedRef.current) {
              setState((prev) => ({
                ...prev,
                progress: {
                  simsDone: msg.simsDone,
                  totalSims: msg.totalSims,
                  bestMove: msg.bestMove,
                  value: msg.value,
                },
              }));
            }
            break;

          case 'search_result':
            if (mountedRef.current) {
              setState((prev) => ({ ...prev, isSearching: false, progress: null }));
            }
            if (msg.success) {
              searchResolverRef.current?.resolve({
                bestMove: msg.bestMove,
                simsDone: msg.simsDone,
                value: msg.value,
                searchMs: msg.searchMs,
              });
            } else {
              searchResolverRef.current?.reject(new Error(msg.error));
            }
            searchResolverRef.current = null;

            break;
        }
      };

      workerRef.current = worker;
      return worker;
    } catch (err) {
      console.warn('[useAIWorker] Failed to create Web Worker:', err);
      if (mountedRef.current) {
        setState((prev) => ({
          ...prev,
          loadError: 'Web Worker not supported',
        }));
      }
      return null;
    }
  }, []);

  // Send load message to a worker
  const sendLoadMessage = useCallback((worker: Worker, info: { url: string; version: string; isRandom: boolean }) => {
    isLoadedRef.current = false;
    loadPromiseRef.current = new Promise<boolean>((resolve) => {
      loadResolverRef.current = { resolve };
    });
    if (mountedRef.current) {
      setState((prev) => ({ ...prev, isLoading: true, isLoaded: false, loadError: null }));
    }

    if (info.isRandom) {
      worker.postMessage({ type: 'load', modelUrl: '', modelVersion: 'random', useRandom: true });
    } else {
      worker.postMessage({ type: 'load', modelUrl: info.url, modelVersion: info.version });
    }

    return loadPromiseRef.current;
  }, []);

  // Terminate current worker and create a fresh one after a delay.
  // The delay gives iOS time to reclaim the old worker's WASM memory
  // before the new worker allocates its own.
  const recycleWorker = useCallback(() => {
    if (workerRef.current) {
      workerRef.current.terminate();
      workerRef.current = null;
    }
    isLoadedRef.current = false;
    // Clear loadPromiseRef so searchFn enters the polling loop
    loadPromiseRef.current = null;

    // Delay creation to let OS reclaim old worker's memory
    setTimeout(() => {
      if (!mountedRef.current || !modelInfoRef.current) return;
      const worker = createWorker();
      if (worker) {
        sendLoadMessage(worker, modelInfoRef.current);
      }
    }, 300);
  }, [createWorker, sendLoadMessage]);

  // Terminate the current worker right away and start a fresh one with the
  // current model. Rejects any pending search; anyone awaiting the old load
  // gets the new load's result instead.
  const resetWorker = useCallback(() => {
    if (workerRef.current) {
      workerRef.current.terminate();
      workerRef.current = null;
    }
    isLoadedRef.current = false;
    searchResolverRef.current?.reject(new Error(SEARCH_CANCELLED));
    searchResolverRef.current = null;
    const staleLoad = loadResolverRef.current;
    loadResolverRef.current = null;
    loadPromiseRef.current = null;
    if (mountedRef.current) {
      setState((prev) => ({ ...prev, isSearching: false, progress: null }));
    }

    const worker = createWorker();
    if (worker && modelInfoRef.current) {
      sendLoadMessage(worker, modelInfoRef.current).then((ok) => staleLoad?.resolve(ok));
    } else {
      staleLoad?.resolve(false);
    }
  }, [createWorker, sendLoadMessage]);

  // Create initial worker on mount
  useEffect(() => {
    createWorker();
    return () => {
      if (workerRef.current) {
        workerRef.current.terminate();
        workerRef.current = null;
      }
    };
  }, [createWorker]);

  // Public: load model (called on mount or model change)
  // Switching models gets a fresh worker: it frees the old model's WebGL
  // context (iOS caps live contexts) and avoids interleaving two loads in one
  // worker.
  const switchModel = useCallback((info: { url: string; version: string; isRandom: boolean }) => {
    const hadModel = modelInfoRef.current !== null;
    modelInfoRef.current = info;
    if (hadModel || !workerRef.current) {
      resetWorker();
    } else {
      sendLoadMessage(workerRef.current, info);
    }
  }, [resetWorker, sendLoadMessage]);

  const loadModel = useCallback((modelUrl: string, modelVersion: string) => {
    switchModel({ url: modelUrl, version: modelVersion, isRandom: false });
  }, [switchModel]);

  const loadRandomEvaluator = useCallback(() => {
    switchModel({ url: '', version: 'random', isRandom: true });
  }, [switchModel]);

  // Public: run MCTS search — waits for load if needed, recycles worker after
  const searchFn = useCallback(
    async (
      engineState: EngineState,
      config?: Partial<MCTSConfig>,
      retried = false,
    ): Promise<{ bestMove: number; simsDone: number; value: number }> => {
      // Wait for worker to be loaded (may be loading after a recycle).
      // Poll until loadPromiseRef appears (set by sendLoadMessage after recycle).
      for (let attempt = 0; attempt < 20 && !isLoadedRef.current; attempt++) {
        if (loadPromiseRef.current) {
          const loaded = await loadPromiseRef.current;
          if (!loaded) throw new Error('Model failed to load');
          break;
        }
        // Worker is being recycled — wait for sendLoadMessage to set the promise
        await new Promise(r => setTimeout(r, 100));
      }
      if (!workerRef.current || !isLoadedRef.current) {
        throw new Error('Worker not initialized or model not loaded');
      }

      if (mountedRef.current) {
        setState((prev) => ({ ...prev, isSearching: true, progress: null }));
      }

      // Only one search at a time: a new one supersedes (and cancels) any
      // still running, so a stale result can never resolve the new request.
      if (searchResolverRef.current) {
        resetWorker();
        return searchFn(engineState, config, retried);
      }

      let result: { bestMove: number; simsDone: number; value: number };
      try {
        result = await new Promise<{ bestMove: number; simsDone: number; value: number }>(
          (resolve, reject) => {
            searchResolverRef.current = { resolve, reject };
            workerRef.current!.postMessage({
              type: 'search',
              state: serializeState(engineState),
              config,
            });
          },
        );
      } catch (err) {
        // GPU context reclaimed by iOS (typically after backgrounding):
        // rebuild the worker and retry once.
        if (!retried && err instanceof Error && err.message === GL_CONTEXT_LOST) {
          resetWorker();
          return searchFn(engineState, config, true);
        }
        throw err;
      }

      // Recycle the worker to fully release WASM linear memory.
      // Both 'wasm' and 'webgpu' backends use ONNX Runtime WASM internally,
      // so memory grows monotonically. Recycle to prevent slowdown.
      // Skip for 'gpu' (custom WebGL), 'pure-ts', and 'random'.
      const backend = activeBackendRef.current;
      if (backend === 'wasm' || backend === 'webgpu') {
        recycleWorker();
      }

      return result;
    },
    [recycleWorker, resetWorker],
  );

  const abort = useCallback(() => {
    workerRef.current?.postMessage({ type: 'abort' });
  }, []);

  const cancelSearch = useCallback(() => {
    if (searchResolverRef.current) resetWorker();
  }, [resetWorker]);

  const waitForLoad = useCallback(async (): Promise<boolean> => {
    if (isLoadedRef.current) return true;
    if (!loadPromiseRef.current) return false;
    return loadPromiseRef.current;
  }, []);

  return {
    ...state,
    loadModel,
    loadRandomEvaluator,
    search: searchFn,
    abort,
    cancelSearch,
    waitForLoad,
  };
}
