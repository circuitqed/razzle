/**
 * React hook for managing the client-side AI Web Worker.
 *
 * Handles worker lifecycle, model loading, and MCTS search.
 * BigInt values are serialized as strings when communicating with the worker.
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import type { EngineState } from '../engine/state';
import type { MCTSConfig } from '../engine/mcts';

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
  search: (state: EngineState, config?: Partial<MCTSConfig>) => Promise<{ bestMove: number; simsDone: number; value: number }>;
  abort: () => void;
  /** Resolves true when model finishes loading, false if load fails or wasn't started */
  waitForLoad: () => Promise<boolean>;
}

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
  const [state, setState] = useState<AIWorkerState>({
    isLoaded: false,
    isLoading: false,
    isSearching: false,
    isRandom: false,
    backend: null,
    loadError: null,
    progress: null,
  });

  // Pending search promise resolver
  const searchResolverRef = useRef<{
    resolve: (result: { bestMove: number; simsDone: number; value: number }) => void;
    reject: (error: Error) => void;
  } | null>(null);

  // Pending load promise resolver + shared promise for waitForLoad
  const loadResolverRef = useRef<{
    resolve: () => void;
    reject: (error: Error) => void;
  } | null>(null);
  const loadPromiseRef = useRef<Promise<boolean> | null>(null);

  // Create worker on mount
  useEffect(() => {
    const worker = new Worker(
      new URL('../workers/ai.worker.ts', import.meta.url),
      { type: 'module' },
    );

    worker.onmessage = (event) => {
      const msg = event.data;
      switch (msg.type) {
        case 'loaded':
          if (msg.success) {
            setState((prev) => ({
              ...prev,
              isLoaded: true,
              isLoading: false,
              isRandom: msg.isRandom ?? false,
              backend: msg.backend ?? (msg.isRandom ? 'random' : 'wasm'),
              loadError: null,
            }));
            loadResolverRef.current?.resolve();
          } else {
            setState((prev) => ({
              ...prev,
              isLoaded: false,
              isLoading: false,
              loadError: msg.error,
            }));
            loadResolverRef.current?.reject(new Error(msg.error));
          }
          loadResolverRef.current = null;
          break;

        case 'loading_progress':
          // Could show download/init progress in the future
          break;

        case 'search_progress':
          setState((prev) => ({
            ...prev,
            progress: {
              simsDone: msg.simsDone,
              totalSims: msg.totalSims,
              bestMove: msg.bestMove,
              value: msg.value,
            },
          }));
          break;

        case 'search_result':
          setState((prev) => ({ ...prev, isSearching: false, progress: null }));
          if (msg.success) {
            searchResolverRef.current?.resolve({
              bestMove: msg.bestMove,
              simsDone: msg.simsDone,
              value: msg.value,
            });
          } else {
            searchResolverRef.current?.reject(new Error(msg.error));
          }
          searchResolverRef.current = null;
          break;
      }
    };

    workerRef.current = worker;

    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  const loadModel = useCallback((modelUrl: string, modelVersion: string) => {
    if (!workerRef.current) return;
    setState((prev) => ({ ...prev, isLoading: true, loadError: null }));
    loadPromiseRef.current = new Promise<boolean>((resolve) => {
      loadResolverRef.current = {
        resolve: () => resolve(true),
        reject: () => resolve(false),
      };
    });
    workerRef.current.postMessage({
      type: 'load',
      modelUrl,
      modelVersion,
    });
  }, []);

  const loadRandomEvaluator = useCallback(() => {
    if (!workerRef.current) return;
    setState((prev) => ({ ...prev, isLoading: true, loadError: null }));
    workerRef.current.postMessage({
      type: 'load',
      modelUrl: '',
      modelVersion: 'random',
      useRandom: true,
    });
  }, []);

  const searchFn = useCallback(
    (
      engineState: EngineState,
      config?: Partial<MCTSConfig>,
    ): Promise<{ bestMove: number; simsDone: number; value: number }> => {
      return new Promise((resolve, reject) => {
        if (!workerRef.current) {
          reject(new Error('Worker not initialized'));
          return;
        }
        if (!state.isLoaded) {
          reject(new Error('Model not loaded'));
          return;
        }

        searchResolverRef.current = { resolve, reject };
        setState((prev) => ({ ...prev, isSearching: true, progress: null }));

        workerRef.current.postMessage({
          type: 'search',
          state: serializeState(engineState),
          config,
        });
      });
    },
    [state.isLoaded],
  );

  const abort = useCallback(() => {
    workerRef.current?.postMessage({ type: 'abort' });
  }, []);

  const waitForLoad = useCallback(async (): Promise<boolean> => {
    if (state.isLoaded) return true;
    if (!loadPromiseRef.current) return false;
    return loadPromiseRef.current;
  }, [state.isLoaded]);

  return {
    ...state,
    loadModel,
    loadRandomEvaluator,
    search: searchFn,
    abort,
    waitForLoad,
  };
}
