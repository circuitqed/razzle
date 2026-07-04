import { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import type { GameState, Player } from '../types';
import { decodeMove, squareToAlgebraic } from '../types';
import * as api from '../api/engine';
import { getOnnxModelInfoByName } from '../api/engine';
import { logger } from '../utils/logger';
import { playMoveSound, playPassSound, playWinSound, playLoseSound } from '../utils/sounds';
import { useAIWorker } from './useAIWorker';
import { useBoardInteraction } from './useBoardInteraction';
import type { EngineState } from '../engine/state';

interface UseGameOptions {
  vsAI?: boolean;
  aiSimulations?: number;
  aiModel?: string;  // Model path or 'random_weights'
  playerColor?: number;  // 0 = blue, 1 = red (only for AI games)
}

interface MoveRecord {
  move: number;
  algebraic: string;
  player: Player;
}

interface LastMove {
  from: number;
  to: number;
}

interface UseGameReturn {
  gameState: GameState | null;
  selectedSquare: number | null;
  isLoading: boolean;
  error: string | null;
  aiThinking: boolean;
  aiModelLoading: boolean;
  canEndTurn: boolean;
  mustPass: boolean;
  isPassing: boolean;
  lastMove: LastMove | null;
  lastTurnAnimMoves: LastMove[] | undefined;
  moveHistory: MoveRecord[];
  rawMoves: number[];
  evaluation: number | null;
  aiProgress: { simsDone: number; totalSims: number } | null;
  viewPly: number | null;
  isViewingHistory: boolean;
  startNewGame: () => Promise<void>;
  resumeGame: (gameId: string) => Promise<boolean>;
  handleSquareClick: (square: number) => void;
  handleDragMove: (from: number, to: number) => void;
  endTurn: () => void;
  undoMove: () => Promise<void>;
  resign: () => Promise<void>;
  cancelPass: () => void;
  goToMove: (ply: number) => void;
  goForward: () => void;
  goBack: () => void;
  goToStart: () => void;
  goToEnd: () => void;
}

const END_TURN_MOVE = -1;

/** Convert API GameState to client-side EngineState for local AI search. */
function apiStateToEngineState(gs: GameState): EngineState {
  return {
    pieces: [BigInt(gs.board.p1_pieces), BigInt(gs.board.p2_pieces)],
    balls: [BigInt(gs.board.p1_ball), BigInt(gs.board.p2_ball)],
    currentPlayer: gs.current_player,
    touchedMask: BigInt(gs.touched_mask),
    hasPassed: gs.has_passed,
    lastKnightDst: gs.last_knight_dst ?? -1,
    ply: gs.ply,
  };
}

export function useGame(options: UseGameOptions = {}): UseGameReturn {
  const { vsAI = true, aiSimulations = 256, aiModel, playerColor = 0 } = options;

  const [gameState, setGameState] = useState<GameState | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [aiThinking, setAiThinking] = useState(false);
  const aiThinkingRef = useRef(false); // synchronous guard against double-trigger
  const [lastMove, setLastMove] = useState<LastMove | null>(null);
  const [moveHistory, setMoveHistory] = useState<MoveRecord[]>([]);
  const [rawMoves, setRawMoves] = useState<number[]>([]);
  const [evaluation, setEvaluation] = useState<number | null>(null);
  const [aiProgress, setAiProgress] = useState<{ simsDone: number; totalSims: number } | null>(null);
  const [lastTurnAnimMoves, setLastTurnAnimMoves] = useState<LastMove[] | undefined>(undefined);

  // Generation counter: incremented on new game / resume so stale async ops bail out
  const gameGenRef = useRef(0);

  // Client-side AI worker
  const aiWorker = useAIWorker();
  const loadedModelRef = useRef<string | null>(null);
  // Set when the model info fetch itself fails (e.g. model missing on the
  // server). Distinct from aiWorker.loadError, which only covers failures
  // AFTER a load was started — without this, the AI-move trigger effect
  // retries forever (red error banner + flickering "thinking" indicator).
  const [modelUnavailable, setModelUnavailable] = useState<string | null>(null);

  // Which player the human controls (for AI games)
  const humanPlayer = vsAI ? playerColor : -1; // -1 means both
  const aiPlayer = vsAI ? (1 - playerColor) : -1;

  // Whether the human can interact with the board
  const isInteractionEnabled = useMemo(() => {
    if (aiThinking) return false;
    if (vsAI && gameState?.current_player === aiPlayer) return false;
    return true;
  }, [aiThinking, vsAI, gameState?.current_player, aiPlayer]);

  // Play sound when game ends
  const prevStatusRef = useRef<string | null>(null);
  useEffect(() => {
    if (!gameState) return;

    const wasPlaying = prevStatusRef.current === 'playing';
    prevStatusRef.current = gameState.status;

    if (wasPlaying && gameState.status === 'finished' && gameState.winner !== null) {
      if (vsAI) {
        if (gameState.winner === humanPlayer) {
          playWinSound();
        } else {
          playLoseSound();
        }
      } else {
        playWinSound();
      }
    }
  }, [gameState, vsAI, humanPlayer]);

  // Ref for stable access to aiWorker functions (avoids dependency on aiWorker object)
  const aiWorkerRef = useRef(aiWorker);
  aiWorkerRef.current = aiWorker;

  // Load client-side AI model reactively when aiModel changes
  useEffect(() => {
    if (!vsAI) return;

    const modelKey = aiModel ?? 'latest';
    if (loadedModelRef.current === modelKey) return;
    loadedModelRef.current = modelKey;
    setModelUnavailable(null);

    if (modelKey === 'random_weights') {
      aiWorkerRef.current.loadRandomEvaluator();
      logger.info('[useGame] Loading random evaluator for client-side AI');
      return;
    }

    const filename = modelKey === 'latest' ? null : modelKey.split('/').pop();
    const fetchAndLoad = filename
      ? () => getOnnxModelInfoByName(filename)
      : () => api.getOnnxModelInfo();

    fetchAndLoad()
      .then((modelInfo) => {
        if (loadedModelRef.current !== modelKey) return;
        aiWorkerRef.current.loadModel(modelInfo.url, modelInfo.version);
        logger.info('[useGame] Loading client-side AI model:', modelInfo.version);
      })
      .catch((err) => {
        if (loadedModelRef.current !== modelKey) return;
        // No server-side AI fallback exists (it's admin-gated in prod), so a
        // missing model means this level cannot be played right now. Fail
        // once with a clear message; retry only if the user switches levels.
        logger.error('[useGame] AI model unavailable:', err);
        loadedModelRef.current = null; // allow retry on reselect
        setModelUnavailable(modelKey);
        setError(`The AI for this level is unavailable (model ${filename ?? 'latest'} missing on the server). Try a different level.`);
      });
  }, [vsAI, aiModel]);

  // Handle AI move - compute the FULL turn, then apply all at once.
  // AI thinks silently; only the final board state is shown.
  const handleAIMove = useCallback(async (gameId: string) => {
    // Synchronous guard: prevent double-trigger from React batched updates
    if (aiThinkingRef.current) return;
    aiThinkingRef.current = true;

    const gen = gameGenRef.current;
    const isStale = () => gen !== gameGenRef.current;

    setAiThinking(true);
    try {
      let aiMoveCount = 0;
      const MAX_AI_MOVES = 10;
      const aiTurnMoves: number[] = [];
      let aiValue: number | null = null;

      let currentState = await api.getGameState(gameId);
      if (isStale()) return;

      while (currentState.status === 'playing' && currentState.current_player === aiPlayer) {
        aiMoveCount++;
        if (aiMoveCount > MAX_AI_MOVES) {
          logger.error('[useGame] AI move loop exceeded max iterations');
          setError('AI took too many moves in one turn');
          break;
        }

        // Try client-side AI first; wait for model load if still in progress
        let clientReady = aiWorkerRef.current.isLoaded;
        if (!clientReady) {
          // Wait for pending model load (returns immediately if already loaded/failed)
          clientReady = await aiWorkerRef.current.waitForLoad();
          if (isStale()) return;
        }

        logger.info('[useGame] AI making move...', { moveNumber: aiMoveCount, clientSide: clientReady });

        let aiMove: number;

        if (clientReady) {
          try {
            const engineState = apiStateToEngineState(currentState);
            const result = await aiWorkerRef.current.search(engineState, {
              numSimulations: aiSimulations,
            });
            if (isStale()) return;
            aiMove = result.bestMove;
            aiValue = result.value;
            if (result.searchMs) {
              const secs = (result.searchMs / 1000).toFixed(1);
              const msPerSim = (result.searchMs / result.simsDone).toFixed(1);
              const simsPerSec = (1000 * result.simsDone / result.searchMs).toFixed(1);
              logger.info('[useGame] Search perf', {
                sims: result.simsDone, secs, msPerSim, simsPerSec,
                backend: aiWorkerRef.current.backend,
              });
            }
            setAiProgress(null);
            try {
              currentState = await api.makeMove(gameId, aiMove);
            } catch (moveErr: any) {
              // If server rejects the move (stale state), refetch and retry this iteration
              if (moveErr?.status === 400) {
                logger.warn('[useGame] AI move rejected, refetching state', { move: aiMove });
                currentState = await api.getGameState(gameId);
                if (isStale()) return;
                aiMoveCount--; // don't count this failed attempt
                continue;
              }
              throw moveErr;
            }
            if (isStale()) return;
          } catch (clientErr) {
            if (isStale()) return;
            throw clientErr;
          }
        } else {
          // Model not ready — stop silently if still loading, throw if permanently failed
          if (aiWorkerRef.current.isLoading) return;
          throw new Error(`Client AI not ready: ${aiWorkerRef.current.loadError ?? 'not loaded'}`);
        }

        aiTurnMoves.push(aiMove);

        logger.info('[useGame] AI sub-move', {
          move: aiMove,
          newPlayer: currentState.current_player,
        });
      }

      if (isStale()) return;

      // Apply the full AI turn at once
      setGameState(currentState);

      if (aiValue !== undefined && aiValue !== null) {
        setEvaluation(-aiValue);
      }

      // Record all AI moves
      const newRecords: MoveRecord[] = [];
      for (const move of aiTurnMoves) {
        if (move !== END_TURN_MOVE) {
          const { src, dst } = decodeMove(move);
          newRecords.push({
            move,
            algebraic: `${squareToAlgebraic(src)}-${squareToAlgebraic(dst)}`,
            player: aiPlayer as Player,
          });
        }
      }
      setRawMoves(prev => [...prev, ...aiTurnMoves]);
      setMoveHistory(prev => [...prev, ...newRecords]);

      // Set lastMove to last non-END_TURN for animation
      const nonEndTurns = aiTurnMoves.filter(m => m !== END_TURN_MOVE);
      const lastActual = nonEndTurns[nonEndTurns.length - 1];
      if (lastActual !== undefined) {
        const { src, dst } = decodeMove(lastActual);
        setLastMove({ from: src, to: dst });

        // Multi-pass: set waypoint animation data
        if (nonEndTurns.length > 1) {
          setLastTurnAnimMoves(nonEndTurns.map(m => {
            const d = decodeMove(m);
            return { from: d.src, to: d.dst };
          }));
          playPassSound();
        } else {
          setLastTurnAnimMoves(undefined);
          playMoveSound();
        }
      }

      logger.info('[useGame] AI turn complete', {
        moves: aiTurnMoves,
        currentPlayer: currentState.current_player,
      });
    } catch (err) {
      if (isStale()) return;
      const msg = err instanceof Error ? err.message : 'AI move failed';
      logger.error('[useGame] AI move failed:', err);
      // Don't show "Worker not initialized" to user — the worker is recycling
      // and the retry will handle it silently via the useEffect re-trigger.
      if (!msg.includes('Worker not initialized')) {
        setError(msg);
      }
    } finally {
      if (!isStale()) {
        setAiThinking(false); aiThinkingRef.current = false;
        setAiProgress(null);
      }
    }
  }, [aiSimulations, aiModel, aiPlayer, aiWorker]);

  // Ref for handleAIMove so commitTurn can call it without circular dependency
  const handleAIMoveRef = useRef(handleAIMove);
  handleAIMoveRef.current = handleAIMove;

  // Guard against double-sends (e.g. from event bubbling)
  const commitInProgressRef = useRef(false);

  // Trigger AI move when it's AI's turn and nothing else is in progress.
  // Covers: new game (AI goes first), resumed game (AI's turn), etc.
  useEffect(() => {
    if (!vsAI || !gameState || gameState.status !== 'playing') return;
    if (aiThinking || isLoading) return;
    if (gameState.current_player !== aiPlayer) return;
    // Don't retry if model failed to load and isn't recovering
    if (aiWorker.loadError && !aiWorker.isLoaded && !aiWorker.isLoading) return;
    // Don't retry if the model info fetch failed (model missing server-side)
    if (modelUnavailable) return;
    handleAIMove(gameState.game_id);
  }, [vsAI, gameState?.game_id, gameState?.current_player, gameState?.status, aiPlayer, aiThinking, isLoading, aiWorker.loadError, aiWorker.isLoaded, aiWorker.isLoading, modelUnavailable]);

  // Commit a complete turn: send all sub-moves to the server, update state.
  const commitTurn = useCallback(
    (moves: number[]) => {
      if (!gameState) return;
      if (commitInProgressRef.current) {
        logger.info('[useGame] Ignoring duplicate commitTurn');
        return;
      }
      commitInProgressRef.current = true;

      const gen = gameGenRef.current;
      const isStale = () => gen !== gameGenRef.current;

      setIsLoading(true);
      setError(null);

      const gameId = gameState.game_id;
      const prevPlayer = gameState.current_player;

      (async () => {
        try {
          logger.info('[useGame] Sending turn to server:', moves);
          const currentState = await api.makeTurn(gameId, moves);
          if (isStale()) return;

          setGameState(currentState);

          // Use authoritative move list from server
          if (currentState.moves) {
            setRawMoves(currentState.moves);
          }

          // Record new moves in history
          const newRecords: MoveRecord[] = [];
          for (const move of moves) {
            if (move !== END_TURN_MOVE) {
              const { src, dst } = decodeMove(move);
              newRecords.push({
                move,
                algebraic: `${squareToAlgebraic(src)}-${squareToAlgebraic(dst)}`,
                player: prevPlayer,
              });
            }
          }
          setMoveHistory((prev) => [...prev, ...newRecords]);

          // Set lastMove for animation (human moves don't use multi-pass waypoints)
          setLastTurnAnimMoves(undefined);
          const lastActual = [...moves].reverse().find((m) => m !== END_TURN_MOVE);
          if (lastActual !== undefined) {
            const { src, dst } = decodeMove(lastActual);
            setLastMove({ from: src, to: dst });
          }

          // Trigger AI if needed
          if (vsAI && currentState.status === 'playing' && currentState.current_player === aiPlayer) {
            await handleAIMoveRef.current(currentState.game_id);
          }
        } catch (err) {
          if (isStale()) return;
          logger.error('[useGame] commitTurn failed:', err);
          setError(err instanceof Error ? err.message : 'Move failed');
        } finally {
          commitInProgressRef.current = false;
          if (!isStale()) {
            setIsLoading(false);
          }
        }
      })();
    },
    [gameState, vsAI, aiPlayer]
  );

  // Board interaction hook
  const {
    selectedSquare, clearSelection, handleSquareClick, handleDragMove,
    endTurn, cancelPass, canEndTurn, isPassing, mustPass,
    viewPly, isViewingHistory, effectiveGameState, displayLastMove,
    goToMove, goForward, goBack, goToStart, goToEnd,
  } = useBoardInteraction({
    gameState,
    rawMoves,
    lastMove,
    isInteractionEnabled,
    isLoading,
    commitTurn,
  });

  // Start a new game
  const startNewGame = useCallback(async () => {
    gameGenRef.current++;  // Invalidate any in-flight AI/commit operations
    setAiThinking(false); aiThinkingRef.current = false;
    setIsLoading(true);
    setError(null);
    clearSelection();
    goToEnd();
    setLastMove(null);
    setLastTurnAnimMoves(undefined);
    setMoveHistory([]);
    setRawMoves([]);
    setEvaluation(null);

    try {
      const { game_id } = await api.createGame({
        player1_type: 'human',
        player2_type: vsAI ? 'ai' : 'human',
        ai_simulations: aiSimulations,
      });
      const state = await api.getGameState(game_id);
      setGameState(state);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start game');
    } finally {
      setIsLoading(false);
    }
  }, [vsAI, aiSimulations, clearSelection, goToEnd]);

  // Resume an existing game (e.g., after page refresh)
  const resumeGame = useCallback(async (gameId: string): Promise<boolean> => {
    gameGenRef.current++;  // Invalidate any in-flight AI/commit operations
    setAiThinking(false); aiThinkingRef.current = false;
    setIsLoading(true);
    setError(null);
    clearSelection();
    goToEnd();
    setLastMove(null);
    setLastTurnAnimMoves(undefined);
    setMoveHistory([]);
    setRawMoves([]);
    setEvaluation(null);

    try {
      const state = await api.getGameState(gameId);

      // Only resume games that are still in progress
      if (state.status !== 'playing') {
        return false;
      }

      setGameState(state);

      // Restore move history from server
      if (state.moves) {
        setRawMoves(state.moves);

        // Derive lastMove from move history
        for (let i = state.moves.length - 1; i >= 0; i--) {
          if (state.moves[i] !== END_TURN_MOVE) {
            const { src, dst } = decodeMove(state.moves[i]);
            setLastMove({ from: src, to: dst });
            break;
          }
        }

        // Rebuild moveHistory records
        const records: MoveRecord[] = [];
        for (const move of state.moves) {
          if (move !== END_TURN_MOVE) {
            const { src, dst } = decodeMove(move);
            // Approximate the player from the move sequence
            records.push({
              move,
              algebraic: `${squareToAlgebraic(src)}-${squareToAlgebraic(dst)}`,
              player: 0 as Player, // Approximate - not critical for display
            });
          }
        }
        setMoveHistory(records);
      }

      // AI triggering handled by the useEffect that watches current_player
      return true;
    } catch (err) {
      logger.error('[useGame] Failed to resume game:', err);
      return false;
    } finally {
      setIsLoading(false);
    }
  }, [clearSelection, goToEnd]);

  // Undo last move
  const undoMove = useCallback(async () => {
    if (!gameState) return;

    goToEnd();
    setIsLoading(true);
    setError(null);
    clearSelection();

    try {
      let state = await api.undoMove(gameState.game_id);
      setMoveHistory(prev => prev.slice(0, -1));
      setRawMoves(prev => prev.slice(0, -1));
      if (vsAI && state.current_player === aiPlayer) {
        state = await api.undoMove(gameState.game_id);
        setMoveHistory(prev => prev.slice(0, -1));
        setRawMoves(prev => prev.slice(0, -1));
      }
      setGameState(state);
      const newHistory = moveHistory.slice(0, vsAI && state.current_player === humanPlayer ? -2 : -1);
      if (newHistory.length === 0) {
        setLastMove(null);
      } else {
        const lastRecord = newHistory[newHistory.length - 1];
        const { src, dst } = decodeMove(lastRecord.move);
        setLastMove({ from: src, to: dst });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Undo failed');
    } finally {
      setIsLoading(false);
    }
  }, [gameState, vsAI, aiPlayer, humanPlayer, moveHistory, goToEnd, clearSelection]);

  // Resign from the game
  const resign = useCallback(async () => {
    if (!gameState || gameState.status !== 'playing') return;

    setIsLoading(true);
    setError(null);
    goToEnd();

    try {
      const newState = await api.resignGame(gameState.game_id, humanPlayer === -1 ? 0 : humanPlayer);
      setGameState(newState);
      clearSelection();
      playLoseSound();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Resign failed');
    } finally {
      setIsLoading(false);
    }
  }, [gameState, humanPlayer, goToEnd, clearSelection]);

  return {
    gameState: effectiveGameState,
    selectedSquare,
    isLoading,
    error,
    aiThinking,
    aiModelLoading: aiWorker.isLoading,
    canEndTurn,
    mustPass,
    isPassing,
    lastMove: displayLastMove,
    lastTurnAnimMoves,
    moveHistory,
    rawMoves,
    evaluation,
    aiProgress,
    viewPly,
    isViewingHistory,
    startNewGame,
    resumeGame,
    handleSquareClick,
    handleDragMove,
    endTurn,
    undoMove,
    resign,
    cancelPass,
    goToMove,
    goForward,
    goBack,
    goToStart,
    goToEnd,
  };
}
