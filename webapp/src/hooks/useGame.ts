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
  moveHistory: MoveRecord[];
  rawMoves: number[];
  evaluation: number | null;
  aiProgress: { simsDone: number; totalSims: number } | null;
  viewPly: number | null;
  isViewingHistory: boolean;
  startNewGame: () => Promise<void>;
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
  const [lastMove, setLastMove] = useState<LastMove | null>(null);
  const [moveHistory, setMoveHistory] = useState<MoveRecord[]>([]);
  const [rawMoves, setRawMoves] = useState<number[]>([]);
  const [evaluation, setEvaluation] = useState<number | null>(null);
  const [aiProgress, setAiProgress] = useState<{ simsDone: number; totalSims: number } | null>(null);

  // Client-side AI worker
  const aiWorker = useAIWorker();
  const loadedModelRef = useRef<string | null>(null);

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
        logger.info('[useGame] ONNX model not available, using server AI:', err);
      });
  }, [vsAI, aiModel]);

  // Handle AI move - compute the FULL turn, then apply all at once.
  // AI thinks silently; only the final board state is shown.
  const handleAIMove = useCallback(async (gameId: string) => {
    setAiThinking(true);
    try {
      let aiMoveCount = 0;
      const MAX_AI_MOVES = 10;
      const aiTurnMoves: number[] = [];
      let aiValue: number | null = null;

      let currentState = await api.getGameState(gameId);

      while (currentState.status === 'playing' && currentState.current_player === aiPlayer) {
        aiMoveCount++;
        if (aiMoveCount > MAX_AI_MOVES) {
          logger.error('[useGame] AI move loop exceeded max iterations');
          setError('AI took too many moves in one turn');
          break;
        }

        const clientReady = aiWorker.isLoaded;
        if (!clientReady && !aiWorker.isLoading) {
          logger.warn('[useGame] Client-side AI not available', {
            loadError: aiWorker.loadError,
            backend: aiWorker.backend,
          });
        }

        logger.info('[useGame] AI making move...', { moveNumber: aiMoveCount, clientSide: clientReady });

        let aiMove: number;

        if (clientReady) {
          try {
            const engineState = apiStateToEngineState(currentState);
            const result = await aiWorker.search(engineState, {
              numSimulations: aiSimulations,
            });
            aiMove = result.bestMove;
            aiValue = result.value;
            setAiProgress(null);
            currentState = await api.makeMove(gameId, aiMove);
          } catch (clientErr) {
            logger.error('[useGame] Client-side AI failed, falling back to server:', clientErr);
            const aiResponse = await api.getAIMove(gameId, { simulations: aiSimulations, model: aiModel });
            currentState = aiResponse.game_state;
            aiMove = aiResponse.move;
            aiValue = aiResponse.value;
          }
        } else {
          const aiResponse = await api.getAIMove(gameId, { simulations: aiSimulations, model: aiModel });
          currentState = aiResponse.game_state;
          aiMove = aiResponse.move;
          aiValue = aiResponse.value;
        }

        aiTurnMoves.push(aiMove);

        logger.info('[useGame] AI sub-move', {
          move: aiMove,
          newPlayer: currentState.current_player,
        });
      }

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
      const lastActual = [...aiTurnMoves].reverse().find(m => m !== END_TURN_MOVE);
      if (lastActual !== undefined) {
        const { src, dst } = decodeMove(lastActual);
        setLastMove({ from: src, to: dst });
        // Play sound for the turn result
        if (aiTurnMoves.some(m => m !== END_TURN_MOVE && newRecords.length > 0)) {
          // If there were passes (more than one non-END_TURN move), play pass sound
          const nonEndTurns = aiTurnMoves.filter(m => m !== END_TURN_MOVE);
          if (nonEndTurns.length > 1) {
            playPassSound();
          } else {
            playMoveSound();
          }
        }
      }

      logger.info('[useGame] AI turn complete', {
        moves: aiTurnMoves,
        currentPlayer: currentState.current_player,
      });
    } catch (err) {
      logger.error('[useGame] AI move failed:', err);
      setError(err instanceof Error ? err.message : 'AI move failed');
    } finally {
      setAiThinking(false);
      setAiProgress(null);
    }
  }, [aiSimulations, aiModel, aiPlayer, aiWorker]);

  // Ref for handleAIMove so commitTurn can call it without circular dependency
  const handleAIMoveRef = useRef(handleAIMove);
  handleAIMoveRef.current = handleAIMove;

  // Trigger AI move when it's AI's turn (e.g., after new game where AI goes first)
  useEffect(() => {
    if (!vsAI || !gameState || gameState.status !== 'playing') return;
    if (aiThinking || isLoading) return;
    if (gameState.current_player !== aiPlayer) return;
    if (gameState.ply === 0 && aiPlayer === 0) {
      handleAIMove(gameState.game_id);
    }
  }, [vsAI, gameState?.game_id, gameState?.current_player, gameState?.ply, aiPlayer, aiThinking, isLoading]);

  // Commit a complete turn: send all sub-moves to the server, update state.
  const commitTurn = useCallback(
    (moves: number[]) => {
      if (!gameState) return;

      setIsLoading(true);
      setError(null);

      const gameId = gameState.game_id;
      const prevPlayer = gameState.current_player;

      (async () => {
        try {
          let currentState = gameState;
          for (const move of moves) {
            currentState = await api.makeMove(gameId, move);
          }

          setGameState(currentState);

          // Record moves
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
          setRawMoves((prev) => [...prev, ...moves]);
          setMoveHistory((prev) => [...prev, ...newRecords]);

          // Set lastMove for animation
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
          logger.error('[useGame] commitTurn failed:', err);
          setError(err instanceof Error ? err.message : 'Move failed');
        } finally {
          setIsLoading(false);
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
    setIsLoading(true);
    setError(null);
    clearSelection();
    goToEnd();
    setLastMove(null);
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
    moveHistory,
    rawMoves,
    evaluation,
    aiProgress,
    viewPly,
    isViewingHistory,
    startNewGame,
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
