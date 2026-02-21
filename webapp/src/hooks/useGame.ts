import { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import type { GameState, Player } from '../types';
import { encodeMove, decodeMove, TOTAL_SQUARES, squareToAlgebraic } from '../types';
import * as api from '../api/engine';
import { getOnnxModelInfoByName } from '../api/engine';
import { logger } from '../utils/logger';
import { playMoveSound, playPassSound, playEndTurnSound, playWinSound, playLoseSound, playSelectSound } from '../utils/sounds';
import { useAIWorker } from './useAIWorker';
import { replayToPosition, getLastMoveAtPosition } from '../utils/replay';
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
  /** Whether a pass chain is in progress (ball has been passed at least once) */
  isPassing: boolean;
  lastMove: LastMove | null;
  moveHistory: MoveRecord[];
  /** Raw move integers including END_TURN (-1) for proper notation display */
  rawMoves: number[];
  /** AI's evaluation of the position (from human's perspective: +1 = human winning) */
  evaluation: number | null;
  /** Client-side AI search progress */
  aiProgress: { simsDone: number; totalSims: number } | null;
  /** Current ply being viewed (null = live position) */
  viewPly: number | null;
  /** Whether we're viewing a historical position */
  isViewingHistory: boolean;
  startNewGame: () => Promise<void>;
  handleSquareClick: (square: number) => void;
  handleDragMove: (from: number, to: number) => void;
  endTurn: () => Promise<void>;
  undoMove: () => Promise<void>;
  resign: () => Promise<void>;
  /** Cancel the current pass chain (undo all passes, return to pre-pass state) */
  cancelPass: () => Promise<void>;
  /** Navigate to a specific ply in history (view-only) */
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
  const [selectedSquare, setSelectedSquare] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [aiThinking, setAiThinking] = useState(false);
  const [lastMove, setLastMove] = useState<LastMove | null>(null);
  const [moveHistory, setMoveHistory] = useState<MoveRecord[]>([]);
  const [rawMoves, setRawMoves] = useState<number[]>([]);
  const [evaluation, setEvaluation] = useState<number | null>(null);
  const [aiProgress, setAiProgress] = useState<{ simsDone: number; totalSims: number } | null>(null);
  const [viewPly, setViewPly] = useState<number | null>(null);

  // Client-side AI worker
  const aiWorker = useAIWorker();
  const loadedModelRef = useRef<string | null>(null);

  // Ref to track move-in-progress synchronously (avoids closure staleness)
  const moveInProgress = useRef(false);

  // Track pass chain length for cancelPass
  const passChainStartRef = useRef<number>(0);

  // Which player the human controls (for AI games)
  const humanPlayer = vsAI ? playerColor : -1; // -1 means both
  const aiPlayer = vsAI ? (1 - playerColor) : -1;

  // Check if end turn is available
  const canEndTurn = useMemo(() => {
    if (!gameState) return false;
    return gameState.legal_moves.includes(END_TURN_MOVE);
  }, [gameState]);

  // Whether a pass chain is in progress
  const isPassing = useMemo(() => {
    return gameState?.has_passed ?? false;
  }, [gameState]);

  // Check if player must pass (forced pass rule - opponent moved adjacent to ball)
  const mustPass = useMemo(() => {
    if (!gameState || gameState.has_passed) return false;

    // Get ball position for current player
    const ballBitboard = gameState.current_player === 0
      ? gameState.board.p1_ball
      : gameState.board.p2_ball;

    // Find the ball square (first set bit)
    let ballSquare = -1;
    for (let i = 0; i < TOTAL_SQUARES; i++) {
      if ((BigInt(ballBitboard) & (BigInt(1) << BigInt(i))) !== BigInt(0)) {
        ballSquare = i;
        break;
      }
    }

    if (ballSquare === -1) return false;

    // Check if all legal moves (except end turn) are passes from the ball position
    const realMoves = gameState.legal_moves.filter(m => m !== END_TURN_MOVE);
    if (realMoves.length === 0) return false;

    return realMoves.every(move => {
      const { src } = decodeMove(move);
      return src === ballSquare;
    });
  }, [gameState]);

  // History navigation state
  const isViewingHistory = viewPly !== null;
  const totalPlies = rawMoves.length;

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

  // Helper to record a move and play sound
  const recordMove = useCallback((move: number, player: Player, isPass: boolean) => {
    // Always add to rawMoves for proper notation display
    setRawMoves(prev => [...prev, move]);

    if (move === END_TURN_MOVE) return; // Don't record end turn in formatted history
    const { src, dst } = decodeMove(move);
    const algebraic = `${squareToAlgebraic(src)}-${squareToAlgebraic(dst)}`;
    setLastMove({ from: src, to: dst });
    setMoveHistory(prev => [...prev, { move, algebraic, player }]);

    // Play appropriate sound
    if (isPass) {
      playPassSound();
    } else {
      playMoveSound();
    }
  }, []);

  // Ref for stable access to aiWorker functions (avoids dependency on aiWorker object)
  const aiWorkerRef = useRef(aiWorker);
  aiWorkerRef.current = aiWorker;

  // Load client-side AI model reactively when aiModel changes
  useEffect(() => {
    if (!vsAI) return;

    // Determine which model key we want loaded
    const modelKey = aiModel ?? 'latest';

    // Skip if already loaded or currently loading this model
    if (loadedModelRef.current === modelKey) return;

    // Mark this model as the target (prevents duplicate loads and retries)
    loadedModelRef.current = modelKey;

    if (modelKey === 'random_weights') {
      aiWorkerRef.current.loadRandomEvaluator();
      logger.info('[useGame] Loading random evaluator for client-side AI');
      return;
    }

    // Extract filename from path for the by-name endpoint
    const filename = modelKey === 'latest' ? null : modelKey.split('/').pop();

    const fetchAndLoad = filename
      ? () => getOnnxModelInfoByName(filename)
      : () => api.getOnnxModelInfo();

    fetchAndLoad()
      .then((modelInfo) => {
        // Check we haven't switched models while fetching
        if (loadedModelRef.current !== modelKey) return;
        aiWorkerRef.current.loadModel(modelInfo.url, modelInfo.version);
        logger.info('[useGame] Loading client-side AI model:', modelInfo.version);
      })
      .catch((err) => {
        logger.info('[useGame] ONNX model not available, using server AI:', err);
        // Don't reset loadedModelRef - prevents infinite retry loop on devices
        // where ONNX isn't supported. Server-side AI will be used as fallback.
      });
  }, [vsAI, aiModel]);

  // Start a new game
  const startNewGame = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    setSelectedSquare(null);
    setLastMove(null);
    setMoveHistory([]);
    setRawMoves([]);
    setEvaluation(null);
    setViewPly(null);
    passChainStartRef.current = 0;

    try {
      const { game_id } = await api.createGame({
        player1_type: 'human',
        player2_type: vsAI ? 'ai' : 'human',
        ai_simulations: aiSimulations,
      });
      const state = await api.getGameState(game_id);
      setGameState(state);

      // If human plays as red (player 1), AI needs to move first
      if (vsAI && aiPlayer === 0 && state.status === 'playing' && state.current_player === 0) {
        // AI is player 0, needs to move first
        setIsLoading(false);
        // Trigger AI move after state is set - handled by the effect below
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start game');
    } finally {
      setIsLoading(false);
    }
  }, [vsAI, aiSimulations, aiPlayer]);

  // Handle AI move - loop until AI's turn is complete
  // Uses client-side AI if loaded, falls back to server AI
  const handleAIMove = useCallback(async (gameId: string) => {
    setAiThinking(true);
    try {
      let aiMoveCount = 0;
      const MAX_AI_MOVES = 10;

      let currentState = await api.getGameState(gameId);

      while (currentState.status === 'playing' && currentState.current_player === aiPlayer) {
        aiMoveCount++;
        if (aiMoveCount > MAX_AI_MOVES) {
          logger.error('[useGame] AI move loop exceeded max iterations');
          setError('AI took too many moves in one turn');
          break;
        }

        // Use client-side AI if ready, otherwise fall back to server
        const clientReady = aiWorker.isLoaded;
        if (!clientReady && !aiWorker.isLoading) {
          logger.warn('[useGame] Client-side AI not available', {
            loadError: aiWorker.loadError,
            backend: aiWorker.backend,
          });
        }

        logger.info('[useGame] AI making move...', { moveNumber: aiMoveCount, clientSide: clientReady });

        const prevPlayer = currentState.current_player;
        let aiMove: number;
        let aiValue: number | null = null;

        // Try client-side AI first
        if (clientReady) {
          try {
            const engineState = apiStateToEngineState(currentState);
            const result = await aiWorker.search(engineState, {
              numSimulations: aiSimulations,
            });

            aiMove = result.bestMove;
            aiValue = result.value;
            setAiProgress(null);

            // Sync move to server
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

        setGameState(currentState);

        if (aiValue !== undefined && aiValue !== null) {
          setEvaluation(-aiValue);
        }

        const wasPass = currentState.current_player === prevPlayer;
        recordMove(aiMove, aiPlayer as Player, wasPass);

        logger.info('[useGame] AI move complete', {
          move: aiMove,
          wasPass,
          newPlayer: currentState.current_player,
          evaluation: aiValue,
        });
      }

      logger.info('[useGame] AI turn complete, current_player:', currentState.current_player);
    } catch (err) {
      logger.error('[useGame] AI move failed:', err);
      setError(err instanceof Error ? err.message : 'AI move failed');
    } finally {
      setAiThinking(false);
      setAiProgress(null);
    }
  }, [aiSimulations, aiModel, aiPlayer, recordMove, aiWorker]);

  // Trigger AI move when it's AI's turn (e.g., after new game where AI goes first)
  useEffect(() => {
    if (!vsAI || !gameState || gameState.status !== 'playing') return;
    if (aiThinking || isLoading) return;
    if (gameState.current_player !== aiPlayer) return;
    // Only trigger if no moves have been made yet and AI goes first,
    // or after we've returned from a move
    if (gameState.ply === 0 && aiPlayer === 0) {
      handleAIMove(gameState.game_id);
    }
  }, [vsAI, gameState?.game_id, gameState?.current_player, gameState?.ply, aiPlayer, aiThinking, isLoading]);

  // End turn explicitly (after passing) - "Complete Pass"
  const endTurn = useCallback(async () => {
    logger.info('[useGame] endTurn called:', { gameId: gameState?.game_id, canEndTurn });
    if (!gameState || !canEndTurn) return;

    // Return to live view if viewing history
    setViewPly(null);
    setIsLoading(true);
    setError(null);

    try {
      logger.info('[useGame] Sending END_TURN_MOVE:', END_TURN_MOVE);
      const newState = await api.makeMove(gameState.game_id, END_TURN_MOVE);
      setGameState(newState);
      setSelectedSquare(null);
      setRawMoves(prev => [...prev, END_TURN_MOVE]);
      passChainStartRef.current = 0;
      playEndTurnSound();

      // If vs AI and it's now AI's turn
      if (vsAI && newState.status === 'playing' && newState.current_player === aiPlayer) {
        await handleAIMove(newState.game_id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'End turn failed');
    } finally {
      setIsLoading(false);
    }
  }, [gameState, canEndTurn, vsAI, aiPlayer, handleAIMove]);

  // Cancel the current pass chain
  const cancelPass = useCallback(async () => {
    if (!gameState || !isPassing) return;

    setIsLoading(true);
    setError(null);

    try {
      // Count how many pass moves we need to undo
      // Walk backwards from the end of rawMoves until we hit a non-pass move or the start
      let undoCount = 0;
      for (let i = rawMoves.length - 1; i >= 0; i--) {
        const move = rawMoves[i];
        if (move === END_TURN_MOVE) break; // Shouldn't happen during a pass, but safety
        undoCount++;
        // Check if the move before this was also a pass (same player)
        // We keep undoing until we get back to the knight move state
      }

      // Undo each pass move
      let state = gameState;
      for (let i = 0; i < undoCount; i++) {
        state = await api.undoMove(state.game_id);
      }

      setGameState(state);
      setSelectedSquare(null);
      setMoveHistory(prev => prev.slice(0, -undoCount));
      setRawMoves(prev => prev.slice(0, -undoCount));
      passChainStartRef.current = 0;

      // Restore last move highlight
      const newRawMoves = rawMoves.slice(0, -undoCount);
      const lastNonEndTurn = [...newRawMoves].reverse().find(m => m !== END_TURN_MOVE);
      if (lastNonEndTurn !== undefined) {
        const { src, dst } = decodeMove(lastNonEndTurn);
        setLastMove({ from: src, to: dst });
      } else {
        setLastMove(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Cancel pass failed');
    } finally {
      setIsLoading(false);
    }
  }, [gameState, isPassing, rawMoves]);

  // Handle square click
  const handleSquareClick = useCallback((square: number) => {
    logger.info('[useGame] handleSquareClick:', { square, selectedSquare, gameId: gameState?.game_id, moveInProgress: moveInProgress.current });
    if (!gameState || gameState.status !== 'playing' || aiThinking || isLoading) return;

    // Return to live view if viewing history
    if (isViewingHistory) {
      setViewPly(null);
      return;
    }

    // Use ref for synchronous check to avoid closure staleness race condition
    if (moveInProgress.current) {
      logger.info('[useGame] Ignoring click - move in progress');
      return;
    }

    // If AI's turn in vs AI mode, ignore clicks
    if (vsAI && gameState.current_player === aiPlayer) return;

    const { board, legal_moves, current_player } = gameState;

    // Check if clicking on own piece or ball
    const isOwnPiece = (player: Player, sq: number) => {
      const pieces = BigInt(player === 0 ? board.p1_pieces : board.p2_pieces);
      const ball = BigInt(player === 0 ? board.p1_ball : board.p2_ball);
      const mask = BigInt(1) << BigInt(sq);
      return ((pieces | ball) & mask) !== BigInt(0);
    };

    // During a pass chain, lock interaction to valid receivers only
    if (isPassing && selectedSquare !== null) {
      const moveEncoded = encodeMove(selectedSquare, square);
      if (legal_moves.includes(moveEncoded)) {
        // Valid pass destination - execute it
        moveInProgress.current = true;
        setIsLoading(true);
        setError(null);

        api.makeMove(gameState.game_id, moveEncoded)
          .then(async (newState) => {
            setGameState(newState);
            const wasPass = newState.current_player === current_player;
            recordMove(moveEncoded, current_player, wasPass);

            if (wasPass) {
              setSelectedSquare(square);
            } else {
              setSelectedSquare(null);
              passChainStartRef.current = 0;
              if (vsAI && newState.status === 'playing' && newState.current_player === aiPlayer) {
                await handleAIMove(newState.game_id);
              }
            }
          })
          .catch((err) => {
            logger.error('[useGame] Pass move failed:', err);
            setError(err instanceof Error ? err.message : 'Move failed');
          })
          .finally(() => {
            moveInProgress.current = false;
            setIsLoading(false);
          });
      }
      // During pass, ignore all other clicks (no deselect, no selecting other pieces)
      return;
    }

    // If a piece is selected, check if this is a valid move destination
    if (selectedSquare !== null) {
      const moveEncoded = encodeMove(selectedSquare, square);
      if (legal_moves.includes(moveEncoded)) {
        // Valid move - execute it
        moveInProgress.current = true;
        setIsLoading(true);
        setError(null);

        logger.info('[useGame] Making move:', {
          gameId: gameState.game_id,
          moveEncoded,
          from: selectedSquare,
          to: square,
          legalMoves: legal_moves
        });
        api.makeMove(gameState.game_id, moveEncoded)
          .then(async (newState) => {
            setGameState(newState);

            // Check if this was a pass (ball moved but turn didn't change)
            const wasPass = newState.current_player === current_player;
            recordMove(moveEncoded, current_player, wasPass);

            if (wasPass) {
              // Keep selection on the new ball position for chaining passes
              setSelectedSquare(square);
              if (passChainStartRef.current === 0) {
                passChainStartRef.current = rawMoves.length;
              }
            } else {
              setSelectedSquare(null);
              passChainStartRef.current = 0;
              // If vs AI and it's now AI's turn
              if (vsAI && newState.status === 'playing' && newState.current_player === aiPlayer) {
                await handleAIMove(newState.game_id);
              }
            }
          })
          .catch((err) => {
            logger.error('[useGame] Move failed:', err);
            setError(err instanceof Error ? err.message : 'Move failed');
          })
          .finally(() => {
            moveInProgress.current = false;
            setIsLoading(false);
          });
        return;
      }
    }

    // Check if clicking on own piece to select/deselect
    if (isOwnPiece(current_player, square)) {
      const newSelection = selectedSquare === square ? null : square;
      setSelectedSquare(newSelection);
      if (newSelection !== null) {
        playSelectSound();
      }
      return;
    }

    // Clicking elsewhere - deselect
    if (selectedSquare !== null) {
      setSelectedSquare(null);
    }
  }, [gameState, selectedSquare, vsAI, aiPlayer, aiThinking, isLoading, isPassing, isViewingHistory, handleAIMove, rawMoves, recordMove]);

  // Undo last move
  const undoMove = useCallback(async () => {
    if (!gameState) return;

    // Return to live view if viewing history
    setViewPly(null);

    setIsLoading(true);
    setError(null);
    setSelectedSquare(null);

    try {
      let state = await api.undoMove(gameState.game_id);
      setMoveHistory(prev => prev.slice(0, -1)); // Remove last move
      setRawMoves(prev => prev.slice(0, -1)); // Remove from raw moves too
      if (vsAI && state.current_player === aiPlayer) {
        state = await api.undoMove(gameState.game_id);
        setMoveHistory(prev => prev.slice(0, -1)); // Remove AI's move too
        setRawMoves(prev => prev.slice(0, -1));
      }
      setGameState(state);
      // Update lastMove to previous move or null
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
  }, [gameState, vsAI, aiPlayer, humanPlayer, moveHistory]);

  // Handle drag-and-drop move
  const handleDragMove = useCallback((from: number, to: number) => {
    if (!gameState || gameState.status !== 'playing' || aiThinking || isLoading) return;
    if (isViewingHistory) {
      setViewPly(null);
      return;
    }
    if (vsAI && gameState.current_player === aiPlayer) return;

    const moveEncoded = encodeMove(from, to);
    if (!gameState.legal_moves.includes(moveEncoded)) return;

    const { current_player } = gameState;

    moveInProgress.current = true;
    setIsLoading(true);
    setError(null);
    setSelectedSquare(null);

    api.makeMove(gameState.game_id, moveEncoded)
      .then(async (newState) => {
        setGameState(newState);
        const wasPass = newState.current_player === current_player;
        recordMove(moveEncoded, current_player, wasPass);

        if (wasPass) {
          setSelectedSquare(to);
        } else if (vsAI && newState.status === 'playing' && newState.current_player === aiPlayer) {
          await handleAIMove(newState.game_id);
        }
      })
      .catch((err) => {
        logger.error('[useGame] Drag move failed:', err);
        setError(err instanceof Error ? err.message : 'Move failed');
      })
      .finally(() => {
        moveInProgress.current = false;
        setIsLoading(false);
      });
  }, [gameState, vsAI, aiPlayer, aiThinking, isLoading, isViewingHistory, handleAIMove, recordMove]);

  // Resign from the game
  const resign = useCallback(async () => {
    if (!gameState || gameState.status !== 'playing') return;

    setIsLoading(true);
    setError(null);
    setViewPly(null);

    try {
      const newState = await api.resignGame(gameState.game_id, humanPlayer === -1 ? 0 : humanPlayer);
      setGameState(newState);
      setSelectedSquare(null);
      playLoseSound();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Resign failed');
    } finally {
      setIsLoading(false);
    }
  }, [gameState, humanPlayer]);

  // History navigation (view-only)
  const goToMove = useCallback((ply: number) => {
    const clamped = Math.max(0, Math.min(totalPlies, ply));
    if (clamped === totalPlies) {
      setViewPly(null); // Back to live
    } else {
      setViewPly(clamped);
    }
  }, [totalPlies]);

  const goForward = useCallback(() => {
    if (viewPly === null) return; // Already at live
    goToMove(viewPly + 1);
  }, [viewPly, goToMove]);

  const goBack = useCallback(() => {
    const current = viewPly ?? totalPlies;
    goToMove(current - 1);
  }, [viewPly, totalPlies, goToMove]);

  const goToStart = useCallback(() => {
    goToMove(0);
  }, [goToMove]);

  const goToEnd = useCallback(() => {
    setViewPly(null);
  }, []);

  // Compute what to display when viewing history
  const displayState = useMemo(() => {
    if (viewPly === null || !gameState) return null;
    return replayToPosition(rawMoves, viewPly);
  }, [viewPly, rawMoves, gameState]);

  const displayLastMove = useMemo(() => {
    if (viewPly === null) return lastMove;
    return getLastMoveAtPosition(rawMoves, viewPly);
  }, [viewPly, rawMoves, lastMove]);

  // Build the effective game state for rendering
  const effectiveGameState = useMemo(() => {
    if (!gameState) return null;
    if (!displayState) return gameState;
    // Override visual state with history position
    return {
      ...gameState,
      board: displayState.board,
      current_player: displayState.currentPlayer as Player,
      touched_mask: displayState.touchedMask,
      has_passed: displayState.hasPassed,
      ply: displayState.ply,
      legal_moves: [], // No moves while viewing history
    };
  }, [gameState, displayState]);

  return {
    gameState: effectiveGameState,
    selectedSquare: isViewingHistory ? null : selectedSquare,
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
