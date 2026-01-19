import { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import type { GameState, Player } from '../types';
import { encodeMove, decodeMove, TOTAL_SQUARES, squareToAlgebraic } from '../types';
import * as api from '../api/engine';
import { logger } from '../utils/logger';
import { playMoveSound, playPassSound, playEndTurnSound, playWinSound, playLoseSound, playSelectSound } from '../utils/sounds';

interface UseGameOptions {
  vsAI?: boolean;
  aiSimulations?: number;
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
  canEndTurn: boolean;
  mustPass: boolean;
  lastMove: LastMove | null;
  moveHistory: MoveRecord[];
  startNewGame: () => Promise<void>;
  handleSquareClick: (square: number) => void;
  handleDragMove: (from: number, to: number) => void;
  endTurn: () => Promise<void>;
  undoMove: () => Promise<void>;
}

const END_TURN_MOVE = -1;

export function useGame(options: UseGameOptions = {}): UseGameReturn {
  const { vsAI = true, aiSimulations = 800 } = options;

  const [gameState, setGameState] = useState<GameState | null>(null);
  const [selectedSquare, setSelectedSquare] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [aiThinking, setAiThinking] = useState(false);
  const [lastMove, setLastMove] = useState<LastMove | null>(null);
  const [moveHistory, setMoveHistory] = useState<MoveRecord[]>([]);

  // Ref to track move-in-progress synchronously (avoids closure staleness)
  const moveInProgress = useRef(false);

  // Check if end turn is available
  const canEndTurn = useMemo(() => {
    if (!gameState) return false;
    return gameState.legal_moves.includes(END_TURN_MOVE);
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

  // Play sound when game ends
  const prevStatusRef = useRef<string | null>(null);
  useEffect(() => {
    if (!gameState) return;

    const wasPlaying = prevStatusRef.current === 'playing';
    prevStatusRef.current = gameState.status;

    if (wasPlaying && gameState.status === 'finished' && gameState.winner !== null) {
      // In AI mode, player is 0 (blue)
      if (vsAI) {
        if (gameState.winner === 0) {
          playWinSound();
        } else {
          playLoseSound();
        }
      } else {
        // In PvP, just play a neutral end sound (win sound for winner)
        playWinSound();
      }
    }
  }, [gameState, vsAI]);

  // Start a new game
  const startNewGame = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    setSelectedSquare(null);
    setLastMove(null);
    setMoveHistory([]);

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
  }, [vsAI, aiSimulations]);

  // Helper to record a move and play sound
  const recordMove = useCallback((move: number, player: Player, isPass: boolean) => {
    if (move === END_TURN_MOVE) return; // Don't record end turn as a move
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

  // Handle AI move - loop until AI's turn is complete
  const handleAIMove = useCallback(async (gameId: string) => {
    setAiThinking(true);
    try {
      // Use the state from the response instead of fetching again
      let aiMoveCount = 0;
      const MAX_AI_MOVES = 10; // Safety limit to prevent infinite loops

      let currentState = await api.getGameState(gameId);

      // Keep making moves while it's still AI's turn (player 1) and game is playing
      while (currentState.status === 'playing' && currentState.current_player === 1) {
        aiMoveCount++;
        if (aiMoveCount > MAX_AI_MOVES) {
          logger.error('[useGame] AI move loop exceeded max iterations');
          setError('AI took too many moves in one turn');
          break;
        }

        logger.info('[useGame] AI making move...', { moveNumber: aiMoveCount });
        const prevPlayer = currentState.current_player;
        const aiResponse = await api.getAIMove(gameId, { simulations: aiSimulations });

        // Use the state from the AI response instead of fetching again
        currentState = aiResponse.game_state;
        setGameState(currentState); // Update UI after each AI move

        // A pass doesn't change the player, a knight move does
        const wasPass = currentState.current_player === prevPlayer;
        recordMove(aiResponse.move, 1, wasPass);

        logger.info('[useGame] AI move complete', {
          move: aiResponse.algebraic,
          wasPass,
          newPlayer: currentState.current_player
        });
      }

      logger.info('[useGame] AI turn complete, current_player:', currentState.current_player);
    } catch (err) {
      logger.error('[useGame] AI move failed:', err);
      setError(err instanceof Error ? err.message : 'AI move failed');
    } finally {
      setAiThinking(false);
    }
  }, [aiSimulations, recordMove]);

  // End turn explicitly (after passing)
  const endTurn = useCallback(async () => {
    logger.info('[useGame] endTurn called:', { gameId: gameState?.game_id, canEndTurn });
    if (!gameState || !canEndTurn) return;

    setIsLoading(true);
    setError(null);

    try {
      logger.info('[useGame] Sending END_TURN_MOVE:', END_TURN_MOVE);
      const newState = await api.makeMove(gameState.game_id, END_TURN_MOVE);
      setGameState(newState);
      setSelectedSquare(null);
      playEndTurnSound();

      // If vs AI and it's now AI's turn
      if (vsAI && newState.status === 'playing' && newState.current_player === 1) {
        await handleAIMove(newState.game_id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'End turn failed');
    } finally {
      setIsLoading(false);
    }
  }, [gameState, canEndTurn, vsAI, handleAIMove]);

  // Handle square click
  const handleSquareClick = useCallback((square: number) => {
    logger.info('[useGame] handleSquareClick:', { square, selectedSquare, gameId: gameState?.game_id, moveInProgress: moveInProgress.current });
    if (!gameState || gameState.status !== 'playing' || aiThinking || isLoading) return;

    // Use ref for synchronous check to avoid closure staleness race condition
    if (moveInProgress.current) {
      logger.info('[useGame] Ignoring click - move in progress');
      return;
    }

    // If AI's turn in vs AI mode, ignore clicks
    if (vsAI && gameState.current_player === 1) return;

    const { board, legal_moves, current_player } = gameState;

    // Check if clicking on own piece or ball
    const isOwnPiece = (player: Player, sq: number) => {
      const pieces = BigInt(player === 0 ? board.p1_pieces : board.p2_pieces);
      const ball = BigInt(player === 0 ? board.p1_ball : board.p2_ball);
      const mask = BigInt(1) << BigInt(sq);
      return ((pieces | ball) & mask) !== BigInt(0);
    };

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
            } else {
              setSelectedSquare(null);
              // If vs AI and it's now AI's turn
              if (vsAI && newState.status === 'playing' && newState.current_player === 1) {
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
  }, [gameState, selectedSquare, vsAI, aiThinking, isLoading, handleAIMove]);

  // Undo last move
  const undoMove = useCallback(async () => {
    if (!gameState) return;

    setIsLoading(true);
    setError(null);
    setSelectedSquare(null);

    try {
      let state = await api.undoMove(gameState.game_id);
      setMoveHistory(prev => prev.slice(0, -1)); // Remove last move
      if (vsAI && state.current_player === 1) {
        state = await api.undoMove(gameState.game_id);
        setMoveHistory(prev => prev.slice(0, -1)); // Remove AI's move too
      }
      setGameState(state);
      // Update lastMove to previous move or null
      const newHistory = moveHistory.slice(0, vsAI && state.current_player === 0 ? -2 : -1);
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
  }, [gameState, vsAI, moveHistory]);

  // Handle drag-and-drop move
  const handleDragMove = useCallback((from: number, to: number) => {
    if (!gameState || gameState.status !== 'playing' || aiThinking || isLoading) return;
    if (vsAI && gameState.current_player === 1) return;

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
        } else if (vsAI && newState.status === 'playing' && newState.current_player === 1) {
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
  }, [gameState, vsAI, aiThinking, isLoading, handleAIMove, recordMove]);

  return {
    gameState,
    selectedSquare,
    isLoading,
    error,
    aiThinking,
    canEndTurn,
    mustPass,
    lastMove,
    moveHistory,
    startNewGame,
    handleSquareClick,
    handleDragMove,
    endTurn,
    undoMove,
  };
}
