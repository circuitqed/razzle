import { useState, useCallback } from 'react';
import type { GameState, Player } from '../types';
import { encodeMove } from '../types';
import * as api from '../api/engine';

interface UseGameOptions {
  vsAI?: boolean;
  aiSimulations?: number;
}

interface UseGameReturn {
  gameState: GameState | null;
  selectedSquare: number | null;
  isLoading: boolean;
  error: string | null;
  aiThinking: boolean;
  startNewGame: () => Promise<void>;
  handleSquareClick: (square: number) => void;
  undoMove: () => Promise<void>;
}

export function useGame(options: UseGameOptions = {}): UseGameReturn {
  const { vsAI = true, aiSimulations = 800 } = options;

  const [gameState, setGameState] = useState<GameState | null>(null);
  const [selectedSquare, setSelectedSquare] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [aiThinking, setAiThinking] = useState(false);

  // Start a new game
  const startNewGame = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    setSelectedSquare(null);

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

  // Handle AI move
  const handleAIMove = useCallback(async (gameId: string) => {
    setAiThinking(true);
    try {
      const aiResponse = await api.getAIMove(gameId, { simulations: aiSimulations });
      const newState = await api.makeMove(gameId, aiResponse.move);
      setGameState(newState);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'AI move failed');
    } finally {
      setAiThinking(false);
    }
  }, [aiSimulations]);

  // Handle square click
  const handleSquareClick = useCallback((square: number) => {
    if (!gameState || gameState.status !== 'playing' || aiThinking) return;

    // If AI's turn in vs AI mode, ignore clicks
    if (vsAI && gameState.current_player === 1) return;

    const { board, legal_moves } = gameState;

    // Check if clicking on own piece
    const isOwnPiece = (player: Player, sq: number) => {
      const bb = BigInt(player === 0 ? board.p1_pieces : board.p2_pieces);
      return (bb & (BigInt(1) << BigInt(sq))) !== BigInt(0);
    };

    if (isOwnPiece(gameState.current_player, square)) {
      // Select this piece (or deselect if already selected)
      setSelectedSquare(selectedSquare === square ? null : square);
      return;
    }

    // If a piece is selected, try to move to this square
    if (selectedSquare !== null) {
      const moveEncoded = encodeMove(selectedSquare, square);
      if (legal_moves.includes(moveEncoded)) {
        // Valid move - execute it
        setIsLoading(true);
        setError(null);

        api.makeMove(gameState.game_id, moveEncoded)
          .then(async (newState) => {
            setGameState(newState);
            setSelectedSquare(null);

            // If vs AI and game is still playing and it's AI's turn
            if (vsAI && newState.status === 'playing' && newState.current_player === 1) {
              await handleAIMove(newState.game_id);
            }
          })
          .catch((err) => {
            setError(err instanceof Error ? err.message : 'Move failed');
          })
          .finally(() => {
            setIsLoading(false);
          });
      } else {
        // Invalid move - deselect
        setSelectedSquare(null);
      }
    }
  }, [gameState, selectedSquare, vsAI, aiThinking, handleAIMove]);

  // Undo last move
  const undoMove = useCallback(async () => {
    if (!gameState) return;

    setIsLoading(true);
    setError(null);
    setSelectedSquare(null);

    try {
      // In vs AI mode, undo twice to get back to human's turn
      let state = await api.undoMove(gameState.game_id);
      if (vsAI && state.current_player === 1) {
        state = await api.undoMove(gameState.game_id);
      }
      setGameState(state);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Undo failed');
    } finally {
      setIsLoading(false);
    }
  }, [gameState, vsAI]);

  return {
    gameState,
    selectedSquare,
    isLoading,
    error,
    aiThinking,
    startNewGame,
    handleSquareClick,
    undoMove,
  };
}
