import { useEffect } from 'react';
import Board from './components/Board';
import { useGame } from './hooks/useGame';

export default function App() {
  const {
    gameState,
    selectedSquare,
    isLoading,
    error,
    aiThinking,
    startNewGame,
    handleSquareClick,
    undoMove,
  } = useGame({ vsAI: true, aiSimulations: 800 });

  // Start a game on mount
  useEffect(() => {
    startNewGame();
  }, []);

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-4">
      <h1 className="text-3xl font-bold mb-6">Razzle Dazzle</h1>

      {error && (
        <div className="mb-4 px-4 py-2 bg-red-600 text-white rounded">
          {error}
        </div>
      )}

      {isLoading && !gameState && (
        <div className="text-gray-400">Loading...</div>
      )}

      {gameState && (
        <>
          <Board
            board={gameState.board}
            currentPlayer={gameState.current_player}
            legalMoves={gameState.legal_moves}
            selectedSquare={selectedSquare}
            onSquareClick={handleSquareClick}
          />

          {/* Game status */}
          <div className="mt-4 text-center">
            {gameState.status === 'won' && (
              <div className="text-2xl font-bold text-yellow-400">
                {gameState.winner === 0 ? 'Blue' : 'Red'} Wins!
              </div>
            )}
            {gameState.status === 'draw' && (
              <div className="text-2xl font-bold text-gray-400">
                Draw!
              </div>
            )}
            {aiThinking && (
              <div className="text-blue-400 animate-pulse">
                AI is thinking...
              </div>
            )}
          </div>

          {/* Controls */}
          <div className="mt-4 flex gap-4">
            <button
              onClick={startNewGame}
              disabled={isLoading}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded font-medium transition-colors"
            >
              New Game
            </button>
            <button
              onClick={undoMove}
              disabled={isLoading || aiThinking || gameState.ply === 0}
              className="px-4 py-2 bg-gray-600 hover:bg-gray-700 disabled:bg-gray-800 disabled:text-gray-500 rounded font-medium transition-colors"
            >
              Undo
            </button>
          </div>

          {/* Game info */}
          <div className="mt-4 text-sm text-gray-400">
            Ply: {gameState.ply}
          </div>
        </>
      )}

      {/* Instructions */}
      <div className="mt-8 text-sm text-gray-500 max-w-md text-center">
        <p>Click a piece to select it, then click a highlighted square to move.</p>
        <p className="mt-1">Blue aims for the top (red zone), Red aims for the bottom (blue zone).</p>
      </div>
    </div>
  );
}
