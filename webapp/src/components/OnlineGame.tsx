/**
 * Online Game component.
 * Main view for playing an online multiplayer game.
 */

import { useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import Board from './Board';
import MoveHistory from './MoveHistory';
import OnlineGameOverlay from './OnlineGameOverlay';
import { useOnlineGame } from '../hooks/useOnlineGame';
import type { OnlineOpponentInfo } from '../api/online';

interface OnlineGameProps {
  gameId: string;
  onGameEnd?: (winner: number | null, reason: string) => void;
}

export default function OnlineGame({ gameId, onGameEnd }: OnlineGameProps) {
  const navigate = useNavigate();

  const handleGameEnd = useCallback(
    (winner: number | null, reason: string) => {
      onGameEnd?.(winner, reason);
    },
    [onGameEnd]
  );

  const handleOpponentJoined = useCallback((opponent: OnlineOpponentInfo) => {
    console.log('Opponent joined:', opponent);
  }, []);

  const {
    gameState,
    myColor,
    isMyTurn,
    opponent,
    opponentConnected,
    connectionStatus,
    selectedSquare,
    isLoading,
    error,
    canEndTurn,
    mustPass,
    lastMove,
    rawMoves,
    disconnectWarning,
    handleSquareClick,
    handleDragMove,
    endTurn,
    leaveGame,
    reconnect,
  } = useOnlineGame({
    gameId,
    onGameEnd: handleGameEnd,
    onOpponentJoined: handleOpponentJoined,
  });

  const handleLeaveGame = async () => {
    try {
      await leaveGame();
      navigate('/');
    } catch (err) {
      console.error('Failed to leave game:', err);
      navigate('/');
    }
  };

  // Determine if we need to flip the board
  // Red player (color 1) sees the board flipped
  const shouldFlipBoard = myColor === 1;

  // Get winner text
  const getWinnerText = () => {
    if (!gameState || gameState.winner === null) return '';
    if (gameState.winner === myColor) {
      return 'You Win!';
    }
    return 'You Lose';
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-2 sm:p-4">
      <h1 className="text-2xl sm:text-3xl font-bold mb-3 sm:mb-4">Razzle Dazzle</h1>

      {/* Online Game Header */}
      {myColor !== null && (
        <OnlineGameOverlay
          myColor={myColor}
          isMyTurn={isMyTurn}
          opponent={opponent}
          opponentConnected={opponentConnected}
          connectionStatus={connectionStatus}
          disconnectWarning={disconnectWarning}
          onLeaveGame={handleLeaveGame}
          onReconnect={reconnect}
        />
      )}

      {error && (
        <div className="mb-4 px-4 py-2 bg-red-600 text-white rounded">{error}</div>
      )}

      {!gameState && connectionStatus === 'connected' && (
        <div className="text-gray-400">Loading game...</div>
      )}

      {gameState && (
        <>
          <div className="flex flex-col sm:flex-row gap-4 items-center sm:items-start w-full max-w-md sm:max-w-none sm:w-auto">
            <Board
              board={gameState.board}
              currentPlayer={gameState.current_player}
              legalMoves={isMyTurn ? gameState.legal_moves : []}
              selectedSquare={selectedSquare}
              onSquareClick={handleSquareClick}
              onDragMove={handleDragMove}
              touchedMask={gameState.touched_mask}
              mustPass={mustPass}
              flipped={shouldFlipBoard}
              lastMove={lastMove}
            />
            <div className="hidden sm:block">
              <MoveHistory moves={rawMoves} />
            </div>
          </div>

          {/* Game status */}
          <div className="mt-4 text-center h-8 flex items-center justify-center">
            {gameState.status === 'finished' && gameState.winner !== null && (
              <div
                className={`text-2xl font-bold ${
                  gameState.winner === myColor ? 'text-green-400' : 'text-red-400'
                }`}
              >
                {getWinnerText()}
              </div>
            )}
            {gameState.status === 'draw' && (
              <div className="text-2xl font-bold text-gray-400">Draw!</div>
            )}
            {mustPass && gameState.status === 'playing' && (
              <div className="text-yellow-400 animate-pulse">
                Forced to pass! Opponent moved adjacent to your ball.
              </div>
            )}
          </div>

          {/* Controls */}
          <div className="mt-4 flex flex-wrap justify-center gap-2 sm:gap-4">
            {canEndTurn && (
              <button
                onClick={endTurn}
                disabled={isLoading}
                className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded font-medium transition-colors animate-pulse"
              >
                End Turn
              </button>
            )}
            <button
              onClick={() => navigate('/')}
              className="px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded font-medium transition-colors"
            >
              Back to Menu
            </button>
          </div>

          {/* Game info */}
          <div className="mt-4 text-sm text-gray-400 text-center">
            <span>Ply: {gameState.ply}</span>
            <span className="ml-4">Online Game</span>
          </div>
        </>
      )}

      {/* Instructions */}
      <div className="mt-8 text-sm text-gray-500 max-w-md text-center">
        <p>Click a piece to select it, then click a highlighted square to move or pass.</p>
        <p className="mt-1">After passing, click "End Turn" to finish your turn.</p>
        <p className="mt-1">
          {myColor === 0 ? 'You are Blue - aim for the top!' : 'You are Red - aim for the bottom!'}
        </p>
      </div>
    </div>
  );
}
