/**
 * Online Game Overlay component.
 * Shows opponent info, connection status, and turn indicator during online games.
 */

import { useState, useEffect } from 'react';
import type { OnlineOpponentInfo } from '../api/online';
import type { ConnectionStatus } from '../hooks/useOnlineGame';

interface OnlineGameOverlayProps {
  myColor: 0 | 1;
  isMyTurn: boolean;
  opponent: OnlineOpponentInfo | null;
  opponentConnected: boolean;
  connectionStatus: ConnectionStatus;
  disconnectWarning: { gracePeriod: number; startTime: number } | null;
  onLeaveGame: () => void;
  onReconnect: () => void;
}

export default function OnlineGameOverlay({
  myColor,
  isMyTurn,
  opponent,
  opponentConnected,
  connectionStatus,
  disconnectWarning,
  onLeaveGame,
  onReconnect,
}: OnlineGameOverlayProps) {
  const [showLeaveConfirm, setShowLeaveConfirm] = useState(false);
  const [timeRemaining, setTimeRemaining] = useState<number | null>(null);

  // Update countdown timer for disconnect warning
  useEffect(() => {
    if (!disconnectWarning) {
      setTimeRemaining(null);
      return;
    }

    const updateTimer = () => {
      const elapsed = (Date.now() - disconnectWarning.startTime) / 1000;
      const remaining = Math.max(0, disconnectWarning.gracePeriod - elapsed);
      setTimeRemaining(Math.ceil(remaining));
    };

    updateTimer();
    const interval = setInterval(updateTimer, 1000);
    return () => clearInterval(interval);
  }, [disconnectWarning]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <>
      {/* Top bar with game info */}
      <div className="bg-gray-800 rounded-lg px-4 py-2 mb-4 flex items-center justify-between">
        {/* Opponent info */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span
              className={`w-3 h-3 rounded-full ${
                myColor === 0 ? 'bg-red-500' : 'bg-blue-500'
              }`}
            ></span>
            <span className="text-white font-medium">
              {opponent?.display_name || 'Opponent'}
            </span>
            {opponent?.elo_rating && (
              <span className="text-gray-500 text-sm">
                ({Math.round(opponent.elo_rating)})
              </span>
            )}
          </div>

          {/* Connection status */}
          <div className="flex items-center gap-1">
            <span
              className={`w-2 h-2 rounded-full ${
                opponentConnected ? 'bg-green-500' : 'bg-red-500 animate-pulse'
              }`}
            ></span>
            <span className="text-xs text-gray-500">
              {opponentConnected ? 'Online' : 'Disconnected'}
            </span>
          </div>
        </div>

        {/* Your info and turn indicator */}
        <div className="flex items-center gap-3">
          <div
            className={`px-3 py-1 rounded text-sm font-medium ${
              isMyTurn
                ? 'bg-green-600 text-white animate-pulse'
                : 'bg-gray-700 text-gray-400'
            }`}
          >
            {isMyTurn ? 'Your Turn' : "Opponent's Turn"}
          </div>

          <div className="flex items-center gap-2">
            <span className="text-gray-400 text-sm">You:</span>
            <span
              className={`w-3 h-3 rounded-full ${
                myColor === 0 ? 'bg-blue-500' : 'bg-red-500'
              }`}
            ></span>
          </div>

          {/* Leave button */}
          <button
            onClick={() => setShowLeaveConfirm(true)}
            className="px-2 py-1 text-sm text-gray-400 hover:text-red-400 transition-colors"
            title="Leave game"
          >
            Leave
          </button>
        </div>
      </div>

      {/* Connection status overlay */}
      {connectionStatus !== 'connected' && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-40">
          <div className="bg-gray-800 rounded-lg p-6 text-center max-w-sm">
            {connectionStatus === 'connecting' && (
              <>
                <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                <p className="text-white">Connecting...</p>
              </>
            )}
            {connectionStatus === 'reconnecting' && (
              <>
                <div className="w-12 h-12 border-4 border-yellow-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                <p className="text-white mb-2">Reconnecting...</p>
                <p className="text-gray-400 text-sm">Please wait</p>
              </>
            )}
            {connectionStatus === 'disconnected' && (
              <>
                <div className="text-red-500 text-5xl mb-4">!</div>
                <p className="text-white mb-2">Connection Lost</p>
                <p className="text-gray-400 text-sm mb-4">
                  Unable to connect to the game server.
                </p>
                <button
                  onClick={onReconnect}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded font-medium"
                >
                  Retry Connection
                </button>
              </>
            )}
          </div>
        </div>
      )}

      {/* Opponent disconnect warning */}
      {disconnectWarning && timeRemaining !== null && (
        <div className="fixed top-20 left-1/2 transform -translate-x-1/2 z-30">
          <div className="bg-yellow-600 text-white px-4 py-2 rounded-lg shadow-lg flex items-center gap-3">
            <span className="animate-pulse">!</span>
            <span>
              Opponent disconnected. Game will forfeit in {formatTime(timeRemaining)}
            </span>
          </div>
        </div>
      )}

      {/* Leave confirmation dialog */}
      {showLeaveConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-lg p-6 max-w-sm w-full mx-4">
            <h3 className="text-xl font-bold text-white mb-2">Leave Game?</h3>
            <p className="text-gray-400 mb-4">
              Leaving the game will count as a forfeit. Your opponent will win.
            </p>
            <div className="flex gap-2">
              <button
                onClick={() => {
                  setShowLeaveConfirm(false);
                  onLeaveGame();
                }}
                className="flex-1 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded font-medium"
              >
                Leave Game
              </button>
              <button
                onClick={() => setShowLeaveConfirm(false)}
                className="flex-1 px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded font-medium"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
