/**
 * Online Game Overlay component.
 * Shows connection status modals and opponent disconnect warning during online games.
 */

import { useState, useEffect } from 'react';
import type { ConnectionStatus } from '../hooks/useOnlineGame';

interface OnlineGameOverlayProps {
  connectionStatus: ConnectionStatus;
  disconnectWarning: { gracePeriod: number; startTime: number } | null;
  onReconnect: () => void;
}

export default function OnlineGameOverlay({
  connectionStatus,
  disconnectWarning,
  onReconnect,
}: OnlineGameOverlayProps) {
  const [disconnectCountdown, setDisconnectCountdown] = useState<number | null>(null);

  // Update countdown timer for disconnect warning
  useEffect(() => {
    if (!disconnectWarning) {
      setDisconnectCountdown(null);
      return;
    }

    const updateTimer = () => {
      const elapsed = (Date.now() - disconnectWarning.startTime) / 1000;
      const remaining = Math.max(0, disconnectWarning.gracePeriod - elapsed);
      setDisconnectCountdown(Math.ceil(remaining));
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
      {disconnectWarning && disconnectCountdown !== null && (
        <div className="fixed top-20 left-1/2 transform -translate-x-1/2 z-30">
          <div className="bg-yellow-600 text-white px-4 py-2 rounded-lg shadow-lg flex items-center gap-3">
            <span className="animate-pulse">!</span>
            <span>
              Opponent disconnected. Game will forfeit in {formatTime(disconnectCountdown)}
            </span>
          </div>
        </div>
      )}

    </>
  );
}
