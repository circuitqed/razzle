/**
 * Online Game Overlay component.
 * Shows connection status modals and opponent disconnect warning during online games.
 */

import type { ConnectionStatus } from '../hooks/useOnlineGame';

interface OnlineGameOverlayProps {
  connectionStatus: ConnectionStatus;
  onReconnect: () => void;
}

export default function OnlineGameOverlay({
  connectionStatus,
  onReconnect,
}: OnlineGameOverlayProps) {
  return (
    <>
      {/* Connection status overlay — only for YOUR connection issues */}
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
    </>
  );
}
