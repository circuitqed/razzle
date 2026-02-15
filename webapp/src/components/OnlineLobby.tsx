/**
 * Online Lobby component for creating and joining online games.
 */

import { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import * as onlineApi from '../api/online';

interface OnlineLobbyProps {
  isOpen: boolean;
  onClose: () => void;
  onGameCreated: (gameId: string, joinCode: string, hostColor: number) => void;
  onGameJoined: (gameId: string, yourColor: number) => void;
  onOpenLogin: () => void;
}

export default function OnlineLobby({
  isOpen,
  onClose,
  onGameCreated,
  onGameJoined,
  onOpenLogin,
}: OnlineLobbyProps) {
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState<'create' | 'join'>('create');
  const [hostColor, setHostColor] = useState<0 | 1>(0); // 0 = blue, 1 = red
  const [joinCode, setJoinCode] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [myGames, setMyGames] = useState<onlineApi.MyOnlineGamesResponse | null>(null);

  // Reset form when modal opens
  useEffect(() => {
    if (isOpen) {
      setError(null);
      setJoinCode('');
      if (user) {
        loadMyGames();
      }
    }
  }, [isOpen, user]);

  const loadMyGames = async () => {
    try {
      const games = await onlineApi.getMyOnlineGames();
      setMyGames(games);
    } catch (err) {
      console.error('Failed to load games:', err);
    }
  };

  if (!isOpen) return null;

  const handleCreateGame = async () => {
    if (!user) {
      onOpenLogin();
      return;
    }

    setError(null);
    setIsLoading(true);

    try {
      const result = await onlineApi.createOnlineGame(hostColor);
      onGameCreated(result.game_id, result.join_code, result.host_color);
    } catch (err) {
      setError(
        err instanceof onlineApi.OnlineAPIError ? err.message : 'Failed to create game'
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleJoinGame = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!user) {
      onOpenLogin();
      return;
    }

    if (!joinCode.trim()) {
      setError('Please enter a game code');
      return;
    }

    setError(null);
    setIsLoading(true);

    try {
      const result = await onlineApi.joinOnlineGame(joinCode.trim());
      onGameJoined(result.game_id, result.your_color);
    } catch (err) {
      if (err instanceof onlineApi.OnlineAPIError) {
        if (err.status === 404) {
          setError('Game not found. Check the code and try again.');
        } else if (err.status === 409) {
          setError('Game is no longer available.');
        } else {
          setError(err.message);
        }
      } else {
        setError('Failed to join game');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleResumeGame = (gameId: string, yourColor: number) => {
    onGameJoined(gameId, yourColor);
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-6 max-w-md w-full mx-4 max-h-[90vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold text-white">Play Online</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white text-2xl leading-none"
          >
            &times;
          </button>
        </div>

        {!user && (
          <div className="mb-4 p-3 bg-yellow-600 bg-opacity-20 border border-yellow-600 rounded">
            <p className="text-yellow-400 text-sm">
              You need to{' '}
              <button onClick={onOpenLogin} className="underline hover:text-yellow-300">
                log in
              </button>{' '}
              to play online.
            </p>
          </div>
        )}

        {error && (
          <div className="mb-4 bg-red-600 text-white px-3 py-2 rounded text-sm">
            {error}
          </div>
        )}

        {/* Tabs */}
        <div className="flex gap-2 mb-4">
          <button
            onClick={() => setActiveTab('create')}
            className={`flex-1 px-4 py-2 rounded font-medium transition-colors ${
              activeTab === 'create'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Create Game
          </button>
          <button
            onClick={() => setActiveTab('join')}
            className={`flex-1 px-4 py-2 rounded font-medium transition-colors ${
              activeTab === 'join'
                ? 'bg-green-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Join Game
          </button>
        </div>

        {/* Create Game Tab */}
        {activeTab === 'create' && (
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-300 mb-2">Choose your color</label>
              <div className="flex gap-2">
                <button
                  onClick={() => setHostColor(0)}
                  className={`flex-1 px-4 py-3 rounded font-medium transition-colors flex items-center justify-center gap-2 ${
                    hostColor === 0
                      ? 'bg-blue-600 text-white ring-2 ring-blue-400'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  <span className="w-4 h-4 rounded-full bg-blue-400"></span>
                  Blue (First)
                </button>
                <button
                  onClick={() => setHostColor(1)}
                  className={`flex-1 px-4 py-3 rounded font-medium transition-colors flex items-center justify-center gap-2 ${
                    hostColor === 1
                      ? 'bg-red-600 text-white ring-2 ring-red-400'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  <span className="w-4 h-4 rounded-full bg-red-400"></span>
                  Red (Second)
                </button>
              </div>
            </div>

            <button
              onClick={handleCreateGame}
              disabled={isLoading || !user}
              className="w-full px-4 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded font-medium transition-colors"
            >
              {isLoading ? 'Creating...' : 'Create Game'}
            </button>
          </div>
        )}

        {/* Join Game Tab */}
        {activeTab === 'join' && (
          <form onSubmit={handleJoinGame} className="space-y-4">
            <div>
              <label className="block text-sm text-gray-300 mb-2">Enter game code</label>
              <input
                type="text"
                value={joinCode}
                onChange={(e) => setJoinCode(e.target.value.toUpperCase())}
                placeholder="e.g., ABC123"
                maxLength={8}
                className="w-full px-4 py-3 bg-gray-700 text-white text-center text-2xl tracking-widest rounded border border-gray-600 focus:border-green-500 focus:outline-none uppercase"
                autoFocus
              />
            </div>

            <button
              type="submit"
              disabled={isLoading || !user || !joinCode.trim()}
              className="w-full px-4 py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white rounded font-medium transition-colors"
            >
              {isLoading ? 'Joining...' : 'Join Game'}
            </button>
          </form>
        )}

        {/* My Active Games */}
        {user && myGames && (myGames.active.length > 0 || myGames.waiting.length > 0) && (
          <div className="mt-6 pt-4 border-t border-gray-700">
            <h3 className="text-sm font-medium text-gray-400 mb-2">Your Games</h3>

            {myGames.waiting.length > 0 && (
              <div className="mb-3">
                <p className="text-xs text-gray-500 mb-1">Waiting for opponent</p>
                {myGames.waiting.map((game) => (
                  <div
                    key={game.game_id}
                    className="flex items-center justify-between bg-gray-700 rounded p-2 mb-1"
                  >
                    <div className="flex items-center gap-2">
                      <span
                        className={`w-3 h-3 rounded-full ${
                          game.your_color === 0 ? 'bg-blue-400' : 'bg-red-400'
                        }`}
                      ></span>
                      <span className="text-sm text-gray-300 font-mono">
                        {game.join_code}
                      </span>
                    </div>
                    <button
                      onClick={() => handleResumeGame(game.game_id, game.your_color)}
                      className="px-2 py-1 text-xs bg-yellow-600 hover:bg-yellow-700 text-white rounded"
                    >
                      Open
                    </button>
                  </div>
                ))}
              </div>
            )}

            {myGames.active.length > 0 && (
              <div>
                <p className="text-xs text-gray-500 mb-1">In progress</p>
                {myGames.active.map((game) => (
                  <div
                    key={game.game_id}
                    className="flex items-center justify-between bg-gray-700 rounded p-2 mb-1"
                  >
                    <div className="flex items-center gap-2">
                      <span
                        className={`w-3 h-3 rounded-full ${
                          game.your_color === 0 ? 'bg-blue-400' : 'bg-red-400'
                        }`}
                      ></span>
                      <span className="text-sm text-gray-300">
                        vs {game.opponent_name || 'Opponent'}
                      </span>
                      {game.is_your_turn && (
                        <span className="text-xs bg-green-600 px-1 rounded">Your turn</span>
                      )}
                    </div>
                    <button
                      onClick={() => handleResumeGame(game.game_id, game.your_color)}
                      className="px-2 py-1 text-xs bg-blue-600 hover:bg-blue-700 text-white rounded"
                    >
                      Resume
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
