import { useState, useEffect, useCallback } from 'react';
import * as gamesApi from '../api/games';
import type { GameSummary } from '../api/games';
import { useAuth } from '../contexts/AuthContext';

interface GameBrowserProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectGame: (gameId: string) => void;
}

export default function GameBrowser({ isOpen, onClose, onSelectGame }: GameBrowserProps) {
  const { user } = useAuth();
  const [games, setGames] = useState<GameSummary[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);

  // Filters
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [resultFilter, setResultFilter] = useState<string>('');
  const [myGamesOnly, setMyGamesOnly] = useState(false);

  const fetchGames = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const params: Parameters<typeof gamesApi.listGames>[0] = {
        page,
        per_page: 15,
      };

      if (statusFilter) {
        params.status = statusFilter;
      }
      if (resultFilter) {
        params.winner = parseInt(resultFilter);
      }
      if (myGamesOnly && user) {
        params.player_id = user.user_id;
      }

      const response = await gamesApi.listGames(params);
      setGames(response.games);
      setTotalPages(response.total_pages);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load games');
    } finally {
      setIsLoading(false);
    }
  }, [page, statusFilter, resultFilter, myGamesOnly, user]);

  // Load games when filters change
  useEffect(() => {
    if (isOpen) {
      fetchGames();
    }
  }, [isOpen, fetchGames]);

  // Reset page when filters change
  useEffect(() => {
    setPage(1);
  }, [statusFilter, resultFilter, myGamesOnly]);

  if (!isOpen) return null;

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    // Use Intl.DateTimeFormat for proper localization
    return new Intl.DateTimeFormat(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }).format(date);
  };

  const getResultText = (game: GameSummary) => {
    if (game.status === 'playing') return 'In Progress';
    if (game.winner === null) return 'Draw';
    return game.winner === 0 ? 'Blue Won' : 'Red Won';
  };

  const getResultColor = (game: GameSummary) => {
    if (game.status === 'playing') return 'text-yellow-400';
    if (game.winner === null) return 'text-gray-400';
    return game.winner === 0 ? 'text-blue-400' : 'text-red-400';
  };

  const getPlayersText = (game: GameSummary) => {
    // Extract just the filename from model path (e.g., "/path/to/model.pt" -> "model.pt")
    const getModelName = (modelPath: string | null) => {
      if (!modelPath) return 'AI';
      const parts = modelPath.split('/');
      return parts[parts.length - 1];
    };

    const getPlayerName = (username: string | null, isAI: boolean) => {
      if (isAI) {
        return getModelName(game.ai_model_version);
      }
      // Use username if available, otherwise "Human"
      return username || 'Human';
    };

    const p1 = getPlayerName(game.player1_username, game.player1_type === 'ai');
    const p2 = getPlayerName(game.player2_username, game.player2_type === 'ai');
    return `${p1} vs ${p2}`;
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-6 max-w-4xl w-full mx-4 max-h-[90vh] flex flex-col">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold text-white">Game History</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Filters */}
        <div className="flex flex-wrap gap-4 mb-4">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Status</label>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="px-3 py-1 bg-gray-700 text-white rounded border border-gray-600 text-sm"
            >
              <option value="">All</option>
              <option value="playing">In Progress</option>
              <option value="finished">Finished</option>
            </select>
          </div>

          <div>
            <label className="block text-xs text-gray-400 mb-1">Result</label>
            <select
              value={resultFilter}
              onChange={(e) => setResultFilter(e.target.value)}
              className="px-3 py-1 bg-gray-700 text-white rounded border border-gray-600 text-sm"
            >
              <option value="">All</option>
              <option value="0">Blue Won</option>
              <option value="1">Red Won</option>
            </select>
          </div>

          {user && (
            <div className="flex items-end">
              <label className="flex items-center gap-2 text-sm text-gray-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={myGamesOnly}
                  onChange={(e) => setMyGamesOnly(e.target.checked)}
                  className="rounded"
                />
                My Games Only
              </label>
            </div>
          )}

          <div className="flex-1"></div>

          <div className="flex items-end">
            <button
              onClick={fetchGames}
              disabled={isLoading}
              className="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm transition-colors"
            >
              Refresh
            </button>
          </div>
        </div>

        {error && (
          <div className="bg-red-600 text-white px-3 py-2 rounded text-sm mb-4">
            {error}
          </div>
        )}

        {/* Games list */}
        <div className="flex-1 overflow-y-auto">
          {isLoading && games.length === 0 ? (
            <div className="text-center text-gray-400 py-8">Loading...</div>
          ) : games.length === 0 ? (
            <div className="text-center text-gray-400 py-8">No games found</div>
          ) : (
            <table className="w-full text-sm">
              <thead className="text-gray-400 border-b border-gray-700">
                <tr>
                  <th className="text-left py-2 px-2">ID</th>
                  <th className="text-left py-2 px-2">Date</th>
                  <th className="text-left py-2 px-2">Players</th>
                  <th className="text-left py-2 px-2">Result</th>
                  <th className="text-center py-2 px-2">Moves</th>
                  <th className="text-right py-2 px-2">Actions</th>
                </tr>
              </thead>
              <tbody>
                {games.map((game) => (
                  <tr key={game.game_id} className="border-b border-gray-700 hover:bg-gray-700/50">
                    <td className="py-2 px-2">
                      <code className="text-xs text-gray-400 bg-gray-700 px-1.5 py-0.5 rounded font-mono">
                        {game.game_id.slice(0, 8)}
                      </code>
                    </td>
                    <td className="py-2 px-2 text-gray-300">
                      {formatDate(game.created_at)}
                    </td>
                    <td className="py-2 px-2 text-gray-300">
                      {getPlayersText(game)}
                    </td>
                    <td className={`py-2 px-2 ${getResultColor(game)}`}>
                      {getResultText(game)}
                    </td>
                    <td className="py-2 px-2 text-center text-gray-300">
                      {game.move_count}
                    </td>
                    <td className="py-2 px-2 text-right">
                      <button
                        onClick={() => onSelectGame(game.game_id)}
                        className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded text-xs transition-colors"
                      >
                        Replay
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex justify-center items-center gap-4 mt-4 pt-4 border-t border-gray-700">
            <button
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={page === 1 || isLoading}
              className="px-3 py-1 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-500 text-white rounded text-sm transition-colors"
            >
              Previous
            </button>
            <span className="text-gray-400 text-sm">
              Page {page} of {totalPages}
            </span>
            <button
              onClick={() => setPage(p => Math.min(totalPages, p + 1))}
              disabled={page === totalPages || isLoading}
              className="px-3 py-1 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-500 text-white rounded text-sm transition-colors"
            >
              Next
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
