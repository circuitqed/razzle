import { useState, useEffect, useCallback } from 'react';
import * as historyApi from '../api/history';
import type { HistoryGame, GameSummaryStats, RecordCounts } from '../api/history';
import { useAuth } from '../contexts/AuthContext';
import { useDialogA11y } from '../hooks/useDialogA11y';
import { readLocalProgress } from '../utils/aiLevelSync';

interface MyGamesProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectGame: (gameId: string) => void;
  onBrowseAll: () => void;
  onOpenLogin: () => void;
  onOpenRegister: () => void;
}

const PER_PAGE = 15;

function formatDate(dateStr: string): string {
  return new Intl.DateTimeFormat(undefined, {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  }).format(new Date(dateStr));
}

function recordText(r: RecordCounts): string {
  return `${r.wins}W ${r.losses}L${r.draws ? ` ${r.draws}D` : ''}`;
}

const RESULT_STYLE: Record<string, { text: string; className: string }> = {
  win: { text: 'Won', className: 'text-green-400' },
  loss: { text: 'Lost', className: 'text-red-400' },
  draw: { text: 'Draw', className: 'text-gray-400' },
  in_progress: { text: 'In progress', className: 'text-yellow-400' },
  abandoned: { text: 'Abandoned', className: 'text-gray-500' },
  aborted: { text: 'Aborted', className: 'text-gray-500' },
};

function opponentText(g: HistoryGame): string {
  const o = g.opponent;
  if (o.type === 'ai') {
    const detail = o.ai_level ? `Level ${o.ai_level}` : (o.ai_simulations ? `${o.ai_simulations} sims` : '');
    return detail ? `AI · ${detail}` : 'AI';
  }
  if (o.type === 'human') return o.username ? `@${o.username}` : o.name;
  return 'Pass & play';
}

function StatTile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-gray-700/60 rounded-lg px-3 py-2">
      <div className="text-xs text-gray-400">{label}</div>
      <div className="text-lg font-semibold text-white">{value}</div>
      {sub && <div className="text-xs text-gray-400">{sub}</div>}
    </div>
  );
}

export default function MyGames({ isOpen, onClose, onSelectGame, onBrowseAll, onOpenLogin, onOpenRegister }: MyGamesProps) {
  const dialogRef = useDialogA11y(onClose);
  const { user } = useAuth();
  const [summary, setSummary] = useState<GameSummaryStats | null>(null);
  const [games, setGames] = useState<HistoryGame[]>([]);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!user) return;
    setIsLoading(true);
    setError(null);
    try {
      const [s, h] = await Promise.all([
        historyApi.getMySummary(),
        historyApi.getMyGames(page, PER_PAGE),
      ]);
      setSummary(s);
      setGames(h.games);
      setTotalPages(h.total_pages);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load your games');
    } finally {
      setIsLoading(false);
    }
  }, [user, page]);

  useEffect(() => {
    if (isOpen) load();
  }, [isOpen, load]);

  if (!isOpen) return null;

  // Local progress may be ahead of the server until the next sync lands.
  const local = readLocalProgress();
  const highest = Math.max(summary?.highest_ai_level_beaten ?? 0, local.highestBeaten);
  const currentLevel = local.level;

  return (
    <div ref={dialogRef} role="dialog" aria-modal="true" aria-label="My games"
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-4 sm:p-6 max-w-2xl w-full mx-4 max-h-[90vh] flex flex-col">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold text-white">My Games</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-white transition-colors" aria-label="Close">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {!user ? (
          <div className="text-center py-8 px-2">
            <p className="text-gray-200 mb-1">Keep your games and track your progress.</p>
            <p className="text-sm text-gray-400 mb-5">
              With a free account, every game you play is saved here with your record against each AI level.
            </p>
            <div className="flex justify-center gap-3">
              <button onClick={onOpenRegister} className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded font-medium transition-colors">
                Sign up
              </button>
              <button onClick={onOpenLogin} className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded font-medium transition-colors">
                Log in
              </button>
            </div>
            <button onClick={onBrowseAll} className="mt-6 text-sm text-gray-400 hover:text-white transition-colors">
              Find a player
            </button>
          </div>
        ) : (
          <>
            {/* Summary */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
              <StatTile label="vs AI" value={summary ? recordText(summary.vs_ai) : '–'} />
              <StatTile label="vs People" value={summary ? recordText(summary.vs_human) : '–'} />
              <StatTile label="Highest level beaten" value={highest ? `Level ${highest}` : '–'} />
              <StatTile label="Auto-match level" value={`Level ${currentLevel}`} />
            </div>

            {summary && summary.vs_ai.by_level.length > 0 && (
              <div className="flex flex-wrap gap-1.5 mb-3" aria-label="Record by AI level">
                {summary.vs_ai.by_level.map(l => (
                  <span key={l.level} className="text-xs bg-gray-700 text-gray-300 rounded px-2 py-0.5">
                    L{l.level}: {recordText(l)}
                  </span>
                ))}
              </div>
            )}

            {error && <div className="bg-red-600 text-white px-3 py-2 rounded text-sm mb-3">{error}</div>}

            {/* Recent games */}
            <div className="flex-1 overflow-y-auto min-h-0">
              {isLoading && games.length === 0 ? (
                <div className="text-center text-gray-400 py-8">Loading...</div>
              ) : games.length === 0 ? (
                <div className="text-center text-gray-400 py-8">No games yet — finished games will show up here.</div>
              ) : (
                <ul className="divide-y divide-gray-700">
                  {games.map(g => {
                    const style = g.result ? RESULT_STYLE[g.result] : null;
                    return (
                      <li key={g.game_id} onClick={() => onSelectGame(g.game_id)}
                        className="flex items-center gap-3 py-2 px-1 -mx-1 rounded cursor-pointer hover:bg-gray-700/50">
                        <span className={`w-3 h-3 rounded-full shrink-0 ${g.your_color === 0 ? 'bg-blue-500' : 'bg-red-500'}`}
                          title={g.your_color === 0 ? 'You played blue' : 'You played red'} />
                        <div className="flex-1 min-w-0">
                          <div className="text-sm text-gray-200 truncate">{opponentText(g)}</div>
                          <div className="text-xs text-gray-500">
                            {formatDate(g.finished_at ?? g.updated_at)} · {g.move_count} moves{g.resigned ? ' · resigned' : ''}
                          </div>
                        </div>
                        {style && <span className={`text-sm font-medium ${style.className}`}>{style.text}</span>}
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>

            <div className="flex flex-wrap justify-between items-center gap-2 mt-3 pt-3 border-t border-gray-700">
              <button onClick={onBrowseAll} className="text-sm text-gray-400 hover:text-white transition-colors">
                Find a player
              </button>
              {totalPages > 1 && (
                <div className="flex items-center gap-3">
                  <button
                    onClick={() => setPage(p => Math.max(1, p - 1))}
                    disabled={page === 1 || isLoading}
                    className="px-3 py-1 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-500 text-white rounded text-sm transition-colors"
                  >
                    Previous
                  </button>
                  <span className="text-gray-400 text-sm">{page} / {totalPages}</span>
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
          </>
        )}
      </div>
    </div>
  );
}
