import { useState, useEffect, useCallback, useRef } from 'react';
import { useAuth } from '../contexts/AuthContext';
import * as onlineApi from '../api/online';
import type { ModelInfo } from '../api/engine';
import { getAutoMatchLevel, setAutoMatchLevel, getTierSettings, getLevelLabel, TIERS } from '../utils/autoMatch';

type GameMode = 'ai' | 'pvp' | 'online';
type ColorChoice = 'blue' | 'red' | 'random';
type OnlineTab = 'create' | 'join' | 'lobby';

const LOBBY_FILTER_CATEGORIES = ['bullet', 'blitz', 'rapid', 'correspondence'] as const;
type LobbyFilterCategory = typeof LOBBY_FILTER_CATEGORIES[number];

const LOBBY_FILTER_LABELS: Record<LobbyFilterCategory, string> = {
  bullet: 'Bullet',
  blitz: 'Blitz',
  rapid: 'Rapid',
  correspondence: 'Corr.',
};

const LOBBY_FILTERS_STORAGE_KEY = 'knightball_lobby_filters';

function loadSavedLobbyFilters(): Set<LobbyFilterCategory> {
  try {
    const saved = localStorage.getItem(LOBBY_FILTERS_STORAGE_KEY);
    if (saved) {
      const arr = JSON.parse(saved) as string[];
      const valid = arr.filter((s): s is LobbyFilterCategory =>
        LOBBY_FILTER_CATEGORIES.includes(s as LobbyFilterCategory)
      );
      if (valid.length > 0) return new Set(valid);
    }
  } catch {}
  return new Set(LOBBY_FILTER_CATEGORIES); // all selected by default
}

function saveLobbyFilters(filters: Set<LobbyFilterCategory>) {
  localStorage.setItem(LOBBY_FILTERS_STORAGE_KEY, JSON.stringify([...filters]));
}

function categorizeGame(game: onlineApi.PublicGameSummary): LobbyFilterCategory {
  if (game.game_mode === 'correspondence') return 'correspondence';
  if (!game.time_control) return 'rapid'; // untimed realtime = rapid
  if (game.time_control <= 180) return 'bullet';
  if (game.time_control <= 600) return 'blitz';
  return 'rapid';
}

// Simulation options (powers of 2)
const SIMULATION_OPTIONS = [
  { value: 64, label: '64' },
  { value: 128, label: '128' },
  { value: 256, label: '256' },
  { value: 512, label: '512' },
  { value: 1024, label: '1K' },
  { value: 2048, label: '2K' },
  { value: 4096, label: '4K' },
  { value: 8192, label: '8K' },
  { value: 16384, label: '16K' },
  { value: 32768, label: '32K' },
  { value: 65536, label: '64K' },
];

// Bot difficulty presets from pegasus training run Elo tournament
export const BOT_PRESETS = [
  { id: 'beginner', name: 'Beginner', model: 'pegasus_iter_010.pt', sims: 16, elo: 661 },
  { id: 'easy', name: 'Easy', model: 'pegasus_iter_075.pt', sims: 64, elo: 998 },
  { id: 'medium', name: 'Medium', model: 'pegasus_iter_150.pt', sims: 256, elo: 1070 },
  { id: 'hard', name: 'Hard', model: 'pegasus_iter_250.pt', sims: 1024, elo: 1219 },
  { id: 'expert', name: 'Expert', model: 'pegasus_iter_250.pt', sims: 4096, elo: 1300 },
] as const;

export type BotDifficulty = 'auto' | typeof BOT_PRESETS[number]['id'] | 'custom';

function formatTimeControl(game: onlineApi.PublicGameSummary): string {
  if (game.game_mode === 'correspondence') {
    return `${game.days_per_move}d/move`;
  }
  if (!game.time_control) return 'No limit';
  const mins = Math.floor(game.time_control / 60);
  return game.increment ? `${mins}+${game.increment}` : `${mins}+0`;
}

function formatTimeSince(iso: string): string {
  const diffMs = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diffMs / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

function formatDeadlineShort(deadline: string): string {
  const diffMs = new Date(deadline).getTime() - Date.now();
  if (diffMs <= 0) return 'Expired';
  const hours = Math.floor(diffMs / 3600000);
  const days = Math.floor(hours / 24);
  const h = hours % 24;
  if (days > 0) return `${days}d ${h}h`;
  if (hours > 0) return `${hours}h`;
  return `${Math.floor(diffMs / 60000)}m`;
}

export interface NewGameSettings {
  mode: GameMode;
  model?: string;       // undefined = latest
  simulations: number;
  colorChoice: ColorChoice;
  timeControl?: number | null;  // seconds per player, null = untimed (PvP only)
  increment?: number;  // seconds added per turn (PvP only)
  difficulty?: BotDifficulty;  // Named difficulty level, or 'custom' for manual selection
}

interface NewGameDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onStartGame: (settings: NewGameSettings) => void;
  onGameCreated: (gameId: string, joinCode: string, hostColor: number) => void;
  onGameJoined: (gameId: string, yourColor: number) => void;
  availableModels: ModelInfo[];
  currentSettings: NewGameSettings;
}

export default function NewGameDialog({
  isOpen,
  onClose,
  onStartGame,
  onGameCreated,
  onGameJoined,
  availableModels,
  currentSettings,
}: NewGameDialogProps) {
  const { user } = useAuth();

  // Shared state
  const [mode, setMode] = useState<GameMode>(currentSettings.mode);
  const [model, setModel] = useState<string | undefined>(currentSettings.model);
  const [simulations, setSimulations] = useState(currentSettings.simulations);
  const [difficulty, setDifficulty] = useState<BotDifficulty>(currentSettings.difficulty ?? 'auto');
  const [selectedLevel, setSelectedLevel] = useState(getAutoMatchLevel);
  const [colorChoice, setColorChoice] = useState<ColorChoice>(currentSettings.colorChoice);
  const [timeControl, setTimeControl] = useState<number | null>(currentSettings.timeControl ?? null);
  const [increment, setIncrement] = useState<number>(currentSettings.increment ?? 0);
  const [timePreset, setTimePreset] = useState<string>('none');
  const [customMinutes, setCustomMinutes] = useState(5);
  const [customIncrement, setCustomIncrement] = useState(0);

  // Online-specific state
  const [onlineTab, setOnlineTab] = useState<OnlineTab>('lobby');
  const [onlineColorChoice, setOnlineColorChoice] = useState<ColorChoice>('random');
  const [onlineTimePreset, setOnlineTimePreset] = useState<string>('none');
  const [onlineTimeControl, setOnlineTimeControl] = useState<number | null>(null);
  const [onlineIncrement, setOnlineIncrement] = useState<number>(0);
  const [onlineCustomMinutes, setOnlineCustomMinutes] = useState(5);
  const [onlineCustomIncrement, setOnlineCustomIncrement] = useState(0);
  const [gameMode, setGameMode] = useState<string>('realtime');
  const [daysPerMove, setDaysPerMove] = useState<number>(2);
  const [isPublic, setIsPublic] = useState(true);
  const [joinCode, setJoinCode] = useState('');
  const [onlineError, setOnlineError] = useState<string | null>(null);
  const [onlineLoading, setOnlineLoading] = useState(false);
  const [myGames, setMyGames] = useState<onlineApi.MyOnlineGamesResponse | null>(null);

  // Lobby state
  const [lobbyGames, setLobbyGames] = useState<onlineApi.PublicGameSummary[]>([]);
  const [lobbyFilters, setLobbyFilters] = useState<Set<LobbyFilterCategory>>(loadSavedLobbyFilters);
  const [lobbyLoading, setLobbyLoading] = useState(false);
  const refreshTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const toggleLobbyFilter = (cat: LobbyFilterCategory) => {
    setLobbyFilters(prev => {
      const next = new Set(prev);
      if (next.has(cat)) {
        // Don't allow deselecting all
        if (next.size > 1) next.delete(cat);
      } else {
        next.add(cat);
      }
      saveLobbyFilters(next);
      return next;
    });
  };

  const allFiltersSelected = lobbyFilters.size === LOBBY_FILTER_CATEGORIES.length;

  const filteredLobbyGames = lobbyGames.filter(game =>
    allFiltersSelected || lobbyFilters.has(categorizeGame(game))
  );

  const loadLobbyGames = useCallback(async () => {
    setLobbyLoading(true);
    try {
      const result = await onlineApi.getPublicLobby(); // fetch all, filter client-side
      setLobbyGames(result.games);
    } catch (err) {
      console.error('Failed to load lobby:', err);
    } finally {
      setLobbyLoading(false);
    }
  }, []);

  const loadMyGames = async () => {
    try {
      const games = await onlineApi.getMyOnlineGames();
      setMyGames(games);
    } catch (err) {
      console.error('Failed to load games:', err);
    }
  };

  // Sync with current settings when dialog opens
  useEffect(() => {
    if (isOpen) {
      setMode(currentSettings.mode);
      setModel(currentSettings.model);
      setSimulations(currentSettings.simulations);
      setDifficulty(currentSettings.difficulty ?? 'auto');
      setColorChoice(currentSettings.colorChoice);
      setTimeControl(currentSettings.timeControl ?? null);
      setIncrement(currentSettings.increment ?? 0);
      setOnlineError(null);
      setJoinCode('');
      if (user) {
        loadMyGames();
      }
    }
  }, [isOpen, currentSettings, user]);

  // Auto-refresh lobby every 10 seconds when lobby tab is active and mode is online
  useEffect(() => {
    if (isOpen && mode === 'online' && onlineTab === 'lobby') {
      loadLobbyGames();
      refreshTimerRef.current = setInterval(loadLobbyGames, 10000);
    }
    return () => {
      if (refreshTimerRef.current) {
        clearInterval(refreshTimerRef.current);
        refreshTimerRef.current = null;
      }
    };
  }, [isOpen, mode, onlineTab, loadLobbyGames]);

  if (!isOpen) return null;

  const handleStart = () => {
    // Resolve model and sims from difficulty preset
    let resolvedModel = model;
    let resolvedSims = simulations;
    if (difficulty === 'auto') {
      const tier = getTierSettings(getAutoMatchLevel());
      resolvedModel = tier.model;
      resolvedSims = tier.sims;
    } else if (difficulty !== 'custom') {
      const preset = BOT_PRESETS.find(p => p.id === difficulty);
      if (preset) {
        resolvedModel = preset.model;
        resolvedSims = preset.sims;
      }
    }
    onStartGame({
      mode,
      model: resolvedModel,
      simulations: resolvedSims,
      colorChoice,
      timeControl: mode === 'pvp' ? timeControl : null,
      increment: mode === 'pvp' ? increment : 0,
      difficulty,
    });
  };

  const handleCreateGame = async () => {
    setOnlineError(null);
    setOnlineLoading(true);

    // Resolve color: random picks 0 or 1
    const hostColor = onlineColorChoice === 'random'
      ? (Math.random() < 0.5 ? 0 : 1)
      : onlineColorChoice === 'blue' ? 0 : 1;

    try {
      const result = await onlineApi.createOnlineGame(
        hostColor,
        gameMode === 'correspondence' ? null : onlineTimeControl,
        gameMode === 'correspondence' ? 0 : onlineIncrement,
        gameMode,
        gameMode === 'correspondence' ? daysPerMove : null,
        isPublic,
      );
      onClose();
      onGameCreated(result.game_id, result.join_code, result.host_color);
    } catch (err) {
      setOnlineError(
        err instanceof onlineApi.OnlineAPIError ? err.message : 'Failed to create game'
      );
    } finally {
      setOnlineLoading(false);
    }
  };

  const handleJoinGame = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!joinCode.trim()) {
      setOnlineError('Please enter a game code');
      return;
    }

    setOnlineError(null);
    setOnlineLoading(true);

    try {
      const result = await onlineApi.joinOnlineGame(joinCode.trim());
      onClose();
      onGameJoined(result.game_id, result.your_color);
    } catch (err) {
      if (err instanceof onlineApi.OnlineAPIError) {
        if (err.status === 404) {
          setOnlineError('Game not found. Check the code and try again.');
        } else if (err.status === 409) {
          setOnlineError('Game is no longer available.');
        } else {
          setOnlineError(err.message);
        }
      } else {
        setOnlineError('Failed to join game');
      }
    } finally {
      setOnlineLoading(false);
    }
  };

  const handleJoinLobbyGame = async (lobbyJoinCode: string) => {
    setOnlineError(null);
    setOnlineLoading(true);

    try {
      const result = await onlineApi.joinOnlineGame(lobbyJoinCode);
      onClose();
      onGameJoined(result.game_id, result.your_color);
    } catch (err) {
      if (err instanceof onlineApi.OnlineAPIError) {
        if (err.status === 409) {
          setOnlineError('Game is no longer available.');
          loadLobbyGames();
        } else {
          setOnlineError(err.message);
        }
      } else {
        setOnlineError('Failed to join game');
      }
    } finally {
      setOnlineLoading(false);
    }
  };

  const handleQuickMatch = async () => {
    setOnlineError(null);
    setOnlineLoading(true);

    try {
      // Derive time controls from selected filters
      const hasRealtime = lobbyFilters.has('bullet') || lobbyFilters.has('blitz') || lobbyFilters.has('rapid');
      const onlyCorrespondence = !hasRealtime && lobbyFilters.has('correspondence');

      let qmMode = 'realtime';
      let qmTimeControl: number | null = null;
      let qmIncrement = 0;
      let qmDaysPerMove: number | null = null;

      if (onlyCorrespondence) {
        qmMode = 'correspondence';
        qmDaysPerMove = 2;
      } else if (lobbyFilters.size === 1) {
        // Single realtime category selected — use its default time control
        const category = [...lobbyFilters][0];
        if (category === 'bullet') { qmTimeControl = 120; }
        else if (category === 'blitz') { qmTimeControl = 300; }
        else if (category === 'rapid') { qmTimeControl = 600; }
      }
      // Multiple categories or all selected: no specific time control (server picks)

      const result = await onlineApi.quickMatch(qmMode, 0, qmTimeControl, qmIncrement, qmDaysPerMove);
      onClose();
      if (result.action === 'joined') {
        onGameJoined(result.game_id, result.your_color);
      } else {
        onGameCreated(result.game_id, result.join_code!, result.your_color);
      }
    } catch (err) {
      setOnlineError(
        err instanceof onlineApi.OnlineAPIError ? err.message : 'Failed to find a match'
      );
    } finally {
      setOnlineLoading(false);
    }
  };

  const handleResumeGame = (gameId: string, yourColor: number) => {
    onClose();
    onGameJoined(gameId, yourColor);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />

      {/* Dialog */}
      <div className="relative bg-gray-800 rounded-lg shadow-xl max-w-md w-full p-6 max-h-[90vh] overflow-y-auto">
        <h2 className="text-xl font-bold mb-4">New Game</h2>

        {/* Mode selection */}
        <div className="mb-4">
          <label className="block text-xs text-gray-400 mb-2">Game Mode</label>
          <div className="flex gap-2">
            <button
              onClick={() => setMode('ai')}
              className={`flex-1 px-3 py-2 rounded font-medium text-sm transition-colors ${
                mode === 'ai'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              vs AI
            </button>
            <button
              onClick={() => setMode('pvp')}
              className={`flex-1 px-3 py-2 rounded font-medium text-sm transition-colors ${
                mode === 'pvp'
                  ? 'bg-purple-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Local 2P
            </button>
            <button
              onClick={() => setMode('online')}
              className={`flex-1 px-3 py-2 rounded font-medium text-sm transition-colors ${
                mode === 'online'
                  ? 'bg-green-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Online
            </button>
          </div>
        </div>

        {/* AI settings */}
        {mode === 'ai' && (
          <>
            <div className="mb-3">
              <label className="block text-xs text-gray-400 mb-2">Difficulty</label>
              {/* Auto-match toggle */}
              <div className="flex items-center gap-2 mb-2">
                <button
                  onClick={() => setDifficulty('auto')}
                  className={`flex-1 px-2 py-2 rounded text-sm font-medium transition-colors ${
                    difficulty === 'auto'
                      ? 'bg-green-600 text-white ring-2 ring-green-400'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  Auto — {getLevelLabel(selectedLevel)}
                </button>
                <button
                  onClick={() => setDifficulty('custom')}
                  className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                    difficulty === 'custom'
                      ? 'bg-gray-500 text-white ring-2 ring-gray-400'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  Custom
                </button>
              </div>
              {difficulty === 'auto' && (
                <>
                  <p className="text-xs text-gray-500 mb-2">
                    Adjusts after each game. Or pick a level:
                  </p>
                  <div className="grid grid-cols-5 gap-1">
                    {TIERS.map((tier, i) => {
                      const level = i + 1;
                      const isCurrentAuto = level === selectedLevel;
                      return (
                        <button
                          key={level}
                          onClick={() => {
                            setAutoMatchLevel(level);
                            setSelectedLevel(level);
                          }}
                          className={`px-1 py-1.5 rounded text-xs font-medium transition-colors ${
                            isCurrentAuto
                              ? 'bg-green-600 text-white ring-1 ring-green-400'
                              : 'bg-gray-700 text-gray-400 hover:bg-gray-600 hover:text-white'
                          }`}
                          title={tier.label}
                        >
                          {level}
                        </button>
                      );
                    })}
                  </div>
                </>
              )}
            </div>

            {/* Custom model/sims controls */}
            {difficulty === 'custom' && (
              <>
                <div className="mb-3">
                  <label className="block text-xs text-gray-400 mb-1">Model</label>
                  <select
                    value={model || ''}
                    onChange={(e) => setModel(e.target.value || undefined)}
                    className="w-full px-3 py-1.5 bg-gray-700 text-white rounded border border-gray-600 text-sm"
                  >
                    <option value="">Latest</option>
                    {availableModels.map((m) => (
                      <option key={m.path} value={m.path}>
                        {m.name}{m.name !== 'random_weights' && !m.has_onnx ? ' *' : ''}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="mb-3">
                  <label className="block text-xs text-gray-400 mb-1">Simulations</label>
                  <select
                    value={simulations}
                    onChange={(e) => setSimulations(Number(e.target.value))}
                    className="w-full px-3 py-1.5 bg-gray-700 text-white rounded border border-gray-600 text-sm"
                  >
                    {SIMULATION_OPTIONS.map((opt) => (
                      <option key={opt.value} value={opt.value}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                </div>
              </>
            )}

            <div className="mb-4">
              <label className="block text-xs text-gray-400 mb-1">Play as</label>
              <div className="flex gap-2">
                <button
                  onClick={() => setColorChoice('random')}
                  className={`flex-1 px-3 py-1.5 rounded text-sm transition-colors ${
                    colorChoice === 'random'
                      ? 'bg-gray-500 text-white ring-2 ring-white/50'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  Random
                </button>
                <button
                  onClick={() => setColorChoice('blue')}
                  className={`flex-1 px-3 py-1.5 rounded text-sm transition-colors ${
                    colorChoice === 'blue'
                      ? 'bg-blue-600 text-white ring-2 ring-white/50'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  Blue
                </button>
                <button
                  onClick={() => setColorChoice('red')}
                  className={`flex-1 px-3 py-1.5 rounded text-sm transition-colors ${
                    colorChoice === 'red'
                      ? 'bg-red-600 text-white ring-2 ring-white/50'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  Red
                </button>
              </div>
            </div>
          </>
        )}

        {/* PvP settings */}
        {mode === 'pvp' && (
          <div className="mb-4 space-y-3">
            <div className="text-sm text-gray-400">
              Two players on the same device. Blue moves first.
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">Time control</label>
              <div className="flex flex-wrap gap-2">
                {([
                  { key: 'none', label: 'No limit', time: null, inc: 0 },
                  { key: 'bullet', label: 'Bullet 2+0', time: 120, inc: 0 },
                  { key: 'blitz', label: 'Blitz 5+0', time: 300, inc: 0 },
                  { key: 'rapid', label: 'Rapid 10+0', time: 600, inc: 0 },
                  { key: 'custom', label: 'Custom', time: null, inc: 0 },
                ] as { key: string; label: string; time: number | null; inc: number }[]).map((opt) => (
                  <button
                    key={opt.key}
                    onClick={() => {
                      setTimePreset(opt.key);
                      if (opt.key === 'custom') {
                        setTimeControl(customMinutes * 60);
                        setIncrement(customIncrement);
                      } else {
                        setTimeControl(opt.time);
                        setIncrement(opt.inc);
                      }
                    }}
                    className={`px-2 py-1 rounded text-xs transition-colors ${
                      timePreset === opt.key
                        ? 'bg-purple-600 text-white ring-1 ring-purple-400'
                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    }`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
              {timePreset === 'custom' && (
                <div className="flex gap-3 mt-2">
                  <div className="flex-1">
                    <label className="block text-xs text-gray-500 mb-1">Minutes</label>
                    <input
                      type="number"
                      min={1}
                      max={60}
                      value={customMinutes}
                      onChange={(e) => {
                        const v = Math.max(1, Math.min(60, Number(e.target.value) || 1));
                        setCustomMinutes(v);
                        setTimeControl(v * 60);
                      }}
                      className="w-full px-2 py-1 bg-gray-700 text-white rounded border border-gray-600 text-sm"
                    />
                  </div>
                  <div className="flex-1">
                    <label className="block text-xs text-gray-500 mb-1">Increment (sec)</label>
                    <input
                      type="number"
                      min={0}
                      max={30}
                      value={customIncrement}
                      onChange={(e) => {
                        const v = Math.max(0, Math.min(30, Number(e.target.value) || 0));
                        setCustomIncrement(v);
                        setIncrement(v);
                      }}
                      className="w-full px-2 py-1 bg-gray-700 text-white rounded border border-gray-600 text-sm"
                    />
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Online settings - inline lobby */}
        {mode === 'online' && (
          <div className="mb-4">
            {onlineError && (
              <div className="mb-3 bg-red-600 text-white px-3 py-2 rounded text-sm">
                {onlineError}
              </div>
            )}

            {/* Online sub-tabs */}
            <div className="flex gap-2 mb-3">
              <button
                onClick={() => setOnlineTab('create')}
                className={`flex-1 px-3 py-1.5 rounded font-medium transition-colors text-sm ${
                  onlineTab === 'create'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                Create
              </button>
              <button
                onClick={() => setOnlineTab('join')}
                className={`flex-1 px-3 py-1.5 rounded font-medium transition-colors text-sm ${
                  onlineTab === 'join'
                    ? 'bg-green-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                Join Code
              </button>
              <button
                onClick={() => setOnlineTab('lobby')}
                className={`flex-1 px-3 py-1.5 rounded font-medium transition-colors text-sm ${
                  onlineTab === 'lobby'
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                Lobby
              </button>
            </div>

            {/* Create Game sub-tab */}
            {onlineTab === 'create' && (
              <div className="space-y-3">
                {/* Color picker with Random */}
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Play as</label>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setOnlineColorChoice('random')}
                      className={`flex-1 px-3 py-2 rounded font-medium text-sm transition-colors flex items-center justify-center gap-1.5 ${
                        onlineColorChoice === 'random'
                          ? 'bg-gray-500 text-white ring-2 ring-white/50'
                          : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                      }`}
                    >
                      Random
                    </button>
                    <button
                      onClick={() => setOnlineColorChoice('blue')}
                      className={`flex-1 px-3 py-2 rounded font-medium text-sm transition-colors flex items-center justify-center gap-1.5 ${
                        onlineColorChoice === 'blue'
                          ? 'bg-blue-600 text-white ring-2 ring-blue-400'
                          : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                      }`}
                    >
                      <span className="w-3 h-3 rounded-full bg-blue-400"></span>
                      Blue
                    </button>
                    <button
                      onClick={() => setOnlineColorChoice('red')}
                      className={`flex-1 px-3 py-2 rounded font-medium text-sm transition-colors flex items-center justify-center gap-1.5 ${
                        onlineColorChoice === 'red'
                          ? 'bg-red-600 text-white ring-2 ring-red-400'
                          : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                      }`}
                    >
                      <span className="w-3 h-3 rounded-full bg-red-400"></span>
                      Red
                    </button>
                  </div>
                </div>

                {/* Time Control */}
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Time control</label>
                  <div className="flex flex-wrap gap-2">
                    {([
                      { key: 'none', label: 'No limit', time: null, inc: 0 },
                      { key: 'bullet', label: '2+0', time: 120, inc: 0 },
                      { key: 'blitz', label: '5+0', time: 300, inc: 0 },
                      { key: 'rapid', label: '10+0', time: 600, inc: 0 },
                      { key: 'custom', label: 'Custom', time: null, inc: 0 },
                      { key: 'correspondence', label: 'Corr.', time: null, inc: 0 },
                    ] as { key: string; label: string; time: number | null; inc: number }[]).map((opt) => (
                      <button
                        key={opt.key}
                        onClick={() => {
                          setOnlineTimePreset(opt.key);
                          if (opt.key === 'correspondence') {
                            setGameMode('correspondence');
                            setOnlineTimeControl(null);
                            setOnlineIncrement(0);
                          } else {
                            setGameMode('realtime');
                            if (opt.key === 'custom') {
                              setOnlineTimeControl(onlineCustomMinutes * 60);
                              setOnlineIncrement(onlineCustomIncrement);
                            } else {
                              setOnlineTimeControl(opt.time);
                              setOnlineIncrement(opt.inc);
                            }
                          }
                        }}
                        className={`px-2 py-1.5 rounded font-medium text-xs transition-colors ${
                          onlineTimePreset === opt.key
                            ? opt.key === 'correspondence'
                              ? 'bg-amber-600 text-white ring-1 ring-amber-400'
                              : 'bg-purple-600 text-white ring-1 ring-purple-400'
                            : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                        }`}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                  {onlineTimePreset === 'custom' && (
                    <div className="flex gap-3 mt-2">
                      <div className="flex-1">
                        <label className="block text-xs text-gray-500 mb-1">Minutes</label>
                        <input
                          type="number"
                          min={1}
                          max={60}
                          value={onlineCustomMinutes}
                          onChange={(e) => {
                            const v = Math.max(1, Math.min(60, Number(e.target.value) || 1));
                            setOnlineCustomMinutes(v);
                            setOnlineTimeControl(v * 60);
                          }}
                          className="w-full px-2 py-1 bg-gray-700 text-white rounded border border-gray-600 text-sm"
                        />
                      </div>
                      <div className="flex-1">
                        <label className="block text-xs text-gray-500 mb-1">Increment (sec)</label>
                        <input
                          type="number"
                          min={0}
                          max={30}
                          value={onlineCustomIncrement}
                          onChange={(e) => {
                            const v = Math.max(0, Math.min(30, Number(e.target.value) || 0));
                            setOnlineCustomIncrement(v);
                            setOnlineIncrement(v);
                          }}
                          className="w-full px-2 py-1 bg-gray-700 text-white rounded border border-gray-600 text-sm"
                        />
                      </div>
                    </div>
                  )}
                  {onlineTimePreset === 'correspondence' && (
                    <div className="mt-2">
                      <label className="block text-xs text-gray-500 mb-1">Days per move</label>
                      <div className="flex gap-2">
                        {[1, 2, 3].map((d) => (
                          <button
                            key={d}
                            onClick={() => setDaysPerMove(d)}
                            className={`flex-1 px-2 py-1.5 rounded font-medium text-xs transition-colors ${
                              daysPerMove === d
                                ? 'bg-amber-600 text-white ring-1 ring-amber-400'
                                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                            }`}
                          >
                            {d}d
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Public / Private toggle */}
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Visibility</label>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setIsPublic(false)}
                      className={`flex-1 px-3 py-1.5 rounded font-medium text-sm transition-colors ${
                        !isPublic
                          ? 'bg-gray-600 text-white ring-2 ring-gray-400'
                          : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                      }`}
                    >
                      Private
                    </button>
                    <button
                      onClick={() => setIsPublic(true)}
                      className={`flex-1 px-3 py-1.5 rounded font-medium text-sm transition-colors ${
                        isPublic
                          ? 'bg-purple-600 text-white ring-2 ring-purple-400'
                          : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                      }`}
                    >
                      Public
                    </button>
                  </div>
                </div>

                <button
                  onClick={handleCreateGame}
                  disabled={onlineLoading}
                  className="w-full px-4 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded font-medium transition-colors"
                >
                  {onlineLoading ? 'Creating...' : 'Create Game'}
                </button>
              </div>
            )}

            {/* Join Code sub-tab */}
            {onlineTab === 'join' && (
              <form onSubmit={handleJoinGame} className="space-y-3">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Enter game code</label>
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
                  disabled={onlineLoading || !joinCode.trim()}
                  className="w-full px-4 py-2.5 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white rounded font-medium transition-colors"
                >
                  {onlineLoading ? 'Joining...' : 'Join Game'}
                </button>
              </form>
            )}

            {/* Lobby sub-tab */}
            {onlineTab === 'lobby' && (
              <div className="space-y-3">
                {/* Quick Match button */}
                <button
                  onClick={handleQuickMatch}
                  disabled={onlineLoading}
                  className="w-full px-4 py-2.5 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 disabled:from-gray-600 disabled:to-gray-600 text-white rounded-lg font-bold text-base transition-all"
                >
                  {onlineLoading ? 'Matching...' : 'Quick Match'}
                </button>

                {/* Multi-select filter */}
                <div className="flex gap-1.5">
                  {LOBBY_FILTER_CATEGORIES.map((cat) => (
                    <button
                      key={cat}
                      onClick={() => toggleLobbyFilter(cat)}
                      className={`flex-1 px-2 py-1 rounded text-xs font-medium transition-colors ${
                        lobbyFilters.has(cat)
                          ? cat === 'correspondence'
                            ? 'bg-amber-600 text-white'
                            : 'bg-purple-600 text-white'
                          : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
                      }`}
                    >
                      {LOBBY_FILTER_LABELS[cat]}
                    </button>
                  ))}
                </div>

                {/* Game list */}
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {lobbyLoading && lobbyGames.length === 0 && (
                    <p className="text-center text-gray-400 text-sm py-4">Loading...</p>
                  )}

                  {!lobbyLoading && filteredLobbyGames.length === 0 && (
                    <div className="text-center py-4">
                      <p className="text-gray-400 text-sm mb-1">No public games available.</p>
                      <p className="text-gray-500 text-xs">Create one or use Quick Match!</p>
                    </div>
                  )}

                  {filteredLobbyGames.map((game) => (
                    <div
                      key={game.game_id}
                      className="flex items-center justify-between bg-gray-700 rounded-lg p-2.5"
                    >
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-0.5">
                          <span className="text-sm text-white font-medium truncate">
                            {game.host_username || game.host_display_name || 'Anonymous'}
                          </span>
                          <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${
                            game.game_mode === 'correspondence'
                              ? 'bg-amber-600/30 text-amber-300'
                              : 'bg-blue-600/30 text-blue-300'
                          }`}>
                            {game.game_mode === 'correspondence' ? 'Corr' : 'Live'}
                          </span>
                        </div>
                        <div className="flex items-center gap-2 text-xs text-gray-400">
                          <span>{formatTimeControl(game)}</span>
                          <span>&middot;</span>
                          <span>{formatTimeSince(game.created_at)}</span>
                        </div>
                      </div>
                      <button
                        onClick={() => handleJoinLobbyGame(game.join_code)}
                        disabled={onlineLoading}
                        className="ml-3 px-3 py-1.5 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white rounded text-sm font-medium transition-colors"
                      >
                        Join
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* My Active Games */}
            {user && myGames && (myGames.active.length > 0 || myGames.waiting.length > 0) && (
              <div className="mt-3 pt-3 border-t border-gray-700">
                <h3 className="text-xs font-medium text-gray-400 mb-2">Your Games</h3>

                {myGames.waiting.length > 0 && (
                  <div className="mb-2">
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
                          {game.game_mode === 'correspondence' && (
                            <span className="text-xs text-amber-400" title="Correspondence">&#128197;</span>
                          )}
                          <span className="text-sm text-gray-300">
                            vs {game.opponent_username || game.opponent_name || 'Opponent'}
                          </span>
                          {game.is_your_turn && (
                            <span className="text-xs bg-green-600 px-1 rounded">Your turn</span>
                          )}
                          {game.is_your_turn && game.game_mode === 'correspondence' && game.move_deadline && (
                            <span className="text-xs text-amber-300">{formatDeadlineShort(game.move_deadline)}</span>
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
        )}

        {/* Buttons - only show for non-online modes */}
        {mode !== 'online' && (
          <div className="flex gap-3">
            <button
              onClick={onClose}
              className="flex-1 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded font-medium transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleStart}
              className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded font-medium transition-colors"
            >
              Start Game
            </button>
          </div>
        )}

        {/* Cancel button for online mode */}
        {mode === 'online' && (
          <div className="mt-3">
            <button
              onClick={onClose}
              className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded font-medium transition-colors text-sm"
            >
              Cancel
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
