import { useState, useEffect } from 'react';
import type { ModelInfo } from '../api/engine';

type GameMode = 'ai' | 'pvp' | 'online';
type ColorChoice = 'blue' | 'red' | 'random';

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

export interface NewGameSettings {
  mode: GameMode;
  model?: string;       // undefined = latest
  simulations: number;
  colorChoice: ColorChoice;
}

interface NewGameDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onStartGame: (settings: NewGameSettings) => void;
  onPlayOnline: () => void;
  availableModels: ModelInfo[];
  currentSettings: NewGameSettings;
}

export default function NewGameDialog({
  isOpen,
  onClose,
  onStartGame,
  onPlayOnline,
  availableModels,
  currentSettings,
}: NewGameDialogProps) {
  const [mode, setMode] = useState<GameMode>(currentSettings.mode);
  const [model, setModel] = useState<string | undefined>(currentSettings.model);
  const [simulations, setSimulations] = useState(currentSettings.simulations);
  const [colorChoice, setColorChoice] = useState<ColorChoice>(currentSettings.colorChoice);

  // Sync with current settings when dialog opens
  useEffect(() => {
    if (isOpen) {
      setMode(currentSettings.mode);
      setModel(currentSettings.model);
      setSimulations(currentSettings.simulations);
      setColorChoice(currentSettings.colorChoice);
    }
  }, [isOpen, currentSettings]);

  if (!isOpen) return null;

  const handleStart = () => {
    if (mode === 'online') {
      onPlayOnline();
      return;
    }
    onStartGame({ mode, model, simulations, colorChoice });
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />

      {/* Dialog */}
      <div className="relative bg-gray-800 rounded-lg shadow-xl max-w-sm w-full p-6">
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
          <div className="mb-4 text-sm text-gray-400">
            Two players on the same device. Blue moves first.
          </div>
        )}

        {/* Online info */}
        {mode === 'online' && (
          <div className="mb-4 text-sm text-gray-400">
            Create or join an online game. Requires login.
          </div>
        )}

        {/* Buttons */}
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
            {mode === 'online' ? 'Go to Lobby' : 'Start Game'}
          </button>
        </div>
      </div>
    </div>
  );
}
