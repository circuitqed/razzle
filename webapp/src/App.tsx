import { useEffect, useState } from 'react';
import Board from './components/Board';
import MoveHistory from './components/MoveHistory';
import ConfirmDialog from './components/ConfirmDialog';
import RulesModal from './components/RulesModal';
import TrainingDashboard from './components/TrainingDashboard';
import LoginModal from './components/LoginModal';
import RegisterModal from './components/RegisterModal';
import UserMenu from './components/UserMenu';
import GameBrowser from './components/GameBrowser';
import ReplayViewer from './components/ReplayViewer';
import AnalysisBoard from './components/AnalysisBoard';
import { useGame } from './hooks/useGame';
import { setSoundEnabled, isSoundEnabled } from './utils/sounds';
import { healthCheck, listModels, type ModelInfo } from './api/engine';
import { AuthProvider } from './contexts/AuthContext';

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

type GameMode = 'ai' | 'pvp';

function AppContent() {
  const [gameMode, setGameMode] = useState<GameMode>('ai');
  const [flipBoard, setFlipBoard] = useState(false);
  const [soundOn, setSoundOn] = useState(isSoundEnabled());
  const [showNewGameConfirm, setShowNewGameConfirm] = useState(false);
  const [showRules, setShowRules] = useState(false);
  const [showTraining, setShowTraining] = useState(false);
  const [currentModel, setCurrentModel] = useState<string | null>(null);

  // AI settings
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | undefined>(undefined); // undefined = latest
  const [selectedSimulations, setSelectedSimulations] = useState(800);

  // Auth modals
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [showRegisterModal, setShowRegisterModal] = useState(false);

  // Game browser and replay
  const [showGameBrowser, setShowGameBrowser] = useState(false);
  const [replayGameId, setReplayGameId] = useState<string | null>(null);

  // Analysis board
  const [showAnalysisBoard, setShowAnalysisBoard] = useState(false);

  const toggleSound = () => {
    const newValue = !soundOn;
    setSoundOn(newValue);
    setSoundEnabled(newValue);
  };

  const {
    gameState,
    selectedSquare,
    isLoading,
    error,
    aiThinking,
    canEndTurn,
    mustPass,
    lastMove,
    rawMoves,
    startNewGame,
    handleSquareClick,
    handleDragMove,
    endTurn,
    undoMove,
  } = useGame({ vsAI: gameMode === 'ai', aiSimulations: selectedSimulations, aiModel: selectedModel });

  // Fetch model info and available models from server
  const fetchModelInfo = async () => {
    try {
      // Fetch available models
      const modelsResponse = await listModels();
      setAvailableModels(modelsResponse.models);

      // Set current model display name
      if (modelsResponse.current) {
        const modelName = modelsResponse.current.split('/').pop() || modelsResponse.current;
        setCurrentModel(modelName);
      }
    } catch (e) {
      console.error('Failed to fetch model info:', e);
      // Fallback to health check
      try {
        const health = await healthCheck();
        if (health.model) {
          const modelName = health.model.split('/').pop() || health.model;
          setCurrentModel(modelName);
        }
      } catch {
        // ignore
      }
    }
  };

  // Start a game on mount and when mode changes
  useEffect(() => {
    startNewGame();
    fetchModelInfo();
  }, [gameMode]);

  // Auto-flip board in PvP mode based on current player
  const shouldFlip = gameMode === 'pvp' && flipBoard && gameState?.current_player === 1;

  // Get winner text based on mode
  const getWinnerText = () => {
    if (!gameState || gameState.winner === null) return '';
    if (gameMode === 'ai') {
      return gameState.winner === 0 ? 'You Win!' : 'AI Wins!';
    }
    return gameState.winner === 0 ? 'Blue Wins!' : 'Red Wins!';
  };

  // Handle mode change - start new game with new mode
  const handleModeChange = (mode: GameMode) => {
    if (mode !== gameMode) {
      setGameMode(mode);
      // Game will restart on next render with new mode
    }
  };

  // Start new game with current mode
  const handleNewGame = () => {
    // Show confirmation if game is in progress (more than 1 ply)
    if (gameState && gameState.status === 'playing' && gameState.ply > 0) {
      setShowNewGameConfirm(true);
    } else {
      startNewGame();
      fetchModelInfo();  // Check for new model on new game
    }
  };

  const confirmNewGame = () => {
    setShowNewGameConfirm(false);
    startNewGame();
    fetchModelInfo();  // Check for new model on new game
  };

  const cancelNewGame = () => {
    setShowNewGameConfirm(false);
  };

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger shortcuts when typing in input fields
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      // Handle modals
      if (showNewGameConfirm) {
        if (e.key === 'Escape') {
          cancelNewGame();
        } else if (e.key === 'Enter') {
          confirmNewGame();
        }
        return;
      }

      if (showRules) {
        if (e.key === 'Escape') {
          setShowRules(false);
        }
        return;
      }

      switch (e.key.toLowerCase()) {
        case 'n':
          handleNewGame();
          break;
        case 'u':
          if (gameState && gameState.ply > 0 && !isLoading && !aiThinking) {
            undoMove();
          }
          break;
        case 'e':
          if (canEndTurn && !isLoading && !aiThinking) {
            endTurn();
          }
          break;
        case 'm':
          toggleSound();
          break;
        case 'escape':
          // Close any open modals
          setShowGameBrowser(false);
          setReplayGameId(null);
          setShowAnalysisBoard(false);
          break;
        case '?':
        case '/':
          setShowRules(true);
          break;
        case 't':
          setShowTraining((prev) => !prev);
          break;
        case 'b':
          setShowGameBrowser((prev) => !prev);
          break;
        case 'a':
          setShowAnalysisBoard((prev) => !prev);
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [gameState, isLoading, aiThinking, canEndTurn, showNewGameConfirm, showRules, handleNewGame, undoMove, endTurn, toggleSound]);

  const handleSelectGameForReplay = (gameId: string) => {
    setShowGameBrowser(false);
    setReplayGameId(gameId);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-2 sm:p-4">
      {/* Header with user menu */}
      <div className="absolute top-4 right-4">
        <UserMenu
          onOpenLogin={() => setShowLoginModal(true)}
          onOpenRegister={() => setShowRegisterModal(true)}
          onOpenBrowser={() => setShowGameBrowser(true)}
        />
      </div>

      <h1 className="text-2xl sm:text-3xl font-bold mb-3 sm:mb-4">Razzle Dazzle</h1>

      {/* Game mode selector */}
      <div className="mb-4 flex gap-2">
        <button
          onClick={() => handleModeChange('ai')}
          className={`px-4 py-2 rounded font-medium transition-colors ${
            gameMode === 'ai'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          vs AI
        </button>
        <button
          onClick={() => handleModeChange('pvp')}
          className={`px-4 py-2 rounded font-medium transition-colors ${
            gameMode === 'pvp'
              ? 'bg-purple-600 text-white'
              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
          }`}
        >
          2 Player
        </button>
      </div>

      {/* AI settings */}
      {gameMode === 'ai' && (
        <div className="mb-4 flex flex-wrap gap-4 items-end justify-center">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Model</label>
            <select
              value={selectedModel || ''}
              onChange={(e) => setSelectedModel(e.target.value || undefined)}
              className="px-3 py-1.5 bg-gray-700 text-white rounded border border-gray-600 text-sm min-w-[140px]"
            >
              <option value="">Latest</option>
              {availableModels.map((model) => (
                <option key={model.path} value={model.path}>
                  {model.name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Simulations</label>
            <select
              value={selectedSimulations}
              onChange={(e) => setSelectedSimulations(Number(e.target.value))}
              className="px-3 py-1.5 bg-gray-700 text-white rounded border border-gray-600 text-sm"
            >
              {SIMULATION_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      )}

      {/* PvP options */}
      {gameMode === 'pvp' && (
        <div className="mb-4">
          <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
            <input
              type="checkbox"
              checked={flipBoard}
              onChange={(e) => setFlipBoard(e.target.checked)}
              className="rounded"
            />
            Flip board for Red's turn
          </label>
        </div>
      )}

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
          {/* Turn indicator */}
          <div className="mb-2 text-center">
            <span
              className={`inline-block px-3 py-1 rounded text-white text-sm font-medium ${
                gameState.current_player === 0 ? 'bg-blue-500' : 'bg-red-500'
              }`}
            >
              {gameState.current_player === 0 ? 'Blue' : 'Red'}'s Turn
            </span>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 items-center sm:items-start w-full max-w-md sm:max-w-none sm:w-auto">
            <Board
              board={gameState.board}
              currentPlayer={gameState.current_player}
              legalMoves={gameState.legal_moves}
              selectedSquare={selectedSquare}
              onSquareClick={handleSquareClick}
              onDragMove={handleDragMove}
              touchedMask={gameState.touched_mask}
              mustPass={mustPass}
              flipped={shouldFlip}
              lastMove={lastMove}
            />
            <div className="hidden sm:block">
              <MoveHistory moves={rawMoves} />
            </div>
          </div>

          {/* Game status - fixed height to prevent layout shift */}
          <div className="mt-4 text-center h-8 flex items-center justify-center">
            {gameState.status === 'finished' && gameState.winner !== null && (
              <div className="text-2xl font-bold text-yellow-400">
                {getWinnerText()}
              </div>
            )}
            {(gameState.status === 'draw' || (gameState.status === 'finished' && gameState.winner === null)) && (
              <div className="text-2xl font-bold text-gray-400">
                Draw!
              </div>
            )}
            {aiThinking && (
              <div className="text-blue-400 animate-pulse">
                AI is thinking...
              </div>
            )}
            {mustPass && !aiThinking && gameState.status === 'playing' && (
              <div className="text-yellow-400 animate-pulse">
                Forced to pass! Opponent moved adjacent to your ball.
              </div>
            )}
          </div>

          {/* Controls */}
          <div className="mt-4 flex flex-wrap justify-center gap-2 sm:gap-4">
            <button
              onClick={handleNewGame}
              disabled={isLoading}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded font-medium transition-colors"
            >
              New Game
            </button>
            {canEndTurn && (
              <button
                onClick={endTurn}
                disabled={isLoading || aiThinking}
                className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded font-medium transition-colors animate-pulse"
              >
                End Turn
              </button>
            )}
            <button
              onClick={undoMove}
              disabled={isLoading || aiThinking || gameState.ply === 0}
              className="px-4 py-2 bg-gray-600 hover:bg-gray-700 disabled:bg-gray-800 disabled:text-gray-500 rounded font-medium transition-colors"
            >
              Undo
            </button>
            <button
              onClick={toggleSound}
              className="px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded font-medium transition-colors"
              title={soundOn ? 'Mute sounds' : 'Enable sounds'}
            >
              {soundOn ? '🔊' : '🔇'}
            </button>
            <button
              onClick={() => setShowRules(true)}
              className="px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded font-medium transition-colors"
              title="Show rules"
            >
              ?
            </button>
            <button
              onClick={() => setShowTraining(true)}
              className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded font-medium transition-colors"
              title="Training Dashboard"
            >
              📊
            </button>
            <button
              onClick={() => setShowGameBrowser(true)}
              className="px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded font-medium transition-colors"
              title="Game History (B)"
            >
              📜
            </button>
            <button
              onClick={() => setShowAnalysisBoard(true)}
              className="px-4 py-2 bg-orange-600 hover:bg-orange-700 rounded font-medium transition-colors"
              title="Analysis Board (A)"
            >
              🔬
            </button>
          </div>

          {/* Game info */}
          <div className="mt-4 text-sm text-gray-400 text-center">
            <span>Ply: {gameState.ply}</span>
            {gameMode === 'ai' && (
              <span className="ml-4 text-gray-500">
                {selectedModel ? selectedModel.split('/').pop() : (currentModel || 'Latest')} @ {selectedSimulations} sims
              </span>
            )}
          </div>
        </>
      )}

      {/* Instructions */}
      <div className="mt-8 text-sm text-gray-500 max-w-md text-center">
        <p>Click a piece to select it, then click a highlighted square to move or pass.</p>
        <p className="mt-1">After passing, click "End Turn" to finish your turn.</p>
        <p className="mt-1">Blue aims for the top, Red aims for the bottom.</p>
        <p className="mt-2 text-xs hidden sm:block">
          Shortcuts: N=New Game, U=Undo, E=End Turn, M=Mute, B=Browse, A=Analysis
        </p>
      </div>

      {/* Confirm New Game Dialog */}
      <ConfirmDialog
        isOpen={showNewGameConfirm}
        title="Start New Game?"
        message="Your current game will be lost. Are you sure you want to start a new game?"
        confirmText="New Game"
        cancelText="Cancel"
        onConfirm={confirmNewGame}
        onCancel={cancelNewGame}
      />

      {/* Rules Modal */}
      <RulesModal isOpen={showRules} onClose={() => setShowRules(false)} />

      {/* Training Dashboard */}
      {showTraining && (
        <TrainingDashboard onClose={() => setShowTraining(false)} refreshInterval={10000} />
      )}

      {/* Auth Modals */}
      <LoginModal
        isOpen={showLoginModal}
        onClose={() => setShowLoginModal(false)}
        onSwitchToRegister={() => {
          setShowLoginModal(false);
          setShowRegisterModal(true);
        }}
      />
      <RegisterModal
        isOpen={showRegisterModal}
        onClose={() => setShowRegisterModal(false)}
        onSwitchToLogin={() => {
          setShowRegisterModal(false);
          setShowLoginModal(true);
        }}
      />

      {/* Game Browser */}
      <GameBrowser
        isOpen={showGameBrowser}
        onClose={() => setShowGameBrowser(false)}
        onSelectGame={handleSelectGameForReplay}
      />

      {/* Replay Viewer */}
      {replayGameId && (
        <ReplayViewer
          gameId={replayGameId}
          onClose={() => setReplayGameId(null)}
        />
      )}

      {/* Analysis Board */}
      <AnalysisBoard
        isOpen={showAnalysisBoard}
        onClose={() => setShowAnalysisBoard(false)}
      />
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}
