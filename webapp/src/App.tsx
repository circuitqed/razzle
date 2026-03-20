import { Component, Suspense, lazy, useEffect, useState, useCallback, useMemo, useRef } from 'react';
import type { ReactNode, ErrorInfo } from 'react';
import { BrowserRouter, Routes, Route, useParams, useNavigate, useLocation } from 'react-router-dom';
import { GoogleOAuthProvider } from '@react-oauth/google';
import BugReportDialog from './components/BugReportDialog';
import GameView from './components/GameView';
import RulesModal from './components/RulesModal';
import LoginModal from './components/LoginModal';
import RegisterModal from './components/RegisterModal';
import ForgotPasswordModal from './components/ForgotPasswordModal';
import UsernamePickerModal from './components/UsernamePickerModal';
import EmailVerificationBanner from './components/EmailVerificationBanner';
import VerifyEmailPage from './components/VerifyEmailPage';
import ResetPasswordPage from './components/ResetPasswordPage';
import GoogleCallbackPage from './components/GoogleCallbackPage';
import UserMenu from './components/UserMenu';
import GameBrowser from './components/GameBrowser';
import ReplayViewer from './components/ReplayViewer';
import OpeningExplorer from './components/OpeningExplorer';
import WaitingForOpponent from './components/WaitingForOpponent';
import OnlineGame from './components/OnlineGame';
import { TermsPage, PrivacyPage } from './components/LegalPages';
import AboutPage from './components/AboutPage';
import NewGameDialog from './components/NewGameDialog';
import type { NewGameSettings } from './components/NewGameDialog';
import { BOT_PRESETS } from './components/NewGameDialog';
import { useGame } from './hooks/useGame';
import { setSoundEnabled, isSoundEnabled } from './utils/sounds';
import { adjustAfterGame, getAutoMatchLevel, getLevelLabel, getTierSettings } from './utils/autoMatch';
import { listModels, type ModelInfo } from './api/engine';
import * as onlineApi from './api/online';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import LandingPage from './components/LandingPage';
import Tutorial from './components/Tutorial';

// Lazy-loaded heavy routes
const TrainingDashboard = lazy(() => import('./components/TrainingDashboard'));
const AnalysisBoard = lazy(() => import('./components/AnalysisBoard'));

const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID || '';

const GAME_STORAGE_KEY = 'knightball_current_game';
const HAS_PLAYED_KEY = 'knightball_has_played';

function readSavedGame(): { gameId: string; settings: NewGameSettings; playerColor: number } | null {
  try {
    const json = localStorage.getItem(GAME_STORAGE_KEY);
    if (json) return JSON.parse(json);
  } catch { /* ignore corrupt data */ }
  return null;
}

function AppContent() {
  // Restore saved game state if available
  const savedGame = useMemo(() => readSavedGame(), []);

  // Landing page & tutorial state
  const [showLanding, setShowLanding] = useState(() => {
    try { return !localStorage.getItem(HAS_PLAYED_KEY); } catch { return false; }
  });
  const [showTutorial, setShowTutorial] = useState(false);

  // Game settings (persisted across new games)
  const [settings, setSettings] = useState<NewGameSettings>(() => {
    if (savedGame?.settings) return savedGame.settings;
    // Resolve auto-match tier for initial default
    const tier = getTierSettings(getAutoMatchLevel());
    return {
      mode: 'ai' as const,
      model: tier.model,
      simulations: tier.sims,
      colorChoice: 'random' as const,
      difficulty: 'auto',
    };
  });
  const [playerColor, setPlayerColor] = useState(savedGame?.playerColor ?? 0);
  const [gameGeneration, setGameGeneration] = useState(0); // Bumped to force new game

  const [flipBoard, setFlipBoard] = useState(false);
  const [soundOn, setSoundOn] = useState(isSoundEnabled());
  const [showRules, setShowRules] = useState(false);
  const [showBugReport, setShowBugReport] = useState(false);
  const [showNewGameDialog, setShowNewGameDialog] = useState(false);

  // Available models from server
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);

  // Auto-match level toast
  const [levelToast, setLevelToast] = useState<string | null>(null);

  // Auth modals
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [showRegisterModal, setShowRegisterModal] = useState(false);
  const [showForgotPassword, setShowForgotPassword] = useState(false);
  const [usernamePickerData, setUsernamePickerData] = useState<{
    tempToken: string;
    suggestedName: string;
  } | null>(null);

  // Game browser and replay
  const [showGameBrowser, setShowGameBrowser] = useState(false);
  const [replayGameId, setReplayGameId] = useState<string | null>(null);

  // Analysis board
  const [showAnalysisBoard, setShowAnalysisBoard] = useState(false);

  // Training dashboard
  const [showTrainingDashboard, setShowTrainingDashboard] = useState(false);

  // Online multiplayer
  const [waitingGame, setWaitingGame] = useState<{
    gameId: string;
    joinCode: string;
    hostColor: number;
  } | null>(null);

  const navigate = useNavigate();
  const location = useLocation();
  const { user } = useAuth();

  // If arriving from an online game, don't resume saved AI game
  const fromOnline = (location.state as { fromOnline?: boolean })?.fromOnline === true;

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
    aiModelLoading,
    canEndTurn,
    mustPass,
    isPassing,
    lastMove,
    lastTurnAnimMoves,
    rawMoves,
    evaluation,
    viewPly,
    isViewingHistory,
    startNewGame,
    resumeGame,
    handleSquareClick,
    handleDragMove,
    endTurn,
    undoMove,
    resign,
    cancelPass,
    goForward,
    goBack,
    goToStart,
    goToEnd,
  } = useGame({
    vsAI: settings.mode === 'ai',
    aiSimulations: settings.simulations,
    aiModel: settings.model,
    playerColor,
  });

  // Fetch available models from server
  const fetchModels = useCallback(async () => {
    try {
      const modelsResponse = await listModels();
      setAvailableModels(modelsResponse.models);
    } catch (e) {
      console.error('Failed to fetch models:', e);
    }
  }, []);

  // On mount: resume saved game or start new (skip if landing page is showing)
  useEffect(() => {
    if (showLanding) return;
    if (savedGame && !fromOnline) {
      resumeGame(savedGame.gameId).then(success => {
        if (!success) {
          localStorage.removeItem(GAME_STORAGE_KEY);
          startNewGame();
        }
      });
    } else {
      if (fromOnline) {
        localStorage.removeItem(GAME_STORAGE_KEY);
      }
      startNewGame();
    }
    fetchModels();
  }, [showLanding]);

  // Persist active game to localStorage so it survives refresh
  useEffect(() => {
    if (!gameState) return;
    if (gameState.status === 'playing' && gameState.ply > 0) {
      localStorage.setItem(GAME_STORAGE_KEY, JSON.stringify({
        gameId: gameState.game_id,
        settings,
        playerColor,
      }));
    } else if (gameState.status === 'finished') {
      localStorage.removeItem(GAME_STORAGE_KEY);
    }
  }, [gameState?.game_id, gameState?.status, gameState?.ply, settings, playerColor]);

  // Adjust auto-match level when an AI game finishes
  const prevGameStatusRef = useRef<string | null>(null);
  useEffect(() => {
    if (!gameState) return;
    const wasPlaying = prevGameStatusRef.current === 'playing';
    prevGameStatusRef.current = gameState.status;

    if (
      wasPlaying &&
      gameState.status === 'finished' &&
      gameState.winner !== null &&
      settings.mode === 'ai' &&
      settings.difficulty === 'auto'
    ) {
      const won = gameState.winner === playerColor;
      const oldLevel = getAutoMatchLevel();
      const newLevel = adjustAfterGame(won);
      if (newLevel !== oldLevel) {
        const label = getLevelLabel(newLevel);
        if (won) {
          setLevelToast(`Level up! Now at ${label}`);
        } else {
          setLevelToast(`Dropped to ${label}`);
        }
      }
    }
  }, [gameState?.status, gameState?.winner, settings.mode, settings.difficulty, playerColor]);

  // Auto-dismiss level toast
  useEffect(() => {
    if (!levelToast) return;
    const timer = setTimeout(() => setLevelToast(null), 3000);
    return () => clearTimeout(timer);
  }, [levelToast]);

  // Resolve color choice for new games
  const resolveColor = useCallback((choice: NewGameSettings['colorChoice']) => {
    if (choice === 'random') return Math.random() < 0.5 ? 0 : 1;
    return choice === 'blue' ? 0 : 1;
  }, []);

  // Handle new game from dialog
  const handleStartGame = useCallback((newSettings: NewGameSettings) => {
    localStorage.removeItem(GAME_STORAGE_KEY);
    setSettings(newSettings);
    setShowNewGameDialog(false);
    const color = newSettings.mode === 'ai' ? resolveColor(newSettings.colorChoice) : 0;
    setPlayerColor(color);
    setGameGeneration(g => g + 1); // Force new game even if settings unchanged
  }, [resolveColor]);

  // Re-start game when generation changes (after dialog)
  const initialMountRef = useRef(true);
  useEffect(() => {
    if (initialMountRef.current) {
      initialMountRef.current = false;
      return; // Skip initial mount (handled by the mount effect above)
    }
    startNewGame();
  }, [gameGeneration]);

  // Handle new game button
  const handleNewGameClick = () => {
    if (gameState && gameState.status === 'playing' && gameState.ply > 0) {
      // Game in progress - show dialog (which doubles as confirmation)
      setShowNewGameDialog(true);
    } else {
      setShowNewGameDialog(true);
    }
  };

  // Quick new game (same settings, new random color if applicable)
  const handleQuickNewGame = useCallback(() => {
    localStorage.removeItem(GAME_STORAGE_KEY);
    if (settings.mode === 'ai') {
      setPlayerColor(resolveColor(settings.colorChoice));
      // Re-resolve auto-match tier in case level changed
      if (settings.difficulty === 'auto') {
        const tier = getTierSettings(getAutoMatchLevel());
        setSettings(prev => ({ ...prev, model: tier.model, simulations: tier.sims }));
      }
    }
    setGameGeneration(g => g + 1);
    fetchModels();
  }, [settings, resolveColor, fetchModels]);

  // Board flipping: default is player looking up
  const shouldFlipBoard = useMemo(() => {
    if (settings.mode === 'pvp') {
      // In PvP, flip when it's red's turn if auto-flip is on
      return flipBoard && gameState?.current_player === 1;
    }
    // In AI mode, flip if player is red or manual flip
    const baseFlip = playerColor === 1;
    return flipBoard ? !baseFlip : baseFlip;
  }, [settings.mode, flipBoard, gameState?.current_player, playerColor]);

  // Player names
  const playerName = useMemo(() => {
    if (settings.mode === 'ai') {
      return user?.username || user?.display_name || 'You';
    }
    return 'Blue';
  }, [settings.mode, user]);

  const opponentName = useMemo(() => {
    if (settings.mode === 'ai') {
      // Auto-match: show current level
      if (settings.difficulty === 'auto') {
        return `AI (${getLevelLabel(getAutoMatchLevel())})`;
      }
      // Use friendly difficulty name if available
      if (settings.difficulty && settings.difficulty !== 'custom') {
        const preset = BOT_PRESETS.find(p => p.id === settings.difficulty);
        if (preset) return `AI (${preset.name})`;
      }
      // Custom mode: show model + sims
      let modelName: string;
      if (settings.model) {
        modelName = settings.model.split('/').pop()?.replace('.pt', '') || 'AI';
      } else {
        const latest = availableModels.find(m => m.name !== 'random_weights');
        modelName = latest ? latest.name.replace('.pt', '') : 'AI';
      }
      return `${modelName} - ${settings.simulations} sims`;
    }
    return 'Red';
  }, [settings.mode, settings.model, settings.simulations, settings.difficulty, availableModels]);

  // Labels positioned relative to board orientation
  // In AI mode, human is always at the bottom (shouldFlipBoard ensures this)
  // In PvP mode, names follow the flip
  const bottomName = settings.mode === 'ai' ? playerName : (shouldFlipBoard ? 'Red' : 'Blue');
  const topName = settings.mode === 'ai' ? opponentName : (shouldFlipBoard ? 'Blue' : 'Red');

  // Winner text + color
  const winnerDisplay = useMemo(() => {
    if (!gameState || gameState.winner === null) return null;
    const colorClass = gameState.winner === 0 ? 'text-blue-400' : 'text-red-400';
    if (settings.mode === 'ai') {
      const text = gameState.winner === playerColor
        ? 'You Win!'
        : `${opponentName} Wins!`;
      return { text, colorClass };
    }
    const text = gameState.winner === 0 ? 'Blue Wins!' : 'Red Wins!';
    return { text, colorClass };
  }, [gameState, settings.mode, playerColor, opponentName]);

  // Turn indicator text (consolidated with AI thinking)
  const turnIndicator = useMemo(() => {
    if (!gameState) return null;
    if (gameState.status === 'finished') return null;

    const isPlayerTurn = settings.mode !== 'ai' || gameState.current_player === playerColor;
    const color = gameState.current_player === 0 ? 'Blue' : 'Red';
    const colorClass = gameState.current_player === 0 ? 'bg-blue-500' : 'bg-red-500';

    let text = `${color}'s Turn`;
    if (settings.mode === 'ai') {
      text = isPlayerTurn ? 'Your Turn' : `${opponentName}`;
    }

    return { text, colorClass, isThinking: aiThinking };
  }, [gameState, settings.mode, playerColor, opponentName, aiThinking]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      if (showNewGameDialog || showRules) {
        if (e.key === 'Escape') {
          setShowNewGameDialog(false);
          setShowRules(false);
        }
        return;
      }

      switch (e.key.toLowerCase()) {
        case 'n':
          handleNewGameClick();
          break;
        case 'u':
          if (gameState && gameState.ply > 0 && !isLoading && !aiThinking) undoMove();
          break;
        case 'e':
          if (canEndTurn && !isLoading && !aiThinking) endTurn();
          break;
        case 'm':
          toggleSound();
          break;
        case 'f':
          setFlipBoard(f => !f);
          break;
        case 'arrowleft':
          e.preventDefault();
          goBack();
          break;
        case 'arrowright':
          e.preventDefault();
          goForward();
          break;
        case 'home':
          e.preventDefault();
          goToStart();
          break;
        case 'end':
          e.preventDefault();
          goToEnd();
          break;
        case 'escape':
          setShowGameBrowser(false);
          setReplayGameId(null);
          setShowAnalysisBoard(false);
          setShowTrainingDashboard(false);
          if (isViewingHistory) goToEnd();
          break;
        case '?':
        case '/':
          setShowRules(true);
          break;
        case 'b':
          setShowGameBrowser(prev => !prev);
          break;
        case 't':
          setShowTrainingDashboard(prev => !prev);
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [gameState, isLoading, aiThinking, canEndTurn, showNewGameDialog, showRules, isViewingHistory]);

  // Tab title: show "(Your Turn)" when it's the human's turn in AI mode
  useEffect(() => {
    const isPlayerTurn =
      gameState?.status === 'playing' &&
      settings.mode === 'ai' &&
      gameState.current_player === playerColor;
    document.title = isPlayerTurn ? '(Your Turn) KnightBall' : 'KnightBall';

    const handleVisibility = () => {
      if (document.visibilityState === 'visible') {
        document.title = 'KnightBall';
      }
    };
    document.addEventListener('visibilitychange', handleVisibility);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibility);
      document.title = 'KnightBall';
    };
  }, [gameState?.status, gameState?.current_player, settings.mode, playerColor]);

  const handleSelectGameForReplay = (gameId: string) => {
    setShowGameBrowser(false);
    setReplayGameId(gameId);
  };

  // Online multiplayer handlers
  const handleGameCreated = (gameId: string, joinCode: string, hostColor: number) => {
    setShowNewGameDialog(false);
    setWaitingGame({ gameId, joinCode, hostColor });
  };

  const handleGameJoined = (gameId: string, _yourColor: number) => {
    setShowNewGameDialog(false);
    setWaitingGame(null);
    navigate(`/online/${gameId}`);
  };

  const handleCancelWaiting = async () => {
    if (waitingGame) {
      try {
        await onlineApi.leaveOnlineGame(waitingGame.gameId);
      } catch (err) {
        console.error('Failed to cancel game:', err);
      }
    }
    setWaitingGame(null);
  };

  // Landing page handlers
  const handlePlayNow = useCallback(() => {
    try { localStorage.setItem(HAS_PLAYED_KEY, 'true'); } catch { /* ignore */ }
    setShowLanding(false);
    // The mount useEffect will fire when showLanding becomes false
  }, []);

  const handleStartTutorial = useCallback(() => {
    try { localStorage.setItem(HAS_PLAYED_KEY, 'true'); } catch { /* ignore */ }
    setShowLanding(false);
    setShowTutorial(true);
  }, []);

  const handleTutorialDone = useCallback(() => {
    setShowTutorial(false);
    // If no game was started yet (edge case), start one now.
    if (!gameState) {
      startNewGame();
      fetchModels();
    }
  }, [gameState, startNewGame, fetchModels]);

  // WebSocket for waiting game
  useEffect(() => {
    if (!waitingGame) return;
    const ws = onlineApi.connectOnlineGameWebSocket(waitingGame.gameId, {
      onPlayerJoined: () => {
        setWaitingGame(null);
        navigate(`/online/${waitingGame.gameId}`);
      },
      onError: (error) => console.error('WebSocket error:', error),
      onClose: () => {},
    });
    return () => { ws.close(); };
  }, [waitingGame, navigate]);

  // Show landing page for first-time visitors
  if (showLanding) {
    return <LandingPage onPlayNow={handlePlayNow} onTutorial={handleStartTutorial} />;
  }

  return (
    <div className="h-[100dvh] bg-gray-900 text-white flex flex-col overflow-hidden">
      {/* Tutorial overlay */}
      {showTutorial && (
        <Tutorial onComplete={handleTutorialDone} onSkip={handleTutorialDone} />
      )}

      {/* Email verification banner */}
      <EmailVerificationBanner />

      {/* Header bar */}
      <header className="flex items-center justify-end px-4 py-2 shrink-0">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowGameBrowser(true)}
            className="p-1.5 text-gray-400 hover:text-white transition-colors"
            title="Game history (B)"
          >
{'\u{1F4CB}'}
          </button>
          <button
            onClick={() => setShowBugReport(true)}
            className="p-1.5 text-gray-400 hover:text-white transition-colors"
            title="Report a bug"
          >
{'\u{1F41B}'}
          </button>
          <button
            onClick={() => setShowTutorial(true)}
            className="p-1.5 text-gray-400 hover:text-white transition-colors"
            title="How to Play"
          >
{'\u{1F393}'}
          </button>
          <UserMenu
            onOpenLogin={() => setShowLoginModal(true)}
            onOpenRegister={() => setShowRegisterModal(true)}
            onOpenBrowser={() => setShowGameBrowser(true)}
          />
        </div>
      </header>

      <main className="flex-1 min-h-0 flex flex-col items-center p-2 sm:p-4 pb-4 overflow-y-auto">
        <GameView
          gameState={gameState}
          selectedSquare={selectedSquare}
          isLoading={isLoading}
          error={error}
          canEndTurn={canEndTurn}
          mustPass={mustPass}
          isPassing={isPassing}
          lastMove={lastMove}
          lastTurnMoves={lastTurnAnimMoves}
          rawMoves={rawMoves}
          viewPly={viewPly}
          isViewingHistory={isViewingHistory}
          handleSquareClick={handleSquareClick}
          handleDragMove={handleDragMove}
          endTurn={endTurn}
          goForward={goForward}
          goBack={goBack}
          goToStart={goToStart}
          goToEnd={goToEnd}
          flipped={shouldFlipBoard}
          topName={topName}
          bottomName={bottomName}
          cancelPass={cancelPass}
          evaluation={settings.mode === 'ai' ? evaluation : undefined}
          playerColor={playerColor}
          aiThinking={aiThinking}
          statusLine={
            <>
              {gameState?.status === 'finished' && winnerDisplay && (
                <span className={`text-xl font-bold ${winnerDisplay.colorClass}`}>{winnerDisplay.text}</span>
              )}
              {turnIndicator && gameState?.status === 'playing' && (
                <>
                  <span className={`inline-block px-3 py-0.5 rounded text-white text-sm font-medium ${turnIndicator.colorClass}`}>
                    {turnIndicator.text}
                  </span>
                  {turnIndicator.isThinking && (
                    <span className="text-blue-400 text-sm animate-pulse">thinking...</span>
                  )}
                  {aiModelLoading && !turnIndicator.isThinking && (
                    <span className="text-yellow-400 text-xs animate-pulse">loading model...</span>
                  )}
                  {mustPass && !aiThinking && (
                    <span className="text-yellow-400 text-xs">forced pass</span>
                  )}
                </>
              )}
            </>
          }
          extraActions={
            <>
              {/* New Game */}
              {gameState && (
                <button
                  onClick={gameState.status === 'finished' ? handleQuickNewGame : handleNewGameClick}
                  disabled={isLoading}
                  className="px-3 py-2 sm:px-4 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded font-medium transition-colors text-sm sm:text-base"
                >
                  New Game
                </button>
              )}

              {/* Resign - only during active game, no confirmation needed vs AI */}
              {gameState && gameState.ply > 0 && gameState.status === 'playing' && !isPassing && !isViewingHistory && (
                <button
                  onClick={() => resign()}
                  disabled={isLoading}
                  className="px-3 py-2 sm:px-4 bg-gray-600 hover:bg-red-700 text-gray-300 hover:text-white disabled:bg-gray-600 rounded font-medium transition-colors text-sm sm:text-base"
                >
                  Resign
                </button>
              )}

              {/* Undo - not during pass */}
              {gameState && settings.mode === 'ai' && !isPassing && !isViewingHistory && (
                <button
                  onClick={undoMove}
                  disabled={isLoading || aiThinking || gameState.ply === 0}
                  className="px-3 py-2 sm:px-4 bg-gray-600 hover:bg-gray-700 disabled:bg-gray-800 disabled:text-gray-500 rounded font-medium transition-colors text-sm sm:text-base"
                >
                  Undo
                </button>
              )}

              {/* Sound toggle */}
              <button
                onClick={toggleSound}
                className="px-3 py-2 bg-gray-600 hover:bg-gray-700 rounded font-medium transition-colors text-sm"
                title={soundOn ? 'Mute (M)' : 'Unmute (M)'}
              >
                {soundOn ? '\u{1F50A}' : '\u{1F507}'}
              </button>

              {/* Flip board */}
              <button
                onClick={() => setFlipBoard(f => !f)}
                className="px-3 py-2 bg-gray-600 hover:bg-gray-700 rounded font-medium transition-colors text-sm"
                title="Flip board (F)"
              >
                {'\u{21C5}'}
              </button>

              {/* Rules */}
              <button
                onClick={() => setShowRules(true)}
                className="px-3 py-2 bg-gray-600 hover:bg-gray-700 rounded font-medium transition-colors text-sm"
                title="Rules (?)"
              >
                ?
              </button>

            </>
          }
        />

        {/* Instructions - collapsed on mobile */}
        <div className="mt-4 text-xs text-gray-600 max-w-md text-center hidden sm:block">
          <p>Click a piece to select, then click a destination. After passing, click "Complete Pass".</p>
          <p className="mt-1">
            Keys: N=New Game, U=Undo, E=End Turn, F=Flip, M=Mute, ?=Rules, {'\u{2190}\u{2192}'}=History
          </p>
        </div>
        <div className="mt-2 text-xs text-gray-700">
          <a href="/about" className="hover:text-gray-500">About</a>
          {' \u00B7 '}
          <a href="/terms" className="hover:text-gray-500">Terms</a>
          {' \u00B7 '}
          <a href="/privacy" className="hover:text-gray-500">Privacy</a>
        </div>
      </main>

      {/* New Game Dialog */}
      <NewGameDialog
        isOpen={showNewGameDialog}
        onClose={() => setShowNewGameDialog(false)}
        onStartGame={handleStartGame}
        onGameCreated={handleGameCreated}
        onGameJoined={handleGameJoined}
        availableModels={availableModels}
        currentSettings={settings}
      />

      {/* Rules Modal */}
      <RulesModal isOpen={showRules} onClose={() => setShowRules(false)} onStartTutorial={() => { setShowRules(false); setShowTutorial(true); }} />

      {/* Bug Report Dialog */}
      <BugReportDialog
        isOpen={showBugReport}
        onClose={() => setShowBugReport(false)}
        gameState={gameState}
        rawMoves={rawMoves}
        gameMode={settings.mode}
        aiModel={settings.model}
      />

      {/* Auth Modals */}
      <LoginModal
        isOpen={showLoginModal}
        onClose={() => setShowLoginModal(false)}
        onSwitchToRegister={() => {
          setShowLoginModal(false);
          setShowRegisterModal(true);
        }}
        onForgotPassword={() => {
          setShowLoginModal(false);
          setShowForgotPassword(true);
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
      <ForgotPasswordModal
        isOpen={showForgotPassword}
        onClose={() => setShowForgotPassword(false)}
        onSwitchToLogin={() => {
          setShowForgotPassword(false);
          setShowLoginModal(true);
        }}
      />
      <UsernamePickerModal
        isOpen={!!usernamePickerData}
        onClose={() => setUsernamePickerData(null)}
        tempToken={usernamePickerData?.tempToken || ''}
        suggestedName={usernamePickerData?.suggestedName || ''}
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

      {/* Training Dashboard */}
      {showTrainingDashboard && (
        <TrainingDashboard
          onClose={() => setShowTrainingDashboard(false)}
          refreshInterval={10000}
        />
      )}

      {/* Waiting for Opponent */}
      {waitingGame && (
        <WaitingForOpponent
          joinCode={waitingGame.joinCode}
          hostColor={waitingGame.hostColor}
          onCancel={handleCancelWaiting}
        />
      )}

      {/* Auto-match level toast */}
      {levelToast && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50">
          <div className="px-4 py-2 bg-gray-800 border border-gray-600 rounded-lg shadow-lg text-sm text-white whitespace-nowrap">
            {levelToast}
          </div>
        </div>
      )}

      {/* Cookie consent banner */}
    </div>
  );
}

// 404 page
function NotFoundPage() {
  const navigate = useNavigate();
  return (
    <div className="min-h-screen bg-gray-900 flex flex-col items-center justify-center text-white">
      <h1 className="text-6xl font-bold mb-4">404</h1>
      <p className="text-gray-400 mb-6">Page not found</p>
      <button
        onClick={() => navigate('/')}
        className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded font-medium transition-colors"
      >
        Go Home
      </button>
    </div>
  );
}

// Standalone Training Dashboard page for /dashboard route (requires login)
function TrainingDashboardPage() {
  const { isAuthenticated, isLoading } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      navigate('/');
    }
  }, [isAuthenticated, isLoading, navigate]);

  if (isLoading || !isAuthenticated) {
    return <div className="min-h-screen bg-gray-900 flex items-center justify-center text-gray-400">Loading...</div>;
  }

  return (
    <TrainingDashboard refreshInterval={10000} />
  );
}

// Online game page wrapper
function OnlineGamePage() {
  const { gameId } = useParams<{ gameId: string }>();
  const navigate = useNavigate();

  if (!gameId) {
    navigate('/');
    return null;
  }

  return (
    <OnlineGame
      gameId={gameId}
      onGameEnd={() => {}}
    />
  );
}

// Join game via code page
function JoinGamePage() {
  const { code } = useParams<{ code: string }>();
  const navigate = useNavigate();
  const [error, setError] = useState<string | null>(null);
  const [isJoining, setIsJoining] = useState(false);

  useEffect(() => {
    if (!code) {
      navigate('/');
      return;
    }

    const joinGame = async () => {
      setIsJoining(true);
      try {
        const result = await onlineApi.joinOnlineGame(code);
        navigate(`/online/${result.game_id}`);
      } catch (err) {
        if (err instanceof onlineApi.OnlineAPIError) {
          setError(err.message);
        } else {
          setError('Failed to join game');
        }
        setTimeout(() => navigate('/'), 3000);
      } finally {
        setIsJoining(false);
      }
    };

    joinGame();
  }, [code, navigate]);

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-4">
      <h1 className="text-2xl font-bold mb-4">KnightBall</h1>
      {isJoining && (
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-green-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p>Joining game {code}...</p>
        </div>
      )}
      {error && (
        <div className="text-center">
          <div className="bg-red-600 px-4 py-2 rounded mb-4">{error}</div>
          <p className="text-gray-400">Redirecting to home...</p>
        </div>
      )}
    </div>
  );
}

// Error boundary to prevent white-screen crashes
class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('[ErrorBoundary]', error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-4">
          <h1 className="text-2xl font-bold mb-4">Something went wrong</h1>
          <p className="text-gray-400 mb-4 text-center max-w-md">{this.state.error.message}</p>
          <button
            onClick={() => { this.setState({ error: null }); window.location.reload(); }}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded font-medium"
          >
            Reload
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

function AppContentWrapper() {
  return <AppContent />;
}

export default function App() {
  return (
    <ErrorBoundary>
      <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
        <BrowserRouter>
          <AuthProvider>
            <Suspense fallback={<div className="min-h-screen bg-gray-900 flex items-center justify-center text-gray-400">Loading...</div>}>
              <Routes>
                <Route path="/" element={<AppContentWrapper />} />
                <Route path="/openings" element={<OpeningExplorer />} />
                <Route path="/dashboard" element={<TrainingDashboardPage />} />
                <Route path="/online/:gameId" element={<OnlineGamePage />} />
                <Route path="/join/:code" element={<JoinGamePage />} />
                <Route path="/auth/google/callback" element={<GoogleCallbackPage />} />
                <Route path="/verify-email" element={<VerifyEmailPage />} />
                <Route path="/reset-password" element={<ResetPasswordPage />} />
                <Route path="/terms" element={<TermsPage />} />
                <Route path="/privacy" element={<PrivacyPage />} />
                <Route path="/about" element={<AboutPage />} />
                <Route path="*" element={<NotFoundPage />} />
              </Routes>
            </Suspense>
          </AuthProvider>
        </BrowserRouter>
      </GoogleOAuthProvider>
    </ErrorBoundary>
  );
}
