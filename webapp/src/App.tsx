import { Component, useEffect, useState, useCallback, useMemo, useRef } from 'react';
import type { ReactNode, ErrorInfo } from 'react';
import { BrowserRouter, Routes, Route, useParams, useNavigate } from 'react-router-dom';
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
import OnlineLobby from './components/OnlineLobby';
import WaitingForOpponent from './components/WaitingForOpponent';
import OnlineGame from './components/OnlineGame';
import EvaluationMeter from './components/EvaluationMeter';
import NewGameDialog from './components/NewGameDialog';
import type { NewGameSettings } from './components/NewGameDialog';
import { useGame } from './hooks/useGame';
import { setSoundEnabled, isSoundEnabled } from './utils/sounds';
import { listModels, type ModelInfo } from './api/engine';
import { formatMovesForDisplay } from './utils/replay';
import * as onlineApi from './api/online';
import { AuthProvider, useAuth } from './contexts/AuthContext';

function AppContent() {
  // Game settings (persisted across new games)
  const [settings, setSettings] = useState<NewGameSettings>({
    mode: 'ai',
    model: undefined,
    simulations: 256,
    colorChoice: 'random',
  });
  const [playerColor, setPlayerColor] = useState(0); // Actual color for current game
  const [gameGeneration, setGameGeneration] = useState(0); // Bumped to force new game

  const [flipBoard, setFlipBoard] = useState(false);
  const [soundOn, setSoundOn] = useState(isSoundEnabled());
  const [showResignConfirm, setShowResignConfirm] = useState(false);
  const [showRules, setShowRules] = useState(false);
  const [showNewGameDialog, setShowNewGameDialog] = useState(false);

  // Available models from server
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);

  // Auth modals
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [showRegisterModal, setShowRegisterModal] = useState(false);

  // Game browser and replay
  const [showGameBrowser, setShowGameBrowser] = useState(false);
  const [replayGameId, setReplayGameId] = useState<string | null>(null);

  // Analysis board
  const [showAnalysisBoard, setShowAnalysisBoard] = useState(false);

  // Training dashboard
  const [showTrainingDashboard, setShowTrainingDashboard] = useState(false);

  // Online multiplayer
  const [showOnlineLobby, setShowOnlineLobby] = useState(false);
  const [waitingGame, setWaitingGame] = useState<{
    gameId: string;
    joinCode: string;
    hostColor: number;
  } | null>(null);

  const navigate = useNavigate();
  const { user } = useAuth();

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
    rawMoves,
    evaluation,
    viewPly,
    isViewingHistory,
    startNewGame,
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

  // Auto-start game on mount
  useEffect(() => {
    startNewGame();
    fetchModels();
  }, []);

  // Resolve color choice for new games
  const resolveColor = useCallback((choice: NewGameSettings['colorChoice']) => {
    if (choice === 'random') return Math.random() < 0.5 ? 0 : 1;
    return choice === 'blue' ? 0 : 1;
  }, []);

  // Handle new game from dialog
  const handleStartGame = useCallback((newSettings: NewGameSettings) => {
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
    if (settings.mode === 'ai') {
      setPlayerColor(resolveColor(settings.colorChoice));
    }
    startNewGame();
    fetchModels();
  }, [settings, resolveColor, startNewGame, fetchModels]);

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
      return user?.display_name || user?.username || 'You';
    }
    return 'Blue';
  }, [settings.mode, user]);

  const opponentName = useMemo(() => {
    if (settings.mode === 'ai') {
      let modelName: string;
      if (settings.model) {
        modelName = settings.model.split('/').pop()?.replace('.pt', '') || 'AI';
      } else {
        // Resolve "latest" to the actual model name from available models
        const latest = availableModels.find(m => m.name !== 'random_weights');
        modelName = latest ? latest.name.replace('.pt', '') : 'AI';
      }
      return `${modelName} - ${settings.simulations} sims`;
    }
    return 'Red';
  }, [settings.mode, settings.model, settings.simulations, availableModels]);

  // Labels positioned relative to board orientation
  // In AI mode, human is always at the bottom (shouldFlipBoard ensures this)
  // In PvP mode, names follow the flip
  const bottomName = settings.mode === 'ai' ? playerName : (shouldFlipBoard ? 'Red' : 'Blue');
  const topName = settings.mode === 'ai' ? opponentName : (shouldFlipBoard ? 'Blue' : 'Red');

  // Winner text
  const getWinnerText = () => {
    if (!gameState || gameState.winner === null) return '';
    if (settings.mode === 'ai') {
      return gameState.winner === playerColor ? 'You Win!' : `${opponentName} Wins!`;
    }
    return gameState.winner === 0 ? 'Blue Wins!' : 'Red Wins!';
  };

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

  // Compact move history for inline display
  const formattedTurns = useMemo(() => {
    return formatMovesForDisplay(rawMoves);
  }, [rawMoves]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      if (showNewGameDialog || showResignConfirm || showRules) {
        if (e.key === 'Escape') {
          setShowNewGameDialog(false);
          setShowResignConfirm(false);
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
  }, [gameState, isLoading, aiThinking, canEndTurn, showNewGameDialog, showResignConfirm, showRules, isViewingHistory]);

  const handleSelectGameForReplay = (gameId: string) => {
    setShowGameBrowser(false);
    setReplayGameId(gameId);
  };

  // Online multiplayer handlers
  const handleGameCreated = (gameId: string, joinCode: string, hostColor: number) => {
    setShowOnlineLobby(false);
    setWaitingGame({ gameId, joinCode, hostColor });
  };

  const handleGameJoined = (gameId: string, _yourColor: number) => {
    setShowOnlineLobby(false);
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

  // WebSocket for waiting game
  useEffect(() => {
    if (!waitingGame) return;
    const ws = onlineApi.connectOnlineGameWebSocket(waitingGame.gameId, {
      onPlayerJoined: () => {
        setWaitingGame(null);
        navigate(`/online/${waitingGame.gameId}`);
      },
      onError: (error) => console.error('WebSocket error:', error),
      onClose: () => console.log('WebSocket closed'),
    });
    return () => { ws.close(); };
  }, [waitingGame, navigate]);

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col">
      {/* Header bar */}
      <header className="flex items-center justify-between px-4 py-2 shrink-0">
        <div className="w-20" /> {/* Spacer for balance */}
        <h1 className="text-xl sm:text-2xl font-bold">Razzle Dazzle</h1>
        <UserMenu
          onOpenLogin={() => setShowLoginModal(true)}
          onOpenRegister={() => setShowRegisterModal(true)}
          onOpenBrowser={() => setShowGameBrowser(true)}
        />
      </header>

      <main className="flex-1 flex flex-col items-center p-2 sm:p-4 pb-4">

        {error && (
          <div className="mb-2 px-4 py-1.5 bg-red-600 text-white rounded text-sm">
            {error}
          </div>
        )}

        {isLoading && !gameState && (
          <div className="text-gray-400">Loading...</div>
        )}

        {gameState && (
          <>
            {/* Status line: turn indicator / game over / AI thinking */}
            <div className="mb-2 text-center h-7 flex items-center justify-center gap-2">
              {gameState.status === 'finished' && gameState.winner !== null && (
                <span className="text-xl font-bold text-yellow-400">{getWinnerText()}</span>
              )}
              {gameState.status === 'finished' && gameState.winner === null && (
                <span className="text-xl font-bold text-gray-400">Draw!</span>
              )}
              {turnIndicator && gameState.status === 'playing' && (
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
              {isViewingHistory && (
                <span className="text-gray-400 text-xs ml-2 sm:hidden">
                  (move {viewPly}/{rawMoves.length})
                </span>
              )}
            </div>

            {/* Opponent name above board */}
            <div className="text-xs text-gray-500 mb-1">{topName}</div>

            {/* Board + eval meter + move history (desktop) row */}
            <div className="flex gap-2 items-start">
              <Board
                board={gameState.board}
                currentPlayer={gameState.current_player}
                legalMoves={gameState.legal_moves}
                selectedSquare={selectedSquare}
                onSquareClick={handleSquareClick}
                onDragMove={handleDragMove}
                touchedMask={gameState.touched_mask}
                mustPass={mustPass}
                flipped={shouldFlipBoard}
                lastMove={lastMove}
                animate={!isViewingHistory}
              />
              {/* Eval meter - AI mode only, right of board */}
              {settings.mode === 'ai' && (
                <EvaluationMeter value={evaluation} />
              )}
              {/* Desktop move history panel */}
              <div className="hidden sm:block">
                <MoveHistory moves={rawMoves} />
              </div>
            </div>

            {/* Player name below board */}
            <div className="text-xs text-gray-500 mt-1">{bottomName}</div>

            {/* Desktop: navigation buttons below board */}
            <div className="hidden sm:flex items-center gap-2 mt-2 justify-center">
              <button
                onClick={goToStart}
                disabled={rawMoves.length === 0}
                className="px-2 py-1 text-sm bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 rounded transition-colors"
                title="Go to start (Home)"
              >
                {'\u{25C0}\u{25C0}'}
              </button>
              <button
                onClick={goBack}
                disabled={rawMoves.length === 0 && viewPly === null}
                className="px-2 py-1 text-sm bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 rounded transition-colors"
                title="Back (Left arrow)"
              >
                {'\u{25C0}'}
              </button>
              {isViewingHistory && (
                <span className="text-gray-400 text-xs">
                  move {viewPly}/{rawMoves.length}
                </span>
              )}
              <button
                onClick={goForward}
                disabled={viewPly === null}
                className="px-2 py-1 text-sm bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 rounded transition-colors"
                title="Forward (Right arrow)"
              >
                {'\u{25B6}'}
              </button>
              <button
                onClick={goToEnd}
                disabled={viewPly === null}
                className="px-2 py-1 text-sm bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 rounded transition-colors"
                title="Go to end (End)"
              >
                {'\u{25B6}\u{25B6}'}
              </button>
            </div>

            {/* Mobile: compact move history bar + navigation */}
            <div className="mt-2 w-full max-w-[400px] sm:hidden">
              <div className="flex items-center gap-1 justify-center">
                <button
                  onClick={goToStart}
                  disabled={rawMoves.length === 0}
                  className="px-2 py-1 text-sm bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 rounded transition-colors"
                  title="Go to start (Home)"
                >
                  {'\u{25C0}\u{25C0}'}
                </button>
                <button
                  onClick={goBack}
                  disabled={rawMoves.length === 0 && viewPly === null}
                  className="px-2 py-1 text-sm bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 rounded transition-colors"
                  title="Back (Left arrow)"
                >
                  {'\u{25C0}'}
                </button>

                {/* Scrollable move history */}
                <div className="flex-1 overflow-x-auto whitespace-nowrap bg-gray-800 rounded px-2 py-1 text-xs min-h-[28px] flex items-center gap-1 scrollbar-thin">
                  {formattedTurns.length === 0 && (
                    <span className="text-gray-600 italic">No moves</span>
                  )}
                  {formattedTurns.map((turn: { blue: string; red: string }, idx: number) => (
                    <span key={idx} className="inline-flex items-center gap-0.5">
                      <span className="text-gray-500">{idx + 1}.</span>
                      {turn.blue && <span className="text-blue-400">{turn.blue}</span>}
                      {turn.red && <span className="text-red-400">{turn.red}</span>}
                    </span>
                  ))}
                </div>

                <button
                  onClick={goForward}
                  disabled={viewPly === null}
                  className="px-2 py-1 text-sm bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 rounded transition-colors"
                  title="Forward (Right arrow)"
                >
                  {'\u{25B6}'}
                </button>
                <button
                  onClick={goToEnd}
                  disabled={viewPly === null}
                  className="px-2 py-1 text-sm bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 rounded transition-colors"
                  title="Go to end (End)"
                >
                  {'\u{25B6}\u{25B6}'}
                </button>
              </div>
            </div>

            {/* Action buttons */}
            <div className="mt-3 flex flex-wrap justify-center gap-2">
              {/* Complete Pass / Cancel Pass (during pass chain) */}
              {isPassing && !isViewingHistory && gameState.status === 'playing' && (
                <>
                  <button
                    onClick={endTurn}
                    disabled={isLoading || aiThinking || !canEndTurn}
                    className="px-3 py-2 sm:px-4 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded font-medium transition-colors animate-pulse text-sm sm:text-base"
                  >
                    Complete Pass
                  </button>
                  <button
                    onClick={cancelPass}
                    disabled={isLoading || aiThinking}
                    className="px-3 py-2 sm:px-4 bg-gray-600 hover:bg-gray-700 rounded font-medium transition-colors text-sm sm:text-base"
                  >
                    Cancel Pass
                  </button>
                </>
              )}

              {/* End Turn (non-pass, e.g. forced pass scenario where canEndTurn is true but no pass chain yet) */}
              {canEndTurn && !isPassing && !isViewingHistory && gameState.status === 'playing' && (
                <button
                  onClick={endTurn}
                  disabled={isLoading || aiThinking}
                  className="px-3 py-2 sm:px-4 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded font-medium transition-colors animate-pulse text-sm sm:text-base"
                >
                  End Turn
                </button>
              )}

              {/* New Game */}
              <button
                onClick={gameState.status === 'finished' ? handleQuickNewGame : handleNewGameClick}
                disabled={isLoading}
                className="px-3 py-2 sm:px-4 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded font-medium transition-colors text-sm sm:text-base"
              >
                New Game
              </button>

              {/* Resign - only during active game */}
              {gameState.ply > 0 && gameState.status === 'playing' && !isPassing && !isViewingHistory && (
                <button
                  onClick={() => setShowResignConfirm(true)}
                  disabled={isLoading}
                  className="px-3 py-2 sm:px-4 bg-gray-600 hover:bg-red-700 text-gray-300 hover:text-white disabled:bg-gray-600 rounded font-medium transition-colors text-sm sm:text-base"
                >
                  Resign
                </button>
              )}

              {/* Undo - not during pass */}
              {settings.mode === 'ai' && !isPassing && !isViewingHistory && (
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
            </div>
          </>
        )}

        {/* Instructions - collapsed on mobile */}
        <div className="mt-4 text-xs text-gray-600 max-w-md text-center hidden sm:block">
          <p>Click a piece to select, then click a destination. After passing, click "Complete Pass".</p>
          <p className="mt-1">
            Keys: N=New Game, U=Undo, E=End Turn, F=Flip, M=Mute, ?=Rules, {'\u{2190}\u{2192}'}=History
          </p>
        </div>
      </main>

      {/* New Game Dialog */}
      <NewGameDialog
        isOpen={showNewGameDialog}
        onClose={() => setShowNewGameDialog(false)}
        onStartGame={handleStartGame}
        onPlayOnline={() => {
          setShowNewGameDialog(false);
          setShowOnlineLobby(true);
        }}
        availableModels={availableModels}
        currentSettings={settings}
      />

      {/* Confirm Resign Dialog */}
      <ConfirmDialog
        isOpen={showResignConfirm}
        title="Resign Game?"
        message="Are you sure you want to resign? Your opponent will win."
        confirmText="Resign"
        cancelText="Cancel"
        onConfirm={() => {
          setShowResignConfirm(false);
          resign();
        }}
        onCancel={() => setShowResignConfirm(false)}
      />

      {/* Rules Modal */}
      <RulesModal isOpen={showRules} onClose={() => setShowRules(false)} />

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

      {/* Training Dashboard */}
      {showTrainingDashboard && (
        <TrainingDashboard
          onClose={() => setShowTrainingDashboard(false)}
          refreshInterval={10000}
        />
      )}

      {/* Online Lobby */}
      <OnlineLobby
        isOpen={showOnlineLobby}
        onClose={() => setShowOnlineLobby(false)}
        onGameCreated={handleGameCreated}
        onGameJoined={handleGameJoined}
      />

      {/* Waiting for Opponent */}
      {waitingGame && (
        <WaitingForOpponent
          joinCode={waitingGame.joinCode}
          hostColor={waitingGame.hostColor}
          onCancel={handleCancelWaiting}
        />
      )}
    </div>
  );
}

// Standalone Training Dashboard page for /dashboard route
function TrainingDashboardPage() {
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
      onGameEnd={(winner, reason) => {
        console.log('Game ended:', { winner, reason });
      }}
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
      <h1 className="text-2xl font-bold mb-4">Razzle Dazzle</h1>
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
      <BrowserRouter>
        <AuthProvider>
          <Routes>
            <Route path="/" element={<AppContentWrapper />} />
            <Route path="/dashboard" element={<TrainingDashboardPage />} />
            <Route path="/online/:gameId" element={<OnlineGamePage />} />
            <Route path="/join/:code" element={<JoinGamePage />} />
          </Routes>
        </AuthProvider>
      </BrowserRouter>
    </ErrorBoundary>
  );
}
