/**
 * Online Game component.
 * Thin wrapper around GameView that adds online-specific overlay.
 * All gameplay UI is shared with local/AI games via GameView.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import BugReportDialog from './BugReportDialog';
import GameView from './GameView';
import OnlineGameOverlay from './OnlineGameOverlay';
import ConfirmDialog from './ConfirmDialog';
import RulesModal from './RulesModal';
import { PlayerClock, CorrespondenceClock } from './GameClock';
import { useOnlineGame } from '../hooks/useOnlineGame';
import { setSoundEnabled, isSoundEnabled } from '../utils/sounds';
import type { OnlineOpponentInfo } from '../api/online';

const FIRST_MOVE_TIMEOUT = 30;

interface OnlineGameProps {
  gameId: string;
  onGameEnd?: (winner: number | null, reason: string) => void;
}

export default function OnlineGame({ gameId, onGameEnd }: OnlineGameProps) {
  const navigate = useNavigate();
  const [soundOn, setSoundOn] = useState(isSoundEnabled());
  const [flipBoard, setFlipBoard] = useState(false);
  const [showResignConfirm, setShowResignConfirm] = useState(false);
  const [showRules, setShowRules] = useState(false);
  const [showBugReport, setShowBugReport] = useState(false);

  const handleGameEnd = useCallback(
    (winner: number | null, reason: string) => {
      onGameEnd?.(winner, reason);
    },
    [onGameEnd]
  );

  const handleOpponentJoined = useCallback((opponent: OnlineOpponentInfo) => {
    console.log('Opponent joined:', opponent);
  }, []);

  const {
    gameState,
    myColor,
    isMyTurn,
    opponent,
    opponentConnected,
    connectionStatus,
    onlineStatus,
    selectedSquare,
    isLoading,
    error,
    canEndTurn,
    mustPass,
    isPassing,
    rawMoves,
    lastTurnAnimMoves,
    disconnectWarning,
    handleSquareClick,
    handleDragMove,
    endTurn,
    cancelPass,
    leaveGame,
    reconnect,
    viewPly,
    isViewingHistory,
    effectiveGameState,
    displayLastMove,
    goForward,
    goBack,
    goToStart,
    goToEnd,
    rematchState,
    rematchGameId,
    requestRematch,
    acceptRematch,
    declineRematch,
  } = useOnlineGame({
    gameId,
    onGameEnd: handleGameEnd,
    onOpponentJoined: handleOpponentJoined,
  });

  // Redirect to home if the game is already over on initial load,
  // or if the game doesn't exist (error from status fetch).
  const hasRedirected = useRef(false);
  useEffect(() => {
    if (hasRedirected.current) return;
    // Game not found or not accessible
    if (error && !gameState) {
      hasRedirected.current = true;
      navigate('/');
      return;
    }
    // Game already finished/abandoned when we loaded the page (no live WS state yet)
    if ((onlineStatus === 'finished' || onlineStatus === 'abandoned') && !gameState) {
      hasRedirected.current = true;
      navigate('/');
    }
  }, [onlineStatus, error, gameState, navigate]);

  const handleResign = async () => {
    setShowResignConfirm(false);
    try {
      await leaveGame();
      // Stay on the page — the WS game_over/game_abandoned handler will
      // update the state, showing the result + rematch/back buttons.
    } catch (err) {
      console.error('Failed to leave game:', err);
    }
  };

  const toggleSound = () => {
    const newValue = !soundOn;
    setSoundOn(newValue);
    setSoundEnabled(newValue);
  };

  // Board flipping: red player sees board flipped by default
  const shouldFlipBoard = useMemo(() => {
    const baseFlip = myColor === 1;
    return flipBoard ? !baseFlip : baseFlip;
  }, [myColor, flipBoard]);

  // Player names
  const opponentDisplayName = useMemo(() => {
    if (opponent?.display_name) return opponent.display_name;
    return myColor === 0 ? 'Red' : 'Blue';
  }, [myColor, opponent]);

  // Forfeit countdown: show after 30s of opponent disconnect
  const FORFEIT_DISPLAY_DELAY = 30;
  const [forfeitCountdown, setForfeitCountdown] = useState<number | null>(null);

  useEffect(() => {
    if (!disconnectWarning) {
      setForfeitCountdown(null);
      return;
    }
    const tick = () => {
      const elapsed = (Date.now() - disconnectWarning.startTime) / 1000;
      const remaining = disconnectWarning.gracePeriod - elapsed;
      if (elapsed >= FORFEIT_DISPLAY_DELAY && remaining > 0) {
        setForfeitCountdown(Math.ceil(remaining));
      } else if (remaining <= 0) {
        setForfeitCountdown(0);
      } else {
        setForfeitCountdown(null);
      }
    };
    tick();
    const interval = setInterval(tick, 1000);
    return () => clearInterval(interval);
  }, [disconnectWarning]);

  // Rich name elements: opponent gets rating + connection dot + forfeit countdown
  const opponentNameEl = useMemo(() => (
    <div className="flex items-center gap-1.5">
      <span>{opponentDisplayName}</span>
      {opponent?.elo_rating && <span className="text-gray-600">({Math.round(opponent.elo_rating)})</span>}
      <span className={`w-1.5 h-1.5 rounded-full ${opponentConnected ? 'bg-green-500' : 'bg-red-500 animate-pulse'}`} />
      {forfeitCountdown !== null && (
        <span className="text-red-400 text-xs">
          forfeit in {forfeitCountdown}s
        </span>
      )}
    </div>
  ), [opponentDisplayName, opponent?.elo_rating, opponentConnected, forfeitCountdown]);

  const myNameEl = useMemo(() => (
    <span>{myColor === 0 ? 'Blue (You)' : 'Red (You)'}</span>
  ), [myColor]);

  // Names follow board orientation: player whose territory is at bottom gets bottomName
  // Not flipped: bottom = player 0 (blue), top = player 1 (red)
  // Flipped: bottom = player 1 (red), top = player 0 (blue)
  const p0Name = myColor === 0 ? myNameEl : opponentNameEl;
  const p1Name = myColor === 1 ? myNameEl : opponentNameEl;
  const topName = shouldFlipBoard ? p0Name : p1Name;
  const bottomName = shouldFlipBoard ? p1Name : p0Name;
  const topPlayerIndex = shouldFlipBoard ? 0 : 1;
  const bottomPlayerIndex = shouldFlipBoard ? 1 : 0;

  // Clock state — use isMyTurn (derived from raw server state) so clocks
  // don't switch when viewing history (effectiveGameState.current_player changes).
  const liveCurrentPlayer = myColor !== null ? (isMyTurn ? myColor : 1 - myColor) : null;
  const gameOver = gameState?.status !== 'playing';
  const isCorrespondence = gameState?.game_mode === 'correspondence';
  const timeRemaining = gameState?.time_remaining;
  const moveDeadline = gameState?.move_deadline;

  // Clocks visible immediately but don't tick until both players have completed one turn each.
  // Use ply count: ply >= 2 means both players have moved at least once.
  // Latch: once started, stays started (don't stop when viewing history at ply < 2).
  const clocksStartedRef = useRef(false);
  const clocksStarted = useMemo(() => {
    if (clocksStartedRef.current) return true;
    const started = (gameState?.ply ?? 0) >= 2;
    if (started) clocksStartedRef.current = true;
    return started;
  }, [gameState?.ply]);

  // 30s first-move countdown (ply 0 with opponent connected)
  const firstMoveStartRef = useRef<number | null>(null);
  const [firstMoveCountdown, setFirstMoveCountdown] = useState<number | null>(null);

  useEffect(() => {
    const isFirstMove = !isViewingHistory && gameState?.ply === 0 && gameState?.status === 'playing' && opponentConnected;
    if (isFirstMove) {
      if (firstMoveStartRef.current === null) {
        firstMoveStartRef.current = performance.now();
      }
      const tick = () => {
        if (!firstMoveStartRef.current) return;
        const elapsed = (performance.now() - firstMoveStartRef.current) / 1000;
        setFirstMoveCountdown(Math.max(0, FIRST_MOVE_TIMEOUT - elapsed));
      };
      tick();
      const interval = setInterval(tick, 100);
      return () => clearInterval(interval);
    } else {
      firstMoveStartRef.current = null;
      setFirstMoveCountdown(null);
    }
  }, [gameState?.ply, gameState?.status, opponentConnected, isViewingHistory]);

  // Navigate to rematch game when created
  useEffect(() => {
    if (rematchGameId) {
      navigate(`/online/${rematchGameId}`);
    }
  }, [rematchGameId, navigate]);

  const topClock = useMemo(() => {
    if (myColor === null) return undefined;
    if (isCorrespondence && moveDeadline) {
      return (
        <CorrespondenceClock
          deadline={moveDeadline}
          label=""
          isActive={clocksStarted && liveCurrentPlayer === topPlayerIndex}
          gameOver={gameOver}
        />
      );
    }
    if (timeRemaining) {
      return (
        <PlayerClock
          time={timeRemaining[topPlayerIndex]}
          label=""
          isActive={clocksStarted && liveCurrentPlayer === topPlayerIndex}
          gameOver={gameOver}
        />
      );
    }
    return undefined;
  }, [myColor, topPlayerIndex, isCorrespondence, moveDeadline, timeRemaining, liveCurrentPlayer, gameOver, clocksStarted]);

  const bottomClock = useMemo(() => {
    if (myColor === null) return undefined;
    if (isCorrespondence && moveDeadline) {
      return (
        <CorrespondenceClock
          deadline={moveDeadline}
          label=""
          isActive={clocksStarted && liveCurrentPlayer === bottomPlayerIndex}
          gameOver={gameOver}
        />
      );
    }
    if (timeRemaining) {
      return (
        <PlayerClock
          time={timeRemaining[bottomPlayerIndex]}
          label=""
          isActive={clocksStarted && liveCurrentPlayer === bottomPlayerIndex}
          gameOver={gameOver}
        />
      );
    }
    return undefined;
  }, [myColor, bottomPlayerIndex, isCorrespondence, moveDeadline, timeRemaining, liveCurrentPlayer, gameOver, clocksStarted]);

  // Winner text + color
  const winnerDisplay = useMemo(() => {
    if (!gameState || gameState.winner === null) return null;
    const winnerName = gameState.winner === myColor
      ? 'You'
      : (opponent?.display_name || (gameState.winner === 0 ? 'Blue' : 'Red'));
    const colorClass = gameState.winner === 0 ? 'text-blue-400' : 'text-red-400';
    return { text: `${winnerName} Win${gameState.winner === myColor ? '' : 's'}!`, colorClass };
  }, [gameState, myColor, opponent]);

  // Turn indicator
  const turnIndicator = useMemo(() => {
    if (!gameState || gameState.status !== 'playing') return null;
    const colorClass = gameState.current_player === 0 ? 'bg-blue-500' : 'bg-red-500';
    const text = isMyTurn ? 'Your Turn' : `${opponentDisplayName}'s Turn`;
    return { text, colorClass };
  }, [gameState, isMyTurn, opponentDisplayName]);

  return (
    <div className="h-[100dvh] bg-gray-900 text-white flex flex-col overflow-hidden">
      {/* Header bar with online overlay */}
      <header className="flex items-center justify-between px-4 py-2 shrink-0">
        <div className="w-20" />
        <h1 className="text-xl sm:text-2xl font-bold">Razzle Dazzle</h1>
        <div className="w-20 flex justify-end">
          <button
            onClick={() => setShowBugReport(true)}
            className="p-1.5 text-gray-400 hover:text-white transition-colors"
            title="Report a bug"
          >
{'\u{1F41B}'}
          </button>
        </div>
      </header>

      {/* Online-specific overlay (connection modals) */}
      {myColor !== null && (
        <OnlineGameOverlay
          connectionStatus={connectionStatus}
          onReconnect={reconnect}
        />
      )}

      <main className="flex-1 min-h-0 flex flex-col items-center p-2 sm:p-4 pb-4 overflow-y-auto">
        <GameView
          gameState={effectiveGameState}
          selectedSquare={selectedSquare}
          isLoading={isLoading}
          error={error}
          canEndTurn={canEndTurn}
          mustPass={mustPass}
          isPassing={isPassing}
          lastMove={displayLastMove}
          lastTurnMoves={lastTurnAnimMoves}
          rawMoves={rawMoves}
          viewPly={viewPly}
          isViewingHistory={isViewingHistory}
          handleSquareClick={handleSquareClick}
          handleDragMove={handleDragMove}
          endTurn={endTurn}
          cancelPass={cancelPass}
          goForward={goForward}
          goBack={goBack}
          goToStart={goToStart}
          goToEnd={goToEnd}
          flipped={shouldFlipBoard}
          topName={topName}
          bottomName={bottomName}
          topClock={topClock}
          bottomClock={bottomClock}
          statusLine={
            <>
              {gameState?.status === 'finished' && winnerDisplay && (
                <span className={`text-xl font-bold ${winnerDisplay.colorClass}`}>{winnerDisplay.text}</span>
              )}
              {turnIndicator && (
                <>
                  <span className={`inline-block px-3 py-0.5 rounded text-white text-sm font-medium ${turnIndicator.colorClass}`}>
                    {turnIndicator.text}
                  </span>
                  {firstMoveCountdown !== null && (
                    <span className={`font-mono text-sm ${firstMoveCountdown < 10 ? 'text-red-400' : 'text-gray-400'}`}>
                      {Math.ceil(firstMoveCountdown)}s
                    </span>
                  )}
                  {mustPass && isMyTurn && (
                    <span className="text-yellow-400 text-xs">forced pass</span>
                  )}
                </>
              )}
            </>
          }
          extraActions={
            <>
              {/* Resign - only during active game */}
              {gameState && gameState.ply > 0 && gameState.status === 'playing' && !isPassing && !isViewingHistory && (
                <button
                  onClick={() => setShowResignConfirm(true)}
                  disabled={isLoading}
                  className="px-3 py-2 sm:px-4 bg-gray-600 hover:bg-red-700 text-gray-300 hover:text-white disabled:bg-gray-600 rounded font-medium transition-colors text-sm sm:text-base"
                >
                  Resign
                </button>
              )}

              {/* Back to Menu - after game ends */}
              {gameState && gameState.status !== 'playing' && (
                <button
                  onClick={() => navigate('/', { state: { fromOnline: true } })}
                  className="px-3 py-2 sm:px-4 bg-gray-600 hover:bg-gray-700 rounded font-medium transition-colors text-sm sm:text-base"
                >
                  Back to Menu
                </button>
              )}

              {/* Rematch - after game ends */}
              {gameState && gameState.status !== 'playing' && rematchState === 'none' && (
                <button
                  onClick={requestRematch}
                  className="px-3 py-2 sm:px-4 bg-blue-600 hover:bg-blue-700 rounded font-medium transition-colors text-sm sm:text-base"
                >
                  Rematch
                </button>
              )}
              {gameState && gameState.status !== 'playing' && rematchState === 'sent' && (
                <button
                  disabled
                  className="px-3 py-2 sm:px-4 bg-gray-500 rounded font-medium text-sm sm:text-base cursor-not-allowed"
                >
                  Rematch Sent...
                </button>
              )}
              {gameState && gameState.status !== 'playing' && rematchState === 'received' && (
                <>
                  <button
                    onClick={acceptRematch}
                    className="px-3 py-2 sm:px-4 bg-green-600 hover:bg-green-700 rounded font-medium transition-colors text-sm sm:text-base"
                  >
                    Accept Rematch
                  </button>
                  <button
                    onClick={declineRematch}
                    className="px-3 py-2 sm:px-4 bg-gray-600 hover:bg-gray-700 rounded font-medium transition-colors text-sm sm:text-base"
                  >
                    Decline
                  </button>
                </>
              )}

              {/* Sound toggle */}
              <button
                onClick={toggleSound}
                className="px-3 py-2 bg-gray-600 hover:bg-gray-700 rounded font-medium transition-colors text-sm"
                title={soundOn ? 'Mute' : 'Unmute'}
              >
                {soundOn ? '\u{1F50A}' : '\u{1F507}'}
              </button>

              {/* Flip board */}
              <button
                onClick={() => setFlipBoard(f => !f)}
                className="px-3 py-2 bg-gray-600 hover:bg-gray-700 rounded font-medium transition-colors text-sm"
                title="Flip board"
              >
                {'\u{21C5}'}
              </button>

              {/* Rules */}
              <button
                onClick={() => setShowRules(true)}
                className="px-3 py-2 bg-gray-600 hover:bg-gray-700 rounded font-medium transition-colors text-sm"
                title="Rules"
              >
                ?
              </button>

            </>
          }
        />
      </main>

      {/* Resign Confirmation */}
      <ConfirmDialog
        isOpen={showResignConfirm}
        title="Resign Game?"
        message="Are you sure you want to resign? Your opponent will win."
        confirmText="Resign"
        cancelText="Cancel"
        onConfirm={handleResign}
        onCancel={() => setShowResignConfirm(false)}
      />

      {/* Rules Modal */}
      <RulesModal isOpen={showRules} onClose={() => setShowRules(false)} />

      {/* Bug Report Dialog */}
      <BugReportDialog
        isOpen={showBugReport}
        onClose={() => setShowBugReport(false)}
        gameState={gameState}
        rawMoves={rawMoves}
        gameMode="online"
        gameId={gameId}
      />
    </div>
  );
}
