/**
 * Shared game view component used by both local (AppContent) and online (OnlineGame) modes.
 * Renders: board, move history, history navigation, pass/end-turn controls, status line.
 * Parent provides game state + handlers (from useGame or useOnlineGame) and mode-specific UI.
 */

import { useMemo, useRef, useEffect } from 'react';
import type { GameState } from '../types';
import Board from './Board';
import MoveHistory from './MoveHistory';
import EvaluationMeter from './EvaluationMeter';
import { formatMovesForDisplay } from '../utils/replay';

interface LastMove {
  from: number;
  to: number;
}

export interface GameViewProps {
  // Game state (from useBoardInteraction via parent hook)
  gameState: GameState | null;
  selectedSquare: number | null;
  isLoading: boolean;
  error: string | null;
  canEndTurn: boolean;
  mustPass: boolean;
  isPassing: boolean;
  lastMove: LastMove | null;
  lastTurnMoves?: LastMove[];
  rawMoves: number[];
  viewPly: number | null;
  isViewingHistory: boolean;

  // Handlers (from useBoardInteraction via parent hook)
  handleSquareClick: (square: number) => void;
  handleDragMove: (from: number, to: number) => void;
  endTurn: () => void;
  goForward: () => void;
  goBack: () => void;
  goToStart: () => void;
  goToEnd: () => void;

  // Display options
  flipped?: boolean;
  topName?: React.ReactNode;
  bottomName?: React.ReactNode;

  // Optional mode-specific features
  cancelPass?: () => void;
  evaluation?: number | null;
  aiThinking?: boolean;
  aiModelLoading?: boolean;
  /** Human player's color (0=blue, 1=red) — used for eval percentage coloring */
  playerColor?: number;

  // Clock elements rendered adjacent to player names
  topClock?: React.ReactNode;
  bottomClock?: React.ReactNode;

  // Custom status line content (turn indicator, etc.)
  statusLine?: React.ReactNode;

  // Extra action buttons (New Game, Undo, Resign, Sound, Flip, Rules, etc.)
  extraActions?: React.ReactNode;
}

export default function GameView({
  gameState,
  selectedSquare,
  isLoading,
  error,
  canEndTurn,
  mustPass,
  isPassing,
  lastMove,
  lastTurnMoves,
  rawMoves,
  viewPly,
  isViewingHistory,
  handleSquareClick,
  handleDragMove,
  endTurn,
  goForward,
  goBack,
  goToStart,
  goToEnd,
  flipped = false,
  topName,
  bottomName,
  cancelPass,
  evaluation,
  aiThinking = false,
  playerColor = 0,
  topClock,
  bottomClock,
  statusLine,
  extraActions,
}: GameViewProps) {
  // Format moves for compact mobile display
  const formattedTurns = useMemo(() => {
    return formatMovesForDisplay(rawMoves);
  }, [rawMoves]);

  // Effective ply for mobile move highlighting
  const effectivePly = viewPly ?? rawMoves.length;
  const mobileHistoryRef = useRef<HTMLDivElement>(null);

  // Auto-scroll mobile history to keep highlighted move visible
  useEffect(() => {
    if (!mobileHistoryRef.current) return;
    const highlighted = mobileHistoryRef.current.querySelector('[data-highlighted="true"]');
    if (highlighted) {
      highlighted.scrollIntoView({ inline: 'nearest', behavior: 'smooth' });
    }
  }, [effectivePly]);

  return (
    <>
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
            {gameState.status === 'finished' && gameState.winner !== null && statusLine}
            {gameState.status === 'finished' && gameState.winner === null && (
              <span className="text-xl font-bold text-gray-400">Draw!</span>
            )}
            {gameState.status === 'playing' && statusLine}
          </div>

          {/* Board + eval meter + move history (desktop) row */}
          <div className="flex gap-2 items-start">
            {/* Board column: names + board share same width */}
            <div className="inline-block w-full max-w-[400px] sm:max-w-none sm:w-auto">
              {/* Opponent name + clock above board */}
              {(topName || topClock) && (
                <div className="flex items-center justify-between mb-1" style={{ paddingLeft: '3.846%' }}>
                  {topName && <div className="text-xs text-gray-500">{topName}</div>}
                  {topClock && <div className="flex-shrink-0">{topClock}</div>}
                </div>
              )}
              <Board
                board={gameState.board}
                currentPlayer={gameState.current_player}
                legalMoves={gameState.legal_moves}
                selectedSquare={selectedSquare}
                onSquareClick={handleSquareClick}
                onDragMove={handleDragMove}
                touchedMask={gameState.touched_mask}
                mustPass={mustPass}
                flipped={flipped}
                lastMove={lastMove}
                lastTurnMoves={lastTurnMoves}
                animate={!isViewingHistory}
              />
              {/* Player name + clock below board */}
              {(bottomName || bottomClock) && (
                <div className="flex items-center justify-between mt-1" style={{ paddingLeft: '3.846%' }}>
                  {bottomName && <div className="text-xs text-gray-500">{bottomName}</div>}
                  {bottomClock && <div className="flex-shrink-0">{bottomClock}</div>}
                </div>
              )}
            </div>
            {/* Eval meter - only when evaluation is provided */}
            {evaluation !== undefined && evaluation !== null && (
              <EvaluationMeter value={evaluation} flipped={flipped} playerColor={playerColor} />
            )}
            {/* Desktop: move history panel + navigation buttons */}
            <div className="hidden sm:flex sm:flex-col sm:gap-2">
              <MoveHistory moves={rawMoves} viewPly={viewPly} />
              <div className="flex items-center gap-2 justify-center">
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
              <div ref={mobileHistoryRef} className="flex-1 overflow-x-auto whitespace-nowrap bg-gray-800 rounded px-2 py-1 text-xs min-h-[28px] flex items-center gap-1 scrollbar-thin">
                {formattedTurns.length === 0 && (
                  <span className="text-gray-600 italic">No moves</span>
                )}
                {formattedTurns.map((turn, idx) => {
                  const blueIsCurrent = turn.bluePlyEnd !== undefined && turn.bluePlyEnd === effectivePly;
                  const blueIsFuture = turn.bluePlyEnd !== undefined && turn.bluePlyEnd > effectivePly;
                  const redIsCurrent = turn.redPlyEnd !== undefined && turn.redPlyEnd === effectivePly;
                  const redIsFuture = turn.redPlyEnd !== undefined && turn.redPlyEnd > effectivePly;

                  const blueClass = blueIsCurrent
                    ? 'text-blue-300 bg-blue-900/50 rounded px-0.5'
                    : blueIsFuture ? 'text-gray-600' : 'text-blue-400';
                  const redClass = redIsCurrent
                    ? 'text-red-300 bg-red-900/50 rounded px-0.5'
                    : redIsFuture ? 'text-gray-600' : 'text-red-400';
                  const numClass = (blueIsFuture && redIsFuture) || (!turn.blue && redIsFuture)
                    ? 'text-gray-700' : 'text-gray-500';

                  return (
                    <span key={idx} className="inline-flex items-center gap-0.5">
                      <span className={numClass}>{idx + 1}.</span>
                      {turn.blue && (
                        <span className={blueClass} data-highlighted={blueIsCurrent || undefined}>
                          {turn.blue}
                        </span>
                      )}
                      {turn.red && (
                        <span className={redClass} data-highlighted={redIsCurrent || undefined}>
                          {turn.red}
                        </span>
                      )}
                    </span>
                  );
                })}
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
                {cancelPass && (
                  <button
                    onClick={cancelPass}
                    disabled={isLoading || aiThinking}
                    className="px-3 py-2 sm:px-4 bg-gray-600 hover:bg-gray-700 rounded font-medium transition-colors text-sm sm:text-base"
                  >
                    Cancel Pass
                  </button>
                )}
              </>
            )}

            {/* End Turn (non-pass) */}
            {canEndTurn && !isPassing && !isViewingHistory && gameState.status === 'playing' && (
              <button
                onClick={endTurn}
                disabled={isLoading || aiThinking}
                className="px-3 py-2 sm:px-4 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded font-medium transition-colors animate-pulse text-sm sm:text-base"
              >
                End Turn
              </button>
            )}

            {/* Mode-specific action buttons (New Game, Resign, Undo, etc.) */}
            {extraActions}
          </div>
        </>
      )}
    </>
  );
}
