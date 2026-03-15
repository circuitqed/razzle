import { useEffect } from 'react';
import Board from './Board';
import { useOpeningExplorer } from '../hooks/useOpeningExplorer';
import type { BookMove, Turn } from '../hooks/useOpeningExplorer';

function formatCount(n: number): string {
  return n.toLocaleString();
}

function WinRateBar({ winRate }: { winRate: number }) {
  const bluePercent = Math.round(winRate * 100);
  const redPercent = 100 - bluePercent;
  return (
    <div className="flex h-4 rounded overflow-hidden bg-gray-700 min-w-[80px]">
      {bluePercent > 0 && (
        <div
          className="bg-blue-500 flex items-center justify-center text-[10px] text-white font-medium leading-none"
          style={{ width: `${bluePercent}%` }}
        >
          {bluePercent >= 15 ? `${bluePercent}%` : ''}
        </div>
      )}
      {redPercent > 0 && (
        <div
          className="bg-red-500 flex items-center justify-center text-[10px] text-white font-medium leading-none"
          style={{ width: `${redPercent}%` }}
        >
          {redPercent >= 15 ? `${redPercent}%` : ''}
        </div>
      )}
    </div>
  );
}

function MoveRow({ move, onPlay }: { move: BookMove; onPlay: () => void }) {
  const typeIcon = move.type === 'knight' ? '\u265E' : move.type === 'pass' ? '\u2192' : '\u23F9';
  return (
    <tr
      className="hover:bg-gray-700/50 cursor-pointer transition-colors"
      onClick={onPlay}
    >
      <td className="py-1.5 px-2 font-mono text-sm whitespace-nowrap">
        <span className="text-gray-400 mr-1">{typeIcon}</span>
        {move.label}
      </td>
      <td className="py-1.5 px-2 text-right text-sm text-gray-300 tabular-nums">
        {formatCount(move.count)}
      </td>
      <td className="py-1.5 px-2 w-full min-w-[100px]">
        <WinRateBar winRate={move.winRate} />
      </td>
    </tr>
  );
}

function MoveHistory({
  turns,
  currentIndex,
  onGoTo,
}: {
  turns: Turn[];
  currentIndex: number;
  onGoTo: (index: number) => void;
}) {
  if (turns.length === 0) return null;

  // Group turns into numbered moves (Blue + Red pairs)
  const moveNumbers: { num: number; blue?: Turn; red?: Turn }[] = [];
  let moveNum = 1;
  for (let i = 0; i < turns.length; i++) {
    const turn = turns[i];
    if (turn.player === 0) {
      moveNumbers.push({ num: moveNum, blue: turn });
    } else {
      if (moveNumbers.length > 0 && !moveNumbers[moveNumbers.length - 1].red && moveNumbers[moveNumbers.length - 1].blue) {
        moveNumbers[moveNumbers.length - 1].red = turn;
        moveNum++;
      } else {
        moveNumbers.push({ num: moveNum, red: turn });
        moveNum++;
      }
    }
  }

  return (
    <div className="flex flex-wrap gap-x-1 gap-y-0.5 text-xs font-mono px-1">
      {moveNumbers.map((m) => (
        <span key={m.num} className="flex items-center gap-0.5">
          <span className="text-gray-600">{m.num}.</span>
          {m.blue && (
            <button
              className={`px-0.5 rounded transition-colors ${
                currentIndex >= m.blue.startIndex && currentIndex <= m.blue.endIndex
                  ? 'text-blue-300 bg-blue-900/40'
                  : 'text-blue-400/70 hover:text-blue-300'
              }`}
              onClick={() => onGoTo(m.blue!.endIndex)}
            >
              {m.blue.display}
            </button>
          )}
          {m.red && (
            <button
              className={`px-0.5 rounded transition-colors ${
                currentIndex >= m.red.startIndex && currentIndex <= m.red.endIndex
                  ? 'text-red-300 bg-red-900/40'
                  : 'text-red-400/70 hover:text-red-300'
              }`}
              onClick={() => onGoTo(m.red!.endIndex)}
            >
              {m.red.display}
            </button>
          )}
        </span>
      ))}
    </div>
  );
}

export default function OpeningExplorer() {
  const {
    loading,
    error,
    bookSize,
    currentState,
    currentIndex,
    historyLength,
    bookMoves,
    boardState,
    turns,
    playMove,
    goBack,
    goForward,
    goToStart,
    goToPosition,
  } = useOpeningExplorer();

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      switch (e.key) {
        case 'ArrowLeft':
          e.preventDefault();
          goBack();
          break;
        case 'ArrowRight':
          e.preventDefault();
          goForward();
          break;
        case 'Home':
          e.preventDefault();
          goToStart();
          break;
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [goBack, goForward, goToStart]);

  const turnLabel = currentState.currentPlayer === 0 ? 'Blue' : 'Red';
  const turnColorClass = currentState.currentPlayer === 0 ? 'text-blue-400' : 'text-red-400';

  return (
    <div className="min-h-[100dvh] bg-gray-900 text-white flex flex-col">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-2 shrink-0">
        <a href="/" className="text-gray-400 hover:text-white transition-colors text-sm">
          &larr; Back
        </a>
        <h1 className="text-xl sm:text-2xl font-bold">Opening Explorer</h1>
        <div className="w-16" />
      </header>

      {/* Main content */}
      <main className="flex-1 flex flex-col lg:flex-row items-start justify-center gap-4 p-2 sm:p-4">
        {/* Board */}
        <div className="w-full lg:w-auto flex justify-center shrink-0">
          <div className="w-full max-w-[min(60vh,28rem)]">
            <Board
              board={boardState}
              currentPlayer={currentState.currentPlayer as 0 | 1}
              legalMoves={[]}
              selectedSquare={null}
              onSquareClick={() => {}}
              flipped={false}
              touchedMask={currentState.touchedMask.toString()}
              animate={false}
            />
          </div>
        </div>

        {/* Explorer panel */}
        <div className="w-full lg:w-80 flex flex-col gap-3">
          {/* Status line */}
          <div className="flex items-center justify-between text-sm px-1">
            <span className={turnColorClass}>
              {turnLabel} to move &middot; Ply {currentState.ply}
            </span>
            {!loading && (
              <span className="text-gray-500">
                {bookSize.toLocaleString()} positions
              </span>
            )}
          </div>

          {/* Move history (grouped by turn) */}
          <MoveHistory
            turns={turns}
            currentIndex={currentIndex}
            onGoTo={goToPosition}
          />

          {/* Nav buttons */}
          <div className="flex gap-1 px-1">
            <button
              onClick={goToStart}
              disabled={currentIndex === 0}
              className="px-2 py-1 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 rounded text-sm transition-colors"
              title="Go to start (Home)"
            >
              |&lt;
            </button>
            <button
              onClick={goBack}
              disabled={currentIndex === 0}
              className="px-2 py-1 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 rounded text-sm transition-colors"
              title="Go back (Left)"
            >
              &lt;
            </button>
            <button
              onClick={goForward}
              disabled={currentIndex >= historyLength - 1}
              className="px-2 py-1 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:text-gray-600 rounded text-sm transition-colors"
              title="Go forward (Right)"
            >
              &gt;
            </button>
          </div>

          {/* Move table */}
          {loading ? (
            <div className="text-gray-400 text-center py-8 animate-pulse">
              Loading opening book...
            </div>
          ) : error ? (
            <div className="text-red-400 text-center py-8">
              Failed to load book: {error}
            </div>
          ) : bookMoves.length > 0 ? (
            <div className="overflow-y-auto">
              <table className="w-full">
                <thead>
                  <tr className="text-gray-500 text-xs uppercase border-b border-gray-700">
                    <th className="py-1 px-2 text-left">Move</th>
                    <th className="py-1 px-2 text-right">Games</th>
                    <th className="py-1 px-2 text-left">Win Rate</th>
                  </tr>
                </thead>
                <tbody>
                  {bookMoves.map((move) => (
                    <MoveRow
                      key={move.actions.join(',')}
                      move={move}
                      onPlay={() => playMove(move.actions)}
                    />
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-gray-500 text-center py-8">
              No book data for this position
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
