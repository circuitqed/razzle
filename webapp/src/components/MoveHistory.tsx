import { useMemo, useRef, useEffect } from 'react';
import { formatMovesForDisplay } from '../utils/replay';

interface MoveHistoryProps {
  /** Raw move integers from the game */
  moves: number[];
  /** Current viewing ply (null = live position) */
  viewPly?: number | null;
}

export default function MoveHistory({ moves, viewPly }: MoveHistoryProps) {
  // Fixed height to match board (8 rows × 50px = 400px)
  const FIXED_HEIGHT = 'h-[400px]';
  const scrollRef = useRef<HTMLDivElement>(null);

  const formattedTurns = useMemo(() => {
    return formatMovesForDisplay(moves);
  }, [moves]);

  // The effective ply for highlighting: null means live (= all moves visible)
  const effectivePly = viewPly ?? moves.length;

  // Auto-scroll to keep the highlighted move visible
  useEffect(() => {
    if (!scrollRef.current) return;
    const highlighted = scrollRef.current.querySelector('[data-highlighted="true"]');
    if (highlighted) {
      highlighted.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }, [effectivePly]);

  if (moves.length === 0) {
    return (
      <div className={`w-48 ${FIXED_HEIGHT} bg-gray-800 rounded p-3 text-sm`}>
        <h3 className="font-semibold mb-2 text-gray-300">Move History</h3>
        <p className="text-gray-500 italic">No moves yet</p>
      </div>
    );
  }

  return (
    <div className={`w-48 ${FIXED_HEIGHT} bg-gray-800 rounded p-3 text-sm flex flex-col`}>
      <h3 className="font-semibold mb-2 text-gray-300">Move History</h3>
      <div ref={scrollRef} className="space-y-1 flex-1 overflow-y-auto overflow-x-hidden">
        {formattedTurns.map((turn, idx) => {
          // Determine state for each cell: highlighted, future, or normal
          const blueIsCurrent = turn.bluePlyEnd !== undefined && turn.bluePlyEnd === effectivePly;
          const blueIsFuture = turn.bluePlyEnd !== undefined && turn.bluePlyEnd > effectivePly;
          const redIsCurrent = turn.redPlyEnd !== undefined && turn.redPlyEnd === effectivePly;
          const redIsFuture = turn.redPlyEnd !== undefined && turn.redPlyEnd > effectivePly;

          const blueClass = blueIsCurrent
            ? 'text-blue-300 bg-blue-900/50 rounded px-1 -mx-1'
            : blueIsFuture
              ? 'text-gray-600'
              : 'text-blue-400';
          const redClass = redIsCurrent
            ? 'text-red-300 bg-red-900/50 rounded px-1 -mx-1'
            : redIsFuture
              ? 'text-gray-600'
              : 'text-red-400';
          const numClass = (blueIsFuture && redIsFuture) || (!turn.blue && redIsFuture)
            ? 'text-gray-700'
            : 'text-gray-500';

          return (
            <div key={idx} className="flex gap-2">
              <span className={`${numClass} w-6`}>{idx + 1}.</span>
              <span
                className={`${blueClass} flex-1`}
                data-highlighted={blueIsCurrent || undefined}
              >
                {turn.blue || '...'}
              </span>
              <span
                className={`${redClass} flex-1`}
                data-highlighted={redIsCurrent || undefined}
              >
                {turn.red || '...'}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
