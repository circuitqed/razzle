import { formatMovesForDisplay, type FormattedTurn } from '../utils/replay';

interface MoveHistoryProps {
  /** Raw move integers from the game */
  moves: number[];
}

export default function MoveHistory({ moves }: MoveHistoryProps) {
  // Fixed height to match board (8 rows × 50px = 400px)
  const FIXED_HEIGHT = 'h-[400px]';

  if (moves.length === 0) {
    return (
      <div className={`w-48 ${FIXED_HEIGHT} bg-gray-800 rounded p-3 text-sm`}>
        <h3 className="font-semibold mb-2 text-gray-300">Move History</h3>
        <p className="text-gray-500 italic">No moves yet</p>
      </div>
    );
  }

  // Format moves with pass chains (e.g., "d1-c1-b1" instead of "d1-c1, c1-b1, End Turn")
  const formattedTurns = formatMovesForDisplay(moves);

  // Count actual moves (excluding end turns which are -1)
  const actualMoveCount = moves.filter(m => m !== -1).length;

  return (
    <div className={`w-48 ${FIXED_HEIGHT} bg-gray-800 rounded p-3 text-sm flex flex-col`}>
      <h3 className="font-semibold mb-2 text-gray-300">Move History</h3>
      <div className="space-y-1 flex-1 overflow-y-auto">
        {formattedTurns.map((turn, idx) => (
          <div key={idx} className="flex gap-2">
            <span className="text-gray-500 w-6">{idx + 1}.</span>
            <span className="text-blue-400 flex-1">
              {turn.blue || '...'}
            </span>
            <span className="text-red-400 flex-1">
              {turn.red || '...'}
            </span>
          </div>
        ))}
      </div>
      <div className="mt-2 pt-2 border-t border-gray-700 text-gray-500 text-xs">
        {actualMoveCount} move{actualMoveCount !== 1 ? 's' : ''} played
      </div>
    </div>
  );
}
