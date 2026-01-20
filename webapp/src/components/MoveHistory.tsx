interface MoveRecord {
  move: number;
  algebraic: string;
  player: 0 | 1;
}

interface MoveHistoryProps {
  moves: MoveRecord[];
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

  // Group moves by turn (pairs of moves)
  const groupedMoves: { blue: string[]; red: string[] }[] = [];
  let currentTurn = { blue: [] as string[], red: [] as string[] };

  for (const record of moves) {
    if (record.player === 0) {
      // Blue's move - if we have red moves pending, start new turn
      if (currentTurn.red.length > 0) {
        groupedMoves.push(currentTurn);
        currentTurn = { blue: [], red: [] };
      }
      currentTurn.blue.push(record.algebraic);
    } else {
      currentTurn.red.push(record.algebraic);
    }
  }
  // Push last turn if it has any moves
  if (currentTurn.blue.length > 0 || currentTurn.red.length > 0) {
    groupedMoves.push(currentTurn);
  }

  return (
    <div className={`w-48 ${FIXED_HEIGHT} bg-gray-800 rounded p-3 text-sm flex flex-col`}>
      <h3 className="font-semibold mb-2 text-gray-300">Move History</h3>
      <div className="space-y-1 flex-1 overflow-y-auto">
        {groupedMoves.map((turn, idx) => (
          <div key={idx} className="flex gap-2">
            <span className="text-gray-500 w-6">{idx + 1}.</span>
            <span className="text-blue-400 flex-1">
              {turn.blue.join(', ') || '...'}
            </span>
            <span className="text-red-400 flex-1">
              {turn.red.join(', ') || '...'}
            </span>
          </div>
        ))}
      </div>
      <div className="mt-2 pt-2 border-t border-gray-700 text-gray-500 text-xs">
        {moves.length} move{moves.length !== 1 ? 's' : ''} played
      </div>
    </div>
  );
}
