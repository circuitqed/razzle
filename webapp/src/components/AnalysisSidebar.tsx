import EvaluationBar from './EvaluationBar';
import type { MoveAnalysis } from '../api/games';

interface AnalysisSidebarProps {
  value: number;
  currentPlayer: 0 | 1;
  topMoves: MoveAnalysis[];
  isAnalyzing: boolean;
  onMoveHover?: (move: number | null) => void;
}

export default function AnalysisSidebar({
  value,
  currentPlayer,
  topMoves,
  isAnalyzing,
  onMoveHover,
}: AnalysisSidebarProps) {
  // Format value as percentage
  const formatValue = (v: number) => {
    const pct = v * 100;
    return pct >= 0 ? `+${pct.toFixed(0)}%` : `${pct.toFixed(0)}%`;
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4 w-64">
      {/* Header with evaluation bar */}
      <div className="flex gap-4 mb-4">
        <EvaluationBar value={value} currentPlayer={currentPlayer} height={120} />
        <div className="flex-1">
          <h3 className="font-medium text-gray-300 mb-1">Evaluation</h3>
          <div className={`text-2xl font-bold ${
            value > 0.05 ? 'text-green-400' :
            value < -0.05 ? 'text-red-400' :
            'text-gray-400'
          }`}>
            {formatValue(value)}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {value > 0.05 ? (currentPlayer === 0 ? 'Blue advantage' : 'Red advantage') :
             value < -0.05 ? (currentPlayer === 0 ? 'Red advantage' : 'Blue advantage') :
             'Equal position'}
          </div>
          {isAnalyzing && (
            <div className="text-xs text-blue-400 mt-2 animate-pulse">
              Analyzing...
            </div>
          )}
        </div>
      </div>

      {/* Top moves */}
      <div>
        <h4 className="text-sm font-medium text-gray-400 mb-2">Top Moves</h4>
        {topMoves.length === 0 ? (
          <div className="text-sm text-gray-500 italic">
            {isAnalyzing ? 'Calculating...' : 'No analysis available'}
          </div>
        ) : (
          <div className="space-y-1">
            {topMoves.slice(0, 5).map((move, idx) => (
              <div
                key={move.move}
                className={`flex items-center gap-2 p-2 rounded text-sm cursor-pointer transition-colors ${
                  idx === 0 ? 'bg-green-900/50 text-green-300' : 'bg-gray-700/50 text-gray-300 hover:bg-gray-700'
                }`}
                onMouseEnter={() => onMoveHover?.(move.move)}
                onMouseLeave={() => onMoveHover?.(null)}
              >
                <span className="w-5 text-center font-medium text-gray-500">
                  {idx + 1}.
                </span>
                <span className="font-mono flex-1">{move.algebraic}</span>
                <span className={`text-xs ${
                  move.value > 0 ? 'text-green-400' : move.value < 0 ? 'text-red-400' : 'text-gray-400'
                }`}>
                  {formatValue(move.value)}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="mt-4 pt-4 border-t border-gray-700">
        <h4 className="text-xs font-medium text-gray-500 mb-2">Move Quality</h4>
        <div className="grid grid-cols-2 gap-1 text-xs">
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-green-500"></span>
            <span className="text-gray-400">Best</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-blue-500"></span>
            <span className="text-gray-400">Good</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-yellow-500"></span>
            <span className="text-gray-400">Inaccuracy</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-orange-500"></span>
            <span className="text-gray-400">Mistake</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-red-500"></span>
            <span className="text-gray-400">Blunder</span>
          </div>
        </div>
      </div>
    </div>
  );
}
