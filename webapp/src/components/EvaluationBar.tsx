interface EvaluationBarProps {
  value: number; // -1 to 1 (from perspective of current player)
  currentPlayer: 0 | 1;
  height?: number;
}

export default function EvaluationBar({ value, currentPlayer, height = 300 }: EvaluationBarProps) {
  // Convert value to percentage (0% = P2 winning, 100% = P1 winning)
  // value is from perspective of current player
  // Positive = current player winning, negative = opponent winning
  const p1Advantage = currentPlayer === 0 ? value : -value;
  const percentage = (p1Advantage + 1) / 2 * 100;

  // Clamp between 5% and 95% for visual purposes
  const displayPercentage = Math.max(5, Math.min(95, percentage));

  // Format value for display
  const displayValue = Math.abs(value * 100).toFixed(0);
  const advantageText = value > 0.02
    ? (currentPlayer === 0 ? 'Blue' : 'Red')
    : value < -0.02
    ? (currentPlayer === 0 ? 'Red' : 'Blue')
    : 'Even';

  return (
    <div
      className="w-6 rounded overflow-hidden flex flex-col relative bg-gray-700"
      style={{ height }}
      title={`${advantageText} ${value > 0.02 || value < -0.02 ? `+${displayValue}%` : ''}`}
    >
      {/* Red (P2) section at top */}
      <div
        className="bg-red-500 transition-all duration-300"
        style={{ height: `${100 - displayPercentage}%` }}
      />
      {/* Blue (P1) section at bottom */}
      <div
        className="bg-blue-500 transition-all duration-300"
        style={{ height: `${displayPercentage}%` }}
      />

      {/* Center line */}
      <div
        className="absolute left-0 right-0 h-px bg-gray-400"
        style={{ top: '50%' }}
      />

      {/* Value indicator */}
      <div
        className="absolute left-0 right-0 flex justify-center transition-all duration-300"
        style={{ top: `${100 - displayPercentage}%`, transform: 'translateY(-50%)' }}
      >
        <div className="bg-gray-900 text-white text-xs px-1 rounded">
          {Math.abs(p1Advantage * 100).toFixed(0)}
        </div>
      </div>
    </div>
  );
}
