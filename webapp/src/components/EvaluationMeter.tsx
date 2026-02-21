/**
 * Evaluation Meter component.
 * Displays AI's evaluation of the position as a vertical bar.
 * Red (top) = good for AI, Blue (bottom) = good for player.
 *
 * Uses self-stretch to match the board SVG height, then excludes
 * the bottom label area (3.4% of total) so the bar aligns with
 * just the board squares.
 */

interface EvaluationMeterProps {
  /** Evaluation from player's perspective: +1 = player winning, -1 = AI winning */
  value: number | null;
}

// The board SVG viewBox is 414 tall (400 squares + 14 label padding).
// The bar should span only the squares portion = 400/414 ≈ 96.6%.
const LABEL_BOTTOM_PCT = 3.4; // 14/414 * 100

export default function EvaluationMeter({ value }: EvaluationMeterProps) {
  // Indicator position as percentage from top
  // value +1 (blue winning) = 100% from top (at blue/bottom)
  // value -1 (red winning) = 0% from top (at red/top)
  // value 0 (equal) = 50% from top (middle)
  const indicatorPosition = value !== null ? (value + 1) * 50 : 50;
  const clampedPosition = Math.max(0, Math.min(100, indicatorPosition));

  // Determine which side is winning
  const blueWinning = value !== null && value > 0;
  const redWinning = value !== null && value < 0;

  return (
    <div
      className="self-stretch flex flex-col items-center"
      style={{ paddingBottom: `${LABEL_BOTTOM_PCT}%` }}
    >
      {/* Meter container - gradient from red (top) to blue (bottom) */}
      <div
        className="relative w-3 flex-1 bg-gradient-to-b from-red-600 via-gray-500 to-blue-600 rounded-full overflow-hidden"
        title={value !== null ? `Evaluation: ${value > 0 ? '+' : ''}${(value * 100).toFixed(0)}%` : 'No evaluation yet'}
      >
        {/* Darkened overlay on the losing side */}
        {/* When blue winning: darken red (top) from 0 to indicator */}
        {blueWinning && (
          <div
            className="absolute top-0 left-0 right-0 bg-gray-900/60 transition-all duration-500 ease-out"
            style={{ height: `${clampedPosition}%` }}
          />
        )}
        {/* When red winning: darken blue (bottom) from indicator to 100 */}
        {redWinning && (
          <div
            className="absolute bottom-0 left-0 right-0 bg-gray-900/60 transition-all duration-500 ease-out"
            style={{ height: `${100 - clampedPosition}%` }}
          />
        )}

        {/* Center line indicator */}
        <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-white/40" />

        {/* Current position indicator - white line at the boundary */}
        <div
          className="absolute left-0 right-0 h-0.5 bg-white shadow-lg transition-all duration-500 ease-out"
          style={{ top: `${clampedPosition}%` }}
        />
      </div>

      {/* Numeric value shown below */}
      {value !== null && (
        <div className="text-xs text-gray-400 tabular-nums mt-1">
          {value > 0 ? '+' : ''}{(value * 100).toFixed(0)}%
        </div>
      )}
    </div>
  );
}
