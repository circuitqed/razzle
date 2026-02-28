/**
 * Evaluation Meter component.
 * Displays AI's evaluation of the position as a vertical bar
 * that spans the board-squares region (excluding labels).
 * The percentage text sits directly below, aligned with the board's file labels.
 */

import { useMemo } from 'react';

interface EvaluationMeterProps {
  /** Evaluation from player's perspective: +1 = player winning, -1 = AI winning */
  value: number | null;
  /** When true, blue is at top and red at bottom (matches flipped board) */
  flipped?: boolean;
  /** Human player's color (0=blue, 1=red) — used for eval percentage coloring */
  playerColor?: number;
}

export default function EvaluationMeter({ value, flipped = false, playerColor = 0 }: EvaluationMeterProps) {
  // Convert from human-perspective value to blue-perspective value.
  // value > 0 means human winning; we need blue advantage for bar positioning.
  const blueAdv = value !== null ? (playerColor === 0 ? value : -value) : null;

  // Indicator position as percentage from top.
  // Normal:  red at top, blue at bottom → blue winning = 100% from top (near blue end)
  // Flipped: blue at top, red at bottom → blue winning = 0% from top (near blue end)
  const rawPosition = blueAdv !== null ? (blueAdv + 1) * 50 : 50;
  const indicatorPosition = flipped ? 100 - rawPosition : rawPosition;
  const clampedPosition = Math.max(0, Math.min(100, indicatorPosition));

  // Which color is at top/bottom depends on flip
  const topWinning = flipped
    ? (blueAdv !== null && blueAdv > 0)   // blue winning, blue at top
    : (blueAdv !== null && blueAdv < 0);  // red winning, red at top
  const bottomWinning = flipped
    ? (blueAdv !== null && blueAdv < 0)   // red winning, red at bottom
    : (blueAdv !== null && blueAdv > 0);  // blue winning, blue at bottom

  const gradientClass = flipped
    ? 'bg-gradient-to-b from-blue-600 via-gray-500 to-red-600'
    : 'bg-gradient-to-b from-red-600 via-gray-500 to-blue-600';

  // Eval percentage display
  const evalText = useMemo(() => {
    if (value === null) return null;
    const pct = (value * 100).toFixed(0);
    const text = `${value > 0 ? '+' : ''}${pct}%`;
    const isBlueWinning = (value > 0 && playerColor === 0) || (value < 0 && playerColor === 1);
    const colorClass = value === 0
      ? 'text-gray-400'
      : isBlueWinning ? 'text-blue-400' : 'text-red-400';
    return { text, colorClass };
  }, [value, playerColor]);

  return (
    <div className="self-stretch flex flex-col items-center">
      {/* Gradient bar — flex ratio 400 matches board squares height (8 rows * 50px) */}
      <div
        className={`relative w-3 ${gradientClass} rounded-full overflow-hidden`}
        style={{ flex: '400 0 0' }}
        title={value !== null ? `Evaluation: ${value > 0 ? '+' : ''}${(value * 100).toFixed(0)}%` : 'No evaluation yet'}
      >
        {/* Darkened overlay on the losing side */}
        {topWinning && (
          <div
            className="absolute bottom-0 left-0 right-0 bg-gray-900/60 transition-all duration-500 ease-out"
            style={{ height: `${100 - clampedPosition}%` }}
          />
        )}
        {bottomWinning && (
          <div
            className="absolute top-0 left-0 right-0 bg-gray-900/60 transition-all duration-500 ease-out"
            style={{ height: `${clampedPosition}%` }}
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
      {/* Percentage text — flex ratio 14 matches board label pad (file labels) */}
      <div style={{ flex: '14 0 0' }} className="flex items-end justify-center">
        {evalText && (
          <span className={`text-[10px] font-medium tabular-nums leading-none ${evalText.colorClass}`}>
            {evalText.text}
          </span>
        )}
      </div>
    </div>
  );
}
