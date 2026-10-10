import type { Player } from '../types';

interface PieceProps {
  player: Player;
  hasBall: boolean;
  isSelected?: boolean;
  isIneligible?: boolean;
  mustPass?: boolean;
}

const COLORS = {
  0: { base: '#3b82f6', line: '#1d4ed8' },
  1: { base: '#ef4444', line: '#b91c1c' },
} as const;

// Lines parallel to the diamond's upper-left edge, 7 apart, running from the
// lower-left edge to the upper-right edge. Plain geometry rather than an SVG
// <pattern>: WebKit sometimes drops url(#id) fills after a re-render, leaving
// the piece empty.
const HATCH_LINES = [5, 12, 19].map((d) => {
  const t = d / (17 * Math.SQRT2);
  return [8 + 17 * t, 25 + 17 * t, 25 + 17 * t, 8 + 17 * t];
});

/**
 * A knight: a diamond. Solid with a light top facet when it can receive a
 * pass; lined (thin darker diagonal lines, no facet) when it can't yet,
 * i.e. it touched the ball and hasn't made a knight move since (all pieces
 * at the start).
 */
export default function Piece({ player, hasBall, isSelected, isIneligible, mustPass }: PieceProps) {
  const c = COLORS[player];
  const strokeColor = isSelected ? '#fbbf24' : '#1f2937';
  const strokeWidth = isSelected ? 3 : 1.5;

  return (
    <g>
      {/* Silhouette: same outline for both states, so pieces line up */}
      <polygon points="25,8 42,25 25,42 8,25" fill={c.base} />
      {isIneligible
        ? HATCH_LINES.map(([x1, y1, x2, y2], i) => (
            <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={c.line} strokeWidth={1.6} />
          ))
        : // Solid: a light facet on the upper half gives it some depth
          <polygon points="25,10.5 39.5,25 10.5,25" fill="#ffffff" opacity={0.22} />}
      <polygon
        points="25,8 42,25 25,42 8,25"
        fill="none"
        stroke={strokeColor}
        strokeWidth={strokeWidth}
        strokeLinejoin="round"
      />

      {/* Ball indicator */}
      {hasBall && (
        <>
          {/* Pulsing glow effect when must pass */}
          {mustPass && (
            <circle
              cx="25"
              cy="25"
              r="12"
              fill="none"
              stroke="#fbbf24"
              strokeWidth="3"
              opacity="0.7"
            >
              <animate
                attributeName="r"
                values="10;14;10"
                dur="1s"
                repeatCount="indefinite"
              />
              <animate
                attributeName="opacity"
                values="0.7;0.3;0.7"
                dur="1s"
                repeatCount="indefinite"
              />
            </circle>
          )}
          <circle
            cx="25"
            cy="25"
            r="8"
            fill="#fbbf24"
            stroke="#92400e"
            strokeWidth="1.5"
          />
        </>
      )}

    </g>
  );
}
