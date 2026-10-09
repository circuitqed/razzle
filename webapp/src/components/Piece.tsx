import type { Player } from '../types';

interface PieceProps {
  player: Player;
  hasBall: boolean;
  isSelected?: boolean;
  isIneligible?: boolean;
  mustPass?: boolean;
}

const COLORS = {
  0: { base: '#3b82f6', ring: '#2563eb', tint: 'rgba(59, 130, 246, 0.1)' },
  1: { base: '#ef4444', ring: '#dc2626', tint: 'rgba(239, 68, 68, 0.1)' },
} as const;

/**
 * A knight: a diamond. Solid with a light top facet when it can receive a
 * pass; hollow (a coloured ring over a faint tint) when it can't yet, i.e. it
 * was touched by a pass and hasn't moved since (all pieces at the start).
 */
export default function Piece({ player, hasBall, isSelected, isIneligible, mustPass }: PieceProps) {
  const c = COLORS[player];
  const strokeColor = isSelected ? '#fbbf24' : '#1f2937';
  const strokeWidth = isSelected ? 3 : 1.5;

  return (
    <g>
      {/* Silhouette: same outline for both states, so pieces line up */}
      <polygon
        points="25,8 42,25 25,42 8,25"
        fill={isIneligible ? c.tint : c.base}
        stroke={strokeColor}
        strokeWidth={strokeWidth}
        strokeLinejoin="round"
      />
      {isIneligible ? (
        // Hollow: a thick ring in the player's colour just inside the outline
        <polygon
          points="25,11.5 38.5,25 25,38.5 11.5,25"
          fill="none"
          stroke={c.ring}
          strokeWidth={4}
          strokeLinejoin="round"
        />
      ) : (
        // Solid: a light facet on the upper half gives it some depth
        <polygon points="25,10.5 39.5,25 10.5,25" fill="#ffffff" opacity={0.22} />
      )}

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
