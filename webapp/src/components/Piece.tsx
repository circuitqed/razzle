import type { Player } from '../types';

interface PieceProps {
  player: Player;
  hasBall: boolean;
  isSelected?: boolean;
  isIneligible?: boolean;
  mustPass?: boolean;
  onClick?: () => void;
}

export default function Piece({ player, hasBall, isSelected, isIneligible, mustPass, onClick }: PieceProps) {
  const baseColor = player === 0 ? '#3b82f6' : '#ef4444';
  const strokeColor = isSelected ? '#fbbf24' : '#1f2937';
  const strokeWidth = isSelected ? 3 : 1.5;

  return (
    <g onClick={onClick} style={{ cursor: onClick ? 'pointer' : 'default' }}>
      {/* Main piece - diamond shape */}
      <polygon
        points="25,8 42,25 25,42 8,25"
        fill={baseColor}
        stroke={strokeColor}
        strokeWidth={strokeWidth}
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

      {/* Ineligible indicator - small X in corner */}
      {isIneligible && (
        <g>
          <line x1="38" y1="5" x2="45" y2="12" stroke="#dc2626" strokeWidth="2" strokeLinecap="round" />
          <line x1="45" y1="5" x2="38" y2="12" stroke="#dc2626" strokeWidth="2" strokeLinecap="round" />
        </g>
      )}
    </g>
  );
}
