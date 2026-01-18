import type { Player } from '../types';

interface PieceProps {
  player: Player;
  hasBall: boolean;
  isSelected?: boolean;
  onClick?: () => void;
}

export default function Piece({ player, hasBall, isSelected, onClick }: PieceProps) {
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
        <circle
          cx="25"
          cy="25"
          r="8"
          fill="#fbbf24"
          stroke="#92400e"
          strokeWidth="1.5"
        />
      )}
    </g>
  );
}
