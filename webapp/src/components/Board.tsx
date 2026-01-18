import { useMemo } from 'react';
import type { BoardState, Player } from '../types';
import {
  BOARD_COLS,
  BOARD_ROWS,
  hasPiece,
  decodeMove,
} from '../types';
import Piece from './Piece';

interface BoardProps {
  board: BoardState;
  currentPlayer: Player;
  legalMoves: number[];
  selectedSquare: number | null;
  onSquareClick: (square: number) => void;
  flipped?: boolean;
}

const SQUARE_SIZE = 50;
const BOARD_WIDTH = BOARD_COLS * SQUARE_SIZE;
const BOARD_HEIGHT = BOARD_ROWS * SQUARE_SIZE;

export default function Board({
  board,
  currentPlayer,
  legalMoves,
  selectedSquare,
  onSquareClick,
  flipped = false,
}: BoardProps) {
  // Parse legal moves to get destination squares for selected piece
  const legalDestinations = useMemo(() => {
    if (selectedSquare === null) return new Set<number>();
    return new Set(
      legalMoves
        .map(decodeMove)
        .filter(({ src }) => src === selectedSquare)
        .map(({ dst }) => dst)
    );
  }, [selectedSquare, legalMoves]);

  // Get all squares with pieces that can move
  const movablePieces = useMemo(() => {
    return new Set(legalMoves.map((m) => decodeMove(m).src));
  }, [legalMoves]);

  // Render a single square
  const renderSquare = (row: number, col: number) => {
    const square = row * BOARD_COLS + col;
    const isLight = (row + col) % 2 === 0;

    // Determine visual position (flip if needed)
    const visualRow = flipped ? BOARD_ROWS - 1 - row : row;
    const visualCol = flipped ? BOARD_COLS - 1 - col : col;
    const x = visualCol * SQUARE_SIZE;
    const y = (BOARD_ROWS - 1 - visualRow) * SQUARE_SIZE;

    // Check what's on this square
    const hasP1Piece = hasPiece(board.p1_pieces, square);
    const hasP2Piece = hasPiece(board.p2_pieces, square);
    const hasP1Ball = hasPiece(board.p1_ball, square);
    const hasP2Ball = hasPiece(board.p2_ball, square);

    const isSelected = selectedSquare === square;
    const isLegalDest = legalDestinations.has(square);
    const canMove = movablePieces.has(square);

    // Determine background color
    let bgColor = isLight ? '#f0d9b5' : '#b58863';
    if (isSelected) {
      bgColor = '#bbcb2b';
    } else if (isLegalDest) {
      bgColor = isLight ? '#acd490' : '#87b560';
    }

    // Goal zones (rows 0 and 7)
    if (row === 0) {
      bgColor = isLegalDest ? '#60a5fa' : '#93c5fd';
    } else if (row === BOARD_ROWS - 1) {
      bgColor = isLegalDest ? '#f87171' : '#fca5a5';
    }

    return (
      <g key={square} onClick={() => onSquareClick(square)}>
        {/* Square background */}
        <rect
          x={x}
          y={y}
          width={SQUARE_SIZE}
          height={SQUARE_SIZE}
          fill={bgColor}
          stroke="#374151"
          strokeWidth="0.5"
          style={{ cursor: isLegalDest || canMove ? 'pointer' : 'default' }}
        />

        {/* Legal move indicator (dot for empty squares) */}
        {isLegalDest && !hasP1Piece && !hasP2Piece && (
          <circle
            cx={x + SQUARE_SIZE / 2}
            cy={y + SQUARE_SIZE / 2}
            r={8}
            fill="rgba(0, 0, 0, 0.2)"
          />
        )}

        {/* Pieces */}
        {(hasP1Piece || hasP2Piece) && (
          <g transform={`translate(${x}, ${y})`}>
            <Piece
              player={hasP1Piece ? 0 : 1}
              hasBall={hasP1Ball || hasP2Ball}
              isSelected={isSelected}
              onClick={canMove || isSelected ? () => onSquareClick(square) : undefined}
            />
          </g>
        )}
      </g>
    );
  };

  // Generate all squares
  const squares = [];
  for (let row = 0; row < BOARD_ROWS; row++) {
    for (let col = 0; col < BOARD_COLS; col++) {
      squares.push(renderSquare(row, col));
    }
  }

  return (
    <div className="inline-block">
      {/* Player indicator */}
      <div className="mb-2 text-center">
        <span
          className={`inline-block px-3 py-1 rounded text-white text-sm font-medium ${
            currentPlayer === 0 ? 'bg-blue-500' : 'bg-red-500'
          }`}
        >
          {currentPlayer === 0 ? 'Blue' : 'Red'}'s Turn
        </span>
      </div>

      <svg
        width={BOARD_WIDTH}
        height={BOARD_HEIGHT}
        viewBox={`0 0 ${BOARD_WIDTH} ${BOARD_HEIGHT}`}
        className="border-2 border-gray-700 rounded"
      >
        {squares}

        {/* File labels (a-g) */}
        {Array.from({ length: BOARD_COLS }, (_, col) => {
          const visualCol = flipped ? BOARD_COLS - 1 - col : col;
          const label = String.fromCharCode('a'.charCodeAt(0) + col);
          return (
            <text
              key={`file-${col}`}
              x={visualCol * SQUARE_SIZE + SQUARE_SIZE / 2}
              y={BOARD_HEIGHT - 4}
              textAnchor="middle"
              fontSize="10"
              fill="#374151"
            >
              {label}
            </text>
          );
        })}

        {/* Rank labels (1-8) */}
        {Array.from({ length: BOARD_ROWS }, (_, row) => {
          const visualRow = flipped ? BOARD_ROWS - 1 - row : row;
          const label = row + 1;
          return (
            <text
              key={`rank-${row}`}
              x={4}
              y={(BOARD_ROWS - 1 - visualRow) * SQUARE_SIZE + SQUARE_SIZE / 2 + 4}
              fontSize="10"
              fill="#374151"
            >
              {label}
            </text>
          );
        })}
      </svg>
    </div>
  );
}
