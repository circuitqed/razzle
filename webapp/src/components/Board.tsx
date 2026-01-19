import { useMemo, useState, useEffect, useRef } from 'react';
import type { BoardState, Player } from '../types';
import {
  BOARD_COLS,
  BOARD_ROWS,
  hasPiece,
  decodeMove,
} from '../types';
import Piece from './Piece';

interface LastMove {
  from: number;
  to: number;
}

interface BoardProps {
  board: BoardState;
  currentPlayer: Player;
  legalMoves: number[];
  selectedSquare: number | null;
  onSquareClick: (square: number) => void;
  onDragMove?: (from: number, to: number) => void;
  flipped?: boolean;
  touchedMask: number; // Bitboard of ineligible pieces
  mustPass?: boolean; // Forced pass situation
  lastMove?: LastMove | null; // Last move for highlighting
}

const SQUARE_SIZE = 50;
const BOARD_WIDTH = BOARD_COLS * SQUARE_SIZE;
const BOARD_HEIGHT = BOARD_ROWS * SQUARE_SIZE;
const ANIMATION_DURATION = 200; // ms

// Helper to get visual position for a square
function getSquarePosition(square: number, flipped: boolean) {
  const row = Math.floor(square / BOARD_COLS);
  const col = square % BOARD_COLS;
  const visualRow = flipped ? BOARD_ROWS - 1 - row : row;
  const visualCol = flipped ? BOARD_COLS - 1 - col : col;
  return {
    x: visualCol * SQUARE_SIZE,
    y: (BOARD_ROWS - 1 - visualRow) * SQUARE_SIZE,
  };
}

interface AnimatingPiece {
  fromSquare: number;
  toSquare: number;
  player: Player;
  hasBall: boolean;
  progress: number;
}

export default function Board({
  board,
  currentPlayer,
  legalMoves,
  selectedSquare,
  onSquareClick,
  onDragMove,
  flipped = false,
  touchedMask,
  mustPass = false,
  lastMove = null,
}: BoardProps) {
  // Drag state - drag only activates after moving past threshold
  const [pendingDragSquare, setPendingDragSquare] = useState<number | null>(null);
  const [draggingSquare, setDraggingSquare] = useState<number | null>(null);
  const [dragPosition, setDragPosition] = useState<{ x: number; y: number } | null>(null);
  const dragStartRef = useRef<{ x: number; y: number; pointerId: number } | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const DRAG_THRESHOLD = 8; // Pixels moved before drag activates

  // Handle pointer down - just record start position, don't interfere with clicks
  const handlePiecePointerDown = (square: number, e: React.PointerEvent) => {
    if (!movablePieces.has(square)) return;

    const hasP1 = hasPiece(board.p1_pieces, square) || hasPiece(board.p1_ball, square);
    const piecePlayer = hasP1 ? 0 : 1;
    if (piecePlayer !== currentPlayer) return;

    const svg = svgRef.current;
    if (svg) {
      const rect = svg.getBoundingClientRect();
      dragStartRef.current = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
        pointerId: e.pointerId,
      };
      setPendingDragSquare(square);
    }
  };

  // Handle pointer move - only start drag after threshold
  const handlePointerMove = (e: React.PointerEvent) => {
    const svg = svgRef.current;
    if (!svg) return;

    const rect = svg.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // If already dragging, update position
    if (draggingSquare !== null) {
      setDragPosition({ x, y });
      return;
    }

    // Check if we should start dragging
    if (pendingDragSquare !== null && dragStartRef.current) {
      const dx = x - dragStartRef.current.x;
      const dy = y - dragStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) > DRAG_THRESHOLD) {
        // Start dragging - capture pointer now
        (e.target as Element).setPointerCapture(dragStartRef.current.pointerId);
        setDraggingSquare(pendingDragSquare);
        setDragPosition({ x, y });
        setPendingDragSquare(null);
      }
    }
  };

  // Handle pointer up - complete drag if active
  const handlePointerUp = (e: React.PointerEvent) => {
    // Clear pending drag state
    setPendingDragSquare(null);
    dragStartRef.current = null;

    // If not actively dragging, do nothing (let click events handle it)
    if (draggingSquare === null) return;

    const svg = svgRef.current;
    if (svg && onDragMove) {
      const rect = svg.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      const visualCol = Math.floor(x / SQUARE_SIZE);
      const visualRow = BOARD_ROWS - 1 - Math.floor(y / SQUARE_SIZE);
      const col = flipped ? BOARD_COLS - 1 - visualCol : visualCol;
      const row = flipped ? BOARD_ROWS - 1 - visualRow : visualRow;

      if (row >= 0 && row < BOARD_ROWS && col >= 0 && col < BOARD_COLS) {
        const targetSquare = row * BOARD_COLS + col;
        const destinations = getLegalDestinationsForSquare(draggingSquare);

        if (destinations.has(targetSquare)) {
          onDragMove(draggingSquare, targetSquare);
        }
      }
    }

    setDraggingSquare(null);
    setDragPosition(null);
  };

  // Animation state
  const [animatingPiece, setAnimatingPiece] = useState<AnimatingPiece | null>(null);
  const prevLastMoveRef = useRef<LastMove | null>(null);
  const animationRef = useRef<number | null>(null);

  // Trigger animation when lastMove changes
  useEffect(() => {
    const prevMove = prevLastMoveRef.current;
    prevLastMoveRef.current = lastMove;

    // If lastMove changed and there's a new move, animate it
    if (lastMove && (!prevMove || prevMove.from !== lastMove.from || prevMove.to !== lastMove.to)) {
      // Determine what piece moved and if it has a ball
      const fromSquare = lastMove.from;
      const toSquare = lastMove.to;

      // Check if this was a piece move or ball pass by looking at current board state
      // (the move already happened, so we check the destination)
      const hasP1AtDest = hasPiece(board.p1_pieces, toSquare) || hasPiece(board.p1_ball, toSquare);
      const hasP1Ball = hasPiece(board.p1_ball, toSquare);
      const hasP2Ball = hasPiece(board.p2_ball, toSquare);

      const player: Player = hasP1AtDest ? 0 : 1;
      const hasBall = hasP1Ball || hasP2Ball;

      // Start animation
      const startTime = performance.now();

      const animate = (currentTime: number) => {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / ANIMATION_DURATION, 1);

        setAnimatingPiece({
          fromSquare,
          toSquare,
          player,
          hasBall,
          progress,
        });

        if (progress < 1) {
          animationRef.current = requestAnimationFrame(animate);
        } else {
          setAnimatingPiece(null);
        }
      };

      // Cancel any existing animation
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }

      animationRef.current = requestAnimationFrame(animate);
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [lastMove, board]);

  // Get destinations for the selected piece (knight moves and passes)
  const legalDestinations = useMemo(() => {
    if (selectedSquare === null) return new Set<number>();
    return new Set(
      legalMoves
        .filter(m => m >= 0) // Exclude end turn (-1)
        .map(decodeMove)
        .filter(({ src }) => src === selectedSquare)
        .map(({ dst }) => dst)
    );
  }, [selectedSquare, legalMoves]);

  // Get all squares with pieces that can move/pass
  const movablePieces = useMemo(() => {
    return new Set(
      legalMoves
        .filter(m => m >= 0)
        .map((m) => decodeMove(m).src)
    );
  }, [legalMoves]);

  // Get legal destinations for a specific source square
  const getLegalDestinationsForSquare = (square: number) => {
    return new Set(
      legalMoves
        .filter(m => m >= 0)
        .map(decodeMove)
        .filter(({ src }) => src === square)
        .map(({ dst }) => dst)
    );
  };

  // Legal destinations for dragging piece
  const dragDestinations = useMemo(() => {
    if (draggingSquare === null) return new Set<number>();
    return getLegalDestinationsForSquare(draggingSquare);
  }, [draggingSquare, legalMoves]);

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
    const isDragDest = dragDestinations.has(square);
    const canMove = movablePieces.has(square);
    const isLastMoveFrom = lastMove?.from === square;
    const isLastMoveTo = lastMove?.to === square;
    const isDragging = draggingSquare === square;

    // Check if this piece is ineligible to receive passes
    const isIneligible = hasPiece(touchedMask, square);

    // Determine background color
    let bgColor = isLight ? '#f0d9b5' : '#b58863';
    if (isSelected || isDragging) {
      bgColor = '#bbcb2b';
    } else if (isLegalDest || isDragDest) {
      bgColor = isLight ? '#acd490' : '#87b560';
    } else if (isLastMoveFrom || isLastMoveTo) {
      // Highlight last move with a subtle yellow tint
      bgColor = isLight ? '#f7e896' : '#c9b458';
    }

    // Goal zones (rows 0 and 7)
    if (row === 0) {
      bgColor = (isLegalDest || isDragDest) ? '#60a5fa' : (isLastMoveFrom || isLastMoveTo) ? '#93c5fd' : '#93c5fd';
    } else if (row === BOARD_ROWS - 1) {
      bgColor = (isLegalDest || isDragDest) ? '#f87171' : (isLastMoveFrom || isLastMoveTo) ? '#fca5a5' : '#fca5a5';
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
        {(isLegalDest || isDragDest) && !hasP1Piece && !hasP2Piece && (
          <circle
            cx={x + SQUARE_SIZE / 2}
            cy={y + SQUARE_SIZE / 2}
            r={8}
            fill="rgba(0, 0, 0, 0.2)"
          />
        )}

        {/* Pieces - hide piece only when actively dragging it */}
        {(hasP1Piece || hasP2Piece) &&
          !isDragging &&
          !(animatingPiece && animatingPiece.toSquare === square && animatingPiece.progress < 1) && (
          <g
            transform={`translate(${x}, ${y})`}
            onPointerDown={(e) => handlePiecePointerDown(square, e)}
            style={{ cursor: canMove ? 'grab' : 'default', touchAction: 'none' }}
          >
            <Piece
              player={hasP1Piece ? 0 : 1}
              hasBall={hasP1Ball || hasP2Ball}
              isSelected={isSelected}
              isIneligible={isIneligible}
              mustPass={mustPass && (hasP1Ball || hasP2Ball) && (hasP1Piece ? 0 : 1) === currentPlayer}
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
    <div className="inline-block w-full max-w-[350px] sm:max-w-none sm:w-auto">
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
        ref={svgRef}
        width="100%"
        height="auto"
        viewBox={`0 0 ${BOARD_WIDTH} ${BOARD_HEIGHT}`}
        className="border-2 border-gray-700 rounded w-full sm:w-[350px]"
        preserveAspectRatio="xMidYMid meet"
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
        style={{ touchAction: 'none' }}
      >
        {squares}

        {/* Dragging piece ghost */}
        {draggingSquare !== null && dragPosition && (() => {
          const hasP1 = hasPiece(board.p1_pieces, draggingSquare) || hasPiece(board.p1_ball, draggingSquare);
          const piecePlayer = hasP1 ? 0 : 1;
          const hasBall = hasPiece(board.p1_ball, draggingSquare) || hasPiece(board.p2_ball, draggingSquare);
          return (
            <g
              transform={`translate(${dragPosition.x - SQUARE_SIZE / 2}, ${dragPosition.y - SQUARE_SIZE / 2})`}
              style={{ pointerEvents: 'none', opacity: 0.8 }}
            >
              <Piece player={piecePlayer} hasBall={hasBall} />
            </g>
          );
        })()}

        {/* Animating piece overlay */}
        {animatingPiece && animatingPiece.progress < 1 && (() => {
          const fromPos = getSquarePosition(animatingPiece.fromSquare, flipped);
          const toPos = getSquarePosition(animatingPiece.toSquare, flipped);
          // Ease-out interpolation
          const t = 1 - Math.pow(1 - animatingPiece.progress, 3);
          const x = fromPos.x + (toPos.x - fromPos.x) * t;
          const y = fromPos.y + (toPos.y - fromPos.y) * t;
          return (
            <g transform={`translate(${x}, ${y})`}>
              <Piece
                player={animatingPiece.player}
                hasBall={animatingPiece.hasBall}
              />
            </g>
          );
        })()}

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
