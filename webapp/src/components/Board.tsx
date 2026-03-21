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
  onSquareDoubleClick?: (square: number) => void;
  onDragMove?: (from: number, to: number) => void;
  flipped?: boolean;
  touchedMask: string | number; // Bitboard of ineligible pieces (string for JS precision)
  mustPass?: boolean; // Forced pass situation
  lastMove?: LastMove | null; // Last move for highlighting
  lastTurnMoves?: LastMove[]; // All moves in last opponent turn (for multi-pass animation)
  animate?: boolean; // Whether to animate piece movement (default true)
  animationDuration?: number; // ms per segment (default ANIMATION_DURATION)
  suggestedMoves?: number[]; // Moves to highlight green; other legal moves shown gray
}

const SQUARE_SIZE = 50;
const BOARD_WIDTH = BOARD_COLS * SQUARE_SIZE;
const BOARD_HEIGHT = BOARD_ROWS * SQUARE_SIZE;
const LABEL_PAD_LEFT = 14; // Space for rank labels (1-8) outside the board
const LABEL_PAD_BOTTOM = 14; // Space for file labels (a-g) outside the board
const ANIMATION_DURATION = 350; // ms - all piece/ball animations

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
  toSquare: number; // Final destination (used for ball hiding at dest)
  player: Player;
  hasBall: boolean;
  isBallOnly: boolean; // True if this is a pass (only ball moves, piece stays)
  progress: number;
  fromOverride?: { x: number; y: number } | null; // SVG position override (for drag-drop)
  waypoints?: { x: number; y: number }[]; // Multi-pass path (all positions from start to end)
  passSquares?: number[]; // Square indices for each pass: [from, mid1, mid2, ..., to]
}

export default function Board({
  board,
  currentPlayer,
  legalMoves,
  selectedSquare,
  onSquareClick,
  onSquareDoubleClick,
  onDragMove,
  flipped = false,
  touchedMask,
  mustPass = false,
  lastMove = null,
  lastTurnMoves,
  animate = true,
  animationDuration = ANIMATION_DURATION,
  suggestedMoves,
}: BoardProps) {
  // Drag state - drag only activates after moving past threshold
  const [pendingDragSquare, setPendingDragSquare] = useState<number | null>(null);
  const [draggingSquare, setDraggingSquare] = useState<number | null>(null);
  const [dragPosition, setDragPosition] = useState<{ x: number; y: number } | null>(null);
  const dragStartRef = useRef<{ x: number; y: number; pointerId: number } | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const DRAG_THRESHOLD = 8; // Pixels moved before drag activates

  // Track SVG position of the last drag-drop so the animation starts from there
  const lastDragDropRef = useRef<{ from: number; to: number; x: number; y: number } | null>(null);

  // Suppress click events immediately after a drag-drop completes
  // (pointerup fires before click, so drag can complete and then click fires on the original square)
  const dragJustCompletedRef = useRef(false);

  // Convert client coordinates to SVG coordinates (accounts for viewBox offset and scale)
  const clientToSvg = (clientX: number, clientY: number): { x: number; y: number } | null => {
    const svg = svgRef.current;
    if (!svg) return null;
    const rect = svg.getBoundingClientRect();
    const viewBoxW = BOARD_WIDTH + LABEL_PAD_LEFT;
    const viewBoxH = BOARD_HEIGHT + LABEL_PAD_BOTTOM;
    return {
      x: (clientX - rect.left) / rect.width * viewBoxW - LABEL_PAD_LEFT,
      y: (clientY - rect.top) / rect.height * viewBoxH,
    };
  };

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
    const clientRelX = e.clientX - rect.left;
    const clientRelY = e.clientY - rect.top;

    // If already dragging, update position in SVG coords
    if (draggingSquare !== null) {
      const svgPos = clientToSvg(e.clientX, e.clientY);
      if (svgPos) setDragPosition(svgPos);
      return;
    }

    // Check if we should start dragging (threshold in CSS pixels)
    if (pendingDragSquare !== null && dragStartRef.current) {
      const dx = clientRelX - dragStartRef.current.x;
      const dy = clientRelY - dragStartRef.current.y;
      if (Math.sqrt(dx * dx + dy * dy) > DRAG_THRESHOLD) {
        (e.target as Element).setPointerCapture(dragStartRef.current.pointerId);
        setDraggingSquare(pendingDragSquare);
        const svgPos = clientToSvg(e.clientX, e.clientY);
        if (svgPos) setDragPosition(svgPos);
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

    // Mark that a drag just completed to suppress the subsequent click event
    dragJustCompletedRef.current = true;

    if (onDragMove) {
      const svgPos = clientToSvg(e.clientX, e.clientY);
      if (svgPos) {
        const visualCol = Math.floor(svgPos.x / SQUARE_SIZE);
        const visualRow = BOARD_ROWS - 1 - Math.floor(svgPos.y / SQUARE_SIZE);
        const col = flipped ? BOARD_COLS - 1 - visualCol : visualCol;
        const row = flipped ? BOARD_ROWS - 1 - visualRow : visualRow;

        if (row >= 0 && row < BOARD_ROWS && col >= 0 && col < BOARD_COLS) {
          const targetSquare = row * BOARD_COLS + col;
          const destinations = getLegalDestinationsForSquare(draggingSquare);

          if (destinations.has(targetSquare)) {
            // Save drop position so the animation starts from here, not the original square
            lastDragDropRef.current = {
              from: draggingSquare,
              to: targetSquare,
              x: svgPos.x - SQUARE_SIZE / 2,
              y: svgPos.y - SQUARE_SIZE / 2,
            };
            onDragMove(draggingSquare, targetSquare);
          }
        }
      }
    }

    setDraggingSquare(null);
    setDragPosition(null);
  };

  // Animation state
  const [animatingPiece, setAnimatingPiece] = useState<AnimatingPiece | null>(null);
  const prevLastMoveRef = useRef<LastMove | null>(null);
  const prevLastTurnMovesRef = useRef<LastMove[] | undefined>(undefined);
  const animationRef = useRef<number | null>(null);

  // Trigger animation when lastMove or lastTurnMoves changes
  useEffect(() => {
    const prevMove = prevLastMoveRef.current;
    prevLastMoveRef.current = lastMove;
    const prevTurnMoves = prevLastTurnMovesRef.current;
    prevLastTurnMovesRef.current = lastTurnMoves;

    if (!animate) return;

    // --- Multi-pass animation (ball travels through waypoints) ---
    const turnMovesChanged = lastTurnMoves && lastTurnMoves.length > 1 &&
      (!prevTurnMoves || prevTurnMoves.length !== lastTurnMoves.length ||
       prevTurnMoves.some((m, i) => m.from !== lastTurnMoves[i].from || m.to !== lastTurnMoves[i].to));

    if (turnMovesChanged && lastTurnMoves.length > 1) {
      const firstFrom = lastTurnMoves[0].from;
      const finalTo = lastTurnMoves[lastTurnMoves.length - 1].to;

      // Determine player from board state at final destination
      const hasP1AtDest = hasPiece(board.p1_pieces, finalTo) || hasPiece(board.p1_ball, finalTo);
      const player: Player = hasP1AtDest ? 0 : 1;

      // Build waypoints: start position + each destination
      const waypoints = [
        getSquarePosition(firstFrom, flipped),
        ...lastTurnMoves.map(m => getSquarePosition(m.to, flipped)),
      ];

      // Track square indices for progressive ineligibility display
      const passSquares = [firstFrom, ...lastTurnMoves.map(m => m.to)];

      const numSegments = lastTurnMoves.length;
      const totalDuration = animationDuration * numSegments;
      const startTime = performance.now();

      const animateFn = (currentTime: number) => {
        const progress = Math.min((currentTime - startTime) / totalDuration, 1);

        setAnimatingPiece({
          fromSquare: firstFrom,
          toSquare: finalTo,
          player,
          hasBall: true,
          isBallOnly: true,
          progress,
          waypoints,
          passSquares,
        });

        if (progress < 1) {
          animationRef.current = requestAnimationFrame(animateFn);
        } else {
          setAnimatingPiece(null);
        }
      };

      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      animationRef.current = requestAnimationFrame(animateFn);
      return;
    }

    // --- Single-move animation (existing behavior) ---
    if (lastMove && (!prevMove || prevMove.from !== lastMove.from || prevMove.to !== lastMove.to)) {
      const fromSquare = lastMove.from;
      const toSquare = lastMove.to;

      const hasP1AtDest = hasPiece(board.p1_pieces, toSquare) || hasPiece(board.p1_ball, toSquare);
      const hasP1Ball = hasPiece(board.p1_ball, toSquare);
      const hasP2Ball = hasPiece(board.p2_ball, toSquare);

      const hasP1PieceAtSource = hasPiece(board.p1_pieces, fromSquare);
      const hasP2PieceAtSource = hasPiece(board.p2_pieces, fromSquare);
      const isBallOnly = hasP1PieceAtSource || hasP2PieceAtSource;

      const player: Player = hasP1AtDest ? 0 : 1;
      const hasBall = hasP1Ball || hasP2Ball;

      let animFromOverride: { x: number; y: number } | null = null;
      if (
        lastDragDropRef.current &&
        lastDragDropRef.current.from === fromSquare &&
        lastDragDropRef.current.to === toSquare
      ) {
        animFromOverride = { x: lastDragDropRef.current.x, y: lastDragDropRef.current.y };
        lastDragDropRef.current = null;
      }

      const duration = animationDuration;
      const startTime = performance.now();

      const animateFn = (currentTime: number) => {
        const progress = Math.min((currentTime - startTime) / duration, 1);

        setAnimatingPiece({
          fromSquare,
          toSquare,
          player,
          hasBall,
          isBallOnly,
          progress,
          fromOverride: animFromOverride,
        });

        if (progress < 1) {
          animationRef.current = requestAnimationFrame(animateFn);
        } else {
          setAnimatingPiece(null);
        }
      };

      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      animationRef.current = requestAnimationFrame(animateFn);
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
        setAnimatingPiece(null);
      }
    };
  }, [lastMove, lastTurnMoves, board, animationDuration]);

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

  // Suggested destinations (for tutorial — green highlight vs gray for others)
  const suggestedDestinations = useMemo(() => {
    if (!suggestedMoves || selectedSquare === null) return null;
    return new Set(
      suggestedMoves
        .filter(m => m >= 0)
        .map(decodeMove)
        .filter(({ src }) => src === selectedSquare)
        .map(({ dst }) => dst)
    );
  }, [suggestedMoves, selectedSquare]);

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
    // During multi-pass animation, only show X after ball has left that square
    let isIneligible = hasPiece(touchedMask, square);
    if (isIneligible && animatingPiece?.passSquares && animatingPiece.progress < 1) {
      const pts = animatingPiece.passSquares;
      const numSeg = pts.length - 1;
      const completedSeg = Math.floor(animatingPiece.progress * numSeg);
      // Square is revealed only if ball has departed it (it's at index <= completedSeg)
      const sqIdx = pts.indexOf(square);
      if (sqIdx >= 0 && sqIdx > completedSeg) {
        isIneligible = false; // Ball hasn't left this square yet
      }
    }

    // Determine background color
    let bgColor = isLight ? '#f0d9b5' : '#b58863';
    if (isSelected || isDragging) {
      bgColor = '#bbcb2b';
    } else if (isLegalDest || isDragDest) {
      // If suggestedMoves is active, show suggested in green, others in gray
      const isSuggested = !suggestedDestinations || suggestedDestinations.has(square);
      bgColor = isSuggested
        ? (isLight ? '#acd490' : '#87b560')  // green
        : (isLight ? '#c8c8c8' : '#888888'); // gray
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
      <g
        key={square}
        onClick={() => {
          // Suppress click events that fire immediately after a drag completes
          if (dragJustCompletedRef.current) {
            dragJustCompletedRef.current = false;
            return;
          }
          onSquareClick(square);
        }}
        onDoubleClick={() => onSquareDoubleClick?.(square)}
      >
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

        {/* Pieces - hide piece only when actively dragging it or during knight move animation */}
        {(hasP1Piece || hasP2Piece) &&
          !isDragging &&
          !(animatingPiece && animatingPiece.toSquare === square && animatingPiece.progress < 1 && !animatingPiece.isBallOnly) && (
          <g
            transform={`translate(${x}, ${y})`}
            onPointerDown={(e) => handlePiecePointerDown(square, e)}
            style={{ cursor: canMove ? 'grab' : 'default', touchAction: 'none' }}
          >
            <Piece
              player={hasP1Piece ? 0 : 1}
              hasBall={
                (hasP1Ball || hasP2Ball) &&
                // During pass animation, hide ball at destination until animation completes
                !(animatingPiece && animatingPiece.toSquare === square && animatingPiece.isBallOnly && animatingPiece.progress < 1)
              }
              isSelected={isSelected}
              isIneligible={isIneligible}
              mustPass={mustPass && (hasP1Ball || hasP2Ball) && (hasP1Piece ? 0 : 1) === currentPlayer}
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
    <div className="inline-block w-full max-w-[400px] sm:max-w-none sm:w-auto">
      <svg
        ref={svgRef}
        width="100%"
        height="auto"
        viewBox={`${-LABEL_PAD_LEFT} 0 ${BOARD_WIDTH + LABEL_PAD_LEFT} ${BOARD_HEIGHT + LABEL_PAD_BOTTOM}`}
        className="w-full sm:w-[400px]"
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

        {/* Animating piece/ball overlay */}
        {animatingPiece && animatingPiece.progress < 1 && (() => {
          let x: number, y: number;

          try {
            if (animatingPiece.waypoints && animatingPiece.waypoints.length > 1) {
              // Multi-pass: interpolate along waypoint path
              const pts = animatingPiece.waypoints;
              const numSeg = pts.length - 1;
              const scaled = animatingPiece.progress * numSeg;
              const seg = Math.min(Math.floor(scaled), numSeg - 1);
              if (seg < 0 || !pts[seg] || !pts[seg + 1]) {
                return null; // Defensive: bail if waypoints are invalid
              }
              const segT = scaled - seg;
              const t = 1 - Math.pow(1 - segT, 3); // ease-out per segment
              x = pts[seg].x + (pts[seg + 1].x - pts[seg].x) * t;
              y = pts[seg].y + (pts[seg + 1].y - pts[seg].y) * t;
            } else {
              // Single move: from -> to
              const fromPos = animatingPiece.fromOverride ?? getSquarePosition(animatingPiece.fromSquare, flipped);
              const toPos = getSquarePosition(animatingPiece.toSquare, flipped);
              const t = 1 - Math.pow(1 - animatingPiece.progress, 3);
              x = fromPos.x + (toPos.x - fromPos.x) * t;
              y = fromPos.y + (toPos.y - fromPos.y) * t;
            }
          } catch (err) {
            console.error('[Board] Animation rendering error:', err, animatingPiece);
            return null; // Skip animation frame rather than crash
          }

          if (animatingPiece.isBallOnly) {
            // Pass animation - only the ball moves
            return (
              <g transform={`translate(${x + SQUARE_SIZE / 2}, ${y + SQUARE_SIZE / 2})`}>
                <circle
                  r="8"
                  fill="#fbbf24"
                  stroke="#92400e"
                  strokeWidth="1.5"
                />
              </g>
            );
          }

          // Knight move animation - whole piece moves
          return (
            <g transform={`translate(${x}, ${y})`}>
              <Piece
                player={animatingPiece.player}
                hasBall={animatingPiece.hasBall}
              />
            </g>
          );
        })()}

        {/* File labels (a-g) — below the board */}
        {Array.from({ length: BOARD_COLS }, (_, col) => {
          const visualCol = flipped ? BOARD_COLS - 1 - col : col;
          const label = String.fromCharCode('a'.charCodeAt(0) + col);
          return (
            <text
              key={`file-${col}`}
              x={visualCol * SQUARE_SIZE + SQUARE_SIZE / 2}
              y={BOARD_HEIGHT + LABEL_PAD_BOTTOM - 3}
              textAnchor="middle"
              fontSize="10"
              fill="#9ca3af"
            >
              {label}
            </text>
          );
        })}

        {/* Rank labels (1-8) — left of the board */}
        {Array.from({ length: BOARD_ROWS }, (_, row) => {
          const visualRow = flipped ? BOARD_ROWS - 1 - row : row;
          const label = row + 1;
          return (
            <text
              key={`rank-${row}`}
              x={-3}
              y={(BOARD_ROWS - 1 - visualRow) * SQUARE_SIZE + SQUARE_SIZE / 2 + 4}
              textAnchor="end"
              fontSize="10"
              fill="#9ca3af"
            >
              {label}
            </text>
          );
        })}
      </svg>
    </div>
  );
}
