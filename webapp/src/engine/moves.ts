/**
 * Move generation for Razzle Dazzle.
 *
 * Handles knight moves and ball passes with proper rule enforcement.
 */

import {
  NUM_SQUARES,
  KNIGHT_ATTACKS,
  bit,
  iterBits,
  sqToRowCol,
  isValidSq,
  rowcolToSq,
  END_TURN_MOVE,
} from './bitboard';
import type { EngineState } from './state';
import { getOccupied, getEmpty } from './state';

export function encodeMove(src: number, dst: number): number {
  return src * NUM_SQUARES + dst;
}

export function decodeMove(move: number): [number, number] {
  return [Math.floor(move / NUM_SQUARES), move % NUM_SQUARES];
}

/** Generate all legal knight moves for the current player. */
export function getKnightMoves(state: EngineState): number[] {
  const moves: number[] = [];
  const p = state.currentPlayer;
  const movablePieces = state.pieces[p] & ~state.balls[p];
  const empty = getEmpty(state);

  for (const src of iterBits(movablePieces)) {
    const targets = KNIGHT_ATTACKS[src] & empty;
    for (const dst of iterBits(targets)) {
      moves.push(encodeMove(src, dst));
    }
  }
  return moves;
}

/** Generate all legal ball passes for the current player. */
export function getPassMoves(state: EngineState): number[] {
  const moves: number[] = [];
  const p = state.currentPlayer;
  const ballPos = state.balls[p];
  if (ballPos === 0n) return moves;

  // Find ball square
  let ballSq = -1;
  for (const sq of iterBits(ballPos)) {
    ballSq = sq;
    break;
  }
  if (ballSq === -1) return moves;

  const myPieces = state.pieces[p];
  const occupied = getOccupied(state);
  const validReceivers = myPieces & ~state.touchedMask;

  const directions: [number, number][] = [
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1], // N, S, E, W
    [1, 1],
    [1, -1],
    [-1, 1],
    [-1, -1], // Diagonals
  ];

  const [ballRow, ballCol] = sqToRowCol(ballSq);

  for (const [dr, dc] of directions) {
    let r = ballRow + dr;
    let c = ballCol + dc;
    while (isValidSq(r, c)) {
      const sq = rowcolToSq(r, c);
      const sqBit = bit(sq);
      if (occupied & sqBit) {
        // Hit something - if it's our piece and can receive, it's a valid pass
        if (validReceivers & sqBit) {
          moves.push(encodeMove(ballSq, sq));
        }
        break; // Can't pass through pieces
      }
      r += dr;
      c += dc;
    }
  }

  return moves;
}

/**
 * Check if current player MUST pass (opponent moved adjacent to ball).
 */
export function mustPass(state: EngineState): boolean {
  if (state.lastKnightDst < 0) return false;

  const p = state.currentPlayer;
  const ballPos = state.balls[p];
  if (ballPos === 0n) return false;

  let ballSq = -1;
  for (const sq of iterBits(ballPos)) {
    ballSq = sq;
    break;
  }
  if (ballSq === -1) return false;

  const [ballRow, ballCol] = sqToRowCol(ballSq);
  const [dstRow, dstCol] = sqToRowCol(state.lastKnightDst);

  const rowDiff = Math.abs(ballRow - dstRow);
  const colDiff = Math.abs(ballCol - dstCol);

  return rowDiff <= 1 && colDiff <= 1 && (rowDiff > 0 || colDiff > 0);
}

/** Get all legal moves for the current player. */
export function getLegalMoves(state: EngineState): number[] {
  const forcedPass = mustPass(state);
  const passMoves = getPassMoves(state);

  if (forcedPass && passMoves.length > 0) {
    return passMoves;
  }

  if (state.hasPassed) {
    return [...passMoves, END_TURN_MOVE];
  }

  return [...passMoves, ...getKnightMoves(state)];
}

/**
 * Get a boolean mask indicating which moves are legal.
 * Returns array of length NUM_SQUARES * NUM_SQUARES (3136).
 * END_TURN (-1) is NOT included in this mask; check separately.
 */
export function getMoveMask(state: EngineState): boolean[] {
  const mask = new Array<boolean>(NUM_SQUARES * NUM_SQUARES).fill(false);
  for (const move of getLegalMoves(state)) {
    if (move >= 0) {
      mask[move] = true;
    }
  }
  return mask;
}
