/**
 * Game state representation for Razzle Dazzle.
 *
 * Uses BigInt bitboards for efficient state manipulation.
 */

import {
  NUM_SQUARES,
  VALID_MASK,
  P1_START_PIECES,
  P1_START_BALL,
  P2_START_PIECES,
  P2_START_BALL,
  ROW_1_MASK,
  ROW_8_MASK,
  bit,
} from './bitboard';

export interface EngineState {
  pieces: [bigint, bigint]; // [p1_pieces, p2_pieces]
  balls: [bigint, bigint]; // [p1_ball, p2_ball]
  currentPlayer: number; // 0 or 1
  touchedMask: bigint; // Pieces ineligible to receive passes
  hasPassed: boolean; // Whether a pass has been made this turn
  lastKnightDst: number; // Destination of opponent's last knight move, or -1
  ply: number;
}

export function newGame(): EngineState {
  return {
    pieces: [P1_START_PIECES, P2_START_PIECES],
    balls: [P1_START_BALL, P2_START_BALL],
    currentPlayer: 0,
    touchedMask: 0n,
    hasPassed: false,
    lastKnightDst: -1,
    ply: 0,
  };
}

export function copyState(s: EngineState): EngineState {
  return {
    pieces: [s.pieces[0], s.pieces[1]],
    balls: [s.balls[0], s.balls[1]],
    currentPlayer: s.currentPlayer,
    touchedMask: s.touchedMask,
    hasPassed: s.hasPassed,
    lastKnightDst: s.lastKnightDst,
    ply: s.ply,
  };
}

export function getOccupied(s: EngineState): bigint {
  return s.pieces[0] | s.pieces[1] | s.balls[0] | s.balls[1];
}

export function getEmpty(s: EngineState): bigint {
  return ~getOccupied(s) & VALID_MASK;
}

export function isTerminal(s: EngineState): boolean {
  if (s.balls[0] & ROW_8_MASK) return true;
  if (s.balls[1] & ROW_1_MASK) return true;
  if (s.ply > 200) return true;
  return false;
}

export function getWinner(s: EngineState): number | null {
  if (s.balls[0] & ROW_8_MASK) return 0;
  if (s.balls[1] & ROW_1_MASK) return 1;
  if (s.ply > 200) return 1 - s.currentPlayer;
  return null;
}

export function getResult(s: EngineState, player: number): number {
  const winner = getWinner(s);
  if (winner === null) return 0.5;
  return winner === player ? 1.0 : 0.0;
}

/**
 * Apply a move to the state IN PLACE.
 *
 * Move encoding:
 * - Knight move: src * 56 + dst
 * - Pass: src * 56 + dst (src is ball position)
 * - End turn: -1
 */
export function applyMove(s: EngineState, move: number): void {
  if (move === -1) {
    // End turn: switch player, do NOT reset touchedMask
    s.currentPlayer = 1 - s.currentPlayer;
    s.hasPassed = false;
    s.lastKnightDst = -1;
    s.ply++;
    return;
  }

  const src = Math.floor(move / NUM_SQUARES);
  const dst = move % NUM_SQUARES;
  const srcBit = bit(src);
  const dstBit = bit(dst);
  const p = s.currentPlayer;

  if (s.balls[p] & srcBit) {
    // Ball pass: move ball from src to dst
    s.balls[p] = dstBit;
    s.touchedMask |= srcBit | dstBit;
    s.hasPassed = true;
  } else {
    // Knight move: move piece from src to dst
    s.pieces[p] = (s.pieces[p] & ~srcBit) | dstBit;
    // Clear ineligibility only for the source bit
    s.touchedMask &= ~srcBit;
    s.lastKnightDst = dst;
    s.currentPlayer = 1 - p;
    s.hasPassed = false;
    s.ply++;
  }
}
