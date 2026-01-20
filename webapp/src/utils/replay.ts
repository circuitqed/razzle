/**
 * Utilities for game replay and position reconstruction.
 *
 * This module provides client-side move application to avoid
 * repeated server calls during replay navigation.
 */

import { BoardState, decodeMove } from '../types';

// Initial board configuration
// Blue (P1) has 5 pieces on row 0 (b1-f1), ball on d1
// Red (P2) has 5 pieces on row 7 (b8-f8), ball on d8
// Board is 8 rows x 7 columns (56 squares)
// Square index = row * 7 + col (a=0, b=1, c=2, d=3, e=4, f=5, g=6)

// Use BigInt literals for precise values, then convert to string
const INITIAL_P1_PIECES = '62';        // bits 1-5: b1,c1,d1,e1,f1
const INITIAL_P1_BALL = '8';           // bit 3: d1
const INITIAL_P2_PIECES = '34902897112121344';  // bits 50-54: b8,c8,d8,e8,f8
const INITIAL_P2_BALL = '4503599627370496';     // bit 52: d8

export interface ReplayState {
  board: BoardState;
  currentPlayer: number;
  touchedMask: string;  // String for JS precision
  hasPassed: boolean;
  ply: number;
  lastKnightDst: number;
}

export function getInitialState(): ReplayState {
  return {
    board: {
      p1_pieces: INITIAL_P1_PIECES,
      p1_ball: INITIAL_P1_BALL,
      p2_pieces: INITIAL_P2_PIECES,
      p2_ball: INITIAL_P2_BALL,
    },
    currentPlayer: 0,
    touchedMask: '0',
    hasPassed: false,
    ply: 0,
    lastKnightDst: -1,
  };
}

/**
 * Apply a move to the replay state and return a new state.
 * This is a simplified version - doesn't validate legality.
 */
export function applyMove(state: ReplayState, move: number): ReplayState {
  // End turn move
  if (move === -1) {
    return {
      ...state,
      currentPlayer: 1 - state.currentPlayer,
      touchedMask: '0',
      hasPassed: false,
      ply: state.ply + 1,
      lastKnightDst: -1,
    };
  }

  const { src, dst } = decodeMove(move);
  const player = state.currentPlayer;

  // Convert to BigInt for bitwise operations
  const srcMask = BigInt(1) << BigInt(src);
  const dstMask = BigInt(1) << BigInt(dst);

  const p1Pieces = BigInt(state.board.p1_pieces);
  const p1Ball = BigInt(state.board.p1_ball);
  const p2Pieces = BigInt(state.board.p2_pieces);
  const p2Ball = BigInt(state.board.p2_ball);

  // Determine if this is a ball pass or knight move
  const isBallMove = player === 0
    ? (p1Ball & srcMask) !== BigInt(0)
    : (p2Ball & srcMask) !== BigInt(0);

  let newP1Pieces = p1Pieces;
  let newP1Ball = p1Ball;
  let newP2Pieces = p2Pieces;
  let newP2Ball = p2Ball;
  let newTouchedMask = BigInt(state.touchedMask);
  let newHasPassed = state.hasPassed;
  let turnEnds = false;
  let newLastKnightDst = state.lastKnightDst;

  if (isBallMove) {
    // Ball pass - move ball to destination, mark both squares as touched
    if (player === 0) {
      newP1Ball = (newP1Ball & ~srcMask) | dstMask;
    } else {
      newP2Ball = (newP2Ball & ~srcMask) | dstMask;
    }
    newTouchedMask = newTouchedMask | srcMask | dstMask;
    newHasPassed = true;
  } else {
    // Knight move - move piece and its ball if it has one
    if (player === 0) {
      newP1Pieces = (newP1Pieces & ~srcMask) | dstMask;
      if ((newP1Ball & srcMask) !== BigInt(0)) {
        newP1Ball = (newP1Ball & ~srcMask) | dstMask;
      }
    } else {
      newP2Pieces = (newP2Pieces & ~srcMask) | dstMask;
      if ((newP2Ball & srcMask) !== BigInt(0)) {
        newP2Ball = (newP2Ball & ~srcMask) | dstMask;
      }
    }
    // Knight move ends the turn (unless in forced pass situation handled by server)
    turnEnds = true;
    newLastKnightDst = dst;
  }

  return {
    board: {
      p1_pieces: String(newP1Pieces),
      p1_ball: String(newP1Ball),
      p2_pieces: String(newP2Pieces),
      p2_ball: String(newP2Ball),
    },
    currentPlayer: turnEnds ? 1 - player : player,
    touchedMask: turnEnds ? '0' : String(newTouchedMask),
    hasPassed: turnEnds ? false : newHasPassed,
    ply: state.ply + 1,
    lastKnightDst: turnEnds ? newLastKnightDst : state.lastKnightDst,
  };
}

/**
 * Reconstruct all positions from a list of moves.
 * Returns an array of states, one for each position in the game
 * (including the initial position).
 */
export function reconstructPositions(moves: number[]): ReplayState[] {
  const positions: ReplayState[] = [getInitialState()];
  let state = getInitialState();

  for (const move of moves) {
    state = applyMove(state, move);
    positions.push(state);
  }

  return positions;
}

/**
 * Get the state at a specific ply (move index).
 */
export function replayToPosition(moves: number[], ply: number): ReplayState {
  let state = getInitialState();
  const movesToApply = moves.slice(0, ply);

  for (const move of movesToApply) {
    state = applyMove(state, move);
  }

  return state;
}

/**
 * Get the last move info (from/to squares) for a given position.
 */
export function getLastMoveAtPosition(moves: number[], ply: number): { from: number; to: number } | null {
  if (ply === 0) return null;

  const move = moves[ply - 1];
  if (move === -1) return null;

  const { src, dst } = decodeMove(move);
  return { from: src, to: dst };
}
