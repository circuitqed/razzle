/**
 * Board symmetry transformations for Razzle Dazzle.
 *
 * 180-degree rotation is used to normalize player perspective.
 * Policy arrays need to be rotated when evaluating from player 1's perspective.
 */

import { ROWS, COLS, NUM_SQUARES, NUM_ACTIONS, END_TURN_ACTION } from './bitboard';

function rotateSquare180(sq: number): number {
  const row = Math.floor(sq / COLS);
  const col = sq % COLS;
  const newRow = ROWS - 1 - row;
  const newCol = COLS - 1 - col;
  return newRow * COLS + newCol;
}

function rotateMove180(move: number): number {
  if (move === END_TURN_ACTION) return move;
  const src = Math.floor(move / NUM_SQUARES);
  const dst = move % NUM_SQUARES;
  const newSrc = rotateSquare180(src);
  const newDst = rotateSquare180(dst);
  return newSrc * NUM_SQUARES + newDst;
}

// Precompute move rotation map: MOVE_ROTATION_MAP[i] = rotated index of action i
export const MOVE_ROTATION_MAP: Int32Array = new Int32Array(NUM_ACTIONS);

for (let i = 0; i < NUM_ACTIONS; i++) {
  MOVE_ROTATION_MAP[i] = rotateMove180(i);
}

/**
 * Rotate a policy array 180 degrees using the precomputed map.
 * Input/output: Float32Array of length NUM_ACTIONS (3137).
 */
export function rotatePolicy180(policy: Float32Array): Float32Array {
  const rotated = new Float32Array(NUM_ACTIONS);
  for (let i = 0; i < NUM_ACTIONS; i++) {
    rotated[i] = policy[MOVE_ROTATION_MAP[i]];
  }
  return rotated;
}
