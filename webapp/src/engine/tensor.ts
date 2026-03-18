/**
 * Neural network tensor conversion for Razzle Dazzle.
 *
 * Converts an EngineState to a Float32Array suitable for ONNX inference.
 * Output shape: (7, 8, 7) flattened = 392 floats in CHW order.
 */

import { ROWS, COLS } from './bitboard';
import { iterBits } from './bitboard';
import type { EngineState } from './state';

/**
 * Convert state to neural network input tensor.
 *
 * The board is always presented from the current player's perspective:
 * - Current player's pieces at the "bottom" (low row indices)
 * - Goal at the "top" (high row indices)
 * For player 1, the entire tensor is rotated 180 degrees.
 *
 * Planes:
 *  0: Current player's pieces
 *  1: Current player's ball
 *  2: Opponent's pieces
 *  3: Opponent's ball
 *  4: Touched mask
 *  5: Always 1 (reserved)
 *  6: Has passed indicator (all 1s if hasPassed, 0s otherwise)
 *
 * Returns Float32Array of length 392 (7 * 8 * 7) in CHW order.
 */
export function stateToTensor(state: EngineState, out?: Float32Array): Float32Array {
  const data = out ?? new Float32Array(7 * ROWS * COLS);
  if (out) data.fill(0);
  const p = state.currentPlayer;
  const opp = 1 - p;

  // Helper: set a bit in the tensor
  const set = (plane: number, row: number, col: number) => {
    data[plane * ROWS * COLS + row * COLS + col] = 1.0;
  };

  // Plane 0: Current player's pieces
  for (const sq of iterBits(state.pieces[p])) {
    const row = Math.floor(sq / COLS);
    const col = sq % COLS;
    set(0, row, col);
  }

  // Plane 1: Current player's ball
  for (const sq of iterBits(state.balls[p])) {
    const row = Math.floor(sq / COLS);
    const col = sq % COLS;
    set(1, row, col);
  }

  // Plane 2: Opponent's pieces
  for (const sq of iterBits(state.pieces[opp])) {
    const row = Math.floor(sq / COLS);
    const col = sq % COLS;
    set(2, row, col);
  }

  // Plane 3: Opponent's ball
  for (const sq of iterBits(state.balls[opp])) {
    const row = Math.floor(sq / COLS);
    const col = sq % COLS;
    set(3, row, col);
  }

  // Plane 4: Touched mask
  for (const sq of iterBits(state.touchedMask)) {
    const row = Math.floor(sq / COLS);
    const col = sq % COLS;
    set(4, row, col);
  }

  // Plane 5: Always 1
  const plane5Offset = 5 * ROWS * COLS;
  for (let i = 0; i < ROWS * COLS; i++) {
    data[plane5Offset + i] = 1.0;
  }

  // Plane 6: Has passed indicator
  if (state.hasPassed) {
    const plane6Offset = 6 * ROWS * COLS;
    for (let i = 0; i < ROWS * COLS; i++) {
      data[plane6Offset + i] = 1.0;
    }
  }

  // Rotate 180 for player 1: flip both spatial axes
  if (p === 1) {
    rotateTensor180InPlace(data);
  }

  return data;
}

/**
 * Rotate tensor 180 degrees in place.
 * For each plane, (row, col) -> (ROWS-1-row, COLS-1-col).
 */
function rotateTensor180InPlace(data: Float32Array): void {
  const planeSize = ROWS * COLS;
  for (let plane = 0; plane < 7; plane++) {
    const offset = plane * planeSize;
    // Swap elements symmetrically: element at index i swaps with element at (planeSize - 1 - i)
    for (let i = 0; i < Math.floor(planeSize / 2); i++) {
      const j = planeSize - 1 - i;
      const tmp = data[offset + i];
      data[offset + i] = data[offset + j];
      data[offset + j] = tmp;
    }
  }
}
