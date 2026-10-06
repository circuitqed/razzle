/**
 * Neural network tensor conversion for Razzle Dazzle.
 *
 * Converts an EngineState to a Float32Array suitable for ONNX inference.
 * Output shape: (planes, 8, 7) flattened in CHW order — 7 planes (392 floats)
 * for v1 networks, 9 planes (504 floats) for v2.
 */

import { ROWS, COLS } from './bitboard';
import { iterBits } from './bitboard';
import type { EngineState } from './state';
import { mustPass, getPassMoves } from './moves';

/**
 * True if the side to move must pass now: opponent's last knight move landed
 * next to our ball, we haven't passed yet this turn, and a pass exists.
 * Mirrors Python GameState.is_forced_pass() / C razzle_state_extra_planes.
 */
export function isForcedPass(state: EngineState): boolean {
  return !state.hasPassed && mustPass(state) && getPassMoves(state).length > 0;
}

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
 * v2 networks (numPlanes = 9) add:
 *  7: Opponent's last knight-move destination (one-hot; all 0 if lastKnightDst = -1)
 *  8: Forced pass (all 1s if the side to move must pass now, see isForcedPass)
 *
 * @param numPlanes 7 or 9. Defaults to out.length / 56 when `out` is given, else 7.
 * Returns Float32Array of length numPlanes * 56 in CHW order.
 */
export function stateToTensor(state: EngineState, out?: Float32Array, numPlanes?: number): Float32Array {
  const planes = numPlanes ?? (out ? out.length / (ROWS * COLS) : 7);
  if (planes !== 7 && planes !== 9) {
    throw new Error(`stateToTensor: numPlanes must be 7 or 9, got ${planes}`);
  }
  if (out && out.length !== planes * ROWS * COLS) {
    throw new Error(`stateToTensor: out has ${out.length} floats, expected ${planes * ROWS * COLS}`);
  }
  const data = out ?? new Float32Array(planes * ROWS * COLS);
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

  if (planes === 9) {
    // Plane 7: opponent's last knight destination (in board coordinates;
    // rotated together with the other planes below)
    if (state.lastKnightDst >= 0) {
      const sq = state.lastKnightDst;
      set(7, Math.floor(sq / COLS), sq % COLS);
    }
    // Plane 8: forced pass
    if (isForcedPass(state)) {
      data.fill(1.0, 8 * ROWS * COLS, 9 * ROWS * COLS);
    }
  }

  // Rotate 180 for player 1: flip both spatial axes
  if (p === 1) {
    rotateTensor180InPlace(data, planes);
  }

  return data;
}

/**
 * Rotate tensor 180 degrees in place.
 * For each plane, (row, col) -> (ROWS-1-row, COLS-1-col).
 */
function rotateTensor180InPlace(data: Float32Array, numPlanes: number): void {
  const planeSize = ROWS * COLS;
  for (let plane = 0; plane < numPlanes; plane++) {
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
