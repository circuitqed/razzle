/**
 * Bitboard utilities for Razzle Dazzle.
 *
 * Board layout (8 rows x 7 cols = 56 squares):
 *
 *   8 | 49 50 51 52 53 54 55
 *   7 | 42 43 44 45 46 47 48
 *   6 | 35 36 37 38 39 40 41
 *   5 | 28 29 30 31 32 33 34
 *   4 | 21 22 23 24 25 26 27
 *   3 | 14 15 16 17 18 19 20
 *   2 |  7  8  9 10 11 12 13
 *   1 |  0  1  2  3  4  5  6
 *     +---------------------
 *        a  b  c  d  e  f  g
 *
 * Square index = row * 7 + col (row 0 = rank 1, col 0 = file a)
 * All bitboard operations use BigInt (56-bit boards exceed JS Number precision).
 */

export const ROWS = 8;
export const COLS = 7;
export const NUM_SQUARES = ROWS * COLS; // 56
export const NUM_ACTIONS = NUM_SQUARES * NUM_SQUARES + 1; // 3137
export const END_TURN_ACTION = NUM_SQUARES * NUM_SQUARES; // 3136
export const END_TURN_MOVE = -1;

export const VALID_MASK = (1n << 56n) - 1n;

// Starting positions
export const P1_START_PIECES = 0b0111110n; // bits 1-5 (b1-f1)
export const P1_START_BALL = 0b0001000n; // bit 3 (d1)
export const P2_START_PIECES = P1_START_PIECES << 49n; // row 8
export const P2_START_BALL = P1_START_BALL << 49n; // d8

// Goal rows
export const ROW_1_MASK = 0b1111111n; // bits 0-6
export const ROW_8_MASK = ROW_1_MASK << 49n; // bits 49-55

// Knight move deltas (row_delta, col_delta)
const KNIGHT_DELTAS: [number, number][] = [
  [-2, -1],
  [-2, 1],
  [-1, -2],
  [-1, 2],
  [1, -2],
  [1, 2],
  [2, -1],
  [2, 1],
];

// Utility functions
export function sqToRowCol(sq: number): [number, number] {
  return [Math.floor(sq / COLS), sq % COLS];
}

export function rowcolToSq(row: number, col: number): number {
  return row * COLS + col;
}

export function isValidSq(row: number, col: number): boolean {
  return row >= 0 && row < ROWS && col >= 0 && col < COLS;
}

export function bit(sq: number): bigint {
  return 1n << BigInt(sq);
}

export function popcount(bb: bigint): number {
  let count = 0;
  let v = bb;
  while (v !== 0n) {
    v &= v - 1n;
    count++;
  }
  return count;
}

export function lsb(bb: bigint): number {
  if (bb === 0n) return -1;
  // (bb & -bb) isolates lowest bit
  const isolated = bb & -bb;
  // Count trailing zeros
  let idx = 0;
  let v = isolated;
  while (v > 1n) {
    v >>= 1n;
    idx++;
  }
  return idx;
}

export function* iterBits(bb: bigint): Generator<number> {
  let v = bb;
  while (v !== 0n) {
    const sq = lsb(v);
    yield sq;
    v &= v - 1n; // Clear LSB
  }
}

export function sqToAlgebraic(sq: number): string {
  const [row, col] = sqToRowCol(sq);
  return String.fromCharCode('a'.charCodeAt(0) + col) + String(row + 1);
}

export function algebraicToSq(s: string): number {
  const col = s.charCodeAt(0) - 'a'.charCodeAt(0);
  const row = parseInt(s[1]) - 1;
  return rowcolToSq(row, col);
}

// Precomputed tables
export const KNIGHT_ATTACKS: bigint[] = new Array(NUM_SQUARES).fill(0n);

function initKnightAttacks(): void {
  for (let sq = 0; sq < NUM_SQUARES; sq++) {
    const [row, col] = sqToRowCol(sq);
    let attacks = 0n;
    for (const [dr, dc] of KNIGHT_DELTAS) {
      const r = row + dr;
      const c = col + dc;
      if (isValidSq(r, c)) {
        attacks |= bit(rowcolToSq(r, c));
      }
    }
    KNIGHT_ATTACKS[sq] = attacks;
  }
}

// Initialize at module load
initKnightAttacks();
