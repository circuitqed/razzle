// Board state using bitboards
export interface BoardState {
  p1_pieces: number;
  p1_ball: number;
  p2_pieces: number;
  p2_ball: number;
}

// Game status
export type GameStatus = 'playing' | 'finished' | 'won' | 'draw';

// Player number
export type Player = 0 | 1;

// Full game state from API
export interface GameState {
  game_id: string;
  board: BoardState;
  current_player: Player;
  legal_moves: number[];
  status: GameStatus;
  winner: Player | null;
  ply: number;
  // Bitboard of pieces that are ineligible to receive passes
  // A piece becomes ineligible when it passes or receives the ball
  touched_mask: number;
  // Whether the current player has passed this turn (can only pass more or end turn)
  has_passed: boolean;
}

// AI move response
export interface AIMoveResponse {
  move: number;
  algebraic: string;
  policy: number[];
  value: number;
  visits: number;
  time_ms: number;
  top_moves: TopMove[];
  game_state: GameState;
}

export interface TopMove {
  move: number;
  algebraic: string;
  visits: number;
  value: number;
}

// Legal move with details
export interface LegalMove {
  move: number;
  algebraic: string;
  type: 'knight' | 'pass';
}

// WebSocket message types
export type WSMessageType = 'state' | 'thinking' | 'game_over' | 'error' | 'move' | 'ai_move' | 'undo';

export interface WSMessage<T = unknown> {
  type: WSMessageType;
  data: T;
}

export interface ThinkingData {
  simulations_done: number;
  simulations_total: number;
  current_best: string;
  value: number;
}

export interface GameOverData {
  winner: Player;
  reason: string;
}

export interface ErrorData {
  message: string;
  code: string;
}

// Board constants
export const BOARD_COLS = 7;
export const BOARD_ROWS = 8;
export const TOTAL_SQUARES = BOARD_COLS * BOARD_ROWS;

// Utility functions for move encoding/decoding
export function encodeMove(src: number, dst: number): number {
  return src * TOTAL_SQUARES + dst;
}

export function decodeMove(move: number): { src: number; dst: number } {
  const src = Math.floor(move / TOTAL_SQUARES);
  const dst = move % TOTAL_SQUARES;
  return { src, dst };
}

// Convert square index to row/col
export function squareToCoords(square: number): { row: number; col: number } {
  const row = Math.floor(square / BOARD_COLS);
  const col = square % BOARD_COLS;
  return { row, col };
}

// Convert row/col to square index
export function coordsToSquare(row: number, col: number): number {
  return row * BOARD_COLS + col;
}

// Check if a bit is set in a bitboard
export function hasPiece(bitboard: number, square: number): boolean {
  // JavaScript bitwise ops work on 32-bit ints, need BigInt for 64-bit
  const bb = BigInt(bitboard);
  const mask = BigInt(1) << BigInt(square);
  return (bb & mask) !== BigInt(0);
}

// Get all piece positions from a bitboard
export function getPiecePositions(bitboard: number): number[] {
  const positions: number[] = [];
  const bb = BigInt(bitboard);
  for (let i = 0; i < TOTAL_SQUARES; i++) {
    if ((bb & (BigInt(1) << BigInt(i))) !== BigInt(0)) {
      positions.push(i);
    }
  }
  return positions;
}

// Square notation (a1-g8)
export function squareToAlgebraic(square: number): string {
  const { row, col } = squareToCoords(square);
  const file = String.fromCharCode('a'.charCodeAt(0) + col);
  const rank = row + 1;
  return `${file}${rank}`;
}

export function algebraicToSquare(algebraic: string): number {
  const file = algebraic.charCodeAt(0) - 'a'.charCodeAt(0);
  const rank = parseInt(algebraic[1]) - 1;
  return coordsToSquare(rank, file);
}
