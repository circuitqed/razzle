import { vi } from 'vitest'
import type { GameState, AIMoveResponse } from '../types'

// Mock game state for testing
export const mockInitialGameState: GameState = {
  game_id: 'test-game-123',
  board: {
    p1_pieces: 62,        // Initial P1 pieces
    p1_ball: 8,           // Ball on d1
    p2_pieces: 4432676798464,  // Initial P2 pieces
    p2_ball: 562949953421312,  // Ball on d8
  },
  current_player: 0,
  legal_moves: [184, 185, 240, 241, 296, 297],  // Sample legal moves
  status: 'playing',
  winner: null,
  ply: 0,
}

export const mockGameStateAfterMove: GameState = {
  ...mockInitialGameState,
  current_player: 1,
  ply: 1,
}

export const mockAIMoveResponse: AIMoveResponse = {
  move: 3080,
  algebraic: 'd8-c6',
  policy: [],
  value: 0.1,
  visits: 800,
  time_ms: 500,
  top_moves: [
    { move: 3080, algebraic: 'd8-c6', visits: 400, value: 0.15 },
  ],
}

// Create mock API module
export function createMockApi() {
  return {
    createGame: vi.fn().mockResolvedValue({ game_id: 'test-game-123' }),
    getGameState: vi.fn().mockResolvedValue(mockInitialGameState),
    makeMove: vi.fn().mockResolvedValue(mockGameStateAfterMove),
    getAIMove: vi.fn().mockResolvedValue(mockAIMoveResponse),
    undoMove: vi.fn().mockResolvedValue(mockInitialGameState),
    getLegalMoves: vi.fn().mockResolvedValue({ moves: [] }),
    healthCheck: vi.fn().mockResolvedValue({ status: 'ok', version: '0.1.0' }),
  }
}
