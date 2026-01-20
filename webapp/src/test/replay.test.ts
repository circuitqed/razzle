import { describe, it, expect } from 'vitest'
import {
  getInitialState,
  applyMove,
  reconstructPositions,
  replayToPosition,
  getLastMoveAtPosition,
} from '../utils/replay'

describe('Replay Utilities', () => {
  describe('getInitialState', () => {
    it('returns correct initial state', () => {
      const state = getInitialState()

      expect(state.currentPlayer).toBe(0)
      expect(state.ply).toBe(0)
      expect(state.hasPassed).toBe(false)
      expect(state.touchedMask).toBe('0')
      // Board values are strings to preserve precision for JS
      expect(state.board.p1_pieces).not.toBe('0')
      expect(state.board.p2_pieces).not.toBe('0')
      expect(state.board.p1_ball).not.toBe('0')
      expect(state.board.p2_ball).not.toBe('0')
    })
  })

  describe('applyMove', () => {
    it('handles end turn move', () => {
      const state = getInitialState()
      const newState = applyMove(state, -1)

      expect(newState.currentPlayer).toBe(1)
      expect(newState.ply).toBe(1)
      expect(newState.hasPassed).toBe(false)
      expect(newState.touchedMask).toBe('0')
    })

    it('increments ply on each move', () => {
      let state = getInitialState()
      expect(state.ply).toBe(0)

      // Apply end turn moves to increment ply
      state = applyMove(state, -1)
      expect(state.ply).toBe(1)

      state = applyMove(state, -1)
      expect(state.ply).toBe(2)
    })

    it('switches player on end turn', () => {
      let state = getInitialState()
      expect(state.currentPlayer).toBe(0)

      state = applyMove(state, -1)
      expect(state.currentPlayer).toBe(1)

      state = applyMove(state, -1)
      expect(state.currentPlayer).toBe(0)
    })
  })

  describe('reconstructPositions', () => {
    it('returns initial position for empty moves', () => {
      const positions = reconstructPositions([])

      expect(positions.length).toBe(1)
      expect(positions[0].ply).toBe(0)
      expect(positions[0].currentPlayer).toBe(0)
    })

    it('returns correct number of positions', () => {
      const moves = [-1, -1, -1]  // Three end turn moves
      const positions = reconstructPositions(moves)

      expect(positions.length).toBe(4)  // Initial + 3 moves
      expect(positions[0].ply).toBe(0)
      expect(positions[1].ply).toBe(1)
      expect(positions[2].ply).toBe(2)
      expect(positions[3].ply).toBe(3)
    })

    it('tracks player changes correctly', () => {
      const moves = [-1, -1]  // Two end turn moves
      const positions = reconstructPositions(moves)

      expect(positions[0].currentPlayer).toBe(0)
      expect(positions[1].currentPlayer).toBe(1)
      expect(positions[2].currentPlayer).toBe(0)
    })
  })

  describe('replayToPosition', () => {
    it('returns initial state for ply 0', () => {
      const moves = [-1, -1, -1]
      const state = replayToPosition(moves, 0)

      expect(state.ply).toBe(0)
      expect(state.currentPlayer).toBe(0)
    })

    it('returns correct state for middle position', () => {
      const moves = [-1, -1, -1]
      const state = replayToPosition(moves, 2)

      expect(state.ply).toBe(2)
      expect(state.currentPlayer).toBe(0)  // After 2 end turns, back to player 0
    })

    it('returns final state for max ply', () => {
      const moves = [-1, -1, -1]
      const state = replayToPosition(moves, 3)

      expect(state.ply).toBe(3)
    })
  })

  describe('getLastMoveAtPosition', () => {
    it('returns null for ply 0', () => {
      const moves = [100, 200, 300]  // Some encoded moves
      const lastMove = getLastMoveAtPosition(moves, 0)

      expect(lastMove).toBeNull()
    })

    it('returns null for end turn moves', () => {
      const moves = [-1, -1]
      const lastMove = getLastMoveAtPosition(moves, 1)

      expect(lastMove).toBeNull()
    })

    it('returns from/to for regular moves', () => {
      // Encoded move: src * 56 + dst
      // Move from square 3 to square 10: 3 * 56 + 10 = 178
      const moves = [178]
      const lastMove = getLastMoveAtPosition(moves, 1)

      expect(lastMove).not.toBeNull()
      expect(lastMove?.from).toBe(3)
      expect(lastMove?.to).toBe(10)
    })
  })
})
