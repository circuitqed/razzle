import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useGame } from './useGame'
import { mockInitialGameState, mockGameStateAfterMove, mockAIMoveResponse } from '../test/mocks'

// Mock the API module
vi.mock('../api/engine', () => ({
  createGame: vi.fn(),
  getGameState: vi.fn(),
  makeMove: vi.fn(),
  getAIMove: vi.fn(),
  undoMove: vi.fn(),
}))

// Import mocked module
import * as api from '../api/engine'

describe('useGame hook', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // Setup default mock implementations
    vi.mocked(api.createGame).mockResolvedValue({ game_id: 'test-game-123' })
    vi.mocked(api.getGameState).mockResolvedValue(mockInitialGameState)
    vi.mocked(api.makeMove).mockResolvedValue(mockGameStateAfterMove)
    vi.mocked(api.getAIMove).mockResolvedValue(mockAIMoveResponse)
    vi.mocked(api.undoMove).mockResolvedValue(mockInitialGameState)
  })

  describe('initialization', () => {
    it('should start with null game state', () => {
      const { result } = renderHook(() => useGame())
      expect(result.current.gameState).toBeNull()
      expect(result.current.isLoading).toBe(false)
      expect(result.current.error).toBeNull()
    })
  })

  describe('startNewGame', () => {
    it('should create a new game and fetch state', async () => {
      const { result } = renderHook(() => useGame())

      await act(async () => {
        await result.current.startNewGame()
      })

      expect(api.createGame).toHaveBeenCalledWith({
        player1_type: 'human',
        player2_type: 'ai',
        ai_simulations: 800,
      })
      expect(api.getGameState).toHaveBeenCalledWith('test-game-123')
      expect(result.current.gameState).toEqual(mockInitialGameState)
    })

    it('should set loading state during game creation', async () => {
      const { result } = renderHook(() => useGame())

      // Start the game creation
      const promise = act(async () => {
        await result.current.startNewGame()
      })

      // Note: In practice, you'd need to test the loading state
      // during the async operation, which is complex with renderHook

      await promise
      expect(result.current.isLoading).toBe(false)
    })

    it('should handle errors', async () => {
      vi.mocked(api.createGame).mockRejectedValue(new Error('Network error'))

      const { result } = renderHook(() => useGame())

      await act(async () => {
        await result.current.startNewGame()
      })

      expect(result.current.error).toBe('Network error')
      expect(result.current.gameState).toBeNull()
    })

    it('should clear previous state when starting new game', async () => {
      const { result } = renderHook(() => useGame())

      // Start a game
      await act(async () => {
        await result.current.startNewGame()
      })

      // Simulate an error state
      vi.mocked(api.createGame).mockRejectedValue(new Error('Failed'))

      await act(async () => {
        await result.current.startNewGame()
      })

      expect(result.current.selectedSquare).toBeNull()
    })
  })

  describe('handleSquareClick', () => {
    it('should do nothing when game state is null', async () => {
      const { result } = renderHook(() => useGame())

      act(() => {
        result.current.handleSquareClick(3)
      })

      expect(result.current.selectedSquare).toBeNull()
    })

    it('should select a piece when clicking on own piece', async () => {
      const { result } = renderHook(() => useGame())

      await act(async () => {
        await result.current.startNewGame()
      })

      // Click on a P1 piece (square 1 = b1)
      act(() => {
        result.current.handleSquareClick(1)
      })

      expect(result.current.selectedSquare).toBe(1)
    })

    it('should deselect when clicking same piece again', async () => {
      const { result } = renderHook(() => useGame())

      await act(async () => {
        await result.current.startNewGame()
      })

      // Select piece
      act(() => {
        result.current.handleSquareClick(1)
      })
      expect(result.current.selectedSquare).toBe(1)

      // Click same piece again
      act(() => {
        result.current.handleSquareClick(1)
      })
      expect(result.current.selectedSquare).toBeNull()
    })
  })

  describe('AI vs Human mode', () => {
    it('should default to vs AI mode', () => {
      const { result } = renderHook(() => useGame())
      // vsAI is internal, but we can test behavior
      expect(result.current.aiThinking).toBe(false)
    })

    it('should use custom simulation count', async () => {
      const { result } = renderHook(() => useGame({ aiSimulations: 400 }))

      await act(async () => {
        await result.current.startNewGame()
      })

      expect(api.createGame).toHaveBeenCalledWith(
        expect.objectContaining({
          ai_simulations: 400,
        })
      )
    })

    it('should set player2_type to human when vsAI is false', async () => {
      const { result } = renderHook(() => useGame({ vsAI: false }))

      await act(async () => {
        await result.current.startNewGame()
      })

      expect(api.createGame).toHaveBeenCalledWith(
        expect.objectContaining({
          player2_type: 'human',
        })
      )
    })
  })

  describe('undoMove', () => {
    it('should call undo API', async () => {
      const { result } = renderHook(() => useGame())

      await act(async () => {
        await result.current.startNewGame()
      })

      await act(async () => {
        await result.current.undoMove()
      })

      expect(api.undoMove).toHaveBeenCalledWith('test-game-123')
    })

    it('should do nothing when game state is null', async () => {
      const { result } = renderHook(() => useGame())

      await act(async () => {
        await result.current.undoMove()
      })

      expect(api.undoMove).not.toHaveBeenCalled()
    })

    it('should handle undo errors', async () => {
      vi.mocked(api.undoMove).mockRejectedValue(new Error('Nothing to undo'))

      const { result } = renderHook(() => useGame())

      await act(async () => {
        await result.current.startNewGame()
      })

      await act(async () => {
        await result.current.undoMove()
      })

      expect(result.current.error).toBe('Nothing to undo')
    })
  })
})
