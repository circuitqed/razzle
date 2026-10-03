/**
 * useGame with the on-device backend: when offline, games are created and
 * played locally (human turns, AI turns, undo) with no server round-trips.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act, waitFor } from '@testing-library/react'
import { newGame } from '../engine/state'
import { getLegalMoves } from '../engine/moves'
import type { EngineState } from '../engine/state'

vi.mock('../api/engine', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./../api/engine')>()
  return {
    ...actual,
    createGame: vi.fn(),
    getGameState: vi.fn(),
    makeMove: vi.fn(),
    makeTurn: vi.fn(),
    undoMove: vi.fn(),
    resignGame: vi.fn(),
  }
})

vi.mock('../engine/bundledModels', () => ({
  resolveModelInfo: vi.fn().mockResolvedValue({ version: 'test', url: '', size_bytes: 0 }),
}))

// Fake AI: always plays the first legal move, instantly.
const fakeWorker = {
  isLoaded: true,
  isLoading: false,
  isSearching: false,
  isRandom: false,
  backend: 'test',
  loadError: null,
  progress: null,
  loadModel: vi.fn(),
  loadRandomEvaluator: vi.fn(),
  search: vi.fn(async (state: EngineState) => ({ bestMove: getLegalMoves(state)[0], simsDone: 1, value: 0 })),
  abort: vi.fn(),
  cancelSearch: vi.fn(),
  waitForLoad: vi.fn().mockResolvedValue(true),
}
vi.mock('./useAIWorker', () => ({
  useAIWorker: () => fakeWorker,
  SEARCH_CANCELLED: 'Search cancelled',
}))

import { useGame } from './useGame'
import * as serverApi from '../api/engine'

const onLine = vi.spyOn(navigator, 'onLine', 'get')

beforeEach(() => {
  localStorage.clear()
  vi.clearAllMocks()
  onLine.mockReturnValue(false)
})
afterEach(() => onLine.mockReset())

describe('useGame offline (on-device backend)', () => {
  it('plays human and AI turns locally and undoes back to the human turn', async () => {
    const { result } = renderHook(() => useGame({ vsAI: true, playerColor: 0, aiModel: 'pegasus_iter_050.pt' }))

    await act(async () => { await result.current.startNewGame() })
    const gs = result.current.gameState!
    expect(gs.game_id.startsWith('local-')).toBe(true)
    expect(gs.ply).toBe(0)

    // Human knight move (knight moves commit immediately)
    const start = newGame()
    const knight = getLegalMoves(start).find(m => m >= 0)!
    await act(async () => { result.current.handleDragMove(Math.floor(knight / 56), knight % 56) })

    // AI replies on-device; play returns to the human
    await waitFor(() => {
      expect(result.current.gameState!.current_player).toBe(0)
      expect(result.current.rawMoves.length).toBeGreaterThanOrEqual(2)
    })
    expect(fakeWorker.search).toHaveBeenCalled()
    expect(result.current.rawMoves[0]).toBe(knight)

    // Undo rewinds both the AI reply and the human move
    await act(async () => { await result.current.undoMove() })
    expect(result.current.rawMoves).toEqual([])
    expect(result.current.gameState!.ply).toBe(0)
    expect(result.current.gameState!.current_player).toBe(0)

    // Nothing touched the server
    expect(serverApi.createGame).not.toHaveBeenCalled()
    expect(serverApi.makeTurn).not.toHaveBeenCalled()
    expect(serverApi.makeMove).not.toHaveBeenCalled()
  })

  it('lets the AI open when the human plays red', async () => {
    const { result } = renderHook(() => useGame({ vsAI: true, playerColor: 1, aiModel: 'pegasus_iter_050.pt' }))
    await act(async () => { await result.current.startNewGame() })
    await waitFor(() => expect(result.current.gameState!.current_player).toBe(1))
    expect(result.current.rawMoves.length).toBeGreaterThan(0)
  })

  it('resumes a saved on-device game with correct player attribution', async () => {
    const { result } = renderHook(() => useGame({ vsAI: false }))
    await act(async () => { await result.current.startNewGame() })
    const id = result.current.gameState!.game_id
    const knight = getLegalMoves(newGame()).find(m => m >= 0)!
    await act(async () => { result.current.handleDragMove(Math.floor(knight / 56), knight % 56) })
    await waitFor(() => expect(result.current.rawMoves).toEqual([knight]))

    const { result: resumed } = renderHook(() => useGame({ vsAI: false }))
    let ok = false
    await act(async () => { ok = await resumed.current.resumeGame(id) })
    expect(ok).toBe(true)
    expect(resumed.current.rawMoves).toEqual([knight])
    expect(resumed.current.moveHistory[0].player).toBe(0)
    expect(resumed.current.gameState!.current_player).toBe(1)
  })

  it('does not spin retrying when the AI fails deterministically', async () => {
    fakeWorker.search.mockRejectedValue(new Error('boom'))
    const { result } = renderHook(() => useGame({ vsAI: true, playerColor: 1, aiModel: 'pegasus_iter_050.pt' }))
    await act(async () => { await result.current.startNewGame() })
    await waitFor(() => expect(result.current.error).toBe('boom'))
    await act(async () => { await new Promise(r => setTimeout(r, 200)) })
    expect(fakeWorker.search).toHaveBeenCalledTimes(1)
    fakeWorker.search.mockReset().mockImplementation(async (state: EngineState) => ({ bestMove: getLegalMoves(state)[0], simsDone: 1, value: 0 }))
  })
})
