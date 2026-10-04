import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'

vi.mock('../utils/sounds', () => ({
  playMoveSound: vi.fn(),
  playPassSound: vi.fn(),
  playEndTurnSound: vi.fn(),
  playSelectSound: vi.fn(),
}))

import { useTutorial } from './useTutorial'
import { TUTORIAL_STEPS, toHookSteps } from '../data/tutorialSteps'
import { getLegalMoves } from '../engine/moves'

const mv = (src: number, dst: number) => src * 56 + dst
const has = (bb: bigint, sq: number) => (bb & (1n << BigInt(sq))) !== 0n

const hookSteps = toHookSteps()

function setup() {
  return renderHook(() => useTutorial({ steps: hookSteps, onComplete: vi.fn(), onSkip: vi.fn() }))
}

type R = ReturnType<typeof setup>['result']

function drag(result: R, from: number, to: number) {
  act(() => { result.current.handleDragMove(from, to) })
  act(() => { vi.advanceTimersByTime(2000) })
}

function next(result: R) {
  act(() => { result.current.nextStep() })
  act(() => { vi.advanceTimersByTime(2000) })
}

function goToStep(result: R, id: string) {
  const target = TUTORIAL_STEPS.findIndex((s) => s.id === id)
  while (result.current.currentStep < target) {
    // Use the intended (first allowed) move(s) to clear earlier steps.
    const step = TUTORIAL_STEPS[result.current.currentStep]
    const m = (step.suggestedMoves ?? step.allowedMoves)[0]
    drag(result, Math.floor(m / 56), m % 56)
    for (const c of step.chainMoves ?? []) {
      if (result.current.gameState!.legal_moves.includes(c)) drag(result, Math.floor(c / 56), c % 56)
    }
    if (result.current.canEndTurn) act(() => { result.current.endTurn() })
    expect(result.current.stepComplete).toBe(true)
    next(result)
  }
}

describe('useTutorial', () => {
  beforeEach(() => { vi.useFakeTimers() })
  afterEach(() => { vi.useRealTimers() })

  it('completes every step with the intended moves', () => {
    const { result } = setup()
    goToStep(result, TUTORIAL_STEPS[TUTORIAL_STEPS.length - 1].id)
    const last = TUTORIAL_STEPS[TUTORIAL_STEPS.length - 1]
    const m = last.allowedMoves[0]
    drag(result, Math.floor(m / 56), m % 56)
    expect(result.current.stepComplete).toBe(true)
  })

  it('eligible-receivers: the board reflects where the learner actually moved (e5->d3)', () => {
    const { result } = setup()
    goToStep(result, 'eligible-receivers')
    drag(result, 32, 17) // e5 -> d3, one of the two suggested squares
    const s = result.current.engineState
    // The knight must be where the learner put it, not teleported to c4.
    expect(has(s.pieces[0], 17)).toBe(true)
    expect(has(s.pieces[0], 23)).toBe(false)
    // ...and the follow-up pass to it must be available.
    expect(result.current.gameState!.legal_moves).toContain(mv(24, 17))
    drag(result, 24, 17)
    expect(result.current.stepComplete).toBe(true)
    expect(has(result.current.engineState.balls[0], 17)).toBe(true)
  })

  it('eligible-receivers: an off-line knight move is rejected with a hint, board unchanged', () => {
    const { result } = setup()
    goToStep(result, 'eligible-receivers')
    const before = result.current.engineState
    // e5 -> f3 is a legal knight move, but f3 is not on a passing line from d4.
    expect(getLegalMoves(before)).toContain(mv(32, 19))
    drag(result, 32, 19)
    const s = result.current.engineState
    expect(s).toEqual(before)
    expect(has(s.pieces[0], 23)).toBe(false) // no teleport to c4
    expect(result.current.stepComplete).toBe(false)
    expect(result.current.wrongMoveMessage).toBeTruthy()
    // The intended move still works afterwards.
    drag(result, 32, 23)
    expect(result.current.wrongMoveMessage).toBeNull()
    drag(result, 24, 23)
    expect(result.current.stepComplete).toBe(true)
  })

  it('every step: any legal-but-unintended move is rejected and leaves the board unchanged', () => {
    const { result } = setup()
    for (let i = 0; i < TUTORIAL_STEPS.length; i++) {
      goToStep(result, TUTORIAL_STEPS[i].id)
      const before = result.current.engineState
      const allowed = new Set(result.current.gameState!.legal_moves)
      const unintended = getLegalMoves(before).filter((m) => m >= 0 && !allowed.has(m))
      for (const m of unintended) {
        drag(result, Math.floor(m / 56), m % 56)
        expect(result.current.engineState).toEqual(before)
        expect(result.current.stepComplete).toBe(false)
        expect(result.current.wrongMoveMessage).toBeTruthy()
      }
    }
  })

  it('chain steps: unintended moves mid-chain (incl. ending early) are rejected', () => {
    for (const step of TUTORIAL_STEPS.filter((s) => s.chainMoves)) {
      const { result, unmount } = setup()
      goToStep(result, step.id)
      const m = step.allowedMoves[0]
      drag(result, Math.floor(m / 56), m % 56)
      const before = result.current.engineState
      expect(result.current.canEndTurn).toBe(false)
      const allowed = new Set(result.current.gameState!.legal_moves)
      expect(allowed.size).toBeGreaterThan(0)
      for (const u of getLegalMoves(before).filter((x) => x >= 0 && !allowed.has(x))) {
        drag(result, Math.floor(u / 56), u % 56)
        expect(result.current.engineState).toEqual(before)
        expect(result.current.wrongMoveMessage).toBeTruthy()
      }
      act(() => { result.current.endTurn() }) // not allowed yet: no-op
      expect(result.current.engineState).toEqual(before)
      expect(result.current.stepComplete).toBe(false)
      unmount()
    }
  })

  it('clicks are ignored while the opponent move is animating', () => {
    const { result } = setup()
    goToStep(result, 'score')
    // finish score step, then advance to forced-pass without flushing timers
    drag(result, 45, 52)
    next(result)
    drag(result, 32, 23)
    drag(result, 24, 23)
    act(() => { result.current.nextStep() })
    expect(result.current.showingPreState).toBe(true)
    act(() => { result.current.handleDragMove(24, 22) })
    expect(result.current.stepComplete).toBe(false)
    act(() => { vi.advanceTimersByTime(2000) })
    expect(result.current.showingPreState).toBe(false)
    drag(result, 24, 22)
    expect(result.current.stepComplete).toBe(true)
  })
})
