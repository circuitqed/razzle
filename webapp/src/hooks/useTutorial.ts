/**
 * Hook for KnightBall's interactive tutorial.
 *
 * Manages a sequence of guided tutorial steps, each with a preset board
 * position and a restricted set of allowed moves. The hook converts the
 * internal EngineState to the GameState type expected by the Board
 * component, and handles click-to-move / drag-to-move interaction with
 * the same select-then-execute pattern used in useBoardInteraction.
 */

import { useState, useCallback, useMemo, useEffect } from 'react';
import type { GameState, Player } from '../types';
import { encodeMove, TOTAL_SQUARES } from '../types';
import type { EngineState } from '../engine/state';
import { copyState, applyMove } from '../engine/state';
import { getLegalMoves } from '../engine/moves';
import {
  playMoveSound,
  playPassSound,
  playEndTurnSound,
  playSelectSound,
} from '../utils/sounds';

// ---------------------------------------------------------------------------
// Tutorial step definition
// ---------------------------------------------------------------------------

export interface TutorialStep {
  /** Short title shown above the board */
  title: string;
  /** Main instruction text */
  instruction: string;
  /** Secondary hint (shown after a delay or on hover) */
  hint: string;
  /** The board state for this step */
  state: EngineState;
  /** Encoded moves the player is allowed to make (subset of legal moves) */
  allowedMoves: number[];
  /** If true the step requires the player to press "End Turn" after moves */
  requireEndTurn?: boolean;
  /** State to load after the first move (for chain steps) */
  nextStepState?: EngineState;
  /** Continuation moves after nextStepState loads. Step completes after these. */
  chainMoves?: number[];
  /** Squares to highlight as hints (e.g. valid sources / destinations) */
  highlightSquares?: number[];
  /** State to show before the interactive state (opponent about to move). */
  preState?: EngineState;
  preMessage?: string;
  preMove?: { from: number; to: number };
  /** Message shown after completing the step. */
  completionMessage?: string;
  /** State shown briefly during chain transition (opponent moving). */
  chainPreState?: EngineState;
  chainPreMessage?: string;
  chainPreMove?: { from: number; to: number };
}

// ---------------------------------------------------------------------------
// Return type
// ---------------------------------------------------------------------------

export interface UseTutorialReturn {
  currentStep: number;
  totalSteps: number;
  stepTitle: string;
  stepInstruction: string;
  stepHint: string;
  gameState: GameState | null;
  engineState: EngineState;
  showingPreState: boolean;
  preMessage: string | null;
  preAnimMove: { from: number; to: number } | null;
  completionMessage: string | null;
  selectedSquare: number | null;
  highlightSquares: number[];
  isPassing: boolean;
  canEndTurn: boolean;
  stepComplete: boolean;
  handleSquareClick: (square: number) => void;
  handleDragMove: (from: number, to: number) => void;
  endTurn: () => void;
  nextStep: () => void;
  skipTutorial: () => void;
}

// ---------------------------------------------------------------------------
// localStorage key
// ---------------------------------------------------------------------------

const TUTORIAL_COMPLETE_KEY = 'knightball_tutorial_complete';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Dummy game_id used for the tutorial GameState wrapper. */
const TUTORIAL_GAME_ID = '__tutorial__';

const END_TURN_MOVE = -1;

/**
 * Build a GameState object (as expected by the Board component) from an
 * EngineState and a set of legal moves. BigInt values are serialised to
 * strings because GameState uses string bitboards.
 */
function engineToGameState(eng: EngineState, legalMoves: number[]): GameState {
  return {
    game_id: TUTORIAL_GAME_ID,
    board: {
      p1_pieces: eng.pieces[0].toString(),
      p1_ball: eng.balls[0].toString(),
      p2_pieces: eng.pieces[1].toString(),
      p2_ball: eng.balls[1].toString(),
    },
    current_player: eng.currentPlayer as Player,
    legal_moves: legalMoves,
    status: 'playing',
    winner: null,
    ply: eng.ply,
    touched_mask: eng.touchedMask.toString(),
    has_passed: eng.hasPassed,
    last_knight_dst: eng.lastKnightDst,
  };
}

/**
 * Check whether `square` holds one of the current player's pieces (including
 * the ball).
 */
function isOwnPiece(eng: EngineState, square: number): boolean {
  const mask = BigInt(1) << BigInt(square);
  const p = eng.currentPlayer;
  return ((eng.pieces[p] | eng.balls[p]) & mask) !== BigInt(0);
}

/** Check whether the current player's ball sits on `square`. */
function isBallAt(eng: EngineState, square: number): boolean {
  const mask = BigInt(1) << BigInt(square);
  return (eng.balls[eng.currentPlayer] & mask) !== BigInt(0);
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export interface UseTutorialOptions {
  steps: TutorialStep[];
  onComplete: () => void;
  onSkip: () => void;
}

export function useTutorial(options: UseTutorialOptions): UseTutorialReturn {
  const { steps, onComplete, onSkip } = options;

  // -- Core state -----------------------------------------------------------
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [localState, setLocalState] = useState<EngineState>(() =>
    copyState(steps[0].state)
  );
  const [selectedSquare, setSelectedSquare] = useState<number | null>(null);
  const [stepComplete, setStepComplete] = useState(false);
  const [inChain, setInChain] = useState(false);
  const [chainDone, setChainDone] = useState(false);
  const [showingPreState, setShowingPreState] = useState(false);
  const [preAnimMove, setPreAnimMove] = useState<{ from: number; to: number } | null>(null);

  // Handle preState: show the "before" state briefly, then animate the
  // opponent's move by switching to the "after" state with lastMove set.
  useEffect(() => {
    if (!step.preState) return;
    // Show the pre-state (before opponent moves)
    setShowingPreState(true);
    setPreAnimMove(null);
    setLocalState(copyState(step.preState));
    // Brief pause, then animate the opponent's move
    const timer = setTimeout(() => {
      setLocalState(copyState(step.state));
      if (step.preMove) setPreAnimMove(step.preMove);
      // Clear after animation completes (Board animation is 350ms)
      setTimeout(() => {
        setShowingPreState(false);
        setPreAnimMove(null);
      }, 400);
    }, 200);
    return () => clearTimeout(timer);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentStepIndex]);

  // -- Derived step data ----------------------------------------------------
  const step = steps[currentStepIndex];

  // Compute full legal moves from the engine, then intersect with the step's
  // allowed set so the player can only make the intended move(s).
  const filteredLegalMoves = useMemo(() => {
    if (stepComplete) return [];
    const engineLegal = getLegalMoves(localState);
    // After chain passes are done, only allow END_TURN
    if (chainDone) {
      return engineLegal.filter((m) => m === END_TURN_MOVE);
    }
    // In chain phase, use chainMoves instead of allowedMoves
    const allowed = new Set(inChain && step.chainMoves ? step.chainMoves : step.allowedMoves);
    return engineLegal.filter((m) => allowed.has(m));
  }, [localState, step.allowedMoves, step.chainMoves, stepComplete, inChain, chainDone]);

  const isPassing = localState.hasPassed;

  const canEndTurn = useMemo(() => {
    if (stepComplete) return false;
    if (!step.requireEndTurn && !chainDone) return false;
    return filteredLegalMoves.includes(END_TURN_MOVE);
  }, [filteredLegalMoves, step.requireEndTurn, stepComplete, chainDone]);

  // Build the GameState fed to the Board component.
  const gameState = useMemo<GameState | null>(() => {
    // When step is complete and waiting for the player to advance, show no
    // legal moves so the board is non-interactive.
    const moves = stepComplete ? [] : filteredLegalMoves;
    return engineToGameState(localState, moves);
  }, [localState, filteredLegalMoves, stepComplete]);

  const highlightSquares = useMemo(() => {
    return step.highlightSquares ?? [];
  }, [step.highlightSquares]);

  // -- Move execution -------------------------------------------------------

  /**
   * After a move has been applied, decide whether the step is complete.
   *
   * - If `requireEndTurn` is set, the step is NOT complete yet (player needs
   *   to press End Turn).
   * - If `nextStepState` exists, load that state (e.g. to show an opponent
   *   reply) and mark the step complete.
   * - Otherwise mark the step complete immediately.
   */
  const afterMove = useCallback(
    (newState: EngineState, wasKnightMove: boolean) => {
      // Chain step phase 1: first move loads continuation state
      if (step.nextStepState && step.chainMoves && !inChain) {
        if (step.chainPreState) {
          // Show opponent moving with animation before chain continuation
          setShowingPreState(true);
          setPreAnimMove(null);
          setLocalState(copyState(step.chainPreState));
          setTimeout(() => {
            setLocalState(copyState(step.nextStepState!));
            if (step.chainPreMove) setPreAnimMove(step.chainPreMove);
            setTimeout(() => {
              setShowingPreState(false);
              setPreAnimMove(null);
              setInChain(true);
            }, 500);
          }, 600);
        } else {
          setLocalState(copyState(step.nextStepState));
          setInChain(true);
        }
        return;
      }

      // Chain step phase 2: chain passes done, need END_TURN
      if (inChain && step.requireEndTurn) {
        setLocalState(newState);
        setChainDone(true);
        setSelectedSquare(null);
        return;
      }

      // Non-chain step with requireEndTurn (pass must be followed by END_TURN)
      if (step.requireEndTurn && !wasKnightMove && !inChain) {
        setLocalState(newState);
        return;
      }

      if (step.nextStepState && !step.chainMoves) {
        setLocalState(copyState(step.nextStepState));
      } else {
        setLocalState(newState);
      }
      setStepComplete(true);
      setSelectedSquare(null);
    },
    [step.requireEndTurn, step.nextStepState, step.chainMoves, inChain, chainDone]
  );

  /** Execute an encoded move on the local state. */
  const executeMove = useCallback(
    (move: number) => {
      const next = copyState(localState);
      const wasBall = isBallAt(localState, Math.floor(move / TOTAL_SQUARES));
      applyMove(next, move);

      if (wasBall) {
        playPassSound();
        // After a pass, auto-select the destination so the player can
        // continue a pass chain (if allowed) or end their turn.
        const dst = move % TOTAL_SQUARES;
        setSelectedSquare(dst);
        afterMove(next, false);
      } else {
        playMoveSound();
        // Knight move ends the turn automatically.
        setSelectedSquare(null);
        afterMove(next, true);
      }
    },
    [localState, afterMove]
  );

  // -- Interaction handlers -------------------------------------------------

  const handleSquareClick = useCallback(
    (square: number) => {
      if (stepComplete) return;

      // During a pass chain, lock interaction to valid pass receivers.
      if (isPassing && selectedSquare !== null) {
        const moveEncoded = encodeMove(selectedSquare, square);
        if (filteredLegalMoves.includes(moveEncoded)) {
          executeMove(moveEncoded);
        }
        return;
      }

      // Second click: try to execute a move from the selected square.
      if (selectedSquare !== null) {
        const moveEncoded = encodeMove(selectedSquare, square);
        if (filteredLegalMoves.includes(moveEncoded)) {
          executeMove(moveEncoded);
          return;
        }
      }

      // First click (or re-selection): select own piece if it has moves.
      if (isOwnPiece(localState, square)) {
        const hasMove = filteredLegalMoves.some((m) => {
          if (m === END_TURN_MOVE) return false;
          const src = Math.floor(m / TOTAL_SQUARES);
          return src === square;
        });

        if (hasMove) {
          const next = selectedSquare === square ? null : square;
          setSelectedSquare(next);
          if (next !== null) {
            playSelectSound();
          }
          return;
        }
      }

      // Click on empty / opponent square -> deselect.
      if (selectedSquare !== null) {
        setSelectedSquare(null);
      }
    },
    [
      stepComplete,
      selectedSquare,
      isPassing,
      filteredLegalMoves,
      localState,
      executeMove,
    ]
  );

  const handleDragMove = useCallback(
    (from: number, to: number) => {
      if (stepComplete) return;

      const moveEncoded = encodeMove(from, to);
      if (!filteredLegalMoves.includes(moveEncoded)) return;

      executeMove(moveEncoded);
    },
    [stepComplete, filteredLegalMoves, executeMove]
  );

  // -- End turn -------------------------------------------------------------

  const endTurn = useCallback(() => {
    if (!canEndTurn) return;

    playEndTurnSound();
    const next = copyState(localState);
    applyMove(next, END_TURN_MOVE);
    setLocalState(next);
    setStepComplete(true);
    setSelectedSquare(null);
  }, [canEndTurn, localState, step.nextStepState]);

  // -- Navigation -----------------------------------------------------------

  const nextStep = useCallback(() => {
    const nextIndex = currentStepIndex + 1;
    if (nextIndex >= steps.length) {
      // Tutorial finished -- persist and notify parent.
      try {
        localStorage.setItem(TUTORIAL_COMPLETE_KEY, 'true');
      } catch {
        // localStorage may be unavailable; ignore.
      }
      onComplete();
      return;
    }

    setCurrentStepIndex(nextIndex);
    setLocalState(copyState(steps[nextIndex].state));
    setSelectedSquare(null);
    setStepComplete(false);
    setInChain(false);
    setChainDone(false);
  }, [currentStepIndex, steps, onComplete]);

  const skipTutorial = useCallback(() => {
    try {
      localStorage.setItem(TUTORIAL_COMPLETE_KEY, 'true');
    } catch {
      // Ignore.
    }
    onSkip();
  }, [onSkip]);

  // -- Return ---------------------------------------------------------------

  return {
    currentStep: currentStepIndex,
    totalSteps: steps.length,
    stepTitle: step.title,
    stepInstruction: step.instruction,
    stepHint: step.hint,
    gameState,
    engineState: localState,
    showingPreState,
    preMessage: showingPreState ? (step.chainPreMessage ?? step.preMessage ?? null) : null,
    preAnimMove,
    completionMessage: stepComplete ? (step.completionMessage ?? null) : null,
    selectedSquare,
    highlightSquares,
    isPassing,
    canEndTurn,
    stepComplete,
    handleSquareClick,
    handleDragMove,
    endTurn,
    nextStep,
    skipTutorial,
  };
}
