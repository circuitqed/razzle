/**
 * Shared hook for board interaction logic used by both useGame (local/AI)
 * and useOnlineGame (online multiplayer).
 *
 * Sub-moves (individual knight moves, pass legs) are applied locally using
 * the TypeScript game engine and buffered in pendingMoves[].
 * Only when the turn is complete (END_TURN) is the full batch sent to
 * the server/WebSocket via the commitTurn callback.
 *
 * This means:
 *  - Cancel pass is trivial: discard the buffer and revert to committed state
 *  - No animation glitches: board state and lastMove are always consistent
 *  - The server only sees complete turns
 */

import { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import type { GameState, Player } from '../types';
import { encodeMove, decodeMove, TOTAL_SQUARES } from '../types';
import {
  playMoveSound,
  playPassSound,
  playEndTurnSound,
  playSelectSound,
} from '../utils/sounds';
import { replayToPosition, getLastMoveAtPosition, getInitialState, applyMove as replayApplyMove } from '../utils/replay';
import type { EngineState } from '../engine/state';
import { copyState, applyMove as engineApplyMove } from '../engine/state';
import { getLegalMoves as engineGetLegalMoves } from '../engine/moves';

const END_TURN_MOVE = -1;

interface LastMove {
  from: number;
  to: number;
}

export interface UseBoardInteractionOptions {
  /** Committed server state (updated after each server/WS round-trip) */
  gameState: GameState | null;
  /** Committed move history */
  rawMoves: number[];
  /** Last move from committed state */
  lastMove: LastMove | null;
  /** Whether the user can interact (useGame: !aiThinking && humanTurn; online: isMyTurn) */
  isInteractionEnabled: boolean;
  isLoading: boolean;
  /** Called when the turn is complete. Receives ALL moves for the turn
   *  (sub-moves + END_TURN). Parent sends them to the server. */
  commitTurn: (moves: number[]) => void;
}

export interface UseBoardInteractionReturn {
  selectedSquare: number | null;
  clearSelection: () => void;
  handleSquareClick: (square: number) => void;
  handleDragMove: (from: number, to: number) => void;
  endTurn: () => void;
  cancelPass: () => void;
  canEndTurn: boolean;
  isPassing: boolean;
  mustPass: boolean;
  /** Whether the player has uncommitted local moves (knight or passes) */
  hasPendingMoves: boolean;
  // History navigation
  viewPly: number | null;
  isViewingHistory: boolean;
  effectiveGameState: GameState | null;
  displayLastMove: LastMove | null;
  goToMove: (ply: number) => void;
  goForward: () => void;
  goBack: () => void;
  goToStart: () => void;
  goToEnd: () => void;
}

/** Convert API GameState → EngineState */
function apiToEngine(gs: GameState): EngineState {
  return {
    pieces: [BigInt(gs.board.p1_pieces), BigInt(gs.board.p2_pieces)],
    balls: [BigInt(gs.board.p1_ball), BigInt(gs.board.p2_ball)],
    currentPlayer: gs.current_player,
    touchedMask: BigInt(gs.touched_mask),
    hasPassed: gs.has_passed,
    lastKnightDst: gs.last_knight_dst ?? -1,
    ply: gs.ply,
  };
}

/** Build a GameState suitable for rendering from local EngineState */
function engineToDisplayState(
  base: GameState,
  eng: EngineState,
  legalMoves: number[]
): GameState {
  return {
    ...base,
    board: {
      p1_pieces: eng.pieces[0].toString(),
      p1_ball: eng.balls[0].toString(),
      p2_pieces: eng.pieces[1].toString(),
      p2_ball: eng.balls[1].toString(),
    },
    current_player: eng.currentPlayer as Player,
    legal_moves: legalMoves,
    touched_mask: eng.touchedMask.toString(),
    has_passed: eng.hasPassed,
    ply: eng.ply,
  };
}

export function useBoardInteraction(
  options: UseBoardInteractionOptions
): UseBoardInteractionReturn {
  const {
    gameState,
    rawMoves,
    lastMove,
    isInteractionEnabled,
    isLoading,
    commitTurn,
  } = options;

  const [selectedSquare, setSelectedSquare] = useState<number | null>(null);
  const [viewPly, setViewPly] = useState<number | null>(null);

  // --- Local turn buffer ---
  // Sub-moves are applied to localEngine and stored in pendingMoves.
  // Only sent to server when the turn ends (END_TURN).
  const [pendingMoves, setPendingMoves] = useState<number[]>([]);
  const [localEngine, setLocalEngine] = useState<EngineState | null>(null);

  // Clear local state when committed gameState advances (turn was committed)
  const prevLoadingRef = useRef(isLoading);
  useEffect(() => {
    if (prevLoadingRef.current && !isLoading && pendingMoves.length > 0) {
      setPendingMoves([]);
      setLocalEngine(null);
    }
    prevLoadingRef.current = isLoading;
  }, [isLoading, pendingMoves.length]);

  // --- Effective state (local override or committed) ---

  const localLegalMoves = useMemo(() => {
    if (!localEngine) return null;
    return engineGetLegalMoves(localEngine);
  }, [localEngine]);

  /** The game state used for rendering and interaction (local if mid-turn, else committed) */
  const liveGameState = useMemo(() => {
    if (!gameState) return null;
    if (localEngine && localLegalMoves && pendingMoves.length > 0) {
      return engineToDisplayState(gameState, localEngine, localLegalMoves);
    }
    return gameState;
  }, [gameState, localEngine, localLegalMoves, pendingMoves.length]);

  // Last move for highlighting during local pass chain
  const liveLastMove = useMemo((): LastMove | null => {
    if (pendingMoves.length > 0) {
      // Show the last sub-move in the pending buffer
      const last = pendingMoves[pendingMoves.length - 1];
      if (last !== END_TURN_MOVE) {
        const { src, dst } = decodeMove(last);
        return { from: src, to: dst };
      }
    }
    return lastMove;
  }, [pendingMoves, lastMove]);

  // --- Derived state (from effective state) ---

  const canEndTurn = useMemo(() => {
    if (!liveGameState) return false;
    return liveGameState.legal_moves.includes(END_TURN_MOVE);
  }, [liveGameState]);

  const isPassing = useMemo(() => {
    return liveGameState?.has_passed ?? false;
  }, [liveGameState]);

  const mustPass = useMemo(() => {
    if (!liveGameState || liveGameState.has_passed) return false;

    const ballBitboard =
      liveGameState.current_player === 0
        ? liveGameState.board.p1_ball
        : liveGameState.board.p2_ball;

    let ballSquare = -1;
    for (let i = 0; i < TOTAL_SQUARES; i++) {
      if ((BigInt(ballBitboard) & (BigInt(1) << BigInt(i))) !== BigInt(0)) {
        ballSquare = i;
        break;
      }
    }
    if (ballSquare === -1) return false;

    const realMoves = liveGameState.legal_moves.filter((m) => m !== END_TURN_MOVE);
    if (realMoves.length === 0) return false;

    return realMoves.every((move) => {
      const { src } = decodeMove(move);
      return src === ballSquare;
    });
  }, [liveGameState]);

  // --- History navigation ---

  const isViewingHistory = viewPly !== null;
  const totalPlies = rawMoves.length;

  const clearSelection = useCallback(() => {
    setSelectedSquare(null);
  }, []);

  // --- Helpers ---

  /** Get the current engine state (local if mid-turn, else from committed gameState) */
  const getCurrentEngine = useCallback((): EngineState | null => {
    if (localEngine) return localEngine;
    if (gameState) return apiToEngine(gameState);
    return null;
  }, [localEngine, gameState]);

  /** Apply a sub-move locally (buffer it, don't send to server) */
  const applyLocalMove = useCallback(
    (move: number) => {
      const current = getCurrentEngine();
      if (!current) return;
      const next = copyState(current);
      engineApplyMove(next, move);
      setLocalEngine(next);
      setPendingMoves((prev) => [...prev, move]);
    },
    [getCurrentEngine]
  );

  /** Check if a square has the current player's ball */
  const isBallAtSquare = useCallback(
    (currentPlayer: Player, sq: number): boolean => {
      const state = liveGameState;
      if (!state) return false;
      const ballBb = BigInt(
        currentPlayer === 0 ? state.board.p1_ball : state.board.p2_ball
      );
      return (ballBb & (BigInt(1) << BigInt(sq))) !== BigInt(0);
    },
    [liveGameState]
  );

  // --- Click handler ---

  const handleSquareClick = useCallback(
    (square: number) => {
      if (
        !liveGameState ||
        liveGameState.status !== 'playing' ||
        !isInteractionEnabled ||
        isLoading
      )
        return;

      // Return to live view if viewing history
      if (isViewingHistory) {
        setViewPly(null);
        return;
      }

      const { board, legal_moves, current_player } = liveGameState;

      const isOwnPiece = (player: Player, sq: number) => {
        const pieces = BigInt(
          player === 0 ? board.p1_pieces : board.p2_pieces
        );
        const ball = BigInt(player === 0 ? board.p1_ball : board.p2_ball);
        const mask = BigInt(1) << BigInt(sq);
        return ((pieces | ball) & mask) !== BigInt(0);
      };

      // During a pass chain, lock interaction to valid receivers only
      if (isPassing && selectedSquare !== null) {
        const moveEncoded = encodeMove(selectedSquare, square);
        if (legal_moves.includes(moveEncoded)) {
          playPassSound();
          applyLocalMove(moveEncoded);
          setSelectedSquare(square);
        }
        return;
      }

      // If a piece is selected, check if this is a valid move destination
      if (selectedSquare !== null) {
        const moveEncoded = encodeMove(selectedSquare, square);
        if (legal_moves.includes(moveEncoded)) {
          const isPass = isBallAtSquare(current_player, selectedSquare);

          if (isPass) {
            playPassSound();
            // Buffer pass locally for chaining
            applyLocalMove(moveEncoded);
            setSelectedSquare(square);
          } else {
            playMoveSound();
            // Knight move: commit immediately (turn ends automatically)
            setSelectedSquare(null);
            commitTurn([...pendingMoves, moveEncoded]);
          }
          return;
        }
      }

      // Select/deselect own piece
      if (isOwnPiece(current_player, square)) {
        const newSelection = selectedSquare === square ? null : square;
        setSelectedSquare(newSelection);
        if (newSelection !== null) {
          playSelectSound();
        }
        return;
      }

      // Clicking elsewhere — deselect
      if (selectedSquare !== null) {
        setSelectedSquare(null);
      }
    },
    [
      liveGameState,
      selectedSquare,
      isInteractionEnabled,
      isLoading,
      isPassing,
      isViewingHistory,
      applyLocalMove,
      isBallAtSquare,
      commitTurn,
      pendingMoves,
    ]
  );

  // --- Drag handler ---

  const handleDragMove = useCallback(
    (from: number, to: number) => {
      if (
        !liveGameState ||
        liveGameState.status !== 'playing' ||
        !isInteractionEnabled ||
        isLoading
      )
        return;

      if (isViewingHistory) {
        setViewPly(null);
        return;
      }

      const moveEncoded = encodeMove(from, to);
      if (!liveGameState.legal_moves.includes(moveEncoded)) return;

      const isPass = isBallAtSquare(liveGameState.current_player, from);

      if (isPass) {
        playPassSound();
        // Buffer pass locally for chaining
        applyLocalMove(moveEncoded);
        setSelectedSquare(to);
      } else {
        playMoveSound();
        // Knight move: commit immediately (turn ends automatically)
        setSelectedSquare(null);
        commitTurn([...pendingMoves, moveEncoded]);
      }
    },
    [
      liveGameState,
      isInteractionEnabled,
      isLoading,
      isViewingHistory,
      applyLocalMove,
      isBallAtSquare,
      commitTurn,
      pendingMoves,
    ]
  );

  // --- End turn ---

  const endTurn = useCallback(() => {
    if (!canEndTurn || !isInteractionEnabled) return;
    setViewPly(null);
    playEndTurnSound();
    setSelectedSquare(null);
    // Commit the full turn: all pending sub-moves + END_TURN
    commitTurn([...pendingMoves, END_TURN_MOVE]);
  }, [canEndTurn, isInteractionEnabled, pendingMoves, commitTurn]);

  // --- Cancel pass ---

  const cancelPass = useCallback(() => {
    setPendingMoves([]);
    setLocalEngine(null);
    setSelectedSquare(null);
  }, []);

  // --- History navigation (steps in full turns, not individual sub-moves) ---

  // Compute turn boundaries: ply indices where a full turn ends.
  // A turn ends after a knight move or after END_TURN (-1).
  // Boundaries are the ply AFTER the turn-ending move (i.e. the start of the next turn).
  const turnBoundaries = useMemo(() => {
    const boundaries: number[] = [0]; // Always include ply 0 (initial position)
    let state = getInitialState();
    for (let i = 0; i < rawMoves.length; i++) {
      const move = rawMoves[i];
      if (move === END_TURN_MOVE) {
        state = replayApplyMove(state, move);
        boundaries.push(i + 1);
      } else {
        // Check if this is a knight move (not a ball pass) by checking ball at src
        const { src } = decodeMove(move);
        const srcMask = BigInt(1) << BigInt(src);
        const isBall = state.currentPlayer === 0
          ? (BigInt(state.board.p1_ball) & srcMask) !== BigInt(0)
          : (BigInt(state.board.p2_ball) & srcMask) !== BigInt(0);
        state = replayApplyMove(state, move);
        if (!isBall) {
          // Knight move ends the turn
          boundaries.push(i + 1);
        }
      }
    }
    // Always include the end position if not already there
    if (boundaries[boundaries.length - 1] !== rawMoves.length) {
      boundaries.push(rawMoves.length);
    }
    return boundaries;
  }, [rawMoves]);

  const goToMove = useCallback(
    (ply: number) => {
      const clamped = Math.max(0, Math.min(totalPlies, ply));
      if (clamped === totalPlies) {
        setViewPly(null);
      } else {
        setViewPly(clamped);
      }
    },
    [totalPlies]
  );

  const goForward = useCallback(() => {
    const current = viewPly ?? totalPlies;
    if (current >= totalPlies) return;
    // Find the next turn boundary after current
    for (const b of turnBoundaries) {
      if (b > current) {
        goToMove(b);
        return;
      }
    }
    goToMove(totalPlies);
  }, [viewPly, totalPlies, turnBoundaries, goToMove]);

  const goBack = useCallback(() => {
    const current = viewPly ?? totalPlies;
    if (current <= 0) return;
    // Find the previous turn boundary before current
    for (let i = turnBoundaries.length - 1; i >= 0; i--) {
      if (turnBoundaries[i] < current) {
        goToMove(turnBoundaries[i]);
        return;
      }
    }
    goToMove(0);
  }, [viewPly, totalPlies, turnBoundaries, goToMove]);

  const goToStart = useCallback(() => {
    goToMove(0);
  }, [goToMove]);

  const goToEnd = useCallback(() => {
    setViewPly(null);
  }, []);

  // --- Display state for history ---

  const displayState = useMemo(() => {
    if (viewPly === null || !gameState) return null;
    return replayToPosition(rawMoves, viewPly);
  }, [viewPly, rawMoves, gameState]);

  const displayLastMove = useMemo(() => {
    if (viewPly === null) return liveLastMove;
    return getLastMoveAtPosition(rawMoves, viewPly);
  }, [viewPly, rawMoves, liveLastMove]);

  // Build effective game state for rendering.
  // Priority: history view > local mid-turn > committed server state.
  const effectiveGameState = useMemo(() => {
    if (!gameState) return null;
    if (displayState) {
      return {
        ...gameState,
        board: displayState.board,
        current_player: displayState.currentPlayer as Player,
        touched_mask: displayState.touchedMask,
        has_passed: displayState.hasPassed,
        ply: displayState.ply,
        legal_moves: [] as number[],
      };
    }
    return liveGameState;
  }, [gameState, displayState, liveGameState]);

  return {
    selectedSquare: isViewingHistory ? null : selectedSquare,
    clearSelection,
    handleSquareClick,
    handleDragMove,
    endTurn,
    cancelPass,
    canEndTurn,
    isPassing,
    mustPass,
    hasPendingMoves: pendingMoves.length > 0,
    viewPly,
    isViewingHistory,
    effectiveGameState,
    displayLastMove,
    goToMove,
    goForward,
    goBack,
    goToStart,
    goToEnd,
  };
}
