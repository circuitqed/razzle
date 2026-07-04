import { useState, useEffect, useCallback, useMemo } from 'react';
import { OpeningBook } from '../engine/openingBook';
import { newGame, copyState, applyMove } from '../engine/state';
import type { EngineState } from '../engine/state';
import { sqToAlgebraic, NUM_SQUARES, END_TURN_ACTION } from '../engine/bitboard';
import { getOpeningName } from '../engine/openingNames';
import type { BoardState } from '../types';
import { API_BASE } from '../api/base';

export interface BookMove {
  actions: number[]; // sequence of actions forming this complete turn
  label: string; // e.g. "f1-e3", "d1-c1, c1-b1", "End Turn"
  type: 'knight' | 'pass' | 'end_turn';
  count: number;
  winRate: number; // win rate from P1 perspective (for consistent blue/red display)
  frequency: number; // count / total
}

/** A turn groups one or more atomic actions by the same player. */
export interface Turn {
  player: number; // 0 or 1
  /** Algebraic labels for each action in this turn (e.g. ["d1-c1", "End Turn"] or ["f1-e3"]) */
  labels: string[];
  /** Combined display label (e.g. "d1-c1" or "d1-c1, c1-b1") */
  display: string;
  /** Index into positionStack where this turn starts */
  startIndex: number;
  /** Index into positionStack after this turn ends */
  endIndex: number;
}

export interface OpeningExplorerState {
  loading: boolean;
  error: string | null;
  bookSize: number;
  currentState: EngineState;
  currentIndex: number; // index into positionStack (0 = start)
  historyLength: number; // total positions in stack
  bookMoves: BookMove[];
  boardState: BoardState;
  turns: Turn[]; // move history grouped by player turn
  currentTurnIndex: number; // which turn we're currently in/after (-1 = before any)
  openingName: string | null; // named opening for current line, if any
  playMove: (actions: number[]) => void;
  goBack: () => void;
  goForward: () => void;
  goToStart: () => void;
  goToPosition: (index: number) => void;
}

function engineToBoardState(state: EngineState): BoardState {
  return {
    p1_pieces: state.pieces[0].toString(),
    p1_ball: state.balls[0].toString(),
    p2_pieces: state.pieces[1].toString(),
    p2_ball: state.balls[1].toString(),
  };
}

function getMoveLabel(state: EngineState, action: number): { label: string; type: 'knight' | 'pass' | 'end_turn' } {
  if (action === -1 || action === END_TURN_ACTION) {
    return { label: 'End Turn', type: 'end_turn' };
  }

  const src = Math.floor(action / NUM_SQUARES);
  const dst = action % NUM_SQUARES;
  const p = state.currentPlayer;
  const srcBit = 1n << BigInt(src);

  const isPass = (state.balls[p] & srcBit) !== 0n;
  return {
    label: `${sqToAlgebraic(src)}-${sqToAlgebraic(dst)}`,
    type: isPass ? 'pass' : 'knight',
  };
}

/**
 * Trace pass chains through the book to find all complete turns
 * starting from a mid-chain position. Returns chains ending with END_TURN.
 */
function tracePassChains(
  book: OpeningBook,
  state: EngineState,
  actionsSoFar: number[],
  pathSoFar: string[],
  depth = 0,
): { actions: number[]; label: string; count: number; winRate: number }[] {
  if (depth > 10) return []; // safety limit
  const raw = book.lookupRaw(state);
  if (!raw) return [];

  const results: { actions: number[]; label: string; count: number; winRate: number }[] = [];
  for (const [action, count, winRate] of raw.moves) {
    const { type } = getMoveLabel(state, action);

    if (type === 'end_turn') {
      // Chain complete — use END_TURN's count and winRate
      results.push({
        actions: [...actionsSoFar, action],
        label: pathSoFar.join('-'),
        count,
        winRate,
      });
    } else if (type === 'pass') {
      // Continue chain — append destination square to path
      const dst = action % NUM_SQUARES;
      const nextState = copyState(state);
      applyMove(nextState, action);
      const subChains = tracePassChains(
        book,
        nextState,
        [...actionsSoFar, action],
        [...pathSoFar, sqToAlgebraic(dst)],
        depth + 1,
      );
      results.push(...subChains);
    }
    // Knight moves in mid-chain shouldn't happen, ignore
  }

  return results;
}

/**
 * Build turn-level moves from book data at the given position.
 * Knight moves are complete turns (single action).
 * Pass chains are traced through the book and grouped into single entries.
 */
function buildTurnMoves(book: OpeningBook, state: EngineState): BookMove[] {
  const raw = book.lookupRaw(state);
  if (!raw) return [];

  const turnMoves: BookMove[] = [];

  for (const [action, count, winRate] of raw.moves) {
    const { label, type } = getMoveLabel(state, action);
    const p1WinRate = state.currentPlayer === 0 ? winRate : 1 - winRate;

    if (type === 'knight') {
      turnMoves.push({
        actions: [action],
        label,
        type: 'knight',
        count,
        winRate: p1WinRate,
        frequency: 0,
      });
    } else if (type === 'end_turn') {
      turnMoves.push({
        actions: [action],
        label: 'End Turn',
        type: 'end_turn',
        count,
        winRate: p1WinRate,
        frequency: 0,
      });
    } else {
      // Pass — trace complete chains through the book
      const src = Math.floor(action / NUM_SQUARES);
      const dst = action % NUM_SQUARES;
      const nextState = copyState(state);
      applyMove(nextState, action);
      const chains = tracePassChains(book, nextState, [action], [sqToAlgebraic(src), sqToAlgebraic(dst)]);

      if (chains.length > 0) {
        for (const chain of chains) {
          const chainP1WinRate = state.currentPlayer === 0 ? chain.winRate : 1 - chain.winRate;
          turnMoves.push({
            actions: chain.actions,
            label: chain.label,
            type: 'pass',
            count: chain.count,
            winRate: chainP1WinRate,
            frequency: 0,
          });
        }
      } else {
        // Chain goes out of book — show as atomic pass
        turnMoves.push({
          actions: [action],
          label,
          type: 'pass',
          count,
          winRate: p1WinRate,
          frequency: 0,
        });
      }
    }
  }

  // Compute frequencies
  const total = turnMoves.reduce((sum, m) => sum + m.count, 0);
  for (const m of turnMoves) {
    m.frequency = total > 0 ? m.count / total : 0;
  }

  // Sort by count descending
  turnMoves.sort((a, b) => b.count - a.count);

  return turnMoves;
}

/**
 * Group atomic moves into player turns.
 * A turn ends when the current player changes (knight move or end turn).
 */
function buildTurns(
  positionStack: EngineState[],
  moveStack: number[],
  upTo: number,
): Turn[] {
  const turns: Turn[] = [];
  let i = 0;
  while (i < upTo) {
    const player = positionStack[i].currentPlayer;
    const startIndex = i;
    const labels: string[] = [];

    while (i < upTo) {
      const { label, type } = getMoveLabel(positionStack[i], moveStack[i]);
      // Don't include "End Turn" in the display — it's implicit
      if (type !== 'end_turn') {
        labels.push(label);
      }
      i++;
      // Turn ends when player changes
      if (positionStack[i].currentPlayer !== player) {
        break;
      }
    }

    // For pass chains, show as a path (d8-e8-c6) instead of comma-separated
    let display: string;
    if (labels.length === 0) {
      display = 'End Turn';
    } else if (labels.length === 1) {
      display = labels[0];
    } else {
      // Multiple moves = pass chain — merge into path format
      const parts = [labels[0].split('-')[0]];
      for (const l of labels) {
        parts.push(l.split('-')[1]);
      }
      display = parts.join('-');
    }

    turns.push({
      player,
      labels,
      display,
      startIndex,
      endIndex: i,
    });
  }
  return turns;
}

export function useOpeningExplorer(): OpeningExplorerState {
  const [book, setBook] = useState<OpeningBook | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Position stack: positionStack[0] is the starting position
  // moveStack[i] is the action that leads from positionStack[i] to positionStack[i+1]
  const [positionStack, setPositionStack] = useState<EngineState[]>([newGame()]);
  const [moveStack, setMoveStack] = useState<number[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);

  // Load book on mount
  useEffect(() => {
    let cancelled = false;
    OpeningBook.load(`${API_BASE}/opening-book`)
      .then((b) => {
        if (!cancelled) {
          setBook(b);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err.message);
          setLoading(false);
        }
      });
    return () => { cancelled = true; };
  }, []);

  const currentState = positionStack[currentIndex];

  const bookMoves = useMemo((): BookMove[] => {
    if (!book) return [];
    return buildTurnMoves(book, currentState);
  }, [book, currentState]);

  const turns = useMemo(
    () => buildTurns(positionStack, moveStack, currentIndex),
    [positionStack, moveStack, currentIndex],
  );

  // Which turn the currentIndex falls within or after
  const currentTurnIndex = useMemo(() => {
    for (let t = turns.length - 1; t >= 0; t--) {
      if (currentIndex >= turns[t].endIndex) return t;
    }
    return -1;
  }, [turns, currentIndex]);

  // Named opening for current line
  const openingName = useMemo(
    () => getOpeningName(turns.map((t) => t.display)),
    [turns],
  );

  const playMove = useCallback((actions: number[]) => {
    setPositionStack((prev) => {
      const newStack = prev.slice(0, currentIndex + 1);
      let state = prev[currentIndex];
      for (const action of actions) {
        const next = copyState(state);
        const internalAction = action === END_TURN_ACTION ? -1 : action;
        applyMove(next, internalAction);
        newStack.push(next);
        state = next;
      }
      return newStack;
    });
    setMoveStack((prev) => {
      const newMoves = prev.slice(0, currentIndex);
      for (const action of actions) {
        const internalAction = action === END_TURN_ACTION ? -1 : action;
        newMoves.push(internalAction);
      }
      return newMoves;
    });
    setCurrentIndex((prev) => prev + actions.length);
  }, [currentIndex]);

  const goBack = useCallback(() => {
    setCurrentIndex((prev) => Math.max(0, prev - 1));
  }, []);

  const goForward = useCallback(() => {
    setCurrentIndex((prev) => Math.min(positionStack.length - 1, prev + 1));
  }, [positionStack.length]);

  const goToStart = useCallback(() => {
    setCurrentIndex(0);
  }, []);

  const goToPosition = useCallback((index: number) => {
    setCurrentIndex(Math.max(0, Math.min(positionStack.length - 1, index)));
  }, [positionStack.length]);

  const boardState = useMemo(() => engineToBoardState(currentState), [currentState]);

  return {
    loading,
    error,
    bookSize: book?.size ?? 0,
    currentState,
    currentIndex,
    historyLength: positionStack.length,
    bookMoves,
    boardState,
    turns,
    currentTurnIndex,
    openingName,
    playMove,
    goBack,
    goForward,
    goToStart,
    goToPosition,
  };
}
