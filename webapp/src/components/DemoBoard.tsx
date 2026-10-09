import { useEffect, useMemo, useState } from 'react';
import Board from './Board';
import { newGame, applyMove, copyState, type EngineState } from '../engine/state';
import { decodeMove } from '../types';

/**
 * A short game between two KnightBall AIs (blue: distill_v2_48x6 @64 sims,
 * red: distill_v2_16x2 @4), replayed on a loop for the landing page. It
 * shows knight moves, a four-pass chain (d1-d3-c3-c6-c7) and a goal.
 * Moves are src*56+dst; -1 ends a pass turn.
 */
export const DEMO_GAME: number[] = [
  298, 3063, 72, 2837, 1039, 2892, 129, 2948, 2053, 2111, -1, 984, 2037, 241,
  2096, 1780, 1184, 1829, 469, 185, 968, 933, 2116, -1, 1184, 2122, 469, 2514,
];

const STEP_MS = 1100;
const END_PAUSE_MS = 2600;

function boardOf(s: EngineState) {
  return {
    p1_pieces: s.pieces[0].toString(),
    p1_ball: s.balls[0].toString(),
    p2_pieces: s.pieces[1].toString(),
    p2_ball: s.balls[1].toString(),
  };
}

export default function DemoBoard({ moves = DEMO_GAME }: { moves?: number[] }) {
  // Every position of the game, computed once.
  const positions = useMemo(() => {
    const out: EngineState[] = [newGame()];
    for (const m of moves) {
      const next = copyState(out[out.length - 1]);
      applyMove(next, m);
      out.push(next);
    }
    return out;
  }, [moves]);

  const reducedMotion = typeof window !== 'undefined'
    && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  const [step, setStep] = useState(0);

  useEffect(() => {
    if (reducedMotion || moves.length === 0) return;
    const atEnd = step >= moves.length;
    // End-of-turn markers don't move anything: take them without a pause.
    const delay = atEnd ? END_PAUSE_MS : moves[step] === -1 ? 0 : STEP_MS;
    const id = setTimeout(() => setStep((s) => (s >= moves.length ? 0 : s + 1)), delay);
    return () => clearTimeout(id);
  }, [step, moves, reducedMotion]);

  // With reduced motion, show the position just before the goal.
  const shown = reducedMotion ? Math.max(0, moves.length - 1) : step;
  const state = positions[shown];
  const last = shown > 0 ? moves[shown - 1] : -1;
  const lastMove = last >= 0 ? decodeMove(last) : null;
  const scored = shown === moves.length && moves.length > 0;

  return (
    <div className="relative">
      <Board
        board={boardOf(state)}
        currentPlayer={state.currentPlayer as 0 | 1}
        legalMoves={[]}
        selectedSquare={null}
        onSquareClick={() => {}}
        touchedMask={state.touchedMask.toString()}
        lastMove={lastMove ? { from: lastMove.src, to: lastMove.dst } : null}
        animate={!reducedMotion}
        fluid
      />
      {scored && (
        <div className="absolute inset-x-0 top-[42%] flex justify-center pointer-events-none">
          <span className="rounded-full bg-gray-900/85 px-4 py-1.5 text-sm font-semibold text-blue-300 shadow-lg">
            Blue scores!
          </span>
        </div>
      )}
    </div>
  );
}
