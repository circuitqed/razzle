import { describe, it, expect } from 'vitest';
import { search } from '../mcts';
import type { Evaluator } from '../evaluator';
import { newGame, type EngineState } from '../state';
import { getLegalMoves } from '../moves';
import { NUM_ACTIONS } from '../bitboard';

/** Uniform priors except one favoured move; value 0 everywhere. */
function favouring(move: number): Evaluator {
  return {
    async evaluate(state: EngineState) {
      const policy = new Float32Array(NUM_ACTIONS);
      for (const m of getLegalMoves(state)) if (m >= 0) policy[m] = 0.01;
      policy[move] = 0.9;
      return { policy, value: 0 };
    },
  };
}

describe('low-sim search follows the network', () => {
  // Levels 1-7 search 1-32 sims. The first simulation used to see zero root
  // visits (no exploration term), ignore the priors and take the first legal
  // move, so 1-4 sim levels played the lowest-numbered move every turn.
  const legal = getLegalMoves(newGame()).filter((m) => m >= 0);
  const favourite = legal[legal.length - 1]; // last in move order, never the default

  for (const sims of [1, 2, 4, 8]) {
    it(`plays the top-prior move at ${sims} sims`, async () => {
      const res = await search(newGame(), favouring(favourite), { numSimulations: sims });
      expect(res.bestMove).toBe(favourite);
    });
  }
});
