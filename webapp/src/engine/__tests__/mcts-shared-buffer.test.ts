import { describe, it, expect } from 'vitest';
import { search, type MCTSNode } from '../mcts';
import type { Evaluator } from '../evaluator';
import { newGame, applyMove, type EngineState } from '../state';
import { NUM_ACTIONS, END_TURN_ACTION } from '../bitboard';
import { getLegalMoves } from '../moves';

/** Position-dependent, sharply peaked priors over the legal moves. */
function h(s: EngineState, m: number): number {
  const v = Number((s.pieces[0] ^ (s.pieces[1] << 3n) ^ s.balls[0] ^ (s.balls[1] << 7n)) % 1000003n) + m * 7919;
  return ((v * 2654435761) >>> 0) / 4294967296 + 0.01;
}

function fill(s: EngineState, out: Float32Array): number {
  out.fill(0);
  for (const m of getLegalMoves(s)) out[m === -1 ? END_TURN_ACTION : m] = h(s, m) ** 4;
  return h(s, 1) - 0.5;
}

/** Reuses one output buffer across calls, like PureTSEvaluator. */
class SharedBufferEvaluator implements Evaluator {
  private buf = new Float32Array(NUM_ACTIONS);
  async evaluate(state: EngineState) {
    const value = fill(state, this.buf);
    return { policy: this.buf, value };
  }
}

/** Fresh array per call (always correct). */
class CopyingEvaluator implements Evaluator {
  async evaluate(state: EngineState) {
    const p = new Float32Array(NUM_ACTIONS);
    const value = fill(state, p);
    return { policy: p, value };
  }
}

function midgame(): EngineState {
  const s = newGame();
  let seed = 7;
  for (let i = 0; i < 9; i++) {
    const m = getLegalMoves(s);
    seed = (seed * 48271) % 2147483647;
    applyMove(s, m[seed % m.length]);
  }
  return s;
}

function visits(root: MCTSNode): Map<number, number> {
  return new Map([...root.children].map(([m, c]) => [m, c.visitCount]));
}

describe('batched MCTS with evaluators that reuse an output buffer', () => {
  it('expands each leaf with its own priors (same result as a copying evaluator)', async () => {
    const cfg = { numSimulations: 400, batchSize: 8, earlyTermination: false, temperature: 0 };
    const shared = await search(midgame(), new SharedBufferEvaluator(), cfg);
    const copying = await search(midgame(), new CopyingEvaluator(), cfg);
    expect(visits(shared.rootNode)).toEqual(visits(copying.rootNode));
    expect(shared.bestMove).toBe(copying.bestMove);
  });
});
