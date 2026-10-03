import { describe, it, expect } from 'vitest';
import { search, statesEqual, type MCTSNode } from '../mcts';
import type { Evaluator } from '../evaluator';
import { newGame, applyMove, copyState, type EngineState } from '../state';
import { getLegalMoves, getPassMoves } from '../moves';
import { NUM_ACTIONS } from '../bitboard';

/** Deterministic evaluator (uniform policy, mild value) that counts state evaluations. */
function countingEvaluator(): Evaluator & { calls: number } {
  const ev = {
    calls: 0,
    async evaluate(state: EngineState) {
      ev.calls++;
      const policy = new Float32Array(NUM_ACTIONS).fill(1 / NUM_ACTIONS);
      return { policy, value: (state.ply % 7) / 10 - 0.3 };
    },
  };
  return ev;
}

/** Play deterministic knight moves until the side to move has a legal pass. */
function positionWithPass(): EngineState {
  const state = newGame();
  for (let i = 0; i < 60; i++) {
    if (getPassMoves(state).length > 0) return state;
    const knight = getLegalMoves(state).find(m => m >= 0 && !getPassMoves(state).includes(m))!;
    applyMove(state, knight);
  }
  throw new Error('no pass position found');
}

const CFG = { numSimulations: 400, earlyTermination: false, batchSize: 1, temperature: 0 };

/** The most-visited pass child of the root, if the root explored one. */
function visitedPassChild(root: MCTSNode): MCTSNode {
  const passes = getPassMoves(root.state);
  let best: MCTSNode | null = null;
  for (const m of passes) {
    const c = root.children.get(m);
    if (c && c.isExpanded && (!best || c.visitCount > best.visitCount)) best = c;
  }
  if (!best) throw new Error('no expanded pass child');
  return best;
}

describe('MCTS subtree reuse (pass chains)', () => {
  it('continues from a matching subtree with far fewer evaluations', async () => {
    const start = positionWithPass();
    const first = await search(start, countingEvaluator(), CFG);
    const child = visitedPassChild(first.rootNode);
    expect(child.state.currentPlayer).toBe(start.currentPlayer); // same player moves again

    const fresh = countingEvaluator();
    await search(copyState(child.state), fresh, CFG);

    const reused = countingEvaluator();
    const priorVisits = child.visitCount;
    const result = await search(copyState(child.state), reused, CFG, undefined, undefined, child);

    expect(priorVisits).toBeGreaterThan(0);
    expect(reused.calls).toBeLessThan(fresh.calls);
    expect(result.rootNode).toBe(child);
    expect(getLegalMoves(child.state)).toContain(result.bestMove);
  });

  it('ignores a subtree whose position does not match', async () => {
    const start = positionWithPass();
    const first = await search(start, countingEvaluator(), CFG);
    const child = visitedPassChild(first.rootNode);

    const fresh = countingEvaluator();
    const r = await search(copyState(start), fresh, CFG, undefined, undefined, child);
    expect(r.rootNode).not.toBe(child);
    expect(statesEqual(r.rootNode.state, start)).toBe(true);
  });
});
