/**
 * Monte Carlo Tree Search for Razzle Dazzle (browser).
 *
 * Single-threaded PUCT implementation for client-side inference.
 * Async because ONNX inference returns Promises.
 */

import { END_TURN_ACTION } from './bitboard';
import { getLegalMoves } from './moves';
import {
  copyState,
  applyMove,
  isTerminal,
  getResult,
  type EngineState,
} from './state';
import type { Evaluator } from './evaluator';

export interface MCTSConfig {
  numSimulations: number;
  cPuct: number;
  temperature: number;
  earlyTermination: boolean;
  visitConcentrationThreshold: number;
  minSimsForEarlyStop: number;
  checkEarlyStopInterval: number;
  passQuiescence: boolean;
  passQuiescenceMaxDepth: number;
  /** Batch size for parallel leaf collection with virtual loss. 0 = auto (~5% of numSimulations). */
  batchSize: number;
  /** Virtual loss applied during batched selection to encourage exploration diversity. */
  virtualLoss: number;
}

export const DEFAULT_CONFIG: MCTSConfig = {
  numSimulations: 256,
  cPuct: 1.5,
  temperature: 0,
  earlyTermination: true,
  visitConcentrationThreshold: 0.85,
  minSimsForEarlyStop: 100,
  checkEarlyStopInterval: 50,
  passQuiescence: true,
  passQuiescenceMaxDepth: 10,
  batchSize: 0,
  virtualLoss: 3,
};

function autoBatchSize(numSims: number): number {
  const target = numSims * 0.05;
  if (target <= 4) return 1; // disable batching for very low sim counts
  if (target <= 8) return 8;
  return 1 << Math.floor(Math.log2(target));
}

export interface MCTSNode {
  state: EngineState;
  prior: number;
  visitCount: number;
  valueSum: number;
  virtualLoss: number;
  children: Map<number, MCTSNode>;
  isExpanded: boolean;
}

function createNode(state: EngineState, prior: number = 0): MCTSNode {
  return {
    state,
    prior,
    visitCount: 0,
    valueSum: 0,
    virtualLoss: 0,
    children: new Map(),
    isExpanded: false,
  };
}

function nodeValue(node: MCTSNode): number {
  if (node.visitCount === 0) return 0;
  return node.valueSum / node.visitCount;
}

function ucbScore(
  child: MCTSNode,
  parentVisits: number,
  parentPlayer: number,
  cPuct: number,
): number {
  // Use adjusted visit counts (includes virtual loss) for exploration diversity
  const adjustedChildVisits = child.visitCount + child.virtualLoss;
  const exploration =
    cPuct *
    child.prior *
    Math.sqrt(parentVisits) /
    (1 + adjustedChildVisits);
  const q = nodeValue(child);
  if (child.state.currentPlayer !== parentPlayer) {
    return -q + exploration;
  }
  return q + exploration;
}

function selectChild(
  node: MCTSNode,
  cPuct: number,
): [number, MCTSNode] {
  let bestScore = -Infinity;
  let bestAction = -1;
  let bestChild: MCTSNode | null = null;
  const parentPlayer = node.state.currentPlayer;
  const adjustedParentVisits = node.visitCount + node.virtualLoss;

  for (const [action, child] of node.children) {
    const score = ucbScore(child, adjustedParentVisits, parentPlayer, cPuct);
    if (score > bestScore) {
      bestScore = score;
      bestAction = action;
      bestChild = child;
    }
  }

  return [bestAction, bestChild!];
}

function expandNode(node: MCTSNode, policy: Float32Array): void {
  if (node.isExpanded) return;

  const legalMoves = getLegalMoves(node.state);

  const getPrior = (move: number): number => {
    if (move === -1) return policy[END_TURN_ACTION];
    return policy[move];
  };

  let policySum = 0;
  for (const move of legalMoves) {
    policySum += getPrior(move);
  }

  for (const move of legalMoves) {
    const childState = copyState(node.state);
    applyMove(childState, move);
    const prior = policySum > 0 ? getPrior(move) / policySum : 1.0 / legalMoves.length;
    const child = createNode(childState, prior);

    // Seed terminal children with actual game result so MCTS
    // immediately knows about wins/losses regardless of NN prior
    if (isTerminal(childState)) {
      const result = getResult(childState, childState.currentPlayer);
      const value = 2 * result - 1; // [-1, +1] from child's perspective
      child.visitCount = 1;
      child.valueSum = value;
    }

    node.children.set(move, child);
  }

  node.isExpanded = true;
}

export interface SearchResult {
  bestMove: number;
  rootNode: MCTSNode;
  simsDone: number;
  value: number;
}

export interface SearchProgress {
  simsDone: number;
  totalSims: number;
  bestMove: number;
  value: number;
}

/**
 * Run MCTS search. Async because ONNX inference is async.
 *
 * @param abortSignal - If provided, checked each simulation; if true, search stops early.
 * @param onProgress - Called periodically with search progress.
 */
export async function search(
  state: EngineState,
  evaluator: Evaluator,
  config: Partial<MCTSConfig> = {},
  abortSignal?: { aborted: boolean },
  onProgress?: (progress: SearchProgress) => void,
): Promise<SearchResult> {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  const root = createNode(copyState(state));

  // Evaluate and expand root
  const { policy: rootPolicy, value: rootValue } = await evaluator.evaluate(
    root.state,
  );
  expandNode(root, rootPolicy);

  // Check for immediate winning move among children (e.g. winning pass)
  // This catches wins that the neural net might give low prior to
  const rootPlayer = root.state.currentPlayer;
  for (const [action, child] of root.children) {
    if (isTerminal(child.state)) {
      const result = getResult(child.state, rootPlayer);
      if (result === 1.0) {
        return { bestMove: action, rootNode: root, simsDone: 0, value: 1.0 };
      }
    }
  }

  // Pass quiescence for root
  let initialValue = rootValue;
  if (cfg.passQuiescence && root.state.hasPassed) {
    initialValue = await quiescenceSearch(root, evaluator, cfg, 0);
    root.visitCount = 1;
    root.valueSum = initialValue;
  }

  let simsDone = 0;
  const progressInterval = 50;
  const batchSize = cfg.batchSize > 0 ? cfg.batchSize : autoBatchSize(cfg.numSimulations);

  while (simsDone < cfg.numSimulations) {
    if (abortSignal?.aborted) break;

    const currentBatch = Math.min(batchSize, cfg.numSimulations - simsDone);

    if (currentBatch <= 1) {
      // Single simulation (no batching overhead)
      await simulate(root, evaluator, cfg);
      simsDone++;
    } else {
      // Batched: collect multiple leaves with virtual loss, evaluate together
      await simulateBatch(root, evaluator, cfg, currentBatch);
      simsDone += currentBatch;
    }

    // Report progress periodically
    if (onProgress && simsDone % progressInterval < currentBatch) {
      const best = getBestMoveFromRoot(root);
      onProgress({
        simsDone,
        totalSims: cfg.numSimulations,
        bestMove: best.move,
        value: best.value,
      });
    }

    // Early termination check
    if (
      cfg.earlyTermination &&
      simsDone >= cfg.minSimsForEarlyStop &&
      simsDone % cfg.checkEarlyStopInterval < currentBatch
    ) {
      if (shouldStopEarly(root, cfg)) break;
    }
  }

  const best = getBestMoveFromRoot(root);

  return {
    bestMove: best.move,
    rootNode: root,
    simsDone,
    value: best.value,
  };
}

function getBestMoveFromRoot(root: MCTSNode): { move: number; value: number } {
  let bestMove = -1;
  let bestVisits = -1;
  let bestValue = 0;
  const rootPlayer = root.state.currentPlayer;

  for (const [action, child] of root.children) {
    if (child.visitCount > bestVisits) {
      bestVisits = child.visitCount;
      bestMove = action;
      // Convert to root player's perspective
      const q = nodeValue(child);
      bestValue = child.state.currentPlayer !== rootPlayer ? -q : q;
    }
  }

  return { move: bestMove, value: bestValue };
}

async function simulate(
  root: MCTSNode,
  evaluator: Evaluator,
  cfg: MCTSConfig,
): Promise<void> {
  let node = root;
  const path: MCTSNode[] = [node];

  // SELECT: traverse tree to leaf
  while (node.isExpanded && node.children.size > 0) {
    if (isTerminal(node.state)) break;
    [, node] = selectChild(node, cfg.cPuct);
    path.push(node);
  }

  let leafPlayer: number;
  let value: number;

  // EVALUATE
  if (isTerminal(node.state)) {
    leafPlayer = node.state.currentPlayer;
    const result = getResult(node.state, leafPlayer);
    value = 2 * result - 1;
  } else {
    const { policy, value: evalValue } = await evaluator.evaluate(node.state);
    expandNode(node, policy);

    // Pass quiescence
    if (cfg.passQuiescence && node.state.hasPassed) {
      value = await quiescenceSearch(node, evaluator, cfg, 0);
    } else {
      value = evalValue;
    }
    leafPlayer = node.state.currentPlayer;
  }

  // BACKUP
  for (const n of path) {
    n.visitCount++;
    if (n.state.currentPlayer === leafPlayer) {
      n.valueSum += value;
    } else {
      n.valueSum -= value;
    }
  }
}

/**
 * Batched simulation: collect multiple leaves with virtual loss, evaluate
 * them together, then expand and backup. Mirrors the Python MCTS batching.
 */
async function simulateBatch(
  root: MCTSNode,
  evaluator: Evaluator,
  cfg: MCTSConfig,
  batchSize: number,
): Promise<void> {
  // Phase 1: SELECT — collect leaves with virtual loss
  const paths: MCTSNode[][] = [];
  const leaves: MCTSNode[] = [];
  const leafPlayers: number[] = [];
  const terminalBackups: Array<{ path: MCTSNode[]; value: number; leafPlayer: number }> = [];

  for (let i = 0; i < batchSize; i++) {
    let node = root;
    const path: MCTSNode[] = [node];

    // Traverse tree, applying virtual loss to encourage diverse exploration
    while (node.isExpanded && node.children.size > 0) {
      if (isTerminal(node.state)) break;
      node.virtualLoss += cfg.virtualLoss;
      [, node] = selectChild(node, cfg.cPuct);
      path.push(node);
    }

    node.virtualLoss += cfg.virtualLoss;

    if (isTerminal(node.state)) {
      const leafPlayer = node.state.currentPlayer;
      const result = getResult(node.state, leafPlayer);
      const value = 2 * result - 1;
      terminalBackups.push({ path, value, leafPlayer });
    } else {
      paths.push(path);
      leaves.push(node);
      leafPlayers.push(node.state.currentPlayer);
    }
  }

  // Phase 2: EVALUATE — all non-terminal leaves (batched if supported)
  if (leaves.length > 0) {
    let results: Array<{ policy: Float32Array; value: number }>;
    if (evaluator.evaluateBatch) {
      results = await evaluator.evaluateBatch(leaves.map(l => l.state));
    } else {
      // Sequential evaluation — ONNX Runtime may not handle concurrent session.run()
      results = [];
      for (const leaf of leaves) {
        results.push(await evaluator.evaluate(leaf.state));
      }
    }

    // Phase 3: EXPAND and BACKUP
    for (let i = 0; i < leaves.length; i++) {
      const leaf = leaves[i];
      const { policy, value: evalValue } = results[i];
      const path = paths[i];
      const leafPlayer = leafPlayers[i];

      // Expand
      if (!leaf.isExpanded) {
        expandNode(leaf, policy);
      }

      // Pass quiescence
      let value = evalValue;
      if (cfg.passQuiescence && leaf.state.hasPassed) {
        value = await quiescenceSearch(leaf, evaluator, cfg, 0);
      }

      // Backup: propagate value, remove virtual loss
      for (const n of path) {
        n.visitCount++;
        n.virtualLoss -= cfg.virtualLoss;
        if (n.state.currentPlayer === leafPlayer) {
          n.valueSum += value;
        } else {
          n.valueSum -= value;
        }
      }
    }
  }

  // Phase 4: BACKUP terminal paths
  for (const { path, value, leafPlayer } of terminalBackups) {
    for (const n of path) {
      n.visitCount++;
      n.virtualLoss -= cfg.virtualLoss;
      if (n.state.currentPlayer === leafPlayer) {
        n.valueSum += value;
      } else {
        n.valueSum -= value;
      }
    }
  }
}

async function quiescenceSearch(
  node: MCTSNode,
  evaluator: Evaluator,
  cfg: MCTSConfig,
  depth: number,
): Promise<number> {
  const callerPlayer = node.state.currentPlayer;

  if (depth >= cfg.passQuiescenceMaxDepth) {
    const { value } = await evaluator.evaluate(node.state);
    return value;
  }

  if (isTerminal(node.state)) {
    const result = getResult(node.state, callerPlayer);
    return 2 * result - 1;
  }

  // Turn ended (hasPassed=false) - evaluate this position
  if (!node.state.hasPassed) {
    const { value } = await evaluator.evaluate(node.state);
    if (node.state.currentPlayer !== callerPlayer) {
      return -value;
    }
    return value;
  }

  // Mid-pass: search all children and take max
  if (node.children.size === 0) {
    const { value } = await evaluator.evaluate(node.state);
    return value;
  }

  let bestValue = -Infinity;
  for (const [, child] of node.children) {
    if (!child.isExpanded && !isTerminal(child.state)) {
      const { policy } = await evaluator.evaluate(child.state);
      expandNode(child, policy);
    }

    let childValue = await quiescenceSearch(child, evaluator, cfg, depth + 1);

    if (child.state.currentPlayer !== callerPlayer) {
      childValue = -childValue;
    }

    // Seed child's Q value
    if (child.visitCount === 0) {
      child.visitCount = 1;
      if (child.state.currentPlayer !== callerPlayer) {
        child.valueSum = -childValue;
      } else {
        child.valueSum = childValue;
      }
    }

    bestValue = Math.max(bestValue, childValue);
  }

  return bestValue;
}

function shouldStopEarly(root: MCTSNode, cfg: MCTSConfig): boolean {
  if (root.children.size === 0) return true;

  let totalVisits = 0;
  let maxVisits = 0;
  for (const [, child] of root.children) {
    totalVisits += child.visitCount;
    if (child.visitCount > maxVisits) {
      maxVisits = child.visitCount;
    }
  }

  if (totalVisits === 0) return false;
  return maxVisits / totalVisits >= cfg.visitConcentrationThreshold;
}

/**
 * Get visit-count-based policy from root node.
 * Returns Float32Array of length NUM_ACTIONS.
 *
 * @param temperature - Controls policy sharpness:
 *   0: one-hot on highest-visited move (greedy)
 *   1: proportional to raw visit counts
 *   >0, !=1: visits^(1/temp) then normalize
 */
export function getPolicy(root: MCTSNode, temperature: number = 1.0): Float32Array {
  const policySize = END_TURN_ACTION + 1; // 3137
  const policy = new Float32Array(policySize);

  const actionToIndex = (action: number): number =>
    action === -1 ? END_TURN_ACTION : action;

  let totalVisits = 0;
  for (const [, child] of root.children) {
    totalVisits += child.visitCount;
  }

  if (totalVisits > 0 && temperature > 0) {
    if (temperature === 1.0) {
      // Proportional to visits
      for (const [action, child] of root.children) {
        policy[actionToIndex(action)] = child.visitCount / totalVisits;
      }
    } else {
      // Apply temperature: visits^(1/temp) then normalize
      const invTemp = 1.0 / temperature;
      let sum = 0;
      const scaled: Array<[number, number]> = [];
      for (const [action, child] of root.children) {
        const v = Math.pow(child.visitCount, invTemp);
        scaled.push([action, v]);
        sum += v;
      }
      for (const [action, v] of scaled) {
        policy[actionToIndex(action)] = v / sum;
      }
    }
  } else if (root.children.size > 0) {
    // Greedy (temperature == 0): one-hot on highest visits
    let bestAction = -1;
    let bestVisits = -1;
    for (const [action, child] of root.children) {
      if (child.visitCount > bestVisits) {
        bestVisits = child.visitCount;
        bestAction = action;
      }
    }
    policy[actionToIndex(bestAction)] = 1.0;
  }

  return policy;
}

/**
 * Select a move from the root node based on visit counts and temperature.
 *
 * @param temperature - Controls selection:
 *   0: return highest-visited move (greedy/deterministic)
 *   >0: sample from temperature-scaled visit distribution
 */
export function selectMove(root: MCTSNode, temperature: number = 0): number {
  if (root.children.size === 0) {
    throw new Error('No legal moves');
  }

  const actions: number[] = [];
  const visits: number[] = [];
  for (const [action, child] of root.children) {
    actions.push(action);
    visits.push(child.visitCount);
  }

  if (temperature === 0) {
    // Greedy: pick highest visit count
    let bestIdx = 0;
    for (let i = 1; i < visits.length; i++) {
      if (visits[i] > visits[bestIdx]) {
        bestIdx = i;
      }
    }
    return actions[bestIdx];
  }

  // Sample from temperature-scaled distribution
  const totalVisits = visits.reduce((a, b) => a + b, 0);
  if (totalVisits === 0) {
    // Uniform random
    return actions[Math.floor(Math.random() * actions.length)];
  }

  let probs: number[];
  if (temperature === 1.0) {
    probs = visits.map((v) => v / totalVisits);
  } else {
    const invTemp = 1.0 / temperature;
    const scaled = visits.map((v) => Math.pow(v, invTemp));
    const sum = scaled.reduce((a, b) => a + b, 0);
    probs = scaled.map((v) => v / sum);
  }

  // Weighted random selection
  const r = Math.random();
  let cumulative = 0;
  for (let i = 0; i < probs.length; i++) {
    cumulative += probs[i];
    if (r < cumulative) {
      return actions[i];
    }
  }
  return actions[actions.length - 1];
}
