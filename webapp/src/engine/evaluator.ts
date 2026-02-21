/**
 * Evaluator interfaces and implementations for Razzle Dazzle MCTS.
 *
 * - OnnxEvaluator: Uses ONNX Runtime Web for neural network inference.
 * - PureTSEvaluator: Pure TypeScript inference — no WASM, iOS-compatible.
 * - RandomEvaluator: Uniform policy over legal moves, value=0 (for testing).
 */

import { NUM_ACTIONS, END_TURN_ACTION } from './bitboard';
import { getLegalMoves } from './moves';
import { stateToTensor } from './tensor';
import { rotatePolicy180 } from './symmetry';
import type { EngineState } from './state';
import type { PureTSModel } from './inference';

export interface Evaluator {
  evaluate(state: EngineState): Promise<{ policy: Float32Array; value: number }>;
}

/**
 * ONNX Runtime Web evaluator.
 *
 * Wraps an ort.InferenceSession and runs neural network inference.
 * The network outputs log-probabilities; we exponentiate to get probabilities.
 * For player 1, we rotate the policy back to the original perspective.
 */
export class OnnxEvaluator implements Evaluator {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private session: any;

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  constructor(session: any) {
    this.session = session;
  }

  async evaluate(
    state: EngineState,
  ): Promise<{ policy: Float32Array; value: number }> {
    const tensor = stateToTensor(state);

    // Create ONNX tensor: shape [1, 7, 8, 7]
    // ort is loaded in the worker context
    const ort = self.ort ?? (await import('onnxruntime-web'));
    const inputTensor = new ort.Tensor('float32', tensor, [1, 7, 8, 7]);
    const feeds = { board_input: inputTensor };

    const results = await this.session.run(feeds);

    // Copy data out of WASM heap BEFORE disposing tensors.
    // Tensor .data is a view into WASM linear memory — reading after
    // dispose is use-after-free that corrupts/crashes after a few inferences.
    const logPolicy = new Float32Array(results.policy.data as Float32Array);
    const value = (results.value.data as Float32Array)[0];

    // Dispose ONNX tensors to free WASM heap memory.
    inputTensor.dispose();
    results.policy.dispose();
    results.value.dispose();

    // Exponentiate log-softmax to get probabilities
    let policy = new Float32Array(NUM_ACTIONS);
    for (let i = 0; i < NUM_ACTIONS; i++) {
      policy[i] = Math.exp(logPolicy[i]);
    }

    // Rotate policy back for player 1 (the network sees a rotated board)
    if (state.currentPlayer === 1) {
      policy = rotatePolicy180(policy);
    }

    return { policy, value };
  }
}

/**
 * Pure TypeScript evaluator — no WASM, no SharedArrayBuffer.
 *
 * Uses PureTSModel for inference (im2col + GEMM in plain JS).
 * This is the iOS-compatible inference path. On desktop, OnnxEvaluator
 * is preferred for its WASM SIMD acceleration.
 */
export class PureTSEvaluator implements Evaluator {
  private model: PureTSModel;

  constructor(model: PureTSModel) {
    this.model = model;
  }

  async evaluate(
    state: EngineState,
  ): Promise<{ policy: Float32Array; value: number }> {
    const tensor = stateToTensor(state);

    const { policy: logPolicy, value } = this.model.forward(tensor);

    // Exponentiate log-softmax to get probabilities
    let policy = new Float32Array(NUM_ACTIONS);
    for (let i = 0; i < NUM_ACTIONS; i++) {
      policy[i] = Math.exp(logPolicy[i]);
    }

    // Rotate policy back for player 1 (the network sees a rotated board)
    if (state.currentPlayer === 1) {
      policy = rotatePolicy180(policy);
    }

    return { policy, value };
  }
}

/**
 * Random evaluator for testing without a model.
 * Returns uniform policy over legal moves and value = 0.
 */
export class RandomEvaluator implements Evaluator {
  async evaluate(
    state: EngineState,
  ): Promise<{ policy: Float32Array; value: number }> {
    const legalMoves = getLegalMoves(state);
    const policy = new Float32Array(NUM_ACTIONS);

    if (legalMoves.length > 0) {
      const prob = 1.0 / legalMoves.length;
      for (const move of legalMoves) {
        const idx = move === -1 ? END_TURN_ACTION : move;
        policy[idx] = prob;
      }
    }

    return { policy, value: 0.0 };
  }
}

// Declare ort on globalThis for worker context
declare global {
  // eslint-disable-next-line no-var
  var ort: typeof import('onnxruntime-web') | undefined;
}
