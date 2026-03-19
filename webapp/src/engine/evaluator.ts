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
import type { WebGLModel } from './webglInference';
import type { GPUForwardPass } from './webglForwardPass';

export interface Evaluator {
  evaluate(state: EngineState): Promise<{ policy: Float32Array; value: number }>;
  /** Optional batch evaluation — processes multiple states in a single GPU call. */
  evaluateBatch?(states: EngineState[]): Promise<Array<{ policy: Float32Array; value: number }>>;
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
  private tensorBuf: Float32Array;
  private policyBuf: Float32Array;
  private evalCount = 0;
  private totalMs = 0;
  private lastReport = 0;

  constructor(model: PureTSModel) {
    this.model = model;
    this.tensorBuf = new Float32Array(7 * 8 * 7); // 392
    this.policyBuf = new Float32Array(NUM_ACTIONS);
  }

  async evaluate(
    state: EngineState,
  ): Promise<{ policy: Float32Array; value: number }> {
    const t0 = performance.now();
    stateToTensor(state, this.tensorBuf);

    const { policy: logPolicy, value } = this.model.forward(this.tensorBuf);
    this.totalMs += performance.now() - t0;
    this.evalCount++;
    // Log every 50 evals
    if (this.evalCount - this.lastReport >= 50) {
      const avg = (this.totalMs / this.evalCount).toFixed(1);
      const rate = (1000 / (this.totalMs / this.evalCount)).toFixed(1);
      console.log(`[PureTSEvaluator] ${this.evalCount} evals, avg ${avg}ms/eval, ${rate} evals/sec, total ${(this.totalMs / 1000).toFixed(1)}s`);
      this.lastReport = this.evalCount;
    }

    // Exponentiate log-softmax to get probabilities
    const policy = this.policyBuf;
    for (let i = 0; i < NUM_ACTIONS; i++) {
      policy[i] = Math.exp(logPolicy[i]);
    }

    // Rotate policy back for player 1 (the network sees a rotated board)
    if (state.currentPlayer === 1) {
      // rotatePolicy180 returns a new array — copy back into our buffer
      const rotated = rotatePolicy180(policy);
      policy.set(rotated);
    }

    return { policy, value };
  }
}

/**
 * WebGL-accelerated evaluator — uses custom WebGL GEMM for GPU inference.
 * Produces identical results to PureTSEvaluator (verified by tests).
 */
export class WebGLEvaluator implements Evaluator {
  private model: WebGLModel;
  private tensorBuf: Float32Array;
  private policyBuf: Float32Array;

  constructor(model: WebGLModel) {
    this.model = model;
    this.tensorBuf = new Float32Array(7 * 8 * 7);
    this.policyBuf = new Float32Array(NUM_ACTIONS);
  }

  async evaluate(
    state: EngineState,
  ): Promise<{ policy: Float32Array; value: number }> {
    stateToTensor(state, this.tensorBuf);

    const { policy: logPolicy, value } = this.model.forward(this.tensorBuf);

    const policy = this.policyBuf;
    for (let i = 0; i < NUM_ACTIONS; i++) {
      policy[i] = Math.exp(logPolicy[i]);
    }

    if (state.currentPlayer === 1) {
      const rotated = rotatePolicy180(policy);
      policy.set(rotated);
    }

    return { policy, value };
  }

  dispose(): void {
    this.model.dispose();
  }
}

/**
 * GPU-resident evaluator — keeps activations on GPU, only reads back final output.
 * Verified accurate to 3e-5 against CPU on 200 positions.
 */
export class GPUEvaluator implements Evaluator {
  private model: GPUForwardPass;
  private tensorBuf: Float32Array;
  private policyBuf: Float32Array;

  constructor(model: GPUForwardPass) {
    this.model = model;
    this.tensorBuf = new Float32Array(7 * 8 * 7);
    this.policyBuf = new Float32Array(NUM_ACTIONS);
  }

  async evaluate(
    state: EngineState,
  ): Promise<{ policy: Float32Array; value: number }> {
    stateToTensor(state, this.tensorBuf);
    const { policy: logPolicy, value } = this.model.forward(this.tensorBuf);

    const policy = this.policyBuf;
    for (let i = 0; i < NUM_ACTIONS; i++) {
      policy[i] = Math.exp(logPolicy[i]);
    }

    if (state.currentPlayer === 1) {
      const rotated = rotatePolicy180(policy);
      policy.set(rotated);
    }

    return { policy, value };
  }

  /**
   * Batch evaluation: run N forward passes in a single GPU submission.
   * All N inputs are stacked into one wide texture, processed together,
   * and read back with just 2 readPixels calls total.
   */
  async evaluateBatch(
    states: EngineState[],
  ): Promise<Array<{ policy: Float32Array; value: number }>> {
    if (states.length === 0) return [];

    // Convert states to tensors
    const tensors = states.map(s => {
      const buf = new Float32Array(7 * 8 * 7);
      stateToTensor(s, buf);
      return buf;
    });

    // Batched GPU forward pass
    const gpuResults = this.model.forwardBatch(tensors);

    // Post-process: exp + rotation
    return gpuResults.map((r, i) => {
      const policy = new Float32Array(NUM_ACTIONS);
      for (let j = 0; j < NUM_ACTIONS; j++) {
        policy[j] = Math.exp(r.policy[j]);
      }
      if (states[i].currentPlayer === 1) {
        const rotated = rotatePolicy180(policy);
        policy.set(rotated);
      }
      return { policy, value: r.value };
    });
  }

  dispose(): void {
    this.model.dispose();
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
