/**
 * Test entry point for WebGL inference validation.
 * Built as a separate Vite entry point so it can import all engine modules.
 * Runs full MCTS search with both CPU and GPU evaluators and compares results.
 */

import { createModelFromOnnx } from './engine/inference';
import { createGPUModelFromOnnx } from './engine/webglForwardPass';
import { PureTSEvaluator, GPUEvaluator } from './engine/evaluator';
import { search, DEFAULT_CONFIG } from './engine/mcts';
import type { MCTSConfig } from './engine/mcts';
import { getLegalMoves } from './engine/moves';
import { applyMove, copyState, newGame } from './engine/state';
import type { EngineState } from './engine/state';

// ---- UI helpers ----
const logEl = document.getElementById('log')!;
const resultEl = document.getElementById('result')!;

function p(msg: string, cls?: string) {
  const d = document.createElement('div');
  d.textContent = msg;
  if (cls) d.className = cls;
  logEl.appendChild(d);
  window.scrollTo(0, document.body.scrollHeight);
}

function fmtMove(idx: number): string {
  if (idx === -1 || idx === 3136) return 'END_TURN';
  const src = Math.floor(idx / 56), dst = idx % 56;
  const cols = 'abcdefg';
  return cols[src % 7] + (Math.floor(src / 7) + 1) + cols[dst % 7] + (Math.floor(dst / 7) + 1);
}

// ---- Generate test positions by playing random moves ----
function generatePositions(count: number): EngineState[] {
  const positions: EngineState[] = [];
  const rng = { seed: 42, next() { this.seed = (this.seed * 1664525 + 1013904223) & 0xffffffff; return (this.seed >>> 0) / 0xffffffff; } };

  for (let game = 0; game < 20 && positions.length < count; game++) {
    let state = newGame();
    const maxPly = 5 + Math.floor(rng.next() * 25);

    for (let ply = 0; ply < maxPly; ply++) {
      const moves = getLegalMoves(state);
      if (moves.length === 0) break;

      // Sample positions at various ply depths
      if (ply > 0 && (ply === 3 || ply === 8 || ply === 15 || ply === 22)) {
        positions.push(copyState(state));
        if (positions.length >= count) break;
      }

      const move = moves[Math.floor(rng.next() * moves.length)];
      state = copyState(state);
      applyMove(state, move);
    }
  }

  return positions;
}

// ---- Main test ----
async function run() {
  const results: any = {
    userAgent: navigator.userAgent,
    gpu: null,
    mctsComparisons: [],
    summary: null,
    error: null,
  };

  try {
    // Check WebGL2
    p('=== WebGL2 Check ===', 'info');
    const testCanvas = document.createElement('canvas');
    const testGl = testCanvas.getContext('webgl2');
    if (!testGl) throw new Error('WebGL2 not available');
    if (!testGl.getExtension('EXT_color_buffer_float')) throw new Error('No EXT_color_buffer_float');
    const debugExt = testGl.getExtension('WEBGL_debug_renderer_info');
    const renderer = debugExt ? testGl.getParameter(debugExt.UNMASKED_RENDERER_WEBGL) : testGl.getParameter(testGl.RENDERER);
    p('Renderer: ' + renderer, 'info');
    results.gpu = { renderer };
    const loseCtx = testGl.getExtension('WEBGL_lose_context');
    if (loseCtx) loseCtx.loseContext();

    // Download ONNX model
    p('\n=== Downloading Model ===', 'info');
    const infoResp = await fetch('/api/models/onnx/by-name/iter_300.pt');
    if (!infoResp.ok) throw new Error('Model info failed: ' + infoResp.status);
    const modelInfo = await infoResp.json();
    const filename = modelInfo.filename || modelInfo.url?.split('/').pop();
    p('Downloading ' + filename + '...');
    const modelResp = await fetch('/api/models/onnx/' + filename);
    if (!modelResp.ok) throw new Error('Model download failed: ' + modelResp.status);
    const modelBuffer = await modelResp.arrayBuffer();
    p('Downloaded ' + (modelBuffer.byteLength / 1024 / 1024).toFixed(1) + ' MB');

    // Create models
    p('\n=== Creating Models ===', 'info');
    const cpuModel = createModelFromOnnx(modelBuffer);
    p('CPU model: ' + cpuModel.config.numFilters + 'f, ' + cpuModel.config.numBlocks + 'b');

    const gpuCanvas = document.createElement('canvas');
    const gpuModel = createGPUModelFromOnnx(modelBuffer, gpuCanvas);
    p('GPU model created');

    const cpuEval = new PureTSEvaluator(cpuModel);
    const gpuEval = new GPUEvaluator(gpuModel);

    // Generate test positions
    p('\n=== Generating Test Positions ===', 'info');
    const positions = generatePositions(8);
    p('Generated ' + positions.length + ' positions');

    // Run MCTS on each position with both evaluators
    p('\n=== Running MCTS Comparisons ===', 'info');
    p('Config: 256 sims, temperature=0 (deterministic)', 'dim');
    p('');

    const mctsConfig: Partial<MCTSConfig> = {
      numSimulations: 256,
    };

    let allMatch = true;

    for (let i = 0; i < positions.length; i++) {
      const state = positions[i];
      p('[Position ' + i + '] ply=' + state.ply + ' player=' + state.currentPlayer, 'info');

      // CPU MCTS
      const cpuStart = performance.now();
      const cpuResult = await search(state, cpuEval, { ...DEFAULT_CONFIG, ...mctsConfig }, { aborted: false });
      const cpuMs = performance.now() - cpuStart;

      // GPU MCTS
      const gpuStart = performance.now();
      const gpuResult = await search(state, gpuEval, { ...DEFAULT_CONFIG, ...mctsConfig }, { aborted: false });
      const gpuMs = performance.now() - gpuStart;

      // Compare best moves
      const cpuBest = cpuResult.bestMove;
      const gpuBest = gpuResult.bestMove;
      const moveMatch = cpuBest === gpuBest;

      // Compare visit distributions for top moves
      const cpuVisits: Array<{ move: number; visits: number }> = [];
      const gpuVisits: Array<{ move: number; visits: number }> = [];

      cpuResult.rootNode.children.forEach((child, action) => {
        if (child.visitCount > 0) cpuVisits.push({ move: action, visits: child.visitCount });
      });
      gpuResult.rootNode.children.forEach((child, action) => {
        if (child.visitCount > 0) gpuVisits.push({ move: action, visits: child.visitCount });
      });

      cpuVisits.sort((a, b) => b.visits - a.visits);
      gpuVisits.sort((a, b) => b.visits - a.visits);

      // Compare value
      const cpuValue = cpuResult.value;
      const gpuValue = gpuResult.value;
      const valueDiff = Math.abs(cpuValue - gpuValue);

      // Check if visit distributions match
      let visitsMatch = true;
      const maxVisits = Math.max(cpuVisits.length, gpuVisits.length);
      for (let v = 0; v < Math.min(5, maxVisits); v++) {
        const cv = cpuVisits[v], gv = gpuVisits[v];
        if (!cv || !gv || cv.move !== gv.move || Math.abs(cv.visits - gv.visits) > 1) {
          visitsMatch = false;
          break;
        }
      }

      const posMatch = moveMatch && visitsMatch;
      if (!posMatch) allMatch = false;

      p('  Best move:  CPU=' + fmtMove(cpuBest) + '  GPU=' + fmtMove(gpuBest) + '  ' +
        (moveMatch ? 'MATCH' : 'DIFFER!'), moveMatch ? 'pass' : 'fail');
      p('  Value:      CPU=' + cpuValue.toFixed(4) + '  GPU=' + gpuValue.toFixed(4) +
        '  diff=' + valueDiff.toExponential(2), 'dim');
      p('  Time:       CPU=' + cpuMs.toFixed(0) + 'ms  GPU=' + gpuMs.toFixed(0) + 'ms  (' +
        (cpuMs / gpuMs).toFixed(1) + 'x speedup)', 'dim');

      // Show top 5 visit counts
      p('  Visit counts (top 5):', 'dim');
      for (let v = 0; v < Math.min(5, maxVisits); v++) {
        const cv = cpuVisits[v], gv = gpuVisits[v];
        const cm = cv ? fmtMove(cv.move).padEnd(7) + ' ' + String(cv.visits).padStart(4) : '---';
        const gm = gv ? fmtMove(gv.move).padEnd(7) + ' ' + String(gv.visits).padStart(4) : '---';
        const vmatch = cv && gv && cv.move === gv.move && Math.abs(cv.visits - gv.visits) <= 1;
        p('    CPU: ' + cm + '   GPU: ' + gm + '  ' + (vmatch ? '==' : '!='), vmatch ? 'dim' : 'warn');
      }

      results.mctsComparisons.push({
        ply: state.ply,
        player: state.currentPlayer,
        cpuBestMove: cpuBest,
        gpuBestMove: gpuBest,
        moveMatch,
        cpuValue,
        gpuValue,
        valueDiff,
        cpuMs: Math.round(cpuMs),
        gpuMs: Math.round(gpuMs),
        cpuTopVisits: cpuVisits.slice(0, 5),
        gpuTopVisits: gpuVisits.slice(0, 5),
        visitsMatch,
      });

      // Clean up tree memory
      cpuResult.rootNode.children.clear();
      gpuResult.rootNode.children.clear();

      p('');
    }

    // Summary
    p('========================================', 'info');
    p('=== MCTS COMPARISON SUMMARY ===', 'info');
    p('========================================', 'info');
    const matchCount = results.mctsComparisons.filter((c: any) => c.moveMatch).length;
    const visitMatchCount = results.mctsComparisons.filter((c: any) => c.visitsMatch).length;
    p('Positions: ' + positions.length);
    p('Best move matches: ' + matchCount + '/' + positions.length, matchCount === positions.length ? 'pass' : 'fail');
    p('Visit distribution matches: ' + visitMatchCount + '/' + positions.length, visitMatchCount === positions.length ? 'pass' : 'warn');

    const avgCpuMs = results.mctsComparisons.reduce((s: number, c: any) => s + c.cpuMs, 0) / positions.length;
    const avgGpuMs = results.mctsComparisons.reduce((s: number, c: any) => s + c.gpuMs, 0) / positions.length;
    p('Avg search time: CPU=' + avgCpuMs.toFixed(0) + 'ms  GPU=' + avgGpuMs.toFixed(0) + 'ms  (' +
      (avgCpuMs / avgGpuMs).toFixed(1) + 'x speedup)', 'info');

    results.summary = { positions: positions.length, moveMatches: matchCount, visitMatches: visitMatchCount, avgCpuMs, avgGpuMs };
    resultEl.dataset.status = allMatch ? 'pass' : 'fail';

    const banner = document.createElement('div');
    banner.className = 'summary ' + (allMatch ? 'pass' : 'fail');
    banner.textContent = allMatch
      ? 'ALL ' + positions.length + ' MCTS SEARCHES MATCH (moves + visit counts)'
      : matchCount + '/' + positions.length + ' move matches, ' + visitMatchCount + '/' + positions.length + ' visit matches';
    logEl.appendChild(banner);

    gpuModel.dispose();

  } catch (e: any) {
    p('\nERROR: ' + e.message, 'fail');
    if (e.stack) p(e.stack, 'dim');
    results.error = e.message;
    resultEl.dataset.status = 'fail';
  }

  // Send results
  try {
    await fetch('/api/logs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: 'webgl-mcts-test',
        entries: [{
          timestamp: new Date().toISOString(),
          level: 'info',
          message: '[webgl-mcts-test] Results',
          data: {
          userAgent: results.userAgent,
          gpu: results.gpu,
          summary: results.summary,
          error: results.error,
          // Compact per-position: just move match + visit match
          positions: results.mctsComparisons.map((c: any) => ({
            ply: c.ply, moveMatch: c.moveMatch, visitsMatch: c.visitsMatch,
            cpuMs: c.cpuMs, gpuMs: c.gpuMs, valueDiff: c.valueDiff,
          })),
        },
        }],
      }),
    });
    p('Results sent to server', 'dim');
  } catch (e: any) {
    p('Failed to send: ' + e.message, 'warn');
  }
}

run();
