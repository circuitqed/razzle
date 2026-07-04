/**
 * Comprehensive on-device self-test suite for the native iOS app.
 *
 * Runs inside the real WKWebView (Capacitor) and exercises the exact code
 * paths the shipping app uses. Test groups map to bug classes found during
 * development (see git history / docs/IOS_HANDOFF.md):
 *
 *  env       — WebGL2/OffscreenCanvas/WebGPU/IndexedDB capability survey;
 *              service worker must NOT be active in the native app
 *              (stale-SW-cache bug class: 4693084, 8cd874c)
 *  rules     — TS engine rule regression: pass/END_TURN flow (#1, #2,
 *              49c0bfe, 852e053), ineligibility/touchedMask (fb90bda, #3),
 *              bitboard 56-bit precision (5c54f29), forced pass, ply-cap
 *              termination — via directed scenarios + seeded random-game
 *              invariant sweeps
 *  inference — CPU (pure-TS) and GPU (WebGL2 GEMM) forward passes vs Python
 *              reference fixtures (catches the "ONNX Runtime WebGL plays
 *              randomly" class, efdd7ca) on THIS device's GPU
 *  mcts      — CPU-vs-GPU full MCTS agreement (best move + visit counts),
 *              deterministic at temperature 0 (value backprop / perspective
 *              classes: 1bf6125, 490622d, 04d9ced; batch deadlock: 2e28821)
 *  cache     — IndexedDB model cache: second load must skip the download
 *  game      — full games through the REAL ai.worker: MCTS vs random mover,
 *              AI as player 0 AND player 1 (perspective bugs), every AI move
 *              legality-checked (illegal-move class)
 *  soak      — repeated worker searches; worker must survive and stay fast
 *              (iOS memory-kill class that motivated dropping WASM)
 *  backend   — prod REST/WS integration: health, models, game CRUD, undo,
 *              resign, server-AI posture, worker-context CORS, anon identity
 *              (X-Anon-Id) persistence, WSS auth
 *
 * Query params:
 *   ?groups=env,rules,...   run a subset (default: all)
 *   ?model=local            use bundled /test-model.onnx for game/soak
 *   ?gamesims=128 ?mctssims=128 ?soakn=25   tuning knobs
 *
 * Output: DOM log, console.log('[native-test] ...') for os_log capture,
 * one 'RESULT: ...' line and one 'RESULT_JSON: {...}' line at the end,
 * and <div id="result" data-status="pass|fail">.
 */

import {
  newGame, copyState, applyMove, isTerminal, getWinner, getEmpty,
} from './engine/state';
import type { EngineState } from './engine/state';
import { getLegalMoves, getPassMoves, getKnightMoves, mustPass, decodeMove } from './engine/moves';
import { END_TURN_MOVE } from './engine/bitboard';
import { createModelFromOnnx } from './engine/inference';
import { createGPUModelFromOnnx } from './engine/webglForwardPass';
import { PureTSEvaluator, GPUEvaluator } from './engine/evaluator';
import { search, DEFAULT_CONFIG } from './engine/mcts';
import { clearModelCache } from './engine/modelCache';
import { API_BASE, isNativeApp, gameWebSocketUrl, installNativeIdentity } from './api/base';
import { TIERS } from './utils/autoMatch';

installNativeIdentity();

// ---------------------------------------------------------------- reporting

const logEl = document.getElementById('log')!;
const resultEl = document.getElementById('result')!;

function p(msg: string, cls?: string) {
  const d = document.createElement('div');
  d.textContent = msg;
  if (cls) d.className = cls;
  logEl.appendChild(d);
  console.log('[native-test] ' + msg);
  window.scrollTo(0, document.body.scrollHeight);
}

const params = new URLSearchParams(location.search);
const failures: string[] = [];
const groupResults: Record<string, { pass: number; fail: number; skipped?: boolean }> = {};
let currentGroup = '';

function check(name: string, ok: boolean, detail?: string) {
  p(`${ok ? 'PASS' : 'FAIL'}: ${name}${detail ? ' — ' + detail : ''}`, ok ? 'pass' : 'fail');
  const g = groupResults[currentGroup] ?? (groupResults[currentGroup] = { pass: 0, fail: 0 });
  if (ok) g.pass++;
  else {
    g.fail++;
    failures.push(`[${currentGroup}] ${name}`);
  }
}

// ------------------------------------------------------------ worker helpers

interface SerializedEngineState {
  pieces: [string, string];
  balls: [string, string];
  currentPlayer: number;
  touchedMask: string;
  hasPassed: boolean;
  lastKnightDst: number;
  ply: number;
}

function serializeState(s: EngineState): SerializedEngineState {
  return {
    pieces: [s.pieces[0].toString(), s.pieces[1].toString()],
    balls: [s.balls[0].toString(), s.balls[1].toString()],
    currentPlayer: s.currentPlayer,
    touchedMask: s.touchedMask.toString(),
    hasPassed: s.hasPassed,
    lastKnightDst: s.lastKnightDst,
    ply: s.ply,
  };
}

function createAIWorker(): Worker {
  return new Worker(new URL('./workers/ai.worker.ts', import.meta.url), { type: 'module' });
}

interface LoadResult { success: boolean; backend?: string; error?: string; downloaded: boolean }

function loadModel(worker: Worker, modelUrl: string, modelVersion: string): Promise<LoadResult> {
  return new Promise((resolve) => {
    let downloaded = false;
    const onMsg = (event: MessageEvent) => {
      const msg = event.data;
      if (msg.type === 'loading_progress' && msg.stage === 'downloading') downloaded = true;
      if (msg.type === 'loaded') {
        worker.removeEventListener('message', onMsg);
        resolve({ ...msg, downloaded });
      }
    };
    worker.addEventListener('message', onMsg);
    worker.postMessage({ type: 'load', modelUrl, modelVersion });
  });
}

interface SearchResult { success: boolean; bestMove: number; simsDone: number; value: number; error?: string }

function workerSearch(worker: Worker, state: EngineState, numSimulations: number, timeoutMs = 120000): Promise<SearchResult> {
  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      worker.removeEventListener('message', onMsg);
      resolve({ success: false, bestMove: -99, simsDone: 0, value: 0, error: 'search timed out' });
    }, timeoutMs);
    const onMsg = (event: MessageEvent) => {
      const msg = event.data;
      if (msg.type === 'search_result') {
        clearTimeout(timer);
        worker.removeEventListener('message', onMsg);
        resolve(msg);
      }
    };
    worker.addEventListener('message', onMsg);
    worker.postMessage({ type: 'search', state: serializeState(state), config: { numSimulations } });
  });
}

// ---------------------------------------------------------------- utilities

function makeRng(seed: number) {
  return {
    seed,
    next() { this.seed = (this.seed * 1664525 + 1013904223) & 0xffffffff; return (this.seed >>> 0) / 0xffffffff; },
  };
}

function popcount(b: bigint): number {
  let n = 0;
  while (b) { b &= b - 1n; n++; }
  return n;
}

async function getModelUrl(): Promise<{ url: string; version: string }> {
  if (params.get('model') === 'local') {
    return { url: '/test-model.onnx', version: 'local-test' };
  }
  const resp = await fetch(`${API_BASE}/models/onnx/latest`, { credentials: 'include' });
  if (!resp.ok) throw new Error('model info fetch failed: ' + resp.status);
  const info = await resp.json();
  return { url: `${API_BASE}${info.url}`, version: info.version };
}

/**
 * Model matching the reference fixtures (fixtures.model), for Python-reference
 * checks. Returns null when that model is no longer available anywhere —
 * callers should then skip reference checks (CPU-vs-GPU agreement still runs).
 */
/**
 * Fetch a bundled file, or null if absent. Capacitor's WKURLSchemeHandler
 * REJECTS the fetch promise for missing local files ("Load failed") rather
 * than returning a 404 — treat rejection as absence.
 */
async function fetchLocal(path: string): Promise<ArrayBuffer | null> {
  try {
    const resp = await fetch(path);
    if (resp.ok && (resp.headers.get('content-type') ?? '').indexOf('html') === -1) {
      return await resp.arrayBuffer();
    }
  } catch { /* missing bundled file */ }
  return null;
}

async function getFixtureModelBuffer(fixturesModel: string): Promise<ArrayBuffer | null> {
  // Bundled copy first (runner drops it in dist/), then on-demand API export.
  const local = await fetchLocal('/test-model-fixtures.onnx');
  if (local) return local;
  try {
    const infoResp = await fetch(`${API_BASE}/models/onnx/by-name/${encodeURIComponent(fixturesModel)}`, { credentials: 'include' });
    if (!infoResp.ok) return null;
    const info = await infoResp.json();
    const modelResp = await fetch(`${API_BASE}${info.url}`, { credentials: 'include' });
    if (!modelResp.ok) return null;
    return await modelResp.arrayBuffer();
  } catch {
    return null;
  }
}

/** Any usable model (weights don't matter — CPU and GPU must agree on the SAME weights). */
async function getAnyModelBuffer(): Promise<ArrayBuffer> {
  for (const path of ['/test-model-fixtures.onnx', '/test-model.onnx']) {
    const local = await fetchLocal(path);
    if (local) return local;
  }
  const model = await getModelUrl();
  const resp = await fetch(model.url, { credentials: 'include' });
  if (!resp.ok) throw new Error('model download failed: ' + resp.status);
  return resp.arrayBuffer();
}

/** Deterministic sample positions at assorted plies (same scheme as test-webgl-entry). */
function generatePositions(count: number): EngineState[] {
  const positions: EngineState[] = [];
  const rng = makeRng(42);
  for (let game = 0; game < 20 && positions.length < count; game++) {
    let state = newGame();
    const maxPly = 5 + Math.floor(rng.next() * 25);
    for (let ply = 0; ply < maxPly; ply++) {
      const moves = getLegalMoves(state);
      if (moves.length === 0) break;
      if (ply > 0 && (ply === 3 || ply === 8 || ply === 15 || ply === 22)) {
        positions.push(copyState(state));
        if (positions.length >= count) break;
      }
      state = copyState(state);
      applyMove(state, moves[Math.floor(rng.next() * moves.length)]);
    }
  }
  return positions;
}

// ------------------------------------------------------------------- groups

async function groupEnv() {
  p('=== env: capabilities ===', 'info');
  p('userAgent: ' + navigator.userAgent, 'dim');
  p('origin: ' + location.origin + '  isNativeApp: ' + isNativeApp + '  API_BASE: ' + API_BASE, 'dim');

  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2');
  check('WebGL2 (main thread)', !!gl);
  if (gl) check('EXT_color_buffer_float', !!gl.getExtension('EXT_color_buffer_float'));
  check('OffscreenCanvas', typeof OffscreenCanvas !== 'undefined');
  check('IndexedDB', typeof indexedDB !== 'undefined');
  check('localStorage', (() => { try { localStorage.setItem('__t', '1'); localStorage.removeItem('__t'); return true; } catch { return false; } })());

  if (isNativeApp && 'serviceWorker' in navigator) {
    // Both checks matter: getRegistrations() finds installed SWs; controller
    // catches this page already being served THROUGH one (stale-SW class —
    // the exact failure mode that plagued the PWA).
    const regs = await navigator.serviceWorker.getRegistrations().catch(() => []);
    check('no service worker registered in native app', regs.length === 0, `${regs.length} registration(s)`);
    check('page not controlled by a service worker', navigator.serviceWorker.controller === null);
  }

  // Informational WebGPU survey (no pass/fail): historically present-but-
  // broken in iOS Safari workers; gated off in WKWebView.
  const mainGpu = 'gpu' in navigator;
  let mainAdapter = false;
  if (mainGpu) {
    try { mainAdapter = (await (navigator as any).gpu.requestAdapter()) !== null; } catch { /* unavailable */ }
  }
  p(`WebGPU main thread: present=${mainGpu} adapter=${mainAdapter}`, 'info');
  const workerProbe = await new Promise<string>((resolve) => {
    const src = `
      (async () => {
        const res = { gpu: 'gpu' in navigator, adapter: false, err: null };
        if (res.gpu) {
          try { res.adapter = (await navigator.gpu.requestAdapter()) !== null; }
          catch (e) { res.err = String(e); }
        }
        postMessage(res);
      })();
    `;
    const w = new Worker(URL.createObjectURL(new Blob([src], { type: 'application/javascript' })));
    const timer = setTimeout(() => { w.terminate(); resolve('probe timed out (likely crashed)'); }, 10000);
    w.onmessage = (e) => {
      clearTimeout(timer); w.terminate();
      resolve(`present=${e.data.gpu} adapter=${e.data.adapter}${e.data.err ? ' err=' + e.data.err : ''}`);
    };
    w.onerror = (e) => { clearTimeout(timer); w.terminate(); resolve('worker error: ' + e.message); };
  });
  p('WebGPU in worker: ' + workerProbe, 'info');
}

async function groupRules() {
  p('', undefined);
  p('=== rules: engine regression ===', 'info');

  // -- directed: initial position sanity
  {
    const s = newGame();
    check('initial: 5 pieces + 1 ball each, ball on own piece',
      popcount(s.pieces[0]) === 5 && popcount(s.pieces[1]) === 5 &&
      popcount(s.balls[0]) === 1 && popcount(s.balls[1]) === 1 &&
      (s.balls[0] & s.pieces[0]) === s.balls[0] && (s.balls[1] & s.pieces[1]) === s.balls[1]);
    check('initial: all pieces ineligible → no pass moves', getPassMoves(s).length === 0);
    check('initial: knight moves exist, END_TURN not legal',
      getKnightMoves(s).length > 0 && !getLegalMoves(s).includes(END_TURN_MOVE));
    check('initial: ball-holding piece cannot move',
      getKnightMoves(s).every((m) => (s.balls[0] & (1n << BigInt(decodeMove(m)[0]))) === 0n));
  }

  // -- directed: knight move switches player and increments ply
  {
    const s = newGame();
    const m = getKnightMoves(s)[0];
    const before = copyState(s);
    applyMove(s, m);
    check('knight move: player switches, ply increments, lastKnightDst set',
      s.currentPlayer === 1 && s.ply === 1 && s.lastKnightDst === decodeMove(m)[1]);
    check('copyState: original not mutated by applyMove on copy', (() => {
      const c = copyState(before);
      applyMove(c, getKnightMoves(before)[0]);
      return before.currentPlayer === 0 && before.ply === 0;
    })());
  }

  // -- directed: pass flow (bug classes #1, #2 — pass rejected, end-turn)
  {
    // Random-walk until the side to move has a pass available.
    const rng = makeRng(7);
    let s = newGame();
    let found = false;
    for (let i = 0; i < 400 && !found; i++) {
      if (isTerminal(s)) { s = newGame(); continue; }
      if (!s.hasPassed && getPassMoves(s).length > 0) { found = true; break; }
      const moves = getLegalMoves(s);
      s = copyState(s);
      applyMove(s, moves[Math.floor(rng.next() * moves.length)]);
    }
    check('pass scenario reachable via random play', found);
    if (found) {
      const passer = s.currentPlayer;
      const pass = getPassMoves(s)[0];
      const [src, dst] = decodeMove(pass);
      check('pass listed in getLegalMoves', getLegalMoves(s).includes(pass));
      s = copyState(s);
      applyMove(s, pass);
      check('pass: same player keeps the turn, hasPassed set, ball moved',
        s.currentPlayer === passer && s.hasPassed &&
        (s.balls[passer] & (1n << BigInt(dst))) !== 0n);
      check('pass: both src and receiver become ineligible (touchedMask)',
        (s.touchedMask & (1n << BigInt(src))) !== 0n && (s.touchedMask & (1n << BigInt(dst))) !== 0n);
      check('pass: receiver not offered as a target again this turn',
        getPassMoves(s).every((m2) => decodeMove(m2)[1] !== src || true) &&
        getPassMoves(s).every((m2) => (s.touchedMask & (1n << BigInt(decodeMove(m2)[1]))) === 0n));
      const legals = getLegalMoves(s);
      check('after pass: END_TURN legal, knight moves not', legals.includes(END_TURN_MOVE) && legals.every((m2) => m2 === END_TURN_MOVE || getPassMoves(s).includes(m2)));
      const beforeEnd = s.ply;
      s = copyState(s);
      applyMove(s, END_TURN_MOVE);
      check('END_TURN: player switches, ply increments, hasPassed cleared',
        s.currentPlayer === 1 - passer && s.ply === beforeEnd + 1 && !s.hasPassed);
    }
  }

  // -- directed: forced pass (mustPass) returns only passes
  {
    const rng = makeRng(99);
    let s = newGame();
    let seen = false;
    for (let i = 0; i < 2000 && !seen; i++) {
      if (isTerminal(s)) { s = newGame(); continue; }
      if (!s.hasPassed && mustPass(s) && getPassMoves(s).length > 0) {
        seen = true;
        const legals = getLegalMoves(s);
        check('forced pass: legal moves are exactly the passes',
          legals.length === getPassMoves(s).length && legals.every((m) => getPassMoves(s).includes(m)));
        break;
      }
      const moves = getLegalMoves(s);
      s = copyState(s);
      applyMove(s, moves[Math.floor(rng.next() * moves.length)]);
    }
    if (!seen) p('forced-pass position not reached in sweep (informational)', 'warn');
  }

  // -- sweep: seeded random games, invariants every ply (bitboard precision class)
  {
    const rng = makeRng(1234);
    let games = 0, plies = 0, bad: string | null = null;
    const VALID = (1n << 56n) - 1n;
    outer: for (let g = 0; g < 10; g++) {
      let s = newGame();
      games++;
      for (let i = 0; i < 400; i++) {
        if (isTerminal(s)) {
          if (getWinner(s) === null) { bad = `terminal but no winner (game ${g})`; break outer; }
          break;
        }
        const moves = getLegalMoves(s);
        if (moves.length === 0) { bad = `no legal moves in non-terminal state (game ${g} ply ${s.ply})`; break outer; }
        const mv = moves[Math.floor(rng.next() * moves.length)];
        s = copyState(s);
        applyMove(s, mv);
        plies++;
        for (const pl of [0, 1]) {
          if (popcount(s.pieces[pl]) !== 5) { bad = `piece count ${popcount(s.pieces[pl])} for p${pl}`; break outer; }
          if (popcount(s.balls[pl]) !== 1) { bad = `ball count for p${pl}`; break outer; }
          if ((s.balls[pl] & s.pieces[pl]) !== s.balls[pl]) { bad = `ball off own piece p${pl}`; break outer; }
          if ((s.pieces[pl] | s.balls[pl]) !== ((s.pieces[pl] | s.balls[pl]) & VALID)) { bad = 'bits above square 55'; break outer; }
        }
        if ((s.pieces[0] & s.pieces[1]) !== 0n) { bad = 'piece overlap'; break outer; }
        if ((getEmpty(s) & (s.pieces[0] | s.pieces[1])) !== 0n) { bad = 'empty/occupied disagree'; break outer; }
      }
      if (!isTerminal(s) && s.ply <= 200) { bad = null; /* long game hit sweep cap — fine */ }
    }
    check('random-game invariant sweep (10 games)', bad === null, bad ?? `${games} games, ${plies} plies checked`);
  }

  // -- directed: ply-cap termination rule
  {
    const s = newGame();
    s.ply = 201;
    check('ply>200: terminal, non-mover wins', isTerminal(s) && getWinner(s) === 1 - s.currentPlayer);
  }
}

async function groupInference() {
  p('', undefined);
  p('=== inference: GPU-vs-CPU agreement + Python reference ===', 'info');

  const fixResp = await fetch('/inference-fixtures.json');
  if (!fixResp.ok) { check('fixtures available', false, 'fetch ' + fixResp.status); return; }
  const fixtures = await fixResp.json();
  check('fixtures structure', fixtures.positions?.length > 0,
    `${fixtures.positions.length} positions for ${fixtures.model}`);

  const tensors: Float32Array[] = fixtures.positions.map((pos: any) => {
    const tensor = new Float32Array(7 * 56);
    for (let c = 0; c < 7; c++)
      for (let r = 0; r < 8; r++)
        for (let col = 0; col < 7; col++)
          tensor[c * 56 + r * 7 + col] = pos.tensor[c][r][col];
    return tensor;
  });

  // Python-reference model may no longer exist server-side (models get
  // renamed/pruned between training runs). Reference checks are best-depth;
  // GPU-vs-CPU agreement below is the essential on-device check.
  const refBuffer = await getFixtureModelBuffer(fixtures.model);

  // -- essential: GPU and CPU must agree on the SAME weights (the class of
  //    bug where the WebGL path silently computes garbage → random play)
  let modelBuffer: ArrayBuffer;
  try {
    modelBuffer = refBuffer ?? await getAnyModelBuffer();
  } catch (e: any) {
    check('any model available for agreement check', false, e.message);
    return;
  }
  const cpuModel = createModelFromOnnx(modelBuffer);
  const gpuCanvas = document.createElement('canvas');
  const gpuModel = createGPUModelFromOnnx(modelBuffer, gpuCanvas);

  let xMaxPolicy = 0, xMaxValue = 0;
  for (const tensor of tensors) {
    const cpu = cpuModel.forward(tensor);
    const gpu = gpuModel.forward(tensor);
    for (let i = 0; i < 3137; i++) {
      xMaxPolicy = Math.max(xMaxPolicy, Math.abs(gpu.policy[i] - cpu.policy[i]));
    }
    xMaxValue = Math.max(xMaxValue, Math.abs(gpu.value - cpu.value));
  }
  check('GPU vs CPU policy agreement (20 positions)', xMaxPolicy < 0.01, `maxΔ=${xMaxPolicy.toExponential(2)}`);
  check('GPU vs CPU value agreement', xMaxValue < 0.005, `maxΔ=${xMaxValue.toExponential(2)}`);
  gpuModel.dispose();

  // -- depth: compare against Python reference outputs when the exact
  //    fixtures model is available
  if (!refBuffer) {
    p(`SKIP: Python-reference checks — fixtures model '${fixtures.model}' not available ` +
      '(regenerate fixtures against a current model to restore this depth)', 'warn');
    return;
  }
  const refCpu = createModelFromOnnx(refBuffer);
  const refGpuModel = createGPUModelFromOnnx(refBuffer, document.createElement('canvas'));
  let cpuMaxPolicy = 0, cpuMaxValue = 0, gpuMaxPolicy = 0, gpuMaxValue = 0;
  for (let t = 0; t < tensors.length; t++) {
    const pos = fixtures.positions[t];
    const cpu = refCpu.forward(tensors[t]);
    const gpu = refGpuModel.forward(tensors[t]);
    for (let i = 0; i < 3137; i++) {
      cpuMaxPolicy = Math.max(cpuMaxPolicy, Math.abs(cpu.policy[i] - pos.policy[i]));
      gpuMaxPolicy = Math.max(gpuMaxPolicy, Math.abs(gpu.policy[i] - pos.policy[i]));
    }
    cpuMaxValue = Math.max(cpuMaxValue, Math.abs(cpu.value - pos.value));
    gpuMaxValue = Math.max(gpuMaxValue, Math.abs(gpu.value - pos.value));
  }
  refGpuModel.dispose();
  // Same tolerances as the vitest reference test (float32 vs float64).
  check('CPU policy vs Python', cpuMaxPolicy < 0.05, `maxΔ=${cpuMaxPolicy.toExponential(2)}`);
  check('CPU value vs Python', cpuMaxValue < 0.01, `maxΔ=${cpuMaxValue.toExponential(2)}`);
  check('GPU policy vs Python', gpuMaxPolicy < 0.05, `maxΔ=${gpuMaxPolicy.toExponential(2)}`);
  check('GPU value vs Python', gpuMaxValue < 0.01, `maxΔ=${gpuMaxValue.toExponential(2)}`);
}

async function groupMcts() {
  p('', undefined);
  p('=== mcts: CPU-vs-GPU search agreement ===', 'info');

  const sims = parseInt(params.get('mctssims') ?? '128', 10);
  let modelBuffer: ArrayBuffer;
  try {
    modelBuffer = await getAnyModelBuffer();
  } catch (e: any) {
    check('model available', false, e.message);
    return;
  }
  const cpuEval = new PureTSEvaluator(createModelFromOnnx(modelBuffer));
  const gpuCanvas = document.createElement('canvas');
  const gpuModel = createGPUModelFromOnnx(modelBuffer, gpuCanvas);
  const gpuEval = new GPUEvaluator(gpuModel);

  // batchSize 1 forces the sequential search path for BOTH evaluators.
  // GPUEvaluator supports evaluateBatch, so at the default (auto) batch size
  // it takes the virtual-loss batched path while PureTSEvaluator goes
  // sequential — legitimately different trees, not an accuracy bug.
  const cfg = { ...DEFAULT_CONFIG, numSimulations: sims, batchSize: 1 };
  const positions = generatePositions(6);
  let moveMatches = 0, valueMaxDiff = 0, cpuMsTot = 0, gpuMsTot = 0;
  for (let i = 0; i < positions.length; i++) {
    const st = positions[i];
    const t0 = performance.now();
    const cpu = await search(st, cpuEval, cfg, { aborted: false });
    const t1 = performance.now();
    const gpu = await search(st, gpuEval, cfg, { aborted: false });
    const t2 = performance.now();
    cpuMsTot += t1 - t0; gpuMsTot += t2 - t1;
    if (cpu.bestMove === gpu.bestMove) moveMatches++;
    else p(`  pos ${i} (ply ${st.ply}): CPU best=${cpu.bestMove} GPU best=${gpu.bestMove}`, 'warn');
    valueMaxDiff = Math.max(valueMaxDiff, Math.abs(cpu.value - gpu.value));
    const legal = getLegalMoves(st);
    if (!legal.includes(cpu.bestMove) || !legal.includes(gpu.bestMove)) {
      check(`pos ${i}: best moves legal`, false, `cpu=${cpu.bestMove} gpu=${gpu.bestMove}`);
    }
    cpu.rootNode.children.clear();
    gpu.rootNode.children.clear();
  }
  gpuModel.dispose();

  check(`best-move agreement (${sims} sims, temp 0)`, moveMatches === positions.length, `${moveMatches}/${positions.length}`);
  check('root value agreement', valueMaxDiff < 0.02, `maxΔ=${valueMaxDiff.toExponential(2)}`);
  p(`timing: CPU ${(cpuMsTot / positions.length).toFixed(0)}ms/search, GPU ${(gpuMsTot / positions.length).toFixed(0)}ms/search`, 'info');
}

async function groupCache() {
  p('', undefined);
  p('=== cache: IndexedDB model cache ===', 'info');

  let model: { url: string; version: string };
  try {
    model = await getModelUrl();
  } catch (e: any) {
    check('model info', false, e.message);
    return;
  }

  await clearModelCache().catch(() => { /* first run */ });

  const w1 = createAIWorker();
  const t0 = performance.now();
  const first = await loadModel(w1, model.url, model.version);
  const firstMs = performance.now() - t0;
  w1.terminate();
  check('first load succeeds (cache cleared)', first.success, `backend=${first.backend} ${firstMs.toFixed(0)}ms downloaded=${first.downloaded}`);
  check('first load downloads', first.downloaded);

  const w2 = createAIWorker();
  const t1 = performance.now();
  const second = await loadModel(w2, model.url, model.version);
  const secondMs = performance.now() - t1;
  w2.terminate();
  check('second load succeeds', second.success, `${secondMs.toFixed(0)}ms downloaded=${second.downloaded}`);
  check('second load served from IndexedDB (no download)', !second.downloaded);
}

async function playFullGame(worker: Worker, aiPlayer: number, sims: number): Promise<{ winner: number | null; plies: number; aiMoves: number; illegal: number; simsPerSec: number }> {
  const rng = makeRng(555 + aiPlayer);
  let state = newGame();
  let plies = 0, aiMoves = 0, illegal = 0, searchMs = 0, simsDone = 0;
  while (!isTerminal(state) && plies < 400) {
    let move: number;
    if (state.currentPlayer === aiPlayer) {
      const t = performance.now();
      const result = await workerSearch(worker, state, sims);
      if (!result.success) throw new Error('worker search failed: ' + result.error);
      searchMs += performance.now() - t;
      simsDone += result.simsDone;
      aiMoves++;
      move = result.bestMove;
      if (!getLegalMoves(state).includes(move)) { illegal++; move = getLegalMoves(state)[0]; }
    } else {
      const moves = getLegalMoves(state);
      move = moves[Math.floor(rng.next() * moves.length)];
    }
    state = copyState(state);
    applyMove(state, move);
    plies++;
  }
  return { winner: getWinner(state), plies, aiMoves, illegal, simsPerSec: simsDone / (searchMs / 1000) };
}

async function groupGame() {
  p('', undefined);
  p('=== game: full games through the real ai.worker ===', 'info');

  const sims = parseInt(params.get('gamesims') ?? '128', 10);
  let model: { url: string; version: string };
  try {
    model = await getModelUrl();
  } catch (e: any) {
    check('model info', false, e.message);
    return;
  }
  p('model: ' + model.version, 'dim');

  const worker = createAIWorker();
  const loaded = await loadModel(worker, model.url, model.version);
  check('worker model load', loaded.success, loaded.success ? `backend=${loaded.backend}` : loaded.error);
  if (!loaded.success) { worker.terminate(); return; }
  if (/iPhone|iPad|iPod/i.test(navigator.userAgent)) {
    check('iOS backend is WebGL GPU', loaded.backend === 'gpu', `backend=${loaded.backend}`);
  }

  // AI as player 0 AND player 1 — catches value-perspective bugs where the
  // net/search optimizes for the wrong side (1bf6125, 490622d, 04d9ced).
  for (const aiPlayer of [0, 1]) {
    try {
      const g = await playFullGame(worker, aiPlayer, sims);
      p(`  as P${aiPlayer}: winner=${g.winner} plies=${g.plies} aiMoves=${g.aiMoves} ${g.simsPerSec.toFixed(1)} sims/sec`, 'dim');
      check(`AI (as player ${aiPlayer}) beats random mover`, g.winner === aiPlayer, `winner=${g.winner}`);
      check(`AI (as player ${aiPlayer}) plays only legal moves`, g.illegal === 0, `${g.illegal} illegal`);
    } catch (e: any) {
      check(`AI (as player ${aiPlayer}) full game`, false, e.message);
    }
  }
  worker.terminate();
}

async function groupSoak() {
  p('', undefined);
  p('=== soak: repeated searches (memory/stability) ===', 'info');

  const n = parseInt(params.get('soakn') ?? '25', 10);
  let model: { url: string; version: string };
  try {
    model = await getModelUrl();
  } catch (e: any) {
    check('model info', false, e.message);
    return;
  }

  const worker = createAIWorker();
  const loaded = await loadModel(worker, model.url, model.version);
  if (!loaded.success) { check('worker model load', false, loaded.error); worker.terminate(); return; }

  const state = generatePositions(1)[0] ?? newGame();
  const times: number[] = [];
  let failed: string | null = null;
  for (let i = 0; i < n; i++) {
    const t = performance.now();
    const r = await workerSearch(worker, state, 128, 60000);
    if (!r.success) { failed = `search ${i + 1}/${n}: ${r.error}`; break; }
    times.push(performance.now() - t);
    if ((i + 1) % 5 === 0) p(`  ${i + 1}/${n} searches, last ${times[times.length - 1].toFixed(0)}ms`, 'dim');
  }
  worker.terminate();

  check(`${n} consecutive searches all succeed (worker survives)`, failed === null, failed ?? undefined);
  if (failed === null && times.length >= 10) {
    const early = times.slice(0, 5).sort((a, b) => a - b)[2];
    const late = times.slice(-5).sort((a, b) => a - b)[2];
    check('no pathological slowdown (median last5 ≤ 3× median first5)', late <= early * 3,
      `first5 med ${early.toFixed(0)}ms, last5 med ${late.toFixed(0)}ms`);
  }
}

async function groupBackend() {
  p('', undefined);
  p('=== backend: prod integration (' + API_BASE + ') ===', 'info');

  try {
    const resp = await fetch(`${API_BASE}/health`, { credentials: 'include' });
    const health = await resp.json();
    check('GET /health', resp.ok, JSON.stringify(health));
  } catch (e: any) {
    check('GET /health', false, e.message + ' (CORS?)');
    return;
  }

  // Every difficulty-tier model must resolve — a pruned model file silently
  // kills client AI for those levels, and server AI is 403-disabled, so
  // affected levels have NO working AI at all (worse for Level 1: default).
  try {
    const tierModels = [...new Set(TIERS.map((t) => t.model))];
    const missing: string[] = [];
    for (const m of tierModels) {
      const r = await fetch(`${API_BASE}/models/onnx/by-name/${encodeURIComponent(m)}`, { credentials: 'include' });
      if (!r.ok) missing.push(`${m} (${r.status})`);
    }
    check(`all ${tierModels.length} difficulty-tier models resolve`, missing.length === 0,
      missing.length ? 'missing: ' + missing.join(', ') : undefined);
  } catch (e: any) {
    check('all difficulty-tier models resolve', false, e.message);
  }

  try {
    const resp = await fetch(`${API_BASE}/models/onnx/latest`, { credentials: 'include' });
    const info = await resp.json();
    const iterOk = /iter_(\d+)/.test(info.version);
    check('GET /models/onnx/latest', resp.ok && iterOk, info.version);

    // Worker-context CORS: the shipping app downloads the model INSIDE the
    // worker, where native HTTP plugins can't help — must pass browser CORS.
    const workerCors = await new Promise<string>((resolve) => {
      const src = `
        onmessage = async (e) => {
          try {
            const r = await fetch(e.data);
            postMessage(r.ok ? 'ok' : 'status ' + r.status);
          } catch (err) { postMessage('error: ' + err.message); }
        };
      `;
      const w = new Worker(URL.createObjectURL(new Blob([src], { type: 'application/javascript' })));
      const timer = setTimeout(() => { w.terminate(); resolve('timeout'); }, 30000);
      w.onmessage = (ev) => { clearTimeout(timer); w.terminate(); resolve(ev.data); };
      w.postMessage(`${API_BASE}${info.url}`);
    });
    check('model download passes CORS from a worker', workerCors === 'ok', workerCors);
  } catch (e: any) {
    check('GET /models/onnx/latest', false, e.message);
  }

  // Game CRUD + undo + resign (server-side pass/undo bug classes: #1, 8362dec)
  let gameId: string | null = null;
  try {
    const resp = await fetch(`${API_BASE}/games`, {
      method: 'POST', credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ player1_type: 'human', player2_type: 'ai', client_type: 'webapp-native-test' }),
    });
    gameId = (await resp.json()).game_id;
    check('POST /games', resp.ok && !!gameId, gameId ?? undefined);
  } catch (e: any) {
    check('POST /games', false, e.message);
  }

  if (gameId) {
    try {
      const lm = await (await fetch(`${API_BASE}/games/${gameId}/legal-moves`, { credentials: 'include' })).json();
      const moveResp = await fetch(`${API_BASE}/games/${gameId}/move`, {
        method: 'POST', credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ move: lm.moves[0].move }),
      });
      check('POST /games/{id}/move (first legal move accepted)', moveResp.ok);

      const undoResp = await fetch(`${API_BASE}/games/${gameId}/undo`, { method: 'POST', credentials: 'include' });
      const undone = undoResp.ok ? await undoResp.json() : null;
      check('POST /games/{id}/undo returns to ply 0', undoResp.ok && undone?.ply === 0, `ply=${undone?.ply}`);
    } catch (e: any) {
      check('move/undo roundtrip', false, e.message);
    }

    // Server AI is expected to be admin-gated in prod (403 by design).
    try {
      const aiResp = await fetch(`${API_BASE}/games/${gameId}/ai`, {
        method: 'POST', credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ simulations: 16 }),
      });
      check('POST /games/{id}/ai reachable (403-by-design or 200)', aiResp.ok || aiResp.status === 403,
        aiResp.status === 403 ? 'server AI disabled by design' : `status ${aiResp.status}`);
    } catch (e: any) {
      check('POST /games/{id}/ai reachable', false, e.message);
    }

    try {
      const ok = await new Promise<boolean>((resolve) => {
        const ws = new WebSocket(gameWebSocketUrl(gameId!));
        const timer = setTimeout(() => { ws.close(); resolve(false); }, 8000);
        ws.onmessage = () => { clearTimeout(timer); ws.close(); resolve(true); };
        ws.onerror = () => { clearTimeout(timer); resolve(false); };
      });
      check('WSS local-game socket receives state', ok);
    } catch (e: any) {
      check('WSS local-game socket receives state', false, e.message);
    }

    try {
      const resignResp = await fetch(`${API_BASE}/games/${gameId}/resign`, {
        method: 'POST', credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ player: 0 }),
      });
      const resigned = resignResp.ok ? await resignResp.json() : null;
      check('POST /games/{id}/resign ends game', resignResp.ok && resigned?.status !== 'playing', `status=${resigned?.status}`);
    } catch (e: any) {
      check('POST /games/{id}/resign ends game', false, e.message);
    }
  }

  // Anonymous identity persistence + WSS auth (native identity plumbing).
  let onlineGameId: string | null = null;
  try {
    const resp = await fetch(`${API_BASE}/games/online`, {
      method: 'POST', credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ host_color: 0, game_mode: 'realtime', is_public: false, client_type: 'webapp-native-test' }),
    });
    const data = await resp.json();
    onlineGameId = data.game_id;
    check('POST /games/online', resp.ok && !!onlineGameId, `join_code=${data.join_code}`);

    const mine = await (await fetch(`${API_BASE}/games/online/mine`, { credentials: 'include' })).json();
    const found = [...(mine.active ?? []), ...(mine.waiting ?? [])].some((g: any) => g.game_id === onlineGameId);
    check('anon identity persists across requests', found, found ? 'own game visible in /mine' : 'identity dropped');
  } catch (e: any) {
    check('anon identity persists across requests', false, e.message);
  }

  if (onlineGameId) {
    try {
      const result = await new Promise<string>((resolve) => {
        const ws = new WebSocket(gameWebSocketUrl(onlineGameId!));
        const timer = setTimeout(() => { ws.close(); resolve('timeout'); }, 8000);
        ws.onmessage = (event) => {
          const msg = JSON.parse(event.data);
          if (msg.type === 'error') { clearTimeout(timer); ws.close(); resolve(msg.data?.code ?? 'error'); }
          else { clearTimeout(timer); ws.close(); resolve('ok'); }
        };
        ws.onclose = (e) => { clearTimeout(timer); resolve('closed:' + e.code); };
        ws.onerror = () => {};
      });
      check('online-game WSS authenticates', result === 'ok', result);
    } catch (e: any) {
      check('online-game WSS authenticates', false, e.message);
    }
    try {
      await fetch(`${API_BASE}/games/online/${onlineGameId}/leave`, { method: 'POST', credentials: 'include' });
    } catch { /* best-effort cleanup */ }
  }
}

// --------------------------------------------------------------------- main

const ALL_GROUPS: Record<string, () => Promise<void>> = {
  env: groupEnv,
  rules: groupRules,
  inference: groupInference,
  mcts: groupMcts,
  cache: groupCache,
  game: groupGame,
  soak: groupSoak,
  backend: groupBackend,
};

async function run() {
  const requested = (params.get('groups') ?? Object.keys(ALL_GROUPS).join(','))
    .split(',').map((s) => s.trim()).filter((s) => s in ALL_GROUPS);
  // Back-compat with the v1 harness flag.
  const groups = params.get('skipbackend') === '1' ? requested.filter((g) => g !== 'backend') : requested;

  const t0 = performance.now();
  for (const g of groups) {
    currentGroup = g;
    groupResults[g] = { pass: 0, fail: 0 };
    try {
      await ALL_GROUPS[g]();
    } catch (e: any) {
      check(`${g}: uncaught exception`, false, e.message);
      if (e.stack) p(e.stack, 'dim');
    }
  }
  const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

  p('', undefined);
  const ok = failures.length === 0;
  const banner = document.createElement('div');
  banner.className = 'summary ' + (ok ? 'pass' : 'fail');
  banner.textContent = ok ? `ALL TESTS PASSED (${groups.join(', ')}; ${elapsed}s)` : `${failures.length} FAILURE(S): ` + failures.join('; ');
  logEl.appendChild(banner);
  console.log('[native-test] RESULT_JSON: ' + JSON.stringify({ ok, elapsed: +elapsed, groups: groupResults, failures }));
  p(ok ? 'RESULT: ALL TESTS PASSED' : 'RESULT: FAILURES: ' + failures.join('; '), ok ? 'pass' : 'fail');
  resultEl.dataset.status = ok ? 'pass' : 'fail';
}

run();
