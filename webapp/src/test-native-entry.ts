/**
 * Native app self-test entry point.
 *
 * Exercises the REAL app AI path — ai.worker.ts in a Web Worker (WebGL2 GEMM
 * on iOS) — by playing a full game of MCTS-AI vs a random mover. A working
 * net should beat a random mover essentially always; a broken/garbage
 * inference path plays ~randomly and loses about half the time.
 *
 * Also checks backend connectivity (REST, server AI, WebSocket, anonymous
 * cookie persistence) against API_BASE, which resolves to the production
 * host in the native app.
 *
 * Every line is mirrored to console.log with a [native-test] prefix so it
 * shows up in the native (Xcode / os_log) console when run inside Capacitor.
 */

import { newGame, copyState, applyMove, isTerminal, getWinner } from './engine/state';
import type { EngineState } from './engine/state';
import { getLegalMoves } from './engine/moves';
import { API_BASE, isNativeApp, gameWebSocketUrl } from './api/base';

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

function check(name: string, ok: boolean, detail?: string) {
  p(`${ok ? 'PASS' : 'FAIL'}: ${name}${detail ? ' — ' + detail : ''}`, ok ? 'pass' : 'fail');
  if (!ok) failures.push(name);
}

// ---- Worker plumbing (same protocol as useAIWorker) ----

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

function loadModel(worker: Worker, modelUrl: string, modelVersion: string): Promise<{ success: boolean; backend?: string; error?: string }> {
  return new Promise((resolve) => {
    const onMsg = (event: MessageEvent) => {
      const msg = event.data;
      if (msg.type === 'loaded') {
        worker.removeEventListener('message', onMsg);
        resolve(msg);
      }
    };
    worker.addEventListener('message', onMsg);
    worker.postMessage({ type: 'load', modelUrl, modelVersion });
  });
}

function workerSearch(worker: Worker, state: EngineState, numSimulations: number): Promise<{ success: boolean; bestMove: number; simsDone: number; value: number; error?: string }> {
  return new Promise((resolve) => {
    const onMsg = (event: MessageEvent) => {
      const msg = event.data;
      if (msg.type === 'search_result') {
        worker.removeEventListener('message', onMsg);
        resolve(msg);
      }
    };
    worker.addEventListener('message', onMsg);
    worker.postMessage({ type: 'search', state: serializeState(state), config: { numSimulations } });
  });
}

// ---- Random opponent ----

const rng = { seed: 20260704, next() { this.seed = (this.seed * 1664525 + 1013904223) & 0xffffffff; return (this.seed >>> 0) / 0xffffffff; } };

function randomMove(state: EngineState): number {
  const moves = getLegalMoves(state);
  return moves[Math.floor(rng.next() * moves.length)];
}

// ---- Tests ----

async function testEnvironment() {
  p('=== Environment ===', 'info');
  p('userAgent: ' + navigator.userAgent, 'dim');
  p('origin: ' + location.origin, 'dim');
  p('isNativeApp: ' + isNativeApp + '  API_BASE: ' + API_BASE, 'dim');
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2');
  check('WebGL2 (main thread)', !!gl);
  if (gl) check('EXT_color_buffer_float', !!gl.getExtension('EXT_color_buffer_float'));
  check('OffscreenCanvas', typeof OffscreenCanvas !== 'undefined');
}

async function getModelUrl(): Promise<{ url: string; version: string }> {
  if (params.get('model') === 'local') {
    return { url: '/test-model.onnx', version: 'local-test' };
  }
  const resp = await fetch(`${API_BASE}/models/onnx/latest`);
  if (!resp.ok) throw new Error('model info fetch failed: ' + resp.status);
  const info = await resp.json();
  return { url: `${API_BASE}${info.url}`, version: info.version };
}

async function testWorkerAIGame(): Promise<void> {
  p('', undefined);
  p('=== Worker AI: full game vs random mover ===', 'info');

  const model = await getModelUrl();
  p('model: ' + model.version + ' @ ' + model.url, 'dim');

  const worker = createAIWorker();
  const t0 = performance.now();
  const loaded = await loadModel(worker, model.url, model.version);
  check('worker model load', loaded.success, loaded.success ? `backend=${loaded.backend} in ${(performance.now() - t0).toFixed(0)}ms` : loaded.error);
  if (!loaded.success) { worker.terminate(); return; }

  // On iOS the whole point is the WebGL2 path; pure-ts fallback means WebGL broke.
  const expectGpu = /iPhone|iPad|iPod/i.test(navigator.userAgent);
  if (expectGpu) {
    check('iOS backend is WebGL GPU', loaded.backend === 'gpu', `backend=${loaded.backend}`);
  }

  const SIMS = 128;
  const MAX_PLIES = 400;
  const aiPlayer = 0;
  let state = newGame();
  let plies = 0;
  let aiMoves = 0;
  let totalSearchMs = 0;
  let totalSims = 0;

  while (!isTerminal(state) && plies < MAX_PLIES) {
    let move: number;
    if (state.currentPlayer === aiPlayer) {
      const t = performance.now();
      const result = await workerSearch(worker, state, SIMS);
      if (!result.success) {
        check('worker search', false, result.error);
        worker.terminate();
        return;
      }
      totalSearchMs += performance.now() - t;
      totalSims += result.simsDone;
      aiMoves++;
      move = result.bestMove;
    } else {
      move = randomMove(state);
    }
    state = copyState(state);
    applyMove(state, move);
    plies++;
    if (plies % 20 === 0) p(`  ply ${plies}...`, 'dim');
  }

  const winner = getWinner(state);
  const simsPerSec = totalSims / (totalSearchMs / 1000);
  p(`game over: winner=${winner === null ? 'none (ply cap)' : winner} plies=${plies} aiMoves=${aiMoves}`, 'info');
  p(`search perf: ${simsPerSec.toFixed(1)} sims/sec avg (${(totalSearchMs / aiMoves).toFixed(0)}ms/move @ ${SIMS} sims)`, 'info');
  check('AI (player 0) beats random mover', winner === aiPlayer, `winner=${winner}`);

  worker.terminate();
}

async function testBackend(): Promise<void> {
  p('', undefined);
  p('=== Backend connectivity (' + API_BASE + ') ===', 'info');

  // Health
  try {
    const resp = await fetch(`${API_BASE}/health`, { credentials: 'include' });
    const health = await resp.json();
    check('GET /api/health', resp.ok, JSON.stringify(health));
  } catch (e: any) {
    check('GET /api/health', false, e.message + ' (CORS not deployed yet?)');
    return; // no point continuing
  }

  // Model info (what the real app fetches before the worker downloads it)
  try {
    const resp = await fetch(`${API_BASE}/models/onnx/latest`, { credentials: 'include' });
    const info = await resp.json();
    check('GET /api/models/onnx/latest', resp.ok, info.version);
  } catch (e: any) {
    check('GET /api/models/onnx/latest', false, e.message);
  }

  // Create a local-vs-AI game + server AI move (the server-AI fallback path)
  let gameId: string | null = null;
  try {
    const resp = await fetch(`${API_BASE}/games`, {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ player1_type: 'human', player2_type: 'ai', client_type: 'webapp-native-test' }),
    });
    const data = await resp.json();
    gameId = data.game_id;
    check('POST /api/games (create)', resp.ok && !!gameId, gameId ?? undefined);
  } catch (e: any) {
    check('POST /api/games (create)', false, e.message);
  }

  if (gameId) {
    // Make one human move so it's the AI's turn, then request a server AI move
    try {
      const stateResp = await fetch(`${API_BASE}/games/${gameId}/legal-moves`, { credentials: 'include' });
      const { moves } = await stateResp.json();
      const moveResp = await fetch(`${API_BASE}/games/${gameId}/move`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ move: moves[0].move }),
      });
      check('POST /api/games/{id}/move', moveResp.ok);
    } catch (e: any) {
      check('POST /api/games/{id}/move', false, e.message);
    }

    try {
      const t = performance.now();
      const aiResp = await fetch(`${API_BASE}/games/${gameId}/ai`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ simulations: 16 }),
      });
      check('POST /api/games/{id}/ai (server-AI fallback)', aiResp.ok, `${(performance.now() - t).toFixed(0)}ms`);
    } catch (e: any) {
      check('POST /api/games/{id}/ai (server-AI fallback)', false, e.message);
    }

    // WebSocket to the local game (auth-optional for non-online games)
    try {
      const ok = await new Promise<boolean>((resolve) => {
        const ws = new WebSocket(gameWebSocketUrl(gameId!));
        const timer = setTimeout(() => { ws.close(); resolve(false); }, 8000);
        ws.onmessage = () => { clearTimeout(timer); ws.close(); resolve(true); };
        ws.onerror = () => { clearTimeout(timer); resolve(false); };
      });
      check('WSS game socket connects + receives state', ok);
    } catch (e: any) {
      check('WSS game socket connects + receives state', false, e.message);
    }
  }

  // Anonymous cookie persistence: create an online game, then verify
  // /games/online/mine (a second request) sees it — only works if the
  // anon session cookie survived between requests.
  let onlineGameId: string | null = null;
  try {
    const resp = await fetch(`${API_BASE}/games/online`, {
      method: 'POST',
      credentials: 'include',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ host_color: 0, game_mode: 'realtime', is_public: false, client_type: 'webapp-native-test' }),
    });
    const data = await resp.json();
    onlineGameId = data.game_id;
    check('POST /api/games/online (create)', resp.ok && !!onlineGameId, `join_code=${data.join_code}`);

    const mineResp = await fetch(`${API_BASE}/games/online/mine`, { credentials: 'include' });
    const mine = await mineResp.json();
    const found = [...(mine.active ?? []), ...(mine.waiting ?? [])].some((g: any) => g.game_id === onlineGameId);
    check('anon cookie persists across requests', found, found ? 'own game visible in /mine' : 'game NOT in /mine — cookie dropped');
  } catch (e: any) {
    check('anon cookie persists across requests', false, e.message);
  }

  // Cookie auth over the WS handshake (what online multiplayer needs)
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
      check('online-game WSS auth via cookie', result === 'ok', result);
    } catch (e: any) {
      check('online-game WSS auth via cookie', false, e.message);
    }

    // Clean up the waiting game
    try {
      await fetch(`${API_BASE}/games/online/${onlineGameId}/leave`, { method: 'POST', credentials: 'include' });
    } catch { /* best-effort cleanup */ }
  }
}

async function run() {
  try {
    await testEnvironment();
    await testWorkerAIGame();
    if (params.get('skipbackend') !== '1') {
      await testBackend();
    }
  } catch (e: any) {
    p('UNCAUGHT: ' + e.message, 'fail');
    if (e.stack) p(e.stack, 'dim');
    failures.push('uncaught: ' + e.message);
  }

  p('', undefined);
  const ok = failures.length === 0;
  const banner = document.createElement('div');
  banner.className = 'summary ' + (ok ? 'pass' : 'fail');
  banner.textContent = ok ? 'ALL TESTS PASSED' : `${failures.length} FAILURE(S): ` + failures.join('; ');
  logEl.appendChild(banner);
  p(ok ? 'RESULT: ALL TESTS PASSED' : 'RESULT: FAILURES: ' + failures.join('; '), ok ? 'pass' : 'fail');
  resultEl.dataset.status = ok ? 'pass' : 'fail';
}

run();
