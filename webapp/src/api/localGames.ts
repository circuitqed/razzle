/**
 * On-device game backend for AI and pass-and-play games.
 *
 * Mirrors the subset of the server game API that useGame needs (create, get,
 * move, turn, undo, resign) but runs entirely on the client TypeScript engine,
 * so games work offline and moves apply instantly. Used by the native app.
 *
 * A game is stored as its move list and replayed from the start position on
 * load — replay is microseconds, and it means persisted games can never hold
 * an inconsistent board.
 *
 * Finished games are queued and replayed to the server (POST /games + /turn)
 * when the network is available, so they still show up in game history and
 * training data. Sync is best-effort and never blocks play.
 */

import type { GameState } from '../types';
import type { EngineState } from '../engine/state';
import { newGame, applyMove, isTerminal, getWinner, copyState } from '../engine/state';
import { getLegalMoves } from '../engine/moves';
import { EngineAPIError } from './engine';
import * as serverApi from './engine';

export const LOCAL_GAME_PREFIX = 'local-';
const STORE_KEY = 'knightball_local_games';
const SYNC_QUEUE_KEY = 'knightball_local_sync_queue';
const MAX_STORED_GAMES = 5;
const MAX_SYNC_QUEUE = 50;

interface StoredGame {
  id: string;
  moves: number[];
  resignedBy: 0 | 1 | null;
  player2Type: 'human' | 'ai';
  aiSimulations?: number;
  /** AI-game metadata for account history (optional: absent in older stored games). */
  humanColor?: 0 | 1;
  aiLevel?: number;
  aiModel?: string;
  updatedAt: number;
}

export function isLocalGameId(gameId: string): boolean {
  return gameId.startsWith(LOCAL_GAME_PREFIX);
}

// ---------------------------------------------------------------------------
// Persistence

function loadStore(): Record<string, StoredGame> {
  try {
    const json = localStorage.getItem(STORE_KEY);
    if (json) return JSON.parse(json);
  } catch { /* corrupt or unavailable storage */ }
  return {};
}

function saveStore(store: Record<string, StoredGame>): void {
  // Keep only the most recently touched games; finished ones are synced and
  // the app only ever resumes the current game.
  const kept = Object.values(store)
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .slice(0, MAX_STORED_GAMES);
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(Object.fromEntries(kept.map(g => [g.id, g]))));
  } catch { /* storage full/unavailable — game continues in memory only */ }
}

function getStored(gameId: string): StoredGame {
  const game = loadStore()[gameId];
  if (!game) throw new EngineAPIError(404, 'NOT_FOUND', 'Game not found');
  return game;
}

function putStored(game: StoredGame): void {
  const store = loadStore();
  store[game.id] = { ...game, updatedAt: Date.now() };
  saveStore(store);
}

// ---------------------------------------------------------------------------
// State

function replay(moves: number[]): EngineState {
  const state = newGame();
  for (const m of moves) applyMove(state, m);
  return state;
}

function isOver(game: StoredGame, state: EngineState): boolean {
  return game.resignedBy !== null || isTerminal(state);
}

function toGameState(game: StoredGame, state: EngineState): GameState {
  const over = isOver(game, state);
  const winner = game.resignedBy !== null ? (1 - game.resignedBy) : getWinner(state);
  return {
    game_id: game.id,
    board: {
      p1_pieces: state.pieces[0].toString(),
      p1_ball: state.balls[0].toString(),
      p2_pieces: state.pieces[1].toString(),
      p2_ball: state.balls[1].toString(),
    },
    current_player: state.currentPlayer as 0 | 1,
    legal_moves: over ? [] : getLegalMoves(state),
    status: over ? 'finished' : 'playing',
    winner: winner as 0 | 1 | null,
    ply: state.ply,
    touched_mask: state.touchedMask.toString(),
    has_passed: state.hasPassed,
    last_knight_dst: state.lastKnightDst,
    time_control: null,
    increment: 0,
    time_remaining: null,
    game_mode: 'realtime',
    moves: [...game.moves],
  };
}

function finish(game: StoredGame, state: EngineState): GameState {
  putStored(game);
  if (isOver(game, state)) enqueueSync(game);
  return toGameState(game, state);
}

// ---------------------------------------------------------------------------
// API (same shapes as ./engine)

export async function createGame(options?: {
  player2_type?: 'human' | 'ai';
  ai_simulations?: number;
  human_color?: 0 | 1;
  ai_level?: number;
  ai_model?: string;
}): Promise<{ game_id: string }> {
  const id = `${LOCAL_GAME_PREFIX}${crypto.randomUUID()}`;
  putStored({
    id,
    moves: [],
    resignedBy: null,
    player2Type: options?.player2_type ?? 'ai',
    aiSimulations: options?.ai_simulations,
    humanColor: options?.human_color,
    aiLevel: options?.ai_level,
    aiModel: options?.ai_model,
    updatedAt: Date.now(),
  });
  return { game_id: id };
}

export async function getGameState(gameId: string): Promise<GameState> {
  const game = getStored(gameId);
  return toGameState(game, replay(game.moves));
}

export async function makeMove(gameId: string, move: number): Promise<GameState> {
  const game = getStored(gameId);
  const state = replay(game.moves);
  if (isOver(game, state)) throw new EngineAPIError(409, 'GAME_FINISHED', 'Game already finished');
  if (!getLegalMoves(state).includes(move)) {
    throw new EngineAPIError(400, 'INVALID_MOVE', `Invalid move: ${move}`);
  }
  applyMove(state, move);
  game.moves.push(move);
  return finish(game, state);
}

/** Apply a complete turn atomically (same validation as the server's process_turn). */
export async function makeTurn(gameId: string, moves: number[]): Promise<GameState> {
  if (moves.length === 0) throw new EngineAPIError(400, 'EMPTY_TURN', 'No moves provided');
  const game = getStored(gameId);
  const state = replay(game.moves);
  if (isOver(game, state)) throw new EngineAPIError(409, 'GAME_FINISHED', 'Game already finished');

  const work = copyState(state);
  const startPlayer = work.currentPlayer;
  const applied: number[] = [];
  for (const move of moves) {
    if (!getLegalMoves(work).includes(move)) {
      throw new EngineAPIError(400, 'INVALID_MOVE', `Invalid move: ${move}`);
    }
    applyMove(work, move);
    applied.push(move);
    if (isTerminal(work)) break;
  }
  if (work.currentPlayer === startPlayer && !isTerminal(work)) {
    throw new EngineAPIError(400, 'INCOMPLETE_TURN', 'Incomplete turn: player did not change');
  }
  game.moves.push(...applied);
  return finish(game, work);
}

/** Undo the last sub-move (matches server granularity). */
export async function undoMove(gameId: string): Promise<GameState> {
  const game = getStored(gameId);
  if (game.moves.length === 0) throw new EngineAPIError(400, 'NOTHING_TO_UNDO', 'Nothing to undo');
  if (game.resignedBy !== null) throw new EngineAPIError(409, 'GAME_FINISHED', 'Game already finished');
  game.moves.pop();
  const state = replay(game.moves);
  putStored(game);
  return toGameState(game, state);
}

export async function resignGame(gameId: string, player?: number): Promise<GameState> {
  const game = getStored(gameId);
  const state = replay(game.moves);
  if (isOver(game, state)) throw new EngineAPIError(409, 'GAME_FINISHED', 'Game already finished');
  game.resignedBy = (player === 1 ? 1 : 0);
  return finish(game, state);
}

// ---------------------------------------------------------------------------
// Server sync of finished games

interface SyncEntry {
  id: string;
  moves: number[];
  resignedBy: 0 | 1 | null;
  player2Type: 'human' | 'ai';
  aiSimulations?: number;
  humanColor?: 0 | 1;
  aiLevel?: number;
  aiModel?: string;
}

function loadQueue(): SyncEntry[] {
  try {
    const json = localStorage.getItem(SYNC_QUEUE_KEY);
    if (json) return JSON.parse(json);
  } catch { /* ignore */ }
  return [];
}

function saveQueue(queue: SyncEntry[]): void {
  try {
    localStorage.setItem(SYNC_QUEUE_KEY, JSON.stringify(queue.slice(-MAX_SYNC_QUEUE)));
  } catch { /* ignore */ }
}

function enqueueSync(game: StoredGame): void {
  // Games nobody moved in aren't worth recording.
  if (game.moves.length === 0) return;
  const queue = loadQueue();
  if (queue.some(e => e.id === game.id)) return;
  queue.push({
    id: game.id,
    moves: [...game.moves],
    resignedBy: game.resignedBy,
    player2Type: game.player2Type,
    aiSimulations: game.aiSimulations,
    humanColor: game.humanColor,
    aiLevel: game.aiLevel,
    aiModel: game.aiModel,
  });
  saveQueue(queue);
  void flushSyncQueue();
}

/** Split a flat move list into turns: a turn ends when the player changes or the game ends. */
export function splitIntoTurns(moves: number[]): number[][] {
  const turns: number[][] = [];
  const state = newGame();
  let current: number[] = [];
  for (const m of moves) {
    const player = state.currentPlayer;
    applyMove(state, m);
    current.push(m);
    if (state.currentPlayer !== player || isTerminal(state)) {
      turns.push(current);
      current = [];
    }
  }
  if (current.length) turns.push(current);
  return turns;
}

let inflight: Promise<void> | null = null;

/**
 * Replay queued finished games to the server. Network errors leave the entry
 * queued for next time; a server rejection (4xx) drops it so a bad entry
 * can't wedge the queue. Concurrent callers share the in-flight flush.
 */
export function flushSyncQueue(): Promise<void> {
  if (!inflight) {
    inflight = doFlush().finally(() => { inflight = null; });
  }
  return inflight;
}

async function doFlush(): Promise<void> {
  if (typeof navigator !== 'undefined' && navigator.onLine === false) return;
  let queue = loadQueue();
  while (queue.length > 0) {
    const entry = queue[0];
    try {
      const { game_id } = await serverApi.createGame({
        player1_type: 'human',
        player2_type: entry.player2Type,
        ai_simulations: entry.aiSimulations,
        // Sent with the user's credentials, so a signed-in player's on-device
        // games land in their account history with the right seat and level.
        human_color: entry.humanColor,
        ai_level: entry.aiLevel,
        ai_model: entry.aiModel,
      });
      for (const turn of splitIntoTurns(entry.moves)) {
        await serverApi.makeTurn(game_id, turn);
      }
      if (entry.resignedBy !== null) {
        await serverApi.resignGame(game_id, entry.resignedBy);
      }
    } catch (err) {
      const status = err instanceof EngineAPIError ? err.status : 0;
      // offline / server down / rate-limited (POST /games is limited per IP) — retry later
      if (status < 400 || status >= 500 || status === 429) return;
    }
    queue = loadQueue().filter(e => e.id !== entry.id);
    saveQueue(queue);
  }
}
