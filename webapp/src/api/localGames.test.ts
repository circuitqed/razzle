import { describe, it, expect, beforeEach, vi } from 'vitest';

const server = vi.hoisted(() => ({
  createGame: vi.fn(),
  makeTurn: vi.fn(),
  resignGame: vi.fn(),
}));

vi.mock('./engine', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./engine')>();
  return { ...actual, ...server };
});

import * as local from './localGames';
import { EngineAPIError } from './engine';
import { newGame, applyMove, isTerminal, getWinner } from '../engine/state';
import { getLegalMoves } from '../engine/moves';

const END_TURN = -1;

/** Deterministic PRNG so failures reproduce. */
function rng(seed: number) {
  return () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
}

/** Pick a full random turn (sub-moves until the player changes or game ends). */
function randomTurn(moves: number[], rand: () => number): number[] {
  const state = newGame();
  for (const m of moves) applyMove(state, m);
  const player = state.currentPlayer;
  const turn: number[] = [];
  while (state.currentPlayer === player && !isTerminal(state)) {
    const legal = getLegalMoves(state);
    const m = legal[Math.floor(rand() * legal.length)];
    applyMove(state, m);
    turn.push(m);
  }
  return turn;
}

async function playRandomGame(seed: number) {
  const rand = rng(seed);
  const { game_id } = await local.createGame({ player2_type: 'ai', ai_simulations: 64 });
  let gs = await local.getGameState(game_id);
  while (gs.status === 'playing') {
    gs = await local.makeTurn(game_id, randomTurn(gs.moves!, rand));
  }
  return { game_id, gs };
}

beforeEach(async () => {
  await local.flushSyncQueue(); // let earlier tests' background syncs settle
  localStorage.clear();
  server.createGame.mockReset().mockResolvedValue({ game_id: 'srv-1' });
  server.makeTurn.mockReset().mockResolvedValue({});
  server.resignGame.mockReset().mockResolvedValue({});
});

describe('local game backend', () => {
  it('creates a game at the start position with engine legal moves', async () => {
    const { game_id } = await local.createGame();
    expect(local.isLocalGameId(game_id)).toBe(true);
    const gs = await local.getGameState(game_id);
    expect(gs.status).toBe('playing');
    expect(gs.ply).toBe(0);
    expect(gs.current_player).toBe(0);
    expect(gs.legal_moves.sort()).toEqual(getLegalMoves(newGame()).sort());
    expect(gs.moves).toEqual([]);
  });

  it('persists across reloads of the module state (localStorage)', async () => {
    const { game_id } = await local.createGame();
    const first = (await local.getGameState(game_id)).legal_moves[0];
    await local.makeMove(game_id, first);
    const gs = await local.getGameState(game_id);
    expect(gs.moves).toEqual([first]);
    expect(JSON.parse(localStorage.getItem('knightball_local_games')!)[game_id].moves).toEqual([first]);
  });

  it('rejects illegal moves with 400 and finished games with 409', async () => {
    const { game_id } = await local.createGame();
    await expect(local.makeMove(game_id, 9999)).rejects.toMatchObject({ status: 400 });
    await local.resignGame(game_id, 0);
    const legal = getLegalMoves(newGame())[0];
    await expect(local.makeMove(game_id, legal)).rejects.toMatchObject({ status: 409 });
  });

  it('returns 404 for unknown games', async () => {
    await expect(local.getGameState('local-nope')).rejects.toBeInstanceOf(EngineAPIError);
  });

  it('rejects incomplete turns atomically', async () => {
    // Find an opening position where a pass is legal: play knight moves until one is.
    const { game_id } = await local.createGame();
    const rand = rng(7);
    for (let i = 0; i < 40; i++) {
      const gs = await local.getGameState(game_id);
      const state = newGame();
      for (const m of gs.moves!) applyMove(state, m);
      const pass = gs.legal_moves.find(m => m !== END_TURN && (state.balls[state.currentPlayer] >> BigInt(Math.floor(m / 56))) & 1n);
      if (pass !== undefined) {
        const before = gs.moves!.length;
        await expect(local.makeTurn(game_id, [pass])).rejects.toMatchObject({ code: 'INCOMPLETE_TURN' });
        expect((await local.getGameState(game_id)).moves!.length).toBe(before);
        await local.makeTurn(game_id, [pass, END_TURN]);
        return;
      }
      await local.makeTurn(game_id, randomTurn(gs.moves!, rand));
    }
    throw new Error('never found a legal pass');
  });

  it('plays random games to completion matching the engine result', async () => {
    for (const seed of [1, 2, 3]) {
      const { gs } = await playRandomGame(seed);
      const state = newGame();
      for (const m of gs.moves!) applyMove(state, m);
      expect(gs.status).toBe('finished');
      expect(gs.winner).toBe(getWinner(state));
      expect(gs.legal_moves).toEqual([]);
    }
  });

  it('undo pops one sub-move and refuses on an empty game', async () => {
    const { game_id } = await local.createGame();
    await expect(local.undoMove(game_id)).rejects.toMatchObject({ status: 400 });
    const m = (await local.getGameState(game_id)).legal_moves[0];
    await local.makeMove(game_id, m);
    const gs = await local.undoMove(game_id);
    expect(gs.moves).toEqual([]);
    expect(gs.current_player).toBe(0);
  });

  it('resign awards the game to the opponent', async () => {
    const { game_id } = await local.createGame();
    const gs = await local.resignGame(game_id, 1);
    expect(gs.status).toBe('finished');
    expect(gs.winner).toBe(0);
  });
});

describe('splitIntoTurns', () => {
  it('reassembles a game into turns that each change the player', async () => {
    const { gs } = await playRandomGame(11);
    const turns = local.splitIntoTurns(gs.moves!);
    expect(turns.flat()).toEqual(gs.moves);
    const state = newGame();
    for (const turn of turns) {
      const player = state.currentPlayer;
      for (const m of turn) applyMove(state, m);
      expect(state.currentPlayer !== player || isTerminal(state)).toBe(true);
    }
  });
});

describe('server sync', () => {
  it('replays a finished game to the server turn by turn, then clears the queue', async () => {
    const { gs } = await playRandomGame(5);
    await local.flushSyncQueue();
    expect(server.createGame).toHaveBeenCalledTimes(1);
    expect(server.createGame).toHaveBeenCalledWith(expect.objectContaining({ player2_type: 'ai', ai_simulations: 64 }));
    const sent = server.makeTurn.mock.calls.map(c => c[1]);
    expect(sent).toEqual(local.splitIntoTurns(gs.moves!));
    expect(server.makeTurn.mock.calls.every(c => c[0] === 'srv-1')).toBe(true);
    expect(JSON.parse(localStorage.getItem('knightball_local_sync_queue') ?? '[]')).toEqual([]);
  });

  it('replays resignations', async () => {
    const { game_id } = await local.createGame();
    await local.makeMove(game_id, (await local.getGameState(game_id)).legal_moves[0]);
    await local.resignGame(game_id, 0);
    await local.flushSyncQueue();
    expect(server.resignGame).toHaveBeenCalledWith('srv-1', 0);
  });

  it('keeps the game queued when the network is down', async () => {
    server.createGame.mockRejectedValue(new TypeError('Failed to fetch'));
    await playRandomGame(6);
    await local.flushSyncQueue();
    expect(JSON.parse(localStorage.getItem('knightball_local_sync_queue')!)).toHaveLength(1);
  });

  it('drops a game the server rejects so the queue cannot wedge', async () => {
    server.makeTurn.mockRejectedValue(new EngineAPIError(400, 'INVALID_MOVE', 'bad'));
    await playRandomGame(8);
    await local.flushSyncQueue();
    expect(JSON.parse(localStorage.getItem('knightball_local_sync_queue')!)).toEqual([]);
  });

  it('does not queue games with no moves', async () => {
    const { game_id } = await local.createGame();
    await local.resignGame(game_id, 0);
    expect(localStorage.getItem('knightball_local_sync_queue')).toBeNull();
  });
});
