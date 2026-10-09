import { describe, it, expect } from 'vitest';
import { DEMO_GAME } from './DemoBoard';
import { newGame, applyMove, isTerminal, getWinner } from '../engine/state';
import { getLegalMoves } from '../engine/moves';

describe('DEMO_GAME', () => {
  it('is a legal game that blue wins on the last move', () => {
    const s = newGame();
    DEMO_GAME.forEach((m, i) => {
      expect(getLegalMoves(s), `move ${i}`).toContain(m);
      expect(isTerminal(s)).toBe(false);
      applyMove(s, m);
    });
    expect(isTerminal(s)).toBe(true);
    expect(getWinner(s)).toBe(0);
  });
});
