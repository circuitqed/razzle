import { describe, it, expect } from 'vitest';
import {
  ROWS,
  COLS,
  NUM_SQUARES,
  NUM_ACTIONS,
  END_TURN_MOVE,
  END_TURN_ACTION,
  P1_START_PIECES,
  P1_START_BALL,
  P2_START_PIECES,
  P2_START_BALL,
  ROW_1_MASK,
  ROW_8_MASK,
  KNIGHT_ATTACKS,
  bit,
  popcount,
  lsb,
  iterBits,
  sqToRowCol,
  rowcolToSq,
  sqToAlgebraic,
  algebraicToSq,
} from '../bitboard';
import {
  newGame,
  copyState,
  applyMove,
  isTerminal,
  getWinner,
  getOccupied,
  // EngineState type used indirectly
} from '../state';
import { getLegalMoves, getKnightMoves, getPassMoves, mustPass, encodeMove, decodeMove } from '../moves';
import { stateToTensor } from '../tensor';
import { MOVE_ROTATION_MAP, rotatePolicy180 } from '../symmetry';
import { RandomEvaluator } from '../evaluator';
import { search } from '../mcts';

// ─── Bitboard tests ───

describe('bitboard', () => {
  it('has correct constants', () => {
    expect(ROWS).toBe(8);
    expect(COLS).toBe(7);
    expect(NUM_SQUARES).toBe(56);
    expect(NUM_ACTIONS).toBe(3137);
    expect(END_TURN_ACTION).toBe(3136);
  });

  it('sqToRowCol and rowcolToSq are inverse', () => {
    for (let sq = 0; sq < NUM_SQUARES; sq++) {
      const [r, c] = sqToRowCol(sq);
      expect(rowcolToSq(r, c)).toBe(sq);
    }
  });

  it('sqToAlgebraic and algebraicToSq roundtrip', () => {
    expect(sqToAlgebraic(0)).toBe('a1');
    expect(sqToAlgebraic(3)).toBe('d1');
    expect(sqToAlgebraic(52)).toBe('d8');
    expect(sqToAlgebraic(55)).toBe('g8');
    expect(algebraicToSq('a1')).toBe(0);
    expect(algebraicToSq('d1')).toBe(3);
    expect(algebraicToSq('d8')).toBe(52);
  });

  it('bit and popcount', () => {
    expect(bit(0)).toBe(1n);
    expect(bit(3)).toBe(8n);
    expect(popcount(0n)).toBe(0);
    expect(popcount(P1_START_PIECES)).toBe(5);
    expect(popcount(P2_START_PIECES)).toBe(5);
  });

  it('lsb', () => {
    expect(lsb(0n)).toBe(-1);
    expect(lsb(1n)).toBe(0);
    expect(lsb(8n)).toBe(3);
    expect(lsb(P1_START_PIECES)).toBe(1); // bit 1 = b1
  });

  it('iterBits', () => {
    const bits = [...iterBits(P1_START_PIECES)];
    expect(bits).toEqual([1, 2, 3, 4, 5]); // b1, c1, d1, e1, f1
  });

  it('KNIGHT_ATTACKS are precomputed correctly', () => {
    // d1 (sq 3) should have knight moves to: b2(8), c3(16), e3(18), f2(12)
    // Actually d1 is row 0, col 3. Knight deltas: (2,-1)=>(2,2)=sq16, (2,1)=>(2,4)=sq18,
    // (1,-2)=>(1,1)=sq8, (1,2)=>(1,5)=sq12
    // Negative row deltas would go off board
    const attacks = KNIGHT_ATTACKS[3];
    expect(attacks & bit(8)).not.toBe(0n); // b2
    expect(attacks & bit(12)).not.toBe(0n); // f2
    expect(attacks & bit(16)).not.toBe(0n); // c3
    expect(attacks & bit(18)).not.toBe(0n); // e3
    // Should not have d3 (17)
    expect(attacks & bit(17)).toBe(0n);
  });

  it('starting positions have 5 pieces each', () => {
    expect(popcount(P1_START_PIECES)).toBe(5);
    expect(popcount(P2_START_PIECES)).toBe(5);
  });

  it('ROW_1_MASK and ROW_8_MASK cover correct rows', () => {
    for (let col = 0; col < COLS; col++) {
      expect(ROW_1_MASK & bit(rowcolToSq(0, col))).not.toBe(0n);
      expect(ROW_8_MASK & bit(rowcolToSq(7, col))).not.toBe(0n);
    }
    // Row 2 should NOT be in ROW_1_MASK
    expect(ROW_1_MASK & bit(rowcolToSq(1, 0))).toBe(0n);
  });
});

// ─── State tests ───

describe('state', () => {
  it('newGame creates correct starting position', () => {
    const s = newGame();
    expect(s.currentPlayer).toBe(0);
    expect(s.hasPassed).toBe(false);
    expect(s.lastKnightDst).toBe(-1);
    expect(s.ply).toBe(0);
    expect(s.pieces[0]).toBe(P1_START_PIECES);
    expect(s.pieces[1]).toBe(P2_START_PIECES);
    expect(s.balls[0]).toBe(P1_START_BALL);
    expect(s.balls[1]).toBe(P2_START_BALL);
    expect(s.touchedMask).toBe(0n);
  });

  it('initial state is not terminal', () => {
    expect(isTerminal(newGame())).toBe(false);
    expect(getWinner(newGame())).toBeNull();
  });

  it('copyState makes independent copy', () => {
    const s = newGame();
    const c = copyState(s);
    applyMove(c, getLegalMoves(c)[0]);
    // Original should be unchanged
    expect(s.currentPlayer).toBe(0);
    expect(s.ply).toBe(0);
  });

  it('getOccupied returns all occupied squares', () => {
    const s = newGame();
    const occ = getOccupied(s);
    // 5 pieces + 1 ball per player = 12 occupied (ball overlaps a piece)
    // Actually ball is ON one of the pieces, so 10 occupied
    expect(popcount(occ)).toBe(10);
  });
});

// ─── Move tests ───

describe('moves', () => {
  it('encodeMove and decodeMove roundtrip', () => {
    expect(encodeMove(3, 17)).toBe(3 * 56 + 17);
    expect(decodeMove(3 * 56 + 17)).toEqual([3, 17]);
  });

  it('initial position has legal moves', () => {
    const s = newGame();
    const moves = getLegalMoves(s);
    expect(moves.length).toBeGreaterThan(0);
    // Should only be knight moves (no pass or end turn from starting position)
    expect(moves.includes(END_TURN_MOVE)).toBe(false);
  });

  it('initial position has only knight moves (ball has no untouched receivers yet)', () => {
    const s = newGame();
    const knightMoves = getKnightMoves(s);
    const passMoves = getPassMoves(s);
    const legalMoves = getLegalMoves(s);
    // All pieces are fresh, so ball CAN pass to them
    // Ball at d1 (sq3), pieces at b1(1), c1(2), d1(3), e1(4), f1(5)
    // Ball can pass horizontally to c1(2) and e1(4) - adjacent pieces
    // But d1 has the ball, so valid receivers are other pieces NOT in touchedMask
    // Since touchedMask is 0, all other pieces can receive
    expect(legalMoves.length).toBe(knightMoves.length + passMoves.length);
  });

  it('knight move changes player', () => {
    const s = newGame();
    const knightMoves = getKnightMoves(s);
    expect(knightMoves.length).toBeGreaterThan(0);
    const move = knightMoves[0];
    applyMove(s, move);
    expect(s.currentPlayer).toBe(1);
    expect(s.ply).toBe(1);
    expect(s.hasPassed).toBe(false);
  });

  it('ball pass does not change player', () => {
    const s = newGame();
    const passMoves = getPassMoves(s);
    if (passMoves.length > 0) {
      const prevPlayer = s.currentPlayer;
      applyMove(s, passMoves[0]);
      expect(s.currentPlayer).toBe(prevPlayer);
      expect(s.hasPassed).toBe(true);
    }
  });

  it('after pass, can only pass more or end turn', () => {
    const s = newGame();
    const passMoves = getPassMoves(s);
    if (passMoves.length > 0) {
      applyMove(s, passMoves[0]);
      const legal = getLegalMoves(s);
      // Should have END_TURN and possibly more passes, but no knight moves
      expect(legal.includes(END_TURN_MOVE)).toBe(true);
      for (const m of legal) {
        if (m !== END_TURN_MOVE) {
          // Should be a pass move (from ball position)
          const [src] = decodeMove(m);
          const ballSq = lsb(s.balls[s.currentPlayer]);
          expect(src).toBe(ballSq);
        }
      }
    }
  });

  it('end turn switches player and resets hasPassed', () => {
    const s = newGame();
    const passMoves = getPassMoves(s);
    if (passMoves.length > 0) {
      applyMove(s, passMoves[0]);
      expect(s.hasPassed).toBe(true);
      applyMove(s, END_TURN_MOVE);
      expect(s.hasPassed).toBe(false);
      expect(s.currentPlayer).toBe(1);
    }
  });

  it('touched mask persists across end turn', () => {
    const s = newGame();
    const passMoves = getPassMoves(s);
    if (passMoves.length > 0) {
      applyMove(s, passMoves[0]);
      const maskAfterPass = s.touchedMask;
      expect(maskAfterPass).not.toBe(0n);
      applyMove(s, END_TURN_MOVE);
      // touchedMask should NOT be reset
      expect(s.touchedMask).toBe(maskAfterPass);
    }
  });

  it('knight move clears source bit from touchedMask', () => {
    const s = newGame();
    // Make a pass to set touchedMask
    const passMoves = getPassMoves(s);
    if (passMoves.length > 0) {
      const passMove = passMoves[0];
      const [, dst] = decodeMove(passMove);
      applyMove(s, passMove);
      const dstBit = bit(dst);
      // dst bit should be in touchedMask
      expect(s.touchedMask & dstBit).not.toBe(0n);
      // End turn
      applyMove(s, END_TURN_MOVE);
      // Now P2 moves a knight - that piece's src bit should be cleared
      const p2Knights = getKnightMoves(s);
      if (p2Knights.length > 0) {
        const knightMove = p2Knights[0];
        const [src] = decodeMove(knightMove);
        const srcBit = bit(src);
        // Only if src is in touchedMask (likely not from starting, but let's set it up)
        applyMove(s, knightMove);
        expect(s.touchedMask & srcBit).toBe(0n);
      }
    }
  });
});

// ─── Terminal detection ───

describe('terminal detection', () => {
  it('p1 wins when ball reaches row 8', () => {
    const s = newGame();
    // Manually set p1's ball to row 8
    s.balls[0] = bit(rowcolToSq(7, 3)); // d8
    expect(isTerminal(s)).toBe(true);
    expect(getWinner(s)).toBe(0);
  });

  it('p2 wins when ball reaches row 1', () => {
    const s = newGame();
    s.balls[1] = bit(rowcolToSq(0, 3)); // d1
    expect(isTerminal(s)).toBe(true);
    expect(getWinner(s)).toBe(1);
  });

  it('ply > 200 is terminal', () => {
    const s = newGame();
    s.ply = 201;
    expect(isTerminal(s)).toBe(true);
    expect(getWinner(s)).toBe(1); // opponent of current player (0) wins
  });
});

// ─── Tensor tests ───

describe('tensor', () => {
  it('produces correct shape', () => {
    const s = newGame();
    const t = stateToTensor(s);
    expect(t.length).toBe(7 * 8 * 7); // 392
  });

  it('has pieces in correct planes for P1', () => {
    const s = newGame();
    const t = stateToTensor(s);

    // Plane 0 (P1 pieces at row 0, cols 1-5)
    for (let col = 1; col <= 5; col++) {
      expect(t[0 * 56 + 0 * 7 + col]).toBe(1.0);
    }
    // a1 and g1 should be 0
    expect(t[0 * 56 + 0 * 7 + 0]).toBe(0.0);
    expect(t[0 * 56 + 0 * 7 + 6]).toBe(0.0);

    // Plane 1 (P1 ball at d1 = row 0, col 3)
    expect(t[1 * 56 + 0 * 7 + 3]).toBe(1.0);

    // Plane 2 (P2 pieces at row 7, cols 1-5)
    for (let col = 1; col <= 5; col++) {
      expect(t[2 * 56 + 7 * 7 + col]).toBe(1.0);
    }

    // Plane 5 should be all 1s
    for (let i = 0; i < 56; i++) {
      expect(t[5 * 56 + i]).toBe(1.0);
    }

    // Plane 6 should be all 0s (hasPassed=false)
    for (let i = 0; i < 56; i++) {
      expect(t[6 * 56 + i]).toBe(0.0);
    }
  });

  it('rotates for P2 perspective', () => {
    const s = newGame();
    s.currentPlayer = 1;
    const t = stateToTensor(s);

    // For P2, the board is rotated 180. "Current player" is P2.
    // After rotation, P2's pieces (originally row 7) should appear at row 0.
    // Plane 0 = current player's pieces = P2 pieces, after rotation at row 0, cols 1-5
    for (let col = 1; col <= 5; col++) {
      expect(t[0 * 56 + 0 * 7 + col]).toBe(1.0);
    }

    // Plane 2 = opponent's pieces = P1 pieces, after rotation at row 7, cols 1-5
    for (let col = 1; col <= 5; col++) {
      expect(t[2 * 56 + 7 * 7 + col]).toBe(1.0);
    }
  });

  it('hasPassed sets plane 6 to all 1s', () => {
    const s = newGame();
    // Make a pass to set hasPassed
    const passMoves = getPassMoves(s);
    if (passMoves.length > 0) {
      applyMove(s, passMoves[0]);
      expect(s.hasPassed).toBe(true);
      const t = stateToTensor(s);
      for (let i = 0; i < 56; i++) {
        expect(t[6 * 56 + i]).toBe(1.0);
      }
    }
  });
});

// ─── Symmetry tests ───

describe('symmetry', () => {
  it('MOVE_ROTATION_MAP has correct size', () => {
    expect(MOVE_ROTATION_MAP.length).toBe(NUM_ACTIONS);
  });

  it('END_TURN maps to itself', () => {
    expect(MOVE_ROTATION_MAP[END_TURN_ACTION]).toBe(END_TURN_ACTION);
  });

  it('rotation is self-inverse', () => {
    for (let i = 0; i < NUM_ACTIONS; i++) {
      const rotated = MOVE_ROTATION_MAP[i];
      expect(MOVE_ROTATION_MAP[rotated]).toBe(i);
    }
  });

  it('rotatePolicy180 is self-inverse', () => {
    const policy = new Float32Array(NUM_ACTIONS);
    for (let i = 0; i < NUM_ACTIONS; i++) {
      policy[i] = i / NUM_ACTIONS; // Unique values
    }
    const rotated = rotatePolicy180(policy);
    const back = rotatePolicy180(rotated);
    for (let i = 0; i < NUM_ACTIONS; i++) {
      expect(Math.abs(back[i] - policy[i])).toBeLessThan(1e-6);
    }
  });
});

// ─── MCTS with RandomEvaluator ───

describe('MCTS with RandomEvaluator', () => {
  it('completes search and returns a legal move', async () => {
    const s = newGame();
    const evaluator = new RandomEvaluator();
    const result = await search(s, evaluator, { numSimulations: 50 });

    expect(result.simsDone).toBeGreaterThan(0);
    expect(result.simsDone).toBeLessThanOrEqual(50);

    // bestMove should be in legal moves
    const legal = getLegalMoves(s);
    expect(legal).toContain(result.bestMove);
  });

  it('respects abort signal', async () => {
    const s = newGame();
    const evaluator = new RandomEvaluator();
    const abortFlag = { aborted: true }; // Abort immediately

    const result = await search(s, evaluator, { numSimulations: 1000 }, abortFlag);
    // Should complete quickly since abort is already set
    expect(result.simsDone).toBeLessThan(1000);
  });
});

// ─── mustPass tests ───

describe('forced pass rule', () => {
  it('mustPass returns false when no last knight move', () => {
    const s = newGame();
    expect(mustPass(s)).toBe(false);
  });

  it('mustPass returns true when last knight dst is adjacent to ball', () => {
    const s = newGame();
    // Simulate: opponent moved a knight to c2 (sq 9), ball is at d1 (sq 3)
    // c2 is adjacent to d1 (row diff 1, col diff 1)
    s.lastKnightDst = 9; // c2
    // Ball at d1 (default for P1)
    expect(mustPass(s)).toBe(true);
  });

  it('mustPass returns false when last knight dst is not adjacent', () => {
    const s = newGame();
    // e3 (sq 18) is not adjacent to d1 (sq 3) - row diff 2
    s.lastKnightDst = 18;
    expect(mustPass(s)).toBe(false);
  });
});
