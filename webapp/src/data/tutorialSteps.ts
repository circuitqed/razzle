import type { EngineState } from '../engine/state';

export interface TutorialStep {
  id: string;
  title: string;
  instruction: string;
  hint: string;
  boardState: EngineState;
  playerColor: number;
  allowedMoves: number[];
  highlightSquares?: number[];
  requireEndTurn?: boolean;
  autoAdvance?: boolean;
  nextStepState?: EngineState;
  chainMoves?: number[];
  /** State to show briefly before the interactive state (opponent about to move). */
  preState?: EngineState;
  preMessage?: string;
  /** The opponent's move to animate during preState→mainState transition. */
  preMove?: { from: number; to: number };
  /** Message shown after the user completes the step. */
  completionMessage?: string;
  /** State shown briefly during chain transition (opponent moving). */
  chainPreState?: EngineState;
  chainPreMessage?: string;
  /** The opponent's move to animate during chain pre-transition. */
  chainPreMove?: { from: number; to: number };
  /** Moves to visually emphasize (green). Other allowed moves shown as gray. */
  suggestedMoves?: number[];
}

const b = (sq: number): bigint => 1n << BigInt(sq);
const bits = (...sqs: number[]): bigint => sqs.reduce((acc, sq) => acc | b(sq), 0n);
const mv = (src: number, dst: number): number => src * 56 + dst;

// Board:
//   8 | 49 50 51 52 53 54 55    (goal for player 0)
//   7 | 42 43 44 45 46 47 48
//   6 | 35 36 37 38 39 40 41
//   5 | 28 29 30 31 32 33 34
//   4 | 21 22 23 24 25 26 27
//   3 | 14 15 16 17 18 19 20
//   2 |  7  8  9 10 11 12 13
//   1 |  0  1  2  3  4  5  6
//        a  b  c  d  e  f  g
//
// 5 pieces + 1 ball per player. No captures. Ball carrier is in pieces[].

// ─── Step 1: Move a Knight ──────────────────────────────────────────
const step1State: EngineState = {
  pieces: [bits(1, 2, 3, 4, 5), bits(50, 51, 52, 53, 54)],
  balls: [b(3), b(52)],
  currentPlayer: 0,
  touchedMask: 0n,
  hasPassed: false,
  lastKnightDst: -1,
  ply: 0,
};

// ─── Step 2: Pass the Ball ──────────────────────────────────────────
// Ball at d1=3. Two receivers: a4=21 (NW diagonal) and d3=17 (north).
// Other pieces (f4=26, g5=34) off all lines from d1, d3, and a4.
const step2State: EngineState = {
  pieces: [bits(3, 17, 21, 26, 34), bits(42, 48, 50, 52, 54)],
  balls: [b(3), b(52)],
  currentPlayer: 0,
  touchedMask: 0n,
  hasPassed: false,
  lastKnightDst: -1,
  ply: 4,
};

// ─── Step 3: Chain Passes ───────────────────────────────────────────
// Ball at c3=16, teammates at c5=30 and e7=46.
// Pass c3→c5, then c5→e7 (NE diagonal), then Complete Pass.
const step3State: EngineState = {
  pieces: [bits(1, 5, 16, 30, 46), bits(42, 48, 50, 53, 54)],
  balls: [b(16), b(52)],
  currentPlayer: 0,
  touchedMask: 0n,
  hasPassed: false,
  lastKnightDst: -1,
  ply: 6,
};

const step3After1stPass: EngineState = {
  pieces: [bits(1, 5, 16, 30, 46), bits(42, 48, 50, 53, 54)],
  balls: [b(30), b(52)],
  currentPlayer: 0,
  touchedMask: bits(16, 30),
  hasPassed: true,
  lastKnightDst: -1,
  ply: 6,
};

// ─── Step 4: Score! ─────────────────────────────────────────────────
// Ball at d7=45, teammate at d8=52. Pass scores.
const step4State: EngineState = {
  pieces: [bits(8, 22, 32, 45, 52), bits(42, 44, 48, 50, 54)],
  balls: [b(45), b(50)],
  currentPlayer: 0,
  touchedMask: 0n,
  hasPassed: false,
  lastKnightDst: -1,
  ply: 20,
};

// ─── Step 5: Clearing Touched Pieces ────────────────────────────────
// Ball at d4=24. Touched pieces at e5=32 (NE diagonal from d4) and
// g7=48 (also NE but behind e5). Other pieces a3=14, g3=20 (off lines).
// Opponent at d6=38 (blocks north from d4), a7=42, b8=50, d8=52, f8=54.
//
// User moves e5=32 to c4=23 (west of d4) or d3=17 (south of d4).
// After move + opponent response, pass to the cleared piece.
const step5State: EngineState = {
  pieces: [bits(14, 20, 24, 32, 48), bits(38, 42, 50, 52, 54)],
  balls: [b(24), b(52)],
  currentPlayer: 0,
  touchedMask: bits(24, 32, 48),
  hasPassed: false,
  lastKnightDst: -1,
  ply: 10,
};

// We use a dynamic approach: the hook applies the actual move, then
// loads this template state (representing the opponent having responded).
// The nextStepState is for e5→c4 case. For e5→d3, the chainMoves
// also include mv(24,17) so either works.
// After e5→c4=23, opponent moves d6=38→f5=33 (knight move).
const step5AfterMove: EngineState = {
  pieces: [bits(14, 20, 23, 24, 48), bits(33, 42, 50, 52, 54)],
  balls: [b(24), b(52)],
  currentPlayer: 0,
  touchedMask: bits(24, 48),
  hasPassed: false,
  lastKnightDst: -1,
  ply: 12,
};

// ─── Step 6: Forced Pass ────────────────────────────────────────────
// Show the board BEFORE opponent moves (preState), then after a delay
// transition to the state where opponent knight is at e5=32 adjacent
// to ball at d4=24. Must pass to b4=22.

// Before: opponent knight at d7=45, about to move to e5=32
const step6PreState: EngineState = {
  pieces: [bits(2, 5, 22, 24, 39), bits(42, 45, 48, 50, 54)],
  balls: [b(24), b(50)],
  currentPlayer: 1, // opponent's turn (about to move)
  touchedMask: 0n,
  hasPassed: false,
  lastKnightDst: -1,
  ply: 8,
};

// After: opponent moved c6=37→e5=32, now adjacent to ball at d4=24
const step6State: EngineState = {
  pieces: [bits(2, 5, 22, 24, 39), bits(32, 42, 48, 50, 54)],
  balls: [b(24), b(50)],
  currentPlayer: 0,
  touchedMask: 0n,
  hasPassed: false,
  lastKnightDst: 32,
  ply: 9,
};

// ─── Export ─────────────────────────────────────────────────────────

// Pre-state for eligible receivers: before opponent moves, show the state
// right after the user's knight move (opponent about to respond)
const step5PreOpponent: EngineState = {
  pieces: [bits(14, 20, 23, 24, 48), bits(38, 42, 50, 52, 54)],  // d6=38 still here, about to move
  balls: [b(24), b(52)],
  currentPlayer: 1,
  touchedMask: bits(24, 48),
  hasPassed: false,
  lastKnightDst: -1,
  ply: 11,
};

export const TUTORIAL_STEPS: TutorialStep[] = [
  {
    id: 'move-knight',
    title: 'Move a Knight',
    instruction: 'Knights move in an L-shape, just like chess. Moving a knight ends your turn. Tap any knight, then tap a highlighted square.',
    hint: 'The piece carrying the ball can\'t move as a knight — it can only pass.',
    completionMessage: 'Great move!',
    boardState: step1State,
    playerColor: 0,
    allowedMoves: [
      mv(1, 10), mv(1, 14), mv(1, 16),
      mv(2, 7), mv(2, 11), mv(2, 15), mv(2, 17),
      mv(4, 9), mv(4, 13), mv(4, 17), mv(4, 19),
      mv(5, 10), mv(5, 18), mv(5, 20),
    ],
    highlightSquares: [],
    autoAdvance: true,
  },
  {
    id: 'pass-ball',
    title: 'Pass the Ball',
    instruction: 'The ball travels in a straight line — horizontally, vertically, or diagonally — to the nearest friendly piece. Tap the ball carrier, then tap a teammate.',
    hint: 'Passing does NOT end your turn.',
    completionMessage: 'Nice pass!',
    boardState: step2State,
    playerColor: 0,
    allowedMoves: [mv(3, 17), mv(3, 21)],
    highlightSquares: [3],
    autoAdvance: true,
  },
  {
    id: 'chain-passes',
    title: 'Chain Passes',
    instruction: 'You can pass multiple times in one turn! Pieces that already touched the ball (marked X) can\'t receive again. Pass north to c5, then diagonally to e7, then tap "Complete Pass."',
    hint: 'Plan your pass chains to move the ball across the board quickly.',
    completionMessage: 'Excellent chain!',
    boardState: step3State,
    playerColor: 0,
    allowedMoves: [mv(16, 30)],
    highlightSquares: [16, 30, 46],
    nextStepState: step3After1stPass,
    chainMoves: [mv(30, 46)],
    requireEndTurn: true,
  },
  {
    id: 'score',
    title: 'Score!',
    instruction: 'Get your ball to the opponent\'s back row to win the game. Pass the ball to your piece on row 8!',
    hint: 'The highlighted row is the goal.',
    completionMessage: 'Goal! You win!',
    boardState: step4State,
    playerColor: 0,
    allowedMoves: [mv(45, 52)],
    highlightSquares: [49, 50, 51, 52, 53, 54, 55],
    autoAdvance: true,
  },
  {
    id: 'eligible-receivers',
    title: 'Eligible Receivers',
    instruction: 'Pieces that touched the ball can\'t receive again until they make a knight move. Your nearby teammates are all ineligible (X). Move the knight at e5 to make it eligible, then pass to it!',
    hint: 'Moving a piece clears its X marker.',
    completionMessage: 'Well done! You cleared the piece and passed to it.',
    boardState: step5State,
    playerColor: 0,
    allowedMoves: [
      mv(32, 17), mv(32, 19), mv(32, 23), mv(32, 27),  // all legal knight moves from e5
      mv(32, 37), mv(32, 41), mv(32, 45), mv(32, 47),
    ],
    highlightSquares: [32],
    suggestedMoves: [mv(32, 23), mv(32, 17)], // c4 and d3 land on pass lines from d4
    nextStepState: step5AfterMove,
    chainPreState: step5PreOpponent,
    chainPreMessage: 'The opponent is making their move...',
    chainPreMove: { from: 38, to: 33 }, // d6→f5 (knight move)
    chainMoves: [mv(24, 23), mv(24, 17)],
  },
  {
    id: 'forced-pass',
    title: 'Forced Pass',
    instruction: 'When an opponent\'s knight lands next to your ball carrier, you must pass the ball before doing anything else. Pass to safety!',
    hint: 'The glowing ball means you\'re forced to pass. If you have no eligible receivers, the forced pass is skipped.',
    completionMessage: 'Quick thinking! Remember: if you have no eligible receivers, the forced pass is skipped and you can move normally.',
    boardState: step6State,
    preState: step6PreState,
    preMessage: 'The opponent is moving their knight next to your ball...',
    preMove: { from: 45, to: 32 }, // d7→e5
    playerColor: 0,
    allowedMoves: [mv(24, 22)],
    highlightSquares: [24, 32],
    autoAdvance: true,
  },
];
