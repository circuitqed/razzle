# Razzle Dazzle - Official Rules

*Based on the original rules from [SuperDuperGames](https://superdupergames.org/rules/razzle.pdf)*

## Overview

Razzle Dazzle is a two-player abstract strategy game played on an 8×7 board. Each player controls 5 pieces that move like chess knights, plus a ball. The objective is to advance your ball to your opponent's back row.

## Board Setup

```
  8 | .  o  o  O  o  o  . |  ← Player 2 (O) starts here
  7 | .  .  .  .  .  .  . |
  6 | .  .  .  .  .  .  . |
  5 | .  .  .  .  .  .  . |
  4 | .  .  .  .  .  .  . |
  3 | .  .  .  .  .  .  . |
  2 | .  .  .  .  .  .  . |
  1 | .  x  x  X  x  x  . |  ← Player 1 (X) starts here
    +---------------------+
       a  b  c  d  e  f  g

Legend:
  X/O = piece with ball (uppercase)
  x/o = piece without ball (lowercase)
  .   = empty square
```

- **Player 1 (X)**: Pieces on b1, c1, d1, e1, f1 with ball on d1
- **Player 2 (O)**: Pieces on b8, c8, d8, e8, f8 with ball on d8

## Turn Structure

On your turn, you must do ONE of the following:
1. **Move a piece** - make one knight move, turn ends
2. **Pass the ball** - pass one or more times, then turn ends

You **cannot** do both in the same turn. It's either moving OR passing, not both.

## Piece Movement

- Pieces move exactly like chess knights: in an "L" shape (2 squares in one direction, 1 square perpendicular)
- A piece can only move to an **empty square**
- The piece holding the ball **cannot move** - you must pass the ball away first
- Moving a piece **ends your turn**

### Knight Move Pattern
```
    . . . . . . .
    . . 2 . 2 . .
    . 1 . . . 1 .
    . . . N . . .
    . 1 . . . 1 .
    . . 2 . 2 . .
    . . . . . . .
```
From position N, the knight can move to any square marked 1 or 2.

## Ball Passing

- The ball can be passed in a **straight line** (horizontal, vertical, or diagonal) to any of your pieces
- The pass cannot go through other pieces (blocked by any piece in the way)
- You may pass the ball **multiple times** in a single turn

### Ineligible Receivers (Critical Rule!)

When a piece passes OR receives the ball:
1. That piece becomes an **ineligible receiver**
2. An ineligible piece **cannot receive passes** until it moves
3. This restriction **persists across turns** - it doesn't reset when your turn ends!
4. Once an ineligible piece makes a knight move, it becomes eligible again

**Example**: If piece on c1 passes to e1, BOTH c1 and e1 become ineligible. On future turns, you cannot pass to c1 or e1 until they move.

## Forced Pass Rule

If your opponent's **most recent move** places one of their pieces **adjacent** to your ball (including diagonally adjacent), you **must pass** if possible.

- "Adjacent" means any of the 8 surrounding squares
- This only applies if the opponent JUST moved there (not if they were already adjacent)
- If you cannot pass (no eligible receivers in line), you may move a piece instead

## Winning the Game

- **Player 1 wins** by getting their ball to row 8 (the opponent's back row)
- **Player 2 wins** by getting their ball to row 1 (the opponent's back row)

The ball reaches the goal row when passed to a piece on that row.

## Draw Conditions

The game is drawn if:
- The same board position occurs three times (threefold repetition)
- An excessive number of moves occur without progress (200+ ply in our implementation)

## Move Notation

Moves are written in algebraic notation:
- Knight move: `b1-c3` (piece moves from b1 to c3)
- Ball pass: `d1-e1` (ball passes from d1 to e1)
- End turn: `end` (explicitly end turn after passing)

## Summary of Key Rules

1. ✓ On your turn: EITHER move a piece OR pass the ball (not both)
2. ✓ Pieces move like knights to empty squares
3. ✓ Piece with ball cannot move
4. ✓ Ball passes in straight lines, blocked by pieces
5. ✓ Multiple passes allowed per turn (but no knight move after passing)
6. ✓ Passers AND receivers become ineligible until they move
7. ✓ Forced pass when opponent moves adjacent to your ball
8. ✓ Win by getting ball to opponent's back row
