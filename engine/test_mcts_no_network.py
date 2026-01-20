#!/usr/bin/env python3
"""
Test MCTS without neural network to verify the search logic is correct.

Uses a dummy evaluator that returns uniform policy and neutral value.
MCTS should still find wins/losses by exploring to terminal states.

Board layout (8 rows x 7 cols = 56 squares):
  8 | 49 50 51 52 53 54 55  <- P0 goal row
  7 | 42 43 44 45 46 47 48
  6 | 35 36 37 38 39 40 41
  5 | 28 29 30 31 32 33 34
  4 | 21 22 23 24 25 26 27
  3 | 14 15 16 17 18 19 20
  2 |  7  8  9 10 11 12 13
  1 |  0  1  2  3  4  5  6  <- P1 goal row
    +---------------------
       a  b  c  d  e  f  g

Square = row * 7 + col
"""

import numpy as np
from razzle.core.state import GameState
from razzle.core.bitboard import rowcol_to_sq, sq_to_algebraic
from razzle.core.moves import move_to_algebraic, get_legal_moves
from razzle.ai.mcts import MCTS, MCTSConfig
from razzle.ai.network import NUM_ACTIONS


class DummyEvaluator:
    """Returns uniform policy and neutral value - lets MCTS figure it out via terminals."""
    def evaluate(self, state: GameState) -> tuple[np.ndarray, float]:
        policy = np.ones(NUM_ACTIONS, dtype=np.float32) / NUM_ACTIONS
        value = 0.0
        return policy, value

    def evaluate_batch(self, states: list[GameState]) -> list[tuple[np.ndarray, float]]:
        return [self.evaluate(s) for s in states]


def sq(file: str, rank: int) -> int:
    """Convert algebraic-like notation to square index."""
    col = ord(file) - ord('a')
    row = rank - 1
    return rowcol_to_sq(row, col)


def test_winning_position():
    """Test a position where P0 can win by passing to the goal."""
    print("="*60)
    print("Test: P0 one pass from winning")
    print("="*60)

    # P0's ball at f7 (sq 47), P0 knight at f8 (sq 54) to receive winning pass
    # Passing f7->f8 wins because f8 is on row 8 (P0's goal)
    state = GameState()

    # Clear the board
    state.pieces = [0, 0]
    state.balls = [0, 0]

    # P0 (Blue): ball at f7, knights at f8 and d4
    sq_f7 = sq('f', 7)  # 6*7 + 5 = 47
    sq_f8 = sq('f', 8)  # 7*7 + 5 = 54
    sq_d4 = sq('d', 4)  # 3*7 + 3 = 24

    state.balls[0] = 1 << sq_f7
    state.pieces[0] = (1 << sq_f8) | (1 << sq_d4)

    # P1 (Red): knights far away, ball at b5
    sq_a1 = sq('a', 1)  # 0
    sq_b1 = sq('b', 1)  # 1
    sq_g1 = sq('g', 1)  # 6
    sq_a2 = sq('a', 2)  # 7
    sq_b5 = sq('b', 5)  # 4*7 + 1 = 29

    state.pieces[1] = (1 << sq_a1) | (1 << sq_b1) | (1 << sq_g1) | (1 << sq_a2)
    state.balls[1] = 1 << sq_b5

    state.current_player = 0
    state.ply = 50

    print(f"\nP0 ball at f7 (sq {sq_f7}), P0 knight at f8 (sq {sq_f8})")
    print(f"Winning move: pass f7->f8 (ball reaches row 8)")
    print(f"Current player: {state.current_player} ({'Blue' if state.current_player == 0 else 'Red'})")

    # The winning move is pass f7->f8
    winning_move = sq_f7 * 56 + sq_f8
    print(f"Expected winning move encoding: {winning_move} = {move_to_algebraic(winning_move)}")

    # Verify legal moves include the winning move
    legal = get_legal_moves(state)
    print(f"Legal moves ({len(legal)}): {[move_to_algebraic(m) for m in legal[:10]]}...")
    assert winning_move in legal, f"Winning move {winning_move} not in legal moves!"

    # Run MCTS with dummy evaluator
    evaluator = DummyEvaluator()

    for sims in [100, 400, 800]:
        config = MCTSConfig(
            num_simulations=sims,
            temperature=0,
            dirichlet_epsilon=0
        )
        mcts = MCTS(evaluator, config)

        root = mcts.search(state, add_noise=False)
        best_move = mcts.select_move(root)

        top_moves = []
        for action, child in sorted(root.children.items(), key=lambda x: -x[1].visit_count)[:5]:
            top_moves.append({
                'move': move_to_algebraic(action),
                'visits': child.visit_count,
                'value': child.value
            })

        print(f"\nSimulations: {sims}")
        print(f"Best move: {move_to_algebraic(best_move)}")
        print(f"Top moves:")
        for m in top_moves:
            print(f"  {m['move']}: visits={m['visits']}, value={m['value']:.3f}")

        if best_move == winning_move:
            print(f"✓ MCTS found winning move!")
        else:
            print(f"✗ MCTS did NOT find winning move")


def test_defensive_position():
    """Test a position where current player must block or lose."""
    print("\n" + "="*60)
    print("Test: P1 must block P0's winning pass")
    print("="*60)

    # P0 (Blue) has ball at f7, knight at f8 - one pass from winning
    # P1 (Red) to move - P1's ball is far from goal, so P1 must block or lose
    state = GameState()

    state.pieces = [0, 0]
    state.balls = [0, 0]

    # P0: ball at f7, knights at f8 and d4
    sq_f7 = sq('f', 7)  # 47
    sq_f8 = sq('f', 8)  # 54
    sq_d4 = sq('d', 4)  # 24

    state.balls[0] = 1 << sq_f7
    state.pieces[0] = (1 << sq_f8) | (1 << sq_d4)

    # P1: knight at d5 can move to e7 or f6 (adjacent to f7, forcing pass)
    # P1's ball is at b6 - far from P1's goal (row 1)
    sq_d5 = sq('d', 5)  # 31
    sq_a3 = sq('a', 3)  # 14
    sq_b4 = sq('b', 4)  # 22
    sq_a5 = sq('a', 5)  # 28
    sq_b6 = sq('b', 6)  # 36 - ball (far from row 1)

    state.pieces[1] = (1 << sq_d5) | (1 << sq_a3) | (1 << sq_b4) | (1 << sq_a5)
    state.balls[1] = 1 << sq_b6

    state.current_player = 1  # Red to move
    state.ply = 51

    # Defensive moves: d5->e7 or d5->f6 (adjacent to f7, forcing P0 to pass away from f8)
    sq_e7 = sq('e', 7)  # 46
    sq_f6 = sq('f', 6)  # 40

    print(f"\nP1 to move. P0 threatens f7->f8 winning pass.")
    print(f"P1's ball at b6 - too far from goal row 1 to win quickly")
    print(f"P1 knight at d5 (sq {sq_d5}) can move to e7 (sq {sq_e7}) or f6 (sq {sq_f6})")
    print(f"Both are adjacent to P0's ball at f7, forcing P0 to pass away from the goal.")
    print(f"Current player: {state.current_player} ({'Blue' if state.current_player == 0 else 'Red'})")

    defensive_move1 = sq_d5 * 56 + sq_e7
    defensive_move2 = sq_d5 * 56 + sq_f6
    print(f"Good defensive moves: d5-e7 ({defensive_move1}), d5-f6 ({defensive_move2})")

    # Verify legal moves
    legal = get_legal_moves(state)
    print(f"Legal moves ({len(legal)}): {[move_to_algebraic(m) for m in legal[:10]]}...")

    evaluator = DummyEvaluator()

    for sims in [400, 800, 1600, 3200]:
        config = MCTSConfig(
            num_simulations=sims,
            temperature=0,
            dirichlet_epsilon=0
        )
        mcts = MCTS(evaluator, config)

        root = mcts.search(state, add_noise=False)
        best_move = mcts.select_move(root)

        top_moves = []
        for action, child in sorted(root.children.items(), key=lambda x: -x[1].visit_count)[:5]:
            top_moves.append({
                'move': move_to_algebraic(action),
                'action': action,
                'visits': child.visit_count,
                'value': child.value
            })

        print(f"\nSimulations: {sims}")
        print(f"Best move: {move_to_algebraic(best_move)}")
        print(f"Top moves:")
        for m in top_moves:
            status = "✓ BLOCKS" if m['action'] in [defensive_move1, defensive_move2] else ""
            print(f"  {m['move']}: visits={m['visits']}, value={m['value']:.3f} {status}")


if __name__ == '__main__':
    test_winning_position()
    test_defensive_position()
