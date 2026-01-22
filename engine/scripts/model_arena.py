#!/usr/bin/env python3
"""
Model Arena - Compare models by having them play against each other.
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.core.state import GameState
from razzle.core.moves import get_legal_moves
from razzle.ai.network import RazzleNet
from razzle.ai.mcts import MCTS, MCTSConfig
from razzle.ai.evaluator import BatchedEvaluator


@dataclass
class MatchResult:
    model1_wins: int = 0
    model2_wins: int = 0
    draws: int = 0

    @property
    def total(self) -> int:
        return self.model1_wins + self.model2_wins + self.draws

    def model1_win_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.model1_wins / self.total

    def model2_win_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.model2_wins / self.total


def play_game(mcts1: MCTS, mcts2: MCTS, verbose: bool = False) -> int:
    """
    Play a single game between two MCTS instances.

    Returns:
        1 if mcts1 wins, -1 if mcts2 wins, 0 for draw
    """
    state = GameState.new_game()
    move_count = 0
    max_moves = 500  # Prevent infinite games

    while not state.is_terminal() and move_count < max_moves:
        mcts = mcts1 if state.current_player == 0 else mcts2

        # Search
        root = mcts.search(state, add_noise=False)

        # Select best move (temperature = 0 for deterministic play)
        best_move = None
        best_visits = -1
        for move, child in root.children.items():
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_move = move

        if best_move is None:
            # No legal moves (shouldn't happen)
            break

        state.apply_move(best_move)
        move_count += 1

        if verbose:
            print(f"Move {move_count}: Player {1 - state.current_player} plays {best_move}")

    if state.is_terminal():
        # Game ended - check winner
        winner = state.get_winner()
        if winner == 0:
            return 1  # Player 0 (mcts1) wins
        elif winner == 1:
            return -1  # Player 1 (mcts2) wins
        else:
            return 0  # Draw
    else:
        # Max moves reached
        return 0


def run_match(
    model1_path: str,
    model2_path: str,
    num_games: int = 20,
    simulations: int = 200,
    device: str = 'cuda',
    verbose: bool = False
) -> MatchResult:
    """
    Run a match between two models.
    Each model plays as both Player 0 and Player 1.
    """
    # Load models
    print(f"Loading model 1: {model1_path}")
    net1 = RazzleNet.load(model1_path, device=device)
    net1.eval()

    print(f"Loading model 2: {model2_path}")
    net2 = RazzleNet.load(model2_path, device=device)
    net2.eval()

    # Create evaluators
    eval1 = BatchedEvaluator(net1, device=device)
    eval2 = BatchedEvaluator(net2, device=device)

    # Create MCTS instances
    config = MCTSConfig(num_simulations=simulations)
    mcts1 = MCTS(eval1, config)
    mcts2 = MCTS(eval2, config)

    result = MatchResult()

    # Play games with alternating colors
    games_per_side = num_games // 2

    print(f"\nPlaying {games_per_side} games with Model 1 as Player 0...")
    for i in range(games_per_side):
        outcome = play_game(mcts1, mcts2, verbose=verbose)
        if outcome == 1:
            result.model1_wins += 1
        elif outcome == -1:
            result.model2_wins += 1
        else:
            result.draws += 1

        if (i + 1) % 5 == 0:
            print(f"  Games: {i+1}/{games_per_side}, Model1: {result.model1_wins}, Model2: {result.model2_wins}, Draws: {result.draws}")

    print(f"\nPlaying {games_per_side} games with Model 2 as Player 0...")
    for i in range(games_per_side):
        # Swap order - model2 is player 0, model1 is player 1
        outcome = play_game(mcts2, mcts1, verbose=verbose)
        # Note: outcome is from model2's perspective as player 0
        if outcome == 1:
            result.model2_wins += 1
        elif outcome == -1:
            result.model1_wins += 1
        else:
            result.draws += 1

        if (i + 1) % 5 == 0:
            total_played = games_per_side + i + 1
            print(f"  Games: {i+1}/{games_per_side}, Model1: {result.model1_wins}, Model2: {result.model2_wins}, Draws: {result.draws}")

    return result


def main():
    parser = argparse.ArgumentParser(description='Compare models by having them play against each other')
    parser.add_argument('model1', type=str, help='Path to first model')
    parser.add_argument('model2', type=str, help='Path to second model')
    parser.add_argument('--games', type=int, default=20, help='Number of games to play (default: 20)')
    parser.add_argument('--simulations', type=int, default=200, help='MCTS simulations per move (default: 200)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--verbose', action='store_true', help='Print moves')

    args = parser.parse_args()

    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    result = run_match(
        args.model1,
        args.model2,
        num_games=args.games,
        simulations=args.simulations,
        device=args.device,
        verbose=args.verbose
    )

    print("\n" + "=" * 50)
    print("MATCH RESULTS")
    print("=" * 50)
    print(f"Model 1: {Path(args.model1).name}")
    print(f"Model 2: {Path(args.model2).name}")
    print(f"Games played: {result.total}")
    print(f"Model 1 wins: {result.model1_wins} ({result.model1_win_rate()*100:.1f}%)")
    print(f"Model 2 wins: {result.model2_wins} ({result.model2_win_rate()*100:.1f}%)")
    print(f"Draws: {result.draws}")
    print("=" * 50)


if __name__ == '__main__':
    main()
