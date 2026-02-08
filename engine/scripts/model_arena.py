#!/usr/bin/env python3
"""
Model Arena - Compare models by having them play against each other.
"""

import argparse
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np
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


def play_game(
    mcts1: MCTS,
    mcts2: MCTS,
    verbose: bool = False,
    opening_moves: int = 0,
    opening_temperature: float = 1.0,
) -> int:
    """
    Play a single game between two MCTS instances.

    Args:
        mcts1: MCTS instance for player 0
        mcts2: MCTS instance for player 1
        verbose: Print moves as they happen
        opening_moves: Number of moves to use temperature-based selection
        opening_temperature: Temperature for opening moves (higher = more random)

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

        # Select move - use temperature for opening moves
        if move_count < opening_moves and opening_temperature > 0:
            # Temperature-based selection for opening diversity
            moves = list(root.children.keys())
            visits = np.array([root.children[m].visit_count for m in moves], dtype=np.float32)

            if visits.sum() > 0:
                # Apply temperature: higher temp = more uniform distribution
                visits = np.power(visits, 1.0 / opening_temperature)
                probs = visits / visits.sum()
                best_move = moves[np.random.choice(len(moves), p=probs)]
            else:
                best_move = random.choice(moves) if moves else None
        else:
            # Greedy selection (temperature = 0)
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
    verbose: bool = False,
    opening_moves: int = 3,
    opening_temperature: float = 1.0,
    parallel_games: int = 1,
) -> MatchResult:
    """
    Run a match between two models.
    Each model plays as both Player 0 and Player 1.

    Args:
        model1_path: Path to first model
        model2_path: Path to second model
        num_games: Total number of games to play
        simulations: MCTS simulations per move
        device: Device for inference (cuda/cpu)
        verbose: Print moves as they happen
        opening_moves: Number of moves to use temperature-based selection (default: 3)
        opening_temperature: Temperature for opening moves (default: 1.0)
        parallel_games: Number of games to run in parallel (default: 1)
    """
    # Load models
    print(f"Loading model 1: {model1_path}")
    net1 = RazzleNet.load(model1_path, device=device)
    net1.eval()

    print(f"Loading model 2: {model2_path}")
    net2 = RazzleNet.load(model2_path, device=device)
    net2.eval()

    # Create evaluators (shared across threads - thread-safe for inference)
    eval1 = BatchedEvaluator(net1, device=device)
    eval2 = BatchedEvaluator(net2, device=device)

    config = MCTSConfig(num_simulations=simulations)
    result = MatchResult()

    # Play games with alternating colors
    games_per_side = num_games // 2

    def play_single_game(model1_first: bool) -> int:
        """Play a single game with thread-local MCTS instances."""
        # Create fresh MCTS instances per game (tree state is not shared)
        mcts1 = MCTS(eval1, config)
        mcts2 = MCTS(eval2, config)

        if model1_first:
            return play_game(mcts1, mcts2, verbose=verbose,
                           opening_moves=opening_moves,
                           opening_temperature=opening_temperature)
        else:
            # Swap order - model2 is player 0
            outcome = play_game(mcts2, mcts1, verbose=verbose,
                              opening_moves=opening_moves,
                              opening_temperature=opening_temperature)
            # Flip outcome to be from model1's perspective
            return -outcome

    if parallel_games <= 1:
        # Sequential execution (original behavior)
        print(f"\nPlaying {games_per_side} games with Model 1 as Player 0...")
        for i in range(games_per_side):
            outcome = play_single_game(model1_first=True)
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
            outcome = play_single_game(model1_first=False)
            if outcome == 1:
                result.model1_wins += 1
            elif outcome == -1:
                result.model2_wins += 1
            else:
                result.draws += 1
            if (i + 1) % 5 == 0:
                print(f"  Games: {i+1}/{games_per_side}, Model1: {result.model1_wins}, Model2: {result.model2_wins}, Draws: {result.draws}")
    else:
        # Parallel execution
        print(f"\nPlaying {num_games} games in parallel ({parallel_games} at a time)...")

        # Create list of games: True = model1 first, False = model2 first
        games_to_play = [True] * games_per_side + [False] * games_per_side
        completed = 0

        with ThreadPoolExecutor(max_workers=parallel_games) as executor:
            futures = {executor.submit(play_single_game, m1_first): m1_first
                      for m1_first in games_to_play}

            for future in as_completed(futures):
                outcome = future.result()
                if outcome == 1:
                    result.model1_wins += 1
                elif outcome == -1:
                    result.model2_wins += 1
                else:
                    result.draws += 1

                completed += 1
                if completed % 5 == 0:
                    print(f"  Games: {completed}/{num_games}, Model1: {result.model1_wins}, Model2: {result.model2_wins}, Draws: {result.draws}")

    return result


def main():
    parser = argparse.ArgumentParser(description='Compare models by having them play against each other')
    parser.add_argument('model1', type=str, help='Path to first model')
    parser.add_argument('model2', type=str, help='Path to second model')
    parser.add_argument('--games', type=int, default=20, help='Number of games to play (default: 20)')
    parser.add_argument('--simulations', type=int, default=200, help='MCTS simulations per move (default: 200)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--verbose', action='store_true', help='Print moves')
    parser.add_argument('--opening-moves', type=int, default=3,
                        help='Number of opening moves with temperature (default: 3)')
    parser.add_argument('--opening-temperature', type=float, default=1.0,
                        help='Temperature for opening moves (default: 1.0)')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of games to run in parallel (default: 1)')

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
        verbose=args.verbose,
        opening_moves=args.opening_moves,
        opening_temperature=args.opening_temperature,
        parallel_games=args.parallel,
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
