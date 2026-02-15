#!/usr/bin/env python3
"""
Model Analysis Tool for Razzle Dazzle.

Analyzes a trained model's performance by:
1. Computing policy accuracy (top-1, top-3) from self-play
2. Measuring value calibration (predicted vs actual outcomes)
3. Reporting game quality metrics (length, diversity)

Usage:
    python scripts/analyze_model.py output/models/iter_500.pt --games 100
    python scripts/analyze_model.py output/models/iter_500.pt --games 50 --simulations 400
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.core.state import GameState
from razzle.core.moves import get_legal_moves
from razzle.ai.network import RazzleNet, NUM_ACTIONS, END_TURN_ACTION
from razzle.ai.mcts import MCTS, MCTSConfig
from razzle.ai.evaluator import BatchedEvaluator
from razzle.training.metrics import (
    compute_policy_metrics,
    compute_value_metrics,
    compute_value_calibration,
    compute_calibration_error,
    format_calibration_table,
    compute_game_diversity,
)


@dataclass
class GameData:
    """Data from a single game for analysis."""
    moves: list[int]
    result: float  # +1 for P0 win, -1 for P1 win, 0 for draw
    states: list[np.ndarray]  # Board tensors at each position
    policies: list[np.ndarray]  # MCTS policies at each position
    players: list[int]  # Current player at each position


def move_to_index(move: int) -> int:
    """Convert game move to network action index."""
    return END_TURN_ACTION if move == -1 else move


def play_analysis_game(
    mcts: MCTS,
    record_data: bool = True,
    verbose: bool = False,
) -> GameData:
    """
    Play a self-play game and record data for analysis.

    Args:
        mcts: MCTS instance with network evaluator
        record_data: Whether to record states and policies
        verbose: Print move-by-move output

    Returns:
        GameData with all recorded information
    """
    state = GameState.new_game()
    move_count = 0
    max_moves = 500

    moves = []
    states = []
    policies = []
    players = []

    while not state.is_terminal() and move_count < max_moves:
        # Record state before move
        if record_data:
            states.append(state.to_tensor())
            players.append(state.current_player)

        # Search
        root = mcts.search(state, add_noise=True)

        # Get MCTS policy from visit counts
        if record_data:
            policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
            total_visits = sum(child.visit_count for child in root.children.values())
            if total_visits > 0:
                for move, child in root.children.items():
                    policy[move_to_index(move)] = child.visit_count / total_visits
            policies.append(policy)

        # Select move (temperature-based early, greedy later)
        if move_count < 30:
            # Temperature-based selection
            visits = np.array([
                root.children[m].visit_count if m in root.children else 0
                for m in range(-1, NUM_ACTIONS - 1)  # -1 to 3135 (excludes END_TURN special)
            ])
            # Use END_TURN separately
            if -1 in root.children:
                visits = visits  # Already included
            if visits.sum() > 0:
                probs = visits / visits.sum()
                move_idx = np.random.choice(len(visits), p=probs)
                best_move = move_idx - 1  # Convert back to move (-1 to 3135)
            else:
                best_move = None
        else:
            # Greedy selection
            best_move = None
            best_visits = -1
            for move, child in root.children.items():
                if child.visit_count > best_visits:
                    best_visits = child.visit_count
                    best_move = move

        if best_move is None:
            break

        moves.append(best_move)
        state.apply_move(best_move)
        move_count += 1

        if verbose:
            print(f"Move {move_count}: Player {1 - state.current_player} plays {best_move}")

    # Determine result
    if state.is_terminal():
        winner = state.get_winner()
        if winner == 0:
            result = 1.0
        elif winner == 1:
            result = -1.0
        else:
            result = 0.0
    else:
        result = 0.0  # Max moves reached

    return GameData(
        moves=moves,
        result=result,
        states=states,
        policies=policies,
        players=players,
    )


def analyze_model(
    model_path: str,
    num_games: int = 100,
    simulations: int = 200,
    device: str = 'cuda',
    verbose: bool = False,
) -> dict:
    """
    Comprehensive model analysis.

    Args:
        model_path: Path to model checkpoint
        num_games: Number of self-play games to generate
        simulations: MCTS simulations per move
        device: Device for inference
        verbose: Print detailed output

    Returns:
        Dictionary of analysis results
    """
    print(f"Loading model: {model_path}")
    network = RazzleNet.load(model_path, device=device)
    network.eval()

    print(f"Model parameters: {network.num_parameters():,}")

    # Create MCTS
    evaluator = BatchedEvaluator(network, device=device)
    config = MCTSConfig(num_simulations=simulations)
    mcts = MCTS(evaluator, config)

    # Generate self-play games
    print(f"\nGenerating {num_games} self-play games ({simulations} sims/move)...")

    all_games: list[GameData] = []
    p0_wins = 0
    p1_wins = 0
    draws = 0

    for i in range(num_games):
        game = play_analysis_game(mcts, record_data=True, verbose=verbose)
        all_games.append(game)

        if game.result > 0:
            p0_wins += 1
        elif game.result < 0:
            p1_wins += 1
        else:
            draws += 1

        if (i + 1) % 10 == 0:
            print(f"  Games: {i + 1}/{num_games}, P0: {p0_wins}, P1: {p1_wins}, Draws: {draws}")

    # Collect all positions for analysis
    all_states = []
    all_policies = []
    all_values = []
    all_legal_masks = []

    for game in all_games:
        for j, (state_tensor, policy, player) in enumerate(
            zip(game.states, game.policies, game.players)
        ):
            all_states.append(state_tensor)
            all_policies.append(policy)

            # Value target from this player's perspective
            if game.result == 0:
                value = 0.0
            elif player == 0:
                value = game.result
            else:
                value = -game.result
            all_values.append(value)

            # Reconstruct legal mask (policy has non-zero where legal)
            legal_mask = (policy > 0).astype(np.float32)
            all_legal_masks.append(legal_mask)

    # Convert to tensors
    states_tensor = torch.from_numpy(np.stack(all_states)).to(device)
    policies_array = np.stack(all_policies)
    values_array = np.array(all_values, dtype=np.float32)
    legal_masks_array = np.stack(all_legal_masks)

    # Get network predictions
    print("\nComputing network predictions...")
    with torch.no_grad():
        pred_logits, pred_values, pred_difficulty = network(states_tensor)
        pred_values = pred_values.squeeze(-1).cpu().numpy()
        pred_logits = pred_logits.cpu().numpy()

    # Compute metrics
    print("\nComputing metrics...")

    # Policy metrics
    policy_metrics = compute_policy_metrics(pred_logits, policies_array, legal_masks_array)

    # Value metrics
    value_metrics = compute_value_metrics(pred_values)

    # Value calibration
    calibration = compute_value_calibration(pred_values, values_array, num_bins=10)
    calibration_error = compute_calibration_error(calibration)

    # Game diversity
    move_sequences = [game.moves for game in all_games]
    diversity_metrics = compute_game_diversity(move_sequences)

    # Compile results
    results = {
        # Game statistics
        'num_games': num_games,
        'p0_wins': p0_wins,
        'p1_wins': p1_wins,
        'draws': draws,
        'p0_win_rate': p0_wins / num_games,
        'total_positions': len(all_states),

        # Policy metrics
        'policy_top1_accuracy': policy_metrics.top1_accuracy,
        'policy_top3_accuracy': policy_metrics.top3_accuracy,
        'policy_entropy': policy_metrics.entropy,
        'policy_legal_mass': policy_metrics.legal_mass,

        # Value metrics
        'value_mean': value_metrics.mean,
        'value_std': value_metrics.std,
        'value_extremity': value_metrics.extremity,
        'value_calibration_error': calibration_error,
        'calibration': calibration,

        # Game diversity
        **diversity_metrics,
    }

    return results


def print_results(results: dict):
    """Pretty print analysis results."""
    print("\n" + "=" * 60)
    print("MODEL ANALYSIS RESULTS")
    print("=" * 60)

    print("\n--- Game Statistics ---")
    print(f"Games played: {results['num_games']}")
    print(f"Player 0 wins: {results['p0_wins']} ({results['p0_win_rate']*100:.1f}%)")
    print(f"Player 1 wins: {results['p1_wins']} ({(results['p1_wins']/results['num_games'])*100:.1f}%)")
    print(f"Draws: {results['draws']}")
    print(f"Total positions analyzed: {results['total_positions']}")

    print("\n--- Policy Quality ---")
    print(f"Top-1 accuracy: {results['policy_top1_accuracy']*100:.1f}%")
    print(f"Top-3 accuracy: {results['policy_top3_accuracy']*100:.1f}%")
    print(f"Average entropy: {results['policy_entropy']:.3f}")
    print(f"Legal move mass: {results['policy_legal_mass']*100:.1f}%")

    print("\n--- Value Predictions ---")
    print(f"Mean value: {results['value_mean']:+.3f}")
    print(f"Std value: {results['value_std']:.3f}")
    print(f"Extremity (avg |value|): {results['value_extremity']:.3f}")
    print(f"Calibration error (MAE): {results['value_calibration_error']:.3f}")

    print("\n--- Game Diversity ---")
    print(f"Average game length: {results['avg_game_length']:.1f} moves")
    print(f"Game length std: {results['game_length_std']:.1f}")
    print(f"Unique openings (5-move): {results['unique_openings']}")
    print(f"Opening diversity: {results['opening_diversity']*100:.1f}%")

    print("\n--- Value Calibration Table ---")
    print(format_calibration_table(results['calibration']))

    print("\n" + "=" * 60)

    # Interpretation
    print("\n--- Interpretation ---")

    # Policy
    if results['policy_top1_accuracy'] > 0.6:
        print("  Policy: GOOD - Network accurately predicts MCTS moves")
    elif results['policy_top1_accuracy'] > 0.4:
        print("  Policy: MODERATE - Some agreement with MCTS")
    else:
        print("  Policy: WEAK - Network and MCTS disagree significantly")

    # Value
    if results['value_calibration_error'] < 0.3:
        print("  Value: WELL-CALIBRATED - Predictions match outcomes")
    elif results['value_calibration_error'] < 0.5:
        print("  Value: MODERATELY CALIBRATED - Some prediction bias")
    else:
        print("  Value: POORLY CALIBRATED - Predictions don't match outcomes")

    if results['value_extremity'] > 0.8:
        print("  Value: OVERCONFIDENT - Predictions too extreme")
    elif results['value_extremity'] < 0.3:
        print("  Value: UNDERCONFIDENT - Predictions too conservative")

    # Legal moves
    if results['policy_legal_mass'] < 0.95:
        print(f"  Warning: {(1-results['policy_legal_mass'])*100:.1f}% probability on illegal moves")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze a trained Razzle Dazzle model'
    )
    parser.add_argument('model', type=str, help='Path to model checkpoint')
    parser.add_argument('--games', type=int, default=100,
                        help='Number of self-play games (default: 100)')
    parser.add_argument('--simulations', type=int, default=200,
                        help='MCTS simulations per move (default: 200)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print moves during games')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to JSON file')

    args = parser.parse_args()

    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Run analysis
    results = analyze_model(
        args.model,
        num_games=args.games,
        simulations=args.simulations,
        device=args.device,
        verbose=args.verbose,
    )

    # Print results
    print_results(results)

    # Save to file if requested
    if args.output:
        import json
        output_path = Path(args.output)

        # Convert non-serializable items
        json_results = {k: v for k, v in results.items() if k != 'calibration'}
        json_results['calibration'] = [
            {
                'predicted': b.predicted_mean,
                'actual': b.actual_mean,
                'count': b.count,
                'error': b.error,
            }
            for b in results['calibration']
        ]

        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
