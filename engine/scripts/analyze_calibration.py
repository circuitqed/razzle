#!/usr/bin/env python3
"""
Value Calibration Analysis for Razzle Dazzle.

Deep-dive analysis of value head calibration:
1. Calibration by game phase (opening, middle, endgame)
2. Calibration by position features (material, ball progress)
3. Overconfidence/underconfidence diagnosis
4. Reliability diagrams

Usage:
    python scripts/analyze_calibration.py output/models/iter_500.pt --games 100
    python scripts/analyze_calibration.py output/models/iter_500.pt --games 200 --bins 20
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
from razzle.core.bitboard import ROWS, COLS, iter_bits, popcount
from razzle.core.moves import get_legal_moves
from razzle.ai.network import RazzleNet, NUM_ACTIONS, END_TURN_ACTION
from razzle.ai.mcts import MCTS, MCTSConfig
from razzle.ai.evaluator import BatchedEvaluator
from razzle.training.metrics import (
    compute_value_calibration,
    compute_calibration_error,
    CalibrationBucket,
)


@dataclass
class PositionData:
    """Data for a single position."""
    state_tensor: np.ndarray
    player: int
    ply: int
    game_result: float
    value_target: float  # From player's perspective
    # Position features
    ball_row: int  # 0-7 for current player's ball
    opp_ball_row: int
    pieces_count: int
    opp_pieces_count: int


def extract_position_features(state: GameState) -> dict:
    """Extract features from a game state for analysis."""
    p = state.current_player

    # Ball positions (row 0-7)
    my_ball_sq = list(iter_bits(state.balls[p]))[0] if state.balls[p] else 0
    opp_ball_sq = list(iter_bits(state.balls[1 - p]))[0] if state.balls[1 - p] else 0

    my_ball_row = my_ball_sq // COLS
    opp_ball_row = opp_ball_sq // COLS

    # Piece counts
    my_pieces = popcount(state.pieces[p])
    opp_pieces = popcount(state.pieces[1 - p])

    return {
        'ball_row': my_ball_row if p == 0 else (ROWS - 1 - my_ball_row),
        'opp_ball_row': opp_ball_row if p == 1 else (ROWS - 1 - opp_ball_row),
        'pieces_count': my_pieces,
        'opp_pieces_count': opp_pieces,
    }


def play_calibration_games(
    network: RazzleNet,
    num_games: int,
    simulations: int,
    device: str,
) -> list[PositionData]:
    """
    Play self-play games and collect position data for calibration analysis.
    """
    evaluator = BatchedEvaluator(network, device=device)
    config = MCTSConfig(num_simulations=simulations)
    mcts = MCTS(evaluator, config)

    all_positions: list[PositionData] = []

    for game_idx in range(num_games):
        state = GameState.new_game()
        move_count = 0
        max_moves = 500

        # Collect positions during game
        game_positions: list[PositionData] = []

        while not state.is_terminal() and move_count < max_moves:
            # Extract features
            features = extract_position_features(state)

            # Record position
            game_positions.append(PositionData(
                state_tensor=state.to_tensor(),
                player=state.current_player,
                ply=state.ply,
                game_result=0.0,  # Will be filled after game ends
                value_target=0.0,
                ball_row=features['ball_row'],
                opp_ball_row=features['opp_ball_row'],
                pieces_count=features['pieces_count'],
                opp_pieces_count=features['opp_pieces_count'],
            ))

            # MCTS search and move selection
            root = mcts.search(state, add_noise=True)

            # Greedy selection (temperature 0 for cleaner calibration)
            best_move = None
            best_visits = -1
            for move, child in root.children.items():
                if child.visit_count > best_visits:
                    best_visits = child.visit_count
                    best_move = move

            if best_move is None:
                break

            state.apply_move(best_move)
            move_count += 1

        # Determine game result
        if state.is_terminal():
            winner = state.get_winner()
            if winner == 0:
                game_result = 1.0
            elif winner == 1:
                game_result = -1.0
            else:
                game_result = 0.0
        else:
            game_result = 0.0

        # Fill in game results and value targets
        for pos in game_positions:
            pos.game_result = game_result
            if game_result == 0:
                pos.value_target = 0.0
            elif pos.player == 0:
                pos.value_target = game_result
            else:
                pos.value_target = -game_result

        all_positions.extend(game_positions)

        if (game_idx + 1) % 10 == 0:
            print(f"  Games: {game_idx + 1}/{num_games}")

    return all_positions


def analyze_calibration_by_phase(
    positions: list[PositionData],
    pred_values: np.ndarray,
) -> dict:
    """
    Analyze calibration by game phase (opening, middle, endgame).
    """
    # Define phases by ply
    opening_mask = np.array([p.ply < 20 for p in positions])
    middle_mask = np.array([(p.ply >= 20) and (p.ply < 50) for p in positions])
    endgame_mask = np.array([p.ply >= 50 for p in positions])

    targets = np.array([p.value_target for p in positions])

    results = {}

    for name, mask in [
        ('opening', opening_mask),
        ('middle', middle_mask),
        ('endgame', endgame_mask),
    ]:
        if np.sum(mask) > 10:
            cal = compute_value_calibration(pred_values[mask], targets[mask], num_bins=5)
            results[name] = {
                'count': int(np.sum(mask)),
                'calibration_error': compute_calibration_error(cal),
                'mean_pred': float(np.mean(pred_values[mask])),
                'mean_actual': float(np.mean(targets[mask])),
                'pred_std': float(np.std(pred_values[mask])),
            }
        else:
            results[name] = None

    return results


def analyze_calibration_by_ball_progress(
    positions: list[PositionData],
    pred_values: np.ndarray,
) -> dict:
    """
    Analyze calibration by ball progress (how far the ball has advanced).
    """
    ball_rows = np.array([p.ball_row for p in positions])
    targets = np.array([p.value_target for p in positions])

    results = {}

    # Group by ball position
    for row_range, name in [
        ((0, 2), 'ball_back'),
        ((2, 5), 'ball_middle'),
        ((5, 8), 'ball_advanced'),
    ]:
        mask = (ball_rows >= row_range[0]) & (ball_rows < row_range[1])
        if np.sum(mask) > 10:
            cal = compute_value_calibration(pred_values[mask], targets[mask], num_bins=5)
            results[name] = {
                'count': int(np.sum(mask)),
                'calibration_error': compute_calibration_error(cal),
                'mean_pred': float(np.mean(pred_values[mask])),
                'mean_actual': float(np.mean(targets[mask])),
            }
        else:
            results[name] = None

    return results


def diagnose_calibration_issues(
    pred_values: np.ndarray,
    targets: np.ndarray,
) -> dict:
    """
    Diagnose calibration issues (overconfidence, underconfidence, bias).
    """
    # Overconfidence: predictions more extreme than reality
    # Underconfidence: predictions less extreme than reality

    pred_extremity = np.mean(np.abs(pred_values))
    target_extremity = np.mean(np.abs(targets))

    # Bias: systematic over/under prediction
    bias = np.mean(pred_values - targets)

    # Win/loss accuracy: when network is confident, is it right?
    confident_positive = pred_values > 0.5
    confident_negative = pred_values < -0.5

    if np.sum(confident_positive) > 0:
        positive_accuracy = np.mean(targets[confident_positive] > 0)
    else:
        positive_accuracy = None

    if np.sum(confident_negative) > 0:
        negative_accuracy = np.mean(targets[confident_negative] < 0)
    else:
        negative_accuracy = None

    # Diagnose
    issues = []

    if pred_extremity > target_extremity * 1.2:
        issues.append("OVERCONFIDENT: Predictions too extreme")
    elif pred_extremity < target_extremity * 0.8:
        issues.append("UNDERCONFIDENT: Predictions too conservative")

    if abs(bias) > 0.1:
        if bias > 0:
            issues.append(f"POSITIVE BIAS: Predictions skewed positive by {bias:.2f}")
        else:
            issues.append(f"NEGATIVE BIAS: Predictions skewed negative by {abs(bias):.2f}")

    return {
        'pred_extremity': float(pred_extremity),
        'target_extremity': float(target_extremity),
        'bias': float(bias),
        'positive_confidence_accuracy': positive_accuracy,
        'negative_confidence_accuracy': negative_accuracy,
        'issues': issues,
    }


def print_reliability_diagram(
    pred_values: np.ndarray,
    targets: np.ndarray,
    num_bins: int = 10,
):
    """
    Print ASCII reliability diagram.
    """
    calibration = compute_value_calibration(pred_values, targets, num_bins)

    print("\nReliability Diagram (Predicted vs Actual)")
    print("=" * 60)
    print("Perfect calibration would show '|' at each prediction level")
    print()

    # Create diagram
    for bucket in calibration:
        pred = bucket.predicted_mean
        actual = bucket.actual_mean

        # Scale to 0-50 for display
        pred_pos = int((pred + 1) / 2 * 50)
        actual_pos = int((actual + 1) / 2 * 50)

        line = [' '] * 51
        line[pred_pos] = '|'  # Predicted
        line[actual_pos] = '*'  # Actual

        # If they overlap
        if pred_pos == actual_pos:
            line[pred_pos] = 'O'

        print(f"[{pred:+.2f}] {''.join(line)} n={bucket.count}")

    print()
    print("Legend: | = predicted, * = actual, O = match")


def main():
    parser = argparse.ArgumentParser(
        description='Deep analysis of value calibration for Razzle Dazzle models'
    )
    parser.add_argument('model', type=str, help='Path to model checkpoint')
    parser.add_argument('--games', type=int, default=100,
                        help='Number of self-play games (default: 100)')
    parser.add_argument('--simulations', type=int, default=200,
                        help='MCTS simulations per move (default: 200)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--bins', type=int, default=10,
                        help='Number of calibration bins (default: 10)')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to JSON file')

    args = parser.parse_args()

    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Load model
    print(f"Loading model: {args.model}")
    network = RazzleNet.load(args.model, device=args.device)
    network.eval()

    # Generate games
    print(f"\nGenerating {args.games} self-play games...")
    positions = play_calibration_games(
        network, args.games, args.simulations, args.device
    )
    print(f"Collected {len(positions)} positions")

    # Get predictions
    print("\nComputing predictions...")
    states = torch.from_numpy(
        np.stack([p.state_tensor for p in positions])
    ).to(args.device)

    with torch.no_grad():
        _, pred_values, _ = network(states)
        pred_values = pred_values.squeeze(-1).cpu().numpy()

    targets = np.array([p.value_target for p in positions])

    # Overall calibration
    print("\n" + "=" * 60)
    print("VALUE CALIBRATION ANALYSIS")
    print("=" * 60)

    overall_cal = compute_value_calibration(pred_values, targets, args.bins)
    overall_error = compute_calibration_error(overall_cal)

    print(f"\nOverall Calibration Error: {overall_error:.4f}")

    # Detailed calibration table
    print("\n--- Calibration by Prediction Bucket ---")
    print(f"{'Predicted':>12} {'Actual':>12} {'Error':>10} {'Count':>8}")
    print("-" * 50)
    for bucket in overall_cal:
        print(f"{bucket.predicted_mean:>+12.3f} {bucket.actual_mean:>+12.3f} "
              f"{bucket.error:>10.3f} {bucket.count:>8}")

    # Phase analysis
    print("\n--- Calibration by Game Phase ---")
    phase_results = analyze_calibration_by_phase(positions, pred_values)
    for phase, data in phase_results.items():
        if data:
            print(f"  {phase.upper():10} n={data['count']:5}  "
                  f"error={data['calibration_error']:.3f}  "
                  f"pred={data['mean_pred']:+.3f}  actual={data['mean_actual']:+.3f}")

    # Ball progress analysis
    print("\n--- Calibration by Ball Progress ---")
    ball_results = analyze_calibration_by_ball_progress(positions, pred_values)
    for pos_name, data in ball_results.items():
        if data:
            print(f"  {pos_name:15} n={data['count']:5}  "
                  f"error={data['calibration_error']:.3f}  "
                  f"pred={data['mean_pred']:+.3f}  actual={data['mean_actual']:+.3f}")

    # Diagnosis
    print("\n--- Calibration Diagnosis ---")
    diagnosis = diagnose_calibration_issues(pred_values, targets)

    print(f"  Prediction extremity: {diagnosis['pred_extremity']:.3f}")
    print(f"  Target extremity:     {diagnosis['target_extremity']:.3f}")
    print(f"  Bias:                 {diagnosis['bias']:+.3f}")

    if diagnosis['positive_confidence_accuracy'] is not None:
        print(f"  Confident positive accuracy: {diagnosis['positive_confidence_accuracy']*100:.1f}%")
    if diagnosis['negative_confidence_accuracy'] is not None:
        print(f"  Confident negative accuracy: {diagnosis['negative_confidence_accuracy']*100:.1f}%")

    if diagnosis['issues']:
        print("\n  ISSUES DETECTED:")
        for issue in diagnosis['issues']:
            print(f"    - {issue}")
    else:
        print("\n  No major calibration issues detected.")

    # Reliability diagram
    print_reliability_diagram(pred_values, targets, min(args.bins, 10))

    # Save results
    if args.output:
        import json
        output_path = Path(args.output)

        json_results = {
            'num_games': args.games,
            'num_positions': len(positions),
            'overall_calibration_error': float(overall_error),
            'overall_calibration': [
                {
                    'predicted': b.predicted_mean,
                    'actual': b.actual_mean,
                    'count': b.count,
                    'error': b.error,
                }
                for b in overall_cal
            ],
            'phase_analysis': phase_results,
            'ball_progress_analysis': ball_results,
            'diagnosis': {
                'pred_extremity': diagnosis['pred_extremity'],
                'target_extremity': diagnosis['target_extremity'],
                'bias': diagnosis['bias'],
                'issues': diagnosis['issues'],
            },
        }

        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
