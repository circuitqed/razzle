#!/usr/bin/env python3
"""
Analyze policy accuracy - how well does raw NN policy match MCTS results?

Metrics:
- Top-1 accuracy: Does raw policy's best move match MCTS's best move?
- Top-3 accuracy: Is MCTS's best move in raw policy's top 3?
- Average rank: What rank does MCTS's best move have in raw policy?
- KL divergence: How different are the distributions?
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.core.state import GameState
from razzle.core.moves import get_legal_moves
from razzle.ai.network import RazzleNet
from razzle.ai.mcts import MCTS, MCTSConfig
from razzle.ai.evaluator import BatchedEvaluator


@dataclass
class PolicyAccuracyMetrics:
    """Metrics for a single position."""
    top1_match: bool = False       # Raw policy top move == MCTS top move
    top3_match: bool = False       # MCTS top move in raw policy top 3
    mcts_move_rank: int = 0        # Rank of MCTS move in raw policy (0-indexed)
    kl_divergence: float = 0.0     # KL(MCTS || raw)
    raw_entropy: float = 0.0       # Entropy of raw policy
    mcts_entropy: float = 0.0      # Entropy of MCTS policy
    game_phase: str = "opening"    # opening/midgame/endgame


@dataclass
class AggregateMetrics:
    """Aggregated metrics across positions."""
    top1_accuracy: float = 0.0
    top3_accuracy: float = 0.0
    avg_rank: float = 0.0
    avg_kl: float = 0.0
    avg_raw_entropy: float = 0.0
    avg_mcts_entropy: float = 0.0
    # By phase
    opening_top1: float = 0.0
    midgame_top1: float = 0.0
    endgame_top1: float = 0.0
    num_positions: int = 0


def get_game_phase(move_number: int) -> str:
    """Classify position by game phase."""
    if move_number < 10:
        return "opening"
    elif move_number < 30:
        return "midgame"
    else:
        return "endgame"


def analyze_position(
    state: GameState,
    evaluator: BatchedEvaluator,
    mcts: MCTS,
    move_number: int
) -> PolicyAccuracyMetrics:
    """Analyze policy accuracy for a single position."""
    legal_moves = get_legal_moves(state)
    if len(legal_moves) <= 1:
        return None  # Skip trivial positions

    # Get raw policy from network
    policy, value = evaluator.evaluate(state)

    # Run MCTS search
    root = mcts.search(state, add_noise=False)

    # Get MCTS policy from visit counts
    total_visits = sum(child.visit_count for child in root.children.values())
    if total_visits == 0:
        return None

    mcts_policy = np.zeros_like(policy)
    for move, child in root.children.items():
        mcts_policy[move] = child.visit_count / total_visits

    # Mask to legal moves only
    legal_mask = np.zeros(len(policy), dtype=bool)
    for move in legal_moves:
        legal_mask[move] = True

    # Get raw policy rankings (among legal moves)
    raw_probs = policy.copy()
    raw_probs[~legal_mask] = -np.inf
    raw_ranking = np.argsort(-raw_probs)  # Descending order

    # Top move from each
    raw_top_move = raw_ranking[0]
    mcts_top_move = max(root.children.items(), key=lambda x: x[1].visit_count)[0]

    # Compute metrics
    metrics = PolicyAccuracyMetrics()
    metrics.top1_match = (raw_top_move == mcts_top_move)
    metrics.top3_match = mcts_top_move in raw_ranking[:3]

    # Rank of MCTS move in raw policy
    for rank, move in enumerate(raw_ranking):
        if move == mcts_top_move:
            metrics.mcts_move_rank = rank
            break

    # KL divergence (MCTS || raw) - only over legal moves
    eps = 1e-8
    raw_legal = np.clip(policy[legal_mask], eps, 1.0)
    raw_legal = raw_legal / raw_legal.sum()  # Renormalize
    mcts_legal = np.clip(mcts_policy[legal_mask], eps, 1.0)
    mcts_legal = mcts_legal / mcts_legal.sum()

    metrics.kl_divergence = np.sum(mcts_legal * np.log(mcts_legal / raw_legal))

    # Entropy
    metrics.raw_entropy = -np.sum(raw_legal * np.log(raw_legal))
    metrics.mcts_entropy = -np.sum(mcts_legal * np.log(mcts_legal))

    metrics.game_phase = get_game_phase(move_number)

    return metrics


def play_and_analyze(
    evaluator: BatchedEvaluator,
    mcts: MCTS,
    num_games: int = 10,
    verbose: bool = False
) -> list[PolicyAccuracyMetrics]:
    """Play games and collect policy accuracy metrics."""
    all_metrics = []

    for game_idx in range(num_games):
        state = GameState.new_game()
        move_number = 0

        while not state.is_terminal() and move_number < 100:
            # Analyze this position
            metrics = analyze_position(state, evaluator, mcts, move_number)
            if metrics is not None:
                all_metrics.append(metrics)

            # Make a move (using MCTS)
            root = mcts.search(state, add_noise=False)
            best_move = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
            state.apply_move(best_move)
            move_number += 1

        if verbose:
            print(f"  Game {game_idx + 1}/{num_games}: {move_number} moves, {len(all_metrics)} positions analyzed", flush=True)

    return all_metrics


def aggregate_metrics(metrics_list: list[PolicyAccuracyMetrics]) -> AggregateMetrics:
    """Aggregate metrics across positions."""
    if not metrics_list:
        return AggregateMetrics()

    agg = AggregateMetrics()
    agg.num_positions = len(metrics_list)

    agg.top1_accuracy = sum(m.top1_match for m in metrics_list) / len(metrics_list)
    agg.top3_accuracy = sum(m.top3_match for m in metrics_list) / len(metrics_list)
    agg.avg_rank = sum(m.mcts_move_rank for m in metrics_list) / len(metrics_list)
    agg.avg_kl = sum(m.kl_divergence for m in metrics_list) / len(metrics_list)
    agg.avg_raw_entropy = sum(m.raw_entropy for m in metrics_list) / len(metrics_list)
    agg.avg_mcts_entropy = sum(m.mcts_entropy for m in metrics_list) / len(metrics_list)

    # By phase
    opening = [m for m in metrics_list if m.game_phase == "opening"]
    midgame = [m for m in metrics_list if m.game_phase == "midgame"]
    endgame = [m for m in metrics_list if m.game_phase == "endgame"]

    if opening:
        agg.opening_top1 = sum(m.top1_match for m in opening) / len(opening)
    if midgame:
        agg.midgame_top1 = sum(m.top1_match for m in midgame) / len(midgame)
    if endgame:
        agg.endgame_top1 = sum(m.top1_match for m in endgame) / len(endgame)

    return agg


def analyze_model(
    model_path: str,
    device: str = 'cuda',
    num_games: int = 5,
    simulations: int = 100,
    verbose: bool = False
) -> AggregateMetrics:
    """Analyze a single model."""
    # Load model
    net = RazzleNet.load(model_path, device=device)
    net.eval()

    # Create evaluator and MCTS
    evaluator = BatchedEvaluator(net, device=device)
    config = MCTSConfig(num_simulations=simulations)
    mcts = MCTS(evaluator, config)

    # Play and analyze
    metrics_list = play_and_analyze(evaluator, mcts, num_games=num_games, verbose=verbose)

    return aggregate_metrics(metrics_list)


def main():
    parser = argparse.ArgumentParser(description='Analyze policy accuracy across training')
    parser.add_argument('--models-dir', type=str, default='output/models', help='Directory with model checkpoints')
    parser.add_argument('--games', type=int, default=5, help='Games per model')
    parser.add_argument('--simulations', type=int, default=100, help='MCTS simulations per move')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--step', type=int, default=50, help='Analyze every N iterations')
    parser.add_argument('--output', type=str, default='plots/policy_accuracy.png', help='Output plot path')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Unbuffered output
    sys.stdout.reconfigure(line_buffering=True)

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU", flush=True)
        args.device = 'cpu'

    # Find model checkpoints
    models_dir = Path(args.models_dir)
    model_files = sorted(models_dir.glob('iter_*.pt'))

    if not model_files:
        print(f"No models found in {models_dir}", flush=True)
        return

    print(f"Found {len(model_files)} models", flush=True)

    # Select models to analyze (every N iterations)
    iterations = []
    results = []

    for model_path in model_files:
        # Extract iteration number
        name = model_path.stem  # e.g., "iter_010"
        try:
            iter_num = int(name.split('_')[1])
        except (IndexError, ValueError):
            continue

        if iter_num % args.step != 0 and model_path != model_files[-1]:
            continue  # Skip unless it's on the step or the last model

        print(f"\nAnalyzing {model_path.name}...", flush=True)

        try:
            metrics = analyze_model(
                str(model_path),
                device=args.device,
                num_games=args.games,
                simulations=args.simulations,
                verbose=args.verbose
            )

            iterations.append(iter_num)
            results.append(metrics)

            print(f"  Top-1: {metrics.top1_accuracy:.1%}, Top-3: {metrics.top3_accuracy:.1%}, "
                  f"Avg Rank: {metrics.avg_rank:.1f}, KL: {metrics.avg_kl:.3f}", flush=True)
            print(f"  By phase - Opening: {metrics.opening_top1:.1%}, "
                  f"Midgame: {metrics.midgame_top1:.1%}, Endgame: {metrics.endgame_top1:.1%}", flush=True)

        except Exception as e:
            print(f"  Error: {e}", flush=True)
            continue

    if len(iterations) < 2:
        print("Not enough models analyzed for plotting", flush=True)
        return

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Top-1 and Top-3 accuracy
    ax1 = axes[0, 0]
    ax1.plot(iterations, [r.top1_accuracy for r in results], 'b-o', label='Top-1 Accuracy')
    ax1.plot(iterations, [r.top3_accuracy for r in results], 'g-o', label='Top-3 Accuracy')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Policy Accuracy vs MCTS')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Plot 2: Average rank of MCTS move
    ax2 = axes[0, 1]
    ax2.plot(iterations, [r.avg_rank for r in results], 'r-o')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Average Rank')
    ax2.set_title('Rank of MCTS Top Move in Raw Policy')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()  # Lower rank is better

    # Plot 3: KL divergence
    ax3 = axes[1, 0]
    ax3.plot(iterations, [r.avg_kl for r in results], 'm-o')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('KL Divergence')
    ax3.set_title('KL Divergence (MCTS || Raw Policy)')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Accuracy by game phase
    ax4 = axes[1, 1]
    ax4.plot(iterations, [r.opening_top1 for r in results], 'b-o', label='Opening')
    ax4.plot(iterations, [r.midgame_top1 for r in results], 'g-o', label='Midgame')
    ax4.plot(iterations, [r.endgame_top1 for r in results], 'r-o', label='Endgame')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Top-1 Accuracy')
    ax4.set_title('Policy Accuracy by Game Phase')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)

    plt.tight_layout()

    # Save plot
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}", flush=True)

    # Print summary
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)

    if results:
        first = results[0]
        last = results[-1]
        print(f"First model (iter {iterations[0]}):", flush=True)
        print(f"  Top-1: {first.top1_accuracy:.1%}, Top-3: {first.top3_accuracy:.1%}", flush=True)
        print(f"Last model (iter {iterations[-1]}):", flush=True)
        print(f"  Top-1: {last.top1_accuracy:.1%}, Top-3: {last.top3_accuracy:.1%}", flush=True)
        print(f"  Change: {(last.top1_accuracy - first.top1_accuracy):+.1%} Top-1, "
              f"{(last.top3_accuracy - first.top3_accuracy):+.1%} Top-3", flush=True)


if __name__ == '__main__':
    main()
