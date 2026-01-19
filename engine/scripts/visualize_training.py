#!/usr/bin/env python3
"""
Visualize training progress for Razzle Dazzle.

Shows:
- Loss curves (policy, value, total)
- Win rates over iterations
- Game length trends
- Training throughput
- Model strength evaluation

Can read from either:
- training_log.json (new format, preferred)
- Legacy pickle game files
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not installed. Install with: pip install matplotlib")


def load_training_log(output_dir: Path) -> dict:
    """Load training data from training_log.json (new format)."""
    log_path = output_dir / "training_log.json"
    if not log_path.exists():
        return None

    with open(log_path, 'r') as f:
        return json.load(f)


def load_legacy_history(output_dir: Path) -> dict:
    """Load training data from pickle files (legacy format)."""
    history = {
        'iterations': [],
        'games': [],
        'p1_wins': [],
        'p2_wins': [],
        'draws': [],
        'avg_game_length': [],
        'training_examples': [],
    }

    # Find all game directories
    game_dirs = sorted(output_dir.glob("games_iter_*"))

    for game_dir in game_dirs:
        iter_num = int(game_dir.name.split('_')[-1])

        # Load games
        games = []
        for game_file in sorted(game_dir.glob("*.pkl")):
            with open(game_file, 'rb') as f:
                games.append(pickle.load(f))

        if not games:
            continue

        # Compute statistics
        p1_wins = sum(1 for g in games if g.result == 1.0)
        p2_wins = sum(1 for g in games if g.result == -1.0)
        draws = sum(1 for g in games if g.result == 0.0)
        avg_length = sum(len(g.moves) for g in games) / len(games)
        total_examples = sum(len(g.states) for g in games)

        history['iterations'].append(iter_num)
        history['games'].append(len(games))
        history['p1_wins'].append(p1_wins)
        history['p2_wins'].append(p2_wins)
        history['draws'].append(draws)
        history['avg_game_length'].append(avg_length)
        history['training_examples'].append(total_examples)

    return history


def convert_log_to_history(log: dict) -> dict:
    """Convert new JSON log format to history dict for plotting."""
    history = {
        'iterations': [],
        'games': [],
        'p1_wins': [],
        'p2_wins': [],
        'draws': [],
        'avg_game_length': [],
        'min_game_length': [],
        'max_game_length': [],
        'std_game_length': [],
        'training_examples': [],
        'loss': [],
        'policy_loss': [],
        'value_loss': [],
        'selfplay_time': [],
        'training_time': [],
        'total_time': [],
        'win_rate_vs_random': [],
        'elo': [],
    }

    for it in log.get('iterations', []):
        history['iterations'].append(it['iteration'])
        history['games'].append(it['num_games'])
        history['p1_wins'].append(it['p1_wins'])
        history['p2_wins'].append(it['p2_wins'])
        history['draws'].append(it['draws'])
        history['avg_game_length'].append(it['avg_game_length'])
        history['min_game_length'].append(it.get('min_game_length', 0))
        history['max_game_length'].append(it.get('max_game_length', 0))
        history['std_game_length'].append(it.get('std_game_length', 0))
        history['training_examples'].append(it['training_examples'])
        history['loss'].append(it.get('final_loss', 0))
        history['policy_loss'].append(it.get('final_policy_loss', 0))
        history['value_loss'].append(it.get('final_value_loss', 0))
        history['selfplay_time'].append(it.get('selfplay_time_sec', 0))
        history['training_time'].append(it.get('training_time_sec', 0))
        history['total_time'].append(it.get('total_time_sec', 0))
        history['win_rate_vs_random'].append(it.get('win_rate_vs_random'))
        history['elo'].append(it.get('elo_rating'))

    return history


def evaluate_models(output_dir: Path, num_games: int = 10) -> list:
    """Evaluate models against random to measure improvement."""
    from razzle.ai.network import RazzleNet
    from razzle.ai.mcts import MCTS, MCTSConfig
    from razzle.ai.evaluator import BatchedEvaluator, DummyEvaluator
    from razzle.core.state import GameState

    model_files = sorted(output_dir.glob("model_iter_*.pt"))
    if not model_files:
        return []

    results = []
    random_eval = DummyEvaluator()

    for model_file in model_files:
        iter_num = int(model_file.stem.split('_')[-1])
        print(f"Evaluating model iter {iter_num}...")

        network = RazzleNet.load(model_file, device='cpu')
        model_eval = BatchedEvaluator(network, device='cpu')

        wins = 0
        losses = 0
        draws = 0
        game_lengths = []

        for game_num in range(num_games):
            state = GameState.new_game()
            model_plays_first = game_num % 2 == 0
            move_count = 0

            config = MCTSConfig(num_simulations=50, temperature=0.0, batch_size=8)

            while not state.is_terminal() and move_count < 200:
                player = state.current_player
                is_model_turn = (player == 0) == model_plays_first

                if is_model_turn:
                    mcts = MCTS(model_eval, config)
                else:
                    mcts = MCTS(random_eval, config)

                root = mcts.search_batched(state, add_noise=False)
                move = mcts.select_move(root)
                state.apply_move(move)
                move_count += 1

            game_lengths.append(move_count)
            winner = state.get_winner()

            if winner is None:
                draws += 1
            else:
                model_won = (winner == 0) == model_plays_first
                if model_won:
                    wins += 1
                else:
                    losses += 1

        win_rate = wins / num_games if num_games > 0 else 0
        avg_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0

        results.append({
            'iteration': iter_num,
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': win_rate,
            'avg_game_length': avg_length
        })
        print(f"  Win rate vs random: {win_rate:.1%} ({wins}W/{losses}L/{draws}D), avg length: {avg_length:.0f}")

    return results


def compute_elo(output_dir: Path, games_per_match: int = 6) -> list:
    """Compute ELO ratings by having models play against each other."""
    from razzle.ai.network import RazzleNet
    from razzle.ai.mcts import MCTS, MCTSConfig
    from razzle.ai.evaluator import BatchedEvaluator
    from razzle.core.state import GameState

    model_files = sorted(output_dir.glob("model_iter_*.pt"))
    if len(model_files) < 2:
        return []

    # Initialize ELO ratings at 1000
    elo = {int(f.stem.split('_')[-1]): 1000.0 for f in model_files}
    K = 32  # ELO K-factor

    print("Computing ELO ratings...")

    # Each model plays against adjacent models
    for i in range(len(model_files) - 1):
        iter_a = int(model_files[i].stem.split('_')[-1])
        iter_b = int(model_files[i + 1].stem.split('_')[-1])

        print(f"  Match: iter {iter_a} vs iter {iter_b}")

        net_a = RazzleNet.load(model_files[i], device='cpu')
        net_b = RazzleNet.load(model_files[i + 1], device='cpu')
        eval_a = BatchedEvaluator(net_a, device='cpu')
        eval_b = BatchedEvaluator(net_b, device='cpu')

        wins_a = 0
        wins_b = 0

        config = MCTSConfig(num_simulations=50, temperature=0.0, batch_size=8)

        for game_num in range(games_per_match):
            state = GameState.new_game()
            a_plays_first = game_num % 2 == 0
            move_count = 0

            while not state.is_terminal() and move_count < 150:
                player = state.current_player
                is_a_turn = (player == 0) == a_plays_first

                if is_a_turn:
                    mcts = MCTS(eval_a, config)
                else:
                    mcts = MCTS(eval_b, config)

                root = mcts.search_batched(state, add_noise=False)
                move = mcts.select_move(root)
                state.apply_move(move)
                move_count += 1

            winner = state.get_winner()
            if winner is not None:
                a_won = (winner == 0) == a_plays_first
                if a_won:
                    wins_a += 1
                else:
                    wins_b += 1

        # Update ELO based on results
        total_games = wins_a + wins_b
        if total_games > 0:
            score_a = wins_a / total_games
            expected_a = 1 / (1 + 10 ** ((elo[iter_b] - elo[iter_a]) / 400))
            elo[iter_a] += K * (score_a - expected_a)
            elo[iter_b] += K * ((1 - score_a) - (1 - expected_a))

        print(f"    Results: {wins_a}-{wins_b}, ELO: {iter_a}={elo[iter_a]:.0f}, {iter_b}={elo[iter_b]:.0f}")

    return [{'iteration': k, 'elo': v} for k, v in sorted(elo.items())]


def plot_training_progress(history: dict, eval_results: list = None, elo_results: list = None, save_path: Path = None):
    """Create comprehensive visualization of training progress."""
    if not HAS_MATPLOTLIB:
        print_text_summary(history, eval_results, elo_results)
        return

    # Determine number of plots based on available data
    has_loss = any(history.get('loss', []))
    has_time = any(history.get('total_time', []))
    has_eval = eval_results is not None and len(eval_results) > 0
    has_elo = elo_results is not None and len(elo_results) > 0

    # Create figure with appropriate layout
    n_rows = 2
    n_cols = 3 if (has_loss or has_time) else 2

    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle('Razzle Dazzle Training Progress', fontsize=14, fontweight='bold')

    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)

    iterations = history['iterations']

    # Plot 1: Win distribution
    ax1 = fig.add_subplot(gs[0, 0])
    width = 0.25
    x = np.arange(len(iterations))
    ax1.bar(x - width, history['p1_wins'], width, label='P1 Wins', color='#2196F3', alpha=0.8)
    ax1.bar(x, history['p2_wins'], width, label='P2 Wins', color='#F44336', alpha=0.8)
    ax1.bar(x + width, history['draws'], width, label='Draws', color='#9E9E9E', alpha=0.8)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Games')
    ax1.set_title('Win Distribution')
    ax1.set_xticks(x)
    ax1.set_xticklabels(iterations)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Game length (key early indicator)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(iterations, history['avg_game_length'], 'g-o', linewidth=2, markersize=6, label='Average')

    # Show min/max range if available
    if any(history.get('min_game_length', [])) and any(history.get('max_game_length', [])):
        ax2.fill_between(iterations,
                         history['min_game_length'],
                         history['max_game_length'],
                         alpha=0.2, color='green', label='Min-Max Range')
    else:
        ax2.fill_between(iterations, history['avg_game_length'], alpha=0.2, color='green')

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Moves')
    ax2.set_title('Game Length (Early Learning Indicator)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Loss curves (if available)
    if has_loss:
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(iterations, history['loss'], 'b-o', linewidth=2, label='Total Loss', markersize=6)
        ax3.plot(iterations, history['policy_loss'], 'r--s', linewidth=1.5, label='Policy Loss', markersize=4)
        ax3.plot(iterations, history['value_loss'], 'g--^', linewidth=1.5, label='Value Loss', markersize=4)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training Loss')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3)

    # Plot 4: Training throughput
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(iterations, history['training_examples'], color='#9C27B0', alpha=0.7)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Examples')
    ax4.set_title('Training Examples per Iteration')
    ax4.grid(axis='y', alpha=0.3)

    # Plot 5: Win rate vs random / ELO
    ax5 = fig.add_subplot(gs[1, 1])
    if has_eval:
        eval_iters = [r['iteration'] for r in eval_results]
        win_rates = [r['win_rate'] for r in eval_results]
        ax5.plot(eval_iters, win_rates, 'b-o', linewidth=2, markersize=8, label='Win Rate')
        ax5.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random baseline')
        ax5.set_ylim(0, 1)
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Win Rate')
        ax5.set_title('Win Rate vs Random')
        ax5.legend(loc='lower right', fontsize=8)
        ax5.grid(True, alpha=0.3)
    elif has_elo:
        elo_iters = [r['iteration'] for r in elo_results]
        elos = [r['elo'] for r in elo_results]
        ax5.plot(elo_iters, elos, 'b-o', linewidth=2, markersize=8)
        ax5.axhline(y=1000, color='r', linestyle='--', alpha=0.7, label='Starting ELO')
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('ELO Rating')
        ax5.set_title('ELO Progression')
        ax5.legend(loc='lower right', fontsize=8)
        ax5.grid(True, alpha=0.3)
    else:
        # Check if we have win_rate in history
        if any(w is not None for w in history.get('win_rate_vs_random', [])):
            win_rates = [w if w is not None else 0.5 for w in history['win_rate_vs_random']]
            ax5.plot(iterations, win_rates, 'b-o', linewidth=2, markersize=8)
            ax5.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
            ax5.set_ylim(0, 1)
            ax5.set_title('Win Rate vs Random')
        else:
            ax5.text(0.5, 0.5, 'Run with --evaluate\nto see model strength',
                     ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Win Rate vs Random')
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Win Rate')
        ax5.grid(True, alpha=0.3)

    # Plot 6: Timing breakdown (if available)
    if has_time:
        ax6 = fig.add_subplot(gs[1, 2])
        selfplay_times = history.get('selfplay_time', [0] * len(iterations))
        training_times = history.get('training_time', [0] * len(iterations))

        width = 0.35
        x = np.arange(len(iterations))
        ax6.bar(x - width/2, selfplay_times, width, label='Self-play', color='#FF9800', alpha=0.8)
        ax6.bar(x + width/2, training_times, width, label='Training', color='#4CAF50', alpha=0.8)
        ax6.set_xlabel('Iteration')
        ax6.set_ylabel('Time (seconds)')
        ax6.set_title('Time per Iteration')
        ax6.set_xticks(x)
        ax6.set_xticklabels(iterations)
        ax6.legend(loc='upper right', fontsize=8)
        ax6.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def print_text_summary(history: dict, eval_results: list = None, elo_results: list = None):
    """Print text summary when matplotlib is not available."""
    print("\n" + "=" * 80)
    print("Training Progress Summary")
    print("=" * 80)

    # Check if we have detailed game length stats
    has_length_details = any(history.get('min_game_length', []))

    # Table header
    if has_length_details:
        print(f"\n{'Iter':>4} | {'Games':>5} | {'P1':>3} | {'P2':>3} | {'Draw':>4} | {'AvgLen':>6} | {'Range':>11} | {'Loss':>7}")
        print("-" * 80)
    else:
        print(f"\n{'Iter':>4} | {'Games':>5} | {'P1':>3} | {'P2':>3} | {'Draw':>4} | {'AvgLen':>6} | {'Examples':>8} | {'Loss':>7}")
        print("-" * 70)

    for i, iter_num in enumerate(history['iterations']):
        loss_str = f"{history['loss'][i]:.4f}" if history.get('loss') and i < len(history['loss']) and history['loss'][i] else "   -   "

        if has_length_details:
            min_len = history['min_game_length'][i] if i < len(history.get('min_game_length', [])) else 0
            max_len = history['max_game_length'][i] if i < len(history.get('max_game_length', [])) else 0
            range_str = f"({min_len:3d}-{max_len:3d})" if min_len and max_len else "     -     "
            print(f"{iter_num:>4} | {history['games'][i]:>5} | {history['p1_wins'][i]:>3} | "
                  f"{history['p2_wins'][i]:>3} | {history['draws'][i]:>4} | "
                  f"{history['avg_game_length'][i]:>6.1f} | {range_str:>11} | {loss_str}")
        else:
            print(f"{iter_num:>4} | {history['games'][i]:>5} | {history['p1_wins'][i]:>3} | "
                  f"{history['p2_wins'][i]:>3} | {history['draws'][i]:>4} | "
                  f"{history['avg_game_length'][i]:>6.1f} | {history['training_examples'][i]:>8} | {loss_str}")

    # Totals
    print("-" * 70)
    total_games = sum(history['games'])
    total_examples = sum(history['training_examples'])
    total_p1 = sum(history['p1_wins'])
    total_p2 = sum(history['p2_wins'])
    total_draws = sum(history['draws'])
    print(f"{'Total':>4} | {total_games:>5} | {total_p1:>3} | {total_p2:>3} | {total_draws:>4} | "
          f"{'':>6} | {total_examples:>8} |")

    # Timing info
    if history.get('total_time') and any(history['total_time']):
        total_time = sum(history['total_time'])
        print(f"\nTotal training time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        avg_iter_time = total_time / len(history['iterations'])
        print(f"Average time per iteration: {avg_iter_time:.1f} seconds")

    if eval_results:
        print("\n" + "-" * 40)
        print("Model Evaluation vs Random:")
        for r in eval_results:
            extra = f", avg length: {r['avg_game_length']:.0f}" if 'avg_game_length' in r else ""
            print(f"  Iter {r['iteration']}: {r['win_rate']:.1%} win rate ({r['wins']}W/{r['losses']}L/{r['draws']}D){extra}")

    if elo_results:
        print("\n" + "-" * 40)
        print("ELO Ratings (model vs model):")
        for r in elo_results:
            diff = r['elo'] - 1000
            sign = '+' if diff >= 0 else ''
            print(f"  Iter {r['iteration']}: ELO {r['elo']:.0f} ({sign}{diff:.0f})")


def main():
    parser = argparse.ArgumentParser(description='Visualize Razzle Dazzle training')
    parser.add_argument('--output', type=Path, default=Path('output'), help='Training output directory')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate models against random')
    parser.add_argument('--elo', action='store_true', help='Compute ELO ratings between models')
    parser.add_argument('--eval-games', type=int, default=10, help='Games per evaluation')
    parser.add_argument('--save', type=Path, help='Save plot to file')
    parser.add_argument('--text', action='store_true', help='Force text output (no plots)')

    args = parser.parse_args()

    if not args.output.exists():
        print(f"Output directory not found: {args.output}")
        return

    # Try to load from JSON log first, fall back to legacy format
    print("Loading training history...")
    log = load_training_log(args.output)

    if log:
        print("Using training_log.json format")
        history = convert_log_to_history(log)
    else:
        print("Using legacy pickle format")
        history = load_legacy_history(args.output)

    if not history['iterations']:
        print("No training data found")
        return

    print(f"Found {len(history['iterations'])} iterations")

    # Evaluate models if requested
    eval_results = None
    if args.evaluate:
        print("\nEvaluating models vs random...")
        eval_results = evaluate_models(args.output, args.eval_games)

    # Compute ELO if requested
    elo_results = None
    if args.elo:
        print("\nComputing ELO ratings...")
        elo_results = compute_elo(args.output, games_per_match=args.eval_games)

    # Create visualization
    if args.text or not HAS_MATPLOTLIB:
        print_text_summary(history, eval_results, elo_results)
    else:
        plot_training_progress(history, eval_results, elo_results, args.save)


if __name__ == '__main__':
    main()
