#!/usr/bin/env python3
"""
Diagnostic script to analyze neural network policy outputs.

Checks whether the network has learned to assign probability mass
to legal moves vs illegal moves.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.core.state import GameState
from razzle.core.moves import get_legal_moves, move_to_algebraic
from razzle.ai.network import RazzleNet, NUM_ACTIONS, END_TURN_ACTION
from razzle.ai.evaluator import BatchedEvaluator


def analyze_policy(network: RazzleNet, state: GameState, device: str = 'cpu') -> dict:
    """
    Analyze network policy output for a given state.

    Returns dict with:
    - legal_moves: list of legal moves
    - legal_prob_mass: total probability on legal moves
    - illegal_prob_mass: total probability on illegal moves
    - top_legal: top 5 legal moves with probs
    - top_illegal: top 5 illegal moves with probs (if any have significant mass)
    """
    # Get network output
    tensor = torch.from_numpy(state.to_tensor()).unsqueeze(0).to(device)
    network.eval()
    with torch.no_grad():
        log_policy, value = network(tensor)

    policy = torch.exp(log_policy).squeeze(0).cpu().numpy()
    value = value.item()

    # Get legal moves
    legal_moves = set(get_legal_moves(state))

    # Map END_TURN (-1) to its policy index
    legal_indices = set()
    for m in legal_moves:
        if m == -1:
            legal_indices.add(END_TURN_ACTION)
        else:
            legal_indices.add(m)

    # Calculate probability mass
    legal_prob_mass = sum(policy[i] for i in legal_indices)
    illegal_prob_mass = 1.0 - legal_prob_mass

    # Top legal moves
    legal_probs = [(m, policy[END_TURN_ACTION if m == -1 else m]) for m in legal_moves]
    legal_probs.sort(key=lambda x: x[1], reverse=True)
    top_legal = legal_probs[:5]

    # Top illegal moves (indices not in legal_indices)
    illegal_probs = [(i, policy[i]) for i in range(NUM_ACTIONS) if i not in legal_indices]
    illegal_probs.sort(key=lambda x: x[1], reverse=True)
    top_illegal = illegal_probs[:5]

    return {
        'legal_moves': legal_moves,
        'num_legal': len(legal_moves),
        'legal_prob_mass': legal_prob_mass,
        'illegal_prob_mass': illegal_prob_mass,
        'top_legal': top_legal,
        'top_illegal': top_illegal,
        'value': value,
    }


def diagnose_model(model_path: str, num_positions: int = 20, device: str = 'cpu'):
    """
    Diagnose a trained model by analyzing policy outputs on random positions.
    """
    print(f"Loading model from {model_path}")
    network = RazzleNet.load(model_path, device=device)
    network.eval()

    print(f"\nAnalyzing {num_positions} positions...\n")

    legal_masses = []
    illegal_masses = []

    # Test starting position
    print("=" * 60)
    print("STARTING POSITION")
    print("=" * 60)
    state = GameState.new_game()
    result = analyze_policy(network, state, device)
    print_analysis(state, result)
    legal_masses.append(result['legal_prob_mass'])
    illegal_masses.append(result['illegal_prob_mass'])

    # Play some random moves and analyze
    print("\n" + "=" * 60)
    print("RANDOM GAME POSITIONS")
    print("=" * 60)

    for i in range(num_positions - 1):
        # Play random moves to get varied positions
        state = GameState.new_game()
        num_moves = np.random.randint(1, 30)

        for _ in range(num_moves):
            if state.is_terminal():
                break
            legal = get_legal_moves(state)
            if not legal:
                break
            move = np.random.choice(legal)
            state.apply_move(move)

        if state.is_terminal():
            continue

        result = analyze_policy(network, state, device)
        legal_masses.append(result['legal_prob_mass'])
        illegal_masses.append(result['illegal_prob_mass'])

        if i < 5:  # Only print first 5 random positions
            print(f"\nPosition after {num_moves} random moves:")
            print_analysis(state, result)

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Positions analyzed: {len(legal_masses)}")
    print(f"\nProbability mass on LEGAL moves:")
    print(f"  Mean:   {np.mean(legal_masses):.4f}")
    print(f"  Std:    {np.std(legal_masses):.4f}")
    print(f"  Min:    {np.min(legal_masses):.4f}")
    print(f"  Max:    {np.max(legal_masses):.4f}")
    print(f"\nProbability mass on ILLEGAL moves:")
    print(f"  Mean:   {np.mean(illegal_masses):.4f}")
    print(f"  Std:    {np.std(illegal_masses):.4f}")
    print(f"  Min:    {np.min(illegal_masses):.4f}")
    print(f"  Max:    {np.max(illegal_masses):.4f}")

    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    mean_legal = np.mean(legal_masses)
    if mean_legal > 0.9:
        print("GOOD: Network strongly favors legal moves (>90% mass)")
    elif mean_legal > 0.7:
        print("OK: Network somewhat favors legal moves (70-90% mass)")
    elif mean_legal > 0.5:
        print("POOR: Network weakly favors legal moves (50-70% mass)")
    else:
        print("BAD: Network assigns more mass to illegal moves than legal!")
        print("     This suggests the network hasn't learned move legality.")

    # Compare to uniform baseline
    # With ~15 legal moves out of 3137, uniform would give ~0.5% to legal
    print(f"\nFor reference: Uniform random policy would give ~0.5% to legal moves")
    print(f"               Your network gives {mean_legal*100:.1f}% to legal moves")

    if mean_legal > 0.005:
        improvement = mean_legal / 0.005
        print(f"               That's {improvement:.0f}x better than random")


def print_analysis(state: GameState, result: dict):
    """Print analysis results for a position."""
    print(state)
    print(f"\nLegal moves: {result['num_legal']}")
    print(f"Probability mass on legal moves: {result['legal_prob_mass']:.4f} ({result['legal_prob_mass']*100:.1f}%)")
    print(f"Probability mass on illegal moves: {result['illegal_prob_mass']:.4f} ({result['illegal_prob_mass']*100:.1f}%)")
    print(f"Network value: {result['value']:.4f}")

    print(f"\nTop legal moves:")
    for move, prob in result['top_legal']:
        alg = move_to_algebraic(move)
        print(f"  {alg:8s} {prob:.4f} ({prob*100:.1f}%)")

    if result['top_illegal'][0][1] > 0.01:  # Only show if significant
        print(f"\nTop illegal moves (should be ~0):")
        for idx, prob in result['top_illegal'][:3]:
            # Decode the illegal move for inspection
            if idx == END_TURN_ACTION:
                desc = "END_TURN (illegal here)"
            else:
                src, dst = idx // 56, idx % 56
                desc = f"sq{src}->sq{dst}"
            print(f"  {desc:20s} {prob:.4f} ({prob*100:.1f}%)")


def compare_with_uniform():
    """Show what a uniform (untrained) network looks like."""
    print("=" * 60)
    print("BASELINE: UNTRAINED NETWORK (UNIFORM POLICY)")
    print("=" * 60)

    from razzle.ai.network import create_network
    network = create_network(num_filters=64, num_blocks=6)

    state = GameState.new_game()
    result = analyze_policy(network, state)

    print(f"\nUntrained network on starting position:")
    print(f"Legal moves: {result['num_legal']}")
    print(f"Probability on legal: {result['legal_prob_mass']:.4f} ({result['legal_prob_mass']*100:.2f}%)")
    print(f"Probability on illegal: {result['illegal_prob_mass']:.4f} ({result['illegal_prob_mass']*100:.2f}%)")
    print(f"\nExpected for uniform over {NUM_ACTIONS} actions with {result['num_legal']} legal:")
    expected = result['num_legal'] / NUM_ACTIONS
    print(f"  {expected:.4f} ({expected*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description='Diagnose neural network policy outputs')
    parser.add_argument('model_path', type=str, nargs='?', default=None,
                        help='Path to model file (.pt)')
    parser.add_argument('--positions', type=int, default=20,
                        help='Number of positions to analyze')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('--baseline', action='store_true',
                        help='Show baseline untrained network stats')

    args = parser.parse_args()

    if args.baseline:
        compare_with_uniform()
        print()

    if args.model_path:
        diagnose_model(args.model_path, args.positions, args.device)
    elif not args.baseline:
        # If no model and no baseline, show baseline by default
        compare_with_uniform()
        print("\nTo analyze a trained model, provide the model path:")
        print("  python diagnose_policy.py output/models/iter_230.pt")


if __name__ == '__main__':
    main()
