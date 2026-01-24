#!/usr/bin/env python3
"""Measure actual MCTS search depth and tree structure."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from collections import defaultdict
from razzle.core.state import GameState
from razzle.ai.network import RazzleNet
from razzle.ai.mcts import MCTS, MCTSConfig, Node
from razzle.ai.evaluator import BatchedEvaluator


def measure_tree(root: Node) -> dict:
    """Measure tree statistics."""
    depths = []
    visits_by_depth = defaultdict(int)
    nodes_by_depth = defaultdict(int)

    def traverse(node: Node, depth: int):
        if node.visit_count > 0:
            depths.append(depth)
            visits_by_depth[depth] += node.visit_count
            nodes_by_depth[depth] += 1

        for child in node.children.values():
            traverse(child, depth + 1)

    traverse(root, 0)

    max_depth = max(depths) if depths else 0

    # Find depth where 90% of visits occur
    total_visits = sum(visits_by_depth.values())
    cumulative = 0
    depth_90 = 0
    for d in range(max_depth + 1):
        cumulative += visits_by_depth[d]
        if cumulative >= 0.9 * total_visits:
            depth_90 = d
            break

    return {
        'max_depth': max_depth,
        'depth_90_pct': depth_90,
        'visits_by_depth': dict(visits_by_depth),
        'nodes_by_depth': dict(nodes_by_depth),
        'total_nodes': len(depths),
    }


def main():
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load model
    model_path = Path('output/models/iter_100.pt')
    if not model_path.exists():
        model_path = Path('output/models/iter_050.pt')
    if not model_path.exists():
        print("No model found")
        return

    print(f"Loading {model_path}")
    net = RazzleNet.load(model_path, device=device)
    net.eval()

    evaluator = BatchedEvaluator(net, device=device)

    # Test different simulation counts
    for sims in [100, 400, 800, 1600]:
        config = MCTSConfig(num_simulations=sims)
        mcts = MCTS(evaluator, config)

        # Run from starting position
        state = GameState.new_game()
        root = mcts.search(state, add_noise=False)

        stats = measure_tree(root)

        print(f"\n=== {sims} simulations ===")
        print(f"Max depth: {stats['max_depth']} plies")
        print(f"90% of visits within: {stats['depth_90_pct']} plies")
        print(f"Total nodes: {stats['total_nodes']}")
        print(f"Visits by depth: ", end="")
        for d in range(min(8, stats['max_depth'] + 1)):
            v = stats['visits_by_depth'].get(d, 0)
            print(f"d{d}={v} ", end="")
        print()
        print(f"Nodes by depth: ", end="")
        for d in range(min(8, stats['max_depth'] + 1)):
            n = stats['nodes_by_depth'].get(d, 0)
            print(f"d{d}={n} ", end="")
        print()

        # Effective branching factor
        if stats['total_nodes'] > 1:
            # B^d = total_nodes, so B = total_nodes^(1/d)
            avg_depth = sum(d * stats['nodes_by_depth'][d] for d in stats['nodes_by_depth']) / stats['total_nodes']
            if avg_depth > 0:
                eff_branch = stats['total_nodes'] ** (1 / stats['max_depth'])
                print(f"Effective branching factor: {eff_branch:.1f}")


if __name__ == '__main__':
    main()
