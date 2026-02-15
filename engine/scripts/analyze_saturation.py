#!/usr/bin/env python3
"""
Offline training experiments to diagnose training saturation.

Experiments:
1. Train/val split - detect overfitting vs capacity limit
2. Large network comparison - detect capacity saturation
3. Learning rate sensitivity - can we push past the plateau

Usage:
    # Step 1: Convert games to numpy (slow, only needed once)
    python scripts/analyze_saturation.py --convert

    # Step 2: Run experiments (uses cached data)
    python scripts/analyze_saturation.py --experiment all
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from razzle.ai.network import RazzleNet, create_network, NUM_ACTIONS, END_TURN_ACTION
from razzle.core.state import GameState
from razzle.core.moves import get_legal_moves
from razzle.core.symmetry import rotate_policy_180
from razzle.training.trainer import Trainer, TrainingConfig, RazzleDataset


CACHE_DIR = Path("output/saturation_analysis")
CACHE_FILE = CACHE_DIR / "training_data.npz"
DB_PATH = Path("server/games.db")


@dataclass
class TrainingGame:
    moves: list[int]
    result: float
    visit_counts: list[dict[int, int]]
    model_version: str


def load_games_from_db(max_games: int = 0) -> list[TrainingGame]:
    """Load training games from database (newest first)."""
    db = sqlite3.connect(str(DB_PATH))
    c = db.cursor()
    query = "SELECT moves, result, visit_counts, model_version FROM training_games ORDER BY id DESC"
    if max_games > 0:
        query += f" LIMIT {max_games}"
    c.execute(query)

    games = []
    for row in c.fetchall():
        moves = json.loads(row[0])
        result = row[1]
        vc_raw = json.loads(row[2])
        visit_counts = [{int(k): v for k, v in vc.items()} for vc in vc_raw]
        model_version = row[3] or "unknown"
        games.append(TrainingGame(moves=moves, result=result,
                                   visit_counts=visit_counts, model_version=model_version))
    db.close()
    return games


def games_to_training_data(games: list[TrainingGame]) -> tuple:
    """Convert games to numpy arrays."""
    all_states = []
    all_policies = []
    all_values = []
    all_legal_masks = []
    skipped = 0

    for gi, game in enumerate(games):
        if gi % 1000 == 0 and gi > 0:
            print(f"  Converting game {gi}/{len(games)}... ({len(all_states)} positions)")

        if len(game.moves) != len(game.visit_counts):
            skipped += 1
            continue

        state = GameState.new_game()
        states, policies, legal_masks, players = [], [], [], []

        try:
            for move, visit_counts in zip(game.moves, game.visit_counts):
                states.append(state.to_tensor())
                players.append(state.current_player)

                legal_mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
                for m in get_legal_moves(state):
                    legal_mask[END_TURN_ACTION if m == -1 else m] = 1.0
                legal_masks.append(legal_mask)

                policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
                total_visits = sum(visit_counts.values())
                if total_visits > 0:
                    for m, count in visit_counts.items():
                        policy[END_TURN_ACTION if m == -1 else m] = count / total_visits
                policies.append(policy)

                state.apply_move(move)
        except Exception:
            skipped += 1
            continue

        for i in range(len(states)):
            player = players[i]
            value = max(-0.99, min(0.99, game.result if player == 0 else -game.result))
            all_states.append(states[i])
            if player == 1:
                all_policies.append(rotate_policy_180(policies[i]))
                all_legal_masks.append(rotate_policy_180(legal_masks[i]))
            else:
                all_policies.append(policies[i])
                all_legal_masks.append(legal_masks[i])
            all_values.append(value)

    if skipped:
        print(f"  Skipped {skipped} games with errors")

    return (np.stack(all_states), np.stack(all_policies),
            np.array(all_values, dtype=np.float32), np.stack(all_legal_masks))


def convert_and_cache(max_games: int = 3000):
    """Convert games to numpy and cache to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading up to {max_games} games from database (newest first)...")
    games = load_games_from_db(max_games=max_games)
    print(f"Loaded {len(games)} games")
    print(f"Model versions: {dict(sorted(Counter(g.model_version for g in games).items()))}")

    CHUNK_SIZE = 1000
    chunk_files = []

    t0 = time.time()
    total_positions = 0
    for chunk_start in range(0, len(games), CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(games))
        chunk_games = games[chunk_start:chunk_end]
        print(f"\nChunk {chunk_start//CHUNK_SIZE + 1}: games {chunk_start}-{chunk_end}...")

        states, policies, values, legal_masks = games_to_training_data(chunk_games)
        total_positions += len(states)

        chunk_file = CACHE_DIR / f"chunk_{chunk_start:05d}.npz"
        np.savez_compressed(chunk_file, states=states, policies=policies.astype(np.float16),
                            values=values, legal_masks=legal_masks.astype(np.float16))
        chunk_files.append(str(chunk_file))
        print(f"  Saved {len(states)} positions")
        del states, policies, values, legal_masks

    elapsed = time.time() - t0
    print(f"\nConverted {total_positions} positions in {elapsed:.1f}s")

    print("Concatenating chunks...")
    all_s, all_p, all_v, all_m = [], [], [], []
    for cf in chunk_files:
        data = np.load(cf)
        all_s.append(data["states"])
        all_p.append(data["policies"])
        all_v.append(data["values"])
        all_m.append(data["legal_masks"])

    np.savez_compressed(CACHE_FILE,
                        states=np.concatenate(all_s),
                        policies=np.concatenate(all_p),
                        values=np.concatenate(all_v),
                        legal_masks=np.concatenate(all_m))
    print(f"Saved {total_positions} positions ({CACHE_FILE.stat().st_size / 1e6:.1f} MB)")

    for cf in chunk_files:
        Path(cf).unlink()


def load_cached_data():
    """Load cached training data."""
    if not CACHE_FILE.exists():
        print("Run with --convert first")
        sys.exit(1)

    print(f"Loading cached data...")
    data = np.load(CACHE_FILE)
    states = data["states"].astype(np.float32)
    policies = data["policies"].astype(np.float32)
    values = data["values"].astype(np.float32)
    legal_masks = data["legal_masks"].astype(np.float32)
    print(f"Loaded {len(states)} positions")
    return states, policies, values, legal_masks


def split_data(states, policies, values, legal_masks, val_fraction=0.2, seed=42):
    """Split data into train/val sets."""
    rng = np.random.RandomState(seed)
    n = len(states)
    indices = rng.permutation(n)
    val_size = int(n * val_fraction)
    val_idx, train_idx = indices[:val_size], indices[val_size:]

    return {
        "train": (states[train_idx], policies[train_idx], values[train_idx], legal_masks[train_idx]),
        "val": (states[val_idx], policies[val_idx], values[val_idx], legal_masks[val_idx]),
    }


def evaluate_model(network, states, policies, values, legal_masks, device, batch_size=1024):
    """Evaluate a model on data without training."""
    network.eval()
    dataset = RazzleDataset(states, policies, values, legal_masks)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    total_p, total_v, total_ill = 0.0, 0.0, 0.0
    total_n = 0
    all_preds = []
    correct_top1 = 0

    with torch.no_grad():
        for batch in loader:
            s, p, v, m = [x.to(device) for x in batch]
            log_pol, pred_v, _ = network(s)
            pred_v = pred_v.squeeze(-1)
            probs = torch.exp(log_pol)

            # Policy loss
            masked_p = p * m
            masked_log = log_pol * m
            p_loss = -torch.sum(masked_p * masked_log, dim=1).mean()

            # Value MSE
            v_loss = F.mse_loss(pred_v, v)

            # Illegal penalty
            ill = torch.sum(probs * (1.0 - m), dim=1).mean()

            # Top-1 accuracy
            masked_probs = probs * m
            pred_moves = masked_probs.argmax(dim=1)
            target_moves = p.argmax(dim=1)
            correct_top1 += (pred_moves == target_moves).sum().item()

            bs = s.shape[0]
            total_p += p_loss.item() * bs
            total_v += v_loss.item() * bs
            total_ill += ill.item() * bs
            total_n += bs
            all_preds.append(pred_v.cpu().numpy())

    pred_values = np.concatenate(all_preds)
    sign_acc = float(np.mean(np.sign(pred_values) == np.sign(values)))
    top1_acc = correct_top1 / total_n

    return {
        "policy_loss": total_p / total_n,
        "value_mse": total_v / total_n,
        "illegal_penalty": total_ill / total_n,
        "value_sign_accuracy": sign_acc,
        "policy_top1_accuracy": top1_acc,
    }


def train_epochs(network, train_data, val_data, device, epochs=5, batch_size=512, lr=0.001, label=""):
    """Train for N epochs tracking train and val metrics."""
    t_s, t_p, t_v, t_m = train_data
    dataset = RazzleDataset(t_s, t_p, t_v, t_m)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=1e-4)

    params = sum(p.numel() for p in network.parameters())
    print(f"\n  Training {label} ({params:,} params), {len(t_s)} train, {len(val_data[0])} val, lr={lr}")

    results = []
    for epoch in range(epochs):
        t0 = time.time()
        network.train()
        ep_p, ep_v, ep_ill, n_batch = 0.0, 0.0, 0.0, 0

        for batch in loader:
            s, p, v, m = [x.to(device) for x in batch]
            log_pol, pred_v, pred_d = network(s)
            pred_v = pred_v.squeeze(-1)
            probs = torch.exp(log_pol)

            p_loss = -torch.sum(p * m * log_pol * m, dim=1).mean()
            v_diff = pred_v - v
            v_loss = torch.mean(v_diff ** 2)
            ill = torch.sum(probs * (1.0 - m), dim=1).mean()
            loss = p_loss + 20.0 * v_loss + 1.0 * torch.mean(v_diff ** 4) + ill

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_p += p_loss.item()
            ep_v += v_loss.item()
            ep_ill += ill.item()
            n_batch += 1

        train_p = ep_p / n_batch
        train_v = ep_v / n_batch

        # Validate
        val_metrics = evaluate_model(network, *val_data, device)
        elapsed = time.time() - t0

        r = {"epoch": epoch + 1, "train_policy": train_p, "train_value": train_v,
             "val_policy": val_metrics["policy_loss"], "val_value": val_metrics["value_mse"],
             "val_sign_acc": val_metrics["value_sign_accuracy"],
             "val_top1": val_metrics["policy_top1_accuracy"], "time": elapsed}
        results.append(r)

        print(f"    Ep {epoch+1:>2}: train_p={train_p:.4f} train_v={train_v:.5f} | "
              f"val_p={r['val_policy']:.4f} val_v={r['val_value']:.5f} "
              f"sign={r['val_sign_acc']*100:.1f}% top1={r['val_top1']*100:.1f}% "
              f"({elapsed:.0f}s)")

    return results


def run_experiments(device):
    """Run all saturation experiments."""
    states, policies, values, legal_masks = load_cached_data()
    splits = split_data(states, policies, values, legal_masks)
    val_data = splits["val"]
    train_data = splits["train"]

    print(f"Train: {len(train_data[0])} positions, Val: {len(val_data[0])} positions")

    # ================================================================
    # EXPERIMENT 1: Evaluate existing models on train/val split
    # ================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: GENERALIZATION GAP ACROSS TRAINING")
    print("=" * 70)

    models_dir = Path("output/models")
    checkpoints = sorted(models_dir.glob("iter_*.pt"))

    # Evaluate every 5th model + last few
    eval_models = []
    for cp in checkpoints:
        iter_num = int(cp.stem.split("_")[1])
        if iter_num <= 5 or iter_num % 10 == 0 or iter_num >= len(checkpoints) - 3:
            eval_models.append(cp)

    print(f"\nEvaluating {len(eval_models)} checkpoints on train/val split...")
    print(f"{'Model':>12} | {'Train P':>8} {'Train V':>8} {'Trn Top1':>8} | "
          f"{'Val P':>8} {'Val V':>8} {'Val Top1':>8} {'Val Sgn%':>8} | {'P Gap':>6} {'V Gap':>6}")
    print("-" * 110)

    all_results = []
    for model_path in eval_models:
        net = RazzleNet.load(model_path, device=device).to(device)
        train_m = evaluate_model(net, *train_data, device)
        val_m = evaluate_model(net, *val_data, device)
        p_gap = val_m["policy_loss"] - train_m["policy_loss"]
        v_gap = val_m["value_mse"] - train_m["value_mse"]

        print(f"{model_path.stem:>12} | {train_m['policy_loss']:>8.4f} {train_m['value_mse']:>8.5f} "
              f"{train_m['policy_top1_accuracy']*100:>7.1f}% | "
              f"{val_m['policy_loss']:>8.4f} {val_m['value_mse']:>8.5f} "
              f"{val_m['policy_top1_accuracy']*100:>7.1f}% {val_m['value_sign_accuracy']*100:>7.1f}% | "
              f"{p_gap:>+6.3f} {v_gap:>+6.4f}")

        all_results.append({
            "model": model_path.stem,
            "train": train_m, "val": val_m, "p_gap": p_gap, "v_gap": v_gap
        })
        del net

    # Summary
    last = all_results[-1]
    print(f"\nSUMMARY (latest model {last['model']}):")
    print(f"  Policy gap (val - train): {last['p_gap']:+.4f}")
    print(f"  Value gap (val - train):  {last['v_gap']:+.5f}")
    if abs(last['p_gap']) < 0.05:
        print("  --> Minimal overfitting: network is NOT memorizing training data")
        print("  --> Plateau likely due to capacity limit or data quality, not overfitting")
    elif last['p_gap'] > 0.1:
        print("  --> Significant overfitting detected")
        print("  --> More data or regularization could help")

    # ================================================================
    # EXPERIMENT 2: Fresh medium retrain (5 epochs from scratch)
    # ================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: FRESH RETRAIN vs ONLINE-TRAINED (Medium)")
    print("=" * 70)
    print("Training fresh medium network from scratch on all data...")

    net_fresh = create_network(preset='medium', device=device)
    fresh_results = train_epochs(net_fresh, train_data, val_data, device,
                                  epochs=5, batch_size=512, lr=0.001, label="medium-fresh")

    # Compare best fresh with iter_052
    best_fresh = min(fresh_results, key=lambda r: r["val_policy"])
    iter052_val = all_results[-1]["val"]  # From experiment 1

    print(f"\n  Comparison after {len(fresh_results)} epochs from scratch:")
    print(f"    iter_052 val policy: {iter052_val['policy_loss']:.4f}, val sign%: {iter052_val['value_sign_accuracy']*100:.1f}%")
    print(f"    Fresh    val policy: {best_fresh['val_policy']:.4f}, val sign%: {best_fresh['val_sign_acc']*100:.1f}%")

    del net_fresh

    # ================================================================
    # EXPERIMENT 3: Fine-tune iter_052 with lower LR
    # ================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: LEARNING RATE SENSITIVITY (Fine-tune iter_052)")
    print("=" * 70)

    for lr in [0.0005, 0.0001, 0.00005]:
        net_ft = RazzleNet.load(models_dir / "iter_052.pt", device=device).to(device)
        ft_results = train_epochs(net_ft, train_data, val_data, device,
                                   epochs=3, batch_size=512, lr=lr, label=f"iter_052+lr={lr}")
        best_ft = min(ft_results, key=lambda r: r["val_policy"])
        improvement = iter052_val['policy_loss'] - best_ft['val_policy']
        print(f"    LR={lr}: best val_policy={best_ft['val_policy']:.4f} "
              f"(improvement: {improvement:+.4f})")
        del net_ft

    # ================================================================
    # EXPERIMENT 4: Large network capacity test (just evaluate + 3 epochs)
    # ================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: LARGE NETWORK CAPACITY TEST")
    print("=" * 70)

    # Check available memory
    import psutil
    avail_gb = psutil.virtual_memory().available / 1e9
    print(f"Available memory: {avail_gb:.1f} GB")

    if avail_gb < 8:
        print("Not enough memory for large network training, skipping")
        print("(Need ~8GB free, consider running with fewer positions)")
    else:
        net_large = create_network(preset='large', device=device)
        large_params = sum(p.numel() for p in net_large.parameters())
        print(f"Large network: {large_params:,} params (vs medium: 2,386,075)")

        large_results = train_epochs(net_large, train_data, val_data, device,
                                      epochs=5, batch_size=256, lr=0.0005, label="large-fresh")
        best_large = min(large_results, key=lambda r: r["val_policy"])
        print(f"\n  Large best val_policy: {best_large['val_policy']:.4f} vs "
              f"medium best: {best_fresh['val_policy']:.4f}")
        improvement = best_fresh['val_policy'] - best_large['val_policy']
        if improvement > 0.05:
            print("  --> CAPACITY LIMITED: Large network learns significantly better")
        elif improvement > 0:
            print("  --> MARGINAL: Slight improvement, medium near its limit")
        else:
            print("  --> NOT CAPACITY LIMITED: Data quality is the bottleneck")
        del net_large

    # ================================================================
    # FINAL DIAGNOSIS
    # ================================================================
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    gap = last['p_gap']
    if abs(gap) < 0.05:
        print("1. No significant overfitting (gap < 0.05)")
        print("   The network generalizes well to unseen data from same distribution")
    else:
        print(f"1. Generalization gap: {gap:+.4f}")

    print("\n2. The plateau is likely caused by:")
    print("   - Self-play data from weak models is limited in quality")
    print("   - The MCTS policies (targets) are imperfect teachers")
    print("   - More training iterations generate better games which improve targets")
    print("   - This is the expected AZ training dynamic: each iteration is a small step")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Training saturation analysis")
    parser.add_argument("--convert", action="store_true")
    parser.add_argument("--experiment", type=str, choices=["all"], default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-games", type=int, default=3000)
    args = parser.parse_args()

    if not args.convert and not args.experiment:
        parser.print_help()
        return

    if args.convert:
        convert_and_cache(max_games=args.max_games)

    if args.experiment:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        run_experiments(args.device)


if __name__ == "__main__":
    main()
