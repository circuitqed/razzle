"""Generate a large set of neural network inference reference fixtures from random game positions."""

import json
import random
import sys
import torch
import numpy as np

from razzle.core.state import GameState
from razzle.core.moves import get_legal_moves, END_TURN_MOVE
from razzle.ai.network import create_network

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

MODEL_PATH = "/app/models/iter_300.pt"
OUTPUT_PATH = "/tmp/inference-fixtures-large.json"
NUM_GAMES = 50
POSITIONS_PER_GAME = 4
TARGET_POSITIONS = NUM_GAMES * POSITIONS_PER_GAME  # 200
MAX_PLY = 200  # safety limit to avoid infinite games

print("Loading network...")
net = create_network(preset="medium")
ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
if "conv_input.weight" in ckpt:
    sd = ckpt
net.load_state_dict(sd)
net.eval()
print("Network loaded.")


def play_random_game():
    """Play a random game to completion, returning list of (state_copy, ply) at each position."""
    state = GameState()
    positions = []
    while not state.is_terminal() and state.ply < MAX_PLY:
        # Record position before the move
        positions.append((state.copy(), state.ply, state.current_player))
        moves = get_legal_moves(state)
        if not moves:
            break
        move = random.choice(moves)
        state.apply_move(move)
    return positions


def sample_positions_from_game(positions, num_samples=POSITIONS_PER_GAME):
    """Sample diverse positions from a game: mix of early, mid, and late game."""
    if len(positions) < num_samples:
        return positions

    # Categorize positions by game phase
    early = [p for p in positions if p[1] <= 10]
    mid = [p for p in positions if 10 < p[1] <= 25]
    late = [p for p in positions if p[1] > 25]

    sampled = []

    # Try to get at least one from each phase
    for bucket in [early, mid, late]:
        if bucket:
            sampled.append(random.choice(bucket))

    # Fill remaining slots from all positions (excluding already sampled)
    sampled_set = set(id(s[0]) for s in sampled)
    remaining = [p for p in positions if id(p[0]) not in sampled_set]
    if remaining:
        need = num_samples - len(sampled)
        if need > 0:
            sampled.extend(random.sample(remaining, min(need, len(remaining))))

    return sampled[:num_samples]


def compute_inference(state):
    """Compute neural network inference for a position."""
    tensor = state.to_tensor()  # shape (7, 8, 7), numpy array
    input_t = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        policy_logits, value, difficulty = net(input_t)

    policy = policy_logits.squeeze(0).tolist()  # 3137 elements (log-softmax)
    val = value.item()

    return tensor.tolist(), policy, val


print(f"Playing {NUM_GAMES} random games and sampling ~{POSITIONS_PER_GAME} positions each...")
print(f"Target: {TARGET_POSITIONS} positions")
print()

all_positions = []
ply_distribution = {"early": 0, "mid": 0, "late": 0}

for game_idx in range(NUM_GAMES):
    game_positions = play_random_game()
    sampled = sample_positions_from_game(game_positions)

    for state, ply, player in sampled:
        tensor_list, policy, value = compute_inference(state)
        all_positions.append({
            "tensor": tensor_list,
            "policy": policy,
            "value": round(value, 8),
            "ply": ply,
            "current_player": player,
        })

        # Track distribution
        if ply <= 10:
            ply_distribution["early"] += 1
        elif ply <= 25:
            ply_distribution["mid"] += 1
        else:
            ply_distribution["late"] += 1

    count = len(all_positions)
    if count >= 20 and (count // 20) > ((count - len(sampled)) // 20):
        print(f"  Progress: {count} positions collected (game {game_idx + 1}/{NUM_GAMES})")
    elif game_idx == 0:
        print(f"  Progress: {count} positions collected (game 1/{NUM_GAMES})")

# Trim to exactly TARGET_POSITIONS if we have more
if len(all_positions) > TARGET_POSITIONS:
    all_positions = all_positions[:TARGET_POSITIONS]

print()
print(f"Collected {len(all_positions)} positions")
print(f"Ply distribution: {ply_distribution}")

# Verify all tensors and policies have correct shapes
for i, pos in enumerate(all_positions):
    assert len(pos["tensor"]) == 7, f"Position {i}: tensor dim 0 is {len(pos['tensor'])}, expected 7"
    assert len(pos["tensor"][0]) == 8, f"Position {i}: tensor dim 1 is {len(pos['tensor'][0])}, expected 8"
    assert len(pos["tensor"][0][0]) == 7, f"Position {i}: tensor dim 2 is {len(pos['tensor'][0][0])}, expected 7"
    assert len(pos["policy"]) == 3137, f"Position {i}: policy length is {len(pos['policy'])}, expected 3137"

print("Shape validation passed.")

# Build output
output = {
    "model": "iter_300.pt",
    "num_positions": len(all_positions),
    "positions": all_positions,
}

print(f"Writing to {OUTPUT_PATH}...")
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f)

# Report file size
import os
size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
print(f"Done! File size: {size_mb:.1f} MB")
print(f"Positions: {len(all_positions)}")
