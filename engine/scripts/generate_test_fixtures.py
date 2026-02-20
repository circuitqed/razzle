#!/usr/bin/env python3
"""Generate reference test fixtures for cross-validating the TypeScript engine port.

Outputs JSON to stdout with states, legal moves, and tensor data at each step
of several game sequences, including real self-play games.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from razzle.core.state import GameState
from razzle.core.moves import get_legal_moves, move_to_algebraic, algebraic_to_move


def state_to_dict(state: GameState) -> dict:
    """Serialize game state to a JSON-compatible dict.

    Bitboard values are stored as strings to avoid JS Number precision loss
    (56-bit bitboards exceed 2^53 safe integer limit).
    """
    return {
        "pieces": [str(state.pieces[0]), str(state.pieces[1])],
        "balls": [str(state.balls[0]), str(state.balls[1])],
        "currentPlayer": state.current_player,
        "touchedMask": str(state.touched_mask),
        "hasPassed": state.has_passed,
        "lastKnightDst": state.last_knight_dst,
        "ply": state.ply,
    }


def snapshot(state: GameState, description: str, move=None, move_alg=None) -> dict:
    """Capture a snapshot of the current state."""
    legal = get_legal_moves(state)
    tensor = state.to_tensor()
    s = {
        "description": description,
        "state": state_to_dict(state),
        "legalMoves": sorted(legal),
        "tensor": tensor.flatten().tolist(),
        "isTerminal": state.is_terminal(),
        "winner": state.get_winner(),
    }
    if move is not None:
        s["moveApplied"] = move
    if move_alg is not None:
        s["moveAlgebraic"] = move_alg
    return s


def generate_sequence(name: str, moves_algebraic: list[str]) -> dict:
    """Play through a sequence of moves, capturing state at each step."""
    state = GameState.new_game()
    snapshots = [snapshot(state, "initial")]

    for i, move_str in enumerate(moves_algebraic):
        move = -1 if move_str == "end" else algebraic_to_move(move_str)
        state.apply_move(move)
        snapshots.append(snapshot(
            state,
            f"after {move_str} (move {i+1})",
            move=move,
            move_alg=move_str,
        ))

    return {"name": name, "moves": moves_algebraic, "snapshots": snapshots}


def generate_random_game(name: str, seed: int, max_moves: int = 60) -> dict:
    """Play a random game capturing every state. Uses raw move ints."""
    rng = random.Random(seed)
    state = GameState.new_game()
    snapshots = [snapshot(state, "initial")]
    moves_played = []

    for i in range(max_moves):
        if state.is_terminal():
            break
        legal = get_legal_moves(state)
        if not legal:
            break
        move = rng.choice(legal)
        move_alg = move_to_algebraic(move)
        moves_played.append(move_alg)
        state.apply_move(move)
        snapshots.append(snapshot(
            state,
            f"after {move_alg} (move {i+1})",
            move=move,
            move_alg=move_alg,
        ))

    return {"name": name, "moves": moves_played, "snapshots": snapshots}


def main():
    sequences = []

    # --- Scripted sequences testing specific rules ---

    # 1: Simple knight moves by both players
    sequences.append(generate_sequence(
        "knight_moves_only",
        ["b1-c3", "b8-c6", "f1-e3", "f8-e6"]
    ))

    # 2: Pass then end turn
    sequences.append(generate_sequence(
        "pass_and_end_turn",
        ["d1-e1", "end", "d8-e8", "end"]
    ))

    # 3: Pass chain (multiple passes in one turn)
    sequences.append(generate_sequence(
        "pass_chain",
        ["d1-c1", "c1-b1", "end", "d8-c8", "end"]
    ))

    # 4: Touched mask persists across end turn
    sequences.append(generate_sequence(
        "touched_mask_persistence",
        ["d1-e1", "end", "b8-c6", "d1-c1"]
    ))

    # 5: Multiple knight moves
    sequences.append(generate_sequence(
        "mixed_game_start",
        ["b1-a3", "b8-a6", "c1-b3", "c8-b6", "e1-d3", "e8-d6"]
    ))

    # 6: Forced pass scenario
    sequences.append(generate_sequence(
        "forced_pass_test",
        ["b1-c3", "f8-e6", "c3-d1"]
    ))

    # 7: Longer scripted game with passes and knight moves
    sequences.append(generate_sequence(
        "longer_game",
        ["d1-c1", "c1-b1", "end", "d8-c8", "end", "c1-b3", "c8-b6"]
    ))

    # 8: P2 tensor rotation check
    sequences.append(generate_sequence(
        "p2_tensor_check",
        ["b1-c3"]
    ))

    # --- Random self-play games (reproducible via seed) ---

    # 3 random games with different seeds, capturing full state at every move
    for i, seed in enumerate([42, 123, 999]):
        sequences.append(generate_random_game(
            f"random_game_{i+1}_seed{seed}",
            seed=seed,
            max_moves=80,
        ))

    output = {
        "description": "Cross-validation fixtures for TypeScript engine port",
        "sequences": sequences,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
