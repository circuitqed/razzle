"""
Self-play game generation for training.

Generates games by playing the neural network against itself using MCTS.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import pickle
import numpy as np
from pathlib import Path

from ..core.state import GameState
from ..core.moves import move_to_algebraic
from ..ai.mcts import MCTS, MCTSConfig
from ..ai.network import RazzleNet, NUM_ACTIONS
from ..ai.evaluator import BatchedEvaluator, DummyEvaluator


@dataclass
class GameRecord:
    """Record of a self-play game for training."""
    states: list[np.ndarray]  # State tensors
    policies: list[np.ndarray]  # MCTS policies
    result: float  # Final result: 1.0 = player 0 wins, -1.0 = player 1 wins, 0 = draw
    moves: list[int] = field(default_factory=list)  # Move history

    def training_examples(self) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """
        Convert game record to training examples.

        Returns list of (state, policy, value) tuples.
        Value is from the perspective of the player to move in that state.
        """
        examples = []
        for i, (state, policy) in enumerate(zip(self.states, self.policies)):
            # Player to move alternates (player 0 moves on even indices)
            player_to_move = i % 2
            # Value from perspective of player to move
            if self.result == 0:
                value = 0.0
            elif player_to_move == 0:
                value = self.result
            else:
                value = -self.result
            examples.append((state, policy, value))
        return examples


class SelfPlay:
    """
    Generates training data through self-play.
    """

    def __init__(
        self,
        network: Optional[RazzleNet] = None,
        device: str = 'cpu',
        num_simulations: int = 800,
        temperature_moves: int = 30,  # Use temperature for first N moves
        temperature: float = 1.0,
        batch_size: int = 1
    ):
        if network is not None:
            self.evaluator = BatchedEvaluator(network, batch_size=batch_size, device=device)
        else:
            self.evaluator = DummyEvaluator()

        self.num_simulations = num_simulations
        self.temperature_moves = temperature_moves
        self.temperature = temperature

    def play_game(self, verbose: bool = False) -> GameRecord:
        """
        Play a full game of self-play.

        Returns a GameRecord with states, policies, and result.
        """
        state = GameState.new_game()
        states = []
        policies = []
        moves = []

        move_count = 0

        while not state.is_terminal():
            # Configure MCTS
            temp = self.temperature if move_count < self.temperature_moves else 0.0
            config = MCTSConfig(
                num_simulations=self.num_simulations,
                temperature=temp
            )
            mcts = MCTS(self.evaluator, config)

            # Search
            root = mcts.search(state, add_noise=True)

            # Record state and policy
            states.append(state.to_tensor())
            policies.append(mcts.get_policy(root))

            # Select move
            move = mcts.select_move(root)
            moves.append(move)

            if verbose:
                print(f"Move {move_count + 1}: {move_to_algebraic(move)}")
                print(state)

            # Apply move
            state.apply_move(move)
            move_count += 1

            # Safety limit
            if move_count > 500:
                break

        # Determine result
        winner = state.get_winner()
        if winner == 0:
            result = 1.0
        elif winner == 1:
            result = -1.0
        else:
            result = 0.0

        if verbose:
            print(f"Game over after {move_count} moves. Winner: {winner}")
            print(state)

        return GameRecord(
            states=states,
            policies=policies,
            result=result,
            moves=moves
        )

    def generate_games(
        self,
        num_games: int,
        output_dir: Optional[Path] = None,
        verbose: bool = False
    ) -> list[GameRecord]:
        """
        Generate multiple self-play games.

        If output_dir is provided, saves games incrementally.
        """
        games = []

        for i in range(num_games):
            if verbose:
                print(f"\n=== Game {i + 1}/{num_games} ===")

            game = self.play_game(verbose=verbose)
            games.append(game)

            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                path = output_dir / f"game_{i:05d}.pkl"
                with open(path, 'wb') as f:
                    pickle.dump(game, f)

        return games


def load_games(directory: Path) -> list[GameRecord]:
    """Load all game records from a directory."""
    games = []
    for path in sorted(directory.glob("*.pkl")):
        with open(path, 'rb') as f:
            games.append(pickle.load(f))
    return games


def games_to_training_data(
    games: list[GameRecord]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert list of games to training arrays.

    Returns (states, policies, values) arrays.
    """
    all_examples = []
    for game in games:
        all_examples.extend(game.training_examples())

    states = np.stack([e[0] for e in all_examples])
    policies = np.stack([e[1] for e in all_examples])
    values = np.array([e[2] for e in all_examples], dtype=np.float32)

    return states, policies, values
