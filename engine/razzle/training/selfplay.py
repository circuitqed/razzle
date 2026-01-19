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
    result: float  # Final result: 1.0 = player 0 wins, -1.0 = player 1 wins
    moves: list[int] = field(default_factory=list)  # Move history
    ball_progress: list[tuple[float, float]] = field(default_factory=list)  # Ball advancement per state

    def training_examples(self, use_ball_shaping: bool = True) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """
        Convert game record to training examples.

        Returns list of (state, policy, value) tuples.
        Value is from the perspective of the player to move in that state.

        If use_ball_shaping is True, adds a small bonus based on ball advancement.
        """
        examples = []
        for i, (state, policy) in enumerate(zip(self.states, self.policies)):
            # Player to move alternates (player 0 moves on even indices)
            player_to_move = i % 2

            # Base value from game result
            if self.result == 0:
                base_value = 0.0
            elif player_to_move == 0:
                base_value = self.result
            else:
                base_value = -self.result

            # Add ball advancement shaping (small bonus for progress)
            if use_ball_shaping and self.ball_progress and i < len(self.ball_progress):
                p0_progress, p1_progress = self.ball_progress[i]
                # p0_progress is 0-1 (how far P0's ball has advanced toward row 8)
                # p1_progress is 0-1 (how far P1's ball has advanced toward row 1)
                # Small bonus: 0.1 * (my_progress - opponent_progress)
                if player_to_move == 0:
                    shaping = 0.1 * (p0_progress - p1_progress)
                else:
                    shaping = 0.1 * (p1_progress - p0_progress)
                value = 0.9 * base_value + 0.1 * shaping
            else:
                value = base_value

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
        ball_progress = []

        move_count = 0

        while not state.is_terminal():
            # Track ball positions for reward shaping
            # P0 wants ball to reach row 8 (index 7), starts at row 1 (index 0)
            # P1 wants ball to reach row 1 (index 0), starts at row 8 (index 7)
            p0_ball_row = self._get_ball_row(state, 0)
            p1_ball_row = self._get_ball_row(state, 1)
            # Normalize progress to 0-1 range
            p0_progress = p0_ball_row / 7.0  # 0 = start, 1 = goal
            p1_progress = (7 - p1_ball_row) / 7.0  # 7 = start, 0 = goal -> inverted
            ball_progress.append((p0_progress, p1_progress))

            # Configure MCTS
            temp = self.temperature if move_count < self.temperature_moves else 0.0
            config = MCTSConfig(
                num_simulations=self.num_simulations,
                temperature=temp,
                batch_size=16  # Use batched search for efficiency
            )
            mcts = MCTS(self.evaluator, config)

            # Search with batching for better performance
            root = mcts.search_batched(state, add_noise=True)

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

            # Safety limit - reduced since games should be shorter
            if move_count > 300:
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
            moves=moves,
            ball_progress=ball_progress
        )

    def _get_ball_row(self, state: GameState, player: int) -> int:
        """Get the row (0-7) of the specified player's ball."""
        for sq in range(56):
            if state.balls[player] & (1 << sq):
                return sq // 7
        return 0  # Fallback

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
    games: list[GameRecord],
    use_ball_shaping: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert list of games to training arrays.

    Args:
        games: List of game records
        use_ball_shaping: Add small reward for ball advancement

    Returns (states, policies, values) arrays.
    """
    all_examples = []
    for game in games:
        all_examples.extend(game.training_examples(use_ball_shaping=use_ball_shaping))

    states = np.stack([e[0] for e in all_examples])
    policies = np.stack([e[1] for e in all_examples])
    values = np.array([e[2] for e in all_examples], dtype=np.float32)

    return states, policies, values
