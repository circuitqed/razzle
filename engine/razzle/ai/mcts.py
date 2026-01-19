"""
Monte Carlo Tree Search for Razzle Dazzle.

Implements PUCT (Polynomial Upper Confidence Trees) as used in AlphaZero.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Protocol
import math
import numpy as np

from ..core.state import GameState
from ..core.moves import get_legal_moves, decode_move, move_to_algebraic
from .network import NUM_ACTIONS


class Evaluator(Protocol):
    """Protocol for position evaluators."""
    def evaluate(self, state: GameState) -> tuple[np.ndarray, float]:
        """Return (policy, value) for state."""
        ...


@dataclass
class MCTSConfig:
    """Configuration for MCTS."""
    num_simulations: int = 800
    c_puct: float = 1.5  # Exploration constant
    dirichlet_alpha: float = 0.3  # Noise for root exploration
    dirichlet_epsilon: float = 0.25  # Weight of noise at root
    temperature: float = 1.0  # Temperature for move selection
    batch_size: int = 8  # Number of parallel simulations for batched search
    virtual_loss: int = 3  # Virtual loss to encourage exploration in parallel search


@dataclass
class Node:
    """A node in the MCTS tree."""
    state: GameState
    parent: Optional[Node] = None
    parent_action: Optional[int] = None
    prior: float = 0.0

    # Statistics
    visit_count: int = 0
    value_sum: float = 0.0
    virtual_loss: int = 0  # For parallel search

    # Children (lazily initialized)
    children: dict[int, Node] = field(default_factory=dict)
    is_expanded: bool = False

    @property
    def value(self) -> float:
        """Average value (Q) of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def adjusted_visit_count(self) -> int:
        """Visit count including virtual losses."""
        return self.visit_count + self.virtual_loss

    def ucb_score(self, parent_visits: int, c_puct: float) -> float:
        """
        Compute PUCT score for node selection.

        UCB = Q + c_puct * P * sqrt(parent_visits) / (1 + child_visits)
        """
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.adjusted_visit_count)
        return self.value + exploration

    def select_child(self, c_puct: float) -> tuple[int, Node]:
        """Select best child according to PUCT."""
        best_score = float('-inf')
        best_action = -1
        best_child = None

        parent_visits = self.adjusted_visit_count

        for action, child in self.children.items():
            score = child.ucb_score(parent_visits, c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, policy: np.ndarray) -> None:
        """Expand node with children based on policy."""
        if self.is_expanded:
            return

        legal_moves = get_legal_moves(self.state)

        # Helper to get prior for a move, handling END_TURN (-1) specially
        def get_prior(move: int) -> float:
            if move == -1:  # END_TURN_MOVE
                # End turn gets a small fixed prior - MCTS will adjust via visits
                return 0.1
            return policy[move]

        # Mask and renormalize policy
        policy_sum = sum(get_prior(m) for m in legal_moves)
        if policy_sum > 0:
            for move in legal_moves:
                child_state = self.state.copy()
                child_state.apply_move(move)
                prior = get_prior(move) / policy_sum
                self.children[move] = Node(
                    state=child_state,
                    parent=self,
                    parent_action=move,
                    prior=prior
                )
        else:
            # Uniform if policy is all zeros
            uniform = 1.0 / len(legal_moves) if legal_moves else 0.0
            for move in legal_moves:
                child_state = self.state.copy()
                child_state.apply_move(move)
                self.children[move] = Node(
                    state=child_state,
                    parent=self,
                    parent_action=move,
                    prior=uniform
                )

        self.is_expanded = True

    def add_dirichlet_noise(self, alpha: float, epsilon: float) -> None:
        """Add Dirichlet noise to root node priors for exploration."""
        if not self.children:
            return

        actions = list(self.children.keys())
        noise = np.random.dirichlet([alpha] * len(actions))

        for i, action in enumerate(actions):
            child = self.children[action]
            child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]


class MCTS:
    """
    Monte Carlo Tree Search with PUCT selection.

    Supports both synchronous single-threaded and batched evaluation.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        config: Optional[MCTSConfig] = None
    ):
        self.evaluator = evaluator
        self.config = config or MCTSConfig()

    def search(self, state: GameState, add_noise: bool = True) -> Node:
        """
        Run MCTS from given state.

        Returns the root node with visit statistics.
        """
        root = Node(state=state.copy())

        # Evaluate and expand root
        policy, value = self.evaluator.evaluate(root.state)
        root.expand(policy)

        if add_noise and self.config.dirichlet_epsilon > 0:
            root.add_dirichlet_noise(
                self.config.dirichlet_alpha,
                self.config.dirichlet_epsilon
            )

        # Run simulations
        for _ in range(self.config.num_simulations):
            self._simulate(root)

        return root

    def _simulate(self, root: Node) -> None:
        """Run one MCTS simulation."""
        node = root
        path = [node]

        # SELECT: traverse tree to leaf
        while node.is_expanded and node.children:
            if node.state.is_terminal():
                break
            _, node = node.select_child(self.config.c_puct)
            path.append(node)

        # EVALUATE
        if node.state.is_terminal():
            # Terminal node - use actual game result
            value = node.state.get_result(node.state.current_player)
            value = 2 * value - 1  # Convert from [0,1] to [-1,1]
        else:
            # Expand and evaluate with neural network
            policy, value = self.evaluator.evaluate(node.state)
            node.expand(policy)

        # BACKUP: propagate value up the tree
        # Value is from perspective of node's current player
        # As we go up, we flip perspective
        for i, n in enumerate(reversed(path)):
            n.visit_count += 1
            # Flip value at each level (my good = parent bad)
            sign = 1 if (i % 2 == 0) else -1
            n.value_sum += sign * value

    def get_policy(self, root: Node) -> np.ndarray:
        """
        Get move probabilities based on visit counts.

        Returns array of shape (3136,) with probabilities.
        Note: END_TURN (-1) is not included in the policy array.
        """
        policy = np.zeros(NUM_ACTIONS, dtype=np.float32)

        # Filter out END_TURN for policy array (it has no slot)
        valid_children = {a: c for a, c in root.children.items() if a >= 0}
        total_visits = sum(c.visit_count for c in valid_children.values())

        if total_visits > 0 and self.config.temperature > 0:
            if self.config.temperature == 1.0:
                # Standard proportional to visits
                for action, child in valid_children.items():
                    policy[action] = child.visit_count / total_visits
            else:
                # Apply temperature
                visits = np.array([c.visit_count for c in valid_children.values()], dtype=np.float32)
                visits = np.power(visits, 1.0 / self.config.temperature)
                visits_sum = visits.sum()
                for i, (action, child) in enumerate(valid_children.items()):
                    policy[action] = visits[i] / visits_sum
        elif valid_children:
            # Greedy (temperature = 0)
            best_action = max(valid_children.keys(), key=lambda a: valid_children[a].visit_count)
            policy[best_action] = 1.0

        return policy

    def select_move(self, root: Node) -> int:
        """Select move based on visit counts and temperature."""
        if not root.children:
            raise ValueError("No legal moves")

        if self.config.temperature == 0:
            # Greedy
            return max(root.children.keys(), key=lambda a: root.children[a].visit_count)
        else:
            # Sample according to policy
            policy = self.get_policy(root)
            actions = list(root.children.keys())
            probs = np.array([policy[a] for a in actions])
            probs = probs / probs.sum()  # Normalize to handle floating point errors
            return np.random.choice(actions, p=probs)

    def get_best_move(self, state: GameState, add_noise: bool = False) -> int:
        """Convenience method: search and return best move."""
        root = self.search(state, add_noise=add_noise)
        return self.select_move(root)

    def search_batched(self, state: GameState, add_noise: bool = True) -> Node:
        """
        Run batched MCTS from given state.

        Uses virtual loss to select multiple leaves in parallel,
        then batch evaluates them for better GPU utilization.

        Returns the root node with visit statistics.
        """
        root = Node(state=state.copy())

        # Evaluate and expand root
        policy, value = self.evaluator.evaluate(root.state)
        root.expand(policy)

        if add_noise and self.config.dirichlet_epsilon > 0:
            root.add_dirichlet_noise(
                self.config.dirichlet_alpha,
                self.config.dirichlet_epsilon
            )

        # Run simulations in batches
        remaining = self.config.num_simulations
        batch_size = self.config.batch_size

        while remaining > 0:
            current_batch = min(batch_size, remaining)
            self._simulate_batch(root, current_batch)
            remaining -= current_batch

        return root

    def _simulate_batch(self, root: Node, batch_size: int) -> None:
        """Run a batch of MCTS simulations with virtual loss."""
        paths: list[list[Node]] = []
        leaves: list[Node] = []
        terminal_paths: list[tuple[list[Node], float]] = []

        # SELECT: traverse tree to find multiple leaves
        for _ in range(batch_size):
            node = root
            path = [node]

            # Apply virtual loss as we descend
            while node.is_expanded and node.children:
                if node.state.is_terminal():
                    break
                node.virtual_loss += self.config.virtual_loss
                _, node = node.select_child(self.config.c_puct)
                path.append(node)

            # Apply virtual loss to leaf
            node.virtual_loss += self.config.virtual_loss

            if node.state.is_terminal():
                # Terminal node - record for later backup
                value = node.state.get_result(node.state.current_player)
                value = 2 * value - 1  # Convert from [0,1] to [-1,1]
                terminal_paths.append((path, value))
            else:
                paths.append(path)
                leaves.append(node)

        # EVALUATE: batch evaluate all non-terminal leaves
        if leaves:
            states = [leaf.state for leaf in leaves]
            results = self.evaluator.evaluate_batch(states)

            # EXPAND and BACKUP for each leaf
            for path, leaf, (policy, value) in zip(paths, leaves, results):
                # Expand leaf
                if not leaf.is_expanded:
                    leaf.expand(policy)

                # Backup value and remove virtual loss
                for i, n in enumerate(reversed(path)):
                    n.visit_count += 1
                    n.virtual_loss -= self.config.virtual_loss
                    # Flip value at each level
                    sign = 1 if (i % 2 == 0) else -1
                    n.value_sum += sign * value

        # BACKUP terminal paths
        for path, value in terminal_paths:
            for i, n in enumerate(reversed(path)):
                n.visit_count += 1
                n.virtual_loss -= self.config.virtual_loss
                sign = 1 if (i % 2 == 0) else -1
                n.value_sum += sign * value

    def analyze(self, root: Node, top_k: int = 5) -> list[dict]:
        """
        Analyze search results.

        Returns list of top moves with statistics.
        """
        moves = []
        for action, child in root.children.items():
            moves.append({
                'move': action,
                'algebraic': move_to_algebraic(action),
                'visits': child.visit_count,
                'value': child.value,
                'prior': child.prior
            })

        moves.sort(key=lambda m: m['visits'], reverse=True)
        return moves[:top_k]


def play_move(
    state: GameState,
    evaluator: Evaluator,
    num_simulations: int = 800,
    temperature: float = 0.0,
    batched: bool = False,
    batch_size: int = 8
) -> tuple[int, Node]:
    """
    Play a single move using MCTS.

    Args:
        state: Current game state
        evaluator: Position evaluator (neural network or dummy)
        num_simulations: Number of MCTS simulations
        temperature: Temperature for move selection (0 = greedy)
        batched: Use batched search for better GPU utilization
        batch_size: Number of parallel simulations per batch

    Returns (move, root_node).
    """
    config = MCTSConfig(
        num_simulations=num_simulations,
        temperature=temperature,
        batch_size=batch_size
    )
    mcts = MCTS(evaluator, config)

    if batched:
        root = mcts.search_batched(state, add_noise=False)
    else:
        root = mcts.search(state, add_noise=False)

    move = mcts.select_move(root)
    return move, root
