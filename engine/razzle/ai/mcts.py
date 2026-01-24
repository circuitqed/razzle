"""
Monte Carlo Tree Search for Razzle Dazzle.

Implements PUCT (Polynomial Upper Confidence Trees) as used in AlphaZero.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Protocol, TYPE_CHECKING
import math
import time
import numpy as np

from ..core.state import GameState
from ..core.moves import get_legal_moves, decode_move, move_to_algebraic
from .network import NUM_ACTIONS, END_TURN_ACTION

if TYPE_CHECKING:
    from .time_manager import TimeManager


class Evaluator(Protocol):
    """Protocol for position evaluators."""
    def evaluate(self, state: GameState) -> tuple[np.ndarray, float]:
        """Return (policy, value) for state."""
        ...

    def evaluate_with_difficulty(self, state: GameState) -> tuple[np.ndarray, float, float]:
        """Return (policy, value, difficulty) for state."""
        ...


@dataclass
class MCTSConfig:
    """Configuration for MCTS."""
    num_simulations: int = 800
    c_puct: float = 1.5  # Exploration constant
    dirichlet_alpha: float = 0.3  # Noise for root exploration
    dirichlet_epsilon: float = 0.25  # Weight of noise at root
    temperature: float = 1.0  # Temperature for move selection
    batch_size: int = 32  # Number of parallel simulations for batched search
    virtual_loss: int = 3  # Virtual loss to encourage exploration in parallel search

    # Early termination settings
    early_termination: bool = True  # Enable early stopping when search converges
    visit_concentration_threshold: float = 0.85  # Stop if top move has this fraction of visits
    min_sims_for_early_stop: int = 100  # Minimum sims before early termination allowed
    check_early_stop_interval: int = 50  # Check every N simulations

    # Pass quiescence: continue search through pass sequences to avoid evaluating mid-turn
    pass_quiescence: bool = True  # Don't evaluate mid-pass positions
    pass_quiescence_max_depth: int = 10  # Max passes to extend (safety limit)


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

    def ucb_score(self, parent_visits: int, c_puct: float, parent_player: int) -> float:
        """
        Compute PUCT score for node selection.

        UCB = Q + c_puct * P * sqrt(parent_visits) / (1 + child_visits)

        Note: Values are stored from each node's current_player perspective.
        When turn switches (knight move), we negate because opponent's loss = our gain.
        When turn stays (ball pass), value is already from our perspective.
        """
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.adjusted_visit_count)
        if self.state.current_player != parent_player:
            return -self.value + exploration
        else:
            return self.value + exploration

    def select_child(self, c_puct: float) -> tuple[int, Node]:
        """Select best child according to PUCT."""
        best_score = float('-inf')
        best_action = -1
        best_child = None

        parent_visits = self.adjusted_visit_count
        parent_player = self.state.current_player

        for action, child in self.children.items():
            score = child.ucb_score(parent_visits, c_puct, parent_player)
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

        # Helper to get prior for a move
        def get_prior(move: int) -> float:
            if move == -1:  # END_TURN_MOVE
                # END_TURN is at index END_TURN_ACTION (3136) in policy array
                return policy[END_TURN_ACTION]
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

    def _quiescence_search(self, node: Node, depth: int = 0) -> float:
        """
        Search through pass sequence to find best outcome.

        Since the same player is moving throughout a pass sequence,
        we take the max value over all branches. This ensures we
        properly evaluate positions where a pass chain leads to a win.

        Args:
            node: Current node (must be expanded)
            depth: Current depth in quiescence search

        Returns:
            Best value achievable from this position
        """
        # Safety limit to prevent infinite loops
        if depth >= self.config.pass_quiescence_max_depth:
            _, value = self.evaluator.evaluate(node.state)
            return value

        # Terminal state - return actual result
        if node.state.is_terminal():
            result = node.state.get_result(node.state.current_player)
            return 2 * result - 1  # Convert [0,1] to [-1,1]

        # Turn ended (has_passed=False) - evaluate this position
        if not node.state.has_passed:
            _, value = self.evaluator.evaluate(node.state)
            return value

        # Mid-pass: search all children and take max
        # (same player is still moving, so we want the best outcome)
        if not node.children:
            # No children somehow - evaluate here
            _, value = self.evaluator.evaluate(node.state)
            return value

        best_value = float('-inf')
        for action, child in node.children.items():
            # Expand child if needed
            if not child.is_expanded and not child.state.is_terminal():
                policy, _ = self.evaluator.evaluate(child.state)
                child.expand(policy)

            child_value = self._quiescence_search(child, depth + 1)
            best_value = max(best_value, child_value)

        return best_value

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

        # EVALUATE (with pass quiescence)
        if node.state.is_terminal():
            # Terminal node - use actual game result
            leaf_player = node.state.current_player
            value = node.state.get_result(leaf_player)
            value = 2 * value - 1  # Convert from [0,1] to [-1,1]
        else:
            # Expand the node first
            policy, value = self.evaluator.evaluate(node.state)
            node.expand(policy)

            # Pass quiescence: if we're mid-pass, search all continuations
            # and take the max value (same player is moving throughout)
            if self.config.pass_quiescence and node.state.has_passed:
                value = self._quiescence_search(node, depth=0)

            leaf_player = node.state.current_player

        # BACKUP: propagate value up the tree
        # Value is from perspective of leaf's current player
        # Only flip sign when the player actually changes between nodes
        for n in reversed(path):
            n.visit_count += 1
            # Value sign depends on whether this node's player matches leaf's player
            if n.state.current_player == leaf_player:
                n.value_sum += value
            else:
                n.value_sum -= value

    def get_policy(self, root: Node) -> np.ndarray:
        """
        Get move probabilities based on visit counts.

        Returns array of shape (NUM_ACTIONS,) with probabilities.
        END_TURN (-1) is mapped to index END_TURN_ACTION (3136).
        """
        policy = np.zeros(NUM_ACTIONS, dtype=np.float32)

        # Map action to policy index (END_TURN -1 -> END_TURN_ACTION)
        def action_to_index(action: int) -> int:
            return END_TURN_ACTION if action == -1 else action

        total_visits = sum(c.visit_count for c in root.children.values())

        if total_visits > 0 and self.config.temperature > 0:
            if self.config.temperature == 1.0:
                # Standard proportional to visits
                for action, child in root.children.items():
                    policy[action_to_index(action)] = child.visit_count / total_visits
            else:
                # Apply temperature
                actions = list(root.children.keys())
                visits = np.array([root.children[a].visit_count for a in actions], dtype=np.float32)
                visits = np.power(visits, 1.0 / self.config.temperature)
                visits_sum = visits.sum()
                for i, action in enumerate(actions):
                    policy[action_to_index(action)] = visits[i] / visits_sum
        elif root.children:
            # Greedy (temperature = 0)
            best_action = max(root.children.keys(), key=lambda a: root.children[a].visit_count)
            policy[action_to_index(best_action)] = 1.0

        return policy

    def select_move(self, root: Node) -> int:
        """Select move based on visit counts and temperature."""
        if not root.children:
            raise ValueError("No legal moves")

        actions = list(root.children.keys())
        visits = np.array([root.children[a].visit_count for a in actions], dtype=np.float32)

        if self.config.temperature == 0:
            # Greedy - pick highest visit count
            return actions[np.argmax(visits)]
        else:
            # Sample proportional to visit counts (with temperature)
            if visits.sum() == 0:
                # No visits yet - uniform random
                probs = np.ones(len(actions)) / len(actions)
            elif self.config.temperature == 1.0:
                probs = visits / visits.sum()
            else:
                # Apply temperature
                visits_temp = np.power(visits, 1.0 / self.config.temperature)
                probs = visits_temp / visits_temp.sum()

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
        leaf_players: list[int] = []
        terminal_paths: list[tuple[list[Node], float, int]] = []  # (path, value, leaf_player)

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
                leaf_player = node.state.current_player
                value = node.state.get_result(leaf_player)
                value = 2 * value - 1  # Convert from [0,1] to [-1,1]
                terminal_paths.append((path, value, leaf_player))
            else:
                paths.append(path)
                leaves.append(node)
                leaf_players.append(node.state.current_player)

        # EVALUATE: batch evaluate all non-terminal leaves
        if leaves:
            states = [leaf.state for leaf in leaves]
            results = self.evaluator.evaluate_batch(states)

            # EXPAND and BACKUP for each leaf
            for path, leaf, leaf_player, (policy, value) in zip(paths, leaves, leaf_players, results):
                # Expand leaf
                if not leaf.is_expanded:
                    leaf.expand(policy)

                # Backup value and remove virtual loss
                # Only flip sign when player changes, not at every level
                for n in reversed(path):
                    n.visit_count += 1
                    n.virtual_loss -= self.config.virtual_loss
                    if n.state.current_player == leaf_player:
                        n.value_sum += value
                    else:
                        n.value_sum -= value

        # BACKUP terminal paths
        for path, value, leaf_player in terminal_paths:
            for n in reversed(path):
                n.visit_count += 1
                n.virtual_loss -= self.config.virtual_loss
                if n.state.current_player == leaf_player:
                    n.value_sum += value
                else:
                    n.value_sum -= value

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

    def _should_stop_early(self, root: Node, sims_done: int) -> bool:
        """
        Check if we should stop search early due to convergence.

        Returns True if:
        - Early termination is enabled
        - Minimum simulations have been done
        - Top move has enough visit concentration
        """
        if not self.config.early_termination:
            return False

        if sims_done < self.config.min_sims_for_early_stop:
            return False

        if not root.children:
            return True  # No children = terminal state

        # Calculate visit concentration
        total_visits = sum(c.visit_count for c in root.children.values())
        if total_visits == 0:
            return False

        max_visits = max(c.visit_count for c in root.children.values())
        concentration = max_visits / total_visits

        return concentration >= self.config.visit_concentration_threshold

    def search_timed(
        self,
        state: GameState,
        time_manager: 'TimeManager',
        add_noise: bool = True
    ) -> tuple[Node, dict]:
        """
        Run MCTS with dynamic time management based on position difficulty.

        Uses the neural network's difficulty prediction to allocate simulations:
        - Easy positions (low difficulty) get fewer simulations
        - Hard positions (high difficulty) get more simulations

        Args:
            state: Current game state
            time_manager: TimeManager for tracking time budget
            add_noise: Add Dirichlet noise at root for exploration

        Returns:
            (root_node, info_dict) where info_dict contains search statistics
        """
        start_time = time.time()

        root = Node(state=state.copy())

        # Get evaluation with difficulty prediction
        if hasattr(self.evaluator, 'evaluate_with_difficulty'):
            policy, value, difficulty = self.evaluator.evaluate_with_difficulty(root.state)
        else:
            # Fallback for evaluators without difficulty
            policy, value = self.evaluator.evaluate(root.state)
            difficulty = 0.5  # Neutral difficulty

        root.expand(policy)

        if add_noise and self.config.dirichlet_epsilon > 0:
            root.add_dirichlet_noise(
                self.config.dirichlet_alpha,
                self.config.dirichlet_epsilon
            )

        # Determine target simulations based on difficulty
        target_sims = time_manager.get_target_simulations(difficulty)

        # Run simulations in batches
        sims_done = 0
        batch_size = self.config.batch_size
        early_stopped = False

        while sims_done < target_sims:
            current_batch = min(batch_size, target_sims - sims_done)
            self._simulate_batch(root, current_batch)
            sims_done += current_batch

            # Check for early termination
            if (sims_done % self.config.check_early_stop_interval == 0 and
                self._should_stop_early(root, sims_done)):
                early_stopped = True
                break

        elapsed = time.time() - start_time

        # Update time manager
        time_manager.update(elapsed, sims_done)

        info = {
            'difficulty': difficulty,
            'target_simulations': target_sims,
            'simulations_done': sims_done,
            'elapsed_time': elapsed,
            'early_stopped': early_stopped,
            'sims_per_second': sims_done / elapsed if elapsed > 0 else 0,
        }

        return root, info


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


def play_move_timed(
    state: GameState,
    evaluator: Evaluator,
    time_manager: 'TimeManager',
    temperature: float = 0.0,
    batch_size: int = 16
) -> tuple[int, Node, dict]:
    """
    Play a single move using MCTS with dynamic time management.

    The number of simulations is determined by the time manager based on:
    - Remaining time budget
    - Position difficulty (from neural network)

    Args:
        state: Current game state
        evaluator: Position evaluator (must support evaluate_with_difficulty)
        time_manager: TimeManager tracking game time
        temperature: Temperature for move selection (0 = greedy)
        batch_size: Number of parallel simulations per batch

    Returns (move, root_node, info_dict).
    """
    config = MCTSConfig(
        num_simulations=time_manager.config.max_simulations,  # Will be limited by time
        temperature=temperature,
        batch_size=batch_size
    )
    mcts = MCTS(evaluator, config)

    root, info = mcts.search_timed(state, time_manager, add_noise=False)
    move = mcts.select_move(root)

    return move, root, info
