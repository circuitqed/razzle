"""
Training metrics computation for Razzle Dazzle.

Provides functions to compute policy accuracy, value calibration,
and other training quality metrics.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PolicyMetrics:
    """Policy quality metrics."""
    top1_accuracy: float  # % where argmax(pred) == argmax(target)
    top3_accuracy: float  # % where target's top move is in pred's top 3
    entropy: float        # Average entropy of predictions (over legal moves)
    legal_mass: float     # Average probability mass on legal moves
    effective_branching_factor: float  # exp(entropy) - how many moves probability is spread across
    policy_confidence: float  # Average max probability on top legal move


@dataclass
class ValueMetrics:
    """Value prediction metrics."""
    mean: float       # Mean predicted value
    std: float        # Std of predictions
    extremity: float  # Mean of abs(predictions) - how confident


@dataclass
class CalibrationBucket:
    """Single calibration bucket."""
    predicted_mean: float  # Mean predicted value in bucket
    actual_mean: float     # Mean actual outcome in bucket
    count: int             # Number of samples in bucket
    error: float           # Absolute error between predicted and actual


def compute_policy_accuracy(
    pred_logits: np.ndarray | torch.Tensor,
    target_policy: np.ndarray | torch.Tensor,
    legal_mask: Optional[np.ndarray | torch.Tensor] = None,
) -> tuple[float, float]:
    """
    Compute top-1 and top-3 accuracy on legal moves.

    Args:
        pred_logits: Predicted log-probabilities (batch, num_actions)
        target_policy: Target policy distribution (batch, num_actions)
        legal_mask: Optional mask of legal moves (batch, num_actions)

    Returns:
        (top1_accuracy, top3_accuracy) as floats in [0, 1]
    """
    # Convert to numpy if needed
    if isinstance(pred_logits, torch.Tensor):
        pred_logits = pred_logits.detach().cpu().numpy()
    if isinstance(target_policy, torch.Tensor):
        target_policy = target_policy.detach().cpu().numpy()
    if legal_mask is not None and isinstance(legal_mask, torch.Tensor):
        legal_mask = legal_mask.detach().cpu().numpy()

    batch_size = pred_logits.shape[0]
    top1_correct = 0
    top3_correct = 0

    for i in range(batch_size):
        # Get predicted probabilities (exp of log probs)
        pred_probs = np.exp(pred_logits[i])

        # Mask illegal moves if mask provided
        if legal_mask is not None:
            pred_probs = pred_probs * legal_mask[i]

        # Get target's best move (argmax of target policy)
        target_best = np.argmax(target_policy[i])

        # Get prediction's top 3
        pred_top3 = np.argsort(pred_probs)[-3:][::-1]  # Top 3, descending
        pred_best = pred_top3[0]

        # Check accuracy
        if pred_best == target_best:
            top1_correct += 1
        if target_best in pred_top3:
            top3_correct += 1

    return top1_correct / batch_size, top3_correct / batch_size


def compute_policy_metrics(
    pred_logits: np.ndarray | torch.Tensor,
    target_policy: np.ndarray | torch.Tensor,
    legal_mask: Optional[np.ndarray | torch.Tensor] = None,
) -> PolicyMetrics:
    """
    Compute comprehensive policy metrics.

    Args:
        pred_logits: Predicted log-probabilities (batch, num_actions)
        target_policy: Target policy distribution (batch, num_actions)
        legal_mask: Optional mask of legal moves (batch, num_actions)

    Returns:
        PolicyMetrics dataclass
    """
    # Convert to numpy if needed
    if isinstance(pred_logits, torch.Tensor):
        pred_logits = pred_logits.detach().cpu().numpy()
    if isinstance(target_policy, torch.Tensor):
        target_policy = target_policy.detach().cpu().numpy()
    if legal_mask is not None and isinstance(legal_mask, torch.Tensor):
        legal_mask = legal_mask.detach().cpu().numpy()

    batch_size = pred_logits.shape[0]

    # Compute accuracies
    top1_acc, top3_acc = compute_policy_accuracy(pred_logits, target_policy, legal_mask)

    # Compute entropy, legal mass, and confidence
    total_entropy = 0.0
    total_legal_mass = 0.0
    total_confidence = 0.0

    for i in range(batch_size):
        pred_probs = np.exp(pred_logits[i])
        pred_probs = np.clip(pred_probs, 1e-10, 1.0)  # Avoid log(0)

        # Legal mass and confidence
        if legal_mask is not None:
            legal_mass = np.sum(pred_probs * legal_mask[i])

            # Entropy over legal moves only (renormalized)
            legal_probs = pred_probs * legal_mask[i]
            legal_probs = legal_probs / (legal_probs.sum() + 1e-10)  # Renormalize
            legal_probs = np.clip(legal_probs, 1e-10, 1.0)
            entropy = -np.sum(legal_probs * np.log(legal_probs) * (legal_probs > 1e-9))

            # Policy confidence: max probability on legal moves (after renormalization)
            confidence = np.max(legal_probs)
        else:
            legal_mass = 1.0  # No mask means all legal
            entropy = -np.sum(pred_probs * np.log(pred_probs))
            confidence = np.max(pred_probs)

        total_entropy += entropy
        total_legal_mass += legal_mass
        total_confidence += confidence

    avg_entropy = total_entropy / batch_size

    return PolicyMetrics(
        top1_accuracy=top1_acc,
        top3_accuracy=top3_acc,
        entropy=avg_entropy,
        legal_mass=total_legal_mass / batch_size,
        effective_branching_factor=float(np.exp(avg_entropy)),
        policy_confidence=total_confidence / batch_size,
    )


def compute_value_metrics(
    pred_values: np.ndarray | torch.Tensor,
) -> ValueMetrics:
    """
    Compute value prediction metrics.

    Args:
        pred_values: Predicted values (batch,) in [-1, 1]

    Returns:
        ValueMetrics dataclass
    """
    if isinstance(pred_values, torch.Tensor):
        pred_values = pred_values.detach().cpu().numpy()

    # Flatten if needed
    pred_values = pred_values.flatten()

    return ValueMetrics(
        mean=float(np.mean(pred_values)),
        std=float(np.std(pred_values)),
        extremity=float(np.mean(np.abs(pred_values))),
    )


def compute_value_calibration(
    pred_values: np.ndarray | torch.Tensor,
    actual_outcomes: np.ndarray | torch.Tensor,
    num_bins: int = 10,
) -> list[CalibrationBucket]:
    """
    Compute value calibration metrics.

    Bins predictions into ranges and compares average predicted value
    to average actual outcome in each bin.

    Args:
        pred_values: Predicted values (batch,) in [-1, 1]
        actual_outcomes: Actual game outcomes (batch,) in [-1, 1]
        num_bins: Number of calibration bins

    Returns:
        List of CalibrationBucket for non-empty bins
    """
    if isinstance(pred_values, torch.Tensor):
        pred_values = pred_values.detach().cpu().numpy()
    if isinstance(actual_outcomes, torch.Tensor):
        actual_outcomes = actual_outcomes.detach().cpu().numpy()

    # Flatten
    pred_values = pred_values.flatten()
    actual_outcomes = actual_outcomes.flatten()

    # Create bins from -1 to 1
    bins = np.linspace(-1, 1, num_bins + 1)
    calibration = []

    for i in range(num_bins):
        lower, upper = bins[i], bins[i + 1]

        # Handle edge case for last bin (include upper bound)
        if i == num_bins - 1:
            mask = (pred_values >= lower) & (pred_values <= upper)
        else:
            mask = (pred_values >= lower) & (pred_values < upper)

        count = np.sum(mask)
        if count > 0:
            pred_mean = float(np.mean(pred_values[mask]))
            actual_mean = float(np.mean(actual_outcomes[mask]))
            calibration.append(CalibrationBucket(
                predicted_mean=pred_mean,
                actual_mean=actual_mean,
                count=int(count),
                error=abs(pred_mean - actual_mean),
            ))

    return calibration


def compute_calibration_error(calibration: list[CalibrationBucket]) -> float:
    """
    Compute weighted mean absolute calibration error.

    Args:
        calibration: List of calibration buckets

    Returns:
        Weighted MAE (weighted by bucket size)
    """
    if not calibration:
        return 0.0

    total_error = 0.0
    total_count = 0

    for bucket in calibration:
        total_error += bucket.error * bucket.count
        total_count += bucket.count

    return total_error / total_count if total_count > 0 else 0.0


def format_calibration_table(calibration: list[CalibrationBucket]) -> str:
    """
    Format calibration data as a readable table.

    Args:
        calibration: List of calibration buckets

    Returns:
        Formatted string table
    """
    lines = [
        "Value Calibration Table",
        "=" * 50,
        f"{'Predicted':>12} {'Actual':>12} {'Error':>10} {'Count':>8}",
        "-" * 50,
    ]

    for bucket in calibration:
        lines.append(
            f"{bucket.predicted_mean:>+12.3f} {bucket.actual_mean:>+12.3f} "
            f"{bucket.error:>10.3f} {bucket.count:>8}"
        )

    lines.append("-" * 50)
    mae = compute_calibration_error(calibration)
    lines.append(f"Weighted MAE: {mae:.4f}")

    return "\n".join(lines)


def _is_pass_move(state, move: int) -> bool:
    """Check if a move is a ball pass (ball at src, piece at dst)."""
    from ..core.bitboard import bit

    if move == -1:
        return False
    src = move // 56
    dst = move % 56
    p = state.current_player
    return bool((state.balls[p] & bit(src)) and (state.pieces[p] & bit(dst)))


def _count_turns(move_sequence: list[int]) -> int:
    """
    Count turns (player switches) in a move sequence.

    A turn ends when:
    - A knight move is made (not a pass)
    - END_TURN (-1) is played

    This requires replaying the game to distinguish passes from knight moves.
    """
    from ..core.state import GameState

    state = GameState.new_game()
    turns = 0

    for move in move_sequence:
        if move == -1:  # END_TURN
            turns += 1
        elif not _is_pass_move(state, move):
            turns += 1  # Knight move ends a turn

        state.apply_move(move)

    return turns


def _get_first_n_turns(move_sequence: list[int], n: int) -> tuple:
    """
    Extract moves comprising the first N turns.

    Returns a tuple of moves that can be used as an opening fingerprint.
    Each turn includes all passes plus the final knight move or END_TURN.
    """
    from ..core.state import GameState

    state = GameState.new_game()
    turns = 0
    opening_moves = []

    for move in move_sequence:
        if turns >= n:
            break

        opening_moves.append(move)

        if move == -1:  # END_TURN
            turns += 1
        elif not _is_pass_move(state, move):
            turns += 1  # Knight move ends a turn

        state.apply_move(move)

    return tuple(opening_moves)


def compute_pass_stats(move_sequence: list[int]) -> dict:
    """
    Compute pass-related statistics for a game.

    Returns:
        Dictionary with:
        - pass_decisions: turns that included at least one pass
        - knight_decisions: turns with no passes (just a knight move)
        - total_passes: total number of individual pass actions
        - pass_decision_rate: pass_decisions / total_decisions
    """
    from ..core.state import GameState

    state = GameState.new_game()
    pass_decisions = 0
    knight_decisions = 0
    total_passes = 0
    current_turn_has_pass = False

    for move in move_sequence:
        if move == -1:  # END_TURN
            # Turn ends - must have had passes to need END_TURN
            pass_decisions += 1
            current_turn_has_pass = False
        elif _is_pass_move(state, move):
            total_passes += 1
            current_turn_has_pass = True
        else:
            # Knight move - ends the turn
            if current_turn_has_pass:
                pass_decisions += 1
            else:
                knight_decisions += 1
            current_turn_has_pass = False

        state.apply_move(move)

    total_decisions = pass_decisions + knight_decisions
    pass_decision_rate = pass_decisions / total_decisions if total_decisions > 0 else 0.0

    return {
        'pass_decisions': pass_decisions,
        'knight_decisions': knight_decisions,
        'total_passes': total_passes,
        'pass_decision_rate': pass_decision_rate,
    }


def compute_game_diversity(
    move_sequences: list[list[int]],
    depth: int = 5,
) -> dict:
    """
    Compute game diversity metrics.

    Args:
        move_sequences: List of move sequences from multiple games
        depth: Number of turns to consider for opening diversity

    Returns:
        Dictionary with diversity metrics
    """
    if not move_sequences:
        return {
            'total_games': 0,
            'unique_openings': 0,
            'opening_diversity': 0.0,
            'avg_game_length': 0.0,
            'game_length_std': 0.0,
        }

    # Game length statistics (in turns, not atomic actions)
    lengths = [_count_turns(seq) for seq in move_sequences]
    avg_length = np.mean(lengths)
    std_length = np.std(lengths) if len(lengths) > 1 else 0.0

    # Opening diversity (unique sequences of first N turns)
    openings = set()
    for seq in move_sequences:
        opening = _get_first_n_turns(seq, depth)
        openings.add(opening)

    unique_openings = len(openings)
    opening_diversity = unique_openings / len(move_sequences)

    return {
        'total_games': len(move_sequences),
        'unique_openings': unique_openings,
        'opening_diversity': opening_diversity,
        'avg_game_length': float(avg_length),
        'game_length_std': float(std_length),
    }


def compute_batch_metrics(
    network,
    states: torch.Tensor,
    policies: torch.Tensor,
    values: torch.Tensor,
    legal_masks: Optional[torch.Tensor] = None,
) -> dict:
    """
    Compute all training metrics for a batch.

    This is a convenience function for use in the training loop.

    Args:
        network: RazzleNet model
        states: Input states (batch, 7, 8, 7)
        policies: Target policies (batch, num_actions)
        values: Target values (batch,)
        legal_masks: Optional legal move masks (batch, num_actions)

    Returns:
        Dictionary of metrics
    """
    network.eval()
    with torch.no_grad():
        pred_logits, pred_values, pred_difficulty = network(states)
        pred_values = pred_values.squeeze(-1)

    # Policy metrics
    policy_metrics = compute_policy_metrics(pred_logits, policies, legal_masks)

    # Value metrics
    value_metrics = compute_value_metrics(pred_values)

    return {
        # Policy
        'policy_top1_accuracy': policy_metrics.top1_accuracy,
        'policy_top3_accuracy': policy_metrics.top3_accuracy,
        'policy_entropy': policy_metrics.entropy,
        'policy_legal_mass': policy_metrics.legal_mass,
        # Value
        'value_mean': value_metrics.mean,
        'value_std': value_metrics.std,
        'value_extremity': value_metrics.extremity,
    }
