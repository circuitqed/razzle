"""
Board symmetry transformations for Razzle Dazzle.

The board is 8 rows x 7 columns. We support:
- 180° rotation: Normalizes player perspective (both players see "advance toward far rank")
- Horizontal flip: Data augmentation (mirror left/right)

180° rotation is used to normalize the board so both players see the same
spatial pattern - their pieces start at the "bottom" and the goal is at the "top".
This reduces the learning problem since the network doesn't need to learn
two separate perspectives.
"""

import numpy as np
from functools import lru_cache

from .bitboard import ROWS, COLS, NUM_SQUARES

# Total actions: 56*56 + 1 (END_TURN)
TOTAL_SQUARES = NUM_SQUARES  # 56
NUM_ACTIONS = TOTAL_SQUARES * TOTAL_SQUARES + 1  # 3137
END_TURN_INDEX = NUM_ACTIONS - 1  # 3136


def rotate_square_180(sq: int) -> int:
    """
    Rotate a square index 180 degrees.

    Square (row, col) becomes (7-row, 6-col).
    This is equivalent to viewing the board from the opposite side.

    Args:
        sq: Square index (0-55)

    Returns:
        Rotated square index
    """
    row = sq // COLS
    col = sq % COLS
    new_row = ROWS - 1 - row
    new_col = COLS - 1 - col
    return new_row * COLS + new_col


def flip_square_horizontal(sq: int) -> int:
    """
    Flip a square horizontally (mirror across vertical axis).

    Square (row, col) becomes (row, 6-col).

    Args:
        sq: Square index (0-55)

    Returns:
        Flipped square index
    """
    row = sq // COLS
    col = sq % COLS
    new_col = COLS - 1 - col
    return row * COLS + new_col


def rotate_move_180(move: int) -> int:
    """
    Rotate a move index 180 degrees.

    Move is encoded as src * 56 + dst.

    Args:
        move: Move index (0-3136)

    Returns:
        Rotated move index
    """
    if move == END_TURN_INDEX:
        return move
    src = move // TOTAL_SQUARES
    dst = move % TOTAL_SQUARES
    new_src = rotate_square_180(src)
    new_dst = rotate_square_180(dst)
    return new_src * TOTAL_SQUARES + new_dst


def flip_move_horizontal(move: int) -> int:
    """
    Flip a move horizontally.

    Args:
        move: Move index (0-3136)

    Returns:
        Flipped move index
    """
    if move == END_TURN_INDEX:
        return move
    src = move // TOTAL_SQUARES
    dst = move % TOTAL_SQUARES
    new_src = flip_square_horizontal(src)
    new_dst = flip_square_horizontal(dst)
    return new_src * TOTAL_SQUARES + new_dst


def rotate_bitboard_180(bb: int) -> int:
    """
    Rotate a bitboard 180 degrees.

    Args:
        bb: Bitboard (56-bit integer)

    Returns:
        Rotated bitboard
    """
    result = 0
    for sq in range(NUM_SQUARES):
        if bb & (1 << sq):
            result |= 1 << rotate_square_180(sq)
    return result


def flip_bitboard_horizontal(bb: int) -> int:
    """
    Flip a bitboard horizontally.

    Args:
        bb: Bitboard (56-bit integer)

    Returns:
        Flipped bitboard
    """
    result = 0
    for sq in range(NUM_SQUARES):
        if bb & (1 << sq):
            result |= 1 << flip_square_horizontal(sq)
    return result


# Precompute move rotation/flip maps for efficient policy transformation
def _build_move_rotation_map() -> np.ndarray:
    """Build mapping from move index to rotated move index."""
    mapping = np.zeros(NUM_ACTIONS, dtype=np.int32)
    for move in range(NUM_ACTIONS):
        mapping[move] = rotate_move_180(move)
    return mapping


def _build_move_flip_map() -> np.ndarray:
    """Build mapping from move index to horizontally flipped move index."""
    mapping = np.zeros(NUM_ACTIONS, dtype=np.int32)
    for move in range(NUM_ACTIONS):
        mapping[move] = flip_move_horizontal(move)
    return mapping


# Precomputed maps (built once at module load)
MOVE_ROTATION_MAP = _build_move_rotation_map()
MOVE_FLIP_MAP = _build_move_flip_map()


def rotate_policy_180(policy: np.ndarray) -> np.ndarray:
    """
    Rotate policy array 180 degrees.

    Since 180° rotation is self-inverse, this function can be used both to:
    - Transform policy targets for training (original -> rotated space)
    - Transform network output back (rotated -> original space)

    Args:
        policy: Policy array of shape (..., NUM_ACTIONS)

    Returns:
        Rotated policy array
    """
    # Use gather indexing - works with batched inputs
    return policy[..., MOVE_ROTATION_MAP]


def flip_policy_horizontal(policy: np.ndarray) -> np.ndarray:
    """
    Flip policy array horizontally.

    Since horizontal flip is self-inverse, this function can be used both ways.

    Args:
        policy: Policy array of shape (..., NUM_ACTIONS)

    Returns:
        Flipped policy array
    """
    return policy[..., MOVE_FLIP_MAP]


def rotate_tensor_180(tensor: np.ndarray) -> np.ndarray:
    """
    Rotate a board tensor 180 degrees.

    Args:
        tensor: Board tensor of shape (C, H, W) or (N, C, H, W)

    Returns:
        Rotated tensor (flipped along both spatial dimensions)
    """
    # Flip along both height and width axes
    if tensor.ndim == 3:
        return np.flip(np.flip(tensor, axis=1), axis=2).copy()
    elif tensor.ndim == 4:
        return np.flip(np.flip(tensor, axis=2), axis=3).copy()
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {tensor.ndim}D")


def flip_tensor_horizontal(tensor: np.ndarray) -> np.ndarray:
    """
    Flip a board tensor horizontally.

    Args:
        tensor: Board tensor of shape (C, H, W) or (N, C, H, W)

    Returns:
        Flipped tensor
    """
    if tensor.ndim == 3:
        return np.flip(tensor, axis=2).copy()
    elif tensor.ndim == 4:
        return np.flip(tensor, axis=3).copy()
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {tensor.ndim}D")
