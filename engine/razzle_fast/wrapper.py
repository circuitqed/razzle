"""
Pythonic wrappers around the C engine using ctypes.

FastState is a drop-in replacement for GameState.
FastMCTS drives the C tree with Python NN evaluation.
"""

from __future__ import annotations

import ctypes
import os
import numpy as np
from numpy.ctypeslib import ndpointer
from typing import Optional

# Constants
NUM_SQ = 56
NUM_ACTIONS = 3137
END_TURN_ACTION = 3136
TENSOR_SIZE = 7 * 8 * 7  # 392 floats

# Load the shared library
_here = os.path.dirname(os.path.abspath(__file__))
_lib_path = os.path.join(_here, "libcore.so")
if not os.path.exists(_lib_path):
    raise ImportError(
        f"C library not found at {_lib_path}. "
        f"Build it with: cd engine && python3 razzle_fast/_build.py"
    )
_lib = ctypes.CDLL(_lib_path)


# ============================================================
# C struct definitions
# ============================================================

class CRazzleState(ctypes.Structure):
    _fields_ = [
        ("pieces", ctypes.c_uint64 * 2),
        ("balls", ctypes.c_uint64 * 2),
        ("touched_mask", ctypes.c_uint64),
        ("current_player", ctypes.c_int32),
        ("has_passed", ctypes.c_int32),
        ("last_knight_dst", ctypes.c_int32),
        ("ply", ctypes.c_int32),
    ]


class CMCTSNode(ctypes.Structure):
    _fields_ = [
        ("state", CRazzleState),
        ("parent", ctypes.c_int32),
        ("first_child", ctypes.c_int32),
        ("next_sibling", ctypes.c_int32),
        ("parent_action", ctypes.c_int32),
        ("prior", ctypes.c_double),
        ("visit_count", ctypes.c_int32),
        ("value_sum", ctypes.c_double),
        ("virtual_loss", ctypes.c_int32),
        ("is_terminal", ctypes.c_int32),
        ("is_expanded", ctypes.c_int32),
        ("num_children", ctypes.c_int32),
    ]


class CMCTSTree(ctypes.Structure):
    _fields_ = [
        ("nodes", ctypes.POINTER(CMCTSNode)),
        ("capacity", ctypes.c_int32),
        ("count", ctypes.c_int32),
        ("root", ctypes.c_int32),
        ("path_buf", ctypes.POINTER(ctypes.c_int32)),
        ("path_lens", ctypes.POINTER(ctypes.c_int32)),
        ("leaf_indices", ctypes.POINTER(ctypes.c_int32)),
        ("max_batch", ctypes.c_int32),
        ("max_depth", ctypes.c_int32),
    ]


# ============================================================
# Function signatures
# ============================================================

# State operations
_lib.razzle_state_init.argtypes = [ctypes.POINTER(CRazzleState)]
_lib.razzle_state_init.restype = None

_lib.razzle_state_copy.argtypes = [ctypes.POINTER(CRazzleState), ctypes.POINTER(CRazzleState)]
_lib.razzle_state_copy.restype = None

_lib.razzle_state_apply_move.argtypes = [ctypes.POINTER(CRazzleState), ctypes.c_int]
_lib.razzle_state_apply_move.restype = None

_lib.razzle_state_is_terminal.argtypes = [ctypes.POINTER(CRazzleState)]
_lib.razzle_state_is_terminal.restype = ctypes.c_int

_lib.razzle_state_get_winner.argtypes = [ctypes.POINTER(CRazzleState)]
_lib.razzle_state_get_winner.restype = ctypes.c_int

_lib.razzle_state_get_result.argtypes = [ctypes.POINTER(CRazzleState), ctypes.c_int]
_lib.razzle_state_get_result.restype = ctypes.c_float

_lib.razzle_state_get_legal_moves.argtypes = [ctypes.POINTER(CRazzleState), ctypes.POINTER(ctypes.c_int)]
_lib.razzle_state_get_legal_moves.restype = ctypes.c_int

_lib.razzle_state_to_tensor.argtypes = [ctypes.POINTER(CRazzleState), ctypes.POINTER(ctypes.c_float)]
_lib.razzle_state_to_tensor.restype = None

_lib.razzle_state_equals.argtypes = [ctypes.POINTER(CRazzleState), ctypes.POINTER(CRazzleState)]
_lib.razzle_state_equals.restype = ctypes.c_int

# MCTS operations
_lib.razzle_mcts_create.argtypes = [
    ctypes.POINTER(CRazzleState), ctypes.c_int, ctypes.c_int, ctypes.c_int
]
_lib.razzle_mcts_create.restype = ctypes.POINTER(CMCTSTree)

_lib.razzle_mcts_free.argtypes = [ctypes.POINTER(CMCTSTree)]
_lib.razzle_mcts_free.restype = None

_lib.razzle_mcts_expand_root.argtypes = [ctypes.POINTER(CMCTSTree), ctypes.POINTER(ctypes.c_float)]
_lib.razzle_mcts_expand_root.restype = None

_lib.razzle_mcts_add_dirichlet_noise.argtypes = [
    ctypes.POINTER(CMCTSTree), ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_int
]
_lib.razzle_mcts_add_dirichlet_noise.restype = None

_lib.razzle_mcts_select_leaves.argtypes = [
    ctypes.POINTER(CMCTSTree), ctypes.c_int, ctypes.c_int, ctypes.c_float,
    ctypes.POINTER(ctypes.c_float)
]
_lib.razzle_mcts_select_leaves.restype = ctypes.c_int

_lib.razzle_mcts_expand_and_backup.argtypes = [
    ctypes.POINTER(CMCTSTree), ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int
]
_lib.razzle_mcts_expand_and_backup.restype = None

_lib.razzle_mcts_get_policy.argtypes = [
    ctypes.POINTER(CMCTSTree), ctypes.POINTER(ctypes.c_float), ctypes.c_float
]
_lib.razzle_mcts_get_policy.restype = None

_lib.razzle_mcts_root_visits.argtypes = [ctypes.POINTER(CMCTSTree)]
_lib.razzle_mcts_root_visits.restype = ctypes.c_int

_lib.razzle_mcts_should_stop_early.argtypes = [ctypes.POINTER(CMCTSTree), ctypes.c_int, ctypes.c_float]
_lib.razzle_mcts_should_stop_early.restype = ctypes.c_int

_lib.razzle_mcts_check_immediate_win.argtypes = [ctypes.POINTER(CMCTSTree)]
_lib.razzle_mcts_check_immediate_win.restype = ctypes.c_int

_lib.razzle_mcts_get_root_children.argtypes = [
    ctypes.POINTER(CMCTSTree),
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)
]
_lib.razzle_mcts_get_root_children.restype = ctypes.c_int


# ============================================================
# Helper: numpy <-> ctypes
# ============================================================

def _np_to_cfloat_ptr(arr: np.ndarray):
    """Get a ctypes float* from a contiguous float32 numpy array."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


# ============================================================
# FastState
# ============================================================

class FastState:
    """C-backed game state, API-compatible with GameState."""

    __slots__ = ("_cs",)

    def __init__(self, _cs=None):
        if _cs is not None:
            self._cs = _cs
        else:
            self._cs = CRazzleState()
            _lib.razzle_state_init(ctypes.byref(self._cs))

    @classmethod
    def from_game_state(cls, gs) -> FastState:
        """Convert a Python GameState to FastState."""
        cs = CRazzleState()
        cs.pieces[0] = gs.pieces[0]
        cs.pieces[1] = gs.pieces[1]
        cs.balls[0] = gs.balls[0]
        cs.balls[1] = gs.balls[1]
        cs.touched_mask = gs.touched_mask
        cs.current_player = gs.current_player
        cs.has_passed = int(gs.has_passed)
        cs.last_knight_dst = gs.last_knight_dst
        cs.ply = gs.ply
        return cls(_cs=cs)

    def to_game_state(self):
        """Convert back to Python GameState."""
        from razzle.core.state import GameState
        return GameState(
            pieces=(int(self._cs.pieces[0]), int(self._cs.pieces[1])),
            balls=(int(self._cs.balls[0]), int(self._cs.balls[1])),
            current_player=int(self._cs.current_player),
            touched_mask=int(self._cs.touched_mask),
            has_passed=bool(self._cs.has_passed),
            last_knight_dst=int(self._cs.last_knight_dst),
            ply=int(self._cs.ply),
        )

    def copy(self) -> FastState:
        new_cs = CRazzleState()
        _lib.razzle_state_copy(ctypes.byref(self._cs), ctypes.byref(new_cs))
        return FastState(_cs=new_cs)

    def apply_move(self, move: int) -> None:
        _lib.razzle_state_apply_move(ctypes.byref(self._cs), move)

    def is_terminal(self) -> bool:
        return bool(_lib.razzle_state_is_terminal(ctypes.byref(self._cs)))

    def get_winner(self) -> Optional[int]:
        w = _lib.razzle_state_get_winner(ctypes.byref(self._cs))
        return None if w < 0 else int(w)

    def get_result(self, player: int) -> float:
        return float(_lib.razzle_state_get_result(ctypes.byref(self._cs), player))

    def get_legal_moves(self) -> list[int]:
        moves_buf = (ctypes.c_int * 256)()
        n = _lib.razzle_state_get_legal_moves(ctypes.byref(self._cs), moves_buf)
        return [int(moves_buf[i]) for i in range(n)]

    def to_tensor(self) -> np.ndarray:
        buf = np.zeros(TENSOR_SIZE, dtype=np.float32)
        _lib.razzle_state_to_tensor(ctypes.byref(self._cs), _np_to_cfloat_ptr(buf))
        return buf.reshape(7, 8, 7)

    @property
    def current_player(self) -> int:
        return int(self._cs.current_player)

    @property
    def has_passed(self) -> bool:
        return bool(self._cs.has_passed)

    @property
    def ply(self) -> int:
        return int(self._cs.ply)

    @property
    def pieces(self) -> tuple[int, int]:
        return (int(self._cs.pieces[0]), int(self._cs.pieces[1]))

    @property
    def balls(self) -> tuple[int, int]:
        return (int(self._cs.balls[0]), int(self._cs.balls[1]))

    @property
    def touched_mask(self) -> int:
        return int(self._cs.touched_mask)

    @property
    def last_knight_dst(self) -> int:
        return int(self._cs.last_knight_dst)

    def __eq__(self, other):
        if not isinstance(other, FastState):
            return NotImplemented
        return bool(_lib.razzle_state_equals(ctypes.byref(self._cs), ctypes.byref(other._cs)))

    def __repr__(self):
        return (
            f"FastState(player={self.current_player}, ply={self.ply}, "
            f"passed={self.has_passed})"
        )


# ============================================================
# FastMCTS
# ============================================================

class FastMCTS:
    """
    C-backed MCTS search.

    Uses the C tree for selection/expansion/backup, with Python NN for evaluation.
    Compatible with the existing evaluator interface (evaluate_batch).
    """

    def __init__(self, evaluator, config=None):
        self.evaluator = evaluator
        if config is None:
            from razzle.ai.mcts import MCTSConfig
            config = MCTSConfig()
        self.config = config

    def search_batched(self, state, add_noise: bool = True):
        """Run MCTS search. Returns _MCTSResult with policy and selected_move."""
        policy, value, selected_move, _ = self._search_with_tree(state, add_noise)
        return _MCTSResult(policy, selected_move, value)

    def search_with_visits(self, state, add_noise: bool = True):
        """Run MCTS search. Returns (_MCTSResult, visit_counts_dict)."""
        policy, value, selected_move, visit_counts = self._search_with_tree(state, add_noise, extract_visits=True)
        return _MCTSResult(policy, selected_move, value), visit_counts

    def search_fast(self, state, add_noise: bool = True) -> tuple[np.ndarray, int]:
        """Fast path: returns (policy, selected_move) directly."""
        policy, _, selected_move, _ = self._search_with_tree(state, add_noise)
        return policy, selected_move

    def _search_with_tree(self, state, add_noise: bool, extract_visits: bool = False):
        """Core search. Returns (policy, root_value, selected_move, visit_counts)."""
        # Convert state
        if isinstance(state, FastState):
            cs = state._cs
        else:
            fs = FastState.from_game_state(state)
            cs = fs._cs

        cfg = self.config
        max_nodes = max(200000, cfg.num_simulations * 40)
        max_depth = 200

        tree = _lib.razzle_mcts_create(ctypes.byref(cs), max_nodes, cfg.batch_size, max_depth)
        if not tree:
            raise MemoryError("Failed to allocate MCTS tree")

        try:
            policy, root_value, selected_move = self._run_search(tree, cs, add_noise)
            visit_counts = None
            if extract_visits:
                visit_counts = self._extract_visit_counts(tree)
            return policy, root_value, selected_move, visit_counts
        finally:
            _lib.razzle_mcts_free(tree)

    def _extract_visit_counts(self, tree) -> dict[int, int]:
        """Extract visit counts from root's children as {move: visits} dict.

        Actions use the Python convention: -1 for END_TURN, src*56+dst for moves.
        """
        max_children = 256
        actions = (ctypes.c_int * max_children)()
        visits = (ctypes.c_int * max_children)()
        priors = (ctypes.c_float * max_children)()
        values = (ctypes.c_float * max_children)()

        n = _lib.razzle_mcts_get_root_children(
            tree, actions, visits, priors, values
        )

        vc = {}
        for i in range(n):
            v = int(visits[i])
            if v > 0:
                vc[int(actions[i])] = v
        return vc

    def _run_search(self, tree, root_cs, add_noise: bool):
        cfg = self.config

        # Evaluate root state
        root_gs = _cs_to_game_state(root_cs)
        policy_np, value = self.evaluator.evaluate(root_gs)

        # Expand root
        _lib.razzle_mcts_expand_root(tree, _np_to_cfloat_ptr(policy_np))

        # Check immediate win
        win_move = _lib.razzle_mcts_check_immediate_win(tree)
        if win_move != -2:
            policy_out = np.zeros(NUM_ACTIONS, dtype=np.float32)
            action_idx = END_TURN_ACTION if win_move == -1 else win_move
            policy_out[action_idx] = 1.0
            return policy_out, 1.0, win_move

        # Add Dirichlet noise
        if add_noise and cfg.dirichlet_epsilon > 0:
            root_node = tree.contents.nodes[tree.contents.root]
            num_children = root_node.num_children
            if num_children > 0:
                noise = np.random.dirichlet(
                    [cfg.dirichlet_alpha] * num_children
                ).astype(np.float32)
                _lib.razzle_mcts_add_dirichlet_noise(
                    tree, cfg.dirichlet_epsilon,
                    _np_to_cfloat_ptr(noise), num_children
                )

        # Run simulations
        sims_done = 0
        batch_size = cfg.batch_size
        tensors_buf = np.zeros(batch_size * TENSOR_SIZE, dtype=np.float32)
        vloss = cfg.virtual_loss

        while sims_done < cfg.num_simulations:
            current_batch = min(batch_size, cfg.num_simulations - sims_done)

            leaf_count = _lib.razzle_mcts_select_leaves(
                tree, current_batch, vloss, cfg.c_puct,
                _np_to_cfloat_ptr(tensors_buf)
            )

            if leaf_count > 0:
                # Build GameStates for evaluator
                leaf_states = []
                for k in range(leaf_count):
                    batch_idx = tree.contents.leaf_indices[k]
                    path_offset = batch_idx * tree.contents.max_depth
                    path_len = tree.contents.path_lens[batch_idx]
                    leaf_node_idx = tree.contents.path_buf[path_offset + path_len - 1]
                    leaf_node = tree.contents.nodes[leaf_node_idx]
                    gs = _cstate_to_game_state(leaf_node.state)
                    leaf_states.append(gs)

                results = self.evaluator.evaluate_batch(leaf_states)

                # Pack for C
                policies_np = np.zeros((leaf_count, NUM_ACTIONS), dtype=np.float32)
                values_np = np.zeros(leaf_count, dtype=np.float32)
                for k, (pol, val) in enumerate(results):
                    policies_np[k] = pol
                    values_np[k] = val

                _lib.razzle_mcts_expand_and_backup(
                    tree, leaf_count,
                    _np_to_cfloat_ptr(policies_np),
                    _np_to_cfloat_ptr(values_np),
                    vloss
                )

            sims_done += current_batch

            # Early termination
            if (cfg.early_termination and
                sims_done >= cfg.min_sims_for_early_stop and
                sims_done % cfg.check_early_stop_interval < batch_size):
                if _lib.razzle_mcts_should_stop_early(
                    tree, cfg.min_sims_for_early_stop,
                    cfg.visit_concentration_threshold
                ):
                    break

        # Extract policy
        policy_out = np.zeros(NUM_ACTIONS, dtype=np.float32)
        _lib.razzle_mcts_get_policy(tree, _np_to_cfloat_ptr(policy_out), cfg.temperature)
        # Re-read since C wrote to the buffer
        # (numpy array was passed by pointer, so it was written in-place)

        root_value = 0.0
        root_visits = _lib.razzle_mcts_root_visits(tree)
        if root_visits > 0:
            root_node = tree.contents.nodes[tree.contents.root]
            root_value = root_node.value_sum / root_visits

        selected_move = self._select_move_from_policy(policy_out, cfg.temperature)
        return policy_out, root_value, selected_move

    def _select_move_from_policy(self, policy: np.ndarray, temperature: float) -> int:
        """Select a move from the policy distribution."""
        if temperature == 0:
            action_idx = int(np.argmax(policy))
        else:
            nonzero = policy > 0
            if not np.any(nonzero):
                action_idx = int(np.argmax(policy))
            else:
                action_idx = int(np.random.choice(NUM_ACTIONS, p=policy))

        return -1 if action_idx == END_TURN_ACTION else action_idx

    def get_policy(self, result) -> np.ndarray:
        return result.policy

    def select_move(self, result) -> int:
        return result.selected_move


class _MCTSResult:
    """Lightweight result object from FastMCTS search."""
    __slots__ = ("policy", "selected_move", "root_value")

    def __init__(self, policy: np.ndarray, selected_move: int, root_value: float):
        self.policy = policy
        self.selected_move = selected_move
        self.root_value = root_value


# ============================================================
# Internal helpers
# ============================================================

def _cs_to_game_state(cs):
    """Convert a ctypes CRazzleState (pointer-like) to Python GameState."""
    from razzle.core.state import GameState
    return GameState(
        pieces=(int(cs.pieces[0]), int(cs.pieces[1])),
        balls=(int(cs.balls[0]), int(cs.balls[1])),
        current_player=int(cs.current_player),
        touched_mask=int(cs.touched_mask),
        has_passed=bool(cs.has_passed),
        last_knight_dst=int(cs.last_knight_dst),
        ply=int(cs.ply),
    )


def _cstate_to_game_state(rs):
    """Convert an embedded CRazzleState struct to Python GameState."""
    from razzle.core.state import GameState
    return GameState(
        pieces=(int(rs.pieces[0]), int(rs.pieces[1])),
        balls=(int(rs.balls[0]), int(rs.balls[1])),
        current_player=int(rs.current_player),
        touched_mask=int(rs.touched_mask),
        has_passed=bool(rs.has_passed),
        last_knight_dst=int(rs.last_knight_dst),
        ply=int(rs.ply),
    )
