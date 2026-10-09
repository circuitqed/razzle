"""C search options (razzle_mcts_set_search_options): virtual loss in Q, first-play urgency."""

import ctypes

import numpy as np

from razzle_fast.wrapper import _lib, CRazzleState, _np_to_cfloat_ptr

_lib.razzle_mcts_set_search_options.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float]
_lib.razzle_mcts_set_search_options.restype = None
_lib.razzle_state_init.argtypes = [ctypes.POINTER(CRazzleState)]

NUM_ACTIONS = 3137
TENSOR = 7 * 8 * 7


def _search(vloss_q=0, fpu_mode=0, fpu=0.0, batch=8, rounds=30, seed=0, set_options=True):
    """Batched search on random priors/values; returns (duplicate leaves, root visits)."""
    rng = np.random.default_rng(seed)
    cs = CRazzleState()
    _lib.razzle_state_init(ctypes.byref(cs))
    tree = _lib.razzle_mcts_create(ctypes.byref(cs), 100_000, batch, 256)
    if set_options:
        _lib.razzle_mcts_set_search_options(ctypes.cast(tree, ctypes.c_void_p), vloss_q, fpu_mode, fpu)
    pol = rng.random(NUM_ACTIONS).astype(np.float32)
    pol /= pol.sum()
    _lib.razzle_mcts_expand_root(tree, _np_to_cfloat_ptr(pol))
    buf = np.zeros(batch * TENSOR, np.float32)
    dup = 0
    for _ in range(rounds):
        n = _lib.razzle_mcts_select_leaves(tree, batch, 3, ctypes.c_float(1.5), _np_to_cfloat_ptr(buf))
        leaves = buf[: n * TENSOR].reshape(n, TENSOR)
        dup += n - len({x.tobytes() for x in leaves})
        p = rng.random((n, NUM_ACTIONS)).astype(np.float32)
        p /= p.sum(1, keepdims=True)
        v = rng.uniform(-1, 1, n).astype(np.float32)
        _lib.razzle_mcts_expand_and_backup(tree, n, _np_to_cfloat_ptr(p), _np_to_cfloat_ptr(v), 3)
    visits = (ctypes.c_int * 256)()
    actions = (ctypes.c_int * 256)()
    priors = (ctypes.c_float * 256)()
    vals = (ctypes.c_float * 256)()
    k = _lib.razzle_mcts_get_root_children(tree, actions, visits, priors, vals)
    out = sorted((actions[i], visits[i]) for i in range(k))
    _lib.razzle_mcts_free(tree)
    return dup, out


def test_defaults_are_the_original_search():
    for seed in range(3):
        assert _search(seed=seed, set_options=False) == _search(seed=seed)


def test_vloss_q_spreads_batches():
    base = sum(_search(seed=s)[0] for s in range(20))
    vlq = sum(_search(vloss_q=1, seed=s)[0] for s in range(20))
    assert vlq < base


def test_fpu_changes_the_search():
    assert _search(fpu_mode=1, fpu=0.5)[1] != _search()[1]
