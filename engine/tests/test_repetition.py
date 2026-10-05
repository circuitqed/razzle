"""Repetition rule variant: threefold repetition of a turn-start position is a draw.

Covers the Python engine, the C engine's search (history-aware draw detection), and
parity between them. The rule is off by default; these tests switch it on locally.
"""
import ctypes
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE / 'scripts' / 'distill'))

import razzle.core.state as state_mod  # noqa: E402
from razzle.core.state import GameState  # noqa: E402
from razzle.core.moves import get_legal_moves  # noqa: E402

arena = pytest.importorskip('arena')


@pytest.fixture
def rule_on(monkeypatch):
    monkeypatch.setattr(state_mod, 'REPETITION_DRAW', True)
    monkeypatch.setattr(arena, 'REPETITION_DRAW', True)


def knight_moves(s):
    return [m for m in get_legal_moves(s) if m != -1 and not (s.balls[s.current_player] >> (m // 56)) & 1]


def shuffle_cycle(s):
    """Both players move a knight out and back: returns the 4 moves applied."""
    played = []
    out = {}
    for t in range(4):
        p = s.current_player
        if t < 2:
            m = knight_moves(s)[0]
            out[p] = m
        else:
            src, dst = divmod(out[p], 56)
            m = dst * 56 + src
            assert m in get_legal_moves(s), 'return move must be legal'
        s.apply_move(m)
        played.append(m)
    return played


def test_rule_off_by_default():
    s = GameState.new_game()
    assert s.position_counts is None
    for _ in range(4):
        shuffle_cycle(s)
    assert not s.is_terminal()


def test_threefold_shuffle_is_a_draw(rule_on):
    """Draw exactly when some turn-start position occurs for the third time."""
    probe = GameState.new_game()
    seq = []
    for _ in range(4):
        seq += shuffle_cycle(probe)
    s = GameState.new_game()
    seen = {s.position_key(): 1}
    draw_at = None
    for i, m in enumerate(seq):
        s.apply_move(m)
        k = s.position_key()
        seen[k] = seen.get(k, 0) + 1
        expected = seen[k] >= 3
        assert s.is_terminal() == expected, i
        if expected:
            draw_at = i
            break
    assert draw_at is not None and 8 <= draw_at < 12      # during the third cycle
    assert s.get_winner() is None and s.get_result(0) == 0.5
    s.undo_move()                    # undo rewinds the count
    assert not s.is_terminal()


def test_mid_pass_states_do_not_count(rule_on):
    s = GameState.new_game()
    # play until a pass is available, then check a mid-pass state never registers
    rng = random.Random(3)
    for _ in range(200):
        if s.is_terminal():
            break
        moves = get_legal_moves(s)
        passes = [m for m in moves if m != -1 and (s.balls[s.current_player] >> (m // 56)) & 1]
        if passes and not s.has_passed:
            before = dict(s.position_counts)
            s.apply_move(passes[0])
            assert s.has_passed and s.position_counts == before
            return
        s.apply_move(rng.choice(moves))
    pytest.skip('no pass position reached')


def test_python_c_parity_random_shuffling_games(rule_on):
    rng = random.Random(0)
    for game in range(300):
        s = GameState.new_game()
        cs = arena.CRazzleState()
        arena._lib.razzle_state_init(ctypes.byref(cs))
        rep = arena.RepetitionTracker(cs)
        last = {0: None, 1: None}
        for ply in range(400):
            py_draw = s.is_repetition_draw()
            c_draw = rep.is_draw()
            assert py_draw == c_draw, (game, ply)
            if s.is_terminal() or arena._lib.razzle_state_is_terminal(ctypes.byref(cs)) or c_draw:
                break
            moves = get_legal_moves(s)
            p = s.current_player
            back = None
            if last[p] is not None:
                src, dst = divmod(last[p], 56)
                back = dst * 56 + src
            m = back if back in moves and rng.random() < 0.8 else rng.choice(moves)
            if m != -1 and not (s.balls[p] >> (m // 56)) & 1:
                last[p] = m
            s.apply_move(m)
            arena._lib.razzle_state_apply_move(ctypes.byref(cs), m)
            rep.add(cs)


class _UniformModel:
    def __call__(self, x, players, extra):
        n = len(players)
        return np.full((n, 3137), 1.0 / 3137, np.float32), np.zeros(n, np.float32)


def test_search_scores_third_repetition_as_draw(rule_on):
    s = GameState.new_game()
    cs = arena.CRazzleState()
    arena._lib.razzle_state_init(ctypes.byref(cs))
    rep = arena.RepetitionTracker(cs)
    # two full cycles plus the first three moves of a third: the next move recreates X a third time
    seq = []
    probe = GameState.new_game()
    for _ in range(3):
        seq += shuffle_cycle(probe)
    for m in seq[:-1]:
        s.apply_move(m)
        arena._lib.razzle_state_apply_move(ctypes.byref(cs), m)
        rep.add(cs)
    final = seq[-1]
    assert final in get_legal_moves(s)
    search = arena.Search(cs, 256, 8, history=rep.hist)
    model = _UniformModel()
    while not search.finished():
        x, pl, ex = search.request()
        if len(pl):
            search.deliver(*model(x, pl, ex))
        else:
            search.deliver(np.zeros((0, 3137), np.float32), np.zeros(0, np.float32))
    tc = search.tree.contents
    c = tc.nodes[tc.root].first_child
    found = False
    while c >= 0:
        node = tc.nodes[c]
        if node.parent_action == final:
            found = True
            assert node.is_terminal == 1, 'third repetition must be terminal'
            assert node.visit_count == 0 or abs(node.value_sum) < 1e-9, 'draw scores 0'
        else:
            assert node.is_terminal == 0
        c = node.next_sibling
    assert found
    search.free()
