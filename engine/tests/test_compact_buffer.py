"""CompactReplayBuffer: packing round-trips exactly, eviction keeps the newest positions."""
import numpy as np

from razzle.training.compact_buffer import CompactReplayBuffer, pack_chunk


def _batch(rng, n, start_id, planes=9):
    st = (rng.random((n, planes, 8, 7)) < 0.15).astype(np.float32)
    leg = np.zeros((n, 3137), np.float32)
    pol = np.zeros((n, 3137), np.float32)
    for i in range(n):
        legal = rng.choice(3137, rng.integers(1, 40), replace=False)
        leg[i, legal] = 1
        if rng.random() < 0.5:
            p = rng.random(len(legal)).astype(np.float32)
            pol[i, legal] = (p / p.sum()).astype(np.float16)   # stored as float16
    ids = np.arange(start_id, start_id + n, dtype=np.float32)   # value doubles as position id
    pw = (rng.random(n) < 0.7).astype(np.float32)
    return st, pol, ids, leg, pw


def test_sample_reproduces_positions_exactly():
    rng = np.random.default_rng(1)
    buf = CompactReplayBuffer(min_positions=500, max_positions=500, fraction=1.0)
    originals = {}
    nid = 0
    for n in (200, 150, 300):          # 650 added, window 500 -> first chunk partially trimmed
        st, pol, ids, leg, pw = _batch(rng, n, nid)
        nid += n
        buf.add(st, pol, ids, leg, policy_weights=pw)
        for i in range(n):
            originals[int(ids[i])] = (st[i], pol[i], leg[i], pw[i])
    assert len(buf) == 500
    s, p, v, l, w = buf.sample(2000, np.random.default_rng(2))
    assert s.shape == (2000, 9, 8, 7)
    seen = set(int(x) for x in v)
    assert min(seen) >= 150           # oldest 150 positions evicted
    assert len(seen) > 400            # covers the window
    for k in range(2000):
        o = originals[int(v[k])]
        np.testing.assert_array_equal(s[k], o[0])
        np.testing.assert_array_equal(p[k], o[1])
        np.testing.assert_array_equal(l[k], o[2])
        assert w[k] == o[3]


def test_add_chunk_matches_add_and_checks_planes():
    rng = np.random.default_rng(3)
    st, pol, ids, leg, pw = _batch(rng, 50, 0)
    a, b = CompactReplayBuffer(min_positions=100), CompactReplayBuffer(min_positions=100)
    a.add(st, pol, ids, leg, policy_weights=pw)
    b.add_chunk(pack_chunk(st, pol, ids, leg, pw))
    r = np.random.default_rng(4)
    xa = a.sample(64, r)
    xb = b.sample(64, np.random.default_rng(4))
    for u, t in zip(xa, xb):
        np.testing.assert_array_equal(u, t)
    st7 = _batch(rng, 5, 100, planes=7)
    try:
        b.add(*st7[:4], policy_weights=st7[4])
        assert False, 'expected plane mismatch error'
    except ValueError:
        pass


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(5)
    buf = CompactReplayBuffer(min_positions=1000)
    for k in range(3):
        buf.add(*_batch(rng, 100, 100 * k)[:4])
    buf.save(tmp_path / 'w.npz')
    b2 = CompactReplayBuffer(min_positions=1000)
    b2.load(tmp_path / 'w.npz')
    assert len(b2) == 300 and b2.planes == 9 and b2.total_positions_seen == 300
    x, y = buf.sample(32, np.random.default_rng(6)), b2.sample(32, np.random.default_rng(6))
    for u, t in zip(x, y):
        np.testing.assert_array_equal(u, t)
