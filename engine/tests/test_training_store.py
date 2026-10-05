"""Self-play game storage: separate compressed training.db, daily archive, safe resets, legacy migration."""
import gzip
import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from server import persistence


@pytest.fixture
def data_dir(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "games.db"
        persistence.init_db(db)
        monkeypatch.setattr(persistence, "DEFAULT_DB_PATH", db)
        yield Path(d)


GAME = dict(worker_id="w1", moves=[66, 2835, -1], result=1.0,
            visit_counts=[{"66": 10, "67": 5}, {}, {"-1": 1}], model_version="v2run_iter_003")


def test_roundtrip_compressed_and_archived(data_dir):
    gid = persistence.save_training_game(**GAME)
    games, pending = persistence.get_pending_training_games(limit=10)
    assert pending == 1 and games[0]["id"] == gid
    assert games[0]["moves"] == GAME["moves"] and games[0]["visit_counts"] == GAME["visit_counts"]

    # stored in training.db (compressed blob), not games.db
    with sqlite3.connect(data_dir / "training.db") as c:
        run, blob = c.execute("SELECT run_name, data FROM selfplay_games").fetchone()
    assert run == "v2run" and isinstance(blob, bytes)
    with sqlite3.connect(data_dir / "games.db") as c:
        assert c.execute("SELECT COUNT(*) FROM training_games").fetchone()[0] == 0

    # appended to today's archive
    files = list((data_dir / "training_archive").glob("selfplay_*.jsonl.gz"))
    assert len(files) == 1
    rec = json.loads(gzip.open(files[0], "rt").readline())
    assert rec["moves"] == GAME["moves"] and rec["id"] == gid


def test_fetch_marks_used_and_reset_keeps_games(data_dir):
    for _ in range(3):
        persistence.save_training_game(**GAME)
    persistence.get_pending_training_games(limit=1)          # 1 used, 2 pending
    out = persistence.clear_training_data()
    assert out["games_deleted"] == 2                          # retired, not deleted
    stats = persistence.get_training_games_stats()
    assert stats["total"] == 3 and stats["pending"] == 0
    with sqlite3.connect(data_dir / "training.db") as c:
        assert dict(c.execute("SELECT status, COUNT(*) FROM selfplay_games GROUP BY status").fetchall()) == \
            {"archived": 2, "used": 1}


def test_migrate_legacy_rows_idempotent(data_dir):
    with sqlite3.connect(data_dir / "games.db") as c:
        for i in range(7):
            c.execute("INSERT INTO training_games (worker_id, moves, result, visit_counts, model_version, status, created_at)"
                      " VALUES (?, ?, ?, ?, ?, 'used', '2026-04-01T00:00:00Z')",
                      ("old", json.dumps([1, 2, 3 + i]), -1.0, json.dumps([{"1": 3}, {}, {}]), "gryphon_iter_010"))
    assert persistence.migrate_legacy_training_games(log=lambda *_: None) == {"copied": 7, "skipped": 0}
    assert persistence.migrate_legacy_training_games(log=lambda *_: None) == {"copied": 0, "skipped": 7}
    games, total = persistence.get_all_training_games(limit=100)
    assert total == 7 and sorted(g["moves"][2] for g in games) == list(range(3, 10))


def test_training_db_uses_wal(tmp_path, monkeypatch):
    """Exports / dashboard reads must not block self-play inserts."""
    import sqlite3
    from server import persistence
    monkeypatch.setattr(persistence, "DEFAULT_DB_PATH", tmp_path / "games.db")
    persistence.init_db(tmp_path / "games.db")
    persistence.save_training_game("w", [1, 2], 1.0, [{}, {}], "run_iter_001")
    db = persistence.training_db_path()
    mode = sqlite3.connect(str(db)).execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"
    # a long-running reader doesn't block a writer
    reader = sqlite3.connect(str(db))
    reader.execute("BEGIN")
    reader.execute("SELECT count(*) FROM selfplay_games").fetchone()
    persistence.save_training_game("w", [3], -1.0, [{}], "run_iter_001")
    reader.rollback()
    assert persistence.get_training_games_stats()["total"] >= 2


def test_search_values_round_trip(tmp_path, monkeypatch):
    from server import persistence
    monkeypatch.setattr(persistence, "DEFAULT_DB_PATH", tmp_path / "games.db")
    persistence.init_db(tmp_path / "games.db")
    persistence.save_training_game("w", [1, 2, 3], 1.0, [{}, {5: 2}, {}], "run_iter_001",
                                   search_values=[None, 0.25, -0.5])
    persistence.save_training_game("w", [4], -1.0, [{}], "run_iter_001")     # old-style game
    games, pending = persistence.get_pending_training_games(limit=10, mark_used=False)
    assert pending == 2
    assert games[0]["search_values"] == [None, 0.25, -0.5]
    assert games[1]["search_values"] is None


def test_value_mix_in_training_targets():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import trainer as T
    from razzle.training.api_client import TrainingGame
    from razzle.core.state import GameState
    from razzle.core.moves import get_legal_moves
    # two knight moves: P0 moves, then P1 moves (positions for P0 and P1); P0 wins
    s = GameState.new_game(); m0 = get_legal_moves(s)[0]; s.apply_move(m0); m1 = get_legal_moves(s)[0]
    g = TrainingGame(id=1, worker_id="w", moves=[m0, m1], result=1.0, visit_counts=[{m0: 3}, {m1: 3}],
                     model_version="r_iter_000", created_at="", search_values=[0.2, 0.6])
    _, _, v_plain, _, _ = T.games_to_training_data([g], num_planes=9)
    _, _, v_mix, _, _ = T.games_to_training_data([g], num_planes=9, value_mix=0.5)
    assert v_plain[0] == 0.99 and v_plain[1] == -0.99            # outcome from the side to move
    assert abs(v_mix[0] - (0.5 * 0.2 + 0.5 * 1.0)) < 1e-6         # P0: outcome +1
    assert abs(v_mix[1] - (0.5 * 0.6 + 0.5 * -1.0)) < 1e-6        # P1: outcome -1
