"""cleanup_old_games must keep every game with moves (players' history); only empty games go."""
import sqlite3
import tempfile
from pathlib import Path

from server import persistence


def _insert(conn, gid, updated, winner=None, resigned=None, user=None, moves='[1]'):
    conn.execute(
        "INSERT INTO games (game_id, player1_type, player2_type, state_json, moves_json, created_at, updated_at,"
        " winner, resigned_by, player1_user_id) VALUES (?, 'human', 'ai', '{}', ?, ?, ?, ?, ?, ?)",
        (gid, moves, updated, updated, winner, resigned, user))


def test_cleanup_keeps_history():
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "games.db"
        persistence.init_db(db)
        old, new = "2020-01-01T00:00:00Z", "2999-01-01T00:00:00Z"
        with sqlite3.connect(db) as c:
            _insert(c, "finished_old", old, winner=1)
            _insert(c, "resigned_old", old, resigned=0)
            _insert(c, "linked_old", old, user="u1")
            _insert(c, "nowinner_old", old)          # realistic: local AI games have winner NULL
            _insert(c, "empty_old", old, moves='[]', winner=None)
            _insert(c, "fresh", new)
        deleted = persistence.cleanup_old_games(max_age_days=7, db_path=db)
        with sqlite3.connect(db) as c:
            left = {r[0] for r in c.execute("SELECT game_id FROM games")}
        assert left == {"finished_old", "resigned_old", "linked_old", "nowinner_old", "fresh"}
        assert deleted == 1   # only the empty game
