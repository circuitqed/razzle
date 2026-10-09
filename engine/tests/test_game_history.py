"""
Tests for account game history: AI-game metadata, result recording,
/me/games, /me/summary and /me/ai-progress.
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import persistence
from server.main import app, auth_limiter, game_create_limiter, games
from razzle.core.state import GameState
from razzle.core.moves import get_legal_moves


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        persistence.init_db(db_path)
        yield db_path


@pytest.fixture
def client(temp_db, monkeypatch):
    monkeypatch.setattr(persistence, 'DEFAULT_DB_PATH', temp_db)
    auth_limiter._hits.clear()
    game_create_limiter._hits.clear()
    with TestClient(app, base_url="https://testserver") as client:
        yield client


def register(client, username="alice"):
    r = client.post('/auth/register', json={'username': username, 'password': 'password123'})
    assert r.status_code == 200
    return r.json()['user']['user_id']


def db_row(db, game_id):
    with sqlite3.connect(db) as c:
        c.row_factory = sqlite3.Row
        return c.execute("SELECT * FROM games WHERE game_id = ?", (game_id,)).fetchone()


def play_first_turn(client, game_id):
    """Submit one legal turn (a knight move ends the turn)."""
    state = games[game_id].state
    knight = next(m for m in get_legal_moves(state) if m != -1)
    r = client.post(f'/games/{game_id}/turn', json={'moves': [knight]})
    assert r.status_code == 200, r.text


class TestCreateAIGame:
    def test_metadata_and_user_recorded(self, client, temp_db):
        uid = register(client)
        r = client.post('/games', json={
            'player2_type': 'ai', 'ai_simulations': 64,
            'ai_level': 4, 'ai_model': 'pegasus_iter_050.pt',
        })
        gid = r.json()['game_id']
        row = db_row(temp_db, gid)
        assert row['player1_user_id'] == uid
        assert row['ai_level'] == 4
        assert row['ai_model_version'] == 'pegasus_iter_050.pt'
        assert row['ai_simulations'] == 64
        assert row['player1_player_id'] == f'human_{uid}'
        assert row['player2_player_id'] == 'ai_pegasus_iter_050.pt_64'

    def test_human_as_red_takes_seat_two(self, client, temp_db):
        uid = register(client)
        gid = client.post('/games', json={'player2_type': 'ai', 'human_color': 1, 'ai_level': 2}).json()['game_id']
        row = db_row(temp_db, gid)
        assert (row['player1_type'], row['player2_type']) == ('ai', 'human')
        assert row['player1_user_id'] is None
        assert row['player2_user_id'] == uid

    def test_older_client_without_metadata(self, client, temp_db):
        gid = client.post('/games', json={'player1_type': 'human', 'player2_type': 'ai'}).json()['game_id']
        row = db_row(temp_db, gid)
        assert row['ai_level'] is None
        assert row['player1_user_id'] is None

    def test_login_mid_game_links_human_seat(self, client, temp_db):
        gid = client.post('/games', json={'player2_type': 'ai', 'human_color': 1}).json()['game_id']
        uid = register(client)
        play_first_turn(client, gid)
        assert db_row(temp_db, gid)['player2_user_id'] == uid


class TestResultAndHistory:
    def test_resigned_ai_game_in_history_and_summary(self, client, temp_db):
        register(client)
        gid = client.post('/games', json={'player2_type': 'ai', 'ai_level': 3, 'ai_simulations': 16}).json()['game_id']
        play_first_turn(client, gid)
        # AI resigns -> user wins at level 3
        assert client.post(f'/games/{gid}/resign', json={'player': 1}).status_code == 200
        row = db_row(temp_db, gid)
        assert row['winner'] == 0 and row['finished_at'] is not None

        hist = client.get('/me/games').json()
        assert hist['total'] == 1
        g = hist['games'][0]
        assert g['game_id'] == gid
        assert g['mode'] == 'ai' and g['result'] == 'win' and g['resigned']
        assert g['opponent']['type'] == 'ai' and g['opponent']['ai_level'] == 3
        assert g['move_count'] == 1

        summary = client.get('/me/summary').json()
        assert summary['vs_ai']['wins'] == 1
        assert summary['vs_ai']['by_level'] == [{'level': 3, 'wins': 1, 'losses': 0, 'draws': 0, 'games': 1}]
        assert summary['highest_ai_level_beaten'] == 3
        assert summary['vs_human']['games'] == 0

    def test_loss_as_red(self, client):
        register(client)
        gid = client.post('/games', json={'player2_type': 'ai', 'human_color': 1, 'ai_level': 5}).json()['game_id']
        play_first_turn(client, gid)  # AI (blue) moves first
        client.post(f'/games/{gid}/resign', json={'player': 1})  # human resigns
        g = client.get('/me/games').json()['games'][0]
        assert g['your_color'] == 1 and g['result'] == 'loss'
        assert client.get('/me/summary').json()['highest_ai_level_beaten'] == 0

    def test_terminal_state_recorded_without_elo(self, temp_db, monkeypatch):
        """Anonymous games still get their result recorded (no ELO ids)."""
        monkeypatch.setattr(persistence, 'DEFAULT_DB_PATH', temp_db)
        from server.main import Game
        game = Game("g1")
        persistence.save_game("g1", game.state, moves=[1])
        game.resigned_by = 1
        game.check_and_update_elo()
        row = db_row(temp_db, "g1")
        assert row['winner'] == 0 and row['finished_at'] is not None

    def test_stale_unfinished_game_is_abandoned(self, client, temp_db):
        register(client)
        gid = client.post('/games', json={'player2_type': 'ai', 'ai_level': 2}).json()['game_id']
        play_first_turn(client, gid)
        assert client.get('/me/games').json()['games'][0]['result'] == 'in_progress'
        with sqlite3.connect(temp_db) as c:
            c.execute("UPDATE games SET updated_at = '2020-01-01T00:00:00Z' WHERE game_id = ?", (gid,))
        assert client.get('/me/games').json()['games'][0]['result'] == 'abandoned'
        listed = client.get('/games').json()['games']
        assert [g['status'] for g in listed if g['game_id'] == gid] == ['abandoned']
        # Correspondence games are never auto-abandoned
        with sqlite3.connect(temp_db) as c:
            c.execute("UPDATE games SET game_mode = 'correspondence' WHERE game_id = ?", (gid,))
        assert client.get('/me/games').json()['games'][0]['result'] == 'in_progress'

    def test_admin_flag_in_auth_me(self, client, temp_db):
        uid = register(client)
        assert client.get('/auth/me').json()['is_admin'] is False
        with sqlite3.connect(temp_db) as c:
            c.execute("UPDATE users SET is_admin = 1 WHERE user_id = ?", (uid,))
        assert client.get('/auth/me').json()['is_admin'] is True

    def test_online_game_vs_human(self, temp_db, monkeypatch):
        monkeypatch.setattr(persistence, 'DEFAULT_DB_PATH', temp_db)
        a = persistence.create_user("alice", "password123")
        b = persistence.create_user("bob", "password123", display_name="Bob")
        g = persistence.create_online_game(a["user_id"], host_color=0)
        persistence.join_online_game(g["join_code"], b["user_id"])
        persistence.append_moves(g["game_id"], [5])
        persistence.update_online_game_status(g["game_id"], "finished", 1)
        hist = persistence.get_user_game_history(a["user_id"])
        entry = hist["games"][0]
        assert entry["mode"] == "online" and entry["result"] == "loss"
        assert entry["opponent"]["name"] == "Bob"
        s = persistence.get_user_game_summary(b["user_id"])
        assert s["vs_human"] == {"wins": 1, "losses": 0, "draws": 0, "games": 1}

    def test_history_requires_auth(self, client):
        assert client.get('/me/games').status_code == 401
        assert client.get('/me/summary').status_code == 401
        assert client.get('/me/ai-progress').status_code == 401

    def test_pagination_and_empty_games_skipped(self, client, temp_db):
        uid = register(client)
        state = GameState.new_game()
        for i in range(5):
            persistence.save_game(f"h{i}", state, moves=[1, 2], player1_user_id=uid)
        persistence.save_game("empty", state, moves=[], player1_user_id=uid)
        r = client.get('/me/games?page=2&per_page=2').json()
        assert r['total'] == 5 and r['total_pages'] == 3 and len(r['games']) == 2

    def test_finished_linked_games_survive_cleanup_and_startup(self, temp_db, monkeypatch):
        monkeypatch.setattr(persistence, 'DEFAULT_DB_PATH', temp_db)
        state = GameState.new_game()
        persistence.save_game("old", state, moves=[1])
        persistence.record_game_finished("old", None)
        with sqlite3.connect(temp_db) as c:
            c.execute("UPDATE games SET updated_at = '2020-01-01T00:00:00Z'")
        persistence.cleanup_old_games(max_age_days=7)
        assert db_row(temp_db, "old") is not None
        # Finished games aren't reloaded into memory on startup
        assert persistence.load_all_games(active_within_days=7) == []


class TestAIProgress:
    def test_merge_rules(self, client):
        register(client)
        assert client.get('/me/ai-progress').json() == {
            'current_level': None, 'current_level_updated_at': None, 'highest_level_beaten': 0}

        r = client.put('/me/ai-progress', json={
            'current_level': 6, 'current_level_updated_at': '2026-01-02T00:00:00.000Z',
            'highest_level_beaten': 5}).json()
        assert r['current_level'] == 6 and r['highest_level_beaten'] == 5

        # Older device: stale level ignored, highest never decreases
        r = client.put('/me/ai-progress', json={
            'current_level': 2, 'current_level_updated_at': '2026-01-01T00:00:00Z',
            'highest_level_beaten': 3}).json()
        assert r['current_level'] == 6 and r['highest_level_beaten'] == 5

        # Newer change wins (even if lower), higher peak raises highest
        r = client.put('/me/ai-progress', json={
            'current_level': 4, 'current_level_updated_at': '2026-01-03T00:00:00Z',
            'highest_level_beaten': 7}).json()
        assert r['current_level'] == 4 and r['highest_level_beaten'] == 7

        summary = client.get('/me/summary').json()
        assert summary['auto_match_level'] == 4
        assert summary['highest_ai_level_beaten'] == 7

    def test_future_timestamp_clamped(self, client):
        register(client)
        client.put('/me/ai-progress', json={
            'current_level': 9, 'current_level_updated_at': '2999-01-01T00:00:00Z'})
        r = client.put('/me/ai-progress', json={'current_level': 3}).json()
        assert r['current_level'] == 3

    def test_validation(self, client):
        register(client)
        assert client.put('/me/ai-progress', json={'current_level': 0}).status_code == 422
