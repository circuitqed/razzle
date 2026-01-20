"""
Tests for authentication endpoints.
"""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import persistence
from server.main import app


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        persistence.init_db(db_path)
        yield db_path


@pytest.fixture
def client(temp_db, monkeypatch):
    """Create a test client with temporary database."""
    # Override the default db path before any database operations
    monkeypatch.setattr(persistence, 'DEFAULT_DB_PATH', temp_db)

    with TestClient(app) as client:
        yield client


class TestUserRegistration:
    """Tests for user registration endpoint."""

    def test_register_success(self, client):
        """Test successful user registration."""
        response = client.post('/auth/register', json={
            'username': 'newuser',
            'password': 'password123',
            'display_name': 'New User'
        })

        assert response.status_code == 200
        data = response.json()
        assert data['message'] == 'Account created successfully'
        assert data['user']['username'] == 'newuser'
        assert data['user']['display_name'] == 'New User'
        assert 'user_id' in data['user']
        assert 'razzle_auth' in response.cookies

    def test_register_without_display_name(self, client):
        """Test registration without display name uses username."""
        response = client.post('/auth/register', json={
            'username': 'simpleuser',
            'password': 'password123'
        })

        assert response.status_code == 200
        data = response.json()
        assert data['user']['display_name'] == 'simpleuser'

    def test_register_duplicate_username(self, client):
        """Test registration with existing username fails."""
        # Register first user
        client.post('/auth/register', json={
            'username': 'existinguser',
            'password': 'password123'
        })

        # Try to register again with same username
        response = client.post('/auth/register', json={
            'username': 'existinguser',
            'password': 'differentpassword'
        })

        assert response.status_code == 409
        assert 'already exists' in response.json()['detail']

    def test_register_username_case_insensitive(self, client):
        """Test that username is case-insensitive."""
        client.post('/auth/register', json={
            'username': 'TestUser',
            'password': 'password123'
        })

        response = client.post('/auth/register', json={
            'username': 'testuser',
            'password': 'password456'
        })

        assert response.status_code == 409

    def test_register_short_username(self, client):
        """Test registration with too short username fails."""
        response = client.post('/auth/register', json={
            'username': 'ab',
            'password': 'password123'
        })

        assert response.status_code == 422  # Validation error

    def test_register_short_password(self, client):
        """Test registration with too short password fails."""
        response = client.post('/auth/register', json={
            'username': 'validuser',
            'password': '12345'
        })

        assert response.status_code == 422  # Validation error

    def test_register_invalid_username_chars(self, client):
        """Test registration with invalid username characters fails."""
        response = client.post('/auth/register', json={
            'username': 'user@name',
            'password': 'password123'
        })

        assert response.status_code == 422


class TestUserLogin:
    """Tests for user login endpoint."""

    @pytest.fixture(autouse=True)
    def setup_user(self, client):
        """Create a test user before each login test."""
        client.post('/auth/register', json={
            'username': 'logintest',
            'password': 'testpassword'
        })

    def test_login_success(self, client):
        """Test successful login."""
        response = client.post('/auth/login', json={
            'username': 'logintest',
            'password': 'testpassword'
        })

        assert response.status_code == 200
        data = response.json()
        assert data['message'] == 'Login successful'
        assert data['user']['username'] == 'logintest'
        assert 'razzle_auth' in response.cookies

    def test_login_wrong_password(self, client):
        """Test login with wrong password fails."""
        response = client.post('/auth/login', json={
            'username': 'logintest',
            'password': 'wrongpassword'
        })

        assert response.status_code == 401
        assert 'Invalid' in response.json()['detail']

    def test_login_nonexistent_user(self, client):
        """Test login with nonexistent user fails."""
        response = client.post('/auth/login', json={
            'username': 'nonexistent',
            'password': 'testpassword'
        })

        assert response.status_code == 401

    def test_login_case_insensitive_username(self, client):
        """Test login username is case-insensitive."""
        response = client.post('/auth/login', json={
            'username': 'LOGINTEST',
            'password': 'testpassword'
        })

        assert response.status_code == 200


class TestAuthenticatedEndpoints:
    """Tests for authenticated endpoints."""

    def test_get_me_authenticated(self, client):
        """Test getting current user when authenticated."""
        # Register first to get cookies
        client.post('/auth/register', json={
            'username': 'authuser',
            'password': 'password123'
        })

        response = client.get('/auth/me')

        assert response.status_code == 200
        data = response.json()
        assert data['username'] == 'authuser'

    def test_get_me_unauthenticated(self, client):
        """Test getting current user when not authenticated."""
        response = client.get('/auth/me')

        assert response.status_code == 401

    def test_logout(self, client):
        """Test logout endpoint works."""
        # Register first
        client.post('/auth/register', json={
            'username': 'logoutuser',
            'password': 'password123'
        })

        # Logout
        response = client.post('/auth/logout')
        assert response.status_code == 200
        assert 'Logged out' in response.json()['message']


class TestPasswordHashing:
    """Tests for password hashing functionality."""

    def test_password_not_stored_plaintext(self, temp_db):
        """Test that passwords are hashed, not stored plaintext."""
        persistence.create_user('hashtest', 'secretpassword', db_path=temp_db)

        with persistence.get_connection(temp_db) as conn:
            row = conn.execute(
                "SELECT password_hash FROM users WHERE username = ?",
                ('hashtest',)
            ).fetchone()

        assert row is not None
        assert row['password_hash'] != 'secretpassword'
        assert '$' in row['password_hash']  # Contains salt separator

    def test_verify_correct_password(self, temp_db):
        """Test password verification with correct password."""
        persistence.create_user('verifytest', 'mypassword', db_path=temp_db)
        user = persistence.authenticate_user('verifytest', 'mypassword', db_path=temp_db)

        assert user is not None
        assert user['username'] == 'verifytest'

    def test_verify_wrong_password(self, temp_db):
        """Test password verification with wrong password."""
        persistence.create_user('verifytest2', 'mypassword', db_path=temp_db)
        user = persistence.authenticate_user('verifytest2', 'wrongpassword', db_path=temp_db)

        assert user is None
