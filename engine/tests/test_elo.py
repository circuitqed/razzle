"""Unit tests for ELO rating calculation."""

import pytest
from razzle.training.elo import (
    EloRating,
    expected_score,
    update_rating,
    update_ratings_from_match,
    compute_all_ratings,
    elo_win_probability,
    rating_difference_for_win_rate,
    format_rating,
    get_iteration_from_version,
)


class TestExpectedScore:
    """Tests for expected_score function."""

    def test_equal_ratings(self):
        """Equal ratings should give 50% expected score."""
        assert expected_score(1000, 1000) == pytest.approx(0.5)
        assert expected_score(1500, 1500) == pytest.approx(0.5)

    def test_higher_rating_advantage(self):
        """Higher rated player should have > 50% expected score."""
        assert expected_score(1100, 1000) > 0.5
        assert expected_score(1200, 1000) > expected_score(1100, 1000)

    def test_lower_rating_disadvantage(self):
        """Lower rated player should have < 50% expected score."""
        assert expected_score(1000, 1100) < 0.5

    def test_symmetry(self):
        """Expected scores should sum to 1."""
        e_a = expected_score(1100, 1000)
        e_b = expected_score(1000, 1100)
        assert e_a + e_b == pytest.approx(1.0)

    def test_standard_400_difference(self):
        """400 point difference should give ~91% expected score."""
        # 400 points difference: E = 1 / (1 + 10^(-1)) = 1 / 1.1 ≈ 0.909
        assert expected_score(1400, 1000) == pytest.approx(0.909, rel=0.01)

    def test_standard_200_difference(self):
        """200 point difference should give ~76% expected score."""
        # 200 points difference: E = 1 / (1 + 10^(-0.5)) ≈ 0.76
        assert expected_score(1200, 1000) == pytest.approx(0.76, rel=0.01)


class TestEloRating:
    """Tests for EloRating dataclass."""

    def test_default_values(self):
        """Test default rating values."""
        rating = EloRating()
        assert rating.rating == 1000.0
        assert rating.games_played == 0

    def test_k_factor_provisional(self):
        """K-factor should be 32 for provisional players (<30 games)."""
        rating = EloRating(games_played=0)
        assert rating.k_factor == 32.0

        rating = EloRating(games_played=29)
        assert rating.k_factor == 32.0

    def test_k_factor_established(self):
        """K-factor should be 16 for established players (>=30 games)."""
        rating = EloRating(games_played=30)
        assert rating.k_factor == 16.0

        rating = EloRating(games_played=100)
        assert rating.k_factor == 16.0


class TestUpdateRating:
    """Tests for update_rating function."""

    def test_win_increases_rating(self):
        """Winning should increase rating."""
        rating = EloRating(rating=1000, games_played=0)
        new_rating = update_rating(rating, 1000, 1.0)
        assert new_rating.rating > rating.rating
        assert new_rating.games_played == 1

    def test_loss_decreases_rating(self):
        """Losing should decrease rating."""
        rating = EloRating(rating=1000, games_played=0)
        new_rating = update_rating(rating, 1000, 0.0)
        assert new_rating.rating < rating.rating

    def test_draw_equal_ratings_no_change(self):
        """Draw against equal opponent should not change rating significantly."""
        rating = EloRating(rating=1000, games_played=0)
        new_rating = update_rating(rating, 1000, 0.5)
        # With K=32 and E=0.5, change = 32 * (0.5 - 0.5) = 0
        assert new_rating.rating == pytest.approx(rating.rating, abs=0.001)

    def test_upset_win_big_gain(self):
        """Winning against higher rated opponent gives bigger gain."""
        rating = EloRating(rating=1000, games_played=0)

        # Win against equal opponent
        new_rating_equal = update_rating(rating, 1000, 1.0)

        # Win against stronger opponent
        new_rating_upset = update_rating(rating, 1200, 1.0)

        assert new_rating_upset.rating > new_rating_equal.rating


class TestUpdateRatingsFromMatch:
    """Tests for update_ratings_from_match function."""

    def test_clean_sweep(self):
        """Test when one player wins all games."""
        r1 = EloRating(rating=1000, games_played=0)
        r2 = EloRating(rating=1000, games_played=0)

        new_r1, new_r2 = update_ratings_from_match(r1, r2, wins1=10, wins2=0, draws=0)

        assert new_r1.rating > 1000
        assert new_r2.rating < 1000
        assert new_r1.games_played == 10
        assert new_r2.games_played == 10

    def test_even_match(self):
        """Test when players split games evenly."""
        r1 = EloRating(rating=1000, games_played=0)
        r2 = EloRating(rating=1000, games_played=0)

        new_r1, new_r2 = update_ratings_from_match(r1, r2, wins1=5, wins2=5, draws=0)

        # Ratings should remain roughly equal
        # Note: Some drift is expected due to sequential processing
        assert abs(new_r1.rating - new_r2.rating) < 100

    def test_all_draws(self):
        """Test when all games are draws."""
        r1 = EloRating(rating=1000, games_played=0)
        r2 = EloRating(rating=1000, games_played=0)

        new_r1, new_r2 = update_ratings_from_match(r1, r2, wins1=0, wins2=0, draws=10)

        # Ratings should stay close to original
        assert new_r1.rating == pytest.approx(1000, abs=10)
        assert new_r2.rating == pytest.approx(1000, abs=10)

    def test_rating_conservation(self):
        """Total rating change should approximately sum to zero."""
        r1 = EloRating(rating=1000, games_played=0)
        r2 = EloRating(rating=1000, games_played=0)

        new_r1, new_r2 = update_ratings_from_match(r1, r2, wins1=7, wins2=3, draws=2)

        # Total rating change should be close to zero
        # (not exact due to K-factor changes during match)
        total_before = r1.rating + r2.rating
        total_after = new_r1.rating + new_r2.rating
        assert abs(total_after - total_before) < 50


class TestComputeAllRatings:
    """Tests for compute_all_ratings function."""

    def test_empty_matches(self):
        """Test with no matches."""
        ratings = compute_all_ratings([])
        assert ratings == {}

    def test_single_match(self):
        """Test with a single match."""
        matches = [{
            "model1_version": "initial",
            "model2_version": "iter_001",
            "model1_wins": 3,
            "model2_wins": 7,
            "draws": 0,
        }]

        ratings = compute_all_ratings(matches, anchor_model="initial")

        assert "initial" in ratings
        assert "iter_001" in ratings
        assert ratings["initial"].rating == pytest.approx(1000.0)  # Anchored
        assert ratings["iter_001"].rating > 1000.0  # Won more

    def test_anchor_normalization(self):
        """Test that anchor model stays at anchor rating."""
        matches = [
            {
                "model1_version": "initial",
                "model2_version": "iter_001",
                "model1_wins": 2,
                "model2_wins": 8,
                "draws": 0,
            },
            {
                "model1_version": "iter_001",
                "model2_version": "iter_002",
                "model1_wins": 3,
                "model2_wins": 7,
                "draws": 0,
            },
        ]

        ratings = compute_all_ratings(matches, anchor_model="initial", anchor_rating=1000.0)

        assert ratings["initial"].rating == pytest.approx(1000.0)

    def test_chain_of_improvements(self):
        """Test that later models can be rated higher than earlier ones."""
        matches = [
            {"model1_version": "initial", "model2_version": "iter_001",
             "model1_wins": 2, "model2_wins": 8, "draws": 0},
            {"model1_version": "iter_001", "model2_version": "iter_002",
             "model1_wins": 2, "model2_wins": 8, "draws": 0},
            {"model1_version": "iter_002", "model2_version": "iter_003",
             "model1_wins": 2, "model2_wins": 8, "draws": 0},
        ]

        ratings = compute_all_ratings(matches, anchor_model="initial")

        # Each model should be stronger than the previous
        assert ratings["iter_001"].rating > ratings["initial"].rating
        assert ratings["iter_002"].rating > ratings["iter_001"].rating
        assert ratings["iter_003"].rating > ratings["iter_002"].rating


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_elo_win_probability(self):
        """Test win probability calculation."""
        # Same as expected_score
        assert elo_win_probability(1000, 1000) == pytest.approx(0.5)
        assert elo_win_probability(1400, 1000) == pytest.approx(0.909, rel=0.01)

    def test_rating_difference_for_win_rate(self):
        """Test rating difference calculation."""
        # 50% win rate = 0 difference
        assert rating_difference_for_win_rate(0.5) == pytest.approx(0.0, abs=0.001)

        # ~76% win rate = 200 point difference
        diff = rating_difference_for_win_rate(0.76)
        assert diff == pytest.approx(200.0, rel=0.05)

        # ~91% win rate = 400 point difference
        diff = rating_difference_for_win_rate(0.909)
        assert diff == pytest.approx(400.0, rel=0.05)

    def test_rating_difference_invalid_input(self):
        """Test rating difference with invalid inputs."""
        with pytest.raises(ValueError):
            rating_difference_for_win_rate(0.0)
        with pytest.raises(ValueError):
            rating_difference_for_win_rate(1.0)
        with pytest.raises(ValueError):
            rating_difference_for_win_rate(-0.1)
        with pytest.raises(ValueError):
            rating_difference_for_win_rate(1.1)

    def test_format_rating(self):
        """Test rating formatting."""
        rating = EloRating(rating=1234.56, games_played=42)
        formatted = format_rating(rating)
        assert "1235" in formatted  # Rounded
        assert "42" in formatted

    def test_get_iteration_from_version(self):
        """Test iteration extraction from version string."""
        assert get_iteration_from_version("initial") == 0
        assert get_iteration_from_version("iter_001") == 1
        assert get_iteration_from_version("iter_099") == 99
        assert get_iteration_from_version("iter_100") == 100
        assert get_iteration_from_version("unknown") == -1
        assert get_iteration_from_version("iter_abc") == -1
