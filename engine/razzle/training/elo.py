"""
ELO rating calculation for model arena evaluation.

Implements standard ELO rating system to measure model strength
across training iterations.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EloRating:
    """ELO rating for a model."""
    rating: float = 1000.0
    games_played: int = 0

    @property
    def k_factor(self) -> float:
        """
        K-factor determines how much ratings change per game.
        Higher K = more volatile ratings.
        Use higher K for players with fewer games (provisional period).
        """
        if self.games_played < 30:
            return 32.0
        return 16.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """
    Calculate expected score for player A against player B.

    Uses the standard ELO formula:
    E_A = 1 / (1 + 10^((R_B - R_A) / 400))

    Args:
        rating_a: ELO rating of player A
        rating_b: ELO rating of player B

    Returns:
        Expected score for player A (0.0 to 1.0)
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_rating(
    rating: EloRating,
    opponent_rating: float,
    score: float
) -> EloRating:
    """
    Update a rating based on a single game result.

    Args:
        rating: Current rating to update
        opponent_rating: Opponent's rating
        score: Actual score (1.0 for win, 0.5 for draw, 0.0 for loss)

    Returns:
        New EloRating with updated values
    """
    expected = expected_score(rating.rating, opponent_rating)
    new_rating = rating.rating + rating.k_factor * (score - expected)

    return EloRating(
        rating=new_rating,
        games_played=rating.games_played + 1
    )


def update_ratings_from_match(
    rating1: EloRating,
    rating2: EloRating,
    wins1: int,
    wins2: int,
    draws: int
) -> tuple[EloRating, EloRating]:
    """
    Update two ratings based on a match result.

    A match consists of multiple games between two players.
    Each game (win, loss, or draw) updates the ratings.

    Args:
        rating1: Rating of player 1
        rating2: Rating of player 2
        wins1: Number of wins for player 1
        wins2: Number of wins for player 2
        draws: Number of draws

    Returns:
        Tuple of (new_rating1, new_rating2)
    """
    r1 = EloRating(rating=rating1.rating, games_played=rating1.games_played)
    r2 = EloRating(rating=rating2.rating, games_played=rating2.games_played)

    # Process player 1 wins
    for _ in range(wins1):
        old_r2 = r2.rating
        r1 = update_rating(r1, old_r2, 1.0)
        r2 = update_rating(r2, r1.rating - r1.k_factor * (1.0 - expected_score(r1.rating - r1.k_factor, old_r2)), 0.0)
        # Simpler: just update both simultaneously based on pre-game ratings
        # Actually let's do it properly - both see the same opponent rating

    # Reset and do it properly - update both players simultaneously per game
    r1 = EloRating(rating=rating1.rating, games_played=rating1.games_played)
    r2 = EloRating(rating=rating2.rating, games_played=rating2.games_played)

    total_games = wins1 + wins2 + draws

    # Process all games, updating both ratings after each
    for _ in range(wins1):
        old_r1, old_r2 = r1.rating, r2.rating
        r1 = EloRating(
            rating=old_r1 + r1.k_factor * (1.0 - expected_score(old_r1, old_r2)),
            games_played=r1.games_played + 1
        )
        r2 = EloRating(
            rating=old_r2 + r2.k_factor * (0.0 - expected_score(old_r2, old_r1)),
            games_played=r2.games_played + 1
        )

    for _ in range(wins2):
        old_r1, old_r2 = r1.rating, r2.rating
        r1 = EloRating(
            rating=old_r1 + r1.k_factor * (0.0 - expected_score(old_r1, old_r2)),
            games_played=r1.games_played + 1
        )
        r2 = EloRating(
            rating=old_r2 + r2.k_factor * (1.0 - expected_score(old_r2, old_r1)),
            games_played=r2.games_played + 1
        )

    for _ in range(draws):
        old_r1, old_r2 = r1.rating, r2.rating
        r1 = EloRating(
            rating=old_r1 + r1.k_factor * (0.5 - expected_score(old_r1, old_r2)),
            games_played=r1.games_played + 1
        )
        r2 = EloRating(
            rating=old_r2 + r2.k_factor * (0.5 - expected_score(old_r2, old_r1)),
            games_played=r2.games_played + 1
        )

    return r1, r2


def compute_all_ratings(
    matches: list[dict],
    anchor_model: str = "initial",
    anchor_rating: float = 1000.0
) -> dict[str, EloRating]:
    """
    Compute all ratings from match history.

    Processes matches in chronological order, updating ratings
    as each match is processed.

    Args:
        matches: List of match dictionaries with keys:
            - model1_version: str
            - model2_version: str
            - model1_wins: int
            - model2_wins: int
            - draws: int
        anchor_model: Model version to anchor at fixed rating
        anchor_rating: Rating to anchor the anchor_model at

    Returns:
        Dictionary mapping model version to EloRating
    """
    ratings: dict[str, EloRating] = {}

    def get_rating(model: str) -> EloRating:
        if model not in ratings:
            ratings[model] = EloRating(rating=anchor_rating, games_played=0)
        return ratings[model]

    # Process matches in order
    for match in matches:
        model1 = match["model1_version"]
        model2 = match["model2_version"]
        wins1 = match["model1_wins"]
        wins2 = match["model2_wins"]
        draws = match["draws"]

        r1 = get_rating(model1)
        r2 = get_rating(model2)

        new_r1, new_r2 = update_ratings_from_match(r1, r2, wins1, wins2, draws)

        ratings[model1] = new_r1
        ratings[model2] = new_r2

    # Normalize ratings so anchor model is at anchor_rating
    if anchor_model in ratings:
        offset = anchor_rating - ratings[anchor_model].rating
        for model in ratings:
            ratings[model] = EloRating(
                rating=ratings[model].rating + offset,
                games_played=ratings[model].games_played
            )

    return ratings


def elo_win_probability(rating_a: float, rating_b: float) -> float:
    """
    Calculate win probability for player A against player B.

    This is just the expected score, which represents the probability
    of winning (or half the probability of drawing).

    Args:
        rating_a: ELO rating of player A
        rating_b: ELO rating of player B

    Returns:
        Win probability for player A (0.0 to 1.0)
    """
    return expected_score(rating_a, rating_b)


def rating_difference_for_win_rate(win_rate: float) -> float:
    """
    Calculate the rating difference needed to achieve a given win rate.

    Args:
        win_rate: Desired win probability (0.0 to 1.0, exclusive of 0 and 1)

    Returns:
        Rating difference (positive means A is stronger)
    """
    if win_rate <= 0.0 or win_rate >= 1.0:
        raise ValueError("Win rate must be between 0 and 1 (exclusive)")

    # From E_A = 1 / (1 + 10^((R_B - R_A) / 400))
    # Solving for R_A - R_B:
    # R_A - R_B = 400 * log10((1 - E_A) / E_A) * -1
    # R_A - R_B = 400 * log10(E_A / (1 - E_A))
    import math
    return 400.0 * math.log10(win_rate / (1.0 - win_rate))


def format_rating(rating: EloRating) -> str:
    """Format a rating for display."""
    return f"{rating.rating:.0f} ({rating.games_played} games)"


def get_iteration_from_version(version: str) -> int:
    """
    Extract iteration number from model version string.

    Examples:
        "initial" -> 0
        "iter_001" -> 1
        "iter_099" -> 99

    Args:
        version: Model version string

    Returns:
        Iteration number (0 for initial model)
    """
    if version == "initial":
        return 0
    if version.startswith("iter_"):
        try:
            return int(version[5:])
        except ValueError:
            return -1
    return -1
