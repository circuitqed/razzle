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
    anchor_rating: float = 1000.0,
    iterations: int = 200,
) -> dict[str, EloRating]:
    """
    Compute all ratings from match history using Bradley-Terry MLE.

    Uses iterative Bradley-Terry maximum likelihood estimation, which
    converges to stable ratings regardless of match processing order.

    Args:
        matches: List of match dictionaries with keys:
            - model1_version: str
            - model2_version: str
            - model1_wins: int
            - model2_wins: int
            - draws: int
        anchor_model: Model version to anchor at fixed rating
        anchor_rating: Rating to anchor the anchor_model at
        iterations: Number of Bradley-Terry iterations (default 200)

    Returns:
        Dictionary mapping model version to EloRating
    """
    import math

    # Collect players and game counts
    players: list[str] = []
    player_set: set[str] = set()
    games_played: dict[str, int] = {}

    for match in matches:
        m1 = match["model1_version"]
        m2 = match["model2_version"]
        total = match["model1_wins"] + match["model2_wins"] + match["draws"]
        for p in (m1, m2):
            if p not in player_set:
                player_set.add(p)
                players.append(p)
            games_played[p] = games_played.get(p, 0) + total

    if not players:
        return {}

    n = len(players)
    idx = {p: i for i, p in enumerate(players)}

    # Build win matrix: wins[i][j] = score of i against j
    # Draws count as 0.5 for each side
    wins = [[0.0] * n for _ in range(n)]
    total = [[0.0] * n for _ in range(n)]

    for match in matches:
        i = idx[match["model1_version"]]
        j = idx[match["model2_version"]]
        w1 = match["model1_wins"]
        w2 = match["model2_wins"]
        d = match["draws"]
        wins[i][j] += w1 + d * 0.5
        wins[j][i] += w2 + d * 0.5
        g = w1 + w2 + d
        total[i][j] += g
        total[j][i] += g

    # Bradley-Terry iterative update
    # strength[i] is proportional to P(i beats j) = s[i] / (s[i] + s[j])
    strength = [1.0] * n

    for _ in range(iterations):
        new_strength = [0.0] * n
        for i in range(n):
            w_i = sum(wins[i])
            if w_i == 0:
                new_strength[i] = strength[i]
                continue
            denom = 0.0
            for j in range(n):
                if total[i][j] > 0:
                    denom += total[i][j] / (strength[i] + strength[j])
            new_strength[i] = w_i / denom if denom > 0 else strength[i]
        # Normalize to geometric mean = 1
        log_mean = sum(math.log(max(s, 1e-30)) for s in new_strength) / n
        factor = math.exp(-log_mean)
        strength = [s * factor for s in new_strength]

    # Convert to Elo scale: Elo = 400 * log10(strength) + anchor_rating
    ratings: dict[str, EloRating] = {}
    for p, s in zip(players, strength):
        elo = 400.0 * math.log10(max(s, 1e-30))
        ratings[p] = EloRating(rating=elo, games_played=games_played[p])

    # Normalize so anchor model is at anchor_rating
    # Try exact match first, then prefix match (e.g. "initial" matches "initial_s1")
    anchor_key = None
    if anchor_model in ratings:
        anchor_key = anchor_model
    else:
        # Find lowest-rated player whose name starts with anchor_model
        candidates = [(k, v.rating) for k, v in ratings.items()
                      if k.startswith(anchor_model)]
        if candidates:
            anchor_key = min(candidates, key=lambda x: x[1])[0]

    if anchor_key:
        offset = anchor_rating - ratings[anchor_key].rating
    else:
        # No anchor found — center around anchor_rating
        avg = sum(r.rating for r in ratings.values()) / len(ratings)
        offset = anchor_rating - avg

    for model in ratings:
        ratings[model] = EloRating(
            rating=ratings[model].rating + offset,
            games_played=ratings[model].games_played,
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
