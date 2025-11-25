"""
Kicktipp scoring system implementation.
"""

from .config import POINTS_EXACT, POINTS_GOAL_DIFF, POINTS_TENDENCY, POINTS_WRONG


def calculate_kicktipp_points(
    predicted_home: int,
    predicted_away: int,
    actual_home: int,
    actual_away: int
) -> int:
    """
    Calculate points according to Kicktipp.de rules.

    Args:
        predicted_home: Predicted home team score
        predicted_away: Predicted away team score
        actual_home: Actual home team score
        actual_away: Actual away team score

    Returns:
        Points earned (4 = exact, 3 = goal diff, 2 = tendency, 0 = wrong)
    """
    # Exact result
    if predicted_home == actual_home and predicted_away == actual_away:
        return POINTS_EXACT

    # Calculate differences
    pred_diff = predicted_home - predicted_away
    actual_diff = actual_home - actual_away

    # Right goal difference
    if pred_diff == actual_diff:
        return POINTS_GOAL_DIFF

    # Right tendency (win/draw/loss)
    pred_tendency = 0 if pred_diff == 0 else (1 if pred_diff > 0 else -1)
    actual_tendency = 0 if actual_diff == 0 else (1 if actual_diff > 0 else -1)

    if pred_tendency == actual_tendency:
        return POINTS_TENDENCY

    return POINTS_WRONG


def get_tendency(home_score: int, away_score: int) -> str:
    """Get match tendency as string."""
    if home_score > away_score:
        return 'home_win'
    elif away_score > home_score:
        return 'away_win'
    return 'draw'
