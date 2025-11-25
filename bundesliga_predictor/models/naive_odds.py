"""
Naive Odds-Based Model for baseline prediction.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd

from .base import BaseModel
from ..config import DEFAULT_HOME_WIN_SCORE, DEFAULT_DRAW_SCORE, DEFAULT_AWAY_WIN_SCORE


class NaiveOddsModel(BaseModel):
    """
    Naive model that uses betting odds to determine the favorite,
    then predicts a typical scoreline based on the result type.

    This model doesn't require training - it just uses odds directly.
    Serves as a strong baseline that is hard to beat.
    """

    def __init__(self):
        super().__init__(
            name="Naive Odds-based",
            model_id="model4"
        )
        self._is_trained = True  # No training needed

    def train(
        self,
        X: np.ndarray,
        y_home: np.ndarray,
        y_away: np.ndarray,
        scorelines: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        No training needed for naive model.

        Returns:
            Empty metrics dictionary
        """
        return {'note': 'No training required for naive odds model'}

    def predict(
        self,
        X: np.ndarray,
        odds_home: Optional[float] = None,
        odds_draw: Optional[float] = None,
        odds_away: Optional[float] = None,
        **kwargs
    ) -> Tuple[int, int]:
        """
        Predict based on odds.

        Args:
            X: Feature vector (can extract odds from it if not provided separately)
            odds_home: Home win odds
            odds_draw: Draw odds
            odds_away: Away win odds

        Returns:
            Tuple of (predicted_home_score, predicted_away_score)
        """
        # Try to get odds from kwargs or feature vector
        if odds_home is None and len(X.shape) > 1 and X.shape[1] >= 22:
            # Odds are at indices 19, 20, 21 in feature vector
            odds_home = X[0, 19] if not np.isnan(X[0, 19]) else None
            odds_draw = X[0, 20] if not np.isnan(X[0, 20]) else None
            odds_away = X[0, 21] if not np.isnan(X[0, 21]) else None

        return self._predict_from_odds(odds_home, odds_draw, odds_away)

    def _predict_from_odds(
        self,
        odds_home: Optional[float],
        odds_draw: Optional[float],
        odds_away: Optional[float]
    ) -> Tuple[int, int]:
        """
        Predict scoreline based on odds.

        Logic:
        - If home has lowest odd → predict 2-1 (home win)
        - If away has lowest odd → predict 1-2 (away win)
        - If draw has lowest odd or home/away very close → predict 1-1

        Args:
            odds_home: Home win odds
            odds_draw: Draw odds
            odds_away: Away win odds

        Returns:
            Predicted (home_score, away_score)
        """
        # Default to draw if no valid odds
        if (odds_home is None or odds_draw is None or odds_away is None or
            pd.isna(odds_home) or pd.isna(odds_draw) or pd.isna(odds_away)):
            return DEFAULT_DRAW_SCORE

        # Find minimum odd (favorite)
        min_odd = min(odds_home, odds_draw, odds_away)

        # Check if home and away odds are very close (within 5%)
        if abs(odds_home - odds_away) / min(odds_home, odds_away) < 0.05:
            return DEFAULT_DRAW_SCORE

        # Return scoreline based on favorite
        if min_odd == odds_home:
            return DEFAULT_HOME_WIN_SCORE
        elif min_odd == odds_away:
            return DEFAULT_AWAY_WIN_SCORE
        else:
            return DEFAULT_DRAW_SCORE

    def predict_with_details(
        self,
        X: np.ndarray,
        odds_home: Optional[float] = None,
        odds_draw: Optional[float] = None,
        odds_away: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Get prediction with odds analysis."""
        home_score, away_score = self.predict(X, odds_home, odds_draw, odds_away)

        # Determine prediction type
        if home_score > away_score:
            prediction_type = "home_win"
        elif away_score > home_score:
            prediction_type = "away_win"
        else:
            prediction_type = "draw"

        result = {
            'home_score': home_score,
            'away_score': away_score,
            'scoreline': f"{home_score}-{away_score}",
            'model_id': self.model_id,
            'model_name': self.name,
            'prediction_type': prediction_type,
        }

        # Add odds info if available
        if odds_home is not None and not pd.isna(odds_home):
            result['odds'] = {
                'home': odds_home,
                'draw': odds_draw,
                'away': odds_away,
            }
            result['favorite'] = self._determine_favorite(odds_home, odds_draw, odds_away)

        return result

    def _determine_favorite(
        self,
        odds_home: float,
        odds_draw: float,
        odds_away: float
    ) -> str:
        """Determine who is the favorite based on odds."""
        min_odd = min(odds_home, odds_draw, odds_away)
        if min_odd == odds_home:
            return "home"
        elif min_odd == odds_away:
            return "away"
        return "draw"
