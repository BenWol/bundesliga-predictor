"""
Smart Odds Model for Bundesliga prediction.

Enhanced odds-based model that predicts diverse scorelines based on
odds strength, not just favorite detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any

from ..base import BaseModel


class SmartOddsModel(BaseModel):
    """
    Enhanced odds-based model that predicts more diverse scorelines
    based on odds strength, not just favorite detection.
    """

    def __init__(self):
        super().__init__(
            name="Smart Odds",
            model_id="smart_odds"
        )
        self._is_trained = True  # No training needed

        # Scoreline mappings based on favorite strength
        self.scorelines = {
            'strong_home': [(3, 1), (2, 0), (3, 0)],
            'medium_home': [(2, 1), (1, 0)],
            'strong_away': [(1, 3), (0, 2), (0, 3)],
            'medium_away': [(1, 2), (0, 1)],
            'balanced': [(1, 1), (2, 2), (0, 0)]
        }

    def train(
        self,
        X: np.ndarray,
        y_home: np.ndarray,
        y_away: np.ndarray,
        scorelines: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """No training needed."""
        return {'n_samples': len(X)}

    def predict(
        self,
        X: np.ndarray,
        odds_home: Optional[float] = None,
        odds_draw: Optional[float] = None,
        odds_away: Optional[float] = None,
        **kwargs
    ) -> Tuple[int, int]:
        """Predict based on odds strength with diverse scorelines."""
        if odds_home is None or odds_draw is None or odds_away is None:
            return (1, 1)

        try:
            if pd.isna(odds_home) or pd.isna(odds_draw) or pd.isna(odds_away):
                return (1, 1)
        except (TypeError, ValueError):
            return (1, 1)

        # Calculate implied probabilities
        total = 1/odds_home + 1/odds_draw + 1/odds_away
        prob_home = (1/odds_home) / total
        prob_draw = (1/odds_draw) / total
        prob_away = (1/odds_away) / total

        # Calculate expected goals from odds
        home_xg = -np.log(prob_away + prob_draw * 0.5) * 1.2
        away_xg = -np.log(prob_home + prob_draw * 0.5) * 1.2
        home_xg = np.clip(home_xg, 0.5, 4.0)
        away_xg = np.clip(away_xg, 0.5, 4.0)

        # Use expected goals directly for prediction
        home_score = int(np.round(home_xg))
        away_score = int(np.round(away_xg))

        return (home_score, away_score)

    def predict_with_details(
        self,
        X: np.ndarray,
        odds_home: Optional[float] = None,
        odds_draw: Optional[float] = None,
        odds_away: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Get prediction with additional details."""
        home, away = self.predict(X, odds_home=odds_home, odds_draw=odds_draw, odds_away=odds_away)

        details = {
            'home_score': home,
            'away_score': away,
            'scoreline': f"{home}-{away}",
            'model_id': self.model_id,
            'model_name': self.name,
        }

        if odds_home and odds_draw and odds_away:
            try:
                total = 1/odds_home + 1/odds_draw + 1/odds_away
                details['prob_home'] = (1/odds_home) / total
                details['prob_draw'] = (1/odds_draw) / total
                details['prob_away'] = (1/odds_away) / total
            except (TypeError, ValueError, ZeroDivisionError):
                pass

        return details
