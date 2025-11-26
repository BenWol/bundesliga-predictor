"""
Probability-Maximizing Model for Bundesliga prediction.

Model that explicitly maximizes expected Kicktipp points
by considering the full probability distribution.
"""

import numpy as np
from scipy.stats import poisson
from sklearn.ensemble import GradientBoostingRegressor
from typing import Dict, Tuple, Optional, Any

from ..base import BaseModel


class ProbabilityMaxModel(BaseModel):
    """
    Model that explicitly maximizes expected Kicktipp points
    by considering the full probability distribution.
    """

    def __init__(self):
        super().__init__(
            name="Probability Max",
            model_id="probability_max"
        )
        self._home_model = None
        self._away_model = None

    def train(
        self,
        X: np.ndarray,
        y_home: np.ndarray,
        y_away: np.ndarray,
        scorelines: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Train xG models."""
        self._home_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )
        self._away_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=43
        )

        self._home_model.fit(X, y_home)
        self._away_model.fit(X, y_away)
        self._is_trained = True

        return {'n_samples': len(X)}

    def predict(self, X: np.ndarray, **kwargs) -> Tuple[int, int]:
        """Find scoreline that maximizes expected Kicktipp points."""
        if not self._is_trained:
            raise RuntimeError("Model must be trained first")

        home_lambda = max(0.5, self._home_model.predict(X)[0])
        away_lambda = max(0.5, self._away_model.predict(X)[0])

        # Calculate probability distribution
        max_goals = 6
        probs = np.zeros((max_goals, max_goals))
        for h in range(max_goals):
            for a in range(max_goals):
                probs[h, a] = poisson.pmf(h, home_lambda) * poisson.pmf(a, away_lambda)

        # For each possible prediction, calculate expected points
        best_pred = (1, 1)
        best_expected = 0

        for pred_h in range(max_goals):
            for pred_a in range(max_goals):
                expected_pts = 0

                for actual_h in range(max_goals):
                    for actual_a in range(max_goals):
                        prob = probs[actual_h, actual_a]
                        pts = self._kicktipp_points(pred_h, pred_a, actual_h, actual_a)
                        expected_pts += prob * pts

                if expected_pts > best_expected:
                    best_expected = expected_pts
                    best_pred = (pred_h, pred_a)

        return best_pred

    def _kicktipp_points(
        self,
        pred_h: int,
        pred_a: int,
        actual_h: int,
        actual_a: int
    ) -> int:
        """Calculate Kicktipp points."""
        if pred_h == actual_h and pred_a == actual_a:
            return 4
        elif (pred_h - pred_a) == (actual_h - actual_a):
            return 3
        elif (pred_h > pred_a and actual_h > actual_a) or \
             (pred_h < pred_a and actual_h < actual_a) or \
             (pred_h == pred_a and actual_h == actual_a):
            return 2
        else:
            return 0

    def get_expected_goals(self, X: np.ndarray) -> Tuple[float, float]:
        """Get expected goals."""
        if not self._is_trained:
            raise RuntimeError("Model must be trained first")

        return (self._home_model.predict(X)[0], self._away_model.predict(X)[0])

    def predict_with_details(self, X: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Get prediction with additional details."""
        home, away = self.predict(X, **kwargs)
        xg_home, xg_away = self.get_expected_goals(X)

        return {
            'home_score': home,
            'away_score': away,
            'scoreline': f"{home}-{away}",
            'model_id': self.model_id,
            'model_name': self.name,
            'xg_home': xg_home,
            'xg_away': xg_away,
        }
