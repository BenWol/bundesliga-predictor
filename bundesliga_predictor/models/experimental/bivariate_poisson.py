"""
Bivariate Poisson Model for Bundesliga prediction.

Captures correlation between home and away goals - high-scoring games
tend to be high for both teams.
"""

import numpy as np
from scipy.stats import poisson
from sklearn.ensemble import GradientBoostingRegressor
from typing import Dict, Tuple, Optional, Any

from ..base import BaseModel


class BivariatePoissonModel(BaseModel):
    """
    Bivariate Poisson model that captures correlation between
    home and away goals - high-scoring games tend to be high for both.
    """

    def __init__(self):
        super().__init__(
            name="Bivariate Poisson",
            model_id="bivariate_poisson"
        )
        self._home_model = None
        self._away_model = None
        self._correlation = 0.0  # Learned correlation parameter

    def train(
        self,
        X: np.ndarray,
        y_home: np.ndarray,
        y_away: np.ndarray,
        scorelines: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Train models and estimate correlation."""
        # Train expected goal models
        self._home_model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )

        self._away_model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            random_state=43
        )

        self._home_model.fit(X, y_home)
        self._away_model.fit(X, y_away)

        # Estimate correlation from residuals
        home_pred = self._home_model.predict(X)
        away_pred = self._away_model.predict(X)

        home_resid = y_home - home_pred
        away_resid = y_away - away_pred

        # Correlation of residuals (bounded)
        self._correlation = np.clip(np.corrcoef(home_resid, away_resid)[0, 1], -0.3, 0.3)

        self._is_trained = True
        return {'correlation': self._correlation, 'n_samples': len(X)}

    def predict(self, X: np.ndarray, **kwargs) -> Tuple[int, int]:
        """Predict using bivariate Poisson distribution."""
        if not self._is_trained:
            raise RuntimeError("Model must be trained first")

        home_lambda = max(0.5, self._home_model.predict(X)[0])
        away_lambda = max(0.5, self._away_model.predict(X)[0])

        # Find most likely scoreline considering correlation
        best_score = None
        best_prob = 0

        for h in range(6):
            for a in range(6):
                # Bivariate Poisson approximation with correlation adjustment
                base_prob = poisson.pmf(h, home_lambda) * poisson.pmf(a, away_lambda)

                # Adjust for correlation - increase prob of both high or both low
                mean_h, mean_a = home_lambda, away_lambda
                corr_adj = 1 + self._correlation * ((h - mean_h) * (a - mean_a)) / (mean_h * mean_a + 1e-6)
                corr_adj = max(0.5, min(1.5, corr_adj))

                adj_prob = base_prob * corr_adj

                if adj_prob > best_prob:
                    best_prob = adj_prob
                    best_score = (h, a)

        return best_score or (1, 1)

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
            'correlation': self._correlation,
        }
