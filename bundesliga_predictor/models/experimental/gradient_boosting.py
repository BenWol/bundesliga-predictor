"""
Gradient Boosting Model for Bundesliga prediction.

Often outperforms Random Forest for tabular data with complex interactions.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from typing import Dict, Tuple, Optional, Any

from ..base import BaseModel


class GradientBoostingModel(BaseModel):
    """
    Gradient Boosting model - often outperforms Random Forest for
    tabular data with complex interactions.
    """

    def __init__(self):
        super().__init__(
            name="Gradient Boosting",
            model_id="gradient_boosting"
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
        """Train separate GB models for home and away goals."""
        self._home_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            min_samples_split=10,
            subsample=0.8,
            random_state=42
        )

        self._away_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            min_samples_split=10,
            subsample=0.8,
            random_state=43
        )

        self._home_model.fit(X, y_home)
        self._away_model.fit(X, y_away)
        self._is_trained = True

        return {'n_samples': len(X)}

    def predict(self, X: np.ndarray, **kwargs) -> Tuple[int, int]:
        """Predict using trained GB models."""
        if not self._is_trained:
            raise RuntimeError("Model must be trained first")

        home_pred = self._home_model.predict(X)[0]
        away_pred = self._away_model.predict(X)[0]

        return (max(0, int(round(home_pred))), max(0, int(round(away_pred))))

    def get_expected_goals(self, X: np.ndarray) -> Tuple[float, float]:
        """Get raw expected goals predictions."""
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
