"""
Poisson Regression Model for goal-based prediction.
"""

from typing import Tuple, Optional, Dict, Any, List
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import poisson

from .base import BaseModel
from ..config import RF_N_ESTIMATORS, RF_MAX_DEPTH_POISSON, MODEL_RANDOM_STATE


class PoissonRegressionModel(BaseModel):
    """
    Poisson-based model that predicts expected goals (lambda) for each team.

    Uses separate Random Forest regressors for home and away expected goals,
    then applies Poisson distribution to get most likely scoreline.
    """

    def __init__(self):
        super().__init__(
            name="Poisson Regression",
            model_id="model3"
        )
        self._home_model: Optional[RandomForestRegressor] = None
        self._away_model: Optional[RandomForestRegressor] = None

    def train(
        self,
        X: np.ndarray,
        y_home: np.ndarray,
        y_away: np.ndarray,
        scorelines: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Train separate models for home and away goals.

        Args:
            X: Feature matrix
            y_home: Home scores
            y_away: Away scores
            scorelines: Not used for this model

        Returns:
            Dictionary with training metrics
        """
        self._home_model = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH_POISSON,
            random_state=MODEL_RANDOM_STATE,
            n_jobs=-1
        )

        self._away_model = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH_POISSON,
            random_state=MODEL_RANDOM_STATE + 1,
            n_jobs=-1
        )

        self._home_model.fit(X, y_home)
        self._away_model.fit(X, y_away)
        self._is_trained = True

        return {
            'n_samples': len(X),
            'n_features': X.shape[1],
        }

    def predict(self, X: np.ndarray, **kwargs) -> Tuple[int, int]:
        """
        Predict most likely scoreline using Poisson distribution.

        Args:
            X: Feature vector for single match

        Returns:
            Tuple of (predicted_home_score, predicted_away_score)
        """
        if not self._is_trained or self._home_model is None or self._away_model is None:
            raise RuntimeError("Model must be trained before prediction")

        home_lambda = max(0, self._home_model.predict(X)[0])
        away_lambda = max(0, self._away_model.predict(X)[0])

        # Most likely score is floor of lambda (mode of Poisson)
        home_score = int(np.floor(home_lambda))
        away_score = int(np.floor(away_lambda))

        return home_score, away_score

    def predict_with_details(self, X: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Get prediction with expected goals and probability distribution."""
        if not self._is_trained or self._home_model is None or self._away_model is None:
            raise RuntimeError("Model must be trained before prediction")

        home_lambda = max(0, self._home_model.predict(X)[0])
        away_lambda = max(0, self._away_model.predict(X)[0])

        home_score = int(np.floor(home_lambda))
        away_score = int(np.floor(away_lambda))

        # Calculate probability distribution
        scoreline_probs = self._calculate_scoreline_probs(home_lambda, away_lambda)

        return {
            'home_score': home_score,
            'away_score': away_score,
            'scoreline': f"{home_score}-{away_score}",
            'model_id': self.model_id,
            'model_name': self.name,
            'home_lambda': float(home_lambda),
            'away_lambda': float(away_lambda),
            'top_3_scorelines': scoreline_probs[:3],
        }

    def get_expected_goals(self, X: np.ndarray) -> Tuple[float, float]:
        """
        Get expected goals (lambda) for each team.

        Args:
            X: Feature vector for single match

        Returns:
            Tuple of (home_expected_goals, away_expected_goals)
        """
        if not self._is_trained or self._home_model is None or self._away_model is None:
            raise RuntimeError("Model must be trained before prediction")

        home_lambda = max(0, self._home_model.predict(X)[0])
        away_lambda = max(0, self._away_model.predict(X)[0])

        return float(home_lambda), float(away_lambda)

    def _calculate_scoreline_probs(
        self,
        home_lambda: float,
        away_lambda: float,
        max_goals: int = 6
    ) -> List[Tuple[str, float]]:
        """
        Calculate probability for each scoreline.

        Args:
            home_lambda: Expected home goals
            away_lambda: Expected away goals
            max_goals: Maximum goals to consider

        Returns:
            List of (scoreline, probability) sorted by probability
        """
        probs = {}
        for h in range(max_goals):
            for a in range(max_goals):
                prob = poisson.pmf(h, home_lambda) * poisson.pmf(a, away_lambda)
                probs[f"{h}-{a}"] = prob

        return sorted(probs.items(), key=lambda x: x[1], reverse=True)
