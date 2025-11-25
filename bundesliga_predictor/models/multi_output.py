"""
Multi-Output Regression Model for score prediction.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base import BaseModel
from ..config import RF_N_ESTIMATORS, RF_MAX_DEPTH_REGRESSION, MODEL_RANDOM_STATE


class MultiOutputRegressionModel(BaseModel):
    """
    Multi-output regression model that predicts home and away scores simultaneously.

    Uses Random Forest to directly predict numerical scores.
    """

    def __init__(self):
        super().__init__(
            name="Multi-Output Regression",
            model_id="model1"
        )
        self._model: Optional[RandomForestRegressor] = None

    def train(
        self,
        X: np.ndarray,
        y_home: np.ndarray,
        y_away: np.ndarray,
        scorelines: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Train the multi-output regression model.

        Args:
            X: Feature matrix
            y_home: Home scores
            y_away: Away scores
            scorelines: Not used for this model

        Returns:
            Dictionary with training metrics
        """
        # Combine scores into multi-output target
        y = np.column_stack([y_home, y_away])

        self._model = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH_REGRESSION,
            random_state=MODEL_RANDOM_STATE,
            n_jobs=-1
        )

        self._model.fit(X, y)
        self._is_trained = True

        # Return basic metrics
        return {
            'n_samples': len(X),
            'n_features': X.shape[1],
        }

    def predict(self, X: np.ndarray, **kwargs) -> Tuple[int, int]:
        """
        Predict home and away scores.

        Args:
            X: Feature vector for single match

        Returns:
            Tuple of (predicted_home_score, predicted_away_score)
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model must be trained before prediction")

        pred = self._model.predict(X)[0]
        home_score = max(0, int(round(pred[0])))
        away_score = max(0, int(round(pred[1])))

        return home_score, away_score

    def predict_with_details(self, X: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Get prediction with raw regression values."""
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model must be trained before prediction")

        pred = self._model.predict(X)[0]
        home_score = max(0, int(round(pred[0])))
        away_score = max(0, int(round(pred[1])))

        return {
            'home_score': home_score,
            'away_score': away_score,
            'scoreline': f"{home_score}-{away_score}",
            'model_id': self.model_id,
            'model_name': self.name,
            'raw_home': float(pred[0]),
            'raw_away': float(pred[1]),
        }
