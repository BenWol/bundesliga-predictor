"""
Multi-Class Classification Model for scoreline prediction.
"""

from typing import Tuple, Optional, Dict, Any, List
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .base import BaseModel
from ..config import RF_N_ESTIMATORS, RF_MAX_DEPTH_CLASSIFICATION, MODEL_RANDOM_STATE


class MultiClassClassificationModel(BaseModel):
    """
    Multi-class classification model that predicts scorelines directly.

    Each unique scoreline (e.g., "2-1", "0-0") is a class.
    """

    def __init__(self):
        super().__init__(
            name="Multi-Class Classification",
            model_id="model2"
        )
        self._model: Optional[RandomForestClassifier] = None

    def train(
        self,
        X: np.ndarray,
        y_home: np.ndarray,
        y_away: np.ndarray,
        scorelines: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Train the classification model.

        Args:
            X: Feature matrix
            y_home: Not directly used (scorelines used instead)
            y_away: Not directly used (scorelines used instead)
            scorelines: Scoreline strings (e.g., "2-1") - required

        Returns:
            Dictionary with training metrics
        """
        if scorelines is None:
            # Generate scorelines from scores if not provided
            scorelines = np.array([
                f"{min(int(h), 5)}-{min(int(a), 5)}"
                for h, a in zip(y_home, y_away)
            ])

        self._model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH_CLASSIFICATION,
            random_state=MODEL_RANDOM_STATE,
            n_jobs=-1,
            min_samples_split=5
        )

        self._model.fit(X, scorelines)
        self._is_trained = True

        # Get class distribution
        unique, counts = np.unique(scorelines, return_counts=True)
        top_5 = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:5]

        return {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_classes': len(unique),
            'top_5_scorelines': top_5,
        }

    def predict(self, X: np.ndarray, **kwargs) -> Tuple[int, int]:
        """
        Predict scoreline.

        Args:
            X: Feature vector for single match

        Returns:
            Tuple of (predicted_home_score, predicted_away_score)
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model must be trained before prediction")

        scoreline = self._model.predict(X)[0]
        home_score, away_score = map(int, scoreline.split('-'))

        return home_score, away_score

    def predict_with_details(self, X: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Get prediction with probability distribution."""
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model must be trained before prediction")

        scoreline = self._model.predict(X)[0]
        home_score, away_score = map(int, scoreline.split('-'))

        # Get probabilities for top scorelines
        proba = self._model.predict_proba(X)[0]
        top_3_idx = np.argsort(proba)[-3:][::-1]
        top_3 = [
            (self._model.classes_[i], float(proba[i]))
            for i in top_3_idx
        ]

        return {
            'home_score': home_score,
            'away_score': away_score,
            'scoreline': scoreline,
            'model_id': self.model_id,
            'model_name': self.name,
            'top_3_scorelines': top_3,
            'probability': float(proba[np.where(self._model.classes_ == scoreline)[0][0]]),
        }

    def get_top_n_predictions(self, X: np.ndarray, n: int = 3) -> List[Tuple[str, float]]:
        """
        Get top N most likely scorelines with probabilities.

        Args:
            X: Feature vector for single match
            n: Number of predictions to return

        Returns:
            List of (scoreline, probability) tuples
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model must be trained before prediction")

        proba = self._model.predict_proba(X)[0]
        top_idx = np.argsort(proba)[-n:][::-1]

        return [
            (self._model.classes_[i], float(proba[i]))
            for i in top_idx
        ]
