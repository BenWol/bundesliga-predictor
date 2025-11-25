"""
Base model class for all prediction models.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import numpy as np

from ..scoring import calculate_kicktipp_points


class BaseModel(ABC):
    """
    Abstract base class for all prediction models.

    All models must implement:
    - train(): Train the model on data
    - predict(): Make a prediction for a single match
    - name: Human-readable model name
    """

    def __init__(self, name: str, model_id: str):
        """
        Initialize base model.

        Args:
            name: Human-readable model name
            model_id: Short identifier (e.g., 'model1')
        """
        self.name = name
        self.model_id = model_id
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self._is_trained

    @abstractmethod
    def train(
        self,
        X: np.ndarray,
        y_home: np.ndarray,
        y_away: np.ndarray,
        scorelines: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Train the model.

        Args:
            X: Feature matrix
            y_home: Home scores
            y_away: Away scores
            scorelines: Scoreline strings (e.g., "2-1") - needed for classification

        Returns:
            Dictionary with training metrics
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> Tuple[int, int]:
        """
        Predict score for a single match.

        Args:
            X: Feature vector for single match
            **kwargs: Additional arguments (e.g., odds for naive model)

        Returns:
            Tuple of (predicted_home_score, predicted_away_score)
        """
        pass

    def predict_with_details(self, X: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Predict with additional details.

        Args:
            X: Feature vector for single match
            **kwargs: Additional arguments

        Returns:
            Dictionary with prediction and additional info
        """
        home, away = self.predict(X, **kwargs)
        return {
            'home_score': home,
            'away_score': away,
            'scoreline': f"{home}-{away}",
            'model_id': self.model_id,
            'model_name': self.name,
        }

    def evaluate(
        self,
        X: np.ndarray,
        y_home: np.ndarray,
        y_away: np.ndarray,
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate model on test data using Kicktipp scoring.

        Args:
            X: Feature matrix
            y_home: Actual home scores
            y_away: Actual away scores
            **kwargs: Additional arguments for predict

        Returns:
            Dictionary with evaluation metrics
        """
        total_points = 0
        exact_matches = 0
        goal_diff_matches = 0
        tendency_matches = 0

        for i in range(len(X)):
            pred_home, pred_away = self.predict(X[i:i+1], **kwargs)
            actual_home, actual_away = int(y_home[i]), int(y_away[i])

            points = calculate_kicktipp_points(pred_home, pred_away, actual_home, actual_away)
            total_points += points

            if points == 4:
                exact_matches += 1
            elif points == 3:
                goal_diff_matches += 1
            elif points == 2:
                tendency_matches += 1

        n_matches = len(X)
        max_points = n_matches * 4

        return {
            'total_points': total_points,
            'max_possible': max_points,
            'avg_points_per_match': total_points / n_matches if n_matches > 0 else 0,
            'kicktipp_percentage': (total_points / max_points) * 100 if max_points > 0 else 0,
            'exact_matches': exact_matches,
            'goal_diff_matches': goal_diff_matches,
            'tendency_matches': tendency_matches,
            'wrong_predictions': n_matches - exact_matches - goal_diff_matches - tendency_matches,
            'exact_accuracy': exact_matches / n_matches if n_matches > 0 else 0,
        }

    def __repr__(self) -> str:
        status = "trained" if self._is_trained else "not trained"
        return f"{self.__class__.__name__}(name='{self.name}', {status})"
