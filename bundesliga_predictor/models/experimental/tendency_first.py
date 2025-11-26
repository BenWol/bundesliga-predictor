"""
Tendency-First Model for Bundesliga prediction.

Two-stage model: First predict tendency (home/draw/away),
then predict most likely scoreline within that tendency.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from collections import Counter
from typing import Dict, Tuple, Optional, Any

from ..base import BaseModel


class TendencyFirstModel(BaseModel):
    """
    Two-stage model: First predict tendency (home/draw/away),
    then predict most likely scoreline within that tendency.
    """

    def __init__(self):
        super().__init__(
            name="Tendency First",
            model_id="tendency_first"
        )
        self._tendency_model = None
        self._home_scorelines = {}  # tendency -> [(scoreline, prob), ...]
        self._xg_model_home = None
        self._xg_model_away = None

    def train(
        self,
        X: np.ndarray,
        y_home: np.ndarray,
        y_away: np.ndarray,
        scorelines: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Train tendency classifier and learn scoreline distributions."""
        # Create tendency labels
        tendencies = []
        for h, a in zip(y_home, y_away):
            if h > a:
                tendencies.append('H')
            elif h < a:
                tendencies.append('A')
            else:
                tendencies.append('D')

        tendencies = np.array(tendencies)

        # Train tendency classifier
        self._tendency_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )
        self._tendency_model.fit(X, tendencies)

        # Train xG models for score prediction
        self._xg_model_home = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )
        self._xg_model_away = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            random_state=43
        )
        self._xg_model_home.fit(X, y_home)
        self._xg_model_away.fit(X, y_away)

        # Learn scoreline distributions per tendency
        for tendency in ['H', 'D', 'A']:
            mask = tendencies == tendency
            scores = [f"{int(h)}-{int(a)}" for h, a in zip(y_home[mask], y_away[mask])]
            counter = Counter(scores)
            total = sum(counter.values())
            self._home_scorelines[tendency] = [
                (s.split('-'), c/total) for s, c in counter.most_common(10)
            ]

        self._is_trained = True
        return {'n_samples': len(X)}

    def predict(self, X: np.ndarray, **kwargs) -> Tuple[int, int]:
        """Predict tendency first, then scoreline."""
        if not self._is_trained:
            raise RuntimeError("Model must be trained first")

        # Predict tendency
        tendency_probs = self._tendency_model.predict_proba(X)[0]
        classes = self._tendency_model.classes_
        tendency = classes[np.argmax(tendency_probs)]

        # Get xG predictions
        home_xg = max(0, self._xg_model_home.predict(X)[0])
        away_xg = max(0, self._xg_model_away.predict(X)[0])

        # Adjust xG based on tendency
        if tendency == 'H':
            # Ensure home wins
            home_score = max(1, int(round(home_xg)))
            away_score = min(home_score - 1, int(round(away_xg)))
            away_score = max(0, away_score)
        elif tendency == 'A':
            # Ensure away wins
            away_score = max(1, int(round(away_xg)))
            home_score = min(away_score - 1, int(round(home_xg)))
            home_score = max(0, home_score)
        else:
            # Draw - same scores
            avg = (home_xg + away_xg) / 2
            home_score = away_score = int(round(avg))

        return (home_score, away_score)

    def get_tendency_probabilities(self, X: np.ndarray) -> Dict[str, float]:
        """Get tendency probabilities."""
        if not self._is_trained:
            raise RuntimeError("Model must be trained first")

        probs = self._tendency_model.predict_proba(X)[0]
        classes = self._tendency_model.classes_
        return {c: float(p) for c, p in zip(classes, probs)}

    def predict_with_details(self, X: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Get prediction with additional details."""
        home, away = self.predict(X, **kwargs)
        tendency_probs = self.get_tendency_probabilities(X)

        return {
            'home_score': home,
            'away_score': away,
            'scoreline': f"{home}-{away}",
            'model_id': self.model_id,
            'model_name': self.name,
            'tendency_probabilities': tendency_probs,
            'predicted_tendency': max(tendency_probs.items(), key=lambda x: x[1])[0],
        }
