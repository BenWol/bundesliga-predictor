"""
Context-Aware Model that adjusts predictions based on match context.

This model wraps base predictions and adjusts them for:
- Derby matches (more goals, more unpredictable)
- Relegation battles (tighter, fewer goals)
- Title race matches (high-scoring, favorite usually wins)
- Big club dynamics (expectations affect performance)
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np

from .base import BaseModel


class ContextAwareModel(BaseModel):
    """
    Context-aware model that adjusts predictions based on match context.

    Uses feature indices to detect context and adjust accordingly.
    """

    def __init__(self, base_model: BaseModel):
        """
        Initialize with a base model to wrap.

        Args:
            base_model: The underlying prediction model
        """
        super().__init__(
            name=f"Context-Aware {base_model.name}",
            model_id=f"context_{base_model.model_id}"
        )
        self.base_model = base_model

        # Feature indices for context features (at end of feature vector)
        # Features are: [19 historical] + [20 odds] + [10 context]
        self.idx_is_derby = 39
        self.idx_home_big_club = 40
        self.idx_away_big_club = 41
        self.idx_home_momentum = 42
        self.idx_away_momentum = 43
        self.idx_home_consistency = 44
        self.idx_away_consistency = 45
        self.idx_position_diff = 46
        self.idx_title_race = 47
        self.idx_relegation = 48

    def train(
        self,
        X: np.ndarray,
        y_home: np.ndarray,
        y_away: np.ndarray,
        scorelines: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Train the underlying base model."""
        # Train on original features (first 39)
        X_base = X[:, :39] if X.shape[1] > 39 else X
        result = self.base_model.train(X_base, y_home, y_away, scorelines)
        self._is_trained = True
        return result

    def predict(self, X: np.ndarray, **kwargs) -> Tuple[int, int]:
        """Predict with context adjustments."""
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction")

        # Get base prediction
        X_base = X[:, :39] if X.shape[1] > 39 else X
        base_home, base_away = self.base_model.predict(X_base, **kwargs)

        # If no context features, return base prediction
        if X.shape[1] <= 39:
            return (base_home, base_away)

        # Extract context features
        context = self._extract_context(X)

        # Apply adjustments
        adj_home, adj_away = self._apply_context_adjustments(
            base_home, base_away, context
        )

        return (adj_home, adj_away)

    def _extract_context(self, X: np.ndarray) -> Dict[str, float]:
        """Extract context from feature vector."""
        return {
            'is_derby': X[0, self.idx_is_derby] if X.shape[1] > self.idx_is_derby else 0,
            'home_big_club': X[0, self.idx_home_big_club] if X.shape[1] > self.idx_home_big_club else 0,
            'away_big_club': X[0, self.idx_away_big_club] if X.shape[1] > self.idx_away_big_club else 0,
            'home_momentum': X[0, self.idx_home_momentum] if X.shape[1] > self.idx_home_momentum else 0,
            'away_momentum': X[0, self.idx_away_momentum] if X.shape[1] > self.idx_away_momentum else 0,
            'home_consistency': X[0, self.idx_home_consistency] if X.shape[1] > self.idx_home_consistency else 0.5,
            'away_consistency': X[0, self.idx_away_consistency] if X.shape[1] > self.idx_away_consistency else 0.5,
            'position_diff': X[0, self.idx_position_diff] if X.shape[1] > self.idx_position_diff else 0,
            'title_race': X[0, self.idx_title_race] if X.shape[1] > self.idx_title_race else 0,
            'relegation': X[0, self.idx_relegation] if X.shape[1] > self.idx_relegation else 0,
        }

    def _apply_context_adjustments(
        self,
        home: int,
        away: int,
        context: Dict[str, float]
    ) -> Tuple[int, int]:
        """Apply context-based adjustments to prediction."""

        # Derby matches: tend to be more goals and unpredictable
        if context['is_derby'] > 0.5:
            # Derbies are often high-scoring and tight
            if home == away:  # Draw prediction
                # Derbies rarely end 0-0, more likely 1-1 or 2-2
                if home == 0:
                    home, away = 1, 1
            else:
                # Keep the prediction but maybe increase goals slightly
                total = home + away
                if total < 2:
                    # Increase total goals in derbies
                    if home > away:
                        home += 1
                    else:
                        away += 1

        # Relegation battles: tighter, more defensive
        if context['relegation'] > 0.5:
            # Relegation matches tend to be low-scoring
            total = home + away
            if total > 3:
                # Reduce goals
                diff = home - away
                if diff > 0:
                    home = 2
                    away = 1
                elif diff < 0:
                    home = 1
                    away = 2
                else:
                    home = away = 1

        # Title race: favorites usually deliver
        if context['title_race'] > 0.5:
            # Title contenders are more reliable
            if context['position_diff'] > 0.3:  # Home team significantly better
                if home <= away:  # Model thinks draw or away win
                    home = max(home + 1, 2)
            elif context['position_diff'] < -0.3:  # Away team significantly better
                if away <= home:
                    away = max(away + 1, 2)

        # Momentum adjustment
        momentum_diff = context['home_momentum'] - context['away_momentum']
        if abs(momentum_diff) > 0.3:
            # Strong momentum difference
            if momentum_diff > 0.3 and home <= away:
                # Home team in better form, boost slightly
                if home == away:
                    home += 1
            elif momentum_diff < -0.3 and away <= home:
                # Away team in better form
                if home == away:
                    away += 1

        # Big club underperformance check
        if context['home_big_club'] > 0.5 and context['away_big_club'] < 0.5:
            # Home big club vs smaller team - big clubs sometimes struggle
            if context['home_consistency'] < 0.4:
                # Inconsistent big club - don't trust high predictions
                if home > 2:
                    home = 2

        return (max(0, home), max(0, away))

    def predict_with_details(self, X: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Get prediction with context details."""
        home, away = self.predict(X, **kwargs)

        # Get base prediction for comparison
        X_base = X[:, :39] if X.shape[1] > 39 else X
        base_home, base_away = self.base_model.predict(X_base, **kwargs)

        context = self._extract_context(X) if X.shape[1] > 39 else {}

        return {
            'home_score': home,
            'away_score': away,
            'scoreline': f"{home}-{away}",
            'model_id': self.model_id,
            'model_name': self.name,
            'base_prediction': f"{base_home}-{base_away}",
            'context_applied': context,
            'was_adjusted': (home != base_home or away != base_away),
        }


class DerbySpecialistModel(BaseModel):
    """
    Specialized model for derby matches.

    Uses different logic for derbies vs regular matches.
    Historical analysis shows derbies have:
    - Higher average goals
    - More variance in outcomes
    - Underdogs perform better
    """

    def __init__(self, base_model: BaseModel):
        super().__init__(
            name="Derby Specialist",
            model_id="derby_specialist"
        )
        self.base_model = base_model
        self.idx_is_derby = 39

    def train(self, X, y_home, y_away, scorelines=None):
        """Train base model."""
        X_base = X[:, :39] if X.shape[1] > 39 else X
        result = self.base_model.train(X_base, y_home, y_away, scorelines)
        self._is_trained = True
        return result

    def predict(self, X: np.ndarray, **kwargs) -> Tuple[int, int]:
        """Predict with derby adjustments."""
        if not self._is_trained:
            raise RuntimeError("Model must be trained")

        X_base = X[:, :39] if X.shape[1] > 39 else X
        base_home, base_away = self.base_model.predict(X_base, **kwargs)

        # Check if derby
        is_derby = X[0, self.idx_is_derby] > 0.5 if X.shape[1] > self.idx_is_derby else False

        if not is_derby:
            return (base_home, base_away)

        # Derby adjustments
        # 1. Derbies rarely end 0-0
        if base_home == 0 and base_away == 0:
            return (1, 1)

        # 2. Derbies often have at least 2 total goals
        if base_home + base_away < 2:
            if base_home >= base_away:
                return (max(1, base_home), base_away)
            else:
                return (base_home, max(1, base_away))

        # 3. In derbies, 2-1 and 2-2 are common scorelines
        if base_home > base_away:
            # Home win predicted - 2-1 is most common derby home win
            return (2, 1)
        elif base_away > base_home:
            # Away win - 1-2 is common
            return (1, 2)
        else:
            # Draw - 1-1 or 2-2
            if base_home >= 2:
                return (2, 2)
            return (1, 1)


class RelegationBattleModel(BaseModel):
    """
    Specialized model for relegation battles.

    These matches tend to be:
    - Low-scoring (defensive)
    - Tight (small margins)
    - High stakes = more caution
    """

    def __init__(self, base_model: BaseModel):
        super().__init__(
            name="Relegation Battle",
            model_id="relegation_specialist"
        )
        self.base_model = base_model
        self.idx_relegation = 48

    def train(self, X, y_home, y_away, scorelines=None):
        X_base = X[:, :39] if X.shape[1] > 39 else X
        result = self.base_model.train(X_base, y_home, y_away, scorelines)
        self._is_trained = True
        return result

    def predict(self, X: np.ndarray, **kwargs) -> Tuple[int, int]:
        if not self._is_trained:
            raise RuntimeError("Model must be trained")

        X_base = X[:, :39] if X.shape[1] > 39 else X
        base_home, base_away = self.base_model.predict(X_base, **kwargs)

        # Check if relegation battle
        is_relegation = X[0, self.idx_relegation] > 0.5 if X.shape[1] > self.idx_relegation else False

        if not is_relegation:
            return (base_home, base_away)

        # Relegation battle adjustments
        # 1. Total goals usually 2 or less
        total = base_home + base_away
        if total > 3:
            # Scale down
            ratio = base_home / (total + 0.1)
            base_home = int(round(ratio * 2.5))
            base_away = int(round((1 - ratio) * 2.5))

        # 2. Common scorelines: 1-0, 0-1, 1-1, 2-1, 0-0
        if base_home == base_away:
            # Draw predicted
            return (1, 1) if base_home >= 1 else (0, 0)
        elif base_home > base_away:
            # Home win - usually tight
            return (1, 0) if base_home - base_away == 1 else (2, 1)
        else:
            # Away win
            return (0, 1) if base_away - base_home == 1 else (1, 2)
