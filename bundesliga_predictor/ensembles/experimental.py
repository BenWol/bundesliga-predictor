"""
Experimental ensemble strategies.

These ensembles are being tested for potential inclusion in the main
prediction pipeline. Use them in backtest.py to compare performance.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional


class OptimizedConsensusEnsemble:
    """
    Optimized consensus ensemble with:
    1. Dynamic threshold based on historical accuracy
    2. Weighted consensus (better models count more)
    3. Smarter fallback selection
    """

    def __init__(self, model_weights: Dict[str, float] = None):
        self.name = "Optimized Consensus"

        # Learned weights from backtest analysis
        self.model_weights = model_weights or {
            'model4': 1.5,   # Naive odds - strong
            'model1': 1.2,   # Multi-output - decent
            'model3': 0.8,   # Poisson - tends to be conservative
            'model2': 0.6,   # Classification - worst performer
        }

        self.consensus_threshold = 2.5  # Weighted threshold

    def combine(
        self,
        predictions: Dict[str, Tuple[int, int]],
        **kwargs
    ) -> Tuple[Tuple[int, int], str, Dict[str, Any]]:
        """Combine with weighted consensus."""
        # Calculate weighted votes
        weighted_votes = {}

        for model_id, pred in predictions.items():
            weight = self.model_weights.get(model_id, 1.0)
            if pred not in weighted_votes:
                weighted_votes[pred] = {'weight': 0, 'models': []}
            weighted_votes[pred]['weight'] += weight
            weighted_votes[pred]['models'].append(model_id)

        # Find best prediction
        best_pred = max(weighted_votes.items(), key=lambda x: x[1]['weight'])
        best_scoreline = best_pred[0]
        best_weight = best_pred[1]['weight']

        if best_weight >= self.consensus_threshold:
            return (
                best_scoreline,
                'weighted_consensus',
                {'weight': best_weight, 'models': best_pred[1]['models']}
            )
        else:
            # Smart fallback
            if 'model4' in predictions:
                model4_pred = predictions['model4']

                # If model4 predicts draw but no other model agrees, reconsider
                if model4_pred == (1, 1):
                    non_draw_preds = [p for p in predictions.values() if p != (1, 1)]
                    if len(non_draw_preds) >= 3:
                        counter = Counter(non_draw_preds)
                        alt_pred = counter.most_common(1)[0][0]
                        return (alt_pred, 'draw_override', {'original': model4_pred})

                return (model4_pred, 'fallback', {'model': 'model4'})

            return (best_scoreline, 'best_available', {'weight': best_weight})


class HybridEnsemble:
    """
    Hybrid ensemble that combines multiple strategies:
    1. If 4 models agree -> use that (very high confidence)
    2. If 3 models agree -> use weighted consensus
    3. If 2+ models agree on tendency and odds confirm -> use it
    4. Otherwise -> smart model selection based on odds strength
    """

    def __init__(self):
        self.name = "Hybrid Ensemble"

    def combine(
        self,
        predictions: Dict[str, Tuple[int, int]],
        odds_home: Optional[float] = None,
        odds_draw: Optional[float] = None,
        odds_away: Optional[float] = None,
        **kwargs
    ) -> Tuple[Tuple[int, int], str, Dict[str, Any]]:
        """Hybrid combination strategy."""

        # Count exact prediction matches
        pred_counts = Counter(predictions.values())
        most_common, count = pred_counts.most_common(1)[0]

        # Strategy 1: Very high consensus (4 models agree)
        if count >= 4:
            return (most_common, 'unanimous', {'count': count})

        # Strategy 2: High consensus (3 models agree)
        if count >= 3:
            return (most_common, 'consensus', {'count': count})

        # Strategy 3: Check tendency consensus
        tendencies = {}
        for model_id, pred in predictions.items():
            if pred[0] > pred[1]:
                t = 'H'
            elif pred[0] < pred[1]:
                t = 'A'
            else:
                t = 'D'

            if t not in tendencies:
                tendencies[t] = []
            tendencies[t].append((model_id, pred))

        best_tendency = max(tendencies.items(), key=lambda x: len(x[1]))
        tendency = best_tendency[0]
        agreeing = best_tendency[1]

        # If 3+ agree on tendency, use smart selection
        if len(agreeing) >= 3:
            odds_support = self._check_odds_support(tendency, odds_home, odds_draw, odds_away)

            if odds_support:
                model4_pred = predictions.get('model4')
                if model4_pred:
                    model4_tendency = 'H' if model4_pred[0] > model4_pred[1] else \
                                     ('A' if model4_pred[0] < model4_pred[1] else 'D')
                    if model4_tendency == tendency:
                        return (model4_pred, 'tendency_odds_confirmed', {'tendency': tendency})

                preds_in_tendency = [p[1] for p in agreeing]
                if tendency == 'H':
                    best = max(preds_in_tendency, key=lambda x: x[0])
                elif tendency == 'A':
                    best = max(preds_in_tendency, key=lambda x: x[1])
                else:
                    avg = int(round(np.mean([p[0] for p in preds_in_tendency])))
                    best = (avg, avg)

                return (best, 'tendency_consensus', {'tendency': tendency, 'odds_support': True})

        # Strategy 4: Use odds-based selection
        if odds_home and odds_draw and odds_away:
            return self._odds_based_selection(predictions, odds_home, odds_draw, odds_away)

        # Fallback
        if 'model4' in predictions:
            return (predictions['model4'], 'fallback', {})

        return (most_common, 'majority', {'count': count})

    def _check_odds_support(
        self,
        tendency: str,
        odds_home: Optional[float],
        odds_draw: Optional[float],
        odds_away: Optional[float]
    ) -> bool:
        """Check if odds support the given tendency."""
        if not all([odds_home, odds_draw, odds_away]):
            return True

        try:
            if any(pd.isna([odds_home, odds_draw, odds_away])):
                return True
        except (TypeError, ValueError):
            return True

        min_odds = min(odds_home, odds_draw, odds_away)

        if tendency == 'H' and min_odds == odds_home:
            return True
        if tendency == 'A' and min_odds == odds_away:
            return True
        if tendency == 'D' and min_odds == odds_draw:
            return True

        # Allow if difference is small
        if tendency == 'H' and odds_home <= min_odds * 1.2:
            return True
        if tendency == 'A' and odds_away <= min_odds * 1.2:
            return True

        return False

    def _odds_based_selection(
        self,
        predictions: Dict[str, Tuple[int, int]],
        odds_home: float,
        odds_draw: float,
        odds_away: float
    ) -> Tuple[Tuple[int, int], str, Dict[str, Any]]:
        """Select prediction based on odds strength."""
        try:
            total = 1/odds_home + 1/odds_draw + 1/odds_away
            prob_home = (1/odds_home) / total
            prob_draw = (1/odds_draw) / total
            prob_away = (1/odds_away) / total
        except (TypeError, ValueError, ZeroDivisionError):
            if 'model4' in predictions:
                return (predictions['model4'], 'odds_error', {})
            return (list(predictions.values())[0], 'first', {})

        # Strong favorite detection
        if prob_home > 0.55:
            home_preds = [(m, p) for m, p in predictions.items() if p[0] > p[1]]
            if home_preds:
                best = max(home_preds, key=lambda x: x[1][0] - x[1][1])
                return (best[1], 'strong_home', {'prob': prob_home, 'model': best[0]})

        if prob_away > 0.55:
            away_preds = [(m, p) for m, p in predictions.items() if p[1] > p[0]]
            if away_preds:
                best = max(away_preds, key=lambda x: x[1][1] - x[1][0])
                return (best[1], 'strong_away', {'prob': prob_away, 'model': best[0]})

        if prob_draw > 0.33:
            draw_preds = [(m, p) for m, p in predictions.items() if p[0] == p[1]]
            if draw_preds:
                return (draw_preds[0][1], 'high_draw_prob', {'prob': prob_draw})

        # Default to model4
        if 'model4' in predictions:
            return (predictions['model4'], 'odds_fallback', {})

        return (list(predictions.values())[0], 'first', {})


class AdaptiveScorelineEnsemble:
    """
    Ensemble that adapts scoreline predictions based on context:
    - Uses more aggressive scorelines for strong favorites
    - Uses conservative scorelines for balanced matches
    """

    def __init__(self):
        self.name = "Adaptive Scoreline"

        self.scorelines = {
            'strong_home': [(3, 1), (3, 0), (2, 0), (4, 1)],
            'medium_home': [(2, 1), (2, 0), (1, 0)],
            'balanced': [(1, 1), (2, 2), (0, 0), (2, 1), (1, 2)],
            'medium_away': [(1, 2), (0, 2), (0, 1)],
            'strong_away': [(1, 3), (0, 3), (0, 2), (1, 4)],
        }

    def combine(
        self,
        predictions: Dict[str, Tuple[int, int]],
        odds_home: Optional[float] = None,
        odds_draw: Optional[float] = None,
        odds_away: Optional[float] = None,
        **kwargs
    ) -> Tuple[Tuple[int, int], str, Dict[str, Any]]:
        """Adaptive combination based on odds."""

        # Determine match category
        category = self._get_match_category(odds_home, odds_draw, odds_away)

        # Get consensus prediction
        pred_counts = Counter(predictions.values())
        most_common, count = pred_counts.most_common(1)[0]

        # If consensus aligns with category, use it
        if count >= 3:
            return (most_common, 'consensus', {'count': count, 'category': category})

        # Otherwise, select best scoreline for category
        valid_scorelines = self.scorelines.get(category, self.scorelines['balanced'])

        # Find prediction closest to valid scorelines
        best_pred = None
        best_dist = float('inf')

        for pred in predictions.values():
            for template in valid_scorelines:
                dist = abs(pred[0] - template[0]) + abs(pred[1] - template[1])
                if dist < best_dist:
                    best_dist = dist
                    best_pred = pred

        if best_pred is None:
            best_pred = valid_scorelines[0]

        if best_dist > 2:
            best_pred = valid_scorelines[0]

        return (best_pred, 'adaptive', {'category': category, 'distance': best_dist})

    def _get_match_category(
        self,
        odds_home: Optional[float],
        odds_draw: Optional[float],
        odds_away: Optional[float]
    ) -> str:
        """Categorize match based on odds."""
        if not all([odds_home, odds_draw, odds_away]):
            return 'balanced'

        try:
            if any(pd.isna([odds_home, odds_draw, odds_away])):
                return 'balanced'

            total = 1/odds_home + 1/odds_draw + 1/odds_away
            prob_home = (1/odds_home) / total
            prob_away = (1/odds_away) / total

            if prob_home > 0.55:
                return 'strong_home'
            elif prob_home > 0.42:
                return 'medium_home'
            elif prob_away > 0.55:
                return 'strong_away'
            elif prob_away > 0.42:
                return 'medium_away'
            else:
                return 'balanced'
        except (TypeError, ValueError, ZeroDivisionError):
            return 'balanced'


class BayesianOptimalEnsemble:
    """
    Uses Bayesian reasoning to select the optimal prediction.
    Maximizes expected Kicktipp points given all available information.
    """

    def __init__(self):
        self.name = "Bayesian Optimal"

        # Prior probabilities from historical Bundesliga data
        self.scoreline_priors = {
            (1, 0): 0.095, (0, 0): 0.082, (1, 1): 0.115, (2, 1): 0.118,
            (2, 0): 0.089, (0, 1): 0.072, (1, 2): 0.077, (2, 2): 0.052,
            (3, 1): 0.056, (3, 0): 0.042, (0, 2): 0.041, (1, 3): 0.032,
            (3, 2): 0.028, (0, 3): 0.022, (4, 1): 0.021, (2, 3): 0.019,
            (4, 0): 0.015, (3, 3): 0.012, (4, 2): 0.011, (5, 1): 0.008,
        }

    def combine(
        self,
        predictions: Dict[str, Tuple[int, int]],
        odds_home: Optional[float] = None,
        odds_draw: Optional[float] = None,
        odds_away: Optional[float] = None,
        xg_home: Optional[float] = None,
        xg_away: Optional[float] = None,
        **kwargs
    ) -> Tuple[Tuple[int, int], str, Dict[str, Any]]:
        """Find Bayesian optimal prediction."""

        # Get xG estimates
        if xg_home is None or xg_away is None:
            xg_home, xg_away = self._estimate_xg(odds_home, odds_draw, odds_away)

        # Calculate posterior probabilities for each scoreline
        posteriors = {}
        total_prob = 0

        for h in range(7):
            for a in range(7):
                # Likelihood from Poisson model
                likelihood = poisson.pmf(h, xg_home) * poisson.pmf(a, xg_away)

                # Prior from historical data
                prior = self.scoreline_priors.get((h, a), 0.005)

                # Model agreement bonus
                model_bonus = 1.0
                pred_tuple = (h, a)
                agreeing_models = sum(1 for p in predictions.values() if p == pred_tuple)
                model_bonus += agreeing_models * 0.3

                # Posterior (unnormalized)
                posterior = likelihood * prior * model_bonus
                posteriors[(h, a)] = posterior
                total_prob += posterior

        # Normalize
        for key in posteriors:
            posteriors[key] /= total_prob

        # Find prediction that maximizes expected Kicktipp points
        best_pred = (1, 1)
        best_expected = 0

        for pred_h in range(6):
            for pred_a in range(6):
                expected = 0

                for (actual_h, actual_a), prob in posteriors.items():
                    pts = self._kicktipp_points(pred_h, pred_a, actual_h, actual_a)
                    expected += prob * pts

                if expected > best_expected:
                    best_expected = expected
                    best_pred = (pred_h, pred_a)

        return (best_pred, 'bayesian_optimal', {'expected': best_expected})

    def _estimate_xg(
        self,
        odds_home: Optional[float],
        odds_draw: Optional[float],
        odds_away: Optional[float]
    ) -> Tuple[float, float]:
        """Estimate xG from odds."""
        if not all([odds_home, odds_draw, odds_away]):
            return 1.5, 1.3
        try:
            if any(pd.isna([odds_home, odds_draw, odds_away])):
                return 1.5, 1.3
            total = 1/odds_home + 1/odds_draw + 1/odds_away
            prob_home = (1/odds_home) / total
            prob_draw = (1/odds_draw) / total
            prob_away = (1/odds_away) / total
            xg_home = -np.log(prob_away + prob_draw * 0.5) * 1.2
            xg_away = -np.log(prob_home + prob_draw * 0.5) * 1.2
            return np.clip(xg_home, 0.5, 4.0), np.clip(xg_away, 0.5, 4.0)
        except (TypeError, ValueError, ZeroDivisionError):
            return 1.5, 1.3

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
        return 0


class AggressiveScorelineEnsemble:
    """
    More aggressive scoreline predictions for strong favorites.
    Uses 3-0, 3-1, 0-3 etc. when confidence is high.
    """

    def __init__(self):
        self.name = "Aggressive Scoreline"

    def combine(
        self,
        predictions: Dict[str, Tuple[int, int]],
        odds_home: Optional[float] = None,
        odds_draw: Optional[float] = None,
        odds_away: Optional[float] = None,
        xg_home: Optional[float] = None,
        xg_away: Optional[float] = None,
        **kwargs
    ) -> Tuple[Tuple[int, int], str, Dict[str, Any]]:
        """Aggressive predictions for strong favorites."""

        strength, favorite = self._get_favorite_strength(odds_home, odds_draw, odds_away)

        # Get tendency consensus
        tendencies = {}
        for model_id, pred in predictions.items():
            if pred[0] > pred[1]:
                t = 'H'
            elif pred[0] < pred[1]:
                t = 'A'
            else:
                t = 'D'
            if t not in tendencies:
                tendencies[t] = []
            tendencies[t].append(model_id)

        majority = max(tendencies.items(), key=lambda x: len(x[1]))
        tendency = majority[0]
        tendency_count = len(majority[1])

        if xg_home is None or xg_away is None:
            xg_home, xg_away = self._estimate_xg(odds_home, odds_draw, odds_away)

        # Aggressive logic
        if strength > 0.55 and tendency_count >= 3:
            if favorite == 'H' and tendency == 'H':
                home = max(3, int(round(xg_home)))
                away = min(1, int(round(xg_away)))
                return ((home, away), 'aggressive_home', {'strength': strength})

            elif favorite == 'A' and tendency == 'A':
                away = max(3, int(round(xg_away)))
                home = min(1, int(round(xg_home)))
                return ((home, away), 'aggressive_away', {'strength': strength})

        if strength > 0.45 and tendency_count >= 3:
            if favorite == 'H' and tendency == 'H':
                home = max(2, int(round(xg_home)))
                away = max(0, min(home - 1, int(round(xg_away))))
                return ((home, away), 'medium_home', {'strength': strength})

            elif favorite == 'A' and tendency == 'A':
                away = max(2, int(round(xg_away)))
                home = max(0, min(away - 1, int(round(xg_home))))
                return ((home, away), 'medium_away', {'strength': strength})

        # Standard consensus
        pred_counts = Counter(predictions.values())
        most_common, count = pred_counts.most_common(1)[0]

        if count >= 3:
            return (most_common, 'consensus', {'count': count})

        if 'model4' in predictions:
            return (predictions['model4'], 'fallback', {})

        return (most_common, 'majority', {})

    def _get_favorite_strength(
        self,
        odds_home: Optional[float],
        odds_draw: Optional[float],
        odds_away: Optional[float]
    ) -> Tuple[float, Optional[str]]:
        """Get favorite strength and identity."""
        if not all([odds_home, odds_draw, odds_away]):
            return 0.33, None
        try:
            if any(pd.isna([odds_home, odds_draw, odds_away])):
                return 0.33, None
            total = 1/odds_home + 1/odds_draw + 1/odds_away
            prob_home = (1/odds_home) / total
            prob_away = (1/odds_away) / total
            if prob_home > prob_away:
                return prob_home, 'H'
            else:
                return prob_away, 'A'
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.33, None

    def _estimate_xg(
        self,
        odds_home: Optional[float],
        odds_draw: Optional[float],
        odds_away: Optional[float]
    ) -> Tuple[float, float]:
        """Estimate xG."""
        if not all([odds_home, odds_draw, odds_away]):
            return 1.5, 1.3
        try:
            if any(pd.isna([odds_home, odds_draw, odds_away])):
                return 1.5, 1.3
            total = 1/odds_home + 1/odds_draw + 1/odds_away
            prob_home = (1/odds_home) / total
            prob_draw = (1/odds_draw) / total
            prob_away = (1/odds_away) / total
            xg_home = -np.log(prob_away + prob_draw * 0.5) * 1.2
            xg_away = -np.log(prob_home + prob_draw * 0.5) * 1.2
            return np.clip(xg_home, 0.5, 4.0), np.clip(xg_away, 0.5, 4.0)
        except (TypeError, ValueError, ZeroDivisionError):
            return 1.5, 1.3


class UltimateTendencyEnsemble:
    """
    Enhanced tendency consensus with smarter expert selection
    and dynamic scoreline adjustment.
    """

    def __init__(self):
        self.name = "Ultimate Tendency"

    def combine(
        self,
        predictions: Dict[str, Tuple[int, int]],
        odds_home: Optional[float] = None,
        odds_draw: Optional[float] = None,
        odds_away: Optional[float] = None,
        xg_home: Optional[float] = None,
        xg_away: Optional[float] = None,
        **kwargs
    ) -> Tuple[Tuple[int, int], str, Dict[str, Any]]:
        """Ultimate combination strategy."""

        # Get tendency for each prediction
        tendencies = {}
        for model_id, pred in predictions.items():
            if pred[0] > pred[1]:
                t = 'H'
            elif pred[0] < pred[1]:
                t = 'A'
            else:
                t = 'D'

            if t not in tendencies:
                tendencies[t] = []
            tendencies[t].append((model_id, pred))

        # Find majority tendency
        best_tendency = max(tendencies.items(), key=lambda x: len(x[1]))
        tendency = best_tendency[0]
        agreeing = best_tendency[1]

        # Check if odds support this tendency
        odds_tendency = self._get_odds_tendency(odds_home, odds_draw, odds_away)
        odds_aligned = (odds_tendency == tendency) or (odds_tendency is None)

        # Calculate expected goals
        if xg_home is None or xg_away is None:
            xg_home, xg_away = self._estimate_xg_from_odds(odds_home, odds_draw, odds_away)

        # Strategy 1: Strong consensus (3+ agree) + odds aligned
        if len(agreeing) >= 3 and odds_aligned:
            return self._smart_scoreline(tendency, xg_home, xg_away, predictions)

        # Strategy 2: Strong consensus but odds disagree - use model4
        if len(agreeing) >= 3 and not odds_aligned:
            if 'model4' in predictions:
                return (predictions['model4'], 'odds_override',
                       {'tendency': tendency, 'odds_tendency': odds_tendency})

        # Strategy 3: Weak consensus (2 agree) + strong odds signal
        if len(agreeing) >= 2:
            if odds_tendency and self._is_strong_favorite(odds_home, odds_draw, odds_away):
                return self._odds_based_scoreline(odds_tendency, xg_home, xg_away)

            return self._conservative_scoreline(tendency, xg_home, xg_away, predictions)

        # Fallback to model4
        if 'model4' in predictions:
            return (predictions['model4'], 'fallback', {})

        return (list(predictions.values())[0], 'first', {})

    def _get_odds_tendency(
        self,
        odds_home: Optional[float],
        odds_draw: Optional[float],
        odds_away: Optional[float]
    ) -> Optional[str]:
        """Get tendency from odds."""
        if not all([odds_home, odds_draw, odds_away]):
            return None

        try:
            if any(pd.isna([odds_home, odds_draw, odds_away])):
                return None

            min_odds = min(odds_home, odds_draw, odds_away)

            if min_odds == odds_home:
                return 'H'
            elif min_odds == odds_away:
                return 'A'
            else:
                return 'D'
        except (TypeError, ValueError):
            return None

    def _is_strong_favorite(
        self,
        odds_home: Optional[float],
        odds_draw: Optional[float],
        odds_away: Optional[float]
    ) -> bool:
        """Check if there's a strong favorite."""
        if not all([odds_home, odds_draw, odds_away]):
            return False

        try:
            total = 1/odds_home + 1/odds_draw + 1/odds_away
            prob_home = (1/odds_home) / total
            prob_away = (1/odds_away) / total

            return max(prob_home, prob_away) > 0.50
        except (TypeError, ValueError, ZeroDivisionError):
            return False

    def _estimate_xg_from_odds(
        self,
        odds_home: Optional[float],
        odds_draw: Optional[float],
        odds_away: Optional[float]
    ) -> Tuple[float, float]:
        """Estimate expected goals from odds."""
        if not all([odds_home, odds_draw, odds_away]):
            return 1.5, 1.3

        try:
            if any(pd.isna([odds_home, odds_draw, odds_away])):
                return 1.5, 1.3

            total = 1/odds_home + 1/odds_draw + 1/odds_away
            prob_home = (1/odds_home) / total
            prob_draw = (1/odds_draw) / total
            prob_away = (1/odds_away) / total

            xg_home = -np.log(prob_away + prob_draw * 0.5) * 1.2
            xg_away = -np.log(prob_home + prob_draw * 0.5) * 1.2

            return np.clip(xg_home, 0.5, 4.0), np.clip(xg_away, 0.5, 4.0)
        except (TypeError, ValueError, ZeroDivisionError):
            return 1.5, 1.3

    def _smart_scoreline(
        self,
        tendency: str,
        xg_home: float,
        xg_away: float,
        predictions: Dict[str, Tuple[int, int]]
    ) -> Tuple[Tuple[int, int], str, Dict[str, Any]]:
        """Generate smart scoreline based on tendency and xG."""
        if tendency == 'H':
            home = max(1, int(round(xg_home)))
            away = min(home - 1, int(round(xg_away)))
            away = max(0, away)
            return ((home, away), 'smart_home', {'xg': (xg_home, xg_away)})

        elif tendency == 'A':
            away = max(1, int(round(xg_away)))
            home = min(away - 1, int(round(xg_home)))
            home = max(0, home)
            return ((home, away), 'smart_away', {'xg': (xg_home, xg_away)})

        else:
            avg = (xg_home + xg_away) / 2
            score = int(round(avg))
            return ((score, score), 'smart_draw', {'xg': (xg_home, xg_away)})

    def _conservative_scoreline(
        self,
        tendency: str,
        xg_home: float,
        xg_away: float,
        predictions: Dict[str, Tuple[int, int]]
    ) -> Tuple[Tuple[int, int], str, Dict[str, Any]]:
        """Conservative scoreline for uncertain situations."""
        if tendency == 'H':
            return ((2, 1), 'conservative_home', {})
        elif tendency == 'A':
            return ((1, 2), 'conservative_away', {})
        else:
            return ((1, 1), 'conservative_draw', {})

    def _odds_based_scoreline(
        self,
        tendency: str,
        xg_home: float,
        xg_away: float
    ) -> Tuple[Tuple[int, int], str, Dict[str, Any]]:
        """Odds-based scoreline for strong favorites."""
        if tendency == 'H':
            home = max(2, int(round(xg_home)))
            away = max(0, int(round(xg_away)) - 1)
            return ((home, away), 'odds_home', {})
        elif tendency == 'A':
            away = max(2, int(round(xg_away)))
            home = max(0, int(round(xg_home)) - 1)
            return ((home, away), 'odds_away', {})
        else:
            return ((1, 1), 'odds_draw', {})


class SuperConsensusEnsemble:
    """
    Super consensus that requires 3+ models AND odds alignment.
    Falls back to weighted model selection otherwise.
    """

    def __init__(self):
        self.name = "Super Consensus"

        self.model_weights = {
            'model4': 1.8,   # Naive odds - very strong
            'model1': 1.2,   # Multi-output
            'model3': 0.9,   # Poisson
            'model2': 0.6,   # Classification - weak
        }

    def combine(
        self,
        predictions: Dict[str, Tuple[int, int]],
        odds_home: Optional[float] = None,
        odds_draw: Optional[float] = None,
        odds_away: Optional[float] = None,
        **kwargs
    ) -> Tuple[Tuple[int, int], str, Dict[str, Any]]:
        """Combine with super consensus."""

        # Check for exact consensus
        pred_counts = Counter(predictions.values())
        most_common, count = pred_counts.most_common(1)[0]

        # Get odds tendency
        odds_tendency = self._get_odds_tendency(odds_home, odds_draw, odds_away)
        pred_tendency = 'H' if most_common[0] > most_common[1] else \
                       ('A' if most_common[0] < most_common[1] else 'D')

        # Super consensus: 3+ agree AND matches odds
        if count >= 3 and (odds_tendency == pred_tendency or odds_tendency is None):
            return (most_common, 'super_consensus', {'count': count, 'odds_aligned': True})

        # Strong consensus (4): use regardless of odds
        if count == 4:
            return (most_common, 'unanimous', {'count': count})

        # Tendency consensus
        tendencies = {}
        for model_id, pred in predictions.items():
            if pred[0] > pred[1]:
                t = 'H'
            elif pred[0] < pred[1]:
                t = 'A'
            else:
                t = 'D'

            if t not in tendencies:
                tendencies[t] = []
            tendencies[t].append((model_id, pred))

        best_tendency = max(tendencies.items(), key=lambda x: len(x[1]))
        tendency = best_tendency[0]
        agreeing = best_tendency[1]

        # 3+ agree on tendency AND matches odds
        if len(agreeing) >= 3 and (odds_tendency == tendency or odds_tendency is None):
            best_pred = None
            best_weight = 0

            for model_id, pred in agreeing:
                weight = self.model_weights.get(model_id, 1.0)
                if weight > best_weight:
                    best_weight = weight
                    best_pred = pred

            return (best_pred, 'tendency_consensus', {'tendency': tendency, 'weight': best_weight})

        # Weighted fallback
        weighted_votes = {}
        for model_id, pred in predictions.items():
            weight = self.model_weights.get(model_id, 1.0)
            if pred not in weighted_votes:
                weighted_votes[pred] = 0
            weighted_votes[pred] += weight

        best = max(weighted_votes.items(), key=lambda x: x[1])

        return (best[0], 'weighted_vote', {'weight': best[1]})

    def _get_odds_tendency(
        self,
        odds_home: Optional[float],
        odds_draw: Optional[float],
        odds_away: Optional[float]
    ) -> Optional[str]:
        """Get tendency from odds."""
        if not all([odds_home, odds_draw, odds_away]):
            return None

        try:
            if any(pd.isna([odds_home, odds_draw, odds_away])):
                return None

            min_odds = min(odds_home, odds_draw, odds_away)

            if min_odds == odds_home:
                return 'H'
            elif min_odds == odds_away:
                return 'A'
            else:
                return 'D'
        except (TypeError, ValueError):
            return None


class MaxPointsEnsemble:
    """
    Ensemble specifically designed to maximize Kicktipp points.

    Key insights from analysis:
    1. Model4 (Naive Odds) is surprisingly strong - trust it as baseline
    2. When 3+ models agree on EXACT score, that's very valuable (higher chance of 4 pts)
    3. When tendency consensus differs from odds, be careful
    4. For strong favorites (prob > 0.55), use more aggressive scorelines
    5. Draw prediction is risky - only predict draws when very confident
    """

    def __init__(self):
        self.name = "Max Points"

        # Common scorelines by tendency (based on Bundesliga data)
        self.home_scorelines = [(2, 1), (2, 0), (1, 0), (3, 1), (3, 0)]
        self.away_scorelines = [(1, 2), (0, 2), (0, 1), (1, 3), (0, 3)]
        self.draw_scorelines = [(1, 1), (2, 2), (0, 0)]

    def combine(
        self,
        predictions: Dict[str, Tuple[int, int]],
        odds_home: Optional[float] = None,
        odds_draw: Optional[float] = None,
        odds_away: Optional[float] = None,
        xg_home: Optional[float] = None,
        xg_away: Optional[float] = None,
        **kwargs
    ) -> Tuple[Tuple[int, int], str, Dict[str, Any]]:
        """Maximize expected Kicktipp points."""

        # Count exact prediction matches
        pred_counts = Counter(predictions.values())
        most_common, count = pred_counts.most_common(1)[0]

        # STRATEGY 1: If 4 models agree on exact score -> HIGH VALUE, use it
        if count == 4:
            return (most_common, 'unanimous', {'count': count, 'expected_pts': 'high'})

        # STRATEGY 2: If 3+ models agree on exact score AND it's a common scoreline -> use it
        if count >= 3:
            common_scores = [(2, 1), (1, 1), (1, 0), (2, 0), (1, 2), (0, 1), (0, 0)]
            if most_common in common_scores:
                return (most_common, 'strong_consensus', {'count': count})

        # Get tendency consensus
        tendencies = {}
        for model_id, pred in predictions.items():
            if pred[0] > pred[1]:
                t = 'H'
            elif pred[0] < pred[1]:
                t = 'A'
            else:
                t = 'D'

            if t not in tendencies:
                tendencies[t] = []
            tendencies[t].append((model_id, pred))

        best_tendency = max(tendencies.items(), key=lambda x: len(x[1]))
        tendency = best_tendency[0]
        tendency_count = len(best_tendency[1])

        # Get odds-implied probabilities and tendency
        probs = self._get_probabilities(odds_home, odds_draw, odds_away)
        odds_tendency = self._get_odds_tendency(odds_home, odds_draw, odds_away)

        # STRATEGY 3: Strong tendency consensus (3+) aligned with odds
        if tendency_count >= 3 and tendency == odds_tendency:
            return self._select_optimal_scoreline(tendency, probs, predictions)

        # STRATEGY 4: Strong favorite (prob > 0.55) - trust the odds
        if probs:
            if probs['H'] > 0.55:
                return self._select_optimal_scoreline('H', probs, predictions)
            elif probs['A'] > 0.55:
                return self._select_optimal_scoreline('A', probs, predictions)

        # STRATEGY 5: Models agree on tendency (3+) but odds disagree
        # This is risky - use model4 as it incorporates odds
        if tendency_count >= 3 and odds_tendency and tendency != odds_tendency:
            if 'model4' in predictions:
                return (predictions['model4'], 'odds_priority', {'tendency': tendency, 'odds_tendency': odds_tendency})

        # STRATEGY 6: Weak signals - fall back to model4 (most reliable single model)
        if 'model4' in predictions:
            return (predictions['model4'], 'fallback', {})

        # Last resort
        return (most_common, 'majority', {'count': count})

    def _get_probabilities(
        self,
        odds_home: Optional[float],
        odds_draw: Optional[float],
        odds_away: Optional[float]
    ) -> Optional[Dict[str, float]]:
        """Get implied probabilities from odds."""
        if not all([odds_home, odds_draw, odds_away]):
            return None

        try:
            if any(pd.isna([odds_home, odds_draw, odds_away])):
                return None

            total = 1/odds_home + 1/odds_draw + 1/odds_away
            return {
                'H': (1/odds_home) / total,
                'D': (1/odds_draw) / total,
                'A': (1/odds_away) / total,
            }
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    def _get_odds_tendency(
        self,
        odds_home: Optional[float],
        odds_draw: Optional[float],
        odds_away: Optional[float]
    ) -> Optional[str]:
        """Get tendency from odds."""
        if not all([odds_home, odds_draw, odds_away]):
            return None

        try:
            if any(pd.isna([odds_home, odds_draw, odds_away])):
                return None

            min_odds = min(odds_home, odds_draw, odds_away)

            if min_odds == odds_home:
                return 'H'
            elif min_odds == odds_away:
                return 'A'
            else:
                return 'D'
        except (TypeError, ValueError):
            return None

    def _select_optimal_scoreline(
        self,
        tendency: str,
        probs: Optional[Dict[str, float]],
        predictions: Dict[str, Tuple[int, int]]
    ) -> Tuple[Tuple[int, int], str, Dict[str, Any]]:
        """Select the optimal scoreline for a given tendency."""

        # Get predictions that match the tendency
        matching_preds = []
        for model_id, pred in predictions.items():
            if tendency == 'H' and pred[0] > pred[1]:
                matching_preds.append(pred)
            elif tendency == 'A' and pred[0] < pred[1]:
                matching_preds.append(pred)
            elif tendency == 'D' and pred[0] == pred[1]:
                matching_preds.append(pred)

        # If multiple models agree on a specific score, use that
        if matching_preds:
            pred_counts = Counter(matching_preds)
            best_match, match_count = pred_counts.most_common(1)[0]
            if match_count >= 2:
                return (best_match, 'tendency_match', {'count': match_count})

        # Otherwise, use common scorelines
        if tendency == 'H':
            # For strong favorites, use 2-0 or 3-0; otherwise 2-1
            if probs and probs['H'] > 0.60:
                return ((2, 0), 'strong_home', {'prob': probs['H']})
            return ((2, 1), 'home_win', {})
        elif tendency == 'A':
            if probs and probs['A'] > 0.60:
                return ((0, 2), 'strong_away', {'prob': probs['A']})
            return ((1, 2), 'away_win', {})
        else:
            # Draw - 1-1 is most common
            return ((1, 1), 'draw', {})
