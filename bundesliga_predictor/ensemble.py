"""
Ensemble strategies for combining multiple model predictions.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter


class BaseEnsemble(ABC):
    """
    Abstract base class for ensemble strategies.
    """

    def __init__(self, name: str):
        """
        Initialize ensemble.

        Args:
            name: Human-readable name for the ensemble strategy
        """
        self.name = name

    @abstractmethod
    def combine(
        self,
        predictions: Dict[str, Tuple[int, int]],
        **kwargs
    ) -> Tuple[Tuple[int, int], str, Any]:
        """
        Combine predictions from multiple models.

        Args:
            predictions: Dict mapping model_id to (home_score, away_score)
            **kwargs: Additional context (e.g., model performance history)

        Returns:
            Tuple of:
            - (home_score, away_score): Final prediction
            - strategy_used: Description of how prediction was made
            - details: Additional details about the decision
        """
        pass


class ConsensusEnsemble(BaseEnsemble):
    """
    Consensus-based ensemble strategy.

    If 3+ models agree on a prediction, use that prediction.
    Otherwise, fall back to the best individual model (Naive Odds).

    This strategy outperformed all individual models in backtesting:
    - Ensemble: 1.52 pts/match
    - Model4 (Naive Odds): 1.49 pts/match
    """

    def __init__(self, consensus_threshold: int = 3, fallback_model: str = 'model4'):
        """
        Initialize consensus ensemble.

        Args:
            consensus_threshold: Minimum models that must agree (default: 3)
            fallback_model: Model to use when no consensus (default: model4)
        """
        super().__init__(name="Consensus Ensemble")
        self.consensus_threshold = consensus_threshold
        self.fallback_model = fallback_model

    def combine(
        self,
        predictions: Dict[str, Tuple[int, int]],
        **kwargs
    ) -> Tuple[Tuple[int, int], str, Any]:
        """
        Combine predictions using consensus voting.

        Args:
            predictions: Dict mapping model_id to (home_score, away_score)

        Returns:
            Tuple of (prediction, strategy_used, details)
        """
        # Count occurrences of each prediction
        pred_list = list(predictions.values())
        counts = Counter(pred_list)
        most_common, count = counts.most_common(1)[0]

        if count >= self.consensus_threshold:
            # Strong consensus found
            agreeing_models = [
                model_id for model_id, pred in predictions.items()
                if pred == most_common
            ]
            return (
                most_common,
                'consensus',
                {
                    'count': count,
                    'agreeing_models': agreeing_models,
                    'threshold': self.consensus_threshold,
                }
            )
        else:
            # No consensus - use fallback model
            if self.fallback_model in predictions:
                fallback_pred = predictions[self.fallback_model]
            else:
                # If fallback not available, use most common
                fallback_pred = most_common

            return (
                fallback_pred,
                'fallback',
                {
                    'fallback_model': self.fallback_model,
                    'max_agreement': count,
                    'threshold': self.consensus_threshold,
                }
            )

    def get_prediction_summary(
        self,
        predictions: Dict[str, Tuple[int, int]],
        model_names: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Get a detailed summary of the ensemble decision.

        Args:
            predictions: Model predictions
            model_names: Optional mapping of model_id to display name

        Returns:
            Dictionary with full decision details
        """
        pred, strategy, details = self.combine(predictions)

        if model_names is None:
            model_names = {k: k for k in predictions.keys()}

        summary = {
            'final_prediction': f"{pred[0]}-{pred[1]}",
            'home_score': pred[0],
            'away_score': pred[1],
            'strategy': strategy,
            'individual_predictions': {
                model_names.get(k, k): f"{v[0]}-{v[1]}"
                for k, v in predictions.items()
            },
        }

        if strategy == 'consensus':
            summary['consensus_count'] = details['count']
            summary['agreeing_models'] = [
                model_names.get(m, m) for m in details['agreeing_models']
            ]
            summary['reason'] = f"{details['count']} models agree"
        else:
            summary['fallback_used'] = model_names.get(details['fallback_model'], details['fallback_model'])
            summary['reason'] = f"No consensus (max {details['max_agreement']} agree), using {summary['fallback_used']}"

        return summary


class MajorityVoteEnsemble(BaseEnsemble):
    """
    Simple majority vote ensemble.

    Uses the most common prediction, regardless of agreement count.
    Breaks ties using model priority.
    """

    def __init__(self, priority_order: Optional[List[str]] = None):
        """
        Initialize majority vote ensemble.

        Args:
            priority_order: List of model_ids in priority order for tie-breaking
        """
        super().__init__(name="Majority Vote Ensemble")
        self.priority_order = priority_order or ['model4', 'model1', 'model3', 'model2']

    def combine(
        self,
        predictions: Dict[str, Tuple[int, int]],
        **kwargs
    ) -> Tuple[Tuple[int, int], str, Any]:
        """
        Combine predictions using majority vote.

        Args:
            predictions: Dict mapping model_id to (home_score, away_score)

        Returns:
            Tuple of (prediction, strategy_used, details)
        """
        pred_list = list(predictions.values())
        counts = Counter(pred_list)
        most_common = counts.most_common()

        # Check for ties
        top_count = most_common[0][1]
        tied_preds = [p for p, c in most_common if c == top_count]

        if len(tied_preds) == 1:
            # Clear winner
            return (
                tied_preds[0],
                'majority',
                {'count': top_count, 'tied': False}
            )
        else:
            # Tie - use priority order
            for model_id in self.priority_order:
                if model_id in predictions and predictions[model_id] in tied_preds:
                    return (
                        predictions[model_id],
                        'tiebreak',
                        {'count': top_count, 'tied': True, 'tiebreaker': model_id}
                    )

            # Fallback to first tied prediction
            return (
                tied_preds[0],
                'first_tied',
                {'count': top_count, 'tied': True}
            )


class WeightedEnsemble(BaseEnsemble):
    """
    Weighted ensemble that uses model performance history.

    Models with better historical performance get more weight.
    """

    def __init__(self, default_weights: Optional[Dict[str, float]] = None):
        """
        Initialize weighted ensemble.

        Args:
            default_weights: Default weights for each model
        """
        super().__init__(name="Weighted Ensemble")
        self.default_weights = default_weights or {
            'model1': 1.0,
            'model2': 1.0,
            'model3': 1.0,
            'model4': 1.0,
        }

    def combine(
        self,
        predictions: Dict[str, Tuple[int, int]],
        weights: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> Tuple[Tuple[int, int], str, Any]:
        """
        Combine predictions using weighted voting.

        Args:
            predictions: Dict mapping model_id to (home_score, away_score)
            weights: Optional performance-based weights

        Returns:
            Tuple of (prediction, strategy_used, details)
        """
        if weights is None:
            weights = self.default_weights

        # Calculate weighted votes
        weighted_votes: Dict[Tuple[int, int], float] = {}
        for model_id, pred in predictions.items():
            weight = weights.get(model_id, 1.0)
            weighted_votes[pred] = weighted_votes.get(pred, 0) + weight

        # Find prediction with highest weighted vote
        best_pred = max(weighted_votes.items(), key=lambda x: x[1])

        return (
            best_pred[0],
            'weighted',
            {
                'weighted_votes': {f"{k[0]}-{k[1]}": v for k, v in weighted_votes.items()},
                'winning_weight': best_pred[1],
            }
        )
