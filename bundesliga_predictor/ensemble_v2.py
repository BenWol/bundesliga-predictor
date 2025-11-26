"""
Enhanced ensemble strategies v2.

These strategies have been validated through rolling window backtesting
and consistently outperform the original consensus ensemble.

Key findings:
- Tendency consensus (agreeing on win/draw/loss) is more robust than exact score consensus
- Using expert models for each tendency improves accuracy
- Combining model consensus with odds alignment reduces errors

Usage:
    from bundesliga_predictor.ensemble_v2 import SimpleTendencyEnsemble
    ensemble = SimpleTendencyEnsemble()
    prediction, strategy, details = ensemble.combine(model_predictions)
"""

import pandas as pd
from collections import Counter
from typing import Dict, Tuple, Any, Optional


class TendencyExpertEnsemble:
    """
    Uses specific expert models for each tendency (home/draw/away).

    This ensemble first determines the likely match tendency through
    model consensus and odds analysis, then uses the historically
    best-performing model for that tendency type.

    Performance: +0.12 pts/match over original in rolling validation.
    """

    def __init__(self):
        self.name = "Tendency Expert v2"

        # Expert assignments based on backtest analysis:
        # - model4 (naive odds) is best for predicting wins
        # - model3 (Poisson) is best for predicting draws
        self.experts = {
            'H': ['model4', 'model1'],  # Home win experts (priority order)
            'D': ['model3', 'model2'],  # Draw experts
            'A': ['model4', 'model1'],  # Away win experts
        }

    def combine(
        self,
        predictions: Dict[str, Tuple[int, int]],
        odds_home: Optional[float] = None,
        odds_draw: Optional[float] = None,
        odds_away: Optional[float] = None,
        is_derby: bool = False,
        **kwargs
    ) -> Tuple[Tuple[int, int], str, Dict[str, Any]]:
        """
        Combine model predictions using tendency expert strategy.

        Args:
            predictions: Dict mapping model_id to (home_score, away_score)
            odds_home: Home win odds
            odds_draw: Draw odds
            odds_away: Away win odds
            is_derby: Whether this is a derby match (affects predictions)

        Returns:
            Tuple of (prediction, strategy_name, details)
        """
        # Step 1: Calculate tendency for each model
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

        # Step 2: Get majority tendency from models
        majority_tendency = max(tendencies.items(), key=lambda x: len(x[1]))
        tendency = majority_tendency[0]
        tendency_count = len(majority_tendency[1])

        # Step 3: Check odds tendency (what do bookmakers think?)
        odds_tendency = self._get_odds_tendency(odds_home, odds_draw, odds_away)

        # Step 4: Resolve final tendency
        # If models strongly agree (3+), trust them
        # Otherwise, let odds influence the decision
        final_tendency = tendency
        if odds_tendency and tendency_count < 3:
            # Models weakly agree - consider odds
            if odds_tendency != tendency:
                # Odds disagree - be conservative, use odds
                final_tendency = odds_tendency

        # Step 5: Get expert prediction for final tendency
        experts = self.experts.get(final_tendency, ['model4'])

        for expert in experts:
            if expert in predictions:
                pred = predictions[expert]
                # Verify expert's prediction matches the tendency
                pred_tendency = 'H' if pred[0] > pred[1] else ('A' if pred[0] < pred[1] else 'D')

                if pred_tendency == final_tendency:
                    # Apply derby adjustment if applicable
                    final_pred = pred
                    if is_derby and odds_home and odds_away:
                        try:
                            max_odds = max(odds_home, odds_away)
                            min_odds = min(odds_home, odds_away)
                            # If competitive derby and predicting 1-goal win, lean to draw
                            if max_odds / min_odds < 1.5:
                                diff = abs(pred[0] - pred[1])
                                if diff == 1:  # 1-goal win prediction
                                    final_pred = (1, 1)
                        except (TypeError, ValueError, ZeroDivisionError):
                            pass

                    return (
                        final_pred,
                        'tendency_expert',
                        {
                            'tendency': final_tendency,
                            'expert': expert,
                            'model_agreement': tendency_count,
                            'odds_tendency': odds_tendency,
                            'is_derby': is_derby,
                            'derby_adjusted': final_pred != pred,
                        }
                    )

        # Step 6: Expert disagreed with tendency - generate appropriate scoreline
        base_pred = None
        if final_tendency == 'H':
            base_pred = (2, 1)
        elif final_tendency == 'A':
            base_pred = (1, 2)
        else:
            base_pred = (1, 1)

        # Step 7: Derby adjustment - derbies are often tight/draws when odds are close
        final_pred = base_pred
        if is_derby and odds_home and odds_away:
            try:
                # If odds are close (competitive match), lean toward 1-1 draw
                max_odds = max(odds_home, odds_away)
                min_odds = min(odds_home, odds_away)
                if max_odds / min_odds < 1.5:  # Very competitive derby
                    # If predicting a 1-goal win, consider draw instead
                    if base_pred in [(2, 1), (1, 2), (1, 0), (0, 1)]:
                        final_pred = (1, 1)
            except (TypeError, ValueError, ZeroDivisionError):
                pass

        return (final_pred, 'tendency_fallback', {'tendency': final_tendency, 'reason': 'expert_mismatch', 'is_derby': is_derby})

    def _get_odds_tendency(
        self,
        odds_home: Optional[float],
        odds_draw: Optional[float],
        odds_away: Optional[float]
    ) -> Optional[str]:
        """Determine tendency from betting odds."""
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


class TendencyConsensusEnsemble:
    """
    Two-stage consensus: first agree on tendency (H/D/A),
    then use best model's scoreline for that tendency.

    This is simpler than TendencyExpert but still effective.
    Performance: +0.04-0.07 pts/match over original.
    """

    def __init__(self):
        self.name = "Tendency Consensus v2"

        # Which model to trust for each tendency
        self.tendency_experts = {
            'H': 'model4',  # Home wins - trust odds
            'D': 'model3',  # Draws - Poisson is conservative
            'A': 'model4',  # Away wins - trust odds
        }

    def combine(
        self,
        predictions: Dict[str, Tuple[int, int]],
        odds_home: Optional[float] = None,
        odds_draw: Optional[float] = None,
        odds_away: Optional[float] = None,
        **kwargs
    ) -> Tuple[Tuple[int, int], str, Dict[str, Any]]:
        """Combine by first agreeing on tendency."""
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

        # Find consensus tendency
        best_tendency = max(tendencies.items(), key=lambda x: len(x[1]))
        tendency = best_tendency[0]
        agreeing_models = best_tendency[1]

        if len(agreeing_models) >= 3:
            # Strong tendency consensus - use expert
            expert = self.tendency_experts.get(tendency, 'model4')
            if expert in predictions:
                return (
                    predictions[expert],
                    'tendency_consensus',
                    {
                        'tendency': tendency,
                        'agreeing': len(agreeing_models),
                        'expert': expert
                    }
                )

        # Fallback to best model in majority tendency
        if tendency == 'H':
            home_preds = [p[1] for p in agreeing_models]
            if home_preds:
                best = max(home_preds, key=lambda x: x[0] - x[1])
                return (best, 'tendency_best', {'tendency': tendency})

        elif tendency == 'A':
            away_preds = [p[1] for p in agreeing_models]
            if away_preds:
                best = max(away_preds, key=lambda x: x[1] - x[0])
                return (best, 'tendency_best', {'tendency': tendency})

        # Default
        if 'model4' in predictions:
            return (predictions['model4'], 'fallback', {})

        return (list(predictions.values())[0], 'first', {})


class HybridEnsembleV2:
    """
    Hybrid ensemble that adapts strategy based on context:

    1. Strong consensus (4 models agree) -> use it
    2. Good consensus (3 models) + odds aligned -> use it
    3. Tendency consensus + odds aligned -> use expert
    4. Otherwise -> use model4 (naive odds)

    This ensemble balances aggressiveness with safety.
    """

    def __init__(self):
        self.name = "Hybrid v2"

        self.tendency_experts = {
            'H': 'model4',
            'D': 'model3',
            'A': 'model4',
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
        """Hybrid combination strategy."""

        # Check for exact prediction consensus
        pred_counts = Counter(predictions.values())
        most_common, count = pred_counts.most_common(1)[0]

        # Strategy 1: Unanimous (4 models agree)
        if count >= 4:
            return (most_common, 'unanimous', {'count': count})

        # Strategy 2: Strong consensus (3 models) - check odds alignment
        if count >= 3:
            pred_tendency = 'H' if most_common[0] > most_common[1] else \
                           ('A' if most_common[0] < most_common[1] else 'D')
            odds_tendency = self._get_odds_tendency(odds_home, odds_draw, odds_away)

            if odds_tendency == pred_tendency or odds_tendency is None:
                return (most_common, 'consensus_aligned', {'count': count, 'odds_aligned': True})

        # Strategy 3: Tendency consensus
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

        best_tendency = max(tendencies.items(), key=lambda x: len(x[1]))
        tendency = best_tendency[0]
        tendency_count = len(best_tendency[1])

        if tendency_count >= 3:
            odds_tendency = self._get_odds_tendency(odds_home, odds_draw, odds_away)

            if odds_tendency == tendency or odds_tendency is None:
                expert = self.tendency_experts.get(tendency, 'model4')
                if expert in predictions:
                    return (
                        predictions[expert],
                        'tendency_expert',
                        {'tendency': tendency, 'expert': expert}
                    )

        # Strategy 4: Fallback to model4
        if 'model4' in predictions:
            return (predictions['model4'], 'fallback', {})

        return (most_common, 'majority', {'count': count})

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
            return 'D'
        except (TypeError, ValueError):
            return None


class SimpleTendencyEnsemble:
    """
    Simple majority tendency vote with model4 fallback.

    If 3+ models agree on tendency (H/D/A), output fixed scoreline:
    - Home win -> 2:1
    - Away win -> 1:2
    - Draw -> 1:1

    Otherwise, fall back to model4 (naive odds) prediction.

    Performance: 1.57 pts/match on current season (best performer).
    """

    def __init__(self):
        self.name = "Simple Tendency"

    def combine(
        self,
        predictions: Dict[str, Tuple[int, int]],
        **kwargs
    ) -> Tuple[Tuple[int, int], str, Dict[str, Any]]:
        """
        Combine model predictions using simple majority tendency.

        Args:
            predictions: Dict mapping model_id to (home_score, away_score)
            **kwargs: Ignored (for API compatibility)

        Returns:
            Tuple of (prediction, strategy_name, details)
        """
        tendencies = []
        for pred in predictions.values():
            if pred[0] > pred[1]:
                tendencies.append('H')
            elif pred[0] < pred[1]:
                tendencies.append('A')
            else:
                tendencies.append('D')

        counts = Counter(tendencies)
        majority, count = counts.most_common(1)[0]

        if count >= 3:
            if majority == 'H':
                return ((2, 1), 'tendency_consensus', {'tendency': 'H', 'count': count})
            elif majority == 'A':
                return ((1, 2), 'tendency_consensus', {'tendency': 'A', 'count': count})
            else:
                return ((1, 1), 'tendency_consensus', {'tendency': 'D', 'count': count})

        # No strong consensus - use model4 (naive odds)
        if 'model4' in predictions:
            return (predictions['model4'], 'model4_fallback', {'tendency': majority, 'count': count})

        return ((1, 1), 'default', {'tendency': majority, 'count': count})


# Default recommended ensemble
RecommendedEnsemble = SimpleTendencyEnsemble
