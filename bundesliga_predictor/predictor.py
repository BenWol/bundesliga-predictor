"""
Main predictor class that orchestrates all components.
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    MODEL_NAMES, TRAIN_TEST_SPLIT_RATIO, MIN_TRAINING_SAMPLES,
    RECOMMENDATIONS_FILE
)
from .data import DataLoader, FixturesFetcher
from .features import FeatureExtractor
from .scoring import calculate_kicktipp_points
from .ensemble import ConsensusEnsemble
from .models import (
    BaseModel,
    MultiOutputRegressionModel,
    MultiClassClassificationModel,
    PoissonRegressionModel,
    NaiveOddsModel,
)


class BundesligaPredictor:
    """
    Main predictor class for Bundesliga match predictions.

    This is the entry point for making predictions. It:
    1. Loads and processes match data
    2. Trains all models
    3. Makes predictions for upcoming matches
    4. Combines predictions using ensemble strategy

    Usage:
        predictor = BundesligaPredictor()
        predictions = predictor.predict_next_matchday()
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the predictor.

        Args:
            verbose: Whether to print progress messages
        """
        self.verbose = verbose
        self._data_loader = DataLoader()
        self._fixtures_fetcher = FixturesFetcher()
        self._ensemble = ConsensusEnsemble()

        # Initialize models
        self._models: Dict[str, BaseModel] = {
            'model1': MultiOutputRegressionModel(),
            'model2': MultiClassClassificationModel(),
            'model3': PoissonRegressionModel(),
            'model4': NaiveOddsModel(),
        }

        self._df: Optional[pd.DataFrame] = None
        self._feature_extractor: Optional[FeatureExtractor] = None
        self._is_trained = False
        self._training_metrics: Dict[str, Any] = {}

    def _log(self, message: str):
        """Print message if verbose mode is on."""
        if self.verbose:
            print(message)

    def load_data(self) -> pd.DataFrame:
        """
        Load match data from cache.

        Returns:
            DataFrame with match data
        """
        self._log("\n1. Loading match data...")
        self._df = self._data_loader.load_cached_matches()
        self._feature_extractor = FeatureExtractor(self._df)

        self._log(f"   Date range: {self._df['date'].min().date()} to {self._df['date'].max().date()}")
        return self._df

    def train_models(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, float]]:
        """
        Train all models on the data.

        Args:
            df: Optional DataFrame to use (uses cached data if not provided)

        Returns:
            Dictionary of training metrics per model
        """
        if df is None:
            if self._df is None:
                self.load_data()
            df = self._df

        self._log("\n2. Creating features...")
        X, y_home, y_away, scorelines = self._feature_extractor.extract_training_features(df)
        self._log(f"   Created features for {len(X)} matches")
        self._log(f"   Unique scorelines: {len(np.unique(scorelines))}")

        # Temporal split
        split_idx = int(len(X) * TRAIN_TEST_SPLIT_RATIO)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_home_train, y_home_test = y_home[:split_idx], y_home[split_idx:]
        y_away_train, y_away_test = y_away[:split_idx], y_away[split_idx:]
        scorelines_train, scorelines_test = scorelines[:split_idx], scorelines[split_idx:]

        self._log(f"   Training on {len(X_train)} matches, testing on {len(X_test)}")

        self._log("\n3. Training models...")
        self._log("-" * 70)

        metrics = {}

        for model_id, model in self._models.items():
            if model_id == 'model4':
                # Naive model doesn't need training
                model.train(X_train, y_home_train, y_away_train)
                eval_metrics = model.evaluate(X_test, y_home_test, y_away_test)
            else:
                # Train ML models
                model.train(X_train, y_home_train, y_away_train, scorelines_train)
                eval_metrics = model.evaluate(X_test, y_home_test, y_away_test)

            metrics[model_id] = eval_metrics

            self._log(f"\n[{model_id[-1]}] {model.name}:")
            self._log(f"    Exact Score Accuracy: {eval_metrics['exact_accuracy']:.1%}")
            self._log(f"    Kicktipp Score: {eval_metrics['total_points']}/{eval_metrics['max_possible']} "
                     f"({eval_metrics['kicktipp_percentage']:.1f}%)")
            self._log(f"    Avg Points/Match: {eval_metrics['avg_points_per_match']:.2f}")

        self._log("-" * 70)

        # Summary
        self._log("\n📊 MODEL COMPARISON (individual models):")
        for model_id in ['model1', 'model2', 'model3', 'model4']:
            m = metrics[model_id]
            self._log(f"   [{model_id[-1]}] {MODEL_NAMES[model_id]}: {m['avg_points_per_match']:.2f} pts/match")

        best_id = max(metrics.keys(), key=lambda k: metrics[k]['avg_points_per_match'])
        best_score = metrics[best_id]['avg_points_per_match']
        self._log(f"\n   Best individual: {MODEL_NAMES[best_id]} ({best_score:.2f} pts/match)")
        self._log(f"\n   🎯 USING: Consensus Ensemble (combines all 4 models)")
        self._log(f"      Strategy: If 3+ models agree → use consensus, else → Model4 fallback")

        self._is_trained = True
        self._training_metrics = metrics
        return metrics

    def predict_next_matchday(self) -> List[Dict[str, Any]]:
        """
        Predict scores for the next matchday.

        Returns:
            List of prediction dictionaries
        """
        if not self._is_trained:
            self.load_data()
            self.train_models()

        self._log("\n4. Fetching next matchday fixtures...")
        fixtures = self._fixtures_fetcher.get_next_matchday()

        if not fixtures:
            self._log("   No fixtures found for next matchday.")
            return []

        matchday_num = fixtures[0].get('matchday', '?')
        self._log(f"   Found {len(fixtures)} fixtures for Matchday {matchday_num}")

        self._log("\n5. Making predictions...")
        upcoming_odds = self._fixtures_fetcher.load_upcoming_odds()

        predictions = []
        for fixture in fixtures:
            pred = self._predict_match(fixture, upcoming_odds)
            predictions.append(pred)

        return predictions

    def _predict_match(
        self,
        fixture: Dict[str, Any],
        upcoming_odds: Optional[List[Dict]]
    ) -> Dict[str, Any]:
        """Make prediction for a single match."""
        home_team = fixture['homeTeam']['name']
        away_team = fixture['awayTeam']['name']
        match_date = pd.Timestamp(fixture['utcDate'])

        # Get odds
        odds_home, odds_draw, odds_away = self._fixtures_fetcher.find_fixture_odds(
            home_team, away_team, upcoming_odds
        )

        # Extract features
        X = self._feature_extractor.extract_match_features(
            home_team, away_team, match_date, odds_home, odds_draw, odds_away
        )

        # Get predictions from all models
        model_predictions = {}
        details = {}

        for model_id, model in self._models.items():
            if model_id == 'model4':
                pred = model.predict_with_details(X, odds_home=odds_home, odds_draw=odds_draw, odds_away=odds_away)
            else:
                pred = model.predict_with_details(X)

            model_predictions[model_id] = (pred['home_score'], pred['away_score'])
            details[model_id] = pred

        # Ensemble prediction
        ensemble_pred, strategy, ensemble_details = self._ensemble.combine(model_predictions)

        return {
            'home_team': home_team,
            'away_team': away_team,
            'date': match_date.strftime('%Y-%m-%d %H:%M'),
            'matchday': fixture.get('matchday', '?'),
            'odds': {
                'home': odds_home,
                'draw': odds_draw,
                'away': odds_away,
            },
            'predictions': {
                model_id: {
                    'scoreline': f"{p[0]}-{p[1]}",
                    'home': p[0],
                    'away': p[1],
                    'details': details[model_id],
                }
                for model_id, p in model_predictions.items()
            },
            'ensemble': {
                'scoreline': f"{ensemble_pred[0]}-{ensemble_pred[1]}",
                'home': ensemble_pred[0],
                'away': ensemble_pred[1],
                'strategy': strategy,
                'details': ensemble_details,
            },
        }

    def print_predictions(self, predictions: List[Dict[str, Any]]):
        """
        Print predictions in a formatted way.

        Args:
            predictions: List of prediction dictionaries
        """
        if not predictions:
            print("No predictions to display.")
            return

        matchday = predictions[0].get('matchday', '?')

        print("\n" + "=" * 70)
        print(f"PREDICTIONS FOR MATCHDAY {matchday}")
        print("=" * 70)

        for pred in predictions:
            print(f"\n{pred['home_team']} vs {pred['away_team']}")
            print(f"  Date: {pred['date']}")

            if pred['odds']['home']:
                print(f"  Odds: {pred['odds']['home']:.2f} - {pred['odds']['draw']:.2f} - {pred['odds']['away']:.2f}")

            print()
            for model_id in ['model1', 'model2', 'model3', 'model4']:
                name = MODEL_NAMES[model_id]
                scoreline = pred['predictions'][model_id]['scoreline']
                print(f"  [{model_id[-1]}] {name:<28} {scoreline}")

            # Ensemble
            ens = pred['ensemble']
            print()
            print(f"  [E] ENSEMBLE:                    {ens['scoreline']}  <-- USE THIS")
            if ens['strategy'] == 'consensus':
                count = ens['details']['count']
                print(f"      Strategy: CONSENSUS ({count}/4 models agree)")
            else:
                print(f"      Strategy: {MODEL_NAMES.get(ens['details']['fallback_model'], 'fallback')}")

        # Summary table
        print("\n" + "=" * 70)
        print("ENSEMBLE PREDICTIONS SUMMARY")
        print("=" * 70)
        print(f"{'Match':<45} {'Prediction':>12} {'Strategy':>12}")
        print("-" * 70)

        for pred in predictions:
            match_name = f"{pred['home_team'][:20]} vs {pred['away_team'][:20]}"
            ens = pred['ensemble']
            strategy = "CONS" if ens['strategy'] == 'consensus' else "M4(fallback)"
            print(f"{match_name:<45} {ens['scoreline']:>12} {strategy:>12}")

        print("-" * 70)
        print("\nStrategy: CONS = Consensus (3+ agree), M4 = Naive Odds fallback")

    def run(self) -> List[Dict[str, Any]]:
        """
        Run the full prediction pipeline.

        This is the main entry point for weekly predictions.

        Returns:
            List of prediction dictionaries
        """
        print("=" * 70)
        print("Bundesliga Match Predictor")
        print("=" * 70)

        predictions = self.predict_next_matchday()
        self.print_predictions(predictions)

        return predictions

    def backtest_ensemble(self) -> Dict[str, Any]:
        """
        Run backtest of ensemble on current season.

        Returns:
            Dictionary with backtest results
        """
        if self._df is None:
            self.load_data()

        # Determine current season start
        now = datetime.now()
        if now.month >= 8:
            season_start = pd.Timestamp(f'{now.year}-08-01', tz='UTC')
        else:
            season_start = pd.Timestamp(f'{now.year-1}-08-01', tz='UTC')

        current_season = self._df[self._df['date'] >= season_start].copy()

        self._log(f"\nBacktesting on {len(current_season)} matches from current season...")

        # This would implement full backtesting logic
        # For now, return placeholder
        return {'note': 'Full backtest available in backtest module'}


def main():
    """Main entry point for command-line usage."""
    predictor = BundesligaPredictor()
    predictor.run()


if __name__ == "__main__":
    main()
