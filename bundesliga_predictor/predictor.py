"""
Main predictor class that orchestrates all components.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    MODEL_NAMES, TRAIN_TEST_SPLIT_RATIO, MIN_TRAINING_SAMPLES,
    RECOMMENDATIONS_FILE, TRAINING_SEASONS, CACHE_DIR
)
from .data import DataLoader, FixturesFetcher
from .features import FeatureExtractor
from .scoring import calculate_kicktipp_points
from .ensemble import ConsensusEnsemble
from .ensemble_v2 import (
    HybridEnsembleV2,
    SimpleTendencyEnsemble,
    TendencyExpertEnsemble,
    TendencyConsensusEnsemble,
)
from .ensembles.experimental import (
    OptimizedConsensusEnsemble,
    HybridEnsemble,
    AdaptiveScorelineEnsemble,
    BayesianOptimalEnsemble,
    AggressiveScorelineEnsemble,
    UltimateTendencyEnsemble,
    SuperConsensusEnsemble,
    MaxPointsEnsemble,
)
from .models import (
    BaseModel,
    MultiOutputRegressionModel,
    MultiClassClassificationModel,
    PoissonRegressionModel,
    NaiveOddsModel,
)
from .models.experimental import (
    GradientBoostingModel,
    BivariatePoissonModel,
    SmartOddsModel,
    TendencyFirstModel,
    ProbabilityMaxModel,
)

# Backtest results file path
BACKTEST_RESULTS_FILE = os.path.join(CACHE_DIR, 'backtest_results.json')

# Map of model keys to classes (for experimental models)
MODEL_CLASSES = {
    'model1': MultiOutputRegressionModel,
    'model2': MultiClassClassificationModel,
    'model3': PoissonRegressionModel,
    'model4': NaiveOddsModel,
    'gradient_boosting': GradientBoostingModel,
    'bivariate_poisson': BivariatePoissonModel,
    'smart_odds': SmartOddsModel,
    'tendency_first': TendencyFirstModel,
    'probability_max': ProbabilityMaxModel,
}

# Core models that are always used
CORE_MODELS = ['model1', 'model2', 'model3', 'model4']

# Map of ensemble class names to classes
ENSEMBLE_CLASSES = {
    'ConsensusEnsemble': ConsensusEnsemble,
    'SimpleTendencyEnsemble': SimpleTendencyEnsemble,
    'TendencyExpertEnsemble': TendencyExpertEnsemble,
    'TendencyConsensusEnsemble': TendencyConsensusEnsemble,
    'HybridEnsembleV2': HybridEnsembleV2,
    'OptimizedConsensusEnsemble': OptimizedConsensusEnsemble,
    'HybridEnsemble': HybridEnsemble,
    'AdaptiveScorelineEnsemble': AdaptiveScorelineEnsemble,
    'BayesianOptimalEnsemble': BayesianOptimalEnsemble,
    'AggressiveScorelineEnsemble': AggressiveScorelineEnsemble,
    'UltimateTendencyEnsemble': UltimateTendencyEnsemble,
    'SuperConsensusEnsemble': SuperConsensusEnsemble,
    'MaxPointsEnsemble': MaxPointsEnsemble,
}


def load_backtest_results():
    """
    Load backtest results including best model and best ensemble.

    Returns:
        Dict with 'best_model' and 'best_ensemble' info, or defaults if not available
    """
    defaults = {
        'best_model': {
            'key': 'model4',
            'display_name': 'Naive Odds',
            'avg_points': None,
        },
        'best_ensemble': {
            'instance': HybridEnsembleV2(),
            'class_name': 'HybridEnsembleV2',
            'display_name': 'Hybrid V2',
            'avg_points': None,
        }
    }

    if not os.path.exists(BACKTEST_RESULTS_FILE):
        return defaults

    try:
        with open(BACKTEST_RESULTS_FILE, 'r') as f:
            results = json.load(f)

        # Load best model info
        best_model = results.get('best_model', {})
        if best_model:
            defaults['best_model'] = {
                'key': best_model.get('key', 'model4'),
                'display_name': best_model.get('display_name', 'Naive Odds'),
                'avg_points': best_model.get('avg_points'),
            }

        # Load best ensemble info
        best_ensemble = results.get('best_ensemble', {})
        if best_ensemble:
            class_name = best_ensemble.get('class_name', 'HybridEnsembleV2')
            ensemble_class = ENSEMBLE_CLASSES.get(class_name, HybridEnsembleV2)
            defaults['best_ensemble'] = {
                'instance': ensemble_class(),
                'class_name': class_name,
                'display_name': best_ensemble.get('display_name', 'Hybrid V2'),
                'avg_points': best_ensemble.get('avg_points'),
            }

        return defaults

    except (json.JSONDecodeError, KeyError):
        return defaults


def load_best_ensemble():
    """
    Load the best ensemble from backtest results.
    Legacy function for backwards compatibility.

    Returns:
        Tuple of (ensemble_instance, ensemble_name, avg_points) or (HybridEnsembleV2(), 'HybridEnsembleV2', None) as default
    """
    results = load_backtest_results()
    best = results['best_ensemble']
    return best['instance'], best['class_name'], best['avg_points']


# Bundesliga derbies and rivalries - used for derby-aware predictions
DERBIES = {
    ('bayern', 'dortmund'), ('dortmund', 'bayern'),  # Der Klassiker
    ('dortmund', 'schalke'), ('schalke', 'dortmund'),  # Revierderby
    ('gladbach', 'koln'), ('koln', 'gladbach'),  # Rhine Derby
    ('hamburg', 'bremen'), ('bremen', 'hamburg'),  # Nordderby
    ('hamburg', 'st pauli'), ('st pauli', 'hamburg'),  # Hamburg Derby
    ('frankfurt', 'mainz'), ('mainz', 'frankfurt'),  # Rhein-Main Derby
    ('leverkusen', 'koln'), ('koln', 'leverkusen'),  # Rhineland Derby
    ('union berlin', 'hertha'), ('hertha', 'union berlin'),  # Berlin Derby
    ('leipzig', 'union berlin'), ('union berlin', 'leipzig'),  # East German Derby
}


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

        # Load backtest results (best model and best ensemble)
        backtest_results = load_backtest_results()

        # Best model info
        self._best_model_key = backtest_results['best_model']['key']
        self._best_model_name = backtest_results['best_model']['display_name']
        self._best_model_score = backtest_results['best_model']['avg_points']

        # Best ensemble info
        self._ensemble = backtest_results['best_ensemble']['instance']
        self._ensemble_name = backtest_results['best_ensemble']['class_name']
        self._ensemble_display_name = backtest_results['best_ensemble']['display_name']
        self._ensemble_score = backtest_results['best_ensemble']['avg_points']

        # Initialize core models
        self._models: Dict[str, BaseModel] = {
            'model1': MultiOutputRegressionModel(),
            'model2': MultiClassClassificationModel(),
            'model3': PoissonRegressionModel(),
            'model4': NaiveOddsModel(),
        }

        # Add best model if it's experimental (not a core model)
        if self._best_model_key not in CORE_MODELS:
            model_class = MODEL_CLASSES.get(self._best_model_key)
            if model_class:
                self._models[self._best_model_key] = model_class()
                self._log(f"   Added experimental model: {self._best_model_name}")

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

        self._log(f"   Date range: {self._df['date'].min().date()} to {self._df['date'].max().date()}")
        return self._df

    def _get_training_data(self) -> pd.DataFrame:
        """
        Get training data: last N seasons + current season data.

        Uses TRAINING_SEASONS config to determine how many previous seasons.
        """
        if self._df is None:
            self.load_data()

        df = self._df.copy()

        if 'Season' in df.columns:
            seasons = sorted(df['Season'].unique())
            current_season = seasons[-1]

            # Get last N seasons before current + current season data
            training_seasons = seasons[-(TRAINING_SEASONS + 1):]
            training_df = df[df['Season'].isin(training_seasons)].copy()

            self._log(f"   Training seasons: {', '.join(training_seasons)}")
        else:
            # Fallback: use date-based selection
            now = datetime.now()
            if now.month >= 8:
                current_season_start = pd.Timestamp(f'{now.year}-08-01')
            else:
                current_season_start = pd.Timestamp(f'{now.year-1}-08-01')

            training_start = current_season_start - pd.DateOffset(years=TRAINING_SEASONS)
            training_df = df[df['date'] >= training_start].copy()

            self._log(f"   Training from: {training_start.date()}")

        self._log(f"   Training matches: {len(training_df)}")
        return training_df

    def train_models(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, float]]:
        """
        Train all models on the data.

        Uses last N seasons + current season data for training (walk-forward style).

        Args:
            df: Optional DataFrame to use (uses training data selection if not provided)

        Returns:
            Dictionary of training metrics per model
        """
        if df is None:
            df = self._get_training_data()

        # Create feature extractor with training data
        self._feature_extractor = FeatureExtractor(df)

        self._log("\n2. Creating features...")
        X, y_home, y_away, scorelines = self._feature_extractor.extract_training_features(df)
        self._log(f"   Created features for {len(X)} matches")
        self._log(f"   Unique scorelines: {len(np.unique(scorelines))}")

        # Temporal split for validation
        split_idx = int(len(X) * TRAIN_TEST_SPLIT_RATIO)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_home_train, y_home_test = y_home[:split_idx], y_home[split_idx:]
        y_away_train, y_away_test = y_away[:split_idx], y_away[split_idx:]
        scorelines_train, scorelines_test = scorelines[:split_idx], scorelines[split_idx:]

        self._log(f"   Training on {len(X_train)} matches, validating on {len(X_test)}")

        self._log("\n3. Training models...")
        self._log("-" * 70)

        metrics = {}

        for model_id, model in self._models.items():
            # Models that don't need scorelines for training
            no_scoreline_models = ['model4', 'smart_odds']
            if model_id in no_scoreline_models:
                model.train(X_train, y_home_train, y_away_train)
                eval_metrics = model.evaluate(X_test, y_home_test, y_away_test)
            else:
                # Train ML models with scorelines
                model.train(X_train, y_home_train, y_away_train, scorelines_train)
                eval_metrics = model.evaluate(X_test, y_home_test, y_away_test)

            metrics[model_id] = eval_metrics

            # Use model name for display, with special handling for experimental models
            display_label = model_id[-1] if model_id.startswith('model') else model_id[0].upper()
            self._log(f"\n[{display_label}] {model.name}:")
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

        # Show which ensemble is being used
        ensemble_display = self._ensemble_name.replace('Ensemble', '').replace('V2', ' V2')
        if self._ensemble_score:
            self._log(f"\n   🎯 USING: {ensemble_display} (from backtest: {self._ensemble_score:.2f} pts/match)")
        else:
            self._log(f"\n   🎯 USING: {ensemble_display} (default - run backtest to optimize)")

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

    def _normalize_team_name(self, name: str) -> str:
        """Normalize team name for derby detection."""
        name = str(name).lower().strip()

        replacements = {
            'fc bayern münchen': 'bayern', 'bayern munich': 'bayern',
            'borussia dortmund': 'dortmund', 'bvb': 'dortmund',
            'borussia mönchengladbach': 'gladbach', "m'gladbach": 'gladbach',
            'fc schalke 04': 'schalke', 'schalke 04': 'schalke',
            '1. fc köln': 'koln', 'fc köln': 'koln', 'koln': 'koln',
            'hamburger sv': 'hamburg', 'hsv': 'hamburg',
            'sv werder bremen': 'bremen', 'werder bremen': 'bremen',
            'fc st. pauli': 'st pauli', 'fc st. pauli 1910': 'st pauli',
            'eintracht frankfurt': 'frankfurt', 'sge': 'frankfurt',
            '1. fsv mainz 05': 'mainz', 'mainz 05': 'mainz',
            'bayer 04 leverkusen': 'leverkusen', 'leverkusen': 'leverkusen',
            '1. fc union berlin': 'union berlin', 'union berlin': 'union berlin',
            'hertha bsc': 'hertha', 'hertha berlin': 'hertha',
            'rb leipzig': 'leipzig', 'rasenballsport leipzig': 'leipzig',
        }

        for old, new in replacements.items():
            if old in name:
                return new
        return name

    def _is_derby(self, home_team: str, away_team: str) -> bool:
        """Check if a match is a derby."""
        home_norm = self._normalize_team_name(home_team)
        away_norm = self._normalize_team_name(away_team)
        return (home_norm, away_norm) in DERBIES

    def _predict_match(
        self,
        fixture: Dict[str, Any],
        upcoming_odds: Optional[List[Dict]]
    ) -> Dict[str, Any]:
        """Make prediction for a single match."""
        home_team = fixture['homeTeam']['name']
        away_team = fixture['awayTeam']['name']
        # Convert to tz-naive timestamp to match training data
        match_date = pd.Timestamp(fixture['utcDate']).tz_localize(None)

        # Get odds
        odds_home, odds_draw, odds_away = self._fixtures_fetcher.find_fixture_odds(
            home_team, away_team, upcoming_odds
        )

        # Check if this is a derby
        is_derby = self._is_derby(home_team, away_team)

        # Extract features
        X = self._feature_extractor.extract_match_features(
            home_team, away_team, match_date, odds_home, odds_draw, odds_away
        )

        # Get predictions from all models
        model_predictions = {}
        details = {}

        for model_id, model in self._models.items():
            # Models that need odds for prediction
            odds_models = ['model4', 'smart_odds']
            if model_id in odds_models:
                pred = model.predict_with_details(X, odds_home=odds_home, odds_draw=odds_draw, odds_away=odds_away)
            else:
                pred = model.predict_with_details(X)

            model_predictions[model_id] = (pred['home_score'], pred['away_score'])
            details[model_id] = pred

        # Ensemble prediction (pass odds and derby flag for improved predictions)
        ensemble_pred, strategy, ensemble_details = self._ensemble.combine(
            model_predictions,
            odds_home=odds_home,
            odds_draw=odds_draw,
            odds_away=odds_away,
            is_derby=is_derby
        )

        return {
            'home_team': home_team,
            'away_team': away_team,
            'date': match_date.strftime('%Y-%m-%d %H:%M'),
            'matchday': fixture.get('matchday', '?'),
            'is_derby': is_derby,
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
            # Print core models
            for model_id in CORE_MODELS:
                name = MODEL_NAMES[model_id]
                scoreline = pred['predictions'][model_id]['scoreline']
                # Mark best model
                marker = " *" if model_id == self._best_model_key else ""
                print(f"  [{model_id[-1]}] {name:<28} {scoreline}{marker}")

            # Print best model if it's experimental
            if self._best_model_key not in CORE_MODELS and self._best_model_key in pred['predictions']:
                scoreline = pred['predictions'][self._best_model_key]['scoreline']
                print(f"  [B] {self._best_model_name:<28} {scoreline} *")

            # Ensemble
            ens = pred['ensemble']
            print()
            print(f"  [E] ENSEMBLE:                    {ens['scoreline']}  <-- USE THIS")
            if ens['strategy'] in ['consensus', 'tendency_consensus', 'tendency_expert']:
                print(f"      Strategy: {ens['strategy'].upper()}")
            elif 'fallback_model' in ens.get('details', {}):
                print(f"      Strategy: {MODEL_NAMES.get(ens['details']['fallback_model'], 'fallback')}")
            else:
                print(f"      Strategy: {ens['strategy']}")

        # Best Model predictions table
        print("\n" + "=" * 70)
        model_title = f"BEST MODEL: {self._best_model_name}"
        if self._best_model_score:
            model_title += f" ({self._best_model_score:.2f} pts/match)"
        print(model_title)
        print("=" * 70)
        print(f"{'Match':<50} {'Prediction':>12}")
        print("-" * 70)

        for pred in predictions:
            match_name = f"{pred['home_team'][:22]} vs {pred['away_team'][:22]}"
            best_model_pred = pred['predictions'].get(self._best_model_key, {})
            scoreline = best_model_pred.get('scoreline', '?-?')
            print(f"{match_name:<50} {scoreline:>12}")

        print("-" * 70)

        # Best Ensemble predictions table
        print("\n" + "=" * 70)
        ensemble_title = f"BEST ENSEMBLE: {self._ensemble_display_name}"
        if self._ensemble_score:
            ensemble_title += f" ({self._ensemble_score:.2f} pts/match)"
        print(ensemble_title)
        print("=" * 70)
        print(f"{'Match':<50} {'Prediction':>12}")
        print("-" * 70)

        for pred in predictions:
            match_name = f"{pred['home_team'][:22]} vs {pred['away_team'][:22]}"
            ens = pred['ensemble']
            print(f"{match_name:<50} {ens['scoreline']:>12}")

        print("-" * 70)

        # Note about source
        if self._best_model_score or self._ensemble_score:
            print("\nScores from walk-forward backtest on current season.")

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
