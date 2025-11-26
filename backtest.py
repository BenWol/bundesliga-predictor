#!/usr/bin/env python3
"""
Comprehensive backtest for all models and ensemble strategies.

Tests:
1. All core models (model1-4)
2. All experimental models
3. All ensemble strategies
4. Rolling window validation for robust estimates

Usage:
    uv run python backtest.py                  # Full backtest
    uv run python backtest.py --quick          # Current season only
    uv run python backtest.py --models-only    # Skip ensembles
    uv run python backtest.py --ensembles-only # Skip individual models
"""

import argparse
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional

from bundesliga_predictor import BundesligaPredictor
from bundesliga_predictor.scoring import calculate_kicktipp_points
from bundesliga_predictor.config import MODEL_NAMES

# Core models
from bundesliga_predictor.models import (
    MultiOutputRegressionModel,
    MultiClassClassificationModel,
    PoissonRegressionModel,
    NaiveOddsModel,
)

# Experimental models
from bundesliga_predictor.models.experimental import (
    GradientBoostingModel,
    BivariatePoissonModel,
    SmartOddsModel,
    TendencyFirstModel,
    ProbabilityMaxModel,
)

# Ensembles
from bundesliga_predictor.ensemble import ConsensusEnsemble
from bundesliga_predictor.ensemble_v2 import (
    SimpleTendencyEnsemble,
    TendencyExpertEnsemble,
    TendencyConsensusEnsemble,
    HybridEnsembleV2,
)
from bundesliga_predictor.ensembles.experimental import (
    OptimizedConsensusEnsemble,
    HybridEnsemble,
    AdaptiveScorelineEnsemble,
    BayesianOptimalEnsemble,
    AggressiveScorelineEnsemble,
    UltimateTendencyEnsemble,
    SuperConsensusEnsemble,
    MaxPointsEnsemble,
)


# Extended model names for display
EXTENDED_MODEL_NAMES = {
    **MODEL_NAMES,
    'gradient_boosting': 'Gradient Boosting',
    'bivariate_poisson': 'Bivariate Poisson',
    'smart_odds': 'Smart Odds',
    'tendency_first': 'Tendency First',
    'probability_max': 'Probability Max',
}


def create_all_models() -> Dict[str, Any]:
    """Create all available models."""
    return {
        # Core models
        'model1': MultiOutputRegressionModel(),
        'model2': MultiClassClassificationModel(),
        'model3': PoissonRegressionModel(),
        'model4': NaiveOddsModel(),
        # Experimental models
        'gradient_boosting': GradientBoostingModel(),
        'bivariate_poisson': BivariatePoissonModel(),
        'smart_odds': SmartOddsModel(),
        'tendency_first': TendencyFirstModel(),
        'probability_max': ProbabilityMaxModel(),
    }


def create_all_ensembles() -> Dict[str, Any]:
    """Create all available ensembles."""
    return {
        # Original
        'consensus': ConsensusEnsemble(),
        # V2 (validated)
        'simple_tendency': SimpleTendencyEnsemble(),
        'tendency_expert': TendencyExpertEnsemble(),
        'tendency_consensus': TendencyConsensusEnsemble(),
        'hybrid_v2': HybridEnsembleV2(),
        # Experimental
        'optimized_consensus': OptimizedConsensusEnsemble(),
        'hybrid': HybridEnsemble(),
        'adaptive_scoreline': AdaptiveScorelineEnsemble(),
        'bayesian_optimal': BayesianOptimalEnsemble(),
        'aggressive_scoreline': AggressiveScorelineEnsemble(),
        'ultimate_tendency': UltimateTendencyEnsemble(),
        'super_consensus': SuperConsensusEnsemble(),
        'max_points': MaxPointsEnsemble(),
    }


def train_models(models: Dict, feature_extractor, train_df: pd.DataFrame):
    """Train all models on training data."""
    X_train, y_home, y_away, scorelines = feature_extractor.extract_training_features(train_df)

    for model_id, model in models.items():
        try:
            if model_id == 'model4' or model_id == 'smart_odds':
                model.train(X_train, y_home, y_away)
            else:
                model.train(X_train, y_home, y_away, scorelines)
        except Exception as e:
            print(f"  Warning: Could not train {model_id}: {e}")


def evaluate_on_match(
    models: Dict,
    ensembles: Dict,
    feature_extractor,
    match: pd.Series,
    core_model_ids: List[str] = None
) -> Dict[str, int]:
    """Evaluate all models and ensembles on a single match."""
    results = {}

    home_team = match['home_team']
    away_team = match['away_team']
    actual_home = int(match['home_score'])
    actual_away = int(match['away_score'])

    odds_home = match.get('odds_home')
    odds_draw = match.get('odds_draw')
    odds_away = match.get('odds_away')

    # Extract features
    try:
        X = feature_extractor.extract_match_features(
            home_team, away_team, match['date'],
            odds_home, odds_draw, odds_away
        )
    except Exception as e:
        return {}

    # Evaluate each model
    model_preds = {}
    xg_home, xg_away = None, None

    for model_id, model in models.items():
        try:
            if model_id in ['model4', 'smart_odds']:
                pred = model.predict(X, odds_home=odds_home, odds_draw=odds_draw, odds_away=odds_away)
            else:
                pred = model.predict(X)

            model_preds[model_id] = pred
            pts = calculate_kicktipp_points(pred[0], pred[1], actual_home, actual_away)
            results[model_id] = pts

            # Get xG for ensemble use
            if model_id == 'model3':
                try:
                    xg_home, xg_away = model.get_expected_goals(X)
                except:
                    pass
        except Exception as e:
            pass

    # Filter to core models for ensemble
    if core_model_ids is None:
        core_model_ids = ['model1', 'model2', 'model3', 'model4']

    core_preds = {k: v for k, v in model_preds.items() if k in core_model_ids}

    if len(core_preds) < 4:
        return results

    # Evaluate each ensemble
    for ens_id, ensemble in ensembles.items():
        try:
            # Different ensembles have different signatures
            try:
                ens_pred, _, _ = ensemble.combine(
                    core_preds,
                    odds_home=odds_home,
                    odds_draw=odds_draw,
                    odds_away=odds_away,
                    xg_home=xg_home,
                    xg_away=xg_away
                )
            except TypeError:
                try:
                    ens_pred, _, _ = ensemble.combine(
                        core_preds,
                        odds_home=odds_home,
                        odds_draw=odds_draw,
                        odds_away=odds_away
                    )
                except TypeError:
                    ens_pred, _, _ = ensemble.combine(core_preds)

            pts = calculate_kicktipp_points(ens_pred[0], ens_pred[1], actual_home, actual_away)
            results[f"ens_{ens_id}"] = pts
        except Exception as e:
            pass

    return results


def run_current_season_backtest(
    predictor: BundesligaPredictor,
    include_models: bool = True,
    include_ensembles: bool = True,
    walk_forward: bool = True,
    training_seasons: int = 2
) -> Dict[str, Dict]:
    """
    Run backtest on current season using walk-forward methodology.

    Walk-forward backtesting:
    1. Train on last N seasons before matchday 1
    2. Predict matchday 1
    3. Add matchday 1 results to training data
    4. Retrain and predict matchday 2
    5. Continue until all matchdays are predicted

    Args:
        training_seasons: Number of previous seasons to use for training (default: 2)
    """
    print("\n" + "=" * 80)
    print("CURRENT SEASON BACKTEST" + (" (Walk-Forward)" if walk_forward else " (Static)"))
    print("=" * 80)

    df = predictor._df.copy()

    # Determine current season and training seasons
    if 'Season' in df.columns:
        seasons = sorted(df['Season'].unique())
        current_season_name = seasons[-1]  # Most recent season
        current_season = df[df['Season'] == current_season_name].copy()

        # Use only the last N seasons before current for training
        training_season_names = seasons[-(training_seasons + 1):-1]  # e.g., last 2 before current
        historical = df[df['Season'].isin(training_season_names)].copy()
    else:
        now = datetime.now()
        if now.month >= 8:
            season_start = pd.Timestamp(f'{now.year}-08-01')
        else:
            season_start = pd.Timestamp(f'{now.year-1}-08-01')
        current_season = df[df['date'] >= season_start].copy()
        # Approximate: use last N years
        training_start = season_start - pd.DateOffset(years=training_seasons)
        historical = df[(df['date'] >= training_start) & (df['date'] < season_start)].copy()
        current_season_name = f"{season_start.year}/{season_start.year + 1}"
        training_season_names = [f"~{training_seasons} seasons"]

    print(f"\nCurrent season: {current_season_name}")
    print(f"Training seasons: {', '.join(training_season_names) if isinstance(training_season_names, list) else training_season_names}")
    print(f"Historical matches (training base): {len(historical)}")
    print(f"Current season matches to predict: {len(current_season)}")

    # Create models and ensembles
    models = create_all_models() if include_models else {
        'model1': MultiOutputRegressionModel(),
        'model2': MultiClassClassificationModel(),
        'model3': PoissonRegressionModel(),
        'model4': NaiveOddsModel()
    }
    ensembles = create_all_ensembles() if include_ensembles else {}

    # Initialize results
    results = {k: {'total': 0, 'exact': 0, 'diff': 0, 'tend': 0, 'wrong': 0, 'n': 0}
               for k in list(models.keys()) + [f"ens_{e}" for e in ensembles.keys()]}

    if walk_forward:
        # Walk-forward: Group matches by matchday (approximate by date)
        current_season = current_season.sort_values('date').reset_index(drop=True)

        # Group by date (matches on same day are same matchday)
        current_season['matchday_group'] = (
            current_season['date'].diff().dt.days.fillna(0) > 3
        ).cumsum()

        matchday_groups = current_season.groupby('matchday_group')
        n_matchdays = len(matchday_groups)

        print(f"Matchdays to process: {n_matchdays}")
        print("\nWalk-forward backtesting...")

        # Start with historical data
        training_data = historical.copy()

        from bundesliga_predictor.features import FeatureExtractor

        for md_idx, (_, matchday_matches) in enumerate(matchday_groups):
            md_num = md_idx + 1

            # Create fresh feature extractor with current training data
            feature_extractor = FeatureExtractor(training_data)

            # Train models on all available data
            train_models(models, feature_extractor, training_data)

            # Predict each match in this matchday
            for _, match in matchday_matches.iterrows():
                match_results = evaluate_on_match(models, ensembles, feature_extractor, match)

                for key, pts in match_results.items():
                    if key in results:
                        results[key]['total'] += pts
                        results[key]['n'] += 1

                        if pts == 4:
                            results[key]['exact'] += 1
                        elif pts == 3:
                            results[key]['diff'] += 1
                        elif pts == 2:
                            results[key]['tend'] += 1
                        else:
                            results[key]['wrong'] += 1

            # Add this matchday's results to training data for next iteration
            training_data = pd.concat([training_data, matchday_matches], ignore_index=True)

            # Progress indicator
            if md_num % 5 == 0 or md_num == n_matchdays:
                print(f"  Completed matchday {md_num}/{n_matchdays} "
                      f"(training size: {len(training_data)})")
    else:
        # Static training (original behavior)
        print("\nTraining models (static)...")
        from bundesliga_predictor.features import FeatureExtractor
        feature_extractor = FeatureExtractor(historical)
        train_models(models, feature_extractor, historical)

        print("Evaluating on current season...")
        for _, match in current_season.iterrows():
            match_results = evaluate_on_match(models, ensembles, feature_extractor, match)

            for key, pts in match_results.items():
                if key in results:
                    results[key]['total'] += pts
                    results[key]['n'] += 1

                    if pts == 4:
                        results[key]['exact'] += 1
                    elif pts == 3:
                        results[key]['diff'] += 1
                    elif pts == 2:
                        results[key]['tend'] += 1
                    else:
                        results[key]['wrong'] += 1

    return results


def run_rolling_validation(
    predictor: BundesligaPredictor,
    n_seasons: int = 3,
    include_models: bool = True,
    include_ensembles: bool = True
) -> Dict[str, Dict]:
    """Run rolling window validation across multiple seasons."""
    print("\n" + "=" * 80)
    print(f"ROLLING WINDOW VALIDATION (last {n_seasons} seasons)")
    print("=" * 80)

    df = predictor._df.sort_values('date').reset_index(drop=True)

    # Get seasons
    if 'Season' in df.columns:
        seasons = sorted(df['Season'].unique())
        test_seasons = seasons[-n_seasons:] if len(seasons) >= n_seasons else seasons
    else:
        # Approximate seasons by year
        df['season_year'] = df['date'].dt.year
        seasons = sorted(df['season_year'].unique())
        test_seasons = seasons[-n_seasons:] if len(seasons) >= n_seasons else seasons

    print(f"Testing on seasons: {test_seasons}")

    # Initialize results
    all_results = {}

    for test_season in test_seasons:
        print(f"\nSeason {test_season}...")

        if 'Season' in df.columns:
            train_df = df[df['Season'] < test_season].copy()
            test_df = df[df['Season'] == test_season].copy()
        else:
            train_df = df[df['season_year'] < test_season].copy()
            test_df = df[df['season_year'] == test_season].copy()

        if len(train_df) < 300:
            print(f"  Skipping - not enough training data ({len(train_df)} matches)")
            continue

        print(f"  Training on {len(train_df)} matches, testing on {len(test_df)}")

        # Create fresh feature extractor
        from bundesliga_predictor.features import FeatureExtractor
        feature_extractor = FeatureExtractor(train_df)

        # Create and train models
        models = create_all_models() if include_models else {'model1': MultiOutputRegressionModel(), 'model2': MultiClassClassificationModel(), 'model3': PoissonRegressionModel(), 'model4': NaiveOddsModel()}
        ensembles = create_all_ensembles() if include_ensembles else {}

        train_models(models, feature_extractor, train_df)

        # Evaluate
        for _, match in test_df.iterrows():
            match_results = evaluate_on_match(models, ensembles, feature_extractor, match)

            for key, pts in match_results.items():
                if key not in all_results:
                    all_results[key] = {'total': 0, 'exact': 0, 'diff': 0, 'tend': 0, 'wrong': 0, 'n': 0}

                all_results[key]['total'] += pts
                all_results[key]['n'] += 1

                if pts == 4:
                    all_results[key]['exact'] += 1
                elif pts == 3:
                    all_results[key]['diff'] += 1
                elif pts == 2:
                    all_results[key]['tend'] += 1
                else:
                    all_results[key]['wrong'] += 1

    return all_results


def print_results(results: Dict[str, Dict], title: str, benchmark: float = 1.65):
    """Print results in a formatted table."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if not results:
        print("No results to display.")
        return

    # Separate models and ensembles
    model_results = {k: v for k, v in results.items() if not k.startswith('ens_')}
    ensemble_results = {k: v for k, v in results.items() if k.startswith('ens_')}

    # Print models
    if model_results:
        print(f"\n{'MODEL':<25} {'Total':>6} {'Avg':>7} {'Exact':>6} {'Diff':>6} {'Tend':>6} {'Wrong':>6} {'vs Bench':>8}")
        print("-" * 80)

        sorted_models = sorted(model_results.items(), key=lambda x: x[1]['total'] / x[1]['n'] if x[1]['n'] > 0 else 0, reverse=True)

        for name, r in sorted_models:
            if r['n'] == 0:
                continue
            avg = r['total'] / r['n']
            display_name = EXTENDED_MODEL_NAMES.get(name, name)[:24]
            diff = avg - benchmark
            marker = "*" if avg >= benchmark else ""
            print(f"{display_name:<25} {r['total']:>6} {avg:>7.2f} {r['exact']:>6} {r['diff']:>6} {r['tend']:>6} {r['wrong']:>6} {diff:>+7.2f}{marker}")

    # Print ensembles
    if ensemble_results:
        print(f"\n{'ENSEMBLE':<25} {'Total':>6} {'Avg':>7} {'Exact':>6} {'Diff':>6} {'Tend':>6} {'Wrong':>6} {'vs Bench':>8}")
        print("-" * 80)

        sorted_ensembles = sorted(ensemble_results.items(), key=lambda x: x[1]['total'] / x[1]['n'] if x[1]['n'] > 0 else 0, reverse=True)

        for name, r in sorted_ensembles:
            if r['n'] == 0:
                continue
            avg = r['total'] / r['n']
            display_name = name.replace('ens_', '').replace('_', ' ').title()[:24]
            diff = avg - benchmark
            marker = "*" if avg >= benchmark else ""
            print(f"{display_name:<25} {r['total']:>6} {avg:>7.2f} {r['exact']:>6} {r['diff']:>6} {r['tend']:>6} {r['wrong']:>6} {diff:>+7.2f}{marker}")

    # Summary
    print("-" * 80)
    print(f"* = beats benchmark of {benchmark:.2f} pts/match")


BACKTEST_RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'backtest_results.json')


def save_backtest_results(current_results: Dict, benchmark: float = 1.65):
    """Save backtest results to JSON file for use by predictor."""
    # Find best ensemble
    ensemble_results = {k: v for k, v in current_results.items() if k.startswith('ens_')}

    if not ensemble_results:
        return

    # Calculate averages and find best
    ensemble_scores = {}
    for name, r in ensemble_results.items():
        if r['n'] > 0:
            avg = r['total'] / r['n']
            ensemble_scores[name] = {
                'avg_points': avg,
                'total_points': r['total'],
                'matches': r['n'],
                'exact': r['exact'],
                'diff': r['diff'],
                'tend': r['tend'],
                'wrong': r['wrong'],
            }

    if not ensemble_scores:
        return

    best_ensemble = max(ensemble_scores.keys(), key=lambda k: ensemble_scores[k]['avg_points'])
    best_score = ensemble_scores[best_ensemble]['avg_points']

    # Map ensemble key to class name
    ensemble_class_map = {
        'ens_consensus': 'ConsensusEnsemble',
        'ens_simple_tendency': 'SimpleTendencyEnsemble',
        'ens_tendency_expert': 'TendencyExpertEnsemble',
        'ens_tendency_consensus': 'TendencyConsensusEnsemble',
        'ens_hybrid_v2': 'HybridEnsembleV2',
        'ens_optimized_consensus': 'OptimizedConsensusEnsemble',
        'ens_hybrid': 'HybridEnsemble',
        'ens_adaptive_scoreline': 'AdaptiveScorelineEnsemble',
        'ens_bayesian_optimal': 'BayesianOptimalEnsemble',
        'ens_aggressive_scoreline': 'AggressiveScorelineEnsemble',
        'ens_ultimate_tendency': 'UltimateTendencyEnsemble',
        'ens_super_consensus': 'SuperConsensusEnsemble',
        'ens_max_points': 'MaxPointsEnsemble',
    }

    results_data = {
        'updated_at': datetime.now().isoformat(),
        'benchmark': benchmark,
        'best_ensemble': {
            'key': best_ensemble,
            'class_name': ensemble_class_map.get(best_ensemble, 'HybridEnsembleV2'),
            'avg_points': best_score,
            'beats_benchmark': best_score >= benchmark,
        },
        'all_ensembles': ensemble_scores,
    }

    with open(BACKTEST_RESULTS_FILE, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nBacktest results saved to {os.path.basename(BACKTEST_RESULTS_FILE)}")
    print(f"Best ensemble: {best_ensemble.replace('ens_', '').replace('_', ' ').title()} ({best_score:.2f} pts/match)")


def print_summary(current_results: Dict, rolling_results: Dict, benchmark: float = 1.65):
    """Print final summary and recommendations."""
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)

    # Find best performers
    def get_best(results, prefix=''):
        filtered = {k: v for k, v in results.items() if k.startswith(prefix)}
        if not filtered:
            return None, 0
        best = max(filtered.items(), key=lambda x: x[1]['total'] / x[1]['n'] if x[1]['n'] > 0 else 0)
        return best[0], best[1]['total'] / best[1]['n'] if best[1]['n'] > 0 else 0

    # Best model
    best_model_current, best_model_current_avg = get_best(current_results, '')
    best_model_current = best_model_current if not best_model_current.startswith('ens_') else None

    best_model_rolling, best_model_rolling_avg = get_best(rolling_results, '')
    best_model_rolling = best_model_rolling if not best_model_rolling or not best_model_rolling.startswith('ens_') else None

    # Best ensemble
    best_ens_current, best_ens_current_avg = get_best(current_results, 'ens_')
    best_ens_rolling, best_ens_rolling_avg = get_best(rolling_results, 'ens_')

    print(f"\nBenchmark: {benchmark:.2f} pts/match")

    if best_model_current:
        print(f"\nBest Model (Current Season): {EXTENDED_MODEL_NAMES.get(best_model_current, best_model_current)}")
        print(f"  -> {best_model_current_avg:.2f} pts/match ({'+' if best_model_current_avg >= benchmark else ''}{best_model_current_avg - benchmark:.2f} vs benchmark)")

    if best_model_rolling:
        print(f"\nBest Model (Rolling): {EXTENDED_MODEL_NAMES.get(best_model_rolling, best_model_rolling)}")
        print(f"  -> {best_model_rolling_avg:.2f} pts/match ({'+' if best_model_rolling_avg >= benchmark else ''}{best_model_rolling_avg - benchmark:.2f} vs benchmark)")

    if best_ens_current:
        ens_name = best_ens_current.replace('ens_', '').replace('_', ' ').title()
        print(f"\nBest Ensemble (Current Season): {ens_name}")
        print(f"  -> {best_ens_current_avg:.2f} pts/match ({'+' if best_ens_current_avg >= benchmark else ''}{best_ens_current_avg - benchmark:.2f} vs benchmark)")

    if best_ens_rolling:
        ens_name = best_ens_rolling.replace('ens_', '').replace('_', ' ').title()
        print(f"\nBest Ensemble (Rolling): {ens_name}")
        print(f"  -> {best_ens_rolling_avg:.2f} pts/match ({'+' if best_ens_rolling_avg >= benchmark else ''}{best_ens_rolling_avg - benchmark:.2f} vs benchmark)")

    # Recommendation
    print("\n" + "-" * 80)
    print("RECOMMENDATION:")
    print("-" * 80)

    if best_ens_rolling_avg >= benchmark:
        ens_name = best_ens_rolling.replace('ens_', '').replace('_', ' ').title()
        print(f"\n  USE: {ens_name}")
        print(f"  Achieves {best_ens_rolling_avg:.2f} pts/match (beats benchmark by {best_ens_rolling_avg - benchmark:.2f})")
    elif best_ens_current_avg >= benchmark:
        ens_name = best_ens_current.replace('ens_', '').replace('_', ' ').title()
        print(f"\n  CONSIDER: {ens_name}")
        print(f"  Achieves {best_ens_current_avg:.2f} pts/match on current season")
        print("  Note: Rolling validation did not confirm this result")
    else:
        print(f"\n  No ensemble beats the benchmark of {benchmark:.2f} pts/match")
        print("  Consider tuning ensemble parameters or trying new strategies")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive backtest for Bundesliga predictor')
    parser.add_argument('--quick', '-q', action='store_true', help='Quick mode - current season only')
    parser.add_argument('--static', action='store_true', help='Use static training (train once at season start)')
    parser.add_argument('--models-only', action='store_true', help='Only test individual models')
    parser.add_argument('--ensembles-only', action='store_true', help='Only test ensembles')
    parser.add_argument('--benchmark', type=float, default=1.65, help='Benchmark to compare against (default: 1.65)')
    parser.add_argument('--seasons', type=int, default=3, help='Number of seasons for rolling validation')
    parser.add_argument('--training-seasons', type=int, default=2, help='Number of previous seasons for training (default: 2)')

    args = parser.parse_args()

    include_models = not args.ensembles_only
    include_ensembles = not args.models_only
    walk_forward = not args.static

    # Initialize predictor
    predictor = BundesligaPredictor(verbose=False)
    predictor.load_data()

    print(f"Loaded {len(predictor._df)} matches")

    # Run current season backtest
    current_results = run_current_season_backtest(
        predictor,
        include_models=include_models,
        include_ensembles=include_ensembles,
        walk_forward=walk_forward,
        training_seasons=args.training_seasons
    )
    print_results(current_results, "CURRENT SEASON RESULTS", args.benchmark)

    # Save backtest results for predictor to use
    if include_ensembles:
        save_backtest_results(current_results, args.benchmark)

    # Run rolling validation (unless quick mode)
    rolling_results = {}
    if not args.quick:
        rolling_results = run_rolling_validation(
            predictor,
            n_seasons=args.seasons,
            include_models=include_models,
            include_ensembles=include_ensembles
        )
        print_results(rolling_results, "ROLLING VALIDATION RESULTS", args.benchmark)

    # Print summary
    print_summary(current_results, rolling_results, args.benchmark)


if __name__ == "__main__":
    main()
