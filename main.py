#!/usr/bin/env python3
"""
Unified Bundesliga Prediction Pipeline

Runs the complete prediction workflow:
1. Fetch latest data
2. Run backtest to find best model/ensemble
3. Generate predictions using the best performer
4. Submit to Kicktipp after user confirmation

Usage:
    uv run python main.py
    uv run python main.py --dry-run    # Skip actual submission
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

import pandas as pd
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# Set environment to reduce sklearn verbosity
os.environ['PYTHONWARNINGS'] = 'ignore'


def fetch_data_quiet():
    """Fetch data with minimal output."""
    import requests
    import pandas as pd
    import io
    from dotenv import load_dotenv

    load_dotenv()

    FOOTBALL_API_KEY = os.getenv('FOOTBALL_API_KEY', 'DEMO')
    ODDS_API_KEY = os.getenv('ODDS_API_KEY', '')
    BASE_URL = 'https://api.football-data.org/v4'
    ODDS_BASE_URL = 'https://api.the-odds-api.com/v4'
    BUNDESLIGA_ID = 'BL1'

    current_year = datetime.now().year
    current_month = datetime.now().month
    current_season = current_year if current_month >= 8 else current_year - 1

    tasks = [
        'Fetching match data',
        'Fetching historical odds',
        'Fetching current odds',
        'Saving cache files'
    ]

    pbar = tqdm(tasks, desc="Data fetch", ncols=60, leave=True)

    # Task 1: Fetch match data from API
    pbar.set_description("Fetching matches")
    headers = {'X-Auth-Token': FOOTBALL_API_KEY} if FOOTBALL_API_KEY != 'DEMO' else {}
    seasons_data = []
    for i in range(7, -1, -1):
        season = current_season - i
        try:
            url = f"{BASE_URL}/competitions/{BUNDESLIGA_ID}/matches?season={season}"
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200 and 'matches' in resp.json():
                seasons_data.extend(resp.json()['matches'])
        except:
            pass
    pbar.update(1)

    # Task 2: Fetch historical odds from football-data.co.uk
    pbar.set_description("Fetching hist. odds")
    all_odds = []
    for i in range(7, -1, -1):
        season_start = current_season - i
        season_code = f"{str(season_start)[-2:]}{str(season_start + 1)[-2:]}"
        url = f"https://www.football-data.co.uk/mmz4281/{season_code}/D1.csv"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                df = pd.read_csv(io.StringIO(resp.text))
                df['Season'] = f"{season_start}/{season_start + 1}"
                all_odds.append(df)
        except:
            pass
    pbar.update(1)

    # Task 3: Fetch current odds
    pbar.set_description("Fetching curr. odds")
    current_odds = None
    if ODDS_API_KEY:
        try:
            url = f"{ODDS_BASE_URL}/sports/soccer_germany_bundesliga/odds/"
            params = {'apiKey': ODDS_API_KEY, 'regions': 'eu', 'markets': 'h2h', 'oddsFormat': 'decimal'}
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                current_odds = resp.json()
        except:
            pass
    pbar.update(1)

    # Task 4: Save to cache
    pbar.set_description("Saving cache")
    if seasons_data:
        with open('bundesliga_matches.json', 'w') as f:
            json.dump({'fetched_at': datetime.now().isoformat(), 'total_matches': len(seasons_data), 'matches': seasons_data}, f, indent=2)

    if all_odds:
        combined_df = pd.concat(all_odds, ignore_index=True)
        combined_df.to_csv('bundesliga_historical_odds.csv', index=False)

    if current_odds:
        with open('bundesliga_odds.json', 'w') as f:
            json.dump({'fetched_at': datetime.now().isoformat(), 'total_matches': len(current_odds), 'odds': current_odds}, f, indent=2)
    pbar.update(1)

    pbar.close()
    return len(seasons_data), len(all_odds) if all_odds else 0


def run_backtest_quiet(training_seasons=2):
    """Run quick backtest with minimal output."""
    from bundesliga_predictor import BundesligaPredictor
    from bundesliga_predictor.scoring import calculate_kicktipp_points
    from bundesliga_predictor.features import FeatureExtractor
    from bundesliga_predictor.models import (
        MultiOutputRegressionModel, MultiClassClassificationModel,
        PoissonRegressionModel, NaiveOddsModel
    )
    from bundesliga_predictor.models.experimental import (
        GradientBoostingModel, BivariatePoissonModel, SmartOddsModel,
        TendencyFirstModel, ProbabilityMaxModel
    )
    from bundesliga_predictor.ensemble import ConsensusEnsemble
    from bundesliga_predictor.ensemble_v2 import (
        SimpleTendencyEnsemble, TendencyExpertEnsemble,
        TendencyConsensusEnsemble, HybridEnsembleV2
    )
    from bundesliga_predictor.ensembles.experimental import (
        OptimizedConsensusEnsemble, HybridEnsemble, AdaptiveScorelineEnsemble,
        BayesianOptimalEnsemble, AggressiveScorelineEnsemble,
        UltimateTendencyEnsemble, SuperConsensusEnsemble, MaxPointsEnsemble
    )

    # Suppress output during data loading
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        predictor = BundesligaPredictor(verbose=False)
        predictor.load_data()

    df = predictor._df.copy()

    if 'Season' in df.columns:
        seasons = sorted(df['Season'].unique())
        current_season_name = seasons[-1]
        current_season = df[df['Season'] == current_season_name].copy()
        training_season_names = seasons[-(training_seasons + 1):-1]
        historical = df[df['Season'].isin(training_season_names)].copy()
    else:
        return {}, {}

    # Create models and ensembles
    models = {
        'model1': MultiOutputRegressionModel(),
        'model2': MultiClassClassificationModel(),
        'model3': PoissonRegressionModel(),
        'model4': NaiveOddsModel(),
        'gradient_boosting': GradientBoostingModel(),
        'bivariate_poisson': BivariatePoissonModel(),
        'smart_odds': SmartOddsModel(),
        'tendency_first': TendencyFirstModel(),
        'probability_max': ProbabilityMaxModel(),
    }

    ensembles = {
        'consensus': ConsensusEnsemble(),
        'simple_tendency': SimpleTendencyEnsemble(),
        'tendency_expert': TendencyExpertEnsemble(),
        'tendency_consensus': TendencyConsensusEnsemble(),
        'hybrid_v2': HybridEnsembleV2(),
        'optimized_consensus': OptimizedConsensusEnsemble(),
        'hybrid': HybridEnsemble(),
        'adaptive_scoreline': AdaptiveScorelineEnsemble(),
        'bayesian_optimal': BayesianOptimalEnsemble(),
        'aggressive_scoreline': AggressiveScorelineEnsemble(),
        'ultimate_tendency': UltimateTendencyEnsemble(),
        'super_consensus': SuperConsensusEnsemble(),
        'max_points': MaxPointsEnsemble(),
    }

    results = {k: {'total': 0, 'n': 0} for k in list(models.keys()) + [f"ens_{e}" for e in ensembles.keys()]}

    current_season = current_season.sort_values('date').reset_index(drop=True)
    current_season['matchday_group'] = (current_season['date'].diff().dt.days.fillna(0) > 3).cumsum()
    matchday_groups = current_season.groupby('matchday_group')

    training_data = historical.copy()

    pbar = tqdm(matchday_groups, desc="Backtest", ncols=60, leave=True)

    for _, matchday_matches in pbar:
        feature_extractor = FeatureExtractor(training_data)
        X_train, y_home, y_away, scorelines = feature_extractor.extract_training_features(training_data)

        # Train models
        for model_id, model in models.items():
            try:
                if model_id in ['model4', 'smart_odds']:
                    model.train(X_train, y_home, y_away)
                else:
                    model.train(X_train, y_home, y_away, scorelines)
            except:
                pass

        # Evaluate on each match
        for _, match in matchday_matches.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            actual_home = int(match['home_score'])
            actual_away = int(match['away_score'])
            odds_home = match.get('odds_home')
            odds_draw = match.get('odds_draw')
            odds_away = match.get('odds_away')

            try:
                X = feature_extractor.extract_match_features(home_team, away_team, match['date'], odds_home, odds_draw, odds_away)
            except:
                continue

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
                    results[model_id]['total'] += pts
                    results[model_id]['n'] += 1
                    if model_id == 'model3':
                        try:
                            xg_home, xg_away = model.get_expected_goals(X)
                        except:
                            pass
                except:
                    pass

            core_preds = {k: v for k, v in model_preds.items() if k in ['model1', 'model2', 'model3', 'model4']}
            if len(core_preds) < 4:
                continue

            for ens_id, ensemble in ensembles.items():
                try:
                    try:
                        ens_pred, _, _ = ensemble.combine(core_preds, odds_home=odds_home, odds_draw=odds_draw, odds_away=odds_away, xg_home=xg_home, xg_away=xg_away)
                    except TypeError:
                        try:
                            ens_pred, _, _ = ensemble.combine(core_preds, odds_home=odds_home, odds_draw=odds_draw, odds_away=odds_away)
                        except TypeError:
                            ens_pred, _, _ = ensemble.combine(core_preds)
                    pts = calculate_kicktipp_points(ens_pred[0], ens_pred[1], actual_home, actual_away)
                    results[f"ens_{ens_id}"]['total'] += pts
                    results[f"ens_{ens_id}"]['n'] += 1
                except:
                    pass

        training_data = pd.concat([training_data, matchday_matches], ignore_index=True)

    pbar.close()

    # Separate and calculate averages
    model_results = {}
    ensemble_results = {}

    for k, v in results.items():
        if v['n'] > 0:
            avg = v['total'] / v['n']
            if k.startswith('ens_'):
                ensemble_results[k] = avg
            else:
                model_results[k] = avg

    return model_results, ensemble_results, results  # Also return raw results for saving


def save_backtest_results(results, benchmark=1.65):
    """Save backtest results to JSON file for predictor to use."""
    model_results = {k: v for k, v in results.items() if not k.startswith('ens_')}
    ensemble_results = {k: v for k, v in results.items() if k.startswith('ens_')}

    MODEL_NAMES = {
        'model1': 'Multi-Output Regression',
        'model2': 'Multi-Class Classification',
        'model3': 'Poisson Regression',
        'model4': 'Naive Odds',
        'gradient_boosting': 'Gradient Boosting',
        'bivariate_poisson': 'Bivariate Poisson',
        'smart_odds': 'Smart Odds',
        'tendency_first': 'Tendency First',
        'probability_max': 'Probability Max',
    }

    ENSEMBLE_CLASS_MAP = {
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

    # Calculate scores
    model_scores = {}
    for name, r in model_results.items():
        if r['n'] > 0:
            model_scores[name] = {
                'avg_points': r['total'] / r['n'],
                'total_points': r['total'],
                'matches': r['n'],
            }

    ensemble_scores = {}
    for name, r in ensemble_results.items():
        if r['n'] > 0:
            ensemble_scores[name] = {
                'avg_points': r['total'] / r['n'],
                'total_points': r['total'],
                'matches': r['n'],
            }

    # Find best
    best_model = max(model_scores.keys(), key=lambda k: model_scores[k]['avg_points']) if model_scores else None
    best_model_score = model_scores[best_model]['avg_points'] if best_model else 0

    best_ensemble = max(ensemble_scores.keys(), key=lambda k: ensemble_scores[k]['avg_points']) if ensemble_scores else None
    best_ensemble_score = ensemble_scores[best_ensemble]['avg_points'] if best_ensemble else 0

    results_data = {
        'updated_at': datetime.now().isoformat(),
        'benchmark': benchmark,
        'best_model': {
            'key': best_model,
            'display_name': MODEL_NAMES.get(best_model, best_model),
            'avg_points': best_model_score,
            'beats_benchmark': best_model_score >= benchmark,
        } if best_model else None,
        'best_ensemble': {
            'key': best_ensemble,
            'class_name': ENSEMBLE_CLASS_MAP.get(best_ensemble, 'HybridEnsembleV2'),
            'display_name': best_ensemble.replace('ens_', '').replace('_', ' ').title() if best_ensemble else None,
            'avg_points': best_ensemble_score,
            'beats_benchmark': best_ensemble_score >= benchmark,
        } if best_ensemble else None,
        'all_models': model_scores,
        'all_ensembles': ensemble_scores,
    }

    with open('backtest_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)


def run_predictions_quiet():
    """Run predictions with minimal output."""
    from bundesliga_predictor import BundesligaPredictor

    pbar = tqdm(['Loading', 'Training', 'Predicting', 'Saving'], desc="Predict", ncols=60, leave=True)

    pbar.set_description("Loading data")
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        predictor = BundesligaPredictor(verbose=False)
        predictor.load_data()
    pbar.update(1)

    pbar.set_description("Training models")
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        predictor.train_models()
    pbar.update(1)

    pbar.set_description("Predicting")
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        predictions = predictor.predict_next_matchday()
    pbar.update(1)

    pbar.set_description("Saving results")
    best_model_key = predictor._best_model_key
    best_model_name = predictor._best_model_name
    best_model_score = predictor._best_model_score
    best_ensemble_name = predictor._ensemble_display_name
    best_ensemble_score = predictor._ensemble_score

    save_data = {
        'generated_at': datetime.now().isoformat(),
        'matchday': predictions[0].get('matchday', '?') if predictions else '?',
        'best_model': {
            'name': best_model_name,
            'avg_points': best_model_score,
            'predictions': [
                {'home_team': p['home_team'], 'away_team': p['away_team'], 'date': p['date'],
                 'home_score': p['predictions'][best_model_key]['home'],
                 'away_score': p['predictions'][best_model_key]['away'],
                 'scoreline': p['predictions'][best_model_key]['scoreline']}
                for p in predictions
            ]
        },
        'best_ensemble': {
            'name': best_ensemble_name,
            'avg_points': best_ensemble_score,
            'predictions': [
                {'home_team': p['home_team'], 'away_team': p['away_team'], 'date': p['date'],
                 'home_score': p['ensemble']['home'], 'away_score': p['ensemble']['away'],
                 'scoreline': p['ensemble']['scoreline'], 'strategy': p['ensemble']['strategy']}
                for p in predictions
            ]
        },
        'odds': [{'home_team': p['home_team'], 'away_team': p['away_team'], 'odds': p['odds']} for p in predictions],
    }

    with open('latest_predictions.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    pbar.update(1)

    pbar.close()
    return save_data


def print_results_table(model_results, ensemble_results):
    """Print combined results table sorted by performance."""
    MODEL_NAMES = {
        'model1': 'Multi-Output Regression',
        'model2': 'Multi-Class Classification',
        'model3': 'Poisson Regression',
        'model4': 'Naive Odds',
        'gradient_boosting': 'Gradient Boosting',
        'bivariate_poisson': 'Bivariate Poisson',
        'smart_odds': 'Smart Odds',
        'tendency_first': 'Tendency First',
        'probability_max': 'Probability Max',
    }

    all_results = []
    for k, v in model_results.items():
        all_results.append((MODEL_NAMES.get(k, k), v, 'Model', k))
    for k, v in ensemble_results.items():
        name = k.replace('ens_', '').replace('_', ' ').title()
        all_results.append((name, v, 'Ensemble', k))

    all_results.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 56)
    print(f"{'Name':<30} {'Pts/Match':>10} {'Type':>12}")
    print("=" * 56)

    for name, avg, type_, key in all_results:
        marker = " *" if avg >= 1.65 else ""
        print(f"{name[:29]:<30} {avg:>10.2f} {type_:>12}{marker}")

    print("-" * 56)
    print("* = beats benchmark (1.65 pts/match)")

    return all_results


def submit_to_kicktipp(predictions_data, use_model, dry_run=False):
    """Submit predictions to Kicktipp."""
    from bundesliga_predictor.kicktipp import KicktippClient

    if use_model:
        source = predictions_data['best_model']
    else:
        source = predictions_data['best_ensemble']

    predictions = source['predictions']

    pbar = tqdm(['Login', 'Submit'], desc="Kicktipp", ncols=60, leave=True)

    try:
        client = KicktippClient()
        pbar.set_description("Logging in")
        client.login()
        pbar.update(1)

        pred_for_kicktipp = [
            {'home_team': p['home_team'], 'away_team': p['away_team'],
             'ensemble': {'home': p['home_score'], 'away': p['away_score']}}
            for p in predictions
        ]

        pbar.set_description("Submitting")
        results = client.submit_from_predictor_results(pred_for_kicktipp, overwrite=True, dry_run=dry_run)
        pbar.update(1)
        pbar.close()

        success_count = sum(1 for success, _ in results.values() if success)
        return success_count, len(results)

    except Exception as e:
        pbar.close()
        raise RuntimeError(str(e))


def main():
    parser = argparse.ArgumentParser(description='Unified Bundesliga prediction pipeline')
    parser.add_argument('--dry-run', action='store_true', help='Skip actual submission')
    parser.add_argument('--skip-fetch', action='store_true', help='Skip data fetching')
    parser.add_argument('--skip-backtest', action='store_true', help='Skip backtesting')
    args = parser.parse_args()

    print("\n" + "=" * 56)
    print("        BUNDESLIGA PREDICTION PIPELINE")
    print("=" * 56 + "\n")

    # Step 1: Fetch data
    if not args.skip_fetch:
        matches, odds = fetch_data_quiet()
        print(f"  -> {matches} matches, {odds} seasons of odds\n")

    # Step 2: Run backtest
    if not args.skip_backtest:
        model_results, ensemble_results, raw_results = run_backtest_quiet()
        all_results = print_results_table(model_results, ensemble_results)

        # Save results so predictor uses fresh data
        save_backtest_results(raw_results)

        # Find overall best
        best_name, best_avg, best_type, best_key = all_results[0]
        use_model = (best_type == 'Model')
        print(f"\nBest overall: {best_name} ({best_avg:.2f} pts/match)")
    else:
        # Load from saved backtest results
        with open('backtest_results.json', 'r') as f:
            backtest_data = json.load(f)

        best_model_avg = backtest_data['best_model']['avg_points'] if backtest_data['best_model'] else 0
        best_ens_avg = backtest_data['best_ensemble']['avg_points'] if backtest_data['best_ensemble'] else 0

        if best_model_avg > best_ens_avg:
            best_name = backtest_data['best_model']['display_name']
            best_avg = best_model_avg
            use_model = True
        else:
            best_name = backtest_data['best_ensemble']['display_name']
            best_avg = best_ens_avg
            use_model = False

        print(f"\nBest (from cache): {best_name} ({best_avg:.2f} pts/match)")

    # Step 3: Generate predictions
    print()
    predictions_data = run_predictions_quiet()

    # Step 4: Show predictions
    matchday = predictions_data['matchday']
    if use_model:
        source = predictions_data['best_model']
        source_type = 'model'
    else:
        source = predictions_data['best_ensemble']
        source_type = 'ensemble'

    print(f"\n" + "=" * 56)
    print(f"  PREDICTIONS FOR MATCHDAY {matchday}")
    print(f"  Using: {source['name']} ({source_type}, {source['avg_points']:.2f} pts/match)")
    print("=" * 56)

    for pred in source['predictions']:
        home = pred['home_team'][:22]
        away = pred['away_team'][:22]
        score = pred['scoreline']
        print(f"  {home:<22} vs {away:<22}  {score}")

    print("=" * 56)

    # Step 5: Wait for confirmation
    print("\nPress ENTER to submit to Kicktipp" + (" (dry-run)" if args.dry_run else "") + ", or Ctrl+C to cancel...")

    try:
        input()
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 0

    # Step 6: Submit
    print()
    try:
        success, total = submit_to_kicktipp(predictions_data, use_model, dry_run=args.dry_run)
        print(f"\n{'Would submit' if args.dry_run else 'Submitted'}: {success}/{total} predictions")
    except RuntimeError as e:
        print(f"\nSubmission failed: {e}")
        return 1
    except ValueError as e:
        print(f"\nConfiguration error: {e}")
        print("Set KICKTIPP_EMAIL, KICKTIPP_PASSWORD, KICKTIPP_COMMUNITY in .env")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
