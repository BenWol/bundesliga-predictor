"""
Unified pipeline functions for data fetching, backtesting, prediction, and submission.

Usage:
    from bundesliga_predictor.pipeline import fetch_data, run_backtest, run_predict, submit_kicktipp
"""

import io
import json
import os
import warnings
from datetime import datetime

import pandas as pd
import requests
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Model and ensemble name mappings
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


def fetch_data(verbose=False):
    """
    Fetch matches and odds data from APIs.

    Returns:
        tuple: (num_matches, num_seasons_of_odds)
    """
    from dotenv import load_dotenv
    load_dotenv()

    football_api_key = os.getenv('FOOTBALL_API_KEY', 'DEMO')
    odds_api_key = os.getenv('ODDS_API_KEY', '')

    now = datetime.now()
    current_season = now.year if now.month >= 8 else now.year - 1

    steps = ['matches', 'hist_odds', 'curr_odds', 'save']
    pbar = tqdm(steps, desc="Fetch", ncols=60, disable=verbose)

    # Fetch matches from football-data.org
    pbar.set_description("Matches")
    headers = {'X-Auth-Token': football_api_key} if football_api_key != 'DEMO' else {}
    matches = []
    for i in range(7, -1, -1):
        season = current_season - i
        try:
            url = f"https://api.football-data.org/v4/competitions/BL1/matches?season={season}"
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.ok and 'matches' in resp.json():
                matches.extend(resp.json()['matches'])
        except Exception:
            pass
    pbar.update(1)

    # Fetch historical odds from football-data.co.uk
    pbar.set_description("Hist odds")
    odds_dfs = []
    for i in range(7, -1, -1):
        s = current_season - i
        url = f"https://www.football-data.co.uk/mmz4281/{str(s)[-2:]}{str(s+1)[-2:]}/D1.csv"
        try:
            resp = requests.get(url, timeout=30)
            if resp.ok:
                df = pd.read_csv(io.StringIO(resp.text))
                df['Season'] = f"{s}/{s+1}"
                odds_dfs.append(df)
        except Exception:
            pass
    pbar.update(1)

    # Fetch current odds from the-odds-api.com
    pbar.set_description("Curr odds")
    curr_odds = None
    if odds_api_key:
        try:
            url = "https://api.the-odds-api.com/v4/sports/soccer_germany_bundesliga/odds/"
            resp = requests.get(
                url,
                params={'apiKey': odds_api_key, 'regions': 'eu', 'markets': 'h2h', 'oddsFormat': 'decimal'},
                timeout=30
            )
            if resp.ok:
                curr_odds = resp.json()
        except Exception:
            pass
    pbar.update(1)

    # Save data
    pbar.set_description("Save")
    if matches:
        with open('bundesliga_matches.json', 'w') as f:
            json.dump({'fetched_at': datetime.now().isoformat(), 'matches': matches}, f)
    if odds_dfs:
        pd.concat(odds_dfs, ignore_index=True).to_csv('bundesliga_historical_odds.csv', index=False)
    if curr_odds:
        with open('bundesliga_odds.json', 'w') as f:
            json.dump({'fetched_at': datetime.now().isoformat(), 'odds': curr_odds}, f)
    pbar.update(1)
    pbar.close()

    if verbose:
        print(f"Fetched {len(matches)} matches, {len(odds_dfs)} seasons of odds")

    return len(matches), len(odds_dfs)


def _backtest_matchday(args):
    """
    Backtest a single matchday (helper for parallel execution).

    Args:
        args: tuple of (training_data, matchday_matches)

    Returns:
        dict: Results with total points and count per model/ensemble
    """
    from .scoring import calculate_kicktipp_points
    from .features import FeatureExtractor
    from .models import (
        MultiOutputRegressionModel, MultiClassClassificationModel,
        PoissonRegressionModel, NaiveOddsModel
    )
    from .models.experimental import (
        GradientBoostingModel, BivariatePoissonModel, SmartOddsModel,
        TendencyFirstModel, ProbabilityMaxModel
    )
    from .ensemble import ConsensusEnsemble
    from .ensemble_v2 import (
        SimpleTendencyEnsemble, TendencyExpertEnsemble,
        TendencyConsensusEnsemble, HybridEnsembleV2
    )
    from .ensembles.experimental import (
        OptimizedConsensusEnsemble, HybridEnsemble, AdaptiveScorelineEnsemble,
        BayesianOptimalEnsemble, AggressiveScorelineEnsemble,
        UltimateTendencyEnsemble, SuperConsensusEnsemble, MaxPointsEnsemble
    )

    training_data, matchday_matches = args

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

    # Train on all data up to this matchday
    feature_extractor = FeatureExtractor(training_data)
    X_train, y_home, y_away, scorelines = feature_extractor.extract_training_features(training_data)

    for model_id, model in models.items():
        try:
            if model_id in ['model4', 'smart_odds']:
                model.train(X_train, y_home, y_away)
            else:
                model.train(X_train, y_home, y_away, scorelines)
        except Exception:
            pass

    # Evaluate each match
    results = {k: {'total': 0, 'n': 0} for k in list(models.keys()) + [f"ens_{e}" for e in ensembles.keys()]}

    for _, match in matchday_matches.iterrows():
        home_team, away_team = match['home_team'], match['away_team']
        actual_home, actual_away = int(match['home_score']), int(match['away_score'])
        odds_home = match.get('odds_home')
        odds_draw = match.get('odds_draw')
        odds_away = match.get('odds_away')

        try:
            X = feature_extractor.extract_match_features(home_team, away_team, match['date'], odds_home, odds_draw, odds_away)
        except Exception:
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
                    except Exception:
                        pass
            except Exception:
                pass

        core_preds = {k: v for k, v in model_preds.items() if k in ['model1', 'model2', 'model3', 'model4']}
        if len(core_preds) < 4:
            continue

        for ens_id, ensemble in ensembles.items():
            try:
                try:
                    ens_pred, _, _ = ensemble.combine(
                        core_preds,
                        odds_home=odds_home, odds_draw=odds_draw, odds_away=odds_away,
                        xg_home=xg_home, xg_away=xg_away
                    )
                except TypeError:
                    try:
                        ens_pred, _, _ = ensemble.combine(
                            core_preds,
                            odds_home=odds_home, odds_draw=odds_draw, odds_away=odds_away
                        )
                    except TypeError:
                        ens_pred, _, _ = ensemble.combine(core_preds)
                pts = calculate_kicktipp_points(ens_pred[0], ens_pred[1], actual_home, actual_away)
                results[f"ens_{ens_id}"]['total'] += pts
                results[f"ens_{ens_id}"]['n'] += 1
            except Exception:
                pass

    return results


def _save_backtest_results(results, benchmark=1.65):
    """Save backtest results to JSON file."""
    model_scores = {
        k: {'avg_points': v['total'] / v['n'], 'total_points': v['total'], 'matches': v['n']}
        for k, v in results.items() if not k.startswith('ens_') and v['n'] > 0
    }
    ensemble_scores = {
        k: {'avg_points': v['total'] / v['n'], 'total_points': v['total'], 'matches': v['n']}
        for k, v in results.items() if k.startswith('ens_') and v['n'] > 0
    }

    best_model = max(model_scores, key=lambda k: model_scores[k]['avg_points']) if model_scores else None
    best_ensemble = max(ensemble_scores, key=lambda k: ensemble_scores[k]['avg_points']) if ensemble_scores else None

    data = {
        'updated_at': datetime.now().isoformat(),
        'benchmark': benchmark,
        'best_model': {
            'key': best_model,
            'display_name': MODEL_NAMES.get(best_model, best_model),
            'avg_points': model_scores[best_model]['avg_points'],
            'beats_benchmark': model_scores[best_model]['avg_points'] >= benchmark,
        } if best_model else None,
        'best_ensemble': {
            'key': best_ensemble,
            'class_name': ENSEMBLE_CLASS_MAP.get(best_ensemble, 'HybridEnsembleV2'),
            'display_name': best_ensemble.replace('ens_', '').replace('_', ' ').title() if best_ensemble else None,
            'avg_points': ensemble_scores[best_ensemble]['avg_points'],
            'beats_benchmark': ensemble_scores[best_ensemble]['avg_points'] >= benchmark,
        } if best_ensemble else None,
        'all_models': model_scores,
        'all_ensembles': ensemble_scores,
    }

    with open('backtest_results.json', 'w') as f:
        json.dump(data, f, indent=2)


def run_backtest(verbose=False, training_seasons=2, n_jobs=-1):
    """
    Walk-forward backtest with parallel execution across matchdays.

    Each matchday trains on all prior data and predicts that matchday,
    simulating real-world usage.

    Args:
        verbose: Show detailed output instead of progress bars
        training_seasons: Number of historical seasons for initial training
        n_jobs: Number of parallel jobs (-1 = all cores)

    Returns:
        dict: Results with total points and count per model/ensemble
    """
    from joblib import Parallel, delayed
    from . import BundesligaPredictor

    # Load data
    predictor = BundesligaPredictor(verbose=False)
    predictor.load_data()
    df = predictor._df.copy()

    # Split: historical seasons for initial training, current season for testing
    seasons = sorted(df['Season'].unique())
    current_season = seasons[-1]
    train_seasons = seasons[-(training_seasons + 1):-1]

    historical = df[df['Season'].isin(train_seasons)].copy()
    current = df[df['Season'] == current_season].copy().sort_values('date')

    # Group current season by matchday
    current['matchday_group'] = (current['date'].diff().dt.days.fillna(0) > 3).cumsum()
    matchdays = list(current.groupby('matchday_group'))

    if verbose:
        print(f"Training base: {len(historical)} matches from {train_seasons}")
        print(f"Testing: {len(current)} matches across {len(matchdays)} matchdays (parallel)")

    # Pre-compute training data for each matchday
    tasks = []
    cumulative_data = historical.copy()
    for _, matchday_matches in matchdays:
        tasks.append((cumulative_data.copy(), matchday_matches.copy()))
        cumulative_data = pd.concat([cumulative_data, matchday_matches], ignore_index=True)

    # Run in parallel with progress tracking
    if verbose:
        matchday_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_backtest_matchday)(task) for task in tasks
        )
    else:
        # Use tqdm with joblib's return_as='generator' for real progress
        with tqdm(total=len(tasks), desc="Backtest", ncols=60) as pbar:
            matchday_results = []
            for result in Parallel(n_jobs=n_jobs, return_as='generator')(
                delayed(_backtest_matchday)(task) for task in tasks
            ):
                matchday_results.append(result)
                pbar.update(1)

    # Aggregate results
    final_results = {}
    for md_result in matchday_results:
        for k, v in md_result.items():
            if k not in final_results:
                final_results[k] = {'total': 0, 'n': 0}
            final_results[k]['total'] += v['total']
            final_results[k]['n'] += v['n']

    _save_backtest_results(final_results)
    return final_results


def run_predict(verbose=False):
    """
    Generate predictions for the next matchday.

    Returns:
        dict: Prediction data with best model and ensemble predictions
    """
    from . import BundesligaPredictor

    pbar = tqdm(['Load', 'Train', 'Predict', 'Save'], desc="Predict", ncols=60, disable=verbose)

    pbar.set_description("Load")
    predictor = BundesligaPredictor(verbose=False)
    predictor.load_data()
    pbar.update(1)

    pbar.set_description("Train")
    predictor.train_models()
    pbar.update(1)

    pbar.set_description("Predict")
    predictions = predictor.predict_next_matchday()
    pbar.update(1)

    pbar.set_description("Save")
    best_model_key = predictor._best_model_key

    # Get current odds for the predictions
    odds_data = []
    for p in predictions:
        odds_data.append({
            'home_team': p['home_team'],
            'away_team': p['away_team'],
            'odds': {
                'home': p.get('odds_home'),
                'draw': p.get('odds_draw'),
                'away': p.get('odds_away'),
            }
        })

    save_data = {
        'generated_at': datetime.now().isoformat(),
        'matchday': predictions[0].get('matchday', '?') if predictions else '?',
        'best_model': {
            'name': predictor._best_model_name,
            'avg_points': predictor._best_model_score,
            'predictions': [
                {
                    'home_team': p['home_team'],
                    'away_team': p['away_team'],
                    'date': p['date'],
                    'home_score': p['predictions'][best_model_key]['home'],
                    'away_score': p['predictions'][best_model_key]['away'],
                    'scoreline': p['predictions'][best_model_key]['scoreline'],
                }
                for p in predictions
            ]
        },
        'best_ensemble': {
            'name': predictor._ensemble_display_name,
            'avg_points': predictor._ensemble_score,
            'predictions': [
                {
                    'home_team': p['home_team'],
                    'away_team': p['away_team'],
                    'date': p['date'],
                    'home_score': p['ensemble']['home'],
                    'away_score': p['ensemble']['away'],
                    'scoreline': p['ensemble']['scoreline'],
                    'strategy': p['ensemble']['strategy'],
                }
                for p in predictions
            ]
        },
        'odds': odds_data,
    }

    with open('latest_predictions.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    pbar.update(1)
    pbar.close()

    return save_data


def submit_kicktipp(predictions_data, use_model, dry_run=False, verbose=False):
    """
    Submit predictions to Kicktipp.

    Args:
        predictions_data: Dict with 'best_model' and 'best_ensemble' predictions
        use_model: If True, use best_model predictions; else use best_ensemble
        dry_run: If True, don't actually submit
        verbose: Show detailed output

    Returns:
        tuple: (successful_count, total_count)
    """
    from .kicktipp import KicktippClient

    source = predictions_data['best_model'] if use_model else predictions_data['best_ensemble']

    pbar = tqdm(['Login', 'Submit'], desc="Submit", ncols=60, disable=verbose)

    client = KicktippClient()
    pbar.set_description("Login")
    client.login()
    pbar.update(1)

    pbar.set_description("Submit")
    pred_list = [
        {
            'home_team': p['home_team'],
            'away_team': p['away_team'],
            'ensemble': {'home': p['home_score'], 'away': p['away_score']}
        }
        for p in source['predictions']
    ]
    results = client.submit_from_predictor_results(pred_list, overwrite=True, dry_run=dry_run)
    pbar.update(1)
    pbar.close()

    return sum(1 for s, _ in results.values() if s), len(results)


def print_results_table(results):
    """
    Print a formatted results table sorted by performance.

    Args:
        results: Dict from run_backtest()

    Returns:
        list: Sorted list of (name, avg_points, type, key) tuples
    """
    all_results = []
    for k, v in results.items():
        if v['n'] > 0:
            avg = v['total'] / v['n']
            if k.startswith('ens_'):
                name = k.replace('ens_', '').replace('_', ' ').title()
                all_results.append((name, avg, 'Ensemble', k))
            else:
                all_results.append((MODEL_NAMES.get(k, k), avg, 'Model', k))

    all_results.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 56)
    print(f"{'Name':<30} {'Pts/Match':>10} {'Type':>12}")
    print("=" * 56)
    for name, avg, type_, _ in all_results:
        marker = " *" if avg >= 1.65 else ""
        print(f"{name[:29]:<30} {avg:>10.2f} {type_:>12}{marker}")
    print("-" * 56)
    print("* = beats benchmark (1.65 pts/match)")

    return all_results
