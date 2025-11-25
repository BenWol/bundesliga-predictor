#!/usr/bin/env python3
"""
Backtest ensemble strategy on current season.

Usage:
    uv run python backtest.py
"""

import pandas as pd
from datetime import datetime

from bundesliga_predictor import BundesligaPredictor
from bundesliga_predictor.scoring import calculate_kicktipp_points
from bundesliga_predictor.config import MODEL_NAMES


def main():
    predictor = BundesligaPredictor(verbose=False)
    predictor.load_data()

    # Get current season data
    now = datetime.now()
    if now.month >= 8:
        season_start = pd.Timestamp(f'{now.year}-08-01', tz='UTC')
    else:
        season_start = pd.Timestamp(f'{now.year-1}-08-01', tz='UTC')

    df = predictor._df
    current_season = df[df['date'] >= season_start].copy()

    print(f"\nBacktesting on {len(current_season)} matches from current season")
    print("=" * 60)

    # Train on historical data (before current season)
    historical = df[df['date'] < season_start].copy()
    predictor.train_models(historical)

    # Track points per model
    points = {model_id: 0 for model_id in predictor._models}
    points['ensemble'] = 0

    for _, match in current_season.iterrows():
        home_team = match['home_team']
        away_team = match['away_team']
        actual_home = match['home_score']
        actual_away = match['away_score']

        # Get features
        X = predictor._feature_extractor.extract_match_features(
            home_team, away_team, match['date'],
            match.get('odds_home'), match.get('odds_draw'), match.get('odds_away')
        )

        # Get predictions from all models
        model_preds = {}
        for model_id, model in predictor._models.items():
            if model_id == 'model4':
                pred = model.predict(X, odds_home=match.get('odds_home'),
                                    odds_draw=match.get('odds_draw'),
                                    odds_away=match.get('odds_away'))
            else:
                pred = model.predict(X)
            model_preds[model_id] = pred
            pts = calculate_kicktipp_points(pred[0], pred[1], actual_home, actual_away)
            points[model_id] += pts

        # Ensemble prediction
        ens_pred, _, _ = predictor._ensemble.combine(model_preds)
        pts = calculate_kicktipp_points(ens_pred[0], ens_pred[1], actual_home, actual_away)
        points['ensemble'] += pts

    # Print results
    n_matches = len(current_season)
    print(f"\nResults ({n_matches} matches):")
    print("-" * 40)

    results = []
    for model_id, total_pts in points.items():
        avg = total_pts / n_matches if n_matches > 0 else 0
        name = MODEL_NAMES.get(model_id, 'Ensemble')
        results.append((name, total_pts, avg))

    results.sort(key=lambda x: x[2], reverse=True)

    for name, total, avg in results:
        marker = " <-- BEST" if results[0][0] == name else ""
        print(f"{name:<30} {total:>4} pts  ({avg:.2f}/match){marker}")


if __name__ == "__main__":
    main()
