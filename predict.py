#!/usr/bin/env python3
"""
Weekly Bundesliga Match Predictor

Run this script every week to get predictions for the next matchday.

Usage:
    python predict.py                  # Full prediction with training
    python predict.py --quick          # Quick prediction (skip training output)
    python predict.py --json           # Output predictions as JSON

Prerequisites:
    Run 'python fetch_data.py' first to download match data.
"""

import argparse
import json
import os
import sys
from datetime import datetime

from bundesliga_predictor import BundesligaPredictor
from bundesliga_predictor.config import CACHE_DIR

# Standard predictions file
PREDICTIONS_FILE = os.path.join(CACHE_DIR, 'latest_predictions.json')


def main():
    parser = argparse.ArgumentParser(
        description='Bundesliga Match Predictor - Get predictions for the next matchday'
    )
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick mode - minimal output'
    )
    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output predictions as JSON'
    )
    parser.add_argument(
        '--save', '-s',
        type=str,
        default=None,
        help='Save predictions to file'
    )

    args = parser.parse_args()

    try:
        # Initialize predictor
        predictor = BundesligaPredictor(verbose=not args.quick)

        # Run prediction pipeline
        if args.json:
            # Quiet mode for JSON output
            predictor.verbose = False
            predictor.load_data()
            predictor.train_models()
            predictions = predictor.predict_next_matchday()

            # Convert to JSON-serializable format
            output = {
                'generated_at': __import__('datetime').datetime.now().isoformat(),
                'matchday': predictions[0]['matchday'] if predictions else None,
                'predictions': []
            }

            for pred in predictions:
                output['predictions'].append({
                    'home_team': pred['home_team'],
                    'away_team': pred['away_team'],
                    'date': pred['date'],
                    'ensemble_prediction': pred['ensemble']['scoreline'],
                    'strategy': pred['ensemble']['strategy'],
                    'all_predictions': {
                        model_id: p['scoreline']
                        for model_id, p in pred['predictions'].items()
                    },
                    'odds': pred['odds'],
                })

            print(json.dumps(output, indent=2))

        else:
            # Normal mode with formatted output
            predictions = predictor.run()

        # Always save predictions to standard location
        if predictions:
            # Get best model key from predictor
            best_model_key = predictor._best_model_key
            best_model_name = predictor._best_model_name
            best_model_score = predictor._best_model_score
            best_ensemble_name = predictor._ensemble_display_name
            best_ensemble_score = predictor._ensemble_score

            save_data = {
                'generated_at': datetime.now().isoformat(),
                'matchday': predictions[0].get('matchday', '?'),
                'best_model': {
                    'name': best_model_name,
                    'avg_points': best_model_score,
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
                    'name': best_ensemble_name,
                    'avg_points': best_ensemble_score,
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
                'odds': [
                    {
                        'home_team': p['home_team'],
                        'away_team': p['away_team'],
                        'odds': p['odds'],
                    }
                    for p in predictions
                ],
            }
            with open(PREDICTIONS_FILE, 'w') as f:
                json.dump(save_data, f, indent=2)
            if not args.json:
                print(f"\nPredictions saved to {PREDICTIONS_FILE}")

        # Also save to custom file if requested
        if args.save and predictions:
            with open(args.save, 'w') as f:
                json.dump(save_data, f, indent=2)
            if not args.json:
                print(f"Also saved to {args.save}")

        return 0

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease run 'python fetch_data.py' first to download match data.")
        return 1

    except Exception as e:
        print(f"\n❌ Error: {e}")
        if not args.quick:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
