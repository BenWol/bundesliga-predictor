#!/usr/bin/env python3
"""
Unified Bundesliga Prediction Pipeline

Usage:
    uv run python main.py              # Full pipeline
    uv run python main.py --dry-run    # Skip submission
    uv run python main.py --verbose    # Show detailed output
"""

import argparse
import json
import sys
import warnings

warnings.filterwarnings('ignore')

from bundesliga_predictor import (
    fetch_data,
    run_backtest,
    run_predict,
    submit_kicktipp,
    print_results_table,
)


def main():
    parser = argparse.ArgumentParser(description='Bundesliga prediction pipeline')
    parser.add_argument('--dry-run', action='store_true', help='Skip actual submission')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    parser.add_argument('--skip-fetch', action='store_true', help='Skip data fetching')
    parser.add_argument('--skip-backtest', action='store_true', help='Skip backtesting')
    args = parser.parse_args()

    print("\n" + "=" * 56)
    print("        BUNDESLIGA PREDICTION PIPELINE")
    print("=" * 56 + "\n")

    # 1. Fetch data
    if not args.skip_fetch:
        fetch_data(verbose=args.verbose)
        print()

    # 2. Backtest
    if not args.skip_backtest:
        results = run_backtest(verbose=args.verbose)
        all_results = print_results_table(results)
        best_name, best_avg, best_type, best_key = all_results[0]
        use_model = (best_type == 'Model')
        print(f"\nBest: {best_name} ({best_avg:.2f} pts/match)")
    else:
        with open('backtest_results.json') as f:
            data = json.load(f)
        model_avg = data['best_model']['avg_points'] if data['best_model'] else 0
        ens_avg = data['best_ensemble']['avg_points'] if data['best_ensemble'] else 0
        if model_avg > ens_avg:
            best_name, best_avg, use_model = data['best_model']['display_name'], model_avg, True
        else:
            best_name, best_avg, use_model = data['best_ensemble']['display_name'], ens_avg, False
        print(f"Best (cached): {best_name} ({best_avg:.2f} pts/match)")

    # 3. Predict
    print()
    predictions = run_predict(verbose=args.verbose)

    # 4. Show predictions
    source = predictions['best_model'] if use_model else predictions['best_ensemble']
    source_type = 'model' if use_model else 'ensemble'

    print(f"\n" + "=" * 56)
    print(f"  MATCHDAY {predictions['matchday']} - {source['name']} ({source_type})")
    print("=" * 56)
    for p in source['predictions']:
        print(f"  {p['home_team'][:22]:<22} vs {p['away_team'][:22]:<22}  {p['scoreline']}")
    print("=" * 56)

    # 5. Confirm
    mode = " (dry-run)" if args.dry_run else ""
    print(f"\nPress ENTER to submit{mode}, Ctrl+C to cancel...")
    try:
        input()
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 0

    # 6. Submit
    print()
    try:
        ok, total = submit_kicktipp(predictions, use_model, args.dry_run, args.verbose)
        action = "Would submit" if args.dry_run else "Submitted"
        print(f"\n{action}: {ok}/{total} predictions")
    except Exception as e:
        print(f"\nSubmission failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
