#!/usr/bin/env python3
"""
Submit predictions to Kicktipp.de.

This script reads the latest predictions from predict.py and submits them.

Usage:
    uv run python submit_to_kicktipp.py           # Submit predictions
    uv run python submit_to_kicktipp.py --dry-run # Show what would be submitted
    uv run python submit_to_kicktipp.py --no-overwrite  # Don't overwrite existing tips

Prerequisites:
    1. Run 'python predict.py' first to generate predictions
    2. Set credentials in .env:
       KICKTIPP_EMAIL=your_email@example.com
       KICKTIPP_PASSWORD=your_password
       KICKTIPP_COMMUNITY=wolter
"""

import argparse
import json
import os
import sys
from datetime import datetime

from bundesliga_predictor.config import CACHE_DIR
from bundesliga_predictor.kicktipp import KicktippClient

# Standard predictions file (same as predict.py)
PREDICTIONS_FILE = os.path.join(CACHE_DIR, 'latest_predictions.json')


def load_predictions():
    """
    Load predictions from file.

    Returns:
        Tuple of (predictions_data, error_message)
    """
    if not os.path.exists(PREDICTIONS_FILE):
        return None, "No predictions file found."

    with open(PREDICTIONS_FILE, 'r') as f:
        data = json.load(f)

    # Check if predictions are from today
    generated_at = datetime.fromisoformat(data['generated_at'])
    today = datetime.now().date()

    if generated_at.date() != today:
        days_ago = (today - generated_at.date()).days
        return None, (
            f"Predictions are {days_ago} day(s) old (from {generated_at.strftime('%Y-%m-%d %H:%M')}).\n"
            f"Please run 'uv run python predict.py' first to generate fresh predictions."
        )

    return data, None


def main():
    parser = argparse.ArgumentParser(
        description='Submit predictions to Kicktipp.de'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be submitted without actually submitting'
    )
    parser.add_argument(
        '--no-overwrite',
        action='store_true',
        help='Do not overwrite existing predictions'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Submit even if predictions are not from today'
    )

    args = parser.parse_args()

    # Load predictions
    print("=" * 60)
    print("Loading predictions...")
    print("=" * 60)

    data, error = load_predictions()

    if error and not args.force:
        print(f"\n❌ {error}")
        sys.exit(1)
    elif error and args.force:
        print(f"\n⚠️  Warning: {error}")
        print("Continuing anyway due to --force flag.\n")
        # Re-load without date check
        with open(PREDICTIONS_FILE, 'r') as f:
            data = json.load(f)

    predictions = data['predictions']
    matchday = data.get('matchday', '?')
    generated_at = data['generated_at']

    print(f"\nPredictions for Matchday {matchday}")
    print(f"Generated: {generated_at}")
    print("-" * 40)

    for pred in predictions:
        home = pred['home_team'][:20]
        away = pred['away_team'][:20]
        score = pred['scoreline']
        print(f"  {home:<20} vs {away:<20}  ->  {score}")

    # Submit to Kicktipp
    print("\n" + "=" * 60)
    print("Submitting to Kicktipp...")
    print("=" * 60)

    try:
        client = KicktippClient()
        print(f"\nLogging in as {client.email}...")
        client.login()
        print(f"Logged in successfully!")
        print(f"Community: {client.community}")

        # Convert predictions format for kicktipp client
        pred_for_kicktipp = [
            {
                'home_team': p['home_team'],
                'away_team': p['away_team'],
                'ensemble': {
                    'home': p['home_score'],
                    'away': p['away_score'],
                }
            }
            for p in predictions
        ]

        results = client.submit_from_predictor_results(
            pred_for_kicktipp,
            overwrite=not args.no_overwrite,
            dry_run=args.dry_run
        )

        # Print results
        print("\nSubmission results:")
        print("-" * 40)

        success_count = 0
        for match, (success, message) in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {match[:40]:<40} {message}")
            if success:
                success_count += 1

        print("-" * 40)

        if args.dry_run:
            print(f"\nDRY RUN: Would submit {success_count}/{len(results)} predictions")
        else:
            print(f"\nSubmitted {success_count}/{len(results)} predictions successfully!")

    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        print("\nMake sure you have set the following in your .env file:")
        print("  KICKTIPP_EMAIL=your_email@example.com")
        print("  KICKTIPP_PASSWORD=your_password")
        print("  KICKTIPP_COMMUNITY=wolter")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
