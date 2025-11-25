#!/usr/bin/env python3
"""
Show upcoming fixtures for next matchday.

Usage:
    uv run python show_fixtures.py
"""

from bundesliga_predictor.data import FixturesFetcher


def main():
    fetcher = FixturesFetcher()
    fixtures = fetcher.get_next_matchday()

    if not fixtures:
        print("No upcoming fixtures found.")
        return

    matchday = fixtures[0].get('matchday', '?')
    print(f"\nUpcoming Fixtures - Matchday {matchday}")
    print("=" * 50)

    for f in fixtures:
        home = f['homeTeam']['name']
        away = f['awayTeam']['name']
        date = f['utcDate'][:16].replace('T', ' ')
        print(f"{date}  {home} vs {away}")


if __name__ == "__main__":
    main()
