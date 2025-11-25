"""
Data loading and management for match data and odds.
"""

import json
import os
import requests
import unicodedata
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
import numpy as np

from .config import (
    FOOTBALL_API_KEY, FOOTBALL_API_BASE_URL, ODDS_API_KEY, ODDS_API_BASE_URL,
    BUNDESLIGA_ID, MATCHES_CACHE_FILE, ODDS_CACHE_FILE, HISTORICAL_ODDS_FILE
)


class DataLoader:
    """Handles loading and processing of match and odds data."""

    def __init__(self):
        """Initialize data loader."""
        self._matches_df: Optional[pd.DataFrame] = None

    def load_cached_matches(self) -> pd.DataFrame:
        """
        Load match data from cache file.

        Returns:
            DataFrame with match data

        Raises:
            FileNotFoundError: If cache file doesn't exist
        """
        if not os.path.exists(MATCHES_CACHE_FILE):
            raise FileNotFoundError(
                f"Cache file '{MATCHES_CACHE_FILE}' not found.\n"
                f"Please run 'python -m bundesliga_predictor.fetch' first."
            )

        with open(MATCHES_CACHE_FILE, 'r') as f:
            cache_data = json.load(f)

        print(f"Loading cached data from {os.path.basename(MATCHES_CACHE_FILE)}")
        print(f"  Fetched at: {cache_data['fetched_at']}")
        print(f"  Total matches: {cache_data['total_matches']}")

        df = self._process_matches(cache_data['matches'])

        # Load and merge odds
        df = self._load_and_merge_odds(df)

        self._matches_df = df
        return df

    def _process_matches(self, matches: List[Dict]) -> pd.DataFrame:
        """Process raw match data into DataFrame."""
        processed = []

        for match in matches:
            if match['status'] == 'FINISHED' and match.get('score', {}).get('fullTime'):
                processed.append({
                    'date': match['utcDate'],
                    'home_team': match['homeTeam']['name'],
                    'away_team': match['awayTeam']['name'],
                    'home_score': match['score']['fullTime'].get('home', 0),
                    'away_score': match['score']['fullTime'].get('away', 0),
                    'matchday': match.get('matchday'),
                })

        df = pd.DataFrame(processed)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

        print(f"  Processed {len(df)} finished matches")
        return df

    def _load_and_merge_odds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Load historical odds and merge with match data."""
        if not os.path.exists(HISTORICAL_ODDS_FILE):
            print(f"  Warning: {os.path.basename(HISTORICAL_ODDS_FILE)} not found")
            return df

        odds_df = pd.read_csv(HISTORICAL_ODDS_FILE)
        print(f"  Loaded {len(odds_df)} matches with historical odds")

        if 'Date' in odds_df.columns:
            odds_df['Date'] = pd.to_datetime(odds_df['Date'], format='%d/%m/%Y', errors='coerce')

        # Add odds columns
        df['odds_home'] = np.nan
        df['odds_draw'] = np.nan
        df['odds_away'] = np.nan

        matched_count = 0
        for idx, match_row in df.iterrows():
            home_norm = self._normalize_team_name(match_row['home_team'])
            away_norm = self._normalize_team_name(match_row['away_team'])

            for _, odds_row in odds_df.iterrows():
                odds_home_norm = self._normalize_team_name(str(odds_row.get('HomeTeam', '')))
                odds_away_norm = self._normalize_team_name(str(odds_row.get('AwayTeam', '')))

                home_match = self._teams_match(home_norm, odds_home_norm)
                away_match = self._teams_match(away_norm, odds_away_norm)

                if home_match and away_match:
                    if 'AvgH' in odds_row and pd.notna(odds_row.get('AvgH')):
                        df.at[idx, 'odds_home'] = odds_row['AvgH']
                        df.at[idx, 'odds_draw'] = odds_row['AvgD']
                        df.at[idx, 'odds_away'] = odds_row['AvgA']
                        matched_count += 1
                    elif 'PSH' in odds_row and pd.notna(odds_row.get('PSH')):
                        df.at[idx, 'odds_home'] = odds_row['PSH']
                        df.at[idx, 'odds_draw'] = odds_row['PSD']
                        df.at[idx, 'odds_away'] = odds_row['PSA']
                        matched_count += 1
                    break

        print(f"  Matched odds for {matched_count}/{len(df)} matches ({matched_count/len(df)*100:.1f}%)")
        return df

    @staticmethod
    def _normalize_team_name(name: str) -> str:
        """Normalize team name for matching."""
        name = name.lower()
        name = unicodedata.normalize('NFKD', name)
        name = ''.join([c for c in name if not unicodedata.combining(c)])

        city_vars = {'munchen': 'munich', 'koln': 'cologne'}
        for german, english in city_vars.items():
            name = name.replace(german, english)

        prefixes = ['fc', 'sv', 'tsg', 'vfl', 'bv', 'sc', '1.', 'bor.', 'fsv']
        suffixes = ['1910', '1846', '1899', '1904', '04', '05', '06', '09', 'ev', 'e.v.']

        words = [w.replace('.', '') for w in name.split()
                 if w not in prefixes and w not in suffixes and w]
        return ' '.join(words)

    @staticmethod
    def _teams_match(name1: str, name2: str) -> bool:
        """Check if two normalized team names match."""
        return (name1 in name2 or name2 in name1 or
                all(word in name2 for word in name1.split() if len(word) > 3))

    def get_matches_df(self) -> pd.DataFrame:
        """Get cached matches DataFrame, loading if necessary."""
        if self._matches_df is None:
            return self.load_cached_matches()
        return self._matches_df


class FixturesFetcher:
    """Fetches upcoming fixtures from the API."""

    def __init__(self):
        """Initialize fixtures fetcher."""
        self._headers = {'X-Auth-Token': FOOTBALL_API_KEY} if FOOTBALL_API_KEY != 'DEMO' else {}

    def get_next_matchday(self) -> List[Dict[str, Any]]:
        """
        Get fixtures for the next matchday.

        Returns:
            List of fixture dictionaries
        """
        url = f"{FOOTBALL_API_BASE_URL}/competitions/{BUNDESLIGA_ID}/matches"

        try:
            response = requests.get(url, headers=self._headers)
            response.raise_for_status()
            data = response.json()

            # Get scheduled matches
            upcoming = [m for m in data.get('matches', [])
                       if m['status'] in ['SCHEDULED', 'TIMED']]

            if not upcoming:
                return []

            # Group by matchday
            matchdays: Dict[int, List] = {}
            for match in upcoming:
                md = match.get('matchday')
                if md:
                    matchdays.setdefault(md, []).append(match)

            # Return next matchday
            if matchdays:
                next_md = min(matchdays.keys())
                return matchdays[next_md]

            return []

        except requests.exceptions.RequestException as e:
            print(f"Error fetching fixtures: {e}")
            return []

    def load_upcoming_odds(self) -> Optional[List[Dict]]:
        """Load odds for upcoming matches from cache."""
        if not os.path.exists(ODDS_CACHE_FILE):
            print(f"  Warning: {os.path.basename(ODDS_CACHE_FILE)} not found")
            return None

        with open(ODDS_CACHE_FILE, 'r') as f:
            cache = json.load(f)

        odds = cache.get('odds', [])
        print(f"  Loaded odds for {len(odds)} upcoming matches")
        return odds

    def find_fixture_odds(
        self,
        home_team: str,
        away_team: str,
        odds_data: Optional[List[Dict]]
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Find odds for a specific fixture."""
        if not odds_data:
            return None, None, None

        home_norm = DataLoader._normalize_team_name(home_team)
        away_norm = DataLoader._normalize_team_name(away_team)

        for odds_match in odds_data:
            odds_home_norm = DataLoader._normalize_team_name(odds_match['home_team'])
            odds_away_norm = DataLoader._normalize_team_name(odds_match['away_team'])

            if (DataLoader._teams_match(home_norm, odds_home_norm) and
                DataLoader._teams_match(away_norm, odds_away_norm)):

                # Extract odds from consistent bookmakers
                consistent = {'pinnacle', 'williamhill', 'onexbet', 'betfair_ex_eu'}
                home_odds, draw_odds, away_odds = [], [], []

                for bm in odds_match.get('bookmakers', []):
                    if bm['key'] in consistent:
                        market = bm['markets'][0]
                        outcomes = {o['name']: o['price'] for o in market['outcomes']}
                        home_odds.append(outcomes.get(odds_match['home_team']))
                        draw_odds.append(outcomes.get('Draw'))
                        away_odds.append(outcomes.get(odds_match['away_team']))

                if home_odds and draw_odds and away_odds:
                    return (
                        sum(home_odds) / len(home_odds),
                        sum(draw_odds) / len(draw_odds),
                        sum(away_odds) / len(away_odds)
                    )

        return None, None, None
