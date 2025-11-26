"""
Enhanced feature engineering v2 with:
- Extended historical data support
- Match context features (derbies, position, momentum)
- Additional statistical features

This module provides backward-compatible feature extraction while
adding new predictive features when available.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
from typing import Dict, List, Tuple, Optional
import json
import os

from .config import ODDS_FEATURE_NAMES


# Bundesliga derbies for real-time detection
DERBIES = {
    ('bayern', 'dortmund'): 'Der Klassiker',
    ('dortmund', 'bayern'): 'Der Klassiker',
    ('dortmund', 'schalke'): 'Revierderby',
    ('schalke', 'dortmund'): 'Revierderby',
    ('gladbach', 'koln'): 'Rhine Derby',
    ('koln', 'gladbach'): 'Rhine Derby',
    ('hamburg', 'bremen'): 'Nordderby',
    ('bremen', 'hamburg'): 'Nordderby',
    ('hamburg', 'st pauli'): 'Hamburg Derby',
    ('st pauli', 'hamburg'): 'Hamburg Derby',
    ('frankfurt', 'mainz'): 'Rhein-Main Derby',
    ('mainz', 'frankfurt'): 'Rhein-Main Derby',
    ('leverkusen', 'koln'): 'Rhineland Derby',
    ('koln', 'leverkusen'): 'Rhineland Derby',
    ('union berlin', 'hertha'): 'Berlin Derby',
    ('hertha', 'union berlin'): 'Berlin Derby',
    ('leipzig', 'union berlin'): 'East German Derby',
    ('union berlin', 'leipzig'): 'East German Derby',
}

BIG_CLUBS = ['bayern', 'dortmund', 'schalke', 'hamburg', 'bremen',
             'stuttgart', 'gladbach', 'frankfurt', 'leverkusen', 'koln']


def normalize_team_name(name: str) -> str:
    """Normalize team name to simple form."""
    if pd.isna(name):
        return ""

    name = str(name).lower().strip()

    replacements = {
        'bayern munich': 'bayern', 'bayern munchen': 'bayern', 'fc bayern': 'bayern',
        'borussia dortmund': 'dortmund', 'bor. dortmund': 'dortmund',
        'borussia monchengladbach': 'gladbach', "borussia m'gladbach": 'gladbach',
        "m'gladbach": 'gladbach', 'monchengladbach': 'gladbach',
        'bayer leverkusen': 'leverkusen', 'bayer 04 leverkusen': 'leverkusen',
        'rb leipzig': 'leipzig', 'rasenballsport leipzig': 'leipzig',
        'eintracht frankfurt': 'frankfurt', 'ein frankfurt': 'frankfurt',
        'vfb stuttgart': 'stuttgart', 'sc freiburg': 'freiburg',
        'fsv mainz': 'mainz', 'mainz 05': 'mainz', '1. fsv mainz 05': 'mainz',
        'fc koln': 'koln', '1. fc koln': 'koln', 'fc cologne': 'koln',
        'werder bremen': 'bremen', 'sv werder bremen': 'bremen',
        'vfl wolfsburg': 'wolfsburg', 'tsg hoffenheim': 'hoffenheim',
        'tsg 1899 hoffenheim': 'hoffenheim', 'fc augsburg': 'augsburg',
        'hertha berlin': 'hertha', 'hertha bsc': 'hertha',
        '1. fc union berlin': 'union berlin', 'fc union berlin': 'union berlin',
        '1. fc heidenheim 1846': 'heidenheim', 'fc st. pauli': 'st pauli',
        'fc st. pauli 1910': 'st pauli', 'hamburger sv': 'hamburg',
        'fc schalke 04': 'schalke', 'schalke 04': 'schalke',
    }

    for old, new in replacements.items():
        if old in name:
            return new

    return name


class FeatureExtractorV2:
    """
    Enhanced feature extractor with context features.

    This is backward compatible with the original FeatureExtractor
    but adds new features when extended data is available.
    """

    def __init__(self, df: pd.DataFrame, extended_data_path: str = None):
        """
        Initialize with match data.

        Args:
            df: DataFrame with match data
            extended_data_path: Optional path to extended historical data CSV
        """
        self.df = df
        self.extended_df = None
        self.context_data = None

        # Try to load extended data
        if extended_data_path and os.path.exists(extended_data_path):
            self._load_extended_data(extended_data_path)
        else:
            # Try default path
            default_path = 'bundesliga_historical_odds_extended.csv'
            if os.path.exists(default_path):
                self._load_extended_data(default_path)

        # Load context data if available
        context_path = 'bundesliga_match_context.json'
        if os.path.exists(context_path):
            try:
                with open(context_path, 'r') as f:
                    self.context_data = json.load(f)
            except:
                pass

    def _load_extended_data(self, path: str):
        """Load extended historical data."""
        try:
            self.extended_df = pd.read_csv(path)
            self.extended_df['Date'] = pd.to_datetime(self.extended_df['Date'])
            print(f"  Loaded extended data: {len(self.extended_df)} matches")
        except Exception as e:
            print(f"  Warning: Could not load extended data: {e}")

    def calculate_team_stats(self, team: str, before_date: pd.Timestamp) -> Dict[str, float]:
        """Calculate team statistics up to a given date."""
        team_matches = self.df[self.df['date'] < before_date]

        # Home matches
        home_matches = team_matches[team_matches['home_team'] == team]
        home_wins = len(home_matches[home_matches['home_score'] > home_matches['away_score']])
        home_draws = len(home_matches[home_matches['home_score'] == home_matches['away_score']])
        home_goals_for = home_matches['home_score'].sum()
        home_goals_against = home_matches['away_score'].sum()

        # Away matches
        away_matches = team_matches[team_matches['away_team'] == team]
        away_wins = len(away_matches[away_matches['away_score'] > away_matches['home_score']])
        away_draws = len(away_matches[away_matches['away_score'] == away_matches['home_score']])
        away_goals_for = away_matches['away_score'].sum()
        away_goals_against = away_matches['home_score'].sum()

        # Combined
        total_matches = len(home_matches) + len(away_matches)
        total_wins = home_wins + away_wins
        total_draws = home_draws + away_draws
        total_goals_for = home_goals_for + away_goals_for
        total_goals_against = home_goals_against + away_goals_against

        # Recent form (last 5)
        recent_matches = team_matches[
            (team_matches['home_team'] == team) | (team_matches['away_team'] == team)
        ].tail(5)

        recent_points = 0
        for _, match in recent_matches.iterrows():
            if match['home_team'] == team:
                if match['home_score'] > match['away_score']:
                    recent_points += 3
                elif match['home_score'] == match['away_score']:
                    recent_points += 1
            else:
                if match['away_score'] > match['home_score']:
                    recent_points += 3
                elif match['away_score'] == match['home_score']:
                    recent_points += 1

        return {
            'win_rate': total_wins / total_matches if total_matches > 0 else 0,
            'draw_rate': total_draws / total_matches if total_matches > 0 else 0,
            'goals_for_avg': total_goals_for / total_matches if total_matches > 0 else 0,
            'goals_against_avg': total_goals_against / total_matches if total_matches > 0 else 0,
            'goal_difference': total_goals_for - total_goals_against,
            'recent_form': recent_points / 15.0,
            'home_win_rate': home_wins / len(home_matches) if len(home_matches) > 0 else 0,
            'away_win_rate': away_wins / len(away_matches) if len(away_matches) > 0 else 0,
            'home_goals_avg': home_goals_for / len(home_matches) if len(home_matches) > 0 else 0,
            'away_goals_avg': away_goals_for / len(away_matches) if len(away_matches) > 0 else 0,
        }

    def calculate_h2h_stats(
        self,
        home_team: str,
        away_team: str,
        before_date: pd.Timestamp
    ) -> Dict[str, float]:
        """Calculate head-to-head statistics."""
        h2h_matches = self.df[
            (self.df['date'] < before_date) &
            (((self.df['home_team'] == home_team) & (self.df['away_team'] == away_team)) |
             ((self.df['home_team'] == away_team) & (self.df['away_team'] == home_team)))
        ]

        h2h_home_wins = 0
        h2h_goals_for = 0
        h2h_goals_against = 0

        for _, h2h in h2h_matches.iterrows():
            if h2h['home_team'] == home_team:
                if h2h['home_score'] > h2h['away_score']:
                    h2h_home_wins += 1
                h2h_goals_for += h2h['home_score']
                h2h_goals_against += h2h['away_score']
            else:
                if h2h['away_score'] > h2h['home_score']:
                    h2h_home_wins += 1
                h2h_goals_for += h2h['away_score']
                h2h_goals_against += h2h['home_score']

        h2h_total = len(h2h_matches)

        return {
            'h2h_home_win_rate': h2h_home_wins / h2h_total if h2h_total > 0 else 0.5,
            'h2h_goals_for_avg': h2h_goals_for / h2h_total if h2h_total > 0 else 1.5,
            'h2h_goals_against_avg': h2h_goals_against / h2h_total if h2h_total > 0 else 1.5,
        }

    def calculate_context_features(
        self,
        home_team: str,
        away_team: str,
        match_date: pd.Timestamp
    ) -> Dict[str, float]:
        """Calculate match context features."""
        home_norm = normalize_team_name(home_team)
        away_norm = normalize_team_name(away_team)

        features = {
            'is_derby': 0.0,
            'home_is_big_club': 0.0,
            'away_is_big_club': 0.0,
            'home_momentum': 0.0,
            'away_momentum': 0.0,
            'home_consistency': 0.5,
            'away_consistency': 0.5,
            'position_diff': 0.0,
            'title_race_match': 0.0,
            'relegation_match': 0.0,
        }

        # Derby check
        if (home_norm, away_norm) in DERBIES or (away_norm, home_norm) in DERBIES:
            features['is_derby'] = 1.0

        # Big club check
        if home_norm in BIG_CLUBS:
            features['home_is_big_club'] = 1.0
        if away_norm in BIG_CLUBS:
            features['away_is_big_club'] = 1.0

        # Try to get context from extended data
        if self.extended_df is not None:
            try:
                # Find matching match in extended data
                mask = (
                    (self.extended_df['HomeTeam'].apply(normalize_team_name) == home_norm) &
                    (self.extended_df['AwayTeam'].apply(normalize_team_name) == away_norm) &
                    (self.extended_df['Date'].dt.date == match_date.date())
                )

                match_row = self.extended_df[mask]
                if len(match_row) > 0:
                    row = match_row.iloc[0]

                    if 'home_momentum' in row and pd.notna(row['home_momentum']):
                        features['home_momentum'] = float(row['home_momentum'])
                    if 'away_momentum' in row and pd.notna(row['away_momentum']):
                        features['away_momentum'] = float(row['away_momentum'])
                    if 'home_consistency' in row and pd.notna(row['home_consistency']):
                        features['home_consistency'] = float(row['home_consistency'])
                    if 'away_consistency' in row and pd.notna(row['away_consistency']):
                        features['away_consistency'] = float(row['away_consistency'])
                    if 'position_diff' in row and pd.notna(row['position_diff']):
                        features['position_diff'] = float(row['position_diff']) / 17.0  # Normalize
                    if 'home_title_race' in row and row['home_title_race']:
                        features['title_race_match'] = 1.0
                    if 'away_title_race' in row and row['away_title_race']:
                        features['title_race_match'] = 1.0
                    if 'home_relegation' in row and row['home_relegation']:
                        features['relegation_match'] = 1.0
                    if 'away_relegation' in row and row['away_relegation']:
                        features['relegation_match'] = 1.0

            except Exception as e:
                pass  # Use defaults

        # Calculate momentum from our own data if not found
        if features['home_momentum'] == 0.0:
            features['home_momentum'], features['home_consistency'] = \
                self._calculate_momentum(home_team, match_date)
        if features['away_momentum'] == 0.0:
            features['away_momentum'], features['away_consistency'] = \
                self._calculate_momentum(away_team, match_date)

        return features

    def _calculate_momentum(self, team: str, before_date: pd.Timestamp) -> Tuple[float, float]:
        """Calculate form momentum from our data."""
        team_matches = self.df[
            ((self.df['home_team'] == team) | (self.df['away_team'] == team)) &
            (self.df['date'] < before_date)
        ].tail(10)

        if len(team_matches) < 4:
            return 0.0, 0.5

        points = []
        for _, match in team_matches.iterrows():
            if match['home_team'] == team:
                if match['home_score'] > match['away_score']:
                    pts = 3
                elif match['home_score'] == match['away_score']:
                    pts = 1
                else:
                    pts = 0
            else:
                if match['away_score'] > match['home_score']:
                    pts = 3
                elif match['away_score'] == match['home_score']:
                    pts = 1
                else:
                    pts = 0
            points.append(pts)

        # Momentum: recent vs earlier
        if len(points) >= 6:
            recent = np.mean(points[-3:])
            earlier = np.mean(points[-6:-3])
            momentum = (recent - earlier) / 3.0
        else:
            momentum = 0.0

        # Consistency
        consistency = 1.0 - (np.std(points) / 1.5) if len(points) > 1 else 0.5
        consistency = max(0, min(1, consistency))

        return momentum, consistency

    def extract_match_features(
        self,
        home_team: str,
        away_team: str,
        match_date: pd.Timestamp,
        odds_home: Optional[float] = None,
        odds_draw: Optional[float] = None,
        odds_away: Optional[float] = None
    ) -> np.ndarray:
        """
        Extract all features for a single match.

        Returns feature vector compatible with original (39 features)
        plus optional extended features (10 additional).
        """
        home_stats = self.calculate_team_stats(home_team, match_date)
        away_stats = self.calculate_team_stats(away_team, match_date)
        h2h_stats = self.calculate_h2h_stats(home_team, away_team, match_date)
        context = self.calculate_context_features(home_team, away_team, match_date)

        # Original 19 historical features
        feature_list = [
            home_stats['win_rate'],
            home_stats['draw_rate'],
            home_stats['goals_for_avg'],
            home_stats['goals_against_avg'],
            home_stats['goal_difference'],
            home_stats['recent_form'],
            home_stats['home_win_rate'],
            home_stats['home_goals_avg'],
            away_stats['win_rate'],
            away_stats['draw_rate'],
            away_stats['goals_for_avg'],
            away_stats['goals_against_avg'],
            away_stats['goal_difference'],
            away_stats['recent_form'],
            away_stats['away_win_rate'],
            away_stats['away_goals_avg'],
            h2h_stats['h2h_home_win_rate'],
            h2h_stats['h2h_goals_for_avg'],
            h2h_stats['h2h_goals_against_avg'],
        ]

        # Original 20 odds features
        odds_features = extract_odds_features(odds_home, odds_draw, odds_away)
        for feat_name in ODDS_FEATURE_NAMES:
            feature_list.append(odds_features[feat_name])

        # NEW: 10 context features
        feature_list.extend([
            context['is_derby'],
            context['home_is_big_club'],
            context['away_is_big_club'],
            context['home_momentum'],
            context['away_momentum'],
            context['home_consistency'],
            context['away_consistency'],
            context['position_diff'],
            context['title_race_match'],
            context['relegation_match'],
        ])

        return np.array([feature_list])

    def extract_training_features(
        self,
        df: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract features for all matches."""
        if df is None:
            df = self.df

        features = []
        home_scores = []
        away_scores = []
        scorelines = []

        for _, match in df.iterrows():
            odds_home = match.get('odds_home') if 'odds_home' in match else None
            odds_draw = match.get('odds_draw') if 'odds_draw' in match else None
            odds_away = match.get('odds_away') if 'odds_away' in match else None

            X = self.extract_match_features(
                match['home_team'],
                match['away_team'],
                match['date'],
                odds_home,
                odds_draw,
                odds_away
            )

            features.append(X[0])
            home_scores.append(int(match['home_score']))
            away_scores.append(int(match['away_score']))

            h_score = min(int(match['home_score']), 5)
            a_score = min(int(match['away_score']), 5)
            scorelines.append(f"{h_score}-{a_score}")

        return (
            np.array(features),
            np.array(home_scores),
            np.array(away_scores),
            np.array(scorelines)
        )


def extract_odds_features(
    odds_home: Optional[float],
    odds_draw: Optional[float],
    odds_away: Optional[float]
) -> Dict[str, float]:
    """Extract odds-based features (same as original)."""
    features = {}

    # Raw odds with defaults
    features['odds_home'] = odds_home if odds_home and not pd.isna(odds_home) else 2.5
    features['odds_draw'] = odds_draw if odds_draw and not pd.isna(odds_draw) else 3.5
    features['odds_away'] = odds_away if odds_away and not pd.isna(odds_away) else 2.5

    # Implied probabilities
    if odds_home and odds_draw and odds_away and not any(pd.isna([odds_home, odds_draw, odds_away])):
        prob_h = 1 / odds_home
        prob_d = 1 / odds_draw
        prob_a = 1 / odds_away
        total = prob_h + prob_d + prob_a

        features['prob_home'] = prob_h / total
        features['prob_draw'] = prob_d / total
        features['prob_away'] = prob_a / total
        features['bookmaker_margin'] = total - 1.0

        features['home_away_odds_ratio'] = odds_away / odds_home
        features['home_draw_odds_ratio'] = odds_draw / odds_home
        features['away_draw_odds_ratio'] = odds_draw / odds_away

        home_xg = -np.log(features['prob_away'] + features['prob_draw'] * 0.5) * 1.2
        away_xg = -np.log(features['prob_home'] + features['prob_draw'] * 0.5) * 1.2
        features['odds_xg_home'] = np.clip(home_xg, 0.3, 4.0)
        features['odds_xg_away'] = np.clip(away_xg, 0.3, 4.0)
        features['odds_xg_diff'] = features['odds_xg_home'] - features['odds_xg_away']
        features['odds_xg_total'] = features['odds_xg_home'] + features['odds_xg_away']

        probs = np.array([features['prob_home'], features['prob_draw'], features['prob_away']])
        features['market_entropy'] = -np.sum(probs * np.log(probs + 1e-10))

        features['favorite_strength'] = max(probs)
        features['underdog_chance'] = min(features['prob_home'], features['prob_away'])
        features['odds_spread'] = max(odds_home, odds_away) - min(odds_home, odds_away)

        likely_h, likely_a = _calculate_likely_score(features['odds_xg_home'], features['odds_xg_away'])
        features['odds_likely_home_score'] = likely_h
        features['odds_likely_away_score'] = likely_a
    else:
        # Defaults
        features['prob_home'] = 0.33
        features['prob_draw'] = 0.33
        features['prob_away'] = 0.33
        features['bookmaker_margin'] = 0.1
        features['home_away_odds_ratio'] = 1.0
        features['home_draw_odds_ratio'] = 1.0
        features['away_draw_odds_ratio'] = 1.0
        features['odds_xg_home'] = 1.5
        features['odds_xg_away'] = 1.5
        features['odds_xg_diff'] = 0.0
        features['odds_xg_total'] = 3.0
        features['market_entropy'] = 1.0
        features['favorite_strength'] = 0.4
        features['underdog_chance'] = 0.3
        features['odds_spread'] = 1.0
        features['odds_likely_home_score'] = 1
        features['odds_likely_away_score'] = 1

    return features


def _calculate_likely_score(home_xg: float, away_xg: float, max_goals: int = 5) -> Tuple[int, int]:
    """Calculate most likely scoreline using Poisson."""
    max_prob = 0
    best_home = 1
    best_away = 1

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            prob = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
            if prob > max_prob:
                max_prob = prob
                best_home = h
                best_away = a

    return best_home, best_away


# Extended feature names for reference
CONTEXT_FEATURE_NAMES = [
    'is_derby',
    'home_is_big_club',
    'away_is_big_club',
    'home_momentum',
    'away_momentum',
    'home_consistency',
    'away_consistency',
    'position_diff',
    'title_race_match',
    'relegation_match',
]
