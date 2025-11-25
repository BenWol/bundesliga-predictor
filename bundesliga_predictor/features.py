"""
Feature engineering for match prediction.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
from typing import Dict, List, Tuple, Optional

from .config import ODDS_FEATURE_NAMES


class FeatureExtractor:
    """Extracts features from match data for ML models."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with historical match data.

        Args:
            df: DataFrame with columns: date, home_team, away_team, home_score, away_score
        """
        self.df = df

    def calculate_team_stats(self, team: str, before_date: pd.Timestamp) -> Dict[str, float]:
        """
        Calculate team statistics up to (but not including) a given date.

        Args:
            team: Team name
            before_date: Calculate stats from matches before this date

        Returns:
            Dictionary of team statistics
        """
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

        # Combined stats
        total_matches = len(home_matches) + len(away_matches)
        total_wins = home_wins + away_wins
        total_draws = home_draws + away_draws
        total_goals_for = home_goals_for + away_goals_for
        total_goals_against = home_goals_against + away_goals_against

        # Recent form (last 5 matches)
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
        """
        Calculate head-to-head statistics.

        Args:
            home_team: Home team name
            away_team: Away team name
            before_date: Calculate stats from matches before this date

        Returns:
            Dictionary of H2H statistics
        """
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

        Args:
            home_team: Home team name
            away_team: Away team name
            match_date: Match date
            odds_home: Home win odds (optional)
            odds_draw: Draw odds (optional)
            odds_away: Away win odds (optional)

        Returns:
            Feature vector as numpy array
        """
        home_stats = self.calculate_team_stats(home_team, match_date)
        away_stats = self.calculate_team_stats(away_team, match_date)
        h2h_stats = self.calculate_h2h_stats(home_team, away_team, match_date)

        # Build feature list
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

        # Add odds features
        odds_features = extract_odds_features(odds_home, odds_draw, odds_away)
        for feat_name in ODDS_FEATURE_NAMES:
            feature_list.append(odds_features[feat_name])

        return np.array([feature_list])

    def extract_training_features(
        self,
        df: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features for all matches in the dataset.

        Args:
            df: DataFrame to extract features from (uses self.df if not provided)

        Returns:
            Tuple of (features, home_scores, away_scores, scorelines)
        """
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

            # Cap scores at 5 for scoreline classification
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
    """
    Extract all odds-based features for a single match.

    Args:
        odds_home: Home win odds
        odds_draw: Draw odds
        odds_away: Away win odds

    Returns:
        Dictionary of odds features
    """
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

        # Odds ratios
        features['home_away_odds_ratio'] = odds_away / odds_home
        features['home_draw_odds_ratio'] = odds_draw / odds_home
        features['away_draw_odds_ratio'] = odds_draw / odds_away

        # Expected goals from odds
        home_xg = -np.log(features['prob_away'] + features['prob_draw'] * 0.5) * 1.2
        away_xg = -np.log(features['prob_home'] + features['prob_draw'] * 0.5) * 1.2
        features['odds_xg_home'] = np.clip(home_xg, 0.3, 4.0)
        features['odds_xg_away'] = np.clip(away_xg, 0.3, 4.0)
        features['odds_xg_diff'] = features['odds_xg_home'] - features['odds_xg_away']
        features['odds_xg_total'] = features['odds_xg_home'] + features['odds_xg_away']

        # Market entropy
        probs = np.array([features['prob_home'], features['prob_draw'], features['prob_away']])
        features['market_entropy'] = -np.sum(probs * np.log(probs + 1e-10))

        # Favorite/underdog metrics
        features['favorite_strength'] = max(probs)
        features['underdog_chance'] = min(features['prob_home'], features['prob_away'])
        features['odds_spread'] = max(odds_home, odds_away) - min(odds_home, odds_away)

        # Most likely scoreline from Poisson
        likely_h, likely_a = _calculate_likely_score(features['odds_xg_home'], features['odds_xg_away'])
        features['odds_likely_home_score'] = likely_h
        features['odds_likely_away_score'] = likely_a
    else:
        # Default values when odds not available
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
    """Calculate most likely scoreline using Poisson distribution."""
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
