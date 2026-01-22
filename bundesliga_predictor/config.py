"""
Configuration settings for the Bundesliga Predictor.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
FOOTBALL_API_KEY = os.getenv('FOOTBALL_API_KEY', 'DEMO')
ODDS_API_KEY = os.getenv('ODDS_API_KEY', '')

# Kicktipp Configuration
KICKTIPP_EMAIL = os.getenv('KICKTIPP_EMAIL', '')
KICKTIPP_PASSWORD = os.getenv('KICKTIPP_PASSWORD', '')
KICKTIPP_COMMUNITY = os.getenv('KICKTIPP_COMMUNITY', '')

FOOTBALL_API_BASE_URL = 'https://api.football-data.org/v4'
ODDS_API_BASE_URL = 'https://api.the-odds-api.com/v4'
BUNDESLIGA_ID = 'BL1'

# Cache files
CACHE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATCHES_CACHE_FILE = os.path.join(CACHE_DIR, 'bundesliga_matches.json')
ODDS_CACHE_FILE = os.path.join(CACHE_DIR, 'bundesliga_odds.json')
HISTORICAL_ODDS_FILE = os.path.join(CACHE_DIR, 'bundesliga_historical_odds.csv')
RECOMMENDATIONS_FILE = os.path.join(CACHE_DIR, 'team_model_recommendations.json')

# Model Configuration
MODEL_RANDOM_STATE = 42
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH_REGRESSION = 12
RF_MAX_DEPTH_CLASSIFICATION = 15
RF_MAX_DEPTH_POISSON = 10

# Training Configuration
TRAIN_TEST_SPLIT_RATIO = 0.8
MIN_TRAINING_SAMPLES = 50
TRAINING_SEASONS = 2  # Number of previous seasons to use for training

# Kicktipp Scoring
POINTS_EXACT = 4
POINTS_GOAL_DIFF = 3
POINTS_TENDENCY = 2
POINTS_WRONG = 0

# Model names for display
MODEL_NAMES = {
    'model1': 'Multi-Output Regression',
    'model2': 'Multi-Class Classification',
    'model3': 'Poisson Regression',
    'model4': 'Naive Odds-based',
    'gradient_boosting': 'Gradient Boosting',
    'bivariate_poisson': 'Bivariate Poisson',
    'smart_odds': 'Smart Odds',
    'tendency_first': 'Tendency First',
    'probability_max': 'Probability Max',
}

# Default scorelines for naive model
DEFAULT_HOME_WIN_SCORE = (2, 1)
DEFAULT_DRAW_SCORE = (1, 1)
DEFAULT_AWAY_WIN_SCORE = (1, 2)

# Feature names for historical stats
HISTORICAL_FEATURE_NAMES = [
    'home_win_rate', 'home_draw_rate', 'home_goals_for_avg', 'home_goals_against_avg',
    'home_goal_difference', 'home_recent_form', 'home_home_win_rate', 'home_home_goals_avg',
    'away_win_rate', 'away_draw_rate', 'away_goals_for_avg', 'away_goals_against_avg',
    'away_goal_difference', 'away_recent_form', 'away_away_win_rate', 'away_away_goals_avg',
    'h2h_home_win_rate', 'h2h_goals_for_avg', 'h2h_goals_against_avg',
]

# Odds feature names
ODDS_FEATURE_NAMES = [
    'odds_home', 'odds_draw', 'odds_away',
    'prob_home', 'prob_draw', 'prob_away',
    'bookmaker_margin',
    'home_away_odds_ratio', 'home_draw_odds_ratio', 'away_draw_odds_ratio',
    'odds_xg_home', 'odds_xg_away', 'odds_xg_diff', 'odds_xg_total',
    'market_entropy', 'favorite_strength', 'underdog_chance', 'odds_spread',
    'odds_likely_home_score', 'odds_likely_away_score',
]
