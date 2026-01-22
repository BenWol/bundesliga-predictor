# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Bundesliga football match prediction system that uses machine learning to predict match scores. It combines multiple models with an ensemble strategy that outperforms individual models.

**Key Results from Backtesting (2025/2026 Season, Walk-Forward):**
- Ultimate Tendency: 1.56 pts/match (best performer)
- Tendency Expert: 1.52 pts/match
- Hybrid V2 Ensemble: 1.48 pts/match
- Model4 (Naive Odds): 1.48 pts/match

**Default Ensemble:** Hybrid V2 (adaptive consensus + tendency expert + model4 fallback)

**Training Data:** Last 2 seasons + current season matchdays (walk-forward)

**Benchmark Target:** 1.65 pts/match

## Quick Start

### Setup
```bash
uv pip install -r requirements.txt
```

### Full Pipeline (MAIN ENTRY POINT)
```bash
# Full pipeline: fetch -> backtest -> predict -> submit to Kicktipp
uv run python main.py

# Dry run (skip actual submission)
uv run python main.py --dry-run

# Skip data fetching (use cached data)
uv run python main.py --skip-fetch

# Skip backtesting (use cached results)
uv run python main.py --skip-backtest

# Verbose mode (detailed output instead of progress bars)
uv run python main.py --verbose
```

The pipeline:
1. Fetches latest data (matches + odds)
2. Runs walk-forward backtest to find best model/ensemble
3. Generates predictions using the best performer
4. Shows results table and predictions
5. Waits for confirmation, then submits to Kicktipp

**Walk-Forward Methodology:**
The backtest uses walk-forward validation with parallel execution:
1. Train on last N seasons before matchday 1
2. Predict matchday 1, then add results to training data
3. Retrain and predict matchday 2
4. Continue until all matchdays are predicted

This mirrors real-world usage where each week you train on all available data.

### Running with API keys
Create a `.env` file:
```
FOOTBALL_API_KEY=your_key_here
ODDS_API_KEY=your_odds_api_key_here
KICKTIPP_EMAIL=your_email
KICKTIPP_PASSWORD=your_password
KICKTIPP_COMMUNITY=your_community_name
```

## Project Structure

```
main.py                            # Single entry point: fetch -> backtest -> predict -> submit
bundesliga_predictor/              # Main package (OOP structure)
├── __init__.py                   # Package exports
├── pipeline.py                   # Pipeline functions (fetch, backtest, predict, submit)
├── config.py                     # Configuration settings
├── data.py                       # Data loading and management
├── features.py                   # Feature engineering (39 features)
├── features_v2.py                # Extended features (49 features with context)
├── scoring.py                    # Kicktipp scoring system
├── kicktipp.py                   # Kicktipp client for submitting predictions
├── ensemble.py                   # Original consensus ensemble
├── ensemble_v2.py                # Validated v2 ensembles
├── predictor.py                  # Main predictor class
├── ensembles/                    # Ensemble strategies
│   ├── __init__.py
│   └── experimental.py           # Experimental ensembles
└── models/                       # ML models
    ├── __init__.py
    ├── base.py                   # Abstract base class
    ├── multi_output.py           # Model1: Multi-output regression
    ├── classification.py         # Model2: Multi-class classification
    ├── poisson.py                # Model3: Poisson regression
    ├── naive_odds.py             # Model4: Odds-based baseline
    ├── context_aware.py          # Context-aware model wrappers
    └── experimental/             # Experimental models
        ├── gradient_boosting.py  # Gradient Boosting
        ├── bivariate_poisson.py  # Bivariate Poisson
        ├── smart_odds.py         # Smart Odds
        ├── tendency_first.py     # Tendency-First model
        └── probability_max.py    # Probability-Maximizing model
```

## Architecture

### Core Models

1. **Multi-Output Regression** (`model1`): Random Forest predicting home and away scores directly
2. **Multi-Class Classification** (`model2`): Random Forest classifying scorelines (e.g., "2-1")
3. **Poisson Regression** (`model3`): Separate models for expected goals, then Poisson distribution
4. **Naive Odds** (`model4`): Uses betting odds to determine favorite, predicts typical scoreline

### Experimental Models

5. **Gradient Boosting**: Often outperforms Random Forest for tabular data
6. **Bivariate Poisson**: Captures correlation between home/away goals
7. **Smart Odds**: Enhanced odds-based predictions using xG
8. **Tendency First**: Two-stage model - predicts tendency then scoreline
9. **Probability Max**: Maximizes expected Kicktipp points

### Ensemble Strategies

**Validated (V2):**
- **Tendency Expert**: Uses best model for each tendency (H/D/A) - currently best
- **Tendency Consensus**: Two-stage consensus approach
- **Hybrid V2**: Combines multiple strategies adaptively

**Experimental:**
- Optimized Consensus, Hybrid, Adaptive Scoreline, Bayesian Optimal
- Aggressive Scoreline, Ultimate Tendency, Super Consensus, Max Points

### Feature Engineering

39 base features + 10 context features:
- Team statistics (win rate, goals, form) - 16 features
- Head-to-head statistics - 3 features
- Odds-based features (probabilities, xG, entropy) - 20 features
- Context features (derby, big clubs, momentum, position) - 10 features

### Kicktipp Scoring

- Exact score: 4 points
- Correct goal difference: 3 points
- Correct tendency (win/draw/loss): 2 points
- Wrong: 0 points

## Usage Examples

### Pipeline API
```python
from bundesliga_predictor import fetch_data, run_backtest, run_predict, submit_kicktipp

# Fetch latest data
fetch_data()

# Run walk-forward backtest (parallel)
results = run_backtest()

# Generate predictions
predictions = run_predict()

# Submit to Kicktipp (use_model=False means use best ensemble)
submit_kicktipp(predictions, use_model=False, dry_run=True)
```

### Predictor API
```python
from bundesliga_predictor import BundesligaPredictor

predictor = BundesligaPredictor()
predictions = predictor.predict_next_matchday()

for pred in predictions:
    print(f"{pred['home_team']} vs {pred['away_team']}")
    print(f"  Ensemble: {pred['ensemble']['scoreline']}")
```

### Using Experimental Models
```python
from bundesliga_predictor.models.experimental import GradientBoostingModel

model = GradientBoostingModel()
model.train(X_train, y_home, y_away, scorelines)
home, away = model.predict(X_test)
```

### Using Custom Ensembles
```python
from bundesliga_predictor.ensembles import MaxPointsEnsemble

ensemble = MaxPointsEnsemble()
prediction, strategy, details = ensemble.combine(
    model_predictions,
    odds_home=1.5, odds_draw=4.0, odds_away=6.0
)
```

## API Integration

- **Football Data API**: https://api.football-data.org/v4
- **The Odds API**: https://api.the-odds-api.com/v4
- **Historical Odds**: https://www.football-data.co.uk (CSV)

## Data Files

- `bundesliga_historical_odds.csv` - **Primary data source**: Historical match results + betting odds from football-data.co.uk (7+ seasons)
- `bundesliga_matches.json` - Football-data.org API cache (fallback)
- `bundesliga_odds.json` - Current odds for upcoming matches from The Odds API
- `latest_predictions.json` - Most recent predictions

**Data Flow:**
1. `fetch_data()` downloads data from football-data.org API and football-data.co.uk CSV
2. The CSV is the primary source (has more historical data with odds already included)
3. For predictions, fixtures come from football-data.org API, odds from The Odds API

## Dependencies

- requests, pandas, numpy, scikit-learn, scipy, python-dotenv, tqdm
