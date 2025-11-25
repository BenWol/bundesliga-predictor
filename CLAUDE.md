# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Bundesliga football match prediction system that uses machine learning to predict match scores. It combines multiple models (Multi-Output Regression, Classification, Poisson Regression, and Naive Odds-based) with a consensus ensemble strategy that outperforms individual models.

**Key Results from Backtesting:**
- Ensemble (Consensus): 1.52 pts/match ⭐
- Model4 (Naive Odds): 1.49 pts/match
- Model1 (Multi-Output): 1.42 pts/match
- Model3 (Poisson): 1.16 pts/match
- Model2 (Classification): 1.00 pts/match

## Quick Start

### Setup
```bash
uv pip install -r requirements.txt
```

### First-time data fetch (required before predictions)
```bash
uv run python fetch_data.py
```

### Weekly Predictions (MAIN ENTRY POINT)
```bash
# Get predictions for next matchday
uv run python predict.py

# Quick mode (minimal output)
uv run python predict.py --quick

# JSON output
uv run python predict.py --json

# Save predictions to file
uv run python predict.py --save predictions.json
```

### Running with API keys
Create a `.env` file:
```
FOOTBALL_API_KEY=your_key_here
ODDS_API_KEY=your_odds_api_key_here
```

## Project Structure

```
bundesliga_predictor/          # Main package (OOP structure)
├── __init__.py               # Package exports
├── config.py                 # Configuration settings
├── data.py                   # Data loading and management
├── features.py               # Feature engineering
├── scoring.py                # Kicktipp scoring system
├── ensemble.py               # Ensemble strategies
├── predictor.py              # Main predictor class
└── models/                   # ML models
    ├── base.py               # Abstract base class
    ├── multi_output.py       # Multi-output regression
    ├── classification.py     # Multi-class classification
    ├── poisson.py           # Poisson regression
    └── naive_odds.py        # Odds-based baseline

predict.py                    # Main entry point script
fetch_data.py                 # Data fetching utility
```

## Architecture

### Models

1. **Multi-Output Regression** (`model1`): Random Forest predicting home and away scores directly
2. **Multi-Class Classification** (`model2`): Random Forest classifying scorelines (e.g., "2-1")
3. **Poisson Regression** (`model3`): Separate models for expected goals, then Poisson distribution
4. **Naive Odds** (`model4`): Uses betting odds to determine favorite, predicts typical scoreline

### Ensemble Strategy

The **Consensus Ensemble** is the winning strategy:
- If 3+ models agree on a prediction → use that prediction
- Otherwise → fall back to Model4 (Naive Odds)

This simple strategy outperforms all individual models because:
- Consensus indicates higher confidence
- Model4 is a strong fallback (hard to beat)

### Feature Engineering

39 features per match:
- Team statistics (win rate, goals, form) - 16 features
- Head-to-head statistics - 3 features
- Odds-based features (probabilities, xG, entropy) - 20 features

### Kicktipp Scoring

- Exact score: 4 points
- Correct goal difference: 3 points
- Correct tendency (win/draw/loss): 2 points
- Wrong: 0 points

## Usage Examples

### Python API
```python
from bundesliga_predictor import BundesligaPredictor

predictor = BundesligaPredictor()
predictions = predictor.predict_next_matchday()

for pred in predictions:
    print(f"{pred['home_team']} vs {pred['away_team']}")
    print(f"  Ensemble: {pred['ensemble']['scoreline']}")
```

### Individual Models
```python
from bundesliga_predictor.models import PoissonRegressionModel

model = PoissonRegressionModel()
model.train(X_train, y_home, y_away)
home, away = model.predict(X_test)
```

## API Integration

- **Football Data API**: https://api.football-data.org/v4
- **The Odds API**: https://api.the-odds-api.com/v4
- **Historical Odds**: https://www.football-data.co.uk (CSV)

## Data Files

- `bundesliga_matches.json` - Cached match data
- `bundesliga_historical_odds.csv` - Historical betting odds
- `bundesliga_odds.json` - Current odds for upcoming matches
- `team_model_recommendations.json` - Backtest recommendations

## Dependencies

- requests, pandas, numpy, scikit-learn, scipy, python-dotenv, tqdm
