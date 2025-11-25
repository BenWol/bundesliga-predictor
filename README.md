# Bundesliga Match Prediction

Machine learning-based prediction system for Bundesliga football matches using historical data and Random Forest models.

## Quick Start

1. **Install dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

2. **Fetch historical data (one-time setup):**
   ```bash
   uv run python fetch_data.py
   ```
   This downloads ~8 seasons of Bundesliga data and caches it locally in `bundesliga_matches.json`.

3. **Run predictions:**
   ```bash
   # Exact score predictions with Kicktipp scoring
   uv run python score_prediction.py

   # Backtest to see which model performs best this season
   uv run python backtest_season.py

   # Win/draw/loss predictions
   uv run python bundesliga_predictor.py
   ```

## Features

### Three Prediction Models
1. **Multi-Output Regression** - Predicts home and away scores simultaneously
2. **Multi-Class Classification** - Predicts most likely scoreline from historical patterns
3. **Poisson Regression** - Models goals as Poisson distributions

### Kicktipp.de Scoring
All models are evaluated using the Kicktipp.de scoring system:
- Exact result: 4 points
- Correct goal difference: 3 points
- Correct tendency (win/draw/loss): 2 points
- Wrong prediction: 0 points

### Season Backtesting
Simulates using models throughout the current season to determine which performs best:
- Trains on past data before each matchday
- Predicts that week's matches
- Tracks cumulative Kicktipp scores
- Shows which model would be winning

## Data Caching

To minimize API calls, the system uses a two-stage approach:
1. **fetch_data.py** - Downloads all data once and caches it
2. **data_loader.py** - Shared module that all scripts use to load cached data

This allows you to run predictions and experiments without making repeated API calls.

## Configuration

Set your API key in `.env`:
```
FOOTBALL_API_KEY=your_key_here
```

Get a free API key from: https://www.football-data.org/

## Project Structure

- `fetch_data.py` - Data fetching and caching
- `data_loader.py` - Shared data loading utility
- `score_prediction.py` - Score predictions with 3-model comparison
- `backtest_season.py` - Season simulation to find best model
- `bundesliga_predictor.py` - Win/draw/loss predictions
- `bundesliga_matches.json` - Cached match data (auto-generated)

## Model Features

Each model uses **39 features** per match (19 historical + 20 odds-based):

### Historical Statistics (19 features)
- Team win/draw/loss rates (overall, home, away)
- Goals for/against averages
- Goal difference
- Recent form (last 5 matches)
- Head-to-head statistics

### Odds-Based Features (20 features) 🆕
- Raw betting odds and implied probabilities
- Expected goals derived from market prices
- Market efficiency indicators (entropy, favorite strength)
- Odds ratios capturing relative team strength
- Most likely scoreline from Poisson model

**Why odds help**: Bookmakers aggregate information from injuries, form, motivation, and public sentiment that isn't captured in historical stats alone.

All models use temporal train/test split (80/20 chronological) to prevent data leakage.

See [ODDS_FEATURES_README.md](ODDS_FEATURES_README.md) for detailed feature descriptions.
