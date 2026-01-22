# Bundesliga Match Prediction

Machine learning-based prediction system for Bundesliga football matches using multiple models and ensemble strategies.

## Quick Start

1. **Install dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

2. **Set up API keys** (create `.env` file):
   ```
   FOOTBALL_API_KEY=your_key_here
   ODDS_API_KEY=your_odds_api_key_here
   KICKTIPP_EMAIL=your_email
   KICKTIPP_PASSWORD=your_password
   KICKTIPP_COMMUNITY=your_community_name
   ```

3. **Run the full pipeline:**
   ```bash
   uv run python main.py
   ```

   This will:
   - Fetch latest match and odds data
   - Run walk-forward backtest to find the best model/ensemble
   - Generate predictions for the next matchday
   - Show results and wait for confirmation
   - Submit to Kicktipp

## CLI Options

```bash
uv run python main.py              # Full pipeline
uv run python main.py --dry-run    # Skip actual submission
uv run python main.py --skip-fetch # Use cached data
uv run python main.py --skip-backtest  # Use cached backtest results
uv run python main.py --verbose    # Detailed output instead of progress bars
```

## Models

### Core Models
1. **Multi-Output Regression** - Random Forest predicting home/away scores directly
2. **Multi-Class Classification** - Classifies most likely scoreline
3. **Poisson Regression** - Models goals as Poisson distributions
4. **Naive Odds** - Uses betting odds to predict typical scorelines

### Experimental Models
5. **Gradient Boosting** - Often outperforms Random Forest for tabular data
6. **Bivariate Poisson** - Captures correlation between home/away goals
7. **Smart Odds** - Enhanced odds-based predictions using xG
8. **Tendency First** - Two-stage model (tendency → scoreline)
9. **Probability Max** - Maximizes expected Kicktipp points

### Ensemble Strategies
- **Tendency Consensus** - Best performer (~1.50 pts/match)
- **Tendency Expert** - Uses best model per tendency
- Multiple other ensemble strategies

## Kicktipp Scoring

- Exact result: 4 points
- Correct goal difference: 3 points
- Correct tendency (win/draw/loss): 2 points
- Wrong prediction: 0 points

**Benchmark target:** 1.65 pts/match

## Features

Each model uses **39+ features** per match:

### Historical Statistics (19 features)
- Team win/draw/loss rates (overall, home, away)
- Goals for/against averages
- Goal difference and recent form
- Head-to-head statistics

### Odds-Based Features (20 features)
- Raw betting odds and implied probabilities
- Expected goals derived from market prices
- Market efficiency indicators (entropy, favorite strength)
- Odds ratios capturing relative team strength

## Python API

```python
from bundesliga_predictor import fetch_data, run_backtest, run_predict, submit_kicktipp

fetch_data()
results = run_backtest()
predictions = run_predict()
submit_kicktipp(predictions, use_model=False, dry_run=True)
```

## Project Structure

```
main.py                     # Single entry point
bundesliga_predictor/       # Main package
├── pipeline.py            # Pipeline functions (fetch, backtest, predict, submit)
├── predictor.py           # Main predictor class
├── models/                # ML models
├── ensembles/             # Ensemble strategies
├── features.py            # Feature engineering
└── kicktipp.py            # Kicktipp client
```

## Data Sources

- **Football Data API**: https://api.football-data.org/v4
- **Historical Odds**: https://www.football-data.co.uk (CSV)
- **Current Odds**: https://api.the-odds-api.com/v4
