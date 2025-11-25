# Odds-Based Feature Engineering Enhancements

## Overview

The prediction models have been significantly enhanced with **20 new odds-based features** derived from betting market data. These features capture market sentiment, implied probabilities, and expected goal estimates that complement traditional team statistics.

## Feature Categories

### 1. Raw Odds (3 features)
- `odds_home`: Decimal odds for home win (e.g., 2.15)
- `odds_draw`: Decimal odds for draw
- `odds_away`: Decimal odds for away win

### 2. Implied Probabilities (3 features)
- `prob_home`: Market-implied probability of home win (normalized, removes bookmaker margin)
- `prob_draw`: Market-implied probability of draw
- `prob_away`: Market-implied probability of away win

**Why useful**: Bookmakers aggregate information from many sources. These probabilities often outperform naive statistical models.

### 3. Market Efficiency Indicators (1 feature)
- `bookmaker_margin`: The overround/vig (typically 5-10%)

**Why useful**: Lower margins indicate more efficient markets with better price discovery.

### 4. Odds Ratios (3 features)
- `home_away_odds_ratio`: Relative strength indicator (odds_away / odds_home)
- `home_draw_odds_ratio`: How likely is draw vs home win
- `away_draw_odds_ratio`: How likely is draw vs away win

**Why useful**: Captures relative team strength better than absolute odds.

### 5. Expected Goals from Odds (4 features)
- `odds_xg_home`: Estimated home team expected goals from odds
- `odds_xg_away`: Estimated away team expected goals from odds
- `odds_xg_diff`: Goal difference expectation
- `odds_xg_total`: Total goals expectation

**Why useful**: Translates win probabilities into goal expectations using empirical Bundesliga relationships.

### 6. Market Uncertainty Indicators (4 features)
- `market_entropy`: Shannon entropy of outcome probabilities (higher = more uncertain)
- `favorite_strength`: Probability of most likely outcome
- `underdog_chance`: Probability of underdog winning
- `odds_spread`: Range between highest and lowest odds

**Why useful**: Identifies matches where upsets are more likely vs. predictable outcomes.

### 7. Most Likely Scoreline (2 features)
- `odds_likely_home_score`: Most probable home score from Poisson model fitted to odds
- `odds_likely_away_score`: Most probable away score from Poisson model fitted to odds

**Why useful**: Provides a baseline prediction directly from market expectations.

## Data Sources

### Historical Matches
- **Source**: football-data.co.uk CSV files (8 seasons)
- **Bookmakers**: Average of Pinnacle, William Hill, 1xBet, Betfair
- **Coverage**: ~2,400 historical matches with odds
- **Matching**: Automated team name normalization (handles Bayern München/Munich, FC Köln/Cologne, etc.)

### Upcoming Matches
- **Source**: The Odds API (bundesliga_odds.json)
- **Bookmakers**: Same 4 consistent bookmakers as historical data
- **Consistency**: Ensures training and prediction use identical odds formats

## Integration

### Modified Files

1. **data_loader.py**
   - Enhanced `get_cached_dataframe()` to load and merge historical odds
   - Added team name normalization
   - Matches odds to ~90% of historical games

2. **odds_features.py** (NEW)
   - Comprehensive feature extraction module
   - All 20 odds features with sensible defaults for missing data
   - Reusable across all prediction scripts

3. **score_prediction.py**
   - Updated `create_features_and_scores()` to include odds
   - Updated `create_fixture_features()` to load upcoming odds
   - Now uses **39 total features** (19 historical + 20 odds)

4. **backtest_season.py**
   - Updated `create_features_for_match()` to accept odds parameters
   - Updated `create_training_features()` to extract odds from dataframe
   - Enables fair backtesting with historical odds

## Expected Improvements

### Why Odds Features Help

1. **Information Aggregation**: Bookmakers synthesize:
   - Team news (injuries, suspensions)
   - Motivation factors (relegation battles, title races)
   - Recent form beyond simple win/loss
   - Public sentiment and betting patterns

2. **Residual Prediction**: Odds capture factors not in historical stats:
   - Coaching changes
   - Squad depth and rotation
   - Psychological factors (derby matches, pressure situations)

3. **Calibration**: Bookmaker probabilities are well-calibrated (they have strong financial incentives to be accurate)

### Kicktipp Scoring Optimization

The models are specifically optimized for Kicktipp.de scoring:
- **Exact result (4 pts)**: Odds-based xG helps predict likely scorelines
- **Correct difference (3 pts)**: Odds ratios capture goal difference expectations
- **Correct tendency (2 pts)**: Implied probabilities directly predict win/draw/loss

## Usage

### Run Enhanced Predictions

```bash
# Fetch data with odds (run once)
python fetch_data.py

# Run predictions with odds features
python score_prediction.py

# Backtest with odds features
python backtest_season.py
```

### Feature Statistics

After loading data, you'll see:
```
Loaded 2,448 matches with historical odds
Matched odds for 2,203/2,448 matches (90.0%)
```

The remaining 10% use sensible default values (neutral market: 2.5/3.5/2.5 odds).

## Technical Details

### Missing Data Handling

When odds are unavailable, features use neutral defaults:
- Odds: 2.5 (home), 3.5 (draw), 2.5 (away) ≈ 33% each
- xG: 1.5 goals per team (Bundesliga average)
- Entropy: 1.0 (maximum uncertainty)

This ensures models can always make predictions without errors.

### Performance Considerations

- Feature extraction: ~0.1ms per match
- No API calls during prediction (uses cached data)
- Models automatically handle increased feature dimensionality

## Next Steps

To further optimize:

1. **Feature Selection**: Use Random Forest feature importance to identify top 15-20 features
2. **Hyperparameter Tuning**: Re-optimize `max_depth`, `n_estimators` for expanded feature space
3. **Ensemble Weighting**: Weight models based on odds confidence (lower entropy = trust model more)
4. **Value Betting**: Compare model predictions to odds to identify +EV bets

## Summary

The addition of 20 odds-based features provides the models with:
- ✅ Market consensus on match outcomes
- ✅ Expected goals estimates from betting markets
- ✅ Uncertainty/confidence indicators
- ✅ Calibrated probability estimates
- ✅ Consistent data sources for training and prediction

These enhancements should significantly improve Kicktipp scoring, especially for:
- Matches with clear favorites (odds help predict scorelines)
- Close matches (odds entropy indicates unpredictability)
- Situational factors not captured in historical stats
