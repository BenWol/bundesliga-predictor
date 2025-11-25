#!/usr/bin/env python3
"""
Fetch Bundesliga Data and Cache Locally
Run this script once to download all historical data and save it locally.
Other scripts will then use the cached data instead of making API calls.
"""

import requests
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import io

# Load environment variables
load_dotenv()

FOOTBALL_API_KEY = os.getenv('FOOTBALL_API_KEY', 'DEMO')
ODDS_API_KEY = os.getenv('ODDS_API_KEY', '')  # Get from the-odds-api.com
BASE_URL = 'https://api.football-data.org/v4'
ODDS_BASE_URL = 'https://api.the-odds-api.com/v4'
BUNDESLIGA_ID = 'BL1'
CACHE_FILE = 'bundesliga_matches.json'
ODDS_CACHE_FILE = 'bundesliga_odds.json'
HISTORICAL_ODDS_CACHE_FILE = 'bundesliga_historical_odds.csv'

def fetch_competition_matches(competition_id, season=None):
    """Fetch matches from a competition."""
    headers = {'X-Auth-Token': FOOTBALL_API_KEY} if FOOTBALL_API_KEY != 'DEMO' else {}

    if season:
        url = f"{BASE_URL}/competitions/{competition_id}/matches?season={season}"
    else:
        url = f"{BASE_URL}/competitions/{competition_id}/matches"

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching matches: {e}")
        return None

def fetch_all_historical_data():
    """Fetch all available historical data."""
    current_year = datetime.now().year
    current_month = datetime.now().month

    if current_month >= 8:
        current_season = current_year
    else:
        current_season = current_year - 1

    # Try to fetch last 8 seasons
    num_seasons = 8
    seasons_data = []

    print("=" * 70)
    print("Fetching Bundesliga Historical Data")
    print("=" * 70)
    print()

    for i in range(num_seasons - 1, -1, -1):
        season = current_season - i
        print(f"Fetching data for season {season}/{season+1}...")
        data = fetch_competition_matches(BUNDESLIGA_ID, season)
        if data and 'matches' in data:
            seasons_data.extend(data['matches'])
            print(f"   ✓ Retrieved {len(data['matches'])} matches")
        else:
            print(f"   ✗ No data available for this season")

    return seasons_data

def fetch_historical_odds_from_football_data():
    """Fetch historical betting odds from football-data.co.uk CSV files."""
    print("\n" + "=" * 70)
    print("Fetching Historical Betting Odds from football-data.co.uk")
    print("=" * 70)
    print()

    # Determine which seasons to fetch (last 8 seasons)
    current_year = datetime.now().year
    current_month = datetime.now().month

    if current_month >= 8:
        current_season = current_year
    else:
        current_season = current_year - 1

    num_seasons = 8
    all_odds_data = []

    for i in range(num_seasons - 1, -1, -1):
        season_start = current_season - i
        season_end = season_start + 1

        # Format: D1.csv for Bundesliga Division 1
        # football-data.co.uk uses two-digit years: e.g., 2324 for 2023/2024
        season_code = f"{str(season_start)[-2:]}{str(season_end)[-2:]}"
        url = f"https://www.football-data.co.uk/mmz4281/{season_code}/D1.csv"

        try:
            print(f"Fetching odds for season {season_start}/{season_end}...")
            response = requests.get(url)
            response.raise_for_status()

            # Read CSV data
            df = pd.read_csv(io.StringIO(response.text))

            # Add season identifier
            df['Season'] = f"{season_start}/{season_end}"

            all_odds_data.append(df)
            print(f"   ✓ Retrieved {len(df)} matches with odds")

        except requests.exceptions.RequestException as e:
            print(f"   ✗ Error fetching season {season_start}/{season_end}: {e}")
            continue
        except Exception as e:
            print(f"   ✗ Error parsing CSV for {season_start}/{season_end}: {e}")
            continue

    if all_odds_data:
        combined_df = pd.concat(all_odds_data, ignore_index=True)
        print(f"\n   ✓ Total matches with odds: {len(combined_df)}")
        return combined_df
    else:
        print("   ✗ No historical odds data retrieved")
        return None

def fetch_odds_for_bundesliga():
    """Fetch current odds for upcoming Bundesliga matches from The Odds API."""
    if not ODDS_API_KEY:
        print("\n⚠️  No ODDS_API_KEY found in .env file")
        print("   Sign up at https://the-odds-api.com to get a free API key")
        print("   Add it to your .env file as: ODDS_API_KEY=your_key_here")
        return None

    print("\n" + "=" * 70)
    print("Fetching Betting Odds from The Odds API")
    print("=" * 70)
    print()

    url = f"{ODDS_BASE_URL}/sports/soccer_germany_bundesliga/odds/"
    params = {
        'apiKey': ODDS_API_KEY,
        'regions': 'eu',  # European bookmakers
        'markets': 'h2h',  # Head-to-head (match winner) market
        'oddsFormat': 'decimal'
    }

    try:
        print("Fetching current Bundesliga odds...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        odds_data = response.json()

        print(f"   ✓ Retrieved odds for {len(odds_data)} upcoming matches")
        print(f"   Remaining requests: {response.headers.get('x-requests-remaining', 'N/A')}")

        return odds_data
    except requests.exceptions.RequestException as e:
        print(f"   ✗ Error fetching odds: {e}")
        return None

def save_to_cache(matches_data):
    """Save match data to local JSON file."""
    cache_data = {
        'fetched_at': datetime.now().isoformat(),
        'total_matches': len(matches_data),
        'matches': matches_data
    }

    with open(CACHE_FILE, 'w') as f:
        json.dump(cache_data, f, indent=2)

    print()
    print("=" * 70)
    print(f"✓ Data saved to {CACHE_FILE}")
    print(f"  Total matches: {len(matches_data)}")
    print(f"  Fetched at: {cache_data['fetched_at']}")
    print("=" * 70)

def save_odds_to_cache(odds_data):
    """Save odds data to local JSON file."""
    if not odds_data:
        return

    cache_data = {
        'fetched_at': datetime.now().isoformat(),
        'total_matches': len(odds_data),
        'odds': odds_data
    }

    with open(ODDS_CACHE_FILE, 'w') as f:
        json.dump(cache_data, f, indent=2)

    print()
    print("=" * 70)
    print(f"✓ Odds data saved to {ODDS_CACHE_FILE}")
    print(f"  Total matches with odds: {len(odds_data)}")
    print(f"  Fetched at: {cache_data['fetched_at']}")
    print("=" * 70)

def save_historical_odds_to_cache(historical_odds_df):
    """Save historical odds DataFrame to CSV file."""
    if historical_odds_df is None or historical_odds_df.empty:
        return

    historical_odds_df.to_csv(HISTORICAL_ODDS_CACHE_FILE, index=False)

    print()
    print("=" * 70)
    print(f"✓ Historical odds saved to {HISTORICAL_ODDS_CACHE_FILE}")
    print(f"  Total matches with odds: {len(historical_odds_df)}")
    print(f"  Fetched at: {datetime.now().isoformat()}")
    print("=" * 70)

def normalize_team_name(name):
    """Normalize team name for matching by removing special characters and common prefixes/suffixes."""
    import unicodedata

    # Convert to lowercase
    name = name.lower()

    # Remove accents and special characters
    name = unicodedata.normalize('NFKD', name)
    name = ''.join([c for c in name if not unicodedata.combining(c)])

    # Handle city name variations (München/Munich, etc.)
    city_variations = {
        'munchen': 'munich',
        'munich': 'munich',
        'koln': 'cologne',
        'cologne': 'cologne',
    }

    # Apply city variations
    for german, english in city_variations.items():
        if german in name:
            name = name.replace(german, english)

    # Remove common prefixes and suffixes
    prefixes = ['fc', 'sv', 'tsg', 'vfl', 'bv', 'sc', '1.', 'bor.', 'fsv']
    suffixes = ['1910', '1846', '1899', '1904', '04', '05', '06', '09', 'ev', 'e.v.']

    words = name.split()
    filtered_words = []
    for word in words:
        # Skip if it's a prefix or suffix
        if word in prefixes or word in suffixes:
            continue
        # Remove dots
        word = word.replace('.', '')
        if word:
            filtered_words.append(word)

    return ' '.join(filtered_words)

def find_matching_odds(match_home, match_away, odds_data):
    """Find the odds data entry that best matches the given match teams."""
    match_home_norm = normalize_team_name(match_home)
    match_away_norm = normalize_team_name(match_away)

    for odds_match in odds_data:
        odds_home_norm = normalize_team_name(odds_match['home_team'])
        odds_away_norm = normalize_team_name(odds_match['away_team'])

        # Check if normalized names match (allowing partial matches)
        home_match = (match_home_norm in odds_home_norm or odds_home_norm in match_home_norm or
                     all(word in odds_home_norm for word in match_home_norm.split() if len(word) > 3))
        away_match = (match_away_norm in odds_away_norm or odds_away_norm in match_away_norm or
                     all(word in odds_away_norm for word in match_away_norm.split() if len(word) > 3))

        if home_match and away_match:
            return odds_match

    return None

def match_historical_odds_to_matches(matches_data, historical_odds_df):
    """Match historical odds data to match data."""
    if historical_odds_df is None or historical_odds_df.empty:
        return

    print("\n" + "=" * 70)
    print("Matching Historical Odds to Matches")
    print("=" * 70)
    print()

    # Get finished matches
    finished_matches = [m for m in matches_data if m['status'] == 'FINISHED']
    print(f"Finished matches from football-data.org: {len(finished_matches)}")
    print(f"Matches with odds from football-data.co.uk: {len(historical_odds_df)}")
    print()

    # Sample match check
    matched = 0
    for match in finished_matches[:10]:  # Show first 10 as examples
        home_team = match['homeTeam']['name']
        away_team = match['awayTeam']['name']
        match_date = match['utcDate'][:10]  # YYYY-MM-DD

        # Normalize team names for matching
        home_norm = normalize_team_name(home_team)
        away_norm = normalize_team_name(away_team)

        # Try to find matching row in odds data
        found = False
        for _, odds_row in historical_odds_df.iterrows():
            odds_home_norm = normalize_team_name(str(odds_row.get('HomeTeam', '')))
            odds_away_norm = normalize_team_name(str(odds_row.get('AwayTeam', '')))

            # Check if teams match
            home_match = (home_norm in odds_home_norm or odds_home_norm in home_norm or
                         all(word in odds_home_norm for word in home_norm.split() if len(word) > 3))
            away_match = (away_norm in odds_away_norm or odds_away_norm in away_norm or
                         all(word in odds_away_norm for word in away_norm.split() if len(word) > 3))

            if home_match and away_match:
                matched += 1
                found = True

                # Show available odds columns
                odds_cols = [col for col in odds_row.index if any(x in col for x in ['H', 'D', 'A']) and
                            col not in ['HomeTeam', 'AwayTeam', 'FTR', 'HTR', 'FTHG', 'FTAG', 'HTHG', 'HTAG']]

                # Get average odds if available
                if 'AvgH' in odds_row and pd.notna(odds_row.get('AvgH')):
                    print(f"✓ {home_team} vs {away_team} ({match_date})")
                    print(f"  Matched with: {odds_row.get('HomeTeam')} vs {odds_row.get('AwayTeam')}")
                    print(f"  Avg Odds - Home: {odds_row.get('AvgH', 'N/A'):.2f}, "
                          f"Draw: {odds_row.get('AvgD', 'N/A'):.2f}, "
                          f"Away: {odds_row.get('AvgA', 'N/A'):.2f}")
                elif 'B365H' in odds_row and pd.notna(odds_row.get('B365H')):
                    print(f"✓ {home_team} vs {away_team} ({match_date})")
                    print(f"  Matched with: {odds_row.get('HomeTeam')} vs {odds_row.get('AwayTeam')}")
                    print(f"  Bet365 Odds - Home: {odds_row.get('B365H', 'N/A'):.2f}, "
                          f"Draw: {odds_row.get('B365D', 'N/A'):.2f}, "
                          f"Away: {odds_row.get('B365A', 'N/A'):.2f}")
                break

        if not found and matched < 5:  # Only show first few mismatches
            print(f"✗ {home_team} vs {away_team} - No matching odds found")

    print(f"\n✓ Matched {matched} out of 10 sample matches")
    print(f"  This suggests approximately {(matched/10)*100:.0f}% match rate")
    print("=" * 70)

def extract_consistent_odds(odds_match):
    """
    Extract odds from bookmakers that are also in football-data.co.uk CSVs.
    Returns average odds across these bookmakers for consistency with training data.
    """
    # Bookmakers that appear in BOTH sources
    consistent_bookmakers = {
        'pinnacle': 'Pinnacle',
        'williamhill': 'William Hill',
        'onexbet': '1xBet',
        'betfair_ex_eu': 'Betfair'
    }

    home_odds = []
    draw_odds = []
    away_odds = []
    bookmaker_details = []

    for bookmaker in odds_match.get('bookmakers', []):
        if bookmaker['key'] in consistent_bookmakers:
            market = bookmaker['markets'][0]
            outcomes = {o['name']: o['price'] for o in market['outcomes']}

            home_odds.append(outcomes.get(odds_match['home_team']))
            draw_odds.append(outcomes.get('Draw'))
            away_odds.append(outcomes.get(odds_match['away_team']))
            bookmaker_details.append(consistent_bookmakers[bookmaker['key']])

    if home_odds and draw_odds and away_odds:
        # Calculate average (matching AvgH, AvgD, AvgA from CSV)
        avg_home = sum(home_odds) / len(home_odds)
        avg_draw = sum(draw_odds) / len(draw_odds)
        avg_away = sum(away_odds) / len(away_odds)
        return avg_home, avg_draw, avg_away, bookmaker_details

    return None, None, None, []

def match_odds_to_matches(matches_data, odds_data):
    """Check how many matches can be mapped to odds data."""
    if not odds_data:
        return

    print("\n" + "=" * 70)
    print("Checking Odds Matching for Upcoming Matches")
    print("=" * 70)
    print()

    # Get upcoming/scheduled matches from match data
    upcoming_matches = [m for m in matches_data if m['status'] in ['SCHEDULED', 'TIMED']]
    print(f"Upcoming matches from football-data.org: {len(upcoming_matches)}")
    print(f"Matches with odds from The Odds API: {len(odds_data)}")
    print()

    # Try to match by team names - check ALL upcoming matches
    matched = 0
    matched_with_consistent_odds = 0
    shown_examples = 0
    max_examples = 10  # Show first 10 as examples

    for match in upcoming_matches:
        home_team = match['homeTeam']['name']
        away_team = match['awayTeam']['name']

        # Look for corresponding odds using improved matching
        odds_match = find_matching_odds(home_team, away_team, odds_data)

        if odds_match:
            matched += 1

            # Extract odds from consistent bookmakers
            avg_home, avg_draw, avg_away, bookmakers = extract_consistent_odds(odds_match)

            if avg_home:
                matched_with_consistent_odds += 1
                if shown_examples < max_examples:
                    print(f"✓ {home_team} vs {away_team}")
                    print(f"  Matched with: {odds_match['home_team']} vs {odds_match['away_team']}")
                    print(f"  Avg Odds ({len(bookmakers)} bookmakers): Home: {avg_home:.2f}, "
                          f"Draw: {avg_draw:.2f}, Away: {avg_away:.2f}")
                    print(f"  Bookmakers: {', '.join(bookmakers)}")
                    shown_examples += 1
            else:
                if shown_examples < max_examples:
                    print(f"⚠ {home_team} vs {away_team} - Matched but no consistent bookmaker odds")
                    shown_examples += 1
        else:
            if shown_examples < max_examples:
                print(f"✗ {home_team} vs {away_team} - No matching odds found")
                shown_examples += 1

    if shown_examples >= max_examples:
        print(f"\n... (showing first {max_examples} matches)")

    print(f"\nMatched {matched} out of {len(upcoming_matches)} total upcoming matches")
    print(f"Consistent bookmaker odds available: {matched_with_consistent_odds}/{matched}")
    print(f"\nNote: Using odds from Pinnacle, William Hill, 1xBet, Betfair")
    print(f"      (same bookmakers as in historical training data)")
    print("=" * 70)

def main():
    """Main execution function."""
    # Fetch match data
    matches_data = fetch_all_historical_data()

    if not matches_data:
        print("\nError: Could not fetch any match data.")
        return

    # Save match data to cache
    save_to_cache(matches_data)

    # Fetch historical betting odds from football-data.co.uk
    historical_odds_df = fetch_historical_odds_from_football_data()

    # Save historical odds to cache
    save_historical_odds_to_cache(historical_odds_df)

    # Check mapping between historical matches and odds
    match_historical_odds_to_matches(matches_data, historical_odds_df)

    # Fetch current odds data for upcoming matches
    odds_data = fetch_odds_for_bundesliga()

    # Save current odds data to cache
    save_odds_to_cache(odds_data)

    # Check mapping between upcoming matches and current odds
    match_odds_to_matches(matches_data, odds_data)

    print()
    print("=" * 70)
    print("✓ Data fetching complete!")
    print("=" * 70)
    print()
    print("You can now run the other scripts without making API calls:")
    print("  - python score_prediction.py")
    print("  - python backtest_season.py")
    print("  - python bundesliga_predictor.py")
    print()
    print("Data files created:")
    print(f"  - {CACHE_FILE} (match results)")
    print(f"  - {HISTORICAL_ODDS_CACHE_FILE} (historical betting odds)")
    print(f"  - {ODDS_CACHE_FILE} (current odds for upcoming matches)")
    print("=" * 70)

if __name__ == "__main__":
    main()
