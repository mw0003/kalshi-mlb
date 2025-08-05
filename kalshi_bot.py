"""
Multi-Sport Betting Bot with Composite Odds

Required Environment Variables:
- KALSHI_API_KEY: Your Kalshi API key
- KALSHI_RSA_PRIVATE_KEY: Your Kalshi RSA private key in PEM format
- ODDS_API_KEY: Your The Odds API key

Example usage:
export KALSHI_API_KEY="your-kalshi-api-key"
export KALSHI_RSA_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----"
export ODDS_API_KEY="your-odds-api-key"
python kalshi_bot.py
"""

import requests
import pandas as pd
import pytz
import statsapi
from datetime import datetime, timedelta
import base64
import json
import time as time_module
import uuid
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
from IPython.display import display
import os


# 🔐 Kalshi Credentials
API_KEY = os.getenv("KALSHI_API_KEY", "your-kalshi-api-key-here")
RSA_PRIVATE_KEY_PEM = os.getenv("KALSHI_RSA_PRIVATE_KEY", """-----BEGIN RSA PRIVATE KEY-----
your-rsa-private-key-here
-----END RSA PRIVATE KEY-----""")

# 🔑 Load RSA Key
private_key = serialization.load_pem_private_key(
    RSA_PRIVATE_KEY_PEM.encode(), password=None, backend=default_backend()
)

def sign_request(method, path, key_id, private_key):
    timestamp = str(int(time_module.time() * 1000))
    message = timestamp + method + path.split('?')[0]
    signature = private_key.sign(
        message.encode('utf-8'),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256()
    )
    return {
        "Content-Type": "application/json",
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode('utf-8'),
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
    }

# 💰 Get Kalshi Balance
def get_balance():
    path = "/trade-api/v2/portfolio/balance"
    url = f"https://api.elections.kalshi.com{path}"
    headers = sign_request("GET", path, API_KEY, private_key)
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json().get("balance", 0) / 100

# 🕒 Date & bankroll
eastern = pytz.timezone('US/Eastern')
today = datetime.now(eastern).date()

def get_or_cache_daily_bankroll():
    filename = "/home/walkwalkm1/bankroll_cache.json"
    today_str = str(today)
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
        if today_str in data:
            return data[today_str]
    bankroll = get_balance()
    data = {}
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
    data[today_str] = bankroll
    with open(filename, "w") as f:
        json.dump(data, f)
    return bankroll

bankroll = get_or_cache_daily_bankroll()
print(f"\n💵 Kalshi Balance: ${bankroll:.2f}")

# 📈 Kalshi & FanDuel data
def fetch_kalshi_mlb_odds_active_only():
    url = "https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KXMLBGAME"
    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    rows = []
    for market in data.get("markets", []):
        if market.get("status") != "active":
            continue
        parts = market.get("ticker", "").split('-')
        if len(parts) < 3:
            continue
        rows.append({
            "Market Ticker": market.get("ticker"),
            "Game Title": market.get("title"),
            "Team": parts[-1],
            "Kalshi YES Ask (¢)": market.get("yes_ask")
        })
    return pd.DataFrame(rows)

def fetch_composite_odds(api_key, sport="baseball_mlb"):
    print(f"📡 Starting odds fetch for sport: {sport}")
    url = f'https://api.the-odds-api.com/v4/sports/{sport}/odds'
    params = {
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "bookmakers": "fanduel,pinnacle,draftkings",
        "apiKey": api_key
    }
    
    print(f"🌐 Making API request to: {url}")
    print(f"📋 Request params: {params}")
    
    response = requests.get(url, params=params)
    print(f"📊 API response status: {response.status_code}")
    response.raise_for_status()
    
    sportsbook_odds = {"fanduel": {}, "pinnacle": {}, "draftkings": {}}
    bookmaker_counts = {"fanduel": 0, "pinnacle": 0, "draftkings": 0}
    
    games_data = response.json()
    print(f"🎯 Received {len(games_data)} games from API")
    
    for i, game in enumerate(games_data):
        print(f"🏟️ Processing game {i+1}: {game.get('away_team', 'Unknown')} @ {game.get('home_team', 'Unknown')}")
        
        start_time = pd.to_datetime(game.get("commence_time"), utc=True).tz_convert('US/Eastern')
        if start_time.date() != today:
            print(f"⏭️ Skipping game - not today ({start_time.date()})")
            continue
            
        now = datetime.now(eastern)
        time_until_start = (start_time - now).total_seconds() / 3600
        if time_until_start > 1 and time_until_start > 0:
            print(f"⏰ Skipping game - starts in {time_until_start:.1f} hours (>1 hour)")
            continue
            
        game_bookmakers = []
        for bookmaker in game.get("bookmakers", []):
            book_key = bookmaker["key"]
            game_bookmakers.append(book_key)
            if book_key in sportsbook_odds:
                if book_key in bookmaker_counts:
                    bookmaker_counts[book_key] += 1
                    
                market = next((m for m in bookmaker.get("markets", []) if m["key"] == "h2h"), None)
                if market:
                    for outcome in market.get("outcomes", []):
                        team = outcome["name"].strip().replace("Oakland Athletics", "Athletics")
                        sportsbook_odds[book_key][team] = outcome["price"]
                        print(f"💰 {book_key}: {team} = {outcome['price']}")
        
        print(f"📚 Bookmakers for this game: {game_bookmakers}")
    
    print(f"📊 Bookmaker coverage: {bookmaker_counts}")
    print(f"✅ Processed odds successfully")
    return sportsbook_odds

def build_opponent_map_with_timing():
    games = statsapi.schedule(start_date=str(today), end_date=str(today))
    matchup = {}
    game_timing = {}
    
    for game in games:
        away = game['away_name'].replace("Oakland Athletics", "Athletics")
        home = game['home_name'].replace("Oakland Athletics", "Athletics")
        matchup[away] = home
        matchup[home] = away
        
        game_time = pd.to_datetime(game['game_datetime']).tz_convert('US/Eastern')
        now = datetime.now(eastern)
        time_until_start = (game_time - now).total_seconds() / 3600  # hours
        
        is_eligible = (game.get('status') in ['In Progress', 'Live'] or 
                      (time_until_start <= 1 and time_until_start >= 0))
        
        game_timing[away] = is_eligible
        game_timing[home] = is_eligible
    
    return matchup, game_timing


def american_to_implied_prob(odds):
    if pd.isna(odds):
        return None
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

def devig_sportsbook_odds(odds_dict, opponent_map):
    """
    Remove vig from moneyline odds using normalization method.
    """
    devigged_odds = {}
    processed_games = set()
    
    for team, american_odds in odds_dict.items():
        if pd.isna(american_odds) or team in processed_games:
            continue
            
        opponent = opponent_map.get(team)
        if not opponent or opponent not in odds_dict:
            continue
            
        opponent_odds = odds_dict[opponent]
        if pd.isna(opponent_odds):
            continue
        
        team_implied_prob = american_to_implied_prob(american_odds)
        opponent_implied_prob = american_to_implied_prob(opponent_odds)
        
        if team_implied_prob is None or opponent_implied_prob is None:
            continue
        
        total_implied_prob = team_implied_prob + opponent_implied_prob
        
        team_fair_prob = team_implied_prob / total_implied_prob
        opponent_fair_prob = opponent_implied_prob / total_implied_prob
        
        devigged_odds[team] = 1 / team_fair_prob
        devigged_odds[opponent] = 1 / opponent_fair_prob
        
        processed_games.add(team)
        processed_games.add(opponent)
    
    return devigged_odds

def devig_composite_odds(sportsbook_odds, opponent_map, weights={"pinnacle": 0.5, "fanduel": 0.25, "draftkings": 0.25}):
    """
    Create composite devigged odds from multiple sportsbooks with weighted averages.
    """
    print(f"🧮 Starting composite odds devigging...")
    print(f"📊 Processing sportsbooks: {list(sportsbook_odds.keys())}")
    print(f"⚖️ Using bookmaker weights: {weights}")
    
    devigged_books = {}
    for book, odds_dict in sportsbook_odds.items():
        if odds_dict:
            print(f"📚 Devigging {book} with {len(odds_dict)} teams")
            devigged_books[book] = devig_sportsbook_odds(odds_dict, opponent_map)
            print(f"✅ {book} devigged: {len(devigged_books[book])} teams")
        else:
            print(f"⚠️ No odds data for {book}")
    
    composite_odds = {}
    all_teams = set()
    for book_odds in devigged_books.values():
        all_teams.update(book_odds.keys())
    
    print(f"👥 Processing {len(all_teams)} unique teams across all books")
    
    successful_composites = 0
    for team in all_teams:
        print(f"🏷️ Processing team: {team}")
        weighted_prob = 0
        total_weight = 0
        
        for book, book_odds in devigged_books.items():
            if team in book_odds and book in weights:
                prob = 1 / book_odds[team]
                weighted_prob += prob * weights[book]
                total_weight += weights[book]
                print(f"   📊 {book}: prob={prob:.4f}, weight={weights[book]}")
        
        if total_weight > 0:
            final_prob = weighted_prob / total_weight
            composite_odds[team] = 1 / final_prob
            successful_composites += 1
            print(f"🎯 Final composite odds for {team}: {composite_odds[team]:.4f}")
        else:
            print(f"❌ No valid odds for {team}")
    
    print(f"📈 Composite devigging summary: {successful_composites}/{len(all_teams)} teams processed successfully")
    return composite_odds

# 🔢 Kelly

def kelly_wager(fair_odds, your_odds, bankroll):
    try:
        if pd.isna(fair_odds) or pd.isna(your_odds):
            return 0
        return max((((1 / fair_odds) * (your_odds - 1)) - (1 - (1 / fair_odds))) / (your_odds - 1) * bankroll, 0)
    except:
        return 0

# Team abbreviation maps for different sports
mlb_team_abbr_to_name = {
    "ATL": "Atlanta Braves", "AZ": "Arizona Diamondbacks", "DET": "Detroit Tigers", "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox", "CLE": "Cleveland Guardians", "KC": "Kansas City Royals", "HOU": "Houston Astros",
    "NYM": "New York Mets", "WSH": "Washington Nationals", "PHI": "Philadelphia Phillies", "CHC": "Chicago Cubs",
    "SF": "San Francisco Giants", "TEX": "Texas Rangers", "SEA": "Seattle Mariners", "MIA": "Miami Marlins",
    "CWS": "Chicago White Sox", "MIN": "Minnesota Twins", "LAA": "Los Angeles Angels", "NYY": "New York Yankees",
    "TOR": "Toronto Blue Jays", "PIT": "Pittsburgh Pirates", "LAD": "Los Angeles Dodgers",
    "MIL": "Milwaukee Brewers", "STL": "St. Louis Cardinals", "COL": "Colorado Rockies",
    "CIN": "Cincinnati Reds", "SD": "San Diego Padres", "TB": "Tampa Bay Rays", "OAK": "Oakland Athletics", "ATH": "Athletics"
}

nfl_team_abbr_to_name = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers", "CHI": "Chicago Bears", "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
    "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers", "LAR": "Los Angeles Rams", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SF": "San Francisco 49ers",
    "SEA": "Seattle Seahawks", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders"
}

wnba_team_abbr_to_name = {
    "ATL": "Atlanta Dream", "CHI": "Chicago Sky", "CONN": "Connecticut Sun", "DAL": "Dallas Wings",
    "IND": "Indiana Fever", "LV": "Las Vegas Aces", "MIN": "Minnesota Lynx", "NY": "New York Liberty",
    "PHX": "Phoenix Mercury", "SEA": "Seattle Storm", "WAS": "Washington Mystics", "LA": "Los Angeles Sparks"
}

team_abbr_to_name = mlb_team_abbr_to_name

def fetch_sport_opportunities(sport, api_key):
    """Fetch opportunities for a specific sport"""
    print(f"🏆 fetch_sport_opportunities called for sport: {sport}")
    
    tournament_configs = {
        "wta": "KXWTAMATCH",
        "atp": "KXATPMATCH",
        "mls": "KXMLSGAME",
    }
    
    sport_configs = {
        "mlb": {
            "api_sport": "baseball_mlb",
            "kalshi_series": "KXMLBGAME",
            "team_map": mlb_team_abbr_to_name,
            "market_type": "2way"
        },
        "nfl": {
            "api_sport": "americanfootball_nfl", 
            "kalshi_series": "KXNFLGAME",
            "team_map": nfl_team_abbr_to_name,
            "market_type": "2way"
        },
        "wnba": {
            "api_sport": "basketball_wnba",
            "kalshi_series": "KXWNBAGAME", 
            "team_map": wnba_team_abbr_to_name,
            "market_type": "2way"
        },
        "tennis_wta": {
            "api_sport": "tennis_wta",
            "kalshi_series": tournament_configs["wta"],
            "team_map": {},
            "market_type": "2way"
        },
        "tennis_atp": {
            "api_sport": "tennis_atp", 
            "kalshi_series": tournament_configs["atp"],
            "team_map": {},
            "market_type": "2way"
        },
        "soccer_mls": {
            "api_sport": "soccer_usa_mls",
            "kalshi_series": tournament_configs["mls"],
            "team_map": {},
            "market_type": "3way"
        }
    }
    
    print(f"📋 Available sport configs: {list(sport_configs.keys())}")
    print(f"🎯 Tournament configs available: {list(tournament_configs.keys())}")
    
    if sport not in sport_configs:
        print(f"❌ Sport '{sport}' not found in configurations")
        return pd.DataFrame()
    
    config = sport_configs[sport]
    print(f"⚙️ Using config for {sport}: {config}")
    print(f"🎮 Market type: {config['market_type']}")
    
    print(f"📊 Fetching Kalshi odds for {sport}...")
    if sport == "mlb":
        kalshi_df = fetch_kalshi_mlb_odds_active_only()
    else:
        kalshi_df = fetch_kalshi_sport_odds(config["kalshi_series"])
    
    print(f"📈 Kalshi DataFrame shape: {kalshi_df.shape}")
    if kalshi_df.empty:
        print(f"⚠️ No Kalshi data found for {sport}")
        return pd.DataFrame()
    
    print(f"🔍 Pre-filtering markets to optimize API usage...")
    already_bet_teams = get_already_bet_teams()
    kalshi_df, filtered_count = filter_kalshi_markets_by_existing_bets(kalshi_df, already_bet_teams)
    
    if kalshi_df.empty:
        print(f"⚠️ No remaining markets after filtering already bet teams for {sport}")
        return pd.DataFrame()
    
    print(f"📊 Remaining markets after pre-filtering: {len(kalshi_df)} (saved {filtered_count} API calls)")
    
    print(f"🎯 Checking API call limit before fetching sportsbook odds...")
    if count_api_call():
        print(f"📡 Fetching composite odds from sportsbooks for {config['api_sport']}...")
        sportsbook_odds = fetch_composite_odds(api_key, config["api_sport"])
        print(f"💰 Sportsbook odds keys: {list(sportsbook_odds.keys()) if sportsbook_odds else 'None'}")
        
        opponent_map = {}
        print(f"🔗 Building opponent map for {len(kalshi_df)} teams/players...")
        
        if sport.startswith("tennis"):
            print(f"🎾 Tennis sport detected: {sport} - using dynamic name matching")
            tennis_team_map = build_tennis_team_map(sportsbook_odds, kalshi_df)
            config["team_map"] = tennis_team_map
            print(f"🎯 Updated tennis team map with {len(tennis_team_map)} matched players")
        
        for team1, team2 in zip(kalshi_df["Team"], kalshi_df["Team"]):
            pass
        
        print(f"🧮 Devigging composite odds using {config['market_type']} logic...")
        if config["market_type"] == "3way":
            print(f"⚽ Using 3-way devigging for soccer markets")
            composite_odds = devig_soccer_odds(sportsbook_odds)
        else:
            print(f"🏀 Using 2-way devigging for standard markets")
            composite_odds = devig_composite_odds(sportsbook_odds, opponent_map)
        print(f"✅ Composite odds calculated: {len(composite_odds)} entries")
    else:
        print(f"🚫 API call limit reached, skipping sportsbook odds fetch")
        composite_odds = {}
    
    print(f"🏷️ Adding sport metadata and calculations...")
    kalshi_df["Sport"] = sport.upper()
    kalshi_df["Team Name"] = kalshi_df["Team"].map(config["team_map"]) if config["team_map"] else kalshi_df["Team"]
    kalshi_df["Composite Fair Odds"] = kalshi_df["Team Name"].map(composite_odds)
    
    kalshi_df["Kalshi %"] = kalshi_df["Kalshi YES Ask (¢)"] / 100
    kalshi_df["Decimal Odds (Kalshi)"] = 1 / kalshi_df["Kalshi %"]
    
    print(f"📊 Calculating raw edge for {len(kalshi_df)} opportunities...")
    raw_edge = kalshi_df["Decimal Odds (Kalshi)"] * (1 / kalshi_df["Composite Fair Odds"]) - 1
    kalshi_df["% Edge"] = raw_edge.apply(lambda x: f"{round(x * 100, 1)}%" if pd.notna(x) else None)
    kalshi_df["numeric_edge"] = raw_edge
    
    print(f"🎯 Returning {len(kalshi_df)} opportunities for {sport}")
    return kalshi_df

def fetch_kalshi_sport_odds(series_ticker):
    """Generic function to fetch Kalshi odds for any sport series"""
    url = f"https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker={series_ticker}"
    headers = {"accept": "application/json"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        rows = []
        for market in data.get("markets", []):
            if market.get("status") != "active":
                continue
            parts = market.get("ticker", "").split('-')
            if len(parts) < 3:
                continue
            rows.append({
                "Market Ticker": market.get("ticker"),
                "Game Title": market.get("title"),
                "Team": parts[-1],
                "Kalshi YES Ask (¢)": market.get("yes_ask")
            })
        return pd.DataFrame(rows)
    except:
        return pd.DataFrame()

def devig_soccer_odds(odds_dict):
    """Special devigging for soccer 3-way markets (Win/Loss/Draw)"""
    devigged_odds = {}
    
    for game_id, outcomes in odds_dict.items():
        if len(outcomes) != 3:  # Should have home, away, draw
            continue
            
        total_implied_prob = 0
        implied_probs = {}
        
        for outcome, american_odds in outcomes.items():
            prob = american_to_implied_prob(american_odds)
            if prob is not None:
                implied_probs[outcome] = prob
                total_implied_prob += prob
        
        # Normalize probabilities
        if total_implied_prob > 0:
            for outcome, prob in implied_probs.items():
                fair_prob = prob / total_implied_prob
                devigged_odds[f"{game_id}_{outcome}"] = 1 / fair_prob
    
    return devigged_odds

def normalize_tennis_player_name(full_name):
    """
    Normalize tennis player name to match Kalshi format
    Tries to extract last name and convert to uppercase
    Returns None if normalization fails - implements try-match-ignore-mismatch logic
    """
    if not full_name or not isinstance(full_name, str):
        print(f"⚠️ Invalid tennis player name: {full_name}")
        return None
    
    try:
        name = full_name.strip()
        
        if "," in name:
            last_name = name.split(",")[0].strip()
            print(f"🎾 Tennis name format detected: Last,First -> '{last_name}'")
        elif ". " in name:
            parts = name.split(". ")
            last_name = parts[-1].strip() if len(parts) > 1 else name
            print(f"🎾 Tennis name format detected: Initial.Last -> '{last_name}'")
        else:
            parts = name.split()
            last_name = parts[-1] if parts else name
            print(f"🎾 Tennis name format detected: First Last -> '{last_name}'")
        
        normalized = last_name.upper()
        
        char_map = {
            'Ć': 'C', 'Č': 'C', 'Ž': 'Z', 'Š': 'S', 'Đ': 'DJ', 'Ð': 'D',
            'Ñ': 'N', 'Ü': 'U', 'Ö': 'O', 'Ä': 'A', 'É': 'E',
            'È': 'E', 'À': 'A', 'Ì': 'I', 'Ò': 'O', 'Ù': 'U'
        }
        
        for original, replacement in char_map.items():
            normalized = normalized.replace(original, replacement)
        
        print(f"🎾 Tennis name normalized: '{full_name}' → '{normalized}'")
        return normalized
        
    except Exception as e:
        print(f"❌ Error normalizing tennis name '{full_name}': {e} - ignoring mismatch")
        return None

def match_tennis_player_to_kalshi(odds_player_name, kalshi_markets):
    """
    Try to match tennis player name from odds API to Kalshi market
    Returns matching Kalshi market data or None if no match found
    Implements try-match-ignore-mismatch logic as requested
    """
    normalized_name = normalize_tennis_player_name(odds_player_name)
    if not normalized_name:
        print(f"🚫 Could not normalize tennis player name: '{odds_player_name}' - ignoring")
        return None
    
    for _, market_row in kalshi_markets.iterrows():
        kalshi_team = market_row.get("Team", "")
        if kalshi_team == normalized_name:
            print(f"✅ Tennis match found: '{odds_player_name}' → '{kalshi_team}'")
            return market_row.to_dict()
    
    print(f"❌ No Kalshi match for tennis player: '{odds_player_name}' (normalized: '{normalized_name}') - ignoring mismatch")
    return None

def build_tennis_team_map(sportsbook_odds, kalshi_df):
    """
    Build dynamic team map for tennis by attempting to match player names
    Returns dictionary mapping sportsbook names to Kalshi team codes
    Implements try-match-ignore-mismatch logic
    """
    print(f"🎾 Building tennis team map from {len(sportsbook_odds)} sportsbook entries and {len(kalshi_df)} Kalshi markets")
    
    tennis_team_map = {}
    matches_found = 0
    mismatches_ignored = 0
    
    all_player_names = set()
    for game_id, odds_data in sportsbook_odds.items():
        if isinstance(odds_data, dict):
            all_player_names.update(odds_data.keys())
    
    print(f"🔍 Found {len(all_player_names)} unique player names in sportsbook data")
    
    for player_name in all_player_names:
        match = match_tennis_player_to_kalshi(player_name, kalshi_df)
        if match:
            tennis_team_map[player_name] = match["Team"]
            matches_found += 1
        else:
            mismatches_ignored += 1
    
    print(f"🎯 Tennis team map results: {matches_found} matches found, {mismatches_ignored} mismatches ignored")
    print(f"📋 Tennis team map: {tennis_team_map}")
    
    return tennis_team_map

def get_already_bet_teams():
    """
    Get teams/players we've already bet on today to avoid duplicate API calls
    Returns set of team abbreviations from executed orders
    """
    print(f"🔍 Checking for already bet teams to optimize API calls...")
    
    try:
        executed_team_abbrs = set()
        for order in get_todays_orders():
            if (
                order.get("status") in ("executed", "resting") and
                order.get("action") == "buy" and
                order.get("ticker")
            ):
                parts = order["ticker"].split('-')
                if len(parts) >= 3:
                    abbr = parts[-1]
                    executed_team_abbrs.add(abbr)
        
        print(f"✅ Found {len(executed_team_abbrs)} already bet teams: {sorted(executed_team_abbrs)}")
        return executed_team_abbrs
        
    except Exception as e:
        print(f"⚠️ Error fetching today's orders: {e} - proceeding without pre-filtering")
        return set()

def filter_kalshi_markets_by_existing_bets(kalshi_df, already_bet_teams):
    """
    Filter out Kalshi markets for teams we've already bet on
    Returns filtered DataFrame and count of filtered markets
    """
    if kalshi_df.empty or not already_bet_teams:
        return kalshi_df, 0
    
    original_count = len(kalshi_df)
    
    filtered_df = kalshi_df[~kalshi_df["Team"].isin(already_bet_teams)].reset_index(drop=True)
    
    filtered_count = original_count - len(filtered_df)
    print(f"🚫 Pre-filtered {filtered_count} markets for already bet teams")
    
    if filtered_count > 0:
        filtered_teams = kalshi_df[kalshi_df["Team"].isin(already_bet_teams)]["Team"].tolist()
        print(f"📋 Filtered teams: {sorted(set(filtered_teams))}")
    
    return filtered_df, filtered_count

testing_mode = True  # Set to False for production

def count_api_call():
    """Track API calls for logging purposes - no artificial limits in production"""
    print(f"📞 Making API call...")
    return True

def get_eligible_kalshi_markets_count():
    """Count eligible Kalshi markets across all supported sports"""
    try:
        kalshi_df = fetch_kalshi_mlb_odds_active_only()
        return len(kalshi_df)
    except:
        return 0

def get_dynamic_kelly_multiplier():
    """Calculate dynamic Kelly multiplier based on available markets"""
    print(f"🎯 Calculating dynamic Kelly multiplier...")
    market_count = get_eligible_kalshi_markets_count()
    print(f"📊 Eligible Kalshi markets count: {market_count}")
    
    if market_count < 10:
        multiplier = 0.75
        print(f"🔥 High Kelly multiplier (75%) - few markets available ({market_count} < 10)")
    elif market_count < 20:
        multiplier = 0.65
        print(f"📈 Medium-high Kelly multiplier (65%) - moderate markets ({market_count} < 20)")
    elif market_count < 30:
        multiplier = 0.60
        print(f"📊 Medium Kelly multiplier (60%) - good market count ({market_count} < 30)")
    else:
        multiplier = 0.50
        print(f"🎯 Conservative Kelly multiplier (50%) - many markets available ({market_count} >= 30)")
    
    return multiplier

print("🚀 Starting multi-sport betting bot...")
print(f"🧪 Testing mode: {testing_mode}")

print("⚾ Fetching MLB Kalshi odds...")
kalshi_df = fetch_kalshi_mlb_odds_active_only()
print(f"📊 MLB Kalshi DataFrame shape: {kalshi_df.shape}")

print("🎯 Checking API call limit for sportsbook odds...")
if count_api_call():
    print("📡 Fetching composite sportsbook odds for MLB...")
    sportsbook_odds = fetch_composite_odds(os.getenv("ODDS_API_KEY", "your-odds-api-key-here"))
    print(f"💰 Sportsbook odds received: {len(sportsbook_odds) if sportsbook_odds else 0} games")
    
    print("🕐 Building opponent map with timing filters...")
    opponent_map, game_timing = build_opponent_map_with_timing()
    print(f"⏰ Game timing data: {len(game_timing)} games checked")
    
    eligible_teams = {team for team, is_eligible in game_timing.items() if is_eligible}
    print(f"✅ Eligible teams (starting soon/in progress): {len(eligible_teams)} - {eligible_teams}")
    
    kalshi_df = kalshi_df[kalshi_df["Team Name"].isin(eligible_teams)].reset_index(drop=True)
    print(f"🎯 Filtered Kalshi DataFrame shape: {kalshi_df.shape}")
else:
    print("🚫 API call limit reached, skipping sportsbook odds fetch")
    sportsbook_odds = {}
    opponent_map = {}
    game_timing = {}

composite_odds = devig_composite_odds(sportsbook_odds, opponent_map)

kalshi_df["Team Name"] = kalshi_df["Team"].map(team_abbr_to_name)
kalshi_df["Opponent Name"] = kalshi_df["Team Name"].map(opponent_map)

kalshi_df["Composite Fair Odds"] = kalshi_df["Team Name"].map(composite_odds)

kalshi_df["Kalshi %"] = kalshi_df["Kalshi YES Ask (¢)"] / 100
kalshi_df["Decimal Odds (Kalshi)"] = 1 / kalshi_df["Kalshi %"]

raw_edge = kalshi_df["Decimal Odds (Kalshi)"] * (1 / kalshi_df["Composite Fair Odds"]) - 1
kalshi_df["% Edge"] = raw_edge.apply(lambda x: f"{round(x * 100, 1)}%" if pd.notna(x) else None)
kalshi_df["numeric_edge"] = raw_edge

print(f"🎯 Calculating dynamic Kelly multiplier for wager sizing...")
dynamic_kelly = get_dynamic_kelly_multiplier()
print(f"💰 Using dynamic Kelly multiplier: {dynamic_kelly}")

kalshi_df["$ Wager"] = kalshi_df.apply(
    lambda row: (
        f"${round(min(kelly_wager(row['Composite Fair Odds'], row['Decimal Odds (Kalshi)'], bankroll), bankroll * 0.3))}"
        if pd.notna(row["Composite Fair Odds"]) and pd.notna(row["Decimal Odds (Kalshi)"])
        else "$0"
    ),
    axis=1
)

kalshi_df["Kalshi YES Ask (¢)"] = kalshi_df["Kalshi YES Ask (¢)"].astype(int)
kalshi_df = kalshi_df.sort_values(by="numeric_edge", ascending=False).reset_index(drop=True)

def store_odds_timeseries():
    filename = "/home/walkwalkm1/odds_timeseries.json"
    timestamp = datetime.now(eastern).isoformat()
    
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
    else:
        data = []
    
    for _, row in kalshi_df.iterrows():
        if pd.notna(row["Composite Fair Odds"]) and pd.notna(row["Kalshi %"]):
            data.append({
                "timestamp": timestamp,
                "team": row["Team Name"],
                "kalshi_implied_odds": row["Kalshi %"],
                "composite_devigged_odds": 1 / row["Composite Fair Odds"],
                "expected_value": row["numeric_edge"]
            })
    
    cutoff_date = datetime.now(eastern) - timedelta(days=7)
    data = [d for d in data if datetime.fromisoformat(d["timestamp"]) > cutoff_date]
    
    with open(filename, "w") as f:
        json.dump(data, f)

store_odds_timeseries()

kalshi_df = kalshi_df.drop(columns=["numeric_edge"])

final_df = kalshi_df[[
    "Team Name", "Opponent Name",
    "Kalshi YES Ask (¢)", "Composite Fair Odds",
    "% Edge", "$ Wager", "Market Ticker"
]]

#print("\n📊 Full Table:")
#display(final_df)

filtered_df = final_df[
    (final_df["Kalshi YES Ask (¢)"] >= 60) &
    (final_df["Kalshi YES Ask (¢)"] <= 95) &
    (final_df["% Edge"].str.replace('%', '').astype(float) >= 4) &
    (final_df["% Edge"].str.replace('%', '').astype(float) < 9.1)
].reset_index(drop=True)

seen_teams = set()
filtered_cleaned = []
for _, row in filtered_df.iterrows():
    team = row["Team Name"]
    opponent = row["Opponent Name"]
    if opponent in seen_teams:
        continue
    seen_teams.add(team)
    filtered_cleaned.append(row)

filtered_df = pd.DataFrame(filtered_cleaned).reset_index(drop=True)

# 📥 Get today's orders (Eastern Time aware)
def get_todays_orders():
    path = "/trade-api/v2/portfolio/orders"
    url = f"https://api.elections.kalshi.com{path}"
    headers = sign_request("GET", path, API_KEY, private_key)

    from datetime import time  # needed for time.min / time.max
    today_start_dt = eastern.localize(datetime.combine(today, time.min))
    today_end_dt = eastern.localize(datetime.combine(today, time.max))

    today_start = int(today_start_dt.timestamp())
    today_end = int(today_end_dt.timestamp())

    params = {"min_ts": today_start, "max_ts": today_end}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json().get("orders", [])

# 🚫 Exclude previously bet games (buy-only)
executed_team_abbrs = set()
for order in get_todays_orders():
    if (
        order.get("status") in ("executed","resting") and
        order.get("action") == "buy" and
        order.get("ticker")
    ):
        parts = order["ticker"].split('-')
        if len(parts) >= 3:
            abbr = parts[-1]
            executed_team_abbrs.add(abbr)

# 🔍 Debug print to verify which teams are being excluded
print(f"✅ Executed team abbreviations (BUY only): {sorted(executed_team_abbrs)}")

# Map to full team names
executed_team_names = {team_abbr_to_name[abbr] for abbr in executed_team_abbrs if abbr in team_abbr_to_name}
print(f"✅ Full team names we already bet on: {sorted(executed_team_names)}")

# Identify rows being filtered out
if not filtered_df.empty:
    flagged_rows = filtered_df[
        filtered_df["Team Name"].isin(executed_team_names) |
        filtered_df["Opponent Name"].isin(executed_team_names)
    ]

    print(f"\n🚫 Rows removed due to duplicate or opponent bets: {len(flagged_rows)}")
    if not flagged_rows.empty:
        display(flagged_rows[["Team Name", "Opponent Name", "Market Ticker"]])

    # Apply the actual filtering
    filtered_df = filtered_df[
        ~filtered_df["Team Name"].isin(executed_team_names) &
        ~filtered_df["Opponent Name"].isin(executed_team_names)
    ].reset_index(drop=True)

    print("\n📈 Final filtered_df after exclusions:")
    display(filtered_df)
else:
    print("⚠️ filtered_df is empty before exclusions.")

# 🛒 Submit orders
def submit_order(market_ticker, side, quantity, price):
    path = "/trade-api/v2/portfolio/orders"
    url = f"https://api.elections.kalshi.com{path}"
    headers = sign_request("POST", path, API_KEY, private_key)
    payload = {
        "action": "buy",
        "type": "limit",
        "side": side,
        "ticker": market_ticker,
        "count": quantity,
        "yes_price": price,
        "client_order_id": str(uuid.uuid4())
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

print(f"\n🛒 Processing {len(filtered_df)} potential orders...")
for i, row in filtered_df.iterrows():
    try:
        print(f"\n📋 Order {i+1}/{len(filtered_df)}: {row['Team Name']}")
        
        team_abbr = [abbr for abbr, name in team_abbr_to_name.items() if name == row["Team Name"]]
        opp_abbr = [abbr for abbr, name in team_abbr_to_name.items() if name == row["Opponent Name"]]
        print(f"🔍 Team abbreviations - Team: {team_abbr}, Opponent: {opp_abbr}")

        if (team_abbr and team_abbr[0] in executed_team_abbrs) or (opp_abbr and opp_abbr[0] in executed_team_abbrs):
            print(f"⚠️ Skipping {row['Team Name']} vs {row['Opponent Name']} — already bet.")
            continue

        ticker = row["Market Ticker"]
        team = row["Team Name"]
        base_wager = float(row["$ Wager"].strip("$") or 0)
        wager_dollars = dynamic_kelly * base_wager
        price = int(row["Kalshi YES Ask (¢)"])
        cost_per_contract = price / 100
        suggested_contracts = int(wager_dollars // cost_per_contract)
        contracts = min(suggested_contracts, int(0.2 * bankroll / cost_per_contract))
        total_cost = contracts * cost_per_contract
        
        print(f"💰 Wager calculation:")
        print(f"   Base wager: ${base_wager}")
        print(f"   Kelly multiplier: {dynamic_kelly}")
        print(f"   Adjusted wager: ${wager_dollars:.2f}")
        print(f"   Price per contract: {price}¢")
        print(f"   Suggested contracts: {suggested_contracts}")
        print(f"   Final contracts: {contracts}")
        print(f"   Total cost: ${total_cost:.2f}")

        if contracts < 1 or total_cost > bankroll:
            print(f"🚫 {team} — Not enough bankroll to place even 1 contract.")
            continue

        if testing_mode:
            print(f"🧪 TEST MODE: Would place order for {team} → {contracts} contracts at {price}¢")
            result = {"status": "test_mode"}
        else:
            print(f"🚀 LIVE MODE: Placing order for {team}")
            result = submit_order(ticker, "yes", contracts, price)
            print(f"▶️ {team} → {contracts} contracts at {price}¢ → ✅ {result}")
            bankroll -= total_cost
        
        expected_value_before = (int(row["Kalshi YES Ask (¢)"]) / 100 - 1) * 100
        expected_value_after = float(row["% Edge"].strip("%"))
        
        order_data = {
            "timestamp": datetime.now(eastern).isoformat(),
            "team": team,
            "ticker": ticker,
            "contracts": contracts,
            "price": price,
            "total_cost": total_cost,
            "expected_value_before_devig": expected_value_before,
            "expected_value_after_devig": expected_value_after
        }
        
        orders_filename = "/home/walkwalkm1/placed_orders.json"
        if os.path.exists(orders_filename):
            with open(orders_filename, "r") as f:
                orders_data = json.load(f)
        else:
            orders_data = []
        
        orders_data.append(order_data)
        
        with open(orders_filename, "w") as f:
            json.dump(orders_data, f)

    except Exception as e:
        #print(f"❌ {row['Team Name']} — Error: {e}")
        pass

