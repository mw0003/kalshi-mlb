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
import json
import os
import base64
import json
import time as time_module
import uuid
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
from IPython.display import display
import os


# --- Import Credentials ---
from credentials import KALSHI_API_KEY, KALSHI_RSA_PRIVATE_KEY, ODDS_API_KEY, BANKROLL_CACHE_PATH, PLACED_ORDERS_PATH, ODDS_TIMESERIES_PATH
NDJSON_ONLY = os.getenv("NDJSON_ONLY") == "1"

def get_available_events_cache_path():
    """Get path to available events cache file"""
    return os.path.join(os.path.dirname(__file__), "available_events_cache.json")

def store_daily_available_events():
    """Store count of available events for each sport for today"""
    try:
        cache_path = get_available_events_cache_path()
        
        cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                cache = json.load(f)
        
        eastern = pytz.timezone("US/Eastern")
        today = datetime.now(eastern).date()
        today_str = str(today)
        
        if today_str in cache:
            print(f"📅 Available events already cached for {today_str}")
            return
        
        sport_configs = {
            "MLB": {"series_ticker": "KXMLBGAME"},
            "NFL": {"series_ticker": "KXNFLGAME"},
            "NBA": {"series_ticker": "KXNBAGAME"},
            "NCAAF": {"series_ticker": "KXNCAAFGAME"},
            "NCAAB": {"series_ticker": "KXCBBGAME"},
            "NHL": {"series_ticker": "KXNHLGAME"},
            "WNBA": {"series_ticker": "KXWNBAGAME"},
            "MLS": {"series_ticker": "KXMLSGAME"},
            "EPL": {"series_ticker": "KXEPLGAME"},
            "UCL": {"series_ticker": "KXUCLGAME"},
            "EURO": {"series_ticker": "KXEUROGAME"},
            "COPA": {"series_ticker": "KXCOPAGAME"},
            "TENNIS": {"series_ticker": "KXTENNISGAME"}
        }
        
        today_events = {}
        kalshi_date_str = today.strftime("%y%b%d").upper()
        
        for sport, config in sport_configs.items():
            series_ticker = config["series_ticker"]
            try:
                url = f"https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker={series_ticker}"
                headers = {"accept": "application/json"}
                response = requests.get(url, headers=headers)
                print(f"🔍 DEBUG: {sport} - Response status: {response.status_code}")
                print(f"🔍 DEBUG: {sport} - Response headers: {response.headers}")
                response.raise_for_status()
                data = response.json()
                
                active_markets = []
                for market in data.get("markets", []):
                    if market.get("status") == "active":
                        ticker = market.get("ticker", "")
                        if kalshi_date_str in ticker:
                            active_markets.append(market)
                
                event_count = len(active_markets) // 2 if len(active_markets) > 0 else 0
                today_events[sport] = event_count
                print(f"📅 {sport}: {event_count} events available for {kalshi_date_str}")
                
            except Exception as e:
                print(f"❌ Error fetching {sport} events: {e}")
                today_events[sport] = 0
        
        cache[today_str] = today_events
        
        with open(cache_path, 'w') as f:
            json.dump(cache, f, indent=2)
        
        print(f"✅ Stored available events for {today_str}")
        
    except Exception as e:
        print(f"❌ Error storing daily available events: {e}")

def get_ndjson_path(sport_code, dt):
    day = dt.strftime("%Y-%m-%d")
    base_dir = os.path.join("odds_timeseries", str(sport_code).upper())
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{day}.ndjson")

def write_ndjson_line(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


# 🔑 Load RSA Key
private_key = serialization.load_pem_private_key(
    KALSHI_RSA_PRIVATE_KEY.encode(), password=None, backend=default_backend()
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
    headers = sign_request("GET", path, KALSHI_API_KEY, private_key)
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json().get("balance", 0) / 100

# 🕒 Date & bankroll
eastern = pytz.timezone('US/Eastern')
today = datetime.now(eastern).date()
now = datetime.now(eastern)

def get_or_cache_daily_bankroll():
    filename = BANKROLL_CACHE_PATH
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
    """Fetch MLB odds with today-only filtering"""
    return fetch_kalshi_sport_odds("KXMLBGAME")

def extract_event_id(ticker):
    """Extract event ID from ticker by removing team abbreviation
    Example: KXMLBGAME-25AUG17PITCHC-CHC -> KXMLBGAME-25AUG17PITCHC
    """
    if not ticker:
        return ""
    parts = ticker.split('-')
    if len(parts) >= 3:
        return '-'.join(parts[:-1])
    return ticker

def build_opponent_map():
    """Build opponent mapping for MLB using statsapi"""
    games = statsapi.schedule(start_date=str(today), end_date=str(today))
    matchup = {}
    for game in games:
        away = game['away_name'].replace("Oakland Athletics", "Athletics")
        home = game['home_name'].replace("Oakland Athletics", "Athletics")
        matchup[away] = home
        matchup[home] = away
    return matchup

def fetch_composite_odds(api_key, sport="baseball_mlb"):
    url = f'https://api.the-odds-api.com/v4/sports/{sport}/odds'
    params = {
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "bookmakers": "fanduel,draftkings,betmgm,caesars,espnbet",
        "apiKey": api_key
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    sportsbook_odds = {"fanduel": {}, "draftkings": {}, "betmgm": {}, "caesars": {}, "espnbet": {}}
    opponent_map = {}
    
    games_data = response.json()
    today_games = 0
    
    for game in games_data:
        start_time = pd.to_datetime(game.get("commence_time"), utc=True).tz_convert('US/Eastern')
        if start_time.date() != today:
            continue
            
        today_games += 1
        away_team = game.get('away_team', '').strip().replace("Oakland Athletics", "Athletics")
        home_team = game.get('home_team', '').strip().replace("Oakland Athletics", "Athletics")
        if away_team and home_team:
            opponent_map[away_team] = home_team
            opponent_map[home_team] = away_team
            
        for bookmaker in game.get("bookmakers", []):
            book_key = bookmaker["key"]
            if book_key in sportsbook_odds:
                market = next((m for m in bookmaker.get("markets", []) if m["key"] == "h2h"), None)
                if market:
                    for outcome in market.get("outcomes", []):
                        team = outcome["name"].strip().replace("Oakland Athletics", "Athletics")
                        
                        if team.lower() == "draw":
                            draw_key = f"Draw_{away_team}_vs_{home_team}"
                            sportsbook_odds[book_key][draw_key] = outcome["price"]
                        else:
                            sportsbook_odds[book_key][team] = outcome["price"]
    
    print(f"📊 {sport}: {today_games} games today, {len(opponent_map)//2} matchups processed")
    return sportsbook_odds, opponent_map

def get_espn_scoreboard_json(league):
    url = f"https://site.api.espn.com/apis/site/v2/sports/{league}/scoreboard"
    response = requests.get(url)
    return response.json()

def parse_games(scoreboard_json):
    games_today = []
    today = now.date()

    print(f"🔍 DEBUG: Parsing games at {now.strftime('%Y-%m-%d %H:%M:%S ET')}")

    for event in scoreboard_json.get("events", []):
        comp = event["competitions"][0]
        home = comp["competitors"][0] if comp["competitors"][0]["homeAway"] == "home" else comp["competitors"][1]
        away = comp["competitors"][0] if comp["competitors"][0]["homeAway"] == "away" else comp["competitors"][1]
        start_time = datetime.fromisoformat(event["date"]).astimezone(eastern)
        time_until_start = (start_time - now).total_seconds() / 3600
        status = comp["status"]["type"]["name"].lower()
        
        is_today = start_time.date() == today
        is_eligible = status == "in" or time_until_start <= 1

        teams_in_game = [home["team"]["displayName"], away["team"]["displayName"]]
        
        print(f"  📅 Game: {' vs '.join(teams_in_game)}")
        print(f"     ⏰ Start: {start_time.strftime('%Y-%m-%d %H:%M:%S ET')} ({time_until_start:.1f}h from now)")
        print(f"     📊 Status: {status}")
        print(f"     📅 Is Today: {is_today}")
        print(f"     ✅ Eligible: {is_eligible} (status='{status}' or time_until_start={time_until_start:.1f}h <= 1)")

        if is_today and is_eligible:
            games_today.append(home["team"]["displayName"])
            games_today.append(away["team"]["displayName"])
            print(f"     ➕ Added teams: {home['team']['displayName']}, {away['team']['displayName']}")
    
    print(f"🎯 DEBUG: Final eligible teams: {games_today}")
    return games_today

def get_eligible_teams(sport=None):
    sport_to_league_map = {
        "mlb": "baseball/mlb",
        "nfl": "football/nfl", 
        "wnba": "basketball/wnba",
        "mls": "soccer/usa.1",
        "epl": "soccer/eng.1",
        "college_football": "football/college-football"
    }
    
    if sport and sport in sport_to_league_map:
        league = sport_to_league_map[sport]
        scoreboard = get_espn_scoreboard_json(league)
        return set(parse_games(scoreboard))
    else:
        leagues = ["baseball/mlb", "football/nfl", "basketball/wnba", "soccer/usa.1","soccer/eng.1", "football/college-football"]
        all_teams = set()
        for league in leagues:
            scoreboard = get_espn_scoreboard_json(league)
            eligible_teams = parse_games(scoreboard)
            all_teams.update(eligible_teams)
        return all_teams

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
    print(f"🔧 Starting sportsbook devigging for {len(odds_dict)} teams")
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
        
        print(f"🏈 Processing game: {team} vs {opponent}")
        print(f"   📊 Raw American odds - {team}: {american_odds}, {opponent}: {opponent_odds}")
        
        team_implied_prob = american_to_implied_prob(american_odds)
        opponent_implied_prob = american_to_implied_prob(opponent_odds)
        
        if team_implied_prob is None or opponent_implied_prob is None:
            continue
        
        total_implied_prob = team_implied_prob + opponent_implied_prob
        print(f"   🧮 Implied probs - {team}: {team_implied_prob:.4f}, {opponent}: {opponent_implied_prob:.4f}")
        print(f"   📈 Total implied prob (with vig): {total_implied_prob:.4f} (vig: {(total_implied_prob - 1) * 100:.2f}%)")
        
        team_fair_prob = team_implied_prob / total_implied_prob
        opponent_fair_prob = opponent_implied_prob / total_implied_prob
        print(f"   ✨ Fair probs after devig - {team}: {team_fair_prob:.4f}, {opponent}: {opponent_fair_prob:.4f}")
        
        devigged_odds[team] = 1 / team_fair_prob
        devigged_odds[opponent] = 1 / opponent_fair_prob
        print(f"   🎯 Final devigged odds - {team}: {devigged_odds[team]:.4f}, {opponent}: {devigged_odds[opponent]:.4f}")
        
        processed_games.add(team)
        processed_games.add(opponent)
    
    print(f"✅ Sportsbook devigging complete: {len(devigged_odds)} teams processed")
    return devigged_odds

def devig_composite_odds(sportsbook_odds, opponent_map, weights=None):
    """
    Create composite devigged odds from multiple sportsbooks using simple averages.
    Now uses simple average of available sportsbooks instead of weighted averages.
    """
    print(f"🧮 Starting composite odds devigging with simple averaging...")
    print(f"📊 Processing sportsbooks: {list(sportsbook_odds.keys())}")
    
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
        available_probs = []
        
        for book, book_odds in devigged_books.items():
            if team in book_odds:
                prob = 1 / book_odds[team]
                available_probs.append(prob)
                print(f"   📊 {book}: prob={prob:.4f}, odds={book_odds[team]:.4f}")
        
        if available_probs:
            # Simple average of available probabilities
            avg_prob = sum(available_probs) / len(available_probs)
            composite_odds[team] = 1 / avg_prob
            successful_composites += 1
            print(f"🎯 Final composite odds for {team}: {composite_odds[team]:.4f} (avg of {len(available_probs)} books)")
        else:
            print(f"❌ No valid odds for {team}")
    
    print(f"📈 Composite devigging summary: {successful_composites}/{len(all_teams)} teams processed successfully")
    return composite_odds

# 🔢 Kelly

def kelly_wager(fair_odds, your_odds, bankroll):
    try:
        if pd.isna(fair_odds) or pd.isna(your_odds):
            print(f"   ⚠️ Kelly calculation skipped - missing odds (fair: {fair_odds}, your: {your_odds})")
            return 0
        
        print(f"   💰 Kelly calculation inputs:")
        print(f"      Fair odds: {fair_odds:.4f}")
        print(f"      Your odds (Kalshi): {your_odds:.4f}")
        print(f"      Bankroll: ${bankroll:.2f}")
        
        fair_prob = 1 / fair_odds
        edge = (fair_prob * (your_odds - 1)) - (1 - fair_prob)
        kelly_fraction = edge / (your_odds - 1)
        kelly_amount = kelly_fraction * bankroll
        final_amount = max(kelly_amount, 0)
        
        print(f"      Fair probability: {fair_prob:.4f}")
        print(f"      Edge: {edge:.4f}")
        print(f"      Kelly fraction: {kelly_fraction:.4f}")
        print(f"      Kelly amount: ${kelly_amount:.2f}")
        print(f"      Final amount (max 0): ${final_amount:.2f}")
        
        return final_amount
    except Exception as e:
        print(f"   ❌ Kelly calculation error: {e}")
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
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAC": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
    "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers", "LA": "Los Angeles Rams", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SF": "San Francisco 49ers",
    "SEA": "Seattle Seahawks", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders"
}

wnba_team_abbr_to_name = {
    "ATL": "Atlanta Dream", "CHI": "Chicago Sky", "CONN": "Connecticut Sun", "DAL": "Dallas Wings",
    "IND": "Indiana Fever", "LV": "Las Vegas Aces", "MIN": "Minnesota Lynx", "NY": "New York Liberty",
    "PHX": "Phoenix Mercury", "SEA": "Seattle Storm", "WSH": "Washington Mystics", "LA": "Los Angeles Sparks",
    "GS": "Golden State Valkyries"
}

epl_team_abbr_to_name = {
    "LFC": "Liverpool", "BOU": "Bournemouth", "AVL": "Aston Villa", "NEW": "Newcastle United",
    "ARS": "Arsenal", "CHE": "Chelsea", "MCI": "Manchester City", "MUN": "Manchester United",
    "TOT": "Tottenham Hotspur", "WHU": "West Ham United", "CRY": "Crystal Palace", "BRI": "Brighton & Hove Albion",
    "FUL": "Fulham", "WOL": "Wolverhampton Wanderers", "EVE": "Everton", "BRE": "Brentford",
    "NFO": "Nottingham Forest", "LEI": "Leicester City", "IPS": "Ipswich Town", "SOU": "Southampton", "BUR": "Burnley", "LEE": "Leeds United", "SUN": "Sunderland"
}

def load_college_football_teams():
    """Load college football team mappings from JSON file"""
    json_path = os.path.join(os.path.dirname(__file__), 'college_football_teams.json')
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ College football teams JSON not found at {json_path}, using fallback mapping")
        return {
            "STAN": "Stanford", "HAW": "Hawaii", "OHIO": "Ohio State", "RUTG": "Rutgers",
            "WYO": "Wyoming", "AKR": "Akron", "DSU": "Delaware State", "DEL": "Delaware",
            "ALST": "Alabama State", "UAB": "UAB", "MICH": "Michigan", "BAMA": "Alabama",
            "UGA": "Georgia", "CLEM": "Clemson", "ND": "Notre Dame", "USC": "USC",
            "UCLA": "UCLA", "ORE": "Oregon", "WASH": "Washington", "UTAH": "Utah"
        }
    except json.JSONDecodeError as e:
        print(f"⚠️ Error parsing college football teams JSON: {e}, using fallback mapping")
        return {
            "STAN": "Stanford", "HAW": "Hawaii", "OHIO": "Ohio State", "RUTG": "Rutgers",
            "WYO": "Wyoming", "AKR": "Akron", "DSU": "Delaware State", "DEL": "Delaware",
            "ALST": "Alabama State", "UAB": "UAB", "MICH": "Michigan", "BAMA": "Alabama",
            "UGA": "Georgia", "CLEM": "Clemson", "ND": "Notre Dame", "USC": "USC",
            "UCLA": "UCLA", "ORE": "Oregon", "WASH": "Washington", "UTAH": "Utah"
        }

college_football_team_abbr_to_name = load_college_football_teams()

mls_team_abbr_to_name = {
    "HOU": "Houston Dynamo",
    "NSH": "Nashville SC", 
    "NYC": "New York City FC",
    "SD": "San Diego FC",
    "SJ": "San Jose Earthquakes",
    "VAN": "Vancouver Whitecaps FC",
    "TIE": "Draw",
    "ATL": "Atlanta United FC",
    "ATX": "Austin FC",
    "CHA": "Charlotte FC",
    "CHI": "Chicago Fire",
    "CIN": "FC Cincinnati",
    "COL": "Colorado Rapids",
    "CLB": "Columbus Crew SC",
    "DCU": "D.C. United",
    "DAL": "FC Dallas",
    "MIA": "Inter Miami CF",
    "LAG": "LA Galaxy",
    "LAFC": "Los Angeles FC",
    "MIN": "Minnesota United FC",
    "MTL": "CF Montreal",
    "NYR": "New York Red Bulls",
    "NE": "New England Revolution",
    "NYF": "New York City FC",
    "ORL": "Orlando City SC",
    "PHI": "Philadelphia Union",
    "POR": "Portland Timbers",
    "RSL": "Real Salt Lake",
    "SEA": "Seattle Sounders FC",
    "KC": "Sporting Kansas City",
    "STL": "St. Louis City SC",
    "TOR": "Toronto FC"
}

all_team_mappings = {
    "MLB": mlb_team_abbr_to_name,
    "NFL": nfl_team_abbr_to_name, 
    "WNBA": wnba_team_abbr_to_name,
    "EPL": epl_team_abbr_to_name,
    "MLS": mls_team_abbr_to_name,
    "COLLEGE_FOOTBALL": college_football_team_abbr_to_name
}

combined_team_abbr_to_name = {}
for sport, mapping in all_team_mappings.items():
    for abbr, name in mapping.items():
        combined_team_abbr_to_name[f"{sport}_{abbr}"] = name
        if abbr not in combined_team_abbr_to_name:
            combined_team_abbr_to_name[abbr] = name

team_abbr_to_name = combined_team_abbr_to_name

def fetch_sport_opportunities(sport, api_key):
    """Fetch opportunities for a specific sport"""
    
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
        "epl": {
            "api_sport": "soccer_epl",
            "kalshi_series": "KXEPLGAME",
            "team_map": epl_team_abbr_to_name,
            "market_type": "3way"
        },
        "mls": {
            "api_sport": "soccer_usa_mls",
            "kalshi_series": "KXMLSGAME",
            "team_map": mls_team_abbr_to_name,
            "market_type": "3way"
        },
        "college_football": {
            "api_sport": "americanfootball_ncaaf",
            "kalshi_series": "KXNCAAFGAME",
            "team_map": college_football_team_abbr_to_name,
            "market_type": "2way"
        }
    }
    
    if sport not in sport_configs:
        return pd.DataFrame()
    
    config = sport_configs[sport]
    
    if sport == "mlb":
        kalshi_df = fetch_kalshi_mlb_odds_active_only()
    else:
        kalshi_df = fetch_kalshi_sport_odds(config["kalshi_series"])
        print(f"🔍 DEBUG: {sport} - Raw Kalshi teams from API: {list(kalshi_df['Team'].unique()) if not kalshi_df.empty else 'No markets'}")
        
        if kalshi_df.empty:
            print(f"🔍 DEBUG: {sport} - No Kalshi markets found, skipping processing")
            return pd.DataFrame()
        
        eligible_teams = get_eligible_teams(sport)
        kalshi_df["Team Name"] = kalshi_df["Team"].map(config["team_map"]) if config["team_map"] else kalshi_df["Team"]
        
        if config["team_map"]:
            unmapped_teams = kalshi_df[kalshi_df["Team Name"].isna()]["Team"].unique()
            if len(unmapped_teams) > 0:
                print(f"🔍 DEBUG: {sport} - Unmapped teams: {list(unmapped_teams)}")
        
        print(f"🔍 DEBUG: {sport} - Before ESPN eligibility filter: {len(kalshi_df)} Kalshi markets")
        print(f"🔍 DEBUG: {sport} - Kalshi teams: {list(kalshi_df['Team Name'].unique())}")
        print(f"🔍 DEBUG: {sport} - ESPN eligible teams: {eligible_teams}")
        
        if sport == "mls":
            espn_to_our_mapping = {}
            for our_team in kalshi_df["Team Name"].unique():
                if pd.isna(our_team):
                    continue
                for espn_team in eligible_teams:
                    if our_team == espn_team:
                        espn_to_our_mapping[espn_team] = our_team
                    elif our_team.replace(" FC", "") == espn_team.replace(" FC", ""):
                        espn_to_our_mapping[espn_team] = our_team
                    elif "Whitecaps" in our_team and "Whitecaps" in espn_team:
                        espn_to_our_mapping[espn_team] = our_team
                    elif "Dynamo" in our_team and "Dynamo" in espn_team:
                        espn_to_our_mapping[espn_team] = our_team
            
            print(f"🔍 DEBUG: {sport} - ESPN to Our team mapping: {espn_to_our_mapping}")
            
            eligible_our_teams = set(espn_to_our_mapping.values())
            kalshi_df = kalshi_df[kalshi_df["Team Name"].isin(eligible_our_teams)].reset_index(drop=True)
        else:
            kalshi_df = kalshi_df[kalshi_df["Team Name"].isin(eligible_teams)].reset_index(drop=True)
        
        print(f"🔍 DEBUG: {sport} - After ESPN eligibility filter: {len(kalshi_df)} Kalshi markets")
        if len(kalshi_df) == 0:
            print(f"🔍 DEBUG: {sport} - All teams were filtered out by ESPN eligibility check")

    
    if kalshi_df.empty:
        return pd.DataFrame()
    
    
    if kalshi_df.empty:
        return pd.DataFrame()
    
    if count_api_call():
        sportsbook_odds, opponent_map = fetch_composite_odds(api_key, config["api_sport"])
        
        if config["market_type"] == "3way":
            composite_odds = devig_soccer_odds(sportsbook_odds, opponent_map)
        else:
            composite_odds = devig_composite_odds(sportsbook_odds, opponent_map)
    else:
        composite_odds = {}
        opponent_map = {}
        sportsbook_odds = {}
    
    kalshi_df["Sport"] = sport.upper()
    kalshi_df["Team Name"] = kalshi_df["Team"].map(config["team_map"]) if config["team_map"] else kalshi_df["Team"]
    
    if config["market_type"] == "2way" and opponent_map:
        kalshi_df["Opponent Name"] = kalshi_df["Team Name"].map(opponent_map)
    elif sport == "mlb":
        mlb_opponent_map = build_opponent_map()
        kalshi_df["Opponent Name"] = kalshi_df["Team Name"].map(mlb_opponent_map)
    else:
        kalshi_df["Opponent Name"] = ""
    
    kalshi_df["Composite Fair Odds"] = kalshi_df["Team Name"].map(composite_odds)
    
    kalshi_df["Kalshi %"] = kalshi_df["Kalshi YES Ask (¢)"] / 100
    kalshi_df["Decimal Odds (Kalshi)"] = 1 / kalshi_df["Kalshi %"]
    
    per_book_american = {}
    per_book_prob = {}
    for team in kalshi_df["Team Name"].unique():
        if pd.isna(team):
            continue
        per_book_american[team] = {}
        per_book_prob[team] = {}
        for book, odds_by_team in sportsbook_odds.items():
            if team in odds_by_team:
                price = odds_by_team[team]
                per_book_american[team][book] = price
                per_book_prob[team][book] = american_to_implied_prob(price)
    kalshi_df["Per-Book American Odds"] = kalshi_df["Team Name"].map(per_book_american)
    kalshi_df["Per-Book Implied Prob"] = kalshi_df["Team Name"].map(per_book_prob)

    # Calculate fee based on Kalshi price
    kalshi_df["Fee"] = kalshi_df["Kalshi YES Ask (¢)"].astype(int).apply(get_kalshi_fee_rate)
    
    print(f"\n🧮 Starting raw edge calculation for {len(kalshi_df)} opportunities...")
    
    # Adjusted edge = (Decimal Odds * True Prob) - 1 - Fee
    raw_edge_list = []
    for idx, row in kalshi_df.iterrows():
        team_name = row["Team Name"]
        kalshi_decimal_odds = row["Decimal Odds (Kalshi)"]
        composite_fair_odds = row["Composite Fair Odds"]
        fee = row["Fee"]
        kalshi_price_cents = row["Kalshi YES Ask (¢)"]
        
        print(f"\n📊 Edge calculation for {team_name}:")
        print(f"   💵 Kalshi price: {kalshi_price_cents}¢")
        print(f"   🎲 Kalshi decimal odds: {kalshi_decimal_odds:.4f}")
        print(f"   🎯 Composite fair odds: {composite_fair_odds:.4f}")
        print(f"   💸 Fee rate: {fee:.4f} ({fee * 100:.2f}%)")
        
        if pd.notna(composite_fair_odds):
            true_prob = 1 / composite_fair_odds
            expected_return = kalshi_decimal_odds * true_prob
            raw_edge_value = expected_return - 1 - fee
            
            print(f"   📈 True probability: {true_prob:.4f}")
            print(f"   💰 Expected return: {expected_return:.4f}")
            print(f"   ⚡ Raw edge (before fee): {expected_return - 1:.4f} ({(expected_return - 1) * 100:.2f}%)")
            print(f"   🎯 Final edge (after fee): {raw_edge_value:.4f} ({raw_edge_value * 100:.2f}%)")
        else:
            raw_edge_value = None
            print(f"   ❌ No composite fair odds available - edge calculation skipped")
        
        raw_edge_list.append(raw_edge_value)
    
    raw_edge = pd.Series(raw_edge_list, index=kalshi_df.index)
    kalshi_df["% Edge"] = raw_edge.apply(lambda x: f"{round(x * 100, 1)}%" if pd.notna(x) else None)
    kalshi_df["numeric_edge"] = raw_edge
    
    print(f"\n✅ Edge calculation complete for {len(kalshi_df)} opportunities")
    
    return kalshi_df

def get_kalshi_fee_rate(price_cents):
    if 30 <= price_cents <= 40:
        return 0.016
    elif 41 <= price_cents <= 59:
        return 0.017
    elif 60 <= price_cents <= 69:
        return 0.016
    elif 70 <= price_cents <= 79:
        return 0.0135
    elif 80 <= price_cents <= 90:
        return 0.01
    else:
        return 0.0  # if you want to ignore prices outside this range

def fetch_kalshi_sport_odds(series_ticker):
    """Generic function to fetch Kalshi odds for any sport series - filters for today's games only"""
    if series_ticker == "KXNCAAFGAME":
        url = f"https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker={series_ticker}&status=open&limit=1000"
    else:
        url = f"https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker={series_ticker}"
    headers = {"accept": "application/json"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        rows = []
        
        today_str = today.strftime("%y%b%d").upper()
        
        for market in data.get("markets", []):
            if market.get("status") != "active":
                continue
            
            ticker = market.get("ticker", "")
            parts = ticker.split('-')
            if len(parts) < 3:
                continue
            
            if today_str not in ticker:
                continue
                
            rows.append({
                "Market Ticker": ticker,
                "Game Title": market.get("title"),
                "Team": parts[-1],
                "Kalshi YES Ask (¢)": market.get("yes_ask")
            })
        
        print(f"📅 {series_ticker}: Found {len(rows)} markets for today ({today_str})")
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"❌ Error fetching {series_ticker} markets: {e}")
        return pd.DataFrame()

def devig_soccer_odds(sportsbook_odds, opponent_map):
    """Special devigging for soccer 3-way markets (Win/Loss/Draw)"""
    print(f"🧮 Starting soccer odds devigging with game-by-game processing...")
    print(f"📊 Processing sportsbooks: {list(sportsbook_odds.keys())}")
    
    devigged_books = {}
    for book, odds_dict in sportsbook_odds.items():
        if odds_dict:
            print(f"📚 Processing {book} with {len(odds_dict)} teams")
            devigged_books[book] = devig_sportsbook_odds_soccer(odds_dict, opponent_map)
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
        available_probs = []
        
        for book, book_odds in devigged_books.items():
            if team in book_odds:
                prob = 1 / book_odds[team]
                available_probs.append(prob)
                print(f"   📊 {book}: prob={prob:.4f}, odds={book_odds[team]:.4f}")
        
        if available_probs:
            # Simple average of available probabilities
            avg_prob = sum(available_probs) / len(available_probs)
            composite_odds[team] = 1 / avg_prob
            successful_composites += 1
            print(f"🎯 Final composite odds for {team}: {composite_odds[team]:.4f} (avg of {len(available_probs)} books)")
        else:
            print(f"❌ No valid odds for {team}")
    
    print(f"📈 Soccer composite devigging summary: {successful_composites}/{len(all_teams)} teams processed successfully")
    return composite_odds

def devig_sportsbook_odds_soccer(odds_dict, opponent_map):
    """Devigger for a single sportsbook's 3-way soccer odds with proper game-by-game normalization"""
    
    print(f"🔍 DEBUG: Starting 3-way soccer devigging for {len(odds_dict)} teams")
    print(f"🔍 DEBUG: Found odds for teams: {list(odds_dict.keys())}")
    
    games = {}
    processed_teams = set()
    
    for team in odds_dict.keys():
        if team.lower().startswith("draw_") or team in processed_teams:
            continue
            
        opponent = opponent_map.get(team)
        if opponent and opponent in odds_dict:
            draw_key = f"Draw_{team}_vs_{opponent}"
            alt_draw_key = f"Draw_{opponent}_vs_{team}"
            
            game_teams = [team, opponent]
            
            if draw_key in odds_dict:
                game_teams.append(draw_key)
            elif alt_draw_key in odds_dict:
                game_teams.append(alt_draw_key)
            else:
                print(f"🔍 DEBUG: Warning: No Draw outcome found for game {team} vs {opponent}")
            
            game_id = f"{team}_vs_{opponent}"
            games[game_id] = game_teams
            processed_teams.add(team)
            processed_teams.add(opponent)
            
            print(f"🔍 DEBUG: Created game {game_id} with teams: {game_teams}")
    
    devigged_odds = {}
    
    for game_id, game_teams in games.items():
        print(f"🏈 Processing game: {game_id}")
        
        game_odds = {team: odds_dict[team] for team in game_teams if team in odds_dict}
        
        if len(game_odds) < 2:
            print(f"   ⚠️ Skipping game {game_id}: insufficient odds data")
            continue
        
        implied_probs = {}
        total_implied_prob = 0
        
        print(f"   🔍 Converting American odds to implied probabilities:")
        for team in game_odds:
            american_odds = game_odds[team]
            prob = american_to_implied_prob(american_odds)
            if prob is not None:
                implied_probs[team] = prob
                total_implied_prob += prob
                display_name = "Draw" if team.lower().startswith("draw_") else team
                print(f"      📊 {display_name}: {american_odds} → {prob:.4f} ({prob*100:.2f}%)")
        
        print(f"   📈 Total implied prob (with vig): {total_implied_prob:.4f} ({total_implied_prob*100:.2f}%)")
        print(f"   🧮 Vig amount: {(total_implied_prob - 1.0):.4f} ({(total_implied_prob - 1.0)*100:.2f}%)")
        
        if total_implied_prob > 0:
            print(f"   🔍 Normalizing probabilities and converting back to odds:")
            for team in implied_probs:
                normalized_prob = implied_probs[team] / total_implied_prob
                decimal_odds = 1 / normalized_prob
                
                output_name = "Draw" if team.lower().startswith("draw_") else team
                devigged_odds[output_name] = decimal_odds
                
                display_name = "Draw" if team.lower().startswith("draw_") else team
                print(f"      🎯 {display_name}: prob {implied_probs[team]:.4f} → normalized {normalized_prob:.4f} → odds {decimal_odds:.4f}")
            
            total_normalized = sum(implied_probs[team] / total_implied_prob for team in implied_probs)
            print(f"   ✅ Verification: normalized probabilities sum to {total_normalized:.4f}")
    
    print(f"🔍 DEBUG: Devigging complete. Processed {len(devigged_odds)} outcomes across {len(games)} games")
    return devigged_odds


# def get_already_bet_teams():
#     """
#     Get teams/players we've already bet on today to avoid duplicate API calls
#     Returns set of team abbreviations from executed orders
#     """
#     print(f"🔍 Checking for already bet teams to optimize API calls...")
#     
#     try:
#         executed_team_abbrs = set()
#         for order in get_todays_orders():
#             if (
#                 order.get("status") in ("executed", "resting") and
#                 order.get("action") == "buy" and
#                 order.get("ticker")
#             ):
#                 parts = order["ticker"].split('-')
#                 if len(parts) >= 3:
#                     abbr = parts[-1]
#                     executed_team_abbrs.add(abbr)
#         
#         print(f"✅ Found {len(executed_team_abbrs)} already bet teams: {sorted(executed_team_abbrs)}")
#         return executed_team_abbrs
#         
#     except Exception as e:
#         print(f"⚠️ Error fetching today's orders: {e} - proceeding without pre-filtering")
#         return set()

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

testing_mode = False  # Set to False for production

def count_api_call():
    """Track API calls for logging purposes - no artificial limits in production"""
    print(f"📞 Making API call...")
    return True

def get_dynamic_kelly_multiplier():
    """Return fixed Kelly multiplier of 0.65"""
    return 0.5

print("🚀 Starting multi-sport betting bot...")
print(f"🧪 Testing mode: {testing_mode}")

sports_to_process = ["mlb", "nfl", "wnba", "epl", "mls", "college_football"]
api_key = ODDS_API_KEY

all_sport_dataframes = []
for sport in sports_to_process:
    try:
        sport_df = fetch_sport_opportunities(sport, api_key)
        if not sport_df.empty:
            print(f"✅ {sport.upper()}: {len(sport_df)} opportunities")
            all_sport_dataframes.append(sport_df)
    except Exception as e:
        print(f"❌ Error processing {sport.upper()}: {e}")
        if sport in ["epl", "college_football"]:
            print(f"🔍 DEBUG: {sport.upper()} error details - Exception type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
        continue

if all_sport_dataframes:
    kalshi_df = pd.concat(all_sport_dataframes, ignore_index=True)
    
    
    print("\n" + "="*80)
    print("📋 FULL MULTI-SPORT DATAFRAME (Today's Games Only)")
    print("="*80)
    if not kalshi_df.empty:
        display_columns = ["Sport", "Team Name", "Opponent Name", "Kalshi YES Ask (¢)", "Composite Fair Odds", "% Edge", "Market Ticker"]
        available_columns = [col for col in display_columns if col in kalshi_df.columns]
        print(kalshi_df[available_columns].to_string(index=False))
    else:
        print("No games found for today")
    print("="*80 + "\n")
else:
    kalshi_df = pd.DataFrame()

if not kalshi_df.empty:
    dynamic_kelly = get_dynamic_kelly_multiplier()

    print(f"\n💰 Calculating Kelly wager amounts for {len(kalshi_df)} opportunities...")
    
    wager_amounts = []
    for idx, row in kalshi_df.iterrows():
        team_name = row["Team Name"]
        print(f"\n🎯 Kelly wager calculation for {team_name}:")
        
        if pd.notna(row["Composite Fair Odds"]) and pd.notna(row["Decimal Odds (Kalshi)"]):
            kelly_amount = kelly_wager(row['Composite Fair Odds'], row['Decimal Odds (Kalshi)'], bankroll)
            max_wager = bankroll * 0.3
            final_wager = min(kelly_amount, max_wager)
            
            print(f"   🔒 Max wager cap (30% bankroll): ${max_wager:.2f}")
            print(f"   ✅ Final wager amount: ${final_wager:.2f}")
            
            wager_amounts.append(f"${round(final_wager)}")
        else:
            print(f"   ❌ Missing odds data - wager set to $0")
            wager_amounts.append("$0")
    
    kalshi_df["$ Wager"] = wager_amounts
    print(f"\n✅ Kelly wager calculations complete")

    kalshi_df["Kalshi YES Ask (¢)"] = kalshi_df["Kalshi YES Ask (¢)"].astype(int)
    kalshi_df = kalshi_df.sort_values(by="numeric_edge", ascending=False).reset_index(drop=True)
else:
    print("⚠️ Skipping further processing due to empty dataset")


def store_odds_timeseries():
    filename = ODDS_TIMESERIES_PATH
    timestamp = datetime.now(eastern).isoformat()
    
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
    else:
        data = []
    
    for _, row in kalshi_df.iterrows():
        if pd.notna(row["Composite Fair Odds"]) and pd.notna(row["Kalshi %"]):
            per_book_american = row.get("Per-Book American Odds", {}) or {}
            per_book_prob = row.get("Per-Book Implied Prob", {}) or {}

            game_id = None
            ticker = row.get("Market Ticker", "")
            if ticker:
                # Example: KXMLBGAME-25JUL27NYMSF-NYM -> game_id could be "25JUL27NYMSF"
                parts = ticker.split("-")
                if len(parts) >= 2:
                    game_id = parts[1]

            row_obj = {
                "timestamp": timestamp,
                "sport": row.get("Sport", "UNKNOWN"),
                "team": row["Team Name"],
                "kalshi_implied_odds": row["Kalshi %"],
                "composite_devigged_odds": 1 / row["Composite Fair Odds"],
                "expected_value": row["numeric_edge"],
                "per_book_american_odds": per_book_american,
                "per_book_implied_prob": per_book_prob,
                "composite_source_books": sorted(list(per_book_american.keys())),
                "game_id": game_id,
                "market_ticker": ticker
            }

            data.append(row_obj)

            ndjson_path = get_ndjson_path(row_obj["sport"], datetime.now(eastern).date())
            write_ndjson_line(ndjson_path, row_obj)
    
    if not NDJSON_ONLY:
        with open(filename, "w") as f:
            json.dump(data, f)

store_odds_timeseries()

store_daily_available_events()

if not kalshi_df.empty and "numeric_edge" in kalshi_df.columns:
    kalshi_df = kalshi_df.drop(columns=["numeric_edge"])

if kalshi_df.empty:
    print("⚠️ No betting opportunities found - kalshi_df is empty")
    final_df = pd.DataFrame(columns=[
        "Team Name", "Opponent Name", "Kalshi YES Ask (¢)", 
        "Composite Fair Odds", "% Edge", "$ Wager", "Market Ticker"
    ])
else:
    required_columns = ["Team Name", "Opponent Name", "Kalshi YES Ask (¢)", 
                       "Composite Fair Odds", "% Edge", "$ Wager", "Market Ticker"]
    missing_columns = [col for col in required_columns if col not in kalshi_df.columns]
    
    if missing_columns:
        print(f"⚠️ Missing required columns: {missing_columns}")
        print(f"Available columns: {list(kalshi_df.columns)}")
        final_df = pd.DataFrame(columns=required_columns)
    else:
        final_df = kalshi_df[required_columns]

#print("\n📊 Full Table:")
#display(final_df)

print(f"\n🔍 Applying betting criteria filters...")
print(f"📋 Filter criteria:")
print(f"   💰 Kalshi price: 35¢ - 90¢")
print(f"   📈 Edge: 3% - 15%")

if not final_df.empty:
    print(f"\n🎯 Filtering {len(final_df)} opportunities...")
    
    price_filter = (final_df["Kalshi YES Ask (¢)"] >= 35) & (final_df["Kalshi YES Ask (¢)"] <= 90)
    edge_filter = (final_df["% Edge"].str.replace('%', '').astype(float) >= 3) & (final_df["% Edge"].str.replace('%', '').astype(float) < 15)
    
    print(f"   💵 Price filter passed: {price_filter.sum()}/{len(final_df)} opportunities")
    print(f"   📊 Edge filter passed: {edge_filter.sum()}/{len(final_df)} opportunities")
    
    combined_filter = price_filter & edge_filter
    print(f"   ✅ Both filters passed: {combined_filter.sum()}/{len(final_df)} opportunities")
    
    filtered_df = final_df[combined_filter].reset_index(drop=True)
    
    if not filtered_df.empty:
        print(f"\n📋 Teams passing all filters:")
        for idx, row in filtered_df.iterrows():
            print(f"   🏆 {row['Team Name']}: {row['Kalshi YES Ask (¢)']}¢, {row['% Edge']} edge")
    else:
        print(f"   ❌ No opportunities passed all filters")
else:
    print(f"   ⚠️ No opportunities to filter - final_df is empty")
    filtered_df = pd.DataFrame(columns=[
        "Team Name", "Opponent Name", "Kalshi YES Ask (¢)", 
        "Composite Fair Odds", "% Edge", "$ Wager", "Market Ticker"
    ])

print("\n" + "="*80)
print("🎯 FILTERED DATAFRAME (After Betting Criteria Applied)")
print("Criteria: Kalshi Ask 60-95¢, Edge 4-9.1%")
print("="*80)
if not filtered_df.empty:
    print(filtered_df.to_string(index=False))
    print(f"\n📊 Filtered results: {len(filtered_df)} opportunities from {len(final_df)} total")
else:
    print("No opportunities meet the betting criteria")
print("="*80 + "\n")

if not filtered_df.empty:
    print(f"\n🧹 Cleaning duplicate events from {len(filtered_df)} opportunities...")
    seen_events = set()
    filtered_cleaned = []
    
    for _, row in filtered_df.iterrows():
        team = row["Team Name"]
        opponent = row.get("Opponent Name", "")
        event_id = extract_event_id(row["Market Ticker"])
        
        if event_id in seen_events:
            print(f"   🚫 Skipping {team} - already processed event {event_id}")
            continue
        
        seen_events.add(event_id)
        filtered_cleaned.append(row)
        print(f"   ✅ Added {team} vs {opponent} - blocking event {event_id}")
    
    print(f"   📊 Cleaned results: {len(filtered_cleaned)} opportunities from {len(filtered_df)} total")
    filtered_df = pd.DataFrame(filtered_cleaned).reset_index(drop=True)
    
    if not filtered_df.empty and "% Edge" in filtered_df.columns:
        filtered_df["numeric_edge_temp"] = filtered_df["% Edge"].str.replace('%', '').astype(float)
        filtered_df = filtered_df.sort_values(by="numeric_edge_temp", ascending=False).reset_index(drop=True)
        filtered_df = filtered_df.drop(columns=["numeric_edge_temp"])
        print(f"   📈 Sorted {len(filtered_df)} opportunities by edge (highest first)")
else:
    print("⚠️ No opportunities to clean - filtered_df is empty")

# 📥 Get today's orders (Eastern Time aware)
def get_todays_orders():
    path = "/trade-api/v2/portfolio/orders"
    url = f"https://api.elections.kalshi.com{path}"
    headers = sign_request("GET", path, KALSHI_API_KEY, private_key)

    from datetime import time  # needed for time.min / time.max
    today_start_dt = eastern.localize(datetime.combine(today, time.min))
    today_end_dt = eastern.localize(datetime.combine(today, time.max))

    today_start = int(today_start_dt.timestamp())
    today_end = int(today_end_dt.timestamp())

    params = {"min_ts": today_start, "max_ts": today_end}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json().get("orders", [])

# 🚫 Exclude previously bet events (buy-only)
executed_event_ids = set()
for order in get_todays_orders():
    if (
        order.get("status") in ("executed","resting") and
        order.get("action") == "buy" and
        order.get("ticker")
    ):
        ticker = order["ticker"]
        event_id = extract_event_id(ticker)
        if event_id:
            executed_event_ids.add(event_id)

# 🔍 Debug print to verify which events are being excluded
print(f"✅ Executed event IDs (BUY only): {sorted(executed_event_ids)}")

# Identify rows being filtered out
if not filtered_df.empty:
    filtered_df["Event ID"] = filtered_df["Market Ticker"].apply(extract_event_id)
    
    flagged_rows = filtered_df[
        filtered_df["Event ID"].isin(executed_event_ids)
    ]

    print(f"\n🚫 Rows removed due to duplicate event bets: {len(flagged_rows)}")
    if not flagged_rows.empty:
        display(flagged_rows[["Team Name", "Opponent Name", "Market Ticker", "Event ID"]])

    # Apply the actual filtering
    filtered_df = filtered_df[
        ~filtered_df["Event ID"].isin(executed_event_ids)
    ].reset_index(drop=True)
    
    filtered_df = filtered_df.drop(columns=["Event ID"])
    
    if not filtered_df.empty and "% Edge" in filtered_df.columns:
        filtered_df["numeric_edge_temp"] = filtered_df["% Edge"].str.replace('%', '').astype(float)
        filtered_df = filtered_df.sort_values(by="numeric_edge_temp", ascending=False).reset_index(drop=True)
        filtered_df = filtered_df.drop(columns=["numeric_edge_temp"])
        print(f"   📈 Final sort: {len(filtered_df)} opportunities ordered by edge (highest first)")

    print("\n📈 Final filtered_df after exclusions:")
    display(filtered_df)
else:
    print("⚠️ filtered_df is empty before exclusions.")

# 🛒 Submit orders
def submit_order(market_ticker, side, quantity, price):
    path = "/trade-api/v2/portfolio/orders"
    url = f"https://api.elections.kalshi.com{path}"
    headers = sign_request("POST", path, KALSHI_API_KEY, private_key)
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
        
        current_event_id = extract_event_id(row["Market Ticker"])
        
        if current_event_id in executed_event_ids:
            print(f"⚠️ Skipping {row['Team Name']} vs {row['Opponent Name']} — already bet on event {current_event_id}")
            continue

        ticker = row["Market Ticker"]
        team = row["Team Name"]
        base_wager = float(row["$ Wager"].strip("$") or 0)
        wager_dollars = dynamic_kelly * base_wager
        price = int(row["Kalshi YES Ask (¢)"])
        cost_per_contract = price / 100
        suggested_contracts = int(wager_dollars // cost_per_contract)
        max_contracts_20pct = int(0.2 * bankroll / cost_per_contract)
        contracts = min(suggested_contracts, max_contracts_20pct)
        total_cost = contracts * cost_per_contract
        
        print(f"💰 Position sizing calculation:")
        print(f"   📊 Base Kelly wager: ${base_wager:.2f}")
        print(f"   🔢 Kelly multiplier: {dynamic_kelly}")
        print(f"   💵 Adjusted wager: ${wager_dollars:.2f}")
        print(f"   💰 Price per contract: {price}¢ (${cost_per_contract:.2f})")
        print(f"   🎯 Suggested contracts (from Kelly): {suggested_contracts}")
        print(f"   🛡️ Max contracts (20% bankroll limit): {max_contracts_20pct}")
        print(f"   ✅ Final contracts: {contracts}")
        print(f"   💸 Total cost: ${total_cost:.2f}")
        print(f"   📊 Position as % of bankroll: {(total_cost / bankroll) * 100:.2f}%")

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
        
        orders_filename = PLACED_ORDERS_PATH
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

