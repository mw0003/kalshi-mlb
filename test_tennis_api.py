#!/usr/bin/env python3
"""
Test tennis API integration to analyze player name formatting and odds comparison
between Odds API and Kalshi tennis markets.
Limited to 20 API calls as requested.
"""

import os
import sys
import requests
import pandas as pd
from datetime import datetime, date
import pytz

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from kalshi_bot import fetch_kalshi_sport_odds, fetch_composite_odds, count_api_call, api_calls_made, max_api_calls
    print("✅ Successfully imported kalshi_bot functions")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("⚠️ Will create mock functions for testing")

max_api_calls = 20
api_calls_made = 0

def test_tennis_name_formatting():
    """Test how tennis player names are formatted in different APIs"""
    print("🎾 Testing Tennis API Name Formatting")
    print("=" * 50)
    
    odds_api_key = os.getenv('ODDS_API_KEY')
    if not odds_api_key:
        print("❌ No ODDS_API_KEY found in environment")
        print("💡 Using mock data for demonstration")
        return test_with_mock_data()
    
    print(f"📊 API call budget: {max_api_calls} calls")
    print(f"🎯 Current usage: {api_calls_made}/{max_api_calls}")
    
    tennis_sports = {
        "WTA": "tennis_wta",
        "ATP": "tennis_atp"
    }
    
    results = {}
    
    for league, sport_code in tennis_sports.items():
        print(f"\n🏆 Testing {league} ({sport_code})")
        print("-" * 30)
        
        if api_calls_made >= max_api_calls:
            print(f"🚫 API call limit reached ({max_api_calls})")
            break
            
        try:
            print(f"📡 Fetching {league} odds from Odds API...")
            odds_data = fetch_tennis_odds_api(odds_api_key, sport_code)
            
            print(f"🎯 Fetching {league} odds from Kalshi...")
            kalshi_series = "KXWTAMATCH" if league == "WTA" else "KXATPMATCH"
            kalshi_data = fetch_kalshi_tennis_data(kalshi_series)
            
            analysis = analyze_name_formatting(odds_data, kalshi_data, league)
            results[league] = analysis
            
        except Exception as e:
            print(f"❌ Error testing {league}: {e}")
            results[league] = {"error": str(e)}
    
    print_tennis_analysis(results)
    return results

def fetch_tennis_odds_api(api_key, sport):
    """Fetch tennis odds from The Odds API"""
    global api_calls_made
    
    if api_calls_made >= max_api_calls:
        print(f"🚫 API call limit reached")
        return {}
    
    url = f'https://api.the-odds-api.com/v4/sports/{sport}/odds'
    params = {
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "bookmakers": "fanduel,pinnacle,draftkings",
        "apiKey": api_key
    }
    
    print(f"🌐 Making API call {api_calls_made + 1}/{max_api_calls}")
    response = requests.get(url, params=params)
    api_calls_made += 1
    
    if response.status_code != 200:
        print(f"❌ API error: {response.status_code}")
        return {}
    
    data = response.json()
    print(f"📊 Received {len(data)} matches from Odds API")
    
    matches = []
    for match in data:
        match_info = {
            "match_id": match.get("id"),
            "commence_time": match.get("commence_time"),
            "home_team": match.get("home_team"),
            "away_team": match.get("away_team"),
            "bookmakers": {}
        }
        
        for bookmaker in match.get("bookmakers", []):
            book_name = bookmaker["key"]
            market = next((m for m in bookmaker.get("markets", []) if m["key"] == "h2h"), None)
            if market:
                outcomes = {}
                for outcome in market.get("outcomes", []):
                    outcomes[outcome["name"]] = outcome["price"]
                match_info["bookmakers"][book_name] = outcomes
        
        matches.append(match_info)
        print(f"🎾 Match: {match_info['away_team']} vs {match_info['home_team']}")
        
        for book, odds in match_info["bookmakers"].items():
            players = list(odds.keys())
            if len(players) >= 2:
                print(f"   📚 {book}: {players[0]} vs {players[1]}")
    
    return matches

def fetch_kalshi_tennis_data(series_ticker):
    """Fetch tennis data from Kalshi API"""
    print(f"🎯 Fetching Kalshi data for {series_ticker}")
    
    try:
        if 'fetch_kalshi_sport_odds' in globals():
            df = fetch_kalshi_sport_odds(series_ticker)
            print(f"📈 Kalshi DataFrame shape: {df.shape}")
            
            if not df.empty:
                print("🏷️ Sample Kalshi player names:")
                for i, team in enumerate(df["Team"].head(10)):
                    print(f"   {i+1}. {team}")
            
            return df
        else:
            print("⚠️ fetch_kalshi_sport_odds not available, using mock data")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Error fetching Kalshi data: {e}")
        return pd.DataFrame()

def analyze_name_formatting(odds_data, kalshi_data, league):
    """Analyze differences in player name formatting"""
    print(f"\n🔍 Analyzing {league} Name Formatting")
    print("-" * 40)
    
    analysis = {
        "league": league,
        "odds_api_players": [],
        "kalshi_players": [],
        "name_patterns": {},
        "potential_matches": [],
        "formatting_issues": []
    }
    
    for match in odds_data:
        if match.get("home_team") and match.get("away_team"):
            analysis["odds_api_players"].extend([
                match["home_team"], 
                match["away_team"]
            ])
    
    if not kalshi_data.empty and "Team" in kalshi_data.columns:
        analysis["kalshi_players"] = kalshi_data["Team"].tolist()
    
    print(f"📊 Found {len(analysis['odds_api_players'])} players in Odds API")
    print(f"📊 Found {len(analysis['kalshi_players'])} players in Kalshi")
    
    odds_patterns = analyze_name_patterns(analysis["odds_api_players"])
    kalshi_patterns = analyze_name_patterns(analysis["kalshi_players"])
    
    analysis["name_patterns"] = {
        "odds_api": odds_patterns,
        "kalshi": kalshi_patterns
    }
    
    potential_matches = find_potential_name_matches(
        analysis["odds_api_players"], 
        analysis["kalshi_players"]
    )
    analysis["potential_matches"] = potential_matches
    
    return analysis

def analyze_name_patterns(names):
    """Analyze common patterns in player names"""
    patterns = {
        "total_names": len(names),
        "has_comma": 0,
        "has_initial": 0,
        "has_accent": 0,
        "word_count": {},
        "examples": names[:5]
    }
    
    for name in names:
        if "," in name:
            patterns["has_comma"] += 1
        if ". " in name or " ." in name:
            patterns["has_initial"] += 1
        if any(ord(char) > 127 for char in name):
            patterns["has_accent"] += 1
            
        word_count = len(name.split())
        patterns["word_count"][word_count] = patterns["word_count"].get(word_count, 0) + 1
    
    return patterns

def find_potential_name_matches(odds_names, kalshi_names):
    """Find potential matches between name lists"""
    matches = []
    
    for odds_name in odds_names[:10]:  # Limit to first 10 for testing
        for kalshi_name in kalshi_names[:10]:
            similarity = calculate_name_similarity(odds_name, kalshi_name)
            if similarity > 0.6:  # 60% similarity threshold
                matches.append({
                    "odds_name": odds_name,
                    "kalshi_name": kalshi_name,
                    "similarity": similarity
                })
    
    return sorted(matches, key=lambda x: x["similarity"], reverse=True)

def calculate_name_similarity(name1, name2):
    """Calculate simple name similarity"""
    words1 = set(name1.lower().split())
    words2 = set(name2.lower().split())
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0

def print_tennis_analysis(results):
    """Print comprehensive analysis of tennis API comparison"""
    print("\n" + "=" * 60)
    print("🎾 TENNIS API COMPARISON ANALYSIS")
    print("=" * 60)
    
    for league, analysis in results.items():
        if "error" in analysis:
            print(f"\n❌ {league}: {analysis['error']}")
            continue
            
        print(f"\n🏆 {league} Analysis:")
        print("-" * 30)
        
        odds_patterns = analysis["name_patterns"]["odds_api"]
        kalshi_patterns = analysis["name_patterns"]["kalshi"]
        
        print(f"📊 Odds API: {odds_patterns['total_names']} players")
        print(f"📊 Kalshi: {kalshi_patterns['total_names']} players")
        
        print(f"\n🔍 Name Format Patterns:")
        print(f"   Odds API - Commas: {odds_patterns['has_comma']}, Initials: {odds_patterns['has_initial']}")
        print(f"   Kalshi - Commas: {kalshi_patterns['has_comma']}, Initials: {kalshi_patterns['has_initial']}")
        
        print(f"\n📝 Example Names:")
        print(f"   Odds API: {odds_patterns['examples']}")
        print(f"   Kalshi: {kalshi_patterns['examples']}")
        
        matches = analysis["potential_matches"]
        if matches:
            print(f"\n🎯 Top Potential Matches:")
            for match in matches[:5]:
                print(f"   {match['similarity']:.2f}: '{match['odds_name']}' ↔ '{match['kalshi_name']}'")
        else:
            print(f"\n⚠️ No potential matches found")
    
    print(f"\n📞 Total API calls used: {api_calls_made}/{max_api_calls}")

def test_with_mock_data():
    """Test with mock data when API key not available"""
    print("🧪 Using mock data for demonstration")
    
    mock_results = {
        "WTA": {
            "league": "WTA",
            "odds_api_players": [
                "Iga Swiatek", "Coco Gauff", "A. Sabalenka", "J. Pegula",
                "E. Rybakina", "O. Jabeur", "C. Garcia", "M. Sakkari"
            ],
            "kalshi_players": [
                "Swiatek, Iga", "Gauff, Coco", "Sabalenka, Aryna", "Pegula, Jessica",
                "Rybakina, Elena", "Jabeur, Ons", "Garcia, Caroline", "Sakkari, Maria"
            ],
            "name_patterns": {
                "odds_api": {"total_names": 8, "has_comma": 0, "has_initial": 2, "examples": ["Iga Swiatek", "A. Sabalenka"]},
                "kalshi": {"total_names": 8, "has_comma": 8, "has_initial": 0, "examples": ["Swiatek, Iga", "Gauff, Coco"]}
            },
            "potential_matches": [
                {"odds_name": "Iga Swiatek", "kalshi_name": "Swiatek, Iga", "similarity": 1.0},
                {"odds_name": "Coco Gauff", "kalshi_name": "Gauff, Coco", "similarity": 1.0}
            ]
        }
    }
    
    print_tennis_analysis(mock_results)
    return mock_results

if __name__ == "__main__":
    print("🎾 Starting Tennis API Integration Test")
    print(f"⏰ Test started at: {datetime.now()}")
    print(f"🎯 API call limit: {max_api_calls}")
    
    results = test_tennis_name_formatting()
    
    print(f"\n✅ Test completed")
    print(f"📞 Final API usage: {api_calls_made}/{max_api_calls}")
