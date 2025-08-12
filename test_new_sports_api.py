#!/usr/bin/env python3

import requests
import json
from credentials import ODDS_API_KEY

def test_odds_api_sports():
    """Test available sports from Odds API"""
    print("🔍 Testing Odds API - Available Sports")
    print("="*50)
    
    url = "https://api.the-odds-api.com/v4/sports"
    params = {
        'apiKey': ODDS_API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        sports = response.json()
        
        print(f"✅ Found {len(sports)} available sports")
        
        soccer_sports = [s for s in sports if 'soccer' in s.get('key', '').lower()]
        football_sports = [s for s in sports if 'football' in s.get('key', '').lower()]
        
        print("\n🏈 Football Sports:")
        for sport in football_sports:
            print(f"   {sport['key']}: {sport['title']}")
            
        print("\n⚽ Soccer Sports:")
        for sport in soccer_sports:
            print(f"   {sport['key']}: {sport['title']}")
            
        return sports
        
    except Exception as e:
        print(f"❌ Error fetching sports: {e}")
        return []

def test_epl_soccer_api():
    """Test EPL soccer API call"""
    print("\n🔍 Testing EPL Soccer API")
    print("="*50)
    
    url = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"
    params = {
        'apiKey': ODDS_API_KEY,
        'regions': 'us',
        'markets': 'h2h',
        'oddsFormat': 'american',
        'dateFormat': 'iso'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        games = response.json()
        
        print(f"✅ Found {len(games)} EPL games")
        
        if games:
            print("\n📊 Sample EPL game:")
            game = games[0]
            print(f"   Game: {game.get('home_team')} vs {game.get('away_team')}")
            print(f"   Start time: {game.get('commence_time')}")
            
            if game.get('bookmakers'):
                bookmaker = game['bookmakers'][0]
                print(f"   Bookmaker: {bookmaker.get('title')}")
                
                if bookmaker.get('markets'):
                    market = bookmaker['markets'][0]
                    print(f"   Market: {market.get('key')}")
                    print("   Outcomes:")
                    for outcome in market.get('outcomes', []):
                        print(f"     {outcome.get('name')}: {outcome.get('price')}")
        
        return games
        
    except Exception as e:
        print(f"❌ Error fetching EPL data: {e}")
        return []

def test_college_football_api():
    """Test college football API call (limited to 5 games)"""
    print("\n🔍 Testing College Football API")
    print("="*50)
    
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds"
    params = {
        'apiKey': ODDS_API_KEY,
        'regions': 'us',
        'markets': 'h2h',
        'oddsFormat': 'american',
        'dateFormat': 'iso'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        games = response.json()
        
        print(f"✅ Found {len(games)} college football games")
        
        sample_games = games[:5]
        print(f"📊 Analyzing first {len(sample_games)} games for team name formats:")
        
        team_names = set()
        for i, game in enumerate(sample_games, 1):
            home_team = game.get('home_team', '')
            away_team = game.get('away_team', '')
            team_names.add(home_team)
            team_names.add(away_team)
            
            print(f"\n   Game {i}: {away_team} @ {home_team}")
            print(f"   Start time: {game.get('commence_time')}")
            
            if game.get('bookmakers'):
                bookmaker = game['bookmakers'][0]
                if bookmaker.get('markets'):
                    market = bookmaker['markets'][0]
                    print(f"   Sample odds from {bookmaker.get('title')}:")
                    for outcome in market.get('outcomes', []):
                        print(f"     {outcome.get('name')}: {outcome.get('price')}")
        
        print(f"\n📝 Unique team names found ({len(team_names)}):")
        for team in sorted(team_names):
            print(f"   - {team}")
            
        return sample_games, team_names
        
    except Exception as e:
        print(f"❌ Error fetching college football data: {e}")
        return [], set()

def analyze_team_mapping_challenges(college_teams):
    """Analyze team name mapping challenges for college football"""
    print("\n🔍 Team Name Mapping Analysis")
    print("="*50)
    
    kalshi_abbrevs = {
        "STAN": "Stanford",
        "HAW": "Hawaii", 
        "OHIO": "Ohio State",
        "RUTG": "Rutgers",
        "WYO": "Wyoming",
        "AKR": "Akron",
        "DSU": "Delaware State",
        "DEL": "Delaware",
        "ALST": "Alabama State",
        "UAB": "UAB"
    }
    
    print("🎯 Known Kalshi abbreviations:")
    for abbrev, full_name in kalshi_abbrevs.items():
        print(f"   {abbrev}: {full_name}")
    
    print(f"\n🔍 Mapping challenges for {len(college_teams)} API team names:")
    
    mapping_challenges = []
    potential_matches = {}
    
    for api_team in college_teams:
        matches = []
        for abbrev, kalshi_name in kalshi_abbrevs.items():
            if kalshi_name.lower() in api_team.lower() or api_team.lower() in kalshi_name.lower():
                matches.append((abbrev, kalshi_name))
        
        if matches:
            potential_matches[api_team] = matches
            print(f"   ✅ {api_team} -> Potential matches: {matches}")
        else:
            mapping_challenges.append(api_team)
            print(f"   ❓ {api_team} -> No obvious match found")
    
    print(f"\n📊 Summary:")
    print(f"   - Teams with potential matches: {len(potential_matches)}")
    print(f"   - Teams needing manual mapping: {len(mapping_challenges)}")
    
    if mapping_challenges:
        print(f"\n⚠️  Teams requiring manual investigation:")
        for team in mapping_challenges:
            print(f"     - {team}")
    
    return potential_matches, mapping_challenges

if __name__ == "__main__":
    print("🚀 Testing New Sports API Integration")
    print("="*60)
    
    sports = test_odds_api_sports()
    
    epl_games = test_epl_soccer_api()
    
    college_games, college_teams = test_college_football_api()
    
    if college_teams:
        potential_matches, challenges = analyze_team_mapping_challenges(college_teams)
    
    print(f"\n🎉 Testing complete!")
    print(f"   - EPL games found: {len(epl_games) if epl_games else 0}")
    print(f"   - College football games sampled: {len(college_games) if college_games else 0}")
    print(f"   - Unique college teams analyzed: {len(college_teams) if college_teams else 0}")
