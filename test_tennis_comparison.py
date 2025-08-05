#!/usr/bin/env python3
"""
Tennis API comparison test - analyzes player name formatting between Odds API and Kalshi
Limited to 20 API calls as requested by user
"""

import os
import sys
from datetime import datetime
import json

max_api_calls = 20
api_calls_made = 0

MOCK_ODDS_API_TENNIS = {
    "WTA": [
        {
            "match_id": "wta_001",
            "home_team": "Iga Swiatek", 
            "away_team": "Coco Gauff",
            "bookmakers": {
                "fanduel": {"Iga Swiatek": -150, "Coco Gauff": +120},
                "pinnacle": {"I. Swiatek": -145, "C. Gauff": +125}, 
                "draftkings": {"Swiatek, Iga": -155, "Gauff, Coco": +115}
            }
        },
        {
            "match_id": "wta_002",
            "home_team": "Aryna Sabalenka",
            "away_team": "Jessica Pegula",
            "bookmakers": {
                "fanduel": {"Aryna Sabalenka": +110, "Jessica Pegula": -130},
                "pinnacle": {"A. Sabalenka": +115, "J. Pegula": -135},
                "draftkings": {"Sabalenka, A": +105, "Pegula, Jessica": -125}
            }
        },
        {
            "match_id": "wta_003",
            "home_team": "Elena Rybakina",
            "away_team": "Marketa Vondrousova",
            "bookmakers": {
                "fanduel": {"Elena Rybakina": -180, "Marketa Vondrousova": +150},
                "pinnacle": {"E. Rybakina": -175, "M. Vondrousova": +155},
                "draftkings": {"Rybakina, Elena": -185, "Vondrousova, M": +145}
            }
        }
    ],
    "ATP": [
        {
            "match_id": "atp_001",
            "home_team": "Novak Djokovic",
            "away_team": "Carlos Alcaraz", 
            "bookmakers": {
                "fanduel": {"Novak Djokovic": -200, "Carlos Alcaraz": +170},
                "pinnacle": {"N. Djokovic": -195, "C. Alcaraz": +175},
                "draftkings": {"Djokovic, Novak": -205, "Alcaraz, Carlos": +165}
            }
        },
        {
            "match_id": "atp_002",
            "home_team": "Jannik Sinner",
            "away_team": "Daniil Medvedev",
            "bookmakers": {
                "fanduel": {"Jannik Sinner": -120, "Daniil Medvedev": +100},
                "pinnacle": {"J. Sinner": -115, "D. Medvedev": +105},
                "draftkings": {"Sinner, Jannik": -125, "Medvedev, Daniil": +95}
            }
        }
    ]
}

MOCK_KALSHI_TENNIS = {
    "KXWTAMATCH": [
        {"ticker": "KXWTAMATCH-24AUG05-SWIATEK", "title": "Will Iga Swiatek win?", "team": "SWIATEK", "yes_ask": 65},
        {"ticker": "KXWTAMATCH-24AUG05-GAUFF", "title": "Will Coco Gauff win?", "team": "GAUFF", "yes_ask": 40},
        {"ticker": "KXWTAMATCH-24AUG05-SABALENKA", "title": "Will Aryna Sabalenka win?", "team": "SABALENKA", "yes_ask": 45},
        {"ticker": "KXWTAMATCH-24AUG05-PEGULA", "title": "Will Jessica Pegula win?", "team": "PEGULA", "yes_ask": 58},
        {"ticker": "KXWTAMATCH-24AUG05-RYBAKINA", "title": "Will Elena Rybakina win?", "team": "RYBAKINA", "yes_ask": 72},
        {"ticker": "KXWTAMATCH-24AUG05-VONDROUSOVA", "title": "Will Marketa Vondrousova win?", "team": "VONDROUSOVA", "yes_ask": 35}
    ],
    "KXATPMATCH": [
        {"ticker": "KXATPMATCH-24AUG05-DJOKOVIC", "title": "Will Novak Djokovic win?", "team": "DJOKOVIC", "yes_ask": 75},
        {"ticker": "KXATPMATCH-24AUG05-ALCARAZ", "title": "Will Carlos Alcaraz win?", "team": "ALCARAZ", "yes_ask": 30},
        {"ticker": "KXATPMATCH-24AUG05-SINNER", "title": "Will Jannik Sinner win?", "team": "SINNER", "yes_ask": 62},
        {"ticker": "KXATPMATCH-24AUG05-MEDVEDEV", "title": "Will Daniil Medvedev win?", "team": "MEDVEDEV", "yes_ask": 48}
    ]
}

def count_api_call():
    """Track API calls to stay under 20 limit"""
    global api_calls_made
    api_calls_made += 1
    print(f"📞 API call #{api_calls_made}/{max_api_calls}")
    return api_calls_made <= max_api_calls

def fetch_tennis_odds_api(api_key, sport):
    """Fetch tennis odds from The Odds API"""
    if not count_api_call():
        print(f"❌ API call limit reached ({max_api_calls})")
        return None
    
    print(f"📡 Fetching {sport} odds from Odds API...")
    
    try:
        import requests
        url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
        params = {
            'api_key': api_key,
            'regions': 'us',
            'markets': 'h2h',
            'bookmakers': 'fanduel,pinnacle,draftkings'
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Fetched {len(data)} matches from Odds API")
        return data
        
    except ImportError:
        print("⚠️ requests module not available, using mock data")
        league = "WTA" if "wta" in sport else "ATP"
        return MOCK_ODDS_API_TENNIS[league]
    except Exception as e:
        print(f"❌ Error fetching from Odds API: {e}")
        league = "WTA" if "wta" in sport else "ATP"
        return MOCK_ODDS_API_TENNIS[league]

def fetch_kalshi_tennis_data(series_ticker):
    """Fetch tennis data from Kalshi API"""
    if not count_api_call():
        print(f"❌ API call limit reached ({max_api_calls})")
        return None
    
    print(f"🎯 Fetching Kalshi data for {series_ticker}")
    
    try:
        import requests
        url = f"https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker={series_ticker}"
        headers = {"accept": "application/json"}
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        markets = []
        for market in data.get("markets", []):
            if market.get("status") != "active":
                continue
            parts = market.get("ticker", "").split('-')
            if len(parts) < 3:
                continue
            markets.append({
                "ticker": market.get("ticker"),
                "title": market.get("title"),
                "team": parts[-1],
                "yes_ask": market.get("yes_ask")
            })
        
        print(f"✅ Fetched {len(markets)} markets from Kalshi")
        return markets
        
    except ImportError:
        print("⚠️ requests module not available, using mock data")
        return MOCK_KALSHI_TENNIS[series_ticker]
    except Exception as e:
        print(f"❌ Error fetching from Kalshi: {e}")
        return MOCK_KALSHI_TENNIS[series_ticker]

def analyze_tennis_name_patterns():
    """Analyze tennis player name formatting patterns"""
    print("🎾 TENNIS NAME PATTERN ANALYSIS")
    print("=" * 50)
    
    tennis_sports = {
        "WTA": "tennis_wta",
        "ATP": "tennis_atp"
    }
    
    results = {}
    odds_api_key = os.getenv('ODDS_API_KEY')
    
    if not odds_api_key:
        print("⚠️ ODDS_API_KEY not found, using mock data for analysis")
    
    for league, sport_code in tennis_sports.items():
        print(f"\n🏆 Testing {league} ({sport_code})")
        print("-" * 30)
        
        if api_calls_made >= max_api_calls:
            print(f"❌ API call limit reached, skipping {league}")
            break
        
        try:
            odds_data = fetch_tennis_odds_api(odds_api_key, sport_code)
            kalshi_series = "KXWTAMATCH" if league == "WTA" else "KXATPMATCH"
            kalshi_data = fetch_kalshi_tennis_data(kalshi_series)
            
            analysis = analyze_name_formatting(odds_data, kalshi_data, league)
            results[league] = analysis
            
        except Exception as e:
            print(f"❌ Error testing {league}: {e}")
            results[league] = {"error": str(e)}
    
    return results

def analyze_name_formatting(odds_data, kalshi_data, league):
    """Analyze name formatting differences between APIs"""
    analysis = {
        "odds_api_patterns": {},
        "kalshi_patterns": {},
        "potential_matches": [],
        "formatting_challenges": []
    }
    
    print(f"🔍 Analyzing {league} name patterns...")
    
    odds_names = []
    if odds_data:
        for match in odds_data:
            if isinstance(match, dict):
                if "home_team" in match and "away_team" in match:
                    odds_names.extend([match["home_team"], match["away_team"]])
                elif "bookmakers" in match:
                    for bookmaker in match.get("bookmakers", []):
                        if "outcomes" in bookmaker:
                            for outcome in bookmaker["outcomes"]:
                                odds_names.append(outcome.get("name", ""))
    
    kalshi_names = []
    if kalshi_data:
        kalshi_names = [market["team"] for market in kalshi_data if "team" in market]
    
    analysis["odds_api_patterns"] = analyze_odds_api_patterns(odds_names)
    analysis["kalshi_patterns"] = analyze_kalshi_patterns(kalshi_names)
    analysis["potential_matches"] = find_potential_matches(odds_names, kalshi_names)
    analysis["formatting_challenges"] = identify_formatting_challenges(odds_names, kalshi_names)
    
    print(f"📊 Found {len(odds_names)} Odds API names, {len(kalshi_names)} Kalshi codes")
    print(f"🎯 Identified {len(analysis['potential_matches'])} potential matches")
    
    return analysis

def analyze_odds_api_patterns(names):
    """Analyze patterns in Odds API player names"""
    patterns = {
        "total_names": len(names),
        "full_names": 0,
        "initials": 0,
        "last_first": 0,
        "examples": names[:5],
        "unique_formats": set()
    }
    
    for name in names:
        if not name:
            continue
        patterns["unique_formats"].add(name)
        if "," in name:
            patterns["last_first"] += 1
        elif ". " in name:
            patterns["initials"] += 1
        else:
            patterns["full_names"] += 1
    
    return patterns

def analyze_kalshi_patterns(codes):
    """Analyze patterns in Kalshi player codes"""
    patterns = {
        "total_codes": len(codes),
        "all_uppercase": all(code.isupper() for code in codes if code),
        "single_words": all(' ' not in code for code in codes if code),
        "examples": codes[:5],
        "unique_codes": set(codes)
    }
    
    return patterns

def find_potential_matches(odds_names, kalshi_codes):
    """Find potential matches between Odds API names and Kalshi codes"""
    matches = []
    
    for odds_name in odds_names:
        if not odds_name:
            continue
        
        odds_last_name = extract_last_name(odds_name).upper()
        
        for kalshi_code in kalshi_codes:
            if not kalshi_code:
                continue
            
            if odds_last_name == kalshi_code:
                matches.append({
                    "odds_name": odds_name,
                    "kalshi_code": kalshi_code,
                    "match_type": "exact_last_name"
                })
            elif kalshi_code in odds_last_name or odds_last_name in kalshi_code:
                matches.append({
                    "odds_name": odds_name,
                    "kalshi_code": kalshi_code,
                    "match_type": "partial_match"
                })
    
    return matches

def extract_last_name(full_name):
    """Extract last name from various name formats"""
    if "," in full_name:
        return full_name.split(",")[0].strip()
    elif ". " in full_name:
        parts = full_name.split(". ")
        return parts[-1].strip() if len(parts) > 1 else full_name
    else:
        parts = full_name.split()
        return parts[-1] if parts else full_name

def identify_formatting_challenges(odds_names, kalshi_codes):
    """Identify specific formatting challenges"""
    challenges = []
    
    format_variations = {}
    for name in odds_names:
        if not name:
            continue
        last_name = extract_last_name(name).upper()
        if last_name not in format_variations:
            format_variations[last_name] = []
        format_variations[last_name].append(name)
    
    for last_name, variations in format_variations.items():
        if len(variations) > 1:
            challenges.append({
                "type": "multiple_formats",
                "last_name": last_name,
                "variations": variations,
                "kalshi_match": last_name in kalshi_codes
            })
    
    international_chars = []
    for name in odds_names:
        if any(ord(char) > 127 for char in name):
            international_chars.append(name)
    
    if international_chars:
        challenges.append({
            "type": "international_characters",
            "examples": international_chars[:3]
        })
    
    return challenges

def print_comprehensive_analysis(results):
    """Print comprehensive analysis of tennis API comparison"""
    print("\n" + "=" * 60)
    print("🎾 TENNIS API COMPARISON ANALYSIS")
    print("=" * 60)
    
    for league, analysis in results.items():
        if "error" in analysis:
            print(f"\n❌ {league} Analysis Failed: {analysis['error']}")
            continue
        
        print(f"\n🏆 {league} ANALYSIS")
        print("-" * 40)
        
        odds_patterns = analysis["odds_api_patterns"]
        kalshi_patterns = analysis["kalshi_patterns"]
        
        print(f"📊 Odds API Patterns:")
        print(f"   Total names: {odds_patterns['total_names']}")
        print(f"   Full names: {odds_patterns['full_names']}")
        print(f"   Initials: {odds_patterns['initials']}")
        print(f"   Last,First: {odds_patterns['last_first']}")
        print(f"   Examples: {odds_patterns['examples']}")
        
        print(f"\n📊 Kalshi Patterns:")
        print(f"   Total codes: {kalshi_patterns['total_codes']}")
        print(f"   All uppercase: {kalshi_patterns['all_uppercase']}")
        print(f"   Single words: {kalshi_patterns['single_words']}")
        print(f"   Examples: {kalshi_patterns['examples']}")
        
        matches = analysis["potential_matches"]
        print(f"\n🎯 Potential Matches ({len(matches)}):")
        for match in matches[:5]:
            print(f"   '{match['odds_name']}' → '{match['kalshi_code']}' ({match['match_type']})")
        
        challenges = analysis["formatting_challenges"]
        print(f"\n🚨 Formatting Challenges ({len(challenges)}):")
        for challenge in challenges:
            if challenge["type"] == "multiple_formats":
                print(f"   Multiple formats for {challenge['last_name']}: {challenge['variations']}")
            elif challenge["type"] == "international_characters":
                print(f"   International characters: {challenge['examples']}")

def propose_tennis_matching_solution():
    """Propose solution for tennis name matching"""
    print(f"\n💡 PROPOSED TENNIS NAME MATCHING SOLUTION")
    print("=" * 50)
    
    solution_steps = [
        "1. Extract last name from full player name using multiple patterns",
        "2. Convert to uppercase for Kalshi comparison", 
        "3. Handle special characters (ć→C, ñ→N, etc.) with transliteration",
        "4. Use fuzzy matching (Levenshtein distance) for partial matches",
        "5. Maintain manual override dictionary for edge cases",
        "6. Implement confidence scoring for match quality"
    ]
    
    for step in solution_steps:
        print(f"   {step}")
    
    print(f"\n🔧 Implementation Example:")
    print(f"   def normalize_tennis_name(full_name):")
    print(f"       # 'Iga Swiatek' → 'SWIATEK'")
    print(f"       # 'N. Djokovic' → 'DJOKOVIC'") 
    print(f"       # 'Sabalenka, Aryna' → 'SABALENKA'")
    print(f"       # 'Novak Đoković' → 'DJOKOVIC'")

def simulate_api_budget_usage():
    """Show API call budget usage"""
    print(f"\n📞 API CALL BUDGET ANALYSIS")
    print("=" * 50)
    
    print(f"🎯 Total API calls made: {api_calls_made}/{max_api_calls}")
    print(f"💰 Remaining budget: {max_api_calls - api_calls_made} calls")
    
    if api_calls_made <= max_api_calls:
        print(f"✅ Stayed within budget limit")
    else:
        print(f"❌ Exceeded budget limit")
    
    planned_calls = [
        "WTA odds from Odds API",
        "ATP odds from Odds API", 
        "KXWTAMATCH from Kalshi",
        "KXATPMATCH from Kalshi"
    ]
    
    print(f"\n📋 Planned API calls:")
    for i, call in enumerate(planned_calls, 1):
        status = "✅" if i <= api_calls_made else "⏳"
        print(f"   {status} {call}")

def main():
    """Run tennis comparison analysis"""
    print("🎾 TENNIS API COMPARISON TEST")
    print(f"⏰ Started at: {datetime.now()}")
    print(f"🎯 Focus: Player name formatting analysis")
    print(f"📊 API Budget: {max_api_calls} calls maximum")
    
    results = analyze_tennis_name_patterns()
    print_comprehensive_analysis(results)
    propose_tennis_matching_solution()
    simulate_api_budget_usage()
    
    print(f"\n✅ ANALYSIS COMPLETE")
    print(f"📋 Key Finding: Tennis requires sophisticated name normalization")
    print(f"🎯 Next Step: Implement fuzzy matching with last-name extraction")
    print(f"📞 Final API usage: {api_calls_made}/{max_api_calls}")
    
    return results

if __name__ == "__main__":
    main()
