#!/usr/bin/env python3
"""
Test script for multi-sport betting expansion
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kalshi_bot import (
    american_to_implied_prob, 
    devig_sportsbook_odds, 
    devig_composite_odds,
    get_dynamic_kelly_multiplier,
    api_calls_made,
    max_api_calls
)

def test_american_to_implied_prob():
    """Test American odds to implied probability conversion"""
    print("Testing American odds conversion...")
    
    assert abs(american_to_implied_prob(150) - 0.4) < 0.01, "Positive odds conversion failed"
    
    assert abs(american_to_implied_prob(-150) - 0.6) < 0.01, "Negative odds conversion failed"
    
    print("✅ American odds conversion tests passed")

def test_devigging():
    """Test devigging logic"""
    print("Testing devigging logic...")
    
    odds_dict = {
        "Team A": -110,
        "Team B": -110
    }
    
    opponent_map = {
        "Team A": "Team B",
        "Team B": "Team A"
    }
    
    devigged = devig_sportsbook_odds(odds_dict, opponent_map)
    
    assert 1.9 < devigged["Team A"] < 2.1, f"Devigged odds out of range: {devigged['Team A']}"
    assert 1.9 < devigged["Team B"] < 2.1, f"Devigged odds out of range: {devigged['Team B']}"
    
    print("✅ Devigging tests passed")

def test_composite_odds():
    """Test composite odds calculation"""
    print("Testing composite odds calculation...")
    
    sportsbook_odds = {
        "pinnacle": {"Team A": -105, "Team B": -105},
        "fanduel": {"Team A": -110, "Team B": -110},
        "draftkings": {"Team A": -115, "Team B": -115}
    }
    
    opponent_map = {
        "Team A": "Team B", 
        "Team B": "Team A"
    }
    
    composite = devig_composite_odds(sportsbook_odds, opponent_map)
    
    assert "Team A" in composite, "Team A missing from composite odds"
    assert "Team B" in composite, "Team B missing from composite odds"
    
    print("✅ Composite odds tests passed")

def test_dynamic_kelly():
    """Test dynamic Kelly multiplier"""
    print("Testing dynamic Kelly multiplier...")
    
    multiplier = get_dynamic_kelly_multiplier()
    assert 0.5 <= multiplier <= 0.75, f"Kelly multiplier out of range: {multiplier}"
    
    print(f"✅ Dynamic Kelly multiplier: {multiplier}")

def test_api_tracking():
    """Test API call tracking"""
    print("Testing API call tracking...")
    
    print(f"API calls made: {api_calls_made}/{max_api_calls}")
    assert api_calls_made <= max_api_calls, "API call limit exceeded"
    
    print("✅ API tracking tests passed")

def main():
    """Run all tests"""
    print("🧪 Running multi-sport betting tests...\n")
    
    try:
        test_american_to_implied_prob()
        test_devigging()
        test_composite_odds()
        test_dynamic_kelly()
        test_api_tracking()
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
