#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_integration():
    """Simple test to verify new sports are integrated"""
    print("🔍 Testing basic integration...")
    
    try:
        from kalshi_bot import epl_team_abbr_to_name, college_football_team_abbr_to_name
        
        print(f"✅ EPL teams mapped: {len(epl_team_abbr_to_name)}")
        print(f"✅ College football teams mapped: {len(college_football_team_abbr_to_name)}")
        
        from kalshi_bot import devig_sportsbook_odds, devig_soccer_odds
        
        two_way_odds = {"Team A": -110, "Team B": -110}
        opponent_map = {"Team A": "Team B", "Team B": "Team A"}
        devigged_2way = devig_sportsbook_odds(two_way_odds, opponent_map)
        print(f"✅ 2-way devigging works: {len(devigged_2way)} teams")
        
        three_way_odds = {
            "game1": {"Liverpool": -200, "Bournemouth": 400, "Draw": 300}
        }
        devigged_3way = devig_soccer_odds(three_way_odds)
        print(f"✅ 3-way devigging works: {len(devigged_3way)} outcomes")
        
        print("🎉 All basic integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing New Sports Basic Integration")
    print("="*50)
    
    if test_basic_integration():
        sys.exit(0)
    else:
        sys.exit(1)
