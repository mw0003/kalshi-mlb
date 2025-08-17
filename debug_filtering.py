#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kalshi_bot import get_espn_scoreboard_json, parse_games, get_eligible_teams

def debug_nfl_filtering():
    print("=== DEBUG: NFL Filtering Logic ===")
    
    try:
        nfl_data = get_espn_scoreboard_json("americanfootball_nfl")
        print(f"NFL API Response received: {len(nfl_data.get('events', []))} events")
        
        eligible_teams = parse_games(nfl_data, "americanfootball_nfl")
        print(f"NFL eligible teams after parsing: {eligible_teams}")
        
        if eligible_teams:
            for team in eligible_teams:
                print(f"  - {team}")
        else:
            print("  No eligible NFL teams found")
            
    except Exception as e:
        print(f"Error in NFL filtering debug: {e}")

def debug_mls_filtering():
    print("\n=== DEBUG: MLS Filtering Logic ===")
    
    try:
        mls_data = get_espn_scoreboard_json("soccer_mls")
        print(f"MLS API Response received: {len(mls_data.get('events', []))} events")
        
        eligible_teams = parse_games(mls_data, "soccer_mls")
        print(f"MLS eligible teams after parsing: {eligible_teams}")
        
        if eligible_teams:
            for team in eligible_teams:
                print(f"  - {team}")
        else:
            print("  No eligible MLS teams found")
            
    except Exception as e:
        print(f"Error in MLS filtering debug: {e}")

def debug_all_eligible_teams():
    print("\n=== DEBUG: All Eligible Teams ===")
    
    try:
        all_eligible = get_eligible_teams()
        print(f"Total eligible teams across all sports: {len(all_eligible)}")
        
        for sport, teams in all_eligible.items():
            print(f"{sport}: {teams}")
            
    except Exception as e:
        print(f"Error in all eligible teams debug: {e}")

if __name__ == "__main__":
    debug_nfl_filtering()
    debug_mls_filtering()
    debug_all_eligible_teams()
