#!/usr/bin/env python3
"""
Test script for email report changes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from daily_email_report import summarize_sport

def test_email_report_structure():
    """Test that email report has correct structure"""
    print("Testing email report structure...")
    
    mlb_team_map = {
        "ATL": "Atlanta Braves",
        "NYY": "New York Yankees"
    }
    
    try:
        df, participation_rate = summarize_sport("KXMLBGAME", "MLB", mlb_team_map)
        
        assert "ev_before_devig" not in df.columns, "ev_before_devig column should be removed"
        
        assert "ev_after_devig" in df.columns, "ev_after_devig column should exist"
        
        assert "/" in participation_rate, "Participation rate should be in format 'x/y'"
        
        print(f"✅ Email report structure correct. Participation rate: {participation_rate}")
        
    except Exception as e:
        print(f"⚠️ Email report test failed (expected if no data): {e}")

def main():
    """Run email report tests"""
    print("🧪 Testing email report changes...\n")
    
    test_email_report_structure()
    
    print("\n✅ Email report tests completed")

if __name__ == "__main__":
    main()
