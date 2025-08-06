#!/usr/bin/env python3
"""
Test syntax and imports for the updated kalshi_bot.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work correctly"""
    print("🧪 Testing imports...")
    try:
        import kalshi_bot
        print("✅ kalshi_bot imported successfully")
        
        assert hasattr(kalshi_bot, 'fetch_sport_opportunities')
        assert hasattr(kalshi_bot, 'fetch_composite_odds')
        assert hasattr(kalshi_bot, 'devig_composite_odds')
        assert hasattr(kalshi_bot, 'count_api_call')
        assert hasattr(kalshi_bot, 'get_dynamic_kelly_multiplier')
        print("✅ All key functions found")
        
        print("🎯 Testing tournament configurations...")
        result = kalshi_bot.fetch_sport_opportunities('tennis_wta', 'test-key')
        print("✅ Tennis WTA configuration accessible")
        
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("🎉 All syntax and import tests passed!")
    else:
        print("💥 Tests failed!")
        sys.exit(1)
